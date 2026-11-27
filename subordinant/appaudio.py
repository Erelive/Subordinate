"""Capture of one application's audio (WASAPI process loopback).

LoopbackCapture hears everything the PC plays, so a game running beside the
stream being captioned is mixed into what the recogniser hears. Since build
20348 Windows will instead hand over the mix of a single process tree, and
ProcessCapture is LoopbackCapture's surface on top of that: the pipeline reads
either one the same way.

Plain ctypes against the COM vtables, as audio.default_output_id does: the
alternative is a dependency (comtypes, pycaw, psutil) for a few dozen calls.

What is odd here, and why:

  - The process *tree* is captured, from the topmost process of the app.
    Browsers and Electron apps play from a child utility process, so the pid
    that owns the audio session is not the one worth remembering: it changes
    whenever the browser restarts its audio service. An app is therefore
    identified by image name and captured from its root.

  - The device is a virtual one with no format of its own (GetMixFormat is
    not implemented); it converts to whatever Initialize asks for. It also
    activates happily for a pid that does not exist and then delivers
    silence, so whether the app is running is checked here, with a process
    handle, never inferred from activation succeeding.

  - Its queue is three 10 ms packets deep whatever buffer Initialize is asked
    for (measured on build 26300), and packets that do not fit are dropped
    with no discontinuity flag. A reader that is 40 ms late has lost audio
    and is not told. So the capture thread does nothing but wait, copy and
    count; the three COM calls it makes per packet do not release the GIL,
    because every release is an invitation for another thread to hold it for
    a switch interval; and what was lost is replaced by silence of the same
    length, so that the stream keeps its place in time.

  - Silence. A classic loopback stream stops delivering while nothing plays,
    and depending on the Windows build this one either does the same or
    delivers packets of zeros. The voice detector needs a continuous stream
    to see that someone stopped talking, so Pacer keeps what is handed on in
    step with the wall clock: when packets stop, zeros are synthesised up to
    just short of now, and never past it.

  - One thread owns the stream's COM objects. start() and the watcher ask
    that thread to open or close; they never touch an interface themselves,
    which leaves nothing to lock and makes stop() safe from anywhere.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import queue
import sys
import threading
import time
import uuid
from ctypes import POINTER, byref, c_void_p, wintypes
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, NamedTuple

import numpy as np

from .audio import AudioDeviceError, CaptureBase

log = logging.getLogger(__name__)

# First build with process loopback (Windows Server 2022; every Windows 11).
MIN_BUILD = 20348

WATCH_INTERVAL_SEC = 1.5
# Longest the capture thread sleeps. Well inside the device's 30 ms queue.
WAIT_MS = 20
# No packet for this long means the stream has stopped, not that one is late.
# Packets are 10 ms apart; this is also how far behind the clock a running
# stream may fall before the shortfall is taken to be lost audio.
SILENCE_GAP_SEC = 0.04
# Synthesised silence stops this far short of now. A packet that arrives next
# covers the 10 ms before it was delivered, and must not overlap the silence.
SILENCE_SLACK_SEC = 0.02
# How often a running stream's clock is forgiven its drift against the PC's.
REANCHOR_SEC = 10.0
# More silence than this owed at once is a suspended PC, not a quiet app.
MAX_FILL_SEC = 2.0

CAPTURE_RATE = 48000
CAPTURE_CHANNELS = 2
ACTIVATE_TIMEOUT_SEC = 5.0


def supported() -> bool:
    """Whether this Windows can capture a single application."""
    return sys.platform == "win32" and sys.getwindowsversion().build >= MIN_BUILD


# -- which apps are playing ------------------------------------------------


@dataclass
class AudioApp:
    exe: str  # image name, "chrome.exe"
    name: str  # for people: display_name(exe)
    pid: int  # root of the process tree to capture
    active: bool  # has a stream running right now


class Proc(NamedTuple):
    parent: int
    exe: str


def display_name(exe: str) -> str:
    """The short name people know a program by: "chrome.exe" -> "chrome".

    Made from the image name alone, so that it is the same whether or not
    the program is running: the picker, the status line and the notices all
    use it and have to agree. Never raises.
    """
    name = os.path.basename(str(exe).strip())
    return name[:-4] if name.lower().endswith(".exe") else name


def _same_image(exe: str, wanted: str) -> bool:
    """Image names compare without case, and "chrome" will do for "chrome.exe"."""
    return display_name(exe).lower() == display_name(wanted).lower()


def root_pid(pid: int, table: Mapping[int, Proc]) -> int:
    """The topmost ancestor of `pid` that is still the same program.

    Climbs while the parent has the same image name. A parent that has
    exited leaves its pid behind in the child's record, where Windows may
    have reused it for something else, hence the check on the name and the
    guard against going round in a circle.
    """
    if pid not in table:
        return pid
    exe = table[pid].exe.lower()
    seen = {pid}
    while True:
        parent = table[pid].parent
        above = table.get(parent)
        if above is None or parent in seen or above.exe.lower() != exe:
            return pid
        seen.add(parent)
        pid = parent


def group_sessions(
    sessions: Iterable[tuple[int, bool]],
    table: Mapping[int, Proc],
    skip_pid: int | None = None,
) -> list[AudioApp]:
    """One entry per program from (pid, active) audio sessions.

    Active ones first, then alphabetical. Pid 0 is the system-sounds session.
    """
    found: dict[str, AudioApp] = {}
    for pid, active in sessions:
        if pid == 0 or pid == skip_pid or pid not in table:
            continue
        root = root_pid(pid, table)
        exe = table[root].exe
        known = found.get(exe.lower())
        if known is None:
            found[exe.lower()] = AudioApp(exe, display_name(exe), root, active)
        elif active and not known.active:
            # Two instances of one program: the one making sound is the one meant.
            known.pid, known.active = root, True
    return sorted(found.values(), key=lambda app: (not app.active, app.exe.lower()))


def pick_root(
    exe: str,
    table: Mapping[int, Proc],
    sessions: Iterable[tuple[int, bool]] = (),
) -> int | None:
    """The root pid to capture for an image name, or None if it is not running.

    With several instances running, one whose tree is playing wins, then one
    that at least has an audio session, then the lowest pid.
    """
    roots = {root_pid(pid, table) for pid, proc in table.items() if _same_image(proc.exe, exe)}
    if not roots:
        return None
    with_session = None
    for pid, active in sessions:
        if pid not in table:
            continue
        root = root_pid(pid, table)
        if root not in roots:
            continue
        if active:
            return root
        if with_session is None:
            with_session = root
    return with_session if with_session is not None else min(roots)


def list_audio_apps() -> list[AudioApp]:
    """Programs with an audio session on the default playback device.

    Never raises: a machine with no playback device, or anything else going
    wrong, is an empty list.
    """
    try:
        return group_sessions(_audio_sessions(), _process_table(), skip_pid=os.getpid())
    except Exception:  # noqa: BLE001 - a picker with nothing in it beats a crash
        log.debug("could not list audio apps", exc_info=True)
        return []


def find_app(exe: str) -> int | None:
    """Root pid to capture for an image name such as "chrome.exe", or None."""
    try:
        table = _process_table()
    except Exception:  # noqa: BLE001
        log.debug("could not list processes", exc_info=True)
        return None
    try:
        sessions = _audio_sessions()
    except Exception:  # noqa: BLE001 - no playback device; any instance will do
        sessions = []
    return pick_root(exe, table, sessions)


# -- keeping time ----------------------------------------------------------


class Pacer:
    """Decides how much silence keeps a packet stream in step with the clock.

    Two things make a stream fall short of real time: the device stops
    delivering because the app is silent or gone, and the device drops
    packets because they were read too late. Both show up the same way, as
    fewer frames than seconds, and both are made up with zeros.

    Pure arithmetic on the times it is given, so that it can be tested
    without a device. All times are on one monotonic clock, in seconds.
    """

    def __init__(self, rate: int, now: float):
        self.rate = rate
        self._origin = now  # when frame 0 was due
        self._frames = 0  # frames accounted for since then, real or not
        self._last_packet = now
        self._anchored = now
        self.synthesising = False  # zeros have gone out since the last packet

    def packets(self, now: float, frames: int) -> int:
        """Account for `frames` just read. Returns the silence due before them."""
        behind = int((now - self._origin) * self.rate) - self._frames - frames
        pad = 0
        # Coming out of synthesised silence the shortfall is paid exactly, so
        # that the stream starts level. In a running stream a small one is
        # only the packet still being filled, and is left alone.
        if behind > 0 and (self.synthesising or behind >= SILENCE_GAP_SEC * self.rate):
            pad = self._capped(behind)
        self._frames += pad + frames
        if pad or now - self._anchored >= REANCHOR_SEC:
            # The device's clock is not the PC's. Left to add up, the
            # difference would one day look like a gap and be filled.
            self._origin = now - self._frames / self.rate
            self._anchored = now
        self.synthesising = False
        self._last_packet = now
        return pad

    def idle(self, now: float) -> int:
        """Nothing was waiting to be read. Returns the silence due now."""
        if now - self._last_packet < SILENCE_GAP_SEC:
            return 0
        owed = int((now - self._origin - SILENCE_SLACK_SEC) * self.rate) - self._frames
        if owed <= 0:
            return 0
        owed = self._capped(owed)
        self._frames += owed
        self.synthesising = True
        return owed

    def _capped(self, frames: int) -> int:
        limit = int(MAX_FILL_SEC * self.rate)
        if frames > limit:
            self._origin += (frames - limit) / self.rate
            return limit
        return frames


# -- Win32 and COM plumbing ------------------------------------------------


class GUID(ctypes.Structure):
    _fields_ = [
        ("a", wintypes.DWORD),
        ("b", wintypes.WORD),
        ("c", wintypes.WORD),
        ("d", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> GUID:
    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


_CLSID_DEVICE_ENUMERATOR = _guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
_IID_DEVICE_ENUMERATOR = _guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
_IID_SESSION_MANAGER2 = _guid("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
_IID_SESSION_CONTROL2 = _guid("{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}")
_IID_AUDIO_CLIENT = _guid("{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}")
_IID_CAPTURE_CLIENT = _guid("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
_IID_UNKNOWN = _guid("{00000000-0000-0000-C000-000000000046}")
_IID_AGILE_OBJECT = _guid("{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}")
_IID_COMPLETION_HANDLER = _guid("{41D949AB-9862-444A-80F6-C261334DA5EB}")

_PROCESS_LOOPBACK_DEVICE = "VAD\\Process_Loopback"
_VT_BLOB = 65
_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
_LOOPBACK_MODE_INCLUDE_TREE = 0

_CLSCTX_ALL = 23
_COINIT_MULTITHREADED = 0
_S_OK = 0
_E_NOINTERFACE = -2147467262  # 0x80004002
_E_POINTER = -2147467261  # 0x80004003
_AUDCLNT_S_BUFFER_EMPTY = 0x08890001
_AUDCLNT_SHAREMODE_SHARED = 0
_AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
_AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
_AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
_AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
_AUDCLNT_BUFFERFLAGS_SILENT = 0x2
_AUDIO_SESSION_ACTIVE = 1
_AUDIO_SESSION_EXPIRED = 2
_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3

_TH32CS_SNAPPROCESS = 0x2
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_WAIT_TIMEOUT = 258
_INVALID_HANDLE = c_void_p(-1).value


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ("ActivationType", ctypes.c_int),
        # AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, the only member of the union
        ("TargetProcessId", wintypes.DWORD),
        ("ProcessLoopbackMode", ctypes.c_int),
    ]


class BLOB(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", c_void_p)]


class PROPVARIANT(ctypes.Structure):
    """Only the VT_BLOB arm of the union, which is as large as any of them."""

    _fields_ = [
        ("vt", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("wReserved2", wintypes.WORD),
        ("wReserved3", wintypes.WORD),
        ("blob", BLOB),
    ]


class WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def activation_params(pid: int) -> tuple[PROPVARIANT, AUDIOCLIENT_ACTIVATION_PARAMS]:
    """What ActivateAudioInterfaceAsync needs to capture `pid` and its children.

    The variant points into the params, so the caller holds on to both.
    """
    params = AUDIOCLIENT_ACTIVATION_PARAMS(
        _ACTIVATION_TYPE_PROCESS_LOOPBACK, pid, _LOOPBACK_MODE_INCLUDE_TREE
    )
    variant = PROPVARIANT()
    variant.vt = _VT_BLOB
    variant.blob.cbSize = ctypes.sizeof(params)
    variant.blob.pBlobData = ctypes.addressof(params)
    return variant, params


def wave_format(rate: int, channels: int, float32: bool) -> WAVEFORMATEX:
    bits = 32 if float32 else 16
    align = channels * bits // 8
    return WAVEFORMATEX(
        _WAVE_FORMAT_IEEE_FLOAT if float32 else _WAVE_FORMAT_PCM,
        channels,
        rate,
        rate * align,
        align,
        bits,
        0,
    )


# Private library objects: argtypes set on ctypes.windll's shared ones would
# be seen by every other module that uses them.
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ole32 = ctypes.WinDLL("ole32")

_kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, POINTER(PROCESSENTRY32W)]
_kernel32.Process32NextW.argtypes = [wintypes.HANDLE, POINTER(PROCESSENTRY32W)]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.CreateEventW.restype = wintypes.HANDLE
_kernel32.CreateEventW.argtypes = [c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]

_ole32.CoInitializeEx.restype = ctypes.c_long
_ole32.CoInitializeEx.argtypes = [c_void_p, wintypes.DWORD]
_ole32.CoUninitialize.restype = None
_ole32.CoCreateInstance.restype = ctypes.HRESULT
_ole32.CoCreateInstance.argtypes = [
    POINTER(GUID),
    c_void_p,
    wintypes.DWORD,
    POINTER(GUID),
    POINTER(c_void_p),
]

# A prototype for calls that keep the GIL. PYFUNCTYPE is cdecl, which is the
# same thing as stdcall only on 64-bit Windows; a 32-bit build falls back to
# the ordinary kind and is merely slower to get the GIL back.
_HELD = ctypes.PYFUNCTYPE if ctypes.sizeof(c_void_p) == 8 else ctypes.WINFUNCTYPE


def _slot(pointer, index: int, restype, *argtypes, hold_gil: bool = False):
    """Entry `index` of a COM interface's vtable. Call it with the pointer first."""
    vtable = ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents
    prototype = (_HELD if hold_gil else ctypes.WINFUNCTYPE)(restype, c_void_p, *argtypes)
    return prototype(vtable[index])


def _release(pointer) -> None:
    if pointer:
        _slot(pointer, 2, ctypes.c_ulong)(pointer)


class _Com:
    """COM on the calling thread for the length of a with block."""

    def __enter__(self) -> "_Com":
        # Fails with RPC_E_CHANGED_MODE on a thread that is already a
        # single-threaded apartment (the GUI's). COM is up there all the
        # same, and that initialisation is not ours to undo.
        self._ours = _ole32.CoInitializeEx(None, _COINIT_MULTITHREADED) >= 0
        return self

    def __exit__(self, *_exc) -> None:
        if self._ours:
            _ole32.CoUninitialize()


class _AudioPriority:
    """Runs the calling thread as an audio thread (MMCSS) inside a with block.

    The device forgives 30 ms of lateness and no more, and the PC is by
    definition busy with a game. Does nothing if Windows declines.
    """

    def __enter__(self) -> "_AudioPriority":
        self._task = None
        try:
            self._avrt = ctypes.WinDLL("avrt")
            self._avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
            self._avrt.AvSetMmThreadCharacteristicsW.argtypes = [
                wintypes.LPCWSTR,
                POINTER(wintypes.DWORD),
            ]
            self._avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
            self._task = self._avrt.AvSetMmThreadCharacteristicsW("Audio", byref(wintypes.DWORD()))
        except (OSError, AttributeError):
            log.debug("no MMCSS for the capture thread", exc_info=True)
        return self

    def __exit__(self, *_exc) -> None:
        if self._task:
            self._avrt.AvRevertMmThreadCharacteristics(self._task)


def _process_table() -> dict[int, Proc]:
    """Every running process: pid -> (parent pid, image name)."""
    snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        table: dict[int, Proc] = {}
        more = _kernel32.Process32FirstW(snapshot, byref(entry))
        while more:
            table[entry.th32ProcessID] = Proc(entry.th32ParentProcessID, entry.szExeFile)
            more = _kernel32.Process32NextW(snapshot, byref(entry))
        return table
    finally:
        _kernel32.CloseHandle(snapshot)


def _audio_sessions() -> list[tuple[int, bool]]:
    """(pid, has a running stream) for each session on the default playback device.

    Raises OSError when there is no such device.
    """
    found: list[tuple[int, bool]] = []
    with _Com():
        held: list[c_void_p] = []

        def out() -> c_void_p:
            held.append(c_void_p())
            return held[-1]

        def call(pointer, index: int, argtypes: tuple, *args) -> None:
            _slot(pointer, index, ctypes.HRESULT, *argtypes)(pointer, *args)

        out_pointer = POINTER(c_void_p)
        try:
            enumerator = out()
            _ole32.CoCreateInstance(
                byref(_CLSID_DEVICE_ENUMERATOR),
                None,
                _CLSCTX_ALL,
                byref(_IID_DEVICE_ENUMERATOR),
                byref(enumerator),
            )
            device = out()
            # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender, eConsole)
            call(enumerator, 4, (ctypes.c_int, ctypes.c_int, out_pointer), 0, 0, byref(device))
            manager = out()
            call(  # IMMDevice::Activate
                device,
                3,
                (POINTER(GUID), wintypes.DWORD, c_void_p, out_pointer),
                byref(_IID_SESSION_MANAGER2),
                _CLSCTX_ALL,
                None,
                byref(manager),
            )
            listing = out()
            # IAudioSessionManager2::GetSessionEnumerator, then GetCount
            call(manager, 5, (out_pointer,), byref(listing))
            count = ctypes.c_int()
            call(listing, 3, (POINTER(ctypes.c_int),), byref(count))
            for index in range(count.value):
                try:
                    control = out()
                    # IAudioSessionEnumerator::GetSession
                    call(listing, 4, (ctypes.c_int, out_pointer), index, byref(control))
                    control2 = out()
                    call(  # QueryInterface
                        control,
                        0,
                        (POINTER(GUID), out_pointer),
                        byref(_IID_SESSION_CONTROL2),
                        byref(control2),
                    )
                    pid = wintypes.DWORD()
                    state = ctypes.c_int()
                    # IAudioSessionControl2::GetProcessId, IAudioSessionControl::GetState
                    call(control2, 14, (POINTER(wintypes.DWORD),), byref(pid))
                    call(control2, 3, (POINTER(ctypes.c_int),), byref(state))
                except OSError:
                    continue  # a session that went away while it was being asked
                if state.value != _AUDIO_SESSION_EXPIRED:
                    found.append((pid.value, state.value == _AUDIO_SESSION_ACTIVE))
        finally:
            for pointer in reversed(held):
                _release(pointer)
    return found


# -- activation ------------------------------------------------------------
#
# ActivateAudioInterfaceAsync reports back through a COM object the caller
# supplies. Ours is a vtable of four ctypes callbacks. The callbacks are
# module-level and live for ever; each activation gets its own tiny object,
# kept alive by _completions until Windows has released its last reference,
# which may be some time after the caller stopped caring.

_QUERY_INTERFACE = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(GUID), POINTER(c_void_p))
_ADD_REF = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)
_ACTIVATE_COMPLETED = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, c_void_p)

_completions: dict[int, "_Completion"] = {}
_completions_lock = threading.Lock()
# IAgileObject matters: without it activation refuses a handler that was not
# created on its own thread, with E_ILLEGAL_METHOD_CALL.
_COMPLETION_IIDS = frozenset(
    bytes(iid) for iid in (_IID_UNKNOWN, _IID_AGILE_OBJECT, _IID_COMPLETION_HANDLER)
)


def _completion_query_interface(this, riid, out) -> int:
    if not out:
        return _E_POINTER
    if riid and bytes(riid.contents) in _COMPLETION_IIDS:
        out[0] = this
        _completion_add_ref(this)
        return _S_OK
    out[0] = None
    return _E_NOINTERFACE


def _completion_add_ref(this) -> int:
    with _completions_lock:
        completion = _completions.get(this)
        if completion is None:
            return 0
        completion.refs += 1
        return completion.refs


def _completion_release(this) -> int:
    with _completions_lock:
        completion = _completions.get(this)
        if completion is None:
            return 0
        completion.refs -= 1
        if completion.refs == 0:
            del _completions[this]
        return completion.refs


def _completion_activate_completed(this, _operation) -> int:
    with _completions_lock:
        completion = _completions.get(this)
    if completion is not None:
        completion.done.set()
    return _S_OK


_COMPLETION_CALLBACKS = (
    _QUERY_INTERFACE(_completion_query_interface),
    _ADD_REF(_completion_add_ref),
    _ADD_REF(_completion_release),
    _ACTIVATE_COMPLETED(_completion_activate_completed),
)
_COMPLETION_VTABLE = (c_void_p * 4)(
    *(ctypes.cast(callback, c_void_p) for callback in _COMPLETION_CALLBACKS)
)


class _Completion:
    """An IActivateAudioInterfaceCompletionHandler that sets an Event."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.refs = 1
        # A COM object is a pointer to its vtable; this one is nothing more.
        self._object = c_void_p(ctypes.addressof(_COMPLETION_VTABLE))
        self.pointer = ctypes.addressof(self._object)
        with _completions_lock:
            _completions[self.pointer] = self

    def release(self) -> None:
        _completion_release(self.pointer)


@functools.lru_cache(maxsize=1)
def _activate_audio_interface_async():
    function = ctypes.WinDLL("mmdevapi").ActivateAudioInterfaceAsync
    function.restype = ctypes.HRESULT
    function.argtypes = [
        wintypes.LPCWSTR,
        POINTER(GUID),
        POINTER(PROPVARIANT),
        c_void_p,
        POINTER(c_void_p),
    ]
    return function


def _activate(pid: int) -> c_void_p:
    """An IAudioClient, not yet initialised, on the loopback of `pid`'s tree."""
    variant, _params = activation_params(pid)
    completion = _Completion()
    operation = c_void_p()
    try:
        _activate_audio_interface_async()(
            _PROCESS_LOOPBACK_DEVICE,
            byref(_IID_AUDIO_CLIENT),
            byref(variant),
            completion.pointer,
            byref(operation),
        )
        if not completion.done.wait(ACTIVATE_TIMEOUT_SEC):
            raise OSError("Windows did not answer the request to capture it")
        result = ctypes.c_long()
        unknown = c_void_p()
        # IActivateAudioInterfaceAsyncOperation::GetActivateResult
        _slot(operation, 3, ctypes.HRESULT, POINTER(ctypes.c_long), POINTER(c_void_p))(
            operation, byref(result), byref(unknown)
        )
        try:
            if result.value < 0:
                raise ctypes.WinError(result.value)
            client = c_void_p()
            _slot(unknown, 0, ctypes.HRESULT, POINTER(GUID), POINTER(c_void_p))(
                unknown, byref(_IID_AUDIO_CLIENT), byref(client)
            )
            return client
        finally:
            _release(unknown)
    finally:
        _release(operation)
        completion.release()


class _Stream:
    """A running process-loopback stream. Created, read and closed on one thread."""

    def __init__(self, pid: int):
        self.rate = CAPTURE_RATE
        self.channels = CAPTURE_CHANNELS
        self.float32 = True
        self.flags = 0
        self._client = c_void_p()
        self._capture = c_void_p()
        self._event = None
        try:
            self._open(pid)
        except BaseException:
            self.close()
            raise

    def _open(self, pid: int) -> None:
        plain = _AUDCLNT_STREAMFLAGS_LOOPBACK | _AUDCLNT_STREAMFLAGS_EVENTCALLBACK
        convert = (
            plain | _AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | _AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
        )
        refused: OSError | None = None
        for float32, flags in ((True, plain), (True, convert), (False, plain), (False, convert)):
            # A fresh client per attempt: one that has refused a format is
            # not promised to accept another.
            client = _activate(pid)
            wave = wave_format(self.rate, self.channels, float32)
            try:
                # IAudioClient::Initialize. A second of buffer is asked for
                # and, on the builds measured, ignored.
                _slot(
                    client,
                    3,
                    ctypes.HRESULT,
                    ctypes.c_int,
                    wintypes.DWORD,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                    POINTER(WAVEFORMATEX),
                    c_void_p,
                )(client, _AUDCLNT_SHAREMODE_SHARED, flags, 10_000_000, 0, byref(wave), None)
            except OSError as exc:
                _release(client)
                refused = exc
                continue
            self._client, self.float32, self.flags = client, float32, flags
            break
        else:
            raise refused if refused is not None else OSError("no capture format accepted")

        self._frame_bytes = self.channels * (4 if self.float32 else 2)
        self._event = _kernel32.CreateEventW(None, False, False, None)
        if not self._event:
            raise ctypes.WinError(ctypes.get_last_error())
        client = self._client
        # IAudioClient::SetEventHandle, GetService, Start
        _slot(client, 13, ctypes.HRESULT, wintypes.HANDLE)(client, self._event)
        _slot(client, 14, ctypes.HRESULT, POINTER(GUID), POINTER(c_void_p))(
            client, byref(_IID_CAPTURE_CLIENT), byref(self._capture)
        )
        capture = self._capture
        # IAudioCaptureClient::GetNextPacketSize, GetBuffer, ReleaseBuffer.
        # Bound once and holding the GIL: see the module docstring.
        self._next_size = _slot(capture, 5, ctypes.HRESULT, POINTER(ctypes.c_uint32), hold_gil=True)
        self._get_buffer = _slot(
            capture,
            3,
            ctypes.HRESULT,
            POINTER(c_void_p),
            POINTER(ctypes.c_uint32),
            POINTER(wintypes.DWORD),
            c_void_p,
            c_void_p,
            hold_gil=True,
        )
        self._release_buffer = _slot(capture, 4, ctypes.HRESULT, ctypes.c_uint32, hold_gil=True)
        self._size = ctypes.c_uint32()
        self._data = c_void_p()
        self._frames = ctypes.c_uint32()
        self._flags = wintypes.DWORD()
        _slot(client, 10, ctypes.HRESULT)(client)

    def wait(self, milliseconds: int) -> None:
        """Sleep until the device has a packet, or for `milliseconds`."""
        _kernel32.WaitForSingleObject(self._event, milliseconds)

    def read(self, sink: bytearray) -> int:
        """Append every waiting packet to `sink` as float32. Returns the frames added."""
        total = 0
        capture = self._capture
        while True:
            self._next_size(capture, byref(self._size))
            if not self._size.value:
                return total
            status = self._get_buffer(
                capture, byref(self._data), byref(self._frames), byref(self._flags), None, None
            )
            frames = self._frames.value
            if status == _AUDCLNT_S_BUFFER_EMPTY or not frames:
                return total
            if self._flags.value & _AUDCLNT_BUFFERFLAGS_SILENT or not self._data:
                # The buffer's contents are undefined when this flag is set.
                sink += bytes(frames * self.channels * 4)
            elif self.float32:
                sink += ctypes.string_at(self._data, frames * self._frame_bytes)
            else:
                pcm = np.frombuffer(
                    ctypes.string_at(self._data, frames * self._frame_bytes), dtype=np.int16
                )
                sink += (pcm.astype(np.float32) / 32768.0).tobytes()
            self._release_buffer(capture, frames)
            total += frames

    def close(self) -> None:
        if self._client:
            try:
                _slot(self._client, 11, ctypes.HRESULT)(self._client)  # Stop
            except OSError as exc:
                log.debug("error stopping the stream: %s", exc)
        _release(self._capture)
        _release(self._client)
        self._capture = c_void_p()
        self._client = c_void_p()
        if self._event:
            _kernel32.CloseHandle(self._event)
            self._event = None


class _Target:
    """The process being captured, held open so that its exit can be seen.

    Holding a handle also keeps Windows from giving the pid to something
    else while this still believes in it.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._lock = threading.Lock()
        self._handle = _kernel32.OpenProcess(
            _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        # A process this one may not open (elevated, protected) can still be
        # captured; it just has to be looked for in the process list instead.
        self._denied = not self._handle and ctypes.get_last_error() == _ERROR_ACCESS_DENIED

    def alive(self) -> bool:
        with self._lock:
            if self._handle:
                return _kernel32.WaitForSingleObject(self._handle, 0) == _WAIT_TIMEOUT
        if self._denied:
            try:
                return self.pid in _process_table()
            except OSError:
                return True
        return False

    def close(self) -> None:
        with self._lock:
            if self._handle:
                _kernel32.CloseHandle(self._handle)
                self._handle = None
            self._denied = False


class _Request:
    """Asks the capture thread to switch to `pid`; None means close."""

    def __init__(self, pid: int | None):
        self.pid = pid
        self.done = threading.Event()
        self.error: AudioDeviceError | None = None


# -- the capture -----------------------------------------------------------


class ProcessCapture(CaptureBase):
    """Captures one application into a queue of mono 16 kHz float32 blocks.

    `exe` is an image name such as "chrome.exe"; the instance to capture is
    found when start() is called, and found again if the app is closed and
    reopened. Passing `pid` pins the capture to that one process tree
    instead: when it exits the stream goes quiet and stays quiet.

    The stream never stops while this is running. A silent app, an app that
    has been closed and audio lost to a late read all come out as silence of
    the right length; silence_sec says how much of the output was that.
    """

    def __init__(
        self,
        exe: str,
        frames_per_buffer: int = 1024,
        max_queue_sec: float = 15.0,
        on_event: Callable[[str, str], None] | None = None,
        pid: int | None = None,
    ):
        # on_event kinds: "lost" when the app was closed (or its stream
        # failed and would not reopen), "restored" when it is back.
        super().__init__(frames_per_buffer, max_queue_sec, on_event)
        self.exe = exe
        self._label = display_name(exe)
        self._pinned = pid
        self.device_name = f"{self._label} (this app only)"
        # Silence put into the stream here rather than captured, in seconds,
        # and the part of it that stands in for audio the device dropped.
        self.silence_sec = 0.0
        self.lost_sec = 0.0
        self._late_reads = 0

        self._stream: _Stream | None = None
        self._pacer: Pacer | None = None
        self._block_bytes = 0
        self._requests: queue.SimpleQueue[_Request] = queue.SimpleQueue()
        self._target: _Target | None = None
        self._capturer: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        self._lost = False
        self._stop_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not supported():
            raise AudioDeviceError(
"capturing one app needs Windows 11."
            )
        pid = self._pinned if self._pinned is not None else find_app(self.exe)
        target = _Target(pid) if pid is not None else None
        if target is None or not target.alive():
            if target is not None:
                target.close()
            raise AudioDeviceError(f"{self._label} is not running.")
        self._target = target
        self._capturer = threading.Thread(
            target=self._capture_loop, name="appaudio-capture", daemon=True
        )
        self._capturer.start()
        try:
            self._switch(pid)
        except AudioDeviceError:
            self.stop()
            raise
        self._start_converter()
        self._watcher = threading.Thread(
            target=self._watch_loop, name="appaudio-watch", daemon=True
        )
        self._watcher.start()

    def stop(self) -> None:
        """Safe to call twice, and from any thread including on_event's."""
        self._stop.set()
        with self._stop_lock:
            current = threading.current_thread()
            for name in ("_capturer", "_watcher"):
                thread = getattr(self, name)
                if thread is not None and thread is not current:
                    thread.join(timeout=2.0)
                setattr(self, name, None)
            # Unblock the converter, then the consumer.
            self._stop_converter()
            if self._target is not None:
                self._target.close()
            self._close_output()

    @property
    def buffer_latency_sec(self) -> float:
        """Delay built into capture itself: the packets that make up one block."""
        if not self.native_rate:
            return 0.0
        packet = self.native_rate // 100
        return -(-self.frames_per_buffer // packet) * packet / self.native_rate

    # -- capture thread ----------------------------------------------------

    def _capture_loop(self) -> None:
        # Audio waiting to be handed on as one block, interleaved float32.
        pending = bytearray()
        with _Com(), _AudioPriority():
            try:
                while not self._stop.is_set():
                    self._serve_requests(pending)
                    stream = self._stream
                    mark = len(pending)
                    frames = 0
                    if stream is None:
                        self._stop.wait(WAIT_MS / 1000)
                    else:
                        stream.wait(WAIT_MS)
                        try:
                            frames = stream.read(pending)
                        except OSError as exc:
                            # The watcher sees there is no stream and reopens.
                            log.warning("capture of %s failed: %s", self._label, exc)
                            del pending[mark:]
                            stream.close()
                            self._stream = None
                    pacer = self._pacer
                    if pacer is None:
                        continue
                    now = time.perf_counter()
                    frame_bytes = self.channels * 4
                    if frames:
                        late = not pacer.synthesising
                        pad = pacer.packets(now, frames)
                        if pad:
                            pending[mark:mark] = bytes(pad * frame_bytes)
                            if late:
                                self.lost_sec += pad / pacer.rate
                                self._late_reads += 1
                                if self._late_reads % 50 == 1:
                                    log.warning(
                                        "capture thread ran late %d times, %.2f s of %s lost",
                                        self._late_reads,
                                        self.lost_sec,
                                        self._label,
                                    )
                    else:
                        pad = pacer.idle(now)
                        if pad:
                            pending += bytes(pad * frame_bytes)
                    if pad:
                        self.silence_sec += pad / pacer.rate
                    if len(pending) >= self._block_bytes:
                        self._push(now, bytes(pending))
                        pending.clear()
            except Exception:  # noqa: BLE001 - windowed builds have no stderr to die on
                log.exception("capture of %s stopped unexpectedly", self._label)
            finally:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None

    def _serve_requests(self, pending: bytearray) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if request.pid is not None:
                try:
                    self._open_stream(request.pid, pending)
                except OSError as exc:
                    request.error = AudioDeviceError(
                        f"Windows would not let {self._label} be captured ({exc})."
                    )
            request.done.set()

    def _open_stream(self, pid: int, pending: bytearray) -> None:
        stream = _Stream(pid)
        if (stream.rate, stream.channels) != (self.native_rate, self.channels):
            if pending and self.native_rate:
                self._push(time.perf_counter(), bytes(pending))
            pending.clear()
            self.native_rate = stream.rate
            self.channels = stream.channels
            self._block_bytes = self.frames_per_buffer * stream.channels * 4
            self._set_format(stream.rate, stream.channels)
            self._pacer = Pacer(stream.rate, time.perf_counter())
        self._stream = stream
        log.info(
            "capturing %r  pid %d  %d Hz  %d ch  %s  flags 0x%08X",
            self.device_name,
            pid,
            stream.rate,
            stream.channels,
            "float32" if stream.float32 else "16-bit",
            stream.flags,
        )

    def _switch(self, pid: int | None) -> None:
        """Have the capture thread open `pid`, or close with None. Blocks."""
        request = _Request(pid)
        self._requests.put(request)
        capturer = self._capturer
        while not request.done.wait(0.1):
            if self._stop.is_set() or capturer is None or not capturer.is_alive():
                raise AudioDeviceError(f"capture of {self._label} was stopped.")
        if request.error is not None:
            raise request.error

    # -- watcher thread ----------------------------------------------------

    def _watch_loop(self) -> None:
        """Notice the app closing, and pick it up again when it comes back."""
        with _Com():
            while not self._stop.wait(WATCH_INTERVAL_SEC):
                target = self._target
                alive = target is not None and target.alive()
                if alive and self._stream is not None:
                    continue
                if self._stop.is_set():
                    return
                try:
                    if not alive and not self._lost:
                        self._lost = True
                        self._switch(None)
                        log.warning("%s (pid %d) was closed", self._label, target.pid)
                        self.on_event("lost", f"{self._label} was closed")
                    if alive:
                        pid = target.pid
                    elif self._pinned is not None:
                        continue
                    else:
                        pid = find_app(self.exe)
                        if pid is None:
                            continue
                        fresh = _Target(pid)
                        if not fresh.alive():
                            fresh.close()
                            continue
                        self._target = fresh
                        target.close()
                    self._switch(pid)
                except AudioDeviceError as exc:
                    if self._stop.is_set():
                        return
                    if not self._lost:
                        self._lost = True
                        log.warning("capture of %s lost: %s", self._label, exc)
                        self.on_event("lost", str(exc))
                    continue
                self.reopens += 1
                if self._lost:
                    self._lost = False
                    self.on_event("restored", self.device_name)
