"""WASAPI loopback capture.

Loopback records what Windows is *playing* rather than what a microphone hears,
so it captures any app's output with no virtual cable and no user setup. The
default render device is resolved to its matching loopback device, opened at its
native rate, then downmixed to mono and resampled to 16 kHz for the model.

Three things here exist because a capture that only works on the happy path
fails silently on the others:

  - The output queue is bounded. If the consumer stalls - a model load, a
    slow pass on the CPU - the oldest audio is discarded and counted rather
    than accumulating until the process runs out of memory.
  - Every block carries the time it was captured, so the pipeline can report
    how stale the audio it is working on is, instead of assuming zero.
  - The default output device is watched. Switching from speakers to a
    headset leaves the old loopback stream open and perfectly healthy and
    completely silent; without the watcher the app would sit on "listening"
    forever with nothing to listen to.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from .config import SAMPLE_RATE

log = logging.getLogger(__name__)

# (capture time on the perf_counter clock, mono 16 kHz float32 samples)
Block = tuple[float, np.ndarray]


class AudioDeviceError(RuntimeError):
    """Raised when no usable WASAPI loopback device can be opened."""


def find_default_loopback(pa: pyaudio.PyAudio) -> dict:
    """Return device info for the loopback counterpart of the default speakers."""
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise AudioDeviceError("WASAPI host API unavailable") from exc

    default_index = wasapi.get("defaultOutputDevice", -1)
    if default_index < 0:
        raise AudioDeviceError(
            "Windows reports no default output device. Plug in or enable "
            "speakers or headphones."
        )

    speakers = pa.get_device_info_by_index(default_index)
    if speakers.get("isLoopbackDevice"):
        return speakers

    # Loopback devices are exposed as separate input devices whose name embeds
    # the render device's name, e.g. "Speakers (Realtek) [Loopback]".
    for candidate in pa.get_loopback_device_info_generator():
        if speakers["name"] in candidate["name"]:
            return candidate

    raise AudioDeviceError(
        f"no loopback device found for default output {speakers['name']!r}"
    )


def list_loopback_devices(pa: pyaudio.PyAudio) -> list[dict]:
    return list(pa.get_loopback_device_info_generator())


def default_output_id() -> str | None:
    """Windows' ID for the current default playback endpoint, or None.

    PortAudio enumerates devices once, when it initialises, so asking it
    again says nothing about what has changed since. This asks Core Audio
    directly. Plain ctypes against the COM vtables, because the alternative
    is a dependency (comtypes, pycaw) for two method calls.
    """
    try:
        import ctypes
        from ctypes import POINTER, byref, c_void_p, c_wchar_p, wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("a", wintypes.DWORD),
                ("b", wintypes.WORD),
                ("c", wintypes.WORD),
                ("d", ctypes.c_ubyte * 8),
            ]

        def guid(a, b, c, *d):
            return GUID(a, b, c, (ctypes.c_ubyte * 8)(*d))

        clsid = guid(0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E)
        iid = guid(0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6)

        ole32 = ctypes.OleDLL("ole32")
        # S_FALSE / RPC_E_CHANGED_MODE both mean COM is already up on this
        # thread, which is all that is needed.
        try:
            ole32.CoInitializeEx(None, 0)
        except OSError:
            pass

        enumerator = c_void_p()
        ole32.CoCreateInstance(byref(clsid), None, 23, byref(iid), byref(enumerator))
        if not enumerator:
            return None

        def method(obj, index, restype, *argtypes):
            vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
            return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtable[index])

        release = method(enumerator, 2, ctypes.c_ulong)
        try:
            device = c_void_p()
            # eRender = 0, eConsole = 0
            get_default = method(
                enumerator, 4, ctypes.HRESULT, ctypes.c_int, ctypes.c_int, POINTER(c_void_p)
            )
            get_default(enumerator, 0, 0, byref(device))
            if not device:
                return None
            try:
                ident = c_wchar_p()
                method(device, 5, ctypes.HRESULT, POINTER(c_wchar_p))(device, byref(ident))
                value = ident.value
                free = ctypes.windll.ole32.CoTaskMemFree
                free.restype = None
                free.argtypes = [c_void_p]
                free(ctypes.cast(ident, c_void_p))
                return value
            finally:
                method(device, 2, ctypes.c_ulong)(device)
        finally:
            release(enumerator)
    except Exception:  # noqa: BLE001 - no endpoint, or not Windows: nothing to compare
        return None


class LoopbackCapture:
    """Captures system audio into a queue of mono 16 kHz float32 blocks.

    The PyAudio callback runs on a realtime-ish thread, so it only copies bytes
    into a queue. Downmix and resample happen on the consumer thread, where a
    stall costs latency instead of dropped frames.
    """

    def __init__(
        self,
        frames_per_buffer: int = 1024,
        device_index: int | None = None,
        max_queue_sec: float = 15.0,
        on_event: Callable[[str, str], None] | None = None,
        watch_device: bool = True,
    ):
        self.frames_per_buffer = frames_per_buffer
        self.device_index = device_index
        self.max_queue_sec = max_queue_sec
        # (kind, message): "changed" when the default device moved and capture
        # followed it, "lost" when it could not be reopened, "restored" when a
        # lost device came back.
        self.on_event = on_event or (lambda _kind, _msg: None)
        self.watch_device = watch_device and device_index is None

        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._raw: queue.Queue = queue.Queue(maxsize=256)
        self._out: deque[Block] = deque()
        self._out_samples = 0
        self._cond = threading.Condition()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._device_lock = threading.Lock()
        self._endpoint_id: str | None = None
        self._lost = False

        self.device_name = ""
        self.native_rate = 0
        self.channels = 0
        # Callback blocks dropped because the converter was starved.
        self.dropped_blocks = 0
        # Converted audio discarded because the consumer fell too far behind.
        self.dropped_sec = 0.0
        self.reopens = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._open()
        self._worker = threading.Thread(
            target=self._convert_loop, name="audio-convert", daemon=True
        )
        self._worker.start()
        if self.watch_device:
            self._endpoint_id = default_output_id()
            self._watcher = threading.Thread(
                target=self._watch_loop, name="audio-watch", daemon=True
            )
            self._watcher.start()

    def _open(self) -> None:
        """Open the loopback stream on the current default device."""
        with self._device_lock:
            self._pa = pyaudio.PyAudio()
            try:
                if self.device_index is None:
                    info = find_default_loopback(self._pa)
                else:
                    info = self._pa.get_device_info_by_index(self.device_index)
                    if not info.get("isLoopbackDevice"):
                        raise AudioDeviceError(
                            f"device {self.device_index} is not a loopback device"
                        )

                self.device_name = str(info["name"])
                self.native_rate = int(info["defaultSampleRate"])
                self.channels = int(info["maxInputChannels"])
                log.info(
                    "capturing %r  %d Hz  %d ch",
                    self.device_name,
                    self.native_rate,
                    self.channels,
                )
                # Tell the converter the format before any audio in it arrives.
                self._raw.put(("format", self.native_rate, self.channels))
                self._stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.native_rate,
                    input=True,
                    input_device_index=int(info["index"]),
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=self._callback,
                )
                self._stream.start_stream()
            except Exception as exc:
                self._close_device()
                if isinstance(exc, AudioDeviceError):
                    raise
                raise AudioDeviceError(f"could not open the audio device: {exc}") from exc

    def _close_device(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except OSError as exc:  # stream may already be dead after a device change
                log.debug("error closing stream: %s", exc)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except OSError as exc:
                log.debug("error terminating PortAudio: %s", exc)
            self._pa = None

    def stop(self) -> None:
        self._stop.set()
        with self._device_lock:
            self._close_device()
        # Unblock the converter, then the consumer.
        try:
            self._raw.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._watcher is not None:
            self._watcher.join(timeout=2.0)
            self._watcher = None
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # -- data --------------------------------------------------------------

    def read_all(self, timeout: float | None = None) -> list[Block] | None:
        """Every block that is ready, oldest first. Returns None once stopped.

        Waits up to `timeout` for the first one and then takes whatever else
        has arrived without waiting further. Taking everything is the point:
        the caller is about to spend a few hundred milliseconds on inference,
        and audio left behind here would be that much staler when it is
        finally looked at.
        """
        with self._cond:
            if not self._out and not self._closed:
                self._cond.wait(timeout)
            if not self._out:
                return None if self._closed else []
            blocks = list(self._out)
            self._out.clear()
            self._out_samples = 0
            return blocks

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Block for the next mono 16 kHz block. Returns None once stopped."""
        with self._cond:
            if not self._out and not self._closed:
                self._cond.wait(timeout)
            if not self._out:
                return None if self._closed else np.empty(0, dtype=np.float32)
            _t, block = self._out.popleft()
            self._out_samples -= block.size
            return block

    def pending_blocks(self) -> int:
        return len(self._out)

    def pending_sec(self) -> float:
        """Converted audio waiting to be read."""
        return self._out_samples / SAMPLE_RATE

    @property
    def buffer_latency_sec(self) -> float:
        """Delay built into capture itself: one device buffer."""
        if not self.native_rate:
            return 0.0
        return self.frames_per_buffer / self.native_rate

    # -- internals ---------------------------------------------------------

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ARG002
        if status:
            log.debug("stream status flags: %s", status)
        try:
            self._raw.put_nowait((time.perf_counter(), in_data))
        except queue.Full:
            # Converter is starved of CPU. Dropping is better than growing an
            # unbounded backlog behind a live stream.
            self.dropped_blocks += 1
            if self.dropped_blocks % 50 == 1:
                log.warning("audio backlog full, dropped %d blocks", self.dropped_blocks)
        return (None, pyaudio.paContinue)

    def _convert_loop(self) -> None:
        resampler = None
        channels = 1
        max_samples = int(self.max_queue_sec * SAMPLE_RATE)
        while not self._stop.is_set():
            item = self._raw.get()
            if item is None:
                break
            if item[0] == "format":
                # ResampleStream keeps filter state across chunks, so block
                # boundaries don't produce clicks the way independent per-chunk
                # resampling does. A new device means a new one.
                _tag, rate, channels = item
                resampler = soxr.ResampleStream(
                    rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
                )
                continue
            if resampler is None:
                continue
            stamp, block = item
            frames = np.frombuffer(block, dtype=np.float32)
            if channels > 1:
                # Trailing partial frame would break the reshape; drop it.
                usable = (frames.size // channels) * channels
                mono = frames[:usable].reshape(-1, channels).mean(axis=1)
            else:
                mono = frames
            if mono.size == 0:
                continue
            out = resampler.resample_chunk(np.ascontiguousarray(mono, dtype=np.float32))
            if not out.size:
                continue
            out = out.astype(np.float32, copy=False)
            with self._cond:
                self._out.append((stamp, out))
                self._out_samples += out.size
                while self._out_samples > max_samples and len(self._out) > 1:
                    _t, old = self._out.popleft()
                    self._out_samples -= old.size
                    self.dropped_sec += old.size / SAMPLE_RATE
                self._cond.notify()

    def _watch_loop(self) -> None:
        """Follow the default output device when Windows changes it."""
        while not self._stop.wait(1.5):
            current = default_output_id()
            stream_dead = False
            with self._device_lock:
                if self._stream is not None:
                    try:
                        stream_dead = not self._stream.is_active()
                    except OSError:
                        stream_dead = True
            moved = current is not None and current != self._endpoint_id
            if not (moved or stream_dead or self._lost):
                continue
            if self._stop.is_set():
                return
            previous = self.device_name
            with self._device_lock:
                self._close_device()
            try:
                self._open()
            except AudioDeviceError as exc:
                if not self._lost:
                    self._lost = True
                    log.warning("audio device lost: %s", exc)
                    self.on_event("lost", str(exc))
                continue
            self._endpoint_id = current
            self.reopens += 1
            if self._lost:
                self._lost = False
                self.on_event("restored", self.device_name)
            elif self.device_name != previous:
                self.on_event("changed", self.device_name)
