"""Per-application capture, as far as it can be tested without a sound card.

The Windows calls themselves are exercised by tools.app_audio_check against
real audio. What is here is everything around them: which process stands for
an app, how much silence keeps a stream in step with the clock, the layout of
the structures Windows is handed, and the capture and watcher threads run
against a scripted stand-in for the device.
"""

import ctypes
import os
import struct
import threading
import time
import unittest
from ctypes import POINTER, byref, c_void_p
from unittest import mock

import numpy as np

from subordinant import appaudio
from subordinant.appaudio import (
    MAX_FILL_SEC,
    SILENCE_GAP_SEC,
    SILENCE_SLACK_SEC,
    AudioApp,
    Pacer,
    Proc,
    ProcessCapture,
    display_name,
    group_sessions,
    pick_root,
    root_pid,
)
from subordinant.audio import AudioDeviceError, CaptureBase
from subordinant.config import SAMPLE_RATE

# A desktop: a browser with its helpers, a chat client that spells its own
# name two ways, a game, and two separate copies of a media player.
TABLE = {
    0: Proc(0, "[System Process]"),
    4: Proc(0, "System"),
    100: Proc(50, "explorer.exe"),  # parent 50 exited long ago
    200: Proc(100, "chrome.exe"),
    201: Proc(200, "chrome.exe"),
    202: Proc(201, "chrome.exe"),
    300: Proc(100, "Discord.exe"),
    301: Proc(300, "discord.exe"),
    400: Proc(100, "game.exe"),
    500: Proc(100, "mpv.exe"),
    510: Proc(100, "mpv.exe"),
}


class Roots(unittest.TestCase):
    def test_child_resolves_to_topmost_process_of_the_same_program(self):
        self.assertEqual(root_pid(202, TABLE), 200)
        self.assertEqual(root_pid(201, TABLE), 200)

    def test_root_is_its_own_root(self):
        self.assertEqual(root_pid(200, TABLE), 200)
        self.assertEqual(root_pid(400, TABLE), 400)

    def test_stops_at_a_parent_that_is_another_program(self):
        # explorer started chrome; that does not make chrome part of explorer.
        self.assertEqual(TABLE[root_pid(202, TABLE)].exe, "chrome.exe")

    def test_image_names_compare_without_case(self):
        self.assertEqual(root_pid(301, TABLE), 300)

    def test_unknown_pid_is_returned_as_is(self):
        self.assertEqual(root_pid(9999, TABLE), 9999)

    def test_parent_that_no_longer_exists(self):
        self.assertEqual(root_pid(100, TABLE), 100)

    def test_parent_pid_reused_by_something_else(self):
        table = {10: Proc(20, "vlc.exe"), 20: Proc(1, "notepad.exe")}
        self.assertEqual(root_pid(10, table), 10)

    def test_a_cycle_does_not_hang(self):
        table = {1: Proc(2, "a.exe"), 2: Proc(1, "a.exe")}
        self.assertIn(root_pid(1, table), (1, 2))
        self.assertEqual(root_pid(0, TABLE), 0)  # the idle process is its own parent


class Grouping(unittest.TestCase):
    def test_one_entry_per_program_active_first_then_alphabetical(self):
        sessions = [(0, True), (202, False), (301, False), (400, True), (500, False)]
        apps = group_sessions(sessions, TABLE)
        self.assertEqual(
            apps,
            [
                AudioApp("game.exe", "game", 400, True),
                AudioApp("chrome.exe", "chrome", 200, False),
                AudioApp("Discord.exe", "Discord", 300, False),
                AudioApp("mpv.exe", "mpv", 500, False),
            ],
        )

    def test_helper_processes_collapse_into_their_app(self):
        apps = group_sessions([(201, False), (202, True)], TABLE)
        self.assertEqual(apps, [AudioApp("chrome.exe", "chrome", 200, True)])

    def test_skips_system_sounds_own_process_and_the_dead(self):
        apps = group_sessions([(0, True), (400, True), (9999, True)], TABLE, skip_pid=400)
        self.assertEqual(apps, [])

    def test_of_two_instances_the_one_playing_is_reported(self):
        apps = group_sessions([(500, False), (510, True)], TABLE)
        self.assertEqual(apps, [AudioApp("mpv.exe", "mpv", 510, True)])
        apps = group_sessions([(510, True), (500, False)], TABLE)
        self.assertEqual(apps, [AudioApp("mpv.exe", "mpv", 510, True)])


class Finding(unittest.TestCase):
    def test_none_when_not_running(self):
        self.assertIsNone(pick_root("firefox.exe", TABLE))

    def test_finds_the_root_not_the_helpers(self):
        self.assertEqual(pick_root("chrome.exe", TABLE), 200)

    def test_name_is_forgiving(self):
        self.assertEqual(pick_root("CHROME.EXE", TABLE), 200)
        self.assertEqual(pick_root("chrome", TABLE), 200)
        self.assertEqual(pick_root("discord.exe", TABLE), 300)

    def test_prefers_the_instance_with_an_audio_session(self):
        self.assertEqual(pick_root("mpv.exe", TABLE), 500)
        self.assertEqual(pick_root("mpv.exe", TABLE, [(510, False)]), 510)

    def test_prefers_the_instance_that_is_playing(self):
        self.assertEqual(pick_root("mpv.exe", TABLE, [(500, False), (510, True)]), 510)

    def test_sessions_of_other_programs_do_not_count(self):
        self.assertEqual(pick_root("mpv.exe", TABLE, [(400, True), (9999, True)]), 500)

    def test_session_in_a_helper_counts_for_its_app(self):
        table = dict(TABLE)
        table.update({600: Proc(100, "chrome.exe"), 601: Proc(600, "chrome.exe")})
        self.assertEqual(pick_root("chrome.exe", table, [(601, True)]), 600)


class Names(unittest.TestCase):
    def test_display_name(self):
        self.assertEqual(display_name("chrome.exe"), "chrome")
        self.assertEqual(display_name("Discord.EXE"), "Discord")
        self.assertEqual(display_name("C:\\Program Files\\mpv\\mpv.exe"), "mpv")
        self.assertEqual(display_name("Code - Insiders.exe"), "Code - Insiders")
        self.assertEqual(display_name("System"), "System")

    def test_display_name_never_raises(self):
        for odd in ("", "   ", ".exe", None, 42, b"chrome.exe"):
            self.assertIsInstance(display_name(odd), str)

    def test_picker_status_line_and_notices_agree(self):
        name = display_name("chrome.exe")
        self.assertEqual(group_sessions([(202, True)], TABLE)[0].name, name)
        self.assertEqual(ProcessCapture("chrome.exe").device_name, f"{name} (this app only)")


RATE = 48000
PACKET = RATE // 100  # the device delivers 10 ms at a time


def run_packets(pacer: Pacer, start: float, seconds: float, period: float = 0.01):
    """Feed steady packets. Returns (time of the last one, silence the pacer asked for)."""
    pads = 0
    now = start
    for i in range(1, int(round(seconds / 0.01)) + 1):
        now = start + i * period
        pads += pacer.packets(now, PACKET)
    return now, pads


class Pacing(unittest.TestCase):
    def test_steady_stream_needs_no_silence(self):
        pacer = Pacer(RATE, 100.0)
        _now, pads = run_packets(pacer, 100.0, 30.0)
        self.assertEqual(pads, 0)

    def test_jittery_but_complete_stream_needs_no_silence(self):
        # Woken 25 ms late, three packets at once, then back on time.
        pacer = Pacer(RATE, 0.0)
        now, pads = run_packets(pacer, 0.0, 1.0)
        pads += pacer.packets(now + 0.035, 3 * PACKET)
        pads += pacer.packets(now + 0.040, PACKET)
        self.assertEqual(pads, 0)

    def test_a_late_packet_is_not_yet_silence(self):
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        self.assertEqual(pacer.idle(now + SILENCE_GAP_SEC - 0.005), 0)
        self.assertFalse(pacer.synthesising)

    def test_silence_is_filled_to_just_short_of_now(self):
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        first = pacer.idle(now + 0.05)
        self.assertEqual(first, int((0.05 - SILENCE_SLACK_SEC) * RATE))
        self.assertTrue(pacer.synthesising)
        # ...and keeps pace from there on, 20 ms at a time.
        total = first
        for i in range(1, 51):
            total += pacer.idle(now + 0.05 + i * 0.02)
        self.assertAlmostEqual(total / RATE, 1.05 - SILENCE_SLACK_SEC, places=3)

    def test_never_runs_ahead_of_the_clock(self):
        pacer = Pacer(RATE, 0.0)
        emitted = 0
        now = 0.0
        for step in range(2000):
            now += 0.003 + (step % 7) * 0.004
            if 300 <= step < 900 or step >= 1500:
                emitted += pacer.idle(now)
            else:
                emitted += pacer.packets(now, PACKET) + PACKET if step % 3 == 0 else pacer.idle(now)
            self.assertLessEqual(emitted, int(now * RATE) + PACKET)

    def test_silent_from_the_start(self):
        pacer = Pacer(RATE, 5.0)
        total = sum(pacer.idle(5.0 + i * 0.02) for i in range(1, 101))
        self.assertAlmostEqual(total / RATE, 2.0 - SILENCE_SLACK_SEC, places=3)

    def test_sound_resuming_lines_up_with_the_clock(self):
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        silence = sum(pacer.idle(now + i * 0.02) for i in range(1, 51))  # one second
        resumed = now + 1.0 + 0.013
        pad = pacer.packets(resumed, PACKET)
        self.assertFalse(pacer.synthesising)
        # A second of packets, the silence, the top-up and the new packet
        # add up to exactly the time that has passed.
        total = RATE + silence + pad + PACKET
        self.assertAlmostEqual(total / RATE, resumed, places=3)
        # ...and the stream then carries on with nothing added.
        _now, pads = run_packets(pacer, resumed, 5.0)
        self.assertEqual(pads, 0)

    def test_audio_lost_to_a_late_read_is_replaced(self):
        # Stalled for 300 ms; the device kept only the last three packets.
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        pad = pacer.packets(now + 0.3, 3 * PACKET)
        self.assertAlmostEqual(pad / RATE, 0.27, places=3)
        _now, pads = run_packets(pacer, now + 0.3, 5.0)
        self.assertEqual(pads, 0)

    def test_small_losses_add_up_until_they_are_worth_replacing(self):
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        pads = []
        for _ in range(4):
            now += 0.045  # 45 ms late: three packets survive, 15 ms does not
            pads.append(pacer.packets(now, 3 * PACKET))
            now, more = run_packets(pacer, now, 0.2)
            pads.append(more)
        self.assertEqual(pads[0], 0)
        self.assertAlmostEqual(sum(pads) / RATE, 0.06, delta=SILENCE_GAP_SEC)

    def test_a_slow_device_clock_is_not_mistaken_for_lost_audio(self):
        # 300 parts per million slow, for an hour: a second short by the end.
        pacer = Pacer(RATE, 0.0)
        _now, pads = run_packets(pacer, 0.0, 3600.0, period=0.01 * 1.0003)
        self.assertEqual(pads, 0)

    def test_a_fast_device_clock_only_delays_the_next_silence(self):
        pacer = Pacer(RATE, 0.0)
        now, pads = run_packets(pacer, 0.0, 600.0, period=0.01 * 0.9997)
        self.assertEqual(pads, 0)
        total = sum(pacer.idle(now + i * 0.02) for i in range(1, 51))
        self.assertAlmostEqual(total / RATE, 1.0 - SILENCE_SLACK_SEC, delta=0.02)

    def test_a_suspended_pc_is_not_paid_back(self):
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        woke = now + 3 * 3600
        self.assertEqual(pacer.idle(woke), int(MAX_FILL_SEC * RATE))
        self.assertEqual(pacer.idle(woke + 0.02), int(0.02 * RATE))
        pacer = Pacer(RATE, 0.0)
        now, _ = run_packets(pacer, 0.0, 1.0)
        self.assertEqual(pacer.packets(now + 3 * 3600, PACKET), int(MAX_FILL_SEC * RATE))


class Structures(unittest.TestCase):
    def test_activation_params_are_twelve_bytes(self):
        self.assertEqual(ctypes.sizeof(appaudio.AUDIOCLIENT_ACTIVATION_PARAMS), 12)

    @unittest.skipUnless(ctypes.sizeof(c_void_p) == 8, "64-bit layout")
    def test_propvariant_layout(self):
        self.assertEqual(ctypes.sizeof(appaudio.PROPVARIANT), 24)
        self.assertEqual(appaudio.PROPVARIANT.blob.offset, 8)
        self.assertEqual(appaudio.PROPVARIANT.blob.offset + appaudio.BLOB.pBlobData.offset, 16)

    def test_activation_blob_names_the_process_and_its_tree(self):
        variant, params = appaudio.activation_params(4321)
        self.assertEqual(variant.vt, 65)  # VT_BLOB
        self.assertEqual(variant.blob.cbSize, 12)
        self.assertEqual(variant.blob.pBlobData, ctypes.addressof(params))
        # PROCESS_LOOPBACK, pid, INCLUDE_TARGET_PROCESS_TREE
        self.assertEqual(
            ctypes.string_at(variant.blob.pBlobData, 12), struct.pack("<iIi", 1, 4321, 0)
        )

    def test_wave_formats(self):
        self.assertEqual(ctypes.sizeof(appaudio.WAVEFORMATEX), 18)
        wave = appaudio.wave_format(48000, 2, float32=True)
        self.assertEqual(
            bytes(wave), struct.pack("<HHIIHHH", 3, 2, 48000, 384000, 8, 32, 0)
        )
        wave = appaudio.wave_format(48000, 2, float32=False)
        self.assertEqual(
            bytes(wave), struct.pack("<HHIIHHH", 1, 2, 48000, 192000, 4, 16, 0)
        )

    def test_guids_have_the_windows_byte_order(self):
        self.assertEqual(
            bytes(appaudio._IID_AGILE_OBJECT),
            bytes.fromhex("942bea94cce9e049c0ffee64ca8f5b90"),
        )

    @unittest.skipUnless(ctypes.sizeof(c_void_p) == 8, "64-bit layout")
    def test_process_entry_size(self):
        self.assertEqual(ctypes.sizeof(appaudio.PROCESSENTRY32W), 568)


class CompletionHandler(unittest.TestCase):
    """The COM object Windows calls back, driven the way Windows drives it."""

    def setUp(self):
        self.completion = appaudio._Completion()
        self.pointer = c_void_p(self.completion.pointer)

    def query(self, iid):
        out = c_void_p()
        result = appaudio._slot(
            self.pointer, 0, ctypes.c_long, POINTER(appaudio.GUID), POINTER(c_void_p)
        )(self.pointer, byref(iid), byref(out))
        return result, out.value

    def test_answers_for_iagileobject(self):
        # Without this, activation fails with E_ILLEGAL_METHOD_CALL.
        for iid in (
            appaudio._IID_AGILE_OBJECT,
            appaudio._IID_UNKNOWN,
            appaudio._IID_COMPLETION_HANDLER,
        ):
            result, out = self.query(iid)
            self.assertEqual(result, 0)
            self.assertEqual(out, self.completion.pointer)
        self.assertEqual(self.completion.refs, 4)

    def test_refuses_other_interfaces(self):
        result, out = self.query(appaudio._IID_AUDIO_CLIENT)
        self.assertEqual(result & 0xFFFFFFFF, 0x80004002)  # E_NOINTERFACE
        self.assertIsNone(out)
        self.assertEqual(self.completion.refs, 1)

    def test_completion_sets_the_event(self):
        self.assertFalse(self.completion.done.is_set())
        result = appaudio._slot(self.pointer, 3, ctypes.c_long, c_void_p)(self.pointer, None)
        self.assertEqual(result, 0)
        self.assertTrue(self.completion.done.is_set())

    def test_lives_until_the_last_reference_goes(self):
        add_ref = appaudio._slot(self.pointer, 1, ctypes.c_ulong)
        release = appaudio._slot(self.pointer, 2, ctypes.c_ulong)
        self.assertEqual(add_ref(self.pointer), 2)
        self.completion.release()  # ours; Windows still holds one
        self.assertIn(self.completion.pointer, appaudio._completions)
        self.assertEqual(release(self.pointer), 0)
        self.assertNotIn(self.completion.pointer, appaudio._completions)
        # A stray call after that is answered, not followed into freed memory.
        self.assertEqual(release(self.pointer), 0)

    def tearDown(self):
        appaudio._completions.pop(self.completion.pointer, None)


class Feed(CaptureBase):
    """The smallest possible source: whatever the test pushes."""

    def start(self, rate=48000, channels=2):
        self.native_rate, self.channels = rate, channels
        self._set_format(rate, channels)
        self._start_converter()

    def stop(self):
        self._stop.set()
        self._stop_converter()
        self._close_output()


def drain(capture, seconds: float) -> list:
    blocks = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        got = capture.read_all(timeout=0.05)
        if got is None:
            break
        blocks.extend(got)
    return blocks


class Converter(unittest.TestCase):
    def test_downmixes_and_resamples_to_mono_16k(self):
        feed = Feed()
        feed.start()
        stereo = np.empty(960, dtype=np.float32)
        stereo[0::2], stereo[1::2] = 0.5, 0.1
        for i in range(100):  # one second
            feed._push(float(i), stereo.tobytes())
        blocks = drain(feed, 0.5)
        feed.stop()
        audio = np.concatenate([b for _t, b in blocks])
        self.assertAlmostEqual(audio.size / SAMPLE_RATE, 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.median(audio)), 0.3, places=3)
        self.assertEqual(audio.dtype, np.float32)
        stamps = [t for t, _b in blocks]
        self.assertEqual(stamps, sorted(stamps))
        self.assertLessEqual(stamps[-1], 99.0)

    def test_queue_is_bounded_in_seconds(self):
        feed = Feed(max_queue_sec=0.2)
        feed.start(channels=1)
        block = np.zeros(480, dtype=np.float32).tobytes()
        for _ in range(200):  # two seconds, nobody reading
            feed._push(0.0, block)
            time.sleep(0.0005)
        time.sleep(0.2)
        self.assertLess(feed.pending_sec(), 0.25)
        self.assertGreater(feed.dropped_sec, 1.0)
        feed.stop()

    def test_stop_ends_the_stream(self):
        feed = Feed()
        feed.start()
        feed.stop()
        self.assertIsNone(feed.read_all(timeout=0.01))
        self.assertIsNone(feed.read(timeout=0.01))


LEVEL = 0.25
LOUD_PACKET = np.full(PACKET * 2, LEVEL, dtype=np.float32).tobytes()


class FakeStream:
    """The device, scripted: 10 ms stereo packets while `playing(t)` is true.

    Unlike the real one it never drops a packet, so a slow test machine
    changes nothing but the timing.
    """

    rate = 48000
    channels = 2
    float32 = True
    flags = 0
    playing = staticmethod(lambda _t: True)
    opened: list = []

    def __init__(self, pid: int):
        FakeStream.opened.append(pid)
        self._start = time.perf_counter()
        self._delivered = 0
        self.closed = False

    def wait(self, _milliseconds: int) -> None:
        time.sleep(0.004)

    def read(self, sink: bytearray) -> int:
        frames = 0
        due = int((time.perf_counter() - self._start) * 100)
        while self._delivered < due:
            at = self._delivered / 100
            self._delivered += 1
            if FakeStream.playing(at):
                sink += LOUD_PACKET
                frames += PACKET
        return frames

    def close(self) -> None:
        self.closed = True


class FakeTarget:
    alive_pids: set = set()

    def __init__(self, pid: int):
        self.pid = pid

    def alive(self) -> bool:
        return self.pid in FakeTarget.alive_pids

    def close(self) -> None:
        pass


class CaptureCase(unittest.TestCase):
    """ProcessCapture's own threads, with the device and the process faked."""

    def setUp(self):
        FakeStream.opened = []
        FakeStream.playing = staticmethod(lambda _t: True)
        FakeTarget.alive_pids = {4242}
        self.found = [4242]
        self.events: list = []
        self.happened = threading.Condition()
        for name, value in (
            ("_Stream", FakeStream),
            ("_Target", FakeTarget),
            ("supported", lambda: True),
            ("find_app", lambda _exe: self.found[0]),
            ("WATCH_INTERVAL_SEC", 0.05),
        ):
            patcher = mock.patch.object(appaudio, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def capture(self, **kwargs) -> ProcessCapture:
        cap = ProcessCapture("chrome.exe", on_event=self.on_event, **kwargs)
        self.addCleanup(cap.stop)
        return cap

    def on_event(self, kind, message):
        with self.happened:
            self.events.append((kind, message))
            self.happened.notify_all()

    def wait_for(self, count: int, timeout: float = 3.0) -> None:
        with self.happened:
            self.happened.wait_for(lambda: len(self.events) >= count, timeout)


class Capture(CaptureCase):
    def test_reports_what_it_opened(self):
        cap = self.capture()
        cap.start()
        self.assertEqual(FakeStream.opened, [4242])
        self.assertEqual(cap.device_name, "chrome (this app only)")
        self.assertEqual((cap.native_rate, cap.channels), (48000, 2))
        # 1024 frames take three 10 ms packets to arrive.
        self.assertAlmostEqual(cap.buffer_latency_sec, 0.03)

    def test_audio_comes_through_stamped(self):
        cap = self.capture()
        started = time.perf_counter()
        cap.start()
        blocks = drain(cap, 0.6)
        cap.stop()
        audio = np.concatenate([b for _t, b in blocks])
        self.assertAlmostEqual(audio.size / SAMPLE_RATE, 0.6, delta=0.15)
        self.assertAlmostEqual(float(np.median(audio)), LEVEL, places=3)
        for stamp, _block in blocks:
            self.assertTrue(started <= stamp <= time.perf_counter())
        self.assertEqual(cap.silence_sec, 0.0)

    def test_stream_keeps_time_while_the_app_is_silent(self):
        # Sound, 0.6 s of nothing at all from the device, sound again.
        FakeStream.playing = staticmethod(lambda t: not 0.4 <= t < 1.0)
        cap = self.capture()
        cap.start()
        started = time.perf_counter()
        blocks = drain(cap, 1.5)
        wall = time.perf_counter() - started
        cap.stop()
        audio = np.concatenate([b for _t, b in blocks])
        self.assertAlmostEqual(audio.size / SAMPLE_RATE, wall, delta=0.15)
        self.assertAlmostEqual(cap.silence_sec, 0.6, delta=0.1)
        self.assertEqual(cap.lost_sec, 0.0)
        loud = np.abs(audio) > LEVEL / 2
        edges = np.flatnonzero(np.diff(loud.astype(np.int8))) / SAMPLE_RATE
        # loud -> quiet -> loud, and the quiet is as long as it really was
        self.assertEqual(len(edges), 2, edges)
        self.assertAlmostEqual(edges[1] - edges[0], 0.6, delta=0.06)
        self.assertAlmostEqual(edges[0], 0.4, delta=0.1)

    def test_not_running_is_an_audio_device_error(self):
        self.found[0] = None
        cap = self.capture()
        with self.assertRaises(AudioDeviceError) as raised:
            cap.start()
        self.assertEqual(str(raised.exception), "chrome is not running.")
        self.assertEqual(FakeStream.opened, [])

    def test_found_but_already_gone(self):
        FakeTarget.alive_pids = set()
        with self.assertRaises(AudioDeviceError):
            self.capture().start()

    def test_old_windows_is_an_audio_device_error(self):
        with mock.patch.object(appaudio, "supported", lambda: False):
            with self.assertRaises(AudioDeviceError) as raised:
                self.capture().start()
        self.assertIn("Windows 11", str(raised.exception))

    def test_refusal_by_windows_is_an_audio_device_error_and_leaves_no_threads(self):
        def refuse(_pid):
            raise OSError(-2147024891, "Access is denied")

        before = threading.active_count()
        with mock.patch.object(appaudio, "_Stream", refuse):
            cap = self.capture()
            with self.assertRaises(AudioDeviceError) as raised:
                cap.start()
        self.assertIn("chrome", str(raised.exception))
        self.assertEqual(threading.active_count(), before)
        self.assertIsNone(cap.read_all(timeout=0.01))

    def test_stop_twice_and_before_start(self):
        cap = self.capture()
        cap.stop()
        cap.stop()
        self.assertIsNone(cap.read_all(timeout=0.01))
        cap = self.capture()
        cap.start()
        before = threading.active_count()
        cap.stop()
        cap.stop()
        self.assertEqual(threading.active_count(), before - 3)
        drain(cap, 0.2)
        self.assertIsNone(cap.read_all(timeout=0.01))

    def test_stop_from_another_thread_while_reading(self):
        cap = self.capture()
        cap.start()
        threading.Timer(0.2, cap.stop).start()
        deadline = time.monotonic() + 3.0
        while cap.read_all(timeout=0.05) is not None:
            self.assertLess(time.monotonic(), deadline, "read_all never saw the stop")


class Watcher(CaptureCase):
    def test_app_closed_then_reopened(self):
        cap = self.capture()
        cap.start()
        FakeTarget.alive_pids = set()
        self.found[0] = None
        self.wait_for(1)
        self.assertEqual(self.events, [("lost", "chrome was closed")])
        # Still a stream, though there is no app: silence at the pace of the clock.
        drain(cap, 0.1)
        before = cap.silence_sec
        audio = np.concatenate([b for _t, b in drain(cap, 0.5)])
        self.assertAlmostEqual(audio.size / SAMPLE_RATE, 0.5, delta=0.15)
        self.assertEqual(float(np.max(np.abs(audio))), 0.0)
        self.assertGreater(cap.silence_sec, before + 0.3)
        self.assertEqual(cap.reopens, 0)

        FakeTarget.alive_pids = {5151}
        self.found[0] = 5151
        self.wait_for(2)
        self.assertEqual(self.events[1], ("restored", "chrome (this app only)"))
        self.assertEqual(FakeStream.opened, [4242, 5151])
        self.assertEqual(cap.reopens, 1)
        drain(cap, 0.2)
        audio = np.concatenate([b for _t, b in drain(cap, 0.3)])
        self.assertAlmostEqual(float(np.median(audio)), LEVEL, places=3)

    def test_pinned_pid_is_not_replaced_by_another_instance(self):
        cap = self.capture(pid=4242)
        cap.start()
        FakeTarget.alive_pids = {5151}
        self.found[0] = 5151
        self.wait_for(1)
        self.assertEqual(self.events, [("lost", "chrome was closed")])
        time.sleep(0.3)
        self.assertEqual(FakeStream.opened, [4242])
        self.assertEqual(cap.reopens, 0)

    def test_failed_stream_is_reopened_without_fuss(self):
        cap = self.capture()
        cap.start()
        first = cap._stream
        with mock.patch.object(first, "read", side_effect=OSError("device gone")):
            deadline = time.monotonic() + 3.0
            while cap.reopens == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(cap.reopens, 1)
        self.assertTrue(first.closed)
        self.assertEqual(FakeStream.opened, [4242, 4242])
        self.assertEqual(self.events, [])  # the app never went away
        drain(cap, 0.2)
        audio = np.concatenate([b for _t, b in drain(cap, 0.3)])
        self.assertAlmostEqual(float(np.median(audio)), LEVEL, places=3)

    def test_reopen_that_fails_is_reported_once_and_retried(self):
        cap = self.capture()
        cap.start()
        first = cap._stream

        def refuse(_pid):
            raise OSError(-2147024891, "Access is denied")

        with mock.patch.object(appaudio, "_Stream", refuse):
            with mock.patch.object(first, "read", side_effect=OSError("device gone")):
                self.wait_for(1)
            time.sleep(0.2)  # several more polls, each refused
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0][0], "lost")
        self.assertIn("chrome", self.events[0][1])
        self.wait_for(2)
        self.assertEqual(self.events[1], ("restored", "chrome (this app only)"))
        self.assertEqual(cap.reopens, 1)

    def test_new_instance_that_cannot_be_opened_is_retried(self):
        cap = self.capture()
        cap.start()

        def refuse(_pid):
            raise OSError(-2147024891, "Access is denied")

        with mock.patch.object(appaudio, "_Stream", refuse):
            FakeTarget.alive_pids = {5151}
            self.found[0] = 5151
            self.wait_for(1)
            time.sleep(0.2)
        self.assertEqual(self.events, [("lost", "chrome was closed")])
        self.wait_for(2)
        self.assertEqual(self.events[1], ("restored", "chrome (this app only)"))
        self.assertEqual(FakeStream.opened, [4242, 5151])

    def test_stop_from_inside_the_event_callback(self):
        cap = self.capture()
        stopped = threading.Event()

        def on_event(_kind, _message):
            cap.stop()
            stopped.set()

        cap.on_event = on_event
        cap.start()
        FakeTarget.alive_pids = set()
        self.found[0] = None
        self.assertTrue(stopped.wait(3.0))
        drain(cap, 1.0)  # what was already converted is still handed over
        self.assertIsNone(cap.read_all(timeout=0.01))


class NeverRaises(unittest.TestCase):
    def test_no_playback_device_is_an_empty_list(self):
        with mock.patch.object(appaudio, "_audio_sessions", side_effect=OSError("no endpoint")):
            self.assertEqual(appaudio.list_audio_apps(), [])

    def test_no_process_list_is_an_empty_list(self):
        with mock.patch.object(appaudio, "_process_table", side_effect=OSError("denied")):
            self.assertEqual(appaudio.list_audio_apps(), [])
            self.assertIsNone(appaudio.find_app("chrome.exe"))

    def test_find_app_works_without_a_playback_device(self):
        with (
            mock.patch.object(appaudio, "_audio_sessions", side_effect=OSError("no endpoint")),
            mock.patch.object(appaudio, "_process_table", return_value=TABLE),
        ):
            self.assertEqual(appaudio.find_app("chrome.exe"), 200)
            self.assertIsNone(appaudio.find_app("firefox.exe"))

    def test_own_process_is_not_offered(self):
        me = os.getpid()
        table = {me: Proc(1, "python.exe"), 400: Proc(1, "game.exe")}
        with (
            mock.patch.object(appaudio, "_audio_sessions", return_value=[(me, True), (400, False)]),
            mock.patch.object(appaudio, "_process_table", return_value=table),
        ):
            self.assertEqual(appaudio.list_audio_apps(), [AudioApp("game.exe", "game", 400, False)])

    def test_real_calls_do_not_raise_on_this_machine(self):
        self.assertIsInstance(appaudio.supported(), bool)
        self.assertIsInstance(appaudio.list_audio_apps(), list)
        self.assertIsNone(appaudio.find_app("no-such-program-7f3a.exe"))
        table = appaudio._process_table()
        self.assertIn(os.getpid(), table)
        self.assertIn(root_pid(os.getpid(), table), table)


if __name__ == "__main__":
    unittest.main()
