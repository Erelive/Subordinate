"""WASAPI loopback capture.

Loopback records what Windows is *playing* rather than what a microphone hears,
so it captures any app's output with no virtual cable and no user setup. The
default render device is resolved to its matching loopback device, opened at its
native rate, then downmixed to mono and resampled to 16 kHz for the model.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from .config import SAMPLE_RATE

log = logging.getLogger(__name__)


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
        raise AudioDeviceError("Windows reports no default output device")

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


class LoopbackCapture:
    """Captures system audio into a queue of mono 16 kHz float32 blocks.

    The PyAudio callback runs on a realtime-ish thread, so it only copies bytes
    into a queue. Downmix and resample happen on the consumer thread, where a
    stall costs latency instead of dropped frames.
    """

    def __init__(self, frames_per_buffer: int = 1024, device_index: int | None = None):
        self.frames_per_buffer = frames_per_buffer
        self.device_index = device_index

        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._raw: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
        self._out: queue.Queue[np.ndarray | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        self.device_name = ""
        self.native_rate = 0
        self.channels = 0
        self.dropped_blocks = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        if self.device_index is None:
            info = find_default_loopback(self._pa)
        else:
            info = self._pa.get_device_info_by_index(self.device_index)
            if not info.get("isLoopbackDevice"):
                raise AudioDeviceError(f"device {self.device_index} is not a loopback device")

        self.device_name = str(info["name"])
        self.native_rate = int(info["defaultSampleRate"])
        self.channels = int(info["maxInputChannels"])
        log.info(
            "capturing %r  %d Hz  %d ch", self.device_name, self.native_rate, self.channels
        )

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.native_rate,
            input=True,
            input_device_index=int(info["index"]),
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._callback,
        )

        self._worker = threading.Thread(target=self._convert_loop, name="audio-convert", daemon=True)
        self._worker.start()
        self._stream.start_stream()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except OSError as exc:  # stream may already be dead after a device change
                log.debug("error closing stream: %s", exc)
            self._stream = None
        # Unblock the converter, then the consumer.
        self._raw.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._out.put(None)

    # -- data --------------------------------------------------------------

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Block for the next mono 16 kHz block. Returns None once stopped."""
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return np.empty(0, dtype=np.float32)

    def pending_blocks(self) -> int:
        return self._out.qsize()

    # -- internals ---------------------------------------------------------

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ARG002
        if status:
            log.debug("stream status flags: %s", status)
        try:
            self._raw.put_nowait(in_data)
        except queue.Full:
            # Converter is starved of CPU. Dropping is better than growing an
            # unbounded backlog behind a live stream.
            self.dropped_blocks += 1
            if self.dropped_blocks % 50 == 1:
                log.warning("audio backlog full, dropped %d blocks", self.dropped_blocks)
        return (None, pyaudio.paContinue)

    def _convert_loop(self) -> None:
        # ResampleStream keeps filter state across chunks, so block boundaries
        # don't produce clicks the way independent per-chunk resampling does.
        resampler = soxr.ResampleStream(
            self.native_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
        )
        while not self._stop.is_set():
            block = self._raw.get()
            if block is None:
                break
            frames = np.frombuffer(block, dtype=np.float32)
            if self.channels > 1:
                # Trailing partial frame would break the reshape; drop it.
                usable = (frames.size // self.channels) * self.channels
                mono = frames[:usable].reshape(-1, self.channels).mean(axis=1)
            else:
                mono = frames
            if mono.size == 0:
                continue
            out = resampler.resample_chunk(np.ascontiguousarray(mono, dtype=np.float32))
            if out.size:
                self._out.put(out.astype(np.float32, copy=False))
