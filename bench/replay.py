"""Play a WAV into the pipeline at real-time pace, in place of the loopback device.

The whole point of the pipeline benchmark is latency, and latency only means
something against a clock. So this does not hand the file over as fast as it
can be read: block N becomes readable N * 20 ms after the start and not
before, exactly as a capture device would deliver it. Inference that runs slow
therefore falls behind here in the same way it would live.

It presents the same surface as subordinant.audio.LoopbackCapture, so the
pipeline cannot tell the difference and none of its code is bypassed except
the WASAPI call and the resampler.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

SR = 16000
BLOCK = 320  # 20 ms, about what a 1024-frame buffer at 48 kHz delivers


class FileCapture:
    def __init__(
        self,
        audio: np.ndarray,
        tail_sec: float = 3.0,
        name: str = "file",
        realtime: bool = True,
    ):
        # realtime=False hands the file over as fast as the pipeline asks for
        # it, a tenth of a second per read. That models a recogniser that
        # costs nothing: every pass lands exactly on its interval. Recognition
        # and segmentation come out as they would live and the run takes a
        # third of the time, which is what iterating on accuracy needs.
        # Latency figures from such a run mean nothing and are not reported.
        self.realtime = realtime
        self._released = 0
        # Digital silence after the last sample, the way a player that has
        # finished keeps its stream open for a moment. It gives the pipeline
        # the trailing quiet it needs to close the final utterance.
        self._audio = np.concatenate(
            (audio.astype(np.float32), np.zeros(int(tail_sec * SR), dtype=np.float32))
        )
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device_name = f"replay: {name}"
        self.native_rate = SR
        self.channels = 1
        self.dropped_blocks = 0
        # perf_counter() at which audio time zero was released. Everything the
        # harness records is expressed relative to this.
        self.t0: float | None = None
        self.finished = threading.Event()

    # -- LoopbackCapture surface ------------------------------------------

    def start(self) -> None:
        if not self.realtime:
            self.t0 = time.perf_counter()
            return
        self._thread = threading.Thread(target=self._feed, name="replay", daemon=True)
        self._thread.start()

    def _pull(self, timeout: float | None):
        """Fast replay: the next tenth of a second, or silence once it is over."""
        if self._stop.is_set():
            return None
        total = self._audio.size // BLOCK
        if self._released >= total:
            self.finished.set()
            time.sleep(min(timeout or 0.02, 0.02))
            return []
        now = time.perf_counter()
        blocks = []
        for _ in range(5):
            if self._released >= total:
                break
            i = self._released
            blocks.append((now, self._audio[i * BLOCK : (i + 1) * BLOCK]))
            self._released += 1
        return blocks

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)

    def read(self, timeout: float | None = None):
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return np.empty(0, dtype=np.float32)
        if item is None:
            return None
        return item[1] if isinstance(item, tuple) else item

    def read_all(self, timeout: float | None = None):
        """Every block that is ready, as [(capture_time, samples), ...].

        Blocks up to `timeout` for the first one. Returns None once stopped.
        """
        if not self.realtime:
            return self._pull(timeout)
        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            return []
        if first is None:
            return None
        blocks = [first]
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)  # leave the sentinel for the next call
                break
            blocks.append(item)
        return blocks

    def pending_blocks(self) -> int:
        return self._queue.qsize()

    def pending_sec(self) -> float:
        return self._queue.qsize() * BLOCK / SR

    # -- feeder ------------------------------------------------------------

    def _feed(self) -> None:
        self.t0 = time.perf_counter()
        n_blocks = self._audio.size // BLOCK
        for i in range(n_blocks):
            if self._stop.is_set():
                return
            due = self.t0 + (i + 1) * BLOCK / SR
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            block = self._audio[i * BLOCK : (i + 1) * BLOCK]
            self._queue.put((time.perf_counter(), block))
        self.finished.set()

    def audio_clock(self) -> float:
        """Seconds of the file that have been released so far."""
        if not self.realtime:
            return self._released * BLOCK / SR
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0
