"""Performance counters: is the app actually keeping up with the audio?

The pipeline writes into one Metrics object from its three threads and the GUI
reads a snapshot of it about once a second. Nothing here is locked. Every
write is a single append or a single assignment, each atomic under the GIL,
and a snapshot that is one sample stale is still a correct picture of a
system that changes over hundreds of milliseconds.

What is measured, and against what clock:

  capture   age of the newest audio block at the moment the pipeline picked
            it up, plus one device buffer. How stale the input is before any
            model has looked at it.
  pass      wall time of one recognition pass over the rolling buffer.
  rtf       recognition seconds per second of audio taken in. Under 1.0 the
            recogniser is ahead of the speaker.
  source    how long after a word was spoken its Japanese became final.
  preview   how long after a sentence began the first English appeared.
  final     how long after the last word of a sentence its finished English
            appeared. This is the number a viewer experiences as "delay".
  backlog   captured audio waiting to be looked at.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from collections import deque

log = logging.getLogger(__name__)


class Rolling:
    """The last `size` samples of one quantity."""

    def __init__(self, size: int = 200):
        self._values: deque[float] = deque(maxlen=size)
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        self._values.append(float(value))
        self.count += 1
        self.total += float(value)

    @property
    def last(self) -> float | None:
        return self._values[-1] if self._values else None

    def mean(self) -> float | None:
        values = list(self._values)
        return sum(values) / len(values) if values else None

    def percentile(self, q: float) -> float | None:
        values = sorted(self._values)
        if not values:
            return None
        index = min(len(values) - 1, max(0, int(round(q / 100.0 * (len(values) - 1)))))
        return values[index]

    def max(self) -> float | None:
        values = list(self._values)
        return max(values) if values else None

    def lifetime_mean(self) -> float | None:
        return self.total / self.count if self.count else None


class Metrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.capture_latency = Rolling(100)
        self.pass_sec = Rolling(200)
        self.source_lag = Rolling(200)
        self.preview_latency = Rolling(100)
        self.final_latency = Rolling(100)
        self.mt_final_sec = Rolling(100)
        self.second_look_sec = Rolling(100)
        self.mt_preview_sec = Rolling(200)
        self.llm_first_token_sec = Rolling(100)
        self.diarize_sec = 0.0

        self.passes = 0
        self.infer_sec = 0.0
        self.audio_sec = 0.0  # audio taken into the recogniser
        self.speech_sec = 0.0  # ...of which was inside an utterance
        self.utterances = 0
        self.previews = 0
        self.commits = 0
        self.mt_failures = 0
        self.fallback_lines = 0  # finals produced by the fallback translator
        self.second_looks = 0  # finished lines read again
        self.second_look_changes = 0  # ...and corrected
        self.second_look_rejected = 0  # ...where the model's reply was thrown away
        self.second_look_skipped = 0  # lines let go unchecked to catch up

        self.backlog_sec = 0.0
        self.backlog_max_sec = 0.0
        self.backlog_drops = 0  # times the recogniser gave up on a buffer
        self.backlog_dropped_sec = 0.0
        self.capture_dropped_blocks = 0  # device callbacks lost
        self.capture_dropped_sec = 0.0  # queued audio discarded as too old
        self.device_reopens = 0
        self.recoveries = 0  # failed passes that were retried

        # How utterances ended. Each of these is a decision the pipeline made
        # in place of waiting out the endpoint silence and running one more
        # pass; together they are most of the difference between a caption
        # that is right and on time and one that is neither.
        self.eager_endpoints = 0  # closed early because the text read as finished
        self.passes_reused = 0  # closing passes not run: nothing new to hear
        self.deferred_closes = 0  # pauses not taken as endings: text unfinished
        self.revised_lines = 0  # lines re-read whole and changed by it
        self.tail_drops = 0  # words dropped as placed after the speech ended
        self.pause_splits = 0  # buffer cut at a pause after a line was taken
        self.stall_splits = 0  # buffer cut because the recogniser stopped reading

        self.speakers = 0
        self.engine = ""  # e.g. "Kotoba v2.0 / cuda float16"
        self.translator = ""
        self.device_name = ""

    # -- derived -----------------------------------------------------------

    @property
    def rtf(self) -> float:
        """Recognition seconds per second of speech recognised.

        Against speech rather than against everything captured: counting the
        silence between sentences in the denominator would make any model
        look fast on a quiet stream.
        """
        return self.infer_sec / self.speech_sec if self.speech_sec else 0.0

    def recent_rtf(self, interval_sec: float) -> float | None:
        """RTF implied by the recent passes, were they run every interval."""
        mean = self.pass_sec.mean()
        if mean is None or interval_sec <= 0:
            return None
        return mean / interval_sec

    def keeping_up(self) -> tuple[bool, str]:
        """(ok, one-line reason). The verdict the GUI shows as a badge."""
        if self.passes < 3:
            return True, "Waiting for speech"
        recent = self.pass_sec.mean() or 0.0
        if self.backlog_sec > 2.0:
            return False, f"Falling behind: {self.backlog_sec:.1f}s of audio waiting"
        if recent > 1.5:
            return False, f"Recognition is slow ({recent * 1000:.0f} ms per pass)"
        final = self.final_latency.percentile(50)
        if final is not None and final > 4.0:
            return False, f"Translations are arriving {final:.1f}s late"
        return True, "Keeping up with real time"

    def snapshot(self) -> dict:
        def ms(rolling: Rolling, kind: str = "mean") -> float | None:
            value = (
                rolling.mean()
                if kind == "mean"
                else rolling.percentile(90)
                if kind == "p90"
                else rolling.max()
                if kind == "max"
                else rolling.last
            )
            return None if value is None else round(value * 1000.0, 1)

        def sec(rolling: Rolling, q: float | None = None) -> float | None:
            value = rolling.mean() if q is None else rolling.percentile(q)
            return None if value is None else round(value, 3)

        ok, health = self.keeping_up()
        gpu = gpu_memory()
        return {
            "uptime_sec": round(time.perf_counter() - self.started, 1),
            "keeping_up": ok,
            "health": health,
            "engine": self.engine,
            "translator": self.translator,
            "device_name": self.device_name,
            "capture_latency_ms": ms(self.capture_latency),
            "capture_latency_max_ms": ms(self.capture_latency, "max"),
            "passes": self.passes,
            "mean_pass_ms": ms(self.pass_sec),
            "p90_pass_ms": ms(self.pass_sec, "p90"),
            "last_pass_ms": ms(self.pass_sec, "last"),
            "average_pass_ms": (
                round(1000.0 * self.infer_sec / self.passes, 1) if self.passes else None
            ),
            "rtf": round(self.rtf, 3),
            "audio_sec": round(self.audio_sec, 1),
            "speech_sec": round(self.speech_sec, 1),
            "source_lag_sec": sec(self.source_lag),
            "source_lag_p90_sec": sec(self.source_lag, 90),
            "preview_latency_sec": sec(self.preview_latency, 50),
            "final_latency_sec": sec(self.final_latency, 50),
            "final_latency_p90_sec": sec(self.final_latency, 90),
            "final_latency_mean_sec": sec(self.final_latency),
            "mt_final_ms": ms(self.mt_final_sec),
            "mt_final_p90_ms": ms(self.mt_final_sec, "p90"),
            "mt_preview_ms": ms(self.mt_preview_sec),
            "llm_first_token_ms": ms(self.llm_first_token_sec),
            "utterances": self.utterances,
            "previews": self.previews,
            "commits": self.commits,
            "mt_failures": self.mt_failures,
            "fallback_lines": self.fallback_lines,
            "second_looks": self.second_looks,
            "second_look_changes": self.second_look_changes,
            "second_look_rejected": self.second_look_rejected,
            "second_look_skipped": self.second_look_skipped,
            "second_look_ms": ms(self.second_look_sec),
            "backlog_sec": round(self.backlog_sec, 2),
            "backlog_max_sec": round(self.backlog_max_sec, 2),
            "drops": self.backlog_drops,
            "backlog_dropped_sec": round(self.backlog_dropped_sec, 1),
            "capture_dropped_blocks": self.capture_dropped_blocks,
            "capture_dropped_sec": round(self.capture_dropped_sec, 1),
            "device_reopens": self.device_reopens,
            "recoveries": self.recoveries,
            "eager_endpoints": self.eager_endpoints,
            "passes_reused": self.passes_reused,
            "deferred_closes": self.deferred_closes,
            "revised_lines": self.revised_lines,
            "tail_drops": self.tail_drops,
            "pause_splits": self.pause_splits,
            "stall_splits": self.stall_splits,
            "speakers": self.speakers,
            "gpu_mem_used_mib": gpu[0] if gpu else None,
            "gpu_mem_total_mib": gpu[1] if gpu else None,
            "gpu_util_pct": gpu[2] if gpu else None,
            "process_rss_mib": process_rss_mib(),
        }


# -- system probes ----------------------------------------------------------

_nvml = None
_nvml_handle = None
_nvml_failed = False


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def gpu_memory() -> tuple[int, int, int] | None:
    """(used MiB, total MiB, utilisation %) for GPU 0, or None without NVIDIA.

    Through NVML, the library nvidia-smi itself is a front end for, loaded
    straight from the driver. A call takes microseconds, where spawning
    nvidia-smi once a second would cost more CPU than the rest of this module.

    The figure is the whole card, not this process: on Windows the driver does
    not report per-process usage to NVML. So it includes the game or browser
    sharing the GPU - which is the number that decides whether a model fits.
    """
    global _nvml, _nvml_handle, _nvml_failed
    if _nvml_failed or sys.platform != "win32":
        return None
    try:
        if _nvml is None:
            lib = ctypes.WinDLL("nvml.dll")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit failed")
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                raise OSError("no GPU 0")
            _nvml, _nvml_handle = lib, handle
        memory = _NvmlMemory()
        if _nvml.nvmlDeviceGetMemoryInfo(_nvml_handle, ctypes.byref(memory)) != 0:
            return None
        util = _NvmlUtilization()
        percent = 0
        if _nvml.nvmlDeviceGetUtilizationRates(_nvml_handle, ctypes.byref(util)) == 0:
            percent = int(util.gpu)
        return int(memory.used // 2**20), int(memory.total // 2**20), percent
    except (OSError, AttributeError) as exc:
        _nvml_failed = True
        log.debug("GPU memory unavailable: %s", exc)
        return None


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_rss_mib() -> int | None:
    """This process's working set, in MiB."""
    if sys.platform != "win32":
        return None
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.WorkingSetSize // 2**20)
    except (OSError, AttributeError):
        return None
