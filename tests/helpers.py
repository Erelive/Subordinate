"""Scripted stand-ins for the recogniser, translators and capture device.

The trick that makes an end-to-end test of the engine loop possible without a
model: the fake audio *is* the transcript. Each "word" is a stretch of constant
amplitude, and the amplitude says which word it is. The fake recogniser reads
the words back out of whatever buffer it is handed, with timestamps relative
to that buffer - exactly the contract Whisper has - so trimming, pre-roll,
agreement and segmentation are all exercised for real.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from subordinant.asr import Word
from subordinant.config import SAMPLE_RATE, Config
from subordinant.pipeline import PipelineEvents

LEVEL_BASE = 0.05
LEVEL_STEP = 0.004


def level_of(index: int) -> float:
    return LEVEL_BASE + LEVEL_STEP * index


class Script:
    """A scripted piece of audio: words with gaps between them."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self._parts: list[np.ndarray] = []

    def silence(self, seconds: float) -> "Script":
        self._parts.append(np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32))
        return self

    def say(self, *words: str, each: float = 0.3) -> "Script":
        """Words spoken back to back, `each` seconds apiece."""
        for word in words:
            index = len(self.words)
            self.words.append(word)
            n = int(each * SAMPLE_RATE)
            # Alternating sign so it has energy without a DC offset; the
            # magnitude is the word's identity.
            signal = np.full(n, level_of(index), dtype=np.float32)
            signal[1::2] *= -1
            self._parts.append(signal)
        return self

    def audio(self) -> np.ndarray:
        return np.concatenate(self._parts) if self._parts else np.zeros(0, np.float32)


class FakeTranscriber:
    """Reads Script words back out of a buffer."""

    device = "cpu"
    compute_type = "fake"
    model_id = "fake"

    def __init__(
        self,
        script: Script,
        delay: float = 0.0,
        cfg: Config | None = None,
        stops_at_pause: bool = False,
        reads_at_most: float | None = None,
    ):
        self.script = script
        self.delay = delay
        self.cfg = cfg
        # Behave like a recogniser trained on single sentences: read as far
        # as the first gap in the speech and return as if that were all.
        self.stops_at_pause = stops_at_pause
        # ...or one that gives up after this many seconds of any buffer, the
        # way the real ones sometimes stop partway through a long sentence.
        self.reads_at_most = reads_at_most
        self.notes: list[str] = []
        self.calls = 0
        self.fail_next = 0
        self.buffer_secs: list[float] = []

    def warmup(self) -> float:
        return 0.0

    def close(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        self.calls += 1
        self.buffer_secs.append(audio.size / SAMPLE_RATE)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("CUDA failed with error out of memory")
        magnitude = np.abs(audio)
        ids = np.rint((magnitude - LEVEL_BASE) / LEVEL_STEP).astype(int)
        ids[magnitude < LEVEL_BASE / 2] = -1
        words: list[Word] = []
        start = 0
        n = ids.size
        while start < n:
            current = ids[start]
            end = start
            while end < n and ids[end] == current:
                end += 1
            # A word cut off by the start of the buffer is still heard, as
            # Whisper would hear a clipped word; a sliver is not.
            if 0 <= current < len(self.script.words) and end - start > SAMPLE_RATE // 20:
                words.append(
                    Word(
                        start / SAMPLE_RATE,
                        end / SAMPLE_RATE,
                        self.script.words[current],
                        0.9,
                    )
                )
            start = end
        if self.reads_at_most is not None:
            words = [w for w in words if w.end <= self.reads_at_most]
        if self.stops_at_pause:
            for i in range(len(words) - 1):
                if words[i + 1].start - words[i].end >= 0.15:
                    return words[: i + 1]
        return words


class FakeTranslator:
    """Upper-cases its input, so every final is checkable against its source."""

    uses_context = False

    def __init__(self, name: str = "nmt", delay: float = 0.0):
        self.name = name
        self.delay = delay
        self.last_error: str | None = None
        self.calls: list[str] = []
        self.fail = False
        self.resets = 0
        self.cfg = None

    def warmup(self) -> None:
        pass

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass

    def translate(self, text: str, on_partial=None, beam_size=None) -> str:
        self.calls.append(text)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            self.last_error = "Lost contact with Ollama (ConnectionError)."
            return ""
        self.last_error = None
        english = f"<{text}>"
        if on_partial is not None:
            on_partial(english[: len(english) // 2])
        return english


class FakeModels:
    """A ModelCache that hands out the fakes above."""

    def __init__(self, transcriber, nmt=None, llm=None, llm_error: str | None = None):
        self._transcriber = transcriber
        self._nmt = nmt
        self._llm = llm
        self._llm_error = llm_error

    def has_transcriber(self, cfg) -> bool:
        return True

    def transcriber(self, cfg):
        self._transcriber.cfg = cfg
        return self._transcriber, False

    def has_nmt(self, cfg) -> bool:
        return True

    def nmt(self, cfg):
        from subordinant.translate import TranslationUnavailable

        if self._nmt is None:
            raise TranslationUnavailable("no NMT model in this test")
        return self._nmt, False

    def llm(self, cfg):
        from subordinant.translate import TranslationUnavailable

        if self._llm_error or self._llm is None:
            raise TranslationUnavailable(self._llm_error or "no LLM in this test")
        return self._llm

    def vad(self, cfg):
        from subordinant.vad import EnergyVAD

        return EnergyVAD(cfg.energy_gate_db), None

    def release(self, include_llm: bool = True) -> None:
        pass


class ScriptSource:
    """Feeds audio to the pipeline in 20 ms blocks.

    realtime=True paces it against the clock, as a device would. False hands
    it over as fast as it is asked for, which is enough for anything that
    depends on the audio rather than on the time.
    """

    BLOCK = 320

    def __init__(self, audio: np.ndarray, realtime: bool = False, hold_open: float = 2.0):
        self._audio = audio
        self._pos = 0
        self.realtime = realtime
        self.device_name = "script"
        self.dropped_blocks = 0
        self.dropped_sec = 0.0
        self.buffer_latency_sec = 0.0
        self._t0: float | None = None
        self._ended_at: float | None = None
        self._hold_open = hold_open
        self._closed = threading.Event()
        self.finished = threading.Event()

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._closed.set()

    def read_all(self, timeout: float | None = None):
        if self._closed.is_set():
            return None
        now = time.perf_counter()
        if self._pos >= self._audio.size:
            self.finished.set()
            if self._ended_at is None:
                self._ended_at = now
            # Stay open and silent for a while, as a stopped player does,
            # then end the stream.
            if now - self._ended_at > self._hold_open:
                return None
            time.sleep(min(timeout or 0.02, 0.02))
            return []
        if self.realtime:
            available = int((now - self._t0) * SAMPLE_RATE)
            if available - self._pos < self.BLOCK:
                time.sleep(min(timeout or 0.02, 0.02))
                return []
            end = min(self._audio.size, available)
        else:
            end = min(self._audio.size, self._pos + self.BLOCK * 5)
        blocks = []
        while self._pos + self.BLOCK <= end:
            chunk = self._audio[self._pos : self._pos + self.BLOCK]
            self._pos += self.BLOCK
            stamp = self._t0 + self._pos / SAMPLE_RATE if self.realtime else now
            blocks.append((stamp, chunk))
        if self._audio.size - self._pos < self.BLOCK:
            self._pos = self._audio.size
        return blocks


@dataclass
class Recorded:
    utterances: list
    finals: list
    drafts: list
    previews: list
    commits: list
    states: list
    notices: list
    captions: list
    metrics: list


class RecordingEvents(PipelineEvents):
    def __init__(self) -> None:
        self.r = Recorded([], [], [], [], [], [], [], [], [])
        self._lock = threading.Lock()

    def on_state(self, state, detail):
        self.r.states.append((state, detail))

    def on_notice(self, level, message):
        self.r.notices.append((level, message))

    def on_caption(self, committed, unstable):
        self.r.captions.append((committed, unstable))

    def on_commit(self, text, speaker, start, end):
        self.r.commits.append(text)

    def on_preview(self, english, source):
        self.r.previews.append((english, source))

    def on_utterance(self, utterance):
        with self._lock:
            self.r.utterances.append(utterance)
            (self.r.finals if utterance.final else self.r.drafts).append(utterance)

    def on_metrics(self, snapshot):
        self.r.metrics.append(snapshot)


def test_config(**overrides) -> Config:
    """A Config for engine tests: energy VAD, no diarization, quick cadence."""
    cfg = Config()
    cfg.vad_mode = "energy"
    cfg.silence_flush_sec = 0.4
    cfg.endpoint_silence_sec = 0.4
    cfg.diarize_enabled = False
    cfg.mt_backend = "ct2"
    cfg.mt_hybrid_preview = True
    cfg.process_interval_sec = 0.3
    cfg.min_chunk_sec = 0.5
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def run_pipeline(cfg, script, models, realtime=False, hold_open=1.5, timeout=30.0):
    """Play a script through a real CaptionPipeline. Returns (pipeline, Recorded)."""
    from subordinant.pipeline import CaptionPipeline

    events = RecordingEvents()
    source = ScriptSource(script.audio(), realtime=realtime, hold_open=hold_open)
    pipeline = CaptionPipeline(cfg, events=events, source=source, models=models)
    pipeline.start()
    deadline = time.monotonic() + timeout
    while pipeline.running and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, events.r
