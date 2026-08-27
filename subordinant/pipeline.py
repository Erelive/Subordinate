"""Wiring: capture thread -> energy gate -> streaming ASR -> caption callback.

Runs on one worker thread. Inference blocks that thread, which is fine: audio
keeps arriving in the capture queue meanwhile, and the queue depth is exactly the
signal used to detect that the GPU has fallen behind real time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .asr import Transcriber, Word
from .audio import LoopbackCapture
from .config import SAMPLE_RATE, Config
from .streaming import StreamingTranscriber, join_words
from .translate import TranslationUnavailable, Translator
from .vad import EnergyGate

log = logging.getLogger(__name__)

CaptionCallback = Callable[[str, str], None]
StatusCallback = Callable[[str], None]
CommitCallback = Callable[[str], None]
# (source text, english) for one finished utterance. Fires only with mt_enabled.
TranslationCallback = Callable[[str, str], None]

# Audio kept before the gate opens, so the attack of the first word isn't lost.
PREROLL_SEC = 0.35

# What Whisper puts at the end of a sentence. The Japanese forms matter most
# here; the ASCII ones cover source languages that use them, and "..." is
# deliberately absent - it marks hesitation mid-sentence far more often than an
# ending, and cutting there would hand MT a fragment.
_SENTENCE_END = ("。", "！", "？", ".", "!", "?")


def _sentence_cut(words: list[Word]) -> int:
    """How many leading words form whole sentences. 0 if none do."""
    for i in range(len(words) - 1, -1, -1):
        text = words[i].text.strip()
        if text.endswith(_SENTENCE_END) and not text.endswith("..."):
            return i + 1
    return 0


@dataclass
class Stats:
    passes: int = 0
    total_infer_sec: float = 0.0
    total_audio_sec: float = 0.0
    drops: int = 0
    commits: int = 0
    total_lag_sec: float = 0.0
    max_lag_sec: float = 0.0
    mt_calls: int = 0
    total_mt_sec: float = 0.0

    @property
    def mean_mt_ms(self) -> float:
        return 1000.0 * self.total_mt_sec / self.mt_calls if self.mt_calls else 0.0

    @property
    def rtf(self) -> float:
        """Inference seconds per second of audio processed. Under 1.0 keeps up."""
        return self.total_infer_sec / self.total_audio_sec if self.total_audio_sec else 0.0

    @property
    def mean_pass_ms(self) -> float:
        return 1000.0 * self.total_infer_sec / self.passes if self.passes else 0.0

    @property
    def mean_lag_sec(self) -> float:
        """How far behind the audio a word is when it finally settles on screen."""
        return self.total_lag_sec / self.commits if self.commits else 0.0


class CaptionPipeline:
    def __init__(
        self,
        cfg: Config,
        on_caption: CaptionCallback,
        on_status: StatusCallback | None = None,
        transcriber: Transcriber | None = None,
        on_commit: CommitCallback | None = None,
        on_translation: TranslationCallback | None = None,
    ):
        self.cfg = cfg
        self.on_caption = on_caption
        self.on_status = on_status or (lambda _msg: None)
        # Fires once per word that becomes final, never re-sent. on_caption by
        # contrast re-sends a rolling window, so it cannot build a transcript.
        self.on_commit = on_commit or (lambda _text: None)
        # Fires once per finished utterance, after translation.
        self.on_translation = on_translation or (lambda _src, _en: None)
        self.transcriber = transcriber
        self.translator = None
        # Stream time up to which source text has been handed to MT.
        self._mt_until = 0.0

        # With MT running, Whisper's job is to produce source-language text -
        # asking it to translate as well would hand the NMT model English and
        # throw away the source column. Enforced here rather than in Config
        # because this is the one path both the GUI and tools.console take.
        if cfg.mt_enabled and cfg.task == "translate":
            log.info("mt_enabled: using task=transcribe, MT does the translating")
            cfg.task = "transcribe"

        self.stats = Stats()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="caption-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        try:
            if self.transcriber is None:
                self.on_status("loading model...")
                self.transcriber = Transcriber(self.cfg)
                self.transcriber.warmup()

            if self.cfg.mt_enabled and self.translator is None:
                self.on_status("loading translation model...")
                try:
                    self.translator = Translator(self.cfg)
                except TranslationUnavailable as exc:
                    # Captions in the source language are still useful, so this
                    # degrades rather than aborting the run.
                    log.warning("translation disabled: %s", exc)
                    self.on_status(f"translation unavailable: {exc}")

            capture = LoopbackCapture(frames_per_buffer=self.cfg.frames_per_buffer)
            self.on_status("opening audio device...")
            capture.start()
            self.on_status(f"listening: {capture.device_name}")
        except Exception as exc:  # noqa: BLE001 - surface any startup failure to the UI
            log.exception("pipeline failed to start")
            self.on_status(f"error: {exc}")
            return

        gate = EnergyGate(self.cfg.energy_gate_db)
        stream = StreamingTranscriber(self.cfg, self.transcriber)
        preroll: deque[np.ndarray] = deque()
        preroll_sec = 0.0
        in_speech = False
        pending_sec = 0.0

        try:
            while not self._stop.is_set():
                block = capture.read(timeout=0.2)
                if block is None:
                    break
                if self._paused.is_set():
                    continue

                state = gate.update(block)
                block_sec = block.size / SAMPLE_RATE

                if block.size:
                    if state.active and not in_speech:
                        # Speech just started: prepend the pre-roll so the first
                        # word isn't clipped by the gate's reaction time.
                        for held in preroll:
                            stream.insert_audio(held)
                            pending_sec += held.size / SAMPLE_RATE
                        preroll.clear()
                        preroll_sec = 0.0
                        in_speech = True

                    if in_speech:
                        stream.insert_audio(block)
                        pending_sec += block_sec
                    else:
                        preroll.append(block)
                        preroll_sec += block_sec
                        while preroll_sec > PREROLL_SEC and len(preroll) > 1:
                            preroll_sec -= preroll.popleft().size / SAMPLE_RATE

                # End of utterance: settle the tail, clear the buffer.
                if in_speech and state.trailing_silence_sec >= self.cfg.silence_flush_sec:
                    if stream.buffer_sec >= self.cfg.min_chunk_sec:
                        self._run_pass(stream, audio_sec=pending_sec)
                    self._announce(stream, stream.end_utterance())
                    self._emit(stream)
                    # Silence: settle whatever is left, whole sentence or not.
                    self._flush_translation(stream, force=True)
                    stream.prune_history(self.cfg.display_history_sec * 2)
                    in_speech = False
                    pending_sec = 0.0
                    continue

                if not in_speech or pending_sec < self.cfg.process_interval_sec:
                    continue

                # pending_sec is how much audio piled up while the last pass ran.
                # Well past the interval means inference is not keeping up.
                if pending_sec > self.cfg.max_backlog_sec:
                    stream.drop_backlog()
                    self.stats.drops += 1
                    self.on_status(f"dropped backlog ({pending_sec:.1f}s behind)")
                    pending_sec = 0.0
                    continue

                self._run_pass(stream, audio_sec=pending_sec)
                pending_sec = 0.0
                self._emit(stream)
                # Mid-speech: send any sentence that has completed, so continuous
                # talk still produces English instead of waiting for a gap.
                self._flush_translation(stream, force=False)
        finally:
            capture.stop()
            log.info(
                "pipeline stopped: %d passes, mean %.0f ms, RTF %.2f, "
                "lag mean %.2fs max %.2fs, %d drops",
                self.stats.passes,
                self.stats.mean_pass_ms,
                self.stats.rtf,
                self.stats.mean_lag_sec,
                self.stats.max_lag_sec,
                self.stats.drops,
            )

    def _run_pass(self, stream: StreamingTranscriber, audio_sec: float = 0.0) -> None:
        t0 = time.perf_counter()
        committed, _unstable = stream.process()
        elapsed = time.perf_counter() - t0
        self.stats.passes += 1
        self.stats.total_infer_sec += elapsed
        self.stats.total_audio_sec += audio_sec or self.cfg.process_interval_sec
        self._announce(stream, committed)

    def _flush_translation(self, stream: StreamingTranscriber, force: bool) -> None:
        """Translate whatever source text has settled but not yet been sent.

        Called after every pass, not only at end of utterance: waiting for
        silence means a conversation with no gaps in it never gets translated at
        all. A sentence ending is the natural unit, with mt_max_utterance_sec as
        the backstop for speech that never provides one.

        Progress is tracked by timestamp rather than by index into the committed
        list, because prune_history() rewrites that list and any saved index
        would then point at the wrong word.
        """
        if self.translator is None:
            return
        pending = [
            w for w in stream.hypothesis.committed if w.start >= self._mt_until
        ]
        if not pending:
            return

        if force:
            cut = len(pending)
        else:
            cut = _sentence_cut(pending)
            if cut == 0:
                # No sentence ending yet. Hold unless this has run long enough
                # that holding would mean showing nothing.
                if pending[-1].end - pending[0].start < self.cfg.mt_max_utterance_sec:
                    return
                cut = len(pending)

        chunk = pending[:cut]
        # Advance regardless of the outcome below, so a chunk that translates to
        # nothing cannot be retried forever.
        self._mt_until = chunk[-1].end

        source = join_words(chunk)
        if not source:
            return
        t0 = time.perf_counter()
        english = self.translator.translate(source)
        self.stats.mt_calls += 1
        self.stats.total_mt_sec += time.perf_counter() - t0
        if english:
            self.on_translation(source, english)

    def _announce(self, stream: StreamingTranscriber, words: list[Word]) -> None:
        text = join_words(words)
        if not text:
            return
        # Distance between where the audio stream is now and the end of the word
        # just settled: the real end-to-end caption latency.
        lag = max(0.0, stream.stream_time - words[-1].end)
        self.stats.commits += 1
        self.stats.total_lag_sec += lag
        self.stats.max_lag_sec = max(self.stats.max_lag_sec, lag)
        self.on_commit(text)

    def _emit(self, stream: StreamingTranscriber) -> None:
        committed, unstable = stream.display_words(self.cfg.display_history_sec)
        self.on_caption(join_words(committed), join_words(unstable))
