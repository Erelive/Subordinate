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
from .diarize import UNKNOWN, DiarizationUnavailable, SpeakerTracker
from .streaming import StreamingTranscriber, join_words
from .translate import TranslationUnavailable, Translator
from .vad import EnergyGate

log = logging.getLogger(__name__)

CaptionCallback = Callable[[str, str], None]
StatusCallback = Callable[[str], None]
# (text, speaker id). The speaker is diarize.UNKNOWN when labelling is off or
# the span could not be attributed; it is an int rather than an optional so it
# can cross a Qt signal unchanged.
CommitCallback = Callable[[str, int], None]
# (source text, english, speaker id) for one finished utterance. Fires only with
# mt_enabled.
TranslationCallback = Callable[[str, str, int], None]
# English for the tail that has not settled yet. Rewrites itself freely, so it
# belongs beside the unconfirmed source text, never in the transcript.
LiveTranslationCallback = Callable[[str], None]

# Audio kept before the gate opens, so the attack of the first word isn't lost.
PREROLL_SEC = 0.35

# What Whisper puts at the end of a sentence. The Japanese forms matter most
# here; the ASCII ones cover source languages that use them, and "..." is
# deliberately absent - it marks hesitation mid-sentence far more often than an
# ending, and cutting there would hand MT a fragment.
_SENTENCE_END = ("。", "！", "？", ".", "!", "?")


# Clause marks. Not a sentence ending, but a far better place to cut than the
# middle of a phrase when the backstop has to fire anyway.
_CLAUSE_END = ("、", "，", ",")


def _sentence_cut(words: list[Word]) -> int:
    """How many leading words form whole sentences. 0 if none do."""
    for i in range(len(words) - 1, -1, -1):
        text = words[i].text.strip()
        if text.endswith(_SENTENCE_END) and not text.endswith("..."):
            return i + 1
    return 0


def _clause_cut(words: list[Word]) -> int:
    """How many leading words end on a clause mark. 0 if none do.

    Used only when the backstop fires: translating a whole clause beats
    translating a phrase severed mid-thought, which is what produced run-ons
    like "はいよろしくお願いしますノエルのブームでしゃべります...".
    """
    for i in range(len(words) - 1, -1, -1):
        if words[i].text.strip().endswith(_CLAUSE_END):
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
    live_mt_calls: int = 0
    total_live_mt_sec: float = 0.0
    total_diarize_sec: float = 0.0

    @property
    def mean_mt_ms(self) -> float:
        return 1000.0 * self.total_mt_sec / self.mt_calls if self.mt_calls else 0.0

    @property
    def mean_live_mt_ms(self) -> float:
        return (
            1000.0 * self.total_live_mt_sec / self.live_mt_calls
            if self.live_mt_calls
            else 0.0
        )

    @property
    def diarize_rtf(self) -> float:
        """Speaker labelling cost as a share of the real-time budget."""
        return (
            self.total_diarize_sec / self.total_audio_sec if self.total_audio_sec else 0.0
        )

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
        on_live_translation: LiveTranslationCallback | None = None,
    ):
        self.cfg = cfg
        self.on_caption = on_caption
        self.on_status = on_status or (lambda _msg: None)
        # Fires once per word that becomes final, never re-sent. on_caption by
        # contrast re-sends a rolling window, so it cannot build a transcript.
        self.on_commit = on_commit or (lambda _text, _spk: None)
        # Fires once per finished utterance, after translation.
        self.on_translation = on_translation or (lambda _src, _en, _spk: None)
        self.on_live_translation = on_live_translation or (lambda _en: None)
        self.transcriber = transcriber
        self.translator = None
        self.tracker: SpeakerTracker | None = None
        # Stream time up to which source text has been handed to MT.
        self._mt_until = 0.0
        # Source text the live preview was last built from, so an unchanged tail
        # costs nothing instead of being re-translated every pass.
        self._live_source = ""

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

            if self.cfg.diarize_enabled and self.tracker is None:
                self.on_status("loading speaker model...")
                try:
                    self.tracker = SpeakerTracker(self.cfg)
                except DiarizationUnavailable as exc:
                    # Unlabelled captions are still captions, so this degrades
                    # rather than aborting the run.
                    log.warning("speaker labelling disabled: %s", exc)
                    self.on_status(f"speaker labelling unavailable: {exc}")

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
                            self._insert(stream, held)
                            pending_sec += held.size / SAMPLE_RATE
                        preroll.clear()
                        preroll_sec = 0.0
                        in_speech = True

                    if in_speech:
                        self._insert(stream, block)
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
                    # The settled translation has just reached the transcript,
                    # so a preview of the same words would only duplicate it.
                    self._clear_live()
                    stream.prune_history(
                        self.cfg.display_history_sec * 2, not_before=self._mt_until
                    )
                    if self.tracker is not None:
                        self.tracker.prune(
                            stream.stream_time - self.cfg.display_history_sec * 2
                        )
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
                # Then preview whatever is still in flight, so the English keeps
                # moving between those settled flushes.
                self._flush_live(stream)
        finally:
            capture.stop()
            log.info(
                "pipeline stopped: %d passes, mean %.0f ms, RTF %.2f, "
                "lag mean %.2fs max %.2fs, %d drops, "
                "diarize RTF %.3f over %d speakers, "
                "MT %d settled + %d preview (%.0f ms mean)",
                self.stats.passes,
                self.stats.mean_pass_ms,
                self.stats.rtf,
                self.stats.mean_lag_sec,
                self.stats.max_lag_sec,
                self.stats.drops,
                self.stats.diarize_rtf,
                self.tracker.speaker_count if self.tracker else 0,
                self.stats.mt_calls,
                self.stats.live_mt_calls,
                self.stats.mean_live_mt_ms,
            )

    def _insert(self, stream: StreamingTranscriber, block: np.ndarray) -> None:
        """Hand one block to the transcriber, and to the speaker tracker.

        These have to stay together. The tracker's clock is the total audio it
        has been given, and it is queried with word timestamps on the
        transcriber's clock, so any block that reaches one and not the other
        offsets every later attribution by that block's duration.
        """
        stream.insert_audio(block)
        if self.tracker is not None:
            self.tracker.feed(block)

    def _run_pass(self, stream: StreamingTranscriber, audio_sec: float = 0.0) -> None:
        t0 = time.perf_counter()
        committed, _unstable = stream.process()
        elapsed = time.perf_counter() - t0
        self.stats.passes += 1
        self.stats.total_infer_sec += elapsed
        self.stats.total_audio_sec += audio_sec or self.cfg.process_interval_sec
        if self.tracker is not None:
            # Before _announce, so the words this pass settled can be attributed.
            t0 = time.perf_counter()
            self.tracker.process()
            self.stats.total_diarize_sec += time.perf_counter() - t0
        self._announce(stream, committed)

    def _speaker_cut(self, words: list[Word]) -> int:
        """How many leading words precede the first change of speaker. 0 if none.

        Windows the tracker could not attribute are skipped rather than treated
        as a change: a brief quiet patch mid-sentence would otherwise split an
        utterance in two and hand MT both halves separately.
        """
        if self.tracker is None:
            return 0
        current = UNKNOWN
        for i, word in enumerate(words):
            speaker = self.tracker.speaker_for(word.start, word.end)
            if speaker == UNKNOWN:
                continue
            if current == UNKNOWN:
                current = speaker
            elif speaker != current:
                return i
        return 0

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
            # A change of voice ends an utterance whatever its length, so it is
            # decided before the hold below rather than after it. Checking it
            # afterwards means a short turn is held waiting for a sentence
            # ending that belongs to the *next* speaker - which is exactly the
            # merging this was meant to stop.
            change = self._speaker_cut(pending)
            # A change one or two words in is almost always the previous turn's
            # tail landing inside a boundary window, not a real turn. Cutting
            # there hands MT a fragment, which usually translates to nothing at
            # all - and because the cut still advances _mt_until, that text is
            # consumed and never appears. Ignore changes too short to be real.
            if change and pending[change - 1].end - pending[0].start < self.cfg.mt_min_chunk_sec:
                change = 0
            cut = _sentence_cut(pending)
            if change and (cut == 0 or change < cut):
                cut = change
            elif cut == 0:
                # No sentence ending and nobody new. Hold, unless this has run
                # long enough that holding would mean showing nothing.
                if pending[-1].end - pending[0].start < self.cfg.mt_max_utterance_sec:
                    return
                # Overdue: cut on the last clause mark if there is one, so the
                # translator gets a whole clause rather than a phrase severed
                # mid-thought.
                cut = _clause_cut(pending) or len(pending)

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
        # Emitted even when the translation came back empty. The text has been
        # consumed either way - _mt_until has moved past it - so dropping it
        # here loses the utterance outright, and the source line is exactly what
        # this window keeps history for. The transcript shows source-only.
        self.on_translation(source, english, self._speaker_of(chunk))

    def _flush_live(self, stream: StreamingTranscriber) -> None:
        """Translate the tail that has not settled, for the preview line.

        Deliberately includes the unstable words as well as the settled ones:
        the point is to be current, and this text is already displayed as a
        guess. It will rewrite itself, sometimes reversing meaning, because a
        partial Japanese sentence does not determine an English prefix - see the
        note on mt_live_enabled in config.py.
        """
        if self.translator is None or not self.cfg.mt_live_enabled:
            return

        tail = [w for w in stream.hypothesis.committed if w.start >= self._mt_until]
        tail += stream.hypothesis.unstable
        source = join_words(tail)

        if len(source) < self.cfg.mt_live_min_chars:
            # Nothing worth previewing. Clear it rather than leaving the last
            # preview stranded under a new utterance.
            if self._live_source:
                self._live_source = ""
                self.on_live_translation("")
            return
        if source == self._live_source:
            return  # tail unchanged; the previous preview still stands

        self._live_source = source
        t0 = time.perf_counter()
        english = self.translator.translate(source)
        self.stats.live_mt_calls += 1
        self.stats.total_live_mt_sec += time.perf_counter() - t0
        self.on_live_translation(english)

    def _clear_live(self) -> None:
        if self._live_source:
            self._live_source = ""
            self.on_live_translation("")

    def _speaker_of(self, words: list[Word]) -> int:
        """Who said this run of words, over its whole span."""
        if self.tracker is None or not words:
            return UNKNOWN
        return self.tracker.speaker_for(words[0].start, words[-1].end)

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
        self.on_commit(text, self._speaker_of(words))

    def _emit(self, stream: StreamingTranscriber) -> None:
        committed, unstable = stream.display_words(self.cfg.display_history_sec)
        self.on_caption(join_words(committed), join_words(unstable))
