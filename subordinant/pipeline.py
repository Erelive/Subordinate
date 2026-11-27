"""Wiring: capture -> speech gate -> streaming recogniser -> segmenter -> translators.

Up to four threads, each doing the one thing that would otherwise block another:

  engine    reads captured audio, runs the voice detector and the recogniser,
            decides where sentences end. Never translates.
  final     turns each finished sentence into its permanent English line.
            May take hundreds of milliseconds per line (an LLM); the engine
            does not wait for it.
  refine    reads each finished line again a few seconds later, with the
            lines that followed in view, and corrects it once (refine.py).
            Only exists when there is an instruction model to do it with,
            and always gives way to the thread above.
  preview   keeps a provisional English line for the sentence still being
            spoken. Latest request wins: a preview nobody will see is never
            computed.

The capture device has two more of its own (see audio.py), and the GUI its
own. Nothing here touches Qt; everything the outside world learns arrives
through PipelineEvents, called from whichever thread has the news.

What changed from the single-threaded loop this replaces, and why:

  The loop read exactly one capture block per iteration and ran a pass every
  0.35 s of audio. When a pass took longer than that, the unread audio piled
  up in the capture queue where nothing measured it; the "drop the backlog"
  check compared a counter that was reset every pass against 6 s and so could
  never fire. Measured on the benchmark's music-and-pauses sample, the result
  was captions 11-22 s behind with zero reported drops. The loop now takes
  everything that has arrived before each pass, so a slow pass makes the next
  one cover more audio rather than queue behind it.

  Utterances ended when the output went quiet. They now end when the speaker
  does (vad.py), and sentences inside running speech are cut at breaths the
  grammar agrees with (segment.py) instead of at a three-second timer.

  Translation ran on the recognition thread for the NMT route. It never does
  now, so a preview cannot delay a pass.

How an utterance ends, since most of both the accuracy and the delay is here:

  The line that is kept is read once, whole. Words settle one at a time while
  someone is talking, each on the strength of the half second that followed
  it, and that is what the live caption shows. When the speaker stops, the
  last pass has heard the entire utterance with silence after it; its reading
  replaces the piecemeal one before anything is translated (see
  HypothesisBuffer.settle_all in streaming.py).

  A pause is not an ending until the text agrees. If what was said stops on a
  particle, or is three characters long, the audio is kept and the recogniser
  goes on listening to the same buffer (a deferred close). Cutting there and
  starting fresh made the second half of 何さん、だっけ come back as さんだ
  and っけけ: half a word is not something a recogniser can hear.

  And an ending is not held up once the text is sure of it. A sentence that
  reads as finished is closed a quarter of a second into the pause rather
  than after the full endpoint (the eager endpoint), and the pass that
  already heard it is not run again just because silence was added.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .asr import _HALLUCINATIONS, ModelLoadError, Word, _is_oom, is_cached
from .audio import AudioDeviceError, LoopbackCapture
from .config import SAMPLE_RATE, Config
from .diarize import UNKNOWN, DiarizationUnavailable, SpeakerTracker
from .metrics import Metrics
from .models import ModelCache
from .profiles import short_model_name
from .segment import (
    _letter_count,
    has_content,
    is_fragment,
    join_words,
    looks_complete,
    looks_incomplete,
    nearest_boundary,
    overdue_cut,
    pause_boundaries,
    pause_cut,
    sentence_cut,
)
from .streaming import StreamingTranscriber
from .translate import TranslationUnavailable
from .vad import AUDIO, CLOSE, PAUSE, SpeechGate

log = logging.getLogger(__name__)

# Pipeline states. The GUI shows exactly one of these at a time, with `detail`
# as the sentence under it.
IDLE = "idle"
LOADING = "loading"
LISTENING = "listening"  # running, nobody speaking
RECOGNIZING = "recognizing"  # speech in progress
TRANSLATING = "translating"  # speech over, a final line still in flight
PAUSED = "paused"
ERROR = "error"

# After this many passes fail in a row the run is stopped rather than retried.
_MAX_PASS_FAILURES = 3

# A pass counts as having heard the end of the speech only if it ran with at
# least this much quiet after the last voiced frame. Less, and the recogniser
# was looking at a word that might still have been going.
_HEARD_QUIET_SEC = 0.09

# If this much speech has gone by since a pause and the recogniser's reading
# still stops at that pause, it is not going to read on by itself: it has
# taken what came before the pause for the whole utterance. See
# StreamingTranscriber.split for what is done about it and why.
_STALL_SEC = 1.0

# The same, when there is no pause to point at: the reading ends this far
# short of the audio. Longer than the above because a reading normally trails
# the newest sample by a few tenths of a second, and by the length of any
# quiet at the end of the buffer.
_STALL_NO_PAUSE_SEC = 1.5

# Quiet kept at the head of the buffer when it is cut at a pause, so the first
# word after the pause has its soft start and a moment of silence before it.
_LEAD_IN_SEC = 0.3

# A word the recogniser places this long after the voice detector heard the
# speaker stop was not spoken. Whisper fills silence at the end of a buffer
# with what usually follows speech in its training data - バイバイ,
# ありがとうございました - and at the end of an utterance there is no later
# pass to disagree with it. The margin covers the detector releasing a little
# before a trailing vowel has died away, and word timings being good to about
# a tenth of a second.
_TAIL_SLACK_SEC = 0.2

# Finished lines kept for the second look to read around, and how many may
# be waiting for theirs before the oldest is let go unchecked. Falling that
# far behind means the model cannot keep up, and a correction to a line that
# scrolled away a minute ago is not worth making the next one later still.
_SEEN_LINES = 48
_REFINE_BACKLOG = 10

# Two lines are one sentence only if the second began this soon after the
# first ended. Longer, and the speaker stopped and started again.
_JOIN_GAP_SEC = 2.5


@dataclass
class Utterance:
    """One translation unit: what was said, and what it means.

    Emitted more than once with the same id when a draft precedes the final
    line. `final` is False for the draft and for partial updates to it, True
    exactly once, and nothing with that id is sent after that.
    """

    id: int
    source: str
    english: str = ""
    speaker: int = UNKNOWN
    start: float = 0.0  # stream clock, seconds
    end: float = 0.0
    wall_time: float = 0.0  # time.time() at which the last word was spoken
    final: bool = True
    # Which translator wrote `english`: "llm", "nmt", "whisper", or "".
    engine: str = ""
    # Seconds from the last word being spoken to this line becoming final.
    latency: float | None = None
    # Set on the one later update a final line can receive: its second look
    # changed it. first_* are what it said until then.
    revised: bool = False
    first_source: str = ""
    first_english: str = ""


class PipelineEvents:
    """What the pipeline reports. Subclass and override what is needed.

    Every method is called from a worker thread. An implementation that
    touches a GUI must marshal to its own thread; the app does that with
    queued Qt signals.
    """

    def on_state(self, state: str, detail: str) -> None:
        """The pipeline moved to `state`. `detail` is a sentence for the user."""

    def on_notice(self, level: str, message: str) -> None:
        """Something the user should know. level is "info", "warning" or "error"."""

    def on_caption(self, committed: str, unstable: str) -> None:
        """The Japanese of the sentence in progress: settled part, then guess.

        The settled part usually only grows, but it is not final until the
        sentence is: a later pass that has heard more may correct it."""

    def on_commit(self, text: str, speaker: int, start: float, end: float) -> None:
        """Source text that just became final: it has been cut into a line
        and handed on. Never re-sent, never revised."""

    def on_preview(self, english: str, source: str) -> None:
        """Provisional English for the sentence in progress. "" clears it."""

    def on_utterance(self, utterance: Utterance) -> None:
        """A translation unit: draft (final=False) or finished (final=True)."""

    def on_revision(self, utterance: Utterance) -> None:
        """A finished line was corrected on its second look. At most once
        per line, some seconds after on_utterance sent it as final."""

    def on_metrics(self, snapshot: dict) -> None:
        """Performance counters, about once a second."""


@dataclass
class _Job:
    utterance: Utterance
    spoken_at: float  # perf_counter time the last word ended
    preview_gen: int
    words: list = field(default_factory=list)


@dataclass
class _Seen:
    """A finished line, as the second look will read it."""

    utterance: Utterance
    done_at: float  # perf_counter time it became final


class CaptionPipeline:
    def __init__(
        self,
        cfg: Config,
        events: PipelineEvents | None = None,
        source=None,
        models: ModelCache | None = None,
        wrap_translator: Callable | None = None,
    ):
        # A private copy. The GUI edits its own Config freely; nothing it does
        # reaches a running pass except through the setters below.
        self.cfg = cfg.copy()
        self.events = events or PipelineEvents()
        # Anything with LoopbackCapture's surface. The benchmark passes a file
        # player here; the app passes nothing and gets the real device.
        self._source = source
        self.models = models or ModelCache()
        self._owns_models = models is None
        self._wrap = wrap_translator or (lambda translator: translator)

        cfg = self.cfg
        # With a translator running, Whisper's job is to produce Japanese -
        # asking it to translate as well would hand the translator English
        # and throw the source text away.
        if cfg.mt_enabled and cfg.task == "translate":
            log.info("mt_enabled: using task=transcribe, the translator translates")
            cfg.task = "transcribe"
        self._direct = cfg.task == "translate"

        self.metrics = Metrics()
        self.stats = self.metrics  # the name the tools knew it by
        self.transcriber = None
        self.tracker: SpeakerTracker | None = None
        self._final_tr = None
        self._preview_tr = None
        self._nmt = None

        self.state = IDLE
        self._detail = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._live = cfg.mt_live_enabled

        # The recogniser's stream for the run in progress. It owns the words:
        # see _open.
        self._stream: StreamingTranscriber | None = None
        self._holding = False
        # Silence required before the utterance that has stopped is treated
        # as finished, or None when nothing is waiting. See _close_utterance.
        self._deferred_hold: float | None = None
        # gate.voiced_total as of the last successful pass over the current
        # buffer, and how much quiet that pass had at its end. None when no
        # pass has looked at this buffer.
        self._pass_voiced: int | None = None
        self._pass_quiet = 0.0
        # Content characters handed to the translator so far this run, and,
        # for each pause the recogniser has looked at from inside, how many
        # had been recognised by then - keyed by when the pause began. That
        # is where the pause falls in the text, known without trusting a
        # word timestamp. See segment.pause_boundaries.
        self._sent_letters = 0
        self._pause_marks: dict[float, int] = {}
        # Set when the buffer was cut inside a pause that is still going on:
        # gate.voiced_total as it was then. Until it moves, the buffer holds
        # nothing but that pause, and a recogniser shown only silence makes
        # something up.
        self._quiet_base: int | None = None
        # Start of the pause the last pass's reading stopped at, if speech
        # has gone on well past it. See _unstall.
        self._stalled_at: float | None = None
        self._next_id = 1
        self._gate: SpeechGate | None = None
        # perf_counter time corresponding to stream time 0, for the utterance
        # in progress. wall(t) = _clock + t.
        self._clock = time.perf_counter()
        self._caption = ("", "")

        self._final_q: queue.Queue[_Job | None] = queue.Queue()
        self._final_thread: threading.Thread | None = None
        self._final_busy = False
        self._llm_failures = 0
        self._llm_replaced = False
        self._fallback_announced = False

        # Stream slang: see glossary.py. Used here only to put right the
        # spellings a recogniser gives words it does not know.
        self._glossary = None
        if cfg.slang_glossary and not self._direct:
            from . import glossary

            self._glossary = glossary.default()

        # The second look. _seen is the tail of the transcript, oldest first;
        # _seen_pos is the first line in it not yet looked at again.
        self._refiner = None
        self._refine_thread: threading.Thread | None = None
        self._refine_cond = threading.Condition()
        self._seen: list[_Seen] = []
        self._seen_pos = 0

        self._preview_cond = threading.Condition()
        self._preview_req: tuple[int, str, float] | None = None
        self._preview_thread: threading.Thread | None = None
        self._preview_source = ""
        self._preview_shown = ""
        # Bumped at every cut. A preview computed for text that has since
        # been cut is for a sentence that is no longer in progress.
        self._epoch = 0
        # Bumped every time a preview reaches the screen.
        self._preview_gen = 0
        self._preview_fresh = True
        self._preview_at = 0.0
        # Normalised text of the last unit sent for translation, to recognise
        # a boundary echo of it; see _emit_unit.
        self._last_unit = ""
        self._pass_failures = 0
        self._last_metrics = 0.0
        self._context_reset_done = True

    @property
    def _open(self) -> list[Word]:
        """Settled words not yet handed to the translator, oldest first.

        A copy of what the stream holds. They can still change - a later
        pass may revise them - until _flush takes them, after which they are
        a line in the transcript and never change again.
        """
        stream = self._stream
        return stream.pending if stream is not None else []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="caption-engine", daemon=True
        )
        self._thread.start()

    def stop(self, wait: float = 5.0) -> None:
        self._stop.set()
        with self._preview_cond:
            self._preview_cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            if wait > 0:
                thread.join(timeout=wait)
            if thread.is_alive():
                # Still winding down, and still holding the models. Keep the
                # handle so `running` stays true until it has really gone:
                # the caller uses that to decide when the models may be
                # released or handed to the next run.
                return
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_live_preview(self, enabled: bool) -> None:
        """Turn the provisional English on or off without restarting."""
        self._live = enabled
        self.cfg.mt_live_enabled = enabled
        if not enabled:
            self._preview_source = ""
            self._emit_preview("", "")

    # -- reporting ---------------------------------------------------------

    def _emit(self, name: str, *args) -> None:
        """Call one PipelineEvents method. A listener that raises must not
        take the engine down with it."""
        try:
            getattr(self.events, name)(*args)
        except Exception:  # noqa: BLE001
            log.exception("%s handler failed", name)

    def _set_state(self, state: str, detail: str = "") -> None:
        if state == self.state and detail == self._detail:
            return
        self.state, self._detail = state, detail
        log.info("state: %s%s", state, f" - {detail}" if detail else "")
        self._emit("on_state", state, detail)

    def _notice(self, level: str, message: str) -> None:
        getattr(log, "warning" if level != "info" else "info")("%s", message)
        self._emit("on_notice", level, message)

    def _fail(self, message: str) -> None:
        self._set_state(ERROR, message)

    # -- startup -----------------------------------------------------------

    def _run(self) -> None:
        source = None
        failed = False
        try:
            self._load()
            if self._stop.is_set():
                return
            source = self._open_source()
            if source is None:
                return
            self._start_workers()
            self._loop(source)
        except (ModelLoadError, TranslationUnavailable) as exc:
            failed = True
            self._fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            failed = True
            log.exception("pipeline failed")
            self._fail(self._explain(exc))
        finally:
            if source is not None and self._source is None:
                try:
                    source.stop()
                except Exception:  # noqa: BLE001
                    log.debug("error stopping capture", exc_info=True)
            self._stop_workers()
            self._summary()
            if not failed:
                self._set_state(IDLE, "")

    @staticmethod
    def _explain(exc: BaseException) -> str:
        """An exception as a sentence someone can act on."""
        if _is_oom(exc):
            return (
                "The GPU ran out of memory. Close other GPU-heavy apps, turn on "
                "VRAM saver in Settings, or pick the Fast profile."
            )
        return f"Something went wrong: {type(exc).__name__}: {exc}"

    def _load(self) -> None:
        cfg = self.cfg
        if cfg.vad_mode != "silero":
            # Nothing upstream is rejecting music, so Whisper's own pass has to.
            cfg.vad_filter = True

        name = short_model_name(cfg.model)
        if self.models.has_transcriber(cfg):
            self._set_state(LOADING, f"Preparing {name}")
        elif is_cached(cfg.model):
            self._set_state(LOADING, f"Loading recognition model: {name}")
        else:
            self._set_state(
                LOADING,
                f"Downloading recognition model: {name} (first run only, about 1.5 GB)",
            )
        self.transcriber, fresh = self.models.transcriber(cfg)
        for note in self.transcriber.notes:
            self._notice("warning", note)
        if fresh:
            self.transcriber.warmup()
        if self._stop.is_set():
            return
        if self.transcriber.device == "cpu" and cfg.device != "cpu":
            # Adapt the cadence to what a CPU can do, rather than run passes
            # back to back and call it real time.
            cfg.process_interval_sec = max(cfg.process_interval_sec, 1.0)
            cfg.beam_size = 1
            cfg.max_buffer_sec = min(cfg.max_buffer_sec, 8.0)
        self.metrics.engine = (
            f"{short_model_name(self.transcriber.model_id)} on "
            f"{self.transcriber.device} ({self.transcriber.compute_type})"
        )

        self._load_translators()
        if self._stop.is_set():
            return

        vad, note = self.models.vad(cfg)
        if note:
            self._notice("warning", note)
            cfg.vad_filter = True
        self._vad = vad

        if cfg.diarize_enabled and self.tracker is None:
            self._set_state(LOADING, "Loading speaker model")
            try:
                self.tracker = SpeakerTracker(cfg)
            except DiarizationUnavailable as exc:
                # Unlabelled captions are still captions, so this degrades
                # rather than aborting the run.
                self._notice("warning", f"Speaker labels are off: {exc}")

    def _load_translators(self) -> None:
        cfg = self.cfg
        if self._direct or not cfg.mt_enabled:
            self.metrics.translator = (
                "Whisper (direct)" if self._direct else "none - Japanese only"
            )
            return

        want_llm = cfg.mt_backend == "llm"
        need_nmt = (not want_llm) or cfg.mt_hybrid_preview or cfg.mt_fallback_to_nmt

        final = None
        if want_llm:
            self._set_state(
                LOADING, f"Loading translation model: {short_model_name(cfg.llm_model)}"
            )
            try:
                final = self.models.llm(cfg)
            except TranslationUnavailable as exc:
                if not cfg.mt_fallback_to_nmt:
                    raise
                self._notice(
                    "warning", f"{exc} Using the built-in translator for now."
                )
        if self._stop.is_set():
            return

        nmt = None
        if need_nmt or final is None:
            if not self.models.has_nmt(cfg):
                self._set_state(LOADING, "Loading translation model: Sugoi v4")
            try:
                nmt, fresh = self.models.nmt(cfg)
                if fresh:
                    nmt.warmup()
            except TranslationUnavailable as exc:
                if final is None:
                    # Captions in Japanese are still useful, so this degrades
                    # rather than aborting the run.
                    self._notice(
                        "warning",
                        f"Translation is off - showing Japanese only. {exc}",
                    )
                else:
                    self._notice("warning", f"No draft translations: {exc}")

        self._nmt = self._wrap(nmt) if nmt is not None else None
        if final is not None:
            final.reset()
            self._final_tr = self._wrap(final)
            self._preview_tr = self._nmt if cfg.mt_hybrid_preview else None
            self.metrics.translator = short_model_name(cfg.llm_model)
            if self._preview_tr is not None:
                self.metrics.translator += " (drafts: Sugoi v4)"
        else:
            self._final_tr = self._nmt
            self._preview_tr = self._nmt
            self.metrics.translator = "Sugoi v4" if self._nmt else "none - Japanese only"
        if not self._stop.is_set():
            self._load_refiner()

    def _load_refiner(self) -> None:
        """Set up the second look (refine.py)."""
        cfg = self.cfg
        self._refiner = None
        if not cfg.refine_enabled or self._final_tr is None:
            return
        from .refine import Refiner

        try:
            refiner = Refiner(cfg)
        except TranslationUnavailable as exc:
            # Only possible with refine_fix_words, which needs a model.
            self._notice("warning", f"The second look is off: {exc}")
            return
        if refiner.uses_model:
            llm_running = getattr(self._final_tr, "name", "") == "llm"
            if not llm_running or refiner.model != cfg.llm_model:
                self._set_state(
                    LOADING,
                    f"Loading second-look model: {short_model_name(refiner.model)}",
                )
                refiner.warmup()
                self.models.note_llm(cfg.llm_endpoint, refiner.model)
        self._refiner = refiner

    def _open_source(self):
        """Open the capture device, waiting for one to exist if need be."""
        if self._source is not None:
            self._source.start()
            self.metrics.device_name = getattr(self._source, "device_name", "")
            return self._source

        announced = False
        self._set_state(LOADING, "Opening audio device")
        app = self.cfg.capture_app.strip()
        if app:
            from . import appaudio

            if not appaudio.supported():
                self._notice(
                    "warning",
                    "Listening to one app needs Windows 11, so this is "
                    "listening to everything the PC plays instead.",
                )
                app = ""
        if app:
            # Windows keeps only 30 ms of a single app's audio waiting to be
            # collected, and says nothing when it overflows. The thread that
            # collects it needs the interpreter for a moment every 10 ms; at
            # Python's default of handing it over every 5 ms, three busy
            # threads are enough to make it late. Measured in appaudio's own
            # tests: 0.8 s lost in 7 at the default, none at this.
            sys.setswitchinterval(0.001)
        while not self._stop.is_set():
            if app:
                capture = appaudio.ProcessCapture(
                    app,
                    frames_per_buffer=self.cfg.frames_per_buffer,
                    max_queue_sec=self.cfg.audio_queue_sec,
                    on_event=self._on_app_event,
                )
            else:
                capture = LoopbackCapture(
                    frames_per_buffer=self.cfg.frames_per_buffer,
                    max_queue_sec=self.cfg.audio_queue_sec,
                    on_event=self._on_device_event,
                )
            try:
                capture.start()
            except AudioDeviceError as exc:
                if not announced:
                    announced = True
                    text = str(exc)
                    self._set_state(
                        ERROR,
                        f"{text[:1].upper()}{text[1:]} Captions start when it does."
                        if app
                        else f"No audio device to listen to: {exc} Waiting for one.",
                    )
                self._stop.wait(2.0)
                continue
            self.metrics.device_name = capture.device_name
            return capture
        return None

    def _on_app_event(self, kind: str, message: str) -> None:
        """The one app being listened to closed, or came back."""
        self.metrics.device_reopens += 1
        if kind == "lost":
            self._notice("warning", f"{message}. Captions resume when it is running again.")
        else:
            self.metrics.device_name = message
            self._notice("info", f"Listening to {message} again.")

    def _on_device_event(self, kind: str, message: str) -> None:
        self.metrics.device_reopens += 1
        if kind == "lost":
            self._notice("warning", f"The audio device went away: {message}")
        elif kind == "restored":
            self.metrics.device_name = message
            self._notice("info", f"Audio is back: now listening to {message}")
        else:
            self.metrics.device_name = message
            self._notice("info", f"Output device changed: now listening to {message}")

    def _start_workers(self) -> None:
        if self._final_tr is not None:
            self._final_thread = threading.Thread(
                target=self._final_worker, name="caption-final", daemon=True
            )
            self._final_thread.start()
        if self._preview_tr is not None:
            self._preview_thread = threading.Thread(
                target=self._preview_worker, name="caption-preview", daemon=True
            )
            self._preview_thread.start()
        if self._refiner is not None:
            self._refine_thread = threading.Thread(
                target=self._refine_worker, name="caption-refine", daemon=True
            )
            self._refine_thread.start()

    def _stop_workers(self) -> None:
        """Sentinel the workers and give them a moment to notice.

        The joins are bounded because an LLM request in flight runs to its
        timeout and cannot be interrupted. Outliving this is harmless: the
        threads are daemons, and _stop is set, so they emit nothing.
        """
        self._stop.set()
        if self._final_thread is not None:
            self._final_q.put(None)
            self._final_thread.join(timeout=2.0)
            self._final_thread = None
        if self._preview_thread is not None:
            with self._preview_cond:
                self._preview_cond.notify_all()
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None
        if self._refine_thread is not None:
            with self._refine_cond:
                self._refine_cond.notify_all()
            self._refine_thread.join(timeout=2.0)
            self._refine_thread = None
        if self._refiner is not None:
            self._refiner.close()

    # -- engine loop -------------------------------------------------------

    def _loop(self, source) -> None:
        cfg = self.cfg
        silero = cfg.vad_mode == "silero" and type(self._vad).__name__ == "SileroVAD"
        gate = SpeechGate(
            self._vad,
            threshold=cfg.vad_threshold,
            endpoint_silence_sec=(
                cfg.endpoint_silence_sec if silero else cfg.silence_flush_sec
            ),
            soft_pause_sec=cfg.soft_pause_sec,
        )
        stream = StreamingTranscriber(cfg, self.transcriber)
        self._gate = gate
        self._stream = stream

        new_audio = 0.0  # seconds given to the recogniser since its last pass
        pause_hint = False
        last_audio = time.perf_counter()
        silence_fed = 0.0
        was_paused = False
        buffer_latency = getattr(source, "buffer_latency_sec", 0.0)
        self._set_state(LISTENING, self.metrics.device_name)

        while not self._stop.is_set():
            blocks = source.read_all(timeout=0.05)
            if blocks is None:
                break
            now = time.perf_counter()

            if self._paused.is_set():
                if not was_paused:
                    was_paused = True
                    self._enter_pause(stream, gate)
                    new_audio = 0.0
                continue
            if was_paused:
                was_paused = False
                last_audio, silence_fed = now, 0.0
                self._set_state(LISTENING, self.metrics.device_name)

            events = []
            if blocks:
                last_audio, silence_fed = now, 0.0
                newest = blocks[-1][0]
                self.metrics.capture_latency.add(now - newest + buffer_latency)
                backlog = now - blocks[0][0]
                self.metrics.backlog_sec = backlog
                self.metrics.backlog_max_sec = max(self.metrics.backlog_max_sec, backlog)
                audio = (
                    blocks[0][1]
                    if len(blocks) == 1
                    else np.concatenate([b for _t, b in blocks])
                )
                self.metrics.audio_sec += audio.size / SAMPLE_RATE
                events = gate.feed(audio)
                emitted = any(e.kind == AUDIO for e in events)
            else:
                emitted = False
                newest = now
                self.metrics.backlog_sec = 0.0
                # The device is delivering nothing at all, which is what a
                # paused player looks like to loopback. Count it as silence.
                idle = now - last_audio
                if idle - silence_fed >= 0.1:
                    events = gate.feed_silence(idle - silence_fed)
                    silence_fed = idle

            for event in events:
                if event.kind == AUDIO:
                    self._insert(stream, event.audio)
                    new_audio += event.audio.size / SAMPLE_RATE
                elif event.kind == PAUSE:
                    pause_hint = True
                elif event.kind == CLOSE:
                    self._close_utterance(stream, gate)
                    new_audio = 0.0
                    pause_hint = False
            if emitted:
                # The newest sample the gate has passed on was captured
                # `buffered` before the newest block arrived.
                self._clock = newest - gate.buffered_sec - stream.stream_time

            if gate.in_utterance:
                self._context_reset_done = False
                if new_audio > cfg.max_backlog_sec:
                    # One pass took longer than the whole backlog allowance.
                    # Catch up rather than transcribe audio nobody is still
                    # watching the picture for.
                    self.metrics.backlog_drops += 1
                    self.metrics.backlog_dropped_sec += stream.buffer_sec
                    stream.drop_backlog()
                    self._pass_voiced = None
                    self._deferred_hold = None
                    self._notice(
                        "warning",
                        f"Recognition fell {new_audio:.0f}s behind and skipped "
                        "ahead. Try the Fast profile.",
                    )
                    new_audio = 0.0
                    continue
                due = new_audio >= cfg.process_interval_sec or (
                    pause_hint and new_audio >= 0.1
                )
                if self._quiet_base is not None and gate.voiced_total != self._quiet_base:
                    self._quiet_base = None
                if (
                    due
                    and self._quiet_base is None
                    and stream.buffer_sec >= cfg.min_chunk_sec
                ):
                    ok = self._pass(stream, new_audio)
                    new_audio = 0.0
                    pause_hint = False
                    if ok:
                        self._emit_caption(stream)
                        self._flush(stream, gate, closed=False)
                        self._request_preview(stream)
                if self._eager_ready(stream, gate):
                    # The sentence reads as finished and the speaker has
                    # paused. Take it now; the rest of the endpoint would
                    # only be waiting to be told what the text already says.
                    gate.end_utterance()
                    self.metrics.eager_endpoints += 1
                    self._close_utterance(stream, gate)
                    new_audio = 0.0
                    pause_hint = False
            else:
                if (
                    self._deferred_hold is not None
                    and gate.silence_sec >= self._deferred_hold
                ):
                    # They did not go on. What there is, is the sentence.
                    self._finalize(stream, gate)
                    new_audio = 0.0
                if self._holding:
                    # Text held back because it ended on a dangling particle.
                    # Look again as the silence grows.
                    self._flush(stream, gate, closed=True)
                if (
                    not self._context_reset_done
                    and gate.silence_sec >= cfg.llm_context_reset_sec
                ):
                    self._context_reset_done = True
                    if self._final_tr is not None:
                        self._final_tr.reset()

            self._refresh_state(gate)
            self._tick(source)

    def _enter_pause(self, stream: StreamingTranscriber, gate: SpeechGate) -> None:
        """Drop everything in flight. Resuming starts on fresh audio."""
        stream.reset()
        gate.reset()
        self._pause_marks.clear()
        self._holding = False
        self._deferred_hold = None
        self._pass_voiced = None
        self._quiet_base = None
        self._epoch += 1
        self._preview_source = ""
        self._emit_preview("", "")
        self._caption = ("", "")
        self._emit("on_caption", "", "")
        self._set_state(PAUSED, "")

    def _refresh_state(self, gate: SpeechGate) -> None:
        if self.state == PAUSED:
            return
        if gate.in_utterance:
            state = RECOGNIZING
        elif self._final_busy or not self._final_q.empty():
            state = TRANSLATING
        else:
            state = LISTENING
        if state != self.state:
            self._set_state(state, self.metrics.device_name)

    def _tick(self, source) -> None:
        now = time.perf_counter()
        if now - self._last_metrics < 1.0:
            return
        self._last_metrics = now
        metrics = self.metrics
        metrics.capture_dropped_blocks = getattr(source, "dropped_blocks", 0)
        # lost_sec is the one-app capture's own count of audio Windows
        # discarded before it could be collected.
        metrics.capture_dropped_sec = getattr(source, "dropped_sec", 0.0) + getattr(
            source, "lost_sec", 0.0
        )
        if self.tracker is not None:
            metrics.speakers = self.tracker.speaker_count
        self._emit("on_metrics", metrics.snapshot())

    # -- recognition -------------------------------------------------------

    def _insert(self, stream: StreamingTranscriber, block: np.ndarray) -> None:
        """Hand one block to the transcriber, and to the speaker tracker.

        These have to stay together. The tracker's clock is the total audio it
        has been given, and it is queried with word timestamps on the
        transcriber's clock, so any block that reaches one and not the other
        offsets every later attribution by that block's duration.
        """
        stream.insert_audio(block)
        self.metrics.speech_sec += block.size / SAMPLE_RATE
        if self.tracker is not None:
            self.tracker.feed(block)

    def _pass(
        self, stream: StreamingTranscriber, audio_sec: float, force: bool = False
    ) -> bool:
        """Run the recogniser once. False if the pass failed and was absorbed."""
        t0 = time.perf_counter()
        gate = self._gate
        pause = gate.open_pause if gate is not None else None
        try:
            settled, _unstable = stream.process(
                force=force, quiet_from=pause.start if pause is not None else None
            )
        except Exception as exc:  # noqa: BLE001 - CUDA faults, OOM, anything
            self._pass_failures += 1
            log.exception("recognition pass failed")
            if self._pass_failures >= _MAX_PASS_FAILURES:
                raise
            # Usually memory pressure from another program. The buffer is the
            # only thing this pass can give up, so give it up and go again.
            self.metrics.recoveries += 1
            stream.discard_audio()
            self._pass_voiced = None
            self._deferred_hold = None
            self._notice(
                "warning",
                "A recognition pass failed"
                + (" (the GPU is out of memory)" if _is_oom(exc) else "")
                + ". Skipped a moment of audio and carried on.",
            )
            return False
        self._pass_failures = 0
        if gate is not None:
            self._pass_voiced = gate.voiced_total
            self._pass_quiet = gate.trailing_quiet_sec if gate.in_utterance else gate.silence_sec
        elapsed = time.perf_counter() - t0
        self.metrics.passes += 1
        self.metrics.pass_sec.add(elapsed)
        self.metrics.infer_sec += elapsed
        if self.tracker is not None:
            t0 = time.perf_counter()
            self.tracker.process()
            self.metrics.diarize_sec += time.perf_counter() - t0
        self._note_settled(stream, settled)
        if pause is not None:
            # The buffer ended inside this pause, so what was just read is
            # what was said before it.
            heard = stream.pending + stream.hypothesis.unstable
            self._pause_marks[round(pause.start, 3)] = self._sent_letters + sum(
                _letter_count(w.text) for w in heard
            )
        if gate is not None and not force:
            self._unstall(stream, gate)
        return True

    @staticmethod
    def _cut_point(pause, stream: StreamingTranscriber) -> float:
        """Where in a pause to cut the buffer: late, but not at its very end."""
        end = min(pause.end, stream.stream_time)
        return max(pause.start + 0.05, end - _LEAD_IN_SEC)

    def _unstall(self, stream: StreamingTranscriber, gate: SpeechGate) -> None:
        """Notice the recogniser has stopped reading, and move it on.

        Two ways it happens, both measured on the benchmark's samples.

        At a pause: the reading ends at a pause, the speaker went on a second
        ago, and nothing of what they have said since is in it. The model
        has taken what came before the pause for the whole utterance.

        In mid-sentence: the reading simply stops, a second and a half or
        more short of the audio, and stays there pass after pass while the
        speaker carries on. Left alone this lost forty characters of one
        sentence - in some real-time runs and not others, since whether a
        pass lands on a buffer length the model stops early on is a matter
        of timing.

        Either way everything read so far is settled as read and the buffer
        is cut - in the pause if there is one, at the end of the last word
        read if there is not - so the next pass starts on the speech that
        was being ignored.

        Only when two passes running have stopped at the same place.
        Settling "as read" makes whatever was read permanent, and one pass
        is not grounds for that: the recogniser reads the breath before a
        first word as a phrase on one pass and reads straight through it,
        correctly, on the next.
        """
        said = stream.hypothesis.last_pass
        read_until = said[-1].end if said else stream.buffer_start
        where = cut = None
        for pause in gate.pauses:
            if pause.start < stream.buffer_start or pause.end < read_until - 0.15:
                continue
            if stream.stream_time - pause.end >= _STALL_SEC:
                where, cut = pause.start, self._cut_point(pause, stream)
            break
        if (
            where is None
            and said
            and gate.in_utterance
            and stream.stream_time - read_until >= _STALL_NO_PAUSE_SEC
        ):
            where = cut = read_until
        previous, self._stalled_at = self._stalled_at, where
        if where is None or previous is None or abs(previous - where) > 0.3:
            return
        self._stalled_at = None
        self.metrics.stall_splits += 1
        self._split(stream, gate, cut)

    def _split(self, stream: StreamingTranscriber, gate: SpeechGate, at: float) -> None:
        """Cut the buffer at stream time `at`, settling what was said before it."""
        self._note_settled(stream, stream.split(at))
        pause = gate.open_pause
        if pause is not None and at >= pause.start:
            # Cut inside a pause that is still going on: nobody has spoken
            # since, and there is nothing in the buffer to read.
            self._quiet_base = gate.voiced_total
        else:
            # Speech was left in the buffer, and no pass has read it as the
            # buffer now stands.
            self._quiet_base = None
            self._pass_voiced = None

    def _heard_the_end(self, gate: SpeechGate) -> bool:
        """Has a pass already looked at everything said, with quiet after it?

        True when nothing has been voiced since the last pass and that pass
        ran far enough into the pause to have heard the last word finish.
        Running the recogniser again would hand it the same speech plus more
        silence, and get the same answer a hundred milliseconds later.
        """
        return (
            self._pass_voiced is not None
            and self._pass_voiced == gate.voiced_total
            and self._pass_quiet >= _HEARD_QUIET_SEC
        )

    def _hold_sec(self, text: str) -> float:
        """How much silence it takes to believe this text is a whole sentence.

        The speaker has stopped. How long to wait before believing they have
        finished depends on how finished it sounds: a predicate, not at all;
        a dangling particle, a good while; anything else, somewhere between.
        """
        cfg = self.cfg
        if self._direct or not text or not has_content(text):
            return 0.0
        if looks_complete(text) and not is_fragment(text):
            return 0.0
        if looks_incomplete(text):
            return cfg.long_pause_sec
        return (cfg.endpoint_silence_sec + cfg.long_pause_sec) / 2

    def _reads_finished(self, stream: StreamingTranscriber, gate: SpeechGate) -> bool:
        """The speaker is pausing, and what a pass heard of it is a sentence."""
        cfg = self.cfg
        if cfg.eager_endpoint_sec <= 0 or self._direct or not gate.in_utterance:
            return False
        if gate.open_pause is None or not self._heard_the_end(gate):
            return False
        text = join_words(self._open + stream.hypothesis.unstable)
        return bool(text) and looks_complete(text) and not is_fragment(text)

    def _eager_ready(self, stream: StreamingTranscriber, gate: SpeechGate) -> bool:
        """May the open utterance be closed before the endpoint silence is up?

        Yes, once a pass made during the pause reads as a finished sentence
        and the pause has lasted eager_endpoint_sec. Both halves were
        measured, because each looked like the place to save time:

        The wait cannot be much shorter. Closing 0.2 s into a pause put the
        error rate up from 15.6% to 16.2%, nearly all of it on the samples
        where people pause mid-sentence: at 0.2 s the speaker has often only
        drawn breath, and the line is cut in the middle of なんとなく.

        A second pass to confirm the first buys nothing. Requiring two
        agreeing passes gave 15.4% against 15.5% for one pass and the same
        0.3 s of patience - they are two readings of the same audio by the
        same model, and agree with each other whether or not they are right.
        It was the waiting that helped, and waiting is free.
        """
        if gate.trailing_quiet_sec < self.cfg.eager_endpoint_sec:
            return False
        return self._reads_finished(stream, gate)

    def _close_utterance(self, stream: StreamingTranscriber, gate: SpeechGate) -> None:
        """The speaker has stopped. Finish the sentence, or wait to be sure."""
        heard_enough = gate.utterance_speech_sec >= self.cfg.min_speech_sec
        if not heard_enough:
            if self._deferred_hold is not None:
                # A click or a breath after speech that is still being held.
                # It is in the buffer now and does no harm there; the wait
                # for the speaker to go on simply starts again.
                return
            # A click or a breath. Whisper would caption it anyway.
            stream.discard_audio()
            self._pass_voiced = None
            self._emit_caption(stream)
            self._flush(stream, gate, closed=True)
            return

        if self._heard_the_end(gate):
            self.metrics.passes_reused += 1
        elif stream.buffer_sec > 0:
            # One last look with the whole utterance in the buffer.
            self._pass(stream, 0.0, force=True)

        pending = self._open + stream.hypothesis.unstable
        text = join_words(pending)
        hold = self._hold_sec(text)
        overdue = bool(pending) and (
            pending[-1].end - pending[0].start >= self.cfg.mt_max_utterance_sec
        )
        if hold > gate.silence_sec and not overdue:
            # It does not read as finished: it stops on a particle, or it is
            # a syllable or two. Keep the audio and keep listening. If the
            # speaker goes on, the recogniser hears one sentence instead of
            # two halves; if they do not, the loop finalises it once the
            # silence has lasted `hold`.
            self._deferred_hold = hold
            self.metrics.deferred_closes += 1
            self._emit_caption(stream)
            self._flush(stream, gate, closed=False)
            self._request_preview(stream)
            return
        self._finalize(stream, gate)

    def _finalize(self, stream: StreamingTranscriber, gate: SpeechGate) -> None:
        """Settle everything said so far and hand it on."""
        self._deferred_hold = None
        self._quiet_base = None
        if not self._heard_the_end(gate) and stream.buffer_sec > 0:
            self._pass(stream, 0.0, force=True)
        self._read_to_the_end(stream, gate)
        # Nothing more is coming, so there is nothing for a later pass to
        # agree with: everything is settled, on the last pass's reading of
        # the whole utterance.
        self._note_settled(stream, stream.end_utterance())
        self._pass_voiced = None
        if not self._direct and gate.pauses:
            # When the speaker stopped, on the stream clock: the start of the
            # pause that closed the utterance. Words placed after it were not
            # said.
            unspoken = stream.hypothesis.drop_tail(gate.pauses[-1].start + _TAIL_SLACK_SEC)
            if unspoken:
                self.metrics.tail_drops += 1
                log.info("dropped unspoken tail: %r", join_words(unspoken))
        self._emit_caption(stream)
        self._flush(stream, gate, closed=True)
        stream.prune_history(30.0)
        gate.prune(stream.stream_time - 60.0)
        if len(self._pause_marks) > 400:
            horizon = stream.stream_time - 60.0
            self._pause_marks = {t: n for t, n in self._pause_marks.items() if t >= horizon}
        if self.tracker is not None:
            self.tracker.prune(stream.stream_time - 60.0)

    def _read_to_the_end(self, stream: StreamingTranscriber, gate: SpeechGate) -> None:
        """Make sure the last reading got as far as the speaker did.

        The same failure _unstall watches for while someone is talking, met
        at the end instead: the final pass stops well short of where the
        speech stopped. Nothing later will pick the rest up - this is the
        last look - so what was read is settled, the buffer is cut where the
        reading ended, and the remainder is read by itself. A few times at
        most; a recogniser that keeps stopping is not going to be talked
        round.
        """
        if not gate.pauses or self._direct:
            return
        speech_end = gate.pauses[-1].start
        for _ in range(4):
            said = stream.hypothesis.last_pass
            if not said:
                return
            read_until = said[-1].end
            # The same allowance as in mid-speech, though there is no lag
            # to allow for here. A tighter one was tried: at 0.8 s a laugh
            # after the last word was sent back to the recogniser by itself
            # and came back as a sentence nobody said.
            if (
                speech_end - read_until < _STALL_NO_PAUSE_SEC
                or read_until <= stream.buffer_start + 0.2
            ):
                return
            self.metrics.stall_splits += 1
            self._note_settled(stream, stream.split(read_until))
            if not self._pass(stream, 0.0, force=True):
                return

    def _wall(self, stream_time: float) -> float:
        """perf_counter time at which a moment on the stream clock happened."""
        return self._clock + stream_time

    def _note_settled(self, stream: StreamingTranscriber, words: list[Word]) -> None:
        """Record how long the words that just settled took to do so."""
        revised = stream.hypothesis.revisions
        if revised:
            self.metrics.revised_lines += revised
            stream.hypothesis.revisions = 0
        if not words:
            return
        # From the word being spoken to its text being shown as settled.
        lag = max(0.0, time.perf_counter() - self._wall(words[-1].end))
        self.metrics.commits += 1
        self.metrics.source_lag.add(lag)

    def _emit_caption(self, stream: StreamingTranscriber) -> None:
        committed = join_words(self._open)
        unstable = join_words(stream.hypothesis.unstable)
        if self._direct:
            # Whisper is writing English itself; the line in progress is the
            # preview and there is no Japanese to show.
            self._emit_preview(f"{committed} {unstable}".strip(), "")
            return
        if (committed, unstable) == self._caption:
            return
        self._caption = (committed, unstable)
        self._emit("on_caption", committed, unstable)

    # -- segmentation ------------------------------------------------------

    def _speaker_of(self, words: list[Word]) -> int:
        """Who said this run of words, over its whole span."""
        if self.tracker is None or not words:
            return UNKNOWN
        return self.tracker.speaker_for(words[0].start, words[-1].end)

    def _speaker_cut(self, words: list[Word]) -> int:
        """How many leading words precede the first change of speaker. 0 if none.

        Windows the tracker could not attribute are skipped rather than treated
        as a change: a brief quiet patch mid-sentence would otherwise split an
        utterance in two and hand the translator both halves separately.

        A change one or two words in is almost always the previous turn's tail
        landing inside a boundary window, not a real turn, so changes that
        would strand less than mt_min_chunk_sec are ignored.
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
                if words[i - 1].end - words[0].start < self.cfg.mt_min_chunk_sec:
                    return 0
                return i
        return 0

    def _decide(
        self, pending: list[Word], stream: StreamingTranscriber, gate: SpeechGate, closed: bool
    ) -> int:
        """How many leading words of `pending` to translate now. 0 to wait."""
        cfg = self.cfg
        unstable = stream.hypothesis.unstable
        pauses = list(gate.pauses)
        if gate.open_pause is not None:
            pauses.append(gate.open_pause)
        for pause in pauses:
            pause.letters = self._pause_marks.get(round(pause.start, 3))
        base = self._sent_letters

        next_word = unstable[0].text if unstable else ""
        next_start = unstable[0].start if unstable else None
        # A sentence mark on the last settled word is trusted only if the
        # speaker has actually stopped there.
        paused_at_end = closed or len(pending) in pause_boundaries(
            pending, pauses, next_start, base
        )
        cut = sentence_cut(pending, next_word=next_word, trailing_ok=paused_at_end)
        if not closed and not self._direct:
            cut = max(
                cut,
                pause_cut(
                    pending,
                    pauses,
                    next_word=next_word,
                    next_start=next_start,
                    base_letters=base,
                ),
            )
        change = self._speaker_cut(pending)
        if change:
            # The tracker says where the voice changed to within a second or
            # so. The line is cut at the pause nearest that, not at the word.
            change = nearest_boundary(
                pending, pause_boundaries(pending, pauses, next_start, base), change
            )
        if change and (cut == 0 or change < cut):
            cut = change

        span = pending[-1].end - pending[0].start
        if closed:
            rest = pending[cut:]
            if rest:
                # The speaker has stopped. How long to wait before believing
                # they have finished depends on how finished it sounds:
                # a predicate, not at all; a dangling particle, a good while;
                # anything else, somewhere between.
                release = gate.silence_sec >= self._hold_sec(join_words(rest))
                if release or span >= cfg.mt_max_utterance_sec:
                    cut = len(pending)
            return cut

        if cut == 0 and (
            span >= cfg.mt_max_utterance_sec or len(join_words(pending)) >= 110
        ):
            # Overdue. Something has to be shown; cut at the best place there is.
            cut = overdue_cut(pending, pauses, next_word=next_word, base_letters=base)
        return cut

    def _flush(self, stream: StreamingTranscriber, gate: SpeechGate, closed: bool) -> None:
        """Send whatever has become a finished unit to the translator.

        Called after every pass, not only at end of utterance: waiting for
        silence means a conversation with no gaps in it never gets translated
        at all.
        """
        pending = self._open
        if not pending:
            self._holding = False
            return
        cut = self._decide(pending, stream, gate, closed)
        if cut <= 0:
            self._holding = closed
            return
        # From here these words are a line of the transcript. Nothing that
        # is recognised later can change them.
        chunk = stream.take(cut)
        self._sent_letters += sum(_letter_count(w.text) for w in chunk)
        self._holding = closed and bool(stream.pending)
        text = join_words(chunk)
        if text:
            self._emit(
                "on_commit", text, self._speaker_of(chunk), chunk[0].start, chunk[-1].end
            )
        if not closed:
            self._split_after(chunk, stream, gate)
        self._emit_unit(chunk, stream)
        # The words just cut are no longer "in progress".
        self._emit_caption(stream)

    def _split_after(
        self, chunk: list[Word], stream: StreamingTranscriber, gate: SpeechGate
    ) -> None:
        """A line was cut while the speaker is still going. If it ended at a
        pause, cut the audio there as well.

        The line's audio is of no further use to the recogniser - its text is
        fixed - and left in the buffer it does harm: see
        StreamingTranscriber.split.
        """
        rest = stream.pending or stream.hypothesis.unstable
        after = (chunk[-1].start + chunk[-1].end) / 2.0
        until = (rest[0].start + rest[0].end) / 2.0 if rest else float("inf")
        pauses = list(gate.pauses)
        if gate.open_pause is not None:
            pauses.append(gate.open_pause)
        for pause in reversed(pauses):
            if pause.end <= stream.buffer_start:
                break
            # The pause has to be the one the line ended at, not some later
            # one: cutting there would throw away speech nobody has read.
            if pause.start > chunk[-1].end + 0.4:
                continue
            if after < pause.center <= until:
                self.metrics.pause_splits += 1
                self._split(stream, gate, self._cut_point(pause, stream))
                return

    def _emit_unit(self, chunk: list[Word], stream: StreamingTranscriber) -> None:
        source = join_words(chunk)
        # The words are consumed either way; text with nothing in it is dropped
        # here rather than shown as an empty caption.
        if not source or not has_content(source):
            return
        if self._glossary is not None:
            # Slang the recogniser spelt as something else, put right before
            # anything translates it or the transcript shows it.
            source = self._glossary.respell(source)
        if source.strip().lower() in _HALLUCINATIONS:
            return
        # A syllable or two that repeats how the previous unit ended is that
        # unit's last token heard twice across the boundary, not speech. On
        # its own it translates to something confident and invented - a lone
        # だ came back as "That's right."
        core = "".join(ch for ch in source if ch.isalnum())
        if len(core) <= 2 and self._last_unit.endswith(core):
            return
        self._last_unit = core

        spoken_at = self._wall(chunk[-1].end)
        utterance = Utterance(
            id=self._next_id,
            source=source,
            speaker=self._speaker_of(chunk),
            start=chunk[0].start,
            end=chunk[-1].end,
            wall_time=time.time() - max(0.0, time.perf_counter() - spoken_at),
            final=False,
        )
        self._next_id += 1
        self._epoch += 1
        self._preview_fresh = True
        self.metrics.utterances += 1

        if self._direct:
            utterance.english, utterance.source = source, ""
            utterance.engine = "whisper"
            self._finish(utterance, spoken_at)
            return
        if self._final_tr is None:
            self._finish(utterance, spoken_at)
            return

        # With an LLM finishing the line there is a wait of a few hundred
        # milliseconds. If a draft of exactly this text is already on screen,
        # move it into the transcript as provisional now, so the line appears
        # at once and is corrected once, in place.
        slow = getattr(self._final_tr, "name", "") == "llm"
        if slow and self._preview_shown and self._preview_source == source:
            utterance.english = self._preview_shown
            utterance.engine = "nmt"
            self._send(utterance)
            self._preview_source = ""
            self._emit_preview("", "")
        self._final_q.put(_Job(utterance, spoken_at, self._preview_gen))

    def _send(self, utterance: Utterance) -> None:
        # A copy, because the worker goes on to change this object and the
        # listener may read what it was given on another thread, later.
        self._emit("on_utterance", replace(utterance))

    def _finish(self, utterance: Utterance, spoken_at: float) -> None:
        utterance.final = True
        utterance.latency = max(0.0, time.perf_counter() - spoken_at)
        self.metrics.final_latency.add(utterance.latency)
        self._send(utterance)
        if self._refiner is not None and utterance.source and not self._stop.is_set():
            with self._refine_cond:
                self._seen.append(_Seen(replace(utterance), time.perf_counter()))
                extra = len(self._seen) - _SEEN_LINES
                if extra > 0:
                    del self._seen[:extra]
                    self._seen_pos = max(0, self._seen_pos - extra)
                self._refine_cond.notify()

    # -- final translation -------------------------------------------------

    def _final_worker(self) -> None:
        """Translate finished units, in order, off the engine thread.

        One worker, so lines are finished in the order they were spoken.
        """
        while True:
            job = self._final_q.get()
            if job is None:
                return
            # Past this point the run is over: drain without translating, so
            # teardown is not held up by work nobody will see.
            if self._stop.is_set():
                continue
            self._final_busy = True
            try:
                self._translate_final(job)
            except Exception:  # noqa: BLE001 - one bad line must not end the run
                log.exception("final translation failed")
                self._finish(job.utterance, job.spoken_at)
            finally:
                self._final_busy = False
                # The moment a line has just been finished is the moment the
                # next one is furthest off, so it is when a second look is
                # least likely to be in its way.
                with self._refine_cond:
                    self._refine_cond.notify()

    def _translate_final(self, job: _Job) -> None:
        utterance = job.utterance
        translator = self._final_tr
        is_llm = getattr(translator, "name", "") == "llm"
        had_draft = bool(utterance.english)
        t0 = time.perf_counter()
        first_token: list[float] = []

        def on_partial(part: str) -> None:
            if not first_token:
                first_token.append(time.perf_counter() - t0)
            # A draft that is already a whole sentence is better to look at
            # than half of a better one. Stream only when there is nothing
            # on the line yet.
            if had_draft or self._stop.is_set() or not self._live:
                return
            utterance.english = part
            utterance.engine = "llm"
            self._send(utterance)

        english = translator.translate(
            utterance.source, on_partial=on_partial if is_llm else None
        )
        engine = getattr(translator, "name", "")
        error = getattr(translator, "last_error", None)
        if first_token:
            self.metrics.llm_first_token_sec.add(first_token[0])

        if not english and error:
            self.metrics.mt_failures += 1
            english, engine = self._fallback(utterance.source, error, is_llm)
        elif is_llm:
            self._llm_failures = 0
        elif self._llm_replaced and english:
            # The LLM was given up on; every line since is a fallback line.
            self.metrics.fallback_lines += 1

        self.metrics.mt_final_sec.add(time.perf_counter() - t0)
        if self._stop.is_set():
            return
        utterance.english = english
        utterance.engine = engine if english else ""
        self._finish(utterance, job.spoken_at)
        # The NMT route leaves its preview up until the real line lands, so
        # the English never blinks out between the two. Take it down now,
        # unless a preview for the next sentence has already replaced it.
        if self._preview_gen == job.preview_gen and not is_llm:
            self._preview_source = ""
            self._emit_preview("", "")

    def _fallback(self, source: str, error: str, was_llm: bool) -> tuple[str, str]:
        """Finish a line the main translator could not. Returns (english, engine)."""
        nmt = self._nmt
        if not was_llm or nmt is None or not self.cfg.mt_fallback_to_nmt:
            if not self._fallback_announced:
                self._fallback_announced = True
                self._notice("warning", f"A line could not be translated: {error}")
            return "", ""
        self._llm_failures += 1
        english = nmt.translate(source, beam_size=self.cfg.mt_beam_size)
        self.metrics.fallback_lines += 1
        if not self._fallback_announced:
            self._fallback_announced = True
            self._notice(
                "warning",
                f"{error} Finishing lines with the built-in translator meanwhile.",
            )
        if self._llm_failures >= 3 and self._final_tr is not nmt:
            # Stop paying the timeout on every line. Start again to retry.
            self._final_tr = nmt
            self._llm_replaced = True
            self.metrics.translator = "Sugoi v4 (fallback)"
            self._notice(
                "warning",
                "The translation model is still not answering, so the built-in "
                "translator has taken over. Press Stop then Start to try it again.",
            )
        return english, "nmt"

    # -- second look -------------------------------------------------------

    def _refine_pick(self):
        """The next line due its second look, with its neighbours and whether
        the next line follows straight on from it, or None.

        Called with _refine_cond held. A line is due when enough later lines
        exist to read it against, or when it has waited long enough that no
        more are coming soon. Nothing is due while a first translation is in
        hand or queued: the model is one model, and a line nobody has seen
        yet goes before a line being improved.
        """
        cfg = self.cfg
        # Joining a line to the next takes one line after it. Only the model
        # path has a use for more.
        lookahead = cfg.refine_lookahead if self._refiner.uses_model else 1
        while len(self._seen) - self._seen_pos > _REFINE_BACKLOG:
            self._seen_pos += 1
            self.metrics.second_look_skipped += 1
        waiting = len(self._seen) - self._seen_pos
        if waiting <= 0 or self._final_busy or not self._final_q.empty():
            return None
        index = self._seen_pos
        seen = self._seen[index]
        age = time.perf_counter() - seen.done_at
        if waiting - 1 < lookahead and age < cfg.refine_wait_sec:
            return None
        self._seen_pos += 1
        # Lines from before a long silence are another conversation.
        horizon = seen.done_at - cfg.llm_context_reset_sec
        first = max(0, index - cfg.refine_context_lines)
        before = [
            (s.utterance.source, s.utterance.english)
            for s in self._seen[first:index]
            if s.done_at >= horizon
        ]
        following = self._seen[index + 1 : index + 1 + lookahead]
        after = [(s.utterance.source, s.utterance.english) for s in following]
        # Did the next line come straight after this one, in the same voice?
        follows_on = bool(following) and (
            0.0 <= following[0].utterance.start - seen.utterance.end <= _JOIN_GAP_SEC
            and following[0].utterance.speaker == seen.utterance.speaker
        )
        return seen, before, after, follows_on

    def _refine_worker(self) -> None:
        from .refine import Line

        refiner = self._refiner
        failures = 0
        while not self._stop.is_set():
            with self._refine_cond:
                picked = self._refine_pick()
                if picked is None:
                    self._refine_cond.wait(0.5)
                    continue
            seen, before, after, follows_on = picked
            utterance = seen.utterance
            t0 = time.perf_counter()
            revision = refiner.refine(
                Line(utterance.source, utterance.english),
                [Line(*pair) for pair in before],
                [Line(*pair) for pair in after],
                follows_on=follows_on,
            )
            verdict = refiner.last_verdict
            if verdict == "skipped":
                continue
            self.metrics.second_look_sec.add(time.perf_counter() - t0)
            self.metrics.second_looks += 1
            if verdict == "error":
                failures += 1
                if failures >= 3:
                    # Stop paying the timeout on every line. First
                    # translations are unaffected; Start again to retry.
                    self._notice(
                        "warning",
                        f"The second look has stopped: {refiner.last_error}",
                    )
                    return
                continue
            failures = 0
            if verdict.startswith("bad"):
                self.metrics.second_look_rejected += 1
            if revision is None or self._stop.is_set():
                continue
            first_source, first_english = utterance.source, utterance.english
            with self._refine_cond:
                # Under the lock because _refine_pick reads these lines as
                # the context of the next one.
                utterance.source = revision.source
                utterance.english = revision.english
                utterance.revised = True
                utterance.first_source = first_source
                utterance.first_english = first_english
                if not utterance.engine:
                    utterance.engine = "llm"
            self.metrics.second_look_changes += 1
            amend = getattr(self._final_tr, "amend", None)
            if amend is not None:
                amend(first_source, revision.source, revision.english)
            self._emit("on_revision", replace(utterance))

    # -- preview -----------------------------------------------------------

    def _request_preview(self, stream: StreamingTranscriber) -> None:
        """Ask for provisional English for the sentence in progress.

        Built from the settled words plus, when the recogniser is fairly sure
        of it, its current guess at what follows. It will rewrite itself,
        sometimes reversing meaning, because a partial Japanese sentence does
        not determine an English prefix - which is why it is shown as a
        preview and never enters the transcript.
        """
        if self._preview_tr is None or not self._live or self._direct:
            return
        tail = list(self._open)
        unstable = stream.hypothesis.unstable
        if unstable:
            confidence = sum(w.probability for w in unstable) / len(unstable)
            if confidence >= self.cfg.mt_preview_min_confidence:
                tail += unstable
        source = join_words(tail)

        if len(source) < self.cfg.mt_live_min_chars or not has_content(source):
            # Nothing worth previewing. Clear it rather than leaving the last
            # preview stranded over a new sentence - but only once the line
            # it belonged to has landed, which _translate_final handles.
            return
        if source == self._preview_source:
            return  # unchanged; the preview on screen still stands
        now = time.perf_counter()
        if (
            not self._preview_fresh
            and now - self._preview_at < self.cfg.mt_preview_interval_sec
        ):
            return  # too soon after the last one to be readable
        self._preview_at = now
        self._preview_source = source
        started = self._wall(tail[0].start)
        with self._preview_cond:
            self._preview_req = (self._epoch, source, started)
            self._preview_cond.notify()

    def _preview_worker(self) -> None:
        while not self._stop.is_set():
            with self._preview_cond:
                while self._preview_req is None and not self._stop.is_set():
                    self._preview_cond.wait(0.5)
                request, self._preview_req = self._preview_req, None
            if request is None or self._stop.is_set():
                continue
            epoch, source, started = request
            t0 = time.perf_counter()
            try:
                english = self._preview_tr.translate(
                    source, beam_size=self.cfg.mt_preview_beam_size
                )
            except Exception:  # noqa: BLE001 - a preview is never worth a crash
                log.exception("preview translation failed")
                continue
            self.metrics.mt_preview_sec.add(time.perf_counter() - t0)
            # The sentence this was for has been cut since it was asked for;
            # the transcript has, or is about to have, the real thing.
            if epoch != self._epoch or self._stop.is_set() or not self._live:
                continue
            if not english:
                continue
            if self._preview_fresh:
                self._preview_fresh = False
                self.metrics.preview_latency.add(
                    max(0.0, time.perf_counter() - started)
                )
            self.metrics.previews += 1
            self._emit_preview(english, source)

    def _emit_preview(self, english: str, source: str) -> None:
        if english == self._preview_shown:
            return
        self._preview_shown = english
        self._preview_gen += 1
        self._emit("on_preview", english, source)

    # -- teardown ----------------------------------------------------------

    def _summary(self) -> None:
        s = self.metrics.snapshot()
        log.info(
            "pipeline stopped: %d passes, mean %s ms, RTF %.2f, source lag %s s, "
            "final latency p50 %s s / p90 %s s, %d utterances, %d backlog drops, "
            "MT final %s ms, preview %s ms, %d MT failures",
            s["passes"],
            s["mean_pass_ms"],
            s["rtf"],
            s["source_lag_sec"],
            s["final_latency_sec"],
            s["final_latency_p90_sec"],
            s["utterances"],
            s["drops"],
            s["mt_final_ms"],
            s["mt_preview_ms"],
            s["mt_failures"],
        )
