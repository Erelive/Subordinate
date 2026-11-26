"""Deciding when someone is speaking, and when they have stopped.

Two detectors live here, and the difference between them is the difference
between the old pipeline and this one.

EnergyGate answers "is anything playing". It was the only gate the pipeline
had, and it meant an utterance ended when the *audio* went quiet for 0.7 s.
Over a game soundtrack or a stream's background music that never happens, so
the only things that ever closed a sentence were Whisper's own punctuation and
a timer - measured on the benchmark set, the finished English landed a median
2.4 s after the speaker had stopped.

SileroVAD answers "is anyone *talking*". It runs on the CPU in about a
millisecond per 32 ms frame, and its answer goes quiet between sentences
whether or not music is playing. SpeechGate turns that frame-by-frame answer
into the three things the pipeline needs:

  - which audio is worth sending to the recogniser at all (music with nobody
    talking never reaches the GPU, which is also what stops Whisper
    hallucinating lyrics-shaped text into silence);
  - the moment an utterance ends, a few hundred milliseconds after the last
    word rather than whenever the room goes quiet;
  - where the short pauses *inside* running speech are, on the same clock as
    the recogniser's word timestamps, so a sentence can be cut at a breath
    instead of at an arbitrary word.

The Silero model is the one faster-whisper already ships for its own
vad_filter, so this costs no download and no new dependency.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

# Floor for the dB conversion, so digital silence doesn't produce -inf.
_EPS = 1e-10


@dataclass
class GateState:
    active: bool  # audio present in the block just fed
    trailing_silence_sec: float  # continuous silence up to now


class EnergyGate:
    """RMS threshold. Kept as the fallback when the VAD model cannot load."""

    def __init__(self, threshold_db: float = -48.0):
        self.threshold_db = threshold_db
        self._tail = np.empty(0, dtype=np.float32)
        self._silence_sec = 0.0
        self._saw_audio = False

    def reset(self) -> None:
        self._tail = np.empty(0, dtype=np.float32)
        self._silence_sec = 0.0
        self._saw_audio = False

    @property
    def trailing_silence_sec(self) -> float:
        return self._silence_sec

    def update(self, chunk: np.ndarray) -> GateState:
        """Feed a mono 16 kHz block; returns whether it carried audio."""
        if chunk.size:
            self._tail = np.concatenate((self._tail, chunk)) if self._tail.size else chunk

        n_frames = self._tail.size // FRAME_LEN
        active_any = False
        if n_frames:
            frames = self._tail[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
            self._tail = self._tail[n_frames * FRAME_LEN :]

            rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
            db = 20.0 * np.log10(np.maximum(rms, _EPS))
            loud = db > self.threshold_db
            active_any = bool(loud.any())

            if active_any:
                self._saw_audio = True
                # Silence resumes only after the last loud frame in this block.
                trailing_quiet = int(n_frames - 1 - np.flatnonzero(loud)[-1])
                self._silence_sec = trailing_quiet * FRAME_MS / 1000.0
            elif self._saw_audio:
                self._silence_sec += n_frames * FRAME_MS / 1000.0

        return GateState(active=active_any, trailing_silence_sec=self._silence_sec)


# -- Silero -----------------------------------------------------------------

# Silero v5 at 16 kHz takes exactly this many samples per step and nothing
# else; the model has no notion of a variable hop.
VAD_FRAME = 512
VAD_FRAME_SEC = VAD_FRAME / SAMPLE_RATE  # 32 ms
# Each frame is prefixed with the tail of the one before it.
_VAD_CONTEXT = 64


class VadUnavailable(RuntimeError):
    """Raised when the Silero model cannot be loaded."""


class SileroVAD:
    """Streaming speech probability, one value per 32 ms frame.

    faster-whisper's own wrapper only scores a whole buffer from a zero state,
    which is the right shape for its vad_filter and the wrong one here: scoring
    a stream that way would mean re-running the recurrent half over everything
    heard so far on every block. This drives the same two ONNX graphs directly
    and carries the LSTM state and the 64-sample context between calls, so
    each frame is scored exactly once.
    """

    def __init__(self) -> None:
        try:
            import onnxruntime
            from faster_whisper.utils import get_assets_path
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise VadUnavailable(f"missing dependency for VAD: {exc}") from exc

        import os

        assets = get_assets_path()
        encoder = os.path.join(assets, "silero_encoder_v5.onnx")
        decoder = os.path.join(assets, "silero_decoder_v5.onnx")
        if not (os.path.exists(encoder) and os.path.exists(decoder)):
            raise VadUnavailable(f"Silero VAD model files not found in {assets}")

        opts = onnxruntime.SessionOptions()
        # One thread each. The work is tiny, and more threads here would only
        # contend with the capture converter and the loop feeding the GPU.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        try:
            self._encoder = onnxruntime.InferenceSession(
                encoder, providers=["CPUExecutionProvider"], sess_options=opts
            )
            self._decoder = onnxruntime.InferenceSession(
                decoder, providers=["CPUExecutionProvider"], sess_options=opts
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt model file lands here
            raise VadUnavailable(f"could not load Silero VAD: {exc}") from exc
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_VAD_CONTEXT, dtype=np.float32)

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        """Speech probability for each row of an (n, 512) float32 array."""
        n = frames.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # Row k is prefixed with the last 64 samples of row k-1; row 0 with
        # what was carried over from the previous call.
        contexts = np.empty((n, _VAD_CONTEXT), dtype=np.float32)
        contexts[0] = self._context
        if n > 1:
            contexts[1:] = frames[:-1, -_VAD_CONTEXT:]
        self._context = frames[-1, -_VAD_CONTEXT:].copy()
        batch = np.concatenate((contexts, frames), axis=1)

        # The convolutional half has no state, so every frame goes through it
        # in one call; only the recurrent half has to be stepped in order.
        encoded = self._encoder.run(None, {"input": batch})[0].reshape(n, 1, 128)
        probs = np.empty(n, dtype=np.float32)
        state = self._state
        for k in range(n):
            out, state = self._decoder.run(None, {"input": encoded[k], "state": state})
            probs[k] = float(np.asarray(out).reshape(-1)[0])
        self._state = state
        return probs


class EnergyVAD:
    """Stand-in for SileroVAD built on signal level, with the same surface.

    Used when the ONNX model will not load. It cannot tell speech from music,
    so over a soundtrack utterances end only when everything goes quiet - the
    old behaviour, kept as a floor rather than as a choice.
    """

    def __init__(self, threshold_db: float = -48.0):
        self.threshold = 10 ** (threshold_db / 20.0)

    def reset(self) -> None:
        pass

    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        return (rms > self.threshold).astype(np.float32)


# -- gate -------------------------------------------------------------------

AUDIO = "audio"  # payload: samples to append to the recogniser's stream
CLOSE = "close"  # the utterance has ended; settle it
PAUSE = "pause"  # a short pause inside running speech was just confirmed


@dataclass
class GateEvent:
    kind: str
    audio: np.ndarray | None = None


@dataclass
class Pause:
    """A stretch with nobody talking, on the recogniser's stream clock."""

    start: float
    end: float
    # How much had been recognised when the recogniser looked at the audio up
    # to this pause: a count of content characters, from the start of the
    # run. Filled in by the pipeline when it has one; see segment.py for what
    # it is used for.
    letters: int | None = None

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


class SpeechGate:
    """Frames in, utterances out.

    The gate owns the decision of which audio the recogniser sees. While an
    utterance is open every frame is passed through, pauses included, so the
    stream stays contiguous and word timestamps stay meaningful. Outside one,
    frames are held in a short pre-roll and otherwise discarded.

    Stream clock. `stream_time` counts the audio emitted as AUDIO events, in
    seconds. StreamingTranscriber's clock counts the audio inserted into it.
    The pipeline inserts exactly what this emits, so the two are the same
    clock, which is what lets a Pause recorded here be compared with a Word
    timestamp produced there.
    """

    def __init__(
        self,
        vad,
        threshold: float = 0.5,
        endpoint_silence_sec: float = 0.45,
        soft_pause_sec: float = 0.16,
        preroll_sec: float = 0.3,
        min_speech_sec: float = 0.2,
    ):
        self.vad = vad
        self.threshold = threshold
        # Hysteresis, as Silero's own reference implementation uses: a frame
        # must clear `threshold` to start speech, but only has to stay above
        # this to continue it. Without the gap a voiced fricative flaps the
        # gate open and shut inside one word.
        self.release = max(threshold - 0.15, 0.05)
        self.endpoint_frames = max(1, round(endpoint_silence_sec / VAD_FRAME_SEC))
        self.soft_pause_frames = max(2, round(soft_pause_sec / VAD_FRAME_SEC))
        self.preroll_frames = max(1, round(preroll_sec / VAD_FRAME_SEC))
        self.min_speech_frames = max(1, round(min_speech_sec / VAD_FRAME_SEC))

        self._pending = np.empty(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self.preroll_frames)
        self.in_utterance = False
        self._voiced_last = False
        # Consecutive non-speech frames, counted whether or not an utterance
        # is open; this is "how long since anyone spoke".
        self._quiet_frames = 0
        self._speech_frames = 0  # voiced frames in the open utterance
        # Voiced frames ever. It moves if and only if someone has spoken, so
        # two readings that match mean "nothing was said in between".
        self.voiced_total = 0
        self._emitted = 0  # samples handed out as AUDIO, ever
        self._pause_start: float | None = None
        self._pause_announced = False
        self.pauses: deque[Pause] = deque()
        self._ever_spoke = False
        self.last_probability = 0.0

    # -- state -------------------------------------------------------------

    @property
    def stream_time(self) -> float:
        return self._emitted / SAMPLE_RATE

    @property
    def silence_sec(self) -> float:
        """How long it has been since a voiced frame. inf before the first."""
        if not self._ever_spoke:
            return float("inf")
        return self._quiet_frames * VAD_FRAME_SEC

    @property
    def trailing_quiet_sec(self) -> float:
        """Quiet at the end of the open utterance so far. 0 outside one."""
        if not self.in_utterance:
            return 0.0
        return self._quiet_frames * VAD_FRAME_SEC

    def end_utterance(self) -> bool:
        """Close the open utterance now, without waiting out the silence.

        For the pipeline, when it has decided from the text that the sentence
        is over. The quiet that follows is then ordinary silence between
        utterances: it goes to the pre-roll, and the counter of how long
        nobody has spoken keeps running from where it was.
        """
        if not self.in_utterance:
            return False
        self._close([])
        return True

    @property
    def buffered_sec(self) -> float:
        """Audio taken in but not yet framed. Under one frame, ~32 ms."""
        return self._pending.size / SAMPLE_RATE

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.empty(0, dtype=np.float32)
        self._preroll.clear()
        self.in_utterance = False
        self._voiced_last = False
        self._quiet_frames = 0
        self._speech_frames = 0
        self._pause_start = None
        self._pause_announced = False

    def prune(self, before: float) -> None:
        """Forget pauses older than `before` on the stream clock."""
        while self.pauses and self.pauses[0].end < before:
            self.pauses.popleft()

    # -- input -------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> list[GateEvent]:
        """Take captured audio. Returns what the pipeline should do, in order."""
        if samples.size:
            self._pending = (
                np.concatenate((self._pending, samples)) if self._pending.size else samples
            )
        n = self._pending.size // VAD_FRAME
        if n == 0:
            return []
        frames = self._pending[: n * VAD_FRAME].reshape(n, VAD_FRAME)
        self._pending = self._pending[n * VAD_FRAME :]
        probs = self.vad.probabilities(np.ascontiguousarray(frames, dtype=np.float32))
        return self._advance(frames, probs)

    def feed_silence(self, seconds: float) -> list[GateEvent]:
        """Account for wall-clock time in which the device delivered nothing.

        WASAPI loopback stops producing samples altogether when nothing is
        rendering - a paused video is not silence, it is no data. The old loop
        only measured silence in samples it received, so the last sentence
        before a pause was never closed until something else played. This
        advances the same counters by the clock instead.

        The quiet is not inserted into the recogniser's stream: nothing was
        said in it, and the stream clock only counts audio that exists.
        """
        n = int(seconds / VAD_FRAME_SEC)
        if n <= 0:
            return []
        # Whatever was half-framed or held as pre-roll belongs to audio from
        # before the gap; it must not be glued onto what comes after it.
        self._pending = np.empty(0, dtype=np.float32)
        self._preroll.clear()
        self.vad.reset()
        self._voiced_last = False
        events: list[GateEvent] = []
        if self.in_utterance and self._pause_start is None:
            self._pause_start = self.stream_time
        self._quiet_frames += n
        if self.in_utterance and self._quiet_frames >= self.endpoint_frames:
            self._close(events)
        return events

    # -- internals ---------------------------------------------------------

    def _advance(self, frames: np.ndarray, probs: np.ndarray) -> list[GateEvent]:
        events: list[GateEvent] = []
        run: list[np.ndarray] = []  # frames to emit as one AUDIO event

        def flush_run() -> None:
            if run:
                audio = np.concatenate(run)
                run.clear()
                events.append(GateEvent(AUDIO, audio))

        for frame, prob in zip(frames, probs):
            prob = float(prob)
            self.last_probability = prob
            voiced = prob >= (self.release if self._voiced_last else self.threshold)
            self._voiced_last = voiced

            if not self.in_utterance:
                if voiced:
                    self.voiced_total += 1
                    self._ever_spoke = True
                    self.in_utterance = True
                    self._quiet_frames = 0
                    self._speech_frames = 1
                    self._pause_start = None
                    # The pre-roll goes first, so the attack of the opening
                    # word is not lost to the gate's reaction time.
                    for held in self._preroll:
                        run.append(held)
                        self._emitted += held.size
                    self._preroll.clear()
                    run.append(frame)
                    self._emitted += frame.size
                else:
                    self._quiet_frames += 1
                    self._preroll.append(frame.copy())
                continue

            # Inside an utterance: everything passes through.
            frame_start = self._emitted / SAMPLE_RATE
            run.append(frame)
            self._emitted += frame.size
            if voiced:
                self._preroll.clear()
                self._speech_frames += 1
                self.voiced_total += 1
                if self._pause_start is not None:
                    if self._quiet_frames >= self.soft_pause_frames:
                        self.pauses.append(Pause(self._pause_start, frame_start))
                    self._pause_start = None
                    self._pause_announced = False
                self._quiet_frames = 0
                continue

            if self._pause_start is None:
                self._pause_start = frame_start
            self._quiet_frames += 1
            # Kept as pre-roll too. If the utterance is closed during this
            # quiet and the speaker starts again a moment later, the frames
            # just before the detector fires - the soft start of the first
            # word - are these ones, and the next utterance needs them even
            # though this one has already had them.
            self._preroll.append(frame.copy())
            if self._quiet_frames == self.soft_pause_frames and not self._pause_announced:
                # Long enough to be a breath rather than a stop consonant. The
                # pipeline uses this to run a pass now instead of waiting out
                # the interval, so a sentence that just ended is seen sooner.
                self._pause_announced = True
                flush_run()
                events.append(GateEvent(PAUSE))
            if self._quiet_frames >= self.endpoint_frames:
                flush_run()
                self._close(events)

        flush_run()
        return events

    def _close(self, events: list[GateEvent]) -> None:
        if self._pause_start is not None:
            self.pauses.append(Pause(self._pause_start, self.stream_time))
        self._pause_start = None
        self._pause_announced = False
        self.in_utterance = False
        self._voiced_last = False
        events.append(GateEvent(CLOSE))

    @property
    def open_pause(self) -> Pause | None:
        """The pause in progress, if speech has stopped but not yet ended."""
        if not self.in_utterance or self._pause_start is None:
            return None
        if self._quiet_frames < self.soft_pause_frames:
            return None
        return Pause(self._pause_start, self.stream_time)

    @property
    def utterance_speech_sec(self) -> float:
        """Voiced time in the utterance that is open or was last closed."""
        return self._speech_frames * VAD_FRAME_SEC


def build_vad(mode: str = "silero", energy_db: float = -48.0):
    """The detector `mode` asks for, falling back to energy if it will not load.

    Returns (vad, note). note is None normally, or a sentence for the user
    explaining what was substituted.
    """
    if mode == "energy":
        return EnergyVAD(energy_db), None
    try:
        return SileroVAD(), None
    except VadUnavailable as exc:
        log.warning("Silero VAD unavailable, using the energy gate: %s", exc)
        return (
            EnergyVAD(energy_db),
            "Voice detection model could not load; using a simple volume gate. "
            "Sentences will end late when music is playing.",
        )
