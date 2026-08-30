"""Turning repeated whole-buffer transcriptions into stable streaming captions.

Whisper is not a streaming model: it transcribes a whole buffer at once, and the
answer for a given word can change as more audio arrives. Re-running it every
half second and printing the raw output produces text that visibly rewrites
itself.

LocalAgreement-2 (Machacek et al., 2023) fixes that. Run inference on the growing
buffer; a word is *committed* only once two consecutive passes agree on it. The
committed prefix never changes again, so it can be shown as settled text, while
the disagreeing tail is shown dimmed as a live guess. Once the buffer grows past
its cap, the audio behind the last committed word is discarded.
"""

from __future__ import annotations

import logging

import numpy as np

from .asr import Transcriber, Word
from .config import SAMPLE_RATE, Config

log = logging.getLogger(__name__)


# Sentence and clause marks, in both the Japanese and ASCII forms Whisper emits.
_PUNCT = "。、．，…「」『』.,!?！？;:；："

# How much repeated text it takes to believe Whisper actually repeated itself,
# rather than two different words happening to share a character. Three is
# enough to exclude single particles while still catching the real case, which
# is a whole word or phrase re-emitted at the head of the next pass. The cost of
# being wrong in this direction is one duplicated short word; in the other, it
# is a silently deleted one.
_MIN_REPEAT_CHARS = 3


def _norm(text: str) -> str:
    """Normalise a word for the agreement comparison, ignoring punctuation.

    Whisper attaches a sentence mark only once it has heard enough to decide the
    sentence ended, so the same word arrives as 'します' on one pass and
    'します。' on the next. Comparing those literally makes LocalAgreement stall
    at exactly the sentence boundaries - the words whose punctuation matters most
    - and what finally reaches the caption, and the translation stage after it,
    is an unpunctuated run-on.

    Ignoring the marks here lets the word commit, and because flush() keeps the
    incoming copy, the punctuated form is the one that survives.
    """
    stripped = text.strip().lower()
    core = stripped.strip(_PUNCT)
    # A token that is nothing but punctuation still has to compare as itself,
    # or every one of them would normalise to "" and match each other.
    return core or stripped


class HypothesisBuffer:
    """Holds the previous hypothesis and commits the prefix that repeats."""

    def __init__(self) -> None:
        self.committed: list[Word] = []  # everything settled so far
        self.buffer: list[Word] = []  # previous pass, still unconfirmed
        self._incoming: list[Word] = []
        self.last_committed_time = 0.0

    def insert(self, words: list[Word], offset: float) -> None:
        """Add a pass's words, whose timestamps are relative to `offset`."""
        shifted = [
            Word(w.start + offset, w.end + offset, w.text, w.probability) for w in words
        ]
        # Anything starting before what is already settled is stale.
        self._incoming = [w for w in shifted if w.start > self.last_committed_time - 0.1]

        if not self._incoming or not self.committed:
            return

        # Whisper sometimes repeats the last committed words at the head of the
        # next pass. Detect a repeated 1..5-gram and drop it, otherwise the same
        # words get committed twice.
        if abs(self._incoming[0].start - self.last_committed_time) >= 1.0:
            return
        max_n = min(len(self.committed), len(self._incoming), 5)
        for n in range(1, max_n + 1):
            tail = " ".join(_norm(w.text) for w in self.committed[-n:])
            head = " ".join(_norm(w.text) for w in self._incoming[:n])
            # Length is the evidence, not the match. Japanese tokenises into one
            # and two character pieces - た, て, の, し - which recur constantly,
            # so a short match is a coincidence rather than a repeat, and
            # deleting it removes the real first word of the next utterance. It
            # showed up as transcript lines beginning mid-word.
            #
            # Timestamps cannot settle this: the buffer is trimmed to
            # last_committed_time, so a genuine repeat and a genuinely new word
            # both begin at almost exactly that point.
            if tail == head and len(tail.replace(" ", "")) >= _MIN_REPEAT_CHARS:
                del self._incoming[:n]
                break

    def flush(self) -> list[Word]:
        """Commit the longest prefix this pass and the previous one agree on."""
        newly: list[Word] = []
        while self._incoming and self.buffer:
            if _norm(self._incoming[0].text) != _norm(self.buffer[0].text):
                break
            word = self._incoming.pop(0)
            self.buffer.pop(0)
            newly.append(word)

        self.committed.extend(newly)
        if newly:
            self.last_committed_time = newly[-1].end
        self.buffer = self._incoming
        self._incoming = []
        return newly

    def force_commit(self) -> list[Word]:
        """Commit the unconfirmed tail as-is. Used at end of utterance."""
        newly = self.buffer
        self.buffer = []
        self._incoming = []
        if newly:
            self.committed.extend(newly)
            self.last_committed_time = newly[-1].end
        return newly

    @property
    def unstable(self) -> list[Word]:
        return list(self.buffer)


class StreamingTranscriber:
    def __init__(self, cfg: Config, transcriber: Transcriber):
        self.cfg = cfg
        self.asr = transcriber
        self.hypothesis = HypothesisBuffer()
        self._audio = np.empty(0, dtype=np.float32)
        # Absolute stream time of self._audio[0], in seconds.
        self._offset = 0.0

    # -- state -------------------------------------------------------------

    @property
    def buffer_sec(self) -> float:
        return self._audio.size / SAMPLE_RATE

    @property
    def stream_time(self) -> float:
        return self._offset + self.buffer_sec

    def insert_audio(self, chunk: np.ndarray) -> None:
        if chunk.size:
            self._audio = np.concatenate((self._audio, chunk))

    # -- processing --------------------------------------------------------

    def process(self) -> tuple[list[Word], list[Word]]:
        """Run one pass. Returns (newly committed words, unstable tail)."""
        if self.buffer_sec < self.cfg.min_chunk_sec:
            return [], self.hypothesis.unstable

        words = self.asr.transcribe(self._audio)
        self.hypothesis.insert(words, self._offset)
        committed = self.hypothesis.flush()

        if self.buffer_sec > self.cfg.max_buffer_sec:
            self._trim()
        return committed, self.hypothesis.unstable

    def end_utterance(self) -> list[Word]:
        """Silence detected: settle the tail and drop the buffer."""
        committed = self.hypothesis.force_commit()
        self._offset = self.stream_time
        self._audio = np.empty(0, dtype=np.float32)
        return committed

    def drop_backlog(self) -> None:
        """Give up on the current buffer without emitting a partial guess.

        Called when inference cannot keep up; keeping the buffer would make
        captions drift further behind the audio on every pass.
        """
        log.warning("dropping %.1fs of backlog to catch up", self.buffer_sec)
        self.hypothesis.buffer = []
        self._offset = self.stream_time
        self._audio = np.empty(0, dtype=np.float32)

    def _trim(self) -> None:
        """Discard audio behind the last committed word."""
        cut = self.hypothesis.last_committed_time
        # Nothing committed yet in a long buffer: the tail is likely unusable
        # (music, or speech the model won't settle on). Keep the last few
        # seconds so a starting utterance isn't cut mid-word.
        if cut <= self._offset:
            cut = self.stream_time - self.cfg.min_chunk_sec
        keep_from = max(0.0, cut - self._offset)
        samples = int(keep_from * SAMPLE_RATE)
        if samples <= 0:
            return
        self._audio = self._audio[samples:]
        self._offset += samples / SAMPLE_RATE

    # -- display -----------------------------------------------------------

    def display_words(self, history_sec: float) -> tuple[list[Word], list[Word]]:
        """Committed words recent enough to show, plus the unstable tail."""
        cutoff = self.stream_time - history_sec
        committed = [w for w in self.hypothesis.committed if w.end >= cutoff]
        return committed, self.hypothesis.unstable

    def prune_history(self, keep_sec: float, not_before: float = float("inf")) -> None:
        """Drop settled words older than keep_sec, but never past `not_before`.

        The caller passes how far translation has got as `not_before`. Without
        it, a long stretch of unbroken speech can push words out of this list
        before the MT stage ever sees them, and the utterance simply vanishes -
        no source line, no English, nothing to notice it by.
        """
        cutoff = min(self.stream_time - keep_sec, not_before)
        if len(self.hypothesis.committed) > 512:
            self.hypothesis.committed = [
                w for w in self.hypothesis.committed if w.end >= cutoff
            ]


def join_words(words: list[Word]) -> str:
    # Whisper word tokens carry their own leading space.
    return "".join(w.text for w in words).strip()
