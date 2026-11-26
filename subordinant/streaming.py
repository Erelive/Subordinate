"""Turning repeated whole-buffer transcriptions into stable streaming captions.

Whisper is not a streaming model: it transcribes a whole buffer at once, and the
answer for a given word can change as more audio arrives. Re-running it every
third of a second and printing the raw output produces text that visibly
rewrites itself.

LocalAgreement-2 (Machacek et al., 2023) is the starting point. Run inference
on the growing buffer; a word is *settled* once two consecutive passes agree on
it, and shown as such, while the disagreeing tail is shown dimmed as a guess.

Where this departs from the paper, and why. There, a settled word is settled
for good. Here it is settled until something better is known, and fixed for
good only once it has been handed on - to the translator, or out of the audio
buffer. The difference is what happens to a wrong early guess. Measured on the
benchmark set, the first two passes over the opening half second of an
utterance agreed on a hallucinated phrase often enough to matter, and under
the paper's rule every later pass - each of which read the same audio
correctly - was discarded as disagreeing with "settled" text. Worse, the
correct readings no longer lined up with the settled one, so the join between
old and new words fell back on timestamps and doubled syllables down the rest
of the line. Live recognition was making nearly twice the errors of the same
model run on the finished audio.

So words live in three places:

  sent   handed to the translator. Never changes.
  held   settled, and no longer revisable because the audio they came from has
         left the buffer. Waiting to be handed on.
  soft   settled for now. Still in the buffer, so every pass re-reads them,
         and if two consecutive passes agree with each other against them,
         they are replaced.

When the utterance ends there is one more reading to take into account: the
last pass, which heard all of it with silence after. That reading replaces the
soft words wholesale - it is the only one made with the whole sentence in view.

Once the buffer grows past its cap the oldest audio is discarded, keeping a
couple of seconds of settled speech at its head as context.
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

# Settled speech kept at the head of the buffer when it is trimmed, as left
# context for the recogniser. See StreamingTranscriber._context_cut.
_TRIM_CONTEXT_SEC = 2.5

# How much of the fixed text is lined up against a new hypothesis to find where
# "already said" ends. Only its tail matters - the question is where it stops.
# The longest is tried first, because a long match cannot be a coincidence;
# the shorter ones are there for when the recogniser has since changed its
# mind about something earlier in the line, which spoils a long match without
# making the last few characters any less findable.
_ANCHOR_CHARS = (24, 12, 6)


def _norm(text: str) -> str:
    """Normalise a word for comparison, ignoring punctuation.

    Whisper attaches a sentence mark only once it has heard enough to decide the
    sentence ended, so the same word arrives as 'します' on one pass and
    'します。' on the next. Comparing those literally makes agreement stall at
    exactly the sentence boundaries - the words whose punctuation matters most.
    """
    stripped = text.strip().lower()
    core = stripped.strip(_PUNCT)
    # A token that is nothing but punctuation still has to compare as itself,
    # or every one of them would normalise to "" and match each other.
    return core or stripped


def _letters(text: str) -> list[tuple[int, str]]:
    """(index into text, character) for the characters that carry content."""
    return [(i, ch.lower()) for i, ch in enumerate(text) if ch.isalnum()]


def _content(words: list[Word]) -> str:
    """The content characters of a run of words, as one string."""
    return "".join(ch for w in words for _i, ch in _letters(w.text))


def _restated(
    fixed: list[Word], hypothesis: list[Word], anchor: float | None = None
) -> list[Word] | None:
    """The part of a hypothesis that comes after text which is already fixed.

    Every pass re-transcribes the whole buffer, so its output restates words
    that have already been handed on before it gets to anything new. The job
    is to find where "already said" ends.

    Timestamps were the first answer - drop whatever starts before the last
    fixed word ended - and they are not good enough. Whisper's word timings
    come from cross-attention and wander by a syllable from pass to pass, so
    the last fixed token regularly reappears with a start time a few tens of
    milliseconds late, passes the filter as new, and is kept again. On
    Japanese, where a token is one or two characters, that showed up as a
    doubled character at nearly every join: フランススの授業業があります.

    Text does not wander. The tail of the fixed text is located in the
    hypothesis by edit distance, and the cut is made where the match ends.
    Only the tail, and anywhere in the hypothesis: the first version of this
    insisted that the whole of the fixed text line up from the hypothesis's
    first character, and one early disagreement - a hallucinated opening the
    recogniser later thought better of - was enough to fail the match and
    send every later join back to the clock.

    The cut is made at a character, not at a word. Whisper's tokenisation
    moves between passes as well: 大事 + だ on one, 大事 + だと on the next. A
    cut restricted to word boundaries has to either repeat that だ or lose
    the と; cutting inside the token keeps exactly the と.

    Where the tail matches equally well in more than one place - a phrase
    said twice - the place nearest `anchor`, the time the fixed text ended,
    is the one meant. Remaining ties go to the later cut: they arise when the
    hypothesis has respelled the end of what was fixed, and the fixed text
    stands regardless, so the respelling is what has to go.

    Returns None when the two do not line up well enough to trust, and the
    caller falls back to timestamps.
    """
    whole = _content(fixed)
    if not whole or not hypothesis:
        return None
    # Every content character of the hypothesis, with the word it is in and
    # where in that word's text it sits.
    owners: list[tuple[int, int]] = []
    chars: list[str] = []
    for index, word in enumerate(hypothesis):
        for raw, ch in _letters(word.text):
            owners.append((index, raw))
            chars.append(ch)
    if not chars:
        return None

    m = len(chars)
    codes = np.fromiter((ord(c) for c in chars), dtype=np.uint32, count=m)
    steps = np.arange(m + 1)
    candidates = None
    tried = 0
    for size in _ANCHOR_CHARS:
        settled = whole[-size:]
        n = len(settled)
        if n == tried:
            break  # the fixed text is shorter than this; already tried it all
        tried = n
        # Starting anywhere in the hypothesis is free; ending is what is scored.
        row = np.zeros(m + 1, dtype=np.int32)
        for i in range(1, n + 1):
            cur = np.empty(m + 1, dtype=np.int32)
            cur[0] = i
            cost = (codes != ord(settled[i - 1])).astype(np.int32)
            cur[1:] = np.minimum(row[1:] + 1, row[:-1] + cost)
            row = np.minimum.accumulate(cur - steps) + steps
        best = int(row.min())
        # A character or two can be found, wrongly, almost anywhere; a tail
        # that short has to match exactly to be believed.
        if best <= (max(1, n // 4) if n >= 4 else 0):
            candidates = np.flatnonzero(row == best)
            break
    if candidates is None:
        return None
    position = int(candidates[-1])
    if anchor is not None and len(candidates) > 1:

        def when(pos: int) -> float:
            """Stream time at which the hypothesis reaches character `pos`."""
            if pos >= m:
                return hypothesis[-1].end
            index, _raw = owners[pos]
            return hypothesis[index].start

        nearest = min(abs(when(int(pos)) - anchor) for pos in candidates)
        position = max(
            int(pos) for pos in candidates if abs(when(int(pos)) - anchor) <= nearest + 0.05
        )
    if position >= m:
        return []  # the hypothesis says nothing the fixed text does not
    if position == 0:
        return None  # "matched" by deleting the whole tail: not a match

    index, raw = owners[position]
    word = hypothesis[index]
    first_raw = _letters(word.text)[0][0]
    if raw == first_raw:
        return list(hypothesis[index:])
    # The cut falls inside this token. Keep its tail, and move its start time
    # forward by the share of the token that was already said.
    share = raw / max(1, len(word.text))
    start = word.start + (word.end - word.start) * share
    tail = Word(start, word.end, word.text[raw:], word.probability)
    return [tail] + list(hypothesis[index + 1 :])


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two short strings."""
    if not a or not b:
        return len(a) + len(b)
    codes = np.fromiter((ord(c) for c in b), dtype=np.uint32, count=len(b))
    steps = np.arange(len(b) + 1)
    row = steps.astype(np.int32)
    for i, ch in enumerate(a, start=1):
        cur = np.empty(len(b) + 1, dtype=np.int32)
        cur[0] = i
        cost = (codes != ord(ch)).astype(np.int32)
        cur[1:] = np.minimum(row[1:] + 1, row[:-1] + cost)
        row = np.minimum.accumulate(cur - steps) + steps
    return int(row[-1])


def _agreed(previous: list[Word], newest: list[Word], settled_before: float) -> list[Word]:
    """The leading words of `newest` that `previous` said too.

    Compared character by character rather than token by token, because the
    same text is cut into different tokens from one pass to the next, and
    with punctuation ignored, because it comes and goes. Only whole tokens of
    the newest pass are returned.

    ...and only those ending before `settled_before`. Two passes a third of a
    second apart can agree about a word and both be wrong, when the word is
    the one the buffer ends in the middle of: each has heard the same first
    half and makes the same guess at the rest. Agreement is only evidence
    once the audio has moved on past the word.
    """
    a, b = _content(previous), _content(newest)
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    out: list[Word] = []
    used = 0
    for word in newest:
        size = len(_letters(word.text))
        if used + size > common or word.end > settled_before:
            break
        out.append(word)
        used += size
    # Marks alone are not agreement about anything.
    return out if used else []


class HypothesisBuffer:
    """What has been recognised so far, and how sure of it we are."""

    def __init__(self) -> None:
        self.sent: list[Word] = []  # handed on; never changes
        self.held: list[Word] = []  # settled and fixed, not yet handed on
        self.soft: list[Word] = []  # settled for now; a later pass may revise
        self._unstable: list[Word] = []  # the newest pass's guess at the rest
        # Everything the two most recent passes said, on the stream clock.
        self.last_pass: list[Word] = []
        self._prev_pass: list[Word] = []
        self._offset = 0.0  # stream time the newest pass's buffer began at
        # Nothing before this moment may be taken from a hypothesis: set when
        # audio is thrown away, so text is not matched against words that
        # were never recognised.
        self._floor = 0.0
        self.revisions = 0  # times settled words were replaced

    # -- views -------------------------------------------------------------

    @property
    def committed(self) -> list[Word]:
        """Every settled word, oldest first."""
        return self.sent + self.held + self.soft

    @property
    def pending(self) -> list[Word]:
        """Settled words not yet handed on, oldest first."""
        return self.held + self.soft

    @property
    def unstable(self) -> list[Word]:
        return list(self._unstable)

    @property
    def last_committed_time(self) -> float:
        for words in (self.soft, self.held, self.sent):
            if words:
                return max(words[-1].end, self._floor)
        return self._floor

    @last_committed_time.setter
    def last_committed_time(self, value: float) -> None:
        self._floor = value

    # -- one pass ----------------------------------------------------------

    def insert(self, words: list[Word], offset: float) -> None:
        """Add a pass's words, whose timestamps are relative to `offset`."""
        shifted = [
            Word(w.start + offset, w.end + offset, w.text, w.probability) for w in words
        ]
        self._prev_pass = self.last_pass
        self.last_pass = shifted
        self._offset = offset
        # Soft words whose audio has left the buffer can no longer be re-read.
        while self.soft and self.soft[0].start < offset - 0.05:
            self.held.append(self.soft.pop(0))

    def _beyond_fixed(self, hypothesis: list[Word]) -> list[Word]:
        """The part of a hypothesis after the words that can no longer change."""
        fixed = self.sent + self.held
        if not fixed:
            return [w for w in hypothesis if w.start > self._floor - 0.1]
        fixed_end = max(fixed[-1].end, self._floor)
        # Fixed words whose audio is still in the buffer. The hypothesis has
        # to restate these before it says anything new.
        covered = [w for w in fixed if w.start >= self._offset - 0.05]
        if covered and covered[-1].end >= self._floor:
            fresh = _restated(covered, hypothesis, anchor=covered[-1].end)
            if fresh is not None:
                return fresh

        # Nothing fixed is in the buffer - a new utterance, or audio was
        # dropped - or the text did not line up. Fall back to the clock:
        # anything starting before what is already fixed is stale.
        if not covered:
            # ...where "already fixed" cannot reach into the buffer. A word's
            # timing is routinely stretched across the pause that follows it,
            # and a buffer that was cut in that pause begins before the time
            # the word claims to have ended at. Taken at its word, that would
            # make the first thing said after the pause look stale.
            fixed_end = min(fixed_end, max(self._offset, self._floor))
        fresh = [w for w in hypothesis if w.start > fixed_end - 0.1]
        if not fresh or abs(fresh[0].start - fixed_end) >= 1.0:
            return fresh
        # Whisper sometimes repeats the last words at the head of the next
        # pass. Detect a repeated 1..5-gram and drop it, otherwise the same
        # words are kept twice.
        max_n = min(len(fixed), len(fresh), 5)
        for n in range(1, max_n + 1):
            tail = " ".join(_norm(w.text) for w in fixed[-n:])
            head = " ".join(_norm(w.text) for w in fresh[:n])
            # Length is the evidence, not the match. Japanese tokenises into one
            # and two character pieces - た, て, の, し - which recur constantly,
            # so a short match is a coincidence rather than a repeat, and
            # deleting it removes the real first word of the next utterance. It
            # showed up as transcript lines beginning mid-word.
            if tail == head and len(tail.replace(" ", "")) >= _MIN_REPEAT_CHARS:
                del fresh[:n]
                break
        return fresh

    def flush(self, settled_before: float = float("inf")) -> list[Word]:
        """Work out what is settled after the pass just inserted.

        Returns the words that became settled for the first time, which is
        what a caller measuring delay wants; the settled text itself is in
        `pending`.

        The settled words are whatever the last two passes agree on, from
        the end of the fixed text onward, with three cases:

          - they agree on at least what was settled: it grows, or stays.
          - they agree on less than was settled, and do not contradict it:
            it stays. The newer passes are simply less sure of the end.
          - they agree with each other and contradict what was settled: they
            win. Two readings made with more audio outvote one made with
            less. Settled words from before the point where the new reading
            begins are kept - a reading that starts late has said nothing
            about them, and recognisers do drop the start of a buffer.
        """
        fresh = self._beyond_fixed(self.last_pass)
        previous = self._beyond_fixed(self._prev_pass) if self._prev_pass else []
        agreed = _agreed(previous, fresh, settled_before)
        old = self.soft
        new_text, old_text = _content(agreed), _content(old)

        newly: list[Word] = []
        if new_text.startswith(old_text):
            if len(new_text) > len(old_text) or not old:
                skipped = 0
                for word in agreed:
                    if skipped >= len(old_text):
                        newly.append(word)
                    skipped += len(_letters(word.text))
            self.soft = agreed if agreed else old
            rest = fresh[len(agreed) :] if agreed else fresh
        elif old_text.startswith(new_text):
            rest = _restated(old, fresh, anchor=old[-1].end)
            if rest is None:
                rest = [w for w in fresh if w.start > old[-1].end - 0.1]
        else:
            # Old words that lie mostly before the point the new reading
            # starts at are not contradicted by it - if it really does start
            # late. A reading that begins within a syllable of where the old
            # one did is a re-reading of the same audio, timed a little
            # differently, and keeping the old first token in front of it
            # leaves a stray character (カ + この大学).
            begins = agreed[0].start
            kept = []
            if begins - old[0].start >= 0.3:
                for word in old:
                    if (word.start + word.end) / 2.0 > begins:
                        break
                    kept.append(word)
            self.soft = kept + agreed
            self.revisions += 1
            newly = list(agreed)
            rest = fresh[len(agreed) :]
        self._unstable = rest
        return newly

    def settle_all(self, before: float = float("inf")) -> list[Word]:
        """End of utterance: settle everything, on the last pass's reading.

        Returns the words that had not been settled before.

        `before` limits it to the words that lie ahead of that moment, for
        when an utterance is being split at a pause rather than ended: what
        was said before the pause is finished with, what follows is not.

        The soft words were agreed a few at a time while the utterance was
        still being spoken, each on the strength of the fraction of a second
        that followed it. The last pass read the same audio once, start to
        finish, with the silence after it. That reading is taken whole.

        Unless it is plainly broken. A pass that comes back far shorter than
        what was settled has dropped speech, and one that comes back far
        longer has looped; in either case the settled words, which two passes
        agreed on, stand, with the unconfirmed tail added as it is.
        """
        def ahead(words: list[Word]) -> tuple[list[Word], list[Word]]:
            cut = len(words)
            while cut > 0 and (words[cut - 1].start + words[cut - 1].end) / 2.0 >= before:
                cut -= 1
            return words[:cut], words[cut:]

        soft, soft_later = ahead(self.soft)
        tail, tail_later = ahead(self._unstable) if not soft_later else ([], self._unstable)
        chosen = soft + tail
        later = soft_later + tail_later
        whole = self._beyond_fixed(self.last_pass) if self.last_pass else []
        if whole:
            whole, whole_later = ahead(whole)
            was, now = _content(chosen), _content(whole)
            short = len(now) < 0.6 * len(_content(soft))
            runaway = len(now) > 2 * len(was) + 12
            if not short and not runaway:
                if now != was:
                    self.revisions += 1
                chosen, later = whole, whole_later
        self.held.extend(chosen)
        self.soft = []
        self._unstable = later
        if before == float("inf"):
            self.last_pass = []
            self._prev_pass = []
        return tail

    # The name the first version had, kept for callers and tests.
    force_commit = settle_all

    # -- handing on --------------------------------------------------------

    def take(self, count: int) -> list[Word]:
        """Hand on the first `count` pending words. They never change again."""
        from_held = min(count, len(self.held))
        from_soft = min(count - from_held, len(self.soft))
        chunk = self.held[:from_held] + self.soft[:from_soft]
        del self.held[:from_held]
        del self.soft[:from_soft]
        self.sent.extend(chunk)
        return chunk

    def harden(self) -> None:
        """Fix the settled words as they are: their audio is being discarded."""
        self.held.extend(self.soft)
        self.soft = []

    def drop_tail(self, after: float) -> list[Word]:
        """Remove trailing pending words that begin at or after `after`."""
        dropped: list[Word] = []
        for words in (self.soft, self.held):
            while words and words[-1].start >= after:
                dropped.insert(0, words.pop())
            if words:
                break
        return dropped

    def discard_unstable(self) -> None:
        self._unstable = []
        self.last_pass = []
        self._prev_pass = []


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

    @property
    def buffer_start(self) -> float:
        """Stream time of the oldest audio still in the buffer."""
        return self._offset

    @property
    def pending(self) -> list[Word]:
        """Settled words not yet handed on."""
        return self.hypothesis.pending

    def insert_audio(self, chunk: np.ndarray) -> None:
        if chunk.size:
            self._audio = np.concatenate((self._audio, chunk))

    # -- processing --------------------------------------------------------

    def process(
        self, force: bool = False, quiet_from: float | None = None
    ) -> tuple[list[Word], list[Word]]:
        """Run one pass. Returns (newly settled words, unstable tail).

        force runs it however little audio there is. That is for the last
        pass of an utterance: a one-word reply is shorter than min_chunk_sec,
        and waiting for more of it means never transcribing it at all.

        quiet_from is when the speaker stopped, if they are pausing now.
        Words are normally held back until the audio has run on past them,
        in case the buffer ends in the middle of one; a word followed by
        silence is not in the middle of anything, and need not wait.
        """
        if self.buffer_sec < self.cfg.min_chunk_sec and not force:
            return [], self.hypothesis.unstable
        if self._audio.size == 0:
            return [], self.hypothesis.unstable

        words = self.asr.transcribe(self._audio)
        self.hypothesis.insert(words, self._offset)
        settled_before = self.stream_time - self.cfg.commit_holdback_sec
        if quiet_from is not None:
            # The buffer ends in silence, not in the middle of a word. Word
            # timings stretch the last word some way into that silence, so
            # the limit is lifted rather than moved to where the voice
            # stopped.
            settled_before = float("inf")
        settled = self.hypothesis.flush(settled_before)

        if self.buffer_sec > self.cfg.max_buffer_sec:
            self._trim()
        return settled, self.hypothesis.unstable

    def end_utterance(self) -> list[Word]:
        """The speaker has stopped: settle everything and drop the buffer."""
        settled = self.hypothesis.settle_all()
        self._offset = self.stream_time
        self._audio = np.empty(0, dtype=np.float32)
        return settled

    def take(self, count: int) -> list[Word]:
        """Hand on the first `count` pending words."""
        return self.hypothesis.take(count)

    def split(self, at: float) -> list[Word]:
        """End the utterance as far as `at`, and keep listening to the rest.

        Everything recognised before that moment is settled on the last
        pass's reading and its audio is dropped; the audio from there on
        stays, as the start of a fresh buffer. `at` should be inside a pause.

        Why the audio has to go: the Japanese recognisers this app uses were
        trained on single sentences, and given a buffer holding a finished
        sentence, a pause, and the start of another, they transcribe the
        first and stop. The second is not misheard, it is not read at all -
        measured on the benchmark's continuous-speech sample, for three
        seconds at a time, until the model flipped to reading only the
        second and the opening words of it were lost for good. A buffer
        that starts at the pause has nothing in it to stop at.
        """
        at = min(max(at, self._offset), self.stream_time)
        settled = self.hypothesis.settle_all(before=at)
        samples = int((at - self._offset) * SAMPLE_RATE)
        if samples > 0:
            self._audio = self._audio[samples:]
            self._offset += samples / SAMPLE_RATE
        return settled

    def drop_backlog(self) -> None:
        """Give up on the current buffer without emitting a partial guess.

        Called when inference cannot keep up; keeping the buffer would make
        captions drift further behind the audio on every pass.
        """
        log.warning("dropping %.1fs of backlog to catch up", self.buffer_sec)
        self.discard_audio()

    def discard_audio(self) -> None:
        """Drop the buffer and every unconfirmed guess; keep what is settled.

        Settled words not yet handed on stay where they are: they were heard
        and agreed on, and whatever went wrong with the audio since - a pass
        that failed, a click that was not speech - is no reason to lose them.
        """
        self.hypothesis.harden()
        self.hypothesis.discard_unstable()
        # Anything after this point starts on fresh audio, so the next words
        # must not be compared against where the discarded ones would have
        # ended.
        self.hypothesis.last_committed_time = self.stream_time
        self._offset = self.stream_time
        self._audio = np.empty(0, dtype=np.float32)

    def _context_cut(self) -> float:
        """Where to cut a full buffer so it still opens on settled speech.

        The buffer used to be cut at the end of the last settled word, which
        left the recogniser starting cold on whatever came next: no left
        context, a first word clipped by however far the word timing was off,
        and nothing settled left in the buffer to line the next hypothesis up
        against - so the join was made by timestamps, and showed as doubled
        syllables (ニュースス) and words split across lines (台 | 風).

        This keeps the last couple of seconds of settled speech in the
        buffer. The recogniser restates them at the head of every pass, they
        are matched by text and discarded, and what follows is heard in
        context. The cut is placed in the widest gap between words near the
        target, so the buffer begins at a breath where there is one.
        """
        settled_end = self.hypothesis.last_committed_time
        keep = min(_TRIM_CONTEXT_SEC, self.cfg.max_buffer_sec / 3.0)
        target = settled_end - keep
        if target <= self._offset + 0.2:
            # Too little is settled to keep that much. While the buffer is
            # only a little over, leave it; well over, cut at the settled
            # text as before rather than let every pass grow without limit.
            if self.buffer_sec <= self.cfg.max_buffer_sec * 1.5:
                return self._offset
            return settled_end
        words = [w for w in self.hypothesis.committed if w.start > self._offset]
        best, best_gap = target, -1.0
        for prev, word in zip(words, words[1:]):
            if abs(word.start - target) > 1.0:
                continue
            gap = word.start - prev.end
            if gap > best_gap + 0.02 or (
                abs(gap - best_gap) <= 0.02 and abs(word.start - target) < abs(best - target)
            ):
                best_gap = gap
                best = (prev.end + word.start) / 2.0 if gap > 0 else word.start
        return best

    def _trim(self) -> None:
        """Discard old audio, keeping a little settled speech as context."""
        cut = self.hypothesis.last_committed_time
        # Nothing settled yet in a long buffer: the tail is likely unusable
        # (music, or speech the model won't settle on). Keep the last few
        # seconds so a starting utterance isn't cut mid-word.
        unsettled = cut <= self._offset
        if unsettled:
            cut = self.stream_time - self.cfg.min_chunk_sec
        else:
            cut = self._context_cut()
        keep_from = max(0.0, cut - self._offset)
        samples = int(keep_from * SAMPLE_RATE)
        if samples <= 0:
            return
        self._audio = self._audio[samples:]
        self._offset += samples / SAMPLE_RATE
        if unsettled:
            # Guesses about the audio just discarded are not to be matched
            # against readings of what is left.
            self.hypothesis.last_committed_time = self._offset
        # What is settled now stays as it is. The words kept at the head of
        # the buffer are there to be context, not to be read again: the
        # buffer may begin partway through one of them, and a re-reading
        # that starts with half a word is not an improvement on the text.
        self.hypothesis.harden()

    # -- display -----------------------------------------------------------

    def display_words(self, history_sec: float) -> tuple[list[Word], list[Word]]:
        """Settled words recent enough to show, plus the unstable tail."""
        cutoff = self.stream_time - history_sec
        committed = [w for w in self.hypothesis.committed if w.end >= cutoff]
        return committed, self.hypothesis.unstable

    def prune_history(self, keep_sec: float, not_before: float = float("inf")) -> None:
        """Forget words handed on more than keep_sec ago.

        Only words that have been handed on are ever dropped: anything still
        waiting for the translator stays however old it is, or a long stretch
        of unbroken speech could push words out before they were translated,
        and the utterance would simply vanish.
        """
        cutoff = min(self.stream_time - keep_sec, not_before)
        sent = self.hypothesis.sent
        # Cheap test first: this is called after every utterance, and nearly
        # always there is nothing old enough to drop.
        if sent and sent[0].end < cutoff:
            self.hypothesis.sent = [w for w in sent if w.end >= cutoff]

    def reset(self) -> None:
        """Forget everything. Used after a pause or a recovered failure."""
        self.hypothesis = HypothesisBuffer()
        self._offset = self.stream_time
        self._audio = np.empty(0, dtype=np.float32)


def join_words(words: list[Word]) -> str:
    # Whisper word tokens carry their own leading space.
    return "".join(w.text for w in words).strip()
