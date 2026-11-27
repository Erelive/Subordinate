"""The session's transcript, as data.

Every finished line is kept here - Japanese, English, when it was said, who
said it - so it can be scrolled back through, searched and copied. Nothing is
written to disk: the transcript lives for the session and is gone when the
app closes, unless the user copies or exports it.

It is bounded. A six-hour stream is several thousand lines, and an unbounded
list of them is a slow leak that only shows on exactly the long sessions the
app is for. Past `max_entries` the oldest lines are dropped.

No Qt in this module; ui/transcript.py wraps it in a model.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

from .diarize import UNKNOWN, speaker_label


@dataclass
class Entry:
    id: int
    source: str
    english: str
    speaker: int = UNKNOWN
    wall_time: float = 0.0
    final: bool = True
    engine: str = ""
    latency: float | None = None
    # Corrected by the second look, and what it said before that.
    revised: bool = False
    first_source: str = ""
    first_english: str = ""

    @property
    def clock(self) -> str:
        """Time of day the line was spoken, HH:MM:SS."""
        if not self.wall_time:
            return ""
        return time.strftime("%H:%M:%S", time.localtime(self.wall_time))

    @property
    def primary(self) -> str:
        """The line to read first: the English, or the Japanese if there is none."""
        return self.english or self.source

    def haystack(self) -> str:
        return _fold(f"{self.english}\n{self.source}")


def _fold(text: str) -> str:
    """Case- and width-insensitive form for searching.

    NFKC so that typing ｱﾆﾒ or ＡＢＣ finds アニメ and ABC: the width a
    Japanese IME happens to be in should not decide whether a search works.
    """
    return unicodedata.normalize("NFKC", text).casefold()


class History:
    def __init__(self, max_entries: int = 3000):
        self.max_entries = max(1, int(max_entries))
        self.entries: list[Entry] = []
        self._index: dict[int, Entry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def upsert(self, utterance) -> tuple[int, bool, int]:
        """Add a line, or update the draft of it already here.

        Returns (row, is_new, dropped). `dropped` is how many of the oldest
        lines were removed to stay within the limit - the caller needs it to
        keep a view's row numbers honest.

        A final line is never changed by a later update. The pipeline does
        not send one, but the transcript is where "finished text is not
        rewritten" is a promise to the reader, so it is enforced here too.
        The one exception has its own door: see revise.
        """
        existing = self._index.get(utterance.id)
        if existing is not None:
            row = self.entries.index(existing)
            if existing.final:
                return row, False, 0
            existing.source = utterance.source
            existing.english = utterance.english
            existing.speaker = utterance.speaker
            existing.final = utterance.final
            existing.engine = utterance.engine
            existing.latency = utterance.latency
            return row, False, 0

        entry = Entry(
            id=utterance.id,
            source=utterance.source,
            english=utterance.english,
            speaker=utterance.speaker,
            wall_time=utterance.wall_time,
            final=utterance.final,
            engine=utterance.engine,
            latency=utterance.latency,
        )
        self.entries.append(entry)
        self._index[entry.id] = entry
        dropped = 0
        while len(self.entries) > self.max_entries:
            old = self.entries.pop(0)
            self._index.pop(old.id, None)
            dropped += 1
        return len(self.entries) - 1, True, dropped

    def revise(self, utterance) -> int | None:
        """Apply a second-look correction to a finished line. Returns its row.

        Once per line, and only to a line that is final: a correction to a
        draft is just the draft's next version, and a second correction
        would make the transcript something that keeps moving. None if the
        line is not here any more or does not qualify.
        """
        entry = self._index.get(utterance.id)
        if entry is None or not entry.final or entry.revised:
            return None
        entry.first_source, entry.first_english = entry.source, entry.english
        entry.source = utterance.source
        entry.english = utterance.english
        entry.engine = utterance.engine or entry.engine
        entry.revised = True
        return self.entries.index(entry)

    def clear(self) -> None:
        self.entries.clear()
        self._index.clear()

    def latest(self) -> Entry | None:
        """The most recent line that has a translation, drafts included."""
        for entry in reversed(self.entries):
            if entry.primary:
                return entry
        return None

    def search(self, query: str) -> list[Entry]:
        """Lines containing every word of `query`, in either language."""
        terms = [t for t in _fold(query).split() if t]
        if not terms:
            return list(self.entries)
        return [e for e in self.entries if all(t in e.haystack() for t in terms)]

    @staticmethod
    def matches(entry: Entry, query: str) -> bool:
        terms = [t for t in _fold(query).split() if t]
        if not terms:
            return True
        haystack = entry.haystack()
        return all(t in haystack for t in terms)

    @staticmethod
    def format_entry(
        entry: Entry, source: bool = True, timestamps: bool = False, english: bool = True
    ) -> str:
        head = []
        if timestamps and entry.clock:
            head.append(f"[{entry.clock}]")
        who = speaker_label(entry.speaker)
        if who:
            head.append(f"{who}:")
        prefix = " ".join(head)
        lines = []
        if english and entry.english:
            lines.append(entry.english)
        if (source or not lines) and entry.source:
            lines.append(entry.source)
        if not lines:
            return ""
        if prefix:
            lines[0] = f"{prefix} {lines[0]}"
        return "\n".join(lines)

    def as_text(
        self,
        source: bool = True,
        timestamps: bool = True,
        entries: list[Entry] | None = None,
    ) -> str:
        """The transcript as plain text, one line or pair per utterance."""
        chosen = self.entries if entries is None else entries
        blocks = [self.format_entry(e, source=source, timestamps=timestamps) for e in chosen]
        separator = "\n\n" if source else "\n"
        return separator.join(b for b in blocks if b)
