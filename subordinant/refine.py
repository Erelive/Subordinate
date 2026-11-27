"""The second look: one finished line, read again once more is known.

Every line in the transcript is translated the moment its speaker stops,
which is before the next line exists. A few seconds later more is known, and
the line can be corrected in place, once.

What is done with that, by default, needs no model:

  A line that stops on のに, けど, から or ので, and is followed at once by
  another from the same speaker, was the first half of a sentence. Its
  translation reads as a whole one, because the translator was not told
  otherwise. Its ending is reopened - "..., but" - so that the two lines
  read as the one sentence they were. (join_to_next)

That is all, and this module is mostly a record of why. The idea was larger:
show the line to an instruction model again, between its neighbours, and let
it repair misheard Japanese and re-translate with hindsight. Each form of
that was built and measured (bench/refine_bench.py, BENCHMARKS.md) with the
two models to hand, a 14B Japanese-to-English specialist and a 4B general
one:

  - Asked for the line back, Japanese and English, the model rewords English
    that was fine and "corrects" Japanese by borrowing a word from the next
    line. Net effect on real recogniser output: none, for most of a second
    per line that first translations then queued behind.

  - Asked instead to name misheard words as pairs, and held to pairs that
    are one kanji exchanged for another (the shape a homophone error has),
    it fixes the errors planted in a test script and, on real output, turns
    歯が痛い into 天が痛い. More characters wrong after than before.

  - Offered "this was a question" and "the subject was someone else" as
    verdicts, it declared three plain statements in twenty-eight lines to
    be questions. Nothing in the text can check either claim.

  - Asked only how to reopen the ending of a line already known to run on,
    it does it - but then all it is choosing is "but" or "so", which the
    particle the line ends on already says.

The model path is still here, behind cfg.refine_fix_words, because what
failed is these models and not the approach: the guards (judge) keep it
from doing worse than a few wrong kanji, and the benchmark says in three
minutes whether a different model does better. It is off, and nothing in
the app turns it on.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import requests

from . import glossary
from .config import Config
from .llm_translate import (
    _LANG_NAME,
    _is_local,
    find_ollama,
    has_model,
    server_reachable,
    start_server,
)
from .translate import TranslationUnavailable

log = logging.getLogger(__name__)

_SYSTEM = """You proofread one line of a live {src}-to-English subtitle track, a few seconds after it was shown.

The {src} text came from speech recognition. Now and then a word in it is misheard as a different word with exactly the same pronunciation, written with a different kanji. Its first translation was written before the following lines had been spoken.

You are given the lines before it, the line itself, and the lines after it. Most lines need nothing. Reply with a JSON object with three fields, in this order:

"check": one of
  "fine" - the first translation stands. This is the usual answer.
  "continues" - the line stops mid-sentence (on のに, けど, から, て, で, と and the like), the next line finishes the thought, and the first translation reads as a finished sentence.
  "misheard" - a word in it was misheard, and the first translation follows the wrong word.

"fixes": the misheard words, if any, as a list of {{"heard": the word as it is written in the line, "said": the word that was said}}. Only where the word as written makes no sense in this line and a word pronounced exactly the same does. A word that makes sense as written is never a fix, and neither is a word taken from another line. Usually an empty list.

"english": an empty string if the check is "fine". If it is "continues", the first translation with only its ending changed so that it stays open - a comma, or a joining word such as "but" or "so" - and reads on into the next line. If it is "misheard", a natural English subtitle for the corrected line, and only that line. {honorifics} Spell names as the earlier English lines do. Never include what the lines before or after say."""

_HONORIFICS_KEEP = "Keep honorifics attached to names (-san, -chan, -kun, senpai)."
_HONORIFICS_DROP = "Drop honorifics or use a natural English form of address."

_NAMES = """

Names and terms used in this audio. Use these spellings, and these readings for any kanji given one:
{terms}"""

# "question" and "subject" were verdicts too, and are the reason hindsight was
# wanted in the first place. They are gone because the models here cannot be
# trusted with them: offered the choice, the 14B declared three plain
# statements in twenty-eight lines to be questions and put a question mark on
# each, and swapped a subject that was right. What is left is what could be
# checked - see judge.
CHECKS = ("fine", "continues", "misheard")

# Field order matters: Ollama constrains the reply to this shape in this
# order, so the model commits to what the later lines showed before it writes
# anything else - and when that is "fine", there is almost nothing else.
_SCHEMA = {
    "type": "object",
    "properties": {
        "check": {"type": "string", "enum": list(CHECKS)},
        "fixes": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"heard": {"type": "string"}, "said": {"type": "string"}},
                "required": ["heard", "said"],
            },
        },
        "english": {"type": "string"},
    },
    "required": ["check", "fixes", "english"],
}

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class Line:
    """One transcript line as the second look sees it."""

    source: str
    english: str = ""


@dataclass
class Reply:
    check: str = "fine"
    fixes: list[tuple[str, str]] = field(default_factory=list)
    english: str = ""


@dataclass
class Revision:
    source: str
    english: str
    source_changed: bool
    english_changed: bool


def _letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _is_kanji(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def is_sound_alike_swap(heard: str, said: str) -> bool:
    """Is this the shape of a homophone error: one kanji exchanged, nothing else?

    There is no pronunciation dictionary here to ask whether two words sound
    the same, so the test is on the writing. A recogniser that mishears a
    word keeps its sound, and in Japanese that nearly always means the same
    kana around a different kanji: 以外/意外, 書いて/描いて, 機会/機械. A model
    that is "correcting" from context instead swaps in a different word
    altogether - other kana, another length, katakana for kanji - and that
    is exactly what this refuses. It also refuses real mishearings that
    change two characters; those are left as heard, which is the safe side.
    """
    if heard == said or len(heard) != len(said) or not 1 <= len(heard) <= 12:
        return False
    changed = [(a, b) for a, b in zip(heard, said) if a != b]
    return len(changed) == 1 and _is_kanji(changed[0][0]) and _is_kanji(changed[0][1])


def apply_fixes(source: str, fixes: list[tuple[str, str]]) -> str:
    """`source` with each acceptable fix made. Anything doubtful is skipped."""
    for heard, said in fixes:
        # Once, or not at all: a word that occurs twice gives no way to tell
        # which one was meant.
        if is_sound_alike_swap(heard, said) and source.count(heard) == 1:
            source = source.replace(heard, said)
    return source


def english_is_plausible(english: str, first: str, source: str) -> bool:
    """Is this a subtitle for this line, rather than a note or two lines' worth?"""
    english = english.strip()
    if not _letters(english):
        return False
    # Japanese left in the English is the model quoting instead of translating.
    if len(_CJK.findall(english)) > 2:
        return False
    # English runs about three characters to a Japanese one. Half as long
    # again as the first translation, or several times what the line could
    # carry, means the neighbours were translated into it.
    ceiling = 6 * max(4, len(_letters(source))) + 24
    if first.strip():
        ceiling = min(ceiling, int(1.5 * len(first)) + 15)
    return len(english) <= ceiling


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) >= 4}


def borrows_from(english: str, first: str, neighbours: list[str]) -> bool:
    """Is this another line's translation, put here?

    The commonest way a second look goes wrong: shown three lines and asked
    about the middle one, the model answers with the one after it. That reads
    as perfectly good English of a plausible length, so it is caught by what
    it is made of - mostly a neighbour's words, and more of theirs than of
    this line's own first translation.
    """
    words = _content_words(english)
    if len(words) < 2:
        return False
    own = len(words & _content_words(first)) / len(words)
    for other in neighbours:
        shared = len(words & _content_words(other)) / len(words)
        if shared >= 0.5 and shared > own:
            return True
    return False


def same_text(a: str, b: str) -> bool:
    """Equal as a reader would see it: case, spacing and a final stop aside."""
    return a.strip().rstrip(".!").lower() == b.strip().rstrip(".!").lower()


def build_request(line: Line, before: list[Line], after: list[Line], notes: list[str]) -> str:
    """The user message: the line between its neighbours."""
    parts: list[str] = []
    if before:
        parts.append("Lines before:")
        for item in before:
            parts.append(f"- {item.source}")
            if item.english:
                parts.append(f"  ({item.english})")
    parts.append("The line to check:")
    parts.append(f"- {line.source}")
    if line.english:
        parts.append(f"  First translation: {line.english}")
    else:
        parts.append("  First translation: (none - write the English whatever the check)")
    if after:
        parts.append("Lines after:")
        parts.extend(f"- {item.source}" for item in after)
    else:
        parts.append("Lines after: (nothing has been said since)")
    if notes:
        parts.append("Term notes for the line (stream and net slang):")
        parts.extend(f"- {note}" for note in notes)
    return "\n".join(parts)


def parse_reply(content: str) -> Reply | None:
    """The model's reply, or None if it is not one."""
    content = content.strip()
    try:
        data = json.loads(content)
    except ValueError:
        match = _JSON_OBJECT.search(content)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    english = data.get("english", "")
    if not isinstance(english, str):
        return None
    check = data.get("check")
    fixes = []
    raw = data.get("fixes")
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            heard, said = item.get("heard"), item.get("said")
            if isinstance(heard, str) and isinstance(said, str):
                fixes.append((heard.strip(), said.strip()))
    return Reply(check if check in CHECKS else "fine", fixes, " ".join(english.split()))


# How a line that runs on ends, and the English word that carries it over.
# Longest first. Only the endings that leave a clause hanging on purpose: て
# and で end half of all spoken sentences, finished or not.
_RUNS_ON = (
    ("けれども", "but"), ("けれど", "but"), ("けども", "but"), ("のに", "but"),
    ("けど", "but"), ("ですが", "but"), ("ますが", "but"), ("んだが", "but"),
    ("ので", "so"), ("から", "so"),
)
_TRAILING = " \u3000。、，,.…・ー〜~！!"
# English that already says how it joins what follows. Adding "but" after
# "though" is how "took 12 hours though, but" happens.
_JOINED = re.compile(
    r"\b(but|though|although|however|so|because|since|if|when|while|until|and)\b", re.I
)


def runs_on(source: str) -> str | None:
    """The joining word for a line that stops mid-sentence, or None.

    Decided from the line's own last word. A question, or a line too short
    to be a clause, does not run on whatever it ends with.
    """
    text = source.strip()
    if text.endswith(("?", "？")):
        return None
    core = text.rstrip(_TRAILING)
    if len(_letters(core)) < 6:
        return None
    for ending, word in _RUNS_ON:
        if core.endswith(ending):
            return word
    return None


def join_to_next(line: Line) -> str | None:
    """The line's English with its ending reopened, or None to leave it.

    Only a translation that closes with a full stop is touched: one that
    ends on a comma, a dash or an ellipsis is open already, and a question
    or an exclamation is not something to append "but" to.
    """
    word = runs_on(line.source)
    english = line.english.strip()
    if word is None or len(english.split()) < 3:
        return None
    if not english.endswith(".") or english.endswith(".."):
        return None
    stem = english[:-1].rstrip()
    if _JOINED.search(stem):
        return stem + ","
    return f"{stem}, {word}"


def opens_the_ending(english: str, first: str) -> bool:
    """Is `english` the first translation with only its ending left open?"""
    stem = first.strip().rstrip(".!?,;- ").lower()
    new = english.strip().lower()
    if not stem or not new.startswith(stem):
        return False
    tail = new[len(stem):].strip(" ,.-\u2014\u2026")
    return tail in ("", "but", "so", "and", "then", "though", "because", "when", "if")


def judge(
    line: Line,
    reply: Reply,
    neighbours: list[str] | None = None,
    runs_on: bool = False,
    fix_words: bool = False,
) -> tuple[Revision | None, str]:
    """Decide what, if anything, of a reply is kept. Returns (revision, verdict).

    The verdict is "changed", "same" or "bad-english" - the last being a
    reply whose English was thrown away with nothing else to keep, which is
    worth counting: a model that is thrown away half the time is the wrong
    model for this.

    `runs_on` is the caller's own finding that the Japanese stops on a
    dangling particle and another line follows. The model's word for it is
    not taken: asked, it finds half of all lines unfinished.

    The English is only allowed to change for a reason. Asked to look at a
    line again, a model rewords it - "she'd come" becomes "she'd be coming" -
    and a caption that changes under the reader for no gain is worse than
    one that stays put. A line with no English yet is the exception:
    anything plausible beats nothing.
    """
    source = apply_fixes(line.source, reply.fixes) if fix_words else line.source
    source_changed = source != line.source
    first = line.english
    english = first
    rejected = False
    if reply.english and not same_text(reply.english, first):
        if not first.strip() or source_changed:
            wanted = True
        elif reply.check == "continues" and runs_on:
            wanted = opens_the_ending(reply.english, first)
        else:
            wanted = False
        if wanted:
            if english_is_plausible(reply.english, first, line.source) and not borrows_from(
                reply.english, first, neighbours or []
            ):
                english = reply.english
            else:
                rejected = True
    english_changed = english != first
    if not source_changed and not english_changed:
        return None, "bad-english" if rejected else "same"
    return Revision(source, english, source_changed, english_changed), "changed"


def worth_a_look(line: Line, has_next: bool) -> bool:
    """Could the model path change this line at all?

    It can supply a translation where there was none, fix a kanji, or open
    the ending of a line that runs on, so a line that offers none of those
    is not sent. Each call is time the first translations cannot use.
    """
    if not line.english.strip():
        return True
    return any(_is_kanji(ch) for ch in line.source) or (
        has_next and runs_on(line.source) is not None
    )


class Refiner:
    """Gives lines their second look. One instance per run.

    By default that is join_to_next and nothing else, and no model is
    involved. With cfg.refine_fix_words it asks the model, and `judge`
    decides what to keep of the answer.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.refine_model.strip() or cfg.llm_model
        self.last_error: str | None = None
        self.last_verdict = ""
        self._url = cfg.llm_endpoint.rstrip("/")
        self._system = _SYSTEM.format(
            src=_LANG_NAME.get(cfg.mt_source_lang, cfg.mt_source_lang),
            honorifics=_HONORIFICS_KEEP if cfg.llm_keep_honorifics else _HONORIFICS_DROP,
        )
        terms = cfg.initial_prompt.strip()
        if terms:
            self._system += _NAMES.format(terms=terms)
        self._glossary = glossary.default() if cfg.slang_glossary else None
        self._fix_words = cfg.refine_fix_words
        self._session = requests.Session()
        if self._fix_words:
            self._check_server()

    @property
    def uses_model(self) -> bool:
        return self._fix_words

    def _check_server(self) -> None:
        if not server_reachable(self._url) and not start_server(self._url):
            if _is_local(self._url) and find_ollama() is None:
                raise TranslationUnavailable(
                    "it runs through Ollama, which is not installed (ollama.com)."
                )
            raise TranslationUnavailable(f"Ollama is not answering at {self._url}.")
        if not has_model(self._url, self.model):
            raise TranslationUnavailable(
                f"the model {self.model} has not been downloaded (ollama pull {self.model})."
            )

    def warmup(self) -> None:
        """Load the weights now if this is not the model already translating."""
        try:
            self._ask(build_request(Line("こんにちは", "Hello."), [], [], []), 180.0)
        except Exception as exc:  # noqa: BLE001 - a slow warmup must not block the run
            log.warning("second-look warmup failed, continuing: %s", exc)

    def close(self) -> None:
        self._session.close()

    def notes_for(self, line: Line) -> list[str]:
        """Term notes for the line being checked, and only for it.

        The neighbours' terms were tried too, and were applied: with the next
        line's note about お披露目 in view, the model wrote お披露目 into this
        one. A note is an instruction, and it is carried out wherever it is
        seen.
        """
        if self._glossary is None:
            return []
        return self._glossary.notes(line.source)

    def _ask(self, request: str, timeout: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": request},
            ],
            "stream": False,
            "format": _SCHEMA,
            "keep_alive": self.cfg.llm_keep_alive,
            "options": {
                # Zero, where the first pass uses a little: this is a check,
                # and a check that answers differently each time is not one.
                "temperature": 0.0,
                "num_predict": max(160, self.cfg.llm_max_tokens * 2),
                "num_ctx": self.cfg.llm_num_ctx,
                "seed": 7,
            },
        }
        resp = self._session.post(f"{self._url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content") or ""

    def refine(
        self,
        line: Line,
        before: list[Line],
        after: list[Line],
        follows_on: bool = True,
    ) -> Revision | None:
        """A revision of `line`, or None when it stands as it is.

        `follows_on` is whether the next line came straight after this one
        from the same speaker - without that, a line ending in けど is just
        a line ending in けど.

        Never raises. A failure sets last_error and returns None; last_verdict
        says which kind of None it was.
        """
        self.last_error = None
        self.last_verdict = ""
        if not line.source.strip():
            return None
        joined = join_to_next(line) if after and follows_on else None
        if not self._fix_words:
            if joined is None:
                self.last_verdict = "skipped"
                return None
            self.last_verdict = "changed"
            return Revision(line.source, joined, False, True)
        if not worth_a_look(line, bool(after)):
            self.last_verdict = "skipped"
            return None
        request = build_request(line, before, after, self.notes_for(line))
        t0 = time.perf_counter()
        try:
            content = self._ask(request, self.cfg.llm_timeout_sec)
        except requests.Timeout:
            self.last_error = "The second-look model took too long to answer."
            self.last_verdict = "error"
            return None
        except requests.RequestException as exc:
            self.last_error = f"Lost contact with Ollama ({type(exc).__name__})."
            self.last_verdict = "error"
            return None
        except Exception as exc:  # noqa: BLE001 - one bad line must not end the run
            log.exception("second look failed for %r", line.source[:80])
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_verdict = "error"
            return None
        reply = parse_reply(content)
        if reply is None:
            self.last_verdict = "bad-reply"
            log.debug("second look: unreadable reply %r", content[:120])
            return None
        neighbours = [item.english for item in before + after if item.english]
        revision, self.last_verdict = judge(
            line, reply, neighbours, runs_on=joined is not None, fix_words=True
        )
        if revision is None and joined is not None:
            # The model had nothing; the rule still applies.
            revision, self.last_verdict = Revision(line.source, joined, False, True), "changed"
        elif revision is not None and joined is not None and not revision.english_changed:
            revision.english, revision.english_changed = joined, True
        log.debug(
            "second look %s (%s) in %.0f ms: %r %r / %r",
            self.last_verdict,
            reply.check,
            (time.perf_counter() - t0) * 1000,
            line.source,
            reply.fixes,
            reply.english,
        )
        return revision
