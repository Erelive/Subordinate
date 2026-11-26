"""Where one translation unit ends and the next begins.

This decides what the translator is handed, and that decides most of what the
viewer thinks of the translation. A translator given a whole sentence is
usually right; given a sentence severed mid-verb it is fluently wrong, because
Japanese puts the verb last and hangs negation and tense on it. Measured on
the benchmark set with the previous fixed-timer cut:

    これは日本で | っていない食べ物です   ->  "...in Japan." / "It's not food."

The source said "this is a food that isn't sold in Japan". Nothing downstream
can recover that; the cut has to be right.

Three kinds of evidence are used, strongest first.

  Punctuation. When the recogniser writes 。 it has heard a sentence end.
  Reliable where it exists - and Kotoba-Whisper never writes any.

  Pauses. People breathe between sentences. The voice detector records every
  pause on the same clock as the word timestamps, so a pause can be matched to
  the word boundary it fell on. A long one ends the utterance outright. A
  short one is only a candidate: speakers also pause after は and を, in the
  middle of a thought.

  Grammar. So a short pause is believed only if the text before it reads as
  finished - it ends in a predicate or a sentence-final particle - and the
  text after it does not open with something that could only be a
  continuation. Conversely a long pause is doubted, for a little while, if
  the text before it ends in a particle that demands more.

None of this parses Japanese. It is suffix matching, chosen so that each list
errs in the cheap direction: a missed boundary costs latency (the next signal
catches it), a false one costs a broken sentence.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence


class _Word(Protocol):
    start: float
    end: float
    text: str


class _Pause(Protocol):
    start: float
    end: float

    @property
    def center(self) -> float: ...


# What a recogniser puts at the end of a sentence. "..." is deliberately
# absent - it marks hesitation mid-sentence far more often than an ending.
SENTENCE_END = ("。", "！", "？", ".", "!", "?")

# Clause marks. Not a sentence ending, but a far better place to cut than the
# middle of a phrase when the backstop has to fire anyway.
CLAUSE_END = ("、", "，", ",")

# An utterance with no word in it. Whisper emits these for breaths, laughs cut
# short and false starts: the chunk is a run of ellipses and nothing else. It
# reaches the translator as "...", comes back as "..." or empty, and renders
# as a caption line that says nothing - which reads as a missed translation
# rather than as the silence it actually was.
_NO_CONTENT = re.compile(r"^[\s.。、,，…‥・~〜～ー\-—_!?！？「」『』（）()\[\]'\"♪]*$")

# Trailing decoration that says nothing about whether the sentence is over:
# closing quotes, elongation (だねー, だねぇ), small-tsu cut-offs.
_TAIL_NOISE = " \t　」』）)\"'ー〜～…‥っッぁぃぅぇぉ♪"

# Endings that mean the sentence has, grammatically, arrived somewhere. Every
# entry is here because it is rarely anything except final. The bare
# dictionary-form endings (う く す つ ...) are deliberately absent: half the
# adverbs in the language end in one of them.
_COMPLETE = (
    # polite predicates
    "ます", "ました", "ません", "ましょう", "です", "でした", "でしょう",
    "ください", "なさい", "ございます",
    # plain copula and past
    "だった", "である", "だろう", "だろ", "でしょ", "じゃない", "じゃん",
    "た", "だ",
    # negatives and desideratives
    "ない", "たい", "なかった",
    # verbs in dictionary form that close a sentence far more often than not
    "する", "いる", "ある", "なる", "れる", "える", "くる", "みる", "てる", "でる",
    "思う", "言う", "違う", "ちゃう", "じゃう", "しまう", "もらう",
    # sentence-final particles. の and bare な are left out on purpose: both
    # are at least as often the middle of a noun phrase (私の, 静かな).
    "よ", "ね", "か", "わ", "ぞ", "ぜ",
    "もん", "かな", "だな", "よな", "かも", "かしら", "っけ",
    # set phrases, listed whole because their last syllable is a particle
    "ありがとう", "おはよう", "こんにちは", "こんばんは", "ごめん", "はい", "うん",
    "いいえ",
)

# Endings after which a Japanese sentence is, grammatically, still owed
# something: case and topic particles, conjunctive forms, quotatives, fillers.
# けど and から are here although speakers trail off on them constantly -
# holding those a moment longer costs half a second when they were final, and
# saves the sentence when they were not.
_INCOMPLETE = (
    "は", "が", "を", "に", "へ", "と", "で", "も", "や",
    "から", "ので", "のに", "けど", "けれど", "けども", "けれども",
    "て", "し", "たり", "ば", "たら", "なら", "とか", "って", "という", "っていう",
    "ながら", "つつ", "より", "ように", "ために",
    # words that end in a syllable _COMPLETE would otherwise claim
    "まだ", "また", "ただ", "あなた", "あの", "その", "この", "どの",
    "何か", "なにか", "なんか", "どこか", "いつか", "誰か", "だれか", "かどうか",
    "そして", "でも", "だから", "それで", "しかし", "あと", "つまり",
    "えっと", "えーと", "まあ", "ちょっと",
)

# Particles that are also tossed into the middle of a sentence as fillers:
# それでね, 私はさ. After one of these the sentence is as finished as the
# text in front of the particle is.
_INTERJECTORY = "よねさな"

# How a continuation opens. If the word after a candidate boundary is one of
# these, the boundary is inside a sentence no matter how finished the text
# before it looks: 行きます | けど is one sentence, not two.
#
# The short ones are matched against a whole recogniser token rather than as
# a prefix. As prefixes they would veto every sentence that happens to start
# with the same syllable - かなり, なんだか, よく.
_CONTINUES_EXACT = frozenset(
    ("ね", "よ", "か", "わ", "な", "の", "し", "が", "と", "から", "もん", "とか")
)
_CONTINUES_PREFIX = (
    "けど", "けれど", "ので", "のに", "って", "んです", "んだ", "んで",
    "でしょ", "だろ", "という", "と思",
)


# Whole utterances that are complete at two or three characters. Anything
# else that short is more likely a fragment of something than a sentence.
_SHORT_REPLIES = frozenset(
    ("はい", "うん", "ええ", "いえ", "いや", "そう", "ねえ", "ほら", "よし", "うそ", "まじ", "なに")
)


def is_fragment(text: str) -> bool:
    """Too short to be translated on its own with any confidence.

    A two-character unit handed to a translator comes back as a confident
    sentence about nothing: さんだ became "It's Sanda." Short units are held
    a little longer in case they are the start of something, unless they are
    one of the handful of words that are whole replies by themselves.
    """
    core = "".join(ch for ch in text if ch.isalnum())
    if len(core) >= 4:
        return False
    return core not in _SHORT_REPLIES


def has_content(text: str) -> bool:
    """False for text that is only punctuation, ellipses and the like."""
    return not _NO_CONTENT.match(text)


def join_words(words: Sequence[_Word]) -> str:
    # Whisper word tokens carry their own leading space.
    return "".join(w.text for w in words).strip()


def _core(text: str) -> str:
    """Text with trailing punctuation and decoration removed."""
    text = text.strip()
    trimmed = text.rstrip(_TAIL_NOISE + "".join(SENTENCE_END))
    return trimmed


def ends_sentence(text: str) -> bool:
    """Did the recogniser mark this as a finished sentence?"""
    text = text.strip()
    return text.endswith(SENTENCE_END) and not text.endswith("...")


def _classify(core: str) -> int:
    """+1 reads as finished, -1 as dangling, 0 no opinion.

    The longest matching ending decides. かも ends in も and is final; とか
    ends in か and is not; whichever list holds the longer match knows more
    about the word than the one that only recognised its last syllable.
    """
    if not core:
        return 0
    done = max((len(e) for e in _COMPLETE if core.endswith(e)), default=0)
    dangling = max((len(e) for e in _INCOMPLETE if core.endswith(e)), default=0)
    if not done and not dangling:
        return 0
    if dangling >= done:
        return -1
    if done == 1 and core[-1] in _INTERJECTORY and len(core) > 1:
        if _classify(core[:-1]) < 0:
            return -1
    return 1


def looks_complete(text: str) -> bool:
    """Does this read as a finished Japanese sentence? Conservative."""
    if ends_sentence(text):
        return True
    stripped = text.strip()
    if stripped.endswith(CLAUSE_END) or stripped.endswith(("…", "‥", "...")):
        return False
    return _classify(_core(stripped)) > 0


def looks_incomplete(text: str) -> bool:
    """Is this visibly waiting for more - a dangling particle or clause mark?"""
    if ends_sentence(text):
        return False
    stripped = text.strip()
    if stripped.endswith(CLAUSE_END) or stripped.endswith(("…", "‥", "...")):
        return True
    return _classify(_core(stripped)) < 0


def opens_as_continuation(word: str) -> bool:
    """Would this recogniser token, coming next, belong to the sentence before?"""
    token = word.strip().rstrip("、，,。")
    return token in _CONTINUES_EXACT or token.startswith(_CONTINUES_PREFIX)


# -- word boundaries --------------------------------------------------------

# Kana that cannot begin a word: the small ones, the sokuon, the long-vowel
# mark, and ん. A cut that leaves one of these at the head of a line has cut
# a word in two.
_NEVER_INITIAL = "ぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッーんン"

# The one-kana particles. A kanji followed by one of these is the end of a
# word; a kanji followed by any other hiragana is usually its own ending
# (行|きます, 食|べる).
_PARTICLES = "はがをにでともへやのかねよ"


def _script(ch: str) -> str:
    """Which writing system a character belongs to: h, k, c, l, or o."""
    code = ord(ch)
    if 0x3041 <= code <= 0x309F:
        return "h"
    if 0x30A0 <= code <= 0x30FF or 0xFF66 <= code <= 0xFF9F:
        return "k"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or ch in "々〆":
        return "c"
    if ch.isascii() and ch.isalnum() or 0xFF10 <= code <= 0xFF5A:
        return "l"
    return "o"


def splits_word(before: str, after: str) -> bool:
    """Would a cut between these two texts probably fall inside a word?

    Japanese is written without spaces and the recogniser's tokens are not
    words, so a boundary between two tokens says nothing by itself. The
    script does: words are mostly runs of one script, with a hiragana ending.
    Two kanji either side of a cut are nearly always one compound (台|風),
    as are two katakana (ニュ|ース) or two Latin letters (N|HK); a hiragana
    followed by anything else is nearly always a new word.

    Deliberately one-sided: it reports the cuts it is confident are wrong and
    says nothing about the rest. Used to choose between neighbouring
    boundaries, never to forbid a cut outright.
    """
    left = next((ch for ch in reversed(before) if ch.isalnum()), "")
    right = next((ch for ch in after if not ch.isspace()), "")
    if not left or not right or not right.isalnum():
        return False
    if right in _NEVER_INITIAL or left in "っッ":
        return True
    a, b = _script(left), _script(right)
    if a == b and a in "ckl":
        return True
    if a == "c" and b == "h" and right not in _PARTICLES:
        return True
    return False


# -- cuts over a list of words ---------------------------------------------


def sentence_cut(
    words: Sequence[_Word], next_word: str = "", trailing_ok: bool = True
) -> int:
    """How many leading words form whole punctuated sentences. 0 if none.

    A sentence mark is the recogniser's opinion, formed on the audio it had
    at the time, and it revises that opinion: なさそうです。 on one pass
    becomes なさそうですけどね on the next, by which point the first version
    has been translated as a complete sentence. So a mark is only acted on
    when what follows it is consistent with a sentence having ended.

    next_word is the first token after `words`, if the recogniser has one in
    flight. trailing_ok says whether a mark on the very last word, with
    nothing known to follow it, may be trusted - true once the speaker has
    paused, false while they are still talking and the next word is simply
    not in yet.
    """
    for i in range(len(words) - 1, -1, -1):
        if not ends_sentence(words[i].text):
            continue
        last = i == len(words) - 1
        following = next_word if last else words[i + 1].text
        if following:
            if opens_as_continuation(following):
                continue
        elif not trailing_ok:
            continue
        return i + 1
    return 0


def clause_cut(words: Sequence[_Word]) -> int:
    """How many leading words end on a clause mark. 0 if none do."""
    for i in range(len(words) - 1, -1, -1):
        if words[i].text.strip().endswith(CLAUSE_END):
            return i + 1
    return 0


def _letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def pause_boundaries(
    words: Sequence[_Word],
    pauses: Sequence[_Pause],
    next_start: float | None = None,
    base_letters: int | None = None,
) -> list[int]:
    """Word counts at which a recorded pause falls, ascending.

    A boundary of n means "after the first n words".

    There are two ways to place a pause among the words, and the better one
    does not use word timings at all. When the recogniser ran while the
    speaker was pausing, the buffer it was given ended in that pause, so
    what it returned is exactly the text that precedes the pause. The
    pipeline records how long that text was (pause.letters, counted from the
    start of the run), and base_letters says how much of the run came before
    `words`; the difference is how far into `words` the pause falls.

    Without that, the pause is placed by time. Word timestamps from
    cross-attention are good to a tenth of a second or so, which is about the
    length of the pauses being matched, so nothing compares edges: a pause
    belongs to the boundary whose neighbouring words have their midpoints
    either side of the pause's centre. On Japanese, where a token is a
    character or two, that is routinely one token out - enough to put the
    か of かなり on the wrong side.

    next_start is when the first word *after* this list begins, if known - the
    unstable tail. Without it, a pause after the last word cannot be told
    apart from one in the middle of a word still being recognised.
    """
    if not words or not pauses:
        return []
    mids = [(w.start + w.end) / 2.0 for w in words]
    counts = [_letter_count(w.text) for w in words]
    total = sum(counts)
    found: set[int] = set()
    for pause in pauses:
        mark = getattr(pause, "letters", None)
        if mark is not None and base_letters is not None:
            target = mark - base_letters
            if target <= 0 or target > total + 1:
                continue  # the pause is before these words, or beyond them
            if target >= total:
                found.add(len(words))
                continue
            run = 0
            for n, size in enumerate(counts, start=1):
                run += size
                if run >= target:
                    # The nearer word boundary, if the mark falls inside a
                    # token: the text has been re-read since it was taken.
                    found.add(n if run - target <= target - (run - size) else n - 1)
                    break
            found.discard(0)
            continue
        c = pause.center
        if c <= mids[0]:
            continue
        if c > mids[-1]:
            # After the last word. Only a boundary if nothing still being
            # recognised sits between that word and the pause.
            if pause.end >= words[-1].end - 0.15 and (
                next_start is None or next_start >= pause.start - 0.1
            ):
                found.add(len(words))
            continue
        for n in range(1, len(words)):
            if mids[n - 1] < c <= mids[n]:
                found.add(n)
                break
    return sorted(found)


def nearest_boundary(
    words: Sequence[_Word], boundaries: Sequence[int], index: int, within_sec: float = 1.0
) -> int:
    """The boundary closest in time to where word `index` begins. 0 if none is near.

    For a change of speaker. The speaker tracker labels windows of a second
    or two, so the word at which it reports a new voice is good to about
    that; cutting the text exactly there puts the cut in the middle of
    whatever word the window happened to end on (この動画ハ | プニング).
    People take turns at pauses. The pause nearest the reported change is
    where the change was, and if there is no pause near it, the report is
    more likely a wobble in the tracker than a second person.
    """
    if not 0 <= index < len(words):
        return 0
    when = words[index].start
    best, best_gap = 0, within_sec
    for n in boundaries:
        if not 1 <= n <= len(words):
            continue
        gap = abs(words[n - 1].end - when)
        if gap <= best_gap:
            best, best_gap = n, gap
    return best


def pause_cut(
    words: Sequence[_Word],
    pauses: Sequence[_Pause],
    next_word: str = "",
    next_start: float | None = None,
    min_chars: int = 3,
    base_letters: int | None = None,
) -> int:
    """Latest pause boundary where the text before it reads as finished.

    next_word and next_start describe the first token after `words`, if the
    recogniser has one in flight; they let a boundary at the very end be
    judged the same way as one in the middle.

    The boundary is allowed to slide one word either way, because the
    recogniser's "words" in Japanese are sub-word pieces and a pause matched
    to the gap before す is really the gap after ます.
    """
    boundaries = pause_boundaries(words, pauses, next_start, base_letters)
    for n in reversed(boundaries):
        for candidate in (n, n + 1, n - 1):
            if candidate < 1 or candidate > len(words):
                continue
            before = join_words(words[:candidate])
            if (
                len(_core(before)) < min_chars
                or is_fragment(before)
                or not looks_complete(before)
            ):
                continue
            following = words[candidate].text if candidate < len(words) else next_word
            if following and opens_as_continuation(following):
                continue
            if following and splits_word(before, following):
                continue
            return candidate
    return 0


def overdue_cut(
    words: Sequence[_Word],
    pauses: Sequence[_Pause],
    next_word: str = "",
    base_letters: int | None = None,
) -> int:
    """Where to cut when the hold has run out and no sentence has ended.

    Something has to be translated now. In order of preference: the last
    pause, wherever it fell, since a breath is at least a phrase boundary; the
    last clause mark; the last particle, which ends a phrase by definition;
    and only then everything - backed off, if need be, to the last place that
    is not the middle of a word.

    A pause is matched to a word boundary through word timings that are good
    to a syllable at best, so the boundary it lands on may be one token out.
    The neighbours are tried too, and one that does not split a word is
    preferred to one that does: the pause after みたい must not take the 台
    of 台風 with it.
    """
    if not words:
        return 0

    def following(n: int) -> str:
        return words[n].text if n < len(words) else next_word

    def clean(n: int) -> bool:
        return not splits_word(join_words(words[:n]), following(n))

    boundaries = [
        n
        for n in pause_boundaries(words, pauses, base_letters=base_letters)
        if 1 <= n < len(words)
    ]
    # Not a boundary so early that it strands a syllable ahead of the cut.
    for n in reversed(boundaries):
        for candidate in (n, n - 1, n + 1):
            if not 1 <= candidate < len(words):
                continue
            if len(_core(join_words(words[:candidate]))) >= 4 and clean(candidate):
                return candidate
    cut = clause_cut(words)
    if cut:
        return cut
    for n in range(len(words) - 1, 1, -1):
        if _classify(_core(join_words(words[:n]))) < 0 and clean(n):
            return n
    for n in range(len(words), max(1, len(words) - 4), -1):
        if clean(n):
            return n
    return len(words)


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, keeping each mark with its sentence.

    Sentence-level NMT models are trained on single sentences and tend to
    translate the first of several and drop the rest, so multi-sentence
    utterances are split before they are handed over.
    """
    parts: list[str] = []
    current = []
    for index, ch in enumerate(text):
        current.append(ch)
        if ch in "。！？!?":
            # Keep runs like ？！ and a closing quote with the sentence.
            following = text[index + 1 : index + 2]
            if following and following in "。！？!?」』）)":
                continue
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
    piece = "".join(current).strip()
    if piece:
        if parts and not has_content(piece):
            parts[-1] += piece
        else:
            parts.append(piece)
    return parts
