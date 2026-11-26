"""Text scoring shared by the bench scripts: CER, chrF, and alignment.

Self-contained on purpose. The usual packages for these (jiwer, sacrebleu) are
fine, but each is a dependency whose only job would be forty lines of
arithmetic, and the alignment below is needed anyway - it is what turns "this
text was emitted at this moment" into a latency.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

# Everything that is not a letter, digit or kana/kanji. Recognisers disagree
# freely about punctuation and spacing in Japanese, and none of that is an
# error a viewer would notice, so it is not counted as one.
_DROP = re.compile(r"[^\w]|_", re.UNICODE)


def normalize_ja(text: str) -> str:
    """Canonical form for comparing Japanese transcripts.

    NFKC folds full-width digits and Latin (１００ｍ -> 100m) so a model is not
    marked wrong for its choice of width. Kanji/kana choices are left alone:
    a model that writes わたし for 私 is charged for it, as in every published
    Japanese CER.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return _DROP.sub("", text)


def _distance_matrix(ref: str, hyp: str) -> np.ndarray:
    """Full Levenshtein table, rows over ref and columns over hyp.

    One vectorised row at a time. The insertion term makes a row depend on
    itself, which is handled by the running-minimum identity
    d[j] = min_k<=j (t[k] + j - k).
    """
    n, m = len(ref), len(hyp)
    h = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32) if m else np.zeros(0, np.uint32)
    table = np.zeros((n + 1, m + 1), dtype=np.int32)
    table[0] = np.arange(m + 1)
    steps = np.arange(m + 1)
    for i in range(1, n + 1):
        prev = table[i - 1]
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        if m:
            cost = (h != ord(ref[i - 1])).astype(np.int32)
            row[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        table[i] = np.minimum.accumulate(row - steps) + steps
    return table


def align(ref: str, hyp: str) -> tuple[list[int], dict]:
    """Map every hyp character to the ref character it corresponds to.

    Returns (mapping, counts). mapping[j] is the index into ref that hyp[j]
    matched or substituted, or -1 if hyp[j] is an insertion. counts holds the
    substitution / deletion / insertion totals of the optimal path.
    """
    n, m = len(ref), len(hyp)
    table = _distance_matrix(ref, hyp)
    mapping = [-1] * m
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if table[i, j] == table[i - 1, j - 1] + cost:
                mapping[j - 1] = i - 1
                subs += cost
                i, j = i - 1, j - 1
                continue
        if j > 0 and table[i, j] == table[i, j - 1] + 1:
            ins += 1
            j -= 1
            continue
        dels += 1
        i -= 1
    return mapping, {"sub": subs, "del": dels, "ins": ins, "ref_len": n, "hyp_len": m}


def cer(ref: str, hyp: str) -> dict:
    """Character error rate between two transcripts, after normalize_ja."""
    ref_n, hyp_n = normalize_ja(ref), normalize_ja(hyp)
    _mapping, counts = align(ref_n, hyp_n)
    errors = counts["sub"] + counts["del"] + counts["ins"]
    counts["cer"] = errors / max(1, len(ref_n))
    return counts


# -- chrF -------------------------------------------------------------------


def _char_ngrams(text: str, n: int) -> Counter:
    # Whitespace removed, as sacreBLEU's chrF does by default.
    text = "".join(text.split())
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(hyp: str, ref: str, order: int = 6, beta: float = 2.0) -> float:
    """chrF2 on a 0-100 scale: character n-gram F-score, recall-weighted.

    Chosen over BLEU for this job because the outputs are short and the
    references single: BLEU on a ten-word line is mostly zeros, while chrF
    still separates "close" from "wrong". It tracks sacreBLEU's chrF2 closely
    but is not guaranteed digit-identical, so compare numbers produced here
    with each other and not with published tables.
    """
    hyp, ref = hyp.lower(), ref.lower()
    precisions, recalls = [], []
    for n in range(1, order + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


# -- timing -----------------------------------------------------------------


class ReferenceTimeline:
    """When each character of a sample's reference transcript was spoken.

    Built from the sidecar. If build_samples' alignment pass has run, each
    segment carries per-character times measured from the audio; otherwise
    they are interpolated linearly across the segment, which is good to a few
    hundred milliseconds on clean clips and worse on ones with lead-in noise.
    """

    def __init__(self, sidecar: dict):
        self.text = ""
        self.times: list[float] = []  # time each character has *finished*
        self.segment_of: list[int] = []
        self.segments = sidecar["segments"]
        for index, seg in enumerate(self.segments):
            chars = normalize_ja(seg["ja"])
            measured = seg.get("char_times")
            for k, _ch in enumerate(chars):
                if measured and len(measured) == len(chars):
                    t = float(measured[k])
                else:
                    span = seg["end"] - seg["start"]
                    t = seg["start"] + span * (k + 1) / max(1, len(chars))
                self.times.append(t)
                self.segment_of.append(index)
            self.text += chars

    def locate(self, hyp_stream: str) -> list[float | None]:
        """Audio time for each character of a normalised hypothesis stream.

        Insertions take the time of the nearest matched character before them,
        so a hallucinated tail still gets a time rather than being dropped
        from the latency figures.
        """
        mapping, _counts = align(self.text, hyp_stream)
        out: list[float | None] = []
        last: float | None = None
        for ref_index in mapping:
            if ref_index >= 0:
                last = self.times[ref_index]
            out.append(last)
        return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
