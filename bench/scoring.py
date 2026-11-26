"""Turn a recorded pipeline run into numbers.

A run is a list of timestamped events - text that was committed, previews that
were shown, translations that were finalised - plus the sample's sidecar,
which says what was really said and when. Everything here is a comparison
between those two.

The one idea doing most of the work: every event carries Japanese source
text, and that text can be found in the reference transcript. Where it ends in
the reference tells us what moment of audio the event had caught up to, and
the event's own timestamp says when the viewer saw it. The difference is the
latency a person would have experienced - measured from the audio, not from
the pipeline's own bookkeeping, so it cannot flatter the thing being measured.
"""

from __future__ import annotations

import numpy as np

from bench.textmetrics import (
    ReferenceTimeline,
    align,
    cer,
    chrf,
    normalize_ja,
    percentile,
)

# How far back in the reference an event's text may be searched for. Bounded
# so a phrase that recurs (ごめんなさい is in the samples four times) matches
# its nearest earlier occurrence rather than its first.
LOOKBACK_SEC = 30.0
# An event cannot be about audio that had not been played yet. A little slack
# covers the aligner's own timing error.
LOOKAHEAD_SEC = 0.4


def locate_end(ref: str, hyp: str) -> tuple[int, int]:
    """Where in ref the text hyp most plausibly ends. Returns (end, distance).

    Semi-global edit distance: hyp must be consumed whole, but may start and
    stop anywhere in ref at no cost. Ties go to the latest end. The texts
    this is used on are previews, which by construction end at the most
    recent audio; the caller bounds how recent that is allowed to be.
    """
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return 0, m
    r = np.frombuffer(ref.encode("utf-32-le"), dtype=np.uint32)
    steps = np.arange(n + 1)
    prev = np.zeros(n + 1, dtype=np.int32)
    for j in range(1, m + 1):
        cur = np.empty(n + 1, dtype=np.int32)
        cur[0] = j
        cost = (r != ord(hyp[j - 1])).astype(np.int32)
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = np.minimum.accumulate(cur - steps) + steps
    best = int(prev.min())
    end = int(np.flatnonzero(prev == best)[-1])
    return end, best


class Locator:
    """Finds event text in a sample's reference, near the event's own time."""

    def __init__(self, sidecar: dict):
        self.timeline = ReferenceTimeline(sidecar)
        self._times = np.asarray(self.timeline.times, dtype=np.float64)

    def span(self, source: str, at: float) -> tuple[int, int] | None:
        """Reference character range [start, end) that `source` corresponds to.

        None if the text cannot be placed - too short, or nothing like it was
        said in the window, which is what a hallucination looks like.
        """
        hyp = normalize_ja(source)
        if len(hyp) < 2:
            return None
        lo = int(np.searchsorted(self._times, at - LOOKBACK_SEC, side="left"))
        hi = int(np.searchsorted(self._times, at + LOOKAHEAD_SEC, side="right"))
        if hi <= lo:
            return None
        end, distance = locate_end(self.timeline.text[lo:hi], hyp)
        if distance > 0.6 * len(hyp):
            return None
        end += lo
        return max(lo, end - len(hyp)), end

    def sequence(self, events: list[dict], key: str) -> list[tuple[int, int] | None]:
        """Reference spans for a run of events whose texts follow one another.

        Committed text and finished translations are each a single stream cut
        into pieces: placing every piece independently would let a two-
        character piece match any of the dozen places those two characters
        occur. Aligning the whole stream once and reading each piece's place
        off the alignment has no such freedom.

        A piece the alignment puts at audio that had not been played when the
        piece was emitted is a misalignment, and is placed on its own instead.
        """
        norms = [normalize_ja(e.get(key, "")) for e in events]
        mapping, _counts = align(self.timeline.text, "".join(norms))
        spans: list[tuple[int, int] | None] = []
        position = 0
        for event, norm in zip(events, norms):
            hits = [m for m in mapping[position : position + len(norm)] if m >= 0]
            position += len(norm)
            span = (min(hits), max(hits) + 1) if hits else None
            if span is not None and self.time_of(span[1] - 1) > event["t"] + 1.0:
                span = self.span(event.get(key, ""), event["t"])
            spans.append(span)
        return spans

    def time_of(self, index: int) -> float:
        """Audio time at which reference character `index` finished."""
        index = min(max(index, 0), len(self._times) - 1)
        return float(self._times[index])


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 3),
        "p50": round(percentile(values, 50), 3),
        "p90": round(percentile(values, 90), 3),
        "max": round(float(np.max(values)), 3),
    }


def score(record: dict, sidecar: dict) -> dict:
    """Score one recorded run of one sample.

    `record` holds lists of events, each a dict with a time `t` on the audio
    clock (seconds since the sample started playing):

        commits   {t, text}                 source text that became final
        previews  {t, source, english}      provisional English
        finals    {t, source, english}      English that will not change
    """
    locator = Locator(sidecar)
    timeline = locator.timeline
    segments = sidecar["segments"]
    seg_of = timeline.segment_of

    # -- recognition -------------------------------------------------------
    ref_text = "".join(s["ja"] for s in segments)
    hyp_text = "".join(c["text"] for c in record["commits"])
    asr = cer(ref_text, hyp_text)
    # The same measure over the Japanese that ended up in the transcript. The
    # commit stream above is what the recogniser settled word by word; this is
    # what was kept, cut into lines and handed to the translator - the text a
    # viewer can scroll back to, and the text the English was made from.
    kept_text = "".join(f.get("source", "") for f in record["finals"])
    transcript = cer(ref_text, kept_text) if kept_text else None

    # -- translation -------------------------------------------------------
    ref_en = " ".join(s["en"] for s in segments)
    hyp_en = " ".join(f["english"] for f in record["finals"] if f["english"])
    translation_chrf = chrf(hyp_en, ref_en) if hyp_en else 0.0

    # -- latency -----------------------------------------------------------
    n_seg = len(segments)
    seg_chars = [0] * n_seg
    for owner in seg_of:
        seg_chars[owner] += 1
    seg_first_char = [0] * n_seg
    running = 0
    for k in range(n_seg):
        seg_first_char[k] = running
        running += seg_chars[k]

    first_english: list[float | None] = [None] * n_seg
    final_done: list[float | None] = [None] * n_seg
    covered_until = [0] * n_seg  # chars of each segment covered by finals

    def touches(span: tuple[int, int], k: int) -> int:
        a = max(span[0], seg_first_char[k])
        b = min(span[1], seg_first_char[k] + seg_chars[k])
        return max(0, b - a)

    source_lag: list[float] = []
    for commit, span in zip(
        record["commits"], locator.sequence(record["commits"], "text")
    ):
        if span:
            source_lag.append(commit["t"] - locator.time_of(span[1] - 1))

    final_lag: list[float] = []
    unplaced_finals = 0
    final_spans = locator.sequence(record["finals"], "source")
    placed = [
        ("preview", e, locator.span(e.get("source", ""), e["t"]))
        for e in record["previews"]
    ] + [("final", e, span) for e, span in zip(record["finals"], final_spans)]
    for kind, event, span in placed:
        if True:
            if not event.get("english"):
                continue
            if span is None:
                if kind == "final":
                    unplaced_finals += 1
                continue
            if kind == "final":
                final_lag.append(event["t"] - locator.time_of(span[1] - 1))
            for k in range(seg_of[span[0]], seg_of[span[1] - 1] + 1):
                hit = touches(span, k)
                # English "for" a sentence means it reached a few characters
                # in, not that it brushed the first one.
                if hit >= min(3, seg_chars[k]):
                    if first_english[k] is None or event["t"] < first_english[k]:
                        first_english[k] = event["t"]
                if kind == "final":
                    reached = span[1] - seg_first_char[k]
                    covered_until[k] = max(covered_until[k], min(reached, seg_chars[k]))
                    # Done once the final text reaches the sentence's last
                    # few characters; recognisers drop trailing particles.
                    if (
                        final_done[k] is None
                        and covered_until[k] >= seg_chars[k] - max(2, seg_chars[k] // 8)
                    ):
                        final_done[k] = event["t"]

    first_after_start = [
        first_english[k] - segments[k].get("speech_start", segments[k]["start"])
        for k in range(n_seg)
        if first_english[k] is not None
    ]
    final_after_end = [
        final_done[k] - segments[k].get("speech_end", segments[k]["end"])
        for k in range(n_seg)
        if final_done[k] is not None
    ]

    # -- stability ---------------------------------------------------------
    shown = [p["english"] for p in record["previews"] if p.get("english")]
    kept: list[float] = []
    for before, after in zip(shown, shown[1:]):
        common = 0
        for x, y in zip(before, after):
            if x != y:
                break
            common += 1
        kept.append(common / max(1, len(before)))
    final_sources = [normalize_ja(f["source"]) for f in record["finals"]]
    duplicates = sum(
        1
        for a, b in zip(final_sources, final_sources[1:])
        if a and a == b
    )

    # English produced while nobody was speaking: any final that could not be
    # placed in the reference at all.
    return {
        "sentences": n_seg,
        "cer": round(asr["cer"], 4),
        "cer_detail": {k: asr[k] for k in ("sub", "del", "ins", "ref_len", "hyp_len")},
        "transcript_cer": round(transcript["cer"], 4) if transcript else None,
        "transcript_cer_detail": (
            {k: transcript[k] for k in ("sub", "del", "ins", "ref_len", "hyp_len")}
            if transcript
            else None
        ),
        "chrf": round(translation_chrf, 2),
        "source_lag": _stats(source_lag),
        "first_english_after_start": _stats(first_after_start),
        "final_after_sentence_end": _stats(final_after_end),
        "final_lag": _stats(final_lag),
        "sentences_finalised": sum(1 for x in final_done if x is not None),
        "sentences_with_english": sum(1 for x in first_english if x is not None),
        "finals": len(record["finals"]),
        "empty_finals": sum(1 for f in record["finals"] if not f["english"]),
        "unplaced_finals": unplaced_finals,
        "duplicate_finals": duplicates,
        "previews": len(shown),
        "previews_per_sentence": round(len(shown) / max(1, n_seg), 2),
        # Share of the previous preview that survives into the next one. 1.0
        # is a preview that only ever grows; low is one that keeps rewriting.
        "preview_prefix_kept": round(float(np.mean(kept)), 3) if kept else None,
    }


def merge(per_sample: dict[str, dict]) -> dict:
    """Pool per-sample scores into one row for a whole run."""
    if not per_sample:
        return {}
    ref_len = sum(s["cer_detail"]["ref_len"] for s in per_sample.values())
    errors = sum(
        s["cer_detail"]["sub"] + s["cer_detail"]["del"] + s["cer_detail"]["ins"]
        for s in per_sample.values()
    )

    def pooled(key: str, field: str) -> float | None:
        values = [s[key][field] for s in per_sample.values() if s[key].get("n")]
        return round(float(np.mean(values)), 3) if values else None

    kept = [s.get("transcript_cer_detail") for s in per_sample.values()]
    kept_errors = sum(d["sub"] + d["del"] + d["ins"] for d in kept if d)
    kept_len = sum(d["ref_len"] for d in kept if d)

    return {
        "cer": round(errors / max(1, ref_len), 4),
        "transcript_cer": round(kept_errors / kept_len, 4) if kept_len else None,
        "chrf": round(float(np.mean([s["chrf"] for s in per_sample.values()])), 2),
        "source_lag_p50": pooled("source_lag", "p50"),
        "first_english_p50": pooled("first_english_after_start", "p50"),
        "final_after_end_p50": pooled("final_after_sentence_end", "p50"),
        "final_after_end_p90": pooled("final_after_sentence_end", "p90"),
        "final_lag_p50": pooled("final_lag", "p50"),
        "sentences": sum(s["sentences"] for s in per_sample.values()),
        "sentences_finalised": sum(s["sentences_finalised"] for s in per_sample.values()),
        "unplaced_finals": sum(s["unplaced_finals"] for s in per_sample.values()),
        "duplicate_finals": sum(s["duplicate_finals"] for s in per_sample.values()),
    }
