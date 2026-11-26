"""Put benchmark runs side by side.

    python -m bench.report base-fast fast
    python -m bench.report bench/results/base-fast.json some/other/run.json
    python -m bench.report --samples base-fast fast

Each argument is a label under bench/results/ or a path to a results file.
The first one is the reference the others are compared against.

Two error rates are shown, because they answer different questions. "live" is
the Japanese as it settled word by word while people were talking - what the
caption line shows. "kept" is the Japanese in the finished transcript lines,
which is the text the translator was actually given. Runs recorded before the
second figure existed have it worked out here from their saved transcripts.

Latency columns are blank for a fast replay: the audio was not played against
a clock, so there is nothing to measure them with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.textmetrics import cer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SAMPLES = ROOT / "samples"


def load(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = RESULTS / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("label", path.stem)
    return data


def kept_cer(run: dict, sample_id: str) -> tuple[int, int] | None:
    """(errors, reference length) of the transcript lines for one sample."""
    detail = run["samples"][sample_id].get("transcript_cer_detail")
    if detail:
        return detail["sub"] + detail["del"] + detail["ins"], detail["ref_len"]
    finals = run.get("transcripts", {}).get(sample_id, {}).get("finals")
    sidecar = SAMPLES / f"{sample_id}.json"
    if not finals or not sidecar.exists():
        return None
    reference = "".join(
        s["ja"] for s in json.loads(sidecar.read_text(encoding="utf-8"))["segments"]
    )
    hypothesis = "".join(f.get("source", "") for f in finals)
    if not hypothesis:
        return None
    counts = cer(reference, hypothesis)
    return counts["sub"] + counts["del"] + counts["ins"], counts["ref_len"]


def summarise(run: dict) -> dict:
    summary = dict(run.get("summary", {}))
    errors = length = 0
    for sample_id in run["samples"]:
        got = kept_cer(run, sample_id)
        if got:
            errors += got[0]
            length += got[1]
    summary["kept_cer"] = errors / length if length else None
    engine = run.get("engine", {})
    passes = sum(e.get("passes", 0) for e in engine.values())
    infer = sum(
        e.get("passes", 0) * (e.get("average_pass_ms") or e.get("mean_pass_ms") or 0)
        for e in engine.values()
    )
    summary["pass_ms"] = infer / passes if passes else None
    summary["passes"] = passes
    # How long after a word is spoken its Japanese is shown as settled, by
    # the pipeline's own clock. The scorer's source_lag is not used for this:
    # it times committed text, and what "committed" means changed - it was
    # each word as it settled, and is now each line as it is handed to the
    # translator - so the two are not the same measurement across runs.
    lags = [e["source_lag_sec"] for e in engine.values() if e.get("source_lag_sec") is not None]
    summary["settle_sec"] = sum(lags) / len(lags) if lags else None
    speech = sum(e.get("speech_sec", 0.0) for e in engine.values())
    summary["gpu_share"] = (
        sum(e.get("rtf", 0.0) * e.get("speech_sec", 0.0) for e in engine.values()) / speech
        if speech
        else None
    )
    for key in (
        "eager_endpoints", "passes_reused", "deferred_closes", "revised_lines",
        "tail_drops", "pause_splits", "stall_splits",
    ):
        summary[key] = sum(e.get(key, 0) for e in engine.values())
    summary["realtime"] = run.get("realtime", True)
    return summary


def _fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "sec":
        return f"{value:.2f}s"
    if kind == "ms":
        return f"{value:.0f}ms"
    if kind == "num":
        return f"{value:.1f}"
    return str(value)


ROWS = [
    ("Japanese errors, live (CER)", "cer", "pct", False),
    ("Japanese errors, kept (CER)", "kept_cer", "pct", False),
    ("English vs reference (chrF)", "chrf", "num", False),
    ("Japanese settles after word", "settle_sec", "sec", True),
    ("First English after start", "first_english_p50", "sec", True),
    ("Final English after end, p50", "final_after_end_p50", "sec", True),
    ("Final English after end, p90", "final_after_end_p90", "sec", True),
    ("Recognition pass, mean", "pass_ms", "ms", True),
    ("GPU time per second of speech", "gpu_share", "sec", True),
    ("Recognition passes", "passes", "int", False),
    ("Sentences finalised", "sentences_finalised", "int", False),
    ("Unplaced lines", "unplaced_finals", "int", False),
    ("Duplicate lines", "duplicate_finals", "int", False),
    ("Closed early (text complete)", "eager_endpoints", "int", False),
    ("Closing passes not needed", "passes_reused", "int", False),
    ("Pauses held (text unfinished)", "deferred_closes", "int", False),
    ("Settled text revised", "revised_lines", "int", False),
    ("Unspoken tails dropped", "tail_drops", "int", False),
    ("Buffer cut at a pause", "pause_splits", "int", False),
    ("Recogniser stalls broken", "stall_splits", "int", False),
]


def table(runs: list[dict]) -> str:
    summaries = [summarise(run) for run in runs]
    labels = [run["label"] for run in runs]
    width = max(14, *(len(label) + 2 for label in labels))
    lines = [f"{'':32s}" + "".join(f"{label:>{width}s}" for label in labels)]
    for title, key, kind, timed in ROWS:
        cells = []
        for summary in summaries:
            value = summary.get(key)
            if timed and not summary["realtime"]:
                value = None
            cells.append(f"{_fmt(value, kind):>{width}s}")
        lines.append(f"{title:32s}" + "".join(cells))
    return "\n".join(lines)


def per_sample(runs: list[dict]) -> str:
    labels = [run["label"] for run in runs]
    width = max(22, *(len(label) + 2 for label in labels))
    lines = [f"{'live / kept CER':24s}" + "".join(f"{label:>{width}s}" for label in labels)]
    for sample_id in runs[0]["samples"]:
        cells = []
        for run in runs:
            if sample_id not in run["samples"]:
                cells.append(f"{'-':>{width}s}")
                continue
            live = run["samples"][sample_id]["cer"]
            got = kept_cer(run, sample_id)
            kept = f"{got[0] / got[1] * 100:.1f}%" if got else "-"
            cells.append(f"{f'{live * 100:.1f}% / {kept}':>{width}s}")
        lines.append(f"{sample_id:24s}" + "".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="labels or paths; the first is the reference")
    parser.add_argument("--samples", action="store_true", help="add the per-sample error rates")
    args = parser.parse_args(argv)
    runs = [load(name) for name in args.runs]
    print(table(runs))
    if args.samples:
        print()
        print(per_sample(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
