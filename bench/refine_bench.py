"""Measure the second look on text, without audio.

    python -m bench.refine_bench                       # the app's two LLMs
    python -m bench.refine_bench --llm qwen3:4b-instruct
    python -m bench.refine_bench --first nmt --show    # NMT first pass, print every change

The audio samples cannot test this: their sentences are unrelated to each
other, and the second look works by reading a line between its neighbours. So
this uses a scripted stream (refine_cases.json) in which each line is given
three ways - as said, as a recogniser typically hears it, and in English.

For each model: translate the heard lines in order (the first pass, exactly as
the app does it), then give each line its second look with the next lines in
view, and compare, before and after:

  Japanese errors   characters of the heard line that differ from what was
                    said (CER). Lower after means words were repaired; higher
                    means the model "corrected" something that was right.
  English           chrF against the reference. It under-rates a translation
                    that is free rather than literal, so read --show as well.

It also counts how many replies were thrown away by the guards in refine.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench.textmetrics import cer, chrf
from subordinant import glossary
from subordinant.config import Config
from subordinant.profiles import QWEN3_4B, SUGOI_14B
from subordinant.refine import Line, Refiner

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _errors(ref: str, hyp: str) -> tuple[int, int]:
    counts = cer(ref, hyp)
    return counts["sub"] + counts["del"] + counts["ins"], counts["ref_len"]


def first_pass(cfg: Config, heard: list[str], engine: str) -> list[str]:
    if engine == "nmt":
        from subordinant.translate import Translator

        translator = Translator(cfg)
        return [translator.translate(text) for text in heard]
    from subordinant.llm_translate import LLMTranslator

    translator = LLMTranslator(cfg)
    return [translator.translate(text) for text in heard]


def run(model: str, lines: list[dict], args) -> dict:
    cfg = Config()
    cfg.mt_backend = "llm"
    cfg.llm_model = model
    cfg.refine_model = args.refine_model or ""
    cfg.llm_context_utterances = args.context
    cfg.slang_glossary = not args.no_glossary
    cfg.refine_fix_words = args.fix_words
    # What the pipeline hands on: the recogniser's text with the known
    # mishearings of slang respelt (CaptionPipeline._emit_unit).
    heard = [
        glossary.default().respell(row["heard"]) if cfg.slang_glossary else row["heard"]
        for row in lines
    ]
    first = first_pass(cfg, heard, args.first)
    if args.no_second_look:
        refiner = None
    else:
        refiner = Refiner(cfg)
    final_ja = list(heard)
    final_en = list(first)
    verdicts: dict[str, int] = {}
    times = []
    changes = []
    for i, row in enumerate(lines):
        before = [
            Line(final_ja[j], final_en[j]) for j in range(max(0, i - cfg.refine_context_lines), i)
        ]
        if refiner is None:
            break
        after = [
            Line(heard[j], first[j])
            for j in range(i + 1, min(len(lines), i + 1 + cfg.refine_lookahead))
        ]
        t0 = time.perf_counter()
        revision = refiner.refine(Line(heard[i], first[i]), before, after)
        times.append(time.perf_counter() - t0)
        verdicts[refiner.last_verdict] = verdicts.get(refiner.last_verdict, 0) + 1
        if refiner.last_verdict == "skipped":
            times.pop()
        if revision is not None:
            final_ja[i], final_en[i] = revision.source, revision.english
            changes.append(i)

    ja_heard = [_errors(r["said"], r["heard"]) for r in lines]
    ja_before = [_errors(r["said"], heard[i]) for i, r in enumerate(lines)]
    ja_after = [_errors(r["said"], final_ja[i]) for i, r in enumerate(lines)]
    total = sum(n for _e, n in ja_before)
    result = {
        "model": model,
        "first": args.first,
        "glossary": cfg.slang_glossary,
        "cer_heard": sum(e for e, _n in ja_heard) / total,
        "cer_before": sum(e for e, _n in ja_before) / total,
        "cer_after": sum(e for e, _n in ja_after) / total,
        "lines_repaired": sum(a[0] < b[0] for a, b in zip(ja_after, ja_before)),
        "lines_damaged": sum(a[0] > b[0] for a, b in zip(ja_after, ja_before)),
        "chrf_before": sum(chrf(first[i], r["en"]) for i, r in enumerate(lines)) / len(lines),
        "chrf_after": sum(chrf(final_en[i], r["en"]) for i, r in enumerate(lines)) / len(lines),
        "verdicts": verdicts,
        "ms_per_line": 1000 * sum(times) / len(times) if times else None,
        "lines": [
            {
                "said": r["said"], "heard": heard[i], "repaired": final_ja[i],
                "first": first[i], "second": final_en[i], "reference": r["en"],
            }
            for i, r in enumerate(lines)
        ],
    }
    if args.show:
        for i in changes:
            row = result["lines"][i]
            print(f"\n  [{i + 1}] {row['heard']}")
            if row["repaired"] != row["heard"]:
                print(f"      JA -> {row['repaired']}   (said: {row['said']})")
            print(f"      1st   {row['first']}")
            if row["second"] != row["first"]:
                print(f"      2nd   {row['second']}")
            print(f"      ref   {row['reference']}")
    return result


def run_transcripts(model: str, label: str, args) -> dict:
    """The second look over what the recogniser really produced.

    Takes the finished lines of a saved bench.run (its Japanese and first
    English, per sample) and gives each its second look, as the pipeline
    would. These are the audio samples, whose sentences do not follow from
    each other - the worst case for a pass that reads a line against its
    neighbours, and so the place to see what it does when the context is no
    help: every change it makes here is suspect.
    """
    run_file = RESULTS / f"{label}.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    cfg = Config()
    cfg.mt_backend = "llm"
    cfg.llm_model = model
    cfg.slang_glossary = not args.no_glossary
    cfg.refine_fix_words = args.fix_words
    refiner = Refiner(cfg)
    before_err = after_err = length = looked = 0
    changes = []
    times = []
    for sample_id, transcript in saved["transcripts"].items():
        finals = [f for f in transcript["finals"] if f.get("source")]
        sidecar = ROOT / "samples" / f"{sample_id}.json"
        if not finals or not sidecar.exists():
            continue
        reference = "".join(
            seg["ja"] for seg in json.loads(sidecar.read_text(encoding="utf-8"))["segments"]
        )
        heard = [f.get("first_source", f["source"]) for f in finals]
        first = [f.get("first_english", f["english"]) for f in finals]
        final_ja, final_en = list(heard), list(first)
        for i in range(len(finals)):
            lo = max(0, i - cfg.refine_context_lines)
            before = [Line(final_ja[j], final_en[j]) for j in range(lo, i)]
            after = [
                Line(heard[j], first[j])
                for j in range(i + 1, min(len(finals), i + 1 + cfg.refine_lookahead))
            ]
            t0 = time.perf_counter()
            revision = refiner.refine(Line(heard[i], first[i]), before, after)
            if refiner.last_verdict != "skipped":
                looked += 1
                times.append(time.perf_counter() - t0)
            if revision is not None:
                final_ja[i], final_en[i] = revision.source, revision.english
                changes.append((sample_id, heard[i], revision.source, first[i], revision.english))
        e0, n = _errors(reference, "".join(heard))
        e1, _n = _errors(reference, "".join(final_ja))
        before_err, after_err, length = before_err + e0, after_err + e1, length + n
    print(f"\n  {looked} lines checked, {len(changes)} changed")
    for sample_id, ja0, ja1, en0, en1 in changes:
        print(f"\n  [{sample_id}]")
        if ja0 != ja1:
            print(f"      JA   {ja0}\n        -> {ja1}")
        if en0 != en1:
            print(f"      EN   {en0}\n        -> {en1}")
    print(
        f"\n  Japanese errors  {before_err / length * 100:.2f}% -> {after_err / length * 100:.2f}%"
        f"   ({before_err} -> {after_err} characters)\n"
        f"  per checked line {1000 * sum(times) / max(1, len(times)):.0f} ms"
    )
    return {
        "model": model, "transcripts": label, "checked": looked, "changed": len(changes),
        "errors_before": before_err, "errors_after": after_err, "reference_chars": length,
        "changes": [list(c) for c in changes],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", nargs="*", help="Ollama tags; default: the app's catalogue")
    parser.add_argument("--first", choices=("llm", "nmt"), default="llm",
                        help="what writes the first translation")
    parser.add_argument("--refine-model", default="", help="second-look model, if different")
    parser.add_argument("--context", type=int, default=4, help="first-pass lines of context")
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--no-second-look", action="store_true", help="first pass only")
    parser.add_argument("--fix-words", action="store_true",
                        help="the model path: let it change misheard words (refine_fix_words)")
    parser.add_argument("--show", action="store_true", help="print every line that changed")
    parser.add_argument("--label", default="refine")
    parser.add_argument(
        "--transcripts", metavar="RUN",
        help="instead of the script, check the finished lines of this saved bench.run",
    )
    args = parser.parse_args(argv)
    if args.transcripts:
        results = []
        for model in args.llm or [SUGOI_14B]:
            print(f"\n== {model}  (second look over {args.transcripts}) ==")
            results.append(run_transcripts(model, args.transcripts, args))
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"{args.label}-{args.transcripts}.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

    cases = json.loads((ROOT / "refine_cases.json").read_text(encoding="utf-8"))
    lines = cases["lines"]
    hits = sum(bool(glossary.default().lookup(r["heard"])) for r in lines)
    print(f"{len(lines)} lines, {hits} with a glossary term")
    results = []
    for model in args.llm or [QWEN3_4B, SUGOI_14B]:
        print(f"\n== {model}  (first pass: {args.first}) ==")
        result = run(model, lines, args)
        results.append(result)
        print(
            f"\n  Japanese errors  as heard {result['cer_heard'] * 100:.1f}%, "
            f"slang respelt {result['cer_before'] * 100:.1f}%, "
            f"after the second look {result['cer_after'] * 100:.1f}%   "
            f"({result['lines_repaired']} lines repaired, {result['lines_damaged']} damaged)\n"
            f"  English chrF     first pass {result['chrf_before']:.1f}, "
            f"after the second look {result['chrf_after']:.1f}\n"
            f"  replies          {result['verdicts']}\n"
            f"  per line sent    {result['ms_per_line'] or 0:.0f} ms"
        )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
