"""Translation engines compared on the same Japanese: quality, and time per line.

    python -m bench.mt_bench                      built-in NMT + every installed LLM
    python -m bench.mt_bench --llm qwen3:4b-instruct gemma4:12b-it-qat
    python -m bench.mt_bench --no-nmt

Every sentence in bench/manifest.json is translated by each engine and scored
with chrF against the manifest's English reference. Sentences are fed in the
order they occur in the samples, so an engine that uses context gets the
context it would get live.

Each engine is run on two versions of the input:

  clean      the dataset transcript, punctuated - the engine's ceiling.
  asr-like   the same text with punctuation stripped, which is what
             Kotoba-Whisper actually hands the translator.

The gap between the two is how much an engine depends on punctuation it will
not be given.

chrF with one reference is a blunt instrument: it rewards matching the
reference's word choices and cannot tell a good paraphrase from a bad one.
It is used here to rank engines against each other on identical input, where
that bias is the same for all of them. Read the outputs too - they are in
bench/results/mt.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from bench.build_samples import load_manifest
from bench.run import gpu_mem_now
from bench.textmetrics import chrf

RESULTS = Path(__file__).resolve().parent / "results"

_PUNCT = "。、，,！!？?「」『』…‥"

# The prompt the app used before the overhaul, kept to check the new one is
# an improvement rather than a change.
OLD_PROMPT = """You are a live subtitle translator. Translate each Japanese line into natural English.

Rules:
- Output ONLY the English translation. No notes, no romaji, no quotes, no explanation.
- The line comes from live speech and may be cut off mid-sentence. Translate exactly what is there and stop. Never invent an ending.
- Keep the speaker's register: casual speech stays casual.
- Render names consistently with how you rendered them in earlier lines.
- If the line carries no translatable content, output nothing at all."""


def sentences() -> list[dict]:
    """Every (ja, en) pair, in sample order, each sample a separate conversation."""
    manifest = load_manifest()
    out = []
    seen: set[tuple[str, str]] = set()
    for spec in manifest["samples"]:
        for item in spec["items"]:
            clip = item if isinstance(item, str) else item.get("clip")
            if clip is None:
                continue
            key = (spec["id"], clip)
            if key in seen:
                continue
            seen.add(key)
            meta = manifest["clips"][clip]
            out.append({"sample": spec["id"], "clip": clip, "ja": meta["ja"], "en": meta["en"]})
    return out


def strip_punct(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT)


def run_engine(label: str, translate, reset, rows: list[dict], variant: str) -> dict:
    outputs, times, scores = [], [], []
    current = None
    for row in rows:
        if row["sample"] != current:
            current = row["sample"]
            reset()
        source = row["ja"] if variant == "clean" else strip_punct(row["ja"])
        t0 = time.perf_counter()
        english = translate(source)
        times.append(time.perf_counter() - t0)
        outputs.append({"ja": source, "en": english, "ref": row["en"]})
        scores.append(chrf(english, row["en"]))
    ordered = sorted(times)
    result = {
        "chrf": round(statistics.mean(scores), 2),
        "chrf_corpus": round(
            chrf(" ".join(o["en"] for o in outputs), " ".join(o["ref"] for o in outputs)), 2
        ),
        "empty": sum(1 for o in outputs if not o["en"]),
        "mean_ms": round(1000 * statistics.mean(times), 1),
        "p50_ms": round(1000 * ordered[len(ordered) // 2], 1),
        "p90_ms": round(1000 * ordered[int(len(ordered) * 0.9)], 1),
        "max_ms": round(1000 * ordered[-1], 1),
        "outputs": outputs,
    }
    print(
        f"  {label:38s} {variant:8s} chrF {result['chrf']:5.1f}  "
        f"mean {result['mean_ms']:6.0f} ms  p90 {result['p90_ms']:6.0f} ms  "
        f"empty {result['empty']}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", nargs="*", help="Ollama tags; default: the app's catalogue")
    parser.add_argument("--no-nmt", action="store_true")
    parser.add_argument("--old-prompt", action="store_true", help="also run the previous prompt")
    parser.add_argument("--context", nargs="*", type=int, default=None,
                        help="context sizes to try per LLM (default: the app's)")
    parser.add_argument("--out", default="mt")
    args = parser.parse_args(argv)

    from subordinant.config import Config

    rows = sentences()
    print(f"{len(rows)} sentences")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.out}.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save() -> None:
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.no_nmt:
        from subordinant.translate import Translator

        cfg = Config()
        before = gpu_mem_now()
        nmt = Translator(cfg)
        nmt.warmup()
        added = (gpu_mem_now() or 0) - (before or 0)
        for beam in (1, 2, 5, 10):
            label = f"sugoi-v4 beam {beam}"
            results[label] = {"vram_added_mib": added}
            for variant in ("clean", "asr-like"):
                results[label][variant] = run_engine(
                    label, lambda s, b=beam: nmt.translate(s, beam_size=b), lambda: None, rows, variant
                )
            save()
        nmt.close()

    from subordinant.llm_translate import LLMTranslator, installed_models
    from subordinant.profiles import TRANSLATORS
    from subordinant.translate import TranslationUnavailable

    cfg = Config()
    tags = args.llm
    if tags is None:
        tags = [e["needs"] for e in TRANSLATORS.values() if e.get("needs")]
    have = set(installed_models(cfg.llm_endpoint))
    for tag in tags:
        if tag not in have and f"{tag}:latest" not in have:
            print(f"  {tag}: not installed in Ollama, skipped")
            continue
        contexts = args.context if args.context is not None else [cfg.llm_context_utterances]
        for context in contexts:
            for prompt_name in (["new", "old"] if args.old_prompt else ["new"]):
                cfg = Config()
                cfg.mt_backend = "llm"
                cfg.llm_model = tag
                cfg.llm_context_utterances = context
                before = gpu_mem_now()
                try:
                    llm = LLMTranslator(cfg)
                except TranslationUnavailable as exc:
                    print(f"  {tag}: {exc}")
                    break
                if prompt_name == "old":
                    llm._system = OLD_PROMPT
                added = (gpu_mem_now() or 0) - (before or 0)
                label = f"{tag} ctx{context}" + (" old-prompt" if prompt_name == "old" else "")
                results[label] = {"vram_added_mib": added}
                for variant in ("clean", "asr-like"):
                    results[label][variant] = run_engine(
                        label, llm.translate, llm.reset, rows, variant
                    )
                save()
        # Make room for the next model: two 9 GB models do not share a card.
        try:
            llm.unload()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)

    print(f"\n{'engine':40s} {'clean':>6s} {'asr-like':>9s} {'mean ms':>8s} {'p90 ms':>7s}")
    for label, r in results.items():
        if "clean" not in r:
            continue
        print(
            f"{label:40s} {r['clean']['chrf']:6.1f} {r['asr-like']['chrf']:9.1f} "
            f"{r['asr-like']['mean_ms']:8.0f} {r['asr-like']['p90_ms']:7.0f}"
        )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
