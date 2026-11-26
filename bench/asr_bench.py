"""Recognition models compared on the same audio: accuracy, and cost per pass.

    python -m bench.asr_bench                       every model in the catalogue
    python -m bench.asr_bench --models kotoba anime
    python -m bench.asr_bench --qwen                also try Qwen3-ASR (needs torch)

Two different questions, answered separately because the live pipeline cares
about both and they do not rank the models the same way.

Accuracy: each sentence of each sample is cut out by the sidecar's times and
transcribed on its own, with the decoding options the pipeline uses, then the
sample is scored as character error rate against the dataset transcript. That
is the model's ceiling at the pipeline's operating point - a few seconds of
audio per call - with segmentation taken out of the picture.

It is deliberately not a whole-file transcription. faster-whisper's long-form
mode walks a file in 30 s windows using the timestamp tokens the model emits,
and the distilled models (Kotoba, Anime-Whisper) emit few of them: they skip
whole stretches, and score 40-50% CER on a file they transcribe at 15% a
sentence at a time. The live pipeline never puts them in that mode, so a
number taken there would rank the models by something that does not happen.
--long-form reports it anyway, for the record.

Cost: the pipeline does not transcribe a file, it re-transcribes a rolling
buffer several times a second. What decides whether a model can keep up is the
wall time of one pass over a buffer of N seconds, word timestamps on. That is
measured directly, on real speech, at several buffer lengths.

Results: bench/results/asr.json, and a table on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from bench.build_samples import load_sample, sample_ids
from bench.run import gpu_mem_now
from bench.textmetrics import cer

RESULTS = Path(__file__).resolve().parent / "results"
SR = 16000

# short name -> (model id, beam, no_repeat_ngram_size)
MODELS = {
    "kotoba": ("kotoba-tech/kotoba-whisper-v2.0-faster", 5, 0),
    "anime": ("flyfront/anime-whisper-faster", 5, 5),
    "turbo": ("large-v3-turbo", 5, 0),
    "large-v3": ("large-v3", 5, 0),
}


def _transcribe(model, audio, beam, no_repeat, words=False, vad=True, max_new_tokens=None):
    kwargs = dict(
        language="ja",
        task="transcribe",
        beam_size=beam,
        no_repeat_ngram_size=no_repeat,
        word_timestamps=words,
        vad_filter=vad,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    if max_new_tokens:
        kwargs["max_new_tokens"] = max_new_tokens
    segments, _info = model.transcribe(audio, **kwargs)
    return "".join(seg.text for seg in segments)


def bench_whisper(
    name: str, samples: list[str], beams: list[int], long_form: bool = False
) -> dict:
    from subordinant import cuda
    from subordinant.asr import repair_alignment_heads, repair_tokenizer, resolve_model

    cuda.bootstrap()
    from faster_whisper import WhisperModel

    model_id, beam, no_repeat = MODELS[name]
    repair_alignment_heads(model_id)
    repair_tokenizer(model_id)
    before = gpu_mem_now()
    t0 = time.perf_counter()
    model = WhisperModel(resolve_model(model_id), device="cuda", compute_type="float16")
    load_sec = time.perf_counter() - t0
    _transcribe(model, np.zeros(SR, dtype=np.float32), 1, 0)  # warm
    out: dict = {
        "model": model_id,
        "beam": beam,
        "mode": "long-form" if long_form else "per-sentence",
        "load_sec": round(load_sec, 2),
        "samples": {},
    }

    total_err = total_ref = 0
    total_audio = total_time = 0.0
    for sample_id in samples:
        audio, sidecar = load_sample(sample_id)
        ref = "".join(s["ja"] for s in sidecar["segments"])
        t0 = time.perf_counter()
        if long_form:
            hyp = _transcribe(model, audio, beam, no_repeat)
        else:
            parts = []
            for seg in sidecar["segments"]:
                a = max(0, int((seg["start"] - 0.2) * SR))
                b = min(audio.size, int((seg["end"] + 0.2) * SR))
                parts.append(_transcribe(model, audio[a:b], beam, no_repeat, vad=False))
            hyp = "".join(parts)
        elapsed = time.perf_counter() - t0
        score = cer(ref, hyp)
        errors = score["sub"] + score["del"] + score["ins"]
        total_err += errors
        total_ref += score["ref_len"]
        total_audio += audio.size / SR
        total_time += elapsed
        out["samples"][sample_id] = {
            "cer": round(score["cer"], 4),
            "sec": round(elapsed, 2),
            "hyp": hyp,
        }
        print(f"  {name:9s} {sample_id:22s} CER {score['cer']:6.1%}  {elapsed:5.1f}s", flush=True)
    out["cer"] = round(total_err / max(1, total_ref), 4)
    out["offline_rtf"] = round(total_time / total_audio, 4)

    # Cost of one streaming pass, by buffer length, on continuous real speech.
    audio, _sidecar = load_sample("08_long_continuous")
    passes: dict = {}
    for b in beams:
        row = {}
        for seconds in (2, 4, 8, 12):
            times = []
            for k in range(6):
                start = int((5 + k * 17) * SR)
                window = audio[start : start + seconds * SR]
                t0 = time.perf_counter()
                # The pipeline's own call: word timestamps on, and no inner
                # VAD because the stream is gated before it gets here.
                _transcribe(model, window, b, no_repeat, words=True, vad=False)
                times.append(time.perf_counter() - t0)
            row[str(seconds)] = round(1000 * statistics.median(times[1:]), 1)
        passes[str(b)] = row
        print(f"  {name:9s} beam {b}: pass ms by buffer {row}", flush=True)
    out["pass_ms"] = passes
    after = gpu_mem_now()
    if before is not None and after is not None:
        out["gpu_mem_added_mib"] = after - before
    del model
    return out


def bench_qwen(samples: list[str], repo: str) -> dict | None:
    """Qwen3-ASR through transformers, if this environment can run it."""
    try:
        import torch
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        print(f"  qwen3-asr: skipped - {type(exc).__name__}: {exc}")
        return {"skipped": f"{type(exc).__name__}: {exc}"}

    before = gpu_mem_now()
    t0 = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(repo)
        model = AutoModelForMultimodalLM.from_pretrained(
            repo, device_map="cuda", dtype=torch.bfloat16
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  qwen3-asr: skipped - {type(exc).__name__}: {exc}")
        return {"skipped": f"{type(exc).__name__}: {str(exc)[:300]}"}
    out: dict = {
        "model": repo,
        "transformers": transformers.__version__,
        "load_sec": round(time.perf_counter() - t0, 2),
        "samples": {},
    }

    forced = {"language": "ja"}

    def run(chunk: np.ndarray) -> str:
        try:
            inputs = processor.apply_transcription_request(
                audio=chunk, sampling_rate=SR, **forced
            )
        except ValueError:
            # Some fine-tunes ship a chat template without the language slot.
            # Let the model identify the language itself, as its card does.
            forced.clear()
            inputs = processor.apply_transcription_request(audio=chunk, sampling_rate=SR)
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ids = ids[:, inputs["input_ids"].shape[1] :]
        return processor.decode(ids, return_format="transcription_only")[0]

    run(np.zeros(SR, dtype=np.float32))
    total_err = total_ref = 0
    total_audio = total_time = 0.0
    for sample_id in samples:
        audio, sidecar = load_sample(sample_id)
        ref = "".join(s["ja"] for s in sidecar["segments"])
        # No long-form mode is assumed: feed it the sentences one at a time,
        # cut where the sidecar says they are. That is kinder to it than the
        # whole-file run the Whisper models get, so read the CER as a ceiling.
        t0 = time.perf_counter()
        parts = []
        for seg in sidecar["segments"]:
            a = max(0, int((seg["start"] - 0.2) * SR))
            b = min(audio.size, int((seg["end"] + 0.2) * SR))
            parts.append(run(audio[a:b]))
        elapsed = time.perf_counter() - t0
        hyp = "".join(parts)
        score = cer(ref, hyp)
        total_err += score["sub"] + score["del"] + score["ins"]
        total_ref += score["ref_len"]
        total_audio += audio.size / SR
        total_time += elapsed
        out["samples"][sample_id] = {"cer": round(score["cer"], 4), "sec": round(elapsed, 2), "hyp": hyp}
        print(f"  qwen3-asr {sample_id:22s} CER {score['cer']:6.1%}  {elapsed:5.1f}s", flush=True)
    out["cer"] = round(total_err / max(1, total_ref), 4)
    out["offline_rtf"] = round(total_time / total_audio, 4)

    audio, _sidecar = load_sample("08_long_continuous")
    row = {}
    for seconds in (2, 4, 8, 12):
        times = []
        for k in range(5):
            start = int((5 + k * 17) * SR)
            t0 = time.perf_counter()
            run(audio[start : start + seconds * SR])
            times.append(time.perf_counter() - t0)
        row[str(seconds)] = round(1000 * statistics.median(times[1:]), 1)
    out["pass_ms"] = {"greedy": row}
    print(f"  qwen3-asr pass ms by buffer {row}", flush=True)
    after = gpu_mem_now()
    if before is not None and after is not None:
        out["gpu_mem_added_mib"] = after - before
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--beams", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--long-form", action="store_true",
                        help="transcribe each sample whole instead of per sentence")
    parser.add_argument("--beam", type=int, help="override the catalogue beam for accuracy")
    parser.add_argument("--qwen-repo", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--out", default="asr")
    args = parser.parse_args(argv)

    samples = args.samples or sample_ids()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.out}.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in args.models:
        if args.beam:
            MODELS[name] = (MODELS[name][0], args.beam, MODELS[name][2])
        key = name + (f"-beam{args.beam}" if args.beam else "") + ("-longform" if args.long_form else "")
        results[key] = bench_whisper(name, samples, args.beams, args.long_form)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.qwen:
        key = "qwen3-asr" if args.qwen_repo.startswith("Qwen/") else args.qwen_repo.split("/")[-1]
        results[key] = bench_qwen(samples, args.qwen_repo)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'model':12s} {'CER':>7s} {'offline RTF':>12s} {'VRAM':>7s}   pass ms @ 2/4/8/12 s")
    for name, r in results.items():
        if not r or "cer" not in r:
            continue
        first = next(iter(r.get("pass_ms", {}).values()), {})
        cost = " / ".join(str(first.get(k, "-")) for k in ("2", "4", "8", "12"))
        print(
            f"{name:12s} {r['cer']:7.1%} {r['offline_rtf']:12.3f} "
            f"{r.get('gpu_mem_added_mib', 0):6d}M   {cost}"
        )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
