"""Measure when each reference character is spoken, and store it in the sidecars.

    python -m bench.align_samples

Latency is "when did the text appear" minus "when was it said", and the second
half needs better than a guess. The recipe knows where each clip starts and
ends, but broadcast clips carry seconds of ambience around the words, and a
linear spread across the clip then misplaces the speech by that much.

So this runs Whisper large-v3 over each finished sample offline - whole file,
full context, word timestamps - and aligns its output to the reference text
character by character. A reference character inherits the time of the
recognised character it lines up with; ones the recogniser missed are
interpolated between their neighbours.

This uses a recogniser to produce ground truth for benchmarking recognisers,
which is only sound because of what is being borrowed: the *timing* of words
whose identity is already known from the dataset transcript. large-v3's
cross-attention timing is good to roughly 50-100 ms offline, an order of
magnitude tighter than the latencies being measured.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bench.build_samples import SAMPLES, load_sample, sample_ids
from bench.textmetrics import align, normalize_ja

ALIGNER = "large-v3"


def main() -> int:
    from subordinant import cuda

    cuda.bootstrap()
    from faster_whisper import WhisperModel

    model = WhisperModel(ALIGNER, device="cuda", compute_type="float16")
    for sample_id in sample_ids():
        audio, sidecar = load_sample(sample_id)
        segments, _info = model.transcribe(
            audio,
            language="ja",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        hyp_chars: list[str] = []
        hyp_times: list[float] = []
        for seg in segments:
            for word in seg.words or ():
                chars = normalize_ja(word.word)
                for k, ch in enumerate(chars):
                    hyp_chars.append(ch)
                    # Spread a multi-character word evenly over its span.
                    hyp_times.append(
                        word.start + (word.end - word.start) * (k + 1) / len(chars)
                    )
        hyp = "".join(hyp_chars)

        ref_chars: list[str] = []
        owner: list[int] = []
        for index, seg in enumerate(sidecar["segments"]):
            chars = normalize_ja(seg["ja"])
            ref_chars.extend(chars)
            owner.extend([index] * len(chars))
        ref = "".join(ref_chars)

        mapping, counts = align(ref, hyp)
        times = np.full(len(ref), np.nan)
        for hyp_index, ref_index in enumerate(mapping):
            if ref_index >= 0 and ref[ref_index] == hyp[hyp_index]:
                times[ref_index] = hyp_times[hyp_index]

        # Fill the characters the recogniser missed, inside each segment only,
        # and clamp to the segment so one bad match cannot drag a time into
        # the neighbouring sentence.
        matched = 0
        for index, seg in enumerate(sidecar["segments"]):
            idx = [i for i, o in enumerate(owner) if o == index]
            if not idx:
                continue
            seg_times = times[idx[0] : idx[-1] + 1].copy()
            seg_times[(seg_times < seg["start"] - 0.3) | (seg_times > seg["end"] + 0.3)] = np.nan
            known = np.flatnonzero(~np.isnan(seg_times))
            matched += known.size
            if known.size == 0:
                seg_times = np.linspace(seg["start"], seg["end"], len(idx) + 1)[1:]
            else:
                xs = np.concatenate(([-1], known, [len(idx)]))
                ys = np.concatenate(([seg["start"]], seg_times[known], [seg["end"]]))
                seg_times = np.interp(np.arange(len(idx)), xs, ys)
            # Monotonic within the segment, whatever the aligner did.
            seg_times = np.maximum.accumulate(seg_times)
            seg["char_times"] = [round(float(t), 3) for t in seg_times]
            seg["speech_start"] = round(float(seg_times[0]) - 0.15, 3)
            seg["speech_end"] = round(float(seg_times[-1]), 3)

        sidecar["aligner"] = ALIGNER
        (SAMPLES / f"{sample_id}.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        errors = counts["sub"] + counts["del"] + counts["ins"]
        print(
            f"  {sample_id}: {matched}/{len(ref)} characters timed directly, "
            f"aligner CER {errors / max(1, len(ref)):.1%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
