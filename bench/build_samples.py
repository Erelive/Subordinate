"""Build the benchmark WAVs described by bench/manifest.json.

    python -m bench.build_samples [--force]

Every clip is real recorded Japanese with a published transcript, pulled from
four public datasets by (file, row) so the result is the same everywhere. Only
the parquet row groups that hold a wanted row are read - a few tens of MB over
HTTP range requests, or nothing at all if the files are already in the Hugging
Face cache.

Each sample is written as 16 kHz mono PCM beside a JSON sidecar giving, for
every sentence in it, when it starts and ends in the file, the Japanese that
was said and an English reference. That sidecar is the ground truth the other
bench scripts score against: accuracy from the text, latency from the times.

Nothing here is random at run time. Gaps come from a seeded generator, the
time-stretch is a deterministic WSOLA, and the "music" is synthesised from
sines so no copyrighted audio is involved.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
# Clip audio decoded once and kept, so rebuilding after a recipe change does
# not go back to the network.
CLIP_CACHE = ROOT / ".cache"

# Loudness every clip is brought to before mixing. The datasets differ by
# 20 dB between them, and a pipeline's gate and VAD should be judged on speech
# at a consistent level rather than on who recorded quietest.
TARGET_RMS_DB = -23.0


# -- reading clips ----------------------------------------------------------


def _open_parquet(repo: str, filename: str):
    """A pyarrow ParquetFile for a dataset file, local if cached."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, try_to_load_from_cache

    cached = try_to_load_from_cache(repo, filename, repo_type="dataset")
    if isinstance(cached, str):
        return pq.ParquetFile(cached), None
    fs = HfFileSystem()
    handle = fs.open(f"datasets/{repo}/{filename}", "rb")
    return pq.ParquetFile(handle), handle


def _decode(data: bytes) -> np.ndarray:
    """Any container PyAV can read -> mono float32 at 16 kHz."""
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(data), sampling_rate=SR).astype(np.float32)


def load_clips(manifest: dict) -> dict[str, np.ndarray]:
    """Decoded, trimmed, level-matched audio for every clip in the manifest."""
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    clips: dict[str, np.ndarray] = {}
    wanted: dict[str, list[tuple[str, int]]] = {}
    for clip_id, spec in manifest["clips"].items():
        cached = CLIP_CACHE / f"{clip_id}.npy"
        if cached.exists():
            clips[clip_id] = np.load(cached)
        else:
            wanted.setdefault(spec["source"], []).append((clip_id, spec["row"]))

    for source, rows in wanted.items():
        info = manifest["sources"][source]
        print(f"reading {len(rows)} clips from {info['repo']} ...", flush=True)
        pf, handle = _open_parquet(info["repo"], info["file"])
        try:
            # Row -> (row group, offset within it).
            starts = [0]
            for g in range(pf.num_row_groups):
                starts.append(starts[-1] + pf.metadata.row_group(g).num_rows)
            by_group: dict[int, list[tuple[str, int]]] = {}
            for clip_id, row in rows:
                group = next(g for g in range(pf.num_row_groups) if row < starts[g + 1])
                by_group.setdefault(group, []).append((clip_id, row - starts[group]))
            for group, members in sorted(by_group.items()):
                table = pf.read_row_group(group, columns=["audio", "transcription"])
                audio_col = table.column("audio")
                text_col = table.column("transcription")
                for clip_id, offset in members:
                    expected = manifest["clips"][clip_id]["ja"]
                    got = text_col[offset].as_py().strip()
                    # Common Voice transcripts carry a stray trailing ".".
                    if got.rstrip(".") != expected.rstrip("."):
                        raise SystemExit(
                            f"{clip_id}: the dataset row says {got!r} but the "
                            f"manifest expects {expected!r} - the dataset "
                            "changed, or the row index is wrong"
                        )
                    audio = _decode(audio_col[offset].as_py()["bytes"])
                    audio = level(trim_silence(audio))
                    np.save(CLIP_CACHE / f"{clip_id}.npy", audio)
                    clips[clip_id] = audio
        finally:
            if handle is not None:
                handle.close()
    return clips


# -- signal helpers ---------------------------------------------------------


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = x.size // frame
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1))


def trim_silence(x: np.ndarray, pad_ms: int = 80) -> np.ndarray:
    """Cut leading and trailing quiet, keeping a short pad either side.

    The datasets pad clips by anything from nothing to two seconds. Trimming
    is what makes the recipe's gap figures the real gaps, and the sidecar's
    start and end times the real edges of speech.
    """
    frame = SR // 50  # 20 ms
    rms = _frame_rms(x, frame)
    if rms.size == 0:
        return x
    threshold = max(float(rms.max()) * 10 ** (-32 / 20), 1e-4)
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return x
    pad = SR * pad_ms // 1000
    start = max(0, int(loud[0]) * frame - pad)
    end = min(x.size, (int(loud[-1]) + 1) * frame + pad)
    return x[start:end]


def level(x: np.ndarray) -> np.ndarray:
    """Scale to TARGET_RMS_DB measured over the frames that carry speech."""
    rms = _frame_rms(x, SR // 50)
    if rms.size == 0 or rms.max() <= 0:
        return x
    active = rms[rms > rms.max() * 0.1]
    current = float(np.sqrt(np.mean(np.square(active))))
    gain = 10 ** (TARGET_RMS_DB / 20) / max(current, 1e-6)
    out = x * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.97:
        out *= 0.97 / peak
    return out.astype(np.float32)


def time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    """Change speed without changing pitch (WSOLA). rate > 1 is faster.

    Resampling would be simpler and would also turn every speaker into a
    chipmunk, which is not what fast speech is. WSOLA slides each analysis
    window by up to 10 ms to line it up with the waveform already written, so
    pitch periods stay intact.
    """
    if abs(rate - 1.0) < 1e-3:
        return x
    win = SR * 30 // 1000
    hop_out = win // 2
    hop_in = int(round(hop_out * rate))
    tol = SR * 10 // 1000
    window = np.hanning(win).astype(np.float32)

    n_out = int(x.size / rate)
    out = np.zeros(n_out + win, dtype=np.float32)
    norm = np.zeros(n_out + win, dtype=np.float32)
    padded = np.pad(x, (tol, win + tol + hop_out))

    pos_in, pos_out = tol, 0
    follow: np.ndarray | None = None
    while pos_out + win <= out.size and pos_in + win + tol <= padded.size:
        delta = 0
        if follow is not None:
            region = padded[pos_in - tol : pos_in + tol + win]
            delta = int(np.argmax(np.correlate(region, follow, mode="valid"))) - tol
        start = pos_in + delta
        out[pos_out : pos_out + win] += padded[start : start + win] * window
        norm[pos_out : pos_out + win] += window
        follow = padded[start + hop_out : start + hop_out + win]
        if follow.size < win:
            break
        pos_in += hop_in
        pos_out += hop_out
    out /= np.maximum(norm, 1e-3)
    return out[:n_out]


def split_point(x: np.ndarray) -> int:
    """The quietest moment in the middle half of a clip - where a breath is."""
    frame = SR * 40 // 1000
    rms = _frame_rms(x, frame)
    lo, hi = int(rms.size * 0.25), int(rms.size * 0.75)
    if hi <= lo:
        return x.size // 2
    return (lo + int(np.argmin(rms[lo:hi]))) * frame + frame // 2


def music(seconds: float, seed: int = 7) -> np.ndarray:
    """Synthetic backing track: pad chords, a bass line and a soft beat.

    Deliberately busy in the 200 Hz - 3 kHz band where speech lives, because
    that is what makes real background music hard on a recogniser. Unit-ish
    level; the caller sets the mix.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    bpm = 100.0
    beat = 60.0 / bpm
    # Am - F - C - G, one bar each.
    chords = [
        (220.00, 261.63, 329.63),
        (174.61, 220.00, 261.63),
        (261.63, 329.63, 392.00),
        (196.00, 246.94, 293.66),
    ]
    out = np.zeros(n, dtype=np.float64)
    bar = beat * 4
    for i in range(int(seconds / bar) + 1):
        a, b = int(i * bar * SR), min(n, int((i + 1) * bar * SR))
        if a >= n:
            break
        seg_t = t[a:b] - i * bar
        env = np.minimum(1.0, seg_t / 0.08) * np.minimum(1.0, (bar - seg_t) / 0.15)
        chord = chords[i % len(chords)]
        for f in chord:
            for harmonic, amp in ((1, 0.5), (2, 0.22), (3, 0.12), (4, 0.06)):
                out[a:b] += amp * env * np.sin(2 * np.pi * f * harmonic * seg_t)
        # Bass on the root, an octave down, on every beat.
        root = chord[0] / 2
        for k in range(4):
            ba = a + int(k * beat * SR)
            bb = min(b, ba + int(beat * 0.9 * SR))
            if ba >= bb:
                continue
            bt = np.arange(bb - ba) / SR
            out[ba:bb] += 0.8 * np.exp(-bt * 3.0) * np.sin(2 * np.pi * root * bt)
    # Hi-hat on the off-beats: filtered noise bursts.
    hat = int(0.05 * SR)
    for k in range(int(seconds / (beat / 2))):
        ha = int(k * beat / 2 * SR)
        if ha + hat >= n:
            break
        burst = rng.standard_normal(hat) * np.exp(-np.arange(hat) / (0.012 * SR))
        out[ha : ha + hat] += 0.25 * np.diff(burst, prepend=0.0)
    out /= max(1e-9, float(np.max(np.abs(out))))
    return out.astype(np.float32)


# -- assembly ---------------------------------------------------------------


def build_sample(spec: dict, clips: dict[str, np.ndarray], manifest: dict):
    """Lay one recipe out on a timeline. Returns (audio, segments)."""
    # Seeded per sample id, so adding a sample does not reshuffle the others.
    rng = np.random.default_rng(abs(hash_id(spec["id"])) % (2**32))
    tempo = float(spec.get("tempo", 1.0))
    gap_lo, gap_hi = spec.get("gap", [0.5, 0.8])
    snr = spec.get("music_snr_db")

    parts: list[np.ndarray] = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    cursor = parts[0].size
    segments: list[dict] = []
    music_spans: list[tuple[int, int]] = []

    def add(chunk: np.ndarray) -> None:
        nonlocal cursor
        parts.append(chunk)
        cursor += chunk.size

    for item in spec["items"]:
        if isinstance(item, dict) and "silence" in item:
            add(np.zeros(int(item["silence"] * SR), dtype=np.float32))
            continue
        if isinstance(item, dict) and "music" in item:
            n = int(item["music"] * SR)
            music_spans.append((cursor, cursor + n))
            add(np.zeros(n, dtype=np.float32))
            continue

        clip_id = item if isinstance(item, str) else item["clip"]
        audio = time_stretch(clips[clip_id], tempo)
        pause = item.get("split_pause") if isinstance(item, dict) else None
        start = cursor
        pause_at = None
        if pause:
            cut = split_point(audio)
            pause_at = (start + cut) / SR
            fade = SR // 100
            head, tail = audio[:cut].copy(), audio[cut:].copy()
            head[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            tail[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            add(head)
            add(np.zeros(int(pause * SR), dtype=np.float32))
            add(tail)
        else:
            add(audio)
        meta = manifest["clips"][clip_id]
        segment = {
            "clip": clip_id,
            "start": round(start / SR, 3),
            "end": round(cursor / SR, 3),
            "ja": meta["ja"],
            "en": meta["en"],
        }
        if pause_at is not None:
            segment["pause_at"] = round(pause_at, 3)
            segment["pause_sec"] = pause
        segments.append(segment)
        add(np.zeros(int(rng.uniform(gap_lo, gap_hi) * SR), dtype=np.float32))

    add(np.zeros(int(1.5 * SR), dtype=np.float32))
    audio = np.concatenate(parts)

    speech_rms = 10 ** (TARGET_RMS_DB / 20)
    if snr is not None:
        bed = music(audio.size / SR)[: audio.size]
        bed *= speech_rms * 10 ** (-snr / 20) / max(1e-9, float(np.sqrt(np.mean(bed**2))))
        audio = audio + bed
    for a, b in music_spans:
        bed = music((b - a) / SR, seed=11)[: b - a]
        bed *= speech_rms / max(1e-9, float(np.sqrt(np.mean(bed**2))))
        fade = SR // 5
        bed[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        bed[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        audio[a:b] += bed

    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32), segments, [
        (round(a / SR, 3), round(b / SR, 3)) for a, b in music_spans
    ]


def hash_id(text: str) -> int:
    """A stable hash. Python's own is salted per process, which would make the
    gaps differ between two runs of this script."""
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SR or wav.getnchannels() != 1:
            raise ValueError(f"{path} is not 16 kHz mono")
        data = wav.readframes(wav.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def load_manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def load_sample(sample_id: str) -> tuple[np.ndarray, dict]:
    """(audio, sidecar) for a built sample. Raises if it has not been built."""
    wav = SAMPLES / f"{sample_id}.wav"
    if not wav.exists():
        raise FileNotFoundError(
            f"{wav} is missing - run: python -m bench.build_samples"
        )
    meta = json.loads((SAMPLES / f"{sample_id}.json").read_text(encoding="utf-8"))
    return read_wav(wav), meta


def sample_ids() -> list[str]:
    return [s["id"] for s in load_manifest()["samples"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild existing WAVs")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    clips = load_clips(manifest)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for spec in manifest["samples"]:
        wav = SAMPLES / f"{spec['id']}.wav"
        if wav.exists() and not args.force:
            print(f"  {spec['id']}: already built")
            continue
        audio, segments, music_spans = build_sample(spec, clips, manifest)
        write_wav(wav, audio)
        sidecar = {
            "id": spec["id"],
            "title": spec["title"],
            "about": spec["about"],
            "duration": round(audio.size / SR, 3),
            "segments": segments,
            "music_only": music_spans,
        }
        (SAMPLES / f"{spec['id']}.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        speech = sum(s["end"] - s["start"] for s in segments)
        print(
            f"  {spec['id']}: {audio.size / SR:6.1f}s, {len(segments)} sentences, "
            f"{speech:5.1f}s of speech"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
