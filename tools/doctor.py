"""Environment check: audio devices, GPU, model load, rough inference speed.

Run this before anything else. It answers the two questions that decide whether
the app can work on this machine: can we open a loopback device, and is the GPU
fast enough to transcribe faster than real time.

    .venv\\Scripts\\python.exe -m tools.doctor
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from subordinant.config import SAMPLE_RATE, Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def check_audio() -> bool:
    import pyaudiowpatch as pyaudio

    from subordinant.audio import AudioDeviceError, find_default_loopback

    pa = pyaudio.PyAudio()
    try:
        loopbacks = list(pa.get_loopback_device_info_generator())
        print(f"\nWASAPI loopback devices ({len(loopbacks)}):")
        for d in loopbacks:
            print(
                f"  [{d['index']:>3}] {d['name']}  "
                f"{int(d['defaultSampleRate'])} Hz  {d['maxInputChannels']} ch"
            )
        try:
            default = find_default_loopback(pa)
        except AudioDeviceError as exc:
            print(f"  FAIL: {exc}")
            return False
        print(f"\n  default -> [{default['index']}] {default['name']}")
        return True
    finally:
        pa.terminate()


def check_capture(seconds: float = 3.0) -> bool:
    from subordinant.audio import LoopbackCapture

    print(f"\nCapturing {seconds:.0f}s of system audio (play something now)...")
    cap = LoopbackCapture()
    try:
        cap.start()
    except Exception as exc:  # noqa: BLE001 - report any device failure verbatim
        print(f"  FAIL: {exc}")
        return False

    collected: list[np.ndarray] = []
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            block = cap.read(timeout=0.5)
            if block is None:
                break
            if block.size:
                collected.append(block)
    finally:
        cap.stop()

    if not collected:
        print("  FAIL: no audio blocks received")
        return False

    audio = np.concatenate(collected)
    rms = float(np.sqrt(np.mean(audio**2)))
    db = 20 * np.log10(max(rms, 1e-10))
    print(
        f"  got {audio.size / SAMPLE_RATE:.2f}s at {SAMPLE_RATE} Hz, "
        f"level {db:.1f} dBFS, dropped {cap.dropped_blocks} blocks"
    )
    if db < -70:
        print("  WARN: effectively silent - was anything playing?")
    return True


def check_model(cfg: Config) -> bool:
    from subordinant.asr import Transcriber

    print(f"\nLoading {cfg.model} ({cfg.compute_type}) on {cfg.device}...")
    print("  first run downloads ~1.6 GB from Hugging Face")
    try:
        asr = Transcriber(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")
        return False
    print(f"  running on: {asr.device}")
    if asr.device != cfg.device:
        print(f"  WARN: wanted {cfg.device}, fell back to {asr.device}")

    asr.warmup()

    # Rough speed check. Synthetic audio decodes to little text, so this
    # measures encoder throughput and understates a real workload - treat it as
    # an upper bound on speed, not a promise.
    for seconds in (5.0, 15.0):
        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.02).astype(np.float32)
        t0 = time.perf_counter()
        asr.transcribe(audio)
        elapsed = time.perf_counter() - t0
        print(
            f"  {seconds:>4.0f}s buffer -> {elapsed * 1000:6.0f} ms  "
            f"(RTF {elapsed / seconds:.3f})"
        )
    return True


def main() -> int:
    cfg = Config.load()
    ok = check_audio()
    ok = check_capture() and ok
    ok = check_model(cfg) and ok
    print("\nOK" if ok else "\nSome checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
