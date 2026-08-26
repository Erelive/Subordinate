"""Capture system audio for N seconds and report what arrived.

    .venv\\Scripts\\python.exe -m tools.capture_check [seconds]

Play something first - WASAPI loopback delivers nothing at all while the render
device is idle, so silence here is ambiguous between "broken" and "nothing on".
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from subordinant.audio import LoopbackCapture
from subordinant.config import SAMPLE_RATE
from subordinant.vad import EnergyGate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    cap = LoopbackCapture()
    cap.start()

    gate = EnergyGate()
    blocks: list[np.ndarray] = []
    active_blocks = 0
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            block = cap.read(timeout=0.3)
            if block is None:
                break
            if block.size:
                blocks.append(block)
                if gate.update(block).active:
                    active_blocks += 1
    finally:
        cap.stop()

    if not blocks:
        print("FAIL: no audio blocks arrived")
        return 1

    audio = np.concatenate(blocks)
    duration = audio.size / SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(
        f"blocks={len(blocks)} active={active_blocks} "
        f"duration={duration:.2f}s peak={20 * np.log10(max(peak, 1e-10)):.1f} dBFS "
        f"rms={20 * np.log10(max(rms, 1e-10)):.1f} dBFS dropped={cap.dropped_blocks}"
    )
    # Loopback only produces samples while something renders, so the captured
    # duration being short of the wall clock is normal, not a fault.
    print(f"wall={seconds:.1f}s captured={duration:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
