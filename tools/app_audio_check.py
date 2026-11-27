"""List the apps that are playing sound, or capture one of them and report.

    .venv\\Scripts\\python.exe -m tools.app_audio_check
    .venv\\Scripts\\python.exe -m tools.app_audio_check chrome.exe [seconds]

Unlike tools.capture_check, silence here is not ambiguous: a capture of one
app comes out at wall-clock length whether or not the app is playing, so a
short capture means something is wrong and a quiet one means the app was
quiet. The last line says how much of it was silence put there by the
capture itself rather than delivered by Windows.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from subordinant import appaudio
from subordinant.audio import AudioDeviceError
from subordinant.config import SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def db(value: float) -> float:
    return 20 * np.log10(max(value, 1e-10))


def list_apps() -> int:
    if not appaudio.supported():
        print("This Windows cannot capture a single app (needs build 20348 or later).")
        return 1
    apps = appaudio.list_audio_apps()
    if not apps:
        print("No app has an audio session on the default output device.")
        return 0
    print(f"{'':8} {'exe':28} {'name':24} pid")
    for app in apps:
        state = "playing" if app.active else "quiet"
        print(f"{state:8} {app.exe:28} {app.name:24} {app.pid}")
    return 0


def capture(exe: str, seconds: float) -> int:
    cap = appaudio.ProcessCapture(
        exe, on_event=lambda kind, message: print(f"  event: {kind}: {message}")
    )
    try:
        cap.start()
    except AudioDeviceError as exc:
        print(f"FAIL: {exc}")
        return 1

    blocks: list[np.ndarray] = []
    started = time.perf_counter()
    deadline = started + seconds
    try:
        while time.perf_counter() < deadline:
            block = cap.read(timeout=0.3)
            if block is None:
                break
            if block.size:
                blocks.append(block)
    finally:
        wall = time.perf_counter() - started
        cap.stop()

    if not blocks:
        print("FAIL: no audio blocks arrived")
        return 1

    audio = np.concatenate(blocks)
    duration = audio.size / SAMPLE_RATE
    print(
        f"{cap.device_name}: {cap.native_rate} Hz, {cap.channels} ch\n"
        f"wall={wall:.2f}s captured={duration:.2f}s ({100 * duration / wall:.1f}%)\n"
        f"rms={db(float(np.sqrt(np.mean(audio**2)))):.1f} dBFS "
        f"peak={db(float(np.max(np.abs(audio)))):.1f} dBFS\n"
        f"synthesised silence={cap.silence_sec:.2f}s, of which standing in for "
        f"audio lost to late reads={cap.lost_sec:.2f}s; "
        f"dropped blocks={cap.dropped_blocks} reopens={cap.reopens}"
    )
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        return list_apps()
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    return capture(sys.argv[1], seconds)


if __name__ == "__main__":
    sys.exit(main())
