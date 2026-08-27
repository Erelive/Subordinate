"""Run the caption pipeline with plain console output, no GUI.

Useful for measuring latency and accuracy without the overlay in the way.

    .venv\\Scripts\\python.exe -m tools.console
"""

from __future__ import annotations

import sys
import time

from subordinant import logsetup
from subordinant.config import Config
from subordinant.pipeline import CaptionPipeline

logsetup.configure()


def main() -> int:
    # Optional run length, so the pipeline can be exercised unattended.
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = Config.load()
    last = ""
    transcript: list[str] = []

    def on_commit(text: str, speaker: int = -1) -> None:
        transcript.append(text)

    def on_caption(committed: str, unstable: str) -> None:
        nonlocal last
        # Show only the tail, so the line stays readable in a terminal.
        line = (committed + " " + unstable).strip()[-160:]
        if line == last:
            return
        last = line
        print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)

    def on_status(msg: str) -> None:
        print(f"[{msg}]", flush=True)

    def on_translation(source: str, english: str, speaker: int = -1) -> None:
        # One finished utterance, both languages - the pair is the point, so
        # print them together rather than interleaved with the streaming line.
        who = f"S{speaker + 1} " if speaker >= 0 else "   "
        print(f"  {who}JA  {source}", flush=True)
        print(f"  {who}EN  {english}", flush=True)

    pipeline = CaptionPipeline(
        cfg, on_caption, on_status, on_commit=on_commit, on_translation=on_translation
    )
    pipeline.start()
    print(f"running for {seconds:.0f}s" if seconds else "Ctrl+C to stop.", flush=True)
    deadline = time.monotonic() + seconds if seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        pipeline.stop()
        s = pipeline.stats
        print("\n--- final transcript ---")
        print(" ".join(transcript).strip() or "(nothing recognised)")
        print(
            f"\n{s.passes} passes, mean {s.mean_pass_ms:.0f} ms/pass, RTF {s.rtf:.2f}\n"
            f"caption lag: mean {s.mean_lag_sec:.2f}s, max {s.max_lag_sec:.2f}s "
            f"over {s.commits} commits, {s.drops} backlog drops"
        )
        if s.mt_calls:
            # Per pass, not per call, is the number that matters for keeping up.
            amortised = 1000.0 * s.total_mt_sec / s.passes if s.passes else 0.0
            print(
                f"translation: {s.mt_calls} utterances, mean {s.mean_mt_ms:.0f} ms each, "
                f"{amortised:.0f} ms/pass amortised"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
