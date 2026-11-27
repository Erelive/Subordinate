"""Run the caption pipeline with plain console output, no GUI.

Useful for measuring latency and accuracy on real audio without the window in
the way: it listens to whatever the PC is playing, with the saved settings.

    .venv\\Scripts\\python.exe -m tools.console          # until Ctrl+C
    .venv\\Scripts\\python.exe -m tools.console 30       # for 30 seconds
    .venv\\Scripts\\python.exe -m tools.console 30 fast  # with a profile
"""

from __future__ import annotations

import sys
import time

from subordinant import logsetup
from subordinant.config import Config
from subordinant.pipeline import CaptionPipeline, PipelineEvents
from subordinant.profiles import PROFILES, apply_profile

logsetup.configure()


class _Console(PipelineEvents):
    def __init__(self) -> None:
        self.last = ""
        self.lines: list[str] = []

    def on_state(self, state: str, detail: str) -> None:
        if state in ("loading", "error", "paused", "idle"):
            print(f"[{state}{': ' + detail if detail else ''}]", flush=True)

    def on_notice(self, level: str, message: str) -> None:
        print(f"[{level}] {message}", flush=True)

    def on_caption(self, committed: str, unstable: str) -> None:
        # Show only the tail, so the line stays readable in a terminal. The
        # guess is bracketed: it is the part that may still change.
        line = (committed + (f" ({unstable})" if unstable else "")).strip()[-160:]
        if line and line != self.last:
            self.last = line
            print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)

    def on_utterance(self, utterance) -> None:
        if not utterance.final:
            return
        # One finished line, both languages - the pair is the point, so they
        # are printed together rather than interleaved with the live one.
        who = f"S{utterance.speaker + 1} " if utterance.speaker >= 0 else "   "
        delay = f"  [{utterance.latency:.2f}s]" if utterance.latency is not None else ""
        if utterance.source:
            print(f"  {who}JA  {utterance.source}", flush=True)
            self.lines.append(utterance.source)
        if utterance.english:
            print(f"  {who}EN  {utterance.english}{delay}", flush=True)


    def on_revision(self, utterance) -> None:
        if utterance.first_source != utterance.source:
            print(f"   *  JA  {utterance.first_source}  ->  {utterance.source}", flush=True)
        if utterance.first_english != utterance.english:
            print(f"   *  EN  {utterance.first_english}  ->  {utterance.english}", flush=True)


def main() -> int:
    # Optional run length, so the pipeline can be exercised unattended.
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = Config.load()
    if len(sys.argv) > 2:
        if sys.argv[2] not in PROFILES:
            print(f"unknown profile {sys.argv[2]!r}; expected one of {', '.join(PROFILES)}")
            return 2
        apply_profile(cfg, sys.argv[2])

    events = _Console()
    pipeline = CaptionPipeline(cfg, events=events)
    pipeline.start()
    print(f"running for {seconds:.0f}s" if seconds else "Ctrl+C to stop.", flush=True)
    deadline = time.monotonic() + seconds if seconds else None
    try:
        while pipeline.running and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        pipeline.stop()
        s = pipeline.metrics.snapshot()
        print("\n--- final transcript ---")
        print(" ".join(events.lines).strip() or "(nothing recognised)")

        def show(value, unit: str = "") -> str:
            return "-" if value is None else f"{value}{unit}"

        print(
            f"\n{s['engine']}  ->  {s['translator']}\n"
            f"{s['passes']} passes, mean {show(s['mean_pass_ms'], ' ms')}, RTF {s['rtf']:.2f}, "
            f"{s['drops']} backlog drops\n"
            f"Japanese settled {show(s['source_lag_sec'], ' s')} after each word; "
            f"finished English {show(s['final_latency_sec'], ' s')} after each sentence "
            f"(worst 1 in 10: {show(s['final_latency_p90_sec'], ' s')})\n"
            f"{s['utterances']} lines: {s['eager_endpoints'] + s['pause_splits']} finished "
            f"early, {s['deferred_closes']} pauses waited out, {s['revised_lines']} "
            f"corrections on re-reading\n"
            f"second look: {s['second_look_changes']} lines corrected; "
            f"audio dropped: {s.get('capture_dropped_sec', 0.0):.2f} s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
