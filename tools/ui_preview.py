"""Render the interface to PNG files without running the pipeline.

    .venv\\Scripts\\python.exe -m tools.ui_preview [output-dir] [dark|light]

Fills the window with a scripted conversation and photographs each page, the
performance panel and the overlay. Nothing is shown on screen, no model is
loaded and no audio is captured, so it runs in a second and is safe to run
while the real app is open. Useful for checking a layout or theme change, and
it is how the screenshots in the README were made.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from subordinant import logsetup  # noqa: F401 - importing the package pins the MSVC runtime
from subordinant.config import Config
from subordinant.history import History
from subordinant.pipeline import Utterance

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QColor, QPainter, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

LINES = [
    (0, "ねえ、今日の配信見た？", "Hey, did you watch today's stream?"),
    (1, "見た見た！最後のボス戦、めちゃくちゃ熱かったよね。", "I did, I did! That last boss fight was insanely hype, right?"),
    (0, "あそこで回復アイテム使い切っちゃうとは思わなかった。", "I didn't think she'd burn through all her healing items right there."),
    (1, "でも結局ノーデスでクリアしたんだから、やっぱり上手いよ。", "But she still cleared it without dying in the end, so she really is good."),
    (0, "天音さん、来週はコラボって言ってたけど、誰とやるんだろう。", "Amane-san said next week's a collab, but I wonder who it's with."),
    (1, "まだ内緒らしいよ。告知は金曜日だって。", "Apparently it's still a secret. She said the announcement's on Friday."),
    (0, "楽しみだなあ。絶対リアタイで見る。", "I can't wait. I'm definitely watching it live."),
]


def populate(model, speakers: bool = False) -> None:
    now = time.time() - 60
    for index, (speaker, ja, en) in enumerate(LINES):
        model.add(
            Utterance(
                id=index + 1,
                source=ja,
                english=en,
                speaker=speaker if speakers else -1,
                wall_time=now + index * 6.5,
                final=True,
                engine="llm",
                latency=0.7,
            )
        )
    # One line still being finished: a draft, to show how it differs.
    model.add(
        Utterance(
            id=len(LINES) + 1,
            source="じゃあ、そのときまた一緒に見ようか。",
            english="Well then, let's watch it together again then.",
            wall_time=now + len(LINES) * 6.5,
            final=False,
            engine="nmt",
        )
    )


METRICS = {
    "keeping_up": True,
    "health": "Keeping up with real time",
    "engine": "Kotoba-Whisper v2.0 on cuda (float16)",
    "translator": "Qwen3 4B Instruct (drafts: Sugoi v4)",
    "device_name": "Speakers (High Definition Audio Device) [Loopback]",
    "capture_latency_ms": 24.0,
    "passes": 1482,
    "mean_pass_ms": 118.0,
    "p90_pass_ms": 163.0,
    "rtf": 0.34,
    "source_lag_sec": 0.47,
    "preview_latency_sec": 0.92,
    "final_latency_sec": 0.71,
    "final_latency_p90_sec": 1.24,
    "mt_final_ms": 164.0,
    "mt_preview_ms": 31.0,
    "llm_first_token_ms": 48.0,
    "utterances": 212,
    "fallback_lines": 0,
    "eager_endpoints": 41,
    "pause_splits": 96,
    "deferred_closes": 37,
    "revised_lines": 58,
    "backlog_sec": 0.02,
    "backlog_max_sec": 0.31,
    "backlog_dropped_sec": 0.0,
    "capture_dropped_sec": 0.0,
    "capture_dropped_blocks": 0,
    "gpu_mem_used_mib": 7820,
    "gpu_mem_total_mib": 16376,
    "gpu_util_pct": 38,
    "process_rss_mib": 1480,
}


def grab(widget, path: Path) -> None:
    widget.grab().save(str(path))
    print(f"  wrote {path}")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ui_preview")
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv[:1])
    from subordinant.overlay import CaptionOverlay
    from subordinant.ui.transcript import TranscriptModel
    from subordinant.ui.window import MainWindow

    cfg = Config()
    cfg.theme = theme
    cfg.window_geometry = ""
    history = History(cfg.history_max_entries)
    model = TranscriptModel(history)
    window = MainWindow(cfg, model, history)
    # Laid out and painted, but never put on screen.
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1040, 720)
    window.show()
    populate(model)
    window.set_state("recognizing", "Speakers (High Definition Audio Device)")
    live = window.live_page.live
    live.set_caption("それにしても、今回のイベントは", "報酬が")
    live.set_preview("That said, the rewards for this event are...")
    window.update_metrics(METRICS)
    app.processEvents()
    window.live_page.transcript.jump_to_latest()
    app.processEvents()
    grab(window, out / f"live-{theme}.png")

    window.live_page.metrics_button.setChecked(True)
    window.notify(
        "warning",
        "Ollama is not answering at http://localhost:11434. Using the built-in "
        "translator for now.",
        "Open settings",
        "settings",
    )
    app.processEvents()
    window.live_page.transcript.jump_to_latest()
    app.processEvents()
    grab(window, out / f"live-metrics-{theme}.png")
    window.live_page.metrics_button.setChecked(False)
    window.dismiss_notice()

    window.set_state("idle", "")
    model.clear()
    window.set_ready_summary(
        [
            "Balanced profile",
            "Hears with Kotoba-Whisper v2.0",
            "Translates with Qwen3 4B Instruct, drafts by Sugoi v4",
        ]
    )
    app.processEvents()
    grab(window, out / f"idle-{theme}.png")

    populate(model, speakers=True)
    window.show_page("history")
    window.history_page.search.setText("コラボ")
    app.processEvents()
    grab(window, out / f"history-{theme}.png")
    window.history_page.search.setText("")

    window.show_page("settings")
    app.processEvents()
    page = window.settings_page
    bar = page._scroll.verticalScrollBar()
    for index, (_button, heading) in enumerate(page._sections):
        bar.setValue(max(0, heading.y() - 14))
        app.processEvents()
        grab(window, out / f"settings-{index + 1}-{theme}.png")

    # The overlay, composited over a stand-in for bright video.
    overlay = CaptionOverlay(cfg)
    overlay.resize(1200, overlay.height())
    for name, show_source in (("overlay", False), ("overlay-ja", True)):
        cfg.overlay_show_source = show_source
        overlay.apply_settings()
        overlay.resize(1200, overlay.height())
        overlay._final = LINES[1][2]
        overlay._final_source = LINES[1][1]
        overlay._final_is_draft = False
        overlay._preview = "I didn't think she'd use up all her healing..."
        overlay._caption = "あそこで回復アイテム使い切っちゃうとは"
        canvas = QPixmap(overlay.size())
        painter = QPainter(canvas)
        painter.fillRect(canvas.rect(), QColor("#6f8fb5"))
        painter.fillRect(0, canvas.height() // 2, canvas.width(), canvas.height() // 2, QColor("#d9c7a3"))
        painter.end()
        painter = QPainter(canvas)
        overlay.render(painter, flags=overlay.RenderFlag.DrawChildren)
        painter.end()
        canvas.save(str(out / f"{name}.png"))
        print(f"  wrote {out / (name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
