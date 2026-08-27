"""SubOrdinant - live captions for Windows system audio.

    .venv\\Scripts\\pythonw.exe main.py

The overlay is click-through and has no taskbar entry, so the tray icon is the
only way to control or quit it.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import signal
import sys
from dataclasses import fields

# A DLL conflict inside CTranslate2 or Qt faults the process rather than raising,
# so without this a packaged build dies leaving nothing behind to debug.
faulthandler.enable()

# Imported before PyQt6 on purpose: the package __init__ pins the system MSVC
# runtime, which has to happen before Qt puts its own copy on the search path.
from subordinant import logsetup
from subordinant.config import Config

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

from subordinant.asr import translate_warning  # noqa: E402
from subordinant.overlay import CaptionOverlay, make_tray_icon  # noqa: E402
from subordinant.pipeline import CaptionPipeline  # noqa: E402

log = logging.getLogger("subordinant")


class Bridge(QObject):
    """Carries pipeline output from the worker thread to the Qt main thread.

    Signals emitted across threads are delivered queued, which is what makes it
    safe for the worker to call these directly.
    """

    caption = pyqtSignal(str, str)
    status = pyqtSignal(str)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="SubOrdinant", description=__doc__)
    parser.add_argument("--model", help="faster-whisper model name or local path")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="inference device")
    parser.add_argument("--compute-type", help="e.g. float16, int8_float16, int8")
    parser.add_argument("--language", help="spoken language code, e.g. en")
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        help="translate outputs English for any source language",
    )
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="write the resulting settings to %%APPDATA%%\\SubOrdinant\\config.json",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.load()
    known = {f.name for f in fields(Config)}
    for key, value in vars(args).items():
        if value is None or key not in known:
            continue
        setattr(cfg, key, value)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logsetup.configure(args.verbose)
    cfg = build_config(args)
    if args.save_config:
        print(f"wrote {cfg.save()}")

    app = QApplication(sys.argv[:1])
    # The overlay hides itself when idle; that must not end the process.
    app.setQuitOnLastWindowClosed(False)

    overlay = CaptionOverlay(cfg)
    bridge = Bridge()
    bridge.caption.connect(overlay.set_caption, Qt.ConnectionType.QueuedConnection)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(
            None, "SubOrdinant", "No system tray available - cannot run without it."
        )
        return 1

    tray = QSystemTrayIcon(make_tray_icon())
    tray.setToolTip("SubOrdinant - starting...")
    menu = QMenu()
    pause_action = menu.addAction("Pause captions")
    pause_action.setCheckable(True)
    menu.addSeparator()
    quit_action = menu.addAction("Quit")
    tray.setContextMenu(menu)
    tray.show()

    pipeline = CaptionPipeline(
        cfg,
        on_caption=bridge.caption.emit,
        on_status=bridge.status.emit,
    )

    def on_status(msg: str) -> None:
        log.info("status: %s", msg)
        tray.setToolTip(f"SubOrdinant - {msg}")
        if msg.startswith("error:"):
            tray.showMessage("SubOrdinant", msg, QSystemTrayIcon.MessageIcon.Critical)

    bridge.status.connect(on_status, Qt.ConnectionType.QueuedConnection)

    def on_pause(checked: bool) -> None:
        pipeline.set_paused(checked)
        pause_action.setText("Resume captions" if checked else "Pause captions")
        if checked:
            overlay.clear()
            overlay.hide()
        tray.setToolTip("SubOrdinant - paused" if checked else "SubOrdinant - listening")

    pause_action.toggled.connect(on_pause)

    def shutdown() -> None:
        pipeline.stop()
        tray.hide()
        app.quit()

    quit_action.triggered.connect(shutdown)
    app.aboutToQuit.connect(pipeline.stop)

    # Qt's event loop swallows SIGINT unless Python gets a chance to run.
    signal.signal(signal.SIGINT, lambda *_: shutdown())
    heartbeat = QTimer()
    heartbeat.start(200)
    heartbeat.timeout.connect(lambda: None)

    pipeline.start()
    # Windows shows one balloon at a time, so these are exclusive: the greeting
    # would otherwise replace the warning a second after it appeared. Raised
    # here rather than from the pipeline because a translate/model mismatch is
    # knowable from the config alone, before the model load spends minutes on it.
    mismatch = translate_warning(cfg)
    if mismatch:
        tray.showMessage(
            "SubOrdinant", mismatch, QSystemTrayIcon.MessageIcon.Warning, 10000
        )
    else:
        tray.showMessage(
            "SubOrdinant",
            "Listening to system audio. Quit from this tray icon.",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
