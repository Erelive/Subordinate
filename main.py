"""SubOrdinant - live captions for Windows system audio.

    .venv\\Scripts\\pythonw.exe main.py

The overlay is click-through and has no taskbar entry, so the tray icon is the
only way to control or quit it.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import signal
import sys
from dataclasses import fields
from pathlib import Path

# A DLL conflict inside CTranslate2 or Qt faults the process rather than raising,
# so without this a packaged build dies leaving nothing behind to debug.
#
# A windowed build has no console: sys.stderr is None, and faulthandler.enable()
# raises RuntimeError rather than falling back, which killed SubOrdinant.exe
# before any of its own code ran. Point it at a file instead - a native fault
# leaves nothing else behind, so this is exactly when the output matters most.
#
# The directory is resolved here rather than through config.config_dir() because
# this has to run before `subordinant` is imported: importing the package pins
# the MSVC runtime, and that is itself a load-library step worth having a
# faulthandler for.
_crash_log = None  # module-level so it is not closed by garbage collection
if sys.stderr is not None:
    faulthandler.enable()
else:
    try:
        _dir = Path(os.environ.get("APPDATA") or Path.home()) / "SubOrdinant"
        _dir.mkdir(parents=True, exist_ok=True)
        _crash_log = open(_dir / "crash.log", "a", buffering=1, encoding="utf-8")
        faulthandler.enable(_crash_log)
    except OSError:
        pass  # no crash log is survivable; failing to start is not

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
from subordinant.window import ENGINES, TranscriptWindow  # noqa: E402

log = logging.getLogger("subordinant")


class Bridge(QObject):
    """Carries pipeline output from the worker thread to the Qt main thread.

    Signals emitted across threads are delivered queued, which is what makes it
    safe for the worker to call these directly.
    """

    caption = pyqtSignal(str, str)
    status = pyqtSignal(str)
    translation = pyqtSignal(str, str)
    commit = pyqtSignal(str)


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
    parser.add_argument(
        "--mt",
        dest="mt_enabled",
        action="store_true",
        default=None,  # None so it cannot override a saved config's value
        help="translate finished utterances with a dedicated NMT model instead "
        "of Whisper's --task translate; shows source text and English",
    )
    parser.add_argument("--mt-model", dest="mt_model", help="NMT model to use")
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

    window = TranscriptWindow(cfg)
    bridge = Bridge()
    bridge.caption.connect(window.set_live, Qt.ConnectionType.QueuedConnection)
    bridge.translation.connect(
        window.add_utterance, Qt.ConnectionType.QueuedConnection
    )
    # Without MT there are no utterance pairs, so committed text is the only
    # thing that would ever reach the transcript.
    if not cfg.mt_enabled:
        bridge.commit.connect(
            window.add_source_only, Qt.ConnectionType.QueuedConnection
        )

    overlay = CaptionOverlay(cfg) if cfg.overlay_enabled else None
    if overlay is not None:
        bridge.caption.connect(overlay.set_caption, Qt.ConnectionType.QueuedConnection)
        bridge.translation.connect(
            overlay.set_translation, Qt.ConnectionType.QueuedConnection
        )

    window.apply_settings(cfg.window_opacity, cfg.window_on_top)
    window.show()

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

    def build_pipeline() -> CaptionPipeline:
        return CaptionPipeline(
            cfg,
            on_caption=bridge.caption.emit,
            on_status=bridge.status.emit,
            on_translation=bridge.translation.emit,
            on_commit=bridge.commit.emit,
        )

    pipeline = build_pipeline()
    window.load_settings(cfg)

    def apply_settings() -> None:
        """Rebuild the pipeline for a new language or translation engine.

        Both change which models are loaded, and those are constructed once
        inside the worker, so the pipeline is torn down and replaced rather than
        mutated underneath a running thread.
        """
        nonlocal pipeline
        language, engine = window.pending_settings()
        cfg.language = language
        for key, value in ENGINES[engine].items():
            setattr(cfg, key, value)

        window.set_status("applying settings...")
        window.apply_button.setEnabled(False)
        pipeline.stop()

        # commit only feeds the transcript when nothing else will, so the
        # connection has to follow whether MT is now on.
        try:
            bridge.commit.disconnect(window.add_source_only)
        except TypeError:
            pass  # not connected
        if not cfg.mt_enabled:
            bridge.commit.connect(
                window.add_source_only, Qt.ConnectionType.QueuedConnection
            )

        pipeline = build_pipeline()
        pipeline.set_paused(window.pause_box.isChecked())
        pipeline.start()
        window.show_applied()

    window.apply_button.clicked.connect(apply_settings)

    def on_status(msg: str) -> None:
        log.info("status: %s", msg)
        tray.setToolTip(f"SubOrdinant - {msg}")
        window.set_status(msg)
        if msg.startswith("error:"):
            tray.showMessage("SubOrdinant", msg, QSystemTrayIcon.MessageIcon.Critical)

    bridge.status.connect(on_status, Qt.ConnectionType.QueuedConnection)

    # Pause has two controls - the tray item and the window checkbox - and each
    # one's toggled signal would drive the other straight back into here. Block
    # the signals while syncing rather than relying on the values converging.
    syncing = False

    def on_pause(checked: bool) -> None:
        nonlocal syncing
        if syncing:
            return
        syncing = True
        try:
            pipeline.set_paused(checked)
            pause_action.setText("Resume captions" if checked else "Pause captions")
            pause_action.setChecked(checked)
            window.pause_box.setChecked(checked)
        finally:
            syncing = False
        if checked and overlay is not None:
            overlay.clear()
            overlay.hide()
        tray.setToolTip("SubOrdinant - paused" if checked else "SubOrdinant - listening")

    pause_action.toggled.connect(on_pause)
    window.pause_box.toggled.connect(on_pause)

    def shutdown() -> None:
        # Persist what the window's controls were left on, so the next launch
        # comes back the way it was put down.
        cfg.window_opacity, cfg.window_on_top = window.current_settings()
        try:
            cfg.save()
        except OSError as exc:  # noqa: BLE001 - never block quitting on this
            log.warning("could not save window settings: %s", exc)
        pipeline.stop()
        tray.hide()
        window.close()
        app.quit()

    quit_action.triggered.connect(shutdown)
    # Resolved at call time, not bound now: applying settings replaces the
    # pipeline object, and binding the method here would leave this stopping a
    # pipeline that is already dead while the live one keeps its thread.
    app.aboutToQuit.connect(lambda: pipeline.stop())

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
