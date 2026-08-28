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
_null_out = None
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

    # Give the interpreter real streams before importing anything else.
    #
    # Plenty of libraries write to stdout/stderr without checking - tqdm, which
    # huggingface_hub uses for download progress, is the one that bit here:
    # fetching the translation model died with "'NoneType' object has no
    # attribute 'write'" and the app sat on "loading translation model..."
    # forever. Guarding each caller individually is whack-a-mole; giving them
    # something writable fixes the whole class at once.
    #
    # stderr goes to the crash log rather than nowhere, because Python's default
    # excepthook prints tracebacks there - otherwise an unhandled exception in a
    # windowed build leaves no trace at all.
    try:
        _null_out = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _null_out
        sys.stderr = _crash_log or _null_out
    except OSError:
        pass

# Progress bars have nowhere useful to go in a windowed build, and writing them
# to the crash log would bury the tracebacks it exists for.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

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
    # The trailing int is the speaker id, diarize.UNKNOWN when unlabelled. Qt
    # signals need a concrete type, which is why it is a sentinel int rather
    # than an optional.
    translation = pyqtSignal(str, str, int)
    commit = pyqtSignal(str, int)
    # English for speech still in progress; replaces itself every pass.
    live_translation = pyqtSignal(str)


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
    parser.add_argument(
        "--speakers",
        dest="diarize_enabled",
        action="store_true",
        default=None,  # None so it cannot override a saved config's value
        help="label each line with who said it, and split lines when the "
        "speaking voice changes",
    )
    parser.add_argument(
        "--initial-prompt",
        dest="initial_prompt",
        help="names and terms to expect, so Whisper stops mishearing them; "
        "keep it short or it starts reciting the prompt",
    )
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="load each model and exit, reporting what worked; run it from "
        "SubOrdinant-debug.exe to check a packaged build",
    )
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


def selftest(cfg: Config) -> int:
    """Load every native subsystem and report, without opening a window.

    This exists because the failures that packaging introduces are DLL
    conflicts, and those fault the process rather than raising - a bundle can be
    built, signed and shipped and still die the moment a model is constructed.
    Three separate libraries here resolve a DLL by base name that something else
    on the machine also provides, Windows' own onnxruntime.dll among them, and
    none of that is visible until the load is actually attempted.

    Run it from SubOrdinant-debug.exe: the windowed build has no console for the
    output to reach.
    """
    ok = True

    def report(name: str, fn) -> None:
        nonlocal ok
        try:
            print(f"  {name}: {fn()}", flush=True)
        except Exception as exc:  # noqa: BLE001 - reporting is the whole job
            ok = False
            print(f"  {name}: FAILED - {type(exc).__name__}: {exc}", flush=True)

    print(f"SubOrdinant selftest (frozen={getattr(sys, 'frozen', False)})", flush=True)

    from subordinant import cuda

    # These two ran at import time - importing the package is what pins them -
    # so calling them again returns an empty list. Report the state instead;
    # "(none)" from an idempotent call reads like a failure when it is the
    # opposite.
    report("msvc runtime", lambda: "pinned" if cuda._preloaded else "NOT pinned")
    report("cuda dirs", lambda: "registered" if cuda._done else "not registered")
    report("onnxruntime", lambda: cuda.preload_onnxruntime() or "(already pinned)")

    def load_asr() -> str:
        from subordinant.asr import Transcriber

        Transcriber(cfg).warmup()
        return f"{cfg.model} on {cfg.device}"

    report("whisper", load_asr)

    if cfg.mt_enabled:
        def load_mt() -> str:
            from subordinant.translate import Translator

            return Translator(cfg).translate("これはテストです。") or "(empty)"

        report("translation", load_mt)

    if cfg.diarize_enabled:
        def load_spk() -> str:
            import numpy as np

            from subordinant.config import SAMPLE_RATE
            from subordinant.diarize import SpeakerTracker

            tracker = SpeakerTracker(cfg)
            # A second of noise: the point is that compute() runs at all, which
            # is where a wrongly-bound onnxruntime takes the process down.
            rng = np.random.default_rng(0)
            tracker.feed(rng.standard_normal(SAMPLE_RATE * 2).astype(np.float32) * 0.1)
            tracker.process()
            return f"dim ok, {tracker.speaker_count} speaker(s) from noise"

        report("speakers", load_spk)

    def open_audio() -> str:
        from subordinant.audio import LoopbackCapture

        capture = LoopbackCapture(frames_per_buffer=cfg.frames_per_buffer)
        capture.start()
        name = capture.device_name
        capture.stop()
        return name

    report("audio", open_audio)

    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logsetup.configure(args.verbose)
    cfg = build_config(args)
    if args.save_config:
        print(f"wrote {cfg.save()}")
    if args.selftest:
        return selftest(cfg)

    app = QApplication(sys.argv[:1])
    # The overlay hides itself when idle; that must not end the process.
    app.setQuitOnLastWindowClosed(False)

    window = TranscriptWindow(cfg)
    bridge = Bridge()
    bridge.caption.connect(window.set_live, Qt.ConnectionType.QueuedConnection)
    bridge.translation.connect(
        window.add_utterance, Qt.ConnectionType.QueuedConnection
    )
    bridge.live_translation.connect(
        window.set_live_translation, Qt.ConnectionType.QueuedConnection
    )
    # Without MT there are no utterance pairs, so committed text is the only
    # thing that would ever reach the transcript.
    if not cfg.mt_enabled:
        bridge.commit.connect(
            window.add_source_only, Qt.ConnectionType.QueuedConnection
        )

    # Built whether or not it is switched on: an unshown QWidget costs nothing,
    # and constructing it up front is what lets the toolbar toggle take effect
    # without a restart. What actually turns it off is disconnecting it - a
    # merely hidden overlay would still run its full relayout-and-repaint on
    # every caption, which is the cost the toggle exists to avoid.
    overlay = CaptionOverlay(cfg)

    def set_overlay_enabled(on: bool) -> None:
        cfg.overlay_enabled = on  # saved on exit, so the choice survives a restart
        if on:
            bridge.caption.connect(
                overlay.set_caption, Qt.ConnectionType.QueuedConnection
            )
            bridge.translation.connect(
                overlay.set_translation, Qt.ConnectionType.QueuedConnection
            )
            bridge.live_translation.connect(
                overlay.set_live_translation, Qt.ConnectionType.QueuedConnection
            )
            return
        try:
            bridge.caption.disconnect(overlay.set_caption)
            bridge.translation.disconnect(overlay.set_translation)
            bridge.live_translation.disconnect(overlay.set_live_translation)
        except TypeError:
            pass  # not connected
        # Clear as well as hide: otherwise switching back on redisplays the line
        # that was on screen when it went off, until the next caption arrives.
        overlay.clear()
        overlay.hide()

    window.overlay_box.toggled.connect(set_overlay_enabled)
    # apply_settings drives the checkbox, whose toggled signal then runs
    # set_overlay_enabled - but only if the value changes from the unchecked
    # default, so call it directly rather than relying on that.
    set_overlay_enabled(cfg.overlay_enabled)

    window.apply_settings(
        cfg.window_opacity,
        cfg.window_on_top,
        cfg.mt_live_enabled,
        cfg.overlay_enabled,
    )
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
            on_live_translation=bridge.live_translation.emit,
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
        language, engine, diarize, speaker_cap, prompt = window.pending_settings()
        cfg.language = language
        cfg.diarize_enabled = diarize
        cfg.diarize_max_speakers = speaker_cap
        cfg.initial_prompt = prompt
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
        if checked:
            # Harmless when the overlay is switched off - it is already hidden.
            overlay.clear()
            overlay.hide()
        tray.setToolTip("SubOrdinant - paused" if checked else "SubOrdinant - listening")

    pause_action.toggled.connect(on_pause)
    window.pause_box.toggled.connect(on_pause)

    def on_live_preview(enabled: bool) -> None:
        # Read fresh on every pass by the worker, so no restart is needed. A
        # bool assignment is atomic under the GIL; the worst case is that one
        # in-flight pass uses the previous value.
        cfg.mt_live_enabled = enabled
        if not enabled:
            # _flush_live returns early once this is off, so it never emits the
            # empty preview that would clear these. Do it here or the last guess
            # stays on screen for the rest of the session.
            window.set_live_translation("")
            overlay.set_live_translation("")

    window.live_preview_box.toggled.connect(on_live_preview)

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
