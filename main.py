"""SubOrdinant - live Japanese-to-English captions for Windows system audio.

    .venv\\Scripts\\pythonw.exe main.py

Opens the main window. Pick a profile, press Start, play something in
Japanese: the English appears in the window and, if the overlay is on, over
the screen. Everything runs on this machine.
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
# The Hub client phones home with usage statistics unless told not to. Nothing
# about this app should leave the machine except a model download the user
# asked for.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Imported before PyQt6 on purpose: the package __init__ pins the system MSVC
# runtime, which has to happen before Qt puts its own copy on the search path.
from subordinant import logsetup  # noqa: E402
from subordinant.config import Config  # noqa: E402
from subordinant.profiles import PROFILES, apply_profile  # noqa: E402

log = logging.getLogger("subordinant")

_instance_mutex = None  # held for the life of the process


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="SubOrdinant", description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="start from this performance profile instead of the saved settings",
    )
    parser.add_argument(
        "--start", action="store_true", help="begin translating immediately"
    )
    parser.add_argument(
        "--minimized", action="store_true", help="start in the tray without a window"
    )
    parser.add_argument("--model", help="recognition model name or local path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="inference device")
    parser.add_argument("--compute-type", help="e.g. float16, int8_float16, int8")
    parser.add_argument(
        "--initial-prompt",
        dest="initial_prompt",
        help="names and terms to expect, e.g. '天音 = Amane, ホロライブ'",
    )
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
    if not cfg.restore_last_config and not args.profile:
        # Appearance and window settings were loaded and are kept; the engine
        # goes back to the default profile.
        apply_profile(cfg, "balanced")
    if args.profile:
        apply_profile(cfg, args.profile)
    known = {f.name for f in fields(Config)}
    for key, value in vars(args).items():
        if value is None or key not in known or key == "profile":
            continue
        setattr(cfg, key, value)
    if args.start:
        cfg.autostart_translation = True
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

    def load_vad() -> str:
        import numpy as np

        from subordinant.vad import VAD_FRAME, SileroVAD

        probs = SileroVAD().probabilities(np.zeros((4, VAD_FRAME), dtype=np.float32))
        return f"silero ok, silence scores {float(probs.max()):.2f}"

    report("voice detector", load_vad)

    def load_asr() -> str:
        from subordinant.asr import Transcriber

        asr = Transcriber(cfg)
        asr.warmup()
        return f"{asr.model_id} on {asr.device} ({asr.compute_type})"

    report("recognition", load_asr)

    if cfg.mt_enabled:
        def load_nmt() -> str:
            from subordinant.translate import Translator

            return Translator(cfg).translate("これはテストです。") or "(empty)"

        report("translation (built-in)", load_nmt)
        if cfg.mt_backend == "llm":
            def load_llm() -> str:
                from subordinant.llm_translate import LLMTranslator

                return LLMTranslator(cfg).translate("これはテストです。") or "(empty)"

            report(f"translation ({cfg.llm_model})", load_llm)

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

        capture = LoopbackCapture(frames_per_buffer=cfg.frames_per_buffer, watch_device=False)
        capture.start()
        name = capture.device_name
        capture.stop()
        return name

    report("audio", open_audio)

    def gpu() -> str:
        from subordinant.metrics import gpu_memory

        info = gpu_memory()
        return "no NVIDIA GPU visible" if info is None else f"{info[0]} / {info[1]} MiB in use"

    report("gpu memory", gpu)

    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def _set_taskbar_identity() -> None:
    """Tell Windows this process is its own app, not the launcher's.

    Without an explicit AppUserModelID a source checkout inherits python.exe's
    identity: the taskbar button groups under Python and shows the Python
    icon, whatever setWindowIcon says. A frozen build gets its identity from
    the executable, so this is a no-op there.

    Cosmetic, so every failure is swallowed - this must not be what stops the
    app from starting.
    """
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SubOrdinant.Captions"
        )
    except Exception:  # noqa: BLE001 - cosmetic; never worth failing over
        log.debug("could not set AppUserModelID", exc_info=True)


def _already_running() -> bool:
    """Is another copy of the app running in this session?

    Two copies would both capture the same audio and both draw captions over
    the screen. With "launch on startup" on, opening the app from the Start
    menu is exactly how a second one would come to exist.
    """
    global _instance_mutex
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        _instance_mutex = kernel32.CreateMutexW(None, False, "Local\\SubOrdinant.SingleInstance")
        return kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    except Exception:  # noqa: BLE001 - if this cannot be checked, just run
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logsetup.configure(args.verbose)
    cfg = build_config(args)
    if args.save_config:
        print(f"wrote {cfg.save()}")
    if args.selftest:
        return selftest(cfg)

    # Qt is imported here rather than at the top so that --selftest and
    # --save-config work on a machine with no display.
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from subordinant.overlay import app_icon
    from subordinant.ui.controller import Controller

    _set_taskbar_identity()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("SubOrdinant")
    # Closing the window to the tray, or the overlay hiding itself when idle,
    # must not end the process.
    app.setQuitOnLastWindowClosed(False)
    # Inherited by every window and dialog.
    app.setWindowIcon(app_icon())

    if _already_running():
        QMessageBox.information(
            None,
            "SubOrdinant",
            "SubOrdinant is already running. Look for its icon in the system tray.",
        )
        return 0

    controller = Controller(
        app, cfg, start_hidden=args.minimized or cfg.start_minimized
    )
    if controller.tray is None and (args.minimized or cfg.start_minimized):
        # Nothing to restore the window from, so do not hide it.
        controller.show_window()

    # Qt's event loop swallows SIGINT unless Python gets a chance to run.
    signal.signal(signal.SIGINT, lambda *_: controller.quit())
    from PyQt6.QtCore import QTimer

    heartbeat = QTimer()
    heartbeat.start(250)
    heartbeat.timeout.connect(lambda: None)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
