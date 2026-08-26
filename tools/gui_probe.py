"""Narrow down where the GUI process dies during startup.

Loads Qt and the ASR model in a controlled order, printing each step, with
faulthandler on so a native crash still leaves a stack behind.

    .venv\\Scripts\\python.exe -u -m tools.gui_probe [qt-first|model-first]
"""

from __future__ import annotations

import faulthandler
import sys

faulthandler.enable()


def step(msg: str) -> None:
    print(f"-- {msg}", flush=True)


def load_qt():
    step("importing PyQt6")
    from PyQt6.QtWidgets import QApplication

    step("creating QApplication")
    app = QApplication(sys.argv[:1])
    step("QApplication created")
    return app


def load_model():
    step("importing subordinant.asr (runs cuda.bootstrap)")
    from subordinant.asr import Transcriber
    from subordinant.config import Config

    step("constructing WhisperModel")
    asr = Transcriber(Config.load())
    step(f"model loaded on {asr.device}")
    step("warmup")
    asr.warmup()
    step("warmup done")
    return asr


def main() -> int:
    order = sys.argv[1] if len(sys.argv) > 1 else "qt-first"
    if order == "qt-first":
        app = load_qt()  # noqa: F841 - kept alive deliberately
        load_model()
    elif order == "preload-qt-first":
        step("preloading system MSVC runtime")
        from subordinant.cuda import preload_msvc_runtime

        step(f"preloaded: {preload_msvc_runtime()}")
        app = load_qt()  # noqa: F841
        load_model()
    elif order == "import-only":
        # Does merely importing PyQt6 poison the process, or only building a
        # QApplication? Decides how much main.py has to defer.
        step("importing PyQt6 (no QApplication)")
        import PyQt6.QtWidgets  # noqa: F401

        step("PyQt6 imported")
        load_model()
    else:
        load_model()
        load_qt()
    step("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
