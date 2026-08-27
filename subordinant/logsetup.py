"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging
import sys

from .config import config_dir

LOG_NAME = "subordinant.log"


def configure(verbose: bool = False) -> None:
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    else:
        # A windowed PyInstaller build has no console, so sys.stderr is None and
        # a StreamHandler would silently drop every record. Without a file there
        # would be nothing at all to debug a packaged run with.
        try:
            path = config_dir()
            path.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path / LOG_NAME, encoding="utf-8"))
        except OSError:
            handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # faster-whisper logs a line per inference pass, which at two passes a
    # second buries everything else.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    # These dump full HTTP headers at DEBUG, which buries our own output
    # whenever --verbose is on.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
