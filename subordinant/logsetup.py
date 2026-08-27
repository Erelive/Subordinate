"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging
import sys

from .config import config_dir

LOG_NAME = "subordinant.log"


def has_console() -> bool:
    """Is there a real console to write to?

    Checks sys.__stderr__ rather than sys.stderr: a windowed build has neither,
    but main.py replaces sys.stderr with the crash log so that libraries which
    write to it unconditionally keep working. Only the original tells the truth,
    and logging must not follow the substitute - normal records would then be
    interleaved into the file that exists for crashes.
    """
    return sys.__stderr__ is not None


def configure(verbose: bool = False) -> None:
    handlers: list[logging.Handler] = []
    if has_console():
        handlers.append(logging.StreamHandler())
    else:
        # A windowed PyInstaller build has no console, so a StreamHandler would
        # silently drop every record. Without a file there would be nothing at
        # all to debug a packaged run with.
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
