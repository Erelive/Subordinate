"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging


def configure(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # faster-whisper logs a line per inference pass, which at two passes a
    # second buries everything else.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    # These dump full HTTP headers at DEBUG, which buries our own output
    # whenever --verbose is on.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
