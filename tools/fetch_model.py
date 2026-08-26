"""Pre-download model weights so first launch isn't a silent multi-minute wait.

    .venv\\Scripts\\python.exe -m tools.fetch_model [model-name]
"""

from __future__ import annotations

import sys
import time

from subordinant import cuda
from subordinant.config import Config

cuda.bootstrap()

from faster_whisper.utils import download_model  # noqa: E402


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else Config.load().model
    print(f"fetching {name} ...")
    t0 = time.perf_counter()
    path = download_model(name)
    print(f"cached at {path} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
