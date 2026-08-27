"""Pre-download model weights so first launch isn't a silent multi-minute wait.

    .venv\\Scripts\\python.exe -m tools.fetch_model [model-name]
    .venv\\Scripts\\python.exe -m tools.fetch_model --mt [hf-repo]

--mt fetches the NMT translation model instead of a Whisper one.
"""

from __future__ import annotations

import sys
import time

from subordinant import cuda
from subordinant.config import Config

cuda.bootstrap()

from faster_whisper.utils import download_model  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    cfg = Config.load()

    if args and args[0] == "--mt":
        from huggingface_hub import snapshot_download

        name = args[1] if len(args) > 1 else cfg.mt_model
        print(f"fetching translation model {name} ...")
        t0 = time.perf_counter()
        path = snapshot_download(name)
        print(f"cached at {path} ({time.perf_counter() - t0:.0f}s)")
        return 0

    name = args[0] if args else cfg.model
    print(f"fetching {name} ...")
    t0 = time.perf_counter()
    path = download_model(name)
    print(f"cached at {path} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
