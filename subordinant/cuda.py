"""Make the pip-installed CUDA libraries findable on Windows.

nvidia-cublas-cu12 and nvidia-cudnn-cu12 drop their DLLs under
site-packages\\nvidia\\*\\bin, which is not on the DLL search path. CTranslate2
loads them by name at import time, so this has to run *before* faster_whisper is
imported or the model falls back to a "Library cublas64_12.dll is not found"
error.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_done = False
_preloaded = False

# PyQt6 ships its own copy of the MSVC runtime in PyQt6\Qt6\bin and puts that
# directory on the DLL search path when imported. CTranslate2's native library
# then resolves these by base name and can bind to PyQt6's build instead of the
# system one, which faults the process on model construction. Loading the
# System32 copies first pins the correct build: Windows resolves a later request
# for the same base name to the module already loaded.
_MSVC_RUNTIME = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "concrt140.dll",
)


def preload_msvc_runtime() -> list[str]:
    """Pin the system MSVC runtime before any package can supply its own."""
    global _preloaded
    if _preloaded or sys.platform != "win32":
        return []
    _preloaded = True

    import ctypes

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    loaded: list[str] = []
    for name in _MSVC_RUNTIME:
        path = system32 / name
        if not path.exists():
            continue
        try:
            ctypes.WinDLL(str(path))
        except OSError as exc:
            log.debug("could not preload %s: %s", path, exc)
            continue
        loaded.append(name)
    log.debug("preloaded MSVC runtime: %s", ", ".join(loaded) or "(none)")
    return loaded


def _roots() -> list[Path]:
    roots = [Path(entry) for entry in sys.path if entry]
    # PyInstaller lays the same tree out next to the executable.
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        roots.append(Path(frozen))
    return roots


def _candidate_dirs() -> list[Path]:
    """Directories holding the CUDA libraries shipped by the nvidia-* wheels.

    Both cuBLAS and the cuDNN sub-libraries have to be reachable. CTranslate2
    already pins the top-level libraries it vendors (ctranslate2.dll,
    cudnn64_9.dll, libiomp5md.dll) by loading them from its own package
    directory with an absolute path at import time, so registering these cannot
    displace them - but the vendored cudnn64_9.dll is only a loader stub, and
    the real cudnn_*64_9.dll files it pulls in live here.
    """
    dirs: list[Path] = []
    for root in _roots():
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            dirs.extend(sorted(p for p in nvidia.glob("*/bin") if p.is_dir()))
    return dirs


def bootstrap() -> list[Path]:
    """Register bundled CUDA DLL directories. Safe to call more than once."""
    global _done
    if _done or sys.platform != "win32":
        return []
    _done = True

    added: list[Path] = []
    seen: set[str] = set()
    for d in _candidate_dirs():
        key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(str(d))
        except OSError as exc:
            log.debug("could not register %s: %s", d, exc)
            continue
        # ctranslate2 resolves some libraries through PATH rather than the
        # per-process DLL directory list, so set both.
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        added.append(d)

    if added:
        log.debug("registered CUDA DLL dirs: %s", ", ".join(str(p) for p in added))
    else:
        log.debug("no bundled CUDA DLL dirs found; relying on system CUDA")
    return added
