# PyInstaller spec for SubOrdinant.
#
#   .venv\Scripts\pyinstaller.exe SubOrdinant.spec --noconfirm
#
# Model weights are deliberately NOT bundled. They are ~1.6 GB and would triple
# an already large installer; the app downloads them to the Hugging Face cache on
# first run instead, so an update ships without them.
#
# The CUDA libraries from the nvidia-* wheels must keep their
# nvidia\<lib>\bin\*.dll layout inside the bundle, because subordinant.cuda
# looks for exactly that tree under sys._MEIPASS to register the DLL
# directories at startup.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

SITE_PACKAGES = Path(SPECPATH) / ".venv" / "Lib" / "site-packages"
SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"


# PyQt6 vendors an MSVC runtime from 2019 (14.26) in PyQt6\Qt6\bin and adds that
# directory to the DLL search path. CTranslate2 is linked against a newer one and
# faults on model construction if it binds to Qt's copy. Keeping exactly one
# build - the system's - is what stops that, so every copy is dropped from the
# bundle below and replaced with the System32 set.
CRT_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)

binaries = []
datas = []

# ctranslate2 loads every DLL beside itself at import, including its vendored
# cuDNN stub and OpenMP runtime, so all of them have to travel together.
binaries += [
    (str(dll), "ctranslate2") for dll in (SITE_PACKAGES / "ctranslate2").glob("*.dll")
]

# Preserve nvidia\<lib>\bin\ exactly; see the note above.
for dll in SITE_PACKAGES.glob("nvidia/*/bin/*.dll"):
    binaries.append((str(dll), str(dll.parent.relative_to(SITE_PACKAGES))))

# Silero VAD ONNX weights and the tokenizer files faster-whisper ships.
datas += collect_data_files("faster_whisper")

# onnxruntime's DLLs must keep their onnxruntime\capi\ layout, because
# subordinant.cuda.preload_onnxruntime looks for exactly that path under
# sys._MEIPASS. It has to load this copy by absolute path before sherpa-onnx
# binds to the older onnxruntime.dll that Windows 11 ships in System32 - which
# is a segfault, not an error. collect_dynamic_libs already preserves that
# layout; the requirement is noted here so it is not "tidied" into the root.
for package in ("onnxruntime", "av", "pyaudiowpatch", "soxr"):
    binaries += collect_dynamic_libs(package)

# sentencepiece is copied by hand rather than declared as a hidden import.
#
# PyInstaller imports every collected package in one isolated subprocess to walk
# its binary dependencies, and that subprocess has already imported PyQt6. Qt puts
# its vendored MSVC runtime on the DLL search path, _sentencepiece.pyd binds to
# that instead of the system build, and the subprocess dies with an access
# violation (0xC0000005) - the same collision described at the top of this file.
# The app avoids it by pinning the system runtime in subordinant/__init__ before
# Qt loads; PyInstaller's analysis subprocess has no such protection and no hook
# to add one.
#
# Copying the files directly keeps the package out of that import list. It is
# safe because sentencepiece has nothing to discover: one statically linked .pyd
# and a few pure-Python modules, no sibling DLLs.
SENTENCEPIECE = SITE_PACKAGES / "sentencepiece"
binaries += [(str(p), "sentencepiece") for p in SENTENCEPIECE.glob("*.pyd")]
datas += [(str(p), "sentencepiece") for p in SENTENCEPIECE.glob("*.py")]

a = Analysis(
    ["main.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "subordinant",
        "pyaudiowpatch",
        "soxr",
        # Imported lazily inside Translator.__init__, so static analysis misses
        # it. sentencepiece is deliberately absent - see the note above.
        "huggingface_hub",
        # Likewise lazy, inside SpeakerTracker.__init__. Unlike sentencepiece
        # this one is safe to let PyInstaller import for analysis - checked
        # directly, it survives being imported after PyQt6 without the runtime
        # pin, so it does not need the copy-by-hand treatment.
        "sherpa_onnx",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Excluded from analysis, not from the bundle - the files are copied in
        # by hand above. PyInstaller's modulegraph scans bytecode, so it finds
        # the lazy import inside Translator.__init__ even without a hidden
        # import, and importing it for dependency analysis crashes the isolated
        # subprocess. See the note beside SENTENCEPIECE above.
        "sentencepiece",
        # Pulled in transitively but unused, and each costs tens of megabytes.
        "tkinter",
        "matplotlib",
        "scipy",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtQuick",
        "PyQt6.QtQml",
        "PyQt6.Qt3DCore",
        "PyQt6.QtMultimedia",
        "PyQt6.QtBluetooth",
        "PyQt6.QtDesigner",
        "PyQt6.QtPdf",
    ],
    noarchive=False,
)

# Collapse every MSVC runtime copy down to the system build; see CRT_DLLS above.
# numpy's privately renamed copy (msvcp140-<hash>.dll) has its own base name and
# cannot collide, so it is left alone.
a.binaries = [
    entry for entry in a.binaries if Path(entry[0]).name.lower() not in CRT_DLLS
]
a.binaries += [
    (name, str(SYSTEM32 / name), "BINARY")
    for name in CRT_DLLS
    if (SYSTEM32 / name).exists()
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubOrdinant",
    debug=False,
    strip=False,
    upx=False,
    # No console window: the tray icon is the app's only chrome.
    console=False,
)

# Same app with a console attached. Without it a failure inside a bundled native
# library is completely silent - there is nowhere for the traceback to go.
exe_debug = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubOrdinant-debug",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    exe_debug,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SubOrdinant",
)
