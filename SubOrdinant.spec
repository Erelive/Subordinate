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

for package in ("onnxruntime", "av", "pyaudiowpatch", "soxr"):
    binaries += collect_dynamic_libs(package)

a = Analysis(
    ["main.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "subordinant",
        "pyaudiowpatch",
        "soxr",
    ],
    hookspath=[],
    runtime_hooks=[],
    # Pulled in transitively but unused, and each costs tens of megabytes.
    excludes=[
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
