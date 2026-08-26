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

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

SITE_PACKAGES = Path(SPECPATH) / ".venv" / "Lib" / "site-packages"

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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SubOrdinant",
)
