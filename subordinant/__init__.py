"""SubOrdinant - live on-screen captions for Windows system audio.

Importing this package pins the system MSVC runtime before anything else can
load a different copy of it. PyQt6 bundles its own build in PyQt6\\Qt6\\bin and
adds that directory to the DLL search path on import; if CTranslate2's native
library binds to that build instead of the system one, constructing a Whisper
model faults the process with an access violation. Pinning here means module
import order elsewhere in the app does not have to be load-bearing.
"""

from . import cuda as _cuda

_cuda.preload_msvc_runtime()

__version__ = "0.1.0"
