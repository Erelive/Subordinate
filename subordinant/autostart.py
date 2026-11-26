"""Launch on Windows startup.

One value under HKCU\\...\\Run, which is what the Startup tab in Task Manager
reads and what the user can switch off there without opening this app. Per
user, so it needs no elevation.

The registry is the truth, not the config file: the user may have removed the
entry from Task Manager, or moved the app. Config.launch_on_startup mirrors
this on launch, never the other way round.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE = "SubOrdinant"


def command() -> str:
    """The command line that starts this app, quietly, in the tray."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --minimized'
    # From a checkout: pythonw, so no console window appears at logon.
    python = Path(sys.executable)
    windowless = python.with_name("pythonw.exe")
    launcher = windowless if windowless.exists() else python
    script = Path(__file__).resolve().parent.parent / "main.py"
    return f'"{launcher}" "{script}" --minimized'


def is_enabled() -> bool:
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, _VALUE)
            return True
    except OSError:
        return False


def set_enabled(enabled: bool) -> bool:
    """Add or remove the Run entry. Returns whether it worked."""
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enabled:
                winreg.SetValueEx(key, _VALUE, 0, winreg.REG_SZ, command())
            else:
                try:
                    winreg.DeleteValue(key, _VALUE)
                except FileNotFoundError:
                    pass
        return True
    except OSError as exc:
        log.warning("could not change the startup entry: %s", exc)
        return False
