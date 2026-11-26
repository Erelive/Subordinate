"""The main window.

Frameless, with its own title bar, so the chrome can carry what a title bar
normally wastes: the app's state, and the three places to go. The window
still behaves like a native one - it snaps to screen edges, maximises on a
double-click, resizes from its borders and gets the system shadow and rounded
corners - because those are done by Windows itself: the window keeps the
styles of a normal resizable window and only answers "where is the title
bar?" differently. See _NativeFrame.

Three pages:

  Live      the transcript, the sentence in progress, and the controls.
  History   the same transcript, searchable, with export.
  Settings  everything in Config, without a text editor.

The window knows nothing about the pipeline. It emits what the user asked for
and is told what to show; ui/controller.py is the only thing that talks to
both.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

from PyQt6.QtCore import QByteArray, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..history import History
from ..profiles import PROFILE_BLURBS, PROFILE_LABELS, PROFILES, detect_profile
from . import icons
from .live import LiveCard
from .metrics import MetricsPanel
from .settings import SettingsPage
from .theme import Palette, palette, stylesheet
from .transcript import Look, SearchProxy, TranscriptModel, TranscriptView
from .widgets import Banner, IconButton, SegmentedControl, StatusPill, ToggleChip

log = logging.getLogger(__name__)

TITLE_HEIGHT = 46
RESIZE_BORDER = 6


class _NativeFrame:
    """Windows plumbing for a frameless window that still acts native.

    Qt's FramelessWindowHint creates a bare popup: no resize borders, no
    snapping, no shadow. Rather than reimplement each of those, this gives
    the window back the *styles* of a normal one (WS_THICKFRAME, WS_CAPTION)
    so that Windows provides them, and then removes the visible frame by
    answering WM_NCCALCSIZE with "the client area is the whole window".
    WM_NCHITTEST tells Windows which part of our own drawing is the title bar
    and which are the resize borders.

    Everything here is best-effort. If any call fails the window is simply a
    frameless Qt window, dragged by startSystemMove - less polished, fully
    usable.
    """

    WM_NCCALCSIZE = 0x0083
    WM_NCHITTEST = 0x0084
    HTCLIENT, HTCAPTION = 1, 2
    HTLEFT, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT = 10, 11, 12, 13, 14
    HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT = 15, 16, 17

    def __init__(self) -> None:
        self.active = False

    def install(self, hwnd: int, dark: bool, border: str) -> None:
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            dwmapi = ctypes.windll.dwmapi
            get_style = user32.GetWindowLongPtrW
            set_style = user32.SetWindowLongPtrW
            get_style.restype = ctypes.c_ssize_t
            get_style.argtypes = [wintypes.HWND, ctypes.c_int]
            set_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            style = get_style(hwnd, -16)
            # THICKFRAME | CAPTION | MAXIMIZEBOX | MINIMIZEBOX
            set_style(hwnd, -16, style | 0x00040000 | 0x00C00000 | 0x00010000 | 0x00020000)
            self.active = True
            self.recolor(hwnd, dark, border)
            # Round corners on Windows 11; an error on 10, which is ignored.
            corner = ctypes.c_int(2)
            dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd), 33, ctypes.byref(corner), ctypes.sizeof(corner)
            )
            # SWP_FRAMECHANGED | NOMOVE | NOSIZE | NOZORDER | NOACTIVATE
            user32.SetWindowPos(wintypes.HWND(hwnd), None, 0, 0, 0, 0, 0x0020 | 0x0002 | 0x0001 | 0x0004 | 0x0010)
        except Exception:  # noqa: BLE001 - cosmetic; never worth failing over
            log.debug("native frame unavailable", exc_info=True)
            self.active = False

    def recolor(self, hwnd: int, dark: bool, border: str) -> None:
        if sys.platform != "win32":
            return
        try:
            dwmapi = ctypes.windll.dwmapi
            flag = ctypes.c_int(1 if dark else 0)
            dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd), 20, ctypes.byref(flag), ctypes.sizeof(flag)
            )
            color = QColor(border)
            # COLORREF is 0x00BBGGRR.
            ref = ctypes.c_int(color.red() | (color.green() << 8) | (color.blue() << 16))
            dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd), 34, ctypes.byref(ref), ctypes.sizeof(ref)
            )
        except Exception:  # noqa: BLE001
            log.debug("could not recolor frame", exc_info=True)

    def handle(self, window: "MainWindow", message) -> tuple[bool, int]:
        if not self.active:
            return False, 0
        try:
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == self.WM_NCCALCSIZE and msg.wParam:
                user32 = ctypes.windll.user32
                if user32.IsZoomed(msg.hWnd):
                    # A maximised window overhangs the screen by its frame
                    # width on every side. Pull the client area back in, or
                    # the edges of the interface are off-screen.
                    rect = ctypes.cast(msg.lParam, ctypes.POINTER(wintypes.RECT)).contents
                    frame = user32.GetSystemMetrics(32) + user32.GetSystemMetrics(92)
                    rect.left += frame
                    rect.top += frame
                    rect.right -= frame
                    rect.bottom -= frame
                return True, 0
            if msg.message == self.WM_NCHITTEST:
                return self._hit(window, msg)
        except Exception:  # noqa: BLE001 - a bad hit test must not take the window down
            log.debug("native event failed", exc_info=True)
        return False, 0

    def _hit(self, window: "MainWindow", msg) -> tuple[bool, int]:
        user32 = ctypes.windll.user32
        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
        rect = wintypes.RECT()
        user32.GetWindowRect(msg.hWnd, ctypes.byref(rect))
        if not user32.IsZoomed(msg.hWnd):
            border = int(RESIZE_BORDER * window.devicePixelRatioF())
            left = x < rect.left + border
            right = x >= rect.right - border
            top = y < rect.top + border
            bottom = y >= rect.bottom - border
            if top and left:
                return True, self.HTTOPLEFT
            if top and right:
                return True, self.HTTOPRIGHT
            if bottom and left:
                return True, self.HTBOTTOMLEFT
            if bottom and right:
                return True, self.HTBOTTOMRIGHT
            if left:
                return True, self.HTLEFT
            if right:
                return True, self.HTRIGHT
            if top:
                return True, self.HTTOP
            if bottom:
                return True, self.HTBOTTOM
        pos = window.mapFromGlobal(QCursor.pos())
        if window.is_drag_area(pos):
            return True, self.HTCAPTION
        return False, 0


class TitleBar(QWidget):
    def __init__(self, pal: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(TITLE_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 6, 0)
        layout.setSpacing(10)

        self.logo = QLabel()
        self.logo.setFixedSize(22, 22)
        layout.addWidget(self.logo)
        self.name = QLabel("SubOrdinant")
        self.name.setObjectName("appName")
        layout.addWidget(self.name)
        layout.addSpacing(14)

        self.status = StatusPill(pal)
        layout.addWidget(self.status, stretch=1)

        # How far behind the speaker the finished English is running. In the
        # title bar so it can be seen from any page, and at a glance: it is
        # the one number that says whether the app is doing its job.
        self.delay = QLabel("")
        self.delay.setObjectName("delayChip")
        self.delay.setToolTip(
            "How long after a sentence ends its finished English appears "
            "(typical recent sentence)."
        )
        self.delay.hide()
        layout.addWidget(self.delay, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(6)

        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.nav_buttons: dict[str, QToolButton] = {}
        for key, label in (("live", "Live"), ("history", "History"), ("settings", "Settings")):
            button = QToolButton()
            button.setObjectName("nav")
            button.setText(label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(button)
            self.nav_buttons[key] = button
            layout.addWidget(button)
        self.nav_buttons["live"].setChecked(True)
        layout.addSpacing(8)

        self.minimize = IconButton("minimize", pal, "Minimize", size=16)
        self.minimize.setObjectName("chrome")
        self.maximize = IconButton("maximize", pal, "Maximize", size=16)
        self.maximize.setObjectName("chrome")
        self.close_button = IconButton("close", pal, "Close", size=16)
        self.close_button.setObjectName("chromeClose")
        for button in (self.minimize, self.maximize, self.close_button):
            button.setFixedSize(40, 30)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            layout.addWidget(button)
        self.set_palette(pal)

    def set_palette(self, pal: Palette) -> None:
        self.logo.setPixmap(icons.pixmap("logo", pal.accent, 22))
        self.status.set_palette(pal)
        for button in (self.minimize, self.maximize, self.close_button):
            button.set_palette(pal)


class LivePage(QWidget):
    def __init__(self, cfg: Config, pal: Palette, model: TranscriptModel, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self._pal = pal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 12)
        layout.setSpacing(10)

        # -- controls --
        bar = QHBoxLayout()
        bar.setSpacing(8)
        options = [(key, PROFILE_LABELS[key]) for key in PROFILES] + [("custom", "Custom")]
        self.profile = SegmentedControl(options)
        self.profile.set_tooltips(PROFILE_BLURBS)
        bar.addWidget(self.profile)
        bar.addSpacing(6)

        # One button that is Start or Stop, whichever applies. Showing both
        # with one of them greyed out spent the widest part of the toolbar
        # on a control that could not be pressed.
        self.start = QPushButton("Start")
        self.start.setObjectName("primary")
        self.start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start.setMinimumWidth(112)
        self.start.setToolTip("Start translating  (F5)")
        bar.addWidget(self.start)
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("stop")
        self.stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop.setMinimumWidth(112)
        self.stop.setToolTip("Stop translating  (F5)")
        self.stop.hide()
        bar.addWidget(self.stop)
        self.pause = QPushButton("Pause")
        self.pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause.setMinimumWidth(92)
        self.pause.setToolTip("Stop listening for a moment without unloading anything  (F6)")
        self.pause.setEnabled(False)
        bar.addWidget(self.pause)
        bar.addStretch(1)

        self.show_source = ToggleChip(
            "source", "Japanese", pal, "Show the original Japanese under each line"
        )
        self.overlay = ToggleChip(
            "captions", "Overlay", pal, "Draw captions over the screen as well"
        )
        self.metrics_button = ToggleChip(
            "chart", "Stats", pal, "Show delay and performance figures"
        )
        self.copy_latest = IconButton("copy", pal, "Copy the latest translation")
        self.copy_all = IconButton("copy_all", pal, "Copy the whole transcript")
        self.clear = IconButton("trash", pal, "Clear the transcript")
        self._icon_buttons = [
            self.show_source, self.overlay, self.metrics_button,
            self.copy_latest, self.copy_all, self.clear,
        ]
        for button in (self.show_source, self.overlay, self.metrics_button):
            bar.addWidget(button)
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedSize(1, 20)
        divider.setStyleSheet("max-height: 20px; min-height: 20px;")
        bar.addWidget(divider)
        for button in (self.copy_latest, self.copy_all, self.clear):
            bar.addWidget(button)
        layout.addLayout(bar)

        self.banner = Banner(pal)
        layout.addWidget(self.banner)

        # -- transcript, with the metrics panel beside it when shown --
        middle = QHBoxLayout()
        middle.setSpacing(10)
        well = QFrame()
        well.setObjectName("well")
        well_layout = QVBoxLayout(well)
        well_layout.setContentsMargins(2, 6, 2, 6)
        self.look = Look(
            pal, float(cfg.translation_font_size), cfg.show_source, cfg.show_timestamps
        )
        self.transcript = TranscriptView(self.look)
        self.transcript.setModel(model)
        self.transcript.empty_title = "Ready to translate"
        self.transcript.empty_text = "Press Start, then play anything in Japanese."
        well_layout.addWidget(self.transcript)
        middle.addWidget(well, stretch=1)
        self.metrics = MetricsPanel(pal)
        self.metrics.setVisible(cfg.metrics_visible)
        middle.addWidget(self.metrics)
        layout.addLayout(middle, stretch=1)

        self.live = LiveCard(pal, float(cfg.translation_font_size))
        self.live.set_show_source(cfg.show_source)
        layout.addWidget(self.live)

        # -- footer --
        foot = QHBoxLayout()
        foot.setContentsMargins(4, 0, 4, 0)
        self.health = QLabel("")
        self.health.setObjectName("hint")
        foot.addWidget(self.health)
        foot.addStretch(1)
        self.engine = QLabel("")
        self.engine.setObjectName("hint")
        foot.addWidget(self.engine)
        layout.addLayout(foot)

    def set_palette(self, pal: Palette) -> None:
        self._pal = pal
        for button in self._icon_buttons:
            button.set_palette(pal)
        self.banner.set_palette(pal)
        self.live.set_palette(pal)
        self.metrics.set_palette(pal)
        self.look.palette = pal
        self.transcript.set_look(self.look)

    def set_look(self, cfg: Config) -> None:
        self.look.english_pt = float(cfg.translation_font_size)
        self.look.show_source = cfg.show_source
        self.look.show_timestamps = cfg.show_timestamps
        self.transcript.set_look(self.look)
        self.live.set_english_size(self.look.english_pt)
        self.live.set_show_source(cfg.show_source)


class HistoryPage(QWidget):
    export_requested = pyqtSignal()
    copy_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self, cfg: Config, pal: Palette, model: TranscriptModel, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self._model = model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 12)
        layout.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search this session, in English or Japanese")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(320)
        bar.addWidget(self.search, stretch=1)
        self.count = QLabel("")
        self.count.setObjectName("hint")
        bar.addWidget(self.count)
        self.copy = QPushButton("Copy")
        self.copy.setToolTip("Copy the lines shown")
        self.export = QPushButton("Export...")
        self.export.setToolTip("Save the lines shown as a text file")
        self.clear = QPushButton("Clear")
        self.clear.setObjectName("danger")
        for button in (self.copy, self.export, self.clear):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            bar.addWidget(button)
        layout.addLayout(bar)
        self.copy.clicked.connect(self.copy_requested.emit)
        self.export.clicked.connect(self.export_requested.emit)
        self.clear.clicked.connect(self.clear_requested.emit)

        self.proxy = SearchProxy(self)
        self.proxy.setSourceModel(model)
        well = QFrame()
        well.setObjectName("well")
        well_layout = QVBoxLayout(well)
        well_layout.setContentsMargins(2, 6, 2, 6)
        self.look = Look(pal, float(cfg.translation_font_size), True, True)
        self.view = TranscriptView(self.look)
        self.view.follow = False
        self.view.setModel(self.proxy)
        self.view.empty_text = "Nothing to show yet."
        well_layout.addWidget(self.view)
        layout.addWidget(well, stretch=1)

        self.search.textChanged.connect(self._search)
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            signal.connect(lambda *_a: self._recount())
        self._recount()

    def _search(self, text: str) -> None:
        self.proxy.set_query(text)
        self.view.empty_text = (
            f"No line contains “{text.strip()}”." if text.strip() else "Nothing to show yet."
        )
        self._recount()

    def _recount(self) -> None:
        shown, total = self.proxy.rowCount(), self._model.rowCount()
        if self.proxy.query:
            self.count.setText(f"{shown} of {total} lines")
        else:
            self.count.setText(f"{total} line{'s' if total != 1 else ''}")

    def visible_entries(self) -> list:
        from .transcript import ENTRY_ROLE

        return [
            self.proxy.index(row, 0).data(ENTRY_ROLE) for row in range(self.proxy.rowCount())
        ]

    def set_palette(self, pal: Palette) -> None:
        self.look.palette = pal
        self.view.set_look(self.look)

    def set_look(self, cfg: Config) -> None:
        self.look.english_pt = float(cfg.translation_font_size)
        self.view.set_look(self.look)


class MainWindow(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    pause_requested = pyqtSignal(bool)
    profile_chosen = pyqtSignal(str)
    clear_requested = pyqtSignal()
    copy_latest_requested = pyqtSignal()
    copy_all_requested = pyqtSignal()
    overlay_toggled = pyqtSignal(bool)
    source_toggled = pyqtSignal(bool)
    metrics_toggled = pyqtSignal(bool)
    banner_action = pyqtSignal(str)
    closing = pyqtSignal()
    page_changed = pyqtSignal(str)

    def __init__(self, cfg: Config, model: TranscriptModel, history: History):
        super().__init__()
        self.cfg = cfg
        self._history = history
        self._pal = palette(cfg.theme)
        self._frame = _NativeFrame()
        self._paused = False
        self._running = False
        self._drag_from: QPoint | None = None
        self.setWindowTitle("SubOrdinant")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.setMinimumSize(820, 560)
        self.resize(1040, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._root = QWidget()
        self._root.setObjectName("root")
        outer.addWidget(self._root)
        layout = QVBoxLayout(self._root)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.title = TitleBar(self._pal)
        layout.addWidget(self.title)

        self.stack = QStackedWidget()
        self.live_page = LivePage(cfg, self._pal, model)
        self.history_page = HistoryPage(cfg, self._pal, model)
        self.settings_page = SettingsPage(cfg, self._pal)
        self._pages = {
            "live": self.live_page,
            "history": self.history_page,
            "settings": self.settings_page,
        }
        for page in self._pages.values():
            self.stack.addWidget(page)
        layout.addWidget(self.stack, stretch=1)

        # -- wiring --
        for key, button in self.title.nav_buttons.items():
            button.clicked.connect(lambda _c, k=key: self.show_page(k))
        self.title.minimize.clicked.connect(self.showMinimized)
        self.title.maximize.clicked.connect(self._toggle_maximize)
        self.title.close_button.clicked.connect(self.close)

        live = self.live_page
        live.start.clicked.connect(self.start_requested.emit)
        live.stop.clicked.connect(self.stop_requested.emit)
        live.pause.clicked.connect(lambda: self.pause_requested.emit(not self._paused))
        live.profile.changed.connect(self.profile_chosen.emit)
        live.clear.clicked.connect(self.clear_requested.emit)
        live.copy_latest.clicked.connect(self.copy_latest_requested.emit)
        live.copy_all.clicked.connect(self.copy_all_requested.emit)
        live.overlay.toggled.connect(self.overlay_toggled.emit)
        live.show_source.toggled.connect(self.source_toggled.emit)
        live.metrics_button.toggled.connect(self._toggle_metrics)
        live.banner.action.connect(self.banner_action.emit)
        live.transcript.copied.connect(lambda _t: self.toast("Copied"))
        self.history_page.view.copied.connect(lambda _t: self.toast("Copied"))
        self.history_page.copy_requested.connect(self._copy_history)
        self.history_page.export_requested.connect(self._export_history)
        self.history_page.clear_requested.connect(self.clear_requested.emit)

        # Keys for the things done while watching something else.
        for keys, action in (
            ("F5", self._toggle_run),
            ("F6", lambda: live.pause.click() if live.pause.isEnabled() else None),
            ("Ctrl+1", lambda: self.show_page("live")),
            ("Ctrl+2", lambda: self.show_page("history")),
            ("Ctrl+3", lambda: self.show_page("settings")),
            ("Ctrl+F", lambda: self.show_page("history")),
            ("Ctrl+,", lambda: self.show_page("settings")),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(action)

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.live_page.health.setText(self._health))
        self._health = ""

        self.apply_theme()
        self.sync_from_config()
        self.set_state("idle", "")

    def _toggle_run(self) -> None:
        if self._running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_ready_summary(self, lines: list[str]) -> None:
        """What pressing Start will run, shown while the transcript is empty."""
        self.live_page.transcript.empty_text = "\n".join(
            [*lines, "", "Press Start, then play anything in Japanese."]
        )
        self.live_page.transcript.viewport().update()

    # -- native frame ------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._frame.active:
            self._frame.install(int(self.winId()), self._pal.name == "dark", self._pal.border_strong)

    def nativeEvent(self, event_type, message):  # noqa: N802
        # "Not handled" is returned directly rather than by calling the base
        # class. QWidget.nativeEvent does nothing, and calling it through
        # PyQt6 faults the process with an access violation - found the hard
        # way, with no traceback, the first time this window was shown.
        if bytes(event_type) == b"windows_generic_MSG":
            handled, result = self._frame.handle(self, message)
            if handled:
                return True, result
        return False, 0

    def is_drag_area(self, pos: QPoint) -> bool:
        """Is this point part of the title bar that drags the window?"""
        if pos.y() < 0 or pos.y() > TITLE_HEIGHT or pos.x() < 0 or pos.x() > self.width():
            return False
        child = self.childAt(pos)
        return child is None or child in (
            self._root, self.title, self.title.logo, self.title.name,
            self.title.status, self.title.delay,
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Only reached without the native frame; with it, Windows moves the
        # window itself and Qt never sees these clicks.
        if event.button() == Qt.MouseButton.LeftButton and self.is_drag_area(event.pos()):
            handle = self.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.is_drag_area(event.pos()):
            self._toggle_maximize()
            return
        super().mouseDoubleClickEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        # Qt sends change events from inside the constructor, before the
        # title bar exists. An exception escaping a virtual method called by
        # Qt ends the process without a traceback, so this guard is not
        # decoration.
        if getattr(self, "title", None) is None:
            return
        maximized = self.isMaximized()
        self.title.maximize.set_glyph("restore" if maximized else "maximize")
        self.title.maximize.setToolTip("Restore" if maximized else "Maximize")

    def _toggle_maximize(self) -> None:
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cfg.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        event.ignore()
        self.closing.emit()

    def restore_geometry(self) -> None:
        if not self.cfg.window_geometry:
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                self.move(
                    area.x() + (area.width() - self.width()) // 2,
                    area.y() + (area.height() - self.height()) // 2,
                )
            return
        try:
            self.restoreGeometry(QByteArray.fromBase64(self.cfg.window_geometry.encode("ascii")))
        except Exception:  # noqa: BLE001 - a bad blob just means default placement
            log.debug("could not restore window geometry", exc_info=True)

    # -- appearance --------------------------------------------------------

    def apply_theme(self) -> None:
        self._pal = palette(self.cfg.theme)
        self.setStyleSheet(stylesheet(self._pal, self.cfg.ui_font_size))
        self.title.set_palette(self._pal)
        self.live_page.set_palette(self._pal)
        self.history_page.set_palette(self._pal)
        self.settings_page.set_palette(self._pal)
        self.live_page.set_look(self.cfg)
        self.history_page.set_look(self.cfg)
        if self._frame.active:
            self._frame.recolor(int(self.winId()), self._pal.name == "dark", self._pal.border_strong)

    @property
    def pal(self) -> Palette:
        return self._pal

    def sync_from_config(self) -> None:
        """Point the Live page's controls at what the config says."""
        live = self.live_page
        name = detect_profile(self.cfg)
        live.profile.set_option_visible("custom", name == "custom")
        live.profile.set_current(name)
        for button, value in (
            (live.show_source, self.cfg.show_source),
            (live.overlay, self.cfg.overlay_enabled),
            (live.metrics_button, self.cfg.metrics_visible),
        ):
            button.blockSignals(True)
            button.setChecked(value)
            button.blockSignals(False)
            button._refresh()
        live.metrics.setVisible(self.cfg.metrics_visible)
        live.set_look(self.cfg)
        self.history_page.set_look(self.cfg)

    def _toggle_metrics(self, on: bool) -> None:
        self.live_page.metrics.setVisible(on)
        self.metrics_toggled.emit(on)

    # -- navigation --------------------------------------------------------

    def show_page(self, key: str) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        self.title.nav_buttons[key].setChecked(True)
        if key == "history":
            self.history_page.search.setFocus()
        self.page_changed.emit(key)

    # -- state -------------------------------------------------------------

    def set_state(self, state: str, detail: str) -> None:
        self.title.status.set_state(state, detail)
        live = self.live_page
        live.live.set_state(state)
        running = state in ("loading", "listening", "recognizing", "translating", "paused")
        self._running = running
        self._paused = state == "paused"
        live.start.setVisible(not running)
        live.start.setText("Start" if state != "error" else "Try again")
        live.stop.setVisible(running)
        live.stop.setText("Cancel" if state == "loading" else "Stop")
        live.pause.setEnabled(state in ("listening", "recognizing", "translating", "paused"))
        live.pause.setText("Resume" if self._paused else "Pause")
        if not running:
            live.live.clear()
            live.metrics.reset() if state == "idle" else None
            self._health = ""
            live.health.setText("")
            self.title.delay.hide()

    def set_engine_label(self, text: str) -> None:
        self.live_page.engine.setText(text)

    def update_metrics(self, m: dict) -> None:
        self.live_page.metrics.update_metrics(m)
        parts = [m.get("health", "")]
        final = m.get("final_latency_sec")
        chip = self.title.delay
        if final is not None and self._running:
            chip.setText(f"{final:.1f} s behind")
            tone = "ok" if m.get("keeping_up", True) and final < 2.5 else "warn"
            if chip.property("tone") != tone:
                chip.setProperty("tone", tone)
                chip.style().unpolish(chip)
                chip.style().polish(chip)
            chip.show()
        if final is not None:
            parts.append(f"{final:.1f} s behind")
        if m.get("mean_pass_ms") is not None:
            parts.append(f"{m['mean_pass_ms']:.0f} ms a pass")
        self._health = "   ·   ".join(p for p in parts if p)
        if not self._toast_timer.isActive():
            self.live_page.health.setText(self._health)
        self.live_page.health.setStyleSheet(
            "" if m.get("keeping_up", True) else f"color: {self._pal.warn};"
        )
        self.set_engine_label(
            "   ·   ".join(p for p in (m.get("engine", ""), m.get("translator", "")) if p)
        )

    def toast(self, text: str, ms: int = 1600) -> None:
        """A word in the footer, briefly: confirmation that a click did something."""
        self.live_page.health.setText(text)
        self._toast_timer.start(ms)

    def notify(self, level: str, message: str, action_label: str = "", action_key: str = "") -> None:
        self.live_page.banner.show_message(level, message, action_label, action_key)

    def dismiss_notice(self) -> None:
        self.live_page.banner.hide()

    # -- history actions ---------------------------------------------------

    def _copy_history(self) -> None:
        entries = self.history_page.visible_entries()
        text = self._history.as_text(source=True, timestamps=True, entries=entries)
        if text:
            QGuiApplication.clipboard().setText(text)
            self.history_page.count.setText(f"Copied {len(entries)} lines")
            QTimer.singleShot(1600, self.history_page._recount)

    def _export_history(self) -> None:
        entries = self.history_page.visible_entries()
        if not entries:
            return
        import time

        suggested = f"SubOrdinant {time.strftime('%Y-%m-%d %H%M')}.txt"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export transcript", suggested, "Text file (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig") as handle:
                handle.write(self._history.as_text(source=True, timestamps=True, entries=entries))
                handle.write("\n")
        except OSError as exc:
            self.history_page.count.setText(f"Could not save: {exc}")
            return
        self.history_page.count.setText("Saved")
        QTimer.singleShot(1600, self.history_page._recount)
