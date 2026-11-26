"""Small controls the pages are built from."""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .theme import STATE_COLORS, STATE_LABELS, Palette


class Themed:
    """Mixin for widgets that paint with palette colours themselves.

    Stylesheets reach anything Qt draws. They do not reach paintEvent, so
    widgets that draw their own pixels are handed the palette directly and
    told when it changes.
    """

    _palette: Palette

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()  # type: ignore[attr-defined]


class StatusPill(QWidget, Themed):
    """A coloured dot and a short label: what the app is doing right now.

    The dot breathes while the app is working - loading, hearing speech,
    translating - and holds still when it is idle or waiting, so the two can
    be told apart from across the room without reading the text.
    """

    _BUSY = {"loading", "recognizing", "translating"}

    def __init__(self, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self._palette = palette
        self._state = "idle"
        self._label = STATE_LABELS["idle"]
        self._detail = ""
        self._pulse = 1.0
        self._anim = QPropertyAnimation(self, b"pulse", self)
        self._anim.setDuration(1100)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.5, 0.35)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(120)

    def _get_pulse(self) -> float:
        return self._pulse

    def _set_pulse(self, value: float) -> None:
        self._pulse = value
        self.update()

    pulse = pyqtProperty(float, _get_pulse, _set_pulse)

    def set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self._label = STATE_LABELS.get(state, state.title())
        self._detail = detail
        if state in self._BUSY:
            if self._anim.state() != QPropertyAnimation.State.Running:
                self._anim.start()
        else:
            self._anim.stop()
            self._pulse = 1.0
        self.setToolTip(f"{self._label} - {detail}" if detail else self._label)
        self.update()

    @property
    def state(self) -> str:
        return self._state

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(260, 30)

    def paintEvent(self, event) -> None:  # noqa: ARG002, N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self._palette.color(STATE_COLORS.get(self._state, "text_3"))
        cy = self.height() / 2.0
        halo = QColor(color)
        halo.setAlphaF(0.22 * self._pulse)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QRectF(2, cy - 8, 16, 16))
        dot = QColor(color)
        dot.setAlphaF(0.55 + 0.45 * self._pulse)
        p.setBrush(dot)
        p.drawEllipse(QRectF(6, cy - 4, 8, 8))

        font = self.font()
        font.setWeight(600)
        p.setFont(font)
        metrics = QFontMetrics(font)
        x = 26
        p.setPen(self._palette.color("text"))
        p.drawText(
            QRectF(x, 0, self.width() - x, self.height()),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._label,
        )
        x += metrics.horizontalAdvance(self._label) + 10
        if self._detail and x < self.width() - 20:
            font.setWeight(400)
            p.setFont(font)
            metrics = QFontMetrics(font)
            p.setPen(self._palette.color("text_3"))
            text = metrics.elidedText(
                self._detail, Qt.TextElideMode.ElideRight, int(self.width() - x)
            )
            p.drawText(
                QRectF(x, 0, self.width() - x, self.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                text,
            )
        p.end()


class ToggleSwitch(QAbstractButton, Themed):
    """An on/off switch that slides."""

    def __init__(self, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self._palette = palette
        self._position = 0.0
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(38, 22)
        self._anim = QPropertyAnimation(self, b"position", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get_position(self) -> float:
        return self._position

    def _set_position(self, value: float) -> None:
        self._position = value
        self.update()

    position = pyqtProperty(float, _get_position, _set_position)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._position)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:  # noqa: N802
        # Set programmatically, as when a page loads: jump, do not slide.
        super().setChecked(on)
        self._anim.stop()
        self._position = 1.0 if on else 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002, N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        off = self._palette.color("border_strong")
        on = self._palette.color("accent")
        t = self._position
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            track.setAlphaF(0.45)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 11, 11)
        knob = QColor("#ffffff") if self._palette.name == "dark" or t > 0.5 else QColor("#ffffff")
        p.setBrush(knob)
        x = 3 + t * (self.width() - 22)
        p.drawEllipse(QRectF(x, 3, 16, 16))
        p.end()


class SegmentedControl(QFrame):
    """A row of mutually exclusive choices. Emits changed(key)."""

    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("segmented")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for key, label in options:
            button = QPushButton(label)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, k=key: self.changed.emit(k))
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)

    def set_current(self, key: str) -> None:
        """Select `key` without emitting. An unknown key selects nothing."""
        button = self._buttons.get(key)
        self._group.setExclusive(False)
        for b in self._buttons.values():
            b.setChecked(b is button)
        self._group.setExclusive(True)

    def current(self) -> str | None:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return None

    def set_tooltips(self, tips: dict[str, str]) -> None:
        for key, tip in tips.items():
            if key in self._buttons:
                self._buttons[key].setToolTip(tip)

    def set_option_visible(self, key: str, visible: bool) -> None:
        if key in self._buttons:
            self._buttons[key].setVisible(visible)


class IconButton(QToolButton, Themed):
    """A tool button whose glyph follows the theme."""

    def __init__(
        self,
        name: str,
        palette: Palette,
        tooltip: str = "",
        checkable: bool = False,
        size: int = 18,
        color: str = "text_2",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._name = name
        self._size = size
        self._color = color
        self._palette = palette
        self.setCheckable(checkable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(icons.size(size))
        if tooltip:
            self.setToolTip(tooltip)
        self.toggled.connect(lambda _on: self._refresh())
        self._refresh()

    def set_glyph(self, name: str) -> None:
        self._name = name
        self._refresh()

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._refresh()

    def _refresh(self) -> None:
        tone = "accent" if self.isCheckable() and self.isChecked() else self._color
        self.setIcon(
            icons.icon(
                self._name,
                self._palette.color(tone),
                self._size,
                disabled=self._palette.text_3,
            )
        )


class ToggleChip(QPushButton, Themed):
    """An on/off control that says what it is: a glyph and a word.

    The Live page's toggles were glyphs alone, and a glyph alone has to be
    hovered over to be understood. These are the three things people switch
    while watching, so they are spelled out.
    """

    def __init__(
        self,
        name: str,
        label: str,
        palette: Palette,
        tooltip: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(label, parent)
        self.setObjectName("chip")
        self._name = name
        self._palette = palette
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(icons.size(16))
        if tooltip:
            self.setToolTip(tooltip)
        self.toggled.connect(lambda _on: self._refresh())
        self._refresh()

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._refresh()

    def _refresh(self) -> None:
        tone = "accent" if self.isChecked() else "text_2"
        self.setIcon(
            icons.icon(self._name, self._palette.color(tone), 16, disabled=self._palette.text_3)
        )


class Card(QFrame):
    """A titled panel. Rows are added with add_row / add_widget."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 14, 18, 14)
        self._layout.setSpacing(0)
        self._rows = 0
        if title:
            label = QLabel(title.upper())
            label.setObjectName("cardTitle")
            self._layout.addWidget(label)
            self._layout.addSpacing(6)

    def add_widget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def add_row(self, row: "SettingRow") -> "SettingRow":
        if self._rows:
            line = QFrame()
            line.setObjectName("divider")
            line.setFixedHeight(1)
            self._layout.addWidget(line)
            # Hidden together: a row that is not shown must not leave its
            # separator behind to double up with the next one.
            row.divider = line
        self._layout.addWidget(row)
        self._rows += 1
        return row


class SettingRow(QWidget):
    """One setting: what it is and what it does on the left, the control on
    the right. `help` is the sentence that makes the control understandable
    without opening a manual."""

    def __init__(
        self,
        title: str,
        help: str = "",  # noqa: A002 - the natural name for the argument
        control: QWidget | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 11, 0, 11)
        layout.setSpacing(18)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("rowTitle")
        text.addWidget(self.title)
        self.help = QLabel(help)
        self.help.setObjectName("rowHelp")
        self.help.setWordWrap(True)
        self.help.setVisible(bool(help))
        text.addWidget(self.help)
        layout.addLayout(text, stretch=1)
        self.control = control
        self.divider: QFrame | None = None
        if control is not None:
            layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

    def setVisible(self, visible: bool) -> None:  # noqa: N802
        super().setVisible(visible)
        if self.divider is not None:
            self.divider.setVisible(visible)

    def set_help(self, text: str) -> None:
        self.help.setText(text)
        self.help.setVisible(bool(text))


class Banner(QFrame):
    """A one-line notice with an optional action and a dismiss button.

    This is how the pipeline's on_notice reaches the user: a fallback was
    used, the output device changed, a model is missing. It stays until
    dismissed or replaced, unlike a tray balloon, which is gone before it
    has been read.
    """

    action = pyqtSignal(str)

    def __init__(self, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("banner")
        self._palette = palette
        self._action_key = ""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 6, 7)
        layout.setSpacing(10)
        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        layout.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._text, stretch=1)
        self._button = QPushButton()
        self._button.setObjectName("link")
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.clicked.connect(lambda: self.action.emit(self._action_key))
        layout.addWidget(self._button)
        self._dismiss = IconButton("close", palette, "Dismiss", size=14, color="text_3")
        self._dismiss.clicked.connect(self.hide)
        layout.addWidget(self._dismiss)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._dismiss.set_palette(palette)

    def show_message(
        self,
        level: str,
        message: str,
        action_label: str = "",
        action_key: str = "",
        timeout_ms: int = 0,
    ) -> None:
        tone = {"error": "error", "warning": "warn"}.get(level, "accent")
        self._dot.setStyleSheet(
            f"background: {getattr(self._palette, tone)}; border-radius: 4px;"
        )
        self._text.setText(message)
        self._action_key = action_key
        self._button.setText(action_label)
        self._button.setVisible(bool(action_label))
        self.setProperty("level", level)
        self.style().unpolish(self)
        self.style().polish(self)
        self.show()
        self._timer.stop()
        if timeout_ms:
            self._timer.start(timeout_ms)


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    line.setFixedHeight(1)
    return line
