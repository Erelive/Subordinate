"""Always-on-top caption overlay.

A frameless, click-through, translucent window near the bottom (or top) of the
primary screen: English subtitles for whatever is playing, with the Japanese
optionally underneath.

Two things about how it is built are deliberate.

It never changes size while it is running. The window is as wide as the
caption is allowed to be and as tall as the most text it will ever show, and
stays that way; what changes is the plate painted inside it, which hugs the
text. Resizing a layered window means the compositor re-creates its surface,
and doing that on every word is what made the old overlay shimmer. A window
that never moves or resizes is also the kindest thing to put over a game.

It shows two lines of thought at most. The most recent finished sentence, in
white, stays for a few seconds or until the next one lands. Under it, dimmer
and in italics, is the draft of the sentence being spoken now. The draft can
rewrite itself; the finished line never does.

Known limit: an always-on-top window cannot draw over a game running in
exclusive fullscreen. Borderless-windowed mode works.
"""

from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPixmap,
    QTextLayout,
    QTextOption,
)
from PyQt6.QtWidgets import QWidget

from .config import Config, asset_path

log = logging.getLogger(__name__)

PADDING_X = 22
PADDING_Y = 12
RADIUS = 12
BLOCK_GAP = 6
# Lines each kind of text may take before it is cut from the front.
FINAL_LINES = 3
DRAFT_LINES = 2
SOURCE_LINES = 1


def app_icon() -> QIcon:
    """The app icon, for the window, the taskbar and the tray.

    assets/subordinant.ico carries nine sizes so Windows can pick one per
    context; see tools/make_icon.py, which generates it.

    Falls back to drawing the glyph if the file is missing or unreadable.
    That matters more than it looks: with the window closed to the tray, the
    tray icon is the only way back to the app, and an icon-less tray entry is
    an invisible one.
    """
    path = asset_path("subordinant.ico")
    if path.exists():
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
        log.warning("%s exists but Qt could not read it; drawing icon", path)
    else:
        log.warning("%s missing; drawing icon", path)
    return _drawn_icon()


def _drawn_icon() -> QIcon:
    """The caption-bar glyph, drawn rather than loaded."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(28, 28, 30, 235))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(4, 12, 56, 40), 8, 8)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRoundedRect(QRectF(13, 25, 38, 6), 3, 3)
    painter.setBrush(QColor(123, 140, 255))
    painter.drawRoundedRect(QRectF(13, 37, 24, 6), 3, 3)
    painter.end()
    return QIcon(pixmap)


class _Block:
    """One piece of text laid out for drawing."""

    def __init__(self, text: str, font: QFont, color: QColor, width: float, max_lines: int):
        self.color = color
        self.layout = QTextLayout(text, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layout.setTextOption(option)
        self.height, self.width = self._lay(width)
        # Too long for its allowance: drop words from the front until it
        # fits. The end of a sentence is the part still being listened to.
        guard = 0
        while self.layout.lineCount() > max_lines and guard < 200:
            guard += 1
            cut = text.find(" ", max(1, len(text) // 12))
            text = text[cut + 1 :] if cut > 0 else text[max(1, len(text) // 8) :]
            self.layout = QTextLayout("... " + text, font)
            self.layout.setTextOption(option)
            self.height, self.width = self._lay(width)

    def _lay(self, width: float) -> tuple[float, float]:
        self.layout.beginLayout()
        height = 0.0
        natural = 0.0
        while True:
            line = self.layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            line.setPosition(QPointF(0.0, height))
            height += line.height()
            natural = max(natural, line.naturalTextWidth())
        self.layout.endLayout()
        return height, natural


class CaptionOverlay(QWidget):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            # Tool keeps it out of the taskbar and the alt-tab list.
            | Qt.WindowType.Tool
            # Clicks pass through to whatever is underneath.
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowTitle("SubOrdinant captions")

        self._final = ""  # the last finished (or drafted) line
        self._final_source = ""
        self._final_is_draft = False
        self._final_at = 0.0
        self._final_id = -1
        self._preview = ""  # English for the sentence in progress
        self._caption = ""  # its Japanese, as heard so far
        self._enabled = True

        self._expire = QTimer(self)
        self._expire.setSingleShot(True)
        self._expire.timeout.connect(self._expired)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self._fade_done)
        self._hiding = False

        self.apply_settings()

    # -- settings ----------------------------------------------------------

    def apply_settings(self) -> None:
        """Re-read fonts, colours, size and position from the config."""
        cfg = self.cfg
        families = [cfg.font_family, "Segoe UI", "Yu Gothic UI", "Meiryo UI"]
        self._font = QFont()
        self._font.setFamilies(families)
        self._font.setPointSizeF(cfg.font_size)
        self._font.setWeight(QFont.Weight.DemiBold)
        self._draft_font = QFont(self._font)
        self._draft_font.setWeight(QFont.Weight.Medium)
        self._draft_font.setItalic(True)
        self._source_font = QFont()
        self._source_font.setFamilies(["Yu Gothic UI", "Meiryo UI", cfg.font_family])
        self._source_font.setPointSizeF(max(9.0, cfg.font_size * 0.62))
        self._source_font.setWeight(QFont.Weight.Medium)

        self._plate = QColor(8, 9, 12, int(round(255 * cfg.overlay_opacity / 100.0)))
        self._final_color = QColor(cfg.committed_color)
        self._draft_color = QColor("#c9cfda")
        self._source_color = QColor("#aab2c0")
        self._place()
        self.update()

    def _place(self) -> None:
        """Fix the window's geometry. Called on settings changes only."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        width = int(area.width() * self.cfg.max_width_frac)
        line = QFontMetricsF(self._font).lineSpacing()
        source_line = QFontMetricsF(self._source_font).lineSpacing()
        height = int(
            (FINAL_LINES + DRAFT_LINES) * line
            + 2 * SOURCE_LINES * source_line
            + 3 * BLOCK_GAP
            + 2 * PADDING_Y
            + 8
        )
        height = min(height, area.height())
        margin = int(area.height() * self.cfg.bottom_margin_frac)
        x = area.x() + (area.width() - width) // 2
        if self.cfg.overlay_position == "top":
            y = area.y() + margin
        else:
            y = area.y() + area.height() - height - margin
        self.setGeometry(x, y, width, height)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.clear()

    # -- content -----------------------------------------------------------

    def show_utterance(self, utterance) -> None:
        """A line from the transcript: a draft of it, or the finished thing."""
        if not self._enabled:
            return
        text = (utterance.english or utterance.source).strip()
        if not text:
            return
        self._final = text
        self._final_source = utterance.source.strip() if utterance.english else ""
        self._final_is_draft = not utterance.final
        self._final_id = utterance.id
        self._final_at = time.monotonic()
        self._refresh()

    def set_preview(self, english: str) -> None:
        if not self._enabled:
            return
        english = english.strip()
        if english == self._preview:
            return
        self._preview = english
        self._refresh()

    def set_caption(self, committed: str, unstable: str) -> None:
        """The Japanese of the sentence in progress."""
        if not self._enabled:
            return
        caption = f"{committed}{unstable}".strip()
        if caption == self._caption:
            return
        self._caption = caption
        # Only drawn beside a preview, and only if Japanese is switched on,
        # so by itself it is not a reason to repaint.
        if self.cfg.overlay_show_source or not self.cfg.mt_enabled:
            self._refresh()

    def clear(self) -> None:
        self._final = self._final_source = self._preview = self._caption = ""
        self._expire.stop()
        self._fade.stop()
        self._hiding = False
        self.hide()

    # -- visibility --------------------------------------------------------

    def _has_content(self) -> bool:
        if self._final or self._preview:
            return True
        # With no translator the Japanese is the caption.
        return bool(self._caption and not self.cfg.mt_enabled)

    def _refresh(self) -> None:
        if not self._has_content():
            self._fade_out()
            return
        if self._final:
            remaining = self.cfg.display_history_sec - (time.monotonic() - self._final_at)
            self._expire.start(max(50, int(remaining * 1000)))
        if not self.isVisible() or self._hiding:
            self._fade_in()
        self.update()

    def _expired(self) -> None:
        """The finished line has been up long enough."""
        self._final = self._final_source = ""
        self._refresh()

    def _fade_in(self) -> None:
        self._hiding = False
        self._fade.stop()
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
        self._fade.setDuration(120)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(1.0)
        self._fade.start()

    def _fade_out(self) -> None:
        if not self.isVisible() or self._hiding:
            return
        self._hiding = True
        self._fade.stop()
        self._fade.setDuration(260)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _fade_done(self) -> None:
        if self._hiding:
            self._hiding = False
            self.hide()

    # -- painting ----------------------------------------------------------

    def _blocks(self, width: float) -> list[_Block]:
        blocks: list[_Block] = []
        show_source = self.cfg.overlay_show_source
        if self._final:
            blocks.append(
                _Block(
                    self._final,
                    self._draft_font if self._final_is_draft else self._font,
                    self._draft_color if self._final_is_draft else self._final_color,
                    width,
                    FINAL_LINES,
                )
            )
            if show_source and self._final_source:
                blocks.append(
                    _Block(self._final_source, self._source_font, self._source_color, width, SOURCE_LINES)
                )
        if self._preview:
            blocks.append(
                _Block(self._preview, self._draft_font, self._draft_color, width, DRAFT_LINES)
            )
        if self._caption and (show_source or not self.cfg.mt_enabled) and (
            self._preview or not self.cfg.mt_enabled
        ):
            blocks.append(
                _Block(self._caption, self._source_font, self._source_color, width, SOURCE_LINES)
            )
        return blocks

    def paintEvent(self, event) -> None:  # noqa: ARG002, N802 - Qt naming
        text_width = self.width() - 2 * PADDING_X
        blocks = self._blocks(text_width)
        if not blocks:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        content_h = sum(b.height for b in blocks) + BLOCK_GAP * (len(blocks) - 1)
        content_w = max(b.width for b in blocks)
        plate_w = min(self.width(), content_w + 2 * PADDING_X)
        plate_h = min(self.height(), content_h + 2 * PADDING_Y)
        plate_x = (self.width() - plate_w) / 2.0
        # Anchored to the edge the overlay sits at, so the plate grows away
        # from that edge and the newest line stays put.
        plate_y = 0.0 if self.cfg.overlay_position == "top" else self.height() - plate_h

        if self._plate.alpha() > 0:
            path = QPainterPath()
            path.addRoundedRect(QRectF(plate_x, plate_y, plate_w, plate_h), RADIUS, RADIUS)
            painter.fillPath(path, self._plate)

        y = plate_y + PADDING_Y
        shadow = QColor(0, 0, 0, 215)
        for block in blocks:
            origin = QPointF(PADDING_X, y)
            # A dark halo, so the text reads over bright video even with the
            # plate turned all the way down.
            painter.setPen(shadow)
            for dx, dy in ((-1.2, 0.0), (1.2, 0.0), (0.0, -1.2), (0.0, 1.6), (1.0, 1.4)):
                block.layout.draw(painter, origin + QPointF(dx, dy))
            painter.setPen(block.color)
            block.layout.draw(painter, origin)
            y += block.height + BLOCK_GAP
        painter.end()
