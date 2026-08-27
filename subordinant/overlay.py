"""Always-on-top caption overlay.

A frameless, click-through, translucent window pinned near the bottom of the
primary screen. Committed text is drawn solid; the unconfirmed tail is dimmed, so
a word visibly "settles" instead of the whole line rewriting itself.

Known limit: an always-on-top window cannot draw over a game running in
exclusive fullscreen. Borderless-windowed mode works.
"""

from __future__ import annotations

import html
import logging

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPixmap,
    QTextBlockFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import QWidget

from .config import Config

log = logging.getLogger(__name__)

PADDING = 18
RADIUS = 12
LINE_HEIGHT_PCT = 125


def _parse_rgba(value: str) -> QColor:
    """Accept 'rgba(r, g, b, a)' or '#rrggbb'."""
    text = value.strip()
    if text.lower().startswith("rgba"):
        parts = text[text.index("(") + 1 : text.rindex(")")].split(",")
        nums = [float(p) for p in parts]
        if len(nums) == 4:
            r, g, b, a = nums
            # Accept both 0-255 and 0-1 alpha conventions.
            alpha = int(a * 255) if a <= 1.0 else int(a)
            return QColor(int(r), int(g), int(b), alpha)
    color = QColor(text)
    return color if color.isValid() else QColor(0, 0, 0, 190)


def make_tray_icon() -> QIcon:
    """Draw a caption-bar glyph so no image asset has to ship with the app."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(28, 28, 30, 235))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(4, 12, 56, 40), 8, 8)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRoundedRect(QRectF(13, 25, 38, 6), 3, 3)
    painter.drawRoundedRect(QRectF(13, 37, 24, 6), 3, 3)
    painter.end()
    return QIcon(pixmap)


class CaptionOverlay(QWidget):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._bg = _parse_rgba(cfg.background_rgba)

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

        self._doc = QTextDocument()
        self._doc.setDefaultFont(QFont(cfg.font_family, cfg.font_size))
        self._doc.setDocumentMargin(0)

        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self.hide)

        self._committed = ""
        self._unstable = ""
        self._translation = ""

    # -- content -----------------------------------------------------------

    @pyqtSlot(str, str)
    def set_caption(self, committed: str, unstable: str) -> None:
        committed = committed.strip()
        unstable = unstable.strip()
        if committed == self._committed and unstable == self._unstable:
            return
        self._committed, self._unstable = committed, unstable
        self._render()

    @pyqtSlot(str, str, int)
    def set_translation(  # noqa: ARG002
        self, source: str, english: str, speaker: int = -1
    ) -> None:
        """Show the English for the utterance that just closed.

        Replaces rather than appends: one finished sentence at a time is what
        the MT stage produces, and the source text above it is already the
        running context.

        The speaker id is accepted and ignored. The overlay shows one line over
        a video, where a "Speaker 2" tag costs more width than it earns; the
        transcript window is where labelling pays off.
        """
        english = english.strip()
        if english == self._translation:
            return
        self._translation = english
        self._render()

    def _render(self) -> None:
        committed, unstable = self._committed, self._unstable
        translation = self._translation

        if not committed and not unstable and not translation:
            self._idle.start(int(self.cfg.hide_after_idle_sec * 1000))
            return
        self._idle.stop()

        pieces = []
        if translation:
            # The translation is what the viewer is here to read, so it gets the
            # committed colour and the transcription drops to the dim one - the
            # source text is reference, not the caption.
            pieces.append(
                f'<span style="color:{self.cfg.committed_color}">'
                f"{html.escape(translation)}</span><br>"
            )
        source_color = (
            self.cfg.unstable_color if translation else self.cfg.committed_color
        )
        if committed:
            pieces.append(
                f'<span style="color:{source_color}">{html.escape(committed)}</span>'
            )
        if unstable:
            pieces.append(
                f'<span style="color:{self.cfg.unstable_color}">'
                f"{html.escape(unstable)}</span>"
            )
        # Zero margins: QTextDocument's default paragraph margins otherwise leave
        # a band of dead space under the last line. Line spacing is applied
        # through QTextBlockFormat rather than CSS - Qt's rich text does not read
        # a CSS line-height the way a browser does, and setting it there produces
        # wildly wrong block heights.
        self._doc.setHtml('<p style="margin:0">' + " ".join(pieces) + "</p>")
        cursor = QTextCursor(self._doc)
        cursor.select(QTextCursor.SelectionType.Document)
        block = QTextBlockFormat()
        block.setLineHeight(
            LINE_HEIGHT_PCT, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        cursor.mergeBlockFormat(block)
        self._relayout()
        if not self.isVisible():
            self.show()
        self.update()

    def clear(self) -> None:
        self._translation = ""
        self.set_caption("", "")

    # -- geometry ----------------------------------------------------------

    def _relayout(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        max_text_width = int(area.width() * self.cfg.max_width_frac) - 2 * PADDING

        # Lay out once unconstrained to find the natural width, then clamp, so
        # short captions get a box that hugs the text instead of a full-width bar.
        self._doc.setTextWidth(max_text_width)
        text_width = min(self._doc.idealWidth(), float(max_text_width))
        self._doc.setTextWidth(text_width)
        size = self._doc.size()

        width = int(text_width) + 2 * PADDING
        # Never let a layout mistake push the box past the screen; a caption that
        # is clipped at the bottom is still readable, one drawn off-screen is not.
        height = min(int(size.height()) + 2 * PADDING, area.height())
        x = area.x() + (area.width() - width) // 2
        y = area.y() + area.height() - height - int(area.height() * self.cfg.bottom_margin_frac)
        log.debug(
            "relayout screen=%s avail=%dx%d+%d+%d dpr=%.2f max_text=%d "
            "ideal=%.0f text_w=%.0f box=%dx%d at %d,%d",
            screen.name(),
            area.width(),
            area.height(),
            area.x(),
            area.y(),
            screen.devicePixelRatio(),
            max_text_width,
            self._doc.idealWidth(),
            text_width,
            width,
            height,
            x,
            y,
        )
        self.setGeometry(x, y, width, height)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ARG002, N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), RADIUS, RADIUS)
        painter.fillPath(path, self._bg)

        painter.translate(PADDING, PADDING)
        self._doc.drawContents(painter)
        painter.end()
