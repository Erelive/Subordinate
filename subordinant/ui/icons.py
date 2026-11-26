"""The interface glyphs, drawn with QPainter.

Drawn rather than shipped for two reasons. They take the theme's colours
directly, so switching between dark and light needs no second set of files.
And the packaged build carries no icon assets that can go missing - the same
reasoning as the tray icon's drawn fallback in overlay.py.

Every glyph is described on a 20x20 grid and rendered at the device pixel
ratio, so it stays sharp at 150% scaling.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

GRID = 20.0


def _pen(color: QColor, width: float = 1.6) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _play(p: QPainter, c: QColor) -> None:
    path = QPainterPath()
    path.moveTo(7.0, 4.6)
    path.lineTo(15.4, 10.0)
    path.lineTo(7.0, 15.4)
    path.closeSubpath()
    p.setPen(_pen(c, 1.4))
    p.setBrush(c)
    p.drawPath(path)


def _pause(p: QPainter, c: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(5.6, 4.8, 3.2, 10.4), 1.0, 1.0)
    p.drawRoundedRect(QRectF(11.2, 4.8, 3.2, 10.4), 1.0, 1.0)


def _stop(p: QPainter, c: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(5.4, 5.4, 9.2, 9.2), 1.8, 1.8)


def _copy(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(7.2, 7.2, 8.6, 8.6), 1.8, 1.8)
    path = QPainterPath()
    path.moveTo(12.8, 4.2)
    path.lineTo(6.0, 4.2)
    path.quadTo(4.2, 4.2, 4.2, 6.0)
    path.lineTo(4.2, 12.8)
    p.drawPath(path)


def _copy_all(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(4.4, 3.6, 11.2, 12.8), 2.0, 2.0)
    for y in (7.2, 10.0, 12.8):
        p.drawLine(QPointF(7.2, y), QPointF(12.8, y))


def _trash(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(4.4, 6.0), QPointF(15.6, 6.0))
    p.drawLine(QPointF(8.2, 6.0), QPointF(8.8, 4.0))
    p.drawLine(QPointF(8.8, 4.0), QPointF(11.2, 4.0))
    p.drawLine(QPointF(11.2, 4.0), QPointF(11.8, 6.0))
    path = QPainterPath()
    path.moveTo(5.8, 6.2)
    path.lineTo(6.6, 15.0)
    path.quadTo(6.7, 16.2, 7.9, 16.2)
    path.lineTo(12.1, 16.2)
    path.quadTo(13.3, 16.2, 13.4, 15.0)
    path.lineTo(14.2, 6.2)
    p.drawPath(path)


def _gear(p: QPainter, c: QColor) -> None:
    # Three sliders: reads as "settings" at 16 px, where a cog turns to mush.
    p.setPen(_pen(c))
    for y, knob in ((5.6, 12.6), (10.0, 7.4), (14.4, 11.0)):
        p.drawLine(QPointF(4.0, y), QPointF(16.0, y))
        p.setBrush(c)
        p.drawEllipse(QPointF(knob, y), 1.9, 1.9)


def _search(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(8.8, 8.8), 4.6, 4.6)
    p.drawLine(QPointF(12.3, 12.3), QPointF(16.0, 16.0))


def _chart(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    path = QPainterPath()
    path.moveTo(3.6, 12.4)
    path.lineTo(7.0, 12.4)
    path.lineTo(8.8, 6.0)
    path.lineTo(11.6, 15.0)
    path.lineTo(13.2, 9.6)
    path.lineTo(16.4, 9.6)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def _minimize(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.3))
    p.drawLine(QPointF(5.4, 10.4), QPointF(14.6, 10.4))


def _maximize(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(5.4, 5.4, 9.2, 9.2), 1.4, 1.4)


def _restore(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(5.0, 7.2, 7.8, 7.8), 1.3, 1.3)
    path = QPainterPath()
    path.moveTo(7.6, 7.0)
    path.lineTo(7.6, 5.6)
    path.quadTo(7.6, 5.0, 8.4, 5.0)
    path.lineTo(14.2, 5.0)
    path.quadTo(15.0, 5.0, 15.0, 5.8)
    path.lineTo(15.0, 11.6)
    path.quadTo(15.0, 12.4, 14.2, 12.4)
    path.lineTo(13.0, 12.4)
    p.drawPath(path)


def _close(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.3))
    p.drawLine(QPointF(5.8, 5.8), QPointF(14.2, 14.2))
    p.drawLine(QPointF(14.2, 5.8), QPointF(5.8, 14.2))


def _captions(p: QPainter, c: QColor) -> None:
    # The overlay: a screen with a caption bar along its bottom edge.
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.4, 4.6, 13.2, 10.8), 2.0, 2.0)
    p.drawLine(QPointF(6.2, 12.0), QPointF(13.8, 12.0))


def _source(p: QPainter, c: QColor) -> None:
    # "Show Japanese": two lines of text, the second shorter.
    p.setPen(_pen(c))
    p.drawLine(QPointF(4.4, 7.0), QPointF(15.6, 7.0))
    p.drawLine(QPointF(4.4, 12.6), QPointF(11.6, 12.6))


def _download(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.drawLine(QPointF(10.0, 4.0), QPointF(10.0, 12.2))
    p.drawLine(QPointF(6.6, 9.0), QPointF(10.0, 12.4))
    p.drawLine(QPointF(13.4, 9.0), QPointF(10.0, 12.4))
    p.drawLine(QPointF(4.6, 15.8), QPointF(15.4, 15.8))


def _chevron(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.4))
    p.drawLine(QPointF(6.4, 8.2), QPointF(10.0, 11.8))
    p.drawLine(QPointF(13.6, 8.2), QPointF(10.0, 11.8))


def _check(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 1.8))
    p.drawLine(QPointF(5.2, 10.4), QPointF(8.6, 13.8))
    p.drawLine(QPointF(8.6, 13.8), QPointF(14.8, 6.6))


def _logo(p: QPainter, c: QColor) -> None:
    """The app mark: a caption plate, lower bar in the accent colour.

    Same idea as assets/subordinant.ico - the plate the app draws on screen,
    with the second line coloured the way unsettled text is.
    """
    p.setPen(Qt.PenStyle.NoPen)
    plate = QColor(c)
    plate.setAlpha(46)
    p.setBrush(plate)
    p.drawRoundedRect(QRectF(1.6, 3.4, 16.8, 13.2), 3.6, 3.6)
    p.setBrush(QColor("#eef0f4") if c.lightness() > 120 else QColor("#20242b"))
    p.drawRoundedRect(QRectF(4.6, 7.0, 10.8, 2.3), 1.1, 1.1)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(4.6, 11.0, 7.0, 2.3), 1.1, 1.1)


_GLYPHS = {
    "play": _play,
    "pause": _pause,
    "stop": _stop,
    "copy": _copy,
    "copy_all": _copy_all,
    "trash": _trash,
    "settings": _gear,
    "search": _search,
    "chart": _chart,
    "minimize": _minimize,
    "maximize": _maximize,
    "restore": _restore,
    "close": _close,
    "captions": _captions,
    "source": _source,
    "download": _download,
    "chevron": _chevron,
    "check": _check,
    "logo": _logo,
}


def pixmap(name: str, color: str | QColor, size: int = 18, dpr: float = 2.0) -> QPixmap:
    """One glyph as a pixmap `size` logical pixels square."""
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / GRID, size / GRID)
    _GLYPHS[name](painter, QColor(color))
    painter.end()
    return pm


def icon(name: str, color: str | QColor, size: int = 18, disabled: str | None = None) -> QIcon:
    result = QIcon()
    result.addPixmap(pixmap(name, color, size), QIcon.Mode.Normal)
    if disabled:
        result.addPixmap(pixmap(name, disabled, size), QIcon.Mode.Disabled)
    return result


def size(px: int = 18) -> QSize:
    return QSize(px, px)
