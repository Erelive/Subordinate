"""The transcript: a list of finished lines, painted.

A model/view list with a painting delegate rather than a text box or a stack
of label widgets. A text box cannot give each line its own hover state and
copy button; a few thousand label widgets is a few thousand native layouts
every time the window is resized. A delegate draws only the rows on screen
and measures the rest from a cache.

Each row is one utterance:

    12:04:31   I went to Tokyo with my friends today.     <- the English
               今日は友達と東京に行ってきた。              <- the Japanese

The English is the largest and brightest text in the window, because it is
what the window is for. A line that is still a draft is drawn dimmer and in
italics, and brightens in place when its final version arrives.

A finished line can change once more: a few seconds later the second look may
correct it. That line carries a small dot beside it from then on, and hovering
shows what it said before, so a reader who saw the first version is not left
wondering whether they misread it.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPointF,
    QRect,
    QRectF,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QTextLayout,
    QTextOption,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QWidget,
)

from ..diarize import UNKNOWN, speaker_label
from ..history import Entry, History
from . import icons
from .theme import SPEAKER_COLORS, Palette, display_font, ui_font

ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1

PAD_X = 16
PAD_Y = 9
GAP = 3  # between the English and the Japanese
COPY_BOX = 26


class TranscriptModel(QAbstractListModel):
    def __init__(self, history: History, parent=None):
        super().__init__(parent)
        self.history = history

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.history.entries)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.history.entries):
            return None
        entry = self.history.entries[index.row()]
        if role == ENTRY_ROLE:
            return entry
        if role == Qt.ItemDataRole.DisplayRole:
            return entry.primary
        if role == Qt.ItemDataRole.ToolTipRole and entry.revised:
            lines = ["Corrected on a second look."]
            if entry.first_english and entry.first_english != entry.english:
                lines.append(f"First shown as: {entry.first_english}")
            if entry.first_source and entry.first_source != entry.source:
                lines.append(f"First heard as: {entry.first_source}")
            return "\n".join(lines)
        return None

    def add(self, utterance) -> None:
        """Insert a new line or update a draft in place."""
        existing = utterance.id in self.history._index
        if existing:
            row, _new, _dropped = self.history.upsert(utterance)
            index = self.index(row)
            self.dataChanged.emit(index, index, [ENTRY_ROLE, Qt.ItemDataRole.DisplayRole])
            return
        row = len(self.history.entries)
        self.beginInsertRows(QModelIndex(), row, row)
        _row, _new, dropped = self.history.upsert(utterance)
        self.endInsertRows()
        if dropped:
            # upsert has already removed them; tell the views after the fact.
            self.beginResetModel()
            self.endResetModel()

    def revise(self, utterance) -> None:
        """Apply a second-look correction to a line already shown as final."""
        row = self.history.revise(utterance)
        if row is None:
            return
        index = self.index(row)
        self.dataChanged.emit(
            index,
            index,
            [ENTRY_ROLE, Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
        )

    def clear(self) -> None:
        self.beginResetModel()
        self.history.clear()
        self.endResetModel()


class SearchProxy(QSortFilterProxyModel):
    """Filters the transcript to lines containing every word of the query."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._query = ""

    def set_query(self, query: str) -> None:
        self._query = query.strip()
        self.invalidateFilter()

    @property
    def query(self) -> str:
        return self._query

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return True
        entry = self.sourceModel().data(self.sourceModel().index(row, 0, parent), ENTRY_ROLE)
        return entry is not None and History.matches(entry, self._query)


@dataclass
class Look:
    """Everything that decides how a row is drawn."""

    palette: Palette
    english_pt: float = 15.0
    show_source: bool = True
    show_timestamps: bool = True

    def key(self) -> tuple:
        return (self.palette.name, self.english_pt, self.show_source, self.show_timestamps)


class TranscriptDelegate(QStyledItemDelegate):
    def __init__(self, look: Look, parent=None):
        super().__init__(parent)
        self.look = look
        self._heights: dict[tuple, float] = {}
        self._refonts()

    def set_look(self, look: Look) -> None:
        self.look = look
        self._heights.clear()
        self._refonts()

    def _refonts(self) -> None:
        pt = self.look.english_pt
        self.english_font = display_font(pt, QFont.Weight.Medium)
        self.draft_font = display_font(pt, QFont.Weight.Normal)
        self.draft_font.setItalic(True)
        self.source_font = ui_font(max(8.0, pt * 0.8))
        self.meta_font = ui_font(max(7.5, pt * 0.62))
        self.tag_font = ui_font(max(7.5, pt * 0.64), QFont.Weight.DemiBold)
        self._gutter = (
            QFontMetricsF(self.meta_font).horizontalAdvance("00:00:00") + 16
            if self.look.show_timestamps
            else 0.0
        )

    # -- measuring ---------------------------------------------------------

    @staticmethod
    def _layout(text: str, font: QFont, width: float) -> tuple[QTextLayout, float]:
        layout = QTextLayout(text, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        height = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(40.0, width))
            line.setPosition(QPointF(0.0, height))
            height += line.height()
        layout.endLayout()
        return layout, height

    def _text_width(self, row_width: int) -> float:
        return row_width - 2 * PAD_X - self._gutter - COPY_BOX

    def _parts(self, entry: Entry) -> tuple[str, str]:
        """(primary line, secondary line). The secondary may be empty."""
        if entry.english:
            return entry.english, entry.source if self.look.show_source else ""
        return entry.source, ""

    def _height(self, entry: Entry, row_width: int) -> float:
        key = (entry.id, entry.english, entry.source, entry.final, entry.speaker, row_width)
        cached = self._heights.get(key)
        if cached is not None:
            return cached
        width = self._text_width(row_width)
        primary, secondary = self._parts(entry)
        height = 2.0 * PAD_Y
        if entry.speaker != UNKNOWN:
            height += QFontMetricsF(self.tag_font).height() + 1
        font = self.english_font if entry.final else self.draft_font
        if not entry.english:
            font = self.source_font
        height += self._layout(primary, font, width)[1]
        if secondary:
            height += GAP + self._layout(secondary, self.source_font, width)[1]
        if len(self._heights) > 20000:
            self._heights.clear()
        self._heights[key] = height
        return height

    def sizeHint(self, option, index: QModelIndex) -> QSize:  # noqa: N802
        entry = index.data(ENTRY_ROLE)
        view = self.parent()
        width = view.viewport().width() if view is not None else option.rect.width()
        if entry is None or width <= 0:
            return QSize(max(width, 100), 40)
        return QSize(width, int(self._height(entry, width) + 0.999))

    def copy_rect(self, row_rect: QRect) -> QRect:
        return QRect(
            row_rect.right() - PAD_X - COPY_BOX + 8, row_rect.top() + PAD_Y - 2, 22, 22
        )

    # -- painting ----------------------------------------------------------

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        entry: Entry | None = index.data(ENTRY_ROLE)
        if entry is None:
            return
        p = self.look.palette
        rect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(p.color("raised"))
            painter.drawRoundedRect(QRectF(rect.adjusted(4, 1, -4, -1)), 7, 7)
            painter.drawPixmap(
                self.copy_rect(rect).topLeft() + QPointF(2, 2).toPoint(),
                icons.pixmap("copy", p.text_3, 18, painter.device().devicePixelRatioF()),
            )

        x = rect.left() + PAD_X
        y = float(rect.top() + PAD_Y)
        width = self._text_width(rect.width())

        if self.look.show_timestamps and entry.clock:
            painter.setFont(self.meta_font)
            painter.setPen(p.color("text_3"))
            ascent = QFontMetricsF(
                self.english_font if entry.english else self.source_font
            ).ascent()
            tag_height = (
                QFontMetricsF(self.tag_font).height() + 1 if entry.speaker != UNKNOWN else 0
            )
            # On the baseline of the first line of text, so the two read as
            # one row whatever their sizes.
            painter.drawText(QPointF(x, y + tag_height + ascent), entry.clock)
        x += self._gutter

        if entry.speaker != UNKNOWN:
            painter.setFont(self.tag_font)
            painter.setPen(QColor(SPEAKER_COLORS[entry.speaker % len(SPEAKER_COLORS)]))
            metrics = QFontMetricsF(self.tag_font)
            painter.drawText(QPointF(x, y + metrics.ascent()), speaker_label(entry.speaker))
            y += metrics.height() + 1

        primary, secondary = self._parts(entry)
        if not entry.english:
            font, color = self.source_font, p.color("text")
        elif entry.final:
            font, color = self.english_font, p.color("text")
        else:
            font, color = self.draft_font, p.color("draft")
        layout, height = self._layout(primary, font, width)
        if entry.revised:
            # Level with the middle of the first line of text.
            first_line = QFontMetricsF(font).height()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(p.color("accent"))
            painter.drawEllipse(QPointF(x - 8.0, y + first_line / 2.0 + 0.5), 2.5, 2.5)
        painter.setPen(color)
        layout.draw(painter, QPointF(x, y))
        y += height

        if secondary:
            y += GAP
            layout, _h = self._layout(secondary, self.source_font, width)
            painter.setPen(p.color("text_2") if entry.final else p.color("text_3"))
            layout.draw(painter, QPointF(x, y))
        painter.restore()


class TranscriptView(QListView):
    """The scrolling list. Follows new lines unless the reader has scrolled up."""

    copied = pyqtSignal(str)

    def __init__(self, look: Look, parent: QWidget | None = None):
        super().__init__(parent)
        self._look = look
        self._delegate = TranscriptDelegate(look, self)
        self.setItemDelegate(self._delegate)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(22)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setUniformItemSizes(False)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMouseTracking(True)
        self.setFrameShape(QListView.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.empty_title = ""
        self.empty_text = ""
        self.follow = True
        self._was_at_bottom = True

        self._jump = QPushButton("Jump to latest", self)
        self._jump.setCursor(Qt.CursorShape.PointingHandCursor)
        self._jump.hide()
        self._jump.clicked.connect(self.jump_to_latest)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)

    # -- look --------------------------------------------------------------

    def set_look(self, look: Look) -> None:
        self._look = look
        self._delegate.set_look(look)
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    # -- following ---------------------------------------------------------

    def _at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 6

    def _scrolled(self, _value: int) -> None:
        self._was_at_bottom = self._at_bottom()
        self._jump.setVisible(self.follow and not self._was_at_bottom)

    def _range_changed(self, _low: int, high: int) -> None:
        # New text arrived. Stay at the end if the reader was already there;
        # leave them where they are if they had scrolled up to read.
        if self.follow and self._was_at_bottom:
            self.verticalScrollBar().setValue(high)

    def jump_to_latest(self) -> None:
        self._was_at_bottom = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self._jump.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        was = self._was_at_bottom
        super().resizeEvent(event)
        self._delegate._heights.clear()
        self.scheduleDelayedItemsLayout()
        self._jump.adjustSize()
        self._jump.move(
            (self.width() - self._jump.width()) // 2,
            self.height() - self._jump.height() - 14,
        )
        if self.follow and was:
            QTimer.singleShot(0, self.jump_to_latest)

    # -- interaction -------------------------------------------------------

    def _entry_at(self, pos) -> tuple[Entry | None, QRect]:
        index = self.indexAt(pos)
        if not index.isValid():
            return None, QRect()
        return index.data(ENTRY_ROLE), self.visualRect(index)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        entry, rect = self._entry_at(event.pos())
        over_copy = entry is not None and self._delegate.copy_rect(rect).contains(event.pos())
        self.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor if over_copy else Qt.CursorShape.ArrowCursor
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            entry, rect = self._entry_at(event.pos())
            if entry is not None and self._delegate.copy_rect(rect).contains(event.pos()):
                self._copy(entry.primary)
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        entry, _rect = self._entry_at(event.pos())
        if entry is not None:
            self._copy(entry.primary)

    def _copy(self, text: str) -> None:
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        self.copied.emit(text)

    def _menu(self, pos) -> None:
        entry, _rect = self._entry_at(pos)
        if entry is None:
            return
        menu = QMenu(self)
        if entry.english:
            menu.addAction("Copy English", lambda: self._copy(entry.english))
        if entry.source:
            menu.addAction("Copy Japanese", lambda: self._copy(entry.source))
        if entry.english and entry.source:
            menu.addAction(
                "Copy both",
                lambda: self._copy(History.format_entry(entry, source=True)),
            )
        if entry.revised and entry.first_english:
            menu.addSeparator()
            menu.addAction(
                "Copy the first version", lambda: self._copy(entry.first_english)
            )
        menu.exec(self.viewport().mapToGlobal(pos))

    # -- empty state -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        model = self.model()
        if model is not None and model.rowCount() == 0 and (self.empty_text or self.empty_title):
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            area = self.viewport().rect().adjusted(40, 0, -40, 0)
            flags = int(Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap)
            title_font = display_font(15.0, QFont.Weight.DemiBold)
            body_font = ui_font(10.5)
            title_h = (
                QFontMetricsF(title_font).height() + 10 if self.empty_title else 0.0
            )
            body_h = (
                QFontMetricsF(body_font)
                .boundingRect(QRectF(area), flags, self.empty_text)
                .height()
                if self.empty_text
                else 0.0
            )
            top = area.top() + (area.height() - title_h - body_h) / 2.0
            if self.empty_title:
                painter.setPen(self._look.palette.color("text_2"))
                painter.setFont(title_font)
                painter.drawText(
                    QRectF(area.left(), top, area.width(), title_h), flags, self.empty_title
                )
            if self.empty_text:
                painter.setPen(self._look.palette.color("text_3"))
                painter.setFont(body_font)
                painter.drawText(
                    QRectF(area.left(), top + title_h, area.width(), body_h + 4),
                    flags,
                    self.empty_text,
                )
            painter.end()
