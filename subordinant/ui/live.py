"""The sentence in progress.

Pinned under the transcript, this card is the only place text is allowed to
change under the reader's eyes. It shows two things about the sentence being
spoken right now:

  - a provisional English line, in the draft colour. It can rewrite itself,
    sometimes into its own opposite, because a Japanese sentence does not
    commit to its meaning until its verb, which comes last.
  - the Japanese as the recogniser currently hears it: the part it has
    settled on, and after it, dimmer, the part it is still guessing at.

When the sentence ends, its final English appears in the transcript above and
this card empties. Nothing shown here is ever copied into the transcript as
is. The card keeps a fixed height, so text arriving and leaving does not make
the transcript above it jump.
"""

from __future__ import annotations

import html

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .theme import Palette, display_font, ui_font

_IDLE_HINTS = {
    "idle": "The sentence being spoken appears here as it is heard.",
    "loading": "Getting the models ready...",
    "listening": "Listening for Japanese speech...",
    "paused": "Paused. Nothing is being listened to.",
    "error": "",
}


class LiveCard(QFrame):
    def __init__(self, palette: Palette, english_pt: float = 15.0, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self._palette = palette
        self._english_pt = english_pt
        self._state = "idle"
        self._english = ""
        self._committed = ""
        self._unstable = ""
        self._show_source = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 14)
        layout.setSpacing(5)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._tag = QLabel("LIVE")
        self._tag.setObjectName("liveTag")
        head.addWidget(self._tag)
        self._note = QLabel("")
        self._note.setObjectName("hint")
        head.addWidget(self._note)
        head.addStretch(1)
        layout.addLayout(head)

        self._en = QLabel()
        self._en.setWordWrap(True)
        self._en.setTextFormat(Qt.TextFormat.PlainText)
        self._en.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._en)
        # A quick fade when the preview is replaced: the eye reads a change of
        # text as a flicker if it is instant, and as an update if it is not.
        self._fade = QGraphicsOpacityEffect(self._en)
        self._fade.setOpacity(1.0)
        self._en.setGraphicsEffect(self._fade)
        self._anim = QPropertyAnimation(self._fade, b"opacity", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._ja = QLabel()
        self._ja.setWordWrap(True)
        self._ja.setTextFormat(Qt.TextFormat.RichText)
        self._ja.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._ja)
        layout.addStretch(1)
        self._restyle()

    # -- appearance --------------------------------------------------------

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._restyle()

    def set_english_size(self, pt: float) -> None:
        self._english_pt = pt
        self._restyle()

    def set_show_source(self, show: bool) -> None:
        self._show_source = show
        self._render()

    def _english_css(self, color: str, italic: bool = True) -> str:
        # Size and style go in the widget's own stylesheet. The application
        # stylesheet sets a font size on everything, and a stylesheet rule
        # outranks setFont, so a font set the ordinary way is silently
        # ignored.
        style = "italic" if italic else "normal"
        return (
            f"color: {color}; font-size: {self._english_pt:.1f}pt; "
            f"font-style: {style}; font-weight: 400;"
        )

    def _restyle(self) -> None:
        font = display_font(self._english_pt, QFont.Weight.Normal)
        font.setItalic(True)
        self._en.setFont(font)
        source_pt = max(8.0, self._english_pt * 0.8)
        source_font = ui_font(source_pt)
        self._ja.setFont(source_font)
        self._ja.setStyleSheet(f"font-size: {source_pt:.1f}pt;")
        # Room for two lines of the English and one of the Japanese, so the
        # common case never resizes the card.
        from PyQt6.QtGui import QFontMetricsF

        en_line = QFontMetricsF(font).lineSpacing()
        ja_line = QFontMetricsF(source_font).lineSpacing()
        self.setMinimumHeight(int(en_line * 2 + ja_line * 1 + 62))
        self._render()

    # -- content -----------------------------------------------------------

    def set_state(self, state: str) -> None:
        self._state = state
        # Lit while someone is being heard, so "nothing is happening" and
        # "nothing is being said" can be told apart without reading.
        active = "true" if state in ("recognizing", "translating") else "false"
        if self._tag.property("active") != active:
            self._tag.setProperty("active", active)
            self._tag.style().unpolish(self._tag)
            self._tag.style().polish(self._tag)
        self._render()

    def set_preview(self, english: str) -> None:
        english = english.strip()
        if english == self._english:
            return
        had = bool(self._english)
        self._english = english
        self._render()
        if english and had:
            self._anim.stop()
            self._anim.setStartValue(0.45)
            self._anim.setEndValue(1.0)
            self._anim.start()

    def set_caption(self, committed: str, unstable: str) -> None:
        committed, unstable = committed.strip(), unstable.strip()
        if (committed, unstable) == (self._committed, self._unstable):
            return
        self._committed, self._unstable = committed, unstable
        self._render()

    def clear(self) -> None:
        self._english = self._committed = self._unstable = ""
        self._render()

    def _render(self) -> None:
        p = self._palette
        has_text = bool(self._english or self._committed or self._unstable)
        if not has_text:
            self._en.setText(_IDLE_HINTS.get(self._state, ""))
            self._en.setStyleSheet(self._english_css(p.text_3, italic=False))
            self._ja.setText("")
            self._note.setText("")
            return
        self._en.setStyleSheet(self._english_css(p.draft))
        self._en.setText(self._english)
        self._note.setText("draft - may change" if self._english else "")
        if self._show_source:
            self._ja.setText(
                f'<span style="color:{p.text_2}">{html.escape(self._committed)}</span>'
                f'<span style="color:{p.text_3}">{html.escape(self._unstable)}</span>'
            )
        else:
            self._ja.setText("")
