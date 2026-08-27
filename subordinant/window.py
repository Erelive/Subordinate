"""The main application window: a readable, scrollable transcript.

This is the app's primary surface, not the overlay. The overlay draws on the
primary screen and cannot know better - system audio is one mixed stream with no
screen affinity, so nothing in the signal says which monitor the sound came from.
On a two-monitor setup that puts captions for a video on the second screen over
whatever is on the first, and against an exclusive-fullscreen game it cannot draw
at all. A window the user places solves both by putting the decision with the
only party that has the information.

It also keeps history. The overlay shows a rolling display_history_sec window and
discards the rest; here every committed utterance stays for the session, so the
original Japanese can be checked after the fact - which is the entire point when
the translation mangles a name.
"""

from __future__ import annotations

import html
import logging

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import Config

log = logging.getLogger(__name__)

# Opacity below this makes the window effectively invisible and unrecoverable
# without knowing where it is, so the slider stops short of it.
MIN_OPACITY = 25

# Translation engines, as config overrides. Each is a whole coherent setup
# rather than a single switch, because the choice implies a model and a Whisper
# task as well: Sugoi needs Whisper transcribing so it has source text to work
# from, while Whisper's own translation needs large-v3 - large-v3-turbo accepts
# task="translate" and returns the source language untouched.
ENGINES: dict[str, dict] = {
    "Sugoi v4 (recommended)": {
        "mt_enabled": True,
        "model": "large-v3-turbo",
        "task": "transcribe",
    },
    "Whisper large-v3": {
        "mt_enabled": False,
        "model": "large-v3",
        "task": "translate",
    },
    "No translation": {
        "mt_enabled": False,
        "model": "large-v3-turbo",
        "task": "transcribe",
    },
}

# Source language of the audio, not of the captions.
LANGUAGES = [
    ("Japanese", "ja"),
    ("English", "en"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
]

_SENTENCE_END = ("。", "！", "？", ".", "!", "?")


class TranscriptWindow(QMainWindow):
    """Scrollable session transcript, with the live line pinned underneath."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SubOrdinant")
        self.resize(760, 520)
        # True while committed fragments are still accumulating into one line.
        self._open_line = False
        self._applied: tuple[str, str] = ("", "")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addLayout(self._build_settings())
        layout.addLayout(self._build_toolbar())

        # The settled transcript. Read-only but selectable, so a name can be
        # copied out; that is a large part of why the history exists.
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont(cfg.font_family, cfg.font_size - 8))
        self._log.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self._log, stretch=1)

        # Whisper's unconfirmed tail - it rewrites itself, so it is kept out of
        # the transcript above and shown separately.
        self._live = QLabel()
        self._live.setWordWrap(True)
        self._live.setFont(QFont(cfg.font_family, cfg.font_size - 8))
        self._live.setStyleSheet(f"color: {cfg.unstable_color};")
        layout.addWidget(self._live)

        self._status = QLabel("starting...")
        self._status.setStyleSheet(f"color: {cfg.unstable_color};")
        layout.addWidget(self._status)

    # -- settings ----------------------------------------------------------

    def _build_settings(self) -> QHBoxLayout:
        """Audio language and translation engine, with an explicit Apply.

        Both changes reload models and so restart the pipeline, which takes a few
        seconds - too disruptive to fire on every keystroke of a combo box, hence
        Apply rather than applying live.
        """
        bar = QHBoxLayout()

        bar.addWidget(QLabel("Audio language"))
        self._language = QComboBox()
        for label, code in LANGUAGES:
            self._language.addItem(label, code)
        bar.addWidget(self._language)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Translate with"))
        self._engine = QComboBox()
        for name in ENGINES:
            self._engine.addItem(name)
        bar.addWidget(self._engine)

        self._apply = QPushButton("Apply")
        self._apply.setEnabled(False)
        bar.addWidget(self._apply)

        # Only offer Apply once something actually differs from what is running.
        self._language.currentIndexChanged.connect(self._settings_changed)
        self._engine.currentIndexChanged.connect(self._settings_changed)

        bar.addStretch(1)
        return bar

    def _settings_changed(self) -> None:
        self._apply.setEnabled(self.pending_settings() != self._applied)

    def pending_settings(self) -> tuple[str, str]:
        """(language code, engine name) currently selected in the controls."""
        return self._language.currentData(), self._engine.currentText()

    def show_applied(self) -> None:
        """Mark the current selection as the one that is running."""
        self._applied = self.pending_settings()
        self._apply.setEnabled(False)

    def load_settings(self, cfg: Config) -> None:
        """Point the controls at what cfg currently describes."""
        idx = self._language.findData(cfg.language)
        if idx >= 0:
            self._language.setCurrentIndex(idx)
        for name, over in ENGINES.items():
            if all(getattr(cfg, k) == v for k, v in over.items()):
                self._engine.setCurrentText(name)
                break
        self.show_applied()

    @property
    def apply_button(self) -> QPushButton:
        return self._apply

    # -- toolbar -----------------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        self._pin = QCheckBox("Pin on top")
        self._pin.toggled.connect(self._set_on_top)
        bar.addWidget(self._pin)

        self._pause = QCheckBox("Pause")
        bar.addWidget(self._pause)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Opacity"))
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(MIN_OPACITY, 100)
        self._opacity.setValue(100)
        self._opacity.setFixedWidth(120)
        self._opacity.valueChanged.connect(
            lambda v: self.setWindowOpacity(v / 100.0)
        )
        bar.addWidget(self._opacity)

        bar.addStretch(1)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)
        return bar

    def _set_on_top(self, on: bool) -> None:
        # Qt drops the native window when flags change, so it has to be shown
        # again - and re-showing resets opacity, which must be re-applied.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()
        self.setWindowOpacity(self._opacity.value() / 100.0)

    @property
    def pause_box(self) -> QCheckBox:
        return self._pause

    def apply_settings(self, opacity_pct: int, on_top: bool) -> None:
        """Apply saved window settings without going through the widgets twice."""
        self._opacity.setValue(max(MIN_OPACITY, min(100, opacity_pct)))
        self._pin.setChecked(on_top)

    def current_settings(self) -> tuple[int, bool]:
        """Opacity percent and pin state, for writing back to the config."""
        return self._opacity.value(), self._pin.isChecked()

    # -- content -----------------------------------------------------------

    def _at_bottom(self) -> bool:
        """Is the view scrolled to the end?

        Appending must not yank the view back down while the user is reading
        earlier text - being able to look back is the reason this window exists.
        """
        bar = self._log.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _append(self, content: str, plain: bool = False, new_block: bool = False) -> None:
        follow = self._at_bottom()
        cursor = QTextCursor(self._log.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if new_block and not self._log.document().isEmpty():
            # insertHtml merges a <div> into the block already at the cursor
            # rather than starting a new one, so the break has to be explicit.
            cursor.insertBlock()
        if plain:
            # Continuations go in as plain text, not HTML: HTML collapses a
            # leading space, which is what ran fragments together into
            # "is socute!". Inserting text also keeps the character format
            # already in effect, so the colour carries over.
            cursor.insertText(content)
        else:
            cursor.insertHtml(content)
        if follow:
            bar = self._log.verticalScrollBar()
            bar.setValue(bar.maximum())

    @pyqtSlot(str, str)
    def add_utterance(self, source: str, english: str) -> None:
        """One finished utterance: the source text and its translation."""
        source, english = source.strip(), english.strip()
        if not source and not english:
            return
        parts = []
        if english:
            parts.append(
                f'<div style="color:{self.cfg.committed_color}">'
                f"{html.escape(english)}</div>"
            )
        if source:
            parts.append(
                f'<div style="color:{self.cfg.unstable_color}">'
                f"{html.escape(source)}</div>"
            )
        self._append("".join(parts) + "<div>&nbsp;</div>")

    @pyqtSlot(str)
    def add_source_only(self, text: str) -> None:
        """Committed text with no translation - MT off, or it returned nothing.

        This arrives as fragments, a few words per pass, already stripped. They
        have to be re-joined with a space or the transcript reads
        "is socute!What?Is this aLeon?"; and a fragment ending a sentence starts
        the next block, so the log breaks into readable lines instead of growing
        as one unbroken paragraph.
        """
        text = text.strip()
        if not text:
            return
        if self._open_line:
            self._append(" " + text, plain=True)
        else:
            self._append(
                f'<span style="color:{self.cfg.committed_color}">'
                f"{html.escape(text)}</span>",
                new_block=True,
            )
        self._open_line = not text.endswith(_SENTENCE_END)

    @pyqtSlot(str, str)
    def set_live(self, committed: str, unstable: str) -> None:
        self._live.setText(f"{committed} {unstable}".strip())

    @pyqtSlot(str)
    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    def clear(self) -> None:
        self._log.clear()
        self._live.clear()
        self._open_line = False
