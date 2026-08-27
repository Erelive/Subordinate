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

import logging

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import (
    QColor,
    QFont,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import Config
from .diarize import UNKNOWN, speaker_label

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

# Source language of the audio, not of the captions. Kept to the four the
# translation models are actually set up for - Whisper handles far more, but
# offering them would imply a working translation path that does not exist.
LANGUAGES = [
    ("Japanese", "ja"),
    ("English", "en"),
    ("Chinese", "zh"),
    ("Korean", "ko"),
]

_SENTENCE_END = ("。", "！", "？", ".", "!", "?")

# One palette for both the widget stylesheet and the transcript HTML, so the two
# cannot drift apart. Dark throughout on purpose: the default Qt chrome is light
# and sat badly against a dark transcript, and this window is meant to be usable
# at low opacity over video, where a light background glares.
BG = "#15171b"  # window
SURFACE = "#1c1f25"  # transcript and inputs
SURFACE_HI = "#242830"  # hover
BORDER = "#2c313a"
ACCENT = "#5b8def"
TEXT = "#eceef2"  # translation - what the reader is here for
TEXT_SOURCE = "#a8b0bd"  # source text: subordinate to the translation, but the
# whole point is that it stays readable, so it sits well above the label grey
TEXT_DIM = "#8a919e"  # labels
TEXT_FAINT = "#5d646f"  # status, and the unconfirmed tail

# One hue per speaker, in the order they are first heard. Distinguishing them at
# a glance is the whole point of the label, so these are chosen to stay apart
# under the window's low-opacity mode rather than to be pretty. The list is as
# long as Config.diarize_max_speakers; beyond that it wraps.
SPEAKER_COLORS = (
    "#5b8def",  # blue
    "#e0a14b",  # amber
    "#5fbf87",  # green
    "#c678dd",  # violet
    "#e06c75",  # red
    "#56b6c2",  # cyan
)

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QLabel {{ background: transparent; color: {TEXT_DIM}; }}
QLabel#live {{ color: {TEXT_FAINT}; padding: 2px 4px; }}
QLabel#liveEnglish {{ color: {TEXT_SOURCE}; padding: 2px 4px; }}
QLabel#status {{ color: {TEXT_FAINT}; font-size: 11px; padding: 0 4px; }}

QTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 12px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}

QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:hover {{ border-color: #39404b; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit[text=""] {{ color: {TEXT_FAINT}; }}

QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-width: 118px;
    color: {TEXT};
}}
QComboBox:hover {{ background: {SURFACE_HI}; border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {SURFACE_HI}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {SURFACE}; }}
QPushButton#apply:enabled {{
    background: {ACCENT}; border-color: {ACCENT}; color: #ffffff;
}}
QPushButton#apply:enabled:hover {{ background: #6d9bf5; }}

QCheckBox {{ spacing: 7px; color: {TEXT_DIM}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{
    height: 4px; background: {BORDER}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 13px; height: 13px; margin: -5px 0;
    background: {TEXT}; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #3a404b; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QFrame#divider {{ background: {BORDER}; max-height: 1px; border: none; }}
"""


class TranscriptWindow(QMainWindow):
    """Scrollable session transcript, with the live line pinned underneath."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SubOrdinant")
        self.resize(760, 520)
        # True while committed fragments are still accumulating into one line.
        self._open_line = False
        # Who that open line is tagged with, so a change of voice can break it.
        self._open_speaker = UNKNOWN
        self._applied: tuple[str, str, bool, int, str] = ("", "", False, 0, "")

        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(10)

        layout.addLayout(self._build_settings())
        layout.addWidget(self._divider())
        layout.addLayout(self._build_toolbar())

        # The settled transcript. Read-only but selectable, so a name can be
        # copied out; that is a large part of why the history exists.
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont(cfg.font_family, cfg.font_size - 9))
        self._log.setFrameShape(QFrame.Shape.NoFrame)  # the stylesheet draws it
        self._log.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self._log, stretch=1)

        # The in-flight English, above the source it came from: it is the line
        # the reader is actually watching, and putting it second would make them
        # track a moving target down the window. Brighter than the source tail
        # but still short of transcript white, because it rewrites itself.
        self._live_en = QLabel()
        self._live_en.setObjectName("liveEnglish")
        self._live_en.setWordWrap(True)
        self._live_en.setFont(QFont(cfg.font_family, cfg.font_size - 9))
        self._live_en.setMinimumHeight(22)
        self._live_en.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self._live_en)

        # Whisper's unconfirmed tail - it rewrites itself, so it is kept out of
        # the transcript above and shown separately, dimmer than settled text.
        self._live = QLabel()
        self._live.setObjectName("live")
        self._live.setWordWrap(True)
        self._live.setFont(QFont(cfg.font_family, cfg.font_size - 10))
        # Two lines' worth, so arriving text cannot make the window jump.
        self._live.setMinimumHeight(34)
        self._live.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self._live)

        self._status = QLabel("starting...")
        self._status.setObjectName("status")
        layout.addWidget(self._status)

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        return line

    # -- settings ----------------------------------------------------------

    def _build_settings(self) -> QVBoxLayout:
        """Audio language and translation engine, with an explicit Apply.

        Both changes reload models and so restart the pipeline, which takes a few
        seconds - too disruptive to fire on every keystroke of a combo box, hence
        Apply rather than applying live.
        """
        rows = QVBoxLayout()
        rows.setSpacing(7)
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Audio"))
        self._language = QComboBox()
        for label, code in LANGUAGES:
            self._language.addItem(label, code)
        bar.addWidget(self._language)

        bar.addSpacing(10)
        bar.addWidget(QLabel("Translate with"))
        self._engine = QComboBox()
        for name in ENGINES:
            self._engine.addItem(name)
        self._engine.setMinimumWidth(180)
        bar.addWidget(self._engine)

        bar.addSpacing(10)
        # Sits with the model settings rather than the toolbar because it loads
        # a model of its own, so changing it goes through Apply like the rest.
        self._diarize = QCheckBox("Label speakers")
        self._diarize.setToolTip(
            "Tag each line with who said it, and split lines when the voice "
            "changes.\nNumbers are assigned in the order people are first "
            "heard, and\ncan shift if someone is quiet for a long stretch."
        )
        bar.addWidget(self._diarize)

        # Capping this is the most reliable defence against one person becoming
        # several - an excited or put-on voice embeds quite differently from the
        # same person talking normally, and no threshold separates that cleanly.
        # The viewer usually knows the answer, so let them say it.
        self._speaker_count = QComboBox()
        self._speaker_count.addItem("Auto", 6)
        for n in range(2, 9):
            self._speaker_count.addItem(f"{n} people", n)
        self._speaker_count.setToolTip(
            "How many people are talking. Setting it stops one voice being "
            "split\ninto several; leave on Auto if you do not know."
        )
        self._speaker_count.setMinimumWidth(92)
        self._speaker_count.setEnabled(False)
        bar.addWidget(self._speaker_count)

        # Meaningless while labelling is off, so it follows the checkbox.
        self._diarize.toggled.connect(self._speaker_count.setEnabled)
        self._speaker_count.currentIndexChanged.connect(self._settings_changed)

        self._apply = QPushButton("Apply")
        self._apply.setObjectName("apply")  # the stylesheet accents it when live
        self._apply.setEnabled(False)
        bar.addWidget(self._apply)

        # Only offer Apply once something actually differs from what is running.
        self._language.currentIndexChanged.connect(self._settings_changed)
        self._engine.currentIndexChanged.connect(self._settings_changed)
        self._diarize.toggled.connect(self._settings_changed)

        bar.addStretch(1)
        rows.addLayout(bar)

        # Names Whisper has never heard come out as whatever they sound like -
        # Noel as "Noelle", Subaru as "Sabaru" - and the translator then renders
        # the wrong word faithfully, so nothing downstream can tell. Listing them
        # here biases the decoder before it guesses. Measured on English TTS:
        # 1/2 names correct without, 2/2 with.
        names = QHBoxLayout()
        names.setSpacing(8)
        names.addWidget(QLabel("Names"))
        self._prompt = QLineEdit()
        self._prompt.setPlaceholderText(
            "Names and terms to expect, comma separated - e.g. ラミィ, ノエル, スバル, ホロライブ"
        )
        self._prompt.setToolTip(
            "Words the stream will use that Whisper would otherwise mishear.\n"
            "Keep it short: this shares the decoder's context with the audio,\n"
            "and a long list makes Whisper recite it instead of transcribing."
        )
        self._prompt.setClearButtonEnabled(True)
        self._prompt.textChanged.connect(self._settings_changed)
        # Enter applies, since typing here has no other terminator.
        self._prompt.returnPressed.connect(
            lambda: self._apply.isEnabled() and self._apply.click()
        )
        names.addWidget(self._prompt, stretch=1)
        rows.addLayout(names)
        return rows

    def _settings_changed(self) -> None:
        self._apply.setEnabled(self.pending_settings() != self._applied)

    def pending_settings(self) -> tuple[str, str, bool, int, str]:
        """(language, engine, labelling, speaker cap, name glossary)."""
        return (
            self._language.currentData(),
            self._engine.currentText(),
            self._diarize.isChecked(),
            self._speaker_count.currentData(),
            self._prompt.text().strip(),
        )

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
        self._prompt.setText(cfg.initial_prompt)
        self._diarize.setChecked(cfg.diarize_enabled)
        self._speaker_count.setEnabled(cfg.diarize_enabled)
        idx = self._speaker_count.findData(cfg.diarize_max_speakers)
        self._speaker_count.setCurrentIndex(idx if idx >= 0 else 0)
        self.show_applied()

    @property
    def apply_button(self) -> QPushButton:
        return self._apply

    # -- toolbar -----------------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self._pin = QCheckBox("Pin on top")
        self._pin.toggled.connect(self._set_on_top)
        bar.addWidget(self._pin)

        self._pause = QCheckBox("Pause")
        bar.addWidget(self._pause)

        # Not in the settings row, and so not behind Apply: this changes nothing
        # about which models are loaded, only whether the tail gets previewed, so
        # it can take effect on the next pass.
        self._live_preview = QCheckBox("Live preview")
        self._live_preview.setToolTip(
            "Translate speech before the sentence ends, above the Japanese.\n"
            "It updates constantly and will rewrite itself - Japanese puts the\n"
            "verb and its negation last, so a half-sentence can mean the\n"
            "opposite. The transcript below is unaffected."
        )
        bar.addWidget(self._live_preview)

        bar.addSpacing(8)
        bar.addWidget(QLabel("Opacity"))
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(MIN_OPACITY, 100)
        self._opacity.setValue(100)
        self._opacity.setFixedWidth(110)
        self._opacity.valueChanged.connect(self._on_opacity)
        bar.addWidget(self._opacity)

        # The percentage matters when the window is translucent over video and
        # the slider itself has gone hard to see.
        self._opacity_value = QLabel("100%")
        self._opacity_value.setFixedWidth(38)
        bar.addWidget(self._opacity_value)

        bar.addStretch(1)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)
        return bar

    def _on_opacity(self, value: int) -> None:
        self.setWindowOpacity(value / 100.0)
        self._opacity_value.setText(f"{value}%")

    def _set_on_top(self, on: bool) -> None:
        # Qt drops the native window when flags change, so it has to be shown
        # again - and re-showing resets opacity, which must be re-applied.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()
        self.setWindowOpacity(self._opacity.value() / 100.0)

    @property
    def pause_box(self) -> QCheckBox:
        return self._pause

    @property
    def live_preview_box(self) -> QCheckBox:
        return self._live_preview

    def apply_settings(
        self, opacity_pct: int, on_top: bool, live_preview: bool = True
    ) -> None:
        """Apply saved window settings without going through the widgets twice."""
        self._opacity.setValue(max(MIN_OPACITY, min(100, opacity_pct)))
        self._pin.setChecked(on_top)
        self._live_preview.setChecked(live_preview)

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

    def _line(
        self,
        text: str,
        color: str,
        top_margin: int = 0,
        speaker: int = UNKNOWN,
    ) -> None:
        """Append one paragraph in the given colour, optionally speaker-tagged.

        Deliberately built from block and character formats rather than HTML.
        insertHtml nests each <div> inside the block already at the cursor, so
        consecutive entries indented further and further to the right; and it
        collapses a leading space, which ran committed fragments together into
        "is socute!". Formats have neither problem, and give real block margins,
        which Qt's rich text ignores between top-level blocks anyway.
        """
        follow = self._at_bottom()
        cursor = QTextCursor(self._log.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._log.document().isEmpty():
            cursor.insertBlock()

        block = QTextBlockFormat()
        block.setTopMargin(top_margin)
        block.setBottomMargin(0)
        block.setLeftMargin(0)
        block.setIndent(0)
        cursor.setBlockFormat(block)

        # The tag is a run of differently-formatted characters in the same block,
        # not a block of its own, so the line still wraps and selects as one unit.
        if speaker != UNKNOWN:
            tag = QTextCharFormat()
            tag.setForeground(QColor(SPEAKER_COLORS[speaker % len(SPEAKER_COLORS)]))
            tag.setFontWeight(QFont.Weight.DemiBold)
            cursor.setCharFormat(tag)
            cursor.insertText(f"{speaker_label(speaker)}  ")

        char = QTextCharFormat()
        char.setForeground(QColor(color))
        cursor.setCharFormat(char)
        cursor.insertText(text)

        if follow:
            bar = self._log.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _continue_line(self, text: str) -> None:
        """Extend the paragraph at the end, keeping its format."""
        follow = self._at_bottom()
        cursor = QTextCursor(self._log.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if follow:
            bar = self._log.verticalScrollBar()
            bar.setValue(bar.maximum())

    @pyqtSlot(str, str, int)
    def add_utterance(self, source: str, english: str, speaker: int = UNKNOWN) -> None:
        """One finished utterance: the source text and its translation."""
        source, english = source.strip(), english.strip()
        if not source and not english:
            return
        # The gap goes above the translation, so each pair reads as one unit
        # instead of the source drifting toward the next utterance's heading.
        # The tag goes on the first line of the pair only - repeating it on the
        # source line would say the same thing twice about one utterance.
        if english:
            # The translation leads on contrast, but the source is not a
            # footnote - being able to read it is the reason the history exists,
            # so it keeps the same size and only steps down in colour.
            self._line(english, TEXT, top_margin=10, speaker=speaker)
            if source:
                self._line(source, TEXT_SOURCE)
        elif source:
            self._line(source, TEXT_SOURCE, top_margin=10, speaker=speaker)
        self._open_line = False
        self._open_speaker = UNKNOWN

    @pyqtSlot(str, int)
    def add_source_only(self, text: str, speaker: int = UNKNOWN) -> None:
        """Committed text with no translation - MT off, or it returned nothing.

        This arrives as fragments, a few words per pass, already stripped. They
        have to be re-joined with a space or the transcript reads
        "is socute!What?Is this aLeon?"; and a fragment ending a sentence starts
        the next block, so the log breaks into readable lines instead of growing
        as one unbroken paragraph.

        A change of speaker breaks the line too. Without that, one person's
        fragments keep extending a line tagged with the previous speaker's name,
        which is worse than no tag at all.
        """
        text = text.strip()
        if not text:
            return
        changed = speaker != self._open_speaker and speaker != UNKNOWN
        if self._open_line and not changed:
            self._continue_line(" " + text)
        else:
            self._line(text, TEXT, top_margin=6, speaker=speaker)
            self._open_speaker = speaker
        self._open_line = not text.endswith(_SENTENCE_END)

    @pyqtSlot(str, str)
    def set_live(self, committed: str, unstable: str) -> None:
        self._live.setText(f"{committed} {unstable}".strip())

    @pyqtSlot(str)
    def set_live_translation(self, english: str) -> None:
        """English for speech still in progress. Rewrites itself by design.

        Kept out of the transcript on purpose: a partial Japanese sentence can
        translate to the opposite of the finished one, so this is a guess shown
        where guesses live, and the settled version follows underneath it.
        """
        self._live_en.setText(english.strip())

    @pyqtSlot(str)
    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    def clear(self) -> None:
        self._log.clear()
        self._live.clear()
        self._live_en.clear()
        self._open_line = False
        self._open_speaker = UNKNOWN
