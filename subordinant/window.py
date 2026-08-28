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

from .config import TRANSCRIBERS, Config
from .diarize import UNKNOWN, speaker_label

log = logging.getLogger(__name__)

# Opacity below this makes the window effectively invisible and unrecoverable
# without knowing where it is, so the slider stops short of it.
MIN_OPACITY = 25

# Translation engines, as config overrides. Each is a whole coherent setup
# rather than a single switch, because the choice implies a Whisper task as
# well: Sugoi needs Whisper transcribing so it has source text to work from,
# while Whisper's own translation needs task="translate".
#
# Which model does the transcribing is a separate choice, in config.TRANSCRIBERS
# - it used to live here, which conflated two independent decisions and left no
# way to pick a better Japanese model without also changing how translation
# worked. The one place they are not independent is below.
#
# Which keys say *which* engine this is, as opposed to how it is tuned. Applying
# an engine sets every key in its entry; recognising one only compares these.
# Otherwise a config saved before a tuning value changed stops matching its own
# engine, and the picker silently shows the wrong one while something else runs.
#
# mt_max_utterance_sec is here rather than treated as tuning because the Sugoi
# entries below differ by nothing else. Leave it out and they all match on
# (mt_enabled, task), so load_settings picks whichever comes first and the
# picker shows 2.0 s while 8.0 s is running.
ENGINE_IDENTITY = ("mt_enabled", "task", "mt_max_utterance_sec")

# How long speech with no sentence ending in it is held before translating it
# anyway. It is offered here, spelled out, because it is the single control that
# decides whether spontaneous speech reads as sentences or as debris, and the
# right value depends on the source rather than on the machine.
#
# Whisper only emits 。 and 、 when it hears a clear sentence-final fall, which
# scripted material has and live talk does not. With no punctuation to cut on,
# _flush_translation falls through to this backstop and cuts wherever this many
# seconds of audio happened to land - which on fast speech is mid-word. Measured
# on a real caption run, the same clause cut two ways:
#
#     ...見えてしま | います...   -> "...I could see what was inside." /
#                                   "He's here. Do you want to comment?"
#     ...見えてしまいますよ       -> "...you can see what's inside." /
#                                   "Guess I'll make that comment."
#
# The second half of a severed verb translates as a whole sentence, fluently and
# entirely invented, because the translator never sees the words either side of
# the cut. Longer holds mean fewer cuts and so fewer of these, paid for in
# latency - which the live preview (mt_live_enabled) largely hides, since the
# unfinished tail keeps moving while the settled text waits.
ENGINES: dict[str, dict] = {
    "Sugoi v4 - utterance 2 s": {
        "mt_enabled": True,
        "task": "transcribe",
        "mt_max_utterance_sec": 2.0,
    },
    "Sugoi v4 - utterance 3 s": {
        "mt_enabled": True,
        "task": "transcribe",
        "mt_max_utterance_sec": 3.0,
    },
    "Sugoi v4 - utterance 5 s": {
        "mt_enabled": True,
        "task": "transcribe",
        "mt_max_utterance_sec": 5.0,
    },
    "Sugoi v4 - utterance 8 s": {
        "mt_enabled": True,
        "task": "transcribe",
        "mt_max_utterance_sec": 8.0,
    },
        "Sugoi v4 (utterance 10.0 s - most context)": {
        "mt_enabled": True,
        "task": "transcribe",
        "mt_max_utterance_sec": 10.0,
    },
    "Whisper large-v3": {
        "mt_enabled": False,
        "task": "translate",
    },
    "No translation": {
        "mt_enabled": False,
        "task": "transcribe",
    },
}

_ENGINE_TIP = (
    "What turns the Japanese into English.\n"
    "Sugoi is a dedicated translator and the recommended route.\n"
    "\n"
    "The utterance figure is how long speech with no sentence ending is\n"
    "held before translating it anyway. Scripted material - anime, a read\n"
    "script - carries punctuation and rarely reaches it, so any value\n"
    "behaves the same. Live conversation carries none, so this decides\n"
    "where lines get cut: short holds cut mid-word and the translator\n"
    "invents a sentence around the fragment, long holds read better and\n"
    "arrive later. Try 5 s on a stream, 2-3 s on anime."
)

# Engines where one model does both jobs, so the engine dictates the transcriber
# and the picker for it is locked. large-v3-turbo accepts task="translate" and
# returns the source language untouched, and the distilled models cannot
# translate at all, so this route specifically needs full large-v3.
ENGINE_TRANSCRIBER = {"Whisper large-v3": "Whisper large-v3"}

_TRANSCRIBER_TIP = (
    "Which model turns audio into text.\n"
    "large-v3-turbo handles any language and is the safe default.\n"
    "Kotoba is Japanese-only, and both faster and more accurate on it.\n"
    "large-v3 is the strongest, but costs roughly twice the time per pass."
)


# Whisper's beam width, offered so it can be compared on real audio. 0 means
# "whatever the model above asks for" - the per-model defaults in TRANSCRIBERS,
# which differ because the models can afford different amounts of it.
#
# Worth knowing before reaching for it: on clean single-speaker audio this
# changes almost nothing. Measured over 300 CommonVoice utterances, Kotoba
# scored 89.84 / 89.77 / 89.79 percent at beam 5 / 10 / 20 - noise. What beam
# search actually buys is the *overlapping* speech case, where greedy decoding
# lost 39 percent of the words and beam 5 lost 15 percent. So the reason to
# raise it is two people talking over each other, not a clearer transcript of
# one person.
BEAMS: list[tuple[str, int]] = [
    ("Auto", 0),
    ("1", 1),
    ("3", 3),
    ("5", 5),
    ("10", 10),
    ("20", 20),
]

_BEAM_TIP = (
    "How many decodings Whisper keeps in flight before picking one.\n"
    "Auto uses each model's own default: 5 for the turbo models, 1 for\n"
    "large-v3, which cannot afford more.\n"
    "\n"
    "It costs time per pass, and the models differ sharply. Measured at\n"
    "a full 12 s buffer, as RTF against the 0.35 s interval - over 1.0\n"
    "means the backlog stops draining and audio is eventually discarded:\n"
    "\n"
    "   Kotoba     beam 1 0.40   beam 5 0.53   beam 10 1.20   beam 20 3.34\n"
    "   turbo      beam 1 0.42   beam 5 0.62\n"
    "   large-v3   beam 1 1.21   beam 3 1.18   beam 5 1.46\n"
    "\n"
    "On one clear speaker it buys almost nothing. It earns its cost on\n"
    "people talking over each other, where beam 1 loses 39% of the words\n"
    "and beam 5 loses 15%."
)

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


def _cap_width(combo: QComboBox, chars: int) -> None:
    """Stop a combo asking for room for its longest entry.

    A combo sizes itself to its widest item, and setMinimumWidth only raises
    that floor - it cannot lower it. Unchecked, the utterance variants took the
    bar's preferred width to 2470 px; with this it is 1906, slightly under what
    it was before they existed.

    This bounds the *preferred* width only. minimumSizeHint keeps accounting for
    the longest item whatever the size-adjust policy or maximum say, so the
    window's hard minimum still grew - 1410 px to 1614 px when the Beam control
    went in. If that ever has to come down, the fix is splitting this bar over
    two rows, not tuning the numbers here. Text past the cap elides, and the
    full label is still readable in the popup.
    """
    width = combo.fontMetrics().horizontalAdvance("W" * chars) + 34
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(chars)
    combo.setMaximumWidth(width)


def _engine_for(cfg: Config) -> str:
    """Which engine entry describes cfg. Never returns nothing.

    Exact match first. Failing that, mt_max_utterance_sec is snapped to the
    nearest entry that agrees on everything else, because a config carrying a
    hold this list does not offer - hand-edited, or saved before the list
    changed - would otherwise match nothing and leave the picker showing its
    first item while something else entirely was running. That is the failure
    ENGINE_IDENTITY exists to prevent, and adding a tuning value to it reopens
    it unless the miss is caught here.
    """
    others = [k for k in ENGINE_IDENTITY if k != "mt_max_utterance_sec"]
    for name, over in ENGINES.items():
        if all(getattr(cfg, k) == over[k] for k in ENGINE_IDENTITY if k in over):
            return name
    near = [
        (abs(over["mt_max_utterance_sec"] - cfg.mt_max_utterance_sec), name)
        for name, over in ENGINES.items()
        if "mt_max_utterance_sec" in over
        and all(getattr(cfg, k) == over[k] for k in others if k in over)
    ]
    if near:
        return min(near)[1]
    return next(iter(ENGINES))


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
        bar.addWidget(QLabel("Transcribe with"))
        self._transcriber = QComboBox()
        for name in TRANSCRIBERS:
            self._transcriber.addItem(name)
        _cap_width(self._transcriber, 18)
        self._transcriber.setToolTip(_TRANSCRIBER_TIP)
        bar.addWidget(self._transcriber)

        # Sits with the transcriber because it tunes that model rather than
        # being a choice of its own, and its cost depends on which one is up.
        beam_label = QLabel("Beam")
        beam_label.setToolTip(_BEAM_TIP)
        bar.addWidget(beam_label)
        self._beam = QComboBox()
        for label, value in BEAMS:
            self._beam.addItem(label, value)
        _cap_width(self._beam, 4)
        self._beam.setToolTip(_BEAM_TIP)
        bar.addWidget(self._beam)

        bar.addSpacing(10)
        bar.addWidget(QLabel("Translate with"))
        self._engine = QComboBox()
        for name in ENGINES:
            self._engine.addItem(name)
        _cap_width(self._engine, 20)
        self._engine.setToolTip(_ENGINE_TIP)
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
        self._transcriber.currentIndexChanged.connect(self._settings_changed)
        self._beam.currentIndexChanged.connect(self._settings_changed)
        self._engine.currentIndexChanged.connect(self._settings_changed)
        self._diarize.toggled.connect(self._settings_changed)
        # Whisper's own translation is one model doing both jobs, so picking it
        # settles the transcriber too - shown rather than silently overridden.
        self._engine.currentIndexChanged.connect(self._sync_transcriber)
        self._sync_transcriber()

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

    def _sync_transcriber(self) -> None:
        """Lock the transcriber to the engine where the engine implies one."""
        forced = ENGINE_TRANSCRIBER.get(self._engine.currentText())
        if forced:
            self._transcriber.setCurrentText(forced)
        self._transcriber.setEnabled(forced is None)
        self._transcriber.setToolTip(
            f"{self._engine.currentText()} translates with the "
            "transcription model itself, so it sets this one too."
            if forced
            else _TRANSCRIBER_TIP
        )

    def _settings_changed(self) -> None:
        self._apply.setEnabled(self.pending_settings() != self._applied)

    def pending_settings(self) -> tuple[str, str, int, str, bool, int, str]:
        """(language, transcriber, beam, engine, labelling, speaker cap, glossary).

        beam is 0 for "whatever the transcriber asks for".
        """
        return (
            self._language.currentData(),
            self._transcriber.currentText(),
            self._beam.currentData(),
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
        self._engine.setCurrentText(_engine_for(cfg))
        for name, over in TRANSCRIBERS.items():
            if cfg.model == over["model"]:
                self._transcriber.setCurrentText(name)
                break
        # Show Auto rather than the number whenever the running beam is simply
        # what this model asks for, so the control reads as "not overridden".
        default = TRANSCRIBERS.get(self._transcriber.currentText(), {}).get("beam_size")
        idx = self._beam.findData(0 if cfg.beam_size == default else cfg.beam_size)
        self._beam.setCurrentIndex(idx if idx >= 0 else 0)
        self._sync_transcriber()
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

        # Also outside Apply: switching the overlay on or off only connects or
        # disconnects an already-built widget, so it costs nothing to reload.
        self._overlay = QCheckBox("Screen overlay")
        self._overlay.setToolTip(
            "Draw captions directly on the screen as well as here.\n"
            "Click-through, and always on the primary monitor - system audio\n"
            "carries nothing that says which screen it came from. It cannot\n"
            "draw over an exclusive-fullscreen game; borderless windowed works."
        )
        bar.addWidget(self._overlay)

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

    @property
    def overlay_box(self) -> QCheckBox:
        return self._overlay

    def apply_settings(
        self,
        opacity_pct: int,
        on_top: bool,
        live_preview: bool = True,
        overlay: bool = False,
    ) -> None:
        """Apply saved window settings without going through the widgets twice."""
        self._opacity.setValue(max(MIN_OPACITY, min(100, opacity_pct)))
        self._pin.setChecked(on_top)
        self._live_preview.setChecked(live_preview)
        self._overlay.setChecked(overlay)

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
