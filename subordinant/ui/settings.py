"""The settings page.

Every control here is bound to one field of Config. Changing a control writes
that field and emits changed(field); the controller decides what follows -
most appearance settings apply on the spot, anything that changes which
models are loaded waits for the pipeline to be restarted, and everything is
saved. Nobody has to open config.json.

The page is organised the way the choices are made rather than the way the
code is: what to translate with, how hard to push the machine, what it looks
like, how the app behaves on the desktop.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import TRANSCRIBERS, Config
from ..profiles import (
    CUSTOM_LLM,
    PROFILE_BLURBS,
    PROFILE_LABELS,
    PROFILES,
    RECOGNIZER_NOTES,
    TRANSLATORS,
    apply_profile,
    detect_profile,
    recognizer_name,
    translator_name,
)
from .theme import Palette
from .widgets import Card, SegmentedControl, SettingRow, ToggleSwitch

# Settings that change which models are loaded or how the pipeline is wired.
# A running pipeline has its own copy of the config, so these take effect the
# next time it starts.
ENGINE_KEYS = frozenset(
    {
        "profile", "model", "device", "compute_type", "cpu_model", "language", "task",
        "beam_size", "no_repeat_ngram_size", "initial_prompt",
        "mt_enabled", "mt_backend", "mt_model", "mt_beam_size", "mt_max_utterance_sec",
        "mt_hybrid_preview", "mt_fallback_to_nmt", "mt_preview_min_confidence",
        "mt_preview_interval_sec",
        "llm_model", "llm_endpoint", "llm_context_utterances", "llm_context_reset_sec",
        "llm_keep_honorifics",
        "diarize_enabled", "diarize_max_speakers",
        "process_interval_sec", "min_chunk_sec", "max_buffer_sec", "commit_holdback_sec",
        "vad_mode", "vad_threshold", "endpoint_silence_sec", "eager_endpoint_sec",
        "soft_pause_sec",
        "long_pause_sec", "vram_saver",
    }
)


class _Slider(QWidget):
    """A slider over a float range with its value written beside it."""

    changed = pyqtSignal(float)

    def __init__(self, low: float, high: float, step: float, fmt: str, width: int = 170):
        super().__init__()
        self._low, self._step, self._fmt = low, step, fmt
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._value = QLabel()
        self._value.setObjectName("secondary")
        self._value.setMinimumWidth(52)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._value)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, int(round((high - low) / step)))
        self._slider.setFixedWidth(width)
        self._slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._slider.valueChanged.connect(self._moved)
        # The wheel scrolls the page, not whichever slider is under the cursor.
        self._slider.wheelEvent = lambda event: event.ignore()
        layout.addWidget(self._slider)

    def value(self) -> float:
        return self._low + self._slider.value() * self._step

    def set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(round((value - self._low) / self._step)))
        self._slider.blockSignals(False)
        self._value.setText(self._fmt.format(self.value()))

    def _moved(self, _tick: int) -> None:
        self._value.setText(self._fmt.format(self.value()))
        self.changed.emit(self.value())


class _Combo(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()  # see _Slider


class SettingsPage(QWidget):
    changed = pyqtSignal(str)  # a Config field was written
    download_requested = pyqtSignal(str)  # an Ollama tag
    open_logs_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, cfg: Config, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("page")
        self.cfg = cfg
        self._palette = palette
        self._loading = False
        # (load, widget) pairs: load() reads cfg into the widget.
        self._loaders: list = []
        self._switches: list[ToggleSwitch] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- section list --
        side = QVBoxLayout()
        side.setContentsMargins(18, 18, 6, 18)
        side.setSpacing(3)
        self._side_group = QButtonGroup(self)
        self._side_group.setExclusive(True)
        self._sections: list[tuple[QPushButton, QWidget]] = []
        self._side = side
        side_holder = QWidget()
        side_holder.setLayout(side)
        side_holder.setFixedWidth(168)
        root.addWidget(side_holder)

        # -- scrolling body --
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("page")
        self._body = QVBoxLayout(body)
        self._body.setContentsMargins(12, 18, 24, 28)
        self._body.setSpacing(12)
        self._scroll.setWidget(body)
        root.addWidget(self._scroll, stretch=1)

        self._build_translation()
        self._build_performance()
        self._build_appearance()
        self._build_general()
        self._body.addStretch(1)
        side.addStretch(1)
        self._scroll.verticalScrollBar().valueChanged.connect(self._sync_side)
        self.refresh()

    # -- scaffolding -------------------------------------------------------

    def _section(self, title: str, blurb: str) -> None:
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        if self._sections:
            self._body.addSpacing(14)
        self._body.addWidget(heading)
        sub = QLabel(blurb)
        sub.setObjectName("rowHelp")
        sub.setWordWrap(True)
        self._body.addWidget(sub)
        button = QPushButton(title)
        button.setObjectName("side")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumWidth(140)
        button.clicked.connect(lambda _c, w=heading: self._goto(w))
        self._side_group.addButton(button)
        self._side.addWidget(button)
        self._sections.append((button, heading))
        if len(self._sections) == 1:
            button.setChecked(True)

    def _card(self, title: str = "") -> Card:
        card = Card(title)
        self._body.addWidget(card)
        return card

    def _goto(self, widget: QWidget) -> None:
        self._scroll.verticalScrollBar().setValue(max(0, widget.y() - 14))

    def _sync_side(self, value: int) -> None:
        current = self._sections[0][0]
        for button, heading in self._sections:
            if heading.y() - 40 <= value:
                current = button
        current.setChecked(True)

    def _set(self, key: str, value) -> None:
        if self._loading or getattr(self.cfg, key) == value:
            return
        setattr(self.cfg, key, value)
        if key in ENGINE_KEYS and key != "profile":
            # A hand-edited engine setting means the profile label may no
            # longer be true.
            self.cfg.profile = detect_profile(self.cfg)
            self._load_profile()
        self.changed.emit(key)

    def _toggle(self, card: Card, key: str, title: str, help: str = "") -> SettingRow:  # noqa: A002
        switch = ToggleSwitch(self._palette)
        switch.toggled.connect(lambda on, k=key: self._set(k, bool(on)))
        self._switches.append(switch)
        self._loaders.append(lambda s=switch, k=key: s.setChecked(bool(getattr(self.cfg, k))))
        return card.add_row(SettingRow(title, help, switch))

    def _slider(
        self, card: Card, key: str, title: str, help: str,  # noqa: A002
        low: float, high: float, step: float, fmt: str, scale: float = 1.0, integer: bool = False,
    ) -> SettingRow:
        slider = _Slider(low, high, step, fmt)

        def write(value: float, k=key) -> None:
            raw = value / scale
            self._set(k, int(round(raw)) if integer else round(raw, 4))

        slider.changed.connect(write)
        self._loaders.append(lambda s=slider, k=key: s.set_value(getattr(self.cfg, k) * scale))
        return card.add_row(SettingRow(title, help, slider))

    def _combo(
        self, card: Card, key: str, title: str, help: str, options: list[tuple[str, object]],  # noqa: A002
        width: int = 230,
    ) -> tuple[SettingRow, _Combo]:
        combo = _Combo()
        combo.setFixedWidth(width)
        for label, value in options:
            combo.addItem(label, value)
        combo.currentIndexChanged.connect(lambda _i, c=combo, k=key: self._set(k, c.currentData()))

        def load(c=combo, k=key) -> None:
            index = c.findData(getattr(self.cfg, k))
            c.blockSignals(True)
            c.setCurrentIndex(index if index >= 0 else 0)
            c.blockSignals(False)

        self._loaders.append(load)
        return card.add_row(SettingRow(title, help, combo)), combo

    # -- Translation -------------------------------------------------------

    def _build_translation(self) -> None:
        self._section("Translation", "What listens, what translates, and how much it remembers.")

        card = self._card("Languages")
        languages = QLabel("Japanese  ->  English")
        languages.setObjectName("secondary")
        card.add_row(
            SettingRow(
                "Source and target",
                "Japanese speech in, English text out. The models are chosen and "
                "tuned for this pair.",
                languages,
            )
        )

        card = self._card("Profile")
        self._profile = SegmentedControl(
            [(key, PROFILE_LABELS[key]) for key in PROFILES]
        )
        self._profile.set_tooltips(PROFILE_BLURBS)
        self._profile.changed.connect(self._choose_profile)
        self._profile_row = card.add_row(
            SettingRow("Performance profile", PROFILE_BLURBS["balanced"], self._profile)
        )

        card = self._card("Models")
        self._recognizer = _Combo()
        self._recognizer.setFixedWidth(280)
        for name in TRANSCRIBERS:
            self._recognizer.addItem(name)
        self._recognizer.currentTextChanged.connect(self._choose_recognizer)
        self._recognizer_row = card.add_row(
            SettingRow("Recognition model", "", self._recognizer)
        )

        self._translator = _Combo()
        self._translator.setFixedWidth(280)
        for name in TRANSLATORS:
            self._translator.addItem(name)
        self._translator.addItem(CUSTOM_LLM)
        self._translator.currentTextChanged.connect(self._choose_translator)
        self._translator_row = card.add_row(
            SettingRow("Translation model", "", self._translator)
        )

        self._custom_tag = QLineEdit()
        self._custom_tag.setFixedWidth(280)
        self._custom_tag.setPlaceholderText("an Ollama tag, e.g. gemma3:12b")
        self._custom_tag.editingFinished.connect(
            lambda: self._set_llm_tag(self._custom_tag.text().strip())
        )
        self._custom_row = card.add_row(
            SettingRow(
                "Ollama model",
                "Any model Ollama has pulled. Use an instruction model, not a "
                "reasoning one: a model that thinks out loud puts the thinking "
                "in the captions.",
                self._custom_tag,
            )
        )

        status = QWidget()
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(10)
        self._status_text = QLabel("")
        self._status_text.setObjectName("secondary")
        status_layout.addWidget(self._status_text)
        self._download = QPushButton("Download")
        self._download.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download.clicked.connect(
            lambda: self.download_requested.emit(self.cfg.llm_model)
        )
        status_layout.addWidget(self._download)
        self._status_row = card.add_row(SettingRow("Status", "", status))
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.hide()
        card.add_widget(self._progress)

        card = self._card("Context")
        names = QLineEdit()
        names.setFixedWidth(280)
        names.setPlaceholderText("天音 = Amane, ラミィ, ホロライブ")
        names.setClearButtonEnabled(True)
        names.editingFinished.connect(lambda: self._set("initial_prompt", names.text().strip()))
        self._loaders.append(lambda: names.setText(self.cfg.initial_prompt))
        card.add_row(
            SettingRow(
                "Names and terms",
                "Words the audio will use that would otherwise be misheard or "
                "misread. Give a reading where it matters. Keep it short.",
                names,
            )
        )
        self._slider(
            card, "llm_context_utterances", "Lines of context",
            "How many previous lines the translation model is shown. More keeps "
            "names and subjects straight; too many lets old topics bleed in. "
            "Only LLM translators use it.",
            0, 10, 1, "{:.0f}", integer=True,
        )
        self._combo(
            card, "llm_context_reset_sec", "Forget context after silence",
            "Start fresh once nobody has spoken for this long.",
            [("30 seconds", 30.0), ("1 minute", 60.0), ("2 minutes", 120.0),
             ("5 minutes", 300.0), ("10 minutes", 600.0)],
            width=160,
        )
        self._toggle(
            card, "llm_keep_honorifics", "Keep honorifics",
            "Leave -san, -chan, senpai and the like in the English.",
        )
        self._toggle(
            card, "mt_live_enabled", "Draft translation while speaking",
            "Show provisional English before the sentence is finished. It can "
            "change; the finished line never does.",
        )
        self._toggle(
            card, "mt_hybrid_preview", "Fast drafts with an LLM translator",
            "Draft with the built-in translator and let the LLM write the final "
            "line. Uses about 1 GB more VRAM.",
        )
        self._toggle(
            card, "mt_fallback_to_nmt", "Fall back to the built-in translator",
            "If the LLM cannot be reached, keep translating instead of showing "
            "Japanese only.",
        )

        card = self._card("Speakers")
        self._toggle(
            card, "diarize_enabled", "Label speakers",
            "Tag each line with who said it and start a new line when the voice "
            "changes. Numbers are assigned in the order voices are first heard.",
        )
        self._combo(
            card, "diarize_max_speakers", "Number of people",
            "If you know it, say so: it stops one excited voice being split in two.",
            [("Up to 6", 6)] + [(f"{n} people", n) for n in (2, 3, 4, 5, 8)],
            width=160,
        )

    # -- Performance -------------------------------------------------------

    def _build_performance(self) -> None:
        self._section("Performance", "How the work is shared between speed, accuracy and memory.")

        card = self._card("Hardware")
        self._combo(
            card, "device", "Run on",
            "Automatic uses the graphics card and falls back to the processor, "
            "with a smaller model, if it cannot.",
            [("Automatic", "auto"), ("Graphics card (CUDA)", "cuda"), ("Processor (CPU)", "cpu")],
        )
        self._combo(
            card, "compute_type", "Precision",
            "16-bit is the right choice on any recent NVIDIA card. Mixed 8-bit "
            "halves the recogniser's memory for a small loss of accuracy.",
            [("16-bit float (recommended)", "float16"), ("Mixed 8-bit (less memory)", "int8_float16"),
             ("32-bit float", "float32")],
        )
        self._toggle(
            card, "vram_saver", "VRAM saver",
            "8-bit weights for the recogniser and the built-in translator. "
            "For cards with 8 GB or less, or running beside a game.",
        )
        self._toggle(
            card, "unload_on_stop", "Free memory when stopped",
            "Release the models on Stop. The next Start takes a few seconds longer.",
        )

        card = self._card("Recognition")
        self._slider(
            card, "process_interval_sec", "Processing interval",
            "How often the recogniser looks at the audio. Lower is more "
            "responsive and works the GPU harder.",
            0.15, 1.0, 0.05, "{:.2f} s",
        )
        self._slider(
            card, "max_buffer_sec", "Context length",
            "How much recent audio the recogniser sees at once. More context "
            "is more accurate and makes each pass slower.",
            4.0, 20.0, 1.0, "{:.0f} s",
        )
        self._slider(
            card, "commit_holdback_sec", "Caution with the newest words",
            "How far the audio must run past a word before its text is treated "
            "as settled. Higher is more accurate and makes the live Japanese "
            "trail the speaker by that much more.",
            0.0, 1.0, 0.05, "{:.2f} s",
        )
        self._combo(
            card, "beam_size", "Beam width",
            "How many alternatives the recogniser keeps in play. Wider helps "
            "most when people talk over each other.",
            [(str(n), n) for n in (1, 2, 3, 5, 8, 10)], width=110,
        )

        card = self._card("Speech segmentation")
        self._slider(
            card, "endpoint_silence_sec", "End-of-sentence pause",
            "How long a pause ends a sentence. This is the main delay setting: "
            "the final English cannot arrive sooner than this after the last word.",
            0.2, 1.2, 0.05, "{:.2f} s",
        )
        self._slider(
            card, "eager_endpoint_sec", "Finish complete sentences early",
            "When what was said already reads as a whole sentence, take it "
            "this far into the pause instead of waiting out the full one above. "
            "0 turns this off.",
            0.0, 0.6, 0.05, "{:.2f} s",
        )
        self._slider(
            card, "long_pause_sec", "Patience for unfinished sentences",
            "A sentence that stops on a dangling particle is held this long in "
            "case the speaker was only drawing breath.",
            0.5, 3.0, 0.1, "{:.1f} s",
        )
        self._slider(
            card, "mt_max_utterance_sec", "Longest hold",
            "Speech with no pause and no sentence ending is translated anyway "
            "after this long.",
            2.0, 20.0, 0.5, "{:.1f} s",
        )
        self._slider(
            card, "vad_threshold", "Voice detection threshold",
            "How sure the detector must be that someone is talking. Raise it if "
            "music keeps being transcribed; lower it if quiet speech is missed.",
            0.2, 0.9, 0.05, "{:.2f}",
        )

    # -- Appearance --------------------------------------------------------

    def _build_appearance(self) -> None:
        self._section("Appearance", "The window, and the captions drawn over your screen.")

        card = self._card("Window")
        theme = SegmentedControl([("dark", "Dark"), ("light", "Light")])
        theme.changed.connect(lambda key: self._set("theme", key))
        self._loaders.append(lambda: theme.set_current(self.cfg.theme))
        card.add_row(SettingRow("Theme", "", theme))
        self._slider(
            card, "ui_font_size", "Interface text size", "",
            8, 14, 1, "{:.0f} pt", integer=True,
        )
        self._slider(
            card, "translation_font_size", "Translation text size",
            "Size of the English in the transcript. The Japanese follows it.",
            10, 28, 1, "{:.0f} pt", integer=True,
        )
        self._toggle(card, "show_source", "Show the original Japanese",
                     "Under each translation, for checking a name or a nuance.")
        self._toggle(card, "show_timestamps", "Show timestamps", "")

        card = self._card("Overlay")
        self._toggle(
            card, "overlay_enabled", "Show captions over the screen",
            "A click-through caption plate on the primary display. Works over "
            "borderless-windowed games; exclusive fullscreen cannot be drawn over.",
        )
        self._toggle(card, "overlay_show_source", "Japanese on the overlay too",
                     "Under the English, smaller.")
        self._slider(card, "overlay_opacity", "Background opacity",
                     "Of the plate behind the text. The text stays solid.",
                     0, 100, 1, "{:.0f}%", integer=True)
        position = SegmentedControl([("bottom", "Bottom"), ("top", "Top")])
        position.changed.connect(lambda key: self._set("overlay_position", key))
        self._loaders.append(lambda: position.set_current(self.cfg.overlay_position))
        card.add_row(SettingRow("Position", "", position))
        self._slider(card, "max_width_frac", "Width",
                     "How much of the screen's width a caption may use before wrapping.",
                     30, 100, 1, "{:.0f}%", scale=100.0)
        self._slider(card, "bottom_margin_frac", "Distance from the edge", "",
                     0, 40, 1, "{:.0f}%", scale=100.0)
        self._slider(card, "font_size", "Caption text size", "",
                     14, 60, 1, "{:.0f} pt", integer=True)
        self._slider(card, "display_history_sec", "Caption stays for",
                     "How long a finished caption remains if nothing replaces it.",
                     2.0, 30.0, 1.0, "{:.0f} s")

    # -- General -----------------------------------------------------------

    def _build_general(self) -> None:
        self._section("General", "How the app behaves on your desktop.")
        card = self._card("Startup")
        self._toggle(card, "launch_on_startup", "Launch when Windows starts", "")
        self._toggle(card, "start_minimized", "Start minimized",
                     "Open in the tray instead of showing the window.")
        self._toggle(card, "autostart_translation", "Start translating on launch",
                     "Skip the Start button.")
        self._toggle(
            card, "restore_last_config", "Remember the last configuration",
            "Come back with the profile and models used last time. Off, every "
            "launch starts from Balanced.",
        )
        card = self._card("Window")
        self._toggle(card, "minimize_to_tray", "Keep running in the tray when closed",
                     "Closing the window hides it. Quit from the tray icon.")

        card = self._card("Maintenance")
        logs = QPushButton("Open folder")
        logs.setCursor(Qt.CursorShape.PointingHandCursor)
        logs.clicked.connect(self.open_logs_requested.emit)
        card.add_row(SettingRow("Logs and settings file", "Where the log and config.json live.", logs))
        reset = QPushButton("Reset")
        reset.setObjectName("danger")
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(self.reset_requested.emit)
        card.add_row(SettingRow("Reset all settings", "Back to the defaults. The transcript is not affected.", reset))

    # -- model choices -----------------------------------------------------

    def _choose_profile(self, name: str) -> None:
        if self._loading:
            return
        apply_profile(self.cfg, name)
        self.refresh()
        self.changed.emit("profile")

    def _choose_recognizer(self, name: str) -> None:
        if self._loading or name not in TRANSCRIBERS:
            return
        changed = False
        for key, value in TRANSCRIBERS[name].items():
            if getattr(self.cfg, key) != value:
                setattr(self.cfg, key, value)
                changed = True
        if changed:
            self.cfg.profile = detect_profile(self.cfg)
            self.refresh()
            self.changed.emit("model")

    def _choose_translator(self, name: str) -> None:
        if self._loading:
            return
        if name == CUSTOM_LLM:
            self.cfg.mt_enabled, self.cfg.task, self.cfg.mt_backend = True, "transcribe", "llm"
        elif name in TRANSLATORS:
            for key, value in TRANSLATORS[name]["set"].items():
                setattr(self.cfg, key, value)
        self.cfg.profile = detect_profile(self.cfg)
        self.refresh()
        self.changed.emit("mt_backend")

    def _set_llm_tag(self, tag: str) -> None:
        if tag and tag != self.cfg.llm_model:
            self.cfg.llm_model = tag
            self.cfg.profile = detect_profile(self.cfg)
            self.refresh()
            self.changed.emit("llm_model")

    # -- loading -----------------------------------------------------------

    def _load_profile(self) -> None:
        name = detect_profile(self.cfg)
        self._profile.set_current(name)
        self._profile_row.set_help(PROFILE_BLURBS.get(name, PROFILE_BLURBS["custom"]))

    def refresh(self) -> None:
        """Point every control at what the config currently says."""
        self._loading = True
        try:
            for load in self._loaders:
                load()
            self._load_profile()
            recognizer = recognizer_name(self.cfg)
            if self._recognizer.findText(recognizer) < 0:
                self._recognizer.addItem(recognizer)
            self._recognizer.setCurrentText(recognizer)
            direct = self.cfg.task == "translate" and not self.cfg.mt_enabled
            self._recognizer.setEnabled(not direct)
            self._recognizer_row.set_help(
                "Set by the translation model: Whisper is doing both jobs."
                if direct
                else RECOGNIZER_NOTES.get(recognizer, "")
            )
            translator = translator_name(self.cfg)
            self._translator.setCurrentText(translator)
            custom = translator == CUSTOM_LLM
            self._translator_row.set_help(
                "A model you have pulled into Ollama yourself."
                if custom
                else TRANSLATORS.get(translator, {}).get("note", "")
            )
            self._custom_row.setVisible(custom)
            self._custom_tag.setText(self.cfg.llm_model if custom else "")
            uses_llm = self.cfg.mt_enabled and self.cfg.mt_backend == "llm"
            self._status_row.setVisible(uses_llm)
            if not uses_llm:
                self._progress.hide()
        finally:
            self._loading = False

    def set_translator_status(self, text: str, can_download: bool, tone: str = "") -> None:
        """Whether the chosen LLM is ready, from the controller's Ollama check."""
        color = getattr(self._palette, tone) if tone else self._palette.text_2
        self._status_text.setText(text)
        self._status_text.setStyleSheet(f"color: {color};")
        self._download.setVisible(can_download)
        self._download.setEnabled(True)

    def set_download_progress(self, fraction: float | None, text: str) -> None:
        self._download.setEnabled(False)
        self._progress.show()
        if fraction is None:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 1000)
            self._progress.setValue(int(fraction * 1000))
        self._status_text.setText(text)
        self._status_text.setStyleSheet(f"color: {self._palette.text_2};")

    def end_download(self) -> None:
        self._progress.hide()
        self._download.setEnabled(True)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        for switch in self._switches:
            switch.set_palette(palette)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._sync_side(self._scroll.verticalScrollBar().value()))
