"""What connects the interface to the pipeline.

The pipeline runs on its own threads and reports through PipelineEvents; the
interface lives on Qt's thread and must only be touched from it. Everything
crossing between them goes through Bridge's signals, which Qt delivers queued
- so a worker thread can announce a caption without ever holding up, or being
held up by, the window.

The controller also owns the things that outlive any one run: the loaded
models, the transcript, the settings, the overlay and the tray icon. A run is
cheap to throw away and start again; those are not.
"""

from __future__ import annotations

import logging
import os
import threading

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .. import autostart
from ..config import Config, config_dir
from ..history import History
from ..models import ModelCache
from ..overlay import CaptionOverlay, app_icon
from ..pipeline import ERROR, IDLE, LOADING, PAUSED, CaptionPipeline, PipelineEvents
from ..profiles import PROFILE_LABELS, apply_profile, detect_profile, short_model_name
from .settings import ENGINE_KEYS
from .theme import STATE_LABELS
from .transcript import TranscriptModel
from .window import MainWindow

log = logging.getLogger(__name__)

# Settings whose effect is on how things are drawn, applied the moment they
# change.
_LOOK_KEYS = {
    "theme", "ui_font_size", "translation_font_size", "show_source", "show_timestamps",
}
_OVERLAY_KEYS = {
    "overlay_enabled", "overlay_show_source", "overlay_opacity", "overlay_position",
    "max_width_frac", "bottom_margin_frac", "font_size", "display_history_sec",
}


class Bridge(QObject):
    """Pipeline events as Qt signals. The first argument is always the run's
    generation, so that the last words of a run that has been replaced are
    not mistaken for the first words of its replacement."""

    state = pyqtSignal(int, str, str)
    notice = pyqtSignal(int, str, str)
    caption = pyqtSignal(int, str, str)
    preview = pyqtSignal(int, str, str)
    utterance = pyqtSignal(int, object)
    metrics = pyqtSignal(int, dict)
    # From helper threads that are not a pipeline.
    translator_status = pyqtSignal(str, bool, str)
    download_progress = pyqtSignal(object, str)
    download_done = pyqtSignal(bool, str)


class _Events(PipelineEvents):
    def __init__(self, bridge: Bridge, generation: int):
        self._bridge = bridge
        self._generation = generation

    def on_state(self, state, detail):
        self._bridge.state.emit(self._generation, state, detail)

    def on_notice(self, level, message):
        self._bridge.notice.emit(self._generation, level, message)

    def on_caption(self, committed, unstable):
        self._bridge.caption.emit(self._generation, committed, unstable)

    def on_preview(self, english, source):
        self._bridge.preview.emit(self._generation, english, source)

    def on_utterance(self, utterance):
        self._bridge.utterance.emit(self._generation, utterance)

    def on_metrics(self, snapshot):
        self._bridge.metrics.emit(self._generation, snapshot)


class Controller(QObject):
    def __init__(self, app: QApplication, cfg: Config, start_hidden: bool = False):
        super().__init__()
        self.app = app
        self.cfg = cfg
        self.models = ModelCache()
        self.history = History(cfg.history_max_entries)
        self.model = TranscriptModel(self.history)
        self.bridge = Bridge()
        self._pipeline: CaptionPipeline | None = None
        self._generation = 0
        self._state = IDLE
        self._want_start = False  # start again as soon as the old run is gone
        self._restart_pending = False  # engine settings changed while running
        self._quitting = False
        self._tray_hint_shown = False
        self._downloading = False

        cfg.launch_on_startup = autostart.is_enabled()

        self.window = MainWindow(cfg, self.model, self.history)
        self.overlay = CaptionOverlay(cfg)
        self.overlay.set_enabled(cfg.overlay_enabled)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save)

        self._build_tray()
        self._wire()

        for note in cfg.load_notes:
            self.window.notify("warning", note)
        self.window.restore_geometry()
        if not start_hidden:
            self.window.show()
        self._refresh_engine_label()
        if cfg.autostart_translation:
            QTimer.singleShot(200, self.start)

    # -- wiring ------------------------------------------------------------

    def _wire(self) -> None:
        queued = Qt.ConnectionType.QueuedConnection
        b = self.bridge
        b.state.connect(self._on_state, queued)
        b.notice.connect(self._on_notice, queued)
        b.caption.connect(self._on_caption, queued)
        b.preview.connect(self._on_preview, queued)
        b.utterance.connect(self._on_utterance, queued)
        b.metrics.connect(self._on_metrics, queued)
        b.translator_status.connect(self.window.settings_page.set_translator_status, queued)
        b.download_progress.connect(self.window.settings_page.set_download_progress, queued)
        b.download_done.connect(self._download_done, queued)

        w = self.window
        w.start_requested.connect(self.start)
        w.stop_requested.connect(self.stop)
        w.pause_requested.connect(self.set_paused)
        w.profile_chosen.connect(self._choose_profile)
        w.clear_requested.connect(self.clear)
        w.copy_latest_requested.connect(self.copy_latest)
        w.copy_all_requested.connect(self.copy_all)
        w.overlay_toggled.connect(lambda on: self._set("overlay_enabled", on))
        w.source_toggled.connect(lambda on: self._set("show_source", on))
        w.metrics_toggled.connect(lambda on: self._set("metrics_visible", on))
        w.banner_action.connect(self._banner_action)
        w.closing.connect(self._window_closing)
        w.page_changed.connect(self._page_changed)

        s = w.settings_page
        s.changed.connect(self._setting_changed)
        s.download_requested.connect(self.download_model)
        s.open_logs_requested.connect(self._open_logs)
        s.reset_requested.connect(self._reset_settings)

        self.app.aboutToQuit.connect(self._shutdown)

    def _build_tray(self) -> None:
        self.tray: QSystemTrayIcon | None = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("SubOrdinant")
        menu = QMenu()
        self._tray_show = menu.addAction("Open SubOrdinant", self.show_window)
        menu.addSeparator()
        self._tray_start = menu.addAction("Start translating", self._tray_toggle_run)
        self._tray_pause = menu.addAction("Pause", lambda: self.set_paused(self._state != PAUSED))
        self._tray_overlay = menu.addAction("Captions over the screen")
        self._tray_overlay.setCheckable(True)
        self._tray_overlay.setChecked(self.cfg.overlay_enabled)
        self._tray_overlay.toggled.connect(lambda on: self._set("overlay_enabled", on))
        menu.addSeparator()
        menu.addAction("Quit", self.quit)
        self._tray_menu = menu  # keep a reference; the tray does not own it
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    # -- run control -------------------------------------------------------

    def start(self) -> None:
        """Begin translating with the current settings."""
        if self._pipeline is not None and self._pipeline.running:
            if self._state in (IDLE, ERROR):
                # The previous run is on its way out. Start when it has gone.
                self._want_start = True
            return
        self._want_start = False
        self._restart_pending = False
        self.window.dismiss_notice()
        self._generation += 1
        self._pipeline = CaptionPipeline(
            self.cfg,
            events=_Events(self.bridge, self._generation),
            models=self.models,
        )
        self._on_state(self._generation, LOADING, "Starting")
        self._pipeline.start()
        self._refresh_engine_label()

    def stop(self) -> None:
        self._want_start = False
        pipeline = self._pipeline
        if pipeline is None:
            return
        log.info("stop requested")
        # Not joined here: the engine thread may be mid-pass, and waiting for
        # it on the GUI thread is a frozen window. It reports "idle" when it
        # has actually finished.
        pipeline.stop(wait=0.0)
        self.overlay.clear()
        if not pipeline.running:
            self._on_state(self._generation, IDLE, "")

    def restart(self) -> None:
        """Stop, and start again once the old run has let go of the models."""
        if self._pipeline is None or not self._pipeline.running:
            self.start()
            return
        log.info("restarting the engine")
        self._want_start = True
        self._pipeline.stop(wait=0.0)
        self.overlay.clear()
        self.window.set_state(LOADING, "Restarting")

    def set_paused(self, paused: bool) -> None:
        if self._pipeline is None or not self._pipeline.running:
            return
        self._pipeline.set_paused(paused)
        if paused:
            self.overlay.clear()

    def _choose_profile(self, name: str) -> None:
        if name == "custom" or name == detect_profile(self.cfg):
            self.window.sync_from_config()
            return
        log.info("profile chosen: %s (was %s)", name, detect_profile(self.cfg))
        apply_profile(self.cfg, name)
        self.window.settings_page.refresh()
        self.window.sync_from_config()
        self._refresh_engine_label()
        self._save_timer.start()
        if self._pipeline is not None and self._pipeline.running:
            # Choosing a profile is asking for that profile now, not later.
            self.restart()

    # -- pipeline events (GUI thread) -------------------------------------

    def _current(self, generation: int) -> bool:
        return generation == self._generation

    def _on_state(self, generation: int, state: str, detail: str) -> None:
        if not self._current(generation):
            return
        self._state = state
        self.window.set_state(state, detail)
        self._update_tray(state, detail)
        if state == ERROR:
            self.overlay.clear()
            self.window.notify("error", detail, "Try again", "retry")
            if self.tray is not None and not self.window.isVisible():
                self.tray.showMessage(
                    "SubOrdinant", detail, QSystemTrayIcon.MessageIcon.Critical, 8000
                )
        if state in (IDLE, ERROR):
            QTimer.singleShot(60, self._run_ended)

    def _run_ended(self) -> None:
        """The engine thread has reported its last state. Clean up after it."""
        pipeline = self._pipeline
        if pipeline is not None and pipeline.running:
            # It announced the end a moment before its thread returned.
            QTimer.singleShot(80, self._run_ended)
            return
        if self._want_start and not self._quitting:
            self.start()
            return
        if self.cfg.unload_on_stop and self._state == IDLE:
            threading.Thread(target=self.models.release, name="release", daemon=True).start()

    def _on_notice(self, generation: int, level: str, message: str) -> None:
        if not self._current(generation):
            return
        self.window.notify(level, message)

    def _on_caption(self, generation: int, committed: str, unstable: str) -> None:
        if not self._current(generation):
            return
        self.window.live_page.live.set_caption(committed, unstable)
        self.overlay.set_caption(committed, unstable)

    def _on_preview(self, generation: int, english: str, _source: str) -> None:
        if not self._current(generation):
            return
        self.window.live_page.live.set_preview(english)
        self.overlay.set_preview(english)

    def _on_utterance(self, generation: int, utterance) -> None:
        if not self._current(generation):
            return
        self.model.add(utterance)
        self.overlay.show_utterance(utterance)

    def _on_metrics(self, generation: int, snapshot: dict) -> None:
        if self._current(generation):
            self.window.update_metrics(snapshot)

    # -- transcript actions ------------------------------------------------

    def clear(self) -> None:
        self.model.clear()
        self.overlay.clear()
        self.window.toast("Transcript cleared")

    def copy_latest(self) -> None:
        entry = self.history.latest()
        if entry is None:
            self.window.toast("Nothing to copy yet")
            return
        QGuiApplication.clipboard().setText(entry.primary)
        self.window.toast("Copied the latest translation")

    def copy_all(self) -> None:
        text = self.history.as_text(
            source=self.cfg.show_source, timestamps=self.cfg.show_timestamps
        )
        if not text:
            self.window.toast("Nothing to copy yet")
            return
        QGuiApplication.clipboard().setText(text)
        self.window.toast(f"Copied {len(self.history)} lines")

    # -- settings ----------------------------------------------------------

    def _set(self, key: str, value) -> None:
        if getattr(self.cfg, key) == value:
            return
        setattr(self.cfg, key, value)
        self._setting_changed(key)
        self.window.settings_page.refresh()

    def _setting_changed(self, key: str) -> None:
        cfg = self.cfg
        self._save_timer.start()
        if key in _LOOK_KEYS:
            if key in ("theme", "ui_font_size"):
                self.window.apply_theme()
            self.window.sync_from_config()
        if key in _OVERLAY_KEYS:
            self.overlay.set_enabled(cfg.overlay_enabled)
            self.overlay.apply_settings()
            if hasattr(self, "_tray_overlay"):
                self._tray_overlay.blockSignals(True)
                self._tray_overlay.setChecked(cfg.overlay_enabled)
                self._tray_overlay.blockSignals(False)
            self.window.sync_from_config()
        if key == "metrics_visible":
            self.window.sync_from_config()
        if key == "launch_on_startup":
            if not autostart.set_enabled(cfg.launch_on_startup):
                cfg.launch_on_startup = autostart.is_enabled()
                self.window.settings_page.refresh()
                self.window.notify("warning", "Could not change the Windows startup entry.")
        if key == "history_max_entries":
            self.history.max_entries = cfg.history_max_entries
        if key == "mt_live_enabled" and self._pipeline is not None:
            self._pipeline.set_live_preview(cfg.mt_live_enabled)
            if not cfg.mt_live_enabled:
                self.window.live_page.live.set_preview("")
                self.overlay.set_preview("")
        if key in ENGINE_KEYS:
            self.window.sync_from_config()
            self._refresh_engine_label()
            if key in ("mt_backend", "llm_model", "profile", "llm_endpoint"):
                self.check_translator()
            if self._pipeline is not None and self._pipeline.running and self._state != ERROR:
                self._restart_pending = True
                self.window.notify(
                    "info",
                    "Settings changed. They take effect when translation restarts.",
                    "Restart now",
                    "restart",
                )

    def _banner_action(self, key: str) -> None:
        log.info("banner action: %s", key)
        self.window.dismiss_notice()
        if key == "retry":
            self.start()
        elif key == "restart":
            self.restart()
        elif key == "settings":
            self.window.show_page("settings")

    def _refresh_engine_label(self) -> None:
        cfg = self.cfg
        if cfg.task == "translate" and not cfg.mt_enabled:
            translator = "direct"
        elif not cfg.mt_enabled:
            translator = "no translation"
        elif cfg.mt_backend == "llm":
            translator = short_model_name(cfg.llm_model)
        else:
            translator = "Sugoi v4"
        self.window.set_engine_label(f"{short_model_name(cfg.model)}   ·   {translator}")
        profile = PROFILE_LABELS.get(detect_profile(cfg), "Custom")
        drafts = (
            ", drafts by Sugoi v4"
            if cfg.mt_enabled and cfg.mt_backend == "llm" and cfg.mt_hybrid_preview
            else ""
        )
        self.window.set_ready_summary(
            [
                f"{profile} profile",
                f"Hears with {short_model_name(cfg.model)}",
                "Japanese captions only"
                if translator == "no translation"
                else "Whisper translates directly"
                if translator == "direct"
                else f"Translates with {translator}{drafts}",
            ]
        )

    def _reset_settings(self) -> None:
        fresh = Config()
        keep = {"window_geometry"}
        for name in vars(fresh):
            if name not in keep and name != "load_notes":
                setattr(self.cfg, name, getattr(fresh, name))
        autostart.set_enabled(False)
        self.window.settings_page.refresh()
        self.window.apply_theme()
        self.window.sync_from_config()
        self.overlay.set_enabled(self.cfg.overlay_enabled)
        self.overlay.apply_settings()
        self._refresh_engine_label()
        self._save_timer.start()
        self.window.notify("info", "Settings were reset to their defaults.")
        if self._pipeline is not None and self._pipeline.running:
            self.restart()

    def _open_logs(self) -> None:
        path = config_dir()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # noqa: S606 - opening the user's own folder
        except OSError as exc:
            self.window.notify("warning", f"Could not open {path}: {exc}")

    def _save(self) -> None:
        try:
            self.cfg.save()
        except OSError as exc:
            log.warning("could not save settings: %s", exc)

    # -- Ollama ------------------------------------------------------------

    def _page_changed(self, key: str) -> None:
        if key == "settings":
            self.check_translator()

    def check_translator(self) -> None:
        """Find out, off the GUI thread, whether the chosen LLM can be used."""
        cfg = self.cfg
        if not (cfg.mt_enabled and cfg.mt_backend == "llm") or self._downloading:
            return
        endpoint, model = cfg.llm_endpoint, cfg.llm_model
        self.bridge.translator_status.emit("Checking...", False, "")

        def work() -> None:
            from ..llm_translate import find_ollama, has_model, server_reachable

            if not server_reachable(endpoint):
                if find_ollama() is None:
                    self.bridge.translator_status.emit(
                        "Ollama is not installed. Get it from ollama.com.", False, "warn"
                    )
                else:
                    self.bridge.translator_status.emit(
                        "Ollama is not running. It will be started when you press Start.",
                        False,
                        "",
                    )
                return
            if has_model(endpoint, model):
                self.bridge.translator_status.emit("Installed and ready", False, "ok")
            else:
                self.bridge.translator_status.emit("Not downloaded yet", True, "warn")

        threading.Thread(target=work, name="ollama-check", daemon=True).start()

    def download_model(self, tag: str) -> None:
        if self._downloading or not tag:
            return
        self._downloading = True
        endpoint = self.cfg.llm_endpoint
        self.bridge.download_progress.emit(None, f"Starting download of {tag}...")

        def work() -> None:
            from ..llm_translate import pull_model, server_reachable, start_server
            from ..translate import TranslationUnavailable

            try:
                if not server_reachable(endpoint) and not start_server(endpoint):
                    raise TranslationUnavailable("Ollama is not running and could not be started.")

                def progress(fraction, status) -> None:
                    label = status or "Downloading"
                    if fraction is not None:
                        label = f"{label}  {fraction * 100:.0f}%"
                    self.bridge.download_progress.emit(fraction, label)

                pull_model(endpoint, tag, on_progress=progress, should_stop=lambda: self._quitting)
                self.bridge.download_done.emit(True, "")
            except TranslationUnavailable as exc:
                self.bridge.download_done.emit(False, str(exc))

        threading.Thread(target=work, name="ollama-pull", daemon=True).start()

    def _download_done(self, ok: bool, message: str) -> None:
        self._downloading = False
        self.window.settings_page.end_download()
        if ok:
            self.check_translator()
        else:
            self.window.settings_page.set_translator_status(message, True, "error")

    # -- tray and window ---------------------------------------------------

    def _update_tray(self, state: str, detail: str) -> None:
        if self.tray is None:
            return
        label = STATE_LABELS.get(state, state)
        self.tray.setToolTip(f"SubOrdinant - {label}" + (f"\n{detail[:60]}" if detail else ""))
        running = state not in (IDLE, ERROR)
        self._tray_start.setText("Stop translating" if running else "Start translating")
        self._tray_pause.setEnabled(running and state != LOADING)
        self._tray_pause.setText("Resume" if state == PAUSED else "Pause")

    def _tray_toggle_run(self) -> None:
        if self._state in (IDLE, ERROR):
            self.start()
        else:
            self.stop()

    def _tray_activated(self, reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_window()

    def show_window(self) -> None:
        self.window.show()
        if self.window.isMinimized():
            self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _window_closing(self) -> None:
        if self.cfg.minimize_to_tray and self.tray is not None and not self._quitting:
            self.window.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                self.tray.showMessage(
                    "SubOrdinant is still running",
                    "It stays in the tray. Right-click the icon to quit.",
                    QSystemTrayIcon.MessageIcon.Information,
                    4000,
                )
            return
        self.quit()

    def quit(self) -> None:
        self._quitting = True
        self.app.quit()

    def _shutdown(self) -> None:
        """Last thing before the process exits. Must not block for long."""
        self._quitting = True
        self._want_start = False
        try:
            if self.window.isVisible():
                self.cfg.window_geometry = bytes(
                    self.window.saveGeometry().toBase64()
                ).decode("ascii")
            self.cfg.save()
        except OSError as exc:
            log.warning("could not save settings: %s", exc)
        if self._pipeline is not None:
            self._pipeline.stop(wait=3.0)
        self.overlay.clear()
        if self.tray is not None:
            self.tray.hide()
