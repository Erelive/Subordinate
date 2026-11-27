"""The performance panel.

Hidden until asked for, so the main window stays a transcript. When shown it
answers one question first - is the app keeping up with the audio? - and then
breaks the delay down by stage so the answer can be acted on: a slow
recognition pass, a slow translator and a starved GPU all look the same from
the caption's side, and need different fixes.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .theme import Palette
from .widgets import divider


def _sec(value) -> str:
    return "-" if value is None else f"{value:.2f} s"


def _ms(value) -> str:
    return "-" if value is None else f"{value:.0f} ms"


class _Group(QWidget):
    """A heading over rows of label / value."""

    def __init__(self, title: str, rows: list[tuple[str, str, str]]):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        head = QLabel(title.upper())
        head.setObjectName("cardTitle")
        layout.addWidget(head)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(5)
        grid.setColumnStretch(0, 1)
        self.values: dict[str, QLabel] = {}
        for row, (key, label, tip) in enumerate(rows):
            name = QLabel(label)
            name.setObjectName("secondary")
            name.setToolTip(tip)
            value = QLabel("-")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setToolTip(tip)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self.values[key] = value
        layout.addLayout(grid)


class MetricsPanel(QFrame):
    def __init__(self, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self._palette = palette
        self.setFixedWidth(272)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        self._verdict = QLabel("Not running")
        self._verdict.setObjectName("metricValue")
        self._verdict.setWordWrap(True)
        layout.addWidget(self._verdict)
        self._reason = QLabel("Start translating to see live figures.")
        self._reason.setObjectName("rowHelp")
        self._reason.setWordWrap(True)
        layout.addWidget(self._reason)
        layout.addWidget(divider())

        self._groups = [
            _Group(
                "Delay",
                [
                    ("final", "Finished translation", "From the last word of a sentence to its final English. Median of recent sentences."),
                    ("final_p90", "... worst 1 in 10", "The same, 90th percentile: what a bad sentence costs."),
                    ("preview", "First draft", "From the start of a sentence to the first provisional English."),
                    ("source", "Japanese text", "From a word being spoken to its text being settled."),
                ],
            ),
            _Group(
                "Recognition",
                [
                    ("pass", "Time per pass", "Wall time of one recognition pass over the rolling buffer. Mean of recent passes."),
                    ("pass_p90", "... worst 1 in 10", "90th percentile pass time."),
                    ("rtf", "Real-time factor", "Recognition seconds per second of speech. Under 1.0 is faster than the speaker."),
                    ("passes", "Passes", "Recognition passes run this session."),
                ],
            ),
            _Group(
                "Sentences",
                [
                    ("early", "Finished early", "Sentences taken before the full end-of-sentence pause, because the text already read as complete."),
                    ("held", "Waited for more", "Pauses not treated as endings, because the text stopped mid-thought."),
                    ("revised", "Corrected", "Times the settled Japanese was replaced by a later, better reading before being translated."),
                ],
            ),
            _Group(
                "Translation",
                [
                    ("mt_final", "Final line", "Time the translator takes for one finished sentence."),
                    ("mt_preview", "Draft", "Time for one provisional translation."),
                    ("first_token", "LLM first word", "How long the LLM takes to begin answering."),
                    ("lines", "Lines", "Sentences translated this session."),
                    ("fallback", "By fallback", "Lines finished by the built-in translator because the chosen one failed."),
                    ("second", "Second look", "Finished lines corrected once the line after them had been heard."),
                ],
            ),
            _Group(
                "Audio",
                [
                    ("capture", "Capture latency", "Age of the newest audio when the pipeline picked it up."),
                    ("backlog", "Waiting", "Captured audio not yet looked at. Should stay near zero."),
                    ("backlog_max", "... peak", "The most audio that has been waiting at once."),
                    ("dropped", "Dropped", "Audio discarded because the app could not keep up."),
                ],
            ),
            _Group(
                "System",
                [
                    ("vram", "GPU memory", "Memory in use on the whole graphics card, other programs included."),
                    ("gpu", "GPU load", "How busy the graphics card is."),
                    ("ram", "App memory", "This program's working set. The LLM lives in Ollama and is not counted."),
                ],
            ),
        ]
        for group in self._groups:
            layout.addWidget(group)

        layout.addWidget(divider())
        self._engine = QLabel("")
        self._engine.setObjectName("rowHelp")
        self._engine.setWordWrap(True)
        layout.addWidget(self._engine)
        layout.addStretch(1)
        self._values = {k: v for g in self._groups for k, v in g.values.items()}

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def reset(self) -> None:
        self._verdict.setText("Not running")
        self._verdict.setStyleSheet("")
        self._reason.setText("Start translating to see live figures.")

    def update_metrics(self, m: dict) -> None:
        p = self._palette
        ok = m.get("keeping_up", True)
        self._verdict.setText("Keeping up" if ok else "Falling behind")
        self._verdict.setStyleSheet(f"color: {p.ok if ok else p.warn};")
        self._reason.setText(m.get("health", ""))

        v = self._values
        v["final"].setText(_sec(m.get("final_latency_sec")))
        v["final_p90"].setText(_sec(m.get("final_latency_p90_sec")))
        v["preview"].setText(_sec(m.get("preview_latency_sec")))
        v["source"].setText(_sec(m.get("source_lag_sec")))
        v["pass"].setText(_ms(m.get("mean_pass_ms")))
        v["pass_p90"].setText(_ms(m.get("p90_pass_ms")))
        v["rtf"].setText(f"{m.get('rtf', 0.0):.2f}" if m.get("passes") else "-")
        v["passes"].setText(str(m.get("passes", 0)))
        v["early"].setText(str(m.get("eager_endpoints", 0) + m.get("pause_splits", 0)))
        v["held"].setText(str(m.get("deferred_closes", 0)))
        v["revised"].setText(str(m.get("revised_lines", 0)))
        v["mt_final"].setText(_ms(m.get("mt_final_ms")))
        v["mt_preview"].setText(_ms(m.get("mt_preview_ms")))
        v["first_token"].setText(_ms(m.get("llm_first_token_ms")))
        v["lines"].setText(str(m.get("utterances", 0)))
        v["fallback"].setText(str(m.get("fallback_lines", 0)))
        v["second"].setText(str(m.get("second_look_changes", 0)))
        v["capture"].setText(_ms(m.get("capture_latency_ms")))
        v["backlog"].setText(_sec(m.get("backlog_sec")))
        v["backlog_max"].setText(_sec(m.get("backlog_max_sec")))
        dropped = m.get("backlog_dropped_sec", 0.0) + m.get("capture_dropped_sec", 0.0)
        blocks = m.get("capture_dropped_blocks", 0)
        v["dropped"].setText(
            f"{dropped:.1f} s" + (f" + {blocks} blocks" if blocks else "")
        )
        v["dropped"].setStyleSheet(f"color: {p.warn};" if dropped or blocks else "")
        used, total = m.get("gpu_mem_used_mib"), m.get("gpu_mem_total_mib")
        if used is not None and total:
            v["vram"].setText(f"{used / 1024:.1f} / {total / 1024:.0f} GB")
            v["vram"].setStyleSheet(f"color: {p.warn};" if used / total > 0.93 else "")
        else:
            v["vram"].setText("n/a")
        gpu = m.get("gpu_util_pct")
        v["gpu"].setText("n/a" if gpu is None else f"{gpu}%")
        ram = m.get("process_rss_mib")
        v["ram"].setText("n/a" if ram is None else f"{ram / 1024:.1f} GB")

        lines = [m.get("engine", ""), m.get("translator", ""), m.get("device_name", "")]
        self._engine.setText("\n".join(line for line in lines if line))
