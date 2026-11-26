"""Palettes and the application stylesheet.

One palette object is the single source for every colour in the app - the
stylesheet below, the transcript's painted rows, the icons, the overlay's
accent - so the dark and light themes cannot drift apart widget by widget.

The look is deliberately quiet: one accent colour, used for the thing that is
active and for nothing else; hairline borders instead of shadows; and the
translation text as the brightest thing on screen, because it is what the
window is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str  # window
    surface: str  # cards, inputs
    raised: str  # hover, selected rows
    sunken: str  # the transcript well
    border: str
    border_strong: str
    text: str  # the translation; primary labels
    text_2: str  # the Japanese; descriptions
    text_3: str  # timestamps, hints, disabled
    accent: str
    accent_hover: str
    accent_soft: str  # tinted backgrounds behind accent text
    on_accent: str
    ok: str
    warn: str
    error: str
    draft: str  # provisional English

    def color(self, name: str) -> QColor:
        return QColor(getattr(self, name))


DARK = Palette(
    name="dark",
    bg="#0f1114",
    surface="#16191e",
    raised="#1e2229",
    sunken="#0b0d10",
    border="#23272f",
    border_strong="#323844",
    text="#eceef2",
    text_2="#a2a9b6",
    text_3="#666e7c",
    accent="#7b8cff",
    accent_hover="#93a1ff",
    accent_soft="#1c2040",
    on_accent="#0b0d17",
    ok="#4cc38a",
    warn="#e2a54a",
    error="#f0616d",
    draft="#8b93a3",
)

LIGHT = Palette(
    name="light",
    bg="#f4f5f7",
    surface="#ffffff",
    raised="#eceef2",
    sunken="#fafbfc",
    border="#e1e4ea",
    border_strong="#cdd2db",
    text="#14171c",
    text_2="#535b69",
    text_3="#8a919e",
    accent="#4c5bd4",
    accent_hover="#3d4bc0",
    accent_soft="#e8eaff",
    on_accent="#ffffff",
    ok="#1f9d63",
    warn="#b7791f",
    error="#d63b4a",
    draft="#7a8290",
)

PALETTES = {"dark": DARK, "light": LIGHT}

# One hue per speaker, in the order they are first heard. Chosen to stay
# apart from each other and from the accent on both themes.
SPEAKER_COLORS = (
    "#6ea8fe",
    "#e0a14b",
    "#5fbf87",
    "#c58af9",
    "#f0717b",
    "#4fc3d0",
)

# Colour of the status dot for each pipeline state.
STATE_COLORS = {
    "idle": "text_3",
    "loading": "accent",
    "listening": "ok",
    "recognizing": "accent",
    "translating": "warn",
    "paused": "text_3",
    "error": "error",
}

STATE_LABELS = {
    "idle": "Stopped",
    "loading": "Getting ready",
    "listening": "Listening",
    "recognizing": "Hearing Japanese",
    "translating": "Translating",
    "paused": "Paused",
    "error": "Problem",
}


def palette(name: str) -> Palette:
    return PALETTES.get(name, DARK)


_UI_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Yu Gothic UI", "Meiryo UI"]
_DISPLAY_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Yu Gothic UI", "Meiryo UI"]


def _available(families: list[str]) -> list[str]:
    have = set(QFontDatabase.families())
    picked = [f for f in families if f in have]
    return picked or ["Segoe UI"]


def ui_font(size: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Interface text. Latin from Segoe, kana and kanji from Yu Gothic UI."""
    font = QFont()
    font.setFamilies(_available(_UI_FAMILIES))
    font.setPointSizeF(size)
    font.setWeight(weight)
    return font


def display_font(size: float, weight: QFont.Weight = QFont.Weight.Medium) -> QFont:
    """The translation, and headings: the optical size cut for larger text."""
    font = QFont()
    font.setFamilies(_available(_DISPLAY_FAMILIES))
    font.setPointSizeF(size)
    font.setWeight(weight)
    return font


def _chevron_file(p: Palette) -> str:
    """Path of the combo-box arrow for this palette, written on first use.

    A stylesheet can only take an image from a file, and the app ships none,
    so the one glyph a stylesheet needs is rendered into the temp directory.
    If that fails the combo boxes simply have no arrow.
    """
    import tempfile
    from pathlib import Path

    from . import icons

    try:
        folder = Path(tempfile.gettempdir()) / "SubOrdinant-ui"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"chevron-{p.name}.png"
        if not path.exists():
            icons.pixmap("chevron", p.text_2, 14).save(str(path))
        return path.as_posix()
    except OSError:
        return ""


def stylesheet(p: Palette, base_pt: int = 10) -> str:
    """The application stylesheet for a palette and a base text size."""
    small = max(8, base_pt - 1)
    chevron = _chevron_file(p)
    return f"""
* {{
    font-family: "Segoe UI Variable Text", "Segoe UI", "Yu Gothic UI", "Meiryo UI";
    font-size: {base_pt}pt;
    outline: none;
}}
QWidget {{ color: {p.text}; background: transparent; }}
QWidget#root {{ background: {p.bg}; border: 1px solid {p.border}; }}
QWidget#titleBar {{ background: {p.bg}; }}
QWidget#page {{ background: {p.bg}; }}

QLabel {{ background: transparent; }}
QLabel#appName {{ font-size: {base_pt + 1}pt; font-weight: 600; color: {p.text}; }}
QLabel#muted, QLabel#hint {{ color: {p.text_3}; font-size: {small}pt; }}
QLabel#secondary {{ color: {p.text_2}; }}
QLabel#sectionTitle {{ font-size: {base_pt + 4}pt; font-weight: 600; color: {p.text}; }}
QLabel#cardTitle {{
    font-size: {small}pt; font-weight: 600; color: {p.text_3};
    letter-spacing: 1px;
}}
QLabel#rowTitle {{ color: {p.text}; }}
QLabel#rowHelp {{ color: {p.text_3}; font-size: {small}pt; }}
QLabel#metricValue {{ font-size: {base_pt + 3}pt; font-weight: 600; color: {p.text}; }}
QLabel#metricLabel {{ color: {p.text_3}; font-size: {small}pt; }}

QFrame#card {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame#well {{
    background: {p.sunken};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame#divider {{ background: {p.border}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#banner {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: 8px;
}}
QFrame#banner[level="warning"] {{ border-color: {p.warn}; }}
QFrame#banner[level="error"] {{ border-color: {p.error}; }}

/* -- buttons -- */
QPushButton {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 6px 14px;
    color: {p.text};
}}
QPushButton:hover {{ background: {p.raised}; border-color: {p.border_strong}; }}
QPushButton:pressed {{ background: {p.border}; }}
QPushButton:disabled {{ color: {p.text_3}; background: {p.bg}; border-color: {p.border}; }}
QPushButton#primary {{
    background: {p.accent}; border-color: {p.accent}; color: {p.on_accent};
    font-weight: 600; padding: 7px 20px;
}}
QPushButton#primary:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton#primary:disabled {{
    background: {p.raised}; border-color: {p.border}; color: {p.text_3};
}}
QPushButton#danger:hover {{ border-color: {p.error}; color: {p.error}; }}
QPushButton#stop {{
    background: {p.surface}; border-color: {p.error}; color: {p.error};
    font-weight: 600; padding: 7px 20px;
}}
QPushButton#stop:hover {{ background: {p.error}; color: {p.on_accent}; }}
QPushButton#chip {{
    background: transparent; border: 1px solid {p.border}; border-radius: 7px;
    padding: 5px 11px 5px 9px; color: {p.text_2};
}}
QPushButton#chip:hover {{ background: {p.raised}; color: {p.text}; }}
QPushButton#chip:checked {{
    background: {p.accent_soft}; border-color: {p.accent_soft}; color: {p.text};
}}
QLabel#delayChip {{
    border: 1px solid {p.border}; border-radius: 9px; padding: 2px 9px;
    color: {p.text_2}; font-size: {small}pt; font-weight: 600;
}}
QLabel#delayChip[tone="ok"] {{ color: {p.ok}; }}
QLabel#delayChip[tone="warn"] {{ color: {p.warn}; border-color: {p.warn}; }}
QLabel#liveTag {{
    font-size: {small}pt; font-weight: 600; color: {p.text_3}; letter-spacing: 1px;
}}
QLabel#liveTag[active="true"] {{ color: {p.accent}; }}
QPushButton#link {{
    background: transparent; border: none; color: {p.accent}; padding: 2px 4px;
}}
QPushButton#link:hover {{ color: {p.accent_hover}; }}

QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 7px;
    padding: 5px;
    color: {p.text_2};
}}
QToolButton:hover {{ background: {p.raised}; }}
QToolButton:pressed {{ background: {p.border}; }}
QToolButton:checked {{ background: {p.accent_soft}; }}
QToolButton#nav {{
    padding: 6px 12px; color: {p.text_2}; font-weight: 500;
}}
QToolButton#nav:hover {{ color: {p.text}; }}
QToolButton#nav:checked {{ background: {p.raised}; color: {p.text}; }}
QToolButton#chrome {{ border-radius: 6px; padding: 6px 10px; }}
QToolButton#chromeClose:hover {{ background: {p.error}; }}
QPushButton#side {{
    background: transparent; border: none; border-radius: 7px;
    padding: 7px 12px; color: {p.text_2}; text-align: left;
}}
QPushButton#side:hover {{ color: {p.text}; background: {p.surface}; }}
QPushButton#side:checked {{ background: {p.raised}; color: {p.text}; font-weight: 600; }}

/* segmented control */
QFrame#segmented {{
    background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px;
}}
QPushButton#segment {{
    background: transparent; border: none; border-radius: 6px;
    padding: 5px 14px; color: {p.text_2};
}}
QPushButton#segment:hover {{ color: {p.text}; }}
QPushButton#segment:checked {{
    background: {p.raised}; color: {p.text}; font-weight: 600;
}}
QPushButton#segment:disabled {{ color: {p.text_3}; }}

/* -- inputs -- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 5px 9px;
    color: {p.text};
    selection-background-color: {p.accent};
    selection-color: {p.on_accent};
    min-height: 18px;
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {p.border_strong};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled, QComboBox:disabled {{ color: {p.text_3}; background: {p.bg}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: 6px;
    padding: 4px;
    selection-background-color: {p.raised};
    selection-color: {p.text};
    color: {p.text};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {p.border_strong}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {p.text}; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {p.accent_hover}; }}
QSlider::handle:horizontal:disabled {{ background: {p.text_3}; }}

/* -- scrolling -- */
QScrollArea {{ border: none; background: transparent; }}
QAbstractScrollArea {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{
    background: {p.border_strong}; border-radius: 3px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_3}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QScrollBar:horizontal {{ height: 0; }}

QListView {{ background: transparent; border: none; }}

QToolTip {{
    background: {p.raised}; color: {p.text};
    border: 1px solid {p.border_strong}; border-radius: 6px; padding: 5px 8px;
}}
QMenu {{
    background: {p.surface}; border: 1px solid {p.border_strong};
    border-radius: 8px; padding: 5px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; color: {p.text}; }}
QMenu::item:selected {{ background: {p.raised}; }}
QMenu::item:disabled {{ color: {p.text_3}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 5px 6px; }}

QProgressBar {{
    background: {p.border}; border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 2px; }}
"""
