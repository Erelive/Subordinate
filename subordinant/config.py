"""Runtime configuration.

Defaults are tuned for an 8 GB laptop GPU. Values can be overridden by a JSON
file at %APPDATA%\\SubOrdinant\\config.json; any key not present there keeps the
default below.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

# The model rate Whisper expects. Not configurable - changing it breaks the model.
SAMPLE_RATE = 16000


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "SubOrdinant"


@dataclass
class Config:
    # --- model ---
    model: str = "large-v3-turbo"
    device: str = "cuda"  # "cuda" or "cpu"
    compute_type: str = "float16"  # "int8_float16" halves VRAM, slightly worse
    language: str = "en"
    task: str = "transcribe"  # v2: "translate" for any-language -> English
    beam_size: int = 1  # greedy; beams cost latency for little gain here

    # --- streaming ---
    # How much fresh audio to accumulate before running inference again. Lower
    # means more responsive captions but more GPU work per second of audio.
    process_interval_sec: float = 0.5
    # Don't run inference at all until this much speech exists; very short
    # buffers produce garbage and hallucinations.
    min_chunk_sec: float = 1.0
    # Hard cap on the rolling buffer. Longer context is more accurate but each
    # inference re-transcribes the whole buffer, so cost grows with length and a
    # long buffer is what pushes RTF towards 1.0 during continuous speech.
    max_buffer_sec: float = 12.0
    # Silence this long ends the utterance: commit everything, clear the buffer.
    silence_flush_sec: float = 0.7
    # If the worker falls this far behind real time, drop the backlog rather
    # than let caption latency grow without bound.
    max_backlog_sec: float = 6.0

    # --- voice activity ---
    # Cheap pre-gate only: decides whether audio is worth sending to the GPU.
    # The real speech/non-speech decision is faster-whisper's bundled Silero VAD.
    energy_gate_db: float = -48.0
    vad_filter: bool = True

    # --- overlay ---
    font_family: str = "Segoe UI"
    font_size: int = 26
    # Fraction of screen width the caption box may occupy before wrapping.
    max_width_frac: float = 0.72
    # Distance from the bottom of the screen, as a fraction of screen height.
    bottom_margin_frac: float = 0.08
    committed_color: str = "#ffffff"
    unstable_color: str = "#9aa0a6"
    background_rgba: str = "rgba(0, 0, 0, 190)"
    # How much committed text to keep on screen, in seconds of speech.
    display_history_sec: float = 12.0
    # Hide the overlay entirely after this long with nothing to show.
    hide_after_idle_sec: float = 4.0

    # --- audio ---
    # Capture buffer size in frames, at the device's native rate. Smaller means
    # lower capture latency and more callbacks.
    frames_per_buffer: int = 1024

    @classmethod
    def load(cls) -> "Config":
        path = config_dir() / "config.json"
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable config at %s: %s", path, exc)
            return cfg
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                log.warning("unknown config key %r ignored", key)
        return cfg

    def save(self) -> Path:
        path = config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
