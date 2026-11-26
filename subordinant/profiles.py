"""Performance profiles, and the catalogue of models the Settings page offers.

A profile is a named set of Config values. Choosing one in the GUI writes
those values and nothing else, so everything a profile does is visible in the
Settings page afterwards and any of it can be overridden - at which point the
profile reads "Custom".

The three differ in which models run and in how long the pipeline is willing
to wait before committing to a sentence. They are not three labels on one
configuration: Fast runs no LLM at all, Balanced and Accuracy run different
ones, and the recogniser changes too. The numbers behind each choice are in
BENCHMARKS.md; each value below says which measurement it came from.
"""

from __future__ import annotations

from .config import TRANSCRIBERS, Config

KOTOBA = "kotoba-tech/kotoba-whisper-v2.0-faster"
ANIME_WHISPER = "flyfront/anime-whisper-faster"
SUGOI_14B = "hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M"
QWEN3_4B = "qwen3:4b-instruct"

PROFILES: dict[str, dict] = {
    # Minimum delay. One small recogniser, one small translator, both in this
    # process, nothing to install. English is final roughly as fast as the
    # recogniser can settle the Japanese.
    "fast": {
        "model": KOTOBA,
        # Greedy decoding. Beam search earns its cost on overlapping voices
        # and costs a third more per pass on everything else.
        "beam_size": 1,
        "no_repeat_ngram_size": 0,
        "process_interval_sec": 0.25,
        "min_chunk_sec": 0.6,
        "commit_holdback_sec": 0.3,
        "max_buffer_sec": 8.0,
        # Short enough to catch the gap between two back-to-back sentences.
        "endpoint_silence_sec": 0.32,
        "eager_endpoint_sec": 0.3,
        "long_pause_sec": 0.8,
        "mt_max_utterance_sec": 5.0,
        "mt_backend": "ct2",
        "mt_beam_size": 5,
        "mt_preview_min_confidence": 0.35,
        "llm_context_utterances": 0,
    },
    # The default. The fast recogniser with a wider beam, drafts from the NMT
    # model the moment there is text, and a small instruction model that
    # rewrites each finished line with the previous ones in view - which is
    # what keeps names and dropped subjects straight.
    "balanced": {
        "model": KOTOBA,
        "beam_size": 5,
        "no_repeat_ngram_size": 0,
        "process_interval_sec": 0.35,
        "min_chunk_sec": 0.8,
        "commit_holdback_sec": 0.4,
        "max_buffer_sec": 10.0,
        "endpoint_silence_sec": 0.45,
        "eager_endpoint_sec": 0.3,
        "long_pause_sec": 1.1,
        "mt_max_utterance_sec": 7.0,
        "mt_backend": "llm",
        "llm_model": QWEN3_4B,
        "mt_beam_size": 5,
        "mt_preview_min_confidence": 0.5,
        "llm_context_utterances": 4,
    },
    # Best output. A recogniser tuned on voice acting, more patience before
    # deciding a sentence is over, and a 14B Japanese-to-English specialist
    # with a longer memory. Needs about 11 GB of VRAM all told.
    "accuracy": {
        "model": ANIME_WHISPER,
        "beam_size": 5,
        "no_repeat_ngram_size": 5,
        "process_interval_sec": 0.45,
        "min_chunk_sec": 1.0,
        "commit_holdback_sec": 0.6,
        "max_buffer_sec": 12.0,
        "endpoint_silence_sec": 0.6,
        "eager_endpoint_sec": 0.35,
        "long_pause_sec": 1.4,
        "mt_max_utterance_sec": 10.0,
        "mt_backend": "llm",
        "llm_model": SUGOI_14B,
        "mt_beam_size": 10,
        "mt_preview_min_confidence": 0.6,
        "llm_context_utterances": 6,
    },
}

# Applied with every profile, so that switching profiles cannot leave behind
# a mode the previous one did not set and this one does not expect.
_COMMON = {
    "language": "ja",
    "task": "transcribe",
    "mt_enabled": True,
    "mt_hybrid_preview": True,
    "vad_mode": "silero",
}

PROFILE_LABELS = {
    "fast": "Fast",
    "balanced": "Balanced",
    "accuracy": "Accuracy",
    "custom": "Custom",
}

PROFILE_BLURBS = {
    "fast": "Lowest delay. Built-in translator only - nothing else to install.",
    "balanced": "Quick drafts, then a context-aware rewrite of each line. Recommended.",
    "accuracy": "Best recognition and translation. More patient, needs ~11 GB of VRAM.",
    "custom": "Settings changed by hand.",
}


def apply_profile(cfg: Config, name: str) -> Config:
    """Write a profile's values onto cfg. Returns cfg for chaining."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; expected one of {list(PROFILES)}")
    for key, value in {**_COMMON, **PROFILES[name]}.items():
        setattr(cfg, key, value)
    cfg.profile = name
    return cfg


def detect_profile(cfg: Config) -> str:
    """Which profile cfg currently matches exactly, or "custom"."""
    for name, values in PROFILES.items():
        merged = {**_COMMON, **values}
        if all(getattr(cfg, key) == value for key, value in merged.items()):
            return name
    return "custom"


# -- model catalogue --------------------------------------------------------

# What the "Recognition model" picker shows for each TRANSCRIBERS entry.
RECOGNIZER_NOTES = {
    "Whisper large-v3-turbo": "Any language. ~1.6 GB VRAM.",
    "Kotoba-Whisper v2.0 (Japanese)": "Fastest, Japanese only. ~1.6 GB VRAM.",
    "Anime-Whisper (Japanese)": "Tuned on anime and game voice acting. ~1.6 GB VRAM.",
    "Whisper large-v3": "Strongest general model, about 2x the cost. ~3 GB VRAM.",
}

# Translation engines, as config overrides. `needs` is what has to exist
# outside this process for the engine to work.
TRANSLATORS: dict[str, dict] = {
    "Sugoi v4 (built-in)": {
        "set": {"mt_enabled": True, "task": "transcribe", "mt_backend": "ct2"},
        "needs": None,
        "note": "300M NMT model in this process. Fastest; one sentence at a "
        "time, no memory of earlier lines. ~1 GB VRAM.",
    },
    "Qwen3 4B Instruct (Ollama)": {
        "set": {
            "mt_enabled": True,
            "task": "transcribe",
            "mt_backend": "llm",
            "llm_model": QWEN3_4B,
        },
        "needs": QWEN3_4B,
        "note": "Small instruction model. Sees earlier lines, so names and "
        "dropped subjects hold. ~3.5 GB VRAM in Ollama.",
    },
    "Sugoi 14B Ultra (Ollama)": {
        "set": {
            "mt_enabled": True,
            "task": "transcribe",
            "mt_backend": "llm",
            "llm_model": SUGOI_14B,
        },
        "needs": SUGOI_14B,
        "note": "Japanese-to-English specialist, the most idiomatic output "
        "here. ~9.5 GB VRAM in Ollama.",
    },
    "Whisper large-v3 (speech to English directly)": {
        "set": {"mt_enabled": False, "task": "translate", "model": "large-v3", "beam_size": 1},
        "needs": None,
        "note": "One model does both jobs. English only - the Japanese is "
        "not kept - and weakest on names.",
    },
    "None (Japanese captions only)": {
        "set": {"mt_enabled": False, "task": "transcribe"},
        "needs": None,
        "note": "Transcription without translation.",
    },
}

CUSTOM_LLM = "Other Ollama model..."


def recognizer_name(cfg: Config) -> str:
    """Catalogue name of the recogniser cfg uses, or the raw model string."""
    for name, values in TRANSCRIBERS.items():
        if values["model"] == cfg.model:
            return name
    return cfg.model


def translator_name(cfg: Config) -> str:
    """Catalogue name of the translation engine cfg uses."""
    if cfg.task == "translate" and not cfg.mt_enabled:
        return "Whisper large-v3 (speech to English directly)"
    if not cfg.mt_enabled:
        return "None (Japanese captions only)"
    if cfg.mt_backend == "ct2":
        return "Sugoi v4 (built-in)"
    for name, entry in TRANSLATORS.items():
        if entry["set"].get("llm_model") == cfg.llm_model:
            return name
    return CUSTOM_LLM


def short_model_name(model: str) -> str:
    """A model id trimmed to something that fits a status line."""
    known = {
        KOTOBA: "Kotoba-Whisper v2.0",
        ANIME_WHISPER: "Anime-Whisper",
        "large-v3-turbo": "Whisper large-v3-turbo",
        "large-v3": "Whisper large-v3",
        SUGOI_14B: "Sugoi 14B Ultra",
        QWEN3_4B: "Qwen3 4B Instruct",
        "entai2965/sugoi-v4-ja-en-ctranslate2": "Sugoi v4",
    }
    if model in known:
        return known[model]
    return model.rsplit("/", 1)[-1]
