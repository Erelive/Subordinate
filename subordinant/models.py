"""Loaded models, kept between runs.

Pressing Stop and then Start, or changing the hold time and applying it,
should not cost a multi-second model load when the models have not changed.
The pipeline is rebuilt each time, cheaply; the weights live here and are
handed to whichever pipeline is current.

The cache holds at most one recogniser and one NMT model. Asking for a
different one releases the old one *first*, so switching from Kotoba to
large-v3 never has both on the GPU at once - on a 16 GB card with a 9 GB LLM
resident, that overlap is the difference between loading and failing.
"""

from __future__ import annotations

import gc
import logging
import threading

from .config import Config

log = logging.getLogger(__name__)


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._asr = None
        self._asr_key: tuple | None = None
        self._nmt = None
        self._nmt_key: tuple | None = None
        self._vad = None
        self._vad_key: tuple | None = None
        self._vad_note: str | None = None
        self._llm_seen: set[tuple[str, str]] = set()

    # -- recogniser --------------------------------------------------------

    @staticmethod
    def _asr_identity(cfg: Config) -> tuple:
        return (cfg.model, cfg.device, cfg.compute_type, cfg.vram_saver, cfg.cpu_model)

    def has_transcriber(self, cfg: Config) -> bool:
        return self._asr is not None and self._asr_key == self._asr_identity(cfg)

    def transcriber(self, cfg: Config):
        """The recogniser cfg describes, loading it if it is not the one held."""
        from .asr import Transcriber

        with self._lock:
            key = self._asr_identity(cfg)
            if self._asr is not None and self._asr_key == key:
                # Same weights; the decoding options are read from cfg on
                # every call, so pointing it at the new one is all it takes
                # to change beam width or the prompt.
                self._asr.cfg = cfg
                return self._asr, False
            self._drop_asr()
            self._asr = Transcriber(cfg)
            self._asr_key = key
            return self._asr, True

    def _drop_asr(self) -> None:
        if self._asr is not None:
            self._asr.close()
            self._asr = None
            self._asr_key = None
            gc.collect()

    # -- NMT ---------------------------------------------------------------

    @staticmethod
    def _nmt_identity(cfg: Config) -> tuple:
        return (
            cfg.mt_model,
            cfg.device,
            cfg.vram_saver,
            cfg.mt_source_lang,
            cfg.mt_target_lang,
        )

    def has_nmt(self, cfg: Config) -> bool:
        return self._nmt is not None and self._nmt_key == self._nmt_identity(cfg)

    def nmt(self, cfg: Config):
        from .translate import Translator

        with self._lock:
            key = self._nmt_identity(cfg)
            if self._nmt is not None and self._nmt_key == key:
                self._nmt.cfg = cfg
                return self._nmt, False
            self._drop_nmt()
            self._nmt = Translator(cfg)
            self._nmt_key = key
            return self._nmt, True

    def _drop_nmt(self) -> None:
        if self._nmt is not None:
            self._nmt.close()
            self._nmt = None
            self._nmt_key = None
            gc.collect()

    # -- LLM ---------------------------------------------------------------

    def llm(self, cfg: Config):
        """A translator bound to the Ollama model cfg names.

        Built fresh every time: the object is a prompt, a history and an HTTP
        session, while the weights it refers to are held by Ollama. What is
        remembered is which models have been warmed, so release() knows what
        to ask Ollama to let go of.
        """
        from .llm_translate import LLMTranslator

        translator = LLMTranslator(cfg)
        self._llm_seen.add((cfg.llm_endpoint, cfg.llm_model))
        return translator

    # -- VAD ---------------------------------------------------------------

    def vad(self, cfg: Config):
        """(detector, note). The note is set when a fallback was substituted."""
        from .vad import build_vad

        with self._lock:
            key = (cfg.vad_mode, cfg.energy_gate_db)
            if self._vad is None or self._vad_key != key:
                self._vad, self._vad_note = build_vad(cfg.vad_mode, cfg.energy_gate_db)
                self._vad_key = key
            self._vad.reset()
            return self._vad, self._vad_note

    # -- release -----------------------------------------------------------

    def release(self, include_llm: bool = True) -> None:
        """Give all model memory back. The next Start pays the load again."""
        with self._lock:
            self._drop_asr()
            self._drop_nmt()
            self._vad = None
            self._vad_key = None
            if include_llm and self._llm_seen:
                import requests

                for endpoint, model in list(self._llm_seen):
                    try:
                        requests.post(
                            f"{endpoint.rstrip('/')}/api/generate",
                            json={"model": model, "keep_alive": 0},
                            timeout=5.0,
                        )
                    except requests.RequestException as exc:
                        log.debug("could not unload %s: %s", model, exc)
                self._llm_seen.clear()
        log.info("released model memory")
