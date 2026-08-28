"""Machine translation stage: source-language text -> English.

Whisper can translate on its own with task="translate", but that was a training
side-objective rather than a translation system, and it shows: on Japanese it
frequently returns the source language untouched, and it mangles proper nouns.
A dedicated NMT model does the job better *and* cheaper, because of where it sits
in the loop.

Whisper re-runs on the rolling buffer every process_interval_sec. This does not.
It runs once per committed utterance - roughly once per eight passes - so its
cost amortises to a few milliseconds per pass and stays out of the real-time
budget. Measured on an RTX 4080 SUPER: ~46 ms per utterance at beam 1, ~6 ms/pass
amortised, against the ~300 ms/pass that a second Whisper pass would have cost.

The tradeoff is that a translation cannot be streamed the way transcription can.
LocalAgreement-2 commits a stable prefix, which works because ASR output is
monotonic; translation is not. Japanese is SOV and English is SVO, so a partial
Japanese sentence does not determine an English prefix. Translating whole
utterances is what avoids captions that rewrite themselves.
"""

from __future__ import annotations

import logging
import time

from . import cuda
from .config import Config

# The Translator loads cuBLAS by name exactly as the Whisper model does, so the
# nvidia wheel directories have to be registered before ctranslate2 is imported.
cuda.bootstrap()

import ctranslate2  # noqa: E402

log = logging.getLogger(__name__)


class TranslationUnavailable(RuntimeError):
    """Raised when the MT model cannot be loaded."""


def build_translator(cfg: Config):
    """The translator cfg.mt_backend asks for.

    Both backends expose exactly translate(str) -> str and raise
    TranslationUnavailable from their constructor, so the pipeline holds one
    slot and neither knows about the other. Imported lazily because the LLM
    backend reaches for the network at import time and the ct2 one must not
    depend on it being reachable.
    """
    if cfg.mt_backend == "llm":
        from .llm_translate import LLMTranslator

        return LLMTranslator(cfg)
    return Translator(cfg)


class Translator:
    """Wraps a CTranslate2 NMT model plus its SentencePiece tokenizers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import sentencepiece
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise TranslationUnavailable(
                f"missing dependency for translation: {exc}"
            ) from exc

        t0 = time.perf_counter()
        try:
            path = snapshot_download(cfg.mt_model)
        except Exception as exc:  # noqa: BLE001 - network, auth, disk all land here
            raise TranslationUnavailable(
                f"could not fetch {cfg.mt_model}: {exc}"
            ) from exc

        try:
            self._translator = ctranslate2.Translator(path, device=cfg.device)
            self._device = cfg.device
        except (RuntimeError, ValueError) as exc:
            if cfg.device != "cuda":
                raise TranslationUnavailable(str(exc)) from exc
            log.warning("MT on CPU (%s)", exc)
            self._translator = ctranslate2.Translator(path, device="cpu")
            self._device = "cpu"

        # Sugoi ships separate source and target SentencePiece models under spm/.
        self._src = sentencepiece.SentencePieceProcessor(
            f"{path}/spm/spm.{cfg.mt_source_lang}.nopretok.model"
        )
        self._tgt = sentencepiece.SentencePieceProcessor(
            f"{path}/spm/spm.{cfg.mt_target_lang}.nopretok.model"
        )
        log.info(
            "loaded MT %s on %s in %.1fs",
            cfg.mt_model,
            self._device,
            time.perf_counter() - t0,
        )

    def translate(self, text: str) -> str:
        """Translate one utterance. Returns "" for empty or failed input."""
        text = text.strip()
        if not text:
            return ""
        try:
            tokens = self._src.encode(text, out_type=str)
            if not tokens:
                return ""
            results = self._translator.translate_batch(
                source=[tokens],
                beam_size=self.cfg.mt_beam_size,
                # Utterances are single sentences. Without the repeat guards a
                # garbled one decodes into hundreds of repeated words - see the
                # notes on these keys in config.py.
                max_decoding_length=self.cfg.mt_max_decoding_length,
                repetition_penalty=self.cfg.mt_repetition_penalty,
                no_repeat_ngram_size=self.cfg.mt_no_repeat_ngram_size,
            )
            # <unk> renders as a literal box in the overlay, so drop it rather
            # than show it.
            return self._tgt.decode(results[0].hypotheses[0]).replace("<unk>", "").strip()
        except Exception:  # noqa: BLE001 - a bad utterance must not kill the pipeline
            log.exception("translation failed for %r", text[:80])
            return ""
