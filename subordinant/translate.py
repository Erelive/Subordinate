"""Machine translation stage: Japanese text -> English, with a dedicated NMT model.

Whisper can translate on its own with task="translate", but that was a training
side-objective rather than a translation system, and it shows: on Japanese it
frequently returns the source language untouched, and it mangles proper nouns.
A dedicated NMT model does the job better *and* cheaper, because of where it sits
in the loop.

Whisper re-runs on the rolling buffer every process_interval_sec. This does not.
It runs once per finished utterance, plus a narrow-beam preview of the sentence
in progress, so its cost stays out of the recogniser's real-time budget.

This model has two jobs in the app, and they pull in different directions:

  - As the *final* translator (the Fast profile) it wants a wide beam and
    whole sentences.
  - As the *preview* translator (every profile) it wants to be quick, and its
    output is replaced within a second or two - by its own wide-beam pass, or
    by an LLM's line on the profiles that use one.

Hence translate() takes the beam as an argument rather than reading one
setting: the same loaded weights serve both.

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
from .segment import has_content, split_sentences

# The Translator loads cuBLAS by name exactly as the Whisper model does, so the
# nvidia wheel directories have to be registered before ctranslate2 is imported.
cuda.bootstrap()

import ctranslate2  # noqa: E402

log = logging.getLogger(__name__)


class TranslationUnavailable(RuntimeError):
    """Raised when a translation engine cannot be loaded.

    The message is shown to the user as-is, so it names what is missing and,
    where there is one, the command or setting that fixes it.
    """


def build_translator(cfg: Config):
    """The translator cfg.mt_backend asks for.

    Both backends expose translate(str, on_partial=None) -> str, reset(), and
    a last_error attribute, and raise TranslationUnavailable from their
    constructor, so the pipeline holds one slot and neither knows about the
    other. Imported lazily because the LLM backend reaches for the network at
    construction and the ct2 one must not depend on it being reachable.
    """
    if cfg.mt_backend == "llm":
        from .llm_translate import LLMTranslator

        return LLMTranslator(cfg)
    return Translator(cfg)


def _snapshot(repo: str) -> str:
    """Local directory for a Hub repo, touching the network only if needed."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    try:
        return snapshot_download(repo)
    except Exception as exc:  # noqa: BLE001 - network, auth, disk all land here
        raise TranslationUnavailable(
            f"The translation model {repo} is not installed and could not be "
            f"downloaded ({type(exc).__name__}). Connect to the internet once "
            "so it can be fetched."
        ) from exc


def is_cached(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class Translator:
    """Wraps a CTranslate2 NMT model plus its SentencePiece tokenizers."""

    name = "nmt"
    # This model sees one sentence and nothing else. The pipeline uses the
    # flag to decide whether keeping conversational context is worth anything.
    uses_context = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_error: str | None = None
        try:
            import sentencepiece
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise TranslationUnavailable(
                f"missing dependency for translation: {exc}"
            ) from exc

        t0 = time.perf_counter()
        path = _snapshot(cfg.mt_model)

        # 8-bit weights when memory is tight. Measured on Sugoi v4 the speed
        # is unchanged either way - a 300M model on a batch of one is bound by
        # sequential decode steps, not arithmetic - so this is purely a VRAM
        # choice.
        compute = "int8_float16" if cfg.vram_saver else "default"
        want_cuda = cfg.device in ("auto", "cuda")
        self._translator = None
        errors: list[str] = []
        for device, ctype in (
            [("cuda", compute), ("cuda", "int8")] if want_cuda else []
        ) + ([("cpu", "int8")] if cfg.device != "cuda" else []):
            try:
                self._translator = ctranslate2.Translator(
                    path, device=device, compute_type=ctype
                )
                self.device = device
                break
            except (RuntimeError, ValueError) as exc:
                errors.append(f"{device}/{ctype}: {exc}")
                log.warning("MT load failed on %s (%s): %s", device, ctype, exc)
        if self._translator is None:
            raise TranslationUnavailable(
                "Could not load the translation model: " + "; ".join(errors)
            )

        # Sugoi ships separate source and target SentencePiece models under spm/.
        try:
            self._src = sentencepiece.SentencePieceProcessor(
                f"{path}/spm/spm.{cfg.mt_source_lang}.nopretok.model"
            )
            self._tgt = sentencepiece.SentencePieceProcessor(
                f"{path}/spm/spm.{cfg.mt_target_lang}.nopretok.model"
            )
        except Exception as exc:  # noqa: BLE001 - missing or corrupt tokenizer files
            raise TranslationUnavailable(
                f"The translation model in {path} is incomplete ({exc}). Delete "
                "that folder and start again to re-download it."
            ) from exc
        log.info(
            "loaded MT %s on %s in %.1fs",
            cfg.mt_model,
            self.device,
            time.perf_counter() - t0,
        )

    def warmup(self) -> None:
        """One throwaway call, so the first real line is not the slow one."""
        self.translate("これはテストです。", beam_size=1)

    def reset(self) -> None:
        """Nothing to forget: the model has no memory between calls."""

    def close(self) -> None:
        translator = self._translator
        self._translator = None
        if translator is not None:
            try:
                translator.unload_model()
            except Exception:  # noqa: BLE001 - dropping the reference frees it too
                log.debug("unload_model failed", exc_info=True)

    def translate(self, text: str, on_partial=None, beam_size: int | None = None) -> str:
        """Translate one utterance. Returns "" for empty or failed input.

        on_partial is accepted and ignored. This backend produces a whole
        line in tens of milliseconds, so there is nothing to stream; the
        argument exists so both backends answer the same call.

        An utterance holding several sentences is split and translated as a
        batch. The model was trained on single sentences and, handed three,
        tends to translate the first and summarise or drop the rest.
        """
        text = text.strip()
        if not text or self._translator is None:
            return ""
        try:
            sentences = [s for s in split_sentences(text) if has_content(s)] or [text]
            batches = [self._src.encode(s, out_type=str) for s in sentences]
            batches = [tokens for tokens in batches if tokens]
            if not batches:
                return ""
            results = self._translator.translate_batch(
                source=batches,
                beam_size=beam_size or self.cfg.mt_beam_size,
                # Without the repeat guards a garbled utterance decodes into
                # hundreds of repeated words - see the notes on these keys in
                # config.py.
                max_decoding_length=self.cfg.mt_max_decoding_length,
                repetition_penalty=self.cfg.mt_repetition_penalty,
                no_repeat_ngram_size=self.cfg.mt_no_repeat_ngram_size,
            )
            pieces = [
                # <unk> renders as a literal box in the overlay, so drop it
                # rather than show it.
                self._tgt.decode(r.hypotheses[0]).replace("<unk>", "").strip()
                for r in results
            ]
            self.last_error = None
            return " ".join(p for p in pieces if p)
        except Exception as exc:  # noqa: BLE001 - a bad utterance must not kill the pipeline
            log.exception("translation failed for %r", text[:80])
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""
