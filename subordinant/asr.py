"""faster-whisper wrapper producing word-level hypotheses."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cuda
from .config import (
    DECODER_LAYERS,
    MONOLINGUAL,
    SAMPLE_RATE,
    TOKENIZER_DONOR,
    Config,
)

cuda.bootstrap()  # must precede the faster_whisper import

from faster_whisper import WhisperModel  # noqa: E402

log = logging.getLogger(__name__)

# Whisper emits these on silence and music no matter what the audio contains -
# they are memorised from subtitle training data. Dropped only when one of them
# is the entire output of a pass, so genuine speech is never truncated.
_HALLUCINATIONS = {
    "thank you.",
    "thank you",
    "thanks for watching!",
    "thanks for watching.",
    "you",
    "bye.",
    "subtitles by the amara.org community",
    "subs by www.zeoranger.co.uk",
    "please subscribe!",
    "♪",
    # Japanese sources have their own set, memorised from YouTube subtitles.
    # These leak through untranslated even with task="translate", because the
    # model is not translating anything - it is reciting a caption card.
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "ご視聴ありがとうございます",
    "ご視聴ありがとうございます。",
    "最後までご視聴いただきありがとうございました",
    "最後までご視聴いただきありがとうございました。",
    "チャンネル登録お願いします",
    "チャンネル登録をお願いします",
    "おやすみなさい",
    "おやすみなさい。",
}


# Models that are weak at task="translate" - not models that refuse it. Both
# accept the task and run: the turbo checkpoints are multilingual and carry the
# <|translate|> token, they are just pruned to four decoder layers and fine-tuned
# on transcription only, so translation is unreliable and commonly comes back in
# the source language. The distil-* ones are English-only distillations, so a
# non-English source is the harder problem for them.
#
# Nothing errors in either case, which is the reason to say something here: bad
# captions otherwise look like an audio or language problem.
_WEAK_TRANSLATE = ("turbo", "distil", "kotoba", "anime")

# Models an initial prompt makes worse instead of better. Anime-Whisper's card
# is explicit that one "causes hallucinations and significantly degraded
# performance", which is the reverse of what a prompt does everywhere else
# here - biasing the decoder towards names it would otherwise guess at.
#
# Nothing errors, and the captions do not look broken; they just quietly get
# worse, which is the whole reason to say something.
_PROMPT_HOSTILE = ("anime-whisper",)

# Decoder attention heads at d_model 1280, which is what every model in
# DECODER_LAYERS is distilled from.
_ALIGNMENT_HEADS = 20


def translate_warning(cfg: Config) -> str | None:
    """Return a warning if cfg asks a weak-at-translation model to translate."""
    if cfg.task != "translate":
        return None
    name = cfg.model.lower()
    if not any(marker in name for marker in _WEAK_TRANSLATE):
        return None
    return (
        f"{cfg.model} is fine-tuned for transcription; translation is unreliable "
        "and may come back in the source language. If captions look wrong, "
        "try --model large-v3."
    )


def transcribe_warning(cfg: Config) -> str | None:
    """Return a warning if cfg points a single-language model at another one."""
    only = MONOLINGUAL.get(cfg.model)
    if only is None or cfg.language == only:
        return None
    return (
        f"{cfg.model} is trained on {only!r} only; it will still accept "
        f"language={cfg.language!r} and return confident nonsense. Switch the "
        "audio language or pick a multilingual model."
    )


def prompt_warning(cfg: Config) -> str | None:
    """Return a warning if cfg prompts a model that prompting degrades."""
    if not cfg.initial_prompt.strip():
        return None
    name = cfg.model.lower()
    if not any(marker in name for marker in _PROMPT_HOSTILE):
        return None
    return (
        f"Names go to the translator on {cfg.model}, not the transcriber - "
        "its model card reports that a decoder prompt causes hallucinations. "
        "Write readings for kanji names (天音 = Amane) to get the most of it."
    )


class ModelLoadError(RuntimeError):
    """The recognition model could not be fetched or constructed.

    The message is written for the status line: it says what is missing and
    what to do about it.
    """


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text


def _snapshot(model_id: str) -> Path | None:
    """The local directory holding model_id, downloading it if need be.

    The cache is tried first with the network forbidden. A model that is
    already installed is then found without a single request leaving the
    machine, which is both what "works offline" has to mean and a second or
    two saved on every start.
    """
    path = Path(model_id)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached; fall through to the network
        pass
    try:
        return Path(snapshot_download(model_id))
    except Exception as exc:  # noqa: BLE001 - network, auth and disk all land here
        log.warning("could not resolve %s: %s", model_id, exc)
        return None


def is_cached(model: str) -> bool:
    """Is this recognition model already on disk?"""
    if Path(model).is_dir():
        return True
    from faster_whisper.utils import download_model

    try:
        download_model(model, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - anything else means "not usable offline"
        return False


def resolve_model(model: str) -> str:
    """Local path of a recognition model, fetching it only if it is absent."""
    if Path(model).is_dir():
        return model
    from faster_whisper.utils import download_model

    try:
        return download_model(model, local_files_only=True)
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    try:
        return download_model(model)
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - network, disk, auth
        raise ModelLoadError(
            f"The recognition model {model} is not installed and could not be "
            f"downloaded ({type(exc).__name__}). Connect to the internet once "
            "so it can be fetched, or pick a model that is already installed."
        ) from exc


def repair_tokenizer(model_id: str) -> None:
    """Supply a tokenizer.json to conversions published without one.

    Without this the model decodes against whisper-tiny's vocabulary, which
    is one token shorter than the large-v3 one it was trained with, so every
    control token from 50358 up - the tasks, <|notimestamps|>, every timestamp
    - comes back shifted. Captions scramble and word timings go meaningless
    while the text tokens underneath still decode, so nothing raises.

    See TOKENIZER_DONOR in config for why the donor is the right vocabulary.
    Only the one file is fetched, not the donor's 1.5 GB of weights.
    """
    donor = TOKENIZER_DONOR.get(model_id)
    if donor is None:
        return  # not a conversion we know to be missing one

    path = _snapshot(model_id)
    if path is None:
        return
    target = path / "tokenizer.json"
    if target.exists():
        return  # already repaired, or shipped correct

    try:
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(donor, "tokenizer.json")
        shutil.copyfile(source, target)
    except Exception as exc:  # noqa: BLE001 - network, auth and disk all land here
        log.warning(
            "could not supply tokenizer.json for %s from %s: %s. Captions "
            "from this model will be scrambled - it will decode against the "
            "wrong vocabulary and nothing will raise.",
            model_id,
            donor,
            exc,
        )
        return
    log.info("supplied tokenizer.json for %s from %s", model_id, donor)


def repair_alignment_heads(model_id: str) -> None:
    """Drop alignment heads that name decoder layers the model does not have.

    Distil models are shipped with the alignment_heads of the model they were
    distilled from, which name decoder layers 7-25 on a decoder that has two.
    faster-whisper reads them straight out of config.json when word_timestamps
    is on and indexes the list of layers with them, so the mismatch is an
    out-of-bounds read inside CTranslate2: the process segfaults, with no
    traceback and nothing written to the log. Repairing the file is the only
    place to intervene, because the heads are read during construction.

    Rewrites the snapshot in place - the config is a few hundred bytes beside a
    1.5 GB model.bin, the edit is idempotent, and the model cannot be used for
    anything else in this program without it.
    """
    layers = DECODER_LAYERS.get(model_id)
    if layers is None:
        return  # not a model we know to be mis-shipped

    path = _snapshot(model_id)
    if path is None:
        return

    config = path / "config.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", config, exc)
        return

    heads = data.get("alignment_heads") or []
    if all(layer < layers for layer, _head in heads):
        return  # already repaired, or shipped correct

    # Every head of the last decoder layer. The published pairs are a curated
    # subset chosen for attending to time cleanly, and no such list exists for
    # these models; taking the whole layer is the closest available substitute,
    # and it puts word boundaries within ~40 ms of turbo's on the same audio.
    data["alignment_heads"] = [[layers - 1, h] for h in range(_ALIGNMENT_HEADS)]
    try:
        config.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        log.warning("could not repair %s: %s", config, exc)
        return
    log.info(
        "repaired alignment heads for %s: %d pairs named decoder layers past %d",
        model_id,
        len(heads),
        layers - 1,
    )


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Things that happened while loading that the user should be told:
        # a fallback to 8-bit weights, to the CPU, to a smaller model. Empty
        # when everything loaded as asked.
        self.notes: list[str] = []
        checks = (translate_warning, transcribe_warning, prompt_warning)
        for warning in (check(cfg) for check in checks):
            if warning:
                log.warning("%s", warning)

        t0 = time.perf_counter()
        last_error: BaseException | None = None
        for device, compute_type, model_id in self._attempts(cfg):
            try:
                # Both repairs must precede construction: the heads are read
                # as the model is built, and a bad set faults the process
                # instead of raising, while the tokenizer is resolved once and
                # kept for the model's lifetime.
                repair_alignment_heads(model_id)
                repair_tokenizer(model_id)
                path = resolve_model(model_id)
                self.model = WhisperModel(
                    path, device=device, compute_type=compute_type
                )
            except ModelLoadError:
                raise
            except Exception as exc:  # noqa: BLE001 - CUDA, cuDNN, memory, bad files
                last_error = exc
                log.warning(
                    "could not load %s on %s (%s): %s",
                    model_id,
                    device,
                    compute_type,
                    exc,
                )
                continue
            self.device = device
            self.compute_type = compute_type
            self.model_id = model_id
            break
        else:
            raise ModelLoadError(
                f"Could not load the recognition model {cfg.model}: {last_error}"
            )

        if self.device != "cuda" and cfg.device != "cpu":
            reason = (
                "the GPU ran out of memory"
                if last_error is not None and _is_oom(last_error)
                else "the GPU could not be used"
            )
            swapped = (
                f" with the smaller {self.model_id} model"
                if self.model_id != cfg.model
                else ""
            )
            self.notes.append(
                f"Running on the CPU{swapped} because {reason}. Expect later "
                "captions and lower accuracy; see the log for the cause."
            )
        elif self.compute_type != self._wanted_compute(cfg) and self.device == "cuda":
            self.notes.append(
                "Loaded the recognition model with 8-bit weights because the "
                "GPU did not have room for the full-precision ones."
            )
        log.info(
            "loaded %s on %s (%s) in %.1fs",
            self.model_id,
            self.device,
            self.compute_type,
            time.perf_counter() - t0,
        )

    @staticmethod
    def _wanted_compute(cfg: Config) -> str:
        if cfg.vram_saver and cfg.compute_type in ("float16", "float32", "auto"):
            return "int8_float16"
        return cfg.compute_type

    @classmethod
    def _attempts(cls, cfg: Config) -> list[tuple[str, str, str]]:
        """(device, compute type, model) to try, best first.

        On the GPU the only thing to give up is precision: the same model in
        8-bit weights needs about half the memory, which is usually the
        difference when something else already has the card. Giving up the
        GPU also means giving up the model - the ones tuned for it manage a
        pass every several seconds on a CPU - unless the user asked for the
        CPU outright, in which case their model choice stands.
        """
        attempts: list[tuple[str, str, str]] = []
        if cfg.device in ("auto", "cuda"):
            wanted = cls._wanted_compute(cfg)
            attempts.append(("cuda", wanted, cfg.model))
            if wanted != "int8_float16":
                attempts.append(("cuda", "int8_float16", cfg.model))
        if cfg.device == "cpu":
            compute = cfg.compute_type if cfg.compute_type in ("int8", "float32") else "int8"
            attempts.append(("cpu", compute, cfg.model))
        elif cfg.device == "auto":
            heavy = any(tag in cfg.model.lower() for tag in ("large", "kotoba", "anime", "turbo"))
            attempts.append(("cpu", "int8", cfg.cpu_model if heavy else cfg.model))
        return attempts

    def warmup(self) -> float:
        """Run one throwaway inference so the first caption isn't the slow one."""
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
        t0 = time.perf_counter()
        # task must match the real passes: translate runs a different decoder
        # prompt, and warming the wrong one leaves the first caption slow, which
        # is the entire thing this method exists to prevent.
        segments, _ = self.model.transcribe(
            noise,
            language=self.cfg.language,
            task=self.cfg.task,
            beam_size=1,
            # Warming the same decoder path the real passes use, prompt included
            # - a prompt changes the context length and so the cost.
            initial_prompt=self.cfg.initial_prompt or None,
        )
        list(segments)  # the generator is lazy; force it
        elapsed = time.perf_counter() - t0
        log.info("warmup pass: %.2fs", elapsed)
        return elapsed

    def _asr_prompt(self) -> str | None:
        """The Names box as a decoder prompt, where the model tolerates one.

        None rather than "" so faster-whisper skips the prompt path entirely
        when unset - and None as well on a model listed in _PROMPT_HOSTILE,
        which a prompt makes measurably worse rather than better.

        Nothing is lost by that. A name reaching the captions wrong is
        usually not the transcriber mishearing the audio - it is the
        translator picking a reading for kanji it guessed at, 天音 coming
        back as "Tenki" rather than "Amane". The box goes to the translation
        stage as a glossary either way; see LLMTranslator.
        """
        if not self.cfg.initial_prompt:
            return None
        if any(m in self.model_id.lower() for m in _PROMPT_HOSTILE):
            return None
        return self.cfg.initial_prompt

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        """Transcribe a buffer of mono 16 kHz float32 audio into timed words."""
        if audio.size == 0:
            return []

        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            task=self.cfg.task,
            beam_size=self.cfg.beam_size,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            word_timestamps=True,
            # Off when the stream is already gated by the neural VAD: the
            # same detector would score the same audio a second time, on the
            # CPU, every pass. On with the energy gate, where it is the only
            # thing standing between music and the decoder.
            vad_filter=self.cfg.vad_filter or self.cfg.vad_mode != "silero",
            initial_prompt=self._asr_prompt(),
            # Streaming re-transcribes an overlapping buffer every pass. Feeding
            # the previous text back in makes Whisper loop on its own output.
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
        )

        words: list[Word] = []
        texts: list[str] = []
        for seg in segments:
            texts.append(seg.text)
            for w in seg.words or ():
                words.append(
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        text=w.word,
                        probability=float(w.probability),
                    )
                )

        whole = "".join(texts).strip().lower()
        if whole in _HALLUCINATIONS:
            return []
        return words

    def close(self) -> None:
        """Release the model's memory now rather than at garbage collection."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.model.unload_model()
        except Exception:  # noqa: BLE001 - best effort; dropping the ref frees it too
            log.debug("unload_model failed", exc_info=True)
        self.model = None
