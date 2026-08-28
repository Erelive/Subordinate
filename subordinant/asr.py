"""faster-whisper wrapper producing word-level hypotheses."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cuda
from .config import DECODER_LAYERS, MONOLINGUAL, SAMPLE_RATE, Config

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
_WEAK_TRANSLATE = ("turbo", "distil", "kotoba")

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

    path = Path(model_id)
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_id))
        except Exception as exc:  # noqa: BLE001 - network, auth and disk all land here
            log.warning("could not resolve %s to repair it: %s", model_id, exc)
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
        for warning in (translate_warning(cfg), transcribe_warning(cfg)):
            if warning:
                log.warning("%s", warning)
        # Must precede construction: the heads are read as the model is built,
        # and a bad set faults the process instead of raising.
        repair_alignment_heads(cfg.model)
        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(
                cfg.model, device=cfg.device, compute_type=cfg.compute_type
            )
            self.device = cfg.device
        except (RuntimeError, ValueError) as exc:
            if cfg.device != "cuda":
                raise
            log.warning("CUDA unavailable (%s); falling back to CPU int8", exc)
            self.model = WhisperModel(cfg.model, device="cpu", compute_type="int8")
            self.device = "cpu"
        log.info(
            "loaded %s on %s in %.1fs", cfg.model, self.device, time.perf_counter() - t0
        )

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

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        """Transcribe a buffer of mono 16 kHz float32 audio into timed words."""
        if audio.size == 0:
            return []

        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            task=self.cfg.task,
            beam_size=self.cfg.beam_size,
            word_timestamps=True,
            vad_filter=self.cfg.vad_filter,
            # Names and terms this stream is going to use. None rather than ""
            # so faster-whisper skips the prompt path entirely when unset.
            initial_prompt=self.cfg.initial_prompt or None,
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
