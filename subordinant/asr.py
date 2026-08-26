"""faster-whisper wrapper producing word-level hypotheses."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from . import cuda
from .config import SAMPLE_RATE, Config

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
}


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
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
        segments, _ = self.model.transcribe(noise, language=self.cfg.language, beam_size=1)
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
