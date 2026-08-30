"""Speaker attribution: who is speaking, on the same clock as the words.

Offline diarization clusters embeddings over a whole recording, which is what
makes its labels globally consistent - "Speaker 2" means the same person on page
one and page ten. A live stream has no whole recording, so this does the online
version: embed a window of speech, compare it against a bank of running
centroids, and either match an existing speaker or open a new one.

The consequence is worth stating plainly, because it shows up in use: labels
drift. Someone who stops talking for a few minutes and comes back may not match
their old centroid closely enough, and gets a fresh number. That is inherent to
deciding without the future, not a bug to be fixed here.

What it does buy is the thing that matters most for a caption: knowing *that*
the voice changed. A speaker change is a hard boundary between utterances, so
the pipeline can cut there and stop merging two people's speech into a single
translation unit.

Window and hop
--------------
These are separate on purpose. Embedding quality depends almost entirely on how
much speech goes into one embedding - measured on two voices, the *lowest*
same-speaker similarity was 0.24 at a 1 s window, 0.61 at 1.5 s and 0.85 at 3 s,
against a between-speaker maximum that barely moved from 0.21. A short window
therefore makes one person look like two, which is the over-splitting failure.

But the window is also the resolution at which a change of voice can be seen, and
a 3 s resolution is far too coarse to cut a caption on. So the window is long (3 s
of trailing context, for the embedding) while the hop is short (1 s, the span each
label actually covers). Every hop embeds the last `window` seconds and labels only
the most recent `hop` of it.

Clock alignment
---------------
speaker_for() is queried with Word timestamps, which are absolute stream time as
StreamingTranscriber counts it: the total duration of audio handed to
insert_audio(). This tracker therefore has to be fed *exactly* those same blocks,
at the same points - pre-roll replay included. Feeding it raw capture blocks
instead would advance its clock during the silence the gate drops, and every
attribution would be wrong by the accumulated difference.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import cuda
from .config import SAMPLE_RATE, Config

log = logging.getLogger(__name__)

# The embedding models are published as individual ONNX files in one repo, so
# this fetches a single file. snapshot_download would pull all twenty-one.
MODEL_REPO = "csukuangfj/speaker-embedding-models"

# No speaker could be determined for a span: too quiet, or before enough audio
# existed to embed. Callers show the text unlabelled rather than guessing.
UNKNOWN = -1


class DiarizationUnavailable(RuntimeError):
    """Raised when the speaker embedding model cannot be loaded."""


@dataclass
class Segment:
    """One hop of stream time, labelled."""

    start: float
    end: float
    speaker: int


class SpeakerTracker:
    """Online speaker labelling over a bank of running centroids."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Must happen before sherpa_onnx is imported; see cuda.preload_onnxruntime.
        cuda.preload_onnxruntime()

        try:
            import sherpa_onnx
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise DiarizationUnavailable(
                f"missing dependency for speaker labelling: {exc}"
            ) from exc

        try:
            path = hf_hub_download(MODEL_REPO, cfg.diarize_model)
        except Exception as exc:  # noqa: BLE001 - network, auth, disk all land here
            raise DiarizationUnavailable(
                f"could not fetch {cfg.diarize_model}: {exc}"
            ) from exc

        try:
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=path,
                    # One thread is plenty at ~33 ms per 3 s window, and this runs
                    # on the pipeline worker, where extra threads would contend
                    # with the loop feeding the GPU.
                    num_threads=1,
                    provider="cpu",
                )
            )
        except Exception as exc:  # noqa: BLE001 - a bad model file lands here
            raise DiarizationUnavailable(f"could not load {path}: {exc}") from exc

        self._window_sec = float(cfg.diarize_window_sec)
        self._hop_sec = float(cfg.diarize_hop_sec)
        # A window shorter than the minimum embed length can never produce a
        # long enough chunk, so every window would be rejected and the feature
        # would silently do nothing at all - no labels, no error, no clue.
        # Shortening a window in config.json is exactly how someone would hit
        # that, so clamp instead of failing.
        self._min_embed_sec = min(cfg.diarize_min_embed_sec, self._window_sec)
        if self._min_embed_sec < cfg.diarize_min_embed_sec:
            log.warning(
                "diarize_min_embed_sec %.1fs exceeds diarize_window_sec %.1fs; "
                "using %.1fs. Short windows split one voice into several.",
                cfg.diarize_min_embed_sec,
                self._window_sec,
                self._min_embed_sec,
            )
        # Audio still needed for a future window, and the absolute stream time
        # its first sample sits at.
        self._audio = np.empty(0, dtype=np.float32)
        self._origin = 0.0
        # Absolute stream time of the end of everything fed so far.
        self._time = 0.0
        # End of the next hop to label.
        self._next_end = self._hop_sec

        # Unit-norm centroids, and how many windows each is the mean of.
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        # An embedding that matched nobody, with the span it came from, waiting
        # to be confirmed. See _classify for why one window is not enough.
        self._provisional: np.ndarray | None = None
        self._provisional_span: tuple[float, float] | None = None
        self.timeline: deque[Segment] = deque()

        log.info(
            "loaded speaker model %s (dim %d, window %.1fs hop %.1fs)",
            cfg.diarize_model,
            self._extractor.dim,
            self._window_sec,
            self._hop_sec,
        )

    # -- input -------------------------------------------------------------

    def feed(self, audio: np.ndarray) -> None:
        """Take a block of audio. Must be the same block given to insert_audio."""
        if audio.size:
            self._audio = np.concatenate((self._audio, audio))
            self._time += audio.size / SAMPLE_RATE

    def process(self) -> list[Segment]:
        """Label every hop that has completed. Cheap; call often."""
        produced: list[Segment] = []
        while self._time >= self._next_end:
            end = self._next_end
            # The label covers one hop; the embedding sees up to `window` of
            # trailing context, or everything there is if the stream is younger.
            win_start = max(self._origin, end - self._window_sec)
            i0 = int(round((win_start - self._origin) * SAMPLE_RATE))
            i1 = int(round((end - self._origin) * SAMPLE_RATE))
            chunk = self._audio[i0:i1]
            span = (max(0.0, end - self._hop_sec), end)

            speaker = self._classify(chunk, span)
            if speaker != UNKNOWN:
                self.timeline.append(Segment(span[0], span[1], speaker))
                produced.append(self.timeline[-1])

            self._next_end += self._hop_sec

        self._trim()
        return produced

    def _trim(self) -> None:
        """Drop audio no future window can reach."""
        keep_from = max(self._origin, self._next_end - self._window_sec - self._hop_sec)
        drop = int(round((keep_from - self._origin) * SAMPLE_RATE))
        if drop > 0:
            self._audio = self._audio[drop:]
            self._origin = keep_from

    # -- labelling ---------------------------------------------------------

    def _embed(self, chunk: np.ndarray) -> np.ndarray | None:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=chunk)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        # Cosine similarity is a plain dot product once both sides are unit
        # length, which is why centroids are kept normalised too.
        return vec / norm if norm else None

    def _classify(self, chunk: np.ndarray, span: tuple[float, float]) -> int:
        # Too little audio to identify anyone. Embeddings from very short spans
        # are close to worthless - same-speaker similarity at 1 s was no better
        # than between-speaker - and admitting them is what invents speakers.
        if chunk.size < int(self._min_embed_sec * SAMPLE_RATE):
            return UNKNOWN

        # Near-silence and room tone produce embeddings that are not about a
        # voice at all. Letting them into the bank drags centroids together and
        # merges speakers that were previously distinct.
        if float(np.sqrt(np.mean(np.square(chunk)))) < self.cfg.diarize_min_rms:
            return UNKNOWN

        emb = self._embed(chunk)
        if emb is None:
            return UNKNOWN

        if self._centroids:
            sims = np.fromiter(
                (float(emb @ c) for c in self._centroids), dtype=np.float32
            )
            best = int(sims.argmax())
            # Past the cap, force the best match rather than opening a new
            # speaker. This is the lever that matters when the count is known:
            # set diarize_max_speakers to how many people are actually talking
            # and one person in two moods can no longer become two speakers.
            at_cap = len(self._centroids) >= self.cfg.diarize_max_speakers
            if sims[best] >= self.cfg.diarize_threshold or at_cap:
                self._update(best, emb)
                self._provisional = None
                self._provisional_span = None
                return best

        # Matched nobody. One window is not enough to open a speaker on, because
        # the window spanning a hand-off contains two voices and matches neither
        # - and left to itself it became a permanent extra identity sitting
        # between two real people. Requiring a second, agreeing window discards
        # those.
        if self._provisional is not None and (
            float(emb @ self._provisional) >= self.cfg.diarize_threshold
        ):
            # Seeded from the confirming window alone. The one being confirmed
            # may itself be a blend of two voices, which is what made it fail to
            # match in the first place; averaging it in would carry that into
            # the new centroid.
            self._centroids.append(emb)
            self._counts.append(1)
            speaker = len(self._centroids) - 1

            # Backfill the window that was held as provisional. It was this
            # speaker all along, and without this the first thing anyone says
            # goes out unlabelled - which is exactly what happened to the
            # opening lines of a stream.
            if self._provisional_span is not None:
                start, end = self._provisional_span
                self.timeline.append(Segment(start, end, speaker))
            self._provisional = None
            self._provisional_span = None
            return speaker

        self._provisional = emb
        self._provisional_span = span
        return UNKNOWN

    def _update(self, index: int, emb: np.ndarray) -> None:
        """Fold an embedding into a centroid as a running mean."""
        n = self._counts[index]
        merged = (self._centroids[index] * n + emb) / (n + 1)
        norm = float(np.linalg.norm(merged))
        if norm:
            self._centroids[index] = merged / norm
        self._counts[index] = n + 1

    # -- queries -----------------------------------------------------------

    def speaker_for(self, start: float, end: float) -> int:
        """Who dominates the span, by overlap duration. UNKNOWN if unresolvable."""
        if end <= start:
            end = start + 1e-3
        totals: dict[int, float] = {}
        for seg in self.timeline:
            if seg.end <= start:
                continue
            if seg.start >= end:
                break  # the timeline is append-only, and therefore sorted
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > 0:
                totals[seg.speaker] = totals.get(seg.speaker, 0.0) + overlap
        if totals:
            return max(totals, key=lambda k: totals[k])

        # Nothing covers this span. Rather than give up - which shows the line
        # with no speaker at all - take the nearest label within one window, so
        # words in a brief unlabelled gap join the turn around them.
        return self._nearest(start, end)

    def _nearest(self, start: float, end: float) -> int:
        best, best_gap = UNKNOWN, self._window_sec
        for seg in self.timeline:
            gap = start - seg.end if seg.end <= start else seg.start - end
            if 0 <= gap <= best_gap:
                best, best_gap = seg.speaker, gap
        return best

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def prune(self, before: float) -> None:
        """Drop timeline entries older than `before`. Centroids are kept.

        The bank is what lets a returning speaker keep their number, so it
        deliberately outlives the timeline.
        """
        while self.timeline and self.timeline[0].end < before:
            self.timeline.popleft()


def speaker_label(speaker: int) -> str:
    """Display name for a speaker id."""
    return "" if speaker == UNKNOWN else f"Speaker {speaker + 1}"
