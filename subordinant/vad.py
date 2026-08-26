"""Cheap energy gate in front of the GPU.

This is deliberately *not* the speech/non-speech decision - that is made by the
Silero VAD bundled with faster-whisper, on the audio that gets this far. All this
does is answer "is anything playing at all", which is enough to keep the GPU idle
during digital silence and to detect the end of an utterance so the streaming
buffer can be flushed.

Music and other loud non-speech will pass this gate. That is expected; Silero
rejects it downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE

FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

# Floor for the dB conversion, so digital silence doesn't produce -inf.
_EPS = 1e-10


@dataclass
class GateState:
    active: bool  # audio present in the block just fed
    trailing_silence_sec: float  # continuous silence up to now


class EnergyGate:
    def __init__(self, threshold_db: float = -48.0):
        self.threshold_db = threshold_db
        self._tail = np.empty(0, dtype=np.float32)
        self._silence_sec = 0.0
        self._saw_audio = False

    def reset(self) -> None:
        self._tail = np.empty(0, dtype=np.float32)
        self._silence_sec = 0.0
        self._saw_audio = False

    @property
    def trailing_silence_sec(self) -> float:
        return self._silence_sec

    def update(self, chunk: np.ndarray) -> GateState:
        """Feed a mono 16 kHz block; returns whether it carried audio."""
        if chunk.size:
            self._tail = np.concatenate((self._tail, chunk)) if self._tail.size else chunk

        n_frames = self._tail.size // FRAME_LEN
        active_any = False
        if n_frames:
            frames = self._tail[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
            self._tail = self._tail[n_frames * FRAME_LEN :]

            rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
            db = 20.0 * np.log10(np.maximum(rms, _EPS))
            loud = db > self.threshold_db
            active_any = bool(loud.any())

            if active_any:
                self._saw_audio = True
                # Silence resumes only after the last loud frame in this block.
                trailing_quiet = int(n_frames - 1 - np.flatnonzero(loud)[-1])
                self._silence_sec = trailing_quiet * FRAME_MS / 1000.0
            elif self._saw_audio:
                self._silence_sec += n_frames * FRAME_MS / 1000.0

        return GateState(active=active_any, trailing_silence_sec=self._silence_sec)
