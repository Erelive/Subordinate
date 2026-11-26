import unittest

import numpy as np

from subordinant.config import SAMPLE_RATE
from subordinant.vad import (
    AUDIO,
    CLOSE,
    PAUSE,
    VAD_FRAME,
    VAD_FRAME_SEC,
    EnergyVAD,
    SpeechGate,
)


def tone(seconds: float, level: float = 0.1) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    signal = np.full(n, level, dtype=np.float32)
    signal[1::2] *= -1
    return signal


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def gate(**kwargs) -> SpeechGate:
    return SpeechGate(EnergyVAD(-48.0), **kwargs)


def emitted(events) -> float:
    """Seconds of audio in a list of gate events."""
    return sum(e.audio.size for e in events if e.kind == AUDIO) / SAMPLE_RATE


class Gate(unittest.TestCase):
    def test_silence_emits_nothing(self):
        g = gate()
        self.assertEqual(g.feed(quiet(2.0)), [])
        self.assertFalse(g.in_utterance)
        self.assertEqual(g.stream_time, 0.0)

    def test_speech_opens_with_preroll_and_closes_after_silence(self):
        g = gate(endpoint_silence_sec=0.4, preroll_sec=0.3)
        events = g.feed(np.concatenate((quiet(1.0), tone(1.0))))
        self.assertTrue(g.in_utterance)
        # A second of speech plus the pre-roll that preceded it.
        self.assertAlmostEqual(emitted(events), 1.3, delta=0.07)

        events = g.feed(quiet(1.0))
        self.assertEqual([e.kind for e in events if e.kind != AUDIO], [PAUSE, CLOSE])
        self.assertFalse(g.in_utterance)
        # The trailing quiet up to the endpoint was passed through, the rest
        # was not.
        self.assertAlmostEqual(emitted(events), 0.4, delta=0.07)
        self.assertGreaterEqual(g.silence_sec, 0.9)

    def test_short_pause_is_recorded_on_the_stream_clock(self):
        g = gate(endpoint_silence_sec=0.6, soft_pause_sec=0.16, preroll_sec=0.0)
        g.feed(tone(1.0))
        g.feed(quiet(0.3))
        self.assertIsNotNone(g.open_pause)
        g.feed(tone(1.0))
        self.assertTrue(g.in_utterance)
        self.assertEqual(len(g.pauses), 1)
        pause = g.pauses[0]
        self.assertAlmostEqual(pause.start, 1.0, delta=0.07)
        self.assertAlmostEqual(pause.end, 1.3, delta=0.07)
        self.assertIsNone(g.open_pause)
        # Everything was passed through, pause included, so the stream stays
        # contiguous and word timestamps keep their meaning.
        self.assertAlmostEqual(g.stream_time, 2.3, delta=0.07)

    def test_a_glitch_is_not_a_pause(self):
        g = gate(endpoint_silence_sec=0.6, soft_pause_sec=0.16, preroll_sec=0.0)
        g.feed(tone(1.0))
        g.feed(quiet(0.06))
        g.feed(tone(1.0))
        self.assertEqual(len(g.pauses), 0)

    def test_block_size_does_not_change_the_answer(self):
        audio = np.concatenate((quiet(0.5), tone(1.2), quiet(0.3), tone(0.8), quiet(1.0)))
        results = []
        for block in (160, 320, 341, 4000, audio.size):
            g = gate(endpoint_silence_sec=0.45)
            kinds, total = [], 0
            for i in range(0, audio.size, block):
                for event in g.feed(audio[i : i + block]):
                    if event.kind == AUDIO:
                        total += event.audio.size
                    else:
                        kinds.append(event.kind)
            results.append((tuple(kinds), total, len(g.pauses)))
        self.assertEqual(len(set(results)), 1, results)

    def test_frames_are_whole(self):
        g = gate(preroll_sec=0.0)
        events = g.feed(tone(1.0))
        for event in events:
            if event.kind == AUDIO:
                self.assertEqual(event.audio.size % VAD_FRAME, 0)

    def test_device_going_idle_closes_the_utterance(self):
        # Loopback delivers no samples at all while nothing is rendering. The
        # utterance must still end.
        g = gate(endpoint_silence_sec=0.4)
        g.feed(tone(1.0))
        self.assertTrue(g.in_utterance)
        self.assertEqual(g.feed_silence(0.2), [])
        self.assertTrue(g.in_utterance)
        events = g.feed_silence(0.3)
        self.assertEqual([e.kind for e in events], [CLOSE])
        self.assertFalse(g.in_utterance)
        # No audio existed in that gap, so the stream clock did not move.
        self.assertAlmostEqual(g.stream_time, 1.0, delta=0.07)
        self.assertEqual(len(g.pauses), 1)

    def test_silence_counter_keeps_growing_after_close(self):
        g = gate(endpoint_silence_sec=0.4)
        g.feed(tone(1.0))
        g.feed(quiet(0.5))
        before = g.silence_sec
        g.feed(quiet(1.0))
        self.assertAlmostEqual(g.silence_sec - before, 1.0, delta=VAD_FRAME_SEC * 2)
        g.feed_silence(2.0)
        self.assertGreater(g.silence_sec, before + 2.9)

    def test_utterance_can_be_closed_before_the_endpoint(self):
        g = gate(endpoint_silence_sec=0.6, preroll_sec=0.0)
        g.feed(tone(1.0))
        g.feed(quiet(0.25))
        self.assertTrue(g.in_utterance)
        self.assertAlmostEqual(g.trailing_quiet_sec, 0.25, delta=0.07)
        self.assertTrue(g.end_utterance())
        self.assertFalse(g.in_utterance)
        self.assertFalse(g.end_utterance())  # nothing left to close
        # The pause that ended it is on record like any other.
        self.assertEqual(len(g.pauses), 1)
        self.assertAlmostEqual(g.pauses[0].start, 1.0, delta=0.07)
        # More quiet is just quiet: nothing is emitted and nothing closes twice.
        self.assertEqual(g.feed(quiet(1.0)), [])

    def test_voiced_total_moves_only_while_someone_speaks(self):
        g = gate(endpoint_silence_sec=0.6)
        g.feed(tone(16 * VAD_FRAME_SEC))  # whole frames, so none is half quiet
        spoken = g.voiced_total
        self.assertEqual(spoken, 16)
        g.feed(quiet(10 * VAD_FRAME_SEC))
        self.assertEqual(g.voiced_total, spoken)
        g.feed(tone(6 * VAD_FRAME_SEC))
        self.assertEqual(g.voiced_total, spoken + 6)

    def test_quiet_before_an_early_close_leads_in_the_next_utterance(self):
        # Closed 0.25 s into a pause; the speaker starts again at once. The
        # quiet before the close is the only lead-in the next word has.
        g = gate(endpoint_silence_sec=0.6, preroll_sec=0.3)
        g.feed(tone(1.0))
        g.feed(quiet(0.25))
        g.end_utterance()
        events = g.feed(tone(0.5))
        self.assertTrue(g.in_utterance)
        self.assertGreater(emitted(events), 0.5 + 0.15)

    def test_reset(self):
        g = gate()
        g.feed(tone(1.0))
        g.reset()
        self.assertFalse(g.in_utterance)
        self.assertIsNone(g.open_pause)

    def test_prune(self):
        g = gate(endpoint_silence_sec=0.6, preroll_sec=0.0)
        for _ in range(3):
            g.feed(tone(0.5))
            g.feed(quiet(0.3))
        self.assertEqual(len(g.pauses), 2)
        g.prune(g.pauses[1].end - 0.01)
        self.assertEqual(len(g.pauses), 1)


class Silero(unittest.TestCase):
    """The real model, on synthetic input. Checks the streaming wrapper, not
    the model's judgement: state carried across calls must give the same
    probabilities as one call over everything."""

    def setUp(self):
        from subordinant.vad import SileroVAD, VadUnavailable

        try:
            self.vad = SileroVAD()
        except VadUnavailable as exc:
            self.skipTest(str(exc))

    def test_chunking_does_not_change_probabilities(self):
        rng = np.random.default_rng(0)
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        # Something voiced-ish: a harmonic stack with a moving pitch.
        f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        signal = sum(np.sin(k * phase) / k for k in range(1, 8)) * 0.1
        signal = (signal + rng.standard_normal(t.size) * 0.002).astype(np.float32)
        frames = signal[: (signal.size // VAD_FRAME) * VAD_FRAME].reshape(-1, VAD_FRAME)

        self.vad.reset()
        whole = self.vad.probabilities(frames)
        self.vad.reset()
        parts = np.concatenate(
            [self.vad.probabilities(frames[i : i + 7]) for i in range(0, len(frames), 7)]
        )
        np.testing.assert_allclose(whole, parts, atol=1e-4)

    def test_digital_silence_is_not_speech(self):
        self.vad.reset()
        probs = self.vad.probabilities(np.zeros((40, VAD_FRAME), dtype=np.float32))
        self.assertLess(float(probs.max()), 0.3)


if __name__ == "__main__":
    unittest.main()
