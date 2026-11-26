import unittest

import numpy as np

from subordinant.asr import Word
from subordinant.config import SAMPLE_RATE, Config
from subordinant.streaming import HypothesisBuffer, StreamingTranscriber, join_words


def ws(*texts: str, at: float = 0.0, each: float = 0.3) -> list[Word]:
    return [Word(at + i * each, at + (i + 1) * each, t, 0.9) for i, t in enumerate(texts)]


def text(words) -> str:
    return join_words(words)


class LocalAgreement(unittest.TestCase):
    def test_nothing_commits_on_a_single_sighting(self):
        h = HypothesisBuffer()
        h.insert(ws("今日", "は"), 0.0)
        self.assertEqual(h.flush(), [])
        self.assertEqual(text(h.unstable), "今日は")

    def test_agreed_prefix_commits_and_the_rest_waits(self):
        h = HypothesisBuffer()
        h.insert(ws("今日", "は", "雨"), 0.0)
        h.flush()
        h.insert(ws("今日", "は", "晴れ", "です"), 0.0)
        self.assertEqual(text(h.flush()), "今日は")
        self.assertEqual(text(h.unstable), "晴れです")
        self.assertEqual(text(h.committed), "今日は")

    def test_one_dissenting_pass_changes_nothing(self):
        h = HypothesisBuffer()
        for hypothesis in (
            ("今日", "は"),
            ("今日", "は", "いい"),
            ("昨日", "は", "いい", "天気"),  # the recogniser changes its mind, once
        ):
            h.insert(ws(*hypothesis), 0.0)
            h.flush()
        self.assertEqual(text(h.pending), "今日は")

    def test_two_passes_agreeing_against_settled_text_replace_it(self):
        # The opening half second is misheard, twice, and settles. Every later
        # pass reads it correctly. The later reading has to win: under the
        # rule that settled text never changes, this line was wrong for good.
        h = HypothesisBuffer()
        for hypothesis in (
            ("話", "で"),
            ("話", "で", "十分"),
            ("住民", "に", "よる", "と"),
            ("住民", "に", "よる", "と", "過去"),
        ):
            h.insert(ws(*hypothesis), 0.0)
            h.flush()
        self.assertEqual(text(h.pending), "住民によると")
        self.assertEqual(h.revisions, 1)

    def test_revision_with_shifted_timing_leaves_nothing_behind(self):
        # The corrected reading starts a syllable later by the clock than the
        # mistaken one did. It is the same audio; none of the old text stays.
        h = HypothesisBuffer()
        wrong = ws("カ", "ナダ", "学", each=0.15)
        h.insert(wrong, 0.0)
        h.flush()
        h.insert(wrong, 0.0)
        h.flush()
        right = ws("この", "大学", "アメリカ", at=0.1, each=0.15)
        h.insert(right, 0.0)
        h.flush()
        h.insert(right, 0.0)
        h.flush()
        self.assertEqual(text(h.pending), "この大学アメリカ")

    def test_text_that_was_handed_on_is_never_revised(self):
        h = HypothesisBuffer()
        first = ws("話", "で", "十分")
        h.insert(first, 0.0)
        h.flush()
        h.insert(first, 0.0)
        h.flush()
        self.assertEqual(text(h.take(3)), "話で十分")
        for hypothesis in (
            ("住民", "に", "よる", "と"),
            ("住民", "に", "よる", "と", "過去"),
        ):
            h.insert(ws(*hypothesis), 0.0)
            h.flush()
        self.assertEqual(text(h.sent), "話で十分")

    def test_words_at_the_edge_of_the_buffer_are_held_back(self):
        # Both passes agree on 天気, but the audio ends 0.1 s after it: the
        # word may be half of something longer.
        h = HypothesisBuffer()
        words = ws("今日", "は", "天気")  # 0.0-0.9
        h.insert(words, 0.0)
        h.flush(settled_before=1.0 - 0.4)
        h.insert(words, 0.0)
        self.assertEqual(text(h.flush(settled_before=1.0 - 0.4)), "今日は")
        self.assertEqual(text(h.unstable), "天気")
        # Once the audio has moved on, it settles.
        h.insert(words, 0.0)
        self.assertEqual(text(h.flush(settled_before=1.5 - 0.4)), "天気")

    def test_punctuation_does_not_block_agreement(self):
        # The same word arrives as します on one pass and します。 on the next.
        h = HypothesisBuffer()
        h.insert(ws("勉強", "します"), 0.0)
        h.flush()
        h.insert(ws("勉強", "します。", "次"), 0.0)
        committed = h.flush()
        # ...and the punctuated form is the one that is kept.
        self.assertEqual(text(committed), "勉強します。")

    def test_force_commit_takes_the_tail(self):
        h = HypothesisBuffer()
        h.insert(ws("そう", "です", "ね"), 0.0)
        h.flush()
        self.assertEqual(text(h.force_commit()), "そうですね")
        self.assertEqual(h.unstable, [])
        self.assertEqual(text(h.pending), "そうですね")

    def test_the_last_pass_is_taken_whole_when_the_utterance_ends(self):
        # Settled piecemeal as 元気すぎて; the pass that heard the whole
        # sentence says 研究しすぎて. Nothing has been handed on yet, so the
        # line that is kept is the one read with everything in view.
        h = HypothesisBuffer()
        early = ws("元気", "すぎて")
        h.insert(early, 0.0)
        h.flush()
        h.insert(early, 0.0)
        h.flush()
        self.assertEqual(text(h.pending), "元気すぎて")
        h.insert(ws("研究", "し", "すぎて", "普通", "の", "人"), 0.0)
        h.flush()
        h.settle_all()
        self.assertEqual(text(h.pending), "研究しすぎて普通の人")

    def test_a_last_pass_that_dropped_speech_is_not_trusted(self):
        h = HypothesisBuffer()
        said = ws("今日", "は", "友達", "と", "東京", "に", "行き", "ました")
        h.insert(said, 0.0)
        h.flush()
        h.insert(said, 0.0)
        h.flush()
        h.insert(ws("ました"), 0.0)  # the final pass lost most of it
        h.flush()
        h.settle_all()
        self.assertEqual(text(h.pending), "今日は友達と東京に行きました")

    def test_unspoken_tail_can_be_dropped(self):
        h = HypothesisBuffer()
        h.insert(ws("思い出せ", "ない", "みたい", "バイバイ", each=0.5), 0.0)
        h.flush()
        h.settle_all()
        # The speaker stopped at 1.5 s; バイバイ is placed after that.
        self.assertEqual(text(h.drop_tail(1.5 + 0.0)), "バイバイ")
        self.assertEqual(text(h.pending), "思い出せないみたい")

    def test_repeated_head_is_dropped(self):
        h = HypothesisBuffer()
        h.insert(ws("ありがとう", "ございます"), 0.0)
        h.flush()
        h.insert(ws("ありがとう", "ございます", "では"), 0.0)
        h.flush()
        # The next pass, on a trimmed buffer, re-emits the last committed word.
        h.insert(ws("ございます", "では", "また"), 0.6)
        h.flush()
        self.assertEqual(text(h.committed).count("ございます"), 1)

    def test_a_shared_particle_is_not_a_repeat(self):
        h = HypothesisBuffer()
        h.insert(ws("行っ", "た"), 0.0)
        h.flush()
        h.insert(ws("行っ", "た", "た"), 0.0)  # e.g. 行った、たくさん
        h.flush()
        h.insert(ws("た", "くさん"), 0.6)
        h.flush()
        h.insert(ws("た", "くさん", "ある"), 0.6)
        h.flush()
        self.assertEqual(text(h.committed), "行ったたくさん")


class Restated(unittest.TestCase):
    """Re-transcribed text is matched to what is settled by content.

    These are the cases the timestamp filter got wrong on Japanese, each
    taken from a real benchmark transcript.
    """

    def commit(self, h: HypothesisBuffer, *hypotheses) -> None:
        for hypothesis in hypotheses:
            h.insert(hypothesis, 0.0)
            h.flush()

    def test_last_token_is_not_committed_twice_when_its_timestamp_wanders(self):
        h = HypothesisBuffer()
        first = ws("フラン", "ス")
        self.commit(h, first, first)
        self.assertEqual(text(h.committed), "フランス")
        # The same audio again, with every timestamp 150 ms late: by the
        # clock, ス now starts after the committed ス ended.
        late = [Word(w.start + 0.25, w.end + 0.25, w.text, 0.9) for w in ws("フラン", "ス", "の", "授業")]
        self.commit(h, late, late)
        self.assertEqual(text(h.committed), "フランスの授業")

    def test_cut_inside_a_token_when_tokenisation_changes(self):
        h = HypothesisBuffer()
        first = ws("大事", "だ")
        self.commit(h, first, first)
        # Same words, different pieces: だ is now fused with と.
        second = ws("大事", "だと", "思い", "ます")
        self.commit(h, second, second)
        self.assertEqual(text(h.committed), "大事だと思います")

    def test_respelled_ending_does_not_duplicate(self):
        h = HypothesisBuffer()
        first = ws("日本", "で", "打")
        self.commit(h, first, first)
        # The recogniser changes its mind about the last character. What is
        # committed stands; the respelling must not be appended to it.
        second = ws("日本", "で", "売", "って", "いない")
        self.commit(h, second, second)
        # ...and since two passes now agree on the respelling, and nothing
        # has been handed on, it is the respelling that is kept.
        self.assertEqual(text(h.committed), "日本で売っていない")

    def test_respelling_after_hand_on_does_not_duplicate(self):
        h = HypothesisBuffer()
        first = ws("日本", "で", "打")
        self.commit(h, first, first)
        h.take(3)
        second = ws("日本", "で", "売", "って", "いない")
        self.commit(h, second, second)
        self.assertEqual(text(h.committed), "日本で打っていない")

    def test_join_survives_an_opening_the_recogniser_took_back(self):
        # A line was handed on with a hallucinated opening. Later passes read
        # the opening differently, so the fixed text and the hypothesis agree
        # only at the end - which is the only place they need to.
        h = HypothesisBuffer()
        first = ws("話", "で", "十分", "あり", "ました", "けど", "ね", "過去", "に", "も", "似た", "よう")
        self.commit(h, first, first)
        h.take(len(first))
        second = ws("住民", "に", "よる", "と", "過去", "に", "も", "似た", "よう", "な", "訪問", "が")
        self.commit(h, second, second)
        self.assertEqual(text(h.pending), "な訪問が")

    def test_genuine_repetition_is_kept(self):
        h = HypothesisBuffer()
        first = ws("そう", "そう")
        self.commit(h, first, first)
        second = ws("そう", "そう", "そう", "です")
        self.commit(h, second, second)
        self.assertEqual(text(h.committed), "そうそうそうです")

    def test_punctuation_in_the_hypothesis_does_not_disturb_the_match(self):
        h = HypothesisBuffer()
        first = ws("そう", "です")
        self.commit(h, first, first)
        second = ws("そう", "です。", "次", "は")
        self.commit(h, second, second)
        # The newer reading is the one kept, mark and all: the mark is how
        # the recogniser says a sentence ended.
        self.assertEqual(text(h.committed), "そうです。次は")

    def test_unrelated_hypothesis_falls_back_to_the_clock(self):
        h = HypothesisBuffer()
        first = ws("最初", "の", "文")
        self.commit(h, first, first)
        # Nothing like the settled text: only words after it in time are new.
        other = ws("全然", "違う", "話", "です", at=0.9)
        self.commit(h, other, other)
        self.assertEqual(text(h.committed), "最初の文全然違う話です")


class Recorded:
    """A recogniser that returns scripted hypotheses and logs what it was given."""

    def __init__(self, hypotheses):
        self._hypotheses = list(hypotheses)
        self.seen: list[float] = []

    def transcribe(self, audio):
        self.seen.append(audio.size / SAMPLE_RATE)
        return self._hypotheses.pop(0) if self._hypotheses else []


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class Stream(unittest.TestCase):
    def cfg(self, **over) -> Config:
        cfg = Config()
        cfg.min_chunk_sec = 1.0
        cfg.max_buffer_sec = 4.0
        for key, value in over.items():
            setattr(cfg, key, value)
        return cfg

    def test_short_buffer_is_not_transcribed_unless_forced(self):
        asr = Recorded([ws("はい")])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(0.4))
        self.assertEqual(s.process(), ([], []))
        self.assertEqual(asr.seen, [])
        s.process(force=True)
        self.assertEqual(len(asr.seen), 1)

    def test_empty_buffer_is_never_transcribed(self):
        asr = Recorded([ws("はい")])
        s = StreamingTranscriber(self.cfg(), asr)
        s.process(force=True)
        self.assertEqual(asr.seen, [])

    def test_clock_survives_end_of_utterance(self):
        asr = Recorded([ws("一", "二"), ws("一", "二")])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(1.5))
        s.process()
        s.process()
        s.end_utterance()
        self.assertAlmostEqual(s.stream_time, 1.5)
        self.assertEqual(s.buffer_sec, 0.0)
        s.insert_audio(audio(1.0))
        self.assertAlmostEqual(s.stream_time, 2.5)

    def test_timestamps_are_absolute_after_a_new_utterance(self):
        asr = Recorded([ws("前"), ws("後", at=0.2)])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(1.2))
        s.process()
        s.end_utterance()
        s.insert_audio(audio(1.2))
        s.process()
        tail = s.end_utterance()
        self.assertAlmostEqual(tail[0].start, 1.2 + 0.2)

    def test_buffer_is_trimmed_behind_committed_words(self):
        hyp = ws("一", "二", "三", each=1.0)
        asr = Recorded([hyp, hyp, hyp])
        s = StreamingTranscriber(self.cfg(max_buffer_sec=4.0), asr)
        s.insert_audio(audio(4.5))
        s.process()
        s.process()  # agreement: all three commit, buffer is over the cap
        self.assertLess(s.buffer_sec, 2.0)
        self.assertAlmostEqual(s.stream_time, 4.5)

    def test_drop_backlog_keeps_the_clock_and_the_committed_text(self):
        asr = Recorded([ws("一"), ws("一", "二")])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(2.0))
        s.process()
        s.process()
        s.drop_backlog()
        self.assertAlmostEqual(s.stream_time, 2.0)
        self.assertEqual(s.buffer_sec, 0.0)
        self.assertEqual(text(s.hypothesis.committed), "一")
        self.assertEqual(s.hypothesis.unstable, [])

    def test_discarding_audio_keeps_what_was_settled(self):
        asr = Recorded([ws("一", "二"), ws("一", "二", "三")])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(2.0))
        s.process()
        s.process()
        s.discard_audio()
        self.assertEqual(text(s.pending), "一二")
        self.assertEqual(s.hypothesis.unstable, [])
        self.assertEqual(s.buffer_sec, 0.0)
        self.assertAlmostEqual(s.stream_time, 2.0)

    def test_prune_history(self):
        asr = Recorded([ws("一", "二", each=0.5)] * 2)
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(1.0))
        s.process()
        s.process()
        s.end_utterance()
        s.insert_audio(audio(60.0))
        # Words still waiting to be handed on are kept however old they are.
        s.prune_history(30.0)
        self.assertEqual(text(s.hypothesis.committed), "一二")
        s.take(2)
        s.prune_history(30.0)
        self.assertEqual(s.hypothesis.committed, [])

    def test_trim_keeps_settled_speech_as_context(self):
        # One-second words. Eight settle, then the buffer goes over its cap.
        eight = ws(*[f"語{i}" for i in range(8)], each=1.0)
        ten = ws(*[f"語{i}" for i in range(10)], each=1.0)
        asr = Recorded([eight, eight, ten])
        s = StreamingTranscriber(self.cfg(max_buffer_sec=9.0), asr)
        s.insert_audio(audio(8.5))
        s.process()
        s.process()
        self.assertEqual(len(s.hypothesis.soft), 8)
        s.insert_audio(audio(2.0))
        s.process()
        # Not cut back to the last settled word, at 8.0: a couple of seconds
        # of settled speech stay in the buffer for the recogniser to lean on.
        self.assertLess(s.buffer_start, 8.0 - 2.0)
        self.assertGreater(s.buffer_start, 3.0)
        self.assertAlmostEqual(s.stream_time, 10.5)
        # What was settled is fixed now; it will not be re-read from a
        # buffer that may begin partway through a word.
        self.assertEqual(s.hypothesis.soft, [])
        self.assertEqual(len(s.pending), 8)

    def test_a_word_followed_by_silence_is_not_held_back(self):
        hyp = ws("そう", "です", each=0.5)
        asr = Recorded([hyp, hyp])
        s = StreamingTranscriber(self.cfg(commit_holdback_sec=0.5), asr)
        s.insert_audio(audio(1.2))
        s.process()
        settled, unstable = s.process(quiet_from=1.0)
        # です ends 0.2 s before the buffer does, inside the holdback - but
        # the speaker has stopped, so there is nothing for it to be half of.
        self.assertEqual(text(settled), "そうです")
        self.assertEqual(unstable, [])

    def test_split_settles_what_came_before_and_keeps_the_audio_after(self):
        first = ws("行き", "ました", each=0.5)
        both = first + ws("それ", "で", at=1.4, each=0.3)
        asr = Recorded([first, both])
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(2.2))
        s.process()
        s.process()
        s.split(1.2)  # in the gap between the two phrases
        self.assertEqual(text(s.hypothesis.held), "行きました")
        self.assertEqual(text(s.hypothesis.unstable), "それで")
        self.assertAlmostEqual(s.buffer_start, 1.2)
        self.assertAlmostEqual(s.stream_time, 2.2)

    def test_first_word_after_a_split_is_not_mistaken_for_stale(self):
        # The word before the pause has its timing stretched right across it,
        # so by the clock it "ends" after the next word begins.
        stretched = [Word(0.0, 0.5, "過去に", 0.9), Word(0.5, 1.6, "も", 0.9)]
        asr = Recorded(
            [
                stretched,
                stretched,
                ws("似た", "よう", "な", at=0.2, each=0.3),  # relative to the new buffer
                ws("似た", "よう", "な", "訪問", at=0.2, each=0.3),
            ]
        )
        s = StreamingTranscriber(self.cfg(), asr)
        s.insert_audio(audio(2.0))
        s.process()
        s.process(quiet_from=1.0)
        s.take(2)
        s.split(1.3)
        s.insert_audio(audio(1.5))
        s.process()
        s.process()
        self.assertTrue(text(s.pending).startswith("似た"), text(s.pending))


if __name__ == "__main__":
    unittest.main()
