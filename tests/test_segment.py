import unittest
from dataclasses import dataclass

from subordinant import segment as seg


@dataclass
class W:
    start: float
    end: float
    text: str


@dataclass
class P:
    start: float
    end: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


def words(*texts: str, each: float = 0.3, at: float = 0.0, gaps: dict | None = None):
    """Words laid end to end; gaps maps a word index to silence after it."""
    out, t = [], at
    for i, text in enumerate(texts):
        out.append(W(t, t + each, text))
        t += each + (gaps or {}).get(i, 0.0)
    return out


class Completeness(unittest.TestCase):
    def test_finished_sentences(self):
        for text in (
            "歯が痛いので口を開けることができません",
            "週に4回フランスの授業があります",
            "理解できなくなった",
            "休みたいんだよ",
            "本当ですよね",
            "かもしれないですね",
            "ごめんなさい",
            "こんにちは",
            "そうかも",
            "すごいねぇー",
            "行ったよ！",
        ):
            with self.subTest(text=text):
                self.assertTrue(seg.looks_complete(text))
                self.assertFalse(seg.looks_incomplete(text))

    def test_dangling_sentences(self):
        for text in (
            "これは日本で",
            "私は",
            "それでね",  # ね after a conjunction is a filler, not an ending
            "ですが",
            "ですけど",
            "なんか",  # ends in か, which would be final on its own
            "とか",
            "まだ",  # ends in だ, likewise
            "あの",
            "えっと…",
            "それは、",
        ):
            with self.subTest(text=text):
                self.assertFalse(seg.looks_complete(text))
                self.assertTrue(seg.looks_incomplete(text))

    def test_no_opinion(self):
        # Neither list recognises these, so a long pause finishes them and a
        # short one does not.
        for text in ("いい", "やっぱり"):
            with self.subTest(text=text):
                self.assertFalse(seg.looks_complete(text))
                self.assertFalse(seg.looks_incomplete(text))

    def test_punctuation_wins(self):
        self.assertTrue(seg.looks_complete("それは。"))
        self.assertFalse(seg.looks_incomplete("それは。"))
        # An ellipsis is hesitation, not an ending.
        self.assertFalse(seg.ends_sentence("それは..."))

    def test_content(self):
        self.assertFalse(seg.has_content("...。"))
        self.assertFalse(seg.has_content("  "))
        self.assertTrue(seg.has_content("はい。"))


class Cuts(unittest.TestCase):
    def test_sentence_cut_takes_last_ending(self):
        ws = words("そう", "です。", "でも", "ね、", "いい", "よ。", "あと")
        self.assertEqual(seg.sentence_cut(ws), 6)
        self.assertEqual(seg.sentence_cut(words("そう", "です")), 0)

    def test_sentence_mark_followed_by_a_continuation_is_not_an_ending(self):
        # The recogniser wrote 。 and then heard the sentence carry on.
        ws = words("なさ", "そう", "です。", "けど", "ね")
        self.assertEqual(seg.sentence_cut(ws), 0)
        ws = words("なさ", "そう", "です。")
        self.assertEqual(seg.sentence_cut(ws, next_word="けど"), 0)
        self.assertEqual(seg.sentence_cut(ws, next_word="次"), 3)

    def test_sentence_mark_on_the_last_word_waits_for_evidence(self):
        ws = words("なさ", "そう", "です。")
        # Still talking, next word not in yet: not trusted.
        self.assertEqual(seg.sentence_cut(ws, trailing_ok=False), 0)
        # The speaker has paused: trusted.
        self.assertEqual(seg.sentence_cut(ws, trailing_ok=True), 3)
        # An earlier mark with words after it is trusted either way.
        ws = words("はい。", "それ", "は", "です。")
        self.assertEqual(seg.sentence_cut(ws, trailing_ok=False), 1)

    def test_fragments(self):
        for text in ("さんだ", "っけ", "だ", "か、"):
            with self.subTest(text=text):
                self.assertTrue(seg.is_fragment(text))
        for text in ("はい", "うん。", "なに?", "ありがとう", "そうですね"):
            with self.subTest(text=text):
                self.assertFalse(seg.is_fragment(text))

    def test_pause_cut_does_not_strand_a_fragment(self):
        ws = words("さん", "だ", "それ", "から", gaps={1: 0.25})
        self.assertEqual(seg.pause_cut(ws, [P(0.6, 0.85)]), 0)

    def test_pause_boundary_matches_by_midpoints(self):
        ws = words("あり", "ます", "この", "動画", gaps={1: 0.25})
        # Words 0.0-0.3, 0.3-0.6, [pause 0.6-0.85], 0.85-1.15, ...
        self.assertEqual(seg.pause_boundaries(ws, [P(0.6, 0.85)]), [2])
        # A pause the timestamps place slightly inside the neighbouring word
        # still lands on the same boundary.
        self.assertEqual(seg.pause_boundaries(ws, [P(0.5, 0.8)]), [2])

    def test_pause_cut_needs_a_finished_sentence(self):
        done = words("授業が", "あり", "ます", "この", "動画", gaps={2: 0.25})
        self.assertEqual(seg.pause_cut(done, [P(0.9, 1.15)]), 3)
        dangling = words("これは", "日本", "で", "売って", "いない", gaps={2: 0.25})
        self.assertEqual(seg.pause_cut(dangling, [P(0.9, 1.15)]), 0)

    def test_pause_cut_slides_one_token(self):
        # The pause is matched one token early - between あり and ま - but the
        # sentence only reads as finished after す.
        ws = words("授業が", "あり", "ま", "す", "この", gaps={3: 0.25})
        early = P(0.85, 1.0)  # centre inside ま's span
        self.assertEqual(seg.pause_cut(ws, [early]), 4)

    def test_pause_cut_refuses_a_continuation(self):
        ws = words("行き", "ます", "けど", "明日は", gaps={1: 0.25})
        self.assertEqual(seg.pause_cut(ws, [P(0.6, 0.85)]), 0)
        # ...and the same when the continuation is still unconfirmed.
        ws = words("行き", "ます", gaps={1: 0.0})
        self.assertEqual(
            seg.pause_cut(ws, [P(0.6, 0.85)], next_word="けど", next_start=0.85), 0
        )

    def test_pause_cut_at_the_very_end(self):
        ws = words("行き", "ます")
        self.assertEqual(seg.pause_cut(ws, [P(0.6, 0.8)]), 2)
        # Not a boundary if the recogniser still has a word in flight that
        # starts before the pause does.
        self.assertEqual(
            seg.pause_cut(ws, [P(0.6, 0.8)], next_word="か", next_start=0.45), 0
        )

    def test_a_sentence_starting_with_a_particle_syllable_is_not_a_continuation(self):
        # かなり starts with か. It must not veto the boundary before it.
        ws = words("本当", "です", "かなり", "広い", gaps={1: 0.25})
        self.assertEqual(seg.pause_cut(ws, [P(0.6, 0.85)]), 2)

    def test_overdue_prefers_pause_then_clause_then_particle(self):
        ws = words("今日は", "友達と", "一緒に", "東京に", "行って", gaps={1: 0.2})
        self.assertEqual(seg.overdue_cut(ws, [P(0.6, 0.8)]), 2)
        ws = words("今日は、", "友達と", "一緒に")
        self.assertEqual(seg.overdue_cut(ws, []), 1)
        ws = words("今日", "友達と", "一緒", "東京")
        self.assertEqual(seg.overdue_cut(ws, []), 2)  # after と
        ws = words("東京", "大阪", "京都")
        self.assertEqual(seg.overdue_cut(ws, []), 3)  # nothing better: all of it

    def test_split_sentences(self):
        self.assertEqual(
            seg.split_sentences("そうなの？それはすごいねぇ。っていうか、お嬢様なの？連れてるけど"),
            ["そうなの？", "それはすごいねぇ。", "っていうか、お嬢様なの？", "連れてるけど"],
        )
        self.assertEqual(seg.split_sentences("えっ！？本当？"), ["えっ！？", "本当？"])
        self.assertEqual(seg.split_sentences("句読点なし"), ["句読点なし"])


@dataclass
class MarkedP(P):
    letters: int | None = None


class WordBoundaries(unittest.TestCase):
    def test_cuts_that_fall_inside_a_word(self):
        for before, after in (
            ("思い出せないみたい台", "風や大雨"),  # two kanji: one compound
            ("NHKニュ", "ース"),  # katakana run, and ー cannot begin a word
            ("それはN", "HKです"),
            ("今から行", "きます"),  # a kanji and its own ending
            ("ちょっと待っ", "て"),
            ("こんなにかわい", "ゃ"),
        ):
            with self.subTest(before=before, after=after):
                self.assertTrue(seg.splits_word(before, after))

    def test_cuts_that_fall_between_words(self):
        for before, after in (
            ("思い出せないみたい", "台風や大雨"),
            ("私", "は学生です"),  # kanji then a particle
            ("東京に", "行きます"),
            ("そうですね。", "次は"),
            ("", "こんにちは"),
            ("こんにちは", ""),
        ):
            with self.subTest(before=before, after=after):
                self.assertFalse(seg.splits_word(before, after))

    def test_overdue_cut_does_not_take_half_a_compound(self):
        # The pause is after みたい. Word timings put it one token late, on
        # the far side of 台.
        ws = words("よく", "思い出せ", "ない", "みたい", "台", "風", "や", "大雨", "に")
        pause = P(ws[4].end - 0.05, ws[4].end + 0.05)
        self.assertEqual(seg.pause_boundaries(ws, [pause]), [5])
        self.assertEqual(seg.overdue_cut(ws, [pause]), 4)

    def test_overdue_cut_backs_off_to_a_word_boundary(self):
        # No pause, no clause mark. Taking everything would end the line on
        # 大学 with 院 still to come; the last place that is not inside a
        # word is after は.
        ws = words("今日", "は", "東京", "大", "学")
        self.assertEqual(seg.overdue_cut(ws, [], next_word="院"), 2)


class PauseMarks(unittest.TestCase):
    """A pause placed by what had been read when it was heard, not by time."""

    def test_mark_overrides_a_misleading_timestamp(self):
        ws = words("残念", "だった", "のに", "かな", "り", "広い")
        # By the clock the pause sits after かな...
        by_time = P(ws[3].end - 0.05, ws[3].end + 0.05)
        self.assertEqual(seg.pause_boundaries(ws, [by_time]), [4])
        # ...but seven characters had been read when it was heard: 残念だったのに.
        marked = MarkedP(by_time.start, by_time.end, letters=7)
        self.assertEqual(seg.pause_boundaries(ws, [marked], base_letters=0), [3])

    def test_mark_counts_from_the_start_of_the_run(self):
        ws = words("かなり", "広い", "範囲", "で")
        # 20 characters were handed on before these words; the pause came
        # five further in.
        marked = MarkedP(0.0, 0.1, letters=25)
        self.assertEqual(seg.pause_boundaries(ws, [marked], base_letters=20), [2])

    def test_mark_before_or_beyond_these_words_is_not_a_boundary(self):
        ws = words("かなり", "広い")
        self.assertEqual(
            seg.pause_boundaries(ws, [MarkedP(0.0, 0.1, letters=20)], base_letters=20), []
        )
        self.assertEqual(
            seg.pause_boundaries(ws, [MarkedP(0.0, 0.1, letters=40)], base_letters=20), []
        )

    def test_mark_at_the_end_is_a_boundary_after_the_last_word(self):
        ws = words("そう", "です", "ね")
        self.assertEqual(
            seg.pause_boundaries(ws, [MarkedP(5.0, 5.2, letters=5)], base_letters=0), [3]
        )

    def test_pause_cut_uses_the_mark(self):
        ws = words("行き", "ました", "それ", "で")
        marked = MarkedP(ws[2].end, ws[2].end + 0.2, letters=5)  # after 行きました
        self.assertEqual(seg.pause_cut(ws, [marked], base_letters=0), 2)


class SpeakerChange(unittest.TestCase):
    def test_change_is_moved_to_the_pause_beside_it(self):
        # A pause after 動画 (boundary 2). The tracker reports the new voice
        # two words late, in the middle of ハプニング.
        ws = words("この", "動画", "ハ", "プニング", "を", gaps={1: 0.3})
        self.assertEqual(seg.nearest_boundary(ws, [2], 3), 2)

    def test_change_with_no_pause_near_it_is_not_a_cut(self):
        ws = words("雪", "の", "日", "は", "出", "かけ", "ません", each=0.4)
        self.assertEqual(seg.nearest_boundary(ws, [], 5), 0)
        # A pause three seconds away is not this change.
        self.assertEqual(seg.nearest_boundary(ws, [1], 6, within_sec=1.0), 0)

    def test_nearest_of_several_pauses(self):
        ws = words("あ", "い", "う", "え", "お", "か", each=0.5)
        self.assertEqual(seg.nearest_boundary(ws, [1, 4], 3), 4)


if __name__ == "__main__":
    unittest.main()
