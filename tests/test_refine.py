"""The slang glossary and the second look's guards.

Nothing here talks to a model. What the model says is the part that cannot be
tested; what is done with its reply is the part that has to be right whatever
it says, and that is all plain functions.
"""

from __future__ import annotations

import time
import types
import unittest

from subordinant.config import Config
from subordinant.glossary import Glossary, default, fold
from subordinant.history import History
from subordinant.pipeline import CaptionPipeline, Utterance, _Seen
from subordinant.refine import (
    Line,
    Refiner,
    Reply,
    apply_fixes,
    borrows_from,
    build_request,
    english_is_plausible,
    is_sound_alike_swap,
    join_to_next,
    judge,
    opens_the_ending,
    parse_reply,
    runs_on,
    worth_a_look,
)


class GlossaryLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = default()

    def terms(self, text: str) -> list[str]:
        return [term.forms[0] for _written, term in self.glossary.lookup(text)]

    def test_finds_stream_slang(self) -> None:
        self.assertEqual(self.terms("今日の枠はここまで、スパチャありがとう"), ["枠", "スパチャ"])

    def test_plain_japanese_has_no_notes(self) -> None:
        self.assertEqual(self.glossary.notes("今日は友達と東京に行ってきた。"), [])

    def test_script_and_kana_size_do_not_matter(self) -> None:
        self.assertEqual(fold("スパチャ"), fold("すぱちゃ"))
        self.assertEqual(self.terms("すぱちゃ読みます"), ["スパチャ"])
        self.assertEqual(self.terms("てえてえなあ"), ["てぇてぇ"])

    def test_ordinary_compounds_are_not_slang(self) -> None:
        for text in ("庭の雑草を抜いた", "窓枠を直した", "卒業式に行った", "天井裏にいる"):
            self.assertEqual(self.terms(text), [], text)

    def test_grammar_that_only_looks_like_slang(self) -> None:
        # Folding kana makes ママ and まま one string, and they are not one word.
        self.assertEqual(self.terms("何の進展もないまま終了しました"), [])
        self.assertEqual(self.terms("ママが今描いてくれてる"), ["ママ"])
        self.assertEqual(self.terms("それなら行くよ"), [])
        self.assertEqual(self.terms("それな！"), ["それな"])

    def test_slang_beside_an_ordinary_compound_is_still_found(self) -> None:
        self.assertEqual(self.terms("雑草を抜いてたら転んで草"), ["草"])

    def test_longer_term_hides_the_one_inside_it(self) -> None:
        self.assertEqual(self.terms("箱推しです"), ["箱推し"])
        self.assertEqual(self.terms("次は凸待ちやります"), ["凸待ち"])

    def test_note_says_when_the_word_was_misheard(self) -> None:
        (note,) = self.glossary.notes("来週は突待ちをやります")
        self.assertTrue(note.startswith("凸待ち (misheard here as 突待ち):"), note)
        (note,) = self.glossary.notes("来週は凸待ちをやります")
        self.assertTrue(note.startswith("凸待ち:"), note)

    def test_limit(self) -> None:
        text = "スパチャ 赤スパ 同接 切り抜き 歌枠 初見 古参"
        self.assertEqual(len(self.glossary.lookup(text, limit=3)), 3)

    def test_every_entry_matches_its_own_forms(self) -> None:
        # An entry whose `unless` list swallows its own form can never fire.
        for term in self.glossary._terms:
            for form in term.forms + term.heard:
                found = [t for _w, t in Glossary((term,)).lookup(form)]
                self.assertEqual(found, [term], form)


class Respelling(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = default()

    def test_known_mishearings_are_put_right(self) -> None:
        cases = {
            "来週の土曜日に突待ちをやります。": "来週の土曜日に凸待ちをやります。",
            "同説が5000人超えてる": "同接が5000人超えてる",
            "昨日のコラボ、ていていって言われてて草": "昨日のコラボ、てぇてぇって言われてて草",
            "箱押しの人も単押しの人も": "箱推しの人も単推しの人も",
        }
        for heard, said in cases.items():
            self.assertEqual(self.glossary.respell(heard), said)

    def test_proper_spellings_and_plain_text_are_untouched(self) -> None:
        for text in (
            "来週の土曜日に凸待ちをやります。",
            "スーパーチャットありがとう",  # a spelling, not a mishearing
            "てえてえなあ",
            "今日は友達と東京に行ってきた。",
        ):
            self.assertEqual(self.glossary.respell(text), text)

    def test_every_mishearing_differs_from_every_spelling(self) -> None:
        spellings = {form for term in self.glossary._terms for form in term.forms}
        for term in self.glossary._terms:
            for heard in term.heard:
                self.assertNotIn(heard, spellings)


class WordFixes(unittest.TestCase):
    def test_one_kanji_for_another_is_a_sound_alike(self) -> None:
        for heard, said in (("以外", "意外"), ("書いて", "描いて"), ("機会", "機械"), ("以外と", "意外と")):
            self.assertTrue(is_sound_alike_swap(heard, said), (heard, said))

    def test_a_different_word_is_not(self) -> None:
        for heard, said in (
            ("近藤", "ルヴルベ"),  # another line's word
            ("消費", "視察"),  # two kanji changed
            ("言い", "聞き"),  # the kana changed, so the sound did
            ("した", "やり"),  # grammar, not a word
            ("彼方", "かなた"),  # a respelling: right, but not provably
            ("意外", "意外"),
            ("", ""),
        ):
            self.assertFalse(is_sound_alike_swap(heard, said), (heard, said))

    def test_fix_is_applied_in_place(self) -> None:
        self.assertEqual(apply_fixes("以外と出ないもんだね", [("以外", "意外")]), "意外と出ないもんだね")

    def test_doubtful_fixes_are_skipped(self) -> None:
        line = "消費も少しずつ回復の兆しが見える"
        self.assertEqual(apply_fixes(line, [("消費", "視察")]), line)
        # Not in the line at all.
        self.assertEqual(apply_fixes(line, [("以外", "意外")]), line)
        # Twice in the line: no telling which was meant.
        twice = "以外と以外な結果"
        self.assertEqual(apply_fixes(twice, [("以外", "意外")]), twice)


class EnglishGuards(unittest.TestCase):
    def test_ordinary_line(self) -> None:
        self.assertTrue(english_is_plausible("Did you get scolded?", "She got mad at me.", "怒られた"))

    def test_empty_and_japanese_are_refused(self) -> None:
        self.assertFalse(english_is_plausible("  ", "Hello.", "こんにちは"))
        self.assertFalse(english_is_plausible("今日はありがとう", "Thanks.", "今日はありがとう"))

    def test_two_lines_in_one_is_refused(self) -> None:
        self.assertFalse(
            english_is_plausible(
                "So I'll wrap up for today. Please subscribe and leave a like on the video.",
                "So I'll wrap up for today.",
                "じゃあ今日はこの辺で終わります",
            )
        )

    def test_a_neighbours_line_is_recognised(self) -> None:
        self.assertTrue(
            borrows_from(
                "The girl was looking for her mother",
                "while screaming loudly",
                ["The girl was looking for her mother."],
            )
        )
        self.assertFalse(
            borrows_from(
                "Mom is drawing it for me right now.",
                "Mom is writing it for me right now.",
                ["I'd like to reveal my new outfit soon."],
            )
        )

    def test_opening_an_ending(self) -> None:
        first = "I decided I'm not going to spend any more money on this."
        self.assertTrue(opens_the_ending(first[:-1] + ", but", first))
        self.assertTrue(opens_the_ending(first[:-1] + ",", first))
        self.assertTrue(opens_the_ending(first[:-1] + "...", first))
        self.assertFalse(opens_the_ending("Even though I decided to stop spending,", first))
        self.assertFalse(opens_the_ending(first[:-1] + ", but I spent 30,000 yen", first))


class Judging(unittest.TestCase):
    def test_fixed_word_brings_its_english(self) -> None:
        line = Line("ママが今書いてくれてる", "Mom is writing it for me.")
        reply = Reply("misheard", [("書いて", "描いて")], "Mom is drawing it for me.")
        # Not unless asked: with the models to hand, word fixes do more harm
        # than good, and they are off by default.
        self.assertIsNone(judge(line, reply)[0])
        revision, verdict = judge(line, reply, fix_words=True)
        self.assertEqual(verdict, "changed")
        self.assertEqual(revision.source, "ママが今描いてくれてる")
        self.assertEqual(revision.english, "Mom is drawing it for me.")
        self.assertTrue(revision.source_changed and revision.english_changed)

    def test_fix_without_english_changes_only_the_japanese(self) -> None:
        line = Line("以外と出ないもんだね", "It's surprisingly rare.")
        revision, verdict = judge(line, Reply("fine", [("以外", "意外")], ""), fix_words=True)
        self.assertEqual(verdict, "changed")
        self.assertEqual(revision.english, "It's surprisingly rare.")
        self.assertTrue(revision.source_changed)
        self.assertFalse(revision.english_changed)

    def test_rewording_without_a_reason_is_ignored(self) -> None:
        line = Line("怒られた", "She got mad at me.")
        for check in ("fine", "misheard", "continues"):
            revision, verdict = judge(line, Reply(check, [], "She was angry with me."))
            self.assertIsNone(revision, check)
            self.assertEqual(verdict, "same", check)

    def test_refused_fix_takes_its_english_with_it(self) -> None:
        line = Line("消費も回復の兆しが見える", "Consumption is showing signs of recovery.")
        revision, _verdict = judge(
            line,
            Reply("misheard", [("消費", "視察")], "Inspections are showing signs of recovery."),
            fix_words=True,
        )
        self.assertIsNone(revision)

    def test_sentence_that_runs_on_is_left_open(self) -> None:
        line = Line("課金はもうしないって決めたのに", "I decided not to spend any more.")
        reply = Reply("continues", [], "I decided not to spend any more, but")
        revision, verdict = judge(line, reply, runs_on=True)
        self.assertEqual(verdict, "changed")
        self.assertEqual(revision.english, "I decided not to spend any more, but")
        # The model's word alone is not enough: the Japanese has to say so.
        self.assertIsNone(judge(line, reply, runs_on=False)[0])
        # And it may only open the ending, not rewrite the line.
        rewrite = Reply("continues", [], "Even though I'd sworn off spending,")
        self.assertIsNone(judge(line, rewrite, runs_on=True)[0])

    def test_a_line_with_no_english_takes_any_plausible_one(self) -> None:
        revision, verdict = judge(Line("こんばんは"), Reply("fine", [], "Good evening."))
        self.assertEqual(verdict, "changed")
        self.assertEqual(revision.english, "Good evening.")

    def test_a_neighbours_line_is_thrown_away(self) -> None:
        revision, verdict = judge(
            Line("大声で鳴きながら"),
            Reply("fine", [], "The girl was looking for her mother"),
            neighbours=["The girl was looking for her mother."],
        )
        self.assertIsNone(revision)
        self.assertEqual(verdict, "bad-english")

    def test_which_lines_the_model_is_sent(self) -> None:
        self.assertTrue(worth_a_look(Line("こんばんは"), has_next=False))  # no English yet
        self.assertTrue(worth_a_look(Line("以外と出ない", "x"), has_next=False))  # a kanji
        self.assertTrue(worth_a_look(Line("そうしたいんだけれども", "x"), has_next=True))
        self.assertFalse(worth_a_look(Line("そうしたいんだけれども", "x"), has_next=False))
        self.assertFalse(worth_a_look(Line("ありがとう", "Thanks."), has_next=True))


class Joining(unittest.TestCase):
    """The default second look: a rule, no model."""

    def test_which_lines_run_on(self) -> None:
        self.assertEqual(runs_on("課金はもうしないって決めたのに"), "but")
        self.assertEqual(runs_on("この前の耐久配信は12時間かかったけど、"), "but")
        self.assertEqual(runs_on("今日は早く終わったので…"), "so")
        self.assertEqual(runs_on("もう少しで終わるから"), "so")

    def test_which_do_not(self) -> None:
        for text in (
            "意外と出ないもんだね。",  # finished
            "ゆっくりしていってね",  # て is not a reason
            "えっと…",
            "だけど",  # too short to be a clause
            "それはどこから？",  # a question
            "",
        ):
            self.assertIsNone(runs_on(text), text)

    def test_ending_is_reopened(self) -> None:
        line = Line("課金はもうしないって決めたのに", "I decided not to spend any more.")
        self.assertEqual(join_to_next(line), "I decided not to spend any more, but")

    def test_no_second_joining_word(self) -> None:
        line = Line("この前の耐久配信は12時間かかったけど", "The last one took 12 hours though.")
        self.assertEqual(join_to_next(line), "The last one took 12 hours though,")
        line = Line("で、ゲームの話なんだけど", "So, let's talk about the game.")
        self.assertEqual(join_to_next(line), "So, let's talk about the game,")

    def test_english_that_is_already_open_or_not_a_statement_is_left(self) -> None:
        source = "課金はもうしないって決めたのに"
        for english in (
            "I decided not to spend any more...",
            "I decided not to spend any more,",
            "Didn't I decide not to spend any more?",
            "No more spending!",
            "Um.",
            "",
        ):
            self.assertIsNone(join_to_next(Line(source, english)), english)

    def test_refiner_joins_without_a_model(self) -> None:
        # No server is contacted: constructing this would fail if it were.
        cfg = Config()
        cfg.llm_endpoint = "http://127.0.0.1:9"
        refiner = Refiner(cfg)
        self.assertFalse(refiner.uses_model)
        line = Line("課金はもうしないって決めたのに", "I decided not to spend any more.")
        after = [Line("気づいたら3万円使ってた", "Before I knew it I'd spent 30,000 yen.")]
        revision = refiner.refine(line, [], after)
        self.assertEqual(revision.english, "I decided not to spend any more, but")
        self.assertEqual(revision.source, line.source)
        self.assertFalse(revision.source_changed)
        # Nothing after it, or the next line is not its continuation: left.
        self.assertIsNone(refiner.refine(line, [], []))
        self.assertIsNone(refiner.refine(line, [], after, follows_on=False))
        self.assertEqual(refiner.last_verdict, "skipped")
        # A finished line: left.
        self.assertIsNone(refiner.refine(Line("そうなんだ", "I see."), [], after))


class Replies(unittest.TestCase):
    def test_json(self) -> None:
        reply = parse_reply(
            '{"check": "misheard", "fixes": [{"heard": "以外", "said": "意外"}], '
            '"english": "It\'s  surprisingly rare."}'
        )
        self.assertEqual(reply, Reply("misheard", [("以外", "意外")], "It's surprisingly rare."))

    def test_the_usual_reply_is_nearly_empty(self) -> None:
        self.assertEqual(parse_reply('{"check": "fine", "fixes": [], "english": ""}'), Reply())

    def test_json_inside_other_text_and_unknown_check(self) -> None:
        reply = parse_reply('Here you go:\n{"check": "question", "english": "Ah?"}\nDone.')
        self.assertEqual(reply, Reply("fine", [], "Ah?"))

    def test_not_a_reply(self) -> None:
        for content in ("", "Did you get scolded?", '{"english": 3}', "[1, 2]"):
            self.assertIsNone(parse_reply(content), content)

    def test_malformed_fixes_are_dropped(self) -> None:
        reply = parse_reply('{"check": "misheard", "fixes": ["x", {"heard": 1, "said": "a"}], "english": ""}')
        self.assertEqual(reply.fixes, [])

    def test_request_puts_the_line_between_its_neighbours(self) -> None:
        text = build_request(
            Line("怒られた", "She got mad."),
            [Line("マネージャーさんに怒られるかな", "Will my manager be mad?")],
            [Line("まだ言ってないからセーフ", "I haven't told her yet.")],
            ["x: y"],
        )
        order = [text.index(part) for part in (
            "マネージャーさんに怒られるかな", "The line to check", "怒られた\n",
            "First translation: She got mad.", "まだ言ってないからセーフ", "x: y",
        )]
        self.assertEqual(order, sorted(order))
        # The following lines' English is for the guards, not for the model:
        # shown it, the model copies it.
        self.assertNotIn("I haven't told her yet.", text)


class TranscriptRevision(unittest.TestCase):
    def test_final_line_is_corrected_once(self) -> None:
        history = History()
        history.upsert(Utterance(id=1, source="以外と出ない", english="Other than that.", final=True))
        row = history.revise(
            Utterance(id=1, source="意外と出ない", english="Surprisingly rare.", revised=True)
        )
        self.assertEqual(row, 0)
        entry = history.entries[0]
        self.assertTrue(entry.revised)
        self.assertEqual((entry.source, entry.english), ("意外と出ない", "Surprisingly rare."))
        self.assertEqual((entry.first_source, entry.first_english), ("以外と出ない", "Other than that."))
        # A second correction is refused: the line has stopped moving.
        self.assertIsNone(history.revise(Utterance(id=1, source="x", english="y")))
        self.assertEqual(history.entries[0].english, "Surprisingly rare.")

    def test_drafts_and_missing_lines_are_left_alone(self) -> None:
        history = History()
        history.upsert(Utterance(id=1, source="あ", english="Ah", final=False))
        self.assertIsNone(history.revise(Utterance(id=1, source="あ", english="Oh")))
        self.assertIsNone(history.revise(Utterance(id=9, source="あ", english="Oh")))


class Scheduling(unittest.TestCase):
    """Which line gets its second look, and when."""

    def setUp(self) -> None:
        cfg = Config()
        cfg.refine_lookahead = 2
        cfg.refine_wait_sec = 5.0
        cfg.refine_context_lines = 2
        self.pipeline = CaptionPipeline(cfg)
        # The model path, which is the one that waits for two lines.
        self.pipeline._refiner = types.SimpleNamespace(uses_model=True)

    def add(self, ident: int, age: float = 0.0, speaker: int = -1, gap: float = 0.5) -> None:
        start = (self.pipeline._seen[-1].utterance.end + gap) if self.pipeline._seen else 0.0
        self.pipeline._seen.append(
            _Seen(
                Utterance(
                    id=ident, source=f"ja{ident}", english=f"en{ident}",
                    speaker=speaker, start=start, end=start + 2.0,
                ),
                time.perf_counter() - age,
            )
        )

    def pick(self):
        with self.pipeline._refine_cond:
            return self.pipeline._refine_pick()

    def test_waits_for_the_lines_that_follow(self) -> None:
        self.add(1)
        self.assertIsNone(self.pick())
        self.add(2)
        self.assertIsNone(self.pick())
        self.add(3)
        seen, before, after, follows_on = self.pick()
        self.assertEqual(seen.utterance.id, 1)
        self.assertEqual(before, [])
        self.assertEqual(after, [("ja2", "en2"), ("ja3", "en3")])
        self.assertTrue(follows_on)
        # Line 2 has only one line after it, and is not old yet.
        self.assertIsNone(self.pick())

    def test_a_join_needs_only_the_next_line(self) -> None:
        self.pipeline._refiner = types.SimpleNamespace(uses_model=False)
        self.add(1)
        self.assertIsNone(self.pick())
        self.add(2)
        seen, _before, after, follows_on = self.pick()
        self.assertEqual(seen.utterance.id, 1)
        self.assertEqual(after, [("ja2", "en2")])
        self.assertTrue(follows_on)

    def test_a_line_after_a_gap_or_in_another_voice_does_not_follow_on(self) -> None:
        self.pipeline._refiner = types.SimpleNamespace(uses_model=False)
        self.add(1)
        self.add(2, gap=4.0)
        self.assertFalse(self.pick()[3])
        self.add(3, speaker=1)
        self.assertFalse(self.pick()[3])

    def test_a_line_before_a_silence_is_checked_after_the_wait(self) -> None:
        self.add(1, age=6.0)
        seen, _before, after, follows_on = self.pick()
        self.assertEqual(seen.utterance.id, 1)
        self.assertEqual(after, [])
        self.assertFalse(follows_on)

    def test_gives_way_to_first_translations(self) -> None:
        for ident in (1, 2, 3):
            self.add(ident)
        self.pipeline._final_busy = True
        self.assertIsNone(self.pick())
        self.pipeline._final_busy = False
        self.assertIsNotNone(self.pick())

    def test_context_is_the_corrected_lines_before_it(self) -> None:
        for ident in (1, 2, 3, 4, 5):
            self.add(ident)
        self.pick()
        self.pipeline._seen[0].utterance.english = "en1 corrected"
        self.pick()
        seen, before, _after, _follows = self.pick()
        self.assertEqual(seen.utterance.id, 3)
        self.assertEqual(before, [("ja1", "en1 corrected"), ("ja2", "en2")])

    def test_lines_from_before_a_long_silence_are_not_context(self) -> None:
        self.pipeline.cfg.llm_context_reset_sec = 30.0
        self.add(1, age=100.0)
        self.pipeline._seen_pos = 1
        self.add(2, age=6.0)
        _seen, before, _after, _follows = self.pick()
        self.assertEqual(before, [])

    def test_falling_behind_skips_the_oldest(self) -> None:
        for ident in range(1, 16):
            self.add(ident)
        seen, _before, _after, _follows = self.pick()
        self.assertEqual(seen.utterance.id, 6)
        self.assertEqual(self.pipeline.metrics.second_look_skipped, 5)


if __name__ == "__main__":
    unittest.main()
