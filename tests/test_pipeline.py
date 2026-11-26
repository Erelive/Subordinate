"""The engine loop, end to end, with scripted models.

What these pin down is ordering and bookkeeping: every word that was spoken
reaches a final line exactly once, in order, whatever the recogniser's timing
does - and when something fails, the run says so and carries on.
"""

import time
import unittest

from subordinant.pipeline import ERROR, IDLE, LISTENING, PAUSED, RECOGNIZING, CaptionPipeline
from tests.helpers import (
    FakeModels,
    FakeTranscriber,
    FakeTranslator,
    RecordingEvents,
    Script,
    ScriptSource,
    run_pipeline,
    test_config,
)


def spoken(script: Script) -> str:
    return "".join(script.words)


def finals_text(recorded) -> str:
    return "".join(u.source for u in recorded.finals)


class Flow(unittest.TestCase):
    def test_every_word_is_finalised_once_in_order(self):
        script = (
            Script()
            .silence(0.5)
            .say("今日は", "友達と", "東京に", "行き", "ました。")
            .silence(1.0)
            .say("すごく", "楽し", "かった", "です。")
            .silence(1.5)
        )
        nmt = FakeTranslator()
        cfg = test_config()
        pipeline, r = run_pipeline(cfg, script, FakeModels(FakeTranscriber(script), nmt))

        self.assertEqual(finals_text(r), spoken(script))
        self.assertEqual("".join(r.commits), spoken(script))
        self.assertEqual(
            [u.source for u in r.finals],
            ["今日は友達と東京に行きました。", "すごく楽しかったです。"],
        )
        for utterance in r.finals:
            self.assertEqual(utterance.english, f"<{utterance.source}>")
            self.assertEqual(utterance.engine, "nmt")
            self.assertIsNotNone(utterance.latency)
        ids = [u.id for u in r.finals]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(r.states[-1][0], IDLE)
        self.assertIn(RECOGNIZING, [s for s, _ in r.states])
        self.assertIn(LISTENING, [s for s, _ in r.states])

    def test_sentence_is_cut_mid_speech_without_waiting_for_silence(self):
        # No pause anywhere: the only signal is the recogniser's punctuation.
        script = Script().silence(0.3)
        for i in range(6):
            script.say(f"文{i}", "です。", each=0.35)
        script.silence(1.5)
        cfg = test_config()
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator())
        )
        self.assertEqual(finals_text(r), spoken(script))
        self.assertGreaterEqual(len(r.finals), 3)
        for utterance in r.finals:
            self.assertTrue(utterance.source.endswith("。"))

    def test_dangling_utterance_waits_for_its_other_half(self):
        # A pause long enough to end the utterance, after a particle that
        # says the sentence is not over. The two halves must be translated as
        # one line.
        script = (
            Script()
            .silence(0.3)
            .say("これは", "日本", "で")
            .silence(0.7)
            .say("売って", "いない", "食べ物", "です")
            .silence(2.0)
        )
        cfg = test_config(long_pause_sec=1.2)
        asr = FakeTranscriber(script)
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        self.assertEqual(
            [u.source for u in r.finals], ["これは日本で売っていない食べ物です"]
        )
        # ...and heard as one: the audio of the first half was still in the
        # buffer when the second arrived, so the recogniser was never asked
        # to make sense of half a sentence by itself.
        self.assertGreaterEqual(pipeline.metrics.deferred_closes, 1)
        self.assertGreater(max(asr.buffer_secs), 0.9 + 1.2)

    def test_dangling_utterance_is_translated_anyway_after_a_long_pause(self):
        script = Script().silence(0.3).say("これは", "日本", "で").silence(3.0)
        cfg = test_config(long_pause_sec=0.8)
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator()), hold_open=2.0
        )
        self.assertEqual([u.source for u in r.finals], ["これは日本で"])

    def test_finished_sentence_does_not_wait_out_the_endpoint(self):
        # A long endpoint, as the Accuracy profile has. The sentence reads as
        # finished, so it is closed a quarter of a second into the pause -
        # and without a further pass over audio that has only gained silence.
        script = (
            Script()
            .silence(0.3)
            .say("今日は", "友達と", "東京に", "行き", "ました。", each=0.4)
            .silence(2.5)
        )
        cfg = test_config(
            endpoint_silence_sec=1.2, silence_flush_sec=1.2, eager_endpoint_sec=0.25
        )
        asr = FakeTranscriber(script)
        asr.device = "cuda"  # the cadence of a GPU run, not the CPU fallback's
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()), realtime=True)
        self.assertEqual([u.source for u in r.finals], [spoken(script)])
        # Well inside the 1.2 s the endpoint alone would have taken.
        self.assertLess(r.finals[0].latency, 0.9)
        # And once the line was taken, the silence that followed was not
        # sent to the recogniser to be made into words.
        last_speech = 0.3 + 5 * 0.4
        self.assertLess(max(asr.buffer_secs), last_speech + 0.9)

    def test_unfinished_sentence_still_waits_for_the_endpoint(self):
        script = (
            Script()
            .silence(0.3)
            .say("今日は", "友達と", "東京", each=0.4)
            .silence(2.5)
        )
        cfg = test_config(
            endpoint_silence_sec=0.8, silence_flush_sec=0.8, eager_endpoint_sec=0.25
        )
        pipeline, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator()), realtime=True
        )
        self.assertEqual([u.source for u in r.finals], [spoken(script)])
        self.assertEqual(pipeline.metrics.eager_endpoints, 0)

    def test_recogniser_that_stops_at_a_pause_is_moved_past_it(self):
        # The first phrase does not read as a sentence, so no line is cut at
        # the pause; and this recogniser, like the ones trained on single
        # sentences, reads up to a pause and no further. Left alone, nothing
        # after the pause would ever be recognised.
        script = (
            Script()
            .silence(0.3)
            .say("東京", "タワー", each=0.4)
            .silence(0.25)
            .say("の", "近く", "に", "ある", "店", "に", "行き", "ました。", each=0.3)
            .silence(1.5)
        )
        asr = FakeTranscriber(script, stops_at_pause=True)
        cfg = test_config(endpoint_silence_sec=0.5, silence_flush_sec=0.5)
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        self.assertEqual(finals_text(r), spoken(script))
        self.assertGreaterEqual(pipeline.metrics.stall_splits, 1)

    def test_recogniser_that_stops_mid_sentence_is_moved_on(self):
        # Six seconds without a pause, and a recogniser that never reads past
        # the first two and a half seconds of whatever it is given. There is
        # no pause to cut at; the buffer has to be cut where the reading
        # stopped, or everything after it goes unheard.
        script = Script().silence(0.3)
        for i in range(12):
            script.say(f"語{i}", "を", each=0.3)
        script.silence(1.5)
        asr = FakeTranscriber(script, reads_at_most=2.5)
        asr.device = "cuda"
        cfg = test_config(max_buffer_sec=20.0, mt_max_utterance_sec=30.0)
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        heard, said = finals_text(r), spoken(script)
        # In order, nothing repeated, and all of it but for whatever this
        # recogniser's last buffer held past its two and a half seconds -
        # under the second and a half that is worth going back for.
        self.assertTrue(said.startswith(heard), heard)
        self.assertGreaterEqual(len(heard), len(said) - 8)
        self.assertGreaterEqual(pipeline.metrics.stall_splits, 2)

    def test_finished_line_takes_its_audio_out_of_the_buffer(self):
        # Two sentences with a breath between them, too short to end the
        # utterance. Once the first is a line, its audio is cut away, so a
        # recogniser that stops at pauses goes straight on to the second.
        script = (
            Script()
            .silence(0.3)
            .say("東京に", "行き", "ました", each=0.4)
            .silence(0.25)
            .say("とても", "楽し", "かった", "です", each=0.4)
            .silence(1.5)
        )
        asr = FakeTranscriber(script, stops_at_pause=True)
        asr.device = "cuda"
        cfg = test_config(endpoint_silence_sec=0.5, silence_flush_sec=0.5, eager_endpoint_sec=0.0)
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        self.assertEqual(
            [u.source for u in r.finals], ["東京に行きました", "とても楽しかったです"]
        )
        self.assertGreaterEqual(pipeline.metrics.pause_splits, 1)
        self.assertEqual(pipeline.metrics.stall_splits, 0)

    def test_short_reply_is_not_lost(self):
        # Shorter than min_chunk_sec, so no ordinary pass ever runs on it.
        script = Script().silence(0.5).say("はい", each=0.35).silence(1.5)
        cfg = test_config(min_chunk_sec=1.0)
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator())
        )
        self.assertEqual([u.source for u in r.finals], ["はい"])

    def test_previews_are_provisional_and_never_enter_the_transcript(self):
        script = (
            Script()
            .silence(0.3)
            .say("今日は", "友達と", "一緒に", "東京に", "行って", "きたんだ", each=0.4)
            .silence(1.5)
        )
        nmt = FakeTranslator()
        cfg = test_config()
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), nmt), realtime=True
        )
        shown = [english for english, _src in r.previews if english]
        self.assertTrue(shown, "no preview was shown during a 2.4 s sentence")
        self.assertEqual(len(r.finals), 1)
        self.assertEqual(r.finals[0].source, spoken(script))
        # The preview is cleared once the real line has landed.
        self.assertEqual(r.previews[-1][0], "")

    def test_live_preview_off(self):
        script = Script().silence(0.3).say("今日は", "いい", "天気", "です。").silence(1.5)
        cfg = test_config(mt_live_enabled=False)
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator()), realtime=True
        )
        self.assertEqual([e for e, _ in r.previews if e], [])
        self.assertEqual(len(r.finals), 1)

    def test_no_translator_still_produces_japanese_lines(self):
        script = Script().silence(0.3).say("今日は", "いい", "天気", "です。").silence(1.5)
        cfg = test_config(mt_enabled=False)
        _p, r = run_pipeline(cfg, script, FakeModels(FakeTranscriber(script)))
        self.assertEqual([u.source for u in r.finals], [spoken(script)])
        self.assertEqual(r.finals[0].english, "")

    def test_fragment_waits_longer_than_a_sentence_does(self):
        # さんだ ends in だ and so reads as finished, but three characters on
        # their own are not something to translate in a hurry.
        script = (
            Script()
            .silence(0.3)
            .say("さん", "だ", each=0.3)
            .silence(0.6)
            .say("っけ", "と", "思い", "ます", each=0.3)
            .silence(2.0)
        )
        cfg = test_config(endpoint_silence_sec=0.4, silence_flush_sec=0.4, long_pause_sec=1.2)
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator())
        )
        self.assertEqual([u.source for u in r.finals], ["さんだっけと思います"])

    def test_boundary_echo_is_not_translated(self):
        # The recogniser hands back the previous sentence's last syllable as
        # an utterance of its own.
        script = (
            Script()
            .silence(0.3)
            .say("きれい", "だ", each=0.4)
            .silence(1.2)
            .say("だ", each=0.3)
            .silence(2.0)
        )
        cfg = test_config(long_pause_sec=0.6)
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator()), hold_open=2.5
        )
        self.assertEqual([u.source for u in r.finals], ["きれいだ"])

    def test_content_free_utterances_are_dropped(self):
        script = Script().silence(0.3).say("...", "…", each=0.4).silence(1.5)
        cfg = test_config()
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), FakeTranslator())
        )
        self.assertEqual(r.finals, [])


class Hybrid(unittest.TestCase):
    def test_llm_finishes_what_the_nmt_drafted(self):
        script = (
            Script()
            .silence(0.3)
            .say("今日は", "友達と", "東京に", "行き", "ました。", each=0.4)
            .silence(1.5)
        )
        nmt, llm = FakeTranslator("nmt"), FakeTranslator("llm", delay=0.2)
        cfg = test_config(mt_backend="llm")
        _p, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), nmt, llm), realtime=True
        )
        self.assertEqual(len(r.finals), 1)
        final = r.finals[0]
        self.assertEqual(final.engine, "llm")
        self.assertEqual(final.source, spoken(script))
        self.assertEqual(llm.calls, [spoken(script)])
        # The draft and the final are the same utterance, corrected in place.
        for draft in r.drafts:
            self.assertEqual(draft.id, final.id)
            self.assertFalse(draft.final)
        # The NMT model did the previewing; the LLM was never asked for one.
        self.assertTrue(nmt.calls)

    def test_missing_llm_falls_back_to_nmt_with_a_notice(self):
        script = Script().silence(0.3).say("今日は", "いい", "天気", "です。").silence(1.5)
        nmt = FakeTranslator("nmt")
        cfg = test_config(mt_backend="llm")
        models = FakeModels(
            FakeTranscriber(script), nmt, llm_error="Ollama is not answering."
        )
        _p, r = run_pipeline(cfg, script, models)
        self.assertEqual([u.engine for u in r.finals], ["nmt"])
        self.assertTrue(any("Ollama is not answering" in m for _lvl, m in r.notices))
        self.assertNotIn(ERROR, [s for s, _ in r.states])

    def test_missing_llm_is_an_error_when_fallback_is_off(self):
        script = Script().silence(0.3).say("今日は").silence(1.0)
        cfg = test_config(mt_backend="llm", mt_fallback_to_nmt=False)
        models = FakeModels(
            FakeTranscriber(script), FakeTranslator(), llm_error="Ollama is not answering."
        )
        _p, r = run_pipeline(cfg, script, models)
        self.assertEqual(r.states[-1][0], ERROR)
        self.assertIn("Ollama is not answering", r.states[-1][1])

    def test_llm_failing_mid_session_hands_over_to_nmt(self):
        script = Script().silence(0.3)
        for i in range(5):
            script.say(f"文{i}", "です。", each=0.35).silence(0.8)
        script.silence(1.0)
        nmt, llm = FakeTranslator("nmt"), FakeTranslator("llm")
        llm.fail = True
        cfg = test_config(mt_backend="llm")
        pipeline, r = run_pipeline(
            cfg, script, FakeModels(FakeTranscriber(script), nmt, llm)
        )
        self.assertEqual(finals_text(r), spoken(script))
        self.assertTrue(all(u.engine == "nmt" for u in r.finals))
        self.assertTrue(all(u.english for u in r.finals))
        # It stops asking the dead model after three failures.
        self.assertEqual(len(llm.calls), 3)
        self.assertGreaterEqual(pipeline.metrics.fallback_lines, 5)
        self.assertTrue(any(level == "warning" for level, _ in r.notices))


class Robustness(unittest.TestCase):
    def test_slow_recogniser_does_not_fall_further_and_further_behind(self):
        # Each pass takes longer than the interval between passes. The old
        # loop queued the difference without limit; this one must make each
        # pass cover everything that has arrived.
        script = Script().silence(0.3)
        for i in range(8):
            script.say(f"文{i}", "です。", each=0.4)
        script.silence(2.0)
        asr = FakeTranscriber(script, delay=0.5)
        cfg = test_config(process_interval_sec=0.25)
        t0 = time.perf_counter()
        pipeline, r = run_pipeline(
            cfg, script, FakeModels(asr, FakeTranslator()), realtime=True
        )
        self.assertEqual(finals_text(r), spoken(script))
        # 6.4 s of speech at one pass per 0.5 s is about 13 passes. Queueing
        # one per 0.25 s would be 26, and would finish far behind the audio.
        self.assertLess(asr.calls, 20)
        self.assertLess(pipeline.metrics.backlog_max_sec, 1.5)
        self.assertEqual(pipeline.metrics.backlog_drops, 0)
        audio_sec = script.audio().size / 16000
        self.assertLess(time.perf_counter() - t0, audio_sec + 3.0)

    def test_failed_pass_is_survived_and_reported(self):
        script = (
            Script()
            .silence(0.3)
            .say("最初", "です。", each=0.4)
            .silence(1.0)
            .say("次", "です。", each=0.4)
            .silence(1.5)
        )
        asr = FakeTranscriber(script)
        asr.fail_next = 1
        cfg = test_config()
        pipeline, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        self.assertNotIn(ERROR, [s for s, _ in r.states])
        self.assertEqual(pipeline.metrics.recoveries, 1)
        self.assertTrue(any("out of memory" in m for _lvl, m in r.notices))
        self.assertIn("次です。", [u.source for u in r.finals])

    def test_repeated_failure_stops_with_an_error_that_says_why(self):
        script = Script().silence(0.3)
        for i in range(6):
            script.say(f"文{i}", "です。", each=0.4).silence(0.7)
        asr = FakeTranscriber(script)
        asr.fail_next = 99
        cfg = test_config()
        _p, r = run_pipeline(cfg, script, FakeModels(asr, FakeTranslator()))
        self.assertEqual(r.states[-1][0], ERROR)
        self.assertIn("out of memory", r.states[-1][1])

    def test_pause_discards_and_resume_starts_clean(self):
        script = (
            Script()
            .silence(0.3)
            .say("聞こえ", "ない", "はず", each=0.5)
            .silence(1.0)
            .say("聞こえ", "ます。", each=0.5)
            .silence(1.5)
        )
        events = RecordingEvents()
        source = ScriptSource(script.audio(), realtime=True, hold_open=1.5)
        cfg = test_config()
        pipeline = CaptionPipeline(
            cfg,
            events=events,
            source=source,
            models=FakeModels(FakeTranscriber(script), FakeTranslator()),
        )
        pipeline.set_paused(True)
        pipeline.start()
        time.sleep(2.2)  # the first sentence goes by while paused
        pipeline.set_paused(False)
        deadline = time.monotonic() + 15
        while pipeline.running and time.monotonic() < deadline:
            time.sleep(0.02)
        pipeline.stop()
        r = events.r
        self.assertIn(PAUSED, [s for s, _ in r.states])
        self.assertEqual([u.source for u in r.finals], ["聞こえます。"])

    def test_listener_that_raises_does_not_stop_the_run(self):
        script = Script().silence(0.3).say("今日は", "いい", "天気", "です。").silence(1.5)

        class Bad(RecordingEvents):
            def on_caption(self, committed, unstable):
                raise ValueError("listener bug")

        events = Bad()
        source = ScriptSource(script.audio(), hold_open=1.5)
        pipeline = CaptionPipeline(
            test_config(),
            events=events,
            source=source,
            models=FakeModels(FakeTranscriber(script), FakeTranslator()),
        )
        pipeline.start()
        deadline = time.monotonic() + 15
        while pipeline.running and time.monotonic() < deadline:
            time.sleep(0.02)
        pipeline.stop()
        self.assertEqual(len(events.r.finals), 1)

    def test_stop_is_prompt(self):
        script = Script().silence(0.3).say(*["語"] * 40, each=0.3)
        events = RecordingEvents()
        source = ScriptSource(script.audio(), realtime=True)
        pipeline = CaptionPipeline(
            test_config(),
            events=events,
            source=source,
            models=FakeModels(FakeTranscriber(script), FakeTranslator()),
        )
        pipeline.start()
        time.sleep(1.0)
        t0 = time.perf_counter()
        pipeline.stop()
        self.assertLess(time.perf_counter() - t0, 3.0)
        self.assertFalse(pipeline.running)


if __name__ == "__main__":
    unittest.main()
