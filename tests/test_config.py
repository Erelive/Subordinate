import json
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from subordinant.config import TRANSCRIBERS, Config
from subordinant.profiles import (
    PROFILES,
    TRANSLATORS,
    apply_profile,
    detect_profile,
    recognizer_name,
    translator_name,
)


class ConfigFile(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._old = os.environ.get("APPDATA")
        os.environ["APPDATA"] = self._dir.name
        self.path = Path(self._dir.name) / "SubOrdinant" / "config.json"
        self.path.parent.mkdir(parents=True)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = self._old
        self._dir.cleanup()

    def write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def test_missing_file_gives_defaults(self):
        self.assertEqual(Config.load(), Config())

    def test_round_trip_writes_only_what_changed(self):
        cfg = Config()
        cfg.font_size = 31
        cfg.initial_prompt = "天音 = Amane"
        cfg.save()
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved, {"font_size": 31, "initial_prompt": "天音 = Amane"})
        loaded = Config.load()
        self.assertEqual(loaded.font_size, 31)
        self.assertEqual(loaded.initial_prompt, "天音 = Amane")
        self.assertEqual(loaded.load_notes, [])

    def test_corrupt_file_is_kept_aside_and_reported(self):
        self.write('{"font_size": 31,')
        cfg = Config.load()
        self.assertEqual(cfg, Config())
        self.assertEqual(len(cfg.load_notes), 1)
        self.assertIn("could not be read", cfg.load_notes[0])
        self.assertFalse(self.path.exists())
        self.assertEqual(len(list(self.path.parent.glob("config.corrupt-*.json"))), 1)

    def test_not_an_object(self):
        self.write("[1, 2, 3]")
        cfg = Config.load()
        self.assertEqual(cfg, Config())
        self.assertTrue(cfg.load_notes)

    def test_wrong_types_fall_back_one_key_at_a_time(self):
        self.write(
            json.dumps(
                {
                    "beam_size": "five",
                    "mt_enabled": "yes",
                    "process_interval_sec": None,
                    "font_size": 40,
                    "theme": "neon",
                }
            )
        )
        cfg = Config.load()
        defaults = Config()
        self.assertEqual(cfg.beam_size, defaults.beam_size)
        self.assertEqual(cfg.mt_enabled, defaults.mt_enabled)
        self.assertEqual(cfg.process_interval_sec, defaults.process_interval_sec)
        self.assertEqual(cfg.theme, defaults.theme)
        self.assertEqual(cfg.font_size, 40)  # the valid one still applies
        self.assertEqual(len(cfg.load_notes), 1)
        for key in ("beam_size", "mt_enabled", "process_interval_sec", "theme"):
            self.assertIn(key, cfg.load_notes[0])

    def test_numbers_are_clamped_not_rejected(self):
        self.write(json.dumps({"process_interval_sec": 0, "beam_size": 500, "overlay_opacity": -5}))
        cfg = Config.load()
        self.assertEqual(cfg.process_interval_sec, 0.1)
        self.assertEqual(cfg.beam_size, 20)
        self.assertEqual(cfg.overlay_opacity, 0)
        self.assertEqual(cfg.load_notes, [])

    def test_int_where_float_is_expected(self):
        self.write(json.dumps({"max_buffer_sec": 9, "beam_size": 3.0}))
        cfg = Config.load()
        self.assertEqual(cfg.max_buffer_sec, 9.0)
        self.assertIsInstance(cfg.max_buffer_sec, float)
        self.assertEqual(cfg.beam_size, 3)
        self.assertIsInstance(cfg.beam_size, int)

    def test_retired_and_unknown_keys_are_ignored(self):
        self.write(json.dumps({"mt_async": True, "window_opacity": 80, "nonsense": 1}))
        cfg = Config.load()
        self.assertEqual(cfg, Config())
        self.assertEqual(cfg.load_notes, [])

    def test_save_is_atomic(self):
        Config().save()
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_copy_is_independent(self):
        cfg = Config()
        clone = cfg.copy()
        clone.beam_size = 9
        self.assertNotEqual(cfg.beam_size, 9)


class Defaults(unittest.TestCase):
    def test_japanese_to_english_out_of_the_box(self):
        cfg = Config()
        self.assertEqual(cfg.language, "ja")
        self.assertEqual(cfg.task, "transcribe")
        self.assertTrue(cfg.mt_enabled)
        self.assertEqual(cfg.mt_source_lang, "ja")
        self.assertEqual(cfg.mt_target_lang, "en")

    def test_defaults_are_the_balanced_profile(self):
        self.assertEqual(detect_profile(Config()), "balanced")
        self.assertEqual(Config().profile, "balanced")


class Profiles(unittest.TestCase):
    def test_three_profiles(self):
        self.assertEqual(set(PROFILES), {"fast", "balanced", "accuracy"})

    def test_profiles_set_the_same_keys(self):
        keys = [set(values) for values in PROFILES.values()]
        union = set().union(*keys)
        # A key one profile sets and another omits would survive a switch
        # between them, silently. llm_model is the exception: Fast has no LLM.
        for name, values in PROFILES.items():
            self.assertLessEqual(union - set(values), {"llm_model"}, name)

    def test_every_key_is_a_real_setting(self):
        known = {f.name for f in fields(Config)}
        for name, values in PROFILES.items():
            self.assertLessEqual(set(values), known, name)

    def test_profiles_are_not_cosmetic(self):
        configs = {name: apply_profile(Config(), name) for name in PROFILES}
        fast, balanced, accuracy = (configs[n] for n in ("fast", "balanced", "accuracy"))
        # Different translation engines...
        self.assertEqual(fast.mt_backend, "ct2")
        self.assertEqual(balanced.mt_backend, "llm")
        self.assertEqual(accuracy.mt_backend, "llm")
        self.assertNotEqual(balanced.llm_model, accuracy.llm_model)
        # ...and progressively more patience before committing to a sentence.
        for key in (
            "process_interval_sec",
            "endpoint_silence_sec",
            "long_pause_sec",
            "mt_max_utterance_sec",
            "max_buffer_sec",
        ):
            self.assertLess(getattr(fast, key), getattr(balanced, key), key)
            self.assertLess(getattr(balanced, key), getattr(accuracy, key), key)
        self.assertLess(fast.beam_size, balanced.beam_size)
        self.assertLess(balanced.llm_context_utterances, accuracy.llm_context_utterances)

    def test_apply_then_detect(self):
        for name in PROFILES:
            cfg = apply_profile(Config(), name)
            self.assertEqual(detect_profile(cfg), name)
            self.assertEqual(cfg.profile, name)

    def test_any_hand_edit_is_custom(self):
        cfg = apply_profile(Config(), "fast")
        cfg.beam_size = 3
        self.assertEqual(detect_profile(cfg), "custom")

    def test_switching_profiles_leaves_nothing_behind(self):
        cfg = apply_profile(Config(), "accuracy")
        apply_profile(cfg, "fast")
        self.assertEqual(detect_profile(cfg), "fast")
        self.assertEqual(cfg.no_repeat_ngram_size, 0)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            apply_profile(Config(), "ludicrous")

    def test_profile_models_are_in_the_catalogue(self):
        models = {entry["model"] for entry in TRANSCRIBERS.values()}
        for name in PROFILES:
            cfg = apply_profile(Config(), name)
            self.assertIn(cfg.model, models)
            self.assertIn(recognizer_name(cfg), TRANSCRIBERS)
            self.assertIn(translator_name(cfg), TRANSLATORS)

    def test_translator_catalogue_round_trips(self):
        for name, entry in TRANSLATORS.items():
            cfg = Config()
            for key, value in entry["set"].items():
                setattr(cfg, key, value)
            self.assertEqual(translator_name(cfg), name)


if __name__ == "__main__":
    unittest.main()
