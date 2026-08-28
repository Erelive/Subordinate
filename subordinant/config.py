"""Runtime configuration.

Defaults are tuned for an 8 GB laptop GPU. Values can be overridden by a JSON
file at %APPDATA%\\SubOrdinant\\config.json; any key not present there keeps the
default below.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

# The model rate Whisper expects. Not configurable - changing it breaks the model.
SAMPLE_RATE = 16000

# Transcription models offered in the GUI, as config overrides. Separate from
# ENGINES in window.py because the two choices are genuinely independent: this
# is what turns audio into text, that is what translates the text afterwards.
# The one exception is Whisper's own task="translate", where a single model does
# both jobs and so dictates this one - see ENGINE_TRANSCRIBER in window.py.
#
# beam_size belongs here rather than with the engine because it is a property of
# the acoustic model: large-v3 cannot afford beam 5 and turbo can. See the note
# on beam_size below.
TRANSCRIBERS: dict[str, dict] = {
    "Whisper large-v3-turbo": {
        "model": "large-v3-turbo",
        "beam_size": 5,
    },
    # Distil-Whisper fine-tuned on Japanese by Kotoba Technologies. Four decoder
    # layers' worth of Whisper large has been cut to two, so it is *faster* than
    # turbo rather than slower - measured on a 12 s buffer, 184 ms against
    # turbo's 210 ms (RTF 0.37 against 0.42) - while being trained specifically
    # on Japanese speech.
    #
    # This is v2.0 and not v2.2 on purpose. v2.1 and v2.2 add punctuation and
    # pyannote speaker diarization *around* the same acoustic weights, in a
    # transformers pipeline that CTranslate2 cannot carry; every
    # "kotoba-whisper-v2.2-faster" conversion on the Hub is a byte-identical
    # copy of this repo (verified by sha256). The parts v2.2 adds are parts this
    # program already does for itself, in diarize.py and the sentence splitter.
    "Kotoba-Whisper v2.0 (Japanese)": {
        "model": "kotoba-tech/kotoba-whisper-v2.0-faster",
        # Set to 10 by request. Measured at a full 12 s buffer, per pass against
        # 0.5 s of fresh audio:
        #
        #     beam 1    141 ms   RTF 0.28
        #     beam 5    186 ms   RTF 0.37
        #     beam 10   421 ms   RTF 0.84   <- here
        #     beam 20  1169 ms   RTF 2.34
        #
        # 0.84 is the same place large-v3 sits, and it is the edge: it leaves
        # ~16% headroom on an idle machine, so a game on the other monitor
        # pushes it past 1.0, the backlog stops draining, and drop_backlog()
        # discards audio that is then never transcribed at all. If captions
        # start missing words under load, this is the first thing to lower.
        #
        # Worth knowing what it buys: on clean single-speaker audio, nothing -
        # beam 5, 10 and 20 returned identical text. The case for it is the one
        # measured for turbo under beam_size below, where beam search cut word
        # loss on *overlapping* speech from 39% to 15%; that is not tested here,
        # so whether 10 beats 5 on a real collab is unmeasured either way.
        "beam_size": 5,
    },
    "Whisper large-v3": {
        "model": "large-v3",
        # Beam 1, unlike the turbo paths, and not for lack of benefit but for
        # lack of room. large-v3 is already an expensive pass; measured at a
        # full 12 s buffer, beam 5 puts it at RTF 1.02 on its own - past the
        # point where the backlog stops draining and audio is discarded.
        #
        #     beam 1   425 ms   0.85      beam 3   413 ms   0.83
        #     beam 5   511 ms   1.02  <- over
        #
        # Beam search pays when the model is uncertain, which is why it rescued
        # turbo on overlapping speech (39% -> 15% word loss). large-v3 is a
        # stronger model and gains least from it, so this is the cheapest place
        # to give it up.
        "beam_size": 1,
    },
}

# Models trained on one language only. Whisper's language token still exists on
# these and setting it to anything else is accepted rather than rejected, so the
# failure is silent bad transcription - hence the warning in asr.py.
MONOLINGUAL: dict[str, str] = {
    "kotoba-tech/kotoba-whisper-v2.0-faster": "ja",
}

# Distil-Whisper models keep the full 32-layer encoder but cut the decoder to
# two layers, and every CTranslate2 conversion of one on the Hub ships the
# alignment_heads of the model it was distilled *from* - pairs naming decoder
# layers 7 through 25. Word timestamps cross-attend to those layers, so
# faster-whisper indexes past the end of a 2-layer decoder and the process dies
# with a segfault rather than an exception. This program needs word timestamps
# for LocalAgreement, so the config has to be repaired before the model loads -
# see repair_alignment_heads() in asr.py.
DECODER_LAYERS: dict[str, int] = {
    "kotoba-tech/kotoba-whisper-v2.0-faster": 2,
}


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "SubOrdinant"


@dataclass
class Config:
    # --- model ---
    model: str = "large-v3-turbo"
    device: str = "cuda"  # "cuda" or "cpu"
    compute_type: str = "float16"  # "int8_float16" halves VRAM, slightly worse
    # Source language of the audio, not of the captions. "ja" with
    # task="translate" gives Japanese audio -> English captions.
    language: str = "en"
    # "translate" always targets English; Whisper has no other target. Needs a
    # full model - see translate_warning() in asr.py. Prefer mt_enabled below:
    # a dedicated NMT model translates better and costs less than this does.
    task: str = "transcribe"
    # Beam search. This was 1 on the grounds that beams cost latency for little
    # gain - which was measured on clean single-speaker audio, where it is true
    # and useless. On overlapping speech, which is what a collab actually is,
    # greedy decoding loses 39% of the words and beam 5 loses 15%:
    #
    #     beam 1   39% lost   146 ms at a full 12 s buffer   RTF 0.29
    #     beam 2   24% lost
    #     beam 3   24% lost
    #     beam 5   15% lost   216 ms                         RTF 0.43
    #
    # Clean audio is unchanged either way (203/204 words). RTF 0.43 leaves ample
    # headroom, so the cost is latency that was never the binding constraint.
    beam_size: int = 5
    # Text shown to Whisper before each pass, to bias it towards words it would
    # otherwise mishear. Proper nouns are the case that matters: a name it has
    # never seen comes out as whatever it sounds like, and the translation stage
    # then faithfully translates the wrong word, so the error is invisible by
    # the time anyone reads it.
    #
    # Keep it short. The prompt occupies the same decoder context the audio
    # competes for, and a long one makes Whisper start reciting it instead of
    # transcribing - the failure looks like hallucinated captions.
    initial_prompt: str = ""

    # --- translation ---
    # Translate committed utterances with a dedicated NMT model instead of
    # Whisper's task="translate". Captions then carry both the source text and
    # the English. See subordinant/translate.py for why this is the cheaper
    # route despite being a second model.
    mt_enabled: bool = False
    mt_model: str = "entai2965/sugoi-v4-ja-en-ctranslate2"
    # Which SentencePiece pair to load out of the model's spm/ directory.
    mt_source_lang: str = "ja"
    mt_target_lang: str = "en"
    # The search converges at 10; above that the beam only costs. Over 12
    # conversational utterances, beam 10, 20 and 32 returned byte-identical
    # output at a flat mean logprob (-0.4687, -0.4687, -0.4685). Only beam 1 is
    # meaningfully worse - it differs from beam 5 on 7 of the 12, logprob
    # -0.4857. Between 5 and 10 there are 3 changes, none of them clear wins.
    #
    # Cost on an RTX 4080 SUPER, mean ms per call at float32:
    #
    #     beam 1   23.5 committed   16.1 live tail
    #     beam 5   30.9             26.8
    #     beam 10  67.2             48.1
    #     beam 20 149.1            111.6   <- 2.5x beam 10 for identical text
    #
    # Cost scales with beam width and no setting avoids that: a 300M model on a
    # batch of one is latency-bound on sequential decode steps rather than
    # throughput-bound, so the idle GPU cannot absorb a wider beam. float16 buys
    # ~15%, int8 nothing, and beam 50 falls off a cliff entirely (745 ms).
    #
    # The live tail is what makes any of this a real cost - it runs every pass,
    # not once per utterance, so at 0.5 s passes beam 10 is 0.09 of the
    # real-time budget against Whisper's 0.31. If it ever needs to come down,
    # lower the preview rather than the committed pass: the preview is a guess
    # that rewrites itself anyway, while committed text amortises over ~8 passes
    # and is nearly free at any beam.
    mt_beam_size: int = 20
    # NMT models degenerate into repetition loops when handed input they cannot
    # parse, and garbled ASR off a noisy stream is exactly that. Without these an
    # utterance can decode into hundreds of repeated words. The cap is the
    # backstop; the penalties stop it well before reaching it.
    mt_repetition_penalty: float = 1.2
    mt_no_repeat_ngram_size: int = 3
    mt_max_decoding_length: int = 128
    # Translation normally fires on a sentence ending. Continuous speech - a
    # conversation between several people, say - may never produce one, and
    # waiting for silence would mean no English at all while anyone is talking.
    # This is the backstop: translate whatever has accumulated once it spans
    # this long, sentence boundary or not.
    mt_max_utterance_sec: float = 3.0
    # Don't cut a translation unit at a change of voice if what comes before it
    # is shorter than this. A word or two stranded ahead of a speaker change is
    # nearly always the tail of the previous turn caught by a boundary window,
    # not a real turn of its own; translating it alone yields nothing useful and
    # consumes the text, so the caption goes missing entirely.
    mt_min_chunk_sec: float = 0.8
    # Translate the unfinished tail every pass and show it as a live preview,
    # so English moves continuously instead of arriving a sentence at a time.
    #
    # This cannot go in the transcript, and the reason is structural rather than
    # a matter of polish. LocalAgreement works because ASR output is monotonic:
    # a committed prefix stays put. Translation is not monotonic, and Japanese is
    # near the worst case - the verb is last, and negation and tense are suffixes
    # on it, so a partial sentence can translate to the opposite of the finished
    # one. Measured with Sugoi:
    #
    #     これは面白い          -> This is interesting.
    #     これは面白くない      -> This isn't funny.
    #     これは面白くないと思わない -> I don't think this is funny.
    #
    # So the preview is shown where rewriting is already the contract - beside
    # the unconfirmed source tail, dimmed - while the transcript keeps taking
    # whole utterances and never rewrites.
    mt_live_enabled: bool = True
    # Don't preview less than this much source text: a two-word fragment
    # translates to noise and just makes the line flicker.
    mt_live_min_chars: int = 4

    # --- speaker labelling ---
    # Label each utterance with who said it, and cut translation units at a
    # change of voice. See subordinant/diarize.py for why the labels drift.
    diarize_enabled: bool = False
    # One file out of csukuangfj/speaker-embedding-models. CAM++ trained on
    # Chinese and English, 28 MB: the closest thing on offer to multilingual,
    # which matters because the audio languages here are ja/zh/ko as well as en.
    # Speaker embeddings key on timbre rather than phonetics, so they transfer
    # across languages better than an ASR model would.
    diarize_model: str = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    # How much speech goes into one embedding. This is the single biggest lever
    # on whether one person gets mistaken for two. Measured on two real voices,
    # the *lowest* same-speaker similarity was 0.24 at a 1 s window, 0.61 at
    # 1.5 s, 0.74 at 2 s and 0.85 at 3 s, while the highest between-speaker
    # similarity barely moved (0.25 -> 0.21). Short windows are what make the
    # same voice look like a stranger.
    # ...but only up to a point, and the point is set by how fast people take
    # turns. A window longer than a turn straddles the hand-off, so the
    # embedding is a blend of two voices - and because blends are folded into
    # the running centroid, the centroids converge until everyone matches
    # speaker 1. Measured on two voices alternating, speakers found (want 2):
    #
    #     window/hop     8 s turns   4 s   2.5 s   1.5 s
    #     3.0 / 1.00         2        1      1       1     <- collapses
    #     2.5 / 0.75         2        2      1       1
    #     2.0 / 0.75         2        2      2       2
    #     1.5 / 1.00         2        2      2       2
    #
    # 2 s is the longest window that survives fast dialogue, and its voiceprints
    # are markedly better than 1.5 s (0.74 against 0.61), so it takes both.
    diarize_window_sec: float = 2.0
    # How much stream time each label covers. Below the window, so a change of
    # voice is still placed finely while being decided on more context.
    diarize_hop_sec: float = 0.75
    # Never classify on less audio than this. Below it, same-speaker similarity
    # is no better than between-speaker, so a label would be a coin flip - and a
    # wrong one opens a speaker that never goes away.
    diarize_min_embed_sec: float = 2.0
    # Cosine similarity at which a window is judged to be an existing speaker
    # rather than a new one. At the 3 s window above, same-speaker stayed above
    # 0.85 and different-speaker below 0.21, so 0.5 sits in open space. That was
    # two TTS voices, male against female - the easy case. Lower this if one
    # person keeps turning into several; raise it if two people share a label.
    diarize_threshold: float = 0.5
    # Past this many, the closest match wins instead of a new speaker opening.
    # The most effective setting here when the answer is known: three people in
    # a collab means 3, and no amount of shouting can then invent a fourth.
    diarize_max_speakers: int = 6
    # Windows quieter than this never reach the model. Room tone and music
    # produce embeddings that are not about a voice, and admitting them drags
    # the centroids together until distinct speakers merge.
    diarize_min_rms: float = 0.005

    # --- streaming ---
    # How much fresh audio to accumulate before running inference again. Lower
    # means more responsive captions but more GPU work per second of audio.
    process_interval_sec: float = 0.5
    # Don't run inference at all until this much speech exists; very short
    # buffers produce garbage and hallucinations.
    min_chunk_sec: float = 1.0
    # Hard cap on the rolling buffer. Longer context is more accurate but each
    # inference re-transcribes the whole buffer, so cost grows with length and a
    # long buffer is what pushes RTF towards 1.0 during continuous speech.
    max_buffer_sec: float = 12.0
    # Silence this long ends the utterance: commit everything, clear the buffer.
    silence_flush_sec: float = 0.7
    # If the worker falls this far behind real time, drop the backlog rather
    # than let caption latency grow without bound.
    max_backlog_sec: float = 6.0

    # --- voice activity ---
    # Cheap pre-gate only: decides whether audio is worth sending to the GPU.
    # The real speech/non-speech decision is faster-whisper's bundled Silero VAD.
    energy_gate_db: float = -48.0
    vad_filter: bool = True

    # --- window ---
    # The main window is the default surface. The overlay draws on the primary
    # screen and cannot know better - system audio is one mixed stream with no
    # screen affinity - so on a multi-monitor setup it puts captions for a video
    # on one screen over whatever is on the other, and it cannot draw over an
    # exclusive-fullscreen game at all. Enable it for single-screen full-screen
    # viewing, where there is nothing to collide with.
    overlay_enabled: bool = False
    window_opacity: int = 100  # percent; the window's slider writes this back
    window_on_top: bool = False

    # --- overlay ---
    font_family: str = "Segoe UI"
    font_size: int = 26
    # Fraction of screen width the caption box may occupy before wrapping.
    max_width_frac: float = 0.72
    # Distance from the bottom of the screen, as a fraction of screen height.
    bottom_margin_frac: float = 0.08
    committed_color: str = "#ffffff"
    unstable_color: str = "#9aa0a6"
    background_rgba: str = "rgba(0, 0, 0, 190)"
    # How much committed text to keep on screen, in seconds of speech.
    display_history_sec: float = 12.0
    # Hide the overlay entirely after this long with nothing to show.
    hide_after_idle_sec: float = 4.0

    # --- audio ---
    # Capture buffer size in frames, at the device's native rate. Smaller means
    # lower capture latency and more callbacks.
    frames_per_buffer: int = 1024

    @classmethod
    def load(cls) -> "Config":
        path = config_dir() / "config.json"
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable config at %s: %s", path, exc)
            return cfg
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                log.warning("unknown config key %r ignored", key)
        return cfg

    def save(self) -> Path:
        """Write the settings that differ from the defaults, and only those.

        Writing every field looks harmless and is not: the app saves on exit, so
        one run freezes the whole default set into the user's config, and every
        default tuned afterwards is silently overridden on their machine
        forever. That happened - diarize_window_sec was raised from 1.5 s to 3 s
        to stop one speaker being split into several, and installs that had run
        once kept the old value and the old bug.

        Recording only what was actually chosen keeps the file a statement of
        intent, so defaults stay live.
        """
        path = config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        defaults = type(self)()
        chosen = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != getattr(defaults, f.name)
        }
        # ensure_ascii=False so a Japanese name glossary stays readable in the
        # file rather than turning into a wall of \u escapes.
        path.write_text(
            json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path
