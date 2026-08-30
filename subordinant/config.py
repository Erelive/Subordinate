"""Runtime configuration.

Defaults are tuned for an 8 GB laptop GPU. Values can be overridden by a JSON
file at %APPDATA%\\SubOrdinant\\config.json; any key not present there keeps the
default below.
"""

from __future__ import annotations

import json
import logging
import os
import sys
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
        "no_repeat_ngram_size": 0,
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
        # Measured at a full 12 s buffer. The ms are a property of the model;
        # the RTF is against process_interval_sec, now 0.35 s rather than the
        # 0.5 s these were first quoted at:
        #
        #     beam 1    141 ms   RTF 0.40
        #     beam 5    186 ms   RTF 0.53   <- here
        #     beam 10   421 ms   RTF 1.20
        #     beam 20  1169 ms   RTF 3.34
        #
        # Beam 10 was affordable at a 0.5 s interval (0.84) and is not at 0.35:
        # the backlog stops draining, and once max_backlog_sec of it piles up
        # drop_backlog() discards audio that is then never transcribed at all.
        # If captions start missing words under load, raise the interval first -
        # this beam is the overlapping-speech insurance described below.
        #
        # Worth knowing what it buys: on clean single-speaker audio, nothing -
        # beam 5, 10 and 20 returned identical text. The case for it is the one
        # measured for turbo under beam_size below, where beam search cut word
        # loss on *overlapping* speech from 39% to 15%; that is not tested here,
        # so whether 10 beats 5 on a real collab is unmeasured either way.
        "beam_size": 5,
        "no_repeat_ngram_size": 0,
    },
    # litagin/anime-whisper, fine-tuned from the same Kotoba v2.0 weights as
    # the entry above on ~5,300 hours of Japanese game and anime voice acting.
    # Identical shape: 32 encoder layers, 2 decoder layers, d_model 1280.
    #
    # The same shape does NOT mean the same cost, which is what this entry
    # first claimed. Measured on a 4080 SUPER with Sugoi 14B resident, beam 5,
    # median of four passes over real Japanese speech, by buffer length:
    #
    #                  2 s     4 s     8 s    12 s     RTF at 12 s
    #     Kotoba      105ms   128ms   125ms   163ms       0.47
    #     Anime       184ms   274ms   213ms   407ms       1.16
    #
    # Two and a half times Kotoba at a full buffer, and over the 0.35 s
    # interval. The weights are the same size; the output is not. This model
    # was trained to emit rich punctuation, ellipses and non-verbal sounds,
    # and an autoregressive decoder pays per token it writes.
    #
    # What that means in practice: a buffer that reaches max_buffer_sec under
    # sustained speech runs a backlog, the same failure large-v3 has. It is
    # fine in ordinary use because utterances end and the buffer resets long
    # before 12 s. If captions start dropping words during a long unbroken
    # stretch, this is why - raise process_interval_sec or switch to Kotoba.
    #
    # On the author's held-out visual-novel set it reads 13.0% CER against
    # Kotoba v2.0's 18.8% and large-v3's 16.5%. That set is in-domain for it
    # and out of domain for the other two, so it is the ceiling rather than
    # the expected gain: a live collab is unscripted, mic'd and overlapping,
    # and none of that is in 5,300 hours of studio voice acting.
    #
    # A third-party CTranslate2 conversion, because there is no official one.
    # It inherits the same mis-copied alignment_heads Kotoba ships - pairs
    # naming decoder layers 7-25 on a decoder that has two - which is why it
    # needs the DECODER_LAYERS entry below. Without it, word timestamps are an
    # out-of-bounds read inside CTranslate2 and the process segfaults. Checked
    # by reading the conversion's config.json, not assumed from the base model.
    #
    # Two ways it behaves unlike every other model here, both from its card:
    #
    #   - An initial prompt *degrades* it - hallucinations and worse text -
    #     where everywhere else a name in the prompt is what stops the decoder
    #     guessing. The Names box wants to be empty on this route; see
    #     prompt_warning() in asr.py, which says so at load.
    #
    #   - It usually omits the sentence-final 。, and _sentence_cut() in
    #     pipeline.py cuts utterances on exactly that. Expect it to lean on the
    #     clause marks and the mt_max_utterance_sec timeout more than the
    #     others do, which shows up as translation pacing rather than as an
    #     error.
    "Anime-Whisper (Japanese)": {
        "model": "flyfront/anime-whisper-faster",
        # Not measured here. Kotoba is the closest guide - same shape, same
        # size, same cost per pass - and beam 5 is where that entry landed.
        "beam_size": 5,
        # The one model here that needs it; see the field in Config. Its card
        # benchmarks at 5, and without it emotive delivery decays into stutter
        # loops that reach the captions as real-looking text. Measured free:
        # 407 ms with it against 413 ms without, at a 12 s buffer.
        "no_repeat_ngram_size": 5,
    },
    "Whisper large-v3": {
        "model": "large-v3",
        # Beam 1, unlike the turbo paths, and not for lack of benefit but for
        # lack of room. large-v3 is already an expensive pass. At a full 12 s
        # buffer, against the 0.35 s process_interval_sec:
        #
        #     beam 1   425 ms   RTF 1.21      beam 3   413 ms   RTF 1.18
        #     beam 5   511 ms   RTF 1.46
        #
        # All of them are over 1.0, beam 1 included - this route no longer fits
        # the interval, where at 0.5 s it just did (0.85). A later interleaved
        # run measured beam 1 lower, at 371 ms, which is still RTF 1.06. So this
        # engine now runs a persistent backlog and will drop audio under
        # sustained speech. It was already the route the README argues against -
        # Sugoi is roughly three times cheaper and better on proper nouns - but
        # anyone choosing it should raise process_interval_sec back to 0.5.
        #
        # Beam search pays when the model is uncertain, which is why it rescued
        # turbo on overlapping speech (39% -> 15% word loss). large-v3 is a
        # stronger model and gains least from it, so this is the cheapest place
        # to give it up.
        "beam_size": 1,
        "no_repeat_ngram_size": 0,
    },
}

# Models trained on one language only. Whisper's language token still exists on
# these and setting it to anything else is accepted rather than rejected, so the
# failure is silent bad transcription - hence the warning in asr.py.
MONOLINGUAL: dict[str, str] = {
    "kotoba-tech/kotoba-whisper-v2.0-faster": "ja",
    "flyfront/anime-whisper-faster": "ja",
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
    # Fine-tuned from the above, and the conversion carries that config over
    # unchanged - same two decoder layers, same heads naming layers 7-25.
    "flyfront/anime-whisper-faster": 2,
}

# CTranslate2 conversions published without a tokenizer.json, mapped to the
# model whose tokenizer is the correct one for them.
#
# faster-whisper falls back to openai/whisper-tiny's tokenizer when a model
# directory has none. For anything descended from large-v3 that is the wrong
# vocabulary by exactly one token: large-v3 added <|yue|> at id 50358, so
# whisper-tiny's 51865-entry vocabulary puts <|translate|> there instead and
# every id above it lands one position off - the task tokens, <|notimestamps|>
# and the entire timestamp range.
#
# Ordinary text sits below 50358 and survives, which is why the failure reads
# as scrambled captions with meaningless word timings rather than as obvious
# noise, and why it is worth repairing rather than detecting downstream.
#
# The donor is the model the broken one was fine-tuned from - fine-tuning does
# not change the vocabulary. Verified before wiring: the two models'
# vocabulary.json files match entry-for-entry across all 51,866 tokens.
TOKENIZER_DONOR: dict[str, str] = {
    "flyfront/anime-whisper-faster": "kotoba-tech/kotoba-whisper-v2.0-faster",
}


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "SubOrdinant"


def asset_path(name: str) -> Path:
    """Locate a file shipped in assets/, from a checkout or a frozen build.

    The spec unpacks assets/ to the bundle root, so the path relative to it is
    the same one the checkout uses - only the root differs.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    root = Path(frozen) if frozen else Path(__file__).resolve().parent.parent
    return root / "assets" / name


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
    #     beam 1   39% lost   146 ms at a full 12 s buffer   RTF 0.42
    #     beam 2   24% lost
    #     beam 3   24% lost
    #     beam 5   15% lost   216 ms                         RTF 0.62
    #
    # Clean audio is unchanged either way (203/204 words). RTF is against the
    # 0.35 s process_interval_sec; 0.62 still leaves headroom, so the cost is
    # latency that was never the binding constraint.
    beam_size: int = 5
    # Blocks the decoder from re-emitting an n-gram it has already produced.
    # 0 is off, which is right for the plain Whisper checkpoints: they rarely
    # loop at temperature 0, and forbidding repeats costs real text in
    # Japanese, where short particles legitimately recur inside one sentence.
    #
    # Anime-Whisper needs it. Trained on emotive speech full of drawn-out and
    # stuttered delivery, it degenerates into stutter loops on exactly that
    # material - the failure looks like ヤバい......バい......ですか, a real
    # word decaying into its own tail. Its card benchmarks the model at 5,
    # which is where the entry in TRANSCRIBERS puts it.
    #
    # Set by every TRANSCRIBERS entry rather than left to this default: the
    # GUI applies an entry's keys with setattr and resets nothing, so a value
    # only some entries carry would survive a switch to a model that does not
    # want it.
    no_repeat_ngram_size: int = 0
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
    # Which translator runs: "ct2" is the CTranslate2 NMT model in mt_model
    # (Sugoi), "llm" is a local instruction model over Ollama - see
    # subordinant/llm_translate.py. They differ by ~10x in cost per call and are
    # not interchangeable at every point in the loop, which is what mt_async and
    # mt_live_enabled below are for.
    mt_backend: str = "ct2"
    # Translate off the ASR thread. Sugoi is 46 ms and is simply called inline;
    # an LLM is 300-600 ms, and that inline would stall the capture loop for the
    # whole call - audio piles up in the queue and drop_backlog() eventually
    # fires, so the price of a slow translator would be discarded audio rather
    # than a late caption. Off-thread the ASR loop never waits, and the caption
    # arrives when it arrives. Left off for ct2, whose measured numbers were all
    # taken on the inline path.
    mt_async: bool = False
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
    #     beam 10  67.2             48.1   <- here
    #     beam 20 149.1            111.6   2.5x beam 10 for identical text
    #
    # Cost scales with beam width and no setting avoids that: a 300M model on a
    # batch of one is latency-bound on sequential decode steps rather than
    # throughput-bound, so the idle GPU cannot absorb a wider beam. float16 buys
    # ~15%, int8 nothing, and beam 50 falls off a cliff entirely (745 ms).
    #
    # The live tail is what makes any of this a real cost - it runs every pass,
    # not once per utterance, so at the 0.35 s passes below beam 10 is 0.14 of
    # the real-time budget against Whisper's 0.77. If it ever needs to come down,
    # lower the preview rather than the committed pass: the preview is a guess
    # that rewrites itself anyway, while committed text amortises over ~8 passes
    # and is nearly free at any beam.
    mt_beam_size: int = 10
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
    #
    # Offered in the GUI rather than left as tuning, because on spontaneous
    # speech it is the main control over legibility: Whisper emits no
    # punctuation there, so this decides where lines are cut, and a cut landing
    # mid-word makes the translator invent a sentence around the fragment. See
    # ENGINES in window.py for the measured example.
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

    # --- LLM translation (mt_backend="llm") ---
    # Ollama's HTTP API, which is why this costs no new Python dependency:
    # requests is already pinned for faster-whisper. The model stays resident in
    # the server across engine switches, so swapping 4B for 8B in the GUI is a
    # different string in one request rather than a model load.
    llm_endpoint: str = "http://localhost:11434"
    # An instruct tag, not plain qwen3, and the distinction is load-bearing.
    # qwen3:4b and qwen3:8b are reasoning models whose ollama template prefills
    # <think> on every assistant turn unconditionally, so they deliberate for
    # tens to hundreds of tokens before translating a sentence they could have
    # translated directly - pure latency here, and it exhausts llm_max_tokens
    # before the answer starts. The -instruct tag never reasons at all.
    llm_model: str = "qwen3:4b-instruct"
    # Ask for reasoning. Off, and left off: deliberation is latency this stage
    # cannot spend. Note that this does not make a reasoning model stop
    # reasoning - nothing sent over the API can, if its template forces the
    # block - it only decides whether the reasoning is asked for. See _post in
    # llm_translate.py for why false is expressed by omitting the flag.
    llm_think: bool = False
    # Ceiling on one translation, in tokens. Same job as mt_max_decoding_length
    # on the ct2 path: an instruction model handed garbled ASR will sometimes
    # comment on it at length instead of translating it.
    llm_max_tokens: int = 128
    # Low but not zero. Greedy decoding on a fragment tends to produce the same
    # stock phrase repeatedly; a little spread reads better without inventing.
    llm_temperature: float = 0.2
    # Context window. 2048 is generous for an utterance plus the few turns of
    # history below, and keeping it small keeps the KV cache small.
    llm_num_ctx: int = 2048
    # How many previous utterances to show the model as prior turns.
    #
    # This is the whole reason an LLM is worth 10x the cost. Sugoi is stateless
    # and sees exactly one sentence, which is why it renames the same character
    # every line and cannot recover a dropped subject - and Japanese drops the
    # subject constantly. Prior turns fix both: the model has already seen how
    # it rendered a name, and it knows who is being talked about.
    #
    # Costs prefill on every call, which is cheap next to decode, so this is the
    # rare knob that buys quality at almost no latency. 0 disables it.
    llm_context_utterances: int = 3
    # Give up on one translation after this long and emit nothing rather than
    # letting a wedged server back the queue up indefinitely. Generous, because
    # off-thread a slow call costs a late caption and nothing else.
    llm_timeout_sec: float = 20.0
    # How long Ollama holds the model in VRAM after the last request. Longer
    # than the default 5 minutes, so a pause in the audio does not mean paying
    # the load again on the next line.
    llm_keep_alive: str = "30m"

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
    #
    # This is the strongest latency lever in the program, and it pays twice. A
    # word waits here once for the pass that first transcribes it and again for
    # the pass that agrees with it - LocalAgreement commits nothing on a single
    # sighting - so the interval appears twice on the path from speech to
    # committed text. Dropping 0.5 to 0.35 takes roughly 0.3 s off it.
    #
    # What it costs: each pass re-transcribes the whole buffer, so its cost is
    # near enough independent of this and RTF is simply cost/interval. Measured
    # over 116 consecutive 12 s buffers with the arguments transcribe() really
    # passes - Kotoba beam 5, float16, temperature 0.0 - the pass ran 270 ms
    # median, 382 ms at p90, 426 ms worst, plus ~48 ms for the Sugoi live tail:
    #
    #     interval   RTF median   RTF p90
    #     0.50 s        0.64        0.86
    #     0.45 s        0.71        0.96
    #     0.40 s        0.80        1.07
    #     0.35 s        0.91        1.23   <- here
    #
    # p90 over 1.0 is not the same as dropping audio: a slow pass only adds to
    # the backlog, and drop_backlog() does not fire until max_backlog_sec of it
    # accumulates, which a 0.91 median drains. The margin is real but thin.
    # That measurement is also the pessimistic end - it was taken on audio the
    # model finds hard, where decoding runs long. The 186 ms figure under
    # TRANSCRIBERS came from ordinary Japanese speech and puts this at RTF 0.67.
    # If captions start missing words under load, raise this to 0.45 before
    # touching beam_size.
    process_interval_sec: float = 0.35
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
