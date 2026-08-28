# Subordinate
Real-time captions for Windows system audio, transcribed locally with Whisper and drawn in an overlay.

# Details

Live on-screen captions for whatever your Windows PC is playing. Audio is
captured straight off the output device, transcribed by a local Whisper model on
the GPU, and drawn in a click-through overlay above the taskbar. Nothing leaves
the machine and there is no network round trip in the caption path.

Measured on an RTX 4060 Laptop (8 GB) with `large-v3-turbo` at float16, over 100
seconds of continuous speech:

| | |
|---|---|
| Caption lag, mean | 0.64 s |
| Caption lag, max | 1.14 s |
| Inference per pass | ~280 ms |
| Real-time factor | 0.60 |
| Backlog drops | 0 |

"Caption lag" is the gap between where the audio stream is and the end of the
word that just became final on screen - the number a viewer actually perceives.

## Status

Windows only. English transcription is the tuned and tested default.

## Requirements

- Windows 10/11
- NVIDIA GPU with current drivers. CPU works but is far slower; the app falls
  back to CPU int8 automatically if CUDA is unavailable.
- Python 3.12 for a source checkout

## Running from source

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m tools.doctor      # check devices, GPU, model
.venv\Scripts\python.exe main.py
```

First launch downloads ~1.6 GB of model weights to the Hugging Face cache.
`tools.fetch_model` does that ahead of time.

The app opens a window holding the running transcript. Both languages stay there
for the session, so the original can be checked after the fact - which is the
point when a translation mangles a name. The toolbar has opacity, pin-on-top and
pause; scrolling up to read back will not be interrupted by new text arriving.

**Audio language**, **Transcribe with** and **Translate with** are set in the
window, so a double-clicked build needs no command line.

Transcription and translation are picked separately, because they are separate
decisions:

| Transcribe with | Speed | Notes |
|---|---|---|
| **Whisper large-v3-turbo** | 210 ms, RTF 0.60 | any language; the safe default |
| **Kotoba-Whisper v2.0** | 184 ms, RTF 0.53 | Japanese only, and *faster* than turbo |
| **Whisper large-v3** | ~2x turbo | strongest, and the only one that can translate |

The **Beam** picker beside this one overrides `beam_size` for whichever model is
selected; Auto keeps the per-model default. Raising it is worth trying on people
talking over each other, where beam 1 loses 39% of the words and beam 5 loses
15%. On a single clear speaker it does almost nothing - over 300 CommonVoice
utterances Kotoba scored 89.84 / 89.77 / 89.79 percent at beam 5 / 10 / 20.

| Translate with | What runs | Notes |
|---|---|---|
| **Sugoi v4** (utterance 2 / 3 / 5 / 8 / 10 s) | Whisper transcribes, Sugoi translates | the fast default; 46 ms a line, keeps the source text |
| **Qwen3 4B Instruct** (utterance 3 / 5 s) | Whisper transcribes, a local LLM translates | needs Ollama; 116 ms a line, and sees the preceding lines, so names hold |
| **Sugoi 14B Ultra** (utterance 3 / 5 s) | Whisper transcribes, a ja->en specialist LLM translates | needs Ollama and 9 GB of VRAM; the most accurate option, 182 ms a line |
| **Whisper large-v3** | one model, `task=translate` | English only, RTF 1.31 - see below; locks the transcriber to large-v3 |
| **No translation** | transcription only | source language only |

### The LLM engines

Sugoi is a 300M NMT model that sees one sentence and nothing else, and its two
standing failures both come from that: it renames the same character on every
line, and it cannot recover a subject the speaker dropped - which Japanese
speakers do constantly. Neither is reachable by tuning a model with no memory.

The Qwen3 engines send each committed utterance to a local instruction model
along with the last three, as prior turns. The model has already seen how it
rendered a name, and it knows who is being talked about. That is the whole
trade: 116 ms a line against Sugoi's 46 ms - mean and median over 14 warm
calls on a 4080 SUPER, max 199 ms - spent on the one thing the NMT route
cannot do.

They need [Ollama](https://ollama.com) running, with the model pulled:

```
winget install Ollama.Ollama
ollama pull qwen3:4b-instruct
```

**It has to be an instruct tag.** Plain `qwen3:4b` and `qwen3:8b` are reasoning
models, and Ollama's template for them prefills a `<think>` block on every
assistant turn unconditionally - there is no `enable_thinking` branch in it, so
nothing sent over the API stops them deliberating. Passing `"think": false` only
switches off Ollama's *parser*, which sends the deliberation down `content` with
its opening tag already eaten by the template, where nothing can strip it and it
renders verbatim as captions. `-instruct` never reasons, so the question does
not arise. There is no `qwen3:8b-instruct`; `qwen2.5:7b-instruct` is the nearest
larger non-reasoning model.

#### Sugoi 14B Ultra

`Sugoi-14B-Ultra` is a Japanese-to-English specialist fine-tune rather than a
general instruction model, and it is the most accurate engine here. It is not in
Ollama's own library - there is no `sugoi` entry there - so it comes from Hugging
Face, which Ollama resolves itself from the `hf.co/` prefix:

```
ollama pull hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M
```

`Q4_K_M` is 9.0 GB and the right quantisation for a 16 GB card: `Q8_0` is 15.7 GB
and leaves nothing for Whisper beside it, and `Q2_K` at 5.8 GB gives the accuracy
back. **This engine wants 12 GB of VRAM or better.** Below that Ollama offloads
layers to the CPU and the per-line cost stops being a caption delay and starts
being seconds.

Measured on a 4080 SUPER, fully GPU-resident, same eight lines through the app's
own prompt path:

| | mean | median | max |
|---|---|---|---|
| Qwen3 4B Instruct | 97 ms | 94 ms | 161 ms |
| Sugoi 14B Ultra | 182 ms | 177 ms | 267 ms |

Twice the cost of the 4B for output that is consistently more idiomatic -
*"My hair doesn't cooperate, and it gets tangled easily"* against the 4B's
*"My hair doesn't stay styled well, and it tends to tangle"*. Both sit far inside
the per-utterance budget, so on a card that fits it there is little reason to
prefer the smaller model.

The model card recommends its own system prompt, a localizer framing that asks
for colloquial and slang vocabulary. It is not used. The app's prompt is kept
instead because it carries the one instruction the card's does not - that the
line comes from live speech, may be cut off mid-sentence, and must never have an
ending invented for it - which is the difference between a caption and a
hallucination on ASR fragments. Measured, the card prompt was also slower
(208 ms) and drifted on content: it rendered 大変そう as "it's going to be a
hassle" where the app prompt gave the correct "it seems harder than usual".

Nothing else changes - the weights live in Ollama rather than in this process,
so the app gains no new Python dependency and the bundle does not grow. If the
server is down or the tag is not pulled, the engine says so at **Apply** and
captions continue in the source language rather than the run failing.

Two things are forced on these engines rather than offered, both by cost:

- **The dimmed live preview is off.** It runs every pass, ~0.35 s apart, on a
  budget of tens of milliseconds. No LLM is close. English arrives a whole
  utterance at a time instead of moving continuously.
- **Translation runs off the ASR thread.** Half a second inline stalls the
  capture loop until `drop_backlog()` discards audio, so a slow translator would
  cost *audio* rather than latency. On the worker it costs neither: the caption
  simply lands a few hundred milliseconds later, on a line that was already
  waiting for an utterance boundary.

4B and 8B are both offered at the same two utterance lengths so switching
between them is a fair comparison - changing the hold at the same time would
confound it. On a 16 GB card both fit beside Kotoba comfortably; gpt-oss-20b,
the obvious other candidate, does not (~12 GB of weights against ~13.5 GB free)
and is a reasoning model besides, which for translation is latency with no
return.

The seconds on the Sugoi entries are `mt_max_utterance_sec`: how long speech
carrying no sentence ending is held before being translated anyway. Whisper only
emits `。` and `、` when it hears a clear sentence-final fall, which scripted
material has and live conversation does not, so on a stream this backstop is what
decides where lines get cut. Cut too early and it lands mid-word - the translator
then renders the orphaned half as a whole sentence, fluently and entirely
invented. `...見えてしま` / `います...` came back as *"I could see what was
inside."* / *"He's here."*, where the uncut `...見えてしまいますよ` gives
*"...you can see what's inside."* Longer holds cut less often and read better,
and arrive later. Try 5 s on a stream, 2-3 s on anime.

Measured at a 12 s buffer on an RTX 4080 SUPER, Japanese audio. Kotoba is a
Distil-Whisper: the encoder is the full 32 layers but the decoder is cut from
32 to 2, which is why it beats turbo on speed while being trained specifically
on Japanese. Pick it only for Japanese - Whisper's language token still exists
on it, so another language is accepted rather than refused and comes back as
confident nonsense.

It is **v2.0 and not v2.2** deliberately. v2.1 and v2.2 wrap the same acoustic
weights in a transformers pipeline that adds punctuation and pyannote speaker
diarization; CTranslate2 cannot carry that pipeline, and every
`kotoba-whisper-v2.2-faster` conversion on the Hub is a byte-identical copy of
v2.0-faster (same sha256). The parts v2.2 adds are parts this program already
does for itself, in `diarize.py` and the sentence splitter.

Changing any of them reloads models, so it takes effect on **Apply** rather than
immediately, and the pipeline restarts - a few seconds of no captions.

### Why a window rather than an overlay

The overlay draws on the primary screen and cannot do better: system audio is one
mixed stream with no screen affinity, so nothing in it says which monitor the
sound came from. With a video on the second screen and a game on the first, the
captions land over the game - and an always-on-top window cannot draw over an
exclusive-fullscreen game at all. A window you place solves both, because the
only party who knows where the captions belong is you.

The overlay is still there for single-screen full-screen viewing, where there is
nothing to collide with. Enable it with `overlay_enabled` in the config. It is
click-through and has no taskbar button, so the tray icon controls it.

### Useful commands

```
.venv\Scripts\python.exe -m tools.doctor           # devices + GPU + speed
.venv\Scripts\python.exe -m tools.capture_check 6  # is loopback capturing?
.venv\Scripts\python.exe -m tools.console 30       # captions in the terminal
.venv\Scripts\python.exe main.py --model large-v3 --language ja --task translate
```

`tools\speak.ps1` speaks text through the speakers with Windows SAPI, which gives
the capture path a repeatable source with known ground truth. `tools\overlay_test.ps1`
launches the overlay, speaks into it, and screenshots the result.

## Japanese to English captions

Translation runs in either of two ways. **`--mt` is the recommended one**: Whisper
transcribes the Japanese, and a dedicated NMT model turns each finished utterance
into English. It is faster than making Whisper translate, reads better, and shows
both languages at once.

**1. Fetch the models.** Whisper's default is fine here - only the NMT model is
new:

```
.venv\Scripts\python.exe -m tools.fetch_model --mt
```

About 1 GB, cached alongside the Whisper weights. Skip it and the first launch
downloads it anyway, silently, while nothing appears on screen.

**2. Run it**, then start the Japanese audio:

```
.venv\Scripts\python.exe main.py --language ja --mt
```

Capture is from the **default playback device**, so anything audible works - a
video, a live stream, a call. Quit from the tray icon. Add `--verbose` on a first
run to watch the models load and confirm the capture device.

The overlay shows the English in white with the Japanese dimmed beneath it. The
Japanese streams word by word; the English lands one sentence at a time, once the
speaker pauses. That is not a tuning problem - see below.

**3. Make it permanent** once it looks right:

```
.venv\Scripts\python.exe main.py --language ja --mt --save-config
```

That writes `%APPDATA%\SubOrdinant\config.json` and plain `main.py` then uses
these settings. Delete that file to return to English transcription.

### Why not just use Whisper large-v3 to translate

It is more accurate per sentence, and it loses words to pay for it. The pipeline
re-transcribes the whole rolling buffer every pass, so cost grows with buffer
length; measured on an RTX 4080 SUPER. RTF is per pass against
`process_interval_sec`, which is 0.35 s:

| engine | 4 s buffer | 8 s | 12 s |
|---|---|---|---|
| `large-v3-turbo` transcribe | 101 ms (RTF 0.29) | 124 ms (0.35) | 152 ms (**0.43**) |
| `large-v3` translate | 214 ms (0.61) | 318 ms (0.91) | 416 ms (**1.19**) |

Turbo at 0.43 has room to spare. `large-v3` translate does not: at a full buffer
it is past 1.0 on its own, so that route needs `process_interval_sec` raised
back to 0.5 to fit. Continuous speech keeps the buffer at its cap, and anything
else using the GPU - a game on the other monitor, most obviously - costs more
still. The backlog then grows every
pass until `max_backlog_sec`, at which point `drop_backlog()` discards that audio
outright. Those words are never transcribed at all, which is what "it misses
words" looks like from the outside.

Turbo plus a dedicated translator sits at 0.30 and does not have that failure
mode. Bigger MT models are affordable here where a bigger *Whisper* is not,
because MT runs once per finished utterance rather than once per pass - roughly
an eighth as often.

### The other way: Whisper's own translation

Whisper can translate without a second model, but needs `large-v3` to do it, and
gives you English only - the Japanese is discarded:

```
.venv\Scripts\python.exe -m tools.fetch_model large-v3
.venv\Scripts\python.exe main.py --model large-v3 --language ja --task translate
```

Measured on an RTX 4080 SUPER at the default 12 s buffer:

| | ms/pass | RTF | Output |
|---|---|---|---|
| `--mt` (turbo + NMT) | 154 + ~8 amortised | **0.46** | Japanese *and* English |
| `--task translate` (large-v3) | 460 | **1.31** | English only |

`--mt` is roughly 3x cheaper because the NMT stage runs once per *utterance* -
about one call in eight passes - rather than on every pass over the rolling
buffer. At ~46 ms per call it amortises to single-digit milliseconds per pass. The
`--task translate` route runs at RTF 1.31 against the 0.35 s interval, so it no
longer fits one: the backlog grows every pass until `max_backlog_sec`, at which
point `drop_backlog()` discards that audio outright. Raise `process_interval_sec`
to 0.5 if you want that route, which puts it back at 0.92.

Quality favours `--mt` too. Whisper's translate task was a training side
objective, not a translation system, and it shows most on proper nouns.

Whisper translates other languages *into English*, and only that direction, in
both modes. There is no setting that produces Japanese captions from English
audio - the model has no target language to choose.

### Why not the default model

`large-v3-turbo` does not translate. It loads, accepts `--task translate`, and
returns **Japanese** - transcribing instead of translating, because the turbo
fine-tune was trained on transcription only. Nothing errors, so it reads as a bug
somewhere else; the app raises a tray warning at startup instead. The `distil-*`
models are English-only distillations and are wrong for a Japanese source for a
different reason.

Turbo remains the better choice for *transcribing* Japanese, which it does well
and roughly 2.5x faster.

### What to expect

`--language ja` describes the audio, not the captions. Leaving it off makes
Whisper re-guess the source language every pass, which it does inconsistently on
a rolling buffer.

`large-v3` costs about 2.5x turbo per pass. Measured on an RTX 4080 SUPER at the
default 12 s buffer: ~400 ms per pass, real-time factor near 0.8 - it keeps up,
without much headroom. On the 8 GB laptop GPU in the table above it will not;
raise `process_interval_sec` or lower `max_buffer_sec` to trade latency or
context for throughput, and watch for `dropped backlog` under `--verbose`.

Proper nouns are the weak point. Names, nicknames and invented terms are
routinely mangled by the translate task, so content that leans on them loses
detail plain transcription would have kept. `--mt` helps twice over: the NMT
model handles them better, and the Japanese stays on screen underneath, so a
mangled name is still recoverable by eye.

Translation quality depends heavily on the transcript being well formed, so
`_norm()` in `subordinant/streaming.py` compares words with punctuation
stripped. Whisper attaches a sentence mark only once it has heard enough to
decide the sentence ended, so the same word arrives as `します` on one pass and
`します。` on the next; comparing those literally made LocalAgreement stall at
exactly the sentence boundaries, and what reached the translator was an
unpunctuated run-on that it rendered badly.

**English arrives a sentence at a time under `--mt`, and cannot be streamed.**
LocalAgreement-2 commits a stable prefix, which works because transcription is
monotonic - the words already settled never change. Translation is not: Japanese
is SOV and English is SVO, so the verb ending a Japanese sentence lands in the
middle of the English one. A partial Japanese sentence does not determine an
English prefix, so streaming it would produce captions that rewrite themselves,
which is the exact behaviour the streaming design exists to eliminate.

Translation fires whenever a sentence completes, not only during pauses - a
conversation between several people can run for minutes without a gap, and
waiting for one would mean no English at all while anyone is talking. Speech that
never produces a sentence ending is translated anyway once it spans
`mt_max_utterance_sec` (6 s by default). Lower that for shorter, more frequent
English; raise it for fewer, longer, better-formed sentences.

Memorised sign-offs such as "Thank you for watching!" also surface during music
and silence. The blocklist in `subordinant/asr.py` drops them when they are a
pass's entire output, but not when they are appended to real speech.

## Why English arrives a sentence at a time

Because a partial Japanese sentence does not determine an English prefix, and
often determines its opposite. Japanese puts the verb last, and negation and
tense are suffixes on that verb, so translating as you go means translating
something that has not decided what it means yet. Real Sugoi output on growing
prefixes of one sentence:

```
これは面白い               -> This is interesting.
これは面白くない           -> This isn't funny.
これは面白くないと         -> This has to be interesting.
これは面白くないと思わない -> I don't think this is funny.
```

Three reversals in four passes. That is why the transcript takes whole
utterances: LocalAgreement works on the source because ASR output is monotonic,
and translation simply is not.

**Live preview** (on by default, toggle in the toolbar) gives the continuous
motion anyway, without putting rewrites in the record. The unfinished tail is
translated every pass and shown above the Japanese it came from, dimmed - the
same contract the unconfirmed source text already has. The settled translation
still drops into the transcript when the sentence ends, unchanged.

It costs 17-46 ms per pass, an added RTF of 0.05-0.13, and only re-runs when the
tail has actually changed. Turn it off if the movement is distracting; nothing
else changes.

## Overlapping speech is the hard limit

Whisper transcribes one stream. When two people talk at once it does not
transcribe both and it does not fail loudly - it returns one of them, or a blend,
and the words of whoever lost are simply absent. Measured over a minute of
continuous speech with known ground truth:

| audio | words lost |
|---|---|
| clean single speaker | 0% |
| two speakers overlapping | 26% |
| two speakers + background music | 39% |

Nothing downstream can recover this. A missing word reaches the translator as an
absence, not an error, so the English reads fluently and is simply wrong. If a
line goes missing from a collab, this is almost always why - not the translator,
and not a dropped buffer.

What helps is beam search. On the same overlapping audio, `beam_size` 5 loses 15%
where greedy loses 39%, and costs RTF 0.62 against 0.42 - latency that was never
the binding constraint here. It is the default for that reason. On clean audio
the two are indistinguishable, which is why the original measurement, taken on
clean audio, concluded beams were not worth it.

## Names, and why translations go wrong

Most bad translations are not the translator's fault. Whisper renders a name it
has never heard as whatever it sounds like, and the MT stage then translates that
wrong word perfectly faithfully, so nothing downstream can tell anything happened.

The **Names** box takes the words a stream will use - `ラミィ, ノエル, スバル,
ホロライブ` - and biases the decoder before it guesses. Measured on English TTS
with the same failure, the transcript went from

```
bare     Lamy and Noelle are talking to Sabaru about the Kolob.
primed   Lamy and Noel are talking to Subaru about the Kolob.
```

Note that "collab" stayed wrong: only words actually in the list get fixed.

Keep the list short. It occupies the same decoder context the audio competes
for, and a long one makes Whisper start reciting the prompt instead of
transcribing - which looks like hallucinated captions, not a bad setting.

How much this is worth is visible by comparing Sugoi on clean input against the
same lines as they came off a stream:

```
clean    自己紹介はラミィとスバルでやろうか
         -> Shall I introduce myself with Rammy and Subaru?
stream   ろよいいはじゃあ半分半分でやる?
         -> Royoiha, would you like to do it in half?
```

The Japanese is already wrong in the second case. No translation model recovers
from that, which is why the glossary matters more than the choice of translator.

## Labelling who is speaking

Tick **Label speakers**, or pass `--speakers`. Each line is then tagged
`Speaker 1`, `Speaker 2` and so on, in a colour per speaker, and a change of
voice starts a new line.

The tag is the smaller half of what this buys. The larger half is that a change
of voice becomes a hard boundary: without it, two people talking without a gap
are handed to the translator as one unit, and one person's sentence gets finished
by the next person's words. That is where run-ons in the transcript came from.

It costs a 28 MB model and almost nothing at runtime - about 33 ms per second of
speech on the CPU, off the GPU path, an RTF around 0.04 against Whisper's 0.53.

**Set the number of people** if you know it. That cap is by far the most reliable
control, because the thing that splits one person into several is the same person
sounding different - shouting, laughing, putting on a voice - and no similarity
threshold separates that from a second person cleanly. Telling it there are three
means there cannot be a fourth.

**What it cannot do.** Offline diarization clusters over a whole recording, which
is what makes its numbering globally consistent. Live, there is no whole
recording: each window is matched against a bank of running centroids and
labelled on the spot. So

- **numbers are not names.** They are assigned in the order people are first
  heard.
- **numbers can drift.** Someone quiet for a long stretch may not match their old
  centroid closely enough and comes back as a new number.
- **overlapping speech breaks it.** Two people talking over each other produce one
  embedding belonging to neither. A collab is full of this.
- **background music and game audio** pollute the embeddings.
- **it hears voices, not intent.** If one person reads out another's lines, the
  labels will be right and still look wrong.

Tuning lives in `config.json`. The two that matter:

`diarize_window_sec` (3 s) is how much speech goes into one embedding, and it is
the biggest single lever on over-splitting. Measured on two voices, the *lowest*
same-speaker similarity was 0.24 at a 1 s window, 0.61 at 1.5 s, 0.74 at 2 s and
0.85 at 3 s, while the highest between-speaker similarity barely moved (0.25 ->
0.21). Short windows make one voice look like a stranger. `diarize_hop_sec` (1 s)
is kept well below it so a change of voice can still be placed to within a
second: each hop embeds three seconds of context but labels only the last one.

`diarize_threshold` (0.5) is the similarity at which a window is judged to be
someone already known: lower it when one person keeps splitting in two, raise it
when two people share a label. Note this is the opposite direction to the speaker
cap, which is usually the better tool.

Two smaller rules are worth knowing because they show up as behaviour. A new
voice is only given a number once a **second** window agrees with the first - the
window spanning a hand-off contains two voices and matches neither, and left alone
it became a permanent extra identity between two real people. And a word in a
brief unlabelled gap adopts the turn around it, so a short pause does not strip
the label off the line.

## Building the distributable

```
.venv\Scripts\pyinstaller.exe SubOrdinant.spec --noconfirm
```

Close any running build first. Windows keeps loaded DLLs locked, so a live
`SubOrdinant.exe` makes the rebuild fail with `PermissionError: [WinError 5]` on
a `.pyd` under `dist\SubOrdinant\_internal\`.

Output lands in `dist\SubOrdinant\` - `SubOrdinant.exe` to double-click, plus
`SubOrdinant-debug.exe` with a console attached, because a failure inside a
bundled native library is otherwise completely silent. Model weights are not
bundled; the app fetches them on first run.

Note what the spec does with `sentencepiece`: it is copied by hand and excluded
from analysis. PyInstaller imports every collected package in one isolated
subprocess to walk binary dependencies, and that subprocess has already imported
PyQt6 - so `_sentencepiece.pyd` binds to Qt's vendored MSVC runtime and dies with
an access violation, the same collision described below. The app avoids it by
pinning the system runtime before Qt loads; the analysis subprocess cannot.
`sherpa_onnx` was checked for the same problem and does not have it, so it is a
plain hidden import.

### Checking a packaged build

```
build\dist-next\SubOrdinant\SubOrdinant-debug.exe --selftest --speakers --mt --language ja
```

Loads every model, opens the audio device, prints what worked, and exits without
a window. Worth running after any packaging change, because the failures
packaging introduces are DLL conflicts - they fault the process instead of
raising, so a bundle can build cleanly and still die the moment a model is
constructed. Use the debug executable: the windowed one has no console for the
output to reach.

## Configuration

Every setting in `subordinant/config.py` can be overridden from
`%APPDATA%\SubOrdinant\config.json`. Write the current values out with:

```
.venv\Scripts\python.exe main.py --save-config
```

The settings worth touching:

- `model` - `distil-large-v3` is faster, `large-v3` is needed for `translate`
- `initial_prompt` - names and terms to expect; the Names box writes this
- `mt_enabled` - translate utterances with a dedicated NMT model (`--mt`)
- `mt_model` - the NMT model; defaults to Sugoi v4 JA->EN
- `mt_source_lang` / `mt_target_lang` - which SentencePiece pair to load
- `mt_beam_size` - beam search is nearly free here; it runs per utterance
- `mt_max_utterance_sec` - translate unpunctuated speech after this long; the
  Translate-with picker offers 2/3/5/8 s, and on live speech it is the main
  control over whether lines read as sentences or as fragments
- `mt_repetition_penalty`, `mt_no_repeat_ngram_size` - stop NMT repeat loops
- `diarize_enabled` - label lines with who said it (`--speakers`)
- `diarize_max_speakers` - how many people are talking; the most useful setting
- `diarize_window_sec` - speech per embedding; short windows over-split
- `diarize_hop_sec` - how precisely a change of voice can be placed
- `diarize_threshold` - lower to split fewer speakers, raise to merge fewer
- `mt_min_chunk_sec` - ignore speaker changes that would strand a fragment
- `mt_live_enabled` - preview the unfinished tail; the toolbar toggles it
- `beam_size` - 5 by default; 1 loses 39% of words on overlapping speech. The
  toolbar's Beam picker overrides it (Auto keeps each model's own default);
  on one clear speaker it changes almost nothing, and it is expensive on
  large-v3 - see the picker's tooltip for the per-model cost
- `compute_type` - `int8_float16` roughly halves VRAM at a small accuracy cost
- `process_interval_sec` - 0.35 s; the strongest latency lever, and it pays
  twice, since LocalAgreement needs two passes to commit a word
- `max_buffer_sec` - longer context is more accurate but pushes RTF towards 1.0
- `font_size`, `max_width_frac`, `bottom_margin_frac` - overlay placement
- `overlay_enabled` - draw the click-through overlay as well as the window
- `window_opacity`, `window_on_top` - written back when the app exits

## How it works

```
WASAPI loopback -> downmix + resample to 16 kHz -> energy gate
   -> rolling buffer -> Whisper (every ~0.35 s) -> LocalAgreement-2 -> overlay
```

**Capture.** WASAPI loopback records the default render device, so any app's
output is picked up with no virtual audio cable. The device runs at 48 kHz
stereo; audio is downmixed to mono and resampled to the 16 kHz Whisper expects.
Note that loopback delivers no samples at all while nothing is playing, which is
why "no audio" and "silence" look identical at this layer.

**Gating.** A cheap RMS energy gate decides whether audio is worth sending to the
GPU. It is not the speech/non-speech decision - music passes it easily. The real
decision is the Silero VAD bundled with faster-whisper, which runs on the audio
that gets through. The gate also detects end of utterance so the buffer can be
flushed.

**Streaming.** Whisper is not a streaming model: it transcribes a whole buffer at
once and its answer for a given word can change as more audio arrives. Printing
raw output every half second produces text that visibly rewrites itself. This
uses LocalAgreement-2 (Machacek et al., 2023): run inference on the growing
buffer, and commit a word only once two consecutive passes agree on it. The
committed prefix is final and drawn in white; the disagreeing tail is drawn
dimmed as a live guess. Once the buffer passes `max_buffer_sec`, audio behind the
last committed word is discarded.

## Known limits

- **Exclusive-fullscreen games.** Affects the optional overlay only: an
  always-on-top window cannot draw over them, and borderless-windowed mode works.
  The main window sidesteps this - put it on another monitor and nothing tries to
  draw over the game at all.
- **Output device changes.** Switching the default output device while running is
  not picked up; restart the app.
- **Music and singing.** Whisper will sometimes produce lyrics-shaped text from
  songs. The VAD suppresses most of it and a small blocklist catches the common
  memorised phrases ("Thanks for watching!"), but not all of it.
- **Bundle size.** The CUDA libraries alone are ~1.3 GB, before the 1.6 GB of
  model weights. This is not a small download.
- **No console in the packaged build.** `SubOrdinant.exe` is windowed, so
  `sys.stdout` and `sys.stderr` start out `None`. Logs go to
  `%APPDATA%\SubOrdinant\subordinant.log` and native faults to `crash.log`
  beside it; `main.py` then replaces the streams with real files, because
  libraries write to them without checking - huggingface_hub's download progress
  bars did exactly that and hung the app on "loading translation model...".
  Code that needs to know whether a console exists must ask
  `logsetup.has_console()`, which checks `sys.__stderr__`; `sys.stderr` is the
  substitute and will lie. Run `SubOrdinant-debug.exe` for the same output on a
  real console.

## Implementation notes

Two Windows-specific traps are worth knowing about, both handled in
`subordinant/cuda.py`:

**PyQt6 ships its own MSVC runtime.** `PyQt6\Qt6\bin` contains `msvcp140.dll` and
`vcruntime140.dll`, and importing PyQt6 puts that directory on the DLL search
path. CTranslate2's native library then resolves those by base name and can bind
to Qt's build instead of the system one, which kills the process with an access
violation the moment a Whisper model is constructed - importing PyQt6 is enough
to trigger it, no `QApplication` required. Importing the `subordinant` package
loads the System32 copies first, which pins them for everything that follows.
This is why `main.py` imports `subordinant` above `PyQt6`.

**The pip CUDA wheels are not optional.** CTranslate2 vendors `ctranslate2.dll`,
`libiomp5md.dll` and a `cudnn64_9.dll` *stub*, and loads all three from its own
directory by absolute path at import. cuBLAS and the real `cudnn_*64_9.dll`
libraries come from the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels,
whose directories are registered at startup.
