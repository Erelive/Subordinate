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

**Audio language** and **Translate with** are set in the window, so a
double-clicked build needs no command line. The three engines are:

| | What runs | Notes |
|---|---|---|
| **Sugoi v4** | turbo transcribes, Sugoi translates | recommended; RTF 0.31, keeps the source text |
| **Whisper large-v3** | one model, `task=translate` | English only, RTF 0.92 |
| **No translation** | turbo transcribes | source language only |

Changing either reloads models, so it takes effect on **Apply** rather than
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
| `--mt` (turbo + NMT) | 154 + ~6 amortised | **0.31** | Japanese *and* English |
| `--task translate` (large-v3) | 460 | **0.92** | English only |

`--mt` is roughly 3x cheaper because the NMT stage runs once per *utterance* -
about one call in eight passes - rather than on every pass over the rolling
buffer. At ~46 ms per call it amortises to single-digit milliseconds per pass. The
`--task translate` route runs at RTF 0.92, which leaves ~40 ms of headroom before
`max_backlog_sec` starts discarding audio outright.

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

## Configuration

Every setting in `subordinant/config.py` can be overridden from
`%APPDATA%\SubOrdinant\config.json`. Write the current values out with:

```
.venv\Scripts\python.exe main.py --save-config
```

The settings worth touching:

- `model` - `distil-large-v3` is faster, `large-v3` is needed for `translate`
- `mt_enabled` - translate utterances with a dedicated NMT model (`--mt`)
- `mt_model` - the NMT model; defaults to Sugoi v4 JA->EN
- `mt_source_lang` / `mt_target_lang` - which SentencePiece pair to load
- `mt_beam_size` - beam search is nearly free here; it runs per utterance
- `mt_max_utterance_sec` - translate unpunctuated speech after this long
- `mt_repetition_penalty`, `mt_no_repeat_ngram_size` - stop NMT repeat loops
- `compute_type` - `int8_float16` roughly halves VRAM at a small accuracy cost
- `process_interval_sec` - lower is more responsive and more GPU work
- `max_buffer_sec` - longer context is more accurate but pushes RTF towards 1.0
- `font_size`, `max_width_frac`, `bottom_margin_frac` - overlay placement
- `overlay_enabled` - draw the click-through overlay as well as the window
- `window_opacity`, `window_on_top` - written back when the app exits

## How it works

```
WASAPI loopback -> downmix + resample to 16 kHz -> energy gate
   -> rolling buffer -> Whisper (every ~0.5 s) -> LocalAgreement-2 -> overlay
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
  `sys.stdout` and `sys.stderr` are `None`. Logs go to
  `%APPDATA%\SubOrdinant\subordinant.log` and native faults to `crash.log`
  beside it. Anything constructing a `StreamHandler`, or calling
  `faulthandler.enable()` with no argument, breaks that build and only that
  build - run `SubOrdinant-debug.exe` to get the same output on a console.

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
