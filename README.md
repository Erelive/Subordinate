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

The overlay is click-through and has no taskbar button, so the **tray icon is the
only way to pause or quit it**.

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

Whisper translates other languages *into English*, and only that direction. There
is no setting that produces Japanese captions from English audio - the model has
no target language to choose.

**1. Fetch the model.** `large-v3` is required and is not the default:

```
.venv\Scripts\python.exe -m tools.fetch_model large-v3
```

About 3 GB. Skip this and the first launch downloads it anyway, silently, while
nothing appears on screen.

**2. Run it**, then start the Japanese audio:

```
.venv\Scripts\python.exe main.py --model large-v3 --language ja --task translate
```

Capture is from the **default playback device**, so anything audible works - a
video, a live stream, a call. Quit from the tray icon. Add `--verbose` on a first
run to watch the model load and confirm the capture device.

**3. Make it permanent** once it looks right:

```
.venv\Scripts\python.exe main.py --model large-v3 --language ja --task translate --save-config
```

That writes `%APPDATA%\SubOrdinant\config.json` and plain `main.py` then uses
these settings. Delete that file to return to English transcription.

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
detail plain transcription would have kept.

Memorised sign-offs such as "Thank you for watching!" also surface during music
and silence. The blocklist in `subordinant/asr.py` drops them when they are a
pass's entire output, but not when they are appended to real speech.

## Building the distributable

```
.venv\Scripts\pyinstaller.exe SubOrdinant.spec --noconfirm
```

Output lands in `dist\SubOrdinant\`. Model weights are not bundled - the app
fetches them on first run.

## Configuration

Every setting in `subordinant/config.py` can be overridden from
`%APPDATA%\SubOrdinant\config.json`. Write the current values out with:

```
.venv\Scripts\python.exe main.py --save-config
```

The settings worth touching:

- `model` - `distil-large-v3` is faster, `large-v3` is needed for `translate`
- `compute_type` - `int8_float16` roughly halves VRAM at a small accuracy cost
- `process_interval_sec` - lower is more responsive and more GPU work
- `max_buffer_sec` - longer context is more accurate but pushes RTF towards 1.0
- `font_size`, `max_width_frac`, `bottom_margin_frac` - overlay placement

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

- **Exclusive-fullscreen games.** An always-on-top window cannot draw over them.
  Borderless-windowed mode works.
- **Output device changes.** Switching the default output device while running is
  not picked up; restart the app.
- **Music and singing.** Whisper will sometimes produce lyrics-shaped text from
  songs. The VAD suppresses most of it and a small blocklist catches the common
  memorised phrases ("Thanks for watching!"), but not all of it.
- **Bundle size.** The CUDA libraries alone are ~1.3 GB, before the 1.6 GB of
  model weights. This is not a small download.

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
