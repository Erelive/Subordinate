# What actually moved the numbers

A record of the changes that improved accuracy or speed, what each one bought,
and how it was measured. Kept separate from the README because the README says
what the app does now, and this says why - which is what you need when deciding
what to try next.

Everything here was measured on an RTX 4080 SUPER. "RTF" is inference seconds per
second of audio; above 1.0 the pipeline falls behind and starts discarding audio.
Ground truth for the word-loss figures is a 60 s TTS passage of 204 known words.

---

## The big wins, in order of size

### 1. Beam search: 39% → 15% word loss on overlapping speech

`beam_size` was 1, with the comment *"beams cost latency for little gain here."*
That was measured on clean single-speaker audio, where it is true. On overlapping
speech it is badly wrong.

Measured on two overlapping speakers with background music:

| beam | words lost | ms at a 12 s buffer | RTF |
|---|---|---|---|
| 1 | 39% | 146 | 0.29 |
| 2 | 24% | not measured | |
| 3 | 24% | 284 (anomalous, see below) | |
| **5** | **15%** | 216 | **0.43** |

The beam-3 worst-case timing came out *above* beam 5, which cannot be right -
beam search cost is not strictly monotonic because of early stopping, and this
was a median of only four runs. Treat it as noise; beam 5 was re-measured and is
solid.

Clean audio is 203/204 words either way, which is exactly why the original
measurement missed it. **Lesson: benchmark on the hard case, not the easy one.**

### 2. Engine choice: turbo + Sugoi instead of Whisper large-v3 translate

Per pass, against 0.5 s of fresh audio:

| engine | 4 s buffer | 8 s | 12 s |
|---|---|---|---|
| `large-v3-turbo` transcribe | 101 ms (0.20) | 124 ms (0.25) | 152 ms (**0.30**) |
| `large-v3` translate | 214 ms (0.43) | 318 ms (0.64) | 416 ms (**0.83**) |

RTF 0.83 leaves 17% headroom on an idle machine. Continuous speech pins the
buffer at its cap and anything else on the GPU pushes it past 1.0, at which point
`drop_backlog()` discards audio outright - words that are never transcribed at
all. This is what "large-v3 misses words" was.

The asymmetry that makes this work: **MT runs once per finished utterance, not
once per pass** - roughly one call in eight. A bigger *translator* is affordable
where a bigger *Whisper* is not.

### 3. Speaker embedding window: 1.5 s → 3 s

Over-splitting (one person labelled as several) is decided by the *lowest*
same-speaker similarity. It rises steeply with window length:

| window | lowest same-speaker | highest different-speaker | margin |
|---|---|---|---|
| 1.0 s | 0.241 | 0.248 | **−0.007** |
| 1.5 s | 0.606 | 0.227 | 0.380 |
| 2.0 s | 0.743 | 0.221 | 0.522 |
| **3.0 s** | **0.850** | 0.212 | **0.638** |

At 1.5 s with a 0.5 threshold there was barely a tenth of headroom, so an excited
or put-on voice opened a new speaker. Window and hop were then **separated**:
3 s of context per embedding, but a 1 s hop, so a change of voice is still placed
to within a second. Costs ~33 ms per second of speech, RTF ~0.03.

The hard guarantee is `diarize_max_speakers`. When the count is known, set it -
no threshold separates "same person shouting" from "second person" reliably.

### 4. Name glossary (`initial_prompt`)

Whisper renders an unfamiliar proper noun as whatever it sounds like, and the
translator then renders *that* faithfully - so the error is invisible downstream.
Measured on English TTS with the same failure:

```
bare     Lamy and Noelle are talking to Sabaru about the Kolob.
primed   Lamy and Noel   are talking to Subaru about the Kolob.
```

Only words in the list get fixed ("collab" stayed wrong). Keep it short: the
prompt shares decoder context with the audio, and a long one makes Whisper recite
it instead of transcribing.

### 5. Translation model: Sugoi v4 stays

Tested and rejected. Sugoi is purpose-trained on Japanese→English conversational
text; the general multilingual models spread their capacity across hundreds of
languages and lose.

| model | size | licence | verdict |
|---|---|---|---|
| **Sugoi v4** | 1.1 GB | non-commercial | best of those tested |
| madlad400-3b | 3.0 GB | MIT/Apache | clearly worse |
| M2M100 | | MIT | worse |
| opus-mt-ja-en | | Apache-2.0 | degenerates badly |

```
怖いよ、これ本当に怖いって
  Sugoi       -> I'm scared. This is really scary.
  madlad400   -> I'm stupid, this is really stupid.
```

**Lesson: bigger is not better across a domain shift.** madlad400 has 3× the
parameters and a better licence, and still lost.

---

## Bugs that were silently costing accuracy

These produced no error. Each one looked like a model limitation until it was
traced.

| # | Bug | Effect |
|---|---|---|
| 1 | `_norm()` compared punctuation literally, so `します` ≠ `します。` | LocalAgreement stalled at exactly the sentence boundaries, so MT received unpunctuated run-ons |
| 2 | Translation only flushed on silence | A conversation with no gaps produced no English at all |
| 3 | Speaker-change cut was evaluated *after* the hold-and-wait early return | The cut never fired for anything under 6 s - i.e. almost always |
| 4 | A 1–2 word fragment stranded before a speaker change was translated alone | Produced nothing, but still advanced the marker, so **the text was consumed and never displayed** |
| 5 | Empty translations dropped the source line too | Whole utterances vanished with no trace |
| 6 | `prune_history` cut at 24 s regardless of translation progress | During long unbroken speech, words were deleted before MT reached them |
| 7 | New speakers opened on a single window | The window spanning a hand-off contains two voices and matched neither, becoming a permanent phantom speaker |
| 8 | `diarize_min_embed_sec` above `diarize_window_sec` | Speaker labelling silently did **nothing** - no labels, no error |
| 9 | `save()` wrote all 44 config fields on exit | One run froze every default into the user's config, so **later tuning was silently overridden forever** |

Number 9 is the one worth remembering, because it silently undoes the others. It
reverted the 1.5 s → 3 s embedding window on an existing install: the fix
shipped, and the one machine that needed it kept the old value and the old bug.
`save()` now writes only fields that differ from their defaults, so the file
records what was chosen rather than a snapshot of every default.

---

## The limits that remain

**Overlapping speech is the hard ceiling.** Whisper transcribes one stream. When
two people talk at once it returns one of them or a blend, and the other's words
are simply absent - no error, no gap.

| audio | beam 1 | beam 5 |
|---|---|---|
| clean single speaker | 0% | 0% |
| two speakers overlapping | 26% | not measured |
| two speakers + background music | 39% | **15%** |

A missing word reaches the translator as an absence rather than an error, so the
English reads fluently and is wrong. Beam search cut the hardest case from 39% to
15%; nothing else in this architecture touches it.

Note the gap in that table: the beam sweep was run only on the hardest audio, so
the overlap-without-music figure at beam 5 is unknown. It is presumably between
0% and 15%, but it was not measured.

**Streaming translation cannot be monotonic.** Japanese puts the verb last, with
negation and tense as suffixes on it, so a partial sentence can translate to the
opposite of the finished one:

```
これは面白い               -> This is interesting.
これは面白くない           -> This isn't funny.
これは面白くないと         -> This has to be interesting.
これは面白くないと思わない -> I don't think this is funny.
```

This is why the live preview is confined to the dimmed line and the transcript
still takes whole utterances.

**Speaker labels drift.** Online clustering decides without the future, so
someone quiet for a long stretch can return as a new number. Offline diarization
does not have this problem and cannot be used live.

---

## Worth trying next

Roughly in order of expected value.

- **Source separation before the ASR.** The only real attack on the remaining
  overlap loss. Heavy - a separate model in front of the pipeline - but it is
  where the largest measurable error now lives.
- **Speaker enrollment.** A few seconds of reference audio per person would turn
  `Speaker 2` into `ラミィ`. The embeddings already exist; it needs a UI and a
  place to store them.
- **Rolling context prompt.** `condition_on_previous_text` is off, and must stay
  off - it makes Whisper loop on its own output. But feeding recent *committed*
  text into `initial_prompt` alongside the glossary would give real context
  without the loop. Untested.
- **Centroid merging.** Would fix label drift by merging speaker centroids that
  converge. Needs a way to remap labels already displayed.
- **Beam > 5, and temperature fallback.** Neither measured. Beam 5 was the best
  of {1,2,3,5}; the curve had not flattened.
- **NLLB-1.3B and m2m100-1.2B.** Untested at that size. Both are general
  multilingual models, so the madlad400 result is a reason for low expectations.
  NLLB is CC-BY-NC - the same licence class as Sugoi, so no worse for this project.
- **Whisper fine-tuned on the domain.** Would attack the name and slang errors at
  the source rather than biasing around them.

## Things that were measured and did *not* help

Recording these so they are not retried.

- **VAD off** - identical accuracy, marginally faster. Left on, since it also
  suppresses hallucinations on silence and music.
- **A 20 s buffer** - *worse* (28% loss vs 15%) and slower. Longer context is not
  automatically better.
- **`_trim()` discarding untranscribed audio** - suspected as the cause of missing
  words, instrumented, and found never to fire. The loss was overlap, not the
  pipeline.
- **Bigger MT models** - see madlad400 above.

## How to measure

The harnesses live in the scratchpad rather than the repo, but the method is:

1. Render known text through Windows SAPI (`tools\speak.ps1`) so there is ground
   truth. Two voices are available, which is enough to synthesise overlap.
2. Drive `StreamingTranscriber` directly, block by block, rather than the audio
   device - it is deterministic and does not need real time.
3. Score with longest-common-subsequence against the reference, so insertions do
   not mask deletions.
4. Report cost as RTF at a *full* buffer, since that is where continuous speech
   sits.

`SubOrdinant-debug.exe --selftest` loads every model and opens the audio device
without a window, which is how a packaged build gets checked - the failures
packaging introduces are DLL conflicts, and those fault the process rather than
raising.
