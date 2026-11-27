# Benchmarks

What the numbers in the code comments and the profiles are based on, and how to
take them again.

Everything here was measured on an RTX 4080 SUPER (16 GB) with a game holding
about 3.5 GB of the card and a varying share of its time, which is how the app
is normally used and is also why some of the latency figures below come with a
warning.

## The test set

`bench/samples/` holds eight recordings, thirteen minutes in all, each with a
reference transcript, a reference English translation and the time every
character was spoken:

| sample | what it stresses |
|---|---|
| `01_slow_clear` | read speech, clean - the easy case |
| `02_conversational` | unscripted talk, fillers, long run-on sentences |
| `03_fast` | fast delivery with almost no gap between sentences |
| `04_multi_speaker` | short turns, different voices |
| `05_casual_slang` | anime and game dialogue |
| `06_background_music` | speech over music |
| `07_pauses` | mid-sentence pauses, long dead air, music with nobody talking |
| `08_long_continuous` | two and a half minutes with no gap long enough to count as a pause |

`bench.run` plays a sample into the real pipeline in place of the audio device -
same voice detector, same recogniser, same translators, same threads - and
`bench/scoring.py` compares what came out with the reference.

- **Japanese errors (CER)** - character error rate of the Japanese in the
  finished transcript lines. Kana/kanji choices count as errors (わたし for 私),
  as in every published Japanese CER, so a few percent of this is spelling
  rather than mishearing.
- **English vs reference (chrF)** - character n-gram overlap between the English
  lines and one reference translation. Good for comparing runs with each other;
  it punishes a correct free translation as much as a wrong one, so do not read
  a couple of points as better or worse.
- **Final English after end** - from the end of a sentence to its finished
  English appearing. The reference marks a sentence's end where its last
  character was aligned, which is up to 0.3 s before the sound actually dies
  away, so about 0.15 s of every figure in this row is the speaker still
  finishing. It is the same offset for every run.

## Before and after the recognition rework

"Before" is the working tree as it stood when the rework started
(`bench/results/base-fast.json`, `base-balanced.json`). Real time, all eight
samples.

| | Fast, before | Fast, after | Balanced, before | Balanced, after |
|---|---|---|---|---|
| Japanese errors (CER) | 18.9% | **15.1%** | 16.5% | **14.6%** |
| English vs reference (chrF) | 56.9 | **59.2** | 55.2 | **56.5** |
| Sentences that got a finished line | 128 / 129 | 129 / 129 | 126 / 129 | 129 / 129 |
| Final English after end, typical | 0.72 s | 0.71 s | 0.90 s | see below |
| Final English after end, worst 1 in 10 | 1.62 s | **1.50 s** | 2.20 s | see below |
| First English after a sentence starts | 1.27 s | 1.29 s | 1.42 s | 1.44 s |
| Live Japanese settles after the word | 0.40 s | 0.58 s | 0.55 s | 0.74 s |

Errors by sample, before -> after:

| sample | Fast | Balanced |
|---|---|---|
| `01_slow_clear` | 9.4% -> 3.0% | 3.8% -> 3.8% |
| `02_conversational` | 22.7% -> 17.6% | 20.4% -> 18.0% |
| `03_fast` | 25.5% -> 18.1% | 23.5% -> 13.1% |
| `04_multi_speaker` | 10.7% -> 7.8% | 16.3% -> 9.6% |
| `05_casual_slang` | 18.5% -> 16.0% | 16.3% -> 16.9% |
| `06_background_music` | 20.1% -> 17.8% | 17.1% -> 18.7% |
| `07_pauses` | 13.0% -> 8.3% | 11.3% -> 9.1% |
| `08_long_continuous` | 20.0% -> 17.6% | 16.2% -> 16.2% |

What that says, plainly:

- **Recognition is markedly better** - a fifth fewer errors on Fast, an eighth
  fewer on Balanced, and no sentence left without a finished line. Balanced got
  slightly worse on two samples (slang, and speech over music) and a great deal
  better on two others.
- **The typical delay barely moved.** Fast was already close to its floor,
  which is: notice the pause, read the utterance with the silence after it,
  translate it. The rework no longer runs a closing pass when nothing new has
  been said, which is worth a pass (0.12-0.19 s) on sentences that end cleanly,
  and gives some of that back on sentences that end on は or けど, which are now
  held with their audio. The worst case improved a little.
- **The live Japanese line trails by about 0.2 s more.** That is
  `commit_holdback_sec`, and it is where a good share of the accuracy came from.
  Finished lines are not delayed by it.

**Balanced's delay could not be measured cleanly.** A game was running on the
same GPU during these runs and its load came and went, so recognition passes
took anywhere from 157 to 330 ms in both the before and the after run, on
different samples. Sample by sample, with the pass time each run actually got:

| sample | pass, before -> after | typical delay, before -> after |
|---|---|---|
| `01_slow_clear` | 168 -> 164 ms | 0.85 -> 0.72 s |
| `02_conversational` | 181 -> 206 ms | 0.94 -> 0.88 s |
| `03_fast` | 184 -> 213 ms | 0.97 -> 1.22 s |
| `04_multi_speaker` | 162 -> 269 ms | 0.80 -> 1.07 s |
| `05_casual_slang` | 160 -> 253 ms | 0.80 -> 1.08 s |
| `06_background_music` | 182 -> 187 ms | 0.89 -> 0.83 s |
| `07_pauses` | 164 -> 201 ms | 0.86 -> 1.01 s |
| `08_long_continuous` | 330 -> 157 ms | 1.12 -> 1.05 s |

Where the two runs got the same GPU (`01`, `06`) the delay is 0.06-0.13 s
shorter than before. Where the after run was starved (`04`, `05`) it is longer,
by about what two slower passes cost. `03_fast` is longer on its own account:
its sentences run on without a gap and several now wait for an ending that the
old pipeline cut through. Take this again with the GPU to itself before quoting
a figure for Balanced.

### Where the errors were

The recogniser on its own, given each reference sentence cut out and handed over
whole, makes about 11% errors on this set (see below). The live pipeline was
making 19%. The gap was not the model:

1. **Wrong early guesses were permanent.** The first two passes over the opening
   half second of an utterance agreed on a hallucinated phrase often enough to
   matter, and settled text could never change, so every later pass - each of
   which read the same audio correctly - was discarded as disagreeing with it.
   Settled words can now be revised until their line is handed to the
   translator.
2. **One early disagreement broke every later join.** New words were lined up
   against *all* of the settled text; once that no longer matched, the join fell
   back on timestamps and doubled syllables down the rest of the line
   (似たようような, ニュースス). Only the tail is lined up now.
3. **Words settled at the edge of the buffer.** The word the buffer ends in the
   middle of is guessed the same wrong way by two consecutive passes.
4. **The buffer was cut mid-word and started cold.** It now keeps a couple of
   seconds of settled speech at its head and is cut in a gap between words.
5. **A finished sentence left in the buffer hid the next one.** Kotoba-Whisper
   and Anime-Whisper transcribe the first sentence in a buffer and stop. In
   `08_long_continuous` the reading stayed on a sentence that had already been
   translated for three seconds at a time while the next was being spoken, then
   jumped to the middle of it. The buffer is now cut at the pause once the
   sentence before it is settled, with a watchdog for when the text test misses.
6. **Pauses were matched to words by timestamps** that are a token out on
   Japanese (かな | り広い範囲). They are matched by what had been read when the
   pause was heard.
7. **A mid-sentence pause split the audio as well as the text.** 何さん、だっけ
   came back as さんだ and っけけ. If the text does not read as finished the
   audio is kept and the recogniser hears the sentence whole.
8. **The recogniser sometimes just stops,** a second or more short of the audio,
   in the middle of a sentence with no pause in it, and stays stopped pass after
   pass. Whether it happens depends on the exact buffer length a pass lands on,
   so the same recording lost forty characters in one real-time run and none in
   the next - `03_fast` came out anywhere from 17.6% to 26.5% on identical
   code. A reading that stays short of the audio for two passes running now has
   what it read settled and the buffer cut where it stopped, and the closing
   pass is no longer skipped after such a cut, which was losing the last words
   of a line.

Run counters from the Fast run, for scale: 129 sentences; 159 corrections to
settled text; 115 buffer cuts at a pause; 16 recogniser stalls broken; 62 pauses
waited out because the text was unfinished; 150 closing passes not run because
nothing new had been said.

## Tried and not kept

- **Closing a sentence 0.2 s into a pause.** It put errors up from 15.6% to
  16.2%, nearly all of it on `03_fast` and `07_pauses`: at 0.2 s the speaker has
  often only drawn breath, and the line is cut in the middle of なんとなく.
  `eager_endpoint_sec` is 0.3 s.
- **A second pass to confirm an ending before taking it.** 15.4% with it, 15.5%
  without, at the same 0.3 s of patience. Two readings of the same audio by the
  same model agree with each other whether or not they are right; it was the
  waiting that helped. The confirming pass cost a pass of delay on every
  sentence and was removed.
- **Re-reading any speech the last pass left unread, down to 0.8 s.** It
  recovered whole sentences the recogniser had skipped - and sent a laugh after a
  last word back to be recognised by itself, which came back as a sentence
  nobody said. The limit is 1.5 s.
- **Whisper large-v3-turbo instead of Kotoba-Whisper** in the same live setting:
  15.4% against 15.6%. Better on slang and on continuous speech (it does not
  stall at sentence ends: 2 stalls against 19), worse over music and with long
  pauses, and a little slower per pass. Not a reason to change the default.
- **Dropping low-confidence text.** The recogniser's word probabilities do not
  separate what it invents from what it hears: a hallucinated line averaged 0.79
  and a correct short one 0.87.

## The recognisers on their own

`python -m bench.asr_bench`. Each reference sentence cut out and transcribed by
itself - the best a recogniser can do on this set, with segmentation taken out
of the question.

| model | CER | pass over a 2 / 4 / 8 / 12 s buffer (beam 1) | VRAM |
|---|---|---|---|
| Kotoba-Whisper v2.0 | 10.8% | 82 / 90 / 97 / 136 ms | 2.2 GB |
| Whisper large-v3-turbo | 9.7% | 87 / 98 / 131 / 171 ms | 2.1 GB |
| Whisper large-v3 | 11.7% | 170 / 217 / 349 / 480 ms | 3.9 GB |
| Anime-Whisper | 16.0% | 158 / 231 / 175 / 201 ms | 2.0 GB |

Beam 5 against beam 1 on Kotoba: 10.8% against 10.9%, for about 20 ms more per
pass. Anime-Whisper is the weakest of the four here; only `05_casual_slang`,
which is its home ground, goes its way (8.3% against 10.8%).

## The translators on their own

`python -m bench.mt_bench`. The 129 reference sentences, translated from the
clean Japanese and from a recogniser's version of it.

| translator | chrF, clean | chrF, recognised | per line | VRAM |
|---|---|---|---|---|
| Sugoi v4, beam 5 | 59.6 | 55.7 | 30 ms | 1.5 GB |
| Sugoi v4, beam 1 | 61.0 | 56.6 | 25 ms | 1.5 GB |
| Qwen3 4B Instruct, 4 lines of context | 55.8 | 54.5 | 116 ms | 3.0 GB |
| Sugoi 14B Ultra, 4 lines of context | 64.8 | 63.0 | 304 ms | 8.8 GB |

Worth knowing when choosing a profile: by this measure the 4B model does not
beat the built-in translator it replaces, on clean text or recognised text, at
four times the cost per line. chrF undersells a model that translates freely, so
this is not proof that its lines read worse - but nothing here says they read
better. The 14B specialist is the one that is clearly ahead.

## Stream slang, and the second look

`python -m bench.refine_bench`. These are measured on text, not audio, because
they depend on lines following from each other and the audio samples are
unrelated sentences. `bench/refine_cases.json` is a scripted VTuber-style
stream of 28 lines, each given three ways: as said, as a recogniser typically
writes it (homophones, slang spelt as something else, no question marks), and
in English. 17 of the 28 contain a term from the glossary. It was written for
this benchmark by the same hand as the glossary, so read it as "does the
mechanism work", not as an accuracy figure for real streams.

### The glossary

First-pass translation of the script, with and without the notes and
respelling from `subordinant/glossary.py`:

| translator | chrF without | chrF with |
|---|---|---|
| Qwen3 4B Instruct | 41.8 | 46.1 |
| Sugoi 14B Ultra | 47.2 | 56 - 60 over four runs |

Japanese errors against what was said: 4.7% as heard, 2.7% once the known
mishearings of slang are respelt (突待ち to 凸待ち, 同説 to 同接, ていてい to
てぇてぇ). That step uses no model. What the notes change, on the 14B:

```
来週の土曜日に突待ちをやります。
  without   I'm doing a tachiai next Saturday.
  with      I'll be doing a call-in stream next Saturday.
昨日のガチャ、天井まで回したのに爆死した
  without   I pulled the gacha all the way to the ceiling yesterday, but I still got nothing.
  with      I pulled the gacha all the way to the pity limit yesterday, but I still got nothing.
支持コメはなしでお願いね
  without   No likes on the comments, please.
  with      No backseat comments, please.
```

The audio samples are the check on the other side - that ordinary Japanese is
left alone. None of their 129 reference sentences contains a glossary term or
a respelling. (Two did at first: ないまま matched ママ, because matching folds
katakana and hiragana together. That entry now matches katakana only.)

### The second look

What ships is a rule: a line that stops on のに, けど, から or ので and is
followed at once by another has the ending of its English reopened. On the
script it joins 3 lines of 28 (14B first pass; 2 with the 4B), on the saved
`balanced` run of the audio samples 3 of 158, and it changes no Japanese. chrF
moves by 0.1 either way, which is to say the metric does not see it; it is for
the reader, who otherwise gets two sentences where one was said.

It is all that is left of a larger idea, and the rest is recorded here because
each part was measured before it was removed. Sugoi 14B throughout unless
said; "real output" is the finished lines of a saved `bench.run`, checked
against the samples' reference text.

| what the model was asked to do | script (28 lines) | real output |
|---|---|---|
| Return the whole line, Japanese and English | Japanese errors 4.7% to 2.9%, no line damaged; English 59.3 to 59.1 | four samples, live, Accuracy profile: kept Japanese 19.5% to 19.5%, English 59.4 to 59.5; final English 0.43 s later (p50), 0.73 s (p90) |
| The same, Qwen3 4B | English 46.2 to 44.6: it writes the next line into this one | not run |
| Name misheard words as pairs; one-kanji swaps only | 2.7% to 2.4%, 1 line repaired, none damaged | 14.64% to 14.84%: 12 lines "fixed" in 158, 7 more characters wrong |
| Also say "this was a question" / "wrong subject" | 3 statements turned into questions, 1 right subject swapped | not run |
| Only reopen the ending of a line known to run on | does it; chooses "but"/"so", which the particle already says | "Um..." to "Um," - replaced by the rule |

Some of what the word-fixing form did to real output:

```
歯が痛いので口を開けることができません      ->  天が痛いので...
  My tooth hurts so I can't open my mouth.  ->  The weather feels so bad that I can't open my mouth.
無罪の人々はもちろん放免された              ->  無事の人々は...
  Of course, the innocent people were released.  ->  Of course, the unharmed people were released.
それ以外はよく思い出せない                  ->  それ意外は...     (以外 was right)
```

The last one is worth a sentence: the instructions at that point gave 以外/意外
as an example of a homophone error, and the model applied the example to the
first line that contained it. The instructions have no examples now.

Two cautions about these figures. The real output is unrelated sentences, which
is the worst case for anything that reads a line against its neighbours - on a
real stream the context would help more than it does here, and no such
recording with a checked transcript was to hand. And the models are a
Japanese-to-English specialist and a 4B generalist, neither of them a
proofreader. The model path is still in the code (`refine_fix_words`,
`refine_model`) so that a different model can be put through the same test:

```
.venv\Scripts\python.exe -m bench.refine_bench --show --fix-words --llm <ollama tag>
.venv\Scripts\python.exe -m bench.refine_bench --transcripts balanced --fix-words --llm <ollama tag>
```

It earns a place when the second of those lowers the error count.

## Taking them again

The audio is not in the repository: it is cut from public speech datasets that
are not ours to pass on. Build it once, which downloads the clips named in
`bench/manifest.json` (a few hundred MB, cached) and takes a couple of minutes:

```
.venv\Scripts\python.exe -m bench.build_samples     # writes bench/samples/*.wav
.venv\Scripts\python.exe -m bench.align_samples     # times each character; needs large-v3
```

Then:

```
.venv\Scripts\python.exe -m bench.run --profile fast                  # real time, 13 min
.venv\Scripts\python.exe -m bench.run --profile balanced
.venv\Scripts\python.exe -m bench.run --profile fast --fast-replay    # accuracy only, ~5 min
.venv\Scripts\python.exe -m bench.run --config my.json --label trial  # any settings
.venv\Scripts\python.exe -m bench.report base-fast fast --samples     # side by side
```

Results are written to `bench/results/<label>.json`. `base-fast` and
`base-balanced` there are the "before" runs above. Give the GPU to the run: a
second model working at the same time shows up as latency that is not the
pipeline's.

To see why a line came out the way it did, the engine counters in a results file
(`engine.<sample>`) say how each utterance ended, and `--fast-replay` on one
sample with `--samples <id>` takes under a minute.
