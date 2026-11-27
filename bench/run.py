"""Run the whole live pipeline over the benchmark samples and score it.

    python -m bench.run --profile balanced
    python -m bench.run --profile fast --samples 02_conversational 06_background_music
    python -m bench.run --config my.json --label experiment
    python -m bench.run --all-profiles

Each sample is replayed in real time through the same code the app runs -
gate, VAD, streaming recogniser, translator, all of it - with only the WASAPI
device swapped for a file. A run of every sample takes about thirteen minutes
per profile because the audio is thirteen minutes long; that is the point.

Results land in bench/results/<label>.json. `python -m bench.report` renders
them side by side.

This file also runs unmodified against the pipeline as it stood before the
translation overhaul (commit 4ecbfde), which is how the baseline figures in
BENCHMARKS.md were taken: check that commit out in a worktree, copy bench/ in
beside it, and run with --config pointing at the settings to measure.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

from bench.build_samples import load_sample, sample_ids
from bench.replay import FileCapture
from bench.scoring import merge, score

RESULTS = Path(__file__).resolve().parent / "results"


class Recorder:
    """Collects what the pipeline emits, stamped on the audio clock."""

    def __init__(self, capture: FileCapture):
        self.capture = capture
        self.commits: list[dict] = []
        self.previews: list[dict] = []
        self.finals: list[dict] = []
        self.statuses: list[dict] = []
        self.states: list[dict] = []
        self.calls: list[dict] = []
        # Source text of the translation call in flight or most recently
        # finished, so a preview can be tied to the text it was made from.
        self.current_source = ""
        self._lock = threading.Lock()

    def now(self) -> float:
        return round(self.capture.audio_clock(), 3)

    def commit(self, text: str, speaker: int = -1) -> None:
        self.commits.append({"t": self.now(), "text": text})

    def caption(self, committed: str, unstable: str) -> None:
        pass  # the commit stream already carries everything scored

    def status(self, message: str) -> None:
        self.statuses.append({"t": self.now(), "message": message})

    def preview(self, english: str, source: str | None = None) -> None:
        if not english:
            return
        self.previews.append(
            {"t": self.now(), "english": english, "source": source or self.current_source}
        )

    def final(self, source: str, english: str, speaker: int = -1, ident: int = 0) -> None:
        self.finals.append(
            {"t": self.now(), "source": source, "english": english, "id": ident}
        )

    def revise(self, ident: int, source: str, english: str) -> None:
        """A second-look correction: the line is scored as it ended up, at the
        time it was first shown. What it said first is kept beside it."""
        for line in reversed(self.finals):
            if line.get("id") == ident:
                line["first_source"], line["first_english"] = line["source"], line["english"]
                line["source"], line["english"] = source, english
                line["revised_at"] = self.now()
                return

    def record(self) -> dict:
        return {
            "commits": self.commits,
            "previews": self.previews,
            "finals": self.finals,
        }


class TimedTranslator:
    """Wraps a translator to time its calls and expose what it is working on."""

    def __init__(self, inner, recorder: Recorder):
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def translate(self, text: str, on_partial=None, **kwargs):
        rec = self._recorder
        rec.current_source = text
        t0 = time.perf_counter()
        first: list[float] = []

        def partial(piece: str) -> None:
            if not first:
                first.append(time.perf_counter() - t0)
            if on_partial is not None:
                on_partial(piece)

        if on_partial is not None:
            result = self._inner.translate(text, on_partial=partial, **kwargs)
        else:
            result = self._inner.translate(text, **kwargs)
        rec.calls.append(
            {
                "sec": round(time.perf_counter() - t0, 4),
                "first_token_sec": round(first[0], 4) if first else None,
                "chars": len(text),
                "streamed": on_partial is not None,
            }
        )
        return result


class SystemSampler(threading.Thread):
    """GPU memory and utilisation, plus this process's CPU and RAM, once a second."""

    def __init__(self) -> None:
        super().__init__(name="bench-sampler", daemon=True)
        self._stop_event = threading.Event()
        self.gpu_mem: list[int] = []
        self.gpu_util: list[int] = []
        self.rss: list[float] = []
        self.cpu: list[float] = []
        self.ollama_rss: list[float] = []

    def run(self) -> None:
        try:
            import psutil

            me = psutil.Process()
            me.cpu_percent(None)
        except ImportError:
            psutil = None
            me = None
        while not self._stop_event.wait(1.0):
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).stdout.strip()
                mem, util = (int(x) for x in out.splitlines()[0].split(","))
                self.gpu_mem.append(mem)
                self.gpu_util.append(util)
            except Exception:  # noqa: BLE001 - no NVIDIA GPU is a valid machine
                pass
            if me is not None:
                self.rss.append(me.memory_info().rss / 2**20)
                # Percent of one core; divide by the core count for "of the CPU".
                self.cpu.append(me.cpu_percent(None))
                total = 0.0
                for proc in psutil.process_iter(["name", "memory_info"]):
                    name = (proc.info.get("name") or "").lower()
                    if name.startswith("ollama") and proc.info.get("memory_info"):
                        total += proc.info["memory_info"].rss / 2**20
                self.ollama_rss.append(total)

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=3.0)

        def peak(values):
            return round(max(values), 1) if values else None

        def mean(values):
            return round(sum(values) / len(values), 1) if values else None

        return {
            "gpu_mem_peak_mib": peak(self.gpu_mem),
            "gpu_util_mean_pct": mean(self.gpu_util),
            "process_rss_peak_mib": peak(self.rss),
            "process_cpu_mean_pct_of_one_core": mean(self.cpu),
            "ollama_rss_peak_mib": peak(self.ollama_rss),
        }


def gpu_mem_now() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


# -- adapters ---------------------------------------------------------------


class _Session:
    """One loaded pipeline that successive samples are played through.

    Models are loaded once and kept, as they are in the app: reloading per
    sample would measure the loader rather than the pipeline and would triple
    the run time.
    """

    def __init__(self, cfg, realtime: bool = True):
        self.cfg = cfg
        self.realtime = realtime
        self.transcriber = None
        self.translators: dict = {}
        self.load_sec: float | None = None
        self.cache = None

    def play(self, audio, name: str) -> tuple[Recorder, dict]:
        import subordinant.pipeline as pipeline_module

        capture = FileCapture(audio, name=name, realtime=self.realtime)
        recorder = Recorder(capture)
        session = self

        accepts_source = (
            "source"
            in inspect.signature(pipeline_module.CaptionPipeline.__init__).parameters
        )
        t_load = time.perf_counter()
        if accepts_source:
            pipe = _build_current(pipeline_module, self, capture, recorder)
        else:
            pipe = _build_legacy(pipeline_module, self, capture, recorder)

        pipe.start()
        # The replay clock starts when the pipeline opens the "device", which
        # is after its models have loaded.
        while capture.t0 is None:
            if not pipe._thread or not pipe._thread.is_alive():
                raise RuntimeError(
                    "pipeline died during startup: "
                    + "; ".join(s["message"] for s in recorder.statuses[-3:])
                )
            time.sleep(0.05)
        if session.load_sec is None:
            session.load_sec = round(time.perf_counter() - t_load, 2)
        capture.finished.wait()
        # Let the last utterance close and its translation come back.
        deadline = time.perf_counter() + 6.0
        quiet_since = time.perf_counter()
        seen = len(recorder.finals)
        while time.perf_counter() < deadline:
            time.sleep(0.1)
            if len(recorder.finals) != seen:
                seen = len(recorder.finals)
                quiet_since = time.perf_counter()
            if time.perf_counter() - quiet_since > 2.0:
                break
        pipe.stop()
        self.transcriber = getattr(pipe, "transcriber", None) or self.transcriber
        stats = _engine_stats(pipe)
        stats["mt_calls"] = _call_stats(recorder.calls)
        return recorder, stats


def _build_legacy(module, session: _Session, capture, recorder):
    """The pipeline as it was before the overhaul: no injectable source."""
    module.LoopbackCapture = lambda frames_per_buffer=1024, **_kw: capture
    original = getattr(module, "_bench_original_build", None) or module.build_translator
    module._bench_original_build = original

    def build(cfg):
        key = (cfg.mt_backend, cfg.mt_model, cfg.llm_model)
        if key not in session.translators:
            session.translators[key] = original(cfg)
        inner = session.translators[key]
        # A fresh conversation per sample: one sample's lines must not be the
        # LLM's context for the next.
        history = getattr(inner, "_history", None)
        if history is not None:
            history.clear()
        return TimedTranslator(inner, recorder)

    module.build_translator = build
    return module.CaptionPipeline(
        session.cfg,
        on_caption=recorder.caption,
        on_status=recorder.status,
        transcriber=session.transcriber,
        on_commit=recorder.commit,
        on_translation=recorder.final,
        on_live_translation=recorder.preview,
    )


def _build_current(module, session: _Session, capture, recorder):
    """The current pipeline, which takes its audio source as an argument."""
    from subordinant.pipeline import CaptionPipeline, PipelineEvents

    from subordinant.models import ModelCache

    if session.cache is None:
        session.cache = ModelCache()

    class Events(PipelineEvents):
        def on_commit(self, text, speaker, start, end):
            recorder.commit(text, speaker)

        def on_preview(self, english, source):
            recorder.preview(english, source)

        def on_utterance(self, utterance):
            if utterance.final:
                # Whisper translating directly has no Japanese to report; the
                # English is the only text there is to place in the reference.
                recorder.final(
                    utterance.source, utterance.english, utterance.speaker, utterance.id
                )
            elif utterance.english:
                # A draft line in the transcript is provisional English too.
                recorder.preview(utterance.english, utterance.source)

        def on_revision(self, utterance):
            recorder.revise(utterance.id, utterance.source, utterance.english)

        def on_state(self, state, detail):
            recorder.states.append({"t": recorder.now(), "state": state, "detail": detail})

        def on_notice(self, level, message):
            recorder.status(f"{level}: {message}")

    pipe = CaptionPipeline(
        session.cfg,
        events=Events(),
        source=capture,
        models=session.cache,
        wrap_translator=lambda inner: TimedTranslator(inner, recorder),
    )
    return pipe


def _engine_stats(pipe) -> dict:
    stats = pipe.stats
    if hasattr(stats, "snapshot"):
        return dict(stats.snapshot())
    return {
        "passes": stats.passes,
        "mean_pass_ms": round(stats.mean_pass_ms, 1),
        "rtf": round(stats.rtf, 3),
        "drops": stats.drops,
        "engine_commit_lag_mean": round(stats.mean_lag_sec, 3),
        "engine_commit_lag_max": round(stats.max_lag_sec, 3),
    }


def _call_stats(calls: list[dict]) -> dict:
    if not calls:
        return {"n": 0}
    secs = sorted(c["sec"] for c in calls)
    firsts = [c["first_token_sec"] for c in calls if c["first_token_sec"] is not None]
    return {
        "n": len(calls),
        "mean_ms": round(1000 * sum(secs) / len(secs), 1),
        "p50_ms": round(1000 * secs[len(secs) // 2], 1),
        "p90_ms": round(1000 * secs[int(len(secs) * 0.9)], 1),
        "max_ms": round(1000 * secs[-1], 1),
        "first_token_mean_ms": (
            round(1000 * sum(firsts) / len(firsts), 1) if firsts else None
        ),
    }


# -- driver -----------------------------------------------------------------


def load_config(profile: str | None, config_path: str | None):
    from subordinant.config import Config

    cfg = Config()
    if profile:
        from subordinant.profiles import apply_profile

        apply_profile(cfg, profile)
    if config_path:
        known = {f.name for f in fields(Config)}
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                print(f"  (ignoring unknown config key {key!r})")
    return cfg


def run(
    cfg, label: str, samples: list[str], note: str = "", realtime: bool = True
) -> dict:
    from subordinant import logsetup

    logsetup.configure(False)
    session = _Session(cfg, realtime=realtime)
    gpu_before = gpu_mem_now()
    sampler = SystemSampler()
    sampler.start()

    per_sample: dict[str, dict] = {}
    transcripts: dict[str, dict] = {}
    engine: dict[str, dict] = {}
    for sample_id in samples:
        audio, sidecar = load_sample(sample_id)
        print(f"[{label}] {sample_id} ({sidecar['duration']:.0f}s) ...", flush=True)
        recorder, stats = session.play(audio, sample_id)
        result = score(recorder.record(), sidecar)
        per_sample[sample_id] = result
        engine[sample_id] = stats
        transcripts[sample_id] = {
            "finals": recorder.finals,
            "statuses": [s for s in recorder.statuses if "listening" not in s["message"]][-8:],
        }
        fa = result["final_after_sentence_end"]
        fe = result["first_english_after_start"]
        print(
            f"    CER {result['cer']:.1%}  transcript CER {result['transcript_cer'] or 0:.1%}  "
            f"chrF {result['chrf']:.1f}  "
            f"first-EN p50 {fe.get('p50', float('nan')):.2f}s  "
            f"final-after-end p50 {fa.get('p50', float('nan')):.2f}s "
            f"p90 {fa.get('p90', float('nan')):.2f}s  "
            f"finalised {result['sentences_finalised']}/{result['sentences']}  "
            f"pass {stats.get('mean_pass_ms', 0):.0f} ms",
            flush=True,
        )

    system = sampler.stop()
    system["gpu_mem_before_mib"] = gpu_before
    if gpu_before is not None and system["gpu_mem_peak_mib"] is not None:
        system["gpu_mem_added_mib"] = round(system["gpu_mem_peak_mib"] - gpu_before, 1)

    out = {
        "label": label,
        "note": note,
        # False for a fast replay: recognition and segmentation are as they
        # would be live, every latency figure is meaningless.
        "realtime": realtime,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            f.name: getattr(cfg, f.name)
            for f in fields(type(cfg))
            if getattr(cfg, f.name) != getattr(type(cfg)(), f.name)
        },
        "load_sec": session.load_sec,
        "summary": merge(per_sample),
        "system": system,
        "samples": per_sample,
        "engine": engine,
        "transcripts": transcripts,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{label}] wrote {path}")
    print(f"[{label}] summary: {json.dumps(out['summary'])}")
    print(f"[{label}] system:  {json.dumps(system)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="fast, balanced or accuracy")
    parser.add_argument("--all-profiles", action="store_true")
    parser.add_argument("--config", help="JSON of Config overrides, applied last")
    parser.add_argument("--label", help="name for the results file")
    parser.add_argument("--note", default="", help="free text stored with the results")
    parser.add_argument("--samples", nargs="*", help="sample ids; default all")
    parser.add_argument(
        "--fast-replay",
        action="store_true",
        help="feed audio as fast as it is consumed: accuracy only, no latency",
    )
    args = parser.parse_args(argv)
    realtime = not args.fast_replay

    samples = args.samples or sample_ids()
    if args.all_profiles:
        from subordinant.profiles import PROFILES

        for name in PROFILES:
            run(
                load_config(name, args.config),
                args.label or name,
                samples,
                args.note,
                realtime,
            )
        return 0
    cfg = load_config(args.profile, args.config)
    label = args.label or args.profile or "custom"
    run(cfg, label, samples, args.note, realtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
