"""Benchmarks: reproducible Japanese test audio, and scripts that measure on it.

    python -m bench.build_samples      build bench/samples/*.wav from the manifest
    python -m bench.asr_bench          recognition accuracy and speed, per model
    python -m bench.mt_bench           translation quality and speed, per engine
    python -m bench.run                the whole live pipeline, per profile

See BENCHMARKS.md for what each one measures and the numbers they produced.
"""
