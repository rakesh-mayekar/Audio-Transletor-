"""Measure load time, RTF, latency and memory for an installed model on THIS machine."""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

from ai.model_manager import ModelManager
from ai.whisper_engine import WhisperEngine
from core.config import paths
from core.config.models import ModelConfig
from scripts import make_test_audio
from tests.ai.conftest import load_wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="small")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    eng = WhisperEngine(ModelManager(a.models_dir or paths.models_dir()))
    cfg = ModelConfig(name=a.model, cpu_threads=a.threads)
    t0 = time.perf_counter()
    eng.load_model(cfg)
    print(f"model={a.model} int8 cpu threads={eng._threads} load={time.perf_counter() - t0:.2f}s")
    eng.translate_audio(load_wav(make_test_audio.make("pt")), "pt")  # warm-up
    allms: list[float] = []
    for code in make_test_audio.SAMPLES:
        audio = load_wav(make_test_audio.OUT / f"{code}.wav")
        runs = [eng.translate_audio(audio, code) for _ in range(a.repeats)]
        ms = [r.inference_ms for r in runs]
        allms += ms
        print(
            f"{code}: audio={runs[0].audio_ms / 1000:.2f}s p50={statistics.median(ms):.0f}ms "
            f"rtf={statistics.median(r.rtf for r in runs):.2f}"
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    p50, worst = statistics.median(allms), max(allms)
    print(f"overall p50={p50:.0f}ms max={worst:.0f}ms peak_rss={rss_mb:.0f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
