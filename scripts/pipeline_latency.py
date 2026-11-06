"""Measure real end-to-end latency at 1x real-time pacing: speaker stops -> caption emitted.

Usage: python scripts/pipeline_latency.py --models-dir models [--repeats 2]
Synthetic speech only (macOS `say` samples): this measures SPEED, not translation quality.
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runner import TranslatorRuntime  # noqa: E402
from core.config.models import AppConfig  # noqa: E402
from core.events import EventBus  # noqa: E402
from core.events.types import CaptionValidated  # noqa: E402
from tests.ai.conftest import load_wav  # noqa: E402

CODES = ["zh", "ja", "pt", "hi"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--verify-interval", type=int, default=1)
    a = ap.parse_args()
    gap = np.zeros(16000 * 3, np.float32)
    sections, ends = [gap], []
    pos = gap.size
    for _ in range(a.repeats):
        for c in CODES:
            w = load_wav(Path("tests/ai/data") / f"{c}.wav")
            sections += [w, gap]
            ends.append((pos + w.size) / 16000)  # when the speaker stops (stream seconds)
            pos += w.size + gap.size
    audio = np.concatenate(sections)

    bus = EventBus()
    caps: list[tuple[float, CaptionValidated]] = []
    bus.subscribe("m", CaptionValidated, handler=lambda e: caps.append((time.monotonic(), e)))
    rt = TranslatorRuntime(
        AppConfig(language={"verify_interval": a.verify_interval}), a.models_dir, bus
    )
    t_load = time.monotonic()
    rt.start()
    print(f"worker+model ready in {time.monotonic() - t_load:.1f}s")
    t0 = time.monotonic()
    rt.run_samples(audio, speed=1.0)
    bus.wait_idle()
    st = rt.pipeline.stats if rt.pipeline else None
    rt.stop()
    bus.close()
    print(f"captions {len(caps)}/{len(ends)}  stats={st}")
    if len(caps) != len(ends):
        print("caption count != utterance count: latency pairing would be wrong, not reporting")
        return
    lat = sorted(when - t0 - end_s for (when, _), end_s in zip(caps, ends, strict=True))
    p90 = lat[int(0.9 * (len(lat) - 1))]
    print(
        f"speaker-stop -> caption latency: min {lat[0]:.2f}s  "
        f"median {statistics.median(lat):.2f}s  p90 {p90:.2f}s  max {lat[-1]:.2f}s"
    )
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    print(f"parent peak RSS: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
