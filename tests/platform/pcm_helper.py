"""Scripted stand-in for an OS audio helper: emits a sine as float32le on stdout."""

import argparse
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=int, default=48000)
ap.add_argument("--channels", type=int, default=2)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--freq", type=float, default=1000.0)
ap.add_argument("--exit-code", type=int, default=0)
ap.add_argument("--die-after", type=float, default=-1.0)  # exit abruptly after N seconds
ap.add_argument("--no-sleep", action="store_true")
a = ap.parse_args()

if a.exit_code and a.die_after < 0:
    print("helper: refusing to start", file=sys.stderr, flush=True)
    sys.exit(a.exit_code)

out = sys.stdout.buffer
step = int(a.rate * 0.02)
n = 0
t0 = time.monotonic()
while n / a.rate < a.seconds:
    t = (np.arange(step) + n) / a.rate
    s = (0.5 * np.sin(2 * np.pi * a.freq * t)).astype("<f4")
    out.write(np.repeat(s[:, None], a.channels, axis=1).tobytes())
    out.flush()
    n += step
    if 0 <= a.die_after <= time.monotonic() - t0:
        print("helper: simulated crash", file=sys.stderr, flush=True)
        sys.exit(a.exit_code or 3)
    if not a.no_sleep:
        time.sleep(0.02)
