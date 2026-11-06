"""Segmenter logic with a scripted VAD (deterministic; no model)."""

import numpy as np

from core.config.models import VadConfig
from core.pipeline.segmenter import WINDOW, SpeechSegmenter
from core.types import AudioFrame, samples_to_ms

WIN_MS = samples_to_ms(WINDOW)  # 32 ms


class ScriptedVad:
    """Probability per window comes from a function of the window's absolute index."""

    def __init__(self, fn):
        self.fn, self.i = fn, 0

    def process(self, samples):
        n = samples.size // WINDOW
        out = [self.fn(self.i + k) for k in range(n)]
        self.i += n
        return out

    def reset(self):
        pass


def run(fn, seconds, cfg=None, chunk=1600, start_ms=1000.0):
    seg = SpeechSegmenter(ScriptedVad(fn), cfg or VadConfig())
    rng = np.random.default_rng(0)
    total = int(seconds * 16000)
    sig = rng.standard_normal(total).astype(np.float32) * 0.01
    out, pos = [], 0
    while pos < total:
        n = min(chunk, total - pos)
        out += seg.feed(AudioFrame(start_ms + pos / 16, sig[pos : pos + n].copy()))
        pos += n
    return seg, out


def span(a_s, b_s):
    return lambda i: 0.9 if a_s <= i * WIN_MS / 1000 < b_s else 0.0


def test_one_utterance_with_preroll_and_endpoint():
    seg, out = run(span(1.0, 2.0), 4.0)
    assert len(out) == 1
    u = out[0]
    assert not u.forced_split
    # starts ~pre-roll (300 ms) before speech start; times are on the capture clock (+1000 ms)
    assert 1000 + 1000 - 340 <= u.start_ms <= 1000 + 1000 - 260
    assert 1000 + 2000 <= u.end_ms <= 1000 + 2000 + 200
    assert abs(u.speech_ms - 1000) < 100
    assert abs(u.samples.size / 16 - u.duration_ms) < 1


def test_short_blip_discarded():
    seg, out = run(span(1.0, 1.1), 3.0)
    assert out == [] and seg.discarded_short == 1


def test_short_pause_does_not_split():
    def f(i):
        return 0.9 if (1.0 <= i * WIN_MS / 1000 < 2.0 or 2.3 <= i * WIN_MS / 1000 < 3.0) else 0.0

    _, out = run(f, 5.0)
    assert len(out) == 1 and out[0].duration_ms > 2000


def test_long_pause_splits_in_two():
    _, out = run(
        lambda i: 0.9 if 1 <= i * WIN_MS / 1000 < 2 or 3 <= i * WIN_MS / 1000 < 4 else 0, 6
    )
    assert len(out) == 2 and out[0].end_ms <= out[1].start_ms + 1


def test_continuous_speech_force_split_with_overlap_and_bound():
    cfg = VadConfig()
    _, out = run(lambda i: 0.9, 30.0, cfg)
    assert len(out) >= 3
    assert all(u.duration_ms <= cfg.max_utterance_s * 1000 + 2 * WIN_MS for u in out)
    assert all(u.forced_split for u in out[:-1] if u.forced_split)
    for a, b in zip(out, out[1:], strict=False):
        assert (
            a.end_ms - b.start_ms >= cfg.overlap_ms - 2 * WIN_MS
        )  # words at the seam are repeated
        assert b.start_ms < a.end_ms


def test_flush_closes_open_utterance():
    seg, out = run(span(1.0, 99.0), 2.0)
    assert out == []
    flushed = seg.flush()
    assert len(flushed) == 1 and flushed[0].speech_ms > 500


def test_gap_in_capture_is_bridged_and_reported():
    seg = SpeechSegmenter(ScriptedVad(lambda i: 0.0), VadConfig())
    s = np.zeros(8000, np.float32)
    seg.feed(AudioFrame(0.0, s))
    seg.feed(AudioFrame(500.0 + 1000.0, s))  # 1 s of capture audio never arrived
    assert 900 < seg.gap_ms_total < 1100


def test_sample_exact_timing_independent_of_chunking():
    ref = None
    for chunk in (160, 1600, 1537, 7919):
        _, out = run(span(1.0, 2.0), 4.0, chunk=chunk)
        sig = [(round(u.start_ms), round(u.end_ms)) for u in out]
        ref = ref or sig
        assert sig == ref
