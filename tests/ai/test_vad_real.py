"""Real Silero VAD on real/synthetic signals (offline: model is bundled in faster-whisper)."""

import numpy as np
import pytest

from ai.vad import SileroStreamingVad
from core.config.models import VadConfig
from core.pipeline.segmenter import SpeechSegmenter
from core.types import AudioFrame

pytestmark = pytest.mark.ai


def segment(audio, chunk=1600, tail_s=2.0):
    seg = SpeechSegmenter(SileroStreamingVad(), VadConfig())
    audio = np.concatenate([audio, np.zeros(int(tail_s * 16000), np.float32)])
    out = []
    for i in range(0, audio.size, chunk):
        out += seg.feed(AudioFrame(i / 16, audio[i : i + chunk].copy()))
    return seg, out + seg.flush()


@pytest.mark.parametrize("code", ["zh", "ja", "pt", "ar", "hi", "en"])
def test_each_sample_is_one_utterance_covering_speech(speech, code):
    a = speech(code)
    _, out = segment(a)
    assert 1 <= len(out) <= 2
    covered = sum(u.speech_ms for u in out) / 1000
    assert covered > 0.6 * a.size / 16000


def test_non_speech_produces_no_utterances():
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 4) / 16000
    cases = {
        "silence": np.zeros(16000 * 4, np.float32),
        "white": (rng.standard_normal(16000 * 4) * 0.05).astype(np.float32),
        "tone": (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
    }
    for name, a in cases.items():
        _, out = segment(a)
        assert out == [], name


def test_stream_state_makes_chunking_irrelevant(speech):
    a = speech("ja")
    ref = [(round(u.start_ms / 32), round(u.end_ms / 32)) for u in segment(a, 512)[1]]
    for chunk in (160, 1600, 4001):
        got = [(round(u.start_ms / 32), round(u.end_ms / 32)) for u in segment(a, chunk)[1]]
        assert got == ref
