"""End-to-end: synthetic recordings -> real VAD -> real Whisper worker process -> validator."""

import time
from pathlib import Path

import numpy as np
import pytest

from ai.inference_worker import InferenceWorker
from ai.vad import SileroStreamingVad
from app.pipeline import TranslationPipeline
from core.config.models import AppConfig
from core.events import EventBus
from core.events.types import AudioDropped, CaptionRejected, CaptionValidated
from core.types import AudioFrame
from tests.ai.conftest import MODELS, requires_model

pytestmark = [pytest.mark.ai, requires_model]


def run(sections, cfg=None):
    cfg = cfg or AppConfig()
    bus = EventBus()
    got = []
    bus.subscribe("t", handler=got.append)
    worker = InferenceWorker(
        "ai.whisper_engine:create_engine",
        {"models_dir": str(Path(MODELS)), "model_config": cfg.model},
        max_pending=4,
    )
    worker.start()
    try:
        pipe = TranslationPipeline(cfg, SileroStreamingVad(), worker, bus)
        audio = np.concatenate(sections)
        for i in range(0, audio.size, 1600):  # 100 ms frames replayed at 4x real time
            time.sleep(0.025)  # staleness is measured on the audio clock: don't outrun inference
            pipe.feed(AudioFrame(i / 16, audio[i : i + 1600].copy()))
            while pipe.poll():
                pass
        pipe.finish()
        pipe.drain(120)
        bus.wait_idle()
        return pipe, got
    finally:
        worker.stop()
        bus.close()


def test_mixed_stream_only_real_speech_becomes_captions(speech):
    sil = np.zeros(16000 * 2, np.float32)
    noise = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.03).astype(np.float32)
    pipe, ev = run([sil, speech("ja"), sil, noise, sil, speech("ja"), sil, speech("fr"), sil])
    caps = [e for e in ev if isinstance(e, CaptionValidated)]
    drops = [e for e in ev if isinstance(e, AudioDropped)]
    print("\nCAPTIONS:", [c.text for c in caps], "\nSTATS:", pipe.stats)
    assert len(caps) >= 2
    assert pipe.stats.utterances == 3  # noise and silence never reach the model
    assert all("weather" in c.text.lower() for c in caps)
    assert [d.reason for d in drops] == ["unsupported_language"]  # French refused, not forced
    assert not [e for e in ev if isinstance(e, CaptionRejected)]


def test_caption_timestamps_lie_inside_source_audio(speech):
    sil = np.zeros(16000 * 2, np.float32)
    a = speech("zh")
    _, ev = run([sil, a, sil])
    for c in (e for e in ev if isinstance(e, CaptionValidated)):
        assert 1500 <= c.start_ms < c.end_ms <= 2000 + a.size / 16 + 1500
