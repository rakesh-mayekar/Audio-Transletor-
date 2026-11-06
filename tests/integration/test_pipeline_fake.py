"""Pipeline orchestration with a scripted worker (no model): routing, drops, rejections."""

import numpy as np
import pytest

from ai.engine import EngineOutput
from ai.inference_worker import WorkerRequest, WorkerResult
from app.pipeline import TranslationPipeline
from core.config.models import AppConfig, LanguageConfig
from core.events import EventBus
from core.events.types import (
    AudioDropped,
    CaptionRejected,
    CaptionValidated,
    LanguageConfirmed,
    TranslationStarted,
)
from core.types import AudioFrame, ModelDiagnostics, TranslationResult

_BUSES: list = []


@pytest.fixture(autouse=True)
def _close_buses():
    yield
    while _BUSES:
        _BUSES.pop().close()


class WindowVad:
    """Voiced when |amplitude| is large; deterministic stand-in for Silero."""

    def process(self, s):
        return [float(np.abs(s[i : i + 512]).mean() > 0.05) for i in range(0, s.size, 512)]

    def reset(self):
        pass


class ScriptedWorker:
    def __init__(self, reply, accept=True):
        self.reply, self.accept, self.reqs, self.out = reply, accept, [], []

    def submit(self, req: WorkerRequest) -> bool:
        if not self.accept:
            return False
        self.reqs.append(req)
        self.out.append(self.reply(req))
        return True

    def poll(self, timeout=0.0):
        return self.out.pop(0) if self.out else None


def tone(sec, amp=0.3):
    return (amp * np.sin(np.arange(int(sec * 16000)) * 0.3)).astype(np.float32)


def stream(pipe, sections):
    audio = np.concatenate(sections)
    for i in range(0, audio.size, 1600):
        pipe.feed(AudioFrame(i / 16, audio[i : i + 1600].copy()))
        while pipe.poll():  # a real loop polls as it feeds
            pass
    pipe.finish()


def ok(lang="ja", text="Hello world today", nsp=0.05, p=0.9):
    def reply(req):
        r = TranslationResult(
            lang,
            text,
            req.audio_start_ms,
            req.audio_start_ms + req.audio.size / 16,
            "small",
            diagnostics=ModelDiagnostics(nsp, -0.3, 1.2, p),
        )
        return WorkerResult(
            req.request_id,
            EngineOutput(
                [r], lang, p, 100.0, 1000.0, detected_language=lang, detected_probability=p
            ),
        )

    return reply


def collect(bus):
    got = []
    bus.subscribe("t", handler=got.append)
    return got


def make(worker, **lang):
    bus = EventBus()
    _BUSES.append(bus)
    cfg = AppConfig(language=LanguageConfig(**lang))
    return TranslationPipeline(cfg, WindowVad(), worker, bus), bus


def test_valid_caption_flow_and_language_confirmation():
    w = ScriptedWorker(ok())
    pipe, bus = make(w)
    got = collect(bus)
    sil = np.zeros(16000 * 2, np.float32)
    stream(pipe, [sil, tone(2), sil, tone(2), sil, tone(2), sil])
    while pipe.poll():
        pass
    bus.wait_idle()
    assert pipe.stats.utterances == 3 and pipe.stats.captions_valid >= 1
    assert w.reqs[0].language is None  # auto-detect first
    assert any(isinstance(e, LanguageConfirmed) and e.language == "ja" for e in got)
    assert w.reqs[-1].language == "ja"  # later requests are pinned to the confirmed language
    assert w.reqs[-1].verify_language and not w.reqs[0].verify_language  # ...but still verified
    bus.close()


def test_manual_language_is_forced_and_never_detected():
    w = ScriptedWorker(ok())
    pipe, bus = make(w, auto_detect=False, manual_source="pt")
    stream(pipe, [np.zeros(8000, np.float32), tone(1.5), np.zeros(32000, np.float32)])
    assert [r.language for r in w.reqs] == ["pt"]
    bus.close()


def test_hallucination_rejected_and_reported():
    pipe, bus = make(ScriptedWorker(ok(text="Thank you for watching!", nsp=0.85)))
    got = collect(bus)
    stream(pipe, [np.zeros(8000, np.float32), tone(1.5), np.zeros(32000, np.float32)])
    while pipe.poll():
        pass
    bus.wait_idle()
    rej = [e for e in got if isinstance(e, CaptionRejected)]
    assert rej and "no_speech" in rej[0].reason
    assert not [e for e in got if isinstance(e, CaptionValidated)]
    bus.close()


def test_backlog_drop_is_visible():
    pipe, bus = make(ScriptedWorker(ok(), accept=False))
    got = collect(bus)
    stream(pipe, [np.zeros(8000, np.float32), tone(1.5), np.zeros(32000, np.float32)])
    bus.wait_idle()
    drops = [e for e in got if isinstance(e, AudioDropped)]
    assert drops and drops[0].reason == "inference_backlog" and pipe.stats.dropped_backlog == 1
    assert not [e for e in got if isinstance(e, TranslationStarted)]
    bus.close()


def test_stale_and_unsupported_results():
    def reply(req):
        if len(pipe_ref) % 2 == 0:
            pipe_ref.append(1)
            return WorkerResult(req.request_id, stale=True)
        pipe_ref.append(1)
        return WorkerResult(
            req.request_id,
            error={
                "code": "LANGUAGE_UNSUPPORTED",
                "message": "x",
                "hint": None,
                "details": {"detected_language": "fr", "probability": 0.97},
            },
        )

    pipe_ref: list[int] = []
    pipe, bus = make(ScriptedWorker(reply))
    got = collect(bus)
    sil = np.zeros(16000, np.float32)
    stream(pipe, [sil, tone(1.5), sil, tone(1.5), sil])
    while pipe.poll():
        pass
    bus.wait_idle()
    reasons = sorted(e.reason for e in got if isinstance(e, AudioDropped))
    assert reasons == ["stale", "unsupported_language"]
    assert pipe.stats.dropped_stale == 1 and pipe.stats.dropped_unsupported == 1
    bus.close()


def test_worker_error_surfaces_without_crashing():
    pipe, bus = make(
        ScriptedWorker(
            lambda r: WorkerResult(
                r.request_id,
                error={"code": "INFERENCE_ERROR", "message": "boom", "hint": None, "details": {}},
            )
        )
    )
    stream(pipe, [np.zeros(8000, np.float32), tone(1.5), np.zeros(32000, np.float32)])
    while pipe.poll():
        pass
    assert pipe.stats.errors == 1
    bus.close()


def test_sustained_language_change_switches_after_hysteresis():
    state = {"n": 0}

    def reply(req):
        state["n"] += 1
        return ok(lang="ja" if state["n"] <= 3 else "pt")(req)

    w = ScriptedWorker(reply)
    pipe, bus = make(w)
    got = collect(bus)
    sil = np.zeros(16000 * 2, np.float32)
    stream(pipe, [sil] + [tone(2), sil] * 8)
    bus.wait_idle()
    langs = [e.language for e in got if isinstance(e, LanguageConfirmed)]
    assert langs[0] == "ja" and langs[-1] == "pt"  # switched, not flapping
    assert len(langs) == 2
    assert pipe.language_machine.detected_language == "pt"


@pytest.mark.parametrize(
    ("interval", "expected"), [(1, [True] * 4), (2, [False, True, False, True]), (0, [False] * 4)]
)
def test_verify_interval_controls_rechecks(interval, expected):
    w = ScriptedWorker(ok())
    bus = EventBus()
    _BUSES.append(bus)
    cfg = AppConfig(language=LanguageConfig(verify_interval=interval, min_detection_speech_s=1.0))
    pipe = TranslationPipeline(cfg, WindowVad(), w, bus)
    sil = np.zeros(16000 * 2, np.float32)
    stream(pipe, [sil] + [tone(2), sil] * 6)
    pinned = [r for r in w.reqs if r.language == "ja"]
    assert [r.verify_language for r in pinned[:4]] == expected
