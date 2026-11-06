"""TranslatorRuntime live loop with a scripted capture helper and scripted worker (no model)."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import TranslationPipeline
from app.runner import TranslatorRuntime
from core.config.models import AppConfig
from core.errors import PermissionDeniedError
from core.events import EventBus
from core.events.types import CaptionValidated, ErrorOccurred
from core.state import AppState
from tests.integration.test_pipeline_fake import ScriptedWorker, WindowVad, ok
from tests.platform.test_process_provider import ScriptedProvider


@pytest.fixture
def runtime():
    bus = EventBus()
    got = []
    bus.subscribe("t", handler=got.append)
    rt = TranslatorRuntime(AppConfig(), Path("unused"), bus)
    rt.state.transition(AppState.STARTING)
    rt.pipeline = TranslationPipeline(rt.cfg, WindowVad(), ScriptedWorker(ok()), bus)
    rt.state.transition(AppState.RUNNING)
    yield rt, bus, got
    bus.close()


def test_live_loop_turns_capture_into_captions(runtime):
    rt, bus, got = runtime
    # helper emits a loud 1 kHz tone (WindowVad treats loud as speech)
    provider = ScriptedProvider("--seconds=60", rate=48000, channels=2)
    stop = threading.Event()
    t = threading.Thread(target=rt.run_provider, args=(provider, stop), daemon=True)
    t.start()
    time.sleep(3.5)
    stop.set()
    t.join(15)
    bus.wait_idle()
    assert not t.is_alive()
    assert [e for e in got if isinstance(e, CaptionValidated)]
    assert provider.get_status().state.value in {"stopped", "failed"}  # cleaned up


def test_capture_failure_is_published_and_raised_not_swallowed(runtime):
    rt, bus, got = runtime
    provider = ScriptedProvider("--exit-code=77")
    with pytest.raises(PermissionDeniedError):
        rt.run_provider(provider, threading.Event())
    bus.wait_idle()
    errs = [e for e in got if isinstance(e, ErrorOccurred)]
    assert errs and errs[0].component == "capture" and not errs[0].recoverable
    assert errs[0].error.hint  # actionable
    assert rt.state.state is AppState.ERROR


def test_replay_samples_and_clean_stop(runtime):
    rt, bus, got = runtime
    sil = np.zeros(16000, np.float32)
    tone = (0.3 * np.sin(np.arange(24000) * 0.3)).astype(np.float32)
    rt.run_samples(np.concatenate([sil, tone, sil, sil]), speed=20.0)
    bus.wait_idle()
    assert [e for e in got if isinstance(e, CaptionValidated)]
    rt.stop()
    assert rt.state.state is AppState.IDLE and rt.pipeline is None
