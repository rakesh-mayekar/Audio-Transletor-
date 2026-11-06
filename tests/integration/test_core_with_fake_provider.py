"""Phase 1 acceptance: the shared core runs end-to-end with a provider, with no OS-specific code."""

import numpy as np

from core.config import AppConfig
from core.events import AudioCaptured, AudioDropped, ErrorOccurred, EventBus
from core.state import ApplicationStateMachine, AppState
from oslayer.fake import FakeAudioCaptureProvider


def test_provider_to_bus_to_subscriber_and_clean_shutdown():
    cfg = AppConfig()
    bus = EventBus()
    app = ApplicationStateMachine()
    received = []
    bus.subscribe(
        "sink",
        AudioCaptured,
        handler=lambda e: received.append(e.frame),
        maxsize=cfg.audio.max_pending_frames,
    )

    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    provider = FakeAudioCaptureProvider(tone)

    app.transition(AppState.STARTING)
    provider.initialize()
    provider.select_device(cfg.audio.device_id)
    provider.start_capture()
    app.transition(AppState.RUNNING)

    pumped = 0
    while pumped < 100:
        f = provider.read_audio(0.2)
        if f is not None:
            bus.publish(AudioCaptured(frame=f))
            pumped += 1

    app.transition(AppState.STOPPING)
    provider.stop_capture()
    provider.cleanup()
    assert bus.wait_idle(5)
    bus.close()
    app.transition(AppState.IDLE)

    assert len(received) == 100
    ts = [f.pts_ms for f in received]
    assert ts == sorted(ts)
    assert all(f.samples.dtype == np.float32 and f.sample_rate == 16000 for f in received)
    assert app.state is AppState.IDLE
    assert provider.get_status().state.value == "stopped"


def test_device_loss_surfaces_as_error_event_and_app_error_state():
    from core.errors import AudioCaptureError

    bus = EventBus()
    sub = bus.subscribe("err", ErrorOccurred, AudioDropped)
    app = ApplicationStateMachine()
    p = FakeAudioCaptureProvider()
    p.initialize()
    app.transition(AppState.STARTING)
    p.start_capture()
    app.transition(AppState.RUNNING)
    p.simulate_device_loss()
    try:
        for _ in range(100):
            p.read_audio(0.05)
    except AudioCaptureError as e:
        bus.publish(ErrorOccurred(error=e, component="capture", recoverable=True))
        app.transition(AppState.ERROR)
    ev = sub.get(0.5)
    assert isinstance(ev, ErrorOccurred) and ev.error.code == "DEVICE_LOST"
    assert app.state is AppState.ERROR
    p.cleanup()
    bus.close()
