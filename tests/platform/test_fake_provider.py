import time

import numpy as np
import pytest

from core.errors import DeviceLostError
from oslayer.base import AudioCaptureProvider, CaptureState
from oslayer.fake import FakeAudioCaptureProvider
from tests.platform.conformance import ProviderConformance


class TestFakeConformance(ProviderConformance):
    def make_provider(self) -> AudioCaptureProvider:
        return FakeAudioCaptureProvider(
            np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
        )


def test_bounded_buffer_drops_oldest_and_counts():
    p = FakeAudioCaptureProvider(max_buffered_frames=4)
    p.initialize()
    p.start_capture()
    time.sleep(0.3)  # consumer idle -> producer must not grow memory
    st = p.get_status()
    assert st.frames_dropped > 0
    assert len(p._buf) <= 4
    first = p.read_audio(0.1)
    assert (
        first is not None and first.pts_ms > 0
    )  # oldest frames were dropped, timestamps preserved
    p.cleanup()


def test_device_loss_raises_typed_error():
    p = FakeAudioCaptureProvider()
    p.initialize()
    p.start_capture()
    p.simulate_device_loss()
    with pytest.raises(DeviceLostError):
        for _ in range(100):
            p.read_audio(0.05)
    assert p.get_status().state is CaptureState.FAILED
    p.cleanup()


def test_context_manager_cleans_up():
    with FakeAudioCaptureProvider() as p:
        p.start_capture()
    assert p.get_status().state is CaptureState.STOPPED
    assert p._thread is None
