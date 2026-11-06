import sys
import time
from pathlib import Path

import numpy as np
import pytest

from core.errors import AudioCaptureError, DeviceLostError, PermissionDeniedError
from oslayer.base import AudioDevice, CaptureState
from oslayer.common.process_capture import PcmProcessCaptureProvider
from tests.platform.conformance import ProviderConformance

HELPER = str(Path(__file__).with_name("pcm_helper.py"))


class ScriptedProvider(PcmProcessCaptureProvider):
    platform_name = "fake"

    def __init__(self, *extra: str, rate: int = 48000, channels: int = 2, **kw):
        super().__init__(**kw)
        self.helper_rate, self.helper_channels = rate, channels
        self._extra = [f"--rate={rate}", f"--channels={channels}", *extra]

    def _discover_devices(self):
        return [AudioDevice("default", "Default output (loopback)", is_default=True)]

    def _build_command(self, device):
        return [sys.executable, "-I", HELPER, *self._extra]


class TestProcessProviderConformance(ProviderConformance):
    def make_provider(self):
        return ScriptedProvider()


@pytest.mark.parametrize(("rate", "channels"), [(48000, 2), (44100, 2), (16000, 1), (96000, 6)])
def test_any_helper_format_is_normalized_to_16k_mono(rate, channels):
    p = ScriptedProvider(rate=rate, channels=channels)
    p.initialize()
    p.start_capture()
    try:
        got, end = [], time.monotonic() + 8
        while sum(f.samples.size for f in got) < 16000 and time.monotonic() < end:
            f = p.read_audio(0.2)
            if f is not None:
                got.append(f)
        x = np.concatenate([f.samples for f in got])
        assert x.size >= 16000
        body = x[2000:16000]
        assert abs(float(np.sqrt(np.mean(body**2))) - 0.5 / np.sqrt(2)) < 0.02  # 1 kHz survived
    finally:
        p.cleanup()


def test_timestamps_follow_sample_clock_and_capture_clock():
    p = ScriptedProvider()
    p.initialize()
    t0 = time.monotonic() * 1000
    p.start_capture()
    frames = [f for f in (p.read_audio(0.5) for _ in range(20)) if f]
    p.cleanup()
    assert frames[0].pts_ms >= t0  # capture clock, not zero-based
    for a, b in zip(frames, frames[1:], strict=False):
        assert b.pts_ms == pytest.approx(a.end_ms, abs=1e-6)


def test_permission_denied_maps_to_typed_error_with_hint():
    p = ScriptedProvider("--exit-code=77")
    p.initialize()
    p.start_capture()
    with pytest.raises(PermissionDeniedError) as ei:
        for _ in range(60):
            p.read_audio(0.1)
    assert ei.value.hint
    assert p.get_status().state is CaptureState.FAILED
    p.cleanup()


def test_unavailable_device_maps_to_device_lost():
    p = ScriptedProvider("--exit-code=69")
    p.initialize()
    p.start_capture()
    with pytest.raises(DeviceLostError):
        for _ in range(60):
            p.read_audio(0.1)
    p.cleanup()


def test_helper_crash_midstream_surfaces_after_buffered_audio_drained():
    p = ScriptedProvider("--die-after=0.3", "--exit-code=3")
    p.initialize()
    p.start_capture()
    n = 0
    with pytest.raises(AudioCaptureError, match="exit=3"):
        for _ in range(200):
            if p.read_audio(0.1) is not None:
                n += 1
    assert n > 5  # audio produced before the crash was still delivered
    p.cleanup()


def test_slow_consumer_bounded_with_drop_accounting():
    p = ScriptedProvider("--seconds=5", "--no-sleep", max_pending_frames=8)
    p.initialize()
    p.start_capture()
    time.sleep(0.8)  # consumer idle while helper floods
    st = p.get_status()
    assert p._q.qsize() <= 8 and st.frames_dropped > 0
    p.cleanup()


def test_cleanup_kills_helper_and_threads():
    import threading

    p = ScriptedProvider()
    p.initialize()
    p.start_capture()
    proc = p._proc
    p.cleanup()
    assert proc.poll() is not None
    assert not [t for t in threading.enumerate() if t.name.startswith("capture-")]
