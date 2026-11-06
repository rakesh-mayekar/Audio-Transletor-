"""Reusable contract tests for ANY AudioCaptureProvider (fake today; Windows/macOS/Linux later).

Usage: subclass ProviderConformance in a test module and implement `make_provider`.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.errors import AudioCaptureError, DeviceNotFoundError
from core.types import SAMPLE_RATE, AudioFrame
from oslayer.base import AudioCaptureProvider, CaptureState, DeviceKind


class ProviderConformance:
    def make_provider(self) -> AudioCaptureProvider:  # pragma: no cover - overridden
        raise NotImplementedError

    @pytest.fixture
    def provider(self):
        p = self.make_provider()
        p.initialize()
        yield p
        p.cleanup()

    def _read_n(self, p: AudioCaptureProvider, n: int, budget_s: float = 5.0) -> list[AudioFrame]:
        out: list[AudioFrame] = []
        end = time.monotonic() + budget_s
        while len(out) < n and time.monotonic() < end:
            f = p.read_audio(0.1)
            if f is not None:
                out.append(f)
        return out

    def test_lifecycle_states(self, provider):
        assert provider.get_status().state is CaptureState.READY
        provider.start_capture()
        assert provider.get_status().state is CaptureState.CAPTURING
        provider.stop_capture()
        assert provider.get_status().state is CaptureState.STOPPED

    def test_lists_no_microphones(self, provider):
        devices = provider.list_devices()
        assert all(d.kind is DeviceKind.SYSTEM_OUTPUT_LOOPBACK for d in devices)

    def test_unknown_device_rejected(self, provider):
        with pytest.raises(DeviceNotFoundError):
            provider.select_device("does-not-exist")

    def test_frames_are_normalized(self, provider):
        provider.start_capture()
        frames = self._read_n(provider, 5)
        assert len(frames) == 5
        for f in frames:
            assert f.sample_rate == SAMPLE_RATE
            assert f.samples.dtype == np.float32 and f.samples.ndim == 1

    def test_timestamps_monotonic_and_contiguous(self, provider):
        provider.start_capture()
        frames = self._read_n(provider, 10)
        for a, b in zip(frames, frames[1:], strict=False):
            assert b.pts_ms >= a.pts_ms
            assert (
                b.pts_ms == pytest.approx(a.end_ms, abs=1e-6) or b.pts_ms > a.end_ms
            )  # gap allowed, overlap not

    def test_read_without_data_times_out_with_none(self, provider):
        t0 = time.monotonic()
        assert provider.read_audio(0.05) is None
        assert time.monotonic() - t0 < 1.0

    def test_cleanup_idempotent_and_releases(self, provider):
        provider.start_capture()
        provider.cleanup()
        provider.cleanup()
        assert provider.get_status().state is not CaptureState.CAPTURING

    def test_start_before_initialize_fails(self):
        p = self.make_provider()
        with pytest.raises(AudioCaptureError):
            p.start_capture()
        p.cleanup()
