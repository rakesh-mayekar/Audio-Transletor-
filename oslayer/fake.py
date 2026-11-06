"""FakeAudioCaptureProvider — a TEST HELPER, not a product feature.

It replays supplied samples through the same contract real providers implement, so the shared
core and the conformance suite can run on any OS. It is never selected by `create_audio_provider`
unless explicitly registered, and it does not count as a platform implementation.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from core.errors import AudioCaptureError, DeviceLostError
from core.types import SAMPLE_RATE, AudioFrame, ms_to_samples
from oslayer.base import (
    AudioCaptureProvider,
    AudioDevice,
    CaptureState,
    CaptureStatus,
    DeviceKind,
)


class FakeAudioCaptureProvider(AudioCaptureProvider):
    platform_name = "fake"

    def __init__(
        self,
        samples: NDArray[np.float32] | None = None,
        frame_ms: float = 20.0,
        max_buffered_frames: int = 64,
        realtime: bool = False,
    ) -> None:
        self._source = samples if samples is not None else np.zeros(SAMPLE_RATE, np.float32)
        self._frame_samples = ms_to_samples(frame_ms)
        self._max = max_buffered_frames
        self._realtime = realtime
        self._frame_ms = frame_ms
        self._devices = [
            AudioDevice("fake-speakers", "Fake Speakers (loopback)", is_default=True),
            AudioDevice("fake-headphones", "Fake Headphones (loopback)"),
            AudioDevice("fake-mic", "Fake Microphone", kind=DeviceKind.MICROPHONE),
        ]
        self._state = CaptureState.UNINITIALIZED
        self._device: AudioDevice | None = None
        self._buf: deque[AudioFrame] = deque()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._emitted = 0
        self._dropped = 0
        self._error: str | None = None
        self._fail_with: Exception | None = None

    # -- test hooks ----------------------------------------------------------
    def simulate_device_loss(self) -> None:
        self._fail_with = DeviceLostError(
            "Fake device unplugged", hint="Reconnect the output device."
        )

    # -- contract ------------------------------------------------------------
    def initialize(self) -> None:
        if self._state is CaptureState.UNINITIALIZED or self._state is CaptureState.STOPPED:
            self._state = CaptureState.READY

    def list_devices(self) -> list[AudioDevice]:
        # microphones are NOT listed (system-audio only)
        return [d for d in self._devices if d.kind is DeviceKind.SYSTEM_OUTPUT_LOOPBACK]

    def select_device(self, device_id: str | None) -> None:
        self._device = self.resolve_device(device_id)

    def start_capture(self) -> None:
        if self._state is CaptureState.UNINITIALIZED:
            raise AudioCaptureError("initialize() must be called first")
        if self._state is CaptureState.CAPTURING:
            return
        if self._device is None:
            self._device = self.resolve_device(None)
        self._stop.clear()
        self._state = CaptureState.CAPTURING
        self._thread = threading.Thread(target=self._produce, name="fake-capture", daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        pos = 0
        n = self._frame_samples
        t0 = time.monotonic()
        while not self._stop.is_set():
            if self._fail_with is not None:
                self._error = str(self._fail_with)
                self._state = CaptureState.FAILED
                with self._cond:
                    self._cond.notify_all()
                return
            chunk = self._source[pos : pos + n]
            if chunk.size < n:  # loop the source, keep timestamps monotonic
                pos = 0
                chunk = self._source[:n]
                if chunk.size < n:
                    chunk = np.pad(chunk, (0, n - chunk.size))
            pts = self._emitted_total * self._frame_ms
            frame = AudioFrame(pts_ms=pts, samples=np.ascontiguousarray(chunk, dtype=np.float32))
            pos += n
            with self._cond:
                if len(self._buf) >= self._max:
                    self._buf.popleft()  # bounded: drop oldest, count it
                    self._dropped += 1
                self._buf.append(frame)
                self._emitted += 1
                self._cond.notify_all()
            if self._realtime:
                target = t0 + (self._emitted_total + 1) * self._frame_ms / 1000.0
                time.sleep(max(0.0, target - time.monotonic()))

    @property
    def _emitted_total(self) -> int:
        return self._emitted + self._dropped

    def read_audio(self, timeout: float = 0.1) -> AudioFrame | None:
        with self._cond:
            if not self._buf and self._state is CaptureState.CAPTURING:
                self._cond.wait(timeout)
            if self._buf:
                return self._buf.popleft()
        if self._state is CaptureState.FAILED and self._fail_with is not None:
            raise self._fail_with
        return None

    def stop_capture(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(2.0)
            self._thread = None
        if self._state is CaptureState.CAPTURING:
            self._state = CaptureState.STOPPED
        with self._cond:
            self._cond.notify_all()

    def get_status(self) -> CaptureStatus:
        return CaptureStatus(
            self._state, self.platform_name, self._device, self._emitted, self._dropped, self._error
        )

    def cleanup(self) -> None:
        self.stop_capture()
        with self._cond:
            self._buf.clear()
        self._state = CaptureState.STOPPED
