"""Base class for providers that read raw float32 PCM from a helper process's stdout.

Why a helper process: macOS (ScreenCaptureKit) and Linux (PulseAudio/PipeWire `parec`) expose
system-audio capture most robustly through a small native tool; isolating it also means a crash
in native audio code cannot take down the app. All shared behaviour - downmix, resampling to
16 kHz, sample-clock timestamps, bounded buffering with drop accounting, failure mapping,
orderly shutdown - lives here and is verified with a scripted helper on every OS. Only the
command line and device discovery are OS-specific.

Helper contract: write interleaved little-endian float32 at (helper_rate, helper_channels) to
stdout; diagnostics to stderr; exit 77 = permission denied, 69 = no such device/unavailable.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from abc import abstractmethod
from collections import deque

import numpy as np

from core.errors import (
    AudioCaptureError,
    DeviceLostError,
    PermissionDeniedError,
    ProviderUnavailableError,
)
from core.types import SAMPLE_RATE, AudioFrame, samples_to_ms
from oslayer.base import (
    AudioCaptureProvider,
    AudioDevice,
    CaptureState,
    CaptureStatus,
)
from oslayer.common.pcm import StreamingResampler, to_mono

_log = logging.getLogger(__name__)
EXIT_PERMISSION = 77
EXIT_UNAVAILABLE = 69
_CHUNK_MS = 20


class PcmProcessCaptureProvider(AudioCaptureProvider):
    helper_rate: int = 48_000
    helper_channels: int = 2
    permission_hint: str = "Grant the required system-audio permission and restart capture."

    def __init__(self, max_pending_frames: int = 256) -> None:
        self._q: queue.Queue[AudioFrame] = queue.Queue(maxsize=max_pending_frames)
        self._state = CaptureState.UNINITIALIZED
        self._device: AudioDevice | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._err_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._emitted = 0
        self._dropped = 0
        self._failure: AudioCaptureError | None = None

    # ----------------------------------------------------------- OS-specific hooks
    @abstractmethod
    def _discover_devices(self) -> list[AudioDevice]: ...

    @abstractmethod
    def _build_command(self, device: AudioDevice | None) -> list[str]: ...

    def _check_available(self) -> None:
        """Raise ProviderUnavailableError (with an install hint) if the helper is missing."""

    # ----------------------------------------------------------------- lifecycle
    def initialize(self) -> None:
        if self._state not in (CaptureState.UNINITIALIZED, CaptureState.STOPPED):
            return
        self._check_available()
        self._state = CaptureState.READY

    def list_devices(self) -> list[AudioDevice]:
        return self._discover_devices()

    def select_device(self, device_id: str | None) -> None:
        self._device = self.resolve_device(device_id)

    def start_capture(self) -> None:
        if self._state is not CaptureState.READY and self._state is not CaptureState.STOPPED:
            raise AudioCaptureError(
                f"Cannot start capture from state {self._state.value}",
                hint="Call initialize() first.",
            )
        cmd = self._build_command(self._device)
        self._stop.clear()
        self._failure = None
        self._emitted = self._dropped = 0
        self._stderr_tail.clear()
        while not self._q.empty():
            self._q.get_nowait()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as e:
            self._state = CaptureState.FAILED
            raise ProviderUnavailableError(
                f"Could not start audio helper: {e}", hint="Check the helper is installed."
            ) from e
        self._state = CaptureState.CAPTURING
        self._reader = threading.Thread(target=self._read_loop, name="capture-reader", daemon=True)
        self._err_thread = threading.Thread(
            target=self._drain_stderr, name="capture-stderr", daemon=True
        )
        self._reader.start()
        self._err_thread.start()

    def read_audio(self, timeout: float = 0.1) -> AudioFrame | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            if self._failure is not None:
                raise self._failure from None
            return None

    def stop_capture(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(3.0)
            for s in (proc.stdout, proc.stderr):
                if s is not None:
                    s.close()
        for t in (self._reader, self._err_thread):
            if t is not None:
                t.join(5.0)
        self._reader = self._err_thread = None
        if self._state in (CaptureState.CAPTURING, CaptureState.RECOVERING):
            self._state = CaptureState.STOPPED

    def get_status(self) -> CaptureStatus:
        return CaptureStatus(
            state=self._state,
            platform=self.platform_name,
            device=self._device,
            frames_emitted=self._emitted,
            frames_dropped=self._dropped,
            last_error=str(self._failure) if self._failure else None,
        )

    def cleanup(self) -> None:
        self.stop_capture()
        if self._state is not CaptureState.FAILED:
            self._state = (
                CaptureState.STOPPED
                if self._state is not CaptureState.UNINITIALIZED
                else self._state
            )

    # ---------------------------------------------------------------- internals
    def _push(self, frame: AudioFrame) -> None:
        while True:
            try:
                self._q.put_nowait(frame)
                self._emitted += 1
                return
            except queue.Full:
                try:
                    self._q.get_nowait()  # drop OLDEST; consumer is too slow
                    self._dropped += 1
                except queue.Empty:
                    pass

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, b""):
                self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        frame_bytes = 4 * self.helper_channels
        chunk = int(self.helper_rate * _CHUNK_MS / 1000) * frame_bytes
        resampler = StreamingResampler(self.helper_rate, SAMPLE_RATE)
        carry = b""
        out_samples = 0
        pts0: float | None = None
        try:
            while not self._stop.is_set():
                data = proc.stdout.read(chunk)
                if not data:
                    break
                if pts0 is None:
                    pts0 = time.monotonic() * 1000.0  # capture clock anchor (first byte)
                data = carry + data
                usable = len(data) - len(data) % frame_bytes
                carry = data[usable:]
                if not usable:
                    continue
                pcm = np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)
                mono = to_mono(pcm, self.helper_channels)
                out = np.nan_to_num(resampler.process(mono), copy=False)
                np.clip(out, -1.0, 1.0, out=out)
                if out.size:
                    self._push(AudioFrame(pts0 + samples_to_ms(out_samples), out))
                    out_samples += out.size
        except (OSError, ValueError):
            if not self._stop.is_set():
                _log.exception("capture reader failed")
        finally:
            if not self._stop.is_set():
                self._on_helper_exit(proc)

    def _on_helper_exit(self, proc: subprocess.Popen[bytes]) -> None:
        try:
            code = proc.wait(2.0)
        except subprocess.TimeoutExpired:
            code = None
        tail = " | ".join(self._stderr_tail)
        with self._lock:
            if code == EXIT_PERMISSION:
                self._failure = PermissionDeniedError(
                    "System-audio capture permission was denied", hint=self.permission_hint
                )
            elif code == EXIT_UNAVAILABLE:
                self._failure = DeviceLostError(
                    "Audio output device is unavailable",
                    hint="Check the output device exists, then select it again.",
                )
            else:
                self._failure = AudioCaptureError(
                    f"Audio helper stopped unexpectedly (exit={code}). {tail}".strip(),
                    hint="See the log for the helper's stderr output.",
                )
            self._state = CaptureState.FAILED


__all__ = ["EXIT_PERMISSION", "EXIT_UNAVAILABLE", "PcmProcessCaptureProvider"]
