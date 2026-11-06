"""TranslatorRuntime: owns the worker process, VAD and pipeline; drives them from one loop."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai.inference_worker import InferenceWorker
from ai.vad import SileroStreamingVad
from app.pipeline import TranslationPipeline
from core.config.models import AppConfig
from core.errors import AudioCaptureError, TranslatorError
from core.events import EventBus
from core.events.types import ErrorOccurred
from core.state import ApplicationStateMachine, AppState
from core.types import SAMPLE_RATE, AudioFrame
from oslayer.base import AudioCaptureProvider

_log = logging.getLogger(__name__)
FRAME = SAMPLE_RATE // 10  # 100 ms


class TranslatorRuntime:
    def __init__(self, config: AppConfig, models_dir: Path, bus: EventBus) -> None:
        self.cfg, self.models_dir, self.bus = config, models_dir, bus
        self.state = ApplicationStateMachine()
        self.worker: InferenceWorker | None = None
        self.pipeline: TranslationPipeline | None = None

    def start(self, load_timeout: float = 180.0) -> None:
        self.state.transition(AppState.STARTING)
        try:
            worker = InferenceWorker(
                "ai.whisper_engine:create_engine",
                {"models_dir": str(self.models_dir), "model_config": self.cfg.model},
                max_pending=4,
            )
            worker.start(timeout=load_timeout)  # raises ModelError (typed) if missing/corrupt
            self.worker = worker
            self.pipeline = TranslationPipeline(self.cfg, SileroStreamingVad(), worker, self.bus)
        except BaseException:
            self.state.transition(AppState.ERROR)
            raise
        self.state.transition(AppState.RUNNING)

    def stop(self) -> None:
        if self.state.state in (AppState.IDLE,):
            return
        if self.state.can_transition(AppState.STOPPING):
            self.state.transition(AppState.STOPPING)
        try:
            if self.pipeline is not None:
                self.pipeline.finish()
                self.pipeline.drain(timeout=30)
        finally:
            if self.worker is not None:
                self.worker.stop()
            self.worker = self.pipeline = None
            if self.state.can_transition(AppState.IDLE):
                self.state.transition(AppState.IDLE)

    # ------------------------------------------------------------------ sources
    def run_samples(
        self, samples: NDArray[np.float32], speed: float = 1.0, stop: threading.Event | None = None
    ) -> None:
        """Replay a recording through the pipeline, paced at `speed` x real time."""
        assert self.pipeline is not None
        t0 = time.monotonic()
        for i in range(0, samples.size, FRAME):
            if stop is not None and stop.is_set():
                break
            self.pipeline.feed(AudioFrame(i * 1000.0 / SAMPLE_RATE, samples[i : i + FRAME].copy()))
            while self.pipeline.poll():
                pass
            lag = (i + FRAME) / SAMPLE_RATE / speed - (time.monotonic() - t0)
            if lag > 0:
                time.sleep(lag)
        self.pipeline.finish()
        self.pipeline.drain(timeout=120)

    def run_provider(self, provider: AudioCaptureProvider, stop: threading.Event) -> None:
        """Live capture loop. Capture errors are published and end the loop (no silent retry,
        no microphone fallback)."""
        assert self.pipeline is not None
        provider.initialize()
        provider.start_capture()
        try:
            while not stop.is_set():
                frame = provider.read_audio(0.05)
                if frame is not None:
                    self.pipeline.feed(frame)
                while self.pipeline.poll():
                    pass
        except AudioCaptureError as e:
            self.bus.publish(ErrorOccurred(error=e, component="capture", recoverable=False))
            if self.state.can_transition(AppState.ERROR):
                self.state.transition(AppState.ERROR)
            raise
        except TranslatorError as e:
            self.bus.publish(ErrorOccurred(error=e, component="runtime", recoverable=False))
            raise
        finally:
            provider.cleanup()
            self.pipeline.finish()
            self.pipeline.drain(timeout=15)
