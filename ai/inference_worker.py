"""Persistent inference worker in a SEPARATE PROCESS.

Why a process: isolates the GIL and native-library crashes from the UI/audio threads, and gives
a hard kill on shutdown. One worker = one model instance. Queues are bounded; when the worker
is saturated `submit()` returns False so the caller can record AudioDropped (never silent).
Requests carry a wall-clock deadline: expired ones are answered "stale" without running the model.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ai.engine import EngineOutput, SpeechTranslationEngine
from core.errors import InferenceError, ModelError, TranslatorError

_log = logging.getLogger(__name__)
_CTX = mp.get_context("spawn")  # identical behaviour on Windows/macOS/Linux


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request_id: str
    audio: NDArray[np.float32]
    language: str | None
    audio_start_ms: float
    deadline_epoch_s: float  # results later than this are useless -> skipped
    verify_language: bool = False


@dataclass(frozen=True, slots=True)
class WorkerResult:
    request_id: str
    output: EngineOutput | None = None
    error: dict[str, Any] | None = None
    stale: bool = False
    queue_wait_ms: float = 0.0


def _worker_main(factory: str, kwargs: dict[str, Any], req_q: Any, res_q: Any) -> None:
    try:
        mod, _, fn = factory.partition(":")
        engine: SpeechTranslationEngine = getattr(importlib.import_module(mod), fn)(**kwargs)
        res_q.put(("ready", None))
    except BaseException as e:  # report load failure to the parent, then exit
        res_q.put(
            (
                "load_error",
                {
                    "message": str(e),
                    "trace": traceback.format_exc(),
                    "code": getattr(e, "code", "MODEL_ERROR"),
                },
            )
        )
        return
    while True:
        req = req_q.get()
        if req is None:
            break
        now = time.time()
        if now > req.deadline_epoch_s:
            res_q.put(("result", WorkerResult(req.request_id, stale=True)))
            continue
        try:
            out = engine.translate_audio(
                req.audio, req.language, req.audio_start_ms, verify_language=req.verify_language
            )
            res_q.put(("result", WorkerResult(req.request_id, output=out)))
        except TranslatorError as e:
            res_q.put(("result", WorkerResult(req.request_id, error=e.to_dict())))
        except Exception as e:
            res_q.put(
                (
                    "result",
                    WorkerResult(
                        req.request_id,
                        error={
                            "code": "INFERENCE_ERROR",
                            "message": str(e),
                            "hint": None,
                            "details": {},
                        },
                    ),
                )
            )
    engine.unload_model()


class InferenceWorker:
    def __init__(self, factory: str, factory_kwargs: dict[str, Any], max_pending: int = 4) -> None:
        self._factory, self._kwargs = factory, factory_kwargs
        self._req_q: Any = _CTX.Queue(maxsize=max_pending)
        self._res_q: Any = _CTX.Queue()
        self._proc: Any = None
        self.submitted = 0
        self.rejected_full = 0

    def start(self, timeout: float = 120.0) -> None:
        if self._proc is not None:
            return
        self._proc = _CTX.Process(
            target=_worker_main,
            args=(self._factory, self._kwargs, self._req_q, self._res_q),
            name="inference-worker",
            daemon=True,
        )
        self._proc.start()
        try:
            kind, payload = self._res_q.get(timeout=timeout)
        except queue.Empty:
            self.stop(force=True)
            raise ModelError("Inference worker did not become ready in time") from None
        if kind == "load_error":
            self.stop(force=True)
            raise ModelError(
                f"Worker failed to load model: {payload['message']}",
                hint=None,
                code=payload["code"],
            )

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def submit(self, req: WorkerRequest) -> bool:
        if not self.alive:
            raise InferenceError(
                "Inference worker is not running",
                hint="Restart translation; check logs for a crash.",
            )
        try:
            self._req_q.put_nowait(req)
        except queue.Full:
            self.rejected_full += 1
            return False
        self.submitted += 1
        return True

    def poll(self, timeout: float = 0.1) -> WorkerResult | None:
        try:
            kind, payload = self._res_q.get(timeout=timeout)
        except queue.Empty:
            return None
        return payload if kind == "result" else None

    def stop(self, force: bool = False, timeout: float = 10.0) -> None:
        p = self._proc
        if p is None:
            return
        try:
            if not force and p.is_alive():
                try:
                    self._req_q.put(None, timeout=1.0)
                except queue.Full:
                    force = True
                p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join(3.0)
            if p.is_alive():
                p.kill()
                p.join(3.0)
        finally:
            self._proc = None
            for q in (self._req_q, self._res_q):
                q.close()
                q.cancel_join_thread()
