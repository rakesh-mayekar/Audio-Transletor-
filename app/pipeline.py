"""TranslationPipeline: frames -> segmenter -> inference -> validation -> events.

Single-threaded by design (feed/poll are called from one loop) so ordering is deterministic and
testable; the inference itself runs in the worker process. Every discarded piece of audio or
text produces an event (AudioDropped / CaptionRejected) - nothing disappears silently.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ai.engine import LanguageDetection
from ai.inference_worker import WorkerRequest, WorkerResult
from ai.language_detector import LanguageDetector
from core.config.models import AppConfig
from core.events import EventBus
from core.events.types import (
    AudioDropped,
    CaptionRejected,
    CaptionValidated,
    ErrorOccurred,
    LanguageConfirmed,
    SpeechDetected,
    TranslationCompleted,
    TranslationStarted,
)
from core.pipeline.segmenter import SpeechSegmenter, Utterance, VadProtocol
from core.state import CaptionLifecycle, LanguageState, LanguageStateMachine
from core.types import AudioFrame, CaptionState
from core.validation import CaptionValidator

_log = logging.getLogger(__name__)


class InferencePort(Protocol):
    def submit(self, req: WorkerRequest) -> bool: ...
    def poll(self, timeout: float = 0.1) -> WorkerResult | None: ...


@dataclass(slots=True)
class PipelineStats:
    utterances: int = 0
    captions_valid: int = 0
    captions_rejected: int = 0
    dropped_backlog: int = 0
    dropped_stale: int = 0
    dropped_unsupported: int = 0
    errors: int = 0


class TranslationPipeline:
    def __init__(
        self,
        config: AppConfig,
        vad: VadProtocol,
        worker: InferencePort,
        bus: EventBus,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = config
        self.bus, self.worker, self._clock = bus, worker, clock
        self.segmenter = SpeechSegmenter(vad, config.vad)
        self.validator = CaptionValidator(config.validation)
        from core.config.languages import SUPPORTED_CODES

        self.language_machine = LanguageStateMachine(frozenset(SUPPORTED_CODES))
        self.detector = LanguageDetector(
            self.language_machine,
            config.language,
            min_probability=config.validation.min_language_probability,
        )
        if not config.language.auto_detect and config.language.manual_source:
            self.language_machine.set_manual(config.language.manual_source)
        self.stats = PipelineStats()
        self._pending: dict[str, Utterance] = {}
        self._seq = 0
        self._last_audio_end_ms = 0.0
        self._announced: str | None = None
        self._since_confirm = 0

    # ----------------------------------------------------------------- input
    @property
    def pending(self) -> int:
        return len(self._pending)

    def feed(self, frame: AudioFrame) -> None:
        self._last_audio_end_ms = frame.end_ms
        for u in self.segmenter.feed(frame):
            self._dispatch(u)

    def finish(self) -> None:
        for u in self.segmenter.flush():
            self._dispatch(u)

    def set_manual_language(self, code: str | None) -> None:
        """None returns to auto-detect."""
        self.finish()
        if code is None:
            if self.language_machine.state is LanguageState.MANUAL_OVERRIDE:
                self.language_machine.clear_manual()
        else:
            self.language_machine.set_manual(code)

    def _dispatch(self, u: Utterance) -> None:
        self.stats.utterances += 1
        self.bus.publish(
            SpeechDetected(utterance_id=u.utterance_id, start_ms=u.start_ms, end_ms=u.end_ms)
        )
        deadline = self._clock() + self.cfg.validation.max_staleness_ms / 1000.0
        pinned = self.detector.active_language
        verify = False
        iv = self.cfg.language.verify_interval
        if pinned is not None and self.language_machine.state is LanguageState.CONFIRMED and iv:
            self._since_confirm += 1
            verify = self._since_confirm % iv == 0
        else:
            self._since_confirm = 0
        req = WorkerRequest(
            request_id=u.utterance_id,
            audio=u.samples,
            language=pinned,
            audio_start_ms=u.start_ms,
            deadline_epoch_s=deadline,
            verify_language=verify,  # auto-pinned language is re-checked; manual is trusted
        )
        if not self.worker.submit(req):
            self.stats.dropped_backlog += 1
            self.bus.publish(
                AudioDropped(
                    start_ms=u.start_ms,
                    end_ms=u.end_ms,
                    reason="inference_backlog",
                    component="pipeline",
                )
            )
            return
        self._pending[u.utterance_id] = u
        self.bus.publish(
            TranslationStarted(utterance_id=u.utterance_id, start_ms=u.start_ms, end_ms=u.end_ms)
        )

    # ---------------------------------------------------------------- output
    def poll(self, timeout: float = 0.0) -> bool:
        res = self.worker.poll(timeout)
        if res is None:
            return False
        u = self._pending.pop(res.request_id, None)
        if u is None:
            return True
        self._handle(u, res)
        return True

    def drain(self, timeout: float = 60.0) -> None:
        end = time.monotonic() + timeout
        while self._pending and time.monotonic() < end:
            self.poll(0.1)

    def _handle(self, u: Utterance, res: WorkerResult) -> None:
        if res.stale:
            self.stats.dropped_stale += 1
            self.bus.publish(
                AudioDropped(
                    start_ms=u.start_ms, end_ms=u.end_ms, reason="stale", component="worker"
                )
            )
            return
        if res.error is not None:
            self._handle_error(u, res.error)
            return
        assert res.output is not None
        out = res.output
        if out.detected_language is not None and out.detected_probability is not None:
            before = self.detector.active_language
            self.detector.observe(
                LanguageDetection(out.detected_language, out.detected_probability, True),
                u.speech_ms / 1000,
            )
            self._announce(before)
        for r in out.results:
            self._seq += 1
            cid = f"cap-{self._seq:06d}"
            life = CaptionLifecycle(cid)
            life.transition(CaptionState.PROCESSING)
            life.transition(CaptionState.CANDIDATE)
            self.bus.publish(
                TranslationCompleted(utterance_id=u.utterance_id, caption_id=cid, result=r)
            )
            v = self.validator.validate(
                r, now_audio_ms=self._last_audio_end_ms, seam=u.overlap_ms > 0
            )
            if v.accepted:
                life.transition(CaptionState.VALIDATED)
                self.stats.captions_valid += 1
                self.bus.publish(
                    CaptionValidated(
                        caption_id=cid,
                        text=v.text,
                        start_ms=r.audio_start_ms,
                        end_ms=r.audio_end_ms,
                    )
                )
            else:
                life.transition(CaptionState.REJECTED)
                self.stats.captions_rejected += 1
                self.bus.publish(CaptionRejected(caption_id=cid, reason=",".join(v.reasons)))

    def _handle_error(self, u: Utterance, err: dict[str, object]) -> None:
        code = str(err.get("code"))
        if code == "LANGUAGE_UNSUPPORTED":
            self.stats.dropped_unsupported += 1
            d = err.get("details") or {}
            assert isinstance(d, dict)
            lang, prob = d.get("detected_language"), d.get("probability")
            if isinstance(lang, str) and isinstance(prob, float):
                self.detector.observe(LanguageDetection(lang, prob, False), u.speech_ms / 1000)
            self.bus.publish(
                AudioDropped(
                    start_ms=u.start_ms,
                    end_ms=u.end_ms,
                    reason="unsupported_language",
                    component="engine",
                )
            )
            return
        self.stats.errors += 1
        from core.errors import InferenceError

        self.bus.publish(
            ErrorOccurred(
                error=InferenceError(str(err.get("message")), hint=str(err.get("hint") or "")),
                component="engine",
            )
        )

    def _announce(self, before: str | None) -> None:
        now = self.detector.active_language
        confirmed = self.detector.machine.state is LanguageState.CONFIRMED
        if now and now != before and now != self._announced and confirmed:
            self._announced = now
            self.bus.publish(LanguageConfirmed(language=now))
