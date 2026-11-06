"""Typed, immutable events. Timestamps with `_ms` suffix are AUDIO-clock times."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.errors import TranslatorError
from core.types import AudioFrame, CaptionState, TranslationResult


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    emitted_at_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True, slots=True, kw_only=True)
class AppStateChanged(Event):
    old: str
    new: str
    reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class LanguageStateChanged(Event):
    old: str
    new: str
    language: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorOccurred(Event):
    error: TranslatorError
    component: str
    recoverable: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioCaptured(Event):
    frame: AudioFrame


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDropped(Event):
    """Audio was discarded; always recorded so loss is never silent."""

    start_ms: float
    end_ms: float
    reason: str
    component: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechDetected(Event):
    utterance_id: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class LanguageConfirmed(Event):
    language: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TranslationStarted(Event):
    utterance_id: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TranslationCompleted(Event):
    utterance_id: str
    caption_id: str
    result: TranslationResult


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptionValidated(Event):
    caption_id: str
    text: str
    start_ms: float
    end_ms: float
    state: CaptionState = CaptionState.VALIDATED


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptionRejected(Event):
    caption_id: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptionDisplayed(Event):
    caption_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptionExpired(Event):
    caption_id: str
