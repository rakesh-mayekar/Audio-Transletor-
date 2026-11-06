"""Core data contracts shared by every layer. No OS-specific or UI imports allowed here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from core.errors import InvalidAudioError

SAMPLE_RATE = 16_000
CHANNELS = 1


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    return round(ms * sample_rate / 1000.0)


def samples_to_ms(samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    return samples * 1000.0 / sample_rate


@dataclass(frozen=True, slots=True, eq=False)
class AudioFrame:
    """Normalized audio: 16 kHz, mono, float32, stamped with the *capture* time.

    `pts_ms` is the capture-clock position of the first sample, never processing time.
    """

    pts_ms: float
    samples: NDArray[np.float32]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise InvalidAudioError(
                f"sample_rate must be {SAMPLE_RATE}, got {self.sample_rate}",
                hint="Resample in the provider or the shared Resampler before emitting frames.",
            )
        s = self.samples
        if not isinstance(s, np.ndarray) or s.dtype != np.float32 or s.ndim != 1:
            raise InvalidAudioError(
                "samples must be a 1-D float32 numpy array (mono)",
                hint="Convert channels to mono and dtype to float32.",
            )
        if s.size == 0:
            raise InvalidAudioError("empty audio frame")
        if self.pts_ms < 0:
            raise InvalidAudioError(f"negative timestamp {self.pts_ms}")
        view = s.view()
        view.flags.writeable = False
        object.__setattr__(self, "samples", view)

    @property
    def duration_ms(self) -> float:
        return samples_to_ms(self.samples.size, self.sample_rate)

    @property
    def end_ms(self) -> float:
        return self.pts_ms + self.duration_ms


class CaptionState(StrEnum):
    CAPTURED = "captured"
    PROCESSING = "processing"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    FINAL = "final"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    """Raw Whisper signals. These are diagnostics, NOT calibrated confidences."""

    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    language_probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Output contract of the translation engine (spec 7.1)."""

    source_language: str
    translated_text: str
    audio_start_ms: float
    audio_end_ms: float
    model: str
    target_language: str = "en"
    status: CaptionState = CaptionState.CANDIDATE
    diagnostics: ModelDiagnostics = field(default_factory=ModelDiagnostics)
    source_text: str | None = None  # optional; never fabricated

    def __post_init__(self) -> None:
        if self.audio_end_ms < self.audio_start_ms:
            raise InvalidAudioError("audio_end_ms precedes audio_start_ms")
