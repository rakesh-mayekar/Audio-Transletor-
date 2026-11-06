"""Engine-independent interface. The pipeline depends on this, never on faster-whisper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from core.config.models import ModelConfig
from core.types import TranslationResult


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    language: str  # Whisper's top guess over ALL languages (may be unsupported)
    probability: float
    supported: bool
    top: tuple[tuple[str, float], ...] = ()  # top few (code, prob) for diagnostics
    supported_mass: float = 0.0  # total probability on the 5 supported languages


@dataclass(frozen=True, slots=True)
class EngineOutput:
    results: list[TranslationResult]
    language: str
    language_probability: float | None
    inference_ms: float
    audio_ms: float
    extra: dict[str, float] = field(default_factory=dict)
    # Result of a real detection pass, if one ran (auto-detect or verification of a pinned language)
    detected_language: str | None = None
    detected_probability: float | None = None

    @property
    def rtf(self) -> float:
        return self.inference_ms / self.audio_ms if self.audio_ms else float("inf")


class SpeechTranslationEngine(ABC):
    @abstractmethod
    def load_model(self, configuration: ModelConfig) -> None: ...

    @abstractmethod
    def translate_audio(
        self,
        audio: NDArray[np.float32],
        language: str | None,
        audio_start_ms: float = 0.0,
        verify_language: bool = False,
    ) -> EngineOutput:
        """Translate 16 kHz mono float32 speech to English. `language=None` runs detection
        first and raises UnsupportedLanguageError for languages outside the supported set.
        `verify_language=True` with a given language also runs detection: a confident unsupported
        result raises; a confident different supported language is used for this utterance."""

    @abstractmethod
    def detect_language(self, audio: NDArray[np.float32]) -> LanguageDetection: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
