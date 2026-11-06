"""Importable (spawn-safe) fake engines for inference-worker process tests."""

from __future__ import annotations

import time

import numpy as np

from ai.engine import EngineOutput, LanguageDetection, SpeechTranslationEngine
from core.errors import InferenceError, ModelError
from core.types import TranslationResult


class FakeEngine(SpeechTranslationEngine):
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self._loaded = True

    def load_model(self, configuration) -> None:
        self._loaded = True

    def translate_audio(self, audio, language, audio_start_ms=0.0, verify_language=False):
        time.sleep(self.delay_s)
        if language == "boom":
            raise InferenceError("kaput")
        ms = audio.size / 16.0
        r = TranslationResult(
            language or "ja", "hello", audio_start_ms, audio_start_ms + ms, "fake"
        )
        return EngineOutput([r], language or "ja", None, self.delay_s * 1000, ms)

    def detect_language(self, audio):
        return LanguageDetection("ja", 0.9, True)

    def unload_model(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create(delay_s: float = 0.0) -> FakeEngine:
    return FakeEngine(delay_s)


def create_failing() -> FakeEngine:
    raise ModelError("cannot load", code="MODEL_MISSING") if False else ModelError("cannot load")


_ = np
