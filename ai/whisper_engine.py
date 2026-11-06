"""Faster-Whisper (CTranslate2) engine. Offline: loads only from a verified local directory.

* task="translate" always (Whisper's own multilingual -> English translation). No separate MT
  stage and no LLM post-processing.
* The segment generator returned by faster-whisper is lazy; it is fully consumed here, inside
  the inference context.
* Whisper's internal no-speech / logprob / compression-ratio *skipping* is disabled so the raw
  segments and their diagnostics reach OUR validator (Phase 7). Nothing is invented or dropped here.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ai.engine import EngineOutput, LanguageDetection, SpeechTranslationEngine
from ai.model_manager import ModelManager
from core.config.languages import SUPPORTED_CODES
from core.config.models import ModelConfig
from core.errors import (
    InferenceError,
    InvalidAudioError,
    ModelError,
    UnsupportedLanguageError,
)
from core.types import SAMPLE_RATE, ModelDiagnostics, TranslationResult

_log = logging.getLogger(__name__)
MIN_AUDIO_MS = 200.0


def default_cpu_threads() -> int:
    """Conservative placeholder (half the logical CPUs) until Phase 10 benchmarks threads."""
    return max(1, (os.cpu_count() or 2) // 2)


class WhisperEngine(SpeechTranslationEngine):
    def __init__(self, manager: ModelManager) -> None:
        self._manager = manager
        self._model: Any = None
        self._cfg: ModelConfig | None = None
        self._threads = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_label(self) -> str:
        return f"whisper-{self._cfg.name}-{self._cfg.compute_type}" if self._cfg else "unloaded"

    def load_model(self, configuration: ModelConfig) -> None:
        if self._model is not None:
            return  # one instance only; unload explicitly to change
        path = self._manager.resolve(configuration.name)  # verifies integrity; never downloads
        # Hard offline guard: any accidental hub call fails instead of reaching the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from faster_whisper import WhisperModel

        threads = configuration.cpu_threads or default_cpu_threads()
        t0 = time.perf_counter()
        try:
            self._model = WhisperModel(
                str(path),
                device=configuration.device,
                compute_type=configuration.compute_type,
                cpu_threads=threads,
                num_workers=configuration.inference_workers,
            )
        except Exception as e:
            raise ModelError(
                f"Failed to load model '{configuration.name}': {e}",
                hint="Check compute_type/device support on this machine.",
            ) from e
        self._cfg, self._threads = configuration, threads
        _log.info(
            "model %s loaded in %.2fs (threads=%d)",
            self.model_label,
            time.perf_counter() - t0,
            threads,
        )

    def unload_model(self) -> None:
        self._model = None
        self._cfg = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate_audio(audio: NDArray[np.float32]) -> None:
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
            raise InvalidAudioError("audio must be a 1-D float32 array at 16 kHz")
        if audio.size * 1000.0 / SAMPLE_RATE < MIN_AUDIO_MS:
            raise InvalidAudioError(
                f"audio shorter than {MIN_AUDIO_MS:.0f} ms",
                hint="Language detection and translation need a complete utterance.",
            )

    def _require_model(self) -> Any:
        if self._model is None or self._cfg is None:
            raise ModelError("Model not loaded", hint="Call load_model() first.")
        return self._model

    # --------------------------------------------------------------- detection
    def detect_language(self, audio: NDArray[np.float32]) -> LanguageDetection:
        model = self._require_model()
        self._validate_audio(audio)
        try:
            lang, prob, all_probs = model.detect_language(
                audio=audio, language_detection_segments=1
            )
        except Exception as e:
            raise InferenceError(f"Language detection failed: {e}") from e
        top = tuple((str(c), float(p)) for c, p in sorted(all_probs, key=lambda x: -x[1])[:5])
        mass = float(sum(p for c, p in all_probs if c in SUPPORTED_CODES))
        return LanguageDetection(str(lang), float(prob), str(lang) in SUPPORTED_CODES, top, mass)

    # ------------------------------------------------------------- translation
    def translate_audio(
        self,
        audio: NDArray[np.float32],
        language: str | None,
        audio_start_ms: float = 0.0,
        verify_language: bool = False,
    ) -> EngineOutput:
        model = self._require_model()
        assert self._cfg is not None
        self._validate_audio(audio)
        if language is not None and language not in SUPPORTED_CODES:
            raise UnsupportedLanguageError(
                f"Language '{language}' is not supported",
                hint="Supported: " + ", ".join(sorted(SUPPORTED_CODES)),
            )
        lang_prob: float | None = None
        det: LanguageDetection | None = None
        if language is None or verify_language:
            det = self.detect_language(audio)
            confident = det.probability >= VERIFY_MIN_PROBABILITY
            if not det.supported and (language is None or confident):
                raise UnsupportedLanguageError(
                    f"Detected '{det.language}' (p={det.probability:.2f}) is unsupported",
                    hint="Select a supported source language manually if this is wrong.",
                    detected_language=det.language,
                    probability=det.probability,
                )
            if language is None or (det.supported and confident):
                language = det.language  # auto-detect, or confident correction of a pinned one
            if language == det.language:
                lang_prob = det.probability
        t0 = time.perf_counter()
        try:
            segments, info = model.transcribe(
                audio,
                task=self._cfg.task,  # "translate"
                language=language,
                beam_size=self._cfg.beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,  # our own Silero VAD runs upstream
                without_timestamps=False,
                no_speech_threshold=None,  # raw diagnostics go to OUR validator,
                log_prob_threshold=None,  # not silently filtered here
                compression_ratio_threshold=None,
            )
            seg_list = list(segments)  # lazy generator: consume inside the inference context
        except Exception as e:
            raise InferenceError(f"Whisper inference failed: {e}") from e
        inference_ms = (time.perf_counter() - t0) * 1000.0
        # When the caller forced the language, Whisper reports probability 1.0 - that is not a
        # measurement, so it is NOT propagated. Only a real detection yields a probability.
        _ = info
        audio_ms = audio.size * 1000.0 / SAMPLE_RATE
        results = [
            TranslationResult(
                source_language=language,
                translated_text=s.text.strip(),
                audio_start_ms=audio_start_ms + float(s.start) * 1000.0,
                audio_end_ms=audio_start_ms + max(float(s.end), float(s.start)) * 1000.0,
                model=self.model_label,
                diagnostics=ModelDiagnostics(
                    no_speech_prob=float(s.no_speech_prob),
                    avg_logprob=float(s.avg_logprob),
                    compression_ratio=float(s.compression_ratio),
                    language_probability=lang_prob,
                ),
            )
            for s in seg_list
        ]
        return EngineOutput(
            results,
            language,
            lang_prob,
            inference_ms,
            audio_ms,
            detected_language=det.language if det else None,
            detected_probability=det.probability if det else None,
        )


VERIFY_MIN_PROBABILITY = 0.5  # below this a detection never overrides or rejects


def create_engine(models_dir: str, model_config: dict[str, Any]) -> WhisperEngine:
    """Factory used by the inference worker process (picklable arguments only)."""
    engine = WhisperEngine(ModelManager(models_dir=Path(models_dir)))
    engine.load_model(ModelConfig.model_validate(model_config))
    return engine
