"""Pydantic configuration models. All values are *starting points*, not tuned constants."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config.languages import SUPPORTED_CODES

CURRENT_SCHEMA_VERSION = 1


class _Section(BaseModel):
    # Unknown keys are ignored so a newer minor config can still be read (forward compat);
    # schema_version is what gates incompatible changes.
    model_config = ConfigDict(extra="ignore", validate_assignment=True, frozen=False)


class LanguageConfig(_Section):
    auto_detect: bool = True
    manual_source: str | None = None
    target: Literal["en"] = "en"
    min_detection_speech_s: float = Field(3.0, ge=1.0, le=30.0)
    switch_hysteresis_utterances: int = Field(2, ge=1, le=10)
    # Re-check an auto-detected, pinned language on every Nth utterance (1 = every one, safest;
    # 0 = never). Each check costs one extra encoder pass (~+650 ms on an M2 Pro, Small INT8).
    verify_interval: int = Field(1, ge=0, le=20)

    @model_validator(mode="after")
    def _check(self) -> LanguageConfig:
        if self.manual_source is not None and self.manual_source not in SUPPORTED_CODES:
            raise ValueError(
                f"manual_source '{self.manual_source}' unsupported; "
                f"use one of {sorted(SUPPORTED_CODES)}"
            )
        if not self.auto_detect and self.manual_source is None:
            raise ValueError("auto_detect is off, so manual_source must be set")
        return self


class AudioConfig(_Section):
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    device_id: str | None = None  # None = platform default output
    ring_buffer_seconds: float = Field(30.0, ge=5.0, le=300.0)
    max_pending_frames: int = Field(256, ge=8, le=10_000)


class VadConfig(_Section):
    threshold: float = Field(0.5, gt=0.0, lt=1.0)
    min_speech_ms: int = Field(250, ge=50, le=5000)
    silence_endpoint_ms: int = Field(600, ge=100, le=3000)
    pre_roll_ms: int = Field(300, ge=0, le=2000)
    max_utterance_s: float = Field(9.0, ge=2.0, le=30.0)
    overlap_ms: int = Field(300, ge=0, le=2000)

    @model_validator(mode="after")
    def _check(self) -> VadConfig:
        if self.max_utterance_s * 1000 <= self.min_speech_ms + self.silence_endpoint_ms:
            raise ValueError("max_utterance_s must exceed min_speech_ms + silence_endpoint_ms")
        return self


class PerformanceProfile(StrEnum):
    LOW_RESOURCE = "low_resource"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class ModelConfig(_Section):
    name: Literal["base", "small", "medium", "large-v3"] = "small"
    compute_type: Literal["int8", "int8_float16", "float16", "float32"] = "int8"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    beam_size: int = Field(1, ge=1, le=10)
    inference_workers: int = Field(1, ge=1, le=1)  # one model instance by design
    cpu_threads: int | None = Field(None, ge=1, le=256)  # None = decided by hardware detector
    task: Literal["translate"] = "translate"  # never "transcribe" for the caption pipeline
    models_dir: str | None = None  # None = per-OS user data dir


class ValidationConfig(_Section):
    """Diagnostic thresholds. Defaults mirror Whisper's own heuristics and are UNTUNED;
    they must be re-evaluated per language against human-reviewed data (Phase 7)."""

    no_speech_prob_max: float = Field(0.6, ge=0.0, le=1.0)
    avg_logprob_min: float = Field(-1.0, le=0.0)
    compression_ratio_max: float = Field(2.4, ge=1.0, le=10.0)
    min_language_probability: float = Field(0.5, ge=0.0, le=1.0)
    max_staleness_ms: int = Field(15_000, ge=1000, le=120_000)
    duplicate_similarity: float = Field(0.9, ge=0.5, le=1.0)


class OverlayConfig(_Section):
    visible: bool = True
    opacity: float = Field(0.85, ge=0.1, le=1.0)
    font_size_pt: int = Field(28, ge=10, le=96)
    click_through: bool = False
    always_on_top: bool = True
    position: tuple[int, int] | None = None  # None = bottom-centre of primary screen
    caption_timeout_s: float = Field(6.0, ge=1.0, le=60.0)
    history_size: int = Field(100, ge=0, le=5000)


class PrivacyConfig(_Section):
    save_caption_history: bool = True
    record_diagnostic_audio: bool = False
    diagnostic_audio_consent: bool = False
    telemetry: Literal[False] = False  # no telemetry in this product
    allow_network_for_model_download: bool = False  # explicit opt-in per install/update

    @model_validator(mode="after")
    def _consent(self) -> PrivacyConfig:
        if self.record_diagnostic_audio and not self.diagnostic_audio_consent:
            raise ValueError("record_diagnostic_audio requires explicit diagnostic_audio_consent")
        return self


class AppConfig(_Section):
    schema_version: int = CURRENT_SCHEMA_VERSION
    profile: PerformanceProfile = PerformanceProfile.LOW_RESOURCE
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
