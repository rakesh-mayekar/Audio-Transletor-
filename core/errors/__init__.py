"""Typed error hierarchy. Every error carries a stable code and an actionable hint.

Errors are never swallowed: components either raise them or publish them as ErrorOccurred events.
"""

from __future__ import annotations

from typing import Any


class TranslatorError(Exception):
    """Base class for all application errors."""

    code: str = "TRANSLATOR_ERROR"

    def __init__(self, message: str, *, hint: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = details

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" + (f" (hint: {self.hint})" if self.hint else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": {
                k: v if isinstance(v, str | int | float | bool | None) else repr(v)
                for k, v in self.details.items()
            },
        }


# --- configuration -----------------------------------------------------------
class ConfigError(TranslatorError):
    code = "CONFIG_INVALID"


class ConfigMigrationError(ConfigError):
    code = "CONFIG_MIGRATION"


# --- state -------------------------------------------------------------------
class InvalidStateTransition(TranslatorError):
    code = "INVALID_STATE_TRANSITION"


# --- platform / audio --------------------------------------------------------
class UnsupportedPlatformError(TranslatorError):
    code = "PLATFORM_UNSUPPORTED"


class ProviderUnavailableError(TranslatorError):
    """A platform provider is not implemented/installed in this build."""

    code = "PROVIDER_UNAVAILABLE"


class AudioCaptureError(TranslatorError):
    code = "AUDIO_CAPTURE"


class PermissionDeniedError(AudioCaptureError):
    code = "PERMISSION_DENIED"


class DeviceNotFoundError(AudioCaptureError):
    code = "DEVICE_NOT_FOUND"


class DeviceLostError(AudioCaptureError):
    code = "DEVICE_LOST"


class InvalidAudioError(TranslatorError):
    code = "AUDIO_INVALID"


# --- models / inference (used from Phase 5) ----------------------------------
class ModelError(TranslatorError):
    code = "MODEL_ERROR"


class ModelMissingError(ModelError):
    code = "MODEL_MISSING"


class ModelIntegrityError(ModelError):
    code = "MODEL_INTEGRITY"


class ModelNetworkNotAllowedError(ModelError):
    code = "MODEL_NETWORK_NOT_ALLOWED"


class InferenceError(TranslatorError):
    code = "INFERENCE_ERROR"


class UnsupportedLanguageError(TranslatorError):
    """Speech detected in a language outside the supported set. Never coerced to a supported one."""

    code = "LANGUAGE_UNSUPPORTED"
