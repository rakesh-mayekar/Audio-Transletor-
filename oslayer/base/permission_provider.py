from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class PermissionKind(StrEnum):
    SYSTEM_AUDIO_CAPTURE = "system_audio_capture"
    SCREEN_RECORDING = "screen_recording"  # macOS ScreenCaptureKit
    AUDIO_SERVICE_ACCESS = "audio_service_access"  # e.g. PipeWire socket access


class PermissionState(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"
    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"  # cannot be determined programmatically on this OS


@dataclass(frozen=True, slots=True)
class PermissionStatus:
    kind: PermissionKind
    state: PermissionState
    instructions: str = ""  # actionable text shown to the user when not granted


class PermissionProvider(ABC):
    @abstractmethod
    def check(self) -> list[PermissionStatus]: ...

    @abstractmethod
    def request(self, kind: PermissionKind) -> PermissionStatus:
        """Trigger the OS prompt where one exists; otherwise return instructions."""
