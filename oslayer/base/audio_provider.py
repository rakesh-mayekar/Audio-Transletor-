"""The ONLY OS-specific audio boundary. Shared code depends on this interface alone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType

from core.errors import AudioCaptureError, DeviceNotFoundError
from core.types import AudioFrame


class DeviceKind(StrEnum):
    SYSTEM_OUTPUT_LOOPBACK = "system_output_loopback"
    MICROPHONE = "microphone"  # modelled only so it can be REJECTED explicitly


@dataclass(frozen=True, slots=True)
class AudioDevice:
    id: str
    name: str
    kind: DeviceKind = DeviceKind.SYSTEM_OUTPUT_LOOPBACK
    is_default: bool = False
    native_sample_rate: int | None = None
    native_channels: int | None = None


class CaptureState(StrEnum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    CAPTURING = "capturing"
    RECOVERING = "recovering"  # device change/loss; provider is trying to reconnect
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaptureStatus:
    state: CaptureState
    platform: str
    device: AudioDevice | None = None
    frames_emitted: int = 0
    frames_dropped: int = 0  # dropped because the consumer was too slow (bounded buffer)
    last_error: str | None = None


class AudioCaptureProvider(ABC):
    """Contract every platform provider must satisfy (verified by the conformance suite).

    Emitted frames MUST be 16 kHz / mono / float32 (see core.types.AudioFrame) with `pts_ms`
    taken from the capture clock, and internal buffering MUST be bounded.
    A provider MUST NOT capture the microphone or fall back to it; on failure it raises
    a typed AudioCaptureError subclass with an actionable hint.
    """

    #: "windows" | "macos" | "linux" | "fake"
    platform_name: str = "unknown"

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def list_devices(self) -> list[AudioDevice]:
        """Playback-loopback devices only. Never microphones."""

    @abstractmethod
    def select_device(self, device_id: str | None) -> None:
        """None selects the system default output. Unknown id -> DeviceNotFoundError."""

    @abstractmethod
    def start_capture(self) -> None: ...

    @abstractmethod
    def read_audio(self, timeout: float = 0.1) -> AudioFrame | None:
        """Next frame, or None if none arrived within `timeout`. Must not block the UI thread
        indefinitely; raises AudioCaptureError if capture has failed unrecoverably."""

    @abstractmethod
    def stop_capture(self) -> None: ...

    @abstractmethod
    def get_status(self) -> CaptureStatus: ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release every OS resource. Idempotent."""

    # -- helpers shared by implementations -----------------------------------
    def resolve_device(self, device_id: str | None) -> AudioDevice | None:
        devices = self.list_devices()
        if device_id is None:
            return next((d for d in devices if d.is_default), devices[0] if devices else None)
        for d in devices:
            if d.id == device_id:
                if d.kind is DeviceKind.MICROPHONE:
                    raise AudioCaptureError(
                        "Microphone devices cannot be used for system-audio capture",
                        hint="Select an output (loopback) device.",
                    )
                return d
        raise DeviceNotFoundError(
            f"Audio device '{device_id}' not found",
            hint="Call list_devices() for current devices; the device may have been unplugged.",
        )

    def __enter__(self) -> AudioCaptureProvider:
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()
