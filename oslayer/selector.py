"""Chooses the platform provider at runtime. Providers are imported lazily, so importing this
module never pulls in Windows/macOS/Linux-only dependencies on the wrong OS."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from enum import StrEnum

from core.errors import ProviderUnavailableError, UnsupportedPlatformError
from oslayer.base import AudioCaptureProvider


class OSName(StrEnum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


def detect_os(platform_string: str | None = None) -> OSName:
    p = platform_string if platform_string is not None else sys.platform
    if p.startswith("win"):
        return OSName.WINDOWS
    if p == "darwin":
        return OSName.MACOS
    if p.startswith("linux"):
        return OSName.LINUX
    raise UnsupportedPlatformError(
        f"Unsupported operating system: {p}",
        hint="Supported: Windows 10/11, macOS, Linux (Ubuntu).",
    )


# "module:Class" — resolved lazily
_AUDIO_PROVIDERS: dict[OSName, str] = {
    OSName.WINDOWS: "oslayer.windows.audio_capture:WindowsAudioCaptureProvider",
    OSName.MACOS: "oslayer.macos.audio_capture:MacOSAudioCaptureProvider",
    OSName.LINUX: "oslayer.linux.audio_capture:LinuxAudioCaptureProvider",
}
_OVERRIDES: dict[OSName, Callable[[], AudioCaptureProvider]] = {}


def register_audio_provider(os_name: OSName, factory: Callable[[], AudioCaptureProvider]) -> None:
    """Explicit override (tests / future plugins). Not used implicitly."""
    _OVERRIDES[os_name] = factory


def clear_overrides() -> None:
    _OVERRIDES.clear()


def create_audio_provider(os_name: OSName | None = None) -> AudioCaptureProvider:
    os_name = os_name or detect_os()
    if os_name in _OVERRIDES:
        return _OVERRIDES[os_name]()
    module_name, _, cls_name = _AUDIO_PROVIDERS[os_name].partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # Only "this provider module doesn't exist" is translated; a missing dependency
        # INSIDE an existing provider is a real error and must surface as such.
        if e.name is not None and (module_name == e.name or module_name.startswith(e.name + ".")):
            raise ProviderUnavailableError(
                f"No audio capture provider is implemented for {os_name.value} in this build",
                hint="No microphone fallback is used. See docs/PHASE0_DISCOVERY.md (roadmap).",
                module=module_name,
            ) from e
        raise
    provider: AudioCaptureProvider = getattr(module, cls_name)()
    return provider
