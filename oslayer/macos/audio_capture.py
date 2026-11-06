"""macOS system-audio capture via a small ScreenCaptureKit helper (needs macOS 13+).

macOS has no loopback device; ScreenCaptureKit is Apple's supported way to capture system
playback and requires the user to grant *Screen Recording* permission (to the terminal or app
that launches the helper). The helper streams 48 kHz stereo float32 to stdout.
Build once with scripts/build_macos_helper.sh (needs Xcode command line tools).
STATUS: helper compiles; real capture requires an interactive permission grant and has not
been run by the automated tests (see docs/PHASE3_4_REPORT.md).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from core.errors import ProviderUnavailableError, UnsupportedPlatformError
from oslayer.base import AudioDevice, DeviceKind
from oslayer.common.process_capture import PcmProcessCaptureProvider

HELPER_NAME = "olt-audio-tap"
SYSTEM_DEVICE = AudioDevice(
    id="system",
    name="System audio (all apps)",
    kind=DeviceKind.SYSTEM_OUTPUT_LOOPBACK,
    is_default=True,
)


def helper_path() -> Path:
    override = os.environ.get("OLT_AUDIO_TAP")
    return Path(override) if override else Path(__file__).with_name(HELPER_NAME)


class MacOSAudioCaptureProvider(PcmProcessCaptureProvider):
    platform_name = "macos"
    helper_rate = 48_000
    helper_channels = 2
    permission_hint = (
        "Open System Settings > Privacy & Security > Screen & System Audio Recording, enable "
        "the app that runs this program (e.g. Terminal), then restart it."
    )

    def _check_available(self) -> None:
        ver = platform.mac_ver()[0]
        if ver and int(ver.split(".")[0]) < 13:
            raise UnsupportedPlatformError(
                f"macOS {ver} is too old for system-audio capture",
                hint="ScreenCaptureKit audio capture needs macOS 13 or newer.",
            )
        p = helper_path()
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise ProviderUnavailableError(
                f"macOS audio helper not found at {p}",
                hint="Build it: bash scripts/build_macos_helper.sh (requires Xcode command line "
                "tools: xcode-select --install).",
            )

    def _discover_devices(self) -> list[AudioDevice]:
        return [SYSTEM_DEVICE]  # ScreenCaptureKit captures the mixed system output

    def _build_command(self, device: AudioDevice | None) -> list[str]:
        return [str(helper_path())]
