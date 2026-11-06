"""Windows system-audio capture: WASAPI loopback through a child-process helper.

No special permission is required for loopback capture on Windows 10/11. Needs the optional
dependency `PyAudioWPatch` (pip install .[windows]).
STATUS: see oslayer/windows/wasapi_helper.py - never run on real Windows.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from core.errors import AudioCaptureError, ProviderUnavailableError
from oslayer.base import AudioDevice, DeviceKind
from oslayer.common.process_capture import PcmProcessCaptureProvider


def parse_device_json(text: str) -> list[AudioDevice]:
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as e:
        raise AudioCaptureError(f"unreadable device list: {e}") from e
    return [
        AudioDevice(
            id=r["id"],
            name=r["name"],
            kind=DeviceKind.SYSTEM_OUTPUT_LOOPBACK,
            is_default=bool(r.get("is_default")),
            native_sample_rate=r.get("rate"),
            native_channels=r.get("channels"),
        )
        for r in rows
    ]


class WindowsAudioCaptureProvider(PcmProcessCaptureProvider):
    platform_name = "windows"
    helper_rate = 16_000  # the helper resamples and downmixes itself
    helper_channels = 1
    permission_hint = "Loopback capture needs no permission; check the output device is enabled."

    def _check_available(self) -> None:
        if importlib.util.find_spec("pyaudiowpatch") is None:
            raise ProviderUnavailableError(
                "PyAudioWPatch is not installed",
                hint="Install it: pip install PyAudioWPatch (provides WASAPI loopback).",
            )

    def _discover_devices(self) -> list[AudioDevice]:
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "oslayer.windows.wasapi_helper", "--list"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if r.returncode != 0:
            raise AudioCaptureError(
                f"Could not list loopback devices: {r.stderr.strip()}",
                hint="Is an audio output device enabled?",
            )
        return parse_device_json(r.stdout)

    def _build_command(self, device: AudioDevice | None) -> list[str]:
        cmd = [sys.executable, "-m", "oslayer.windows.wasapi_helper"]
        return [*cmd, "--device", device.id] if device else cmd
