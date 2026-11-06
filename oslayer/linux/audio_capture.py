"""Linux system-audio capture via the PulseAudio API (`parec`/`pactl`).

Works on PulseAudio and on PipeWire (through pipewire-pulse, the default on current Ubuntu).
System playback is captured from a sink's *monitor* source; microphones are never listed.
STATUS: logic verified with recorded `pactl` output and a scripted helper on macOS; NOT yet run
against a real Linux sound server (see docs/PHASE3_4_REPORT.md).
"""

from __future__ import annotations

import shutil
import subprocess

from core.errors import AudioCaptureError, ProviderUnavailableError
from oslayer.base import AudioDevice, DeviceKind
from oslayer.common.process_capture import PcmProcessCaptureProvider

DEFAULT_MONITOR = "@DEFAULT_MONITOR@"


def parse_pactl_sources(short_listing: str, default_sink: str | None) -> list[AudioDevice]:
    """Parse `pactl list short sources`; keep only sink monitors (= playback loopback)."""
    devices: list[AudioDevice] = []
    for line in short_listing.splitlines():
        cols = line.split("\t") if "\t" in line else line.split()
        if len(cols) < 2:
            continue
        name = cols[1]
        if not name.endswith(".monitor"):
            continue  # real input (microphone, line-in): excluded by design
        sink = name.removesuffix(".monitor")
        devices.append(
            AudioDevice(
                id=name,
                name=f"{sink} (monitor)",
                kind=DeviceKind.SYSTEM_OUTPUT_LOOPBACK,
                is_default=default_sink is not None and sink == default_sink,
            )
        )
    return devices


class LinuxAudioCaptureProvider(PcmProcessCaptureProvider):
    platform_name = "linux"
    helper_rate = 16_000  # parec resamples/downmixes server-side
    helper_channels = 1
    permission_hint = "Check that your user can access the sound server (pactl info)."

    def _check_available(self) -> None:
        missing = [t for t in ("parec", "pactl") if shutil.which(t) is None]
        if missing:
            raise ProviderUnavailableError(
                f"Required tool(s) not found: {', '.join(missing)}",
                hint="Install them: sudo apt install pulseaudio-utils (PipeWire users: "
                "also pipewire-pulse).",
            )

    @staticmethod
    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)  # noqa: S603
        except (OSError, subprocess.TimeoutExpired) as e:
            raise AudioCaptureError(
                f"{args[0]} failed: {e}", hint="Is the sound server running?"
            ) from e
        if r.returncode != 0:
            raise AudioCaptureError(
                f"{' '.join(args)} exited {r.returncode}: {r.stderr.strip()}",
                hint="Is the sound server running? Try: pactl info",
            )
        return r.stdout

    def _discover_devices(self) -> list[AudioDevice]:
        listing = self._run(["pactl", "list", "short", "sources"])
        try:
            default_sink = self._run(["pactl", "get-default-sink"]).strip() or None
        except AudioCaptureError:
            default_sink = None  # older pactl: no default marking
        return parse_pactl_sources(listing, default_sink)

    def _build_command(self, device: AudioDevice | None) -> list[str]:
        return [
            "parec",
            f"--device={device.id if device else DEFAULT_MONITOR}",
            "--format=float32le",
            f"--rate={self.helper_rate}",
            f"--channels={self.helper_channels}",
            "--latency-msec=40",
            "--raw",
        ]
