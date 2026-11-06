"""OS provider logic that can be verified on any machine: parsing, commands, error mapping,
selector wiring. Real capture on Windows/Linux is NOT exercised here (see PHASE3_4_REPORT)."""

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core.errors import ProviderUnavailableError
from oslayer.base import DeviceKind
from oslayer.linux.audio_capture import LinuxAudioCaptureProvider, parse_pactl_sources
from oslayer.macos import audio_capture as mac
from oslayer.selector import OSName, create_audio_provider, detect_os
from oslayer.windows import wasapi_helper
from oslayer.windows.audio_capture import WindowsAudioCaptureProvider, parse_device_json

PACTL = (
    "0\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
    "1\talsa_input.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
    "2\tbluez_output.AA_BB.1.monitor\tmodule-bluez5-device.c\ts16le 2ch 48000Hz\tIDLE\n"
)


def test_linux_lists_only_monitors_never_microphones():
    devs = parse_pactl_sources(PACTL, "bluez_output.AA_BB.1")
    assert [d.id for d in devs] == [
        "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
        "bluez_output.AA_BB.1.monitor",
    ]
    assert all(d.kind is DeviceKind.SYSTEM_OUTPUT_LOOPBACK for d in devs)
    assert [d.is_default for d in devs] == [False, True]
    assert parse_pactl_sources("garbage\n\n", None) == []


def test_linux_command_uses_monitor_and_16k_mono():
    p = LinuxAudioCaptureProvider()
    cmd = p._build_command(None)
    assert cmd[0] == "parec" and "--device=@DEFAULT_MONITOR@" in cmd
    assert "--rate=16000" in cmd and "--channels=1" in cmd and "--format=float32le" in cmd
    dev = parse_pactl_sources(PACTL, None)[0]
    assert f"--device={dev.id}" in p._build_command(dev)


def test_linux_missing_tools_gives_install_hint(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ProviderUnavailableError, match="parec") as e:
        LinuxAudioCaptureProvider().initialize()
    assert "apt install" in (e.value.hint or "")


def test_macos_missing_helper_gives_build_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("OLT_AUDIO_TAP", str(tmp_path / "nope"))
    monkeypatch.setattr("platform.mac_ver", lambda: ("14.5", ("", "", ""), ""))
    with pytest.raises(ProviderUnavailableError) as e:
        mac.MacOSAudioCaptureProvider().initialize()
    assert "build_macos_helper" in (e.value.hint or "")


def test_macos_old_version_rejected(monkeypatch):
    from core.errors import UnsupportedPlatformError

    monkeypatch.setattr("platform.mac_ver", lambda: ("12.7", ("", "", ""), ""))
    with pytest.raises(UnsupportedPlatformError):
        mac.MacOSAudioCaptureProvider().initialize()


def test_macos_single_system_device_and_command(monkeypatch, tmp_path):
    monkeypatch.setenv("OLT_AUDIO_TAP", str(tmp_path / "tap"))
    p = mac.MacOSAudioCaptureProvider()
    assert [d.id for d in p.list_devices()] == ["system"]
    assert p._build_command(None) == [str(tmp_path / "tap")]


@pytest.mark.skipif(sys.platform != "darwin", reason="helper binary is built on macOS only")
def test_macos_helper_binary_present_and_executable_if_built():
    h = mac.helper_path()
    if not h.exists():
        pytest.skip("run scripts/build_macos_helper.sh")
    import os

    assert os.access(h, os.X_OK)


def test_windows_device_json_roundtrip():
    rows = [
        {"id": "7", "name": "Speakers [Loopback]", "is_default": True, "rate": 48000, "channels": 2}
    ]
    d = parse_device_json(json.dumps(rows))[0]
    assert d.id == "7" and d.is_default and d.native_sample_rate == 48000
    from core.errors import AudioCaptureError

    with pytest.raises(AudioCaptureError):
        parse_device_json("not json")


def test_windows_missing_backend_gives_install_hint(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda n: None)
    with pytest.raises(ProviderUnavailableError) as e:
        WindowsAudioCaptureProvider().initialize()
    assert "PyAudioWPatch" in (e.value.hint or "")


class FakePyAudio:
    """Just enough of PyAudioWPatch to drive stream_loopback()."""

    paWASAPI, paFloat32, paContinue = 13, 1, 0

    def __init__(self, devices):
        self._devices = devices
        self.opened = None

    def get_host_api_info_by_type(self, _):
        return {"defaultOutputDevice": 1}

    def get_device_info_by_index(self, i):
        return {"name": "Speakers"}

    def get_loopback_device_info_generator(self):
        yield from self._devices

    def open(self, **kw):
        self.opened = kw
        data = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(4800 * 20) / 48000)).astype("<f4")
        stereo = np.repeat(data[:, None], kw["channels"], axis=1)
        for i in range(0, len(stereo), 960):
            kw["stream_callback"](stereo[i : i + 960].tobytes(), 960, None, 0)
        return types.SimpleNamespace(stop_stream=lambda: None, close=lambda: None)


LOOP = [
    {"index": 7, "name": "Speakers [Loopback]", "defaultSampleRate": 48000, "maxInputChannels": 2}
]


def test_windows_helper_streams_16k_mono_from_loopback():
    pa = FakePyAudio(LOOP)
    out = io.BytesIO()
    assert wasapi_helper.stream_loopback(pa, None, out, max_seconds=1.0) == 0
    x = np.frombuffer(out.getvalue(), dtype="<f4")
    assert pa.opened["input_device_index"] == 7 and pa.opened["rate"] == 48000
    assert abs(float(np.sqrt(np.mean(x[2000:15000] ** 2))) - 0.5 / np.sqrt(2)) < 0.02


def test_windows_helper_without_loopback_device_exits_unavailable():
    assert wasapi_helper.stream_loopback(FakePyAudio([]), None, io.BytesIO(), 1.0) == 69


def test_selector_resolves_each_os_to_its_provider():
    assert type(create_audio_provider(OSName.LINUX)).__name__ == "LinuxAudioCaptureProvider"
    assert type(create_audio_provider(OSName.MACOS)).__name__ == "MacOSAudioCaptureProvider"
    assert type(create_audio_provider(OSName.WINDOWS)).__name__ == "WindowsAudioCaptureProvider"
    assert detect_os("linux") is OSName.LINUX


def test_helper_script_is_importable_without_pyaudio():
    assert Path(wasapi_helper.__file__).exists()
