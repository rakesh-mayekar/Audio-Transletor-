import pytest

from core.errors import ProviderUnavailableError, UnsupportedPlatformError
from oslayer.base import AudioCaptureProvider
from oslayer.fake import FakeAudioCaptureProvider
from oslayer.selector import (
    OSName,
    clear_overrides,
    create_audio_provider,
    detect_os,
    register_audio_provider,
)


@pytest.mark.parametrize(
    "s,expected",
    [
        ("win32", OSName.WINDOWS),
        ("darwin", OSName.MACOS),
        ("linux", OSName.LINUX),
        ("linux2", OSName.LINUX),
    ],
)
def test_detect_os(s, expected):
    assert detect_os(s) is expected


def test_unknown_os_rejected():
    with pytest.raises(UnsupportedPlatformError):
        detect_os("freebsd13")


@pytest.mark.parametrize("os_name", list(OSName))
def test_missing_provider_is_explicit_error_never_a_fallback(os_name, monkeypatch):
    import oslayer.selector as sel

    clear_overrides()
    monkeypatch.setitem(sel._AUDIO_PROVIDERS, os_name, "oslayer.nonexistent.audio:Nope")
    with pytest.raises(ProviderUnavailableError) as ei:
        create_audio_provider(os_name)
    assert "microphone" in ei.value.hint.lower()


def test_override_registration():
    register_audio_provider(OSName.LINUX, FakeAudioCaptureProvider)
    try:
        p = create_audio_provider(OSName.LINUX)
        assert isinstance(p, AudioCaptureProvider)
    finally:
        clear_overrides()


def test_missing_dependency_inside_provider_is_not_masked(tmp_path, monkeypatch):
    import sys

    pkg = tmp_path / "oslayer_fake_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("import definitely_not_installed_xyz\nclass P: pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    from oslayer import selector

    monkeypatch.setitem(selector._AUDIO_PROVIDERS, OSName.WINDOWS, "oslayer_fake_pkg.mod:P")
    with pytest.raises(ModuleNotFoundError) as ei:
        create_audio_provider(OSName.WINDOWS)
    assert ei.value.name == "definitely_not_installed_xyz"
    sys.modules.pop("oslayer_fake_pkg.mod", None)
