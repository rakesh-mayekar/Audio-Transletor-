"""The shipped command line, run as a real subprocess against the real offline model."""

import subprocess
import sys

import pytest

from tests.ai.conftest import MODELS, ROOT, requires_model

pytestmark = pytest.mark.ai


def cli(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "-m", "app", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
    )


@requires_model
def test_wav_replay_prints_english_caption(speech):
    from scripts import make_test_audio  # ensure the sample exists

    speech("zh")
    r = cli(
        "--wav", str(make_test_audio.OUT / "zh.wav"), "--models-dir", str(MODELS), "--speed", "4"
    )
    assert r.returncode == 0, r.stderr
    assert "weather" in r.stdout.lower() and "language: zh" in r.stdout


@requires_model
def test_unsupported_language_is_reported_not_translated(speech):
    from scripts import make_test_audio

    speech("fr")
    r = cli(
        "--wav", str(make_test_audio.OUT / "fr.wav"), "--models-dir", str(MODELS), "--speed", "4"
    )
    assert r.returncode == 0
    assert "unsupported_language" in r.stdout
    assert "weather" not in r.stdout.lower() and "beautiful" not in r.stdout.lower()


def test_missing_model_exits_nonzero_with_actionable_message(tmp_path, speech):
    from scripts import make_test_audio

    speech("pt")
    r = cli("--wav", str(make_test_audio.OUT / "pt.wav"), "--models-dir", str(tmp_path))
    assert r.returncode == 2
    assert "install_models.py" in r.stderr
