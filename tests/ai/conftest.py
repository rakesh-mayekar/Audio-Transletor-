import socket
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from ai.model_manager import ModelManager
from scripts import make_test_audio

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def model_installed(name: str = "small") -> bool:
    try:
        ModelManager(MODELS).verify(name, full=False)
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(not model_installed(), reason="run scripts/install_models.py")


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


@pytest.fixture(scope="session")
def speech():
    def get(code: str) -> np.ndarray:
        p = make_test_audio.OUT / f"{code}.wav"
        if not p.exists():
            if sys.platform != "darwin":
                pytest.skip("synthetic speech generator needs macOS; provide tests/ai/data/*.wav")
            make_test_audio.make(code)
        return load_wav(p)

    return get


@pytest.fixture(scope="session")
def engine():
    from ai.whisper_engine import WhisperEngine
    from core.config.models import ModelConfig

    e = WhisperEngine(ModelManager(MODELS))
    e.load_model(ModelConfig())
    yield e
    e.unload_model()


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a network connection fails the test."""

    def blocked(*a, **k):
        raise AssertionError("NETWORK ACCESS ATTEMPTED during offline inference")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
