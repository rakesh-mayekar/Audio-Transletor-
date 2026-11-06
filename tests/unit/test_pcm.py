import numpy as np
import pytest

from core.errors import InvalidAudioError
from oslayer.common import StreamingResampler, to_mono


def sine(freq, rate, seconds, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


@pytest.mark.parametrize("rate", [48000, 44100, 32000, 22050, 8000, 16000])
def test_inband_tone_preserved(rate):
    r = StreamingResampler(rate, 16000)
    y = r.process(sine(1000, rate, 1.0))
    core = y[400:-400]
    assert abs(rms(core) - 0.5 / np.sqrt(2)) < 0.01
    assert abs(len(y) - 16000) < 40


@pytest.mark.parametrize("rate", [48000, 44100])
def test_out_of_band_tone_rejected(rate):
    y = StreamingResampler(rate, 16000).process(sine(11000, rate, 1.0))
    assert rms(y[400:-400]) < 0.5 / np.sqrt(2) * 0.01  # > 40 dB down


def test_chunking_does_not_change_output():
    x = sine(700, 44100, 1.0) + sine(3100, 44100, 1.0, 0.2)
    whole = StreamingResampler(44100, 16000).process(x)
    for chunk in (1, 7, 441, 1000, 4096):
        r = StreamingResampler(44100, 16000)
        parts = [r.process(x[i : i + chunk]) for i in range(0, x.size, chunk)]
        got = np.concatenate(parts)
        n = min(got.size, whole.size)
        assert abs(got.size - whole.size) <= 1
        assert np.allclose(got[:n], whole[:n], atol=1e-5), chunk


def test_identity_rate_passthrough():
    x = sine(500, 16000, 0.1)
    assert np.array_equal(StreamingResampler(16000, 16000).process(x), x)


def test_to_mono():
    st = np.array([1, 3, 2, 4], np.float32)
    assert np.array_equal(to_mono(st, 2), np.array([2, 3], np.float32))
    assert to_mono(st, 1) is st
    with pytest.raises(InvalidAudioError):
        to_mono(np.zeros(5, np.float32), 2)
    with pytest.raises(InvalidAudioError):
        StreamingResampler(0, 16000)
