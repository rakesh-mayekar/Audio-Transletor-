import numpy as np
import pytest

from core.errors import (
    AudioCaptureError,
    DeviceLostError,
    InvalidAudioError,
    PermissionDeniedError,
    TranslatorError,
)
from core.types import AudioFrame, CaptionState, TranslationResult, ms_to_samples, samples_to_ms


def f32(n):
    return np.zeros(n, np.float32)


def test_audio_frame_valid_and_readonly():
    fr = AudioFrame(100.0, f32(1600))
    assert fr.duration_ms == pytest.approx(100.0)
    assert fr.end_ms == pytest.approx(200.0)
    with pytest.raises(ValueError):
        fr.samples[0] = 1.0


@pytest.mark.parametrize(
    "samples",
    [np.zeros(10, np.float64), np.zeros((10, 2), np.float32), np.zeros(0, np.float32), [0.0, 1.0]],
)
def test_audio_frame_rejects_bad_samples(samples):
    with pytest.raises(InvalidAudioError):
        AudioFrame(0.0, samples)


def test_audio_frame_rejects_wrong_rate_and_negative_pts():
    with pytest.raises(InvalidAudioError):
        AudioFrame(0.0, f32(10), sample_rate=48000)
    with pytest.raises(InvalidAudioError):
        AudioFrame(-1.0, f32(10))


def test_ms_sample_roundtrip():
    assert ms_to_samples(250) == 4000
    assert samples_to_ms(4000) == 250.0


def test_translation_result_defaults_and_validation():
    r = TranslationResult("ja", "Hello", 0, 1000, "whisper-small")
    assert (
        r.target_language == "en" and r.status is CaptionState.CANDIDATE and r.source_text is None
    )
    with pytest.raises(InvalidAudioError):
        TranslationResult("ja", "x", 1000, 0, "m")


def test_error_hierarchy_and_formatting():
    e = PermissionDeniedError("denied", hint="Open Settings", device="x")
    assert isinstance(e, AudioCaptureError) and isinstance(e, TranslatorError)
    assert "PERMISSION_DENIED" in str(e) and "Open Settings" in str(e)
    d = e.to_dict()
    assert d["code"] == "PERMISSION_DENIED" and d["hint"] == "Open Settings"
    assert issubclass(DeviceLostError, AudioCaptureError)
