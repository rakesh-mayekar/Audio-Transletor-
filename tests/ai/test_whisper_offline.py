"""Real-model tests (Whisper Small INT8, CPU). Skipped if the model is not installed.

IMPORTANT: speech here is SYNTHETIC (macOS system voices). These are smoke/regression tests,
not a measurement of real-world translation quality.
"""

import time

import numpy as np
import pytest

from ai.inference_worker import InferenceWorker, WorkerRequest
from core.config import SUPPORTED_CODES
from core.errors import InvalidAudioError, ModelMissingError, UnsupportedLanguageError
from tests.ai.conftest import MODELS, requires_model

pytestmark = [pytest.mark.ai, requires_model]

KEYWORDS = ("weather", "park")
LANGS = ["zh", "ja", "pt", "ar", "hi"]


def test_supported_set_is_exactly_five():
    assert set(LANGS) == SUPPORTED_CODES


@pytest.mark.parametrize("code", LANGS)
def test_language_detection(engine, speech, code, no_network):
    d = engine.detect_language(speech(code))
    assert d.language == code and d.supported and d.probability > 0.5
    assert d.supported_mass >= d.probability


@pytest.mark.parametrize(
    "code",
    [
        "zh",
        "ja",
        "pt",
        "hi",
        pytest.param(
            "ar",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Known: Whisper Small drops 'weather' in this Arabic sample (see PHASE5_REPORT)",
            ),
        ),
    ],
)
def test_translates_to_english_offline(engine, speech, code, no_network):
    audio = speech(code)
    out = engine.translate_audio(audio, code)
    text = " ".join(r.translated_text for r in out.results).lower()
    assert out.results and text.isascii(), text  # English output, not source-language script
    assert all(k in text for k in KEYWORDS), text
    assert out.rtf < 0.7, f"RTF {out.rtf:.2f}"


@pytest.mark.parametrize("code", LANGS)
def test_arabic_and_all_languages_produce_ascii_english(engine, speech, code, no_network):
    out = engine.translate_audio(speech(code), code)
    text = " ".join(r.translated_text for r in out.results)
    assert text.strip() and text.isascii()


def test_auto_mode_detects_then_translates(engine, speech, no_network):
    out = engine.translate_audio(speech("ja"), None)
    assert out.language == "ja" and out.language_probability and out.language_probability > 0.5


def test_explicit_language_has_no_fabricated_probability(engine, speech, no_network):
    out = engine.translate_audio(speech("pt"), "pt")
    assert out.language_probability is None
    assert all(r.diagnostics.language_probability is None for r in out.results)


def test_timestamps_offset_by_audio_start_and_ordered(engine, speech, no_network):
    out = engine.translate_audio(speech("zh"), "zh", audio_start_ms=60_000.0)
    starts = [r.audio_start_ms for r in out.results]
    assert starts == sorted(starts) and starts[0] >= 60_000.0
    assert all(r.audio_end_ms >= r.audio_start_ms for r in out.results)


def test_diagnostics_are_populated(engine, speech, no_network):
    d = engine.translate_audio(speech("hi"), "hi").results[0].diagnostics
    assert None not in (d.no_speech_prob, d.avg_logprob, d.compression_ratio)
    assert d.no_speech_prob < 0.6 and d.avg_logprob > -1.0  # clean speech passes default thresholds


def test_task_is_translate_not_transcribe(engine, speech, no_network):
    # transcribe would return Hindi/Japanese script; translate returns English
    text = engine.translate_audio(speech("ja"), "ja").results[0].translated_text
    assert text.isascii()


@pytest.mark.parametrize("code", ["en", "fr"])
def test_unsupported_language_is_rejected_not_coerced(engine, speech, code, no_network):
    audio = speech(code)
    d = engine.detect_language(audio)
    assert not d.supported
    with pytest.raises(UnsupportedLanguageError):
        engine.translate_audio(audio, None)
    with pytest.raises(UnsupportedLanguageError):
        engine.translate_audio(audio, code)


def test_too_short_or_bad_audio_rejected(engine, no_network):
    with pytest.raises(InvalidAudioError):
        engine.translate_audio(np.zeros(800, np.float32), "ja")
    with pytest.raises(InvalidAudioError):
        engine.translate_audio(np.zeros(16000, np.float64), "ja")


def test_raw_whisper_hallucinates_on_non_speech_but_diagnostics_flag_it(engine, no_network):
    """Documents WHY VAD + validation are mandatory: the model emits text for non-speech.
    The default no_speech_prob threshold (0.6) must catch each of these."""
    rng = np.random.default_rng(0)
    cases = {
        "silence": np.zeros(48000, np.float32),
        "quiet-noise": (rng.standard_normal(48000) * 0.01).astype(np.float32),
        "loud-noise": (rng.standard_normal(48000) * 0.2).astype(np.float32),
        "tone": (0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 16000)).astype(np.float32),
    }
    for name, audio in cases.items():
        for r in engine.translate_audio(audio, "ja").results:
            assert r.diagnostics.no_speech_prob > 0.6, (name, r.translated_text, r.diagnostics)


def test_deterministic_output(engine, speech, no_network):
    a = engine.translate_audio(speech("pt"), "pt").results[0].translated_text
    b = engine.translate_audio(speech("pt"), "pt").results[0].translated_text
    assert a == b


def test_unload_and_reload_single_instance(speech):
    from ai.model_manager import ModelManager
    from ai.whisper_engine import WhisperEngine
    from core.config.models import ModelConfig

    e = WhisperEngine(ModelManager(MODELS))
    e.load_model(ModelConfig())
    first = e._model
    e.load_model(ModelConfig())  # no second instance
    assert e._model is first
    e.unload_model()
    assert not e.is_loaded


def test_missing_model_never_triggers_download(tmp_path, no_network):
    from ai.model_manager import ModelManager
    from ai.whisper_engine import WhisperEngine
    from core.config.models import ModelConfig

    with pytest.raises(ModelMissingError):
        WhisperEngine(ModelManager(tmp_path)).load_model(ModelConfig())


def test_full_pipeline_through_worker_process(speech):
    w = InferenceWorker(
        "ai.whisper_engine:create_engine",
        {"models_dir": str(MODELS), "model_config": {"name": "small"}},
        max_pending=2,
    )
    t0 = time.monotonic()
    w.start(timeout=180)
    load_s = time.monotonic() - t0
    proc = w._proc
    try:
        assert w.submit(WorkerRequest("p", speech("pt"), "pt", 5000.0, time.time() + 60))
        deadline = time.monotonic() + 60
        r = None
        while r is None and time.monotonic() < deadline:
            r = w.poll(0.5)
        assert r and r.error is None, r
        text = r.output.results[0].translated_text.lower()
        assert "park" in text and r.output.results[0].audio_start_ms >= 5000.0
        print(f"worker load {load_s:.1f}s, rtf {r.output.rtf:.2f}")
    finally:
        w.stop()
    assert not proc.is_alive()
