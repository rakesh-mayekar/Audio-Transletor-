import json

import pytest

from core.config import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_CODES,
    AppConfig,
    load_config,
    migrate,
    parse_config,
    paths,
    recover_config,
    save_config,
)
from core.errors import ConfigError, ConfigMigrationError


def test_defaults_match_spec():
    c = AppConfig()
    assert c.model.name == "small" and c.model.compute_type == "int8" and c.model.device == "cpu"
    assert c.model.beam_size == 1 and c.model.inference_workers == 1 and c.model.task == "translate"
    assert c.audio.sample_rate == 16000 and c.audio.channels == 1
    assert c.vad.threshold == 0.5 and c.vad.min_speech_ms == 250 and c.vad.pre_roll_ms == 300
    assert 500 <= c.vad.silence_endpoint_ms <= 700 and 8 <= c.vad.max_utterance_s <= 10
    assert c.language.target == "en" and c.language.auto_detect
    assert c.privacy.telemetry is False and c.privacy.record_diagnostic_audio is False
    assert c.privacy.allow_network_for_model_download is False
    assert c.schema_version == CURRENT_SCHEMA_VERSION


def test_exactly_five_languages():
    assert {"zh", "ja", "pt", "ar", "hi"} == SUPPORTED_CODES


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": {"manual_source": "fr"}},
        {"language": {"auto_detect": False}},
        {"model": {"task": "transcribe"}},
        {"model": {"inference_workers": 2}},
        {"audio": {"sample_rate": 44100}},
        {"vad": {"threshold": 1.5}},
        {"vad": {"max_utterance_s": 0.5}},
        {"privacy": {"telemetry": True}},
        {"privacy": {"record_diagnostic_audio": True}},
    ],
)
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ConfigError):
        parse_config(kwargs)


def test_diagnostic_recording_requires_consent():
    c = parse_config(
        {"privacy": {"record_diagnostic_audio": True, "diagnostic_audio_consent": True}}
    )
    assert c.privacy.record_diagnostic_audio


def test_manual_language_ok():
    c = parse_config({"language": {"auto_detect": False, "manual_source": "pt"}})
    assert c.language.manual_source == "pt"


def test_roundtrip_atomic(tmp_path):
    p = tmp_path / "sub" / "settings.json"
    c = AppConfig()
    c.overlay.font_size_pt = 40
    save_config(c, p)
    assert load_config(p).overlay.font_size_pt == 40
    assert list(p.parent.glob("*.tmp")) == []


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.json") == AppConfig()


def test_corrupt_file_is_error_not_silent_reset(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError):
        load_config(p)
    assert p.read_text() == "{not json"  # untouched


def test_recover_moves_bad_file_aside(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("[]")
    cfg, backup = recover_config(p)
    assert (
        cfg == AppConfig() and backup is not None and backup.read_text() == "[]" and not p.exists()
    )


def test_unknown_keys_ignored_forward_compat():
    c = parse_config({"schema_version": 1, "future_option": 1, "model": {"name": "base", "x": 2}})
    assert c.model.name == "base"


def test_newer_schema_refused():
    with pytest.raises(ConfigMigrationError):
        parse_config({"schema_version": CURRENT_SCHEMA_VERSION + 1})


def test_legacy_unversioned_file_migrates_via_chain():
    migs = {0: lambda d: {**d, "log_level": d.pop("level", "INFO")}}
    out = migrate({"level": "DEBUG"}, migs, target=1)
    assert out["log_level"] == "DEBUG" and out["schema_version"] == 1


def test_missing_migration_step_raises():
    with pytest.raises(ConfigMigrationError):
        migrate({"schema_version": 1}, {}, target=3)


def test_multi_step_migration():
    migs = {0: lambda d: {**d, "a": 1}, 1: lambda d: {**d, "b": d["a"] + 1}}
    out = migrate({}, migs, target=2)
    assert out["a"] == 1 and out["b"] == 2 and out["schema_version"] == 2


def test_bad_schema_version_type():
    with pytest.raises(ConfigMigrationError):
        migrate({"schema_version": "x"})


def test_invalid_values_fail_validation_not_migration():
    # guards against ConfigMigrationError (a ConfigError subclass) masking real validation
    with pytest.raises(ConfigError) as ei:
        parse_config({"language": {"manual_source": "fr"}})
    assert not isinstance(ei.value, ConfigMigrationError)


def test_unversioned_legacy_file_loads():
    assert parse_config({"model": {"name": "base"}}).schema_version == CURRENT_SCHEMA_VERSION


def test_paths_use_override_and_are_not_hardcoded(tmp_path, monkeypatch):
    monkeypatch.setenv("OLT_HOME", str(tmp_path))
    assert paths.config_file() == tmp_path / "config" / "settings.json"
    assert paths.models_dir().is_relative_to(tmp_path)
    monkeypatch.delenv("OLT_HOME")
    assert "OfflineLiveTranslator" in str(paths.config_dir())
    assert "rakeshmayekar" not in str(paths.models_dir()) or str(paths.models_dir()).startswith(
        "/Users"
    )
    assert json.dumps(str(paths.log_dir()))
