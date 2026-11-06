import pytest

from ai.engine import LanguageDetection
from ai.language_detector import LanguageDetector
from core.config import SUPPORTED_CODES, LanguageConfig
from core.state import LanguageMode, LanguageState, LanguageStateMachine


def det(lang, p, supported=None):
    return LanguageDetection(lang, p, lang in SUPPORTED_CODES if supported is None else supported)


@pytest.fixture
def d():
    return LanguageDetector(LanguageStateMachine(SUPPORTED_CODES), LanguageConfig())


def test_no_decision_from_tiny_fragments(d):
    # 0.4 s of speech each, three times = 1.2 s < 3.0 s needed -> still no decision
    for _ in range(3):
        assert d.observe(det("ja", 0.99), 0.4) is LanguageState.DETECTING
    assert d.active_language is None


def test_confirms_with_enough_evidence(d):
    d.observe(det("ja", 0.95), 2.0)
    assert d.observe(det("ja", 0.97), 2.0) is LanguageState.CONFIRMED
    assert d.active_language == "ja"


def test_unsupported_language_is_never_mapped(d):
    d.observe(det("en", 0.98), 2.0)
    assert d.observe(det("en", 0.99), 2.0) is LanguageState.UNSUPPORTED
    assert d.active_language is None


def test_split_evidence_is_low_confidence(d):
    d.observe(det("ja", 0.6), 2.0)
    assert d.observe(det("zh", 0.6), 2.0) is LanguageState.LOW_CONFIDENCE
    assert d.active_language is None


def test_weak_probability_is_low_confidence(d):
    d.observe(det("hi", 0.3), 2.0)
    assert d.observe(det("hi", 0.35), 2.0) is LanguageState.LOW_CONFIDENCE


def test_recovers_from_low_confidence(d):
    d.observe(det("ja", 0.3), 2.0)
    d.observe(det("ja", 0.3), 2.0)
    assert d.machine.state is LanguageState.LOW_CONFIDENCE
    for _ in range(8):
        d.observe(det("ja", 0.97), 2.0)
    assert d.machine.state is LanguageState.CONFIRMED and d.active_language == "ja"


def test_single_odd_utterance_cannot_flip_confirmed_language(d):
    d.observe(det("ja", 0.97), 2.0)
    d.observe(det("ja", 0.97), 2.0)
    assert d.active_language == "ja"
    # sustained pt evidence is needed, one utterance must not flip
    for _ in range(4):
        d.observe(det("ja", 0.97), 2.0)  # keep ja dominant in window
    d.observe(det("pt", 0.99), 2.0)
    assert d.active_language == "ja"


def test_sustained_change_switches_language():
    cfg = LanguageConfig(switch_hysteresis_utterances=2)
    d = LanguageDetector(LanguageStateMachine(SUPPORTED_CODES), cfg, window=3)
    d.observe(det("ja", 0.97), 2.0)
    d.observe(det("ja", 0.97), 2.0)
    assert d.active_language == "ja"
    for _ in range(6):
        d.observe(det("pt", 0.98), 2.0)
    assert d.active_language == "pt"


def test_manual_override_ignores_detection(d):
    d.machine.set_manual("ar")
    assert d.machine.mode is LanguageMode.MANUAL
    for _ in range(5):
        d.observe(det("ja", 0.99), 3.0)
    assert d.machine.state is LanguageState.MANUAL_OVERRIDE and d.active_language == "ar"


def test_reset_keeps_manual_override(d):
    d.machine.set_manual("hi")
    d.reset()
    assert d.active_language == "hi"


def test_reset_clears_auto_state(d):
    d.observe(det("ja", 0.97), 4.0)
    assert d.active_language == "ja"
    d.reset()
    assert d.machine.state is LanguageState.UNKNOWN and d.active_language is None


def test_switch_goes_through_detecting_and_is_observable():
    cfg = LanguageConfig(switch_hysteresis_utterances=2)
    m = LanguageStateMachine(SUPPORTED_CODES)
    seen = []
    m.add_listener(lambda o, n, r: seen.append((o.value, n.value)))
    d = LanguageDetector(m, cfg, window=3)
    for _ in range(2):
        d.observe(det("ja", 0.97), 2.0)
    for _ in range(6):
        d.observe(det("pt", 0.98), 2.0)
    assert ("confirmed", "detecting") in seen and d.active_language == "pt"
