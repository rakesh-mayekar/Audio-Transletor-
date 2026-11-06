import threading

import pytest

from core.config import SUPPORTED_CODES
from core.errors import ConfigError, InvalidStateTransition
from core.state import (
    APP_TRANSITIONS,
    CAPTION_TRANSITIONS,
    LANGUAGE_TRANSITIONS,
    ApplicationStateMachine,
    AppState,
    CaptionLifecycle,
    LanguageMode,
    LanguageState,
    LanguageStateMachine,
)
from core.types import CaptionState


def test_app_happy_path_and_pause_resume():
    m = ApplicationStateMachine()
    seen = []
    m.add_listener(lambda o, n, r: seen.append((o, n, r)))
    for s in (
        AppState.STARTING,
        AppState.RUNNING,
        AppState.PAUSED,
        AppState.RUNNING,
        AppState.STOPPING,
        AppState.IDLE,
    ):
        m.transition(s, "t")
    assert m.state is AppState.IDLE and len(seen) == 6


@pytest.mark.parametrize(
    "bad", [AppState.RUNNING, AppState.PAUSED, AppState.STOPPING, AppState.ERROR]
)
def test_app_illegal_from_idle(bad):
    m = ApplicationStateMachine()
    with pytest.raises(InvalidStateTransition):
        m.transition(bad)
    assert m.state is AppState.IDLE


def test_app_error_recovery_and_table_is_total():
    m = ApplicationStateMachine()
    m.transition(AppState.STARTING)
    m.transition(AppState.ERROR)
    m.transition(AppState.STARTING)
    assert set(APP_TRANSITIONS) == set(AppState)


def test_listener_failure_does_not_block_transition_or_other_listeners():
    m = ApplicationStateMachine()
    called = []
    m.add_listener(lambda *a: 1 / 0)
    m.add_listener(lambda o, n, r: called.append(n))
    m.transition(AppState.STARTING)
    assert m.state is AppState.STARTING and called == [AppState.STARTING]


def test_transition_thread_safety_only_one_winner():
    m = ApplicationStateMachine()
    results = []

    def go():
        try:
            m.transition(AppState.STARTING)
            results.append("ok")
        except InvalidStateTransition:
            results.append("fail")

    ts = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("ok") == 1 and results.count("fail") == 19


def test_language_flow_spec():
    m = LanguageStateMachine(SUPPORTED_CODES)
    assert m.state is LanguageState.UNKNOWN and m.mode is LanguageMode.AUTO
    m.begin_detection()
    m.confirm("ja")
    assert m.state is LanguageState.CONFIRMED and m.detected_language == "ja"


def test_language_low_confidence_and_unsupported_paths():
    m = LanguageStateMachine(SUPPORTED_CODES)
    m.begin_detection()
    m.mark_low_confidence()
    assert m.state is LanguageState.LOW_CONFIDENCE
    m.transition(LanguageState.DETECTING)
    m.mark_unsupported()
    assert m.state is LanguageState.UNSUPPORTED and m.detected_language is None


@pytest.mark.parametrize("start", list(LanguageState))
def test_manual_override_reachable_from_every_state(start):
    m = LanguageStateMachine(SUPPORTED_CODES)
    # walk to `start`
    path = {
        LanguageState.UNKNOWN: [],
        LanguageState.DETECTING: [LanguageState.DETECTING],
        LanguageState.CONFIRMED: [LanguageState.DETECTING, LanguageState.CONFIRMED],
        LanguageState.LOW_CONFIDENCE: [LanguageState.DETECTING, LanguageState.LOW_CONFIDENCE],
        LanguageState.UNSUPPORTED: [LanguageState.DETECTING, LanguageState.UNSUPPORTED],
        LanguageState.MANUAL_OVERRIDE: [LanguageState.MANUAL_OVERRIDE],
    }[start]
    for s in path:
        m.transition(s)
    m.set_manual("hi")
    assert m.state is LanguageState.MANUAL_OVERRIDE and m.mode is LanguageMode.MANUAL
    assert m.manual_language == "hi"
    m.clear_manual()
    assert m.state is LanguageState.DETECTING and m.manual_language is None


@pytest.mark.parametrize("lang", ["en", "fr", "", "ZH", "pt-BR"])
def test_unsupported_language_never_accepted(lang):
    m = LanguageStateMachine(SUPPORTED_CODES)
    with pytest.raises(ConfigError):
        m.set_manual(lang)
    m.begin_detection()
    with pytest.raises(ConfigError):
        m.confirm(lang)
    assert m.state is LanguageState.DETECTING


def test_cannot_confirm_without_detecting():
    m = LanguageStateMachine(SUPPORTED_CODES)
    with pytest.raises(InvalidStateTransition):
        m.confirm("ja")


def test_caption_final_only_via_validated():
    for src in CaptionState:
        allowed = CAPTION_TRANSITIONS[src]
        assert (CaptionState.FINAL in allowed) == (src is CaptionState.VALIDATED)
    c = CaptionLifecycle("c1")
    with pytest.raises(InvalidStateTransition):
        c.transition(CaptionState.FINAL)
    for s in (CaptionState.PROCESSING, CaptionState.CANDIDATE):
        c.transition(s)
    with pytest.raises(InvalidStateTransition):
        c.transition(CaptionState.FINAL)  # candidate -> final forbidden
    c.transition(CaptionState.VALIDATED)
    c.transition(CaptionState.FINAL)
    c.transition(CaptionState.EXPIRED)


def test_caption_terminal_states():
    assert CAPTION_TRANSITIONS[CaptionState.REJECTED] == frozenset()
    assert CAPTION_TRANSITIONS[CaptionState.EXPIRED] == frozenset()
    assert set(CAPTION_TRANSITIONS) == set(CaptionState)
    assert set(LANGUAGE_TRANSITIONS) == set(LanguageState)
