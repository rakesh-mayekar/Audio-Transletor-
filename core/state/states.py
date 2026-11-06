"""Concrete state enums, transition tables and machines (application, language, caption)."""

from __future__ import annotations

from enum import StrEnum

from core.errors import ConfigError
from core.state.machine import StateMachine
from core.types import CaptionState


# ----------------------------------------------------------------- application
class AppState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


APP_TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    AppState.IDLE: frozenset({AppState.STARTING}),
    AppState.STARTING: frozenset({AppState.RUNNING, AppState.ERROR, AppState.STOPPING}),
    AppState.RUNNING: frozenset({AppState.PAUSED, AppState.STOPPING, AppState.ERROR}),
    AppState.PAUSED: frozenset({AppState.RUNNING, AppState.STOPPING, AppState.ERROR}),
    AppState.STOPPING: frozenset({AppState.IDLE, AppState.ERROR}),
    AppState.ERROR: frozenset({AppState.STARTING, AppState.IDLE}),
}


class ApplicationStateMachine(StateMachine[AppState]):
    def __init__(self) -> None:
        super().__init__("application", AppState.IDLE, APP_TRANSITIONS)


# -------------------------------------------------------------------- language
class LanguageState(StrEnum):
    UNKNOWN = "unknown"
    DETECTING = "detecting"
    CONFIRMED = "confirmed"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED = "unsupported"
    MANUAL_OVERRIDE = "manual_override"


class LanguageMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


_ALL_LANG = frozenset(LanguageState)

LANGUAGE_TRANSITIONS: dict[LanguageState, frozenset[LanguageState]] = {
    LanguageState.UNKNOWN: frozenset({LanguageState.DETECTING, LanguageState.MANUAL_OVERRIDE}),
    LanguageState.DETECTING: frozenset(
        {
            LanguageState.CONFIRMED,
            LanguageState.LOW_CONFIDENCE,
            LanguageState.UNSUPPORTED,
            LanguageState.UNKNOWN,
            LanguageState.MANUAL_OVERRIDE,
        }
    ),
    # sustained change / re-evaluation is allowed from any settled auto state
    LanguageState.CONFIRMED: frozenset(
        {LanguageState.DETECTING, LanguageState.UNKNOWN, LanguageState.MANUAL_OVERRIDE}
    ),
    LanguageState.LOW_CONFIDENCE: frozenset(
        {
            LanguageState.DETECTING,
            LanguageState.CONFIRMED,
            LanguageState.UNKNOWN,
            LanguageState.MANUAL_OVERRIDE,
        }
    ),
    LanguageState.UNSUPPORTED: frozenset(
        {LanguageState.DETECTING, LanguageState.UNKNOWN, LanguageState.MANUAL_OVERRIDE}
    ),
    LanguageState.MANUAL_OVERRIDE: frozenset(
        {LanguageState.UNKNOWN, LanguageState.DETECTING, LanguageState.MANUAL_OVERRIDE}
    ),
}


class LanguageStateMachine(StateMachine[LanguageState]):
    """Holds language state plus the AUTO/MANUAL mode and the manual language.

    Detection *evidence* logic arrives in Phase 5; this class only guarantees that
    states are legal, manual override works from any state, and unsupported
    languages can never be selected as a source.
    """

    def __init__(self, supported: frozenset[str]) -> None:
        super().__init__("language", LanguageState.UNKNOWN, LANGUAGE_TRANSITIONS)
        self._supported = supported
        self._manual_language: str | None = None
        self._detected_language: str | None = None

    @property
    def mode(self) -> LanguageMode:
        return (
            LanguageMode.MANUAL
            if self.state is LanguageState.MANUAL_OVERRIDE
            else LanguageMode.AUTO
        )

    @property
    def manual_language(self) -> str | None:
        return self._manual_language

    @property
    def detected_language(self) -> str | None:
        return self._detected_language

    def begin_detection(self) -> None:
        self.transition(LanguageState.DETECTING, "detection started")

    def set_manual(self, language: str) -> None:
        if language not in self._supported:
            raise ConfigError(
                f"Language '{language}' is not supported",
                hint="Supported: " + ", ".join(sorted(self._supported)),
            )
        self._manual_language = language
        self.transition(LanguageState.MANUAL_OVERRIDE, f"manual:{language}")

    def clear_manual(self) -> None:
        self._manual_language = None
        self.transition(LanguageState.DETECTING, "manual override cleared")

    def confirm(self, language: str) -> None:
        if language not in self._supported:
            raise ConfigError(f"Cannot confirm unsupported language '{language}'")
        self._detected_language = language
        self.transition(LanguageState.CONFIRMED, f"confirmed:{language}")

    def mark_unsupported(self) -> None:
        self._detected_language = None
        self.transition(LanguageState.UNSUPPORTED, "language not in supported set")

    def mark_low_confidence(self) -> None:
        self.transition(LanguageState.LOW_CONFIDENCE, "insufficient evidence")


# --------------------------------------------------------------------- caption
CAPTION_TRANSITIONS: dict[CaptionState, frozenset[CaptionState]] = {
    CaptionState.CAPTURED: frozenset(
        {CaptionState.PROCESSING, CaptionState.REJECTED, CaptionState.EXPIRED}
    ),
    CaptionState.PROCESSING: frozenset(
        {CaptionState.CANDIDATE, CaptionState.REJECTED, CaptionState.EXPIRED}
    ),
    CaptionState.CANDIDATE: frozenset(
        {CaptionState.VALIDATED, CaptionState.REJECTED, CaptionState.EXPIRED}
    ),
    # FINAL is reachable ONLY from VALIDATED: model text alone can never become final.
    CaptionState.VALIDATED: frozenset(
        {CaptionState.FINAL, CaptionState.REJECTED, CaptionState.EXPIRED}
    ),
    CaptionState.FINAL: frozenset({CaptionState.EXPIRED}),
    CaptionState.REJECTED: frozenset(),
    CaptionState.EXPIRED: frozenset(),
}


class CaptionLifecycle(StateMachine[CaptionState]):
    def __init__(self, caption_id: str) -> None:
        super().__init__(f"caption:{caption_id}", CaptionState.CAPTURED, CAPTION_TRANSITIONS)
