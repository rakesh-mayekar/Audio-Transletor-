"""Language state driver. Turns per-utterance detections into a stable language state.

Rules (all thresholds from LanguageConfig/ValidationConfig, none hardcoded universal truths):
  * No decision until enough *speech seconds* are accumulated (no tiny-fragment decisions).
  * Votes are weighted by probability x speech seconds.
  * An unsupported winner -> UNSUPPORTED (never mapped onto a supported language).
  * Weak or split evidence -> LOW_CONFIDENCE.
  * Switching an already CONFIRMED language needs `switch_hysteresis_utterances` consecutive
    agreeing observations (one odd word cannot flip the language).
  * MANUAL_OVERRIDE ignores detection entirely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ai.engine import LanguageDetection
from core.config.languages import SUPPORTED_CODES
from core.config.models import LanguageConfig
from core.state import LanguageState, LanguageStateMachine


@dataclass(frozen=True, slots=True)
class Observation:
    language: str
    probability: float
    speech_s: float


class LanguageDetector:
    def __init__(
        self,
        machine: LanguageStateMachine,
        config: LanguageConfig,
        min_probability: float = 0.5,
        min_share: float = 0.6,
        window: int = 8,
    ) -> None:
        self.machine = machine
        self.config = config
        self.min_probability = min_probability
        self.min_share = min_share
        self._obs: deque[Observation] = deque(maxlen=window)
        self._pending_switch: tuple[str, int] | None = None
        self.last_detection: LanguageDetection | None = None  # diagnostics for the UI

    def reset(self) -> None:
        self._obs.clear()
        self._pending_switch = None
        if self.machine.state is not LanguageState.MANUAL_OVERRIDE:
            self.machine.transition(LanguageState.UNKNOWN, "reset")

    @property
    def active_language(self) -> str | None:
        """Language to pass to the engine: manual choice, confirmed language, else None."""
        m = self.machine
        if m.state is LanguageState.MANUAL_OVERRIDE:
            return m.manual_language
        return m.detected_language if m.state is LanguageState.CONFIRMED else None

    def observe(self, detection: LanguageDetection, speech_s: float) -> LanguageState:
        self.last_detection = detection
        m = self.machine
        if m.state is LanguageState.MANUAL_OVERRIDE:
            return m.state
        if m.state is LanguageState.UNKNOWN:
            m.begin_detection()
        self._obs.append(Observation(detection.language, detection.probability, speech_s))

        total = sum(o.speech_s for o in self._obs)
        if total < self.config.min_detection_speech_s:
            return m.state  # not enough evidence yet; stay as-is

        votes: dict[str, float] = {}
        for o in self._obs:
            votes[o.language] = votes.get(o.language, 0.0) + o.probability * o.speech_s
        weight = sum(votes.values())
        top_lang, top_vote = max(votes.items(), key=lambda kv: kv[1])
        share = top_vote / weight if weight else 0.0
        top_obs = [o for o in self._obs if o.language == top_lang]
        mean_p = sum(o.probability for o in top_obs) / len(top_obs)

        if share < self.min_share or mean_p < self.min_probability:
            self._settle(LanguageState.LOW_CONFIDENCE)
            return m.state
        if top_lang not in SUPPORTED_CODES:
            self._settle(LanguageState.UNSUPPORTED)
            return m.state
        if m.state is LanguageState.CONFIRMED and m.detected_language != top_lang:
            n = (
                (self._pending_switch[1] + 1)
                if self._pending_switch and self._pending_switch[0] == top_lang
                else 1
            )
            self._pending_switch = (top_lang, n)
            if n < self.config.switch_hysteresis_utterances:
                return m.state
        self._pending_switch = None
        if m.state is LanguageState.CONFIRMED and m.detected_language == top_lang:
            return m.state
        if m.state is not LanguageState.DETECTING:  # sustained change / recovery re-evaluates first
            m.transition(LanguageState.DETECTING, "language change")
        m.confirm(top_lang)
        return m.state

    def _settle(self, target: LanguageState) -> None:
        m = self.machine
        if m.state is target:
            return
        if target is LanguageState.UNSUPPORTED:
            if m.state is not LanguageState.DETECTING:
                m.transition(LanguageState.DETECTING, "re-evaluating")
            m.mark_unsupported()
        else:
            if m.state is not LanguageState.DETECTING:
                m.transition(LanguageState.DETECTING, "re-evaluating")
            m.mark_low_confidence()
