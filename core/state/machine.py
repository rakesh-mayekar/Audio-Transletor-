"""Generic, thread-safe finite state machine with an explicit transition table."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Generic, TypeVar

from core.errors import InvalidStateTransition

S = TypeVar("S", bound=Enum)
TransitionListener = Callable[[S, S, str], None]  # (old, new, reason)

_log = logging.getLogger(__name__)


class StateMachine(Generic[S]):  # noqa: UP046
    def __init__(
        self,
        name: str,
        initial: S,
        transitions: Mapping[S, frozenset[S]],
    ) -> None:
        self._name = name
        self._state = initial
        self._transitions = transitions
        self._lock = threading.RLock()
        self._listeners: list[Callable[[S, S, str], None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> S:
        with self._lock:
            return self._state

    def can_transition(self, target: S) -> bool:
        with self._lock:
            return target in self._transitions.get(self._state, frozenset())

    def add_listener(self, listener: Callable[[S, S, str], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def transition(self, target: S, reason: str = "") -> S:
        """Move to `target` or raise InvalidStateTransition. Returns the previous state.

        Listeners run outside the lock, in order. A failing listener is logged and
        does not undo the transition or prevent other listeners from running.
        """
        with self._lock:
            old = self._state
            if target not in self._transitions.get(old, frozenset()):
                raise InvalidStateTransition(
                    f"{self._name}: {old.name} -> {target.name} is not allowed",
                    hint="Allowed: "
                    + ", ".join(sorted(s.name for s in self._transitions.get(old, ()))),
                    old=old.name,
                    new=target.name,
                )
            self._state = target
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(old, target, reason)
            except Exception:
                _log.exception("%s: state listener failed (%s -> %s)", self._name, old, target)
        return old
