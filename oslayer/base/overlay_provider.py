from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.config.models import OverlayConfig


@dataclass(frozen=True, slots=True)
class OverlayCapabilities:
    transparent: bool
    always_on_top: bool
    click_through: bool
    positionable: bool  # app can place window at absolute screen coordinates
    over_fullscreen_apps: bool
    notes: tuple[str, ...] = ()  # documented fallbacks/limitations for this OS/compositor


@dataclass(frozen=True, slots=True)
class OverlayApplyResult:
    applied: frozenset[str]
    unsupported: frozenset[str]
    messages: tuple[str, ...] = field(default_factory=tuple)


class OverlayProvider(ABC):
    """Applies OS-specific window behaviour to a UI window. `window` is opaque to the core
    (a QWidget in the UI layer), so this module has no Qt dependency."""

    @abstractmethod
    def capabilities(self) -> OverlayCapabilities: ...

    @abstractmethod
    def apply(self, window: Any, config: OverlayConfig) -> OverlayApplyResult:
        """Apply what the platform supports; report (never hide) what it cannot."""
