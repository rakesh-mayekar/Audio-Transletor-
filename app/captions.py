"""Caption presentation logic (no Qt): what is on screen now, for how long, and history.

The UI polls `visible()` / receives `changed` callbacks; timing is driven by an injected
monotonic clock so behaviour is deterministic in tests.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from core.config.models import OverlayConfig
from core.events import EventBus
from core.events.types import CaptionDisplayed, CaptionExpired, CaptionValidated

MAX_VISIBLE = 2  # two stacked captions at most; older ones are pushed out, not queued


@dataclass(frozen=True, slots=True)
class ShownCaption:
    caption_id: str
    text: str
    start_ms: float
    end_ms: float
    shown_at: float
    expires_at: float


def display_seconds(text: str, base_s: float) -> float:
    """Longer captions stay up longer (≈ 3 words/second reading speed, never below base_s)."""
    return max(base_s, len(text.split()) / 3.0)


class CaptionPresenter:
    def __init__(
        self,
        cfg: OverlayConfig,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        keep_history: bool = True,
    ) -> None:
        self.cfg = cfg
        self._bus = bus
        self._clock = clock
        self._visible: list[ShownCaption] = []
        self._history: deque[ShownCaption] = deque(maxlen=max(1, cfg.history_size))
        self._keep_history = keep_history and cfg.history_size > 0
        self._latest_end_ms = -1.0
        self.on_change: Callable[[], None] | None = None
        self.skipped_out_of_order = 0

    def show(self, e: CaptionValidated) -> bool:
        """Display a validated caption. Returns False if it is older than what is on screen."""
        if e.end_ms <= self._latest_end_ms:
            self.skipped_out_of_order += 1
            return False
        now = self._clock()
        cap = ShownCaption(
            e.caption_id,
            e.text,
            e.start_ms,
            e.end_ms,
            now,
            now + display_seconds(e.text, self.cfg.caption_timeout_s),
        )
        self._latest_end_ms = e.end_ms
        self._visible.append(cap)
        while len(self._visible) > MAX_VISIBLE:
            self._expire(self._visible.pop(0))
        if self._keep_history:
            self._history.append(cap)
        if self._bus:
            self._bus.publish(CaptionDisplayed(caption_id=cap.caption_id))
        self._changed()
        return True

    def tick(self) -> bool:
        """Remove expired captions; True if the screen content changed."""
        now = self._clock()
        gone = [c for c in self._visible if c.expires_at <= now]
        for c in gone:
            self._visible.remove(c)
            self._expire(c)
        if gone:
            self._changed()
        return bool(gone)

    def visible(self) -> list[ShownCaption]:
        return list(self._visible)

    def text(self) -> str:
        return "\n".join(c.text for c in self._visible)

    def history(self) -> list[ShownCaption]:
        return list(self._history)

    def clear(self) -> None:
        for c in self._visible:
            self._expire(c)
        self._visible.clear()
        self._changed()

    def _expire(self, c: ShownCaption) -> None:
        if self._bus:
            self._bus.publish(CaptionExpired(caption_id=c.caption_id))

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()
