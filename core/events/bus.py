"""In-process event bus with BOUNDED per-subscriber queues.

publish() never blocks the producer. If a subscriber is slow, its oldest queued event is
dropped and counted (`dropped`), so back-pressure is visible, memory is bounded, and a slow
consumer can never stall audio capture. Handler exceptions are logged and surfaced through
`on_error`; they never kill the dispatcher thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from core.events.types import Event

_log = logging.getLogger(__name__)

Handler = Callable[[Any], None]
ErrorHandler = Callable[[str, Event, Exception], None]


class Subscription:
    def __init__(
        self,
        name: str,
        types: tuple[type[Event], ...],
        maxsize: int,
        handler: Handler | None,
        on_error: ErrorHandler | None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.name = name
        self.types = types
        self.maxsize = maxsize
        self._handler = handler
        self._on_error = on_error
        self._queue: deque[Event] = deque()
        self._cond = threading.Condition()
        self._busy = False
        self._closed = False
        self.dropped = 0
        self.delivered = 0
        self._thread: threading.Thread | None = None
        if handler is not None:
            self._thread = threading.Thread(target=self._run, name=f"bus-{name}", daemon=True)
            self._thread.start()

    # -- producer side --------------------------------------------------------
    def accepts(self, event: Event) -> bool:
        return isinstance(event, self.types)

    def put(self, event: Event) -> None:
        with self._cond:
            if self._closed:
                return
            if len(self._queue) >= self.maxsize:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append(event)
            self._cond.notify_all()

    # -- consumer side --------------------------------------------------------
    def get(self, timeout: float | None = None) -> Event | None:
        """Pull mode (only for subscriptions created without a handler)."""
        if self._handler is not None:
            raise RuntimeError("get() is only valid for pull-mode subscriptions")
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(timeout)
            if self._queue:
                self.delivered += 1
                return self._queue.popleft()
            return None

    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._queue or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def close(self, join_timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(join_timeout)

    def _run(self) -> None:
        assert self._handler is not None
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue and self._closed:
                    return
                event = self._queue.popleft()
                self._busy = True
            try:
                self._handler(event)
                self.delivered += 1
            except Exception as exc:  # handler bugs must be visible, not fatal
                _log.exception("subscriber %s failed on %s", self.name, type(event).__name__)
                if self._on_error is not None:
                    try:
                        self._on_error(self.name, event, exc)
                    except Exception:
                        _log.exception("on_error handler for %s failed", self.name)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self._closed = False

    def subscribe(
        self,
        name: str,
        *types: type[Event],
        handler: Handler | None = None,
        maxsize: int = 256,
        on_error: ErrorHandler | None = None,
    ) -> Subscription:
        """Subscribe to event types (subclasses match). With `handler`, a dedicated thread
        delivers events in order; without it, use `Subscription.get()`."""
        sub = Subscription(name, types or (Event,), maxsize, handler, on_error)
        with self._lock:
            if self._closed:
                sub.close()
                raise RuntimeError("EventBus is closed")
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        sub.close()

    def publish(self, event: Event) -> int:
        """Non-blocking fan-out. Returns the number of subscribers that received the event."""
        with self._lock:
            subs = list(self._subs)
        n = 0
        for s in subs:
            if s.accepts(event):
                s.put(event)
                n += 1
        return n

    def wait_idle(self, timeout: float = 5.0) -> bool:
        with self._lock:
            subs = list(self._subs)
        deadline = time.monotonic() + timeout
        return all(s.wait_idle(max(0.0, deadline - time.monotonic())) for s in subs)

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                s.name: {"pending": s.pending(), "dropped": s.dropped, "delivered": s.delivered}
                for s in self._subs
            }

    def close(self) -> None:
        """Graceful shutdown: stop accepting, drain handler threads, join them."""
        with self._lock:
            self._closed = True
            subs, self._subs = self._subs, []
        for s in subs:
            s.close()
