import threading
import time

import numpy as np

from core.errors import DeviceLostError
from core.events import (
    AudioCaptured,
    AudioDropped,
    CaptionValidated,
    ErrorOccurred,
    Event,
    EventBus,
    LanguageConfirmed,
)
from core.types import AudioFrame


def frame(i=0):
    return AudioFrame(i * 20.0, np.zeros(320, np.float32))


def test_events_are_immutable_and_typed():
    e = CaptionValidated(caption_id="1", text="hi", start_ms=0, end_ms=1)
    try:
        e.text = "x"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("event must be frozen")
    assert isinstance(e, Event) and e.emitted_at_ns > 0


def test_type_routing_and_subclass_subscription():
    bus = EventBus()
    a = bus.subscribe("a", AudioCaptured)
    allsub = bus.subscribe("all")
    bus.publish(AudioCaptured(frame=frame()))
    bus.publish(LanguageConfirmed(language="ja"))
    assert a.pending() == 1 and allsub.pending() == 2
    assert isinstance(a.get(0.1), AudioCaptured)
    bus.close()


def test_handler_receives_in_order():
    bus = EventBus()
    got = []
    bus.subscribe("h", AudioCaptured, handler=lambda e: got.append(e.frame.pts_ms))
    for i in range(50):
        bus.publish(AudioCaptured(frame=frame(i)))
    assert bus.wait_idle(5)
    assert got == [i * 20.0 for i in range(50)]
    bus.close()


def test_bounded_queue_drops_oldest_and_counts_without_blocking_producer():
    bus = EventBus()
    gate = threading.Event()
    got = []

    def slow(e):
        gate.wait(5)
        got.append(e.frame.pts_ms)

    sub = bus.subscribe("slow", AudioCaptured, handler=slow, maxsize=5)
    t0 = time.monotonic()
    for i in range(200):
        bus.publish(AudioCaptured(frame=frame(i)))
    assert time.monotonic() - t0 < 1.0  # producer never blocked
    assert sub.pending() <= 5 and sub.dropped >= 190
    gate.set()
    assert bus.wait_idle(5)
    assert got[-1] == 199 * 20.0  # newest survived
    assert len(got) <= 6
    bus.close()


def test_handler_exception_reported_and_does_not_kill_dispatcher():
    bus = EventBus()
    errors, ok = [], []

    def h(e):
        if e.language == "bad":
            raise ValueError("boom")
        ok.append(e.language)

    bus.subscribe(
        "h", LanguageConfirmed, handler=h, on_error=lambda n, e, x: errors.append((n, str(x)))
    )
    for lang in ("ja", "bad", "zh"):
        bus.publish(LanguageConfirmed(language=lang))
    assert bus.wait_idle(5)
    assert ok == ["ja", "zh"] and errors == [("h", "boom")]
    bus.close()


def test_close_joins_threads_and_rejects_new_subscribers():
    bus = EventBus()
    sub = bus.subscribe("h", handler=lambda e: None)
    bus.close()
    assert sub._thread is not None and not sub._thread.is_alive()
    try:
        bus.subscribe("late")
    except RuntimeError:
        pass
    else:
        raise AssertionError
    assert bus.publish(AudioDropped(start_ms=0, end_ms=1, reason="r", component="c")) == 0


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    sub = bus.subscribe("p", LanguageConfirmed)
    bus.unsubscribe(sub)
    assert bus.publish(LanguageConfirmed(language="ja")) == 0
    bus.close()


def test_concurrent_publishers_no_loss_within_capacity():
    bus = EventBus()
    count = [0]
    lock = threading.Lock()

    def h(e):
        with lock:
            count[0] += 1

    bus.subscribe("c", LanguageConfirmed, handler=h, maxsize=10_000)

    def pub():
        for _ in range(500):
            bus.publish(LanguageConfirmed(language="ja"))

    ts = [threading.Thread(target=pub) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert bus.wait_idle(10) and count[0] == 4000
    bus.close()


def test_error_event_carries_typed_error():
    e = ErrorOccurred(error=DeviceLostError("gone"), component="capture", recoverable=True)
    assert e.error.code == "DEVICE_LOST"
