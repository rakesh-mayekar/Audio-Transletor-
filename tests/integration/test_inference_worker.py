import time

import numpy as np
import pytest

from ai.inference_worker import InferenceWorker, WorkerRequest
from core.errors import InferenceError, ModelError

FAKE = "tests.ai.fake_engine:create"


def req(i, language="ja", ttl=30.0, n=16000):
    return WorkerRequest(f"r{i}", np.zeros(n, np.float32), language, 1000.0 * i, time.time() + ttl)


def collect(w, n, budget=20.0):
    out, end = [], time.monotonic() + budget
    while len(out) < n and time.monotonic() < end:
        r = w.poll(0.2)
        if r:
            out.append(r)
    return out


@pytest.fixture
def worker():
    w = InferenceWorker(FAKE, {"delay_s": 0.0}, max_pending=4)
    w.start(timeout=60)
    yield w
    w.stop()


def test_roundtrip_in_separate_process_with_timestamp_offset(worker):
    import os

    assert worker._proc.pid != os.getpid()
    assert worker.submit(req(3))
    (r,) = collect(worker, 1)
    assert r.request_id == "r3" and r.error is None and not r.stale
    assert r.output.results[0].audio_start_ms == 3000.0  # absolute audio-clock offset preserved


def test_ordering_preserved(worker):
    for i in range(4):
        assert worker.submit(req(i))
    ids = [r.request_id for r in collect(worker, 4)]
    assert ids == ["r0", "r1", "r2", "r3"]


def test_stale_request_skipped_without_running_model(worker):
    assert worker.submit(req(0, ttl=-1.0))
    (r,) = collect(worker, 1)
    assert r.stale and r.output is None


def test_backpressure_returns_false_instead_of_growing():
    w = InferenceWorker(FAKE, {"delay_s": 0.5}, max_pending=2)
    w.start(timeout=60)
    try:
        accepted = [w.submit(req(i)) for i in range(12)]
        assert accepted.count(False) >= 8 and w.rejected_full >= 8
        assert accepted.count(True) <= 4
    finally:
        w.stop()


def test_engine_error_returned_as_typed_error_and_worker_survives(worker):
    assert worker.submit(req(0, language="boom"))
    assert worker.submit(req(1))
    a, b = collect(worker, 2)
    assert a.error and a.error["code"] == "INFERENCE_ERROR" and "kaput" in a.error["message"]
    assert b.error is None and b.output is not None


def test_load_failure_is_reported_not_hung():
    w = InferenceWorker("tests.ai.fake_engine:create_failing", {})
    with pytest.raises(ModelError, match="cannot load"):
        w.start(timeout=60)
    assert not w.alive


def test_stop_terminates_process_and_is_idempotent():
    w = InferenceWorker(FAKE, {"delay_s": 5.0}, max_pending=2)  # busy worker: forces kill path
    w.start(timeout=60)
    proc = w._proc
    w.submit(req(0))
    t0 = time.monotonic()
    w.stop(timeout=1.0)
    assert time.monotonic() - t0 < 10
    assert not proc.is_alive()
    w.stop()


def test_submit_after_stop_raises():
    w = InferenceWorker(FAKE, {})
    w.start(timeout=60)
    w.stop()
    with pytest.raises(InferenceError):
        w.submit(req(0))
