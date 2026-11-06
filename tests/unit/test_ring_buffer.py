import numpy as np
import pytest

from core.audio.ring_buffer import RingBuffer
from core.errors import InvalidAudioError


def ramp(a, b):
    return np.arange(a, b, dtype=np.float32)


def test_read_back_simple():
    r = RingBuffer(100)
    r.write(ramp(0, 50))
    assert np.array_equal(r.read(10, 20), ramp(10, 20))
    assert r.end == 50 and r.start == 0 and r.dropped_samples == 0


def test_wraparound_and_drop_accounting():
    r = RingBuffer(100)
    r.write(ramp(0, 80))
    r.write(ramp(80, 130))  # overflows by 30
    assert r.dropped_samples == 30 and r.start == 30 and r.end == 130
    assert np.array_equal(r.read(30, 130), ramp(30, 130))
    assert np.array_equal(r.read(95, 110), ramp(95, 110))  # across the seam


def test_write_larger_than_capacity():
    r = RingBuffer(10)
    r.write(ramp(0, 5))
    r.write(ramp(5, 40))
    assert r.start == 30 and np.array_equal(r.read(30, 40), ramp(30, 40))
    assert r.dropped_samples == 30


def test_overwritten_and_future_reads_raise():
    r = RingBuffer(10)
    r.write(ramp(0, 25))
    with pytest.raises(InvalidAudioError):
        r.read(0, 5)
    with pytest.raises(InvalidAudioError):
        r.read(20, 30)
    with pytest.raises(InvalidAudioError):
        RingBuffer(0)


def test_many_random_writes_match_reference():
    rng = np.random.default_rng(1)
    r, ref = RingBuffer(257), np.zeros(0, np.float32)
    for _ in range(200):
        chunk = rng.standard_normal(int(rng.integers(1, 400))).astype(np.float32)
        r.write(chunk)
        ref = np.concatenate([ref, chunk])
        assert np.array_equal(r.read(r.start, r.end), ref[r.start : r.end])
