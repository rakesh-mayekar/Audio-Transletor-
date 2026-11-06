"""Bounded float32 ring buffer addressed by ABSOLUTE sample index (capture-clock aligned)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.errors import InvalidAudioError


class RingBuffer:
    """Keeps the most recent `capacity` samples. Older audio is dropped (and counted)."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise InvalidAudioError("ring buffer capacity must be positive")
        self._cap = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._end = 0  # absolute index one past the newest sample
        self.dropped_samples = 0

    @property
    def end(self) -> int:
        return self._end

    @property
    def start(self) -> int:
        """Absolute index of the oldest retained sample."""
        return max(0, self._end - self._cap)

    def write(self, samples: NDArray[np.float32]) -> None:
        n = samples.size
        if n == 0:
            return
        self.dropped_samples += max(0, (self._end - self.start) + n - self._cap)
        if n > self._cap:  # only the newest `cap` samples can survive
            samples = samples[-self._cap :]
        base = self._end + n - samples.size  # absolute index of samples[0]
        pos = base % self._cap
        first = min(samples.size, self._cap - pos)
        self._buf[pos : pos + first] = samples[:first]
        if first < samples.size:
            self._buf[: samples.size - first] = samples[first:]
        self._end += n

    def read(self, start: int, end: int) -> NDArray[np.float32]:
        """Copy of [start, end). Raises if any part was already overwritten or not yet written."""
        if start > end:
            raise InvalidAudioError(f"bad range {start}>{end}")
        if start < self.start:
            raise InvalidAudioError(
                f"audio [{start},{self.start}) already overwritten", hint="Increase buffer size."
            )
        if end > self._end:
            raise InvalidAudioError(f"audio past write head ({end}>{self._end})")
        n = end - start
        out = np.empty(n, dtype=np.float32)
        pos = start % self._cap
        first = min(n, self._cap - pos)
        out[:first] = self._buf[pos : pos + first]
        if first < n:
            out[first:] = self._buf[: n - first]
        return out
