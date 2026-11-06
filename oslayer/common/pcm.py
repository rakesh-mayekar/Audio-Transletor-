"""PCM helpers shared by every capture provider: channel downmix and streaming resampling."""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.typing import NDArray

from core.errors import InvalidAudioError


def to_mono(interleaved: NDArray[np.float32], channels: int) -> NDArray[np.float32]:
    """Average interleaved channels. Trailing partial frames must be removed by the caller."""
    if channels < 1:
        raise InvalidAudioError(f"invalid channel count {channels}")
    if channels == 1:
        return interleaved
    if interleaved.size % channels:
        raise InvalidAudioError("interleaved buffer is not a whole number of frames")
    out: NDArray[np.float32] = interleaved.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return out


class StreamingResampler:
    """Rational polyphase resampler (windowed-sinc, zero phase) with state across chunks.

    Output is identical regardless of how the input is chunked. Pure numpy, no extra deps.
    Group delay is removed (zero phase) at the cost of `lookahead` input samples of latency
    (~1 ms). Anti-alias cutoff sits at 90% of the lower Nyquist frequency.
    """

    def __init__(self, in_rate: int, out_rate: int, zero_crossings: int = 12) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise InvalidAudioError("sample rates must be positive")
        g = gcd(in_rate, out_rate)
        self.in_rate, self.out_rate = in_rate, out_rate
        self._L, self._M = out_rate // g, in_rate // g  # upsample, downsample
        if self._L == self._M:
            self._h: NDArray[np.float64] | None = None
        else:
            cutoff = 0.9 * 0.5 / max(self._L, self._M)  # cycles per upsampled sample
            half = zero_crossings * max(self._L, self._M)
            n = np.arange(-half, half + 1)
            h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(2 * half + 1, 8.6)
            self._h = h * self._L / h.sum() * 1.0  # unity DC gain after zero stuffing
            self._c = half
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0  # absolute input index of _buf[0]
        self._n_out = 0

    def process(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._h is None:
            return x
        self._buf = np.concatenate([self._buf, x.astype(np.float64)])
        total = self._base + self._buf.size
        L, M, c, h = self._L, self._M, self._c, self._h
        n_max = (total * L - 1 - c) // M  # last n whose full filter support has arrived
        if n_max < self._n_out:
            return np.zeros(0, np.float32)
        n = np.arange(self._n_out, n_max + 1, dtype=np.int64)
        m = n * M
        t_count = (2 * c) // L + 1
        i_lo = -((-(m - c)) // L)  # ceil((m - c) / L)
        idx = i_lo[:, None] + np.arange(t_count)[None, :]
        d = m[:, None] - idx * L
        valid = (np.abs(d) <= c) & (idx >= 0)
        local = np.clip(idx - self._base, 0, self._buf.size - 1)
        taps = h[np.clip(d + c, 0, 2 * c)]
        y = (self._buf[local] * taps * valid).sum(axis=1)
        self._n_out = int(n_max) + 1
        # keep only input that later outputs can still touch
        keep_from = max(0, -((-(self._n_out * M - c)) // L))
        drop = keep_from - self._base
        if drop > 0:
            self._buf = self._buf[drop:]
            self._base += drop
        out: NDArray[np.float32] = y.astype(np.float32)
        return out
