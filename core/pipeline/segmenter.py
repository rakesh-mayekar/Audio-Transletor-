"""Speech segmentation: frames -> VAD windows -> utterances (pre-roll, endpointing, forced split).

Time model: all positions are absolute sample indices counted from the first sample fed in;
`pts_ms` of the FIRST frame anchors them to the capture clock. A gap between frames (dropped
capture audio) is bridged with silence so indices never drift from real time, and is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from core.audio.ring_buffer import RingBuffer
from core.config.models import VadConfig
from core.types import SAMPLE_RATE, AudioFrame, ms_to_samples, samples_to_ms

WINDOW = 512  # Silero window: 32 ms @ 16 kHz


class VadProtocol(Protocol):
    def process(self, samples: NDArray[np.float32]) -> list[float]:
        """Speech probability per consecutive 512-sample window (stateful across calls)."""

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Utterance:
    utterance_id: str
    start_ms: float  # capture clock
    end_ms: float
    samples: NDArray[np.float32]
    speech_ms: float  # voiced time inside the utterance
    forced_split: bool  # ended by max length, not by silence
    overlap_ms: float  # leading audio repeated from the previous (forced-split) utterance

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class SpeechSegmenter:
    def __init__(self, vad: VadProtocol, cfg: VadConfig | None = None) -> None:
        self._vad = vad
        self._cfg = cfg or VadConfig()
        c = self._cfg
        self._min_speech = ms_to_samples(c.min_speech_ms)
        self._endpoint = ms_to_samples(c.silence_endpoint_ms)
        self._pre_roll = ms_to_samples(c.pre_roll_ms)
        self._max_len = ms_to_samples(c.max_utterance_s * 1000)
        self._overlap = ms_to_samples(c.overlap_ms)
        self._ring = RingBuffer(self._max_len + self._pre_roll + self._endpoint + SAMPLE_RATE * 4)
        self._anchor_ms: float | None = None
        self._pending = np.zeros(0, dtype=np.float32)  # < WINDOW samples awaiting VAD
        self._win_index = 0  # windows consumed
        self._in_speech = False
        self._start = 0  # sample idx where current utterance audio begins (incl. pre-roll)
        self._speech_samples = 0
        self._last_voiced_end = 0
        self._seq = 0
        self._carry_overlap = 0
        self.gap_ms_total = 0.0
        self.discarded_short = 0

    # ------------------------------------------------------------------ public
    def feed(self, frame: AudioFrame) -> list[Utterance]:
        if self._anchor_ms is None:
            self._anchor_ms = frame.pts_ms
        expected = self._anchor_ms + samples_to_ms(self._ring.end + self._pending.size)
        gap = frame.pts_ms - expected
        samples = frame.samples
        if gap > 1000.0 / SAMPLE_RATE * 2:  # capture lost audio: bridge with silence
            self.gap_ms_total += gap
            samples = np.concatenate([np.zeros(ms_to_samples(gap), np.float32), samples])
        return self._ingest(samples)

    def flush(self) -> list[Utterance]:
        """Close any open utterance (call on stop / language switch)."""
        out: list[Utterance] = []
        if self._in_speech:
            u = self._close(self._last_voiced_end, forced=False)
            if u:
                out.append(u)
        self._vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        return out

    # ------------------------------------------------------------------ internals
    def _ms(self, idx: int) -> float:
        assert self._anchor_ms is not None
        return self._anchor_ms + samples_to_ms(idx)

    def _ingest(self, samples: NDArray[np.float32]) -> list[Utterance]:
        data = np.concatenate([self._pending, samples]) if self._pending.size else samples
        n_win = data.size // WINDOW
        usable = data[: n_win * WINDOW]
        self._pending = data[n_win * WINDOW :].copy()
        if n_win == 0:
            return []
        self._ring.write(usable)
        probs = self._vad.process(usable)
        out: list[Utterance] = []
        for p in probs:
            w_start = self._win_index * WINDOW
            w_end = w_start + WINDOW
            self._win_index += 1
            out.extend(self._step(p, w_start, w_end))
        return out

    def _step(self, prob: float, w_start: int, w_end: int) -> list[Utterance]:
        voiced = prob >= self._cfg.threshold
        out: list[Utterance] = []
        if not self._in_speech:
            if voiced:
                self._in_speech = True
                self._start = max(self._ring.start, w_start - self._pre_roll - self._carry_overlap)
                self._speech_samples = WINDOW
                self._last_voiced_end = w_end
            return out
        if voiced:
            self._speech_samples += WINDOW
            self._last_voiced_end = w_end
        silent_for = w_end - self._last_voiced_end
        if silent_for >= self._endpoint:
            u = self._close(self._last_voiced_end, forced=False)
            if u:
                out.append(u)
            return out
        if w_end - self._start >= self._max_len:
            u = self._close(w_end, forced=True)
            if u:
                out.append(u)
        return out

    def _close(self, voiced_end: int, *, forced: bool) -> Utterance | None:
        end = min(self._ring.end, voiced_end + (0 if forced else self._pre_roll // 2))
        start, speech = self._start, self._speech_samples
        self._in_speech = False
        self._speech_samples = 0
        self._carry_overlap = 0
        if speech < self._min_speech:
            self.discarded_short += 1
            return None
        self._seq += 1
        overlap_ms = samples_to_ms(self._carry_overlap)
        u = Utterance(
            utterance_id=f"utt-{self._seq:06d}",
            start_ms=self._ms(start),
            end_ms=self._ms(end),
            samples=self._ring.read(start, end),
            speech_ms=samples_to_ms(speech),
            forced_split=forced,
            overlap_ms=overlap_ms,
        )
        if forced:
            # continue immediately, repeating a short tail so words cut at the seam are not lost
            self._in_speech = True
            self._carry_overlap = self._overlap
            self._start = max(self._ring.start, end - self._overlap)
            self._speech_samples = self._overlap
            self._last_voiced_end = end
        return u
