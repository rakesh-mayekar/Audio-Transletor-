"""Caption validation (spec Phase 7). Principle: a MISSING caption is better than a WRONG one.

All numeric thresholds come from ValidationConfig and are UNTUNED defaults (Whisper's own
heuristics). Each rejection carries a machine-readable reason so tuning can be data-driven.
Raw Whisper numbers are diagnostics, not calibrated confidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from core.config.models import ValidationConfig
from core.types import TranslationResult

# Phrases Whisper is known to emit on silence/music (training-data artefacts). Only treated as
# hallucination when the acoustic diagnostics are also unconvincing, so a real speaker saying
# "thank you" is not suppressed.
_HALLUCINATIONS = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thank you so much for watching",
        "please subscribe",
        "subtitles by the amaraorg community",
        "the end",
        "bye",
        "you",
        "so",
    }
)
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_WORD.findall(t))


@dataclass(frozen=True, slots=True)
class Verdict:
    accepted: bool
    text: str  # possibly trimmed (seam overlap removed); original text if rejected
    reasons: tuple[str, ...] = ()


def strip_seam_overlap(prev: str, new: str, min_words: int = 2, max_words: int = 12) -> str:
    """Remove from `new` a leading run of words that repeats the end of `prev` (forced-split
    seam). Needs >= min_words matching words, so a single repeated word is left alone."""
    p, n = prev.split(), new.split()
    pn = [normalize(w) for w in p]
    nn = [normalize(w) for w in n]
    for k in range(min(len(p), len(n), max_words), min_words - 1, -1):
        if pn[-k:] == nn[:k] and all(nn[:k]):
            return " ".join(n[k:])
    return new


class CaptionValidator:
    def __init__(self, cfg: ValidationConfig | None = None) -> None:
        self.cfg = cfg or ValidationConfig()
        self._last_text = ""
        self._last_norm = ""
        self._last_end_ms = -1.0

    def reset(self) -> None:
        self._last_text, self._last_norm, self._last_end_ms = "", "", -1.0

    def validate(
        self, r: TranslationResult, *, now_audio_ms: float | None = None, seam: bool = False
    ) -> Verdict:
        c, d = self.cfg, r.diagnostics
        text = r.translated_text.strip()
        reasons: list[str] = []
        if not text or not _WORD.search(text):
            return Verdict(False, text, ("empty",))
        if d.no_speech_prob is not None and d.no_speech_prob > c.no_speech_prob_max:
            reasons.append("no_speech")
        if d.avg_logprob is not None and d.avg_logprob < c.avg_logprob_min:
            reasons.append("low_logprob")
        if d.compression_ratio is not None and d.compression_ratio > c.compression_ratio_max:
            reasons.append("repetitive")
        if (
            d.language_probability is not None
            and d.language_probability < c.min_language_probability
        ):
            reasons.append("low_language_confidence")
        if self._repeats(text):
            reasons.append("repetition_loop")
        if normalize(text) in _HALLUCINATIONS and (
            (d.no_speech_prob or 0.0) >= 0.3 or (d.avg_logprob is not None and d.avg_logprob < -0.7)
        ):
            reasons.append("known_hallucination")
        if now_audio_ms is not None and now_audio_ms - r.audio_end_ms > c.max_staleness_ms:
            reasons.append("stale")
        if reasons:
            return Verdict(False, text, tuple(reasons))

        if seam and self._last_text:
            text = strip_seam_overlap(self._last_text, text)
            if not normalize(text):
                return Verdict(False, r.translated_text.strip(), ("duplicate",))
        norm = normalize(text)
        overlaps_previous = bool(self._last_norm) and r.audio_start_ms < self._last_end_ms + 1.0
        if (
            overlaps_previous
            and SequenceMatcher(None, self._last_norm, norm).ratio() >= c.duplicate_similarity
        ):
            return Verdict(False, text, ("duplicate",))
        self._last_text, self._last_norm, self._last_end_ms = text, norm, r.audio_end_ms
        return Verdict(True, text)

    @staticmethod
    def _repeats(text: str, limit: int = 4) -> bool:
        """Same word >limit times in a row, or the same 2-3 word phrase looping."""
        w = [normalize(x) for x in text.split()]
        run = 1
        for a, b in zip(w, w[1:], strict=False):
            run = run + 1 if a and a == b else 1
            if run > limit:
                return True
        for n in (2, 3):
            for i in range(len(w) - n * (limit + 1) + 1):
                g = w[i : i + n]
                if all(w[i + n * k : i + n * (k + 1)] == g for k in range(limit + 1)):
                    return True
        return False
