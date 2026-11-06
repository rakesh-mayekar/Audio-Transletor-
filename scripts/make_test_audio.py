"""Generate SYNTHETIC five-language smoke-test speech using macOS's built-in offline voices.

Synthetic TTS speech is a smoke test only. It does NOT measure real-world accuracy (accents,
noise, music, overlapping speakers). Real evaluation needs human-reviewed recordings
(see docs/PHASE5_REPORT.md). Works on macOS only; on other OSes supply your own WAV files.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# language -> (voice, text, English keywords that a correct translation should contain)
SAMPLES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "zh": ("Tingting", "今天天气很好，我们一起去公园散步吧。", ("weather", "park")),
    "ja": (
        "Kyoko",
        "今日はとても良い天気ですね。一緒に公園へ散歩に行きましょう。",
        ("weather", "park"),
    ),
    "pt": ("Luciana", "Hoje o tempo está muito bom. Vamos passear no parque.", ("weather", "park")),
    "ar": ("Majed", "الطقس جميل جدا اليوم. هيا نذهب إلى الحديقة.", ("weather", "park")),
    "hi": ("Lekha", "आज मौसम बहुत अच्छा है। चलो पार्क में टहलने चलते हैं।", ("weather", "park")),
}
# UNSUPPORTED languages, used to prove they are rejected rather than coerced
UNSUPPORTED: dict[str, tuple[str, str]] = {
    "en": ("Samantha", "The weather is very nice today. Let us walk to the park."),
    "fr": ("Thomas", "Il fait très beau aujourd'hui, allons au parc."),
}
OUT = Path(__file__).resolve().parents[1] / "tests" / "ai" / "data"


def make(code: str, out_dir: Path = OUT) -> Path:
    voice, text = (SAMPLES[code][:2]) if code in SAMPLES else UNSUPPORTED[code]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{code}.wav"
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "s.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(dest)],
            check=True,
        )
    return dest


if __name__ == "__main__":
    if sys.platform != "darwin":
        raise SystemExit("make_test_audio.py needs macOS 'say'; supply your own WAVs elsewhere")
    for c in [*SAMPLES, *UNSUPPORTED]:
        print(make(c))
