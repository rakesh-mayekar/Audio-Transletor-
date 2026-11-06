"""Supported source languages. Exactly five; anything else is UNSUPPORTED, never guessed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    code: str  # Whisper language code
    name: str
    note: str


SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
    "zh": LanguageInfo(
        "zh", "Chinese (Mandarin)", "Initial target is Mandarin; other varieties untested."
    ),
    "ja": LanguageInfo("ja", "Japanese", ""),
    "pt": LanguageInfo(
        "pt",
        "Portuguese (Brazilian preferred)",
        "Whisper has a single 'pt' code; Brazilian is a regional preference, not a separate model.",
    ),
    "ar": LanguageInfo(
        "ar",
        "Arabic",
        "Dialects vary widely; accuracy is not expected to be uniform across dialects.",
    ),
    "hi": LanguageInfo("hi", "Hindi", ""),
}
SUPPORTED_CODES: frozenset[str] = frozenset(SUPPORTED_LANGUAGES)
TARGET_LANGUAGE = "en"
