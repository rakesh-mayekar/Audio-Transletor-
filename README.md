# Offline Live Translator

Captures system playback audio, detects the spoken language (Chinese, Japanese, Portuguese, Arabic, Hindi),
translates it to English with a local Whisper model, and shows subtitles. No cloud, no API calls, no telemetry.
Models are installed once with an explicit command; after that everything runs offline.

Start with `docs/FINAL_STATUS.md` (what is and is not verified). Per-phase detail: `docs/PHASE*_REPORT.md`.

    uv venv --python 3.12 && uv pip install -e ".[dev,ai]"
    python scripts/install_models.py --model small --allow-network    # one-time download
    python -m app --doctor
# audio-translation
# Audio-Transletor-
