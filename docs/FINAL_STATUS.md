# Status after Phases 3, 4, 6, 7, 8, 10, 11, 12 (written 2026-11-06)

Everything here was built and checked on ONE machine: MacBook Pro M2 Pro, macOS, Python 3.12.
Nothing has been run on Windows or Linux. "Verified" below means a test ran and passed on that Mac.

## What works, and how it was checked
| Area | State | Evidence |
|---|---|---|
| Offline Whisper translate (Phase 5) | Verified | see PHASE5_REPORT.md; network-blocked tests |
| Ring buffer | Verified | unit tests incl. randomized wrap-around vs reference (found + fixed a real bug) |
| Streaming Silero VAD (bundled in faster-whisper, no download) | Verified | silence / white noise / 440 Hz tone -> zero utterances; 6 spoken samples -> 1 utterance each; chunk-size independent |
| Utterance segmenter (pre-roll, endpoint, force-split + overlap, gap bridging) | Verified | scripted-VAD unit tests, sample-exact timing across chunk sizes |
| Caption validator (no-speech, logprob, compression, repetition, hallucination phrases, duplicates, seam overlap, staleness) | Verified on synthetic data only | 13 unit tests; thresholds are UNTUNED Whisper defaults |
| Pipeline (frames -> VAD -> worker process -> validator -> events) | Verified | scripted-worker tests + real end-to-end (real VAD + real Whisper worker) |
| Language handling | Verified | auto-detect, hysteresis switch, French refused even while Japanese is pinned |
| CLI `python -m app` (`--doctor`, `--wav`, `--live`, `--list-devices`) | `--doctor`, `--wav` verified | subprocess tests against the real model |
| Audio resampling / downmix (any rate -> 16 kHz mono) | Verified | tone preserved, >40 dB alias rejection, chunking-independent |
| Process-based capture provider (shared by all 3 OS) | Verified with a scripted helper | conformance suite + permission / device-lost / crash / slow-consumer / shutdown tests |
| macOS provider + Swift ScreenCaptureKit helper | Helper COMPILES (arm64). Real capture NOT run | needs Screen Recording permission granted interactively |
| Linux provider (`parec`/`pactl`) | Logic verified against recorded `pactl` text. NOT run on Linux | needs a Linux machine |
| Windows provider (WASAPI loopback via PyAudioWPatch) | Logic verified with a fake PyAudio. NOT run on Windows | needs Windows + `pip install PyAudioWPatch` |
| Overlay: caption timing / history logic | Verified | unit tests |
| Overlay: PySide6 window | Written, byte-compiles, NEVER executed | PySide6 not installed |
| Overlay OS capabilities (Wayland limits etc.) | Documented behaviour, not measured | stated in `oslayer/overlay.py` |
| Packaging (PyInstaller spec) | Written, NEVER built | PyInstaller not installed |
| CI (3 OS x py3.11/3.12) | Written, NEVER executed | no repository / runner |

## Measured performance (M2 Pro, Whisper Small INT8, 4 languages x2, 1x real-time pacing)
| Setting | Captions | Speaker stops -> caption (min / median / p90 / max) |
|---|---|---|
| `verify_interval=1` (default: re-check pinned language every utterance) | 8 / 8 | 2.01 / 2.15 / 2.29 / 2.31 s |
| `verify_interval=3` | 8 / 8 | 1.42 / 1.80 / 2.31 / 2.31 s |
Latency = 0.6 s endpoint silence + inference. Inference ~0.8 s pinned, ~1.46 s with the language re-check (a second
encoder pass; faster-whisper's own auto-detect costs the same, so it cannot be avoided, only done less often).
Parent process peak RSS ~103 MB; worker ~865 MB (Phase 5). Only synthetic speech was used.

## Bugs found by the tests in this round (all fixed)
1. Ring buffer returned rotated data after a single write larger than its capacity.
2. After a language was confirmed, the pipeline stopped detecting: French was silently translated through the
   pinned Japanese setting. Now the pinned language is re-verified (configurable) and unsupported audio is refused.
3. Worker error details were stringified, losing the detected language; now primitives survive.
4. An edit of mine corrupted `ai/whisper_engine.py`; recovered exactly (the damage was reversible) and re-verified by the full suite.

## Known problems and limits (not hidden)
- **Arabic quality with Small** is poor on the one test sentence (meaning changed). Medium is untested.
- **Quality was never measured on real recordings.** All speech is macOS `say` synthetic voices.
- Validator thresholds are untuned; a real speaker saying "thank you" in a quiet moment could be suppressed.
- A `resource_tracker: leaked semaphore` warning appeared ONCE at exit of the latency script; 3 later runs
  did not reproduce it. Cause unknown.
- The worker subprocess is not under the socket-block test guard; the network was never physically disabled.
- Live macOS capture, and everything on Windows/Linux, is untested. macOS captures the whole system mix (no per-app).
- The overlay will not appear over exclusive-fullscreen games or native macOS full-screen Spaces.
- Tests were last run on Python 3.12 only; Python 3.11 was not re-run for the new code.

## Things that need downloading / installing (nothing was downloaded in this round)
| Item | Size (approx.) | Needed for |
|---|---|---|
| PySide6 (`pip install '.[gui]'`) | ~200 MB | the floating overlay (`--overlay`) |
| Whisper Medium (`scripts/install_models.py --model medium --allow-network`) | 1.5 GB | testing Arabic / better accuracy |
| PyAudioWPatch (`pip install '.[windows]'`) | small | Windows capture |
| `pulseaudio-utils` (apt) | small | Linux capture |
| Xcode command line tools | already present here | rebuild the macOS helper |
| PyInstaller | ~20 MB | building a distributable app |

## How to run
    python -m app --doctor --models-dir models          # what is installed / available
    python -m app --wav tests/ai/data/zh.wav --models-dir models --speed 4
    python -m app --live --models-dir models            # needs Screen Recording permission on macOS
    python -m app --live --overlay --models-dir models  # needs PySide6
Config option `language.verify_interval` (1 safest, N cheaper, 0 off) trades latency against language-change safety.
