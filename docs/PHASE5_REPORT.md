# Phase 5 — Offline AI Integration: verification report

Machine: MacBook Pro, Apple M2 Pro (12 cores), 16 GB, macOS 26.6.2, Python 3.12.16,
faster-whisper 1.2.1 / CTranslate2 4.8.5. Model: Whisper **Small**, INT8, CPU, beam 1, `task="translate"`.

## Results (actual)
| Check | Result |
|---|---|
| Full test suite | **158 passed, 1 xfailed** (the xfail is the known Arabic quality gap below) |
| Non-AI suite on Python 3.11 | 122 passed |
| ruff / mypy strict | clean / no issues (29 files) |
| Coverage | 94% overall (inference_worker 73%: crash/kill edge branches) |

## What "offline" means here, and how it was proven
- **Inference** makes no network calls. Every real-model test runs under a fixture that makes any
  `socket.connect` / `getaddrinfo` / `create_connection` raise, and the model loads and translates under it.
  The engine also sets `HF_HUB_OFFLINE=1` and loads from a verified local directory path, never a hub id.
- **Obtaining the weights** needs one explicit step. It was a one-time download of the pinned
  `Systran/faster-whisper-small` files (483 MB) with `--allow-network`; without that flag the installer
  and `ModelManager.install()` refuse. A machine with no internet can use `--from-dir <folder>`.
  The only code path that can reach the network is `ModelManager.install`.
- Not covered by the socket guard: the *worker subprocess* in `test_full_pipeline_through_worker_process`
  (the guard patches the parent process). It uses the same local-path loading and `HF_HUB_OFFLINE`.
  A machine-level test with the network physically disabled has not been done.

## Integrity
Revisions are pinned commits. `model.bin` is checked against a pinned SHA-256 (the downloaded file matched),
and all four files are hashed into a manifest at install. Every model load re-verifies (about 1 s for Small).
Tested: corrupt download rejected with nothing installed, bit-flip tampering caught, wrong revision caught.

## Measured performance (Small INT8, 5 clean synthetic utterances of 3.8–5.3 s)
| Threads | Load | Inference p50 | Max | Peak RSS |
|---|---|---|---|---|
| 4 | 1.1 s | 1286 ms | 1696 ms | 918 MB |
| 6 (default = cores/2) | 1.4 s | 1021 ms | 1266 ms | 865 MB |
| 8 | 1.2 s | 1054 ms | 1201 ms | 834 MB |

RTF per utterance 0.19–0.29 (target < 0.7). Inference time is nearly constant (~1 s) regardless of utterance
length because Whisper's encoder always processes a padded 30 s window, so short utterances have a *worse* RTF
and 9 s utterances a better one. Expect caption latency ≈ endpoint silence (0.5–0.7 s) + ~1 s inference.
These numbers are from clean TTS audio on one machine. They are not a general performance claim.

## Translation quality: what is and isn't known
Test speech is **synthetic** (macOS system voices): a smoke/regression test, **not** a quality evaluation.
| Lang | Detected | Output (Small) | Verdict |
|---|---|---|---|
| zh | zh 1.00 | "The weather is very good today, let's go to the park together." | correct |
| ja | ja 0.99 | "Today is a very good weather. Let's go to the park together." | correct, awkward |
| pt | pt 0.97 | "Today the weather is very good, let's go to the park." | correct |
| hi | hi 0.87 | "The weather is very good today. Let's go to the park." | correct |
| ar | ar 1.00 | "I am very happy today, let's go to the garden" | **meaning drift: "weather is beautiful" became "I am very happy"** |

The Arabic miss had the lowest `avg_logprob` (-0.88 vs -0.37 to -0.58), but a single sample proves nothing.
Medium (1.5 GB) is the obvious thing to compare; it has not been installed or tested. Real accuracy needs
human-reviewed recordings with reference translations for all five languages. Not done.

## Hallucination evidence (why VAD + validation are mandatory)
Raw Whisper emits text for non-speech: silence -> "The end", loud noise -> "Thank you for watching!",
quiet noise -> "I'm going to make a new one.", a tone -> "The End". All four had `no_speech_prob` 0.72–0.88,
so the default 0.6 threshold flags them (asserted in a test). Whisper's own internal skip filters are
disabled so these diagnostics reach our validator (Phase 7). Nothing is filtered or invented in the engine.

## Unsupported languages
English and French speech were detected and `translate_audio` raised `UnsupportedLanguageError`; they were
never mapped onto a supported language. Forcing an unsupported code is also rejected.

## Bugs found and fixed during this phase
1. `language_probability` was reported as 1.0 when the caller forced the language: a fabricated confidence.
   Now `None` unless detection actually ran (test added).
2. Switching a CONFIRMED language to another crashed (illegal CONFIRMED to CONFIRMED transition).
   Now routed through DETECTING (test added).
3. A Phase 1 isolation test depended on test order; it now runs in a fresh interpreter.

## Delivered
`ai/model_registry.py`, `ai/model_manager.py`, `ai/engine.py`, `ai/whisper_engine.py`, `ai/language_detector.py`,
`ai/inference_worker.py` (separate spawn process, bounded queue, stale-request skipping, hard kill),
`scripts/install_models.py`, `scripts/make_test_audio.py`, `scripts/benchmark.py`, tests (model manager,
language detector, worker, 5-language real-model suite).

## Not done / limits
- Verified on macOS only. The engine is plain CPU INT8 and should behave the same on Windows/Linux, but that is untested.
- `make_test_audio.py` needs macOS `say`; on other OSes supply WAVs in `tests/ai/data/`.
- Only Small is installed and tested. `base` and `medium` are registered and hash-pinned but unexercised; `large-v3` is not installable yet.
- Thread count default (cores/2) is a placeholder until Phase 10.
- No streaming, VAD or caption validation yet (Phases 6 and 7). The language detector's thresholds are untuned.
- Whisper is utterance-based, not token-streaming.
