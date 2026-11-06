# Phase 1 — Shared Core: verification report

Run on macOS 26.6.2 / arm64 / Python 3.12.16 (and 3.11) on 2026-11-06.

## Results (actual)
| Check | Result |
|---|---|
| `pytest` (3.12) | **96 passed**, 5 consecutive runs, no flakes |
| `pytest` (3.11) | 96 passed |
| `mypy` strict (22 source files) | no issues |
| `ruff check` | all checks passed |
| Coverage (core + oslayer) | 97% (904 stmts, ~30 missed: bus close/edge branches, logging console branch, fake-provider edge branches) |
| Thread-leak guard (autouse) | every test must end with no bus/capture threads alive; verified that it fails on a deliberate leak |

## Delivered
- `core/errors`: typed hierarchy with a code + hint
- `core/types.py`: validated `AudioFrame`, `TranslationResult` and diagnostics contracts
- `core/config`: Pydantic config, exactly 5 languages, schema versioning, migration chain, atomic save, explicit corrupt-file recovery, per-OS dirs via platformdirs
- `core/events`: 13 typed immutable events + bounded, thread-safe bus
- `core/state`: application, language and caption state machines
- `core/logging`: rotation and an audio-redaction filter
- `oslayer/base`: audio/overlay/permission interfaces; `selector` (lazy, no fallback); `fake.py` test provider
- Tests: unit, integration (provider → bus → subscriber, device loss → error state), conformance suite, core-isolation checks
- `.github/workflows/ci.yml`: ubuntu/windows/macos × 3.11/3.12 matrix (**written, not yet executed**: the project is not on GitHub)

## Bugs found and fixed during verification
1. Unversioned settings had no 0→1 migration, so valid configs failed. Added the migration plus a regression test that validation errors are not masked by migration errors.
2. Logging-test handler leak (test bug).

## Not done / honest limits
- **Windows and Linux were not run.** Only macOS executed these tests; the CI matrix will exercise the others once the repo is pushed. Core isolation is enforced statically, but cross-OS behaviour is unproven until CI runs.
- No real audio provider exists yet (Phases 2–4). `create_audio_provider` raises `ProviderUnavailableError` on all three OSes by design.
- The PyInstaller packaging spike from Phase 0 was not done.
- Detection *evidence* logic for the language state machine is Phase 5; Phase 1 only guarantees legal transitions and rejection of unsupported languages.
- Whisper-derived validation thresholds in config are untuned defaults.
