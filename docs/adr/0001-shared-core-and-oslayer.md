# ADR 0001 — Shared core, `oslayer` adapters, no OS branching in core

- **Status:** accepted (Phase 1)
- `core/` contains zero OS-specific code. A test (`tests/unit/test_core_isolation.py`) fails if core imports OS/audio/GUI/ML libraries or reads `sys.platform`.
- The only OS boundaries are `oslayer.base.{AudioCaptureProvider, OverlayProvider, PermissionProvider}`. Implementations live in `oslayer/{windows,macos,linux}` and are imported lazily by `oslayer.selector`.
- The package is named `oslayer`, not `platform`, to avoid shadowing the stdlib module.
- A missing provider raises `ProviderUnavailableError`. There is never a microphone fallback, and the microphone device kind is rejected.
- Every provider must pass `tests/platform/conformance.py` (16 kHz / mono / float32, monotonic capture-clock timestamps, bounded buffers, typed errors, idempotent cleanup).
- Events use a bounded drop-oldest bus: producers never block, and loss is counted and visible.
- FINAL captions are reachable only from VALIDATED in the transition table.
