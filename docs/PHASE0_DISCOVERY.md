# Phase 0 — Discovery & Architecture

Status: **draft for approval**. No implementation code exists. Items marked **[VERIFY]** come from engineering knowledge, not from a test on this machine, and must be confirmed in the named phase before anyone relies on them.

## 1. Development environment (measured 2026-11-06)

| Item | Value |
|---|---|
| OS | macOS 26.6.2 (build 25G83), Darwin 25.6.0 |
| Architecture | arm64 (Apple Silicon) |
| CPU | Apple M2 Pro, 12 cores (physical = logical = 12) |
| RAM | 16 GiB |
| GPU | Apple M2 Pro integrated, 19 cores, Metal 4 |
| Disk | 196 GiB free of 926 GiB |
| Python | **3.9.6 only** (/usr/bin/python3, Apple system Python). No 3.11/3.12, no uv/conda/brew |
| Swift | 6.2.4 (Command Line Tools). **Full Xcode is not installed** |
| Other tools | git, node present. **ffmpeg, cmake, brew absent** |
| Audio | Built-in output "MacBook Pro Speakers" (48 kHz, 2 ch) is the default. Built-in mic is the default input |
| Desktop | macOS Aqua (no X11/Wayland/PipeWire) |
| ML packages | None installed (no PySide6, faster-whisper, numpy, onnxruntime, …) |
| Repo | Not a git repository; working directory is empty |

### Consequences
1. **Only macOS can be tested here.** Windows and Linux providers cannot be validated on this machine. Per rule 14, neither OS may be called "supported" until tested on real hardware or VMs (see R1).
2. **Python must be upgraded** before Phase 1. Target Python 3.11 or 3.12 in a venv (installed via python.org installer, or uv). System Python 3.9 is end-of-life and is not a target.
3. **Native macOS capture is buildable.** SwiftPM works with the Command Line Tools and the macOS SDK includes ScreenCaptureKit **[VERIFY: compile a hello-world SCStream helper in Phase 3]**. Code signing and notarization will need a full Xcode and an Apple Developer account.
4. **faster-whisper/CTranslate2 will run CPU-only on this Mac.** CTranslate2 has no Metal backend, so the M2's GPU is unused **[VERIFY by benchmark]**. This is a real tension with the "Small INT8 on CPU" default and is raised as Decision D1.

## 2. Architecture

```
 Platform layer (per OS)                     Shared core (pure Python, OS-agnostic)
┌───────────────────────────┐   normalized   ┌──────────────────────────────────────────────────────┐
│ WindowsAudioCaptureProvider│  16k mono f32  │ CaptureManager → Resampler/Normalizer → RingBuffer   │
│  WASAPI loopback           │ ─────────────► │   → SileroVAD → UtteranceManager ──┐                 │
│ MacOSAudioCaptureProvider  │  timestamped   │                                    ▼                 │
│  Swift helper + SCKit      │  PCM frames    │   LanguageStateMachine ◄── InferenceWorker (1 proc)  │
│ LinuxAudioCaptureProvider  │                │   (UNKNOWN…MANUAL)        faster-whisper, translate  │
│  PipeWire                  │                │                                    │                 │
└───────────────────────────┘                │   Validator → SegmentMatcher → Synchronizer          │
 Overlay/Permission adapters (per OS)         │                                    │ typed events    │
┌───────────────────────────┐                └────────────────────────────────────┼─────────────────┘
│ OverlayProvider, Permission│ ◄──────────────────────── Qt signals ───────────────┘
│ Provider, global shortcuts │          PySide6 UI: main window + overlay (main thread only)
└───────────────────────────┘
```

### Execution contexts
| Context | Kind | Notes |
|---|---|---|
| Qt main | thread | UI and overlay only |
| Audio capture | thread (Win/Linux) or child process (macOS Swift helper) | Writes into a bounded ring buffer; never blocks |
| Audio processing | thread | Resample, VAD, segmentation. numpy and onnxruntime release the GIL |
| Inference | **separate process** | Persistent model, one instance. Isolates GIL and crashes; allows clean kill on shutdown |
| Caption worker | thread | Validation, dedup, ordering |

All queues are bounded. Overflow policy for inference backlog: drop the *oldest unprocessed* utterance, record a `DroppedAudio` diagnostic event with its timestamps, and never display a stale caption. The staleness threshold is configurable.

### Key design decisions
- **A1.** `AudioCaptureProvider` is the only OS-specific audio boundary. It emits `AudioFrame(pts_ms, samples: float32[n], sample_rate=16000)`. Conversion to 16 kHz mono happens either in the provider (native resamplers are cheap) or in the shared `Resampler`. Either way, the contract is enforced by a conformance test run against every provider.
- **A2.** Timestamps come from the capture clock (sample counters anchored to a monotonic clock), never from processing completion time.
- **A3.** Typed events are frozen dataclasses on an in-process bus. Pydantic is used for config and settings (schema-versioned), not for hot-path audio events.
- **A4.** Language state machine (UNKNOWN, DETECTING, CONFIRMED, LOW_CONFIDENCE, UNSUPPORTED, MANUAL_OVERRIDE) is driven by Whisper's language probabilities over ≥ N seconds of speech. A detected language outside {zh, ja, pt, ar, hi} goes to UNSUPPORTED. In manual mode we pass `language=` explicitly. Unsupported speech is never forced into a supported language.
- **A5.** Engine call is `task="translate"`, `beam_size=1`, with the segment generator consumed inside the inference worker. Per-segment `no_speech_prob`, `avg_logprob`, `compression_ratio` and `language_probability` are carried as diagnostics, not as confidence.
- **A6.** Caption lifecycle: CAPTURED → PROCESSING → CANDIDATE → VALIDATED → FINAL, with REJECTED and EXPIRED as terminal states. FINAL requires passing validation; there is no path that publishes raw model text.
- **A7.** Utterance-based, not token streaming. The UI and docs must say so. For continuous speech the cap is 8–10 s with a small timestamped overlap and dedup by time range and text similarity, not text alone.

### Platform capture approach (to be confirmed in each phase)
| OS | Primary approach | Alternatives considered | Phase |
|---|---|---|---|
| Windows | WASAPI loopback on the default render endpoint via a maintained binding (candidate: PyAudioWPatch; or `soundcard`; or a small native shim) **[VERIFY maintenance status and Python 3.12 wheels]** | Process-loopback (per-app, Win10 2004+) as a later option | 2 |
| macOS | **Swift helper** using ScreenCaptureKit audio capture (`SCStreamConfiguration.capturesAudio`, macOS 13+) streaming PCM to Python over a pipe/socket. Needs Screen Recording permission | Core Audio process taps (macOS 14.2+) use a narrower "system audio" permission and may be the better UX **[VERIFY: both; pick by experiment]**. PyObjC bridging is rejected: callback threading and packaging are fragile | 3 |
| Linux | PipeWire. Resolve the default sink's monitor dynamically (pw-dump / `wpctl`), then stream via `pw-cat --record` or libpipewire bindings **[VERIFY which is more robust]** | PulseAudio compat layer (`parec`) as the fallback | 4 |

No provider may fall back to microphone capture. Failure produces a typed error with actionable instructions.

## 3. Dependency matrix

| Component | Candidate | License **[VERIFY before distribution]** | Notes |
|---|---|---|---|
| UI | PySide6 (Qt 6) | LGPLv3 / commercial | Dynamic linking required for LGPL compliance |
| ASR+translate | faster-whisper + CTranslate2 | MIT / MIT | CPU INT8 on all three OSes |
| Model weights | Systran/faster-whisper-small (CT2 conversion of OpenAI Whisper) | MIT (Whisper weights) | Explicit download only; checksum-pinned |
| VAD | Silero VAD, ONNX via onnxruntime | MIT | Avoid the `torch` dependency (large) |
| Numerics | numpy, scipy or soxr/resampy for resampling | BSD / LGPL(soxr: LGPL-2.1) **[VERIFY]** | Pick a permissive resampler or `scipy.signal.resample_poly` |
| Config | pydantic v2 | MIT | |
| Storage | sqlite3 (stdlib) | PSF | |
| System stats | psutil | BSD | |
| Packaging | PyInstaller (Win/macOS/Linux), Inno Setup or WiX, dmg + codesign + notarytool, AppImage | various | Phase 11 |
| Test | pytest, pytest-qt | MIT | |

Model sizes (approx., INT8): Small ≈ 250 MB on disk **[VERIFY]**; Medium ≈ 770 MB. Confirm the RAM footprint by measurement.

## 4. OS compatibility matrix (target vs. tested)

"Tested" is the only column that permits the word "supported".

| OS | Capture API | Min version (target) | Overlay notes | Tested today |
|---|---|---|---|---|
| Windows 10/11 x64 | WASAPI loopback | Win 10 | Topmost + `WS_EX_TRANSPARENT` click-through feasible | **No** (no Windows host) |
| macOS Apple Silicon | ScreenCaptureKit (or Core Audio tap) | macOS 13 (SCKit audio) / 14.2 (taps) **[VERIFY]** | Floating window level; over full-screen apps needs specific collection behavior | **Dev host only: macOS 26.6.2 / M2 Pro** |
| macOS Intel | same | same | same | **No** |
| Ubuntu 22.04/24.04 x86_64 | PipeWire | 22.04 ships PipeWire but PulseAudio is the default; 24.04 defaults to PipeWire | X11: topmost and click-through feasible. **Wayland: no global positioning or always-on-top for normal clients; overlay may need layer-shell or XWayland fallback** | **No** |

## 5. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Only macOS available; Windows/Linux unverifiable | Certain | High | Request Win and Ubuntu VMs/hardware (or CI runners with loopback devices) before Phases 2 and 4. Until then, code ships labelled "untested" |
| R2 | Whisper **Small translate quality is weak for ja/ar/hi** and may hallucinate on music/noise | High | High | Build the five-language evaluation set early (Phase 5). Benchmark Small vs Medium. Treat Small as a *starting default*, not a given |
| R3 | CTranslate2 has no Metal acceleration; Apple GPU unused | High | Medium | Benchmark CPU INT8 RTF on M2 Pro first. If it misses the RTF < 0.7 target, evaluate whisper.cpp (Metal/Core ML) or MLX behind the same `SpeechTranslationEngine` interface |
| R4 | Screen Recording permission needed for SCKit audio: poor UX, resets on OS updates | High | Medium | Evaluate Core Audio taps; build an explicit permission-guidance UI; never fall back to the mic |
| R5 | Wayland blocks always-on-top / click-through / positioning | High | Medium | Document fallback matrix; test X11 and GNOME/KDE Wayland; consider XWayland or layer-shell on wlroots/KDE |
| R6 | DRM-protected playback (e.g. some browser DRM paths) may be silent to loopback | Low–Med | Medium | Document known limitations after testing |
| R7 | Language auto-detect misfires on music, short clips, code-switching | High | Medium | Min speech duration, hysteresis state machine, UNSUPPORTED path, manual override |
| R8 | Whisper hallucination on silence/music (repetition loops, "thanks for watching") | High | High | VAD gating + layered validator + stale/dup rejection. Measure false captions per minute of non-speech |
| R9 | Latency: utterance-based design adds ≥ endpoint + inference time (likely 1.5–4 s) | High | Medium | State honestly in UI/docs. Measure P50/P95 |
| R10 | Python 3.9 on dev box; packaging with PySide6 + onnxruntime + CT2 is large and fiddly | Med | Medium | Install Python 3.12 now; spike a PyInstaller build in Phase 1 not Phase 11 |
| R11 | Notarization/signing needs full Xcode and a paid Apple Developer account | Med | Medium | Confirm with the user; document the unsigned fallback |
| R12 | Licenses (LGPL PySide6, model weights, resampler) | Low | High | License audit before Phase 11; ship third-party notices |
| R13 | Arabic dialect, Mandarin accent variation reduce accuracy | High | Medium | Document; evaluate on MSA and at least one dialect; no accuracy promises |
| R14 | Multi-process inference + audio threads: shutdown leaks | Med | Medium | Explicit lifecycle tests; kill-on-exit watchdog; 30-minute soak test |

## 6. Project structure

As in the specification (§13/Part 11), with two additions:
- `platform/macos/helper/` — SwiftPM package for the ScreenCaptureKit helper.
- `docs/adr/` — one short Architecture Decision Record per decision above.

Note the Python package name `platform/` shadows the stdlib `platform` module. **Recommendation: rename it to `oslayer/` (or `platforms/`)**. This is Decision D3.

## 7. Roadmap

| Phase | Deliverable | Machine needed | Exit gate |
|---|---|---|---|
| 0 | This document | Mac | Your approval |
| 1 | Shared core + abstraction interfaces + unit tests + PyInstaller spike | Mac | Core runs with a fake provider; tests green |
| 5 (pull forward, see D2) | Model manager + Whisper translate offline, 5-language eval set + Small/Medium benchmark | Mac | Measured RTF and quality per language |
| 3 | macOS capture (Swift helper) | Mac | Real playback captured to WAV, no mic |
| 6–8 | Streaming engine, validation, UI/overlay | Mac | Captions over a video in another app |
| 2, 4 | Windows, Linux capture | **Windows PC, Ubuntu** | WAV from real playback |
| 9–12 | Integration, performance, packaging, release validation | all three | Per spec acceptance criteria |

## 8. Decisions needed from you

- **D1 — Engine on Apple Silicon.** Keep faster-whisper (CPU INT8, matches your spec, uniform across OSes), or add whisper.cpp/MLX later *only if* the benchmark misses the RTF target? **Recommendation: keep faster-whisper behind the engine interface; decide from measured numbers.**
- **D2 — Phase order.** Your spec orders capture before AI. Since this Mac can only test macOS, I recommend Phase 0 → 1 → 5 (benchmark and quality, the biggest unknown) → 3 → rest on the Mac, with Windows and Linux phases when machines are available. OK?
- **D3 — Rename `platform/` to `oslayer/`** to avoid the stdlib clash. OK?
- **D4 — Windows and Linux test access.** Do you have physical machines or VMs, or should those phases be delivered as untested code?
- **D5 — Python.** May I install Python 3.12 (python.org installer or `uv`)? It needs a network download.

## 9. What Phase 0 did *not* do
No network lookups were made, so every **[VERIFY]** item is unconfirmed. No packages were installed, no audio was captured, and no model was downloaded. The Windows and Linux rows reflect design intent only.
