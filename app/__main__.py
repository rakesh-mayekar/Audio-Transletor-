"""Command line entry:  python -m app --doctor | --list-devices | --wav FILE | --live [--overlay]"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from ai.model_manager import ModelManager
from app.runner import TranslatorRuntime
from core.config.languages import SUPPORTED_CODES
from core.config.loader import parse_config
from core.config.paths import models_dir as default_models_dir
from core.errors import TranslatorError
from core.events import EventBus
from core.events.types import (
    AudioDropped,
    CaptionRejected,
    CaptionValidated,
    ErrorOccurred,
    LanguageConfirmed,
)
from oslayer.selector import create_audio_provider, detect_os


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit("WAV must be 16 kHz, mono, 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def doctor(models: Path) -> int:
    ok = True
    print(f"OS: {detect_os().value}   Python: {sys.version.split()[0]}")
    for st in ModelManager(models).list_status():
        if st.installed:
            note = "verified" if st.verified else f"INSTALLED BUT FAILED CHECK: {st.error}"
            print(f"model {st.name:<7} installed, {note} ({st.size_bytes / 1e6:.0f} MB)")
        else:
            print(f"model {st.name:<7} not installed")
    print(f"models dir: {models}")
    print(f"PySide6 (overlay): {'yes' if importlib.util.find_spec('PySide6') else 'NOT installed'}")
    try:
        p = create_audio_provider()
        p.initialize()
        names = [d.name for d in p.list_devices()]
        print(f"audio capture: backend present ({p.platform_name}); devices: {names}")
        print("   (permission / real capture is NOT checked here; run --live to test)")
        p.cleanup()
    except TranslatorError as e:
        ok = False
        print(f"audio capture: UNAVAILABLE - {e}" + (f"\n   hint: {e.hint}" if e.hint else ""))
    return 0 if ok else 1


def attach_console(bus: EventBus) -> None:
    def show(e: object) -> None:
        if isinstance(e, CaptionValidated):
            print(f"[{e.start_ms / 1000:7.2f}s] {e.text}", flush=True)
        elif isinstance(e, CaptionRejected):
            print(f"   (rejected: {e.reason})", flush=True)
        elif isinstance(e, AudioDropped):
            print(
                f"   (dropped {e.start_ms / 1000:.1f}-{e.end_ms / 1000:.1f}s: {e.reason})",
                flush=True,
            )
        elif isinstance(e, LanguageConfirmed):
            print(f"   language: {e.language}", flush=True)
        elif isinstance(e, ErrorOccurred):
            print(f"ERROR [{e.component}] {e.error}  hint: {e.error.hint}", flush=True)

    bus.subscribe("console", handler=show)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app", description="Offline live audio translator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doctor", action="store_true")
    g.add_argument("--list-devices", action="store_true")
    g.add_argument("--wav", type=Path)
    g.add_argument("--live", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--language", choices=sorted(SUPPORTED_CODES), help="force source language")
    ap.add_argument("--device", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="--wav replay speed")
    ap.add_argument("--overlay", action="store_true", help="show the floating caption window")
    a = ap.parse_args(argv)
    models = a.models_dir or default_models_dir()

    if a.doctor:
        return doctor(models)
    if a.list_devices:
        try:
            p = create_audio_provider()
            p.initialize()
            for d in p.list_devices():
                print(f"{d.id}\t{d.name}{'  (default)' if d.is_default else ''}")
            p.cleanup()
            return 0
        except TranslatorError as e:
            print(e if not e.hint else f"{e}\nhint: {e.hint}", file=sys.stderr)
            return 1

    lang = {"auto_detect": False, "manual_source": a.language} if a.language else {}
    cfg = parse_config({"schema_version": 1, "language": lang})
    bus = EventBus()
    attach_console(bus)
    rt = TranslatorRuntime(cfg, models, bus)
    stop = threading.Event()
    try:
        rt.start()
        if a.wav:
            rt.run_samples(load_wav(a.wav), a.speed, stop)
        elif a.overlay:
            return _live_with_overlay(rt, cfg, bus, a.device, stop)
        else:
            provider = create_audio_provider()
            provider.initialize()
            provider.select_device(a.device)
            print("Capturing system audio. Ctrl+C to stop.", flush=True)
            rt.run_provider(provider, stop)
        return 0
    except KeyboardInterrupt:
        stop.set()
        return 0
    except TranslatorError as e:
        print(e if not e.hint else f"{e}\nhint: {e.hint}", file=sys.stderr)
        return 2
    finally:
        rt.stop()
        bus.wait_idle(2.0)
        bus.close()


def _live_with_overlay(rt: TranslatorRuntime, cfg, bus: EventBus, device, stop) -> int:  # type: ignore[no-untyped-def]
    if importlib.util.find_spec("PySide6") is None:
        print("The overlay needs PySide6:  pip install '.[gui]'", file=sys.stderr)
        return 3
    from PySide6.QtWidgets import QApplication

    from app.overlay_qt import OverlayWindow

    app = QApplication(sys.argv[:1])
    win = OverlayWindow(cfg.overlay, bus)
    provider = create_audio_provider()
    provider.initialize()
    provider.select_device(device)
    t = threading.Thread(target=rt.run_provider, args=(provider, stop), name="runtime", daemon=True)
    t.start()
    app.aboutToQuit.connect(stop.set)
    code = app.exec()
    stop.set()
    t.join(10)
    del win
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
