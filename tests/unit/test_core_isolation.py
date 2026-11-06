"""Enforces: the shared core contains no OS-specific code or imports."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORT_PREFIXES = (
    "oslayer",
    "winreg",
    "msvcrt",
    "ctypes.wintypes",
    "AppKit",
    "Foundation",
    "objc",
    "Quartz",
    "ScreenCaptureKit",
    "pyaudiowpatch",
    "pyaudio",
    "sounddevice",
    "soundcard",
    "pipewire",
    "pulsectl",
    "comtypes",
    "pycaw",
    "PySide6",
    "PyQt5",
    "PyQt6",
    "torch",
    "faster_whisper",
)


def _core_files():
    return [p for p in (ROOT / "core").rglob("*.py")]


def test_core_has_no_os_or_ui_imports():
    offenders = []
    for f in _core_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    offenders.append(f"{f.relative_to(ROOT)}: {n}")
    assert not offenders, offenders


def test_core_does_not_branch_on_os():
    offenders = []
    for f in _core_files():
        src = f.read_text()
        for needle in ("sys.platform", "platform.system", "os.name", "platform.uname"):
            if needle in src:
                offenders.append(f"{f.relative_to(ROOT)}: {needle}")
    assert not offenders, offenders


def test_importing_core_loads_no_oslayer_or_gui_modules():
    """Run in a fresh interpreter so other tests' imports cannot affect the result."""
    code = (
        "import sys, importlib\n"
        "for m in ('core.config','core.events','core.state','core.logging','core.errors','core.types'):\n"
        "    importlib.import_module(m)\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'PySide6','torch','faster_whisper','ctranslate2','oslayer','ai'}]\n"
        "print('LOADED', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
