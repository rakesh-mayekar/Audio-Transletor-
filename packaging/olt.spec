# PyInstaller spec for the offline translator.
# STATUS: UNTESTED - PyInstaller was not installed when this was written and no build was run.
# Build (on the TARGET OS; PyInstaller does not cross-compile):
#     pip install pyinstaller && pyinstaller packaging/olt.spec
# Models are NOT bundled (480 MB+, user-selected): install with scripts/install_models.py --from-dir.
# macOS: first run scripts/build_macos_helper.sh so the helper is collected below.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent
datas = collect_data_files("faster_whisper")            # includes the bundled Silero VAD onnx
datas += collect_data_files("certifi")
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
helper = root / "oslayer" / "macos" / "olt-audio-tap"
if sys.platform == "darwin" and helper.exists():
    binaries.append((str(helper), "oslayer/macos"))

a = Analysis(
    [str(root / "app" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["ai.whisper_engine", "ai.vad", "oslayer.linux.audio_capture",
                   "oslayer.macos.audio_capture", "oslayer.windows.audio_capture",
                   "oslayer.windows.wasapi_helper"],
    excludes=["tkinter", "matplotlib"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="offline-live-translator", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="offline-live-translator")
