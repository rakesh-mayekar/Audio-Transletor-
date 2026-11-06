"""Windows WASAPI loopback helper (runs as a child process; see oslayer.common.process_capture).

Streams the default (or chosen) playback device as 16 kHz mono float32 LE to stdout.
Needs the optional `PyAudioWPatch` package (a PyAudio fork with WASAPI loopback).
Exit codes: 0 stop, 69 no loopback device / backend missing.
STATUS: logic verified with a fake PyAudio module on macOS; NEVER run against real WASAPI
(no Windows machine available) - see docs/PHASE3_4_REPORT.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
from typing import Any, BinaryIO

import numpy as np

from oslayer.common.pcm import StreamingResampler, to_mono

EXIT_UNAVAILABLE = 69


def loopback_devices(pa: Any) -> list[dict[str, Any]]:
    """All WASAPI loopback endpoints; the one mirroring the default output is flagged."""
    wasapi = pa.get_host_api_info_by_type(pa.paWASAPI)
    default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    out: list[dict[str, Any]] = []
    for d in pa.get_loopback_device_info_generator():
        out.append(
            {
                "id": str(d["index"]),
                "name": str(d["name"]),
                "is_default": str(d["name"]).startswith(str(default_out["name"])),
                "rate": int(d["defaultSampleRate"]),
                "channels": int(d["maxInputChannels"]),
            }
        )
    return out


def stream_loopback(
    pa: Any, device_id: str | None, out: BinaryIO, max_seconds: float | None
) -> int:
    devs = loopback_devices(pa)
    if not devs:
        print("no WASAPI loopback device found", file=sys.stderr)
        return EXIT_UNAVAILABLE
    dev = next((d for d in devs if d["id"] == device_id), None) if device_id else None
    dev = dev or next((d for d in devs if d["is_default"]), devs[0])
    q: queue.Queue[bytes] = queue.Queue(maxsize=256)

    def cb(data: bytes, frames: int, t: Any, status: Any) -> tuple[None, int]:
        with contextlib.suppress(queue.Full):  # consumer too slow: drop the chunk
            q.put_nowait(data)
        return (None, pa.paContinue)

    stream = pa.open(
        format=pa.paFloat32,
        channels=dev["channels"],
        rate=dev["rate"],
        input=True,
        input_device_index=int(dev["id"]),
        frames_per_buffer=int(dev["rate"] * 0.02),
        stream_callback=cb,
    )
    rs = StreamingResampler(dev["rate"], 16_000)
    produced = 0
    try:
        while max_seconds is None or produced < max_seconds * 16_000:
            try:
                raw = q.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm = np.frombuffer(raw, dtype="<f4")
            pcm = pcm[: pcm.size - pcm.size % dev["channels"]]
            y = rs.process(to_mono(pcm, dev["channels"]))
            if y.size:
                out.write(y.astype("<f4").tobytes())
                out.flush()
                produced += y.size
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        stream.stop_stream()
        stream.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print loopback devices as JSON")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("PyAudioWPatch is not installed (pip install PyAudioWPatch)", file=sys.stderr)
        return EXIT_UNAVAILABLE
    pa = pyaudio.PyAudio()
    try:
        if a.list:
            print(json.dumps(loopback_devices(pa)))
            return 0
        return stream_loopback(pa, a.device, sys.stdout.buffer, a.max_seconds)
    except OSError as e:
        print(f"WASAPI error: {e}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        pa.terminate()


if __name__ == "__main__":
    sys.exit(main())
