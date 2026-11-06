"""Streaming Silero VAD (v6 ONNX bundled with faster-whisper; no download, no network).

faster-whisper's own wrapper resets recurrent state on every call, which is wrong for live
audio, so we drive the same ONNX session statefully: LSTM (h, c) and a 64-sample context are
carried across windows.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

WINDOW = 512
CONTEXT = 64


class SileroStreamingVad:
    def __init__(self) -> None:
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._ctx = np.zeros(CONTEXT, np.float32)

    def process(self, samples: NDArray[np.float32]) -> list[float]:
        if samples.size % WINDOW:
            raise ValueError("samples must be a multiple of 512")
        probs: list[float] = []
        for i in range(0, samples.size, WINDOW):
            w = samples[i : i + WINDOW]
            x = np.concatenate([self._ctx, w])[None, :].astype(np.float32)
            out, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
            probs.append(float(out.reshape(-1)[0]))
            self._ctx = w[-CONTEXT:].copy()
        return probs
