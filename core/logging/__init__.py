"""Logging setup with rotation and a privacy filter that never lets raw audio into logs."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import numpy as np

_HANDLER_TAG = "_olt_handler"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"


def _redact(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype} redacted>"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes len={len(value)} redacted>"
    return value


class RedactAudioFilter(logging.Filter):
    """Replaces audio-like arguments (ndarray/bytes) so raw audio cannot reach a log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_redact(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _redact(v) for k, v in record.args.items()}
        record.msg = _redact(record.msg)
        return True


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    console: bool = True,
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Idempotent: calling again replaces only the handlers this function installed."""
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(h)
        h.close()
    root.setLevel(level)
    fmt = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler())
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_dir / "translator.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        )
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactAudioFilter())
        setattr(h, _HANDLER_TAG, True)
        root.addHandler(h)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
