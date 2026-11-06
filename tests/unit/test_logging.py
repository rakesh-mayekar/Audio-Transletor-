import logging

import numpy as np
import pytest

from core.logging import setup_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    setup_logging("WARNING", None, console=False)


def test_rotating_file_and_redaction(tmp_path):
    setup_logging("INFO", tmp_path, console=False)
    log = logging.getLogger("t")
    log.info("audio %s and %s", np.ones(1000, np.float32), b"\x00" * 64)
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "translator.log").read_text()
    assert "redacted" in text and "1000" in text and "1., 1." not in text
    assert "\\x00" not in text


def test_setup_is_idempotent(tmp_path):
    root = logging.getLogger()
    before = len(root.handlers)
    setup_logging("INFO", tmp_path, console=False)
    n1 = len(root.handlers)
    setup_logging("DEBUG", tmp_path, console=False)
    assert len(root.handlers) == n1 and root.level == logging.DEBUG
    setup_logging("INFO", None, console=False)
    assert len(root.handlers) == before
