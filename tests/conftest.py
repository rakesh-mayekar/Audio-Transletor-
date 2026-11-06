import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_worker_threads():
    """Every test must shut down the bus/capture threads it starts (graceful-shutdown rule)."""
    yield
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        leaked = [
            t.name
            for t in threading.enumerate()
            if t.is_alive() and t.name.startswith(("bus-", "fake-capture", "capture-"))
        ]
        if not leaked:
            return
        time.sleep(0.05)
    pytest.fail(f"leaked worker threads: {leaked}")
