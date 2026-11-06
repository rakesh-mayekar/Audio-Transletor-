"""OS-appropriate application directories. Uses platformdirs, so no OS branching here.

Set OLT_HOME to relocate everything (tests, portable installs).
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

from core import APP_NAME


def _dirs() -> PlatformDirs:
    return PlatformDirs(appname=APP_NAME, appauthor=False)


def _override() -> Path | None:
    v = os.environ.get("OLT_HOME")
    return Path(v) if v else None


def config_dir() -> Path:
    o = _override()
    return (o / "config") if o else Path(_dirs().user_config_dir)


def data_dir() -> Path:
    o = _override()
    return (o / "data") if o else Path(_dirs().user_data_dir)


def models_dir() -> Path:
    return data_dir() / "models"


def log_dir() -> Path:
    o = _override()
    return (o / "logs") if o else Path(_dirs().user_log_dir)


def config_file() -> Path:
    return config_dir() / "settings.json"
