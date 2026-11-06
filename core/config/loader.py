"""Load / save / migrate settings. Atomic writes; corrupt files are never silently reset."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.config.models import CURRENT_SCHEMA_VERSION, AppConfig
from core.errors import ConfigError, ConfigMigrationError

# Migration N maps a dict at schema N to schema N+1. Add entries when the schema changes;
# never edit an existing one (backward compatibility of old settings files).
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # 0 -> 1: files written before schema versioning existed; layout is identical.
    0: lambda d: dict(d),
}


def migrate(
    raw: dict[str, Any],
    migrations: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    target: int = CURRENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    migrations = MIGRATIONS if migrations is None else migrations
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ConfigMigrationError(f"Invalid schema_version: {version!r}")
    if version > target:
        raise ConfigMigrationError(
            f"Settings were written by a newer version (schema {version} > {target})",
            hint="Upgrade the application or remove the settings file.",
        )
    data = dict(raw)
    while version < target:
        step = migrations.get(version)
        if step is None:
            raise ConfigMigrationError(f"No migration from schema {version} to {version + 1}")
        data = step(data)
        version += 1
        data["schema_version"] = version
    data["schema_version"] = target
    return data


def parse_config(raw: dict[str, Any]) -> AppConfig:
    try:
        return AppConfig.model_validate(migrate(raw))
    except ValidationError as e:
        raise ConfigError("Invalid settings", hint=str(e), errors=e.errors()) from e


def load_config(path: Path) -> AppConfig:
    """Return defaults if the file does not exist; raise ConfigError if it is unreadable/invalid."""
    if not path.exists():
        return AppConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(
            f"Cannot read settings file {path}: {e}",
            hint="Fix or delete the file, or call recover_config().",
        ) from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Settings file {path} must contain a JSON object")
    return parse_config(raw)


def recover_config(path: Path) -> tuple[AppConfig, Path | None]:
    """Explicit recovery: move a bad file aside (kept for inspection) and return defaults."""
    try:
        return load_config(path), None
    except ConfigError:
        backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        os.replace(path, backup)
        return AppConfig(), backup


def save_config(config: AppConfig, path: Path) -> None:
    """Atomic write: temp file in the same directory, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump_json(indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
