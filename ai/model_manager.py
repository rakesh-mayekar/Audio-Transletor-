"""Local model manager. The ONLY code in the product allowed to touch the network, and only
inside `install()` when the caller passes allow_network=True. Loading never downloads."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ai.model_registry import REGISTRY, ModelSpec
from core.errors import (
    ModelError,
    ModelIntegrityError,
    ModelMissingError,
    ModelNetworkNotAllowedError,
)

_log = logging.getLogger(__name__)
MANIFEST = "manifest.json"
ProgressFn = Callable[[str], None]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelStatus:
    name: str
    installed: bool
    verified: bool
    path: Path
    size_bytes: int
    revision: str | None
    error: str | None = None


class ModelManager:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)

    # ------------------------------------------------------------------ queries
    def spec(self, name: str) -> ModelSpec:
        try:
            return REGISTRY[name]
        except KeyError:
            raise ModelError(
                f"Model '{name}' is not in the registry",
                hint="Available: " + ", ".join(sorted(REGISTRY)),
            ) from None

    def model_dir(self, name: str) -> Path:
        return self.models_dir / name

    def verify(self, name: str, *, full: bool = True) -> None:
        """Raise ModelMissingError / ModelIntegrityError unless the install is intact.

        full=True re-hashes every file against the manifest (and model.bin against the pinned
        registry hash). full=False checks presence and sizes only.
        """
        spec = self.spec(name)
        d = self.model_dir(name)
        manifest_path = d / MANIFEST
        if not manifest_path.is_file():
            raise ModelMissingError(
                f"Model '{name}' is not installed",
                hint=f"Run: python scripts/install_models.py --model {name} --allow-network "
                f"(one-time download) or --from-dir <folder> (no network).",
                path=str(d),
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded = manifest["files"]
        except (OSError, ValueError, KeyError) as e:
            raise ModelIntegrityError(
                f"Corrupt manifest for '{name}': {e}", hint="Reinstall."
            ) from e
        if manifest.get("revision") != spec.revision:
            raise ModelIntegrityError(
                f"Installed '{name}' revision {manifest.get('revision')} != pinned {spec.revision}",
                hint="Reinstall the model.",
            )
        for f in spec.files:
            p = d / f.name
            if not p.is_file():
                raise ModelIntegrityError(f"Missing file {f.name} in '{name}'", hint="Reinstall.")
            if p.stat().st_size != f.size:
                raise ModelIntegrityError(f"Size mismatch for {f.name}", hint="Reinstall.")
            if full:
                digest = sha256_file(p)
                if f.sha256 is not None and digest != f.sha256:
                    raise ModelIntegrityError(f"SHA-256 mismatch for {f.name}", hint="Reinstall.")
                if recorded.get(f.name, {}).get("sha256") != digest:
                    raise ModelIntegrityError(
                        f"{f.name} differs from the hash recorded at install", hint="Reinstall."
                    )

    def status(self, name: str, *, full: bool = False) -> ModelStatus:
        d = self.model_dir(name)
        spec = self.spec(name)
        if not (d / MANIFEST).is_file():
            return ModelStatus(name, False, False, d, 0, None)
        size = sum(p.stat().st_size for p in d.iterdir() if p.is_file())
        try:
            self.verify(name, full=full)
            return ModelStatus(name, True, True, d, size, spec.revision)
        except ModelError as e:
            return ModelStatus(name, True, False, d, size, spec.revision, str(e))

    def list_status(self) -> list[ModelStatus]:
        return [self.status(n) for n in sorted(REGISTRY)]

    def resolve(self, name: str, *, full_verify: bool = True) -> Path:
        """Return the verified local directory for `name`. Never downloads."""
        self.verify(name, full=full_verify)
        return self.model_dir(name)

    # ------------------------------------------------------------------ install
    def install(
        self, name: str, *, allow_network: bool = False, progress: ProgressFn | None = None
    ) -> Path:
        """Explicit one-time download of pinned files. Refuses unless allow_network=True."""
        spec = self.spec(name)
        if not allow_network:
            raise ModelNetworkNotAllowedError(
                "Downloading a model requires explicit permission",
                hint="Pass --allow-network to the installer, or use --from-dir with a local copy.",
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover
            raise ModelError(
                "huggingface_hub is required for downloads", hint="pip install huggingface_hub"
            ) from e
        self.models_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.models_dir))
        try:
            for f in spec.files:
                if progress:
                    progress(f"downloading {f.name} ({f.size / 1e6:.1f} MB)")
                got = hf_hub_download(
                    repo_id=spec.repo_id, filename=f.name, revision=spec.revision, local_dir=staging
                )
                _ = got
            return self._commit(spec, staging, progress)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def install_from_directory(
        self, name: str, src: Path, *, progress: ProgressFn | None = None
    ) -> Path:
        """Network-free install from a folder containing the pinned files (e.g. USB copy)."""
        spec = self.spec(name)
        src = Path(src)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.models_dir))
        try:
            for f in spec.files:
                s = src / f.name
                if not s.is_file():
                    raise ModelMissingError(f"{f.name} not found in {src}")
                if progress:
                    progress(f"copying {f.name}")
                shutil.copy2(s, staging / f.name)
            return self._commit(spec, staging, progress)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _commit(self, spec: ModelSpec, staging: Path, progress: ProgressFn | None) -> Path:
        files: dict[str, dict[str, object]] = {}
        for f in spec.files:
            p = staging / f.name
            if not p.is_file() or p.stat().st_size != f.size:
                raise ModelIntegrityError(
                    f"{f.name}: missing or wrong size", hint="Retry the install."
                )
            if progress:
                progress(f"verifying {f.name}")
            digest = sha256_file(p)
            if f.sha256 is not None and digest != f.sha256:
                raise ModelIntegrityError(
                    f"{f.name}: SHA-256 does not match the pinned value",
                    hint="The file is corrupt or not the pinned release; nothing was installed.",
                )
            files[f.name] = {"size": f.size, "sha256": digest}
        (staging / MANIFEST).write_text(
            json.dumps(
                {
                    "name": spec.name,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "license": spec.license,
                    "installed_at": int(time.time()),
                    "files": files,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for p in list(staging.iterdir()):  # drop hub bookkeeping (.cache etc.)
            if p.name not in {f.name for f in spec.files} | {MANIFEST}:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        dest = self.model_dir(spec.name)
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
        staging.mkdir()  # so the caller's cleanup of `staging` is a harmless no-op
        _log.info("installed model %s (%s)", spec.name, spec.revision)
        return dest

    def remove(self, name: str) -> bool:
        self.spec(name)
        d = self.model_dir(name)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False
