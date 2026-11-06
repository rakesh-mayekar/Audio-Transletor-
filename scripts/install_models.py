"""Explicit model installer. The ONLY entry point that may use the network (and only with
--allow-network). Use --from-dir for a fully offline install from a local copy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai.model_manager import ModelManager
from ai.model_registry import REGISTRY
from core.config import paths
from core.errors import TranslatorError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(REGISTRY), default="small")
    ap.add_argument("--models-dir", type=Path, default=None, help="default: per-OS user data dir")
    ap.add_argument("--allow-network", action="store_true", help="permit the one-time download")
    ap.add_argument("--from-dir", type=Path, help="install from a local folder (no network)")
    ap.add_argument("--list", action="store_true", help="show installed models and exit")
    ap.add_argument("--verify", action="store_true", help="fully re-hash an installed model")
    a = ap.parse_args(argv)
    mgr = ModelManager(a.models_dir or paths.models_dir())
    try:
        if a.list:
            for s in mgr.list_status():
                state = (
                    "verified"
                    if s.verified
                    else ("INVALID: " + str(s.error) if s.installed else "not installed")
                )
                print(f"{s.name:8} {state:14} {s.size_bytes / 1e6:8.1f} MB  {s.path}")
            return 0
        if a.verify:
            mgr.verify(a.model, full=True)
            print(f"{a.model}: OK")
            return 0
        if a.from_dir:
            mgr.install_from_directory(a.model, a.from_dir, progress=print)
        else:
            mgr.install(a.model, allow_network=a.allow_network, progress=print)
        mgr.verify(a.model, full=True)
        print(f"installed {a.model} -> {mgr.model_dir(a.model)}")
        return 0
    except TranslatorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
