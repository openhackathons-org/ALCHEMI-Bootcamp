#!/usr/bin/env python3
"""Sync runtime-critical files (helpers/, data/hosts/) to ws-loc.

The notebook file itself is NOT synced: VS Code writes the ipynb
locally in WSL, and the Jupyter kernel on ws-loc never opens it - it
just receives cell source over the Jupyter wire protocol. Shipping
the ipynb between machines was a cargo-cult step that erased live
cell outputs on every round trip; this helper replaces it.

What DOES get synced:

  helpers/*.py        - imported by the kernel; changes require a
                        kernel restart to take effect
  data/hosts/*        - loaded via notebook relative paths

Everything else (cached_responses/, outputs/, assets/) is created by
the kernel at runtime directly on ws-loc and stays there.

Usage:
    # from part-1-nim/
    python3 scripts/sync_notebook.py     # default: push helpers + data
    python3 scripts/sync_notebook.py --all  # also push Dockerfile +
                                            # docker-compose.yml for a
                                            # rebuild
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
REMOTE_HOST = "ws-loc"
REMOTE_ROOT = "/tmp/alchemi-playbook-part1"

CORE_PATHS = [
    "helpers/",
    "data/hosts/",
]
BUILD_PATHS = [
    "Dockerfile",
    "docker-compose.yml",
    "environment.yml",
    "scripts/",
]


def _rsync(paths: list[str]) -> None:
    for p in paths:
        src = REPO_DIR / p
        dst = f"{REMOTE_HOST}:{REMOTE_ROOT}/{p}"
        if not src.exists():
            print(f"[skip] {src} does not exist locally")
            continue
        cmd = [
            "rsync", "-az", "--delete" if src.is_dir() else "--no-R",
            "-e", "ssh",
            "--exclude=__pycache__/",
            "--exclude=.pytest_cache/",
            str(src) + ("/" if src.is_dir() else ""),
            dst,
        ]
        subprocess.run(cmd, check=True)
        print(f"[sync] {p} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="Also sync Dockerfile / docker-compose.yml / scripts/ "
                         "for a container rebuild")
    args = ap.parse_args()
    paths = CORE_PATHS + (BUILD_PATHS if args.all else [])
    _rsync(paths)
    print()
    print("Notebook file NOT synced (stays on WSL; kernel gets cell source via"
          " the Jupyter wire protocol).")
    print()
    print("If you changed helpers/, restart the kernel in VS Code / JupyterLab"
          " so the new modules are re-imported.")


if __name__ == "__main__":
    main()
