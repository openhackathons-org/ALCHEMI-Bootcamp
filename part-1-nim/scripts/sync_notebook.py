#!/usr/bin/env python3
"""Cell-level merge between a locally-regenerated .ipynb and a live copy.

Problem: running `scripts/build_notebook.py` produces a fresh notebook
with empty outputs; `scp`-ing it to ws-loc wipes any cell outputs (and
any manual edits the user made in JupyterLab on the remote side). This
helper does a cell-by-cell merge:

  for each cell index i:
    * if cell sources differ → use the local source (author's intent wins),
      but keep the remote cell's outputs + execution_count,
      so if the source didn't change, everything stays identical;
      if it did change, the stale output is replaced with an empty one
      (because the old output no longer matches the new code).
    * if counts differ → warn and preserve local (structural edit).

Usage:
    # from part-1-nim/
    python3 scripts/sync_notebook.py push    # local -> ws-loc
    python3 scripts/sync_notebook.py pull    # ws-loc -> local (preserve outputs)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOCAL_NB = REPO_DIR / "alchemi-mace-water-sorbents.ipynb"
REMOTE_HOST = "ws-loc"
REMOTE_NB = "/tmp/alchemi-playbook-part1/alchemi-mace-water-sorbents.ipynb"


def _fetch_remote() -> dict:
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
        tmp = Path(f.name)
    subprocess.run(
        ["scp", "-q", f"{REMOTE_HOST}:{REMOTE_NB}", str(tmp)],
        check=True,
    )
    nb = json.loads(tmp.read_text())
    tmp.unlink()
    return nb


def _push_remote(nb: dict) -> None:
    with tempfile.NamedTemporaryFile(
        suffix=".ipynb", mode="w", delete=False
    ) as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
        tmp = Path(f.name)
    subprocess.run(
        ["scp", "-q", str(tmp), f"{REMOTE_HOST}:{REMOTE_NB}"],
        check=True,
    )
    tmp.unlink()


def merge(local: dict, remote: dict, direction: str) -> dict:
    """Merge cells; outputs always come from *remote* (the live copy)."""
    if len(local["cells"]) != len(remote["cells"]):
        print(
            f"[warn] cell count differs "
            f"(local={len(local['cells'])}, remote={len(remote['cells'])}); "
            "falling back to local-wins",
        )
        return local

    merged = json.loads(json.dumps(local))  # deep copy
    changed = 0
    for i, (lc, rc) in enumerate(zip(merged["cells"], remote["cells"])):
        if lc["cell_type"] != rc["cell_type"]:
            print(f"[warn] cell {i} type differs; keeping local.")
            continue
        # Source: local wins (author intent).
        src_changed = lc.get("source") != rc.get("source")
        # Outputs: remote wins (live runs) — but only if source hasn't changed.
        if lc["cell_type"] == "code":
            if src_changed:
                lc["outputs"] = []
                lc["execution_count"] = None
                changed += 1
            else:
                lc["outputs"] = rc.get("outputs", [])
                lc["execution_count"] = rc.get("execution_count")
    print(f"cells with source changes: {changed}")
    return merged


def cmd_push() -> None:
    local = json.loads(LOCAL_NB.read_text())
    remote = _fetch_remote()
    merged = merge(local, remote, direction="push")
    _push_remote(merged)
    LOCAL_NB.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"pushed merged notebook to {REMOTE_HOST}:{REMOTE_NB}")


def cmd_pull() -> None:
    remote = _fetch_remote()
    LOCAL_NB.write_text(json.dumps(remote, indent=1) + "\n")
    print(f"pulled remote notebook into {LOCAL_NB}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"push", "pull"}:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "push":
        cmd_push()
    else:
        cmd_pull()
