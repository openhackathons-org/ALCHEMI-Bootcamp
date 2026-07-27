"""Tests for removal of verified AIMNet files from the Docker build layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "build" / "prewarm_aimnet.py"
SPEC = importlib.util.spec_from_file_location("part1_prewarm_aimnet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report(path: Path) -> dict[str, object]:
    return {
        "cache_dir": str(path.parent),
        "checkpoints_retained": True,
        "checkpoints": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": MODULE.sha256_file(path),
            }
        ],
    }


def test_remove_verified_checkpoints_deletes_the_checked_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checked checkpoint")
    report = _report(checkpoint)

    assert MODULE.remove_verified_checkpoints(report) == [str(checkpoint)]
    assert not checkpoint.exists()
    assert report["checkpoints_retained"] is False
    assert report["removed_checkpoint_paths"] == [str(checkpoint)]


def test_remove_verified_checkpoints_refuses_a_changed_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checked checkpoint")
    report = _report(checkpoint)
    checkpoint.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="changed after validation"):
        MODULE.remove_verified_checkpoints(report)
    assert checkpoint.exists()
