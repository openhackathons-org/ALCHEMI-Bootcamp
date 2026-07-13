"""Tests for deterministic water-run artifact manifests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import (  # noqa: E402
    WATER_RUN_MANIFEST_NAME,
    write_water_run_manifest,
)


def test_water_run_manifest_is_complete_normalized_and_repeatable(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "water_spectrum.csv"
    second = nested / "water_cluster.extxyz"
    first.write_bytes(b"frequency,intensity\n1000,0.4\n")
    second.write_bytes(b"3\nframe\n")
    notes = tmp_path / "notes.txt"
    notes.write_text("run note")

    manifest = write_water_run_manifest(
        tmp_path,
        provenance={"commit": "abc123", "root": Path("repo/part-1")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        gates={"passed": np.bool_(True), "errors": np.array([0.0, 1.0e-6])},
    )
    manifest_path = tmp_path / WATER_RUN_MANIFEST_NAME
    first_bytes = manifest_path.read_bytes()
    repeated = write_water_run_manifest(
        tmp_path,
        provenance={"commit": "abc123", "root": Path("repo/part-1")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        gates={"passed": np.bool_(True), "errors": np.array([0.0, 1.0e-6])},
    )

    assert manifest_path.read_bytes() == first_bytes
    assert repeated == manifest == json.loads(first_bytes)
    assert manifest["provenance"]["root"] == "repo/part-1"
    assert manifest["settings"] == {"dt_fs": 0.5, "steps": 55_000}
    assert manifest["gates"] == {"errors": [0.0, 1.0e-6], "passed": True}
    assert [item["path"] for item in manifest["files"]] == [
        "nested/water_cluster.extxyz",
        "notes.txt",
        "water_spectrum.csv",
    ]
    for item, path in zip(manifest["files"], (second, notes, first), strict=True):
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert WATER_RUN_MANIFEST_NAME not in {item["path"] for item in manifest["files"]}


def test_water_run_manifest_inventories_nested_zarr_files(tmp_path: Path) -> None:
    zarr = tmp_path / "water_ir_relaxed.zarr"
    chunks = zarr / "positions" / "c"
    chunks.mkdir(parents=True)
    root_metadata = zarr / "zarr.json"
    array_metadata = zarr / "positions" / "zarr.json"
    chunk = chunks / "0"
    root_metadata.write_text('{"zarr_format": 3}\n')
    array_metadata.write_text('{"node_type": "array"}\n')
    chunk.write_bytes(b"binary chunk")
    run_log = tmp_path / "run.log"
    run_log.write_text("complete\n")

    manifest = write_water_run_manifest(
        tmp_path,
        provenance={},
        settings={},
        gates={},
    )

    assert [item["path"] for item in manifest["files"]] == [
        "run.log",
        "water_ir_relaxed.zarr/positions/c/0",
        "water_ir_relaxed.zarr/positions/zarr.json",
        "water_ir_relaxed.zarr/zarr.json",
    ]
    assert [item["bytes"] for item in manifest["files"]] == [
        path.stat().st_size for path in (run_log, chunk, array_metadata, root_metadata)
    ]


def test_water_run_manifest_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_water_run_manifest(
            tmp_path,
            provenance={},
            settings={"temperature": np.float64(np.nan)},
            gates={},
        )
    assert not (tmp_path / WATER_RUN_MANIFEST_NAME).exists()
