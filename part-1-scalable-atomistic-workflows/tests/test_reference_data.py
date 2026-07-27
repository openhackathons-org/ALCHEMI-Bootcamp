"""Tests for exact B97-3c dimer reference-geometry verification."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.reference_data import (  # noqa: E402
    load_verified_b97_3c_dimer_reference,
)
from reference import water_dimer_b97_3c as canonical  # noqa: E402


def _write_synthetic_reference(path: Path) -> None:
    records = [
        canonical.EnergyRecord(
            point,
            energy_ab_Eh=-152.0 + 1.0e-4 * point.index,
            energy_a_Eh=-76.0,
            energy_b_Eh=-76.0,
        )
        for point in canonical.load_scan_geometries()
    ]
    canonical.write_scan_csv(path, records)
    _seal_reference_bundle(path)


def _default_manifest(path: Path) -> dict[str, object]:
    source_paths = {
        "generator": canonical.GENERATOR_SOURCE,
        "geometry_builder": canonical.STRUCTURE_SOURCE,
        "environment_spec": canonical.ENVIRONMENT_SOURCE,
        "repository_license": canonical.LICENSE_SOURCE,
    }
    return {
        "schema": "alchemi.b97-3c-water-dimer-scan.v1",
        "status": "complete",
        "method": {
            "method": "B97-3c",
            "basis": "def2-mTZVP",
        },
        "scan": {
            "single_point_count": 24,
        },
        "sources": {
            name: {
                "path": str(source_path),
                "sha256": canonical.sha256_file(source_path),
            }
            for name, source_path in source_paths.items()
            if source_path.is_file()
        },
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": canonical.sha256_file(path),
            }
        ],
    }


def _write_checksum_index(path: Path) -> None:
    artifact_dir = path.parent
    covered = (path, artifact_dir / "manifest.json")
    (artifact_dir / "SHA256SUMS").write_text(
        "".join(f"{canonical.sha256_file(item)}  {item.name}\n" for item in covered),
        encoding="utf-8",
    )


def _seal_reference_bundle(path: Path) -> None:
    canonical.write_json(path.parent / "manifest.json", _default_manifest(path))
    _write_checksum_index(path)


def _rewrite_manifest(path: Path, transform: Callable[[dict[str, Any]], None]) -> None:
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform(manifest)
    canonical.write_json(manifest_path, manifest)
    _write_checksum_index(path)


def test_load_verified_reference_matches_live_canonical_geometries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)

    table = load_verified_b97_3c_dimer_reference(path)

    assert isinstance(table, pd.DataFrame)
    assert table["scan_index"].tolist() == list(range(8))
    assert table["requested_oo_distance_angstrom"].tolist() == list(
        canonical.DEFAULT_OO_DISTANCES_ANGSTROM
    )


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("requested_oo_distance_angstrom", 2.51, "requested_oo_distance"),
        ("measured_oo_distance_angstrom", 2.51, "measured_oo_distance"),
        ("ab_geometry_sha256", "0" * 64, "live canonical geometry"),
        ("a_geometry_sha256", "1" * 64, "live canonical geometry"),
        ("b_geometry_sha256", "2" * 64, "live canonical geometry"),
    ],
)
def test_load_verified_reference_rejects_geometry_drift(
    tmp_path: Path,
    column: str,
    replacement: object,
    message: str,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)
    table = pd.read_csv(path)
    table.loc[0, column] = replacement
    table.to_csv(path, index=False)
    _seal_reference_bundle(path)

    with pytest.raises(ValueError, match=message):
        load_verified_b97_3c_dimer_reference(path)


def test_load_verified_reference_rejects_reordered_or_incomplete_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)
    table = pd.read_csv(path)
    table.iloc[::-1].to_csv(path, index=False)
    _seal_reference_bundle(path)
    with pytest.raises(ValueError, match="scan_index row 0"):
        load_verified_b97_3c_dimer_reference(path)

    table.iloc[:-1].to_csv(path, index=False)
    _seal_reference_bundle(path)
    with pytest.raises(ValueError, match="7 rows; expected 8"):
        load_verified_b97_3c_dimer_reference(path)


def test_load_verified_reference_rejects_csv_checksum_tamper(tmp_path: Path) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch for interaction_curve"):
        load_verified_b97_3c_dimer_reference(path)


def test_load_verified_reference_rejects_manifest_checksum_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 mismatch for manifest.json"):
        load_verified_b97_3c_dimer_reference(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("status", "incomplete", "status is not complete"),
        ("method", "HF", "wrong method"),
        ("basis", "sto-3g", "wrong basis"),
        ("single_point_count", 23, "required 24 single-point"),
    ],
)
def test_load_verified_reference_rejects_resealed_manifest_contract_tamper(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)

    def transform(manifest: dict[str, Any]) -> None:
        if field == "status":
            manifest["status"] = replacement
        elif field in {"method", "basis"}:
            manifest["method"][field] = replacement
        else:
            manifest["scan"][field] = replacement

    _rewrite_manifest(path, transform)
    with pytest.raises(ValueError, match=message):
        load_verified_b97_3c_dimer_reference(path)


def test_load_verified_reference_rejects_resealed_source_hash_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)

    def transform(manifest: dict[str, Any]) -> None:
        manifest["sources"]["geometry_builder"]["sha256"] = "0" * 64

    _rewrite_manifest(path, transform)
    with pytest.raises(ValueError, match="does not match live geometry_builder"):
        load_verified_b97_3c_dimer_reference(path)


def test_load_verified_reference_rejects_resealed_file_inventory_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interaction_curve.csv"
    _write_synthetic_reference(path)

    def transform(manifest: dict[str, Any]) -> None:
        manifest["files"][0]["sha256"] = "f" * 64

    _rewrite_manifest(path, transform)
    with pytest.raises(ValueError, match="manifest SHA-256.*inconsistent"):
        load_verified_b97_3c_dimer_reference(path)
