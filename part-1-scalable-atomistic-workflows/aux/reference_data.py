"""Strict geometry verification for the water-dimer B97-3c CSV artifact."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file


CHECKSUM_INDEX_NAME = "SHA256SUMS"
MANIFEST_NAME = "manifest.json"
REFERENCE_CSV_NAME = "interaction_curve.csv"


def _load_checksum_index(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing B97-3c checksum index: {path}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        digest, relative_path = fields
        relative_path = relative_path.removeprefix("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError(f"invalid SHA-256 on SHA256SUMS line {line_number}")
        if relative_path in checksums:
            raise ValueError(f"duplicate SHA256SUMS entry: {relative_path}")
        checksums[relative_path] = digest.lower()
    return checksums


def _verify_indexed_file(
    path: Path, *, relative_path: str, checksums: Mapping[str, str]
) -> str:
    if relative_path not in checksums:
        raise ValueError(f"SHA256SUMS does not cover {relative_path}")
    actual = sha256_file(path)
    if actual != checksums[relative_path]:
        raise ValueError(f"SHA-256 mismatch for {relative_path}")
    return actual


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"B97-3c manifest {label} must be an object")
    return value


def _verify_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    csv_path: Path,
    csv_sha256: str,
    canonical: Any,
) -> None:
    schema = manifest.get("schema")
    if schema is not None and schema != "alchemi.b97-3c-water-dimer-scan.v1":
        raise ValueError(f"unexpected B97-3c dimer manifest schema: {schema!r}")
    if manifest.get("status") != "complete":
        raise ValueError("B97-3c dimer artifact status is not complete")

    method = _require_mapping(manifest.get("method"), "method")
    if str(method.get("method", "")).casefold() != canonical.METHOD.casefold():
        raise ValueError("B97-3c dimer manifest has the wrong method")
    if str(method.get("basis", "")).casefold() != canonical.BASIS.casefold():
        raise ValueError("B97-3c dimer manifest has the wrong basis")

    scan = _require_mapping(manifest.get("scan"), "scan")
    required_single_points = 3 * len(canonical.DEFAULT_OO_DISTANCES_ANGSTROM)
    if scan.get("single_point_count") != required_single_points:
        raise ValueError(
            "B97-3c dimer manifest does not record the required "
            f"{required_single_points} single-point calculations"
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("B97-3c manifest files must be a list")
    csv_records = [
        record
        for record in files
        if isinstance(record, Mapping) and record.get("path") == csv_path.name
    ]
    if len(csv_records) != 1:
        raise ValueError("B97-3c manifest must inventory interaction_curve.csv once")
    csv_record = csv_records[0]
    if csv_record.get("sha256") != csv_sha256:
        raise ValueError("manifest SHA-256 for interaction_curve.csv is inconsistent")
    if csv_record.get("bytes") != csv_path.stat().st_size:
        raise ValueError("manifest byte size for interaction_curve.csv is inconsistent")

    sources = _require_mapping(manifest.get("sources"), "sources")
    live_sources = {
        "generator": canonical.GENERATOR_SOURCE,
        "geometry_builder": canonical.STRUCTURE_SOURCE,
        "environment_spec": canonical.ENVIRONMENT_SOURCE,
        "repository_license": canonical.LICENSE_SOURCE,
    }
    for required_source in ("generator", "geometry_builder"):
        if required_source not in sources:
            raise ValueError(
                f"B97-3c manifest is missing required source: {required_source}"
            )
    for source_name, live_path in live_sources.items():
        if source_name not in sources:
            continue
        record = _require_mapping(sources[source_name], f"sources.{source_name}")
        recorded_hash = record.get("sha256")
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise ValueError(
                f"B97-3c manifest source {source_name} has no valid SHA-256"
            )
        if not live_path.is_file():
            raise ValueError(f"live B97-3c source is missing: {live_path}")
        if recorded_hash.lower() != sha256_file(live_path):
            raise ValueError(
                f"B97-3c manifest source hash does not match live {source_name}"
            )


def _authenticate_reference_bundle(csv_path: Path, canonical: Any) -> None:
    if csv_path.name != REFERENCE_CSV_NAME:
        raise ValueError(f"B97-3c reference CSV must be named {REFERENCE_CSV_NAME}")
    if not csv_path.is_file():
        raise ValueError(f"missing B97-3c interaction curve: {csv_path}")

    artifact_dir = csv_path.parent
    manifest_path = artifact_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing B97-3c manifest: {manifest_path}")
    checksums = _load_checksum_index(artifact_dir / CHECKSUM_INDEX_NAME)
    csv_sha256 = _verify_indexed_file(
        csv_path,
        relative_path=REFERENCE_CSV_NAME,
        checksums=checksums,
    )
    _verify_indexed_file(
        manifest_path,
        relative_path=MANIFEST_NAME,
        checksums=checksums,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("B97-3c manifest is not valid JSON") from exc
    _verify_manifest_contract(
        _require_mapping(manifest, "root"),
        csv_path=csv_path,
        csv_sha256=csv_sha256,
        canonical=canonical,
    )


def _first_mismatch(observed: np.ndarray, expected: np.ndarray) -> int:
    mismatches = np.flatnonzero(observed != expected)
    return int(mismatches[0])


def _require_exact_numeric_column(
    table: pd.DataFrame,
    column: str,
    expected: np.ndarray,
) -> None:
    try:
        observed = table[column].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain numeric values") from exc
    if np.array_equal(observed, expected):
        return
    if observed.shape != expected.shape:
        raise ValueError(
            f"{column} has shape {observed.shape}; expected {expected.shape}"
        )
    row = _first_mismatch(observed, expected)
    raise ValueError(
        f"{column} row {row} is {observed[row]!r}; expected {expected[row]!r}"
    )


def _require_exact_hash_column(
    table: pd.DataFrame,
    column: str,
    expected: list[str],
) -> None:
    observed: list[Any] = table[column].tolist()
    if observed == expected:
        return
    row = next(
        index
        for index, (actual, required) in enumerate(zip(observed, expected, strict=True))
        if actual != required
    )
    raise ValueError(f"{column} row {row} does not match the live canonical geometry")


def load_verified_b97_3c_dimer_reference(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load the B97-3c dimer curve after exact live-geometry verification.

    The sibling checksum index authenticates the CSV and manifest before the
    CSV is parsed. Manifest source hashes and the ordered O--O grid plus all
    AB/A/B geometry hashes must then match the live canonical sources and
    freshly built ASE structures. These checks establish bundle integrity and
    geometry identity; they do not rerun the electronic-structure calculation.
    """

    from reference import water_dimer_b97_3c as canonical

    csv_path = Path(csv_path)
    _authenticate_reference_bundle(csv_path, canonical)
    table = pd.read_csv(csv_path)
    required = set(canonical.CSV_FIELDS)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            "B97-3c dimer CSV is missing required columns: " + ", ".join(missing)
        )

    points = canonical.load_scan_geometries()
    if len(table) != len(points):
        raise ValueError(
            f"B97-3c dimer CSV has {len(table)} rows; expected {len(points)}"
        )

    _require_exact_numeric_column(
        table,
        "scan_index",
        np.asarray([point.index for point in points], dtype=np.float64),
    )
    _require_exact_numeric_column(
        table,
        "requested_oo_distance_angstrom",
        np.asarray(
            [point.requested_oo_distance_angstrom for point in points],
            dtype=np.float64,
        ),
    )
    _require_exact_numeric_column(
        table,
        "measured_oo_distance_angstrom",
        np.asarray(
            [point.measured_oo_distance_angstrom for point in points],
            dtype=np.float64,
        ),
    )
    for role in ("ab", "a", "b"):
        _require_exact_hash_column(
            table,
            f"{role}_geometry_sha256",
            [canonical.geometry_sha256(getattr(point, role)) for point in points],
        )
    return table
