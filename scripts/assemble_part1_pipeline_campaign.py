#!/usr/bin/env python3
"""Seal fetched Compute Lab campaign records into a learner result bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aux.artifacts import sha256_file  # noqa: E402
from aux.pipeline_campaign_results import (  # noqa: E402
    BUNDLE_SCHEMA,
    ROUTE_SPECS,
    RUN_COLUMNS,
    RUN_SCHEMA,
    canonical_json_sha256,
    correctness_record_sha256,
    load_pipeline_campaign_bundle,
    producer_set_sha256,
    runtime_identity_sha256,
    toolkit_runtime_record_sha256,
)
from part1_distributed_campaign_contract import (  # noqa: E402
    CORE_BRANCH,
    CORE_COMMIT,
    CORE_VERSION,
    CAMPAIGN_REPEATS,
    DEFAULT_SYSTEMS,
    OPS_COMMIT,
    OPS_VERSION,
    PRODUCER_FILES,
)


RAW_INDEX_NAME = "raw-SHA256SUMS"
PRODUCER_INDEX_NAME = "producer-SHA256SUMS"
REQUIRED_REPEATS = CAMPAIGN_REPEATS

RAW_KEYS = {
    "backend",
    "correctness_checks",
    "correctness_passed",
    "correctness_tolerances",
    "covalent_oh_gate_passed",
    "duplicate_systems",
    "elapsed_s",
    "error",
    "error_type",
    "gpu_count",
    "gpu_name",
    "hostname_rank0",
    "max_abs_net_charge_e",
    "max_charge_difference_e",
    "max_energy_difference_ev",
    "max_force_difference_ev_per_a",
    "max_handoff_fmax_ev_per_a",
    "max_relaxation_steps_observed",
    "min_interatomic_distance_a",
    "missing_systems",
    "model",
    "nodes",
    "nve_steps_verified",
    "nvt_steps_verified",
    "oxygen_connectivity_gate_passed",
    "partition",
    "peak_memory_bytes_max_rank",
    "pipeline_count",
    "producer_set",
    "purpose",
    "python_hash_seed",
    "python_version",
    "rank_audits",
    "rank_count",
    "repeat",
    "repository_commit",
    "route",
    "run_id",
    "schema",
    "slurm_job_id",
    "stage_1_completions",
    "stage_2_completions",
    "status",
    "success",
    "systems_completed",
    "systems_per_s",
    "systems_requested",
    "timestamp_utc",
    "timing_boundary",
    "toolkit_core_commit",
    "toolkit_core_branch",
    "toolkit_core_clean",
    "toolkit_core_version",
    "toolkit_ops_commit",
    "toolkit_ops_version",
    "torch_version",
    "unexpected_systems",
    "unique_systems_completed",
    "workload",
}


class AssemblyError(ValueError):
    """Raised when fetched raw records cannot form a valid campaign bundle."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_checksum_index(root: Path, name: str) -> dict[Path, str]:
    index = root / name
    if not index.is_file():
        raise AssemblyError(f"missing {name} in {root}")
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(
        index.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise AssemblyError(f"malformed {name} line {line_number}")
        digest, raw_path = fields
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise AssemblyError(f"invalid SHA-256 in {name} line {line_number}")
        raw_path = raw_path.removeprefix("*")
        indexed_path = Path(raw_path)
        if indexed_path.is_absolute():
            # Compute Lab records the original absolute path. After the job
            # directory is fetched, the same artifact lives under local raw/.
            path = root / "raw" / indexed_path.name
        else:
            path = root / indexed_path
        path = path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise AssemblyError(f"{name} path escapes its job directory") from exc
        if path in entries:
            raise AssemblyError(f"duplicate {name} entry for {path.name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise AssemblyError(f"SHA-256 mismatch for {path}")
        entries[path] = digest
    if not entries:
        raise AssemblyError(f"{name} is empty")
    return entries


def _producer_index_name(raw_path: str) -> str:
    """Map a local or Compute Lab producer path to its contract name."""

    normalized = Path(raw_path.removeprefix("*")).as_posix().rstrip("/")
    expected_paths = PRODUCER_FILES
    matches = [
        expected
        for expected in expected_paths
        if normalized == expected or normalized.endswith(f"/{expected}")
    ]
    if len(matches) != 1:
        raise AssemblyError(f"unexpected producer path: {raw_path}")
    return matches[0]


def _read_producer_index(root: Path) -> dict[str, str]:
    index = root / PRODUCER_INDEX_NAME
    if not index.is_file():
        raise AssemblyError(f"missing {PRODUCER_INDEX_NAME} in {root}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        index.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise AssemblyError(
                f"malformed {PRODUCER_INDEX_NAME} line {line_number}"
            )
        digest, raw_path = fields
        relative_path = _producer_index_name(raw_path)
        if relative_path in entries:
            raise AssemblyError(f"duplicate producer entry: {relative_path}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise AssemblyError(f"invalid producer SHA-256: {relative_path}")
        entries[relative_path] = digest
    expected = set(PRODUCER_FILES)
    if set(entries) != expected:
        raise AssemblyError(
            "producer-SHA256SUMS does not cover the exact campaign producer set"
        )
    producer_entries = {
        relative_path: entries[relative_path] for relative_path in PRODUCER_FILES
    }
    for relative_path, digest in producer_entries.items():
        local_file = REPOSITORY_ROOT / relative_path
        if not local_file.is_file() or sha256_file(local_file) != digest:
            raise AssemblyError(
                f"local producer does not match fetched identity: {relative_path}"
            )
    return producer_entries


def _load_job_records(
    root: Path,
) -> tuple[
    list[tuple[Path, Mapping[str, Any]]],
    tuple[Path, Mapping[str, Any]],
    dict[str, str],
]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise AssemblyError(f"campaign job directory not found: {root}")
    indexed_raw = _read_checksum_index(root, RAW_INDEX_NAME)
    raw_json = set((root / "raw").glob("*.json"))
    raw_incomplete = set((root / "raw").glob("*.json.incomplete"))
    raw_artifacts = raw_json | raw_incomplete
    if {path.resolve() for path in raw_artifacts} != set(indexed_raw):
        raise AssemblyError(
            f"{RAW_INDEX_NAME} must cover every raw JSON artifact in {root}"
        )
    producer_files = _read_producer_index(root)
    records: list[tuple[Path, Mapping[str, Any]]] = []
    smoke_records: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(raw_json):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise AssemblyError(f"cannot read raw record {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise AssemblyError(f"raw record must be an object: {path}")
        if value.get("schema") != RUN_SCHEMA:
            raise AssemblyError(f"raw record has an unexpected schema: {path}")
        if value.get("purpose") not in {"campaign", "smoke"}:
            raise AssemblyError(f"raw record has an unexpected purpose: {path}")
        observed = set(value)
        if observed != RAW_KEYS:
            missing = sorted(RAW_KEYS - observed)
            extra = sorted(observed - RAW_KEYS)
            raise AssemblyError(
                f"raw record keys differ in {path} (missing={missing}; extra={extra})"
            )
        if value.get("producer_set") != producer_files:
            raise AssemblyError(f"producer_set differs from its checksum index: {path}")
        expected_runtime = {
            "toolkit_core_branch": CORE_BRANCH,
            "toolkit_core_commit": CORE_COMMIT,
            "toolkit_core_clean": True,
            "toolkit_core_version": CORE_VERSION,
            "toolkit_ops_commit": OPS_COMMIT,
            "toolkit_ops_version": OPS_VERSION,
        }
        for field, expected_value in expected_runtime.items():
            if value.get(field) != expected_value:
                raise AssemblyError(
                    f"stock Toolkit runtime field {field} differs in {path}"
                )
        if value["purpose"] == "smoke":
            smoke_records.append((path, value))
        else:
            records.append((path, value))
    if not records:
        raise AssemblyError(f"no measured campaign records found in {root}")
    if len(smoke_records) != 1:
        raise AssemblyError(
            f"{root} must contain exactly one purpose=smoke acceptance record"
        )
    smoke_path, smoke_record = smoke_records[0]
    if (
        smoke_record.get("success") is not True
        or smoke_record.get("status") != "complete"
        or smoke_record.get("correctness_passed") is not True
    ):
        raise AssemblyError(
            f"smoke acceptance must be successful and complete: {smoke_path}"
        )
    if smoke_record.get("repeat") != 0:
        raise AssemblyError(f"smoke acceptance must use repeat 0: {smoke_path}")
    if smoke_record.get("systems_requested") != DEFAULT_SYSTEMS:
        raise AssemblyError(
            f"smoke acceptance must cover all {DEFAULT_SYSTEMS} systems: {smoke_path}"
        )
    return records, smoke_records[0], producer_files


def _one_common(records: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = {_canonical(record[field]) for record in records}
    if len(values) != 1:
        raise AssemblyError(f"raw campaign field {field!r} differs between records")
    return records[0][field]


def _successful_common(records: Sequence[Mapping[str, Any]], field: str) -> Any:
    successful = [record for record in records if record["success"]]
    if not successful:
        raise AssemblyError("no successful record is available to identify the campaign")
    return _one_common(successful, field)


def _count_or_empty(value: Any, field: str, source: Path) -> int | str:
    if value is None:
        return ""
    if not isinstance(value, list):
        raise AssemblyError(f"{source}: {field} must be a list or null")
    return len(value)


def _csv_row(path: Path, root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    row = {column: record.get(column, "") for column in RUN_COLUMNS}
    row["source_artifact"] = (
        Path(root.name) / path.relative_to(root)
    ).as_posix()
    row["error_type"] = record.get("error_type") or ""
    row["error"] = record.get("error") or ""
    for field in ("missing_systems", "duplicate_systems", "unexpected_systems"):
        row[field] = _count_or_empty(record.get(field), field, path)
    for field, value in list(row.items()):
        if value is None:
            row[field] = ""
    return row


def _acceptance_summary(
    path: Path, root: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    count_fields = (
        "missing_systems",
        "duplicate_systems",
        "unexpected_systems",
    )
    counts: dict[str, int] = {}
    for field in count_fields:
        value = _count_or_empty(record[field], field, path)
        if not isinstance(value, int):
            raise AssemblyError(f"successful smoke acceptance is missing {field}")
        counts[field] = value
    return {
        "stock_toolkit_record_sha256": toolkit_runtime_record_sha256(record),
        "correctness_contract_sha256": correctness_record_sha256(
            record["correctness_checks"], record["correctness_tolerances"]
        ),
        "correctness_passed": record["correctness_passed"],
        "covalent_oh_gate_passed": record["covalent_oh_gate_passed"],
        "duplicate_systems": counts["duplicate_systems"],
        "error": record["error"],
        "error_type": record["error_type"],
        "gpu_count": record["gpu_count"],
        "max_abs_net_charge_e": record["max_abs_net_charge_e"],
        "max_charge_difference_e": record["max_charge_difference_e"],
        "max_energy_difference_ev": record["max_energy_difference_ev"],
        "max_force_difference_ev_per_a": record[
            "max_force_difference_ev_per_a"
        ],
        "max_handoff_fmax_ev_per_a": record["max_handoff_fmax_ev_per_a"],
        "max_relaxation_steps_observed": record[
            "max_relaxation_steps_observed"
        ],
        "min_interatomic_distance_a": record["min_interatomic_distance_a"],
        "missing_systems": counts["missing_systems"],
        "model_record_sha256": canonical_json_sha256(record["model"]),
        "nodes": record["nodes"],
        "nve_steps_verified": record["nve_steps_verified"],
        "nvt_steps_verified": record["nvt_steps_verified"],
        "oxygen_connectivity_gate_passed": record[
            "oxygen_connectivity_gate_passed"
        ],
        "pipeline_count": record["pipeline_count"],
        "producer_set_sha256": producer_set_sha256(record["producer_set"]),
        "purpose": record["purpose"],
        "rank_count": record["rank_count"],
        "raw_record_sha256": sha256_file(path),
        "repeat": record["repeat"],
        "route": record["route"],
        "run_id": record["run_id"],
        "runtime_identity_sha256": runtime_identity_sha256(record),
        "schema": record["schema"],
        "slurm_job_id": str(record["slurm_job_id"]),
        "source_artifact": (Path(root.name) / path.relative_to(root)).as_posix(),
        "stage_1_completions": record["stage_1_completions"],
        "stage_2_completions": record["stage_2_completions"],
        "status": record["status"],
        "success": record["success"],
        "systems_completed": record["systems_completed"],
        "systems_requested": record["systems_requested"],
        "timestamp_utc": record["timestamp_utc"],
        "unexpected_systems": counts["unexpected_systems"],
        "unique_systems_completed": record["unique_systems_completed"],
        "workload_record_sha256": canonical_json_sha256(record["workload"]),
    }


def assemble_pipeline_campaign(
    input_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    generated_at_utc: str | None = None,
) -> Path:
    """Seal three fetched job directories into one checked result bundle."""

    roots = [Path(path).expanduser().resolve() for path in input_dirs]
    if len(roots) != 3 or len(set(roots)) != 3:
        raise AssemblyError("provide exactly three distinct campaign job directories")
    if len({root.name for root in roots}) != len(roots):
        raise AssemblyError("campaign job directory names must be unique")

    artifacts: list[tuple[Path, Path, Mapping[str, Any]]] = []
    acceptance_artifacts: list[tuple[Path, Path, Mapping[str, Any]]] = []
    producer_sets: list[dict[str, str]] = []
    for root in roots:
        records, (smoke_path, smoke_record), producer_set = _load_job_records(root)
        job_routes = {record["route"] for _, record in records}
        if len(job_routes) != 1:
            raise AssemblyError(f"campaign job contains more than one route: {root}")
        job_route = job_routes.pop()
        if smoke_record["route"] != job_route:
            raise AssemblyError(f"smoke acceptance route differs from its job: {root}")
        measured_job_ids = {str(record["slurm_job_id"]) for _, record in records}
        if measured_job_ids != {str(smoke_record["slurm_job_id"])}:
            raise AssemblyError(f"smoke acceptance job identity differs in {root}")
        artifacts.extend((root, path, record) for path, record in records)
        acceptance_artifacts.append((root, smoke_path, smoke_record))
        producer_sets.append(producer_set)
    if len({_canonical(value) for value in producer_sets}) != 1:
        raise AssemblyError("producer set differs between campaign jobs")

    records = [record for _, _, record in artifacts]
    acceptance_records = [record for _, _, record in acceptance_artifacts]
    all_records = [*records, *acceptance_records]
    expected_matrix = {
        (route, repeat)
        for route in ROUTE_SPECS
        for repeat in range(1, REQUIRED_REPEATS + 1)
    }
    observed_matrix = [(record["route"], record["repeat"]) for record in records]
    if len(observed_matrix) != len(set(observed_matrix)):
        raise AssemblyError("raw campaign contains duplicate route/repeat cases")
    if set(observed_matrix) != expected_matrix:
        missing = sorted(expected_matrix - set(observed_matrix))
        extra = sorted(set(observed_matrix) - expected_matrix)
        raise AssemblyError(
            "raw campaign does not contain the requested full matrix "
            f"(missing={missing}; extra={extra})"
        )

    for field in (
        "backend",
        "gpu_name",
        "partition",
        "producer_set",
        "python_hash_seed",
        "python_version",
        "repository_commit",
        "systems_requested",
        "timing_boundary",
        "toolkit_core_commit",
        "toolkit_core_branch",
        "toolkit_core_clean",
        "toolkit_core_version",
        "toolkit_ops_commit",
        "toolkit_ops_version",
        "torch_version",
    ):
        _one_common(all_records, field)
    for field in ("model", "workload", "correctness_checks", "correctness_tolerances"):
        _successful_common(all_records, field)

    workload = _successful_common(all_records, "workload")
    workload_without_fingerprint = dict(workload)
    workload_without_fingerprint.pop("campaign_definition_sha256", None)
    for record in records:
        candidate = record["workload"]
        if not isinstance(candidate, Mapping):
            raise AssemblyError("every raw record must retain its workload settings")
        candidate_without_fingerprint = dict(candidate)
        candidate_without_fingerprint.pop("campaign_definition_sha256", None)
        if candidate_without_fingerprint != workload_without_fingerprint:
            raise AssemblyError("workload settings differ between raw records")

    job_ids: dict[str, str] = {}
    for route in ROUTE_SPECS:
        ids = {str(record["slurm_job_id"]) for record in records if record["route"] == route}
        if len(ids) != 1:
            raise AssemblyError(f"route {route} spans more than one Slurm job")
        job_ids[route] = ids.pop()
    if len(set(job_ids.values())) != len(job_ids):
        raise AssemblyError("each route must come from a distinct Slurm job")

    acceptance_summaries = [
        _acceptance_summary(path, root, record)
        for root, path, record in acceptance_artifacts
    ]
    acceptance_summaries.sort(
        key=lambda summary: list(ROUTE_SPECS).index(summary["route"])
    )

    rows = [
        _csv_row(path, root, record) for root, path, record in artifacts
    ]
    rows.sort(key=lambda row: (list(ROUTE_SPECS).index(row["route"]), row["repeat"]))
    table = pd.DataFrame(rows, columns=RUN_COLUMNS)
    publishable = bool(table["success"].all())

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite campaign bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        runs_path = staging / "runs.csv"
        table.to_csv(runs_path, index=False, lineterminator="\n")
        runs_sha256 = sha256_file(runs_path)
        producer_files = producer_sets[0]
        manifest = {
            "acceptance": {
                "performance_metrics_used": False,
                "purpose": "smoke",
                "required_successes_per_route": 1,
                "runs": acceptance_summaries,
                "systems_total": DEFAULT_SYSTEMS,
            },
            "artifact_id": f"pipeline-campaign-{runs_sha256[:16]}",
            "campaign": {
                "model": _successful_common(all_records, "model"),
                "repeats": REQUIRED_REPEATS,
                "routes": [
                    {"id": route, **spec} for route, spec in ROUTE_SPECS.items()
                ],
                "systems_total": _one_common(records, "systems_requested"),
                "timing_boundary": _one_common(records, "timing_boundary"),
                "workload": workload,
            },
            "correctness": {
                "reference_route": "fused_1gpu",
                "required_checks": _successful_common(
                    all_records, "correctness_checks"
                ),
                **_successful_common(all_records, "correctness_tolerances"),
            },
            "data": {
                "bytes": runs_path.stat().st_size,
                "columns": list(RUN_COLUMNS),
                "file": "runs.csv",
                "row_count": len(table),
                "sha256": runs_sha256,
            },
            "failure_policy": {
                "failed_rows_retained": True,
                "required_failure_fields": ["success", "error_type", "error"],
            },
            "integrity": {
                "checksum_index": "SHA256SUMS",
                "checksum_index_covers_manifest": True,
            },
            "provenance": {
                "backend": _one_common(records, "backend"),
                "generated_at_utc": generated_at_utc or _utc_now(),
                "gpu_name": _one_common(records, "gpu_name"),
                "partition": _one_common(records, "partition"),
                "producer_files_sha256": producer_files,
                "producer_set_sha256": producer_set_sha256(producer_files),
                "python_hash_seed": _one_common(records, "python_hash_seed"),
                "python_version": _one_common(records, "python_version"),
                "repository_commit": _one_common(records, "repository_commit"),
                "site": "Compute Lab",
                "slurm_job_ids_by_route": job_ids,
                "toolkit_core_branch": _one_common(
                    records, "toolkit_core_branch"
                ),
                "toolkit_core_clean": _one_common(records, "toolkit_core_clean"),
                "toolkit_core_commit": _one_common(records, "toolkit_core_commit"),
                "toolkit_core_version": _one_common(
                    records, "toolkit_core_version"
                ),
                "toolkit_ops_commit": _one_common(records, "toolkit_ops_commit"),
                "toolkit_ops_version": _one_common(
                    records, "toolkit_ops_version"
                ),
                "torch_version": _one_common(records, "torch_version"),
            },
            "publication": {
                "ready": publishable,
                "successful_repeats_required_per_route": REQUIRED_REPEATS,
            },
            "schema": BUNDLE_SCHEMA,
            "scope": {
                "fixed_workload_speed_benchmark": True,
                "learner_use": "Plot saved H100 results without a cluster allocation.",
                "scientific_accuracy_benchmark": False,
            },
            "status": "complete",
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        (staging / "SHA256SUMS").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n"
            f"{runs_sha256}  runs.csv\n",
            encoding="utf-8",
        )
        load_pipeline_campaign_bundle(
            staging,
            require_publishable=publishable,
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs=3, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at-utc")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = assemble_pipeline_campaign(
        args.input_dirs,
        output_dir=args.output_dir,
        generated_at_utc=args.generated_at_utc,
    )
    print(f"sealed pipeline campaign bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
