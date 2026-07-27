"""Load the prerecorded Compute Lab pipeline campaign results.

The campaign compares one fixed workload across three routes: all stages on
one GPU, one two-GPU ``DistributedPipeline``, and one four-GPU
``DistributedPipeline`` containing two independent stage pairs. This module
authenticates the small result bundle and returns the
learner-facing timing metrics.  It does not replace the earlier synthetic
distributed-benchmark loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file


BUNDLE_SCHEMA = "alchemi.pipeline-campaign-bundle.v2"
RUN_SCHEMA = "alchemi.pipeline-campaign-run.v2"
CHECKSUM_INDEX_NAME = "SHA256SUMS"
MANIFEST_NAME = "manifest.json"
RUNS_NAME = "runs.csv"
TOOLKIT_CORE_BRANCH = "0.2.0-rc"
TOOLKIT_CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
TOOLKIT_CORE_VERSION = "0.2.0"
TOOLKIT_OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
TOOLKIT_OPS_VERSION = "0.4.0"
REQUIRED_PRODUCER_FILES = {
    "scripts/benchmark_part1_distributed_campaign.py",
    "scripts/part1_distributed_campaign_contract.py",
    "scripts/record_part1_campaign_failure.py",
    "scripts/run_part1_distributed_torchrun.sh",
    "scripts/slurm_part1_distributed_campaign.sbatch",
    "part-1-scalable-atomistic-workflows/aux/artifacts.py",
    "part-1-scalable-atomistic-workflows/aux/checkpoint.py",
    "part-1-scalable-atomistic-workflows/aux/electrostatics.py",
    "part-1-scalable-atomistic-workflows/aux/hooks.py",
    "part-1-scalable-atomistic-workflows/aux/runtime.py",
    "part-1-scalable-atomistic-workflows/aux/structures.py",
}
MODEL_DTYPE_DESCRIPTION = (
    "float32 positions, velocities, forces, AIMNet, and Coulomb pair math; "
    "float64 Coulomb energy accumulation"
)
FIXED_SYSTEMS_TOTAL = 8192
EXPECTED_D3_BJ_PARAMETERS = {"a1": 0.37, "a2": 4.1, "s6": 1.0, "s8": 1.5}
RUNTIME_IDENTITY_FIELDS = (
    "backend",
    "gpu_name",
    "partition",
    "producer_files_sha256",
    "python_hash_seed",
    "python_version",
    "repository_commit",
    "toolkit_core_branch",
    "toolkit_core_clean",
    "toolkit_core_commit",
    "toolkit_core_version",
    "toolkit_ops_commit",
    "toolkit_ops_version",
    "torch_version",
)
TOOLKIT_RUNTIME_FIELDS = (
    "toolkit_core_branch",
    "toolkit_core_clean",
    "toolkit_core_commit",
    "toolkit_core_version",
    "toolkit_ops_commit",
    "toolkit_ops_version",
)
ACCEPTANCE_RUN_KEYS = {
    "stock_toolkit_record_sha256",
    "correctness_contract_sha256",
    "correctness_passed",
    "covalent_oh_gate_passed",
    "duplicate_systems",
    "error",
    "error_type",
    "gpu_count",
    "max_abs_net_charge_e",
    "max_charge_difference_e",
    "max_energy_difference_ev",
    "max_force_difference_ev_per_a",
    "max_handoff_fmax_ev_per_a",
    "max_relaxation_steps_observed",
    "min_interatomic_distance_a",
    "missing_systems",
    "model_record_sha256",
    "nodes",
    "nve_steps_verified",
    "nvt_steps_verified",
    "oxygen_connectivity_gate_passed",
    "pipeline_count",
    "producer_set_sha256",
    "purpose",
    "rank_count",
    "raw_record_sha256",
    "repeat",
    "route",
    "run_id",
    "runtime_identity_sha256",
    "schema",
    "slurm_job_id",
    "source_artifact",
    "stage_1_completions",
    "stage_2_completions",
    "status",
    "success",
    "systems_completed",
    "systems_requested",
    "timestamp_utc",
    "unexpected_systems",
    "unique_systems_completed",
    "workload_record_sha256",
}

EXPECTED_PAIR_BOUNDARIES = {
    "fused_1gpu": [],
    "pipeline_2gpu": [[0, 1]],
    "pipeline_4gpu": [[0, 1], [2, 3]],
}

ROUTE_SPECS: dict[str, dict[str, int]] = {
    "fused_1gpu": {
        "nodes": 1,
        "gpu_count": 1,
        "rank_count": 1,
        "pipeline_count": 0,
    },
    "pipeline_2gpu": {
        "nodes": 2,
        "gpu_count": 2,
        "rank_count": 2,
        "pipeline_count": 1,
    },
    "pipeline_4gpu": {
        "nodes": 4,
        "gpu_count": 4,
        "rank_count": 4,
        "pipeline_count": 2,
    },
}

RUN_COLUMNS = (
    "schema",
    "run_id",
    "timestamp_utc",
    "slurm_job_id",
    "source_artifact",
    "route",
    "repeat",
    "success",
    "status",
    "error_type",
    "error",
    "nodes",
    "gpu_count",
    "rank_count",
    "pipeline_count",
    "systems_requested",
    "systems_completed",
    "unique_systems_completed",
    "missing_systems",
    "duplicate_systems",
    "unexpected_systems",
    "stage_1_completions",
    "stage_2_completions",
    "correctness_passed",
    "max_energy_difference_ev",
    "max_force_difference_ev_per_a",
    "max_charge_difference_e",
    "max_abs_net_charge_e",
    "max_handoff_fmax_ev_per_a",
    "min_interatomic_distance_a",
    "max_relaxation_steps_observed",
    "nvt_steps_verified",
    "nve_steps_verified",
    "covalent_oh_gate_passed",
    "oxygen_connectivity_gate_passed",
    "elapsed_s",
    "systems_per_s",
    "peak_memory_bytes_max_rank",
)

_INTEGER_COLUMNS = (
    "repeat",
    "nodes",
    "gpu_count",
    "rank_count",
    "pipeline_count",
    "systems_requested",
)
_NULLABLE_INTEGER_COLUMNS = (
    "systems_completed",
    "unique_systems_completed",
    "missing_systems",
    "duplicate_systems",
    "unexpected_systems",
    "stage_1_completions",
    "stage_2_completions",
    "max_relaxation_steps_observed",
    "peak_memory_bytes_max_rank",
)
_NULLABLE_BOOL_COLUMNS = (
    "correctness_passed",
    "nvt_steps_verified",
    "nve_steps_verified",
    "covalent_oh_gate_passed",
    "oxygen_connectivity_gate_passed",
)
_FLOAT_COLUMNS = (
    "max_energy_difference_ev",
    "max_force_difference_ev_per_a",
    "max_charge_difference_e",
    "max_abs_net_charge_e",
    "max_handoff_fmax_ev_per_a",
    "min_interatomic_distance_a",
    "elapsed_s",
    "systems_per_s",
)

DEFAULT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "compute_lab_pipeline_campaign"
)


class PipelineCampaignError(ValueError):
    """Raised when a pipeline campaign bundle has an invalid format."""


def producer_set_sha256(files_sha256: Mapping[str, str]) -> str:
    """Return one stable digest for a path-to-SHA-256 producer mapping."""

    canonical = json.dumps(
        dict(files_sha256),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def toolkit_runtime_record_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact stock Toolkit Core and Ops identity."""

    return canonical_json_sha256(
        {field: value[field] for field in TOOLKIT_RUNTIME_FIELDS}
    )


def runtime_identity_sha256(value: Mapping[str, Any]) -> str:
    """Hash the runtime and producer fields shared by every route job."""

    identity = {
        field: value[field]
        for field in RUNTIME_IDENTITY_FIELDS
        if field != "producer_files_sha256"
    }
    identity["producer_files_sha256"] = value.get(
        "producer_files_sha256", value.get("producer_set")
    )
    return canonical_json_sha256(identity)


def correctness_record_sha256(
    checks: list[str], tolerances: Mapping[str, Any]
) -> str:
    """Hash the raw correctness checks and tolerances as one settings record."""

    return canonical_json_sha256(
        {"checks": checks, "tolerances": dict(tolerances)}
    )


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    observed = set(value)
    if observed == expected:
        return
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise PipelineCampaignError(
        f"{field} has the wrong keys ({'; '.join(details)})"
    )


def _require_utc_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineCampaignError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PipelineCampaignError(f"{field} must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PipelineCampaignError(f"{field} must be a UTC timestamp")
    return value


def _require_safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PipelineCampaignError(f"{field} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PipelineCampaignError(f"{field} must be a safe relative path")
    return value


@dataclass(frozen=True)
class PipelineCampaignBundle:
    """Verified campaign rows and a plot-ready route summary."""

    root: Path
    manifest: Mapping[str, Any]
    runs: pd.DataFrame

    @property
    def run_details(self) -> Mapping[str, Any]:
        """Return the recorded software, hardware, and job details."""

        details = self.manifest["provenance"]
        assert isinstance(details, Mapping)
        return details

    @property
    def failed_runs(self) -> pd.DataFrame:
        """Return retained failures without changing their recorded fields."""

        return self.runs.loc[~self.runs["success"]].reset_index(drop=True).copy()

    @property
    def summary(self) -> pd.DataFrame:
        """Return medians, speedup, efficiency, and timed GPU cost by route."""

        return summarize_pipeline_campaign(self.runs)


def _load_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise PipelineCampaignError(f"missing checksum index: {path}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise PipelineCampaignError("malformed SHA256SUMS")
        digest, name = fields
        checksums[name.removeprefix("*")] = digest.lower()
    if set(checksums) != {MANIFEST_NAME, RUNS_NAME}:
        raise PipelineCampaignError(
            "SHA256SUMS must cover exactly manifest.json and runs.csv"
        )
    return checksums


def _verify_file(path: Path, *, name: str, checksums: Mapping[str, str]) -> str:
    if not path.is_file():
        raise PipelineCampaignError(f"missing bundle file: {path}")
    digest = sha256_file(path)
    if checksums[name] != digest:
        raise PipelineCampaignError(f"SHA-256 mismatch for {name}")
    return digest


def _load_runs(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise PipelineCampaignError(f"cannot read runs.csv: {exc}") from exc
    if tuple(raw.columns) != RUN_COLUMNS:
        raise PipelineCampaignError(
            "runs.csv columns must exactly match the campaign run schema"
        )
    if raw.empty:
        raise PipelineCampaignError("runs.csv must not be empty")

    runs = raw.copy()
    success = raw["success"].str.casefold()
    if not success.isin({"true", "false"}).all():
        raise PipelineCampaignError("runs.csv success values must be booleans")
    runs["success"] = success.eq("true")

    for column in _NULLABLE_BOOL_COLUMNS:
        values = raw[column].str.casefold()
        if not values.isin({"", "true", "false"}).all():
            raise PipelineCampaignError(
                f"runs.csv {column} values must be booleans or empty"
            )
        runs[column] = values.map(
            {"": pd.NA, "true": True, "false": False}
        ).astype("boolean")
    for column in _INTEGER_COLUMNS:
        if not raw[column].str.fullmatch(r"[0-9]+").all():
            raise PipelineCampaignError(
                f"runs.csv {column} values must be nonnegative integers"
            )
        runs[column] = raw[column].astype("int64")
    for column in _FLOAT_COLUMNS:
        values = pd.to_numeric(raw[column].replace("", pd.NA), errors="coerce")
        invalid = (raw[column] != "") & (values.isna() | ~np.isfinite(values))
        if invalid.any():
            raise PipelineCampaignError(
                f"runs.csv {column} values must be finite numbers or empty"
            )
        runs[column] = values.astype("Float64")
    for column in _NULLABLE_INTEGER_COLUMNS:
        invalid = (raw[column] != "") & ~raw[column].str.fullmatch(r"[0-9]+")
        if invalid.any():
            raise PipelineCampaignError(
                f"runs.csv {column} values must be nonnegative integers or empty"
            )
        runs[column] = pd.to_numeric(
            raw[column].replace("", pd.NA), errors="raise"
        ).astype("Int64")

    successful = runs["success"]
    runs["gpu_seconds_per_structure"] = pd.Series(
        pd.NA, index=runs.index, dtype="Float64"
    )
    runs.loc[successful, "gpu_seconds_per_structure"] = (
        runs.loc[successful, "elapsed_s"].astype(float)
        * runs.loc[successful, "gpu_count"].astype(float)
        / runs.loc[successful, "systems_requested"].astype(float)
    )
    return runs


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PipelineCampaignError(f"{field} must be a SHA-256 digest")
    return value


def _validate_acceptance(
    acceptance: Any,
    *,
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    correctness_checks: list[str],
    correctness_tolerances: Mapping[str, Any],
    result_tolerances: Mapping[str, float],
    minimum_distance_a: float,
    provenance: Mapping[str, Any],
    job_ids: Mapping[str, str],
) -> None:
    if not isinstance(acceptance, Mapping):
        raise PipelineCampaignError("manifest acceptance must be an object")
    _require_exact_keys(
        acceptance,
        {
            "performance_metrics_used",
            "purpose",
            "required_successes_per_route",
            "runs",
            "systems_total",
        },
        "manifest acceptance",
    )
    if acceptance["purpose"] != "smoke":
        raise PipelineCampaignError("acceptance purpose must be smoke")
    if acceptance["performance_metrics_used"] is not False:
        raise PipelineCampaignError(
            "acceptance runs must not contribute performance metrics"
        )
    if (
        isinstance(acceptance["required_successes_per_route"], bool)
        or acceptance["required_successes_per_route"] != 1
    ):
        raise PipelineCampaignError(
            "acceptance requires exactly one successful run per route"
        )
    if acceptance["systems_total"] != FIXED_SYSTEMS_TOTAL:
        raise PipelineCampaignError(
            f"acceptance must cover the full {FIXED_SYSTEMS_TOTAL}-system campaign"
        )
    acceptance_runs = acceptance["runs"]
    if not isinstance(acceptance_runs, list):
        raise PipelineCampaignError("acceptance runs must be a list")

    expected_identities = {
        "stock_toolkit_record_sha256": toolkit_runtime_record_sha256(provenance),
        "correctness_contract_sha256": correctness_record_sha256(
            correctness_checks, correctness_tolerances
        ),
        "model_record_sha256": canonical_json_sha256(model),
        "producer_set_sha256": producer_set_sha256(
            provenance["producer_files_sha256"]
        ),
        "runtime_identity_sha256": runtime_identity_sha256(provenance),
        "workload_record_sha256": canonical_json_sha256(workload),
    }
    routes: list[str] = []
    run_ids: list[str] = []
    for index, run in enumerate(acceptance_runs):
        field = f"acceptance run {index}"
        if not isinstance(run, Mapping):
            raise PipelineCampaignError(f"{field} must be an object")
        _require_exact_keys(run, ACCEPTANCE_RUN_KEYS, field)
        route = run["route"]
        if not isinstance(route, str) or route not in ROUTE_SPECS:
            raise PipelineCampaignError(f"{field} route is unsupported")
        routes.append(route)
        if run["schema"] != RUN_SCHEMA:
            raise PipelineCampaignError(f"{field} schema is incorrect")
        if run["purpose"] != "smoke" or run["repeat"] != 0:
            raise PipelineCampaignError(
                f"{field} must be the route's purpose=smoke repeat 0 record"
            )
        if run["success"] is not True or run["status"] != "complete":
            raise PipelineCampaignError(f"{field} must be successful and complete")
        if run["error_type"] is not None or run["error"] is not None:
            raise PipelineCampaignError(f"{field} cannot contain an error")
        run_id = run["run_id"]
        if not isinstance(run_id, str) or not run_id.strip():
            raise PipelineCampaignError(f"{field} run_id must be nonempty")
        run_ids.append(run_id)
        _require_utc_timestamp(run["timestamp_utc"], f"{field} timestamp_utc")
        _require_safe_relative_path(
            run["source_artifact"], f"{field} source_artifact"
        )
        _require_sha256(run["raw_record_sha256"], f"{field} raw_record_sha256")
        if run["slurm_job_id"] != job_ids[route]:
            raise PipelineCampaignError(
                f"{field} slurm_job_id does not match its measured route"
            )
        for topology_field, expected in ROUTE_SPECS[route].items():
            value = run[topology_field]
            if isinstance(value, bool) or value != expected:
                raise PipelineCampaignError(
                    f"{field} {topology_field} does not match its route"
                )
        for identity_field, expected in expected_identities.items():
            _require_sha256(run[identity_field], f"{field} {identity_field}")
            if run[identity_field] != expected:
                label = identity_field.replace("_record_sha256", "").replace(
                    "_sha256", ""
                )
                raise PipelineCampaignError(
                    f"{field} {label} identity does not match the measured campaign"
                )

        expected_counts = {
            "systems_requested": FIXED_SYSTEMS_TOTAL,
            "systems_completed": FIXED_SYSTEMS_TOTAL,
            "unique_systems_completed": FIXED_SYSTEMS_TOTAL,
            "missing_systems": 0,
            "duplicate_systems": 0,
            "unexpected_systems": 0,
            "stage_1_completions": FIXED_SYSTEMS_TOTAL,
            "stage_2_completions": FIXED_SYSTEMS_TOTAL,
        }
        for count_field, expected in expected_counts.items():
            value = run[count_field]
            if isinstance(value, bool) or value != expected:
                raise PipelineCampaignError(
                    f"{field} {count_field} does not prove the full campaign"
                )
        for boolean_field in (
            "correctness_passed",
            "nvt_steps_verified",
            "nve_steps_verified",
            "covalent_oh_gate_passed",
            "oxygen_connectivity_gate_passed",
        ):
            if run[boolean_field] is not True:
                raise PipelineCampaignError(f"{field} {boolean_field} must be true")
        for result_field, tolerance in result_tolerances.items():
            value = run[result_field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0.0 <= float(value) <= tolerance
            ):
                raise PipelineCampaignError(
                    f"{field} {result_field} exceeds the correctness tolerance"
                )
        minimum_distance = run["min_interatomic_distance_a"]
        if (
            isinstance(minimum_distance, bool)
            or not isinstance(minimum_distance, (int, float))
            or not np.isfinite(minimum_distance)
            or float(minimum_distance) < minimum_distance_a
        ):
            raise PipelineCampaignError(
                f"{field} min_interatomic_distance_a is below the required minimum"
            )
        relaxation_steps = run["max_relaxation_steps_observed"]
        if (
            isinstance(relaxation_steps, bool)
            or not isinstance(relaxation_steps, int)
            or relaxation_steps < 0
        ):
            raise PipelineCampaignError(
                f"{field} max_relaxation_steps_observed must be nonnegative"
            )
    if set(routes) != set(ROUTE_SPECS) or len(routes) != len(ROUTE_SPECS):
        raise PipelineCampaignError(
            "acceptance runs must contain exactly one successful run for every route"
        )
    if len(run_ids) != len(set(run_ids)):
        raise PipelineCampaignError("acceptance run_id values must be unique")


def load_pipeline_campaign_bundle(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    *,
    require_publishable: bool = True,
) -> PipelineCampaignBundle:
    """Authenticate and load a prerecorded pipeline campaign bundle.

    Set ``require_publishable=False`` only to inspect a complete diagnostic
    bundle that retained a failed measured repeat.
    """

    root = Path(bundle_dir).resolve()
    checksums = _load_checksums(root / CHECKSUM_INDEX_NAME)
    manifest_path = root / MANIFEST_NAME
    runs_path = root / RUNS_NAME
    _verify_file(manifest_path, name=MANIFEST_NAME, checksums=checksums)
    runs_sha256 = _verify_file(runs_path, name=RUNS_NAME, checksums=checksums)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise PipelineCampaignError(f"cannot read manifest.json: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise PipelineCampaignError("manifest root must be an object")
    _require_exact_keys(
        manifest,
        {
            "acceptance",
            "artifact_id",
            "campaign",
            "correctness",
            "data",
            "failure_policy",
            "integrity",
            "publication",
            "provenance",
            "schema",
            "scope",
            "status",
        },
        "manifest root",
    )
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise PipelineCampaignError("unexpected pipeline campaign bundle schema")
    if manifest.get("status") != "complete":
        raise PipelineCampaignError("pipeline campaign bundle is not complete")
    if manifest.get("artifact_id") != f"pipeline-campaign-{runs_sha256[:16]}":
        raise PipelineCampaignError("artifact_id does not match runs.csv")

    data = manifest.get("data")
    if not isinstance(data, Mapping):
        raise PipelineCampaignError("manifest data must be an object")
    if data.get("file") != RUNS_NAME or data.get("sha256") != runs_sha256:
        raise PipelineCampaignError("manifest data identity does not match runs.csv")
    if data.get("columns") != list(RUN_COLUMNS):
        raise PipelineCampaignError("manifest data columns do not match runs.csv")
    if data.get("bytes") != runs_path.stat().st_size:
        raise PipelineCampaignError("manifest runs.csv byte count is inconsistent")

    runs = _load_runs(runs_path)
    if data.get("row_count") != len(runs):
        raise PipelineCampaignError("manifest row count does not match runs.csv")
    publication = manifest.get("publication")
    if not isinstance(publication, Mapping):
        raise PipelineCampaignError("manifest publication must be an object")
    _require_exact_keys(
        publication,
        {"ready", "successful_repeats_required_per_route"},
        "manifest publication",
    )
    if publication["successful_repeats_required_per_route"] != 5:
        raise PipelineCampaignError(
            "publication requires exactly five successful repeats per route"
        )
    if not isinstance(publication["ready"], bool):
        raise PipelineCampaignError("publication ready must be a boolean")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PipelineCampaignError("manifest provenance must be an object")
    _require_exact_keys(
        provenance,
        {
            "backend",
            "generated_at_utc",
            "gpu_name",
            "partition",
            "producer_files_sha256",
            "producer_set_sha256",
            "python_hash_seed",
            "python_version",
            "repository_commit",
            "site",
            "slurm_job_ids_by_route",
            "toolkit_core_branch",
            "toolkit_core_clean",
            "toolkit_core_commit",
            "toolkit_core_version",
            "toolkit_ops_commit",
            "toolkit_ops_version",
            "torch_version",
        },
        "manifest provenance",
    )
    expected_toolkit_runtime = {
        "toolkit_core_branch": TOOLKIT_CORE_BRANCH,
        "toolkit_core_clean": True,
        "toolkit_core_commit": TOOLKIT_CORE_COMMIT,
        "toolkit_core_version": TOOLKIT_CORE_VERSION,
        "toolkit_ops_commit": TOOLKIT_OPS_COMMIT,
        "toolkit_ops_version": TOOLKIT_OPS_VERSION,
    }
    for field, expected in expected_toolkit_runtime.items():
        if provenance[field] != expected:
            raise PipelineCampaignError(
                f"provenance {field} does not match the required stock Toolkit runtime"
            )
    producer_files = provenance.get("producer_files_sha256")
    if not isinstance(producer_files, Mapping) or not producer_files:
        raise PipelineCampaignError(
            "provenance producer_files_sha256 must be a nonempty object"
        )
    if set(producer_files) != REQUIRED_PRODUCER_FILES:
        raise PipelineCampaignError(
            "provenance producer_files_sha256 must cover the exact campaign "
            "producer and helper set"
        )
    for relative_path, digest in producer_files.items():
        _require_safe_relative_path(
            relative_path, "provenance producer_files_sha256 path"
        )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PipelineCampaignError(
                "provenance producer_files_sha256 has an invalid digest"
            )
    if provenance.get("producer_set_sha256") != producer_set_sha256(producer_files):
        raise PipelineCampaignError(
            "provenance producer_set_sha256 does not match producer_files_sha256"
        )
    if provenance["site"] != "Compute Lab":
        raise PipelineCampaignError("provenance site must be Compute Lab")
    if provenance["backend"] != "nccl":
        raise PipelineCampaignError("provenance backend must be nccl")
    for field in ("gpu_name", "partition"):
        value = provenance[field]
        if not isinstance(value, str) or "H100" not in value.upper():
            raise PipelineCampaignError(f"provenance {field} must identify H100")
    if provenance["python_hash_seed"] != "0":
        raise PipelineCampaignError("provenance python_hash_seed must be '0'")
    for field in ("python_version", "torch_version"):
        value = provenance[field]
        if not isinstance(value, str) or not value.strip():
            raise PipelineCampaignError(f"provenance {field} must be nonempty text")
    for field in ("repository_commit", "toolkit_core_commit", "toolkit_ops_commit"):
        value = provenance[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise PipelineCampaignError(f"provenance {field} must be a git commit")
    _require_utc_timestamp(provenance["generated_at_utc"], "generated_at_utc")
    job_ids = provenance.get("slurm_job_ids_by_route")
    if not isinstance(job_ids, Mapping) or set(job_ids) != set(ROUTE_SPECS):
        raise PipelineCampaignError(
            "provenance slurm_job_ids_by_route must cover every route"
        )
    if any(
        not isinstance(job_id, str)
        or re.fullmatch(r"[1-9][0-9]*", job_id) is None
        for job_id in job_ids.values()
    ):
        raise PipelineCampaignError("provenance SLURM job IDs must be positive integers")
    if len(set(job_ids.values())) != len(job_ids):
        raise PipelineCampaignError("each route must use a distinct SLURM job")
    campaign = manifest.get("campaign")
    if not isinstance(campaign, Mapping):
        raise PipelineCampaignError("manifest campaign must be an object")
    _require_exact_keys(
        campaign,
        {
            "model",
            "repeats",
            "routes",
            "systems_total",
            "timing_boundary",
            "workload",
        },
        "manifest campaign",
    )
    expected_routes = [
        {"id": route, **spec} for route, spec in ROUTE_SPECS.items()
    ]
    if campaign.get("routes") != expected_routes:
        raise PipelineCampaignError(
            "campaign routes do not match the three supported topologies"
        )
    timing_boundary = campaign.get("timing_boundary")
    if not isinstance(timing_boundary, str) or not timing_boundary.strip():
        raise PipelineCampaignError("campaign timing_boundary must be nonempty text")
    model = campaign.get("model")
    if not isinstance(model, Mapping):
        raise PipelineCampaignError("campaign model must be an object")
    _require_exact_keys(
        model,
        {
            "checkpoint_sha256",
            "checkpoint_source",
            "components",
            "d3_bj_parameters",
            "d3_parameter_sha256",
            "dtype",
            "eager",
        },
        "campaign model",
    )
    for field in ("checkpoint_sha256", "d3_parameter_sha256"):
        if (
            not isinstance(model[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", model[field]) is None
        ):
            raise PipelineCampaignError(
                f"campaign model {field} must be a SHA-256 digest"
            )
    if not isinstance(model["checkpoint_source"], str) or not model[
        "checkpoint_source"
    ].strip():
        raise PipelineCampaignError(
            "campaign model checkpoint_source must be nonempty text"
        )
    if model["dtype"] != MODEL_DTYPE_DESCRIPTION or model["eager"] is not True:
        raise PipelineCampaignError(
            "campaign model must declare the exact eager mixed-precision "
            "evaluation used by the measured driver"
        )
    d3_bj_parameters = model["d3_bj_parameters"]
    if (
        not isinstance(d3_bj_parameters, Mapping)
        or set(d3_bj_parameters) != set(EXPECTED_D3_BJ_PARAMETERS)
        or any(
            isinstance(d3_bj_parameters[key], bool)
            or not isinstance(d3_bj_parameters[key], (int, float))
            or not np.isfinite(d3_bj_parameters[key])
            or not np.isclose(
                float(d3_bj_parameters[key]), expected, rtol=0.0, atol=1.0e-15
            )
            for key, expected in EXPECTED_D3_BJ_PARAMETERS.items()
        )
    ):
        raise PipelineCampaignError(
            "campaign model D3(BJ) parameters do not match the checkpoint card"
        )
    expected_components = [
        "AIMNet2 B97-3c residual",
        "finite all-pairs Coulomb",
        "pairwise D3(BJ)",
    ]
    if model["components"] != expected_components:
        raise PipelineCampaignError(
            "campaign model components do not match the validated potential"
        )
    workload = campaign.get("workload")
    if not isinstance(workload, Mapping):
        raise PipelineCampaignError("campaign workload must be an object")
    _require_exact_keys(
        workload,
        {
            "atoms_per_system",
            "batch_size",
            "campaign_definition_sha256",
            "campaign_seed",
            "comm_mode",
            "dt_fs",
            "fire_fmax_ev_per_a",
            "friction_per_fs",
            "nve_steps",
            "nvt_steps",
            "pipeline_pair_boundaries",
            "perturbation_description",
            "pipeline_partition_rule",
            "source_structure",
            "stage_names",
            "structure_builder_file",
            "structure_builder_sha256",
            "systems_total",
            "temperature_k",
            "velocity_seed_rule",
        },
        "campaign workload",
    )
    for field in ("campaign_definition_sha256", "structure_builder_sha256"):
        if (
            not isinstance(workload.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", workload[field]) is None
        ):
            raise PipelineCampaignError(
                f"campaign workload {field} must be a SHA-256 digest"
            )
    for field in (
        "atoms_per_system",
        "batch_size",
        "systems_total",
        "nvt_steps",
        "nve_steps",
    ):
        value = workload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PipelineCampaignError(
                f"campaign workload {field} must be a positive integer"
            )
    if workload["campaign_seed"] != 20260714:
        raise PipelineCampaignError(
            "campaign workload campaign_seed does not match the fixed campaign"
        )
    if workload["velocity_seed_rule"] != "910000 + campaign_id":
        raise PipelineCampaignError(
            "campaign workload velocity_seed_rule does not match the fixed campaign"
        )
    if workload["pipeline_partition_rule"] != (
        "4 GPUs: campaign_id % 2; 1 or 2 GPUs: all campaign IDs"
    ):
        raise PipelineCampaignError(
            "campaign workload pipeline_partition_rule does not match the fixed "
            "campaign"
        )
    if workload["pipeline_pair_boundaries"] != EXPECTED_PAIR_BOUNDARIES:
        raise PipelineCampaignError(
            "campaign workload pipeline_pair_boundaries must keep ranks 0-1 and "
            "2-3 as separate four-GPU streams"
        )
    if workload["source_structure"] != "generated cyclic water hexamer":
        raise PipelineCampaignError(
            "campaign workload source_structure does not match the fixed campaign"
        )
    _require_safe_relative_path(
        workload["structure_builder_file"],
        "campaign workload structure_builder_file",
    )
    if workload["comm_mode"] not in {"sync", "async_recv", "fully_async"}:
        raise PipelineCampaignError("campaign workload comm_mode is unsupported")
    for field in (
        "dt_fs",
        "fire_fmax_ev_per_a",
        "friction_per_fs",
        "temperature_k",
    ):
        value = workload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise PipelineCampaignError(
                f"campaign workload {field} must be a finite positive number"
            )
    if (
        not isinstance(workload["perturbation_description"], str)
        or not workload["perturbation_description"].strip()
    ):
        raise PipelineCampaignError(
            "campaign workload perturbation_description must be nonempty text"
        )
    stage_names = workload.get("stage_names")
    if stage_names != ["FIRE2", "NVTLangevin", "NVE"]:
        raise PipelineCampaignError(
            "campaign workload stage_names must match the three scientific stages"
        )
    repeats = campaign.get("repeats")
    if repeats != 5:
        raise PipelineCampaignError("campaign repeats must be exactly five")
    systems_total = campaign.get("systems_total")
    if (
        isinstance(systems_total, bool)
        or not isinstance(systems_total, int)
        or systems_total <= 0
    ):
        raise PipelineCampaignError(
            "campaign systems_total must be a positive integer"
        )
    if systems_total != FIXED_SYSTEMS_TOTAL:
        raise PipelineCampaignError(
            "campaign systems_total must match the fixed "
            f"{FIXED_SYSTEMS_TOTAL}-system campaign"
        )
    if workload["systems_total"] != systems_total:
        raise PipelineCampaignError(
            "campaign workload systems_total does not match campaign systems_total"
        )
    correctness_contract = manifest.get("correctness")
    if not isinstance(correctness_contract, Mapping):
        raise PipelineCampaignError("manifest correctness must be an object")
    _require_exact_keys(
        correctness_contract,
        {
            "charge_atol_e",
            "covalent_oh_cutoff_a",
            "energy_atol_ev",
            "force_atol_ev_per_a",
            "max_abs_net_charge_e",
            "max_handoff_fmax_ev_per_a",
            "min_interatomic_distance_a",
            "oxygen_connectivity_cutoff_a",
            "reference_route",
            "required_checks",
        },
        "manifest correctness",
    )
    if correctness_contract["reference_route"] != "fused_1gpu":
        raise PipelineCampaignError(
            "correctness reference_route must be fused_1gpu"
        )
    required_checks = correctness_contract["required_checks"]
    if (
        not isinstance(required_checks, list)
        or not required_checks
        or any(not isinstance(item, str) or not item.strip() for item in required_checks)
    ):
        raise PipelineCampaignError(
            "correctness required_checks must be nonempty plain text"
        )
    tolerance_fields = {
        "max_energy_difference_ev": "energy_atol_ev",
        "max_force_difference_ev_per_a": "force_atol_ev_per_a",
        "max_charge_difference_e": "charge_atol_e",
        "max_abs_net_charge_e": "max_abs_net_charge_e",
        "max_handoff_fmax_ev_per_a": "max_handoff_fmax_ev_per_a",
    }
    expected_correctness_values = {
        "energy_atol_ev": 5.0e-5,
        "force_atol_ev_per_a": 5.0e-5,
        "charge_atol_e": 5.0e-6,
        "max_abs_net_charge_e": 5.0e-5,
        "max_handoff_fmax_ev_per_a": float(workload["fire_fmax_ev_per_a"]),
        "min_interatomic_distance_a": 0.55,
        "covalent_oh_cutoff_a": 1.25,
        "oxygen_connectivity_cutoff_a": 4.0,
    }
    for field, expected in expected_correctness_values.items():
        value = correctness_contract[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not np.isclose(float(value), expected, rtol=0.0, atol=1.0e-15)
        ):
            raise PipelineCampaignError(
                f"correctness {field} does not match the fixed campaign"
            )
    tolerances: dict[str, float] = {}
    for result_field, manifest_field in tolerance_fields.items():
        value = correctness_contract.get(manifest_field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise PipelineCampaignError(
                f"correctness {manifest_field} must be a finite nonnegative number"
            )
        tolerances[result_field] = float(value)
    correctness_tolerances = {
        field: correctness_contract[field]
        for field in expected_correctness_values
    }
    _validate_acceptance(
        manifest.get("acceptance"),
        model=model,
        workload=workload,
        correctness_checks=required_checks,
        correctness_tolerances=correctness_tolerances,
        result_tolerances=tolerances,
        minimum_distance_a=float(
            correctness_contract["min_interatomic_distance_a"]
        ),
        provenance=provenance,
        job_ids=job_ids,
    )
    expected_matrix = {
        (route, repeat)
        for route in ROUTE_SPECS
        for repeat in range(1, repeats + 1)
    }
    observed_matrix = list(
        runs.loc[:, ["route", "repeat"]].itertuples(index=False, name=None)
    )
    if len(observed_matrix) != len(set(observed_matrix)):
        raise PipelineCampaignError(
            "runs.csv has duplicate route and repeat cases"
        )
    if set(observed_matrix) != expected_matrix:
        raise PipelineCampaignError(
            "runs.csv does not contain the complete requested matrix"
        )
    if runs["run_id"].duplicated().any():
        raise PipelineCampaignError("runs.csv run_id values must be unique")
    for route, spec in ROUTE_SPECS.items():
        route_rows = runs.loc[runs["route"] == route]
        for field, expected in spec.items():
            if not route_rows[field].eq(expected).all():
                raise PipelineCampaignError(
                    f"runs.csv {route} {field} does not match its route topology"
                )
    for row_index, row in runs.iterrows():
        if row["schema"] != RUN_SCHEMA:
            raise PipelineCampaignError(
                f"runs.csv row {row_index}: schema does not match {RUN_SCHEMA}"
            )
        if not str(row["run_id"]).strip():
            raise PipelineCampaignError(
                f"runs.csv row {row_index}: run_id must be nonempty"
            )
        _require_utc_timestamp(row["timestamp_utc"], f"row {row_index} timestamp_utc")
        _require_safe_relative_path(
            row["source_artifact"], f"row {row_index} source_artifact"
        )
        if row["slurm_job_id"] != job_ids[row["route"]]:
            raise PipelineCampaignError(
                f"runs.csv row {row_index}: slurm_job_id does not match its route"
            )
        if int(row["systems_requested"]) != systems_total:
            raise PipelineCampaignError(
                f"runs.csv row {row_index}: systems_requested does not match "
                "the fixed campaign"
            )
        if bool(row["success"]):
            if row["status"] != "complete":
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: successful runs need status=complete"
                )
            if str(row["error_type"]) or str(row["error"]):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: successful runs cannot have errors"
                )
            if pd.isna(row["correctness_passed"]) or not bool(
                row["correctness_passed"]
            ):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: successful runs require "
                    "correctness_passed=true"
                )
            expected_counts = {
                "systems_completed": systems_total,
                "unique_systems_completed": systems_total,
                "missing_systems": 0,
                "duplicate_systems": 0,
                "unexpected_systems": 0,
                "stage_1_completions": systems_total,
                "stage_2_completions": systems_total,
            }
            for field, expected in expected_counts.items():
                if pd.isna(row[field]) or int(row[field]) != expected:
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: {field} does not show the "
                        "complete fixed workload"
                    )
            if pd.isna(row["elapsed_s"]) or float(row["elapsed_s"]) <= 0.0:
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: elapsed_s must be positive"
                )
            if (
                pd.isna(row["peak_memory_bytes_max_rank"])
                or int(row["peak_memory_bytes_max_rank"]) <= 0
            ):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: peak memory must be positive"
                )
            expected_throughput = systems_total / float(row["elapsed_s"])
            if pd.isna(row["systems_per_s"]) or not np.isclose(
                float(row["systems_per_s"]), expected_throughput, rtol=1.0e-6
            ):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: systems_per_s is inconsistent"
                )
            for field, tolerance in tolerances.items():
                value = row[field]
                if pd.isna(value) or not 0.0 <= float(value) <= tolerance:
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: {field} exceeds the "
                        "correctness tolerance"
                    )
            min_distance = row["min_interatomic_distance_a"]
            if pd.isna(min_distance) or float(min_distance) < float(
                correctness_contract["min_interatomic_distance_a"]
            ):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: min_interatomic_distance_a "
                    "is below the required minimum"
                )
            for field in ("nvt_steps_verified", "nve_steps_verified"):
                if pd.isna(row[field]) or not bool(row[field]):
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: {field} is false"
                    )
            for field in (
                "covalent_oh_gate_passed",
                "oxygen_connectivity_gate_passed",
            ):
                if pd.isna(row[field]) or not bool(row[field]):
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: {field} is false"
                    )
            if pd.isna(row["max_relaxation_steps_observed"]):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: "
                    "max_relaxation_steps_observed is missing"
                )
        else:
            if row["status"] != "failed":
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: failed runs need status=failed"
                )
            if not str(row["error_type"]).strip() or not str(row["error"]).strip():
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: failed runs must retain "
                    "error_type and error"
                )
            if pd.isna(row["correctness_passed"]) or bool(
                row["correctness_passed"]
            ):
                raise PipelineCampaignError(
                    f"runs.csv row {row_index}: failed runs need "
                    "correctness_passed=false"
                )
            partial_counts = (
                "systems_completed",
                "unique_systems_completed",
                "missing_systems",
                "duplicate_systems",
                "unexpected_systems",
                "stage_1_completions",
                "stage_2_completions",
            )
            for field in partial_counts:
                if not pd.isna(row[field]) and not 0 <= int(row[field]) <= systems_total:
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: partial {field} is outside "
                        "the fixed campaign"
                    )
            for field in (*_FLOAT_COLUMNS, *_NULLABLE_INTEGER_COLUMNS):
                value = row[field]
                if not pd.isna(value) and float(value) < 0.0:
                    raise PipelineCampaignError(
                        f"runs.csv row {row_index}: partial {field} cannot be negative"
                    )
    successful_by_route = (
        runs.loc[runs["success"]].groupby("route").size().to_dict()
    )
    computed_ready = all(
        successful_by_route.get(route, 0) == 5 for route in ROUTE_SPECS
    )
    if publication["ready"] is not computed_ready:
        raise PipelineCampaignError(
            "publication ready does not match the successful repeat counts"
        )
    if require_publishable and not computed_ready:
        raise PipelineCampaignError("pipeline campaign is not ready for publication")
    return PipelineCampaignBundle(root=root, manifest=manifest, runs=runs)


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if not values.empty else np.nan


def summarize_pipeline_campaign(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize successful repeats while retaining explicit failure counts."""

    rows: list[dict[str, Any]] = []
    for route, spec in ROUTE_SPECS.items():
        route_rows = runs.loc[runs["route"] == route]
        successful = route_rows.loc[route_rows["success"]]
        elapsed = successful["elapsed_s"].astype(float)
        throughput = successful["systems_per_s"].astype(float)
        gpu_cost = successful["gpu_seconds_per_structure"].astype(float)
        rows.append(
            {
                "route": route,
                "gpu_count": spec["gpu_count"],
                "total_runs": int(len(route_rows)),
                "successful_runs": int(len(successful)),
                "failed_runs": int((~route_rows["success"]).sum()),
                "median_elapsed_s": _quantile(elapsed, 0.5),
                "elapsed_q25_s": _quantile(elapsed, 0.25),
                "elapsed_q75_s": _quantile(elapsed, 0.75),
                "median_systems_per_s": _quantile(throughput, 0.5),
                "median_gpu_seconds_per_structure": _quantile(gpu_cost, 0.5),
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["route"] == "fused_1gpu"].iloc[0]
    baseline_elapsed = float(baseline["median_elapsed_s"])
    summary["speedup_vs_1gpu"] = baseline_elapsed / summary["median_elapsed_s"]
    summary["parallel_efficiency_pct"] = (
        100.0 * summary["speedup_vs_1gpu"] / summary["gpu_count"]
    )
    return summary


__all__ = (
    "BUNDLE_SCHEMA",
    "CHECKSUM_INDEX_NAME",
    "DEFAULT_BUNDLE_DIR",
    "FIXED_SYSTEMS_TOTAL",
    "MANIFEST_NAME",
    "PipelineCampaignBundle",
    "PipelineCampaignError",
    "ROUTE_SPECS",
    "RUN_COLUMNS",
    "RUN_SCHEMA",
    "RUNS_NAME",
    "canonical_json_sha256",
    "correctness_record_sha256",
    "load_pipeline_campaign_bundle",
    "producer_set_sha256",
    "runtime_identity_sha256",
    "summarize_pipeline_campaign",
    "toolkit_runtime_record_sha256",
)
