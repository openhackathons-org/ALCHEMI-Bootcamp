"""Read saved H100 domain-decomposition results for the Part 1 notebook.

An absent result directory is represented explicitly as ``not reported``.
An existing bundle is accepted only when its hashes, requested rows, hardware,
settings, and retained failures all agree.  Failed rows stay in every
learner-facing table; plot values are simply missing for failed measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import DOMAIN_METHODOLOGY


BUNDLE_SCHEMA = "alchemi.domain-decomposition-lesson.v3"
MANIFEST_NAME = "manifest.json"
CHECKSUM_INDEX_NAME = "SHA256SUMS"
RAW_RESULTS_NAME = "raw-results.jsonl"
TABLE_NAMES = ("capacity", "parity", "distributed")

CAPACITY_COLUMNS = (
    "case_id",
    "atom_count",
    "molecules_per_species",
    "gpus",
    "success",
    "status",
    "failure_type",
    "failure_stage",
    "error",
    "elapsed_s",
    "peak_memory_bytes_max_rank",
    "energy_ev",
    "force_rms_ev_per_a",
    "force_max_ev_per_a",
    "structure_sha256",
    "settings_sha256",
    "measurement_role",
    "measurement_kind",
)
PARITY_COLUMNS = (
    "case_id",
    "atom_count",
    "force_reference_gpus",
    "energy_reference_gpus",
    "gpus",
    "success",
    "status",
    "failure_type",
    "failure_stage",
    "error",
    "one_gpu_energy_abs_offset_ev",
    "one_gpu_energy_abs_offset_ev_per_atom",
    "distributed_energy_difference_ev",
    "distributed_energy_difference_ev_per_atom",
    "force_rms_difference_ev_per_a",
    "force_max_difference_ev_per_a",
    "energy_tolerance_ev_per_atom",
    "force_tolerance_ev_per_a",
    "distributed_energy_passed",
    "force_passed",
    "parity_passed",
    "structure_sha256",
    "settings_sha256",
    "measurement_role",
    "measurement_kind",
)
DISTRIBUTED_COLUMNS = (
    "case_id",
    "atom_count",
    "molecules_per_species",
    "nodes",
    "gpus",
    "ranks",
    "success",
    "status",
    "failure_type",
    "failure_stage",
    "error",
    "elapsed_s",
    "warmup_count",
    "sample_count",
    "elapsed_samples_s",
    "elapsed_median_s",
    "elapsed_q1_s",
    "elapsed_q3_s",
    "elapsed_iqr_s",
    "peak_memory_bytes_max_rank",
    "owned_atoms_min_rank",
    "owned_atoms_max_rank",
    "spatial_grid",
    "energy_ev",
    "force_rms_ev_per_a",
    "force_max_ev_per_a",
    "structure_sha256",
    "settings_sha256",
    "measurement_role",
    "measurement_kind",
)
TABLE_COLUMNS = {
    "capacity": CAPACITY_COLUMNS,
    "parity": PARITY_COLUMNS,
    "distributed": DISTRIBUTED_COLUMNS,
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_H100 = re.compile(r"(?:^|[^A-Za-z0-9])H100(?:[^A-Za-z0-9]|$)", re.IGNORECASE)
_PARITY_CASE_ID = re.compile(
    r"^parity-pairs-(?P<pair_count>[0-9]{6})-gpus-(?P<gpus>[0-9]{2})$"
)
_DISTRIBUTED_CASE_ID = re.compile(
    r"^(?P<kind>steady-timing|rescue)-pairs-(?P<pair_count>[0-9]{6})-"
    r"gpus-(?P<gpus>[0-9]{2})$"
)
_OOM_TYPES = {"CUDAOutOfMemoryError", "OutOfMemoryError"}
CHARGE_SUM_TOLERANCE_E = DOMAIN_METHODOLOGY.charge_sum_tolerance_e
PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
)
PME_EWALD_FORCE_TOLERANCE_EV_PER_A = (
    DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
)
ATOMS_PER_COMPOSITION_UNIT = DOMAIN_METHODOLOGY.atoms_per_composition_unit
ELECTROSTATICS_VALIDATION_ATOM_COUNT = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
    * ATOMS_PER_COMPOSITION_UNIT
)
_REQUIRED_PARITY_GPUS = frozenset(DOMAIN_METHODOLOGY.distributed_world_sizes)
_REQUIRED_STEADY_TIMING_GPUS = frozenset(
    DOMAIN_METHODOLOGY.steady_timing_world_sizes
)
_REQUIRED_RESCUE_GPUS = frozenset(DOMAIN_METHODOLOGY.distributed_world_sizes)


class DomainLessonResultsError(ValueError):
    """Raised when an existing result bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class DomainLessonView:
    """Notebook-ready saved results, or an explicit not-reported state."""

    available: bool
    reason: str
    root: Path
    manifest: Mapping[str, Any]
    capacity_table: pd.DataFrame
    charge_diagnostics_table: pd.DataFrame
    electrostatics_table: pd.DataFrame
    parity_table: pd.DataFrame
    distributed_table: pd.DataFrame
    plot_data: pd.DataFrame

    @property
    def successful_case_count(self) -> int:
        """Count successful saved cases across the three measurement tables."""

        return sum(
            int(success.sum())
            for success in (
                self.capacity_table["success"].eq(True),
                self.parity_table["passed"].eq(True),
                self.distributed_table["success"].eq(True),
            )
        )

    @property
    def failed_case_count(self) -> int:
        """Count failed saved cases across the three measurement tables."""

        return sum(
            int(failed.sum())
            for failed in (
                self.capacity_table["success"].eq(False),
                self.parity_table["passed"].eq(False),
                self.distributed_table["success"].eq(False),
            )
        )

    @property
    def measured_max_atom_count(self) -> int | None:
        """Return the largest measured case, or ``None`` when results are absent."""

        if not self.available:
            return None
        atom_counts = pd.concat(
            [
                table["atom_count"]
                for table in (
                    self.capacity_table,
                    self.parity_table,
                    self.distributed_table,
                )
                if not table.empty
            ],
            ignore_index=True,
        )
        return int(atom_counts.max())

    @property
    def takeaway(self) -> dict[str, Any]:
        """Return the small set of conclusions shown below the recorded plots."""

        if not self.available:
            raise DomainLessonResultsError(
                "recorded domain-decomposition results are not available"
            )

        successful_capacity = self.capacity_table.loc[
            self.capacity_table["success"].eq(True)
        ]
        oom_capacity = self.capacity_table.loc[
            self.capacity_table["success"].eq(False)
            & self.capacity_table["failure_type"]
            .astype(str)
            .str.contains("OutOfMemory", case=False, na=False)
        ].sort_values("atom_count")
        if successful_capacity.empty or oom_capacity.empty:
            raise DomainLessonResultsError(
                "recorded capacity results need a success and a CUDA OOM"
            )

        first_oom_atoms = int(oom_capacity.iloc[0]["atom_count"])
        rescue_rows = self.distributed_table.loc[
            self.distributed_table["atom_count"].eq(first_oom_atoms)
            & self.distributed_table["world_size"].gt(1)
        ]
        rescue_gpu_counts = tuple(
            sorted(
                int(value)
                for value in rescue_rows.loc[
                    rescue_rows["success"].eq(True), "world_size"
                ]
            )
        )

        scaling_rows = self.distributed_table.loc[
            self.distributed_table["world_size"].gt(1)
            & self.distributed_table["speedup_vs_1gpu"].notna()
        ].sort_values("world_size")
        speedup_by_gpu = tuple(
            (int(row.world_size), float(row.speedup_vs_1gpu))
            for row in scaling_rows.itertuples(index=False)
        )
        parallel_efficiency_by_gpu = tuple(
            (int(row.world_size), float(row.parallel_efficiency))
            for row in scaling_rows.itertuples(index=False)
        )
        energy_comparison_rows = self.parity_table.loc[
            self.parity_table["world_size"].isin(
                DOMAIN_METHODOLOGY.energy_comparison_world_sizes
            )
        ]

        return {
            "largest_successful_single_gpu_atoms": int(
                successful_capacity["atom_count"].max()
            ),
            "first_single_gpu_oom_atoms": first_oom_atoms,
            "rescue_successful_gpu_counts": rescue_gpu_counts,
            "all_one_gpu_force_checks_passed": bool(
                self.parity_table["force_passed"].eq(True).all()
            ),
            "all_distributed_energy_checks_passed": bool(
                energy_comparison_rows["distributed_energy_passed"].eq(True).all()
            ),
            "timed_one_gpu_force_checks_passed": True,
            "timed_distributed_energy_checks_passed": True,
            "rescue_output_comparison_count": max(
                len(rescue_rows.loc[rescue_rows["success"].eq(True)]) - 1,
                0,
            ),
            "speedup_by_gpu": speedup_by_gpu,
            "parallel_efficiency_by_gpu": parallel_efficiency_by_gpu,
        }

    @property
    def bundle_record(self) -> dict[str, str] | None:
        """Return compact file identities for the verified saved result set."""

        if not self.available:
            return None
        source = self.manifest["identity"]["source"]
        return {
            "created_utc": str(self.manifest["created_utc"]),
            "manifest_sha256": _sha256_file(self.root / MANIFEST_NAME),
            "raw_results_sha256": _sha256_file(self.root / RAW_RESULTS_NAME),
            "checksum_index_sha256": _sha256_file(self.root / CHECKSUM_INDEX_NAME),
            "repository_commit": str(source["repository_commit"]),
        }

    @property
    def recorded_run_table(self) -> pd.DataFrame:
        """Identify the saved run without exposing the full manifest."""

        if not self.available:
            return pd.DataFrame(columns=("Recorded result set",))
        identity = self.manifest["identity"]
        source = identity["source"]
        hardware = identity["hardware"]
        gpu_counts = sorted(
            set(self.distributed_table["world_size"].astype(int).tolist())
        )
        return pd.Series(
            {
                "Bundle created (UTC)": self.manifest["created_utc"],
                "Site": hardware["site"],
                "GPU": hardware["gpu_model"],
                "Interconnect": hardware["interconnect"],
                "Measured nodes / GPUs": ", ".join(
                    f"{count} / {count}" for count in gpu_counts
                ),
                "Toolkit version": source["toolkit_version"],
                "Toolkit Core commit": source["toolkit_commit"],
                "Toolkit-Ops commit": source["toolkit_ops_commit"],
                "Tutorial commit": source["repository_commit"],
            },
            name="Recorded result set",
        ).to_frame()

    @property
    def failed_table(self) -> pd.DataFrame:
        """Return every measured failure without dropping its source table."""

        failures: list[pd.DataFrame] = []
        for table_name, table in (
            ("capacity", self.capacity_table),
            ("parity", self.parity_table),
            ("distributed", self.distributed_table),
        ):
            if "success" not in table or table.empty:
                continue
            selected = table.loc[table["success"].eq(False)].copy()
            if not selected.empty:
                selected.insert(0, "table", table_name)
                failures.append(selected)
        if not failures:
            return pd.DataFrame(columns=("table", "case_id", "failure_type", "error"))
        return pd.concat(failures, ignore_index=True, sort=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of compact, sorted, strict JSON."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _not_reported(
    root: Path,
    planned_atom_counts: tuple[int, ...],
    reason: str,
    manifest: Mapping[str, Any] | None = None,
) -> DomainLessonView:
    raw_capacity = pd.DataFrame(
        [
            {
                "case_id": f"planned-{atoms}",
                "atom_count": atoms,
                "success": pd.NA,
                "status": "not reported",
                "failure_type": "",
                "failure_stage": "",
                "error": reason,
                "measurement_role": "capacity",
                "measurement_kind": "not reported",
            }
            for atoms in planned_atom_counts
        ],
        columns=CAPACITY_COLUMNS,
    )
    raw_parity = pd.DataFrame(columns=PARITY_COLUMNS)
    raw_distributed = pd.DataFrame(columns=DISTRIBUTED_COLUMNS)
    plot_data = _plot_data(raw_capacity, raw_distributed)
    return DomainLessonView(
        available=False,
        reason=reason,
        root=root,
        manifest={} if manifest is None else manifest,
        capacity_table=_capacity_view(raw_capacity),
        charge_diagnostics_table=_charge_diagnostics_view(()),
        electrostatics_table=_electrostatics_view(None),
        parity_table=_parity_view(raw_parity),
        distributed_table=_distributed_view(raw_distributed),
        plot_data=plot_data,
    )


def _planned_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("planned_atom_counts must contain positive integers")
        try:
            count = int(value)
            unchanged = float(value) == count
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planned_atom_counts must contain positive integers"
            ) from exc
        if count <= 0 or not unchanged:
            raise ValueError("planned_atom_counts must contain positive integers")
        counts.append(count)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("planned_atom_counts must be nonempty and unique")
    return tuple(counts)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
        unchanged = float(value) == integer
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0 or not unchanged:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise DomainLessonResultsError(f"{context}.{key} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    keys: Sequence[str],
    *,
    context: str,
) -> None:
    missing = set(keys) - set(value)
    if missing:
        raise DomainLessonResultsError(f"{context} is missing {sorted(missing)!r}")


def _read_checksum_index(root: Path) -> dict[str, str]:
    index_path = root / CHECKSUM_INDEX_NAME
    if not index_path.is_file():
        raise DomainLessonResultsError(f"missing {CHECKSUM_INDEX_NAME}")
    checksums: dict[str, str] = {}
    for line_number, raw in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 2 or not _HEX64.fullmatch(parts[0]):
            raise DomainLessonResultsError(
                f"invalid {CHECKSUM_INDEX_NAME} line {line_number}"
            )
        relative = PurePosixPath(parts[1].removeprefix("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise DomainLessonResultsError("checksum path must stay inside the bundle")
        name = relative.as_posix()
        if name in checksums:
            raise DomainLessonResultsError(f"duplicate checksum entry for {name}")
        checksums[name] = parts[0]
    return checksums


def _validate_file(root: Path, relative_name: str, expected_sha256: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DomainLessonResultsError("data path must stay inside the bundle")
    if not _HEX64.fullmatch(str(expected_sha256)):
        raise DomainLessonResultsError(f"invalid SHA-256 for {relative_name}")
    path = root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DomainLessonResultsError("data path must stay inside the bundle") from exc
    if not path.is_file():
        raise DomainLessonResultsError(f"missing data file {relative_name}")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise DomainLessonResultsError(
            f"SHA-256 mismatch for {relative_name}: {observed}"
        )
    return path


def _reject_absolute_json_paths(value: Any, *, context: str) -> None:
    """Reject direct host-path references in portable JSON records."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_json_paths(item, context=f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_json_paths(item, context=f"{context}[{index}]")
        return
    if isinstance(value, str):
        is_windows_path = re.match(r"^[A-Za-z]:[\\/]", value) is not None
        if PurePosixPath(value).is_absolute() or is_windows_path:
            raise DomainLessonResultsError(
                f"{context} contains a host path; bundle references must be relative"
            )


def _read_raw_results(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    metadata = _require_mapping(manifest, "raw_results", context="manifest")
    _require_keys(
        metadata,
        ("file", "sha256", "row_count"),
        context="manifest.raw_results",
    )
    filename = str(metadata["file"])
    if filename != RAW_RESULTS_NAME:
        raise DomainLessonResultsError(
            f"manifest.raw_results.file must be {RAW_RESULTS_NAME}"
        )
    expected_sha256 = str(metadata["sha256"])
    if checksums.get(filename) != expected_sha256:
        raise DomainLessonResultsError(
            "raw results checksum index and manifest disagree"
        )
    path = _validate_file(root, filename, expected_sha256)
    rows: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DomainLessonResultsError(
                f"invalid raw results JSON on line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise DomainLessonResultsError(
                f"raw results line {line_number} must contain an object"
            )
        _reject_absolute_json_paths(
            row,
            context=f"raw results line {line_number}",
        )
        rows.append(row)
    raw_row_count = metadata["row_count"]
    if isinstance(raw_row_count, bool):
        raise DomainLessonResultsError(
            "manifest.raw_results.row_count must be a non-negative integer"
        )
    try:
        expected_rows = int(raw_row_count)
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "manifest.raw_results.row_count must be a non-negative integer"
        ) from exc
    if expected_rows < 0 or float(raw_row_count) != expected_rows:
        raise DomainLessonResultsError(
            "manifest.raw_results.row_count must be a non-negative integer"
        )
    if len(rows) != expected_rows:
        raise DomainLessonResultsError("raw results row count does not match")
    return rows


def _artifact_path(
    record: Mapping[str, Any],
    *,
    context: str,
    directory: str,
    suffix: str,
    checksums: Mapping[str, str],
    root: Path,
) -> str:
    _require_keys(record, ("file", "sha256"), context=context)
    filename = str(record["file"])
    relative = PurePosixPath(filename)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 2
        or relative.parts[0] != directory
        or relative.suffix != suffix
    ):
        raise DomainLessonResultsError(
            f"{context}.file must be a {suffix} file inside {directory}/"
        )
    expected_sha256 = _require_sha256(
        record["sha256"],
        name=f"{context}.sha256",
    )
    if checksums.get(filename) != expected_sha256:
        raise DomainLessonResultsError(
            f"{filename} checksum index and manifest disagree"
        )
    _validate_file(root, filename, expected_sha256)
    return filename


def _validate_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> set[str]:
    artifacts = _require_mapping(manifest, "artifacts", context="manifest")
    _require_keys(
        artifacts,
        ("structures", "case_logs", "files"),
        context="manifest.artifacts",
    )
    structures = artifacts["structures"]
    case_logs = artifacts["case_logs"]
    files = artifacts["files"]
    if not isinstance(structures, list) or not structures:
        raise DomainLessonResultsError(
            "manifest.artifacts.structures must be a nonempty list"
        )
    if not isinstance(case_logs, list) or not case_logs:
        raise DomainLessonResultsError(
            "manifest.artifacts.case_logs must be a nonempty list"
        )
    if not isinstance(files, list) or not files:
        raise DomainLessonResultsError(
            "manifest.artifacts.files must be a nonempty list"
        )

    declared_paths: set[str] = set()
    structure_roles: set[str] = set()
    for index, item in enumerate(structures):
        context = f"manifest.artifacts.structures[{index}]"
        if not isinstance(item, Mapping):
            raise DomainLessonResultsError(f"{context} must be an object")
        _require_keys(item, ("role", "pair_count"), context=context)
        role = str(item["role"]).strip()
        if not role or role in structure_roles:
            raise DomainLessonResultsError("structure artifact roles must be unique")
        structure_roles.add(role)
        try:
            _positive_integer(item["pair_count"], name=f"{context}.pair_count")
        except ValueError as exc:
            raise DomainLessonResultsError(str(exc)) from exc
        filename = _artifact_path(
            item,
            context=context,
            directory="structures",
            suffix=".extxyz",
            checksums=checksums,
            root=root,
        )
        if filename in declared_paths:
            raise DomainLessonResultsError(
                f"artifact file is declared twice: {filename}"
            )
        declared_paths.add(filename)

    case_ids: set[str] = set()
    for index, item in enumerate(case_logs):
        context = f"manifest.artifacts.case_logs[{index}]"
        if not isinstance(item, Mapping):
            raise DomainLessonResultsError(f"{context} must be an object")
        _require_keys(item, ("case_id",), context=context)
        case_id = str(item["case_id"]).strip()
        if not case_id or case_id in case_ids:
            raise DomainLessonResultsError("case log IDs must be unique")
        case_ids.add(case_id)
        filename = _artifact_path(
            item,
            context=context,
            directory="logs",
            suffix=".log",
            checksums=checksums,
            root=root,
        )
        if filename in declared_paths:
            raise DomainLessonResultsError(
                f"artifact file is declared twice: {filename}"
            )
        declared_paths.add(filename)

    for index, item in enumerate(files):
        context = f"manifest.artifacts.files[{index}]"
        if not isinstance(item, Mapping):
            raise DomainLessonResultsError(f"{context} must be an object")
        _require_keys(item, ("role", "file", "sha256"), context=context)
        role = str(item["role"]).strip()
        if not role:
            raise DomainLessonResultsError(f"{context}.role must be nonempty")
        filename = str(item["file"])
        relative = PurePosixPath(filename)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.as_posix() in {MANIFEST_NAME, CHECKSUM_INDEX_NAME}
        ):
            raise DomainLessonResultsError(
                f"{context}.file must stay inside the bundle"
            )
        expected_sha256 = _require_sha256(
            item["sha256"],
            name=f"{context}.sha256",
        )
        if checksums.get(filename) != expected_sha256:
            raise DomainLessonResultsError(
                f"{filename} checksum index and manifest disagree"
            )
        _validate_file(root, filename, expected_sha256)
        if filename in declared_paths:
            raise DomainLessonResultsError(
                f"artifact file is declared twice: {filename}"
            )
        declared_paths.add(filename)
    return declared_paths


def _validate_declared_files(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
    artifact_paths: set[str],
) -> None:
    data = _require_mapping(manifest, "data", context="manifest")
    table_paths = {
        str(_require_mapping(data, name, context="manifest.data")["file"])
        for name in TABLE_NAMES
    }
    raw_results = _require_mapping(manifest, "raw_results", context="manifest")
    declared = {
        MANIFEST_NAME,
        str(raw_results["file"]),
        *table_paths,
        *artifact_paths,
    }
    checksum_paths = set(checksums)
    missing_checksums = declared - checksum_paths
    if missing_checksums:
        raise DomainLessonResultsError(
            f"SHA256SUMS is missing declared files: {sorted(missing_checksums)!r}"
        )
    undeclared_checksums = checksum_paths - declared
    if undeclared_checksums:
        raise DomainLessonResultsError(
            f"SHA256SUMS lists undeclared files: {sorted(undeclared_checksums)!r}"
        )

    files_on_disk = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected_files = declared | {CHECKSUM_INDEX_NAME}
    undeclared_files = files_on_disk - expected_files
    if undeclared_files:
        raise DomainLessonResultsError(
            f"bundle contains undeclared files: {sorted(undeclared_files)!r}"
        )


def _validate_identity(
    manifest: Mapping[str, Any],
) -> tuple[dict[int, str], str]:
    identity = _require_mapping(manifest, "identity", context="manifest")
    hashes = _require_mapping(manifest, "identity_sha256", context="manifest")
    _require_keys(
        identity,
        ("source", "hardware", "settings", "inputs"),
        context="identity",
    )
    _require_keys(
        hashes,
        ("source", "hardware", "settings", "inputs"),
        context="identity_sha256",
    )
    for name in ("source", "hardware", "settings", "inputs"):
        record = _require_mapping(identity, name, context="identity")
        observed = canonical_json_sha256(record)
        expected = str(hashes[name])
        if not _HEX64.fullmatch(expected) or observed != expected:
            raise DomainLessonResultsError(f"{name} identity hash does not match")

    source = _require_mapping(identity, "source", context="identity")
    _require_keys(
        source,
        (
            "repository_commit",
            "repository_dirty",
            "toolkit_commit",
            "toolkit_ops_commit",
            "toolkit_version",
            "domain_methodology",
            "producer_files_sha256",
        ),
        context="identity.source",
    )
    if not _HEX40.fullmatch(str(source["repository_commit"])):
        raise DomainLessonResultsError("repository_commit must be a 40-digit SHA")
    if not _HEX40.fullmatch(str(source["toolkit_commit"])):
        raise DomainLessonResultsError("toolkit_commit must be a 40-digit SHA")
    if not _HEX40.fullmatch(str(source["toolkit_ops_commit"])):
        raise DomainLessonResultsError("toolkit_ops_commit must be a 40-digit SHA")
    if not isinstance(source["repository_dirty"], bool):
        raise DomainLessonResultsError("repository_dirty must be boolean")
    if not str(source["toolkit_version"]).strip():
        raise DomainLessonResultsError("toolkit_version must be reported")
    producer_files = source["producer_files_sha256"]
    if not isinstance(producer_files, Mapping) or not producer_files:
        raise DomainLessonResultsError("producer_files_sha256 must be nonempty")
    if any(
        not str(path).strip() or not _HEX64.fullmatch(str(digest))
        for path, digest in producer_files.items()
    ):
        raise DomainLessonResultsError("producer file hashes must be valid SHA-256")

    expected_config_sha256 = _sha256_file(Path(__file__).with_name("config.py"))
    expected_methodology_values = json.loads(
        json.dumps(
            DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
            allow_nan=False,
        )
    )
    expected_methodology_record = json.loads(
        json.dumps(DOMAIN_METHODOLOGY.as_record(), allow_nan=False)
    )
    source_methodology = _require_mapping(
        source,
        "domain_methodology",
        context="identity.source",
    )
    _require_keys(
        source_methodology,
        ("name", "version", "config_sha256", "record", "resolved_values"),
        context="identity.source.domain_methodology",
    )
    if (
        source_methodology["name"] != DOMAIN_METHODOLOGY.name
        or source_methodology["version"] != DOMAIN_METHODOLOGY.version
        or source_methodology["config_sha256"] != expected_config_sha256
        or source_methodology["record"] != expected_methodology_record
        or source_methodology["resolved_values"] != expected_methodology_values
    ):
        raise DomainLessonResultsError(
            "saved results do not use the current methodology identity and values"
        )
    config_producer_digests = [
        str(digest)
        for path, digest in producer_files.items()
        if PurePosixPath(str(path)).name == "config.py"
    ]
    if config_producer_digests != [expected_config_sha256]:
        raise DomainLessonResultsError(
            "producer files do not contain the current methodology config"
        )

    hardware = _require_mapping(identity, "hardware", context="identity")
    _require_keys(
        hardware,
        (
            "site",
            "site_source",
            "gpu_model",
            "gpu_memory_bytes",
            "gpus_available",
            "nodes_available",
            "resource_count_source",
            "driver_version",
            "cuda_version",
            "interconnect",
            "interconnect_source",
        ),
        context="identity.hardware",
    )
    if not _H100.search(str(hardware["gpu_model"])):
        raise DomainLessonResultsError("saved lesson results must identify H100 GPUs")
    for key in ("gpu_memory_bytes", "gpus_available", "nodes_available"):
        try:
            value = int(hardware[key])
        except (TypeError, ValueError) as exc:
            raise DomainLessonResultsError(
                f"identity.hardware.{key} must be positive"
            ) from exc
        if value <= 0:
            raise DomainLessonResultsError(f"identity.hardware.{key} must be positive")
    for key in (
        "site",
        "site_source",
        "resource_count_source",
        "driver_version",
        "cuda_version",
        "interconnect",
        "interconnect_source",
    ):
        if not str(hardware[key]).strip():
            raise DomainLessonResultsError(f"identity.hardware.{key} is empty")

    settings = _require_mapping(identity, "settings", context="identity")
    _require_keys(
        settings,
        (
            "model_components",
            "domain_methodology",
            "precision",
            "aimnet_checkpoint_sha256",
            "d3_parameters_sha256",
            "pme",
            "ewald_reference",
            "domain",
            "packmol",
            "timing_boundary",
            "timing_measurement_kind",
            "timing_measurement_role",
            "timing_world_sizes",
            "timing_warmup_count",
            "timing_sample_count",
            "timing_model_evaluations_per_workflow",
            "timing_one_rank_run_steps",
            "timing_multi_rank_run_steps",
            "timing_summary",
            "timing_quartile_method",
            "timing_max_relative_iqr",
        ),
        context="identity.settings",
    )
    components = settings["model_components"]
    if not isinstance(components, list) or not components:
        raise DomainLessonResultsError("model_components must be a nonempty list")
    for key in ("aimnet_checkpoint_sha256", "d3_parameters_sha256"):
        if not _HEX64.fullmatch(str(settings[key])):
            raise DomainLessonResultsError(f"{key} must be a SHA-256")
    for key in ("pme", "ewald_reference", "domain", "packmol"):
        if not isinstance(settings[key], Mapping) or not settings[key]:
            raise DomainLessonResultsError(f"identity.settings.{key} must be reported")
    if (
        not str(settings["precision"]).strip()
        or not str(settings["timing_boundary"]).strip()
    ):
        raise DomainLessonResultsError("precision and timing_boundary must be reported")
    if (
        settings["timing_measurement_kind"] != "steady_partition_run_gather"
        or settings["timing_measurement_role"] != "steady_timing"
        or list(settings["timing_world_sizes"])
        != list(DOMAIN_METHODOLOGY.steady_timing_world_sizes)
        or settings["timing_warmup_count"]
        != DOMAIN_METHODOLOGY.steady_timing_warmup_count
        or settings["timing_sample_count"]
        != DOMAIN_METHODOLOGY.steady_timing_sample_count
        or settings["timing_model_evaluations_per_workflow"]
        != DOMAIN_METHODOLOGY.steady_timing_model_evaluations_per_workflow
        or settings["timing_one_rank_run_steps"]
        != DOMAIN_METHODOLOGY.steady_timing_run_steps(1)
        or settings["timing_multi_rank_run_steps"]
        != DOMAIN_METHODOLOGY.steady_timing_run_steps(
            DOMAIN_METHODOLOGY.distributed_world_sizes[0]
        )
        or not str(settings["timing_summary"]).strip()
        or settings["timing_quartile_method"] != "inclusive linear interpolation"
        or settings["timing_max_relative_iqr"]
        != DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr
    ):
        raise DomainLessonResultsError(
            "identity.settings steady timing differs from the current methodology"
        )

    settings_methodology = _require_mapping(
        settings,
        "domain_methodology",
        context="identity.settings",
    )
    _require_keys(
        settings_methodology,
        ("name", "version", "config_sha256", "resolved_values"),
        context="identity.settings.domain_methodology",
    )
    if (
        settings_methodology["name"] != DOMAIN_METHODOLOGY.name
        or settings_methodology["version"] != DOMAIN_METHODOLOGY.version
        or settings_methodology["config_sha256"] != expected_config_sha256
        or settings_methodology["resolved_values"] != expected_methodology_values
    ):
        raise DomainLessonResultsError(
            "saved settings do not use the current methodology identity and values"
        )

    pme = settings["pme"]
    expected_pme = {
        "cutoff_a": DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
        "mesh_safety_factor": DOMAIN_METHODOLOGY.pme_mesh_safety_factor,
        "parameter_rule": (
            "estimate_pme_parameters(accuracy, real_space_cutoff, "
            "mesh_safety_factor)"
        ),
        "spline_order": DOMAIN_METHODOLOGY.pme_spline_order,
        "accuracy": DOMAIN_METHODOLOGY.pme_accuracy,
        "hybrid_forces": True,
        "reciprocal_mesh_distribution": "replicated_per_rank",
    }
    if any(pme.get(key) != value for key, value in expected_pme.items()):
        raise DomainLessonResultsError(
            "identity.settings.pme differs from the current methodology"
        )

    ewald_reference = settings["ewald_reference"]
    expected_ewald_reference = {
        "accuracy": DOMAIN_METHODOLOGY.ewald_reference_accuracy,
        "parameter_rule": "estimate_ewald_parameters(accuracy)",
        "scope": "fixed-charge electrostatics validation only",
    }
    if any(
        ewald_reference.get(key) != value
        for key, value in expected_ewald_reference.items()
    ):
        raise DomainLessonResultsError(
            "identity.settings.ewald_reference differs from the current methodology"
        )

    domain = settings["domain"]
    expected_domain = {
        "api": "DomainParallel",
        "skin_a": DOMAIN_METHODOLOGY.domain_halo_skin_a,
        "cutoff_a": max(
            DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a,
            DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
            DOMAIN_METHODOLOGY.d3_cutoff_a,
        ),
        "compile": False,
        "require_nondegenerate": True,
        "grid_dims": DOMAIN_METHODOLOGY.domain_grid_dims,
        "rank_grid_policy": (
            "Toolkit SpatialPartitioner derives cells_per_dim and rank_grid "
            "from each input's actual cell shape and the domain cutoff"
        ),
        "recorded_layout_fields": ["cells_per_dim", "rank_grid"],
    }
    if any(domain.get(key) != value for key, value in expected_domain.items()):
        raise DomainLessonResultsError(
            "identity.settings.domain differs from the current methodology"
        )

    packmol = settings["packmol"]
    expected_packmol = {
        "construction_density_g_cm3": (
            DOMAIN_METHODOLOGY.construction_density_g_cm3
        ),
        "tolerance_a": DOMAIN_METHODOLOGY.packmol_tolerance_a,
        "precision_a": DOMAIN_METHODOLOGY.packmol_precision_a,
        "base_seed": DOMAIN_METHODOLOGY.packmol_seed,
        "periodic_boundary_check": True,
    }
    if any(packmol.get(key) != value for key, value in expected_packmol.items()):
        raise DomainLessonResultsError(
            "identity.settings.packmol differs from the current methodology"
        )

    parity_acceptance = _require_mapping(
        settings,
        "parity_acceptance",
        context="identity.settings",
    )
    expected_parity_acceptance = {
        "declared_before_measurement": True,
        "energy_reference_world_size": (
            DOMAIN_METHODOLOGY.energy_reference_world_size
        ),
        "energy_comparison_world_sizes": list(
            DOMAIN_METHODOLOGY.energy_comparison_world_sizes
        ),
        "energy_one_gpu_comparison": (
            "diagnostic_only_due_different_reduction_path"
        ),
        "energy_rule": (
            "abs(delta_energy_eV) / atom_count <= tolerance_eV_per_atom"
        ),
        "energy_tolerance_ev_per_atom": (
            DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
        ),
        "force_rule": (
            "componentwise abs(delta) <= atol_eV_A + "
            "rtol * abs(reference_component_eV_A)"
        ),
        "force_reference_world_size": (
            DOMAIN_METHODOLOGY.force_reference_world_size
        ),
        "force_comparison_world_sizes": list(
            DOMAIN_METHODOLOGY.force_comparison_world_sizes
        ),
        "force_atol_ev_a": DOMAIN_METHODOLOGY.parity_force_atol_ev_a,
        "force_rtol": DOMAIN_METHODOLOGY.parity_force_rtol,
    }
    if dict(parity_acceptance) != expected_parity_acceptance:
        raise DomainLessonResultsError(
            "identity.settings.parity_acceptance differs from the current methodology"
        )

    inputs = _require_mapping(identity, "inputs", context="identity")
    _require_keys(
        inputs,
        (
            "structures_sha256_by_atom_count",
            "nci_subset_sha256",
            "molecule_pair",
            "construction_density_g_cm3",
        ),
        context="identity.inputs",
    )
    if not _HEX64.fullmatch(str(inputs["nci_subset_sha256"])):
        raise DomainLessonResultsError("nci_subset_sha256 must be a SHA-256")
    raw_structure_hashes = inputs["structures_sha256_by_atom_count"]
    if not isinstance(raw_structure_hashes, Mapping) or not raw_structure_hashes:
        raise DomainLessonResultsError(
            "structures_sha256_by_atom_count must be a nonempty object"
        )
    structure_hashes: dict[int, str] = {}
    for raw_count, raw_digest in raw_structure_hashes.items():
        try:
            atom_count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise DomainLessonResultsError(
                "structure hash keys must be positive atom counts"
            ) from exc
        digest = str(raw_digest)
        if (
            atom_count <= 0
            or str(atom_count) != str(raw_count)
            or not _HEX64.fullmatch(digest)
        ):
            raise DomainLessonResultsError("structures_sha256_by_atom_count is invalid")
        structure_hashes[atom_count] = digest
    if str(inputs["molecule_pair"]) != "phenol + N-methylacetamide":
        raise DomainLessonResultsError("unexpected molecular pair")
    try:
        density = float(inputs["construction_density_g_cm3"])
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "construction_density_g_cm3 must be positive"
        ) from exc
    if not np.isfinite(density) or density <= 0.0:
        raise DomainLessonResultsError("construction_density_g_cm3 must be positive")
    return structure_hashes, str(hashes["settings"])


def _coerce_boolean(series: pd.Series, *, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise DomainLessonResultsError(f"{name} contains an invalid boolean")
    return converted.astype(bool)


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if not _HEX64.fullmatch(digest):
        raise DomainLessonResultsError(f"{name} must be a valid SHA-256")
    return digest


def _finite_charge_number(
    record: Mapping[str, Any],
    name: str,
    *,
    context: str,
) -> float:
    value = record.get(name)
    if isinstance(value, bool):
        raise DomainLessonResultsError(f"{context}.{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(f"{context}.{name} must be finite") from exc
    if not np.isfinite(number):
        raise DomainLessonResultsError(f"{context}.{name} must be finite")
    return number


def _validated_charge_diagnostics(
    value: Any,
    *,
    atom_count: int,
    context: str,
    target_sum_e: float = 0.0,
) -> dict[str, Any]:
    """Validate one unmodified float32 charge tensor summary.

    The residual is recorded as a numerical diagnostic. This general check does
    not turn it into a pass/fail threshold for large one-GPU calculations.
    """

    if not isinstance(value, Mapping):
        raise DomainLessonResultsError(f"{context} must be an object")
    required = (
        "available",
        "finite",
        "dtype",
        "target_sum_e",
        "sum_e",
        "residual_e",
        "abs_residual_per_atom",
        "sum_abs_e",
        "max_abs_e",
        "shape",
        "sha256",
    )
    _require_keys(value, required, context=context)
    if value["available"] is not True:
        raise DomainLessonResultsError(f"{context} are not available")
    if value["finite"] is not True:
        raise DomainLessonResultsError(f"{context} contain non-finite charges")
    if value["dtype"] != "float32":
        raise DomainLessonResultsError(
            f"{context}.dtype must identify the float32 PME charge tensor"
        )
    if atom_count <= 0:
        raise DomainLessonResultsError(f"{context} atom count must be positive")

    expected_target = float(target_sum_e)
    if not np.isfinite(expected_target):
        raise DomainLessonResultsError(
            f"{context} expected total-charge target must be finite"
        )
    observed_target = _finite_charge_number(
        value,
        "target_sum_e",
        context=context,
    )
    if observed_target != expected_target:
        raise DomainLessonResultsError(
            f"{context}.target_sum_e does not match the input total charge"
        )

    shape = value["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        )
        or math.prod(shape) != atom_count
    ):
        raise DomainLessonResultsError(
            f"{context}.shape does not match the atom count"
        )
    digest = _require_sha256(
        value["sha256"],
        name=f"{context}.sha256",
    )

    charge_sum = _finite_charge_number(value, "sum_e", context=context)
    residual = _finite_charge_number(value, "residual_e", context=context)
    residual_per_atom = _finite_charge_number(
        value,
        "abs_residual_per_atom",
        context=context,
    )
    sum_abs = _finite_charge_number(value, "sum_abs_e", context=context)
    max_abs = _finite_charge_number(value, "max_abs_e", context=context)
    if not math.isclose(
        residual,
        charge_sum - observed_target,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise DomainLessonResultsError(
            f"{context}.residual_e is inconsistent with the charge sum"
        )
    if not math.isclose(
        residual_per_atom,
        abs(residual) / atom_count,
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        raise DomainLessonResultsError(
            f"{context}.abs_residual_per_atom is inconsistent"
        )

    magnitude_slack = 1.0e-12 * max(1.0, sum_abs, atom_count * max_abs)
    if (
        residual_per_atom < 0.0
        or sum_abs < 0.0
        or max_abs < 0.0
        or sum_abs + magnitude_slack < abs(charge_sum)
        or max_abs > sum_abs + magnitude_slack
        or atom_count * max_abs + magnitude_slack < abs(charge_sum)
        or sum_abs > atom_count * max_abs + magnitude_slack
    ):
        raise DomainLessonResultsError(
            f"{context} contain inconsistent charge magnitudes"
        )

    return {
        "available": True,
        "finite": True,
        "dtype": "float32",
        "target_sum_e": observed_target,
        "sum_e": charge_sum,
        "residual_e": residual,
        "abs_residual_per_atom": residual_per_atom,
        "sum_abs_e": sum_abs,
        "max_abs_e": max_abs,
        "shape": list(shape),
        "sha256": digest,
    }


def _raw_electrostatics_row(
    raw_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    matches = [
        row for row in raw_rows if row.get("mode") == "electrostatics-validation"
    ]
    if len(matches) != 1:
        raise DomainLessonResultsError(
            "raw results must contain exactly one electrostatics-validation row"
        )
    row = matches[0]
    _require_keys(
        row,
        (
            "status",
            "success",
            "atom_count",
            "input",
            "charges",
            "pme",
            "ewald",
            "comparison",
            "result_file_sha256",
        ),
        context="raw electrostatics row",
    )
    if row["status"] != "complete" or row["success"] is not True:
        raise DomainLessonResultsError(
            "raw electrostatics-validation row did not complete"
        )
    return row


def _validate_electrostatics(
    manifest: Mapping[str, Any],
    *,
    structure_sha256_by_atom_count: Mapping[int, str],
    raw_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the predeclared fixed-charge PME-versus-Ewald check."""

    record = _require_mapping(
        manifest,
        "electrostatics_validation",
        context="manifest",
    )
    required = (
        "status",
        "measurement_kind",
        "fixed_charges",
        "atom_count",
        "structure_sha256",
        "charge_diagnostics",
        "charge_sum_e",
        "charge_sum_tolerance_e",
        "pme_energy_ev",
        "ewald_energy_ev",
        "energy_abs_difference_ev_per_atom",
        "energy_tolerance_ev_per_atom",
        "force_rms_difference_ev_per_a",
        "force_max_difference_ev_per_a",
        "force_tolerance_ev_per_a",
        "charge_sha256",
        "pme_force_sha256",
        "ewald_force_sha256",
        "result_file_sha256",
    )
    _require_keys(record, required, context="electrostatics_validation")
    if record["status"] != "passed":
        raise DomainLessonResultsError("PME-versus-Ewald validation did not pass")
    if record["measurement_kind"] != "measured":
        raise DomainLessonResultsError("PME-versus-Ewald validation must be measured")
    if record["fixed_charges"] is not True:
        raise DomainLessonResultsError(
            "PME and Ewald must use the same fixed charge array"
        )
    try:
        atom_count = int(record["atom_count"])
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "electrostatics_validation.atom_count must be positive"
        ) from exc
    if atom_count <= 0 or float(record["atom_count"]) != atom_count:
        raise DomainLessonResultsError(
            "electrostatics_validation.atom_count must be positive"
        )
    if atom_count != ELECTROSTATICS_VALIDATION_ATOM_COUNT:
        raise DomainLessonResultsError(
            "the strict charge-residual check is reserved for the "
            f"{ELECTROSTATICS_VALIDATION_ATOM_COUNT:,}-atom "
            "PME-versus-Ewald validation"
        )
    expected_structure = structure_sha256_by_atom_count.get(atom_count)
    if expected_structure is None or record["structure_sha256"] != expected_structure:
        raise DomainLessonResultsError(
            "electrostatics validation structure does not match its atom count"
        )

    numeric_names = (
        "charge_sum_e",
        "charge_sum_tolerance_e",
        "pme_energy_ev",
        "ewald_energy_ev",
        "energy_abs_difference_ev_per_atom",
        "energy_tolerance_ev_per_atom",
        "force_rms_difference_ev_per_a",
        "force_max_difference_ev_per_a",
        "force_tolerance_ev_per_a",
    )
    values: dict[str, float] = {}
    for name in numeric_names:
        try:
            value = float(record[name])
        except (TypeError, ValueError) as exc:
            raise DomainLessonResultsError(
                f"electrostatics_validation.{name} must be finite"
            ) from exc
        if not np.isfinite(value):
            raise DomainLessonResultsError(
                f"electrostatics_validation.{name} must be finite"
            )
        values[name] = value

    expected_tolerances = {
        "charge_sum_tolerance_e": CHARGE_SUM_TOLERANCE_E,
        "energy_tolerance_ev_per_atom": PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM,
        "force_tolerance_ev_per_a": PME_EWALD_FORCE_TOLERANCE_EV_PER_A,
    }
    for name, expected in expected_tolerances.items():
        if not math.isclose(values[name], expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise DomainLessonResultsError(
                f"electrostatics_validation.{name} changed from the predeclared value"
            )
    manifest_charge_diagnostics = _validated_charge_diagnostics(
        record["charge_diagnostics"],
        atom_count=atom_count,
        target_sum_e=0.0,
        context="electrostatics_validation.charge_diagnostics",
    )
    if values["charge_sum_e"] != manifest_charge_diagnostics["sum_e"]:
        raise DomainLessonResultsError(
            "electrostatics_validation.charge_sum_e does not match "
            "charge_diagnostics"
        )
    if (
        abs(manifest_charge_diagnostics["residual_e"])
        > CHARGE_SUM_TOLERANCE_E
    ):
        raise DomainLessonResultsError(
            "the 3,200-atom PME-versus-Ewald charge residual exceeds "
            "the predeclared limit"
        )
    if (
        values["energy_abs_difference_ev_per_atom"] < 0.0
        or values["energy_abs_difference_ev_per_atom"]
        > PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM
    ):
        raise DomainLessonResultsError("PME-versus-Ewald energy check failed")
    for name in (
        "force_rms_difference_ev_per_a",
        "force_max_difference_ev_per_a",
    ):
        if values[name] < 0.0:
            raise DomainLessonResultsError(
                f"electrostatics_validation.{name} must be non-negative"
            )
    if values["force_max_difference_ev_per_a"] > PME_EWALD_FORCE_TOLERANCE_EV_PER_A:
        raise DomainLessonResultsError("PME-versus-Ewald force check failed")

    raw = _raw_electrostatics_row(raw_rows)
    try:
        raw_atom_count = int(raw["atom_count"])
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "raw electrostatics atom count must be a positive integer"
        ) from exc
    if (
        raw_atom_count <= 0
        or float(raw["atom_count"]) != raw_atom_count
        or raw_atom_count != atom_count
    ):
        raise DomainLessonResultsError(
            "raw electrostatics atom count does not match the manifest"
        )

    raw_input = _require_mapping(raw, "input", context="raw electrostatics row")
    raw_charges = _require_mapping(
        raw,
        "charges",
        context="raw electrostatics row",
    )
    raw_pme = _require_mapping(raw, "pme", context="raw electrostatics row")
    raw_ewald = _require_mapping(raw, "ewald", context="raw electrostatics row")
    raw_comparison = _require_mapping(
        raw,
        "comparison",
        context="raw electrostatics row",
    )
    raw_pme_forces = _require_mapping(
        raw_pme,
        "forces",
        context="raw electrostatics row.pme",
    )
    raw_ewald_forces = _require_mapping(
        raw_ewald,
        "forces",
        context="raw electrostatics row.ewald",
    )
    _require_keys(
        raw_input,
        ("file_sha256",),
        context="raw electrostatics row.input",
    )
    _require_keys(
        raw_charges,
        (
            "available",
            "finite",
            "dtype",
            "target_sum_e",
            "sum_e",
            "residual_e",
            "abs_residual_per_atom",
            "sum_abs_e",
            "max_abs_e",
            "shape",
            "sha256",
        ),
        context="raw electrostatics row.charges",
    )
    _require_keys(
        raw_pme,
        ("energy_ev",),
        context="raw electrostatics row.pme",
    )
    _require_keys(
        raw_ewald,
        ("energy_ev",),
        context="raw electrostatics row.ewald",
    )
    _require_keys(
        raw_pme_forces,
        ("sha256",),
        context="raw electrostatics row.pme.forces",
    )
    _require_keys(
        raw_ewald_forces,
        ("sha256",),
        context="raw electrostatics row.ewald.forces",
    )
    _require_keys(
        raw_comparison,
        (
            "absolute_energy_difference_ev_per_atom",
            "force_difference_rms_ev_a",
            "force_difference_max_norm_ev_a",
            "passed",
        ),
        context="raw electrostatics row.comparison",
    )
    if raw_charges["available"] is not True:
        raise DomainLessonResultsError(
            "raw electrostatics-validation row must include predicted charges"
        )
    if raw_comparison["passed"] is not True:
        raise DomainLessonResultsError(
            "raw electrostatics-validation comparison did not pass"
        )
    raw_charge_diagnostics = _validated_charge_diagnostics(
        raw_charges,
        atom_count=atom_count,
        target_sum_e=0.0,
        context="raw electrostatics row.charges",
    )

    raw_structure_sha256 = _require_sha256(
        raw_input["file_sha256"],
        name="raw electrostatics input SHA-256",
    )
    if raw_structure_sha256 != str(record["structure_sha256"]):
        raise DomainLessonResultsError(
            "raw electrostatics input SHA-256 does not match the manifest"
        )
    hash_pairs = (
        ("charge", record["charge_sha256"], raw_charges["sha256"]),
        ("PME force", record["pme_force_sha256"], raw_pme_forces["sha256"]),
        ("Ewald force", record["ewald_force_sha256"], raw_ewald_forces["sha256"]),
        (
            "original result file",
            record["result_file_sha256"],
            raw["result_file_sha256"],
        ),
    )
    for label, manifest_digest, raw_digest in hash_pairs:
        expected_digest = _require_sha256(
            manifest_digest,
            name=f"electrostatics_validation {label} SHA-256",
        )
        observed_digest = _require_sha256(
            raw_digest,
            name=f"raw electrostatics {label} SHA-256",
        )
        if observed_digest != expected_digest:
            raise DomainLessonResultsError(
                f"{label} SHA-256 does not match raw results"
            )

    raw_numeric_values = {
        "charge_sum_e": raw_charges["sum_e"],
        "pme_energy_ev": raw_pme["energy_ev"],
        "ewald_energy_ev": raw_ewald["energy_ev"],
        "energy_abs_difference_ev_per_atom": raw_comparison[
            "absolute_energy_difference_ev_per_atom"
        ],
        "force_rms_difference_ev_per_a": raw_comparison["force_difference_rms_ev_a"],
        "force_max_difference_ev_per_a": raw_comparison[
            "force_difference_max_norm_ev_a"
        ],
    }
    for name, raw_value in raw_numeric_values.items():
        try:
            observed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise DomainLessonResultsError(
                f"raw electrostatics {name} must be finite"
            ) from exc
        if not np.isfinite(observed):
            raise DomainLessonResultsError(f"raw electrostatics {name} must be finite")
        if observed != values[name]:
            raise DomainLessonResultsError(
                f"electrostatics_validation.{name} does not match raw results"
            )
    if raw_charge_diagnostics != manifest_charge_diagnostics:
        raise DomainLessonResultsError(
            "electrostatics_validation.charge_diagnostics do not match raw results"
        )


def _read_table(
    root: Path,
    manifest: Mapping[str, Any],
    table_name: str,
    checksums: Mapping[str, str],
) -> pd.DataFrame:
    data = _require_mapping(manifest, "data", context="manifest")
    metadata = _require_mapping(data, table_name, context="manifest.data")
    _require_keys(
        metadata,
        ("file", "sha256", "row_count", "columns", "planned_case_ids"),
        context=f"manifest.data.{table_name}",
    )
    filename = str(metadata["file"])
    if checksums.get(filename) != str(metadata["sha256"]):
        raise DomainLessonResultsError(
            f"{table_name} checksum index and manifest disagree"
        )
    path = _validate_file(root, filename, str(metadata["sha256"]))
    table = pd.read_csv(path, keep_default_na=False)
    expected_columns = TABLE_COLUMNS[table_name]
    if tuple(metadata["columns"]) != expected_columns:
        raise DomainLessonResultsError(
            f"{table_name} manifest columns do not match schema"
        )
    if tuple(table.columns) != expected_columns:
        raise DomainLessonResultsError(f"{table_name} CSV columns do not match schema")
    if len(table) != int(metadata["row_count"]):
        raise DomainLessonResultsError(f"{table_name} row count does not match")
    if table["case_id"].astype(str).duplicated().any():
        raise DomainLessonResultsError(f"{table_name} case_id values must be unique")
    planned = [str(item) for item in metadata["planned_case_ids"]]
    if len(set(planned)) != len(planned) or set(table["case_id"].astype(str)) != set(
        planned
    ):
        raise DomainLessonResultsError(
            f"{table_name} rows do not match the complete planned case list"
        )
    table["success"] = _coerce_boolean(table["success"], name=f"{table_name}.success")
    if "parity_passed" in table:
        parity_values = table["parity_passed"].replace("", pd.NA)
        nonempty = parity_values.notna()
        parsed = pd.Series(pd.NA, index=table.index, dtype="boolean")
        if nonempty.any():
            parsed.loc[nonempty] = _coerce_boolean(
                parity_values.loc[nonempty],
                name="parity.parity_passed",
            )
        table["parity_passed"] = parsed
    return table


def _finite_numeric(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    rows: pd.Series,
    table_name: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(table.loc[rows, column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise DomainLessonResultsError(
                f"{table_name}.{column} must be finite for successful rows"
            )


def _validate_rows(
    table_name: str,
    table: pd.DataFrame,
    *,
    structure_sha256_by_atom_count: Mapping[int, str],
    settings_sha256: str,
) -> None:
    roles = table["measurement_role"].astype(str)
    kinds = table["measurement_kind"].astype(str)
    expected_roles = {
        "capacity": {"capacity"},
        "parity": {"parity"},
        "distributed": {"steady_timing", "rescue"},
    }[table_name]
    if not set(roles).issubset(expected_roles):
        raise DomainLessonResultsError(f"{table_name} contains an invalid measurement role")
    expected_kinds = roles.map(
        {
            "capacity": "cold_one_shot_partition_run_gather",
            "parity": "cold_one_shot_partition_run_gather",
            "rescue": "cold_one_shot_partition_run_gather",
            "steady_timing": "steady_partition_run_gather",
        }
    )
    if not kinds.eq(expected_kinds).all():
        raise DomainLessonResultsError(
            f"{table_name} measurement kind does not match its role"
        )
    if not table["settings_sha256"].astype(str).eq(settings_sha256).all():
        raise DomainLessonResultsError(
            f"{table_name}.settings_sha256 does not match the manifest identity"
        )
    successes = table["success"]
    if not table.loc[successes, "status"].astype(str).eq("complete").all():
        raise DomainLessonResultsError(
            f"{table_name} successful rows must have status complete"
        )
    failed = ~successes
    if failed.any():
        if not table.loc[failed, "status"].astype(str).eq("failed").all():
            raise DomainLessonResultsError(
                f"{table_name} failed rows must have status failed"
            )
        for column in ("failure_type", "failure_stage", "error"):
            if table.loc[failed, column].astype(str).str.strip().eq("").any():
                raise DomainLessonResultsError(
                    f"{table_name} failed rows must report {column}"
                )
    for column in ("failure_type", "failure_stage", "error"):
        if table.loc[successes, column].astype(str).str.strip().ne("").any():
            raise DomainLessonResultsError(
                f"{table_name} successful rows must leave {column} empty"
            )

    integer_columns = {
        "capacity": ("atom_count", "molecules_per_species", "gpus"),
        "parity": (
            "atom_count",
            "force_reference_gpus",
            "energy_reference_gpus",
            "gpus",
        ),
        "distributed": (
            "atom_count",
            "molecules_per_species",
            "nodes",
            "gpus",
            "ranks",
        ),
    }[table_name]
    for column in integer_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.rint(values)).all()
            or np.any(values <= 0)
        ):
            raise DomainLessonResultsError(
                f"{table_name}.{column} must contain positive integers"
            )
        table[column] = values.astype(np.int64)
    expected_structure_hashes = table["atom_count"].map(structure_sha256_by_atom_count)
    if (
        expected_structure_hashes.isna().any()
        or not table["structure_sha256"]
        .astype(str)
        .eq(expected_structure_hashes.astype(str))
        .all()
    ):
        raise DomainLessonResultsError(
            f"{table_name}.structure_sha256 does not match its atom count"
        )

    successful_numeric = {
        "capacity": (
            "elapsed_s",
            "peak_memory_bytes_max_rank",
            "energy_ev",
            "force_rms_ev_per_a",
            "force_max_ev_per_a",
        ),
        "parity": (
            "one_gpu_energy_abs_offset_ev",
            "one_gpu_energy_abs_offset_ev_per_atom",
            "distributed_energy_difference_ev",
            "distributed_energy_difference_ev_per_atom",
            "force_rms_difference_ev_per_a",
            "force_max_difference_ev_per_a",
            "energy_tolerance_ev_per_atom",
            "force_tolerance_ev_per_a",
        ),
        "distributed": (
            "elapsed_s",
            "warmup_count",
            "sample_count",
            "elapsed_median_s",
            "elapsed_q1_s",
            "elapsed_q3_s",
            "elapsed_iqr_s",
            "peak_memory_bytes_max_rank",
            "owned_atoms_min_rank",
            "owned_atoms_max_rank",
            "energy_ev",
            "force_rms_ev_per_a",
            "force_max_ev_per_a",
        ),
    }[table_name]
    _finite_numeric(
        table,
        successful_numeric,
        rows=successes,
        table_name=table_name,
    )
    if table_name == "capacity" and not table["gpus"].eq(1).all():
        raise DomainLessonResultsError("capacity rows must use one GPU")
    if table_name == "parity":
        if not (
            table.loc[
                successes,
                ["distributed_energy_passed", "force_passed", "parity_passed"],
            ]
            .eq(True)
            .all()
            .all()
        ):
            raise DomainLessonResultsError("successful parity rows must pass")
        nonnegative_columns = (
            "one_gpu_energy_abs_offset_ev",
            "one_gpu_energy_abs_offset_ev_per_atom",
            "distributed_energy_difference_ev",
            "distributed_energy_difference_ev_per_atom",
            "force_rms_difference_ev_per_a",
            "force_max_difference_ev_per_a",
        )
        if any(
            (
                pd.to_numeric(table.loc[successes, column], errors="coerce") < 0.0
            ).any()
            for column in nonnegative_columns
        ):
            raise DomainLessonResultsError(
                "parity energy and force differences must be non-negative"
            )
        if not (
            pd.to_numeric(
                table.loc[
                    successes,
                    "distributed_energy_difference_ev_per_atom",
                ],
                errors="coerce",
            )
            <= pd.to_numeric(
                table.loc[successes, "energy_tolerance_ev_per_atom"],
                errors="coerce",
            )
        ).all():
            raise DomainLessonResultsError(
                "distributed energy difference exceeds tolerance"
            )
        if not (
            pd.to_numeric(
                table.loc[successes, "force_max_difference_ev_per_a"],
                errors="coerce",
            )
            <= pd.to_numeric(
                table.loc[successes, "force_tolerance_ev_per_a"], errors="coerce"
            )
        ).all():
            raise DomainLessonResultsError("parity force difference exceeds tolerance")
    if table_name == "distributed":
        matching_topology = table["nodes"].eq(table["gpus"]) & table["gpus"].eq(
            table["ranks"]
        )
        if not matching_topology.all():
            raise DomainLessonResultsError(
                "distributed rows must satisfy nodes == gpus == ranks"
            )
        for index, row in table.loc[successes].iterrows():
            try:
                samples = json.loads(str(row["elapsed_samples_s"]))
                values = np.asarray(samples, dtype=float)
                warmup_count = int(row["warmup_count"])
                sample_count = int(row["sample_count"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DomainLessonResultsError(
                    "distributed timing samples and counts are invalid"
                ) from exc
            if (
                not isinstance(samples, list)
                or values.ndim != 1
                or len(values) != sample_count
                or not np.isfinite(values).all()
                or np.any(values <= 0.0)
            ):
                raise DomainLessonResultsError(
                    "distributed elapsed_samples_s must contain every positive sample"
                )
            role = str(row["measurement_role"])
            expected_counts = (
                (
                    DOMAIN_METHODOLOGY.steady_timing_warmup_count,
                    DOMAIN_METHODOLOGY.steady_timing_sample_count,
                )
                if role == "steady_timing"
                else (0, 1)
            )
            if (warmup_count, sample_count) != expected_counts:
                raise DomainLessonResultsError(
                    "distributed timing counts do not match the measurement role"
                )
            q1, median_value, q3 = np.quantile(
                values,
                (0.25, 0.5, 0.75),
                method="linear",
            )
            expected = {
                "elapsed_s": median_value,
                "elapsed_median_s": median_value,
                "elapsed_q1_s": q1,
                "elapsed_q3_s": q3,
                "elapsed_iqr_s": q3 - q1,
            }
            if any(
                not math.isclose(
                    float(row[name]),
                    float(value),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for name, value in expected.items()
            ):
                raise DomainLessonResultsError(
                    f"distributed timing statistics do not match samples at row {index}"
                )
            relative_iqr = float(q3 - q1) / float(median_value)
            if (
                role == "steady_timing"
                and relative_iqr
                > DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr + 1.0e-12
            ):
                raise DomainLessonResultsError(
                    "steady timing is too variable to report: relative IQR "
                    f"{relative_iqr:.1%} exceeds "
                    f"{DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr:.1%} "
                    f"at row {index}"
                )
        for world_size, success, grid_text in zip(
            table["gpus"].astype(int),
            table["success"].astype(bool),
            table["spatial_grid"],
            strict=True,
        ):
            _parse_spatial_grid(
                grid_text,
                world_size=world_size,
                context="successful distributed row",
                required=success,
            )


def _case_kinds(
    table: pd.DataFrame,
    *,
    pattern: re.Pattern[str],
    table_name: str,
    default_kind: str | None = None,
) -> pd.Series:
    """Check case IDs against their row values and return each case kind."""

    kinds: list[str] = []
    for row in table.itertuples(index=False):
        case_id = str(row.case_id)
        match = pattern.fullmatch(case_id)
        if match is None:
            raise DomainLessonResultsError(
                f"{table_name} case_id {case_id!r} does not match the campaign plan"
            )
        if int(match.group("gpus")) != int(row.gpus):
            raise DomainLessonResultsError(
                f"{table_name} case_id GPU count does not match its gpus column"
            )
        pair_count = int(match.group("pair_count"))
        if pair_count * ATOMS_PER_COMPOSITION_UNIT != int(row.atom_count):
            raise DomainLessonResultsError(
                f"{table_name} rows have a case_id pair count that does not match "
                "atom_count"
            )
        if hasattr(row, "molecules_per_species") and pair_count != int(
            row.molecules_per_species
        ):
            raise DomainLessonResultsError(
                f"{table_name} case_id pair count does not match molecules_per_species"
            )
        kinds.append(
            default_kind if default_kind is not None else str(match.group("kind"))
        )
    return pd.Series(kinds, index=table.index, dtype="string")


def _has_exact_gpu_rows(table: pd.DataFrame, required: frozenset[int]) -> bool:
    observed = table["gpus"].astype(int)
    return len(observed) == len(required) and set(observed) == set(required)


def _validate_measurement_completeness(
    parity: pd.DataFrame,
    distributed: pd.DataFrame,
) -> None:
    """Require every saved row used by the recorded multi-GPU lesson."""

    _case_kinds(
        parity,
        pattern=_PARITY_CASE_ID,
        table_name="parity",
        default_kind="parity",
    )
    if not _has_exact_gpu_rows(parity, _REQUIRED_PARITY_GPUS):
        raise DomainLessonResultsError(
            "parity measurements require exactly one row for each declared "
            "multi-GPU size"
        )
    if not parity["force_reference_gpus"].eq(
        DOMAIN_METHODOLOGY.force_reference_world_size
    ).all():
        raise DomainLessonResultsError(
            "parity force rows must use a one-GPU reference"
        )
    if not parity["energy_reference_gpus"].eq(
        DOMAIN_METHODOLOGY.energy_reference_world_size
    ).all():
        raise DomainLessonResultsError(
            "parity energy rows must use a two-GPU distributed reference"
        )
    if parity["atom_count"].nunique() != 1:
        raise DomainLessonResultsError(
            "parity measurements must use the same structure size"
        )
    if not parity["success"].eq(True).all() or not (
        parity["parity_passed"].eq(True).fillna(False).all()
    ):
        raise DomainLessonResultsError(
            "all declared multi-GPU parity measurements must succeed and pass"
        )

    kinds = _case_kinds(
        distributed,
        pattern=_DISTRIBUTED_CASE_ID,
        table_name="distributed",
    )
    steady = distributed.loc[kinds.eq("steady-timing")]
    if not _has_exact_gpu_rows(steady, _REQUIRED_STEADY_TIMING_GPUS):
        raise DomainLessonResultsError(
            "steady timing requires exactly one row for each declared GPU size"
        )
    if not steady["success"].all():
        raise DomainLessonResultsError("all steady-timing rows must succeed")
    if steady["atom_count"].nunique() != 1:
        raise DomainLessonResultsError(
            "steady-timing measurements must use the same structure"
        )
    if not steady["measurement_role"].astype(str).eq("steady_timing").all():
        raise DomainLessonResultsError("steady-timing case IDs need steady_timing roles")

    rescue = distributed.loc[kinds.eq("rescue")]
    if not _has_exact_gpu_rows(rescue, _REQUIRED_RESCUE_GPUS):
        raise DomainLessonResultsError(
            "rescue measurements require exactly one attempt for each declared "
            "multi-GPU size"
        )
    if not rescue["success"].any():
        raise DomainLessonResultsError("at least one rescue attempt must succeed")
    if rescue["atom_count"].nunique() != 1:
        raise DomainLessonResultsError(
            "rescue attempts must use the same structure size"
        )
    if not rescue["measurement_role"].astype(str).eq("rescue").all():
        raise DomainLessonResultsError("rescue case IDs need rescue roles")


def _validate_selection(
    manifest: Mapping[str, Any],
    capacity: pd.DataFrame,
    parity: pd.DataFrame,
    distributed: pd.DataFrame,
    raw_by_case: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    selection = _require_mapping(manifest, "selection", context="manifest")
    required = (
        "largest_success_pair_count",
        "first_cuda_oom_pair_count",
        "parity_pair_count",
        "capacity_charge_diagnostics",
        "parity_charge_diagnostics",
        "successful_rescue_gpu_counts",
    )
    _require_keys(selection, required, context="manifest.selection")
    try:
        largest_pairs = _positive_integer(
            selection["largest_success_pair_count"],
            name="manifest largest-success 1:1 composition count",
        )
        oom_pairs = _positive_integer(
            selection["first_cuda_oom_pair_count"],
            name="manifest first-CUDA-OOM 1:1 composition count",
        )
        parity_pairs = _positive_integer(
            selection["parity_pair_count"],
            name="manifest parity 1:1 composition count",
        )
    except ValueError as exc:
        raise DomainLessonResultsError(str(exc)) from exc

    successful_capacity = capacity.loc[capacity["success"].eq(True)]
    if successful_capacity.empty:
        raise DomainLessonResultsError("capacity sweep has no successful row")
    largest_success_atoms = int(successful_capacity["atom_count"].max())
    oom_capacity = capacity.loc[
        ~capacity["success"] & capacity["failure_type"].astype(str).isin(_OOM_TYPES)
    ]
    if oom_capacity.empty:
        raise DomainLessonResultsError("capacity sweep has no CUDA OOM row")
    first_oom_atoms = int(oom_capacity.iloc[0]["atom_count"])
    if largest_pairs * ATOMS_PER_COMPOSITION_UNIT != largest_success_atoms:
        raise DomainLessonResultsError(
            "manifest largest successful size does not match the capacity ladder"
        )
    if oom_pairs * ATOMS_PER_COMPOSITION_UNIT != first_oom_atoms:
        raise DomainLessonResultsError(
            "manifest first CUDA OOM size does not match the capacity ladder"
        )
    if not parity["atom_count"].eq(
        parity_pairs * ATOMS_PER_COMPOSITION_UNIT
    ).all():
        raise DomainLessonResultsError(
            "manifest parity size does not match measured parity rows"
        )

    kinds = _case_kinds(
        distributed,
        pattern=_DISTRIBUTED_CASE_ID,
        table_name="distributed",
    )
    steady = distributed.loc[kinds.eq("steady-timing")]
    rescue = distributed.loc[kinds.eq("rescue")]
    if not steady["atom_count"].eq(largest_success_atoms).all():
        raise DomainLessonResultsError(
            "steady-timing rows must use the largest successful one-GPU input"
        )
    if not rescue["atom_count"].eq(first_oom_atoms).all():
        raise DomainLessonResultsError(
            "rescue rows must use the first one-GPU CUDA OOM input"
        )

    raw_rescue_gpus = selection["successful_rescue_gpu_counts"]
    if not isinstance(raw_rescue_gpus, list):
        raise DomainLessonResultsError(
            "manifest.selection.successful_rescue_gpu_counts must be a list"
        )
    try:
        declared_rescue_gpus = [
            _positive_integer(
                value,
                name="manifest successful rescue GPU count",
            )
            for value in raw_rescue_gpus
        ]
    except ValueError as exc:
        raise DomainLessonResultsError(str(exc)) from exc
    observed_rescue_gpus = sorted(
        int(value)
        for value in rescue.loc[rescue["success"].eq(True), "gpus"].tolist()
    )
    if (
        declared_rescue_gpus != sorted(set(declared_rescue_gpus))
        or declared_rescue_gpus != observed_rescue_gpus
    ):
        raise DomainLessonResultsError(
            "manifest successful rescue GPU counts do not match measured rows"
        )

    raw_capacity_diagnostics = selection["capacity_charge_diagnostics"]
    if not isinstance(raw_capacity_diagnostics, list):
        raise DomainLessonResultsError(
            "manifest.selection.capacity_charge_diagnostics must be a list"
        )
    expected_capacity_rows = successful_capacity.reset_index(drop=True)
    if len(raw_capacity_diagnostics) != len(expected_capacity_rows):
        raise DomainLessonResultsError(
            "manifest.selection.capacity_charge_diagnostics must describe every "
            "successful one-GPU capacity row"
        )

    validated_by_pair_count: dict[int, dict[str, Any]] = {}
    for index, expected_row in expected_capacity_rows.iterrows():
        context = f"manifest.selection.capacity_charge_diagnostics[{index}]"
        item = raw_capacity_diagnostics[index]
        if not isinstance(item, Mapping):
            raise DomainLessonResultsError(f"{context} must be an object")
        _require_keys(
            item,
            ("case_id", "pair_count", "atom_count", "charge_diagnostics"),
            context=context,
        )
        case_id = str(item["case_id"])
        expected_case_id = str(expected_row["case_id"])
        if case_id != expected_case_id:
            raise DomainLessonResultsError(
                "manifest.selection.capacity_charge_diagnostics must follow the "
                "successful capacity rows in order"
            )
        try:
            atom_count = _positive_integer(
                item["atom_count"],
                name=f"{context}.atom_count",
            )
            pair_count = _positive_integer(
                item["pair_count"],
                name=f"{context}.pair_count",
            )
        except ValueError as exc:
            raise DomainLessonResultsError(str(exc)) from exc
        if (
            atom_count != int(expected_row["atom_count"])
            or pair_count * ATOMS_PER_COMPOSITION_UNIT != atom_count
        ):
            raise DomainLessonResultsError(
                f"{context} does not match its successful capacity row"
            )

        selected_diagnostics = _validated_charge_diagnostics(
            item["charge_diagnostics"],
            atom_count=atom_count,
            target_sum_e=0.0,
            context=f"{context}.charge_diagnostics",
        )
        raw_row = raw_by_case.get(case_id)
        if raw_row is None:
            raise DomainLessonResultsError(
                f"{context} has no matching raw capacity row"
            )
        raw_charges = _require_mapping(
            raw_row,
            "charges",
            context=f"raw capacity row {case_id!r}",
        )
        observed_diagnostics = _validated_charge_diagnostics(
            raw_charges,
            atom_count=atom_count,
            target_sum_e=0.0,
            context=f"raw capacity row {case_id!r}.charges",
        )
        if selected_diagnostics != observed_diagnostics:
            raise DomainLessonResultsError(
                f"{context}.charge_diagnostics do not match raw results"
            )
        if pair_count in validated_by_pair_count:
            raise DomainLessonResultsError(
                "capacity charge diagnostics contain a duplicate pair count"
            )
        validated_by_pair_count[pair_count] = selected_diagnostics

    expected_parity_diagnostics = validated_by_pair_count.get(parity_pairs)
    if expected_parity_diagnostics is None:
        raise DomainLessonResultsError(
            "manifest parity size has no successful one-GPU charge diagnostics"
        )
    parity_diagnostics = _validated_charge_diagnostics(
        selection["parity_charge_diagnostics"],
        atom_count=parity_pairs * ATOMS_PER_COMPOSITION_UNIT,
        target_sum_e=0.0,
        context="manifest.selection.parity_charge_diagnostics",
    )
    if parity_diagnostics != expected_parity_diagnostics:
        raise DomainLessonResultsError(
            "manifest.selection.parity_charge_diagnostics do not match the "
            "selected one-GPU capacity row"
        )
    return selection


def _raw_failure_peak_memory(row: Mapping[str, Any]) -> int | None:
    values = [
        int(memory["max_allocated_bytes"])
        for record in row.get("rank_records", ())
        if isinstance(record, Mapping)
        and isinstance((memory := record.get("memory")), Mapping)
        and memory.get("max_allocated_bytes") is not None
    ]
    return max(values) if values else None


def _same_number(left: Any, right: Any) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(left_value)
        and np.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=1.0e-12, abs_tol=1.0e-12)
    )


def _layout_triplet(value: Any, *, name: str, context: str) -> tuple[int, int, int]:
    """Read one three-dimensional integer layout from saved JSON."""

    if not isinstance(value, list) or len(value) != 3:
        raise DomainLessonResultsError(
            f"{context}.{name} must contain three positive integers"
        )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise DomainLessonResultsError(
            f"{context}.{name} must contain three positive integers"
        )
    return value[0], value[1], value[2]


def _validate_recorded_layout(
    distributed: Mapping[str, Any],
    *,
    world_size: int,
    context: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Check a layout selected for one input by Toolkit SpatialPartitioner."""

    cells_per_dim = _layout_triplet(
        distributed.get("cells_per_dim"),
        name="cells_per_dim",
        context=context,
    )
    rank_grid = _layout_triplet(
        distributed.get("rank_grid"),
        name="rank_grid",
        context=context,
    )
    if math.prod(rank_grid) != world_size:
        raise DomainLessonResultsError(
            f"{context}.rank_grid does not match {world_size} ranks"
        )
    if any(
        ranks > cells
        for ranks, cells in zip(rank_grid, cells_per_dim, strict=True)
    ):
        raise DomainLessonResultsError(
            f"{context}.rank_grid exceeds cells_per_dim"
        )
    if any(
        cells % ranks != 0
        for ranks, cells in zip(rank_grid, cells_per_dim, strict=True)
    ):
        raise DomainLessonResultsError(
            f"{context}.rank_grid does not divide cells_per_dim"
        )
    return cells_per_dim, rank_grid


def _raw_recorded_layout(
    raw: Mapping[str, Any],
    *,
    world_size: int,
    context: str,
    required: bool = True,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Read one consistent layout from a successful row or failed rank records."""

    distributed = raw.get("distributed")
    if isinstance(distributed, Mapping):
        return _validate_recorded_layout(
            distributed,
            world_size=world_size,
            context=f"{context}.distributed",
        )

    records = raw.get("rank_records")
    if not isinstance(records, list):
        if not required:
            return None
        raise DomainLessonResultsError(
            f"{context} does not report its Toolkit spatial layout"
        )
    layouts: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        if record.get("cells_per_dim") is None and record.get("rank_grid") is None:
            continue
        layouts.append(
            _validate_recorded_layout(
                record,
                world_size=world_size,
                context=f"{context}.rank_records[{index}]",
            )
        )
    if not layouts:
        if not required:
            return None
        raise DomainLessonResultsError(
            f"{context} does not report its Toolkit spatial layout"
        )
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise DomainLessonResultsError(
            f"{context} rank records disagree on the Toolkit spatial layout"
        )
    return layouts[0]


def _parse_spatial_grid(
    value: Any,
    *,
    world_size: int,
    context: str,
    required: bool,
) -> tuple[int, int, int] | None:
    """Read the compact rank-grid column while allowing pre-layout failures."""

    if pd.isna(value) or not str(value).strip():
        if required:
            raise DomainLessonResultsError(
                f"{context} must report its spatial_grid"
            )
        return None
    text = str(value)
    try:
        dimensions = tuple(int(item) for item in text.split("x"))
    except ValueError as exc:
        raise DomainLessonResultsError(
            f"invalid distributed spatial_grid {text!r}"
        ) from exc
    if (
        len(dimensions) != 3
        or any(item <= 0 for item in dimensions)
        or math.prod(dimensions) != world_size
    ):
        raise DomainLessonResultsError(
            f"spatial_grid {text!r} does not match {world_size} ranks"
        )
    return dimensions[0], dimensions[1], dimensions[2]


def _reconcile_raw_row(
    table_name: str,
    row: pd.Series,
    raw: Mapping[str, Any],
    *,
    compare_case_id: bool = True,
    compare_distributed_details: bool = True,
) -> None:
    context = f"{table_name} row {row['case_id']!r}"
    required = (
        "status",
        "success",
        "world_size",
        "pair_count",
        "atom_count",
        "input",
    )
    _require_keys(raw, required, context=f"raw {context}")
    raw_input = _require_mapping(raw, "input", context=f"raw {context}")
    _require_keys(raw_input, ("file_sha256",), context=f"raw {context}.input")
    expected_pair_count = int(row["atom_count"]) // ATOMS_PER_COMPOSITION_UNIT
    if "molecules_per_species" in row:
        expected_pair_count = int(row["molecules_per_species"])
    exact_pairs = (
        ("status", str(row["status"]), str(raw["status"])),
        ("success", bool(row["success"]), raw["success"]),
        ("world_size", int(row["gpus"]), raw["world_size"]),
        ("atom_count", int(row["atom_count"]), raw["atom_count"]),
        (
            "molecules_per_species",
            expected_pair_count,
            raw["pair_count"],
        ),
        (
            "structure_sha256",
            str(row["structure_sha256"]),
            raw_input["file_sha256"],
        ),
        (
            "measurement_role",
            str(row["measurement_role"]),
            raw.get("measurement_role"),
        ),
    )
    if compare_case_id:
        exact_pairs = (
            ("case_id", str(row["case_id"]), raw.get("case_id")),
            *exact_pairs,
        )
    for name, expected, observed in exact_pairs:
        if observed != expected:
            raise DomainLessonResultsError(
                f"{context} does not match raw measurement field {name}"
            )

    if table_name == "parity":
        _raw_recorded_layout(
            raw,
            world_size=int(row["gpus"]),
            context=f"raw {context}",
        )
        return

    if bool(row["success"]):
        timing = _require_mapping(raw, "timing", context=f"raw {context}")
        memory = _require_mapping(raw, "memory", context=f"raw {context}")
        output = _require_mapping(raw, "output", context=f"raw {context}")
        forces = _require_mapping(
            output,
            "forces_source_atom_order",
            context=f"raw {context}.output",
        )
        numeric_pairs = (
            ("elapsed_s", row["elapsed_s"], timing.get("wall_s_max_rank")),
            (
                "peak_memory_bytes_max_rank",
                row["peak_memory_bytes_max_rank"],
                memory.get("max_allocated_bytes"),
            ),
            ("energy_ev", row["energy_ev"], output.get("energy_ev")),
            (
                "force_rms_ev_per_a",
                row["force_rms_ev_per_a"],
                forces.get("rms_ev_a"),
            ),
            (
                "force_max_ev_per_a",
                row["force_max_ev_per_a"],
                forces.get("max_norm_ev_a"),
            ),
        )
        for name, expected, observed in numeric_pairs:
            if not _same_number(expected, observed):
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement field {name}"
                )
        if table_name == "distributed" and compare_distributed_details:
            raw_samples = timing.get("samples_s_max_rank")
            try:
                csv_samples = json.loads(str(row["elapsed_samples_s"]))
            except json.JSONDecodeError as exc:
                raise DomainLessonResultsError(
                    f"{context} has invalid elapsed sample JSON"
                ) from exc
            if raw_samples != csv_samples:
                raise DomainLessonResultsError(
                    f"{context} does not match raw timing samples"
                )
            for name, raw_name in (
                ("warmup_count", "warmup_count"),
                ("sample_count", "sample_count"),
                ("elapsed_median_s", "median_s"),
                ("elapsed_q1_s", "q1_s"),
                ("elapsed_q3_s", "q3_s"),
                ("elapsed_iqr_s", "iqr_s"),
            ):
                if not _same_number(row[name], timing.get(raw_name)):
                    raise DomainLessonResultsError(
                        f"{context} does not match raw timing field {raw_name}"
                    )
            if str(row["measurement_role"]) == "steady_timing":
                world_size = int(row["gpus"])
                methodology = DOMAIN_METHODOLOGY
                expected_run_steps = methodology.steady_timing_run_steps(world_size)
                multi_rank_initial = (
                    methodology.domain_parallel_multi_rank_initial_force_evaluations
                )
                expected_initial_evaluations = (
                    multi_rank_initial if world_size > 1 else 0
                )
                expected_model_evaluations = (
                    methodology.steady_timing_model_evaluations_per_workflow
                )
                if (
                    timing.get("run_steps") != expected_run_steps
                    or timing.get("automatic_multi_rank_force_prime")
                    is not (world_size > 1)
                    or timing.get("automatic_initial_force_evaluations")
                    != expected_initial_evaluations
                    or timing.get("model_evaluations_per_workflow")
                    != expected_model_evaluations
                ):
                    raise DomainLessonResultsError(
                        f"{context} does not record equal steady-timing work"
                    )
            distributed = _require_mapping(
                raw,
                "distributed",
                context=f"raw {context}",
            )
            owned = distributed.get("owned_atom_counts")
            if not isinstance(owned, list) or not owned:
                raise DomainLessonResultsError(
                    f"raw {context}.distributed.owned_atom_counts must be nonempty"
                )
            if (
                int(row["owned_atoms_min_rank"]) != min(int(value) for value in owned)
                or int(row["owned_atoms_max_rank"])
                != max(int(value) for value in owned)
            ):
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement owned atom counts"
                )
            raw_grid = distributed.get("rank_grid")
            csv_grid = _parse_spatial_grid(
                row["spatial_grid"],
                world_size=int(row["gpus"]),
                context=context,
                required=True,
            )
            raw_layout = _raw_recorded_layout(
                raw,
                world_size=int(row["gpus"]),
                context=f"raw {context}",
            )
            if raw_layout is None:
                raise DomainLessonResultsError(
                    f"raw {context} does not report its Toolkit spatial layout"
                )
            _cells_per_dim, validated_grid = raw_layout
            if (
                not isinstance(raw_grid, list)
                or validated_grid != csv_grid
            ):
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement spatial grid"
                )
    else:
        failure = _require_mapping(raw, "failure", context=f"raw {context}")
        for name, column in (
            ("type", "failure_type"),
            ("stage", "failure_stage"),
            ("message", "error"),
        ):
            if str(failure.get(name, "")) != str(row[column]):
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement failure {name}"
                )
        peak = _raw_failure_peak_memory(raw)
        reported_peak = str(row["peak_memory_bytes_max_rank"]).strip()
        if peak is None:
            if reported_peak:
                raise DomainLessonResultsError(
                    f"{context} reports peak memory absent from raw measurement"
                )
        elif not _same_number(row["peak_memory_bytes_max_rank"], peak):
            raise DomainLessonResultsError(
                f"{context} does not match raw measurement peak memory"
            )
        if table_name == "distributed":
            raw_layout = _raw_recorded_layout(
                raw,
                world_size=int(row["gpus"]),
                context=f"raw {context}",
                required=False,
            )
            csv_grid = _parse_spatial_grid(
                row["spatial_grid"],
                world_size=int(row["gpus"]),
                context=context,
                required=False,
            )
            if raw_layout is None and csv_grid is None:
                return
            if raw_layout is None or csv_grid is None:
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement spatial grid"
                )
            _cells_per_dim, raw_grid = raw_layout
            if raw_grid != csv_grid:
                raise DomainLessonResultsError(
                    f"{context} does not match raw measurement spatial grid"
                )


def _reconcile_raw_measurements(
    tables: Mapping[str, pd.DataFrame],
    raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    measurement_rows = [
        row for row in raw_rows if row.get("mode") != "electrostatics-validation"
    ]
    raw_by_case: dict[str, Mapping[str, Any]] = {}
    for row in measurement_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in raw_by_case:
            raise DomainLessonResultsError(
                "raw capacity/parity/distributed case IDs must be nonempty and unique"
            )
        raw_by_case[case_id] = row

    expected_case_ids = set(tables["capacity"]["case_id"].astype(str))
    expected_case_ids.update(tables["parity"]["case_id"].astype(str))
    distributed = tables["distributed"]
    expected_case_ids.update(distributed["case_id"].astype(str))
    if set(raw_by_case) != expected_case_ids:
        raise DomainLessonResultsError(
            "raw capacity/parity/distributed rows do not match the complete CSV case set"
        )

    for table_name in ("capacity", "parity", "distributed"):
        for _, row in tables[table_name].iterrows():
            case_id = str(row["case_id"])
            raw = raw_by_case[case_id]
            if table_name == "distributed":
                expected_mode = (
                    "steady-timing"
                    if row["measurement_role"] == "steady_timing"
                    else "distributed"
                )
            else:
                expected_mode = table_name
            if raw.get("mode") != expected_mode:
                raise DomainLessonResultsError(
                    f"{table_name} row {case_id!r} has the wrong raw measurement mode"
                )
            if table_name == "parity":
                _reconcile_raw_row(table_name, row, raw)
            else:
                _reconcile_raw_row(table_name, row, raw)
    return raw_by_case


def _load_raw_force_array(
    root: Path,
    checksums: Mapping[str, str],
    raw: Mapping[str, Any],
    *,
    context: str,
) -> np.ndarray:
    output = _require_mapping(raw, "output", context=context)
    record = _require_mapping(
        output,
        "forces_source_atom_order_npy",
        context=f"{context}.output",
    )
    _require_keys(record, ("path", "sha256", "shape"), context=f"{context} force array")
    filename = str(record["path"])
    digest = _require_sha256(
        record["sha256"],
        name=f"{context} force array SHA-256",
    )
    if checksums.get(filename) != digest:
        raise DomainLessonResultsError(
            f"{context} force array checksum index and raw result disagree"
        )
    path = _validate_file(root, filename, digest)
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DomainLessonResultsError(f"{context} force array is not valid NPY") from exc
    if (
        values.ndim != 2
        or values.shape[1:] != (3,)
        or list(values.shape) != list(record["shape"])
        or not np.isfinite(values).all()
    ):
        raise DomainLessonResultsError(
            f"{context} force array shape or values are invalid"
        )
    return values.astype(np.float64, copy=False)


def _validate_raw_parity(
    root: Path,
    checksums: Mapping[str, str],
    parity: pd.DataFrame,
    raw_by_case: Mapping[str, Mapping[str, Any]],
) -> None:
    atom_count = int(parity.iloc[0]["atom_count"])
    one_gpu_references = [
        raw
        for raw in raw_by_case.values()
        if raw.get("mode") == "capacity"
        and raw.get("success") is True
        and int(raw.get("atom_count", -1)) == atom_count
    ]
    if len(one_gpu_references) != 1:
        raise DomainLessonResultsError(
            "parity rows need exactly one same-input raw one-GPU reference"
        )
    one_gpu_reference = one_gpu_references[0]
    one_gpu_forces = _load_raw_force_array(
        root,
        checksums,
        one_gpu_reference,
        context="raw one-GPU parity reference",
    )
    if one_gpu_forces.shape != (atom_count, 3):
        raise DomainLessonResultsError(
            "raw one-GPU parity reference force shape does not match atom count"
        )
    one_gpu_output = _require_mapping(
        one_gpu_reference,
        "output",
        context="raw one-GPU parity reference",
    )
    try:
        one_gpu_energy = float(one_gpu_output["energy_ev"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "raw one-GPU parity reference energy must be finite"
        ) from exc
    if not np.isfinite(one_gpu_energy):
        raise DomainLessonResultsError(
            "raw one-GPU parity reference energy must be finite"
        )

    energy_reference_world_size = DOMAIN_METHODOLOGY.energy_reference_world_size
    energy_comparison_world_sizes = frozenset(
        DOMAIN_METHODOLOGY.energy_comparison_world_sizes
    )
    energy_reference_rows = parity.loc[
        parity["gpus"].eq(energy_reference_world_size)
    ]
    if len(energy_reference_rows) != 1:
        raise DomainLessonResultsError(
            "parity rows need exactly one declared distributed energy reference"
        )
    energy_reference_case_id = str(energy_reference_rows.iloc[0]["case_id"])
    distributed_energy_reference = raw_by_case[energy_reference_case_id]
    distributed_energy_output = _require_mapping(
        distributed_energy_reference,
        "output",
        context="raw distributed energy reference",
    )
    try:
        distributed_reference_energy = float(distributed_energy_output["energy_ev"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            "raw distributed reference energy must be finite"
        ) from exc
    if not np.isfinite(distributed_reference_energy):
        raise DomainLessonResultsError(
            "raw distributed reference energy must be finite"
        )

    expected_energy_tolerance_per_atom = (
        DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
    )
    component_tolerances = (
        DOMAIN_METHODOLOGY.parity_force_atol_ev_a
        + DOMAIN_METHODOLOGY.parity_force_rtol * np.abs(one_gpu_forces)
    )
    expected_force_tolerance = float(component_tolerances.max())
    for _, row in parity.iterrows():
        case_id = str(row["case_id"])
        raw = raw_by_case[case_id]
        observed_forces = _load_raw_force_array(
            root,
            checksums,
            raw,
            context=f"raw parity row {case_id!r}",
        )
        if observed_forces.shape != one_gpu_forces.shape:
            raise DomainLessonResultsError(
                f"raw parity row {case_id!r} force shape differs from its reference"
            )
        difference = observed_forces - one_gpu_forces
        force_passed = bool(
            np.less_equal(np.abs(difference), component_tolerances).all()
        )
        if not force_passed:
            raise DomainLessonResultsError(
                f"raw parity row {case_id!r} fails componentwise force parity"
            )
        observed_output = _require_mapping(
            raw,
            "output",
            context=f"raw parity row {case_id!r}",
        )
        try:
            observed_energy = float(observed_output["energy_ev"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainLessonResultsError(
                f"raw parity row {case_id!r} energy must be finite"
            ) from exc
        if not np.isfinite(observed_energy):
            raise DomainLessonResultsError(
                f"raw parity row {case_id!r} energy must be finite"
            )
        one_gpu_energy_offset = abs(observed_energy - one_gpu_energy)
        one_gpu_energy_offset_per_atom = one_gpu_energy_offset / atom_count
        distributed_energy_difference = abs(
            observed_energy - distributed_reference_energy
        )
        distributed_energy_difference_per_atom = (
            distributed_energy_difference / atom_count
        )
        force_rms = float(np.sqrt(np.mean(difference * difference)))
        force_max = float(np.abs(difference).max())
        expected_values = {
            "one_gpu_energy_abs_offset_ev": one_gpu_energy_offset,
            "one_gpu_energy_abs_offset_ev_per_atom": (
                one_gpu_energy_offset_per_atom
            ),
            "distributed_energy_difference_ev": distributed_energy_difference,
            "distributed_energy_difference_ev_per_atom": (
                distributed_energy_difference_per_atom
            ),
            "force_rms_difference_ev_per_a": force_rms,
            "force_max_difference_ev_per_a": force_max,
            "energy_tolerance_ev_per_atom": expected_energy_tolerance_per_atom,
            "force_tolerance_ev_per_a": expected_force_tolerance,
        }
        for name, expected in expected_values.items():
            try:
                observed = float(row[name])
            except (TypeError, ValueError) as exc:
                raise DomainLessonResultsError(
                    f"parity row {case_id!r}.{name} must be finite"
                ) from exc
            if not math.isclose(
                observed,
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"parity row {case_id!r}.{name} does not match raw measurement"
                )
        if (
            int(row["gpus"]) in energy_comparison_world_sizes
            and distributed_energy_difference_per_atom
            > expected_energy_tolerance_per_atom
        ):
            raise DomainLessonResultsError(
                f"raw parity row {case_id!r} fails distributed energy agreement"
            )
        expected_flags = {
            "force_passed": force_passed,
            "distributed_energy_passed": True,
            "parity_passed": force_passed,
        }
        for name, expected in expected_flags.items():
            if bool(row[name]) is not expected:
                raise DomainLessonResultsError(
                    f"parity row {case_id!r}.{name} does not match raw measurement"
                )


def _validate_raw_output_agreement(
    root: Path,
    checksums: Mapping[str, str],
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    context: str,
    require_energy: bool = True,
    require_forces: bool = True,
) -> None:
    """Check selected outputs for two runs of the identical saved input."""

    reference_input = _require_mapping(
        reference,
        "input",
        context=f"{context} reference",
    )
    observed_input = _require_mapping(
        observed,
        "input",
        context=f"{context} observed",
    )
    if reference_input.get("file_sha256") != observed_input.get("file_sha256"):
        raise DomainLessonResultsError(f"{context} uses different input structures")

    reference_forces = _load_raw_force_array(
        root,
        checksums,
        reference,
        context=f"{context} reference",
    )
    observed_forces = _load_raw_force_array(
        root,
        checksums,
        observed,
        context=f"{context} observed",
    )
    if observed_forces.shape != reference_forces.shape:
        raise DomainLessonResultsError(f"{context} force shapes differ")

    reference_output = _require_mapping(
        reference,
        "output",
        context=f"{context} reference",
    )
    observed_output = _require_mapping(
        observed,
        "output",
        context=f"{context} observed",
    )
    try:
        reference_energy = float(reference_output["energy_ev"])
        observed_energy = float(observed_output["energy_ev"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainLessonResultsError(
            f"{context} energies must be finite"
        ) from exc
    if not np.isfinite(reference_energy) or not np.isfinite(observed_energy):
        raise DomainLessonResultsError(f"{context} energies must be finite")

    atom_count = int(reference_forces.shape[0])
    if atom_count <= 0:
        raise DomainLessonResultsError(f"{context} has no atoms")
    energy_difference_per_atom = (
        abs(observed_energy - reference_energy) / atom_count
    )
    component_tolerances = (
        DOMAIN_METHODOLOGY.parity_force_atol_ev_a
        + DOMAIN_METHODOLOGY.parity_force_rtol * np.abs(reference_forces)
    )
    if require_energy and (
        energy_difference_per_atom
        > DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
    ):
        raise DomainLessonResultsError(f"{context} fails energy agreement")
    force_agrees = np.less_equal(
            np.abs(observed_forces - reference_forces),
            component_tolerances,
        ).all()
    if require_forces and not force_agrees:
        raise DomainLessonResultsError(f"{context} fails componentwise force agreement")


def _validate_raw_distributed_output_agreement(
    root: Path,
    checksums: Mapping[str, str],
    distributed: pd.DataFrame,
    raw_by_case: Mapping[str, Mapping[str, Any]],
) -> None:
    """Check the actual timing input and comparable successful OOM retries."""

    steady = distributed.loc[
        distributed["measurement_role"].astype(str).eq("steady_timing")
    ].sort_values("gpus")
    force_reference_world_size = DOMAIN_METHODOLOGY.force_reference_world_size
    one_gpu = steady.loc[steady["gpus"].eq(force_reference_world_size)]
    if len(one_gpu) != 1:
        raise DomainLessonResultsError(
            "steady timing output agreement needs one one-GPU reference"
        )
    one_gpu_reference = raw_by_case[str(one_gpu.iloc[0]["case_id"])]
    energy_reference_world_size = DOMAIN_METHODOLOGY.energy_reference_world_size
    energy_reference = steady.loc[
        steady["gpus"].eq(energy_reference_world_size)
    ]
    if len(energy_reference) != 1:
        raise DomainLessonResultsError(
            "steady timing output agreement needs one distributed energy reference"
        )
    distributed_energy_reference = raw_by_case[
        str(energy_reference.iloc[0]["case_id"])
    ]
    for world_size in DOMAIN_METHODOLOGY.force_comparison_world_sizes:
        row = steady.loc[steady["gpus"].eq(world_size)].iloc[0]
        case_id = str(row["case_id"])
        _validate_raw_output_agreement(
            root,
            checksums,
            one_gpu_reference,
            raw_by_case[case_id],
            context=f"steady timing force row {case_id!r}",
            require_energy=False,
            require_forces=True,
        )
    for world_size in DOMAIN_METHODOLOGY.energy_comparison_world_sizes:
        row = steady.loc[steady["gpus"].eq(world_size)].iloc[0]
        case_id = str(row["case_id"])
        _validate_raw_output_agreement(
            root,
            checksums,
            distributed_energy_reference,
            raw_by_case[case_id],
            context=f"steady timing distributed energy row {case_id!r}",
            require_energy=True,
            require_forces=False,
        )

    successful_rescue = distributed.loc[
        distributed["measurement_role"].astype(str).eq("rescue")
        & distributed["success"].eq(True)
    ].sort_values("gpus")
    if len(successful_rescue) < 2:
        return
    rescue_reference = raw_by_case[str(successful_rescue.iloc[0]["case_id"])]
    for _, row in successful_rescue.iloc[1:].iterrows():
        case_id = str(row["case_id"])
        _validate_raw_output_agreement(
            root,
            checksums,
            rescue_reference,
            raw_by_case[case_id],
            context=f"successful rescue row {case_id!r}",
        )


def _capacity_view(table: pd.DataFrame) -> pd.DataFrame:
    """Return the short capacity table shown in the notebook."""

    view = pd.DataFrame(
        {
            "atom_count": table["atom_count"],
            "success": table["success"],
            "world_size": table["gpus"],
            "torch_peak_allocated_gb": pd.to_numeric(
                table["peak_memory_bytes_max_rank"], errors="coerce"
            )
            / 1.0e9,
            "failure_type": table["failure_type"],
            "failure_stage": table["failure_stage"],
        }
    )
    return view.reset_index(drop=True)


def _charge_diagnostics_view(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return the compact one-GPU charge summary shown to learners."""

    columns = (
        "atom_count",
        "dtype",
        "target_sum_e",
        "charge_sum_e",
        "residual_e",
        "abs_residual_per_atom_e",
    )
    rows = []
    for record in records:
        diagnostics = record["charge_diagnostics"]
        rows.append(
            {
                "atom_count": int(record["atom_count"]),
                "dtype": str(diagnostics["dtype"]),
                "target_sum_e": float(diagnostics["target_sum_e"]),
                "charge_sum_e": float(diagnostics["sum_e"]),
                "residual_e": float(diagnostics["residual_e"]),
                "abs_residual_per_atom_e": float(
                    diagnostics["abs_residual_per_atom"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _electrostatics_view(
    record: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Return the fixed-charge PME-versus-Ewald check shown in the notebook."""

    columns = (
        "atom_count",
        "charge_sum_e",
        "pme_energy_eV",
        "ewald_energy_eV",
        "energy_abs_error_meV_per_atom",
        "force_rms_error_eV_A",
        "force_max_error_eV_A",
        "passed",
    )
    if record is None:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "atom_count": int(record["atom_count"]),
                "charge_sum_e": float(record["charge_sum_e"]),
                "pme_energy_eV": float(record["pme_energy_ev"]),
                "ewald_energy_eV": float(record["ewald_energy_ev"]),
                "energy_abs_error_meV_per_atom": (
                    1.0e3 * float(record["energy_abs_difference_ev_per_atom"])
                ),
                "force_rms_error_eV_A": float(record["force_rms_difference_ev_per_a"]),
                "force_max_error_eV_A": float(record["force_max_difference_ev_per_a"]),
                "passed": record["status"] == "passed",
            }
        ],
        columns=columns,
    )


def _parity_view(table: pd.DataFrame) -> pd.DataFrame:
    """Return the split force, energy, and diagnostic comparison table."""

    view = pd.DataFrame(
        {
            "atom_count": table["atom_count"],
            "world_size": table["gpus"],
            "one_gpu_energy_abs_offset_meV_atom": 1.0e3
            * pd.to_numeric(
                table["one_gpu_energy_abs_offset_ev_per_atom"], errors="coerce"
            ),
            "distributed_energy_error_meV_atom": 1.0e3
            * pd.to_numeric(
                table["distributed_energy_difference_ev_per_atom"],
                errors="coerce",
            ),
            "force_max_abs_error_eV_A": pd.to_numeric(
                table["force_max_difference_ev_per_a"], errors="coerce"
            ),
            "force_rms_error_eV_A": pd.to_numeric(
                table["force_rms_difference_ev_per_a"], errors="coerce"
            ),
            "force_passed": table["force_passed"],
            "distributed_energy_passed": table["distributed_energy_passed"],
            "passed": table["parity_passed"],
        }
    )
    return view.reset_index(drop=True)


def _distributed_view(table: pd.DataFrame) -> pd.DataFrame:
    """Return the short domain-decomposition table shown in the notebook."""

    wall_time = pd.to_numeric(table["elapsed_s"], errors="coerce")
    steady_mask = table["measurement_role"].astype(str).eq("steady_timing")
    steady = table.loc[steady_mask & table["success"].eq(True)]
    complete = (
        len(steady) == len(_REQUIRED_STEADY_TIMING_GPUS)
        and set(steady["gpus"].astype(int)) == set(_REQUIRED_STEADY_TIMING_GPUS)
        and steady["atom_count"].nunique() == 1
    )
    speedup = pd.Series(np.nan, index=table.index, dtype=float)
    relative_iqr = pd.to_numeric(
        table["elapsed_iqr_s"], errors="coerce"
    ) / wall_time
    if complete:
        baseline = steady.loc[steady["gpus"].eq(1)]
        if len(baseline) != 1:
            raise DomainLessonResultsError(
                "complete steady timing needs exactly one one-GPU baseline"
            )
        baseline_time = float(wall_time.loc[baseline.index[0]])
        speedup.loc[steady.index] = baseline_time / wall_time.loc[steady.index]

    view = pd.DataFrame(
        {
            "atom_count": table["atom_count"],
            "world_size": table["gpus"],
            "success": table["success"],
            "measurement_role": table["measurement_role"],
            "wall_time_s": wall_time,
            "wall_time_q1_s": pd.to_numeric(
                table["elapsed_q1_s"], errors="coerce"
            ),
            "wall_time_q3_s": pd.to_numeric(
                table["elapsed_q3_s"], errors="coerce"
            ),
            "wall_time_iqr_s": pd.to_numeric(
                table["elapsed_iqr_s"], errors="coerce"
            ),
            "relative_iqr": relative_iqr,
            "speedup_vs_1gpu": speedup,
            "parallel_efficiency": speedup / table["gpus"],
            "torch_peak_allocated_gb": pd.to_numeric(
                table["peak_memory_bytes_max_rank"], errors="coerce"
            )
            / 1.0e9,
            "owned_atoms_min": pd.to_numeric(
                table["owned_atoms_min_rank"], errors="coerce"
            ),
            "owned_atoms_max": pd.to_numeric(
                table["owned_atoms_max_rank"], errors="coerce"
            ),
            "spatial_grid": table["spatial_grid"],
            "failure_type": table["failure_type"],
            "failure_stage": table["failure_stage"],
            "error": table["error"],
        }
    )
    return view.reset_index(drop=True)


def _plot_data(
    capacity: pd.DataFrame,
    distributed: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not capacity.empty:
        capacity_plot = capacity[
            [
                "case_id",
                "atom_count",
                "gpus",
                "success",
                "status",
                "failure_type",
                "elapsed_s",
                "peak_memory_bytes_max_rank",
            ]
        ].copy()
        capacity_plot.insert(0, "series", "single GPU capacity")
        rows.append(capacity_plot)
    if not distributed.empty:
        distributed_plot = distributed[
            [
                "case_id",
                "atom_count",
                "gpus",
                "success",
                "status",
                "failure_type",
                "elapsed_s",
                "peak_memory_bytes_max_rank",
            ]
        ].copy()
        distributed_plot.insert(0, "series", "domain decomposition")
        rows.append(distributed_plot)
    if not rows:
        return pd.DataFrame(
            columns=(
                "series",
                "case_id",
                "atom_count",
                "gpus",
                "success",
                "status",
                "failure_type",
                "elapsed_s",
                "peak_memory_bytes_max_rank",
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_domain_lesson_view(
    path: str | Path,
    *,
    planned_atom_counts: Sequence[int],
    expected_parity_atom_count: int | None = None,
) -> DomainLessonView:
    """Load strict saved results or return an explicit not-reported view."""

    root = Path(path)
    planned_counts = _planned_counts(planned_atom_counts)
    expected_parity_atoms = (
        None
        if expected_parity_atom_count is None
        else _positive_integer(
            expected_parity_atom_count,
            name="expected_parity_atom_count",
        )
    )
    if not root.exists():
        return _not_reported(
            root,
            planned_counts,
            "Compute Lab domain-decomposition results have not been reported.",
        )
    if not root.is_dir():
        raise DomainLessonResultsError("domain lesson result path must be a directory")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        if not any(root.iterdir()):
            return _not_reported(
                root,
                planned_counts,
                "Compute Lab domain-decomposition results have not been reported.",
            )
        raise DomainLessonResultsError("result directory exists without manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainLessonResultsError("manifest.json is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise DomainLessonResultsError("manifest.json must contain an object")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise DomainLessonResultsError("unexpected domain lesson bundle schema")
    status = str(manifest.get("status", ""))
    if status == "not_reported":
        reason = str(manifest.get("reason", "")).strip()
        if not reason:
            raise DomainLessonResultsError("not_reported manifest needs a reason")
        return _not_reported(root, planned_counts, reason, manifest)
    if status != "complete":
        raise DomainLessonResultsError("bundle status must be complete or not_reported")
    if not str(manifest.get("created_utc", "")).strip():
        raise DomainLessonResultsError("complete bundle must report created_utc")

    failure_policy = _require_mapping(manifest, "failure_policy", context="manifest")
    expected_policy = {
        "failed_rows_retained": True,
        "estimates_allowed": False,
        "capacity_stop_condition": "first_single_gpu_oom",
    }
    if dict(failure_policy) != expected_policy:
        raise DomainLessonResultsError(
            "failure policy does not retain measured failures"
        )

    checksums = _read_checksum_index(root)
    if checksums.get(MANIFEST_NAME) != _sha256_file(manifest_path):
        raise DomainLessonResultsError("manifest checksum is missing or incorrect")
    raw_rows = _read_raw_results(root, manifest, checksums)
    artifact_paths = _validate_artifacts(root, manifest, checksums)
    structure_hashes, settings_sha256 = _validate_identity(manifest)
    _validate_electrostatics(
        manifest,
        structure_sha256_by_atom_count=structure_hashes,
        raw_rows=raw_rows,
    )
    tables = {
        name: _read_table(root, manifest, name, checksums) for name in TABLE_NAMES
    }
    _validate_declared_files(
        root,
        manifest,
        checksums,
        artifact_paths,
    )
    for name, table in tables.items():
        _validate_rows(
            name,
            table,
            structure_sha256_by_atom_count=structure_hashes,
            settings_sha256=settings_sha256,
        )
    capacity = tables["capacity"]
    observed_counts = tuple(capacity["atom_count"].astype(int))
    planned_prefix = planned_counts[: len(observed_counts)]
    if observed_counts != planned_prefix:
        raise DomainLessonResultsError(
            "capacity rows must be an in-order prefix of planned_atom_counts"
        )
    oom_rows = capacity.loc[
        ~capacity["success"] & capacity["failure_type"].astype(str).isin(_OOM_TYPES)
    ]
    if oom_rows.empty:
        raise DomainLessonResultsError(
            "complete capacity sweep must retain the first measured single-GPU OOM"
        )
    first_oom_index = int(oom_rows.index[0])
    if first_oom_index != len(capacity) - 1:
        raise DomainLessonResultsError(
            "capacity sweep must stop after the first measured single-GPU OOM"
        )
    if not capacity.loc[: first_oom_index - 1, "success"].all():
        raise DomainLessonResultsError(
            "capacity sweep contains a failure before the reported OOM"
        )
    if tables["parity"].empty or tables["distributed"].empty:
        raise DomainLessonResultsError(
            "complete bundle needs parity and distributed measurements"
        )
    _validate_measurement_completeness(
        tables["parity"],
        tables["distributed"],
    )
    raw_by_case = _reconcile_raw_measurements(tables, raw_rows)
    selection = _validate_selection(
        manifest,
        capacity,
        tables["parity"],
        tables["distributed"],
        raw_by_case,
    )
    _validate_raw_parity(
        root,
        checksums,
        tables["parity"],
        raw_by_case,
    )
    _validate_raw_distributed_output_agreement(
        root,
        checksums,
        tables["distributed"],
        raw_by_case,
    )
    if expected_parity_atoms is not None:
        try:
            parity_pairs = _positive_integer(
                selection["parity_pair_count"],
                name="manifest parity 1:1 composition count",
            )
        except ValueError as exc:
            raise DomainLessonResultsError(str(exc)) from exc
        if parity_pairs * ATOMS_PER_COMPOSITION_UNIT != expected_parity_atoms:
            raise DomainLessonResultsError(
                "manifest parity size does not match expected_parity_atom_count"
            )
        if not tables["parity"]["atom_count"].eq(expected_parity_atoms).all():
            raise DomainLessonResultsError(
                "parity rows do not match expected_parity_atom_count"
            )

    plot_data = _plot_data(capacity, tables["distributed"])
    return DomainLessonView(
        available=True,
        reason="",
        root=root,
        manifest=manifest,
        capacity_table=_capacity_view(capacity),
        charge_diagnostics_table=_charge_diagnostics_view(
            selection["capacity_charge_diagnostics"]
        ),
        electrostatics_table=_electrostatics_view(
            _require_mapping(
                manifest,
                "electrostatics_validation",
                context="manifest",
            )
        ),
        parity_table=_parity_view(tables["parity"]),
        distributed_table=_distributed_view(tables["distributed"]),
        plot_data=plot_data.reset_index(drop=True),
    )


__all__ = (
    "BUNDLE_SCHEMA",
    "CAPACITY_COLUMNS",
    "CHARGE_SUM_TOLERANCE_E",
    "CHECKSUM_INDEX_NAME",
    "DISTRIBUTED_COLUMNS",
    "DomainLessonResultsError",
    "DomainLessonView",
    "MANIFEST_NAME",
    "PARITY_COLUMNS",
    "PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM",
    "PME_EWALD_FORCE_TOLERANCE_EV_PER_A",
    "canonical_json_sha256",
    "load_domain_lesson_view",
)
