"""Read the short H100 domain-decomposition example used in Part 1.

The recorded example evaluates one unchanged 51,200-atom structure on one,
two, and four H100 GPUs.  Each run partitions once, performs one untimed
warmup, records three energy/force passes, and gathers once.  This module
checks the saved files before it prepares the small tables shown to learners.

Missing results are reported explicitly.  Existing results are never accepted
partially: the three GPU counts, the output comparison, the charge diagnostic,
and the small PME-versus-Ewald check must all be present and consistent.
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


BUNDLE_SCHEMA = "alchemi.domain-decomposition-lesson.v5"
RUNNER_SCHEMA = "alchemi.part1-domain-case.v5"
MANIFEST_NAME = "manifest.json"
CHECKSUM_INDEX_NAME = "SHA256SUMS"
RAW_RESULTS_NAME = "raw-results.jsonl"
TABLE_NAMES = ("distributed",)

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
    "warmup_count",
    "measured_pass_count",
    "pass_times_s",
    "median_s",
    "min_s",
    "max_s",
    "peak_memory_bytes_max_rank",
    "owned_atoms_min_rank",
    "owned_atoms_max_rank",
    "spatial_grid",
    "energy_ev",
    "energy_ev_per_atom",
    "comparison_energy_ev",
    "comparison_energy_ev_per_atom",
    "comparison_energy_statistic",
    "energy_dtype",
    "force_rms_ev_per_a",
    "force_max_ev_per_a",
    "structure_sha256",
    "settings_sha256",
    "input_tensor_sha256",
    "positions_pbc_equivalent",
    "max_minimum_image_displacement_a",
    "measurement_role",
    "measurement_kind",
)

PLOT_COLUMNS = (
    "world_size",
    "owned_atoms_min",
    "owned_atoms_max",
    "pass_1_s",
    "pass_2_s",
    "pass_3_s",
    "median_time_s",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_H100 = re.compile(r"(?:^|[^A-Za-z0-9])H100(?:[^A-Za-z0-9]|$)", re.IGNORECASE)

FIXED_PAIR_COUNT = DOMAIN_METHODOLOGY.fixed_molecules_per_species
FIXED_ATOM_COUNT = FIXED_PAIR_COUNT * DOMAIN_METHODOLOGY.atoms_per_composition_unit
ELECTROSTATICS_ATOM_COUNT = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
    * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
ELECTROSTATICS_PAIRS = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
)
REQUIRED_WORLD_SIZES = tuple(DOMAIN_METHODOLOGY.campaign_world_sizes)
PASS_COUNT = DOMAIN_METHODOLOGY.evaluation_pass_count
WARMUP_COUNT = DOMAIN_METHODOLOGY.evaluation_warmup_count


class DomainLessonResultsError(ValueError):
    """Raised when a saved result set is incomplete or inconsistent."""


@dataclass(frozen=True)
class DomainLessonView:
    """Checked, learner-ready domain-decomposition results."""

    available: bool
    reason: str
    root: Path
    manifest: Mapping[str, Any]
    run_settings_table: pd.DataFrame
    layout_table: pd.DataFrame
    timing_table: pd.DataFrame
    output_agreement_table: pd.DataFrame
    charge_diagnostics_table: pd.DataFrame
    electrostatics_table: pd.DataFrame
    distributed_table: pd.DataFrame
    plot_data: pd.DataFrame

    @property
    def successful_case_count(self) -> int:
        """Count successful fixed-input and electrostatics calculations."""

        if not self.available:
            return 0
        fixed = int(self.distributed_table["success"].eq(True).sum())
        electrostatics = int(self.electrostatics_table["passed"].eq(True).sum())
        return fixed + electrostatics

    @property
    def failed_case_count(self) -> int:
        """Count checked failures retained in the learner tables."""

        if not self.available:
            return 0
        fixed = int(self.distributed_table["success"].eq(False).sum())
        electrostatics = int(self.electrostatics_table["passed"].eq(False).sum())
        return fixed + electrostatics

    @property
    def measured_max_atom_count(self) -> int | None:
        """Return the fixed recorded atom count when results are available."""

        return FIXED_ATOM_COUNT if self.available else None

    @property
    def bundle_record(self) -> dict[str, str] | None:
        """Return compact file and source identities for notebook metadata."""

        if not self.available:
            return None
        source = self.manifest["source"]
        return {
            "created_utc": str(self.manifest["created_utc"]),
            "manifest_sha256": _sha256_file(self.root / MANIFEST_NAME),
            "raw_results_sha256": _sha256_file(self.root / RAW_RESULTS_NAME),
            "checksum_index_sha256": _sha256_file(self.root / CHECKSUM_INDEX_NAME),
            "repository_commit": str(source["tutorial_commit"]),
        }

    @property
    def recorded_run_table(self) -> pd.DataFrame:
        """Identify the saved run without showing the full manifest."""

        if not self.available:
            return pd.DataFrame(columns=("Recorded result set",))
        source = self.manifest["source"]
        hardware = self.manifest["hardware"]
        counts = ", ".join(f"{count} / {count}" for count in REQUIRED_WORLD_SIZES)
        return pd.Series(
            {
                "Bundle created (UTC)": self.manifest["created_utc"],
                "Site": hardware["site"],
                "GPU": hardware["gpu_model"],
                "Interconnect": hardware["interconnect"],
                "Measured nodes / GPUs": counts,
                "Toolkit version": source["toolkit_version"],
                "Toolkit Core commit": source["toolkit_core_commit"],
                "Toolkit-Ops commit": source["toolkit_ops_commit"],
                "Tutorial commit": source["tutorial_commit"],
            },
            name="Recorded result set",
        ).to_frame()

    @property
    def takeaway(self) -> dict[str, Any]:
        """Return only conclusions supported by this short recorded example."""

        if not self.available:
            raise DomainLessonResultsError(
                "recorded domain-decomposition results are not available"
            )
        timings = self.timing_table.set_index("world_size")
        agreement = self.output_agreement_table
        return {
            "fixed_atom_count": FIXED_ATOM_COUNT,
            "world_sizes": REQUIRED_WORLD_SIZES,
            "all_fixed_evaluations_succeeded": bool(
                self.distributed_table["success"].eq(True).all()
            ),
            "positions_pbc_equivalent": bool(
                self.distributed_table["positions_pbc_equivalent"].eq(True).all()
            ),
            "max_minimum_image_displacement_a": float(
                self.distributed_table["max_minimum_image_displacement_a"].max()
            ),
            "all_output_checks_passed": bool(agreement["passed"].eq(True).all()),
            "speedup_by_gpu": tuple(
                (int(world_size), float(timings.loc[world_size, "speedup_vs_1gpu"]))
                for world_size in REQUIRED_WORLD_SIZES[1:]
            ),
            "parallel_efficiency_by_gpu": tuple(
                (
                    int(world_size),
                    float(timings.loc[world_size, "parallel_efficiency"]),
                )
                for world_size in REQUIRED_WORLD_SIZES[1:]
            ),
        }

    @property
    def failed_table(self) -> pd.DataFrame:
        """Return any retained failure rows."""

        if not self.available:
            return pd.DataFrame(
                columns=("case_id", "failure_type", "failure_stage", "error")
            )
        return self.distributed_table.loc[
            self.distributed_table["success"].eq(False),
            ("case_id", "failure_type", "failure_stage", "error"),
        ].reset_index(drop=True)


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


def _empty_view(root: Path, reason: str) -> DomainLessonView:
    return DomainLessonView(
        available=False,
        reason=reason,
        root=root,
        manifest={},
        run_settings_table=pd.DataFrame(columns=("setting", "value")),
        layout_table=pd.DataFrame(
            columns=(
                "world_size",
                "nodes",
                "ranks",
                "spatial_grid",
                "owned_atoms_min",
                "owned_atoms_max",
            )
        ),
        timing_table=pd.DataFrame(
            columns=(
                "world_size",
                "pass_1_s",
                "pass_2_s",
                "pass_3_s",
                "median_time_s",
                "speedup_vs_1gpu",
                "parallel_efficiency",
            )
        ),
        output_agreement_table=pd.DataFrame(
            columns=(
                "world_size",
                "one_gpu_energy_offset_meV_atom",
                "one_gpu_energy_offset_is_diagnostic",
                "energy_statistic",
                "energy_dtype",
                "energy_repeatability_span_meV_atom",
                "energy_repeatability_tolerance_meV_atom",
                "energy_repeatability_check_required",
                "energy_repeatability_passed",
                "energy_reference_world_size",
                "energy_difference_meV_atom",
                "energy_check_required",
                "force_reference_world_size",
                "force_rms_error_eV_A",
                "force_max_error_eV_A",
                "energy_passed",
                "force_passed",
                "passed",
            )
        ),
        charge_diagnostics_table=pd.DataFrame(
            columns=(
                "atom_count",
                "dtype",
                "target_sum_e",
                "charge_sum_e",
                "residual_e",
                "abs_residual_per_atom_e",
                "finite",
            )
        ),
        electrostatics_table=pd.DataFrame(),
        distributed_table=pd.DataFrame(columns=DISTRIBUTED_COLUMNS),
        plot_data=pd.DataFrame(columns=PLOT_COLUMNS),
    )


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


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if not _HEX64.fullmatch(digest):
        raise DomainLessonResultsError(f"{name} must be a SHA-256")
    return digest


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise DomainLessonResultsError(f"{name} must be a positive integer")
    try:
        integer = int(value)
        unchanged = float(value) == integer
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(f"{name} must be a positive integer") from exc
    if integer <= 0 or not unchanged:
        raise DomainLessonResultsError(f"{name} must be a positive integer")
    return integer


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DomainLessonResultsError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainLessonResultsError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise DomainLessonResultsError(f"{name} must be finite")
    return number


def _read_checksum_index(root: Path) -> dict[str, str]:
    path = root / CHECKSUM_INDEX_NAME
    if not path.is_file():
        raise DomainLessonResultsError(f"missing {CHECKSUM_INDEX_NAME}")
    checksums: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
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
            raise DomainLessonResultsError(
                "checksum paths must stay inside the result directory"
            )
        name = relative.as_posix()
        if name in checksums:
            raise DomainLessonResultsError(f"duplicate checksum entry for {name}")
        checksums[name] = parts[0]
    return checksums


def _checked_file(
    root: Path,
    relative_name: str,
    expected_sha256: str,
) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DomainLessonResultsError(
            "saved file paths must stay inside the result directory"
        )
    expected = _require_sha256(expected_sha256, name=relative_name)
    path = root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DomainLessonResultsError(
            "saved file paths must stay inside the result directory"
        ) from exc
    if not path.is_file():
        raise DomainLessonResultsError(f"missing saved file {relative_name}")
    observed = _sha256_file(path)
    if observed != expected:
        raise DomainLessonResultsError(
            f"SHA-256 mismatch for {relative_name}: {observed}"
        )
    return path


def _reject_host_paths(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_host_paths(item, context=f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_host_paths(item, context=f"{context}[{index}]")
        return
    if isinstance(value, str):
        windows_path = re.match(r"^[A-Za-z]:[\\/]", value) is not None
        if PurePosixPath(value).is_absolute() or windows_path:
            raise DomainLessonResultsError(
                f"{context} contains a host path; saved paths must be relative"
            )


def _validate_identity(
    manifest: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], str]:
    """Check the saved source, method, fixed input, execution, and hardware."""

    source = _require_mapping(manifest, "source", context="manifest")
    _require_keys(
        source,
        (
            "tutorial_commit",
            "toolkit_core_commit",
            "toolkit_ops_commit",
            "toolkit_version",
            "nci_subset_sha256",
            "aimnet_checkpoint",
            "aimnet_checkpoint_sha256",
            "d3_parameter_sha256",
        ),
        context="manifest.source",
    )
    for name in ("tutorial_commit", "toolkit_core_commit", "toolkit_ops_commit"):
        if not _HEX40.fullmatch(str(source[name])):
            raise DomainLessonResultsError(
                f"manifest.source.{name} must be a 40-digit SHA"
            )
    for name in (
        "nci_subset_sha256",
        "aimnet_checkpoint_sha256",
        "d3_parameter_sha256",
    ):
        _require_sha256(source[name], name=f"manifest.source.{name}")
    if (
        not str(source["toolkit_version"]).strip()
        or not str(source["aimnet_checkpoint"]).strip()
    ):
        raise DomainLessonResultsError(
            "Toolkit version and AIMNet2 checkpoint must be reported"
        )

    expected_config_hash = _sha256_file(Path(__file__).with_name("config.py"))
    expected_values = json.loads(
        json.dumps(
            DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
            allow_nan=False,
        )
    )
    methodology = _require_mapping(manifest, "methodology", context="manifest")
    _require_keys(
        methodology,
        ("schema", "name", "version", "path", "sha256"),
        context="manifest.methodology",
    )
    if (
        methodology["schema"] != DOMAIN_METHODOLOGY.schema
        or methodology["name"] != DOMAIN_METHODOLOGY.name
        or methodology["version"] != DOMAIN_METHODOLOGY.version
        or methodology["sha256"] != expected_config_hash
    ):
        raise DomainLessonResultsError(
            "saved results do not use the current domain methodology"
        )
    if (
        methodology["path"]
        != "part-1-scalable-atomistic-workflows/aux/domain/config.py"
    ):
        raise DomainLessonResultsError("manifest methodology path is incorrect")

    hardware = _require_mapping(manifest, "hardware", context="manifest")
    _require_keys(
        hardware,
        (
            "site",
            "gpu_model",
            "gpu_memory_bytes",
            "gpus_available",
            "nodes_available",
            "driver_version",
            "cuda_version",
            "interconnect",
            "site_source",
            "resource_count_source",
            "interconnect_source",
            "observed_gpus_by_job",
            "observed_nodes_by_job",
        ),
        context="manifest.hardware",
    )
    if not _H100.search(str(hardware["gpu_model"])):
        raise DomainLessonResultsError("recorded results must identify H100 GPUs")
    for name in ("gpu_memory_bytes", "gpus_available", "nodes_available"):
        _positive_integer(
            hardware[name],
            name=f"manifest.hardware.{name}",
        )
    if int(hardware["gpus_available"]) < max(REQUIRED_WORLD_SIZES):
        raise DomainLessonResultsError(
            "hardware identity reports fewer GPUs than the saved runs use"
        )
    if int(hardware["nodes_available"]) < max(REQUIRED_WORLD_SIZES):
        raise DomainLessonResultsError(
            "hardware identity reports fewer nodes than the saved runs use"
        )
    for name in ("site", "driver_version", "cuda_version", "interconnect"):
        if not str(hardware[name]).strip():
            raise DomainLessonResultsError(f"manifest.hardware.{name} must be reported")
    expected_counts = {str(value): value for value in REQUIRED_WORLD_SIZES}
    if (
        hardware["site_source"] != "operator-declared"
        or hardware["resource_count_source"]
        != "derived from successful per-rank runtime records"
        or hardware["interconnect_source"]
        != "operator-declared; raw GPU topology is retained"
        or hardware["observed_gpus_by_job"] != expected_counts
        or hardware["observed_nodes_by_job"] != expected_counts
    ):
        raise DomainLessonResultsError(
            "manifest hardware sources or observed per-job counts are incorrect"
        )

    settings = _require_mapping(manifest, "settings", context="manifest")
    _require_keys(
        settings,
        ("methodology", "model", "input_structure_sha256"),
        context="manifest.settings",
    )
    if settings["methodology"] != expected_values:
        raise DomainLessonResultsError(
            "saved settings do not use the current domain methodology"
        )
    settings_sha256 = _require_sha256(
        manifest.get("settings_sha256"),
        name="manifest.settings_sha256",
    )
    if settings_sha256 != canonical_json_sha256(settings):
        raise DomainLessonResultsError(
            "manifest settings hash does not match its settings"
        )
    structure_sha256 = _require_sha256(
        settings["input_structure_sha256"],
        name="manifest.settings.input_structure_sha256",
    )

    model = _require_mapping(settings, "model", context="manifest.settings")
    _require_keys(
        model,
        (
            "aimnet_checkpoint",
            "aimnet_compile_model",
            "pme_cutoff_a",
            "pme_mesh_safety_factor",
            "pme_spline_order",
            "pme_accuracy",
            "ewald_reference_accuracy",
            "d3_cutoff_a",
            "d3_smoothing_fraction",
            "d3_parameters",
            "neighbor_adaptation",
            "pipeline_groups",
            "position_invariance",
        ),
        context="manifest.settings.model",
    )
    expected_model_values = {
        "aimnet_checkpoint": source["aimnet_checkpoint"],
        "aimnet_compile_model": False,
        "pme_cutoff_a": DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
        "pme_mesh_safety_factor": DOMAIN_METHODOLOGY.pme_mesh_safety_factor,
        "pme_spline_order": DOMAIN_METHODOLOGY.pme_spline_order,
        "pme_accuracy": DOMAIN_METHODOLOGY.pme_accuracy,
        "ewald_reference_accuracy": DOMAIN_METHODOLOGY.ewald_reference_accuracy,
        "d3_cutoff_a": DOMAIN_METHODOLOGY.d3_cutoff_a,
        "d3_smoothing_fraction": DOMAIN_METHODOLOGY.d3_smoothing_fraction,
        "d3_parameters": "read from AIMNet2 checkpoint metadata",
        "neighbor_adaptation": "never",
        "pipeline_groups": [
            {
                "steps": ["AIMNet2Wrapper", "PMEModelWrapper"],
                "use_autograd": True,
            },
            {
                "steps": ["DFTD3ModelWrapper"],
                "use_autograd": False,
            },
        ],
        "position_invariance": {
            "method": "maximum_minimum_image_displacement",
            "tolerance_a": (DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a),
        },
    }
    if dict(model) != expected_model_values:
        raise DomainLessonResultsError(
            "saved model settings differ from the current methodology"
        )

    input_record = _require_mapping(manifest, "input", context="manifest")
    _require_keys(
        input_record,
        ("molecules_per_species", "atom_count", "structure_sha256"),
        context="manifest.input",
    )
    if (
        input_record["molecules_per_species"] != FIXED_PAIR_COUNT
        or input_record["atom_count"] != FIXED_ATOM_COUNT
        or input_record["structure_sha256"] != structure_sha256
    ):
        raise DomainLessonResultsError(
            "manifest input is not the configured fixed 51,200-atom structure"
        )

    execution = _require_mapping(manifest, "execution", context="manifest")
    _require_keys(
        execution,
        (
            "gpu_counts",
            "warmup_count",
            "measured_pass_count",
            "work_per_measured_pass",
            "publishable_benchmark",
            "observed_speedup",
            "parallel_efficiency",
        ),
        context="manifest.execution",
    )
    if (
        execution["gpu_counts"] != list(REQUIRED_WORLD_SIZES)
        or execution["warmup_count"] != WARMUP_COUNT
        or execution["measured_pass_count"] != PASS_COUNT
        or execution["work_per_measured_pass"]
        != "one fixed-structure energy-and-force evaluation"
        or execution["publishable_benchmark"] is not False
    ):
        raise DomainLessonResultsError(
            "manifest execution does not describe the declared few-pass method"
        )
    expected_world_keys = {str(value) for value in REQUIRED_WORLD_SIZES}
    for name in ("observed_speedup", "parallel_efficiency"):
        values = execution[name]
        if not isinstance(values, Mapping) or set(values) != expected_world_keys:
            raise DomainLessonResultsError(
                f"manifest.execution.{name} must cover 1, 2, and 4 GPUs"
            )
        for world, value in values.items():
            if (
                _finite_number(
                    value,
                    name=f"manifest.execution.{name}.{world}",
                )
                <= 0.0
            ):
                raise DomainLessonResultsError(
                    f"manifest.execution.{name}.{world} must be positive"
                )
    return settings_sha256, source, structure_sha256


def _read_table(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> pd.DataFrame:
    files = _require_mapping(manifest, "files", context="manifest")
    metadata = _require_mapping(files, "distributed.csv", context="manifest.files")
    _require_keys(
        metadata,
        ("sha256", "size_bytes"),
        context="manifest.files.distributed.csv",
    )
    filename = "distributed.csv"
    expected_sha256 = _require_sha256(
        metadata["sha256"],
        name="manifest.files.distributed.csv.sha256",
    )
    if checksums.get(filename) != expected_sha256:
        raise DomainLessonResultsError(
            "distributed.csv checksum index and manifest disagree"
        )
    path = _checked_file(root, filename, expected_sha256)
    expected_size = _positive_integer(
        metadata["size_bytes"],
        name="manifest.files.distributed.csv.size_bytes",
    )
    if path.stat().st_size != expected_size:
        raise DomainLessonResultsError(
            "distributed.csv size does not match the manifest"
        )
    table = pd.read_csv(path, keep_default_na=False)
    if tuple(table.columns) != DISTRIBUTED_COLUMNS:
        raise DomainLessonResultsError(
            "distributed CSV columns do not match the v5 schema"
        )
    case_ids = table["case_id"].astype(str).tolist()
    if len(table) != len(REQUIRED_WORLD_SIZES) or len(set(case_ids)) != len(case_ids):
        raise DomainLessonResultsError(
            "distributed.csv must contain one unique 1/2/4-GPU row"
        )
    return table


def _coerce_bool(series: pd.Series, *, name: str) -> pd.Series:
    values: list[bool] = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif str(value).strip().lower() == "true":
            values.append(True)
        elif str(value).strip().lower() == "false":
            values.append(False)
        else:
            raise DomainLessonResultsError(f"{name} must be boolean")
    return pd.Series(values, index=series.index, dtype=bool)


def _parse_pass_times(value: Any, *, context: str) -> tuple[float, float, float]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise DomainLessonResultsError(f"{context} must contain a JSON list") from exc
    if not isinstance(raw, list) or len(raw) != PASS_COUNT:
        raise DomainLessonResultsError(
            f"{context} must contain exactly {PASS_COUNT} pass times"
        )
    values = tuple(_finite_number(item, name=context) for item in raw)
    if any(item <= 0.0 for item in values):
        raise DomainLessonResultsError(f"{context} values must be positive")
    return values  # type: ignore[return-value]


def _validate_table(
    table: pd.DataFrame,
    *,
    settings_sha256: str,
    structure_sha256: str,
) -> dict[int, tuple[float, float, float]]:
    table["success"] = _coerce_bool(table["success"], name="distributed.success")
    table["positions_pbc_equivalent"] = _coerce_bool(
        table["positions_pbc_equivalent"],
        name="distributed.positions_pbc_equivalent",
    )
    if not table["success"].eq(True).all():
        raise DomainLessonResultsError(
            "all three fixed-input GPU evaluations must succeed"
        )
    if not table["status"].astype(str).eq("complete").all():
        raise DomainLessonResultsError(
            "successful fixed-input rows must have status complete"
        )
    for name in ("failure_type", "failure_stage", "error"):
        if table[name].astype(str).str.strip().ne("").any():
            raise DomainLessonResultsError(
                f"successful fixed-input rows must leave {name} empty"
            )
    if not table["measurement_role"].astype(str).eq("fixed_evaluation").all():
        raise DomainLessonResultsError(
            "distributed rows must use the fixed_evaluation role"
        )
    if (
        not table["measurement_kind"]
        .astype(str)
        .eq("fixed_structure_energy_force_pass")
        .all()
    ):
        raise DomainLessonResultsError(
            "distributed rows have the wrong measurement kind"
        )
    if not table["settings_sha256"].astype(str).eq(settings_sha256).all():
        raise DomainLessonResultsError(
            "distributed settings do not match the manifest identity"
        )
    if not table["structure_sha256"].astype(str).eq(structure_sha256).all():
        raise DomainLessonResultsError(
            "distributed rows do not use the fixed input structure"
        )
    for value in table["input_tensor_sha256"]:
        _require_sha256(value, name="distributed.input_tensor_sha256")
    if table["input_tensor_sha256"].astype(str).nunique() != 1:
        raise DomainLessonResultsError(
            "the 1/2/4-GPU rows do not use the same input tensor"
        )
    if not table["positions_pbc_equivalent"].eq(True).all():
        raise DomainLessonResultsError(
            "a fixed-input evaluation is not PBC-equivalent to its input"
        )
    displacement = np.array(
        [
            _finite_number(
                value,
                name="distributed.max_minimum_image_displacement_a",
            )
            for value in table["max_minimum_image_displacement_a"]
        ]
    )
    if np.any(displacement < 0.0) or np.any(
        displacement > DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
    ):
        raise DomainLessonResultsError(
            "a fixed-input evaluation exceeds the minimum-image position tolerance"
        )
    table["max_minimum_image_displacement_a"] = displacement

    integer_columns = (
        "atom_count",
        "molecules_per_species",
        "nodes",
        "gpus",
        "ranks",
        "warmup_count",
        "measured_pass_count",
        "peak_memory_bytes_max_rank",
        "owned_atoms_min_rank",
        "owned_atoms_max_rank",
    )
    for name in integer_columns:
        values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.rint(values)).all()
            or np.any(values <= 0)
        ):
            raise DomainLessonResultsError(
                f"distributed.{name} must contain positive integers"
            )
        table[name] = values.astype(np.int64)
    if not table["atom_count"].eq(FIXED_ATOM_COUNT).all():
        raise DomainLessonResultsError(
            f"every fixed-input row must contain {FIXED_ATOM_COUNT} atoms"
        )
    if not table["molecules_per_species"].eq(FIXED_PAIR_COUNT).all():
        raise DomainLessonResultsError(
            f"every fixed-input row must contain {FIXED_PAIR_COUNT} molecule pairs"
        )
    if not (table["nodes"].eq(table["gpus"]) & table["gpus"].eq(table["ranks"])).all():
        raise DomainLessonResultsError(
            "fixed-input rows must satisfy nodes == gpus == ranks"
        )
    observed_worlds = tuple(sorted(table["gpus"].astype(int).tolist()))
    if observed_worlds != REQUIRED_WORLD_SIZES:
        raise DomainLessonResultsError(
            "fixed-input results require exactly the declared 1/2/4-GPU rows"
        )
    if not table["warmup_count"].eq(WARMUP_COUNT).all():
        raise DomainLessonResultsError("fixed-input rows have the wrong warmup count")
    if not table["measured_pass_count"].eq(PASS_COUNT).all():
        raise DomainLessonResultsError(
            "fixed-input rows have the wrong measured pass count"
        )
    if not (
        table["comparison_energy_statistic"]
        .astype(str)
        .eq("median_of_three_measured_passes")
        .all()
    ):
        raise DomainLessonResultsError(
            "distributed rows have the wrong comparison energy statistic"
        )
    energy_dtypes = table["energy_dtype"].astype(str)
    expected_energy_dtypes = (
        table["gpus"]
        .astype(int)
        .map(DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size)
    )
    if not energy_dtypes.eq(expected_energy_dtypes).all():
        raise DomainLessonResultsError("distributed rows have the wrong energy dtype")

    pass_times: dict[int, tuple[float, float, float]] = {}
    for index, row in table.iterrows():
        world_size = int(row["gpus"])
        values = _parse_pass_times(
            row["pass_times_s"],
            context=f"distributed pass times for {world_size} GPUs",
        )
        pass_times[world_size] = values
        expected = {
            "median_s": float(np.median(values)),
            "min_s": min(values),
            "max_s": max(values),
        }
        for name, value in expected.items():
            observed = _finite_number(
                row[name],
                name=f"distributed.{name}",
            )
            if not math.isclose(
                observed,
                value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"distributed timing statistics do not match pass times at row {index}"
                )
    for name in (
        "energy_ev",
        "energy_ev_per_atom",
        "comparison_energy_ev",
        "comparison_energy_ev_per_atom",
        "force_rms_ev_per_a",
        "force_max_ev_per_a",
    ):
        values = np.array(
            [_finite_number(value, name=f"distributed.{name}") for value in table[name]]
        )
        if name.startswith("force_") and np.any(values < 0.0):
            raise DomainLessonResultsError(f"distributed.{name} must be non-negative")
    expected_per_atom = table["energy_ev"].astype(float) / FIXED_ATOM_COUNT
    if not np.allclose(
        table["energy_ev_per_atom"].astype(float),
        expected_per_atom,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            "distributed energy per atom does not match total energy"
        )
    expected_comparison_per_atom = (
        table["comparison_energy_ev"].astype(float) / FIXED_ATOM_COUNT
    )
    if not np.allclose(
        table["comparison_energy_ev_per_atom"].astype(float),
        expected_comparison_per_atom,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            "distributed comparison energy per atom does not match total energy"
        )
    for row in table.itertuples(index=False):
        grid = _parse_grid(
            row.spatial_grid,
            world_size=int(row.gpus),
            context="distributed.spatial_grid",
        )
        if int(row.owned_atoms_min_rank) > int(row.owned_atoms_max_rank):
            raise DomainLessonResultsError(
                "owned_atoms_min_rank must not exceed owned_atoms_max_rank"
            )
        if not (
            int(row.owned_atoms_min_rank)
            <= FIXED_ATOM_COUNT / int(row.gpus)
            <= int(row.owned_atoms_max_rank)
        ):
            raise DomainLessonResultsError(
                "owned-atom range does not bracket the mean atoms per rank"
            )
        if math.prod(grid) != int(row.gpus):
            raise DomainLessonResultsError(
                "spatial_grid product must equal the GPU count"
            )
    return pass_times


def _parse_grid(
    value: Any,
    *,
    world_size: int,
    context: str,
) -> tuple[int, int, int]:
    parts = str(value).split("x")
    if len(parts) != 3:
        raise DomainLessonResultsError(f"{context} must contain three dimensions")
    try:
        grid = tuple(int(item) for item in parts)
    except ValueError as exc:
        raise DomainLessonResultsError(
            f"{context} must contain positive integers"
        ) from exc
    if any(item <= 0 for item in grid) or math.prod(grid) != world_size:
        raise DomainLessonResultsError(
            f"{context} must be a positive rank grid for {world_size} GPUs"
        )
    return grid  # type: ignore[return-value]


def _read_raw_results(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    files = _require_mapping(manifest, "files", context="manifest")
    metadata = _require_mapping(files, RAW_RESULTS_NAME, context="manifest.files")
    _require_keys(
        metadata,
        ("sha256", "size_bytes"),
        context=f"manifest.files.{RAW_RESULTS_NAME}",
    )
    expected_sha256 = _require_sha256(
        metadata["sha256"],
        name=f"manifest.files.{RAW_RESULTS_NAME}.sha256",
    )
    if checksums.get(RAW_RESULTS_NAME) != expected_sha256:
        raise DomainLessonResultsError(
            "raw-results.jsonl checksum index and manifest disagree"
        )
    path = _checked_file(root, RAW_RESULTS_NAME, expected_sha256)
    expected_size = _positive_integer(
        metadata["size_bytes"],
        name=f"manifest.files.{RAW_RESULTS_NAME}.size_bytes",
    )
    if path.stat().st_size != expected_size:
        raise DomainLessonResultsError(
            "raw-results.jsonl size does not match the manifest"
        )
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DomainLessonResultsError(
                f"invalid raw result JSON on line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise DomainLessonResultsError(
                f"raw result line {line_number} must be an object"
            )
        _reject_host_paths(row, context=f"raw result line {line_number}")
        rows.append(row)
    if len(rows) != len(REQUIRED_WORLD_SIZES) + 1:
        raise DomainLessonResultsError(
            "raw-results.jsonl must contain three fixed rows and one electrostatics row"
        )
    return rows


def _validate_raw_source(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    context: str,
) -> None:
    source = _require_mapping(row, "source", context=context)
    repository_commit = str(source.get("repository_commit", ""))
    if not _HEX40.fullmatch(repository_commit):
        raise DomainLessonResultsError(
            f"{context}.source.repository_commit must be a 40-digit SHA"
        )
    if source.get("repository_dirty") is not False:
        raise DomainLessonResultsError(
            f"{context}.source.repository_dirty must be false"
        )
    observed = {
        "tutorial_commit": repository_commit,
        "toolkit_core_commit": str(source.get("toolkit_core_commit", "")),
        "toolkit_ops_commit": str(source.get("toolkit_ops_commit", "")),
        "toolkit_version": str(source.get("toolkit_version", "")),
    }
    expected_values = {
        name: str(expected[name])
        for name in (
            "tutorial_commit",
            "toolkit_core_commit",
            "toolkit_ops_commit",
            "toolkit_version",
        )
    }
    if observed != expected_values:
        raise DomainLessonResultsError(
            f"{context} source commit or Toolkit version does not match the manifest"
        )


def _validate_raw_methodology(
    row: Mapping[str, Any],
    *,
    pair_count: int,
    context: str,
) -> None:
    methodology = _require_mapping(row, "methodology", context=context)
    _require_keys(
        methodology,
        ("source", "source_file", "resolved_values", "case_molecules_per_species"),
        context=f"{context}.methodology",
    )
    expected_record = json.loads(
        json.dumps(DOMAIN_METHODOLOGY.as_record(), allow_nan=False)
    )
    expected_values = json.loads(
        json.dumps(
            DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
            allow_nan=False,
        )
    )
    if (
        methodology["source"] != expected_record
        or methodology["resolved_values"] != expected_values
        or methodology["case_molecules_per_species"] != pair_count
    ):
        raise DomainLessonResultsError(
            f"{context} methodology differs from the current fixed method"
        )
    source_file = _require_mapping(
        methodology,
        "source_file",
        context=f"{context}.methodology",
    )
    if source_file.get("sha256") != _sha256_file(Path(__file__).with_name("config.py")):
        raise DomainLessonResultsError(
            f"{context} methodology file hash does not match"
        )


def _check_force_summary(
    summary: Mapping[str, Any],
    *,
    atom_count: int,
    context: str,
) -> None:
    _require_keys(
        summary,
        (
            "shape",
            "dtype",
            "rms_ev_a",
            "max_norm_ev_a",
            "finite",
        ),
        context=context,
    )
    if list(summary["shape"]) != [atom_count, 3]:
        raise DomainLessonResultsError(f"{context}.shape is incorrect")
    if summary["dtype"] != "float32":
        raise DomainLessonResultsError(f"{context}.dtype must be float32")
    if summary["finite"] is not True:
        raise DomainLessonResultsError(f"{context} reports non-finite forces")
    for name in ("rms_ev_a", "max_norm_ev_a"):
        value = _finite_number(summary[name], name=f"{context}.{name}")
        if value < 0.0:
            raise DomainLessonResultsError(f"{context}.{name} must be non-negative")
    for optional in (
        "sum_abs_ev_a",
        "sum_squares_ev2_a2",
    ):
        if optional in summary:
            value = _finite_number(
                summary[optional],
                name=f"{context}.{optional}",
            )
            if value < 0.0:
                raise DomainLessonResultsError(
                    f"{context}.{optional} must be non-negative"
                )
    if "sum_vector_ev_a" in summary:
        values = summary["sum_vector_ev_a"]
        if not isinstance(values, list) or len(values) != 3:
            raise DomainLessonResultsError(
                f"{context}.sum_vector_ev_a must have three values"
            )
        for value in values:
            _finite_number(value, name=f"{context}.sum_vector_ev_a")


def _validate_position_invariance(
    output: Mapping[str, Any],
    *,
    context: str,
) -> tuple[float, tuple[float, float, float]]:
    record = _require_mapping(
        output,
        "position_invariance",
        context=context,
    )
    _require_keys(
        record,
        (
            "method",
            "tolerance_a",
            "warmup_maximum_minimum_image_displacement_a",
            "measured_pass_maximum_minimum_image_displacements_a",
            "final_gather_maximum_minimum_image_displacement_a",
            "maximum_minimum_image_displacement_a",
            "all_within_tolerance",
            "interpretation",
        ),
        context=f"{context}.position_invariance",
    )
    tolerance = _finite_number(
        record["tolerance_a"],
        name=f"{context}.position_invariance.tolerance_a",
    )
    if (
        record["method"] != "maximum_minimum_image_displacement"
        or not math.isclose(
            tolerance,
            DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or record["all_within_tolerance"] is not True
        or not str(record["interpretation"]).strip()
    ):
        raise DomainLessonResultsError(
            f"{context} uses the wrong PBC-equivalent position check"
        )
    warmup = _finite_number(
        record["warmup_maximum_minimum_image_displacement_a"],
        name=(
            f"{context}.position_invariance.warmup_maximum_minimum_image_displacement_a"
        ),
    )
    raw_passes = record["measured_pass_maximum_minimum_image_displacements_a"]
    if not isinstance(raw_passes, list) or len(raw_passes) != PASS_COUNT:
        raise DomainLessonResultsError(
            f"{context} must report one minimum-image displacement per pass"
        )
    pass_values = tuple(
        _finite_number(
            value,
            name=(
                f"{context}.position_invariance."
                "measured_pass_maximum_minimum_image_displacements_a"
            ),
        )
        for value in raw_passes
    )
    final = _finite_number(
        record["final_gather_maximum_minimum_image_displacement_a"],
        name=(
            f"{context}.position_invariance."
            "final_gather_maximum_minimum_image_displacement_a"
        ),
    )
    overall = _finite_number(
        record["maximum_minimum_image_displacement_a"],
        name=(f"{context}.position_invariance.maximum_minimum_image_displacement_a"),
    )
    values = (warmup, *pass_values, final)
    if any(value < 0.0 or value > tolerance for value in values):
        raise DomainLessonResultsError(
            f"{context} exceeds the minimum-image position tolerance"
        )
    if not math.isclose(
        overall,
        max(values),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            f"{context} maximum minimum-image displacement is inconsistent"
        )
    return overall, pass_values  # type: ignore[return-value]


def _load_force_array(
    root: Path,
    output: Mapping[str, Any],
    checksums: Mapping[str, str],
    *,
    atom_count: int,
    context: str,
) -> np.ndarray:
    metadata = _require_mapping(
        output,
        "forces_source_atom_order_npy",
        context=context,
    )
    _require_keys(
        metadata,
        ("path", "sha256", "dtype", "shape"),
        context=f"{context}.forces_source_atom_order_npy",
    )
    filename = str(metadata["path"])
    expected_sha256 = _require_sha256(
        metadata["sha256"],
        name=f"{context}.forces_source_atom_order_npy.sha256",
    )
    if checksums.get(filename) != expected_sha256:
        raise DomainLessonResultsError(
            f"{context} force-array checksum index and raw result disagree"
        )
    path = _checked_file(root, filename, expected_sha256)
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DomainLessonResultsError(
            f"{context} force array could not be read"
        ) from exc
    if array.shape != (atom_count, 3):
        raise DomainLessonResultsError(f"{context} force array has the wrong shape")
    if list(metadata["shape"]) != [atom_count, 3]:
        raise DomainLessonResultsError(
            f"{context} force-array metadata has the wrong shape"
        )
    if str(array.dtype) != str(metadata["dtype"]):
        raise DomainLessonResultsError(
            f"{context} force-array dtype does not match metadata"
        )
    if str(array.dtype) != "float32":
        raise DomainLessonResultsError(
            f"{context} force array must use the recorded float32 precision"
        )
    if not np.isfinite(array).all():
        raise DomainLessonResultsError(f"{context} force array is not finite")
    return array


def _validate_charge_record(
    record: Mapping[str, Any],
    *,
    atom_count: int,
    context: str,
) -> dict[str, Any]:
    _require_keys(
        record,
        (
            "available",
            "dtype",
            "target_sum_e",
            "shape",
            "sha256",
            "finite",
            "sum_e",
            "residual_e",
            "abs_residual_per_atom",
        ),
        context=context,
    )
    if record["available"] is not True or record["finite"] is not True:
        raise DomainLessonResultsError(
            f"{context} must report finite predicted charges"
        )
    if str(record["dtype"]) != "float32":
        raise DomainLessonResultsError(f"{context} must report float32 charges")
    shape = list(record["shape"])
    if (
        math.prod(_positive_integer(value, name=f"{context}.shape") for value in shape)
        != atom_count
    ):
        raise DomainLessonResultsError(f"{context}.shape does not match the atom count")
    _require_sha256(record["sha256"], name=f"{context}.sha256")
    target = _finite_number(record["target_sum_e"], name=f"{context}.target_sum_e")
    charge_sum = _finite_number(record["sum_e"], name=f"{context}.sum_e")
    residual = _finite_number(record["residual_e"], name=f"{context}.residual_e")
    per_atom = _finite_number(
        record["abs_residual_per_atom"],
        name=f"{context}.abs_residual_per_atom",
    )
    if not math.isclose(
        residual,
        charge_sum - target,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            f"{context}.residual_e does not equal sum minus target"
        )
    if not math.isclose(
        per_atom,
        abs(residual) / atom_count,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            f"{context}.abs_residual_per_atom is inconsistent"
        )
    return {
        "atom_count": atom_count,
        "dtype": "float32",
        "target_sum_e": target,
        "charge_sum_e": charge_sum,
        "residual_e": residual,
        "abs_residual_per_atom_e": per_atom,
        "finite": True,
    }


def _validate_unavailable_charges(
    record: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if record.get("available") is not False:
        raise DomainLessonResultsError(
            f"{context} must state that multi-GPU charges are unavailable"
        )
    if not str(record.get("reason", "")).strip():
        raise DomainLessonResultsError(
            f"{context} must explain why charges are unavailable"
        )
    for name in ("dtype", "shape", "sha256", "finite", "sum_e", "residual_e"):
        if record.get(name) is not None:
            raise DomainLessonResultsError(
                f"{context}.{name} must be null when charges are unavailable"
            )


def _validate_raw_distributed(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: pd.DataFrame,
    source_identity: Mapping[str, Any],
    settings_sha256: str,
    structure_sha256: str,
    checksums: Mapping[str, str],
    root: Path,
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, np.ndarray],
    dict[str, Any],
]:
    if len(rows) != len(REQUIRED_WORLD_SIZES):
        raise DomainLessonResultsError(
            "raw results require exactly one fixed-input row per GPU count"
        )
    by_world: dict[int, Mapping[str, Any]] = {}
    force_arrays: dict[int, np.ndarray] = {}
    input_hashes: set[str] = set()
    one_gpu_charge: dict[str, Any] | None = None
    table_by_case = table.set_index("case_id")

    for row in rows:
        context = f"raw fixed-input result {row.get('case_id', '<missing>')}"
        _require_keys(
            row,
            (
                "schema",
                "run_id",
                "case_id",
                "mode",
                "measurement_role",
                "status",
                "success",
                "world_size",
                "pair_count",
                "atom_count",
                "source",
                "methodology",
                "runtime",
                "input",
                "distributed",
                "output",
                "charges",
                "timing",
                "memory",
                "settings_sha256",
                "bundle_source",
                "bundle_job_record",
            ),
            context=context,
        )
        if (
            row["schema"] != RUNNER_SCHEMA
            or row["mode"] != "distributed"
            or row["measurement_role"] != "fixed_evaluation"
            or row["status"] != "complete"
            or row["success"] is not True
        ):
            raise DomainLessonResultsError(
                f"{context} is not a successful v5 fixed evaluation"
            )
        if (
            row["settings_sha256"] != settings_sha256
            or row["bundle_source"] != "manifest.json#source"
            or row["bundle_job_record"]
            != f"manifest.json#job_records/{row['world_size']}"
        ):
            raise DomainLessonResultsError(
                f"{context} does not link to the bundle settings and job record"
            )
        _validate_raw_source(
            row,
            expected=source_identity,
            context=context,
        )
        _validate_raw_methodology(
            row,
            pair_count=FIXED_PAIR_COUNT,
            context=context,
        )
        world_size = _positive_integer(
            row["world_size"],
            name=f"{context}.world_size",
        )
        if world_size not in REQUIRED_WORLD_SIZES or world_size in by_world:
            raise DomainLessonResultsError(
                f"{context} has an unexpected or duplicate GPU count"
            )
        if (
            _positive_integer(row["pair_count"], name=f"{context}.pair_count")
            != FIXED_PAIR_COUNT
            or _positive_integer(row["atom_count"], name=f"{context}.atom_count")
            != FIXED_ATOM_COUNT
        ):
            raise DomainLessonResultsError(
                f"{context} does not use the fixed {FIXED_ATOM_COUNT}-atom input"
            )
        case_id = str(row["case_id"])
        if case_id not in table_by_case.index:
            raise DomainLessonResultsError(
                f"{context} has no matching distributed CSV row"
            )
        csv_row = table_by_case.loc[case_id]
        if int(csv_row["gpus"]) != world_size:
            raise DomainLessonResultsError(
                f"{context} GPU count differs from distributed.csv"
            )
        if not str(row["run_id"]).strip():
            raise DomainLessonResultsError(f"{context}.run_id must be nonempty")
        input_record = _require_mapping(row, "input", context=context)
        _require_keys(
            input_record,
            ("path", "file_sha256", "file_size_bytes", "tensor_sha256"),
            context=f"{context}.input",
        )
        input_digest = _require_sha256(
            input_record["file_sha256"],
            name=f"{context}.input.file_sha256",
        )
        if input_digest != structure_sha256:
            raise DomainLessonResultsError(
                f"{context} input structure hash does not match the manifest"
            )
        input_path = str(input_record["path"])
        if checksums.get(input_path) != input_digest:
            raise DomainLessonResultsError(
                f"{context} input checksum index and raw result disagree"
            )
        checked_input = _checked_file(root, input_path, input_digest)
        if checked_input.stat().st_size != _positive_integer(
            input_record["file_size_bytes"],
            name=f"{context}.input.file_size_bytes",
        ):
            raise DomainLessonResultsError(f"{context} input file size does not match")
        tensor_hash = _require_sha256(
            input_record["tensor_sha256"],
            name=f"{context}.input.tensor_sha256",
        )
        input_hashes.add(tensor_hash)
        if tensor_hash != str(csv_row["input_tensor_sha256"]):
            raise DomainLessonResultsError(
                f"{context} input tensor hash differs from distributed.csv"
            )

        distributed = _require_mapping(row, "distributed", context=context)
        _require_keys(
            distributed,
            (
                "api",
                "cells_per_dim",
                "rank_grid",
                "owned_atom_counts",
                "owned_atom_count_min",
                "owned_atom_count_max",
                "partition_count",
                "gather_count",
            ),
            context=f"{context}.distributed",
        )
        if (
            distributed["api"] != "DomainParallel"
            or distributed["partition_count"] != 1
            or distributed["gather_count"] != 1
        ):
            raise DomainLessonResultsError(
                f"{context} must partition and gather exactly once through DomainParallel"
            )
        cells = _layout_triplet(
            distributed["cells_per_dim"],
            context=f"{context}.distributed.cells_per_dim",
        )
        ranks = _layout_triplet(
            distributed["rank_grid"],
            context=f"{context}.distributed.rank_grid",
        )
        if math.prod(ranks) != world_size:
            raise DomainLessonResultsError(
                f"{context} rank grid does not match its GPU count"
            )
        if any(cell % rank != 0 for cell, rank in zip(cells, ranks, strict=True)):
            raise DomainLessonResultsError(
                f"{context} rank grid does not divide the cell grid"
            )
        owned = distributed["owned_atom_counts"]
        if (
            not isinstance(owned, list)
            or len(owned) != world_size
            or any(
                _positive_integer(value, name=f"{context}.owned_atom_counts") <= 0
                for value in owned
            )
            or sum(int(value) for value in owned) != FIXED_ATOM_COUNT
        ):
            raise DomainLessonResultsError(
                f"{context} owned atom counts do not cover the fixed input"
            )
        if min(int(value) for value in owned) != _positive_integer(
            distributed["owned_atom_count_min"],
            name=f"{context}.distributed.owned_atom_count_min",
        ) or max(int(value) for value in owned) != _positive_integer(
            distributed["owned_atom_count_max"],
            name=f"{context}.distributed.owned_atom_count_max",
        ):
            raise DomainLessonResultsError(
                f"{context} owned-atom range is inconsistent"
            )
        if (
            int(csv_row["owned_atoms_min_rank"]) != min(owned)
            or int(csv_row["owned_atoms_max_rank"]) != max(owned)
            or str(csv_row["spatial_grid"]) != "x".join(str(value) for value in ranks)
        ):
            raise DomainLessonResultsError(
                f"{context} layout differs from distributed.csv"
            )

        timing = _require_mapping(row, "timing", context=context)
        _require_keys(
            timing,
            (
                "pass_times_s",
                "median_s",
                "min_s",
                "max_s",
                "warmup_count",
                "measured_pass_count",
                "requested_steps_per_pass",
                "measured_model_evaluations_per_pass",
                "warmup_requested_steps",
                "warmup_automatic_force_prime_evaluations",
                "warmup_model_evaluations",
                "publishable_benchmark",
                "source_input_sha256",
                "partition_count",
                "gather_count",
            ),
            context=f"{context}.timing",
        )
        raw_times = timing["pass_times_s"]
        if not isinstance(raw_times, list):
            raise DomainLessonResultsError(
                f"{context}.timing.pass_times_s must be a list"
            )
        times = _parse_pass_times(
            json.dumps(raw_times, separators=(",", ":")),
            context=f"{context}.timing.pass_times_s",
        )
        csv_times = _parse_pass_times(
            csv_row["pass_times_s"],
            context=f"{context} CSV pass times",
        )
        if not np.array_equal(np.asarray(times), np.asarray(csv_times)):
            raise DomainLessonResultsError(
                f"{context} pass times differ from distributed.csv"
            )
        expected_timing = {
            "median_s": float(np.median(times)),
            "min_s": min(times),
            "max_s": max(times),
        }
        for name, value in expected_timing.items():
            if not math.isclose(
                _finite_number(timing[name], name=f"{context}.timing.{name}"),
                value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"{context} timing summary does not match the three passes"
                )
        warmup_prime = (
            DOMAIN_METHODOLOGY.domain_parallel_multi_rank_warmup_force_prime_evaluations
            if world_size > 1
            else 0
        )
        if (
            timing["warmup_count"] != WARMUP_COUNT
            or timing["measured_pass_count"] != PASS_COUNT
            or timing["requested_steps_per_pass"] != 1
            or timing["measured_model_evaluations_per_pass"]
            != DOMAIN_METHODOLOGY.measured_model_evaluations_per_pass
            or timing["warmup_requested_steps"] != 1
            or timing["warmup_automatic_force_prime_evaluations"] != warmup_prime
            or timing["warmup_model_evaluations"] != 1 + warmup_prime
            or timing["publishable_benchmark"] is not False
            or timing["partition_count"] != 1
            or timing["gather_count"] != 1
        ):
            raise DomainLessonResultsError(
                f"{context} does not record the declared few-pass method"
            )
        timing_source_hash = _require_sha256(
            timing["source_input_sha256"],
            name=f"{context}.timing.source_input_sha256",
        )
        if timing_source_hash != tensor_hash:
            raise DomainLessonResultsError(
                f"{context} timing does not identify the original input tensor"
            )

        output = _require_mapping(row, "output", context=context)
        _require_keys(
            output,
            (
                "energy_ev",
                "energy_ev_per_atom",
                "energy_dtype",
                "forces_source_atom_order",
                "measured_passes",
                "position_invariance",
            ),
            context=f"{context}.output",
        )
        energy = _finite_number(
            output["energy_ev"],
            name=f"{context}.output.energy_ev",
        )
        energy_per_atom = _finite_number(
            output["energy_ev_per_atom"],
            name=f"{context}.output.energy_ev_per_atom",
        )
        energy_dtype = str(output["energy_dtype"])
        expected_energy_dtype = (
            DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(world_size)
        )
        if (
            energy_dtype != expected_energy_dtype
            or str(csv_row["energy_dtype"]) != energy_dtype
        ):
            raise DomainLessonResultsError(
                f"{context} energy dtype differs from the declared method"
            )
        if not math.isclose(
            energy_per_atom,
            energy / FIXED_ATOM_COUNT,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(f"{context} energy per atom is inconsistent")
        if not math.isclose(
            energy,
            float(csv_row["energy_ev"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            energy_per_atom,
            float(csv_row["energy_ev_per_atom"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(
                f"{context} energy differs from distributed.csv"
            )
        overall_displacement, pass_displacements = _validate_position_invariance(
            output,
            context=f"{context}.output",
        )
        if not math.isclose(
            overall_displacement,
            float(csv_row["max_minimum_image_displacement_a"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not bool(csv_row["positions_pbc_equivalent"]):
            raise DomainLessonResultsError(
                f"{context} position check differs from distributed.csv"
            )
        force_summary = _require_mapping(
            output,
            "forces_source_atom_order",
            context=f"{context}.output",
        )
        _check_force_summary(
            force_summary,
            atom_count=FIXED_ATOM_COUNT,
            context=f"{context}.output.forces_source_atom_order",
        )
        if not math.isclose(
            float(csv_row["force_rms_ev_per_a"]),
            float(force_summary["rms_ev_a"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            float(csv_row["force_max_ev_per_a"]),
            float(force_summary["max_norm_ev_a"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(
                f"{context} force summary differs from distributed.csv"
            )
        passes = output["measured_passes"]
        if not isinstance(passes, list) or len(passes) != PASS_COUNT:
            raise DomainLessonResultsError(
                f"{context} must report all three measured outputs"
            )
        pass_energy_values: list[float] = []
        for pass_index, pass_record in enumerate(passes, start=1):
            if not isinstance(pass_record, Mapping):
                raise DomainLessonResultsError(
                    f"{context} measured pass {pass_index} must be an object"
                )
            _require_keys(
                pass_record,
                (
                    "pass_index",
                    "energy_ev",
                    "energy_ev_per_atom",
                    "energy_dtype",
                    "forces",
                    "maximum_minimum_image_displacement_a",
                ),
                context=f"{context}.output.measured_passes[{pass_index - 1}]",
            )
            if pass_record["pass_index"] != pass_index:
                raise DomainLessonResultsError(
                    f"{context} measured pass indices are not ordered"
                )
            pass_displacement = _finite_number(
                pass_record["maximum_minimum_image_displacement_a"],
                name=(
                    f"{context} pass {pass_index} maximum minimum-image displacement"
                ),
            )
            if not math.isclose(
                pass_displacement,
                pass_displacements[pass_index - 1],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"{context} pass {pass_index} position check is inconsistent"
                )
            pass_energy = _finite_number(
                pass_record["energy_ev"],
                name=f"{context} pass {pass_index} energy",
            )
            if str(pass_record["energy_dtype"]) != energy_dtype:
                raise DomainLessonResultsError(
                    f"{context} pass {pass_index} energy dtype is inconsistent"
                )
            pass_energy_values.append(pass_energy)
            pass_energy_per_atom = _finite_number(
                pass_record["energy_ev_per_atom"],
                name=f"{context} pass {pass_index} energy per atom",
            )
            if not math.isclose(
                pass_energy_per_atom,
                pass_energy / FIXED_ATOM_COUNT,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"{context} pass {pass_index} energy per atom is inconsistent"
                )
            pass_forces = pass_record["forces"]
            if not isinstance(pass_forces, Mapping):
                raise DomainLessonResultsError(
                    f"{context} pass {pass_index} forces must be an object"
                )
            _check_force_summary(
                pass_forces,
                atom_count=FIXED_ATOM_COUNT,
                context=f"{context} pass {pass_index} forces",
            )
        energy_span_per_atom = (
            max(pass_energy_values) - min(pass_energy_values)
        ) / FIXED_ATOM_COUNT
        if world_size > 1 and energy_span_per_atom > (
            DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
        ):
            raise DomainLessonResultsError(
                f"{context} distributed measured energies are not repeatable"
            )
        comparison_energy = _finite_number(
            csv_row["comparison_energy_ev"],
            name=f"{context} comparison energy",
        )
        if not math.isclose(
            comparison_energy,
            float(np.median(pass_energy_values)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(
                f"{context} comparison energy is not the measured-pass median"
            )
        final_pass = passes[-1]
        if not math.isclose(
            energy,
            float(final_pass["energy_ev"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(
                f"{context} final energy does not match measured pass 3"
            )
        final_force_summary = final_pass["forces"]
        if not math.isclose(
            float(force_summary["rms_ev_a"]),
            float(final_force_summary["rms_ev_a"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            float(force_summary["max_norm_ev_a"]),
            float(final_force_summary["max_norm_ev_a"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DomainLessonResultsError(
                f"{context} final force summary does not match measured pass 3"
            )
        force_arrays[world_size] = _load_force_array(
            root,
            output,
            checksums,
            atom_count=FIXED_ATOM_COUNT,
            context=context,
        )

        memory = _require_mapping(row, "memory", context=context)
        _require_keys(
            memory,
            (
                "measured_pass_max_allocated_bytes_per_rank",
                "measured_pass_max_reserved_bytes_per_rank",
                "max_allocated_bytes",
                "max_reserved_bytes",
            ),
            context=f"{context}.memory",
        )
        allocated = _validate_pass_memory(
            memory["measured_pass_max_allocated_bytes_per_rank"],
            world_size=world_size,
            context=(f"{context}.memory.measured_pass_max_allocated_bytes_per_rank"),
        )
        reserved = _validate_pass_memory(
            memory["measured_pass_max_reserved_bytes_per_rank"],
            world_size=world_size,
            context=(f"{context}.memory.measured_pass_max_reserved_bytes_per_rank"),
        )
        if any(
            reserved_value < allocated_value
            for allocated_row, reserved_row in zip(
                allocated,
                reserved,
                strict=True,
            )
            for allocated_value, reserved_value in zip(
                allocated_row,
                reserved_row,
                strict=True,
            )
        ):
            raise DomainLessonResultsError(
                f"{context} reserved memory is smaller than allocated memory"
            )
        if (
            int(memory["max_allocated_bytes"])
            != max(value for values in allocated for value in values)
            or int(memory["max_reserved_bytes"])
            != max(value for values in reserved for value in values)
            or int(csv_row["peak_memory_bytes_max_rank"])
            != int(memory["max_allocated_bytes"])
        ):
            raise DomainLessonResultsError(
                f"{context} memory summary differs from the per-pass values"
            )

        charges = _require_mapping(row, "charges", context=context)
        if world_size == DOMAIN_METHODOLOGY.force_reference_world_size:
            one_gpu_charge = _validate_charge_record(
                charges,
                atom_count=FIXED_ATOM_COUNT,
                context=f"{context}.charges",
            )
        else:
            _validate_unavailable_charges(
                charges,
                context=f"{context}.charges",
            )
        by_world[world_size] = row

    if tuple(sorted(by_world)) != REQUIRED_WORLD_SIZES:
        raise DomainLessonResultsError(
            "raw fixed-input rows do not cover every declared GPU count"
        )
    if len(input_hashes) != 1:
        raise DomainLessonResultsError(
            "the fixed-input rows do not share one input tensor"
        )
    if one_gpu_charge is None:
        raise DomainLessonResultsError(
            "the one-GPU predicted-charge diagnostic is missing"
        )
    return by_world, force_arrays, one_gpu_charge


def _layout_triplet(value: Any, *, context: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise DomainLessonResultsError(f"{context} must have three integers")
    result = tuple(_positive_integer(item, name=context) for item in value)
    return result  # type: ignore[return-value]


def _validate_pass_memory(
    value: Any,
    *,
    world_size: int,
    context: str,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != PASS_COUNT:
        raise DomainLessonResultsError(
            f"{context} must contain one row per measured pass"
        )
    rows: list[tuple[int, ...]] = []
    for pass_index, raw_row in enumerate(value, start=1):
        if not isinstance(raw_row, list) or len(raw_row) != world_size:
            raise DomainLessonResultsError(
                f"{context} pass {pass_index} must contain one value per rank"
            )
        row: list[int] = []
        for raw in raw_row:
            amount = _positive_integer(
                raw,
                name=f"{context} pass {pass_index}",
            )
            row.append(amount)
        rows.append(tuple(row))
    return tuple(rows)


def _measured_energy_values(row: Mapping[str, Any]) -> tuple[float, ...]:
    output = _require_mapping(row, "output", context="fixed result")
    passes = output.get("measured_passes")
    if not isinstance(passes, list) or len(passes) != PASS_COUNT:
        raise DomainLessonResultsError(
            "fixed result has the wrong measured-energy series"
        )
    return tuple(
        _finite_number(
            measured["energy_ev"],
            name=f"fixed result measured energy {index}",
        )
        for index, measured in enumerate(passes, start=1)
    )


def _measured_energy_median(row: Mapping[str, Any]) -> float:
    return float(np.median(_measured_energy_values(row)))


def _measured_energy_span_per_atom(row: Mapping[str, Any]) -> float:
    values = _measured_energy_values(row)
    return (max(values) - min(values)) / FIXED_ATOM_COUNT


def _output_agreement(
    rows_by_world: Mapping[int, Mapping[str, Any]],
    force_arrays: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    force_reference_world = DOMAIN_METHODOLOGY.force_reference_world_size
    energy_reference_world = DOMAIN_METHODOLOGY.energy_reference_world_size
    force_reference = force_arrays[force_reference_world].astype(np.float64)
    energy_reference = _measured_energy_median(rows_by_world[energy_reference_world])
    one_gpu_energy = _measured_energy_median(rows_by_world[force_reference_world])
    records: list[dict[str, Any]] = []
    for world_size in REQUIRED_WORLD_SIZES:
        forces = force_arrays[world_size].astype(np.float64)
        difference = forces - force_reference
        component_limit = (
            DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a
            + DOMAIN_METHODOLOGY.evaluation_force_rtol * np.abs(force_reference)
        )
        force_passed = bool(np.less_equal(np.abs(difference), component_limit).all())
        force_rms = float(np.sqrt(np.mean(difference * difference)))
        force_max = float(np.linalg.norm(difference, axis=1).max())

        energy = _measured_energy_median(rows_by_world[world_size])
        energy_span_per_atom = _measured_energy_span_per_atom(rows_by_world[world_size])
        repeatability_required = world_size > 1
        repeatability_passed = bool(
            energy_span_per_atom
            <= (
                DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
            )
        )
        energy_difference_per_atom = abs(energy - energy_reference) / FIXED_ATOM_COUNT
        energy_check_required = world_size in (
            DOMAIN_METHODOLOGY.energy_comparison_world_sizes
        )
        energy_passed = (
            bool(
                energy_difference_per_atom
                <= DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
            )
            if energy_check_required
            else None
        )
        required_checks_passed = (
            force_passed
            and (not repeatability_required or repeatability_passed)
            and (
                not energy_check_required
                or energy_difference_per_atom
                <= DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
            )
        )
        records.append(
            {
                "world_size": world_size,
                "one_gpu_energy_offset_meV_atom": (
                    1000.0 * (energy - one_gpu_energy) / FIXED_ATOM_COUNT
                ),
                "one_gpu_energy_offset_is_diagnostic": True,
                "energy_statistic": "median_of_three_measured_passes",
                "energy_dtype": str(
                    rows_by_world[world_size]["output"]["energy_dtype"]
                ),
                "energy_repeatability_span_meV_atom": (1000.0 * energy_span_per_atom),
                "energy_repeatability_tolerance_meV_atom": (
                    1000.0
                    * DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
                ),
                "energy_repeatability_check_required": repeatability_required,
                "energy_repeatability_passed": (
                    repeatability_passed if repeatability_required else None
                ),
                "energy_reference_world_size": energy_reference_world,
                "energy_difference_meV_atom": (1000.0 * energy_difference_per_atom),
                "energy_check_required": energy_check_required,
                "force_reference_world_size": force_reference_world,
                "force_rms_error_eV_A": force_rms,
                "force_max_error_eV_A": force_max,
                "energy_passed": energy_passed,
                "force_passed": force_passed,
                "passed": required_checks_passed,
            }
        )
    result = pd.DataFrame.from_records(records)
    if not result["passed"].eq(True).all():
        raise DomainLessonResultsError(
            "saved 1/2/4-GPU energies or forces exceed declared agreement limits"
        )
    return result


def _validate_manifest_output_agreement(
    manifest: Mapping[str, Any],
    *,
    rows_by_world: Mapping[int, Mapping[str, Any]],
    force_arrays: Mapping[int, np.ndarray],
) -> None:
    record = _require_mapping(manifest, "output_agreement", context="manifest")
    _require_keys(
        record,
        (
            "force_reference_gpus",
            "distributed_energy_reference_gpus",
            "one_gpu_energy_offsets_are_diagnostics_only",
            "position_check",
            "comparisons",
            "all_required_checks_passed",
        ),
        context="manifest.output_agreement",
    )
    force_reference_world = DOMAIN_METHODOLOGY.force_reference_world_size
    energy_reference_world = DOMAIN_METHODOLOGY.energy_reference_world_size
    if (
        record["force_reference_gpus"] != force_reference_world
        or record["distributed_energy_reference_gpus"] != energy_reference_world
        or record["one_gpu_energy_offsets_are_diagnostics_only"] is not True
        or record.get("energy_statistic") != "median_of_three_measured_passes"
        or record["all_required_checks_passed"] is not True
    ):
        raise DomainLessonResultsError(
            "manifest output-agreement references or status are incorrect"
        )
    position_check = _require_mapping(
        record,
        "position_check",
        context="manifest.output_agreement",
    )
    _require_keys(
        position_check,
        ("method", "tolerance_a", "meaning"),
        context="manifest.output_agreement.position_check",
    )
    if (
        position_check["method"] != "maximum_minimum_image_displacement"
        or not math.isclose(
            _finite_number(
                position_check["tolerance_a"],
                name="manifest.output_agreement.position_check.tolerance_a",
            ),
            DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not str(position_check["meaning"]).strip()
    ):
        raise DomainLessonResultsError(
            "manifest position check differs from the current methodology"
        )
    comparisons = _require_mapping(
        record,
        "comparisons",
        context="manifest.output_agreement",
    )
    if set(comparisons) != {str(value) for value in REQUIRED_WORLD_SIZES}:
        raise DomainLessonResultsError(
            "manifest output agreement must contain the 1/2/4-GPU rows"
        )

    reference_forces = force_arrays[force_reference_world].astype(np.float64)
    one_gpu_energy = _measured_energy_median(rows_by_world[force_reference_world])
    energy_reference = _measured_energy_median(rows_by_world[energy_reference_world])
    for world_size in REQUIRED_WORLD_SIZES:
        context = f"manifest.output_agreement.comparisons.{world_size}"
        comparison = _require_mapping(
            comparisons,
            str(world_size),
            context="manifest.output_agreement.comparisons",
        )
        _require_keys(
            comparison,
            (
                "one_gpu_energy_offset_ev",
                "one_gpu_energy_abs_offset_ev_per_atom",
                "one_gpu_energy_offset_is_diagnostic_only",
                "energy_statistic",
                "energy_dtype",
                "energy_repeatability_span_ev_per_atom",
                "energy_repeatability_tolerance_ev_per_atom",
                "energy_repeatability_check_required",
                "energy_repeatability_passed",
                "distributed_energy_reference_gpus",
                "distributed_energy_difference_ev",
                "distributed_energy_abs_difference_ev_per_atom",
                "distributed_energy_check_required",
                "distributed_energy_passed",
                "force_rms_difference_ev_per_a_vs_1gpu",
                "force_max_difference_ev_per_a_vs_1gpu",
                "force_max_component_difference_ev_per_a_vs_1gpu",
                "distributed_energy_agreement_tolerance_ev_per_atom",
                "force_atol_ev_per_a",
                "force_rtol",
                "force_passed",
                "position_check",
                "position_tolerance_a",
                "maximum_minimum_image_displacement_a",
                "positions_pbc_equivalent",
                "required_checks_passed",
            ),
            context=context,
        )
        energy = _measured_energy_median(rows_by_world[world_size])
        energy_span_per_atom = _measured_energy_span_per_atom(rows_by_world[world_size])
        one_gpu_offset = energy - one_gpu_energy
        distributed_difference = energy - energy_reference
        force_difference = (
            force_arrays[world_size].astype(np.float64) - reference_forces
        )
        expected_numbers = {
            "one_gpu_energy_offset_ev": one_gpu_offset,
            "one_gpu_energy_abs_offset_ev_per_atom": (
                abs(one_gpu_offset) / FIXED_ATOM_COUNT
            ),
            "energy_repeatability_span_ev_per_atom": energy_span_per_atom,
            "energy_repeatability_tolerance_ev_per_atom": (
                DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
            ),
            "distributed_energy_difference_ev": distributed_difference,
            "distributed_energy_abs_difference_ev_per_atom": (
                abs(distributed_difference) / FIXED_ATOM_COUNT
            ),
            "force_rms_difference_ev_per_a_vs_1gpu": float(
                np.sqrt(np.mean(force_difference * force_difference))
            ),
            "force_max_difference_ev_per_a_vs_1gpu": float(
                np.linalg.norm(force_difference, axis=1).max()
            ),
            "force_max_component_difference_ev_per_a_vs_1gpu": float(
                np.abs(force_difference).max()
            ),
            "distributed_energy_agreement_tolerance_ev_per_atom": (
                DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
            ),
            "force_atol_ev_per_a": (DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a),
            "force_rtol": DOMAIN_METHODOLOGY.evaluation_force_rtol,
            "position_tolerance_a": (
                DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
            ),
            "maximum_minimum_image_displacement_a": float(
                rows_by_world[world_size]["output"]["position_invariance"][
                    "maximum_minimum_image_displacement_a"
                ]
            ),
        }
        for name, expected in expected_numbers.items():
            observed = _finite_number(comparison[name], name=f"{context}.{name}")
            if not math.isclose(
                observed,
                expected,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise DomainLessonResultsError(
                    f"{context}.{name} does not match the saved outputs"
                )
        required_energy_check = world_size in (
            DOMAIN_METHODOLOGY.energy_comparison_world_sizes
        )
        repeatability_required = world_size > 1
        repeatability_passed = bool(
            energy_span_per_atom
            <= (
                DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
            )
        )
        expected_energy_pass = (
            abs(distributed_difference) / FIXED_ATOM_COUNT
            <= DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
        )
        force_limit = (
            DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a
            + DOMAIN_METHODOLOGY.evaluation_force_rtol * np.abs(reference_forces)
        )
        expected_force_pass = bool(
            np.less_equal(np.abs(force_difference), force_limit).all()
        )
        expected_required_pass = (
            expected_force_pass
            and (not repeatability_required or repeatability_passed)
            and (not required_energy_check or expected_energy_pass)
        )
        if (
            comparison["one_gpu_energy_offset_is_diagnostic_only"] is not True
            or comparison["energy_statistic"] != "median_of_three_measured_passes"
            or comparison["energy_dtype"]
            != DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(world_size)
            or comparison["energy_repeatability_check_required"]
            is not repeatability_required
            or comparison["energy_repeatability_passed"]
            != (repeatability_passed if repeatability_required else None)
            or comparison["distributed_energy_reference_gpus"] != energy_reference_world
            or comparison["distributed_energy_check_required"]
            is not required_energy_check
            or comparison["distributed_energy_passed"]
            != (expected_energy_pass if required_energy_check else None)
            or comparison["force_passed"] is not expected_force_pass
            or comparison["position_check"] != "maximum_minimum_image_displacement"
            or comparison["positions_pbc_equivalent"] is not True
            or comparison["required_checks_passed"] is not expected_required_pass
        ):
            raise DomainLessonResultsError(
                f"{context} uses the wrong energy or force check"
            )


def _validate_electrostatics(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    settings_sha256: str,
    checksums: Mapping[str, str],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = "raw electrostatics result"
    _require_keys(
        row,
        (
            "schema",
            "run_id",
            "case_id",
            "mode",
            "measurement_role",
            "status",
            "success",
            "world_size",
            "pair_count",
            "atom_count",
            "source",
            "input",
            "charges",
            "pme",
            "ewald",
            "comparison",
            "bundle_settings_sha256",
            "bundle_source",
            "bundle_job_record",
        ),
        context=context,
    )
    if (
        row["schema"] != RUNNER_SCHEMA
        or row["mode"] != "electrostatics-validation"
        or row["measurement_role"] != "electrostatics_validation"
        or row["status"] != "complete"
        or row["success"] is not True
        or row["world_size"] != 1
        or row["atom_count"] != ELECTROSTATICS_ATOM_COUNT
    ):
        raise DomainLessonResultsError(
            "electrostatics result is not the declared successful 3,200-atom check"
        )
    if (
        row["bundle_settings_sha256"] != settings_sha256
        or row["bundle_source"] != "manifest.json#source"
        or row["bundle_job_record"] != "manifest.json#job_records/1"
        or "settings_sha256" in row
    ):
        raise DomainLessonResultsError(
            "electrostatics result must link to, not claim, the fixed-input settings"
        )
    _validate_raw_source(
        row,
        expected=source_identity,
        context=context,
    )
    _validate_raw_methodology(
        row,
        pair_count=ELECTROSTATICS_PAIRS,
        context=context,
    )
    input_record = _require_mapping(row, "input", context=context)
    _require_keys(
        input_record,
        ("path", "file_sha256", "file_size_bytes"),
        context=f"{context}.input",
    )
    input_path = str(input_record["path"])
    input_digest = _require_sha256(
        input_record["file_sha256"],
        name=f"{context}.input.file_sha256",
    )
    if checksums.get(input_path) != input_digest:
        raise DomainLessonResultsError(
            "electrostatics input checksum index and raw result disagree"
        )
    checked_input = _checked_file(root, input_path, input_digest)
    if checked_input.stat().st_size != _positive_integer(
        input_record["file_size_bytes"],
        name=f"{context}.input.file_size_bytes",
    ):
        raise DomainLessonResultsError("electrostatics input file size does not match")

    validation_metadata = _require_mapping(
        manifest,
        "electrostatics_validation",
        context="manifest",
    )
    _require_keys(
        validation_metadata,
        ("file", "sha256", "passed"),
        context="manifest.electrostatics_validation",
    )
    if (
        validation_metadata["file"] != "electrostatics-validation.json"
        or validation_metadata["passed"] is not True
    ):
        raise DomainLessonResultsError(
            "manifest electrostatics validation is incomplete"
        )
    validation_digest = _require_sha256(
        validation_metadata["sha256"],
        name="manifest.electrostatics_validation.sha256",
    )
    files = _require_mapping(manifest, "files", context="manifest")
    validation_file_metadata = _require_mapping(
        files,
        "electrostatics-validation.json",
        context="manifest.files",
    )
    _require_keys(
        validation_file_metadata,
        ("sha256", "size_bytes"),
        context="manifest.files.electrostatics-validation.json",
    )
    if (
        validation_file_metadata["sha256"] != validation_digest
        or checksums.get("electrostatics-validation.json") != validation_digest
    ):
        raise DomainLessonResultsError(
            "electrostatics-validation.json checksums disagree"
        )
    validation_path = _checked_file(
        root,
        "electrostatics-validation.json",
        validation_digest,
    )
    if validation_path.stat().st_size != _positive_integer(
        validation_file_metadata["size_bytes"],
        name="manifest.files.electrostatics-validation.json.size_bytes",
    ):
        raise DomainLessonResultsError(
            "electrostatics-validation.json size does not match the manifest"
        )
    try:
        saved_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DomainLessonResultsError(
            "electrostatics-validation.json is not valid JSON"
        ) from exc
    if not isinstance(saved_validation, Mapping) or saved_validation != row:
        raise DomainLessonResultsError(
            "electrostatics-validation.json differs from raw-results.jsonl"
        )
    _reject_host_paths(
        saved_validation,
        context="electrostatics-validation.json",
    )
    charge = _validate_charge_record(
        _require_mapping(row, "charges", context=context),
        atom_count=ELECTROSTATICS_ATOM_COUNT,
        context="raw electrostatics result.charges",
    )
    if abs(float(charge["residual_e"])) > DOMAIN_METHODOLOGY.charge_sum_tolerance_e:
        raise DomainLessonResultsError(
            "electrostatics charge residual exceeds the declared limit"
        )
    pme = _require_mapping(row, "pme", context=context)
    ewald = _require_mapping(row, "ewald", context=context)
    comparison = _require_mapping(row, "comparison", context=context)
    _require_keys(pme, ("energy_ev", "forces"), context="electrostatics.pme")
    _require_keys(ewald, ("energy_ev", "forces"), context="electrostatics.ewald")
    _require_keys(
        comparison,
        (
            "absolute_energy_difference_ev",
            "absolute_energy_difference_ev_per_atom",
            "force_difference_rms_ev_a",
            "force_difference_max_norm_ev_a",
            "acceptance",
            "passed",
        ),
        context="electrostatics.comparison",
    )
    pme_energy = _finite_number(pme["energy_ev"], name="electrostatics PME energy")
    ewald_energy = _finite_number(
        ewald["energy_ev"],
        name="electrostatics Ewald energy",
    )
    _check_force_summary(
        _require_mapping(pme, "forces", context="electrostatics.pme"),
        atom_count=ELECTROSTATICS_ATOM_COUNT,
        context="electrostatics.pme.forces",
    )
    _check_force_summary(
        _require_mapping(ewald, "forces", context="electrostatics.ewald"),
        atom_count=ELECTROSTATICS_ATOM_COUNT,
        context="electrostatics.ewald.forces",
    )
    energy_error = abs(pme_energy - ewald_energy)
    energy_error_per_atom = energy_error / ELECTROSTATICS_ATOM_COUNT
    if not math.isclose(
        _finite_number(
            comparison["absolute_energy_difference_ev"],
            name="electrostatics absolute energy difference",
        ),
        energy_error,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        _finite_number(
            comparison["absolute_energy_difference_ev_per_atom"],
            name="electrostatics energy difference per atom",
        ),
        energy_error_per_atom,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise DomainLessonResultsError(
            "electrostatics energy comparison is inconsistent"
        )
    force_rms = _finite_number(
        comparison["force_difference_rms_ev_a"],
        name="electrostatics force RMS difference",
    )
    force_max = _finite_number(
        comparison["force_difference_max_norm_ev_a"],
        name="electrostatics force maximum difference",
    )
    acceptance = _require_mapping(
        comparison,
        "acceptance",
        context="electrostatics.comparison",
    )
    expected_acceptance = {
        "declared_before_measurement": True,
        "absolute_energy_difference_ev_per_atom_max": (
            DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
        ),
        "force_difference_max_norm_ev_a_max": (
            DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
        ),
        "absolute_charge_sum_e_max": DOMAIN_METHODOLOGY.charge_sum_tolerance_e,
    }
    if dict(acceptance) != expected_acceptance:
        raise DomainLessonResultsError(
            "electrostatics limits differ from the current methodology"
        )
    passed = (
        energy_error_per_atom
        <= DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
        and force_max <= DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
        and abs(float(charge["residual_e"]))
        <= DOMAIN_METHODOLOGY.charge_sum_tolerance_e
    )
    if comparison["passed"] is not True or not passed:
        raise DomainLessonResultsError("the saved PME-versus-Ewald check does not pass")
    table = pd.DataFrame(
        [
            {
                "atom_count": ELECTROSTATICS_ATOM_COUNT,
                "charge_sum_e": charge["charge_sum_e"],
                "charge_residual_e": charge["residual_e"],
                "energy_abs_error_meV_per_atom": 1000.0 * energy_error_per_atom,
                "force_rms_error_eV_A": force_rms,
                "force_max_error_eV_A": force_max,
                "passed": True,
            }
        ]
    )
    return table, charge


def _settings_table(manifest: Mapping[str, Any]) -> pd.DataFrame:
    settings = manifest["settings"]
    model = settings["model"]
    execution = manifest["execution"]
    domain_cutoff = max(
        DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a,
        float(model["pme_cutoff_a"]),
        float(model["d3_cutoff_a"]),
    )
    return pd.DataFrame(
        [
            ("Fixed input", f"{FIXED_ATOM_COUNT:,} atoms"),
            ("Molecular composition", "phenol + N-methylacetamide"),
            ("Model", "AIMNet2 + PME electrostatics + D3(BJ)"),
            ("Model tensors, coordinates, forces", "float32"),
            (
                "Total energy",
                "float32 on 1 GPU; float64 distributed reduction on 2/4 GPUs",
            ),
            ("Toolkit API", "DomainParallel"),
            ("Domain cutoff", f"{domain_cutoff:g} Å"),
            ("Domain skin", f"{DOMAIN_METHODOLOGY.domain_halo_skin_a:g} Å"),
            ("GPU counts", ", ".join(map(str, REQUIRED_WORLD_SIZES))),
            ("Untimed warmup passes", execution["warmup_count"]),
            ("Measured passes", execution["measured_pass_count"]),
            (
                "Model evaluations per measured pass",
                DOMAIN_METHODOLOGY.measured_model_evaluations_per_pass,
            ),
            ("Work per measured pass", execution["work_per_measured_pass"]),
        ],
        columns=("setting", "value"),
    )


def _learner_tables(
    table: pd.DataFrame,
    pass_times: Mapping[int, tuple[float, float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = table.sort_values("gpus").reset_index(drop=True)
    layout = pd.DataFrame(
        {
            "world_size": ordered["gpus"].astype(int),
            "nodes": ordered["nodes"].astype(int),
            "ranks": ordered["ranks"].astype(int),
            "spatial_grid": ordered["spatial_grid"].astype(str),
            "owned_atoms_min": ordered["owned_atoms_min_rank"].astype(int),
            "owned_atoms_max": ordered["owned_atoms_max_rank"].astype(int),
        }
    )
    timing_records: list[dict[str, Any]] = []
    reference_median = float(np.median(pass_times[REQUIRED_WORLD_SIZES[0]]))
    for world_size in REQUIRED_WORLD_SIZES:
        values = pass_times[world_size]
        median = float(np.median(values))
        speedup = reference_median / median
        timing_records.append(
            {
                "world_size": world_size,
                "pass_1_s": values[0],
                "pass_2_s": values[1],
                "pass_3_s": values[2],
                "median_time_s": median,
                "speedup_vs_1gpu": speedup,
                "parallel_efficiency": speedup / world_size,
            }
        )
    timing = pd.DataFrame.from_records(timing_records)
    plot = timing[
        [
            "world_size",
            "pass_1_s",
            "pass_2_s",
            "pass_3_s",
            "median_time_s",
        ]
    ].merge(
        layout[["world_size", "owned_atoms_min", "owned_atoms_max"]],
        on="world_size",
        validate="one_to_one",
    )
    plot = plot[list(PLOT_COLUMNS)]
    return layout, timing, plot


def _validate_declared_files(
    root: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> None:
    declared = {MANIFEST_NAME}
    files = _require_mapping(manifest, "files", context="manifest")
    required_top_files = {
        "distributed.csv",
        RAW_RESULTS_NAME,
        "electrostatics-validation.json",
    }
    if set(files) != required_top_files:
        raise DomainLessonResultsError(
            "manifest.files must contain the three learner result files"
        )

    def validate_file_record(
        filename: str,
        record: Mapping[str, Any],
        *,
        context: str,
    ) -> None:
        _require_keys(record, ("sha256", "size_bytes"), context=context)
        expected = _require_sha256(record["sha256"], name=f"{context}.sha256")
        if checksums.get(filename) != expected:
            raise DomainLessonResultsError(
                f"{filename} checksum index and manifest disagree"
            )
        path = _checked_file(root, filename, expected)
        expected_size = _positive_integer(
            record["size_bytes"],
            name=f"{context}.size_bytes",
        )
        if path.stat().st_size != expected_size:
            raise DomainLessonResultsError(
                f"{filename} size does not match the manifest"
            )
        if filename in declared:
            raise DomainLessonResultsError(f"saved file is declared twice: {filename}")
        declared.add(filename)

    for filename, record in files.items():
        if not isinstance(record, Mapping):
            raise DomainLessonResultsError(
                f"manifest.files.{filename} must be an object"
            )
        validate_file_record(
            str(filename),
            record,
            context=f"manifest.files.{filename}",
        )

    job_records = _require_mapping(manifest, "job_records", context="manifest")
    expected_job_keys = {str(value) for value in REQUIRED_WORLD_SIZES}
    if set(job_records) != expected_job_keys:
        raise DomainLessonResultsError(
            "manifest.job_records must contain the 1/2/4-GPU jobs"
        )
    required_job_files = {
        "plan.json",
        "phase-summary.json",
        "collection-summary.json",
        "results.jsonl",
        "part1-runtime.json",
        "d3-cache.json",
        "aimnet-checkpoint-preflight.json",
        "gpu-names.txt",
        "gpu-topology.txt",
        "network-interfaces.txt",
        "producer-SHA256SUMS",
        "artifact-SHA256SUMS",
    }
    producer_maps: list[dict[str, str]] = []
    for world_size in REQUIRED_WORLD_SIZES:
        context = f"manifest.job_records.{world_size}"
        record = _require_mapping(
            job_records,
            str(world_size),
            context="manifest.job_records",
        )
        _require_keys(
            record,
            (
                "world_size",
                "files",
                "producer_checksum_file_sha256",
                "artifact_checksum_file_sha256",
                "verified_producer_file_count",
                "verified_artifact_file_count",
                "producer_files",
            ),
            context=context,
        )
        if record["world_size"] != world_size:
            raise DomainLessonResultsError(f"{context}.world_size is incorrect")
        job_files = _require_mapping(record, "files", context=context)
        if not required_job_files <= set(job_files):
            raise DomainLessonResultsError(
                f"{context}.files is missing required job records"
            )
        if not all(
            any(str(name).startswith(f"{directory}/") for name in job_files)
            for directory in ("inputs", "results", "ranks", "logs")
        ):
            raise DomainLessonResultsError(
                f"{context}.files must retain inputs, results, ranks, and logs"
            )
        by_relative_name: dict[str, str] = {}
        for job_relative, raw_file_record in job_files.items():
            if not isinstance(raw_file_record, Mapping):
                raise DomainLessonResultsError(
                    f"{context}.files.{job_relative} must be an object"
                )
            _require_keys(
                raw_file_record,
                ("path", "sha256", "size_bytes"),
                context=f"{context}.files.{job_relative}",
            )
            bundle_path = str(raw_file_record["path"])
            expected_prefix = f"job-records/gpus-{world_size:02d}/"
            if not bundle_path.startswith(expected_prefix) or bundle_path.removeprefix(
                expected_prefix
            ) != str(job_relative):
                raise DomainLessonResultsError(
                    f"{context}.files.{job_relative}.path is incorrect"
                )
            validate_file_record(
                bundle_path,
                raw_file_record,
                context=f"{context}.files.{job_relative}",
            )
            by_relative_name[str(job_relative)] = bundle_path

        producer_checksum = _require_sha256(
            record["producer_checksum_file_sha256"],
            name=f"{context}.producer_checksum_file_sha256",
        )
        artifact_checksum = _require_sha256(
            record["artifact_checksum_file_sha256"],
            name=f"{context}.artifact_checksum_file_sha256",
        )
        if (
            _sha256_file(root / by_relative_name["producer-SHA256SUMS"])
            != producer_checksum
            or _sha256_file(root / by_relative_name["artifact-SHA256SUMS"])
            != artifact_checksum
        ):
            raise DomainLessonResultsError(
                f"{context} checksum-file hashes do not match"
            )
        producer_files = record["producer_files"]
        if not isinstance(producer_files, Mapping) or not producer_files:
            raise DomainLessonResultsError(
                f"{context}.producer_files must be a nonempty object"
            )
        normalized_producers = {
            str(name): _require_sha256(
                digest,
                name=f"{context}.producer_files.{name}",
            )
            for name, digest in producer_files.items()
        }
        if _positive_integer(
            record["verified_producer_file_count"],
            name=f"{context}.verified_producer_file_count",
        ) != len(normalized_producers):
            raise DomainLessonResultsError(
                f"{context} producer file count is incorrect"
            )
        if (
            _positive_integer(
                record["verified_artifact_file_count"],
                name=f"{context}.verified_artifact_file_count",
            )
            != len(job_files) - 1
        ):
            raise DomainLessonResultsError(f"{context} job file count is incorrect")
        producer_maps.append(normalized_producers)
    if any(value != producer_maps[0] for value in producer_maps[1:]):
        raise DomainLessonResultsError(
            "the 1/2/4-GPU jobs used different producer files"
        )

    if set(checksums) != declared:
        missing = declared - set(checksums)
        extra = set(checksums) - declared
        raise DomainLessonResultsError(
            f"checksum index does not match declared files; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    files_on_disk = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected_files = declared | {CHECKSUM_INDEX_NAME}
    if files_on_disk != expected_files:
        raise DomainLessonResultsError(
            "result directory contains missing or undeclared files"
        )


def load_domain_lesson_view(
    root: str | Path,
    *,
    expected_atom_count: int = FIXED_ATOM_COUNT,
    expected_world_sizes: Sequence[int] = REQUIRED_WORLD_SIZES,
) -> DomainLessonView:
    """Load one complete v5 fixed-input result set."""

    root_path = Path(root)
    if (
        isinstance(expected_atom_count, bool)
        or int(expected_atom_count) != FIXED_ATOM_COUNT
    ):
        raise ValueError(f"v5 records one fixed atom count: {FIXED_ATOM_COUNT}")
    worlds = tuple(int(value) for value in expected_world_sizes)
    if worlds != REQUIRED_WORLD_SIZES:
        raise ValueError("v5 records the configured 1/2/4-GPU world sizes")
    if not root_path.exists():
        return _empty_view(
            root_path,
            "recorded H100 results have not been reported",
        )
    if not root_path.is_dir():
        raise DomainLessonResultsError(
            "domain result path exists but is not a directory"
        )
    manifest_path = root_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DomainLessonResultsError(f"missing {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DomainLessonResultsError("manifest.json is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise DomainLessonResultsError("manifest.json must contain an object")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise DomainLessonResultsError(f"manifest schema must be {BUNDLE_SCHEMA}")
    if manifest.get("status") == "not reported":
        return _empty_view(
            root_path,
            str(manifest.get("reason") or "recorded H100 results are not reported"),
        )
    if manifest.get("status") != "complete":
        raise DomainLessonResultsError(
            "existing v5 result sets must have status complete"
        )

    checksums = _read_checksum_index(root_path)
    manifest_checksum = checksums.get(MANIFEST_NAME)
    if manifest_checksum is None:
        raise DomainLessonResultsError("SHA256SUMS is missing manifest.json")
    _checked_file(root_path, MANIFEST_NAME, manifest_checksum)
    settings_sha256, source_identity, structure_sha256 = _validate_identity(manifest)
    table = _read_table(root_path, manifest, checksums)
    pass_times = _validate_table(
        table,
        settings_sha256=settings_sha256,
        structure_sha256=structure_sha256,
    )
    raw_rows = _read_raw_results(root_path, manifest, checksums)
    fixed_raw = [row for row in raw_rows if row.get("mode") == "distributed"]
    electrostatics_raw = [
        row for row in raw_rows if row.get("mode") == "electrostatics-validation"
    ]
    if len(electrostatics_raw) != 1:
        raise DomainLessonResultsError(
            "raw results require one electrostatics-validation row"
        )
    if any(not str(row.get("run_id", "")).strip() for row in raw_rows):
        raise DomainLessonResultsError(
            "all four raw results must identify their source job"
        )
    rows_by_world, force_arrays, fixed_charge = _validate_raw_distributed(
        fixed_raw,
        table=table,
        source_identity=source_identity,
        settings_sha256=settings_sha256,
        structure_sha256=structure_sha256,
        checksums=checksums,
        root=root_path,
    )
    if electrostatics_raw[0]["run_id"] != rows_by_world[1]["run_id"]:
        raise DomainLessonResultsError(
            "electrostatics validation must come from the one-GPU job"
        )
    output_agreement = _output_agreement(rows_by_world, force_arrays)
    _validate_manifest_output_agreement(
        manifest,
        rows_by_world=rows_by_world,
        force_arrays=force_arrays,
    )
    electrostatics, _ = _validate_electrostatics(
        electrostatics_raw[0],
        manifest=manifest,
        source_identity=source_identity,
        settings_sha256=settings_sha256,
        checksums=checksums,
        root=root_path,
    )
    _validate_declared_files(root_path, manifest, checksums)
    layout, timing, plot = _learner_tables(table, pass_times)
    return DomainLessonView(
        available=True,
        reason="",
        root=root_path,
        manifest=manifest,
        run_settings_table=_settings_table(manifest),
        layout_table=layout,
        timing_table=timing,
        output_agreement_table=output_agreement,
        charge_diagnostics_table=pd.DataFrame([fixed_charge]),
        electrostatics_table=electrostatics,
        distributed_table=table.sort_values("gpus").reset_index(drop=True),
        plot_data=plot,
    )


__all__ = (
    "BUNDLE_SCHEMA",
    "DISTRIBUTED_COLUMNS",
    "DomainLessonResultsError",
    "DomainLessonView",
    "PLOT_COLUMNS",
    "canonical_json_sha256",
    "load_domain_lesson_view",
)
