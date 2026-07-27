#!/usr/bin/env python3
"""Plan, prepare, and assemble the Part 1 domain-decomposition run.

The ``plan`` command uses only the Python standard library. It is safe to run
on a laptop without CUDA, Torch, ALCHEMI Toolkit, ASE, or Packmol.

The ``prepare`` command loads the checked 3,200-atom periodic box used in the
live notebook and builds larger inputs as deterministic integer supercells.
Packmol is not run during the campaign.

The capacity ladder is explicit. Every planned case is run in a fresh
``torchrun`` process by the Slurm launcher. A failed process is turned into a
normal result row by ``record-failure`` and retained by ``assemble``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402


DOMAIN_METHODOLOGY_CONFIG_PATH = (
    PART_DIR / "aux" / "domain" / "config.py"
).resolve()
PLAN_SCHEMA = "alchemi.part1-domain-plan.v2"
INPUT_SCHEMA = "alchemi.part1-domain-input.v2"
BASE_BOX_SCHEMA = "alchemi.part1-domain-base-box.v1"
RESULT_SCHEMA = "alchemi.part1-domain-case.v3"
COLLECTION_SCHEMA = "alchemi.part1-domain-collection.v3"
SELECTION_SCHEMA = "alchemi.part1-domain-selection.v3"
DISTRIBUTED_PLAN_SCHEMA = "alchemi.part1-domain-distributed-plan.v2"
BUNDLE_SCHEMA = "alchemi.domain-decomposition-lesson.v3"
PHASE_SUMMARY_SCHEMA = "alchemi.part1-domain-phase-summary.v1"
CHECKPOINT_PREFLIGHT_SCHEMA = "alchemi.part1-domain-checkpoint-preflight.v1"

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

CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
NCI_SUBSET_SHA256 = "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
AIMNET_CHECKPOINT = "aimnet2-wb97m-d3_0"
AIMNET_CHECKPOINT_SHA256 = (
    "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
)
D3_PARAMETER_SHA256 = "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
NCI_SYSTEM_ID = DOMAIN_METHODOLOGY.nci_system_id
NCI_SCALE = DOMAIN_METHODOLOGY.nci_scale
BASE_PAIR_COUNT = DOMAIN_METHODOLOGY.live_molecules_per_species
BASE_ATOM_COUNT = BASE_PAIR_COUNT * DOMAIN_METHODOLOGY.atoms_per_composition_unit
DEFAULT_BASE_BOX_DIR = (
    PART_DIR / "data" / "domain_decomposition" / "prebuilt_base_box"
)

# Phenol (C6H6O) + N-methylacetamide (C3H7NO), using standard atomic
# weights. ``prepare`` recomputes this mass from the actual ASE templates and
# records both values.
PAIR_MASS_U_FROM_FORMULAS = 167.208
ATOMS_PER_PAIR = DOMAIN_METHODOLOGY.atoms_per_composition_unit
AIMNET_NEIGHBOR_CUTOFF_A = DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a
MOLECULE_COUNT_DEFINITION = (
    "The count is the number of independent phenol molecules and the equal "
    "number of independent N-methylacetamide molecules; it is not a count "
    "of pre-bound dimers."
)

DEFAULT_WORLD_SIZES = DOMAIN_METHODOLOGY.capacity_world_sizes
DEFAULT_DISTRIBUTED_WORLD_SIZES = DOMAIN_METHODOLOGY.distributed_world_sizes
# This is a declared benchmark ladder, not a runtime fit estimate. No case is
# skipped based on an atom or memory threshold.
DEFAULT_CAPACITY_PAIR_COUNTS = DOMAIN_METHODOLOGY.capacity_molecules_per_species
DEFAULT_VALIDATION_PAIRS = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
)
DEFAULT_DENSITY_G_CM3 = DOMAIN_METHODOLOGY.construction_density_g_cm3
DEFAULT_PACKMOL_TOLERANCE_A = DOMAIN_METHODOLOGY.packmol_tolerance_a
DEFAULT_PACKMOL_PRECISION_A = DOMAIN_METHODOLOGY.packmol_precision_a
DEFAULT_PACKMOL_SEED = DOMAIN_METHODOLOGY.packmol_seed
EXPECTED_PACKMOL_VERSION = "21.2.1"

DEFAULT_PME_CUTOFF_A = DOMAIN_METHODOLOGY.pme_realspace_cutoff_a
DEFAULT_PME_MESH_SAFETY_FACTOR = DOMAIN_METHODOLOGY.pme_mesh_safety_factor
DEFAULT_PME_SPLINE_ORDER = DOMAIN_METHODOLOGY.pme_spline_order
DEFAULT_PME_ACCURACY = DOMAIN_METHODOLOGY.pme_accuracy
DEFAULT_EWALD_REFERENCE_ACCURACY = DOMAIN_METHODOLOGY.ewald_reference_accuracy
DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
)
DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A = (
    DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
)
DEFAULT_CHARGE_SUM_TOL_E = DOMAIN_METHODOLOGY.charge_sum_tolerance_e
DEFAULT_PARITY_PAIR_COUNT = DOMAIN_METHODOLOGY.parity_molecules_per_species
# These same-input limits are declared before the multi-GPU differences are
# inspected. Energy is normalized by atom count so atomic baselines do not
# loosen the comparison.
DEFAULT_PARITY_ENERGY_TOL_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
)
DEFAULT_PARITY_FORCE_ATOL_EV_A = DOMAIN_METHODOLOGY.parity_force_atol_ev_a
DEFAULT_PARITY_FORCE_RTOL = DOMAIN_METHODOLOGY.parity_force_rtol
DEFAULT_D3_CUTOFF_A = DOMAIN_METHODOLOGY.d3_cutoff_a
DEFAULT_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction
# The Toolkit 0.2 multi-GPU D3 tests use a 4 A margin for coordination
# numbers around borrowed atoms. This is a declared starting value, not a
# universal bound; the campaign accepts it only after same-input force parity.
DEFAULT_DOMAIN_SKIN_A = DOMAIN_METHODOLOGY.domain_halo_skin_a
DEFAULT_STEADY_TIMING_WARMUP_COUNT = (
    DOMAIN_METHODOLOGY.steady_timing_warmup_count
)
DEFAULT_STEADY_TIMING_SAMPLE_COUNT = DOMAIN_METHODOLOGY.steady_timing_sample_count

CUDA_OOM_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDNN_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def methodology_source_identity() -> dict[str, Any]:
    """Identify the versioned methodology module used to resolve defaults."""

    return {
        "schema": DOMAIN_METHODOLOGY.schema,
        "name": DOMAIN_METHODOLOGY.name,
        "version": DOMAIN_METHODOLOGY.version,
        "path": str(DOMAIN_METHODOLOGY_CONFIG_PATH),
        "sha256": sha256_file(DOMAIN_METHODOLOGY_CONFIG_PATH),
    }


def resolved_methodology_values(
    *,
    capacity_pair_counts: tuple[int, ...] | None = None,
    validation_pairs: int | None = None,
    parity_pairs: int | None = None,
    density_g_cm3: float | None = None,
    pme_cutoff_a: float | None = None,
    pme_mesh_safety_factor: float | None = None,
    pme_spline_order: int | None = None,
    pme_accuracy: float | None = None,
    ewald_reference_accuracy: float | None = None,
    d3_cutoff_a: float | None = None,
    d3_smoothing_fraction: float | None = None,
    domain_skin_a: float | None = None,
    packmol_tolerance_a: float | None = None,
    packmol_precision_a: float | None = None,
    packmol_seed: int | None = None,
    steady_timing_warmup_count: int | None = None,
    steady_timing_sample_count: int | None = None,
) -> dict[str, Any]:
    """Return defaults with every explicit campaign override applied."""

    values = DOMAIN_METHODOLOGY.resolved_values(json_compatible=True)
    overrides = {
        "capacity_molecules_per_species": (
            list(capacity_pair_counts)
            if capacity_pair_counts is not None
            else None
        ),
        "electrostatics_validation_molecules_per_species": validation_pairs,
        "parity_molecules_per_species": parity_pairs,
        "construction_density_g_cm3": density_g_cm3,
        "pme_realspace_cutoff_a": pme_cutoff_a,
        "pme_mesh_safety_factor": pme_mesh_safety_factor,
        "pme_spline_order": pme_spline_order,
        "pme_accuracy": pme_accuracy,
        "ewald_reference_accuracy": ewald_reference_accuracy,
        "d3_cutoff_a": d3_cutoff_a,
        "d3_smoothing_fraction": d3_smoothing_fraction,
        "domain_halo_skin_a": domain_skin_a,
        "packmol_tolerance_a": packmol_tolerance_a,
        "packmol_precision_a": packmol_precision_a,
        "packmol_seed": packmol_seed,
        "steady_timing_warmup_count": steady_timing_warmup_count,
        "steady_timing_sample_count": steady_timing_sample_count,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    return values


def checkpoint_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and verify the exact AIMNet2 checkpoint before launching cases."""

    from importlib import metadata

    from aimnet.calculators.model_registry import get_model_path

    checkpoint = Path(get_model_path(args.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AIMNet2 checkpoint is missing: {checkpoint}")
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256 != AIMNET_CHECKPOINT_SHA256:
        raise ValueError(
            "AIMNet2 checkpoint SHA-256 does not match the declared campaign "
            f"value: {observed_sha256}"
        )
    report = {
        "schema": CHECKPOINT_PREFLIGHT_SCHEMA,
        "created_utc": utc_now(),
        "alias": AIMNET_CHECKPOINT,
        "requested": args.checkpoint,
        "path": str(checkpoint),
        "sha256": observed_sha256,
        "size_bytes": checkpoint.stat().st_size,
        "aimnet_version": metadata.version("aimnet"),
    }
    atomic_write_json(args.output.resolve(), report)
    return report


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_positive_ints(values: Iterable[str | int]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("at least one positive integer is required")
    if len(set(parsed)) != len(parsed):
        raise ValueError("values must be unique")
    return parsed


def equivalent_cubic_length_angstrom(
    pair_count: int,
    density_g_cm3: float,
    pair_mass_u: float = PAIR_MASS_U_FROM_FORMULAS,
) -> float:
    """Return the side of a cube with the input's volume."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0.0:
        raise ValueError("density_g_cm3 must be positive and finite")
    # 1 u = 1.66053906660e-24 g and 1 cm3 = 1e24 A3.
    volume_a3 = pair_count * pair_mass_u * 1.66053906660 / density_g_cm3
    return volume_a3 ** (1.0 / 3.0)


def balanced_repeat_factors(pair_count: int) -> tuple[int, int, int]:
    """Plan balanced x/y/z repeats of the checked 128-pair base box."""

    if pair_count < BASE_PAIR_COUNT or pair_count % BASE_PAIR_COUNT:
        raise ValueError(
            f"pair_count must be a multiple of the {BASE_PAIR_COUNT}-pair base box"
        )
    multiplier = pair_count // BASE_PAIR_COUNT
    if multiplier & (multiplier - 1):
        raise ValueError("pair_count/base pair count must be a power of two")
    exponent = multiplier.bit_length() - 1
    quotient, remainder = divmod(exponent, 3)
    factors = [2**quotient] * 3
    for axis in range(3 - remainder, 3):
        factors[axis] *= 2
    return tuple(factors)


def planned_supercell_geometry(
    pair_count: int,
    density_g_cm3: float,
) -> dict[str, Any]:
    """Return the orthorhombic geometry implied by the base-box repeats."""

    repeat_factors = balanced_repeat_factors(pair_count)
    base_length_a = equivalent_cubic_length_angstrom(
        BASE_PAIR_COUNT,
        density_g_cm3,
    )
    cell_lengths_a = tuple(
        base_length_a * repeat_factor for repeat_factor in repeat_factors
    )
    volume_a3 = math.prod(cell_lengths_a)
    return {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(cell_lengths_a),
        "minimum_cell_length_a": min(cell_lengths_a),
        "equivalent_cubic_length_a": volume_a3 ** (1.0 / 3.0),
        "volume_a3": volume_a3,
    }


def require_planned_supercell_geometry(
    geometry: dict[str, Any],
    *,
    pair_count: int,
    density_g_cm3: float,
) -> None:
    """Require an input cell to match the declared base-box repeats."""

    expected = planned_supercell_geometry(pair_count, density_g_cm3)
    if geometry.get("cell_geometry") != expected["cell_geometry"]:
        raise ValueError("input cell geometry does not match the plan")
    for name in (
        "minimum_cell_length_a",
        "equivalent_cubic_length_a",
        "volume_a3",
    ):
        if not math.isclose(
            float(geometry[name]),
            float(expected[name]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            raise ValueError("input cell geometry does not match the plan")
    if any(
        not math.isclose(
            float(observed),
            float(planned),
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        )
        for observed, planned in zip(
            geometry["cell_lengths_a"],
            expected["cell_lengths_a"],
            strict=True,
        )
    ):
        raise ValueError("input cell geometry does not match the plan")


def _cell_geometry_from_matrix(
    cell_a: Any,
) -> tuple[tuple[float, float, float], float]:
    """Return vector lengths and volume for one orthorhombic cell matrix."""

    try:
        matrix = tuple(
            tuple(float(component) for component in vector) for vector in cell_a
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_a must be a 3 by 3 numeric matrix") from exc
    if len(matrix) != 3 or any(len(vector) != 3 for vector in matrix):
        raise ValueError("cell_a must be a 3 by 3 numeric matrix")
    if any(not math.isfinite(component) for vector in matrix for component in vector):
        raise ValueError("cell_a must contain only finite values")

    cell_lengths_a = tuple(
        math.sqrt(sum(component * component for component in vector))
        for vector in matrix
    )
    if any(length <= 0.0 for length in cell_lengths_a):
        raise ValueError("cell vectors must have positive lengths")
    for first in range(3):
        for second in range(first + 1, 3):
            dot = sum(
                matrix[first][axis] * matrix[second][axis] for axis in range(3)
            )
            scale = cell_lengths_a[first] * cell_lengths_a[second]
            if not math.isclose(dot, 0.0, rel_tol=0.0, abs_tol=1.0e-10 * scale):
                raise ValueError("the domain campaign requires an orthorhombic cell")

    volume_a3 = abs(
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if volume_a3 <= 0.0:
        raise ValueError("cell_a must have a positive volume")
    return cell_lengths_a, volume_a3


def validated_manifest_cell_geometry(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return the explicit geometry in an input manifest."""

    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("input manifest has an unknown schema")
    required_fields = (
        "cell_geometry",
        "cell_a",
        "cell_lengths_a",
        "minimum_cell_length_a",
        "equivalent_cubic_length_a",
        "volume_a3",
    )
    missing = [name for name in required_fields if name not in manifest]
    if missing:
        raise ValueError(
            "input manifest is missing explicit cell geometry: "
            + ", ".join(missing)
        )
    if manifest["cell_geometry"] != "orthorhombic":
        raise ValueError("the domain campaign requires an orthorhombic cell")

    cell_lengths_a, volume_a3 = _cell_geometry_from_matrix(manifest["cell_a"])
    try:
        recorded_lengths = tuple(float(value) for value in manifest["cell_lengths_a"])
        recorded_minimum = float(manifest["minimum_cell_length_a"])
        recorded_equivalent = float(manifest["equivalent_cubic_length_a"])
        recorded_volume = float(manifest["volume_a3"])
    except (TypeError, ValueError) as exc:
        raise ValueError("input manifest cell geometry is not numeric") from exc
    if len(recorded_lengths) != 3:
        raise ValueError("cell_lengths_a must contain x, y, and z lengths")

    equivalent_cubic_length_a = volume_a3 ** (1.0 / 3.0)
    expected_values = (
        (*cell_lengths_a, min(cell_lengths_a), equivalent_cubic_length_a, volume_a3)
    )
    observed_values = (
        (*recorded_lengths, recorded_minimum, recorded_equivalent, recorded_volume)
    )
    if any(
        not math.isclose(observed, expected, rel_tol=1.0e-10, abs_tol=1.0e-10)
        for observed, expected in zip(
            observed_values,
            expected_values,
            strict=True,
        )
    ):
        raise ValueError("input manifest cell geometry is internally inconsistent")
    return {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(cell_lengths_a),
        "minimum_cell_length_a": min(cell_lengths_a),
        "equivalent_cubic_length_a": equivalent_cubic_length_a,
        "volume_a3": volume_a3,
    }


def domain_partition_check(
    *,
    cell_lengths_a: Iterable[float],
    ghost_width_a: float,
) -> dict[str, Any]:
    """Describe the runtime check for a meaningful domain partition."""

    try:
        lengths = tuple(float(value) for value in cell_lengths_a)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_lengths_a must contain numeric values") from exc
    if len(lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in lengths
    ):
        raise ValueError("cell_lengths_a must contain three positive finite values")
    if not math.isfinite(ghost_width_a) or ghost_width_a <= 0.0:
        raise ValueError("ghost_width_a must be positive and finite")
    return {
        "cell_lengths_a": list(lengths),
        "ghost_width_a": ghost_width_a,
        "acceptance_rule": (
            "Toolkit DomainParallel require_nondegenerate=True must confirm "
            "that every rank retains remote atoms"
        ),
        "checked_during_each_multi_gpu_run": True,
        "box_length_only_precheck": "not_used",
        "reason": (
            "Whether a halo covers the full structure depends on the actual "
            "rank layout and atom positions, not only the shortest box axis."
        ),
    }


def expected_pme_setup(
    *,
    cell_lengths_a: Iterable[float],
    real_space_cutoff_a: float,
    accuracy: float,
    mesh_safety_factor: float,
) -> dict[str, Any]:
    """Mirror the pinned Toolkit-Ops PME estimator for one orthorhombic cell."""

    try:
        lengths = tuple(float(value) for value in cell_lengths_a)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_lengths_a must contain numeric values") from exc
    if len(lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in lengths
    ):
        raise ValueError("cell_lengths_a must contain three positive finite values")
    values = {
        "real_space_cutoff_a": real_space_cutoff_a,
        "accuracy": accuracy,
        "mesh_safety_factor": mesh_safety_factor,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("PME estimator inputs must be positive and finite")
    alpha = math.sqrt(-math.log(accuracy)) / real_space_cutoff_a
    raw_mesh_dimensions = tuple(
        mesh_safety_factor
        * 2.0
        * alpha
        * cell_length_a
        / (3.0 * accuracy**0.2)
        for cell_length_a in lengths
    )
    mesh_dimensions = tuple(
        int(2 ** math.ceil(math.log2(raw_mesh_dimension)))
        for raw_mesh_dimension in raw_mesh_dimensions
    )
    return {
        "real_space_cutoff_a": real_space_cutoff_a,
        "alpha_a_inverse": alpha,
        "mesh_dimensions": list(mesh_dimensions),
        "mesh_spacing_a": [
            cell_length_a / mesh_dimension
            for cell_length_a, mesh_dimension in zip(
                lengths,
                mesh_dimensions,
                strict=True,
            )
        ],
        "accuracy": accuracy,
        "mesh_safety_factor": mesh_safety_factor,
        "parameter_rule": (
            "estimate_pme_parameters(accuracy, real_space_cutoff, "
            "mesh_safety_factor)"
        ),
    }


def expected_ewald_reference_setup(
    *,
    atom_count: int,
    volume_a3: float,
    accuracy: float,
) -> dict[str, Any]:
    """Mirror the pinned Toolkit-Ops direct Ewald estimator from cell volume."""

    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if (
        not math.isfinite(volume_a3)
        or volume_a3 <= 0.0
        or not math.isfinite(accuracy)
        or not 0.0 < accuracy < 1.0
    ):
        raise ValueError("Ewald estimator inputs are invalid")
    eta = (volume_a3**2 / atom_count) ** (1.0 / 6.0) / math.sqrt(
        2.0 * math.pi
    )
    error_factor = math.sqrt(-2.0 * math.log(accuracy))
    return {
        "real_space_cutoff_a": error_factor * eta,
        "reciprocal_space_cutoff_a_inverse": error_factor / eta,
        "alpha_a_inverse": 1.0 / (math.sqrt(2.0) * eta),
        "accuracy": accuracy,
        "parameter_rule": "estimate_ewald_parameters(accuracy)",
    }


def capacity_case_id(pair_count: int, world_size: int) -> str:
    return f"capacity-pairs-{pair_count:06d}-gpus-{world_size:02d}"


def validation_case_id(pair_count: int) -> str:
    return f"electrostatics-validation-pairs-{pair_count:06d}-gpus-01"


def parity_case_id(pair_count: int, world_size: int) -> str:
    return f"parity-pairs-{pair_count:06d}-gpus-{world_size:02d}"


def steady_timing_case_id(pair_count: int, world_size: int) -> str:
    return f"steady-timing-pairs-{pair_count:06d}-gpus-{world_size:02d}"


def speed_case_id(pair_count: int, world_size: int) -> str:
    """Return the steady-timing ID for callers of the former helper name."""

    return steady_timing_case_id(pair_count, world_size)


def rescue_case_id(pair_count: int, world_size: int) -> str:
    return f"rescue-pairs-{pair_count:06d}-gpus-{world_size:02d}"


def measurement_role_for_mode(mode: str) -> str:
    """Return the explicit campaign role represented by a runner mode."""

    try:
        return {
            "capacity": "capacity",
            "parity": "parity",
            "distributed": "rescue",
            "steady-timing": "steady_timing",
            "electrostatics-validation": "electrostatics_validation",
        }[mode]
    except KeyError as exc:
        raise ValueError(f"unknown domain campaign mode: {mode}") from exc


def _require_complete_distributed_outcomes(
    selection: dict[str, Any],
    rows_by_case: dict[str, dict[str, Any]],
) -> tuple[int, ...]:
    """Require complete steady timings and a successful OOM retry."""

    largest_success_count = int(selection["largest_success"]["input"]["pair_count"])
    first_oom_count = int(selection["first_cuda_oom"]["input"]["pair_count"])

    for world_size in DOMAIN_METHODOLOGY.steady_timing_world_sizes:
        case_id = steady_timing_case_id(largest_success_count, world_size)
        row = rows_by_case.get(case_id)
        if row is None:
            raise ValueError(f"missing {world_size}-GPU steady-timing case: {case_id}")
        if not bool(row.get("success")):
            raise ValueError(
                f"{world_size}-GPU steady-timing case did not complete: {case_id}"
            )
        if row.get("measurement_role") != "steady_timing":
            raise ValueError(f"steady-timing case has the wrong role: {case_id}")

    successful_rescue_world_sizes: list[int] = []
    for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        case_id = rescue_case_id(first_oom_count, world_size)
        row = rows_by_case.get(case_id)
        if row is None:
            raise ValueError(f"missing {world_size}-GPU OOM retry: {case_id}")
        if bool(row.get("success")):
            successful_rescue_world_sizes.append(world_size)
    if not successful_rescue_world_sizes:
        raise ValueError(
            "the first one-GPU CUDA OOM input did not succeed on any declared "
            "multi-GPU size"
        )
    return tuple(successful_rescue_world_sizes)


def validate_recorded_rank_layout(
    distributed: dict[str, Any],
    *,
    world_size: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Validate a layout recorded by Toolkit for the case's actual cell."""

    try:
        cells_per_dim = tuple(int(value) for value in distributed["cells_per_dim"])
        rank_grid = tuple(int(value) for value in distributed["rank_grid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("distributed result has no valid recorded layout") from exc
    if (
        len(cells_per_dim) != 3
        or len(rank_grid) != 3
        or any(value <= 0 for value in (*cells_per_dim, *rank_grid))
        or math.prod(rank_grid) != world_size
        or any(ranks > cells for ranks, cells in zip(
            rank_grid,
            cells_per_dim,
            strict=True,
        ))
        or any(cells % ranks for ranks, cells in zip(
            rank_grid,
            cells_per_dim,
            strict=True,
        ))
    ):
        raise ValueError(
            "recorded cells_per_dim and rank_grid do not match the world size"
        )
    return cells_per_dim, rank_grid


def input_directory_name(pair_count: int) -> str:
    return f"phenol-nma-pairs-{pair_count:06d}"


def result_filename(case_id: str) -> str:
    return f"{case_id}.json"


def build_plan(
    *,
    run_id: str,
    world_sizes: tuple[int, ...],
    capacity_pair_counts: tuple[int, ...],
    validation_pairs: int,
    density_g_cm3: float,
    pme_cutoff_a: float,
    pme_mesh_safety_factor: float,
    pme_spline_order: int,
    pme_accuracy: float,
    ewald_reference_accuracy: float,
    d3_cutoff_a: float,
    d3_smoothing_fraction: float,
    domain_skin_a: float,
    packmol_tolerance_a: float,
    packmol_precision_a: float,
    packmol_seed: int,
    steady_timing_warmup_count: int = DEFAULT_STEADY_TIMING_WARMUP_COUNT,
    steady_timing_sample_count: int = DEFAULT_STEADY_TIMING_SAMPLE_COUNT,
) -> dict[str, Any]:
    if not run_id or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id may contain only letters, numbers, '-' and '_'")
    unsupported = set(world_sizes) - set(DEFAULT_WORLD_SIZES)
    if unsupported:
        raise ValueError(
            f"world sizes must be selected from {DEFAULT_WORLD_SIZES}; "
            f"got {sorted(unsupported)}"
        )
    if tuple(sorted(capacity_pair_counts)) != capacity_pair_counts:
        raise ValueError("capacity pair counts must be strictly increasing")
    for pair_count in (*capacity_pair_counts, validation_pairs):
        balanced_repeat_factors(pair_count)
    positive_floats = {
        "density_g_cm3": density_g_cm3,
        "pme_cutoff_a": pme_cutoff_a,
        "pme_mesh_safety_factor": pme_mesh_safety_factor,
        "pme_accuracy": pme_accuracy,
        "ewald_reference_accuracy": ewald_reference_accuracy,
        "d3_cutoff_a": d3_cutoff_a,
        "domain_skin_a": domain_skin_a,
        "packmol_tolerance_a": packmol_tolerance_a,
        "packmol_precision_a": packmol_precision_a,
    }
    for name, value in positive_floats.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if pme_spline_order <= 0:
        raise ValueError("pme_spline_order must be positive")
    if steady_timing_warmup_count < 1:
        raise ValueError("steady_timing_warmup_count must be at least 1")
    if steady_timing_sample_count < 5:
        raise ValueError("steady_timing_sample_count must be at least 5")
    if not 0.0 <= d3_smoothing_fraction < 1.0:
        raise ValueError("d3_smoothing_fraction must be in [0, 1)")
    if packmol_precision_a >= packmol_tolerance_a:
        raise ValueError("packmol_precision_a must be smaller than tolerance")

    capacity_cases = []
    for world_size in world_sizes:
        for pair_count in capacity_pair_counts:
            case_id = capacity_case_id(pair_count, world_size)
            geometry = planned_supercell_geometry(pair_count, density_g_cm3)
            capacity_cases.append(
                {
                    "case_id": case_id,
                    "mode": "capacity",
                    "measurement_role": "capacity",
                    "world_size": world_size,
                    "pair_count": pair_count,
                    "molecules_per_species": pair_count,
                    "atom_count": pair_count * ATOMS_PER_PAIR,
                    **geometry,
                    "repeat_factors_xyz": list(
                        balanced_repeat_factors(pair_count)
                    ),
                    "rank_grid_policy": "automatic_from_actual_cell",
                    "input_directory": input_directory_name(pair_count),
                    "result_file": result_filename(case_id),
                    "fresh_process_required": True,
                }
            )

    validation_cases: list[dict[str, Any]] = []
    if 1 in world_sizes:
        case_id = validation_case_id(validation_pairs)
        geometry = planned_supercell_geometry(validation_pairs, density_g_cm3)
        validation_cases.append(
            {
                "case_id": case_id,
                "mode": "electrostatics-validation",
                "measurement_role": "electrostatics_validation",
                "world_size": 1,
                "pair_count": validation_pairs,
                "molecules_per_species": validation_pairs,
                "atom_count": validation_pairs * ATOMS_PER_PAIR,
                **geometry,
                "repeat_factors_xyz": list(
                    balanced_repeat_factors(validation_pairs)
                ),
                "input_directory": input_directory_name(validation_pairs),
                "result_file": result_filename(case_id),
                "fresh_process_required": True,
                "comparison": (
                    "PME and Ewald on the same geometry and AIMNet2-predicted charges"
                ),
            }
        )

    return {
        "schema": PLAN_SCHEMA,
        "created_utc": utc_now(),
        "run_id": run_id,
        "description": (
            "One periodic 1:1 phenol and N-methylacetamide box. The main "
            "calculation is AIMNet2 checkpoint base plus PME electrostatics plus "
            "D3(BJ) dispersion through DomainParallel."
        ),
        "source": {
            "toolkit_core_commit": CORE_COMMIT,
            "toolkit_ops_commit": OPS_COMMIT,
            "nci_subset_sha256": NCI_SUBSET_SHA256,
            "aimnet_checkpoint": AIMNET_CHECKPOINT,
            "aimnet_checkpoint_sha256": AIMNET_CHECKPOINT_SHA256,
            "d3_parameter_sha256": D3_PARAMETER_SHA256,
            "domain_methodology_config": methodology_source_identity(),
        },
        "methodology": {
            "source": DOMAIN_METHODOLOGY.as_record(),
            "source_identity": methodology_source_identity(),
            "resolved_values": resolved_methodology_values(
                capacity_pair_counts=capacity_pair_counts,
                validation_pairs=validation_pairs,
                density_g_cm3=density_g_cm3,
                pme_cutoff_a=pme_cutoff_a,
                pme_mesh_safety_factor=pme_mesh_safety_factor,
                pme_spline_order=pme_spline_order,
                pme_accuracy=pme_accuracy,
                ewald_reference_accuracy=ewald_reference_accuracy,
                d3_cutoff_a=d3_cutoff_a,
                d3_smoothing_fraction=d3_smoothing_fraction,
                domain_skin_a=domain_skin_a,
                packmol_tolerance_a=packmol_tolerance_a,
                packmol_precision_a=packmol_precision_a,
                packmol_seed=packmol_seed,
                steady_timing_warmup_count=steady_timing_warmup_count,
                steady_timing_sample_count=steady_timing_sample_count,
            ),
        },
        "input": {
            "molecules": ["phenol", "N-methylacetamide"],
            "nci_system_id": NCI_SYSTEM_ID,
            "nci_scale": NCI_SCALE,
            "stoichiometry": "1:1",
            "count_definition": MOLECULE_COUNT_DEFINITION,
            "atoms_per_pair": ATOMS_PER_PAIR,
            "pair_mass_u_from_formulas": PAIR_MASS_U_FROM_FORMULAS,
            "construction_density_g_cm3": density_g_cm3,
            "packmol_tolerance_a": packmol_tolerance_a,
            "packmol_precision_a": packmol_precision_a,
            "packmol_base_seed": packmol_seed,
            "packmol_version": EXPECTED_PACKMOL_VERSION,
            "packmol_periodic_boundary_check": True,
            "construction_method": "balanced_integer_supercell_repeat",
            "base_box_schema": BASE_BOX_SCHEMA,
            "base_pair_count": BASE_PAIR_COUNT,
            "base_atom_count": BASE_ATOM_COUNT,
            "interpretation": (
                "Packmol created one checked 128-pair starting geometry. "
                "Larger inputs are balanced integer supercells of that box, "
                "which preserves its composition and construction density. "
                "The geometry is not equilibrated and the density is not a "
                "material prediction."
            ),
        },
        "model": {
            "aimnet_checkpoint": AIMNET_CHECKPOINT,
            "aimnet_compile_model": False,
            "pme_cutoff_a": pme_cutoff_a,
            "pme_mesh_safety_factor": pme_mesh_safety_factor,
            "pme_parameter_rule": (
                "estimate_pme_parameters(accuracy, real_space_cutoff, "
                "mesh_safety_factor)"
            ),
            "pme_spline_order": pme_spline_order,
            "pme_accuracy": pme_accuracy,
            "ewald_reference_accuracy": ewald_reference_accuracy,
            "d3_cutoff_a": d3_cutoff_a,
            "d3_smoothing_fraction": d3_smoothing_fraction,
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
        },
        "distributed": {
            "api": "DomainParallel",
            "mesh_dim_names": ["domain"],
            "grid_dims": DOMAIN_METHODOLOGY.domain_grid_dims,
            "rank_grid_policy": (
                "Toolkit SpatialPartitioner derives cells_per_dim and rank_grid "
                "from each input's actual cell shape and the domain cutoff"
            ),
            "recorded_layout_fields": ["cells_per_dim", "rank_grid"],
            "domain_cutoff_a": max(
                pme_cutoff_a,
                d3_cutoff_a,
                AIMNET_NEIGHBOR_CUTOFF_A,
            ),
            "domain_skin_a": domain_skin_a,
            "compile": False,
            "require_nondegenerate_for_world_size_gt_1": True,
            "pme_reciprocal_mesh": (
                "replicated on every rank in the Toolkit 0.2 version used here"
            ),
        },
        "timing": {
            "cold": {
                "measurement_kind": "cold_one_shot_partition_run_gather",
                "measurement_roles": ["capacity", "parity", "rescue"],
                "boundary": (
                    "After fresh DomainParallel construction and context entry: "
                    "rank barrier and CUDA synchronization; one public partition, "
                    "run, and gather workflow; final CUDA synchronization. Model "
                    "loading the checked base box, constructing its supercell, "
                    "model loading, Batch construction, host-to-device transfer, "
                    "file writes, output checks, context cleanup, and process "
                    "launch are outside this time."
                ),
                "warmup_count": 0,
                "sample_count": 1,
                "work_note": (
                    "These rows answer fit and correctness questions and are not "
                    "used for speedup. Toolkit's multi-rank path performs an "
                    "automatic initial force evaluation."
                ),
            },
            "steady": {
                "measurement_kind": "steady_partition_run_gather",
                "measurement_role": "steady_timing",
                "world_sizes": list(DOMAIN_METHODOLOGY.steady_timing_world_sizes),
                "boundary": (
                    "Each warmup and measured sample creates and enters a fresh "
                    "DomainParallel context outside the timer. After a rank barrier "
                    "and CUDA synchronization, the timer covers exactly one public "
                    "partition, run, and gather workflow plus final CUDA "
                    "synchronization. The elapsed value is reduced with the maximum "
                    "across ranks. Output and unchanged-input checks, statistics, "
                    "context exit, and file writes are outside the timer. One rank "
                    "requests two BaseDynamics steps; several ranks request one "
                    "step after DomainParallel's automatic initial force "
                    "evaluation."
                ),
                "warmup_count": steady_timing_warmup_count,
                "sample_count": steady_timing_sample_count,
                "model_evaluations_per_workflow": (
                    DOMAIN_METHODOLOGY.steady_timing_model_evaluations_per_workflow
                ),
                "one_rank_run_steps": DOMAIN_METHODOLOGY.steady_timing_run_steps(1),
                "multi_rank_run_steps": (
                    DOMAIN_METHODOLOGY.steady_timing_run_steps(
                        DOMAIN_METHODOLOGY.distributed_world_sizes[0]
                    )
                ),
                "summary": "median, Q1, Q3, and IQR of max-rank sample seconds",
                "quartile_method": "inclusive linear interpolation",
                "max_relative_iqr": (
                    DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr
                ),
            },
            "publishable_benchmark": False,
            "interpretation": (
                "Only complete dedicated steady_timing rows on the identical "
                "selected input are used for speedup and parallel efficiency. "
                "Cold capacity, parity, and rescue observations are excluded."
            ),
            "memory_boundary": (
                "For successful cases, per-rank CUDA peaks are reset after model "
                "setup and input transfer, immediately before each public workflow. "
                "The reported peak is the maximum over all workflows and ranks. "
                "Failed cases retain the peak reached before the failure."
            ),
        },
        "validation_cases": validation_cases,
        "validation_acceptance": {
            "declared_before_measurement": True,
            "absolute_energy_difference_ev_per_atom_max": (
                DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM
            ),
            "force_difference_max_norm_ev_a_max": (DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A),
            "absolute_charge_sum_e_max": DEFAULT_CHARGE_SUM_TOL_E,
        },
        "capacity_cases": capacity_cases,
        "capacity_policy": (
            "Run the declared one-GPU ladder in fresh processes and stop after "
            "the first genuine CUDA OOM, retaining that failed attempt. Retry "
            "that exact input on the declared multi-GPU grids. Do not lower "
            "cutoffs, change precision, skip an attempted row, or create an "
            "artificial OOM."
        ),
    }


def prepare_input(args: argparse.Namespace) -> dict[str, Any]:
    """Build one campaign input from the checked 3,200-atom base box."""

    # Heavy, optional imports live only in the command that needs them.
    import numpy as np
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    from aux.domain.packing import build_molecular_supercell

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    extxyz_path = output_dir / "structure.extxyz"
    base_dir = args.base_box_dir.resolve()
    base_manifest_path = base_dir / "manifest.json"
    base_structure_path = base_dir / "structure.extxyz"

    if args.reuse_existing:
        if not manifest_path.is_file() or not extxyz_path.is_file():
            raise FileNotFoundError(
                "reuse requested but manifest.json or structure.extxyz is missing"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != INPUT_SCHEMA:
            raise ValueError("existing input manifest has an unknown schema")
        if manifest["structure"]["sha256"] != sha256_file(extxyz_path):
            raise ValueError("existing structure checksum does not match its manifest")
        geometry = validated_manifest_cell_geometry(manifest)
        require_planned_supercell_geometry(
            geometry,
            pair_count=args.pair_count,
            density_g_cm3=args.density_g_cm3,
        )
        if (
            int(manifest["pair_count"]) != args.pair_count
            or int(manifest["molecules_per_species"]) != args.pair_count
            or manifest["count_definition"] != MOLECULE_COUNT_DEFINITION
            or float(manifest["construction_density_g_cm3"])
            != float(args.density_g_cm3)
            or manifest["construction"]["base_box_manifest_sha256"]
            != sha256_file(base_manifest_path)
            or manifest["construction"]["base_box_structure_sha256"]
            != sha256_file(base_structure_path)
        ):
            raise ValueError("existing input was built with different settings")
        return manifest

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"input directory is not empty: {output_dir}; "
            "use a new directory or --reuse-existing"
        )
    if not base_manifest_path.is_file() or not base_structure_path.is_file():
        raise FileNotFoundError(
            "checked base box needs manifest.json and structure.extxyz: "
            f"{base_dir}"
        )

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != BASE_BOX_SCHEMA:
        raise ValueError("base box manifest has an unknown schema")
    base_structure_record = base_manifest.get("structure", {})
    base_source_record = base_manifest.get("source", {})
    base_packmol_record = base_source_record.get("packmol", {})
    base_methodology_record = base_manifest.get("methodology", {})
    if (
        int(base_structure_record.get("molecules_per_species", -1))
        != BASE_PAIR_COUNT
        or int(base_structure_record.get("atom_count", -1)) != BASE_ATOM_COUNT
        or int(base_structure_record.get("molecule_count", -1))
        != 2 * BASE_PAIR_COUNT
        or base_source_record.get("molecule_counts")
        != {"phenol": BASE_PAIR_COUNT, "N-methylacetamide": BASE_PAIR_COUNT}
    ):
        raise ValueError("base box does not match the declared 128-pair system")
    if base_structure_record["sha256"] != sha256_file(base_structure_path):
        raise ValueError("base box structure checksum does not match its manifest")
    if (
        float(base_structure_record["construction_density_g_cm3"])
        != float(args.density_g_cm3)
        or float(base_packmol_record["tolerance_a"])
        != float(args.tolerance_a)
        or float(base_packmol_record.get("precision_a", math.nan))
        != float(args.precision_a)
        or int(base_packmol_record["seed"]) != int(args.seed)
        or base_packmol_record.get("version") != EXPECTED_PACKMOL_VERSION
        or base_source_record["nci_subset_sha256"] != NCI_SUBSET_SHA256
        or str(base_source_record.get("nci_system_id")) != str(NCI_SYSTEM_ID)
        or float(base_source_record.get("nci_scale", math.nan)) != float(NCI_SCALE)
    ):
        raise ValueError("base box was built with different construction settings")
    if (
        base_methodology_record.get("schema") != DOMAIN_METHODOLOGY.schema
        or base_methodology_record.get("name")
        != DOMAIN_METHODOLOGY.name
        or base_methodology_record.get("version")
        != DOMAIN_METHODOLOGY.version
    ):
        raise ValueError("base box was built with a different domain methodology")
    if args.nci_data is not None:
        nci_data = args.nci_data.resolve()
        if not nci_data.is_file() or sha256_file(nci_data) != NCI_SUBSET_SHA256:
            raise ValueError("NCI data file does not match the checked base box")

    base_atoms = ase_read(base_structure_path, format="extxyz")
    if len(base_atoms) != BASE_ATOM_COUNT:
        raise ValueError("base box structure has the wrong atom count")
    for name, expected in base_structure_record.get("arrays", {}).items():
        if name not in base_atoms.arrays:
            raise ValueError(f"base box structure is missing array {name}")
        values = np.asarray(base_atoms.arrays[name])
        if (
            str(values.dtype) != expected["dtype"]
            or list(values.shape) != expected["shape"]
            or sha256(values.tobytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"base box array {name} does not match its manifest")
    if (
        int(base_atoms.info.get("charge", -1)) != 0
        or int(base_atoms.info.get("pair_count", -1)) != BASE_PAIR_COUNT
        or base_atoms.info.get("count_definition") != MOLECULE_COUNT_DEFINITION
    ):
        raise ValueError("base box structure metadata does not match the lesson")
    packed, repeat_factors = build_molecular_supercell(
        base_atoms,
        base_pair_count=BASE_PAIR_COUNT,
        target_pair_count=args.pair_count,
    )
    if len(packed) != args.pair_count * ATOMS_PER_PAIR:
        raise RuntimeError("expanded structure has the wrong atom count")
    if int(packed.info.get("charge", 0)) != 0:
        raise ValueError("expanded molecular box must be neutral")

    density_from_mass_and_cell = (
        float(np.sum(packed.get_masses()))
        * 1.66053906660
        / float(packed.get_volume())
    )
    if not math.isclose(
        density_from_mass_and_cell,
        args.density_g_cm3,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("expanded box density does not match the checked base box")
    pair_mass_u_ase = float(np.sum(packed.get_masses())) / args.pair_count
    if abs(pair_mass_u_ase - PAIR_MASS_U_FROM_FORMULAS) > 0.1:
        raise ValueError(
            "ASE structure mass does not match the mass derived from formulas: "
            f"{pair_mass_u_ase} versus {PAIR_MASS_U_FROM_FORMULAS}"
        )

    packed.info.update(
        {
            "system": "phenol + N-methylacetamide",
            "pair_count": args.pair_count,
            "molecules_per_species": args.pair_count,
            "count_definition": MOLECULE_COUNT_DEFINITION,
            "charge": 0,
            "construction_density_g_cm3": args.density_g_cm3,
            "packmol_seed": args.seed,
            "packmol_tolerance_a": args.tolerance_a,
            "packmol_precision_a": args.precision_a,
            "nci_system_id": NCI_SYSTEM_ID,
            "nci_scale": NCI_SCALE,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ase_write(extxyz_path, packed, format="extxyz")

    packing_helper_path = (PART_DIR / "aux" / "domain" / "packing.py").resolve()
    base_manifest_sha256 = sha256_file(base_manifest_path)
    base_structure_sha256 = sha256_file(base_structure_path)
    cell_a = np.asarray(packed.cell, dtype=float).tolist()
    cell_lengths_a, volume_a3 = _cell_geometry_from_matrix(cell_a)
    equivalent_cubic_length_a = volume_a3 ** (1.0 / 3.0)
    geometry = {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(cell_lengths_a),
        "minimum_cell_length_a": min(cell_lengths_a),
        "equivalent_cubic_length_a": equivalent_cubic_length_a,
        "volume_a3": volume_a3,
    }
    require_planned_supercell_geometry(
        geometry,
        pair_count=args.pair_count,
        density_g_cm3=args.density_g_cm3,
    )
    manifest = {
        "schema": INPUT_SCHEMA,
        "created_utc": utc_now(),
        "pair_count": args.pair_count,
        "molecules_per_species": args.pair_count,
        "count_definition": MOLECULE_COUNT_DEFINITION,
        "atom_count": len(packed),
        "net_charge_e": 0,
        "cell_geometry": geometry["cell_geometry"],
        "cell_a": cell_a,
        "cell_lengths_a": geometry["cell_lengths_a"],
        "minimum_cell_length_a": geometry["minimum_cell_length_a"],
        "equivalent_cubic_length_a": geometry["equivalent_cubic_length_a"],
        "volume_a3": geometry["volume_a3"],
        "construction_density_g_cm3": args.density_g_cm3,
        "density_from_mass_and_cell_g_cm3": density_from_mass_and_cell,
        "pair_mass_u": pair_mass_u_ase,
        "construction": {
            "method": "balanced_integer_supercell_repeat",
            "base_pair_count": BASE_PAIR_COUNT,
            "repeat_multiplier": args.pair_count // BASE_PAIR_COUNT,
            "repeat_factors_xyz": list(repeat_factors),
            "base_box_manifest": str(base_manifest_path),
            "base_box_manifest_schema": BASE_BOX_SCHEMA,
            "base_box_manifest_sha256": base_manifest_sha256,
            "base_box_structure": str(base_structure_path),
            "base_box_structure_sha256": base_structure_sha256,
            "packmol_rerun": False,
        },
        "packmol": {
            "applied_to": "checked_base_box_only",
            "version": base_packmol_record["version"],
            "seed": base_packmol_record["seed"],
            "tolerance_a": base_packmol_record["tolerance_a"],
            "precision_a": base_packmol_record["precision_a"],
            "periodic_boundary_check": True,
            "periodic_min_distance_a": base_structure_record.get(
                "periodic_min_distance_a"
            ),
            "periodic_min_distance_lower_bound_a": base_structure_record.get(
                "periodic_min_distance_a"
            ),
            "periodic_min_distance_required_a": base_structure_record.get(
                "min_distance_required_a"
            ),
        },
        "source": {
            "nci_subset": base_source_record.get("nci_subset_file"),
            "nci_subset_sha256": NCI_SUBSET_SHA256,
            "nci_system_id": NCI_SYSTEM_ID,
            "nci_scale": NCI_SCALE,
            "fragments": {
                "A": "phenol",
                "B": "N-methylacetamide",
            },
            "packing_helper": str(packing_helper_path),
            "packing_helper_sha256": sha256_file(packing_helper_path),
            "domain_methodology_config": str(DOMAIN_METHODOLOGY_CONFIG_PATH),
            "domain_methodology_config_sha256": sha256_file(
                DOMAIN_METHODOLOGY_CONFIG_PATH
            ),
            "domain_methodology_name": DOMAIN_METHODOLOGY.name,
            "domain_methodology_version": DOMAIN_METHODOLOGY.version,
        },
        "structure": {
            "path": str(extxyz_path),
            "sha256": sha256_file(extxyz_path),
            "format": "extxyz",
            "pbc": [True, True, True],
            "source_atom_id": "0-based stable atom identity",
            "molecule_id": "0-based stable molecule identity",
            "molecule_kind": {"0": "phenol", "1": "N-methylacetamide"},
        },
        "interpretation": (
            "This is an integer supercell of the checked Packmol starting "
            "geometry at a declared construction density. It has not been "
            "equilibrated and is not a density prediction."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_rank_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    if directory.is_dir():
        for path in sorted(directory.glob("rank-*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            row["rank_record_file"] = str(path)
            row["rank_record_sha256"] = sha256_file(path)
            records.append(row)
    return records


def _read_case_result(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"planned case has no result row: {case['case_id']} at {path}"
        )
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"{path} has an unknown result schema")
    if row.get("case_id") != case["case_id"]:
        raise ValueError(f"{path} does not match its planned case")
    if row.get("mode") != case.get("mode"):
        raise ValueError(f"{path} has the wrong planned mode")
    if row.get("measurement_role") != case.get("measurement_role"):
        raise ValueError(f"{path} has the wrong planned measurement role")
    row["result_file"] = str(path)
    row["result_file_sha256"] = sha256_file(path)
    return row


def _capacity_prefix(
    plan: dict[str, Any],
    result_dir: Path,
) -> list[dict[str, Any]]:
    cases = [case for case in plan["capacity_cases"] if int(case["world_size"]) == 1]
    if not cases:
        raise ValueError("capacity selection requires a one-GPU ladder")
    rows: list[dict[str, Any]] = []
    saw_oom = False
    for case in cases:
        path = result_dir / case["result_file"]
        if saw_oom:
            if path.exists():
                raise ValueError(
                    "a measured capacity row exists after the first CUDA OOM: "
                    f"{case['case_id']}"
                )
            continue
        row = _read_case_result(path, case)
        rows.append(row)
        if not bool(row["success"]):
            if row.get("failure", {}).get("is_cuda_oom") is not True:
                raise ValueError(
                    "capacity sweep failed before a genuine CUDA OOM: "
                    f"{case['case_id']}"
                )
            saw_oom = True
    return rows


def _finite_charge_value(
    record: dict[str, Any],
    name: str,
    *,
    context: str,
) -> float:
    value = record.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{context} has an invalid {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} has an invalid {name}") from error
    if not math.isfinite(number):
        raise ValueError(f"{context} has a non-finite {name}")
    return number


def validated_charge_diagnostics(
    record: Any,
    *,
    atom_count: int,
    target_sum_e: float = 0.0,
    context: str = "charge diagnostics",
) -> dict[str, Any]:
    """Validate finite charge metadata without limiting its residual size."""

    if not isinstance(record, dict):
        raise ValueError(f"{context} are missing")
    if atom_count <= 0:
        raise ValueError(f"{context} have an invalid atom count")
    if record.get("available") is not True:
        raise ValueError(f"{context} are unavailable")
    if record.get("finite") is not True:
        raise ValueError(f"{context} report non-finite predicted charges")
    if record.get("dtype") != "float32":
        raise ValueError(f"{context} must describe the float32 PME charge tensor")

    expected_target = float(target_sum_e)
    if not math.isfinite(expected_target):
        raise ValueError(f"{context} have a non-finite expected target")
    observed_target = _finite_charge_value(
        record,
        "target_sum_e",
        context=context,
    )
    if observed_target != expected_target:
        raise ValueError(f"{context} do not match the input total-charge target")

    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
        or math.prod(shape) != atom_count
    ):
        raise ValueError(f"{context} have an inconsistent tensor shape")
    checksum = record.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError(f"{context} have an invalid tensor SHA-256")

    charge_sum = _finite_charge_value(record, "sum_e", context=context)
    residual = _finite_charge_value(record, "residual_e", context=context)
    residual_per_atom = _finite_charge_value(
        record,
        "abs_residual_per_atom",
        context=context,
    )
    sum_abs = _finite_charge_value(record, "sum_abs_e", context=context)
    max_abs = _finite_charge_value(record, "max_abs_e", context=context)
    if not math.isclose(
        residual,
        charge_sum - observed_target,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{context} have an inconsistent charge residual")
    if not math.isclose(
        residual_per_atom,
        abs(residual) / atom_count,
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        raise ValueError(f"{context} have an inconsistent residual per atom")
    magnitude_slack = 1.0e-12 * max(1.0, sum_abs)
    if (
        residual_per_atom < 0.0
        or sum_abs < 0.0
        or max_abs < 0.0
        or sum_abs + magnitude_slack < abs(charge_sum)
        or max_abs > sum_abs + magnitude_slack
    ):
        raise ValueError(f"{context} have inconsistent charge magnitudes")

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
        "sha256": checksum,
    }


def require_fixed_charge_validation_residual(
    diagnostics: dict[str, Any],
    *,
    atom_count: int,
    max_abs_residual_e: float,
) -> None:
    """Apply the strict charge limit only to the 3,200-atom solver check."""

    if atom_count != BASE_ATOM_COUNT:
        raise ValueError(
            "the strict charge-residual limit is reserved for the checked "
            f"{BASE_ATOM_COUNT:,}-atom PME-versus-Ewald validation"
        )
    limit = float(max_abs_residual_e)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("fixed-charge validation has an invalid residual limit")
    if abs(float(diagnostics["residual_e"])) > limit:
        raise ValueError(
            "PME-versus-Ewald fixed-charge validation exceeds the declared "
            "charge-residual limit"
        )


def capacity_charge_diagnostic_records(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize the one-GPU charge diagnostics saved with successful cases."""

    records = []
    for row in rows:
        if not bool(row.get("success")):
            continue
        atom_count = int(row["atom_count"])
        records.append(
            {
                "case_id": str(row["case_id"]),
                "pair_count": int(row["pair_count"]),
                "atom_count": atom_count,
                "charge_diagnostics": validated_charge_diagnostics(
                    row.get("charges"),
                    atom_count=atom_count,
                    target_sum_e=0.0,
                    context=f"{row['case_id']} charge diagnostics",
                ),
            }
        )
    return records


def _selected_input(
    *,
    input_root: Path,
    pair_count: int,
    expected_input: dict[str, Any],
) -> dict[str, Any]:
    directory = input_root / input_directory_name(pair_count)
    structure = directory / "structure.extxyz"
    manifest_path = directory / "manifest.json"
    if not structure.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"selected input is incomplete: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"selected input has unknown schema: {directory}")
    geometry = validated_manifest_cell_geometry(manifest)
    require_planned_supercell_geometry(
        geometry,
        pair_count=pair_count,
        density_g_cm3=float(expected_input["construction_density_g_cm3"]),
    )
    if int(manifest["pair_count"]) != pair_count:
        raise ValueError(f"selected input has wrong pair count: {directory}")
    if (
        int(manifest["molecules_per_species"]) != pair_count
        or manifest["count_definition"] != MOLECULE_COUNT_DEFINITION
        or int(manifest["atom_count"]) != pair_count * ATOMS_PER_PAIR
        or float(manifest["construction_density_g_cm3"])
        != float(expected_input["construction_density_g_cm3"])
        or float(manifest["packmol"]["tolerance_a"])
        != float(expected_input["packmol_tolerance_a"])
        or float(manifest["packmol"].get("precision_a", math.nan))
        != float(expected_input["packmol_precision_a"])
        or int(manifest["packmol"]["seed"]) != int(expected_input["packmol_base_seed"])
        or manifest["packmol"].get("version") != expected_input["packmol_version"]
        or manifest["source"]["nci_subset_sha256"] != NCI_SUBSET_SHA256
        or manifest["construction"].get("method")
        != expected_input["construction_method"]
        or int(manifest["construction"].get("base_pair_count", -1))
        != int(expected_input["base_pair_count"])
        or manifest["construction"].get("base_box_manifest_schema")
        != expected_input["base_box_schema"]
        or manifest["construction"].get("repeat_factors_xyz")
        != list(balanced_repeat_factors(pair_count))
    ):
        raise ValueError(f"selected input settings do not match plan: {directory}")
    if manifest["structure"]["sha256"] != sha256_file(structure):
        raise ValueError(f"selected input checksum mismatch: {directory}")
    return {
        "pair_count": pair_count,
        "molecules_per_species": pair_count,
        "atom_count": pair_count * ATOMS_PER_PAIR,
        "directory": str(directory.resolve()),
        "structure": {
            "path": str(structure.resolve()),
            "sha256": sha256_file(structure),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        **geometry,
    }


def select_capacity(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    result_dir = args.result_dir.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("capacity plan has an unknown schema")
    if {int(case["world_size"]) for case in plan["capacity_cases"]} != {1}:
        raise ValueError("capacity plan must contain only one-GPU cases")

    rows = _capacity_prefix(plan, result_dir)
    if not rows or rows[-1].get("failure", {}).get("is_cuda_oom") is not True:
        raise ValueError(
            "capacity ladder must reach and retain its first genuine CUDA OOM"
        )
    successful = [row for row in rows if bool(row["success"])]
    if not successful:
        raise ValueError("capacity ladder has no successful case before the OOM")

    validation_cases = plan["validation_cases"]
    if len(validation_cases) != 1:
        raise ValueError("capacity plan must contain one electrostatics validation")
    validation = _read_case_result(
        result_dir / validation_cases[0]["result_file"],
        validation_cases[0],
    )
    if validation.get("mode") != "electrostatics-validation":
        raise ValueError("validation result has the wrong mode")
    measured_acceptance = validation["comparison"]["acceptance"]
    if measured_acceptance != plan["validation_acceptance"]:
        raise ValueError(
            "PME-versus-Ewald acceptance limits differ from the predeclared plan"
        )
    if validation["comparison"].get("passed") is not True:
        raise ValueError("PME-versus-Ewald fixed-charge validation did not pass")
    validation_atom_count = int(validation["atom_count"])
    validation_charge_diagnostics = validated_charge_diagnostics(
        validation.get("charges"),
        atom_count=validation_atom_count,
        target_sum_e=0.0,
        context="PME-versus-Ewald charge diagnostics",
    )
    require_fixed_charge_validation_residual(
        validation_charge_diagnostics,
        atom_count=validation_atom_count,
        max_abs_residual_e=float(
            measured_acceptance["absolute_charge_sum_e_max"]
        ),
    )
    validation_settings = validation["settings"]
    validation_geometry = validated_manifest_cell_geometry(
        validation["input"]["manifest"]
    )
    require_planned_supercell_geometry(
        validation_geometry,
        pair_count=int(validation["pair_count"]),
        density_g_cm3=float(plan["input"]["construction_density_g_cm3"]),
    )
    expected_pme = expected_pme_setup(
        cell_lengths_a=validation_geometry["cell_lengths_a"],
        real_space_cutoff_a=float(plan["model"]["pme_cutoff_a"]),
        accuracy=float(plan["model"]["pme_accuracy"]),
        mesh_safety_factor=float(plan["model"]["pme_mesh_safety_factor"]),
    )
    expected_ewald = expected_ewald_reference_setup(
        atom_count=int(validation["atom_count"]),
        volume_a3=float(validation_geometry["volume_a3"]),
        accuracy=float(plan["model"]["ewald_reference_accuracy"]),
    )
    observed_pme = validation_settings["pme"]
    observed_ewald = validation_settings["ewald_reference"]
    if (
        not math.isclose(
            float(observed_pme["real_space_cutoff_a"]),
            expected_pme["real_space_cutoff_a"],
            rel_tol=1.0e-6,
        )
        or not math.isclose(
            float(observed_pme["alpha_a_inverse"]),
            expected_pme["alpha_a_inverse"],
            rel_tol=1.0e-6,
        )
        or list(observed_pme["mesh_dimensions"])
        != expected_pme["mesh_dimensions"]
        or any(
            not math.isclose(float(observed), expected, rel_tol=1.0e-6)
            for observed, expected in zip(
                observed_pme["mesh_spacing_a"],
                expected_pme["mesh_spacing_a"],
                strict=True,
            )
        )
        or float(observed_pme["mesh_safety_factor"])
        != float(plan["model"]["pme_mesh_safety_factor"])
        or observed_pme.get("parameter_rule")
        != plan["model"]["pme_parameter_rule"]
        or int(observed_pme["spline_order"])
        != int(plan["model"]["pme_spline_order"])
        or float(observed_pme["accuracy"])
        != float(plan["model"]["pme_accuracy"])
        or any(
            not math.isclose(
                float(observed_ewald[name]),
                expected_ewald[name],
                rel_tol=1.0e-6,
            )
            for name in (
                "real_space_cutoff_a",
                "reciprocal_space_cutoff_a_inverse",
                "alpha_a_inverse",
                "accuracy",
            )
        )
        or observed_ewald.get("parameter_rule")
        != expected_ewald["parameter_rule"]
        or not math.isclose(
            float(validation_settings["minimum_cell_length_a"]),
            float(validation_geometry["minimum_cell_length_a"]),
            rel_tol=1.0e-6,
        )
        or 2.0 * float(observed_ewald["real_space_cutoff_a"])
        >= float(validation_geometry["minimum_cell_length_a"])
        or validation_settings["compile_model"] is not False
    ):
        raise ValueError("electrostatics solver settings differ from the plan")

    capacity_charge_diagnostics = capacity_charge_diagnostic_records(successful)
    parity_pair_count = int(args.parity_pairs)
    parity_rows = [
        row for row in successful if int(row["pair_count"]) == parity_pair_count
    ]
    if len(parity_rows) != 1:
        raise ValueError(
            f"declared parity input {parity_pair_count} pairs did not complete "
            "on one GPU"
        )
    parity_charge_diagnostics = next(
        record["charge_diagnostics"]
        for record in capacity_charge_diagnostics
        if record["case_id"] == parity_rows[0]["case_id"]
    )
    input_root = args.input_root.resolve()
    parity_reference_input = _selected_input(
        input_root=input_root,
        pair_count=parity_pair_count,
        expected_input=plan["input"],
    )
    ghost_width_a = float(plan["distributed"]["domain_cutoff_a"]) + float(
        plan["distributed"]["domain_skin_a"]
    )
    parity_partition_check = domain_partition_check(
        cell_lengths_a=parity_reference_input["cell_lengths_a"],
        ghost_width_a=ghost_width_a,
    )

    largest_success = max(successful, key=lambda row: int(row["pair_count"]))
    first_oom = rows[-1]
    largest_success_input = _selected_input(
        input_root=input_root,
        pair_count=int(largest_success["pair_count"]),
        expected_input=plan["input"],
    )
    steady_case_id = steady_timing_case_id(
        int(largest_success["pair_count"]),
        1,
    )
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_utc": utc_now(),
        "run_id": plan["run_id"],
        "capacity_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "capacity_result_dir": str(result_dir),
        "capacity_attempted_case_ids": [row["case_id"] for row in rows],
        "capacity_attempted_pair_counts": [int(row["pair_count"]) for row in rows],
        "capacity_charge_diagnostics": capacity_charge_diagnostics,
        "largest_success": {
            "case_id": largest_success["case_id"],
            "result_file": largest_success["result_file"],
            "result_file_sha256": largest_success["result_file_sha256"],
            "input": largest_success_input,
        },
        "steady_timing_case": {
            "case_id": steady_case_id,
            "result_file": result_filename(steady_case_id),
            "mode": "steady-timing",
            "series": "steady_timing",
            "measurement_role": "steady_timing",
            "world_size": 1,
            "pair_count": int(largest_success["pair_count"]),
            "molecules_per_species": int(largest_success["pair_count"]),
            "atom_count": int(largest_success["atom_count"]),
            "rank_grid_policy": "automatic_from_actual_cell",
            "input": largest_success_input,
            "fresh_process_required": True,
            "warmup_count": int(plan["timing"]["steady"]["warmup_count"]),
            "sample_count": int(plan["timing"]["steady"]["sample_count"]),
        },
        "first_cuda_oom": {
            "case_id": first_oom["case_id"],
            "result_file": first_oom["result_file"],
            "result_file_sha256": first_oom["result_file_sha256"],
            "failure_stage": first_oom["failure"]["stage"],
            "input": _selected_input(
                input_root=input_root,
                pair_count=int(first_oom["pair_count"]),
                expected_input=plan["input"],
            ),
        },
        "parity_reference": {
            "case_id": parity_rows[0]["case_id"],
            "result_file": parity_rows[0]["result_file"],
            "result_file_sha256": parity_rows[0]["result_file_sha256"],
            "input": parity_reference_input,
            "acceptance": {
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
                    "abs(delta_energy_eV) / atom_count <= "
                    "tolerance_eV_per_atom"
                ),
                "energy_tolerance_ev_per_atom": (
                    DEFAULT_PARITY_ENERGY_TOL_EV_PER_ATOM
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
                "force_atol_ev_a": DEFAULT_PARITY_FORCE_ATOL_EV_A,
                "force_rtol": DEFAULT_PARITY_FORCE_RTOL,
            },
            "charge_diagnostics": parity_charge_diagnostics,
            "partition_check": parity_partition_check,
        },
        "electrostatics_validation": {
            "case_id": validation["case_id"],
            "result_file": validation["result_file"],
            "result_file_sha256": validation["result_file_sha256"],
            "input_file_sha256": validation["input"]["file_sha256"],
            "charge_diagnostics": validation_charge_diagnostics,
            "charge_sha256": validation["charges"]["sha256"],
            "charge_sum_e": validation["charges"]["sum_e"],
            "pme_energy_ev": validation["pme"]["energy_ev"],
            "pme_force_sha256": validation["pme"]["forces"]["sha256"],
            "ewald_energy_ev": validation["ewald"]["energy_ev"],
            "ewald_force_sha256": validation["ewald"]["forces"]["sha256"],
            "comparison": validation["comparison"],
            "settings": validation["settings"],
        },
        "settings": {
            "model": plan["model"],
            "distributed": plan["distributed"],
            "input": plan["input"],
            "timing": plan["timing"],
            "validation_acceptance": plan["validation_acceptance"],
        },
        "methodology": {
            **plan["methodology"],
            "resolved_values": {
                **plan["methodology"]["resolved_values"],
                "parity_molecules_per_species": parity_pair_count,
            },
        },
        "source": plan["source"],
    }
    atomic_write_json(args.output.resolve(), selection)
    return selection


def derive_distributed_plan(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selection has an unknown schema")
    world_size = int(args.world_size)
    if world_size not in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        raise ValueError(
            "distributed world size must be selected from "
            f"{DEFAULT_DISTRIBUTED_WORLD_SIZES}"
        )

    cases: list[dict[str, Any]] = []
    source = selection["parity_reference"]["input"]
    cases.append(
        {
            "case_id": parity_case_id(source["pair_count"], world_size),
            "mode": "parity",
            "series": "parity",
            "measurement_role": "parity",
            "world_size": world_size,
            "pair_count": source["pair_count"],
            "molecules_per_species": source["pair_count"],
            "atom_count": source["atom_count"],
            "rank_grid_policy": "automatic_from_actual_cell",
            "input": source,
            "fresh_process_required": True,
            "warmup_count": 0,
            "sample_count": 1,
        }
    )
    for series, key, id_builder in (
        ("steady_timing", "largest_success", steady_timing_case_id),
        ("rescue", "first_cuda_oom", rescue_case_id),
    ):
        source = selection[key]["input"]
        steady_timing = series == "steady_timing"
        cases.append(
            {
                "case_id": id_builder(source["pair_count"], world_size),
                "mode": "steady-timing" if steady_timing else "distributed",
                "series": series,
                "measurement_role": series,
                "world_size": world_size,
                "pair_count": source["pair_count"],
                "molecules_per_species": source["pair_count"],
                "atom_count": source["atom_count"],
                "rank_grid_policy": "automatic_from_actual_cell",
                "input": source,
                "fresh_process_required": True,
                "warmup_count": (
                    int(selection["settings"]["timing"]["steady"]["warmup_count"])
                    if steady_timing
                    else 0
                ),
                "sample_count": (
                    int(selection["settings"]["timing"]["steady"]["sample_count"])
                    if steady_timing
                    else 1
                ),
            }
        )
    plan = {
        "schema": DISTRIBUTED_PLAN_SCHEMA,
        "created_utc": utc_now(),
        "run_id": f"{selection['run_id']}-gpus-{world_size:02d}",
        "world_size": world_size,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "cases": cases,
        "parity_acceptance": selection["parity_reference"]["acceptance"],
        "settings": selection["settings"],
        "methodology": selection["methodology"],
        "source": selection["source"],
    }
    atomic_write_json(args.output.resolve(), plan)
    return plan


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(path)


def _failure_csv_fields(row: dict[str, Any]) -> dict[str, Any]:
    failure = row.get("failure", {})
    return {
        "failure_type": str(failure.get("type") or "LaunchFailure"),
        "failure_stage": str(failure.get("stage") or "launcher"),
        "error": str(failure.get("message") or "case process failed"),
    }


def _peak_memory_from_failed_row(row: dict[str, Any]) -> int | str:
    values = [
        int(record["memory"]["max_allocated_bytes"])
        for record in row.get("rank_records", [])
        if isinstance(record.get("memory"), dict)
        and record["memory"].get("max_allocated_bytes") is not None
    ]
    return max(values) if values else ""


def _result_metrics(row: dict[str, Any]) -> dict[str, Any]:
    if not bool(row["success"]):
        return {
            **_failure_csv_fields(row),
            "elapsed_s": "",
            "warmup_count": "",
            "sample_count": "",
            "elapsed_samples_s": "",
            "elapsed_median_s": "",
            "elapsed_q1_s": "",
            "elapsed_q3_s": "",
            "elapsed_iqr_s": "",
            "peak_memory_bytes_max_rank": _peak_memory_from_failed_row(row),
            "energy_ev": "",
            "force_rms_ev_per_a": "",
            "force_max_ev_per_a": "",
        }
    timing = row["timing"]
    elapsed_samples = [float(value) for value in timing["samples_s_max_rank"]]
    forces = row["output"]["forces_source_atom_order"]
    return {
        "failure_type": "",
        "failure_stage": "",
        "error": "",
        "elapsed_s": float(timing["median_s"]),
        "warmup_count": int(timing["warmup_count"]),
        "sample_count": int(timing["sample_count"]),
        "elapsed_samples_s": json.dumps(elapsed_samples, separators=(",", ":")),
        "elapsed_median_s": float(timing["median_s"]),
        "elapsed_q1_s": float(timing["q1_s"]),
        "elapsed_q3_s": float(timing["q3_s"]),
        "elapsed_iqr_s": float(timing["iqr_s"]),
        "peak_memory_bytes_max_rank": int(row["memory"]["max_allocated_bytes"]),
        "energy_ev": float(row["output"]["energy_ev"]),
        "force_rms_ev_per_a": float(forces["rms_ev_a"]),
        "force_max_ev_per_a": float(forces["max_norm_ev_a"]),
    }


def _load_force_array(row: dict[str, Any]) -> Any:
    import numpy as np

    if not bool(row["success"]):
        raise ValueError(f"failed case has no force array: {row['case_id']}")
    record = row["output"]["forces_source_atom_order_npy"]
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"force artifact checksum failed: {row['case_id']}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != list(record["shape"]) or not np.isfinite(array).all():
        raise ValueError(f"force artifact is invalid: {row['case_id']}")
    return array


def _read_derived_results(
    directory: Path,
    selection_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = directory / "derived-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != DISTRIBUTED_PLAN_SCHEMA:
        raise ValueError(f"unknown derived plan schema: {plan_path}")
    if plan["selection"]["sha256"] != selection_sha256:
        raise ValueError(f"derived job used a different selection: {plan_path}")
    result_dir = directory / "results"
    rows = []
    for case in plan["cases"]:
        path = result_dir / result_filename(case["case_id"])
        row = _read_case_result(path, case)
        if row["mode"] != case["mode"]:
            raise ValueError(f"result mode mismatch: {path}")
        if row["input"]["file_sha256"] != case["input"]["structure"]["sha256"]:
            raise ValueError(
                f"distributed case did not reuse the selected input: {path}"
            )
        rows.append(row)
    return plan, rows


def _output_comparison(
    reference: dict[str, Any],
    row: dict[str, Any],
    acceptance: dict[str, Any],
    *,
    require_energy: bool = True,
    require_forces: bool = True,
) -> dict[str, Any]:
    """Compare energy and forces for two runs of the identical structure."""

    import numpy as np

    if not bool(reference.get("success")) or not bool(row.get("success")):
        raise ValueError("output comparison requires two successful runs")
    if row["input"]["file_sha256"] != reference["input"]["file_sha256"]:
        raise ValueError("output comparison used different input structures")
    reference_forces = _load_force_array(reference).astype(np.float64)
    observed_forces = _load_force_array(row).astype(np.float64)
    if observed_forces.shape != reference_forces.shape:
        raise ValueError("output comparison force arrays have different shapes")

    atom_count = int(reference_forces.shape[0])
    if atom_count <= 0:
        raise ValueError("output comparison needs at least one atom")
    difference = observed_forces - reference_forces
    energy_difference = abs(
        float(row["output"]["energy_ev"]) - float(reference["output"]["energy_ev"])
    )
    energy_difference_per_atom = energy_difference / atom_count
    energy_tolerance_per_atom = float(
        acceptance["energy_tolerance_ev_per_atom"]
    )
    component_tolerances = float(acceptance["force_atol_ev_a"]) + float(
        acceptance["force_rtol"]
    ) * np.abs(reference_forces)
    force_rms = float(np.sqrt(np.mean(difference * difference)))
    force_max = float(np.abs(difference).max())
    force_tolerance = float(component_tolerances.max())
    force_passed = bool(np.less_equal(np.abs(difference), component_tolerances).all())
    energy_passed = energy_difference_per_atom <= energy_tolerance_per_atom
    passed = (
        (energy_passed or not require_energy)
        and (force_passed or not require_forces)
    )
    return {
        "passed": passed,
        "energy_required": require_energy,
        "energy_passed": energy_passed,
        "forces_required": require_forces,
        "forces_passed": force_passed,
        "energy_difference_ev": energy_difference,
        "energy_difference_ev_per_atom": energy_difference_per_atom,
        "energy_tolerance_ev_per_atom": energy_tolerance_per_atom,
        "force_rms_difference_ev_per_a": force_rms,
        "force_max_difference_ev_per_a": force_max,
        "force_tolerance_ev_per_a": force_tolerance,
    }


def _parity_comparison(
    selection: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Compare distributed forces with the selected one-GPU reference."""

    reference_path = Path(selection["parity_reference"]["result_file"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema") != RESULT_SCHEMA or not bool(reference.get("success")):
        raise ValueError("the selected one-GPU parity reference is not usable")
    return _output_comparison(
        reference,
        row,
        selection["parity_reference"]["acceptance"],
        require_energy=False,
        require_forces=True,
    )


def _distributed_energy_comparison(
    reference: dict[str, Any],
    row: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Compare energies produced by the same distributed reduction path."""

    return _output_comparison(
        reference,
        row,
        selection["parity_reference"]["acceptance"],
        require_energy=True,
        require_forces=False,
    )


def _steady_timing_output_comparison(
    selection: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Compare a timed multi-GPU output with the timed one-GPU output."""

    reference_path = (
        Path(selection["capacity_result_dir"])
        / selection["steady_timing_case"]["result_file"]
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema") != RESULT_SCHEMA or not bool(reference.get("success")):
        raise ValueError("the one-GPU steady-timing reference is not usable")
    return _output_comparison(
        reference,
        row,
        selection["parity_reference"]["acceptance"],
        require_energy=False,
        require_forces=True,
    )


def _electrostatics_phase_summary(phase_dir: Path) -> dict[str, Any]:
    plan_path = phase_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validation_cases = plan.get("validation_cases", [])
    if len(validation_cases) != 1:
        raise ValueError("the capacity plan must contain one validation case")
    case = validation_cases[0]
    row = _read_case_result(phase_dir / "results" / case["result_file"], case)
    acceptance_matches = row.get("comparison", {}).get("acceptance") == plan.get(
        "validation_acceptance"
    )
    accepted = (
        bool(row.get("success"))
        and acceptance_matches
        and (row.get("comparison", {}).get("passed") is True)
    )
    return {
        "schema": PHASE_SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "phase": "electrostatics-validation",
        "status": "accepted" if accepted else "failed",
        "passed": accepted,
        "publishable": False,
        "checks": {
            "case_succeeded": bool(row.get("success")),
            "acceptance_matches_plan": acceptance_matches,
            "pme_ewald_and_charge_limits_passed": (
                row.get("comparison", {}).get("passed") is True
            ),
        },
        "case_id": row["case_id"],
        "message": (
            "Electrostatics validation passed; the capacity ladder may start."
            if accepted
            else "Electrostatics validation failed; the capacity ladder must not run."
        ),
    }


def _capacity_phase_summary(phase_dir: Path) -> dict[str, Any]:
    plan_path = phase_dir / "plan.json"
    selection_path = phase_dir / "selection.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = _capacity_prefix(plan, phase_dir / "results")
    successful = [row for row in rows if bool(row["success"])]
    reached_oom = bool(rows) and (
        rows[-1].get("failure", {}).get("is_cuda_oom") is True
    )
    accepted_validation = _electrostatics_phase_summary(phase_dir)["passed"]
    selection_matches = (
        selection.get("schema") == SELECTION_SCHEMA
        and selection.get("capacity_plan", {}).get("sha256") == sha256_file(plan_path)
        and selection.get("capacity_attempted_case_ids")
        == [row["case_id"] for row in rows]
    )
    steady_case = selection.get("steady_timing_case", {})
    steady_succeeded = False
    if steady_case:
        steady_row = _read_case_result(
            phase_dir / "results" / str(steady_case["result_file"]),
            steady_case,
        )
        steady_succeeded = bool(steady_row.get("success"))
    passed = (
        bool(successful)
        and reached_oom
        and accepted_validation
        and selection_matches
        and steady_succeeded
    )
    return {
        "schema": PHASE_SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "phase": "capacity",
        "status": "complete" if passed else "failed",
        "passed": passed,
        "publishable": False,
        "checks": {
            "electrostatics_validation_accepted": accepted_validation,
            "successful_capacity_case_found": bool(successful),
            "first_natural_cuda_oom_reached": reached_oom,
            "selection_matches_measured_prefix": selection_matches,
            "one_gpu_steady_timing_succeeded": steady_succeeded,
        },
        "attempted_case_ids": [row["case_id"] for row in rows],
        "largest_success_pair_count": (
            max(int(row["pair_count"]) for row in successful) if successful else None
        ),
        "first_cuda_oom_pair_count": (
            int(rows[-1]["pair_count"]) if reached_oom else None
        ),
        "message": (
            "The capacity phase completed, selected exact inputs, and recorded "
            "the dedicated one-GPU steady timing."
            if passed
            else "The capacity phase is incomplete or failed a required check."
        ),
    }


def _distributed_phase_summary(phase_dir: Path) -> dict[str, Any]:
    plan_path = phase_dir / "derived-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection_path = Path(plan["selection"]["path"])
    if sha256_file(selection_path) != plan["selection"]["sha256"]:
        raise ValueError("the capacity selection changed after this job was planned")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checked_plan, rows = _read_derived_results(
        phase_dir,
        plan["selection"]["sha256"],
    )
    rows_by_series = {
        str(case["series"]): row
        for case, row in zip(checked_plan["cases"], rows, strict=True)
    }
    parity_row = rows_by_series["parity"]
    steady_row = rows_by_series["steady_timing"]
    rescue_row = rows_by_series["rescue"]
    parity_execution_succeeded = bool(parity_row.get("success"))
    parity_comparison: dict[str, Any] | None = None
    if parity_execution_succeeded:
        parity_comparison = _parity_comparison(selection, parity_row)
    parity_passed = bool(
        parity_execution_succeeded
        and parity_comparison is not None
        and parity_comparison["passed"]
    )
    steady_execution_succeeded = bool(steady_row.get("success"))
    steady_output_comparison: dict[str, Any] | None = None
    if steady_execution_succeeded:
        steady_output_comparison = _steady_timing_output_comparison(
            selection,
            steady_row,
        )
    steady_passed = bool(
        steady_execution_succeeded
        and steady_output_comparison is not None
        and steady_output_comparison["passed"]
    )
    rescue_succeeded = bool(rescue_row.get("success"))
    passed = parity_passed and steady_passed
    return {
        "schema": PHASE_SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "phase": "distributed",
        "world_size": int(plan["world_size"]),
        "status": "complete" if passed else "failed",
        "passed": passed,
        "publishable": False,
        "checks": {
            "parity_execution_succeeded": parity_execution_succeeded,
            "one_gpu_force_agreement_passed": parity_passed,
            "steady_timing_case_succeeded": steady_execution_succeeded,
            "steady_timing_force_agreement_passed": steady_passed,
            "distributed_energy_agreement_deferred_to_bundle": True,
            "exact_oom_input_rescued": rescue_succeeded,
        },
        "parity_comparison": parity_comparison,
        "steady_timing_output_comparison": steady_output_comparison,
        "case_ids": {series: row["case_id"] for series, row in rows_by_series.items()},
        "message": (
            "One-GPU force checks passed for the agreement and timing inputs. "
            "Distributed-energy agreement is checked after all GPU counts finish. "
            "The exact OOM retry "
            + (
                "also succeeded."
                if rescue_succeeded
                else "did not fit on this GPU count."
            )
            if passed
            else (
                "The distributed phase failed a required one-GPU force check."
            )
        ),
    }


def write_phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Write one phase result even when a required check fails."""

    phase_dir = args.phase_dir.resolve()
    try:
        if args.phase == "electrostatics":
            summary = _electrostatics_phase_summary(phase_dir)
        elif args.phase == "capacity":
            summary = _capacity_phase_summary(phase_dir)
        elif args.phase == "distributed":
            summary = _distributed_phase_summary(phase_dir)
        else:
            raise AssertionError(f"unhandled phase {args.phase}")
    except Exception as exc:
        summary = {
            "schema": PHASE_SUMMARY_SCHEMA,
            "created_utc": utc_now(),
            "phase": args.phase,
            "status": "failed",
            "passed": False,
            "publishable": False,
            "checks": {},
            "error": f"{type(exc).__name__}: {exc}",
            "message": "The phase could not be accepted.",
        }
    atomic_write_json(args.output.resolve(), summary)
    return summary


def read_verified_sha256sums(path: Path) -> dict[Path, str]:
    """Read a GNU SHA256SUMS file and verify every referenced file now."""

    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checksum file is missing: {manifest_path}")
    records: dict[Path, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", raw_line)
        if match is None:
            raise ValueError(
                f"invalid SHA256SUMS line {line_number} in {manifest_path}"
            )
        digest, path_text = match.groups()
        source = Path(path_text)
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        if source in records:
            raise ValueError(f"duplicate checksum entry for {source}")
        if not source.is_file():
            raise FileNotFoundError(f"checksummed file is missing: {source}")
        if sha256_file(source) != digest:
            raise ValueError(f"checksummed file changed after the job: {source}")
        records[source] = digest
    if not records:
        raise ValueError(f"checksum file is empty: {manifest_path}")
    return records


def _producer_digest_by_name(records: dict[Path, str]) -> dict[str, str]:
    by_name: dict[str, str] = {}
    for path, digest in records.items():
        if path.name in by_name:
            raise ValueError(f"producer file name is not unique: {path.name}")
        by_name[path.name] = digest
    return by_name


def _copy_checked_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"copied file checksum changed: {destination}")


def _portable_reference(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _rewrite_bundle_references(
    value: Any,
    *,
    copied_paths: dict[str, str],
    external_paths: dict[str, str],
) -> Any:
    """Replace host paths in copied JSON records with portable references."""

    if isinstance(value, dict):
        return {
            key: _rewrite_bundle_references(
                item,
                copied_paths=copied_paths,
                external_paths=external_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_bundle_references(
                item,
                copied_paths=copied_paths,
                external_paths=external_paths,
            )
            for item in value
        ]
    if isinstance(value, str):
        if value in copied_paths:
            return copied_paths[value]
        if value in external_paths:
            return external_paths[value]
        is_windows_path = re.match(r"^[A-Za-z]:[\\/]", value) is not None
        if Path(value).is_absolute() or is_windows_path:
            raise ValueError(f"bundle JSON still references a host path: {value}")
    return value


def _copy_job_records(
    *,
    job_directories: dict[str, Path],
    artifact_records: dict[str, dict[Path, str]],
    producer_manifest_paths: dict[str, Path],
    artifact_manifest_paths: dict[str, Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], list[Path]]:
    """Copy all job-time campaign files and both checksum lists."""

    copied_records: list[dict[str, Any]] = []
    copied_paths: dict[str, str] = {}
    output_paths: list[Path] = []
    for label, job_dir in job_directories.items():
        destination_root = output_dir / "job-records" / label
        for source, digest in artifact_records[label].items():
            try:
                relative = source.relative_to(job_dir)
            except ValueError as exc:
                raise ValueError(
                    f"job artifact is outside its result directory: {source}"
                ) from exc
            destination = destination_root / relative
            _copy_checked_file(source, destination, expected_sha256=digest)
            portable = _portable_reference(destination, output_dir)
            copied_paths[str(source)] = portable
            source_parent = source.parent
            destination_parent = destination.parent
            while source_parent != job_dir.parent:
                copied_paths.setdefault(
                    str(source_parent),
                    _portable_reference(destination_parent, output_dir),
                )
                if source_parent == job_dir:
                    break
                source_parent = source_parent.parent
                destination_parent = destination_parent.parent
            output_paths.append(destination)
            copied_records.append(
                {
                    "role": "job-file",
                    "job": label,
                    "file": portable,
                    "sha256": digest,
                    "source_sha256": digest,
                }
            )
        for role, source in (
            ("producer-checksums", producer_manifest_paths[label]),
            ("job-file-checksums", artifact_manifest_paths[label]),
        ):
            digest = sha256_file(source)
            destination = destination_root / source.name
            _copy_checked_file(source, destination, expected_sha256=digest)
            portable = _portable_reference(destination, output_dir)
            copied_paths[str(source.resolve())] = portable
            output_paths.append(destination)
            copied_records.append(
                {
                    "role": role,
                    "job": label,
                    "file": portable,
                    "sha256": digest,
                    "source_sha256": digest,
                }
            )
    return copied_records, copied_paths, output_paths


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    capacity_dir = args.capacity_dir.resolve()
    selection_path = capacity_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("capacity selection has an unknown schema")
    selection_sha = sha256_file(selection_path)

    capacity_plan_path = capacity_dir / "plan.json"
    capacity_plan = json.loads(capacity_plan_path.read_text(encoding="utf-8"))
    capacity_rows = _capacity_prefix(capacity_plan, capacity_dir / "results")
    if capacity_rows[-1].get("failure", {}).get("is_cuda_oom") is not True:
        raise ValueError("capacity phase did not end at its first natural CUDA OOM")
    validation_case = capacity_plan["validation_cases"][0]
    validation_row = _read_case_result(
        capacity_dir / "results" / validation_case["result_file"],
        validation_case,
    )
    if validation_row["comparison"].get("passed") is not True:
        raise ValueError("PME-versus-Ewald validation did not pass")
    expected_capacity_charge_diagnostics = capacity_charge_diagnostic_records(
        capacity_rows
    )
    if (
        selection.get("capacity_charge_diagnostics")
        != expected_capacity_charge_diagnostics
    ):
        raise ValueError(
            "capacity charge diagnostics differ from the selected result rows"
        )
    selected_parity_case_id = str(selection["parity_reference"]["case_id"])
    try:
        expected_parity_charge_diagnostics = next(
            record["charge_diagnostics"]
            for record in expected_capacity_charge_diagnostics
            if record["case_id"] == selected_parity_case_id
        )
    except StopIteration as error:
        raise ValueError(
            "the selected parity case has no capacity charge diagnostics"
        ) from error
    if (
        selection["parity_reference"].get("charge_diagnostics")
        != expected_parity_charge_diagnostics
    ):
        raise ValueError(
            "parity charge diagnostics differ from the selected capacity row"
        )
    validation_charge_diagnostics = validated_charge_diagnostics(
        validation_row.get("charges"),
        atom_count=int(validation_row["atom_count"]),
        target_sum_e=0.0,
        context="PME-versus-Ewald charge diagnostics",
    )
    require_fixed_charge_validation_residual(
        validation_charge_diagnostics,
        atom_count=int(validation_row["atom_count"]),
        max_abs_residual_e=float(
            validation_row["comparison"]["acceptance"][
                "absolute_charge_sum_e_max"
            ]
        ),
    )
    if (
        selection["electrostatics_validation"].get("charge_diagnostics")
        != validation_charge_diagnostics
    ):
        raise ValueError(
            "electrostatics charge diagnostics differ from the selected result"
        )
    steady_timing_case = selection["steady_timing_case"]
    steady_timing_one_gpu = _read_case_result(
        capacity_dir / "results" / steady_timing_case["result_file"],
        steady_timing_case,
    )
    if not bool(steady_timing_one_gpu.get("success")):
        raise ValueError("dedicated one-GPU steady-timing case did not complete")

    derived_by_world: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    distributed_directory_by_world: dict[int, Path] = {}
    for raw_directory in args.distributed_dir:
        directory = raw_directory.resolve()
        plan, rows = _read_derived_results(directory, selection_sha)
        world_size = int(plan["world_size"])
        if world_size in derived_by_world:
            raise ValueError(f"duplicate distributed world size {world_size}")
        derived_by_world[world_size] = (plan, rows)
        distributed_directory_by_world[world_size] = directory
    if set(derived_by_world) != set(DEFAULT_DISTRIBUTED_WORLD_SIZES):
        raise ValueError(
            "final bundle requires every declared multi-GPU job: "
            f"{DEFAULT_DISTRIBUTED_WORLD_SIZES}"
        )

    job_directories = {
        "capacity": capacity_dir,
        **{
            f"distributed-{world_size:02d}": distributed_directory_by_world[world_size]
            for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES
        },
    }
    producer_manifest_paths = {
        label: directory / "producer-SHA256SUMS"
        for label, directory in job_directories.items()
    }
    artifact_manifest_paths = {
        label: directory / "artifact-SHA256SUMS"
        for label, directory in job_directories.items()
    }
    producer_records = {
        label: read_verified_sha256sums(path)
        for label, path in producer_manifest_paths.items()
    }
    artifact_records = {
        label: read_verified_sha256sums(path)
        for label, path in artifact_manifest_paths.items()
    }
    producer_digest_maps = {
        label: _producer_digest_by_name(records)
        for label, records in producer_records.items()
    }
    capacity_producers = producer_digest_maps["capacity"]
    for label, records in producer_digest_maps.items():
        if records != capacity_producers:
            raise ValueError(f"producer files changed between jobs: {label}")
    for path in (
        Path(__file__),
        DOMAIN_METHODOLOGY_CONFIG_PATH,
        *args.producer_file,
    ):
        resolved = path.resolve()
        expected = capacity_producers.get(resolved.name)
        if expected is None or sha256_file(resolved) != expected:
            raise ValueError(
                f"current producer does not match job-time checksums: {resolved}"
            )

    phase_summaries: dict[str, dict[str, Any]] = {}
    for label, directory in job_directories.items():
        summary_path = directory / "phase-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema") != PHASE_SUMMARY_SCHEMA
            or summary.get("passed") is not True
        ):
            raise ValueError(f"campaign phase did not pass: {label}")
        phase_summaries[label] = summary
        checkpoint_report = json.loads(
            (directory / "aimnet-checkpoint-preflight.json").read_text(encoding="utf-8")
        )
        if (
            checkpoint_report.get("schema") != CHECKPOINT_PREFLIGHT_SCHEMA
            or checkpoint_report.get("alias") != AIMNET_CHECKPOINT
            or checkpoint_report.get("sha256") != AIMNET_CHECKPOINT_SHA256
        ):
            raise ValueError(f"AIMNet2 checkpoint preflight changed: {label}")

    distributed_rows = [
        row for _plan, rows in derived_by_world.values() for row in rows
    ]
    distributed_rows_by_case = {
        steady_timing_one_gpu["case_id"]: steady_timing_one_gpu,
        **{row["case_id"]: row for row in distributed_rows},
    }
    if len(distributed_rows_by_case) != len(distributed_rows) + 1:
        raise ValueError("distributed jobs produced duplicate case IDs")
    successful_rescue_world_sizes = _require_complete_distributed_outcomes(
        selection,
        distributed_rows_by_case,
    )
    acceptance = selection["parity_reference"]["acceptance"]
    parity_pair_count = int(selection["parity_reference"]["input"]["pair_count"])
    parity_rows_by_world = {
        world_size: distributed_rows_by_case[
            parity_case_id(parity_pair_count, world_size)
        ]
        for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES
    }
    parity_force_comparisons = {
        world_size: _parity_comparison(selection, row)
        for world_size, row in parity_rows_by_world.items()
    }
    energy_reference_world_size = int(acceptance["energy_reference_world_size"])
    energy_comparison_world_sizes = tuple(
        int(value) for value in acceptance["energy_comparison_world_sizes"]
    )
    parity_energy_reference = parity_rows_by_world[energy_reference_world_size]
    parity_energy_comparisons = {
        world_size: _distributed_energy_comparison(
            parity_energy_reference,
            parity_rows_by_world[world_size],
            selection,
        )
        for world_size in energy_comparison_world_sizes
    }
    if not all(
        comparison["forces_passed"]
        for comparison in parity_force_comparisons.values()
    ):
        raise ValueError(
            "the agreement input fails the declared one- versus multi-GPU "
            "force limits"
        )
    if not all(
        comparison["energy_passed"]
        for comparison in parity_energy_comparisons.values()
    ):
        raise ValueError(
            "the agreement input fails the declared distributed energy limits"
        )

    steady_pair_count = int(selection["largest_success"]["input"]["pair_count"])
    steady_rows_by_world = {
        world_size: distributed_rows_by_case[
            steady_timing_case_id(steady_pair_count, world_size)
        ]
        for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES
    }
    steady_force_comparisons = {
        world_size: _steady_timing_output_comparison(selection, row)
        for world_size, row in steady_rows_by_world.items()
    }
    steady_energy_reference = steady_rows_by_world[energy_reference_world_size]
    steady_energy_comparisons = {
        world_size: _distributed_energy_comparison(
            steady_energy_reference,
            steady_rows_by_world[world_size],
            selection,
        )
        for world_size in energy_comparison_world_sizes
    }
    if not all(
        comparison["forces_passed"]
        for comparison in steady_force_comparisons.values()
    ):
        raise ValueError(
            "the timed input fails the declared one- versus multi-GPU force limits"
        )
    if not all(
        comparison["energy_passed"]
        for comparison in steady_energy_comparisons.values()
    ):
        raise ValueError(
            "the timed input fails the declared distributed energy limits"
        )

    successful_rescue_rows = sorted(
        (
            row
            for row in distributed_rows_by_case.values()
            if row.get("measurement_role") == "rescue" and bool(row.get("success"))
        ),
        key=lambda row: int(row["world_size"]),
    )
    rescue_output_comparisons = [
        {
            "reference_world_size": int(successful_rescue_rows[0]["world_size"]),
            "observed_world_size": int(row["world_size"]),
            **_output_comparison(
                successful_rescue_rows[0],
                row,
                acceptance,
            ),
        }
        for row in successful_rescue_rows[1:]
    ]
    if not all(item["passed"] for item in rescue_output_comparisons):
        raise ValueError(
            "successful retries of the one-GPU OOM input disagree in energy or forces"
        )
    successful_rows = [
        row
        for row in [*capacity_rows, *distributed_rows_by_case.values()]
        if bool(row["success"])
    ]
    source_reference = successful_rows[0]["source"]
    if (
        source_reference.get("toolkit_core_commit") != CORE_COMMIT
        or source_reference.get("toolkit_ops_commit") != OPS_COMMIT
        or source_reference.get("toolkit_version") != "0.2.0"
        or source_reference.get("aimnet_checkpoint_sha256") != AIMNET_CHECKPOINT_SHA256
    ):
        raise ValueError("measured source does not match the required versions")
    if bool(source_reference.get("repository_dirty")):
        raise ValueError("publishable results require a clean tutorial checkout")
    planned_methodology = selection["methodology"]
    planned_methodology_identity = planned_methodology["source_identity"]
    if (
        source_reference.get("domain_methodology_name")
        != planned_methodology_identity["name"]
        or source_reference.get("domain_methodology_version")
        != planned_methodology_identity["version"]
        or source_reference.get("domain_methodology_config_sha256")
        != planned_methodology_identity["sha256"]
        or source_reference.get("domain_methodology_record")
        != planned_methodology["source"]
    ):
        raise ValueError("runner methodology source differs from the campaign plan")
    source_keys = (
        "toolkit_core_commit",
        "toolkit_core_source_root",
        "toolkit_core_source_file",
        "toolkit_core_source_file_sha256",
        "toolkit_ops_commit",
        "toolkit_ops_source_root",
        "toolkit_ops_source_file",
        "toolkit_ops_source_file_sha256",
        "toolkit_version",
        "toolkit_ops_version",
        "repository_commit",
        "repository_tree",
        "repository_branch",
        "repository_required_paths",
        "repository_dirty",
        "runtime_software",
        "runner_sha256",
        "aimnet_checkpoint_sha256",
        "domain_methodology_name",
        "domain_methodology_version",
        "domain_methodology_config_file",
        "domain_methodology_config_sha256",
        "domain_methodology_record",
    )
    if any(
        validation_row["source"].get(key) != source_reference.get(key)
        for key in source_keys
    ):
        raise ValueError("electrostatics validation used a different source")
    for row in successful_rows:
        if any(
            row["source"].get(key) != source_reference.get(key) for key in source_keys
        ):
            raise ValueError(f"source identity changed between jobs: {row['case_id']}")
        expected_model = selection["settings"]["model"]
        observed_model = row["model"]
        observed_pme = observed_model["pme"]
        observed_d3 = observed_model["d3"]
        input_geometry = validated_manifest_cell_geometry(
            row["input"]["manifest"]
        )
        require_planned_supercell_geometry(
            input_geometry,
            pair_count=int(row["pair_count"]),
            density_g_cm3=float(
                selection["settings"]["input"]["construction_density_g_cm3"]
            ),
        )
        expected_pme = expected_pme_setup(
            cell_lengths_a=input_geometry["cell_lengths_a"],
            real_space_cutoff_a=float(expected_model["pme_cutoff_a"]),
            accuracy=float(expected_model["pme_accuracy"]),
            mesh_safety_factor=float(expected_model["pme_mesh_safety_factor"]),
        )
        if (
            float(observed_pme["cutoff_a"]) != float(expected_model["pme_cutoff_a"])
            or not math.isclose(
                float(observed_pme["alpha_a_inverse"]),
                expected_pme["alpha_a_inverse"],
                rel_tol=1.0e-6,
            )
            or list(observed_pme["mesh_dimensions"])
            != expected_pme["mesh_dimensions"]
            or any(
                not math.isclose(float(observed), expected, rel_tol=1.0e-6)
                for observed, expected in zip(
                    observed_pme["mesh_spacing_a"],
                    expected_pme["mesh_spacing_a"],
                    strict=True,
                )
            )
            or float(observed_pme["mesh_safety_factor"])
            != float(expected_model["pme_mesh_safety_factor"])
            or observed_pme.get("parameter_rule")
            != expected_model["pme_parameter_rule"]
            or int(observed_pme["spline_order"])
            != int(expected_model["pme_spline_order"])
            or float(observed_pme["accuracy"])
            != float(expected_model["pme_accuracy"])
            or float(observed_d3["cutoff_a"]) != float(expected_model["d3_cutoff_a"])
            or float(observed_d3["smoothing_fraction"])
            != float(expected_model["d3_smoothing_fraction"])
            or observed_d3["parameter_file_sha256"] != D3_PARAMETER_SHA256
            or observed_model.get("neighbor_adaptation")
            != expected_model["neighbor_adaptation"]
        ):
            raise ValueError(f"model settings changed: {row['case_id']}")
        distributed = row["distributed"]
        world_size = int(row["world_size"])
        validate_recorded_rank_layout(distributed, world_size=world_size)
        if (
            float(distributed["domain_cutoff_a"])
            != float(selection["settings"]["distributed"]["domain_cutoff_a"])
            or float(distributed["domain_skin_a"])
            != float(selection["settings"]["distributed"]["domain_skin_a"])
            or distributed["compile"] is not False
            or distributed.get("grid_dims") is not None
            or (world_size > 1 and distributed["require_nondegenerate"] is not True)
        ):
            raise ValueError(f"domain-decomposition settings changed: {row['case_id']}")

    settings_record = {
        "domain_methodology": {
            "name": selection["methodology"]["source_identity"]["name"],
            "version": selection["methodology"]["source_identity"]["version"],
            "config_sha256": selection["methodology"]["source_identity"]["sha256"],
            "resolved_values": selection["methodology"]["resolved_values"],
        },
        "model_components": [
            "AIMNet2 checkpoint base and predicted charges",
            "PME electrostatics",
            "D3(BJ) dispersion",
        ],
        "precision": "float32",
        "aimnet_checkpoint_sha256": AIMNET_CHECKPOINT_SHA256,
        "d3_parameters_sha256": D3_PARAMETER_SHA256,
        "neighbor_adaptation": selection["settings"]["model"]["neighbor_adaptation"],
        "pme": {
            "cutoff_a": selection["settings"]["model"]["pme_cutoff_a"],
            "mesh_safety_factor": selection["settings"]["model"][
                "pme_mesh_safety_factor"
            ],
            "parameter_rule": selection["settings"]["model"]["pme_parameter_rule"],
            "spline_order": selection["settings"]["model"]["pme_spline_order"],
            "accuracy": selection["settings"]["model"]["pme_accuracy"],
            "hybrid_forces": True,
            "reciprocal_mesh_distribution": "replicated_per_rank",
        },
        "ewald_reference": {
            "accuracy": selection["settings"]["model"][
                "ewald_reference_accuracy"
            ],
            "parameter_rule": "estimate_ewald_parameters(accuracy)",
            "scope": "fixed-charge electrostatics validation only",
        },
        "domain": {
            "api": "DomainParallel",
            "skin_a": selection["settings"]["distributed"]["domain_skin_a"],
            "cutoff_a": selection["settings"]["distributed"]["domain_cutoff_a"],
            "compile": False,
            "require_nondegenerate": True,
            "grid_dims": None,
            "rank_grid_policy": (
                "Toolkit SpatialPartitioner derives cells_per_dim and "
                "rank_grid from each input's actual cell shape and the "
                "domain cutoff"
            ),
            "recorded_layout_fields": ["cells_per_dim", "rank_grid"],
            "halo_counts": "not_exposed_by_public_api",
        },
        "packmol": {
            "version": selection["settings"]["input"]["packmol_version"],
            "construction_density_g_cm3": selection["settings"]["input"][
                "construction_density_g_cm3"
            ],
            "tolerance_a": selection["settings"]["input"]["packmol_tolerance_a"],
            "precision_a": selection["settings"]["input"]["packmol_precision_a"],
            "base_seed": selection["settings"]["input"]["packmol_base_seed"],
            "periodic_boundary_check": True,
        },
        "timing_boundary": selection["settings"]["timing"]["steady"]["boundary"],
        "timing_measurement_kind": selection["settings"]["timing"]["steady"][
            "measurement_kind"
        ],
        "timing_measurement_role": "steady_timing",
        "timing_world_sizes": selection["settings"]["timing"]["steady"]["world_sizes"],
        "timing_warmup_count": selection["settings"]["timing"]["steady"][
            "warmup_count"
        ],
        "timing_sample_count": selection["settings"]["timing"]["steady"][
            "sample_count"
        ],
        "timing_model_evaluations_per_workflow": selection["settings"]["timing"][
            "steady"
        ]["model_evaluations_per_workflow"],
        "timing_one_rank_run_steps": selection["settings"]["timing"]["steady"][
            "one_rank_run_steps"
        ],
        "timing_multi_rank_run_steps": selection["settings"]["timing"]["steady"][
            "multi_rank_run_steps"
        ],
        "timing_summary": selection["settings"]["timing"]["steady"]["summary"],
        "timing_quartile_method": selection["settings"]["timing"]["steady"][
            "quartile_method"
        ],
        "timing_max_relative_iqr": selection["settings"]["timing"]["steady"][
            "max_relative_iqr"
        ],
        "publishable_benchmark": selection["settings"]["timing"][
            "publishable_benchmark"
        ],
        "timing_interpretation": selection["settings"]["timing"]["interpretation"],
        "parity_acceptance": selection["parity_reference"]["acceptance"],
        "force_difference_definition": (
            "RMS and maximum absolute difference over Cartesian components"
        ),
    }
    settings_sha = canonical_json_sha256(settings_record)

    capacity_csv_rows: list[dict[str, Any]] = []
    structure_hashes: dict[int, str] = {}
    for row in capacity_rows:
        atom_count = int(row["atom_count"])
        structure_hash = str(row["input"]["file_sha256"])
        structure_hashes[atom_count] = structure_hash
        capacity_csv_rows.append(
            {
                "case_id": row["case_id"],
                "atom_count": atom_count,
                "molecules_per_species": int(row["pair_count"]),
                "gpus": 1,
                "success": bool(row["success"]),
                "status": row["status"],
                **_result_metrics(row),
                "structure_sha256": structure_hash,
                "settings_sha256": settings_sha,
                "measurement_role": "capacity",
                "measurement_kind": "cold_one_shot_partition_run_gather",
            }
        )

    parity_csv_rows: list[dict[str, Any]] = []
    for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        row = parity_rows_by_world[world_size]
        case_id = str(row["case_id"])
        force_comparison = parity_force_comparisons[world_size]
        energy_comparison = parity_energy_comparisons.get(world_size)
        if energy_comparison is None:
            energy_comparison = _distributed_energy_comparison(
                parity_energy_reference,
                parity_energy_reference,
                selection,
            )
        parity_csv_rows.append(
            {
                "case_id": case_id,
                "atom_count": int(row["atom_count"]),
                "force_reference_gpus": (
                    DOMAIN_METHODOLOGY.force_reference_world_size
                ),
                "energy_reference_gpus": energy_reference_world_size,
                "gpus": world_size,
                "success": True,
                "status": "complete",
                "failure_type": "",
                "failure_stage": "",
                "error": "",
                "one_gpu_energy_abs_offset_ev": force_comparison[
                    "energy_difference_ev"
                ],
                "one_gpu_energy_abs_offset_ev_per_atom": force_comparison[
                    "energy_difference_ev_per_atom"
                ],
                "distributed_energy_difference_ev": energy_comparison[
                    "energy_difference_ev"
                ],
                "distributed_energy_difference_ev_per_atom": energy_comparison[
                    "energy_difference_ev_per_atom"
                ],
                "force_rms_difference_ev_per_a": force_comparison[
                    "force_rms_difference_ev_per_a"
                ],
                "force_max_difference_ev_per_a": force_comparison[
                    "force_max_difference_ev_per_a"
                ],
                "energy_tolerance_ev_per_atom": energy_comparison[
                    "energy_tolerance_ev_per_atom"
                ],
                "force_tolerance_ev_per_a": force_comparison[
                    "force_tolerance_ev_per_a"
                ],
                "distributed_energy_passed": bool(
                    energy_comparison["energy_passed"]
                ),
                "force_passed": bool(force_comparison["forces_passed"]),
                "parity_passed": True,
                "structure_sha256": row["input"]["file_sha256"],
                "settings_sha256": settings_sha,
                "measurement_role": "parity",
                "measurement_kind": "cold_one_shot_partition_run_gather",
            }
        )

    def distributed_csv_row(
        row: dict[str, Any],
        *,
        case_id: str | None = None,
        world_size: int,
    ) -> dict[str, Any]:
        atom_count = int(row["atom_count"])
        structure_hash = str(row["input"]["file_sha256"])
        structure_hashes[atom_count] = structure_hash
        metrics = _result_metrics(row)
        owned_counts = (
            [int(value) for value in row["distributed"]["owned_atom_counts"]]
            if bool(row["success"])
            else [
                int(record["owned_atom_count"])
                for record in row.get("rank_records", [])
                if record.get("owned_atom_count") is not None
            ]
        )
        distributed_record = row.get("distributed", {})
        if (
            isinstance(distributed_record, dict)
            and distributed_record.get("cells_per_dim")
            and distributed_record.get("rank_grid")
        ):
            _cells_per_dim, rank_grid = validate_recorded_rank_layout(
                distributed_record,
                world_size=world_size,
            )
            spatial_grid = "x".join(str(value) for value in rank_grid)
        elif bool(row["success"]):
            raise ValueError(
                f"successful distributed row has no recorded layout: "
                f"{row['case_id']}"
            )
        else:
            spatial_grid = ""
        return {
            "case_id": case_id or row["case_id"],
            "atom_count": atom_count,
            "molecules_per_species": int(row["pair_count"]),
            "nodes": world_size,
            "gpus": world_size,
            "ranks": world_size,
            "success": bool(row["success"]),
            "status": row["status"],
            **metrics,
            "owned_atoms_min_rank": min(owned_counts) if owned_counts else "",
            "owned_atoms_max_rank": max(owned_counts) if owned_counts else "",
            "spatial_grid": spatial_grid,
            "structure_sha256": structure_hash,
            "settings_sha256": settings_sha,
            "measurement_role": row["measurement_role"],
            "measurement_kind": (
                "steady_partition_run_gather"
                if row["measurement_role"] == "steady_timing"
                else "cold_one_shot_partition_run_gather"
            ),
        }

    distributed_csv_rows = [
        distributed_csv_row(
            steady_timing_one_gpu,
            world_size=1,
        )
    ]
    for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        for id_builder in (steady_timing_case_id, rescue_case_id):
            pair_count = (
                int(selection["largest_success"]["input"]["pair_count"])
                if id_builder is steady_timing_case_id
                else int(selection["first_cuda_oom"]["input"]["pair_count"])
            )
            row = distributed_rows_by_case[id_builder(pair_count, world_size)]
            distributed_csv_rows.append(distributed_csv_row(row, world_size=world_size))

    steady_timing_rows = [
        row
        for row in distributed_csv_rows
        if row["measurement_role"] == "steady_timing"
    ]
    unstable_timing_rows = [
        row
        for row in steady_timing_rows
        if (
            float(row["elapsed_iqr_s"]) / float(row["elapsed_median_s"])
            > DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr + 1.0e-12
        )
    ]
    if unstable_timing_rows:
        details = ", ".join(
            f"{int(row['gpus'])} GPU: "
            f"{float(row['elapsed_iqr_s']) / float(row['elapsed_median_s']):.1%}"
            for row in unstable_timing_rows
        )
        raise ValueError(
            "steady timing is too variable to report; relative IQR exceeds "
            f"{DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr:.1%} "
            f"for {details}"
        )

    successful_case_rows = [
        row
        for row in [
            *capacity_rows,
            *distributed_rows_by_case.values(),
        ]
        if bool(row["success"])
    ]
    successful_runtime_rows = [
        runtime
        for row in successful_case_rows
        for runtime in row.get("runtime", [])
        if runtime
    ]
    gpu_names = {str(row["gpu_name"]) for row in successful_runtime_rows}
    gpu_memory = {int(row["gpu_total_memory_bytes"]) for row in successful_runtime_rows}
    cuda_versions = {str(row["torch_cuda_version"]) for row in successful_runtime_rows}
    driver_versions = {
        str(row["driver_version"])
        for row in successful_runtime_rows
        if row.get("driver_version")
    }
    if (
        len(gpu_names) != 1
        or not all("H100" in value for value in gpu_names)
        or len(gpu_memory) != 1
        or len(cuda_versions) != 1
        or len(driver_versions) != 1
    ):
        raise ValueError("H100 hardware identity is missing or inconsistent")
    max_observed_gpus = max(
        len([runtime for runtime in row.get("runtime", []) if runtime])
        for row in successful_case_rows
    )
    max_observed_nodes = max(
        len(
            {
                str(runtime["host"])
                for runtime in row.get("runtime", [])
                if runtime and str(runtime.get("host", "")).strip()
            }
        )
        for row in successful_case_rows
    )
    if max_observed_gpus <= 0 or max_observed_nodes <= 0:
        raise ValueError("successful cases do not contain observed GPU and node counts")

    source_record = source_reference
    producer_files = dict(sorted(capacity_producers.items()))
    runner_name = Path(source_record["runner"]).name
    if producer_files.get(runner_name) != source_record["runner_sha256"]:
        raise ValueError("runner identity does not match job-time producer checksums")
    methodology_config_name = Path(
        source_record["domain_methodology_config_file"]
    ).name
    if (
        producer_files.get(methodology_config_name)
        != source_record["domain_methodology_config_sha256"]
    ):
        raise ValueError(
            "methodology config identity does not match job-time producer checksums"
        )
    runtime_software_record = dict(source_record["runtime_software"])
    python_version = runtime_software_record["python_version"]
    runtime_software_record["python_executable"] = f"external:python@{python_version}"
    runtime_software_record["python_prefix"] = (
        f"external:python-prefix@{python_version}"
    )
    source_identity = {
        "repository_commit": source_record["repository_commit"],
        "repository_tree": source_record["repository_tree"],
        "repository_branch": source_record["repository_branch"],
        "repository_dirty": False,
        "toolkit_commit": source_record["toolkit_core_commit"],
        "toolkit_ops_commit": source_record["toolkit_ops_commit"],
        "toolkit_version": source_record["toolkit_version"],
        "toolkit_ops_version": source_record["toolkit_ops_version"],
        "runtime_software": runtime_software_record,
        "aimnet_checkpoint_sha256": source_record["aimnet_checkpoint_sha256"],
        "domain_methodology": {
            "name": source_record["domain_methodology_name"],
            "version": source_record["domain_methodology_version"],
            "config_sha256": source_record["domain_methodology_config_sha256"],
            "record": source_record["domain_methodology_record"],
            "resolved_values": selection["methodology"]["resolved_values"],
        },
        "producer_files_sha256": producer_files,
    }
    hardware_identity = {
        "site": args.site,
        "site_source": "operator-declared",
        "gpu_model": next(iter(gpu_names)),
        "gpu_memory_bytes": next(iter(gpu_memory)),
        "gpus_available": max_observed_gpus,
        "nodes_available": max_observed_nodes,
        "resource_count_source": "derived from successful per-rank runtime records",
        "driver_version": next(iter(driver_versions)),
        "cuda_version": next(iter(cuda_versions)),
        "interconnect": args.interconnect,
        "interconnect_source": "operator-declared; raw GPU topology is retained",
    }
    input_identity = {
        "structures_sha256_by_atom_count": {
            str(atom_count): digest
            for atom_count, digest in sorted(structure_hashes.items())
        },
        "nci_subset_sha256": NCI_SUBSET_SHA256,
        "molecule_pair": "phenol + N-methylacetamide",
        "construction_density_g_cm3": selection["settings"]["input"][
            "construction_density_g_cm3"
        ],
    }
    identity = {
        "source": source_identity,
        "hardware": hardware_identity,
        "settings": settings_record,
        "inputs": input_identity,
    }
    identity_sha = {
        name: canonical_json_sha256(record) for name, record in identity.items()
    }

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"bundle output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_records, copied_paths, generic_paths = _copy_job_records(
        job_directories=job_directories,
        artifact_records=artifact_records,
        producer_manifest_paths=producer_manifest_paths,
        artifact_manifest_paths=artifact_manifest_paths,
        output_dir=output_dir,
    )

    # Copy report-producing source files and the packaged NCI input. Downloaded
    # model and D3 cache files stay external because their redistribution is
    # reviewed separately from this result bundle.
    external_paths: dict[str, str] = {}
    producer_paths_by_name = {
        path.name: (path, digest)
        for path, digest in producer_records["capacity"].items()
    }
    for name, (source, digest) in producer_paths_by_name.items():
        if digest == D3_PARAMETER_SHA256:
            external_paths[str(source)] = f"external:d3-parameter-cache@sha256:{digest}"
            continue
        if digest == NCI_SUBSET_SHA256:
            destination = output_dir / "source-inputs" / source.name
            role = "nci-input"
        elif source.suffix in {".py", ".sh", ".sbatch"}:
            destination = output_dir / "producers" / source.name
            role = "producer"
        else:
            external_paths[str(source)] = f"external:{name}@sha256:{digest}"
            continue
        _copy_checked_file(source, destination, expected_sha256=digest)
        portable = _portable_reference(destination, output_dir)
        copied_paths[str(source)] = portable
        generic_paths.append(destination)
        generic_records.append(
            {
                "role": role,
                "file": portable,
                "sha256": digest,
                "source_sha256": digest,
            }
        )

    embedded_manifest = validation_row["input"].get("manifest")
    if isinstance(embedded_manifest, dict):
        packing_source = embedded_manifest.get("source", {})
        packing_helper_text = packing_source.get("packing_helper")
        packing_helper_sha = packing_source.get("packing_helper_sha256")
        if (
            packing_helper_text
            and packing_helper_sha
            and str(Path(str(packing_helper_text)).resolve()) not in copied_paths
        ):
            packing_helper = Path(str(packing_helper_text)).resolve()
            destination = output_dir / "producers" / "domain-packing.py"
            _copy_checked_file(
                packing_helper,
                destination,
                expected_sha256=str(packing_helper_sha),
            )
            portable = _portable_reference(destination, output_dir)
            copied_paths[str(packing_helper)] = portable
            generic_paths.append(destination)
            generic_records.append(
                {
                    "role": "producer",
                    "file": portable,
                    "sha256": str(packing_helper_sha),
                    "source_sha256": str(packing_helper_sha),
                }
            )

    all_result_rows = [
        validation_row,
        *capacity_rows,
        *distributed_rows_by_case.values(),
    ]
    for row in all_result_rows:
        source = row.get("source") or {}
        external_values = {
            source.get("repository_root"): (
                f"external:tutorial-checkout@{source.get('repository_commit')}"
            ),
            source.get("toolkit_core_source_root"): (
                f"external:toolkit-core@{source.get('toolkit_core_commit')}"
            ),
            source.get("toolkit_core_source_file"): (
                f"external:toolkit-core-source@sha256:"
                f"{source.get('toolkit_core_source_file_sha256')}"
            ),
            source.get("toolkit_ops_source_root"): (
                f"external:toolkit-ops@{source.get('toolkit_ops_commit')}"
            ),
            source.get("toolkit_ops_source_file"): (
                f"external:toolkit-ops-source@sha256:"
                f"{source.get('toolkit_ops_source_file_sha256')}"
            ),
            source.get("aimnet_checkpoint"): (
                f"external:{AIMNET_CHECKPOINT}@sha256:{AIMNET_CHECKPOINT_SHA256}"
            ),
        }
        checkpoint_file = source.get("aimnet_checkpoint_file")
        if isinstance(checkpoint_file, dict):
            external_values[checkpoint_file.get("path")] = (
                f"external:{AIMNET_CHECKPOINT}@sha256:{AIMNET_CHECKPOINT_SHA256}"
            )
        model = row.get("model") or {}
        d3 = model.get("d3") or {}
        external_values[d3.get("parameter_file")] = (
            f"external:d3-parameter-cache@sha256:{D3_PARAMETER_SHA256}"
        )
        parameter_identity = d3.get("parameter_file_identity")
        if isinstance(parameter_identity, dict):
            external_values[parameter_identity.get("path")] = (
                f"external:d3-parameter-cache@sha256:{D3_PARAMETER_SHA256}"
            )
        manifest_record = row.get("input", {}).get("manifest")
        if isinstance(manifest_record, dict):
            executable = manifest_record.get("packmol", {}).get("executable")
            external_values[executable] = f"external:packmol@{EXPECTED_PACKMOL_VERSION}"
        runtime_records = [source.get("runtime_software"), *(row.get("runtime") or [])]
        for rank_record in row.get("rank_records", []):
            rank_source = rank_record.get("source") or {}
            external_values.update(
                {
                    rank_source.get("repository_root"): (
                        f"external:tutorial-checkout@"
                        f"{rank_source.get('repository_commit')}"
                    ),
                    rank_source.get("toolkit_core_source_root"): (
                        f"external:toolkit-core@"
                        f"{rank_source.get('toolkit_core_commit')}"
                    ),
                    rank_source.get("toolkit_core_source_file"): (
                        f"external:toolkit-core-source@sha256:"
                        f"{rank_source.get('toolkit_core_source_file_sha256')}"
                    ),
                    rank_source.get("toolkit_ops_source_root"): (
                        f"external:toolkit-ops@{rank_source.get('toolkit_ops_commit')}"
                    ),
                    rank_source.get("toolkit_ops_source_file"): (
                        f"external:toolkit-ops-source@sha256:"
                        f"{rank_source.get('toolkit_ops_source_file_sha256')}"
                    ),
                    rank_source.get("aimnet_checkpoint"): (
                        f"external:{AIMNET_CHECKPOINT}@sha256:"
                        f"{AIMNET_CHECKPOINT_SHA256}"
                    ),
                }
            )
            rank_checkpoint = rank_source.get("aimnet_checkpoint_file")
            if isinstance(rank_checkpoint, dict):
                external_values[rank_checkpoint.get("path")] = (
                    f"external:{AIMNET_CHECKPOINT}@sha256:{AIMNET_CHECKPOINT_SHA256}"
                )
            runtime_records.extend(
                [rank_source.get("runtime_software"), rank_record.get("runtime")]
            )
        for runtime in runtime_records:
            if not isinstance(runtime, dict):
                continue
            external_values[runtime.get("python_executable")] = (
                f"external:python@{runtime.get('python_version')}"
            )
            external_values[runtime.get("python_prefix")] = (
                f"external:python-prefix@{runtime.get('python_version')}"
            )
        external_paths.update(
            {
                str(path): replacement
                for path, replacement in external_values.items()
                if path
            }
        )

    portable_plan_sources = {
        "capacity-plan.json": capacity_plan,
        "selection.json": selection,
        **{
            f"distributed-{world_size:02d}-plan.json": derived_by_world[world_size][0]
            for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES
        },
    }
    portable_plan_files: dict[str, str] = {}
    portable_plan_digests: dict[str, str] = {}
    for filename, value in portable_plan_sources.items():
        portable_value = _rewrite_bundle_references(
            value,
            copied_paths=copied_paths,
            external_paths=external_paths,
        )
        destination = output_dir / "plans" / filename
        atomic_write_json(destination, portable_value)
        digest = sha256_file(destination)
        portable = _portable_reference(destination, output_dir)
        portable_plan_files[filename] = portable
        portable_plan_digests[filename] = digest
        generic_paths.append(destination)
        generic_records.append(
            {
                "role": "portable-plan",
                "file": portable,
                "sha256": digest,
                "source_sha256": None,
            }
        )

    structure_sources = (
        (
            "validation",
            int(validation_row["pair_count"]),
            Path(validation_row["input"]["path"]).resolve(),
            str(validation_row["input"]["file_sha256"]),
        ),
        (
            "parity",
            int(selection["parity_reference"]["input"]["pair_count"]),
            Path(selection["parity_reference"]["input"]["structure"]["path"]).resolve(),
            str(selection["parity_reference"]["input"]["structure"]["sha256"]),
        ),
        (
            "largest-success",
            int(selection["largest_success"]["input"]["pair_count"]),
            Path(selection["largest_success"]["input"]["structure"]["path"]).resolve(),
            str(selection["largest_success"]["input"]["structure"]["sha256"]),
        ),
        (
            "first-cuda-oom",
            int(selection["first_cuda_oom"]["input"]["pair_count"]),
            Path(selection["first_cuda_oom"]["input"]["structure"]["path"]).resolve(),
            str(selection["first_cuda_oom"]["input"]["structure"]["sha256"]),
        ),
    )
    for role, _pair_count, source_path, expected_sha in structure_sources:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"selected {role} structure is missing: {source_path}"
            )
        if sha256_file(source_path) != expected_sha:
            raise ValueError(f"selected {role} structure checksum does not match")

    log_sources: list[tuple[dict[str, Any], Path]] = [
        (validation_row, capacity_dir),
        *((row, capacity_dir) for row in capacity_rows),
        (steady_timing_one_gpu, capacity_dir),
    ]
    for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        directory = distributed_directory_by_world[world_size]
        plan, rows = derived_by_world[world_size]
        if int(plan["world_size"]) != world_size:
            raise ValueError("distributed result directory order changed")
        log_sources.extend((row, directory) for row in rows)
    log_sources_by_case: dict[str, Path] = {}
    for row, directory in log_sources:
        case_id = str(row["case_id"])
        if case_id in log_sources_by_case:
            raise ValueError(f"duplicate case log requested: {case_id}")
        source_path = (directory / "logs" / f"{case_id}.log").resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"case log is missing: {source_path}")
        case_log = row.get("case_log")
        if (
            isinstance(case_log, dict)
            and case_log.get("sha256") is not None
            and sha256_file(source_path) != case_log["sha256"]
        ):
            raise ValueError(f"failed-case log checksum changed: {case_id}")
        log_sources_by_case[case_id] = source_path

    table_rows = {
        "capacity": (CAPACITY_COLUMNS, capacity_csv_rows),
        "parity": (PARITY_COLUMNS, parity_csv_rows),
        "distributed": (DISTRIBUTED_COLUMNS, distributed_csv_rows),
    }
    data_manifest: dict[str, Any] = {}
    for name, (columns, rows) in table_rows.items():
        filename = f"{name}.csv"
        path = output_dir / filename
        _write_csv(path, columns, rows)
        data_manifest[name] = {
            "file": filename,
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "columns": list(columns),
            "planned_case_ids": [str(row["case_id"]) for row in rows],
        }

    raw_rows = [validation_row, *capacity_rows, steady_timing_one_gpu]
    for world_size in DEFAULT_DISTRIBUTED_WORLD_SIZES:
        raw_rows.extend(derived_by_world[world_size][1])
    raw_rows = [
        _rewrite_bundle_references(
            row,
            copied_paths=copied_paths,
            external_paths=external_paths,
        )
        for row in raw_rows
    ]
    raw_path = output_dir / "raw-results.jsonl"
    atomic_write_jsonl(raw_path, raw_rows)

    structures_dir = output_dir / "structures"
    structures_dir.mkdir()
    structure_records: list[dict[str, Any]] = []
    structure_paths: list[Path] = []
    for role, pair_count, source_path, expected_sha in structure_sources:
        destination = structures_dir / f"{role}-pairs-{pair_count:06d}.extxyz"
        shutil.copy2(source_path, destination)
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(f"copied {role} structure checksum changed")
        structure_paths.append(destination)
        structure_records.append(
            {
                "role": role,
                "pair_count": pair_count,
                "molecules_per_species": pair_count,
                "file": destination.relative_to(output_dir).as_posix(),
                "sha256": expected_sha,
            }
        )

    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    log_records: list[dict[str, Any]] = []
    log_paths: list[Path] = []
    for case_id, source_path in sorted(log_sources_by_case.items()):
        destination = logs_dir / f"{case_id}.log"
        shutil.copy2(source_path, destination)
        digest = sha256_file(destination)
        log_paths.append(destination)
        log_records.append(
            {
                "case_id": case_id,
                "file": destination.relative_to(output_dir).as_posix(),
                "sha256": digest,
            }
        )

    campaign_summary_path = output_dir / "campaign-summary.json"
    campaign_summary = {
        "schema": PHASE_SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "phase": "campaign",
        "status": "complete",
        "passed": True,
        "tutorial_result_set_ready": True,
        "publishable_benchmark": False,
        "checks": {
            "capacity_phase_passed": True,
            "distributed_phase_gpu_counts": list(DEFAULT_DISTRIBUTED_WORLD_SIZES),
            "agreement_input_forces_match_one_gpu": True,
            "agreement_input_distributed_energies_match_two_gpu": True,
            "timed_input_forces_match_one_gpu": True,
            "timed_input_distributed_energies_match_two_gpu": True,
            "one_gpu_to_distributed_energy_offsets_recorded_as_diagnostics": True,
            "exact_oom_input_rescued": True,
            "successful_rescue_outputs_agree": (
                all(item["passed"] for item in rescue_output_comparisons)
                if rescue_output_comparisons
                else None
            ),
            "rescue_output_comparison_count": len(rescue_output_comparisons),
        },
        "successful_rescue_gpu_counts": list(successful_rescue_world_sizes),
        "rescue_output_comparisons": rescue_output_comparisons,
        "message": (
            "All required phases passed, and at least one multi-GPU job "
            "completed the exact input that exhausted one GPU."
        ),
    }
    atomic_write_json(campaign_summary_path, campaign_summary)
    campaign_summary_digest = sha256_file(campaign_summary_path)
    campaign_summary_portable = _portable_reference(
        campaign_summary_path,
        output_dir,
    )
    generic_paths.append(campaign_summary_path)
    generic_records.append(
        {
            "role": "campaign-summary",
            "file": campaign_summary_portable,
            "sha256": campaign_summary_digest,
            "source_sha256": None,
        }
    )

    electrostatics = selection["electrostatics_validation"]
    comparison = electrostatics["comparison"]
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": utc_now(),
        "status": "complete",
        "failure_policy": {
            "failed_rows_retained": True,
            "estimates_allowed": False,
            "capacity_stop_condition": "first_single_gpu_oom",
        },
        "identity": identity,
        "identity_sha256": identity_sha,
        "electrostatics_validation": {
            "status": "passed",
            "measurement_kind": "measured",
            "fixed_charges": True,
            "atom_count": int(validation_row["atom_count"]),
            "structure_sha256": validation_row["input"]["file_sha256"],
            "charge_diagnostics": validation_charge_diagnostics,
            "charge_sum_e": electrostatics["charge_sum_e"],
            "charge_sum_tolerance_e": DEFAULT_CHARGE_SUM_TOL_E,
            "pme_energy_ev": electrostatics["pme_energy_ev"],
            "ewald_energy_ev": electrostatics["ewald_energy_ev"],
            "energy_abs_difference_ev_per_atom": comparison[
                "absolute_energy_difference_ev_per_atom"
            ],
            "energy_tolerance_ev_per_atom": (DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM),
            "force_rms_difference_ev_per_a": comparison["force_difference_rms_ev_a"],
            "force_max_difference_ev_per_a": comparison[
                "force_difference_max_norm_ev_a"
            ],
            "force_tolerance_ev_per_a": DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A,
            "charge_sha256": electrostatics["charge_sha256"],
            "pme_force_sha256": electrostatics["pme_force_sha256"],
            "ewald_force_sha256": electrostatics["ewald_force_sha256"],
            "result_file_sha256": electrostatics["result_file_sha256"],
        },
        "selection": {
            "file": portable_plan_files["selection.json"],
            "sha256": portable_plan_digests["selection.json"],
            "job_file_sha256": selection_sha,
            "largest_success_pair_count": selection["largest_success"]["input"][
                "pair_count"
            ],
            "first_cuda_oom_pair_count": selection["first_cuda_oom"]["input"][
                "pair_count"
            ],
            "parity_pair_count": selection["parity_reference"]["input"]["pair_count"],
            "capacity_charge_diagnostics": selection[
                "capacity_charge_diagnostics"
            ],
            "parity_charge_diagnostics": selection["parity_reference"][
                "charge_diagnostics"
            ],
            "successful_rescue_gpu_counts": list(successful_rescue_world_sizes),
        },
        "data": data_manifest,
        "raw_results": {
            "file": raw_path.name,
            "sha256": sha256_file(raw_path),
            "row_count": len(raw_rows),
        },
        "artifacts": {
            "structures": structure_records,
            "case_logs": log_records,
            "files": generic_records,
        },
        "campaign_summary": {
            "file": campaign_summary_portable,
            "sha256": campaign_summary_digest,
        },
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    checksum_paths = [
        manifest_path,
        raw_path,
        *(output_dir / f"{name}.csv" for name in table_rows),
        *structure_paths,
        *log_paths,
        *generic_paths,
    ]
    if len({path.resolve() for path in checksum_paths}) != len(checksum_paths):
        raise RuntimeError("a bundle file was declared more than once")
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    return manifest


def record_failure(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"result already exists: {output}")
    log_path = args.case_log.resolve()
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    rank_records = _load_rank_records(args.rank_output_dir.resolve())
    combined = (
        log_text
        + "\n"
        + "\n".join(json.dumps(row, sort_keys=True) for row in rank_records)
    )
    is_cuda_oom = any(pattern.search(combined) for pattern in CUDA_OOM_PATTERNS)
    stages = sorted(
        {
            str(row.get("failure", {}).get("stage"))
            for row in rank_records
            if row.get("failure", {}).get("stage")
        }
    )
    input_path = args.input_extxyz.resolve()
    input_sha = sha256_file(input_path) if input_path.is_file() else None
    row = {
        "schema": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "run_id": args.run_id,
        "case_id": args.case_id,
        "mode": args.mode,
        "measurement_role": measurement_role_for_mode(args.mode),
        "status": "failed",
        "success": False,
        "world_size": args.world_size,
        "pair_count": args.pair_count,
        "molecules_per_species": args.pair_count,
        "atom_count": args.pair_count * ATOMS_PER_PAIR,
        "input": {
            "path": str(input_path),
            "file_sha256": input_sha,
        },
        "failure": {
            "type": "CUDAOutOfMemoryError" if is_cuda_oom else "LaunchFailure",
            "is_cuda_oom": is_cuda_oom,
            "stage": (
                stages[0]
                if len(stages) == 1
                else "multiple_ranks"
                if stages
                else "launcher"
            ),
            "rank_stages": stages,
            "exit_code": args.exit_code,
            "message": (
                "A genuine CUDA allocation failure was found in the case log "
                "or rank records."
                if is_cuda_oom
                else "The fresh case process exited unsuccessfully; no CUDA "
                "OOM signature was found."
            ),
        },
        "case_log": {
            "path": str(log_path),
            "sha256": sha256_file(log_path) if log_path.is_file() else None,
        },
        "rank_records": rank_records,
        "charges": {
            "available": False,
            "reason": (
                "This failed case returned no charge output. Successful one-GPU "
                "capacity rows record the actual float32 charge diagnostics; "
                "the strict residual limit applies only to the separate "
                "3,200-atom PME-versus-Ewald validation."
            ),
        },
        "halo_counts": {
            "values": None,
            "reason": "not_exposed_by_public_api",
        },
    }
    atomic_write_json(output, row)
    return row


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan has an unknown schema")
    result_dir = args.result_dir.resolve()
    validation_rows = [
        _read_case_result(result_dir / case["result_file"], case)
        for case in plan["validation_cases"]
    ]
    capacity_rows = _capacity_prefix(plan, result_dir)
    rows = [*validation_rows, *capacity_rows]

    atomic_write_jsonl(args.output_jsonl.resolve(), rows)
    capacity = capacity_rows
    single_gpu_ooms = sorted(
        int(row["pair_count"])
        for row in capacity
        if int(row["world_size"]) == 1
        and row.get("failure", {}).get("is_cuda_oom") is True
    )
    summary = {
        "schema": COLLECTION_SCHEMA,
        "created_utc": utc_now(),
        "run_id": plan["run_id"],
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "rows_jsonl": {
            "path": str(args.output_jsonl.resolve()),
            "sha256": sha256_file(args.output_jsonl.resolve()),
        },
        "declared_capacity_candidates": len(plan["capacity_cases"]),
        "attempted_capacity_rows": len(capacity_rows),
        "planned_rows": len(rows),
        "recorded_rows": len(rows),
        "successful_rows": sum(bool(row["success"]) for row in rows),
        "failed_rows": sum(not bool(row["success"]) for row in rows),
        "capacity_rows": len(capacity),
        "electrostatics_validation_rows": len(rows) - len(capacity),
        "first_measured_single_gpu_cuda_oom_pair_count": (
            single_gpu_ooms[0] if single_gpu_ooms else None
        ),
        "rows": rows,
    }
    atomic_write_json(args.output_summary.resolve(), summary)
    return summary


def _add_shared_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--world-sizes",
        nargs="+",
        default=[str(value) for value in DEFAULT_WORLD_SIZES],
    )
    parser.add_argument(
        "--capacity-pair-counts",
        nargs="+",
        default=[str(value) for value in DEFAULT_CAPACITY_PAIR_COUNTS],
    )
    parser.add_argument(
        "--validation-pairs", type=int, default=DEFAULT_VALIDATION_PAIRS
    )
    parser.add_argument("--density-g-cm3", type=float, default=DEFAULT_DENSITY_G_CM3)
    parser.add_argument("--pme-cutoff-a", type=float, default=DEFAULT_PME_CUTOFF_A)
    parser.add_argument(
        "--pme-mesh-safety-factor",
        type=float,
        default=DEFAULT_PME_MESH_SAFETY_FACTOR,
        help="Multiplier used by Toolkit-Ops' PME mesh estimator.",
    )
    parser.add_argument(
        "--pme-spline-order",
        type=int,
        default=DEFAULT_PME_SPLINE_ORDER,
    )
    parser.add_argument(
        "--pme-accuracy",
        type=float,
        default=DEFAULT_PME_ACCURACY,
    )
    parser.add_argument(
        "--ewald-reference-accuracy",
        type=float,
        default=DEFAULT_EWALD_REFERENCE_ACCURACY,
    )
    parser.add_argument("--d3-cutoff-a", type=float, default=DEFAULT_D3_CUTOFF_A)
    parser.add_argument(
        "--d3-smoothing-fraction",
        type=float,
        default=DEFAULT_D3_SMOOTHING_FRACTION,
    )
    parser.add_argument(
        "--domain-skin-a",
        type=float,
        default=DEFAULT_DOMAIN_SKIN_A,
    )
    parser.add_argument(
        "--packmol-tolerance-a",
        type=float,
        default=DEFAULT_PACKMOL_TOLERANCE_A,
    )
    parser.add_argument(
        "--packmol-precision-a",
        type=float,
        default=DEFAULT_PACKMOL_PRECISION_A,
    )
    parser.add_argument("--packmol-seed", type=int, default=DEFAULT_PACKMOL_SEED)
    parser.add_argument(
        "--steady-timing-warmup-count",
        type=int,
        default=DEFAULT_STEADY_TIMING_WARMUP_COUNT,
    )
    parser.add_argument(
        "--steady-timing-sample-count",
        type=int,
        default=DEFAULT_STEADY_TIMING_SAMPLE_COUNT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and prepare the Part 1 domain-decomposition run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Print or save the full run plan without GPU or Toolkit imports.",
    )
    _add_shared_plan_arguments(plan_parser)
    plan_parser.add_argument("--output", type=Path)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Repeat the checked 3,200-atom periodic base box.",
    )
    prepare_parser.add_argument("--pair-count", type=int, required=True)
    prepare_parser.add_argument(
        "--density-g-cm3",
        type=float,
        default=DEFAULT_DENSITY_G_CM3,
    )
    prepare_parser.add_argument(
        "--tolerance-a",
        type=float,
        default=DEFAULT_PACKMOL_TOLERANCE_A,
    )
    prepare_parser.add_argument(
        "--precision-a",
        type=float,
        default=DEFAULT_PACKMOL_PRECISION_A,
    )
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_PACKMOL_SEED)
    prepare_parser.add_argument(
        "--base-box-dir",
        type=Path,
        default=DEFAULT_BASE_BOX_DIR,
        help="Directory containing the checked base manifest and structure.",
    )
    prepare_parser.add_argument(
        "--packmol",
        default=None,
        help="Accepted for older launchers; Packmol is not run by this command.",
    )
    prepare_parser.add_argument(
        "--nci-data",
        type=Path,
        help="Optional packaged NCI file whose checksum is verified.",
    )
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--reuse-existing", action="store_true")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint-preflight",
        help="Resolve and hash the exact AIMNet2 checkpoint before GPU cases.",
    )
    checkpoint_parser.add_argument("--checkpoint", default=AIMNET_CHECKPOINT)
    checkpoint_parser.add_argument("--output", type=Path, required=True)

    failure_parser = subparsers.add_parser(
        "record-failure",
        help="Convert one failed fresh process into a retained result row.",
    )
    failure_parser.add_argument("--run-id", required=True)
    failure_parser.add_argument("--case-id", required=True)
    failure_parser.add_argument(
        "--mode",
        choices=(
            "capacity",
            "parity",
            "distributed",
            "steady-timing",
            "electrostatics-validation",
        ),
        required=True,
    )
    failure_parser.add_argument("--world-size", type=int, required=True)
    failure_parser.add_argument("--pair-count", type=int, required=True)
    failure_parser.add_argument("--input-extxyz", type=Path, required=True)
    failure_parser.add_argument("--rank-output-dir", type=Path, required=True)
    failure_parser.add_argument("--case-log", type=Path, required=True)
    failure_parser.add_argument("--exit-code", type=int, required=True)
    failure_parser.add_argument("--output", type=Path, required=True)

    select_parser = subparsers.add_parser(
        "select",
        help=(
            "Validate the one-GPU prefix and select parity, steady-timing, and "
            "rescue inputs after the first natural CUDA OOM."
        ),
    )
    select_parser.add_argument("--plan", type=Path, required=True)
    select_parser.add_argument("--result-dir", type=Path, required=True)
    select_parser.add_argument("--input-root", type=Path, required=True)
    select_parser.add_argument(
        "--parity-pairs",
        type=int,
        default=DEFAULT_PARITY_PAIR_COUNT,
    )
    select_parser.add_argument("--output", type=Path, required=True)

    derive_parser = subparsers.add_parser(
        "derive",
        help="Create one 2- or 4-GPU plan from the checked capacity selection.",
    )
    derive_parser.add_argument("--selection", type=Path, required=True)
    derive_parser.add_argument("--world-size", type=int, required=True)
    derive_parser.add_argument("--output", type=Path, required=True)

    phase_parser = subparsers.add_parser(
        "phase-summary",
        help="Check one completed campaign phase and write a plain JSON summary.",
    )
    phase_parser.add_argument(
        "--phase",
        choices=("electrostatics", "capacity", "distributed"),
        required=True,
    )
    phase_parser.add_argument("--phase-dir", type=Path, required=True)
    phase_parser.add_argument("--output", type=Path, required=True)

    bundle_parser = subparsers.add_parser(
        "bundle",
        help=(
            "Combine checked capacity/one-GPU timing and 2/4-GPU jobs into the exact "
            "notebook-facing recorded bundle."
        ),
    )
    bundle_parser.add_argument("--capacity-dir", type=Path, required=True)
    bundle_parser.add_argument(
        "--distributed-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for the 2- and 4-GPU result directories.",
    )
    bundle_parser.add_argument("--site", required=True)
    bundle_parser.add_argument("--interconnect", required=True)
    bundle_parser.add_argument(
        "--producer-file",
        type=Path,
        action="append",
        default=[],
    )
    bundle_parser.add_argument("--output-dir", type=Path, required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help=(
            "Combine validation and the attempted one-GPU prefix, ending at "
            "the first natural CUDA OOM."
        ),
    )
    assemble_parser.add_argument("--plan", type=Path, required=True)
    assemble_parser.add_argument("--result-dir", type=Path, required=True)
    assemble_parser.add_argument("--output-jsonl", type=Path, required=True)
    assemble_parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        plan = build_plan(
            run_id=args.run_id,
            world_sizes=parse_positive_ints(args.world_sizes),
            capacity_pair_counts=parse_positive_ints(args.capacity_pair_counts),
            validation_pairs=args.validation_pairs,
            density_g_cm3=args.density_g_cm3,
            pme_cutoff_a=args.pme_cutoff_a,
            pme_mesh_safety_factor=args.pme_mesh_safety_factor,
            pme_spline_order=args.pme_spline_order,
            pme_accuracy=args.pme_accuracy,
            ewald_reference_accuracy=args.ewald_reference_accuracy,
            d3_cutoff_a=args.d3_cutoff_a,
            d3_smoothing_fraction=args.d3_smoothing_fraction,
            domain_skin_a=args.domain_skin_a,
            packmol_tolerance_a=args.packmol_tolerance_a,
            packmol_precision_a=args.packmol_precision_a,
            packmol_seed=args.packmol_seed,
            steady_timing_warmup_count=args.steady_timing_warmup_count,
            steady_timing_sample_count=args.steady_timing_sample_count,
        )
        text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            atomic_write_json(args.output.resolve(), plan)
        print(text, end="")
        return 0
    if args.command == "prepare":
        manifest = prepare_input(args)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "checkpoint-preflight":
        report = checkpoint_preflight(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "record-failure":
        row = record_failure(args)
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        selection = select_capacity(args)
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    if args.command == "derive":
        plan = derive_distributed_plan(args)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.command == "phase-summary":
        summary = write_phase_summary(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if bool(summary["passed"]) else 1
    if args.command == "bundle":
        manifest = build_bundle(args)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "assemble":
        summary = assemble(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
