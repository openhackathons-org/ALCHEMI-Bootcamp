#!/usr/bin/env python3
"""Prepare and combine the short Part 1 multi-GPU evaluation.

The recorded example uses one fixed 51,200-atom molecular box on 1, 2, and
4 H100 GPUs. Each job prepares the same deterministic integer supercell,
performs one untimed initialization pass, then measures three fixed-structure
energy-and-force passes through Toolkit ``DomainParallel``.

There is no size search and no deliberate out-of-memory run. Failures are
recorded because cluster jobs can fail, but a complete lesson bundle requires
successful results from all three GPU counts.
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
import shutil
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402


DOMAIN_METHODOLOGY_CONFIG_PATH = PART_DIR / "aux" / "domain" / "config.py"
PLAN_SCHEMA = "alchemi.part1-domain-plan.v4"
INPUT_SCHEMA = "alchemi.part1-domain-input.v3"
BASE_BOX_SCHEMA = "alchemi.part1-domain-base-box.v1"
BASE_BOX_METHODOLOGY = {
    "schema": DOMAIN_METHODOLOGY.schema,
    "name": DOMAIN_METHODOLOGY.name,
    "version": DOMAIN_METHODOLOGY.version,
}
RESULT_SCHEMA = "alchemi.part1-domain-case.v5"
COLLECTION_SCHEMA = "alchemi.part1-domain-collection.v5"
BUNDLE_SCHEMA = "alchemi.domain-decomposition-lesson.v5"
PHASE_SUMMARY_SCHEMA = "alchemi.part1-domain-phase-summary.v2"
CHECKPOINT_PREFLIGHT_SCHEMA = "alchemi.part1-domain-checkpoint-preflight.v1"

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
ATOMS_PER_PAIR = DOMAIN_METHODOLOGY.atoms_per_composition_unit
BASE_ATOM_COUNT = BASE_PAIR_COUNT * ATOMS_PER_PAIR
PAIR_MASS_U_FROM_FORMULAS = 167.208
MOLECULE_COUNT_DEFINITION = (
    "The count is the number of independent phenol molecules and the equal "
    "number of independent N-methylacetamide molecules; it is not a count "
    "of pre-bound dimers."
)

DEFAULT_WORLD_SIZES = tuple(DOMAIN_METHODOLOGY.campaign_world_sizes)
DEFAULT_FIXED_PAIR_COUNT = DOMAIN_METHODOLOGY.fixed_molecules_per_species
DEFAULT_VALIDATION_PAIRS = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
)
DEFAULT_WARMUP_COUNT = DOMAIN_METHODOLOGY.evaluation_warmup_count
DEFAULT_PASS_COUNT = DOMAIN_METHODOLOGY.evaluation_pass_count
DEFAULT_DENSITY_G_CM3 = DOMAIN_METHODOLOGY.construction_density_g_cm3
DEFAULT_PACKMOL_TOLERANCE_A = DOMAIN_METHODOLOGY.packmol_tolerance_a
DEFAULT_PACKMOL_PRECISION_A = DOMAIN_METHODOLOGY.packmol_precision_a
DEFAULT_PACKMOL_SEED = DOMAIN_METHODOLOGY.packmol_seed
EXPECTED_PACKMOL_VERSION = "21.2.1"
DEFAULT_BASE_BOX_DIR = PART_DIR / "data" / "domain_decomposition" / "prebuilt_base_box"

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
DEFAULT_DISTRIBUTED_ENERGY_REPEATABILITY_TOL_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
)
DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
)
DEFAULT_EVALUATION_FORCE_ATOL_EV_A = DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a
DEFAULT_EVALUATION_FORCE_RTOL = DOMAIN_METHODOLOGY.evaluation_force_rtol
DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A = (
    DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
)
DEFAULT_D3_CUTOFF_A = DOMAIN_METHODOLOGY.d3_cutoff_a
DEFAULT_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction
DEFAULT_DOMAIN_SKIN_A = DOMAIN_METHODOLOGY.domain_halo_skin_a

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            unknown = set(row) - set(columns)
            if unknown:
                raise ValueError(
                    "CSV row contains unknown fields: " + ", ".join(sorted(unknown))
                )
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(path)


def methodology_source_identity() -> dict[str, Any]:
    return {
        "schema": DOMAIN_METHODOLOGY.schema,
        "name": DOMAIN_METHODOLOGY.name,
        "version": DOMAIN_METHODOLOGY.version,
        "path": str(DOMAIN_METHODOLOGY_CONFIG_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(DOMAIN_METHODOLOGY_CONFIG_PATH),
    }


def resolved_methodology_values() -> dict[str, Any]:
    return DOMAIN_METHODOLOGY.resolved_values(json_compatible=True)


def checkpoint_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from importlib import metadata

    from aimnet.calculators.model_registry import get_model_path

    checkpoint = Path(get_model_path(args.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AIMNet2 checkpoint is missing: {checkpoint}")
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256 != AIMNET_CHECKPOINT_SHA256:
        raise ValueError(
            "AIMNet2 checkpoint SHA-256 does not match the declared value: "
            f"{observed_sha256}"
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
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0.0:
        raise ValueError("density_g_cm3 must be positive and finite")
    volume_a3 = pair_count * pair_mass_u * 1.66053906660 / density_g_cm3
    return volume_a3 ** (1.0 / 3.0)


def balanced_repeat_factors(pair_count: int) -> tuple[int, int, int]:
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
    repeat_factors = balanced_repeat_factors(pair_count)
    base_length_a = equivalent_cubic_length_angstrom(
        BASE_PAIR_COUNT,
        density_g_cm3,
    )
    lengths = tuple(base_length_a * repeat_factor for repeat_factor in repeat_factors)
    volume_a3 = math.prod(lengths)
    return {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(lengths),
        "minimum_cell_length_a": min(lengths),
        "equivalent_cubic_length_a": volume_a3 ** (1.0 / 3.0),
        "volume_a3": volume_a3,
    }


def _cell_geometry_from_matrix(
    cell_a: Any,
) -> tuple[tuple[float, float, float], float]:
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
    lengths = tuple(
        math.sqrt(sum(component * component for component in vector))
        for vector in matrix
    )
    if any(length <= 0.0 for length in lengths):
        raise ValueError("cell vectors must have positive lengths")
    for first in range(3):
        for second in range(first + 1, 3):
            dot = sum(matrix[first][axis] * matrix[second][axis] for axis in range(3))
            if not math.isclose(
                dot,
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-10 * lengths[first] * lengths[second],
            ):
                raise ValueError("the domain example requires an orthorhombic cell")
    volume_a3 = abs(
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if volume_a3 <= 0.0:
        raise ValueError("cell_a must have a positive volume")
    return lengths, volume_a3


def validated_manifest_cell_geometry(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("input manifest has an unknown schema")
    required = (
        "cell_geometry",
        "cell_a",
        "cell_lengths_a",
        "minimum_cell_length_a",
        "equivalent_cubic_length_a",
        "volume_a3",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(
            "input manifest is missing cell geometry: " + ", ".join(missing)
        )
    if manifest["cell_geometry"] != "orthorhombic":
        raise ValueError("the domain example requires an orthorhombic cell")
    lengths, volume_a3 = _cell_geometry_from_matrix(manifest["cell_a"])
    observed = (
        *(float(value) for value in manifest["cell_lengths_a"]),
        float(manifest["minimum_cell_length_a"]),
        float(manifest["equivalent_cubic_length_a"]),
        float(manifest["volume_a3"]),
    )
    expected = (
        *lengths,
        min(lengths),
        volume_a3 ** (1.0 / 3.0),
        volume_a3,
    )
    if len(observed) != len(expected) or any(
        not math.isclose(
            observed_value,
            expected_value,
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        )
        for observed_value, expected_value in zip(
            observed,
            expected,
            strict=True,
        )
    ):
        raise ValueError("input manifest cell geometry is internally inconsistent")
    return {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(lengths),
        "minimum_cell_length_a": min(lengths),
        "equivalent_cubic_length_a": volume_a3 ** (1.0 / 3.0),
        "volume_a3": volume_a3,
    }


def require_planned_supercell_geometry(
    geometry: Mapping[str, Any],
    *,
    pair_count: int,
    density_g_cm3: float,
) -> None:
    expected = planned_supercell_geometry(pair_count, density_g_cm3)
    if geometry.get("cell_geometry") != "orthorhombic":
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


def expected_pme_setup(
    *,
    cell_lengths_a: Iterable[float],
    real_space_cutoff_a: float,
    accuracy: float,
    mesh_safety_factor: float,
) -> dict[str, Any]:
    lengths = tuple(float(value) for value in cell_lengths_a)
    if len(lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in lengths
    ):
        raise ValueError("cell_lengths_a must contain three positive finite values")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            real_space_cutoff_a,
            accuracy,
            mesh_safety_factor,
        )
    ):
        raise ValueError("PME estimator inputs must be positive and finite")
    alpha = math.sqrt(-math.log(accuracy)) / real_space_cutoff_a
    raw_mesh = tuple(
        mesh_safety_factor * 2.0 * alpha * length / (3.0 * accuracy**0.2)
        for length in lengths
    )
    mesh = tuple(int(2 ** math.ceil(math.log2(dimension))) for dimension in raw_mesh)
    return {
        "real_space_cutoff_a": real_space_cutoff_a,
        "alpha_a_inverse": alpha,
        "mesh_dimensions": list(mesh),
        "mesh_spacing_a": [
            length / dimension for length, dimension in zip(lengths, mesh, strict=True)
        ],
        "accuracy": accuracy,
        "mesh_safety_factor": mesh_safety_factor,
        "parameter_rule": (
            "estimate_pme_parameters(accuracy, real_space_cutoff, mesh_safety_factor)"
        ),
    }


def expected_ewald_reference_setup(
    *,
    atom_count: int,
    volume_a3: float,
    accuracy: float,
) -> dict[str, Any]:
    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if (
        not math.isfinite(volume_a3)
        or volume_a3 <= 0.0
        or not math.isfinite(accuracy)
        or not 0.0 < accuracy < 1.0
    ):
        raise ValueError("Ewald estimator inputs are invalid")
    eta = (volume_a3**2 / atom_count) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)
    error_factor = math.sqrt(-2.0 * math.log(accuracy))
    return {
        "real_space_cutoff_a": error_factor * eta,
        "reciprocal_space_cutoff_a_inverse": error_factor / eta,
        "alpha_a_inverse": 1.0 / (math.sqrt(2.0) * eta),
        "accuracy": accuracy,
        "parameter_rule": "estimate_ewald_parameters(accuracy)",
    }


def validate_recorded_rank_layout(
    distributed: Mapping[str, Any],
    *,
    world_size: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
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
        or any(
            ranks > cells or cells % ranks
            for ranks, cells in zip(
                rank_grid,
                cells_per_dim,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "recorded cells_per_dim and rank_grid do not match the world size"
        )
    return cells_per_dim, rank_grid


def input_directory_name(pair_count: int) -> str:
    return f"phenol-nma-pairs-{pair_count:06d}"


def fixed_case_id(pair_count: int, world_size: int) -> str:
    return f"fixed-evaluation-pairs-{pair_count:06d}-gpus-{world_size:02d}"


def validation_case_id(pair_count: int) -> str:
    return f"electrostatics-validation-pairs-{pair_count:06d}-gpus-01"


def result_filename(case_id: str) -> str:
    return f"{case_id}.json"


def build_plan(
    *,
    run_id: str,
    tutorial_commit: str,
    world_size: int,
) -> dict[str, Any]:
    if not run_id or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id may contain only letters, numbers, '-' and '_'")
    if not (
        len(tutorial_commit) == 40
        and all(character in "0123456789abcdef" for character in tutorial_commit)
    ):
        raise ValueError("tutorial_commit must be a full lowercase Git SHA")
    if world_size not in DEFAULT_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {DEFAULT_WORLD_SIZES}; got {world_size}"
        )

    pair_count = DEFAULT_FIXED_PAIR_COUNT
    geometry = planned_supercell_geometry(
        pair_count,
        DEFAULT_DENSITY_G_CM3,
    )
    case_id = fixed_case_id(pair_count, world_size)
    fixed_case = {
        "case_id": case_id,
        "mode": "distributed",
        "measurement_role": "fixed_evaluation",
        "world_size": world_size,
        "pair_count": pair_count,
        "molecules_per_species": pair_count,
        "atom_count": pair_count * ATOMS_PER_PAIR,
        **geometry,
        "repeat_factors_xyz": list(balanced_repeat_factors(pair_count)),
        "input_directory": input_directory_name(pair_count),
        "result_file": result_filename(case_id),
    }
    validation_cases: list[dict[str, Any]] = []
    if world_size == 1:
        validation_pairs = DEFAULT_VALIDATION_PAIRS
        validation_id = validation_case_id(validation_pairs)
        validation_cases.append(
            {
                "case_id": validation_id,
                "mode": "electrostatics-validation",
                "measurement_role": "electrostatics_validation",
                "world_size": 1,
                "pair_count": validation_pairs,
                "molecules_per_species": validation_pairs,
                "atom_count": validation_pairs * ATOMS_PER_PAIR,
                **planned_supercell_geometry(
                    validation_pairs,
                    DEFAULT_DENSITY_G_CM3,
                ),
                "repeat_factors_xyz": list(balanced_repeat_factors(validation_pairs)),
                "input_directory": input_directory_name(validation_pairs),
                "result_file": result_filename(validation_id),
                "comparison": (
                    "PME and direct Ewald on the same geometry and "
                    "AIMNet2-predicted charges"
                ),
            }
        )

    methodology_identity = methodology_source_identity()
    return {
        "schema": PLAN_SCHEMA,
        "created_utc": utc_now(),
        "run_id": run_id,
        "source": {
            "tutorial_commit": tutorial_commit,
            "toolkit_core_commit": CORE_COMMIT,
            "toolkit_ops_commit": OPS_COMMIT,
            "nci_subset_sha256": NCI_SUBSET_SHA256,
            "aimnet_checkpoint": AIMNET_CHECKPOINT,
            "aimnet_checkpoint_sha256": AIMNET_CHECKPOINT_SHA256,
            "d3_parameter_sha256": D3_PARAMETER_SHA256,
            "domain_methodology_config": methodology_identity,
        },
        "methodology": {
            "source": DOMAIN_METHODOLOGY.as_record(),
            "source_identity": methodology_identity,
            "resolved_values": resolved_methodology_values(),
        },
        "input": {
            "molecules": ["phenol", "N-methylacetamide"],
            "nci_system_id": NCI_SYSTEM_ID,
            "nci_scale": NCI_SCALE,
            "stoichiometry": "1:1",
            "count_definition": MOLECULE_COUNT_DEFINITION,
            "atoms_per_pair": ATOMS_PER_PAIR,
            "construction_density_g_cm3": DEFAULT_DENSITY_G_CM3,
            "construction_method": "balanced_integer_supercell_repeat",
            "base_box_schema": BASE_BOX_SCHEMA,
            "base_pair_count": BASE_PAIR_COUNT,
            "base_atom_count": BASE_ATOM_COUNT,
        },
        "model": {
            "aimnet_checkpoint": AIMNET_CHECKPOINT,
            "aimnet_compile_model": False,
            "pme_cutoff_a": DEFAULT_PME_CUTOFF_A,
            "pme_mesh_safety_factor": DEFAULT_PME_MESH_SAFETY_FACTOR,
            "pme_spline_order": DEFAULT_PME_SPLINE_ORDER,
            "pme_accuracy": DEFAULT_PME_ACCURACY,
            "ewald_reference_accuracy": DEFAULT_EWALD_REFERENCE_ACCURACY,
            "d3_cutoff_a": DEFAULT_D3_CUTOFF_A,
            "d3_smoothing_fraction": DEFAULT_D3_SMOOTHING_FRACTION,
            "d3_parameters": "read from AIMNet2 checkpoint metadata",
            "neighbor_adaptation": "never",
            "position_invariance": {
                "method": "maximum_minimum_image_displacement",
                "tolerance_a": (DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A),
            },
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
            "world_size": world_size,
            "mesh_dim_names": ["domain"],
            "grid_dims": DOMAIN_METHODOLOGY.domain_grid_dims,
            "domain_cutoff_a": max(
                DEFAULT_PME_CUTOFF_A,
                DEFAULT_D3_CUTOFF_A,
                DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a,
            ),
            "domain_skin_a": DEFAULT_DOMAIN_SKIN_A,
            "compile": False,
            "require_nondegenerate": world_size > 1,
        },
        "timing": {
            "measurement_kind": "fixed_structure_energy_force_pass",
            "warmup_count": DEFAULT_WARMUP_COUNT,
            "pass_count": DEFAULT_PASS_COUNT,
            "measured_model_evaluations_per_pass": (
                DOMAIN_METHODOLOGY.measured_model_evaluations_per_pass
            ),
            "multi_rank_warmup_force_prime_evaluations": (
                DOMAIN_METHODOLOGY.domain_parallel_multi_rank_warmup_force_prime_evaluations
                if world_size > 1
                else 0
            ),
            "timed_work": (
                "Create one DomainParallel context and partition once. Run one "
                "untimed initialization pass. For each of three measured passes, "
                "synchronize ranks and CUDA, time one public run(n_steps=1), then "
                "synchronize CUDA. Gather once after the measured passes."
            ),
            "publishable_benchmark": False,
            "interpretation": (
                "The three raw pass times and their median describe this short "
                "fixed-input example. They are not a general scaling benchmark."
            ),
        },
        "validation_acceptance": {
            "pme_ewald_energy_difference_ev_per_atom_max": (
                DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM
            ),
            "pme_ewald_force_difference_max_norm_ev_a_max": (
                DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A
            ),
            "absolute_charge_sum_e_max": DEFAULT_CHARGE_SUM_TOL_E,
            "distributed_energy_repeatability_span_ev_per_atom_max": (
                DEFAULT_DISTRIBUTED_ENERGY_REPEATABILITY_TOL_EV_PER_ATOM
            ),
            "distributed_energy_agreement_abs_difference_ev_per_atom_max": (
                DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM
            ),
            "evaluation_force_atol_ev_a": (DEFAULT_EVALUATION_FORCE_ATOL_EV_A),
            "evaluation_force_rtol": DEFAULT_EVALUATION_FORCE_RTOL,
            "evaluation_position_mic_tolerance_a": (
                DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
            ),
        },
        "fixed_case": fixed_case,
        "validation_cases": validation_cases,
        "planned_case_count": 1 + len(validation_cases),
    }


def prepare_input(args: argparse.Namespace) -> dict[str, Any]:
    """Build one deterministic integer supercell from the checked base box."""

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
        require_planned_supercell_geometry(
            validated_manifest_cell_geometry(manifest),
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
            f"checked base box needs manifest.json and structure.extxyz: {base_dir}"
        )

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != BASE_BOX_SCHEMA:
        raise ValueError("base box manifest has an unknown schema")
    structure_record = base_manifest.get("structure", {})
    source_record = base_manifest.get("source", {})
    packmol_record = source_record.get("packmol", {})
    methodology_record = base_manifest.get("methodology", {})
    if (
        int(structure_record.get("molecules_per_species", -1)) != BASE_PAIR_COUNT
        or int(structure_record.get("atom_count", -1)) != BASE_ATOM_COUNT
        or structure_record.get("sha256") != sha256_file(base_structure_path)
        or source_record.get("molecule_counts")
        != {
            "phenol": BASE_PAIR_COUNT,
            "N-methylacetamide": BASE_PAIR_COUNT,
        }
    ):
        raise ValueError("base box does not match the declared input")
    if (
        float(structure_record["construction_density_g_cm3"])
        != float(args.density_g_cm3)
        or float(packmol_record["tolerance_a"]) != float(args.tolerance_a)
        or float(packmol_record["precision_a"]) != float(args.precision_a)
        or int(packmol_record["seed"]) != int(args.seed)
        or packmol_record.get("version") != EXPECTED_PACKMOL_VERSION
        or source_record["nci_subset_sha256"] != NCI_SUBSET_SHA256
        or str(source_record.get("nci_system_id")) != str(NCI_SYSTEM_ID)
        or float(source_record.get("nci_scale")) != float(NCI_SCALE)
        or methodology_record != BASE_BOX_METHODOLOGY
    ):
        raise ValueError("base box was built with different construction settings")
    if args.nci_data is not None:
        nci_data = args.nci_data.resolve()
        if not nci_data.is_file() or sha256_file(nci_data) != NCI_SUBSET_SHA256:
            raise ValueError("NCI data file does not match the checked base box")

    base_atoms = ase_read(base_structure_path, format="extxyz")
    if len(base_atoms) != BASE_ATOM_COUNT:
        raise ValueError("base box structure has the wrong atom count")
    for name, expected in structure_record.get("arrays", {}).items():
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
    fractional_before = np.asarray(
        packed.get_scaled_positions(wrap=False),
        dtype=float,
    )
    outside_before = int(
        np.any(
            (fractional_before < 0.0) | (fractional_before >= 1.0),
            axis=1,
        ).sum()
    )
    packed.wrap(eps=0.0)
    fractional_after = np.asarray(
        packed.get_scaled_positions(wrap=False),
        dtype=float,
    )
    outside_after = int(
        np.any(
            (fractional_after < 0.0) | (fractional_after >= 1.0),
            axis=1,
        ).sum()
    )
    if outside_after != 0:
        raise RuntimeError(
            "expanded structure was not wrapped into the primary periodic cell"
        )
    density = (
        float(np.sum(packed.get_masses())) * 1.66053906660 / float(packed.get_volume())
    )
    if not math.isclose(
        density,
        args.density_g_cm3,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("expanded box density does not match the checked base box")
    pair_mass_u = float(np.sum(packed.get_masses())) / args.pair_count
    if abs(pair_mass_u - PAIR_MASS_U_FROM_FORMULAS) > 0.1:
        raise ValueError("ASE structure mass does not match the molecular formulas")

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

    cell_a = np.asarray(packed.cell, dtype=float).tolist()
    lengths, volume_a3 = _cell_geometry_from_matrix(cell_a)
    geometry = {
        "cell_geometry": "orthorhombic",
        "cell_lengths_a": list(lengths),
        "minimum_cell_length_a": min(lengths),
        "equivalent_cubic_length_a": volume_a3 ** (1.0 / 3.0),
        "volume_a3": volume_a3,
    }
    require_planned_supercell_geometry(
        geometry,
        pair_count=args.pair_count,
        density_g_cm3=args.density_g_cm3,
    )
    packing_helper = PART_DIR / "aux" / "domain" / "packing.py"
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
        "density_from_mass_and_cell_g_cm3": density,
        "pair_mass_u": pair_mass_u,
        "construction": {
            "method": "balanced_integer_supercell_repeat",
            "base_pair_count": BASE_PAIR_COUNT,
            "repeat_multiplier": args.pair_count // BASE_PAIR_COUNT,
            "repeat_factors_xyz": list(repeat_factors),
            "periodic_coordinate_canonicalization": {
                "method": "ase.Atoms.wrap",
                "eps": 0.0,
                "fractional_interval": "[0, 1)",
                "atoms_outside_before": outside_before,
                "atoms_outside_after": outside_after,
            },
            "base_box_manifest": str(base_manifest_path),
            "base_box_manifest_schema": BASE_BOX_SCHEMA,
            "base_box_manifest_sha256": sha256_file(base_manifest_path),
            "base_box_structure": str(base_structure_path),
            "base_box_structure_sha256": sha256_file(base_structure_path),
            "packmol_rerun": False,
        },
        "packmol": {
            "applied_to": "checked_base_box_only",
            "version": packmol_record["version"],
            "seed": packmol_record["seed"],
            "tolerance_a": packmol_record["tolerance_a"],
            "precision_a": packmol_record["precision_a"],
            "periodic_boundary_check": True,
            "periodic_min_distance_a": structure_record.get("periodic_min_distance_a"),
            "periodic_min_distance_required_a": structure_record.get(
                "min_distance_required_a"
            ),
        },
        "source": {
            "nci_subset": source_record.get("nci_subset_file"),
            "nci_subset_sha256": NCI_SUBSET_SHA256,
            "nci_system_id": NCI_SYSTEM_ID,
            "nci_scale": NCI_SCALE,
            "fragments": {
                "A": "phenol",
                "B": "N-methylacetamide",
            },
            "packing_helper": str(packing_helper.resolve()),
            "packing_helper_sha256": sha256_file(packing_helper),
            "domain_methodology_config": str(DOMAIN_METHODOLOGY_CONFIG_PATH.resolve()),
            "domain_methodology_config_sha256": sha256_file(
                DOMAIN_METHODOLOGY_CONFIG_PATH
            ),
            "domain_methodology_name": DOMAIN_METHODOLOGY.name,
            "domain_methodology_version": DOMAIN_METHODOLOGY.version,
            "base_box_methodology": BASE_BOX_METHODOLOGY,
        },
        "structure": {
            "path": str(extxyz_path),
            "sha256": sha256_file(extxyz_path),
            "format": "extxyz",
            "pbc": [True, True, True],
            "source_atom_id": "0-based stable atom identity",
            "molecule_id": "0-based stable molecule identity",
            "molecule_kind": {
                "0": "phenol",
                "1": "N-methylacetamide",
            },
        },
        "interpretation": (
            "This is an integer supercell of the checked Packmol starting "
            "geometry at a declared construction density. It has not been "
            "equilibrated and is not a density prediction."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _read_case_result(
    path: Path,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    row = _load_json(path)
    if row.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"{path} has an unknown result schema")
    for key in ("case_id", "mode", "measurement_role"):
        if row.get(key) != case.get(key):
            raise ValueError(f"{path} has the wrong planned {key}")
    row["result_file"] = str(path)
    row["result_file_sha256"] = sha256_file(path)
    return row


def _planned_cases(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [plan["fixed_case"], *plan.get("validation_cases", [])]


def _validate_fixed_result(
    row: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    case = plan["fixed_case"]
    if not bool(row.get("success")):
        raise ValueError(
            f"{case['case_id']} failed: {row.get('error', 'unknown error')}"
        )
    if int(row["world_size"]) != int(case["world_size"]):
        raise ValueError("fixed result has the wrong world size")
    if int(row["atom_count"]) != int(case["atom_count"]):
        raise ValueError("fixed result has the wrong atom count")
    timing = row.get("timing", {})
    warmup_count = int(timing.get("warmup_count", -1))
    pass_count = int(timing.get("measured_pass_count", -1))
    samples = timing.get("pass_times_s", [])
    if (
        warmup_count != DEFAULT_WARMUP_COUNT
        or pass_count != DEFAULT_PASS_COUNT
        or timing.get("partition_count") != 1
        or timing.get("gather_count") != 1
        or timing.get("requested_steps_per_pass") != 1
        or timing.get("measured_model_evaluations_per_pass") != 1
        or not isinstance(samples, list)
        or len(samples) != DEFAULT_PASS_COUNT
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in samples
        )
    ):
        raise ValueError(
            "fixed result does not contain one warmup and three measured passes"
        )
    validate_recorded_rank_layout(
        row["distributed"],
        world_size=int(case["world_size"]),
    )
    distributed = row["distributed"]
    owned_counts = distributed.get("owned_atom_counts", [])
    if (
        distributed.get("partition_count") != 1
        or distributed.get("gather_count") != 1
        or len(owned_counts) != int(case["world_size"])
        or sum(int(value) for value in owned_counts) != int(case["atom_count"])
    ):
        raise ValueError(
            "fixed result does not record one partition, one gather, and "
            "complete atom ownership"
        )
    if timing.get("source_input_sha256") != row.get("input", {}).get("tensor_sha256"):
        raise ValueError("fixed evaluation changed the source input tensor")
    output = row.get("output", {})
    position_invariance = output.get("position_invariance", {})
    pass_displacements = position_invariance.get(
        "measured_pass_maximum_minimum_image_displacements_a"
    )
    try:
        position_tolerance = float(position_invariance["tolerance_a"])
        warmup_displacement = float(
            position_invariance["warmup_maximum_minimum_image_displacement_a"]
        )
        final_displacement = float(
            position_invariance["final_gather_maximum_minimum_image_displacement_a"]
        )
        maximum_displacement = float(
            position_invariance["maximum_minimum_image_displacement_a"]
        )
        measured_displacements = [float(value) for value in pass_displacements]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "fixed evaluation is missing its minimum-image position check"
        ) from None
    displacement_values = [
        warmup_displacement,
        *measured_displacements,
        final_displacement,
    ]
    if (
        position_invariance.get("method") != "maximum_minimum_image_displacement"
        or not math.isclose(
            position_tolerance,
            DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or len(measured_displacements) != DEFAULT_PASS_COUNT
        or any(
            not math.isfinite(value) or value < 0.0 or value > position_tolerance
            for value in displacement_values
        )
        or not math.isclose(
            maximum_displacement,
            max(displacement_values),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or position_invariance.get("all_within_tolerance") is not True
    ):
        raise ValueError(
            "fixed evaluation is not PBC-equivalent to its source positions"
        )
    measured_passes = output.get("measured_passes")
    if not isinstance(measured_passes, list) or len(measured_passes) != (
        DEFAULT_PASS_COUNT
    ):
        raise ValueError("fixed result does not record all three measured outputs")
    expected_energy_dtype = DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(
        int(case["world_size"])
    )
    for expected_index, measured in enumerate(measured_passes, start=1):
        forces = measured.get("forces", {})
        measured_displacement = float(
            measured.get(
                "maximum_minimum_image_displacement_a",
                math.nan,
            )
        )
        if (
            measured.get("pass_index") != expected_index
            or not math.isfinite(float(measured.get("energy_ev", math.nan)))
            or measured.get("energy_dtype") != expected_energy_dtype
            or forces.get("finite") is not True
            or forces.get("shape") != [int(case["atom_count"]), 3]
            or not math.isclose(
                measured_displacement,
                measured_displacements[expected_index - 1],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "one measured pass has an invalid energy, force summary, "
                "or position check"
            )
    energy_values = [float(measured["energy_ev"]) for measured in measured_passes]
    energy_span_per_atom = (max(energy_values) - min(energy_values)) / int(
        case["atom_count"]
    )
    if (
        int(case["world_size"]) > 1
        and energy_span_per_atom
        > DEFAULT_DISTRIBUTED_ENERGY_REPEATABILITY_TOL_EV_PER_ATOM
    ):
        raise ValueError("measured-pass energies are not mutually consistent")

    last_pass = measured_passes[-1]
    last_forces = last_pass["forces"]
    atom_count = int(case["atom_count"])
    saved_output = row["output"]
    saved_forces = saved_output["forces_source_atom_order"]
    if (
        not math.isfinite(float(saved_output["energy_ev"]))
        or saved_output.get("energy_dtype") != last_pass["energy_dtype"]
        or saved_forces.get("finite") is not True
        or saved_forces.get("shape") != [atom_count, 3]
    ):
        raise ValueError("saved energy or force output is invalid")
    if not math.isclose(
        float(saved_output["energy_ev"]),
        float(last_pass["energy_ev"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("saved energy does not match the last measured pass")
    for name in ("rms_ev_a", "max_norm_ev_a"):
        observed = float(saved_forces[name])
        reference = float(last_forces[name])
        if not math.isclose(
            observed,
            reference,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "saved force summary does not match the last measured pass"
            )
    charges = row.get("charges", {})
    if int(case["world_size"]) == 1:
        try:
            charge_residual = float(charges["residual_e"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("one-GPU fixed result has no charge diagnostic") from exc
        if (
            charges.get("available") is not True
            or charges.get("finite") is not True
            or not math.isfinite(charge_residual)
        ):
            raise ValueError("one-GPU fixed result has an invalid charge diagnostic")
    elif charges.get("available") is not False or not charges.get("reason"):
        raise ValueError("multi-GPU fixed result does not explain unavailable charges")


def _validation_passed(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("success")):
        return False
    comparison = row.get("comparison", {})
    charges = row.get("charges", {})
    pme = row.get("pme", {})
    ewald = row.get("ewald", {})
    timing = row.get("timing", {})
    try:
        values = (
            float(charges["residual_e"]),
            float(pme["energy_ev"]),
            float(ewald["energy_ev"]),
            float(comparison["absolute_energy_difference_ev_per_atom"]),
            float(comparison["force_difference_max_norm_ev_a"]),
            float(timing["wall_s"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        comparison.get("passed") is True
        and charges.get("available") is True
        and charges.get("finite") is True
        and pme.get("forces", {}).get("finite") is True
        and ewald.get("forces", {}).get("finite") is True
        and all(math.isfinite(value) for value in values)
        and abs(values[0]) <= DEFAULT_CHARGE_SUM_TOL_E
        and values[-1] > 0.0
        and bool(timing.get("timed_work"))
    )


def write_phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    phase_dir = args.phase_dir.resolve()
    plan = _load_json(phase_dir / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("job plan has an unknown schema")
    rows = [
        _read_case_result(
            phase_dir / "results" / case["result_file"],
            case,
        )
        for case in _planned_cases(plan)
    ]
    fixed = rows[0]
    errors: list[str] = []
    try:
        _validate_fixed_result(fixed, plan=plan)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    if int(plan["fixed_case"]["world_size"]) == 1:
        if len(rows) != 2 or not _validation_passed(rows[1]):
            errors.append("the 3,200-atom PME-versus-Ewald check failed")
    elif len(rows) != 1:
        errors.append("only the fixed evaluation is allowed above one GPU")

    fixed_input = fixed.get("input", {})
    input_structure_sha256 = (
        fixed_input.get("file_sha256") if isinstance(fixed_input, Mapping) else None
    )
    if input_structure_sha256 is None and isinstance(fixed_input, Mapping):
        legacy_structure = fixed_input.get("structure")
        if isinstance(legacy_structure, Mapping):
            input_structure_sha256 = legacy_structure.get("sha256")
    if (
        not isinstance(input_structure_sha256, str)
        or len(input_structure_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in input_structure_sha256
        )
    ):
        errors.append("fixed result does not record a valid input structure SHA-256")
        input_structure_sha256 = None

    summary = {
        "schema": PHASE_SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "run_id": plan["run_id"],
        "world_size": plan["fixed_case"]["world_size"],
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "planned_case_count": plan["planned_case_count"],
        "completed_case_count": sum(bool(row.get("success")) for row in rows),
        "fixed_case_id": plan["fixed_case"]["case_id"],
        "fixed_result_sha256": fixed["result_file_sha256"],
        "input_structure_sha256": input_structure_sha256,
        "methodology": methodology_source_identity(),
        "electrostatics_validation": (
            {
                "case_id": rows[1]["case_id"],
                "passed": _validation_passed(rows[1]),
                "result_sha256": rows[1]["result_file_sha256"],
            }
            if len(rows) == 2
            else None
        ),
    }
    atomic_write_json(args.output.resolve(), summary)
    if errors:
        raise ValueError("; ".join(errors))
    return summary


def _rank_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("rank-*.json")):
        row = _load_json(path)
        row["record_sha256"] = sha256_file(path)
        records.append(row)
    return records


def _failure_type(log_text: str, exit_code: int) -> str:
    lowered = log_text.lower()
    if "out of memory" in lowered or "outofmemoryerror" in lowered:
        return "CudaOutOfMemory"
    if exit_code == 124:
        return "TimeLimit"
    return "ProcessFailure"


def record_failure(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_extxyz.resolve()
    log_path = args.case_log.resolve()
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    rank_records = _rank_records(args.rank_output_dir.resolve())
    stages = [
        str(record.get("stage")) for record in rank_records if record.get("stage")
    ]
    failure = {
        "schema": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "run_id": args.run_id,
        "case_id": args.case_id,
        "mode": args.mode,
        "measurement_role": (
            "fixed_evaluation"
            if args.mode == "distributed"
            else "electrostatics_validation"
        ),
        "world_size": args.world_size,
        "success": False,
        "status": "failed",
        "error": (
            log_text.strip().splitlines()[-1]
            if log_text.strip()
            else f"runner exited with code {args.exit_code}"
        ),
        "failure": {
            "type": _failure_type(log_text, args.exit_code),
            "stage": stages[-1] if stages else "process",
            "exit_code": args.exit_code,
            "unexpected": True,
        },
        "input": {
            "path": str(input_path),
            "file_sha256": sha256_file(input_path) if input_path.is_file() else None,
            "file_size_bytes": (
                input_path.stat().st_size if input_path.is_file() else None
            ),
            "structure": (
                {
                    "path": str(input_path),
                    "sha256": sha256_file(input_path),
                }
                if input_path.is_file()
                else None
            ),
        },
        "rank_records": rank_records,
        "log": {
            "path": str(log_path),
            "sha256": sha256_file(log_path) if log_path.is_file() else None,
        },
    }
    atomic_write_json(args.output.resolve(), failure)
    return failure


def _force_record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        record = row["output"]["forces_source_atom_order_npy"]
    except (KeyError, TypeError) as exc:
        raise ValueError("fixed result does not record its force array") from exc
    if not isinstance(record, Mapping):
        raise ValueError("fixed result force record is invalid")
    return record


def _load_force_array(row: Mapping[str, Any]) -> Any:
    import numpy as np

    record = _force_record(row)
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError("fixed result force array is missing or changed")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != list(record["shape"]):
        raise ValueError("fixed result force array has the wrong shape")
    return values


def _timing_samples(row: Mapping[str, Any]) -> list[float]:
    timing = row["timing"]
    raw = timing.get("pass_times_s")
    if not isinstance(raw, list) or len(raw) != DEFAULT_PASS_COUNT:
        raise ValueError("fixed result has the wrong measured-pass series")
    return [float(value) for value in raw]


def _timing_median(row: Mapping[str, Any]) -> float:
    timing = row["timing"]
    return float(timing["median_s"])


def _output_metrics(row: Mapping[str, Any]) -> tuple[float, float, float]:
    output = row["output"]
    energy = float(output["energy_ev"])
    summary = output["forces_source_atom_order"]
    force_rms = float(summary["rms_ev_a"])
    force_max = float(summary["max_norm_ev_a"])
    return energy, force_rms, force_max


def _measured_energy_values(row: Mapping[str, Any]) -> list[float]:
    passes = row["output"]["measured_passes"]
    if not isinstance(passes, list) or len(passes) != DEFAULT_PASS_COUNT:
        raise ValueError("fixed result has the wrong measured-energy series")
    values = [float(measured["energy_ev"]) for measured in passes]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("fixed result has a non-finite measured energy")
    return values


def _measured_energy_median(row: Mapping[str, Any]) -> float:
    return float(sorted(_measured_energy_values(row))[DEFAULT_PASS_COUNT // 2])


def _measured_energy_span_per_atom(
    row: Mapping[str, Any],
    *,
    atom_count: int,
) -> float:
    values = _measured_energy_values(row)
    return (max(values) - min(values)) / atom_count


def _methodology_identity_from_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(plan["methodology"]["source_identity"])


def _copy_bundle_file(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"copied file checksum changed: {source}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _verify_checksum_file(path: Path) -> dict[str, str]:
    """Verify and return one GNU sha256sum file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            digest, raw_name = raw_line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not a sha256sum record") from exc
        name = raw_name.removeprefix("*").strip()
        source = Path(name)
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not source.is_file()
            or sha256_file(source) != digest
        ):
            raise ValueError(f"{path}:{line_number} does not match {source}")
        records[str(source)] = digest
    if not records:
        raise ValueError(f"{path} contains no checksum records")
    return records


def _copy_job_records(
    job_dir: Path,
    *,
    world_size: int,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Copy the complete checked job record and return path rewrites."""

    required_top = (
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
    )
    for name in required_top:
        if not (job_dir / name).is_file():
            raise FileNotFoundError(f"{world_size}-GPU job is missing {name}")
    producer_records = _verify_checksum_file(job_dir / "producer-SHA256SUMS")
    artifact_records = _verify_checksum_file(job_dir / "artifact-SHA256SUMS")
    producer_files: dict[str, str] = {}
    producer_rewrites: dict[str, str] = {}
    for raw_path, digest in producer_records.items():
        name = Path(raw_path).name
        if name in producer_files:
            raise ValueError(f"producer checksum list repeats the file name {name}")
        producer_files[name] = digest
        producer_rewrites[raw_path] = (
            f"manifest.json#job_records/{world_size}/producer_files/{name}"
        )

    files = [job_dir / name for name in required_top]
    for name in ("inputs", "results", "ranks", "logs"):
        directory = job_dir / name
        if not directory.is_dir():
            raise FileNotFoundError(f"{world_size}-GPU job is missing {name}/")
        found = sorted(path for path in directory.rglob("*") if path.is_file())
        if not found:
            raise ValueError(f"{world_size}-GPU job has no files under {name}/")
        files.extend(found)
    expected_artifacts = {
        str(path.resolve()) for path in files if path.name != "artifact-SHA256SUMS"
    }
    if set(artifact_records) != expected_artifacts:
        missing = expected_artifacts - set(artifact_records)
        extra = set(artifact_records) - expected_artifacts
        raise ValueError(
            f"{world_size}-GPU artifact checksum list is incomplete: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    destination_root = output_dir / "job-records" / f"gpus-{world_size:02d}"
    rewrites: dict[str, str] = dict(producer_rewrites)
    index: dict[str, dict[str, Any]] = {}
    for source in sorted(set(files)):
        relative = source.relative_to(job_dir)
        destination = destination_root / relative
        copied = _copy_bundle_file(source, destination)
        bundle_relative = str(destination.relative_to(output_dir))
        rewrites[str(source.resolve())] = bundle_relative
        index[str(relative)] = {
            "path": bundle_relative,
            "sha256": copied["sha256"],
            "size_bytes": copied["size_bytes"],
        }
    return rewrites, {
        "world_size": world_size,
        "files": index,
        "producer_checksum_file_sha256": sha256_file(job_dir / "producer-SHA256SUMS"),
        "artifact_checksum_file_sha256": sha256_file(job_dir / "artifact-SHA256SUMS"),
        "verified_producer_file_count": len(producer_records),
        "verified_artifact_file_count": len(artifact_records),
        "producer_files": dict(sorted(producer_files.items())),
    }


def _register_path_rewrite(
    rewrites: dict[str, str],
    raw_path: Any,
    portable_reference: str,
) -> None:
    """Register one absolute host path without changing copied-file rewrites."""

    if not isinstance(raw_path, str):
        return
    path = Path(raw_path)
    if not path.is_absolute():
        return
    resolved = str(path.resolve())
    rewrites.setdefault(resolved, portable_reference)


def _git_source_reference(
    *,
    repository: str,
    commit: Any,
    root: Any,
    path: Any | None = None,
) -> str:
    """Describe a checked Git source without retaining its host checkout path."""

    reference = f"git:{repository}@{commit}"
    if not isinstance(path, str) or not isinstance(root, str):
        return reference
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return reference
    return f"{reference}#{relative.as_posix()}"


def _register_runtime_path_rewrites(
    value: Any,
    *,
    world_size: int,
    rewrites: dict[str, str],
) -> None:
    """Replace Python installation locations with the checked runtime record."""

    if not isinstance(value, Mapping):
        return
    runtime_reference = f"job-records/gpus-{world_size:02d}/part1-runtime.json"
    _register_path_rewrite(
        rewrites,
        value.get("python_executable"),
        f"{runtime_reference}#python_executable",
    )
    _register_path_rewrite(
        rewrites,
        value.get("python_prefix"),
        f"{runtime_reference}#python_prefix",
    )


def _register_row_identity_rewrites(
    row: Mapping[str, Any],
    *,
    world_size: int,
    rewrites: dict[str, str],
) -> None:
    """Map external source locations to portable, checked identities."""

    source = row.get("source")
    tutorial_root: Any = None
    tutorial_commit: Any = None
    if isinstance(source, Mapping):
        tutorial_root = source.get("repository_root")
        tutorial_commit = source.get(
            "repository_commit",
            source.get("tutorial_commit"),
        )
        tutorial_reference = _git_source_reference(
            repository="ALCHEMI-Bootcamp",
            commit=tutorial_commit,
            root=tutorial_root,
        )
        _register_path_rewrite(
            rewrites,
            tutorial_root,
            tutorial_reference,
        )
        for key in (
            "runner",
            "domain_methodology_config_file",
        ):
            raw_path = source.get(key)
            _register_path_rewrite(
                rewrites,
                raw_path,
                _git_source_reference(
                    repository="ALCHEMI-Bootcamp",
                    commit=tutorial_commit,
                    root=tutorial_root,
                    path=raw_path,
                ),
            )
        runner_file = source.get("runner_file")
        if isinstance(runner_file, Mapping):
            raw_path = runner_file.get("path")
            _register_path_rewrite(
                rewrites,
                raw_path,
                _git_source_reference(
                    repository="ALCHEMI-Bootcamp",
                    commit=tutorial_commit,
                    root=tutorial_root,
                    path=raw_path,
                ),
            )

        for prefix, repository in (
            ("toolkit_core", "NVIDIA/nvalchemi-toolkit"),
            ("toolkit_ops", "NVIDIA/nvalchemi-toolkit-ops"),
        ):
            root = source.get(f"{prefix}_source_root")
            commit = source.get(f"{prefix}_commit")
            _register_path_rewrite(
                rewrites,
                root,
                _git_source_reference(
                    repository=repository,
                    commit=commit,
                    root=root,
                ),
            )
            raw_path = source.get(f"{prefix}_source_file")
            _register_path_rewrite(
                rewrites,
                raw_path,
                _git_source_reference(
                    repository=repository,
                    commit=commit,
                    root=root,
                    path=raw_path,
                ),
            )

        checkpoint_reference = (
            "model:AIMNet2/"
            f"{source.get('aimnet_checkpoint_name', AIMNET_CHECKPOINT)}"
            f"@sha256:{source.get('aimnet_checkpoint_sha256')}"
        )
        _register_path_rewrite(
            rewrites,
            source.get("aimnet_checkpoint"),
            checkpoint_reference,
        )
        checkpoint_file = source.get("aimnet_checkpoint_file")
        if isinstance(checkpoint_file, Mapping):
            _register_path_rewrite(
                rewrites,
                checkpoint_file.get("path"),
                checkpoint_reference,
            )
        _register_runtime_path_rewrites(
            source.get("runtime_software"),
            world_size=world_size,
            rewrites=rewrites,
        )

    input_record = row.get("input")
    if isinstance(input_record, Mapping):
        input_manifest = input_record.get("manifest")
        if isinstance(input_manifest, Mapping):
            construction = input_manifest.get("construction")
            if isinstance(construction, Mapping):
                for path_key, hash_key, label in (
                    (
                        "base_box_manifest",
                        "base_box_manifest_sha256",
                        "manifest",
                    ),
                    (
                        "base_box_structure",
                        "base_box_structure_sha256",
                        "structure",
                    ),
                ):
                    _register_path_rewrite(
                        rewrites,
                        construction.get(path_key),
                        (
                            f"input:prebuilt-base-box-{label}"
                            f"@sha256:{construction.get(hash_key)}"
                        ),
                    )
            input_source = input_manifest.get("source")
            if isinstance(input_source, Mapping):
                for path_key in (
                    "packing_helper",
                    "domain_methodology_config",
                ):
                    raw_path = input_source.get(path_key)
                    _register_path_rewrite(
                        rewrites,
                        raw_path,
                        _git_source_reference(
                            repository="ALCHEMI-Bootcamp",
                            commit=tutorial_commit,
                            root=tutorial_root,
                            path=raw_path,
                        ),
                    )
                _register_path_rewrite(
                    rewrites,
                    input_source.get("nci_subset"),
                    (
                        "dataset:NCI-Atlas"
                        f"@sha256:{input_source.get('nci_subset_sha256')}"
                    ),
                )

    methodology = row.get("methodology")
    if isinstance(methodology, Mapping):
        source_file = methodology.get("source_file")
        if isinstance(source_file, Mapping):
            _register_path_rewrite(
                rewrites,
                source_file.get("path"),
                (
                    "manifest.json#job_records/"
                    f"{world_size}/producer_files/{DOMAIN_METHODOLOGY_CONFIG_PATH.name}"
                ),
            )

    runtime_rows = row.get("runtime")
    if isinstance(runtime_rows, list):
        for runtime in runtime_rows:
            _register_runtime_path_rewrites(
                runtime,
                world_size=world_size,
                rewrites=rewrites,
            )

    model = row.get("model")
    if isinstance(model, Mapping):
        d3 = model.get("d3")
        if isinstance(d3, Mapping):
            d3_reference = f"parameters:DFT-D3@sha256:{d3.get('parameter_file_sha256')}"
            _register_path_rewrite(
                rewrites,
                d3.get("parameter_file"),
                d3_reference,
            )
            identity = d3.get("parameter_file_identity")
            if isinstance(identity, Mapping):
                _register_path_rewrite(
                    rewrites,
                    identity.get("path"),
                    d3_reference,
                )


def _rewrite_known_paths(
    value: Any,
    *,
    rewrites: Mapping[str, str],
    context: str = "raw result",
) -> Any:
    """Rewrite copied job paths while preserving the full result structure."""

    if isinstance(value, dict):
        return {
            key: _rewrite_known_paths(
                item,
                rewrites=rewrites,
                context=f"{context}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_known_paths(
                item,
                rewrites=rewrites,
                context=f"{context}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        try:
            resolved = str(Path(value).resolve()) if Path(value).is_absolute() else None
        except (OSError, ValueError):
            resolved = None
        if resolved is not None and resolved in rewrites:
            return rewrites[resolved]
        if resolved is not None:
            raise ValueError(f"{context} contains an undeclared absolute path: {value}")
    return value


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    job_dirs = [path.resolve() for path in args.job_dir]
    if len(job_dirs) != len(DEFAULT_WORLD_SIZES):
        raise ValueError(
            "bundle requires one job directory for each of 1, 2, and 4 GPUs"
        )

    plans: dict[int, dict[str, Any]] = {}
    rows: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    job_dirs_by_world: dict[int, Path] = {}
    for job_dir in job_dirs:
        plan = _load_json(job_dir / "plan.json")
        if plan.get("schema") != PLAN_SCHEMA:
            raise ValueError(f"{job_dir} has an unknown plan schema")
        world_size = int(plan["fixed_case"]["world_size"])
        if world_size in plans:
            raise ValueError(f"duplicate {world_size}-GPU job")
        if world_size not in DEFAULT_WORLD_SIZES:
            raise ValueError(f"unexpected {world_size}-GPU job")
        summary = _load_json(job_dir / "phase-summary.json")
        if (
            summary.get("schema") != PHASE_SUMMARY_SCHEMA
            or summary.get("passed") is not True
            or int(summary.get("world_size", -1)) != world_size
        ):
            raise ValueError(f"{world_size}-GPU job did not pass")
        row = _read_case_result(
            job_dir / "results" / plan["fixed_case"]["result_file"],
            plan["fixed_case"],
        )
        _validate_fixed_result(row, plan=plan)
        plans[world_size] = plan
        rows[world_size] = row
        summaries[world_size] = summary
        job_dirs_by_world[world_size] = job_dir

    if tuple(sorted(plans)) != tuple(sorted(DEFAULT_WORLD_SIZES)):
        raise ValueError("bundle is missing a declared GPU count")

    reference_plan = plans[1]
    identity_fields = (
        "tutorial_commit",
        "toolkit_core_commit",
        "toolkit_ops_commit",
        "nci_subset_sha256",
        "aimnet_checkpoint",
        "aimnet_checkpoint_sha256",
        "d3_parameter_sha256",
    )
    reference_source = reference_plan["source"]
    reference_methodology = _methodology_identity_from_plan(reference_plan)
    reference_input_sha = rows[1]["input"]["file_sha256"]
    reference_atom_count = int(rows[1]["atom_count"])
    settings_record = {
        "methodology": reference_plan["methodology"]["resolved_values"],
        "model": reference_plan["model"],
        "input_structure_sha256": reference_input_sha,
        "position_invariance": {
            "method": "maximum_minimum_image_displacement",
            "tolerance_a": (DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A),
        },
    }
    settings_sha256 = canonical_json_sha256(settings_record)
    for world_size in DEFAULT_WORLD_SIZES:
        plan = plans[world_size]
        if any(
            plan["source"][name] != reference_source[name] for name in identity_fields
        ):
            raise ValueError("jobs were produced from different source inputs")
        if _methodology_identity_from_plan(plan) != reference_methodology:
            raise ValueError("jobs used different methodology files")
        if rows[world_size]["input"]["file_sha256"] != reference_input_sha:
            raise ValueError("jobs did not evaluate the same structure content")
        if int(rows[world_size]["atom_count"]) != reference_atom_count:
            raise ValueError("jobs report different atom counts")

    reference_forces = _load_force_array(rows[1])
    reference_forces_float64 = np.asarray(
        reference_forces,
        dtype=np.float64,
    )
    one_gpu_energy = _measured_energy_median(rows[1])
    two_gpu_energy = _measured_energy_median(rows[2])
    reference_median = _timing_median(rows[1])
    output_comparisons: dict[int, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    portable_rows: list[dict[str, Any]] = []
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"bundle output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path_rewrites: dict[int, dict[str, str]] = {}
    job_record_indexes: dict[str, dict[str, Any]] = {}
    for world_size in DEFAULT_WORLD_SIZES:
        rewrites, record_index = _copy_job_records(
            job_dirs_by_world[world_size],
            world_size=world_size,
            output_dir=output_dir,
        )
        path_rewrites[world_size] = rewrites
        job_record_indexes[str(world_size)] = record_index
    reference_producers = job_record_indexes["1"]["producer_files"]
    if any(
        job_record_indexes[str(world_size)]["producer_files"] != reference_producers
        for world_size in DEFAULT_WORLD_SIZES[1:]
    ):
        raise ValueError("jobs used different producer source files")

    observed_nodes_by_world: dict[int, int] = {}
    observed_gpus_by_world: dict[int, int] = {}
    for world_size in DEFAULT_WORLD_SIZES:
        runtime_rows = rows[world_size].get("runtime")
        if not isinstance(runtime_rows, list) or len(runtime_rows) != world_size:
            raise ValueError(f"{world_size}-GPU result has incomplete runtime records")
        hosts = {str(runtime["host"]) for runtime in runtime_rows}
        gpu_ids = {
            (
                str(runtime["host"]),
                str(runtime.get("gpu_uuid") or runtime["local_rank"]),
            )
            for runtime in runtime_rows
        }
        observed_nodes_by_world[world_size] = len(hosts)
        observed_gpus_by_world[world_size] = len(gpu_ids)
        if (
            observed_nodes_by_world[world_size] != world_size
            or observed_gpus_by_world[world_size] != world_size
        ):
            raise ValueError(
                "the recorded job does not show one distinct H100 GPU per node"
            )
    max_observed_gpus = max(observed_gpus_by_world.values())
    max_observed_nodes = max(observed_nodes_by_world.values())
    raw_source = rows[1].get("source", {})
    toolkit_version = raw_source.get("toolkit_version")
    if (
        not toolkit_version
        or raw_source.get("repository_commit") != reference_source["tutorial_commit"]
        or any(
            rows[world_size].get("source", {}).get("toolkit_version") != toolkit_version
            for world_size in DEFAULT_WORLD_SIZES
        )
    ):
        raise ValueError(
            "raw job source identity does not match the planned tutorial source"
        )
    hardware_reference = rows[1]["runtime"][0]
    for world_size in DEFAULT_WORLD_SIZES:
        for runtime in rows[world_size]["runtime"]:
            if any(
                runtime.get(name) != hardware_reference.get(name)
                for name in (
                    "gpu_name",
                    "gpu_total_memory_bytes",
                    "driver_version",
                    "torch_cuda_version",
                )
            ):
                raise ValueError("jobs used different GPU hardware or CUDA runtimes")

    for world_size in DEFAULT_WORLD_SIZES:
        row = rows[world_size]
        forces = _load_force_array(row)
        if forces.shape != reference_forces.shape:
            raise ValueError("force arrays have different shapes")
        force_difference = (
            np.asarray(forces, dtype=np.float64) - reference_forces_float64
        )
        atom_norms = np.linalg.norm(force_difference, axis=1)
        component_differences = np.abs(force_difference)
        component_limits = (
            DEFAULT_EVALUATION_FORCE_ATOL_EV_A
            + DEFAULT_EVALUATION_FORCE_RTOL * np.abs(reference_forces_float64)
        )
        force_rms_difference = float(np.sqrt(np.mean(np.square(force_difference))))
        force_max_difference = float(np.max(atom_norms))
        force_max_component_difference = float(np.max(component_differences))
        force_components_passed = bool(
            np.all(component_differences <= component_limits)
        )
        saved_energy, force_rms, force_max = _output_metrics(row)
        comparison_energy = _measured_energy_median(row)
        energy_span_per_atom = _measured_energy_span_per_atom(
            row,
            atom_count=reference_atom_count,
        )
        energy_repeatability_required = world_size > 1
        energy_repeatability_passed = (
            energy_span_per_atom
            <= DEFAULT_DISTRIBUTED_ENERGY_REPEATABILITY_TOL_EV_PER_ATOM
        )
        one_gpu_energy_offset = comparison_energy - one_gpu_energy
        one_gpu_energy_abs_offset_per_atom = (
            abs(one_gpu_energy_offset) / reference_atom_count
        )
        distributed_energy_difference = comparison_energy - two_gpu_energy
        distributed_energy_difference_per_atom = (
            abs(distributed_energy_difference) / reference_atom_count
        )
        distributed_energy_required = world_size == 4
        distributed_energy_passed = (
            distributed_energy_difference_per_atom
            <= DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM
        )
        position_invariance = row["output"]["position_invariance"]
        maximum_position_displacement_a = float(
            position_invariance["maximum_minimum_image_displacement_a"]
        )
        positions_pbc_equivalent = bool(
            position_invariance["all_within_tolerance"]
            and maximum_position_displacement_a
            <= DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
        )
        required_checks_passed = (
            force_components_passed
            and (not energy_repeatability_required or energy_repeatability_passed)
            and (not distributed_energy_required or distributed_energy_passed)
            and positions_pbc_equivalent
        )
        if not required_checks_passed:
            raise ValueError(
                f"{world_size}-GPU output fails the declared force or "
                "distributed-energy comparison, or the position check"
            )
        output_comparisons[world_size] = {
            "one_gpu_energy_offset_ev": one_gpu_energy_offset,
            "one_gpu_energy_abs_offset_ev_per_atom": (
                one_gpu_energy_abs_offset_per_atom
            ),
            "one_gpu_energy_offset_is_diagnostic_only": True,
            "energy_statistic": "median_of_three_measured_passes",
            "energy_dtype": str(row["output"]["energy_dtype"]),
            "energy_repeatability_span_ev_per_atom": energy_span_per_atom,
            "energy_repeatability_tolerance_ev_per_atom": (
                DEFAULT_DISTRIBUTED_ENERGY_REPEATABILITY_TOL_EV_PER_ATOM
            ),
            "energy_repeatability_check_required": energy_repeatability_required,
            "energy_repeatability_passed": (
                energy_repeatability_passed if energy_repeatability_required else None
            ),
            "distributed_energy_reference_gpus": 2,
            "distributed_energy_difference_ev": (distributed_energy_difference),
            "distributed_energy_abs_difference_ev_per_atom": (
                distributed_energy_difference_per_atom
            ),
            "distributed_energy_check_required": (distributed_energy_required),
            "distributed_energy_passed": (
                distributed_energy_passed if distributed_energy_required else None
            ),
            "force_rms_difference_ev_per_a_vs_1gpu": (force_rms_difference),
            "force_max_difference_ev_per_a_vs_1gpu": (force_max_difference),
            "force_max_component_difference_ev_per_a_vs_1gpu": (
                force_max_component_difference
            ),
            "distributed_energy_agreement_tolerance_ev_per_atom": (
                DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM
            ),
            "force_acceptance": (
                "abs(delta_component) <= atol + rtol * abs(one_gpu_component)"
            ),
            "force_atol_ev_per_a": DEFAULT_EVALUATION_FORCE_ATOL_EV_A,
            "force_rtol": DEFAULT_EVALUATION_FORCE_RTOL,
            "force_passed": force_components_passed,
            "position_check": "maximum_minimum_image_displacement",
            "position_tolerance_a": (DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A),
            "maximum_minimum_image_displacement_a": (maximum_position_displacement_a),
            "positions_pbc_equivalent": positions_pbc_equivalent,
            "required_checks_passed": True,
        }
        timing_samples = _timing_samples(row)
        median = _timing_median(row)
        distributed = row["distributed"]
        _, rank_grid = validate_recorded_rank_layout(
            distributed,
            world_size=world_size,
        )
        owned_counts = [int(value) for value in distributed["owned_atom_counts"]]
        peak_memory = row.get("memory", {}).get(
            "max_allocated_bytes",
            "",
        )
        timing = row["timing"]
        csv_rows.append(
            {
                "case_id": row["case_id"],
                "atom_count": reference_atom_count,
                "molecules_per_species": DEFAULT_FIXED_PAIR_COUNT,
                "nodes": world_size,
                "gpus": world_size,
                "ranks": world_size,
                "success": True,
                "status": "complete",
                "failure_type": "",
                "failure_stage": "",
                "error": "",
                "warmup_count": DEFAULT_WARMUP_COUNT,
                "measured_pass_count": DEFAULT_PASS_COUNT,
                "pass_times_s": json.dumps(timing_samples),
                "median_s": median,
                "min_s": float(timing["min_s"]),
                "max_s": float(timing["max_s"]),
                "peak_memory_bytes_max_rank": peak_memory,
                "owned_atoms_min_rank": min(owned_counts),
                "owned_atoms_max_rank": max(owned_counts),
                "spatial_grid": "x".join(str(value) for value in rank_grid),
                "energy_ev": saved_energy,
                "energy_ev_per_atom": saved_energy / reference_atom_count,
                "comparison_energy_ev": comparison_energy,
                "comparison_energy_ev_per_atom": (
                    comparison_energy / reference_atom_count
                ),
                "comparison_energy_statistic": ("median_of_three_measured_passes"),
                "energy_dtype": str(row["output"]["energy_dtype"]),
                "force_rms_ev_per_a": force_rms,
                "force_max_ev_per_a": force_max,
                "structure_sha256": reference_input_sha,
                "settings_sha256": settings_sha256,
                "input_tensor_sha256": row["input"]["tensor_sha256"],
                "positions_pbc_equivalent": positions_pbc_equivalent,
                "max_minimum_image_displacement_a": (maximum_position_displacement_a),
                "measurement_role": "fixed_evaluation",
                "measurement_kind": ("fixed_structure_energy_force_pass"),
            }
        )

        _register_row_identity_rewrites(
            row,
            world_size=world_size,
            rewrites=path_rewrites[world_size],
        )
        portable = _rewrite_known_paths(
            row,
            rewrites=path_rewrites[world_size],
        )
        portable["settings_sha256"] = settings_sha256
        portable["bundle_source"] = "manifest.json#source"
        portable["bundle_job_record"] = f"manifest.json#job_records/{world_size}"
        portable_rows.append(portable)

    one_gpu_plan = plans[1]
    validation_case = one_gpu_plan["validation_cases"][0]
    validation_row = _read_case_result(
        job_dirs_by_world[1] / "results" / validation_case["result_file"],
        validation_case,
    )
    if not _validation_passed(validation_row):
        raise ValueError("the PME-versus-Ewald validation did not pass")
    _register_row_identity_rewrites(
        validation_row,
        world_size=1,
        rewrites=path_rewrites[1],
    )
    portable_validation = _rewrite_known_paths(
        validation_row,
        rewrites=path_rewrites[1],
    )
    portable_validation["bundle_settings_sha256"] = settings_sha256
    portable_validation["bundle_source"] = "manifest.json#source"
    portable_validation["bundle_job_record"] = "manifest.json#job_records/1"
    portable_rows.append(portable_validation)

    _write_csv(output_dir / "distributed.csv", DISTRIBUTED_COLUMNS, csv_rows)
    atomic_write_jsonl(output_dir / "raw-results.jsonl", portable_rows)
    atomic_write_json(
        output_dir / "electrostatics-validation.json",
        portable_validation,
    )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": utc_now(),
        "lesson": "Part 1 fixed-input domain decomposition",
        "site": args.site,
        "interconnect": args.interconnect,
        "source": {
            **{name: reference_source[name] for name in identity_fields},
            "toolkit_version": toolkit_version,
        },
        "status": "complete",
        "methodology": reference_methodology,
        "settings": settings_record,
        "settings_sha256": settings_sha256,
        "input": {
            "molecules_per_species": DEFAULT_FIXED_PAIR_COUNT,
            "atom_count": reference_atom_count,
            "structure_sha256": reference_input_sha,
        },
        "execution": {
            "gpu_counts": list(DEFAULT_WORLD_SIZES),
            "warmup_count": DEFAULT_WARMUP_COUNT,
            "measured_pass_count": DEFAULT_PASS_COUNT,
            "work_per_measured_pass": (
                "one fixed-structure energy-and-force evaluation"
            ),
            "publishable_benchmark": False,
            "observed_speedup": {
                str(world_size): reference_median / _timing_median(rows[world_size])
                for world_size in DEFAULT_WORLD_SIZES
            },
            "parallel_efficiency": {
                str(world_size): (
                    reference_median / _timing_median(rows[world_size]) / world_size
                )
                for world_size in DEFAULT_WORLD_SIZES
            },
        },
        "hardware": {
            "site": args.site,
            "site_source": "operator-declared",
            "interconnect": args.interconnect,
            "interconnect_source": ("operator-declared; raw GPU topology is retained"),
            "gpus_available": max_observed_gpus,
            "nodes_available": max_observed_nodes,
            "resource_count_source": (
                "derived from successful per-rank runtime records"
            ),
            "gpu_model": hardware_reference["gpu_name"],
            "gpu_memory_bytes": hardware_reference["gpu_total_memory_bytes"],
            "driver_version": hardware_reference["driver_version"],
            "cuda_version": hardware_reference["torch_cuda_version"],
            "observed_gpus_by_job": {
                str(world_size): observed_gpus_by_world[world_size]
                for world_size in DEFAULT_WORLD_SIZES
            },
            "observed_nodes_by_job": {
                str(world_size): observed_nodes_by_world[world_size]
                for world_size in DEFAULT_WORLD_SIZES
            },
        },
        "output_agreement": {
            "force_reference_gpus": 1,
            "distributed_energy_reference_gpus": 2,
            "one_gpu_energy_offsets_are_diagnostics_only": True,
            "energy_statistic": "median_of_three_measured_passes",
            "position_check": {
                "method": "maximum_minimum_image_displacement",
                "tolerance_a": (DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A),
                "meaning": (
                    "Coordinates may be wrapped to equivalent periodic images; "
                    "each warmup, measured pass, and final gather remains within "
                    "this minimum-image displacement."
                ),
            },
            "comparisons": {
                str(world_size): output_comparisons[world_size]
                for world_size in DEFAULT_WORLD_SIZES
            },
            "all_required_checks_passed": True,
        },
        "electrostatics_validation": {
            "file": "electrostatics-validation.json",
            "sha256": sha256_file(output_dir / "electrostatics-validation.json"),
            "passed": True,
        },
        "job_records": job_record_indexes,
        "files": {},
        "interpretation": (
            "The same fixed structure and complete AIMNet2 plus PME plus D3 "
            "calculation ran through public Toolkit APIs on 1, 2, and 4 H100 "
            "GPUs. The recorded times describe only these three short passes."
        ),
    }
    for name in (
        "distributed.csv",
        "raw-results.jsonl",
        "electrostatics-validation.json",
    ):
        path = output_dir / name
        manifest["files"][name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    atomic_write_json(output_dir / "manifest.json", manifest)
    checksum_lines = []
    for path in sorted(
        candidate
        for candidate in output_dir.rglob("*")
        if candidate.is_file() and candidate.name != "SHA256SUMS"
    ):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return manifest


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan.resolve())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan has an unknown schema")
    result_dir = args.result_dir.resolve()
    rows = [
        _read_case_result(result_dir / case["result_file"], case)
        for case in _planned_cases(plan)
    ]
    atomic_write_jsonl(args.output_jsonl.resolve(), rows)
    summary = {
        "schema": COLLECTION_SCHEMA,
        "created_utc": utc_now(),
        "run_id": plan["run_id"],
        "planned_rows": len(rows),
        "successful_rows": sum(bool(row.get("success")) for row in rows),
        "failed_rows": sum(not bool(row.get("success")) for row in rows),
    }
    atomic_write_json(args.output_summary.resolve(), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and combine the short Part 1 multi-GPU example."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Write the independent plan for one 1-, 2-, or 4-GPU job.",
    )
    plan_parser.add_argument("--run-id", required=True)
    plan_parser.add_argument("--tutorial-commit", required=True)
    plan_parser.add_argument("--world-size", type=int, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Build a deterministic integer supercell of the checked base box.",
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
    )
    prepare_parser.add_argument("--nci-data", type=Path)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--reuse-existing", action="store_true")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint-preflight",
        help="Resolve and verify the pinned AIMNet2 checkpoint.",
    )
    checkpoint_parser.add_argument(
        "--checkpoint",
        default=AIMNET_CHECKPOINT,
    )
    checkpoint_parser.add_argument("--output", type=Path, required=True)

    failure_parser = subparsers.add_parser(
        "record-failure",
        help="Save one unexpected runner failure as a normal result row.",
    )
    failure_parser.add_argument("--run-id", required=True)
    failure_parser.add_argument("--case-id", required=True)
    failure_parser.add_argument(
        "--mode",
        choices=("distributed", "electrostatics-validation"),
        required=True,
    )
    failure_parser.add_argument("--world-size", type=int, required=True)
    failure_parser.add_argument("--input-extxyz", type=Path, required=True)
    failure_parser.add_argument("--rank-output-dir", type=Path, required=True)
    failure_parser.add_argument("--case-log", type=Path, required=True)
    failure_parser.add_argument("--exit-code", type=int, required=True)
    failure_parser.add_argument("--output", type=Path, required=True)

    phase_parser = subparsers.add_parser(
        "phase-summary",
        help="Check one independent GPU job.",
    )
    phase_parser.add_argument("--phase-dir", type=Path, required=True)
    phase_parser.add_argument("--output", type=Path, required=True)

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Combine complete 1-, 2-, and 4-GPU jobs.",
    )
    bundle_parser.add_argument(
        "--job-dir",
        action="append",
        type=Path,
        required=True,
    )
    bundle_parser.add_argument("--site", required=True)
    bundle_parser.add_argument("--interconnect", required=True)
    bundle_parser.add_argument("--output-dir", type=Path, required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="Collect one job's planned rows as JSONL.",
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
            tutorial_commit=args.tutorial_commit,
            world_size=args.world_size,
        )
        atomic_write_json(args.output.resolve(), plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
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
    if args.command == "phase-summary":
        summary = write_phase_summary(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "bundle":
        manifest = build_bundle(args)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "assemble":
        summary = assemble(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
