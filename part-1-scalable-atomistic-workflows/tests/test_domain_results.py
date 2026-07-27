"""Tests for strict cached H100 domain-decomposition lesson results."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain.results import (  # noqa: E402
    BUNDLE_SCHEMA,
    CAPACITY_COLUMNS,
    CHARGE_SUM_TOLERANCE_E,
    DISTRIBUTED_COLUMNS,
    PARITY_COLUMNS,
    PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM,
    PME_EWALD_FORCE_TOLERANCE_EV_PER_A,
    DomainLessonResultsError,
    canonical_json_sha256,
    load_domain_lesson_view,
)
from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402
from aux.plotting import plot_domain_decomposition  # noqa: E402


PLANNED_ATOMS = (3200, 6400, 12800)
STRUCTURE_SHA256_BY_ATOMS = {
    atom_count: str(index) * 64
    for index, atom_count in enumerate(PLANNED_ATOMS, start=1)
}
CELL_GRID_BY_ATOMS = {
    3_200: (2, 2, 2),
    6_400: (2, 2, 4),
    12_800: (2, 4, 4),
}
RANK_GRID_BY_ATOMS_AND_WORLD_SIZE = {
    (3_200, 2): (1, 1, 2),
    (3_200, 4): (1, 2, 2),
    (6_400, 1): (1, 1, 1),
    (6_400, 2): (1, 1, 2),
    (6_400, 4): (1, 1, 4),
    (12_800, 2): (1, 1, 2),
    (12_800, 4): (1, 2, 2),
}
CHARGE_SHA256 = "4" * 64
PME_FORCE_SHA256 = "5" * 64
EWALD_FORCE_SHA256 = "6" * 64
ORIGINAL_RESULT_SHA256 = "7" * 64
CONFIG_PATH = PART_DIR / "aux" / "domain" / "config.py"
ATOMS_PER_COMPOSITION_UNIT = DOMAIN_METHODOLOGY.atoms_per_composition_unit
PARITY_FORCE_BACKGROUND = 2.0e-7
PARITY_FORCE_MAX = 3.0e-6
PARITY_FORCE_RMS = math.sqrt(
    (
        (PLANNED_ATOMS[0] * 3 - 1) * PARITY_FORCE_BACKGROUND**2
        + PARITY_FORCE_MAX**2
    )
    / (PLANNED_ATOMS[0] * 3)
)
PARITY_FORCE_TOLERANCE = (
    DOMAIN_METHODOLOGY.parity_force_atol_ev_a
    + DOMAIN_METHODOLOGY.parity_force_rtol
)


def _capacity_rows(settings_sha256: str) -> pd.DataFrame:
    rows = []
    for atom_count in PLANNED_ATOMS:
        success = atom_count < PLANNED_ATOMS[-1]
        is_speed_reference = atom_count == PLANNED_ATOMS[1]
        rows.append(
            {
                "case_id": f"capacity-{atom_count}",
                "atom_count": atom_count,
                "molecules_per_species": atom_count // ATOMS_PER_COMPOSITION_UNIT,
                "gpus": 1,
                "success": success,
                "status": "complete" if success else "failed",
                "failure_type": "" if success else "CUDAOutOfMemoryError",
                "failure_stage": "" if success else "model_forward",
                "error": "" if success else "CUDA allocator reported out of memory",
                "elapsed_s": (8.0 if is_speed_reference else 2.0) if success else "",
                "peak_memory_bytes_max_rank": (
                    50_000_000_000
                    if is_speed_reference
                    else 60_000_000_000
                    if success
                    else 79_000_000_000
                ),
                "energy_ev": (-40.0 if is_speed_reference else -10.0) if success else "",
                "force_rms_ev_per_a": (
                    0.3 if is_speed_reference else 0.0
                )
                if success
                else "",
                "force_max_ev_per_a": (
                    1.0 if is_speed_reference else 0.0
                )
                if success
                else "",
                "structure_sha256": STRUCTURE_SHA256_BY_ATOMS[atom_count],
                "settings_sha256": settings_sha256,
                "measurement_role": "capacity",
                "measurement_kind": "cold_one_shot_partition_run_gather",
            }
        )
    return pd.DataFrame(rows, columns=CAPACITY_COLUMNS)


def _parity_rows(settings_sha256: str) -> pd.DataFrame:
    one_gpu_energy_offsets = {2: 0.5000, 4: 0.5001}
    distributed_reference_offset = one_gpu_energy_offsets[2]
    return pd.DataFrame(
        [
            {
                "case_id": (
                    f"parity-pairs-"
                    f"{PLANNED_ATOMS[0] // ATOMS_PER_COMPOSITION_UNIT:06d}-"
                    f"gpus-{gpus:02d}"
                ),
                "atom_count": 3200,
                "force_reference_gpus": 1,
                "energy_reference_gpus": 2,
                "gpus": gpus,
                "success": True,
                "status": "complete",
                "failure_type": "",
                "failure_stage": "",
                "error": "",
                "one_gpu_energy_abs_offset_ev": one_gpu_energy_offsets[gpus],
                "one_gpu_energy_abs_offset_ev_per_atom": (
                    one_gpu_energy_offsets[gpus] / 3_200
                ),
                "distributed_energy_difference_ev": abs(
                    one_gpu_energy_offsets[gpus] - distributed_reference_offset
                ),
                "distributed_energy_difference_ev_per_atom": (
                    abs(
                        one_gpu_energy_offsets[gpus]
                        - distributed_reference_offset
                    )
                    / 3_200
                ),
                "force_rms_difference_ev_per_a": PARITY_FORCE_RMS,
                "force_max_difference_ev_per_a": PARITY_FORCE_MAX,
                "energy_tolerance_ev_per_atom": (
                    DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
                ),
                "force_tolerance_ev_per_a": (
                    PARITY_FORCE_TOLERANCE
                ),
                "distributed_energy_passed": True,
                "force_passed": True,
                "parity_passed": True,
                "structure_sha256": STRUCTURE_SHA256_BY_ATOMS[3200],
                "settings_sha256": settings_sha256,
                "measurement_role": "parity",
                "measurement_kind": "cold_one_shot_partition_run_gather",
            }
            for gpus in (2, 4)
        ],
        columns=PARITY_COLUMNS,
    )


def _distributed_rows(
    settings_sha256: str,
    *,
    successful_rescue_gpus: tuple[int, ...] = (4,),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    timing_atom_count = PLANNED_ATOMS[1]
    timing_pairs = timing_atom_count // ATOMS_PER_COMPOSITION_UNIT
    for world_size in (1, 2, 4):
        elapsed_s = 8.0 / world_size
        samples_s = [
            0.90 * elapsed_s,
            0.95 * elapsed_s,
            elapsed_s,
            1.05 * elapsed_s,
            1.10 * elapsed_s,
        ]
        rows.append(
            {
                "case_id": (
                    f"steady-timing-pairs-{timing_pairs:06d}-"
                    f"gpus-{world_size:02d}"
                ),
                "atom_count": timing_atom_count,
                "molecules_per_species": timing_pairs,
                "nodes": world_size,
                "gpus": world_size,
                "ranks": world_size,
                "success": True,
                "status": "complete",
                "failure_type": "",
                "failure_stage": "",
                "error": "",
                "elapsed_s": elapsed_s,
                "warmup_count": DOMAIN_METHODOLOGY.steady_timing_warmup_count,
                "sample_count": DOMAIN_METHODOLOGY.steady_timing_sample_count,
                "elapsed_samples_s": json.dumps(samples_s, separators=(",", ":")),
                "elapsed_median_s": elapsed_s,
                "elapsed_q1_s": 0.95 * elapsed_s,
                "elapsed_q3_s": 1.05 * elapsed_s,
                "elapsed_iqr_s": 0.10 * elapsed_s,
                "peak_memory_bytes_max_rank": 50_000_000_000 // world_size,
                "owned_atoms_min_rank": timing_atom_count // world_size - 5,
                "owned_atoms_max_rank": timing_atom_count // world_size + 5,
                "spatial_grid": "x".join(
                    str(value)
                    for value in RANK_GRID_BY_ATOMS_AND_WORLD_SIZE[
                        (timing_atom_count, world_size)
                    ]
                ),
                "energy_ev": -40.0,
                "force_rms_ev_per_a": 0.3,
                "force_max_ev_per_a": 1.0,
                "structure_sha256": STRUCTURE_SHA256_BY_ATOMS[timing_atom_count],
                "settings_sha256": settings_sha256,
                "measurement_role": "steady_timing",
                "measurement_kind": "steady_partition_run_gather",
            }
        )

    rescue_atom_count = PLANNED_ATOMS[2]
    rescue_pairs = rescue_atom_count // ATOMS_PER_COMPOSITION_UNIT
    for world_size in (2, 4):
        success = world_size in successful_rescue_gpus
        elapsed_s = 16.0 / world_size
        rows.append(
            {
                "case_id": (f"rescue-pairs-{rescue_pairs:06d}-gpus-{world_size:02d}"),
                "atom_count": rescue_atom_count,
                "molecules_per_species": rescue_pairs,
                "nodes": world_size,
                "gpus": world_size,
                "ranks": world_size,
                "success": success,
                "status": "complete" if success else "failed",
                "failure_type": "" if success else "CUDAOutOfMemoryError",
                "failure_stage": "" if success else "model_forward",
                "error": "" if success else "CUDA allocator reported out of memory",
                "elapsed_s": elapsed_s if success else "",
                "warmup_count": 0 if success else "",
                "sample_count": 1 if success else "",
                "elapsed_samples_s": (
                    json.dumps([elapsed_s], separators=(",", ":"))
                    if success
                    else ""
                ),
                "elapsed_median_s": elapsed_s if success else "",
                "elapsed_q1_s": elapsed_s if success else "",
                "elapsed_q3_s": elapsed_s if success else "",
                "elapsed_iqr_s": 0.0 if success else "",
                "peak_memory_bytes_max_rank": (
                    60_000_000_000 // world_size if success else 79_000_000_000
                ),
                "owned_atoms_min_rank": (
                    rescue_atom_count // world_size - 5 if success else ""
                ),
                "owned_atoms_max_rank": (
                    rescue_atom_count // world_size + 5 if success else ""
                ),
                "spatial_grid": "x".join(
                    str(value)
                    for value in RANK_GRID_BY_ATOMS_AND_WORLD_SIZE[
                        (rescue_atom_count, world_size)
                    ]
                ),
                "energy_ev": -80.0 if success else "",
                "force_rms_ev_per_a": 0.4 if success else "",
                "force_max_ev_per_a": 1.2 if success else "",
                "structure_sha256": STRUCTURE_SHA256_BY_ATOMS[rescue_atom_count],
                "settings_sha256": settings_sha256,
                "measurement_role": "rescue",
                "measurement_kind": "cold_one_shot_partition_run_gather",
            }
        )
    return pd.DataFrame(rows, columns=DISTRIBUTED_COLUMNS)


def _identity() -> tuple[dict[str, Any], dict[str, str]]:
    methodology_record = json.loads(
        json.dumps(DOMAIN_METHODOLOGY.as_record(), allow_nan=False)
    )
    methodology_values = json.loads(
        json.dumps(
            DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
            allow_nan=False,
        )
    )
    methodology_identity = {
        "name": DOMAIN_METHODOLOGY.name,
        "version": DOMAIN_METHODOLOGY.version,
        "config_sha256": _sha256(CONFIG_PATH),
        "record": methodology_record,
        "resolved_values": methodology_values,
    }
    identity = {
        "source": {
            "repository_commit": "a" * 40,
            "repository_dirty": False,
            "toolkit_commit": "b" * 40,
            "toolkit_ops_commit": "c" * 40,
            "toolkit_version": "0.2.0",
            "domain_methodology": methodology_identity,
            "producer_files_sha256": {
                "config.py": methodology_identity["config_sha256"],
            },
        },
        "hardware": {
            "site": "Compute Lab",
            "site_source": "operator-declared",
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_memory_bytes": 80_000_000_000,
            "gpus_available": 4,
            "nodes_available": 4,
            "resource_count_source": (
                "derived from successful per-rank runtime records"
            ),
            "driver_version": "590.00",
            "cuda_version": "13.0",
            "interconnect": "InfiniBand",
            "interconnect_source": (
                "operator-declared; raw GPU topology is retained"
            ),
        },
        "settings": {
            "domain_methodology": {
                key: methodology_identity[key]
                for key in ("name", "version", "config_sha256", "resolved_values")
            },
            "model_components": [
                "AIMNet2 checkpoint base and predicted charges",
                "PME electrostatics",
                "D3(BJ) dispersion",
            ],
            "precision": "float32",
            "aimnet_checkpoint_sha256": "d" * 64,
            "d3_parameters_sha256": "e" * 64,
            "neighbor_adaptation": "never",
            "pme": {
                "cutoff_a": DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
                "mesh_safety_factor": (
                    DOMAIN_METHODOLOGY.pme_mesh_safety_factor
                ),
                "parameter_rule": (
                    "estimate_pme_parameters(accuracy, real_space_cutoff, "
                    "mesh_safety_factor)"
                ),
                "spline_order": DOMAIN_METHODOLOGY.pme_spline_order,
                "accuracy": DOMAIN_METHODOLOGY.pme_accuracy,
                "hybrid_forces": True,
                "reciprocal_mesh_distribution": "replicated_per_rank",
            },
            "ewald_reference": {
                "accuracy": DOMAIN_METHODOLOGY.ewald_reference_accuracy,
                "parameter_rule": "estimate_ewald_parameters(accuracy)",
                "scope": "fixed-charge electrostatics validation only",
            },
            "domain": {
                "api": "DomainParallel",
                "cutoff_a": DOMAIN_METHODOLOGY.d3_cutoff_a,
                "skin_a": DOMAIN_METHODOLOGY.domain_halo_skin_a,
                "compile": False,
                "require_nondegenerate": True,
                "grid_dims": DOMAIN_METHODOLOGY.domain_grid_dims,
                "rank_grid_policy": (
                    "Toolkit SpatialPartitioner derives cells_per_dim and "
                    "rank_grid from each input's actual cell shape and the "
                    "domain cutoff"
                ),
                "recorded_layout_fields": ["cells_per_dim", "rank_grid"],
                "halo_counts": "not_exposed_by_public_api",
            },
            "packmol": {
                "version": "21.2.1",
                "construction_density_g_cm3": (
                    DOMAIN_METHODOLOGY.construction_density_g_cm3
                ),
                "tolerance_a": DOMAIN_METHODOLOGY.packmol_tolerance_a,
                "precision_a": DOMAIN_METHODOLOGY.packmol_precision_a,
                "base_seed": DOMAIN_METHODOLOGY.packmol_seed,
                "periodic_boundary_check": True,
            },
            "timing_boundary": (
                "fresh DomainParallel context; synchronized partition, run, gather"
            ),
            "timing_measurement_kind": "steady_partition_run_gather",
            "timing_measurement_role": "steady_timing",
            "timing_world_sizes": list(
                DOMAIN_METHODOLOGY.steady_timing_world_sizes
            ),
            "timing_warmup_count": (
                DOMAIN_METHODOLOGY.steady_timing_warmup_count
            ),
            "timing_sample_count": (
                DOMAIN_METHODOLOGY.steady_timing_sample_count
            ),
            "timing_model_evaluations_per_workflow": (
                DOMAIN_METHODOLOGY.steady_timing_model_evaluations_per_workflow
            ),
            "timing_one_rank_run_steps": (
                DOMAIN_METHODOLOGY.steady_timing_run_steps(1)
            ),
            "timing_multi_rank_run_steps": (
                DOMAIN_METHODOLOGY.steady_timing_run_steps(
                    DOMAIN_METHODOLOGY.distributed_world_sizes[0]
                )
            ),
            "timing_summary": "median, Q1, Q3, and IQR of max-rank sample seconds",
            "timing_quartile_method": "inclusive linear interpolation",
            "timing_max_relative_iqr": (
                DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr
            ),
            "publishable_benchmark": False,
            "timing_interpretation": (
                "same-input repeated public workflows; cold capacity, parity, "
                "and rescue rows excluded from speedup"
            ),
            "parity_acceptance": {
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
            },
            "force_difference_definition": (
                "RMS and maximum absolute difference over Cartesian components"
            ),
        },
        "inputs": {
            "structures_sha256_by_atom_count": {
                str(atom_count): digest
                for atom_count, digest in STRUCTURE_SHA256_BY_ATOMS.items()
            },
            "nci_subset_sha256": "f" * 64,
            "molecule_pair": "phenol + N-methylacetamide",
            "construction_density_g_cm3": 1.0,
        },
    }
    return identity, {
        name: canonical_json_sha256(record) for name, record in identity.items()
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _electrostatics_records() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_record = {
        "status": "passed",
        "measurement_kind": "measured",
        "fixed_charges": True,
        "atom_count": PLANNED_ATOMS[0],
        "structure_sha256": STRUCTURE_SHA256_BY_ATOMS[PLANNED_ATOMS[0]],
        "charge_sum_e": 2.0e-7,
        "charge_sum_tolerance_e": CHARGE_SUM_TOLERANCE_E,
        "pme_energy_ev": -1.0,
        "ewald_energy_ev": -1.00001,
        "energy_abs_difference_ev_per_atom": 3.125e-9,
        "energy_tolerance_ev_per_atom": PME_EWALD_ENERGY_TOLERANCE_EV_PER_ATOM,
        "force_rms_difference_ev_per_a": 2.0e-5,
        "force_max_difference_ev_per_a": 8.0e-5,
        "force_tolerance_ev_per_a": PME_EWALD_FORCE_TOLERANCE_EV_PER_A,
        "charge_sha256": CHARGE_SHA256,
        "pme_force_sha256": PME_FORCE_SHA256,
        "ewald_force_sha256": EWALD_FORCE_SHA256,
        "result_file_sha256": ORIGINAL_RESULT_SHA256,
    }
    raw_record = {
        "mode": "electrostatics-validation",
        "status": "complete",
        "success": True,
        "atom_count": PLANNED_ATOMS[0],
        "input": {
            "file_sha256": STRUCTURE_SHA256_BY_ATOMS[PLANNED_ATOMS[0]],
        },
        "charges": {
            "available": True,
            "sha256": CHARGE_SHA256,
            "sum_e": 2.0e-7,
        },
        "pme": {
            "energy_ev": -1.0,
            "forces": {"sha256": PME_FORCE_SHA256},
        },
        "ewald": {
            "energy_ev": -1.00001,
            "forces": {"sha256": EWALD_FORCE_SHA256},
        },
        "comparison": {
            "absolute_energy_difference_ev_per_atom": 3.125e-9,
            "force_difference_rms_ev_a": 2.0e-5,
            "force_difference_max_norm_ev_a": 8.0e-5,
            "passed": True,
        },
        "result_file_sha256": ORIGINAL_RESULT_SHA256,
    }
    return manifest_record, raw_record


def _raw_measurement_rows(
    root: Path,
    tables: dict[str, pd.DataFrame],
    *,
    componentwise_parity_violation: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build production-shaped raw rows independently from later CSV mutations."""

    force_dir = root / "job-records" / "forces"
    force_dir.mkdir(parents=True)
    artifact_records: list[dict[str, Any]] = []

    def force_record(case_id: str, values: np.ndarray) -> dict[str, Any]:
        path = force_dir / f"{case_id}.npy"
        np.save(path, values.astype(np.float32), allow_pickle=False)
        digest = _sha256(path)
        artifact_records.append(
            {
                "role": "force-array",
                "file": path.relative_to(root).as_posix(),
                "sha256": digest,
            }
        )
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "shape": list(values.shape),
        }

    def common(row: Any, *, mode: str) -> dict[str, Any]:
        raw = {
            "case_id": str(row.case_id),
            "mode": mode,
            "measurement_role": str(row.measurement_role),
            "status": str(row.status),
            "success": bool(row.success),
            "world_size": int(row.gpus),
            "pair_count": int(row.atom_count) // ATOMS_PER_COMPOSITION_UNIT,
            "molecules_per_species": (
                int(row.atom_count) // ATOMS_PER_COMPOSITION_UNIT
            ),
            "atom_count": int(row.atom_count),
            "input": {"file_sha256": str(row.structure_sha256)},
        }
        if bool(row.success):
            elapsed_s = float(row.elapsed_s)
            samples_s = (
                json.loads(str(row.elapsed_samples_s))
                if hasattr(row, "elapsed_samples_s")
                else [elapsed_s]
            )
            warmup_count = (
                int(row.warmup_count) if hasattr(row, "warmup_count") else 0
            )
            sample_count = (
                int(row.sample_count) if hasattr(row, "sample_count") else 1
            )
            q1_s = float(row.elapsed_q1_s) if hasattr(row, "elapsed_q1_s") else elapsed_s
            q3_s = float(row.elapsed_q3_s) if hasattr(row, "elapsed_q3_s") else elapsed_s
            raw.update(
                {
                    "timing": {
                        "wall_s_max_rank": elapsed_s,
                        "samples_s_max_rank": samples_s,
                        "median_s": elapsed_s,
                        "q1_s": q1_s,
                        "q3_s": q3_s,
                        "iqr_s": q3_s - q1_s,
                        "warmup_count": warmup_count,
                        "sample_count": sample_count,
                    },
                    "memory": {
                        "max_allocated_bytes": int(row.peak_memory_bytes_max_rank)
                    },
                    "output": {
                        "energy_ev": float(row.energy_ev),
                        "forces_source_atom_order": {
                            "rms_ev_a": float(row.force_rms_ev_per_a),
                            "max_norm_ev_a": float(row.force_max_ev_per_a),
                        },
                    },
                }
            )
            if str(row.measurement_role) == "steady_timing":
                methodology = DOMAIN_METHODOLOGY
                world_size = int(row.gpus)
                initial_force_evaluations = (
                    methodology.domain_parallel_multi_rank_initial_force_evaluations
                    if world_size > 1
                    else 0
                )
                raw["timing"].update(
                    {
                        "run_steps": methodology.steady_timing_run_steps(world_size),
                        "automatic_multi_rank_force_prime": world_size > 1,
                        "automatic_initial_force_evaluations": (
                            initial_force_evaluations
                        ),
                        "model_evaluations_per_workflow": (
                            methodology.steady_timing_model_evaluations_per_workflow
                        ),
                    }
                )
        else:
            peak_memory = int(row.peak_memory_bytes_max_rank)
            raw.update(
                {
                    "failure": {
                        "type": str(row.failure_type),
                        "stage": str(row.failure_stage),
                        "message": str(row.error),
                        "is_cuda_oom": str(row.failure_type)
                        in {"CUDAOutOfMemoryError", "OutOfMemoryError"},
                    },
                    "rank_records": [
                        {
                            "memory": {"max_allocated_bytes": peak_memory},
                            "owned_atom_count": None,
                        }
                    ],
                }
            )
        return raw

    raw_rows: list[dict[str, Any]] = []
    capacity_force_by_atoms: dict[int, np.ndarray] = {}
    for row in tables["capacity"].itertuples(index=False):
        raw = common(row, mode="capacity")
        if bool(row.success):
            values = np.zeros((int(row.atom_count), 3), dtype=np.float32)
            if int(row.atom_count) == PLANNED_ATOMS[0]:
                values[1, 0] = 1.0
            capacity_force_by_atoms[int(row.atom_count)] = values
            raw["output"]["forces_source_atom_order_npy"] = force_record(
                str(row.case_id), values
            )
        raw_rows.append(raw)

    for table_index, row in enumerate(tables["parity"].itertuples(index=False)):
        difference = np.full(
            (int(row.atom_count), 3),
            PARITY_FORCE_BACKGROUND,
            dtype=np.float32,
        )
        difference[0, 0] = (
            3.0e-3 if componentwise_parity_violation else PARITY_FORCE_MAX
        )
        values = capacity_force_by_atoms[int(row.atom_count)] + difference
        stored_difference = (
            values.astype(np.float64)
            - capacity_force_by_atoms[int(row.atom_count)].astype(np.float64)
        )
        tables["parity"].loc[
            table_index,
            "force_rms_difference_ev_per_a",
        ] = float(np.sqrt(np.mean(stored_difference * stored_difference)))
        tables["parity"].loc[
            table_index,
            "force_max_difference_ev_per_a",
        ] = float(np.abs(stored_difference).max())
        raw = {
            "case_id": str(row.case_id),
            "mode": "parity",
            "measurement_role": "parity",
            "status": "complete",
            "success": True,
            "world_size": int(row.gpus),
            "pair_count": int(row.atom_count) // ATOMS_PER_COMPOSITION_UNIT,
            "molecules_per_species": (
                int(row.atom_count) // ATOMS_PER_COMPOSITION_UNIT
            ),
            "atom_count": int(row.atom_count),
            "input": {"file_sha256": str(row.structure_sha256)},
            "output": {
                "energy_ev": -10.0 + float(row.one_gpu_energy_abs_offset_ev),
                "forces_source_atom_order": {
                    "rms_ev_a": float(np.sqrt(np.mean(values.astype(float) ** 2))),
                    "max_norm_ev_a": float(
                        np.linalg.vector_norm(values.astype(float), axis=1).max()
                    ),
                },
                "forces_source_atom_order_npy": force_record(str(row.case_id), values),
            },
            "timing": {"wall_s_max_rank": 1.0},
            "memory": {"max_allocated_bytes": 1_000_000},
            "distributed": {
                "owned_atom_counts": [int(row.atom_count) // int(row.gpus)]
                * int(row.gpus),
                "cells_per_dim": list(CELL_GRID_BY_ATOMS[int(row.atom_count)]),
                "rank_grid": list(
                    RANK_GRID_BY_ATOMS_AND_WORLD_SIZE[
                        (int(row.atom_count), int(row.gpus))
                    ]
                ),
            },
        }
        raw_rows.append(raw)

    for row in tables["distributed"].itertuples(index=False):
        raw = common(
            row,
            mode=(
                "steady-timing"
                if str(row.measurement_role) == "steady_timing"
                else "distributed"
            ),
        )
        if bool(row.success):
            values = np.zeros((int(row.atom_count), 3), dtype=np.float32)
            raw["output"]["forces_source_atom_order_npy"] = force_record(
                str(row.case_id), values
            )
            raw["distributed"] = {
                "owned_atom_counts": [
                    int(row.owned_atoms_min_rank),
                    int(row.owned_atoms_max_rank),
                ],
                "cells_per_dim": list(CELL_GRID_BY_ATOMS[int(row.atom_count)]),
                "rank_grid": [
                    int(value) for value in str(row.spatial_grid).split("x")
                ],
            }
        else:
            cells_per_dim = list(CELL_GRID_BY_ATOMS[int(row.atom_count)])
            rank_grid = [
                int(value) for value in str(row.spatial_grid).split("x")
            ]
            for record in raw["rank_records"]:
                record["cells_per_dim"] = cells_per_dim
                record["rank_grid"] = rank_grid
        raw_rows.append(raw)

    return raw_rows, artifact_records


def _seal_bundle(
    root: Path,
    *,
    mutate_tables: Any | None = None,
    mutate_raw_rows: Any | None = None,
    mutate_manifest: Any | None = None,
    raw_file_text: str | None = None,
    componentwise_parity_violation: bool = False,
    mutate_identity: Any | None = None,
    rank_grid_override: dict[int, str] | None = None,
    successful_rescue_gpus: tuple[int, ...] = (4,),
) -> Path:
    root.mkdir()
    identity, identity_sha256 = _identity()
    if mutate_identity is not None:
        mutate_identity(identity)
        identity_sha256 = {
            name: canonical_json_sha256(record) for name, record in identity.items()
        }
    settings_sha256 = identity_sha256["settings"]
    raw_tables = {
        "capacity": _capacity_rows(settings_sha256),
        "parity": _parity_rows(settings_sha256),
        "distributed": _distributed_rows(
            settings_sha256,
            successful_rescue_gpus=successful_rescue_gpus,
        ),
    }
    if rank_grid_override:
        for world_size, rank_grid in rank_grid_override.items():
            selected = raw_tables["distributed"]["gpus"].eq(world_size)
            raw_tables["distributed"].loc[selected, "spatial_grid"] = rank_grid
    if componentwise_parity_violation:
        violation_rms = math.sqrt(
            (
                (PLANNED_ATOMS[0] * 3 - 1) * PARITY_FORCE_BACKGROUND**2
                + (3.0e-3) ** 2
            )
            / (PLANNED_ATOMS[0] * 3)
        )
        raw_tables["parity"].loc[:, "force_rms_difference_ev_per_a"] = violation_rms
        raw_tables["parity"].loc[:, "force_max_difference_ev_per_a"] = 3.0e-3
    raw_measurements, force_artifacts = _raw_measurement_rows(
        root,
        raw_tables,
        componentwise_parity_violation=componentwise_parity_violation,
    )
    tables = {name: table.copy(deep=True) for name, table in raw_tables.items()}
    if mutate_tables is not None:
        mutate_tables(tables)
    data: dict[str, Any] = {}
    for name, table in tables.items():
        path = root / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        data[name] = {
            "file": path.name,
            "sha256": _sha256(path),
            "row_count": len(table),
            "columns": list(table.columns),
            "planned_case_ids": table["case_id"].astype(str).tolist(),
        }
    electrostatics, raw_electrostatics = _electrostatics_records()
    raw_rows = [raw_electrostatics, *raw_measurements]
    if mutate_raw_rows is not None:
        mutate_raw_rows(raw_rows)
    raw_path = root / "raw-results.jsonl"
    if raw_file_text is None:
        _write_jsonl(raw_path, raw_rows)
    else:
        raw_path.write_text(raw_file_text, encoding="utf-8")
    structures_dir = root / "structures"
    structures_dir.mkdir()
    structure_path = structures_dir / "validation-pairs-000128.extxyz"
    structure_path.write_text("synthetic extxyz structure\n", encoding="utf-8")
    logs_dir = root / "logs"
    logs_dir.mkdir()
    case_log_path = logs_dir / "electrostatics-validation.log"
    case_log_path.write_text("synthetic completed case log\n", encoding="utf-8")
    records_dir = root / "job-records" / "capacity"
    records_dir.mkdir(parents=True)
    phase_summary_path = records_dir / "phase-summary.json"
    _write_json(
        phase_summary_path,
        {
            "schema": "alchemi.part1-domain-phase-summary.v1",
            "phase": "capacity",
            "status": "complete",
            "passed": True,
        },
    )
    artifacts = {
        "structures": [
            {
                "role": "validation",
                "pair_count": PLANNED_ATOMS[0] // ATOMS_PER_COMPOSITION_UNIT,
                "file": structure_path.relative_to(root).as_posix(),
                "sha256": _sha256(structure_path),
            }
        ],
        "case_logs": [
            {
                "case_id": "electrostatics-validation",
                "file": case_log_path.relative_to(root).as_posix(),
                "sha256": _sha256(case_log_path),
            }
        ],
        "files": [
            {
                "role": "phase-summary",
                "file": phase_summary_path.relative_to(root).as_posix(),
                "sha256": _sha256(phase_summary_path),
            },
            *force_artifacts,
        ],
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": "2026-07-24T12:34:56Z",
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "failure_policy": {
            "failed_rows_retained": True,
            "estimates_allowed": False,
            "capacity_stop_condition": "first_single_gpu_oom",
        },
        "electrostatics_validation": electrostatics,
        "selection": {
            "largest_success_pair_count": (
                PLANNED_ATOMS[1] // ATOMS_PER_COMPOSITION_UNIT
            ),
            "first_cuda_oom_pair_count": (
                PLANNED_ATOMS[2] // ATOMS_PER_COMPOSITION_UNIT
            ),
            "parity_pair_count": (
                PLANNED_ATOMS[0] // ATOMS_PER_COMPOSITION_UNIT
            ),
            "successful_rescue_gpu_counts": list(successful_rescue_gpus),
        },
        "data": data,
        "raw_results": {
            "file": raw_path.name,
            "sha256": _sha256(raw_path),
            "row_count": len(raw_rows),
        },
        "artifacts": artifacts,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    checksum_lines = [f"{_sha256(manifest_path)}  manifest.json"]
    checksum_lines.extend(
        f"{metadata['sha256']}  {metadata['file']}" for metadata in data.values()
    )
    checksum_lines.append(f"{_sha256(raw_path)}  {raw_path.name}")
    for records in artifacts.values():
        checksum_lines.extend(
            f"{record['sha256']}  {record['file']}" for record in records
        )
    (root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return root


def test_missing_results_are_explicitly_not_reported(tmp_path: Path) -> None:
    view = load_domain_lesson_view(
        tmp_path / "not-created",
        planned_atom_counts=PLANNED_ATOMS,
    )

    assert not view.available
    assert "not been reported" in view.reason
    assert view.capacity_table["atom_count"].tolist() == list(PLANNED_ATOMS)
    assert view.capacity_table["success"].isna().all()
    assert view.capacity_table["torch_peak_allocated_gb"].isna().all()
    assert view.electrostatics_table.empty
    assert view.parity_table.empty
    assert view.distributed_table.empty
    assert view.recorded_run_table.empty
    assert view.successful_case_count == 0
    assert view.failed_case_count == 0
    assert view.measured_max_atom_count is None
    with pytest.raises(DomainLessonResultsError, match="not available"):
        _ = view.takeaway


def test_complete_bundle_keeps_oom_and_returns_clear_plot_tables(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(tmp_path / "results")

    view = load_domain_lesson_view(
        root,
        planned_atom_counts=PLANNED_ATOMS,
        expected_parity_atom_count=PLANNED_ATOMS[0],
    )

    assert view.available
    assert view.reason == ""
    assert list(view.capacity_table.columns) == [
        "atom_count",
        "success",
        "world_size",
        "torch_peak_allocated_gb",
        "failure_type",
        "failure_stage",
    ]
    assert view.capacity_table["success"].tolist() == [True, True, False]
    assert view.capacity_table.iloc[-1]["failure_type"] == "CUDAOutOfMemoryError"
    assert view.capacity_table.iloc[-1]["torch_peak_allocated_gb"] == pytest.approx(
        79.0
    )
    assert view.electrostatics_table.iloc[0]["passed"]
    assert view.electrostatics_table.iloc[0]["charge_sum_e"] == pytest.approx(2.0e-7)
    assert view.electrostatics_table.iloc[0][
        "energy_abs_error_meV_per_atom"
    ] == pytest.approx(3.125e-6)
    assert list(view.parity_table.columns) == [
        "atom_count",
        "world_size",
        "one_gpu_energy_abs_offset_meV_atom",
        "distributed_energy_error_meV_atom",
        "force_max_abs_error_eV_A",
        "force_rms_error_eV_A",
        "force_passed",
        "distributed_energy_passed",
        "passed",
    ]
    assert view.parity_table["world_size"].tolist() == [2, 4]
    assert view.parity_table["passed"].all()
    assert list(view.distributed_table.columns) == [
        "atom_count",
        "world_size",
        "success",
        "measurement_role",
        "wall_time_s",
        "wall_time_q1_s",
        "wall_time_q3_s",
        "wall_time_iqr_s",
        "relative_iqr",
        "speedup_vs_1gpu",
        "parallel_efficiency",
        "torch_peak_allocated_gb",
        "owned_atoms_min",
        "owned_atoms_max",
        "spatial_grid",
        "failure_type",
        "failure_stage",
        "error",
    ]
    assert view.distributed_table["spatial_grid"].tolist() == [
        "1x1x1",
        "1x1x2",
        "1x1x4",
        "1x1x2",
        "1x2x2",
    ]
    assert view.distributed_table["world_size"].tolist() == [1, 2, 4, 2, 4]
    assert view.distributed_table.loc[
        view.distributed_table["measurement_role"].eq("steady_timing"),
        "relative_iqr",
    ].to_numpy() == pytest.approx(np.full(3, 0.10))
    assert view.distributed_table["success"].tolist() == [
        True,
        True,
        True,
        False,
        True,
    ]
    assert view.distributed_table["speedup_vs_1gpu"].iloc[:3].tolist() == [
        1.0,
        2.0,
        4.0,
    ]
    assert view.distributed_table["speedup_vs_1gpu"].iloc[3:].isna().all()
    assert view.distributed_table["parallel_efficiency"].iloc[:3].tolist() == [
        1.0,
        1.0,
        1.0,
    ]
    assert view.distributed_table["parallel_efficiency"].iloc[3:].isna().all()
    assert view.distributed_table["measurement_role"].tolist() == [
        "steady_timing",
        "steady_timing",
        "steady_timing",
        "rescue",
        "rescue",
    ]
    assert view.distributed_table["wall_time_q1_s"].iloc[:3].tolist() == [
        7.6,
        3.8,
        1.9,
    ]
    assert view.distributed_table["wall_time_q3_s"].iloc[:3].tolist() == [
        8.4,
        4.2,
        2.1,
    ]
    assert len(view.plot_data) == 8
    assert (~view.plot_data["success"]).sum() == 2
    assert set(view.failed_table["table"]) == {"capacity", "distributed"}
    failed_rescue = view.failed_table.loc[view.failed_table["table"].eq("distributed")]
    assert failed_rescue["world_size"].tolist() == [2]
    assert failed_rescue["wall_time_s"].isna().all()
    assert failed_rescue[["failure_type", "failure_stage", "error"]].to_dict(
        "records"
    ) == [
        {
            "failure_type": "CUDAOutOfMemoryError",
            "failure_stage": "model_forward",
            "error": "CUDA allocator reported out of memory",
        }
    ]
    assert view.recorded_run_table["Recorded result set"].to_dict() == {
        "Bundle created (UTC)": "2026-07-24T12:34:56Z",
        "Site": "Compute Lab",
        "GPU": "NVIDIA H100 80GB HBM3",
        "Interconnect": "InfiniBand",
        "Measured nodes / GPUs": "1 / 1, 2 / 2, 4 / 4",
        "Toolkit version": "0.2.0",
        "Toolkit Core commit": "b" * 40,
        "Toolkit-Ops commit": "c" * 40,
        "Tutorial commit": "a" * 40,
    }
    assert view.successful_case_count == 8
    assert view.failed_case_count == 2
    assert view.measured_max_atom_count == PLANNED_ATOMS[-1]
    assert view.takeaway == {
        "largest_successful_single_gpu_atoms": 6400,
        "first_single_gpu_oom_atoms": 12800,
        "rescue_successful_gpu_counts": (4,),
        "all_one_gpu_force_checks_passed": True,
        "all_distributed_energy_checks_passed": True,
        "timed_one_gpu_force_checks_passed": True,
        "timed_distributed_energy_checks_passed": True,
        "rescue_output_comparison_count": 0,
        "speedup_by_gpu": ((2, 2.0), (4, 4.0)),
        "parallel_efficiency_by_gpu": ((2, 1.0), (4, 1.0)),
    }


def test_complete_capacity_sweep_may_stop_at_first_oom_before_ladder_end(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(tmp_path / "early-oom")

    view = load_domain_lesson_view(
        root,
        planned_atom_counts=(*PLANNED_ATOMS, 25_600, 51_200),
    )

    assert view.available
    assert view.capacity_table["atom_count"].tolist() == list(PLANNED_ATOMS)
    assert view.capacity_table.iloc[-1]["failure_type"] == "CUDAOutOfMemoryError"


def test_loader_rejects_an_estimated_row(tmp_path: Path) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["distributed"].loc[0, "measurement_kind"] = "estimated"

    root = _seal_bundle(tmp_path / "estimated", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="measurement kind"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("elapsed_s", 7.5),
        ("elapsed_q1_s", 1.0),
    ),
)
def test_loader_rejects_stale_distributed_timing_statistics(
    tmp_path: Path,
    column: str,
    value: float,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["distributed"].loc[0, column] = value

    root = _seal_bundle(
        tmp_path / f"stale-{column.replace('_', '-')}",
        mutate_tables=mutate,
    )

    with pytest.raises(
        DomainLessonResultsError,
        match="timing statistics do not match samples",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("case_kind", ("parity", "steady-timing", "rescue"))
def test_loader_rejects_missing_required_four_gpu_row(
    tmp_path: Path,
    case_kind: str,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        table_name = "parity" if case_kind == "parity" else "distributed"
        table = tables[table_name]
        selected = table["case_id"].astype(str).str.startswith(f"{case_kind}-")
        selected &= table["gpus"].eq(4)
        assert selected.sum() == 1
        tables[table_name] = table.loc[~selected].reset_index(drop=True)

    root = _seal_bundle(
        tmp_path / f"missing-{case_kind}-four-gpu",
        mutate_tables=mutate,
    )

    expected_label = case_kind.replace("-", " ")
    with pytest.raises(DomainLessonResultsError, match=f"{expected_label}.*GPU"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("column", ("nodes", "ranks"))
def test_loader_rejects_mismatched_distributed_topology(
    tmp_path: Path,
    column: str,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(
            "steady-timing-pairs-000256-gpus-02"
        )
        assert selected.sum() == 1
        tables["distributed"].loc[selected, column] = 1

    root = _seal_bundle(
        tmp_path / f"mismatched-{column}",
        mutate_tables=mutate,
    )

    with pytest.raises(
        DomainLessonResultsError,
        match="nodes == gpus == ranks",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_accepts_another_valid_rectangular_rank_grid(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(
        tmp_path / "alternate-valid-rank-grid",
        rank_grid_override={4: "2x2x1"},
    )

    view = load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)

    four_gpu = view.distributed_table.loc[
        view.distributed_table["world_size"].eq(4),
        "spatial_grid",
    ]
    assert four_gpu.tolist() == ["2x2x1", "2x2x1"]


def test_loader_rejects_rank_grid_larger_than_cell_grid(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(
        tmp_path / "wrong-rank-grid",
        rank_grid_override={4: "4x1x1"},
    )

    with pytest.raises(DomainLessonResultsError, match="rank_grid exceeds"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("spatial_grid", ("2x2", "2x0x2", "2x2x2"))
def test_loader_rejects_invalid_three_dimensional_rank_grid(
    tmp_path: Path,
    spatial_grid: str,
) -> None:
    root = _seal_bundle(
        tmp_path / f"invalid-rank-grid-{spatial_grid}",
        rank_grid_override={4: spatial_grid},
    )

    with pytest.raises(DomainLessonResultsError, match="spatial_grid"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_noninteger_recorded_cell_grid(tmp_path: Path) -> None:
    case_id = "steady-timing-pairs-000256-gpus-04"

    def mutate(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item.get("case_id") == case_id)
        row["distributed"]["cells_per_dim"] = [2, 2, 4.0]

    root = _seal_bundle(
        tmp_path / "noninteger-cell-grid",
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="positive integers"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_rank_grid_that_does_not_divide_cell_grid(
    tmp_path: Path,
) -> None:
    case_id = "steady-timing-pairs-000256-gpus-04"

    def mutate(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item.get("case_id") == case_id)
        row["distributed"]["cells_per_dim"] = [3, 2, 4]
        row["distributed"]["rank_grid"] = [2, 2, 1]

    def mutate_tables(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(case_id)
        tables["distributed"].loc[selected, "spatial_grid"] = "2x2x1"

    root = _seal_bundle(
        tmp_path / "rank-grid-does-not-divide",
        mutate_tables=mutate_tables,
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="does not divide"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_allows_failed_row_that_never_reached_layout(tmp_path: Path) -> None:
    case_id = "rescue-pairs-000512-gpus-02"

    def mutate_tables(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(case_id)
        assert selected.sum() == 1
        tables["distributed"].loc[selected, "spatial_grid"] = ""

    def mutate_raw(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item.get("case_id") == case_id)
        for record in row["rank_records"]:
            record.pop("cells_per_dim")
            record.pop("rank_grid")

    root = _seal_bundle(
        tmp_path / "failure-before-layout",
        mutate_tables=mutate_tables,
        mutate_raw_rows=mutate_raw,
    )

    view = load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)

    failed = view.distributed_table.loc[
        view.distributed_table["world_size"].eq(2)
        & view.distributed_table["measurement_role"].eq("rescue")
    ]
    assert len(failed) == 1
    assert not bool(failed.iloc[0]["success"])
    assert failed.iloc[0]["spatial_grid"] == ""


@pytest.mark.parametrize("gpus", (2, 4))
def test_loader_requires_each_parity_row_to_succeed(
    tmp_path: Path,
    gpus: int,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        table = tables["parity"]
        selected = table["gpus"].eq(gpus)
        assert selected.sum() == 1
        table.loc[selected, "success"] = False
        table.loc[selected, "status"] = "failed"
        table.loc[selected, "failure_type"] = "RuntimeError"
        table.loc[selected, "failure_stage"] = "distributed_model"
        table.loc[selected, "error"] = "synthetic parity evaluation failure"
        table.loc[
            selected,
            [
                "one_gpu_energy_abs_offset_ev",
                "one_gpu_energy_abs_offset_ev_per_atom",
                "distributed_energy_difference_ev",
                "distributed_energy_difference_ev_per_atom",
                "force_rms_difference_ev_per_a",
                "force_max_difference_ev_per_a",
                "energy_tolerance_ev_per_atom",
                "force_tolerance_ev_per_a",
            ],
        ] = float("nan")
        table.loc[selected, "distributed_energy_passed"] = False
        table.loc[selected, "force_passed"] = False
        table.loc[selected, "parity_passed"] = False

    root = _seal_bundle(
        tmp_path / f"failed-parity-{gpus}-gpu",
        mutate_tables=mutate,
    )

    with pytest.raises(
        DomainLessonResultsError,
        match="all declared multi-GPU parity measurements must succeed and pass",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("gpus", (2, 4))
def test_loader_requires_each_successful_parity_row_to_pass(
    tmp_path: Path,
    gpus: int,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["parity"]["gpus"].eq(gpus)
        assert selected.sum() == 1
        tables["parity"].loc[selected, "parity_passed"] = False

    root = _seal_bundle(
        tmp_path / f"not-passed-parity-{gpus}-gpu",
        mutate_tables=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="parity rows must pass"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_every_steady_timing_row_to_succeed(tmp_path: Path) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        table = tables["distributed"]
        selected = table["case_id"].eq(
            "steady-timing-pairs-000256-gpus-04"
        )
        assert selected.sum() == 1
        table.loc[selected, "success"] = False
        table.loc[selected, "status"] = "failed"
        table.loc[selected, "failure_type"] = "RuntimeError"
        table.loc[selected, "failure_stage"] = "model_forward"
        table.loc[selected, "error"] = "synthetic failed timing row"

    root = _seal_bundle(tmp_path / "failed-steady-timing", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="steady-timing.*succeed"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("world_size", (1, 2, 4))
def test_loader_rejects_unstable_steady_timing_row(
    tmp_path: Path,
    world_size: int,
) -> None:
    def timing_values(median_s: float) -> dict[str, Any]:
        return {
            "samples": [
                0.80 * median_s,
                0.85 * median_s,
                median_s,
                1.15 * median_s,
                1.20 * median_s,
            ],
            "q1": 0.85 * median_s,
            "q3": 1.15 * median_s,
            "iqr": 0.30 * median_s,
        }

    def mutate_tables(tables: dict[str, pd.DataFrame]) -> None:
        table = tables["distributed"]
        selected = table["case_id"].eq(
            f"steady-timing-pairs-000256-gpus-{world_size:02d}"
        )
        assert selected.sum() == 1
        median_s = float(table.loc[selected, "elapsed_median_s"].iloc[0])
        values = timing_values(median_s)
        table.loc[selected, "elapsed_samples_s"] = json.dumps(
            values["samples"],
            separators=(",", ":"),
        )
        table.loc[selected, "elapsed_q1_s"] = values["q1"]
        table.loc[selected, "elapsed_q3_s"] = values["q3"]
        table.loc[selected, "elapsed_iqr_s"] = values["iqr"]

    def mutate_raw(rows: list[dict[str, Any]]) -> None:
        row = next(
            item
            for item in rows
            if item.get("measurement_role") == "steady_timing"
            and item.get("world_size") == world_size
        )
        median_s = float(row["timing"]["median_s"])
        values = timing_values(median_s)
        row["timing"]["samples_s_max_rank"] = values["samples"]
        row["timing"]["q1_s"] = values["q1"]
        row["timing"]["q3_s"] = values["q3"]
        row["timing"]["iqr_s"] = values["iqr"]

    root = _seal_bundle(
        tmp_path / f"unstable-{world_size}",
        mutate_tables=mutate_tables,
        mutate_raw_rows=mutate_raw,
    )

    with pytest.raises(DomainLessonResultsError, match="too variable to report"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("world_size", (2, 4))
def test_loader_checks_output_agreement_on_the_actual_timing_input(
    tmp_path: Path,
    world_size: int,
) -> None:
    case_id = f"steady-timing-pairs-000256-gpus-{world_size:02d}"

    def mutate_tables(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(case_id)
        assert selected.sum() == 1
        tables["distributed"].loc[selected, "energy_ev"] = -39.0

    def mutate_raw(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item.get("case_id") == case_id)
        row["output"]["energy_ev"] = -39.0

    root = _seal_bundle(
        tmp_path / f"timed-output-{world_size}",
        mutate_tables=mutate_tables,
        mutate_raw_rows=mutate_raw,
    )

    with pytest.raises(DomainLessonResultsError, match="fails energy agreement"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_compares_multiple_successful_oom_retries(tmp_path: Path) -> None:
    case_id = "rescue-pairs-000512-gpus-04"

    def mutate_tables(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(case_id)
        assert selected.sum() == 1
        tables["distributed"].loc[selected, "energy_ev"] = -78.0

    def mutate_raw(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item.get("case_id") == case_id)
        row["output"]["energy_ev"] = -78.0

    root = _seal_bundle(
        tmp_path / "rescue-output-disagreement",
        mutate_tables=mutate_tables,
        mutate_raw_rows=mutate_raw,
        successful_rescue_gpus=(2, 4),
    )

    with pytest.raises(
        DomainLessonResultsError,
        match="successful rescue row.*fails energy agreement",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_at_least_one_successful_rescue(tmp_path: Path) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        table = tables["distributed"]
        selected = table["case_id"].astype(str).str.startswith("rescue-")
        assert selected.sum() == 2
        table.loc[selected, "success"] = False
        table.loc[selected, "status"] = "failed"
        table.loc[selected, "failure_type"] = "CUDAOutOfMemoryError"
        table.loc[selected, "failure_stage"] = "model_forward"
        table.loc[selected, "error"] = "CUDA allocator reported out of memory"
        table.loc[
            selected,
                [
                    "elapsed_s",
                    "warmup_count",
                    "sample_count",
                    "elapsed_samples_s",
                    "elapsed_median_s",
                    "elapsed_q1_s",
                    "elapsed_q3_s",
                    "elapsed_iqr_s",
                    "owned_atoms_min_rank",
                "owned_atoms_max_rank",
                "energy_ev",
                "force_rms_ev_per_a",
                "force_max_ev_per_a",
            ],
        ] = ""

    root = _seal_bundle(tmp_path / "no-successful-rescue", mutate_tables=mutate)

    with pytest.raises(
        DomainLessonResultsError,
        match="at least one rescue.*succeed",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_rescue_size_to_equal_first_capacity_oom(
    tmp_path: Path,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["selection"]["first_cuda_oom_pair_count"] = (
            PLANNED_ATOMS[1] // ATOMS_PER_COMPOSITION_UNIT
        )

    root = _seal_bundle(tmp_path / "wrong-rescue-selection", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="first CUDA OOM"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_steady_timing_size_to_equal_largest_capacity_success(
    tmp_path: Path,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["selection"]["largest_success_pair_count"] = (
            PLANNED_ATOMS[0] // ATOMS_PER_COMPOSITION_UNIT
        )

    root = _seal_bundle(
        tmp_path / "wrong-steady-timing-selection",
        mutate_manifest=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="largest successful"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_measured_fixed_charge_electrostatics_check(
    tmp_path: Path,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["electrostatics_validation"]["fixed_charges"] = False

    root = _seal_bundle(tmp_path / "changed-charges", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="same fixed charge"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_post_hoc_solver_tolerance_change(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["electrostatics_validation"]["force_tolerance_ev_per_a"] = 1.0

    root = _seal_bundle(tmp_path / "changed-tolerance", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="predeclared"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_resealed_stale_methodology_values(tmp_path: Path) -> None:
    def mutate(identity: dict[str, Any]) -> None:
        for location in (
            identity["source"]["domain_methodology"]["resolved_values"],
            identity["settings"]["domain_methodology"]["resolved_values"],
        ):
            location["pme_realspace_cutoff_a"] = 5.0

    root = _seal_bundle(
        tmp_path / "stale-methodology",
        mutate_identity=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="current methodology"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_replicated_pme_declaration(tmp_path: Path) -> None:
    def mutate(identity: dict[str, Any]) -> None:
        identity["settings"]["pme"]["reciprocal_mesh_distribution"] = "sharded"

    root = _seal_bundle(
        tmp_path / "wrong-pme-distribution",
        mutate_identity=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="settings.pme"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_raw_results_metadata(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest.pop("raw_results")

    root = _seal_bundle(tmp_path / "missing-raw-metadata", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="manifest.raw_results"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_host_paths_in_raw_results(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["input"]["path"] = "/shared/run/input.extxyz"

    root = _seal_bundle(
        tmp_path / "raw-host-path",
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="host path"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_declared_artifacts(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest.pop("artifacts")

    root = _seal_bundle(tmp_path / "missing-artifacts", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="manifest.artifacts"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize("key", ("structures", "case_logs", "files"))
def test_loader_requires_every_artifact_list(tmp_path: Path, key: str) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["artifacts"].pop(key)

    root = _seal_bundle(tmp_path / f"missing-{key}", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match=key):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "missing-structure")
    (root / "structures" / "validation-pairs-000128.extxyz").unlink()

    with pytest.raises(DomainLessonResultsError, match="missing data file"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_changed_declared_artifact(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "changed-log")
    path = root / "logs" / "electrostatics-validation.log"
    path.write_text("changed after the bundle was sealed\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="SHA-256 mismatch"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_empty_generic_artifact_list(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["artifacts"]["files"] = []

    root = _seal_bundle(tmp_path / "empty-generic-files", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="files must be a nonempty"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_generic_artifact_path_traversal(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["artifacts"]["files"][0]["file"] = "../outside.json"

    root = _seal_bundle(tmp_path / "generic-traversal", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="stay inside the bundle"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_generic_artifact_declared_twice(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        log = manifest["artifacts"]["case_logs"][0]
        manifest["artifacts"]["files"][0] = {
            "role": "duplicate-log",
            "file": log["file"],
            "sha256": log["sha256"],
        }

    root = _seal_bundle(tmp_path / "duplicate-generic-file", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="duplicate"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_generic_artifact_missing_from_checksum_index(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(tmp_path / "missing-generic-checksum")
    checksum_path = root / "SHA256SUMS"
    retained = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if "job-records/capacity/phase-summary.json" not in line
    ]
    checksum_path.write_text("\n".join(retained) + "\n", encoding="utf-8")

    with pytest.raises(
        DomainLessonResultsError,
        match="checksum index and manifest disagree",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_artifact_missing_from_checksum_index(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "missing-artifact-checksum")
    checksum_path = root / "SHA256SUMS"
    retained = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if "structures/validation-pairs-000128.extxyz" not in line
    ]
    checksum_path.write_text("\n".join(retained) + "\n", encoding="utf-8")

    with pytest.raises(
        DomainLessonResultsError,
        match="checksum index and manifest disagree",
    ):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_undeclared_artifact_file(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "undeclared-file")
    (root / "logs" / "unlisted.log").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="undeclared files"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_undeclared_checksum_entry(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "undeclared-checksum")
    path = root / "logs" / "unlisted.log"
    path.write_text("not declared\n", encoding="utf-8")
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        checksum_path.read_text(encoding="utf-8")
        + f"{_sha256(path)}  logs/unlisted.log\n",
        encoding="utf-8",
    )

    with pytest.raises(DomainLessonResultsError, match="lists undeclared files"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_electrostatics_measurement_hashes(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["electrostatics_validation"].pop("charge_sha256")

    root = _seal_bundle(tmp_path / "missing-charge-hash", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="charge_sha256"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize(
    "field",
    (
        "charge_sha256",
        "pme_force_sha256",
        "ewald_force_sha256",
        "result_file_sha256",
    ),
)
def test_loader_rejects_invalid_electrostatics_measurement_hash(
    tmp_path: Path,
    field: str,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["electrostatics_validation"][field] = "not-a-sha256"

    root = _seal_bundle(tmp_path / field, mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="valid SHA-256"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize(
    ("path", "label"),
    (
        (("charges", "sha256"), "charge"),
        (("pme", "forces", "sha256"), "PME force"),
        (("ewald", "forces", "sha256"), "Ewald force"),
        (("result_file_sha256",), "original result file"),
    ),
)
def test_loader_cross_checks_every_raw_electrostatics_hash(
    tmp_path: Path,
    path: tuple[str, ...],
    label: str,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        value: dict[str, Any] = rows[0]
        for key in path[:-1]:
            value = value[key]
        value[path[-1]] = "8" * 64

    root = _seal_bundle(
        tmp_path / label.replace(" ", "-").lower(),
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match=f"{label} SHA-256"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


@pytest.mark.parametrize(
    ("path", "manifest_field"),
    (
        (("charges", "sum_e"), "charge_sum_e"),
        (("pme", "energy_ev"), "pme_energy_ev"),
        (("ewald", "energy_ev"), "ewald_energy_ev"),
        (
            ("comparison", "absolute_energy_difference_ev_per_atom"),
            "energy_abs_difference_ev_per_atom",
        ),
        (
            ("comparison", "force_difference_rms_ev_a"),
            "force_rms_difference_ev_per_a",
        ),
        (
            ("comparison", "force_difference_max_norm_ev_a"),
            "force_max_difference_ev_per_a",
        ),
    ),
)
def test_loader_cross_checks_every_raw_electrostatics_value(
    tmp_path: Path,
    path: tuple[str, ...],
    manifest_field: str,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        value: dict[str, Any] = rows[0]
        for key in path[:-1]:
            value = value[key]
        value[path[-1]] = float(value[path[-1]]) + 0.25

    root = _seal_bundle(tmp_path / manifest_field, mutate_raw_rows=mutate)

    with pytest.raises(DomainLessonResultsError, match=manifest_field):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_raw_result_row_count_mismatch(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["raw_results"]["row_count"] += 1

    root = _seal_bundle(tmp_path / "wrong-raw-count", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="row count"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_the_declared_raw_results_filename(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["raw_results"]["file"] = "measurements.jsonl"

    root = _seal_bundle(tmp_path / "wrong-raw-name", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="raw-results.jsonl"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_invalid_raw_results_json(tmp_path: Path) -> None:
    root = _seal_bundle(
        tmp_path / "invalid-raw-json",
        raw_file_text='{"mode": "electrostatics-validation"\n',
    )

    with pytest.raises(DomainLessonResultsError, match="invalid raw results JSON"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_changed_raw_results_after_sealing(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "changed-raw-results")
    path = root / "raw-results.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="SHA-256 mismatch"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_capacity_csv_value_that_disagrees_with_raw_measurement(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["capacity"].loc[0, "energy_ev"] = -999.0

    root = _seal_bundle(tmp_path / "capacity-raw-mismatch", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="capacity.*raw"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_distributed_csv_value_that_disagrees_with_raw_measurement(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        selected = tables["distributed"]["case_id"].eq(
            "steady-timing-pairs-000256-gpus-04"
        )
        tables["distributed"].loc[selected, "energy_ev"] = -999.0

    root = _seal_bundle(tmp_path / "distributed-raw-mismatch", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="distributed.*raw"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_complete_raw_measurement_case_set(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[:] = [
            row
            for row in rows
            if row.get("case_id")
            != "steady-timing-pairs-000256-gpus-04"
        ]

    root = _seal_bundle(
        tmp_path / "missing-raw-case",
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="complete CSV case set"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_one_raw_electrostatics_row(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows.append(json.loads(json.dumps(rows[0])))

    root = _seal_bundle(
        tmp_path / "duplicate-electrostatics-row",
        mutate_raw_rows=mutate,
    )

    with pytest.raises(DomainLessonResultsError, match="exactly one"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_missing_planned_capacity_row(tmp_path: Path) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["capacity"] = tables["capacity"].iloc[[0, 2]].reset_index(drop=True)

    root = _seal_bundle(tmp_path / "missing-row", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="planned_atom_counts"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_requires_measured_oom_as_last_capacity_row(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        row = tables["capacity"].iloc[-1]
        tables["capacity"].loc[tables["capacity"].index[-1], "success"] = True
        tables["capacity"].loc[tables["capacity"].index[-1], "status"] = "complete"
        for column in ("failure_type", "failure_stage", "error"):
            tables["capacity"].loc[tables["capacity"].index[-1], column] = ""
        tables["capacity"].loc[
            tables["capacity"].index[-1],
            ["elapsed_s", "energy_ev", "force_rms_ev_per_a", "force_max_ev_per_a"],
        ] = [4.0, -20.0, 0.3, 0.9]
        assert row["atom_count"] == PLANNED_ATOMS[-1]

    root = _seal_bundle(tmp_path / "no-oom", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="single-GPU OOM"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_parity_outside_declared_tolerance(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["parity"].loc[0, "force_max_difference_ev_per_a"] = 1.0e-2

    root = _seal_bundle(tmp_path / "bad-parity", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="force.*difference"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_parity_energy_outside_per_atom_tolerance(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        atom_count = int(tables["parity"].loc[0, "atom_count"])
        error_per_atom = (
            2.0 * DOMAIN_METHODOLOGY.parity_energy_tolerance_ev_per_atom
        )
        tables["parity"].loc[
            0,
            "distributed_energy_difference_ev_per_atom",
        ] = error_per_atom
        tables["parity"].loc[0, "distributed_energy_difference_ev"] = (
            error_per_atom * atom_count
        )

    root = _seal_bundle(tmp_path / "bad-parity-energy", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="energy difference"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_negative_absolute_parity_energy_difference(
    tmp_path: Path,
) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["parity"].loc[0, "distributed_energy_difference_ev"] = -1.0

    root = _seal_bundle(tmp_path / "negative-parity-energy", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="non-negative"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_recomputes_componentwise_force_parity_from_raw_arrays(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(
        tmp_path / "componentwise-force-failure",
        componentwise_parity_violation=True,
    )

    with pytest.raises(DomainLessonResultsError, match="componentwise force parity"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_loader_rejects_unexpected_manifest_parity_size(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["selection"]["parity_pair_count"] = 256

    root = _seal_bundle(tmp_path / "wrong-parity-selection", mutate_manifest=mutate)

    with pytest.raises(DomainLessonResultsError, match="manifest parity size"):
        load_domain_lesson_view(
            root,
            planned_atom_counts=PLANNED_ATOMS,
            expected_parity_atom_count=PLANNED_ATOMS[0],
        )


def test_loader_rejects_invalid_manifest_parity_size(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["selection"]["parity_pair_count"] = "many"

    root = _seal_bundle(tmp_path / "invalid-parity-selection", mutate_manifest=mutate)

    with pytest.raises(
        DomainLessonResultsError,
        match="parity 1:1 composition count must be a positive integer",
    ):
        load_domain_lesson_view(
            root,
            planned_atom_counts=PLANNED_ATOMS,
            expected_parity_atom_count=PLANNED_ATOMS[0],
        )


def test_loader_rejects_unexpected_parity_row_size(tmp_path: Path) -> None:
    def mutate(tables: dict[str, pd.DataFrame]) -> None:
        tables["parity"].loc[0, "atom_count"] = PLANNED_ATOMS[1]
        tables["parity"].loc[0, "structure_sha256"] = STRUCTURE_SHA256_BY_ATOMS[
            PLANNED_ATOMS[1]
        ]

    root = _seal_bundle(tmp_path / "wrong-parity-row", mutate_tables=mutate)

    with pytest.raises(DomainLessonResultsError, match="parity rows"):
        load_domain_lesson_view(
            root,
            planned_atom_counts=PLANNED_ATOMS,
            expected_parity_atom_count=PLANNED_ATOMS[0],
        )


def test_loader_rejects_changed_csv_after_sealing(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "tampered")
    path = root / "distributed.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="SHA-256 mismatch"):
        load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)


def test_not_reported_manifest_needs_no_fake_measurements(tmp_path: Path) -> None:
    root = tmp_path / "placeholder"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "schema": BUNDLE_SCHEMA,
            "status": "not_reported",
            "reason": "H100 run is scheduled but has not finished.",
        },
    )

    view = load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)

    assert not view.available
    assert view.reason == "H100 run is scheduled but has not finished."
    assert view.capacity_table["success"].isna().all()


def test_domain_plot_labels_repeated_workflow_time(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    root = _seal_bundle(tmp_path / "plot")
    view = load_domain_lesson_view(root, planned_atom_counts=PLANNED_ATOMS)

    figure, axes = plot_domain_decomposition(
        view.capacity_table,
        view.distributed_table,
    )
    try:
        assert axes[1].get_title() == "Same-input DomainParallel time"
        assert axes[1].get_ylabel() == "partition → 2 evaluations → gather / s"
    finally:
        plt.close(figure)
