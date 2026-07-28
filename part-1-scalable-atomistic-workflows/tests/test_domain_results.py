"""Tests for the fixed-input H100 domain-decomposition result reader."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402
from aux.domain.display import domain_agreement_display_table  # noqa: E402
from aux.domain.results import (  # noqa: E402
    BUNDLE_SCHEMA,
    DISTRIBUTED_COLUMNS,
    PLOT_COLUMNS,
    DomainLessonResultsError,
    canonical_json_sha256,
    load_domain_lesson_view,
)


CONFIG_PATH = PART_DIR / "aux" / "domain" / "config.py"
FIXED_PAIRS = DOMAIN_METHODOLOGY.fixed_molecules_per_species
FIXED_ATOMS = FIXED_PAIRS * DOMAIN_METHODOLOGY.atoms_per_composition_unit
ELECTROSTATICS_PAIRS = (
    DOMAIN_METHODOLOGY.electrostatics_validation_molecules_per_species
)
ELECTROSTATICS_ATOMS = (
    ELECTROSTATICS_PAIRS * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
WORLD_SIZES = tuple(DOMAIN_METHODOLOGY.campaign_world_sizes)
RUN_ID = "part1-fixed-evaluation-test"
REPOSITORY_COMMIT = "a" * 40
TOOLKIT_COMMIT = "b" * 40
OPS_COMMIT = "c" * 40
INPUT_TENSOR_SHA256 = "1" * 64
CHARGE_SHA256 = "3" * 64
ENERGY_PASSES_BY_WORLD = {
    1: (-1250.0, -1244.5, -1239.0),
    2: (-1234.002, -1234.0, -1233.998),
    4: (-1234.001, -1233.999, -1233.997),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _force_summary(array: np.ndarray, *, include_hash: bool = True) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    magnitudes = np.linalg.norm(values, axis=1)
    record: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sum_vector_ev_a": values.sum(axis=0).tolist(),
        "sum_abs_ev_a": float(np.abs(values).sum()),
        "sum_squares_ev2_a2": float((values * values).sum()),
        "rms_ev_a": float(np.sqrt(np.mean(values * values))),
        "max_norm_ev_a": float(magnitudes.max()),
        "finite": True,
    }
    if include_hash:
        record["sha256"] = "4" * 64
    return record


def _charge_record(
    atom_count: int,
    *,
    charge_sum: float,
) -> dict[str, Any]:
    return {
        "available": True,
        "values": "summarized",
        "dtype": "float32",
        "target_sum_e": 0.0,
        "shape": [atom_count, 1],
        "sha256": CHARGE_SHA256,
        "finite": True,
        "sum_e": charge_sum,
        "residual_e": charge_sum,
        "abs_residual_per_atom": abs(charge_sum) / atom_count,
        "sum_abs_e": 100.0,
        "max_abs_e": 0.2,
        "reason": "The exact float32 charge tensor used by PME was summarized.",
    }


def _unavailable_charge_record() -> dict[str, Any]:
    return {
        "available": False,
        "values": None,
        "dtype": None,
        "target_sum_e": 0.0,
        "shape": None,
        "sha256": None,
        "finite": None,
        "sum_e": None,
        "residual_e": None,
        "abs_residual_per_atom": None,
        "sum_abs_e": None,
        "max_abs_e": None,
        "reason": (
            "Toolkit 0.2 returns energy and forces from the distributed "
            "pipeline; the one-GPU row records the predicted charges."
        ),
    }


def _methodology_record(pair_count: int) -> dict[str, Any]:
    return {
        "source": DOMAIN_METHODOLOGY.as_record(),
        "source_file": {
            "path": "part-1-scalable-atomistic-workflows/aux/domain/config.py",
            "sha256": _sha256(CONFIG_PATH),
            "size_bytes": CONFIG_PATH.stat().st_size,
        },
        "resolved_values": DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
        "case_molecules_per_species": pair_count,
    }


def _raw_source() -> dict[str, Any]:
    return {
        "repository_commit": REPOSITORY_COMMIT,
        "repository_dirty": False,
        "toolkit_core_commit": TOOLKIT_COMMIT,
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_version": "0.2.0",
        "domain_methodology_name": DOMAIN_METHODOLOGY.name,
        "domain_methodology_version": DOMAIN_METHODOLOGY.version,
        "domain_methodology_config_sha256": _sha256(CONFIG_PATH),
    }


def _manifest_source() -> dict[str, Any]:
    return {
        "tutorial_commit": REPOSITORY_COMMIT,
        "toolkit_core_commit": TOOLKIT_COMMIT,
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_version": "0.2.0",
        "nci_subset_sha256": "f" * 64,
        "aimnet_checkpoint": "aimnet2-wb97m-d3_0",
        "aimnet_checkpoint_sha256": "d" * 64,
        "d3_parameter_sha256": "e" * 64,
    }


def _manifest_methodology() -> dict[str, Any]:
    return {
        "schema": DOMAIN_METHODOLOGY.schema,
        "name": DOMAIN_METHODOLOGY.name,
        "version": DOMAIN_METHODOLOGY.version,
        "path": "part-1-scalable-atomistic-workflows/aux/domain/config.py",
        "sha256": _sha256(CONFIG_PATH),
    }


def _manifest_settings(structure_sha256: str) -> dict[str, Any]:
    return {
        "methodology": DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
        "model": {
            "aimnet_checkpoint": "aimnet2-wb97m-d3_0",
            "aimnet_compile_model": False,
            "pme_cutoff_a": DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
            "pme_mesh_safety_factor": DOMAIN_METHODOLOGY.pme_mesh_safety_factor,
            "pme_spline_order": DOMAIN_METHODOLOGY.pme_spline_order,
            "pme_accuracy": DOMAIN_METHODOLOGY.pme_accuracy,
            "ewald_reference_accuracy": (DOMAIN_METHODOLOGY.ewald_reference_accuracy),
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
        },
        "input_structure_sha256": structure_sha256,
    }


def _force_arrays() -> dict[int, np.ndarray]:
    reference = np.full((FIXED_ATOMS, 3), 0.1, dtype=np.float32)
    return {
        1: reference,
        2: reference + np.float32(1.0e-4),
        4: reference + np.float32(2.0e-4),
    }


def _distributed_raw_row(
    *,
    world_size: int,
    structure_sha256: str,
    force_path: str,
    force_sha256: str,
    force_array: np.ndarray,
) -> dict[str, Any]:
    times = {
        1: [3.0, 3.1, 2.9],
        2: [1.8, 1.7, 1.9],
        4: [1.2, 1.1, 1.3],
    }[world_size]
    pass_energies = ENERGY_PASSES_BY_WORLD[world_size]
    final_energy = pass_energies[-1]
    rank_grids = {
        1: [1, 1, 1],
        2: [1, 1, 2],
        4: [1, 2, 2],
    }
    owned = [FIXED_ATOMS // world_size] * world_size
    overall_displacement = world_size * 1.0e-7
    pass_displacements = [
        overall_displacement * 0.25,
        overall_displacement * 0.5,
        overall_displacement * 0.75,
    ]
    energy_dtype = DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(world_size)
    pass_summary = _force_summary(force_array, include_hash=False)
    final_summary = _force_summary(force_array)
    measured_passes = [
        {
            "pass_index": index,
            "energy_ev": pass_energies[index - 1],
            "energy_ev_per_atom": pass_energies[index - 1] / FIXED_ATOMS,
            "energy_dtype": energy_dtype,
            "forces": copy.deepcopy(pass_summary),
            "maximum_minimum_image_displacement_a": (pass_displacements[index - 1]),
        }
        for index in range(1, 4)
    ]
    allocated = [
        [10_000_000_000 // world_size + index * 1000] * world_size for index in range(3)
    ]
    reserved = [[value + 1_000_000 for value in values] for values in allocated]
    return {
        "schema": "alchemi.part1-domain-case.v5",
        "created_utc": "2026-07-27T12:00:00Z",
        "run_id": RUN_ID,
        "case_id": f"fixed-evaluation-pairs-{FIXED_PAIRS:06d}-gpus-{world_size:02d}",
        "mode": "distributed",
        "measurement_role": "fixed_evaluation",
        "status": "complete",
        "success": True,
        "world_size": world_size,
        "pair_count": FIXED_PAIRS,
        "molecules_per_species": FIXED_PAIRS,
        "atom_count": FIXED_ATOMS,
        "source": _raw_source(),
        "methodology": _methodology_record(FIXED_PAIRS),
        "runtime": [{"rank": rank} for rank in range(world_size)],
        "input": {
            "path": "structures/fixed.extxyz",
            "file_sha256": structure_sha256,
            "file_size_bytes": 10,
            "tensor_sha256": INPUT_TENSOR_SHA256,
            "manifest": None,
            "manifest_file": None,
        },
        "model": {"name": "AIMNet2 + PME + D3"},
        "distributed": {
            "api": "DomainParallel",
            "mesh_shape": [world_size],
            "mesh_dim_names": ["domain"],
            "grid_dims": None,
            "cells_per_dim": [4, 4, 8],
            "rank_grid": rank_grids[world_size],
            "domain_cutoff_a": 15.0,
            "domain_skin_a": 4.0,
            "compile": False,
            "require_nondegenerate": world_size > 1,
            "owned_atom_counts": owned,
            "owned_atom_count_min": min(owned),
            "owned_atom_count_max": max(owned),
            "halo_atom_counts": None,
            "halo_atom_counts_reason": "not_exposed_by_public_api",
            "partition_count": 1,
            "gather_count": 1,
        },
        "output": {
            "energy_ev": final_energy,
            "energy_ev_per_atom": final_energy / FIXED_ATOMS,
            "energy_dtype": energy_dtype,
            "forces_source_atom_order": final_summary,
            "forces_source_atom_order_npy": {
                "path": force_path,
                "sha256": force_sha256,
                "size_bytes": 1,
                "dtype": str(force_array.dtype),
                "shape": list(force_array.shape),
            },
            "atomic_numbers_source_atom_order_sha256": "5" * 64,
            "source_atom_id_sha256": "6" * 64,
            "measured_passes": measured_passes,
            "position_invariance": {
                "method": "maximum_minimum_image_displacement",
                "tolerance_a": (DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a),
                "warmup_maximum_minimum_image_displacement_a": (
                    overall_displacement * 0.5
                ),
                "measured_pass_maximum_minimum_image_displacements_a": (
                    pass_displacements
                ),
                "final_gather_maximum_minimum_image_displacement_a": (
                    overall_displacement
                ),
                "maximum_minimum_image_displacement_a": overall_displacement,
                "all_within_tolerance": True,
                "interpretation": (
                    "No integration update occurred; periodic coordinate "
                    "wrapping remains PBC-equivalent."
                ),
            },
        },
        "charges": (
            _charge_record(FIXED_ATOMS, charge_sum=1.0e-3)
            if world_size == 1
            else _unavailable_charge_record()
        ),
        "timing": {
            "setup_s_rank0": 1.0,
            "pass_times_s": times,
            "median_s": float(np.median(times)),
            "min_s": min(times),
            "max_s": max(times),
            "measurement_kind": "fixed_structure_energy_force_pass",
            "measurement_role": "fixed_evaluation",
            "warmup_count": 1,
            "measured_pass_count": 3,
            "requested_steps_per_pass": 1,
            "measured_model_evaluations_per_pass": 1,
            "warmup_requested_steps": 1,
            "warmup_automatic_force_prime_evaluations": (1 if world_size > 1 else 0),
            "warmup_model_evaluations": 2 if world_size > 1 else 1,
            "partition_count": 1,
            "gather_count": 1,
            "publishable_benchmark": False,
            "elapsed_reduction": "maximum across ranks via all_reduce",
            "source_input_sha256": INPUT_TENSOR_SHA256,
            "boundary": "one measured DomainParallel.run call per pass",
        },
        "memory": {
            "measured_pass_max_allocated_bytes_per_rank": allocated,
            "measured_pass_max_reserved_bytes_per_rank": reserved,
            "max_allocated_bytes": max(value for row in allocated for value in row),
            "max_reserved_bytes": max(value for row in reserved for value in row),
            "boundary": "measured run calls only",
        },
    }


def _electrostatics_raw_row(
    *,
    structure_sha256: str,
) -> dict[str, Any]:
    pme_energy = -10.0
    ewald_energy = -10.0001
    energy_difference = abs(pme_energy - ewald_energy)
    force = np.full((ELECTROSTATICS_ATOMS, 3), 0.1, dtype=np.float32)
    comparison_force = np.full_like(force, 1.0e-4)
    acceptance = {
        "declared_before_measurement": True,
        "absolute_energy_difference_ev_per_atom_max": (
            DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
        ),
        "force_difference_max_norm_ev_a_max": (
            DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
        ),
        "absolute_charge_sum_e_max": DOMAIN_METHODOLOGY.charge_sum_tolerance_e,
    }
    return {
        "schema": "alchemi.part1-domain-case.v5",
        "created_utc": "2026-07-27T12:00:00Z",
        "run_id": RUN_ID,
        "case_id": "electrostatics-validation",
        "mode": "electrostatics-validation",
        "measurement_role": "electrostatics_validation",
        "status": "complete",
        "success": True,
        "world_size": 1,
        "pair_count": ELECTROSTATICS_PAIRS,
        "molecules_per_species": ELECTROSTATICS_PAIRS,
        "atom_count": ELECTROSTATICS_ATOMS,
        "source": _raw_source(),
        "methodology": _methodology_record(ELECTROSTATICS_PAIRS),
        "runtime": [{"rank": 0}],
        "input": {
            "path": "structures/electrostatics.extxyz",
            "file_sha256": structure_sha256,
            "file_size_bytes": 10,
            "tensor_sha256": "7" * 64,
            "manifest": None,
            "manifest_file": None,
        },
        "settings": {"pme": {}, "ewald": {}},
        "charges": _charge_record(ELECTROSTATICS_ATOMS, charge_sum=2.0e-5),
        "pme": {
            "energy_ev": pme_energy,
            "forces": _force_summary(force),
        },
        "ewald": {
            "energy_ev": ewald_energy,
            "forces": _force_summary(force + comparison_force),
        },
        "comparison": {
            "energy_difference_ev": pme_energy - ewald_energy,
            "absolute_energy_difference_ev": energy_difference,
            "absolute_energy_difference_ev_per_atom": (
                energy_difference / ELECTROSTATICS_ATOMS
            ),
            "force_difference_rms_ev_a": 1.0e-4,
            "force_difference_max_norm_ev_a": float(np.sqrt(3.0) * 1.0e-4),
            "acceptance": acceptance,
            "passed": True,
        },
        "timing": {"wall_s": 1.0},
        "memory": {"max_allocated_bytes": 1000},
    }


def _seal_bundle(
    root: Path,
    *,
    mutate_table: Any | None = None,
    mutate_raw: Any | None = None,
    mutate_manifest: Any | None = None,
) -> Path:
    root.mkdir()
    force_arrays = _force_arrays()
    raw_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    job_record_indexes: dict[str, dict[str, Any]] = {}
    rank_grids = {1: "1x1x1", 2: "1x1x2", 4: "1x2x2"}
    times_by_world = {
        1: [3.0, 3.1, 2.9],
        2: [1.8, 1.7, 1.9],
        4: [1.2, 1.1, 1.3],
    }
    energy_by_world = {
        world_size: ENERGY_PASSES_BY_WORLD[world_size][-1] for world_size in WORLD_SIZES
    }
    fixed_structure_sha256: str | None = None
    electrostatics_structure_path: Path | None = None
    electrostatics_structure_sha256: str | None = None

    for world_size in WORLD_SIZES:
        job_root = root / "job-records" / f"gpus-{world_size:02d}"
        for directory in ("inputs", "results", "ranks", "logs"):
            (job_root / directory).mkdir(parents=True, exist_ok=True)
        fixed_structure = job_root / "inputs" / "fixed.extxyz"
        fixed_structure.write_text(
            "synthetic fixed structure\n",
            encoding="utf-8",
        )
        observed_fixed_sha256 = _sha256(fixed_structure)
        if fixed_structure_sha256 is None:
            fixed_structure_sha256 = observed_fixed_sha256
        assert observed_fixed_sha256 == fixed_structure_sha256

        if world_size == 1:
            electrostatics_structure_path = (
                job_root / "inputs" / "electrostatics.extxyz"
            )
            electrostatics_structure_path.write_text(
                "synthetic electrostatics structure\n",
                encoding="utf-8",
            )
            electrostatics_structure_sha256 = _sha256(electrostatics_structure_path)

        force_path = job_root / "results" / f"forces-gpus-{world_size:02d}.npy"
        np.save(force_path, force_arrays[world_size], allow_pickle=False)
        force_relative = force_path.relative_to(root).as_posix()
        force_sha256 = _sha256(force_path)
        raw_row = _distributed_raw_row(
            world_size=world_size,
            structure_sha256=fixed_structure_sha256,
            force_path=force_relative,
            force_sha256=force_sha256,
            force_array=force_arrays[world_size],
        )
        raw_row["input"]["path"] = fixed_structure.relative_to(root).as_posix()
        raw_row["input"]["file_size_bytes"] = fixed_structure.stat().st_size
        raw_row["output"]["forces_source_atom_order_npy"]["size_bytes"] = (
            force_path.stat().st_size
        )
        raw_rows.append(raw_row)
        timing = times_by_world[world_size]
        memory = raw_row["memory"]
        owned = raw_row["distributed"]["owned_atom_counts"]
        force_summary = raw_row["output"]["forces_source_atom_order"]
        comparison_energy = float(np.median(ENERGY_PASSES_BY_WORLD[world_size]))
        table_rows.append(
            {
                "case_id": raw_row["case_id"],
                "atom_count": FIXED_ATOMS,
                "molecules_per_species": FIXED_PAIRS,
                "nodes": world_size,
                "gpus": world_size,
                "ranks": world_size,
                "success": True,
                "status": "complete",
                "failure_type": "",
                "failure_stage": "",
                "error": "",
                "warmup_count": 1,
                "measured_pass_count": 3,
                "pass_times_s": json.dumps(timing, separators=(",", ":")),
                "median_s": float(np.median(timing)),
                "min_s": min(timing),
                "max_s": max(timing),
                "peak_memory_bytes_max_rank": memory["max_allocated_bytes"],
                "owned_atoms_min_rank": min(owned),
                "owned_atoms_max_rank": max(owned),
                "spatial_grid": rank_grids[world_size],
                "energy_ev": energy_by_world[world_size],
                "energy_ev_per_atom": energy_by_world[world_size] / FIXED_ATOMS,
                "comparison_energy_ev": comparison_energy,
                "comparison_energy_ev_per_atom": comparison_energy / FIXED_ATOMS,
                "comparison_energy_statistic": ("median_of_three_measured_passes"),
                "energy_dtype": (
                    DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(
                        world_size
                    )
                ),
                "force_rms_ev_per_a": force_summary["rms_ev_a"],
                "force_max_ev_per_a": force_summary["max_norm_ev_a"],
                "structure_sha256": fixed_structure_sha256,
                "settings_sha256": "",
                "input_tensor_sha256": INPUT_TENSOR_SHA256,
                "positions_pbc_equivalent": True,
                "max_minimum_image_displacement_a": raw_row["output"][
                    "position_invariance"
                ]["maximum_minimum_image_displacement_a"],
                "measurement_role": "fixed_evaluation",
                "measurement_kind": "fixed_structure_energy_force_pass",
            }
        )

        top_files = {
            "plan.json": "{}\n",
            "phase-summary.json": "{}\n",
            "collection-summary.json": "{}\n",
            "results.jsonl": "{}\n",
            "part1-runtime.json": "{}\n",
            "d3-cache.json": "{}\n",
            "aimnet-checkpoint-preflight.json": "{}\n",
            "gpu-names.txt": "NVIDIA H100 NVL\n",
            "gpu-topology.txt": "synthetic H100 topology\n",
            "network-interfaces.txt": "lo UNKNOWN\nib0 UP\n",
            "producer-SHA256SUMS": (
                f"{_sha256(CONFIG_PATH)}  config.py\n{'9' * 64}  part1_domain_run.py\n"
            ),
            "artifact-SHA256SUMS": f"{'8' * 64}  synthetic-record\n",
        }
        for name, content in top_files.items():
            (job_root / name).write_text(content, encoding="utf-8")
        (job_root / "results" / "fixed.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (job_root / "ranks" / "rank-00.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (job_root / "logs" / "case.log").write_text(
            "synthetic complete job\n",
            encoding="utf-8",
        )
        job_files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in job_root.rglob("*") if item.is_file()):
            relative = path.relative_to(job_root).as_posix()
            job_files[relative] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        producer_files = {
            "config.py": _sha256(CONFIG_PATH),
            "part1_domain_run.py": "9" * 64,
        }
        job_record_indexes[str(world_size)] = {
            "world_size": world_size,
            "files": job_files,
            "producer_checksum_file_sha256": _sha256(job_root / "producer-SHA256SUMS"),
            "artifact_checksum_file_sha256": _sha256(job_root / "artifact-SHA256SUMS"),
            "verified_producer_file_count": len(producer_files),
            "verified_artifact_file_count": len(job_files) - 1,
            "producer_files": producer_files,
        }

    assert fixed_structure_sha256 is not None
    assert electrostatics_structure_path is not None
    assert electrostatics_structure_sha256 is not None
    settings = _manifest_settings(fixed_structure_sha256)
    settings_sha256 = canonical_json_sha256(settings)
    for row in raw_rows:
        row["settings_sha256"] = settings_sha256
        row["bundle_source"] = "manifest.json#source"
        row["bundle_job_record"] = f"manifest.json#job_records/{row['world_size']}"
    for table_row in table_rows:
        table_row["settings_sha256"] = settings_sha256

    electrostatics_row = _electrostatics_raw_row(
        structure_sha256=electrostatics_structure_sha256,
    )
    electrostatics_row["input"]["path"] = electrostatics_structure_path.relative_to(
        root
    ).as_posix()
    electrostatics_row["input"]["file_size_bytes"] = (
        electrostatics_structure_path.stat().st_size
    )
    electrostatics_row["bundle_settings_sha256"] = settings_sha256
    electrostatics_row["bundle_source"] = "manifest.json#source"
    electrostatics_row["bundle_job_record"] = "manifest.json#job_records/1"
    raw_rows.append(electrostatics_row)

    table = pd.DataFrame(table_rows, columns=DISTRIBUTED_COLUMNS)
    if mutate_table is not None:
        mutate_table(table)
    if mutate_raw is not None:
        mutate_raw(raw_rows)

    table_path = root / "distributed.csv"
    table.to_csv(table_path, index=False, lineterminator="\n")
    raw_path = root / "raw-results.jsonl"
    _write_jsonl(raw_path, raw_rows)
    electrostatics_path = root / "electrostatics-validation.json"
    _write_json(electrostatics_path, raw_rows[-1])

    one_gpu_energy = float(
        np.median(
            [
                measured["energy_ev"]
                for measured in raw_rows[0]["output"]["measured_passes"]
            ]
        )
    )
    two_gpu_energy = float(
        np.median(
            [
                measured["energy_ev"]
                for measured in raw_rows[1]["output"]["measured_passes"]
            ]
        )
    )
    reference_forces = force_arrays[1].astype(np.float64)
    output_comparisons: dict[str, dict[str, Any]] = {}
    for world_size in WORLD_SIZES:
        row = next(
            item
            for item in raw_rows
            if item.get("mode") == "distributed"
            and item.get("world_size") == world_size
        )
        measured_energies = [
            float(measured["energy_ev"])
            for measured in row["output"]["measured_passes"]
        ]
        energy = float(np.median(measured_energies))
        energy_span_per_atom = (
            max(measured_energies) - min(measured_energies)
        ) / FIXED_ATOMS
        difference = force_arrays[world_size].astype(np.float64) - reference_forces
        energy_required = world_size == 4
        repeatability_required = world_size > 1
        repeatability_passed = energy_span_per_atom <= (
            DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
        )
        energy_passed = (
            abs(energy - two_gpu_energy) / FIXED_ATOMS
            <= DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
        )
        force_passed = bool(
            np.less_equal(
                np.abs(difference),
                DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a
                + DOMAIN_METHODOLOGY.evaluation_force_rtol * np.abs(reference_forces),
            ).all()
        )
        output_comparisons[str(world_size)] = {
            "one_gpu_energy_offset_ev": energy - one_gpu_energy,
            "one_gpu_energy_abs_offset_ev_per_atom": (
                abs(energy - one_gpu_energy) / FIXED_ATOMS
            ),
            "one_gpu_energy_offset_is_diagnostic_only": True,
            "energy_statistic": "median_of_three_measured_passes",
            "energy_dtype": (
                DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(world_size)
            ),
            "energy_repeatability_span_ev_per_atom": energy_span_per_atom,
            "energy_repeatability_tolerance_ev_per_atom": (
                DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
            ),
            "energy_repeatability_check_required": repeatability_required,
            "energy_repeatability_passed": (
                repeatability_passed if repeatability_required else None
            ),
            "distributed_energy_reference_gpus": 2,
            "distributed_energy_difference_ev": energy - two_gpu_energy,
            "distributed_energy_abs_difference_ev_per_atom": (
                abs(energy - two_gpu_energy) / FIXED_ATOMS
            ),
            "distributed_energy_check_required": energy_required,
            "distributed_energy_passed": energy_passed if energy_required else None,
            "force_rms_difference_ev_per_a_vs_1gpu": float(
                np.sqrt(np.mean(difference * difference))
            ),
            "force_max_difference_ev_per_a_vs_1gpu": float(
                np.linalg.norm(difference, axis=1).max()
            ),
            "force_max_component_difference_ev_per_a_vs_1gpu": float(
                np.abs(difference).max()
            ),
            "distributed_energy_agreement_tolerance_ev_per_atom": (
                DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
            ),
            "force_acceptance": (
                "abs(delta_component) <= atol + rtol * abs(one_gpu_component)"
            ),
            "force_atol_ev_per_a": (DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a),
            "force_rtol": DOMAIN_METHODOLOGY.evaluation_force_rtol,
            "force_passed": force_passed,
            "position_check": "maximum_minimum_image_displacement",
            "position_tolerance_a": (
                DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
            ),
            "maximum_minimum_image_displacement_a": row["output"][
                "position_invariance"
            ]["maximum_minimum_image_displacement_a"],
            "positions_pbc_equivalent": True,
            "required_checks_passed": (
                force_passed
                and (not repeatability_required or repeatability_passed)
                and (not energy_required or energy_passed)
            ),
        }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": "2026-07-27T12:00:00Z",
        "lesson": "Part 1 fixed-input domain decomposition",
        "site": "Compute Lab",
        "interconnect": "H100 NVL cluster",
        "source": _manifest_source(),
        "status": "complete",
        "methodology": _manifest_methodology(),
        "settings": settings,
        "settings_sha256": settings_sha256,
        "input": {
            "molecules_per_species": FIXED_PAIRS,
            "atom_count": FIXED_ATOMS,
            "structure_sha256": fixed_structure_sha256,
        },
        "execution": {
            "gpu_counts": list(WORLD_SIZES),
            "warmup_count": 1,
            "measured_pass_count": 3,
            "work_per_measured_pass": (
                "one fixed-structure energy-and-force evaluation"
            ),
            "publishable_benchmark": False,
            "observed_speedup": {
                "1": 1.0,
                "2": 3.0 / 1.8,
                "4": 2.5,
            },
            "parallel_efficiency": {
                "1": 1.0,
                "2": (3.0 / 1.8) / 2.0,
                "4": 2.5 / 4.0,
            },
        },
        "hardware": {
            "site": "Compute Lab",
            "site_source": "operator-declared",
            "interconnect": "H100 NVL cluster",
            "interconnect_source": ("operator-declared; raw GPU topology is retained"),
            "gpus_available": 4,
            "nodes_available": 4,
            "resource_count_source": (
                "derived from successful per-rank runtime records"
            ),
            "gpu_model": "NVIDIA H100 NVL",
            "gpu_memory_bytes": 100_000_000_000,
            "driver_version": "590.00",
            "cuda_version": "13.0",
            "observed_gpus_by_job": {"1": 1, "2": 2, "4": 4},
            "observed_nodes_by_job": {"1": 1, "2": 2, "4": 4},
        },
        "output_agreement": {
            "force_reference_gpus": 1,
            "distributed_energy_reference_gpus": 2,
            "one_gpu_energy_offsets_are_diagnostics_only": True,
            "energy_statistic": "median_of_three_measured_passes",
            "position_check": {
                "method": "maximum_minimum_image_displacement",
                "tolerance_a": (DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a),
                "meaning": ("Periodic coordinate wrapping remains PBC-equivalent."),
            },
            "comparisons": output_comparisons,
            "all_required_checks_passed": True,
        },
        "electrostatics_validation": {
            "file": electrostatics_path.name,
            "sha256": _sha256(electrostatics_path),
            "passed": True,
        },
        "job_records": job_record_indexes,
        "files": {
            table_path.name: {
                "sha256": _sha256(table_path),
                "size_bytes": table_path.stat().st_size,
            },
            raw_path.name: {
                "sha256": _sha256(raw_path),
                "size_bytes": raw_path.stat().st_size,
            },
            electrostatics_path.name: {
                "sha256": _sha256(electrostatics_path),
                "size_bytes": electrostatics_path.stat().st_size,
            },
        },
        "interpretation": (
            "The same fixed structure ran on one, two, and four H100 GPUs."
        ),
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)

    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )
    return root


def test_missing_results_are_explicitly_not_reported(tmp_path: Path) -> None:
    view = load_domain_lesson_view(tmp_path / "missing")

    assert not view.available
    assert "not been reported" in view.reason
    assert view.run_settings_table.empty
    assert view.layout_table.empty
    assert view.timing_table.empty
    assert view.output_agreement_table.empty
    assert view.charge_diagnostics_table.empty
    assert view.electrostatics_table.empty
    assert view.plot_data.empty
    assert view.successful_case_count == 0
    assert view.failed_case_count == 0
    assert view.measured_max_atom_count is None
    assert view.bundle_record is None


def test_complete_bundle_returns_short_learner_tables(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "results")

    view = load_domain_lesson_view(
        root,
        expected_atom_count=FIXED_ATOMS,
        expected_world_sizes=WORLD_SIZES,
    )

    assert view.available
    assert view.reason == ""
    settings = dict(
        view.run_settings_table.loc[:, ["setting", "value"]].itertuples(
            index=False, name=None
        )
    )
    assert settings["Model tensors, coordinates, forces"] == "float32"
    assert settings["Total energy"] == (
        "float32 on 1 GPU; float64 distributed reduction on 2/4 GPUs"
    )
    assert view.layout_table.to_dict("records") == [
        {
            "world_size": 1,
            "nodes": 1,
            "ranks": 1,
            "spatial_grid": "1x1x1",
            "owned_atoms_min": FIXED_ATOMS,
            "owned_atoms_max": FIXED_ATOMS,
        },
        {
            "world_size": 2,
            "nodes": 2,
            "ranks": 2,
            "spatial_grid": "1x1x2",
            "owned_atoms_min": FIXED_ATOMS // 2,
            "owned_atoms_max": FIXED_ATOMS // 2,
        },
        {
            "world_size": 4,
            "nodes": 4,
            "ranks": 4,
            "spatial_grid": "1x2x2",
            "owned_atoms_min": FIXED_ATOMS // 4,
            "owned_atoms_max": FIXED_ATOMS // 4,
        },
    ]
    assert view.timing_table["world_size"].tolist() == [1, 2, 4]
    assert view.timing_table["pass_1_s"].tolist() == pytest.approx([3.0, 1.8, 1.2])
    assert view.timing_table["pass_2_s"].tolist() == pytest.approx([3.1, 1.7, 1.1])
    assert view.timing_table["pass_3_s"].tolist() == pytest.approx([2.9, 1.9, 1.3])
    assert view.timing_table["median_time_s"].tolist() == pytest.approx([3.0, 1.8, 1.2])
    assert view.timing_table["speedup_vs_1gpu"].tolist() == pytest.approx(
        [1.0, 3.0 / 1.8, 2.5]
    )
    assert list(view.plot_data.columns) == list(PLOT_COLUMNS)
    assert view.plot_data["world_size"].tolist() == [1, 2, 4]
    assert BUNDLE_SCHEMA == "alchemi.domain-decomposition-lesson.v5"
    assert view.manifest["schema"] == BUNDLE_SCHEMA
    assert view.distributed_table["energy_ev"].tolist() == pytest.approx(
        [ENERGY_PASSES_BY_WORLD[world_size][-1] for world_size in WORLD_SIZES]
    )
    assert view.distributed_table["comparison_energy_ev"].tolist() == pytest.approx(
        [
            float(np.median(ENERGY_PASSES_BY_WORLD[world_size]))
            for world_size in WORLD_SIZES
        ]
    )
    assert view.distributed_table["comparison_energy_statistic"].tolist() == [
        "median_of_three_measured_passes",
        "median_of_three_measured_passes",
        "median_of_three_measured_passes",
    ]
    assert view.distributed_table["energy_dtype"].tolist() == [
        "torch.float32",
        "torch.float64",
        "torch.float64",
    ]

    agreement = view.output_agreement_table
    assert agreement["world_size"].tolist() == [1, 2, 4]
    assert agreement["one_gpu_energy_offset_is_diagnostic"].all()
    assert agreement["energy_statistic"].tolist() == [
        "median_of_three_measured_passes",
        "median_of_three_measured_passes",
        "median_of_three_measured_passes",
    ]
    assert agreement["energy_dtype"].tolist() == [
        "torch.float32",
        "torch.float64",
        "torch.float64",
    ]
    assert agreement["energy_repeatability_check_required"].tolist() == [
        False,
        True,
        True,
    ]
    assert pd.isna(agreement.iloc[0]["energy_repeatability_passed"])
    assert agreement.iloc[1:]["energy_repeatability_passed"].tolist() == [True, True]
    assert agreement["energy_repeatability_span_meV_atom"].tolist() == pytest.approx(
        [
            1000.0
            * (
                max(ENERGY_PASSES_BY_WORLD[world_size])
                - min(ENERGY_PASSES_BY_WORLD[world_size])
            )
            / FIXED_ATOMS
            for world_size in WORLD_SIZES
        ]
    )
    expected_repeatability_tolerance_mev = (
        1000.0
        * DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
    )
    assert agreement["energy_repeatability_tolerance_meV_atom"].tolist() == (
        pytest.approx([expected_repeatability_tolerance_mev] * 3)
    )
    assert agreement["energy_check_required"].tolist() == [
        False,
        False,
        True,
    ]
    assert pd.isna(agreement.iloc[0]["energy_passed"])
    assert pd.isna(agreement.iloc[1]["energy_passed"])
    assert bool(agreement.iloc[2]["energy_passed"])
    assert (
        agreement.iloc[0]["energy_repeatability_span_meV_atom"]
        > expected_repeatability_tolerance_mev
    )
    assert (
        agreement.iloc[1]["one_gpu_energy_offset_meV_atom"]
        > 1000.0 * DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom
    )
    assert agreement["passed"].all()
    agreement_display = domain_agreement_display_table(agreement)
    assert agreement_display["Energy repeatability"].tolist() == [
        "Not checked: float32 total",
        "Passed",
        "Passed",
    ]
    assert agreement_display["Distributed energy"].tolist() == [
        "Not compared: float32 total",
        "Reference: 2-GPU float64 reduction",
        "Passed vs 2 GPU",
    ]
    assert agreement_display["Forces"].tolist() == [
        "Reference: 1 GPU",
        "Passed vs 1 GPU",
        "Passed vs 1 GPU",
    ]
    assert view.charge_diagnostics_table.iloc[0]["atom_count"] == FIXED_ATOMS
    assert view.charge_diagnostics_table.iloc[0]["finite"]
    assert view.electrostatics_table.iloc[0]["passed"]
    assert view.successful_case_count == 4
    assert view.failed_case_count == 0
    assert view.measured_max_atom_count == FIXED_ATOMS
    assert view.failed_table.empty
    assert view.takeaway["all_fixed_evaluations_succeeded"]
    assert view.takeaway["positions_pbc_equivalent"]
    assert view.takeaway["max_minimum_image_displacement_a"] == pytest.approx(4.0e-7)
    assert view.takeaway["all_output_checks_passed"]
    assert view.bundle_record is not None
    for world_size in WORLD_SIZES:
        assert (
            "network-interfaces.txt"
            in view.manifest["job_records"][str(world_size)]["files"]
        )
    one_gpu_comparison = view.manifest["output_agreement"]["comparisons"]["1"]
    assert one_gpu_comparison["energy_statistic"] == ("median_of_three_measured_passes")
    assert one_gpu_comparison["energy_repeatability_check_required"] is False
    assert one_gpu_comparison["energy_repeatability_passed"] is None
    assert one_gpu_comparison["energy_repeatability_span_ev_per_atom"] > (
        DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
    )


def test_runner_shaped_records_use_the_expected_output_dtypes(
    tmp_path: Path,
) -> None:
    """Exercise the field names and dtypes emitted by the v5 runner."""

    root = _seal_bundle(tmp_path / "runner-shaped")
    rows = [
        json.loads(line)
        for line in (root / "raw-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    one_gpu = next(
        row
        for row in rows
        if row.get("mode") == "distributed" and row.get("world_size") == 1
    )
    two_gpu = next(
        row
        for row in rows
        if row.get("mode") == "distributed" and row.get("world_size") == 2
    )

    assert one_gpu["charges"]["dtype"] == "float32"
    assert one_gpu["output"]["energy_dtype"] == "torch.float32"
    assert one_gpu["output"]["forces_source_atom_order"]["dtype"] == "float32"
    assert one_gpu["output"]["measured_passes"][0]["forces"]["dtype"] == "float32"
    assert {
        measured_pass["energy_dtype"]
        for measured_pass in one_gpu["output"]["measured_passes"]
    } == {"torch.float32"}
    assert one_gpu["output"]["forces_source_atom_order_npy"]["dtype"] == "float32"
    assert two_gpu["output"]["energy_dtype"] == "torch.float64"
    assert {
        measured_pass["energy_dtype"]
        for measured_pass in two_gpu["output"]["measured_passes"]
    } == {"torch.float64"}
    assert two_gpu["output"]["forces_source_atom_order"]["dtype"] == "float32"
    assert load_domain_lesson_view(root).available


@pytest.mark.parametrize(
    "target",
    (
        "charges",
        "single_energy",
        "multi_energy",
        "force_summary",
        "measured_pass",
        "force_npy",
    ),
)
def test_loader_rejects_invalid_output_dtypes(
    tmp_path: Path,
    target: str,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        one_gpu = rows[0]
        two_gpu = rows[1]
        if target == "charges":
            one_gpu["charges"]["dtype"] = "torch.float32"
        elif target == "single_energy":
            one_gpu["output"]["measured_passes"][0]["energy_dtype"] = "torch.float64"
        elif target == "multi_energy":
            two_gpu["output"]["measured_passes"][0]["energy_dtype"] = "torch.float32"
        elif target == "force_summary":
            one_gpu["output"]["forces_source_atom_order"]["dtype"] = "torch.float32"
        elif target == "measured_pass":
            one_gpu["output"]["measured_passes"][0]["forces"]["dtype"] = "torch.float32"
        else:
            one_gpu["output"]["forces_source_atom_order_npy"]["dtype"] = "torch.float32"

    root = _seal_bundle(tmp_path / f"old-dtype-{target}", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="dtype|float32"):
        load_domain_lesson_view(root)


def test_no_oom_or_capacity_row_is_required(tmp_path: Path) -> None:
    view = load_domain_lesson_view(_seal_bundle(tmp_path / "results"))

    assert len(view.distributed_table) == 3
    assert "capacity" not in view.distributed_table.columns
    assert view.failed_case_count == 0
    assert not hasattr(view, "capacity_table")
    assert not hasattr(view, "parity_table")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("expected_atom_count", FIXED_ATOMS + 1, "one fixed atom count"),
        ("expected_world_sizes", (1, 2), "1/2/4-GPU"),
    ),
)
def test_expected_run_shape_must_match_config(
    tmp_path: Path,
    keyword: str,
    value: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_domain_lesson_view(tmp_path / "missing", **{keyword: value})


def test_loader_requires_all_three_gpu_rows(tmp_path: Path) -> None:
    def mutate(table: pd.DataFrame) -> None:
        table.drop(table.index[table["gpus"].eq(4)], inplace=True)

    root = _seal_bundle(tmp_path / "missing-four", mutate_table=mutate)
    with pytest.raises(DomainLessonResultsError, match="1/2/4-GPU"):
        load_domain_lesson_view(root)


def test_loader_requires_exactly_three_positive_pass_times(tmp_path: Path) -> None:
    def mutate(table: pd.DataFrame) -> None:
        table.loc[table["gpus"].eq(2), "pass_times_s"] = "[1.0,2.0]"

    root = _seal_bundle(tmp_path / "two-passes", mutate_table=mutate)
    with pytest.raises(DomainLessonResultsError, match="exactly 3 pass times"):
        load_domain_lesson_view(root)


def test_loader_recomputes_timing_median(tmp_path: Path) -> None:
    def mutate(table: pd.DataFrame) -> None:
        table.loc[table["gpus"].eq(2), "median_s"] = 99.0

    root = _seal_bundle(tmp_path / "wrong-median", mutate_table=mutate)
    with pytest.raises(
        DomainLessonResultsError,
        match="timing statistics do not match pass times",
    ):
        load_domain_lesson_view(root)


def test_loader_requires_one_partition_and_one_gather(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[1]["distributed"]["partition_count"] = 3

    root = _seal_bundle(tmp_path / "repartitioned", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="partition and gather"):
        load_domain_lesson_view(root)


def test_loader_requires_one_model_evaluation_per_measured_pass(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[2]["timing"]["measured_model_evaluations_per_pass"] = 2

    root = _seal_bundle(tmp_path / "extra-work", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="few-pass method"):
        load_domain_lesson_view(root)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("timing", "source_input_sha256"), "8" * 64),
        (("output", "position_invariance", "all_within_tolerance"), False),
        (
            (
                "output",
                "measured_passes",
                1,
                "maximum_minimum_image_displacement_a",
            ),
            9.0e-5,
        ),
        (
            (
                "output",
                "position_invariance",
                "warmup_maximum_minimum_image_displacement_a",
            ),
            2.0e-4,
        ),
    ),
)
def test_loader_rejects_changed_or_misidentified_positions(
    tmp_path: Path,
    path: tuple[Any, ...],
    value: Any,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        target: Any = rows[0]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    root = _seal_bundle(tmp_path / "changed-positions", mutate_raw=mutate)
    with pytest.raises(
        DomainLessonResultsError,
        match="input tensor|position|minimum-image",
    ):
        load_domain_lesson_view(root)


def test_loader_requires_finite_energy_force_and_charge_outputs(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["output"]["measured_passes"][0]["forces"]["finite"] = False

    root = _seal_bundle(tmp_path / "nonfinite", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="non-finite forces"):
        load_domain_lesson_view(root)


def test_loader_recomputes_force_agreement_from_saved_arrays(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(tmp_path / "bad-force-agreement")
    force_path = root / "job-records" / "gpus-04" / "results" / "forces-gpus-04.npy"
    array = np.load(force_path, allow_pickle=False)
    array[0, 0] += np.float32(0.1)
    np.save(force_path, array, allow_pickle=False)
    digest = _sha256(force_path)

    raw_path = root / "raw-results.jsonl"
    rows = [
        json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    row = next(item for item in rows if item.get("world_size") == 4)
    row["output"]["forces_source_atom_order_npy"]["sha256"] = digest
    _write_jsonl(raw_path, rows)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    force_record = manifest["job_records"]["4"]["files"]["results/forces-gpus-04.npy"]
    force_record["sha256"] = digest
    force_record["size_bytes"] = force_path.stat().st_size
    manifest["files"]["raw-results.jsonl"]["sha256"] = _sha256(raw_path)
    manifest["files"]["raw-results.jsonl"]["size_bytes"] = raw_path.stat().st_size
    _write_json(manifest_path, manifest)
    checksummed = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksummed
        ),
        encoding="utf-8",
    )

    with pytest.raises(DomainLessonResultsError, match="agreement limits"):
        load_domain_lesson_view(root)


def test_loader_rejects_distributed_energy_that_is_not_repeatable(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        two_gpu = next(item for item in rows if item.get("world_size") == 2)
        first_pass = two_gpu["output"]["measured_passes"][0]
        first_pass["energy_ev"] += (
            1.1
            * (
                DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom
            )
            * FIXED_ATOMS
        )
        first_pass["energy_ev_per_atom"] = first_pass["energy_ev"] / FIXED_ATOMS

    root = _seal_bundle(
        tmp_path / "nonrepeatable-distributed-energy",
        mutate_raw=mutate,
    )
    with pytest.raises(DomainLessonResultsError, match="not repeatable"):
        load_domain_lesson_view(root)


def test_loader_rejects_energy_outside_declared_tolerance(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        four = next(item for item in rows if item.get("world_size") == 4)
        for measured_pass in four["output"]["measured_passes"]:
            measured_pass["energy_ev"] += 100.0
            measured_pass["energy_ev_per_atom"] = (
                measured_pass["energy_ev"] / FIXED_ATOMS
            )
        final_pass = four["output"]["measured_passes"][-1]
        four["output"]["energy_ev"] = final_pass["energy_ev"]
        four["output"]["energy_ev_per_atom"] = final_pass["energy_ev_per_atom"]

    def mutate_table(table: pd.DataFrame) -> None:
        selected = table["gpus"].eq(4)
        table.loc[selected, "energy_ev"] += 100.0
        table.loc[selected, "energy_ev_per_atom"] = (
            table.loc[selected, "energy_ev"] / FIXED_ATOMS
        )
        table.loc[selected, "comparison_energy_ev"] += 100.0
        table.loc[selected, "comparison_energy_ev_per_atom"] = (
            table.loc[selected, "comparison_energy_ev"] / FIXED_ATOMS
        )

    root = _seal_bundle(
        tmp_path / "bad-energy",
        mutate_table=mutate_table,
        mutate_raw=mutate,
    )
    with pytest.raises(DomainLessonResultsError, match="agreement limits"):
        load_domain_lesson_view(root)


def test_loader_reserves_strict_charge_limit_for_solver_check(
    tmp_path: Path,
) -> None:
    root = _seal_bundle(tmp_path / "large-charge-residual")
    view = load_domain_lesson_view(root)

    assert abs(view.charge_diagnostics_table.iloc[0]["residual_e"]) > (
        DOMAIN_METHODOLOGY.charge_sum_tolerance_e
    )
    assert view.electrostatics_table.iloc[0]["passed"]


def test_loader_requires_electrostatics_charge_limit(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        charge = rows[-1]["charges"]
        charge["sum_e"] = 1.0e-2
        charge["residual_e"] = 1.0e-2
        charge["abs_residual_per_atom"] = 1.0e-2 / ELECTROSTATICS_ATOMS

    root = _seal_bundle(tmp_path / "bad-electrostatics-charge", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="charge residual"):
        load_domain_lesson_view(root)


def test_loader_requires_current_methodology_identity(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["methodology"]["version"] = "stale"

    root = _seal_bundle(
        tmp_path / "stale-methodology",
        mutate_manifest=mutate,
    )
    with pytest.raises(DomainLessonResultsError, match="current domain methodology"):
        load_domain_lesson_view(root)


def test_loader_requires_same_source_commit(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[2]["source"]["repository_commit"] = "9" * 40

    root = _seal_bundle(tmp_path / "mixed-source", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="source commit"):
        load_domain_lesson_view(root)


def test_loader_rejects_host_paths_in_portable_raw_results(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["input"]["path"] = "/scratch/private/fixed.extxyz"

    root = _seal_bundle(tmp_path / "host-path", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="host path"):
        load_domain_lesson_view(root)


def test_loader_rejects_changed_file_after_bundle_is_sealed(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "changed")
    (root / "distributed.csv").write_text("changed\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="SHA-256 mismatch"):
        load_domain_lesson_view(root)


def test_loader_rejects_undeclared_extra_file(tmp_path: Path) -> None:
    root = _seal_bundle(tmp_path / "extra-file")
    (root / "unrelated.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(DomainLessonResultsError, match="undeclared files"):
        load_domain_lesson_view(root)


def test_loader_checks_per_pass_memory_shape(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[1]["memory"]["measured_pass_max_allocated_bytes_per_rank"] = [
            [1, 1],
            [1, 1],
        ]

    root = _seal_bundle(tmp_path / "bad-memory", mutate_raw=mutate)
    with pytest.raises(DomainLessonResultsError, match="one row per measured pass"):
        load_domain_lesson_view(root)
