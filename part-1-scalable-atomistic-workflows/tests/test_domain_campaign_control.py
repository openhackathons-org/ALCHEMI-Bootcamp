"""CPU-only checks for the short offline H100 example."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_PATH = REPO_ROOT / "scripts" / "slurm_part1_domain_decomposition.sbatch"


def _load_plan_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "part1_domain_plan.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_plan_script()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_checksum_file(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{PLAN.sha256_file(source)}  {source.resolve()}\n" for source in files
        ),
        encoding="utf-8",
    )


def _write_checked_base_box(root: Path) -> Path:
    phenol_symbols = list("C" * 6 + "H" * 6 + "O")
    nma_symbols = list("C" * 3 + "H" * 7 + "NO")
    symbols = phenol_symbols * PLAN.BASE_PAIR_COUNT + nma_symbols * PLAN.BASE_PAIR_COUNT
    atoms = Atoms(symbols, positions=np.zeros((PLAN.BASE_ATOM_COUNT, 3)))
    molecule_id = np.concatenate(
        (
            np.repeat(
                np.arange(PLAN.BASE_PAIR_COUNT, dtype=np.int32),
                len(phenol_symbols),
            ),
            np.repeat(
                np.arange(
                    PLAN.BASE_PAIR_COUNT,
                    2 * PLAN.BASE_PAIR_COUNT,
                    dtype=np.int32,
                ),
                len(nma_symbols),
            ),
        )
    )
    molecule_kind = np.concatenate(
        (
            np.zeros(
                PLAN.BASE_PAIR_COUNT * len(phenol_symbols),
                dtype=np.int32,
            ),
            np.ones(
                PLAN.BASE_PAIR_COUNT * len(nma_symbols),
                dtype=np.int32,
            ),
        )
    )
    template_atom_index = np.concatenate(
        (
            np.tile(
                np.arange(len(phenol_symbols), dtype=np.int32),
                PLAN.BASE_PAIR_COUNT,
            ),
            np.tile(
                np.arange(len(nma_symbols), dtype=np.int32),
                PLAN.BASE_PAIR_COUNT,
            ),
        )
    )
    volume_a3 = (
        float(np.sum(atoms.get_masses())) * 1.66053906660 / PLAN.DEFAULT_DENSITY_G_CM3
    )
    atoms.set_cell([volume_a3 ** (1.0 / 3.0)] * 3)
    atoms.set_pbc(True)
    atoms.info.update(
        {
            "charge": 0,
            "pair_count": PLAN.BASE_PAIR_COUNT,
            "molecules_per_species": PLAN.BASE_PAIR_COUNT,
            "count_definition": PLAN.MOLECULE_COUNT_DEFINITION,
        }
    )
    arrays = {
        "source_atom_id": np.arange(
            PLAN.BASE_ATOM_COUNT,
            dtype=np.int32,
        ),
        "molecule_id": molecule_id,
        "molecule_component": molecule_kind.copy(),
        "molecule_kind": molecule_kind,
        "template_atom_index": template_atom_index,
    }
    for name, values in arrays.items():
        atoms.set_array(name, values)

    root.mkdir(parents=True)
    structure = root / "structure.extxyz"
    ase_write(structure, atoms, format="extxyz")
    manifest = {
        "schema": PLAN.BASE_BOX_SCHEMA,
        "methodology": PLAN.BASE_BOX_METHODOLOGY,
        "source": {
            "nci_subset_file": "../../nci_atlas/nci-atlas-curves.csv.gz",
            "nci_subset_sha256": PLAN.NCI_SUBSET_SHA256,
            "nci_system_id": PLAN.NCI_SYSTEM_ID,
            "nci_scale": PLAN.NCI_SCALE,
            "molecule_counts": {
                "phenol": PLAN.BASE_PAIR_COUNT,
                "N-methylacetamide": PLAN.BASE_PAIR_COUNT,
            },
            "packmol": {
                "version": PLAN.EXPECTED_PACKMOL_VERSION,
                "seed": PLAN.DEFAULT_PACKMOL_SEED,
                "tolerance_a": PLAN.DEFAULT_PACKMOL_TOLERANCE_A,
                "precision_a": PLAN.DEFAULT_PACKMOL_PRECISION_A,
            },
        },
        "structure": {
            "file": structure.name,
            "sha256": PLAN.sha256_file(structure),
            "format": "extxyz",
            "atom_count": PLAN.BASE_ATOM_COUNT,
            "molecule_count": 2 * PLAN.BASE_PAIR_COUNT,
            "molecules_per_species": PLAN.BASE_PAIR_COUNT,
            "construction_density_g_cm3": PLAN.DEFAULT_DENSITY_G_CM3,
            "periodic_min_distance_a": 2.0,
            "min_distance_required_a": 1.999,
            "pbc": [True, True, True],
            "arrays": {
                name: {
                    "dtype": str(values.dtype),
                    "shape": list(values.shape),
                    "sha256": PLAN.sha256(values.tobytes()).hexdigest(),
                }
                for name, values in arrays.items()
            },
        },
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _plan(world_size: int) -> dict[str, object]:
    return PLAN.build_plan(
        run_id=f"fixed-gpus-{world_size}",
        tutorial_commit="1" * 40,
        world_size=world_size,
    )


def test_plan_has_one_fixed_input_and_three_short_passes() -> None:
    assert PLAN.DEFAULT_WORLD_SIZES == (1, 2, 4)
    assert PLAN.DEFAULT_FIXED_PAIR_COUNT == 2_048
    assert PLAN.DEFAULT_FIXED_PAIR_COUNT * PLAN.ATOMS_PER_PAIR == 51_200
    assert PLAN.DEFAULT_WARMUP_COUNT == 1
    assert PLAN.DEFAULT_PASS_COUNT == 3

    plans = [_plan(world_size) for world_size in PLAN.DEFAULT_WORLD_SIZES]
    fixed_cases = [plan["fixed_case"] for plan in plans]

    assert [case["world_size"] for case in fixed_cases] == [1, 2, 4]
    assert {case["pair_count"] for case in fixed_cases} == {2_048}
    assert {case["atom_count"] for case in fixed_cases} == {51_200}
    assert {tuple(case["repeat_factors_xyz"]) for case in fixed_cases} == {(2, 2, 4)}
    assert {case["measurement_role"] for case in fixed_cases} == {"fixed_evaluation"}
    assert len(plans[0]["validation_cases"]) == 1
    assert plans[1]["validation_cases"] == []
    assert plans[2]["validation_cases"] == []
    for plan in plans:
        assert plan["timing"]["warmup_count"] == 1
        assert plan["timing"]["pass_count"] == 3
        assert plan["timing"]["measured_model_evaluations_per_pass"] == 1
        assert plan["timing"]["publishable_benchmark"] is False
        assert (
            plan["validation_acceptance"]["evaluation_position_mic_tolerance_a"]
            == PLAN.DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
        )


@pytest.mark.parametrize("world_size", (0, 3, 8))
def test_plan_rejects_gpu_counts_outside_1_2_4(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size must be one of"):
        _plan(world_size)


def test_campaign_source_has_no_size_search_or_deliberate_failure() -> None:
    planner = (
        (REPO_ROOT / "scripts" / "part1_domain_plan.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    slurm = SBATCH_PATH.read_text(encoding="utf-8").lower()

    for removed in (
        "capacity-pairs",
        "first_cuda_oom",
        "rescue-pairs",
        "steady-timing-pairs",
        "parity-pairs",
        "--dependency",
        "--measurement-role",
        "--warmup-count",
        "--sample-count",
    ):
        assert removed not in slurm
        assert removed not in planner
    assert "for nodes in 1 2 4" in slurm
    assert "one untimed initialization pass" in slurm
    assert "three measured" in slurm
    assert "--nodes 8" not in slurm


def test_prepare_repeats_checked_base_without_running_packmol(
    tmp_path: Path,
) -> None:
    base_dir = _write_checked_base_box(tmp_path / "base")
    output_dir = tmp_path / "pairs-2048"
    args = SimpleNamespace(
        pair_count=PLAN.DEFAULT_FIXED_PAIR_COUNT,
        density_g_cm3=PLAN.DEFAULT_DENSITY_G_CM3,
        tolerance_a=PLAN.DEFAULT_PACKMOL_TOLERANCE_A,
        precision_a=PLAN.DEFAULT_PACKMOL_PRECISION_A,
        seed=PLAN.DEFAULT_PACKMOL_SEED,
        base_box_dir=base_dir,
        nci_data=None,
        output_dir=output_dir,
        reuse_existing=False,
    )

    manifest = PLAN.prepare_input(args)
    expanded = ase_read(output_dir / "structure.extxyz", format="extxyz")

    assert manifest["construction"]["method"] == ("balanced_integer_supercell_repeat")
    assert manifest["construction"]["repeat_factors_xyz"] == [2, 2, 4]
    assert manifest["construction"]["packmol_rerun"] is False
    assert manifest["packmol"]["applied_to"] == "checked_base_box_only"
    assert len(expanded) == 51_200
    assert expanded.pbc.all()
    assert manifest["density_from_mass_and_cell_g_cm3"] == pytest.approx(1.0)
    assert manifest["structure"]["sha256"] == PLAN.sha256_file(
        output_dir / "structure.extxyz"
    )

    reuse_values = vars(args).copy()
    reuse_values["reuse_existing"] = True
    reuse_args = SimpleNamespace(**reuse_values)
    reused = PLAN.prepare_input(reuse_args)
    assert reused == manifest


def test_prepare_wraps_the_checked_box_into_one_periodic_cell(
    tmp_path: Path,
) -> None:
    def prepare(name: str) -> tuple[dict[str, object], Atoms]:
        output_dir = tmp_path / name
        manifest = PLAN.prepare_input(
            SimpleNamespace(
                pair_count=PLAN.BASE_PAIR_COUNT,
                density_g_cm3=PLAN.DEFAULT_DENSITY_G_CM3,
                tolerance_a=PLAN.DEFAULT_PACKMOL_TOLERANCE_A,
                precision_a=PLAN.DEFAULT_PACKMOL_PRECISION_A,
                seed=PLAN.DEFAULT_PACKMOL_SEED,
                base_box_dir=PLAN.DEFAULT_BASE_BOX_DIR,
                nci_data=None,
                output_dir=output_dir,
                reuse_existing=False,
            )
        )
        return manifest, ase_read(
            output_dir / "structure.extxyz",
            format="extxyz",
        )

    first_manifest, first = prepare("first")
    second_manifest, second = prepare("second")
    canonicalization = first_manifest["construction"][
        "periodic_coordinate_canonicalization"
    ]

    assert canonicalization["method"] == "ase.Atoms.wrap"
    assert canonicalization["eps"] == 0.0
    assert canonicalization["atoms_outside_before"] > 0
    assert canonicalization["atoms_outside_after"] == 0
    fractional = first.get_scaled_positions(wrap=False)
    assert np.all(fractional >= 0.0)
    assert np.all(fractional < 1.0)
    assert (
        first_manifest["structure"]["sha256"] == second_manifest["structure"]["sha256"]
    )
    np.testing.assert_array_equal(
        first.arrays["source_atom_id"],
        second.arrays["source_atom_id"],
    )
    np.testing.assert_allclose(first.positions, second.positions, atol=0.0)


def test_input_cell_geometry_rejects_a_false_cubic_summary() -> None:
    base_length_a = PLAN.equivalent_cubic_length_angstrom(
        PLAN.BASE_PAIR_COUNT,
        PLAN.DEFAULT_DENSITY_G_CM3,
    )
    volume_a3 = 2.0 * base_length_a**3
    equivalent = volume_a3 ** (1.0 / 3.0)
    manifest = {
        "schema": PLAN.INPUT_SCHEMA,
        "cell_geometry": "orthorhombic",
        "cell_a": [
            [base_length_a, 0.0, 0.0],
            [0.0, base_length_a, 0.0],
            [0.0, 0.0, 2.0 * base_length_a],
        ],
        "cell_lengths_a": [equivalent] * 3,
        "minimum_cell_length_a": equivalent,
        "equivalent_cubic_length_a": equivalent,
        "volume_a3": volume_a3,
    }
    with pytest.raises(
        ValueError,
        match="cell geometry is internally inconsistent",
    ):
        PLAN.validated_manifest_cell_geometry(manifest)


def _force_record(path: Path, values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sha256": PLAN.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "dtype": str(values.dtype),
        "shape": list(values.shape),
    }


def _force_summary(values: np.ndarray) -> dict[str, object]:
    values64 = values.astype(np.float64)
    magnitudes = np.linalg.vector_norm(values64, axis=1)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sum_vector_ev_a": values64.sum(axis=0).tolist(),
        "sum_abs_ev_a": float(np.abs(values64).sum()),
        "sum_squares_ev2_a2": float(np.square(values64).sum()),
        "rms_ev_a": float(np.sqrt(np.square(values64).mean())),
        "max_norm_ev_a": float(magnitudes.max(initial=0.0)),
        "finite": True,
    }


def _charge_record(atom_count: int) -> dict[str, object]:
    return {
        "available": True,
        "values": None,
        "dtype": "float32",
        "target_sum_e": 0.0,
        "shape": [atom_count, 1],
        "sha256": "c" * 64,
        "finite": True,
        "sum_e": 0.0,
        "residual_e": 0.0,
        "abs_residual_per_atom": 0.0,
        "sum_abs_e": 100.0,
        "max_abs_e": 0.2,
        "reason": "The float32 charge tensor used by PME was summarized.",
    }


def _unavailable_charge_record() -> dict[str, object]:
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
        "reason": "The distributed model group returns energy and forces only.",
    }


def _methodology_record(pair_count: int) -> dict[str, object]:
    config_path = (
        REPO_ROOT
        / "part-1-scalable-atomistic-workflows"
        / "aux"
        / "domain"
        / "config.py"
    )
    return {
        "source": PLAN.DOMAIN_METHODOLOGY.as_record(),
        "source_file": {
            "path": str(config_path.resolve()),
            "sha256": PLAN.sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "resolved_values": (
            PLAN.DOMAIN_METHODOLOGY.resolved_values(
                json_compatible=True,
            )
        ),
        "case_molecules_per_species": pair_count,
    }


def _fixed_row(
    job_dir: Path,
    *,
    world_size: int,
    input_path: Path,
    energy_ev: float = -10_000.0,
    pass_energy_offsets_ev: tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_offset: float = 0.0,
    success: bool = True,
    runner_host_paths: bool = False,
) -> dict[str, object]:
    pair_count = PLAN.DEFAULT_FIXED_PAIR_COUNT
    case_id = PLAN.fixed_case_id(pair_count, world_size)
    forces = np.full(
        (pair_count * PLAN.ATOMS_PER_PAIR, 3),
        force_offset,
        dtype=np.float32,
    )
    force_path = job_dir / "results" / f"{case_id}-forces.npy"
    force_summary = _force_summary(forces)
    rank_grid = {
        1: [1, 1, 1],
        2: [1, 1, 2],
        4: [1, 1, 4],
    }[world_size]
    owned = [forces.shape[0] // world_size] * world_size
    pass_displacements = [
        1.0e-6 * world_size,
        2.0e-6 * world_size,
        3.0e-6 * world_size,
    ]
    warmup_displacement = 0.5e-6 * world_size
    final_displacement = 2.5e-6 * world_size
    maximum_displacement = max(
        warmup_displacement,
        *pass_displacements,
        final_displacement,
    )
    row: dict[str, object] = {
        "schema": PLAN.RESULT_SCHEMA,
        "created_utc": "2026-07-27T00:00:00+00:00",
        "run_id": f"fixed-gpus-{world_size}",
        "case_id": case_id,
        "mode": "distributed",
        "measurement_role": "fixed_evaluation",
        "status": "complete" if success else "failed",
        "success": success,
        "world_size": world_size,
        "pair_count": pair_count,
        "molecules_per_species": pair_count,
        "atom_count": forces.shape[0],
        "source": {
            "tutorial_commit": "1" * 40,
            "toolkit_core_commit": PLAN.CORE_COMMIT,
            "toolkit_ops_commit": PLAN.OPS_COMMIT,
            "toolkit_version": "0.2.0",
            "repository_commit": "1" * 40,
            "repository_dirty": False,
        },
        "methodology": _methodology_record(pair_count),
        "runtime": [
            {
                "rank": rank,
                "local_rank": 0,
                "host": f"h100-node-{rank}",
                "gpu_name": "NVIDIA H100 NVL",
                "gpu_uuid": f"GPU-{world_size}-{rank}",
                "gpu_total_memory_bytes": 100_000_000_000,
                "driver_version": "590.44",
                "torch_cuda_version": "13.0",
            }
            for rank in range(world_size)
        ],
        "input": {
            "path": str(input_path.resolve()),
            "file_sha256": PLAN.sha256_file(input_path),
            "file_size_bytes": input_path.stat().st_size,
            "tensor_sha256": "b" * 64,
        },
        "distributed": {
            "api": "DomainParallel",
            "mesh_shape": [world_size],
            "mesh_dim_names": ["domain"],
            "grid_dims": None,
            "cells_per_dim": [4, 4, 8],
            "rank_grid": rank_grid,
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
        "charges": (
            _charge_record(forces.shape[0])
            if world_size == 1
            else _unavailable_charge_record()
        ),
        "output": {
            "energy_ev": energy_ev,
            "energy_ev_per_atom": energy_ev / forces.shape[0],
            "energy_dtype": (
                PLAN.DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(
                    world_size
                )
            ),
            "forces_source_atom_order": force_summary,
            "forces_source_atom_order_npy": _force_record(
                force_path,
                forces,
            ),
            "measured_passes": [
                {
                    "pass_index": pass_index,
                    "energy_ev": (energy_ev + pass_energy_offsets_ev[pass_index - 1]),
                    "energy_ev_per_atom": (
                        energy_ev + pass_energy_offsets_ev[pass_index - 1]
                    )
                    / forces.shape[0],
                    "energy_dtype": (
                        PLAN.DOMAIN_METHODOLOGY.evaluation_energy_dtype_for_world_size(
                            world_size
                        )
                    ),
                    "forces": force_summary,
                    "maximum_minimum_image_displacement_a": (
                        pass_displacements[pass_index - 1]
                    ),
                }
                for pass_index in (1, 2, 3)
            ],
            "position_invariance": {
                "method": "maximum_minimum_image_displacement",
                "tolerance_a": (PLAN.DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A),
                "warmup_maximum_minimum_image_displacement_a": (warmup_displacement),
                "measured_pass_maximum_minimum_image_displacements_a": (
                    pass_displacements
                ),
                "final_gather_maximum_minimum_image_displacement_a": (
                    final_displacement
                ),
                "maximum_minimum_image_displacement_a": (maximum_displacement),
                "all_within_tolerance": True,
                "interpretation": "PBC-equivalent fixed structure.",
            },
        },
        "timing": {
            "pass_times_s": [
                3.0 / world_size,
                3.1 / world_size,
                2.9 / world_size,
            ],
            "median_s": 3.0 / world_size,
            "min_s": 2.9 / world_size,
            "max_s": 3.1 / world_size,
            "measurement_kind": "fixed_structure_energy_force_pass",
            "measurement_role": "fixed_evaluation",
            "warmup_count": 1,
            "measured_pass_count": 3,
            "source_input_sha256": "b" * 64,
            "partition_count": 1,
            "gather_count": 1,
            "requested_steps_per_pass": 1,
            "measured_model_evaluations_per_pass": 1,
            "warmup_requested_steps": 1,
            "warmup_automatic_force_prime_evaluations": (1 if world_size > 1 else 0),
            "warmup_model_evaluations": 2 if world_size > 1 else 1,
            "publishable_benchmark": False,
        },
        "memory": {
            "max_allocated_bytes": 10_000_000,
            "max_reserved_bytes": 12_000_000,
            "measured_pass_max_allocated_bytes_per_rank": [
                [10_000_000] * world_size for _ in range(3)
            ],
            "measured_pass_max_reserved_bytes_per_rank": [
                [12_000_000] * world_size for _ in range(3)
            ],
        },
    }
    if not success:
        row["error"] = "unexpected runner failure"
    if runner_host_paths:
        _add_runner_host_paths(row)
    return row


def _electrostatics_row(
    *,
    input_path: Path,
    runner_host_paths: bool = False,
) -> dict[str, object]:
    pair_count = PLAN.DEFAULT_VALIDATION_PAIRS
    atom_count = pair_count * PLAN.ATOMS_PER_PAIR
    pme_energy = -1.0
    ewald_energy = -1.0001
    energy_difference = abs(pme_energy - ewald_energy)
    forces = np.zeros((atom_count, 3), dtype=np.float32)
    force_summary = _force_summary(forces)
    row: dict[str, object] = {
        "schema": PLAN.RESULT_SCHEMA,
        "created_utc": "2026-07-27T00:00:00+00:00",
        "run_id": "fixed-gpus-1",
        "case_id": PLAN.validation_case_id(pair_count),
        "mode": "electrostatics-validation",
        "measurement_role": "electrostatics_validation",
        "status": "complete",
        "success": True,
        "world_size": 1,
        "pair_count": pair_count,
        "molecules_per_species": pair_count,
        "atom_count": atom_count,
        "source": {
            "tutorial_commit": "1" * 40,
            "toolkit_core_commit": PLAN.CORE_COMMIT,
            "toolkit_ops_commit": PLAN.OPS_COMMIT,
            "toolkit_version": "0.2.0",
            "repository_commit": "1" * 40,
            "repository_dirty": False,
        },
        "methodology": _methodology_record(pair_count),
        "input": {
            "path": str(input_path.resolve()),
            "file_sha256": PLAN.sha256_file(input_path),
            "file_size_bytes": input_path.stat().st_size,
        },
        "charges": _charge_record(atom_count),
        "pme": {
            "energy_ev": pme_energy,
            "forces": force_summary,
        },
        "ewald": {
            "energy_ev": ewald_energy,
            "forces": force_summary,
        },
        "comparison": {
            "absolute_energy_difference_ev": energy_difference,
            "absolute_energy_difference_ev_per_atom": (energy_difference / atom_count),
            "force_difference_rms_ev_a": 1.0e-4,
            "force_difference_max_norm_ev_a": 1.0e-4,
            "acceptance": {
                "declared_before_measurement": True,
                "absolute_energy_difference_ev_per_atom_max": (
                    PLAN.DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM
                ),
                "force_difference_max_norm_ev_a_max": (
                    PLAN.DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A
                ),
                "absolute_charge_sum_e_max": (PLAN.DEFAULT_CHARGE_SUM_TOL_E),
            },
            "passed": True,
        },
        "timing": {
            "wall_s": 1.0,
            "timed_work": ("AIMNet2 charge forward, PME forward, and Ewald forward"),
        },
    }
    if runner_host_paths:
        _add_runner_host_paths(row)
    return row


def _add_runner_host_paths(row: dict[str, object]) -> None:
    """Use the path fields and nesting written by the real H100 runner."""

    tutorial_root = (
        f"/computelab-cluster/nfedik/alchemi/stage/repo/ALCHEMI-Bootcamp-{'1' * 40}"
    )
    core_root = (
        "/computelab-cluster/nfedik/alchemi/stage/toolkit/"
        f"nvalchemi-toolkit-{PLAN.CORE_COMMIT}"
    )
    ops_root = (
        "/computelab-cluster/nfedik/alchemi/stage/toolkit/"
        f"nvalchemi-toolkit-ops-{PLAN.OPS_COMMIT}"
    )
    checkpoint = (
        "/computelab-cluster/nfedik/alchemi/cache/aimnet/aimnet2-wb97m-d3_0.jpt"
    )
    d3_parameters = (
        "/computelab-cluster/nfedik/alchemi/home/.cache/"
        "nvalchemiops/dftd3_parameters.pt"
    )
    runner = f"{tutorial_root}/scripts/part1_domain_run.py"
    methodology = (
        f"{tutorial_root}/part-1-scalable-atomistic-workflows/aux/domain/config.py"
    )
    runtime = {
        "python_version": "3.12.11",
        "python_executable": (
            "/computelab-cluster/nfedik/alchemi/envs/"
            f"part1-python-{'1' * 40}/bin/python"
        ),
        "python_prefix": (
            f"/computelab-cluster/nfedik/alchemi/envs/part1-python-{'1' * 40}"
        ),
    }
    source = row.setdefault("source", {})
    assert isinstance(source, dict)
    source.update(
        {
            "toolkit_core_source_root": core_root,
            "toolkit_core_source_file": (f"{core_root}/src/nvalchemi/__init__.py"),
            "toolkit_core_source_file_sha256": "d" * 64,
            "toolkit_ops_source_root": ops_root,
            "toolkit_ops_source_file": (f"{ops_root}/src/nvalchemiops/__init__.py"),
            "toolkit_ops_source_file_sha256": "e" * 64,
            "repository_root": tutorial_root,
            "domain_methodology_config_file": methodology,
            "domain_methodology_config_sha256": PLAN.sha256_file(
                REPO_ROOT
                / "part-1-scalable-atomistic-workflows"
                / "aux"
                / "domain"
                / "config.py"
            ),
            "aimnet_checkpoint": checkpoint,
            "aimnet_checkpoint_sha256": PLAN.AIMNET_CHECKPOINT_SHA256,
            "aimnet_checkpoint_file": {
                "path": checkpoint,
                "sha256": PLAN.AIMNET_CHECKPOINT_SHA256,
                "size_bytes": 1,
            },
            "runner": runner,
            "runner_sha256": PLAN.sha256_file(
                REPO_ROOT / "scripts" / "part1_domain_run.py"
            ),
            "runner_file": {
                "path": runner,
                "sha256": PLAN.sha256_file(
                    REPO_ROOT / "scripts" / "part1_domain_run.py"
                ),
                "size_bytes": 1,
            },
            "runtime_software": runtime,
        }
    )
    methodology_record = row["methodology"]
    assert isinstance(methodology_record, dict)
    source_file = methodology_record["source_file"]
    assert isinstance(source_file, dict)
    source_file["path"] = methodology
    runtime_rows = row.setdefault("runtime", [{"rank": 0}])
    assert isinstance(runtime_rows, list)
    for runtime_row in runtime_rows:
        assert isinstance(runtime_row, dict)
        runtime_row.update(runtime)
    if row["mode"] == "distributed":
        row["model"] = {
            "d3": {
                "parameter_file": d3_parameters,
                "parameter_file_sha256": PLAN.D3_PARAMETER_SHA256,
                "parameter_file_identity": {
                    "path": d3_parameters,
                    "sha256": PLAN.D3_PARAMETER_SHA256,
                    "size_bytes": 1,
                },
            }
        }
    input_record = row["input"]
    assert isinstance(input_record, dict)
    input_record["manifest"] = {
        "construction": {
            "base_box_manifest": (
                f"{tutorial_root}/part-1-scalable-atomistic-workflows/"
                "data/domain_decomposition/prebuilt_base_box/manifest.json"
            ),
            "base_box_manifest_sha256": "8" * 64,
            "base_box_structure": (
                f"{tutorial_root}/part-1-scalable-atomistic-workflows/"
                "data/domain_decomposition/prebuilt_base_box/structure.extxyz"
            ),
            "base_box_structure_sha256": "9" * 64,
        },
        "source": {
            "nci_subset": (
                f"{tutorial_root}/part-1-scalable-atomistic-workflows/"
                "data/nci_atlas/nci-atlas-curves.csv.gz"
            ),
            "nci_subset_sha256": PLAN.NCI_SUBSET_SHA256,
            "packing_helper": (
                f"{tutorial_root}/part-1-scalable-atomistic-workflows/"
                "aux/domain/packing.py"
            ),
            "domain_methodology_config": methodology,
        },
        "structure": {"path": input_record["path"]},
    }


def _complete_job(
    root: Path,
    *,
    world_size: int,
    input_content: str = "same fixed structure\n",
    energy_offset_ev: float = 0.0,
    pass_energy_offsets_ev: tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_offset: float = 0.0,
    runner_host_paths: bool = False,
) -> Path:
    job_dir = root / f"job-{world_size}"
    plan = _plan(world_size)
    _write_json(job_dir / "plan.json", plan)
    structure = job_dir / "inputs" / "fixed" / "structure.extxyz"
    structure.parent.mkdir(parents=True)
    structure.write_text(input_content, encoding="utf-8")
    _write_json(
        structure.parent / "manifest.json",
        {"structure": {"sha256": PLAN.sha256_file(structure)}},
    )
    row = _fixed_row(
        job_dir,
        world_size=world_size,
        input_path=structure,
        energy_ev=-10_000.0 + energy_offset_ev,
        pass_energy_offsets_ev=pass_energy_offsets_ev,
        force_offset=force_offset,
        runner_host_paths=runner_host_paths,
    )
    result_path = job_dir / "results" / plan["fixed_case"]["result_file"]
    _write_json(result_path, row)
    if world_size == 1:
        validation_structure = job_dir / "inputs" / "validation" / "structure.extxyz"
        validation_structure.parent.mkdir(parents=True)
        validation_structure.write_text(
            "small validation structure\n",
            encoding="utf-8",
        )
        _write_json(
            validation_structure.parent / "manifest.json",
            {"structure": {"sha256": PLAN.sha256_file(validation_structure)}},
        )
        validation_case = plan["validation_cases"][0]
        _write_json(
            job_dir / "results" / validation_case["result_file"],
            _electrostatics_row(
                input_path=validation_structure,
                runner_host_paths=runner_host_paths,
            ),
        )
    PLAN.write_phase_summary(
        SimpleNamespace(
            phase_dir=job_dir,
            output=job_dir / "phase-summary.json",
        )
    )
    planned_cases = [
        plan["fixed_case"],
        *plan["validation_cases"],
    ]
    for case in planned_cases:
        rank_count = int(case["world_size"])
        for rank in range(rank_count):
            _write_json(
                job_dir / "ranks" / case["case_id"] / f"rank-{rank:02d}.json",
                {
                    "case_id": case["case_id"],
                    "rank": rank,
                    "success": True,
                    "stage": "complete",
                },
            )
        log = job_dir / "logs" / f"{case['case_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("completed\n", encoding="utf-8")

    result_rows = [
        json.loads(
            (job_dir / "results" / case["result_file"]).read_text(encoding="utf-8")
        )
        for case in planned_cases
    ]
    (job_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    _write_json(
        job_dir / "collection-summary.json",
        {
            "schema": PLAN.COLLECTION_SCHEMA,
            "successful_rows": len(result_rows),
            "failed_rows": 0,
        },
    )
    for name in (
        "part1-runtime.json",
        "d3-cache.json",
        "aimnet-checkpoint-preflight.json",
    ):
        _write_json(job_dir / name, {"status": "checked"})
    (job_dir / "gpu-names.txt").write_text(
        "".join("NVIDIA H100 NVL\n" for _ in range(world_size)),
        encoding="utf-8",
    )
    (job_dir / "gpu-topology.txt").write_text(
        "H100 NVL cluster\n",
        encoding="utf-8",
    )
    (job_dir / "network-interfaces.txt").write_text(
        "".join(
            f"h100-node-{rank} interface=enp1s0f0np0 "
            "nccl==enp1s0f0np0 gloo=enp1s0f0np0\n"
            for rank in range(world_size)
        ),
        encoding="utf-8",
    )
    producer_sources = [
        REPO_ROOT / "scripts" / "part1_domain_plan.py",
        REPO_ROOT / "scripts" / "part1_domain_run.py",
    ]
    if runner_host_paths:
        producer_sources.append(
            REPO_ROOT
            / "part-1-scalable-atomistic-workflows"
            / "aux"
            / "domain"
            / "config.py"
        )
    _write_checksum_file(
        job_dir / "producer-SHA256SUMS",
        producer_sources,
    )
    artifact_files = [
        job_dir / "plan.json",
        job_dir / "phase-summary.json",
        job_dir / "collection-summary.json",
        job_dir / "results.jsonl",
        job_dir / "part1-runtime.json",
        job_dir / "d3-cache.json",
        job_dir / "aimnet-checkpoint-preflight.json",
        job_dir / "gpu-names.txt",
        job_dir / "gpu-topology.txt",
        job_dir / "network-interfaces.txt",
        job_dir / "producer-SHA256SUMS",
        *sorted(
            path
            for directory in ("inputs", "results", "ranks", "logs")
            for path in (job_dir / directory).rglob("*")
            if path.is_file()
        ),
    ]
    _write_checksum_file(
        job_dir / "artifact-SHA256SUMS",
        artifact_files,
    )
    return job_dir


def test_phase_summary_requires_pbc_equivalent_positions(
    tmp_path: Path,
) -> None:
    job_dir = _complete_job(tmp_path, world_size=2)
    summary = json.loads((job_dir / "phase-summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["completed_case_count"] == 1

    result_path = next((job_dir / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["output"]["position_invariance"]["all_within_tolerance"] = False
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="not PBC-equivalent"):
        PLAN.write_phase_summary(
            SimpleNamespace(
                phase_dir=job_dir,
                output=job_dir / "phase-summary.json",
            )
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (
            ("output", "measured_passes", 0, "energy_ev", 100.0),
            "measured-pass energies are not mutually consistent",
        ),
        (
            ("output", "energy_ev", 100.0),
            "saved energy does not match the last measured pass",
        ),
        (
            (
                "output",
                "forces_source_atom_order",
                "max_norm_ev_a",
                1.0,
            ),
            "saved force summary does not match the last measured pass",
        ),
    ),
)
def test_phase_summary_checks_all_measured_outputs(
    tmp_path: Path,
    change: tuple[object, ...],
    message: str,
) -> None:
    job_dir = _complete_job(tmp_path, world_size=2)
    result_path = next((job_dir / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    target: object = result
    for key in change[:-2]:
        target = target[key]  # type: ignore[index]
    target[change[-2]] = change[-1]  # type: ignore[index]
    _write_json(result_path, result)

    with pytest.raises(ValueError, match=message):
        PLAN.write_phase_summary(
            SimpleNamespace(
                phase_dir=job_dir,
                output=job_dir / "phase-summary.json",
            )
        )


def test_phase_summary_records_pass_force_variation_but_checks_final_identity(
    tmp_path: Path,
) -> None:
    job_dir = _complete_job(tmp_path, world_size=2)
    result_path = next((job_dir / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["output"]["measured_passes"][0]["forces"]["rms_ev_a"] = 1.0
    result["output"]["measured_passes"][1]["forces"]["max_norm_ev_a"] = 2.0
    _write_json(result_path, result)

    summary = PLAN.write_phase_summary(
        SimpleNamespace(
            phase_dir=job_dir,
            output=job_dir / "phase-summary.json",
        )
    )
    assert summary["passed"] is True

    result["output"]["measured_passes"][2]["forces"]["max_norm_ev_a"] = 1.0
    _write_json(result_path, result)
    with pytest.raises(
        ValueError,
        match="saved force summary does not match the last measured pass",
    ):
        PLAN.write_phase_summary(
            SimpleNamespace(
                phase_dir=job_dir,
                output=job_dir / "phase-summary.json",
            )
        )


@pytest.mark.parametrize("world_size", (2, 4))
def test_phase_summary_enforces_energy_repeatability_above_one_gpu(
    tmp_path: Path,
    world_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="measured-pass energies are not mutually consistent",
    ):
        _complete_job(
            tmp_path,
            world_size=world_size,
            pass_energy_offsets_ev=(20.0, 0.0, 0.0),
        )


def test_bundle_keeps_one_gpu_energy_spread_as_a_diagnostic(
    tmp_path: Path,
) -> None:
    jobs = [
        _complete_job(
            tmp_path,
            world_size=1,
            pass_energy_offsets_ev=(20.0, 0.0, 0.0),
        ),
        _complete_job(
            tmp_path,
            world_size=2,
            energy_offset_ev=10.0,
        ),
        _complete_job(
            tmp_path,
            world_size=4,
            energy_offset_ev=10.01,
        ),
    ]
    manifest = PLAN.build_bundle(
        SimpleNamespace(
            job_dir=jobs,
            site="Compute Lab",
            interconnect="H100 NVL cluster",
            output_dir=tmp_path / "one-gpu-diagnostic",
        )
    )

    comparisons = manifest["output_agreement"]["comparisons"]
    one_gpu = comparisons["1"]
    assert (
        one_gpu["energy_repeatability_span_ev_per_atom"]
        > PLAN.DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM
    )
    assert one_gpu["energy_repeatability_check_required"] is False
    assert one_gpu["energy_repeatability_passed"] is None
    for world_size in ("2", "4"):
        assert comparisons[world_size]["energy_repeatability_check_required"] is True
        assert comparisons[world_size]["energy_repeatability_passed"] is True
    assert manifest["output_agreement"]["all_required_checks_passed"] is True


def test_record_failure_marks_oom_as_unexpected(tmp_path: Path) -> None:
    input_path = tmp_path / "structure.extxyz"
    input_path.write_text("fixed structure\n", encoding="utf-8")
    log_path = tmp_path / "case.log"
    log_path.write_text("CUDA out of memory\n", encoding="utf-8")
    row = PLAN.record_failure(
        SimpleNamespace(
            run_id="fixed-gpus-2",
            case_id=PLAN.fixed_case_id(2_048, 2),
            mode="distributed",
            world_size=2,
            input_extxyz=input_path,
            rank_output_dir=tmp_path / "ranks",
            case_log=log_path,
            exit_code=1,
            output=tmp_path / "result.json",
        )
    )
    assert row["success"] is False
    assert row["failure"] == {
        "type": "CudaOutOfMemory",
        "stage": "process",
        "exit_code": 1,
        "unexpected": True,
    }


def test_failure_phase_summary_falls_back_to_the_nested_structure_digest(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "failed-job"
    plan = _plan(2)
    _write_json(job_dir / "plan.json", plan)
    input_path = job_dir / "inputs" / "fixed" / "structure.extxyz"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("fixed structure\n", encoding="utf-8")
    digest = PLAN.sha256_file(input_path)
    log_path = job_dir / "logs" / "case.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("NCCL initialization failed\n", encoding="utf-8")
    result_path = job_dir / "results" / plan["fixed_case"]["result_file"]
    row = PLAN.record_failure(
        SimpleNamespace(
            run_id=plan["run_id"],
            case_id=plan["fixed_case"]["case_id"],
            mode="distributed",
            world_size=2,
            input_extxyz=input_path,
            rank_output_dir=job_dir / "ranks",
            case_log=log_path,
            exit_code=1,
            output=result_path,
        )
    )
    del row["input"]["file_sha256"]
    _write_json(result_path, row)

    summary_path = job_dir / "phase-summary.json"
    with pytest.raises(ValueError, match="NCCL initialization failed"):
        PLAN.write_phase_summary(
            SimpleNamespace(
                phase_dir=job_dir,
                output=summary_path,
            )
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert summary["status"] == "failed"
    assert summary["input_structure_sha256"] == digest
    assert not any(
        "valid input structure SHA-256" in error for error in summary["errors"]
    )


def test_bundle_requires_complete_matching_1_2_4_results(
    tmp_path: Path,
) -> None:
    jobs = [
        _complete_job(tmp_path, world_size=1),
        _complete_job(
            tmp_path,
            world_size=2,
            energy_offset_ev=10.0,
            force_offset=1.0e-4,
        ),
        _complete_job(
            tmp_path,
            world_size=4,
            energy_offset_ev=10.01,
            force_offset=-1.0e-4,
        ),
    ]
    output_dir = tmp_path / "bundle"
    manifest = PLAN.build_bundle(
        SimpleNamespace(
            job_dir=jobs,
            site="Compute Lab",
            interconnect="H100 NVL cluster",
            output_dir=output_dir,
        )
    )

    assert manifest["schema"] == PLAN.BUNDLE_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["source"]["toolkit_version"] == "0.2.0"
    assert manifest["input"]["atom_count"] == 51_200
    assert manifest["execution"]["gpu_counts"] == [1, 2, 4]
    assert manifest["execution"]["warmup_count"] == 1
    assert manifest["execution"]["measured_pass_count"] == 3
    assert manifest["hardware"]["gpu_model"] == "NVIDIA H100 NVL"
    assert manifest["hardware"]["gpu_memory_bytes"] == 100_000_000_000
    assert manifest["hardware"]["driver_version"] == "590.44"
    assert manifest["hardware"]["cuda_version"] == "13.0"
    assert manifest["hardware"]["gpus_available"] == 4
    assert manifest["hardware"]["nodes_available"] == 4
    assert manifest["settings_sha256"] == PLAN.canonical_json_sha256(
        manifest["settings"]
    )
    assert manifest["output_agreement"]["all_required_checks_passed"] is True
    assert manifest["output_agreement"]["force_reference_gpus"] == 1
    assert manifest["output_agreement"]["distributed_energy_reference_gpus"] == 2
    assert (
        manifest["output_agreement"]["one_gpu_energy_offsets_are_diagnostics_only"]
        is True
    )
    assert (
        manifest["output_agreement"]["position_check"]["tolerance_a"]
        == PLAN.DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
    )
    two_gpu = manifest["output_agreement"]["comparisons"]["2"]
    four_gpu = manifest["output_agreement"]["comparisons"]["4"]
    assert (
        two_gpu["one_gpu_energy_abs_offset_ev_per_atom"]
        > PLAN.DEFAULT_EVALUATION_ENERGY_TOL_EV_PER_ATOM
    )
    assert two_gpu["distributed_energy_check_required"] is False
    assert four_gpu["distributed_energy_check_required"] is True
    assert four_gpu["distributed_energy_passed"] is True
    assert set(manifest["execution"]["observed_speedup"]) == {"1", "2", "4"}

    with (output_dir / "distributed.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == PLAN.DISTRIBUTED_COLUMNS
    assert [int(row["gpus"]) for row in rows] == [1, 2, 4]
    assert {int(row["measured_pass_count"]) for row in rows} == {3}
    assert {json.loads(row["pass_times_s"]).__len__() for row in rows} == {3}
    assert {row["positions_pbc_equivalent"] for row in rows} == {"True"}
    assert all(
        float(row["max_minimum_image_displacement_a"])
        <= PLAN.DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
        for row in rows
    )
    assert (output_dir / "raw-results.jsonl").is_file()
    assert (output_dir / "electrostatics-validation.json").is_file()
    assert (output_dir / "SHA256SUMS").is_file()
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(raw_rows) == 4
    fixed_rows = [row for row in raw_rows if row["mode"] == "distributed"]
    assert [row["world_size"] for row in fixed_rows] == [1, 2, 4]
    assert fixed_rows[0]["charges"]["available"] is True
    assert all(row["charges"]["available"] is False for row in fixed_rows[1:])
    for row in fixed_rows:
        assert len(row["output"]["measured_passes"]) == 3
        assert row["timing"]["partition_count"] == 1
        assert row["timing"]["gather_count"] == 1
        assert (
            len(
                row["output"]["position_invariance"][
                    "measured_pass_maximum_minimum_image_displacements_a"
                ]
            )
            == 3
        )
        assert row["output"]["position_invariance"]["all_within_tolerance"] is True
        assert len(row["memory"]["measured_pass_max_allocated_bytes_per_rank"]) == 3
        assert row["bundle_source"] == "manifest.json#source"
    electrostatics = raw_rows[-1]
    assert electrostatics["mode"] == "electrostatics-validation"
    for name in ("charges", "pme", "ewald", "comparison", "timing"):
        assert name in electrostatics

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
    for world_size in ("1", "2", "4"):
        record = manifest["job_records"][world_size]
        assert required_job_files <= set(record["files"])
        assert any(name.startswith("results/") for name in record["files"])
        assert any(name.startswith("ranks/") for name in record["files"])
        assert any(name.startswith("logs/") for name in record["files"])
        assert any(name.startswith("inputs/") for name in record["files"])
        assert record["verified_producer_file_count"] == 2
        assert record["verified_artifact_file_count"] == (len(record["files"]) - 1)

    from aux.domain.results import load_domain_lesson_view

    loaded = load_domain_lesson_view(output_dir)
    assert loaded.available is True
    assert loaded.successful_case_count == 4


def test_bundle_rewrites_real_runner_host_paths(tmp_path: Path) -> None:
    jobs = [
        _complete_job(
            tmp_path,
            world_size=1,
            runner_host_paths=True,
        ),
        _complete_job(
            tmp_path,
            world_size=2,
            energy_offset_ev=10.0,
            force_offset=1.0e-4,
            runner_host_paths=True,
        ),
        _complete_job(
            tmp_path,
            world_size=4,
            energy_offset_ev=10.01,
            force_offset=-1.0e-4,
            runner_host_paths=True,
        ),
    ]
    output_dir = tmp_path / "portable-bundle"
    PLAN.build_bundle(
        SimpleNamespace(
            job_dir=jobs,
            site="Compute Lab",
            interconnect="H100 NVL cluster",
            output_dir=output_dir,
        )
    )

    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    def absolute_strings(value: object) -> list[str]:
        if isinstance(value, dict):
            return [path for item in value.values() for path in absolute_strings(item)]
        if isinstance(value, list):
            return [path for item in value for path in absolute_strings(item)]
        if isinstance(value, str) and Path(value).is_absolute():
            return [value]
        return []

    assert all(not absolute_strings(row) for row in raw_rows)
    fixed = raw_rows[0]
    assert fixed["source"]["repository_root"].startswith("git:ALCHEMI-Bootcamp@")
    assert fixed["source"]["toolkit_core_source_file"].startswith(
        "git:NVIDIA/nvalchemi-toolkit@"
    )
    assert fixed["source"]["aimnet_checkpoint"].startswith("model:AIMNet2/")
    assert fixed["model"]["d3"]["parameter_file"].startswith(
        "parameters:DFT-D3@sha256:"
    )
    assert fixed["runtime"][0]["python_executable"].endswith(
        "part1-runtime.json#python_executable"
    )
    assert raw_rows[-1]["bundle_settings_sha256"]
    assert "settings_sha256" not in raw_rows[-1]


def test_bundle_rejects_missing_or_different_inputs(tmp_path: Path) -> None:
    jobs = [
        _complete_job(tmp_path, world_size=1),
        _complete_job(tmp_path, world_size=2),
    ]
    with pytest.raises(ValueError, match="one job directory"):
        PLAN.build_bundle(
            SimpleNamespace(
                job_dir=jobs,
                site="Compute Lab",
                interconnect="H100 NVL cluster",
                output_dir=tmp_path / "missing",
            )
        )

    jobs.append(
        _complete_job(
            tmp_path,
            world_size=4,
            input_content="different fixed structure\n",
        )
    )
    with pytest.raises(ValueError, match="same structure content"):
        PLAN.build_bundle(
            SimpleNamespace(
                job_dir=jobs,
                site="Compute Lab",
                interconnect="H100 NVL cluster",
                output_dir=tmp_path / "different",
            )
        )


def test_bundle_rechecks_every_job_file_checksum(tmp_path: Path) -> None:
    jobs = [
        _complete_job(tmp_path, world_size=world_size)
        for world_size in PLAN.DEFAULT_WORLD_SIZES
    ]
    changed_log = next((jobs[2] / "logs").glob("*.log"))
    changed_log.write_text("changed after job completion\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        PLAN.build_bundle(
            SimpleNamespace(
                job_dir=jobs,
                site="Compute Lab",
                interconnect="H100 NVL cluster",
                output_dir=tmp_path / "changed",
            )
        )


def test_slurm_runs_one_fixed_case_per_job_with_runner_derived_counts() -> None:
    source = SBATCH_PATH.read_text(encoding="utf-8")

    assert 'WORLD_SIZE="$SLURM_NNODES"' in source
    assert "config.fixed_molecules_per_species" in source
    assert "config.campaign_world_sizes" in source
    assert "config.evaluation_warmup_count" in source
    assert "config.evaluation_pass_count" in source
    assert 'prepare_input "$FIXED_PAIRS"' in source
    assert 'run_case distributed "$FIXED_PAIRS"' in source
    assert 'if [[ "$WORLD_SIZE" -eq 1 ]]' in source
    assert "run_case electrostatics-validation" in source
    assert '--mode "$mode"' in source
    assert '--world-size "$WORLD_SIZE"' in source
    assert '--pair-count "$pair_count"' in source
    assert "--force-output-npy" in source
    assert "--measurement-role" not in source
    assert "--warmup-count" not in source
    assert "--sample-count" not in source
    assert "--dependency" not in source
    assert "capacity" not in source.lower()
    assert "rescue" not in source.lower()
    assert 'elif [[ ! -s "$CURRENT_OUTPUT" ]]' in source
    assert "The launcher exited successfully but wrote no result JSON." in source
    assert "record_failure 66" in source
    assert 'sha256sum -c "$FINAL_DIR/producer-SHA256SUMS"' in source
    assert '> "$FINAL_DIR/artifact-SHA256SUMS"' in source
