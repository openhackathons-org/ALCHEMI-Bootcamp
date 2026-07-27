"""CPU-only checks for the offline H100 campaign controls."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _force_record(path: Path, values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sha256": PLAN.sha256_file(path),
        "shape": list(values.shape),
    }


def _write_checked_base_box(root: Path) -> Path:
    phenol_symbols = list("C" * 6 + "H" * 6 + "O")
    nma_symbols = list("C" * 3 + "H" * 7 + "NO")
    symbols = (
        phenol_symbols * PLAN.BASE_PAIR_COUNT
        + nma_symbols * PLAN.BASE_PAIR_COUNT
    )
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
        float(np.sum(atoms.get_masses()))
        * 1.66053906660
        / PLAN.DEFAULT_DENSITY_G_CM3
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
    atoms.set_array(
        "source_atom_id",
        np.arange(PLAN.BASE_ATOM_COUNT, dtype=np.int32),
    )
    atoms.set_array("molecule_id", molecule_id)
    atoms.set_array(
        "molecule_component",
        molecule_kind.copy(),
    )
    atoms.set_array("molecule_kind", molecule_kind)
    atoms.set_array("template_atom_index", template_atom_index)

    root.mkdir(parents=True)
    structure = root / "structure.extxyz"
    ase_write(structure, atoms, format="extxyz")
    manifest = {
        "schema": PLAN.BASE_BOX_SCHEMA,
        "methodology": {
            "schema": PLAN.DOMAIN_METHODOLOGY.schema,
            "name": PLAN.DOMAIN_METHODOLOGY.name,
            "version": PLAN.DOMAIN_METHODOLOGY.version,
        },
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
                for name, values in atoms.arrays.items()
                if name
                in {
                    "source_atom_id",
                    "molecule_id",
                    "molecule_component",
                    "molecule_kind",
                    "template_atom_index",
                }
            },
        },
    }
    _write_json(root / "manifest.json", manifest)
    return root


def test_prepare_repeats_checked_base_without_running_packmol(tmp_path: Path) -> None:
    base_dir = _write_checked_base_box(tmp_path / "base")
    output_dir = tmp_path / "pairs-256"
    args = SimpleNamespace(
        pair_count=256,
        density_g_cm3=PLAN.DEFAULT_DENSITY_G_CM3,
        tolerance_a=PLAN.DEFAULT_PACKMOL_TOLERANCE_A,
        precision_a=PLAN.DEFAULT_PACKMOL_PRECISION_A,
        seed=PLAN.DEFAULT_PACKMOL_SEED,
        base_box_dir=base_dir,
        packmol="definitely-not-an-executable",
        nci_data=None,
        output_dir=output_dir,
        reuse_existing=False,
    )

    manifest = PLAN.prepare_input(args)
    expanded = ase_read(output_dir / "structure.extxyz", format="extxyz")

    assert manifest["construction"] == {
        "method": "balanced_integer_supercell_repeat",
        "base_pair_count": 128,
        "repeat_multiplier": 2,
        "repeat_factors_xyz": [1, 1, 2],
        "base_box_manifest": str((base_dir / "manifest.json").resolve()),
        "base_box_manifest_schema": PLAN.BASE_BOX_SCHEMA,
        "base_box_manifest_sha256": PLAN.sha256_file(
            base_dir / "manifest.json"
        ),
        "base_box_structure": str((base_dir / "structure.extxyz").resolve()),
        "base_box_structure_sha256": PLAN.sha256_file(
            base_dir / "structure.extxyz"
        ),
        "packmol_rerun": False,
    }
    assert manifest["packmol"]["applied_to"] == "checked_base_box_only"
    assert len(expanded) == 6_400
    assert expanded.pbc.all()
    base_length_a = float(np.linalg.norm(expanded.cell.array[0]))
    assert manifest["cell_geometry"] == "orthorhombic"
    assert manifest["cell_lengths_a"] == pytest.approx(
        [base_length_a, base_length_a, 2.0 * base_length_a]
    )
    assert manifest["minimum_cell_length_a"] == pytest.approx(base_length_a)
    assert manifest["equivalent_cubic_length_a"] == pytest.approx(
        float(expanded.get_volume()) ** (1.0 / 3.0)
    )
    assert manifest["volume_a3"] == pytest.approx(float(expanded.get_volume()))
    assert "box_length_a" not in manifest
    assert np.array_equal(
        expanded.arrays["source_atom_id"],
        np.arange(6_400, dtype=np.int64),
    )
    assert np.array_equal(
        np.unique(expanded.arrays["molecule_id"]),
        np.arange(512, dtype=np.int64),
    )
    assert np.unique(
        expanded.arrays["molecule_kind"],
        return_counts=True,
    )[1].tolist() == [3_328, 3_072]
    assert manifest["density_from_mass_and_cell_g_cm3"] == pytest.approx(1.0)
    assert manifest["structure"]["sha256"] == PLAN.sha256_file(
        output_dir / "structure.extxyz"
    )


def test_input_cell_geometry_rejects_a_cubic_summary_for_an_elongated_cell() -> None:
    base_length_a = PLAN.equivalent_cubic_length_angstrom(
        PLAN.BASE_PAIR_COUNT,
        PLAN.DEFAULT_DENSITY_G_CM3,
    )
    volume_a3 = 2.0 * base_length_a**3
    equivalent_cubic_length_a = volume_a3 ** (1.0 / 3.0)
    manifest = {
        "schema": PLAN.INPUT_SCHEMA,
        "cell_geometry": "orthorhombic",
        "cell_a": [
            [base_length_a, 0.0, 0.0],
            [0.0, base_length_a, 0.0],
            [0.0, 0.0, 2.0 * base_length_a],
        ],
        "cell_lengths_a": [equivalent_cubic_length_a] * 3,
        "minimum_cell_length_a": equivalent_cubic_length_a,
        "equivalent_cubic_length_a": equivalent_cubic_length_a,
        "volume_a3": volume_a3,
    }

    with pytest.raises(
        ValueError,
        match="cell geometry is internally inconsistent",
    ):
        PLAN.validated_manifest_cell_geometry(manifest)


def test_electrostatics_summary_stops_a_failed_validation(tmp_path: Path) -> None:
    case_id = PLAN.validation_case_id(128)
    plan = {
        "validation_cases": [
            {
                "case_id": case_id,
                "result_file": f"{case_id}.json",
                "mode": "electrostatics-validation",
                "measurement_role": "electrostatics_validation",
            }
        ],
        "validation_acceptance": {"force_limit": 0.005},
    }
    _write_json(tmp_path / "plan.json", plan)
    result = {
        "schema": PLAN.RESULT_SCHEMA,
        "case_id": case_id,
        "mode": "electrostatics-validation",
        "measurement_role": "electrostatics_validation",
        "success": True,
        "comparison": {
            "acceptance": plan["validation_acceptance"],
            "passed": False,
        },
    }
    _write_json(tmp_path / "results" / f"{case_id}.json", result)

    summary = PLAN._electrostatics_phase_summary(tmp_path)

    assert summary["passed"] is False
    assert summary["status"] == "failed"
    assert "must not run" in summary["message"]


def _distributed_phase_directory(
    root: Path,
    *,
    speed_succeeds: bool,
    parity_force_offset: float,
) -> Path:
    pair_count = 128
    world_size = 2
    input_sha = "a" * 64
    reference_force_path = root / "capacity" / "reference-forces.npy"
    observed_force_path = root / "distributed" / "parity-forces.npy"
    steady_force_path = root / "distributed" / "steady-forces.npy"
    reference_forces = np.zeros((4, 3), dtype=np.float32)
    observed_forces = np.full(
        (4, 3),
        parity_force_offset,
        dtype=np.float32,
    )
    reference_result = {
        "schema": PLAN.RESULT_SCHEMA,
        "case_id": PLAN.capacity_case_id(pair_count, 1),
        "success": True,
        "input": {"file_sha256": input_sha},
        "output": {
            "energy_ev": -1.0,
            "forces_source_atom_order_npy": _force_record(
                reference_force_path,
                reference_forces,
            ),
        },
    }
    reference_result_path = root / "capacity" / "reference.json"
    _write_json(reference_result_path, reference_result)
    steady_reference_name = "steady-reference.json"
    _write_json(root / "capacity" / steady_reference_name, reference_result)
    selection = {
        "schema": PLAN.SELECTION_SCHEMA,
        "capacity_result_dir": str((root / "capacity").resolve()),
        "steady_timing_case": {"result_file": steady_reference_name},
        "parity_reference": {
            "result_file": str(reference_result_path.resolve()),
            "acceptance": {
                "energy_tolerance_ev_per_atom": 1.0e-4,
                "force_atol_ev_a": 1.0e-3,
                "force_rtol": 0.0,
            },
        },
    }
    selection_path = root / "capacity" / "selection.json"
    _write_json(selection_path, selection)

    phase_dir = root / "distributed"
    cases = [
        {
            "case_id": PLAN.parity_case_id(pair_count, world_size),
            "mode": "parity",
            "series": "parity",
            "measurement_role": "parity",
            "input": {"structure": {"sha256": input_sha}},
        },
        {
            "case_id": PLAN.steady_timing_case_id(pair_count, world_size),
            "mode": "steady-timing",
            "series": "steady_timing",
            "measurement_role": "steady_timing",
            "input": {"structure": {"sha256": input_sha}},
        },
        {
            "case_id": PLAN.rescue_case_id(pair_count * 2, world_size),
            "mode": "distributed",
            "series": "rescue",
            "measurement_role": "rescue",
            "input": {"structure": {"sha256": input_sha}},
        },
    ]
    derived_plan = {
        "schema": PLAN.DISTRIBUTED_PLAN_SCHEMA,
        "world_size": world_size,
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": PLAN.sha256_file(selection_path),
        },
        "cases": cases,
    }
    _write_json(phase_dir / "derived-plan.json", derived_plan)
    for case in cases:
        success = (
            True
            if case["series"] == "parity"
            else speed_succeeds
            if case["series"] == "steady_timing"
            else False
        )
        row = {
            "schema": PLAN.RESULT_SCHEMA,
            "case_id": case["case_id"],
            "mode": case["mode"],
            "measurement_role": case["measurement_role"],
            "success": success,
            "input": {"file_sha256": input_sha},
        }
        if case["series"] == "parity":
            row["output"] = {
                "energy_ev": -1.0,
                "forces_source_atom_order_npy": _force_record(
                    observed_force_path,
                    observed_forces,
                ),
            }
        elif case["series"] == "steady_timing" and success:
            row["output"] = {
                "energy_ev": -1.0,
                "forces_source_atom_order_npy": _force_record(
                    steady_force_path,
                    reference_forces,
                ),
            }
        _write_json(
            phase_dir / "results" / PLAN.result_filename(case["case_id"]),
            row,
        )
    return phase_dir


def test_distributed_phase_requires_parity_and_steady_timing_but_reports_rescue(
    tmp_path: Path,
) -> None:
    phase_dir = _distributed_phase_directory(
        tmp_path / "accepted",
        speed_succeeds=True,
        parity_force_offset=0.0,
    )

    summary = PLAN._distributed_phase_summary(phase_dir)

    assert summary["passed"] is True
    assert summary["checks"]["one_gpu_force_agreement_passed"] is True
    assert summary["checks"]["steady_timing_case_succeeded"] is True
    assert summary["checks"]["steady_timing_force_agreement_passed"] is True
    assert (
        summary["checks"]["distributed_energy_agreement_deferred_to_bundle"]
        is True
    )
    assert summary["checks"]["exact_oom_input_rescued"] is False


def test_distributed_phase_checks_the_actual_timing_output(tmp_path: Path) -> None:
    phase_dir = _distributed_phase_directory(
        tmp_path / "timing-output-mismatch",
        speed_succeeds=True,
        parity_force_offset=0.0,
    )
    case_id = PLAN.steady_timing_case_id(128, 2)
    result_path = phase_dir / "results" / PLAN.result_filename(case_id)
    row = json.loads(result_path.read_text(encoding="utf-8"))
    force_record = row["output"]["forces_source_atom_order_npy"]
    force_path = Path(force_record["path"])
    np.save(force_path, np.full((4, 3), 0.01, dtype=np.float32))
    force_record["sha256"] = PLAN.sha256_file(force_path)
    _write_json(result_path, row)

    summary = PLAN._distributed_phase_summary(phase_dir)

    assert summary["passed"] is False
    assert summary["checks"]["steady_timing_case_succeeded"] is True
    assert summary["checks"]["steady_timing_force_agreement_passed"] is False


def test_energy_and_force_checks_use_their_declared_references(
    tmp_path: Path,
) -> None:
    phase_dir = _distributed_phase_directory(
        tmp_path / "split-references",
        speed_succeeds=True,
        parity_force_offset=0.0,
    )
    selection = json.loads(
        (tmp_path / "split-references" / "capacity" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    parity_path = (
        phase_dir
        / "results"
        / PLAN.result_filename(PLAN.parity_case_id(128, 2))
    )
    two_gpu = json.loads(parity_path.read_text(encoding="utf-8"))

    one_gpu_force_check = json.loads(json.dumps(two_gpu))
    one_gpu_force_check["output"]["energy_ev"] = -0.5
    force_comparison = PLAN._parity_comparison(selection, one_gpu_force_check)
    assert force_comparison["passed"] is True
    assert force_comparison["forces_passed"] is True
    assert force_comparison["energy_passed"] is False
    assert force_comparison["energy_required"] is False

    four_gpu = json.loads(json.dumps(two_gpu))
    four_gpu["output"]["energy_ev"] = -0.9998
    energy_comparison = PLAN._distributed_energy_comparison(
        two_gpu,
        four_gpu,
        selection,
    )
    assert energy_comparison["passed"] is True
    assert energy_comparison["energy_passed"] is True
    assert energy_comparison["forces_required"] is False

    four_gpu["output"]["energy_ev"] = -0.9990
    failed_energy_comparison = PLAN._distributed_energy_comparison(
        two_gpu,
        four_gpu,
        selection,
    )
    assert failed_energy_comparison["passed"] is False
    assert failed_energy_comparison["energy_passed"] is False

    bundle_source = inspect.getsource(PLAN.build_bundle)
    assert (
        "parity_energy_reference = parity_rows_by_world[energy_reference_world_size]"
        in bundle_source
    )
    assert (
        "steady_energy_reference = steady_rows_by_world[energy_reference_world_size]"
        in bundle_source
    )
    assert "for world_size in energy_comparison_world_sizes" in bundle_source


@pytest.mark.parametrize(
    ("speed_succeeds", "parity_force_offset"),
    ((False, 0.0), (True, 0.01)),
)
def test_distributed_phase_fails_required_checks(
    tmp_path: Path,
    speed_succeeds: bool,
    parity_force_offset: float,
) -> None:
    phase_dir = _distributed_phase_directory(
        tmp_path / "failed",
        speed_succeeds=speed_succeeds,
        parity_force_offset=parity_force_offset,
    )

    summary = PLAN._distributed_phase_summary(phase_dir)

    assert summary["passed"] is False
    assert summary["status"] == "failed"


def test_job_time_checksum_file_is_verified_before_use(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    source.write_text('{"success": true}\n', encoding="utf-8")
    checksum_file = tmp_path / "artifact-SHA256SUMS"
    checksum_file.write_text(
        f"{PLAN.sha256_file(source)}  {source}\n",
        encoding="utf-8",
    )

    assert PLAN.read_verified_sha256sums(checksum_file) == {
        source.resolve(): PLAN.sha256_file(source)
    }

    source.write_text('{"success": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed after the job"):
        PLAN.read_verified_sha256sums(checksum_file)


def test_bundle_reference_rewriter_rejects_unmapped_host_paths() -> None:
    value = {
        "input": "/scratch/run/input.extxyz",
        "checkpoint": "/cache/aimnet.pt",
    }
    rewritten = PLAN._rewrite_bundle_references(
        value,
        copied_paths={"/scratch/run/input.extxyz": "job-records/capacity/input.extxyz"},
        external_paths={
            "/cache/aimnet.pt": "external:aimnet@sha256:abc",
        },
    )
    assert rewritten == {
        "input": "job-records/capacity/input.extxyz",
        "checkpoint": "external:aimnet@sha256:abc",
    }

    with pytest.raises(ValueError, match="host path"):
        PLAN._rewrite_bundle_references(
            {"missed": "/scratch/run/missed.json"},
            copied_paths={},
            external_paths={},
        )


def test_slurm_uses_local_caches_and_checks_each_phase() -> None:
    source = (
        REPO_ROOT / "scripts" / "slurm_part1_domain_decomposition.sbatch"
    ).read_text(encoding="utf-8")

    assert 'export TMPDIR="$NODE_CACHE_ROOT/tmp"' in source
    assert 'export WARP_CACHE_PATH="$NODE_CACHE_ROOT/warp"' in source
    assert 'export CUDA_CACHE_PATH="$NODE_CACHE_ROOT/cuda"' in source
    assert 'METHODOLOGY_CONFIG="$SHARED_REPO/' in source
    assert 'test -f "$METHODOLOGY_CONFIG"' in source
    assert "config.capacity_molecules_per_species" in source
    assert "config.electrostatics_validation_molecules_per_species" in source
    assert "config.parity_molecules_per_species" in source
    assert "config.distributed_world_sizes" in source
    assert '"$PACKING_HELPER" "$METHODOLOGY_CONFIG"' in source
    assert "--producer-file $METHODOLOGY_CONFIG" in source
    assert "ALCHEMI_DOMAIN_PAIR_COUNTS:-128 " not in source
    assert "ALCHEMI_DOMAIN_VALIDATION_PAIRS:-128" not in source
    assert "ALCHEMI_DOMAIN_PARITY_PAIRS:-2048" not in source
    assert "checkpoint-preflight" in source
    assert '--checkpoint "$AIMNET_CHECKPOINT"' in source
    validation_check = source.index("--phase electrostatics")
    capacity_ladder = source.index("reached_oom=false")
    assert validation_check < capacity_ladder
    assert "--phase capacity" in source
    assert "--phase distributed" in source


def test_slurm_runs_each_timing_role_with_the_planned_counts() -> None:
    source = (
        REPO_ROOT / "scripts" / "slurm_part1_domain_decomposition.sbatch"
    ).read_text(encoding="utf-8")

    run_case_start = source.index("run_case() {")
    capacity_start = source.index('if [[ "$PHASE" == capacity ]]', run_case_start)
    run_case = source[run_case_start:capacity_start]
    for declaration in (
        'local measurement_role="$2"',
        'local warmup_count="$3"',
        'local sample_count="$4"',
    ):
        assert declaration in run_case
    for forwarded_argument in (
        '--measurement-role "$measurement_role"',
        '--warmup-count "$warmup_count"',
        '--sample-count "$sample_count"',
    ):
        assert forwarded_argument in run_case

    distributed_start = source.index("else\n  CAPACITY_DIR=", capacity_start)
    capacity_phase = source[capacity_start:distributed_start]
    selection_read = capacity_phase.index(
        'json.load(open(sys.argv[1]))["steady_timing_case"]'
    )
    one_gpu_timing = capacity_phase.index(
        "run_case steady-timing steady_timing",
        selection_read,
    )
    capacity_summary = capacity_phase.index("--phase capacity", one_gpu_timing)
    assert selection_read < one_gpu_timing < capacity_summary
    assert '"$timing_warmup_count" "$timing_sample_count"' in capacity_phase

    distributed_phase = source[distributed_start:]
    derived_plan = distributed_phase.index('"$PLAN_SCRIPT" derive')
    planned_rows = distributed_phase.index(
        'c["mode"], c["measurement_role"], c["warmup_count"], '
        'c["sample_count"]',
        derived_plan,
    )
    planned_loop = distributed_phase.index(
        'for case_row in "${CASE_ROWS[@]}"',
        planned_rows,
    )
    planned_run = distributed_phase.index(
        'run_case "$mode" "$measurement_role" "$warmup_count" "$sample_count"',
        planned_loop,
    )
    assert derived_plan < planned_rows < planned_loop < planned_run
    assert '--world-size "$SLURM_NNODES"' in distributed_phase[
        derived_plan:planned_rows
    ]


def test_plan_separates_cold_roles_from_steady_timing() -> None:
    plan = PLAN.build_plan(
        run_id="cold-timing-test",
        world_sizes=(1,),
        capacity_pair_counts=(128, 256),
        validation_pairs=128,
        density_g_cm3=PLAN.DEFAULT_DENSITY_G_CM3,
        pme_cutoff_a=PLAN.DEFAULT_PME_CUTOFF_A,
        pme_mesh_safety_factor=PLAN.DEFAULT_PME_MESH_SAFETY_FACTOR,
        pme_spline_order=PLAN.DEFAULT_PME_SPLINE_ORDER,
        pme_accuracy=PLAN.DEFAULT_PME_ACCURACY,
        ewald_reference_accuracy=PLAN.DEFAULT_EWALD_REFERENCE_ACCURACY,
        d3_cutoff_a=PLAN.DEFAULT_D3_CUTOFF_A,
        d3_smoothing_fraction=PLAN.DEFAULT_D3_SMOOTHING_FRACTION,
        domain_skin_a=PLAN.DEFAULT_DOMAIN_SKIN_A,
        packmol_tolerance_a=PLAN.DEFAULT_PACKMOL_TOLERANCE_A,
        packmol_precision_a=PLAN.DEFAULT_PACKMOL_PRECISION_A,
        packmol_seed=PLAN.DEFAULT_PACKMOL_SEED,
    )

    assert plan["timing"]["cold"]["measurement_roles"] == [
        "capacity",
        "parity",
        "rescue",
    ]
    assert plan["timing"]["cold"]["warmup_count"] == 0
    assert plan["timing"]["cold"]["sample_count"] == 1
    assert plan["timing"]["steady"]["measurement_role"] == "steady_timing"
    assert plan["timing"]["steady"]["world_sizes"] == [1, 2, 4]
    assert plan["timing"]["steady"]["warmup_count"] >= 1
    assert plan["timing"]["steady"]["sample_count"] >= 5
    assert (
        plan["timing"]["steady"]["max_relative_iqr"]
        == PLAN.DOMAIN_METHODOLOGY.steady_timing_max_relative_iqr
    )
    methodology = PLAN.DOMAIN_METHODOLOGY
    assert (
        plan["timing"]["steady"]["model_evaluations_per_workflow"]
        == methodology.steady_timing_model_evaluations_per_workflow
    )
    assert (
        plan["timing"]["steady"]["one_rank_run_steps"]
        == methodology.steady_timing_run_steps(1)
    )
    assert (
        plan["timing"]["steady"]["multi_rank_run_steps"]
        == methodology.steady_timing_run_steps(
            methodology.distributed_world_sizes[0]
        )
    )
    assert plan["timing"]["publishable_benchmark"] is False
