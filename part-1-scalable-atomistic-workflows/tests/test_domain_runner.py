"""Static and CPU-only checks for the Part 1 fixed-input domain scripts."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain import results as lesson_results  # noqa: E402
from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_script("part1_domain_plan.py")
RUNNER = _load_script("part1_domain_run.py")


def _plan() -> dict[str, object]:
    return PLAN.build_plan(
        run_id="domain-test",
        tutorial_commit="1" * 40,
        world_size=1,
    )


def _clean_git_checkout(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Domain Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "domain@example.com"],
        check=True,
    )
    (path / "tracked.txt").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "pinned source"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def test_plan_uses_one_fixed_input_and_three_short_passes() -> None:
    plan = _plan()
    fixed_case = plan["fixed_case"]  # type: ignore[index]

    assert fixed_case["pair_count"] == PLAN.DEFAULT_FIXED_PAIR_COUNT
    assert fixed_case["atom_count"] == (
        PLAN.DEFAULT_FIXED_PAIR_COUNT * DOMAIN_METHODOLOGY.atoms_per_composition_unit
    )
    assert fixed_case["atom_count"] == 51_200
    assert fixed_case["molecules_per_species"] == fixed_case["pair_count"]
    assert fixed_case["world_size"] == 1
    assert fixed_case["measurement_role"] == "fixed_evaluation"
    assert fixed_case["repeat_factors_xyz"] == [2, 2, 4]
    count_definition = plan["input"]["count_definition"]  # type: ignore[index]
    assert "independent phenol molecules" in count_definition
    assert "not a count of pre-bound dimers" in count_definition
    assert plan["source"]["aimnet_checkpoint"] == "aimnet2-wb97m-d3_0"  # type: ignore[index]
    assert plan["model"]["neighbor_adaptation"] == "never"  # type: ignore[index]
    assert plan["distributed"]["grid_dims"] is None  # type: ignore[index]
    assert plan["timing"]["warmup_count"] == 1  # type: ignore[index]
    assert plan["timing"]["pass_count"] == 3  # type: ignore[index]
    assert plan["timing"]["measured_model_evaluations_per_pass"] == 1  # type: ignore[index]
    assert plan["timing"]["publishable_benchmark"] is False  # type: ignore[index]
    assert plan["model"]["pme_cutoff_a"] == 12.0  # type: ignore[index]
    assert (  # type: ignore[index]
        plan["model"]["pme_mesh_safety_factor"]
        == DOMAIN_METHODOLOGY.pme_mesh_safety_factor
    )
    assert (  # type: ignore[index]
        plan["model"]["pme_accuracy"] == DOMAIN_METHODOLOGY.pme_accuracy
    )
    assert (  # type: ignore[index]
        plan["model"]["ewald_reference_accuracy"]
        == DOMAIN_METHODOLOGY.ewald_reference_accuracy
    )
    assert plan["methodology"]["source"] == DOMAIN_METHODOLOGY.as_record()
    assert plan["methodology"]["source_identity"]["sha256"] == PLAN.sha256_file(
        PLAN.DOMAIN_METHODOLOGY_CONFIG_PATH
    )
    assert (
        plan["methodology"]["resolved_values"]["fixed_molecules_per_species"]
        == PLAN.DEFAULT_FIXED_PAIR_COUNT
    )


def test_planner_and_runner_defaults_come_from_domain_settings() -> None:
    assert PLAN.NCI_SYSTEM_ID == DOMAIN_METHODOLOGY.nci_system_id
    assert PLAN.NCI_SCALE == DOMAIN_METHODOLOGY.nci_scale
    assert (
        PLAN.DEFAULT_FIXED_PAIR_COUNT
        == DOMAIN_METHODOLOGY.fixed_molecules_per_species
        == 2_048
    )
    assert PLAN.DEFAULT_D3_CUTOFF_A == RUNNER.DEFAULT_D3_CUTOFF_A
    assert PLAN.DEFAULT_DOMAIN_SKIN_A == RUNNER.DEFAULT_DOMAIN_SKIN_A
    assert (
        PLAN.DEFAULT_PME_CUTOFF_A
        == RUNNER.DEFAULT_PME_CUTOFF_A
        == DOMAIN_METHODOLOGY.pme_realspace_cutoff_a
        == 12.0
    )
    assert (
        PLAN.DEFAULT_PME_MESH_SAFETY_FACTOR
        == RUNNER.DEFAULT_PME_MESH_SAFETY_FACTOR
        == DOMAIN_METHODOLOGY.pme_mesh_safety_factor
    )
    assert (
        PLAN.DEFAULT_PME_ACCURACY
        == RUNNER.DEFAULT_PME_ACCURACY
        == DOMAIN_METHODOLOGY.pme_accuracy
    )
    assert (
        PLAN.DEFAULT_EWALD_REFERENCE_ACCURACY
        == RUNNER.DEFAULT_EWALD_REFERENCE_ACCURACY
        == DOMAIN_METHODOLOGY.ewald_reference_accuracy
    )
    assert (
        PLAN.ATOMS_PER_PAIR
        == RUNNER.ATOMS_PER_COMPOSITION_UNIT
        == DOMAIN_METHODOLOGY.atoms_per_composition_unit
    )
    assert (
        RUNNER.EXPECTED_AIMNET_NEIGHBOR_CUTOFF_A
        == DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a
    )
    assert (
        PLAN.DEFAULT_WARMUP_COUNT
        == RUNNER.DEFAULT_EVALUATION_WARMUP_COUNT
        == DOMAIN_METHODOLOGY.evaluation_warmup_count
    )
    assert (
        PLAN.DEFAULT_PASS_COUNT
        == RUNNER.DEFAULT_EVALUATION_PASS_COUNT
        == DOMAIN_METHODOLOGY.evaluation_pass_count
    )
    assert (
        RUNNER.DEFAULT_EVALUATION_POSITION_MIC_TOLERANCE_A
        == DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
        == 1.0e-4
    )


def test_runner_matches_the_nci_member_and_declared_d3() -> None:
    assert PLAN.AIMNET_CHECKPOINT == "aimnet2-wb97m-d3_0"
    assert PLAN.AIMNET_CHECKPOINT_SHA256 == RUNNER.CHECKPOINT_SHA256
    assert RUNNER.EXPECTED_D3 == {
        "a1": 0.566,
        "a2": 3.128,
        "s6": 1.0,
        "s8": 0.3908,
    }


def test_runner_uses_the_public_estimator_for_coupled_pme_parameters() -> None:
    source = inspect.getsource(RUNNER.estimate_pme_setup)
    assert "estimate_pme_parameters(" in source
    assert "real_space_cutoff=real_space_cutoff_a" in source
    assert "mesh_safety_factor=mesh_safety_factor" in source

    setup = PLAN.expected_pme_setup(
        cell_lengths_a=(32.1, 64.2, 96.3),
        real_space_cutoff_a=PLAN.DEFAULT_PME_CUTOFF_A,
        accuracy=PLAN.DEFAULT_PME_ACCURACY,
        mesh_safety_factor=PLAN.DEFAULT_PME_MESH_SAFETY_FACTOR,
    )
    assert setup["real_space_cutoff_a"] == 12.0
    assert setup["alpha_a_inverse"] == pytest.approx(
        math.sqrt(-math.log(PLAN.DEFAULT_PME_ACCURACY)) / 12.0
    )
    assert setup["mesh_dimensions"] == [64, 128, 128]
    assert setup["mesh_spacing_a"] == pytest.approx(
        [32.1 / 64.0, 64.2 / 128.0, 96.3 / 128.0]
    )
    assert setup["parameter_rule"] == (
        "estimate_pme_parameters(accuracy, real_space_cutoff, mesh_safety_factor)"
    )


def test_expected_ewald_setup_uses_actual_cell_volume() -> None:
    volume_a3 = 32.1 * 64.2 * 96.3
    setup = PLAN.expected_ewald_reference_setup(
        atom_count=6_400,
        volume_a3=volume_a3,
        accuracy=PLAN.DEFAULT_EWALD_REFERENCE_ACCURACY,
    )
    eta = (volume_a3**2 / 6_400) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)
    error_factor = math.sqrt(-2.0 * math.log(PLAN.DEFAULT_EWALD_REFERENCE_ACCURACY))

    assert setup["real_space_cutoff_a"] == pytest.approx(error_factor * eta)
    assert setup["reciprocal_space_cutoff_a_inverse"] == pytest.approx(
        error_factor / eta
    )
    assert setup["alpha_a_inverse"] == pytest.approx(1.0 / (math.sqrt(2.0) * eta))


def test_recorded_rank_layout_uses_the_actual_rectangular_cell() -> None:
    cells, rank_grid = PLAN.validate_recorded_rank_layout(
        {
            "cells_per_dim": [4, 4, 8],
            "rank_grid": [1, 1, 4],
        },
        world_size=4,
    )

    assert cells == (4, 4, 8)
    assert rank_grid == (1, 1, 4)

    with pytest.raises(ValueError, match="world size"):
        PLAN.validate_recorded_rank_layout(
            {
                "cells_per_dim": [4, 4, 8],
                "rank_grid": [1, 2, 4],
            },
            world_size=4,
        )


def test_runner_and_loader_table_schemas_match() -> None:
    assert PLAN.DISTRIBUTED_COLUMNS == lesson_results.DISTRIBUTED_COLUMNS
    assert "molecules_per_species" in PLAN.DISTRIBUTED_COLUMNS
    assert "measurement_role" in PLAN.DISTRIBUTED_COLUMNS
    assert "pass_times_s" in PLAN.DISTRIBUTED_COLUMNS
    assert "median_s" in PLAN.DISTRIBUTED_COLUMNS
    assert "owned_atoms_min_rank" in PLAN.DISTRIBUTED_COLUMNS
    assert "owned_atoms_max_rank" in PLAN.DISTRIBUTED_COLUMNS
    assert "molecule_pairs" not in PLAN.DISTRIBUTED_COLUMNS
    assert "atom_steps_per_s" not in PLAN.DISTRIBUTED_COLUMNS


def test_runner_keeps_global_energy_separate_from_gathered_atom_fields(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(RUNNER.run_fixed_evaluation)
    order_source = inspect.getsource(RUNNER.source_order_from_gathered_ids)
    diagnostics_source = inspect.getsource(RUNNER.evaluation_output_diagnostics)

    assert "replicated_energy = result_owned.energy.detach().clone()" in source
    assert "gathered = domain.gather(result_owned, dst=0)" in source
    assert "energy_ev = float(replicated_energy.reshape(-1)[0].item())" in source
    assert "sorted_forces = gathered.forces[order]" in source
    assert "float(gathered.energy" not in source
    assert "torch.isfinite(energy)" in diagnostics_source
    assert "torch.isfinite(forces)" in diagnostics_source
    assert "torch.isfinite(gathered.positions)" in source
    assert "torch.isfinite(gathered.forces)" in source
    assert "source_order_from_gathered_ids(" in source
    assert "gathered.source_atom_id" not in source
    assert "source_atom_id is not an exact 0..N-1 permutation" in order_source

    output = tmp_path / "strict.json"
    with pytest.raises(ValueError, match="Out of range float"):
        RUNNER.atomic_write_json(output, {"bad": float("nan")})
    assert not output.exists()


def test_large_finite_charge_residual_is_recorded_per_atom() -> None:
    torch = pytest.importorskip("torch")
    atom_count = 51_200
    charges = torch.zeros(atom_count, dtype=torch.float32)
    charges[0] = -0.00467

    checked = RUNNER.charge_diagnostics(charges, target_sum_e=0.0)

    assert checked["dtype"] == "float32"
    assert checked["shape"] == [atom_count]
    assert checked["sum_e"] == pytest.approx(-0.00467)
    assert checked["residual_e"] == checked["sum_e"]
    assert checked["abs_residual_per_atom"] == pytest.approx(
        abs(checked["residual_e"]) / atom_count
    )
    assert abs(checked["residual_e"]) > PLAN.DEFAULT_CHARGE_SUM_TOL_E


def test_evaluation_output_diagnostics_uses_the_world_size_dtype() -> None:
    torch = pytest.importorskip("torch")

    single_rank = RUNNER.evaluation_output_diagnostics(
        SimpleNamespace(
            energy=torch.tensor([[1.0]], dtype=torch.float32),
            forces=torch.zeros((2, 3), dtype=torch.float32),
        ),
        world_size=1,
    )
    assert single_rank["valid"] is True
    assert single_rank["expected_energy_dtype"] == "torch.float32"
    assert single_rank["energy"]["shape"] == [1, 1]
    assert single_rank["forces"]["shape"] == [2, 3]

    multi_rank = RUNNER.evaluation_output_diagnostics(
        SimpleNamespace(
            energy=torch.tensor([[1.0]], dtype=torch.float64),
            forces=torch.zeros((2, 3), dtype=torch.float32),
        ),
        world_size=2,
    )
    assert multi_rank["valid"] is True
    assert multi_rank["expected_energy_dtype"] == "torch.float64"

    wrong_single_rank = RUNNER.evaluation_output_diagnostics(
        SimpleNamespace(
            energy=torch.tensor([[1.0]], dtype=torch.float64),
            forces=torch.zeros((2, 3), dtype=torch.float32),
        ),
        world_size=1,
    )
    wrong_multi_rank = RUNNER.evaluation_output_diagnostics(
        SimpleNamespace(
            energy=torch.tensor([[1.0]], dtype=torch.float32),
            forces=torch.zeros((2, 3), dtype=torch.float32),
        ),
        world_size=4,
    )
    assert wrong_single_rank["valid"] is False
    assert wrong_multi_rank["valid"] is False
    assert wrong_single_rank["forces"]["finite"] is True
    assert wrong_multi_rank["forces"]["finite"] is True


def test_external_source_ids_restore_toolkit_rank_contiguous_order() -> None:
    torch = pytest.importorskip("torch")

    class RankAssigner:
        def __init__(self) -> None:
            self.calls = 0

        def assign_atoms_to_ranks(self, positions: object) -> object:
            self.calls += 1
            assert positions is source_positions
            return torch.tensor([2, 0, 1, 0, 2, 1])

    source_positions = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    source_ids = torch.tensor([5, 0, 3, 1, 4, 2])
    partitioner = RankAssigner()

    gathered_ids = RUNNER.predict_gathered_source_ids(
        source_atom_ids=source_ids,
        positions=source_positions,
        partitioner=partitioner,
        world_size=3,
    )
    assert partitioner.calls == 1
    assert gathered_ids.tolist() == [0, 1, 3, 2, 5, 4]

    source_order = RUNNER.source_order_from_gathered_ids(
        gathered_ids,
        expected_atom_count=6,
    )
    gathered_values = 10 + gathered_ids
    assert gathered_values[source_order].tolist() == [10, 11, 12, 13, 14, 15]

    class MustNotAssign:
        def assign_atoms_to_ranks(self, positions: object) -> object:
            raise AssertionError("one-rank gather must keep input row order")

    one_rank_ids = RUNNER.predict_gathered_source_ids(
        source_atom_ids=source_ids,
        positions=source_positions,
        partitioner=MustNotAssign(),
        world_size=1,
    )
    assert torch.equal(one_rank_ids, source_ids)
    assert one_rank_ids.data_ptr() != source_ids.data_ptr()


def test_source_input_checksum_includes_ids_kept_outside_batch() -> None:
    torch = pytest.importorskip("torch")

    batch = SimpleNamespace(
        atomic_numbers=torch.tensor([6, 8], dtype=torch.int64),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    source_ids = torch.tensor([0, 1], dtype=torch.int64)

    checksum = RUNNER.source_input_checksum(batch, source_ids)
    assert checksum == RUNNER.source_input_checksum(batch, source_ids.clone())
    assert checksum != RUNNER.source_input_checksum(batch, source_ids.flip(0))

    make_batch_source = inspect.getsource(RUNNER.make_batch)
    evaluation_source = inspect.getsource(RUNNER.run_fixed_evaluation)
    assert "add_node_property" not in make_batch_source
    assert "source_atom_id" not in make_batch_source
    assert "fixed_reference_positions" not in make_batch_source
    assert "gathered.source_atom_id" not in evaluation_source
    assert ".fixed_reference_positions" not in evaluation_source
    assert "predict_gathered_source_ids(" in evaluation_source
    assert "partitioner=derived_layout" in evaluation_source
    assert "owned_reference_positions = owned.positions.detach().clone()" in (
        evaluation_source
    )
    assert evaluation_source.count("owned_reference_positions,") == 2
    assert "diagnostics_by_rank" in evaluation_source
    assert '"gathered_atom_order"' in evaluation_source
    assert "SpatialPartitioner assigns the fixed input to ranks" in evaluation_source


def test_minimum_image_displacement_accepts_periodic_images_and_roundoff() -> None:
    torch = pytest.importorskip("torch")
    cell = torch.diag(torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64)).unsqueeze(
        0
    )
    pbc = torch.tensor([[True, True, True]])
    reference = torch.tensor(
        [[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]],
        dtype=torch.float64,
    )
    lattice_shifts = torch.tensor(
        [[10.0, -11.0, 12.0], [-10.0, 22.0, -12.0]],
        dtype=torch.float64,
    )
    roundoff = torch.tensor(
        [[2.0e-5, -3.0e-5, 4.0e-5], [-1.0e-5, 2.0e-5, -2.0e-5]],
        dtype=torch.float64,
    )
    periodic_image = reference + lattice_shifts + roundoff

    maximum = RUNNER.maximum_minimum_image_displacement_a(
        reference,
        periodic_image,
        cell=cell,
        pbc=pbc,
    )

    assert not torch.equal(periodic_image, reference)
    assert float(maximum.item()) == pytest.approx(
        math.sqrt(2.0**2 + 3.0**2 + 4.0**2) * 1.0e-5,
        abs=1.0e-12,
    )
    assert float(maximum.item()) < (
        DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
    )


def test_minimum_image_displacement_rejects_real_or_nonperiodic_motion() -> None:
    torch = pytest.importorskip("torch")
    cell = torch.diag(torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64)).unsqueeze(
        0
    )
    reference = torch.zeros((1, 3), dtype=torch.float64)

    real_motion = RUNNER.maximum_minimum_image_displacement_a(
        reference,
        torch.tensor([[2.0e-4, 0.0, 0.0]], dtype=torch.float64),
        cell=cell,
        pbc=torch.tensor([[True, True, True]]),
    )
    nonperiodic_shift = RUNNER.maximum_minimum_image_displacement_a(
        reference,
        torch.tensor([[0.0, 11.0, 0.0]], dtype=torch.float64),
        cell=cell,
        pbc=torch.tensor([[True, False, True]]),
    )

    assert float(real_motion.item()) > (
        DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a
    )
    assert float(nonperiodic_shift.item()) == pytest.approx(11.0)


def test_fixed_evaluation_records_pbc_equivalent_position_invariance() -> None:
    source = inspect.getsource(RUNNER.run_fixed_evaluation)

    assert '"position_invariance"' in source
    assert '"method": "maximum_minimum_image_displacement"' in source
    assert '"warmup_maximum_minimum_image_displacement_a"' in source
    assert '"measured_pass_maximum_minimum_image_displacements_a"' in source
    assert '"final_gather_maximum_minimum_image_displacement_a"' in source
    assert '"maximum_minimum_image_displacement_a"' in source
    assert '"all_within_tolerance": True' in source
    assert '"source_input_sha256": input_tensor_hash' in source
    assert "positions_source_atom_order_sha256" not in source
    assert "positions_unchanged" not in source
    assert "source_input_sha256_after" not in source


def test_runtime_gpu_uuid_is_written_as_json_text() -> None:
    runtime_source = inspect.getsource(RUNNER.runtime_row)

    assert "str(gpu_uuid)" in runtime_source
    assert '"gpu_uuid": None if gpu_uuid is None else str(gpu_uuid)' in runtime_source


def test_fixed_timer_uses_one_context_partition_and_gather() -> None:
    source = inspect.getsource(RUNNER.run_fixed_evaluation)

    context_start = source.index("with DomainParallel(")
    partition = source.index("owned = domain.partition(", context_start)
    warmup = source.index("result_owned = domain.run(owned, n_steps=1)", partition)
    measured_loop = source.index(
        "for pass_index in range(1, args.sample_count + 1)",
        warmup,
    )
    timer_start = source.index("started = perf_counter()", measured_loop)
    measured_run = source.index(
        "result_owned = domain.run(result_owned, n_steps=1)",
        timer_start,
    )
    final_sync = source.index("torch.cuda.synchronize(device)", measured_run)
    timer_end = source.index("local_elapsed_s = perf_counter() - started")
    reduction = source.index("dist.all_reduce(max_elapsed", timer_end)
    gather = source.index("gathered = domain.gather(result_owned, dst=0)", reduction)

    assert context_start < partition < warmup < measured_loop
    assert measured_loop < timer_start < measured_run < final_sync < timer_end
    assert timer_end < reduction < gather
    assert source.count("with DomainParallel(") == 1
    assert source.count("domain.partition(") == 1
    assert source.count("domain.gather(") == 1
    assert "evaluator = BaseDynamics(" in source
    assert '"measured_model_evaluations_per_pass"' in source
    assert '"pass_times_s": pass_times_s' in source
    assert '"partition_count": 1' in source
    assert '"gather_count": 1' in source
    assert "n_steps=0" not in source


def test_referenced_files_have_portable_identity_records(tmp_path: Path) -> None:
    structure = tmp_path / "box.extxyz"
    manifest_path = tmp_path / "box.manifest.json"
    structure.write_bytes(b"structure bytes\n")
    manifest_path.write_text('{"atom_count": 25}\n', encoding="utf-8")

    structure_identity = RUNNER.file_identity(structure)
    record = RUNNER.build_input_record(
        input_path=structure,
        tensor_sha256="b" * 64,
        manifest_path=manifest_path,
        manifest={"atom_count": 25},
    )

    assert structure_identity == {
        "path": str(structure.resolve()),
        "sha256": RUNNER.sha256_file(structure),
        "size_bytes": structure.stat().st_size,
    }
    assert record["path"] == structure_identity["path"]
    assert record["file_sha256"] == structure_identity["sha256"]
    assert record["file_size_bytes"] == structure_identity["size_bytes"]
    assert record["manifest_file"] == RUNNER.file_identity(manifest_path)


def test_case_rows_keep_every_generated_file_verifiable() -> None:
    pipeline_source = inspect.getsource(RUNNER.build_complete_pipeline)
    evaluation_source = inspect.getsource(RUNNER.run_fixed_evaluation)
    main_source = inspect.getsource(RUNNER.main)

    assert '"parameter_file_identity": file_identity(d3_parameter_file)' in (
        pipeline_source
    )
    assert '"aimnet_checkpoint_file"' in main_source
    assert '"runner_file"' in main_source
    assert '"forces_source_atom_order_npy"' in evaluation_source
    assert "force_file = file_identity(force_path)" in evaluation_source
    assert "**previous_rank_record" in main_source


def test_runtime_version_check_matches_the_declared_image() -> None:
    versions = dict(RUNNER.EXPECTED_RUNTIME_DISTRIBUTIONS)
    RUNNER.validate_runtime_versions(
        python_major_minor=RUNNER.EXPECTED_PYTHON_MAJOR_MINOR,
        torch_version="2.12.0+cu130",
        torch_cuda_version="13.0",
        distribution_versions=versions,
    )

    wrong = {**versions, "aimnet": "0.1.0"}
    with pytest.raises(RuntimeError, match="aimnet version"):
        RUNNER.validate_runtime_versions(
            python_major_minor=RUNNER.EXPECTED_PYTHON_MAJOR_MINOR,
            torch_version="2.12.0+cu130",
            torch_cuda_version="13.0",
            distribution_versions=wrong,
        )
    with pytest.raises(RuntimeError, match="Python"):
        RUNNER.validate_runtime_versions(
            python_major_minor=(3, 13),
            torch_version="2.12.0+cu130",
            torch_cuda_version="13.0",
            distribution_versions=versions,
        )
    with pytest.raises(RuntimeError, match="Torch"):
        RUNNER.validate_runtime_versions(
            python_major_minor=RUNNER.EXPECTED_PYTHON_MAJOR_MINOR,
            torch_version="2.11.0+cu130",
            torch_cuda_version="13.0",
            distribution_versions=versions,
        )
    with pytest.raises(RuntimeError, match="Torch CUDA"):
        RUNNER.validate_runtime_versions(
            python_major_minor=RUNNER.EXPECTED_PYTHON_MAJOR_MINOR,
            torch_version="2.12.0+cu130",
            torch_cuda_version="12.8",
            distribution_versions=versions,
        )


def test_runtime_rows_must_use_the_same_software_and_gpu() -> None:
    row = {
        "gpu_name": "NVIDIA H100 NVL",
        "gpu_total_memory_bytes": 1,
        "compute_capability": [9, 0],
        "torch_version": "2.12.0+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91002,
        "nccl_version": [2, 27, 5],
        "driver_version": "580.65.06",
        "python_version": "3.12.13",
        "python_executable": "/shared/env/bin/python",
        "python_prefix": "/shared/env",
        "distribution_versions": dict(RUNNER.EXPECTED_RUNTIME_DISTRIBUTIONS),
    }
    software = {
        key: row[key]
        for key in (
            "python_version",
            "python_executable",
            "python_prefix",
            "torch_version",
            "torch_cuda_version",
            "distribution_versions",
        )
    }
    RUNNER.validate_runtime_rows(
        [row, {**row, "rank": 1}],
        expected_software=software,
    )

    with pytest.raises(RuntimeError, match="identity differs"):
        RUNNER.validate_runtime_rows([row, {**row, "torch_version": "2.12.1+cu130"}])
    with pytest.raises(RuntimeError, match="checked source record"):
        RUNNER.validate_runtime_rows(
            [row],
            expected_software={**software, "torch_version": "2.12.1+cu130"},
        )
    with pytest.raises(RuntimeError, match="no per-rank"):
        RUNNER.validate_runtime_rows([])


def test_timing_summary_and_role_counts_are_explicit() -> None:
    summary = RUNNER.summarize_timing_samples([1.0, 2.0, 3.0])
    assert summary == {"median_s": 2.0, "min_s": 1.0, "max_s": 3.0}
    args = argparse.Namespace(
        mode="distributed",
        world_size=2,
        pair_count=DOMAIN_METHODOLOGY.fixed_molecules_per_species,
    )
    RUNNER.validate_measurement_args(args)
    assert args.measurement_role == "fixed_evaluation"
    assert args.warmup_count == 1
    assert args.sample_count == 3

    wrong_size = argparse.Namespace(
        mode="distributed",
        world_size=8,
        pair_count=DOMAIN_METHODOLOGY.fixed_molecules_per_species,
    )
    with pytest.raises(ValueError, match="world size"):
        RUNNER.validate_measurement_args(wrong_size)

    wrong_input = argparse.Namespace(
        mode="distributed",
        world_size=1,
        pair_count=128,
    )
    with pytest.raises(ValueError, match="fixed 2048-pair input"):
        RUNNER.validate_measurement_args(wrong_input)


def test_runner_cli_has_only_the_two_fixed_work_modes() -> None:
    source = inspect.getsource(RUNNER.parse_args)

    assert 'choices=("distributed", "electrostatics-validation")' in source
    assert "--measurement-role" not in source
    assert "--warmup-count" not in source
    assert "--sample-count" not in source
    for removed in ("capacity", "parity", "rescue", "steady-timing"):
        assert removed not in source


def test_planner_builds_one_fixed_case_for_each_gpu_count() -> None:
    plans = {
        world_size: PLAN.build_plan(
            run_id=f"domain-{world_size}",
            tutorial_commit="1" * 40,
            world_size=world_size,
        )
        for world_size in PLAN.DEFAULT_WORLD_SIZES
    }

    for world_size, plan in plans.items():
        fixed = plan["fixed_case"]
        assert fixed["case_id"] == PLAN.fixed_case_id(2_048, world_size)
        assert fixed["world_size"] == world_size
        assert fixed["pair_count"] == 2_048
        assert fixed["atom_count"] == 51_200
        assert plan["planned_case_count"] == (2 if world_size == 1 else 1)
        assert bool(plan["validation_cases"]) is (world_size == 1)

    with pytest.raises(ValueError, match="world_size must be one of"):
        PLAN.build_plan(
            run_id="domain-3",
            tutorial_commit="1" * 40,
            world_size=3,
        )


def test_fixed_input_preparation_does_not_launch_packmol() -> None:
    source = inspect.getsource(PLAN.prepare_input)

    assert "subprocess" not in source
    assert "args.packmol" not in source


def test_runtime_source_check_requires_a_pinned_clean_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "toolkit-ops"
    commit = _clean_git_checkout(checkout)

    observed = RUNNER._verify_clean_git_checkout(
        str(checkout),
        expected_commit=commit,
        label="Toolkit-Ops",
    )
    assert observed == {
        "root": str(checkout.resolve()),
        "commit": commit,
    }

    with pytest.raises(RuntimeError, match="expected commit"):
        RUNNER._verify_clean_git_checkout(
            str(checkout),
            expected_commit="0" * 40,
            label="Toolkit-Ops",
        )

    (checkout / ".gitignore").write_text("ignored.cache\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(checkout), "add", ".gitignore"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "ignore cache"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    (checkout / "ignored.cache").write_text("generated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains ignored files"):
        RUNNER._verify_clean_git_checkout(
            str(checkout),
            expected_commit=commit,
            label="Toolkit-Ops",
        )
    (checkout / "ignored.cache").unlink()

    (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        RUNNER._verify_clean_git_checkout(
            str(checkout),
            expected_commit=commit,
            label="Toolkit-Ops",
        )
    runtime_check = inspect.getsource(RUNNER.verify_runtime_source)
    assert "ALCHEMI_TOOLKIT_CORE_ROOT" in runtime_check
    assert "ALCHEMI_TOOLKIT_OPS_ROOT" in runtime_check


def test_tutorial_runner_requires_one_clean_tracked_revision(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "tutorial"
    _clean_git_checkout(checkout)
    tracked = checkout / "tracked.txt"

    observed = RUNNER._verify_clean_repository(
        checkout,
        required_paths=(tracked,),
    )
    assert observed["commit"]
    assert observed["tree"]
    assert observed["clean"] is True
    assert observed["tracked_required_paths"] == ["tracked.txt"]

    (checkout / "new-file.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        RUNNER._verify_clean_repository(
            checkout,
            required_paths=(tracked,),
        )


def test_runtime_source_check_allows_absent_direct_url_metadata() -> None:
    expected = "1" * 40

    RUNNER._verify_optional_direct_url_commit(
        None,
        expected_commit=expected,
        label="Toolkit-Ops",
    )
    RUNNER._verify_optional_direct_url_commit(
        expected,
        expected_commit=expected,
        label="Toolkit-Ops",
    )
    assert RUNNER.direct_url_commit("package-that-does-not-exist-domain-test") is None


def test_runtime_source_check_rejects_conflicting_direct_url_metadata() -> None:
    with pytest.raises(RuntimeError, match="metadata reports commit"):
        RUNNER._verify_optional_direct_url_commit(
            "1" * 40,
            expected_commit="2" * 40,
            label="Toolkit Core",
        )


def test_launchers_use_both_pinned_source_checkouts() -> None:
    launcher = (REPO_ROOT / "scripts" / "run_part1_domain_decomposition.sh").read_text(
        encoding="utf-8"
    )
    slurm = (
        REPO_ROOT / "scripts" / "slurm_part1_domain_decomposition.sbatch"
    ).read_text(encoding="utf-8")

    assert "ALCHEMI_TOOLKIT_OPS_ROOT" in launcher
    assert 'PYTHONPATH="$CORE_ROOT:$OPS_ROOT' in launcher
    assert 'git -C "$OPS_ROOT" rev-parse HEAD' in slurm
    assert 'git -C "$OPS_ROOT" status --porcelain --untracked-files=all' in slurm
    assert 'export ALCHEMI_TOOLKIT_OPS_ROOT="$OPS_ROOT"' in slurm
