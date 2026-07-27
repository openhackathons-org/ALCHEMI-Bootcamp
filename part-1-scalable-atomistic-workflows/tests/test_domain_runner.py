"""Static and CPU-only checks for the Part 1 domain campaign scripts."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
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
        world_sizes=(1,),
        capacity_pair_counts=PLAN.DEFAULT_CAPACITY_PAIR_COUNTS,
        validation_pairs=PLAN.DEFAULT_VALIDATION_PAIRS,
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


def _selected_input(count: int) -> dict[str, object]:
    return {
        "pair_count": count,
        "atom_count": count * PLAN.ATOMS_PER_PAIR,
        "structure": {"path": f"/tmp/input-{count}.extxyz", "sha256": "a" * 64},
    }


def _campaign_selection() -> dict[str, object]:
    return {
        "largest_success": {"input": _selected_input(4_096)},
        "first_cuda_oom": {"input": _selected_input(8_192)},
    }


def _campaign_rows(
    selection: dict[str, object],
    *,
    successful_rescue_gpus: tuple[int, ...] = (),
) -> dict[str, dict[str, object]]:
    largest = selection["largest_success"]["input"]["pair_count"]  # type: ignore[index]
    first_oom = selection["first_cuda_oom"]["input"]["pair_count"]  # type: ignore[index]
    rows: dict[str, dict[str, object]] = {}
    for world_size in (1, 2, 4):
        rows[PLAN.steady_timing_case_id(largest, world_size)] = {
            "success": True,
            "measurement_role": "steady_timing",
        }
    for world_size in (2, 4):
        rows[PLAN.rescue_case_id(first_oom, world_size)] = {
            "success": world_size in successful_rescue_gpus
        }
    return rows


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


def test_plan_uses_equal_independent_molecule_counts() -> None:
    plan = _plan()

    assert (
        tuple(
            case["pair_count"]
            for case in plan["capacity_cases"]  # type: ignore[index]
        )
        == PLAN.DEFAULT_CAPACITY_PAIR_COUNTS
    )
    assert plan["capacity_cases"][-1]["atom_count"] == (  # type: ignore[index]
        PLAN.DEFAULT_CAPACITY_PAIR_COUNTS[-1]
        * DOMAIN_METHODOLOGY.atoms_per_composition_unit
    )
    assert all(
        case["molecules_per_species"] == case["pair_count"]
        for case in plan["capacity_cases"]  # type: ignore[index]
    )
    count_definition = plan["input"]["count_definition"]  # type: ignore[index]
    assert "independent phenol molecules" in count_definition
    assert "not a count of pre-bound dimers" in count_definition
    assert (
        plan["input"]["packmol_precision_a"]  # type: ignore[index]
        == PLAN.DEFAULT_PACKMOL_PRECISION_A
    )
    assert plan["source"]["aimnet_checkpoint"] == "aimnet2-wb97m-d3_0"  # type: ignore[index]
    assert plan["model"]["neighbor_adaptation"] == "never"  # type: ignore[index]
    assert plan["distributed"]["grid_dims"] is None  # type: ignore[index]
    assert "actual cell shape" in plan["distributed"]["rank_grid_policy"]  # type: ignore[index]
    assert all(
        case["rank_grid_policy"] == "automatic_from_actual_cell"
        for case in plan["capacity_cases"]  # type: ignore[index]
    )
    two_box_case = plan["capacity_cases"][1]  # type: ignore[index]
    base_length_a = PLAN.equivalent_cubic_length_angstrom(
        PLAN.BASE_PAIR_COUNT,
        PLAN.DEFAULT_DENSITY_G_CM3,
    )
    assert two_box_case["cell_geometry"] == "orthorhombic"
    assert two_box_case["cell_lengths_a"] == pytest.approx(
        [base_length_a, base_length_a, 2.0 * base_length_a]
    )
    assert two_box_case["minimum_cell_length_a"] == pytest.approx(base_length_a)
    assert two_box_case["equivalent_cubic_length_a"] == pytest.approx(
        base_length_a * 2.0 ** (1.0 / 3.0)
    )
    assert two_box_case["volume_a3"] == pytest.approx(2.0 * base_length_a**3)
    assert "box_length_a" not in two_box_case
    assert plan["model"]["pme_cutoff_a"] == 12.0  # type: ignore[index]
    assert (  # type: ignore[index]
        plan["model"]["pme_mesh_safety_factor"]
        == DOMAIN_METHODOLOGY.pme_mesh_safety_factor
    )
    assert plan["model"]["pme_parameter_rule"] == (  # type: ignore[index]
        "estimate_pme_parameters(accuracy, real_space_cutoff, mesh_safety_factor)"
    )
    assert (  # type: ignore[index]
        plan["model"]["pme_accuracy"] == DOMAIN_METHODOLOGY.pme_accuracy
    )
    assert (  # type: ignore[index]
        plan["model"]["ewald_reference_accuracy"]
        == DOMAIN_METHODOLOGY.ewald_reference_accuracy
    )
    assert plan["methodology"]["source"] == DOMAIN_METHODOLOGY.as_record()
    assert (
        plan["methodology"]["source_identity"]["sha256"]
        == PLAN.sha256_file(PLAN.DOMAIN_METHODOLOGY_CONFIG_PATH)
    )
    assert (
        plan["methodology"]["resolved_values"][
            "capacity_molecules_per_species"
        ]
        == list(PLAN.DEFAULT_CAPACITY_PAIR_COUNTS)
    )


def test_planner_and_runner_defaults_come_from_domain_settings() -> None:
    assert PLAN.NCI_SYSTEM_ID == DOMAIN_METHODOLOGY.nci_system_id
    assert PLAN.NCI_SCALE == DOMAIN_METHODOLOGY.nci_scale
    assert (
        PLAN.DEFAULT_CAPACITY_PAIR_COUNTS
        == DOMAIN_METHODOLOGY.capacity_molecules_per_species
    )
    assert (
        PLAN.DEFAULT_PARITY_PAIR_COUNT
        == DOMAIN_METHODOLOGY.parity_molecules_per_species
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
        PLAN.AIMNET_NEIGHBOR_CUTOFF_A
        == RUNNER.EXPECTED_AIMNET_NEIGHBOR_CUTOFF_A
        == DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a
    )
    assert (
        PLAN.DEFAULT_STEADY_TIMING_WARMUP_COUNT
        == RUNNER.DEFAULT_STEADY_TIMING_WARMUP_COUNT
        == DOMAIN_METHODOLOGY.steady_timing_warmup_count
    )
    assert (
        PLAN.DEFAULT_STEADY_TIMING_SAMPLE_COUNT
        == RUNNER.DEFAULT_STEADY_TIMING_SAMPLE_COUNT
        == DOMAIN_METHODOLOGY.steady_timing_sample_count
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
    eta = (volume_a3**2 / 6_400) ** (1.0 / 6.0) / math.sqrt(
        2.0 * math.pi
    )
    error_factor = math.sqrt(
        -2.0 * math.log(PLAN.DEFAULT_EWALD_REFERENCE_ACCURACY)
    )

    assert setup["real_space_cutoff_a"] == pytest.approx(error_factor * eta)
    assert setup["reciprocal_space_cutoff_a_inverse"] == pytest.approx(
        error_factor / eta
    )
    assert setup["alpha_a_inverse"] == pytest.approx(
        1.0 / (math.sqrt(2.0) * eta)
    )


def test_domain_partition_check_defers_to_the_actual_toolkit_partition() -> None:
    check = PLAN.domain_partition_check(
        cell_lengths_a=(65.0, 90.0, 120.0),
        ghost_width_a=19.0,
    )

    assert check["cell_lengths_a"] == [65.0, 90.0, 120.0]
    assert check["ghost_width_a"] == 19.0
    assert check["checked_during_each_multi_gpu_run"] is True
    assert check["box_length_only_precheck"] == "not_used"
    assert "require_nondegenerate=True" in check["acceptance_rule"]


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
    assert PLAN.CAPACITY_COLUMNS == lesson_results.CAPACITY_COLUMNS
    assert PLAN.PARITY_COLUMNS == lesson_results.PARITY_COLUMNS
    assert PLAN.DISTRIBUTED_COLUMNS == lesson_results.DISTRIBUTED_COLUMNS
    assert "molecules_per_species" in PLAN.DISTRIBUTED_COLUMNS
    assert "measurement_role" in PLAN.DISTRIBUTED_COLUMNS
    assert "elapsed_samples_s" in PLAN.DISTRIBUTED_COLUMNS
    assert "elapsed_iqr_s" in PLAN.DISTRIBUTED_COLUMNS
    assert "atom_evaluations_per_s" not in PLAN.DISTRIBUTED_COLUMNS
    assert "molecule_pairs" not in PLAN.DISTRIBUTED_COLUMNS
    assert "atom_steps_per_s" not in PLAN.DISTRIBUTED_COLUMNS


def test_runner_keeps_global_energy_separate_from_gathered_atom_fields(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(RUNNER.run_capacity)
    order_source = inspect.getsource(RUNNER.source_order_from_gathered_ids)

    assert "replicated_energy = result_owned.energy.detach().clone()" in source
    assert "gathered = domain.gather(result_owned, dst=0)" in source
    assert "energy_ev = float(replicated_energy.reshape(-1)[0].item())" in source
    assert "sorted_forces = gathered.forces[order]" in source
    assert "float(gathered.energy" not in source
    assert "torch.isfinite(result_owned.energy)" in source
    assert "torch.isfinite(result_owned.forces)" in source
    assert "torch.isfinite(gathered.positions)" in source
    assert "torch.isfinite(gathered.forces)" in source
    assert "source_order_from_gathered_ids(" in source
    assert "gathered.source_atom_id" not in source
    assert "source_atom_id is not an exact 0..N-1 permutation" in order_source

    output = tmp_path / "strict.json"
    with pytest.raises(ValueError, match="Out of range float"):
        RUNNER.atomic_write_json(output, {"bad": float("nan")})
    assert not output.exists()


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
    capacity_source = inspect.getsource(RUNNER.run_capacity)
    assert "source_atom_id" not in make_batch_source
    assert "predict_gathered_source_ids(" in capacity_source
    assert "partitioner=derived_layout" in capacity_source
    assert "world_size > 1 and run_steps != 1" in capacity_source
    assert '"gathered_atom_order"' in capacity_source
    assert "SpatialPartitioner.assign_atoms_to_ranks" in capacity_source


def test_runtime_gpu_uuid_is_written_as_json_text() -> None:
    runtime_source = inspect.getsource(RUNNER.runtime_row)

    assert "str(gpu_uuid)" in runtime_source
    assert '"gpu_uuid": None if gpu_uuid is None else str(gpu_uuid)' in runtime_source


def test_steady_timer_covers_repeated_fresh_public_workflows() -> None:
    source = inspect.getsource(RUNNER.run_capacity)

    context_start = source.index("with DomainParallel(")
    timer_start = source.index("start = perf_counter()", context_start)
    partition = source.index("owned = domain.partition(")
    model_run = source.index("result_owned = domain.run(")
    gather = source.index("domain.gather(")
    final_sync = source.index("torch.cuda.synchronize(device)", gather)
    timer_end = source.index("local_elapsed_s = perf_counter() - start")
    reduction = source.index("dist.all_reduce(max_elapsed", timer_end)
    input_check = source.index("observed_hash = source_input_checksum", timer_end)
    file_write = source.index("# Shared-filesystem writes happen only after")
    energy_snapshot = source.index(
        "replicated_energy = result_owned.energy.detach().clone()"
    )

    assert context_start < timer_start < partition < model_run < gather
    assert gather < final_sync < timer_end < reduction < file_write
    assert timer_end < input_check
    assert timer_end < energy_snapshot
    assert "for warmup_index in range(args.warmup_count)" in source
    assert "for sample_index in range(args.sample_count)" in source
    assert source.count("with DomainParallel(") == 1
    assert "inner = BaseDynamics(" in source
    assert "DOMAIN_METHODOLOGY.steady_timing_run_steps(world_size)" in source
    assert (
        "DOMAIN_METHODOLOGY"
        ".domain_parallel_multi_rank_initial_force_evaluations"
    ) in source
    assert (
        "model_evaluations_per_workflow = run_steps + "
        "automatic_initial_evaluations"
    ) in source
    assert "result_owned = domain.run(owned, n_steps=run_steps)" in source
    assert '"model_evaluations_per_workflow": model_evaluations_per_workflow' in source
    assert '"samples_s_max_rank": samples_s_max_rank' in source


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
    capacity_source = inspect.getsource(RUNNER.run_capacity)
    main_source = inspect.getsource(RUNNER.main)

    assert '"parameter_file_identity": file_identity(d3_parameter_file)' in (
        pipeline_source
    )
    assert '"aimnet_checkpoint_file"' in main_source
    assert '"runner_file"' in main_source
    assert '"forces_source_atom_order_npy"' in capacity_source
    assert "force_file = file_identity(force_path)" in capacity_source
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
    summary = RUNNER.summarize_timing_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary == {"median_s": 3.0, "q1_s": 2.0, "q3_s": 4.0, "iqr_s": 2.0}
    args = argparse.Namespace(
        mode="steady-timing",
        measurement_role="steady_timing",
        warmup_count=1,
        sample_count=5,
    )
    RUNNER.validate_measurement_args(args)
    args.sample_count = 4
    with pytest.raises(ValueError, match="at least five"):
        RUNNER.validate_measurement_args(args)


def test_derive_uses_parity_on_two_and_four_gpus(tmp_path: Path) -> None:
    selection = {
        "schema": PLAN.SELECTION_SCHEMA,
        "run_id": "domain-test",
        "parity_reference": {
            "input": _selected_input(2_048),
            "acceptance": {"energy_atol_ev": 0.1},
        },
        "largest_success": {"input": _selected_input(4_096)},
        "first_cuda_oom": {"input": _selected_input(8_192)},
        "settings": {
            "timing": {
                "steady": {
                    "warmup_count": DOMAIN_METHODOLOGY.steady_timing_warmup_count,
                    "sample_count": DOMAIN_METHODOLOGY.steady_timing_sample_count,
                }
            }
        },
        "methodology": {
            "source": DOMAIN_METHODOLOGY.as_record(),
            "source_identity": PLAN.methodology_source_identity(),
            "resolved_values": DOMAIN_METHODOLOGY.resolved_values(
                json_compatible=True
            ),
        },
        "source": {},
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    observed: dict[int, list[str]] = {}
    for world_size in (2, 4):
        output = tmp_path / f"plan-{world_size}.json"
        plan = PLAN.derive_distributed_plan(
            argparse.Namespace(
                selection=selection_path,
                world_size=world_size,
                output=output,
            )
        )
        observed[world_size] = [case["series"] for case in plan["cases"]]

    assert observed == {
        2: ["parity", "steady_timing", "rescue"],
        4: ["parity", "steady_timing", "rescue"],
    }

    with pytest.raises(ValueError, match="distributed world size"):
        PLAN.derive_distributed_plan(
            argparse.Namespace(
                selection=selection_path,
                world_size=3,
                output=tmp_path / "plan-3.json",
            )
        )


def test_campaign_input_preparation_does_not_launch_packmol() -> None:
    source = inspect.getsource(PLAN.prepare_input)

    assert "subprocess" not in source
    assert "args.packmol" not in source


def test_complete_campaign_requires_every_steady_case_to_succeed() -> None:
    selection = _campaign_selection()
    for world_size in (2, 4):
        rows = _campaign_rows(selection, successful_rescue_gpus=(4,))
        largest = selection["largest_success"]["input"]["pair_count"]  # type: ignore[index]
        rows[PLAN.steady_timing_case_id(largest, world_size)]["success"] = False

        with pytest.raises(
            ValueError,
            match=rf"{world_size}-GPU steady-timing case did not complete",
        ):
            PLAN._require_complete_distributed_outcomes(selection, rows)


def test_complete_campaign_requires_the_oom_input_to_be_rescued() -> None:
    selection = _campaign_selection()
    rows = _campaign_rows(selection)

    with pytest.raises(ValueError, match="did not succeed on any declared"):
        PLAN._require_complete_distributed_outcomes(selection, rows)

    rows = _campaign_rows(selection, successful_rescue_gpus=(4,))
    assert PLAN._require_complete_distributed_outcomes(selection, rows) == (4,)
    assert "_require_complete_distributed_outcomes(" in inspect.getsource(
        PLAN.build_bundle
    )


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
