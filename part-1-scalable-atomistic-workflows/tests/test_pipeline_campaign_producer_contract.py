"""Focused provenance checks for the Compute Lab campaign launcher."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from part1_distributed_campaign_contract import (  # noqa: E402
    BATCHES_PER_PIPELINE_AT_MAX_SCALE,
    CORE_BRANCH,
    CORE_COMMIT,
    CORE_VERSION,
    CURRENT_RC_TIMING_STATUS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SYSTEMS,
    MAX_PIPELINES,
    OPS_COMMIT,
    OPS_VERSION,
    PRODUCER_FILES,
    ROUTE_PAIR_BOUNDARIES,
    pair_boundaries_for_world_size,
)


def _load_driver_function(name: str) -> Any:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, Any] = {
        "Any": Any,
        "Batch": Any,
        "Mapping": Mapping,
        "PipelineModelWrapper": Any,
        "torch": torch,
    }
    exec(compile(module, str(driver), "exec"), namespace)
    return namespace[name]


def test_producer_files_cover_every_local_helper_imported_by_driver() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    imported_helpers = {
        f"part-1-scalable-atomistic-workflows/{node.module.replace('.', '/')}.py"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("aux.")
    }

    assert imported_helpers == {
        "part-1-scalable-atomistic-workflows/aux/artifacts.py",
        "part-1-scalable-atomistic-workflows/aux/checkpoint.py",
        "part-1-scalable-atomistic-workflows/aux/electrostatics.py",
        "part-1-scalable-atomistic-workflows/aux/hooks.py",
        "part-1-scalable-atomistic-workflows/aux/runtime.py",
        "part-1-scalable-atomistic-workflows/aux/structures.py",
    }
    assert imported_helpers <= set(PRODUCER_FILES)


def test_slurm_checksum_index_covers_the_exact_producer_contract() -> None:
    launcher = (
        REPOSITORY_ROOT / "scripts/slurm_part1_distributed_campaign.sbatch"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"^PRODUCER_RELATIVE_PATHS=\(\n(?P<body>.*?)^\)\n",
        launcher,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert match is not None
    recorded_paths = tuple(shlex.split(match.group("body"), comments=True))
    assert recorded_paths == PRODUCER_FILES
    assert 'for relative_path in "${PRODUCER_RELATIVE_PATHS[@]}"' in launcher
    assert 'producer_paths+=("$SHARED_REPO/$relative_path")' in launcher
    assert 'sha256sum "${producer_paths[@]}"' in launcher


def test_campaign_is_pinned_to_the_stock_toolkit_rc() -> None:
    assert CORE_BRANCH == "0.2.0-rc"
    assert CORE_VERSION == "0.2.0"
    assert CORE_COMMIT == "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
    assert OPS_VERSION == "0.4.0"
    assert OPS_COMMIT == "e8e7a7464f6745277a156a3d6f433d06b58c60e3"

    active_paths = (
        "scripts/benchmark_part1_distributed_campaign.py",
        "scripts/part1_distributed_campaign_contract.py",
        "scripts/record_part1_campaign_failure.py",
        "scripts/run_part1_distributed_torchrun.sh",
        "scripts/slurm_part1_distributed_campaign.sbatch",
        "scripts/slurm_part1_pipeline_tuning.sbatch",
        "scripts/assemble_part1_pipeline_campaign.py",
        "scripts/validate_part1_ir_run.py",
        "part-1-scalable-atomistic-workflows/aux/pipeline_campaign_results.py",
    )
    active_text = "\n".join(
        (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        for path in active_paths
    )
    assert "b770ee6963fd2f6137891e408c370012751918e2" not in active_text
    assert "sustained-pipeline-compat.patch" not in active_text
    assert "ALCHEMI_CORE_PATCH_SHA256" not in active_text
    assert "ALCHEMI_CORE_OVERLAY" not in active_text


def test_campaign_checks_reusable_buffer_transfer_before_cuda_or_distributed_init() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_batch_put_roundtrip"
    )
    validator_source = ast.unparse(validator)
    transfer_helper = (
        REPOSITORY_ROOT
        / "part-1-scalable-atomistic-workflows/aux/runtime.py"
    ).read_text(encoding="utf-8")

    runtime_check = source.index("_validate_runtime(core_root)")
    cuda_check = source.index("torch.cuda.is_available()")
    distributed_init = source.index("dist.init_process_group")
    assert runtime_check < cuda_check < distributed_init
    assert "check_batch_buffer_transfer('cpu')" in validator_source
    assert "CURRENT_RC_TIMING_STATUS" in validator_source
    assert 'checks["first_put"]' in transfer_helper
    assert 'checks["repeated_put"]' in transfer_helper
    assert 'checks["zero_then_put"]' in transfer_helper
    assert "all_fields=False" not in transfer_helper
    assert transfer_helper.count("payload_mismatches(") == 4

    launcher = (
        REPOSITORY_ROOT / "scripts/slurm_part1_distributed_campaign.sbatch"
    ).read_text(encoding="utf-8")
    preflight = launcher.index("validate_batch_put_roundtrip")
    first_srun = launcher.index("\nsrun --ntasks")
    assert preflight < first_srun
    assert CURRENT_RC_TIMING_STATUS.startswith("NOT REPORTED:")
    assert "fixes reusable-buffer capacity" in CURRENT_RC_TIMING_STATUS
    assert "Batch.put still copies only float32" in CURRENT_RC_TIMING_STATUS
    assert "integer fields including atomic_numbers" in CURRENT_RC_TIMING_STATUS
    assert (
        "global completion check (all_reduce) after every iteration"
        in CURRENT_RC_TIMING_STATUS
    )
    assert "stage-overlap checks" in CURRENT_RC_TIMING_STATUS


def test_campaign_partitions_full_workload_across_sustained_pipelines() -> None:
    """Every upstream sampler owns its full, balanced campaign partition."""

    assert DEFAULT_BATCH_SIZE >= 256
    assert BATCHES_PER_PIPELINE_AT_MAX_SCALE >= 8
    assert MAX_PIPELINES == 2
    assert DEFAULT_SYSTEMS == (
        DEFAULT_BATCH_SIZE * BATCHES_PER_PIPELINE_AT_MAX_SCALE * MAX_PIPELINES
    )

    partition_ids = _load_driver_function("_partition_campaign_ids")
    two_gpu_partitions = partition_ids(DEFAULT_SYSTEMS, 1)
    four_gpu_partitions = partition_ids(DEFAULT_SYSTEMS, 2)

    assert two_gpu_partitions == [tuple(range(DEFAULT_SYSTEMS))]
    assert [len(ids) for ids in four_gpu_partitions] == [4096, 4096]
    assert four_gpu_partitions[0] == tuple(range(0, DEFAULT_SYSTEMS, 2))
    assert four_gpu_partitions[1] == tuple(range(1, DEFAULT_SYSTEMS, 2))
    assert len(two_gpu_partitions[0]) // DEFAULT_BATCH_SIZE == 16
    assert all(
        len(campaign_ids) // DEFAULT_BATCH_SIZE
        == BATCHES_PER_PIPELINE_AT_MAX_SCALE
        for campaign_ids in four_gpu_partitions
    )

    launcher = (
        REPOSITORY_ROOT / "scripts/slurm_part1_distributed_campaign.sbatch"
    ).read_text(encoding="utf-8")
    assert 'SYSTEMS="${CAMPAIGN_CONTRACT[5]}"' in launcher
    assert 'BATCH_SIZE="${CAMPAIGN_CONTRACT[6]}"' in launcher
    assert 'BALANCE_SYSTEMS="${CAMPAIGN_CONTRACT[7]}"' in launcher
    assert "c.DEFAULT_SYSTEMS" in launcher
    assert "c.DEFAULT_BATCH_SIZE" in launcher
    assert "c.BALANCE_PROBE_SYSTEMS" in launcher


def test_tuning_sweep_derives_multi_batch_workloads_and_keeps_failed_rows() -> None:
    tuning = (
        REPOSITORY_ROOT / "scripts/slurm_part1_pipeline_tuning.sbatch"
    ).read_text(encoding="utf-8")

    assert "ALCHEMI_PROBE_BATCHES_PER_PIPELINE" in tuning
    assert (
        'systems="$((batch_size * BATCHES_PER_PIPELINE * PIPELINE_COUNT))"'
        in tuning
    )
    assert "--purpose tuning" in tuning
    assert 'srun_status="${PIPESTATUS[0]}"' in tuning
    assert '"$FAILURE_RECORDER"' in tuning
    assert 'if [[ "$saved_success" == false ]]' in tuning


def test_campaign_dynamics_tensors_start_in_aimnet_float32() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    dataset = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WaterHexamerDataset"
    )
    dataset_source = ast.unparse(dataset)

    assert "dtype=torch.float32" in dataset_source
    assert "dtype=torch.float64" not in dataset_source


def test_sampler_refill_is_not_blocked_by_rebuilt_neighbor_edges() -> None:
    """RC refill_check reads live edge counts, so zero is not 'disabled'."""

    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    make_sampler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_sampler"
    )
    sampler_call = next(
        node
        for node in ast.walk(make_sampler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SizeAwareSampler"
    )
    keywords = {keyword.arg: keyword.value for keyword in sampler_call.keywords}

    assert isinstance(keywords["max_edges"], ast.Constant)
    assert keywords["max_edges"].value is None
    assert "max_atoms=batch_size * ATOMS_PER_SYSTEM" in ast.unparse(sampler_call)


def test_distributed_sampler_owns_full_partition_with_bounded_active_batch() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_distributed_workflow"
    )
    source = ast.unparse(builder)
    sampler_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_sampler"
    )
    fire_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_fire"
    )
    fire_keywords = {keyword.arg: keyword.value for keyword in fire_call.keywords}

    assert ast.unparse(sampler_call) == (
        "_make_sampler(structures, campaign_ids, device, batch_size)"
    )
    assert "len(campaign_ids) > batch_size" not in source
    assert ast.unparse(fire_keywords["sampler"]) == "sampler"
    assert ast.unparse(fire_keywords["max_batch_size"]) == "batch_size"
    assert ast.unparse(fire_keywords["refill_frequency"]) == "1"


def test_four_rank_topology_has_two_explicit_pair_boundaries() -> None:
    assert pair_boundaries_for_world_size(2) == ((0, 1),)
    assert pair_boundaries_for_world_size(4) == ((0, 1), (2, 3))
    assert ROUTE_PAIR_BOUNDARIES["pipeline_4gpu"] == ((0, 1), (2, 3))
    assert (1, 2) not in ROUTE_PAIR_BOUNDARIES["pipeline_4gpu"]

    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_distributed_workflow"
    )
    fire_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_fire"
    )
    downstream_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedStage"
    )
    fire_keywords = {keyword.arg: keyword.value for keyword in fire_call.keywords}
    downstream_keywords = {
        keyword.arg: keyword.value for keyword in downstream_call.keywords
    }
    assert isinstance(fire_keywords["prior_rank"], ast.Constant)
    assert fire_keywords["prior_rank"].value is None
    assert isinstance(fire_keywords["next_rank"], ast.Name)
    assert fire_keywords["next_rank"].id == "downstream_rank"
    assert isinstance(downstream_keywords["prior_rank"], ast.Name)
    assert downstream_keywords["prior_rank"].id == "upstream_rank"
    assert isinstance(downstream_keywords["next_rank"], ast.Constant)
    assert downstream_keywords["next_rank"].value is None


def test_driver_uses_stock_public_toolkit_imports() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("nvalchemi")
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("nvalchemi")
    )
    assert modules == {
        "nvalchemi",
        "nvalchemi.data",
        "nvalchemi.dynamics",
        "nvalchemi.dynamics.base",
        "nvalchemi.dynamics.hooks",
        "nvalchemi.hooks",
        "nvalchemi.models",
        "nvalchemi.neighbors",
    }
    assert all("._" not in module for module in modules)

    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_runtime"
    )
    validator_source = ast.unparse(validator)
    assert "'status', '--porcelain', '--untracked-files=all'" in validator_source
    assert "nvalchemi.version" in validator_source
    assert "metadata.version('nvalchemi-toolkit-ops')" in validator_source


def test_timeout_writes_checksum_indexes_before_exiting() -> None:
    launcher = (
        REPOSITORY_ROOT / "scripts/slurm_part1_distributed_campaign.sbatch"
    ).read_text(encoding="utf-8")
    timeout_body = launcher.split("record_timeout_failure() {", maxsplit=1)[1]
    timeout_body = timeout_body.split(
        "\n}\n\ntrap record_timeout_failure TERM", maxsplit=1
    )[0]

    checksum_call = timeout_body.rfind("\n  write_checksum_indexes")
    exit_call = timeout_body.rfind("\n  exit 124")
    assert 0 <= checksum_call < exit_call
    assert "\nwrite_checksum_indexes\ndate -Is" in launcher


def test_model_record_includes_checkpoint_d3_bj_parameters() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    model_build = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelBuild"
    )
    fields = {
        node.target.id
        for node in model_build.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    model_record = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_model_record"
    )
    returned = next(
        node.value
        for node in ast.walk(model_record)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    record_keys = {
        key.value for key in returned.keys if isinstance(key, ast.Constant)
    }

    assert "d3_bj_parameters" in fields
    assert "d3_bj_parameters" in record_keys


def test_step_counter_records_the_stage_at_step_entry() -> None:
    """Do not count the final FIRE2 update as the first NVT update."""

    helper = (
        REPOSITORY_ROOT
        / "part-1-scalable-atomistic-workflows/aux/hooks.py"
    )
    tree = ast.parse(helper.read_text(encoding="utf-8"), filename=str(helper))
    counter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StageStepCounterHook"
    )
    stage_assignment = next(
        node
        for node in counter.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "stage"
            for target in node.targets
        )
    )

    assert isinstance(stage_assignment.value, ast.Attribute)
    assert isinstance(stage_assignment.value.value, ast.Name)
    assert stage_assignment.value.value.id == "DynamicsStage"
    assert stage_assignment.value.attr == "BEFORE_STEP"


def test_aggregate_identity_failure_always_has_specific_error_text() -> None:
    error_text = _load_driver_function("_aggregate_failure_error")
    summary = {
        "systems_completed": 511,
        "unique_systems_completed": 511,
        "missing_systems": [37],
        "duplicate_systems": [],
        "unexpected_systems": [],
        "rank_audits": [{"error": None}],
    }

    message = error_text(summary, systems=512)

    assert message
    assert "missing campaign IDs: [37]" in message
    assert "completed 511 of 512 systems" in message


def test_aggregate_failure_has_nonempty_fallback_error_text() -> None:
    error_text = _load_driver_function("_aggregate_failure_error")
    summary = {
        "systems_completed": 512,
        "unique_systems_completed": 512,
        "missing_systems": [],
        "duplicate_systems": [],
        "unexpected_systems": [],
        "rank_audits": [{"error": None}],
    }

    assert error_text(summary, systems=512) == "campaign correctness audit failed"


def test_model_parity_audit_uses_runtime_sized_chunks() -> None:
    audit_outputs = _load_driver_function("_audit_stored_outputs_in_chunks")
    evaluated_batch_sizes: list[int] = []

    class FakeBatch:
        def __init__(self, graph_ids: list[int]) -> None:
            self.graph_ids = graph_ids
            self.num_graphs = len(graph_ids)
            self.energy = torch.tensor(graph_ids, dtype=torch.float64).reshape(-1, 1)
            self.forces = torch.zeros(len(graph_ids), 3, dtype=torch.float64)
            self.charges = torch.zeros(len(graph_ids), 1, dtype=torch.float64)
            self.batch_idx = torch.arange(len(graph_ids), dtype=torch.long)

        def index_select(self, selection: slice) -> "FakeBatch":
            return FakeBatch(self.graph_ids[selection])

        def to(self, device: torch.device) -> "FakeBatch":
            assert device.type == "cpu"
            return self

    class ExactModel:
        model_config = SimpleNamespace(neighbor_config=None)

        def __call__(self, batch: FakeBatch) -> dict[str, torch.Tensor]:
            evaluated_batch_sizes.append(batch.num_graphs)
            return {
                "energy": batch.energy,
                "forces": batch.forces,
                "charges": batch.charges,
            }

    audit_outputs.__globals__["_drop_neighbor_fields"] = lambda batch: None
    audit_outputs.__globals__["compute_neighbors"] = lambda batch, config: None

    result = audit_outputs(
        completed_cpu=FakeBatch(list(range(7))),
        model=ExactModel(),
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert evaluated_batch_sizes == [3, 3, 1]
    assert result["model_parity_batch_size"] == 3
    assert result["model_parity_batches"] == 3
    assert result["max_energy_difference_ev"] == 0.0
    assert result["max_force_difference_ev_per_a"] == 0.0
    assert result["max_charge_difference_e"] == 0.0
    assert result["max_abs_net_charge_e"] == 0.0


def test_campaign_uses_the_official_final_snapshot_hook() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    hook_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "nvalchemi.dynamics.hooks"
        for alias in node.names
    }
    fused_builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_fused_workflow"
    )
    distributed_builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_distributed_workflow"
    )
    fused_source = ast.unparse(fused_builder)
    distributed_source = ast.unparse(distributed_builder)
    downstream_fused = next(
        node
        for node in ast.walk(distributed_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedStage"
    )
    downstream_keywords = {keyword.arg for keyword in downstream_fused.keywords}

    assert "ConvergedSnapshotHook" in hook_imports
    assert "FinalStageCaptureHook" not in class_names
    assert "PendingInputSink" not in class_names
    assert "final_captured" not in driver.read_text(encoding="utf-8")
    assert "nve.register_hook(ConvergedSnapshotHook(sink=sink))" in fused_source
    assert "nve.register_hook(ConvergedSnapshotHook(sink=sink))" in distributed_source
    assert "sinks" not in downstream_keywords


def test_distributed_run_keeps_one_pipeline_for_the_full_partition() -> None:
    driver = REPOSITORY_ROOT / "scripts/benchmark_part1_distributed_campaign.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    distributed_builds = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_distributed_workflow"
    ]
    distributed_runs = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and ast.unparse(node.func.value) == "distributed_build.workflow"
    ]
    source = ast.unparse(main)

    assert len(distributed_builds) == 1
    assert len(distributed_runs) == 1
    assert "_partition_campaign_waves" not in source
    assert "for wave_index" not in source
    assert "del distributed_build" not in source
    assert "with distributed_build.workflow:" in source

    start = source.rfind(
        "start = perf_counter()",
        0,
        source.index("distributed_build = _build_distributed_workflow"),
    )
    build = source.index("distributed_build = _build_distributed_workflow")
    run = source.index("distributed_build.workflow.run()")
    stop = source.index("local_elapsed_s = perf_counter() - start", run)
    assert 0 <= start < build < run < stop
