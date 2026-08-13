"""Behavior checks for the public Part 06 pipeline path."""

from __future__ import annotations

import helpers
import pytest
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import (
    DemoDynamics,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
    HostMemory,
    SizeAwareSampler,
)
from nvalchemi.dynamics.base import BufferConfig
from nvalchemi.hooks import StageTimingHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _demo_stage(model: DemoModelWrapper) -> DemoDynamics:
    return DemoDynamics(
        model=model,
        n_steps=1,
        dt=0.01,
        convergence_hook=None,
        device_type="cpu",
    )


class _ActiveStatusRecorder:
    """Record how many systems are active for one public status code."""

    stage = DynamicsStage.BEFORE_PRE_UPDATE
    frequency = 1

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.active_counts: list[int] = []

    def __call__(self, context: object, stage: DynamicsStage | None = None) -> None:
        observed_stage = stage if stage is not None else getattr(context, "stage", None)
        if observed_stage is not self.stage:
            return
        batch = context.batch  # type: ignore[attr-defined]
        status = batch.status.detach().reshape(-1)
        self.active_counts.append(int(status.eq(self.status_code).sum().cpu()))


def test_fixed_shape_compiled_stage_matches_eager_result() -> None:
    graphs = helpers.make_synthetic_graphs((3, 4), seed=11)
    batch = Batch.from_data_list(graphs)

    def stage(positions: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        centered = positions - positions.mean(dim=0, keepdim=True)
        return centered.square().sum(dim=1) + ptr[-1].to(positions.dtype)

    compiled_stage = torch.compile(stage, backend="eager", fullgraph=True)
    eager = stage(batch.positions, batch.batch_ptr)
    compiled = compiled_stage(batch.positions, batch.batch_ptr)

    torch.testing.assert_close(compiled, eager)
    assert compiled.device == batch.positions.device
    assert compiled.dtype == batch.positions.dtype


def test_sampler_respects_batch_and_atom_limits_and_stamps_ids() -> None:
    graphs = helpers.make_synthetic_graphs((3, 5, 4, 6), seed=9)
    dataset = helpers.AtomicDataset(graphs, device=torch.device("cpu"))
    sampler = SizeAwareSampler(
        dataset,
        max_atoms=9,
        max_edges=None,
        max_batch_size=2,
    )

    batch = sampler.build_initial_batch()

    assert batch.num_graphs <= 2
    assert batch.num_nodes <= 9
    assert batch.system_id.dtype == torch.int64
    assert batch.system_id.unique().numel() == batch.num_graphs
    assert batch.device.type == "cpu"


def test_inflight_fused_stage_preserves_all_systems_and_stage_timing() -> None:
    graphs = helpers.make_synthetic_graphs((3, 4, 3, 4), seed=21)
    dataset = helpers.AtomicDataset(graphs, device=torch.device("cpu"))
    model = DemoModelWrapper(DemoModel(hidden_dim=8)).eval()
    sink = HostMemory(capacity=len(graphs))
    sampler = SizeAwareSampler(
        dataset,
        max_atoms=8,
        max_edges=None,
        max_batch_size=2,
    )
    timer = StageTimingHook("step", enable_nvtx=False)
    monitor = helpers.PipelineMonitor(device=torch.device("cpu"))
    fused = FusedStage(
        sub_stages=[(0, _demo_stage(model)), (1, _demo_stage(model))],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=1,
        device_type="cpu",
    )
    fused.register_fused_hook(timer)
    fused.register_fused_hook(monitor)

    result = fused.run(batch=None, n_steps=20)
    collected = sink.read()

    assert result is None
    assert collected.num_graphs == len(graphs)
    assert sorted(collected.system_id.flatten().tolist()) == list(range(len(graphs)))
    assert sorted(collected.source_index.flatten().tolist()) == list(range(len(graphs)))
    assert timer.summary()["BEFORE_STEP->AFTER_STEP"]["n_samples"] >= 2
    assert not monitor.frame().empty


def test_downstream_fixed_step_budget_runs_before_graduation() -> None:
    graphs = helpers.make_synthetic_graphs((3, 4), seed=23)
    model = DemoModelWrapper(DemoModel(hidden_dim=8)).eval()
    first = _demo_stage(model)
    second = DemoDynamics(
        model=model,
        n_steps=2,
        dt=0.01,
        convergence_hook=None,
        device_type="cpu",
    )
    recorder = _ActiveStatusRecorder(status_code=1)
    second.register_hook(recorder)
    fused = FusedStage(
        sub_stages=[(0, first), (1, second)],
        device_type="cpu",
    )

    result = fused.run(Batch.from_data_list(graphs), n_steps=4)

    assert result is not None
    assert result.status.flatten().tolist() == [2, 2]
    assert any(count == 2 for count in recorder.active_counts)


def test_distributed_pipeline_plan_is_constructible_without_launching_ranks() -> None:
    config = BufferConfig(num_systems=2, num_nodes=16, num_edges=0)
    model = DemoModelWrapper(DemoModel(hidden_dim=8)).eval()
    first_inner = DemoDynamics(
        model=model,
        n_steps=1,
        device_type="cpu",
    )
    second_inner = DemoDynamics(
        model=model,
        n_steps=1,
        device_type="cpu",
    )
    first = FusedStage(
        sub_stages=[(0, first_inner)],
        device_type="cpu",
        buffer_config=config,
        comm_mode="async_recv",
    )
    second = FusedStage(
        sub_stages=[(0, second_inner)],
        device_type="cpu",
        buffer_config=config,
        comm_mode="async_recv",
    )

    pipeline = DistributedPipeline(stages={0: first, 1: second}, synchronized=False)

    assert list(pipeline.stages) == [0, 1]
    assert all(isinstance(stage, FusedStage) for stage in pipeline.stages.values())
    assert pipeline.stages[0].buffer_config == config
    assert pipeline.stages[1].comm_mode == "async_recv"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gpu_buffer_probe_exposes_integer_field_limitation_at_this_pin() -> None:
    report = helpers.probe_gpu_buffer_mixed_dtype(torch.device("cuda"))

    assert report["probe ran"] is True
    assert report["float positions preserved"] is True
    assert report["integer atomic numbers preserved"] is False
    assert report["integer source indices preserved"] is False
