"""Runtime regression for exact stage budgets in a real CPU ``FusedStage``."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(
    os.environ.get(
        "ALCHEMI_REPOSITORY_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))
TOOLKIT_ROOT = os.environ.get("ALCHEMI_TOOLKIT_ROOT")
if TOOLKIT_ROOT:
    sys.path.insert(0, str(Path(TOOLKIT_ROOT).expanduser().resolve()))

torch = pytest.importorskip(
    "torch",
    reason="the fused-stage runtime regression requires PyTorch",
)
pytest.importorskip(
    "nvalchemi",
    reason="the fused-stage runtime regression requires ALCHEMI Toolkit",
)

from nvalchemi.data import AtomicData, Batch, InMemoryDataset  # noqa: E402
from nvalchemi.dynamics import (  # noqa: E402
    BaseDynamics,
    ConvergenceHook,
    DynamicsStage,
    FusedStage,
    HostMemory,
    SizeAwareSampler,
)
import nvalchemi.hooks as toolkit_hooks  # noqa: E402
from nvalchemi.models.demo import DemoModel, DemoModelWrapper  # noqa: E402
from aux.hooks import (  # noqa: E402
    StageStepCounterHook,
    converge_after_steps,
)
from aux.capture import PredictedChargeIRHook  # noqa: E402
from aux.workflow_config import (  # noqa: E402
    IR_CAPTURE_CHARGE_TOLERANCE_E,
)


# This regression targets the Toolkit Core 0.2 release-candidate control flow
# used by the campaign. Older local wheels export ``HookContext`` and order the
# fused hooks differently, so they would test a different implementation.
DynamicsContext = getattr(toolkit_hooks, "DynamicsContext", None)
if DynamicsContext is None:
    pytest.skip(
        "requires the pinned Toolkit Core 0.2 release candidate",
        allow_module_level=True,
    )


def _graph(system_id: int, countdown: float) -> AtomicData:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        dtype=torch.float32,
    )
    graph = AtomicData(
        positions=positions,
        forces=torch.zeros_like(positions),
        velocities=torch.zeros_like(positions),
        atomic_masses=torch.ones(2, dtype=torch.float32),
        atomic_numbers=torch.tensor([1, 1], dtype=torch.int64),
        energy=torch.zeros(1, 1, dtype=torch.float32),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    graph.add_system_property(
        "system_id",
        torch.tensor([[system_id]], dtype=torch.int64),
    )
    graph.add_system_property(
        "convergence_countdown",
        torch.tensor([[countdown]], dtype=torch.float32),
    )
    graph.add_system_property(
        "nvt_steps_done",
        torch.zeros(1, 1, dtype=torch.int64),
    )
    graph.add_system_property(
        "nve_steps_done",
        torch.zeros(1, 1, dtype=torch.int64),
    )
    graph.add_system_property(
        "final_captured",
        torch.zeros(1, 1, dtype=torch.int64),
    )
    return graph


class _AdvanceConvergence:
    """Make two systems leave the first stage on different fused steps."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, context: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        active = context.batch.status.reshape(-1) == 0
        context.batch.convergence_countdown.reshape(-1)[active] -= 1.0


class _CaptureCompletedNVE:
    """Observe final capture at the same AFTER_STEP boundary as the campaign."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self) -> None:
        self.system_ids: list[int] = []

    def __call__(self, context: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        batch = context.batch
        ready = (
            (batch.status.reshape(-1) == 3)
            & (batch.nve_steps_done.reshape(-1) == 2)
            & (batch.final_captured.reshape(-1) == 0)
        )
        batch.final_captured.reshape(-1)[ready] = 1
        for system_id in batch.system_id.reshape(-1)[ready].tolist():
            value = int(system_id)
            if value in self.system_ids:
                raise AssertionError(f"system {value} was captured more than once")
            self.system_ids.append(value)


def _snapshot(batch: Batch) -> dict[str, list[int]]:
    return {
        "status": [int(value) for value in batch.status.reshape(-1).tolist()],
        "nvt": [int(value) for value in batch.nvt_steps_done.reshape(-1).tolist()],
        "nve": [int(value) for value in batch.nve_steps_done.reshape(-1).tolist()],
    }


def _run_counter_probe() -> tuple[list[dict[str, list[int]]], list[int]]:
    model = DemoModelWrapper(DemoModel()).to("cpu")
    convergence = ConvergenceHook(
        criteria={"key": "convergence_countdown", "threshold": 0.0}
    )
    first_stage = BaseDynamics(
        model=model,
        convergence_hook=convergence,
        device_type="cpu",
    )
    first_stage.register_hook(_AdvanceConvergence())
    nvt = BaseDynamics(
        model=model,
        convergence_hook=converge_after_steps("nvt_steps_done", 2),
        device_type="cpu",
    )
    nve = BaseDynamics(
        model=model,
        convergence_hook=converge_after_steps("nve_steps_done", 2),
        device_type="cpu",
    )
    fused = FusedStage(
        sub_stages=[(0, first_stage), (1, nvt), (2, nve)],
        device_type="cpu",
    )

    counter = StageStepCounterHook(
        {1: "nvt_steps_done", 2: "nve_steps_done"}
    )
    capture = _CaptureCompletedNVE()
    fused.register_fused_hook(counter)
    fused.register_fused_hook(capture)

    batch = Batch.from_data_list(
        [_graph(701, 1.0), _graph(702, 2.0)],
        device="cpu",
    )
    batch.status = torch.zeros(batch.num_graphs, 1, dtype=torch.int64)

    history: list[dict[str, list[int]]] = []
    for _ in range(8):
        if fused.all_complete(batch, fused.exit_status):
            break
        batch, _ = fused.step(batch)
        history.append(_snapshot(batch))

    assert fused.all_complete(batch, fused.exit_status)
    return history, capture.system_ids


def test_counter_tracks_real_updates_across_convergence_and_fixed_stages() -> None:
    """A convergence step must not consume the following stage's budget."""

    history, captured = _run_counter_probe()

    assert history == [
        {"status": [1, 0], "nvt": [0, 0], "nve": [0, 0]},
        {"status": [1, 1], "nvt": [1, 0], "nve": [0, 0]},
        {"status": [2, 1], "nvt": [2, 1], "nve": [0, 0]},
        {"status": [2, 2], "nvt": [2, 2], "nve": [1, 0]},
        {"status": [3, 2], "nvt": [2, 2], "nve": [2, 1]},
        {"status": [3, 3], "nvt": [2, 2], "nve": [2, 2]},
    ]
    assert captured == [701, 702]


def test_exact_stage_budgets_end_run_without_global_cutoff() -> None:
    """Per-system convergence counters must be sufficient to drain the run."""

    model = DemoModelWrapper(DemoModel()).to("cpu")
    nvt = BaseDynamics(
        model=model,
        convergence_hook=converge_after_steps("nvt_steps_done", 2),
        device_type="cpu",
    )
    nve = BaseDynamics(
        model=model,
        convergence_hook=converge_after_steps("nve_steps_done", 3),
        device_type="cpu",
    )
    fused = FusedStage(
        sub_stages=[(0, nvt), (1, nve)],
        device_type="cpu",
    )
    fused.register_fused_hook(
        StageStepCounterHook({0: "nvt_steps_done", 1: "nve_steps_done"})
    )
    batch = Batch.from_data_list(
        [_graph(801, 0.0), _graph(802, 0.0)],
        device="cpu",
    )
    batch.status = torch.zeros(batch.num_graphs, 1, dtype=torch.int64)

    result = fused.run(batch)

    assert result is not None
    assert fused.step_count == 5
    assert bool((result.status == fused.exit_status).all())
    assert bool((result.nvt_steps_done == 2).all())
    assert bool((result.nve_steps_done == 3).all())


def test_inflight_queue_preserves_exact_stage_budgets() -> None:
    """Every replacement wave must receive the full per-stage workload."""

    systems = 16
    active_systems = 4
    source = Batch.from_data_list(
        [_graph(900 + index, 0.0) for index in range(systems)],
        device="cpu",
    )
    dataset = InMemoryDataset(in_memory_batch=source, device="cpu")
    try:
        sampler = SizeAwareSampler(
            dataset,
            max_atoms=active_systems * 2,
            max_edges=None,
            max_batch_size=active_systems,
            shuffle=False,
        )
        sink = HostMemory(capacity=systems)
        model = DemoModelWrapper(DemoModel()).to("cpu")
        nvt = BaseDynamics(
            model=model,
            convergence_hook=converge_after_steps("nvt_steps_done", 2),
            device_type="cpu",
        )
        nve = BaseDynamics(
            model=model,
            convergence_hook=converge_after_steps("nve_steps_done", 3),
            device_type="cpu",
        )
        fused = FusedStage(
            sub_stages=[(0, nvt), (1, nve)],
            sampler=sampler,
            sinks=[sink],
            refill_frequency=1,
            device_type="cpu",
        )
        fused.register_fused_hook(
            StageStepCounterHook({0: "nvt_steps_done", 1: "nve_steps_done"})
        )

        result = fused.run(batch=None)
        completed = sink.drain()

        assert result is None
        assert fused.done
        assert sampler.exhausted
        assert fused.step_count == (systems // active_systems) * (2 + 3)
        assert completed.num_graphs == systems
        assert bool((completed.status == fused.exit_status).all())
        assert bool((completed.nvt_steps_done == 2).all())
        assert bool((completed.nve_steps_done == 3).all())
        assert torch.equal(
            torch.sort(completed.system_id.reshape(-1)).values,
            torch.arange(systems),
        )
    finally:
        dataset.close()


def test_ir_capture_uses_step_entry_counters_across_status_migration() -> None:
    """The final update in each stage remains assigned to that stage."""

    graph = _graph(1001, 0.0)
    graph.add_node_property(
        "charges",
        torch.tensor([0.1, -0.1], dtype=torch.float32),
    )
    batch = Batch.from_data_list([graph], device="cpu")
    warmup_status = 7
    production_status = 8
    batch.status = torch.full((1, 1), warmup_status, dtype=torch.int64)
    capture = PredictedChargeIRHook(
        warmup_steps=2,
        n_steps=3,
        dt_fs=0.5,
        warmup_status=warmup_status,
        production_status=production_status,
        charge_tolerance=IR_CAPTURE_CHARGE_TOLERANCE_E,
        compile_reducer=False,
    )
    route = (
        (1, 0, warmup_status),
        (2, 0, production_status),  # NVT migrated after this update.
        (2, 1, production_status),
        (2, 2, production_status),
        (2, 3, production_status + 1),  # NVE migrated after this update.
    )
    for step_count, (nvt_steps, nve_steps, status) in enumerate(route):
        batch.nvt_steps_done.fill_(nvt_steps)
        batch.nve_steps_done.fill_(nve_steps)
        batch.status.fill_(status)
        capture(
            DynamicsContext(batch=batch, step_count=step_count),
            DynamicsStage.AFTER_STEP,
        )

    trajectory = capture.result()

    assert capture.stage_counts == {
        f"status_{warmup_status}_warmup_steps": 2,
        f"status_{production_status}_production_steps": 3,
    }
    assert trajectory.dipoles_e_angstrom.shape == (3, 1, 3)
    assert trajectory.positions_angstrom.shape == (3, 2, 3)


if __name__ == "__main__":
    test_counter_tracks_real_updates_across_convergence_and_fixed_stages()
    test_exact_stage_budgets_end_run_without_global_cutoff()
    test_inflight_queue_preserves_exact_stage_budgets()
    test_ir_capture_uses_step_entry_counters_across_status_migration()
    print("PASS: fused stage counters follow step-entry status")
