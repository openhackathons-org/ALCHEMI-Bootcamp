"""CPU acceptance tests for the pinned BaseDynamics-owned hook lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat
import pytest
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    BaseDynamics,
    ConvergenceHook,
    DynamicsStage,
    HostMemory,
)
from nvalchemi.dynamics.hooks import FreezeAtomsHook, NaNDetectorHook, SnapshotHook
from nvalchemi.hooks import DynamicsContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig

NOTEBOOK = Path(__file__).resolve().parents[1] / "hooks.ipynb"


class ProbeModel(torch.nn.Module, BaseModelMixin):
    """Small deterministic model with the public Toolkit model contract."""

    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs={"energy", "forces"},
            active_outputs={"energy", "forces"},
        )
        self.nonfinite = nonfinite

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:
        return data

    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        energy = torch.zeros(data.num_graphs, 1, dtype=data.positions.dtype)
        node_energy = data.positions.square().sum(dim=1, keepdim=True)
        energy.scatter_add_(0, data.batch_idx.long().unsqueeze(1), node_energy)
        if self.nonfinite:
            energy[0, 0] = torch.nan
        return {"energy": energy, "forces": torch.ones_like(data.positions)}


class ProbeDynamics(BaseDynamics):
    """Minimal host whose update visibly moves every atom."""

    def pre_update(self, batch: Batch) -> None:
        batch.positions.add_(1.0)


class AllConverged:
    """Return every graph index from the convergence boundary."""

    def evaluate(self, batch: Batch) -> torch.Tensor:
        return torch.arange(batch.num_graphs)


class StageRecorder:
    """Record one configured stage through ordinary constructor registration."""

    frequency = 1

    def __init__(
        self,
        stage: DynamicsStage,
        events: list[DynamicsStage],
        contexts: dict[DynamicsStage, DynamicsContext],
    ) -> None:
        self.stage = stage
        self.events = events
        self.contexts = contexts

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        self.events.append(stage)
        self.contexts[stage] = ctx


def energy_history_type() -> type:
    """Load the learner-visible hook class from the notebook."""

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "class EnergyHistoryHook:" in cell.source
    )
    namespace = {
        "DynamicsContext": DynamicsContext,
        "DynamicsStage": DynamicsStage,
    }
    exec(source, namespace)  # noqa: S102 - execute the checked learner class
    return namespace["EnergyHistoryHook"]


def probe_batch() -> Batch:
    """Build one two-atom CPU graph with one frozen category."""

    record = AtomicData(
        atomic_numbers=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocities=torch.ones(2, 3),
        forces=torch.zeros(2, 3),
        energy=torch.zeros(1, 1),
        atom_categories=torch.tensor([-1, 0]),
    )
    record.add_system_property("status", torch.zeros(1, 1, dtype=torch.long))
    batch = Batch.from_data_list([record])
    batch.add_key("system_id", [torch.tensor(0)], level="system")
    return batch


def test_run_owns_registration_context_and_normal_enter_exit() -> None:
    hook = energy_history_type()(frequency=2)
    host = ProbeDynamics(model=ProbeModel(), hooks=[hook], n_steps=5)

    returned = host.run(probe_batch())

    assert returned.num_graphs == 1
    assert sorted({row["step"] for row in hook.rows}) == [0, 2, 4]
    assert hook.events[0] == {"event": "registered", "host": "ProbeDynamics"}
    assert hook.events[1] == {"event": "entered"}
    assert [event["step"] for event in hook.events if event["event"] == "called"] == [
        0,
        2,
        4,
    ]
    assert hook.events[-1] == {"event": "exited"}
    assert not hasattr(hook, "succeeded")
    assert hook.context == {
        "workflow": "ProbeDynamics",
        "batch": "Batch",
        "model": "ProbeModel",
        "step_count": 4,
        "global_rank": 0,
        "converged_mask": None,
    }
    assert hook.last_stage == "AFTER_COMPUTE"


def test_failed_run_closes_hook_without_claiming_success() -> None:
    hook = energy_history_type()(frequency=1)
    host = ProbeDynamics(
        model=ProbeModel(nonfinite=True),
        hooks=[hook, NaNDetectorHook(frequency=1)],
        n_steps=1,
    )

    with pytest.raises(RuntimeError, match="Non-finite values detected"):
        host.run(probe_batch())

    assert hook.events[0]["event"] == "registered"
    assert hook.events[1]["event"] == "entered"
    assert hook.events[-1] == {"event": "exited"}
    assert not hasattr(hook, "succeeded")


def test_step_callbacks_follow_the_nine_stage_sequence() -> None:
    events: list[DynamicsStage] = []
    contexts: dict[DynamicsStage, DynamicsContext] = {}
    hooks = [StageRecorder(stage, events, contexts) for stage in DynamicsStage]
    host = ProbeDynamics(
        model=ProbeModel(),
        hooks=hooks,
        convergence_hook=AllConverged(),
        n_steps=1,
    )

    host.run(probe_batch())

    assert events == list(DynamicsStage)
    assert contexts[DynamicsStage.AFTER_COMPUTE].converged_mask is None
    assert contexts[DynamicsStage.ON_CONVERGE].converged_mask.tolist() == [True]


def test_evaluator_stops_host_while_registered_hook_migrates_status() -> None:
    evaluator = ConvergenceHook.from_fmax(threshold=2.0, frequency=99)
    evaluator_host = ProbeDynamics(
        model=ProbeModel(),
        convergence_hook=evaluator,
        n_steps=3,
    )
    evaluator_batch = evaluator_host.run(probe_batch())

    migrator = ConvergenceHook.from_fmax(
        threshold=2.0,
        source_status=0,
        target_status=1,
        frequency=1,
    )
    registry_host = ProbeDynamics(
        model=ProbeModel(),
        hooks=[migrator],
        convergence_hook=None,
        n_steps=3,
    )
    registry_batch = registry_host.run(probe_batch())

    assert evaluator_host.step_count == 1
    assert evaluator_batch.status.tolist() == [[0]]
    assert registry_host.step_count == 3
    assert registry_batch.status.tolist() == [[1]]


def test_registered_freeze_hook_snapshots_then_restores() -> None:
    batch = probe_batch()
    initial_positions = batch.positions.clone()
    freeze = FreezeAtomsHook(frequency=1)
    host = ProbeDynamics(model=ProbeModel(), hooks=[freeze], n_steps=1)

    host.run(batch)

    torch.testing.assert_close(batch.positions[0], initial_positions[0])
    torch.testing.assert_close(batch.positions[1], initial_positions[1] + 1.0)
    torch.testing.assert_close(batch.velocities[0], torch.zeros(3))
    torch.testing.assert_close(batch.forces[0], torch.zeros(3))


def test_registered_snapshot_hook_obeys_zero_based_frequency() -> None:
    sink = HostMemory(capacity=3)
    snapshot = SnapshotHook(sink=sink, frequency=2)
    host = ProbeDynamics(model=ProbeModel(), hooks=[snapshot], n_steps=5)

    host.run(probe_batch())
    saved = sink.read()

    assert saved.num_graphs == 3
    assert host.step_count == 5
