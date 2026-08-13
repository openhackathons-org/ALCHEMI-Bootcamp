"""CPU behavioral checks for the public BaseDynamics/FIRE2 lifecycle."""

from __future__ import annotations

from typing import Any

import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE2,
    NVE,
    ConvergenceHook,
    FusedStage,
    NVTLangevin,
    initialize_velocities,
)
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class HarmonicModel(torch.nn.Module, BaseModelMixin):
    """Deterministic E=|r|²/2 model with public Toolkit metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs={"energy", "forces"},
            active_outputs={"energy", "forces"},
        )
        self.calls = 0

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:
        return data

    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        self.calls += 1
        node_energy = 0.5 * data.positions.square().sum(dim=1, keepdim=True)
        energy = torch.zeros(
            data.num_graphs,
            1,
            dtype=data.positions.dtype,
            device=data.device,
        )
        energy.scatter_add_(0, data.batch_idx.long().unsqueeze(1), node_energy)
        return {"energy": energy, "forces": -data.positions}


def _record(x: float) -> AtomicData:
    record = AtomicData(
        atomic_numbers=torch.tensor([1]),
        atomic_masses=torch.tensor([1.0]),
        positions=torch.tensor([[x, 0.0, 0.0]]),
        velocities=torch.zeros(1, 3),
        forces=torch.zeros(1, 3),
        energy=torch.zeros(1, 1),
    )
    record.add_system_property("status", torch.zeros(1, 1, dtype=torch.long))
    return record


def _batch(*x: float) -> Batch:
    return Batch.from_data_list([_record(value) for value in x])


def test_fire2_step_updates_owned_batch_and_coherent_outputs() -> None:
    model = HarmonicModel()
    batch = _batch(1.0)
    dynamics = FIRE2(model=model, dt=0.01, maxstep=0.04)
    positions = batch.positions
    energy = batch.energy
    forces = batch.forces

    returned, _ = dynamics.step(batch)
    returned, _ = dynamics.step(batch)

    assert returned is batch
    assert batch.positions is positions
    assert batch.energy is energy
    assert batch.forces is forces
    assert dynamics.step_count == 2
    assert model.calls == 2
    assert batch.positions[0, 0] < 1.0
    torch.testing.assert_close(batch.forces, -batch.positions)
    torch.testing.assert_close(
        batch.energy,
        0.5 * batch.positions.square().sum().reshape(1, 1),
    )


def test_refresh_repairs_outputs_after_converged_position_restore() -> None:
    model = HarmonicModel()
    batch = _batch(0.05, 1.0)
    status_hook = ConvergenceHook.from_fmax(
        0.1, source_status=0, target_status=1
    )
    dynamics = FIRE2(
        model=model,
        dt=0.01,
        maxstep=0.04,
        n_steps=3,
        hooks=[status_hook],
        convergence_hook=ConvergenceHook.from_fmax(0.1),
    )

    returned = dynamics.run(batch)

    assert returned is batch
    assert batch.status.flatten().tolist() == [1, 0]
    torch.testing.assert_close(batch.positions[0, 0], torch.tensor(0.05))
    assert not torch.equal(batch.forces, -batch.positions)

    energy_storage = batch.energy
    force_storage = batch.forces
    outputs = dynamics.compute(batch)

    assert batch.energy is energy_storage
    assert batch.forces is force_storage
    torch.testing.assert_close(batch.energy, outputs["energy"])
    torch.testing.assert_close(batch.forces, outputs["forces"])
    torch.testing.assert_close(batch.forces, -batch.positions)


def test_status_migration_drives_all_system_early_stop() -> None:
    model = HarmonicModel()
    batch = _batch(0.05, 0.08)
    status_hook = ConvergenceHook.from_fmax(
        0.1, source_status=0, target_status=1
    )
    detector = ConvergenceHook.from_fmax(0.1, frequency=999)
    dynamics = FIRE2(
        model=model,
        dt=0.01,
        n_steps=10,
        hooks=[status_hook],
        convergence_hook=detector,
    )

    assert status_hook in dynamics.hooks
    assert detector not in dynamics.hooks
    assert dynamics.convergence_hook is detector

    returned = dynamics.run(batch)

    assert returned is batch
    # Host detection calls evaluate(batch) directly, so registry frequency is irrelevant.
    assert dynamics.step_count == 1
    assert model.calls == 1
    assert batch.status.flatten().tolist() == [1, 1]


def test_public_velocity_initialization_is_seeded_mass_aware_and_com_free() -> None:
    def make_batch() -> Batch:
        records = []
        for offset in (0.2, 0.4):
            record = AtomicData(
                atomic_numbers=torch.tensor([1, 1]),
                atomic_masses=torch.tensor([1.0, 1.0]),
                positions=torch.tensor(
                    [[offset, 0.0, 0.0], [offset + 0.8, 0.0, 0.0]]
                ),
                velocities=torch.zeros(2, 3),
                forces=torch.zeros(2, 3),
                energy=torch.zeros(1, 1),
            )
            records.append(record)
        return Batch.from_data_list(records)

    first = make_batch()
    second = make_batch()
    temperature = torch.tensor([50.0, 100.0])

    initialize_velocities(
        first.velocities,
        first.atomic_masses,
        temperature,
        first.batch_idx,
        random_seed=17,
        remove_com=True,
        rescale=True,
    )
    initialize_velocities(
        second.velocities,
        second.atomic_masses,
        temperature,
        second.batch_idx,
        random_seed=17,
        remove_com=True,
        rescale=True,
    )

    torch.testing.assert_close(first.velocities, second.velocities)
    assert torch.isfinite(first.velocities).all()
    assert bool(first.velocities.square().sum().gt(0))
    for graph in range(first.num_graphs):
        selected = first.batch_idx == graph
        momentum = (
            first.atomic_masses[selected, None] * first.velocities[selected]
        ).sum(dim=0)
        torch.testing.assert_close(momentum, torch.zeros(3), atol=1e-6, rtol=0)


def test_fused_stage_uses_first_stage_model_and_status_codes() -> None:
    model = HarmonicModel()
    nvt = NVTLangevin(
        model=model,
        dt=0.5,
        temperature=50.0,
        friction=0.1,
        random_seed=7,
        n_steps=5,
    )
    nve = NVE(model=model, dt=0.5, n_steps=5)

    fused = nvt + nve

    assert isinstance(fused, FusedStage)
    assert fused.model is model
    assert fused.entry_status == 0
    assert fused.exit_status == 2
    assert [status for status, _ in fused.sub_stages] == [0, 1]
    assert [type(stage).__name__ for _, stage in fused.sub_stages] == [
        "NVTLangevin",
        "NVE",
    ]
