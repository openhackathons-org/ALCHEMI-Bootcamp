"""Deferred CUDA validation for the exact AIMNet2-neighbor-FIRE2 workflow."""

from __future__ import annotations

import os

import helpers
import pytest
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2, ConvergenceHook
from nvalchemi.models import AIMNet2Wrapper
from nvalchemi.neighbors import compute_neighbors

LABELS = (
    "Ethyne",
    "Acetonitrile",
    "Methanol",
    "Acetaldehyde",
    "Acetamide",
    "Pyridine",
    "Phenol",
    "2,3-dimethylbutane",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_N05_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set RUN_N05_CUDA=1 on a free CUDA worker",
)


def _cuda_batch() -> tuple[Batch, list[int]]:
    molecules, metadata = helpers.load_molecule_selection(LABELS)
    records = [
        AtomicData.from_atoms(
            molecule,
            device="cuda",
            dtype=torch.float32,
        )
        for molecule in molecules
    ]
    for record in records:
        record.use_default_velocities()
        record.add_node_property("forces", torch.zeros_like(record.positions))
        record.add_system_property(
            "energy",
            torch.zeros(1, 1, device="cuda"),
        )
        record.add_system_property(
            "status",
            torch.zeros(1, 1, device="cuda", dtype=torch.long),
        )
    return Batch.from_data_list(records, device="cuda"), metadata["atoms"].tolist()


def _cuda_model() -> AIMNet2Wrapper:
    model = AIMNet2Wrapper.from_checkpoint(
        helpers.model_checkpoint(),
        device="cuda",
        compile_model=False,
    ).eval()
    helpers.freeze_model(model)
    model.set_config("active_outputs", {"energy", "forces"})
    return model


def test_exact_aimnet_fire2_outputs_match_restored_positions() -> None:
    batch, atom_counts = _cuda_batch()
    model = _cuda_model()
    monitor = helpers.RelaxationMonitor(LABELS, total_steps=120)
    status_hook = ConvergenceHook.from_fmax(
        0.15,
        source_status=0,
        target_status=1,
    )
    dynamics = FIRE2(
        model=model,
        dt=0.01,
        maxstep=0.04,
        n_steps=120,
        hooks=[*model.make_neighbor_hooks(), status_hook, monitor],
        convergence_hook=ConvergenceHook.from_fmax(0.15),
    )
    starting_batch = batch.clone()

    result = dynamics.run(batch)

    assert result is batch
    assert 1 <= dynamics.step_count <= 120
    assert torch.isfinite(batch.positions).all()
    assert set(batch.status.flatten().tolist()) <= {0, 1}
    assert bool(batch.status.ge(1).any())

    final_positions = batch.positions.detach().clone()
    final_status = batch.status.detach().clone()
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    outputs = dynamics.compute(batch)

    assert torch.equal(batch.positions, final_positions)
    assert torch.equal(batch.status, final_status)
    assert torch.isfinite(batch.energy).all()
    assert torch.isfinite(batch.forces).all()
    torch.testing.assert_close(batch.energy, outputs["energy"])
    torch.testing.assert_close(batch.forces, outputs["forces"])
    direct_outputs = model(batch)
    torch.testing.assert_close(batch.energy, direct_outputs["energy"])
    torch.testing.assert_close(batch.forces, direct_outputs["forces"])

    history = helpers.truncate_history_at_convergence(
        monitor.history_frame(),
        final_batch=batch,
    )
    for graph, rows in history.groupby("graph", sort=False):
        if int(batch.status[graph].item()) >= 1:
            assert int(rows.iloc[-1]["status"]) == 1
            assert rows["status"].ge(1).sum() == 1

    selected = torch.where(batch.status.squeeze(-1) >= 1)[0]
    selected_batch = batch.index_select(selected)
    recovered = batch.to_data_list()
    assert selected_batch.num_graphs == selected.numel()
    assert [record.num_nodes for record in recovered] == atom_counts
    assert all(torch.isfinite(record.energy).all() for record in recovered)
    assert all(torch.isfinite(record.forces).all() for record in recovered)

    one_step = FIRE2(
        model=model,
        dt=0.01,
        maxstep=0.04,
        hooks=model.make_neighbor_hooks(),
        convergence_hook=ConvergenceHook.from_fmax(0.15),
    )
    one_step_batch = starting_batch
    one_step_batch, _ = one_step.step(one_step_batch)
    before = one_step_batch.positions.detach().clone()
    one_step_batch, _ = one_step.step(one_step_batch)
    displacement = torch.linalg.vector_norm(
        one_step_batch.positions - before,
        dim=1,
    )
    assert torch.isfinite(displacement).all()
    assert bool(displacement.gt(0).any())
    one_step_positions = one_step_batch.positions.detach().clone()
    compute_neighbors(
        one_step_batch,
        config=model.model_config.neighbor_config,
    )
    one_step_outputs = one_step.compute(one_step_batch)
    assert torch.equal(one_step_batch.positions, one_step_positions)
    torch.testing.assert_close(one_step_batch.energy, one_step_outputs["energy"])
    torch.testing.assert_close(one_step_batch.forces, one_step_outputs["forces"])
