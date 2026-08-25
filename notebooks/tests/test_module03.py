# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from nvalchemi.data import AtomicData, Batch

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))

from helpers import module03


def test_fractional_batch_multiplier_scales_to_whole_trajectories() -> None:
    half_study = module03._scaled_source_ids(205, 0.5)
    doubled_study = module03._scaled_source_ids(205, 2)

    assert len(half_study) == 103
    assert half_study == tuple(range(103))
    assert len(doubled_study) == 410
    assert doubled_study[:205] == doubled_study[205:]


@pytest.mark.parametrize("batch_multiplier", [0, -0.5, float("nan")])
def test_batch_multiplier_requires_a_positive_finite_number(
    batch_multiplier: float,
) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        module03._scaled_source_ids(205, batch_multiplier)


def _identity_batch(*simulation_ids: int, device: torch.device | str = "cpu") -> Batch:
    records = []
    for simulation_id in simulation_ids:
        record = AtomicData.from_atoms(Atoms("H"), device=device)
        record.add_system_property(
            "simulation_id",
            torch.tensor([[simulation_id]], dtype=torch.long, device=device),
        )
        record.add_system_property(
            "status", torch.tensor([[2]], dtype=torch.long, device=device)
        )
        records.append(record)
    return Batch.from_data_list(records, device=device)


def test_stage_hooks_receive_their_stage_local_rules() -> None:
    temperature = module03.temperature_window_hook(
        total_trajectories=2,
        device="cpu",
        frequency=10,
        window_samples=10,
        required_streak=2,
        target_temperature=300.0,
        temperature_tolerance=60.0,
        trend_tolerance=45.0,
    )
    energy = module03.energy_window_hook(
        total_trajectories=2,
        device="cpu",
        frequency=10,
        window_samples=10,
        required_streak=2,
        span_per_atom=0.00075,
    )

    assert temperature.frequency == energy.frequency == 10
    assert temperature.window_samples == energy.window_samples == 10
    assert temperature.required_streak == energy.required_streak == 2
    assert temperature.target == 300.0
    assert energy.span == 0.00075


def test_temperature_status_control_leaves_the_window_hook_in_control() -> None:
    control = module03.keep_temperature_hook_in_control()
    custom_op = control.criteria[0].custom_op

    assert custom_op is not None
    assert not custom_op(torch.ones(2, 1)).any()


def test_kinetic_energy_is_reduced_by_graph_on_the_batch_device() -> None:
    batch = _identity_batch(0, 1)
    batch["velocities"] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=batch.positions.dtype
    )

    kinetic = module03._kinetic_energy_per_graph(batch)
    masses = batch.atomic_masses.reshape(-1).to(dtype=batch.velocities.dtype)

    assert kinetic.device == batch.positions.device
    assert torch.allclose(kinetic, 0.5 * masses * torch.tensor([1.0, 4.0]))


def test_mixed_trace_rows_requires_a_natural_stage_overlap() -> None:
    snapshots = (
        module03._RouteSnapshot(4, 0, 0, 4),
        module03._RouteSnapshot(0, 4, 0, 4),
    )

    with pytest.raises(RuntimeError, match="natural mixed NVT/NVE state"):
        module03._mixed_trace_rows(snapshots)


def test_completion_identity_only_requires_simulation_id() -> None:
    batches = [_identity_batch(1), _identity_batch(0)]
    timer = SimpleNamespace(elapsed_s=1.0, peak_allocated_gib=0.0)

    result = module03._result_from_batches(
        "test route",
        batches,
        timer,
        expected_trajectories=2,
        device="cpu",
    )

    assert result.completed_systems == 2
    assert all(getattr(batch, "source_id", None) is None for batch in batches)
    assert all(getattr(batch, "replica_id", None) is None for batch in batches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sink requires CUDA")
def test_cuda_completion_sink_preserves_selected_system_fields() -> None:
    batch = _identity_batch(0, 1, device="cuda")
    completed = module03.CudaCompletionSink(capacity=2, device="cuda")

    completed.write(batch, mask=torch.tensor([False, True], device=batch.device))
    completed.write(batch, mask=torch.tensor([True, False], device=batch.device))
    restored = completed.read()

    assert restored.device.type == "cuda"
    assert restored.simulation_id.reshape(-1).tolist() == [1, 0]
    assert restored.status.reshape(-1).tolist() == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sink requires CUDA")
def test_cuda_completion_sink_discards_batch_local_neighbor_caches() -> None:
    first = _identity_batch(0, device="cuda")
    second = _identity_batch(1, device="cuda")
    first["neighbor_matrix"] = torch.zeros(
        (first.positions.shape[0], 48), dtype=torch.long, device="cuda"
    )
    second["neighbor_matrix"] = torch.zeros(
        (second.positions.shape[0], 32), dtype=torch.long, device="cuda"
    )
    completed = module03.CudaCompletionSink(capacity=2, device="cuda")

    completed.write(first)
    completed.write(second)
    restored = completed.read()

    assert restored.simulation_id.reshape(-1).tolist() == [0, 1]
    assert getattr(restored, "neighbor_matrix", None) is None


def test_inflight_check_reads_the_standard_sink_interface() -> None:
    completed_batch = _identity_batch(1, 0)
    completed = SimpleNamespace(read=lambda: completed_batch)
    timer = SimpleNamespace(elapsed_s=1.0, peak_allocated_gib=0.0)

    result = module03.check_inflight_run(
        completed,
        remaining=None,
        timer=timer,
        expected_trajectories=2,
        device="cpu",
        max_batch_size=1,
    )

    assert result.completed_systems == 2
    assert result.replacement_systems == 1


def test_inflight_check_allows_a_study_that_fits_the_initial_batch() -> None:
    completed_batch = _identity_batch(0, 1)
    completed = SimpleNamespace(read=lambda: completed_batch)
    timer = SimpleNamespace(elapsed_s=1.0, peak_allocated_gib=0.0)

    result = module03.check_inflight_run(
        completed,
        remaining=None,
        timer=timer,
        expected_trajectories=2,
        device="cpu",
        max_batch_size=64,
    )

    assert result.completed_systems == 2
    assert result.replacement_systems == 0
