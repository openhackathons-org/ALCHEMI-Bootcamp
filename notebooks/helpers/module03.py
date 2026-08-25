# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Module 03, scale a study."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from numbers import Real
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from uuid import uuid4

import pandas as pd
import torch
from IPython.display import HTML
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    NVE,
    ConvergenceHook,
    DataSink,
    DynamicsStage,
    initialize_velocities,
)
from nvalchemi.hooks import DynamicsContext
from nvalchemi.models import (
    AIMNet2Wrapper,
    DFTD3ModelWrapper,
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
)

from .common import _ignore_known_aimnet2_cutoff_warning
from .model_support import (
    DirectCoulombAdapter,
    d3_parameter_file,
    model_checkpoint,
)
from .ncia250 import load_ncia250
from .viewers import runtime_required_notice

__all__ = [
    "CudaCompletionSink",
    "check_inflight_run",
    "compare_routes",
    "energy_window_hook",
    "group_indices_by_capacity",
    "keep_temperature_hook_in_control",
    "prepare_ncia250_trajectories",
    "show_scheduling_widget",
    "stage_trace_hook",
    "summarize_fused",
    "summarize_sequential",
    "summarize_stage_trace",
    "temperature_window_hook",
    "time_route",
]

_SCHEDULING_WIDGET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "03-campaign-scheduling-widget-v1.html"
)
_NEIGHBOR_FIELDS = (
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "neighbor_list",
    "neighbor_list_shifts",
    "shifts",
    "num_neighbors",
)
_KB_EV_PER_K = 8.617333262e-5
_WIDGET_TOTAL_TRAJECTORIES = 128
_WIDGET_MAX_BATCH_SIZE = 64
_WIDGET_VISIBLE_SLOTS = 16
_PREPARATION_MAX_BATCH_SIZE = 64


def _positive_finite_number(value: Real, *, name: str) -> float:
    """Return one validated positive finite real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return resolved


def _whole_system_count(value: Real, *, name: str) -> int:
    """Round a positive scale result to the nearest whole system."""

    resolved = _positive_finite_number(value, name=name)
    return max(1, math.floor(resolved + 0.5))


def _scaled_source_ids(source_systems: int, batch_multiplier: Real) -> tuple[int, ...]:
    """Repeat source IDs and trim them to the requested scaled study size."""

    if isinstance(source_systems, bool) or not isinstance(source_systems, int):
        raise TypeError("source_systems must be a positive integer")
    if source_systems < 1:
        raise ValueError("source_systems must be a positive integer")
    multiplier = _positive_finite_number(batch_multiplier, name="batch_multiplier")
    target_systems = _whole_system_count(
        source_systems * multiplier, name="scaled study size"
    )
    return tuple(index % source_systems for index in range(target_systems))


def show_scheduling_widget() -> HTML:
    """Embed the scheduling illustration with a preset schematic workload."""

    source = _SCHEDULING_WIDGET.read_text(encoding="utf-8")
    replacements = {
        "__ALCHEMI_TOTAL_SYSTEMS__": str(_WIDGET_TOTAL_TRAJECTORIES),
        "__ALCHEMI_ACTIVE_SYSTEM_LIMIT__": str(_WIDGET_MAX_BATCH_SIZE),
        "__ALCHEMI_VISIBLE_SLOTS__": str(_WIDGET_VISIBLE_SLOTS),
    }
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    root_id = f"alchemi-study-scheduling-{uuid4().hex}"
    return HTML(
        f'<div id="{root_id}" style="width:100%;container-type:inline-size">'
        f"<style>#{root_id}.is-live [data-runtime-note]{{display:none}}</style>"
        + runtime_required_notice(margin="0")
        + '<iframe title="Study scheduling routes" srcdoc="'
        + escape(source, quote=True)
        + '" style="display:block;width:100%;height:470px;'
        + 'height:clamp(330px,calc(42.6cqw + 58px),528px);border:0"></iframe></div>'
        + f'<script>(function(){{const root=document.getElementById("{root_id}");'
        + 'if(root&&root.querySelector("iframe"))root.classList.add("is-live");'
        + "})();</script>"
    )


@dataclass(frozen=True, slots=True)
class _RouteSnapshot:
    """One active-batch observation collected outside route timing."""

    nvt_systems: int
    nve_systems: int
    completed_systems: int
    active_systems: int


class _RouteTraceHook:
    """Record full-batch stage counts during a fused run."""

    stage = DynamicsStage.AFTER_STEP

    def __init__(self, *, frequency: int = 10) -> None:
        self.frequency = frequency
        self._snapshots: list[tuple[torch.Tensor, int]] = []

    def __call__(self, context: Any, stage: Any) -> None:
        del stage
        status = context.batch.status.reshape(-1).detach().clone()
        self._snapshots.append((status, context.batch.num_graphs))

    @property
    def snapshots(self) -> tuple[_RouteSnapshot, ...]:
        rows: list[_RouteSnapshot] = []
        for status, active_systems in self._snapshots:
            rows.append(
                _RouteSnapshot(
                    nvt_systems=int((status == 0).sum().item()),
                    nve_systems=int((status == 1).sum().item()),
                    completed_systems=int((status >= 2).sum().item()),
                    active_systems=active_systems,
                )
            )
        return tuple(rows)


class RouteTimer:
    """Measure one CUDA-synchronized scheduling route."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.elapsed_s = 0.0
        self.peak_allocated_gib = 0.0
        self._started_at = 0.0

    def __enter__(self) -> Self:
        _synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        _synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self._started_at
        if self.device.type == "cuda":
            self.peak_allocated_gib = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Checked completion and synchronized timing for one route."""

    route: str
    completed_systems: int
    elapsed_s: float
    peak_allocated_gib: float
    replacement_systems: int

    @property
    def summary(self) -> dict[str, int | float | str]:
        """Return the compact result shown after each route."""

        return {
            "route": self.route,
            "completed trajectories": self.completed_systems,
            "elapsed [s]": round(self.elapsed_s, 3),
            "throughput [trajectories/s]": round(
                self.completed_systems / self.elapsed_s, 2
            ),
            "peak allocated [GiB]": round(self.peak_allocated_gib, 3),
            "replacement trajectories": self.replacement_systems,
        }


class CudaCompletionSink(DataSink):
    """Retain completed trajectory state on one CUDA device."""

    def __init__(self, capacity: int, device: torch.device | str) -> None:
        self._capacity = capacity
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("CudaCompletionSink requires a CUDA device")
        self._batches: list[Batch] = []
        self._count = 0

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """Select graduated systems and retain their reusable CUDA state."""

        selected = batch
        if mask is not None:
            selected_indices = torch.where(
                mask.to(device=batch.device, dtype=torch.bool)
            )[0]
            if selected_indices.numel() == 0:
                return
            selected = batch.index_select(selected_indices)
        else:
            selected = batch.clone()
        # Neighbor caches are local to one padded active Batch. The model
        # rebuilds them when a completed trajectory enters another workflow.
        _clear_neighbor_fields(selected)
        if not _same_device(selected.device, self._device):
            raise ValueError("Completed systems left the configured CUDA device")
        if self._count + selected.num_graphs > self._capacity:
            raise RuntimeError("Completion sink capacity exceeded")
        self._batches.append(selected)
        self._count += selected.num_graphs

    def read(self) -> Batch:
        """Return all retained CUDA chunks as one Batch."""

        if not self._batches:
            raise RuntimeError("Cannot read from an empty completion sink")
        combined = self._batches[0].clone()
        for batch in self._batches[1:]:
            combined.append(batch)
        return combined

    def zero(self) -> None:
        """Release all retained completion chunks."""

        self._batches.clear()
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device


def _same_device(left: torch.device | str, right: torch.device | str) -> bool:
    """Compare devices, treating an unindexed device as the default index."""

    left, right = torch.device(left), torch.device(right)
    if left.type != right.type:
        return False
    if left.index is None or right.index is None:
        return True
    return left.index == right.index


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _interaction_model(device: torch.device) -> PipelineModelWrapper:
    """Rebuild the composed molecular model used in Module 02."""

    with warnings.catch_warnings():
        _ignore_known_aimnet2_cutoff_warning()
        aimnet = AIMNet2Wrapper.from_checkpoint(
            model_checkpoint(), device=device, dtype=torch.float32
        ).eval()
    aimnet.set_config("active_outputs", {"energy", "forces", "charges"})
    d3 = (
        DFTD3ModelWrapper(
            a1=0.566,
            a2=3.128,
            s8=0.3908,
            cutoff=15.0,
            param_file=d3_parameter_file(),
        )
        .to(device)
        .eval()
    )
    d3.set_config("active_outputs", {"energy", "forces"})
    coulomb = DirectCoulombAdapter(cutoff=15.0).to(device).eval()
    coulomb.set_config("active_outputs", {"energy", "forces"})
    model = (
        PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(aimnet, wire={"charges": "partial_charges"}),
                        coulomb,
                    ],
                    use_autograd=True,
                ),
                PipelineGroup(steps=[d3], use_autograd=False),
            ],
            neighbor_adaptation="always",
        )
        .to(device)
        .eval()
    )
    model.set_config("active_outputs", {"energy", "forces", "charges"})
    return model


class _WindowStatusHook:
    """Track one rolling per-trajectory signal and advance passing systems."""

    stage = DynamicsStage.AFTER_STEP

    def __init__(
        self,
        total_systems: int,
        device: torch.device | str,
        *,
        source_status: int,
        target_status: int,
        frequency: int,
        window_samples: int,
        required_streak: int,
    ) -> None:
        self.frequency = frequency
        self.source_status = source_status
        self.target_status = target_status
        self.window_samples = window_samples
        resolved_device = torch.device(device)
        self.history = torch.zeros(
            total_systems, self.window_samples, device=resolved_device
        )
        self.counts = torch.zeros(
            total_systems, dtype=torch.long, device=resolved_device
        )
        self.streak = torch.zeros(
            total_systems, dtype=torch.long, device=resolved_device
        )
        self.required_streak = required_streak

    def _values(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    def _passes(self, ordered: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, context: Any, stage: Any) -> None:
        del stage
        batch = context.batch
        status = batch.status.reshape(-1)
        active_indices = torch.where(status == self.source_status)[0]
        if active_indices.numel() == 0:
            return
        simulation_ids = batch.simulation_id.reshape(-1).long()[active_indices]
        values = self._values(batch)[active_indices]
        old_counts = self.counts[simulation_ids]
        self.history[simulation_ids, old_counts.remainder(self.window_samples)] = values
        new_counts = old_counts + 1
        self.counts[simulation_ids] = new_counts
        ready = new_counts >= self.window_samples
        if not ready.any():
            return
        ready_indices = active_indices[ready]
        ready_ids = simulation_ids[ready]
        ready_counts = new_counts[ready]
        order = (
            torch.arange(self.window_samples, device=batch.device).unsqueeze(0)
            + ready_counts.remainder(self.window_samples).unsqueeze(1)
        ).remainder(self.window_samples)
        ordered = torch.gather(self.history[ready_ids], 1, order)
        passing = self._passes(ordered)
        streak = torch.where(
            passing,
            self.streak[ready_ids] + 1,
            torch.zeros_like(self.streak[ready_ids]),
        )
        self.streak[ready_ids] = streak
        migrate = streak >= self.required_streak
        if migrate.any():
            status[ready_indices[migrate]] = self.target_status


class _TemperatureWindowHook(_WindowStatusHook):
    def __init__(
        self,
        total_systems: int,
        device: torch.device | str,
        *,
        frequency: int,
        window_samples: int,
        required_streak: int,
        target_temperature: float,
        temperature_tolerance: float,
        trend_tolerance: float,
    ) -> None:
        super().__init__(
            total_systems,
            device,
            source_status=0,
            target_status=1,
            frequency=frequency,
            window_samples=window_samples,
            required_streak=required_streak,
        )
        self.target = target_temperature
        self.tolerance = temperature_tolerance
        self.trend_tolerance = trend_tolerance

    def _values(self, batch: Batch) -> torch.Tensor:
        kinetic = _kinetic_energy_per_graph(batch)
        atom_counts = batch.num_nodes_per_graph.to(dtype=kinetic.dtype)
        return (2.0 * kinetic) / (3.0 * atom_counts * _KB_EV_PER_K)

    def _passes(self, ordered: torch.Tensor) -> torch.Tensor:
        midpoint = self.window_samples // 2
        mean = ordered.mean(dim=1)
        trend = ordered[:, midpoint:].mean(dim=1) - ordered[:, :midpoint].mean(dim=1)
        return (mean.sub(self.target).abs() <= self.tolerance) & (
            trend.abs() <= self.trend_tolerance
        )


class _EnergyWindowHook(_WindowStatusHook):
    def __init__(
        self,
        total_systems: int,
        device: torch.device | str,
        *,
        frequency: int,
        window_samples: int,
        required_streak: int,
        span_per_atom: float,
    ) -> None:
        super().__init__(
            total_systems,
            device,
            source_status=1,
            target_status=2,
            frequency=frequency,
            window_samples=window_samples,
            required_streak=required_streak,
        )
        self.span = span_per_atom

    def _values(self, batch: Batch) -> torch.Tensor:
        kinetic = _kinetic_energy_per_graph(batch)
        potential = batch.energy.reshape(-1)
        return (potential + kinetic) / batch.num_nodes_per_graph

    def _passes(self, ordered: torch.Tensor) -> torch.Tensor:
        return ordered.amax(dim=1).sub(ordered.amin(dim=1)) <= self.span


def group_indices_by_capacity(
    batch: Batch, *, max_batch_size: int, max_atoms: int
) -> list[torch.Tensor]:
    """Return indices packed under the shared system and atom limits."""

    groups: list[torch.Tensor] = []
    start = 0
    counts = batch.num_nodes_per_graph.tolist()
    while start < batch.num_graphs:
        end = start
        atoms = 0
        while end < batch.num_graphs:
            next_atoms = int(counts[end])
            if end > start and (
                end - start >= max_batch_size or atoms + next_atoms > max_atoms
            ):
                break
            atoms += next_atoms
            end += 1
        groups.append(torch.arange(start, end, device=batch.device))
        start = end
    return groups


def _split_batch(batch: Batch, *, max_batch_size: int, max_atoms: int) -> list[Batch]:
    """Pack capacity-limited groups for hidden trajectory preparation."""

    return [
        batch.index_select(indices)
        for indices in group_indices_by_capacity(
            batch, max_batch_size=max_batch_size, max_atoms=max_atoms
        )
    ]


def _clear_neighbor_fields(batch: Batch) -> None:
    for key in _NEIGHBOR_FIELDS:
        if getattr(batch, key, None) is not None:
            del batch[key]


def _kinetic_energy_per_graph(batch: Batch) -> torch.Tensor:
    """Compute graph-level kinetic energy with GPU-resident torch operations."""

    masses = batch.atomic_masses.reshape(-1).to(dtype=batch.velocities.dtype)
    contributions = 0.5 * masses * batch.velocities.square().sum(dim=-1)
    kinetic = torch.zeros(
        batch.num_graphs,
        dtype=contributions.dtype,
        device=contributions.device,
    )
    return kinetic.scatter_add_(0, batch.batch_idx.long(), contributions)


def _run_neighbor_hooks(
    hooks: Sequence[Any],
    batch: Batch,
    model: PipelineModelWrapper,
    workflow: NVE,
) -> None:
    """Run public model neighbor hooks before an explicit model evaluation."""

    context = DynamicsContext(batch=batch, model=model, workflow=workflow)
    for hook in hooks:
        hook(context, DynamicsStage.BEFORE_COMPUTE)


def _prepare_initial_forces(
    records: Sequence[AtomicData],
    model: PipelineModelWrapper,
    device: torch.device,
    *,
    max_batch_size: int,
    max_atoms: int,
) -> tuple[AtomicData, ...]:
    """Compute one common initial force evaluation in capacity-sized groups."""

    full = Batch.from_data_list([record.clone() for record in records], device=device)
    prepared: list[AtomicData] = []
    # The timestep is unused by compute(); this stage only evaluates initial forces.
    evaluator = NVE(model=model, dt=1.0)
    neighbor_hooks = model.make_neighbor_hooks()
    for group in _split_batch(
        full,
        max_batch_size=max_batch_size,
        max_atoms=max_atoms,
    ):
        _run_neighbor_hooks(neighbor_hooks, group, model, evaluator)
        evaluator.compute(group)
        _clear_neighbor_fields(group)
        prepared.extend(group.to_data_list())
    return tuple(prepared)


def prepare_ncia250_trajectories(
    device: torch.device | str,
    *,
    batch_multiplier: Real,
    initial_temperature: float,
    random_seed: int,
) -> tuple[tuple[AtomicData, ...], PipelineModelWrapper]:
    """Load 205 NCIA250 complexes and prepare independent trajectories."""

    resolved_device = torch.device(device)
    atoms, rows, _, _, _ = load_ncia250()
    study_rows = rows[rows["fragment"].eq("AB")].reset_index(drop=True)
    source_systems = len(study_rows)
    model = _interaction_model(resolved_device)
    source_ids = _scaled_source_ids(source_systems, batch_multiplier)
    records: list[AtomicData] = []
    for simulation_id, source_id in enumerate(source_ids):
        row = study_rows.iloc[source_id]
        record = AtomicData.from_atoms(
            atoms[int(row["graph_index"])], device=resolved_device
        )
        record.add_system_property(
            "simulation_id",
            torch.tensor([[simulation_id]], dtype=torch.long, device=resolved_device),
        )
        record.add_system_property(
            "status",
            torch.zeros((1, 1), dtype=torch.long, device=resolved_device),
        )
        record.add_system_property(
            "energy", torch.zeros((1, 1), device=resolved_device)
        )
        record.add_node_property("forces", torch.zeros_like(record.positions))
        records.append(record)

    initialized = Batch.from_data_list(records, device=resolved_device)
    initialized["velocities"] = torch.zeros_like(initialized.positions)
    initialize_velocities(
        initialized.velocities,
        initialized.atomic_masses,
        torch.full(
            (initialized.num_graphs,),
            initial_temperature,
            device=resolved_device,
        ),
        initialized.batch_idx.int(),
        random_seed=random_seed,
        remove_com=True,
        remove_rotations=False,
        rescale=True,
    )
    preparation_max_atoms = math.ceil(
        initialized.num_nodes_per_graph.float().mean().item()
        * _PREPARATION_MAX_BATCH_SIZE
    )
    prepared = _prepare_initial_forces(
        initialized.to_data_list(),
        model,
        resolved_device,
        max_batch_size=_PREPARATION_MAX_BATCH_SIZE,
        max_atoms=preparation_max_atoms,
    )
    _synchronize(resolved_device)
    return prepared, model


def _keep_stage_active(values: torch.Tensor) -> torch.Tensor:
    """Return one false convergence value per active trajectory."""

    return torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)


def keep_temperature_hook_in_control() -> ConvergenceHook:
    """Keep the rolling temperature hook in control of status 0 to 1."""

    return ConvergenceHook(
        criteria={
            "key": "energy",
            "threshold": 0.0,
            "custom_op": _keep_stage_active,
        }
    )


def temperature_window_hook(
    *,
    total_trajectories: int,
    device: torch.device | str,
    frequency: int,
    window_samples: int,
    required_streak: int,
    target_temperature: float,
    temperature_tolerance: float,
    trend_tolerance: float,
) -> _TemperatureWindowHook:
    """Return fresh GPU history for the per-trajectory NVT rule."""

    return _TemperatureWindowHook(
        total_trajectories,
        device,
        frequency=frequency,
        window_samples=window_samples,
        required_streak=required_streak,
        target_temperature=target_temperature,
        temperature_tolerance=temperature_tolerance,
        trend_tolerance=trend_tolerance,
    )


def energy_window_hook(
    *,
    total_trajectories: int,
    device: torch.device | str,
    frequency: int,
    window_samples: int,
    required_streak: int,
    span_per_atom: float,
) -> _EnergyWindowHook:
    """Return fresh GPU history for the per-trajectory NVE rule."""

    return _EnergyWindowHook(
        total_trajectories,
        device,
        frequency=frequency,
        window_samples=window_samples,
        required_streak=required_streak,
        span_per_atom=span_per_atom,
    )


def time_route(device: torch.device | str) -> RouteTimer:
    """Return a synchronized timer for one scheduling route."""

    return RouteTimer(device)


def _batch_identity_tensor(batch: Batch) -> torch.Tensor:
    values = getattr(batch, "simulation_id", None)
    if values is None:
        raise ValueError("Study batches must carry simulation_id")
    return values.reshape(-1).detach().clone()


def _result_from_batches(
    route: str,
    batches: Sequence[Batch],
    timer: RouteTimer,
    *,
    expected_trajectories: int,
    device: torch.device | str,
    replacement_systems: int = 0,
) -> RouteResult:
    if not batches:
        raise ValueError(f"{route} produced no completed batches")
    expected_device = torch.device(device)
    if any(not _same_device(batch.device, expected_device) for batch in batches):
        raise ValueError(f"{route} trajectories left the selected device")
    identities = torch.cat([_batch_identity_tensor(batch) for batch in batches])
    expected = torch.arange(
        expected_trajectories, device=identities.device, dtype=identities.dtype
    )
    if identities.numel() != expected.numel() or not torch.equal(
        torch.sort(identities).values, expected
    ):
        raise ValueError(f"{route} did not complete every trajectory exactly once")
    return RouteResult(
        route=route,
        completed_systems=identities.numel(),
        elapsed_s=timer.elapsed_s,
        peak_allocated_gib=timer.peak_allocated_gib,
        replacement_systems=replacement_systems,
    )


def summarize_sequential(
    results: Sequence[Batch],
    timer: RouteTimer,
    *,
    expected_trajectories: int,
    device: torch.device | str,
) -> RouteResult:
    """Check the sequential-stage outputs and summarize their timing."""

    return _result_from_batches(
        "Run in stages",
        results,
        timer,
        expected_trajectories=expected_trajectories,
        device=device,
    )


def summarize_fused(
    results: Sequence[Batch],
    timer: RouteTimer,
    *,
    expected_trajectories: int,
    device: torch.device | str,
) -> RouteResult:
    """Check the unrefilled FusedStage outputs and summarize their timing."""

    if any(not bool((batch.status.reshape(-1) == 2).all()) for batch in results):
        raise ValueError("FusedStage stopped before every trajectory completed")
    return _result_from_batches(
        "FusedStage",
        results,
        timer,
        expected_trajectories=expected_trajectories,
        device=device,
    )


def _mixed_trace_rows(
    snapshots: Sequence[_RouteSnapshot], *, limit: int = 6
) -> list[_RouteSnapshot]:
    """Return natural mixed-stage observations or fail with a clear cause."""

    mixed = [row for row in snapshots if row.nvt_systems and row.nve_systems]
    if not mixed:
        raise RuntimeError(
            "The fused trace produced no natural mixed NVT/NVE state; "
            "check the rolling-window rules and trace frequency."
        )
    return mixed[:limit]


def stage_trace_hook(*, frequency: int) -> _RouteTraceHook:
    """Return a full-Batch trace hook for one fresh fused route."""

    return _RouteTraceHook(frequency=frequency)


def summarize_stage_trace(trace: _RouteTraceHook, *, limit: int = 6) -> pd.DataFrame:
    """Format the first natural observations with both MD stages active."""

    selected = _mixed_trace_rows(trace.snapshots, limit=limit)
    return pd.DataFrame(
        [
            {
                "NVT trajectories": row.nvt_systems,
                "NVE trajectories": row.nve_systems,
                "completed": row.completed_systems,
                "active Batch": row.active_systems,
            }
            for row in selected
        ]
    )


def check_inflight_run(
    completed: DataSink,
    remaining: Batch | None,
    timer: RouteTimer,
    *,
    expected_trajectories: int,
    device: torch.device | str,
    max_batch_size: int,
) -> RouteResult:
    """Check unique inflight completion and report replacement activity."""

    if remaining is not None:
        raise ValueError("Inflight batching stopped with trajectories still active")
    completed_batch = completed.read()
    initially_active = min(max_batch_size, expected_trajectories)
    result = _result_from_batches(
        "FusedStage + inflight",
        [completed_batch],
        timer,
        expected_trajectories=expected_trajectories,
        device=device,
        replacement_systems=expected_trajectories - initially_active,
    )
    return result


def compare_routes(*results: RouteResult) -> pd.DataFrame:
    """Return comparable route measurements in one compact table."""

    return pd.DataFrame([result.summary for result in results])
