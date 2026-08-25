# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched relaxation execution and the Module 00 time perspective."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import units as ase_units
from nvalchemi.data import AtomicData, Batch

from .catalog import mace_checkpoint_path
from .figures import (
    BORDER,
    MUTED,
    NVIDIA_GREEN,
    NVIDIA_ORANGE,
    TEXT,
    TUTORIAL_FIGURE_WIDTH,
    TUTORIAL_VISUAL_MAX_WIDTH,
    display_figure,
    polish_axis,
)
from .model_support import suppress_known_mace_load_noise
from .viewers import (
    MAX_RELAXATION_SYSTEMS,
    show_relaxation_player,
    snapshot_batches,
)

_DFT_TIME_MULTIPLIER = 1_000


def prepare_dynamics_batch(batch: Batch) -> Batch:
    """Clone a batch and allocate the fields updated by dynamics workflows."""

    prepared = batch.clone()
    # Dynamics owns a fresh autograd graph for each compute step.
    prepared.positions = prepared.positions.detach().clone()
    dtype = prepared.positions.dtype
    device = prepared.device
    counts = prepared.num_nodes_per_graph.tolist()
    if "energy" not in prepared:
        records = prepared.to_data_list()
        for record in records:
            record.add_system_property(
                "energy", torch.zeros((1, 1), dtype=dtype, device=device)
            )
        prepared = Batch.from_data_list(records, device=device)
    if "forces" not in prepared:
        prepared.add_key(
            "forces",
            [torch.zeros((count, 3), dtype=dtype, device=device) for count in counts],
            level="node",
        )
    if "velocities" not in prepared:
        prepared.add_key(
            "velocities",
            [torch.zeros((count, 3), dtype=dtype, device=device) for count in counts],
            level="node",
        )
    return prepared


def system_temperatures(batch: Batch) -> torch.Tensor:
    """Return the instantaneous kinetic temperature of each system in kelvin."""

    masses = batch.atomic_masses.reshape(-1)
    atom_kinetic_energy = 0.5 * masses * batch.velocities.square().sum(dim=1)
    system_kinetic_energy = torch.zeros(
        batch.num_graphs,
        device=batch.device,
        dtype=batch.positions.dtype,
    ).index_add_(0, batch.batch_idx.long(), atom_kinetic_energy)
    degrees_of_freedom = 3.0 * batch.num_nodes_per_graph.to(system_kinetic_energy)
    return 2.0 * system_kinetic_energy / (degrees_of_freedom * ase_units.kB)


def maximum_force_history(
    frames: Sequence[Batch],
    labels: Sequence[str],
) -> dict[str, list[float]]:
    """Return one maximum force in eV/Å per graph and synchronized frame."""

    history = {str(label): [] for label in labels}
    for frame in frames:
        if frame.num_graphs != len(labels):
            raise ValueError("each frame must contain one graph per label")
        for graph, label in enumerate(labels):
            data = frame.get_data(graph)
            forces = getattr(data, "forces", None)
            if forces is None:
                raise ValueError("snapshot frames must include forces")
            maximum = forces.norm(dim=1).max().detach().cpu().item()
            history[str(label)].append(float(maximum))
    return history


@contextmanager
def measure_torch_execution(
    device: torch.device | str,
) -> Iterator[dict[str, float | None]]:
    """Measure synchronized elapsed time and peak allocated CUDA memory."""

    run_device = torch.device(device)
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
        torch.cuda.reset_peak_memory_stats(run_device)

    measurement: dict[str, float | None] = {
        "seconds": None,
        "peak_memory_mib": None,
    }
    start = perf_counter()
    try:
        yield measurement
    finally:
        if run_device.type == "cuda":
            torch.cuda.synchronize(run_device)
        measurement["seconds"] = perf_counter() - start
        if run_device.type == "cuda":
            measurement["peak_memory_mib"] = (
                torch.cuda.max_memory_allocated(run_device) / 1024**2
            )


def run_relaxation_preview(
    batch: Batch,
    model: Any,
    labels: Sequence[str],
    *,
    steps: int,
    frozen_layers: int = 0,
    batch_multiplier: int = 1,
) -> dict[str, Any]:
    """Run a bounded batched relaxation and return inputs for the player."""

    from nvalchemi._typing import AtomCategory
    from nvalchemi.dynamics import FIRE2, HostMemory
    from nvalchemi.dynamics.hooks import FreezeAtomsHook, SnapshotHook
    from nvalchemi.neighbors import compute_neighbors

    if steps < 1:
        raise ValueError("steps must be positive")
    if (
        isinstance(batch_multiplier, bool)
        or not isinstance(batch_multiplier, int)
        or batch_multiplier < 1
    ):
        raise ValueError("batch_multiplier must be a positive integer")
    if batch.num_graphs != len(labels) * batch_multiplier:
        raise ValueError(
            "the batch must contain batch_multiplier copies of every labeled graph"
        )
    if frozen_layers < 0:
        raise ValueError("frozen_layers cannot be negative")

    sink = HostMemory(capacity=(steps + 1) * batch.num_graphs)
    snapshots = SnapshotHook(sink, frequency=1)
    dynamics_batch = prepare_dynamics_batch(batch)
    hooks = [*model.make_neighbor_hooks()]
    if frozen_layers:
        for start, stop in zip(
            dynamics_batch.batch_ptr[:-1],
            dynamics_batch.batch_ptr[1:],
            strict=True,
        ):
            start_index, stop_index = int(start.item()), int(stop.item())
            z = dynamics_batch.positions[start_index:stop_index, 2]
            levels = torch.unique(torch.round(z * 1_000) / 1_000).sort().values
            if len(levels) <= frozen_layers:
                raise ValueError("surface needs more z layers than frozen_layers")
            frozen = z <= levels[frozen_layers - 1] + 1.0e-3
            dynamics_batch.atom_categories[start_index:stop_index][frozen] = (
                AtomCategory.SPECIAL.value
            )
        hooks.insert(0, FreezeAtomsHook())
    hooks.append(snapshots)
    optimizer = FIRE2(
        model=model,
        dt=0.005,
        maxstep=0.03,
        n_steps=steps,
        hooks=hooks,
    )
    compute_neighbors(
        dynamics_batch,
        config=model.model_config.neighbor_config,
    )
    optimizer.compute(dynamics_batch)
    sink.write(dynamics_batch)
    with measure_torch_execution(batch.device) as measurement:
        optimizer.run(dynamics_batch)

    frames = snapshot_batches(sink, batch.num_graphs)
    visible_frames = [
        Batch.from_data_list(
            frame.to_data_list()[: len(labels)],
            device=frame.device,
        )
        for frame in frames
    ]
    run_device = torch.device(batch.device)
    if run_device.type == "cuda":
        device_index = run_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_label = f"cuda:{device_index} · {torch.cuda.get_device_name(run_device)}"
    else:
        device_label = "CPU"

    return {
        "trajectories": visible_frames,
        "labels": [str(label) for label in labels],
        "force_history": maximum_force_history(visible_frames, labels),
        "device": device_label,
        "elapsed_s": measurement["seconds"],
        "convergence": f"{steps}-step preview",
        "batch_multiplier": batch_multiplier,
    }


def relax_selected_structures(
    selection: Any,
    *,
    device: torch.device | str,
    steps: int = 16,
    distortion_std_angstrom: float = 0.10,
    distortion_seed: int = 7,
    batch_multiplier: int,
) -> dict[str, Any]:
    """Relax one picker selection as a single Batch for the Module 00 tour."""

    specs = selection.selected_specs()
    if len(specs) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            f"Choose at most {MAX_RELAXATION_SYSTEMS} systems for the synchronized player"
        )
    if not np.isfinite(distortion_std_angstrom) or distortion_std_angstrom < 0:
        raise ValueError("distortion_std_angstrom must be finite and non-negative")

    run_device = torch.device(device)
    if (
        isinstance(batch_multiplier, bool)
        or not isinstance(batch_multiplier, int)
        or batch_multiplier < 1
    ):
        raise ValueError("batch_multiplier must be a positive integer")
    random = np.random.default_rng(distortion_seed)
    structures = []
    for spec in specs:
        structure = spec.build()
        if distortion_std_angstrom > 0 and len(structure) > 1:
            displacement = random.normal(
                scale=distortion_std_angstrom,
                size=structure.positions.shape,
            )
            displacement -= displacement.mean(axis=0, keepdims=True)
            structure.positions += displacement
        structures.append(structure)

    graphs = [
        AtomicData.from_atoms(structure, device=run_device)
        for _ in range(batch_multiplier)
        for structure in structures
    ]
    batch = Batch.from_data_list(graphs, device=run_device)
    labels = [spec.display_name for spec in specs]

    with suppress_known_mace_load_noise():
        from nvalchemi.models import MACEWrapper

        model = MACEWrapper.from_checkpoint(
            mace_checkpoint_path("medium-mpa-0"),
            device=run_device,
            dtype=torch.float32,
        ).eval()
    model.set_config("active_outputs", {"energy", "forces"})

    preview = run_relaxation_preview(
        batch,
        model,
        labels,
        steps=steps,
        batch_multiplier=batch_multiplier,
    )
    preview["reference_structures"] = dict(zip(labels, structures, strict=True))
    return {
        "batch": batch,
        "batch_multiplier": batch_multiplier,
        "labels": labels,
        "preview": preview,
        "player": show_relaxation_player(**preview),
    }


def _short_duration(seconds: float) -> str:
    """Format seconds for compact labels on the opening comparison plot."""

    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.2f} s"


def plot_relaxation_time_perspective(relaxation: Mapping[str, Any]) -> None:
    """Place one measured Toolkit relaxation beside an illustrative DFT estimate."""

    labels = list(relaxation["labels"])
    preview = relaxation["preview"]
    systems = len(labels)
    batch_multiplier = int(preview.get("batch_multiplier", 1))
    total_calculations = systems * batch_multiplier
    measured_seconds = float(preview["elapsed_s"])
    projected_seconds = measured_seconds * _DFT_TIME_MULTIPLIER

    figure, axis = plt.subplots(figsize=(TUTORIAL_FIGURE_WIDTH, 4.0))
    bars = axis.bar(
        ["MACE in Toolkit\nmeasured", "DFT estimate\nsame work"],
        [measured_seconds, projected_seconds],
        color=[NVIDIA_GREEN, NVIDIA_ORANGE],
        edgecolor="none",
        width=0.48,
        zorder=3,
    )
    axis.set_yscale("log")
    axis.set_ylabel("Elapsed time [s, log scale]", fontsize=11.0)
    axis.bar_label(
        bars,
        labels=[
            _short_duration(measured_seconds),
            _short_duration(projected_seconds),
        ],
        color=TEXT,
        fontweight="bold",
        padding=5,
        fontsize=11.0,
    )
    axis.annotate(
        "MLIP ~O(N)",
        xy=(bars[0].get_x() + bars[0].get_width() / 2, measured_seconds),
        color=NVIDIA_GREEN,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        xytext=(0, 18),
        textcoords="offset points",
    )
    axis.annotate(
        "DFT ~O(N³)",
        xy=(bars[1].get_x() + bars[1].get_width() / 2, projected_seconds),
        color=NVIDIA_ORANGE,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        xytext=(0, 18),
        textcoords="offset points",
    )
    polish_axis(axis, grid="y")
    axis.grid(axis="y", which="minor", visible=False)
    axis.spines["left"].set_color(BORDER)
    axis.spines["bottom"].set_color(BORDER)
    axis.tick_params(axis="x", colors=MUTED, labelsize=11.0)
    axis.tick_params(axis="y", colors=MUTED, labelsize=9.5)
    axis.yaxis.label.set_color(MUTED)
    axis.set_ylim(
        bottom=max(measured_seconds / 4.0, np.finfo(float).tiny),
        top=projected_seconds * 4.0,
    )
    display_figure(
        figure,
        f"MACE in Toolkit is the measured time for {total_calculations} "
        f"calculations on {preview['device']}. The DFT value is an illustrative "
        f"{_DFT_TIME_MULTIPLIER:,.0f}× estimate for the same relaxation work. "
        "Conventional Kohn-Sham DFT is roughly cubic in system size, while "
        "local MLIPs are roughly linear.",
        max_width=TUTORIAL_VISUAL_MAX_WIDTH,
    )
