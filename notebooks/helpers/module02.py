# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper interface used by Module 02, models and simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .common import configure_tutorial
from .model_support import (
    DirectCoulombAdapter,
    d3_parameter_file,
    model_checkpoint,
)
from .nci import (
    checked_nci_interaction_curves,
    interaction_components,
    load_nci_atlas,
    max_system_charge_error,
    nci_curve_deltas,
    plot_nci_interaction_curves,
)
from .ncia250 import load_ncia250
from .relaxation import graph_temperatures
from .viewers import (
    show_bias_comparison_player,
    show_nci_separation_player,
)

__all__ = [
    "DirectCoulombAdapter",
    "checked_nci_interaction_curves",
    "configure_tutorial",
    "d3_parameter_file",
    "equilibrium_nci_complexes",
    "fragment_com_harmonic_bias",
    "fragment_com_separations",
    "graph_temperatures",
    "interaction_components",
    "load_nci_atlas",
    "load_ncia250",
    "max_system_charge_error",
    "model_checkpoint",
    "nci_curve_deltas",
    "nci_scan_player_data",
    "plot_fragment_separation_comparison",
    "plot_nci_interaction_curves",
    "show_bias_comparison_player",
    "show_nci_separation_player",
]


def equilibrium_nci_complexes(
    atoms: Sequence[Any], graph_rows: Any
) -> tuple[list[str], list[Any], list[int]]:
    """Return the three equilibrium AB complexes and their fragment-A sizes."""

    equilibrium = graph_rows[graph_rows["scale"].eq(1.0)]
    complex_rows = equilibrium[equilibrium["fragment"].eq("AB")]
    labels: list[str] = []
    structures: list[Any] = []
    fragment_a_sizes: list[int] = []

    for row in complex_rows.itertuples(index=False):
        fragment_a = equilibrium[
            equilibrium["system_name"].eq(row.system_name)
            & equilibrium["fragment"].eq("A")
        ]
        if len(fragment_a) != 1:
            raise ValueError(f"Expected one fragment A for {row.system_name!r}")
        labels.append(str(row.system_name))
        structures.append(atoms[int(row.graph_index)].copy())
        fragment_a_sizes.append(int(fragment_a.iloc[0]["natoms"]))

    if len(structures) != 3:
        raise ValueError("Expected three equilibrium NCI complexes")
    return labels, structures, fragment_a_sizes


def nci_scan_player_data(
    atoms: Sequence[Any], graph_rows: Any
) -> tuple[dict[str, list[Any]], dict[str, int], list[float]]:
    """Group the three AB scans for synchronized molecular playback."""

    required = {"fragment", "graph_index", "natoms", "scale", "system_name"}
    missing = required - set(graph_rows.columns)
    if missing:
        raise ValueError(f"graph_rows are missing {sorted(missing)!r}")

    complex_rows = graph_rows[graph_rows["fragment"].eq("AB")]
    labels = complex_rows["system_name"].astype(str).drop_duplicates().tolist()
    if len(labels) != 3:
        raise ValueError("Expected three NCI complex scans")

    scales = sorted(complex_rows["scale"].astype(float).unique().tolist())
    trajectories: dict[str, list[Any]] = {}
    fragment_a_sizes: dict[str, int] = {}
    for label in labels:
        rows = complex_rows[complex_rows["system_name"].eq(label)].sort_values("scale")
        observed_scales = rows["scale"].astype(float).tolist()
        if len(rows) != len(scales) or not np.allclose(observed_scales, scales):
            raise ValueError(f"NCI scan scales do not align for {label!r}")

        fragment_a = (
            graph_rows[
                graph_rows["system_name"].eq(label) & graph_rows["fragment"].eq("A")
            ]["natoms"]
            .astype(int)
            .unique()
        )
        if len(fragment_a) != 1:
            raise ValueError(f"Expected one fragment-A size for {label!r}")

        indices = rows["graph_index"].astype(int).tolist()
        if any(index < 0 or index >= len(atoms) for index in indices):
            raise ValueError(f"A graph index is outside atoms for {label!r}")
        trajectories[label] = [atoms[index].copy() for index in indices]
        fragment_a_sizes[label] = int(fragment_a[0])

    return trajectories, fragment_a_sizes, scales


def fragment_com_separations(
    batch: Any, fragment_a_sizes: Sequence[int]
) -> torch.Tensor:
    """Measure mass-weighted fragment center-of-mass separations in Å."""

    if len(fragment_a_sizes) != batch.num_graphs:
        raise ValueError("fragment_a_sizes must contain one split per graph")

    separations = []
    for graph, fragment_a_size in enumerate(fragment_a_sizes):
        graph_data = batch.get_data(graph)
        split = int(fragment_a_size)
        masses = graph_data.atomic_masses.reshape(-1, 1)
        positions = graph_data.positions
        if split < 1 or split >= len(positions):
            raise ValueError("each fragment split must lie inside its graph")
        com_a = (masses[:split] * positions[:split]).sum(0) / masses[:split].sum()
        com_b = (masses[split:] * positions[split:]).sum(0) / masses[split:].sum()
        separations.append(torch.linalg.vector_norm(com_b - com_a))
    return torch.stack(separations)


def fragment_com_harmonic_bias(
    batch: Any,
    fragment_a_sizes: Sequence[int],
    reference_distances: torch.Tensor,
    force_constant_ev_per_a2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return harmonic fragment-COM restraint energies and atomic forces."""

    if len(fragment_a_sizes) != batch.num_graphs:
        raise ValueError("fragment_a_sizes must contain one split per graph")
    if reference_distances.numel() != batch.num_graphs:
        raise ValueError("reference_distances must contain one value per graph")
    if not np.isfinite(force_constant_ev_per_a2) or force_constant_ev_per_a2 <= 0:
        raise ValueError("force_constant_ev_per_a2 must be positive and finite")

    energy = torch.zeros_like(batch.energy)
    forces = torch.zeros_like(batch.forces)
    for graph, fragment_a_size in enumerate(fragment_a_sizes):
        start = int(batch.batch_ptr[graph])
        stop = int(batch.batch_ptr[graph + 1])
        split = start + int(fragment_a_size)
        masses = batch.atomic_masses[start:stop].reshape(-1, 1)
        positions = batch.positions[start:stop]
        local_split = split - start

        mass_a = masses[:local_split]
        mass_b = masses[local_split:]
        com_a = (mass_a * positions[:local_split]).sum(0) / mass_a.sum()
        com_b = (mass_b * positions[local_split:]).sum(0) / mass_b.sum()
        displacement = com_b - com_a
        distance = torch.linalg.vector_norm(displacement)
        direction = displacement / distance.clamp_min(1.0e-12)
        extension = distance - reference_distances[graph]
        force_b = -force_constant_ev_per_a2 * extension * direction

        energy[graph, 0] = 0.5 * force_constant_ev_per_a2 * extension.square()
        forces[start:split] -= force_b * mass_a / mass_a.sum()
        forces[split:stop] += force_b * mass_b / mass_b.sum()
    return energy, forces


def plot_fragment_separation_comparison(
    time_fs: Sequence[float],
    plain: Mapping[str, Sequence[float]],
    biased: Mapping[str, Sequence[float]],
) -> None:
    """Plot matched fragment-separation traces for up to five molecular copies."""

    import matplotlib.pyplot as plt

    labels = list(plain)
    if labels != list(biased):
        raise ValueError("trajectory labels must align")
    if not 1 <= len(labels) <= 5:
        raise ValueError("the Module 02 comparison supports one to five copies")
    times = np.asarray(time_fs, dtype=float)

    columns = min(3, len(labels))
    rows = (len(labels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.2, 3.2 * rows),
        dpi=160,
        squeeze=False,
        sharey=False,
    )
    flat_axes = axes.ravel()
    for axis, label in zip(flat_axes, labels):
        free = np.asarray(plain[label], dtype=float)
        held = np.asarray(biased[label], dtype=float)
        if len(free) != len(times) or len(held) != len(times):
            raise ValueError("time_fs and separation traces must have equal length")
        axis.plot(times, free, color="#6B7280", linewidth=2, label="plain MD")
        axis.plot(times, held, color="#76B900", linewidth=2.4, label="with bias")
        axis.set_title(label.replace(" - ", "\n"), fontsize=10)
        axis.set_xlabel("time [fs]")
        axis.set_ylabel("fragment COM separation R [Å]")
        axis.grid(alpha=0.22)
    for axis in flat_axes[len(labels) :]:
        axis.remove()
    flat_axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    plt.show()
