"""Dependency-light scientific diagnostics for the water IR workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .topology import directed_ring_masks, kabsch_rmsd_frames, longest_false_run

if TYPE_CHECKING:
    from .capture import IRTrajectory


def _to_numpy(value: Any) -> np.ndarray:
    """Detach a Torch-like value if needed, then return a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def mass_only_invariance(
    batch: Any,
    outputs: Mapping[str, Any],
    *,
    atol_energy: float = 1e-5,
    atol_forces: float = 2e-5,
    atol_charges: float = 2e-6,
) -> dict[str, float]:
    """Assert that H/D changes only masses, not PES/model predictions."""

    energies = _to_numpy(outputs["energy"]).reshape(-1)
    forces = _to_numpy(outputs["forces"])
    charges = _to_numpy(outputs["charges"]).reshape(-1)
    positions = _to_numpy(batch.positions)
    atomic_numbers = _to_numpy(batch.atomic_numbers)
    atomic_masses = _to_numpy(batch.atomic_masses)
    boundaries = _to_numpy(batch.batch_ptr).astype(int).tolist()

    deltas: dict[str, float] = {}
    for left, right, label in ((0, 1, "monomer"), (2, 3, "hexamer")):
        l0, l1 = boundaries[left], boundaries[left + 1]
        r0, r1 = boundaries[right], boundaries[right + 1]
        if (l1 - l0) != (r1 - r0):
            raise AssertionError(f"{label}: isotope pair topology differs")
        if not np.array_equal(atomic_numbers[l0:l1], atomic_numbers[r0:r1]):
            raise AssertionError(f"{label}: deuterium must keep atomic number 1")
        np.testing.assert_allclose(
            positions[l0:l1], positions[r0:r1], rtol=1e-7, atol=1e-7
        )

        energy_delta = float(abs(energies[left] - energies[right]))
        force_delta = float(np.max(abs(forces[l0:l1] - forces[r0:r1])))
        charge_delta = float(np.max(abs(charges[l0:l1] - charges[r0:r1])))
        if energy_delta > atol_energy:
            raise AssertionError(f"{label}: isotope energy changed by {energy_delta}")
        if force_delta > atol_forces:
            raise AssertionError(f"{label}: isotope forces changed by {force_delta}")
        if charge_delta > atol_charges:
            raise AssertionError(f"{label}: isotope charges changed by {charge_delta}")
        deltas[f"{label}_energy_eV"] = energy_delta
        deltas[f"{label}_force_eV_A"] = force_delta
        deltas[f"{label}_charge_e"] = charge_delta

    h_mass = atomic_masses[atomic_numbers == 1][0]
    d_start = boundaries[1]
    d_numbers = atomic_numbers[d_start : boundaries[2]]
    d_mass = atomic_masses[d_start : boundaries[2]][d_numbers == 1][0]
    deltas["D_over_H_mass"] = float(d_mass / h_mass)
    return deltas


def _connected_components(points: np.ndarray, cutoff: float) -> int:
    count = len(points)
    adjacency = np.linalg.norm(points[:, None] - points[None, :], axis=-1) < cutoff
    seen: set[int] = set()
    components = 0
    for seed in range(count):
        if seed in seen:
            continue
        components += 1
        stack = [seed]
        seen.add(seed)
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def cluster_integrity(
    trajectory: "IRTrajectory",
    graph_index: int,
    *,
    oxygen_cutoff_angstrom: float = 4.0,
    h_acceptor_cutoff_angstrom: float = 2.5,
    oo_cutoff_angstrom: float = 3.5,
    hbond_angle_cutoff_deg: float = 140.0,
) -> dict[str, float | int | bool]:
    """Summarize covalent, H-bond-ring, and oxygen-skeleton integrity."""

    for name, value in (
        ("oxygen_cutoff_angstrom", oxygen_cutoff_angstrom),
        ("h_acceptor_cutoff_angstrom", h_acceptor_cutoff_angstrom),
        ("oo_cutoff_angstrom", oo_cutoff_angstrom),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(hbond_angle_cutoff_deg) or not (
        0.0 <= hbond_angle_cutoff_deg <= 180.0
    ):
        raise ValueError("hbond_angle_cutoff_deg must lie in [0, 180]")

    start, stop = trajectory.batch_ptr[graph_index : graph_index + 2]
    numbers = trajectory.atomic_numbers[start:stop]
    frames = trajectory.positions_angstrom[:, start:stop]
    oxygen_local = np.flatnonzero(numbers == 8)
    hydrogen_local = np.flatnonzero(numbers == 1)
    if len(oxygen_local) == 0 or len(hydrogen_local) == 0:
        raise ValueError("cluster_integrity requires a water graph")
    reference = frames[0]

    assignment = np.argmin(
        np.linalg.norm(
            reference[hydrogen_local, None] - reference[oxygen_local][None, :],
            axis=-1,
        ),
        axis=1,
    )
    oh_distances = np.linalg.norm(
        frames[:, hydrogen_local] - frames[:, oxygen_local[assignment]], axis=-1
    )

    oxygen_frames = frames[:, oxygen_local]
    donor_oxygen = oxygen_frames[:, assignment]
    hydrogen_frames = frames[:, hydrogen_local]
    h_to_donor = donor_oxygen - hydrogen_frames
    h_to_acceptor = oxygen_frames[:, None, :, :] - hydrogen_frames[:, :, None, :]
    h_acceptor_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    cosine = np.sum(h_to_donor[:, :, None, :] * h_to_acceptor, axis=-1)
    cosine /= np.linalg.norm(h_to_donor, axis=-1)[:, :, None]
    cosine /= np.linalg.norm(h_to_acceptor, axis=-1)
    hbond_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    oo_distance = np.linalg.norm(
        donor_oxygen[:, :, None, :] - oxygen_frames[:, None, :, :], axis=-1
    )
    is_donor = np.arange(len(oxygen_local))[None, None, :] == assignment[None, :, None]
    hbond_mask = (
        (h_acceptor_distance <= float(h_acceptor_cutoff_angstrom))
        & (oo_distance <= float(oo_cutoff_angstrom))
        & (hbond_angle >= float(hbond_angle_cutoff_deg))
        & ~is_donor
    )
    hbond_count = np.sum(hbond_mask, axis=(1, 2))

    adjacency = np.zeros(
        (frames.shape[0], len(oxygen_local), len(oxygen_local)), dtype=bool
    )
    for hydrogen, donor in enumerate(assignment):
        adjacency[:, donor] |= hbond_mask[:, hydrogen]
    ring_masks = directed_ring_masks(adjacency)
    first_initial_loss = np.flatnonzero(~ring_masks.initial_cycle)
    components = [
        _connected_components(frame, oxygen_cutoff_angstrom) for frame in oxygen_frames
    ]
    centered = oxygen_frames - oxygen_frames.mean(axis=1, keepdims=True)
    radius_gyration = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))
    rmsd = kabsch_rmsd_frames(oxygen_frames[0], oxygen_frames)
    return {
        "oxygen_count": int(len(oxygen_local)),
        "max_OH_angstrom": float(oh_distances.max()),
        "max_oxygen_components": int(max(components)),
        "H_bonds_initial": int(hbond_count[0]),
        "H_bonds_min": int(hbond_count.min()),
        "H_bonds_max": int(hbond_count.max()),
        "single_ring_fraction": float(ring_masks.exact_single_ring.mean()),
        "all_frames_single_ring": bool(ring_masks.exact_single_ring.all()),
        "directed_six_cycle_fraction": float(ring_masks.any_cycle.mean()),
        "all_frames_have_directed_six_cycle": bool(ring_masks.any_cycle.all()),
        "initial_ring_fraction": float(ring_masks.initial_cycle.mean()),
        "all_frames_initial_ring": bool(ring_masks.initial_cycle.all()),
        "first_initial_ring_loss_ps": (
            float(first_initial_loss[0] * trajectory.dt_fs / 1000.0)
            if len(first_initial_loss)
            else float("nan")
        ),
        "longest_initial_ring_absence_ps": float(
            longest_false_run(ring_masks.initial_cycle) * trajectory.dt_fs / 1000.0
        ),
        "maximum_directed_cycle_multiplicity": int(ring_masks.cycle_multiplicity.max()),
        "Rg_initial_angstrom": float(radius_gyration[0]),
        "Rg_min_over_initial": float(
            radius_gyration.min() / max(radius_gyration[0], 1e-12)
        ),
        "Rg_max_over_initial": float(
            radius_gyration.max() / max(radius_gyration[0], 1e-12)
        ),
        "oxygen_RMSD_max_angstrom": float(rmsd.max()),
    }
