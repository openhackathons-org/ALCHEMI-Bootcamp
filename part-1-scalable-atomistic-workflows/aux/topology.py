"""NumPy-only topology analysis for the water IR trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from numpy.typing import NDArray


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Return an independent, non-writeable result array."""

    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DirectedRingMasks:
    """Per-frame directed-cycle classifications for one molecular graph."""

    any_cycle: NDArray[np.bool_]
    exact_single_ring: NDArray[np.bool_]
    initial_cycle: NDArray[np.bool_]
    cycle_multiplicity: NDArray[np.int64]
    initial_cycle_nodes: tuple[int, ...] | None


@dataclass(frozen=True)
class WaterTopologyObservables:
    """Frame-resolved topology values for one finite water cluster.

    The calculation is NumPy-only and receives coordinate arrays rather than a
    trajectory object.  Callers keep responsibility for selecting the graph,
    naming the system, and choosing every distance or angle cutoff.
    """

    oxygen_count: int
    hydrogen_bond_count: NDArray[np.int64]
    ring_masks: DirectedRingMasks
    oxygen_component_count: NDArray[np.int64] | None
    oxygen_radius_gyration_angstrom: NDArray[np.float64]
    oxygen_rmsd_angstrom: NDArray[np.float64]
    max_assigned_oh_angstrom: NDArray[np.float64]


def directed_ring_masks(adjacency: np.ndarray) -> DirectedRingMasks:
    """Classify directed Hamiltonian cycles in a frame sequence.

    ``adjacency`` has shape ``(frames, nodes, nodes)``. A cycle can remain
    present while transient extra edges appear; ``exact_single_ring`` is the
    stricter condition that every node also has exactly one incoming and one
    outgoing edge.
    """

    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape (frames, nodes, nodes)")
    if adjacency.shape[0] == 0 or adjacency.shape[1] < 2:
        raise ValueError("adjacency must contain frames and at least two nodes")

    node_count = adjacency.shape[1]
    cycle_nodes: list[tuple[int, ...]] = []
    cycle_presence: list[np.ndarray] = []
    # Pin node 0 first so rotations of one directed cycle are not recounted.
    for tail in itertools.permutations(range(1, node_count)):
        nodes = (0, *tail)
        sources = np.asarray(nodes)
        targets = np.asarray((*tail, 0))
        cycle_nodes.append(nodes)
        cycle_presence.append(adjacency[:, sources, targets].all(axis=1))

    cycle_matrix = np.column_stack(cycle_presence)
    any_cycle = cycle_matrix.any(axis=1)
    cycle_multiplicity = cycle_matrix.sum(axis=1)
    donor_counts = adjacency.sum(axis=2)
    acceptor_counts = adjacency.sum(axis=1)
    exact_single_ring = (
        any_cycle & (donor_counts == 1).all(axis=1) & (acceptor_counts == 1).all(axis=1)
    )

    initial_candidates = np.flatnonzero(cycle_matrix[0])
    if len(initial_candidates):
        initial_index = int(initial_candidates[0])
        initial_cycle = cycle_matrix[:, initial_index]
        initial_cycle_nodes: tuple[int, ...] | None = cycle_nodes[initial_index]
    else:
        initial_cycle = np.zeros(adjacency.shape[0], dtype=bool)
        initial_cycle_nodes = None

    return DirectedRingMasks(
        any_cycle=_readonly_array(any_cycle),
        exact_single_ring=_readonly_array(exact_single_ring),
        initial_cycle=_readonly_array(initial_cycle),
        cycle_multiplicity=_readonly_array(cycle_multiplicity),
        initial_cycle_nodes=initial_cycle_nodes,
    )


def longest_false_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of false entries."""

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    best = current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def kabsch_rmsd_frames(reference: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Align every frame to one reference and return per-frame RMSD."""

    reference = np.asarray(reference, dtype=float)
    frames = np.asarray(frames, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference must have shape (points, 3)")
    if frames.ndim != 3 or frames.shape[1:] != reference.shape:
        raise ValueError("frames must have shape (frames, points, 3)")

    reference_centered = reference - reference.mean(axis=0)
    centered = frames - frames.mean(axis=1, keepdims=True)
    covariance = np.einsum("fni,nj->fij", centered, reference_centered, optimize=True)
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0.0
    if reflected.any():
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vh
    aligned = np.einsum("fni,fij->fnj", centered, rotation, optimize=True)
    return np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=2), axis=1))


def _positive_finite(value: float, *, name: str) -> float:
    """Validate a caller-supplied distance cutoff."""

    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _connected_component_count(points: NDArray[np.float64], cutoff: float) -> int:
    """Count connected components in one oxygen-distance graph."""

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
            for neighbor_value in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor_value)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def water_topology_observables(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    oxygen_connectivity_cutoff_angstrom: float | None,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> WaterTopologyObservables:
    """Calculate the shared water-cluster topology observables.

    Hydrogen ownership is assigned to the nearest oxygen in the first frame,
    then held fixed for the full trajectory.  ``None`` explicitly skips the
    oxygen connected-component calculation for callers that do not report it;
    no numerical cutoff is selected inside this function.
    """

    frames = np.asarray(positions_angstrom, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] == 0 or frames.shape[2] != 3:
        raise ValueError("positions_angstrom must have shape (frames, atoms, 3)")
    if not np.isfinite(frames).all():
        raise ValueError("positions_angstrom contains non-finite values")

    numbers = np.asarray(atomic_numbers)
    if numbers.ndim != 1 or len(numbers) != frames.shape[1]:
        raise ValueError("atomic_numbers must contain one value per atom")
    if not np.issubdtype(numbers.dtype, np.number):
        raise TypeError("atomic_numbers must be numeric")
    if not np.isfinite(numbers).all():
        raise ValueError("atomic_numbers contains non-finite values")

    h_acceptor_cutoff = _positive_finite(
        h_acceptor_cutoff_angstrom,
        name="h_acceptor_cutoff_angstrom",
    )
    oo_cutoff = _positive_finite(
        oo_cutoff_angstrom,
        name="oo_cutoff_angstrom",
    )
    hbond_angle_cutoff = float(hbond_angle_cutoff_deg)
    if not np.isfinite(hbond_angle_cutoff) or not (0.0 <= hbond_angle_cutoff <= 180.0):
        raise ValueError("hbond_angle_cutoff_deg must lie in [0, 180]")
    oxygen_connectivity_cutoff = (
        None
        if oxygen_connectivity_cutoff_angstrom is None
        else _positive_finite(
            oxygen_connectivity_cutoff_angstrom,
            name="oxygen_connectivity_cutoff_angstrom",
        )
    )

    oxygen_local = np.flatnonzero(numbers == 8)
    hydrogen_local = np.flatnonzero(numbers == 1)
    if len(oxygen_local) < 2 or len(hydrogen_local) == 0:
        raise ValueError("water topology requires a multi-water graph")

    oxygen_frames = frames[:, oxygen_local]
    hydrogen_frames = frames[:, hydrogen_local]
    assignment = np.argmin(
        np.linalg.norm(
            hydrogen_frames[0, :, None] - oxygen_frames[0, None, :],
            axis=-1,
        ),
        axis=1,
    )
    donor_oxygen = oxygen_frames[:, assignment]
    assigned_oh = np.linalg.norm(hydrogen_frames - donor_oxygen, axis=-1)

    h_to_donor = donor_oxygen - hydrogen_frames
    h_to_acceptor = oxygen_frames[:, None] - hydrogen_frames[:, :, None]
    h_acceptor_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    cosine = np.sum(h_to_donor[:, :, None] * h_to_acceptor, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine /= np.linalg.norm(h_to_donor, axis=-1)[:, :, None]
        cosine /= np.linalg.norm(h_to_acceptor, axis=-1)
    hbond_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    oo_distance = np.linalg.norm(
        donor_oxygen[:, :, None] - oxygen_frames[:, None],
        axis=-1,
    )
    is_donor = np.arange(len(oxygen_local))[None, None, :] == assignment[None, :, None]
    hbond_mask = (
        (h_acceptor_distance <= h_acceptor_cutoff)
        & (oo_distance <= oo_cutoff)
        & (hbond_angle >= hbond_angle_cutoff)
        & ~is_donor
    )

    adjacency = np.zeros(
        (frames.shape[0], len(oxygen_local), len(oxygen_local)),
        dtype=bool,
    )
    for hydrogen, donor in enumerate(assignment):
        adjacency[:, donor] |= hbond_mask[:, hydrogen]
    ring_masks = directed_ring_masks(adjacency)

    centered = oxygen_frames - oxygen_frames.mean(axis=1, keepdims=True)
    radius_gyration = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))
    rmsd = kabsch_rmsd_frames(oxygen_frames[0], oxygen_frames)
    component_count = (
        None
        if oxygen_connectivity_cutoff is None
        else np.asarray(
            [
                _connected_component_count(frame, oxygen_connectivity_cutoff)
                for frame in oxygen_frames
            ],
            dtype=np.int64,
        )
    )
    return WaterTopologyObservables(
        oxygen_count=int(len(oxygen_local)),
        hydrogen_bond_count=_readonly_array(
            np.sum(hbond_mask, axis=(1, 2), dtype=np.int64)
        ),
        ring_masks=ring_masks,
        oxygen_component_count=(
            None
            if component_count is None
            else _readonly_array(component_count)
        ),
        oxygen_radius_gyration_angstrom=_readonly_array(radius_gyration),
        oxygen_rmsd_angstrom=_readonly_array(rmsd),
        max_assigned_oh_angstrom=_readonly_array(assigned_oh.max(axis=1)),
    )


__all__ = [
    "DirectedRingMasks",
    "WaterTopologyObservables",
    "directed_ring_masks",
    "kabsch_rmsd_frames",
    "longest_false_run",
    "water_topology_observables",
]
