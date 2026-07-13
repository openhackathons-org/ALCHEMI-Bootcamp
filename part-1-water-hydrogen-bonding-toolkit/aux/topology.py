"""NumPy-only topology analysis for the water IR trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


@dataclass(frozen=True)
class DirectedRingMasks:
    """Per-frame directed-cycle classifications for one molecular graph."""

    any_cycle: np.ndarray
    exact_single_ring: np.ndarray
    initial_cycle: np.ndarray
    cycle_multiplicity: np.ndarray
    initial_cycle_nodes: tuple[int, ...] | None


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
        any_cycle
        & (donor_counts == 1).all(axis=1)
        & (acceptor_counts == 1).all(axis=1)
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
        any_cycle=any_cycle,
        exact_single_ring=exact_single_ring,
        initial_cycle=initial_cycle,
        cycle_multiplicity=cycle_multiplicity,
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
    covariance = np.einsum(
        "fni,nj->fij", centered, reference_centered, optimize=True
    )
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0.0
    if reflected.any():
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vh
    aligned = np.einsum("fni,fij->fnj", centered, rotation, optimize=True)
    return np.sqrt(
        np.mean(np.sum((aligned - reference_centered) ** 2, axis=2), axis=1)
    )


__all__ = [
    "DirectedRingMasks",
    "directed_ring_masks",
    "kabsch_rmsd_frames",
    "longest_false_run",
]
