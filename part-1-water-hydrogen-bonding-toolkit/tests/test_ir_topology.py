from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.topology import (
    directed_ring_masks,
    kabsch_rmsd_frames,
    longest_false_run,
)


def test_cycle_with_extra_edge_is_not_misclassified_as_lost() -> None:
    adjacency = np.zeros((3, 6, 6), dtype=bool)
    ring = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
    for donor, acceptor in ring:
        adjacency[:, donor, acceptor] = True
    adjacency[1, 1, 4] = True
    adjacency[2, 5, 0] = False
    adjacency[2, 4, 0] = True
    adjacency[2, 5, 1] = True

    result = directed_ring_masks(adjacency)

    np.testing.assert_array_equal(result.any_cycle, [True, True, False])
    np.testing.assert_array_equal(result.initial_cycle, [True, True, False])
    np.testing.assert_array_equal(result.exact_single_ring, [True, False, False])
    assert result.initial_cycle_nodes == (0, 1, 2, 3, 4, 5)


def test_longest_false_run() -> None:
    assert longest_false_run(np.array([True, False, False, True, False])) == 2


def test_batched_kabsch_removes_rigid_motion() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rigid = reference @ rotation + np.array([4.0, -3.0, 2.0])
    distorted = rigid.copy()
    distorted[1, 2] += 0.4

    rmsd = kabsch_rmsd_frames(reference, np.stack([rigid, distorted]))

    assert rmsd[0] < 1.0e-12
    assert rmsd[1] > 0.02
