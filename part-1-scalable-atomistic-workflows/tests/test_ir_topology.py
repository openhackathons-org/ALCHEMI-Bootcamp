from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.topology import (  # noqa: E402
    directed_ring_masks,
    kabsch_rmsd_frames,
    longest_false_run,
    water_topology_observables,
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
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rigid = reference @ rotation + np.array([4.0, -3.0, 2.0])
    distorted = rigid.copy()
    distorted[1, 2] += 0.4

    rmsd = kabsch_rmsd_frames(reference, np.stack([rigid, distorted]))

    assert rmsd[0] < 1.0e-12
    assert rmsd[1] > 0.02


def _idealized_water_ring() -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(6, dtype=float) * 2.0 * np.pi / 6.0
    oxygen = 2.8 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    next_oxygen = np.roll(oxygen, -1, axis=0)
    donor_direction = next_oxygen - oxygen
    donor_direction /= np.linalg.norm(donor_direction, axis=1, keepdims=True)
    donor_hydrogen = oxygen + 0.96 * donor_direction
    spectator_hydrogen = oxygen + np.array([0.0, 0.0, 0.96])
    positions = np.concatenate((oxygen, donor_hydrogen, spectator_hydrogen))
    numbers = np.concatenate(
        (
            np.full(6, 8, dtype=np.int64),
            np.full(12, 1, dtype=np.int64),
        )
    )
    frames = np.stack((positions, positions + np.array([4.0, -3.0, 2.0])))
    return frames, numbers


def test_water_topology_calculation_owns_shared_frame_observables() -> None:
    frames, numbers = _idealized_water_ring()

    topology = water_topology_observables(
        frames,
        numbers,
        oxygen_connectivity_cutoff_angstrom=4.0,
        h_acceptor_cutoff_angstrom=2.5,
        oo_cutoff_angstrom=3.5,
        hbond_angle_cutoff_deg=140.0,
    )

    assert topology.oxygen_count == 6
    np.testing.assert_array_equal(topology.hydrogen_bond_count, [6, 6])
    np.testing.assert_array_equal(topology.ring_masks.exact_single_ring, [True, True])
    np.testing.assert_array_equal(topology.oxygen_component_count, [1, 1])
    np.testing.assert_allclose(topology.oxygen_radius_gyration_angstrom, [2.8, 2.8])
    np.testing.assert_allclose(topology.oxygen_rmsd_angstrom, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(topology.max_assigned_oh_angstrom, 0.96)


def test_water_topology_has_no_scientific_cutoff_defaults() -> None:
    signature = inspect.signature(water_topology_observables)

    for name in (
        "oxygen_connectivity_cutoff_angstrom",
        "h_acceptor_cutoff_angstrom",
        "oo_cutoff_angstrom",
        "hbond_angle_cutoff_deg",
    ):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
