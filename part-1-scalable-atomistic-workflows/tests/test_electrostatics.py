"""Scientific and batching checks for tutorial-local electrostatics."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from nvalchemi.data import AtomicData, Batch  # noqa: E402

from aux.electrostatics import (  # noqa: E402
    COULOMB_CONSTANT_EV_ANGSTROM,
    DirectCoulombWrapper,
)


def _graph(positions: list[list[float]], charges: list[float]) -> AtomicData:
    data = AtomicData(
        positions=torch.tensor(positions, dtype=torch.float64),
        atomic_numbers=torch.ones(len(positions), dtype=torch.long),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    data.add_node_property(
        "charges",
        torch.tensor(charges, dtype=torch.float64),
    )
    return data


def _legacy_dense_energy(batch: Batch) -> OrderedDict[str, torch.Tensor]:
    """Small-batch reference matching the original dense implementation."""

    positions32 = batch.positions.to(torch.float32)
    charges32 = batch.charges.reshape(-1).to(torch.float32)
    graph_idx = batch.batch_idx.to(torch.long)
    indices = torch.arange(batch.num_nodes, device=batch.device)
    i = indices.repeat_interleave(batch.num_nodes)
    j = indices.repeat(batch.num_nodes)
    same_graph_pair = (i != j) & (graph_idx[i] == graph_idx[j])
    i = i[same_graph_pair]
    j = j[same_graph_pair]
    kernel = charges32[i] * charges32[j] / torch.linalg.vector_norm(
        positions32[i] - positions32[j], dim=-1
    )
    energy = torch.zeros(
        batch.num_graphs,
        dtype=torch.float64,
        device=batch.device,
    )
    energy.index_add_(0, graph_idx[i], kernel.to(torch.float64))
    return OrderedDict(
        energy=(0.5 * COULOMB_CONSTANT_EV_ANGSTROM * energy).unsqueeze(-1)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_ragged_pairs_match_dense_energy_and_gradients(device: str) -> None:
    batch = Batch.from_data_list(
        [
            _graph([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], [0.4, -0.4]),
            _graph(
                [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 1.1, 0.0]],
                [0.3, -0.2, -0.1],
            ),
        ],
        device=device,
    )
    batch.positions.requires_grad_(True)
    batch.charges.requires_grad_(True)

    reference = _legacy_dense_energy(batch)["energy"]
    expected_gradients = torch.autograd.grad(
        reference.sum(),
        (batch.positions, batch.charges),
        retain_graph=True,
    )
    actual = DirectCoulombWrapper()(batch)["energy"]
    actual_gradients = torch.autograd.grad(
        actual.sum(),
        (batch.positions, batch.charges),
    )

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_gradients[0], expected_gradients[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual_gradients[1], expected_gradients[1], rtol=0.0, atol=0.0
    )


def test_pair_builder_scales_with_pairs_inside_each_graph() -> None:
    node_counts = torch.tensor([2, 3, 18, 1], dtype=torch.long)
    node_starts = torch.tensor([0, 2, 5, 23], dtype=torch.long)

    i, j, pair_graph_idx = DirectCoulombWrapper._ordered_pair_indices(
        node_counts,
        node_starts,
    )

    assert i.numel() == int((node_counts * (node_counts - 1)).sum())
    assert j.numel() == i.numel()
    assert pair_graph_idx.numel() == i.numel()
    assert not bool((i == j).any())
    for graph, (start, count) in enumerate(zip(node_starts, node_counts, strict=True)):
        mask = pair_graph_idx == graph
        assert bool((i[mask] >= start).all() and (i[mask] < start + count).all())
        assert bool((j[mask] >= start).all() and (j[mask] < start + count).all())


def test_pair_builder_preserves_graph_major_order() -> None:
    i, j, graph = DirectCoulombWrapper._ordered_pair_indices(
        torch.tensor([2, 3, 1]),
        torch.tensor([0, 2, 5]),
    )

    assert list(zip(i.tolist(), j.tolist(), graph.tolist(), strict=True)) == [
        (0, 1, 0),
        (1, 0, 0),
        (2, 3, 1),
        (2, 4, 1),
        (3, 2, 1),
        (3, 4, 1),
        (4, 2, 1),
        (4, 3, 1),
    ]


def test_two_charge_atomic_data_matches_analytic_energy() -> None:
    distance_a = 2.0
    data = _graph(
        [[0.0, 0.0, 0.0], [distance_a, 0.0, 0.0]],
        [1.0, -1.0],
    )

    energy = DirectCoulombWrapper()(data)["energy"]

    expected = -COULOMB_CONSTANT_EV_ANGSTROM / distance_a
    torch.testing.assert_close(
        energy,
        torch.tensor([[expected]], dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_packed_graphs_equal_individual_calls() -> None:
    graphs = [
        _graph([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], [0.4, -0.4]),
        _graph(
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 1.1, 0.0]],
            [0.3, -0.2, -0.1],
        ),
    ]
    wrapper = DirectCoulombWrapper()
    packed = Batch.from_data_list(graphs, device="cpu")

    packed_energy = wrapper(packed)["energy"]
    individual_energy = torch.cat(
        [wrapper(graph)["energy"] for graph in graphs],
        dim=0,
    )

    torch.testing.assert_close(packed_energy, individual_energy, rtol=0.0, atol=0.0)


def test_packed_pair_indices_are_reused_until_layout_changes() -> None:
    wrapper = DirectCoulombWrapper()
    first = Batch.from_data_list(
        [
            _graph([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], [0.4, -0.4]),
            _graph([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], [0.3, -0.3]),
        ],
        device="cpu",
    )
    second = Batch.from_data_list(
        [
            _graph(
                [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 1.1, 0.0]],
                [0.3, -0.2, -0.1],
            ),
        ],
        device="cpu",
    )

    wrapper(first)
    first_pairs = wrapper._cached_ordered_pairs
    assert first_pairs is not None

    wrapper(first)
    assert wrapper._cached_ordered_pairs is first_pairs

    wrapper(second)
    second_pairs = wrapper._cached_ordered_pairs
    assert second_pairs is not None
    assert second_pairs is not first_pairs

    wrapper(second)
    assert wrapper._cached_ordered_pairs is second_pairs
