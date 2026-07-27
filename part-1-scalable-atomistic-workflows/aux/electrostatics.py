"""Finite-system electrostatics for the composed molecular potential."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig


# AIMNet 0.2.0 uses Hartree * Bohr for a unique pair in eV Angstrom / e^2.
COULOMB_CONSTANT_EV_ANGSTROM = 14.399645351950548


class DirectCoulombWrapper(nn.Module, BaseModelMixin):
    """Differentiable all-pairs 1/r electrostatics for finite systems.

    The wrapper consumes upstream predicted charges and has no spatial
    cutoff. In an autograd ``PipelineGroup``, differentiating its energy
    includes both the fixed-charge force and the charge-response term.
    Periodic systems require a separately declared Ewald or PME boundary.
    """

    def __init__(
        self,
        coulomb_constant: float = COULOMB_CONSTANT_EV_ANGSTROM,
    ) -> None:
        super().__init__()
        self.coulomb_constant = float(coulomb_constant)
        self._cached_batch_ptr: torch.Tensor | None = None
        self._cached_batch_ptr_version: int | None = None
        self._cached_ordered_pairs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            active_outputs={"energy"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charges"}),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        del data, kwargs
        raise NotImplementedError("Direct Coulomb has no learned embeddings")

    def input_data(self) -> set[str]:
        return {"positions", "charges", "batch_idx"}

    @staticmethod
    def _ordered_pair_indices(
        node_counts: torch.Tensor,
        node_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ordered, non-self pairs within each graph of a batch."""

        node_counts = node_counts.to(dtype=torch.long)
        node_starts = node_starts.to(dtype=torch.long)
        pair_counts = node_counts * (node_counts - 1)
        pair_graph_idx = torch.repeat_interleave(
            torch.arange(node_counts.numel(), device=node_counts.device),
            pair_counts,
        )
        pair_starts = torch.cumsum(pair_counts, dim=0) - pair_counts
        local_pair_idx = torch.arange(
            pair_graph_idx.numel(),
            dtype=torch.long,
            device=node_counts.device,
        ) - torch.repeat_interleave(pair_starts, pair_counts)

        counts_for_pair = node_counts[pair_graph_idx]
        i_local = torch.div(
            local_pair_idx,
            counts_for_pair - 1,
            rounding_mode="floor",
        )
        j_local = torch.remainder(local_pair_idx, counts_for_pair - 1)
        j_local = j_local + (j_local >= i_local).to(j_local.dtype)
        starts_for_pair = node_starts[pair_graph_idx]
        return (
            starts_for_pair + i_local,
            starts_for_pair + j_local,
            pair_graph_idx,
        )

    def _batch_ordered_pair_indices(
        self,
        data: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse pair indices while a batch keeps the same graph layout."""

        batch_ptr = data.batch_ptr
        batch_ptr_version = batch_ptr._version
        if (
            self._cached_batch_ptr is batch_ptr
            and self._cached_batch_ptr_version == batch_ptr_version
            and self._cached_ordered_pairs is not None
        ):
            return self._cached_ordered_pairs

        node_counts = data.num_nodes_per_graph.reshape(-1).to(torch.long)
        node_starts = batch_ptr[:-1].reshape(-1).to(torch.long)
        pairs = self._ordered_pair_indices(node_counts, node_starts)
        self._cached_batch_ptr = batch_ptr
        self._cached_batch_ptr_version = batch_ptr_version
        self._cached_ordered_pairs = pairs
        return pairs

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> OrderedDict[str, torch.Tensor]:
        del kwargs
        positions = data.positions
        charges = data.charges.reshape(-1)
        node_count = data.num_nodes

        if isinstance(data, Batch):
            num_graphs = data.num_graphs
            i, j, pair_graph_idx = self._batch_ordered_pair_indices(data)
        else:
            node_counts = torch.tensor(
                [node_count], dtype=torch.long, device=positions.device
            )
            node_starts = torch.zeros(1, dtype=torch.long, device=positions.device)
            num_graphs = 1
            i, j, pair_graph_idx = self._ordered_pair_indices(
                node_counts,
                node_starts,
            )

        # Match AIMNet's float32 pair calculation and float64 accumulation.
        positions32 = positions.to(torch.float32)
        charges32 = charges.to(torch.float32)

        distances = torch.norm(positions32[i] - positions32[j], p=2, dim=-1)
        ordered_pair_kernel = charges32[i] * charges32[j] / distances
        energy = torch.zeros(
            num_graphs, dtype=torch.float64, device=positions.device
        )
        energy.index_add_(
            0,
            pair_graph_idx,
            ordered_pair_kernel.to(torch.float64),
        )
        energy = energy * (0.5 * self.coulomb_constant)
        return OrderedDict(energy=energy.unsqueeze(-1))

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        del path, as_state_dict
        raise NotImplementedError("This tutorial wrapper has no export format")


__all__ = ["DirectCoulombWrapper"]
