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

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> OrderedDict[str, torch.Tensor]:
        del kwargs
        positions = data.positions
        charges = data.charges.reshape(-1)
        node_count = data.num_nodes

        if isinstance(data, Batch):
            graph_idx = data.batch_idx.to(torch.long)
            num_graphs = data.num_graphs
        else:
            graph_idx = torch.zeros(
                node_count, dtype=torch.long, device=positions.device
            )
            num_graphs = 1

        # Match AIMNet's float32 pair calculation and float64 accumulation.
        positions32 = positions.to(torch.float32)
        charges32 = charges.to(torch.float32)
        indices = torch.arange(node_count, device=positions.device)
        i = indices.repeat_interleave(node_count)
        j = indices.repeat(node_count)
        same_graph_pair = (i != j) & (graph_idx[i] == graph_idx[j])
        i = i[same_graph_pair]
        j = j[same_graph_pair]

        distances = torch.norm(positions32[i] - positions32[j], p=2, dim=-1)
        ordered_pair_kernel = charges32[i] * charges32[j] / distances
        energy = torch.zeros(
            num_graphs, dtype=torch.float64, device=positions.device
        )
        energy.index_add_(
            0,
            graph_idx[i],
            ordered_pair_kernel.to(torch.float64),
        )
        energy = energy * (0.5 * self.coulomb_constant)
        return OrderedDict(energy=energy.unsqueeze(-1))

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        del path, as_state_dict
        raise NotImplementedError("This tutorial wrapper has no export format")


__all__ = ["DirectCoulombWrapper"]
