"""Small model wrappers used by the two training examples."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.training import create_model_spec

from .data import make_loader, toy_records


class ToyTransferMLP(torch.nn.Module, BaseModelMixin):
    """Fixed-four-atom MLP with a separately selectable readout."""

    def __init__(self, hidden_features: int = 16) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(12, hidden_features),
            torch.nn.Tanh(),
        )
        self.readout = torch.nn.Linear(hidden_features, 1)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"positions"}),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self,
        data: AtomicData | Batch,
        **kwargs: Any,
    ) -> AtomicData | Batch:
        del kwargs
        return data

    def forward(
        self,
        data: AtomicData | Batch,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        graph_count = data.num_graphs if isinstance(data, Batch) else 1
        features = data.positions.reshape(graph_count, 12)
        return {"energy": self.readout(self.backbone(features))}


def prepare_toy_transfer(
    *,
    device: torch.device,
) -> tuple[ToyTransferMLP, Any, Any, pd.DataFrame]:
    """Pretrain a tiny source task and return a shifted held-out task."""

    torch.manual_seed(71)
    model = ToyTransferMLP().to(device)
    source = toy_records(
        count=24,
        seed=101,
        target_shift=0.0,
        device=device,
    )
    source_loader = make_loader(source, batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-2)
    model.train()
    for _ in range(60):
        for batch in source_loader:
            prediction = model(batch)["energy"]
            loss = torch.nn.functional.mse_loss(prediction, batch.energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    target = toy_records(
        count=16,
        seed=303,
        target_shift=0.8,
        device=device,
    )
    training = target[:12]
    validation = target[12:]
    rows = [
        {
            "sample_id": int(record.sample_id.item()),
            "split": "train" if index < 12 else "validation",
            "target": float(record.energy.detach().cpu().item()),
            "unit": "dimensionless synthetic score",
        }
        for index, record in enumerate(target)
    ]
    return (
        model,
        make_loader(training, batch_size=4),
        make_loader(validation, batch_size=4),
        pd.DataFrame(rows),
    )


class TrainableLennardJones(torch.nn.Module, BaseModelMixin):
    """Differentiable two-parameter LJ wrapper over Toolkit COO neighbors."""

    def __init__(
        self,
        *,
        epsilon_eV: float,
        sigma_A: float,
        cutoff_A: float,
    ) -> None:
        super().__init__()
        if epsilon_eV <= 0 or sigma_A <= 0 or cutoff_A <= 0:
            raise ValueError("epsilon_eV, sigma_A, and cutoff_A must be positive")
        self._initial_epsilon_eV = float(epsilon_eV)
        self._initial_sigma_A = float(sigma_A)
        self.cutoff_A = float(cutoff_A)
        self.log_epsilon = torch.nn.Parameter(
            torch.tensor(math.log(epsilon_eV), dtype=torch.float64)
        )
        self.log_sigma = torch.nn.Parameter(
            torch.tensor(math.log(sigma_A), dtype=torch.float64)
        )
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            required_inputs=frozenset({"neighbor_list"}),
            optional_inputs=frozenset({"neighbor_list_shifts", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff_A,
                format=NeighborListFormat.COO,
                half_list=True,
                skin=0.0,
            ),
        )

    @property
    def epsilon_eV(self) -> torch.Tensor:
        """Positive well depth in eV."""

        return self.log_epsilon.exp()

    @property
    def sigma_A(self) -> torch.Tensor:
        """Positive zero-crossing distance in Å."""

        return self.log_sigma.exp()

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self,
        data: AtomicData | Batch,
        **kwargs: Any,
    ) -> AtomicData | Batch:
        del kwargs
        return data

    def checkpoint_spec(self) -> Any:
        """Return the JSON-serializable constructor recipe for restart."""

        return create_model_spec(
            type(self),
            epsilon_eV=self._initial_epsilon_eV,
            sigma_A=self._initial_sigma_A,
            cutoff_A=self.cutoff_A,
        )

    def _edge_displacements(
        self,
        data: Batch,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        displacement = data.positions[target] - data.positions[source]
        shifts = getattr(data, "neighbor_list_shifts", None)
        if shifts is None:
            return displacement
        cells = getattr(data, "cell", None)
        if cells is None:
            raise ValueError("neighbor_list_shifts require a cell tensor")
        edge_cells = cells.index_select(0, data.batch_idx[source].long())
        cartesian_shifts = torch.einsum(
            "ei,eij->ej",
            shifts.to(dtype=data.positions.dtype),
            edge_cells,
        )
        return displacement + cartesian_shifts

    def forward(
        self,
        data: AtomicData | Batch,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if not isinstance(data, Batch):
            raise TypeError(
                "TrainableLennardJones requires Batch; "
                "wrap one structure with Batch.from_data_list."
            )
        edges = getattr(data, "neighbor_list", None)
        if edges is None:
            raise KeyError("neighbor_list is required; call compute_neighbors first")
        source = edges[:, 0].long()
        target = edges[:, 1].long()
        displacement = self._edge_displacements(data, source, target)
        squared_distance = displacement.square().sum(dim=1)
        if bool(squared_distance.le(0).any()):
            raise ValueError("LJ neighbors must have nonzero separation")

        sigma_over_r_squared = self.sigma_A.square() / squared_distance
        sigma_over_r_six = sigma_over_r_squared.pow(3)
        sigma_over_r_twelve = sigma_over_r_six.square()
        pair_energy = 4.0 * self.epsilon_eV * (sigma_over_r_twelve - sigma_over_r_six)
        force_scale = (
            24.0
            * self.epsilon_eV
            * (sigma_over_r_six - 2.0 * sigma_over_r_twelve)
            / squared_distance
        )
        pair_force = force_scale.unsqueeze(1) * displacement

        edge_graph = data.batch_idx[source].long()
        energy = data.positions.new_zeros(data.num_graphs).index_add(
            0,
            edge_graph,
            pair_energy,
        )
        forces = torch.zeros_like(data.positions)
        forces = forces.index_add(0, source, pair_force)
        forces = forces.index_add(0, target, -pair_force)
        return {
            "energy": energy.unsqueeze(1),
            "forces": forces,
        }
