# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from ase import Atoms, units
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import PipelineGroup, PipelineModelWrapper, PipelineStep
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import compute_neighbors

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import model_support, nci


class _PositionChargeModel(torch.nn.Module, BaseModelMixin):
    """Small differentiable charge model for the composition tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "charges"}),
            active_outputs={"energy", "forces", "charges"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        del data, kwargs
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> Any:
        del kwargs
        inputs = self.adapt_input(data)
        positions = inputs["positions"]
        charges = positions[:, 0] - 0.5
        energy = torch.zeros(
            data.num_graphs,
            dtype=positions.dtype,
            device=positions.device,
        ).index_add_(0, data.batch_idx.to(torch.long), 0.0 * positions[:, 0])
        output = {"energy": energy.unsqueeze(-1), "charges": charges}
        if "forces" in self.model_config.active_outputs:
            output["forces"] = -torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        return self.adapt_output(output, data)


class _QuadraticModel(torch.nn.Module, BaseModelMixin):
    """Analytical energy and forces for the timing smoke test."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        del data, kwargs
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> Any:
        del kwargs
        per_atom = 0.5 * data.positions.square().sum(dim=1)
        energy = torch.zeros(
            data.num_graphs,
            dtype=data.positions.dtype,
            device=data.positions.device,
        ).index_add_(0, data.batch_idx.to(torch.long), per_atom)
        return self.adapt_output(
            {"energy": energy.unsqueeze(-1), "forces": -data.positions},
            data,
        )


def _hydrogen_batch() -> Batch:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return Batch.from_data_list([AtomicData.from_atoms(atoms)])


def _composed_test_model(
    charge_model: BaseModelMixin,
    coulomb: BaseModelMixin,
    d3: BaseModelMixin | None = None,
) -> PipelineModelWrapper:
    groups = [
        PipelineGroup(
            steps=[
                PipelineStep(
                    charge_model,
                    wire={"charges": "partial_charges"},
                ),
                PipelineStep(coulomb),
            ],
            use_autograd=True,
        )
    ]
    if d3 is not None:
        groups.append(PipelineGroup(steps=[PipelineStep(d3)], use_autograd=False))
    model = PipelineModelWrapper(groups=groups, neighbor_adaptation="always").eval()
    model.set_config("active_outputs", {"energy", "forces"})
    return model


@pytest.fixture(scope="module")
def nci_data() -> tuple[list[Atoms], pd.DataFrame, pd.DataFrame]:
    return nci.load_nci_atlas()


def test_nci_subset_identity_and_reference_reduction(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    atoms, graph_rows, references = nci_data

    assert nci.NCI_ATLAS_SHA256 == (
        "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
    )
    assert len(atoms) == len(graph_rows) == 90
    assert len(references) == 30
    assert tuple(references["system_name"].drop_duplicates()) == (
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    )
    assert all(not structure.pbc.any() for structure in atoms)
    assert graph_rows["graph_index"].tolist() == list(range(90))
    assert graph_rows["fragment"].tolist() == ["AB", "A", "B"] * 30

    ionic_equilibrium = references[
        (references["system_id"] == "08.007") & np.isclose(references["scale"], 1.0)
    ].iloc[0]
    assert ionic_equilibrium["ccsd_t_cbs_kcal_mol"] == pytest.approx(-10.237)


def test_checked_nci_curves_align_saved_models_with_current_references(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, references = nci_data

    curves = nci.checked_nci_interaction_curves(references)

    assert len(curves) == 30
    assert tuple(curves["system_name"].drop_duplicates()) == (
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    )
    equilibrium = curves[np.isclose(curves["scale"], 1.0)]
    assert equilibrium.iloc[0]["d3_increment"] == pytest.approx(-3.2267218)
    assert equilibrium.iloc[1]["electrostatic_increment"] == pytest.approx(-2.2557594)


def test_nci_loader_rejects_changed_bytes(tmp_path: Path) -> None:
    source = _NOTEBOOK_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        nci.load_nci_atlas(changed)


def test_direct_coulomb_adapter_returns_energy_and_fixed_charge_forces() -> None:
    batch = _hydrogen_batch()
    coulomb = model_support.DirectCoulombAdapter(cutoff=95.0 * units.Bohr).eval()
    compute_neighbors(batch, config=coulomb.model_config.neighbor_config)
    batch.add_key(
        "partial_charges",
        [torch.tensor([1.0, -1.0])],
        level="node",
    )

    output = coulomb(batch)

    assert "partial_charges" in batch.keys["node"]
    assert output["energy"].item() == pytest.approx(
        -model_support.COULOMB_CONSTANT_EV_ANGSTROM,
        rel=1.0e-6,
    )
    torch.testing.assert_close(
        output["forces"],
        torch.tensor(
            [
                [model_support.COULOMB_CONSTANT_EV_ANGSTROM, 0.0, 0.0],
                [-model_support.COULOMB_CONSTANT_EV_ANGSTROM, 0.0, 0.0],
            ]
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_pipeline_wiring_includes_charge_response_in_forces() -> None:
    batch = _hydrogen_batch()
    charge_model = _PositionChargeModel()
    coulomb = model_support.DirectCoulombAdapter(cutoff=95.0 * units.Bohr)
    pipeline = _composed_test_model(charge_model, coulomb)
    compute_neighbors(batch, config=pipeline.model_config.neighbor_config)

    output = pipeline(batch)

    assert set(output) >= {"energy", "forces", "charges"}
    assert output["energy"].shape == (1, 1)
    assert output["forces"].shape == (2, 3)
    assert torch.isfinite(output["forces"]).all()
    assert output["energy"].item() == pytest.approx(
        -0.25 * model_support.COULOMB_CONSTANT_EV_ANGSTROM,
        rel=1.0e-6,
    )
    assert output["forces"][0, 0].item() < 0.0
    assert output["forces"][1, 0].item() > 0.0


def test_d3_parameter_file_uses_the_verified_runtime_asset() -> None:
    path = model_support.d3_parameter_file()

    assert path.is_file()
    assert path.name == "dftd3_parameters.pt"
