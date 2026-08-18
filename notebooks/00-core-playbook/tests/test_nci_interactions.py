# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import warnings
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

from helpers import core as helpers


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
    return helpers.load_nci_atlas()


def test_nci_subset_identity_and_reference_reduction(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    atoms, graph_rows, references = nci_data

    assert helpers.NCI_ATLAS_SHA256 == (
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

    ionic_equilibrium = references[
        (references["system_id"] == "08.007") & np.isclose(references["scale"], 1.0)
    ].iloc[0]
    assert ionic_equilibrium["ccsd_t_cbs_kcal_mol"] == pytest.approx(-10.237)


def test_nci_equilibrium_references_returns_plain_rows(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, references = nci_data

    rows = helpers.nci_equilibrium_references(references)

    assert len(rows) == 3
    assert rows[0]["system_name"] == "phenol - N-methylacetamide"
    assert rows[0]["dft_d3_kcal_mol"] == pytest.approx(-11.868854)
    assert rows[2]["ccsd_t_cbs_kcal_mol"] == pytest.approx(-10.237)


def test_recurring_nci_complexes_preserve_size_formula_and_charge() -> None:
    complexes = helpers.load_recurring_nci_complexes()

    assert tuple(complexes) == (
        "Propyne–methyl azide",
        "Ammonia–benzoate",
        "Phenol–N-methylacetamide",
    )
    assert [len(structure) for structure in complexes.values()] == [14, 18, 25]
    assert [structure.get_chemical_formula() for structure in complexes.values()] == [
        "C4H7N3",
        "C7H8NO2",
        "C9H13NO2",
    ]
    assert [structure.info["charge"] for structure in complexes.values()] == [0, -1, 0]


def test_select_nci_equilibrium_complex_returns_a_copy(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    atoms, graph_rows, _ = nci_data

    selected = helpers.select_nci_equilibrium_complex(
        atoms, graph_rows, "phenol - N-methylacetamide"
    )

    assert selected.get_chemical_formula() == "C9H13NO2"
    assert selected.info["scale"] == pytest.approx(1.0)
    assert selected.info["fragment"] == "AB"
    source_index = int(
        graph_rows.loc[
            graph_rows["system_name"].eq("phenol - N-methylacetamide")
            & graph_rows["scale"].eq(1.0)
            & graph_rows["fragment"].eq("AB"),
            "graph_index",
        ].item()
    )
    source_positions = atoms[source_index].positions.copy()
    selected.positions += 1.0
    np.testing.assert_array_equal(atoms[source_index].positions, source_positions)


def test_nci_loader_rejects_changed_bytes(tmp_path: Path) -> None:
    source = _NOTEBOOK_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        helpers.load_nci_atlas(changed)


def test_interaction_components_uses_graph_index_and_ab_minus_a_minus_b(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    _, graph_rows, _ = nci_data
    graph_energies = []
    for row in graph_rows.itertuples(index=False):
        fragment_energy = {
            "A": 10.0,
            "B": 20.0,
            "AB": 30.0 + float(row.scale),
        }
        graph_energies.append(fragment_energy[row.fragment])

    curves = helpers.interaction_components(
        graph_rows.sample(frac=1.0, random_state=7),
        {"synthetic": torch.tensor(graph_energies, dtype=torch.float64)},
    )

    expected = curves["scale"].to_numpy() / (units.kcal / units.mol)
    np.testing.assert_allclose(curves["synthetic"], expected)


def test_max_system_charge_error_reduces_atom_charges() -> None:
    first = AtomicData.from_atoms(
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    )
    first.add_system_property("charge", torch.tensor([[0.0]]))
    second = AtomicData.from_atoms(Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    second.add_system_property("charge", torch.tensor([[1.0]]))
    batch = Batch.from_data_list([first, second])

    error = helpers.max_system_charge_error(
        batch,
        torch.tensor([0.25, -0.25, 0.9]),
    )

    assert error == pytest.approx(0.1)


def test_nci_curve_summary_reports_sequential_additions(
    nci_data: tuple[list[Atoms], pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, references = nci_data
    curve_keys = (
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "scale",
    )
    curves = references[list(curve_keys)].copy()
    curves["aimnet_base"] = references["dft_d3_kcal_mol"] + 1.0
    curves["base_d3"] = references["dft_d3_kcal_mol"] + 0.5
    curves["complete"] = references["dft_d3_kcal_mol"]

    summary = helpers.summarize_nci_interaction_curves(curves, references)

    assert len(summary) == 3
    assert all(row["d3_shift"] == pytest.approx(-0.5) for row in summary)
    assert all(row["electrostatic_shift"] == pytest.approx(-0.5) for row in summary)
    assert all(row["dft_mae"] == pytest.approx(0.0) for row in summary)


def test_direct_coulomb_adapter_returns_energy_and_fixed_charge_forces() -> None:
    batch = _hydrogen_batch()
    coulomb = helpers.DirectCoulombAdapter(cutoff=95.0 * units.Bohr).eval()
    compute_neighbors(batch, config=coulomb.model_config.neighbor_config)
    batch.add_key(
        "partial_charges",
        [torch.tensor([1.0, -1.0])],
        level="node",
    )

    output = coulomb(batch)

    assert "partial_charges" in batch.keys["node"]
    assert output["energy"].item() == pytest.approx(
        -helpers.COULOMB_CONSTANT_EV_ANGSTROM,
        rel=1.0e-6,
    )
    torch.testing.assert_close(
        output["forces"],
        torch.tensor(
            [
                [helpers.COULOMB_CONSTANT_EV_ANGSTROM, 0.0, 0.0],
                [-helpers.COULOMB_CONSTANT_EV_ANGSTROM, 0.0, 0.0],
            ]
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_pipeline_wiring_includes_charge_response_in_forces() -> None:
    batch = _hydrogen_batch()
    charge_model = _PositionChargeModel()
    coulomb = helpers.DirectCoulombAdapter(cutoff=95.0 * units.Bohr)
    pipeline = _composed_test_model(charge_model, coulomb)
    compute_neighbors(batch, config=pipeline.model_config.neighbor_config)

    output = pipeline(batch)

    assert set(output) >= {"energy", "forces", "charges"}
    assert output["energy"].shape == (1, 1)
    assert output["forces"].shape == (2, 3)
    assert torch.isfinite(output["forces"]).all()
    assert output["energy"].item() == pytest.approx(
        -0.25 * helpers.COULOMB_CONSTANT_EV_ANGSTROM,
        rel=1.0e-6,
    )
    assert output["forces"][0, 0].item() < 0.0
    assert output["forces"][1, 0].item() > 0.0


def test_component_benchmark_reports_scope_and_restores_model_outputs() -> None:
    atoms = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    ]
    charge_model = _PositionChargeModel()
    d3 = _QuadraticModel()
    coulomb = helpers.DirectCoulombAdapter(cutoff=95.0 * units.Bohr)
    pipeline = _composed_test_model(charge_model, coulomb, d3)
    models = [charge_model, d3, coulomb, pipeline]
    outputs_before = [set(model.model_config.active_outputs) for model in models]

    rows = helpers.benchmark_nci_components(
        atoms,
        charge_model,
        d3,
        coulomb,
        full_model=pipeline,
        device="cpu",
        graph_counts=(3,),
        warmups=1,
        repeats=2,
    )

    assert {row["component"] for row in rows} == {
        "AIMNet base",
        "D3",
        "Electrostatics",
        "Complete pipeline",
    }
    assert all(row["structures"] == 3 for row in rows)
    assert all(row["dataset_repeats"] == 1 for row in rows)
    assert all(row["atoms"] == 6 for row in rows)
    assert all(len(row["repeat_seconds"]) == 2 for row in rows)
    assert all(row["median_seconds"] > 0.0 for row in rows)
    assert all(row["hardware"] for row in rows)
    assert "stored charges" in next(
        row["timing_scope"] for row in rows if row["component"] == "Electrostatics"
    )
    assert "charge-response autograd" in next(
        row["timing_scope"] for row in rows if row["component"] == "Complete pipeline"
    )
    assert [
        set(model.model_config.active_outputs) for model in models
    ] == outputs_before


def test_d3_parameter_file_uses_the_verified_runtime_asset() -> None:
    path = helpers.d3_parameter_file()

    assert path.is_file()
    assert path.name == "dftd3_parameters.pt"


def test_timing_plot_uses_four_measured_calls_without_layout_warnings() -> None:
    rows = []
    for structures, atoms in ((90, 1140), (360, 4560), (1440, 18240)):
        for index, component in enumerate(
            ("AIMNet base", "D3", "Electrostatics", "Complete pipeline"),
            start=1,
        ):
            rows.append(
                {
                    "structures": structures,
                    "atoms": atoms,
                    "component": component,
                    "median_seconds": structures * index / 100_000,
                    "hardware": "Test GPU",
                }
            )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = helpers.plot_nci_component_timings(rows)

    assert result is None
    assert not [item for item in caught if "tight_layout" in str(item.message)]
