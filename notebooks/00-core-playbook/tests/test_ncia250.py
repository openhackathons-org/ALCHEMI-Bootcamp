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
from ase import Atoms
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import core as helpers

_AIMNET_SPECIES = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)


class _QuadraticModel(torch.nn.Module, BaseModelMixin):
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


@pytest.fixture(scope="module")
def ncia250_data() -> tuple[
    list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    return helpers.load_ncia250(supported_atomic_numbers=_AIMNET_SPECIES)


def test_ncia250_identity_inventory_and_checkpoint_subset(
    ncia250_data: tuple[
        list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
    ],
) -> None:
    atoms, graph_rows, references, inventory, stats = ncia250_data

    assert helpers.NCIA250_SHA256 == (
        "34e3c2cec763344dd9be41aa008672c7d052e50db57abe1abc59873d3935c433"
    )
    assert len(inventory) == 250
    assert len(atoms) == len(graph_rows) == 615
    assert len(references) == 205
    assert graph_rows["graph_index"].tolist() == list(range(615))
    assert graph_rows["fragment"].tolist() == ["AB", "A", "B"] * 205
    assert stats == {
        "total_complexes": 250,
        "compatible_complexes": 205,
        "excluded_complexes": 45,
        "compatible_graphs": 615,
        "total_atom_rows": 5962,
        "compatible_atom_rows": 5336,
        "dimer_atoms_min": 2,
        "dimer_atoms_median": 11.0,
        "dimer_atoms_max": 33,
        "compatible_dimer_atoms_min": 4,
        "compatible_dimer_atoms_median": 12.0,
        "compatible_dimer_atoms_max": 33,
        "source_counts": {
            "D1200": 50,
            "HB300SPXx10": 50,
            "HB375x10": 50,
            "R739x5": 50,
            "SH250x10": 50,
        },
        "compatible_source_counts": {
            "D1200": 30,
            "HB300SPXx10": 50,
            "HB375x10": 50,
            "R739x5": 25,
            "SH250x10": 50,
        },
        "element_counts": {
            "Ar": 9,
            "As": 5,
            "B": 21,
            "Br": 45,
            "C": 718,
            "Cl": 53,
            "F": 122,
            "H": 1479,
            "He": 10,
            "I": 38,
            "Kr": 7,
            "N": 152,
            "Ne": 17,
            "O": 175,
            "P": 51,
            "S": 67,
            "Se": 4,
            "Xe": 8,
        },
        "supported_elements": (
            "H",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "As",
            "Se",
            "Br",
            "I",
        ),
        "excluded_elements": ("He", "Ne", "Ar", "Kr", "Xe"),
    }
    assert inventory["model_compatible"].sum() == 205
    assert set(inventory.loc[~inventory["model_compatible"], "charge"]) == {0}
    assert np.isfinite(references["ccsd_t_cbs_kcal_mol"]).all()


def test_ncia250_fragments_reconstruct_each_compatible_complex(
    ncia250_data: tuple[
        list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
    ],
) -> None:
    atoms, _, _, _, _ = ncia250_data

    for index in range(0, len(atoms), 3):
        complex_atoms, fragment_a, fragment_b = atoms[index : index + 3]
        assert len(complex_atoms) == len(fragment_a) + len(fragment_b)
        np.testing.assert_array_equal(
            complex_atoms.numbers,
            np.concatenate((fragment_a.numbers, fragment_b.numbers)),
        )
        np.testing.assert_allclose(
            complex_atoms.positions,
            np.vstack((fragment_a.positions, fragment_b.positions)),
            rtol=0.0,
            atol=0.0,
        )
        assert complex_atoms.info["charge"] == (
            fragment_a.info["charge"] + fragment_b.info["charge"]
        )


def test_ncia250_loader_rejects_changed_bytes(tmp_path: Path) -> None:
    source = _NOTEBOOK_DIR / "data" / "nci_atlas" / "NCIA250.zip"
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        helpers.load_ncia250(
            changed,
            supported_atomic_numbers=_AIMNET_SPECIES,
        )


def test_aimnet_species_come_from_verified_checkpoint() -> None:
    assert helpers.aimnet_checkpoint_species() == _AIMNET_SPECIES


def test_interaction_accuracy_summary_reports_named_error_measures() -> None:
    keys = {
        "subset": ["test", "test", "test"],
        "system_id": ["a", "b", "c"],
        "system_name": ["A", "B", "C"],
        "interaction_class": ["x", "x", "x"],
        "scale": [1.0, 1.0, 1.0],
    }
    curves = pd.DataFrame(
        {
            **keys,
            "aimnet_base": [2.0, 2.0, 4.0],
            "complete": [0.1, 0.4, 0.8],
        }
    )
    references = pd.DataFrame({**keys, "ccsd_t_cbs_kcal_mol": [0.0, 0.0, 0.0]})

    summary = helpers.interaction_accuracy_summary(
        curves,
        references,
        components=("aimnet_base", "complete"),
        tolerance_kcal_mol=0.5,
    ).set_index("component")

    assert summary.loc["aimnet_base", "mae_kcal_mol"] == pytest.approx(8.0 / 3.0)
    assert summary.loc["complete", "median_absolute_error_kcal_mol"] == 0.4
    assert summary.loc["complete", "p90_absolute_error_kcal_mol"] == pytest.approx(
        0.72
    )
    assert summary.loc["complete", "fraction_within_tolerance"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary.loc["complete", "best_for_complexes"] == 3


def _three_complex_graphs() -> tuple[list[AtomicData], pd.DataFrame, Batch]:
    structures = []
    records = []
    for complex_index, distance in enumerate((1.0, 1.4, 1.8)):
        triplet = (
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[distance, 0.0, 0.0]]),
        )
        for fragment, atoms in zip(("AB", "A", "B"), triplet, strict=True):
            records.append(
                {
                    "graph_index": len(structures),
                    "subset": "test",
                    "system_id": str(complex_index),
                    "system_name": f"complex {complex_index}",
                    "interaction_class": "test",
                    "scale": 1.0,
                    "fragment": fragment,
                }
            )
            structures.append(AtomicData.from_atoms(atoms))
    return structures, pd.DataFrame(records), Batch.from_data_list(structures)


def test_batching_benchmark_reports_cpu_sample_and_correctness() -> None:
    graphs, graph_rows, batch = _three_complex_graphs()
    model = _QuadraticModel()

    results = helpers.benchmark_interaction_batching(
        graphs,
        graph_rows,
        batch,
        model,
        warmups=0,
        repeats=2,
        cpu_complexes=2,
    )

    assert results["dataset_complexes"] == 3
    assert results["benchmarked_complexes"] == 2
    assert results["full_dataset"] is False
    assert results["structures"] == 6
    assert results["atoms"] == 8
    assert results["device_label"] == "CPU"
    assert results["speedup"] > 0.0
    assert results["correctness"]["max_energy_delta_ev"] == pytest.approx(0.0)
    assert results["correctness"][
        "max_force_delta_ev_per_angstrom"
    ] == pytest.approx(0.0)
    assert {row["model_calls"] for row in results["routes"]} == {1, 2}
    assert all(len(row["repeat_seconds"]) == 2 for row in results["routes"])


def test_ncia250_plots_render_without_layout_warnings(
    ncia250_data: tuple[
        list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
    ],
) -> None:
    _, _, _, inventory, _ = ncia250_data
    accuracy = pd.DataFrame(
        {
            "component": (
                "aimnet_base",
                "base_d3",
                "complete",
            ),
            "mae_kcal_mol": (8.0, 8.5, 1.5),
        }
    )
    batching = {
        "routes": [
            {
                "route": "serial_triplets",
                "label": "205 × Batch[3]",
                "median_seconds": 2.4,
                "structures": 615,
                "atoms": 5336,
            },
            {
                "route": "combined_batch",
                "label": "1 × Batch[615]",
                "median_seconds": 0.025,
                "structures": 615,
                "atoms": 5336,
            },
        ],
        "speedup": 96.0,
        "hardware": "Test GPU",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        helpers.plot_ncia250_survey(inventory, accuracy)
        helpers.plot_interaction_batching(batching)

    assert not [item for item in caught if "tight_layout" in str(item.message)]
