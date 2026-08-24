# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import module02 as helpers

_AIMNET_SPECIES = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)


def test_ncia250_loader_preserves_inventory_and_checkpoint_scope() -> None:
    structures, graph_rows, references, inventory, stats = helpers.load_ncia250(
        supported_atomic_numbers=_AIMNET_SPECIES
    )

    assert len(inventory) == stats["total_complexes"] == 250
    assert len(references) == stats["compatible_complexes"] == 205
    assert len(structures) == len(graph_rows) == stats["compatible_structures"] == 615
    assert stats["compatible_atom_rows"] == 5336
    assert stats["excluded_elements"] == ("Ar", "He", "Kr", "Ne", "Xe")
    assert graph_rows["graph_index"].tolist() == list(range(615))
    assert graph_rows["fragment"].tolist() == ["AB", "A", "B"] * 205
    assert np.isfinite(references["ccsd_t_cbs_kcal_mol"]).all()


def test_ncia250_fragments_reconstruct_each_compatible_complex() -> None:
    structures, _, _, _, _ = helpers.load_ncia250(
        supported_atomic_numbers=_AIMNET_SPECIES
    )

    for index in range(0, len(structures), 3):
        complex_atoms, fragment_a, fragment_b = structures[index : index + 3]
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
