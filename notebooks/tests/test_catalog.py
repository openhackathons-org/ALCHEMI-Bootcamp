# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))

from helpers.catalog import (
    mace_checkpoint_path,
    opening_performance_catalog,
    structure_from_smiles,
)


def test_catalog_returns_fresh_structures() -> None:
    ibuprofen = opening_performance_catalog()["ibuprofen"]
    first = ibuprofen.build()
    second = ibuprofen.build()
    first.positions[0, 0] += 1.0

    assert not np.array_equal(first.positions, second.positions)


@pytest.mark.parametrize(
    ("key", "anchor"),
    [("co_pt111_4x4", "C"), ("h2o_cu111_4x4", "O")],
)
def test_opening_surface_adsorbates_are_centered(key: str, anchor: str) -> None:
    structure = opening_performance_catalog()[key].build()
    anchor_index = structure.get_chemical_symbols().index(anchor)

    np.testing.assert_allclose(
        structure.get_scaled_positions(wrap=True)[anchor_index, :2],
        [0.5, 0.5],
        atol=1e-12,
    )


def test_smiles_preparation_is_deterministic() -> None:
    first = structure_from_smiles("CCO")
    second = structure_from_smiles("CCO")

    assert first.get_chemical_formula(mode="hill") == "C2H6O"
    np.testing.assert_array_equal(first.numbers, second.numbers)
    np.testing.assert_allclose(first.positions, second.positions, rtol=0.0, atol=0.0)
    assert first.info["embedding"] == "ETKDGv3"
    assert first.info["embedding_seed"] == 0xC0FFEE
    assert first.info["force_field"] in {"MMFF94", "UFF"}


@pytest.mark.parametrize("smiles", ["", "this is not SMILES", "[NH4+]"])
def test_smiles_preparation_errors_are_clear(smiles: str) -> None:
    with pytest.raises(ValueError, match="SMILES|neutral"):
        structure_from_smiles(smiles)


def test_mace_checkpoint_helper_rejects_unpinned_alias() -> None:
    with pytest.raises(ValueError, match="pins 'medium-mpa-0'"):
        mace_checkpoint_path("medium-0b2")
