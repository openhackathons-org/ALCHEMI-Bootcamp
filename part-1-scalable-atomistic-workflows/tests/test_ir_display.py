"""Focused tests for the small learner-facing IR tables."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.experimental_reference import (  # noqa: E402
    load_experimental_water_fundamentals,
)
from aux.ir_display import (  # noqa: E402
    monomer_mode_mapping_display_table,
    prepare_monomer_reference_display,
)
from aux.reference import load_psi4_b973c_ir_artifact  # noqa: E402


def _monomer_references() -> dict[str, object]:
    root = PART_DIR / "reference" / "artifacts"
    return {
        "H2O": load_psi4_b973c_ir_artifact(root / "h2o"),
        "D2O": load_psi4_b973c_ir_artifact(root / "d2o"),
    }


def test_prepares_reference_table_and_reusable_mode_lookups() -> None:
    experimental = load_experimental_water_fundamentals()
    original = experimental.copy(deep=True)

    result = prepare_monomer_reference_display(
        _monomer_references(),
        experimental,
    )

    assert list(result.table.columns) == [
        "System",
        "Mode",
        "B97-3c harmonic (cm-1)",
        "Observed gas phase (cm-1)",
        "Harmonic - observed (cm-1)",
    ]
    assert result.table["System"].tolist() == ["H2O"] * 3 + ["D2O"] * 3
    assert result.table["Mode"].tolist() == [
        "ν1 symmetric stretch",
        "ν2 bend",
        "ν3 antisymmetric stretch",
    ] * 2
    assert result.table["B97-3c harmonic (cm-1)"].tolist() == pytest.approx(
        [3743.13, 1709.54, 3853.99, 2698.32, 1251.19, 2823.06]
    )
    assert result.table["Observed gas phase (cm-1)"].tolist() == [
        3657.1,
        1594.8,
        3755.9,
        2671.7,
        1178.4,
        2787.7,
    ]
    assert result.harmonic_mode_indices == {
        "H2O": (2, 1, 3),
        "D2O": (2, 1, 3),
    }
    assert result.observed_by_mode.loc[
        ("D2O", "antisymmetric_stretch"), "wavenumber_cm1"
    ] == pytest.approx(2787.7)
    pdt.assert_frame_equal(experimental, original)


def test_reference_table_rejects_duplicate_experimental_assignment() -> None:
    experimental = load_experimental_water_fundamentals()
    duplicated = pd.concat([experimental, experimental.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate mode rows"):
        prepare_monomer_reference_display(_monomer_references(), duplicated)


def test_monomer_mapping_table_is_concise_and_does_not_change_full_result() -> None:
    full_table = pd.DataFrame(
        {
            "system": ["cyclic hexamer", "monomer", "monomer", "monomer"],
            "character": [
                "hbonded_oh",
                "bend",
                "antisymmetric_stretch",
                "symmetric_stretch",
            ],
            "H_center_cm-1": [3300.0, 1709.539, 3853.987, 3743.135],
            "D_center_cm-1": [2450.0, 1251.194, 2823.064, 2698.325],
            "H_over_D_center": [1.347, 1.3663, 1.36517, 1.38722],
            "mapping_overlap": [0.94, 0.998765, 0.997654, 0.996543],
            "H_IR_sum_km_mol": [200.0, 65.0, 102.0, 16.0],
        }
    )
    original = full_table.copy(deep=True)

    display = monomer_mode_mapping_display_table(
        SimpleNamespace(table=full_table)
    )

    assert list(display.columns) == [
        "Mode",
        "H2O harmonic (cm-1)",
        "D2O harmonic (cm-1)",
        "H/D frequency ratio",
        "Mode match (0-1)",
    ]
    assert display["Mode"].tolist() == [
        "ν1 symmetric stretch",
        "ν2 bend",
        "ν3 antisymmetric stretch",
    ]
    np.testing.assert_allclose(
        display["H2O harmonic (cm-1)"],
        [3743.1, 1709.5, 3854.0],
    )
    np.testing.assert_allclose(
        display["Mode match (0-1)"],
        [0.9965, 0.9988, 0.9977],
    )
    pdt.assert_frame_equal(full_table, original)


@pytest.mark.parametrize("score", [-0.01, 1.01, np.nan])
def test_monomer_mapping_table_rejects_invalid_match_score(score: float) -> None:
    table = pd.DataFrame(
        {
            "system": ["monomer"] * 3,
            "character": list(("symmetric_stretch", "bend", "antisymmetric_stretch")),
            "H_center_cm-1": [3700.0, 1600.0, 3800.0],
            "D_center_cm-1": [2700.0, 1200.0, 2800.0],
            "H_over_D_center": [1.37, 1.33, 1.36],
            "mapping_overlap": [score, 0.99, 0.98],
        }
    )

    with pytest.raises(ValueError, match="non-finite|between zero and one"):
        monomer_mode_mapping_display_table(SimpleNamespace(table=table))
