"""Tests for compact periodic-domain display tables."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain.display import (  # noqa: E402
    compact_box_summary_table,
    molecule_charge_display_tables,
)


def test_compact_box_summary_keeps_raw_details_unchanged() -> None:
    details = pd.DataFrame(
        [
            {
                "component": "phenol",
                "molecules": 128,
                "atoms_per_molecule": 13,
            },
            {
                "component": "N-methylacetamide",
                "molecules": 128,
                "atoms_per_molecule": 12,
            },
            {
                "component": "periodic box",
                "molecules": 256,
                "atoms_total": 3200,
                "net_charge_e": 0,
                "box_length_a": 49.1256,
                "construction_density_g_cm3": 1.0,
                "periodic_min_distance_a_or_lower_bound": 1.75,
            },
        ]
    )
    original = details.copy(deep=True)

    display = compact_box_summary_table(details)

    assert list(display.columns) == ["Setting", "Checked value"]
    assert display["Setting"].tolist() == [
        "Molecules",
        "Periodic box",
        "Cubic cell",
        "Construction density",
        "Minimum periodic separation",
    ]
    assert display.loc[0, "Checked value"] == (
        "phenol: 128 × 13 atoms; N-methylacetamide: 128 × 12 atoms"
    )
    assert display.loc[1, "Checked value"] == (
        "256 molecules; 3,200 atoms; 0 e target charge"
    )
    pdt.assert_frame_equal(details, original)


def test_molecule_charge_displays_are_compact_and_keep_raw_tables() -> None:
    charges = pd.DataFrame(
        {
            "molecule_id": range(8),
            "component": ["phenol"] * 4 + ["N-methylacetamide"] * 4,
            "predicted_charge_e": [-0.4, -0.2, 0.1, 0.3, -0.3, -0.1, 0.2, 0.4],
        }
    )
    summary = pd.DataFrame(
        {
            "component": ["phenol", "N-methylacetamide", "all molecules"],
            "molecules": [4, 4, 8],
            "mean_charge_e": [-0.05, 0.05, 0.0],
            "mean_abs_charge_e": [0.25, 0.25, 0.25],
            "standard_deviation_e": [0.269, 0.269, 0.274],
            "minimum_charge_e": [-0.4, -0.3, -0.4],
            "maximum_charge_e": [0.3, 0.4, 0.4],
            "total_charge_e": [-0.2, 0.2, 0.0],
        }
    )
    charges_original = charges.copy(deep=True)
    summary_original = summary.copy(deep=True)

    display = molecule_charge_display_tables(
        charges,
        summary,
        extremes_per_side=2,
    )

    assert list(display.summary.columns) == [
        "Molecule",
        "Count",
        "Mean charge / e",
        "Minimum / e",
        "Maximum / e",
        "Total / e",
    ]
    assert list(display.extremes.columns) == [
        "Molecule ID",
        "Molecule",
        "Predicted charge / e",
    ]
    assert display.extremes["Predicted charge / e"].tolist() == [
        -0.4,
        -0.3,
        0.3,
        0.4,
    ]
    pdt.assert_frame_equal(charges, charges_original)
    pdt.assert_frame_equal(summary, summary_original)
