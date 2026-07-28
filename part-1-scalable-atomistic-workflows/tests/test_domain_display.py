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
    domain_agreement_display_table,
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


def test_domain_agreement_display_separates_float32_and_distributed_checks() -> None:
    agreement = pd.DataFrame(
        {
            "world_size": [1, 2, 4],
            "energy_dtype": ["torch.float32", "torch.float64", "torch.float64"],
            "energy_repeatability_span_meV_atom": [0.5, 0.02, 0.03],
            "energy_repeatability_check_required": [False, True, True],
            "energy_repeatability_passed": [None, True, True],
            "energy_reference_world_size": [2, 2, 2],
            "energy_difference_meV_atom": [1.5, 0.0, 0.04],
            "energy_check_required": [False, False, True],
            "energy_passed": [None, None, True],
            "force_reference_world_size": [1, 1, 1],
            "force_rms_error_eV_A": [0.0, 0.0001, 0.0002],
            "force_max_error_eV_A": [0.0, 0.0002, 0.0004],
            "force_passed": [True, True, True],
            "passed": [True, True, True],
        }
    )
    original = agreement.copy(deep=True)

    display = domain_agreement_display_table(agreement)

    assert display["GPUs"].tolist() == [1, 2, 4]
    assert display["Model tensors / coordinates / forces"].tolist() == [
        "float32",
        "float32",
        "float32",
    ]
    assert display["Energy total"].tolist() == [
        "float32 model total",
        "float64 multi-rank reduction",
        "float64 multi-rank reduction",
    ]
    assert display["Energy repeatability"].tolist() == [
        "Not checked: float32 total",
        "Passed",
        "Passed",
    ]
    assert display["Distributed energy"].tolist() == [
        "Not compared: float32 total",
        "Reference: 2-GPU float64 reduction",
        "Passed vs 2 GPU",
    ]
    assert display["Forces"].tolist() == [
        "Reference: 1 GPU",
        "Passed vs 1 GPU",
        "Passed vs 1 GPU",
    ]
    assert "passed" not in display.columns
    pdt.assert_frame_equal(agreement, original)


def test_domain_agreement_display_rejects_ambiguous_energy_dtype() -> None:
    agreement = pd.DataFrame(
        {
            "world_size": [1],
            "energy_dtype": ["torch.float64"],
            "energy_repeatability_span_meV_atom": [0.0],
            "energy_repeatability_check_required": [False],
            "energy_repeatability_passed": [None],
            "energy_reference_world_size": [1],
            "energy_difference_meV_atom": [0.0],
            "energy_check_required": [False],
            "energy_passed": [None],
            "force_reference_world_size": [1],
            "force_rms_error_eV_A": [0.0],
            "force_max_error_eV_A": [0.0],
            "force_passed": [True],
        }
    )

    try:
        domain_agreement_display_table(agreement)
    except ValueError as exc:
        assert "one-GPU total energy must be torch.float32" in str(exc)
    else:
        raise AssertionError("ambiguous one-GPU energy dtype was accepted")
