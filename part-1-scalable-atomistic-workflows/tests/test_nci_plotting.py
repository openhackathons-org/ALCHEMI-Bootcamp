"""Tests for the NCI curve plot and shared Part 1 visual treatment."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.nci_atlas import (  # noqa: E402
    extract_repeated_interaction_reference,
    load_nci_atlas_subset,
)
from aux.nci_plotting import NCI_COLORS, plot_nci_interaction_curves  # noqa: E402
from aux.plotting import COMPONENT_COLORS, DFT_COLOR  # noqa: E402


DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"


def _curves():
    subset = load_nci_atlas_subset(DATA_FILE)
    curves = extract_repeated_interaction_reference(
        subset,
        "ccsd_t_cbs_interaction_energy_kcal_mol",
        output_column="ccsd_t_cbs",
    )
    curves["core"] = curves["ccsd_t_cbs"] + 3.0
    curves["core_plus_d3"] = curves["ccsd_t_cbs"] + 2.0
    curves["core_plus_coulomb"] = curves["ccsd_t_cbs"] + 1.0
    curves["full"] = curves["ccsd_t_cbs"] + 0.2
    curves["full_std"] = 0.1
    curves["dft_full"] = curves["ccsd_t_cbs"] + 0.05
    return curves


def test_nci_colors_reuse_the_part1_palette() -> None:
    assert NCI_COLORS["core"] == COMPONENT_COLORS["residual_interaction_kJ_mol"]
    assert NCI_COLORS["full"] == COMPONENT_COLORS["full_interaction_kJ_mol"]
    assert NCI_COLORS["dft_full"] == DFT_COLOR


def test_plot_returns_three_styled_panels_without_display_side_effects() -> None:
    figure, axes = plot_nci_interaction_curves(_curves())

    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == [
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    ]
    assert axes[0].get_ylabel() == "interaction energy / kcal mol$^{-1}$"
    assert axes[0].spines["top"].get_visible() is False
    assert axes[0].lines[0].get_color() == NCI_COLORS["core"]
    assert figure._suptitle.get_text() == (
        "Interaction curves across three chemical regimes"
    )
    plt.close(figure)


def test_plot_rejects_nonfinite_curve_values() -> None:
    curves = _curves()
    curves.loc[curves.index[0], "full"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        plot_nci_interaction_curves(curves)
