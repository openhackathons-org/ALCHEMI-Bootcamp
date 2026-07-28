"""Compact display tables for the Part 1 periodic-domain lesson.

The packing and result helpers keep complete numerical tables for validation
and saving. These functions work on copies and select only the values learners
need in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _require_columns(table: pd.DataFrame, required: set[str], *, name: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def compact_box_summary_table(details: pd.DataFrame) -> pd.DataFrame:
    """Turn the complete packing table into five checked, readable rows."""

    required = {
        "component",
        "molecules",
        "atoms_per_molecule",
        "atoms_total",
        "net_charge_e",
        "box_length_a",
        "construction_density_g_cm3",
        "periodic_min_distance_a_or_lower_bound",
    }
    _require_columns(details, required, name="box details")
    source = details.copy(deep=True)
    periodic = source.loc[source["component"].eq("periodic box")]
    components = source.loc[~source["component"].eq("periodic box")]
    if len(periodic) != 1 or components.empty:
        raise ValueError(
            "box details must contain molecular components and one periodic box"
        )
    periodic_row = periodic.iloc[0]
    component_text = "; ".join(
        f"{row.component}: {int(row.molecules)} × "
        f"{int(row.atoms_per_molecule)} atoms"
        for row in components.itertuples(index=False)
    )
    numeric = np.asarray(
        [
            periodic_row["molecules"],
            periodic_row["atoms_total"],
            periodic_row["net_charge_e"],
            periodic_row["box_length_a"],
            periodic_row["construction_density_g_cm3"],
            periodic_row["periodic_min_distance_a_or_lower_bound"],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise ValueError("periodic box summary contains non-finite values")
    if numeric[0] <= 0 or numeric[1] <= 0 or np.any(numeric[3:] <= 0):
        raise ValueError("periodic box counts, cell, density, and distance must be positive")

    return pd.DataFrame(
        [
            ("Molecules", component_text),
            (
                "Periodic box",
                f"{int(numeric[0])} molecules; {int(numeric[1]):,} atoms; "
                f"{numeric[2]:g} e target charge",
            ),
            ("Cubic cell", f"{numeric[3]:.3f} Å per side"),
            ("Construction density", f"{numeric[4]:.3f} g cm⁻³"),
            ("Minimum periodic separation", f"at least {numeric[5]:.3f} Å"),
        ],
        columns=("Setting", "Checked value"),
    )


@dataclass(frozen=True)
class MoleculeChargeDisplayTables:
    """Compact component summary and charge extremes."""

    summary: pd.DataFrame
    extremes: pd.DataFrame


def molecule_charge_display_tables(
    molecule_charges: pd.DataFrame,
    component_summary: pd.DataFrame,
    *,
    extremes_per_side: int = 3,
) -> MoleculeChargeDisplayTables:
    """Select readable molecular-charge views without changing raw tables."""

    if (
        isinstance(extremes_per_side, bool)
        or not isinstance(extremes_per_side, int)
        or extremes_per_side <= 0
    ):
        raise ValueError("extremes_per_side must be a positive integer")
    _require_columns(
        molecule_charges,
        {"molecule_id", "component", "predicted_charge_e"},
        name="molecule charges",
    )
    _require_columns(
        component_summary,
        {
            "component",
            "molecules",
            "mean_charge_e",
            "minimum_charge_e",
            "maximum_charge_e",
            "total_charge_e",
        },
        name="component charge summary",
    )
    charges = molecule_charges.copy(deep=True)
    summary = component_summary.copy(deep=True)
    charge_values = charges["predicted_charge_e"].to_numpy(dtype=np.float64)
    summary_values = summary[
        [
            "molecules",
            "mean_charge_e",
            "minimum_charge_e",
            "maximum_charge_e",
            "total_charge_e",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(charge_values).all() or not np.isfinite(summary_values).all():
        raise ValueError("molecular charge tables contain non-finite values")

    summary_display = summary[
        [
            "component",
            "molecules",
            "mean_charge_e",
            "minimum_charge_e",
            "maximum_charge_e",
            "total_charge_e",
        ]
    ].rename(
        columns={
            "component": "Molecule",
            "molecules": "Count",
            "mean_charge_e": "Mean charge / e",
            "minimum_charge_e": "Minimum / e",
            "maximum_charge_e": "Maximum / e",
            "total_charge_e": "Total / e",
        }
    )
    extremes = pd.concat(
        [
            charges.nsmallest(extremes_per_side, "predicted_charge_e"),
            charges.nlargest(extremes_per_side, "predicted_charge_e"),
        ],
        ignore_index=True,
    ).sort_values("predicted_charge_e")
    extremes_display = extremes.rename(
        columns={
            "molecule_id": "Molecule ID",
            "component": "Molecule",
            "predicted_charge_e": "Predicted charge / e",
        }
    ).reset_index(drop=True)
    return MoleculeChargeDisplayTables(
        summary=summary_display,
        extremes=extremes_display,
    )


__all__ = (
    "MoleculeChargeDisplayTables",
    "compact_box_summary_table",
    "molecule_charge_display_tables",
)
