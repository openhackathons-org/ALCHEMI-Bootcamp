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
        f"{row.component}: {int(row.molecules)} × {int(row.atoms_per_molecule)} atoms"
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
        raise ValueError(
            "periodic box counts, cell, density, and distance must be positive"
        )

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


def _display_bool(
    value: object, *, name: str, allow_missing: bool = False
) -> bool | None:
    if pd.isna(value):
        if allow_missing:
            return None
        raise ValueError(f"{name} must be true or false")
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be true or false")
    return bool(value)


def domain_agreement_display_table(agreement: pd.DataFrame) -> pd.DataFrame:
    """Make the dtype and scope of each saved output check explicit.

    The one-GPU total is a float32 model output, so its observed pass-to-pass
    span is a diagnostic rather than the float64 distributed-repeatability
    check used for the two- and four-GPU layouts.
    """

    required = {
        "world_size",
        "energy_dtype",
        "energy_repeatability_span_meV_atom",
        "energy_repeatability_check_required",
        "energy_repeatability_passed",
        "energy_reference_world_size",
        "energy_difference_meV_atom",
        "energy_check_required",
        "energy_passed",
        "force_reference_world_size",
        "force_rms_error_eV_A",
        "force_max_error_eV_A",
        "force_passed",
    }
    _require_columns(agreement, required, name="output agreement")
    columns = (
        "GPUs",
        "Model tensors / coordinates / forces",
        "Energy total",
        "Energy repeatability",
        "Distributed energy",
        "Forces",
        "Energy span / meV atom⁻¹",
        "Energy difference / meV atom⁻¹",
        "Force RMS difference / eV Å⁻¹",
        "Force max difference / eV Å⁻¹",
    )
    source = agreement.copy(deep=True)
    if source.empty:
        return pd.DataFrame(columns=columns)

    worlds = pd.to_numeric(source["world_size"], errors="coerce")
    if (
        worlds.isna().any()
        or not np.equal(worlds.to_numpy(), np.floor(worlds.to_numpy())).all()
        or (worlds <= 0).any()
    ):
        raise ValueError("output agreement world_size values must be positive integers")
    source["world_size"] = worlds.astype(int)
    if source["world_size"].duplicated().any():
        raise ValueError("output agreement must contain one row per GPU layout")
    source = source.sort_values("world_size").reset_index(drop=True)
    world_sizes = set(source["world_size"].tolist())

    numeric_columns = (
        "energy_repeatability_span_meV_atom",
        "energy_difference_meV_atom",
        "force_rms_error_eV_A",
        "force_max_error_eV_A",
    )
    numeric = source.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("output agreement diagnostics must be finite")
    if (numeric.to_numpy(dtype=np.float64) < 0).any():
        raise ValueError("output agreement diagnostics must be non-negative")

    energy_references = pd.to_numeric(
        source["energy_reference_world_size"], errors="coerce"
    )
    force_references = pd.to_numeric(
        source["force_reference_world_size"], errors="coerce"
    )
    if (
        energy_references.isna().any()
        or force_references.isna().any()
        or energy_references.nunique() != 1
        or force_references.nunique() != 1
    ):
        raise ValueError(
            "output agreement must identify one energy and force reference"
        )
    energy_reference = int(energy_references.iloc[0])
    force_reference = int(force_references.iloc[0])
    if energy_reference not in world_sizes or force_reference not in world_sizes:
        raise ValueError("output agreement references must be present in the table")

    records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        world_size = int(row.world_size)
        energy_dtype = str(row.energy_dtype)
        if world_size == 1:
            if energy_dtype != "torch.float32":
                raise ValueError("the one-GPU total energy must be torch.float32")
            energy_total = "float32 model total"
        else:
            if energy_dtype != "torch.float64":
                raise ValueError(
                    "multi-rank total energies must use the float64 reduction"
                )
            energy_total = "float64 multi-rank reduction"

        repeatability_required = _display_bool(
            row.energy_repeatability_check_required,
            name=f"{world_size}-GPU repeatability requirement",
        )
        repeatability_passed = _display_bool(
            row.energy_repeatability_passed,
            name=f"{world_size}-GPU repeatability result",
            allow_missing=True,
        )
        if repeatability_required:
            if repeatability_passed is None:
                raise ValueError(
                    f"{world_size}-GPU repeatability result must be reported"
                )
            repeatability_status = "Passed" if repeatability_passed else "Failed"
        else:
            if repeatability_passed is not None:
                raise ValueError(
                    f"{world_size}-GPU repeatability result must stay unreported"
                )
            repeatability_status = (
                "Not checked: float32 total"
                if energy_dtype == "torch.float32"
                else "Not checked"
            )

        energy_check_required = _display_bool(
            row.energy_check_required,
            name=f"{world_size}-GPU distributed-energy requirement",
        )
        energy_passed = _display_bool(
            row.energy_passed,
            name=f"{world_size}-GPU distributed-energy result",
            allow_missing=True,
        )
        if world_size == energy_reference:
            if energy_check_required or energy_passed is not None:
                raise ValueError(
                    "the distributed-energy reference must not compare with itself"
                )
            distributed_energy_status = (
                f"Reference: {energy_reference}-GPU float64 reduction"
            )
        elif energy_check_required:
            if energy_passed is None:
                raise ValueError(
                    f"{world_size}-GPU distributed-energy result must be reported"
                )
            distributed_energy_status = (
                f"Passed vs {energy_reference} GPU"
                if energy_passed
                else f"Failed vs {energy_reference} GPU"
            )
        else:
            if energy_passed is not None:
                raise ValueError(
                    f"{world_size}-GPU distributed-energy result must stay unreported"
                )
            distributed_energy_status = (
                "Not compared: float32 total"
                if energy_dtype == "torch.float32"
                else "Not checked"
            )

        force_passed = _display_bool(
            row.force_passed,
            name=f"{world_size}-GPU force result",
        )
        if world_size == force_reference:
            if not force_passed:
                raise ValueError("the force reference must agree with itself")
            force_status = f"Reference: {force_reference} GPU"
        else:
            force_status = (
                f"Passed vs {force_reference} GPU"
                if force_passed
                else f"Failed vs {force_reference} GPU"
            )

        records.append(
            {
                "GPUs": world_size,
                "Model tensors / coordinates / forces": "float32",
                "Energy total": energy_total,
                "Energy repeatability": repeatability_status,
                "Distributed energy": distributed_energy_status,
                "Forces": force_status,
                "Energy span / meV atom⁻¹": float(
                    row.energy_repeatability_span_meV_atom
                ),
                "Energy difference / meV atom⁻¹": float(row.energy_difference_meV_atom),
                "Force RMS difference / eV Å⁻¹": float(row.force_rms_error_eV_A),
                "Force max difference / eV Å⁻¹": float(row.force_max_error_eV_A),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


__all__ = (
    "MoleculeChargeDisplayTables",
    "compact_box_summary_table",
    "domain_agreement_display_table",
    "molecule_charge_display_tables",
)
