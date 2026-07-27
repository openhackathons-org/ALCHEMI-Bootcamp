"""Small, display-only tables for the water IR lesson.

The scientific work stays in the reference loaders and mode-mapping helpers.
These functions only arrange their verified results for a learner-facing table.
They always work on copies, so display rounding cannot change values used by
later analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .reference import reference_water_monomer_mode_labels


_MONOMER_LABELS = ("H2O", "D2O")
_MODE_NUMBER = {
    "symmetric_stretch": 1,
    "bend": 2,
    "antisymmetric_stretch": 3,
}
_MODE_DISPLAY = {
    "symmetric_stretch": "ν1 symmetric stretch",
    "bend": "ν2 bend",
    "antisymmetric_stretch": "ν3 antisymmetric stretch",
}


@dataclass(frozen=True)
class MonomerReferenceDisplay:
    """Display table plus values reused by the later harmonic and IR plots."""

    table: pd.DataFrame
    observed_by_mode: pd.DataFrame
    harmonic_mode_indices: dict[str, tuple[int, ...]]


def _required_columns(table: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def prepare_monomer_reference_display(
    references: Mapping[str, Any],
    experimental_fundamentals: pd.DataFrame,
) -> MonomerReferenceDisplay:
    """Match B97-3c monomer modes to observed H2O and D2O positions.

    ``references`` must contain the already loaded and validated B97-3c H2O
    and D2O bundles. ``experimental_fundamentals`` is the checksum-verified
    table returned by :func:`load_experimental_water_fundamentals`. The result
    keeps the raw observed lookup and one-based harmonic mode indices needed by
    later calculations, while ``table`` is rounded and ordered for display.
    """

    missing_references = sorted(set(_MONOMER_LABELS) - set(references))
    if missing_references:
        raise ValueError(
            "missing monomer harmonic references: " + ", ".join(missing_references)
        )
    if not isinstance(experimental_fundamentals, pd.DataFrame):
        raise TypeError("experimental_fundamentals must be a pandas DataFrame")
    _required_columns(
        experimental_fundamentals,
        {"isotopologue", "mode", "wavenumber_cm1"},
        name="experimental fundamentals",
    )

    observed = experimental_fundamentals.copy()
    observed = observed.loc[observed["isotopologue"].isin(_MONOMER_LABELS)]
    duplicated = observed.duplicated(["isotopologue", "mode"], keep=False)
    if duplicated.any():
        raise ValueError("experimental fundamentals contain duplicate mode rows")
    observed_by_mode = observed.set_index(["isotopologue", "mode"])

    expected_pairs = {
        (label, mode) for label in _MONOMER_LABELS for mode in _MODE_NUMBER
    }
    missing_pairs = sorted(expected_pairs - set(observed_by_mode.index))
    if missing_pairs:
        missing_text = ", ".join(f"{label}/{mode}" for label, mode in missing_pairs)
        raise ValueError(f"experimental fundamentals are missing: {missing_text}")

    rows: list[dict[str, Any]] = []
    harmonic_mode_indices: dict[str, tuple[int, ...]] = {}
    for label in _MONOMER_LABELS:
        reference = references[label]
        assignments = tuple(reference_water_monomer_mode_labels(reference))
        frequencies = np.asarray(reference.frequencies_cm1, dtype=np.float64)
        if frequencies.ndim != 1 or len(assignments) != frequencies.size:
            raise ValueError(f"{label} mode labels do not match its frequencies")
        if set(assignments) != set(_MODE_NUMBER) or len(assignments) != 3:
            raise ValueError(f"{label} does not contain the three water monomer modes")
        if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
            raise ValueError(
                f"{label} harmonic frequencies must be finite and positive"
            )

        harmonic_mode_indices[label] = tuple(
            _MODE_NUMBER[mode] for mode in assignments
        )
        for mode, harmonic_cm1 in zip(assignments, frequencies, strict=True):
            observed_cm1 = float(
                observed_by_mode.loc[(label, mode), "wavenumber_cm1"]
            )
            if not np.isfinite(observed_cm1) or observed_cm1 <= 0.0:
                raise ValueError(
                    f"observed frequency for {label}/{mode} must be finite and positive"
                )
            rows.append(
                {
                    "System": label,
                    "Mode": _MODE_DISPLAY[mode],
                    "B97-3c harmonic (cm-1)": harmonic_cm1,
                    "Observed gas phase (cm-1)": observed_cm1,
                    "Harmonic - observed (cm-1)": harmonic_cm1 - observed_cm1,
                    "_mode_order": _MODE_NUMBER[mode],
                }
            )

    table = pd.DataFrame(rows)
    table["_system_order"] = table["System"].map(
        {label: index for index, label in enumerate(_MONOMER_LABELS)}
    )
    table = (
        table.sort_values(["_system_order", "_mode_order"])
        .drop(columns=["_system_order", "_mode_order"])
        .reset_index(drop=True)
        .round(2)
    )
    return MonomerReferenceDisplay(
        table=table,
        observed_by_mode=observed_by_mode,
        harmonic_mode_indices=harmonic_mode_indices,
    )


def monomer_mode_mapping_display_table(mode_mapping: Any) -> pd.DataFrame:
    """Return the three H2O-to-D2O assignments as a concise display table.

    ``mode_mapping`` is the result returned by
    :func:`aux.analysis.h_to_d_mode_mapping_table`. The displayed match score
    is the lowest squared mode overlap along the isotope-mass path. The full
    monomer and cluster table remains available as ``mode_mapping.table``.
    """

    source = getattr(mode_mapping, "table", None)
    if not isinstance(source, pd.DataFrame):
        raise TypeError("mode_mapping must provide a pandas DataFrame as .table")
    required = {
        "system",
        "character",
        "H_center_cm-1",
        "D_center_cm-1",
        "H_over_D_center",
        "mapping_overlap",
    }
    _required_columns(source, required, name="mode mapping table")

    monomer = source.loc[source["system"] == "monomer", sorted(required)].copy()
    if len(monomer) != 3 or set(monomer["character"]) != set(_MODE_NUMBER):
        raise ValueError("mode mapping must contain the three water monomer modes")

    numeric_columns = (
        "H_center_cm-1",
        "D_center_cm-1",
        "H_over_D_center",
        "mapping_overlap",
    )
    numeric = monomer.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("monomer mode mapping contains non-finite values")
    if np.any(numeric[:, :2] <= 0.0):
        raise ValueError("monomer mode frequencies must be positive")
    if np.any(numeric[:, 2] <= 0.0):
        raise ValueError("monomer H/D frequency ratios must be positive")
    overlap = numeric[:, 3]
    if np.any((overlap < 0.0) | (overlap > 1.0)):
        raise ValueError("monomer mode match scores must be between zero and one")

    monomer["_mode_order"] = monomer["character"].map(_MODE_NUMBER)
    monomer["Mode"] = monomer["character"].map(_MODE_DISPLAY)
    display = monomer.sort_values("_mode_order").loc[
        :,
        [
            "Mode",
            "H_center_cm-1",
            "D_center_cm-1",
            "H_over_D_center",
            "mapping_overlap",
        ],
    ]
    display = display.rename(
        columns={
            "H_center_cm-1": "H2O harmonic (cm-1)",
            "D_center_cm-1": "D2O harmonic (cm-1)",
            "H_over_D_center": "H/D frequency ratio",
            "mapping_overlap": "Mode match (0-1)",
        }
    ).reset_index(drop=True)
    return display.round(
        {
            "H2O harmonic (cm-1)": 1,
            "D2O harmonic (cm-1)": 1,
            "H/D frequency ratio": 3,
            "Mode match (0-1)": 4,
        }
    )


__all__ = [
    "MonomerReferenceDisplay",
    "monomer_mode_mapping_display_table",
    "prepare_monomer_reference_display",
]
