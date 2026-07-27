"""Numerical check settings for the Part 1 NCI example.

These values check equivalent calculation routes and basic physical
invariants.  They are not model-accuracy targets.  Keeping them in one named
configuration makes the notebook easier to read while still allowing it to
display every value, unit, and purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

import pandas as pd

from .composition_config import (
    COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
    COMPOSITION_FD_STEP_A,
    COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
)

# Tutorial acceptance threshold for the three selected interaction curves.
# It is deliberately separate from the numerical route-agreement settings
# below: passing it is a focused composition check, not a general accuracy
# guarantee for AIMNet2 or for noncovalent interactions.
NCI_COMPLETE_MAE_LIMIT_KCAL_MOL = 0.5


def _setting(
    *,
    label: str,
    unit: str,
    purpose: str,
) -> object:
    """Create metadata for one learner-facing numerical setting."""

    return field(metadata={"label": label, "unit": unit, "purpose": purpose})


@dataclass(frozen=True)
class NCIValidationSettings:
    """Named tolerances used by the NCI composition checks.

    No field has a fallback value.  A caller must pass one complete, explicit
    settings object so the applied method cannot silently change.
    """

    method_id: str
    charge_atol_e: float = _setting(
        label="Graph charge conservation",
        unit="e",
        purpose="Catch missing or mixed predicted charges between graphs.",
    )
    interaction_energy_atol_eV: float = _setting(
        label="AB − A − B route agreement",
        unit="eV",
        purpose=(
            "Compare component sums and graph orders after reducing absolute "
            "energies to interaction energies."
        ),
    )
    net_force_atol_eV_A: float = _setting(
        label="Net force after translation",
        unit="eV/Å",
        purpose="Check that an isolated complex has no spurious total force.",
    )
    finite_difference_step_A: float = _setting(
        label="Force finite-difference displacement",
        unit="Å",
        purpose="Displace one coordinate for the independent energy derivative.",
    )
    finite_difference_atol_eV_A: float = _setting(
        label="Energy derivative: absolute tolerance",
        unit="eV/Å",
        purpose="Compare the official analytic force with its energy derivative.",
    )
    finite_difference_rtol: float = _setting(
        label="Energy derivative: relative tolerance",
        unit="fraction",
        purpose="Allow finite-difference error to scale with force magnitude.",
    )
    toolkit_official_force_atol_eV_A: float = _setting(
        label="Toolkit force vs official calculator",
        unit="eV/Å",
        purpose="Check the composed Toolkit model against AIMNet2's official route.",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a non-empty string")
        for item in fields(self):
            if item.name == "method_id":
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{item.name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{item.name} must be finite and positive")

    def as_record(self) -> dict[str, str | float]:
        """Return the exact applied values for the saved run summary."""

        return {
            "method_id": self.method_id,
            **{
                item.name: float(getattr(self, item.name))
                for item in fields(self)
                if item.name != "method_id"
            },
        }


NCI_VALIDATION = NCIValidationSettings(
    method_id="nci-composition-check-v2",
    charge_atol_e=2.0e-4,
    interaction_energy_atol_eV=3.0e-3,
    net_force_atol_eV_A=5.0e-3,
    finite_difference_step_A=COMPOSITION_FD_STEP_A,
    finite_difference_atol_eV_A=COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
    finite_difference_rtol=2.0e-2,
    toolkit_official_force_atol_eV_A=(COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A),
)


def nci_validation_settings_table(
    settings: NCIValidationSettings,
) -> pd.DataFrame:
    """Return the eight numerical settings as a compact display table."""

    if not isinstance(settings, NCIValidationSettings):
        raise TypeError("settings must be an NCIValidationSettings instance")
    rows = []
    for item in fields(settings):
        if item.name == "method_id":
            continue
        rows.append(
            {
                "check": item.metadata["label"],
                "value": float(getattr(settings, item.name)),
                "unit": item.metadata["unit"],
                "why it is checked": item.metadata["purpose"],
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "NCI_COMPLETE_MAE_LIMIT_KCAL_MOL",
    "NCI_VALIDATION",
    "NCIValidationSettings",
    "nci_validation_settings_table",
]
