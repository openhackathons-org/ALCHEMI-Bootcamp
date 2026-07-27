"""Pure reductions and tables for the model-composition lesson.

The notebook owns every model call, finite-difference geometry, and numerical
limit.  This module only compares already-computed tensors and formats their
results so the learner-facing cell can stay focused on Toolkit composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class CompositionAgreement:
    """Maximum absolute differences between two complete-model routes."""

    energy_eV: float
    interaction_energy_eV: float
    forces_eV_A: float
    charges_e: float

    def as_dict(self) -> dict[str, float]:
        return {
            "energy_eV": self.energy_eV,
            "interaction_energy_eV": self.interaction_energy_eV,
            "forces_eV_A": self.forces_eV_A,
            "charges_e": self.charges_e,
        }


@dataclass(frozen=True)
class CoulombPairAgreement:
    """Analytic two-particle Coulomb expectation and observed errors."""

    expected_energy_eV: float
    expected_forces_eV_A: np.ndarray
    energy_error_eV: float
    forces_error_eV_A: float


def compare_composition_outputs(
    toolkit_outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
    *,
    interaction_triplets: int,
) -> CompositionAgreement:
    """Compare energy, AB-A-B interaction, force, and charge tensors."""

    if isinstance(interaction_triplets, bool) or interaction_triplets <= 0:
        raise ValueError("interaction_triplets must be a positive integer")
    if int(interaction_triplets) != interaction_triplets:
        raise ValueError("interaction_triplets must be a positive integer")
    interaction_triplets = int(interaction_triplets)

    energy = _array(toolkit_outputs["energy"], name="Toolkit energy").reshape(-1)
    reference_energy = _array(
        reference_outputs["energy"], name="reference energy"
    ).reshape(-1)
    if energy.shape != reference_energy.shape:
        raise ValueError("energy tensors must have the same shape")
    if energy.size != 3 * interaction_triplets:
        raise ValueError("energy rows must contain one AB, A, B triplet per point")

    energy_delta = energy - reference_energy
    triplets = energy_delta.reshape(interaction_triplets, 3)
    interaction_delta = triplets[:, 0] - triplets[:, 1] - triplets[:, 2]

    forces = _array(toolkit_outputs["forces"], name="Toolkit forces")
    reference_forces = _array(reference_outputs["forces"], name="reference forces")
    if forces.shape != reference_forces.shape:
        raise ValueError("force tensors must have the same shape")

    charges = _array(toolkit_outputs["charges"], name="Toolkit charges").reshape(-1)
    reference_charges = _array(
        reference_outputs["charges"], name="reference charges"
    ).reshape(-1)
    if charges.shape != reference_charges.shape:
        raise ValueError("charge tensors must have the same shape")

    return CompositionAgreement(
        energy_eV=float(np.max(np.abs(energy_delta))),
        interaction_energy_eV=float(np.max(np.abs(interaction_delta))),
        forces_eV_A=float(np.max(np.abs(forces - reference_forces))),
        charges_e=float(np.max(np.abs(charges - reference_charges))),
    )


def compare_two_particle_coulomb(
    *,
    positions_angstrom: Any,
    charges_e: Any,
    observed_energy_eV: Any,
    observed_forces_eV_A: Any,
    coulomb_constant_eV_A_per_e2: float,
) -> CoulombPairAgreement:
    """Compare one two-particle result with the analytic 1/r expression."""

    positions = _array(positions_angstrom, name="positions").astype(np.float64)
    charges = _array(charges_e, name="charges").astype(np.float64).reshape(-1)
    observed_forces = _array(observed_forces_eV_A, name="observed forces").astype(
        np.float64
    )
    observed_energy = _array(observed_energy_eV, name="observed energy").reshape(-1)
    if positions.shape != (2, 3) or observed_forces.shape != (2, 3):
        raise ValueError("the analytic Coulomb check requires two 3D particles")
    if charges.shape != (2,) or observed_energy.size != 1:
        raise ValueError(
            "the analytic Coulomb check requires two charges and one energy"
        )
    constant = float(coulomb_constant_eV_A_per_e2)
    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("coulomb_constant_eV_A_per_e2 must be positive and finite")

    displacement = positions[1] - positions[0]
    distance = float(np.linalg.norm(displacement))
    if distance <= 0.0:
        raise ValueError("the two particles must occupy different positions")
    charge_product = float(charges[0] * charges[1])
    expected_energy = constant * charge_product / distance
    force_on_first = -constant * charge_product * displacement / distance**3
    expected_forces = np.stack((force_on_first, -force_on_first))
    return CoulombPairAgreement(
        expected_energy_eV=expected_energy,
        expected_forces_eV_A=expected_forces,
        energy_error_eV=abs(float(observed_energy[0]) - expected_energy),
        forces_error_eV_A=float(np.max(np.abs(observed_forces - expected_forces))),
    )


def central_difference_force(
    energy_minus_eV: float,
    energy_plus_eV: float,
    *,
    displacement_angstrom: float,
) -> float:
    """Return ``-dE/dx`` from symmetric already-computed energies."""

    values = np.asarray(
        [energy_minus_eV, energy_plus_eV, displacement_angstrom],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("finite-difference inputs must be finite")
    if displacement_angstrom <= 0.0:
        raise ValueError("displacement_angstrom must be positive")
    return -float(energy_plus_eV - energy_minus_eV) / (
        2.0 * float(displacement_angstrom)
    )


def build_composition_check_table(
    *,
    agreement: CompositionAgreement,
    coulomb: CoulombPairAgreement,
    finite_difference_energy_route: str,
    finite_difference_step_A: float,
    finite_difference_force_eV_A: float,
    reference_analytic_force_eV_A: float,
    toolkit_force_eV_A: float,
    reference_finite_difference_error_eV_A: float,
    toolkit_finite_difference_error_eV_A: float,
    limits: Mapping[str, float],
) -> pd.DataFrame:
    """Build one readable table without selecting any acceptance limits."""

    required_limits = {
        "energy_eV",
        "interaction_energy_eV",
        "forces_eV_A",
        "charges_e",
        "analytic_coulomb_energy_eV",
        "analytic_coulomb_forces_eV_A",
        "finite_difference_force_error_eV_A",
    }
    if set(limits) != required_limits:
        raise ValueError(
            "limits must contain exactly: " + ", ".join(sorted(required_limits))
        )
    clean_limits = {name: float(value) for name, value in limits.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in clean_limits.values()):
        raise ValueError("every composition-check limit must be positive and finite")

    measured = {
        "Toolkit vs official energy": (agreement.energy_eV, "eV", "energy_eV"),
        "Toolkit vs official interaction energy": (
            agreement.interaction_energy_eV,
            "eV",
            "interaction_energy_eV",
        ),
        "Toolkit vs official force component": (
            agreement.forces_eV_A,
            "eV/A",
            "forces_eV_A",
        ),
        "Toolkit vs official charge": (agreement.charges_e, "e", "charges_e"),
        "Analytic Coulomb energy": (
            coulomb.energy_error_eV,
            "eV",
            "analytic_coulomb_energy_eV",
        ),
        "Analytic Coulomb force component": (
            coulomb.forces_error_eV_A,
            "eV/A",
            "analytic_coulomb_forces_eV_A",
        ),
        "Official analytic vs finite-difference force": (
            float(reference_finite_difference_error_eV_A),
            "eV/A",
            "finite_difference_force_error_eV_A",
        ),
        "Toolkit vs finite-difference force": (
            float(toolkit_finite_difference_error_eV_A),
            "eV/A",
            "finite_difference_force_error_eV_A",
        ),
    }
    rows = []
    for check, (observed, units, limit_name) in measured.items():
        observed = float(observed)
        limit = clean_limits[limit_name]
        rows.append(
            {
                "check": check,
                "max_abs_difference": observed,
                "limit": limit,
                "units": units,
                "passed": observed < limit,
            }
        )

    context = {
        "finite_difference_energy_route": str(finite_difference_energy_route),
        "finite_difference_step_A": float(finite_difference_step_A),
        "finite_difference_force_eV_A": float(finite_difference_force_eV_A),
        "official_analytic_force_eV_A": float(reference_analytic_force_eV_A),
        "toolkit_force_eV_A": float(toolkit_force_eV_A),
    }
    table = pd.DataFrame(rows).set_index("check")
    table.attrs["finite_difference"] = context
    return table


__all__ = [
    "CompositionAgreement",
    "CoulombPairAgreement",
    "build_composition_check_table",
    "central_difference_force",
    "compare_composition_outputs",
    "compare_two_particle_coulomb",
]
