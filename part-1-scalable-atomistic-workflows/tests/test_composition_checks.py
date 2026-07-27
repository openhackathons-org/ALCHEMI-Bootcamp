"""Focused tests for model-composition reductions and table formatting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.composition_checks import (  # noqa: E402
    build_composition_check_table,
    central_difference_force,
    compare_composition_outputs,
    compare_two_particle_coulomb,
)


def _limits() -> dict[str, float]:
    return {
        "energy_eV": 1.0e-3,
        "interaction_energy_eV": 1.0e-3,
        "forces_eV_A": 3.0e-6,
        "charges_e": 2.0e-7,
        "analytic_coulomb_energy_eV": 2.0e-6,
        "analytic_coulomb_forces_eV_A": 2.0e-6,
        "finite_difference_force_error_eV_A": 2.0e-3,
    }


def test_composition_agreement_includes_ab_minus_a_minus_b() -> None:
    reference = {
        "energy": np.array([[-3.0], [-1.0], [-1.0], [-6.0], [-2.0], [-3.0]]),
        "forces": np.zeros((4, 3)),
        "charges": np.zeros(4),
    }
    toolkit = {
        "energy": reference["energy"]
        + np.array([[3.0e-4], [1.0e-4], [-1.0e-4], [-2.0e-4], [0.0], [1.0e-4]]),
        "forces": reference["forces"] + 2.0e-6,
        "charges": reference["charges"] - 1.0e-7,
    }

    result = compare_composition_outputs(toolkit, reference, interaction_triplets=2)

    assert result.energy_eV == pytest.approx(3.0e-4)
    assert result.interaction_energy_eV == pytest.approx(3.0e-4)
    assert result.forces_eV_A == pytest.approx(2.0e-6)
    assert result.charges_e == pytest.approx(1.0e-7)


def test_composition_agreement_rejects_non_triplet_energy_rows() -> None:
    outputs = {
        "energy": np.zeros((4, 1)),
        "forces": np.zeros((2, 3)),
        "charges": np.zeros(2),
    }
    with pytest.raises(ValueError, match="AB, A, B triplet"):
        compare_composition_outputs(outputs, outputs, interaction_triplets=1)


def test_two_particle_coulomb_handles_general_orientation() -> None:
    constant = 14.0
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    charges = np.array([0.75, -0.75])
    expected_energy = constant * np.prod(charges) / 2.0
    expected_force = constant * abs(np.prod(charges)) / 4.0
    forces = np.array([[0.0, expected_force, 0.0], [0.0, -expected_force, 0.0]])

    result = compare_two_particle_coulomb(
        positions_angstrom=positions,
        charges_e=charges,
        observed_energy_eV=expected_energy,
        observed_forces_eV_A=forces,
        coulomb_constant_eV_A_per_e2=constant,
    )

    assert result.expected_energy_eV == pytest.approx(expected_energy)
    np.testing.assert_allclose(result.expected_forces_eV_A, forces)
    assert result.energy_error_eV == pytest.approx(0.0)
    assert result.forces_error_eV_A == pytest.approx(0.0)


def test_central_difference_returns_negative_energy_derivative() -> None:
    assert central_difference_force(
        1.2, 1.0, displacement_angstrom=0.1
    ) == pytest.approx(1.0)


def test_check_table_uses_only_caller_supplied_limits() -> None:
    agreement = compare_composition_outputs(
        {
            "energy": np.array([[0.0], [0.0], [0.0]]),
            "forces": np.zeros((2, 3)),
            "charges": np.zeros(2),
        },
        {
            "energy": np.array([[0.0], [0.0], [0.0]]),
            "forces": np.zeros((2, 3)),
            "charges": np.zeros(2),
        },
        interaction_triplets=1,
    )
    coulomb = compare_two_particle_coulomb(
        positions_angstrom=[[0, 0, 0], [1, 0, 0]],
        charges_e=[1, -1],
        observed_energy_eV=-1.0,
        observed_forces_eV_A=[[1, 0, 0], [-1, 0, 0]],
        coulomb_constant_eV_A_per_e2=1.0,
    )
    table = build_composition_check_table(
        agreement=agreement,
        coulomb=coulomb,
        finite_difference_energy_route="official total energy",
        finite_difference_step_A=0.001,
        finite_difference_force_eV_A=0.2,
        reference_analytic_force_eV_A=0.2,
        toolkit_force_eV_A=0.2,
        reference_finite_difference_error_eV_A=1.0e-5,
        toolkit_finite_difference_error_eV_A=2.0e-5,
        limits=_limits(),
    )

    assert table.shape == (8, 4)
    assert table["passed"].all()
    assert table.attrs["finite_difference"]["finite_difference_step_A"] == 0.001


def test_check_table_rejects_missing_limit() -> None:
    limits = _limits()
    limits.pop("charges_e")
    agreement = compare_composition_outputs(
        {
            "energy": np.zeros((3, 1)),
            "forces": np.zeros((2, 3)),
            "charges": np.zeros(2),
        },
        {
            "energy": np.zeros((3, 1)),
            "forces": np.zeros((2, 3)),
            "charges": np.zeros(2),
        },
        interaction_triplets=1,
    )
    coulomb = compare_two_particle_coulomb(
        positions_angstrom=[[0, 0, 0], [1, 0, 0]],
        charges_e=[1, -1],
        observed_energy_eV=-1.0,
        observed_forces_eV_A=[[1, 0, 0], [-1, 0, 0]],
        coulomb_constant_eV_A_per_e2=1.0,
    )
    with pytest.raises(ValueError, match="limits must contain exactly"):
        build_composition_check_table(
            agreement=agreement,
            coulomb=coulomb,
            finite_difference_energy_route="official",
            finite_difference_step_A=0.001,
            finite_difference_force_eV_A=0.0,
            reference_analytic_force_eV_A=0.0,
            toolkit_force_eV_A=0.0,
            reference_finite_difference_error_eV_A=0.0,
            toolkit_finite_difference_error_eV_A=0.0,
            limits=limits,
        )
