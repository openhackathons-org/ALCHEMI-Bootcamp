"""Checks for the independent NCI force-validation helper."""

from __future__ import annotations

from pathlib import Path
import sys

from ase import Atoms
import pytest
import torch


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.nci_config import NCIValidationSettings  # noqa: E402
from aux.nci_validation import (  # noqa: E402
    build_nci_force_check_table,
    check_nci_force,
    nci_force_check_record,
)


class HarmonicCalculator:
    """Exact E = sum(r^2)/2 route used without runtime patching."""

    def __call__(self, inputs, *, forces: bool):
        assert forces
        positions = inputs["coord"]
        return {
            "energy": (0.5 * positions.square().sum()).reshape(1),
            "forces": -positions,
        }


SETTINGS = NCIValidationSettings(
    method_id="unit-test",
    charge_atol_e=1.0e-8,
    interaction_energy_atol_eV=1.0e-8,
    net_force_atol_eV_A=1.0e-12,
    finite_difference_step_A=1.0e-4,
    finite_difference_atol_eV_A=1.0e-8,
    finite_difference_rtol=1.0e-8,
    toolkit_official_force_atol_eV_A=1.0e-12,
)


def test_nci_force_check_compares_net_force_derivative_and_toolkit_route() -> None:
    atoms = Atoms("HH", positions=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    toolkit_forces = -torch.tensor(atoms.positions, dtype=torch.float64)

    result = check_nci_force(
        example=atoms,
        toolkit_forces=toolkit_forces,
        official_calculator=HarmonicCalculator(),
        device="cpu",
        settings=SETTINGS,
    )

    assert result.atom_index in (0, 1)
    assert result.axis == 0
    assert result.net_force_max_abs_eV_A == 0.0
    assert result.official_finite_difference_error_eV_A < 1.0e-8
    assert result.toolkit_official_error_eV_A == 0.0


def test_force_check_table_reports_all_three_passed_checks() -> None:
    atoms = Atoms("HH", positions=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    result = check_nci_force(
        example=atoms,
        toolkit_forces=-torch.tensor(atoms.positions, dtype=torch.float64),
        official_calculator=HarmonicCalculator(),
        device="cpu",
        settings=SETTINGS,
    )

    table = build_nci_force_check_table(result, SETTINGS)

    assert table["check"].tolist() == [
        "Largest component of the total force",
        "Official force vs energy derivative",
        "Toolkit pipeline vs official calculator",
    ]
    assert table["passed"].tolist() == [True, True, True]
    assert table.loc[1, "coordinate"].startswith("atom ")


def test_force_check_record_preserves_values_and_route_names() -> None:
    atoms = Atoms("HH", positions=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    result = check_nci_force(
        example=atoms,
        toolkit_forces=-torch.tensor(atoms.positions, dtype=torch.float64),
        official_calculator=HarmonicCalculator(),
        device="cpu",
        settings=SETTINGS,
    )

    record = nci_force_check_record(result, SETTINGS)

    assert record["atom_index"] == result.atom_index
    assert record["finite_difference_step_A"] == SETTINGS.finite_difference_step_A
    assert record["toolkit_analytic_force_eV_A"] == result.toolkit_force_eV_A
    assert "PipelineModelWrapper" in record["toolkit_analytic_force_route"]


def test_nci_force_check_requires_explicit_settings() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    with pytest.raises(TypeError, match="NCIValidationSettings"):
        check_nci_force(
            example=atoms,
            toolkit_forces=torch.zeros((1, 3), dtype=torch.float64),
            official_calculator=HarmonicCalculator(),
            device="cpu",
            settings=None,  # type: ignore[arg-type]
        )
