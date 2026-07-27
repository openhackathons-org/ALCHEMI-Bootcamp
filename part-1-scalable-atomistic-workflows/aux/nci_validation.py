"""Independent force checks for the NCI model-composition example.

The notebook shows which structure, checkpoint, Coulomb mode, D3 settings,
finite-difference step, and tolerances are selected.  This module contains the
mechanical coordinate displacements and comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase import Atoms
import numpy as np
import pandas as pd
import torch

from nvalchemi.data import AtomicData, Batch

from .nci_config import NCIValidationSettings


@dataclass(frozen=True)
class NCIForceCheck:
    """Results from one translational and finite-difference force check."""

    atom_index: int
    axis: int
    net_force_max_abs_eV_A: float
    official_analytic_force_eV_A: float
    official_finite_difference_force_eV_A: float
    official_finite_difference_error_eV_A: float
    toolkit_force_eV_A: float
    toolkit_official_error_eV_A: float


def _official_outputs(
    calculator: Any,
    atoms: Atoms,
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    data = AtomicData.from_atoms(atoms, device=device, dtype=torch.float64)
    batch = Batch.from_data_list([data], device=device)
    inputs = {
        "coord": batch.positions,
        "numbers": batch.atomic_numbers,
        "charge": batch.charge.reshape(-1),
        "mol_idx": batch.batch_idx,
    }
    return calculator(inputs, forces=True)


def check_nci_force(
    *,
    example: Atoms,
    toolkit_forces: torch.Tensor,
    official_calculator: Any,
    device: torch.device | str,
    settings: NCIValidationSettings,
) -> NCIForceCheck:
    """Check net force, the official energy derivative, and Toolkit agreement."""

    if not isinstance(settings, NCIValidationSettings):
        raise TypeError("settings must be an NCIValidationSettings instance")
    if toolkit_forces.shape != (len(example), 3):
        raise ValueError("toolkit_forces must contain one 3-vector per atom")

    net_force = toolkit_forces.sum(dim=0)
    torch.testing.assert_close(
        net_force,
        torch.zeros_like(net_force),
        atol=settings.net_force_atol_eV_A,
        rtol=0.0,
    )

    base = _official_outputs(official_calculator, example, device=device)
    flat_index = int(base["forces"].abs().reshape(-1).argmax().detach().cpu())
    atom_index, axis = divmod(flat_index, 3)
    official_force = float(
        base["forces"][atom_index, axis].detach().cpu().reshape(())
    )

    energies = []
    for sign in (-1.0, 1.0):
        displaced = example.copy()
        displaced.positions[atom_index, axis] += (
            sign * settings.finite_difference_step_A
        )
        outputs = _official_outputs(official_calculator, displaced, device=device)
        energies.append(float(outputs["energy"].detach().cpu().reshape(())))
    finite_difference_force = -(
        energies[1] - energies[0]
    ) / (2.0 * settings.finite_difference_step_A)
    np.testing.assert_allclose(
        official_force,
        finite_difference_force,
        rtol=settings.finite_difference_rtol,
        atol=settings.finite_difference_atol_eV_A,
    )

    toolkit_force = float(toolkit_forces[atom_index, axis].detach().cpu())
    np.testing.assert_allclose(
        toolkit_force,
        official_force,
        rtol=0.0,
        atol=settings.toolkit_official_force_atol_eV_A,
    )
    return NCIForceCheck(
        atom_index=atom_index,
        axis=axis,
        net_force_max_abs_eV_A=float(net_force.abs().max().detach().cpu()),
        official_analytic_force_eV_A=official_force,
        official_finite_difference_force_eV_A=finite_difference_force,
        official_finite_difference_error_eV_A=abs(
            official_force - finite_difference_force
        ),
        toolkit_force_eV_A=toolkit_force,
        toolkit_official_error_eV_A=abs(toolkit_force - official_force),
    )


def build_nci_force_check_table(
    result: NCIForceCheck,
    settings: NCIValidationSettings,
) -> pd.DataFrame:
    """Summarize the three force checks without notebook-side table plumbing."""

    if not isinstance(result, NCIForceCheck):
        raise TypeError("result must be an NCIForceCheck instance")
    if not isinstance(settings, NCIValidationSettings):
        raise TypeError("settings must be an NCIValidationSettings instance")
    coordinate = f"atom {result.atom_index}, axis {result.axis}"
    finite_difference_limit = settings.finite_difference_atol_eV_A + (
        settings.finite_difference_rtol
        * abs(result.official_finite_difference_force_eV_A)
    )
    rows = [
        {
            "check": "Largest component of the total force",
            "coordinate": "all atoms",
            "value / eV Å⁻¹": result.net_force_max_abs_eV_A,
            "reference / eV Å⁻¹": 0.0,
            "absolute difference / eV Å⁻¹": result.net_force_max_abs_eV_A,
            "allowed / eV Å⁻¹": settings.net_force_atol_eV_A,
        },
        {
            "check": "Official force vs energy derivative",
            "coordinate": coordinate,
            "value / eV Å⁻¹": result.official_finite_difference_force_eV_A,
            "reference / eV Å⁻¹": result.official_analytic_force_eV_A,
            "absolute difference / eV Å⁻¹": (
                result.official_finite_difference_error_eV_A
            ),
            "allowed / eV Å⁻¹": finite_difference_limit,
        },
        {
            "check": "Toolkit pipeline vs official calculator",
            "coordinate": coordinate,
            "value / eV Å⁻¹": result.toolkit_force_eV_A,
            "reference / eV Å⁻¹": result.official_analytic_force_eV_A,
            "absolute difference / eV Å⁻¹": result.toolkit_official_error_eV_A,
            "allowed / eV Å⁻¹": settings.toolkit_official_force_atol_eV_A,
        },
    ]
    table = pd.DataFrame(rows)
    table["passed"] = (
        table["absolute difference / eV Å⁻¹"] <= table["allowed / eV Å⁻¹"]
    )
    return table


def nci_force_check_record(
    result: NCIForceCheck,
    settings: NCIValidationSettings,
) -> dict[str, int | float | str]:
    """Return the force-check values and calculation routes for saved results."""

    if not isinstance(result, NCIForceCheck):
        raise TypeError("result must be an NCIForceCheck instance")
    if not isinstance(settings, NCIValidationSettings):
        raise TypeError("settings must be an NCIValidationSettings instance")
    return {
        "atom_index": result.atom_index,
        "cartesian_axis": result.axis,
        "finite_difference_step_A": settings.finite_difference_step_A,
        "official_total_energy_route": (
            "official AIMNet2Calculator complete total energy with simple Coulomb "
            "and D3"
        ),
        "official_analytic_force_route": (
            "official AIMNet2Calculator complete-model force"
        ),
        "toolkit_analytic_force_route": (
            "Toolkit PipelineModelWrapper complete-model force"
        ),
        "official_analytic_force_eV_A": result.official_analytic_force_eV_A,
        "official_finite_difference_force_eV_A": (
            result.official_finite_difference_force_eV_A
        ),
        "toolkit_analytic_force_eV_A": result.toolkit_force_eV_A,
        "official_analytic_vs_official_finite_difference_abs_error_eV_A": (
            result.official_finite_difference_error_eV_A
        ),
        "toolkit_analytic_vs_official_analytic_abs_error_eV_A": (
            result.toolkit_official_error_eV_A
        ),
    }


__all__ = [
    "NCIForceCheck",
    "build_nci_force_check_table",
    "check_nci_force",
    "nci_force_check_record",
]
