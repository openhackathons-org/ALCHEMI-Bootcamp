"""Pydantic models and conversion utilities for the Toolkit tutorial."""

from typing import List, Optional

import ase
import numpy as np
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Physical constants (canonical definitions in constants.py)
# ---------------------------------------------------------------------------
from .constants import BOLTZ_EV_K, KE_CONV, P_CONV  # noqa: F401

# ---------------------------------------------------------------------------
# Geometry-relaxation models
# ---------------------------------------------------------------------------


class AtomicStructurePayload(BaseModel):
    """Atomic structure payload used by Toolkit relaxation helpers."""

    coord: List[float]
    numbers: List[int]
    charge: Optional[int] = 0
    mult: Optional[int] = 1
    cell: Optional[List[float]] = None
    pbc: Optional[List[bool]] = None
    structure_id: Optional[str] = None
    active_mask: Optional[List[bool]] = None


class RelaxationRequest(BaseModel):
    """Request model for batch geometry relaxation."""

    atoms: List[AtomicStructurePayload]
    opttol: Optional[float] = None
    opttol_pressure: Optional[float] = None
    cellopt: bool = False
    info: str = ""


class OptimizationResult(AtomicStructurePayload):
    """Result of a single geometry optimisation.

    Older cached JSON responses may store numeric fields as strings.
    Pydantic's float/list[float] validation coerces them back to floats so the
    Toolkit-side analysis sees plain Python numbers.
    """

    converged: bool
    optimizer_nsteps: int = Field(
        default=0,
        description="Number of optimiser steps taken.",
    )
    energy: float
    forces: List[float]
    stress: Optional[List[float]] = None
    charges: Optional[List[float]] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_step_count(cls, data):
        if isinstance(data, dict):
            if "optimizer_nsteps" not in data and "num_optimization_steps" in data:
                data = {**data, "optimizer_nsteps": data["num_optimization_steps"]}
        return data

    @property
    def num_optimization_steps(self) -> int:
        """Back-compat alias for older notebook code that used the long name."""
        return self.optimizer_nsteps


class RelaxationBatchResult(BaseModel):
    """Batch geometry-relaxation result."""

    atoms: List[OptimizationResult]
    info: Optional[str] = ""
    status: Optional[str] = "Success"


# ---------------------------------------------------------------------------
# ASE <-> model conversion utilities
# ---------------------------------------------------------------------------


def ase_to_atomic_data(
    atoms: ase.Atoms,
    structure_id: str | None = None,
    active_mask: list[bool] | None = None,
) -> AtomicStructurePayload:
    """Convert an ASE Atoms object to a Toolkit relaxation payload."""
    data = AtomicStructurePayload(
        coord=atoms.positions.flatten().tolist(),
        numbers=atoms.numbers.tolist(),
        charge=atoms.info.get("charge", 0),
        mult=atoms.info.get("mult", 1),
        structure_id=structure_id,
    )
    if atoms.cell.volume > 0:
        data.cell = atoms.cell.array.flatten().tolist()
        data.pbc = atoms.pbc.tolist()
    if active_mask is not None:
        data.active_mask = active_mask
    return data


def atomic_data_to_ase(atomic_data: AtomicStructurePayload) -> ase.Atoms:
    """Convert a relaxation payload/result to ASE Atoms."""
    atoms = ase.Atoms(
        positions=np.array(atomic_data.coord).reshape(-1, 3),
        numbers=atomic_data.numbers,
    )
    atoms.info["charge"] = atomic_data.charge
    atoms.info["mult"] = atomic_data.mult
    if atomic_data.cell is not None:
        atoms.set_cell(np.array(atomic_data.cell).reshape(3, 3))
    if atomic_data.pbc is not None:
        atoms.set_pbc(atomic_data.pbc)
    if isinstance(atomic_data, OptimizationResult):
        if atomic_data.charges is not None:
            atoms.set_initial_charges(atomic_data.charges)
        if atomic_data.stress is not None:
            atoms.info["stress"] = np.array(atomic_data.stress)
        if atomic_data.energy is not None:
            atoms.info["energy"] = np.array(atomic_data.energy)
        if atomic_data.forces is not None:
            atoms.info["forces"] = np.array(atomic_data.forces).reshape(-1, 3)
    return atoms
