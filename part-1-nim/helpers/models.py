"""Pydantic models and utility functions for ALCHEMI BMD/BGR endpoints."""

from typing import List, Optional

import ase
import ase.io
import numpy as np
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

# ---------------------------------------------------------------------------
# Physical constants (canonical definitions in constants.py)
# ---------------------------------------------------------------------------
from .constants import BOLTZ_EV_K, KE_CONV, P_CONV  # noqa: F401

# ---------------------------------------------------------------------------
# BMD models
# ---------------------------------------------------------------------------


class BMDAtomicData(BaseModel):
    """Atomic data for molecular dynamics simulation."""

    coord: List[float]
    numbers: List[int]
    charge: Optional[int] = 0
    mult: Optional[PositiveInt] = 1
    cell: Optional[List[float]] = None
    pbc: Optional[List[bool]] = None
    structure_id: Optional[str] = None
    mass: Optional[List[float]] = None
    velocity: Optional[List[float]] = None


class BMDConfig(BaseModel):
    """Configuration for molecular dynamics simulations."""

    temperature: PositiveFloat = Field(default=300.0, description="Temperature in K")
    dt: PositiveFloat = Field(default=1.0, description="Time step in fs")
    nvt: bool = Field(default=True, description="Enable stochastic Langevin thermostat")
    friction: PositiveFloat = Field(
        default=1.0, description="Friction coefficient for Langevin thermostat in ps^-1"
    )
    npt: bool = Field(default=False, description="Enable Monte Carlo barostat")
    barostat_prob_shear: NonNegativeFloat = Field(
        default=0.5, description="Probability of shear move in barostat"
    )
    barostat_shear_max: PositiveFloat = Field(
        default=1.0e-3, description="Maximum shear move in barostat"
    )
    barostat_diag_max: PositiveFloat = Field(
        default=3.0e-3, description="Maximum diagonal move in barostat"
    )
    barostat_log_scale_max: PositiveFloat = Field(
        default=5.0e-3, description="Maximum log scale move in barostat"
    )
    pressure: NonNegativeFloat = Field(
        default=0.0, description="External pressure in kBar"
    )
    barostat_every: PositiveInt = Field(
        default=25, description="Frequency of barostat updates"
    )
    barostat_anisotropic: bool = Field(
        default=False, description="Enable anisotropic barostat"
    )
    istep: NonNegativeInt = Field(default=0, description="Simulation step")
    md_time: NonNegativeFloat = Field(
        default=0.0, description="Current simulation time in ps"
    )
    md_time_max: NonNegativeFloat = Field(
        default=10.0, description="Maximum simulation time in ps"
    )
    save_interval: PositiveInt = Field(
        default=100, description="Frequency of trajectory saves in steps"
    )
    efield: List[float] = Field(
        default=[0.0, 0.0, 0.0], description="External electric field vector in uV/A"
    )


class BMDRequest(BaseModel):
    """Request model for MD simulation."""

    atoms: BMDAtomicData
    config: Optional[BMDConfig] = None
    info: Optional[str] = None


class BMDSnapshot(BaseModel):
    """Snapshot of MD simulation state."""

    coord: List[float]
    velocity: List[float]
    energy: float
    charges: Optional[List[float]] = None
    cell: Optional[List[float]] = None
    stress: Optional[List[float]] = None
    istep: Optional[NonNegativeInt] = 0
    md_time: Optional[NonNegativeFloat] = 0.0


class BMDReply(BaseModel):
    """Reply model for MD simulation results."""

    trajectory: Optional[List[BMDSnapshot]] = None
    config: BMDConfig
    status: Optional[str] = "Success"
    info: Optional[str] = None


# ---------------------------------------------------------------------------
# BGR models
# ---------------------------------------------------------------------------


class BGRAtomicData(BaseModel):
    """Atomic data for geometry optimisation."""

    coord: List[float]
    numbers: List[int]
    charge: Optional[int] = 0
    mult: Optional[int] = 1
    cell: Optional[List[float]] = None
    pbc: Optional[List[bool]] = None
    structure_id: Optional[str] = None
    active_mask: Optional[List[bool]] = None


class BGRRequest(BaseModel):
    """Request model for batch geometry relaxation."""

    atoms: List[BGRAtomicData]
    opttol: Optional[float] = None
    opttol_pressure: Optional[float] = None
    cellopt: bool = False
    info: str = ""


class OptimizationResult(BGRAtomicData):
    """Result of a single geometry optimisation.

    BGR NIM 1.0.0 returns numeric fields (coord, cell, forces, stress) as
    strings; pydantic's float/list[float] validation coerces them back to
    floats automatically, so the client side sees plain Python floats.
    """

    converged: bool
    optimizer_nsteps: int = Field(
        default=0,
        description="Number of optimiser steps taken. NIM 1.0.0 returns this "
        "as 'optimizer_nsteps'; earlier prototypes called it "
        "'num_optimization_steps'.",
    )
    energy: float
    forces: List[float]
    stress: Optional[List[float]] = None
    charges: Optional[List[float]] = None

    @property
    def num_optimization_steps(self) -> int:
        """Back-compat alias for older notebook code that used the long name."""
        return self.optimizer_nsteps


class BGRReply(BaseModel):
    """Reply model for batch geometry relaxation results."""

    atoms: List[OptimizationResult]
    info: Optional[str] = ""
    status: Optional[str] = "Success"


# ---------------------------------------------------------------------------
# ASE <-> model conversion utilities
# ---------------------------------------------------------------------------


def read_to_bmd_atomic_data(structure_file: str) -> BMDAtomicData:
    """Read atomic structure from file using ASE and return BMDAtomicData."""
    result = ase.io.read(structure_file)
    assert isinstance(result, ase.Atoms), (
        f"Expected single ase.Atoms, got {type(result)}"
    )
    atoms = result
    data = {
        "coord": atoms.positions.flatten().tolist(),
        "numbers": atoms.numbers.tolist(),
        "charge": atoms.info.get("charge", 0),
        "mult": atoms.info.get("mult", 1),
    }
    if atoms.cell.volume > 0:
        data["cell"] = atoms.cell.array.flatten().tolist()
        data["pbc"] = atoms.get_pbc().tolist()

    for k in ("mass", "velocity"):
        if k in atoms.arrays:
            data[k] = atoms.arrays[k].flatten().tolist()

    return BMDAtomicData(**data)


def ase_to_atomic_data(
    atoms: ase.Atoms,
    structure_id: str | None = None,
    active_mask: list[bool] | None = None,
) -> BGRAtomicData:
    """Convert an ASE Atoms object to an BGRAtomicData instance."""
    data = BGRAtomicData(
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


def atomic_data_to_ase(atomic_data: BGRAtomicData) -> ase.Atoms:
    """Convert BGRAtomicData / OptimizationResult to ASE Atoms."""
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


def ase_to_md_atomic_data(atoms: ase.Atoms) -> BMDAtomicData:
    """Convert an ASE Atoms object to a BMDAtomicData instance."""
    data = {
        "coord": atoms.positions.flatten().tolist(),
        "numbers": atoms.numbers.tolist(),
        "charge": atoms.info.get("charge", 0),
        "mult": atoms.info.get("mult", 1),
    }
    if atoms.cell.volume > 0:
        data["cell"] = atoms.cell.array.flatten().tolist()
        data["pbc"] = atoms.pbc.tolist()
    if "mass" in atoms.arrays:
        data["mass"] = atoms.arrays["mass"].flatten().tolist()
    if hasattr(atoms, "get_velocities"):
        vel = atoms.get_velocities()
        if vel is not None and np.any(vel):
            data["velocity"] = vel.flatten().tolist()
    return BMDAtomicData(**data)
