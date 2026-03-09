"""Pydantic models and utility functions for ALCHEMI BMD/BGR endpoints.

Adapted from examples/bmd_script.py and examples/bgr_script.py.
"""

from typing import List, Optional

import ase
import ase.data
import ase.io
import numpy as np
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KE_CONV = 103.64269667160806       # amu*A^2/fs^2 -> eV
BOLTZ_EV_K = 8.617333262145179e-05  # eV/K
P_CONV = 1.602176634e6             # eV/A^3 -> Bar

# ---------------------------------------------------------------------------
# BMD models (from bmd_script.py)
# ---------------------------------------------------------------------------

class MDAtomicData(BaseModel):
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


class MDConfig(BaseModel):
    """Configuration for molecular dynamics simulations."""
    temperature: PositiveFloat = Field(default=300.0, description="Temperature in K")
    dt: PositiveFloat = Field(default=1.0, description="Time step in fs")
    nvt: bool = Field(default=True, description="Enable stochastic Langevin thermostat")
    friction: PositiveFloat = Field(default=1.0, description="Friction coefficient for Langevin thermostat in ps^-1")
    npt: bool = Field(default=False, description="Enable Monte Carlo barostat")
    barostat_prob_shear: NonNegativeFloat = Field(default=0.5, description="Probability of shear move in barostat")
    barostat_shear_max: PositiveFloat = Field(default=1.0e-3, description="Maximum shear move in barostat")
    barostat_diag_max: PositiveFloat = Field(default=3.0e-3, description="Maximum diagonal move in barostat")
    barostat_log_scale_max: PositiveFloat = Field(default=5.0e-3, description="Maximum log scale move in barostat")
    pressure: NonNegativeFloat = Field(default=0.0, description="External pressure in kBar")
    barostat_every: PositiveInt = Field(default=25, description="Frequency of barostat updates")
    barostat_anisotropic: bool = Field(default=False, description="Enable anisotropic barostat")
    istep: NonNegativeInt = Field(default=0, description="Simulation step")
    md_time: NonNegativeFloat = Field(default=0.0, description="Current simulation time in ps")
    md_time_max: NonNegativeFloat = Field(default=10.0, description="Maximum simulation time in ps")
    save_interval: PositiveInt = Field(default=100, description="Frequency of trajectory saves in steps")
    efield: List[float] = Field(default=[0.0, 0.0, 0.0], description="External electric field vector in uV/A")


class MDRequest(BaseModel):
    """Request model for MD simulation."""
    atoms: MDAtomicData
    config: Optional[MDConfig] = None
    info: Optional[str] = None


class MDSnapshot(BaseModel):
    """Snapshot of MD simulation state."""
    coord: List[float]
    velocity: List[float]
    energy: float
    charges: Optional[List[float]] = None
    cell: Optional[List[float]] = None
    stress: Optional[List[float]] = None
    istep: Optional[NonNegativeInt] = 0
    md_time: Optional[NonNegativeFloat] = 0.0


class MDReply(BaseModel):
    """Reply model for MD simulation results."""
    trajectory: Optional[List[MDSnapshot]] = None
    config: MDConfig
    status: Optional[str] = "Success"
    info: Optional[str] = None

# ---------------------------------------------------------------------------
# BGR models (from bgr_script.py)
# ---------------------------------------------------------------------------

class AtomicData(BaseModel):
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
    atoms: List[AtomicData]
    opttol: Optional[float] = None
    opttol_pressure: Optional[float] = None
    cellopt: bool = False
    info: str = ""


class OptimizationResult(AtomicData):
    """Result of a single geometry optimisation."""
    converged: bool
    num_optimization_steps: int
    energy: float
    forces: List[float]
    stress: Optional[List[float]] = None
    charges: Optional[List[float]] = None


class BGRReply(BaseModel):
    """Reply model for batch geometry relaxation results."""
    atoms: List[OptimizationResult]
    info: Optional[str] = ""
    status: Optional[str] = "Success"

# ---------------------------------------------------------------------------
# ASE <-> model conversion utilities
# ---------------------------------------------------------------------------

def read_structure(structure_file: str) -> MDAtomicData:
    """Read atomic structure from file using ASE and return MDAtomicData."""
    atoms = ase.io.read(structure_file)
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

    return MDAtomicData(**data)


def ase_to_atomic_data(atoms: ase.Atoms, _id: str | None = None) -> AtomicData:
    """Convert an ASE Atoms object to an AtomicData instance."""
    data = AtomicData(
        coord=atoms.positions.flatten().tolist(),
        numbers=atoms.numbers.tolist(),
        charge=atoms.info.get("charge", 0),
        mult=atoms.info.get("mult", 1),
        structure_id=_id,
    )
    if atoms.cell.volume > 0:
        data.cell = atoms.cell.array.flatten().tolist()
        data.pbc = atoms.pbc.tolist()
    return data


def atomic_data_to_ase(atomic_data) -> ase.Atoms:
    """Convert AtomicData / OptimizationResult to ASE Atoms."""
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
    if hasattr(atomic_data, "charges") and atomic_data.charges is not None:
        atoms.set_initial_charges(atomic_data.charges)
    if hasattr(atomic_data, "stress") and atomic_data.stress is not None:
        atoms.info["stress"] = np.array(atomic_data.stress)
    if hasattr(atomic_data, "energy") and atomic_data.energy is not None:
        atoms.info["energy"] = atomic_data.energy
    if hasattr(atomic_data, "forces") and atomic_data.forces is not None:
        atoms.arrays["forces"] = np.array(atomic_data.forces).reshape(-1, 3)
    return atoms


def ase_to_md_atomic_data(atoms: ase.Atoms) -> MDAtomicData:
    """Convert an ASE Atoms object to an MDAtomicData instance."""
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
    return MDAtomicData(**data)
