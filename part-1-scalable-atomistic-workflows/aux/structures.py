"""Deterministic ASE structures used by the water IR tutorial.

This module deliberately stops at ASE ``Atoms`` objects. Converting structures
to Toolkit ``AtomicData`` and assembling a ``Batch`` are learner-facing steps
and belong in the notebook.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from ase import Atoms
from ase.build import molecule


HYDROGEN_MASS_U = 1.00782503223
DEUTERIUM_MASS_U = 2.01410177812
OXYGEN16_MASS_U = 15.99491461957


def make_water_monomer() -> Atoms:
    """Return a centered, nonperiodic ASE water monomer."""

    atoms = molecule("H2O")
    atoms.center(about=(0.0, 0.0, 0.0))
    atoms.set_pbc(False)
    return atoms


def make_water_dimer(oo_distance: float = 2.90) -> Atoms:
    """Return a deterministic near-linear hydrogen-bonded water dimer.

    One donor O-H bond points at the acceptor oxygen. The acceptor molecular
    bisector points away from the donor. This is an illustrative scan seed,
    not a claimed optimized reference structure.
    """

    if oo_distance <= 1.5:
        raise ValueError("oo_distance must be greater than 1.5 Angstrom")

    donor = molecule("H2O")
    donor.translate(-donor.positions[0])
    donor_oh = donor.positions[1] - donor.positions[0]
    donor.rotate(donor_oh, (1.0, 0.0, 0.0), center=donor.positions[0])

    acceptor = molecule("H2O")
    acceptor.translate(-acceptor.positions[0])
    acceptor_oh = acceptor.positions[1:] - acceptor.positions[0]
    acceptor_bisector = np.sum(
        acceptor_oh / np.linalg.norm(acceptor_oh, axis=1)[:, None], axis=0
    )
    acceptor.rotate(
        acceptor_bisector,
        (1.0, 0.0, 0.0),
        center=acceptor.positions[0],
    )
    acceptor.translate((float(oo_distance), 0.0, 0.0))

    atoms = donor + acceptor
    atoms.center(about=(0.0, 0.0, 0.0))
    atoms.set_pbc(False)
    atoms.info["seed"] = "near-linear hydrogen-bonded water dimer"
    atoms.info["oo_distance_angstrom"] = float(oo_distance)
    return atoms


def make_water_dimer_scan(oo_distances: Iterable[float]) -> list[Atoms]:
    """Build one independent dimer for every requested O-O separation."""

    distances = [float(distance) for distance in oo_distances]
    if not distances:
        raise ValueError("oo_distances must contain at least one separation")
    return [make_water_dimer(distance) for distance in distances]


def make_cyclic_water_hexamer(
    oo_distance: float = 2.78,
    oh_distance: float = 0.97,
    hoh_angle_deg: float = 104.5,
) -> Atoms:
    """Construct a deterministic hydrogen-bonded cyclic hexamer seed.

    Each water donates one in-plane hydrogen to the next oxygen. This seed is
    intended for FIRE2 relaxation before production dynamics; it is not
    presented as a reference optimized geometry.
    """

    if oo_distance <= 0.0 or oh_distance <= 0.0:
        raise ValueError("O-O and O-H distances must be positive")
    if not 0.0 < hoh_angle_deg < 180.0:
        raise ValueError("hoh_angle_deg must be between 0 and 180 degrees")

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    radius = float(oo_distance)
    theta = np.deg2rad(hoh_angle_deg)
    rotation = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0.0],
            [np.sin(-theta), np.cos(-theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    oxygens = np.array(
        [
            [
                radius * np.cos(2.0 * np.pi * index / 6.0),
                radius * np.sin(2.0 * np.pi * index / 6.0),
                0.08 * (-1.0 if index % 2 else 1.0),
            ]
            for index in range(6)
        ]
    )

    for index, oxygen in enumerate(oxygens):
        donor = oxygens[(index + 1) % 6] - oxygen
        donor[2] = 0.0
        donor /= np.linalg.norm(donor)
        free = rotation @ donor
        symbols.extend(["O", "H", "H"])
        positions.extend(
            [oxygen, oxygen + oh_distance * donor, oxygen + oh_distance * free]
        )

    atoms = Atoms(symbols=symbols, positions=np.asarray(positions))
    atoms.center(about=(0.0, 0.0, 0.0))
    atoms.set_pbc(False)
    atoms.info["seed"] = "generated cyclic water hexamer"
    return atoms


def set_water_isotope_masses(atoms: Atoms, hydrogen_mass_u: float) -> None:
    """Set explicit H/D and O-16 masses without changing atomic numbers."""

    if hydrogen_mass_u <= 0.0:
        raise ValueError("hydrogen_mass_u must be positive")
    if not np.isin(atoms.numbers, (1, 8)).all():
        raise ValueError("set_water_isotope_masses accepts only water structures")
    masses = np.where(atoms.numbers == 1, hydrogen_mass_u, OXYGEN16_MASS_U)
    atoms.set_masses(masses)


def make_ir_structures(
    *, hexamer: Atoms | None = None
) -> tuple[list[Atoms], list[str]]:
    """Return ``H2O, D2O, (H2O)6, (D2O)6`` as mass-only ASE pairs."""

    monomer = make_water_monomer()
    hexamer_seed = make_cyclic_water_hexamer() if hexamer is None else hexamer.copy()
    hexamer_seed.center(about=(0.0, 0.0, 0.0))
    hexamer_seed.set_pbc(False)

    structures = [
        monomer.copy(),
        monomer.copy(),
        hexamer_seed.copy(),
        hexamer_seed.copy(),
    ]
    for atoms, hydrogen_mass in zip(
        structures,
        (
            HYDROGEN_MASS_U,
            DEUTERIUM_MASS_U,
            HYDROGEN_MASS_U,
            DEUTERIUM_MASS_U,
        ),
        strict=True,
    ):
        set_water_isotope_masses(atoms, hydrogen_mass)

    return structures, ["H2O", "D2O", "(H2O)6", "(D2O)6"]
