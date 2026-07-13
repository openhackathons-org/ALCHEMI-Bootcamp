#!/usr/bin/env python3
"""Generate inspectable B97-3c double-harmonic IR references with Psi4 1.11.

The Hessian is formed explicitly from central differences of full Cartesian
gradients.  Dipole derivatives are formed from the dipoles evaluated at the
same displaced geometries.  One electronic calculation therefore supports
both H and D spectra: isotope substitution changes masses only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHOD = "b97-3c"
BASIS = "def2-mtzvp"
MODEL_CHEMISTRY = f"{METHOD}/{BASIS}"
BOHR_TO_ANGSTROM = 0.529177210903
HYDROGEN_MASS_U = 1.00782503223
DEUTERIUM_MASS_U = 2.01410177812
OXYGEN16_MASS_U = 15.99491461957

# Cordero-style single-bond radii, used only for a topology diagnostic.  The
# electronic calculation does not depend on this table.
COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


@dataclass(frozen=True)
class Geometry:
    """A finite, ordered Cartesian structure."""

    symbols: tuple[str, ...]
    positions_angstrom: np.ndarray
    label: str
    comment: str = ""

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_angstrom, dtype=float)
        if positions.shape != (len(self.symbols), 3):
            raise ValueError(
                f"positions must have shape ({len(self.symbols)}, 3), got "
                f"{positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("positions contain a non-finite value")
        object.__setattr__(self, "positions_angstrom", positions.copy())


def center_on_bounding_box(positions: np.ndarray) -> np.ndarray:
    """Match ``ASE Atoms.center(about=(0, 0, 0))`` for a finite structure."""

    positions = np.asarray(positions, dtype=float)
    return positions - 0.5 * (positions.min(axis=0) + positions.max(axis=0))


def make_h2o_seed() -> Geometry:
    """Return the ASE G2 H2O seed used by ``ase.build.molecule('H2O')``."""

    symbols = ("O", "H", "H")
    positions = np.array(
        [
            [0.0, 0.0, 0.119262],
            [0.0, 0.763239, -0.477047],
            [0.0, -0.763239, -0.477047],
        ]
    )
    return Geometry(
        symbols,
        center_on_bounding_box(positions),
        label="h2o",
        comment="ASE G2 H2O seed; optimized before finite differences",
    )


def make_cyclic_h6_seed(
    oo_distance: float = 2.78,
    oh_distance: float = 0.97,
    hoh_angle_deg: float = 104.5,
) -> Geometry:
    """Match ``helpers.ir.make_cyclic_water_hexamer`` without importing ASE."""

    symbols: list[str] = []
    positions: list[np.ndarray] = []
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
                oo_distance * np.cos(2.0 * np.pi * index / 6.0),
                oo_distance * np.sin(2.0 * np.pi * index / 6.0),
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

    return Geometry(
        tuple(symbols),
        center_on_bounding_box(np.asarray(positions)),
        label="cyclic-h6-seed",
        comment=(
            "Deterministic cyclic (H2O)6 seed from helpers/ir.py; this is not "
            "a published or pre-optimized isomer"
        ),
    )


def read_xyz(path: str | Path) -> Geometry:
    """Read exactly one ordinary XYZ frame, preserving atom order."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: expected an XYZ atom-count and comment line")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path}: first line is not an integer atom count") from exc
    if natoms <= 0:
        raise ValueError(f"{path}: atom count must be positive")
    if len(lines) < natoms + 2:
        raise ValueError(f"{path}: expected {natoms} coordinate lines")

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line_number, line in enumerate(lines[2 : natoms + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: expected symbol x y z")
        symbol = fields[0][0].upper() + fields[0][1:].lower()
        try:
            xyz = [float(value) for value in fields[1:4]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid coordinate") from exc
        symbols.append(symbol)
        positions.append(xyz)

    trailing = [line for line in lines[natoms + 2 :] if line.strip()]
    if trailing:
        raise ValueError(
            f"{path}: contains data after the first frame; provide a single-frame XYZ"
        )
    return Geometry(
        tuple(symbols),
        np.asarray(positions),
        label=path.stem,
        comment=lines[1].strip(),
    )


def write_xyz(path: str | Path, geometry: Geometry, comment: str | None = None) -> None:
    path = Path(path)
    text = [str(len(geometry.symbols)), comment or geometry.comment or geometry.label]
    for symbol, xyz in zip(
        geometry.symbols, geometry.positions_angstrom, strict=True
    ):
        text.append(f"{symbol:<2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_xyz_trajectory(
    path: str | Path,
    symbols: Sequence[str],
    coordinates_bohr: Sequence[np.ndarray],
    energies_Eh: Sequence[float],
) -> None:
    """Write an inspectable multi-frame XYZ optimization trajectory."""

    blocks: list[str] = []
    for step, (coordinates, energy) in enumerate(
        zip(coordinates_bohr, energies_Eh, strict=True)
    ):
        geometry = Geometry(
            tuple(symbols),
            np.asarray(coordinates, dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM,
            label=f"optimization-step-{step}",
        )
        lines = [
            str(len(symbols)),
            f"step={step} energy_Eh={float(energy):.16f} units=angstrom",
        ]
        for symbol, xyz in zip(symbols, geometry.positions_angstrom, strict=True):
            lines.append(
                f"{symbol:<2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}"
            )
        blocks.append("\n".join(lines))
    Path(path).write_text("\n".join(blocks) + "\n", encoding="utf-8")


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def infer_covalent_bonds(
    symbols: Sequence[str], positions_angstrom: np.ndarray
) -> tuple[list[dict[str, Any]], list[str]]:
    """Infer a diagnostic bond graph; this graph never affects the calculation."""

    distances = pairwise_distances(positions_angstrom)
    unsupported = sorted(set(symbols).difference(COVALENT_RADII_ANGSTROM))
    bonds: list[dict[str, Any]] = []
    for i in range(len(symbols)):
        radius_i = COVALENT_RADII_ANGSTROM.get(symbols[i])
        if radius_i is None:
            continue
        for j in range(i + 1, len(symbols)):
            radius_j = COVALENT_RADII_ANGSTROM.get(symbols[j])
            if radius_j is None:
                continue
            threshold = 1.25 * (radius_i + radius_j) + 0.15
            if distances[i, j] <= threshold:
                bonds.append(
                    {
                        "i": i,
                        "j": j,
                        "elements": f"{symbols[i]}-{symbols[j]}",
                        "distance_angstrom": float(distances[i, j]),
                        "threshold_angstrom": float(threshold),
                    }
                )
    return bonds, unsupported


def _angle_degrees(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    left = a - vertex
    right = c - vertex
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def infer_water_hydrogen_bonds(
    symbols: Sequence[str], positions_angstrom: np.ndarray
) -> dict[str, Any] | None:
    """Describe O-H ownership and conventional O-H...O contacts for H/O systems."""

    if not set(symbols).issubset({"H", "O"}):
        return None
    oxygen = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    hydrogen = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    distances = pairwise_distances(positions_angstrom)

    owners: dict[int, int] = {}
    unassigned: list[int] = []
    for h in hydrogen:
        candidates = [(float(distances[h, o]), o) for o in oxygen]
        distance, owner = min(candidates, default=(math.inf, -1))
        if distance <= 1.30:
            owners[h] = owner
        else:
            unassigned.append(h)

    owner_counts = {str(o): sum(owner == o for owner in owners.values()) for o in oxygen}
    hydrogen_bonds: list[dict[str, Any]] = []
    for h, donor in owners.items():
        for acceptor in oxygen:
            if acceptor == donor:
                continue
            ho = float(distances[h, acceptor])
            oo = float(distances[donor, acceptor])
            angle = _angle_degrees(
                positions_angstrom[donor],
                positions_angstrom[h],
                positions_angstrom[acceptor],
            )
            if ho <= 2.50 and oo <= 3.50 and angle >= 140.0:
                hydrogen_bonds.append(
                    {
                        "donor_o": donor,
                        "hydrogen": h,
                        "acceptor_o": acceptor,
                        "h_to_acceptor_angstrom": ho,
                        "o_to_o_angstrom": oo,
                        "o_h_o_angle_deg": angle,
                    }
                )

    return {
        "oxygen_count": len(oxygen),
        "hydrogen_count": len(hydrogen),
        "hydrogen_owner_by_atom": {str(h): owner for h, owner in owners.items()},
        "covalent_h_count_by_oxygen": owner_counts,
        "unassigned_hydrogens": unassigned,
        "all_oxygens_have_two_hydrogens": bool(owner_counts)
        and all(count == 2 for count in owner_counts.values()),
        "hydrogen_bonds": hydrogen_bonds,
        "hydrogen_bond_count": len(hydrogen_bonds),
        "hydrogen_bond_definition": {
            "H_to_acceptor_max_angstrom": 2.50,
            "O_to_O_max_angstrom": 3.50,
            "O_H_O_min_angle_deg": 140.0,
        },
    }


def topology_diagnostics(geometry: Geometry) -> dict[str, Any]:
    bonds, unsupported = infer_covalent_bonds(
        geometry.symbols, geometry.positions_angstrom
    )
    distances = pairwise_distances(geometry.positions_angstrom)
    upper = distances[np.triu_indices(len(geometry.symbols), 1)]
    return {
        "label": geometry.label,
        "atom_count": len(geometry.symbols),
        "formula_counts": {
            symbol: geometry.symbols.count(symbol) for symbol in sorted(set(geometry.symbols))
        },
        "minimum_pair_distance_angstrom": float(upper.min()) if upper.size else None,
        "covalent_bonds": bonds,
        "unsupported_elements_for_bond_diagnostic": unsupported,
        "water_network": infer_water_hydrogen_bonds(
            geometry.symbols, geometry.positions_angstrom
        ),
    }


def is_single_water_ring(topology: Mapping[str, Any]) -> bool:
    """Return whether the diagnostic H-bond graph is one directed ring."""

    network = topology.get("water_network")
    if not isinstance(network, Mapping) or int(network.get("oxygen_count", 0)) < 2:
        return False
    oxygen = {int(index) for index in network["covalent_h_count_by_oxygen"]}
    bonds = network.get("hydrogen_bonds", [])
    donor_counts = {index: 0 for index in oxygen}
    acceptor_counts = {index: 0 for index in oxygen}
    outgoing: dict[int, int] = {}
    for bond in bonds:
        donor = int(bond["donor_o"])
        acceptor = int(bond["acceptor_o"])
        if donor not in oxygen or acceptor not in oxygen:
            return False
        donor_counts[donor] += 1
        acceptor_counts[acceptor] += 1
        outgoing[donor] = acceptor
    if any(count != 1 for count in donor_counts.values()) or any(
        count != 1 for count in acceptor_counts.values()
    ):
        return False
    start = min(oxygen)
    visited: set[int] = set()
    current = start
    while current not in visited:
        visited.add(current)
        current = outgoing[current]
    return current == start and visited == oxygen


def compare_topology(
    initial: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    def pairs(record: Mapping[str, Any]) -> set[tuple[int, int]]:
        return {
            (int(bond["i"]), int(bond["j"]))
            for bond in record["covalent_bonds"]
        }

    initial_pairs = pairs(initial)
    optimized_pairs = pairs(optimized)
    initial_water = initial.get("water_network")
    optimized_water = optimized.get("water_network")
    return {
        "covalent_graph_preserved": initial_pairs == optimized_pairs,
        "bonds_lost": sorted([list(pair) for pair in initial_pairs - optimized_pairs]),
        "bonds_gained": sorted([list(pair) for pair in optimized_pairs - initial_pairs]),
        "initial_hydrogen_bond_count": (
            initial_water["hydrogen_bond_count"] if initial_water else None
        ),
        "optimized_hydrogen_bond_count": (
            optimized_water["hydrogen_bond_count"] if optimized_water else None
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_metadata(psi4: Any) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "psi4": str(psi4.__version__),
        "numpy": np.__version__,
        "qcengine": package_version("qcengine"),
        "qcelemental": package_version("qcelemental"),
        "simple_dftd3": package_version("dftd3"),
        "s-dftd3_executable": shutil.which("s-dftd3"),
        "mctc-gcp_executable": shutil.which("mctc-gcp"),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gradient_and_dipole(
    psi4: Any, molecule: Any
) -> tuple[float, np.ndarray, np.ndarray, Any]:
    gradient, wfn = psi4.gradient(
        MODEL_CHEMISTRY,
        molecule=molecule,
        dertype=1,
        return_wfn=True,
    )
    try:
        dipole = np.asarray(wfn.variable("CURRENT DIPOLE"), dtype=float)
    except KeyError:
        psi4.oeprop(wfn, "DIPOLE")
        dipole = np.asarray(wfn.variable("CURRENT DIPOLE"), dtype=float)
    return (
        float(wfn.variable("CURRENT ENERGY")),
        np.asarray(gradient, dtype=float),
        dipole.reshape(3),
        wfn,
    )


def finite_difference_cartesian(
    psi4: Any,
    molecule: Any,
    step_bohr: float,
) -> dict[str, Any]:
    """Evaluate a 3-point Cartesian Hessian and dipole derivative."""

    if step_bohr <= 0.0:
        raise ValueError("finite-difference step must be positive")
    reference_geometry = np.asarray(molecule.geometry(), dtype=float)
    natoms = molecule.natom()
    ndof = 3 * natoms

    reference_energy, reference_gradient, reference_dipole, reference_wfn = (
        _gradient_and_dipole(psi4, molecule)
    )
    gradients_minus = np.empty((ndof, natoms, 3))
    gradients_plus = np.empty((ndof, natoms, 3))
    dipoles_minus = np.empty((ndof, 3))
    dipoles_plus = np.empty((ndof, 3))
    energies_minus = np.empty(ndof)
    energies_plus = np.empty(ndof)

    for coordinate in range(ndof):
        atom, axis = divmod(coordinate, 3)
        for sign, energies, gradients, dipoles in (
            (-1.0, energies_minus, gradients_minus, dipoles_minus),
            (+1.0, energies_plus, gradients_plus, dipoles_plus),
        ):
            displaced = molecule.clone()
            geometry = reference_geometry.copy()
            geometry[atom, axis] += sign * step_bohr
            displaced.set_geometry(psi4.core.Matrix.from_array(geometry))
            displaced.update_geometry()
            energy, gradient, dipole, _ = _gradient_and_dipole(psi4, displaced)
            energies[coordinate] = energy
            gradients[coordinate] = gradient
            dipoles[coordinate] = dipole
        print(
            f"finite difference {coordinate + 1:4d}/{ndof}: "
            f"atom {atom + 1}, axis {'xyz'[axis]}",
            flush=True,
        )

    # H_ij = d g_i / d x_j.  The raw matrix is retained; its symmetric part is
    # used for normal-mode analysis because a physical Cartesian Hessian is
    # symmetric and the antisymmetric component is finite-difference noise.
    hessian_raw = (
        gradients_plus.reshape(ndof, ndof) - gradients_minus.reshape(ndof, ndof)
    ).T / (2.0 * step_bohr)
    hessian_symmetric = 0.5 * (hessian_raw + hessian_raw.T)
    dipole_derivative = (dipoles_plus - dipoles_minus) / (2.0 * step_bohr)

    return {
        "reference_energy_Eh": reference_energy,
        "reference_gradient_Eh_per_bohr": reference_gradient,
        "reference_dipole_au": reference_dipole,
        "reference_wfn": reference_wfn,
        "energies_minus_Eh": energies_minus,
        "energies_plus_Eh": energies_plus,
        "gradients_minus_Eh_per_bohr": gradients_minus,
        "gradients_plus_Eh_per_bohr": gradients_plus,
        "dipoles_minus_au": dipoles_minus,
        "dipoles_plus_au": dipoles_plus,
        "hessian_raw_Eh_per_bohr2": hessian_raw,
        "hessian_symmetric_Eh_per_bohr2": hessian_symmetric,
        "dipole_derivative_3n_by_3_au": dipole_derivative,
    }


def _mass_vector(molecule: Any, deuterated: bool) -> np.ndarray:
    masses = np.asarray([molecule.mass(i) for i in range(molecule.natom())])
    for atom in range(molecule.natom()):
        symbol = molecule.symbol(atom)
        if symbol == "H":
            masses[atom] = DEUTERIUM_MASS_U if deuterated else HYDROGEN_MASS_U
        elif symbol == "O":
            masses[atom] = OXYGEN16_MASS_U
    return masses


def set_reference_masses(molecule: Any, deuterated: bool = False) -> np.ndarray:
    masses = _mass_vector(molecule, deuterated=deuterated)
    for atom, mass in enumerate(masses):
        molecule.set_mass(atom, float(mass))
    return masses


def analyse_modes(
    psi4: Any,
    molecule: Any,
    wfn: Any,
    hessian: np.ndarray,
    dipole_derivative: np.ndarray,
    deuterated: bool,
) -> tuple[dict[str, Any], str, np.ndarray]:
    """Run Psi4's normal-mode analysis with H or D masses, reusing H and dmu/dR."""

    isotope_molecule = molecule.clone()
    masses = set_reference_masses(isotope_molecule, deuterated=deuterated)
    harmonic_analysis = psi4.driver.qcdb.vib.harmonic_analysis
    vibinfo, text = harmonic_analysis(
        np.asarray(hessian),
        np.asarray(isotope_molecule.geometry()),
        masses,
        wfn.basisset(),
        isotope_molecule.irrep_labels(),
        dipder=np.asarray(dipole_derivative).T,
        project_trans=True,
        project_rot=True,
    )
    return vibinfo, text, masses


def vib_data(vibinfo: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(vibinfo[key].data)


def mode_rows(vibinfo: Mapping[str, Any]) -> list[dict[str, Any]]:
    frequencies = vib_data(vibinfo, "omega")
    intensities = (
        vib_data(vibinfo, "IR_intensity")
        if "IR_intensity" in vibinfo
        else np.full(frequencies.shape, np.nan)
    )
    reduced_masses = vib_data(vibinfo, "mu")
    force_constants = vib_data(vibinfo, "k")
    trv = vib_data(vibinfo, "TRV")
    irreps = vib_data(vibinfo, "gamma")
    return [
        {
            "mode_index": index,
            "classification": str(trv[index]),
            "irrep": "" if irreps[index] is None else str(irreps[index]),
            "frequency_cm-1": float(np.real(frequency)),
            "imaginary_frequency_cm-1": float(np.imag(frequency)),
            "ir_intensity_km_mol": float(intensities[index]),
            "reduced_mass_u": float(reduced_masses[index]),
            "force_constant_mDyne_A": float(force_constants[index]),
        }
        for index, frequency in enumerate(frequencies)
    ]


def write_mode_csv(path: str | Path, vibinfo: Mapping[str, Any]) -> None:
    rows = mode_rows(vibinfo)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def significant_imaginary_modes(
    vibinfo: Mapping[str, Any], threshold_cm1: float
) -> list[dict[str, Any]]:
    return [
        row
        for row in mode_rows(vibinfo)
        if row["classification"] == "V"
        and row["imaginary_frequency_cm-1"] > threshold_cm1
    ]


def _mode_arrays(vibinfo: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "frequency_complex_cm-1": vib_data(vibinfo, "omega"),
        "ir_intensity_km_mol": vib_data(vibinfo, "IR_intensity"),
        "mode_mass_weighted_q": vib_data(vibinfo, "q"),
        "mode_cartesian_w": vib_data(vibinfo, "w"),
        "mode_normalized_cartesian_x": vib_data(vibinfo, "x"),
        "reduced_mass_u": vib_data(vibinfo, "mu"),
        "force_constant_mDyne_A": vib_data(vibinfo, "k"),
        "classification": vib_data(vibinfo, "TRV").astype("U8"),
    }


def _bundle_labels(symbols: Sequence[str], source_label: str) -> tuple[str, str]:
    counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
    if counts == {"H": 2, "O": 1}:
        return "h2o", "d2o"
    if counts == {"H": 12, "O": 6}:
        return "h6", "d6"
    safe_label = "".join(
        character.lower() if character.isalnum() else "-" for character in source_label
    ).strip("-")
    safe_label = safe_label or "system"
    return f"{safe_label}-h", f"{safe_label}-d"


def _vibrational_arrays(vibinfo: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    classification = vib_data(vibinfo, "TRV")
    selection = classification == "V"
    omega = vib_data(vibinfo, "omega")[selection]
    frequencies = np.where(
        np.imag(omega) > np.real(omega), -np.imag(omega), np.real(omega)
    ).astype(float)
    intensities = vib_data(vibinfo, "IR_intensity")[selection].astype(float)
    return frequencies, intensities


def write_isotopologue_bundle(
    output_dir: Path,
    *,
    label: str,
    psi4_version: str,
    charge: int,
    multiplicity: int,
    atomic_numbers: np.ndarray,
    geometry_angstrom: np.ndarray,
    masses_u: np.ndarray,
    hessian: np.ndarray,
    dipole_derivative_3n_by_3_au: np.ndarray,
    vibinfo: Mapping[str, Any],
    step_bohr: float,
    imaginary_threshold_cm1: float,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Write the immutable v1 bundle consumed by the IR mapping layer."""

    frequencies, intensities = _vibrational_arrays(vibinfo)
    selection = vib_data(vibinfo, "TRV") == "V"
    mass_weighted_modes = vib_data(vibinfo, "q")[:, selection].T.reshape(
        len(frequencies), len(atomic_numbers), 3
    )
    orthonormal_error = np.max(
        np.abs(
            mass_weighted_modes.reshape(len(frequencies), -1)
            @ mass_weighted_modes.reshape(len(frequencies), -1).T
            - np.eye(len(frequencies))
        )
    )
    if orthonormal_error > 1.0e-8:
        raise RuntimeError(
            f"{label}: mass-weighted mode orthonormality error "
            f"{orthonormal_error:.3e} exceeds 1e-8"
        )
    centered = geometry_angstrom - geometry_angstrom.mean(axis=0)
    nonlinear = np.linalg.matrix_rank(centered, tol=1.0e-10) > 1
    expected_modes = 3 * len(atomic_numbers) - (6 if nonlinear else 5)
    if len(frequencies) != expected_modes:
        raise RuntimeError(
            f"{label}: Psi4 classified {len(frequencies)} vibrational modes; "
            f"expected {expected_modes}"
        )
    dipole_derivative = np.asarray(dipole_derivative_3n_by_3_au, dtype=float)
    if dipole_derivative.shape != (3 * len(atomic_numbers), 3) or not np.all(
        np.isfinite(dipole_derivative)
    ):
        raise RuntimeError(f"{label}: invalid Cartesian dipole derivative")

    bundle_dir = output_dir / "artifacts" / label
    bundle_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = bundle_dir / "ir_arrays.npz"
    np.savez_compressed(
        arrays_path,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        geometry_angstrom=np.asarray(geometry_angstrom, dtype=np.float64),
        masses_u=np.asarray(masses_u, dtype=np.float64),
        hessian_hartree_per_bohr2=np.asarray(hessian, dtype=np.float64),
        dipole_derivative_3n_by_3_au=dipole_derivative,
        frequencies_cm1=np.asarray(frequencies, dtype=np.float64),
        ir_intensities_km_mol=np.asarray(intensities, dtype=np.float64),
        mass_weighted_modes=np.asarray(mass_weighted_modes, dtype=np.float64),
    )
    arrays_hash = sha256_file(arrays_path)
    write_json(
        bundle_dir / "manifest.json",
        {
            "format": {"name": "alchemi.psi4-b97-3c-ir", "version": 1},
            "artifact_id": f"b97-3c-{label}-{arrays_hash[:16]}",
            "engine": {"name": "Psi4", "version": psi4_version},
            "model_chemistry": {
                "method": "B97-3c",
                "basis": "def2-mTZVP",
                "basis_is_explicit_psi4_1_11_workaround": True,
                "finite_difference_points": 3,
                "finite_difference_step_bohr": step_bohr,
                "hessian_source": "central differences of full Cartesian gradients",
            },
            "molecule": {
                "label": label,
                "charge": charge,
                "multiplicity": multiplicity,
            },
            "arrays": {"file": "ir_arrays.npz", "sha256": arrays_hash},
            "units": {
                "geometry": "angstrom",
                "masses": "unified_atomic_mass_unit",
                "hessian": "hartree_per_bohr2",
                "dipole_derivative": "atomic_unit_dipole_per_bohr",
                "frequencies": "cm^-1",
                "ir_intensities": "km_per_mol",
            },
            "validation": dict(validation),
            "provenance": dict(provenance),
            "normal_modes": {
                "array": "mass_weighted_modes",
                "convention": "q_equals_sqrt_mass_times_cartesian",
                "normalization": "orthonormal_rows",
                "ordering": "frequencies_and_ir_intensities",
            },
            "mode_conventions": {
                "translation_projection": True,
                "rotation_projection": True,
                "order": "ascending Psi4 projected vibrational mode order",
                "mass_weighted_modes": (
                    "shape (n_modes,N,3); rows are q=M^1/2*x eigenvectors, "
                    "Euclidean-orthonormal when flattened, and aligned exactly "
                    "with frequencies_cm1 and ir_intensities_km_mol"
                ),
                "mass_weighted_modes_max_orthonormality_error": float(
                    orthonormal_error
                ),
                "imaginary_frequency_encoding": "negative magnitude in cm^-1",
                "imaginary_threshold_cm-1": imaginary_threshold_cm1,
            },
            "reference_scope": (
                "Full canonical B97-3c endpoint; not a termwise decomposition "
                "of the AIMNet residual and external two-body D3."
            ),
        },
    )


def write_manifest(output_dir: Path, excluded: Iterable[str] = ()) -> None:
    excluded_set = set(excluded)
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded_set or relative.startswith("psi4_scratch/"):
            continue
        # Psi4 creates short-lived cleanup sentinels which may disappear after
        # the manifest is written. They are runtime plumbing, not provenance.
        if path.name.startswith("psi.") and path.name.endswith(".clean"):
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(output_dir / "manifest.json", {"files": rows})


def make_psi4_molecule(
    psi4: Any, geometry: Geometry, charge: int, multiplicity: int
) -> Any:
    lines = [f"{charge} {multiplicity}"]
    for symbol, xyz in zip(
        geometry.symbols, geometry.positions_angstrom, strict=True
    ):
        lines.append(f"{symbol} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}")
    lines.extend(["units angstrom", "symmetry c1", "no_com", "no_reorient"])
    molecule = psi4.geometry("\n".join(lines))
    molecule.set_name(geometry.label)
    set_reference_masses(molecule, deuterated=False)
    molecule.update_geometry()
    return molecule


def prepare_output_directory(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_reference(args: argparse.Namespace) -> int:
    try:
        import psi4
    except ImportError as exc:
        raise RuntimeError(
            "Psi4 is not importable. Create the environment in environment.yml "
            "and run this script with that environment's Python."
        ) from exc

    if str(psi4.__version__).split(".")[:2] != ["1", "11"]:
        raise RuntimeError(
            f"this reference is pinned to Psi4 1.11; found {psi4.__version__}"
        )
    if args.xyz is not None:
        geometry = read_xyz(args.xyz)
        source = {"kind": "xyz", "path": str(Path(args.xyz).resolve())}
    elif args.system == "cyclic-h6-seed":
        geometry = make_cyclic_h6_seed()
        source = {"kind": "built-in", "name": args.system}
    else:
        geometry = make_h2o_seed()
        source = {"kind": "built-in", "name": "h2o"}

    output_dir = prepare_output_directory(args.output)
    # Psi4 writes timer.dat to the process working directory at interpreter
    # shutdown. Keep that generated file with the calculation, not beside this
    # source file. All artifact paths below are already absolute.
    os.chdir(output_dir)
    scratch_dir = output_dir / "psi4_scratch"
    scratch_dir.mkdir(parents=False, exist_ok=False)
    psi4.core.IOManager.shared_object().set_default_path(str(scratch_dir))
    psi4.core.set_output_file(str(output_dir / "psi4.out"), False)
    if args.memory is not None:
        psi4.set_memory(args.memory)
    if args.threads is not None:
        psi4.set_num_threads(args.threads)

    psi4.set_options(
        {
            # Psi4 1.11's composite frequency planner does not reliably carry
            # the integrated basis.  Giving the canonical B97-3c basis
            # explicitly preserves the method while avoiding `(auto)`.
            "basis": BASIS,
            "scf_type": args.scf_type,
            "reference": "rhf" if args.multiplicity == 1 else "uhf",
            "dft_radial_points": args.dft_radial_points,
            "dft_spherical_points": args.dft_spherical_points,
            "e_convergence": args.e_convergence,
            "d_convergence": args.d_convergence,
            "g_convergence": args.g_convergence,
            "scf_initial_accelerator": args.scf_initial_accelerator,
        }
    )

    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input_label": geometry.label,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "method": METHOD,
        "basis": BASIS,
        "model_chemistry": MODEL_CHEMISTRY,
        "scratch_directory": str(scratch_dir),
        "basis_workaround": (
            "Psi4 1.11 composite frequency paths can lose the integrated "
            "basis; def2-mTZVP is supplied explicitly. It is the canonical "
            "B97-3c basis, not a method change."
        ),
        "finite_difference": {
            "coordinates": "all 3N Cartesian coordinates",
            "derivative_source": "full analytic composite gradients",
            "points": 3,
            "step_bohr": args.step_bohr,
        },
        "normal_mode_analysis": {
            "translation_projection": True,
            "rotation_projection": True,
            "imaginary_threshold_cm-1": args.imaginary_threshold_cm1,
        },
        "optimization": {
            "engine": args.optimizer,
            "coordinate_system": args.coordsys if args.optimizer == "geometric" else None,
            "restart_policy": (
                "Pass the saved optimization_failed_last.xyz or any prior XYZ "
                "as --xyz and choose a new output directory."
            ),
        },
        "psi4_options": {
            "scf_type": args.scf_type,
            "dft_radial_points": args.dft_radial_points,
            "dft_spherical_points": args.dft_spherical_points,
            "e_convergence": args.e_convergence,
            "d_convergence": args.d_convergence,
            "g_convergence": args.g_convergence,
            "scf_initial_accelerator": args.scf_initial_accelerator,
            "threads": psi4.get_num_threads(),
            "memory": args.memory,
        },
        "dependencies": dependency_metadata(psi4),
        "reference_scope": (
            "Full canonical B97-3c endpoint. Do not interpret it as a direct "
            "decomposition of the AIMNet residual plus external two-body D3."
        ),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_xyz(output_dir / "input.xyz", geometry)

    molecule = make_psi4_molecule(
        psi4, geometry, charge=args.charge, multiplicity=args.multiplicity
    )
    print(f"optimizing {len(geometry.symbols)} atoms at {MODEL_CHEMISTRY}", flush=True)
    optimize_kwargs: dict[str, Any] = {
        "molecule": molecule,
        "return_wfn": True,
        "return_history": True,
        "engine": args.optimizer,
    }
    if args.optimizer == "geometric":
        optimize_kwargs["optimizer_keywords"] = {"coordsys": args.coordsys}
    try:
        optimized_energy, optimized_wfn, optimization_history = psi4.optimize(
            MODEL_CHEMISTRY,
            **optimize_kwargs,
        )
    except Exception as exc:
        last_molecule = (
            exc.wfn.molecule()
            if isinstance(exc, psi4.OptimizationConvergenceError)
            else molecule
        )
        last_geometry = Geometry(
            tuple(last_molecule.symbol(i) for i in range(last_molecule.natom())),
            np.asarray(last_molecule.geometry()) * BOHR_TO_ANGSTROM,
            label=f"{geometry.label}-optimization-failed-last",
            comment=(
                f"Last {args.optimizer} geometry after optimization failure; "
                "restart by passing this file through --xyz"
            ),
        )
        write_xyz(output_dir / "optimization_failed_last.xyz", last_geometry)
        write_json(
            output_dir / "failure.json",
            {
                "stage": "optimization",
                "optimizer": args.optimizer,
                "coordinate_system": (
                    args.coordsys if args.optimizer == "geometric" else None
                ),
                "message": str(exc),
                "restart_xyz": "optimization_failed_last.xyz",
            },
        )
        write_manifest(output_dir, excluded={"manifest.json"})
        raise

    optimization_coordinates = [
        np.asarray(value, dtype=float) for value in optimization_history["coordinates"]
    ]
    optimization_gradients = [
        np.asarray(value, dtype=float) for value in optimization_history["gradient"]
    ]
    optimization_energies = np.asarray(optimization_history["energy"], dtype=float)
    np.savez_compressed(
        output_dir / "optimization_history.npz",
        energy_Eh=optimization_energies,
        coordinates_bohr=np.asarray(optimization_coordinates),
        gradients_Eh_per_bohr=np.asarray(optimization_gradients),
    )
    write_xyz_trajectory(
        output_dir / "optimization_trajectory.xyz",
        geometry.symbols,
        optimization_coordinates,
        optimization_energies,
    )
    optimized_geometry = Geometry(
        geometry.symbols,
        np.asarray(molecule.geometry()) * BOHR_TO_ANGSTROM,
        label=f"{geometry.label}-b97-3c-optimized",
        comment=f"B97-3c/def2-mTZVP optimized; E = {optimized_energy:.16f} Eh",
    )
    write_xyz(output_dir / "optimized.xyz", optimized_geometry)

    print(
        f"forming 3-point Cartesian Hessian at h={args.step_bohr:g} bohr",
        flush=True,
    )
    finite_difference = finite_difference_cartesian(
        psi4, molecule, step_bohr=args.step_bohr
    )
    reference_wfn = finite_difference.pop("reference_wfn")

    h_vib, h_text, h_masses = analyse_modes(
        psi4,
        molecule,
        reference_wfn,
        finite_difference["hessian_symmetric_Eh_per_bohr2"],
        finite_difference["dipole_derivative_3n_by_3_au"],
        deuterated=False,
    )
    d_vib, d_text, d_masses = analyse_modes(
        psi4,
        molecule,
        reference_wfn,
        finite_difference["hessian_symmetric_Eh_per_bohr2"],
        finite_difference["dipole_derivative_3n_by_3_au"],
        deuterated=True,
    )

    np.save(output_dir / "hessian_raw_Eh_per_bohr2.npy", finite_difference["hessian_raw_Eh_per_bohr2"])
    np.save(output_dir / "hessian_symmetric_Eh_per_bohr2.npy", finite_difference["hessian_symmetric_Eh_per_bohr2"])
    np.save(output_dir / "dipole_derivative_3n_by_3_au.npy", finite_difference["dipole_derivative_3n_by_3_au"])
    np.save(output_dir / "reference_gradient_Eh_per_bohr.npy", finite_difference["reference_gradient_Eh_per_bohr"])
    np.savez_compressed(
        output_dir / "finite_difference_samples.npz",
        energies_minus_Eh=finite_difference["energies_minus_Eh"],
        energies_plus_Eh=finite_difference["energies_plus_Eh"],
        gradients_minus_Eh_per_bohr=finite_difference["gradients_minus_Eh_per_bohr"],
        gradients_plus_Eh_per_bohr=finite_difference["gradients_plus_Eh_per_bohr"],
        dipoles_minus_au=finite_difference["dipoles_minus_au"],
        dipoles_plus_au=finite_difference["dipoles_plus_au"],
        step_bohr=np.asarray(args.step_bohr),
    )
    np.savez_compressed(output_dir / "modes_h.npz", **_mode_arrays(h_vib))
    np.savez_compressed(output_dir / "modes_d.npz", **_mode_arrays(d_vib))
    write_mode_csv(output_dir / "modes_h.csv", h_vib)
    write_mode_csv(output_dir / "modes_d.csv", d_vib)
    (output_dir / "harmonic_analysis_h.txt").write_text(h_text + "\n", encoding="utf-8")
    (output_dir / "harmonic_analysis_d.txt").write_text(d_text + "\n", encoding="utf-8")

    h_label, d_label = _bundle_labels(geometry.symbols, geometry.label)
    atomic_numbers = np.asarray(
        [int(round(molecule.Z(atom))) for atom in range(molecule.natom())],
        dtype=np.int32,
    )

    initial_topology = topology_diagnostics(geometry)
    optimized_topology = topology_diagnostics(optimized_geometry)
    gradient = finite_difference["reference_gradient_Eh_per_bohr"]
    h_imaginary = significant_imaginary_modes(h_vib, args.imaginary_threshold_cm1)
    d_imaginary = significant_imaginary_modes(d_vib, args.imaginary_threshold_cm1)
    gradient_rms = float(np.sqrt(np.mean(gradient**2)))
    gradient_max = float(np.max(np.abs(gradient)))
    raw_hessian = finite_difference["hessian_raw_Eh_per_bohr2"]
    symmetric_hessian = finite_difference["hessian_symmetric_Eh_per_bohr2"]
    asymmetry = raw_hessian - raw_hessian.T
    asymmetry_max = float(np.max(np.abs(asymmetry)))
    hessian_scale = max(float(np.max(np.abs(symmetric_hessian))), 1.0e-30)
    asymmetry_relative = asymmetry_max / hessian_scale
    hessian_ready = (
        asymmetry_relative <= args.hessian_antisymmetry_relative_max
    )
    topology_comparison = compare_topology(initial_topology, optimized_topology)
    initial_is_ring = is_single_water_ring(initial_topology)
    optimized_is_ring = is_single_water_ring(optimized_topology)
    water_network = optimized_topology.get("water_network") or {}
    topology_ready = bool(
        topology_comparison["covalent_graph_preserved"]
        and water_network.get("all_oxygens_have_two_hydrogens", False)
        and (h_label != "h6" or (initial_is_ring and optimized_is_ring))
    )
    gradient_ready = gradient_max <= args.reference_gradient_max_abs
    isotope_ready = bool(
        np.all(
            (h_masses == d_masses)
            | np.array([symbol == "H" for symbol in geometry.symbols])
        )
    )
    reference_ready = bool(
        not h_imaginary
        and not d_imaginary
        and gradient_ready
        and hessian_ready
        and topology_ready
        and isotope_ready
    )
    diagnostics = {
        "optimization": {
            "energy_Eh": float(optimized_energy),
            "reference_energy_Eh": finite_difference["reference_energy_Eh"],
            "gradient_rms_Eh_per_bohr": gradient_rms,
            "gradient_max_abs_Eh_per_bohr": gradient_max,
        },
        "finite_difference": {
            "raw_hessian_max_antisymmetry_Eh_per_bohr2": asymmetry_max,
            "raw_hessian_max_antisymmetry_relative": asymmetry_relative,
            "raw_hessian_max_antisymmetry_relative_limit": (
                args.hessian_antisymmetry_relative_max
            ),
            "analysis_uses_symmetric_part": True,
        },
        "minimum": {
            "imaginary_threshold_cm-1": args.imaginary_threshold_cm1,
            "h_significant_imaginary_modes": h_imaginary,
            "d_significant_imaginary_modes": d_imaginary,
            "h_is_minimum_within_threshold": not h_imaginary,
            "d_is_minimum_within_threshold": not d_imaginary,
        },
        "topology": {
            "initial": initial_topology,
            "optimized": optimized_topology,
            "comparison": topology_comparison,
            "initial_is_single_water_ring": initial_is_ring,
            "optimized_is_single_water_ring": optimized_is_ring,
        },
        "isotopes": {
            "h_masses_u": h_masses,
            "d_masses_u": d_masses,
            "same_geometry_hessian_and_dipole_derivative": True,
            "changed_entries_are_hydrogen_masses_only": isotope_ready,
        },
        "consumer_artifact_gate": {
            "status": "passed" if reference_ready else "failed",
            "reference_ready": reference_ready,
            "reference_gradient_max_abs_limit_Eh_per_bohr": (
                args.reference_gradient_max_abs
            ),
            "reference_gradient_within_limit": gradient_ready,
            "hessian_antisymmetry_within_limit": hessian_ready,
            "h_is_minimum_within_threshold": not h_imaginary,
            "d_is_minimum_within_threshold": not d_imaginary,
            "covalent_and_water_topology_ready": topology_ready,
            "mass_only_isotope_pair": isotope_ready,
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    if args.require_minimum and not reference_ready:
        write_manifest(output_dir, excluded={"manifest.json"})
        print(
            "ERROR: calculation did not pass the consumer reference gate; "
            "no tutorial bundles were written. See diagnostics.json.",
            file=sys.stderr,
        )
        return 2

    provenance = {
        "generator": "reference/b97_3c_ir.py",
        "optimizer": args.optimizer,
        "optimizer_coordinate_system": (
            args.coordsys if args.optimizer == "geometric" else None
        ),
        "scf_type": args.scf_type,
        "scf_initial_accelerator": args.scf_initial_accelerator,
        "dft_grid": {
            "radial_points": args.dft_radial_points,
            "spherical_points": args.dft_spherical_points,
        },
        "energy_convergence": args.e_convergence,
        "density_convergence": args.d_convergence,
        "geometry_convergence": args.g_convergence,
        "dispersion": "D3(BJ)-ATM",
        "short_range_basis_correction": "mctc-gcp B97-3c",
    }
    common_validation = {
        "status": "passed" if reference_ready else "failed",
        "reference_ready": reference_ready,
        "gradient_rms_Eh_per_bohr": gradient_rms,
        "gradient_max_abs_Eh_per_bohr": gradient_max,
        "gradient_max_abs_limit_Eh_per_bohr": args.reference_gradient_max_abs,
        "raw_hessian_max_antisymmetry_relative": asymmetry_relative,
        "raw_hessian_max_antisymmetry_relative_limit": (
            args.hessian_antisymmetry_relative_max
        ),
        "covalent_graph_preserved": topology_comparison[
            "covalent_graph_preserved"
        ],
        "optimized_all_oxygens_have_two_hydrogens": water_network.get(
            "all_oxygens_have_two_hydrogens", False
        ),
        "initial_is_single_water_ring": initial_is_ring,
        "optimized_is_single_water_ring": optimized_is_ring,
        "optimized_hydrogen_bond_count": water_network.get(
            "hydrogen_bond_count", 0
        ),
        "same_geometry_hessian_and_dipole_derivative_for_isotopes": True,
        "changed_entries_are_hydrogen_masses_only": isotope_ready,
    }
    common_bundle = {
        "output_dir": output_dir,
        "psi4_version": str(psi4.__version__),
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "atomic_numbers": atomic_numbers,
        "geometry_angstrom": optimized_geometry.positions_angstrom,
        "hessian": finite_difference["hessian_symmetric_Eh_per_bohr2"],
        "dipole_derivative_3n_by_3_au": finite_difference[
            "dipole_derivative_3n_by_3_au"
        ],
        "step_bohr": args.step_bohr,
        "imaginary_threshold_cm1": args.imaginary_threshold_cm1,
        "provenance": provenance,
    }
    write_isotopologue_bundle(
        label=h_label,
        masses_u=h_masses,
        vibinfo=h_vib,
        validation={
            **common_validation,
            "is_minimum_within_threshold": not h_imaginary,
            "significant_imaginary_modes": h_imaginary,
        },
        **common_bundle,
    )
    write_isotopologue_bundle(
        label=d_label,
        masses_u=d_masses,
        vibinfo=d_vib,
        validation={
            **common_validation,
            "is_minimum_within_threshold": not d_imaginary,
            "significant_imaginary_modes": d_imaginary,
        },
        **common_bundle,
    )
    write_manifest(output_dir, excluded={"manifest.json"})

    h_rows = [row for row in mode_rows(h_vib) if row["classification"] == "V"]
    d_rows = [row for row in mode_rows(d_vib) if row["classification"] == "V"]
    print("H vibrational modes (cm^-1, km/mol):")
    for row in h_rows:
        frequency = row["frequency_cm-1"] or -row["imaginary_frequency_cm-1"]
        print(f"  {frequency:10.4f}  {row['ir_intensity_km_mol']:12.6f}")
    print("D vibrational modes (cm^-1, km/mol):")
    for row in d_rows:
        frequency = row["frequency_cm-1"] or -row["imaginary_frequency_cm-1"]
        print(f"  {frequency:10.4f}  {row['ir_intensity_km_mol']:12.6f}")
    print(f"artifacts: {output_dir}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate B97-3c H/D double-harmonic IR reference artifacts."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--system",
        choices=("h2o", "cyclic-h6-seed"),
        default="h2o",
        help="built-in deterministic starting structure (default: h2o)",
    )
    source.add_argument("--xyz", type=Path, help="single-frame input XYZ")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty output directory; existing artifacts are never overwritten",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--step-bohr",
        type=float,
        default=0.005,
        help="central-difference displacement in bohr (default: 0.005)",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--memory", help='Psi4 memory string, for example "16 GB"; default: Psi4 setting'
    )
    parser.add_argument(
        "--scf-type", choices=("pk", "direct", "df"), default="pk"
    )
    parser.add_argument("--dft-radial-points", type=int, default=99)
    parser.add_argument("--dft-spherical-points", type=int, default=590)
    parser.add_argument("--e-convergence", type=float, default=1.0e-10)
    parser.add_argument("--d-convergence", type=float, default=1.0e-10)
    parser.add_argument("--g-convergence", default="gau_verytight")
    parser.add_argument(
        "--scf-initial-accelerator",
        choices=("ADIIS", "EDIIS", "NONE"),
        default="NONE",
        help=(
            "initial SCF accelerator (default: NONE; avoids an observed "
            "Psi4 1.11 ADIIS minimizer failure while retaining DIIS)"
        ),
    )
    parser.add_argument("--imaginary-threshold-cm1", type=float, default=10.0)
    parser.add_argument(
        "--reference-gradient-max-abs",
        type=float,
        default=1.0e-5,
        help=(
            "largest reference-gradient component allowed in a consumer "
            "bundle, in Eh/bohr (default: 1e-5)"
        ),
    )
    parser.add_argument(
        "--hessian-antisymmetry-relative-max",
        type=float,
        default=1.0e-3,
        help=(
            "largest max-antisymmetry/max-Hessian ratio allowed in a "
            "consumer bundle (default: 1e-3)"
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("geometric", "optking"),
        default="geometric",
        help="geometry optimizer (default: geometric for cluster robustness)",
    )
    parser.add_argument(
        "--coordsys",
        choices=("tric", "cart"),
        default="tric",
        help="geomeTRIC coordinate system (default: tric; ignored by optking)",
    )
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument(
        "--require-minimum",
        dest="require_minimum",
        action="store_true",
        default=True,
        help="require the complete consumer reference gate (default)",
    )
    validation.add_argument(
        "--allow-invalid-reference",
        dest="require_minimum",
        action="store_false",
        help=(
            "write explicitly failed bundles for diagnosis; the tutorial "
            "loader will reject them"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.xyz is not None:
        args.system = None
    if args.multiplicity <= 0:
        parser.error("--multiplicity must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.step_bohr <= 0.0:
        parser.error("--step-bohr must be positive")
    if args.imaginary_threshold_cm1 < 0.0:
        parser.error("--imaginary-threshold-cm1 must be non-negative")
    if args.reference_gradient_max_abs <= 0.0:
        parser.error("--reference-gradient-max-abs must be positive")
    if args.hessian_antisymmetry_relative_max <= 0.0:
        parser.error("--hessian-antisymmetry-relative-max must be positive")
    return run_reference(args)


if __name__ == "__main__":
    raise SystemExit(main())
