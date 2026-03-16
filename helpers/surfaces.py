"""Slab construction, adsorbate placement, and OER analysis utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import ase
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

if TYPE_CHECKING:
    import pymatgen.core

AdsorbateSpecies = Literal["O", "OH", "H2O", "OOH"]


# ---------------------------------------------------------------------------
# Rutile bulk structure
# ---------------------------------------------------------------------------


def build_rutile_bulk(
    metal: str,
    a: float,
    c: float,
    u: float = 0.305,
) -> pymatgen.core.Structure:
    """Build a rutile bulk structure (P4_2/mnm, #136).

    Parameters
    ----------
    metal : str
        Metal element symbol (e.g. "Ir", "Ru", "Ti").
    a, c : float
        Tetragonal lattice parameters in Angstrom.
    u : float
        Oxygen internal coordinate (Wyckoff 4f position (u, u, 0)).
        Typical values: TiO2 ~0.305, RuO2 ~0.306, IrO2 ~0.306.

    References
    ----------
    Lattice parameters and internal coordinates from:
    Bolzan et al., "Powder neutron diffraction study of pyrolusite,
    beta-MnO2", Acta Cryst. B 53, 373-380 (1997).
    Shannon, "Revised effective ionic radii and systematic studies of
    interatomic distances in halides and chalcogenides", Acta Cryst. A
    32, 751-767 (1976).

    Returns
    -------
    pymatgen.core.Structure
    """
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        136,
        Lattice.tetragonal(a, c),
        [metal, "O"],
        [[0.0, 0.0, 0.0], [u, u, 0.0]],
    )


# ---------------------------------------------------------------------------
# Slab construction
# ---------------------------------------------------------------------------


def build_slab(
    bulk: pymatgen.core.Structure,
    miller_index: tuple[int, int, int] = (1, 1, 0),
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (2, 2, 1),
) -> ase.Atoms:
    """Build a surface slab using pymatgen SlabGenerator.

    Selects the most stoichiometric, symmetric termination when available.
    Converts to ASE Atoms with cell and PBC, then applies the supercell.

    Parameters
    ----------
    bulk : pymatgen Structure
        Bulk crystal (e.g. from build_rutile_bulk).
    miller_index : tuple
        Miller index, e.g. (1, 1, 0).
    min_slab_size : float
        Minimum slab thickness in Angstrom.
    min_vacuum_size : float
        Vacuum gap in Angstrom.
    supercell : tuple
        In-plane supercell dimensions, e.g. (2, 2, 1).

    Returns
    -------
    ase.Atoms with cell and PBC=[True, True, True].
    """
    from pymatgen.core.surface import SlabGenerator

    sg = SlabGenerator(
        bulk,
        miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=True,
        in_unit_planes=False,
    )
    slabs = sg.get_slabs()
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slabs for {miller_index}.")

    # Prefer symmetric slabs, then pick the first (usually most stoichiometric)
    symmetric = [s for s in slabs if s.is_symmetric()]
    pmg_slab = symmetric[0] if symmetric else slabs[0]

    atoms = ase.Atoms(
        positions=pmg_slab.cart_coords,
        numbers=pmg_slab.atomic_numbers,
        cell=pmg_slab.lattice.matrix,
        pbc=True,
    )

    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)

    return atoms


# ---------------------------------------------------------------------------
# Active mask (frozen / relaxable atoms)
# ---------------------------------------------------------------------------


def make_active_mask(
    atoms: ase.Atoms,
    bottom_fraction: float = 0.5,
) -> list[bool]:
    """Generate active_mask: True for atoms to relax, False for frozen.

    Atoms in the bottom *bottom_fraction* of the slab height are frozen.

    Parameters
    ----------
    atoms : ase.Atoms
        Slab structure.
    bottom_fraction : float
        Fraction of slab height to freeze (default 0.5 = bottom half).

    Returns
    -------
    list[bool] with len == len(atoms).
    """
    z = atoms.positions[:, 2]
    z_min, z_max = z.min(), z.max()
    threshold = z_min + bottom_fraction * (z_max - z_min)
    return [bool(zi > threshold) for zi in z]


# ---------------------------------------------------------------------------
# Site identification on rutile (110)
# ---------------------------------------------------------------------------


def find_cus_sites(
    atoms: ase.Atoms,
    metal_z: int,
    cutoff_mult: float = 1.2,
) -> np.ndarray:
    """Locate coordinatively unsaturated (cus) metal sites on the surface.

    On rutile (110), cus metals are 5-fold coordinated (one missing oxygen
    neighbour compared to bulk 6-fold coordination).  This function finds
    metal atoms in the top half of the slab with fewer than 6 oxygen
    neighbours.

    Parameters
    ----------
    atoms : ase.Atoms
        Slab structure.
    metal_z : int
        Atomic number of the metal (e.g. 77 for Ir).
    cutoff_mult : float
        Multiplier for natural_cutoffs neighbour search.

    Returns
    -------
    np.ndarray shape (N, 3) — positions of all cus metal sites.
    """
    cutoffs = natural_cutoffs(atoms, mult=cutoff_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    metal_idx = [i for i, z in enumerate(atoms.numbers) if z == metal_z]
    if not metal_idx:
        raise ValueError(f"No atoms with Z={metal_z} found in structure.")

    z_coords = atoms.positions[metal_idx, 2]
    z_mid = (z_coords.min() + z_coords.max()) / 2.0

    cus_positions = []
    for i in metal_idx:
        if atoms.positions[i, 2] < z_mid:
            continue  # skip bottom side of catalyst slab
        indices, _ = nl.get_neighbors(i)
        n_oxygen = sum(1 for j in indices if atoms.numbers[j] == 8)
        if n_oxygen < 6:
            cus_positions.append(atoms.positions[i].copy())

    if not cus_positions:
        # Fallback: return the topmost metal atom
        top_idx = metal_idx[np.argmax(z_coords)]
        cus_positions.append(atoms.positions[top_idx].copy())

    return np.array(cus_positions)


def find_bridge_site(
    atoms: ase.Atoms,
    metal_z: int,
    cutoff_mult: float = 1.2,
) -> np.ndarray:
    """Find the bridge site between two adjacent cus metal atoms.

    Returns the midpoint of the two closest cus sites on the surface.

    Parameters
    ----------
    atoms : ase.Atoms
        Slab structure.
    metal_z : int
        Atomic number of the metal.
    cutoff_mult : float
        Multiplier for natural_cutoffs.

    Returns
    -------
    np.ndarray shape (3,) — Cartesian position of the bridge site.
    """
    cus = find_cus_sites(atoms, metal_z, cutoff_mult)
    if len(cus) < 2:
        raise ValueError("Need at least 2 cus sites for a bridge position.")

    from itertools import combinations

    best_pair, best_dist = None, np.inf
    for i, j in combinations(range(len(cus)), 2):
        d = np.linalg.norm(cus[i] - cus[j])
        if d < best_dist:
            best_dist = d
            best_pair = (i, j)

    return (cus[best_pair[0]] + cus[best_pair[1]]) / 2.0


def find_central_site(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Select the site closest to the centre of the slab's xy-plane.

    Parameters
    ----------
    positions : np.ndarray shape (N, 3)
        Candidate site positions (e.g. from ``find_cus_sites``).
    cell : np.ndarray shape (3, 3)
        Unit cell matrix.

    Returns
    -------
    np.ndarray shape (3,) — position of the most central site.
    """
    center_xy = (cell[0, :2] + cell[1, :2]) / 2.0
    dists = np.linalg.norm(positions[:, :2] - center_xy, axis=1)
    return positions[np.argmin(dists)].copy()


# ---------------------------------------------------------------------------
# Adsorbate construction
# ---------------------------------------------------------------------------


def build_adsorbate(species: AdsorbateSpecies) -> ase.Atoms:
    """Build a small adsorbate molecule.

    All adsorbates have the bonding atom (O) at the origin so that
    ``place_adsorbate`` can position them directly above a surface site.

    Geometries use experimental bond lengths:
      * O-H = 0.957 A (H2O), 0.970 A (OH, OOH)
      * H-O-H = 104.5 deg
      * O-O  = 1.33 A  (OOH peroxo intermediate)
      * O-O-H = 110 deg

    References
    ----------
    H2O geometry: Benedict, Gailer & Plyler, J. Chem. Phys. 24, 1139 (1956).
    OH bond length: Huber & Herzberg, "Molecular Spectra and Molecular
    Structure IV: Constants of Diatomic Molecules" (1979).
    OOH geometry: Binkley & Melius, J. Chem. Phys. 84, 2064 (1986).

    Parameters
    ----------
    species : {"O", "OH", "H2O", "OOH"}

    Returns
    -------
    ase.Atoms
    """
    match species:
        case "O":
            return ase.Atoms("O", positions=[[0.0, 0.0, 0.0]])

        case "OH":
            d_oh = 0.970
            return ase.Atoms(
                "OH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, d_oh],
                ],
            )

        case "H2O":
            d_oh = 0.957
            angle = np.radians(104.5)
            return ase.Atoms(
                "OH2",
                positions=[
                    [0.0, 0.0, 0.0],
                    [d_oh * np.sin(angle / 2), 0.0, d_oh * np.cos(angle / 2)],
                    [-d_oh * np.sin(angle / 2), 0.0, d_oh * np.cos(angle / 2)],
                ],
            )

        case "OOH":
            d_oo = 1.33
            d_oh = 0.970
            ooh_angle = np.radians(110.0)
            return ase.Atoms(
                "O2H",
                positions=[
                    [0.0, 0.0, 0.0],  # O bonding to metal
                    [0.0, 0.0, d_oo],  # O in peroxo bond
                    [
                        0.0,
                        d_oh * np.sin(ooh_angle),
                        d_oo + d_oh * np.cos(ooh_angle),
                    ],
                ],
            )

        case _:
            raise ValueError(f"Unknown adsorbate species: {species!r}")


# ---------------------------------------------------------------------------
# Adsorbate placement
# ---------------------------------------------------------------------------


def _surface_normal(slab: ase.Atoms) -> np.ndarray:
    """Compute the outward surface normal from the slab cell vectors.

    The normal is defined as the cross product of the first two cell
    vectors (a x b), normalised to unit length.  For standard slab cells
    this points along the vacuum direction.
    """
    a_vec = slab.cell[0]
    b_vec = slab.cell[1]
    normal = np.cross(a_vec, b_vec)
    return normal / np.linalg.norm(normal)


def place_adsorbate(
    slab: ase.Atoms,
    adsorbate: ase.Atoms,
    site: np.ndarray,
    height: float = 3.5,
    tilt_angle: float = 0.0,
    frozen_fraction: float = 0.5,
) -> tuple[ase.Atoms, list[bool]]:
    """Place an adsorbate above a surface site.

    The translation direction is computed from the slab cell vectors as
    the surface normal (a x b), so it works for both orthogonal and
    non-orthogonal cells.

    The *height* parameter (default 3.5 A) is above the equilibrium
    metal-oxygen bond distance for transition-metal oxides (Ir-O ~2.0 A,
    Ru-O ~1.9-2.0 A, Ti-O ~1.95 A), giving the optimiser room to find
    the minimum without starting inside the surface.

    References
    ----------
    M-O bond distances: Bolzan et al., Acta Cryst. B 53, 373 (1997);
    Shannon, Acta Cryst. A 32, 751 (1976).

    The *tilt_angle* (default 0 deg = upright) rotates the adsorbate
    about the in-plane a-vector direction before placement.  A 30 deg
    tilt brings one hydrogen of H2O or OH toward a neighbouring bridging
    oxygen, probing whether hydrogen bonding with the surface stabilises
    the adsorption.

    Parameters
    ----------
    slab : ase.Atoms
        Clean slab (no adsorbate yet).
    adsorbate : ase.Atoms
        Molecule from ``build_adsorbate``.
    site : np.ndarray shape (3,)
        Adsorption-site position on the surface.
    height : float
        Distance above the site along the surface normal (Angstrom).
    tilt_angle : float
        Rotation angle in degrees about the in-plane a-vector.
    frozen_fraction : float
        Fraction of slab height to freeze (passed to ``make_active_mask``).

    Returns
    -------
    (combined, active_mask) where *combined* is an ase.Atoms and
    *active_mask* is a list[bool] suitable for ``AtomicData.active_mask``.
    """
    normal_hat = _surface_normal(slab)
    a_hat = slab.cell[0] / np.linalg.norm(slab.cell[0])

    ads = adsorbate.copy()

    # Tilt adsorbate about the in-plane a-vector
    if abs(tilt_angle) > 1e-6:
        ads.rotate(tilt_angle, a_hat, center=(0.0, 0.0, 0.0))

    # Translate along the surface normal to the site + height
    ads.translate(site + height * normal_hat)

    # Combine slab + adsorbate
    combined = slab.copy()
    combined += ads

    # Build active mask: slab frozen/active + adsorbate always active
    slab_mask = make_active_mask(slab, bottom_fraction=frozen_fraction)
    combined_mask = slab_mask + [True] * len(ads)

    return combined, combined_mask


# ---------------------------------------------------------------------------
# Post-relaxation analysis
# ---------------------------------------------------------------------------


def classify_relaxation(
    initial: ase.Atoms,
    relaxed_result,
    slab_n_atoms: int,
    site_initial: np.ndarray,
    metal_z: int,
    height_threshold: float = 5.0,
    site_drift_threshold: float = 1.5,
) -> dict:
    """Classify a relaxed slab+adsorbate structure.

    Parameters
    ----------
    initial : ase.Atoms
        Initial (pre-relaxation) slab+adsorbate structure.
    relaxed_result : OptimizationResult
        BGR result with coord, energy, forces, converged.
    slab_n_atoms : int
        Number of atoms in the bare slab (adsorbate atoms are at the end).
    site_initial : np.ndarray shape (3,)
        Original adsorption-site position on the surface.
    metal_z : int
        Atomic number of the metal.
    height_threshold : float
        If adsorbate is this far above the topmost slab atom, flag desorbed.
    site_drift_threshold : float
        If the adsorbate drifted more than this laterally, flag moved.

    Returns
    -------
    dict with the following keys:

    converged : bool
        Whether the BGR optimiser reported convergence.
    max_force : float
        Maximum force magnitude across all atoms (eV/A).
    energy : float
        Total potential energy of the relaxed structure (eV).
    adsorbate_height : float
        Height of the adsorbate centre-of-mass above the topmost
        slab atom (A).
    nearest_metal_dist : float
        Distance from the bonding atom to the nearest metal (A).
    site_drift : float
        Lateral (xy) displacement of the adsorbate from its initial
        adsorption site (A).
    status : str
        One of ``"converged"``, ``"moved_to_new_site"``,
        ``"dissociated"``, ``"desorbed"``, or ``"needs_review"``.
    """
    from .models import atomic_data_to_ase

    relaxed = atomic_data_to_ase(relaxed_result)

    # Forces
    forces = np.array(relaxed_result.forces).reshape(-1, 3)
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))

    # Adsorbate atoms = those after slab_n_atoms
    ads_pos = relaxed.positions[slab_n_atoms:]
    ads_centre = ads_pos.mean(axis=0)

    # Surface reference: topmost slab atom z
    slab_z_max = relaxed.positions[:slab_n_atoms, 2].max()
    adsorbate_height = float(ads_centre[2] - slab_z_max)

    # Nearest metal distance
    metal_idx = [i for i in range(slab_n_atoms) if relaxed.numbers[i] == metal_z]
    if metal_idx:
        metal_pos = relaxed.positions[metal_idx]
        dists = np.linalg.norm(metal_pos - ads_pos[0], axis=1)
        nearest_metal_dist = float(dists.min())
    else:
        nearest_metal_dist = np.nan

    # Lateral drift from initial site
    site_drift = float(np.linalg.norm(ads_centre[:2] - site_initial[:2]))

    # Initial adsorbate inter-atom distances
    init_ads_pos = initial.positions[slab_n_atoms:]
    n_ads_atoms = len(init_ads_pos)

    # Check for dissociation: adsorbate atoms far apart
    dissociated = False
    if n_ads_atoms > 1:
        relax_ads_dists = []
        for i in range(n_ads_atoms):
            for j in range(i + 1, n_ads_atoms):
                relax_ads_dists.append(np.linalg.norm(ads_pos[i] - ads_pos[j]))
        init_ads_dists = []
        for i in range(n_ads_atoms):
            for j in range(i + 1, n_ads_atoms):
                init_ads_dists.append(np.linalg.norm(init_ads_pos[i] - init_ads_pos[j]))
        # If any bond stretched > 50%, consider dissociated
        for d_init, d_relax in zip(init_ads_dists, relax_ads_dists):
            if d_init > 0.1 and d_relax > 1.5 * d_init:
                dissociated = True
                break

    # Classification
    converged = bool(relaxed_result.converged)
    if adsorbate_height > height_threshold:
        status = "desorbed"
    elif dissociated:
        status = "dissociated"
    elif site_drift > site_drift_threshold:
        status = "moved_to_new_site"
    elif converged:
        status = "converged"
    else:
        status = "needs_review"

    return {
        "converged": converged,
        "max_force": max_force,
        "energy": float(relaxed_result.energy),
        "adsorbate_height": adsorbate_height,
        "nearest_metal_dist": nearest_metal_dist,
        "site_drift": site_drift,
        "status": status,
    }


def compute_adsorption_energy(
    e_slab_ads: float,
    e_slab: float,
    e_gas: float,
) -> float:
    """Compute adsorption energy.

    E_ads = E(slab + adsorbate) - E(clean slab) - E(gas-phase adsorbate)

    A negative value indicates exothermic (favourable) binding.
    """
    return e_slab_ads - e_slab - e_gas


def compute_surface_displacement(
    initial: ase.Atoms,
    relaxed: ase.Atoms,
    slab_n_atoms: int,
) -> np.ndarray:
    """Per-atom displacement magnitude for slab atoms after relaxation.

    Useful for OVITO colour-mapping: blue (minimal) to red (large).
    A good catalyst ideally shows minimal rearrangement of its surface
    atoms during adsorbate binding.

    Parameters
    ----------
    initial : ase.Atoms
        Pre-relaxation structure.
    relaxed : ase.Atoms
        Post-relaxation structure.
    slab_n_atoms : int
        Number of slab atoms (adsorbate atoms are excluded).

    Returns
    -------
    np.ndarray shape (slab_n_atoms,) — displacement magnitudes in Angstrom.
    """
    d = relaxed.positions[:slab_n_atoms] - initial.positions[:slab_n_atoms]
    return np.linalg.norm(d, axis=1)
