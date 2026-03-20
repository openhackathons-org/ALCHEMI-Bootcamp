"""Thermodynamic property extraction and structural analysis from MD trajectories."""

from typing import Optional

import ase
import ase.data
import ase.neighborlist
import numpy as np

from .constants import KE_CONV, BOLTZ_EV_K, P_CONV, AMU_TO_G, ANGSTROM3_TO_CM3
from .models import BMDAtomicData, BMDSnapshot


def trajectory_to_ase_list(
    mdatoms: BMDAtomicData,
    trajectory: list[BMDSnapshot],
) -> list[ase.Atoms]:
    """Convert a list of BMDSnapshots into ASE Atoms frames."""
    base = ase.Atoms(
        positions=np.array(mdatoms.coord).reshape(-1, 3),
        numbers=mdatoms.numbers,
    )
    if mdatoms.cell is not None:
        base.set_cell(np.array(mdatoms.cell).reshape(3, 3))
    if mdatoms.pbc is not None:
        base.set_pbc(mdatoms.pbc)

    frames: list[ase.Atoms] = []
    for snap in trajectory:
        atoms = base.copy()
        atoms.set_positions(np.array(snap.coord).reshape(-1, 3))
        if snap.cell is not None:
            atoms.set_cell(np.array(snap.cell).reshape(3, 3))
        atoms.info["energy"] = snap.energy
        atoms.info["istep"] = snap.istep
        atoms.info["md_time"] = snap.md_time
        if snap.velocity:
            atoms.set_velocities(np.array(snap.velocity).reshape(-1, 3))
        if snap.stress is not None:
            atoms.info["stress"] = np.array(snap.stress).reshape(3, 3)
        frames.append(atoms)
    return frames


def extract_thermo_timeseries(
    mdatoms: BMDAtomicData,
    trajectory: list[BMDSnapshot],
) -> dict[str, np.ndarray]:
    """Extract time-resolved thermodynamic quantities from an MD trajectory.

    Returns a dict with keys: time_ps, temperature_K, e_pot_eV, e_kin_eV,
    e_tot_eV, pressure_kbar, volume_A3, p_xx, p_yy, p_zz.
    """
    if mdatoms.mass is not None:
        mass = np.array(mdatoms.mass)[:, None]
    else:
        mass = ase.data.atomic_masses[mdatoms.numbers][:, None]

    records: dict[str, list[float]] = {
        "time_ps": [],
        "temperature_K": [],
        "e_pot_eV": [],
        "e_kin_eV": [],
        "e_tot_eV": [],
        "pressure_kbar": [],
        "volume_A3": [],
        "p_xx": [],
        "p_yy": [],
        "p_zz": [],
    }

    for snap in trajectory:
        d = snap.model_dump()
        velocity = np.array(d["velocity"]).reshape(-1, 3)

        # Cell
        cell = None
        if d.get("cell") is not None:
            cell = np.array(d["cell"]).reshape(3, 3)
        elif mdatoms.cell is not None:
            cell = np.array(mdatoms.cell).reshape(3, 3)
        remove_rot = cell is None

        # Static stress
        static_stress = (
            np.array(d["stress"]).reshape(3, 3)
            if d.get("stress") is not None
            else np.zeros((3, 3))
        )

        # Kinetic energy
        e_kin = 0.5 * (mass * velocity**2).sum() * KE_CONV
        e_pot = d["energy"]
        e_tot = e_pot + e_kin

        # Temperature
        dof = 3 * len(mass) - 3 - (3 if remove_rot else 0)
        temperature = 2.0 * e_kin / (dof * BOLTZ_EV_K) if dof > 0 else 0.0

        # Volume and pressure
        if cell is not None:
            volume = float(np.linalg.det(cell))
            kinetic_stress = (mass * velocity).T @ velocity / volume * KE_CONV
        else:
            volume = 0.0
            kinetic_stress = np.zeros((3, 3))
        total_stress = static_stress + kinetic_stress
        p = total_stress * P_CONV / 1000.0  # kBar
        p_iso = (p[0, 0] + p[1, 1] + p[2, 2]) / 3.0

        records["time_ps"].append(d["md_time"])
        records["temperature_K"].append(temperature)
        records["e_pot_eV"].append(e_pot)
        records["e_kin_eV"].append(e_kin)
        records["e_tot_eV"].append(e_tot)
        records["pressure_kbar"].append(p_iso)
        records["volume_A3"].append(volume)
        records["p_xx"].append(p[0, 0])
        records["p_yy"].append(p[1, 1])
        records["p_zz"].append(p[2, 2])

    return {k: np.array(v) for k, v in records.items()}


def pick_production_window(
    thermo: dict[str, np.ndarray],
    discard_fraction: float = 0.3,
) -> tuple[int, int]:
    """Return (start_idx, end_idx) that discards the first *discard_fraction* of frames."""
    n = len(thermo["time_ps"])
    start = int(n * discard_fraction)
    return start, n


def compute_density(
    mdatoms: BMDAtomicData,
    volumes: np.ndarray,
) -> float:
    """Compute average density in g/cm^3 from mean volume (A^3) and atomic masses."""
    if mdatoms.mass is not None:
        total_mass_amu = sum(mdatoms.mass)
    else:
        total_mass_amu = float(ase.data.atomic_masses[mdatoms.numbers].sum())
    mean_vol = float(np.mean(volumes))
    return total_mass_amu * AMU_TO_G / (mean_vol * ANGSTROM3_TO_CM3)


def compute_rdf(
    frames: list[ase.Atoms],
    species_pair: tuple[int, int],
    r_max: float = 8.0,
    n_bins: int = 200,
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function g(r) for a pair of species.

    Parameters
    ----------
    frames : list of ASE Atoms (should have PBC + cell)
    species_pair : tuple of two atomic numbers (e.g. (11, 17) for Na-Cl)
    r_max : cutoff distance in Angstrom
    n_bins : number of histogram bins

    Returns
    -------
    r_centres : array of bin centres
    g_r : array of g(r) values
    """
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins, dtype=float)

    z_a, z_b = species_pair
    n_frames_used = 0

    for atoms in frames[start_frame:]:
        idx_a = np.where(atoms.numbers == z_a)[0]
        idx_b = np.where(atoms.numbers == z_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue

        cell = atoms.get_cell()
        n_frames_used += 1

        for i in idx_a:
            diffs = atoms.positions[idx_b] - atoms.positions[i]
            # Minimum image convention
            scaled = diffs @ np.linalg.inv(cell.array)
            scaled -= np.round(scaled)
            diffs = scaled @ cell.array
            dists = np.linalg.norm(diffs, axis=1)

            if z_a == z_b:
                dists = dists[dists > 1e-10]

            counts, _ = np.histogram(dists, bins=r_edges)
            hist += counts

        # Normalise for this frame
        n_a = len(idx_a)
        n_b = len(idx_b)
        if z_a == z_b:
            n_pairs = n_a * (n_b - 1)
        else:
            n_pairs = n_a * n_b
        # Accumulate the normalised histogram
        # (We'll divide by n_frames_used at the end)

    if n_frames_used > 0:
        # Total pairs across all frames
        # Re-compute the average normalisation
        atoms_ref = frames[start_frame]
        idx_a = np.where(atoms_ref.numbers == z_a)[0]
        idx_b = np.where(atoms_ref.numbers == z_b)[0]
        n_a = len(idx_a)
        n_b = len(idx_b)
        if z_a == z_b:
            n_pairs = n_a * (n_b - 1)
        else:
            n_pairs = n_a * n_b

        mean_vol = np.mean([f.get_volume() for f in frames[start_frame:]])
        shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
        ideal_count = (n_pairs / mean_vol) * shell_volumes * n_frames_used
        g_r = np.divide(
            hist, ideal_count, out=np.zeros_like(hist), where=ideal_count > 0
        )
    else:
        g_r = np.zeros(n_bins)

    return r_centres, g_r


def compute_msd(
    frames: list[ase.Atoms],
    species: Optional[int] = None,
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean-squared displacement relative to the first production frame.

    Returns (time_ps, msd_A2). If *species* is given, only that atomic number is used.
    """
    ref = frames[start_frame]
    ref_pos = ref.positions.copy()
    numbers = ref.numbers

    if species is not None:
        mask = numbers == species
    else:
        mask = np.ones(len(numbers), dtype=bool)

    ref_pos_sel = ref_pos[mask]
    times = []
    msds = []

    for frame in frames[start_frame:]:
        delta = frame.positions[mask] - ref_pos_sel
        msd_val = np.mean(np.sum(delta**2, axis=1))
        times.append(frame.info.get("md_time", 0.0))
        msds.append(msd_val)

    return np.array(times), np.array(msds)


def estimate_diffusion_coefficient(
    time_ps: np.ndarray,
    msd: np.ndarray,
    fit_fraction: tuple[float, float] = (0.3, 0.9),
) -> float:
    """Estimate diffusion coefficient D from the linear regime of MSD(t).

    D = MSD / (6 * t) in 3-D.  Returns D in A^2/ps.
    """
    n = len(time_ps)
    i0 = int(n * fit_fraction[0])
    i1 = int(n * fit_fraction[1])
    if i1 - i0 < 2:
        return 0.0
    coeffs = np.polyfit(time_ps[i0:i1], msd[i0:i1], 1)
    slope = coeffs[0]  # A^2/ps
    return slope / 6.0


def thermal_expansion_proxy(
    temperatures: np.ndarray,
    densities: np.ndarray,
) -> float:
    """Estimate volumetric thermal expansion coefficient alpha_V from rho(T).

    alpha_V = -(1/rho) * d(rho)/dT  (units: K^-1).
    Uses a linear fit across the supplied T-rho pairs.
    """
    coeffs = np.polyfit(temperatures, densities, 1)
    drho_dT = coeffs[0]
    rho_mean = np.mean(densities)
    return -drho_dT / rho_mean


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two (N, 3) coordinate arrays.

    Uses SVD-based optimal rotation to minimise RMSD before computing it.
    Both arrays are centred before alignment.
    """
    a = coords_a - coords_a.mean(axis=0)
    b = coords_b - coords_b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T
    a_rot = a @ R.T
    return float(np.sqrt(np.mean(np.sum((a_rot - b) ** 2, axis=1))))


def compute_rmsd(
    frames: list[ase.Atoms],
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMSD relative to the first production frame.

    Returns ``(time_ps, rmsd_A)``.  Follows the same signature pattern as
    :func:`compute_msd`.
    """
    ref = frames[start_frame]
    ref_pos = ref.positions.copy()
    times: list[float] = []
    rmsds: list[float] = []
    for frame in frames[start_frame:]:
        diff = frame.positions - ref_pos
        rmsd_val = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        times.append(frame.info.get("md_time", 0.0))
        rmsds.append(rmsd_val)
    return np.array(times), np.array(rmsds)


def check_bond_integrity(
    frames: list[ase.Atoms],
    ref_frame_idx: int = 0,
    tolerance: float = 0.3,
) -> dict:
    """Check whether covalent bonds are preserved throughout a trajectory.

    Uses ASE ``natural_cutoffs`` (+ *tolerance* Angstrom) to build a bond
    adjacency set for the reference frame and each subsequent frame.

    Returns
    -------
    dict with keys:
        n_broken : int — total bonds lost relative to reference (last frame)
        n_formed : int — total bonds gained relative to reference (last frame)
        integrity_score : float — fraction of frames with identical bond set (1.0 = perfect)
        frame_scores : list[float] — per-frame score (1.0 if bond set matches reference)
    """
    ref = frames[ref_frame_idx]
    cutoffs = ase.neighborlist.natural_cutoffs(ref, mult=1.0)
    cutoffs = [c + tolerance for c in cutoffs]

    def _bond_set(atoms: ase.Atoms) -> set[tuple[int, int]]:
        nl = ase.neighborlist.NeighborList(
            cutoffs, self_interaction=False, bothways=False, skin=0.0
        )
        nl.update(atoms)
        bonds = set()
        for i in range(len(atoms)):
            indices = nl.get_neighbors(i)[0]
            for j in indices:
                bonds.add((min(i, j), max(i, j)))
        return bonds

    ref_bonds = _bond_set(ref)
    frame_scores: list[float] = []
    last_bonds = ref_bonds

    for frame in frames:
        cur_bonds = _bond_set(frame)
        frame_scores.append(1.0 if cur_bonds == ref_bonds else 0.0)
        last_bonds = cur_bonds

    n_broken = len(ref_bonds - last_bonds)
    n_formed = len(last_bonds - ref_bonds)
    integrity_score = sum(frame_scores) / len(frame_scores) if frame_scores else 1.0

    return {
        "n_broken": n_broken,
        "n_formed": n_formed,
        "integrity_score": integrity_score,
        "frame_scores": frame_scores,
    }
