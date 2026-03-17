"""Conformer generation, energy filtering, and CREST-inspired deduplication."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .analysis import kabsch_rmsd
from .constants import KCAL_MOL_TO_EV


def compute_n_conformers(mol: Chem.Mol) -> int:
    """Heuristic conformer count based on rotatable bonds: min(1000, max(200, 3^n_rot))."""
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return min(1000, max(200, 3**n_rot))


def generate_conformers(
    mol: Chem.Mol, n_confs: int, seed: int = 42, rmsd_threshold=0.5
) -> Chem.Mol:
    """Generate 3-D conformers using ETKDGv3 + MMFF optimisation.

    Returns a copy of *mol* with embedded conformers.  Near-identical initial
    geometries are pruned with ``pruneRmsThresh=0.5``.
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = rmsd_threshold
    params.numThreads = 0  # use all available cores
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    return mol


def filter_by_energy(
    energies: np.ndarray,
    threshold_kcal: float = 3.0,
) -> np.ndarray:
    """Return boolean mask of conformers within *threshold_kcal* of the minimum.

    *energies* should be in eV.  The threshold is converted internally
    (1 kcal/mol = 0.0434 eV).
    """
    threshold_ev = threshold_kcal * KCAL_MOL_TO_EV
    e_min = np.min(energies)
    return (energies - e_min) <= threshold_ev


def deduplicate_conformers(
    coords_list: list[np.ndarray],
    energies: np.ndarray,
    rmsd_threshold: float = 0.125,
) -> list[int]:
    """CREST-inspired greedy deduplication by Kabsch-aligned RMSD.

    1. Sort conformers by energy (ascending).
    2. For each conformer, compute RMSD to all previously kept conformers.
    3. Discard if RMSD < *rmsd_threshold* to any kept conformer.

    Returns indices (into the original lists) of unique conformers.
    """
    order = np.argsort(energies)
    kept_indices: list[int] = []
    kept_coords: list[np.ndarray] = []

    for idx in order:
        coords = coords_list[idx]
        is_dup = False
        for kc in kept_coords:
            if kabsch_rmsd(coords, kc) < rmsd_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(int(idx))
            kept_coords.append(coords)

    return kept_indices
