"""Zeolite framework builders for the AWH water-sorbent tutorial.

Loads siliceous CHA and MFI frameworks from the IZA-SC database CIFs
shipped under ``data/hosts/``. Provides:

* :func:`build_siliceous_cha` - pure-Si chabazite (CHA topology)
* :func:`build_siliceous_mfi` - pure-Si silicalite-1 (MFI topology)
* :func:`build_h_cha` - Al-substituted H-chabazite with one Bronsted
  acid site per unit cell (single Si->Al swap on a T-site, charge
  compensated by a proton on the adjacent bridging oxygen)
* :func:`build_h_sapo34` - H-SAPO-34 approximation: CHA topology with
  single Si->Al substitution and H on adjacent O. The full SAPO-34
  Al/P ordering is more complex; for the publication-grade
  substitution pattern see Fischer 2015 supplementary information.

All zeolite frameworks are *bulk* 3D-periodic structures. Unlike the
oxide builders in :mod:`helpers.oxide_slabs`, they are not cleaved into
slabs - H2O is physisorbed *inside* the pore network, not on an
external surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ase
    import pymatgen.core


_HOSTS_DIR = Path(__file__).resolve().parent.parent / "data" / "hosts"
CHA_CIF = _HOSTS_DIR / "CHA_siliceous.cif"
MFI_CIF = _HOSTS_DIR / "MFI_siliceous.cif"


# ---------------------------------------------------------------------------
# Pure-silica frameworks (direct CIF loads)
# ---------------------------------------------------------------------------


def _load_cif_as_ase(cif_path: Path) -> ase.Atoms:
    """Load a CIF via pymatgen and return an ASE Atoms object with PBC."""
    import ase
    from pymatgen.core import Structure

    if not cif_path.is_file():
        raise FileNotFoundError(
            f"Zeolite CIF missing: {cif_path}. "
            "Re-run the CIF download step (see data/hosts/README.md)."
        )
    structure = Structure.from_file(str(cif_path))
    return ase.Atoms(
        positions=structure.cart_coords,
        numbers=structure.atomic_numbers,
        cell=structure.lattice.matrix,
        pbc=True,
    )


def build_siliceous_cha() -> ase.Atoms:
    """Build siliceous chabazite (CHA topology, pure SiO2).

    Source: IZA-SC Database of Zeolite Structures (Baerlocher & McCusker);
    atomic coordinates optimised with DLS76 assuming pure SiO2 composition.
    """
    return _load_cif_as_ase(CHA_CIF)


def build_siliceous_mfi() -> ase.Atoms:
    """Build siliceous MFI (silicalite-1, pure SiO2).

    Source: IZA-SC Database of Zeolite Structures (Baerlocher & McCusker);
    atomic coordinates optimised with DLS76 assuming pure SiO2 composition.
    """
    return _load_cif_as_ase(MFI_CIF)


# ---------------------------------------------------------------------------
# Bronsted-acid (Al-substituted) frameworks
# ---------------------------------------------------------------------------


def _substitute_al_and_protonate(
    atoms: ase.Atoms,
    t_site_index: int | None = None,
    proton_oh_bond: float = 0.98,
) -> ase.Atoms:
    """Replace one Si T-site with Al and place a proton on an adjacent O.

    The resulting structure has composition (Si_{n-1}AlO_{2n})H with a
    Bronsted-acid site (Si-O(H)-Al bridge). Charge is balanced (Al
    contributes -1 vs Si, H contributes +1).

    Parameters
    ----------
    atoms : ase.Atoms
        Siliceous framework with PBC and a full Si/O lattice.
    t_site_index : int or None
        Index of the Si atom to substitute. If None, picks the Si closest
        to the unit-cell centre (makes the acid site reproducible across
        runs). Pass an explicit index for different T-site choices.
    proton_oh_bond : float
        Initial O-H bond length in Angstrom (default 0.98). The BGR
        relaxation will refine this.

    Returns
    -------
    ase.Atoms with one extra H atom.
    """
    import ase

    si_indices = [i for i, z in enumerate(atoms.numbers) if z == 14]
    o_indices = [i for i, z in enumerate(atoms.numbers) if z == 8]
    if not si_indices:
        raise ValueError("No Si atoms in framework - cannot substitute.")

    if t_site_index is None:
        # pick Si closest to cell centre
        center = atoms.cell.array.sum(axis=0) / 2.0
        dists = np.linalg.norm(atoms.positions[si_indices] - center, axis=1)
        t_site_index = si_indices[int(np.argmin(dists))]

    if atoms.numbers[t_site_index] != 14:
        raise ValueError(
            f"Atom {t_site_index} is {atoms.get_chemical_symbols()[t_site_index]}, not Si."
        )

    # Nearest O to the chosen T-site (minimum-image convention)
    cell = atoms.cell.array
    inv_cell = np.linalg.inv(cell)
    t_pos = atoms.positions[t_site_index]
    best_o, best_d = None, np.inf
    for oi in o_indices:
        d = atoms.positions[oi] - t_pos
        frac = d @ inv_cell
        frac -= np.round(frac)
        d_mic = frac @ cell
        r = float(np.linalg.norm(d_mic))
        if r < best_d:
            best_d = r
            best_o = oi
    if best_o is None:
        raise ValueError("No O atom found near the chosen T-site.")

    out = atoms.copy()
    out.numbers[t_site_index] = 13  # Si -> Al

    # Place H along the minimum-image Si-O vector, extended outward from O
    d_so = atoms.positions[best_o] - t_pos
    frac = d_so @ inv_cell
    frac -= np.round(frac)
    d_so_mic = frac @ cell
    o_pos_mic = t_pos + d_so_mic  # real-space O position relative to Al
    direction = d_so_mic / np.linalg.norm(d_so_mic)
    h_pos = o_pos_mic + proton_oh_bond * direction

    h_atom = ase.Atoms("H", positions=[h_pos], cell=cell, pbc=True)
    out = out + h_atom
    return out


def build_h_cha(t_site_index: int | None = None) -> ase.Atoms:
    """Build H-CHA (aluminosilicate chabazite with one Bronsted site).

    Starts from the siliceous CHA framework and substitutes one Si with Al,
    placing a compensating proton on the closest bridging oxygen (the
    standard Si-O(H)-Al motif). This is the S24 Zeolite-class Tier-1
    reference host.
    """
    return _substitute_al_and_protonate(build_siliceous_cha(), t_site_index)


def build_h_sapo34(t_site_index: int | None = None) -> ase.Atoms:
    """Build an H-SAPO-34 approximation (CHA topology, single Bronsted site).

    This returns the same structure as :func:`build_h_cha` - CHA topology
    with one Al substitution and one proton - and uses it as the Tier-3
    reference for Fischer 2015 CP2K PBE-D3 H2O binding on SAPO-34. True
    SAPO-34 has an alternating Al/P pattern on the T-sites with Si
    substitution producing the Bronsted sites, which matters for energetics
    beyond the per-site H2O adsorption level. Within the tutorial scope
    (isolated H2O physisorption on one Bronsted site) this approximation
    is documented in the notebook and defensible; for publication-grade
    Al/P ordering see Fischer 2015 supplementary information.
    """
    return _substitute_al_and_protonate(build_siliceous_cha(), t_site_index)
