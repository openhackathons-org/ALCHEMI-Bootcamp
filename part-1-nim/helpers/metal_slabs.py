"""fcc metal-slab builders for the AdsorbML configuration-search tutorial.

Cu(111) and Pd(111) are the two canonical fcc-metal surfaces in the
Open Catalyst 2020 dataset and in decades of surface-science literature
(CO/Cu(111) top-site, CO/Pd(111) hollow-site, methanol/Cu(111) O-down,
H2O bilayers on both). Both are non-magnetic at the closed-shell
level that MACE-MP-0 was trained on, and both sit inside MPtrj's
demonstrated validation envelope.

Slab convention (OC20-compatible):
- 4 layers total, bottom 2 frozen via active_mask
- (3, 3, 1) supercell in the surface plane (9 surface sites)
- 15 A vacuum along c (inside the 12 A MACE receptive-field cutoff
  plus slack for adsorbates)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .surfaces import build_slab

if TYPE_CHECKING:
    import ase
    import pymatgen.core


# ---------------------------------------------------------------------------
# Experimental lattice constants (300 K, room pressure)
# ---------------------------------------------------------------------------

LATTICE_A_CU = 3.615  # A — Straumanis & Yu 1969, Acta Cryst. A 25, 676
LATTICE_A_PD = 3.891  # A — Arblaster 2012, Platinum Metals Rev. 56, 181


# ---------------------------------------------------------------------------
# Bulk cells (fcc Fm-3m, #225)
# ---------------------------------------------------------------------------


def build_cu_bulk(a: float = LATTICE_A_CU) -> pymatgen.core.Structure:
    """Build fcc Cu primitive cell."""
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(a),
        ["Cu"],
        [[0.0, 0.0, 0.0]],
    )


def build_pd_bulk(a: float = LATTICE_A_PD) -> pymatgen.core.Structure:
    """Build fcc Pd primitive cell."""
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(a),
        ["Pd"],
        [[0.0, 0.0, 0.0]],
    )


# ---------------------------------------------------------------------------
# (111) slabs
# ---------------------------------------------------------------------------


def build_cu111_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> ase.Atoms:
    """Build a Cu(111) slab, OC20 convention (4-layer, 3x3 in-plane).

    The (111) termination is the thermodynamically stable fcc closest-packed
    surface and the one referenced in OC20 and in classical CO/Cu(111)
    surface-science literature (Bagus & Pacchioni 1989).
    """
    return build_slab(
        build_cu_bulk(),
        miller_index=(1, 1, 1),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        supercell=supercell,
    )


def build_pd111_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> ase.Atoms:
    """Build a Pd(111) slab, OC20 convention (4-layer, 3x3 in-plane).

    Canonical hydrogenation catalyst; CO binds at fcc-hollow at low coverage
    per Hammer-Morikawa-Norskov 1996 (PRL 76, 2141).
    """
    return build_slab(
        build_pd_bulk(),
        miller_index=(1, 1, 1),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        supercell=supercell,
    )
