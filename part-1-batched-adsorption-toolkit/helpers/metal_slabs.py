"""fcc metal-slab builders for AdsorbML configuration-search examples.

The Cu/TiO2/TiN nine-surface adsorption screen uses Cu facets. Cu(111) remains
the metal reference builder used by that screen; Pd(111) is retained for
OC20/context checks and Cu/Pd geometry-audit scripts. Both are non-magnetic at
the closed-shell level used by the MACE-family model in this tutorial, and
both sit inside the broad materials benchmark coverage used to evaluate that
model family.

Slab convention (OC20-compatible):
- 4 layers total, bottom 2 frozen via active_mask
- (3, 3, 1) supercell in the surface plane (9 surface sites)
- 15 A vacuum along c (inside the 12 A MACE receptive-field cutoff
  plus slack for adsorbates)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
# Cu slabs
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
    del min_slab_size  # API compatibility; this builder intentionally uses 4 layers.
    from ase.build import fcc111

    nx, ny, nz = supercell
    atoms = fcc111(
        "Cu",
        size=(nx, ny, 4 * nz),
        a=LATTICE_A_CU,
        vacuum=min_vacuum_size / 2.0,
        periodic=True,
    )
    atoms.pbc = [True, True, True]
    return atoms


def build_cu100_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> ase.Atoms:
    """Build a Cu(100) slab, a square low-index fcc terrace."""
    del min_slab_size
    from ase.build import fcc100

    nx, ny, nz = supercell
    atoms = fcc100(
        "Cu",
        size=(nx, ny, 4 * nz),
        a=LATTICE_A_CU,
        vacuum=min_vacuum_size / 2.0,
        periodic=True,
    )
    atoms.pbc = [True, True, True]
    return atoms


def build_cu110_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> ase.Atoms:
    """Build a Cu(110) slab, an open row-like low-index fcc surface."""
    del min_slab_size
    from ase.build import fcc110

    nx, ny, nz = supercell
    atoms = fcc110(
        "Cu",
        size=(nx, ny, 4 * nz),
        a=LATTICE_A_CU,
        vacuum=min_vacuum_size / 2.0,
        periodic=True,
    )
    atoms.pbc = [True, True, True]
    return atoms


def build_pd111_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> ase.Atoms:
    """Build a Pd(111) slab, OC20 convention (4-layer, 3x3 in-plane).

    Canonical hydrogenation catalyst; CO binds at fcc-hollow at low coverage
    per Hammer-Morikawa-Norskov 1996 (PRL 76, 2141).
    """
    del min_slab_size  # API compatibility; this builder intentionally uses 4 layers.
    from ase.build import fcc111

    nx, ny, nz = supercell
    atoms = fcc111(
        "Pd",
        size=(nx, ny, 4 * nz),
        a=LATTICE_A_PD,
        vacuum=min_vacuum_size / 2.0,
        periodic=True,
    )
    atoms.pbc = [True, True, True]
    return atoms
