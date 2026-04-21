"""Oxide bulk and slab builders for the AWH water-sorbent tutorial.

Provides bulk cells and (Miller-indexed) slab constructions for the three
ionic-class hosts in the S24 validation panel of Batatia 2024:

* alpha-Al2O3 (corundum, space group R-3c, #167) - (0001) surface
* TiO2 rutile (P4_2/mnm, #136) - (110) surface
* ZrO2 monoclinic baddeleyite (P2_1/c, #14) - (-1,1,1) surface

All bulks use experimental lattice parameters and Wyckoff positions
from the crystallographic literature. Slab construction delegates to
pymatgen's SlabGenerator via the generic :func:`helpers.surfaces.build_slab`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .surfaces import build_slab

if TYPE_CHECKING:
    import ase
    import pymatgen.core


# ---------------------------------------------------------------------------
# Bulk cells
# ---------------------------------------------------------------------------


def build_alpha_alumina_bulk() -> pymatgen.core.Structure:
    """Build alpha-Al2O3 corundum bulk (R-3c, hexagonal setting).

    Lattice parameters from room-temperature X-ray refinement:
    a = 4.7607 A, c = 12.9947 A.

    References
    ----------
    Lewis et al., "Refinement of the atomic parameters of corundum",
    Acta Cryst. B 38, 1018-1019 (1982).
    """
    from pymatgen.core import Lattice, Structure

    a, c = 4.7607, 12.9947
    return Structure.from_spacegroup(
        "R-3c",
        Lattice.hexagonal(a, c),
        ["Al", "O"],
        [
            [0.0, 0.0, 0.35216],  # Al on 12c
            [0.30624, 0.0, 0.25],  # O on 18e
        ],
    )


def build_rutile_tio2_bulk() -> pymatgen.core.Structure:
    """Build TiO2 rutile bulk (P4_2/mnm, #136).

    Lattice parameters from neutron diffraction:
    a = 4.594 A, c = 2.958 A, O internal coord u = 0.3048.

    References
    ----------
    Bolzan, Fong, Kennedy, Howard, "Powder neutron diffraction study of
    pyrolusite, beta-MnO2", Acta Cryst. B 53, 373-380 (1997).
    """
    from pymatgen.core import Lattice, Structure

    a, c, u = 4.594, 2.958, 0.3048
    return Structure.from_spacegroup(
        136,
        Lattice.tetragonal(a, c),
        ["Ti", "O"],
        [[0.0, 0.0, 0.0], [u, u, 0.0]],
    )


def build_monoclinic_zro2_bulk() -> pymatgen.core.Structure:
    """Build monoclinic ZrO2 (baddeleyite) bulk (P2_1/c, #14).

    Lattice parameters from single-crystal neutron diffraction:
    a = 5.1454 A, b = 5.2075 A, c = 5.3107 A, beta = 99.23 deg.

    References
    ----------
    Howard, Hill, Reichert, "Structures of ZrO2 polymorphs at room
    temperature by high-resolution neutron powder diffraction",
    Acta Cryst. B 44, 116-120 (1988).
    """
    from pymatgen.core import Lattice, Structure

    a, b, c = 5.1454, 5.2075, 5.3107
    beta = 99.23
    lattice = Lattice.from_parameters(a, b, c, 90.0, beta, 90.0)
    # Wyckoff 4e positions for Zr, O1, O2 (Howard 1988, Table 2)
    return Structure.from_spacegroup(
        14,
        lattice,
        ["Zr", "O", "O"],
        [
            [0.2758, 0.0404, 0.2084],
            [0.0700, 0.3317, 0.3447],
            [0.4487, 0.7569, 0.4792],
        ],
    )


# ---------------------------------------------------------------------------
# Slabs
# ---------------------------------------------------------------------------


def build_alpha_alumina_0001_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (1, 1, 1),
) -> ase.Atoms:
    """Build an alpha-Al2O3 (0001) slab.

    The (0001) termination is the thermodynamically stable basal plane of
    corundum and the one referenced in S24 (Ionic sub-category).
    """
    bulk = build_alpha_alumina_bulk()
    return build_slab(
        bulk,
        miller_index=(0, 0, 1),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        supercell=supercell,
    )


def build_tio2_110_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (2, 2, 1),
) -> ase.Atoms:
    """Build a TiO2 rutile (110) slab.

    The (110) surface is the lowest-energy rutile termination and the
    one referenced in S24.
    """
    bulk = build_rutile_tio2_bulk()
    return build_slab(
        bulk,
        miller_index=(1, 1, 0),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        supercell=supercell,
    )


def build_zro2_m111_slab(
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 15.0,
    supercell: tuple[int, int, int] = (1, 1, 1),
) -> ase.Atoms:
    """Build a monoclinic ZrO2 (-1,1,1) slab.

    The (-1,1,1) surface is the most stable termination of baddeleyite
    ZrO2 at room temperature and is closed-shell singlet (no magnetic
    or reducible cations). No published DFT checkpoint exists for H2O
    adsorption on this surface in the S24 panel - it is the Tier-4
    "stretch" host in the tutorial.

    References
    ----------
    Christensen, Carter, "First-principles study of the surfaces of
    zirconia", Phys. Rev. B 58, 8050-8064 (1998).
    """
    bulk = build_monoclinic_zro2_bulk()
    return build_slab(
        bulk,
        miller_index=(-1, 1, 1),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        supercell=supercell,
    )
