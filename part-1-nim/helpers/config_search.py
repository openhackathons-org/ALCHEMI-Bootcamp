"""AdsorbML-style configuration grid generator.

Given (slab, adsorbate) generate N starting configurations on a
systematic grid:

  - **Sites**: top, bridge, fcc-hollow, hcp-hollow (for fcc metals);
    Al-top, O-top, bridge, hollow (for alpha-Al2O3).
  - **Orientations**: rotations about the surface normal + molecule-
    dependent tilts (C-down / O-down for CO, O-down / H-down for H2O,
    O-down / methyl-down for CH3OH).
  - **Heights**: initial z-displacement along the surface normal.

Returns a list of (label, ase.Atoms, active_mask) tuples ready to feed
into a BGR batch relaxation. Each label encodes (adsorbate, host,
site, orientation, height) so the downstream analysis can group and
compare configurations.

Reference: Lan, Palizhati, Hong, Ulissi, Bhowmik, Zitnick 2023,
*npj Computational Materials* 9, 172 (AdsorbML). The 50% single-
starting-point reliability and 87% batch-search reliability headline
numbers come from the benchmark in that paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .surfaces import find_central_site, make_active_mask

if TYPE_CHECKING:
    import ase

# ---------------------------------------------------------------------------
# Adsorbate builders (canonical geometries via ASE)
# ---------------------------------------------------------------------------


def build_co(orient: str = "C-down") -> ase.Atoms:
    """Carbon monoxide, linear, 1.128 A C-O bond (exp).

    `orient` picks which atom sits at the origin (and thus closest to
    the surface when placed via place_adsorbate).
    """
    from ase import Atoms

    d_co = 1.128
    if orient == "C-down":  # C at origin, O up
        return Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d_co]])
    if orient == "O-down":  # O at origin, C up
        return Atoms("OC", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d_co]])
    raise ValueError(f"CO orient must be 'C-down' or 'O-down', got {orient!r}")


def build_h2o(orient: str = "O-down") -> ase.Atoms:
    """Water molecule via ase.build.molecule('H2O'), oriented for placement.

    orient = 'O-down'   -> oxygen at origin, dipole pointing up
    orient = 'H-down'   -> flipped, OH vectors pointing toward surface
    orient = 'flat'     -> molecular plane parallel to the surface
    """
    from ase.build import molecule

    m = molecule("H2O")
    # ASE ships H2O with O at ~(0, 0, 0.119); put O at the origin.
    m.translate(-m.positions[0])
    if orient == "O-down":
        return m
    if orient == "H-down":
        m.rotate(180, "x", center=(0, 0, 0))
        return m
    if orient == "flat":
        m.rotate(90, "x", center=(0, 0, 0))
        return m
    raise ValueError(f"H2O orient must be O-down / H-down / flat, got {orient!r}")


def build_methanol(orient: str = "O-down") -> ase.Atoms:
    """Methanol (CH3OH) from ase.build.molecule.

    orient = 'O-down'      -> oxygen at origin, CH3 group pointing up
    orient = 'methyl-down' -> CH3 nearest the surface, OH up
    """
    from ase.build import molecule

    m = molecule("CH3OH")
    symbols = m.get_chemical_symbols()
    o_idx = symbols.index("O")
    c_idx = symbols.index("C")
    if orient == "O-down":
        m.translate(-m.positions[o_idx])
        # default orientation from ASE has CH3 already above O, leave it
        return m
    if orient == "methyl-down":
        m.translate(-m.positions[c_idx])
        m.rotate(180, "x", center=(0, 0, 0))
        return m
    raise ValueError(
        f"methanol orient must be O-down / methyl-down, got {orient!r}"
    )


ADSORBATE_REGISTRY: dict[str, Callable[[str], "ase.Atoms"]] = {
    "CO": build_co,
    "H2O": build_h2o,
    "CH3OH": build_methanol,
}

ADSORBATE_ORIENTATIONS: dict[str, list[str]] = {
    "CO": ["C-down", "O-down"],
    "H2O": ["O-down", "H-down", "flat"],
    "CH3OH": ["O-down", "methyl-down"],
}


# ---------------------------------------------------------------------------
# Surface site finders
# ---------------------------------------------------------------------------


def _top_layer_positions(slab: ase.Atoms, top_fraction: float = 0.25) -> np.ndarray:
    """Positions of slab atoms in the top `top_fraction` of the z-range."""
    z = slab.positions[:, 2]
    cut = z.min() + (1 - top_fraction) * (z.max() - z.min())
    return slab.positions[z >= cut]


def find_fcc_sites(slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    """Return representative {top, bridge, fcc, hcp} sites for an fcc(111) slab.

    For the tutorial we return only a small number of sites per class
    (all clustered around the slab's central region), not every
    symmetry-distinct site in the cell. That's enough to generate the
    ~12 starting configurations AdsorbML prescribes without bloating
    the batch.
    """
    top_pos = _top_layer_positions(slab, top_fraction=0.15)
    if len(top_pos) < 3:
        raise ValueError("Slab top layer has fewer than 3 atoms; cannot enumerate sites.")

    # Central-most top atom -> top site
    central_idx = int(np.argmin(
        np.linalg.norm(top_pos[:, :2] - (slab.cell.array[0, :2] + slab.cell.array[1, :2]) / 2, axis=1)
    ))
    top_site = top_pos[central_idx].copy()

    # Distances from the central top atom to every other top-layer atom
    d = np.linalg.norm(top_pos - top_site, axis=1)
    d[central_idx] = np.inf
    nearest3 = np.argsort(d)[:3]
    neighbours = top_pos[nearest3]

    # Bridge = midpoint of central <-> nearest neighbour
    bridge_site = (top_site + neighbours[0]) / 2.0
    # fcc hollow = centroid of three nearest neighbours (no second-layer atom directly below)
    fcc_site = neighbours.mean(axis=0)
    # hcp hollow = centroid of central + two nearest neighbours (second-layer atom directly below)
    hcp_site = (top_site + neighbours[0] + neighbours[1]) / 3.0

    return {
        "top": [top_site],
        "bridge": [bridge_site],
        "fcc": [fcc_site],
        "hcp": [hcp_site],
    }


def find_al2o3_0001_sites(slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    """Return representative sites on an alpha-Al2O3(0001) surface.

    alpha-Al2O3(0001) exposes a pattern of aluminium atoms (Al-top),
    oxygen atoms (O-top), two inequivalent bridges (Al-O and O-O), and
    a hollow site at the centre of the hexagonal motif. We return one
    representative of each class near the slab centre.
    """
    top_pos = _top_layer_positions(slab, top_fraction=0.12)
    numbers = slab.numbers
    z_min, z_max = slab.positions[:, 2].min(), slab.positions[:, 2].max()
    cut = z_min + 0.88 * (z_max - z_min)
    top_mask = slab.positions[:, 2] >= cut

    al_top_pos = slab.positions[top_mask & (numbers == 13)]
    o_top_pos = slab.positions[top_mask & (numbers == 8)]
    if len(al_top_pos) == 0 or len(o_top_pos) == 0:
        # (0001) termination is O-heavy; fall back to whatever the top layer has
        fallback = find_central_site(top_pos, slab.cell.array)
        return {"top": [fallback], "bridge": [fallback], "hollow": [fallback]}

    al_site = find_central_site(al_top_pos, slab.cell.array)
    o_site = find_central_site(o_top_pos, slab.cell.array)
    bridge = (al_site + o_site) / 2.0
    hollow = find_central_site(top_pos, slab.cell.array)
    return {
        "al-top": [al_site],
        "o-top": [o_site],
        "bridge": [bridge],
        "hollow": [hollow],
    }


def sites_for_host(host_name: str, slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    if host_name in ("Cu(111)", "Pd(111)"):
        return find_fcc_sites(slab)
    if host_name == "Al2O3(0001)":
        return find_al2o3_0001_sites(slab)
    raise ValueError(f"No site finder registered for host {host_name!r}")


# ---------------------------------------------------------------------------
# Placement + grid generation
# ---------------------------------------------------------------------------


def _surface_normal(slab: ase.Atoms) -> np.ndarray:
    a_vec, b_vec = slab.cell[0], slab.cell[1]
    n = np.cross(a_vec, b_vec)
    return n / np.linalg.norm(n)


def _place(
    slab: ase.Atoms,
    ads: ase.Atoms,
    site: np.ndarray,
    height: float,
    rot_deg: float,
) -> ase.Atoms:
    """Place `ads` above `site` at `height`, rotated by `rot_deg` about z."""
    a = ads.copy()
    if abs(rot_deg) > 1e-6:
        a.rotate(rot_deg, "z", center=(0, 0, 0))
    normal = _surface_normal(slab)
    a.translate(site + height * normal)
    combined = slab.copy() + a
    return combined


@dataclass
class Configuration:
    label: str
    host: str
    adsorbate: str
    site: str
    orientation: str
    rot_deg: float
    height: float
    atoms: ase.Atoms
    active_mask: list[bool]


def build_config_grid(
    host_name: str,
    slab: ase.Atoms,
    adsorbate_name: str,
    sites_filter: list[str] | None = None,
    orientations_filter: list[str] | None = None,
    rotations_deg: tuple[float, ...] = (0.0, 60.0, 120.0),
    heights_A: tuple[float, ...] = (2.2,),
    frozen_fraction: float = 0.5,
) -> list[Configuration]:
    """Generate (site × orientation × rotation × height) starting configs.

    The four axes multiply: for an fcc metal with 4 sites, 3 rotations,
    1 height, and adsorbate-specific orientations (2 for CO, 3 for H2O,
    2 for CH3OH), you get 4*3*1*{2 or 3 or 2} = 24, 36, 24 configs per
    pair. Filters let the notebook scale down to ~12 for runtime or up
    to ~40 for accuracy.
    """
    if adsorbate_name not in ADSORBATE_REGISTRY:
        raise ValueError(f"Unknown adsorbate: {adsorbate_name}")

    site_map = sites_for_host(host_name, slab)
    if sites_filter is not None:
        site_map = {k: v for k, v in site_map.items() if k in sites_filter}
    orientations = ADSORBATE_ORIENTATIONS[adsorbate_name]
    if orientations_filter is not None:
        orientations = [o for o in orientations if o in orientations_filter]

    base_mask = make_active_mask(slab, bottom_fraction=frozen_fraction)
    configs: list[Configuration] = []
    for site_name, positions in site_map.items():
        for pos in positions:
            for orient in orientations:
                ads = ADSORBATE_REGISTRY[adsorbate_name](orient)
                for rot in rotations_deg:
                    for h in heights_A:
                        atoms = _place(slab, ads, pos, height=h, rot_deg=rot)
                        mask = base_mask + [True] * len(ads)
                        label = (
                            f"{adsorbate_name}_{host_name}_"
                            f"{site_name}_{orient}_rot{int(rot)}_h{h:.1f}"
                        )
                        configs.append(Configuration(
                            label=label,
                            host=host_name,
                            adsorbate=adsorbate_name,
                            site=site_name,
                            orientation=orient,
                            rot_deg=float(rot),
                            height=float(h),
                            atoms=atoms,
                            active_mask=mask,
                        ))
    return configs
