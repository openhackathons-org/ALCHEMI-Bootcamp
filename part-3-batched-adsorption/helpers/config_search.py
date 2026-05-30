"""AdsorbML-style configuration grid generator.

This generic helper supports both the Cu/TiO2/TiN nine-surface adsorption
screen and Cu/Pd/Al2O3 auxiliary examples. The alpha-Al2O3 site path is
retained for Cu/Pd/Al2O3 auxiliary scripts and is not part of the
Cu/TiO2/TiN nine-surface adsorption screen.

Given (slab, adsorbate) generate N starting configurations on a
systematic grid:

  - **Sites**: top, bridge, fcc-hollow, hcp-hollow (for fcc metals);
    Al-top, O-top, bridge, hollow (for alpha-Al2O3).
  - **Orientations**: rotations about the surface normal + molecule-
    dependent tilts (C-down / O-down for CO, O-down / H-down for H2O,
    O-down / methyl-down for CH3OH, N-down / H-down for NH3).
  - **Heights**: initial z-displacement along the surface normal.

Returns a list of (label, ase.Atoms, active_mask) tuples ready to feed
into a Toolkit batch relaxation. Each label encodes (adsorbate, host,
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

from .metal_slabs import build_cu111_slab
from .oxide_slabs import build_alpha_alumina_0001_slab, build_tio2_110_slab
from .surfaces import find_central_site, make_active_mask

HOST_BUILDERS = {
    "Cu(111)": lambda: build_cu111_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(3, 3, 1),
    ),
    "Al2O3(0001)": lambda: build_alpha_alumina_0001_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(2, 2, 1),
    ),
    "TiO2(110)": lambda: build_tio2_110_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(2, 2, 1),
    ),
}

if TYPE_CHECKING:
    import ase

# ---------------------------------------------------------------------------
# Adsorbate builders (canonical geometries via ASE)
# ---------------------------------------------------------------------------


def _maximin_anchor_direction(atoms: ase.Atoms, anchor_idx: int) -> np.ndarray:
    """Return a direction that makes *anchor_idx* an exposed lowest atom.

    Methanol's ASE geometry is not pre-oriented for surface adsorption.  For
    O-down starts, simply aligning the C-O bond along z leaves the hydroxyl H
    closer to the surface than O.  A deterministic Fibonacci-sphere search is
    cheap for these small molecules and picks a surface normal for which the
    anchored atom is genuinely the lowest atom.
    """
    rel = atoms.positions - atoms.positions[anchor_idx]
    rel = np.delete(rel, anchor_idx, axis=0)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    best_score = -np.inf
    best = np.array([0.0, 0.0, 1.0])
    for i in range(512):
        z = 1.0 - 2.0 * (i + 0.5) / 512
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        direction = np.array([np.cos(theta) * radius, np.sin(theta) * radius, z])
        score = float(np.min(rel @ direction))
        if score > best_score:
            best_score = score
            best = direction
    if best_score <= 1e-6:
        raise ValueError(
            "Requested anchor atom is not an exposed atom for this molecular geometry."
        )
    return best


def _orient_anchor_lowest(atoms: ase.Atoms, anchor_idx: int) -> ase.Atoms:
    """Translate and rotate a molecule so *anchor_idx* is lowest at z = 0."""
    atoms.translate(-atoms.positions[anchor_idx])
    direction = _maximin_anchor_direction(atoms, anchor_idx)
    atoms.rotate(direction, (0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0))
    atoms.positions[anchor_idx] = [0.0, 0.0, 0.0]
    return atoms


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
        h_idx = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "H"]
        if h_idx and np.mean(m.positions[h_idx, 2]) < 0.0:
            m.rotate(180, "x", center=(0, 0, 0))
        return m
    if orient == "H-down":
        h_idx = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "H"]
        if h_idx and np.mean(m.positions[h_idx, 2]) > 0.0:
            m.rotate(180, "x", center=(0, 0, 0))
        return m
    if orient == "flat":
        # The ASE water geometry lies in a vertical plane after O-centering.
        # Rotate that plane into xy so no atom is deliberately closest to
        # the surface before relaxation.
        m.rotate(90, "y", center=(0, 0, 0))
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
        return _orient_anchor_lowest(m, o_idx)
    if orient == "methyl-down":
        m.translate(-m.positions[c_idx])
        # Point the C-O bond away from the surface.  The methyl H atoms, not
        # the tetrahedral carbon center, are the closest atoms in this decoy
        # orientation.
        m.rotate(m.positions[o_idx] - m.positions[c_idx], (0, 0, 1), center=(0, 0, 0))
        return m
    raise ValueError(f"methanol orient must be O-down / methyl-down, got {orient!r}")


def build_nh3(orient: str = "N-down") -> ase.Atoms:
    """Ammonia (NH3) from ase.build.molecule, oriented for placement.

    orient = 'N-down' -> nitrogen lone-pair donor points toward surface
    orient = 'H-down' -> one or more hydrogens point toward surface
    orient = 'flat'   -> N-H pyramid axis roughly parallel to surface
    """
    from ase.build import molecule

    m = molecule("NH3")
    symbols = m.get_chemical_symbols()
    n_idx = symbols.index("N")
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]
    m.translate(-m.positions[n_idx])
    if orient == "N-down":
        # ASE's NH3 has the H triangle below N; invert so N is the contact atom.
        m.rotate(180, "x", center=(0, 0, 0))
        return m
    if orient == "H-down":
        # Keep ASE's H triangle closest to the surface.
        return m
    if orient == "flat":
        # Put the N -> H-centroid vector roughly parallel to the surface.
        if h_idx:
            axis = m.positions[h_idx].mean(axis=0) - m.positions[n_idx]
            if np.linalg.norm(axis) > 1e-12:
                m.rotate(90, "y", center=(0, 0, 0))
        return m
    raise ValueError(f"NH3 orient must be N-down / H-down / flat, got {orient!r}")


ADSORBATE_REGISTRY: dict[str, Callable[[str], "ase.Atoms"]] = {
    "CO": build_co,
    "H2O": build_h2o,
    "CH3OH": build_methanol,
    "NH3": build_nh3,
}

ADSORBATE_ORIENTATIONS: dict[str, list[str]] = {
    "CO": ["C-down", "O-down"],
    "H2O": ["O-down", "H-down", "flat"],
    "CH3OH": ["O-down", "methyl-down"],
    "NH3": ["N-down", "H-down", "flat"],
}

# ---------------------------------------------------------------------------
# Surface site finders
# ---------------------------------------------------------------------------


def _top_layer_positions(slab: ase.Atoms, top_fraction: float = 0.25) -> np.ndarray:
    """Positions of slab atoms in the top `top_fraction` of the z-range."""
    z = slab.positions[:, 2]
    cut = z.min() + (1 - top_fraction) * (z.max() - z.min())
    return slab.positions[z >= cut]


def _xy_delta_minimum_image(
    p_xy: np.ndarray,
    q_xy: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(p_xy, dtype=float) - np.asarray(q_xy, dtype=float)
    basis = np.vstack([cell[0, :2], cell[1, :2]]).T
    try:
        frac = np.linalg.solve(basis, delta)
    except np.linalg.LinAlgError:
        return delta
    frac -= np.round(frac)
    return basis @ frac


def _wrap_xy_into_cell(xy: np.ndarray, cell: np.ndarray) -> np.ndarray:
    basis = np.vstack([cell[0, :2], cell[1, :2]]).T
    frac = np.linalg.solve(basis, np.asarray(xy, dtype=float))
    frac -= np.floor(frac)
    return basis @ frac


def _unique_periodic_positions(
    positions: list[np.ndarray],
    cell: np.ndarray,
    tol_A: float = 0.25,
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for position in positions:
        wrapped = position.copy()
        wrapped[:2] = _wrap_xy_into_cell(wrapped[:2], cell)
        if not any(
            np.linalg.norm(_xy_delta_minimum_image(wrapped[:2], other[:2], cell))
            < tol_A
            for other in unique
        ):
            unique.append(wrapped)
    return unique


def _z_layers(positions: np.ndarray, tol_A: float = 0.6) -> list[np.ndarray]:
    if len(positions) == 0:
        return []
    ordered = sorted(positions, key=lambda p: float(p[2]), reverse=True)
    layers: list[list[np.ndarray]] = []
    layer_ref_z: list[float] = []
    for position in ordered:
        z = float(position[2])
        if not layers or abs(z - layer_ref_z[-1]) > tol_A:
            layers.append([position])
            layer_ref_z.append(z)
        else:
            layers[-1].append(position)
    return [np.asarray(layer, dtype=float) for layer in layers]


def fcc111_site_candidates(slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    """Enumerate all periodic top/bridge/fcc/hcp candidates in an fcc(111) cell.

    hcp hollows are identified by a second-layer atom below the hollow xy
    position. fcc hollows are the complementary threefold hollows.
    """
    cell = slab.cell.array
    layers = _z_layers(np.asarray(slab.positions, dtype=float))
    if len(layers) < 2 or len(layers[0]) < 3:
        raise ValueError("Need at least two fcc(111) layers for site enumeration.")

    top = layers[0]
    second = layers[1]
    top_z = float(np.max(top[:, 2]))
    shifts = [
        ia * cell[0, :2] + ib * cell[1, :2] for ia in (-1, 0, 1) for ib in (-1, 0, 1)
    ]
    images = [
        (idx, atom[:2] + shift) for idx, atom in enumerate(top) for shift in shifts
    ]

    distances: list[float] = []
    for idx, atom in enumerate(top):
        for image_idx, image_xy in images:
            d = float(np.linalg.norm(image_xy - atom[:2]))
            if (image_idx != idx or d > 1e-6) and d > 1e-6:
                distances.append(d)
    if not distances:
        raise ValueError(
            "Cannot determine fcc(111) in-plane nearest-neighbour distance."
        )
    nn = min(distances)
    neighbour_cut = nn * 1.15

    bridge_candidates: list[np.ndarray] = []
    hollow_candidates: list[np.ndarray] = []
    for atom in top:
        neighbours = [
            image_xy
            for _, image_xy in images
            if 1e-6 < float(np.linalg.norm(image_xy - atom[:2])) < neighbour_cut
        ]
        for neighbour_xy in neighbours:
            bridge_candidates.append(
                np.array(
                    [
                        0.5 * (atom[0] + neighbour_xy[0]),
                        0.5 * (atom[1] + neighbour_xy[1]),
                        top_z,
                    ]
                )
            )
        for i, first_xy in enumerate(neighbours):
            for second_xy in neighbours[i + 1 :]:
                if float(np.linalg.norm(first_xy - second_xy)) < neighbour_cut:
                    hollow_candidates.append(
                        np.array(
                            [
                                (atom[0] + first_xy[0] + second_xy[0]) / 3.0,
                                (atom[1] + first_xy[1] + second_xy[1]) / 3.0,
                                top_z,
                            ]
                        )
                    )

    top_candidates = _unique_periodic_positions(
        [np.array([p[0], p[1], top_z]) for p in top],
        cell,
    )
    bridge_candidates = _unique_periodic_positions(bridge_candidates, cell)
    hollow_candidates = _unique_periodic_positions(hollow_candidates, cell)

    fcc_candidates: list[np.ndarray] = []
    hcp_candidates: list[np.ndarray] = []
    for hollow in hollow_candidates:
        second_layer_distance = min(
            float(np.linalg.norm(_xy_delta_minimum_image(hollow[:2], atom[:2], cell)))
            for atom in second
        )
        if second_layer_distance < nn * 0.3:
            hcp_candidates.append(hollow)
        else:
            fcc_candidates.append(hollow)

    return {
        "top": top_candidates,
        "bridge": bridge_candidates,
        "fcc": fcc_candidates,
        "hcp": hcp_candidates,
    }


def find_fcc_sites(slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    """Return representative {top, bridge, fcc, hcp} sites for an fcc(111) slab.

    For the tutorial we return only a small number of sites per class
    (all clustered around the slab's central region), not every
    symmetry-distinct site in the cell. That's enough to generate the
    ~12 starting configurations AdsorbML prescribes without bloating
    the batch.
    """
    site_map = fcc111_site_candidates(slab)
    return {
        name: [find_central_site(np.asarray(positions), slab.cell.array)]
        for name, positions in site_map.items()
    }


def _nearest_top_neighbours(
    top: np.ndarray,
    central: np.ndarray,
    cell: np.ndarray,
    *,
    count: int = 2,
) -> list[np.ndarray]:
    neighbours: list[tuple[float, np.ndarray]] = []
    for candidate in top:
        delta = _xy_delta_minimum_image(candidate[:2], central[:2], cell)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-8:
            continue
        image = central.copy()
        image[:2] = central[:2] + delta
        image[2] = candidate[2]
        neighbours.append((distance, image))
    neighbours.sort(key=lambda item: item[0])
    return [image for _, image in neighbours[:count]]


def find_elemental_surface_sites(slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    """Return top, bridge, and hollow/trough sites for a one-element slab."""
    top = _top_layer_positions(slab, top_fraction=0.16)
    if len(top) < 3:
        top = _top_layer_positions(slab, top_fraction=0.35)
    if len(top) < 3:
        raise ValueError("Need at least three top-layer atoms for elemental sites.")

    cell = slab.cell.array
    top_site = find_central_site(top, cell)
    neighbours = _nearest_top_neighbours(top, top_site, cell, count=2)
    if len(neighbours) < 2:
        raise ValueError("Cannot identify bridge/hollow sites on elemental slab.")

    bridge = top_site.copy()
    bridge[:2] = 0.5 * (top_site[:2] + neighbours[0][:2])
    bridge[2] = top_site[2]

    hollow = top_site.copy()
    hollow[:2] = (top_site[:2] + neighbours[0][:2] + neighbours[1][:2]) / 3.0
    hollow[2] = top_site[2]
    return {
        "top": [top_site],
        "bridge": [bridge],
        "hollow": [hollow],
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


def find_binary_surface_sites(
    slab: ase.Atoms,
    *,
    cation_symbol: str,
    cation_site_name: str,
    anion_symbol: str,
    anion_site_name: str,
) -> dict[str, list[np.ndarray]]:
    """Return representative top-cation, top-anion, bridge, and hollow sites.

    This finder is deliberately simple: it uses the generated slab geometry to
    pick central sites from top-layer cations and anions. That keeps Miller
    examples programmatic without inventing token-written coordinates.
    """
    from ase.data import atomic_numbers

    cation_number = atomic_numbers[cation_symbol]
    anion_number = atomic_numbers[anion_symbol]
    numbers = slab.numbers
    z_min, z_max = slab.positions[:, 2].min(), slab.positions[:, 2].max()

    for top_fraction in (0.16, 0.25, 0.35, 0.50):
        cut = z_min + (1 - top_fraction) * (z_max - z_min)
        top_mask = slab.positions[:, 2] >= cut
        cation_top_pos = slab.positions[top_mask & (numbers == cation_number)]
        anion_top_pos = slab.positions[top_mask & (numbers == anion_number)]
        if len(cation_top_pos) > 0 and len(anion_top_pos) > 0:
            break
    else:
        fallback = find_central_site(_top_layer_positions(slab), slab.cell.array)
        return {
            cation_site_name: [fallback],
            anion_site_name: [fallback],
            "bridge": [fallback],
            "hollow": [fallback],
        }

    top_pos = slab.positions[top_mask]
    cation_site = find_central_site(cation_top_pos, slab.cell.array)
    anion_site = find_central_site(anion_top_pos, slab.cell.array)
    bridge = (cation_site + anion_site) / 2.0
    hollow = find_central_site(top_pos, slab.cell.array)
    candidates = {
        cation_site_name: [cation_site],
        anion_site_name: [anion_site],
        "bridge": [bridge],
        "hollow": [hollow],
    }
    distinct: dict[str, list[np.ndarray]] = {}
    for name, positions in candidates.items():
        position = positions[0]
        if any(
            abs(float(position[2] - existing[0][2])) < 0.25
            and np.linalg.norm(
                _xy_delta_minimum_image(position[:2], existing[0][:2], slab.cell.array)
            )
            < 0.35
            for existing in distinct.values()
        ):
            continue
        distinct[name] = positions
    return distinct


def find_oxide_sites(
    slab: ase.Atoms,
    *,
    cation_symbol: str,
    cation_site_name: str,
) -> dict[str, list[np.ndarray]]:
    """Return representative top-cation, top-oxygen, bridge, and hollow sites."""
    return find_binary_surface_sites(
        slab,
        cation_symbol=cation_symbol,
        cation_site_name=cation_site_name,
        anion_symbol="O",
        anion_site_name="o-top",
    )


def sites_for_host(host_name: str, slab: ase.Atoms) -> dict[str, list[np.ndarray]]:
    if host_name in ("Cu(111)", "Pd(111)"):
        return find_fcc_sites(slab)
    if host_name in ("Cu(100)", "Cu(110)"):
        return find_elemental_surface_sites(slab)
    if host_name == "Al2O3(0001)":
        return find_al2o3_0001_sites(slab)
    if host_name in ("TiO2(110)", "TiO2(100)", "TiO2(101)"):
        return find_oxide_sites(slab, cation_symbol="Ti", cation_site_name="ti-top")
    if host_name in ("TiN(001)", "TiN(110)", "TiN(210)"):
        return find_binary_surface_sites(
            slab,
            cation_symbol="Ti",
            cation_site_name="ti-top",
            anion_symbol="N",
            anion_site_name="n-top",
        )
    if host_name == "ZrO2(-1,1,1)":
        return find_oxide_sites(slab, cation_symbol="Zr", cation_site_name="zr-top")
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
    adsorbate_name: str,
    slab: ase.Atoms | None = None,
    sites_filter: list[str] | None = None,
    orientations_filter: list[str] | None = None,
    rotations_deg: tuple[float, ...] = (0.0, 60.0, 120.0),
    heights_A: tuple[float, ...] = (2.2,),
    frozen_fraction: float = 0.5,
) -> list[Configuration]:
    """Generate (site × orientation × rotation × height) starting configs.

    The four axes multiply: for an fcc metal with 4 sites, 3 rotations,
    1 height, and adsorbate-specific orientations (2 for CO, 3 for H2O,
    2 for CH3OH, 3 for NH3), you get 4*3*1*{2 or 3 or 2 or 3}
    configs per pair. Filters let the notebook scale down to ~12 for
    runtime or up to ~40 for accuracy. When ``slab`` is omitted, the
    canonical slab for ``host_name`` is built via ``HOST_BUILDERS``.
    """
    if slab is None:
        try:
            slab = HOST_BUILDERS[host_name]()
        except KeyError as exc:
            raise ValueError(f"Unknown host {host_name!r}") from exc
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
                        configs.append(
                            Configuration(
                                label=label,
                                host=host_name,
                                adsorbate=adsorbate_name,
                                site=site_name,
                                orientation=orient,
                                rot_deg=float(rot),
                                height=float(h),
                                atoms=atoms,
                                active_mask=mask,
                            )
                        )
    return configs
