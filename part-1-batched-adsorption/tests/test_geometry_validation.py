"""Executable geometry checks for the AdsorbML tutorial panel."""

import math

import numpy as np
import pytest

from helpers import (
    LATTICE_A_CU,
    LATTICE_A_PD,
    ADSORBATE_ORIENTATIONS,
    build_alpha_alumina_0001_slab,
    build_co,
    build_config_grid,
    build_cu100_slab,
    build_cu111_slab,
    build_cu110_slab,
    build_h2o,
    build_methanol,
    build_nh3,
    build_pd111_slab,
    build_tin_001_slab,
    build_tin_110_slab,
    build_tin_210_slab,
    build_tio2_100_slab,
    build_tio2_101_slab,
    build_tio2_110_slab,
    find_al2o3_0001_sites,
    find_fcc_sites,
    make_active_mask,
    sites_for_host,
    surface_screen_expected_counts,
)


SCREEN_HOST_BUILDERS = {
    "Cu(111)": lambda: build_cu111_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
    "Cu(100)": lambda: build_cu100_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
    "Cu(110)": lambda: build_cu110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
    "TiO2(110)": lambda: build_tio2_110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "TiO2(100)": lambda: build_tio2_100_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "TiO2(101)": lambda: build_tio2_101_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "TiN(001)": lambda: build_tin_001_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "TiN(110)": lambda: build_tin_110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "TiN(210)": lambda: build_tin_210_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
}
SCREEN_ADSORBATES = ("CO", "H2O", "NH3", "CH3OH")
SCREEN_SITES_BY_HOST = {
    "Cu(111)": ("top", "bridge", "fcc"),
    "Cu(100)": ("top", "bridge", "hollow"),
    "Cu(110)": ("top", "bridge", "hollow"),
    "TiO2(110)": ("ti-top", "o-top", "bridge"),
    "TiO2(100)": ("ti-top", "o-top", "bridge"),
    "TiO2(101)": ("ti-top", "o-top", "bridge"),
    "TiN(001)": ("ti-top", "n-top", "bridge"),
    "TiN(110)": ("ti-top", "n-top", "bridge"),
    "TiN(210)": ("ti-top", "n-top", "bridge"),
}
SCREEN_ORIENTATIONS_BY_ADSORBATE = {
    "CO": ("C-down", "O-down"),
    "H2O": ("O-down", "H-down"),
    "NH3": ("N-down", "H-down"),
    "CH3OH": ("O-down", "methyl-down"),
}


def _screen_slabs():
    return {name: builder() for name, builder in SCREEN_HOST_BUILDERS.items()}


def _screen_configs(slabs):
    configs = []
    for host, slab in slabs.items():
        for adsorbate in SCREEN_ADSORBATES:
            configs.extend(
                build_config_grid(
                    host_name=host,
                    slab=slab,
                    adsorbate_name=adsorbate,
                    sites_filter=list(SCREEN_SITES_BY_HOST[host]),
                    orientations_filter=list(SCREEN_ORIENTATIONS_BY_ADSORBATE[adsorbate]),
                    rotations_deg=(0.0,),
                    heights_A=(2.5,),
                )
            )
    return configs


def _z_layers(atoms, tol=0.35):
    """Group atoms into approximate z-layers."""
    layers = []
    for z in sorted(atoms.positions[:, 2]):
        if not layers or abs(z - layers[-1][-1]) > tol:
            layers.append([float(z)])
        else:
            layers[-1].append(float(z))
    return layers


def _nearest_distance(points):
    best = float("inf")
    for i, p in enumerate(points):
        for q in points[i + 1 :]:
            best = min(best, float(np.linalg.norm(p - q)))
    return best


def _nearest_periodic_xy_distance(point, points, cell):
    """Nearest xy distance under 2-D periodic wrapping."""
    basis = cell.array[:2, :2].T
    inv_basis = np.linalg.inv(basis)
    best = float("inf")
    for candidate in points:
        delta = point[:2] - candidate[:2]
        frac = inv_basis @ delta
        frac -= np.round(frac)
        wrapped = basis @ frac
        best = min(best, float(np.linalg.norm(wrapped)))
    return best


def _initial_adsorbate_clearance(config, slab_n):
    slab = config.atoms[:slab_n]
    ads = config.atoms[slab_n:]
    distances = np.linalg.norm(
        ads.positions[:, None, :] - slab.positions[None, :, :],
        axis=2,
    )
    return float(distances.min())


@pytest.mark.parametrize(
    ("builder", "a0", "symbol"),
    [
        (build_cu111_slab, LATTICE_A_CU, "Cu"),
        (build_pd111_slab, LATTICE_A_PD, "Pd"),
    ],
)
def test_fcc111_slabs_have_expected_layer_spacing_and_surface_lattice(
    builder, a0, symbol
):
    slab = builder(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1))
    assert set(slab.get_chemical_symbols()) == {symbol}
    assert len(slab) == 36

    layers = _z_layers(slab)
    assert len(layers) == 4
    assert [len(layer) for layer in layers] == [9, 9, 9, 9]

    layer_centres = np.array([np.mean(layer) for layer in layers])
    spacings = np.diff(layer_centres)
    assert np.median(spacings) == pytest.approx(a0 / math.sqrt(3), rel=0.06)

    top_cut = layer_centres[-1] - 0.2
    top_layer = slab.positions[slab.positions[:, 2] > top_cut]
    assert _nearest_distance(top_layer[:, :2]) == pytest.approx(
        a0 / math.sqrt(2), rel=0.06
    )

    z_span = float(slab.positions[:, 2].max() - slab.positions[:, 2].min())
    assert slab.cell.lengths()[2] - z_span >= 14.5


def test_alpha_alumina_0001_slab_has_expected_composition_and_sites():
    slab = build_alpha_alumina_0001_slab(min_slab_size=8.0, min_vacuum_size=15.0)
    symbols = slab.get_chemical_symbols()
    assert symbols.count("Al") == 48
    assert symbols.count("O") == 72
    assert slab.cell.lengths()[0] >= 8.0
    assert slab.cell.lengths()[1] >= 8.0

    z_span = float(slab.positions[:, 2].max() - slab.positions[:, 2].min())
    assert slab.cell.lengths()[2] - z_span >= 14.5

    sites = find_al2o3_0001_sites(slab)
    assert set(sites) == {"al-top", "o-top", "bridge", "hollow"}
    assert all(len(v) == 1 for v in sites.values())
    assert np.linalg.norm(sites["al-top"][0][:2] - sites["o-top"][0][:2]) > 0.1
    assert np.linalg.norm(sites["bridge"][0][:2] - sites["al-top"][0][:2]) > 0.05


def test_rutile_tio2_110_slab_has_programmatic_sites():
    slab = build_tio2_110_slab(min_slab_size=8.0, min_vacuum_size=15.0)
    symbols = slab.get_chemical_symbols()
    assert symbols.count("Ti") == 24
    assert symbols.count("O") == 48
    assert len(slab) == 72

    sites = sites_for_host("TiO2(110)", slab)
    assert set(sites) == {"ti-top", "o-top", "bridge"}
    assert all(len(v) == 1 for v in sites.values())
    coords = {name: values[0] for name, values in sites.items()}
    assert np.linalg.norm(coords["ti-top"][:2] - coords["o-top"][:2]) > 0.1
    assert np.linalg.norm(coords["bridge"][:2] - coords["ti-top"][:2]) > 0.05

    configs = build_config_grid(
        host_name="TiO2(110)",
        slab=slab,
        adsorbate_name="H2O",
        orientations_filter=["O-down", "flat"],
        rotations_deg=(0.0,),
        heights_A=(2.2,),
    )
    assert len(configs) == 6


@pytest.mark.parametrize(
    ("host_name", "builder", "expected_sites"),
    [
        ("Cu(100)", build_cu100_slab, {"top", "bridge", "hollow"}),
        ("Cu(110)", build_cu110_slab, {"top", "bridge", "hollow"}),
        ("TiO2(100)", build_tio2_100_slab, {"ti-top", "o-top", "bridge"}),
        ("TiO2(101)", build_tio2_101_slab, {"ti-top", "o-top", "bridge"}),
        ("TiN(001)", build_tin_001_slab, {"ti-top", "n-top", "bridge"}),
        ("TiN(110)", build_tin_110_slab, {"ti-top", "n-top", "bridge"}),
        ("TiN(210)", build_tin_210_slab, {"ti-top", "n-top", "bridge"}),
    ],
)
def test_new_surface_screen_slabs_have_registered_sites(host_name, builder, expected_sites):
    slab = builder(min_slab_size=8.0, min_vacuum_size=15.0)
    sites = sites_for_host(host_name, slab)
    assert expected_sites.issubset(sites)
    assert all(len(values) >= 1 for values in sites.values())


def test_surface_screen_builds_216_programmatic_starting_structures():
    slabs = _screen_slabs()
    configs = _screen_configs(slabs)
    expected = surface_screen_expected_counts(
        n_slabs=len(SCREEN_HOST_BUILDERS),
        n_adsorbates=len(SCREEN_ADSORBATES),
        starts_per_pair=6,
    )
    assert len(slabs) == expected["slabs"]
    assert len(configs) == expected["adsorption_relaxations"]

    counts = {}
    for config in configs:
        counts[(config.host, config.adsorbate)] = counts.get((config.host, config.adsorbate), 0) + 1
    assert set(counts.values()) == {expected["starts_per_pair"]}
    assert len(counts) == expected["adsorbate_surface_pairs"]

    labels = [config.label for config in configs]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize(
    ("host_name", "adsorbate"),
    [
        ("Cu(111)", "CO"),
        ("Cu(100)", "H2O"),
        ("Cu(110)", "NH3"),
        ("TiO2(110)", "CH3OH"),
        ("TiO2(100)", "CO"),
        ("TiO2(101)", "H2O"),
        ("TiN(001)", "NH3"),
        ("TiN(110)", "CH3OH"),
        ("TiN(210)", "CO"),
    ],
)
def test_screen_grid_is_exactly_three_sites_by_two_orientations(host_name, adsorbate):
    slabs = _screen_slabs()
    configs = build_config_grid(
        host_name=host_name,
        slab=slabs[host_name],
        adsorbate_name=adsorbate,
        sites_filter=list(SCREEN_SITES_BY_HOST[host_name]),
        orientations_filter=list(SCREEN_ORIENTATIONS_BY_ADSORBATE[adsorbate]),
        rotations_deg=(0.0,),
        heights_A=(2.5,),
    )
    assert len(configs) == 6
    assert len({config.site for config in configs}) == 3
    assert len({config.orientation for config in configs}) == 2


def test_active_mask_freezes_bottom_half_and_relaxes_adsorbate_atoms():
    slab = build_cu111_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1))
    mask = make_active_mask(slab, bottom_fraction=0.5)
    assert len(mask) == len(slab)

    layers = _z_layers(slab)
    layer_centres = np.array([np.mean(layer) for layer in layers])
    split = np.mean(layer_centres[1:3])
    bottom_idx = np.where(slab.positions[:, 2] < split)[0]
    top_idx = np.where(slab.positions[:, 2] > split)[0]
    assert not any(mask[i] for i in bottom_idx)
    assert all(mask[i] for i in top_idx)

    configs = build_config_grid(
        host_name="Cu(111)",
        slab=slab,
        adsorbate_name="CO",
        sites_filter=["top"],
        orientations_filter=["C-down"],
        rotations_deg=(0.0,),
        heights_A=(2.2,),
    )
    assert len(configs) == 1
    config = configs[0]
    assert len(config.active_mask) == len(config.atoms)
    assert all(config.active_mask[-2:])


@pytest.mark.parametrize(
    ("host_name", "slab_builder"),
    [
        ("Cu(111)", build_cu111_slab),
        ("Pd(111)", build_pd111_slab),
    ],
)
def test_co_fcc111_initial_site_grid_has_no_collisions(host_name, slab_builder):
    slab = slab_builder(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1))
    configs = build_config_grid(
        host_name=host_name,
        slab=slab,
        adsorbate_name="CO",
        sites_filter=["top", "bridge", "fcc", "hcp"],
        orientations_filter=["C-down"],
        rotations_deg=(0.0,),
        heights_A=(1.6, 1.8, 2.0, 2.2, 2.4),
    )
    assert len(configs) == 20
    clearances = [_initial_adsorbate_clearance(config, len(slab)) for config in configs]
    assert min(clearances) > 1.5
    assert max(clearances) < 3.1


def test_fcc_site_finder_returns_distinct_central_site_classes():
    slab = build_pd111_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1))
    sites = find_fcc_sites(slab)
    assert set(sites) == {"top", "bridge", "fcc", "hcp"}
    assert all(len(v) == 1 for v in sites.values())

    coords = {name: values[0] for name, values in sites.items()}
    for a, pa in coords.items():
        for b, pb in coords.items():
            if a >= b:
                continue
            assert np.linalg.norm(pa[:2] - pb[:2]) > 0.1

    bridge = coords["bridge"]
    assert np.linalg.norm(bridge[:2] - coords["top"][:2]) < LATTICE_A_PD / math.sqrt(2)


def test_fcc_hcp_hollow_labels_match_second_layer_stacking():
    slab = build_pd111_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1))
    sites = find_fcc_sites(slab)
    layers = _z_layers(slab)
    second_layer_z = np.mean(layers[-2])
    second_layer = slab.positions[np.isclose(slab.positions[:, 2], second_layer_z)]

    hcp_to_second = _nearest_periodic_xy_distance(
        sites["hcp"][0], second_layer, slab.cell
    )
    fcc_to_second = _nearest_periodic_xy_distance(
        sites["fcc"][0], second_layer, slab.cell
    )

    assert hcp_to_second == pytest.approx(0.0, abs=1e-6)
    assert fcc_to_second > 0.5


@pytest.mark.parametrize(
    ("name", "builder", "orientations", "lowest_symbols"),
    [
        ("CO", build_co, ["C-down", "O-down"], [{"C"}, {"O"}]),
        ("H2O", build_h2o, ["O-down", "H-down", "flat"], [{"O"}, {"H"}, None]),
        ("CH3OH", build_methanol, ["O-down", "methyl-down"], [{"O"}, {"H"}]),
        ("NH3", build_nh3, ["N-down", "H-down", "flat"], [{"N"}, {"H"}, None]),
    ],
)
def test_adsorbate_orientation_places_expected_atom_lowest(
    name, builder, orientations, lowest_symbols
):
    assert ADSORBATE_ORIENTATIONS[name] == orientations
    for orient, expected_symbols in zip(orientations, lowest_symbols):
        atoms = builder(orient)
        if orient == "flat":
            assert float(np.ptp(atoms.positions[:, 2])) < 1.7
            continue
        lowest = int(np.argmin(atoms.positions[:, 2]))
        assert atoms.get_chemical_symbols()[lowest] in expected_symbols

        if name == "CH3OH" and orient == "methyl-down":
            symbols = atoms.get_chemical_symbols()
            assert atoms.positions[symbols.index("O"), 2] > atoms.positions[symbols.index("C"), 2]
