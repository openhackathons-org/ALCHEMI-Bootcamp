"""Tests for the OER catalyst screening workflow (surfaces module)."""

import numpy as np
import pytest

from helpers.surfaces import (
    build_rutile_bulk,
    build_slab,
    make_active_mask,
    find_cus_sites,
    find_bridge_site,
    find_central_site,
    build_adsorbate,
    place_adsorbate,
    compute_adsorption_energy,
    compute_surface_displacement,
)
from helpers.constants import EV_PER_OER_STEP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUTILE_PARAMS = {
    "IrO2": ("Ir", 4.499, 3.154, 0.3065),
    "RuO2": ("Ru", 4.492, 3.107, 0.3058),
    "TiO2": ("Ti", 4.594, 2.958, 0.3048),
}

METAL_Z = {"Ir": 77, "Ru": 44, "Ti": 22}


@pytest.fixture
def iro2_bulk():
    metal, a, c, u = RUTILE_PARAMS["IrO2"]
    return build_rutile_bulk(metal, a, c, u)


@pytest.fixture
def iro2_slab():
    metal, a, c, u = RUTILE_PARAMS["IrO2"]
    bulk = build_rutile_bulk(metal, a, c, u)
    return build_slab(
        bulk, (1, 1, 0), min_slab_size=10.0, min_vacuum_size=15.0, supercell=(2, 2, 1)
    )


# ---------------------------------------------------------------------------
# TestRutileBulk
# ---------------------------------------------------------------------------


class TestRutileBulk:
    """Validate rutile bulk structure construction."""

    def test_iro2_atom_count(self, iro2_bulk):
        # P42/mnm conventional cell: 2 metal + 4 oxygen = 6 atoms
        assert len(iro2_bulk) == 6

    def test_iro2_composition(self, iro2_bulk):
        comp = iro2_bulk.composition.reduced_formula
        assert comp == "IrO2"

    def test_ruo2_lattice(self):
        metal, a, c, u = RUTILE_PARAMS["RuO2"]
        bulk = build_rutile_bulk(metal, a, c, u)
        latt = bulk.lattice
        assert abs(latt.a - a) < 0.01
        assert abs(latt.c - c) < 0.01

    def test_tio2_composition(self):
        metal, a, c, u = RUTILE_PARAMS["TiO2"]
        bulk = build_rutile_bulk(metal, a, c, u)
        assert bulk.composition.reduced_formula == "TiO2"

    @pytest.mark.parametrize("name", ["IrO2", "RuO2", "TiO2"])
    def test_all_materials_build(self, name):
        metal, a, c, u = RUTILE_PARAMS[name]
        bulk = build_rutile_bulk(metal, a, c, u)
        assert len(bulk) == 6


# ---------------------------------------------------------------------------
# TestSlabConstruction
# ---------------------------------------------------------------------------


class TestSlabConstruction:
    """Validate slab construction from bulk."""

    def test_slab_has_vacuum(self, iro2_slab):
        # c-axis should be substantially larger than a or b due to vacuum
        cell = iro2_slab.cell.array
        c_len = np.linalg.norm(cell[2])
        a_len = np.linalg.norm(cell[0])
        assert c_len > a_len + 10.0  # vacuum adds >= 15 A

    def test_slab_pbc(self, iro2_slab):
        assert list(iro2_slab.get_pbc()) == [True, True, True]

    def test_slab_has_atoms(self, iro2_slab):
        # 2x2x1 supercell of a (110) slab should have many atoms
        assert len(iro2_slab) >= 24  # at least 4 formula units in supercell

    def test_slab_contains_metal_and_oxygen(self, iro2_slab):
        numbers = set(iro2_slab.numbers)
        assert 77 in numbers  # Ir
        assert 8 in numbers  # O

    def test_slab_layers(self, iro2_slab):
        # Metal atoms should span multiple z-layers
        ir_z = iro2_slab.positions[iro2_slab.numbers == 77, 2]
        unique_layers = len(set(np.round(ir_z, decimals=1)))
        assert unique_layers >= 3


# ---------------------------------------------------------------------------
# TestActiveMask
# ---------------------------------------------------------------------------


class TestActiveMask:
    """Validate active mask generation."""

    def test_mask_length(self, iro2_slab):
        mask = make_active_mask(iro2_slab)
        assert len(mask) == len(iro2_slab)

    def test_half_frozen(self, iro2_slab):
        mask = make_active_mask(iro2_slab, bottom_fraction=0.5)
        n_frozen = sum(1 for m in mask if not m)
        n_active = sum(1 for m in mask if m)
        # Roughly half should be frozen (exact split depends on atom positions)
        assert n_frozen > 0
        assert n_active > 0

    def test_top_atoms_active(self, iro2_slab):
        mask = make_active_mask(iro2_slab)
        z = iro2_slab.positions[:, 2]
        top_idx = np.argmax(z)
        assert mask[top_idx] is True

    def test_bottom_atoms_frozen(self, iro2_slab):
        mask = make_active_mask(iro2_slab)
        z = iro2_slab.positions[:, 2]
        bottom_idx = np.argmin(z)
        assert mask[bottom_idx] is False

    def test_all_bool(self, iro2_slab):
        mask = make_active_mask(iro2_slab)
        assert all(isinstance(m, bool) for m in mask)


# ---------------------------------------------------------------------------
# TestAdsorbatePlacement
# ---------------------------------------------------------------------------


class TestAdsorbatePlacement:
    """Validate adsorbate construction and placement."""

    def test_o_single_atom(self):
        ads = build_adsorbate("O")
        assert len(ads) == 1
        assert ads.numbers[0] == 8

    def test_oh_geometry(self):
        ads = build_adsorbate("OH")
        assert len(ads) == 2
        d = np.linalg.norm(ads.positions[1] - ads.positions[0])
        assert abs(d - 0.970) < 0.01

    def test_h2o_geometry(self):
        ads = build_adsorbate("H2O")
        assert len(ads) == 3
        # O-H bond lengths
        for i in [1, 2]:
            d = np.linalg.norm(ads.positions[i] - ads.positions[0])
            assert abs(d - 0.957) < 0.01

    def test_h2o_angle(self):
        ads = build_adsorbate("H2O")
        v1 = ads.positions[1] - ads.positions[0]
        v2 = ads.positions[2] - ads.positions[0]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        assert abs(angle_deg - 104.5) < 0.5

    def test_ooh_geometry(self):
        ads = build_adsorbate("OOH")
        assert len(ads) == 3
        d_oo = np.linalg.norm(ads.positions[1] - ads.positions[0])
        assert abs(d_oo - 1.33) < 0.01
        d_oh = np.linalg.norm(ads.positions[2] - ads.positions[1])
        assert abs(d_oh - 0.970) < 0.02

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="Unknown adsorbate"):
            build_adsorbate("CH4")

    def test_placement_above_surface(self, iro2_slab):
        ads = build_adsorbate("OH")
        cus = find_cus_sites(iro2_slab, 77)
        combined, mask = place_adsorbate(iro2_slab, ads, cus[0], height=2.0)
        # Adsorbate atoms should be above all slab atoms
        slab_z_max = iro2_slab.positions[:, 2].max()
        ads_z_min = combined.positions[len(iro2_slab) :, 2].min()
        assert ads_z_min > slab_z_max

    def test_mask_includes_adsorbate(self, iro2_slab):
        ads = build_adsorbate("H2O")
        cus = find_cus_sites(iro2_slab, 77)
        combined, mask = place_adsorbate(iro2_slab, ads, cus[0])
        assert len(mask) == len(combined)
        # Last 3 atoms (H2O) should all be active
        assert all(mask[-3:])

    def test_tilted_vs_upright(self, iro2_slab):
        ads = build_adsorbate("OH")
        cus = find_cus_sites(iro2_slab, 77)
        upright, _ = place_adsorbate(iro2_slab, ads, cus[0], tilt_angle=0.0)
        tilted, _ = place_adsorbate(iro2_slab, ads, cus[0], tilt_angle=30.0)
        # H atom position should differ between tilted and upright
        h_upright = upright.positions[-1]
        h_tilted = tilted.positions[-1]
        assert np.linalg.norm(h_upright - h_tilted) > 0.1


# ---------------------------------------------------------------------------
# TestFindSites
# ---------------------------------------------------------------------------


class TestFindSites:
    """Validate surface site identification."""

    def test_cus_sites_exist(self, iro2_slab):
        cus = find_cus_sites(iro2_slab, 77)
        assert len(cus) >= 1

    def test_cus_site_on_surface(self, iro2_slab):
        cus = find_cus_sites(iro2_slab, 77)
        slab_z_max = iro2_slab.positions[:, 2].max()
        slab_z_min = iro2_slab.positions[:, 2].min()
        z_mid = (slab_z_max + slab_z_min) / 2
        # All cus sites should be in the top half
        for pos in cus:
            assert pos[2] > z_mid

    def test_bridge_between_metals(self, iro2_slab):
        bridge = find_bridge_site(iro2_slab, 77)
        cus = find_cus_sites(iro2_slab, 77)
        # Bridge should be between cus sites (not identical to any)
        for pos in cus:
            assert np.linalg.norm(bridge - pos) > 0.5


# ---------------------------------------------------------------------------
# TestCentralSite
# ---------------------------------------------------------------------------


class TestCentralSite:
    """Validate central site selection."""

    def test_picks_central_cus(self, iro2_slab):
        cus = find_cus_sites(iro2_slab, 77)
        cell = iro2_slab.cell.array
        central = find_central_site(cus, cell)
        center_xy = (cell[0, :2] + cell[1, :2]) / 2.0
        # Central site should be closer to xy-center than the average
        d_central = np.linalg.norm(central[:2] - center_xy)
        d_all = [np.linalg.norm(p[:2] - center_xy) for p in cus]
        assert d_central <= min(d_all) + 1e-10

    def test_single_site_returns_itself(self):
        pos = np.array([[5.0, 5.0, 10.0]])
        cell = np.eye(3) * 10.0
        result = find_central_site(pos, cell)
        assert np.allclose(result, pos[0])

    def test_returns_copy(self, iro2_slab):
        cus = find_cus_sites(iro2_slab, 77)
        cell = iro2_slab.cell.array
        central = find_central_site(cus, cell)
        central[0] = 999.0
        # Original should be unmodified
        cus2 = find_cus_sites(iro2_slab, 77)
        assert not np.any(cus2[:, 0] == 999.0)


# ---------------------------------------------------------------------------
# TestAdsorptionEnergy
# ---------------------------------------------------------------------------


class TestAdsorptionEnergy:
    """Validate adsorption energy calculation."""

    def test_simple_calculation(self):
        e_ads = compute_adsorption_energy(-100.0, -80.0, -15.0)
        assert abs(e_ads - (-5.0)) < 1e-10

    def test_negative_means_favorable(self):
        e_ads = compute_adsorption_energy(-100.0, -80.0, -15.0)
        assert e_ads < 0

    def test_positive_means_unfavorable(self):
        e_ads = compute_adsorption_energy(-90.0, -80.0, -15.0)
        assert e_ads > 0


# ---------------------------------------------------------------------------
# TestSurfaceDisplacement
# ---------------------------------------------------------------------------


class TestSurfaceDisplacement:
    """Validate surface displacement computation."""

    def test_zero_displacement(self, iro2_slab):
        disp = compute_surface_displacement(iro2_slab, iro2_slab, len(iro2_slab))
        assert np.allclose(disp, 0.0)

    def test_known_displacement(self):
        import ase

        atoms1 = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        atoms2 = ase.Atoms("H2", positions=[[1, 0, 0], [0, 0, 1]])
        disp = compute_surface_displacement(atoms1, atoms2, 2)
        assert abs(disp[0] - 1.0) < 1e-10
        assert abs(disp[1] - 0.0) < 1e-10

    def test_output_shape(self, iro2_slab):
        disp = compute_surface_displacement(iro2_slab, iro2_slab, len(iro2_slab))
        assert disp.shape == (len(iro2_slab),)


# ---------------------------------------------------------------------------
# TestOERConstant
# ---------------------------------------------------------------------------


class TestOERConstant:
    """Validate OER-related constants."""

    def test_ev_per_oer_step(self):
        assert abs(EV_PER_OER_STEP - 1.23) < 0.01

    def test_four_steps_give_total(self):
        # 4 equal steps of 1.23 eV = 4.92 eV total (standard potential)
        assert abs(4 * EV_PER_OER_STEP - 4.92) < 0.01


# ---------------------------------------------------------------------------
# TestActiveMaskWithAseToAtomicData
# ---------------------------------------------------------------------------


class TestActiveMaskPropagation:
    """Validate active_mask passes through ase_to_atomic_data."""

    def test_active_mask_set(self, iro2_slab):
        from helpers.models import ase_to_atomic_data

        mask = make_active_mask(iro2_slab)
        ad = ase_to_atomic_data(iro2_slab, structure_id="test", active_mask=mask)
        assert ad.active_mask == mask

    def test_active_mask_none_by_default(self, iro2_slab):
        from helpers.models import ase_to_atomic_data

        ad = ase_to_atomic_data(iro2_slab, structure_id="test")
        assert ad.active_mask is None
