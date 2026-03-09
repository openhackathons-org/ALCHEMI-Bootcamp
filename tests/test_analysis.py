"""Test thermodynamic analysis functions using cached MD trajectories."""

import numpy as np
import pytest

from helpers.analysis import (
    compute_density,
    compute_msd,
    compute_rdf,
    estimate_diffusion_coefficient,
    extract_thermo_timeseries,
    pick_production_window,
    thermal_expansion_proxy,
    trajectory_to_ase_list,
)
from helpers.cache import cache_exists, load_cache
from helpers.models import MDAtomicData, MDReply


def _load_npt_data(cache_dir, nacl_md_atoms):
    """Load NVT + NPT cached data and reconstruct NPT seed atoms."""
    nvt = load_cache(cache_dir, "nacl_nvt_equil", MDReply)
    npt = load_cache(cache_dir, "nacl_npt_prod", MDReply)
    last = nvt.trajectory[-1]
    npt_atoms = MDAtomicData(
        coord=last.coord,
        numbers=nacl_md_atoms.numbers,
        cell=last.cell if last.cell else nacl_md_atoms.cell,
        pbc=nacl_md_atoms.pbc,
        velocity=last.velocity,
    )
    return npt_atoms, npt


def _require_npt_cache(cache_dir):
    if not cache_exists(cache_dir, "nacl_nvt_equil"):
        pytest.skip("Cached NVT response not available")
    if not cache_exists(cache_dir, "nacl_npt_prod"):
        pytest.skip("Cached NPT response not available")


class TestThermoExtraction:
    def test_thermo_keys(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)

        expected_keys = {
            "time_ps",
            "temperature_K",
            "e_pot_eV",
            "e_kin_eV",
            "e_tot_eV",
            "pressure_kbar",
            "volume_A3",
            "p_xx",
            "p_yy",
            "p_zz",
        }
        assert set(thermo.keys()) == expected_keys

    def test_thermo_lengths(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)

        n_frames = len(npt.trajectory)
        for key, arr in thermo.items():
            assert len(arr) == n_frames, f"{key} has wrong length"

    def test_temperature_physical_range(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)

        T = thermo["temperature_K"]
        assert T.min() > 0, "Temperature should be positive"
        assert T.max() < 2000, "Temperature unreasonably high"
        # Mean should be near target (300 K) within statistical fluctuation
        assert 100 < T.mean() < 600

    def test_volume_positive(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)

        assert np.all(thermo["volume_A3"] > 0)

    def test_energy_conservation(self, cache_dir, nacl_md_atoms):
        """E_tot = E_pot + E_kin."""
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)

        e_tot_calc = thermo["e_pot_eV"] + thermo["e_kin_eV"]
        np.testing.assert_allclose(thermo["e_tot_eV"], e_tot_calc, rtol=1e-10)


class TestProductionWindow:
    def test_discard_fraction(self):
        thermo = {"time_ps": np.arange(100, dtype=float)}
        s0, s1 = pick_production_window(thermo, discard_fraction=0.3)
        assert s0 == 30
        assert s1 == 100

    def test_zero_discard(self):
        thermo = {"time_ps": np.arange(50, dtype=float)}
        s0, s1 = pick_production_window(thermo, discard_fraction=0.0)
        assert s0 == 0
        assert s1 == 50


class TestDensity:
    def test_nacl_density_physical(self, cache_dir, nacl_md_atoms):
        """NaCl density should be near 2.165 g/cm3 (within 10% for FAST_DEMO)."""
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)
        s0, s1 = pick_production_window(thermo)
        density = compute_density(npt_atoms, thermo["volume_A3"][s0:s1])

        assert 1.5 < density < 3.0, f"Density {density} out of physical range"
        assert abs(density - 2.165) / 2.165 < 0.10, (
            f"Density {density:.3f} deviates >10% from experimental 2.165"
        )


class TestRDF:
    def test_nacl_rdf_first_peak(self, cache_dir, nacl_md_atoms):
        """Na-Cl RDF first peak should be near 2.82 A."""
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        thermo = extract_thermo_timeseries(npt_atoms, npt.trajectory)
        s0, _ = pick_production_window(thermo)

        frames = trajectory_to_ase_list(npt_atoms, npt.trajectory)
        r, g = compute_rdf(frames, species_pair=(11, 17), r_max=8.0, start_frame=s0)

        peak_r = r[np.argmax(g)]
        assert abs(peak_r - 2.82) < 0.3, (
            f"Na-Cl RDF peak at {peak_r:.2f} A, expected ~2.82 A"
        )

    def test_rdf_normalization(self, cache_dir, nacl_md_atoms):
        """g(r) should approach 1 at large r."""
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        frames = trajectory_to_ase_list(npt_atoms, npt.trajectory)

        r, g = compute_rdf(frames, species_pair=(11, 17), r_max=8.0, start_frame=0)
        # At large r, g(r) should tend toward 1
        tail_mean = g[len(g) * 3 // 4 :].mean()
        assert 0.5 < tail_mean < 2.0, f"RDF tail mean {tail_mean} not near 1"


class TestMSD:
    def test_msd_starts_at_zero(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        frames = trajectory_to_ase_list(npt_atoms, npt.trajectory)

        t, msd = compute_msd(frames, species=11, start_frame=0)
        assert msd[0] == pytest.approx(0.0, abs=1e-10)

    def test_msd_non_negative(self, cache_dir, nacl_md_atoms):
        _require_npt_cache(cache_dir)
        npt_atoms, npt = _load_npt_data(cache_dir, nacl_md_atoms)
        frames = trajectory_to_ase_list(npt_atoms, npt.trajectory)

        t, msd = compute_msd(frames, species=11, start_frame=0)
        assert np.all(msd >= 0)

    def test_diffusion_coefficient_non_negative(self):
        t = np.linspace(0, 10, 100)
        msd = 0.5 * t + np.random.normal(0, 0.01, len(t))
        D = estimate_diffusion_coefficient(t, msd)
        assert D >= 0


class TestThermalExpansion:
    def test_thermal_expansion_from_cached_sweep(self, cache_dir, nacl_md_atoms):
        """Estimate alpha_V from temperature sweep."""
        temps = []
        densities = []
        for T in [200, 300, 400]:
            nvt_label = f"nacl_nvt_T{T}"
            npt_label = f"nacl_npt_T{T}"
            if not (cache_exists(cache_dir, nvt_label) and cache_exists(cache_dir, npt_label)):
                pytest.skip(f"Temperature sweep cache missing for T={T}")

            nvt_r = load_cache(cache_dir, nvt_label, MDReply)
            npt_r = load_cache(cache_dir, npt_label, MDReply)
            s = nvt_r.trajectory[-1]
            seed = MDAtomicData(
                coord=s.coord,
                numbers=nacl_md_atoms.numbers,
                cell=s.cell if s.cell else nacl_md_atoms.cell,
                pbc=nacl_md_atoms.pbc,
                velocity=s.velocity,
            )
            th = extract_thermo_timeseries(seed, npt_r.trajectory)
            s0, s1 = pick_production_window(th)
            rho = compute_density(seed, th["volume_A3"][s0:s1])
            temps.append(T)
            densities.append(rho)

        temps = np.array(temps, dtype=float)
        densities = np.array(densities)
        alpha_v = thermal_expansion_proxy(temps, densities)

        # Thermal expansion should be a small positive number (order 1e-4 K^-1)
        assert isinstance(alpha_v, float)
        # Allow wide range for FAST_DEMO short simulations
        assert abs(alpha_v) < 1e-2, f"alpha_V = {alpha_v} seems unreasonably large"

    def test_thermal_expansion_synthetic(self):
        """Synthetic data: known linear density decrease → known alpha_V."""
        temps = np.array([200.0, 300.0, 400.0, 500.0])
        # rho decreasing linearly: drho/dT = -0.001
        densities = 2.2 - 0.001 * (temps - 200)
        alpha_v = thermal_expansion_proxy(temps, densities)
        # alpha_V = -drho_dT / rho_mean = 0.001 / mean(rho)
        expected = 0.001 / densities.mean()
        assert alpha_v == pytest.approx(expected, rel=0.01)
