"""Tests for AdsorbML validation analysis helpers."""

import pytest
from ase.build import fcc111

from helpers.analysis import (
    build_pair_results_table,
    classify_final_site,
    compute_adsorption_energy_ev,
    reference_site_matches,
    summarize_pair_validation,
)
from helpers.config_search import Configuration, build_co, find_fcc_sites
from helpers.references import (
    ACTIVE_ADSORBML_PAIRS,
    ADSORBML_REFERENCES,
    OPTIONAL_ADSORBML_CONTEXT_PAIRS,
    REFERENCE_PUBLICATIONS,
    active_adsorbml_references,
    optional_adsorbml_context_references,
    strict_adsorbml_references,
)


def test_adsorption_energy_convention_negative_exothermic():
    e_ads = compute_adsorption_energy_ev(
        e_slab_ads_ev=-111.0,
        e_clean_slab_ev=-100.0,
        e_gas_ads_ev=-10.5,
    )
    assert e_ads == pytest.approx(-0.5)


def test_reference_rows_are_context_until_exact_dataset_lookup():
    assert len(active_adsorbml_references()) == 9
    assert len(optional_adsorbml_context_references()) == 3
    assert len(ADSORBML_REFERENCES) == 12
    assert set(ACTIVE_ADSORBML_PAIRS).isdisjoint(OPTIONAL_ADSORBML_CONTEXT_PAIRS)
    assert strict_adsorbml_references() == {}
    assert all(r.reference_scope == "context" for r in ADSORBML_REFERENCES.values())


def test_canonical_publication_metadata_present():
    assert REFERENCE_PUBLICATIONS["adsorbml"].doi == "10.1038/s41524-023-01121-5"
    assert REFERENCE_PUBLICATIONS["oc20"].doi == "10.1021/acscatal.0c04525"
    assert REFERENCE_PUBLICATIONS["oc22"].doi == "10.1021/acscatal.2c05426"


def test_reference_site_text_matching():
    assert reference_site_matches("fcc", "fcc-hollow")
    assert reference_site_matches("top", "top (O-down, tilted)")
    assert reference_site_matches("al-top", "Al-top (O-down)")
    assert not reference_site_matches("bridge", "fcc-hollow")
    assert reference_site_matches("top", None) is None


def test_classify_final_site_uses_final_geometry_not_start_label():
    slab = fcc111("Cu", size=(3, 3, 4), vacuum=12.0, periodic=True)
    sites = find_fcc_sites(slab)
    top = sites["top"][0]
    co = build_co("C-down")
    co.translate(top + [0.0, 0.0, 2.0])
    final_atoms = slab + co

    config = Configuration(
        label="CO_Cu111_bridge_C-down_rot0_h2.2",
        host="Cu(111)",
        adsorbate="CO",
        site="bridge",
        orientation="C-down",
        rot_deg=0.0,
        height=2.2,
        atoms=final_atoms,
        active_mask=[True] * len(final_atoms),
    )

    site = classify_final_site(
        host="Cu(111)",
        adsorbate="CO",
        final_atoms=final_atoms,
        config=config,
        slab_atom_count=len(slab),
    )

    assert site.start_site == "bridge"
    assert site.final_site == "top"
    assert site.binding_atom_symbol == "C"


@pytest.mark.parametrize("site_name", ["top", "bridge", "fcc", "hcp"])
def test_classify_final_site_resolves_fcc111_periodic_sites(site_name):
    slab = fcc111("Cu", size=(3, 3, 4), vacuum=12.0, periodic=True)
    sites = find_fcc_sites(slab)
    co = build_co("C-down")
    co.translate(sites[site_name][0] + [0.0, 0.0, 2.0])
    atoms = slab + co
    config = Configuration(
        label=f"CO_Cu111_{site_name}_C-down_rot0_h2.0",
        host="Cu(111)",
        adsorbate="CO",
        site="top",
        orientation="C-down",
        rot_deg=0.0,
        height=2.0,
        atoms=atoms,
        active_mask=[True] * len(atoms),
    )

    site = classify_final_site(
        host="Cu(111)",
        adsorbate="CO",
        final_atoms=atoms,
        config=config,
        slab_atom_count=len(slab),
    )

    assert site.final_site == site_name
    assert site.final_site_distance_A < 0.1


def test_context_references_do_not_pass_strict_validation():
    import pandas as pd

    pair_results = {
        ("Cu(111)", "CO"): pd.DataFrame([
            {
                "E_bind (eV)": -0.75,
                "final_site": "top",
                "binding_atom_symbol": "C",
            }
        ])
    }

    summary = summarize_pair_validation(pair_results, ADSORBML_REFERENCES)
    assert summary.loc[0, "status"] == "context-only"
    assert summary.loc[0, "reference_scope"] == "context"


def test_validation_uses_lowest_reliable_configuration_not_failed_minimum():
    import pandas as pd

    pair_results = {
        ("Cu(111)", "CO"): pd.DataFrame([
            {
                "E_bind (eV)": -9.99,
                "final_site": "bridge",
                "binding_atom_symbol": "C",
                "reliable_for_minimum": False,
            },
            {
                "E_bind (eV)": -0.75,
                "final_site": "top",
                "binding_atom_symbol": "C",
                "reliable_for_minimum": True,
            },
        ])
    }

    summary = summarize_pair_validation(pair_results, ADSORBML_REFERENCES)
    assert summary.loc[0, "E_MACE_eV"] == pytest.approx(-0.75)
    assert summary.loc[0, "MACE_site"] == "top (C-down)"
    assert summary.loc[0, "status"] == "context-only"


def test_validation_flags_pair_with_no_reliable_configuration():
    import pandas as pd

    pair_results = {
        ("Cu(111)", "CO"): pd.DataFrame([
            {
                "E_bind (eV)": -9.99,
                "final_site": "bridge",
                "binding_atom_symbol": "C",
                "reliable_for_minimum": False,
            }
        ])
    }

    summary = summarize_pair_validation(pair_results, ADSORBML_REFERENCES)
    assert summary.loc[0, "status"] == "no-reliable-result"
    assert pd.isna(summary.loc[0, "delta_E_eV"])


def test_pair_results_include_optimizer_steps():
    from helpers.models import OptimizationResult

    slab = fcc111("Cu", size=(3, 3, 4), vacuum=12.0, periodic=True)
    co = build_co("C-down")
    sites = find_fcc_sites(slab)
    co.translate(sites["top"][0] + [0.0, 0.0, 1.8])
    atoms = slab + co
    config = Configuration(
        label="CO_Cu111_top_C-down_rot0_h1.8",
        host="Cu(111)",
        adsorbate="CO",
        site="top",
        orientation="C-down",
        rot_deg=0.0,
        height=1.8,
        atoms=atoms,
        active_mask=[True] * len(atoms),
    )
    result = OptimizationResult(
        coord=atoms.positions.flatten().tolist(),
        numbers=atoms.numbers.tolist(),
        charge=0,
        mult=1,
        cell=atoms.cell.array.flatten().tolist(),
        pbc=atoms.pbc.tolist(),
        converged=True,
        optimizer_nsteps=37,
        energy=-115.0,
        forces=[0.0] * (len(atoms) * 3),
    )

    df = build_pair_results_table(
        host="Cu(111)",
        adsorbate="CO",
        configs=[config],
        opt_results=[result],
        clean_slab_atoms=slab,
        e_clean_slab_ev=-100.0,
        e_gas_ads_ev=-14.0,
        backend="toolkit",
    )

    assert df.loc[0, "optimizer_nsteps"] == 37
    assert df.loc[0, "host"] == "Cu(111)"
    assert df.loc[0, "adsorbate"] == "CO"
    assert df.loc[0, "reference_scope"] == "none"
    assert df.loc[0, "validation_status"] == "not-evaluated"


def test_validation_handles_context_reference_without_energy():
    import pandas as pd

    pair_results = {
        ("Pd(111)", "NH3"): pd.DataFrame([
            {
                "E_bind (eV)": -0.62,
                "final_site": "top",
                "binding_atom_symbol": "N",
                "reliable_for_minimum": True,
            }
        ])
    }

    summary = summarize_pair_validation(pair_results, ADSORBML_REFERENCES)
    assert summary.loc[0, "status"] == "no-reference"
    assert pd.isna(summary.loc[0, "E_ref_eV"])
    assert pd.isna(summary.loc[0, "delta_E_eV"])
