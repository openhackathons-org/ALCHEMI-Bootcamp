"""Focused checks for adsorption structures and result handling."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.adsorption import (  # noqa: E402
    ADSLAB_KEYS,
    ADSORBATES,
    CLEAN_SLAB_KEY,
    FINITE_KEYS,
    GAS_KEYS,
    GEOMETRY_STATUS,
    PERIODIC_KEYS,
    STRUCTURE_KEYS,
    adsorption_energy,
    assemble_adsorption_results,
    build_full_force_table,
    build_initial_structure_set,
    build_placement_table,
    build_structure_inventory_table,
    export_full_forces,
    load_adsorption_methodology,
    load_initial_structure_set,
    split_for_batches,
    summarize_adslab_force_regions,
    validate_structure_set,
    write_initial_structure_set,
)
from aux.adsorption_visualization import _ovito_compatible_copy  # noqa: E402


def test_initial_structure_lineup_has_expected_counts_species_and_pbc() -> None:
    structures = build_initial_structure_set()

    assert tuple(structures) == STRUCTURE_KEYS
    assert len(structures[CLEAN_SLAB_KEY]) == 36
    assert structures[CLEAN_SLAB_KEY].get_chemical_formula() == "Cu36"
    assert tuple(structures[CLEAN_SLAB_KEY].pbc) == (True, True, False)

    expected_gas_atoms = {"CO": 2, "CO2": 3, "NH3": 4, "CH3OH": 6}
    expected_formulas = {"CO": "CO", "CO2": "CO2", "NH3": "H3N", "CH3OH": "CH4O"}
    for name in ADSORBATES:
        gas = structures[GAS_KEYS[name]]
        adslab = structures[ADSLAB_KEYS[name]]
        assert len(gas) == expected_gas_atoms[name]
        assert gas.get_chemical_formula() == expected_formulas[name]
        assert not gas.pbc.any()
        assert len(adslab) == 36 + expected_gas_atoms[name]
        assert tuple(adslab.pbc) == (True, True, False)
        assert (adslab.numbers == 29).sum() == 36
        assert adslab.arrays["is_adsorbate"].sum() == expected_gas_atoms[name]
        assert adslab.info["geometry_status"] == GEOMETRY_STATUS
        assert adslab.info["relaxed"] is False

    assert set(ADSORBATES) == {"CO", "CO2", "NH3", "CH3OH"}
    assert "H2" not in ADSORBATES
    assert "H2O" not in ADSORBATES


def test_ovito_display_copy_converts_boolean_particle_properties() -> None:
    atoms = build_initial_structure_set()[ADSLAB_KEYS["CH3OH"]]
    source_mask = atoms.arrays["is_adsorbate"].copy()

    display_atoms = _ovito_compatible_copy(atoms)

    assert display_atoms is not atoms
    assert atoms.arrays["is_adsorbate"].dtype == np.dtype(bool)
    assert display_atoms.arrays["is_adsorbate"].dtype == np.dtype(np.int8)
    np.testing.assert_array_equal(display_atoms.arrays["is_adsorbate"], source_mask)
    assert display_atoms.arrays["tags"].dtype == atoms.arrays["tags"].dtype
    np.testing.assert_array_equal(display_atoms.arrays["tags"], atoms.arrays["tags"])


def test_structure_set_splits_into_one_periodic_and_one_finite_batch() -> None:
    periodic, finite = split_for_batches(build_initial_structure_set())

    assert tuple(periodic) == PERIODIC_KEYS
    assert tuple(finite) == FINITE_KEYS
    assert len(periodic) == 5
    assert len(finite) == 4
    assert all(tuple(atoms.pbc) == (True, True, False) for atoms in periodic.values())
    assert all(not atoms.pbc.any() for atoms in finite.values())


def test_structure_and_placement_tables_keep_the_versioned_order() -> None:
    structures = load_initial_structure_set()
    inventory = build_structure_inventory_table(structures)
    placements = build_placement_table(load_adsorption_methodology())

    assert inventory["structure"].tolist() == list(STRUCTURE_KEYS)
    assert inventory["atoms"].sum() == sum(map(len, structures.values()))
    assert placements["molecule"].tolist() == list(ADSORBATES)
    assert placements.columns.tolist() == [
        "molecule",
        "site",
        "anchor",
        "height_A",
        "starting_orientation",
    ]


def test_structure_validation_rejects_wrong_boundary_and_relaxation_labels() -> None:
    structures = build_initial_structure_set()
    structures[GAS_KEYS["CO"]].set_pbc(True)
    with pytest.raises(ValueError, match="must be finite"):
        validate_structure_set(structures)

    structures = build_initial_structure_set()
    structures[ADSLAB_KEYS["CO"]].info["relaxed"] = True
    with pytest.raises(ValueError, match="must not be labeled as relaxed"):
        validate_structure_set(structures)


def test_initial_structure_artifacts_round_trip_with_hashes(tmp_path: Path) -> None:
    manifest = write_initial_structure_set(tmp_path)
    loaded = load_initial_structure_set(tmp_path)

    assert manifest["geometry_status"] == GEOMETRY_STATUS
    assert manifest["method_id"] == "cu111-important-molecules-v1"
    assert len(manifest["structures"]) == 9
    assert tuple(loaded) == STRUCTURE_KEYS
    assert sorted(path.name for path in tmp_path.glob("*.extxyz")) == sorted(
        f"{key}_initial.extxyz" for key in STRUCTURE_KEYS
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_initial_structure_set(tmp_path)


def test_checked_in_initial_structures_match_their_manifest() -> None:
    structures = load_initial_structure_set()
    regenerated = build_initial_structure_set()

    assert tuple(structures) == STRUCTURE_KEYS
    assert tuple(structures[CLEAN_SLAB_KEY].pbc) == (True, True, False)
    for key in STRUCTURE_KEYS:
        assert np.array_equal(structures[key].numbers, regenerated[key].numbers)
        assert np.array_equal(structures[key].pbc, regenerated[key].pbc)
        assert np.array_equal(
            structures[key].arrays["is_adsorbate"],
            regenerated[key].arrays["is_adsorbate"],
        )
        assert structures[key].cell.array == pytest.approx(
            regenerated[key].cell.array, abs=1.0e-12
        )
        assert structures[key].positions == pytest.approx(
            regenerated[key].positions, abs=1.0e-8
        )


def test_adsorption_energy_uses_balanced_fixed_geometry_difference() -> None:
    assert adsorption_energy(-16.5, -10.0, -5.0) == pytest.approx(-1.5)
    assert adsorption_energy(-14.0, -10.0, -5.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be finite"):
        adsorption_energy(np.nan, -10.0, -5.0)


def _synthetic_energies() -> tuple[dict[str, float], dict[str, float]]:
    model = {
        CLEAN_SLAB_KEY: -100.0,
        ADSLAB_KEYS["CO"]: -111.0,
        ADSLAB_KEYS["CO2"]: -120.5,
        ADSLAB_KEYS["NH3"]: -130.2,
        ADSLAB_KEYS["CH3OH"]: -139.8,
        GAS_KEYS["CO"]: -10.0,
        GAS_KEYS["CO2"]: -20.0,
        GAS_KEYS["NH3"]: -30.0,
        GAS_KEYS["CH3OH"]: -40.0,
    }
    d3 = {
        CLEAN_SLAB_KEY: -1.0,
        ADSLAB_KEYS["CO"]: -1.3,
        ADSLAB_KEYS["CO2"]: -1.4,
        ADSLAB_KEYS["NH3"]: -1.25,
        ADSLAB_KEYS["CH3OH"]: -1.6,
        GAS_KEYS["CO"]: -0.1,
        GAS_KEYS["CO2"]: -0.2,
        GAS_KEYS["NH3"]: -0.05,
        GAS_KEYS["CH3OH"]: -0.3,
    }
    return model, d3


def _synthetic_adslab_forces() -> dict[str, np.ndarray]:
    atom_counts = {"CO": 38, "CO2": 39, "NH3": 40, "CH3OH": 42}
    forces: dict[str, np.ndarray] = {}
    for name in ADSORBATES:
        array = np.zeros((atom_counts[name], 3), dtype=float)
        array[0] = (3.0, 4.0, 0.0)
        forces[ADSLAB_KEYS[name]] = array
    return forces


def test_result_assembly_returns_every_component_and_force_statistic() -> None:
    model, d3 = _synthetic_energies()
    table = assemble_adsorption_results(
        model_energies_eV=model,
        d3_energies_eV=d3,
        combined_forces_eV_A=_synthetic_adslab_forces(),
    )

    assert table["molecule"].tolist() == list(ADSORBATES)
    assert table["model_adsorption_energy_eV"].tolist() == pytest.approx(
        [-1.0, -0.5, -0.2, 0.2]
    )
    assert table["d3_adsorption_energy_eV"].tolist() == pytest.approx(
        [-0.2, -0.2, -0.2, -0.3]
    )
    assert table["adsorption_energy_eV"].tolist() == pytest.approx(
        [-1.2, -0.7, -0.4, -0.1]
    )
    assert table["fmax_eV_A"].tolist() == pytest.approx([5.0] * 4)
    assert table["force_rms_eV_A"].tolist() == pytest.approx(
        [5.0 / np.sqrt(count) for count in (38, 39, 40, 42)]
    )
    assert table["force_atoms"].tolist() == [38, 39, 40, 42]


def test_result_assembly_rejects_incomplete_energy_components() -> None:
    model, d3 = _synthetic_energies()
    del d3[GAS_KEYS["CO2"]]

    with pytest.raises(ValueError, match="D3 energies: missing co2_gas"):
        assemble_adsorption_results(
            model_energies_eV=model,
            d3_energies_eV=d3,
            combined_forces_eV_A=_synthetic_adslab_forces(),
        )


def test_full_force_export_keeps_every_atom_and_vector(tmp_path: Path) -> None:
    structures = build_initial_structure_set()
    forces = {
        key: np.full((len(atoms), 3), index / 100.0)
        for index, (key, atoms) in enumerate(structures.items())
    }
    table = build_full_force_table(structures, forces)
    output_path = tmp_path / "adsorption_forces.csv"
    record = export_full_forces(
        output_path,
        structures=structures,
        forces_eV_A=forces,
    )
    loaded = pd.read_csv(output_path)

    assert len(table) == sum(len(atoms) for atoms in structures.values())
    assert len(loaded) == len(table) == record["rows"]
    assert record["structures"] == 9
    assert len(record["sha256"]) == 64
    assert loaded.groupby("structure", sort=False).size().to_dict() == {
        key: len(atoms) for key, atoms in structures.items()
    }
    assert set(("fx_eV_A", "fy_eV_A", "fz_eV_A", "force_norm_eV_A")).issubset(
        loaded.columns
    )


def test_adslab_force_summary_splits_adsorbate_and_cu_atoms() -> None:
    structures = build_initial_structure_set()
    forces = {
        key: np.zeros((len(atoms), 3), dtype=float)
        for key, atoms in structures.items()
    }
    for name in ADSORBATES:
        key = ADSLAB_KEYS[name]
        forces[key][0] = (3.0, 4.0, 0.0)
        adsorbate_index = int(np.flatnonzero(structures[key].arrays["is_adsorbate"])[0])
        forces[key][adsorbate_index] = (0.0, 0.0, 2.0)

    summary = summarize_adslab_force_regions(
        build_full_force_table(structures, forces)
    )

    assert len(summary) == 8
    assert summary["region"].tolist() == ["Cu slab", "adsorbate"] * 4
    assert summary.loc[summary["region"] == "Cu slab", "atoms"].tolist() == [36] * 4
    assert summary.loc[summary["region"] == "Cu slab", "fmax_eV_A"].tolist() == [
        5.0
    ] * 4
    assert summary.loc[
        summary["region"] == "adsorbate", "fmax_eV_A"
    ].tolist() == [2.0] * 4
