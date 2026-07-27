"""Focused tests for the packaged NCI Atlas subset and energy reduction."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.nci_atlas import (  # noqa: E402
    CURVE_KEY_COLUMNS,
    EXPECTED_SCALES,
    EXPECTED_SYSTEMS,
    NCI_ATLAS_SUBSET_SHA256,
    assemble_nci_comparison_curves,
    build_graph_index,
    extract_repeated_interaction_reference,
    interaction_metrics,
    load_nci_atlas_subset,
    mean_member_curves,
    reduce_fragment_energies,
    row_to_atoms,
    rows_to_atoms,
    validate_nci_atlas_subset,
)


DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"


@pytest.fixture(scope="module")
def subset() -> pd.DataFrame:
    return load_nci_atlas_subset(DATA_FILE)


def test_packaged_subset_has_expected_identity_and_shape(subset: pd.DataFrame) -> None:
    assert NCI_ATLAS_SUBSET_SHA256 == (
        "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
    )
    assert subset.shape == (90, 14)
    assert tuple(subset["system_id"].drop_duplicates()) == tuple(EXPECTED_SYSTEMS)
    assert tuple(
        sorted(subset[subset["system_id"] == "08.007"]["scale"].unique())
    ) == EXPECTED_SCALES
    assert "08.007" in set(subset["system_id"])


def test_loader_rejects_checksum_mismatch(tmp_path: Path) -> None:
    changed = tmp_path / DATA_FILE.name
    shutil.copyfile(DATA_FILE, changed)
    changed.write_bytes(changed.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_nci_atlas_subset(changed)


def test_validation_rejects_incomplete_curve(subset: pd.DataFrame) -> None:
    incomplete = subset.drop(index=subset.index[0]).reset_index(drop=True)

    with pytest.raises(ValueError, match="89 rows; expected 90"):
        validate_nci_atlas_subset(incomplete)


def test_validation_rejects_fragment_geometry_drift(subset: pd.DataFrame) -> None:
    changed = subset.copy()
    row_index = changed.index[
        (changed["system_id"] == "1.041")
        & np.isclose(changed["scale"], 1.0)
        & (changed["fragment"] == "A")
    ].item()
    coordinates = changed.at[row_index, "positions_angstrom"].split()
    coordinates[0] = f"{float(coordinates[0]) + 0.01:.10f}"
    changed.at[row_index, "positions_angstrom"] = " ".join(coordinates)

    with pytest.raises(ValueError, match="coordinates do not reconstruct"):
        validate_nci_atlas_subset(changed)


def test_rows_convert_to_nonperiodic_ase_without_atomic_charges(
    subset: pd.DataFrame,
) -> None:
    atoms = row_to_atoms(subset.iloc[0])

    assert len(atoms) == int(subset.iloc[0]["natoms"])
    assert not atoms.pbc.any()
    assert atoms.info["charge"] == int(subset.iloc[0]["charge"])
    assert atoms.info["fragment"] == "AB"
    assert "charges" not in atoms.arrays

    all_atoms = rows_to_atoms(subset)
    assert len(all_atoms) == 90
    assert sum(len(atoms) for atoms in all_atoms) == int(subset["natoms"].sum())


def test_graph_index_is_row_aligned_and_contains_source_identifiers(
    subset: pd.DataFrame,
) -> None:
    graph_index = build_graph_index(subset)

    assert graph_index["graph_index"].tolist() == list(range(90))
    assert graph_index["system_id"].tolist() == subset["system_id"].tolist()
    assert graph_index["fragment"].tolist() == subset["fragment"].tolist()
    assert graph_index["source_gradient_block"].is_unique


def _graph_energies(graph_index: pd.DataFrame, multiplier: float) -> np.ndarray:
    values = []
    for row in graph_index.itertuples(index=False):
        interaction = multiplier * float(row.scale)
        fragment_energy = {"A": 10.0, "B": 20.0, "AB": 30.0 + interaction}
        values.append(fragment_energy[row.fragment])
    return np.asarray(values)


def test_reduce_graph_energies_and_summarize_members(subset: pd.DataFrame) -> None:
    graph_index = build_graph_index(subset)
    member_graph_energies = np.stack(
        (_graph_energies(graph_index, 1.0), _graph_energies(graph_index, 2.0))
    )

    member_curves = reduce_fragment_energies(
        graph_index,
        {"model": member_graph_energies},
        unit_scale=2.0,
    )
    assert len(member_curves) == 60
    np.testing.assert_allclose(
        member_curves[member_curves["member"] == 0]["model"],
        member_curves[member_curves["member"] == 0]["scale"] * 2.0,
    )
    np.testing.assert_allclose(
        member_curves[member_curves["member"] == 1]["model"],
        member_curves[member_curves["member"] == 1]["scale"] * 4.0,
    )

    summary = mean_member_curves(
        member_curves,
        ["model"],
        spread_component="model",
    )
    assert len(summary) == 30
    np.testing.assert_allclose(summary["model"], summary["scale"] * 3.0)
    np.testing.assert_allclose(
        summary["model_std"], summary["scale"] * np.sqrt(2.0)
    )


def test_reduction_uses_graph_index_not_dataframe_row_order(
    subset: pd.DataFrame,
) -> None:
    graph_index = build_graph_index(subset)
    energies = _graph_energies(graph_index, 1.0)
    shuffled_index = graph_index.sample(frac=1.0, random_state=7)

    curves = reduce_fragment_energies(shuffled_index, {"model": energies})

    np.testing.assert_allclose(curves["model"], curves["scale"])


def test_reference_extraction_and_metrics(subset: pd.DataFrame) -> None:
    reference = extract_repeated_interaction_reference(
        subset,
        "ccsd_t_cbs_interaction_energy_kcal_mol",
        output_column="ccsd_t_cbs",
    )
    assert len(reference) == 30
    expected = subset[subset["fragment"] == "AB"][
        [*CURVE_KEY_COLUMNS, "ccsd_t_cbs_interaction_energy_kcal_mol"]
    ].rename(
        columns={"ccsd_t_cbs_interaction_energy_kcal_mol": "ccsd_t_cbs"}
    )
    pd.testing.assert_frame_equal(
        reference.reset_index(drop=True), expected.reset_index(drop=True)
    )

    curves = reference.copy()
    curves["prediction"] = curves["ccsd_t_cbs"] + 0.25
    curves["spread"] = 0.10
    metrics = interaction_metrics(
        curves,
        {"prediction vs CC": ("prediction", "ccsd_t_cbs")},
        mean_columns={"ensemble spread": "spread"},
    )
    np.testing.assert_allclose(metrics["prediction vs CC"], 0.25)
    np.testing.assert_allclose(metrics["ensemble spread"], 0.10)


def test_complete_curve_assembly_returns_member_curves_references_and_metrics(
    subset: pd.DataFrame,
) -> None:
    graph_index = build_graph_index(subset)
    member_energies = np.stack(
        (_graph_energies(graph_index, 1.0), _graph_energies(graph_index, 2.0))
    )
    member_curves, curves, metrics = assemble_nci_comparison_curves(
        graph_index,
        subset,
        {"core": member_energies, "full": member_energies},
        d3_graph_energies_eV=np.zeros(len(graph_index)),
        dft_total_energy_column="wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
        cc_interaction_energy_column="ccsd_t_cbs_interaction_energy_kcal_mol",
        comparisons={"complete vs CC": ("full", "ccsd_t_cbs")},
        energy_to_kcal_mol=1.0,
    )

    assert len(member_curves) == 60
    assert len(curves) == 30
    assert {"dft_full", "dft_no_d3", "ccsd_t_cbs", "full_std"}.issubset(
        curves.columns
    )
    assert list(metrics.columns) == ["complete vs CC", "ensemble spread"]


@pytest.mark.parametrize(
    "bad_values",
    [np.zeros(89), np.zeros((2, 89)), np.full(90, np.nan)],
)
def test_reduction_rejects_misaligned_or_nonfinite_values(
    subset: pd.DataFrame, bad_values: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape|non-finite"):
        reduce_fragment_energies(build_graph_index(subset), {"model": bad_values})
