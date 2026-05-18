"""Guardrails for OC20Dense benchmark reporting."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest


PART1 = Path(__file__).resolve().parents[1]
OC20DENSE_ROOT = (
    PART1
    / "outputs"
    / "precomputed"
    / "accuracy"
    / "oc20dense_closed_shell_trajectory_mace_mpa0"
)


def _load_script(relative_path: str):
    os.environ.setdefault("OC20DENSE_LD_REEXEC", "1")
    path = PART1 / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_oc20dense_artifacts() -> Path:
    """Return local OC20Dense output root, or skip clean-clone runs."""
    import pytest

    required = [
        OC20DENSE_ROOT / "tables" / "per_config_results.csv",
        OC20DENSE_ROOT / "tables" / "system_summary.csv",
        OC20DENSE_ROOT / "dft_reference_checks" / "dft_reference_comparison.csv",
        OC20DENSE_ROOT / "dft_final_single_points" / "tables" / "dft_final_sp_system_summary.csv",
        OC20DENSE_ROOT / "mace_adsorption_energy" / "tables" / "mace_adsorption_energies.csv",
        OC20DENSE_ROOT / "reports" / "oc20dense_known_examples_report.md",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(
            "OC20Dense artifact checks require local generated outputs under "
            f"{OC20DENSE_ROOT}; rerun the reproducibility notebook or the "
            "oc20dense recompute script chain to enable these checks."
        )
    return OC20DENSE_ROOT


def test_oc20dense_scripts_mark_mace_eads_reference_status():
    common = _load_script("scripts/_oc20dense_common.py")
    known = _load_script("scripts/run_oc20dense_known_examples.py")
    checks = _load_script("scripts/oc20dense_dft_reference_checks.py")

    expected = "defined_mace_eads_official_surface_neutral_gas_refs"
    assert common.MACE_EADS_REFERENCE_STATUS == expected
    assert known.MACE_EADS_REFERENCE_STATUS == expected
    assert checks.MACE_EADS_REFERENCE_STATUS == expected
    assert common.MACE_RANK_BASIS == "total_energy_within_fixed_system"
    assert known.MACE_RANK_BASIS == "total_energy_within_fixed_system"
    assert checks.MACE_RANK_BASIS == "total_energy_within_fixed_system"


def test_oc20dense_defaults_use_same_closed_shell_system_set():
    common = _load_script("scripts/_oc20dense_common.py")
    known = _load_script("scripts/run_oc20dense_known_examples.py")
    checks = _load_script("scripts/oc20dense_dft_reference_checks.py")

    expected_systems = ("3_2070_48", "72_7104_115", "69_1615_2")
    expected_refs = {"*OH2": "H2O", "*NH3": "NH3", "*N2": "N2"}

    assert common.DEFAULT_SYSTEMS == expected_systems
    assert known.DEFAULT_SYSTEMS == expected_systems
    assert checks.DEFAULT_SYSTEMS == expected_systems
    assert common.CLOSED_SHELL_ADSORBATE_REFERENCES == expected_refs
    assert known.CLOSED_SHELL_ADSORBATE_REFERENCES == expected_refs
    assert checks.CLOSED_SHELL_ADSORBATE_REFERENCES == expected_refs


def test_oc20dense_full_data_root_extends_mapping_lookup(tmp_path, monkeypatch):
    common = _load_script("scripts/_oc20dense_common.py")
    local_root = tmp_path / "local"
    full_root = tmp_path / "full"
    (full_root / "mappings").mkdir(parents=True)
    expected = full_root / "mappings" / "oc20dense_mapping.pkl"
    expected.write_bytes(b"pickle placeholder")

    monkeypatch.setenv("OC20DENSE_FULL_DATA_ROOT", str(full_root))

    assert common.oc20dense_mapping_file(local_root, "oc20dense_mapping.pkl") == expected


def test_oc20dense_precomputed_write_requires_explicit_guard(monkeypatch):
    common = _load_script("scripts/_oc20dense_common.py")
    target = common.PART1 / "outputs" / "precomputed" / "accuracy" / "scratch"

    for name in (
        "REFRESH_SAVED_RESULTS",
        "ALCHEMI_ALLOW_ARTIFACT_OVERWRITE",
        "ALCHEMI_ALLOW_PRECOMPUTED_WRITE",
    ):
        monkeypatch.delenv(name, raising=False)

    try:
        common.require_precomputed_write_allowed(target)
    except PermissionError:
        pass
    else:
        raise AssertionError("precomputed writes should require an explicit guard")

    monkeypatch.setenv("ALCHEMI_ALLOW_PRECOMPUTED_WRITE", "1")
    common.require_precomputed_write_allowed(target)


def test_oc20dense_slim_subset_can_load_selected_initial_structures():
    known = _load_script("scripts/run_oc20dense_known_examples.py")
    data_root = PART1 / "data" / "reference" / "oc20dense"
    initial_dir = data_root / "initial_structures" / "adslab"
    expected_initial = initial_dir / "72_7104_115_rand27_sid3469.extxyz"
    if not expected_initial.exists():
        pytest.skip(
            "Slim OC20Dense validation structures are not bundled in the minimal "
            "GitHub checkout. Restore data/reference/oc20dense locally or set "
            "OC20DENSE_FULL_DATA_ROOT for live validation."
        )

    config = known.SelectedConfig(
        sid=3469,
        system_id="72_7104_115",
        config_id="rand27",
        mpid="mp-614572",
        miller_idx=(1, 1, 2),
        top=True,
        adsorbate="*NH3",
        dft_adsorption_energy_eV=-2.0497137549999707,
        dft_rank=1,
    )
    label, atoms, active_mask = known._atoms_for_selected_config(
        config,
        txn=None,
        initial_structure_dir=initial_dir,
    )

    assert label == "72_7104_115_rand27_sid3469"
    assert len(atoms) == len(active_mask)
    assert set(atoms.get_tags()) == {0, 1, 2}
    assert any(active_mask)


def test_oc20dense_current_tables_expose_rank_basis_and_reference_status():
    root = _require_oc20dense_artifacts()
    tables = root / "tables"
    per_config = _csv_header(tables / "per_config_results.csv")
    summary = _csv_header(tables / "system_summary.csv")
    comparison = _csv_header(
        root / "dft_reference_checks" / "dft_reference_comparison.csv"
    )

    for header in (per_config, summary, comparison):
        assert "mace_rank_basis" in header
        assert "mace_eads_reference_status" in header
    assert "mic_active_atom_rmsd_A" in comparison
    assert "mic_adsorbate_rmsd_A" in comparison


def test_oc20dense_current_outputs_use_closed_shell_system_set():
    root = _require_oc20dense_artifacts()
    expected = {
        "3_2070_48": ("*OH2", "H2O"),
        "72_7104_115": ("*NH3", "NH3"),
        "69_1615_2": ("*N2", "N2"),
    }

    summary = _csv_rows(root / "tables" / "system_summary.csv")
    comparison = _csv_rows(root / "dft_reference_checks" / "dft_reference_comparison.csv")

    assert {
        row["system_id"]: (row["adsorbate"], row["adsorbate_reference_species"])
        for row in summary
    } == expected
    assert {row["system_id"] for row in comparison} == set(expected)
    for row in summary:
        assert row["adsorbate"] != "*COH"
        assert "CH3" not in row["adsorbate"]


def test_oc20dense_model_specific_outputs_are_present():
    root = _require_oc20dense_artifacts()
    per_config = _csv_rows(root / "tables" / "per_config_results.csv")
    summary = _csv_rows(root / "tables" / "system_summary.csv")
    dft_final = _csv_rows(
        root
        / "dft_final_single_points"
        / "tables"
        / "dft_final_sp_system_summary.csv"
    )
    eads_summary = _csv_rows(
        root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_energy_summary.csv"
    )
    report = root / "reports" / "oc20dense_known_examples_report.md"

    assert report.exists()
    expected_systems = {
        "3_2070_48",
        "72_7104_115",
        "69_1615_2",
    }
    assert {row["system_id"] for row in summary} == expected_systems
    assert {row["system_id"] for row in per_config} == expected_systems
    assert {row["adsorbate"] for row in dft_final} == {"*OH2", "*NH3", "*N2"}
    assert {row["adsorbate"] for row in eads_summary} == {"*OH2", "*NH3", "*N2"}
    assert all(row["toolkit_checkpoint"] == "medium-mpa-0" for row in per_config)
    assert all(row["toolkit_head"] in {"", "None", None} for row in per_config)


def test_oc20dense_member_match_uses_exact_system_config_key():
    checks = _load_script("scripts/oc20dense_dft_reference_checks.py")
    key = checks.TrajectoryKey(system_id="8_12_4", config_id="rand42", sid=31534)

    assert checks._member_matches("trajs/8_12_4/8_12_4_rand42.traj.xz", [key]) == key
    assert checks._member_matches("trajs/8_12_4/8_12_4_rand43.traj.xz", [key]) is None


def test_oc20dense_config_ids_select_exact_subset():
    known = _load_script("scripts/run_oc20dense_known_examples.py")
    mapping_by_system = {
        "system-a": [
            (
                101,
                {
                    "config_id": "rand0",
                    "mpid": "mp-test",
                    "miller_idx": (1, 1, 1),
                    "top": True,
                    "adsorbate": "*OH2",
                },
            ),
            (
                102,
                {
                    "config_id": "rand1",
                    "mpid": "mp-test",
                    "miller_idx": (1, 1, 1),
                    "top": True,
                    "adsorbate": "*OH2",
                },
            ),
            (
                103,
                {
                    "config_id": "rand2",
                    "mpid": "mp-test",
                    "miller_idx": (1, 1, 1),
                    "top": True,
                    "adsorbate": "*OH2",
                },
            ),
        ]
    }
    targets = {"system-a": [("rand0", -0.1), ("rand1", -0.3), ("rand2", -0.2)]}

    selected = known._select_system_configs(
        system_id="system-a",
        mapping_by_system=mapping_by_system,
        targets=targets,
        max_configs=1,
        config_ids={"rand0", "rand2"},
    )

    assert [item.config_id for item in selected] == ["rand2", "rand0"]
    assert [item.sid for item in selected] == [103, 101]
    assert [item.dft_rank for item in selected] == [2, 3]


def test_oc20dense_dft_reference_arithmetic_round_trips():
    root = _require_oc20dense_artifacts()
    comparison = _csv_rows(
        root
        / "dft_reference_checks"
        / "dft_reference_comparison.csv"
    )

    for row in comparison:
        from_traj = float(row["dft_adsorption_energy_from_traj_eV"])
        target = float(row["dft_adsorption_energy_target_eV"])
        reported_delta = float(row["dft_traj_minus_target_eV"])
        assert abs((from_traj - target) - reported_delta) <= 1e-12
        assert abs(reported_delta) <= 1e-12


def test_oc20dense_mace_adsorption_energy_subtraction_round_trips():
    root = _require_oc20dense_artifacts()
    eads = _csv_rows(
        root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_energies.csv"
    )

    for row in eads:
        if row["mace_dft_final_sp_total_energy_eV"]:
            dft_final_eads = (
                float(row["mace_dft_final_sp_total_energy_eV"])
                - float(row["mace_surface_dft_final_sp_energy_eV"])
                - float(row["mace_gas_energy_eV"])
            )
            assert abs(dft_final_eads - float(row["mace_dft_final_eads_eV"])) <= 1e-10
        relaxed_eads = (
            float(row["ml_total_energy_eV"])
            - float(row["mace_surface_relaxed_energy_eV"])
            - float(row["mace_gas_energy_eV"])
        )
        assert abs(relaxed_eads - float(row["mace_relaxed_eads_eV"])) <= 1e-10


def test_oc20dense_notebook_live_subset_commands_are_isolated():
    notebook = json.loads(
        (PART1 / "oc20dense-accuracy-reproducibility-check.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert '"--config-ids"' in source
    assert '"--extract-dir"' in source
    assert '"extracted_trajectories"' in source
    assert '"--surface-dir"' in source
    assert '"surface_trajectories"' in source
    assert "include_groups=False" not in source


def test_oc20dense_summarize_median_handles_empty_spearman():
    summarize = _load_script("scripts/summarize_oc20dense_accuracy.py")

    assert math.isnan(summarize._median_or_nan([float("nan")]))
    assert summarize._median_or_nan([1.0, float("nan"), 3.0]) == 2.0
