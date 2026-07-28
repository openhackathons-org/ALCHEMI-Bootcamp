"""Focused checks for assembling the non-teaching Part 1 report."""

from __future__ import annotations

import ast
import inspect
import json
import symtable
import sys
import textwrap
from pathlib import Path

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.notebook_reporting import (  # noqa: E402
    _MANIFEST_CHECK_NAMES,
    _MANIFEST_MODEL_SETTING_NAMES,
    _MANIFEST_RUN_DETAIL_NAMES,
    _MANIFEST_WORKFLOW_SETTING_NAMES,
    _REPORT_REQUIRED_NAMES,
    _RESULT_SUMMARY_NAMES,
    _WATER_RUN_RESULT_NAMES,
    MissingNotebookVariablesError,
    _build_manifest_checks,
    _build_manifest_model_settings,
    _build_manifest_run_details,
    _build_manifest_workflow_settings,
    build_part1_notebook_report,
    build_part1_results_summary,
    build_part1_water_run_results,
)
from aux.run_output import _RUN_DETAIL_SECTION_KEYS  # noqa: E402


NOTEBOOK_PATH = PART_DIR / "alchemi-water-ir.ipynb"


def _indexed_value_keys(function: object) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.slice.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "values"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        )
    }


def test_aimnet_file_identities_are_required_in_the_saved_manifest() -> None:
    assert "aimnet_checkpoint_identities" in _MANIFEST_RUN_DETAIL_NAMES
    assert (
        "aimnet_checkpoint_identities"
        in _RUN_DETAIL_SECTION_KEYS["nci_data"]
    )


def test_missing_notebook_variables_are_reported_together() -> None:
    with pytest.raises(MissingNotebookVariablesError) as caught:
        build_part1_notebook_report({})

    missing_names = caught.value.missing_names
    assert missing_names == tuple(sorted(missing_names))
    assert {
        "ADSORBATES",
        "diagnostic_table",
        "serial_batch_error",
        "spectra",
    } <= set(missing_names)
    assert "manifest_checks" not in missing_names
    assert str(caught.value).startswith(
        "Part 1 notebook report is missing required notebook variables:"
    )


def test_report_builders_declare_every_namespace_value_they_read() -> None:
    builders = (
        (build_part1_results_summary, set(_RESULT_SUMMARY_NAMES)),
        (
            build_part1_water_run_results,
            set(_WATER_RUN_RESULT_NAMES) | {"results_summary"},
        ),
        (_build_manifest_run_details, set(_MANIFEST_RUN_DETAIL_NAMES)),
        (_build_manifest_model_settings, set(_MANIFEST_MODEL_SETTING_NAMES)),
        (_build_manifest_workflow_settings, set(_MANIFEST_WORKFLOW_SETTING_NAMES)),
        (_build_manifest_checks, set(_MANIFEST_CHECK_NAMES)),
    )
    for function, declared_names in builders:
        used_names = _indexed_value_keys(function)
        undeclared = used_names - declared_names
        unused = declared_names - used_names
        assert not undeclared and not unused, (
            f"{function.__name__} notebook-value declaration mismatch; "
            f"undeclared reads: {sorted(undeclared)}; "
            f"unused declarations: {sorted(unused)}"
        )


def test_report_inputs_exist_before_the_summary_cell_runs() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    available: set[str] = set()
    found_summary = False

    for cell in notebook["cells"]:
        if cell.get("id") == "results-summary":
            found_summary = True
            break
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source_text = source if isinstance(source, str) else "".join(source)
        table = symtable.symtable(
            source_text,
            f"{NOTEBOOK_PATH}#{cell.get('id', '<missing-id>')}",
            "exec",
        )
        available.update(
            name
            for name in table.get_identifiers()
            if (
                table.lookup(name).is_assigned()
                or table.lookup(name).is_imported()
                or table.lookup(name).is_namespace()
            )
        )

    assert found_summary
    missing = set(_REPORT_REQUIRED_NAMES) - available
    assert not missing, (
        "results-summary asks for notebook values that no earlier cell defines: "
        f"{sorted(missing)}"
    )


def test_domain_report_settings_come_from_the_versioned_methodology() -> None:
    legacy_names = {
        "DOMAIN_CHARGE_SUM_TOLERANCE_E",
        "DOMAIN_CONSTRUCTION_DENSITY_G_CM3",
        "DOMAIN_LIVE_MOLECULES_PER_SPECIES",
        "DOMAIN_PACKMOL_PRECISION_A",
        "DOMAIN_PACKMOL_SEED",
        "DOMAIN_PACKMOL_TOLERANCE_A",
        "DOMAIN_PLANNED_ATOM_COUNTS",
        "PME_ACCURACY",
        "PME_MESH_SAFETY_FACTOR",
        "PME_REALSPACE_CUTOFF_A",
    }
    assert "DOMAIN_METHODOLOGY" in _REPORT_REQUIRED_NAMES
    assert legacy_names.isdisjoint(_REPORT_REQUIRED_NAMES)


def test_water_run_results_collect_every_existing_report_field() -> None:
    source_by_field = {
        "diagnostics": "diagnostic_table",
        "spectrum_metrics": "metrics",
        "topology_summary": "integrity_table",
        "comparisons": "comparisons",
        "dft_comparison": "reference_metrics",
        "h_to_d_mode_map": "mode_map_table",
        "harmonic_displacements": "harmonic_fd_table",
        "harmonic_convergence": "harmonic_convergence_table",
        "harmonic_checks": "harmonic_validation_table",
        "harmonic_comparison": "harmonic_comparison_table",
        "nci_interaction_curves": "nci_curves",
        "nci_interaction_metrics": "nci_metrics",
        "nci_ensemble_curves": "nci_member_curves",
        "dimer_ablation": "dimer_table",
        "dimer_ablation_mae": "ablation_mae",
        "adsorption_results": "adsorption_results",
        "adsorption_forces": "adsorption_forces",
        "sevennet_graph_mapping": "sevennet_graph_mapping",
        "sevennet_numerical_agreement": "sevennet_numerical_agreement",
        "first_warm_calls": "cold_warm",
        "cpu_gpu_crossover": "crossover",
        "inflight_summary": "inflight_summary",
        "domain_molecule_charges": "domain_molecule_charges",
        "domain_molecule_charge_summary": "domain_molecule_charge_summary",
        "domain_live_summary": "domain_live_summary",
        "topology_timelines": "topology_timelines",
        "spectra": "spectra",
    }
    namespace = {source_name: object() for source_name in source_by_field.values()}
    timings = object()
    namespace["layout_result"] = {"timings": timings}
    results_summary = object()

    results = build_part1_water_run_results(
        namespace,
        results_summary=results_summary,
    )

    for field_name, source_name in source_by_field.items():
        assert getattr(results, field_name) is namespace[source_name]
    assert results.results_summary is results_summary
    assert results.batch_layout_timings is timings
