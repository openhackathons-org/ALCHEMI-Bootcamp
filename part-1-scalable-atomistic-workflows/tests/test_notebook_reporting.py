"""Focused checks for assembling the non-teaching Part 1 report."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.notebook_reporting import (  # noqa: E402
    _MANIFEST_RUN_DETAIL_NAMES,
    MissingNotebookVariablesError,
    build_part1_notebook_report,
    build_part1_water_run_results,
)
from aux.run_output import _RUN_DETAIL_SECTION_KEYS  # noqa: E402


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
