"""Focused checks for the learner-facing Part 1 results summary."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.results_summary import build_results_summary  # noqa: E402


def _inputs() -> dict[str, object]:
    return {
        "batch_results_match": True,
        "serial_batch_error_eV": 1.23e-7,
        "full_pipeline_agreement_error_eV": 4.56e-6,
        "cpu_gpu_crossover_batch_size": 8,
        "cpu_gpu_largest_batch_size": 128,
        "cpu_gpu_largest_batch_speedup": 4.25,
        "cpu_gpu_max_energy_difference_eV": 1.20e-6,
        "sevennet_status": "PASS",
        "sevennet_structure_count": 9,
        "sevennet_batch_count": 2,
        "sevennet_molecule_count": 4,
        "sevennet_max_edge_vector_mapping_difference_A": 5.00e-8,
        "sevennet_repeat_max_energy_difference_eV_per_atom": 1.43e-6,
        "sevennet_repeat_max_force_difference_eV_A": 5.96e-7,
        "sevennet_max_force_eV_A": 1.24,
        "nci_geometry_count": 30,
        "nci_graph_count": 90,
        "nci_max_mae_vs_dft_d3_kcal_mol": 0.38,
        "nci_max_mae_vs_ccsd_t_cbs_kcal_mol": 0.44,
        "harmonic_comparison_reported": True,
        "harmonic_frequency_mae_cm1": 18.4,
        "harmonic_failed_checks": (),
        "inflight_queue_complete": True,
        "inflight_system_count": 2_048,
        "inflight_active_system_count": 256,
        "inflight_nvt_steps": 2,
        "inflight_nve_steps": 3,
        "domain_live_api_passed": True,
        "domain_live_world_size": 1,
        "domain_live_spatially_decomposed": False,
        "domain_live_atom_count": 3_200,
        "domain_live_energy_per_atom_eV": -0.123456,
        "domain_live_max_force_eV_A": 1.234,
        "domain_live_charge_sum_e": 2.30e-6,
        "domain_results_available": True,
        "domain_results_unavailable_reason": None,
        "domain_successful_cases": 4,
        "domain_failed_cases": 0,
        "domain_planned_max_atom_count": 51_200,
        "domain_measured_max_atom_count": 51_200,
        "campaign_available": True,
        "campaign_unavailable_reason": None,
        "campaign_successes": 12,
        "campaign_failures": 0,
        "campaign_systems_total": 8_192,
        "monomer_shown": True,
        "monomer_status": "reported: temperature requirement met",
        "cluster_shown": True,
        "cluster_not_reported_reasons": (),
    }


def test_all_reported_rows_keep_the_exact_order_and_wording() -> None:
    table, not_reported_count = build_results_summary(**_inputs())

    assert list(table.columns) == ["Result", "Status", "Measured", "Applies to"]
    assert table["Result"].tolist() == [
        "Serial and batched energies match",
        "CPU/GPU warm-call crossover",
        "Raw SevenNet-Omni through a custom Toolkit adapter",
        "Complete AIMNet interaction model on NCI Atlas",
        "Full-model harmonic IR vs B97-3c",
        "Inflight queue completed",
        "Live one-GPU DomainParallel API call",
        "Fixed-input DomainParallel passes",
        "DistributedPipeline throughput",
        "H2O/D2O monomer shift",
        "Cluster/monomer shifts",
    ]
    assert table["Status"].tolist() == [
        "PASS",
        "OBSERVED",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "RECORDED",
        "RECORDED",
        "PASS",
        "PASS",
    ]
    assert not_reported_count == 0
    assert table.loc[0, "Measured"] == (
        "checkpoint-base Δ=1.23e-07 eV; complete-model Δ=4.56e-06 eV"
    )
    assert table.loc[1, "Measured"] == (
        "GPU first exceeds CPU at batch size 8; "
        "GPU/CPU throughput at batch 128=4.25×; max energy Δ=1.20e-06 eV"
    )
    assert table.loc[2, "Measured"] == (
        "9 structures in 2 batches; fixed-geometry panel of 4 molecules; "
        "edge-vector mapping Δ=5.00e-08 Å; adapter/pipeline "
        "ΔE=1.43e-06 eV/atom and "
        "ΔF=5.96e-07 eV/Å; max |F|=1.24e+00 eV/Å"
    )
    assert table.loc[2, "Applies to"] == (
        "fixed initial Cu(111) placements; not equilibrium adsorption energies "
        "or a DFT accuracy benchmark"
    )
    assert table.loc[3, "Measured"] == (
        "30 geometries / 90 graphs; max per-system MAE=0.38 kcal/mol vs DFT-D3 "
        "and 0.44 kcal/mol vs CCSD(T)/CBS"
    )
    assert table.loc[3, "Applies to"] == (
        "three curated interaction curves; focused composition check, "
        "not broad MLIP validation"
    )
    assert table.loc[4, "Measured"] == "six-mode frequency MAE=18.4 cm⁻¹"
    assert table.loc[5, "Measured"] == (
        "2,048 systems completed with at most 256 active; 2 NVT + 3 NVE updates each"
    )
    assert table.loc[6, "Measured"] == (
        "3,200 atoms on 1 rank; spatially decomposed=False; "
        "energy/atom=-0.123456 eV; max |F|=1.234 eV/Å; Σq=2.300e-06 e"
    )
    assert table.loc[7, "Measured"] == (
        "4 successful saved cases; 0 failed saved cases; same 51,200-atom input"
    )
    assert table.loc[7, "Applies to"] == (
        "fixed 51,200-atom input; one warm-up and three measured "
        "energy/force passes on 1, 2, 4 GPUs"
    )
    assert table.loc[8, "Measured"] == (
        "12 successful saved timing runs; 0 failed runs"
    )
    assert table.loc[10, "Measured"] == ("temperature and topology requirements met")


def test_unreported_rows_keep_reasons_and_exclude_check_failed_from_count() -> None:
    values = _inputs()
    values.update(
        {
            "batch_results_match": False,
            "harmonic_comparison_reported": False,
            "harmonic_frequency_mae_cm1": None,
            "harmonic_failed_checks": (
                "imaginary_mode_check",
                "hessian_symmetry_check",
            ),
            "domain_results_available": False,
            "domain_results_unavailable_reason": (
                "Recorded DomainParallel results are unavailable because "
                "transfer failed"
            ),
            "domain_successful_cases": 0,
            "domain_failed_cases": 0,
            "domain_measured_max_atom_count": None,
            "campaign_available": False,
            "campaign_unavailable_reason": (
                "Recorded H100 timings are unavailable because transfer failed"
            ),
            "campaign_successes": 0,
            "campaign_failures": 0,
            "monomer_shown": False,
            "monomer_status": "temperature requirement not met",
            "cluster_shown": False,
            "cluster_not_reported_reasons": (
                "ring topology requirement not met",
                "temperature requirement not met",
                "ring topology requirement not met",
            ),
        }
    )

    table, not_reported_count = build_results_summary(**values)

    assert table["Status"].tolist() == [
        "CHECK FAILED",
        "OBSERVED",
        "PASS",
        "PASS",
        "NOT REPORTED",
        "PASS",
        "PASS",
        "NOT REPORTED",
        "NOT REPORTED",
        "NOT REPORTED",
        "NOT REPORTED",
    ]
    assert not_reported_count == 5
    assert table.loc[4, "Measured"] == ("imaginary_mode_check; hessian_symmetry_check")
    assert table.loc[7, "Measured"] == (
        "Recorded DomainParallel results are unavailable because transfer failed; "
        "0 successful saved cases; 0 failed saved cases; "
        "fixed input NOT REPORTED"
    )
    assert table.loc[7, "Applies to"] == (
        "fixed 51,200-atom input; one warm-up and three measured "
        "energy/force passes on 1, 2, 4 GPUs"
    )
    assert table.loc[8, "Measured"] == (
        "Recorded H100 timings are unavailable because transfer failed"
    )
    assert table.loc[8, "Applies to"] == (
        "planned 8,192-hexamer campaign; one trajectory is not split across GPUs"
    )
    assert table.loc[9, "Measured"] == "temperature requirement not met"
    assert table.loc[10, "Measured"] == (
        "ring topology requirement not met; temperature requirement not met"
    )


@pytest.mark.parametrize(
    ("available", "successes", "expected_status", "expected_count"),
    [
        (True, 3, "RECORDED", 0),
        (True, 0, "NOT REPORTED", 1),
        (False, 0, "NOT REPORTED", 1),
    ],
)
def test_campaign_reporting_requires_available_successful_timings(
    available: bool,
    successes: int,
    expected_status: str,
    expected_count: int,
) -> None:
    values = _inputs()
    values.update(
        {
            "campaign_available": available,
            "campaign_unavailable_reason": (
                None if available else "No verified timing bundle is installed"
            ),
            "campaign_successes": successes,
            "campaign_failures": 2,
        }
    )

    table, not_reported_count = build_results_summary(**values)

    assert table.loc[8, "Status"] == expected_status
    assert not_reported_count == expected_count
    if available:
        assert table.loc[8, "Measured"] == (
            f"{successes} successful saved timing runs; 2 failed runs"
        )


@pytest.mark.parametrize(
    (
        "available",
        "reason",
        "successes",
        "failures",
        "measured_max_atoms",
        "expected_status",
        "expected_count",
    ),
    [
        (True, None, 4, 0, 51_200, "RECORDED", 0),
        (
            False,
            "No verified DomainParallel bundle is installed",
            0,
            0,
            None,
            "NOT REPORTED",
            1,
        ),
    ],
)
def test_domain_reporting_distinguishes_recorded_and_unavailable_results(
    available: bool,
    reason: str | None,
    successes: int,
    failures: int,
    measured_max_atoms: int | None,
    expected_status: str,
    expected_count: int,
) -> None:
    values = _inputs()
    values.update(
        {
            "domain_results_available": available,
            "domain_results_unavailable_reason": reason,
            "domain_successful_cases": successes,
            "domain_failed_cases": failures,
            "domain_measured_max_atom_count": measured_max_atoms,
        }
    )

    table, not_reported_count = build_results_summary(**values)

    assert table.loc[7, "Status"] == expected_status
    assert not_reported_count == expected_count
    if available:
        assert table.loc[7, "Measured"] == (
            "4 successful saved cases; 0 failed saved cases; same 51,200-atom input"
        )
    else:
        assert table.loc[7, "Measured"] == (
            "No verified DomainParallel bundle is installed; "
            "0 successful saved cases; 0 failed saved cases; "
            "fixed input NOT REPORTED"
        )


def test_sevennet_status_is_supplied_by_the_notebook() -> None:
    values = _inputs()
    values["sevennet_status"] = "CHECK FAILED"

    table, not_reported_count = build_results_summary(**values)

    assert table.loc[2, "Status"] == "CHECK FAILED"
    assert not_reported_count == 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "harmonic_comparison_reported": False,
                "harmonic_frequency_mae_cm1": None,
                "harmonic_failed_checks": (),
            },
            "harmonic_failed_checks",
        ),
        (
            {
                "harmonic_comparison_reported": False,
                "harmonic_frequency_mae_cm1": 18.4,
                "harmonic_failed_checks": ("tight minimum",),
            },
            "harmonic_frequency_mae_cm1",
        ),
        (
            {
                "harmonic_comparison_reported": True,
                "harmonic_frequency_mae_cm1": None,
            },
            "harmonic_frequency_mae_cm1",
        ),
        (
            {
                "harmonic_comparison_reported": True,
                "harmonic_failed_checks": ("tight minimum",),
            },
            "harmonic_failed_checks",
        ),
        (
            {
                "cluster_shown": False,
                "cluster_not_reported_reasons": (),
            },
            "cluster_not_reported_reasons",
        ),
        (
            {
                "domain_results_available": False,
                "domain_results_unavailable_reason": None,
                "domain_measured_max_atom_count": None,
            },
            "domain_results_unavailable_reason",
        ),
        (
            {
                "campaign_available": False,
                "campaign_unavailable_reason": None,
            },
            "campaign_unavailable_reason",
        ),
    ],
)
def test_unreported_rows_require_a_reason(
    updates: dict[str, object],
    message: str,
) -> None:
    values = _inputs()
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        build_results_summary(**values)


def test_returned_table_is_a_plain_dataframe() -> None:
    table, count = build_results_summary(**_inputs())

    assert type(table) is pd.DataFrame
    assert isinstance(count, int)


def test_available_campaign_rejects_an_unavailable_reason() -> None:
    values = _inputs()
    values["campaign_unavailable_reason"] = "stale failure text"

    with pytest.raises(ValueError, match="campaign_unavailable_reason"):
        build_results_summary(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "domain_results_unavailable_reason": "stale failure text",
            },
            "domain_results_unavailable_reason",
        ),
        (
            {
                "domain_measured_max_atom_count": None,
            },
            "domain_measured_max_atom_count",
        ),
        (
            {
                "domain_results_available": False,
                "domain_results_unavailable_reason": "bundle is not installed",
            },
            "domain_measured_max_atom_count",
        ),
    ],
)
def test_domain_availability_rejects_inconsistent_measurement_state(
    updates: dict[str, object],
    message: str,
) -> None:
    values = _inputs()
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        build_results_summary(**values)
