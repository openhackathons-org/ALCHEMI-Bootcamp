"""Assemble the learner-facing rows in the Part 1 results summary.

The notebook still decides whether each numerical or scientific check passed.
Those decisions, their measured values, and every reason for not reporting a
result are explicit inputs here.  This module owns only stable row wording,
table assembly, and the count used by the final callout.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral, Real

import pandas as pd

from .domain.config import DOMAIN_METHODOLOGY


_COLUMNS = ("Result", "Status", "Measured", "Applies to")
_NOT_REPORTED = "NOT REPORTED"
_DOMAIN_DISTRIBUTED_GPU_TEXT = " or ".join(
    str(world_size) for world_size in DOMAIN_METHODOLOGY.distributed_world_sizes
)


def _bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _count(value: int, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _measurement(value: float, *, name: str, non_negative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _texts(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_text(value, name=f"{name} item") for value in values)
    return result


def build_results_summary(
    *,
    batch_results_match: bool,
    serial_batch_error_eV: float,
    full_pipeline_agreement_error_eV: float,
    cpu_gpu_crossover_batch_size: int | None,
    cpu_gpu_largest_batch_size: int,
    cpu_gpu_largest_batch_speedup: float,
    cpu_gpu_max_energy_difference_eV: float,
    sevennet_status: str,
    sevennet_structure_count: int,
    sevennet_batch_count: int,
    sevennet_molecule_count: int,
    sevennet_max_edge_vector_mapping_difference_A: float,
    sevennet_repeat_max_energy_difference_eV_per_atom: float,
    sevennet_repeat_max_force_difference_eV_A: float,
    sevennet_max_force_eV_A: float,
    nci_geometry_count: int,
    nci_graph_count: int,
    nci_max_mae_vs_dft_d3_kcal_mol: float,
    nci_max_mae_vs_ccsd_t_cbs_kcal_mol: float,
    harmonic_comparison_reported: bool,
    harmonic_frequency_mae_cm1: float | None,
    harmonic_failed_checks: Sequence[str],
    inflight_queue_complete: bool,
    inflight_system_count: int,
    inflight_active_system_count: int,
    inflight_nvt_steps: int,
    inflight_nve_steps: int,
    domain_live_api_passed: bool,
    domain_live_world_size: int,
    domain_live_spatially_decomposed: bool,
    domain_live_atom_count: int,
    domain_live_energy_per_atom_eV: float,
    domain_live_max_force_eV_A: float,
    domain_live_charge_sum_e: float,
    domain_results_available: bool,
    domain_results_unavailable_reason: str | None,
    domain_successful_cases: int,
    domain_failed_cases: int,
    domain_planned_max_atom_count: int,
    domain_measured_max_atom_count: int | None,
    campaign_available: bool,
    campaign_unavailable_reason: str | None,
    campaign_successes: int,
    campaign_failures: int,
    campaign_systems_total: int,
    monomer_shown: bool,
    monomer_status: str,
    cluster_shown: bool,
    cluster_not_reported_reasons: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Return the Part 1 summary table and its ``NOT REPORTED`` row count.

    Reporting decisions are intentionally not inferred here.  In particular,
    the notebook applies the serial/batch tolerances, harmonic numerical
    checks, temperature checks, and topology check before calling this helper.
    """

    batch_match = _bool(batch_results_match, name="batch_results_match")
    serial_error = _measurement(serial_batch_error_eV, name="serial_batch_error_eV")
    complete_error = _measurement(
        full_pipeline_agreement_error_eV,
        name="full_pipeline_agreement_error_eV",
    )
    if cpu_gpu_crossover_batch_size is None:
        crossover_batch_size = None
    else:
        crossover_batch_size = _count(
            cpu_gpu_crossover_batch_size,
            name="cpu_gpu_crossover_batch_size",
            positive=True,
        )
    largest_batch_size = _count(
        cpu_gpu_largest_batch_size,
        name="cpu_gpu_largest_batch_size",
        positive=True,
    )
    largest_batch_speedup = _measurement(
        cpu_gpu_largest_batch_speedup,
        name="cpu_gpu_largest_batch_speedup",
    )
    cpu_gpu_energy_error = _measurement(
        cpu_gpu_max_energy_difference_eV,
        name="cpu_gpu_max_energy_difference_eV",
    )
    sevennet_adapter_status = _text(sevennet_status, name="sevennet_status")
    sevennet_structures = _count(
        sevennet_structure_count,
        name="sevennet_structure_count",
        positive=True,
    )
    sevennet_batches = _count(
        sevennet_batch_count,
        name="sevennet_batch_count",
        positive=True,
    )
    sevennet_molecules = _count(
        sevennet_molecule_count,
        name="sevennet_molecule_count",
        positive=True,
    )
    sevennet_mapping_error = _measurement(
        sevennet_max_edge_vector_mapping_difference_A,
        name="sevennet_max_edge_vector_mapping_difference_A",
    )
    sevennet_repeat_energy_error = _measurement(
        sevennet_repeat_max_energy_difference_eV_per_atom,
        name="sevennet_repeat_max_energy_difference_eV_per_atom",
    )
    sevennet_repeat_force_error = _measurement(
        sevennet_repeat_max_force_difference_eV_A,
        name="sevennet_repeat_max_force_difference_eV_A",
    )
    sevennet_max_force = _measurement(
        sevennet_max_force_eV_A,
        name="sevennet_max_force_eV_A",
    )
    nci_geometries = _count(
        nci_geometry_count, name="nci_geometry_count", positive=True
    )
    nci_graphs = _count(nci_graph_count, name="nci_graph_count", positive=True)
    nci_dft_mae = _measurement(
        nci_max_mae_vs_dft_d3_kcal_mol,
        name="nci_max_mae_vs_dft_d3_kcal_mol",
    )
    nci_cc_mae = _measurement(
        nci_max_mae_vs_ccsd_t_cbs_kcal_mol,
        name="nci_max_mae_vs_ccsd_t_cbs_kcal_mol",
    )

    harmonic_reported = _bool(
        harmonic_comparison_reported,
        name="harmonic_comparison_reported",
    )
    failed_harmonic_checks = _texts(
        harmonic_failed_checks,
        name="harmonic_failed_checks",
    )
    if harmonic_reported:
        if harmonic_frequency_mae_cm1 is None:
            raise ValueError(
                "harmonic_frequency_mae_cm1 is required for a reported result"
            )
        harmonic_mae = _measurement(
            harmonic_frequency_mae_cm1,
            name="harmonic_frequency_mae_cm1",
        )
        if failed_harmonic_checks:
            raise ValueError(
                "harmonic_failed_checks must be empty for a reported result"
            )
    else:
        harmonic_mae = None
        if harmonic_frequency_mae_cm1 is not None:
            raise ValueError(
                "harmonic_frequency_mae_cm1 must be None for an unreported result"
            )
        if not failed_harmonic_checks:
            raise ValueError(
                "harmonic_failed_checks must explain an unreported harmonic result"
            )

    inflight_complete = _bool(
        inflight_queue_complete,
        name="inflight_queue_complete",
    )
    inflight_systems = _count(
        inflight_system_count,
        name="inflight_system_count",
        positive=True,
    )
    inflight_active_systems = _count(
        inflight_active_system_count,
        name="inflight_active_system_count",
        positive=True,
    )
    inflight_nvt = _count(inflight_nvt_steps, name="inflight_nvt_steps")
    inflight_nve = _count(inflight_nve_steps, name="inflight_nve_steps")
    if inflight_active_systems > inflight_systems:
        raise ValueError(
            "inflight_active_system_count cannot exceed inflight_system_count"
        )

    live_domain_passed = _bool(
        domain_live_api_passed,
        name="domain_live_api_passed",
    )
    live_domain_world_size = _count(
        domain_live_world_size,
        name="domain_live_world_size",
        positive=True,
    )
    live_domain_decomposed = _bool(
        domain_live_spatially_decomposed,
        name="domain_live_spatially_decomposed",
    )
    live_domain_atoms = _count(
        domain_live_atom_count,
        name="domain_live_atom_count",
        positive=True,
    )
    live_domain_energy_per_atom = _measurement(
        domain_live_energy_per_atom_eV,
        name="domain_live_energy_per_atom_eV",
        non_negative=False,
    )
    live_domain_max_force = _measurement(
        domain_live_max_force_eV_A,
        name="domain_live_max_force_eV_A",
    )
    live_domain_charge_sum = _measurement(
        domain_live_charge_sum_e,
        name="domain_live_charge_sum_e",
        non_negative=False,
    )

    has_domain_results = _bool(
        domain_results_available,
        name="domain_results_available",
    )
    successful_domain_cases = _count(
        domain_successful_cases,
        name="domain_successful_cases",
    )
    failed_domain_cases = _count(
        domain_failed_cases,
        name="domain_failed_cases",
    )
    planned_domain_max_atoms = _count(
        domain_planned_max_atom_count,
        name="domain_planned_max_atom_count",
        positive=True,
    )
    if has_domain_results:
        if domain_results_unavailable_reason is not None:
            raise ValueError(
                "domain_results_unavailable_reason must be None when "
                "domain_results_available is True"
            )
        if domain_measured_max_atom_count is None:
            raise ValueError(
                "domain_measured_max_atom_count is required when "
                "domain_results_available is True"
            )
        unavailable_domain_reason = None
        measured_domain_max_atoms = _count(
            domain_measured_max_atom_count,
            name="domain_measured_max_atom_count",
            positive=True,
        )
    else:
        if domain_results_unavailable_reason is None:
            raise ValueError(
                "domain_results_unavailable_reason must explain unavailable "
                "DomainParallel results"
            )
        unavailable_domain_reason = _text(
            domain_results_unavailable_reason,
            name="domain_results_unavailable_reason",
        )
        if domain_measured_max_atom_count is not None:
            raise ValueError(
                "domain_measured_max_atom_count must be None when "
                "domain_results_available is False"
            )
        measured_domain_max_atoms = None

    has_campaign = _bool(campaign_available, name="campaign_available")
    if has_campaign:
        if campaign_unavailable_reason is not None:
            raise ValueError(
                "campaign_unavailable_reason must be None when campaign_available "
                "is True"
            )
        unavailable_campaign_reason = None
    else:
        if campaign_unavailable_reason is None:
            raise ValueError(
                "campaign_unavailable_reason must explain an unavailable campaign"
            )
        unavailable_campaign_reason = _text(
            campaign_unavailable_reason,
            name="campaign_unavailable_reason",
        )
    successful_runs = _count(campaign_successes, name="campaign_successes")
    failed_runs = _count(campaign_failures, name="campaign_failures")
    campaign_systems = _count(
        campaign_systems_total,
        name="campaign_systems_total",
        positive=True,
    )

    show_monomer = _bool(monomer_shown, name="monomer_shown")
    monomer_detail = _text(monomer_status, name="monomer_status")
    show_cluster = _bool(cluster_shown, name="cluster_shown")
    cluster_reasons = tuple(
        sorted(
            set(
                _texts(
                    cluster_not_reported_reasons,
                    name="cluster_not_reported_reasons",
                )
            )
        )
    )
    if not show_cluster and not cluster_reasons:
        raise ValueError(
            "cluster_not_reported_reasons must explain unreported cluster shifts"
        )

    table = pd.DataFrame(
        [
            {
                "Result": "Serial and batched energies match",
                "Status": "PASS" if batch_match else "CHECK FAILED",
                "Measured": (
                    f"checkpoint-base Δ={serial_error:.2e} eV; "
                    f"complete-model Δ={complete_error:.2e} eV"
                ),
                "Applies to": "this structure set and model configuration",
            },
            {
                "Result": "CPU/GPU warm-call crossover",
                "Status": "OBSERVED",
                "Measured": (
                    (
                        f"GPU first exceeds CPU at batch size "
                        f"{crossover_batch_size}; "
                    )
                    if crossover_batch_size is not None
                    else (
                        f"GPU crossover not reached through batch size "
                        f"{largest_batch_size}; "
                    )
                )
                + (
                    f"GPU/CPU throughput at batch {largest_batch_size}="
                    f"{largest_batch_speedup:.2f}×; max energy "
                    f"Δ={cpu_gpu_energy_error:.2e} eV"
                ),
                "Applies to": (
                    "synchronized warm AIMNet2 energy calls on this CPU and GPU"
                ),
            },
            {
                "Result": "Raw SevenNet-Omni through a custom Toolkit adapter",
                "Status": sevennet_adapter_status,
                "Measured": (
                    f"{sevennet_structures} structures in {sevennet_batches} batches; "
                    f"fixed-geometry panel of {sevennet_molecules} molecules; "
                    f"edge-vector mapping Δ={sevennet_mapping_error:.2e} Å; "
                    "adapter/pipeline ΔE="
                    f"{sevennet_repeat_energy_error:.2e} eV/atom and "
                    f"ΔF={sevennet_repeat_force_error:.2e} eV/Å; "
                    f"max |F|={sevennet_max_force:.2e} eV/Å"
                ),
                "Applies to": (
                    "fixed initial Cu(111) placements; not equilibrium adsorption "
                    "energies or a DFT accuracy benchmark"
                ),
            },
            {
                "Result": "Complete AIMNet interaction model on NCI Atlas",
                "Status": "PASS",
                "Measured": (
                    f"{nci_geometries} geometries / {nci_graphs} graphs; "
                    f"max per-system MAE={nci_dft_mae:.2f} kcal/mol vs DFT-D3 "
                    f"and {nci_cc_mae:.2f} kcal/mol vs CCSD(T)/CBS"
                ),
                "Applies to": (
                    "three curated interaction curves; focused composition check, "
                    "not broad MLIP validation"
                ),
            },
            {
                "Result": "Full-model harmonic IR vs B97-3c",
                "Status": "PASS" if harmonic_reported else _NOT_REPORTED,
                "Measured": (
                    f"six-mode frequency MAE={harmonic_mae:.1f} cm⁻¹"
                    if harmonic_reported
                    else "; ".join(failed_harmonic_checks)
                ),
                "Applies to": (
                    "H2O/D2O monomer; matched finite-difference and normal-mode "
                    "analysis at the checkpoint target level; frequency score only"
                ),
            },
            {
                "Result": "Inflight queue completed",
                "Status": "PASS" if inflight_complete else "CHECK FAILED",
                "Measured": (
                    f"{inflight_systems:,} systems completed with at most "
                    f"{inflight_active_systems:,} active; "
                    f"{inflight_nvt} NVT + {inflight_nve} NVE updates each"
                ),
                "Applies to": (
                    "queue scheduling and result collection; the short trajectories "
                    "are not scientific MD"
                ),
            },
            {
                "Result": "Live one-GPU DomainParallel API call",
                "Status": "PASS" if live_domain_passed else "CHECK FAILED",
                "Measured": (
                    f"{live_domain_atoms:,} atoms on {live_domain_world_size} rank; "
                    f"spatially decomposed={live_domain_decomposed}; "
                    f"energy/atom={live_domain_energy_per_atom:.6f} eV; "
                    f"max |F|={live_domain_max_force:.3f} eV/Å; "
                    f"Σq={live_domain_charge_sum:.3e} e"
                ),
                "Applies to": (
                    "one unequilibrated periodic Packmol box; API check, not "
                    "multi-GPU scaling"
                ),
            },
            {
                "Result": "DomainParallel multi-GPU scaling",
                "Status": "RECORDED" if has_domain_results else _NOT_REPORTED,
                "Measured": (
                    (
                        f"{successful_domain_cases} successful saved cases; "
                        f"{failed_domain_cases} failed saved cases; "
                        f"measured maximum {measured_domain_max_atoms:,} atoms"
                    )
                    if has_domain_results
                    else (
                        f"{unavailable_domain_reason}; "
                        f"{successful_domain_cases} successful saved cases; "
                        f"{failed_domain_cases} failed saved cases; "
                        f"measured maximum {_NOT_REPORTED}"
                    )
                ),
                "Applies to": (
                    f"planned maximum {planned_domain_max_atoms:,} atoms; "
                    "one periodic system split across "
                    f"{_DOMAIN_DISTRIBUTED_GPU_TEXT} GPUs"
                ),
            },
            {
                "Result": "DistributedPipeline throughput",
                "Status": (
                    "RECORDED" if has_campaign and successful_runs else _NOT_REPORTED
                ),
                "Measured": (
                    (
                        f"{successful_runs} successful saved timing runs; "
                        f"{failed_runs} failed runs"
                    )
                    if has_campaign
                    else unavailable_campaign_reason
                ),
                "Applies to": (
                    f"{'fixed' if has_campaign else 'planned'} "
                    f"{campaign_systems:,}-hexamer campaign; "
                    "one trajectory is not split across GPUs"
                ),
            },
            {
                "Result": "H2O/D2O monomer shift",
                "Status": "PASS" if show_monomer else _NOT_REPORTED,
                "Measured": monomer_detail,
                "Applies to": (
                    "MD, harmonic DFT, and observed positions use separate "
                    "intensity scales"
                ),
            },
            {
                "Result": "Cluster/monomer shifts",
                "Status": "PASS" if show_cluster else _NOT_REPORTED,
                "Measured": (
                    "temperature and topology requirements met"
                    if show_cluster
                    else "; ".join(cluster_reasons)
                ),
                "Applies to": (
                    "shown only when temperatures match and the original ring remains"
                ),
            },
        ],
        columns=_COLUMNS,
    )
    not_reported_count = int(table["Status"].eq(_NOT_REPORTED).sum())
    return table, not_reported_count


__all__ = ["build_results_summary"]
