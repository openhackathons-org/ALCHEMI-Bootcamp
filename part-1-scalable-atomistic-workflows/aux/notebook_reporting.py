"""Assemble the non-teaching report records produced by the Part 1 notebook.

The public builders accept the notebook namespace directly, so the generated
notebook can replace long bookkeeping cells with one call using ``globals()``.
Scientific calculations stay in their teaching cells; this module only derives
the existing summary values and maps already-calculated objects into the saved
report records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
import os
from typing import Any

import pandas as pd

from .artifacts import sha256_file
from .nci_validation import nci_force_check_record
from .results_summary import build_results_summary
from .run_output import WaterRunManifestInput, WaterRunResults


_RESULT_SUMMARY_NAMES = (
    "ADSORBATES",
    "DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON",
    "DOMAIN_PLANNED_ATOM_COUNTS",
    "FULL_SERIAL_BATCH_TOLERANCE_EV",
    "INFLIGHT_ACTIVE_SYSTEMS",
    "INFLIGHT_NVE_STEPS",
    "INFLIGHT_NVT_STEPS",
    "INFLIGHT_SYSTEMS",
    "PLANNED_CAMPAIGN_SYSTEMS_TOTAL",
    "RESIDUAL_SERIAL_BATCH_TOLERANCE_EV",
    "adsorption_structures",
    "comparisons",
    "counts",
    "crossover",
    "domain_charge_sum",
    "domain_energy_ev",
    "domain_fmax_ev_a",
    "domain_live_api_passed",
    "domain_result",
    "domain_view",
    "full_pipeline_agreement_error",
    "harmonic_comparison_reported",
    "harmonic_frequency_mae_cm1",
    "harmonic_validation",
    "inflight",
    "inflight_nve_counts_correct",
    "inflight_nvt_counts_correct",
    "inflight_sampler",
    "nci_curves",
    "nci_graph_index",
    "nci_metrics",
    "serial_batch_error",
    "sevennet_max_edge_vector_mapping_difference_A",
    "sevennet_max_force_eV_A",
    "sevennet_repeat_max_energy_difference_eV_per_atom",
    "sevennet_repeat_max_force_difference_eV_A",
    "unique_ids",
)

_WATER_RUN_RESULT_NAMES = (
    "ablation_mae",
    "adsorption_forces",
    "adsorption_results",
    "cold_warm",
    "comparisons",
    "crossover",
    "diagnostic_table",
    "dimer_table",
    "domain_molecule_charges",
    "domain_molecule_charge_summary",
    "domain_live_summary",
    "harmonic_comparison_table",
    "harmonic_convergence_table",
    "harmonic_fd_table",
    "harmonic_validation_table",
    "inflight_summary",
    "integrity_table",
    "layout_result",
    "metrics",
    "mode_map_table",
    "nci_curves",
    "nci_member_curves",
    "nci_metrics",
    "reference_metrics",
    "sevennet_graph_mapping",
    "sevennet_numerical_agreement",
    "spectra",
    "topology_timelines",
)

_MANIFEST_RUN_DETAIL_NAMES = (
    "DEFAULT_DATA_DIR",
    "DEVICE",
    "D3_PARAMETER_SHA256",
    "MODEL_CHECKPOINT",
    "NCI_CHECKPOINTS",
    "NCI_DATA_FILE",
    "PART_DIR",
    "REFERENCE_ROOT",
    "RUN_ID",
    "SEVENNET_CHECKPOINT_DOI",
    "SEVENNET_CHECKPOINT_URL",
    "SEVENNET_MODALITY",
    "SEVENNET_REFERENCE_METHOD",
    "aimnet_checkpoint_identities",
    "checkpoint_is_override",
    "domain_view",
    "experimental_artifact_id",
    "experimental_data_sha256",
    "harmonic_archive_path",
    "harmonic_archive_sha256",
    "installed_pins",
    "model_card",
    "reference_dirs",
    "sevennet_checkpoint_sha256",
    "torch",
)

_MANIFEST_MODEL_SETTING_NAMES = (
    "COMPILED_EAGER_CHARGE_TOLERANCE_E",
    "COMPILED_EAGER_ENERGY_TOLERANCE_EV",
    "COMPILED_EAGER_FORCE_TOLERANCE_EV_A",
    "COMPILED_REPEAT_CHARGE_TOLERANCE_E",
    "COMPILED_REPEAT_ENERGY_TOLERANCE_EV",
    "COMPILED_REPEAT_FORCE_TOLERANCE_EV_A",
    "COMPONENT_CLOSURE_TOLERANCE_EV",
    "D3_CUTOFF_A",
    "D3_REFERENCE_CUTOFF_BOHR",
    "D3_REFERENCE_SMOOTHING_FRACTION",
    "FULL_SERIAL_BATCH_TOLERANCE_EV",
    "NCI_VALIDATION",
    "PBE_D3_BJ_A1",
    "PBE_D3_BJ_A2_BOHR",
    "PBE_D3_BJ_S6",
    "PBE_D3_BJ_S8",
    "RESIDUAL_SERIAL_BATCH_TOLERANCE_EV",
    "SEVENNET_MODALITY",
    "SEVENNET_MODEL_NAME",
    "SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM",
    "SEVENNET_REPEAT_FORCE_TOL_EV_A",
    "SURFACE_D3_CUTOFF_A",
    "d3_params",
    "nci_curves",
    "nci_graph_index",
)

_MANIFEST_WORKFLOW_SETTING_NAMES = (
    "COVALENT_OH_CUTOFF_A",
    "DOMAIN_CONSTRUCTION_DENSITY_G_CM3",
    "DOMAIN_HALO_SKIN_A",
    "DOMAIN_METHODOLOGY",
    "DOMAIN_PACKMOL_PRECISION_A",
    "DOMAIN_PACKMOL_SEED",
    "DOMAIN_PACKMOL_TOLERANCE_A",
    "DOMAIN_LIVE_MOLECULES_PER_SPECIES",
    "DT_FS",
    "ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM",
    "HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E",
    "HARMONIC_DISPLACEMENT_STEPS_BOHR",
    "HARMONIC_FIRE_INITIAL_DT",
    "HARMONIC_FMAX_EV_A",
    "HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1",
    "HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX",
    "HARMONIC_IMAGINARY_FLOOR_CM1",
    "HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL",
    "HARMONIC_INTENSITY_STEP_REL_TOLERANCE",
    "HARMONIC_MODE_OVERLAP_MIN",
    "HARMONIC_SELECTED_STEP_BOHR",
    "HBOND_ANGLE_CUTOFF_DEG",
    "HBOND_H_ACCEPTOR_CUTOFF_A",
    "HBOND_OO_CUTOFF_A",
    "H_TO_D_COARSE_MASS_PATH_STEPS",
    "H_TO_D_DEGENERACY_TOLERANCE_CM1",
    "H_TO_D_FINE_MASS_PATH_STEPS",
    "INFLIGHT_ACTIVE_SYSTEMS",
    "INFLIGHT_NVE_STEPS",
    "INFLIGHT_NVT_STEPS",
    "INFLIGHT_SYSTEMS",
    "IR_CAPTURE_CHARGE_TOLERANCE_E",
    "IR_CHARGE_NEUTRALITY_TOLERANCE_E",
    "IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM",
    "IR_FIRE_INITIAL_DT",
    "IR_INITIAL_VELOCITY_RANDOM_SEEDS",
    "IR_NVT_FRICTION_PER_FS",
    "IR_NVT_RANDOM_SEED",
    "IR_PRODUCTION_STATUS",
    "IR_WARMUP_STATUS",
    "IR_WELCH_OVERLAP_FRACTION",
    "IR_WELCH_SEGMENT_TIME_FS",
    "MASS_ONLY_CHARGE_TOLERANCE_E",
    "MASS_ONLY_ENERGY_TOLERANCE_EV",
    "MASS_ONLY_FORCE_TOLERANCE_EV_A",
    "MASS_ONLY_POSITION_ATOL_A",
    "MASS_ONLY_POSITION_RTOL",
    "NEIGHBOR_SKIN_A",
    "OH_REGION_WINDOWS_CM1",
    "OXYGEN_CONNECTIVITY_CUTOFF_A",
    "PAIR_TEMPERATURE_RELATIVE_TOLERANCE",
    "PME_ACCURACY",
    "PME_ALPHA_A_INV",
    "PME_MESH_DIMENSIONS",
    "PME_MESH_SAFETY_FACTOR",
    "PME_MESH_SPACING_A",
    "PME_REALSPACE_CUTOFF_A",
    "PRODUCTION_STEPS",
    "TEMPERATURE_K",
    "WARMUP_STEPS",
    "domain_config",
    "domain_cutoff_a",
)

_MANIFEST_CHECK_NAMES = (
    "ADSORBATES",
    "COMPOSITION_FD_ENERGY_ROUTE",
    "DOMAIN_CHARGE_SUM_TOLERANCE_E",
    "FD_STEP_A",
    "INFLIGHT_SYSTEMS",
    "NCI_VALIDATION",
    "adsorption_structures",
    "agreement_errors",
    "analytic_coulomb_errors",
    "cluster_dft_comparison_valid",
    "cluster_intact",
    "compiled_ir_eager_agreement",
    "compiled_ir_repeat_agreement",
    "comparisons",
    "component_closure_error",
    "counts",
    "domain_charge_sum",
    "domain_elapsed_s",
    "domain_energy_ev",
    "domain_fmax_ev_a",
    "domain_peak_memory_gb",
    "domain_result",
    "domain_view",
    "energy_within_advisory",
    "fd_force",
    "fd_force_error",
    "full_pipeline_agreement_error",
    "harmonic_comparison_reported",
    "harmonic_d_convergence",
    "harmonic_fmax_eV_A",
    "harmonic_frequency_mae_cm1",
    "harmonic_h_convergence",
    "harmonic_validation",
    "inflight_nve_counts_correct",
    "inflight_nvt_counts_correct",
    "model_force",
    "nci_charge_conservation_max_abs_e",
    "nci_component_sum_max_abs_eV",
    "nci_force_check",
    "nci_graph_order_max_abs_eV",
    "nci_metrics",
    "official_fd_force_error",
    "official_force",
    "selected_harmonic_estimate",
    "serial_batch_error",
    "sevennet_graph_mapping_passed",
    "sevennet_max_force_eV_A",
    "sevennet_repeat_max_energy_difference_eV_per_atom",
    "sevennet_repeat_max_force_difference_eV_A",
    "stage_counts",
    "unique_ids",
)

_REPORT_REQUIRED_NAMES = tuple(
    sorted(
        set(
            _RESULT_SUMMARY_NAMES
            + _WATER_RUN_RESULT_NAMES
            + _MANIFEST_RUN_DETAIL_NAMES
            + _MANIFEST_MODEL_SETTING_NAMES
            + _MANIFEST_WORKFLOW_SETTING_NAMES
            + _MANIFEST_CHECK_NAMES
        )
    )
)
_UNSET = object()


class MissingNotebookVariablesError(ValueError):
    """Raised when a report builder cannot find all prerequisite cell outputs."""

    def __init__(self, context: str, missing_names: tuple[str, ...]) -> None:
        self.context = context
        self.missing_names = tuple(sorted(missing_names))
        names = ", ".join(self.missing_names)
        super().__init__(f"{context} is missing required notebook variables: {names}")


@dataclass(frozen=True)
class Part1ManifestSections:
    """The four authoring records that become the saved run manifest."""

    run_details: dict[str, dict[str, Any]]
    model_settings: dict[str, dict[str, Any]]
    workflow_settings: dict[str, dict[str, Any]]
    checks: dict[str, dict[str, Any]]

    def as_manifest_input(self) -> WaterRunManifestInput:
        """Validate and flatten the sections for ``save_water_run_outputs``."""

        return WaterRunManifestInput.from_sections(
            run_details=self.run_details,
            settings=self.model_settings | self.workflow_settings,
            checks=self.checks,
        )


@dataclass(frozen=True)
class Part1NotebookReport:
    """Complete bookkeeping output assembled after the teaching calculations."""

    results_summary: pd.DataFrame
    not_reported_count: int
    water_run_results: WaterRunResults
    manifest_input: WaterRunManifestInput


def _require_values(
    namespace: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(namespace, Mapping):
        raise TypeError("namespace must implement collections.abc.Mapping")
    missing_names = tuple(name for name in names if name not in namespace)
    if missing_names:
        raise MissingNotebookVariablesError(context, missing_names)
    return {name: namespace[name] for name in names}


def build_part1_results_summary(
    namespace: Mapping[str, Any],
) -> tuple[pd.DataFrame, int]:
    """Derive the existing learner-facing summary from calculated cell values."""

    values = _require_values(
        namespace,
        _RESULT_SUMMARY_NAMES,
        context="Part 1 results summary",
    )

    batch_results_match = bool(
        values["serial_batch_error"] < values["RESIDUAL_SERIAL_BATCH_TOLERANCE_EV"]
        and values["full_pipeline_agreement_error"]
        < values["FULL_SERIAL_BATCH_TOLERANCE_EV"]
    )
    crossover = values["crossover"]
    cpu_gpu_throughput = crossover.pivot(
        index="batch_size",
        columns="route",
        values="structures_per_s",
    ).sort_index()
    gpu_faster_batches = cpu_gpu_throughput.index[
        cpu_gpu_throughput["GPU"] > cpu_gpu_throughput["CPU"]
    ]
    cpu_gpu_crossover_batch_size = (
        int(gpu_faster_batches[0]) if len(gpu_faster_batches) else None
    )
    cpu_gpu_largest_batch_size = int(cpu_gpu_throughput.index[-1])
    cpu_gpu_largest_batch_speedup = float(
        cpu_gpu_throughput.loc[cpu_gpu_largest_batch_size, "GPU"]
        / cpu_gpu_throughput.loc[cpu_gpu_largest_batch_size, "CPU"]
    )
    cpu_gpu_max_energy_difference_eV = float(
        crossover["max_abs_energy_difference"].max()
    )

    comparisons = values["comparisons"]
    monomer_shown = bool(comparisons.loc["H2O_over_D2O_centroid", "reported"])
    cluster_rows = comparisons.index != "H2O_over_D2O_centroid"
    cluster_shown = bool(comparisons.loc[cluster_rows, "reported"].all())
    harmonic_failed_checks = tuple(
        name for name, passed in values["harmonic_validation"].items() if not passed
    )
    cluster_not_reported_reasons = comparisons.loc[
        cluster_rows & ~comparisons["reported"],
        "status",
    ].tolist()
    inflight_sampler = values["inflight_sampler"]
    inflight = values["inflight"]
    unique_ids = values["unique_ids"]
    counts = values["counts"]
    domain_result = values["domain_result"]
    domain_view = values["domain_view"]
    nci_metrics = values["nci_metrics"]

    return build_results_summary(
        batch_results_match=batch_results_match,
        serial_batch_error_eV=values["serial_batch_error"],
        full_pipeline_agreement_error_eV=values["full_pipeline_agreement_error"],
        cpu_gpu_crossover_batch_size=cpu_gpu_crossover_batch_size,
        cpu_gpu_largest_batch_size=cpu_gpu_largest_batch_size,
        cpu_gpu_largest_batch_speedup=cpu_gpu_largest_batch_speedup,
        cpu_gpu_max_energy_difference_eV=cpu_gpu_max_energy_difference_eV,
        sevennet_status="PASS",
        sevennet_structure_count=len(values["adsorption_structures"]),
        sevennet_batch_count=2,
        sevennet_molecule_count=len(values["ADSORBATES"]),
        sevennet_max_edge_vector_mapping_difference_A=values[
            "sevennet_max_edge_vector_mapping_difference_A"
        ],
        sevennet_repeat_max_energy_difference_eV_per_atom=values[
            "sevennet_repeat_max_energy_difference_eV_per_atom"
        ],
        sevennet_repeat_max_force_difference_eV_A=values[
            "sevennet_repeat_max_force_difference_eV_A"
        ],
        sevennet_max_force_eV_A=values["sevennet_max_force_eV_A"],
        nci_geometry_count=len(values["nci_curves"]),
        nci_graph_count=len(values["nci_graph_index"]),
        nci_max_mae_vs_dft_d3_kcal_mol=float(nci_metrics["complete vs DFT-D3"].max()),
        nci_max_mae_vs_ccsd_t_cbs_kcal_mol=float(nci_metrics["complete vs CC"].max()),
        harmonic_comparison_reported=values["harmonic_comparison_reported"],
        harmonic_frequency_mae_cm1=values["harmonic_frequency_mae_cm1"],
        harmonic_failed_checks=harmonic_failed_checks,
        inflight_queue_complete=bool(
            inflight_sampler.exhausted
            and inflight.done
            and unique_ids.numel() == values["INFLIGHT_SYSTEMS"]
            and not (counts > 1).any().item()
            and values["inflight_nvt_counts_correct"]
            and values["inflight_nve_counts_correct"]
        ),
        inflight_system_count=values["INFLIGHT_SYSTEMS"],
        inflight_active_system_count=values["INFLIGHT_ACTIVE_SYSTEMS"],
        inflight_nvt_steps=values["INFLIGHT_NVT_STEPS"],
        inflight_nve_steps=values["INFLIGHT_NVE_STEPS"],
        domain_live_api_passed=values["domain_live_api_passed"],
        domain_live_world_size=1,
        domain_live_spatially_decomposed=False,
        domain_live_atom_count=int(domain_result.num_nodes),
        domain_live_energy_per_atom_eV=(
            values["domain_energy_ev"] / domain_result.num_nodes
        ),
        domain_live_max_force_eV_A=values["domain_fmax_ev_a"],
        domain_live_charge_sum_e=values["domain_charge_sum"],
        domain_results_available=domain_view.available,
        domain_results_unavailable_reason=(
            None if domain_view.available else domain_view.reason
        ),
        domain_successful_cases=domain_view.successful_case_count,
        domain_failed_cases=domain_view.failed_case_count,
        domain_planned_max_atom_count=max(values["DOMAIN_PLANNED_ATOM_COUNTS"]),
        domain_measured_max_atom_count=domain_view.measured_max_atom_count,
        campaign_available=False,
        campaign_unavailable_reason=values[
            "DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON"
        ],
        campaign_successes=0,
        campaign_failures=0,
        campaign_systems_total=values["PLANNED_CAMPAIGN_SYSTEMS_TOTAL"],
        monomer_shown=monomer_shown,
        monomer_status=comparisons.loc[
            "H2O_over_D2O_centroid",
            "status",
        ],
        cluster_shown=cluster_shown,
        cluster_not_reported_reasons=cluster_not_reported_reasons,
    )


def build_part1_water_run_results(
    namespace: Mapping[str, Any],
    *,
    results_summary: Any = _UNSET,
) -> WaterRunResults:
    """Collect the existing saved-result fields from a notebook namespace.

    ``results_summary`` may be supplied by
    :func:`build_part1_notebook_report`; when omitted, the function reads the
    earlier ``results_summary`` cell output from ``namespace``.
    """

    required_names = _WATER_RUN_RESULT_NAMES
    if results_summary is _UNSET:
        required_names += ("results_summary",)
    values = _require_values(
        namespace,
        required_names,
        context="Part 1 WaterRunResults",
    )
    if results_summary is _UNSET:
        results_summary = values["results_summary"]

    return WaterRunResults(
        diagnostics=values["diagnostic_table"],
        spectrum_metrics=values["metrics"],
        topology_summary=values["integrity_table"],
        comparisons=values["comparisons"],
        dft_comparison=values["reference_metrics"],
        h_to_d_mode_map=values["mode_map_table"],
        harmonic_displacements=values["harmonic_fd_table"],
        harmonic_convergence=values["harmonic_convergence_table"],
        harmonic_checks=values["harmonic_validation_table"],
        harmonic_comparison=values["harmonic_comparison_table"],
        nci_interaction_curves=values["nci_curves"],
        nci_interaction_metrics=values["nci_metrics"],
        nci_ensemble_curves=values["nci_member_curves"],
        dimer_ablation=values["dimer_table"],
        dimer_ablation_mae=values["ablation_mae"],
        adsorption_results=values["adsorption_results"],
        adsorption_forces=values["adsorption_forces"],
        sevennet_graph_mapping=values["sevennet_graph_mapping"],
        sevennet_numerical_agreement=values["sevennet_numerical_agreement"],
        first_warm_calls=values["cold_warm"],
        cpu_gpu_crossover=values["crossover"],
        inflight_summary=values["inflight_summary"],
        domain_molecule_charges=values["domain_molecule_charges"],
        domain_molecule_charge_summary=values["domain_molecule_charge_summary"],
        domain_live_summary=values["domain_live_summary"],
        results_summary=results_summary,
        batch_layout_timings=values["layout_result"]["timings"],
        topology_timelines=values["topology_timelines"],
        spectra=values["spectra"],
    )


def _build_manifest_run_details(
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    domain_view = values["domain_view"]
    part_dir = values["PART_DIR"]
    reference_root = values["REFERENCE_ROOT"]
    harmonic_archive_path = values["harmonic_archive_path"]
    torch = values["torch"]

    return {
        "runtime": {
            "run_id": values["RUN_ID"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(values["DEVICE"]),
            "torch": torch.__version__,
            "aimnet": metadata.version("aimnet"),
            "sevennet": metadata.version("sevenn"),
            "toolkit_core_commit": values["installed_pins"]["Core"],
            "toolkit_ops_commit": values["installed_pins"]["Ops"],
        },
        "aimnet_checkpoint": {
            "checkpoint_source": values["MODEL_CHECKPOINT"],
            "checkpoint_sha256": values["model_card"]["checkpoint_sha256"],
            "checkpoint_override": values["checkpoint_is_override"],
        },
        "nci_data": {
            "nci_checkpoints": values["NCI_CHECKPOINTS"],
            "aimnet_checkpoint_identities": values[
                "aimnet_checkpoint_identities"
            ],
            "nci_subset_sha256": sha256_file(values["NCI_DATA_FILE"]),
        },
        "sevennet_checkpoint": {
            "sevennet_checkpoint_source": values["SEVENNET_CHECKPOINT_URL"],
            "sevennet_checkpoint_sha256": values["sevennet_checkpoint_sha256"],
            "sevennet_checkpoint_doi": values["SEVENNET_CHECKPOINT_DOI"],
            "sevennet_task": values["SEVENNET_MODALITY"],
            "sevennet_reference_method": values["SEVENNET_REFERENCE_METHOD"],
        },
        "source_files": {
            "adsorption_structure_manifest_sha256": sha256_file(
                values["DEFAULT_DATA_DIR"] / "manifest.json"
            ),
            "d3_parameter_file_sha256": values["D3_PARAMETER_SHA256"],
            "notebook_sha256": sha256_file(part_dir / "alchemi-water-ir.ipynb"),
            "dimer_reference_manifest_sha256": sha256_file(
                reference_root / "water_dimer_b97_3c" / "manifest.json"
            ),
        },
        "ir_references": {
            "harmonic_reference_sha256s": {
                label: sha256_file(reference_root / directory / "manifest.json")
                for label, directory in values["reference_dirs"].items()
            },
            "aimnet_harmonic_archive": {
                "path": harmonic_archive_path.name,
                "sha256": values["harmonic_archive_sha256"],
            },
            "experimental_reference_bundle": {
                "artifact_id": values["experimental_artifact_id"],
                "manifest_sha256": sha256_file(
                    part_dir
                    / "reference"
                    / "experimental_water_fundamentals"
                    / "manifest.json"
                ),
                "data_sha256": values["experimental_data_sha256"],
                "checksum_index_sha256": sha256_file(
                    part_dir
                    / "reference"
                    / "experimental_water_fundamentals"
                    / "SHA256SUMS"
                ),
            },
        },
        "distributed_run": {
            "pipeline_campaign_bundle": None,
            "domain_decomposition_bundle": domain_view.bundle_record,
        },
    }


def _build_manifest_model_settings(
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "nci_model": {
            "model": (
                "checkpoint base (E_NN - E_Coulomb^SR) + predicted-charge "
                "full Coulomb + D3(BJ)"
            ),
            "nci_graphs": len(values["nci_graph_index"]),
            "nci_interaction_geometries": len(values["nci_curves"]),
            "nci_reference_levels": [
                "ωB97M-D3(BJ)/def2-TZVPPD",
                "CCSD(T)/CBS interaction energies",
            ],
            "nci_validation": values["NCI_VALIDATION"].as_record(),
        },
        "sevennet_adapter": {
            "custom_adapter_model": values["SEVENNET_MODEL_NAME"],
            "custom_adapter_task": values["SEVENNET_MODALITY"],
            "custom_adapter_scope": (
                "fixed-geometry 2D-periodic Cu(111) and finite molecular "
                "energy/force single points"
            ),
            "custom_adapter_precision": "float32",
            "custom_adapter_compile": False,
            "custom_adapter_energy_repeat_tolerance_eV_per_atom": values[
                "SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM"
            ],
            "custom_adapter_force_repeat_tolerance_eV_A": values[
                "SEVENNET_REPEAT_FORCE_TOL_EV_A"
            ],
            "custom_adapter_geometry_status": (
                "ASE-generated initial placements; not model-relaxed"
            ),
        },
        "surface_dispersion": {
            "surface_d3_cutoff_A": values["SURFACE_D3_CUTOFF_A"],
            "surface_d3_cutoff_bohr": values["D3_REFERENCE_CUTOFF_BOHR"],
            "surface_d3_smoothing_fraction": values["D3_REFERENCE_SMOOTHING_FRACTION"],
            "surface_d3_parameters": {
                "a1": values["PBE_D3_BJ_A1"],
                "a2_bohr": values["PBE_D3_BJ_A2_BOHR"],
                "s6": values["PBE_D3_BJ_S6"],
                "s8": values["PBE_D3_BJ_S8"],
            },
        },
        "model_composition": {
            "electrostatics": "simple nonperiodic all-pairs 1/r; no cutoff",
            "d3_cutoff_A": values["D3_CUTOFF_A"],
            "d3_parameters": values["d3_params"],
            "compile_mode": ("default Torch compile on the fixed 42-atom IR batch"),
            "residual_serial_batch_tolerance_eV": values[
                "RESIDUAL_SERIAL_BATCH_TOLERANCE_EV"
            ],
            "full_serial_batch_tolerance_eV": values["FULL_SERIAL_BATCH_TOLERANCE_EV"],
            "component_closure_tolerance_eV": values["COMPONENT_CLOSURE_TOLERANCE_EV"],
            "compiled_eager_energy_tolerance_eV": values[
                "COMPILED_EAGER_ENERGY_TOLERANCE_EV"
            ],
            "compiled_eager_force_tolerance_eV_A": values[
                "COMPILED_EAGER_FORCE_TOLERANCE_EV_A"
            ],
            "compiled_eager_charge_tolerance_e": values[
                "COMPILED_EAGER_CHARGE_TOLERANCE_E"
            ],
            "compiled_repeat_energy_tolerance_eV": values[
                "COMPILED_REPEAT_ENERGY_TOLERANCE_EV"
            ],
            "compiled_repeat_force_tolerance_eV_A": values[
                "COMPILED_REPEAT_FORCE_TOLERANCE_EV_A"
            ],
            "compiled_repeat_charge_tolerance_e": values[
                "COMPILED_REPEAT_CHARGE_TOLERANCE_E"
            ],
        },
    }


def _build_manifest_workflow_settings(
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    methodology = values["DOMAIN_METHODOLOGY"]
    return {
        "dynamics": {
            "neighbor_skin_A": values["NEIGHBOR_SKIN_A"],
            "fire_initial_dt": values["IR_FIRE_INITIAL_DT"],
            "temperature_K": values["TEMPERATURE_K"],
            "dt_fs": values["DT_FS"],
            "warmup_steps": values["WARMUP_STEPS"],
            "production_steps": values["PRODUCTION_STEPS"],
            "warmup_status": values["IR_WARMUP_STATUS"],
            "production_status": values["IR_PRODUCTION_STATUS"],
            "initial_velocity_random_seeds": values["IR_INITIAL_VELOCITY_RANDOM_SEEDS"],
            "nvt_friction_per_fs": values["IR_NVT_FRICTION_PER_FS"],
            "nvt_random_seed": values["IR_NVT_RANDOM_SEED"],
            "capture_charge_tolerance_e": values["IR_CAPTURE_CHARGE_TOLERANCE_E"],
            "charge_neutrality_tolerance_e": values["IR_CHARGE_NEUTRALITY_TOLERANCE_E"],
            "dipole_origin_tolerance_e_A": values[
                "IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM"
            ],
            "mass_only_position_rtol": values["MASS_ONLY_POSITION_RTOL"],
            "mass_only_position_atol_A": values["MASS_ONLY_POSITION_ATOL_A"],
            "mass_only_energy_tolerance_eV": values["MASS_ONLY_ENERGY_TOLERANCE_EV"],
            "mass_only_force_tolerance_eV_A": values["MASS_ONLY_FORCE_TOLERANCE_EV_A"],
            "mass_only_charge_tolerance_e": values["MASS_ONLY_CHARGE_TOLERANCE_E"],
        },
        "spectrum": {
            "spectrum_segment_time_fs": values["IR_WELCH_SEGMENT_TIME_FS"],
            "spectrum_overlap": values["IR_WELCH_OVERLAP_FRACTION"],
            "spectrum_windows_cm1": values["OH_REGION_WINDOWS_CM1"],
        },
        "isotope_analysis": {
            "pair_temperature_relative_tolerance": values[
                "PAIR_TEMPERATURE_RELATIVE_TOLERANCE"
            ],
            "h_to_d_coarse_mass_path_steps": values["H_TO_D_COARSE_MASS_PATH_STEPS"],
            "h_to_d_fine_mass_path_steps": values["H_TO_D_FINE_MASS_PATH_STEPS"],
            "h_to_d_degeneracy_tolerance_cm1": values[
                "H_TO_D_DEGENERACY_TOLERANCE_CM1"
            ],
        },
        "topology": {
            "oxygen_connectivity_cutoff_A": values["OXYGEN_CONNECTIVITY_CUTOFF_A"],
            "covalent_OH_cutoff_A": values["COVALENT_OH_CUTOFF_A"],
            "hbond_H_acceptor_cutoff_A": values["HBOND_H_ACCEPTOR_CUTOFF_A"],
            "hbond_OO_cutoff_A": values["HBOND_OO_CUTOFF_A"],
            "hbond_angle_cutoff_deg": values["HBOND_ANGLE_CUTOFF_DEG"],
            "energy_excursion_advisory_meV_atom": values[
                "ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM"
            ],
        },
        "harmonic": {
            "harmonic_fmax_eV_A": values["HARMONIC_FMAX_EV_A"],
            "harmonic_fire_initial_dt": values["HARMONIC_FIRE_INITIAL_DT"],
            "harmonic_displacement_steps_bohr": values[
                "HARMONIC_DISPLACEMENT_STEPS_BOHR"
            ].tolist(),
            "harmonic_selected_step_bohr": values["HARMONIC_SELECTED_STEP_BOHR"],
            "harmonic_frequency_step_tolerance_cm1": values[
                "HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1"
            ],
            "harmonic_intensity_step_abs_tolerance_km_mol": values[
                "HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL"
            ],
            "harmonic_intensity_step_relative_tolerance": values[
                "HARMONIC_INTENSITY_STEP_REL_TOLERANCE"
            ],
            "harmonic_mode_overlap_min": values["HARMONIC_MODE_OVERLAP_MIN"],
            "harmonic_hessian_antisymmetry_relative_max": values[
                "HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX"
            ],
            "harmonic_charge_neutrality_tolerance_e": values[
                "HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E"
            ],
            "harmonic_imaginary_floor_cm1": values["HARMONIC_IMAGINARY_FLOOR_CM1"],
        },
        "scaling": {
            "domain_methodology": {
                "source": methodology.as_record(),
                "resolved_values": {
                    **methodology.resolved_values(json_compatible=True),
                    "pme_alpha_a_inv": values["PME_ALPHA_A_INV"],
                    "pme_mesh_dimensions": list(values["PME_MESH_DIMENSIONS"]),
                    "pme_mesh_spacing_a": list(values["PME_MESH_SPACING_A"]),
                    "derived_domain_cutoff_a": values["domain_cutoff_a"],
                },
            },
            "inflight_systems": values["INFLIGHT_SYSTEMS"],
            "inflight_active_systems": values["INFLIGHT_ACTIVE_SYSTEMS"],
            "inflight_nvt_steps": values["INFLIGHT_NVT_STEPS"],
            "inflight_nve_steps": values["INFLIGHT_NVE_STEPS"],
            "domain_live_molecules_per_species": values[
                "DOMAIN_LIVE_MOLECULES_PER_SPECIES"
            ],
            "domain_construction_density_g_cm3": values[
                "DOMAIN_CONSTRUCTION_DENSITY_G_CM3"
            ],
            "domain_packmol_tolerance_a": values["DOMAIN_PACKMOL_TOLERANCE_A"],
            "domain_packmol_precision_a": values["DOMAIN_PACKMOL_PRECISION_A"],
            "domain_packmol_seed": values["DOMAIN_PACKMOL_SEED"],
            "domain_pme_realspace_cutoff_a": values["PME_REALSPACE_CUTOFF_A"],
            "domain_pme_mesh_safety_factor": values["PME_MESH_SAFETY_FACTOR"],
            "domain_pme_alpha_a_inv": values["PME_ALPHA_A_INV"],
            "domain_pme_mesh_dimensions": values["PME_MESH_DIMENSIONS"],
            "domain_pme_mesh_spacing_a": values["PME_MESH_SPACING_A"],
            "domain_pme_accuracy": values["PME_ACCURACY"],
            "domain_ewald_reference_accuracy": (methodology.ewald_reference_accuracy),
            "domain_halo_skin_a": values["DOMAIN_HALO_SKIN_A"],
            "domain_model_cutoff_a": values["domain_cutoff_a"],
            "domain_compile": values["domain_config"].compile,
        },
    }


def _build_manifest_checks(
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    nci_metrics = values["nci_metrics"]
    harmonic_h_convergence = values["harmonic_h_convergence"]
    harmonic_d_convergence = values["harmonic_d_convergence"]
    domain_view = values["domain_view"]
    domain_charge_sum = values["domain_charge_sum"]
    counts = values["counts"]

    return {
        "model_composition": {
            "residual_serial_batch_max_abs_eV": values["serial_batch_error"],
            "full_serial_batch_max_abs_eV": values["full_pipeline_agreement_error"],
            "component_closure_max_abs_eV": values["component_closure_error"],
            "official_calculator_agreement": values["agreement_errors"],
            "analytic_coulomb": values["analytic_coulomb_errors"],
            "compiled_ir_eager_agreement": values["compiled_ir_eager_agreement"],
            "compiled_ir_repeat_agreement": values["compiled_ir_repeat_agreement"],
            "finite_difference_force_energy_route": values[
                "COMPOSITION_FD_ENERGY_ROUTE"
            ],
            "finite_difference_force_step_A": values["FD_STEP_A"],
            "finite_difference_force_reference_eV_A": values["fd_force"],
            "finite_difference_force_official_analytic_eV_A": values["official_force"],
            "finite_difference_force_official_abs_error_eV_A": values[
                "official_fd_force_error"
            ],
            "finite_difference_force_pipeline_eV_A": values["model_force"],
            "finite_difference_force_pipeline_abs_error_eV_A": values["fd_force_error"],
        },
        "nci": {
            "graph_charge_conservation_max_abs_e": values[
                "nci_charge_conservation_max_abs_e"
            ],
            "component_sum_max_abs_eV": values["nci_component_sum_max_abs_eV"],
            "graph_order_max_abs_eV": values["nci_graph_order_max_abs_eV"],
            "nci_complete_max_MAE_vs_DFT_D3_kcal_mol": float(
                nci_metrics["complete vs DFT-D3"].max()
            ),
            "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol": float(
                nci_metrics["complete vs CC"].max()
            ),
            "nci_force_check": nci_force_check_record(
                values["nci_force_check"],
                values["NCI_VALIDATION"],
            ),
        },
        "sevennet": {
            "sevennet_adapter": {
                "graph_mapping_passed": values["sevennet_graph_mapping_passed"],
                "structures": int(len(values["adsorption_structures"])),
                "batches": 2,
                "finite_outputs": True,
                "numerical_max_abs_energy_eV_per_atom": values[
                    "sevennet_repeat_max_energy_difference_eV_per_atom"
                ],
                "numerical_max_abs_forces_eV_A": values[
                    "sevennet_repeat_max_force_difference_eV_A"
                ],
                "max_combined_fmax_eV_A": values["sevennet_max_force_eV_A"],
                "geometry_status": (
                    "ASE-generated initial placements; not model-relaxed"
                ),
                "molecules": list(values["ADSORBATES"]),
                "periodic_pbc": [True, True, False],
            },
        },
        "dynamics": {
            "cluster_integrity_passed": values["cluster_intact"],
            "initial_ring_persisted_all_frames": values["cluster_dft_comparison_valid"],
            "energy_excursion_within_advisory": values["energy_within_advisory"],
            "reported_comparisons": values["comparisons"]["reported"].to_dict(),
            "fused_stage_route_counts": values["stage_counts"],
        },
        "harmonic": {
            "harmonic_checks": values["harmonic_validation"],
            "harmonic_comparison_reported": values["harmonic_comparison_reported"],
            "harmonic_final_fmax_eV_A": values["harmonic_fmax_eV_A"],
            "harmonic_frequency_MAE_vs_B97_3c_cm1": values[
                "harmonic_frequency_mae_cm1"
            ],
            "harmonic_selected_Hessian_antisymmetry_relative": values[
                "selected_harmonic_estimate"
            ].hessian.max_relative_antisymmetry,
            "harmonic_final_frequency_step_change_cm1": {
                "H2O": float(harmonic_h_convergence.frequency_max_abs_change_cm1[-1]),
                "D2O": float(harmonic_d_convergence.frequency_max_abs_change_cm1[-1]),
            },
            "harmonic_final_intensity_step_change_km_mol": {
                "H2O": float(
                    harmonic_h_convergence.ir_intensity_max_abs_change_km_mol[-1]
                ),
                "D2O": float(
                    harmonic_d_convergence.ir_intensity_max_abs_change_km_mol[-1]
                ),
            },
        },
        "scaling": {
            "inflight_completed_systems": values["INFLIGHT_SYSTEMS"],
            "inflight_unique_system_ids": int(values["unique_ids"].numel()),
            "inflight_duplicate_system_ids": int((counts > 1).sum()),
            "inflight_nvt_counts_correct": values["inflight_nvt_counts_correct"],
            "inflight_nve_counts_correct": values["inflight_nve_counts_correct"],
            "domain_world_size": 1,
            "domain_spatially_decomposed": False,
            "domain_atom_count": int(values["domain_result"].num_nodes),
            "domain_energy_eV": values["domain_energy_ev"],
            "domain_force_max_eV_A": values["domain_fmax_ev_a"],
            "domain_charge_sum_e": domain_charge_sum,
            "domain_charge_sum_tolerance_e": values["DOMAIN_CHARGE_SUM_TOLERANCE_E"],
            "domain_charge_neutral": (
                abs(domain_charge_sum) <= values["DOMAIN_CHARGE_SUM_TOLERANCE_E"]
            ),
            "domain_elapsed_s": values["domain_elapsed_s"],
            "domain_peak_memory_GB": values["domain_peak_memory_gb"],
            "domain_recorded_results_available": domain_view.available,
            "domain_recorded_successful_cases": (domain_view.successful_case_count),
            "domain_recorded_failed_cases": domain_view.failed_case_count,
            "pipeline_recorded_results_available": False,
        },
    }


def build_part1_manifest_sections(
    namespace: Mapping[str, Any],
) -> Part1ManifestSections:
    """Build the exact run-detail, setting, and check sections from globals."""

    run_values = _require_values(
        namespace,
        _MANIFEST_RUN_DETAIL_NAMES,
        context="Part 1 manifest run details",
    )
    model_values = _require_values(
        namespace,
        _MANIFEST_MODEL_SETTING_NAMES,
        context="Part 1 manifest model settings",
    )
    workflow_values = _require_values(
        namespace,
        _MANIFEST_WORKFLOW_SETTING_NAMES,
        context="Part 1 manifest workflow settings",
    )
    check_values = _require_values(
        namespace,
        _MANIFEST_CHECK_NAMES,
        context="Part 1 manifest checks",
    )
    return Part1ManifestSections(
        run_details=_build_manifest_run_details(run_values),
        model_settings=_build_manifest_model_settings(model_values),
        workflow_settings=_build_manifest_workflow_settings(workflow_values),
        checks=_build_manifest_checks(check_values),
    )


def build_part1_notebook_report(
    namespace: Mapping[str, Any],
) -> Part1NotebookReport:
    """Build all report objects after the Part 1 calculation cells finish.

    In the notebook, call ``report = build_part1_notebook_report(globals())``.
    Missing prerequisites are reported together before any assembly begins.
    """

    _require_values(
        namespace,
        _REPORT_REQUIRED_NAMES,
        context="Part 1 notebook report",
    )
    results_summary, not_reported_count = build_part1_results_summary(namespace)
    water_run_results = build_part1_water_run_results(
        namespace,
        results_summary=results_summary,
    )
    manifest_sections = build_part1_manifest_sections(namespace)
    return Part1NotebookReport(
        results_summary=results_summary,
        not_reported_count=not_reported_count,
        water_run_results=water_run_results,
        manifest_input=manifest_sections.as_manifest_input(),
    )
