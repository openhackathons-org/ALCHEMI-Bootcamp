"""Write the complete set of Part 1 summary tables and its run manifest.

The notebook computes the scientific results and passes them here explicitly.
This module owns only output mechanics: stable filenames, CSV index choices,
assembly of the shared spectrum table, topology-timeline names, and the final
manifest inventory.  It deliberately contains no Toolkit calls and does not
decide whether a scientific check passed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import WATER_RUN_MANIFEST_NAME, write_water_run_manifest


Table = pd.DataFrame | pd.Series
ManifestRecord = Mapping[str, Any]
ManifestSections = Mapping[str, ManifestRecord]


# These section names are only an authoring aid.  ``WaterRunManifestInput``
# flattens them before saving, so the on-disk manifest keeps its established
# run_details / settings / checks shape.  Listing every key here also makes an
# accidental omission fail before any result file is written.
_RUN_DETAIL_SECTION_KEYS = {
    "runtime": (
        "run_id",
        "slurm_job_id",
        "gpu",
        "torch",
        "aimnet",
        "sevennet",
        "toolkit_core_commit",
        "toolkit_ops_commit",
    ),
    "aimnet_checkpoint": (
        "checkpoint_source",
        "checkpoint_sha256",
        "checkpoint_override",
    ),
    "nci_data": (
        "nci_checkpoints",
        "aimnet_checkpoint_identities",
        "nci_subset_sha256",
    ),
    "sevennet_checkpoint": (
        "sevennet_checkpoint_source",
        "sevennet_checkpoint_sha256",
        "sevennet_checkpoint_doi",
        "sevennet_task",
        "sevennet_reference_method",
    ),
    "source_files": (
        "adsorption_structure_manifest_sha256",
        "d3_parameter_file_sha256",
        "notebook_sha256",
        "dimer_reference_manifest_sha256",
    ),
    "ir_references": (
        "harmonic_reference_sha256s",
        "aimnet_harmonic_archive",
        "experimental_reference_bundle",
    ),
    "distributed_run": (
        "pipeline_campaign_bundle",
        "domain_decomposition_bundle",
    ),
}

_SETTING_SECTION_KEYS = {
    "nci_model": (
        "model",
        "nci_graphs",
        "nci_interaction_geometries",
        "nci_reference_levels",
        "nci_validation",
    ),
    "sevennet_adapter": (
        "custom_adapter_model",
        "custom_adapter_task",
        "custom_adapter_scope",
        "custom_adapter_precision",
        "custom_adapter_compile",
        "custom_adapter_energy_repeat_tolerance_eV_per_atom",
        "custom_adapter_force_repeat_tolerance_eV_A",
        "custom_adapter_geometry_status",
    ),
    "surface_dispersion": (
        "surface_d3_cutoff_A",
        "surface_d3_cutoff_bohr",
        "surface_d3_smoothing_fraction",
        "surface_d3_parameters",
    ),
    "model_composition": (
        "electrostatics",
        "d3_cutoff_A",
        "d3_parameters",
        "compile_mode",
        "residual_serial_batch_tolerance_eV",
        "full_serial_batch_tolerance_eV",
        "component_closure_tolerance_eV",
        "compiled_eager_energy_tolerance_eV",
        "compiled_eager_force_tolerance_eV_A",
        "compiled_eager_charge_tolerance_e",
        "compiled_repeat_energy_tolerance_eV",
        "compiled_repeat_force_tolerance_eV_A",
        "compiled_repeat_charge_tolerance_e",
    ),
    "dynamics": (
        "neighbor_skin_A",
        "fire_initial_dt",
        "temperature_K",
        "dt_fs",
        "warmup_steps",
        "production_steps",
        "warmup_status",
        "production_status",
        "initial_velocity_random_seeds",
        "nvt_friction_per_fs",
        "nvt_random_seed",
        "capture_charge_tolerance_e",
        "charge_neutrality_tolerance_e",
        "dipole_origin_tolerance_e_A",
        "mass_only_position_rtol",
        "mass_only_position_atol_A",
        "mass_only_energy_tolerance_eV",
        "mass_only_force_tolerance_eV_A",
        "mass_only_charge_tolerance_e",
    ),
    "spectrum": (
        "spectrum_segment_time_fs",
        "spectrum_overlap",
        "spectrum_windows_cm1",
    ),
    "isotope_analysis": (
        "pair_temperature_relative_tolerance",
        "h_to_d_coarse_mass_path_steps",
        "h_to_d_fine_mass_path_steps",
        "h_to_d_degeneracy_tolerance_cm1",
    ),
    "topology": (
        "oxygen_connectivity_cutoff_A",
        "covalent_OH_cutoff_A",
        "hbond_H_acceptor_cutoff_A",
        "hbond_OO_cutoff_A",
        "hbond_angle_cutoff_deg",
        "energy_excursion_advisory_meV_atom",
    ),
    "harmonic": (
        "harmonic_fmax_eV_A",
        "harmonic_fire_initial_dt",
        "harmonic_displacement_steps_bohr",
        "harmonic_selected_step_bohr",
        "harmonic_frequency_step_tolerance_cm1",
        "harmonic_intensity_step_abs_tolerance_km_mol",
        "harmonic_intensity_step_relative_tolerance",
        "harmonic_mode_overlap_min",
        "harmonic_hessian_antisymmetry_relative_max",
        "harmonic_charge_neutrality_tolerance_e",
        "harmonic_imaginary_floor_cm1",
    ),
    "scaling": (
        "domain_methodology",
        "inflight_systems",
        "inflight_active_systems",
        "inflight_nvt_steps",
        "inflight_nve_steps",
        "domain_live_molecules_per_species",
        "domain_construction_density_g_cm3",
        "domain_packmol_tolerance_a",
        "domain_packmol_precision_a",
        "domain_packmol_seed",
        "domain_pme_realspace_cutoff_a",
        "domain_pme_mesh_safety_factor",
        "domain_pme_alpha_a_inv",
        "domain_pme_mesh_dimensions",
        "domain_pme_mesh_spacing_a",
        "domain_pme_accuracy",
        "domain_ewald_reference_accuracy",
        "domain_halo_skin_a",
        "domain_model_cutoff_a",
        "domain_compile",
    ),
}

_CHECK_SECTION_KEYS = {
    "model_composition": (
        "residual_serial_batch_max_abs_eV",
        "full_serial_batch_max_abs_eV",
        "component_closure_max_abs_eV",
        "official_calculator_agreement",
        "analytic_coulomb",
        "compiled_ir_eager_agreement",
        "compiled_ir_repeat_agreement",
        "finite_difference_force_energy_route",
        "finite_difference_force_step_A",
        "finite_difference_force_reference_eV_A",
        "finite_difference_force_official_analytic_eV_A",
        "finite_difference_force_official_abs_error_eV_A",
        "finite_difference_force_pipeline_eV_A",
        "finite_difference_force_pipeline_abs_error_eV_A",
    ),
    "nci": (
        "graph_charge_conservation_max_abs_e",
        "component_sum_max_abs_eV",
        "graph_order_max_abs_eV",
        "nci_complete_max_MAE_vs_DFT_D3_kcal_mol",
        "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol",
        "nci_force_check",
    ),
    "sevennet": ("sevennet_adapter",),
    "dynamics": (
        "cluster_integrity_passed",
        "initial_ring_persisted_all_frames",
        "energy_excursion_within_advisory",
        "reported_comparisons",
        "fused_stage_route_counts",
    ),
    "harmonic": (
        "harmonic_checks",
        "harmonic_comparison_reported",
        "harmonic_final_fmax_eV_A",
        "harmonic_frequency_MAE_vs_B97_3c_cm1",
        "harmonic_selected_Hessian_antisymmetry_relative",
        "harmonic_final_frequency_step_change_cm1",
        "harmonic_final_intensity_step_change_km_mol",
    ),
    "scaling": (
        "inflight_completed_systems",
        "inflight_unique_system_ids",
        "inflight_duplicate_system_ids",
        "inflight_nvt_counts_correct",
        "inflight_nve_counts_correct",
        "domain_world_size",
        "domain_spatially_decomposed",
        "domain_atom_count",
        "domain_energy_eV",
        "domain_force_max_eV_A",
        "domain_charge_sum_e",
        "domain_charge_sum_tolerance_e",
        "domain_charge_neutral",
        "domain_elapsed_s",
        "domain_peak_memory_GB",
        "domain_recorded_results_available",
        "domain_recorded_successful_cases",
        "domain_recorded_failed_cases",
        "pipeline_recorded_results_available",
    ),
}


def _format_schema_difference(
    *,
    label: str,
    observed: set[str],
    expected: set[str],
) -> str:
    details = []
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    return f"{label} does not match the required schema: " + "; ".join(details)


def _flatten_manifest_sections(
    sections: ManifestSections,
    *,
    group: str,
    schema: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    if not isinstance(sections, Mapping):
        raise TypeError(f"{group} sections must be a mapping")

    observed_sections = set(sections)
    expected_sections = set(schema)
    if observed_sections != expected_sections:
        raise ValueError(
            _format_schema_difference(
                label=f"{group} sections",
                observed=observed_sections,
                expected=expected_sections,
            )
        )

    flattened: dict[str, Any] = {}
    for section_name, expected_keys in schema.items():
        values = sections[section_name]
        if not isinstance(values, Mapping):
            raise TypeError(f"{group}[{section_name!r}] must be a mapping")
        observed_keys = set(values)
        required_keys = set(expected_keys)
        if observed_keys != required_keys:
            raise ValueError(
                _format_schema_difference(
                    label=f"{group}[{section_name!r}]",
                    observed=observed_keys,
                    expected=required_keys,
                )
            )
        duplicate_keys = flattened.keys() & observed_keys
        if duplicate_keys:
            raise ValueError(
                f"{group} repeats keys across sections: {sorted(duplicate_keys)}"
            )
        flattened.update((key, values[key]) for key in expected_keys)

    return MappingProxyType(flattened)


def _validated_manifest_record(
    values: ManifestRecord,
    *,
    group: str,
    schema: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{group} must be a mapping")
    expected_keys = tuple(key for keys in schema.values() for key in keys)
    observed = set(values)
    expected = set(expected_keys)
    if observed != expected:
        raise ValueError(
            _format_schema_difference(
                label=group,
                observed=observed,
                expected=expected,
            )
        )
    return MappingProxyType({key: values[key] for key in expected_keys})


@dataclass(frozen=True)
class WaterRunManifestInput:
    """Complete manifest values, grouped while authoring and flat when saved.

    Use :meth:`from_sections` to build this object from small records created
    beside the calculation that produced them.  The method requires every
    current manifest key and rejects extras.  It does not read notebook globals,
    calculate scientific results, or supply fallback values.
    """

    run_details: Mapping[str, Any]
    settings: Mapping[str, Any]
    checks: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_details",
            _validated_manifest_record(
                self.run_details,
                group="run_details",
                schema=_RUN_DETAIL_SECTION_KEYS,
            ),
        )
        object.__setattr__(
            self,
            "settings",
            _validated_manifest_record(
                self.settings,
                group="settings",
                schema=_SETTING_SECTION_KEYS,
            ),
        )
        object.__setattr__(
            self,
            "checks",
            _validated_manifest_record(
                self.checks,
                group="checks",
                schema=_CHECK_SECTION_KEYS,
            ),
        )

    @classmethod
    def from_sections(
        cls,
        *,
        run_details: ManifestSections,
        settings: ManifestSections,
        checks: ManifestSections,
    ) -> "WaterRunManifestInput":
        """Validate and flatten the three groups used by the saved manifest."""

        return cls(
            run_details=_flatten_manifest_sections(
                run_details,
                group="run_details",
                schema=_RUN_DETAIL_SECTION_KEYS,
            ),
            settings=_flatten_manifest_sections(
                settings,
                group="settings",
                schema=_SETTING_SECTION_KEYS,
            ),
            checks=_flatten_manifest_sections(
                checks,
                group="checks",
                schema=_CHECK_SECTION_KEYS,
            ),
        )

    def as_save_arguments(self) -> dict[str, Mapping[str, Any]]:
        """Return keyword arguments accepted by ``save_water_run_outputs``."""

        return {
            "run_details": self.run_details,
            "settings": self.settings,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class WaterRunResults:
    """Calculated tables and spectra produced by the water-IR notebook.

    Field names describe scientific results rather than filenames.  The
    filename mapping stays private to this module so notebook cells and the run
    validator cannot silently choose different names.
    """

    diagnostics: Table
    spectrum_metrics: Table
    topology_summary: Table
    comparisons: Table
    dft_comparison: Table
    h_to_d_mode_map: Table
    harmonic_displacements: Table
    harmonic_convergence: Table
    harmonic_checks: Table
    harmonic_comparison: Table
    nci_interaction_curves: Table
    nci_interaction_metrics: Table
    nci_ensemble_curves: Table
    dimer_ablation: Table
    dimer_ablation_mae: Table
    adsorption_results: Table
    adsorption_forces: Table
    sevennet_graph_mapping: Table
    sevennet_numerical_agreement: Table
    first_warm_calls: Table
    cpu_gpu_crossover: Table
    inflight_summary: Table
    domain_molecule_charges: Table
    domain_molecule_charge_summary: Table
    domain_live_summary: Table
    results_summary: Table
    batch_layout_timings: pd.DataFrame | Sequence[Mapping[str, Any]]
    topology_timelines: Mapping[str, Table]
    spectra: Mapping[str, tuple[Any, Any]]


@dataclass(frozen=True)
class SavedWaterRun:
    """Files and manifest returned after a successful save."""

    output_dir: Path
    written_files: tuple[Path, ...]
    spectrum_table: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def relative_files(self) -> tuple[str, ...]:
        """Return written filenames relative to the run directory."""

        return tuple(
            path.relative_to(self.output_dir).as_posix() for path in self.written_files
        )


@dataclass(frozen=True)
class _TableOutput:
    field: str
    filename: str
    index: bool
    header: bool = True


_TABLE_OUTPUTS = (
    _TableOutput("diagnostics", "water_ir_diagnostics.csv", index=True),
    _TableOutput("spectrum_metrics", "water_ir_metrics.csv", index=True),
    _TableOutput("topology_summary", "water_ir_topology.csv", index=True),
    _TableOutput("comparisons", "water_ir_comparisons.csv", index=True),
    _TableOutput("dft_comparison", "water_ir_dft_comparison.csv", index=True),
    _TableOutput("h_to_d_mode_map", "water_ir_h_to_d_mode_map.csv", index=False),
    _TableOutput(
        "harmonic_displacements",
        "water_monomer_harmonic_displacements.csv",
        index=False,
    ),
    _TableOutput(
        "harmonic_convergence",
        "water_monomer_harmonic_convergence.csv",
        index=False,
    ),
    _TableOutput(
        "harmonic_checks",
        "water_monomer_harmonic_checks.csv",
        index=False,
    ),
    _TableOutput(
        "harmonic_comparison",
        "water_monomer_harmonic_comparison.csv",
        index=False,
    ),
    _TableOutput(
        "nci_interaction_curves",
        "nci_interaction_curves.csv",
        index=False,
    ),
    _TableOutput(
        "nci_interaction_metrics",
        "nci_interaction_metrics.csv",
        index=True,
    ),
    _TableOutput(
        "nci_ensemble_curves",
        "nci_ensemble_curves.csv",
        index=False,
    ),
    _TableOutput("dimer_ablation", "water_dimer_ablation.csv", index=False),
    _TableOutput(
        "dimer_ablation_mae",
        "water_dimer_ablation_mae.csv",
        index=True,
    ),
    _TableOutput(
        "adsorption_results",
        "surface_adsorption_energies.csv",
        index=False,
    ),
    _TableOutput(
        "adsorption_forces",
        "surface_adsorption_forces.csv",
        index=False,
    ),
    _TableOutput(
        "sevennet_graph_mapping",
        "sevennet_adapter_graph_mapping.csv",
        index=False,
    ),
    _TableOutput(
        "sevennet_numerical_agreement",
        "sevennet_adapter_numerical_agreement.csv",
        index=False,
    ),
    _TableOutput(
        "first_warm_calls",
        "water_batch_first_warm_calls.csv",
        index=False,
    ),
    _TableOutput(
        "cpu_gpu_crossover",
        "water_batch_cpu_gpu_crossover.csv",
        index=False,
    ),
    _TableOutput("inflight_summary", "inflight_queue_summary.csv", index=False),
    _TableOutput(
        "domain_molecule_charges",
        "domain_molecule_charges.csv",
        index=False,
    ),
    _TableOutput(
        "domain_molecule_charge_summary",
        "domain_molecule_charge_summary.csv",
        index=False,
    ),
    _TableOutput(
        "domain_live_summary",
        "domain_single_gpu_result.csv",
        index=False,
    ),
    _TableOutput("results_summary", "part1_results_summary.csv", index=False),
)

_TOPOLOGY_TIMELINE_FILENAMES = {
    "(H2O)6": "water_ir_h6_topology_timeline.csv",
    "(D2O)6": "water_ir_d6_topology_timeline.csv",
}


def _require_table(value: Any, *, field: str) -> Table:
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{field} must be a pandas DataFrame or Series")
    return value


def _validated_topology_timelines(
    timelines: Mapping[str, Table],
) -> tuple[tuple[str, Table], ...]:
    observed = set(timelines)
    expected = set(_TOPOLOGY_TIMELINE_FILENAMES)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            "topology_timelines must contain the H2O and D2O hexamers exactly: "
            + "; ".join(details)
        )
    return tuple(
        (
            _TOPOLOGY_TIMELINE_FILENAMES[label],
            _require_table(timelines[label], field=f"topology_timelines[{label!r}]"),
        )
        for label in _TOPOLOGY_TIMELINE_FILENAMES
    )


def _spectrum_table(spectra: Mapping[str, tuple[Any, Any]]) -> pd.DataFrame:
    if not spectra:
        raise ValueError("spectra must contain at least one system")

    columns: dict[str, np.ndarray] = {}
    shared_wavenumbers: np.ndarray | None = None
    for label, values in spectra.items():
        if not isinstance(values, tuple) or len(values) != 2:
            raise TypeError(
                f"spectra[{label!r}] must be a (wavenumbers, intensity) tuple"
            )
        wavenumbers = np.asarray(values[0])
        intensity = np.asarray(values[1])
        if wavenumbers.ndim != 1 or intensity.ndim != 1:
            raise ValueError(f"spectra[{label!r}] arrays must be one-dimensional")
        if wavenumbers.shape != intensity.shape:
            raise ValueError(
                f"spectra[{label!r}] wavenumber and intensity lengths differ"
            )
        if not np.isfinite(wavenumbers).all() or not np.isfinite(intensity).all():
            raise ValueError(f"spectra[{label!r}] contains non-finite values")
        if shared_wavenumbers is None:
            shared_wavenumbers = wavenumbers
            columns["wavenumber_cm-1"] = wavenumbers
        elif not np.array_equal(wavenumbers, shared_wavenumbers):
            raise ValueError("all spectra must use the same wavenumber grid")
        columns[f"{label}_PSD_arb"] = intensity

    return pd.DataFrame(columns)


def save_water_run_outputs(
    output_dir: str | Path,
    *,
    results: WaterRunResults,
    run_details: Mapping[str, Any],
    settings: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> SavedWaterRun:
    """Write all final Part 1 tables and the deterministic run manifest.

    ``results`` contains already calculated values.  ``run_details``,
    ``settings``, and ``checks`` are passed unchanged to
    :func:`aux.artifacts.write_water_run_manifest`; keeping these mappings at
    the call site makes every scientific and runtime choice explicit.

    Inputs are validated before any file is written.  Existing files elsewhere
    in the run directory, such as the raw trajectory and Toolkit Zarr store,
    are not modified and are included by the manifest inventory.
    """

    if not isinstance(results, WaterRunResults):
        raise TypeError("results must be a WaterRunResults instance")
    if not isinstance(run_details, Mapping):
        raise TypeError("run_details must be a mapping")
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    if not isinstance(checks, Mapping):
        raise TypeError("checks must be a mapping")

    fixed_tables = tuple(
        (
            spec,
            _require_table(getattr(results, spec.field), field=spec.field),
        )
        for spec in _TABLE_OUTPUTS
    )
    batch_layouts = pd.DataFrame(results.batch_layout_timings)
    topology_tables = _validated_topology_timelines(results.topology_timelines)
    spectrum_table = _spectrum_table(results.spectra)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for spec, table in fixed_tables:
        path = destination / spec.filename
        table.to_csv(path, index=spec.index, header=spec.header)
        written.append(path)

    batch_layout_path = destination / "water_batch_layouts.csv"
    batch_layouts.to_csv(batch_layout_path, index=False)
    written.append(batch_layout_path)

    for filename, table in topology_tables:
        path = destination / filename
        table.to_csv(path, index=False, header=True)
        written.append(path)

    spectrum_path = destination / "water_ir_spectra.csv"
    spectrum_table.to_csv(spectrum_path, index=False)
    written.append(spectrum_path)

    manifest = write_water_run_manifest(
        destination,
        run_details=run_details,
        settings=settings,
        checks=checks,
    )
    written.append(destination / WATER_RUN_MANIFEST_NAME)

    return SavedWaterRun(
        output_dir=destination,
        written_files=tuple(
            sorted(written, key=lambda path: path.relative_to(destination).as_posix())
        ),
        spectrum_table=spectrum_table,
        manifest=manifest,
    )


__all__ = [
    "SavedWaterRun",
    "WaterRunManifestInput",
    "WaterRunResults",
    "save_water_run_outputs",
]
