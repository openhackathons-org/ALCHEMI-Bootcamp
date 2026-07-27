"""Tests for the complete Part 1 run-output writer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import WATER_RUN_MANIFEST_NAME  # noqa: E402
from aux.run_output import (  # noqa: E402
    WaterRunManifestInput,
    WaterRunResults,
    save_water_run_outputs,
)


RUN_DETAIL_SECTION_KEYS = {
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

SETTING_SECTION_KEYS = {
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

CHECK_SECTION_KEYS = {
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


def _section_values(
    schema: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, str]]:
    return {
        section: {key: f"{section}:{key}" for key in keys}
        for section, keys in schema.items()
    }


def _manifest_input() -> WaterRunManifestInput:
    return WaterRunManifestInput.from_sections(
        run_details=_section_values(RUN_DETAIL_SECTION_KEYS),
        settings=_section_values(SETTING_SECTION_KEYS),
        checks=_section_values(CHECK_SECTION_KEYS),
    )


def _run_results(
    *,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    topology_timelines: dict[str, pd.DataFrame] | None = None,
) -> WaterRunResults:
    indexed = pd.DataFrame({"value": [1.25]}, index=["water"])
    rows = pd.DataFrame({"label": ["water"], "value": [1.25]})
    mae = pd.Series(
        [0.63],
        index=["residual + Coulomb + D3"],
        name="MAE_vs_full_B97_3c_kJ_mol",
    )
    grid = np.array([500.0, 1_000.0, 1_500.0])
    return WaterRunResults(
        diagnostics=indexed,
        spectrum_metrics=indexed,
        topology_summary=indexed,
        comparisons=indexed,
        dft_comparison=indexed,
        h_to_d_mode_map=rows,
        harmonic_displacements=rows,
        harmonic_convergence=rows,
        harmonic_checks=rows,
        harmonic_comparison=rows,
        nci_interaction_curves=pd.DataFrame(
            {
                "system_id": ["1.041"],
                "scale": [1.0],
                "full": [-4.32],
                "dft_full": [-4.21],
            }
        ),
        nci_interaction_metrics=pd.DataFrame(
            {
                "complete vs DFT-D3": [0.11],
                "complete vs CC": [0.18],
            },
            index=["1.041"],
        ),
        nci_ensemble_curves=pd.DataFrame(
            {
                "member": [0],
                "system_id": ["1.041"],
                "scale": [1.0],
                "full": [-4.32],
            }
        ),
        dimer_ablation=rows,
        dimer_ablation_mae=mae,
        adsorption_results=pd.DataFrame(
            {
                "molecule": ["CO", "CO2"],
                "model_adsorption_energy_eV": [-0.62, -0.18],
                "d3_adsorption_energy_eV": [-0.08, -0.21],
                "adsorption_energy_eV": [-0.70, -0.39],
                "fmax_eV_A": [1.20, 0.92],
            }
        ),
        adsorption_forces=pd.DataFrame(
            {
                "structure": ["co_on_cu111"],
                "atom_index": [0],
                "element": ["Cu"],
                "fx_eV_A": [0.01],
                "fy_eV_A": [-0.02],
                "fz_eV_A": [0.03],
            }
        ),
        sevennet_graph_mapping=pd.DataFrame(
            {
                "component": ["periodic edge vectors"],
                "toolkit_shape": ["(80, 3)"],
                "sevennet_shape": ["(80, 3)"],
                "exact_match": [True],
                "max_abs_difference": [0.0],
                "units": ["Å"],
                "note": ["target - source + integer shift × cell"],
            }
        ),
        sevennet_numerical_agreement=pd.DataFrame(
            {
                "structure": ["co_on_cu111"],
                "atoms": [38],
                "energy_difference_eV": [1.0e-6],
                "energy_difference_eV_per_atom": [1.0e-6 / 38.0],
                "max_force_component_difference_eV_A": [2.0e-6],
            }
        ),
        first_warm_calls=rows,
        cpu_gpu_crossover=rows,
        inflight_summary=rows,
        domain_molecule_charges=pd.DataFrame(
            {
                "molecule_id": [0, 1],
                "component": ["phenol", "N-methylacetamide"],
                "predicted_charge_e": [0.03, -0.03],
            }
        ),
        domain_molecule_charge_summary=pd.DataFrame(
            {
                "component": ["phenol", "N-methylacetamide", "all molecules"],
                "molecules": [1, 1, 2],
                "mean_charge_e": [0.03, -0.03, 0.0],
                "mean_abs_charge_e": [0.03, 0.03, 0.03],
                "standard_deviation_e": [0.0, 0.0, 0.03],
                "minimum_charge_e": [0.03, -0.03, -0.03],
                "maximum_charge_e": [0.03, -0.03, 0.03],
                "total_charge_e": [0.03, -0.03, 0.0],
            }
        ),
        domain_live_summary=rows,
        results_summary=rows,
        batch_layout_timings=[{"route": "homogeneous", "wall_ms": 1.2}],
        topology_timelines=(
            topology_timelines
            if topology_timelines is not None
            else {
                "(H2O)6": indexed,
                "(D2O)6": indexed,
            }
        ),
        spectra=(
            spectra
            if spectra is not None
            else {
                "H2O": (grid, np.array([0.1, 0.5, 0.2])),
                "D2O": (grid.copy(), np.array([0.2, 0.6, 0.3])),
            }
        ),
    )


def test_manifest_input_flattens_all_current_fields_without_changing_names() -> None:
    run_detail_sections = _section_values(RUN_DETAIL_SECTION_KEYS)
    manifest_input = WaterRunManifestInput.from_sections(
        run_details=run_detail_sections,
        settings=_section_values(SETTING_SECTION_KEYS),
        checks=_section_values(CHECK_SECTION_KEYS),
    )

    expected_run_detail_keys = {
        key for keys in RUN_DETAIL_SECTION_KEYS.values() for key in keys
    }
    expected_setting_keys = {
        key for keys in SETTING_SECTION_KEYS.values() for key in keys
    }
    expected_check_keys = {key for keys in CHECK_SECTION_KEYS.values() for key in keys}
    assert len(expected_run_detail_keys) == 28
    assert len(expected_setting_keys) == 93
    assert len(expected_check_keys) == 52
    assert set(manifest_input.run_details) == expected_run_detail_keys
    assert set(manifest_input.settings) == expected_setting_keys
    assert set(manifest_input.checks) == expected_check_keys

    arguments = manifest_input.as_save_arguments()
    assert set(arguments) == {"run_details", "settings", "checks"}
    assert arguments["run_details"]["run_id"] == "runtime:run_id"
    assert (
        arguments["settings"]["harmonic_imaginary_floor_cm1"]
        == "harmonic:harmonic_imaginary_floor_cm1"
    )
    assert arguments["checks"]["nci_force_check"] == "nci:nci_force_check"

    run_detail_sections["runtime"]["run_id"] = "changed after construction"
    assert manifest_input.run_details["run_id"] == "runtime:run_id"
    with pytest.raises(TypeError):
        manifest_input.settings["dt_fs"] = "changed"  # type: ignore[index]


def test_manifest_input_rejects_missing_or_unexpected_values() -> None:
    run_details = _section_values(RUN_DETAIL_SECTION_KEYS)
    settings = _section_values(SETTING_SECTION_KEYS)
    checks = _section_values(CHECK_SECTION_KEYS)
    del settings["dynamics"]["dt_fs"]

    with pytest.raises(ValueError, match=r"settings\['dynamics'\].*missing.*dt_fs"):
        WaterRunManifestInput.from_sections(
            run_details=run_details,
            settings=settings,
            checks=checks,
        )

    settings = _section_values(SETTING_SECTION_KEYS)
    settings["dynamics"]["unreviewed_default"] = "not allowed"
    with pytest.raises(
        ValueError,
        match=r"settings\['dynamics'\].*unexpected.*unreviewed_default",
    ):
        WaterRunManifestInput.from_sections(
            run_details=run_details,
            settings=settings,
            checks=checks,
        )


def test_manifest_input_saves_the_existing_flat_manifest_schema(
    tmp_path: Path,
) -> None:
    manifest_input = _manifest_input()

    save_water_run_outputs(
        tmp_path,
        results=_run_results(),
        **manifest_input.as_save_arguments(),
    )

    manifest = json.loads((tmp_path / WATER_RUN_MANIFEST_NAME).read_text())
    assert manifest["run_details"] == dict(manifest_input.run_details)
    assert manifest["settings"] == dict(manifest_input.settings)
    assert manifest["checks"] == dict(manifest_input.checks)
    assert not (set(manifest["run_details"]) & set(RUN_DETAIL_SECTION_KEYS))
    assert not (set(manifest["settings"]) & set(SETTING_SECTION_KEYS))
    assert not (set(manifest["checks"]) & set(CHECK_SECTION_KEYS))


def test_save_water_run_outputs_writes_stable_files_and_manifest(
    tmp_path: Path,
) -> None:
    preexisting = tmp_path / "water_ir_trajectory.npz"
    preexisting.write_bytes(b"raw trajectory")

    saved = save_water_run_outputs(
        tmp_path,
        results=_run_results(),
        run_details={"run_id": "3149917", "root": Path("tutorials/v2")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        checks={"passed": np.bool_(True)},
    )

    expected_written = {
        "water_batch_cpu_gpu_crossover.csv",
        "water_batch_first_warm_calls.csv",
        "water_batch_layouts.csv",
        "inflight_queue_summary.csv",
        "domain_molecule_charges.csv",
        "domain_molecule_charge_summary.csv",
        "domain_single_gpu_result.csv",
        "part1_results_summary.csv",
        "water_dimer_ablation.csv",
        "water_dimer_ablation_mae.csv",
        "water_ir_comparisons.csv",
        "water_ir_d6_topology_timeline.csv",
        "water_ir_dft_comparison.csv",
        "water_ir_diagnostics.csv",
        "water_ir_h6_topology_timeline.csv",
        "water_ir_h_to_d_mode_map.csv",
        "water_ir_metrics.csv",
        "water_ir_spectra.csv",
        "water_ir_topology.csv",
        "water_monomer_harmonic_checks.csv",
        "water_monomer_harmonic_comparison.csv",
        "water_monomer_harmonic_convergence.csv",
        "water_monomer_harmonic_displacements.csv",
        "nci_ensemble_curves.csv",
        "nci_interaction_curves.csv",
        "nci_interaction_metrics.csv",
        "surface_adsorption_energies.csv",
        "surface_adsorption_forces.csv",
        "sevennet_adapter_graph_mapping.csv",
        "sevennet_adapter_numerical_agreement.csv",
        WATER_RUN_MANIFEST_NAME,
    }
    assert set(saved.relative_files) == expected_written

    spectrum = pd.read_csv(tmp_path / "water_ir_spectra.csv")
    assert spectrum.columns.tolist() == [
        "wavenumber_cm-1",
        "H2O_PSD_arb",
        "D2O_PSD_arb",
    ]
    np.testing.assert_allclose(spectrum["wavenumber_cm-1"], [500.0, 1_000.0, 1_500.0])
    assert (
        (tmp_path / "water_ir_diagnostics.csv").read_text().startswith(",value\nwater,")
    )
    assert (
        (tmp_path / "water_ir_h_to_d_mode_map.csv")
        .read_text()
        .startswith("label,value\n")
    )
    assert (
        (tmp_path / "water_ir_h6_topology_timeline.csv")
        .read_text()
        .startswith("value\n")
    )
    assert (
        (tmp_path / "water_dimer_ablation_mae.csv")
        .read_text()
        .startswith(",MAE_vs_full_B97_3c_kJ_mol\n")
    )
    assert (
        (tmp_path / "nci_interaction_curves.csv")
        .read_text()
        .startswith("system_id,scale,full,dft_full\n")
    )
    assert (
        (tmp_path / "nci_interaction_metrics.csv")
        .read_text()
        .startswith(",complete vs DFT-D3,complete vs CC\n")
    )
    assert (
        (tmp_path / "nci_ensemble_curves.csv")
        .read_text()
        .startswith("member,system_id,scale,full\n")
    )
    assert (
        (tmp_path / "surface_adsorption_energies.csv")
        .read_text()
        .startswith(
            "molecule,model_adsorption_energy_eV,d3_adsorption_energy_eV,"
            "adsorption_energy_eV,fmax_eV_A\n"
        )
    )
    assert (
        (tmp_path / "surface_adsorption_forces.csv")
        .read_text()
        .startswith("structure,atom_index,element,fx_eV_A,fy_eV_A,fz_eV_A\n")
    )
    assert (
        (tmp_path / "sevennet_adapter_graph_mapping.csv")
        .read_text()
        .startswith(
            "component,toolkit_shape,sevennet_shape,exact_match,max_abs_difference,"
            "units,note\n"
        )
    )
    assert (
        (tmp_path / "sevennet_adapter_numerical_agreement.csv")
        .read_text()
        .startswith(
            "structure,atoms,energy_difference_eV,energy_difference_eV_per_atom,"
            "max_force_component_difference_eV_A\n"
        )
    )
    assert (
        (tmp_path / "domain_molecule_charges.csv")
        .read_text()
        .startswith("molecule_id,component,predicted_charge_e\n")
    )
    assert (
        (tmp_path / "domain_molecule_charge_summary.csv")
        .read_text()
        .startswith(
            "component,molecules,mean_charge_e,mean_abs_charge_e,"
            "standard_deviation_e,minimum_charge_e,maximum_charge_e,"
            "total_charge_e\n"
        )
    )

    manifest_path = tmp_path / WATER_RUN_MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert saved.manifest == manifest
    assert manifest["schema"] == "alchemi.water-ir-run.v2"
    assert manifest["run_details"]["root"] == "tutorials/v2"
    assert manifest["settings"] == {"dt_fs": 0.5, "steps": 55_000}
    assert manifest["checks"] == {"passed": True}
    inventoried = {record["path"] for record in manifest["files"]}
    assert inventoried == (expected_written - {WATER_RUN_MANIFEST_NAME}) | {
        preexisting.name
    }

    repeated = save_water_run_outputs(
        tmp_path,
        results=_run_results(),
        run_details={"run_id": "3149917", "root": Path("tutorials/v2")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        checks={"passed": np.bool_(True)},
    )
    assert manifest_path.read_bytes() == manifest_bytes
    assert repeated.manifest == saved.manifest


def test_save_water_run_outputs_rejects_mismatched_spectrum_grids_before_writing(
    tmp_path: Path,
) -> None:
    results = _run_results(
        spectra={
            "H2O": (np.array([500.0, 1_000.0]), np.array([0.1, 0.2])),
            "D2O": (np.array([500.0, 1_001.0]), np.array([0.3, 0.4])),
        }
    )

    with pytest.raises(ValueError, match="same wavenumber grid"):
        save_water_run_outputs(
            tmp_path,
            results=results,
            run_details={},
            settings={},
            checks={},
        )

    assert not list(tmp_path.glob("*.csv"))


def test_save_water_run_outputs_requires_both_hexamer_timelines(
    tmp_path: Path,
) -> None:
    results = _run_results(topology_timelines={"(H2O)6": pd.DataFrame({"value": [1]})})

    with pytest.raises(ValueError, match="missing.*D2O"):
        save_water_run_outputs(
            tmp_path,
            results=results,
            run_details={},
            settings={},
            checks={},
        )

    assert not list(tmp_path.glob("*.csv"))
