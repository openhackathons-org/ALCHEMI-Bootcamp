"""Focused tests for strict post-run acceptance checks."""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import nbformat
import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "scripts" / "validate_part1_ir_run.py"
SPEC = importlib.util.spec_from_file_location("validate_part1_ir_run", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_validate_notebook_accepts_only_declared_release_link_rebasing(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "part-1"
    release_dir = source_dir / "outputs" / "run-42"
    source_path = source_dir / "tutorial.ipynb"
    reviewed_path = release_dir / "tutorial-reviewed.ipynb"
    source_dir.mkdir()
    release_dir.mkdir(parents=True)
    local_targets = (
        source_dir
        / "assets/images/banner_candidates"
        / "water-ir-v2-04-trajectory-to-spectrum.png",
        source_dir / "COMPUTE_LAB_RUNBOOK.md",
        tmp_path / "part-2-batched-adsorption-toolkit/README.md",
        tmp_path / "THIRD_PARTY_NOTICES.md",
    )
    for target in local_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    markdown = "\n".join(
        f"[Stage 7 script {index}]({reference})"
        for index, reference in enumerate(
            VALIDATOR.LOCAL_MARKDOWN_REFERENCES,
            start=1,
        )
    )
    source = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(markdown, id="intro"),
            nbformat.v4.new_code_cell(
                "answer = 42", id="calculation", execution_count=1
            ),
        ]
    )
    nbformat.write(source, source_path)

    replacements = VALIDATOR.local_reference_replacements(
        source_dir=source_dir,
        output_dir=release_dir,
    )
    reviewed = deepcopy(source)
    for original, replacement in replacements.items():
        reviewed.cells[0].source = reviewed.cells[0].source.replace(
            original, replacement
        )
    reviewed.metadata["alchemi_review"] = {
        "kind": "markdown-only-source-refresh",
        "code_sources_unchanged": True,
        "rebased_local_markdown_references": replacements,
    }
    nbformat.write(reviewed, reviewed_path)

    assert VALIDATOR.validate_notebook(reviewed_path, source_path) == 1

    reviewed.cells[0].source += "\nUndeclared edit."
    nbformat.write(reviewed, reviewed_path)
    with pytest.raises(RuntimeError, match="do not match the source notebook"):
        VALIDATOR.validate_notebook(reviewed_path, source_path)


def test_release_link_rebasing_uses_current_notebook_local_references() -> None:
    assert VALIDATOR.LOCAL_MARKDOWN_REFERENCES == (
        "assets/images/banner_candidates/"
        "water-ir-v2-04-trajectory-to-spectrum.png",
        "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set",
        "COMPUTE_LAB_RUNBOOK.md"
        "#6-check-the-separate-distributedpipeline-campaign",
        "../part-2-batched-adsorption-toolkit/README.md",
        "../THIRD_PARTY_NOTICES.md",
    )


def valid_manifest() -> dict[str, object]:
    return {
        "checks": {
            "residual_serial_batch_max_abs_eV": 1e-7,
            "full_serial_batch_max_abs_eV": 2e-7,
            "component_closure_max_abs_eV": 3e-7,
            "official_calculator_agreement": {
                "energy_eV": 1e-7,
                "interaction_energy_eV": 2e-8,
                "forces_eV_A": 2e-7,
                "charges_e": 3e-9,
            },
            "analytic_coulomb": {
                "energy_eV": 1e-7,
                "forces_eV_A": 2e-7,
            },
            "compiled_ir_eager_agreement": {
                "energy": 1e-7,
                "forces": 2e-7,
                "charges": 3e-9,
            },
            "compiled_ir_repeat_agreement": {
                "energy": 0.0,
                "forces": 0.0,
                "charges": 0.0,
            },
            "finite_difference_force_energy_route": (
                "official AIMNet2Calculator total energy"
            ),
            "finite_difference_force_step_A": 0.003,
            "finite_difference_force_reference_eV_A": 0.1,
            "finite_difference_force_official_analytic_eV_A": 0.0995,
            "finite_difference_force_official_abs_error_eV_A": 0.0005,
            "finite_difference_force_pipeline_eV_A": 0.101,
            "finite_difference_force_pipeline_abs_error_eV_A": 0.001,
        }
    }


def test_load_run_manifest_uses_plain_v2_section_names(tmp_path: Path) -> None:
    path = tmp_path / "water_run_manifest.json"
    expected = {
        "schema": "alchemi.water-ir-run.v2",
        "run_details": {},
        "settings": {},
        "checks": {},
        "files": [],
    }
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert VALIDATOR.load_run_manifest(path) == expected

    legacy = dict(expected)
    legacy["schema"] = "alchemi.water-ir-run.v1"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected run manifest schema"):
        VALIDATOR.load_run_manifest(path)

    old_sections = {
        "schema": "alchemi.water-ir-run.v2",
        "provenance": {},
        "settings": {},
        "gates": {},
        "files": [],
    }
    path.write_text(json.dumps(old_sections), encoding="utf-8")
    with pytest.raises(ValueError, match="incorrect top-level fields"):
        VALIDATOR.load_run_manifest(path)


def test_cluster_check_promotes_saved_positions_before_distance_math() -> None:
    positions = np.array(
        [[[0.0, 0.0, 0.0], [0.9876543, 0.1234567, 0.0], [-0.2, 0.95, 0.0]]],
        dtype=np.float32,
    )
    arrays = {
        "batch_ptr": np.array([0, 3], dtype=np.int64),
        "atomic_numbers": np.array([8, 1, 1], dtype=np.int64),
        "positions_angstrom": positions,
    }
    expected = float(
        np.linalg.norm(
            positions.astype(np.float64)[0, 1:] - positions.astype(np.float64)[0, 0],
            axis=1,
        ).max()
    )
    float32_value = float(np.linalg.norm(positions[0, 1:], axis=1).max())

    result = VALIDATOR.cluster_check_from_trajectory(
        arrays,
        0,
        oxygen_cutoff_angstrom=4.0,
        h_acceptor_cutoff_angstrom=2.5,
        oo_cutoff_angstrom=3.5,
        hbond_angle_cutoff_deg=140.0,
    )

    assert abs(expected - float32_value) > 1.0e-9
    assert result["max_OH_angstrom"] == pytest.approx(expected, abs=1.0e-15)


def write_harmonic_fixture(
    output_dir: Path,
    *,
    minimum_passes: bool = True,
    archive_positions_as_float32: bool = False,
) -> tuple[dict[str, object], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239987, 0.926627, 0.0],
        ]
    )
    atomic_numbers = np.array([8, 1, 1])
    masses = {
        "H2O": np.array([15.999, 1.008, 1.008]),
        "D2O": np.array([15.999, 2.014, 2.014]),
    }
    steps_bohr = np.array([0.020, 0.010, 0.005])
    selected_step_bohr = 0.005
    hessian_eV_per_angstrom2 = np.diag(np.linspace(0.2, 1.0, 9))
    charge_template = np.array([-0.8, 0.4, 0.4])

    archive: dict[str, np.ndarray] = {
        "geometry_angstrom": geometry,
        "atomic_numbers": atomic_numbers,
        "H2O_masses_u": masses["H2O"],
        "D2O_masses_u": masses["D2O"],
        "minimum_forces_eV_per_angstrom": (
            np.zeros((3, 3))
            if minimum_passes
            else np.array([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
        "selected_step_bohr": np.asarray(selected_step_bohr),
    }
    estimates = []
    displacement_rows = []
    for step_bohr in steps_bohr:
        step_angstrom = float(step_bohr * VALIDATOR.ANGSTROM_PER_BOHR)
        displacements = VALIDATOR.symmetric_cartesian_displacements(
            geometry, step_angstrom
        )
        positions = np.concatenate(
            (displacements.plus_angstrom, displacements.minus_angstrom), axis=0
        )
        if archive_positions_as_float32:
            positions = positions.astype(np.float32)
        stored_positions = positions.astype(np.float64, copy=False)
        plus_forces = np.stack(
            [
                (-hessian_eV_per_angstrom2[:, coordinate] * step_angstrom).reshape(3, 3)
                for coordinate in range(9)
            ]
        )
        minus_forces = -plus_forces
        forces = np.concatenate((plus_forces, minus_forces), axis=0)
        charges = np.broadcast_to(charge_template, (18, 3)).copy()
        dipoles = VALIDATOR.molecular_dipoles_from_atomic_predictions(
            stored_positions,
            charges,
            origin_angstrom=stored_positions[:, 0, :],
            neutral_tolerance_e=1.0e-12,
        )
        estimate = VALIDATOR.assemble_harmonic_ir_finite_difference(
            forces_plus_eV_per_angstrom=plus_forces,
            forces_minus_eV_per_angstrom=minus_forces,
            dipoles_plus_e_angstrom=dipoles[:9],
            dipoles_minus_e_angstrom=dipoles[9:],
            step_angstrom=step_angstrom,
        )
        estimates.append(estimate)
        prefix = f"step_{step_bohr:.3f}".replace(".", "p")
        archive[f"{prefix}_positions_angstrom"] = positions
        archive[f"{prefix}_forces_eV_per_angstrom"] = forces
        archive[f"{prefix}_charges_e"] = charges
        archive[f"{prefix}_dipoles_e_angstrom"] = dipoles
        plus_flat = stored_positions[:9].reshape(9, 9)
        minus_flat = stored_positions[9:].reshape(9, 9)
        coordinates = np.arange(9)
        realized_steps = 0.5 * (
            plus_flat[coordinates, coordinates] - minus_flat[coordinates, coordinates]
        )
        realized_error = float(
            np.max(np.abs(realized_steps - step_angstrom)) / step_angstrom
        )
        displacement_rows.append(
            {
                "step_bohr": step_bohr,
                "step_angstrom": step_angstrom,
                "structures_in_call": 18,
                "max_realized_step_relative_error": realized_error,
                "raw_H_max_antisymmetry_relative": (
                    estimate.hessian.max_relative_antisymmetry
                ),
            }
        )

    selected = estimates[-1]
    archive.update(
        {
            "hessian_raw_eV_per_angstrom2": (
                selected.hessian.raw_hessian_eV_per_angstrom2
            ),
            "hessian_eV_per_angstrom2": selected.hessian.hessian_eV_per_angstrom2,
            "hessian_hartree_per_bohr2": selected.hessian_hartree_per_bohr2,
            "dipole_derivative_3n_by_3_au": (selected.dipole_derivative_3n_by_3_au),
        }
    )
    analyses = {
        label: VALIDATOR.analyze_harmonic_ir(
            selected.hessian_hartree_per_bohr2,
            selected.dipole_derivative_3n_by_3_au,
            geometry,
            mass_values,
        )
        for label, mass_values in masses.items()
    }
    convergence = {
        label: VALIDATOR.summarize_harmonic_ir_convergence(
            estimates, geometry, mass_values
        )
        for label, mass_values in masses.items()
    }
    for label, analysis in analyses.items():
        archive[f"{label}_frequencies_cm1"] = analysis.frequencies_cm1
        archive[f"{label}_ir_intensities_km_mol"] = analysis.ir_intensities_km_mol
        archive[f"{label}_mass_weighted_modes"] = analysis.mass_weighted_modes

    archive_path = output_dir / "water_monomer_aimnet_harmonic_ir.npz"
    np.savez_compressed(archive_path, **archive)
    write(
        output_dir / "water_monomer_harmonic_minimum.extxyz",
        Atoms(numbers=atomic_numbers, positions=geometry),
    )
    pd.DataFrame(displacement_rows).to_csv(
        output_dir / "water_monomer_harmonic_displacements.csv", index=False
    )
    pd.DataFrame(
        {
            "isotopologue": ["H2O", "D2O"],
            "max |frequency change|, 0.010→0.005 bohr (cm-1)": [
                convergence[label].frequency_max_abs_change_cm1[-1]
                for label in ("H2O", "D2O")
            ],
            "max |intensity change|, 0.010→0.005 bohr (km/mol)": [
                convergence[label].ir_intensity_max_abs_change_km_mol[-1]
                for label in ("H2O", "D2O")
            ],
            "minimum same-mode overlap": [
                convergence[label].minimum_same_index_mode_squared_overlap[-1]
                for label in ("H2O", "D2O")
            ],
        }
    ).to_csv(output_dir / "water_monomer_harmonic_convergence.csv", index=False)

    checks = {
        "tight minimum": minimum_passes,
        "H2O frequency step stability": True,
        "D2O frequency step stability": True,
        "H2O intensity step stability": True,
        "D2O intensity step stability": True,
        "mode continuity": True,
        "Hessian symmetry": True,
        "no significant imaginary modes": True,
    }
    reported = all(checks.values())
    pd.DataFrame({"check": list(checks), "passed": list(checks.values())}).to_csv(
        output_dir / "water_monomer_harmonic_checks.csv", index=False
    )

    reference_root = ROOT / "part-1-scalable-atomistic-workflows" / "reference"
    references = {
        label: VALIDATOR.load_psi4_b973c_ir_artifact(
            reference_root / "artifacts" / directory
        )
        for label, directory in (("H2O", "h2o"), ("D2O", "d2o"))
    }
    observed = VALIDATOR.load_experimental_water_fundamentals(
        reference_root / "experimental_water_fundamentals"
    ).set_index(["isotopologue", "mode"])
    mode_order = ("symmetric_stretch", "bend", "antisymmetric_stretch")
    mode_number = {"symmetric_stretch": 1, "bend": 2, "antisymmetric_stretch": 3}
    comparison_rows = []
    for label in ("H2O", "D2O"):
        aimnet_labels = VALIDATOR.label_water_monomer_modes(
            geometry,
            atomic_numbers,
            masses[label],
            analyses[label].mass_weighted_modes,
        )
        dft_labels = VALIDATOR.reference_water_monomer_mode_labels(references[label])
        for mode in mode_order:
            aimnet_index = aimnet_labels.index(mode)
            dft_index = dft_labels.index(mode)
            aimnet_frequency = float(analyses[label].frequencies_cm1[aimnet_index])
            dft_frequency = float(references[label].frequencies_cm1[dft_index])
            observed_frequency = float(observed.loc[(label, mode), "wavenumber_cm1"])
            comparison_rows.append(
                {
                    "system": label,
                    "mode": f"ν{mode_number[mode]} {mode.replace('_', ' ')}",
                    "AIMNet+Coulomb+D3_harmonic_cm-1": aimnet_frequency,
                    "AIMNet_point_charge_IR_km_mol": float(
                        analyses[label].ir_intensities_km_mol[aimnet_index]
                    ),
                    "B97-3c_harmonic_cm-1": dft_frequency,
                    "B97-3c_IR_intensity_km_mol": float(
                        references[label].ir_intensities_km_mol[dft_index]
                    ),
                    "observed_gas_cm-1": observed_frequency,
                    "AIMNet+Coulomb+D3_minus_B97-3c_cm-1": (
                        aimnet_frequency - dft_frequency
                    ),
                    "B97-3c_minus_observed_cm-1": (dft_frequency - observed_frequency),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    saved_comparison = comparison if reported else comparison.iloc[0:0]
    saved_comparison.to_csv(
        output_dir / "water_monomer_harmonic_comparison.csv",
        index=False,
    )
    if reported:
        (output_dir / VALIDATOR.HARMONIC_COMPARISON_PLOT).write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
        )

    final_fmax = float(
        np.max(np.linalg.norm(archive["minimum_forces_eV_per_angstrom"], axis=1))
    )
    manifest = {
        "run_details": {
            "aimnet_harmonic_archive": {
                "path": archive_path.name,
                "sha256": VALIDATOR.sha256_file(archive_path),
            }
        },
        "settings": {
            "harmonic_fmax_eV_A": 5.0e-4,
            "harmonic_displacement_steps_bohr": steps_bohr.tolist(),
            "harmonic_selected_step_bohr": selected_step_bohr,
            "harmonic_frequency_step_tolerance_cm1": 2.0,
            "harmonic_intensity_step_abs_tolerance_km_mol": 1.0,
            "harmonic_intensity_step_relative_tolerance": 0.05,
            "harmonic_mode_overlap_min": 0.99,
            "harmonic_hessian_antisymmetry_relative_max": 1.0e-3,
            "harmonic_charge_neutrality_tolerance_e": 5.0e-5,
            "harmonic_imaginary_floor_cm1": -10.0,
        },
        "checks": {
            "harmonic_checks": checks,
            "harmonic_comparison_reported": reported,
            "harmonic_final_fmax_eV_A": final_fmax,
            "harmonic_frequency_MAE_vs_B97_3c_cm1": (
                float(
                    comparison[
                        "AIMNet+Coulomb+D3_minus_B97-3c_cm-1"
                    ].abs().mean()
                )
                if reported
                else None
            ),
            "harmonic_selected_Hessian_antisymmetry_relative": (
                selected.hessian.max_relative_antisymmetry
            ),
            "harmonic_final_frequency_step_change_cm1": {
                label: float(convergence[label].frequency_max_abs_change_cm1[-1])
                for label in ("H2O", "D2O")
            },
            "harmonic_final_intensity_step_change_km_mol": {
                label: float(convergence[label].ir_intensity_max_abs_change_km_mol[-1])
                for label in ("H2O", "D2O")
            },
        },
    }
    return manifest, archive_path


def write_derived_ir_fixture(output_dir: Path) -> dict[str, object]:
    """Write a small deterministic trajectory and its exact derived tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["H2O", "D2O", "(H2O)6", "(D2O)6"]
    reference_root = (
        ROOT / "part-1-scalable-atomistic-workflows" / "reference" / "artifacts"
    )
    reference_directories = {
        "H2O": "h2o",
        "D2O": "d2o",
        "(H2O)6": "h6",
        "(D2O)6": "d6",
    }
    references = {
        label: VALIDATOR.load_psi4_b973c_ir_artifact(
            reference_root / reference_directories[label]
        )
        for label in labels
    }

    dt_fs = 0.5
    frames = 1024
    time_fs = np.arange(frames, dtype=float) * dt_fs
    signal_wavenumbers_cm1 = np.array([3500.0, 2500.0, 3400.0, 2400.0])
    phase = (
        2.0 * np.pi * time_fs[:, None] * signal_wavenumbers_cm1[None, :] * 2.99792458e-5
    )
    dipoles = np.stack(
        (np.sin(phase), 0.4 * np.cos(phase), 0.2 * np.sin(2.0 * phase)),
        axis=-1,
    )
    atomic_numbers = np.concatenate(
        [references[label].atomic_numbers for label in labels]
    )
    atomic_masses = np.concatenate([references[label].masses_u for label in labels])
    initial_positions = np.concatenate(
        [references[label].geometry_angstrom for label in labels], axis=0
    )
    positions = np.broadcast_to(
        initial_positions, (frames, len(initial_positions), 3)
    ).copy()
    atoms_per_graph = np.array(
        [references[label].n_atoms for label in labels], dtype=int
    )
    batch_ptr = np.concatenate(([0], np.cumsum(atoms_per_graph)))
    batch_idx = np.repeat(np.arange(len(labels)), atoms_per_graph)
    temperatures_K = np.array([300.0, 500.0, 300.0, 500.0])
    kinetic = np.broadcast_to(
        1.5 * atoms_per_graph * 8.617333262145e-5 * temperatures_K,
        (frames, len(labels)),
    ).copy()
    trajectory_path = output_dir / "water_ir_trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        dipoles_e_angstrom=dipoles,
        charge_sums_e=np.zeros((frames, len(labels))),
        kinetic_energies_eV=kinetic,
        total_energies_eV=np.zeros((frames, len(labels))),
        positions_angstrom=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses_u=atomic_masses,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        dt_fs=np.asarray(dt_fs),
        labels=np.asarray(labels),
    )

    trajectory, loaded_labels = VALIDATOR.load_ir_trajectory(trajectory_path)
    settings = {
        "dt_fs": dt_fs,
        "spectrum_segment_time_fs": 256.0,
        "spectrum_overlap": 0.5,
        "spectrum_windows_cm1": {
            "H": [2800.0, 4000.0],
            "D": [2000.0, 3100.0],
        },
        "pair_temperature_relative_tolerance": 0.2,
        "covalent_OH_cutoff_A": 1.25,
        "hbond_H_acceptor_cutoff_A": 2.5,
        "hbond_OO_cutoff_A": 3.5,
        "hbond_angle_cutoff_deg": 140.0,
        "h_to_d_coarse_mass_path_steps": 65,
        "h_to_d_fine_mass_path_steps": 129,
        "h_to_d_degeneracy_tolerance_cm1": 2.0,
    }
    spectrum_analysis = VALIDATOR.ir_spectrum_metrics(
        trajectory.dipoles_e_angstrom,
        loaded_labels,
        dt_fs=dt_fs,
        segment_time_fs=settings["spectrum_segment_time_fs"],
        overlap=settings["spectrum_overlap"],
        region_windows_cm1=settings["spectrum_windows_cm1"],
    )
    spectra = spectrum_analysis.spectra
    metrics = spectrum_analysis.metrics
    nve_temperature = (
        2.0
        * trajectory.kinetic_energies_eV
        / (3.0 * atoms_per_graph[None, :] * 8.617333262145e-5)
    )
    comparisons = VALIDATOR.ir_comparison_table(
        metrics,
        nve_temperature,
        loaded_labels,
        pair_temperature_relative_tolerance=settings[
            "pair_temperature_relative_tolerance"
        ],
        cluster_reference_allowed=True,
    ).table
    dft_summary = VALIDATOR.reference_comparison_metrics(
        spectra,
        references,
        loaded_labels,
        dt_fs=dt_fs,
        segment_time_fs=settings["spectrum_segment_time_fs"],
        region_windows_cm1=settings["spectrum_windows_cm1"],
        cluster_reference_allowed=True,
    ).metrics
    mode_map = VALIDATOR.h_to_d_mode_mapping_table(
        references,
        coarse_mass_path_steps=settings["h_to_d_coarse_mass_path_steps"],
        fine_mass_path_steps=settings["h_to_d_fine_mass_path_steps"],
        degeneracy_tolerance_cm1=settings["h_to_d_degeneracy_tolerance_cm1"],
        covalent_oh_cutoff_angstrom=settings["covalent_OH_cutoff_A"],
        h_acceptor_cutoff_angstrom=settings["hbond_H_acceptor_cutoff_A"],
        oo_cutoff_angstrom=settings["hbond_OO_cutoff_A"],
        hbond_angle_cutoff_deg=settings["hbond_angle_cutoff_deg"],
    ).table
    spectrum_table = pd.DataFrame({"wavenumber_cm-1": spectra[loaded_labels[0]][0]})
    for label in loaded_labels:
        spectrum_table[f"{label}_PSD_arb"] = spectra[label][1]

    spectrum_table.to_csv(output_dir / "water_ir_spectra.csv", index=False)
    metrics.to_csv(output_dir / "water_ir_metrics.csv")
    comparisons.to_csv(output_dir / "water_ir_comparisons.csv")
    dft_summary.to_csv(output_dir / "water_ir_dft_comparison.csv")
    mode_map.to_csv(output_dir / "water_ir_h_to_d_mode_map.csv", index=False)
    for graph, slug in ((2, "h6"), (3, "d6")):
        timeline = VALIDATOR.topology_time_series(
            trajectory,
            graph,
            h_acceptor_cutoff_angstrom=settings["hbond_H_acceptor_cutoff_A"],
            oo_cutoff_angstrom=settings["hbond_OO_cutoff_A"],
            hbond_angle_cutoff_deg=settings["hbond_angle_cutoff_deg"],
        )
        timeline.to_csv(
            output_dir / f"water_ir_{slug}_topology_timeline.csv", index=False
        )

    return {
        "settings": settings,
        "checks": {"initial_ring_persisted_all_frames": True},
    }


@pytest.fixture(scope="module")
def derived_ir_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, object]]:
    output_dir = tmp_path_factory.mktemp("derived-ir-run")
    return output_dir, write_derived_ir_fixture(output_dir)


def test_composition_validator_accepts_complete_check_record() -> None:
    result = VALIDATOR.validate_composition_checks(valid_manifest())

    assert result["official_calculator_agreement"]["charges_e"] == 3e-9
    assert result["official_calculator_agreement"]["interaction_energy_eV"] == 2e-8
    assert result["finite_difference_force"]["abs_tolerance_eV_A"] == 0.002
    assert result["finite_difference_force"]["step_A"] == 0.003
    assert result["finite_difference_force"]["energy_route"] == (
        "official AIMNet2Calculator total energy"
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "finite_difference_force_energy_route",
            "Toolkit absolute energy",
            "energy route is not recognized",
        ),
        (
            "finite_difference_force_step_A",
            0.01,
            "step does not match",
        ),
    ],
)
def test_composition_validator_rejects_wrong_force_check_method(
    key: str, value: object, message: str
) -> None:
    manifest = valid_manifest()
    manifest["checks"][key] = value

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_composition_checks(manifest)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("full_serial_batch_max_abs_eV",), 2e-5, "composition check failed"),
        (
            ("official_calculator_agreement", "forces_eV_A"),
            3e-6,
            "composition check failed",
        ),
        (
            ("official_calculator_agreement", "interaction_energy_eV"),
            1e-3,
            "composition check failed",
        ),
        (
            ("analytic_coulomb", "energy_eV"),
            2e-6,
            "composition check failed",
        ),
        (
            ("finite_difference_force_pipeline_abs_error_eV_A",),
            0.002,
            "pipeline finite-difference force error was not reproduced",
        ),
    ],
)
def test_composition_validator_rejects_failed_or_inconsistent_check(
    path: tuple[str, ...], value: float, message: str
) -> None:
    manifest = deepcopy(valid_manifest())
    target = manifest["checks"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_composition_checks(manifest)


def test_execution_runner_is_hashed_and_required_in_bundle() -> None:
    assert VALIDATOR.SOURCE_PATHS == VALIDATOR.load_source_paths(ROOT)
    assert VALIDATOR.SOURCE_MANIFEST_RELATIVE_PATH in VALIDATOR.SOURCE_PATHS
    assert "scripts/run_notebook_no_timeout.py" in VALIDATOR.SOURCE_PATHS
    assert VALIDATOR.BUNDLE_SOURCE_REPOSITORY_PATHS == {
        "alchemi-water-ir-source.ipynb": (
            "part-1-scalable-atomistic-workflows/alchemi-water-ir.ipynb"
        ),
        "run_notebook_no_timeout.py": "scripts/run_notebook_no_timeout.py",
    }
    assert VALIDATOR.BUNDLE_SOURCE_FILES == tuple(
        VALIDATOR.BUNDLE_SOURCE_REPOSITORY_PATHS
    )
    assert VALIDATOR.RUNTIME_CHECK_NAME == "part1-runtime.json"
    assert VALIDATOR.RUNTIME_CHECK_NAME in VALIDATOR.BUNDLE_REQUIRED_FILES
    assert VALIDATOR.D3_CACHE_REPORT_NAME == "part1-d3-cache.json"
    assert VALIDATOR.D3_CACHE_REPORT_NAME in VALIDATOR.BUNDLE_REQUIRED_FILES
    assert VALIDATOR.TIMING_REPORT_NAME == "notebook-timings.json"
    assert VALIDATOR.TIMING_REPORT_NAME in VALIDATOR.BUNDLE_REQUIRED_FILES


@pytest.mark.parametrize(
    "tampered_name",
    tuple(VALIDATOR.BUNDLE_SOURCE_REPOSITORY_PATHS),
)
def test_packaged_source_copies_must_match_runtime_hashes(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    source_hashes: dict[str, str] = {}
    for packaged_name, repository_path in (
        VALIDATOR.BUNDLE_SOURCE_REPOSITORY_PATHS.items()
    ):
        source = ROOT / repository_path
        shutil.copy2(source, tmp_path / packaged_name)
        source_hashes[repository_path] = VALIDATOR.sha256_file(source)

    assert VALIDATOR.validate_packaged_source_copies(
        tmp_path,
        source_hashes,
    ) == {
        packaged_name: source_hashes[repository_path]
        for packaged_name, repository_path in (
            VALIDATOR.BUNDLE_SOURCE_REPOSITORY_PATHS.items()
        )
    }

    with (tmp_path / tampered_name).open("ab") as stream:
        stream.write(b"\nchanged after packaging\n")
    with pytest.raises(RuntimeError, match="packaged source SHA-256"):
        VALIDATOR.validate_packaged_source_copies(
            tmp_path,
            source_hashes,
        )


def write_timing_fixture(
    tmp_path: Path,
    *,
    reviewed: bool = False,
) -> tuple[Path, Path, Path]:
    cells = [nbformat.v4.new_code_cell("setup = True", id="setup")]
    for stage in range(1, 8):
        cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    f'<h2 id="alchemi-stage-{stage}-heading">Stage {stage}</h2>',
                    id=f"stage-{stage}",
                ),
                nbformat.v4.new_code_cell(
                    f"value_{stage} = {stage}",
                    id=f"code-{stage}",
                ),
            ]
        )
    source_notebook = nbformat.v4.new_notebook(cells=cells)
    source_path = tmp_path / "source.ipynb"
    nbformat.write(source_notebook, source_path)

    executed_notebook = deepcopy(source_notebook)
    code_index = 0
    for cell in executed_notebook.cells:
        if cell.cell_type == "code":
            code_index += 1
            cell.execution_count = code_index
    original_executed_path = tmp_path / "executed-original.ipynb"
    nbformat.write(executed_notebook, original_executed_path)
    original_executed_sha256 = VALIDATOR.sha256_file(original_executed_path)

    executed_path = original_executed_path
    if reviewed:
        reviewed_notebook = deepcopy(executed_notebook)
        reviewed_notebook.metadata["alchemi_review"] = {
            "kind": "markdown-only-source-refresh",
            "original_executed_sha256": original_executed_sha256,
        }
        executed_path = tmp_path / "executed-reviewed.ipynb"
        nbformat.write(reviewed_notebook, executed_path)

    cell_timings = []
    stage = 0
    code_index = 0
    elapsed_by_stage: dict[int, float] = {}
    count_by_stage: dict[int, int] = {}
    for cell_index, cell in enumerate(source_notebook.cells):
        if cell.id.startswith("stage-"):
            stage = int(cell.id.split("-", 1)[1])
        if cell.cell_type != "code":
            continue
        code_index += 1
        elapsed = code_index / 10.0
        elapsed_by_stage[stage] = elapsed_by_stage.get(stage, 0.0) + elapsed
        count_by_stage[stage] = count_by_stage.get(stage, 0) + 1
        cell_timings.append(
            {
                "cell_index": cell_index,
                "code_index": code_index,
                "cell_id": cell.id,
                "stage": stage,
                "stage_title": "Setup" if stage == 0 else f"Stage {stage}",
                "first_line": cell.source.splitlines()[0],
                "status": "complete",
                "started_utc": "2026-07-24T12:00:00+00:00",
                "elapsed_s": elapsed,
                "error_type": None,
                "error_message": None,
            }
        )
    stage_timings = [
        {
            "stage": stage,
            "title": "Setup" if stage == 0 else f"Stage {stage}",
            "code_cells_started": count_by_stage[stage],
            "code_cells_completed": count_by_stage[stage],
            "code_cells_failed": 0,
            "elapsed_s": elapsed_by_stage[stage],
        }
        for stage in sorted(count_by_stage)
    ]
    total_code_elapsed_s = sum(elapsed_by_stage.values())
    report = {
        "schema": VALIDATOR.TIMING_REPORT_SCHEMA,
        "status": "complete",
        "started_utc": "2026-07-24T12:00:00+00:00",
        "finished_utc": "2026-07-24T12:01:00+00:00",
        "input_notebook": "source.ipynb",
        "input_notebook_sha256": VALIDATOR.sha256_file(source_path),
        "executed_notebook": "executed.ipynb",
        "executed_notebook_sha256": original_executed_sha256,
        "kernel": "alchemi-main",
        "code_cell_count_expected": len(cell_timings),
        "code_cells_started": len(cell_timings),
        "code_cells_completed": len(cell_timings),
        "code_cells_failed": 0,
        "total_code_elapsed_s": total_code_elapsed_s,
        "total_wall_elapsed_s": total_code_elapsed_s + 1.0,
        "cell_timing_boundary": "execute_cell only",
        "wall_timing_boundary": "kernel setup through shutdown",
        "runner_error_type": None,
        "runner_error_message": None,
        "stage_timings": stage_timings,
        "cell_timings": cell_timings,
    }
    timing_path = tmp_path / VALIDATOR.TIMING_REPORT_NAME
    timing_path.write_text(json.dumps(report), encoding="utf-8")
    return timing_path, source_path, executed_path


def test_notebook_timing_report_is_validated_and_hashed(tmp_path: Path) -> None:
    timing_path, source_path, executed_path = write_timing_fixture(tmp_path)

    result = VALIDATOR.validate_notebook_timing_report(
        timing_path,
        source_notebook=source_path,
        executed_notebook=executed_path,
    )

    assert result["sha256"] == VALIDATOR.sha256_file(timing_path)
    assert result["stage_timings"][0]["title"] == "Setup"
    assert result["stage_timings"][-1]["title"] == "Stage 7"


def test_notebook_timing_report_accepts_the_original_reviewed_notebook_hash(
    tmp_path: Path,
) -> None:
    timing_path, source_path, reviewed_path = write_timing_fixture(
        tmp_path,
        reviewed=True,
    )

    result = VALIDATOR.validate_notebook_timing_report(
        timing_path,
        source_notebook=source_path,
        executed_notebook=reviewed_path,
    )

    assert result["schema"] == VALIDATOR.TIMING_REPORT_SCHEMA


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("failed_status", "not complete"),
        ("source_hash", "source SHA-256"),
        ("cell_stage", "cell identity or stage"),
        ("negative_cell_time", "non-negative"),
        ("wrong_total", "total does not equal"),
        ("wrong_stage_total", "stage total does not equal"),
        ("runner_error", "runner error"),
    ],
)
def test_notebook_timing_report_rejects_incomplete_or_changed_data(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    timing_path, source_path, executed_path = write_timing_fixture(tmp_path)
    report = json.loads(timing_path.read_text(encoding="utf-8"))
    if tamper == "failed_status":
        report["status"] = "failed"
    elif tamper == "source_hash":
        report["input_notebook_sha256"] = "0" * 64
    elif tamper == "cell_stage":
        report["cell_timings"][2]["stage"] = 7
    elif tamper == "negative_cell_time":
        report["cell_timings"][0]["elapsed_s"] = -1.0
    elif tamper == "wrong_total":
        report["total_code_elapsed_s"] += 1.0
    elif tamper == "wrong_stage_total":
        report["stage_timings"][0]["elapsed_s"] += 1.0
    elif tamper == "runner_error":
        report["runner_error_type"] = "RuntimeError"
    else:
        raise AssertionError(f"unknown timing tamper case: {tamper}")
    timing_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match=message):
        VALIDATOR.validate_notebook_timing_report(
            timing_path,
            source_notebook=source_path,
            executed_notebook=executed_path,
        )


def test_stage_7_saved_tables_are_required_outputs() -> None:
    assert {
        "domain_box_evaluated.extxyz",
        "inflight_queue_summary.csv",
        "domain_single_gpu_result.csv",
        "part1_results_summary.csv",
    } <= set(VALIDATOR.REQUIRED_FILES)
    assert VALIDATOR.REQUIRED_DIRECTORIES == ("water_ir_relaxed.zarr",)


def write_domain_box_fixture(
    tmp_path: Path,
) -> tuple[Path, Atoms, dict[str, object]]:
    source_path = (
        ROOT
        / VALIDATOR.DOMAIN_BASE_BOX_RELATIVE_DIR
        / "structure.extxyz"
    )
    atoms = read(source_path)
    atoms.set_positions(np.asarray(atoms.positions, dtype=np.float32))
    atoms.set_cell(np.asarray(atoms.cell.array, dtype=np.float32))
    atoms.set_masses(np.asarray(atoms.get_masses(), dtype=np.float32))
    atoms.info["label"] = "domain-box"
    forces = np.zeros((len(atoms), 3), dtype=np.float32)
    forces[0] = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    charges = np.linspace(-0.25, 0.25, len(atoms), dtype=np.float32)
    energy_ev = -1_937_733.375
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy_ev,
        forces=forces,
        charges=charges,
    )
    path = tmp_path / "domain_box_evaluated.extxyz"
    write(path, atoms, format="extxyz")

    charge_target_e = float(atoms.info["charge"])
    charge_sum_e = float(charges.sum(dtype=np.float64))
    charge_residual_e = charge_sum_e - charge_target_e
    run_manifest: dict[str, object] = {
        "checks": {
            "domain_world_size": 1,
            "domain_spatially_decomposed": False,
            "domain_atom_count": len(atoms),
            "domain_charge_dtype": "float32",
            "domain_charge_finite": True,
            "domain_energy_eV": energy_ev,
            "domain_force_max_eV_A": 3.0,
            "domain_charge_target_e": charge_target_e,
            "domain_charge_sum_e": charge_sum_e,
            "domain_charge_residual_e": charge_residual_e,
            "domain_charge_abs_residual_per_atom": (
                abs(charge_residual_e) / len(atoms)
            ),
        }
    }
    return path, atoms, run_manifest


def test_domain_box_output_round_trip_is_validated(tmp_path: Path) -> None:
    path, _, run_manifest = write_domain_box_fixture(tmp_path)

    result = VALIDATOR.validate_domain_box_output(
        path,
        source_root=ROOT,
        run_manifest=run_manifest,
    )

    assert result["atoms"] == 3_200
    assert result["force_max_eV_A"] == pytest.approx(3.0)
    assert result["identity_arrays"] == list(VALIDATOR.DOMAIN_IDENTITY_ARRAYS)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("second_frame", "exactly one frame"),
        ("missing_charges", "missing results"),
        ("identity", "identity array"),
        ("position", "changed the input coordinates"),
        ("manifest_energy", "energy differs"),
    ],
)
def test_domain_box_output_rejects_incomplete_or_changed_results(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    path, atoms, run_manifest = write_domain_box_fixture(tmp_path)
    if tamper == "second_frame":
        write(path, [atoms, atoms], format="extxyz")
    elif tamper == "missing_charges":
        results = dict(atoms.calc.results)
        results.pop("charges")
        atoms.calc = SinglePointCalculator(atoms, **results)
        write(path, atoms, format="extxyz")
    elif tamper == "identity":
        atoms.arrays["source_atom_id"][0] = len(atoms)
        atoms.calc = SinglePointCalculator(atoms, **atoms.calc.results)
        write(path, atoms, format="extxyz")
    elif tamper == "position":
        results = dict(atoms.calc.results)
        atoms.positions[0, 0] += 1.0e-3
        atoms.calc = SinglePointCalculator(atoms, **results)
        write(path, atoms, format="extxyz")
    elif tamper == "manifest_energy":
        run_manifest["checks"]["domain_energy_eV"] += 1.0
    else:
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises((ValueError, RuntimeError), match=message):
        VALIDATOR.validate_domain_box_output(
            path,
            source_root=ROOT,
            run_manifest=run_manifest,
        )


def packaged_runtime_check() -> dict[str, object]:
    module_names = (
        "aimnet",
        "e3nn",
        "jax",
        "nvalchemi",
        "nvalchemiops",
        "ovito",
        "physicsnemo",
        "sevenn",
        "torch",
        "warp",
    )
    source_paths = VALIDATOR.load_source_paths(ROOT)
    source_revision = VALIDATOR.git_source_revision(ROOT)
    return {
        "schema": "alchemi.part1-runtime-check.v2",
        "source": {
            "checked": True,
            "clean_checkout": True,
            **source_revision,
            "manifest_path": VALIDATOR.SOURCE_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": VALIDATOR.sha256_file(
                ROOT / VALIDATOR.SOURCE_MANIFEST_RELATIVE_PATH
            ),
            "files_sha256": {
                relative: VALIDATOR.sha256_file(ROOT / relative)
                for relative in source_paths
            },
        },
        "python": "3.12.10",
        "python_executable": "/env/bin/python",
        "base_environment": "/env",
        "versions": {
            **VALIDATOR.EXPECTED_RUNTIME_VERSIONS,
            "packmol": VALIDATOR.EXPECTED_PACKMOL_VERSION,
            "torch": "2.12.0+cu130",
            "uv": "0.9.26",
        },
        "resolved_scientific_versions": {
            name: "1.0.0" for name in VALIDATOR.RECORDED_SCIENTIFIC_VERSIONS
        },
        "commits": {
            "nvalchemi-toolkit": VALIDATOR.EXPECTED_TOOLKIT_CORE_COMMIT,
            "nvalchemi-toolkit-ops": VALIDATOR.EXPECTED_TOOLKIT_OPS_COMMIT,
        },
        "cuda_available": True,
        "cuda_device": "NVIDIA H100 NVL",
        "jax_cuda_device": "cuda:0",
        "packmol_binary": "/env/bin/packmol",
        "packmol_check": {
            "atoms": 25,
            "molecules": 2,
            "net_charge_e": 0.0,
            "packmol_precision_a": 1.0e-3,
            "density_from_mass_and_cell_g_cm3": 0.02,
            "periodic_min_distance_lower_bound_a": 2.0,
        },
        "ovito_ase_check": {
            "structures": 2,
            "particle_counts": {"first": 3, "second": 4},
        },
        "toolkit_ops_cuda_check": {
            "directed_edges": 2,
            "jax_segments": 1,
            "segments": 1,
            "warp_segments": 1,
        },
        "module_files": {
            name: f"/env/lib/python3.12/site-packages/{name}/__init__.py"
            for name in module_names
        },
    }


def d3_cache_report() -> dict[str, object]:
    return {
        "schema": "alchemi.part1-d3-cache.v1",
        "parameter_file": "/cache/nvalchemiops/dftd3_parameters.pt",
        "bytes": 1_808_183,
        "sha256": VALIDATOR.EXPECTED_D3_PARAMETER_SHA256,
        "toolkit_version": VALIDATOR.EXPECTED_RUNTIME_VERSIONS[
            "nvalchemi-toolkit"
        ],
    }


def test_d3_cache_report_is_validated_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / VALIDATOR.D3_CACHE_REPORT_NAME
    path.write_text(json.dumps(d3_cache_report()), encoding="utf-8")

    assert VALIDATOR.validate_d3_cache_report(path) == d3_cache_report()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "0" * 64, "SHA-256"),
        ("toolkit_version", "0.1.0", "Toolkit version"),
    ],
)
def test_d3_cache_report_rejects_wrong_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    report = d3_cache_report()
    report[field] = value
    path = tmp_path / VALIDATOR.D3_CACHE_REPORT_NAME
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_d3_cache_report(path)


def test_packaged_runtime_check_is_validated_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / VALIDATOR.RUNTIME_CHECK_NAME
    path.write_text(json.dumps(packaged_runtime_check()), encoding="utf-8")

    result = VALIDATOR.validate_packaged_runtime_check(path, source_root=ROOT)

    assert result["sha256"] == VALIDATOR.sha256_file(path)
    assert result["cuda_device"] == "NVIDIA H100 NVL"
    assert result["versions"]["packmol"] == VALIDATOR.EXPECTED_PACKMOL_VERSION


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("cuda_available",), False, "did not run with CUDA"),
        (("versions", "packmol"), "20.0.0", "Packmol version"),
        (("versions", "uv"), "0.9.25", "uv version"),
        (
            ("commits", "nvalchemi-toolkit"),
            "0" * 40,
            "Toolkit commits",
        ),
        (
            ("source", "checked"),
            False,
            "did not check the tutorial source",
        ),
        (
            ("source", "clean_checkout"),
            False,
            "clean tutorial checkout",
        ),
        (
            (
                "source",
                "files_sha256",
                "scripts/run_notebook_no_timeout.py",
            ),
            "0" * 64,
            "source SHA-256",
        ),
    ],
)
def test_packaged_runtime_check_rejects_wrong_hardware_or_pins(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    report = packaged_runtime_check()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    runtime_path = tmp_path / VALIDATOR.RUNTIME_CHECK_NAME
    runtime_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_packaged_runtime_check(runtime_path, source_root=ROOT)


def test_external_reference_loaders_and_bundles_are_hashed() -> None:
    required = {
        "LICENSE",
        "build/prewarm_d3.py",
        "build/prewarm_sevennet.py",
        "part-1-scalable-atomistic-workflows/aux/adsorption.py",
        "part-1-scalable-atomistic-workflows/aux/adsorption_visualization.py",
        "part-1-scalable-atomistic-workflows/aux/experimental_reference.py",
        "part-1-scalable-atomistic-workflows/aux/domain/__init__.py",
        "part-1-scalable-atomistic-workflows/aux/domain/packing.py",
        "part-1-scalable-atomistic-workflows/aux/domain/results.py",
        "part-1-scalable-atomistic-workflows/aux/framework_comparison.py",
        "part-1-scalable-atomistic-workflows/aux/composition_checks.py",
        "part-1-scalable-atomistic-workflows/aux/harmonic_workflow.py",
        "part-1-scalable-atomistic-workflows/aux/inflight.py",
        "part-1-scalable-atomistic-workflows/aux/ir_display.py",
        "part-1-scalable-atomistic-workflows/aux/models/__init__.py",
        "part-1-scalable-atomistic-workflows/aux/models/sevennet.py",
        "part-1-scalable-atomistic-workflows/aux/models/sevennet_checkpoint.py",
        "part-1-scalable-atomistic-workflows/aux/models/sevennet_checks.py",
        "part-1-scalable-atomistic-workflows/aux/models/sevennet_config.py",
        "part-1-scalable-atomistic-workflows/aux/models/sevennet_lesson.py",
        "part-1-scalable-atomistic-workflows/aux/nci_atlas.py",
        "part-1-scalable-atomistic-workflows/aux/nci_config.py",
        "part-1-scalable-atomistic-workflows/aux/nci_plotting.py",
        "part-1-scalable-atomistic-workflows/aux/nci_validation.py",
        "part-1-scalable-atomistic-workflows/aux/numerical_checks.py",
        "part-1-scalable-atomistic-workflows/aux/pipeline_campaign_results.py",
        "part-1-scalable-atomistic-workflows/aux/pipeline_campaign_view.py",
        "part-1-scalable-atomistic-workflows/aux/results_summary.py",
        "part-1-scalable-atomistic-workflows/aux/run_output.py",
        "part-1-scalable-atomistic-workflows/aux/precision.py",
        "part-1-scalable-atomistic-workflows/aux/workflow_config.py",
        "part-1-scalable-atomistic-workflows/data/nci_atlas/nci-atlas-curves.csv.gz",
        "part-1-scalable-atomistic-workflows/data/adsorption/cu111-important-molecules-v1/methodology.json",
        "part-1-scalable-atomistic-workflows/data/adsorption/cu111-important-molecules-v1/manifest.json",
        "part-1-scalable-atomistic-workflows/reference/experimental_water_fundamentals/SHA256SUMS",
        "part-1-scalable-atomistic-workflows/reference/experimental_water_fundamentals/manifest.json",
        "part-1-scalable-atomistic-workflows/reference/experimental_water_fundamentals/water_gas_phase_fundamentals.csv",
        "scripts/part1_domain_plan.py",
        "scripts/part1_domain_run.py",
        "scripts/run_part1_domain_decomposition.sh",
        "scripts/slurm_part1_domain_decomposition.sbatch",
    }

    assert required <= set(VALIDATOR.SOURCE_PATHS)
    assert not any(
        "data/compute_lab_pipeline_campaign" in path for path in VALIDATOR.SOURCE_PATHS
    )


def test_every_notebook_imported_aux_module_is_hashed() -> None:
    notebook_path = (
        ROOT / "part-1-scalable-atomistic-workflows" / "alchemi-water-ir.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    modules: set[str] = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "aux" or (
                    node.module is not None and node.module.startswith("aux.")
                ):
                    modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name == "aux" or alias.name.startswith("aux.")
                )

    imported_paths = set()
    part_root = ROOT / "part-1-scalable-atomistic-workflows"
    for module in modules:
        candidate = part_root.joinpath(*module.split("."))
        if candidate.with_suffix(".py").is_file():
            source_path = candidate.with_suffix(".py")
        else:
            source_path = candidate / "__init__.py"
        assert source_path.is_file(), module
        imported_paths.add(source_path.relative_to(ROOT).as_posix())

    assert imported_paths <= set(VALIDATOR.SOURCE_PATHS)


def test_nci_validator_schema_matches_notebook_comparisons() -> None:
    notebook_path = (
        ROOT / "part-1-scalable-atomistic-workflows" / "alchemi-water-ir.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    analyze_cell = next(
        cell for cell in notebook.cells if cell.get("id") == "analyze-nci-curves"
    )
    tree = ast.parse(analyze_cell.source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "nci_comparisons"
            for target in node.targets
        )
    )

    notebook_comparisons = ast.literal_eval(assignment.value)
    assert tuple(notebook_comparisons.items()) == tuple(
        VALIDATOR.NCI_COMPARISONS.items()
    )


def write_nci_output_fixture(
    output_dir: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    data_path = (
        ROOT
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "nci_atlas"
        / "nci-atlas-curves.csv.gz"
    )
    source = VALIDATOR.load_nci_atlas_subset(data_path)
    curve_keys = source[list(VALIDATOR.CURVE_KEY_COLUMNS)].drop_duplicates(
        ignore_index=True
    )

    member_tables = []
    curve_axis = np.linspace(-1.5, 1.5, len(curve_keys))
    for member in range(len(VALIDATOR.EXPECTED_NCI_CHECKPOINTS)):
        table = curve_keys.copy()
        table["member"] = member
        table["core"] = curve_axis + 0.01 * member
        table["core_plus_d3"] = table["core"] - 0.20
        table["core_plus_coulomb"] = table["core"] - 0.35
        table["full"] = table["core"] - 0.55
        member_tables.append(table)
    ensemble = pd.concat(member_tables, ignore_index=True)
    ensemble = ensemble[list(VALIDATOR.NCI_ENSEMBLE_COLUMNS)]

    curves = VALIDATOR.mean_member_curves(
        ensemble,
        VALIDATOR.NCI_COMPONENT_COLUMNS,
        spread_component="full",
    )
    graph_index = VALIDATOR.build_graph_index(source)
    dft = VALIDATOR.reduce_fragment_energies(
        graph_index,
        {"dft_full": source["wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"]},
    )
    cc = VALIDATOR.extract_repeated_interaction_reference(
        source,
        "ccsd_t_cbs_interaction_energy_kcal_mol",
        output_column="ccsd_t_cbs",
    )
    curves = curves.merge(
        dft,
        on=list(VALIDATOR.CURVE_KEY_COLUMNS),
        validate="one_to_one",
    )
    curves["dft_no_d3"] = curves["dft_full"] - 0.15
    curves = curves.merge(
        cc,
        on=list(VALIDATOR.CURVE_KEY_COLUMNS),
        validate="one_to_one",
    )
    curves = curves[list(VALIDATOR.NCI_CURVE_COLUMNS)]
    metrics = VALIDATOR.interaction_metrics(
        curves,
        VALIDATOR.NCI_COMPARISONS,
        mean_columns={"ensemble spread": "full_std"},
    ).reset_index()
    metrics = metrics[list(VALIDATOR.NCI_METRIC_COLUMNS)]

    curves_path = output_dir / "nci_interaction_curves.csv"
    metrics_path = output_dir / "nci_interaction_metrics.csv"
    ensemble_path = output_dir / "nci_ensemble_curves.csv"
    curves.to_csv(curves_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    ensemble.to_csv(ensemble_path, index=False)

    force_check = {
        "atom_index": 0,
        "cartesian_axis": 0,
        "finite_difference_step_A": (VALIDATOR.NCI_VALIDATION.finite_difference_step_A),
        "official_total_energy_route": (
            "official AIMNet2Calculator complete total energy with simple Coulomb "
            "and D3"
        ),
        "official_analytic_force_route": (
            "official AIMNet2Calculator complete-model force"
        ),
        "toolkit_analytic_force_route": (
            "Toolkit PipelineModelWrapper complete-model force"
        ),
        "official_analytic_force_eV_A": 0.25,
        "official_finite_difference_force_eV_A": 0.25,
        "toolkit_analytic_force_eV_A": 0.25,
        "official_analytic_vs_official_finite_difference_abs_error_eV_A": 0.0,
        "toolkit_analytic_vs_official_analytic_abs_error_eV_A": 0.0,
    }
    manifest = {
        "run_details": {
            "nci_checkpoints": list(VALIDATOR.EXPECTED_NCI_CHECKPOINTS),
            "nci_subset_sha256": VALIDATOR.sha256_file(data_path),
        },
        "settings": {
            "nci_graphs": len(source),
            "nci_interaction_geometries": len(curve_keys),
            "nci_reference_levels": list(VALIDATOR.EXPECTED_NCI_REFERENCE_LEVELS),
            "nci_validation": VALIDATOR.NCI_VALIDATION.as_record(),
        },
        "checks": {
            "nci_complete_max_MAE_vs_DFT_D3_kcal_mol": float(
                metrics["complete vs DFT-D3"].max()
            ),
            "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol": float(
                metrics["complete vs CC"].max()
            ),
            "nci_force_check": force_check,
        },
    }
    return curves_path, metrics_path, ensemble_path, manifest


def test_nci_outputs_are_tied_to_the_checked_input_and_saved_tables(
    tmp_path: Path,
) -> None:
    curves, metrics, ensemble, manifest = write_nci_output_fixture(tmp_path)

    result = VALIDATOR.validate_nci_outputs(
        curves,
        metrics,
        ensemble,
        manifest,
        source_root=ROOT,
    )

    assert result["input_sha256"] == VALIDATOR.NCI_ATLAS_SUBSET_SHA256
    assert result["graph_rows"] == 90
    assert result["curve_rows"] == 30
    assert result["ensemble_rows"] == 120


def test_nci_outputs_reject_a_run_detail_for_another_input(tmp_path: Path) -> None:
    curves, metrics, ensemble, manifest = write_nci_output_fixture(tmp_path)
    manifest["run_details"]["nci_subset_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="does not match the checked NCI input"):
        VALIDATOR.validate_nci_outputs(
            curves,
            metrics,
            ensemble,
            manifest,
            source_root=ROOT,
        )


def test_nci_outputs_reject_changed_curve_identity(tmp_path: Path) -> None:
    curves, metrics, ensemble, manifest = write_nci_output_fixture(tmp_path)
    table = pd.read_csv(curves)
    table.loc[0, "system_name"] = "different complex"
    table.to_csv(curves, index=False)

    with pytest.raises(RuntimeError, match="checked NCI curve identities"):
        VALIDATOR.validate_nci_outputs(
            curves,
            metrics,
            ensemble,
            manifest,
            source_root=ROOT,
        )


def test_nci_outputs_reject_a_missing_schema_column(tmp_path: Path) -> None:
    curves, metrics, ensemble, manifest = write_nci_output_fixture(tmp_path)
    table = pd.read_csv(ensemble).drop(columns="core_plus_d3")
    table.to_csv(ensemble, index=False)

    with pytest.raises(ValueError, match="columns differ from the notebook schema"):
        VALIDATOR.validate_nci_outputs(
            curves,
            metrics,
            ensemble,
            manifest,
            source_root=ROOT,
        )


def test_nci_outputs_reject_metrics_not_reproduced_from_curves(
    tmp_path: Path,
) -> None:
    curves, metrics, ensemble, manifest = write_nci_output_fixture(tmp_path)
    table = pd.read_csv(metrics)
    table.loc[0, "complete vs CC"] += 0.1
    table.to_csv(metrics, index=False)

    with pytest.raises(RuntimeError, match="do not reproduce the saved curve errors"):
        VALIDATOR.validate_nci_outputs(
            curves,
            metrics,
            ensemble,
            manifest,
            source_root=ROOT,
        )


def write_sevennet_adsorption_fixture(
    output_dir: Path,
    *,
    energy_difference_per_atom: float = 1.0e-6,
    force_difference: float = 2.0e-5,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    energies_path = output_dir / "surface_adsorption_energies.csv"
    forces_path = output_dir / "surface_adsorption_forces.csv"
    mapping_path = output_dir / "sevennet_adapter_graph_mapping.csv"
    agreement_path = output_dir / "sevennet_adapter_numerical_agreement.csv"
    structures = VALIDATOR.load_initial_structure_set(
        ROOT
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "adsorption"
        / "cu111-important-molecules-v1"
    )

    mapping_rows = [
        {
            "component": component,
            "toolkit_shape": "(1,)",
            "sevennet_shape": "(1,)",
            "exact_match": True,
            "max_abs_difference": 0.0,
            "units": "index",
            "note": "fixture",
        }
        for component in VALIDATOR.SEVENNET_MAPPING_COMPONENTS
    ]
    pd.DataFrame(mapping_rows, columns=VALIDATOR.SEVENNET_MAPPING_COLUMNS).to_csv(
        mapping_path, index=False
    )

    agreement_rows = [
        {
            "comparison": "adapter output vs direct raw call",
            "structure": key,
            "atoms": len(structures[key]),
            "energy_difference_eV": (energy_difference_per_atom * len(structures[key])),
            "energy_difference_eV_per_atom": energy_difference_per_atom,
            "max_force_component_difference_eV_A": force_difference,
        }
        for key in VALIDATOR.STRUCTURE_KEYS
    ]
    co_key = VALIDATOR.ADSLAB_KEYS["CO"]
    agreement_rows.append(
        {
            "comparison": "custom adapter vs official SevenNetCalculator",
            "structure": co_key,
            "atoms": len(structures[co_key]),
            "energy_difference_eV": (
                energy_difference_per_atom * len(structures[co_key])
            ),
            "energy_difference_eV_per_atom": energy_difference_per_atom,
            "max_force_component_difference_eV_A": force_difference,
        }
    )
    agreement_rows.extend(
        {
            "comparison": "pipeline output vs explicit component sum",
            "structure": key,
            "atoms": len(structures[key]),
            "energy_difference_eV": (energy_difference_per_atom * len(structures[key])),
            "energy_difference_eV_per_atom": energy_difference_per_atom,
            "max_force_component_difference_eV_A": force_difference,
        }
        for key in VALIDATOR.STRUCTURE_KEYS
    )
    pd.DataFrame(
        agreement_rows,
        columns=VALIDATOR.SEVENNET_AGREEMENT_COLUMNS,
    ).to_csv(agreement_path, index=False)

    force_rows = []
    force_stats: dict[str, tuple[float, float]] = {}
    for structure_index, key in enumerate(VALIDATOR.STRUCTURE_KEYS):
        atoms = structures[key]
        scale = 0.001 * (structure_index + 1)
        vectors = np.tile(np.array([scale, -2.0 * scale, 0.5 * scale]), (len(atoms), 1))
        vectors += np.arange(len(atoms), dtype=float)[:, None] * 1.0e-5
        norms = np.linalg.norm(vectors, axis=1)
        force_stats[key] = (
            float(norms.max()),
            float(np.sqrt(np.mean(norms**2))),
        )
        for atom_index, (atom, position, vector, norm, is_adsorbate) in enumerate(
            zip(
                atoms,
                atoms.positions,
                vectors,
                norms,
                atoms.arrays["is_adsorbate"],
                strict=True,
            )
        ):
            force_rows.append(
                {
                    "structure": key,
                    "role": atoms.info["role"],
                    "atom_index": atom_index,
                    "element": atom.symbol,
                    "is_adsorbate": bool(is_adsorbate),
                    "x_angstrom": position[0],
                    "y_angstrom": position[1],
                    "z_angstrom": position[2],
                    "fx_eV_A": vector[0],
                    "fy_eV_A": vector[1],
                    "fz_eV_A": vector[2],
                    "force_norm_eV_A": norm,
                }
            )
    pd.DataFrame(force_rows, columns=VALIDATOR.ADSORPTION_FORCE_COLUMNS).to_csv(
        forces_path, index=False
    )
    max_all_structure_fmax = max(fmax for fmax, _force_rms in force_stats.values())

    energy_rows = []
    for index, molecule in enumerate(VALIDATOR.ADSORBATES):
        model_clean = -100.0
        model_gas = -10.0 * (index + 1)
        model_adsorption = -1.0 - 0.2 * index
        model_adslab = model_clean + model_gas + model_adsorption
        d3_clean = -2.0
        d3_gas = -0.1 * (index + 1)
        d3_adsorption = -0.05 * (index + 1)
        d3_adslab = d3_clean + d3_gas + d3_adsorption
        adslab_key = VALIDATOR.ADSLAB_KEYS[molecule]
        fmax, force_rms = force_stats[adslab_key]
        energy_rows.append(
            {
                "molecule": molecule,
                "model_adslab_energy_eV": model_adslab,
                "model_clean_slab_energy_eV": model_clean,
                "model_gas_energy_eV": model_gas,
                "model_adsorption_energy_eV": model_adsorption,
                "d3_adslab_energy_eV": d3_adslab,
                "d3_clean_slab_energy_eV": d3_clean,
                "d3_gas_energy_eV": d3_gas,
                "d3_adsorption_energy_eV": d3_adsorption,
                "combined_adslab_energy_eV": model_adslab + d3_adslab,
                "combined_clean_slab_energy_eV": model_clean + d3_clean,
                "combined_gas_energy_eV": model_gas + d3_gas,
                "adsorption_energy_eV": model_adsorption + d3_adsorption,
                "fmax_eV_A": fmax,
                "force_rms_eV_A": force_rms,
                "force_atoms": len(structures[adslab_key]),
            }
        )
    energy_table = pd.DataFrame(
        energy_rows, columns=VALIDATOR.ADSORPTION_ENERGY_COLUMNS
    )
    energy_table.to_csv(energies_path, index=False)

    manifest = {
        "settings": {
            "custom_adapter_energy_repeat_tolerance_eV_per_atom": (
                VALIDATOR.SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM
            ),
            "custom_adapter_force_repeat_tolerance_eV_A": (
                VALIDATOR.SEVENNET_REPEAT_FORCE_TOL_EV_A
            ),
        },
        "checks": {
            "sevennet_adapter": {
                "graph_mapping_passed": True,
                "structures": 9,
                "batches": 2,
                "finite_outputs": True,
                "numerical_max_abs_energy_eV_per_atom": (energy_difference_per_atom),
                "numerical_max_abs_forces_eV_A": force_difference,
                "max_combined_fmax_eV_A": max_all_structure_fmax,
                "geometry_status": VALIDATOR.GEOMETRY_STATUS,
                "molecules": list(VALIDATOR.ADSORBATES),
                "periodic_pbc": [True, True, False],
            }
        },
    }
    return energies_path, forces_path, mapping_path, agreement_path, manifest


def test_sevennet_adsorption_validator_accepts_complete_fixed_geometry_example(
    tmp_path: Path,
) -> None:
    result = VALIDATOR.validate_sevennet_adsorption_outputs(
        *write_sevennet_adsorption_fixture(tmp_path), source_root=ROOT
    )

    assert result["structures"] == 9
    assert result["batches"] == 2
    assert result["molecules"] == ["CO", "CO2", "NH3", "CH3OH"]
    assert result["geometry_status"] == VALIDATOR.GEOMETRY_STATUS
    assert result["forces"]["rows"] == 210


def test_sevennet_adapter_force_max_covers_all_nine_structures(
    tmp_path: Path,
) -> None:
    """The saved all-structure maximum may come from a gas reference."""

    fixture = write_sevennet_adsorption_fixture(tmp_path)
    energies_path, forces_path, mapping_path, agreement_path, manifest = fixture
    forces = pd.read_csv(forces_path)
    gas_rows = forces["structure"] == "co_gas"
    forces.loc[gas_rows, ["fx_eV_A", "fy_eV_A", "fz_eV_A"]] = [2.0, 0.0, 0.0]
    forces.loc[gas_rows, "force_norm_eV_A"] = 2.0
    forces.to_csv(forces_path, index=False)
    manifest["checks"]["sevennet_adapter"]["max_combined_fmax_eV_A"] = 2.0

    result = VALIDATOR.validate_sevennet_adsorption_outputs(
        energies_path,
        forces_path,
        mapping_path,
        agreement_path,
        manifest,
        source_root=ROOT,
    )

    assert result["forces"]["max_fmax_eV_A"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("energy_difference_per_atom", "force_difference", "message"),
    [
        (
            VALIDATOR.SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
            1.0e-5,
            "energy-per-atom agreement",
        ),
        (
            1.0e-6,
            VALIDATOR.SEVENNET_REPEAT_FORCE_TOL_EV_A,
            "force agreement",
        ),
    ],
)
def test_sevennet_adsorption_validator_rejects_repeat_at_strict_limit(
    tmp_path: Path,
    energy_difference_per_atom: float,
    force_difference: float,
    message: str,
) -> None:
    paths = write_sevennet_adsorption_fixture(
        tmp_path,
        energy_difference_per_atom=energy_difference_per_atom,
        force_difference=force_difference,
    )

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_sevennet_adsorption_outputs(*paths, source_root=ROOT)


def test_sevennet_adsorption_validator_recomputes_adsorption_formula(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    table = pd.read_csv(energies)
    table.loc[0, "adsorption_energy_eV"] += 0.2
    table.to_csv(energies, index=False)

    with pytest.raises(RuntimeError, match="adsorption energy formula"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_rejects_nonfinite_all_atom_force(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    table = pd.read_csv(forces)
    table.loc[50, "fx_eV_A"] = np.nan
    table.to_csv(forces, index=False)

    with pytest.raises(RuntimeError, match="non-finite"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_matches_force_statistics_to_full_table(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    table = pd.read_csv(energies)
    table.loc[1, "fmax_eV_A"] += 0.1
    table.to_csv(energies, index=False)

    with pytest.raises(RuntimeError, match="fmax does not match full forces"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_requires_pinned_initial_coordinates(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    table = pd.read_csv(forces)
    table.loc[0, "x_angstrom"] += 0.01
    table.to_csv(forces, index=False)

    with pytest.raises(RuntimeError, match="pinned initial"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_rejects_failed_graph_mapping(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    table = pd.read_csv(mapping)
    table.loc[4, "exact_match"] = False
    table.loc[4, "max_abs_difference"] = 1.0
    table.to_csv(mapping, index=False)

    with pytest.raises(RuntimeError, match="graph fields do not match"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_requires_all_numerical_comparisons(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    pd.read_csv(agreement).iloc[:-1].to_csv(agreement, index=False)

    with pytest.raises(RuntimeError, match="raw-model, official-calculator"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_sevennet_adsorption_validator_requires_unrelaxed_status(
    tmp_path: Path,
) -> None:
    energies, forces, mapping, agreement, manifest = write_sevennet_adsorption_fixture(
        tmp_path
    )
    manifest["checks"]["sevennet_adapter"]["geometry_status"] = "relaxed"

    with pytest.raises(RuntimeError, match="geometry_status"):
        VALIDATOR.validate_sevennet_adsorption_outputs(
            energies, forces, mapping, agreement, manifest, source_root=ROOT
        )


def test_run_details_require_pinned_sevennet_and_structure_manifests() -> None:
    source_notebook = (
        ROOT / "part-1-scalable-atomistic-workflows" / "alchemi-water-ir.ipynb"
    )
    adsorption_manifest = (
        ROOT
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "adsorption"
        / "cu111-important-molecules-v1"
        / "manifest.json"
    )
    nci_subset = (
        ROOT
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "nci_atlas"
        / "nci-atlas-curves.csv.gz"
    )
    details = {
        "checkpoint_sha256": VALIDATOR.EXPECTED_CHECKPOINT_SHA256,
        "d3_parameter_file_sha256": VALIDATOR.EXPECTED_D3_PARAMETER_SHA256,
        "toolkit_core_commit": VALIDATOR.EXPECTED_TOOLKIT_CORE_COMMIT,
        "toolkit_ops_commit": VALIDATOR.EXPECTED_TOOLKIT_OPS_COMMIT,
        "aimnet": VALIDATOR.EXPECTED_AIMNET_VERSION,
        "sevennet": VALIDATOR.EXPECTED_SEVENNET_VERSION,
        "sevennet_checkpoint_source": VALIDATOR.SEVENNET_CHECKPOINT_URL,
        "sevennet_checkpoint_sha256": (VALIDATOR.EXPECTED_SEVENNET_CHECKPOINT_SHA256),
        "sevennet_checkpoint_doi": VALIDATOR.SEVENNET_CHECKPOINT_DOI,
        "sevennet_task": VALIDATOR.SEVENNET_MODALITY,
        "sevennet_reference_method": VALIDATOR.SEVENNET_REFERENCE_METHOD,
        "adsorption_structure_manifest_sha256": VALIDATOR.sha256_file(
            adsorption_manifest
        ),
        "nci_checkpoints": list(VALIDATOR.EXPECTED_NCI_CHECKPOINTS),
        "aimnet_checkpoint_identities": VALIDATOR.CHECKPOINT_IDENTITIES,
        "nci_subset_sha256": VALIDATOR.sha256_file(nci_subset),
        "checkpoint_override": False,
        "torch": "2.12.0+cu130",
        "notebook_sha256": VALIDATOR.sha256_file(source_notebook),
        **VALIDATOR.expected_reference_bundle_details(ROOT),
    }
    expected_domain_bundle = details["domain_decomposition_bundle"]
    assert isinstance(expected_domain_bundle, dict)
    manifest = {"run_details": details}

    result = VALIDATOR.validate_run_details(manifest, source_notebook, ROOT)
    assert result["sevennet_checkpoint_sha256"] == (
        VALIDATOR.EXPECTED_SEVENNET_CHECKPOINT_SHA256
    )
    assert result["nci_subset_sha256"] == VALIDATOR.NCI_ATLAS_SUBSET_SHA256
    assert result["domain_decomposition_bundle"] == expected_domain_bundle

    details["domain_decomposition_bundle"] = {"manifest_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="does not match the live bundle"):
        VALIDATOR.validate_run_details(manifest, source_notebook, ROOT)
    details["domain_decomposition_bundle"] = expected_domain_bundle

    details["adsorption_structure_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="adsorption_structure_manifest_sha256"):
        VALIDATOR.validate_run_details(manifest, source_notebook, ROOT)


def test_domain_producer_hashes_accept_external_d3_filename() -> None:
    source_paths = (
        "scripts/part1_domain_plan.py",
        "scripts/part1_domain_run.py",
        "scripts/run_part1_domain_decomposition.sh",
        "scripts/slurm_part1_domain_decomposition.sbatch",
        "part-1-scalable-atomistic-workflows/aux/domain/packing.py",
        "part-1-scalable-atomistic-workflows/aux/domain/config.py",
        "part-1-scalable-atomistic-workflows/data/nci_atlas/nci-atlas-curves.csv.gz",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/manifest.json",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/structure.extxyz",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/SHA256SUMS",
    )
    producers = {
        Path(relative).name: VALIDATOR.sha256_file(ROOT / relative)
        for relative in source_paths
    }
    producers["site-d3-cache.pt"] = VALIDATOR.EXPECTED_D3_PARAMETER_SHA256

    assert VALIDATOR.validate_domain_producer_hashes(
        producers,
        source_root=ROOT,
    ) == dict(sorted(producers.items()))

    producers["part1_domain_run.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="part1_domain_run.py"):
        VALIDATOR.validate_domain_producer_hashes(
            producers,
            source_root=ROOT,
        )


@pytest.mark.parametrize("minimum_passes", [True, False])
def test_harmonic_validator_rebuilds_arrays_and_accepts_reported_or_withheld_results(
    tmp_path: Path,
    minimum_passes: bool,
) -> None:
    manifest, _ = write_harmonic_fixture(tmp_path, minimum_passes=minimum_passes)

    result = VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)

    assert result["comparison_reported"] is minimum_passes
    assert result["comparison_plot_present"] is minimum_passes
    assert result["checks"]["tight minimum"] is minimum_passes
    if minimum_passes:
        assert result["frequency_MAE_vs_B97_3c_cm1"] > 0.0
    else:
        assert result["frequency_MAE_vs_B97_3c_cm1"] is None
    assert result["candidate_frequency_MAE_vs_B97_3c_cm1"] > 0.0


def test_harmonic_validator_accepts_float32_archived_displacement_metric(
    tmp_path: Path,
) -> None:
    manifest, archive_path = write_harmonic_fixture(
        tmp_path,
        archive_positions_as_float32=True,
    )
    displacement_table = pd.read_csv(
        tmp_path / "water_monomer_harmonic_displacements.csv"
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        expected_errors = []
        for step_bohr in VALIDATOR.HARMONIC_DISPLACEMENT_STEPS_BOHR:
            prefix = f"step_{step_bohr:.3f}".replace(".", "p")
            archived = archive[f"{prefix}_positions_angstrom"]
            assert archived.dtype == np.float32
            positions = archived.astype(np.float64)
            plus_flat = positions[:9].reshape(9, 9)
            minus_flat = positions[9:].reshape(9, 9)
            coordinates = np.arange(9)
            step_angstrom = float(step_bohr * VALIDATOR.ANGSTROM_PER_BOHR)
            realized_steps = 0.5 * (
                plus_flat[coordinates, coordinates]
                - minus_flat[coordinates, coordinates]
            )
            expected_errors.append(
                float(np.max(np.abs(realized_steps - step_angstrom)) / step_angstrom)
            )

    np.testing.assert_allclose(
        displacement_table["max_realized_step_relative_error"],
        expected_errors,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert np.max(expected_errors) > 0.0
    result = VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)
    assert result["comparison_reported"] is True


def test_harmonic_validator_rejects_rows_when_comparison_is_unreported(
    tmp_path: Path,
) -> None:
    manifest, _ = write_harmonic_fixture(tmp_path, minimum_passes=False)
    reported_manifest, _ = write_harmonic_fixture(
        tmp_path / "reported",
        minimum_passes=True,
    )
    reported_table = pd.read_csv(
        tmp_path / "reported" / "water_monomer_harmonic_comparison.csv"
    )
    reported_table.to_csv(
        tmp_path / "water_monomer_harmonic_comparison.csv",
        index=False,
    )

    with pytest.raises(RuntimeError, match="must contain no data rows"):
        VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)

    assert reported_manifest["checks"]["harmonic_comparison_reported"] is True


def test_harmonic_validator_rejects_mae_when_comparison_is_unreported(
    tmp_path: Path,
) -> None:
    manifest, _ = write_harmonic_fixture(tmp_path, minimum_passes=False)
    checks = manifest["checks"]
    assert isinstance(checks, dict)
    checks["harmonic_frequency_MAE_vs_B97_3c_cm1"] = 18.4

    with pytest.raises(RuntimeError, match="frequency MAE must be null"):
        VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)


def test_harmonic_validator_rejects_a_tampered_force_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    manifest, archive_path = write_harmonic_fixture(tmp_path)
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["step_0p005_forces_eV_per_angstrom"][0, 0, 0] += 0.05
    np.savez_compressed(archive_path, **arrays)
    manifest["run_details"]["aimnet_harmonic_archive"]["sha256"] = (
        VALIDATOR.sha256_file(archive_path)
    )

    with pytest.raises(RuntimeError, match="harmonic output mismatch"):
        VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)


def test_harmonic_validator_rejects_loosened_manifest_thresholds(
    tmp_path: Path,
) -> None:
    manifest, _ = write_harmonic_fixture(tmp_path)
    manifest["settings"]["harmonic_frequency_step_tolerance_cm1"] = 100.0

    with pytest.raises(RuntimeError, match="fixed harmonic frequency tolerance"):
        VALIDATOR.validate_harmonic_outputs(tmp_path, ROOT, manifest)


def test_derived_ir_validator_recomputes_every_saved_table(
    derived_ir_fixture: tuple[Path, dict[str, object]],
) -> None:
    output_dir, manifest = derived_ir_fixture

    result = VALIDATOR.validate_derived_ir_outputs(
        output_dir,
        output_dir / "water_ir_trajectory.npz",
        ROOT,
        manifest,
    )

    expected_rows = {
        "water_ir_spectra.csv": 257,
        "water_ir_metrics.csv": 4,
        "water_ir_comparisons.csv": 4,
        "water_ir_dft_comparison.csv": 4,
        "water_ir_h6_topology_timeline.csv": 1024,
        "water_ir_d6_topology_timeline.csv": 1024,
        "water_ir_h_to_d_mode_map.csv": 12,
    }
    assert set(result["tables"]) == set(expected_rows)
    for filename, row_count in expected_rows.items():
        assert result["tables"][filename]["rows"] == row_count
        assert result["tables"][filename]["max_abs_difference"] < 1.0e-9
    comparisons = pd.read_csv(output_dir / "water_ir_comparisons.csv")
    assert comparisons["reported"].tolist() == [False, False, True, True]
    assert comparisons["value"].isna().tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("water_ir_spectra.csv", "numeric"),
        ("water_ir_metrics.csv", "integer"),
        ("water_ir_comparisons.csv", "nan"),
        ("water_ir_dft_comparison.csv", "string"),
        ("water_ir_h6_topology_timeline.csv", "boolean"),
        ("water_ir_d6_topology_timeline.csv", "numeric"),
        ("water_ir_h_to_d_mode_map.csv", "schema"),
    ],
)
def test_derived_ir_validator_rejects_tampered_outputs(
    tmp_path: Path,
    derived_ir_fixture: tuple[Path, dict[str, object]],
    filename: str,
    mutation: str,
) -> None:
    fixture_dir, manifest = derived_ir_fixture
    output_dir = tmp_path / "run"
    shutil.copytree(fixture_dir, output_dir)
    path = output_dir / filename
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if mutation == "numeric":
        column = (
            "H2O_PSD_arb"
            if filename == "water_ir_spectra.csv"
            else "oxygen_RMSD_angstrom"
        )
        table.loc[1, column] = str(float(table.loc[1, column]) + 0.01)
    elif mutation == "integer":
        table.loc[0, "Welch_segments"] = table.loc[0, "Welch_segments"] + ".0"
    elif mutation == "nan":
        assert table.loc[0, "value"] == ""
        table.loc[0, "value"] = "1.0"
    elif mutation == "string":
        table.loc[0, "comparison_scope"] = "quantitative match"
    elif mutation == "boolean":
        table.loc[0, "initial_ring_present"] = "False"
    elif mutation == "schema":
        table = table.rename(columns={"mapping_overlap": "overlap"})
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(f"unknown mutation: {mutation}")
    table.to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="derived IR output mismatch"):
        VALIDATOR.validate_derived_ir_outputs(
            output_dir,
            output_dir / "water_ir_trajectory.npz",
            ROOT,
            manifest,
        )
