"""Regression tests for notebook-facing harmonic workflow post-processing."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.experimental_reference import (  # noqa: E402
    load_experimental_water_fundamentals,
)
from aux.harmonic_ir import (  # noqa: E402
    ANGSTROM_PER_BOHR,
    analyze_harmonic_ir,
    assemble_harmonic_ir_finite_difference,
    molecular_dipoles_from_atomic_predictions,
    summarize_harmonic_ir_convergence,
    symmetric_cartesian_displacements,
)
from aux.harmonic_workflow import (  # noqa: E402
    HarmonicDisplacementResult,
    analyze_harmonic_step_series,
    build_harmonic_archive_arrays,
    build_harmonic_mode_comparison_table,
    collect_harmonic_displacement_result,
    empty_harmonic_mode_comparison_table,
)
from aux.reference import (  # noqa: E402
    label_water_monomer_modes,
    load_psi4_b973c_ir_artifact,
    reference_water_monomer_mode_labels,
)


GEOMETRY_ANGSTROM = np.array(
    [
        [0.000_000, 0.000_000, 0.000_000],
        [0.957_200, 0.000_000, 0.000_000],
        [-0.239_987, 0.927_297, 0.000_000],
    ],
    dtype=np.float64,
)
ATOMIC_NUMBERS = np.array([8, 1, 1], dtype=np.int64)
H2O_MASSES_U = np.array([15.994_914_62, 1.007_825_03, 1.007_825_03])
D2O_MASSES_U = np.array([15.994_914_62, 2.014_101_78, 2.014_101_78])
ISOTOPOLOGUES = (("H2O", H2O_MASSES_U), ("D2O", D2O_MASSES_U))
MODE_ORDER = ("symmetric_stretch", "bend", "antisymmetric_stretch")
MODE_NUMBERS = {"symmetric_stretch": 1, "bend": 2, "antisymmetric_stretch": 3}


def test_empty_mode_comparison_table_preserves_saved_file_schema() -> None:
    table = empty_harmonic_mode_comparison_table()

    assert table.empty
    assert table.columns.tolist() == [
        "system",
        "mode",
        "AIMNet+Coulomb+D3_harmonic_cm-1",
        "AIMNet_point_charge_IR_km_mol",
        "B97-3c_harmonic_cm-1",
        "B97-3c_IR_intensity_km_mol",
        "observed_gas_cm-1",
        "AIMNet+Coulomb+D3_minus_B97-3c_cm-1",
        "B97-3c_minus_observed_cm-1",
    ]


def _quadratic_outputs(
    step_bohr: float,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    step_angstrom = float(step_bohr) * ANGSTROM_PER_BOHR
    displaced = symmetric_cartesian_displacements(GEOMETRY_ANGSTROM, step_angstrom)
    positions = np.concatenate(
        [displaced.plus_angstrom, displaced.minus_angstrom],
        axis=0,
    ).astype(dtype)
    n_coordinates = 3 * GEOMETRY_ANGSTROM.shape[0]
    coupling = np.arange(1, n_coordinates * n_coordinates + 1, dtype=np.float64)
    coupling = coupling.reshape(n_coordinates, n_coordinates) / 200.0
    hessian = coupling.T @ coupling + np.diag(np.linspace(0.75, 1.35, n_coordinates))
    delta = positions.astype(np.float64).reshape(2 * n_coordinates, n_coordinates)
    delta -= GEOMETRY_ANGSTROM.reshape(1, n_coordinates)
    forces = (-(delta @ hessian.T)).reshape(2 * n_coordinates, 3, 3)
    forces = forces.astype(dtype)
    charges = np.broadcast_to(
        np.array([-0.8, 0.4, 0.4], dtype=dtype),
        (2 * n_coordinates, 3),
    ).copy()
    return step_angstrom, positions, forces, charges


def _collect_series() -> tuple[HarmonicDisplacementResult, ...]:
    results = []
    for step_bohr in (0.020, 0.010, 0.005):
        step_angstrom, positions, forces, charges = _quadratic_outputs(step_bohr)
        results.append(
            collect_harmonic_displacement_result(
                step_bohr=step_bohr,
                step_angstrom=step_angstrom,
                n_atoms=3,
                positions_angstrom=positions.reshape(-1, 3),
                forces_eV_per_angstrom=forces.reshape(-1, 3),
                charges_e=charges.reshape(-1, 1),
                dipole_origin_atom_index=0,
                neutral_tolerance_e=1.0e-12,
            )
        )
    return tuple(results)


def _analyze_series(
    results: tuple[HarmonicDisplacementResult, ...],
    **overrides: float | bool,
):
    settings: dict[str, float | bool] = {
        "minimum_passed": True,
        "frequency_step_tolerance_cm1": 1.0e-8,
        "intensity_step_abs_tolerance_km_mol": 1.0e-8,
        "intensity_step_rel_tolerance": 1.0e-8,
        "mode_overlap_min": 0.999_999,
        "hessian_antisymmetry_rel_max": 1.0e-10,
        "imaginary_floor_cm1": -10.0,
    }
    settings.update(overrides)
    return analyze_harmonic_step_series(
        results,
        geometry_angstrom=GEOMETRY_ANGSTROM,
        isotopologues=ISOTOPOLOGUES,
        selected_step_bohr=0.005,
        **settings,
    )


def test_collect_displacement_result_matches_original_notebook_formulas() -> None:
    step_bohr = 0.010
    step_angstrom, positions, forces, charges = _quadratic_outputs(
        step_bohr,
        dtype=np.dtype(np.float32),
    )

    result = collect_harmonic_displacement_result(
        step_bohr=step_bohr,
        step_angstrom=step_angstrom,
        n_atoms=3,
        positions_angstrom=positions.reshape(-1, 3),
        forces_eV_per_angstrom=forces.reshape(-1, 3),
        charges_e=charges.reshape(-1, 1),
        dipole_origin_atom_index=0,
        neutral_tolerance_e=1.0e-6,
    )

    # These are the calculations previously written inline in the notebook.
    n_coordinates = 9
    evaluated_positions = positions.reshape(2 * n_coordinates, 3, 3)
    evaluated_forces = forces.reshape(2 * n_coordinates, 3, 3)
    evaluated_charges = charges.reshape(2 * n_coordinates, 3)
    expected_dipoles = molecular_dipoles_from_atomic_predictions(
        evaluated_positions,
        evaluated_charges,
        origin_angstrom=evaluated_positions[:, 0, :],
        neutral_tolerance_e=1.0e-6,
    )
    expected_estimate = assemble_harmonic_ir_finite_difference(
        forces_plus_eV_per_angstrom=evaluated_forces[:n_coordinates],
        forces_minus_eV_per_angstrom=evaluated_forces[n_coordinates:],
        dipoles_plus_e_angstrom=expected_dipoles[:n_coordinates],
        dipoles_minus_e_angstrom=expected_dipoles[n_coordinates:],
        step_angstrom=step_angstrom,
    )
    coordinate = np.arange(n_coordinates)
    stored = evaluated_positions.astype(np.float64, copy=False)
    realized_half_steps = 0.5 * (
        stored[:n_coordinates].reshape(n_coordinates, n_coordinates)[
            coordinate, coordinate
        ]
        - stored[n_coordinates:].reshape(n_coordinates, n_coordinates)[
            coordinate, coordinate
        ]
    )
    expected_step_error = float(
        np.max(np.abs(realized_half_steps - step_angstrom)) / step_angstrom
    )

    assert tuple(result.table_row()) == (
        "step_bohr",
        "step_angstrom",
        "structures_in_call",
        "max_realized_step_relative_error",
        "raw_H_max_antisymmetry_relative",
    )
    assert result.structures_in_call == 18
    assert result.max_realized_step_relative_error == expected_step_error
    assert result.samples.positions_angstrom.dtype == np.float32
    assert result.samples.forces_eV_per_angstrom.dtype == np.float32
    assert result.samples.charges_e.dtype == np.float32
    assert not result.samples.positions_angstrom.flags.writeable
    np.testing.assert_array_equal(result.samples.positions_angstrom, positions)
    np.testing.assert_allclose(
        result.samples.dipoles_e_angstrom,
        expected_dipoles,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.estimate.hessian.raw_hessian_eV_per_angstrom2,
        expected_estimate.hessian.raw_hessian_eV_per_angstrom2,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.estimate.dipole_derivative_3n_by_3_au,
        expected_estimate.dipole_derivative_3n_by_3_au,
        rtol=0.0,
        atol=0.0,
    )


def test_step_series_matches_original_validation_and_table_formulas() -> None:
    results = _collect_series()
    analysis = _analyze_series(results)
    estimates = [item.estimate for item in results]
    h_convergence = summarize_harmonic_ir_convergence(
        estimates,
        GEOMETRY_ANGSTROM,
        H2O_MASSES_U,
    )
    d_convergence = summarize_harmonic_ir_convergence(
        estimates,
        GEOMETRY_ANGSTROM,
        D2O_MASSES_U,
    )
    selected = results[-1].estimate
    expected_modes = {
        "H2O": analyze_harmonic_ir(
            selected.hessian_hartree_per_bohr2,
            selected.dipole_derivative_3n_by_3_au,
            GEOMETRY_ANGSTROM,
            H2O_MASSES_U,
        ),
        "D2O": analyze_harmonic_ir(
            selected.hessian_hartree_per_bohr2,
            selected.dipole_derivative_3n_by_3_au,
            GEOMETRY_ANGSTROM,
            D2O_MASSES_U,
        ),
    }

    def intensity_is_stable(summary) -> bool:
        absolute = summary.ir_intensity_abs_change_km_mol[-1]
        relative = summary.ir_intensity_relative_change[-1]
        return bool(np.all((absolute <= 1.0e-8) | (relative <= 1.0e-8)))

    expected_validation = {
        "tight minimum": True,
        "H2O frequency step stability": bool(
            h_convergence.frequency_max_abs_change_cm1[-1] <= 1.0e-8
        ),
        "D2O frequency step stability": bool(
            d_convergence.frequency_max_abs_change_cm1[-1] <= 1.0e-8
        ),
        "H2O intensity step stability": intensity_is_stable(h_convergence),
        "D2O intensity step stability": intensity_is_stable(d_convergence),
        "mode continuity": bool(
            min(
                h_convergence.minimum_same_index_mode_squared_overlap[-1],
                d_convergence.minimum_same_index_mode_squared_overlap[-1],
            )
            >= 0.999_999
        ),
        "Hessian symmetry": bool(selected.hessian.max_relative_antisymmetry <= 1.0e-10),
        "no significant imaginary modes": bool(
            all(
                np.all(result.frequencies_cm1 >= -10.0)
                for result in expected_modes.values()
            )
        ),
    }
    expected_validation_table = pd.DataFrame(
        {
            "check": list(expected_validation),
            "passed": list(expected_validation.values()),
        }
    )
    expected_convergence_table = pd.DataFrame(
        {
            "isotopologue": ["H2O", "D2O"],
            "max |frequency change|, 0.010→0.005 bohr (cm-1)": [
                h_convergence.frequency_max_abs_change_cm1[-1],
                d_convergence.frequency_max_abs_change_cm1[-1],
            ],
            "max |intensity change|, 0.010→0.005 bohr (km/mol)": [
                h_convergence.ir_intensity_max_abs_change_km_mol[-1],
                d_convergence.ir_intensity_max_abs_change_km_mol[-1],
            ],
            "minimum same-mode overlap": [
                h_convergence.minimum_same_index_mode_squared_overlap[-1],
                d_convergence.minimum_same_index_mode_squared_overlap[-1],
            ],
        }
    )

    assert dict(analysis.validation_checks) == expected_validation
    assert analysis.comparison_reported == all(expected_validation.values())
    assert tuple(analysis.estimates_by_step_bohr) == (0.020, 0.010, 0.005)
    assert tuple(analysis.samples_by_step_bohr) == (0.020, 0.010, 0.005)
    assert analysis.selected_estimate is selected
    pdt.assert_frame_equal(analysis.validation_table, expected_validation_table)
    pdt.assert_frame_equal(analysis.convergence_table, expected_convergence_table)
    assert tuple(analysis.displacement_table.columns) == (
        "step_bohr",
        "step_angstrom",
        "structures_in_call",
        "max_realized_step_relative_error",
        "raw_H_max_antisymmetry_relative",
    )


def test_every_series_threshold_is_required_and_controls_its_check() -> None:
    signature = inspect.signature(analyze_harmonic_step_series)
    threshold_parameters = (
        "frequency_step_tolerance_cm1",
        "intensity_step_abs_tolerance_km_mol",
        "intensity_step_rel_tolerance",
        "mode_overlap_min",
        "hessian_antisymmetry_rel_max",
        "imaginary_floor_cm1",
    )
    assert all(
        signature.parameters[name].default is inspect.Parameter.empty
        for name in threshold_parameters
    )

    results = _collect_series()
    baseline = _analyze_series(results)
    failed_minimum = _analyze_series(results, minimum_passed=False)
    assert baseline.validation_checks["tight minimum"]
    assert not failed_minimum.validation_checks["tight minimum"]
    assert not failed_minimum.comparison_reported
    with pytest.raises(ValueError, match="between zero and one"):
        _analyze_series(results, mode_overlap_min=1.01)


def test_mode_comparison_table_matches_original_loop_and_column_order() -> None:
    artifact_root = PART_DIR / "reference" / "artifacts"
    references = {
        "H2O": load_psi4_b973c_ir_artifact(
            artifact_root / "h2o",
            frequency_tolerance_cm1=2.0,
        ),
        "D2O": load_psi4_b973c_ir_artifact(
            artifact_root / "d2o",
            frequency_tolerance_cm1=2.0,
        ),
    }
    model_analyses = {
        label: analyze_harmonic_ir(
            reference.hessian_hartree_per_bohr2,
            reference.dipole_derivative_3n_by_3_au,
            reference.geometry_angstrom,
            reference.masses_u,
        )
        for label, reference in references.items()
    }
    observed = load_experimental_water_fundamentals().set_index(
        ["isotopologue", "mode"]
    )

    actual = build_harmonic_mode_comparison_table(
        geometry_angstrom=references["H2O"].geometry_angstrom,
        atomic_numbers=references["H2O"].atomic_numbers,
        isotopologues=tuple(
            (label, references[label].masses_u) for label in ("H2O", "D2O")
        ),
        model_analyses_by_isotopologue=model_analyses,
        references_by_isotopologue=references,
        observed_by_mode=observed,
        mode_order=MODE_ORDER,
        mode_numbers=MODE_NUMBERS,
    )

    # Reproduce the original notebook loop independently.
    expected_rows = []
    for label in ("H2O", "D2O"):
        model_result = model_analyses[label]
        model_labels = label_water_monomer_modes(
            references["H2O"].geometry_angstrom,
            references["H2O"].atomic_numbers,
            references[label].masses_u,
            model_result.mass_weighted_modes,
        )
        reference_labels = reference_water_monomer_mode_labels(references[label])
        for mode in MODE_ORDER:
            model_index = model_labels.index(mode)
            reference_index = reference_labels.index(mode)
            expected_rows.append(
                {
                    "system": label,
                    "mode": f"ν{MODE_NUMBERS[mode]} {mode.replace('_', ' ')}",
                    "AIMNet+Coulomb+D3_harmonic_cm-1": (
                        model_result.frequencies_cm1[model_index]
                    ),
                    "AIMNet_point_charge_IR_km_mol": (
                        model_result.ir_intensities_km_mol[model_index]
                    ),
                    "B97-3c_harmonic_cm-1": (
                        references[label].frequencies_cm1[reference_index]
                    ),
                    "B97-3c_IR_intensity_km_mol": (
                        references[label].ir_intensities_km_mol[reference_index]
                    ),
                    "observed_gas_cm-1": float(
                        observed.loc[(label, mode), "wavenumber_cm1"]
                    ),
                }
            )
    expected = pd.DataFrame(expected_rows)
    expected["AIMNet+Coulomb+D3_minus_B97-3c_cm-1"] = (
        expected["AIMNet+Coulomb+D3_harmonic_cm-1"] - expected["B97-3c_harmonic_cm-1"]
    )
    expected["B97-3c_minus_observed_cm-1"] = (
        expected["B97-3c_harmonic_cm-1"] - expected["observed_gas_cm-1"]
    )
    pdt.assert_frame_equal(actual, expected)


def test_archive_array_names_order_values_and_sample_dtypes_are_unchanged() -> None:
    results = _collect_series()
    analysis = _analyze_series(results)
    minimum_forces = np.arange(9, dtype=np.float32).reshape(3, 3)
    arrays = build_harmonic_archive_arrays(
        geometry_angstrom=GEOMETRY_ANGSTROM,
        atomic_numbers=ATOMIC_NUMBERS,
        isotopologues=ISOTOPOLOGUES,
        minimum_forces_eV_per_angstrom=minimum_forces,
        selected_step_bohr=0.005,
        selected_estimate=analysis.selected_estimate,
        mode_analyses_by_isotopologue=analysis.mode_analyses_by_isotopologue,
        displacement_results=results,
    )

    base_keys = (
        "geometry_angstrom",
        "atomic_numbers",
        "H2O_masses_u",
        "D2O_masses_u",
        "minimum_forces_eV_per_angstrom",
        "selected_step_bohr",
        "hessian_raw_eV_per_angstrom2",
        "hessian_eV_per_angstrom2",
        "hessian_hartree_per_bohr2",
        "dipole_derivative_3n_by_3_au",
        "H2O_frequencies_cm1",
        "D2O_frequencies_cm1",
        "H2O_ir_intensities_km_mol",
        "D2O_ir_intensities_km_mol",
        "H2O_mass_weighted_modes",
        "D2O_mass_weighted_modes",
    )
    sample_keys = tuple(
        f"step_{step_label}_{field}"
        for step_label in ("0p020", "0p010", "0p005")
        for field in (
            "positions_angstrom",
            "forces_eV_per_angstrom",
            "charges_e",
            "dipoles_e_angstrom",
        )
    )
    assert tuple(arrays) == base_keys + sample_keys
    np.testing.assert_array_equal(arrays["geometry_angstrom"], GEOMETRY_ANGSTROM)
    np.testing.assert_array_equal(arrays["atomic_numbers"], ATOMIC_NUMBERS)
    np.testing.assert_array_equal(
        arrays["minimum_forces_eV_per_angstrom"], minimum_forces
    )
    assert arrays["minimum_forces_eV_per_angstrom"].dtype == np.float32
    assert arrays["selected_step_bohr"].shape == ()
    np.testing.assert_array_equal(
        arrays["step_0p005_positions_angstrom"],
        results[-1].samples.positions_angstrom,
    )


def test_helpers_reject_malformed_samples_and_ambiguous_archive_step_names() -> None:
    step_angstrom, positions, forces, charges = _quadratic_outputs(0.010)
    with pytest.raises(ValueError, match="positions_angstrom must have shape"):
        collect_harmonic_displacement_result(
            step_bohr=0.010,
            step_angstrom=step_angstrom,
            n_atoms=3,
            positions_angstrom=positions[:-1],
            forces_eV_per_angstrom=forces,
            charges_e=charges,
            dipole_origin_atom_index=0,
            neutral_tolerance_e=1.0e-12,
        )

    results = _collect_series()
    analysis = _analyze_series(results)
    colliding = HarmonicDisplacementResult(
        step_bohr=0.020_4,
        step_angstrom=results[0].step_angstrom,
        structures_in_call=results[0].structures_in_call,
        max_realized_step_relative_error=results[0].max_realized_step_relative_error,
        estimate=results[0].estimate,
        samples=results[0].samples,
    )
    with pytest.raises(ValueError, match="collide"):
        build_harmonic_archive_arrays(
            geometry_angstrom=GEOMETRY_ANGSTROM,
            atomic_numbers=ATOMIC_NUMBERS,
            isotopologues=ISOTOPOLOGUES,
            minimum_forces_eV_per_angstrom=np.zeros((3, 3)),
            selected_step_bohr=0.005,
            selected_estimate=analysis.selected_estimate,
            mode_analyses_by_isotopologue=analysis.mode_analyses_by_isotopologue,
            displacement_results=(results[0], colliding),
        )
