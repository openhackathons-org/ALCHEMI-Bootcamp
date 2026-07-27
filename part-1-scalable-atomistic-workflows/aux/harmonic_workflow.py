"""Notebook-facing post-processing for the Part 1 harmonic IR workflow.

This module hides array reshaping, table assembly, and archive bookkeeping.
It deliberately does not import ALCHEMI Toolkit or run models, neighbor lists,
relaxation, or dynamics.  The notebook remains responsible for those calls and
passes every scientific threshold explicitly to :func:`analyze_harmonic_step_series`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .harmonic_ir import (
    HarmonicIRConvergence,
    HarmonicIRFiniteDifferenceEstimate,
    HarmonicIRModeAnalysis,
    analyze_harmonic_ir,
    assemble_harmonic_ir_finite_difference,
    molecular_dipoles_from_atomic_predictions,
    summarize_harmonic_ir_convergence,
)
from .reference import (
    HarmonicIRReference,
    label_water_monomer_modes,
    reference_water_monomer_mode_labels,
)


_DISPLACEMENT_COLUMNS = (
    "step_bohr",
    "step_angstrom",
    "structures_in_call",
    "max_realized_step_relative_error",
    "raw_H_max_antisymmetry_relative",
)
_MODE_COMPARISON_BASE_COLUMNS = (
    "system",
    "mode",
    "AIMNet+Coulomb+D3_harmonic_cm-1",
    "AIMNet_point_charge_IR_km_mol",
    "B97-3c_harmonic_cm-1",
    "B97-3c_IR_intensity_km_mol",
    "observed_gas_cm-1",
)
_MODEL_MINUS_REFERENCE_COLUMN = "AIMNet+Coulomb+D3_minus_B97-3c_cm-1"
_REFERENCE_MINUS_OBSERVED_COLUMN = "B97-3c_minus_observed_cm-1"
_MODE_COMPARISON_COLUMNS = (
    *_MODE_COMPARISON_BASE_COLUMNS,
    _MODEL_MINUS_REFERENCE_COLUMN,
    _REFERENCE_MINUS_OBSERVED_COLUMN,
)
_SAMPLE_ARCHIVE_FIELDS = (
    "positions_angstrom",
    "forces_eV_per_angstrom",
    "charges_e",
    "dipoles_e_angstrom",
)


def _numeric_array(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{field} must be a numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} contains non-finite values")
    return array


def _readonly_copy(value: Any, *, field: str) -> np.ndarray:
    result = np.array(_numeric_array(value, field=field), copy=True)
    result.setflags(write=False)
    return result


def _positive_finite(value: float, *, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _nonnegative_finite(value: float, *, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _atom_count(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_atoms must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError("n_atoms must be positive")
    return result


def _reshape_vector_samples(
    value: Any,
    *,
    field: str,
    n_structures: int,
    n_atoms: int,
) -> np.ndarray:
    array = _numeric_array(value, field=field)
    shaped = (n_structures, n_atoms, 3)
    flattened = (n_structures * n_atoms, 3)
    if array.shape == shaped:
        result = array
    elif array.shape == flattened:
        result = array.reshape(shaped)
    else:
        raise ValueError(
            f"{field} must have shape {shaped} or {flattened}; got {array.shape}"
        )
    return _readonly_copy(result, field=field)


def _reshape_scalar_samples(
    value: Any,
    *,
    field: str,
    n_structures: int,
    n_atoms: int,
) -> np.ndarray:
    array = _numeric_array(value, field=field)
    accepted = {
        (n_structures, n_atoms),
        (n_structures, n_atoms, 1),
        (n_structures * n_atoms,),
        (n_structures * n_atoms, 1),
    }
    if array.shape not in accepted:
        expected = sorted(str(shape) for shape in accepted)
        raise ValueError(
            f"{field} must have one scalar per atom in every structure; "
            f"expected one of {expected}, got {array.shape}"
        )
    return _readonly_copy(
        array.reshape(n_structures, n_atoms),
        field=field,
    )


@dataclass(frozen=True)
class HarmonicDisplacementSamples:
    """Model outputs for all ``+/-`` Cartesian displacements at one step."""

    positions_angstrom: np.ndarray
    forces_eV_per_angstrom: np.ndarray
    charges_e: np.ndarray
    dipoles_e_angstrom: np.ndarray

    def as_archive_mapping(self) -> Mapping[str, np.ndarray]:
        """Return the four arrays under the existing NPZ field names."""

        return MappingProxyType(
            {
                "positions_angstrom": self.positions_angstrom,
                "forces_eV_per_angstrom": self.forces_eV_per_angstrom,
                "charges_e": self.charges_e,
                "dipoles_e_angstrom": self.dipoles_e_angstrom,
            }
        )


@dataclass(frozen=True)
class HarmonicDisplacementResult:
    """Finite-difference estimate and inspectable samples for one step."""

    step_bohr: float
    step_angstrom: float
    structures_in_call: int
    max_realized_step_relative_error: float
    estimate: HarmonicIRFiniteDifferenceEstimate
    samples: HarmonicDisplacementSamples

    def table_row(self) -> dict[str, float | int]:
        """Return one row with the notebook's established column names."""

        return {
            "step_bohr": self.step_bohr,
            "step_angstrom": self.step_angstrom,
            "structures_in_call": self.structures_in_call,
            "max_realized_step_relative_error": (self.max_realized_step_relative_error),
            "raw_H_max_antisymmetry_relative": (
                self.estimate.hessian.max_relative_antisymmetry
            ),
        }


@dataclass(frozen=True)
class HarmonicStepSeriesAnalysis:
    """Post-processed displacement series, mode results, and check tables.

    Mapping order follows the explicit ``isotopologues`` sequence passed to
    :func:`analyze_harmonic_step_series`.  DataFrame row and column order is
    stable and matches the original notebook calculations.
    """

    displacement_results: tuple[HarmonicDisplacementResult, ...]
    displacement_table: pd.DataFrame
    estimates_by_step_bohr: Mapping[float, HarmonicIRFiniteDifferenceEstimate]
    samples_by_step_bohr: Mapping[float, Mapping[str, np.ndarray]]
    convergence_by_isotopologue: Mapping[str, HarmonicIRConvergence]
    mode_analyses_by_isotopologue: Mapping[str, HarmonicIRModeAnalysis]
    selected_estimate: HarmonicIRFiniteDifferenceEstimate
    validation_checks: Mapping[str, bool]
    validation_table: pd.DataFrame
    convergence_table: pd.DataFrame
    comparison_reported: bool


def collect_harmonic_displacement_result(
    *,
    step_bohr: float,
    step_angstrom: float,
    n_atoms: int,
    positions_angstrom: Any,
    forces_eV_per_angstrom: Any,
    charges_e: Any,
    dipole_origin_atom_index: int,
    neutral_tolerance_e: float,
) -> HarmonicDisplacementResult:
    """Reshape one complete model call and assemble its finite differences.

    The input order must be all ``+`` displacements followed by all ``-``
    displacements, with Cartesian coordinates ordered atom-major as produced by
    :func:`aux.harmonic_ir.symmetric_cartesian_displacements`.  For ``N``
    atoms, each half contains ``3N`` structures.

    ``dipole_origin_atom_index`` and ``neutral_tolerance_e`` are required so the
    notebook continues to show both scientific choices.  No pass/fail
    tolerance is applied here.
    """

    step_bohr_value = _positive_finite(step_bohr, field="step_bohr")
    step_angstrom_value = _positive_finite(
        step_angstrom,
        field="step_angstrom",
    )
    atom_count = _atom_count(n_atoms)
    if isinstance(dipole_origin_atom_index, (bool, np.bool_)) or not isinstance(
        dipole_origin_atom_index, (int, np.integer)
    ):
        raise TypeError("dipole_origin_atom_index must be an integer")
    origin_index = int(dipole_origin_atom_index)
    if not 0 <= origin_index < atom_count:
        raise ValueError("dipole_origin_atom_index is outside the atom range")
    neutrality = _nonnegative_finite(
        neutral_tolerance_e,
        field="neutral_tolerance_e",
    )

    n_coordinates = 3 * atom_count
    n_structures = 2 * n_coordinates
    positions = _reshape_vector_samples(
        positions_angstrom,
        field="positions_angstrom",
        n_structures=n_structures,
        n_atoms=atom_count,
    )
    forces = _reshape_vector_samples(
        forces_eV_per_angstrom,
        field="forces_eV_per_angstrom",
        n_structures=n_structures,
        n_atoms=atom_count,
    )
    charges = _reshape_scalar_samples(
        charges_e,
        field="charges_e",
        n_structures=n_structures,
        n_atoms=atom_count,
    )
    dipoles = molecular_dipoles_from_atomic_predictions(
        positions,
        charges,
        origin_angstrom=positions[:, origin_index, :],
        neutral_tolerance_e=neutrality,
    )

    stored_positions = positions.astype(np.float64, copy=False)
    plus_positions = stored_positions[:n_coordinates]
    minus_positions = stored_positions[n_coordinates:]
    coordinate = np.arange(n_coordinates)
    realized_half_steps = 0.5 * (
        plus_positions.reshape(n_coordinates, n_coordinates)[coordinate, coordinate]
        - minus_positions.reshape(n_coordinates, n_coordinates)[coordinate, coordinate]
    )
    max_step_relative_error = float(
        np.max(np.abs(realized_half_steps - step_angstrom_value)) / step_angstrom_value
    )

    estimate = assemble_harmonic_ir_finite_difference(
        forces_plus_eV_per_angstrom=forces[:n_coordinates],
        forces_minus_eV_per_angstrom=forces[n_coordinates:],
        dipoles_plus_e_angstrom=dipoles[:n_coordinates],
        dipoles_minus_e_angstrom=dipoles[n_coordinates:],
        step_angstrom=step_angstrom_value,
    )
    return HarmonicDisplacementResult(
        step_bohr=step_bohr_value,
        step_angstrom=step_angstrom_value,
        structures_in_call=n_structures,
        max_realized_step_relative_error=max_step_relative_error,
        estimate=estimate,
        samples=HarmonicDisplacementSamples(
            positions_angstrom=positions,
            forces_eV_per_angstrom=forces,
            charges_e=charges,
            dipoles_e_angstrom=dipoles,
        ),
    )


def _ordered_isotopologues(
    isotopologues: Sequence[tuple[str, Any]],
    *,
    n_atoms: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    try:
        pairs = tuple(isotopologues)
    except TypeError as exc:
        raise TypeError("isotopologues must be an ordered sequence") from exc
    if not pairs:
        raise ValueError("isotopologues must not be empty")

    ordered: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(
                "each isotopologue must be a (label, masses_u) pair; "
                f"item {index} is invalid"
            )
        label, masses_value = pair
        if not isinstance(label, str) or not label:
            raise TypeError("each isotopologue label must be a non-empty string")
        if label in seen:
            raise ValueError(f"duplicate isotopologue label: {label!r}")
        seen.add(label)
        masses = np.asarray(
            _numeric_array(masses_value, field=f"masses for {label}"),
            dtype=np.float64,
        )
        if masses.shape != (n_atoms,):
            raise ValueError(f"masses for {label} must have shape ({n_atoms},)")
        if np.any(masses <= 0.0):
            raise ValueError(f"masses for {label} must be positive")
        masses = np.array(masses, copy=True)
        masses.setflags(write=False)
        ordered.append((label, masses))
    return tuple(ordered)


def analyze_harmonic_step_series(
    displacement_results: Sequence[HarmonicDisplacementResult],
    *,
    geometry_angstrom: Any,
    isotopologues: Sequence[tuple[str, Any]],
    selected_step_bohr: float,
    minimum_passed: bool,
    frequency_step_tolerance_cm1: float,
    intensity_step_abs_tolerance_km_mol: float,
    intensity_step_rel_tolerance: float,
    mode_overlap_min: float,
    hessian_antisymmetry_rel_max: float,
    imaginary_floor_cm1: float,
) -> HarmonicStepSeriesAnalysis:
    """Analyze a displacement series using only caller-supplied criteria.

    Every pass/fail number is a required argument.  The function reproduces
    the original notebook's frequency, intensity, same-index mode-overlap,
    Hessian-symmetry, and imaginary-frequency checks without choosing or
    changing any threshold.
    """

    results = tuple(displacement_results)
    if len(results) < 2:
        raise ValueError("at least two displacement results are required")
    if not all(isinstance(item, HarmonicDisplacementResult) for item in results):
        raise TypeError(
            "displacement_results must contain HarmonicDisplacementResult instances"
        )
    step_keys = tuple(item.step_bohr for item in results)
    if len(set(step_keys)) != len(step_keys):
        raise ValueError("displacement step_bohr values must be unique")

    geometry = np.asarray(
        _numeric_array(geometry_angstrom, field="geometry_angstrom"),
        dtype=np.float64,
    )
    if geometry.ndim != 2 or geometry.shape[1] != 3 or geometry.shape[0] < 2:
        raise ValueError("geometry_angstrom must have shape (n_atoms, 3), n_atoms >= 2")
    n_atoms = int(geometry.shape[0])
    for item in results:
        expected_shape = (2 * 3 * n_atoms, n_atoms, 3)
        if item.samples.positions_angstrom.shape != expected_shape:
            raise ValueError(
                "a displacement result does not match geometry_angstrom; "
                f"expected sample shape {expected_shape}"
            )
    isotope_pairs = _ordered_isotopologues(isotopologues, n_atoms=n_atoms)

    selected_step = _positive_finite(
        selected_step_bohr,
        field="selected_step_bohr",
    )
    estimates = {item.step_bohr: item.estimate for item in results}
    if selected_step not in estimates:
        raise ValueError("selected_step_bohr is not present in displacement_results")
    if not isinstance(minimum_passed, (bool, np.bool_)):
        raise TypeError("minimum_passed must be boolean")

    frequency_tolerance = _nonnegative_finite(
        frequency_step_tolerance_cm1,
        field="frequency_step_tolerance_cm1",
    )
    intensity_absolute_tolerance = _nonnegative_finite(
        intensity_step_abs_tolerance_km_mol,
        field="intensity_step_abs_tolerance_km_mol",
    )
    intensity_relative_tolerance = _nonnegative_finite(
        intensity_step_rel_tolerance,
        field="intensity_step_rel_tolerance",
    )
    overlap_minimum = float(mode_overlap_min)
    if (
        not np.isfinite(overlap_minimum)
        or overlap_minimum < 0.0
        or overlap_minimum > 1.0
    ):
        raise ValueError("mode_overlap_min must be finite and between zero and one")
    antisymmetry_maximum = _nonnegative_finite(
        hessian_antisymmetry_rel_max,
        field="hessian_antisymmetry_rel_max",
    )
    imaginary_floor = float(imaginary_floor_cm1)
    if not np.isfinite(imaginary_floor):
        raise ValueError("imaginary_floor_cm1 must be finite")

    ordered_estimates = [item.estimate for item in results]
    convergence: dict[str, HarmonicIRConvergence] = {}
    mode_analyses: dict[str, HarmonicIRModeAnalysis] = {}
    selected_estimate = estimates[selected_step]
    for label, masses in isotope_pairs:
        convergence[label] = summarize_harmonic_ir_convergence(
            ordered_estimates,
            geometry,
            masses,
        )
        mode_analyses[label] = analyze_harmonic_ir(
            selected_estimate.hessian_hartree_per_bohr2,
            selected_estimate.dipole_derivative_3n_by_3_au,
            geometry,
            masses,
        )

    validation: dict[str, bool] = {"tight minimum": bool(minimum_passed)}
    for label, _ in isotope_pairs:
        summary = convergence[label]
        validation[f"{label} frequency step stability"] = bool(
            summary.frequency_max_abs_change_cm1[-1] <= frequency_tolerance
        )
    for label, _ in isotope_pairs:
        summary = convergence[label]
        absolute = summary.ir_intensity_abs_change_km_mol[-1]
        relative = summary.ir_intensity_relative_change[-1]
        validation[f"{label} intensity step stability"] = bool(
            np.all(
                (absolute <= intensity_absolute_tolerance)
                | (relative <= intensity_relative_tolerance)
            )
        )
    validation["mode continuity"] = bool(
        min(
            convergence[label].minimum_same_index_mode_squared_overlap[-1]
            for label, _ in isotope_pairs
        )
        >= overlap_minimum
    )
    validation["Hessian symmetry"] = bool(
        selected_estimate.hessian.max_relative_antisymmetry <= antisymmetry_maximum
    )
    validation["no significant imaginary modes"] = bool(
        all(
            np.all(result.frequencies_cm1 >= imaginary_floor)
            for result in mode_analyses.values()
        )
    )

    displacement_table = pd.DataFrame.from_records(
        [item.table_row() for item in results],
        columns=_DISPLACEMENT_COLUMNS,
    )
    validation_table = pd.DataFrame.from_records(
        ((name, passed) for name, passed in validation.items()),
        columns=("check", "passed"),
    )
    ordered_by_size = tuple(
        sorted(results, key=lambda item: item.step_angstrom, reverse=True)
    )
    coarse_step = ordered_by_size[-2].step_bohr
    fine_step = ordered_by_size[-1].step_bohr
    frequency_column = (
        f"max |frequency change|, {coarse_step:.3f}→{fine_step:.3f} bohr (cm-1)"
    )
    intensity_column = (
        f"max |intensity change|, {coarse_step:.3f}→{fine_step:.3f} bohr (km/mol)"
    )
    convergence_table = pd.DataFrame.from_records(
        (
            {
                "isotopologue": label,
                frequency_column: (convergence[label].frequency_max_abs_change_cm1[-1]),
                intensity_column: (
                    convergence[label].ir_intensity_max_abs_change_km_mol[-1]
                ),
                "minimum same-mode overlap": (
                    convergence[label].minimum_same_index_mode_squared_overlap[-1]
                ),
            }
            for label, _ in isotope_pairs
        ),
        columns=(
            "isotopologue",
            frequency_column,
            intensity_column,
            "minimum same-mode overlap",
        ),
    )
    sample_mapping = {
        item.step_bohr: item.samples.as_archive_mapping() for item in results
    }
    return HarmonicStepSeriesAnalysis(
        displacement_results=results,
        displacement_table=displacement_table,
        estimates_by_step_bohr=MappingProxyType(estimates),
        samples_by_step_bohr=MappingProxyType(sample_mapping),
        convergence_by_isotopologue=MappingProxyType(convergence),
        mode_analyses_by_isotopologue=MappingProxyType(mode_analyses),
        selected_estimate=selected_estimate,
        validation_checks=MappingProxyType(validation),
        validation_table=validation_table,
        convergence_table=convergence_table,
        comparison_reported=bool(all(validation.values())),
    )


def build_harmonic_mode_comparison_table(
    *,
    geometry_angstrom: Any,
    atomic_numbers: Any,
    isotopologues: Sequence[tuple[str, Any]],
    model_analyses_by_isotopologue: Mapping[str, HarmonicIRModeAnalysis],
    references_by_isotopologue: Mapping[str, HarmonicIRReference],
    observed_by_mode: pd.DataFrame,
    mode_order: Sequence[str],
    mode_numbers: Mapping[str, int],
) -> pd.DataFrame:
    """Match mode characters and build the established seven-value table.

    ``isotopologues`` and ``mode_order`` determine row order.  ``mode_numbers``
    is required because spectroscopic mode numbering is a presentation choice,
    not something this helper should infer.  The two difference columns use
    the exact formulas from the original notebook.
    """

    geometry = np.asarray(
        _numeric_array(geometry_angstrom, field="geometry_angstrom"),
        dtype=np.float64,
    )
    if geometry.ndim != 2 or geometry.shape[1] != 3 or geometry.shape[0] < 2:
        raise ValueError("geometry_angstrom must have shape (n_atoms, 3), n_atoms >= 2")
    numbers = _numeric_array(atomic_numbers, field="atomic_numbers")
    if numbers.shape != (geometry.shape[0],):
        raise ValueError("atomic_numbers must contain one value per atom")
    isotope_pairs = _ordered_isotopologues(
        isotopologues,
        n_atoms=int(geometry.shape[0]),
    )
    if not isinstance(model_analyses_by_isotopologue, Mapping):
        raise TypeError("model_analyses_by_isotopologue must be a mapping")
    if not isinstance(references_by_isotopologue, Mapping):
        raise TypeError("references_by_isotopologue must be a mapping")
    if not isinstance(observed_by_mode, pd.DataFrame):
        raise TypeError("observed_by_mode must be a pandas DataFrame")
    if "wavenumber_cm1" not in observed_by_mode.columns:
        raise ValueError("observed_by_mode must contain wavenumber_cm1")

    modes = tuple(mode_order)
    if not modes or any(not isinstance(mode, str) or not mode for mode in modes):
        raise ValueError("mode_order must contain non-empty mode names")
    if len(set(modes)) != len(modes):
        raise ValueError("mode_order entries must be unique")
    if not isinstance(mode_numbers, Mapping):
        raise TypeError("mode_numbers must be a mapping")
    missing_numbers = [mode for mode in modes if mode not in mode_numbers]
    if missing_numbers:
        raise ValueError(f"mode_numbers is missing {missing_numbers}")
    for mode in modes:
        number = mode_numbers[mode]
        if isinstance(number, (bool, np.bool_)) or not isinstance(
            number,
            (int, np.integer),
        ):
            raise TypeError(f"mode number for {mode!r} must be an integer")

    rows: list[dict[str, Any]] = []
    for label, masses in isotope_pairs:
        try:
            model_result = model_analyses_by_isotopologue[label]
            reference = references_by_isotopologue[label]
        except KeyError as exc:
            raise ValueError(
                f"missing model analysis or reference for {label}"
            ) from exc
        if not isinstance(model_result, HarmonicIRModeAnalysis):
            raise TypeError(f"model analysis for {label} has the wrong type")
        if not isinstance(reference, HarmonicIRReference):
            raise TypeError(f"reference for {label} has the wrong type")
        model_labels = label_water_monomer_modes(
            geometry,
            numbers,
            masses,
            model_result.mass_weighted_modes,
        )
        reference_labels = reference_water_monomer_mode_labels(reference)
        for mode in modes:
            if mode not in model_labels or mode not in reference_labels:
                raise ValueError(f"mode {mode!r} could not be matched for {label}")
            observed_value = observed_by_mode.loc[(label, mode), "wavenumber_cm1"]
            if not np.isscalar(observed_value):
                raise ValueError(
                    f"observed_by_mode must have one row for {(label, mode)!r}"
                )
            observed_cm1 = float(observed_value)
            if not np.isfinite(observed_cm1):
                raise ValueError(
                    f"observed wavenumber is not finite for {label} {mode}"
                )
            model_index = model_labels.index(mode)
            reference_index = reference_labels.index(mode)
            rows.append(
                {
                    "system": label,
                    "mode": f"ν{int(mode_numbers[mode])} {mode.replace('_', ' ')}",
                    "AIMNet+Coulomb+D3_harmonic_cm-1": float(
                        model_result.frequencies_cm1[model_index]
                    ),
                    "AIMNet_point_charge_IR_km_mol": float(
                        model_result.ir_intensities_km_mol[model_index]
                    ),
                    "B97-3c_harmonic_cm-1": float(
                        reference.frequencies_cm1[reference_index]
                    ),
                    "B97-3c_IR_intensity_km_mol": float(
                        reference.ir_intensities_km_mol[reference_index]
                    ),
                    "observed_gas_cm-1": observed_cm1,
                }
            )

    table = pd.DataFrame.from_records(
        rows,
        columns=_MODE_COMPARISON_BASE_COLUMNS,
    )
    table[_MODEL_MINUS_REFERENCE_COLUMN] = (
        table["AIMNet+Coulomb+D3_harmonic_cm-1"] - table["B97-3c_harmonic_cm-1"]
    )
    table[_REFERENCE_MINUS_OBSERVED_COLUMN] = (
        table["B97-3c_harmonic_cm-1"] - table["observed_gas_cm-1"]
    )
    return table


def empty_harmonic_mode_comparison_table() -> pd.DataFrame:
    """Return an empty comparison table with the stable saved-file columns."""

    return pd.DataFrame(columns=_MODE_COMPARISON_COLUMNS)


def build_harmonic_archive_arrays(
    *,
    geometry_angstrom: Any,
    atomic_numbers: Any,
    isotopologues: Sequence[tuple[str, Any]],
    minimum_forces_eV_per_angstrom: Any,
    selected_step_bohr: float,
    selected_estimate: HarmonicIRFiniteDifferenceEstimate,
    mode_analyses_by_isotopologue: Mapping[str, HarmonicIRModeAnalysis],
    displacement_results: Sequence[HarmonicDisplacementResult],
) -> dict[str, np.ndarray]:
    """Assemble the existing NPZ payload without writing a file.

    Array names and insertion order match the original notebook.  Displacement
    samples retain their source dtype.  The caller remains responsible for the
    visible ``np.savez_compressed`` call and output path.
    """

    geometry = np.asarray(
        _numeric_array(geometry_angstrom, field="geometry_angstrom"),
    )
    if geometry.ndim != 2 or geometry.shape[1] != 3 or geometry.shape[0] < 2:
        raise ValueError("geometry_angstrom must have shape (n_atoms, 3), n_atoms >= 2")
    n_atoms = int(geometry.shape[0])
    numbers = _numeric_array(atomic_numbers, field="atomic_numbers")
    if numbers.shape != (n_atoms,):
        raise ValueError("atomic_numbers must contain one value per atom")
    isotope_pairs = _ordered_isotopologues(isotopologues, n_atoms=n_atoms)
    minimum_forces = _numeric_array(
        minimum_forces_eV_per_angstrom,
        field="minimum_forces_eV_per_angstrom",
    )
    if minimum_forces.shape != (n_atoms, 3):
        raise ValueError("minimum_forces_eV_per_angstrom must have shape (n_atoms, 3)")
    selected_step = _positive_finite(
        selected_step_bohr,
        field="selected_step_bohr",
    )
    if not isinstance(selected_estimate, HarmonicIRFiniteDifferenceEstimate):
        raise TypeError("selected_estimate has the wrong type")
    if not isinstance(mode_analyses_by_isotopologue, Mapping):
        raise TypeError("mode_analyses_by_isotopologue must be a mapping")
    results = tuple(displacement_results)
    if not results or not all(
        isinstance(item, HarmonicDisplacementResult) for item in results
    ):
        raise TypeError(
            "displacement_results must contain HarmonicDisplacementResult instances"
        )

    arrays: dict[str, np.ndarray] = {
        "geometry_angstrom": np.array(geometry, copy=True),
        "atomic_numbers": np.array(numbers, copy=True),
    }
    for label, masses in isotope_pairs:
        arrays[f"{label}_masses_u"] = np.array(masses, copy=True)
    arrays["minimum_forces_eV_per_angstrom"] = np.array(
        minimum_forces,
        copy=True,
    )
    arrays["selected_step_bohr"] = np.array(selected_step)
    arrays["hessian_raw_eV_per_angstrom2"] = np.array(
        selected_estimate.hessian.raw_hessian_eV_per_angstrom2,
        copy=True,
    )
    arrays["hessian_eV_per_angstrom2"] = np.array(
        selected_estimate.hessian.hessian_eV_per_angstrom2,
        copy=True,
    )
    arrays["hessian_hartree_per_bohr2"] = np.array(
        selected_estimate.hessian_hartree_per_bohr2,
        copy=True,
    )
    arrays["dipole_derivative_3n_by_3_au"] = np.array(
        selected_estimate.dipole_derivative_3n_by_3_au,
        copy=True,
    )

    for field, attribute in (
        ("frequencies_cm1", "frequencies_cm1"),
        ("ir_intensities_km_mol", "ir_intensities_km_mol"),
        ("mass_weighted_modes", "mass_weighted_modes"),
    ):
        for label, _ in isotope_pairs:
            try:
                analysis = mode_analyses_by_isotopologue[label]
            except KeyError as exc:
                raise ValueError(f"missing mode analysis for {label}") from exc
            if not isinstance(analysis, HarmonicIRModeAnalysis):
                raise TypeError(f"mode analysis for {label} has the wrong type")
            arrays[f"{label}_{field}"] = np.array(
                getattr(analysis, attribute),
                copy=True,
            )

    seen_step_labels: set[str] = set()
    for result in results:
        expected_shape = (2 * 3 * n_atoms, n_atoms, 3)
        if result.samples.positions_angstrom.shape != expected_shape:
            raise ValueError(
                "a displacement result does not match geometry_angstrom; "
                f"expected sample shape {expected_shape}"
            )
        step_label = f"{result.step_bohr:.3f}".replace(".", "p")
        if step_label in seen_step_labels:
            raise ValueError(
                "displacement steps collide after three-decimal archive formatting"
            )
        seen_step_labels.add(step_label)
        sample_mapping = result.samples.as_archive_mapping()
        for name in _SAMPLE_ARCHIVE_FIELDS:
            arrays[f"step_{step_label}_{name}"] = np.array(
                sample_mapping[name],
                copy=True,
            )
    return arrays


__all__ = [
    "HarmonicDisplacementResult",
    "HarmonicDisplacementSamples",
    "HarmonicStepSeriesAnalysis",
    "analyze_harmonic_step_series",
    "build_harmonic_archive_arrays",
    "build_harmonic_mode_comparison_table",
    "collect_harmonic_displacement_result",
    "empty_harmonic_mode_comparison_table",
]
