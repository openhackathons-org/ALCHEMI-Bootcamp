"""Reference harmonic-IR analysis for the water tutorial.

This module deliberately has no Psi4, Torch, Toolkit, or plotting dependency.
It consumes a small, versioned artifact exported from a *completed* Psi4
B97-3c frequency calculation and returns plot-ready arrays and inspectable
mode assignments.  Keeping the reference path separate from the MD runtime
also makes cached results auditable without presenting them as a fresh
calculation.

Artifact format (version 1)
---------------------------
``manifest.json`` and its referenced ``.npz`` file form one immutable bundle::

    {
      "format": {"name": "alchemi.psi4-b97-3c-ir", "version": 1},
      "artifact_id": "h2o-b97-3c-...",
      "engine": {"name": "Psi4", "version": "..."},
      "model_chemistry": {"method": "B97-3c"},
      "molecule": {"label": "H2O", "charge": 0, "multiplicity": 1},
      "arrays": {"file": "ir_arrays.npz", "sha256": "..."},
      "units": {
        "geometry": "angstrom",
        "masses": "unified_atomic_mass_unit",
        "hessian": "hartree_per_bohr2",
        "frequencies": "cm^-1",
        "ir_intensities": "km_per_mol"
      }
    }

The NPZ must contain ``atomic_numbers``, ``geometry_angstrom``, ``masses_u``,
``hessian_hartree_per_bohr2``, ``dipole_derivative_3n_by_3_au``,
``frequencies_cm1``, and ``ir_intensities_km_mol``. Loading recomputes the
vibrational eigensystem from the Cartesian Hessian, checks its frequencies,
and rejects manifests that did not pass the minimum, gradient, Hessian-noise,
topology, and isotope-provenance gates. No pickled arrays are accepted.

The comparison layer never fits a Gaussian or Lorentzian linewidth.  A stick
spectrum is optionally convolved with the *power response of the same discrete
Hann window* used for the finite MD segment.  This makes the envelope a stated
resolution transform, not a model of physical broadening.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


IR_ARTIFACT_SCHEMA = "alchemi.psi4-b97-3c-ir"
IR_ARTIFACT_VERSION = 1

HARTREE_J = 4.359_744_722_207_1e-18
BOHR_M = 5.291_772_109_03e-11
ATOMIC_MASS_UNIT_KG = 1.660_539_066_60e-27
SPEED_OF_LIGHT_CM_S = 2.997_924_58e10
FS_TO_S = 1.0e-15

# Convert sqrt[(Eh / bohr^2) / u] (an angular frequency) to cm^-1.
HESSIAN_EIGENVALUE_TO_CM1 = np.sqrt(
    HARTREE_J / (BOHR_M**2 * ATOMIC_MASS_UNIT_KG)
) / (2.0 * np.pi * SPEED_OF_LIGHT_CM_S)


class IRArtifactError(ValueError):
    """Raised when a cached reference bundle is malformed or inconsistent."""


@dataclass(frozen=True)
class NormalModeSolution:
    """Projected vibrational eigensystem in a mass-weighted basis.

    ``mass_weighted_modes`` has shape ``(n_modes, n_atoms, 3)``.  Each flattened
    row is unit-normalized and mutually orthogonal.  Cartesian displacements
    are obtained by dividing atom ``i`` by ``sqrt(masses_u[i])``.
    """

    frequencies_cm1: np.ndarray
    eigenvalues_hartree_per_bohr2_u: np.ndarray
    mass_weighted_modes: np.ndarray
    external_rank: int


@dataclass(frozen=True)
class HarmonicIRReference:
    """Validated contents of one Psi4 B97-3c harmonic-IR artifact."""

    artifact_id: str
    label: str
    charge: int
    multiplicity: int
    engine_version: str
    atomic_numbers: np.ndarray
    geometry_angstrom: np.ndarray
    masses_u: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    dipole_derivative_3n_by_3_au: np.ndarray
    frequencies_cm1: np.ndarray
    ir_intensities_km_mol: np.ndarray
    mass_weighted_modes: np.ndarray
    manifest: Mapping[str, Any]
    manifest_path: Path | None = None

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def n_modes(self) -> int:
        return int(self.frequencies_cm1.size)


@dataclass(frozen=True)
class SpectralBandSummary:
    """Intensity-weighted location and central 80% span of a spectral band."""

    total_intensity: float
    centroid_cm1: float
    percentile_10_cm1: float
    percentile_90_cm1: float
    width_10_90_cm1: float


@dataclass(frozen=True)
class IRSpectrumComparison:
    """Plot-ready MD curve, raw harmonic sticks, and Hann-resolution envelope."""

    wavenumber_cm1: np.ndarray
    md_intensity: np.ndarray
    md_intensity_normalized: np.ndarray
    stick_wavenumber_cm1: np.ndarray
    stick_intensity_km_mol: np.ndarray
    stick_intensity_normalized: np.ndarray
    reference_envelope: np.ndarray
    reference_envelope_normalized: np.ndarray
    md_summary: SpectralBandSummary
    reference_stick_summary: SpectralBandSummary
    reference_envelope_summary: SpectralBandSummary
    dt_fs: float
    segment_time_fs: float
    segment_frames: int


@dataclass(frozen=True)
class HydrogenBondedOH:
    """One donor O-H coordinate and its geometric acceptor oxygen."""

    oxygen_index: int
    hydrogen_index: int
    acceptor_oxygen_index: int
    hydrogen_acceptor_distance_angstrom: float
    donor_hydrogen_acceptor_angle_deg: float


@dataclass(frozen=True)
class WaterRingModeCharacters:
    """Additive mode-character fractions for a single cyclic water cluster."""

    categories: tuple[str, ...]
    fractions: np.ndarray
    dominant_labels: tuple[str, ...]
    hbonded_oh: tuple[HydrogenBondedOH, ...]
    free_oh: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ModeSubspaceMatch:
    """A near-degenerate block whose individual member labels are non-unique."""

    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    minimum_principal_overlap: float


@dataclass(frozen=True)
class IsotopologueModeMatch:
    """Continuous-mass-path correspondence between two isotope artifacts.

    ``source_to_target[i]`` gives a convenient individual target index.  Within
    a block listed in ``ambiguous_subspaces`` that individual ordering is a
    basis convention; only the source/target *subspaces* are physically robust.
    """

    source_to_target: np.ndarray
    endpoint_squared_overlaps: np.ndarray
    minimum_path_squared_overlaps: np.ndarray
    ambiguous_subspaces: tuple[ModeSubspaceMatch, ...]
    mass_path_fractions: np.ndarray


def _readonly(array: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_method(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IRArtifactError(f"manifest field {field!r} must be an object")
    return value


def _resolve_bundle_file(manifest_path: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IRArtifactError("manifest arrays.file must be a non-empty string")
    candidate = (manifest_path.parent / relative).resolve()
    bundle_root = manifest_path.parent.resolve()
    try:
        candidate.relative_to(bundle_root)
    except ValueError as exc:
        raise IRArtifactError(
            "manifest arrays.file escapes the artifact bundle"
        ) from exc
    if not candidate.is_file():
        raise IRArtifactError(f"artifact array file does not exist: {candidate}")
    return candidate


def _hungarian_minimize(cost: np.ndarray) -> np.ndarray:
    """Return the minimum-cost column for every row (pure NumPy/Python)."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("assignment cost must be a matrix")
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("assignment cost must be non-empty")
    if n_rows > n_cols:
        reverse = _hungarian_minimize(matrix.T)
        result = np.empty(n_rows, dtype=np.int64)
        result.fill(-1)
        for column, row in enumerate(reverse):
            result[row] = column
        if np.any(result < 0):
            raise ValueError("cannot assign every row to a unique column")
        return result

    # Shortest augmenting-path form of the Hungarian algorithm.  The
    # one-based arrays follow the standard derivation and avoid SciPy here.
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)
    for row in range(1, n_rows + 1):
        p[0] = row
        min_value = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, n_cols + 1):
                if used[column]:
                    continue
                reduced = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if reduced < min_value[column]:
                    min_value[column] = reduced
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            if not np.isfinite(delta):
                raise ValueError("assignment cost has no finite solution")
            for column in range(n_cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = np.empty(n_rows, dtype=np.int64)
    for column in range(1, n_cols + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def _maximize_assignment(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("assignment score contains non-finite values")
    return _hungarian_minimize(values.max() - values)


def _external_and_vibrational_bases(
    geometry_angstrom: np.ndarray,
    masses_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    masses = np.asarray(masses_u, dtype=np.float64)
    if geometry.shape != (masses.size, 3):
        raise ValueError("geometry must have shape (len(masses), 3)")
    if masses.size < 2 or np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("at least two finite, positive atomic masses are required")

    centered = geometry - np.average(geometry, axis=0, weights=masses)
    sqrt_mass = np.sqrt(masses)
    columns: list[np.ndarray] = []
    for axis in np.eye(3):
        columns.append((sqrt_mass[:, None] * axis[None, :]).reshape(-1))
    for axis in np.eye(3):
        rotation = np.cross(axis[None, :], centered)
        columns.append((sqrt_mass[:, None] * rotation).reshape(-1))
    raw = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(raw, full_matrices=True)
    threshold = (
        singular[0] * max(raw.shape) * np.finfo(np.float64).eps * 100.0
        if singular.size and singular[0] > 0.0
        else 0.0
    )
    rank = int(np.sum(singular > threshold))
    return left[:, :rank], left[:, rank:]


def solve_vibrational_modes(
    hessian_hartree_per_bohr2: np.ndarray,
    geometry_angstrom: np.ndarray,
    masses_u: np.ndarray,
) -> NormalModeSolution:
    """Project translations/rotations and diagonalize a Cartesian Hessian."""

    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    masses = np.asarray(masses_u, dtype=np.float64)
    degrees = 3 * masses.size
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=np.float64)
    if hessian.shape != (degrees, degrees):
        raise ValueError(f"hessian must have shape ({degrees}, {degrees})")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("hessian contains non-finite values")
    scale = max(float(np.max(np.abs(hessian))), 1.0)
    if np.max(np.abs(hessian - hessian.T)) > 1e-9 * scale:
        raise ValueError("hessian is not symmetric")
    hessian = 0.5 * (hessian + hessian.T)

    repeated_masses = np.repeat(masses, 3)
    inverse_sqrt_mass = 1.0 / np.sqrt(repeated_masses)
    mass_weighted_hessian = (
        inverse_sqrt_mass[:, None]
        * hessian
        * inverse_sqrt_mass[None, :]
    )
    external, vibrational = _external_and_vibrational_bases(geometry, masses)
    reduced = vibrational.T @ mass_weighted_hessian @ vibrational
    eigenvalues, reduced_modes = np.linalg.eigh(0.5 * (reduced + reduced.T))
    modes = (vibrational @ reduced_modes).T.reshape(-1, masses.size, 3)
    frequencies = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * HESSIAN_EIGENVALUE_TO_CM1
    )
    return NormalModeSolution(
        frequencies_cm1=_readonly(frequencies),
        eigenvalues_hartree_per_bohr2_u=_readonly(eigenvalues),
        mass_weighted_modes=_readonly(modes),
        external_rank=int(external.shape[1]),
    )


def load_psi4_b973c_ir_artifact(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    frequency_tolerance_cm1: float = 2.0,
) -> HarmonicIRReference:
    """Load and validate one version-1 Psi4 B97-3c harmonic-IR bundle."""

    manifest_path = Path(path).expanduser()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise IRArtifactError(f"artifact manifest does not exist: {manifest_path}")
    try:
        manifest_object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IRArtifactError(
            f"cannot read artifact manifest: {manifest_path}"
        ) from exc
    manifest = _require_mapping(manifest_object, "<root>")

    format_info = _require_mapping(manifest.get("format"), "format")
    if format_info.get("name") != IR_ARTIFACT_SCHEMA:
        raise IRArtifactError(
            f"unsupported artifact schema {format_info.get('name')!r}; "
            f"expected {IR_ARTIFACT_SCHEMA!r}"
        )
    if format_info.get("version") != IR_ARTIFACT_VERSION:
        raise IRArtifactError(
            f"unsupported artifact version {format_info.get('version')!r}; "
            f"expected {IR_ARTIFACT_VERSION}"
        )

    engine = _require_mapping(manifest.get("engine"), "engine")
    if str(engine.get("name", "")).casefold() != "psi4":
        raise IRArtifactError("reference artifact engine must be Psi4")
    engine_version = str(engine.get("version", "")).strip()
    if not engine_version:
        raise IRArtifactError("manifest engine.version is required")
    chemistry = _require_mapping(
        manifest.get("model_chemistry"), "model_chemistry"
    )
    if _canonical_method(chemistry.get("method")) != "b973c":
        raise IRArtifactError("reference artifact method must be B97-3c")

    expected_units = {
        "geometry": "angstrom",
        "masses": "unified_atomic_mass_unit",
        "hessian": "hartree_per_bohr2",
        "dipole_derivative": "atomic_unit_dipole_per_bohr",
        "frequencies": "cm^-1",
        "ir_intensities": "km_per_mol",
    }
    units = _require_mapping(manifest.get("units"), "units")
    for key, expected in expected_units.items():
        if units.get(key) != expected:
            raise IRArtifactError(
                f"manifest units.{key} must be {expected!r}, got {units.get(key)!r}"
            )

    arrays_info = _require_mapping(manifest.get("arrays"), "arrays")
    arrays_path = _resolve_bundle_file(manifest_path, arrays_info.get("file"))
    expected_checksum = str(arrays_info.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checksum):
        raise IRArtifactError("manifest arrays.sha256 must be a SHA-256 hex digest")
    if verify_checksum:
        actual_checksum = _sha256_file(arrays_path)
        if actual_checksum != expected_checksum:
            raise IRArtifactError(
                "artifact array checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

    required_arrays = {
        "atomic_numbers",
        "geometry_angstrom",
        "masses_u",
        "hessian_hartree_per_bohr2",
        "dipole_derivative_3n_by_3_au",
        "frequencies_cm1",
        "ir_intensities_km_mol",
    }
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            missing = required_arrays.difference(archive.files)
            if missing:
                raise IRArtifactError(
                    f"artifact array file is missing keys: {sorted(missing)}"
                )
            atomic_numbers_raw = np.asarray(archive["atomic_numbers"])
            geometry = np.asarray(archive["geometry_angstrom"], dtype=np.float64)
            masses = np.asarray(archive["masses_u"], dtype=np.float64)
            hessian = np.asarray(
                archive["hessian_hartree_per_bohr2"], dtype=np.float64
            )
            dipole_derivative = np.asarray(
                archive["dipole_derivative_3n_by_3_au"], dtype=np.float64
            )
            frequencies = np.asarray(archive["frequencies_cm1"], dtype=np.float64)
            intensities = np.asarray(
                archive["ir_intensities_km_mol"], dtype=np.float64
            )
            supplied_modes = (
                np.asarray(archive["mass_weighted_modes"], dtype=np.float64)
                if "mass_weighted_modes" in archive.files
                else None
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, IRArtifactError):
            raise
        raise IRArtifactError(f"cannot read artifact arrays: {arrays_path}") from exc

    if atomic_numbers_raw.ndim != 1 or not np.all(
        atomic_numbers_raw == np.round(atomic_numbers_raw)
    ):
        raise IRArtifactError("atomic_numbers must be a one-dimensional integer array")
    atomic_numbers = atomic_numbers_raw.astype(np.int64)
    n_atoms = atomic_numbers.size
    if n_atoms < 2 or np.any(atomic_numbers <= 0):
        raise IRArtifactError("atomic_numbers must contain at least two atoms")
    if geometry.shape != (n_atoms, 3):
        raise IRArtifactError("geometry_angstrom has the wrong shape")
    if masses.shape != (n_atoms,) or np.any(masses <= 0.0):
        raise IRArtifactError("masses_u must contain one positive mass per atom")
    if hessian.shape != (3 * n_atoms, 3 * n_atoms):
        raise IRArtifactError("hessian_hartree_per_bohr2 has the wrong shape")
    if dipole_derivative.shape != (3 * n_atoms, 3):
        raise IRArtifactError("dipole_derivative_3n_by_3_au has the wrong shape")
    if frequencies.ndim != 1 or intensities.shape != frequencies.shape:
        raise IRArtifactError(
            "frequencies and IR intensities must be equal-length vectors"
        )
    for name, values in (
        ("geometry", geometry),
        ("masses", masses),
        ("hessian", hessian),
        ("dipole derivative", dipole_derivative),
        ("frequencies", frequencies),
        ("IR intensities", intensities),
    ):
        if not np.all(np.isfinite(values)):
            raise IRArtifactError(f"artifact {name} contains non-finite values")
    if np.any(intensities < -1e-10):
        raise IRArtifactError("IR intensities must be non-negative")
    intensities = np.maximum(intensities, 0.0)

    try:
        solution = solve_vibrational_modes(hessian, geometry, masses)
    except ValueError as exc:
        raise IRArtifactError(f"invalid artifact Hessian: {exc}") from exc
    if solution.frequencies_cm1.size != frequencies.size:
        raise IRArtifactError(
            "artifact frequency count does not match the projected Hessian: "
            f"{frequencies.size} versus {solution.frequencies_cm1.size}"
        )
    pairing = _hungarian_minimize(
        np.abs(frequencies[:, None] - solution.frequencies_cm1[None, :])
    )
    errors = np.abs(frequencies - solution.frequencies_cm1[pairing])
    if np.max(errors, initial=0.0) > float(frequency_tolerance_cm1):
        worst = int(np.argmax(errors))
        raise IRArtifactError(
            "exported frequencies do not reproduce from the artifact Hessian; "
            f"mode {worst} differs by {errors[worst]:.3f} cm^-1"
        )

    aligned_modes = solution.mass_weighted_modes[pairing]
    if supplied_modes is not None:
        mode_info = _require_mapping(manifest.get("normal_modes"), "normal_modes")
        expected_mode_info = {
            "array": "mass_weighted_modes",
            "convention": "q_equals_sqrt_mass_times_cartesian",
            "normalization": "orthonormal_rows",
            "ordering": "frequencies_and_ir_intensities",
        }
        for key, expected in expected_mode_info.items():
            if mode_info.get(key) != expected:
                raise IRArtifactError(
                    f"manifest normal_modes.{key} must be {expected!r}"
                )
        if supplied_modes.shape != (frequencies.size, n_atoms, 3):
            raise IRArtifactError("mass_weighted_modes has the wrong shape")
        if not np.all(np.isfinite(supplied_modes)):
            raise IRArtifactError("mass_weighted_modes contains non-finite values")
        supplied_flat = supplied_modes.reshape(frequencies.size, -1)
        orthogonality = supplied_flat @ supplied_flat.T
        if np.max(np.abs(orthogonality - np.eye(frequencies.size))) > 1e-6:
            raise IRArtifactError("mass_weighted_modes rows are not orthonormal")
        computed_flat = solution.mass_weighted_modes.reshape(frequencies.size, -1)
        full_subspace_overlap = np.linalg.svd(
            supplied_flat @ computed_flat.T, compute_uv=False
        )
        if np.min(full_subspace_overlap, initial=1.0) < 1.0 - 1e-6:
            raise IRArtifactError(
                "mass_weighted_modes do not span the projected Hessian modes"
            )
        mode_overlap = np.abs(supplied_flat @ computed_flat.T) ** 2
        for mode_index, frequency in enumerate(frequencies):
            compatible = (
                np.abs(solution.frequencies_cm1 - frequency)
                <= float(frequency_tolerance_cm1)
            )
            if not np.any(compatible) or np.sum(
                mode_overlap[mode_index, compatible]
            ) < 1.0 - 1e-5:
                raise IRArtifactError(
                    "mass_weighted_modes ordering is inconsistent with frequencies; "
                    f"mode {mode_index} is not in its Hessian eigenspace"
                )
        aligned_modes = supplied_modes

    molecule = _require_mapping(manifest.get("molecule"), "molecule")
    label = str(molecule.get("label", "")).strip()
    if not label:
        raise IRArtifactError("manifest molecule.label is required")
    artifact_id = str(manifest.get("artifact_id", "")).strip()
    if not artifact_id:
        raise IRArtifactError("manifest artifact_id is required")
    try:
        charge = int(molecule["charge"])
        multiplicity = int(molecule["multiplicity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IRArtifactError(
            "manifest molecule charge and multiplicity must be integers"
        ) from exc
    if multiplicity < 1:
        raise IRArtifactError("manifest molecule.multiplicity must be positive")

    validation = _require_mapping(manifest.get("validation"), "validation")
    if validation.get("status") != "passed" or validation.get(
        "reference_ready"
    ) is not True:
        raise IRArtifactError("reference artifact did not pass its consumer gate")
    if validation.get("is_minimum_within_threshold") is not True or validation.get(
        "significant_imaginary_modes"
    ) != []:
        raise IRArtifactError("reference artifact is not a validated minimum")
    try:
        gradient_max = float(validation["gradient_max_abs_Eh_per_bohr"])
        gradient_limit = float(
            validation["gradient_max_abs_limit_Eh_per_bohr"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IRArtifactError("reference artifact has invalid gradient validation") from exc
    if (
        not np.isfinite(gradient_max)
        or not np.isfinite(gradient_limit)
        or gradient_limit <= 0.0
        or gradient_max > gradient_limit
    ):
        raise IRArtifactError("reference artifact failed its gradient gate")
    try:
        antisymmetry = float(
            validation["raw_hessian_max_antisymmetry_relative"]
        )
        antisymmetry_limit = float(
            validation["raw_hessian_max_antisymmetry_relative_limit"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IRArtifactError(
            "reference artifact has invalid Hessian-consistency validation"
        ) from exc
    if (
        not np.isfinite(antisymmetry)
        or not np.isfinite(antisymmetry_limit)
        or antisymmetry_limit <= 0.0
        or antisymmetry > antisymmetry_limit
    ):
        raise IRArtifactError("reference artifact failed its Hessian-consistency gate")
    for field in (
        "covalent_graph_preserved",
        "optimized_all_oxygens_have_two_hydrogens",
        "same_geometry_hessian_and_dipole_derivative_for_isotopes",
        "changed_entries_are_hydrogen_masses_only",
    ):
        if validation.get(field) is not True:
            raise IRArtifactError(f"reference validation field {field!r} is not true")
    if label in {"h6", "d6"} and validation.get(
        "optimized_is_single_water_ring"
    ) is not True:
        raise IRArtifactError("hexamer reference is not a validated single water ring")

    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    if provenance.get("dispersion") != "D3(BJ)-ATM":
        raise IRArtifactError("reference provenance must declare D3(BJ)-ATM")

    return HarmonicIRReference(
        artifact_id=artifact_id,
        label=label,
        charge=charge,
        multiplicity=multiplicity,
        engine_version=engine_version,
        atomic_numbers=_readonly(atomic_numbers, dtype=np.int64),
        geometry_angstrom=_readonly(geometry),
        masses_u=_readonly(masses),
        hessian_hartree_per_bohr2=_readonly(hessian),
        dipole_derivative_3n_by_3_au=_readonly(dipole_derivative),
        frequencies_cm1=_readonly(frequencies),
        ir_intensities_km_mol=_readonly(intensities),
        mass_weighted_modes=_readonly(aligned_modes),
        manifest=dict(manifest),
        manifest_path=manifest_path,
    )


def independently_max_normalize(*intensities: np.ndarray) -> tuple[np.ndarray, ...]:
    """Normalize each non-negative spectrum to its own maximum.

    Harmonic IR intensities (km/mol) and a classical MD dipole-current PSD do
    not share an absolute ordinate.  Independent normalization therefore
    compares positions and shapes without implying an absolute-intensity test.
    """

    normalized: list[np.ndarray] = []
    for index, values in enumerate(intensities):
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"intensity array {index} contains non-finite values")
        if np.any(array < -1e-14):
            raise ValueError(f"intensity array {index} contains negative values")
        maximum = float(np.max(array, initial=0.0))
        if maximum <= 0.0:
            raise ValueError(f"intensity array {index} has no positive intensity")
        normalized.append(_readonly(np.maximum(array, 0.0) / maximum))
    return tuple(normalized)


def raw_ir_sticks(
    reference: HarmonicIRReference,
    *,
    window_cm1: tuple[float, float] | None = None,
    positive_frequencies_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, unbroadened harmonic positions and raw km/mol intensities."""

    frequency = np.asarray(reference.frequencies_cm1)
    intensity = np.asarray(reference.ir_intensities_km_mol)
    mask = np.ones(frequency.size, dtype=bool)
    if positive_frequencies_only:
        mask &= frequency > 0.0
    if window_cm1 is not None:
        low, high = map(float, window_cm1)
        if not low < high:
            raise ValueError("window_cm1 must have increasing bounds")
        mask &= (frequency >= low) & (frequency <= high)
    order = np.argsort(frequency[mask])
    return _readonly(frequency[mask][order]), _readonly(intensity[mask][order])


def _dirichlet_kernel(frequency_cycles_per_sample: np.ndarray, size: int) -> np.ndarray:
    frequency = np.asarray(frequency_cycles_per_sample, dtype=np.float64)
    return (
        np.exp(-1j * np.pi * frequency * (size - 1))
        * size
        * np.sinc(size * frequency)
        / np.sinc(frequency)
    )


def discrete_hann_power_response(
    frequency_offset_cycles_per_sample: np.ndarray,
    segment_frames: int,
) -> np.ndarray:
    """Normalized squared DTFT of ``np.hanning(segment_frames)``.

    The response is one at zero offset.  It includes the deterministic Hann
    main lobe and sidelobes rather than an adjustable physical linewidth.
    """

    frames = int(segment_frames)
    if frames < 8:
        raise ValueError("segment_frames must be at least 8")
    offset = np.asarray(frequency_offset_cycles_per_sample, dtype=np.float64)
    shift = 1.0 / (frames - 1)
    transform = (
        0.5 * _dirichlet_kernel(offset, frames)
        - 0.25 * _dirichlet_kernel(offset - shift, frames)
        - 0.25 * _dirichlet_kernel(offset + shift, frames)
    )
    zero_amplitude = 0.5 * (frames - 1)
    response = np.abs(transform / zero_amplitude) ** 2
    return np.maximum(response.real, 0.0)


def hann_resolution_envelope(
    stick_wavenumber_cm1: np.ndarray,
    stick_intensity: np.ndarray,
    output_wavenumber_cm1: np.ndarray,
    *,
    dt_fs: float,
    segment_time_fs: float,
) -> np.ndarray:
    """Convolve sticks with the exact finite-segment Hann power response."""

    sticks = np.asarray(stick_wavenumber_cm1, dtype=np.float64)
    weights = np.asarray(stick_intensity, dtype=np.float64)
    output = np.asarray(output_wavenumber_cm1, dtype=np.float64)
    if sticks.ndim != 1 or weights.shape != sticks.shape:
        raise ValueError("stick frequencies and intensities must be equal vectors")
    if output.ndim != 1 or output.size < 2 or np.any(np.diff(output) <= 0.0):
        raise ValueError("output wavenumbers must be a strictly increasing vector")
    if np.any(sticks < 0.0) or np.any(weights < 0.0):
        raise ValueError("stick frequencies and intensities must be non-negative")
    if not np.all(np.isfinite(sticks)) or not np.all(np.isfinite(weights)):
        raise ValueError("sticks contain non-finite values")
    dt = float(dt_fs)
    segment_time = float(segment_time_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be positive")
    if not np.isfinite(segment_time) or segment_time <= 0.0:
        raise ValueError("segment_time_fs must be positive")
    frames = int(round(segment_time / dt))
    if frames < 8:
        raise ValueError("segment_time_fs contains fewer than eight frames")
    nyquist_cm1 = 0.5 / (dt * FS_TO_S * SPEED_OF_LIGHT_CM_S)
    if np.max(sticks, initial=0.0) > nyquist_cm1:
        raise ValueError("a harmonic stick lies above the MD Nyquist wavenumber")
    if np.min(output, initial=0.0) < 0.0 or np.max(
        output, initial=0.0
    ) > nyquist_cm1:
        raise ValueError(
            "output wavenumbers must lie between zero and the MD Nyquist limit"
        )
    offsets = (
        (output[:, None] - sticks[None, :])
        * SPEED_OF_LIGHT_CM_S
        * dt
        * FS_TO_S
    )
    kernel = discrete_hann_power_response(offsets, frames)
    return _readonly(kernel @ weights)


def _clip_density_curve(
    x: np.ndarray,
    y: np.ndarray,
    window_cm1: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if window_cm1 is None:
        return x, y
    low, high = map(float, window_cm1)
    if not low < high:
        raise ValueError("window_cm1 must have increasing bounds")
    if high < x[0] or low > x[-1]:
        raise ValueError("window_cm1 does not overlap the spectrum")
    low = max(low, float(x[0]))
    high = min(high, float(x[-1]))
    interior = (x > low) & (x < high)
    clipped_x = np.concatenate(([low], x[interior], [high]))
    clipped_y = np.concatenate(
        ([np.interp(low, x, y)], y[interior], [np.interp(high, x, y)])
    )
    return clipped_x, clipped_y


def summarize_spectral_band(
    wavenumber_cm1: np.ndarray,
    intensity: np.ndarray,
    *,
    kind: str,
    window_cm1: tuple[float, float] | None = None,
) -> SpectralBandSummary:
    """Compute a centroid and 10--90% width for sticks or a sampled density."""

    x = np.asarray(wavenumber_cm1, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("wavenumber and intensity must be equal non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("spectrum contains non-finite values")
    if np.any(y < -1e-14):
        raise ValueError("spectrum contains negative intensity")
    y = np.maximum(y, 0.0)

    if kind == "sticks":
        mask = np.ones(x.size, dtype=bool)
        if window_cm1 is not None:
            low, high = map(float, window_cm1)
            if not low < high:
                raise ValueError("window_cm1 must have increasing bounds")
            mask &= (x >= low) & (x <= high)
        x_selected = x[mask]
        y_selected = y[mask]
        order = np.argsort(x_selected)
        x_selected = x_selected[order]
        y_selected = y_selected[order]
        total = float(np.sum(y_selected))
        if total <= 0.0:
            raise ValueError("selected stick band has no positive intensity")
        centroid = float(np.sum(x_selected * y_selected) / total)
        cumulative = np.cumsum(y_selected) / total
        q10 = float(x_selected[np.searchsorted(cumulative, 0.10)])
        q90 = float(x_selected[np.searchsorted(cumulative, 0.90)])
    elif kind == "density":
        if x.size < 2 or np.any(np.diff(x) <= 0.0):
            raise ValueError("density wavenumbers must be strictly increasing")
        x_selected, y_selected = _clip_density_curve(x, y, window_cm1)
        dx = np.diff(x_selected)
        interval_area = 0.5 * (y_selected[:-1] + y_selected[1:]) * dx
        total = float(np.sum(interval_area))
        if total <= 0.0:
            raise ValueError("selected density band has no positive area")
        xy = x_selected * y_selected
        centroid = float(
            np.sum(0.5 * (xy[:-1] + xy[1:]) * dx) / total
        )
        cumulative = np.concatenate(([0.0], np.cumsum(interval_area))) / total
        q10 = float(np.interp(0.10, cumulative, x_selected))
        q90 = float(np.interp(0.90, cumulative, x_selected))
    else:
        raise ValueError("kind must be 'sticks' or 'density'")

    return SpectralBandSummary(
        total_intensity=total,
        centroid_cm1=centroid,
        percentile_10_cm1=q10,
        percentile_90_cm1=q90,
        width_10_90_cm1=q90 - q10,
    )


def compare_md_to_harmonic_reference(
    md_wavenumber_cm1: np.ndarray,
    md_intensity: np.ndarray,
    reference: HarmonicIRReference,
    *,
    dt_fs: float,
    segment_time_fs: float,
    summary_window_cm1: tuple[float, float] | None = None,
) -> IRSpectrumComparison:
    """Build an independently normalized MD-versus-harmonic comparison."""

    wavenumber = np.asarray(md_wavenumber_cm1, dtype=np.float64)
    md = np.asarray(md_intensity, dtype=np.float64)
    if wavenumber.ndim != 1 or md.shape != wavenumber.shape:
        raise ValueError("MD wavenumber and intensity must be equal vectors")
    if wavenumber.size < 2 or np.any(np.diff(wavenumber) <= 0.0):
        raise ValueError("MD wavenumbers must be strictly increasing")
    if not np.all(np.isfinite(md)) or np.any(md < -1e-14):
        raise ValueError("MD intensity must be finite and non-negative")
    md = np.maximum(md, 0.0)
    sticks, stick_intensity = raw_ir_sticks(reference)
    if not np.any(stick_intensity > 0.0):
        raise ValueError("reference has no positive harmonic IR intensity")
    envelope = hann_resolution_envelope(
        sticks,
        stick_intensity,
        wavenumber,
        dt_fs=dt_fs,
        segment_time_fs=segment_time_fs,
    )
    md_normalized, envelope_normalized, sticks_normalized = (
        independently_max_normalize(md, envelope, stick_intensity)
    )
    frames = int(round(float(segment_time_fs) / float(dt_fs)))
    return IRSpectrumComparison(
        wavenumber_cm1=_readonly(wavenumber),
        md_intensity=_readonly(md),
        md_intensity_normalized=md_normalized,
        stick_wavenumber_cm1=sticks,
        stick_intensity_km_mol=stick_intensity,
        stick_intensity_normalized=sticks_normalized,
        reference_envelope=envelope,
        reference_envelope_normalized=envelope_normalized,
        md_summary=summarize_spectral_band(
            wavenumber,
            md,
            kind="density",
            window_cm1=summary_window_cm1,
        ),
        reference_stick_summary=summarize_spectral_band(
            sticks,
            stick_intensity,
            kind="sticks",
            window_cm1=summary_window_cm1,
        ),
        reference_envelope_summary=summarize_spectral_band(
            wavenumber,
            envelope,
            kind="density",
            window_cm1=summary_window_cm1,
        ),
        dt_fs=float(dt_fs),
        segment_time_fs=float(segment_time_fs),
        segment_frames=frames,
    )


def _bond_gradient(
    geometry: np.ndarray,
    first: int,
    second: int,
) -> np.ndarray:
    displacement = geometry[first] - geometry[second]
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-12:
        raise ValueError("zero-length bond in internal-coordinate definition")
    unit = displacement / distance
    gradient = np.zeros_like(geometry)
    gradient[first] = unit
    gradient[second] = -unit
    return gradient.reshape(-1)


def _angle_gradient(
    geometry: np.ndarray,
    first: int,
    center: int,
    third: int,
) -> np.ndarray:
    left = geometry[first] - geometry[center]
    right = geometry[third] - geometry[center]
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("zero-length angle arm in internal-coordinate definition")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    sine = float(np.sqrt(max(1.0 - cosine**2, 0.0)))
    if sine <= 1e-8:
        raise ValueError("linear angle has an ill-conditioned gradient")
    left_gradient = (
        cosine * left / left_norm**2 - right / (left_norm * right_norm)
    ) / sine
    right_gradient = (
        cosine * right / right_norm**2 - left / (left_norm * right_norm)
    ) / sine
    gradient = np.zeros_like(geometry)
    gradient[first] = left_gradient
    gradient[third] = right_gradient
    gradient[center] = -(left_gradient + right_gradient)
    return gradient.reshape(-1)


def _covalent_water_bonds(
    geometry: np.ndarray,
    atomic_numbers: np.ndarray,
    covalent_oh_cutoff_angstrom: float,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    oxygen = np.flatnonzero(atomic_numbers == 8)
    hydrogen = np.flatnonzero(atomic_numbers == 1)
    if oxygen.size == 0 or hydrogen.size != 2 * oxygen.size:
        raise ValueError("expected an (H2O)n composition")
    distances = np.linalg.norm(
        geometry[hydrogen, None, :] - geometry[oxygen][None, :, :], axis=-1
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(hydrogen.size), nearest]
    if np.any(nearest_distance > float(covalent_oh_cutoff_angstrom)):
        raise ValueError("at least one hydrogen has no covalent oxygen")
    by_oxygen = {int(index): [] for index in oxygen}
    for hydrogen_index, oxygen_local in zip(hydrogen, nearest, strict=True):
        by_oxygen[int(oxygen[oxygen_local])].append(int(hydrogen_index))
    if any(len(indices) != 2 for indices in by_oxygen.values()):
        raise ValueError("each water oxygen must own exactly two hydrogens")
    return oxygen, by_oxygen


def _water_ring_coordinates(
    geometry: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    covalent_oh_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
    require_single_ring: bool,
) -> tuple[
    dict[str, np.ndarray],
    tuple[HydrogenBondedOH, ...],
    tuple[tuple[int, int], ...],
]:
    oxygen, by_oxygen = _covalent_water_bonds(
        geometry, atomic_numbers, covalent_oh_cutoff_angstrom
    )
    oxygen_set = set(map(int, oxygen))
    hbonded: list[HydrogenBondedOH] = []
    free: list[tuple[int, int]] = []
    bend_rows: list[np.ndarray] = []
    hbonded_rows: list[np.ndarray] = []
    free_rows: list[np.ndarray] = []
    acceptor_counts = {int(index): 0 for index in oxygen}

    for donor, hydrogens in by_oxygen.items():
        bend_rows.append(_angle_gradient(geometry, hydrogens[0], donor, hydrogens[1]))
        for hydrogen in hydrogens:
            candidates: list[tuple[float, float, int]] = []
            for acceptor in oxygen_set.difference({donor}):
                h_to_acceptor = geometry[acceptor] - geometry[hydrogen]
                h_to_donor = geometry[donor] - geometry[hydrogen]
                distance = float(np.linalg.norm(h_to_acceptor))
                oo_distance = float(np.linalg.norm(geometry[acceptor] - geometry[donor]))
                cosine = float(
                    np.dot(h_to_donor, h_to_acceptor)
                    / (np.linalg.norm(h_to_donor) * distance)
                )
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                if (
                    distance <= float(h_acceptor_cutoff_angstrom)
                    and oo_distance <= float(oo_cutoff_angstrom)
                    and angle >= float(hbond_angle_cutoff_deg)
                ):
                    candidates.append((distance, -angle, int(acceptor)))
            coordinate = _bond_gradient(geometry, hydrogen, donor)
            if candidates:
                distance, negative_angle, acceptor = min(candidates)
                hbonded_rows.append(coordinate)
                acceptor_counts[acceptor] += 1
                hbonded.append(
                    HydrogenBondedOH(
                        oxygen_index=donor,
                        hydrogen_index=hydrogen,
                        acceptor_oxygen_index=acceptor,
                        hydrogen_acceptor_distance_angstrom=distance,
                        donor_hydrogen_acceptor_angle_deg=-negative_angle,
                    )
                )
            else:
                free_rows.append(coordinate)
                free.append((donor, hydrogen))

    if require_single_ring:
        donor_counts = {int(index): 0 for index in oxygen}
        outgoing: dict[int, int] = {}
        for bond in hbonded:
            donor_counts[bond.oxygen_index] += 1
            outgoing[bond.oxygen_index] = bond.acceptor_oxygen_index
        if any(count != 1 for count in donor_counts.values()) or any(
            count != 1 for count in acceptor_counts.values()
        ):
            raise ValueError(
                "hydrogen-bond geometry is not one-donor/one-acceptor per water"
            )
        start = int(oxygen[0])
        visited: set[int] = set()
        current = start
        while current not in visited:
            visited.add(current)
            current = outgoing[current]
        if current != start or visited != oxygen_set:
            raise ValueError("hydrogen-bond graph is not a single water ring")
    if not hbonded_rows or not free_rows:
        raise ValueError(
            "ring character requires both hydrogen-bonded and free OH bonds"
        )

    return (
        {
            "bend": np.vstack(bend_rows),
            "hbonded_oh": np.vstack(hbonded_rows),
            "free_oh": np.vstack(free_rows),
        },
        tuple(hbonded),
        tuple(free),
    )


def _orthonormal_row_basis(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.empty((0, values.shape[-1] if values.ndim == 2 else 0))
    _, singular, right = np.linalg.svd(values, full_matrices=False)
    threshold = singular[0] * max(values.shape) * np.finfo(np.float64).eps * 100.0
    rank = int(np.sum(singular > threshold))
    return right[:rank]


def _projection_power(modes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    basis = _orthonormal_row_basis(rows)
    if basis.shape[0] == 0:
        return np.zeros(modes.shape[0])
    return np.sum((modes @ basis.T) ** 2, axis=1)


def water_ring_mode_characters(
    geometry_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    masses_u: np.ndarray,
    mass_weighted_modes: np.ndarray,
    *,
    covalent_oh_cutoff_angstrom: float = 1.25,
    h_acceptor_cutoff_angstrom: float = 2.5,
    oo_cutoff_angstrom: float = 3.5,
    hbond_angle_cutoff_deg: float = 140.0,
    require_single_ring: bool = True,
) -> WaterRingModeCharacters:
    """Partition cyclic-water modes by internal-coordinate subspaces.

    The three intramolecular categories can be non-orthogonal.  Their additive
    shares are therefore the permutation-averaged (Shapley) increments in
    explained squared displacement.  ``intermolecular`` is the orthogonal
    complement of their combined subspace.  The four fractions sum to one for
    every mode without privileging an arbitrary category order. The default
    H-bond definition matches the trajectory and reference diagnostics:
    H...O <= 2.5 A, O...O <= 3.5 A, and O-H...O >= 140 degrees.
    """

    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    numbers = np.asarray(atomic_numbers)
    masses = np.asarray(masses_u, dtype=np.float64)
    modes = np.asarray(mass_weighted_modes, dtype=np.float64)
    if numbers.ndim != 1 or geometry.shape != (numbers.size, 3):
        raise ValueError("geometry and atomic_numbers have incompatible shapes")
    if masses.shape != (numbers.size,) or np.any(masses <= 0.0):
        raise ValueError("masses must contain one positive value per atom")
    if modes.ndim != 3 or modes.shape[1:] != (numbers.size, 3):
        raise ValueError("mass_weighted_modes must have shape (modes, atoms, 3)")
    flattened = modes.reshape(modes.shape[0], -1)
    norms = np.linalg.norm(flattened, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-6):
        raise ValueError("each mass-weighted mode must be unit-normalized")

    coordinate_rows, hbonded, free = _water_ring_coordinates(
        geometry,
        numbers.astype(np.int64),
        covalent_oh_cutoff_angstrom=covalent_oh_cutoff_angstrom,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
        oo_cutoff_angstrom=oo_cutoff_angstrom,
        hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
        require_single_ring=require_single_ring,
    )
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    categories = ("bend", "hbonded_oh", "free_oh")
    rows = {
        name: coordinate_rows[name] * inverse_sqrt_mass[None, :]
        for name in categories
    }

    contribution = {name: np.zeros(flattened.shape[0]) for name in categories}
    for ordering in permutations(categories):
        included: list[np.ndarray] = []
        previous = np.zeros(flattened.shape[0])
        for name in ordering:
            included.append(rows[name])
            current = _projection_power(flattened, np.vstack(included))
            contribution[name] += current - previous
            previous = current
    # There are exactly three intramolecular categories, hence 3! orderings.
    permutation_count = 6.0
    fractions = np.column_stack(
        [np.maximum(contribution[name] / permutation_count, 0.0) for name in categories]
    )
    total_intramolecular = _projection_power(
        flattened, np.vstack([rows[name] for name in categories])
    )
    fractions = np.column_stack(
        (fractions, np.maximum(1.0 - total_intramolecular, 0.0))
    )
    row_sum = fractions.sum(axis=1)
    fractions /= row_sum[:, None]
    all_categories = (*categories, "intermolecular")
    dominant = tuple(all_categories[index] for index in np.argmax(fractions, axis=1))
    return WaterRingModeCharacters(
        categories=all_categories,
        fractions=_readonly(fractions),
        dominant_labels=dominant,
        hbonded_oh=hbonded,
        free_oh=free,
    )


def reference_water_ring_mode_characters(
    reference: HarmonicIRReference,
    **kwargs: Any,
) -> WaterRingModeCharacters:
    """Convenience wrapper for :func:`water_ring_mode_characters`."""

    return water_ring_mode_characters(
        reference.geometry_angstrom,
        reference.atomic_numbers,
        reference.masses_u,
        reference.mass_weighted_modes,
        **kwargs,
    )


def label_water_monomer_modes(
    geometry_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    masses_u: np.ndarray,
    mass_weighted_modes: np.ndarray,
) -> tuple[str, ...]:
    """Assign bend, symmetric-stretch, and antisymmetric-stretch labels."""

    geometry = np.asarray(geometry_angstrom, dtype=np.float64)
    numbers = np.asarray(atomic_numbers)
    masses = np.asarray(masses_u, dtype=np.float64)
    modes = np.asarray(mass_weighted_modes, dtype=np.float64)
    oxygen, by_oxygen = _covalent_water_bonds(geometry, numbers, 1.25)
    if oxygen.size != 1 or modes.shape != (3, numbers.size, 3):
        raise ValueError("monomer labeling requires one H2O and its three modes")
    center = int(oxygen[0])
    hydrogen = by_oxygen[center]
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    stretch_rows = [
        _bond_gradient(geometry, atom, center) * inverse_sqrt_mass
        for atom in hydrogen
    ]
    stretch_rows = [row / np.linalg.norm(row) for row in stretch_rows]
    bend = (
        _angle_gradient(geometry, hydrogen[0], center, hydrogen[1])
        * inverse_sqrt_mass
    )
    ideal = np.vstack(
        (
            bend / np.linalg.norm(bend),
            (stretch_rows[0] + stretch_rows[1])
            / np.linalg.norm(stretch_rows[0] + stretch_rows[1]),
            (stretch_rows[0] - stretch_rows[1])
            / np.linalg.norm(stretch_rows[0] - stretch_rows[1]),
        )
    )
    flattened = modes.reshape(3, -1)
    score = np.abs(ideal @ flattened.T) ** 2
    label_to_mode = _maximize_assignment(score)
    names = ("bend", "symmetric_stretch", "antisymmetric_stretch")
    labels = [""] * 3
    for label, mode in zip(names, label_to_mode, strict=True):
        labels[int(mode)] = label
    return tuple(labels)


def reference_water_monomer_mode_labels(
    reference: HarmonicIRReference,
) -> tuple[str, ...]:
    """Convenience wrapper for :func:`label_water_monomer_modes`."""

    return label_water_monomer_modes(
        reference.geometry_angstrom,
        reference.atomic_numbers,
        reference.masses_u,
        reference.mass_weighted_modes,
    )


def _near_degenerate_groups(
    first_frequencies: np.ndarray,
    second_frequencies: np.ndarray,
    tolerance_cm1: float,
) -> list[np.ndarray]:
    first = np.asarray(first_frequencies, dtype=np.float64)
    second = np.asarray(second_frequencies, dtype=np.float64)
    count = first.size
    adjacency = np.eye(count, dtype=bool)
    for left in range(count):
        for right in range(left + 1, count):
            if (
                abs(first[left] - first[right]) <= tolerance_cm1
                or abs(second[left] - second[right]) <= tolerance_cm1
            ):
                adjacency[left, right] = adjacency[right, left] = True
    groups: list[np.ndarray] = []
    unseen = set(range(count))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(map(int, np.flatnonzero(adjacency[node])))
        unseen.difference_update(component)
        groups.append(np.asarray(sorted(component), dtype=np.int64))
    return groups


def _mark_group_adjacency(adjacency: np.ndarray, groups: Sequence[np.ndarray]) -> None:
    for group in groups:
        if group.size > 1:
            adjacency[np.ix_(group, group)] = True


def _adjacency_groups(adjacency: np.ndarray) -> list[np.ndarray]:
    count = adjacency.shape[0]
    groups: list[np.ndarray] = []
    unseen = set(range(count))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(map(int, np.flatnonzero(adjacency[node])))
        unseen.difference_update(component)
        groups.append(np.asarray(sorted(component), dtype=np.int64))
    return groups


def _align_degenerate_subspaces(
    previous_modes: np.ndarray,
    current_modes: np.ndarray,
    previous_frequencies: np.ndarray,
    current_frequencies: np.ndarray,
    tolerance_cm1: float,
) -> np.ndarray:
    aligned = current_modes.copy()
    for group in _near_degenerate_groups(
        previous_frequencies, current_frequencies, tolerance_cm1
    ):
        if group.size < 2:
            continue
        previous = previous_modes[group]
        current = current_modes[group]
        left, _, right = np.linalg.svd(previous @ current.T)
        rotation = left @ right
        aligned[group] = rotation @ current
    return aligned


def match_isotopologue_modes(
    source: HarmonicIRReference,
    target: HarmonicIRReference,
    *,
    mass_path_steps: int = 65,
    degeneracy_tolerance_cm1: float = 2.0,
    geometry_tolerance_angstrom: float = 1e-8,
    hessian_relative_tolerance: float = 1e-8,
    dipole_derivative_relative_tolerance: float = 1e-10,
) -> IsotopologueModeMatch:
    """Track modes over a continuous source-to-target isotope mass path.

    The Born--Oppenheimer geometry, Cartesian Hessian, and dipole derivative
    must be the same.  At each intermediate mass, maximum squared eigenvector
    overlap supplies the Hungarian assignment.  Near-degenerate blocks are
    propagated with an orthogonal Procrustes alignment and reported as
    subspaces rather than being over-interpreted as unique individual modes.
    """

    if source.atomic_numbers.shape != target.atomic_numbers.shape or not np.array_equal(
        source.atomic_numbers, target.atomic_numbers
    ):
        raise ValueError("isotopologue artifacts must have identical atom ordering")
    if source.n_modes != target.n_modes:
        raise ValueError("isotopologue artifacts have different mode counts")
    if np.max(np.abs(source.geometry_angstrom - target.geometry_angstrom)) > float(
        geometry_tolerance_angstrom
    ):
        raise ValueError("isotopologue artifacts must use the same geometry")
    hessian_scale = max(
        float(np.max(np.abs(source.hessian_hartree_per_bohr2))), 1e-30
    )
    if (
        np.max(
            np.abs(
                source.hessian_hartree_per_bohr2
                - target.hessian_hartree_per_bohr2
            )
        )
        > float(hessian_relative_tolerance) * hessian_scale
    ):
        raise ValueError("isotopologue artifacts must use the same electronic Hessian")
    dipole_scale = max(
        float(np.max(np.abs(source.dipole_derivative_3n_by_3_au))), 1e-30
    )
    if (
        np.max(
            np.abs(
                source.dipole_derivative_3n_by_3_au
                - target.dipole_derivative_3n_by_3_au
            )
        )
        > float(dipole_derivative_relative_tolerance) * dipole_scale
    ):
        raise ValueError(
            "isotopologue artifacts must use the same electronic dipole derivative"
        )
    steps = int(mass_path_steps)
    if steps < 2:
        raise ValueError("mass_path_steps must be at least two")
    tolerance = float(degeneracy_tolerance_cm1)
    if tolerance < 0.0:
        raise ValueError("degeneracy_tolerance_cm1 must be non-negative")

    fractions = np.linspace(0.0, 1.0, steps)
    previous_modes = source.mass_weighted_modes.reshape(source.n_modes, -1).copy()
    previous_frequencies = source.frequencies_cm1.copy()
    minimum_overlap = np.ones(source.n_modes)
    endpoint_path_modes = previous_modes
    endpoint_path_frequencies = previous_frequencies
    ambiguity_adjacency = np.eye(source.n_modes, dtype=bool)
    _mark_group_adjacency(
        ambiguity_adjacency,
        _near_degenerate_groups(
            source.frequencies_cm1,
            source.frequencies_cm1,
            tolerance,
        ),
    )

    for fraction in fractions[1:]:
        masses = (
            (1.0 - fraction) * source.masses_u + fraction * target.masses_u
        )
        solution = solve_vibrational_modes(
            source.hessian_hartree_per_bohr2,
            source.geometry_angstrom,
            masses,
        )
        raw_modes = solution.mass_weighted_modes.reshape(source.n_modes, -1)
        assignment = _maximize_assignment(np.abs(previous_modes @ raw_modes.T) ** 2)
        current_modes = raw_modes[assignment]
        current_frequencies = solution.frequencies_cm1[assignment]
        step_groups = _near_degenerate_groups(
            previous_frequencies,
            current_frequencies,
            tolerance,
        )
        _mark_group_adjacency(ambiguity_adjacency, step_groups)
        current_modes = _align_degenerate_subspaces(
            previous_modes,
            current_modes,
            previous_frequencies,
            current_frequencies,
            tolerance,
        )
        step_overlap = np.abs(np.sum(previous_modes * current_modes, axis=1)) ** 2
        minimum_overlap = np.minimum(minimum_overlap, step_overlap)
        previous_modes = current_modes
        previous_frequencies = current_frequencies
        endpoint_path_modes = current_modes
        endpoint_path_frequencies = current_frequencies

    target_modes = target.mass_weighted_modes.reshape(target.n_modes, -1)
    endpoint_assignment = _maximize_assignment(
        np.abs(endpoint_path_modes @ target_modes.T) ** 2
    )
    endpoint_overlap = np.abs(
        np.sum(endpoint_path_modes * target_modes[endpoint_assignment], axis=1)
    ) ** 2

    ambiguous: list[ModeSubspaceMatch] = []
    target_frequencies_in_path_order = target.frequencies_cm1[endpoint_assignment]
    _mark_group_adjacency(
        ambiguity_adjacency,
        _near_degenerate_groups(
            endpoint_path_frequencies,
            target_frequencies_in_path_order,
            tolerance,
        ),
    )
    for group in _adjacency_groups(ambiguity_adjacency):
        if group.size < 2:
            continue
        source_block = endpoint_path_modes[group]
        target_indices = endpoint_assignment[group]
        target_block = target_modes[target_indices]
        singular = np.linalg.svd(source_block @ target_block.T, compute_uv=False)
        ambiguous.append(
            ModeSubspaceMatch(
                source_indices=tuple(map(int, group)),
                target_indices=tuple(sorted(map(int, target_indices))),
                minimum_principal_overlap=float(np.min(singular) ** 2),
            )
        )

    return IsotopologueModeMatch(
        source_to_target=_readonly(endpoint_assignment, dtype=np.int64),
        endpoint_squared_overlaps=_readonly(endpoint_overlap),
        minimum_path_squared_overlaps=_readonly(minimum_overlap),
        ambiguous_subspaces=tuple(ambiguous),
        mass_path_fractions=_readonly(fractions),
    )


__all__ = [
    "HESSIAN_EIGENVALUE_TO_CM1",
    "IR_ARTIFACT_SCHEMA",
    "IR_ARTIFACT_VERSION",
    "HarmonicIRReference",
    "HydrogenBondedOH",
    "IRArtifactError",
    "IRSpectrumComparison",
    "IsotopologueModeMatch",
    "ModeSubspaceMatch",
    "NormalModeSolution",
    "SpectralBandSummary",
    "WaterRingModeCharacters",
    "compare_md_to_harmonic_reference",
    "discrete_hann_power_response",
    "hann_resolution_envelope",
    "independently_max_normalize",
    "label_water_monomer_modes",
    "load_psi4_b973c_ir_artifact",
    "match_isotopologue_modes",
    "raw_ir_sticks",
    "reference_water_monomer_mode_labels",
    "reference_water_ring_mode_characters",
    "solve_vibrational_modes",
    "summarize_spectral_band",
    "water_ring_mode_characters",
]
