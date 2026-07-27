"""Synthetic tests for the dependency-free harmonic-IR reference helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

import aux.reference as reference_api  # noqa: E402
from aux.reference import core as ir_reference  # noqa: E402


def _water_geometry() -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(104.5)
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.97, 0.0, 0.0],
            [0.97 * np.cos(angle), 0.97 * np.sin(angle), 0.0],
        ]
    )
    return geometry, np.array([8, 1, 1])


def _hessian_from_modes(
    geometry: np.ndarray,
    masses: np.ndarray,
    frequencies_cm1: np.ndarray,
    modes: np.ndarray | None = None,
) -> np.ndarray:
    _, vibrational = ir_reference._external_and_vibrational_bases(geometry, masses)
    if modes is None:
        modes = vibrational.T
    flattened = np.asarray(modes, dtype=np.float64).reshape(len(frequencies_cm1), -1)
    np.testing.assert_allclose(
        flattened @ flattened.T, np.eye(len(flattened)), atol=1e-10
    )
    eigenvalues = (
        np.asarray(frequencies_cm1) / ir_reference.HESSIAN_EIGENVALUE_TO_CM1
    ) ** 2
    weighted = flattened.T @ np.diag(eigenvalues) @ flattened
    sqrt_masses = np.repeat(np.sqrt(masses), 3)
    return sqrt_masses[:, None] * weighted * sqrt_masses[None, :]


def _physical_water_modes(
    geometry: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    stretch_1 = ir_reference._bond_gradient(geometry, 1, 0) * inverse_sqrt_mass
    stretch_2 = ir_reference._bond_gradient(geometry, 2, 0) * inverse_sqrt_mass
    bend = ir_reference._angle_gradient(geometry, 1, 0, 2) * inverse_sqrt_mass
    ideals = np.vstack((bend, stretch_1 + stretch_2, stretch_1 - stretch_2))
    gram = ideals @ ideals.T
    values, vectors = np.linalg.eigh(gram)
    inverse_root = vectors @ np.diag(values**-0.5) @ vectors.T
    return inverse_root @ ideals


def _write_artifact(
    directory: Path,
    *,
    label: str,
    masses: np.ndarray,
    hessian: np.ndarray,
    intensities: np.ndarray | None = None,
    supplied_modes: np.ndarray | None = None,
    dipole_derivative: np.ndarray | None = None,
) -> Path:
    geometry, numbers = _water_geometry()
    solution = ir_reference.solve_vibrational_modes(hessian, geometry, masses)
    if intensities is None:
        intensities = np.array([20.0, 80.0, 50.0])
    if supplied_modes is None:
        supplied_modes = solution.mass_weighted_modes
    if dipole_derivative is None:
        dipole_derivative = np.zeros((3 * len(numbers), 3))
    arrays_path = directory / "ir_arrays.npz"
    np.savez(
        arrays_path,
        atomic_numbers=numbers,
        geometry_angstrom=geometry,
        masses_u=masses,
        hessian_hartree_per_bohr2=hessian,
        dipole_derivative_3n_by_3_au=dipole_derivative,
        frequencies_cm1=solution.frequencies_cm1,
        ir_intensities_km_mol=intensities,
        mass_weighted_modes=supplied_modes,
    )
    checksum = sha256(arrays_path.read_bytes()).hexdigest()
    manifest = {
        "format": {
            "name": ir_reference.IR_ARTIFACT_SCHEMA,
            "version": ir_reference.IR_ARTIFACT_VERSION,
        },
        "artifact_id": f"synthetic-{label.lower()}",
        "engine": {"name": "Psi4", "version": "synthetic-test"},
        "model_chemistry": {"method": "B97-3c"},
        "molecule": {
            "label": label,
            "charge": 0,
            "multiplicity": 1,
        },
        "arrays": {"file": arrays_path.name, "sha256": checksum},
        "normal_modes": {
            "array": "mass_weighted_modes",
            "convention": "q_equals_sqrt_mass_times_cartesian",
            "normalization": "orthonormal_rows",
            "ordering": "frequencies_and_ir_intensities",
        },
        "units": {
            "geometry": "angstrom",
            "masses": "unified_atomic_mass_unit",
            "hessian": "hartree_per_bohr2",
            "dipole_derivative": "atomic_unit_dipole_per_bohr",
            "frequencies": "cm^-1",
            "ir_intensities": "km_per_mol",
        },
        "validation": {
            "status": "passed",
            "reference_ready": True,
            "is_minimum_within_threshold": True,
            "significant_imaginary_modes": [],
            "gradient_rms_Eh_per_bohr": 1.0e-7,
            "gradient_max_abs_Eh_per_bohr": 2.0e-7,
            "gradient_max_abs_limit_Eh_per_bohr": 1.0e-5,
            "raw_hessian_max_antisymmetry_relative": 0.0,
            "raw_hessian_max_antisymmetry_relative_limit": 1.0e-3,
            "covalent_graph_preserved": True,
            "optimized_all_oxygens_have_two_hydrogens": True,
            "initial_is_single_water_ring": False,
            "optimized_is_single_water_ring": False,
            "optimized_hydrogen_bond_count": 0,
            "same_geometry_hessian_and_dipole_derivative_for_isotopes": True,
            "changed_entries_are_hydrogen_masses_only": True,
        },
        "provenance": {"dispersion": "D3(BJ)-ATM"},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _water_hexamer_ring() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 6
    oo_distance = 2.78
    oh_distance = 0.97
    hoh = np.deg2rad(104.5)
    oxygen = np.array(
        [
            [
                oo_distance * np.cos(2.0 * np.pi * index / count),
                oo_distance * np.sin(2.0 * np.pi * index / count),
                0.0,
            ]
            for index in range(count)
        ]
    )
    rotation = np.array(
        [
            [np.cos(-hoh), -np.sin(-hoh), 0.0],
            [np.sin(-hoh), np.cos(-hoh), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    geometry: list[np.ndarray] = []
    numbers: list[int] = []
    for index, center in enumerate(oxygen):
        donor = oxygen[(index + 1) % count] - center
        donor /= np.linalg.norm(donor)
        free = rotation @ donor
        geometry.extend(
            (center, center + oh_distance * donor, center + oh_distance * free)
        )
        numbers.extend((8, 1, 1))
    masses = np.where(np.asarray(numbers) == 8, 15.99491462, 1.00782503)
    return np.asarray(geometry), np.asarray(numbers), masses


class ArtifactAndSpectrumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry, _ = _water_geometry()
        self.h_masses = np.array([15.99491462, 1.00782503, 1.00782503])
        modes = _physical_water_modes(self.geometry, self.h_masses)
        self.hessian = _hessian_from_modes(
            self.geometry,
            self.h_masses,
            np.array([1600.0, 3650.0, 3750.0]),
            modes,
        )

    def test_reference_package_preserves_the_public_core_api(self) -> None:
        self.assertEqual(set(reference_api.__all__), set(ir_reference.__all__))
        for name in ir_reference.__all__:
            self.assertIs(getattr(reference_api, name), getattr(ir_reference, name))

    def test_loads_checksummed_bundle_and_labels_monomer_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _write_artifact(
                Path(temporary),
                label="H2O",
                masses=self.h_masses,
                hessian=self.hessian,
            )
            reference = ir_reference.load_psi4_b973c_ir_artifact(manifest)

        self.assertEqual(reference.label, "H2O")
        self.assertEqual(reference.n_modes, 3)
        self.assertEqual(
            set(ir_reference.reference_water_monomer_mode_labels(reference)),
            {"bend", "symmetric_stretch", "antisymmetric_stretch"},
        )
        np.testing.assert_allclose(
            reference.mass_weighted_modes.reshape(3, -1)
            @ reference.mass_weighted_modes.reshape(3, -1).T,
            np.eye(3),
            atol=1e-10,
        )

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_artifact(
                root,
                label="H2O",
                masses=self.h_masses,
                hessian=self.hessian,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["arrays"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ir_reference.IRArtifactError, "checksum"):
                ir_reference.load_psi4_b973c_ir_artifact(manifest_path)

    def test_failed_scientific_check_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_artifact(
                root,
                label="H2O",
                masses=self.h_masses,
                hessian=self.hessian,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation"]["status"] = "failed"
            manifest["validation"]["reference_ready"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ir_reference.IRArtifactError, "ready for comparison"
            ):
                ir_reference.load_psi4_b973c_ir_artifact(manifest_path)

    def test_preserves_exported_basis_inside_a_degenerate_subspace(self) -> None:
        hessian = _hessian_from_modes(
            self.geometry,
            self.h_masses,
            np.array([1600.0, 3700.0, 3700.0]),
        )
        solution = ir_reference.solve_vibrational_modes(
            hessian, self.geometry, self.h_masses
        )
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0**-0.5, 2.0**-0.5],
                [0.0, -2.0**-0.5, 2.0**-0.5],
            ]
        )
        exported = (
            rotation @ solution.mass_weighted_modes.reshape(3, -1)
        ).reshape(3, 3, 3)
        with tempfile.TemporaryDirectory() as temporary:
            reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    Path(temporary),
                    label="H2O",
                    masses=self.h_masses,
                    hessian=hessian,
                    supplied_modes=exported,
                )
            )
        np.testing.assert_allclose(reference.mass_weighted_modes, exported, atol=1e-12)

    def test_hann_response_matches_direct_discrete_transform(self) -> None:
        frames = 32
        offsets = np.array([-0.03125, -0.007, 0.0, 0.013, 0.0625])
        samples = np.arange(frames)
        window = np.hanning(frames)
        direct = np.array(
            [
                abs(np.sum(window * np.exp(-2j * np.pi * offset * samples))) ** 2
                / np.sum(window) ** 2
                for offset in offsets
            ]
        )
        actual = ir_reference.discrete_hann_power_response(offsets, frames)
        np.testing.assert_allclose(actual, direct, atol=2e-14, rtol=2e-13)

    def test_independent_normalization_and_band_summaries(self) -> None:
        first, second = ir_reference.independently_max_normalize(
            np.array([0.0, 2.0]), np.array([0.0, 50.0])
        )
        np.testing.assert_array_equal(first, second)
        sticks = ir_reference.summarize_spectral_band(
            np.array([1000.0, 2000.0, 3000.0]),
            np.array([1.0, 2.0, 1.0]),
            kind="sticks",
        )
        self.assertAlmostEqual(sticks.centroid_cm1, 2000.0)
        self.assertAlmostEqual(sticks.percentile_10_cm1, 1000.0)
        self.assertAlmostEqual(sticks.percentile_90_cm1, 3000.0)
        self.assertAlmostEqual(sticks.width_10_90_cm1, 2000.0)

    def test_comparison_keeps_raw_sticks_and_uses_hann_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    Path(temporary),
                    label="H2O",
                    masses=self.h_masses,
                    hessian=self.hessian,
                )
            )
        grid = np.linspace(500.0, 4500.0, 4001)
        md = np.exp(-0.5 * ((grid - 1600.0) / 30.0) ** 2)
        md += 0.7 * np.exp(-0.5 * ((grid - 3650.0) / 40.0) ** 2)
        comparison = ir_reference.compare_md_to_harmonic_reference(
            grid,
            md,
            reference,
            dt_fs=0.5,
            segment_time_fs=500.0,
        )
        np.testing.assert_array_equal(
            comparison.stick_intensity_km_mol,
            reference.ir_intensities_km_mol[np.argsort(reference.frequencies_cm1)],
        )
        self.assertEqual(comparison.segment_frames, 1000)
        self.assertAlmostEqual(comparison.md_intensity_normalized.max(), 1.0)
        self.assertAlmostEqual(comparison.reference_envelope_normalized.max(), 1.0)


class ModeCharacterAndMatchingTests(unittest.TestCase):
    def test_ring_character_is_additive_and_recovers_coordinate_subspaces(self) -> None:
        geometry, numbers, masses = _water_hexamer_ring()
        coordinate_rows, _, _ = ir_reference._water_ring_coordinates(
            geometry,
            numbers,
            covalent_oh_cutoff_angstrom=1.25,
            h_acceptor_cutoff_angstrom=2.5,
            oo_cutoff_angstrom=3.5,
            hbond_angle_cutoff_deg=140.0,
            require_single_ring=True,
        )
        inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
        category_modes = []
        for name in ("bend", "hbonded_oh", "free_oh"):
            basis = ir_reference._orthonormal_row_basis(
                coordinate_rows[name] * inverse_sqrt_mass[None, :]
            )
            category_modes.append(basis[0])
        all_internal = ir_reference._orthonormal_row_basis(
            np.vstack(tuple(coordinate_rows.values())) * inverse_sqrt_mass[None, :]
        )
        _, vibrational = ir_reference._external_and_vibrational_bases(geometry, masses)
        intermolecular = None
        for candidate in vibrational.T:
            residual = candidate - (candidate @ all_internal.T) @ all_internal
            if np.linalg.norm(residual) > 0.5:
                intermolecular = residual / np.linalg.norm(residual)
                break
        assert intermolecular is not None
        modes = np.vstack((*category_modes, intermolecular)).reshape(4, -1, 3)

        character = ir_reference.water_ring_mode_characters(
            geometry, numbers, masses, modes
        )
        np.testing.assert_allclose(character.fractions.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(len(character.hbonded_oh), 6)
        self.assertEqual(len(character.free_oh), 6)
        self.assertGreater(character.fractions[0, 0], 0.8)
        self.assertGreater(character.fractions[1, 1], 0.8)
        self.assertGreater(character.fractions[2, 2], 0.8)
        self.assertGreater(character.fractions[3, 3], 0.99)

    def test_continuous_mass_path_matches_h2o_and_d2o(self) -> None:
        geometry, _ = _water_geometry()
        h_masses = np.array([15.99491462, 1.00782503, 1.00782503])
        d_masses = np.array([15.99491462, 2.01410178, 2.01410178])
        h_modes = _physical_water_modes(geometry, h_masses)
        hessian = _hessian_from_modes(
            geometry,
            h_masses,
            np.array([1600.0, 3650.0, 3750.0]),
            h_modes,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h_root = root / "h"
            d_root = root / "d"
            h_root.mkdir()
            d_root.mkdir()
            h_reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    h_root, label="H2O", masses=h_masses, hessian=hessian
                )
            )
            d_reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    d_root, label="D2O", masses=d_masses, hessian=hessian
                )
            )
        match = ir_reference.match_isotopologue_modes(
            h_reference, d_reference, mass_path_steps=11
        )
        self.assertEqual(set(match.source_to_target.tolist()), {0, 1, 2})
        self.assertGreater(float(match.endpoint_squared_overlaps.min()), 0.99)
        self.assertGreater(float(match.minimum_path_squared_overlaps.min()), 0.99)

    def test_isotope_mapping_rejects_a_different_dipole_derivative(self) -> None:
        geometry, numbers = _water_geometry()
        h_masses = np.array([15.99491462, 1.00782503, 1.00782503])
        d_masses = np.array([15.99491462, 2.01410178, 2.01410178])
        hessian = _hessian_from_modes(
            geometry,
            h_masses,
            np.array([1600.0, 3650.0, 3750.0]),
            _physical_water_modes(geometry, h_masses),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h_root = root / "h"
            d_root = root / "d"
            h_root.mkdir()
            d_root.mkdir()
            h_reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    h_root, label="H2O", masses=h_masses, hessian=hessian
                )
            )
            d_reference = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(
                    d_root,
                    label="D2O",
                    masses=d_masses,
                    hessian=hessian,
                    dipole_derivative=np.full((3 * len(numbers), 3), 1.0e-3),
                )
            )

        with self.assertRaisesRegex(ValueError, "same electronic dipole derivative"):
            ir_reference.match_isotopologue_modes(h_reference, d_reference)

    def test_near_degenerate_modes_are_reported_as_a_subspace(self) -> None:
        geometry, _ = _water_geometry()
        masses = np.array([15.99491462, 1.00782503, 1.00782503])
        hessian = _hessian_from_modes(
            geometry, masses, np.array([1600.0, 3700.0, 3700.0])
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            source = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(left, label="H2O", masses=masses, hessian=hessian)
            )
            target = ir_reference.load_psi4_b973c_ir_artifact(
                _write_artifact(right, label="H2O", masses=masses, hessian=hessian)
            )
        match = ir_reference.match_isotopologue_modes(
            source,
            target,
            mass_path_steps=3,
            degeneracy_tolerance_cm1=2.0,
        )
        self.assertEqual(len(match.ambiguous_subspaces), 1)
        self.assertEqual(match.ambiguous_subspaces[0].source_indices, (1, 2))
        self.assertGreater(
            match.ambiguous_subspaces[0].minimum_principal_overlap, 0.999999
        )


if __name__ == "__main__":
    unittest.main()
