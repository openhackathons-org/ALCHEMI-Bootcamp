"""Analytic tests for the finite-difference harmonic-IR helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import harmonic_ir  # noqa: E402


def _forces_for_quadratic(
    positions_angstrom: np.ndarray,
    center_angstrom: np.ndarray,
    hessian_eV_per_angstrom2: np.ndarray,
) -> np.ndarray:
    displacement = positions_angstrom.reshape(positions_angstrom.shape[0], -1)
    displacement = displacement - center_angstrom.reshape(1, -1)
    forces = -(displacement @ hessian_eV_per_angstrom2.T)
    return forces.reshape(positions_angstrom.shape)


class CartesianFiniteDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = np.array(
            [
                [0.15, -0.20, 0.35],
                [1.10, 0.45, -0.30],
            ],
            dtype=np.float64,
        )

    def test_symmetric_displacements_have_one_coordinate_per_structure(self) -> None:
        geometry_before = self.geometry.copy()
        displaced = harmonic_ir.symmetric_cartesian_displacements(
            self.geometry,
            0.0125,
        )

        self.assertEqual(displaced.plus_angstrom.shape, (6, 2, 3))
        self.assertEqual(displaced.minus_angstrom.shape, (6, 2, 3))
        np.testing.assert_array_equal(
            displaced.coordinate_atom_indices,
            np.array([0, 0, 0, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            displaced.coordinate_axes,
            np.array([0, 1, 2, 0, 1, 2]),
        )
        expected = np.eye(6) * displaced.step_angstrom
        np.testing.assert_allclose(
            displaced.plus_angstrom.reshape(6, 6) - self.geometry.reshape(1, 6),
            expected,
            atol=0.0,
        )
        np.testing.assert_allclose(
            displaced.minus_angstrom.reshape(6, 6) - self.geometry.reshape(1, 6),
            -expected,
            atol=0.0,
        )
        np.testing.assert_array_equal(self.geometry, geometry_before)
        self.assertFalse(displaced.plus_angstrom.flags.writeable)

    def test_harmonic_forces_recover_hessian_sign_shape_and_units(self) -> None:
        coupling = np.array(
            [
                [1.0, 0.1, -0.2, 0.3, 0.0, 0.2],
                [0.2, 0.8, 0.0, -0.1, 0.3, 0.0],
                [0.0, 0.2, 0.9, 0.1, -0.2, 0.1],
                [0.1, 0.0, 0.2, 1.1, 0.1, -0.1],
                [-0.1, 0.1, 0.0, 0.2, 0.7, 0.2],
                [0.2, -0.2, 0.1, 0.0, 0.1, 0.9],
            ]
        )
        expected_eV_per_angstrom2 = coupling.T @ coupling + 0.25 * np.eye(6)
        displaced = harmonic_ir.symmetric_cartesian_displacements(
            self.geometry,
            0.02,
        )
        forces_plus = _forces_for_quadratic(
            displaced.plus_angstrom,
            self.geometry,
            expected_eV_per_angstrom2,
        )
        forces_minus = _forces_for_quadratic(
            displaced.minus_angstrom,
            self.geometry,
            expected_eV_per_angstrom2,
        )

        estimate = harmonic_ir.assemble_cartesian_hessian(
            forces_plus,
            forces_minus,
            displaced.step_angstrom,
        )

        self.assertEqual(estimate.hessian_eV_per_angstrom2.shape, (6, 6))
        np.testing.assert_allclose(
            estimate.raw_hessian_eV_per_angstrom2,
            expected_eV_per_angstrom2,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            estimate.hessian_hartree_per_bohr2,
            expected_eV_per_angstrom2
            * harmonic_ir.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        self.assertLess(estimate.max_relative_antisymmetry, 1.0e-13)

    def test_raw_hessian_antisymmetry_is_reported_before_symmetrizing(self) -> None:
        step = 0.01
        raw = np.diag(np.linspace(0.5, 1.0, 6))
        raw[0, 4] = 0.20
        raw[4, 0] = -0.05
        forces_plus = (-step * raw.T).reshape(6, 2, 3)
        forces_minus = (step * raw.T).reshape(6, 2, 3)

        estimate = harmonic_ir.assemble_cartesian_hessian(
            forces_plus,
            forces_minus,
            step,
        )

        expected_antisymmetry = np.max(np.abs(raw - raw.T))
        np.testing.assert_allclose(estimate.raw_hessian_eV_per_angstrom2, raw)
        np.testing.assert_allclose(
            estimate.hessian_eV_per_angstrom2,
            0.5 * (raw + raw.T),
        )
        self.assertAlmostEqual(
            estimate.max_abs_antisymmetry_eV_per_angstrom2,
            expected_antisymmetry,
        )
        self.assertAlmostEqual(
            estimate.max_relative_antisymmetry,
            expected_antisymmetry
            / np.max(np.abs(0.5 * (raw + raw.T))),
        )


class DipoleFiniteDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.958, 0.000, 0.000],
                [-0.240, 0.927, 0.000],
            ],
            dtype=np.float64,
        )
        self.base_charges = np.array([-0.82, 0.41, 0.41])
        first = np.linspace(-0.05, 0.07, 9)
        second = np.linspace(0.03, -0.02, 9)
        self.charge_jacobian = np.vstack((first, second, -(first + second)))
        self.atomic_dipole_jacobian = np.column_stack(
            (
                np.linspace(0.01, 0.03, 9),
                np.linspace(-0.02, 0.01, 9),
                np.linspace(0.015, -0.005, 9),
            )
        )

    def _predictions(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        displacement = positions.reshape(positions.shape[0], -1)
        displacement = displacement - self.geometry.reshape(1, -1)
        charges = self.base_charges + displacement @ self.charge_jacobian.T
        atomic_dipoles = np.zeros_like(positions)
        atomic_dipoles[:, 0, :] = displacement @ self.atomic_dipole_jacobian
        return charges, atomic_dipoles

    def test_position_dependent_neutral_charges_give_origin_invariant_derivative(
        self,
    ) -> None:
        displaced = harmonic_ir.symmetric_cartesian_displacements(
            self.geometry,
            0.015,
        )
        charges_plus, atomic_plus = self._predictions(displaced.plus_angstrom)
        charges_minus, atomic_minus = self._predictions(displaced.minus_angstrom)
        shifted_origin = np.array([3.2, -1.7, 0.8])

        dipoles_plus = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            displaced.plus_angstrom,
            charges_plus,
            atomic_dipoles_e_angstrom=atomic_plus,
            neutral_tolerance_e=1.0e-12,
        )
        dipoles_minus = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            displaced.minus_angstrom,
            charges_minus,
            atomic_dipoles_e_angstrom=atomic_minus,
            neutral_tolerance_e=1.0e-12,
        )
        shifted_plus = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            displaced.plus_angstrom,
            charges_plus,
            atomic_dipoles_e_angstrom=atomic_plus,
            origin_angstrom=shifted_origin,
            neutral_tolerance_e=1.0e-12,
        )
        per_structure_plus = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            displaced.plus_angstrom,
            charges_plus,
            atomic_dipoles_e_angstrom=atomic_plus,
            origin_angstrom=displaced.plus_angstrom[:, 0, :],
            neutral_tolerance_e=1.0e-12,
        )
        shifted_minus = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            displaced.minus_angstrom,
            charges_minus,
            atomic_dipoles_e_angstrom=atomic_minus,
            origin_angstrom=shifted_origin,
            neutral_tolerance_e=1.0e-12,
        )
        np.testing.assert_allclose(shifted_plus, dipoles_plus, atol=3.0e-15)
        np.testing.assert_allclose(shifted_minus, dipoles_minus, atol=3.0e-15)
        np.testing.assert_allclose(per_structure_plus, dipoles_plus, atol=3.0e-15)

        estimate = harmonic_ir.assemble_dipole_derivative(
            dipoles_plus,
            dipoles_minus,
            displaced.step_angstrom,
        )
        shifted_estimate = harmonic_ir.assemble_dipole_derivative(
            shifted_plus,
            shifted_minus,
            displaced.step_angstrom,
        )

        expected = np.empty((9, 3), dtype=np.float64)
        for coordinate in range(9):
            atom, axis = divmod(coordinate, 3)
            expected[coordinate] = (
                self.charge_jacobian[:, coordinate] @ self.geometry
                + self.base_charges[atom] * np.eye(3)[axis]
                + self.atomic_dipole_jacobian[coordinate]
            )
        np.testing.assert_allclose(
            estimate.dipole_derivative_3n_by_3_au,
            expected,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            shifted_estimate.dipole_derivative_3n_by_3_au,
            expected,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            estimate.dipoles_plus_atomic_units,
            dipoles_plus * harmonic_ir.BOHR_PER_ANGSTROM,
            rtol=0.0,
            atol=0.0,
        )

    def test_charged_point_charge_dipole_has_the_expected_origin_shift(self) -> None:
        geometry = self.geometry[:2]
        charges = np.array([0.65, -0.15])
        origin = np.array([1.2, -0.4, 0.7])
        at_zero = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            geometry,
            charges,
        )
        shifted = harmonic_ir.molecular_dipoles_from_atomic_predictions(
            geometry,
            charges,
            origin_angstrom=origin,
        )
        total_charge = float(np.sum(charges))
        np.testing.assert_allclose(shifted - at_zero, -total_charge * origin)
        with self.assertRaisesRegex(ValueError, "not neutral"):
            harmonic_ir.molecular_dipoles_from_atomic_predictions(
                geometry,
                charges,
                neutral_tolerance_e=1.0e-12,
            )


class HarmonicIRModeAnalysisTests(unittest.TestCase):
    artifact_root = PART_DIR / "reference" / "artifacts"

    @staticmethod
    def _artifact_arrays(label: str) -> dict[str, np.ndarray]:
        path = HarmonicIRModeAnalysisTests.artifact_root / label / "ir_arrays.npz"
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}

    def test_projected_modes_and_intensities_reproduce_psi4_artifacts(self) -> None:
        for label in ("h2o", "d2o"):
            with self.subTest(label=label):
                arrays = self._artifact_arrays(label)
                analysis = harmonic_ir.analyze_harmonic_ir(
                    arrays["hessian_hartree_per_bohr2"],
                    arrays["dipole_derivative_3n_by_3_au"],
                    arrays["geometry_angstrom"],
                    arrays["masses_u"],
                )

                self.assertEqual(analysis.external_rank, 6)
                self.assertEqual(analysis.mass_weighted_modes.shape, (3, 3, 3))
                self.assertEqual(
                    analysis.normal_dipole_derivative_modes_by_3_e_per_sqrt_u.shape,
                    (3, 3),
                )
                np.testing.assert_allclose(
                    analysis.frequencies_cm1,
                    arrays["frequencies_cm1"],
                    rtol=0.0,
                    atol=2.0e-7,
                )
                # The checked-in artifact was produced by Psi4.  Its legacy
                # physical constants differ from current CODATA by 1.5e-8 in
                # this conversion, so the comparison is intentionally tight
                # but not bitwise.
                np.testing.assert_allclose(
                    analysis.ir_intensities_km_mol,
                    arrays["ir_intensities_km_mol"],
                    rtol=2.0e-8,
                    atol=2.0e-8,
                )
                expected_intensity = (
                    harmonic_ir.IR_INTENSITY_KM_MOL_PER_E2_PER_U
                    * np.sum(
                        analysis.normal_dipole_derivative_modes_by_3_e_per_sqrt_u
                        ** 2,
                        axis=1,
                    )
                )
                np.testing.assert_allclose(
                    analysis.ir_intensities_km_mol,
                    expected_intensity,
                    rtol=2.0e-15,
                    atol=2.0e-15,
                )

    def test_deuteration_changes_masses_only(self) -> None:
        h2o = self._artifact_arrays("h2o")
        d2o = self._artifact_arrays("d2o")
        for name in (
            "atomic_numbers",
            "geometry_angstrom",
            "hessian_hartree_per_bohr2",
            "dipole_derivative_3n_by_3_au",
        ):
            np.testing.assert_array_equal(h2o[name], d2o[name])

        hydrogen = h2o["atomic_numbers"] == 1
        np.testing.assert_array_equal(
            h2o["masses_u"][~hydrogen],
            d2o["masses_u"][~hydrogen],
        )
        self.assertTrue(np.all(d2o["masses_u"][hydrogen] > h2o["masses_u"][hydrogen]))

        h_analysis = harmonic_ir.analyze_harmonic_ir(
            h2o["hessian_hartree_per_bohr2"],
            h2o["dipole_derivative_3n_by_3_au"],
            h2o["geometry_angstrom"],
            h2o["masses_u"],
        )
        d_analysis = harmonic_ir.analyze_harmonic_ir(
            h2o["hessian_hartree_per_bohr2"],
            h2o["dipole_derivative_3n_by_3_au"],
            h2o["geometry_angstrom"],
            d2o["masses_u"],
        )
        self.assertTrue(np.all(d_analysis.frequencies_cm1 < h_analysis.frequencies_cm1))
        np.testing.assert_allclose(
            d_analysis.frequencies_cm1,
            d2o["frequencies_cm1"],
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            d_analysis.ir_intensities_km_mol,
            d2o["ir_intensities_km_mol"],
            rtol=2.0e-8,
            atol=2.0e-8,
        )


class ConvergenceSummaryTests(unittest.TestCase):
    def test_adjacent_step_summary_tracks_second_order_finite_difference_error(
        self,
    ) -> None:
        geometry = np.array(
            [
                [-0.3, 0.0, 0.2],
                [0.7, 0.4, -0.1],
            ]
        )
        hessian = np.diag(np.linspace(0.8, 1.3, 6))
        hessian[0, 4] = hessian[4, 0] = 0.12
        force_cubic = np.linspace(0.2, 0.7, 6)
        dipole_derivative = np.column_stack(
            (
                np.linspace(0.1, 0.3, 6),
                np.linspace(-0.2, 0.2, 6),
                np.linspace(0.05, -0.1, 6),
            )
        )
        dipole_cubic = np.column_stack(
            (
                np.linspace(0.10, 0.30, 6),
                np.linspace(0.25, 0.05, 6),
                np.linspace(-0.20, 0.10, 6),
            )
        )

        estimates = []
        for step in (0.01, 0.04, 0.02):
            displaced = harmonic_ir.symmetric_cartesian_displacements(
                geometry,
                step,
            )
            plus_x = displaced.plus_angstrom.reshape(6, 6) - geometry.reshape(1, 6)
            minus_x = (
                displaced.minus_angstrom.reshape(6, 6) - geometry.reshape(1, 6)
            )

            def force(flat_displacement: np.ndarray) -> np.ndarray:
                linear = -(flat_displacement @ hessian.T)
                nonlinear = -(flat_displacement**3) * force_cubic
                return (linear + nonlinear).reshape(6, 2, 3)

            def dipole(flat_displacement: np.ndarray) -> np.ndarray:
                return (
                    flat_displacement @ dipole_derivative
                    + (flat_displacement**3) @ dipole_cubic
                )

            estimates.append(
                harmonic_ir.assemble_harmonic_ir_finite_difference(
                    forces_plus_eV_per_angstrom=force(plus_x),
                    forces_minus_eV_per_angstrom=force(minus_x),
                    dipoles_plus_e_angstrom=dipole(plus_x),
                    dipoles_minus_e_angstrom=dipole(minus_x),
                    step_angstrom=step,
                )
            )

        summary = harmonic_ir.summarize_finite_difference_convergence(estimates)

        np.testing.assert_allclose(summary.steps_angstrom, [0.04, 0.02, 0.01])
        np.testing.assert_allclose(summary.coarse_steps_angstrom, [0.04, 0.02])
        np.testing.assert_allclose(summary.fine_steps_angstrom, [0.02, 0.01])
        self.assertEqual(summary.hessian_max_abs_change_hartree_per_bohr2.shape, (2,))
        self.assertEqual(summary.dipole_derivative_max_abs_change_au.shape, (2,))
        self.assertAlmostEqual(
            summary.hessian_max_abs_change_hartree_per_bohr2[0]
            / summary.hessian_max_abs_change_hartree_per_bohr2[1],
            4.0,
            places=10,
        )
        self.assertAlmostEqual(
            summary.dipole_derivative_max_abs_change_au[0]
            / summary.dipole_derivative_max_abs_change_au[1],
            4.0,
            places=10,
        )
        np.testing.assert_allclose(
            summary.hessian_max_relative_antisymmetry,
            0.0,
            atol=1.0e-13,
        )

        spectral = harmonic_ir.summarize_harmonic_ir_convergence(
            estimates,
            geometry,
            np.array([1.0, 2.0]),
        )
        np.testing.assert_allclose(spectral.steps_angstrom, [0.04, 0.02, 0.01])
        self.assertEqual(spectral.frequencies_cm1.shape, (3, 1))
        self.assertEqual(spectral.ir_intensities_km_mol.shape, (3, 1))
        self.assertEqual(spectral.frequency_abs_change_cm1.shape, (2, 1))
        self.assertEqual(spectral.ir_intensity_relative_change.shape, (2, 1))
        self.assertGreater(
            spectral.frequency_max_abs_change_cm1[0],
            spectral.frequency_max_abs_change_cm1[1],
        )
        self.assertGreater(
            spectral.ir_intensity_max_abs_change_km_mol[0],
            spectral.ir_intensity_max_abs_change_km_mol[1],
        )
        np.testing.assert_allclose(
            spectral.minimum_same_index_mode_squared_overlap,
            1.0,
            rtol=0.0,
            atol=2.0e-15,
        )


class HarmonicIntensityTests(unittest.TestCase):
    def test_mode_strengths_apply_inverse_sqrt_mass_and_dipole_projection(self) -> None:
        masses = np.array([1.0, 4.0])
        modes = np.zeros((2, 2, 3))
        modes[0, 0, 0] = 1.0
        modes[1, 1, 1] = 1.0
        derivative = np.zeros((6, 3))
        derivative[0] = [2.0, 0.0, 0.0]
        derivative[4] = [0.0, 6.0, 0.0]

        strengths = harmonic_ir.harmonic_mode_dipole_strengths(
            derivative,
            modes,
            masses,
        )

        # Mode 0: |2 / sqrt(1)|^2 = 4; mode 1: |6 / sqrt(4)|^2 = 9.
        np.testing.assert_allclose(strengths, [4.0, 9.0])
        self.assertFalse(strengths.flags.writeable)
        with self.assertRaisesRegex(ValueError, "finite"):
            harmonic_ir.harmonic_mode_dipole_strengths(
                derivative,
                modes,
                np.array([1.0, np.nan]),
            )


if __name__ == "__main__":
    unittest.main()
