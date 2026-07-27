from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFERENCE_DIR))

import b97_3c_ir as reference  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_h2o_seed_matches_expected_topology(self) -> None:
        geometry = reference.make_h2o_seed()
        diagnostics = reference.topology_diagnostics(geometry)

        self.assertEqual(geometry.symbols, ("O", "H", "H"))
        np.testing.assert_allclose(
            geometry.positions_angstrom.min(axis=0)
            + geometry.positions_angstrom.max(axis=0),
            0.0,
            atol=1.0e-15,
        )
        self.assertEqual(len(diagnostics["covalent_bonds"]), 2)
        self.assertTrue(
            diagnostics["water_network"]["all_oxygens_have_two_hydrogens"]
        )

    def test_cyclic_h6_seed_matches_tutorial_construction(self) -> None:
        geometry = reference.make_cyclic_h6_seed()
        diagnostics = reference.topology_diagnostics(geometry)

        self.assertEqual(len(geometry.symbols), 18)
        self.assertEqual(geometry.symbols, ("O", "H", "H") * 6)
        self.assertEqual(len(diagnostics["covalent_bonds"]), 12)
        self.assertTrue(
            diagnostics["water_network"]["all_oxygens_have_two_hydrogens"]
        )
        self.assertEqual(diagnostics["water_network"]["hydrogen_bond_count"], 6)

    def test_xyz_round_trip(self) -> None:
        geometry = reference.make_h2o_seed()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "water.xyz"
            reference.write_xyz(path, geometry, comment="round trip")
            loaded = reference.read_xyz(path)

        self.assertEqual(loaded.symbols, geometry.symbols)
        self.assertEqual(loaded.comment, "round trip")
        np.testing.assert_allclose(
            loaded.positions_angstrom, geometry.positions_angstrom, atol=5.0e-13
        )

    def test_xyz_rejects_a_second_frame(self) -> None:
        text = "1\nfirst\nH 0 0 0\n1\nsecond\nH 0 0 0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_frames.xyz"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "after the first frame"):
                reference.read_xyz(path)


class _FakeMatrix:
    @staticmethod
    def from_array(array: np.ndarray) -> np.ndarray:
        return np.asarray(array).copy()


class _FakeCore:
    Matrix = _FakeMatrix


class _FakeMolecule:
    def __init__(self, geometry: np.ndarray) -> None:
        self._geometry = np.asarray(geometry, dtype=float).copy()

    def natom(self) -> int:
        return self._geometry.shape[0]

    def geometry(self) -> np.ndarray:
        return self._geometry.copy()

    def clone(self) -> "_FakeMolecule":
        return _FakeMolecule(self._geometry)

    def set_geometry(self, geometry: np.ndarray) -> None:
        self._geometry = np.asarray(geometry, dtype=float).copy()

    def update_geometry(self) -> None:
        pass


class _FakeWavefunction:
    def __init__(self, energy: float, dipole: np.ndarray) -> None:
        self._values = {
            "CURRENT ENERGY": energy,
            "CURRENT DIPOLE": np.asarray(dipole),
        }

    def variable(self, name: str):
        return self._values[name]


class _FakePsi4:
    core = _FakeCore

    def __init__(self, hessian: np.ndarray, dipole_derivative: np.ndarray) -> None:
        self.hessian = np.asarray(hessian)
        self.dipole_derivative = np.asarray(dipole_derivative)

    def gradient(self, _method: str, *, molecule: _FakeMolecule, **_kwargs):
        coordinates = molecule.geometry().reshape(-1)
        energy = 0.5 * coordinates @ self.hessian @ coordinates
        gradient = (self.hessian @ coordinates).reshape(molecule.natom(), 3)
        dipole = coordinates @ self.dipole_derivative
        return gradient, _FakeWavefunction(float(energy), dipole)


class FiniteDifferenceTests(unittest.TestCase):
    def test_cartesian_tensor_orientation(self) -> None:
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(6, 6))
        hessian = raw + raw.T + 8.0 * np.eye(6)
        dipole_derivative = rng.normal(size=(6, 3))
        molecule = _FakeMolecule(rng.normal(size=(2, 3)))

        result = reference.finite_difference_cartesian(
            _FakePsi4(hessian, dipole_derivative), molecule, step_bohr=0.005
        )

        np.testing.assert_allclose(
            result["hessian_raw_Eh_per_bohr2"], hessian, atol=1.0e-11
        )
        np.testing.assert_allclose(
            result["dipole_derivative_3n_by_3_au"],
            dipole_derivative,
            atol=1.0e-11,
        )


class _Datum:
    def __init__(self, data) -> None:
        self.data = np.asarray(data)


class BundleTests(unittest.TestCase):
    def test_v1_bundle_is_numeric_and_self_hashed(self) -> None:
        trv = np.array(["TR"] * 6 + ["V"] * 3)
        vibinfo = {
            "TRV": _Datum(trv),
            "omega": _Datum(np.array([0.0] * 6 + [1000.0, 2000.0, 3000.0])),
            "IR_intensity": _Datum(np.arange(9.0)),
            "q": _Datum(np.eye(9)),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference.write_isotopologue_bundle(
                root,
                label="h2o",
                psi4_version="1.11",
                charge=0,
                multiplicity=1,
                atomic_numbers=np.array([8, 1, 1]),
                geometry_angstrom=np.array(
                    [[0.0, 0.0, 0.1], [0.0, 0.8, -0.4], [0.0, -0.8, -0.4]]
                ),
                masses_u=np.array([16.0, 1.0, 1.0]),
                hessian=np.eye(9),
                dipole_derivative_3n_by_3_au=np.zeros((9, 3)),
                vibinfo=vibinfo,
                step_bohr=0.005,
                imaginary_threshold_cm1=10.0,
                validation={
                    "status": "passed",
                    "reference_ready": True,
                    "is_minimum_within_threshold": True,
                    "significant_imaginary_modes": [],
                },
                provenance={"dispersion": "D3(BJ)-ATM"},
            )
            bundle = root / "artifacts" / "h2o"
            manifest = json.loads((bundle / "manifest.json").read_text())
            arrays = np.load(bundle / "ir_arrays.npz", allow_pickle=False)

            self.assertEqual(
                manifest["format"],
                {"name": "alchemi.psi4-b97-3c-ir", "version": 1},
            )
            self.assertEqual(
                manifest["normal_modes"]["convention"],
                "q_equals_sqrt_mass_times_cartesian",
            )
            self.assertEqual(arrays["frequencies_cm1"].shape, (3,))
            self.assertEqual(
                arrays["dipole_derivative_3n_by_3_au"].shape, (9, 3)
            )
            self.assertEqual(arrays["mass_weighted_modes"].shape, (3, 3, 3))
            self.assertFalse(any(array.dtype.hasobject for array in arrays.values()))
            self.assertEqual(
                reference.sha256_file(bundle / "ir_arrays.npz"),
                manifest["arrays"]["sha256"],
            )

    def test_directory_manifest_is_recursive_and_excludes_psi4_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "result.dat").write_text("result", encoding="utf-8")
            (root / "nested" / "bundle.dat").write_text("bundle", encoding="utf-8")
            (root / "psi.123.clean").write_text("transient", encoding="utf-8")
            (root / "psi4_scratch").mkdir()
            (root / "psi4_scratch" / "scratch.dat").write_text(
                "scratch", encoding="utf-8"
            )

            reference.write_manifest(root, excluded={"manifest.json"})
            manifest = json.loads((root / "manifest.json").read_text())

        self.assertEqual(
            [row["path"] for row in manifest["files"]],
            ["nested/bundle.dat", "result.dat"],
        )


class CommandLineTests(unittest.TestCase):
    def test_robust_scf_default_disables_initial_adiis(self) -> None:
        arguments = reference.build_parser().parse_args(["--output", "unused"])

        self.assertEqual(arguments.scf_initial_accelerator, "NONE")
        self.assertTrue(arguments.require_minimum)


if __name__ == "__main__":
    unittest.main()
