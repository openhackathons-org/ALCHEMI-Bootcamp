from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REFERENCE_DIR.parent
sys.path.insert(0, str(REFERENCE_DIR))

import water_dimer_b97_3c as reference  # noqa: E402


class _FakeAtoms:
    def __init__(self, positions: np.ndarray, distance: float) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.pbc = np.array([False, False, False])
        self.info = {"oo_distance_angstrom": distance}

    def get_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_chemical_symbols(self) -> list[str]:
        return ["O", "H", "H", "O", "H", "H"]


def _fake_point(distance: float = 2.9) -> reference.ScanGeometry:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.25, 0.91, 0.0],
            [distance, 0.0, 0.0],
            [distance + 0.95, 0.0, 0.0],
            [distance - 0.25, -0.91, 0.0],
        ]
    )
    return reference.scan_geometry_from_atoms(
        _FakeAtoms(positions, distance),
        index=0,
        requested_distance_angstrom=distance,
    )


class ScanContractTests(unittest.TestCase):
    def test_declared_scan_is_fixed_and_complete(self) -> None:
        distances = reference.DEFAULT_OO_DISTANCES_ANGSTROM

        self.assertEqual(
            distances,
            (2.50, 2.70, 2.90, 3.20, 3.50, 3.90, 4.40, 5.00),
        )
        self.assertEqual(len(set(distances)), len(distances))
        self.assertEqual(tuple(sorted(distances)), distances)

    def test_frozen_monomers_are_exact_parent_slices(self) -> None:
        point = _fake_point()

        self.assertEqual(point.ab.symbols, ("O", "H", "H", "O", "H", "H"))
        self.assertEqual(point.a.symbols, ("O", "H", "H"))
        self.assertEqual(point.b.symbols, ("O", "H", "H"))
        self.assertTrue(
            np.array_equal(point.a.positions_angstrom, point.ab.positions_angstrom[:3])
        )
        self.assertTrue(
            np.array_equal(point.b.positions_angstrom, point.ab.positions_angstrom[3:])
        )
        self.assertAlmostEqual(point.measured_oo_distance_angstrom, 2.9)

    def test_periodic_or_mislabelled_geometry_is_rejected(self) -> None:
        atoms = _FakeAtoms(_fake_point().ab.positions_angstrom, 2.9)
        atoms.pbc[0] = True

        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            reference.scan_geometry_from_atoms(
                atoms, index=0, requested_distance_angstrom=2.9
            )

        atoms.pbc[:] = False
        atoms.info["oo_distance_angstrom"] = 3.0
        with self.assertRaisesRegex(ValueError, "builder declared"):
            reference.scan_geometry_from_atoms(
                atoms, index=0, requested_distance_angstrom=2.9
            )

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is not installed")
    def test_runtime_builder_parity_is_exact(self) -> None:
        sys.path.insert(0, str(TUTORIAL_DIR))
        from aux.structures import make_water_dimer_scan

        expected = make_water_dimer_scan(reference.DEFAULT_OO_DISTANCES_ANGSTROM)
        observed = reference.load_scan_geometries()

        self.assertEqual(len(observed), len(expected))
        for point, atoms in zip(observed, expected, strict=True):
            self.assertEqual(point.ab.symbols, tuple(atoms.get_chemical_symbols()))
            self.assertTrue(
                np.array_equal(point.ab.positions_angstrom, atoms.get_positions())
            )


class _FakeMolecule:
    def __init__(self, specification: str) -> None:
        self.specification = specification
        self.name = ""

    def set_name(self, name: str) -> None:
        self.name = name

    def update_geometry(self) -> None:
        pass


class _FakePsi4:
    def __init__(self) -> None:
        self.calls: list[tuple[str, _FakeMolecule]] = []

    def geometry(self, specification: str) -> _FakeMolecule:
        return _FakeMolecule(specification)

    def energy(self, method: str, *, molecule: _FakeMolecule) -> float:
        self.calls.append((method, molecule))
        if molecule.name.endswith("-AB"):
            return -152.025
        if molecule.name.endswith("-A"):
            return -76.010
        if molecule.name.endswith("-B"):
            return -76.012
        raise AssertionError(molecule.name)


class EnergyDefinitionTests(unittest.TestCase):
    def test_three_full_endpoints_are_recomputed_and_subtracted(self) -> None:
        psi4 = _FakePsi4()

        record = reference.evaluate_scan_point(psi4, _fake_point())

        self.assertEqual(len(psi4.calls), 3)
        self.assertEqual(
            [molecule.name.rsplit("-", 1)[-1] for _, molecule in psi4.calls],
            ["AB", "A", "B"],
        )
        self.assertTrue(
            all(method == reference.MODEL_CHEMISTRY for method, _ in psi4.calls)
        )
        self.assertAlmostEqual(record.interaction_Eh, -0.003)
        self.assertIn("no_com", psi4.calls[0][1].specification)
        self.assertIn("no_reorient", psi4.calls[0][1].specification)

    def test_method_contract_states_endpoint_and_counterpoise_boundaries(self) -> None:
        contract = reference.method_contract()
        joined = " ".join(contract["components"])

        self.assertIn("D3(BJ)", joined)
        self.assertIn("ATM", joined)
        self.assertIn("gCP", joined)
        self.assertIn("No Boys-Bernardi", contract["counterpoise_policy"])
        self.assertIn("not termwise equivalent", contract["comparison_boundary"])


class ArtifactTests(unittest.TestCase):
    def test_extxyz_contains_all_roles_and_round_trip_safe_hashes(self) -> None:
        point = _fake_point()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structures.extxyz"
            reference.write_structures_extxyz(path, [point])
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual([lines[0], lines[8], lines[13]], ["6", "3", "3"])
        self.assertIn("role=AB", lines[1])
        self.assertIn("role=A", lines[9])
        self.assertIn("role=B", lines[14])
        self.assertIn(reference.geometry_sha256(point.ab), lines[1])

    def test_csv_stores_full_energies_interaction_and_geometry_hashes(self) -> None:
        record = reference.EnergyRecord(_fake_point(), -152.025, -76.010, -76.012)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.csv"
            reference.write_scan_csv(path, [record])
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["interaction_Eh"]), -0.003)
        self.assertEqual(rows[0]["ab_geometry_sha256"], reference.geometry_sha256(record.point.ab))

    def test_checksum_index_covers_manifest_but_not_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.csv").write_text("value\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
            )
            (root / "psi4_scratch").mkdir()
            (root / "psi4_scratch" / "scratch").write_text("x", encoding="utf-8")

            reference.write_sha256sums(root)
            rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()

        self.assertEqual([row.split("  ", 1)[1] for row in rows], ["manifest.json", "scan.csv"])


class CommandLineTests(unittest.TestCase):
    def test_cli_has_no_distance_or_reduced_scan_option(self) -> None:
        parser = reference.build_parser()
        args = parser.parse_args(["--output", "new-output"])
        defaults = parser.parse_args([])
        destinations = {action.dest for action in parser._actions}

        self.assertEqual(args.output, Path("new-output"))
        self.assertEqual(defaults.output, reference.DEFAULT_OUTPUT_DIR)
        self.assertNotIn("distances", destinations)
        self.assertNotIn("reduced", destinations)


if __name__ == "__main__":
    unittest.main()
