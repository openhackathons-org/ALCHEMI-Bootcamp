"""Tests for deterministic water-run artifact manifests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import read


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import (  # noqa: E402
    ORBMOL_RELAXATION_STRUCTURE_FILENAMES,
    WATER_RUN_MANIFEST_NAME,
    WATER_RUN_MANIFEST_SCHEMA,
    graph_atoms_from_batch,
    write_orbmol_relaxation_structures,
    write_water_run_manifest,
)
from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402
from aux.domain.results import load_domain_lesson_view  # noqa: E402


def test_installed_domain_result_set_is_complete() -> None:
    recorded = PART_DIR / "data" / "domain_decomposition" / "recorded"
    fixed_atom_count = (
        DOMAIN_METHODOLOGY.fixed_molecules_per_species
        * DOMAIN_METHODOLOGY.atoms_per_composition_unit
    )

    view = load_domain_lesson_view(
        recorded,
        expected_atom_count=fixed_atom_count,
        expected_world_sizes=DOMAIN_METHODOLOGY.campaign_world_sizes,
    )

    assert view.available
    assert view.bundle_record is not None
    assert view.takeaway["all_fixed_evaluations_succeeded"]
    assert view.takeaway["positions_pbc_equivalent"]
    assert view.takeaway["all_output_checks_passed"]


def test_water_run_manifest_is_complete_normalized_and_repeatable(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "water_spectrum.csv"
    second = nested / "water_cluster.extxyz"
    first.write_bytes(b"frequency,intensity\n1000,0.4\n")
    second.write_bytes(b"3\nframe\n")
    notes = tmp_path / "notes.txt"
    notes.write_text("run note")

    manifest = write_water_run_manifest(
        tmp_path,
        run_details={"commit": "abc123", "root": Path("repo/part-1")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        checks={"passed": np.bool_(True), "errors": np.array([0.0, 1.0e-6])},
    )
    manifest_path = tmp_path / WATER_RUN_MANIFEST_NAME
    first_bytes = manifest_path.read_bytes()
    repeated = write_water_run_manifest(
        tmp_path,
        run_details={"commit": "abc123", "root": Path("repo/part-1")},
        settings={"steps": np.int64(55_000), "dt_fs": np.float64(0.5)},
        checks={"passed": np.bool_(True), "errors": np.array([0.0, 1.0e-6])},
    )

    assert manifest_path.read_bytes() == first_bytes
    assert repeated == manifest == json.loads(first_bytes)
    assert manifest["schema"] == WATER_RUN_MANIFEST_SCHEMA
    assert manifest["run_details"]["root"] == "repo/part-1"
    assert manifest["settings"] == {"dt_fs": 0.5, "steps": 55_000}
    assert manifest["checks"] == {"errors": [0.0, 1.0e-6], "passed": True}
    assert "provenance" not in manifest
    assert "gates" not in manifest
    assert [item["path"] for item in manifest["files"]] == [
        "nested/water_cluster.extxyz",
        "notes.txt",
        "water_spectrum.csv",
    ]
    for item, path in zip(manifest["files"], (second, notes, first), strict=True):
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert WATER_RUN_MANIFEST_NAME not in {item["path"] for item in manifest["files"]}


def test_water_run_manifest_inventories_nested_zarr_files(tmp_path: Path) -> None:
    zarr = tmp_path / "water_ir_relaxed.zarr"
    chunks = zarr / "positions" / "c"
    chunks.mkdir(parents=True)
    root_metadata = zarr / "zarr.json"
    array_metadata = zarr / "positions" / "zarr.json"
    chunk = chunks / "0"
    root_metadata.write_text('{"zarr_format": 3}\n')
    array_metadata.write_text('{"node_type": "array"}\n')
    chunk.write_bytes(b"binary chunk")
    run_log = tmp_path / "run.log"
    run_log.write_text("complete\n")

    manifest = write_water_run_manifest(
        tmp_path,
        run_details={},
        settings={},
        checks={},
    )

    assert [item["path"] for item in manifest["files"]] == [
        "run.log",
        "water_ir_relaxed.zarr/positions/c/0",
        "water_ir_relaxed.zarr/positions/zarr.json",
        "water_ir_relaxed.zarr/zarr.json",
    ]
    assert [item["bytes"] for item in manifest["files"]] == [
        path.stat().st_size for path in (run_log, chunk, array_metadata, root_metadata)
    ]


def test_water_run_manifest_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_water_run_manifest(
            tmp_path,
            run_details={},
            settings={"temperature": np.float64(np.nan)},
            checks={},
        )
    assert not (tmp_path / WATER_RUN_MANIFEST_NAME).exists()


def _orbmol_relaxation_batch(*, displacement: float = 0.0) -> SimpleNamespace:
    dimer_numbers = np.array([8, 1, 1, 8, 1, 1])
    hexamer_numbers = np.tile(np.array([8, 1, 1]), 6)
    numbers = np.concatenate((dimer_numbers, hexamer_numbers))
    positions = np.arange(72, dtype=float).reshape(24, 3) * 0.01
    positions[:, 0] += displacement
    masses = np.where(numbers == 8, 15.999, 1.008)
    return SimpleNamespace(
        batch_ptr=np.array([0, 6, 24]),
        atomic_numbers=numbers,
        atomic_masses=masses,
        positions=positions,
    )


def test_write_orbmol_relaxation_structures_saves_four_inspectable_files(
    tmp_path: Path,
) -> None:
    records = write_orbmol_relaxation_structures(
        tmp_path,
        initial_batch=_orbmol_relaxation_batch(),
        final_batch=_orbmol_relaxation_batch(displacement=0.02),
    )

    assert set(records) == {
        "dimer_initial",
        "dimer_final",
        "hexamer_initial",
        "hexamer_final",
    }
    assert {Path(record["path"]).name for record in records.values()} == set(
        ORBMOL_RELAXATION_STRUCTURE_FILENAMES.values()
    )
    for (system, state), filename in ORBMOL_RELAXATION_STRUCTURE_FILENAMES.items():
        atoms = read(tmp_path / filename, format="extxyz")
        assert atoms.info["system"] == system
        assert atoms.info["relaxation_state"] == state
        assert atoms.info["model"] == "OrbMol-v2"
        assert len(atoms) == (6 if system == "(H2O)2" else 18)
        assert not atoms.pbc.any()


def test_write_orbmol_relaxation_structures_rejects_atom_identity_changes(
    tmp_path: Path,
) -> None:
    initial = _orbmol_relaxation_batch()
    final = _orbmol_relaxation_batch(displacement=0.02)
    final.atomic_numbers = final.atomic_numbers.copy()
    final.atomic_numbers[0] = 1

    with pytest.raises(ValueError, match="atomic_numbers"):
        write_orbmol_relaxation_structures(
            tmp_path,
            initial_batch=initial,
            final_batch=final,
        )

    assert not list(tmp_path.glob("*.extxyz"))


def test_graph_atoms_from_batch_preserves_periodic_box_and_results() -> None:
    reference = Atoms(
        "OH",
        positions=[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
        cell=np.eye(3) * 9.0,
        pbc=True,
    )
    reference.set_array("molecule_id", np.array([0, 0], dtype=np.int64))
    reference.info["construction_density_g_cm3"] = 1.0
    batch = SimpleNamespace(
        batch_ptr=np.array([0, 2]),
        atomic_numbers=np.array([8, 1]),
        atomic_masses=np.array([15.999, 1.008]),
        positions=np.array([[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]]),
        cell=np.array([np.eye(3) * 10.0]),
        pbc=np.array([[True, True, True]]),
        energy=np.array([[-12.5]]),
        forces=np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
        charges=np.array([-0.4, 0.4]),
        charge=np.array([[0.0]]),
    )

    atoms = graph_atoms_from_batch(
        batch,
        0,
        "periodic result",
        reference_atoms=reference,
        include_results=True,
    )

    assert atoms.pbc.all()
    assert np.allclose(atoms.cell, np.eye(3) * 10.0)
    assert np.allclose(atoms.positions, batch.positions)
    assert np.array_equal(atoms.arrays["molecule_id"], np.array([0, 0]))
    assert np.allclose(atoms.arrays["forces"], batch.forces)
    assert np.allclose(atoms.arrays["charges"], batch.charges)
    assert atoms.info["energy"] == pytest.approx(-12.5)
    assert atoms.info["charge"] == 0
    assert atoms.info["construction_density_g_cm3"] == pytest.approx(1.0)
