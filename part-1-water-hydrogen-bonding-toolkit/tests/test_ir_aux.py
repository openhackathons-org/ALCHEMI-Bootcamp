"""CPU tests for dependency-light Part 1 auxiliary mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import (  # noqa: E402
    load_ir_trajectory_arrays,
    save_ir_trajectory,
    trajectory_graph_frames,
)
from aux.checkpoint import validate_b973c_external_components  # noqa: E402
from aux.diagnostics import cluster_integrity, mass_only_invariance  # noqa: E402
from aux.structures import (  # noqa: E402
    DEUTERIUM_MASS_U,
    HYDROGEN_MASS_U,
    make_ir_structures,
    make_water_dimer,
)


@dataclass(frozen=True)
class _Trajectory:
    dipoles_e_angstrom: np.ndarray
    charge_sums_e: np.ndarray
    kinetic_energies_eV: np.ndarray
    total_energies_eV: np.ndarray
    positions_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    atomic_masses_u: np.ndarray
    batch_idx: np.ndarray
    batch_ptr: np.ndarray
    dt_fs: float


def test_ir_structures_are_mass_only_pairs() -> None:
    structures, labels = make_ir_structures()

    assert labels == ["H2O", "D2O", "(H2O)6", "(D2O)6"]
    for left, right in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(
            structures[left].numbers, structures[right].numbers
        )
        np.testing.assert_allclose(
            structures[left].positions, structures[right].positions
        )
        assert not structures[left].pbc.any()
        assert not structures[right].pbc.any()
    h_mass = structures[0].get_masses()[structures[0].numbers == 1][0]
    d_mass = structures[1].get_masses()[structures[1].numbers == 1][0]
    assert h_mass == pytest.approx(HYDROGEN_MASS_U)
    assert d_mass == pytest.approx(DEUTERIUM_MASS_U)


def test_water_dimer_geometry_declares_seed_not_reference() -> None:
    dimer = make_water_dimer(2.90)
    oxygen = dimer.positions[dimer.numbers == 8]
    hydrogen = dimer.positions[dimer.numbers == 1]

    assert len(dimer) == 6
    assert np.linalg.norm(oxygen[1] - oxygen[0]) == pytest.approx(2.90)
    assert np.min(np.linalg.norm(hydrogen - oxygen[1], axis=1)) < 2.0
    assert "seed" in dimer.info


def test_mass_only_diagnostic_accepts_numpy_batch() -> None:
    structures, _ = make_ir_structures()
    numbers = np.concatenate([atoms.numbers for atoms in structures])
    positions = np.concatenate([atoms.positions for atoms in structures])
    masses = np.concatenate([atoms.get_masses() for atoms in structures])
    ptr = np.cumsum([0, *(len(atoms) for atoms in structures)])
    batch = SimpleNamespace(
        positions=positions,
        atomic_numbers=numbers,
        atomic_masses=masses,
        batch_ptr=ptr,
    )
    outputs = {
        "energy": np.array([[1.0], [1.0], [2.0], [2.0]]),
        "forces": np.zeros_like(positions),
        "charges": np.zeros(len(numbers)),
    }

    result = mass_only_invariance(batch, outputs)

    assert result["D_over_H_mass"] == pytest.approx(DEUTERIUM_MASS_U / HYDROGEN_MASS_U)
    outputs["charges"][ptr[1]] = 1e-3
    with pytest.raises(AssertionError, match="monomer: isotope charges changed"):
        mass_only_invariance(batch, outputs)


def _two_graph_trajectory() -> _Trajectory:
    monomer, _, hexamer, _ = make_ir_structures()[0]
    numbers = np.concatenate((monomer.numbers, hexamer.numbers))
    masses = np.concatenate((monomer.get_masses(), hexamer.get_masses()))
    positions = np.concatenate((monomer.positions, hexamer.positions))
    ptr = np.array([0, len(monomer), len(numbers)])
    idx = np.repeat(np.arange(2), np.diff(ptr))
    frames = np.stack((positions, positions))
    return _Trajectory(
        dipoles_e_angstrom=np.zeros((2, 2, 3)),
        charge_sums_e=np.zeros((2, 2)),
        kinetic_energies_eV=np.ones((2, 2)),
        total_energies_eV=np.ones((2, 2)),
        positions_angstrom=frames,
        atomic_numbers=numbers,
        atomic_masses_u=masses,
        batch_idx=idx,
        batch_ptr=ptr,
        dt_fs=0.5,
    )


def test_cluster_integrity_recognizes_generated_ring() -> None:
    trajectory = _two_graph_trajectory()

    result = cluster_integrity(trajectory, 1)

    assert result["oxygen_count"] == 6
    assert result["all_frames_initial_ring"] is True
    assert result["max_oxygen_components"] == 1
    assert result["max_OH_angstrom"] < 1.1


def test_trajectory_archive_round_trip_without_toolkit(tmp_path: Path) -> None:
    trajectory = _two_graph_trajectory()
    path = tmp_path / "trajectory.npz"

    manifest = save_ir_trajectory(path, trajectory, ["H2O", "(H2O)6"])
    arrays, labels = load_ir_trajectory_arrays(path)
    frames = trajectory_graph_frames(trajectory, 1, [0, 1], label="(H2O)6")

    assert manifest["frames"] == 2
    assert len(manifest["sha256"]) == 64
    assert labels == ["H2O", "(H2O)6"]
    np.testing.assert_allclose(
        arrays["positions_angstrom"], trajectory.positions_angstrom
    )
    assert [atoms.info["step"] for atoms in frames] == [0, 1]


def test_checkpoint_contract_is_exact() -> None:
    metadata = {
        "needs_coulomb": True,
        "needs_dispersion": True,
        "coulomb_mode": "sr_embedded",
        "coulomb_sr_rc": 4.6,
        "d3_params": {"s6": 1.0, "s8": 1.5, "a1": 0.37, "a2": 4.1},
    }

    assert validate_b973c_external_components(metadata)["s8"] == 1.5
    metadata["d3_params"] = {**metadata["d3_params"], "s8": 1.4}
    with pytest.raises(ValueError, match="Unexpected B97-3c D3 parameter"):
        validate_b973c_external_components(metadata)


def test_aux_mechanics_do_not_hide_toolkit_batch_or_pipeline_construction() -> None:
    mechanics = (
        "artifacts.py",
        "capture.py",
        "checkpoint.py",
        "diagnostics.py",
        "electrostatics.py",
        "structures.py",
    )
    sources = [(PART_DIR / "aux" / name).read_text() for name in mechanics]

    assert "nvalchemi" not in sources[-1]
    assert all("AtomicData.from_atoms(" not in source for source in sources)
    assert all("Batch.from_data_list(" not in source for source in sources)
    assert all("PipelineModelWrapper(" not in source for source in sources)
