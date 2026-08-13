"""Numerical checks for the generated-argon training example."""

from __future__ import annotations

import math

import helpers
import pytest
import torch
from ase import Atoms
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors

EPSILON_EV = 0.0104
SIGMA_ANGSTROM = 3.40
CUTOFF_ANGSTROM = 7.0


def _records(count: int = 8) -> list[AtomicData]:
    return helpers.generate_argon_records(
        count=count,
        seed=712,
        epsilon_eV=EPSILON_EV,
        sigma_A=SIGMA_ANGSTROM,
        cutoff_A=CUTOFF_ANGSTROM,
        dtype=torch.float64,
        device="cpu",
    )


def test_generated_argon_labels_are_deterministic_and_balanced() -> None:
    first = _records()
    second = _records()

    assert len(first) == 8
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        assert left.num_nodes == 4
        assert left.atomic_numbers.tolist() == [18, 18, 18, 18]
        assert tuple(left.positions.shape) == (4, 3)
        assert tuple(left.energy.shape) == (1, 1)
        assert tuple(left.forces.shape) == (4, 3)
        assert left.positions.dtype == torch.float64
        assert left.energy.dtype == torch.float64
        assert left.forces.dtype == torch.float64
        assert int(left.sample_id.item()) == index
        torch.testing.assert_close(left.positions, right.positions)
        torch.testing.assert_close(left.energy, right.energy)
        torch.testing.assert_close(left.forces, right.forces)
        torch.testing.assert_close(
            left.forces.sum(dim=0),
            torch.zeros(3, dtype=torch.float64),
            atol=2.0e-12,
            rtol=0.0,
        )


def test_reference_force_matches_central_energy_difference() -> None:
    positions = _records(1)[0].positions.clone()
    energy, forces = helpers.lj_energy_forces(
        positions,
        epsilon_eV=EPSILON_EV,
        sigma_A=SIGMA_ANGSTROM,
        cutoff_A=CUTOFF_ANGSTROM,
    )
    step = 1.0e-5
    displaced_plus = positions.clone()
    displaced_minus = positions.clone()
    displaced_plus[0, 1] += step
    displaced_minus[0, 1] -= step

    plus = helpers.lj_energy_forces(
        displaced_plus,
        epsilon_eV=EPSILON_EV,
        sigma_A=SIGMA_ANGSTROM,
        cutoff_A=CUTOFF_ANGSTROM,
    )[0]
    minus = helpers.lj_energy_forces(
        displaced_minus,
        epsilon_eV=EPSILON_EV,
        sigma_A=SIGMA_ANGSTROM,
        cutoff_A=CUTOFF_ANGSTROM,
    )[0]
    finite_difference_force = -(plus - minus) / (2.0 * step)

    assert energy.shape == (1, 1)
    assert forces.shape == (4, 3)
    assert float(forces[0, 1]) == pytest.approx(
        float(finite_difference_force),
        rel=2.0e-6,
        abs=2.0e-8,
    )


def test_neighbor_wrapper_matches_reference_and_is_rigid_motion_invariant() -> None:
    records = _records(2)
    model = helpers.TrainableLennardJones(
        epsilon_eV=EPSILON_EV,
        sigma_A=SIGMA_ANGSTROM,
        cutoff_A=CUTOFF_ANGSTROM,
    ).double()
    batch = Batch.from_data_list(records)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    output = model(batch)

    torch.testing.assert_close(output["energy"], batch.energy, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(output["forces"], batch.forces, atol=2.0e-11, rtol=0.0)

    angle = math.pi / 5.0
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    transformed = [
        AtomicData(
            positions=record.positions @ rotation.T
            + torch.tensor([2.0, -1.0, 0.4], dtype=torch.float64),
            atomic_numbers=record.atomic_numbers.clone(),
        )
        for record in records
    ]
    transformed_batch = Batch.from_data_list(transformed)
    compute_neighbors(
        transformed_batch,
        config=model.model_config.neighbor_config,
    )
    transformed_output = model(transformed_batch)

    torch.testing.assert_close(
        transformed_output["energy"],
        output["energy"],
        atol=2.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed_output["forces"],
        output["forces"] @ rotation.T,
        atol=2.0e-11,
        rtol=0.0,
    )


def test_neighbor_shift_convention_uses_the_nearest_periodic_image() -> None:
    atoms = Atoms(
        "Ar2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    record = AtomicData.from_atoms(atoms, dtype=torch.float64)
    batch = Batch.from_data_list([record])
    model = helpers.TrainableLennardJones(
        epsilon_eV=0.002,
        sigma_A=0.18,
        cutoff_A=1.0,
    ).double()
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    output = model(batch)
    expected_energy, expected_forces = helpers.lj_energy_forces(
        torch.tensor([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=torch.float64),
        epsilon_eV=0.002,
        sigma_A=0.18,
        cutoff_A=1.0,
    )

    torch.testing.assert_close(
        output["energy"],
        expected_energy,
        atol=2.0e-12,
        rtol=0.0,
    )
    assert float(output["forces"].detach().norm()) == pytest.approx(
        float(expected_forces.norm()),
        rel=1.0e-10,
    )


def test_checkpoint_spec_rebuilds_without_pickle() -> None:
    model = helpers.TrainableLennardJones(
        epsilon_eV=0.007,
        sigma_A=3.1,
        cutoff_A=CUTOFF_ANGSTROM,
    ).double()
    rebuilt = model.checkpoint_spec().build()

    assert isinstance(rebuilt, helpers.TrainableLennardJones)
    assert rebuilt.cutoff_A == pytest.approx(CUTOFF_ANGSTROM)
    assert rebuilt.epsilon_eV.item() == pytest.approx(0.007)
    assert rebuilt.sigma_A.item() == pytest.approx(3.1)
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(rebuilt.log_epsilon, model.log_epsilon)
    torch.testing.assert_close(rebuilt.log_sigma, model.log_sigma)


def test_built_in_lj_accepts_transferred_parameters() -> None:
    record = _records(1)[0]
    batch = Batch.from_data_list([record])
    built_in = LennardJonesModelWrapper(
        epsilon=EPSILON_EV,
        sigma=SIGMA_ANGSTROM,
        cutoff=CUTOFF_ANGSTROM,
        switch_width=0.0,
        half_list=True,
    )
    compute_neighbors(batch, config=built_in.model_config.neighbor_config)
    output = built_in(batch)

    torch.testing.assert_close(output["energy"], batch.energy, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(output["forces"], batch.forces, atol=2.0e-11, rtol=0.0)
