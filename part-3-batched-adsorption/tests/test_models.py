"""Test Pydantic models and ASE conversion utilities."""

import pytest

from helpers.models import (
    AtomicStructurePayload,
    OptimizationResult,
    RelaxationBatchResult,
    RelaxationRequest,
    ase_to_atomic_data,
    atomic_data_to_ase,
)


class TestRelaxationModels:
    def test_atomic_data(self):
        ad = AtomicStructurePayload(
            coord=[0.0, 0.0, 0.0],
            numbers=[11],
            cell=[5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
            pbc=[True, True, True],
        )
        assert ad.charge == 0
        assert ad.mult == 1

    def test_relaxation_request(self):
        ad = AtomicStructurePayload(coord=[0.0, 0.0, 0.0], numbers=[11])
        req = RelaxationRequest(atoms=[ad])
        assert len(req.atoms) == 1
        assert req.cellopt is False

    def test_optimization_result_string_coercion(self):
        """Older JSON caches may store numeric values as strings."""
        result = OptimizationResult(
            coord=["1.0", "2.0", "3.0"],
            numbers=[1],
            converged=True,
            optimizer_nsteps="7",
            energy="-10.5",
            forces=["0.0", "0.1", "0.2"],
        )
        assert isinstance(result.energy, float)
        assert isinstance(result.coord[0], float)
        assert result.optimizer_nsteps == 7

    def test_relaxation_batch_result(self):
        result = OptimizationResult(
            coord=[1.0, 2.0, 3.0],
            numbers=[1],
            converged=True,
            optimizer_nsteps=3,
            energy=-10.0,
            forces=[0.0, 0.0, 0.0],
        )
        reply = RelaxationBatchResult(atoms=[result], status="Success")
        assert reply.status == "Success"
        assert len(reply.atoms) == 1


class TestASEConversion:
    def test_ase_to_atomic_data(self, nacl_ase):
        ad = ase_to_atomic_data(nacl_ase, structure_id="test")
        assert ad.structure_id == "test"
        assert len(ad.numbers) == 64

    def test_atomic_data_to_ase(self):
        ad = AtomicStructurePayload(
            coord=[0.0, 0.0, 0.0, 2.82, 2.82, 2.82],
            numbers=[11, 17],
            cell=[5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
            pbc=[True, True, True],
        )
        atoms = atomic_data_to_ase(ad)
        assert len(atoms) == 2
        assert atoms.get_volume() == pytest.approx(5.64**3)
