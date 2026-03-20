"""Test Pydantic models and ASE conversion utilities."""

import pytest

from helpers.models import (
    BMDAtomicData,
    BMDConfig,
    BMDReply,
    BMDRequest,
    BMDSnapshot,
    BGRAtomicData,
    BGRRequest,
    ase_to_atomic_data,
    ase_to_md_atomic_data,
    atomic_data_to_ase,
)


class TestMDModels:
    def test_md_atomic_data_minimal(self):
        atoms = BMDAtomicData(
            coord=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            numbers=[1, 1],
        )
        assert len(atoms.numbers) == 2
        assert len(atoms.coord) == 6
        assert atoms.cell is None
        assert atoms.pbc is None

    def test_md_atomic_data_periodic(self):
        atoms = BMDAtomicData(
            coord=[0.0, 0.0, 0.0],
            numbers=[11],
            cell=[5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
            pbc=[True, True, True],
        )
        assert atoms.pbc == [True, True, True]
        assert len(atoms.cell) == 9

    def test_md_config_defaults(self):
        cfg = BMDConfig()
        assert cfg.temperature == 300.0
        assert cfg.dt == 1.0
        assert cfg.nvt is True
        assert cfg.npt is False
        assert cfg.save_interval == 100

    def test_md_config_custom(self):
        cfg = BMDConfig(
            temperature=500.0,
            dt=0.5,
            nvt=True,
            npt=True,
            friction=2.0,
            pressure=1.0,
            md_time_max=5.0,
            save_interval=10,
        )
        assert cfg.temperature == 500.0
        assert cfg.npt is True
        assert cfg.pressure == 1.0

    def test_md_request_serialization(self):
        atoms = BMDAtomicData(coord=[0.0, 0.0, 0.0], numbers=[1])
        cfg = BMDConfig(md_time_max=0.01)
        req = BMDRequest(atoms=atoms, config=cfg)
        d = req.model_dump()
        assert "atoms" in d
        assert "config" in d
        assert d["config"]["md_time_max"] == 0.01

    def test_md_snapshot(self):
        snap = BMDSnapshot(
            coord=[1.0, 2.0, 3.0],
            velocity=[0.1, 0.2, 0.3],
            energy=-10.5,
            istep=100,
            md_time=0.1,
        )
        assert snap.energy == -10.5
        assert snap.istep == 100

    def test_md_snapshot_string_coercion(self):
        """The live API returns string-typed floats; Pydantic must coerce them."""
        snap = BMDSnapshot(
            coord=["1.0", "2.0", "3.0"],
            velocity=["0.1", "0.2", "0.3"],
            energy="-10.5",
            cell=["5.0", "0.0", "0.0", "0.0", "5.0", "0.0", "0.0", "0.0", "5.0"],
            stress=["0.0"] * 9,
            istep=0,
            md_time="0.001",
        )
        assert isinstance(snap.energy, float)
        assert isinstance(snap.coord[0], float)
        assert isinstance(snap.md_time, float)

    def test_md_reply(self):
        snap = BMDSnapshot(
            coord=[1.0, 2.0, 3.0],
            velocity=[0.1, 0.2, 0.3],
            energy=-10.0,
        )
        cfg = BMDConfig()
        reply = BMDReply(trajectory=[snap], config=cfg, status="Success")
        assert reply.status == "Success"
        assert len(reply.trajectory) == 1


class TestBGRModels:
    def test_atomic_data(self):
        ad = BGRAtomicData(
            coord=[0.0, 0.0, 0.0],
            numbers=[11],
            cell=[5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
            pbc=[True, True, True],
        )
        assert ad.charge == 0
        assert ad.mult == 1

    def test_bgr_request(self):
        ad = BGRAtomicData(coord=[0.0, 0.0, 0.0], numbers=[11])
        req = BGRRequest(atoms=[ad])
        assert len(req.atoms) == 1
        assert req.cellopt is False


class TestASEConversion:
    def test_ase_to_md_atomic_data_roundtrip(self, nacl_ase):
        md_data = ase_to_md_atomic_data(nacl_ase)
        assert len(md_data.numbers) == 64
        assert len(md_data.coord) == 64 * 3
        assert md_data.pbc == [True, True, True]
        assert md_data.cell is not None
        assert len(md_data.cell) == 9

    def test_ase_to_atomic_data(self, nacl_ase):
        ad = ase_to_atomic_data(nacl_ase, structure_id="test")
        assert ad.structure_id == "test"
        assert len(ad.numbers) == 64

    def test_atomic_data_to_ase(self):
        ad = BGRAtomicData(
            coord=[0.0, 0.0, 0.0, 2.82, 2.82, 2.82],
            numbers=[11, 17],
            cell=[5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
            pbc=[True, True, True],
        )
        atoms = atomic_data_to_ase(ad)
        assert len(atoms) == 2
        assert atoms.get_volume() == pytest.approx(5.64**3)
