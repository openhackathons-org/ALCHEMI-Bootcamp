"""Tests for async API client functions."""

import asyncio
import json
import math
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from helpers.api_client import (
    async_run_md,
    async_run_md_or_load_cache,
    async_temperature_sweep,
    snapshot_to_mdatoms,
)
from helpers.cache import cache_exists, save_cache
from helpers.models import BMDAtomicData, BMDConfig, BMDReply, BMDSnapshot


class TestSnapshotToMdatoms:
    @pytest.fixture
    def base_atoms(self):
        return BMDAtomicData(
            coord=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            numbers=[11, 17],
            charge=0,
            mult=1,
            cell=[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
            pbc=[True, True, True],
        )

    def test_basic_conversion(self, base_atoms):
        snap = BMDSnapshot(
            coord=[0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            velocity=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            energy=-50.0,
            cell=[5.1, 0.0, 0.0, 0.0, 5.1, 0.0, 0.0, 0.0, 5.1],
        )
        result = snapshot_to_mdatoms(base_atoms, snap)

        assert result.coord == snap.coord
        assert result.velocity == snap.velocity
        assert result.numbers == base_atoms.numbers
        assert result.pbc == base_atoms.pbc
        assert result.charge == base_atoms.charge
        assert result.mult == base_atoms.mult
        assert result.cell == snap.cell

    def test_preserves_base_cell_when_snap_has_none(self, base_atoms):
        snap = BMDSnapshot(
            coord=[0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            velocity=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            energy=-50.0,
            cell=None,
        )
        result = snapshot_to_mdatoms(base_atoms, snap)
        assert result.cell == base_atoms.cell

    def test_returns_mdatomicdata_type(self, base_atoms):
        snap = BMDSnapshot(
            coord=[0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            velocity=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            energy=-50.0,
        )
        result = snapshot_to_mdatoms(base_atoms, snap)
        assert isinstance(result, BMDAtomicData)


class TestAsyncRunMd:
    @pytest.fixture
    def canned_reply_data(self):
        snap = BMDSnapshot(
            coord=[0.1, 0.2, 0.3],
            velocity=[0.01, 0.02, 0.03],
            energy=-10.0,
            istep=100,
            md_time=0.1,
        )
        cfg = BMDConfig(temperature=300.0, md_time_max=0.1)
        reply = BMDReply(trajectory=[snap], config=cfg, status="Success")
        return json.loads(reply.model_dump_json())

    @pytest.fixture
    def simple_atoms(self):
        return BMDAtomicData(
            coord=[0.0, 0.0, 0.0],
            numbers=[1],
            cell=[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            pbc=[True, True, True],
        )

    @pytest.mark.asyncio
    async def test_successful_response(self, canned_reply_data, simple_atoms):
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=canned_reply_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        cfg = BMDConfig(temperature=300.0, md_time_max=0.1)
        reply = await async_run_md(simple_atoms, cfg, "http://fake:8000", mock_session)

        assert isinstance(reply, BMDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) == 1

    @pytest.mark.asyncio
    async def test_failure_status_raises(self, simple_atoms):
        fail_data = {
            "trajectory": None,
            "config": {"temperature": 300.0, "md_time_max": 0.1},
            "status": "Failed",
            "info": "Simulation diverged",
        }
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=fail_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        cfg = BMDConfig(temperature=300.0, md_time_max=0.1)
        with pytest.raises(RuntimeError, match="MD simulation failed"):
            await async_run_md(simple_atoms, cfg, "http://fake:8000", mock_session)


class TestAsyncRunMdOrLoadCache:
    @pytest.fixture
    def simple_atoms(self):
        return BMDAtomicData(
            coord=[0.0, 0.0, 0.0],
            numbers=[1],
            cell=[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            pbc=[True, True, True],
        )

    @pytest.fixture
    def sample_reply(self):
        snap = BMDSnapshot(
            coord=[0.1, 0.2, 0.3],
            velocity=[0.01, 0.02, 0.03],
            energy=-10.0,
        )
        cfg = BMDConfig(temperature=300.0, md_time_max=0.1)
        return BMDReply(trajectory=[snap], config=cfg, status="Success")

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, simple_atoms, sample_reply):
        with tempfile.TemporaryDirectory() as tmp:
            save_cache(tmp, "test_label", sample_reply)
            mock_session = MagicMock()

            result = await async_run_md_or_load_cache(
                simple_atoms,
                BMDConfig(temperature=300.0),
                "http://fake:8000",
                mock_session,
                tmp,
                "test_label",
                endpoint_live=True,
            )
            assert result.status == "Success"
            # Session should not have been called
            mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cache_no_endpoint_raises(self, simple_atoms):
        with tempfile.TemporaryDirectory() as tmp:
            mock_session = MagicMock()
            with pytest.raises(RuntimeError, match="No cached response"):
                await async_run_md_or_load_cache(
                    simple_atoms,
                    BMDConfig(temperature=300.0),
                    "http://fake:8000",
                    mock_session,
                    tmp,
                    "nonexistent",
                    endpoint_live=False,
                )

    @pytest.mark.asyncio
    async def test_cache_miss_calls_endpoint_and_saves(
        self, simple_atoms, sample_reply
    ):
        with tempfile.TemporaryDirectory() as tmp:
            reply_data = json.loads(sample_reply.model_dump_json())

            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = AsyncMock(return_value=reply_data)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_resp)

            result = await async_run_md_or_load_cache(
                simple_atoms,
                BMDConfig(temperature=300.0),
                "http://fake:8000",
                mock_session,
                tmp,
                "new_label",
                endpoint_live=True,
            )
            assert result.status == "Success"
            assert cache_exists(tmp, "new_label")


class TestAsyncTemperatureSweep:
    def test_empty_temperatures(self):
        atoms = BMDAtomicData(
            coord=[0.0, 0.0, 0.0],
            numbers=[1],
            cell=[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            pbc=[True, True, True],
        )
        temps, densities, replies = asyncio.get_event_loop().run_until_complete(
            async_temperature_sweep(
                temperatures=[],
                mdatoms=atoms,
                server_url="http://fake:8000",
                cache_dir="/tmp/nonexistent",
                endpoint_live=False,
                nvt_time_ps=1.0,
                npt_time_ps=1.0,
                dt=1.0,
                friction=1.0,
                pressure=0.0,
                save_interval=10,
            )
        )
        assert temps == []
        assert densities == []
        assert replies == []

    def test_single_temperature_from_cache(self, cache_dir, nacl_md_atoms):
        """Run sweep for T=200 using cached NVT+NPT responses."""
        if not (
            cache_exists(cache_dir, "nacl_nvt_T200")
            and cache_exists(cache_dir, "nacl_npt_T200")
        ):
            pytest.skip("Cached T=200 responses not available")

        temps, densities, replies = asyncio.get_event_loop().run_until_complete(
            async_temperature_sweep(
                temperatures=[200],
                mdatoms=nacl_md_atoms,
                server_url="http://fake:8000",
                cache_dir=cache_dir,
                endpoint_live=False,
                nvt_time_ps=5.0,
                npt_time_ps=10.0,
                dt=1.0,
                friction=1.0,
                pressure=0.0,
                save_interval=100,
            )
        )
        assert len(temps) == 1
        assert temps[0] == 200
        assert 1.5 <= densities[0] <= 3.0
        assert replies[0] is not None

    def test_results_sorted_by_temperature(self, cache_dir, nacl_md_atoms):
        """Run sweep for T=[250, 200] and verify results are sorted."""
        if not all(
            cache_exists(cache_dir, f"nacl_nvt_T{T}")
            and cache_exists(cache_dir, f"nacl_npt_T{T}")
            for T in [200, 250]
        ):
            pytest.skip("Cached T=200,250 responses not available")

        temps, densities, replies = asyncio.get_event_loop().run_until_complete(
            async_temperature_sweep(
                temperatures=[250, 200],
                mdatoms=nacl_md_atoms,
                server_url="http://fake:8000",
                cache_dir=cache_dir,
                endpoint_live=False,
                nvt_time_ps=5.0,
                npt_time_ps=10.0,
                dt=1.0,
                friction=1.0,
                pressure=0.0,
                save_interval=100,
            )
        )
        assert temps == [200, 250]
        assert len(densities) == 2
        assert all(not math.isnan(d) for d in densities)

    def test_failed_pipeline_returns_nan(self):
        """A temperature with no cache and no endpoint should yield NaN."""
        atoms = BMDAtomicData(
            coord=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            numbers=[11, 17],
            cell=[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
            pbc=[True, True, True],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.warns(RuntimeWarning, match="Pipeline failed"):
                temps, densities, replies = asyncio.get_event_loop().run_until_complete(
                    async_temperature_sweep(
                        temperatures=[999],
                        mdatoms=atoms,
                        server_url="http://fake:8000",
                        cache_dir=tmp,
                        endpoint_live=False,
                        nvt_time_ps=1.0,
                        npt_time_ps=1.0,
                        dt=1.0,
                        friction=1.0,
                        pressure=0.0,
                        save_interval=10,
                    )
                )
            assert len(temps) == 1
            assert math.isnan(densities[0])
            assert replies[0] is None
