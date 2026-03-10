"""Test caching layer."""

import os
import tempfile

import pytest

from helpers.cache import cache_exists, load_cache, save_cache
from helpers.models import BGRReply, MDConfig, MDReply, MDSnapshot


class TestCache:
    @pytest.fixture
    def tmp_cache_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def sample_reply(self):
        snap = MDSnapshot(
            coord=[1.0, 2.0, 3.0],
            velocity=[0.01, 0.02, 0.03],
            energy=-100.5,
            istep=10,
            md_time=0.01,
        )
        cfg = MDConfig(temperature=300.0, md_time_max=0.01)
        return MDReply(trajectory=[snap], config=cfg, status="Success")

    def test_cache_roundtrip(self, tmp_cache_dir, sample_reply):
        save_cache(tmp_cache_dir, "test_run", sample_reply)
        assert cache_exists(tmp_cache_dir, "test_run")

        loaded = load_cache(tmp_cache_dir, "test_run", MDReply)
        assert loaded.status == "Success"
        assert len(loaded.trajectory) == 1
        assert loaded.trajectory[0].energy == pytest.approx(-100.5)

    def test_cache_not_exists(self, tmp_cache_dir):
        assert not cache_exists(tmp_cache_dir, "nonexistent")

    def test_save_creates_directory(self, tmp_cache_dir, sample_reply):
        nested = os.path.join(tmp_cache_dir, "sub", "dir")
        save_cache(nested, "test_run", sample_reply)
        assert cache_exists(nested, "test_run")

    def test_cache_json_format(self, tmp_cache_dir, sample_reply):
        import json

        path = save_cache(tmp_cache_dir, "test_run", sample_reply)
        data = json.loads(path.read_text())
        assert "trajectory" in data
        assert "config" in data
        assert "status" in data


class TestCachedResponses:
    """Verify that cached responses from live endpoint runs are valid."""

    def _skip_if_no_cache(self, cache_dir, label):
        if not cache_exists(cache_dir, label):
            pytest.skip(f"Cached response '{label}' not available")

    def test_h2_hello_world_cached(self, cache_dir):
        self._skip_if_no_cache(cache_dir, "h2_hello_world")
        reply = load_cache(cache_dir, "h2_hello_world", MDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 0

    def test_nacl_nvt_equil_cached(self, cache_dir):
        self._skip_if_no_cache(cache_dir, "nacl_nvt_equil")
        reply = load_cache(cache_dir, "nacl_nvt_equil", MDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 5

    def test_nacl_npt_prod_cached(self, cache_dir):
        self._skip_if_no_cache(cache_dir, "nacl_npt_prod")
        reply = load_cache(cache_dir, "nacl_npt_prod", MDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 10

    @pytest.mark.parametrize("T", [200, 300, 400])
    def test_temperature_sweep_cached(self, cache_dir, T):
        self._skip_if_no_cache(cache_dir, f"nacl_npt_T{T}")
        reply = load_cache(cache_dir, f"nacl_npt_T{T}", MDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 0

    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_conformer_cached(self, cache_dir, i):
        self._skip_if_no_cache(cache_dir, f"naph_conf{i}_nvt_500K")
        reply = load_cache(cache_dir, f"naph_conf{i}_nvt_500K", MDReply)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 0

    def test_nacl_bgr_relax_cached(self, cache_dir):
        self._skip_if_no_cache(cache_dir, "nacl_bgr_relax")
        reply = load_cache(cache_dir, "nacl_bgr_relax", BGRReply)
        assert reply.status == "Success"
        assert len(reply.atoms) == 1
