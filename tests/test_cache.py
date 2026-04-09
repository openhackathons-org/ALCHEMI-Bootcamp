"""Test caching layer."""

import os
import tempfile

import pytest

from helpers.cache import cache_exists, load_cache, save_cache
from helpers.models import BMDConfig, BMDReply, BMDSnapshot


class TestCache:
    @pytest.fixture
    def tmp_cache_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def sample_reply(self):
        snap = BMDSnapshot(
            coord=[1.0, 2.0, 3.0],
            velocity=[0.01, 0.02, 0.03],
            energy=-100.5,
            istep=10,
            md_time=0.01,
        )
        cfg = BMDConfig(temperature=300.0, md_time_max=0.01)
        return BMDReply(trajectory=[snap], config=cfg, status="Success")

    def test_cache_roundtrip(self, tmp_cache_dir, sample_reply):
        save_cache(tmp_cache_dir, "test_run", sample_reply)
        assert cache_exists(tmp_cache_dir, "test_run")

        loaded = load_cache(tmp_cache_dir, "test_run", BMDReply)
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
