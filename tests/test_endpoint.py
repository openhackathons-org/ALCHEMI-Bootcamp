"""Test endpoint connectivity and API hello world.

These tests require the live BMD NIM endpoint on localhost:8000
and/or the BGR NIM endpoint on localhost:8890.
They are skipped if the respective endpoint is unreachable.
"""

import pytest

from helpers.api_client import check_endpoint, run_md, run_bgr
from helpers.models import MDAtomicData, MDConfig


@pytest.fixture
def require_endpoint(endpoint_live):
    if not endpoint_live:
        pytest.skip("BMD endpoint not available at localhost:8000")


@pytest.fixture
def require_bgr_endpoint(bgr_endpoint_live):
    if not bgr_endpoint_live:
        pytest.skip("BGR endpoint not available at localhost:8890")


class TestEndpointHealth:
    def test_health_check_returns_bool(self, server_url):
        result = check_endpoint(server_url)
        assert isinstance(result, bool)

    def test_health_check_unreachable(self):
        result = check_endpoint("http://localhost:59999", timeout=2)
        assert result is False


class TestAPIHelloWorld:
    def test_h2_md_with_box(self, require_endpoint, server_url):
        """H2 molecule in a periodic box — minimal MD run."""
        h2 = MDAtomicData(
            coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
            numbers=[1, 1],
            cell=[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            pbc=[True, True, True],
        )
        cfg = MDConfig(
            temperature=300.0,
            dt=1.0,
            nvt=True,
            npt=False,
            md_time_max=0.01,
            save_interval=1,
        )
        reply = run_md(h2, cfg, server_url, timeout=30)
        assert reply.status == "Success"
        assert len(reply.trajectory) > 0
        assert reply.trajectory[-1].energy < 0  # H2 should have negative PE

    def test_h2_nonperiodic_fails(self, require_endpoint, server_url):
        """Non-periodic H2 should fail (endpoint requires PBC)."""
        h2 = MDAtomicData(
            coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
            numbers=[1, 1],
        )
        cfg = MDConfig(
            temperature=300.0,
            dt=1.0,
            nvt=True,
            npt=False,
            md_time_max=0.01,
            save_interval=1,
        )
        with pytest.raises(Exception):
            run_md(h2, cfg, server_url, timeout=30)


class TestBGREndpointHealth:
    def test_health_check_returns_bool(self, bgr_server_url):
        result = check_endpoint(bgr_server_url)
        assert isinstance(result, bool)

    def test_health_check_unreachable(self):
        result = check_endpoint("http://localhost:59998", timeout=2)
        assert result is False


class TestBGRAPIHelloWorld:
    def test_nacl_bgr_no_cellopt(
        self, require_bgr_endpoint, bgr_server_url, nacl_bgr_atom
    ):
        """NaCl BGR relaxation without cell optimisation."""
        reply = run_bgr([nacl_bgr_atom], bgr_server_url, cellopt=False, timeout=60)
        assert reply.status == "Success"
        assert len(reply.atoms) == 1
        assert reply.atoms[0].converged is True

    def test_nacl_bgr_with_cellopt(
        self, require_bgr_endpoint, bgr_server_url, nacl_bgr_atom
    ):
        """NaCl BGR relaxation with cell optimisation."""
        reply = run_bgr([nacl_bgr_atom], bgr_server_url, cellopt=True, timeout=60)
        assert reply.status == "Success"
        assert len(reply.atoms) == 1
        assert reply.atoms[0].energy < 0
