"""Test BGR NIM endpoint connectivity.

These tests require the live BGR NIM endpoint.
They are skipped if the endpoint is unreachable.
"""

import pytest

from helpers.api_client import check_endpoint, run_bgr
from helpers.models import ase_to_atomic_data


@pytest.fixture
def require_bgr_endpoint(bgr_endpoint_live):
    if not bgr_endpoint_live:
        pytest.skip("BGR endpoint not available")


@pytest.fixture
def nacl_bgr_atom(nacl_ase):
    """Build NaCl 2x2x2 supercell as BGRAtomicData for BGR."""
    return ase_to_atomic_data(nacl_ase, _id="nacl_supercell")


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
