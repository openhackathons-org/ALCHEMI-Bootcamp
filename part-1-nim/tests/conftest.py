"""Shared fixtures for ALCHEMI OER Catalyst Screening tests."""

import os
import sys

import pytest

# Ensure helpers package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLAYBOOK_DIR = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR_OER = os.path.join(PLAYBOOK_DIR, "cached_responses", "oer-catalyst-screening")
OUTPUT_DIR = os.path.join(PLAYBOOK_DIR, "outputs")
BGR_SERVER_URL = "http://localhost:8000"


@pytest.fixture
def output_dir():
    return OUTPUT_DIR


@pytest.fixture
def bgr_server_url():
    return BGR_SERVER_URL


@pytest.fixture
def bgr_endpoint_live(bgr_server_url):
    from helpers.api_client import check_endpoint

    return check_endpoint(bgr_server_url)


@pytest.fixture
def nacl_ase():
    """Build NaCl 2x2x2 supercell as ASE Atoms (general-purpose test structure)."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    nacl = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    nacl.make_supercell((2, 2, 2))
    return AseAtomsAdaptor().get_atoms(nacl)


@pytest.fixture
def oer_cache_dir():
    return CACHE_DIR_OER
