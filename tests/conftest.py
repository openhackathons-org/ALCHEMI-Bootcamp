"""Shared fixtures for ALCHEMI BMD Materials Playbook tests."""

import os
import sys

import pytest

# Ensure helpers package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLAYBOOK_DIR = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR_MATERIALS = os.path.join(PLAYBOOK_DIR, "cached_responses", "materials")
CACHE_DIR_CONFORMER = os.path.join(
    PLAYBOOK_DIR, "cached_responses", "conformer-stability"
)
OUTPUT_DIR = os.path.join(PLAYBOOK_DIR, "outputs")
SERVER_URL = "http://localhost:8000"
BGR_SERVER_URL = "http://localhost:8890"


@pytest.fixture
def cache_dir():
    return CACHE_DIR_MATERIALS


@pytest.fixture
def conformer_cache_dir():
    return CACHE_DIR_CONFORMER


@pytest.fixture
def output_dir():
    return OUTPUT_DIR


@pytest.fixture
def server_url():
    return SERVER_URL


@pytest.fixture
def bgr_server_url():
    return BGR_SERVER_URL


@pytest.fixture
def endpoint_live(server_url):
    from helpers.api_client import check_endpoint

    return check_endpoint(server_url)


@pytest.fixture
def bgr_endpoint_live(bgr_server_url):
    from helpers.api_client import check_endpoint

    return check_endpoint(bgr_server_url)


@pytest.fixture
def nacl_md_atoms():
    """Build NaCl 2x2x2 supercell as MDAtomicData."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    from helpers.models import ase_to_md_atomic_data

    nacl = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    nacl.make_supercell((2, 2, 2))
    nacl_ase = AseAtomsAdaptor().get_atoms(nacl)
    return ase_to_md_atomic_data(nacl_ase)


@pytest.fixture
def nacl_ase():
    """Build NaCl 2x2x2 supercell as ASE Atoms."""
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
def nacl_bgr_atom(nacl_ase):
    """Build NaCl 2x2x2 supercell as AtomicData for BGR."""
    from helpers.models import ase_to_atomic_data

    return ase_to_atomic_data(nacl_ase, _id="nacl_supercell")
