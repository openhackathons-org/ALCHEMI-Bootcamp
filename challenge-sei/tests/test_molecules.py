from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge_utils.molecules import (
    build_molecule,
    known_molecules,
    molecule_formula,
    register_molecule,
)


def _manifest_rows():
    import csv

    with (ROOT / "data" / "molecule_manifest.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_every_manifest_molecule_builds_in_code():
    rows = _manifest_rows()
    assert rows, "molecule manifest is empty"
    for row in rows:
        cid = row["candidate_id"]
        assert cid in known_molecules()
        atoms = build_molecule(cid)
        assert len(atoms) > 0
        # fresh copy each call (mutating one must not affect the registry)
        atoms.positions[0] += 100.0
        assert not np.allclose(build_molecule(cid).positions[0], atoms.positions[0])


def test_formulas_match_manifest():
    for row in _manifest_rows():
        assert molecule_formula(row["candidate_id"]) == row["formula"], row["candidate_id"]


def test_geometries_are_sane():
    # bonded-but-not-fused: nearest neighbour within covalent range, no overlaps,
    # molecule connected (no atom stranded far from all others)
    for cid in known_molecules():
        atoms = build_molecule(cid)
        pos = atoms.get_positions()
        d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 0.85, f"{cid}: fused atoms (min dist {d.min():.2f} A)"
        assert d.min(axis=1).max() < 2.0, f"{cid}: stranded atom (nearest {d.min(axis=1).max():.2f} A)"
        # centred near the centroid (embedded geometries are centroid-centred)
        assert np.abs(pos.mean(axis=0)).max() < 1.0


def test_no_structure_files_referenced():
    # the challenge must not depend on structure files
    manifest = (ROOT / "data" / "molecule_manifest.csv").read_text()
    template = (ROOT / "data" / "custom_molecule_manifest_template.csv").read_text()
    assert "structure_path" not in manifest and ".xyz" not in manifest
    assert "structure_path" not in template and ".xyz" not in template
    assert not (ROOT / "data" / "molecules").exists()


def test_register_and_unknown_molecule():
    from ase import Atoms

    with pytest.raises(KeyError, match="register_molecule"):
        build_molecule("definitely_not_registered")
    register_molecule("test_h2", Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)]))
    assert "test_h2" in known_molecules()
    assert len(build_molecule("test_h2")) == 2
    assert molecule_formula("test_h2") == "H2"
