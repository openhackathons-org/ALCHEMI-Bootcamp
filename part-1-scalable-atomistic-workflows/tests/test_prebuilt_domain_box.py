"""Tests for the checked, prebuilt Part 1 domain-box loader."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import sha256_file  # noqa: E402
from aux.domain import (  # noqa: E402
    PrebuiltDomainBoxBundle,
    PrebuiltDomainBoxError,
    load_prebuilt_domain_box,
)
from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402
from aux.nci_atlas import load_nci_atlas_subset  # noqa: E402


BUNDLE_DIR = (
    PART_DIR / "data" / "domain_decomposition" / "prebuilt_base_box"
)
NCI_DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"


@pytest.fixture(scope="module")
def nci_table() -> pd.DataFrame:
    return load_nci_atlas_subset(NCI_DATA_FILE)


def _copy_bundle(tmp_path: Path) -> Path:
    return Path(
        shutil.copytree(BUNDLE_DIR, tmp_path / "prebuilt-domain-box")
    )


def _load_manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reseal_index(root: Path) -> None:
    lines = [
        f"{sha256_file(root / name)}  {name}"
        for name in ("manifest.json", "structure.extxyz", "preview.png")
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _update_structure_digest(root: Path, manifest: dict[str, object]) -> None:
    path = root / "structure.extxyz"
    digest = sha256_file(path)
    structure = manifest["structure"]
    render = manifest["render"]
    assert isinstance(structure, dict)
    assert isinstance(render, dict)
    structure["sha256"] = digest
    structure["bytes"] = path.stat().st_size
    render["source_sha256"] = digest


def test_loads_the_shipped_plan_structure_and_preview(
    nci_table: pd.DataFrame,
) -> None:
    bundle = load_prebuilt_domain_box(BUNDLE_DIR, nci_table)

    assert isinstance(bundle, PrebuiltDomainBoxBundle)
    assert bundle.bundle_dir == BUNDLE_DIR.resolve()
    assert bundle.preview_path == (BUNDLE_DIR / "preview.png").resolve()
    assert bundle.preview_path.is_file()
    assert bundle.plan.atom_count == 3_200
    assert bundle.plan.molecules_per_species == 128
    assert bundle.plan.molecule_count == 256
    assert bundle.plan.net_charge_e == 0
    assert len(bundle.atoms) == 3_200
    assert bundle.atoms.pbc.tolist() == [True, True, True]
    assert np.allclose(
        bundle.atoms.cell.array,
        np.eye(3) * bundle.plan.box_length_a,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert bundle.validation.density_from_mass_and_cell_g_cm3 == pytest.approx(
        DOMAIN_METHODOLOGY.construction_density_g_cm3
    )
    assert bundle.validation.periodic_min_distance_a is not None
    assert (
        bundle.validation.periodic_min_distance_a
        >= bundle.validation.min_distance_required_a
    )
    assert np.array_equal(
        bundle.atoms.arrays["source_atom_id"],
        np.arange(3_200),
    )
    assert set(np.unique(bundle.atoms.arrays["molecule_id"])) == set(range(256))
    assert np.array_equal(
        bundle.atoms.arrays["molecule_component"],
        bundle.atoms.arrays["molecule_kind"],
    )
    assert bundle.manifest["schema"] == "alchemi.part1-domain-base-box.v1"


def test_rejects_a_changed_indexed_file(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(tmp_path)
    with (root / "structure.extxyz").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(
        PrebuiltDomainBoxError,
        match="SHA-256 mismatch for structure.extxyz",
    ):
        load_prebuilt_domain_box(root, nci_table)


@pytest.mark.parametrize("filename", ["extra.txt", "nested"])
def test_rejects_files_or_directories_outside_the_small_bundle(
    nci_table: pd.DataFrame,
    tmp_path: Path,
    filename: str,
) -> None:
    root = _copy_bundle(tmp_path)
    path = root / filename
    if filename == "nested":
        path.mkdir()
    else:
        path.write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(
        PrebuiltDomainBoxError,
        match="prebuilt domain-box directory has the wrong files",
    ):
        load_prebuilt_domain_box(root, nci_table)


def test_rejects_methodology_drift_even_when_the_index_is_updated(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(tmp_path)
    manifest = _load_manifest(root)
    methodology = manifest["methodology"]
    assert isinstance(methodology, dict)
    methodology["version"] = "changed"
    _write_manifest(root, manifest)
    _reseal_index(root)

    with pytest.raises(
        PrebuiltDomainBoxError,
        match="does not match DOMAIN_METHODOLOGY",
    ):
        load_prebuilt_domain_box(root, nci_table)


def test_rejects_changed_source_atom_identity_after_all_digests_are_updated(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(tmp_path)
    path = root / "structure.extxyz"
    atoms = ase_read(path, format="extxyz")
    atoms.arrays["source_atom_id"][0] = 1
    ase_write(path, atoms, format="extxyz")

    manifest = _load_manifest(root)
    structure = manifest["structure"]
    assert isinstance(structure, dict)
    arrays = structure["arrays"]
    assert isinstance(arrays, dict)
    source_record = arrays["source_atom_id"]
    assert isinstance(source_record, dict)
    reloaded = ase_read(path, format="extxyz")
    source_ids = np.asarray(reloaded.arrays["source_atom_id"])
    source_record["dtype"] = str(source_ids.dtype)
    source_record["shape"] = list(source_ids.shape)
    source_record["sha256"] = _array_sha256(source_ids)
    _update_structure_digest(root, manifest)
    _write_manifest(root, manifest)
    _reseal_index(root)

    with pytest.raises(
        PrebuiltDomainBoxError,
        match="source_atom_id does not match the NCI-derived molecule layout",
    ):
        load_prebuilt_domain_box(root, nci_table)


@pytest.mark.parametrize("change", ["pbc", "cell", "contact"])
def test_rejects_a_structure_that_no_longer_matches_the_plan(
    nci_table: pd.DataFrame,
    tmp_path: Path,
    change: str,
) -> None:
    root = _copy_bundle(tmp_path)
    path = root / "structure.extxyz"
    atoms = ase_read(path, format="extxyz")
    if change == "pbc":
        atoms.set_pbc([True, True, False])
    elif change == "cell":
        atoms.set_cell(atoms.cell.array * 1.01)
    else:
        second_molecule_start = 13
        atoms.positions[second_molecule_start] = atoms.positions[0]
    ase_write(path, atoms, format="extxyz")

    manifest = _load_manifest(root)
    _update_structure_digest(root, manifest)
    _write_manifest(root, manifest)
    _reseal_index(root)

    with pytest.raises(
        PrebuiltDomainBoxError,
        match=(
            "structure.extxyz does not match the construction plan|"
            "prebuilt structure must be periodic"
        ),
    ):
        load_prebuilt_domain_box(root, nci_table)


def test_rejects_a_preview_that_is_not_the_recorded_png(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(tmp_path)
    preview = root / "preview.png"
    payload = preview.read_bytes()
    preview.write_bytes(b"not-png!" + payload[8:])

    manifest = _load_manifest(root)
    render = manifest["render"]
    assert isinstance(render, dict)
    render["output_sha256"] = sha256_file(preview)
    render["bytes"] = preview.stat().st_size
    _write_manifest(root, manifest)
    _reseal_index(root)

    with pytest.raises(PrebuiltDomainBoxError, match="not a valid PNG"):
        load_prebuilt_domain_box(root, nci_table)
