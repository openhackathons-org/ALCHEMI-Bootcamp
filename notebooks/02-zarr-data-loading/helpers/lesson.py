"""Checked data and temporary-workspace support for Part 02."""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Sequence
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from ase import Atoms
from ase.io import read
from IPython.display import HTML
from matplotlib.figure import Figure

EXPECTED_EXTXYZ_SHA256 = (
    "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
)
EXPECTED_MANIFEST_SHA256 = (
    "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def molecule_source_path(root: Path = REPO_ROOT) -> Path:
    """Return the pinned extxyz source after verifying its digest."""

    path = root / "data" / "nci_atlas" / "ir-molecule-library.extxyz"
    if sha256_file(path) != EXPECTED_EXTXYZ_SHA256:
        raise RuntimeError(f"Unexpected molecule-library checksum: {path}")
    return path


def load_molecule_manifest(root: Path = REPO_ROOT) -> pd.DataFrame:
    """Load and verify metadata without materializing extxyz frames."""

    molecule_source_path(root)
    manifest_path = (
        root / "data" / "nci_atlas" / "ir-molecule-library-manifest.json"
    )
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected molecule-manifest checksum: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in manifest["molecules"]:
        rows.append(
            {
                "order": int(record["order"]),
                "label": str(record["label"]),
                "formula": str(record["formula"]),
                "atoms": int(record["atom_count"]),
                "charge": int(record["formal_charge"]),
                "source": f"{record['dataset']} / {record['system_id']}",
            }
        )

    frame = pd.DataFrame(rows).sort_values("order", ignore_index=True)
    if len(frame) != 32 or int(frame["atoms"].sum()) != 322:
        raise RuntimeError("The pinned 32-molecule / 322-atom identity check failed.")
    if frame["charge"].ne(0).any():
        raise RuntimeError("This lesson expects neutral molecules only.")
    return frame


def load_molecule_collection(
    root: Path = REPO_ROOT,
) -> tuple[list[Atoms], pd.DataFrame]:
    """Load the pinned molecule library and verify its manifest."""

    extxyz_path = molecule_source_path(root)
    frame = load_molecule_manifest(root)
    atoms = list(read(extxyz_path, index=":"))
    if len(atoms) != len(frame):
        raise RuntimeError("The manifest and extxyz record counts differ.")

    for structure, record in zip(atoms, frame.itertuples(index=False), strict=True):
        if len(structure) != record.atoms:
            raise RuntimeError(f"Atom-count mismatch for {record.label}.")
        if structure.get_chemical_formula() != record.formula:
            raise RuntimeError(f"Formula mismatch for {record.label}.")
    return atoms, frame


def plot_record_layout(
    atom_counts: Sequence[int],
    atoms_ptr: Sequence[int],
) -> Figure:
    """Plot variable record sizes beside cumulative atom-row boundaries."""

    if len(atoms_ptr) != len(atom_counts) + 1:
        raise ValueError("atoms_ptr must contain one boundary beyond the records.")

    style = REPO_ROOT / "shared" / "alchemi-dark.mplstyle"
    with plt.style.context(style):
        figure, (count_axis, pointer_axis) = plt.subplots(1, 2, figsize=(10, 3.4))
        count_axis.bar(range(len(atom_counts)), atom_counts, color="#76B900")
        count_axis.set(xlabel="Record index", ylabel="Atoms per record")
        pointer_axis.plot(atoms_ptr, color="#76B900", marker="o", markersize=3)
        pointer_axis.set(
            xlabel="Record boundary",
            ylabel="Cumulative atom rows",
        )
        figure.suptitle("Variable records map to contiguous atom rows")
        figure.tight_layout()
    plt.close(figure)
    return figure


def figure_to_html(figure: Figure, alt: str) -> HTML:
    """Render a Matplotlib figure with explicit HTML alternative text."""

    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return HTML(
        f'<img src="data:image/png;base64,{encoded}" alt="{escape(alt)}" '
        'style="display:block;width:100%;max-width:100%;height:auto;">'
    )


def tutorial_workspace() -> tuple[TemporaryDirectory[str], Path]:
    """Create a temporary directory that stays alive for one notebook run."""

    owner = TemporaryDirectory(prefix="alchemi-zarr-")
    return owner, Path(owner.name)
