"""Checked inputs and presentation support for the hooks lesson."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ase import Atoms
from ase.io import read

from ._environment import consume_dm_tree_set_notice

_EXPECTED_EXTXYZ_SHA256 = (
    "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
)
_EXPECTED_MANIFEST_SHA256 = (
    "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
)
_MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_molecule_selection(
    labels: Sequence[str],
    root: Path | None = None,
) -> tuple[list[Atoms], pd.DataFrame]:
    """Load and verify an ordered molecule selection from the shared collection."""

    root = _PROJECT_ROOT if root is None else root
    data_dir = root / "data" / "nci_atlas"
    extxyz_path = data_dir / "ir-molecule-library.extxyz"
    manifest_path = data_dir / "ir-molecule-library-manifest.json"
    if _sha256_file(extxyz_path) != _EXPECTED_EXTXYZ_SHA256:
        raise RuntimeError(f"Unexpected molecule-library checksum: {extxyz_path}")
    if _sha256_file(manifest_path) != _EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected molecule-manifest checksum: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = manifest["molecules"]
    record_by_label = {str(record["label"]): record for record in records}
    missing = [label for label in labels if label not in record_by_label]
    if missing:
        raise KeyError(f"Molecule labels missing from the pinned manifest: {missing}")

    all_atoms = list(read(extxyz_path, index=":"))
    selected_atoms: list[Atoms] = []
    rows: list[dict[str, Any]] = []
    for system_id, label in enumerate(labels):
        record = record_by_label[label]
        atoms = all_atoms[int(record["extxyz_index"])].copy()
        if len(atoms) != int(record["atom_count"]):
            raise RuntimeError(f"Atom-count mismatch for {label}.")
        if atoms.get_chemical_formula() != str(record["formula"]):
            raise RuntimeError(f"Formula mismatch for {label}.")
        atoms.info["charge"] = int(record["formal_charge"])
        selected_atoms.append(atoms)
        rows.append(
            {
                "system_id": system_id,
                "label": label,
                "formula": str(record["formula"]),
                "atoms": int(record["atom_count"]),
                "charge": int(record["formal_charge"]),
                "source": f"{record['dataset']} / {record['system_id']}",
            }
        )

    frame = pd.DataFrame(rows)
    if frame["charge"].ne(0).any():
        raise RuntimeError("The hooks lesson expects neutral molecules.")
    return selected_atoms, frame


def model_checkpoint() -> Path:
    """Resolve and verify the AIMNet checkpoint selected for this lesson."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(_MODEL_ALIAS)).resolve()
    digest = _sha256_file(path)
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"Checkpoint checksum mismatch: {digest}")
    return path


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze model parameters while preserving gradients with respect to positions."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def configure_presentation() -> None:
    """Apply the shared plot style and hide pinned runtime warnings."""

    import matplotlib.pyplot as plt

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")
    consume_dm_tree_set_notice()
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True",
        category=UserWarning,
        module="nvalchemi.models.aimnet2",
    )
    warnings.filterwarnings(
        "ignore",
        message="Can't initialize NVML",
        category=UserWarning,
    )


def plot_energy_history(history: pd.DataFrame) -> Any:
    """Display graph energy changes and the batch mean for scheduled hook calls."""

    import matplotlib.pyplot as plt
    from IPython.display import display

    required = {"step", "molecule", "energy change (meV)"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Energy history is missing columns: {sorted(missing)}")

    comparison_styles = [
        ("#00A3E0", "o", "--"),
        ("#D6A94A", "s", "-."),
        ("#AAB2BD", "^", ":"),
    ]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for style, (label, records) in zip(
        comparison_styles,
        history.groupby("molecule", sort=False),
        strict=False,
    ):
        color, marker, linestyle = style
        axis.plot(
            records["step"],
            records["energy change (meV)"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.88,
            label=label,
        )

    batch_mean = history.groupby("step", as_index=False)["energy change (meV)"].mean()
    axis.plot(
        batch_mean["step"],
        batch_mean["energy change (meV)"],
        color="#76B900",
        marker="D",
        linewidth=3.0,
        label="Batch mean",
    )
    axis.set(
        title="Energy recorded at scheduled model evaluations",
        xlabel="FIRE2 step",
        ylabel="Energy change from first record [meV]",
    )
    axis.legend(title="Hook output", frameon=False)
    figure.tight_layout()
    display(
        figure,
        metadata={
            "alt": "Per-molecule and batch-mean energy changes recorded "
            "at the configured hook steps."
        },
    )
    plt.close(figure)
    return figure
