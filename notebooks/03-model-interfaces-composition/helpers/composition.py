"""Tested setup and result shaping for the model-composition lesson.

Learner cells keep Toolkit model construction, configuration, execution, and
composition visible. This module owns repository paths, immutable asset checks,
repeated ``Batch`` construction, and compact inspection tables.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase import Atoms

MODEL_ALIAS = "aimnet2-wb97m-d3_0"
EV_TO_KCAL_MOL = 23.060548867
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_D3_SHA256 = "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
_NCI_SHA256 = "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_COLUMNS = {
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
    "fragment",
    "charge",
    "natoms",
    "symbols",
    "positions_angstrom",
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    "ccsd_t_cbs_interaction_energy_kcal_mol",
}


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local asset."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_presentation() -> None:
    """Apply the lesson's plotting style and one pinned warning filter."""

    import matplotlib.pyplot as plt

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True",
        category=UserWarning,
        module="nvalchemi.models.aimnet2",
    )


def load_nci_records(root: Path | None = None) -> pd.DataFrame:
    """Load and verify the pinned three-system NCI curve table."""

    root = _PROJECT_ROOT if root is None else root
    path = root / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
    if _sha256_file(path) != _NCI_SHA256:
        raise RuntimeError(f"Unexpected NCI curve-table checksum: {path}")

    records = pd.read_csv(path)
    missing = _REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(f"NCI curve table is missing columns: {sorted(missing)}")
    if set(records["fragment"]) != {"AB", "A", "B"}:
        raise ValueError("NCI curve table must contain AB, A, and B records")
    if not records.groupby(["system_id", "scale"]).size().eq(3).all():
        raise ValueError("Every system and scale must contain one AB/A/B triplet")
    return records


def model_checkpoint() -> Path:
    """Resolve and verify the supported AIMNet2 checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(MODEL_ALIAS)).resolve()
    if _sha256_file(path) != _MODEL_SHA256:
        raise RuntimeError(f"Unexpected AIMNet2 checkpoint checksum: {path}")
    return path


def d3_parameter_file() -> Path:
    """Return the verified runtime D3 table without creating it."""

    configured = os.environ.get("ALCHEMI_D3_PARAM_FILE")
    if not configured:
        raise RuntimeError("The synchronized D3 parameter file is not configured")
    path = Path(configured).resolve()
    if not path.is_file():
        raise RuntimeError(f"The synchronized D3 parameter file is absent: {path}")
    if _sha256_file(path) != _D3_SHA256:
        raise RuntimeError(f"Unexpected D3 parameter file checksum: {path}")
    return path


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze model parameters while retaining gradients on model inputs."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def atoms_from_record(record: Mapping[str, Any]) -> Atoms:
    """Build one finite ASE structure from a checked table record."""

    positions = np.fromstring(str(record["positions_angstrom"]), sep=" ").reshape(-1, 3)
    atoms = Atoms(symbols=str(record["symbols"]).split(), positions=positions)
    if len(atoms) != int(record["natoms"]):
        raise ValueError("Atom count does not match the NCI record")
    atoms.info["charge"] = int(record["charge"])
    return atoms


def build_batch(
    records: pd.DataFrame,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Repeat the already-taught structure-to-Batch path for selected records."""

    from nvalchemi.data import AtomicData, Batch

    structures = [
        atoms_from_record(record) for record in records.to_dict(orient="records")
    ]
    graphs = [
        AtomicData.from_atoms(structure, device=device, dtype=dtype)
        for structure in structures
    ]
    return Batch.from_data_list(graphs, device=device)


def ab_minus_a_minus_b(values: Any, fragments: list[str] | pd.Series) -> Any:
    """Return the AB - A - B value after validating fragment identity."""

    labels = list(fragments)
    if len(labels) != 3 or any(labels.count(name) != 1 for name in ("AB", "A", "B")):
        raise ValueError("Expected one AB, A, and B value")
    index = {name: labels.index(name) for name in ("AB", "A", "B")}
    return values[index["AB"]] - values[index["A"]] - values[index["B"]]


def component_plot_html(
    values: pd.Series,
    *,
    reference_kcal_mol: float,
) -> str:
    """Render a compact component plot with an explicit HTML alt description."""

    import matplotlib.pyplot as plt

    labels = values.index.tolist()
    colors = ["#8B949E", "#A98B5B", "#6E8CA0", "#76B900"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(labels, values.to_numpy(dtype=float), color=colors[: len(values)])
    ax.axhline(
        reference_kcal_mol,
        color="#F3F4F6",
        linestyle="--",
        linewidth=1.2,
        label="CCSD(T)/CBS at this geometry",
    )
    ax.axhline(0.0, color="#7C8794", linewidth=0.8)
    ax.set_ylabel("Interaction energy (kcal/mol)")
    ax.set_title("AB - A - B interaction-energy accounting")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(frameon=False, loc="best")
    for bar, value in zip(bars, values.to_numpy(dtype=float), strict=True):
        offset = -4 if value < 0 else 4
        vertical = "top" if value < 0 else "bottom"
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical,
            fontsize=9,
        )
    fig.tight_layout()

    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    alt = (
        "Bar chart of AB minus A minus B interaction energies in kcal/mol "
        "for the AIMNet2 checkpoint base, finite Coulomb term, D3(BJ) term, "
        "and complete model, with the single-geometry CCSD(T)/CBS reference."
    )
    return (
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}" '
        'style="display:block;box-sizing:border-box;width:100%;max-width:100%;'
        'height:auto;">'
    )


def model_contract_table(model: Any) -> pd.DataFrame:
    """Shape public model metadata into a compact inspection table."""

    config = model.model_config
    neighbor = config.neighbor_config
    parameter = next(model.parameters(), None)
    rows = {
        "available outputs": ", ".join(sorted(config.outputs)),
        "active outputs": ", ".join(sorted(config.active_outputs)),
        "required inputs": ", ".join(sorted(model.input_data())),
        "optional inputs": ", ".join(sorted(config.optional_inputs)) or "none",
        "parameter dtype": (
            str(parameter.dtype).removeprefix("torch.")
            if parameter is not None
            else "none"
        ),
        "parameter device": str(parameter.device) if parameter is not None else "none",
        "supports periodic systems": config.supports_pbc,
        "requires periodic data": config.needs_pbc,
        "neighbor cutoff": f"{neighbor.cutoff:g} Å" if neighbor else "none",
        "neighbor format": neighbor.format.value if neighbor else "none",
        "neighbor convention": (
            "half list" if neighbor and neighbor.half_list else "full list"
        ),
    }
    return pd.DataFrame({"value": rows})


def output_contract_table(
    outputs: Mapping[str, Any],
    *,
    num_graphs: int,
    num_nodes: int,
) -> pd.DataFrame:
    """Shape output keys, tensor levels, units, dtypes, and devices."""

    semantics = {
        "energy": ("graph", "eV", num_graphs),
        "forces": ("atom", "eV/Å", num_nodes),
        "charges": ("atom", "e", num_nodes),
    }
    rows = []
    for name, value in outputs.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        level, unit, expected_rows = semantics.get(name, ("other", "—", None))
        rows.append(
            {
                "output": name,
                "shape": tuple(tensor.shape),
                "level": level,
                "unit": unit,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "device": str(tensor.device),
                "expected rows": expected_rows,
            }
        )
    return pd.DataFrame(rows).set_index("output")


def pipeline_table(model: Any) -> pd.DataFrame:
    """Describe public pipeline groups without exposing private internals."""

    rows = []
    for group_index, group in enumerate(model.groups):
        for step_index, step in enumerate(group.steps):
            rows.append(
                {
                    "group": group_index,
                    "step": step_index,
                    "model": type(step.model).__name__,
                    "derivatives": "shared autograd"
                    if group.use_autograd
                    else "direct",
                    "wire": step.wire or "matching keys",
                }
            )
    return pd.DataFrame(rows).set_index(["group", "step"])


__all__ = [
    "EV_TO_KCAL_MOL",
    "MODEL_ALIAS",
    "ab_minus_a_minus_b",
    "atoms_from_record",
    "build_batch",
    "component_plot_html",
    "configure_presentation",
    "d3_parameter_file",
    "freeze_model",
    "load_nci_records",
    "model_checkpoint",
    "model_contract_table",
    "output_contract_table",
    "pipeline_table",
]
