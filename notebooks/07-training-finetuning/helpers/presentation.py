"""Presentation setup and quantitative plots for Part 07."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import HTML
from nvalchemi.data import AtomicData

NVIDIA_GREEN = "#76B900"
WARM_GOLD = "#D6A94A"
COOL_BLUE = "#5DADE2"
MUTED = "#9AA0A6"


def repo_root(start: Path | None = None) -> Path:
    """Find the tutorial root from a notebook-local path."""

    current = (start or Path(__file__)).resolve()
    for parent in (current, *current.parents):
        if (parent / "environment" / "runtime-pins.toml").is_file():
            return parent
    raise FileNotFoundError("could not locate tutorial root")


def configure_presentation() -> None:
    """Load the shared plot style and compact table defaults."""

    root = repo_root()
    plt.style.use(root / "shared" / "alchemi-dark.mplstyle")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 110)


def render_figure(figure: Any, *, alt_text: str) -> HTML:
    """Return bounded HTML whose embedded PNG keeps explicit alt text."""

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    alt = html.escape(alt_text, quote=True)
    return HTML(
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}" '
        'style="display:block;max-width:100%;height:auto;">'
    )


def plot_toy_history(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> Any:
    """Plot the tiny observed train and held-out loss traces."""

    figure, axis = plt.subplots(figsize=(7.4, 3.7))
    axis.scatter(
        training["completed optimizer updates"],
        training["total loss"],
        marker="o",
        color=NVIDIA_GREEN,
        label="Training minibatch MSE",
    )
    axis.plot(
        validation["completed optimizer updates"],
        validation["total loss"],
        marker="s",
        color=WARM_GOLD,
        label="Validation MSE",
    )
    axis.set(
        xlabel="Completed optimizer updates",
        ylabel="MSE (dimensionless score²)",
        title="Four readout-only fine-tuning updates",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def _pair_distance_rows(
    records: Sequence[AtomicData],
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    split_by_id = split_frame.set_index("sample_id")["split"].to_dict()
    rows: list[dict[str, Any]] = []
    for record in records:
        distances = torch.pdist(record.positions).detach().cpu()
        sample_id = int(record.sample_id.item())
        for distance in distances:
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_by_id[sample_id],
                    "pair distance (Å)": float(distance),
                }
            )
    return pd.DataFrame(rows)


def plot_argon_split(
    records: Sequence[AtomicData],
    split_frame: pd.DataFrame,
) -> Any:
    """Show pair-distance coverage for the deterministic split."""

    pair_frame = _pair_distance_rows(records, split_frame)
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    colors = {"train": NVIDIA_GREEN, "validation": WARM_GOLD}
    for split, group in pair_frame.groupby("split", sort=False):
        axis.scatter(
            group["sample_id"],
            group["pair distance (Å)"],
            s=22,
            alpha=0.8,
            color=colors[split],
            label=split,
        )
    axis.set(
        xlabel="Generated structure ID",
        ylabel="Pair distance (Å)",
        title="Ar4 train and validation coverage",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_component_rmse(
    axis: Any,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    column: str,
    label: str,
) -> None:
    axis.plot(
        training["completed optimizer updates"],
        training[column].clip(lower=0).pow(0.5),
        color=NVIDIA_GREEN,
        label="train",
    )
    axis.plot(
        validation["completed optimizer updates"],
        validation[column].clip(lower=0).pow(0.5),
        marker="s",
        color=WARM_GOLD,
        label="validation",
    )
    axis.set(
        xlabel="Completed optimizer updates",
        ylabel=label,
    )
    axis.legend()


def plot_argon_training(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    reference_epsilon_eV: float,
    reference_sigma_A: float,
) -> Any:
    """Plot optimization loss, physical errors, and parameter traces."""

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    loss_axis, energy_axis, force_axis, parameter_axis = axes.flatten()

    loss_axis.plot(
        training["completed optimizer updates"],
        training["total loss"],
        color=NVIDIA_GREEN,
        label="train",
    )
    loss_axis.plot(
        validation["completed optimizer updates"],
        validation["total loss"],
        marker="s",
        color=WARM_GOLD,
        label="validation",
    )
    loss_axis.set(
        xlabel="Completed optimizer updates",
        ylabel="Scaled total loss (dimensionless)",
    )
    loss_axis.set_yscale("log")
    loss_axis.legend()

    _plot_component_rmse(
        energy_axis,
        training,
        validation,
        column="energy MSE (eV²)",
        label="Energy RMSE (eV)",
    )
    _plot_component_rmse(
        force_axis,
        training,
        validation,
        column="force MSE (eV²/Å²)",
        label="Force RMSE (eV/Å)",
    )

    parameter_axis.plot(
        training["completed optimizer updates"],
        training["epsilon (eV)"],
        color=NVIDIA_GREEN,
        label="fitted epsilon",
    )
    parameter_axis.axhline(
        reference_epsilon_eV,
        color=NVIDIA_GREEN,
        linestyle=":",
        label="reference epsilon",
    )
    parameter_axis.set(
        xlabel="Completed optimizer updates",
        ylabel="Epsilon (eV)",
    )
    sigma_axis = parameter_axis.twinx()
    sigma_axis.plot(
        training["completed optimizer updates"],
        training["sigma (Å)"],
        color=COOL_BLUE,
        label="fitted sigma",
    )
    sigma_axis.axhline(
        reference_sigma_A,
        color=COOL_BLUE,
        linestyle=":",
        label="reference sigma",
    )
    sigma_axis.set_ylabel("Sigma (Å)")
    handles_left, labels_left = parameter_axis.get_legend_handles_labels()
    handles_right, labels_right = sigma_axis.get_legend_handles_labels()
    parameter_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=8,
    )

    figure.suptitle("Generated-Ar fit and held-out checks")
    figure.tight_layout()
    return figure
