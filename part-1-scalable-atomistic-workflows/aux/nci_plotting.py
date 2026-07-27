"""NCI Atlas plots using the shared Part 1 colors and axis style."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import plotting as part1_plotting


NCI_COLORS = {
    "core": part1_plotting.COMPONENT_COLORS["residual_interaction_kJ_mol"],
    "core_plus_d3": part1_plotting.COMPONENT_COLORS[
        "residual_plus_D3_interaction_kJ_mol"
    ],
    "core_plus_coulomb": part1_plotting.COMPONENT_COLORS[
        "residual_plus_Coulomb_interaction_kJ_mol"
    ],
    "full": part1_plotting.COMPONENT_COLORS["full_interaction_kJ_mol"],
    "dft_full": part1_plotting.DFT_COLOR,
    "ccsd_t_cbs": part1_plotting.TEXT_COLOR,
}
NCI_LABELS = {
    "core": "checkpoint base",
    "core_plus_d3": "base + D3",
    "core_plus_coulomb": "base + full Coulomb",
    "full": "complete finite model",
    "dft_full": "DFT-D3",
    "ccsd_t_cbs": "CCSD(T)/CBS",
}
NCI_LINESTYLES = {
    "core": ":",
    "core_plus_d3": "--",
    "core_plus_coulomb": "-.",
    "full": "-",
    "dft_full": "--",
}


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency message
        raise ImportError("Plotting NCI Atlas curves requires matplotlib") from exc
    return plt


def plot_nci_interaction_curves(
    curves: Any,
    *,
    component_columns: Sequence[str] = (
        "core",
        "core_plus_d3",
        "core_plus_coulomb",
        "full",
    ),
    line_reference_columns: Sequence[str] = ("dft_full",),
    point_reference_columns: Sequence[str] = ("ccsd_t_cbs",),
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    spread: tuple[str, str] | None = ("full", "full_std"),
    figure_size: tuple[float, float] | None = None,
    title: str = "Interaction curves across three chemical regimes",
) -> tuple[Any, np.ndarray]:
    """Plot one panel per NCI complex without saving or displaying the figure."""

    component_names = tuple(component_columns)
    line_references = tuple(line_reference_columns)
    point_references = tuple(point_reference_columns)
    required = {
        "system_name",
        "scale",
        *component_names,
        *line_references,
        *point_references,
    }
    if spread is not None:
        required.update(spread)
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"curve table is missing {sorted(missing)!r}")
    if not component_names:
        raise ValueError("at least one model component is required")

    numeric_columns = required - {"system_name"}
    numeric = curves[list(numeric_columns)].apply(
        lambda column: np.asarray(column, dtype=float)
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("curve table contains a non-finite plotted value")
    if spread is not None and (numeric[spread[1]] < 0.0).any():
        raise ValueError("ensemble spread must be non-negative")

    system_names = tuple(dict.fromkeys(str(value) for value in curves["system_name"]))
    if not system_names:
        raise ValueError("curve table contains no systems")
    panel_size = figure_size or (4.4 * len(system_names), 3.8)
    display_labels = NCI_LABELS | dict(labels or {})
    palette = NCI_COLORS | dict(colors or {})

    plt = _pyplot()
    figure, axes_grid = plt.subplots(
        1,
        len(system_names),
        figsize=panel_size,
        sharex=True,
        squeeze=False,
    )
    axes = axes_grid.reshape(-1)
    for axis, system_name in zip(axes, system_names, strict=True):
        group = curves[curves["system_name"].astype(str) == system_name].sort_values(
            "scale"
        )
        if group["scale"].duplicated().any():
            raise ValueError(f"{system_name!r} contains duplicate separation scales")
        for column in component_names:
            axis.plot(
                group["scale"],
                group[column],
                color=palette.get(column),
                linestyle=NCI_LINESTYLES.get(column, "-"),
                marker="o",
                markersize=3.5,
                linewidth=1.6 if column != "full" else 2.0,
                label=display_labels.get(column, column.replace("_", " ")),
            )
        if spread is not None:
            center, width = spread
            axis.fill_between(
                group["scale"],
                group[center] - group[width],
                group[center] + group[width],
                color=palette.get(center),
                alpha=0.15,
                linewidth=0.0,
            )
        for column in line_references:
            axis.plot(
                group["scale"],
                group[column],
                color=palette.get(column),
                linestyle=NCI_LINESTYLES.get(column, "--"),
                linewidth=1.4,
                label=display_labels.get(column, column.replace("_", " ")),
            )
        for column in point_references:
            axis.scatter(
                group["scale"],
                group[column],
                s=20,
                facecolors="white",
                edgecolors=palette.get(column),
                linewidths=1.0,
                label=display_labels.get(column, column.replace("_", " ")),
                zorder=5,
            )
        axis.axhline(0.0, color=part1_plotting.TEXT_COLOR, linewidth=0.7, alpha=0.4)
        axis.set_title(system_name)
        axis.set_xlabel(r"separation $R/R_e$")
        part1_plotting.style_axis(axis)

    axes[0].set_ylabel("interaction energy / kcal mol$^{-1}$")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(title, y=1.13, color=part1_plotting.TEXT_COLOR)
    figure.tight_layout()
    return figure, axes


__all__ = [
    "NCI_COLORS",
    "NCI_LABELS",
    "NCI_LINESTYLES",
    "plot_nci_interaction_curves",
]
