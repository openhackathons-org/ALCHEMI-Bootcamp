"""Plot builders for Part 1.

Every function returns Matplotlib objects and leaves saving or displaying to
the notebook.  Keeping those side effects visible makes the output path and
rendering boundary easy for learners to see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FIGURE_SIZE = (12.0, 7.0)
MD_COLOR = "#276FBF"
DFT_COLOR = "#D95F02"
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#111827"
SYSTEM_DISPLAY_LABELS = {
    "H2O": r"H$_2$O",
    "D2O": r"D$_2$O",
    "(H2O)6": r"cyc-(H$_2$O)$_6$",
    "(D2O)6": r"cyc-(D$_2$O)$_6$",
}
SYSTEM_COLORS = {
    "H2O": MD_COLOR,
    "D2O": DFT_COLOR,
    "(H2O)6": MD_COLOR,
    "(D2O)6": DFT_COLOR,
}
SYSTEM_LINESTYLES = {
    "H2O": "-",
    "D2O": "--",
    "(H2O)6": "-",
    "(D2O)6": "--",
}
COMPONENT_COLORS = {
    "residual_interaction_kJ_mol": "#6B7280",
    "residual_plus_D3_interaction_kJ_mol": "#7C3AED",
    "residual_plus_Coulomb_interaction_kJ_mol": "#0F766E",
    "full_interaction_kJ_mol": "#4D7C0F",
    "B97_3c_interaction_kJ_mol": TEXT_COLOR,
}
COMPONENT_STYLES: dict[str, dict[str, Any]] = {
    "residual_interaction_kJ_mol": {
        "linestyle": ":",
        "marker": "o",
        "markerfacecolor": "none",
    },
    "residual_plus_D3_interaction_kJ_mol": {
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": "none",
    },
    "residual_plus_Coulomb_interaction_kJ_mol": {
        "linestyle": "-.",
        "marker": "^",
        "markerfacecolor": "none",
    },
    "full_interaction_kJ_mol": {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.0,
    },
    "B97_3c_interaction_kJ_mol": {
        "linestyle": "--",
        "marker": "D",
        "markerfacecolor": "white",
        "linewidth": 1.7,
    },
}


def style_axis(axis: Any, *, grid_axis: str = "both") -> None:
    """Apply the shared Part 1 plot treatment to one Matplotlib axis."""

    axis.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    axis.title.set_color(TEXT_COLOR)
    axis.title.set_fontsize(11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")


def display_system_label(label: str) -> str:
    """Return consistent isotope typography without changing data keys."""

    return SYSTEM_DISPLAY_LABELS.get(label, label)


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency message
        raise ImportError("Plotting Part 1 results requires matplotlib") from exc
    return plt


def plot_md_dft_comparison(
    labels: Sequence[str],
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
    reference_comparisons: Mapping[str, Any],
    *,
    wavenumber_limits_cm1: tuple[float, float],
    figure_size: tuple[float, float] = FIGURE_SIZE,
    md_color: str = MD_COLOR,
    dft_color: str = DFT_COLOR,
    title: str = "AIMNet MD with harmonic B97-3c shown for inspection",
    withheld_text: str = (
        "cyclic DFT overlay withheld\nring-persistence gate failed"
    ),
) -> tuple[Any, np.ndarray]:
    """Reproduce the four-lane MD/harmonic comparison without side effects."""

    names = tuple(labels)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("the comparison plot requires four unique system labels")
    low, high = map(float, wavenumber_limits_cm1)
    if not np.isfinite((low, high)).all() or not low < high:
        raise ValueError("wavenumber limits must have increasing finite bounds")

    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=figure_size, sharex=True, sharey=True)
    for panel, (axis, label) in enumerate(zip(axes.flat, names, strict=True)):
        if label in reference_comparisons:
            comparison = reference_comparisons[label]
            mask = (
                (comparison.wavenumber_cm1 >= low)
                & (comparison.wavenumber_cm1 <= high)
            )
            if not np.any(mask):
                raise ValueError(f"{label!r} has no MD points in the visible range")
            wavenumber = comparison.wavenumber_cm1[mask]
            md_visible = comparison.md_intensity_normalized[mask]
            md_visible = md_visible / max(float(md_visible.max()), 1e-30)
            dft_visible = comparison.reference_envelope_normalized[mask]
            dft_visible = dft_visible / max(float(dft_visible.max()), 1e-30)
            axis.plot(
                wavenumber,
                0.58 + 0.34 * md_visible,
                color=md_color,
                linewidth=1.35,
                linestyle="-",
                label="AIMNet MD" if panel == 0 else None,
            )
            axis.plot(
                wavenumber,
                0.08 + 0.34 * dft_visible,
                color=dft_color,
                linewidth=1.1,
                linestyle="--",
                label="B97-3c / 5 ps Hann" if panel == 0 else None,
            )
            stick_mask = (
                (comparison.stick_wavenumber_cm1 >= low)
                & (comparison.stick_wavenumber_cm1 <= high)
            )
            stick_visible = comparison.stick_intensity_normalized[stick_mask]
            if stick_visible.size:
                stick_visible = stick_visible / max(float(stick_visible.max()), 1e-30)
                axis.vlines(
                    comparison.stick_wavenumber_cm1[stick_mask],
                    0.08,
                    0.08 + 0.34 * stick_visible,
                    color="#272727",
                    linewidth=0.8,
                    alpha=0.8,
                    label="raw B97-3c sticks" if panel == 0 else None,
                )
            axis.set_yticks((0.08, 0.58), ("DFT", "MD"))
            panel_title = display_system_label(label)
        else:
            try:
                wavenumber, intensity = spectra[label]
            except KeyError as exc:
                raise ValueError(f"missing MD spectrum for {label!r}") from exc
            wavenumber = np.asarray(wavenumber, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            mask = (wavenumber >= low) & (wavenumber <= high)
            if not np.any(mask):
                raise ValueError(f"{label!r} has no MD points in the visible range")
            normalized = intensity[mask] / max(float(intensity[mask].max()), 1e-30)
            axis.plot(
                wavenumber[mask],
                0.58 + 0.34 * normalized,
                color=md_color,
                linewidth=1.35,
                linestyle="-",
            )
            axis.text(
                0.5,
                0.18,
                withheld_text,
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#7A3E00",
                fontsize=8,
            )
            axis.set_yticks((0.58,), ("MD",))
            panel_title = (
                f"{display_system_label(label)} — cyclic overlay withheld"
            )
        axis.set_title(panel_title)
        axis.set_ylim(0.0, 1.0)
        style_axis(axis, grid_axis="x")

    for axis in axes[-1]:
        axis.set_xlabel("wavenumber / cm$^{-1}$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_topology_timeline(
    timelines: Mapping[str, Any],
    *,
    figure_size: tuple[float, float] = FIGURE_SIZE,
    colors: Mapping[str, str] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot H-bond counts and initial-ring persistence for each trajectory."""

    if not timelines:
        raise ValueError("timelines must not be empty")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True)
    palette = SYSTEM_COLORS | dict(colors or {})
    for label, timeline in timelines.items():
        required = {"time_ps", "H_bonds", "initial_ring_present"}
        missing = required - set(timeline.columns)
        if missing:
            raise ValueError(f"{label!r} timeline is missing {sorted(missing)!r}")
        color = palette.get(label)
        linestyle = SYSTEM_LINESTYLES.get(label, "-")
        axes[0].plot(
            timeline["time_ps"],
            timeline["H_bonds"],
            label=display_system_label(label),
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
        axes[1].step(
            timeline["time_ps"],
            timeline["initial_ring_present"].astype(int),
            where="post",
            label=display_system_label(label),
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
    axes[0].set_ylabel("H bonds")
    axes[1].set_ylabel("initial ring")
    axes[1].set_yticks((0, 1), ("absent", "present"))
    axes[1].set_xlabel("time / ps")
    for axis in axes:
        style_axis(axis)
    axes[0].legend(frameon=False, ncol=max(1, min(4, len(timelines))))
    fig.suptitle("Hydrogen-bond topology across the production trajectory")
    fig.tight_layout()
    return fig, axes


def plot_dimer_interaction_energies(
    table: Any,
    *,
    component_columns: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, Any]:
    """Plot selected ``*_interaction_kJ_mol`` columns against O--O distance."""

    if "distance_angstrom" not in table:
        raise ValueError("table must contain distance_angstrom")
    columns = list(component_columns or ())
    if not columns:
        columns = [
            str(column)
            for column in table.columns
            if str(column).endswith("_interaction_kJ_mol")
        ]
    if not columns:
        raise ValueError("no interaction-energy columns were selected")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"table is missing {sorted(missing)!r}")

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=figure_size)
    display_labels = labels or {}
    palette = COMPONENT_COLORS | dict(colors or {})
    style_overrides = styles or {}
    for column in columns:
        default_label = column.removesuffix("_interaction_kJ_mol").replace("_", " ")
        line_style = dict(COMPONENT_STYLES.get(column, {}))
        line_style.update(style_overrides.get(column, {}))
        line_style.setdefault("marker", "o")
        line_style.setdefault("markersize", 4.0)
        line_style.setdefault("linewidth", 1.25)
        axis.plot(
            table["distance_angstrom"],
            table[column],
            color=palette.get(column),
            label=display_labels.get(column, default_label),
            **line_style,
        )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_xlabel("O–O distance / Å")
    axis.set_ylabel("interaction energy / kJ mol$^{-1}$")
    axis.set_title("Water-dimer interaction ablation against full B97-3c")
    style_axis(axis)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig, axis


__all__ = [
    "COMPONENT_COLORS",
    "COMPONENT_STYLES",
    "DFT_COLOR",
    "FIGURE_SIZE",
    "GRID_COLOR",
    "MD_COLOR",
    "SYSTEM_COLORS",
    "SYSTEM_DISPLAY_LABELS",
    "SYSTEM_LINESTYLES",
    "display_system_label",
    "plot_dimer_interaction_energies",
    "plot_md_dft_comparison",
    "plot_topology_timeline",
    "style_axis",
]
