"""Plot builders for Part 1.

Every function returns Matplotlib objects and leaves saving or displaying to
the notebook. Keeping those side effects visible makes the output path and
rendering step easy for learners to see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FIGURE_SIZE = (12.0, 7.0)
MD_COLOR = "#276FBF"
DFT_COLOR = "#D95F02"
EXPERIMENT_COLOR = "#4D7C0F"
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


def plot_domain_decomposition(
    capacity_table: Any,
    distributed_table: Any,
    *,
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, np.ndarray]:
    """Plot checked Torch allocation and repeated multi-GPU workflow times.

    Failed capacity or distributed attempts stay in the input tables but are
    not plotted as coordinates. This avoids drawing a line through work that
    did not complete.
    """

    required_capacity = {"atom_count", "success", "torch_peak_allocated_gb"}
    required_distributed = {
        "atom_count",
        "world_size",
        "success",
        "measurement_role",
        "wall_time_s",
        "wall_time_q1_s",
        "wall_time_q3_s",
    }
    missing_capacity = required_capacity - set(capacity_table.columns)
    missing_distributed = required_distributed - set(distributed_table.columns)
    if missing_capacity:
        raise ValueError(
            "capacity results are missing " + ", ".join(sorted(missing_capacity))
        )
    if missing_distributed:
        raise ValueError(
            "distributed results are missing " + ", ".join(sorted(missing_distributed))
        )

    capacity_ok = capacity_table.loc[capacity_table["success"].astype(bool)].copy()
    distributed_ok = distributed_table.loc[
        distributed_table["success"].astype(bool)
        & distributed_table["measurement_role"].eq("steady_timing")
    ].copy()
    if capacity_ok.empty or distributed_ok.empty:
        raise ValueError("the domain plot requires checked successful rows")

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=figure_size)

    capacity_ok = capacity_ok.sort_values("atom_count")
    axes[0].plot(
        capacity_ok["atom_count"],
        capacity_ok["torch_peak_allocated_gb"],
        color=MD_COLOR,
        marker="o",
        linewidth=1.6,
    )
    if "device_memory_gb" in capacity_ok:
        hbm_values = np.asarray(
            capacity_ok["device_memory_gb"].dropna().unique(), dtype=float
        )
        if hbm_values.size == 1:
            axes[0].axhline(
                hbm_values[0],
                color="#9A3412",
                linestyle="--",
                linewidth=1.2,
                label=f"device memory ({hbm_values[0]:g} GB)",
            )
            axes[0].legend(frameon=False, fontsize=8)
    axes[0].set(
        title="Single-GPU capacity",
        xlabel="atoms in one periodic system",
        ylabel="Torch peak allocated / GB",
    )
    style_axis(axes[0])

    for atom_count, rows in distributed_ok.groupby("atom_count", sort=True):
        rows = rows.sort_values("world_size")
        median_s = np.asarray(rows["wall_time_s"], dtype=float)
        q1_s = np.asarray(rows["wall_time_q1_s"], dtype=float)
        q3_s = np.asarray(rows["wall_time_q3_s"], dtype=float)
        if (
            not np.isfinite((median_s, q1_s, q3_s)).all()
            or np.any(q1_s > median_s)
            or np.any(q3_s < median_s)
        ):
            raise ValueError("steady timing rows have invalid median or quartiles")
        axes[1].errorbar(
            rows["world_size"],
            median_s,
            yerr=np.vstack((median_s - q1_s, q3_s - median_s)),
            marker="o",
            linewidth=1.6,
            capsize=4,
            label=f"{int(atom_count):,} atoms",
        )
    axes[1].set(
        title="Same-input DomainParallel time",
        xlabel="GPUs",
        ylabel="partition → 2 evaluations → gather / s",
    )
    axes[1].set_xticks(
        sorted(distributed_ok["world_size"].astype(int).unique().tolist())
    )
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])
    figure.tight_layout()
    return figure, axes


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
    omitted_text: str = ("cyclic DFT overlay not shown\ninitial ring changed"),
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
            mask = (comparison.wavenumber_cm1 >= low) & (
                comparison.wavenumber_cm1 <= high
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
            stick_mask = (comparison.stick_wavenumber_cm1 >= low) & (
                comparison.stick_wavenumber_cm1 <= high
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
                omitted_text,
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#7A3E00",
                fontsize=8,
            )
            axis.set_yticks((0.58,), ("MD",))
            panel_title = f"{display_system_label(label)}: cyclic overlay not shown"
        axis.set_title(panel_title)
        axis.set_ylim(0.0, 1.0)
        style_axis(axis, grid_axis="x")

    for axis in axes[-1]:
        axis.set_xlabel("wavenumber / cm$^{-1}$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_monomer_ir_comparison(
    reference_comparisons: Mapping[str, Any],
    experimental_fundamentals: Any,
    *,
    labels: Sequence[str] = ("H2O", "D2O"),
    harmonic_mode_indices: Mapping[str, Sequence[int]] | None = None,
    wavenumber_limits_cm1: tuple[float, float] = (500.0, 4200.0),
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, np.ndarray]:
    """Plot MD, harmonic DFT, and observed monomer fundamentals on separate lanes.

    The three sources do not share an intensity scale.  MD and the finite-window
    DFT envelope are normalized independently within the visible region;
    experiment is represented only by observed fundamental positions because
    the bundled experimental reference intentionally contains no intensities or
    digitized spectrum.
    """

    names = tuple(labels)
    if names != ("H2O", "D2O"):
        raise ValueError("the monomer comparison requires labels ('H2O', 'D2O')")
    low, high = map(float, wavenumber_limits_cm1)
    if not np.isfinite((low, high)).all() or not low < high:
        raise ValueError("wavenumber limits must have increasing finite bounds")
    required_columns = {"isotopologue", "mode_index", "wavenumber_cm1"}
    missing_columns = required_columns - set(experimental_fundamentals.columns)
    if missing_columns:
        raise ValueError(
            "experimental fundamentals are missing "
            + ", ".join(sorted(missing_columns))
        )

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True, sharey=True)
    lane_base = {"experiment": 0.06, "dft": 0.37, "md": 0.68}
    lane_height = 0.23
    for panel, (axis, label) in enumerate(zip(axes, names, strict=True)):
        try:
            comparison = reference_comparisons[label]
        except KeyError as exc:
            raise ValueError(f"missing harmonic comparison for {label!r}") from exc

        wavenumber = np.asarray(comparison.wavenumber_cm1, dtype=float)
        mask = (wavenumber >= low) & (wavenumber <= high)
        if not np.any(mask):
            raise ValueError(f"{label!r} has no MD points in the visible range")
        md = np.asarray(comparison.md_intensity_normalized, dtype=float)[mask]
        md /= max(float(md.max()), 1e-30)
        dft = np.asarray(comparison.reference_envelope_normalized, dtype=float)[mask]
        dft /= max(float(dft.max()), 1e-30)
        axis.plot(
            wavenumber[mask],
            lane_base["md"] + lane_height * md,
            color=MD_COLOR,
            linewidth=1.35,
            label="AIMNet predicted-charge MD" if panel == 0 else None,
        )
        axis.plot(
            wavenumber[mask],
            lane_base["dft"] + lane_height * dft,
            color=DFT_COLOR,
            linewidth=1.1,
            linestyle="--",
            label="B97-3c harmonic / 5 ps Hann" if panel == 0 else None,
        )

        sticks = np.asarray(comparison.stick_wavenumber_cm1, dtype=float)
        stick_mask = (sticks >= low) & (sticks <= high)
        stick_intensity = np.asarray(
            comparison.stick_intensity_normalized, dtype=float
        )[stick_mask]
        if stick_intensity.size:
            stick_intensity /= max(float(stick_intensity.max()), 1e-30)
            axis.vlines(
                sticks[stick_mask],
                lane_base["dft"],
                lane_base["dft"] + lane_height * stick_intensity,
                color="#272727",
                linewidth=0.8,
                alpha=0.8,
                label="raw B97-3c sticks" if panel == 0 else None,
            )
            if harmonic_mode_indices is not None:
                try:
                    mode_indices = tuple(harmonic_mode_indices[label])
                except KeyError as exc:
                    raise ValueError(
                        f"missing harmonic mode indices for {label!r}"
                    ) from exc
                if len(mode_indices) != sticks.size:
                    raise ValueError(
                        f"{label!r} harmonic mode indices do not match its sticks"
                    )
                for mode_index, position, height in zip(
                    np.asarray(mode_indices)[stick_mask],
                    sticks[stick_mask],
                    stick_intensity,
                    strict=True,
                ):
                    axis.annotate(
                        rf"$\nu_{{{int(mode_index)}}}$",
                        (position, lane_base["dft"] + lane_height * height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#272727",
                    )

        observed = experimental_fundamentals[
            experimental_fundamentals["isotopologue"] == label
        ].sort_values("mode_index")
        if len(observed) != 3:
            raise ValueError(f"{label!r} requires exactly three experimental markers")
        observed_wavenumber = observed["wavenumber_cm1"].to_numpy(dtype=float)
        if not np.isfinite(observed_wavenumber).all():
            raise ValueError(
                f"{label!r} experimental markers contain non-finite values"
            )
        axis.vlines(
            observed_wavenumber,
            lane_base["experiment"],
            lane_base["experiment"] + 0.16,
            color=EXPERIMENT_COLOR,
            linewidth=1.25,
            label="observed gas-phase fundamentals" if panel == 0 else None,
        )
        axis.scatter(
            observed_wavenumber,
            np.full(3, lane_base["experiment"] + 0.16),
            marker="^",
            s=26,
            color=EXPERIMENT_COLOR,
            zorder=3,
        )
        for mode_index, position in zip(
            observed["mode_index"].to_numpy(dtype=int),
            observed_wavenumber,
            strict=True,
        ):
            axis.annotate(
                rf"$\nu_{{{mode_index}}}$",
                (position, lane_base["experiment"] + 0.17),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=EXPERIMENT_COLOR,
            )

        axis.set_title(display_system_label(label))
        axis.set_xlim(low, high)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(
            (
                lane_base["experiment"] + 0.08,
                lane_base["dft"] + 0.11,
                lane_base["md"] + 0.11,
            ),
            (
                "experiment\npositions",
                "B97-3c\nharmonic",
                "AIMNet MD\nfinite T",
            ),
        )
        axis.set_xlabel("wavenumber / cm$^{-1}$")
        style_axis(axis, grid_axis="x")

    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Water monomer IR: three sources, three explicit comparison lanes")
    fig.text(
        0.5,
        0.015,
        "MD and DFT are independently max-normalized; experiment shows positions only.",
        ha="center",
        color=TEXT_COLOR,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    return fig, axes


def plot_harmonic_monomer_comparison(
    table: Any,
    *,
    labels: Sequence[str] = ("H2O", "D2O"),
    wavenumber_limits_cm1: tuple[float, float] = (500.0, 4200.0),
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, np.ndarray]:
    """Compare mapped harmonic frequencies and show each model's intensities.

    Both calculations report intensities in ``km / mol``, but they use
    different dipole models: the complete AIMNet checkpoint-base + Coulomb +
    D3 calculation uses AIMNet predicted point charges while B97-3c uses an
    electronic dipole derivative.  The absolute stick heights are therefore
    shown for inspection, not scored against each other.
    Experimental fundamentals supply positions only, so their green baseline
    markers do not encode an intensity.  A marker shape identifies each mapped
    mode across both calculations and experiment.
    """

    numeric_columns = (
        "AIMNet+Coulomb+D3_harmonic_cm-1",
        "AIMNet_point_charge_IR_km_mol",
        "B97-3c_harmonic_cm-1",
        "B97-3c_IR_intensity_km_mol",
        "observed_gas_cm-1",
    )
    frequency_columns = (
        "AIMNet+Coulomb+D3_harmonic_cm-1",
        "B97-3c_harmonic_cm-1",
        "observed_gas_cm-1",
    )
    intensity_columns = (
        "AIMNet_point_charge_IR_km_mol",
        "B97-3c_IR_intensity_km_mol",
    )
    required = {"system", "mode", *numeric_columns}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"harmonic comparison table is missing {sorted(missing)!r}")
    names = tuple(labels)
    if names != ("H2O", "D2O"):
        raise ValueError("the harmonic monomer comparison requires H2O and D2O")
    low, high = map(float, wavenumber_limits_cm1)
    if not np.isfinite((low, high)).all() or not low < high:
        raise ValueError("wavenumber limits must have increasing finite bounds")

    prepared_rows: dict[str, Any] = {}
    mode_order: tuple[str, ...] | None = None
    maximum_intensity = 0.0
    for label in names:
        rows = table[table["system"] == label].copy()
        if len(rows) != 3:
            raise ValueError(f"{label!r} requires exactly three mapped modes")
        try:
            values = rows[list(numeric_columns)].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label!r} comparison columns must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{label!r} comparison contains non-finite values")
        if np.any(rows[list(frequency_columns)].to_numpy(dtype=float) <= 0.0):
            raise ValueError("harmonic and observed wavenumbers must be positive")
        intensities = rows[list(intensity_columns)].to_numpy(dtype=float)
        if np.any(intensities < 0.0):
            raise ValueError("double-harmonic IR intensities must be non-negative")

        mode_names = tuple(str(mode).strip() for mode in rows["mode"])
        if any(not mode for mode in mode_names) or len(set(mode_names)) != 3:
            raise ValueError(f"{label!r} requires three unique, non-empty mode names")
        rows["_mode_name"] = mode_names
        if mode_order is None:
            mode_order = mode_names
        elif set(mode_names) != set(mode_order):
            raise ValueError("H2O and D2O must use the same three mapped modes")
        rows = rows.set_index("_mode_name").loc[list(mode_order)].reset_index()
        prepared_rows[label] = rows
        maximum_intensity = max(
            maximum_intensity,
            float(np.max(intensities, initial=0.0)),
        )

    assert mode_order is not None  # names always contains H2O and D2O
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True, sharey=True)
    mode_markers = dict(zip(mode_order, ("o", "s", "D"), strict=True))
    upper_limit = 1.18 * maximum_intensity if maximum_intensity > 0.0 else 1.0
    for panel, (axis, label) in enumerate(zip(axes, names, strict=True)):
        rows = prepared_rows[label]
        aimnet_position = rows["AIMNet+Coulomb+D3_harmonic_cm-1"].to_numpy(dtype=float)
        aimnet_intensity = rows["AIMNet_point_charge_IR_km_mol"].to_numpy(dtype=float)
        dft_position = rows["B97-3c_harmonic_cm-1"].to_numpy(dtype=float)
        dft_intensity = rows["B97-3c_IR_intensity_km_mol"].to_numpy(dtype=float)
        observed_position = rows["observed_gas_cm-1"].to_numpy(dtype=float)

        for mode, aimnet_x, aimnet_y, dft_x, dft_y, observed_x in zip(
            rows["_mode_name"],
            aimnet_position,
            aimnet_intensity,
            dft_position,
            dft_intensity,
            observed_position,
            strict=True,
        ):
            marker = mode_markers[mode]
            axis.vlines(
                aimnet_x,
                0.0,
                aimnet_y,
                color=MD_COLOR,
                linewidth=2.0,
                zorder=2,
            )
            axis.scatter(
                [aimnet_x],
                [aimnet_y],
                marker=marker,
                s=34,
                color=MD_COLOR,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            axis.vlines(
                dft_x,
                0.0,
                dft_y,
                color=DFT_COLOR,
                linewidth=1.7,
                linestyles="--",
                zorder=2,
            )
            axis.scatter(
                [dft_x],
                [dft_y],
                marker=marker,
                s=34,
                facecolor="white",
                edgecolor=DFT_COLOR,
                linewidth=1.2,
                zorder=4,
            )
            axis.scatter(
                [observed_x],
                [0.0],
                marker=marker,
                s=42,
                facecolor=EXPERIMENT_COLOR,
                edgecolor="white",
                linewidth=0.6,
                clip_on=False,
                zorder=5,
            )

        axis.set_title(display_system_label(label))
        axis.set_xlim(low, high)
        axis.set_ylim(0.0, upper_limit)
        axis.set_xlabel("wavenumber / cm$^{-1}$")
        style_axis(axis)

    from matplotlib.lines import Line2D

    source_handles = (
        Line2D(
            [0],
            [0],
            color=MD_COLOR,
            linewidth=2.0,
            label=(
                "AIMNet + Coulomb + D3 frequency; AIMNet point-charge dipole intensity"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=DFT_COLOR,
            linewidth=1.7,
            linestyle="--",
            label="B97-3c electronic dipole derivative",
        ),
        Line2D(
            [0],
            [0],
            color=EXPERIMENT_COLOR,
            marker="|",
            markersize=9,
            linestyle="none",
            label="experiment (positions only)",
        ),
    )
    mode_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#6B7280",
            marker=mode_markers[mode],
            markersize=6,
            linestyle="none",
            label=mode,
        )
        for mode in mode_order
    )
    axes[0].legend(
        handles=source_handles,
        frameon=False,
        loc="upper left",
        fontsize=8,
        title="Source",
        title_fontsize=8,
    )
    axes[1].legend(
        handles=mode_handles,
        frameon=False,
        loc="upper left",
        fontsize=8,
        title="Mode marker",
        title_fontsize=8,
    )
    axes[0].set_ylabel("model-specific integrated intensity / km mol$^{-1}$")
    fig.suptitle("Water monomer harmonic frequencies and model-specific intensities")
    fig.text(
        0.5,
        0.015,
        "Frequency differences are scored; absolute intensities are not compared "
        "across dipole models. Experiment shows positions only.",
        ha="center",
        color=TEXT_COLOR,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    return fig, axes


def plot_distributed_pipeline_scaling(
    bundle: Any,
    *,
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, np.ndarray]:
    """Plot strong and weak ``DistributedPipeline`` evidence separately.

    The bundle is expected to be returned by
    :func:`aux.benchmark_results.load_distributed_benchmark_bundle`.  Successful
    repeats remain visible behind the medians; failed cases are counted in the
    allocation labels rather than silently dropped.
    """

    strong = bundle.strong_summary.copy()
    weak = bundle.weak_summary.copy()
    runs = bundle.runs.copy()
    required_summary = {
        "nodes",
        "successful_runs",
        "failed_runs",
        "baseline_nodes",
    }
    for name, table in (("strong", strong), ("weak", weak)):
        missing = required_summary - set(table.columns)
        if missing:
            raise ValueError(f"{name} summary is missing {sorted(missing)!r}")
        if table.empty:
            raise ValueError(f"{name} summary is empty")
    required_runs = {"mode", "nodes", "success", "elapsed_s"}
    if required_runs - set(runs.columns):
        raise ValueError("distributed runs are missing plotting columns")

    def baseline(summary: Any) -> tuple[int | None, float | None]:
        usable = summary.loc[summary["successful_runs"] > 0]
        if usable.empty:
            return None, None
        row = usable.iloc[0]
        return int(row["nodes"]), float(row["median_elapsed_s"])

    strong_baseline_nodes, strong_baseline_elapsed = baseline(strong)
    weak_baseline_nodes, weak_baseline_elapsed = baseline(weak)

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    panels = (
        (
            axes[0],
            strong,
            runs[(runs["mode"] == "strong") & runs["success"]],
            "speedup_vs_baseline",
            strong_baseline_nodes,
            strong_baseline_elapsed,
            "speedup vs measured baseline",
            "Fixed total work",
        ),
        (
            axes[1],
            weak,
            runs[(runs["mode"] == "weak") & runs["success"]],
            "weak_elapsed_efficiency_vs_baseline",
            weak_baseline_nodes,
            weak_baseline_elapsed,
            "elapsed-time efficiency",
            "Fixed work per two-GPU pipeline",
        ),
    )
    for panel_index, (
        axis,
        summary,
        successful,
        metric,
        baseline_nodes,
        baseline_elapsed,
        ylabel,
        title,
    ) in enumerate(panels):
        nodes = summary["nodes"].to_numpy(dtype=float)
        values = summary[metric].to_numpy(dtype=float)
        if baseline_elapsed is not None:
            raw_values = baseline_elapsed / successful["elapsed_s"].to_numpy(
                dtype=float
            )
            axis.scatter(
                successful["nodes"],
                raw_values,
                color=MD_COLOR,
                alpha=0.28,
                s=24,
                label="successful repeats" if panel_index == 0 else None,
            )
        axis.plot(
            nodes,
            values,
            color=MD_COLOR,
            marker="o",
            linewidth=1.8,
            label="median" if panel_index == 0 else None,
        )
        if panel_index == 0 and baseline_nodes is not None:
            ideal = nodes / baseline_nodes
            ideal_label = f"ideal from {baseline_nodes}-node baseline"
        elif panel_index == 1 and baseline_nodes is not None:
            ideal = np.ones_like(nodes)
            ideal_label = "ideal constant elapsed time"
        else:
            ideal = None
            ideal_label = None
        if ideal is not None:
            axis.plot(
                nodes,
                ideal,
                color="#6B7280",
                linestyle="--",
                linewidth=1.1,
                label=ideal_label,
            )
        finite_values = values[np.isfinite(values)]
        scale = max(
            1.0,
            float(np.max(finite_values)) if finite_values.size else 0.0,
            float(np.max(ideal)) if ideal is not None else 0.0,
        )
        failed_only_y = 0.04 * scale
        for row in summary.itertuples(index=False):
            row_value = float(getattr(row, metric))
            if np.isfinite(row_value):
                annotation_y = row_value
            else:
                annotation_y = failed_only_y
                axis.scatter(
                    [float(row.nodes)],
                    [annotation_y],
                    color="#B42318",
                    marker="x",
                    s=45,
                    linewidths=1.5,
                    label="all repeats failed",
                )
            axis.annotate(
                f"{int(row.successful_runs)} pass / {int(row.failed_runs)} fail",
                (float(row.nodes), annotation_y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_COLOR,
            )
        axis.set_xticks(nodes.astype(int))
        axis.set_xlabel("H100 nodes (one GPU per node)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(bottom=0.0)
        style_axis(axis)

    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
    )
    fig.suptitle(
        "Saved Toolkit DistributedPipeline timing check",
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        "Small communication test · fixed buffers · one H100 GPU per node",
        ha="center",
        color=TEXT_COLOR,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.86))
    return fig, axes


def plot_pipeline_campaign(
    bundle: Any,
    *,
    figure_size: tuple[float, float] = FIGURE_SIZE,
) -> tuple[Any, np.ndarray]:
    """Plot wall time and GPU cost for the prerecorded water campaign.

    Every successful repeat remains visible. Medians and interquartile ranges
    come from the verified summary returned by
    :func:`aux.pipeline_campaign_results.load_pipeline_campaign_bundle`.
    Failed repeats are counted above their route rather than silently removed.
    """

    route_order = ("fused_1gpu", "pipeline_2gpu", "pipeline_4gpu")
    route_labels = {
        "fused_1gpu": "1 node · 1 GPU\nfused stages",
        "pipeline_2gpu": "2 nodes · 2 GPUs\none pipeline",
        "pipeline_4gpu": "4 nodes · 4 GPUs\ntwo pipelines",
    }
    route_colors = {
        "fused_1gpu": "#6B7280",
        "pipeline_2gpu": "#0F766E",
        "pipeline_4gpu": "#7C3AED",
    }
    runs = bundle.runs.copy()
    summary = bundle.summary.copy().set_index("route")
    required_runs = {
        "route",
        "success",
        "elapsed_s",
        "gpu_seconds_per_structure",
    }
    required_summary = {
        "successful_runs",
        "failed_runs",
        "median_elapsed_s",
        "elapsed_q25_s",
        "elapsed_q75_s",
        "median_gpu_seconds_per_structure",
        "speedup_vs_1gpu",
    }
    if required_runs - set(runs.columns):
        raise ValueError("campaign runs are missing plotting columns")
    if required_summary - set(summary.columns):
        raise ValueError("campaign summary is missing plotting columns")
    if set(route_order) - set(summary.index):
        raise ValueError("campaign summary does not cover all three routes")
    manifest = getattr(bundle, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError("campaign bundle is missing its manifest")
    campaign = manifest.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign manifest is missing campaign settings")
    systems_total = campaign.get("systems_total")
    if (
        isinstance(systems_total, bool)
        or not isinstance(systems_total, int)
        or systems_total <= 0
    ):
        raise ValueError("campaign systems_total must be a positive integer")

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    x = np.arange(len(route_order), dtype=float)
    rng = np.random.default_rng(20260714)

    for route_index, route in enumerate(route_order):
        row = summary.loc[route]
        successful = runs.loc[(runs["route"] == route) & runs["success"]]
        color = route_colors[route]
        jitter = rng.uniform(-0.055, 0.055, size=len(successful))

        elapsed_minutes = successful["elapsed_s"].to_numpy(dtype=float) / 60.0
        axes[0].scatter(
            route_index + jitter,
            elapsed_minutes,
            color=color,
            alpha=0.35,
            s=28,
            zorder=2,
        )
        median_minutes = float(row["median_elapsed_s"]) / 60.0
        lower_minutes = median_minutes - float(row["elapsed_q25_s"]) / 60.0
        upper_minutes = float(row["elapsed_q75_s"]) / 60.0 - median_minutes
        axes[0].errorbar(
            [route_index],
            [median_minutes],
            yerr=[[lower_minutes], [upper_minutes]],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            capsize=4,
            zorder=3,
        )
        axes[0].annotate(
            f"{float(row['speedup_vs_1gpu']):.2f}×",
            (route_index, median_minutes),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            color=TEXT_COLOR,
            fontsize=9,
        )

        gpu_cost = successful["gpu_seconds_per_structure"].to_numpy(dtype=float)
        axes[1].scatter(
            route_index + jitter,
            gpu_cost,
            color=color,
            alpha=0.35,
            s=28,
            zorder=2,
        )
        median_cost = float(row["median_gpu_seconds_per_structure"])
        axes[1].scatter(
            [route_index],
            [median_cost],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=70,
            zorder=3,
        )

        pass_count = int(row["successful_runs"])
        fail_count = int(row["failed_runs"])
        axes[1].annotate(
            f"{pass_count} pass / {fail_count} fail",
            (route_index, median_cost),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            color=TEXT_COLOR,
            fontsize=8,
        )

    tick_labels = [route_labels[route] for route in route_order]
    for axis in axes:
        axis.set_xticks(x, tick_labels)
        axis.set_xlim(-0.45, len(route_order) - 0.55)
        axis.set_ylim(bottom=0.0)
        style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("wall time / min")
    axes[0].set_title("Time to finish the campaign")
    axes[1].set_ylabel("timed H100-seconds / structure")
    axes[1].set_title("Timed GPU work per structure")
    fig.suptitle(f"{systems_total:,} water hexamers: same workload, three node layouts")
    fig.text(
        0.5,
        0.015,
        "Five saved runs per layout · points are runs · large markers are "
        "medians · wall-time bars show the interquartile range.",
        ha="center",
        color=TEXT_COLOR,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
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
    "EXPERIMENT_COLOR",
    "FIGURE_SIZE",
    "GRID_COLOR",
    "MD_COLOR",
    "SYSTEM_COLORS",
    "SYSTEM_DISPLAY_LABELS",
    "SYSTEM_LINESTYLES",
    "display_system_label",
    "plot_dimer_interaction_energies",
    "plot_distributed_pipeline_scaling",
    "plot_pipeline_campaign",
    "plot_md_dft_comparison",
    "plot_harmonic_monomer_comparison",
    "plot_monomer_ir_comparison",
    "plot_topology_timeline",
    "style_axis",
]
