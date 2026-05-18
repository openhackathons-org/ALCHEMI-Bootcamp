"""Plotting helpers for the adsorption-search tutorial.

These functions keep Matplotlib formatting out of the notebook so the notebook
can focus on Toolkit calls, data construction, and analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
from typing import Any

import ase
import numpy as np
import pandas as pd

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _allow_artifact_overwrite() -> bool:
    return os.environ.get("ALCHEMI_ALLOW_ARTIFACT_OVERWRITE", "").strip().lower() in _TRUE_VALUES


def _save_figure(fig, output_path: str, **kwargs) -> str:
    path = Path(output_path)
    if "outputs/precomputed" in path.as_posix() and not _allow_artifact_overwrite():
        if path.exists():
            return str(path)
        raise FileExistsError(
            f"Refusing to create official saved figure without refresh enabled: {path}. "
            "Use a live-run output path or set REFRESH_SAVED_RESULTS = True."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    return str(path)


def plot_h2o_batch_speedup(
    speedup_df: pd.DataFrame,
    examples: list[ase.Atoms],
    output_path: str,
) -> str:
    """Plot the measured Toolkit batch-speedup curve."""
    import matplotlib.pyplot as plt

    nv_green = "#76B900"
    nv_blue = "#00A3E0"
    dark = "#000000"
    light = "#F3F5F7"
    muted = "#A8B0B8"

    _ = examples  # Kept for notebook compatibility; the figure now focuses on throughput.
    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor=dark)
    ax.set_facecolor(dark)
    df = speedup_df.sort_values("batch_size")
    ax.plot(
        df["batch_size"],
        df["speedup_vs_single"],
        color=nv_green,
        marker="o",
        linewidth=3.0,
        markersize=7,
        label="measured",
    )
    ax.plot(
        df["batch_size"],
        df["batch_size"],
        color=muted,
        linestyle="--",
        linewidth=1.5,
        label="ideal linear",
    )
    ax.fill_between(df["batch_size"], df["speedup_vs_single"], color=nv_green, alpha=0.16)
    ax.set_xscale("log", base=2)
    ticks = [int(size) for size in df["batch_size"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(size) for size in ticks])
    ax.set_xlabel("H$_2$O structures in one Toolkit batch", color=light, fontsize=12)
    ax.set_ylabel("speedup vs one-at-a-time", color=light, fontsize=12)
    ax.set_title("Batching amortizes fixed work", color=light, pad=12, fontsize=13)
    ax.tick_params(colors=light, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
    ax.grid(True, color="#2F3A44", linewidth=0.9, alpha=0.75)
    ax.legend(facecolor=dark, edgecolor="#4B5563", labelcolor=light, loc="upper left")

    last = df.iloc[-1]
    ax.scatter(
        [last["batch_size"]],
        [last["speedup_vs_single"]],
        s=150,
        facecolors="none",
        edgecolors=nv_blue,
        linewidths=1.7,
        zorder=4,
    )
    ax.text(
        last["batch_size"],
        last["speedup_vs_single"],
        f"  {last['speedup_vs_single']:.1f}x",
        color=nv_blue,
        fontsize=12,
        va="center",
    )

    fig.suptitle("ALCHEMI Toolkit batched H$_2$O relaxation", color=light, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = _save_figure(
        fig,
        output_path,
        dpi=320,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def plot_adsorption_batch_calibration(
    calibration_df: pd.DataFrame,
    output_path: str,
    *,
    title: str = "Adsorption batch-size calibration",
) -> str:
    """Plot short-relaxation throughput, saturation, and memory use."""
    import matplotlib.pyplot as plt

    nv_green = "#76B900"
    nv_blue = "#00A3E0"
    dark = "#000000"
    light = "#F3F5F7"
    muted = "#A8B0B8"
    grid = "#2F3A44"
    colors = [nv_green, nv_blue, "#F5B642", "#D7E3F4"]

    df = calibration_df.copy()
    if "status" not in df:
        df["status"] = "ok"
    if "gpu_free_drop_gb" not in df and {"gpu_free_before_gb", "gpu_free_after_gb"}.issubset(df.columns):
        df["gpu_free_drop_gb"] = (df["gpu_free_before_gb"] - df["gpu_free_after_gb"]).clip(lower=0.0)
    ok = df[df["status"].eq("ok")].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.9),
        facecolor=dark,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for ax in axes:
        ax.set_facecolor(dark)
        ax.tick_params(colors=light, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#4B5563")
        ax.grid(True, color=grid, linewidth=0.9, alpha=0.75)

    recommendation_rows = []
    for idx, (label, model_df) in enumerate(ok.groupby("label", sort=False)):
        model_df = model_df.sort_values("batch_size")
        color = colors[idx % len(colors)]
        best = model_df.loc[model_df["structures_per_s"].idxmax()]
        best_rate = float(best["structures_per_s"])
        near_best = model_df[model_df["structures_per_s"] >= 0.90 * best_rate]
        recommended = near_best.sort_values("batch_size").iloc[0]
        recommendation_rows.append((label, color, recommended, best))

        axes[0].plot(
            model_df["batch_size"],
            model_df["structures_per_s"],
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=6,
            label=label,
        )
        axes[0].scatter(
            [recommended["batch_size"]],
            [recommended["structures_per_s"]],
            s=95,
            facecolors=dark,
            edgecolors=color,
            linewidths=2.0,
            zorder=4,
        )
        axes[0].annotate(
            f"batch {int(recommended['batch_size'])}",
            xy=(recommended["batch_size"], recommended["structures_per_s"]),
            xytext=(6, 8 if idx == 0 else -18),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

        if "process_peak_reserved_gb" in model_df:
            axes[1].plot(
                model_df["batch_size"],
                model_df["process_peak_reserved_gb"],
                color=color,
                marker="o",
                linewidth=2.8,
                markersize=6,
                alpha=0.98,
                label=label,
            )
            last = model_df.iloc[-1]
            axes[1].annotate(
                f"{float(last['process_peak_reserved_gb']):.1f} GB",
                xy=(last["batch_size"], last["process_peak_reserved_gb"]),
                xytext=(7, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )

    oom = df[df["status"].eq("oom")]
    if len(oom):
        for _, row in oom.iterrows():
            axes[1].scatter(
                [row["batch_size"]],
                [row.get("gpu_free_before_gb", np.nan)],
                marker="x",
                color="#FF6B6B",
                s=70,
                linewidths=2,
                label="OOM" if "OOM" not in axes[1].get_legend_handles_labels()[1] else None,
            )

    if len(df):
        ticks = sorted(int(size) for size in df["batch_size"].dropna().unique())
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(size) for size in ticks])

    axes[0].set_xlabel("structures per Toolkit batch", color=light, fontsize=11)
    axes[0].set_ylabel("structures/s", color=light, fontsize=11)
    axes[0].set_title("Throughput saturation", color=light, fontsize=13, pad=10)
    axes[0].legend(
        facecolor=dark,
        edgecolor="#4B5563",
        labelcolor=light,
        fontsize=8,
        loc="lower right",
    )

    axes[1].set_xlabel("structures per Toolkit batch", color=light, fontsize=11)
    axes[1].set_ylabel("peak reserved GPU memory (GB)", color=light, fontsize=11)
    axes[1].set_title("Measured VRAM footprint", color=light, fontsize=13, pad=10)
    if "process_peak_reserved_gb" in ok and ok["process_peak_reserved_gb"].notna().any():
        ymax = float(ok["process_peak_reserved_gb"].max())
        axes[1].set_ylim(0, max(1.0, ymax * 1.22))
    axes[1].legend(
        facecolor=dark,
        edgecolor="#4B5563",
        labelcolor=light,
        fontsize=8.5,
        loc="upper left",
    )

    summary_lines = [
        f"{label}: recommended batch {int(recommended['batch_size'])}; "
        f"best measured {int(best['batch_size'])}"
        for label, _color, recommended, best in recommendation_rows
    ]
    if summary_lines:
        fig.text(
            0.07,
            0.02,
            " | ".join(summary_lines),
            color=muted,
            fontsize=9,
        )

    fig.suptitle(title, color=light, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    output_path = _save_figure(
        fig,
        output_path,
        dpi=320,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def plot_surface_screen_heatmap(
    heatmap_df: pd.DataFrame,
    output_path: str,
    *,
    title: str = "Best adsorption energy found in six-start searches",
) -> str:
    """Plot the 9-facet surface-screen result as an adsorption heatmap."""
    import matplotlib.pyplot as plt

    nv_green = "#76B900"
    dark = "#000000"
    light = "#F3F5F7"
    grid = "#2F3A44"

    df = heatmap_df.copy()
    if df.empty:
        raise ValueError("Cannot plot an empty surface-screen heatmap.")
    pivot = df.pivot(index="adsorbate", columns="host", values="best_E_ads_eV")
    ranks = df.pivot(index="adsorbate", columns="host", values="rank_within_adsorbate")

    fig, ax = plt.subplots(figsize=(12.5, 4.8), facecolor=dark)
    ax.set_facecolor(dark)
    values = pivot.to_numpy(dtype=float)
    vmax = max(0.0, float(np.nanmax(values)))
    vmin = float(np.nanmin(values))
    image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", color=light)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [label.replace("H2O", "H$_2$O").replace("NH3", "NH$_3$").replace("CH3OH", "CH$_3$OH") for label in pivot.index],
        color=light,
    )
    ax.tick_params(colors=light)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")

    for i, adsorbate in enumerate(pivot.index):
        for j, host in enumerate(pivot.columns):
            value = pivot.loc[adsorbate, host]
            if pd.isna(value):
                continue
            rank = int(ranks.loc[adsorbate, host])
            text_color = "#FFFFFF" if value < (vmin + vmax) / 2 else "#111827"
            ax.text(
                j,
                i,
                f"{value:.2f}\n#{rank}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if rank == 1 else "normal",
            )

    ax.set_title(title, color=light, fontsize=15, pad=14)
    ax.set_xlabel("surface", color=light, fontsize=11)
    ax.set_ylabel("adsorbate", color=light, fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color=grid, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("E$_{ads}$ (eV)", color=light)
    cbar.ax.yaxis.set_tick_params(color=light)
    plt.setp(cbar.ax.get_yticklabels(), color=light)
    cbar.outline.set_edgecolor("#4B5563")

    fig.text(
        0.012,
        0.02,
        "Negative values are exothermic. Rank is within each adsorbate.",
        color=nv_green,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path = _save_figure(
        fig,
        output_path,
        dpi=320,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def plot_adsorption_energy_spread(
    pair_results: Mapping[tuple[str, str], pd.DataFrame],
    reference_lookup: Callable[[str, str], Any],
    output_path: str,
) -> str:
    """Plot the E_ads distribution for each adsorbate/surface pair."""
    import matplotlib.pyplot as plt

    n_pairs = len(pair_results)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(9, max(2.5 * n_pairs, 3)), sharex=False)
    if n_pairs == 1:
        axes = [axes]
    site_markers = {
        "top": "o",
        "bridge": "s",
        "fcc": "^",
        "hcp": "v",
        "al-top": "o",
        "o-top": "D",
        "hollow": "P",
    }

    for ax, ((host, adsorbate), df) in zip(axes, pair_results.items()):
        for site in df["final_site"].unique():
            sub = df[df["final_site"] == site]
            ax.scatter(
                sub["E_ads (eV)"],
                [0] * len(sub),
                marker=site_markers.get(site, "o"),
                s=80,
                alpha=0.7,
                label=f"final {site}",
                edgecolor="black",
                linewidth=0.5,
            )
        emin = df["E_ads (eV)"].min()
        ax.axvline(emin, color="black", ls="--", lw=1, label=f"min = {emin:.3f} eV")
        ref = reference_lookup(host, adsorbate)
        if ref is not None and ref.e_ads_ev is not None:
            if ref.strict_for_parity:
                ax.axvline(
                    ref.e_ads_ev,
                    color="red",
                    ls=":",
                    lw=1.5,
                    label=f"matched reference {ref.e_ads_ev:.2f} eV ({ref.binding_site})",
                )
            else:
                ax.axvline(
                    ref.e_ads_ev,
                    color="#777777",
                    ls=":",
                    lw=1.0,
                    label=f"literature/context value {ref.e_ads_ev:.2f} eV ({ref.binding_site})",
                )
        ax.set_yticks([])
        ax.set_xlabel("E_ads (eV)")
        ax.set_title(
            f"{adsorbate} on {host} - {len(df)} starts - "
            f"spread = {df['E_ads (eV)'].max() - emin:.3f} eV",
            fontsize=10,
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        ax.grid(True, axis="x", ls="--", alpha=0.4)

    fig.tight_layout()
    output_path = _save_figure(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_site_agreement_heatmap(
    summary_df: pd.DataFrame,
    host_names: list[str],
    adsorbates: list[str],
    output_path: str,
) -> str:
    """Plot final-site agreement with reference rows."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    site_df = summary_df.copy()
    site_df["site_score"] = site_df["site_match"].map({True: 1, False: -1}).fillna(0)
    heat = site_df.pivot(index="host", columns="adsorbate", values="site_score")
    heat = heat.reindex(host_names).reindex(columns=adsorbates)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cmap = ListedColormap(["#d62728", "#bdbdbd", "#2ca02c"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(heat.values.astype(float), cmap=cmap, norm=norm)
    ax.set_xticks(range(len(adsorbates)))
    ax.set_xticklabels(adsorbates)
    ax.set_yticks(range(len(host_names)))
    ax.set_yticklabels(host_names)
    for i, host in enumerate(host_names):
        for j, adsorbate in enumerate(adsorbates):
            row = site_df[(site_df["host"] == host) & (site_df["adsorbate"] == adsorbate)]
            if len(row):
                value = row.iloc[0]["site_match"]
                if pd.isna(value):
                    text = "n/a"
                else:
                    text = "match" if bool(value) else "diff"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Final-site agreement with reference")
    fig.tight_layout()
    output_path = _save_figure(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_site_bias(bias_df: pd.DataFrame, output_path: str) -> str:
    """Plot nominated single-start E_ads against the batched minimum."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(2 + 0.4 * len(bias_df), 3)))
    if len(bias_df):
        single_col = (
            "nominated single-start E_ads (eV)"
            if "nominated single-start E_ads (eV)" in bias_df.columns
            else "legacy single-start E_ads (eV)"
        )
        batch_col = (
            "batch minimum E_ads (eV)"
            if "batch minimum E_ads (eV)" in bias_df.columns
            else "batch minimum (eV)"
        )
        y = range(len(bias_df))
        ax.barh(
            [yi - 0.18 for yi in y],
            bias_df[single_col],
            0.35,
            color="#d62728",
            label="nominated single start",
        )
        ax.barh(
            [yi + 0.18 for yi in y],
            bias_df[batch_col],
            0.35,
            color="#1f77b4",
            label="batch minimum",
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(bias_df["pair"])
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("E_ads (eV) - more negative = stronger binding")
        ax.set_title("Starting-site effect: single start vs batched search")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x", ls="--", alpha=0.4)
        for i, row in bias_df.reset_index().iterrows():
            ax.text(
                max(row[single_col], row[batch_col]) + 0.02,
                i,
                f"Delta = {row['starting-site effect (meV)']:+d} meV",
                va="center",
                fontsize=9,
            )

    fig.tight_layout()
    output_path = _save_figure(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


plot_single_start_bias = plot_top_site_bias


def plot_adsorption_summary(
    summary_df: pd.DataFrame,
    mad_guide_ev: float,
    output_path: str,
) -> str:
    """Plot final tutorial examples with reference/context values."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    result_type_color = {
        "validation": "#2ca02c",
        "discovery": "#1f77b4",
        "discrepancy": "#d62728",
        "?": "#888888",
    }
    result_type_label = {
        "validation": "reference check",
        "discovery": "search effect",
        "discrepancy": "needs review",
        "?": "not assigned",
    }
    disc_df = summary_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.5 * len(disc_df) + 2)))
    y = list(range(len(disc_df)))
    for i, row in disc_df.iterrows():
        color = result_type_color.get(row["tier"], "#888")
        ax.plot(
            row["E_MACE_eV"],
            i,
            "^",
            color=color,
            markersize=13,
            zorder=3,
            label=(
                f"MACE ({result_type_label.get(row['tier'], row['tier'])})"
                if i == 0
                else None
            ),
        )
        is_matched_reference = row["reference_scope"] in {"strict", "near-strict"}
        if pd.notna(row["E_ref_eV"]):
            ref_color = color if is_matched_reference else "#777777"
            ax.plot(
                row["E_ref_eV"],
                i,
                "D",
                color=ref_color,
                markersize=9,
                zorder=3,
                markerfacecolor="white",
            )
            if is_matched_reference:
                ax.plot(
                    [row["E_ref_eV"] - mad_guide_ev, row["E_ref_eV"] + mad_guide_ev],
                    [i, i],
                    "-",
                    color=color,
                    lw=2,
                    alpha=0.35,
                    zorder=1,
                )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['pair']} [{row['status']}]" for _, row in disc_df.iterrows()])
    ax.set_xlabel("E_ads (eV) - more negative = stronger binding")
    ax.set_title(
        "Adsorption examples: MACE-MPA-0 result, reference/context value, "
        "and MAD guide for matched references"
    )
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    legend_handles = [
        mpatches.Patch(color=color, label=result_type_label.get(tier, tier))
        for tier, color in result_type_color.items()
        if tier != "?"
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path = _save_figure(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
