# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Notebook widgets for surface-screen result inspection."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from IPython.display import HTML, Markdown, display

from .surface_screen import safe_artifact_label
from .visualization import create_interactive_view, subscript_formula_html

ADSORBATE_ATOM_COUNTS = {"CO": 2, "H2O": 3, "NH3": 4, "CH3OH": 6}


def _find_pair_summary_path(
    surface_screen_paths: Mapping[str, object] | None = None,
) -> Path:
    """Find the current or most recent complete surface-screen winner table."""
    configured: Path | None = None
    if surface_screen_paths:
        if "pair_summary_csv" in surface_screen_paths:
            configured = Path(surface_screen_paths["pair_summary_csv"])
        elif "root" in surface_screen_paths:
            configured = (
                Path(surface_screen_paths["root"]) / "tables" / "pair_summary.csv"
            )

    configured_path = configured if configured is not None else None
    if configured_path is not None:
        relaxed_dir = (
            configured_path.parent.parent / "structures" / "relaxed_adsorption"
        )
        if configured_path.exists() and relaxed_dir.exists():
            return configured_path

    candidates: list[Path] = []
    for root in (Path("outputs/live_runs"), Path("outputs/precomputed")):
        if root.exists():
            candidates.extend(root.glob("**/surface_screen/tables/pair_summary.csv"))

    existing: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path == configured_path or path in seen or not path.exists():
            continue
        seen.add(path)
        relaxed_dir = path.parent.parent / "structures" / "relaxed_adsorption"
        if relaxed_dir.exists():
            existing.append(path)

    if not existing:
        configured_label = configured_path if configured_path is not None else "<none>"
        raise RuntimeError(
            "Cannot load winner table; no complete `pair_summary.csv` + "
            f"`structures/relaxed_adsorption` artifact set was found. Configured path: {configured_label}"
        )
    return max(existing, key=lambda path: path.stat().st_mtime)


def _load_ranked_winners(
    *,
    surface_pair_summary_df: pd.DataFrame | None,
    surface_screen_paths: Mapping[str, object] | None,
) -> tuple[pd.DataFrame, Path]:
    configured_pair_summary_path: Path | None = None
    if surface_screen_paths:
        if "pair_summary_csv" in surface_screen_paths:
            configured_pair_summary_path = Path(
                surface_screen_paths["pair_summary_csv"]
            )
        elif "root" in surface_screen_paths:
            configured_pair_summary_path = (
                Path(surface_screen_paths["root"]) / "tables" / "pair_summary.csv"
            )

    if surface_pair_summary_df is not None and configured_pair_summary_path is not None:
        relaxed_dir = (
            configured_pair_summary_path.parent.parent
            / "structures"
            / "relaxed_adsorption"
        )
        if relaxed_dir.exists():
            pair_summary_path = configured_pair_summary_path
        else:
            pair_summary_path = _find_pair_summary_path(surface_screen_paths)
    else:
        pair_summary_path = _find_pair_summary_path(surface_screen_paths)

    if surface_pair_summary_df is not None:
        summary_df = surface_pair_summary_df.copy()
    else:
        summary_df = pd.read_csv(pair_summary_path)

    ranked = (
        summary_df.sort_values(["adsorbate", "rank_within_adsorbate", "best_E_ads_eV"])
        .groupby("adsorbate", as_index=False)
        .head(1)
        .sort_values("adsorbate")
        .reset_index(drop=True)
    )
    return ranked, pair_summary_path


def _load_result_details(artifact_root: Path) -> dict[str, dict[str, object]]:
    results_path = artifact_root / "tables" / "adsorption_results.csv"
    if not results_path.exists():
        return {}
    details_df = pd.read_csv(results_path)
    if "label" not in details_df.columns:
        return {}
    return details_df.set_index("label").to_dict("index")


def _coherent_adsorbate_positions_in_cell(
    original_atoms, wrapped_atoms, adsorbate: str
):
    """Keep the adsorbate intact after wrapping the slab into the displayed cell."""
    ads_count = ADSORBATE_ATOM_COUNTS.get(str(adsorbate), 0)
    if (
        ads_count <= 0
        or ads_count >= len(original_atoms)
        or original_atoms.cell.rank < 2
    ):
        return wrapped_atoms.positions

    ads_start = len(original_atoms) - ads_count
    original_scaled = original_atoms.get_scaled_positions(wrap=False)
    wrapped_scaled = wrapped_atoms.get_scaled_positions(wrap=False)
    ads_scaled = original_scaled[ads_start:]
    target_center = wrapped_scaled[ads_start:].mean(axis=0)

    shift_ranges = []
    for axis in range(3):
        if bool(original_atoms.pbc[axis]) and axis < 2:
            shift_ranges.append(range(-4, 5))
        else:
            shift_ranges.append((0,))

    best_shift = np.zeros(3)
    best_score = np.inf
    for shift in product(*shift_ranges):
        shift = np.asarray(shift, dtype=float)
        candidate = ads_scaled + shift
        center = candidate.mean(axis=0)
        outside = np.maximum(0.0, -candidate) + np.maximum(0.0, candidate - 1.0)
        score = float(np.sum((center - target_center) ** 2) + 3.0 * np.sum(outside))
        if score < best_score:
            best_score = score
            best_shift = shift

    positions = wrapped_atoms.positions.copy()
    positions[ads_start:] = (
        original_atoms.positions[ads_start:] + best_shift @ original_atoms.cell.array
    )
    return positions


def _display_cell_aligned_atoms(atoms, adsorbate: str):
    """Build a display-only periodic image with the slab inside the drawn cell."""
    if atoms.cell.rank < 2 or len(atoms) == 0:
        return atoms.copy()
    wrapped = atoms.copy()
    wrapped.wrap(eps=1e-7)
    wrapped.positions = _coherent_adsorbate_positions_in_cell(atoms, wrapped, adsorbate)
    return wrapped


def _format_relaxation_status(details: Mapping[str, object], steps: object) -> str:
    converged_value = details.get("converged", None)
    if converged_value is None or pd.isna(converged_value):
        parts = ["optimizer status unavailable"]
    else:
        converged = str(converged_value).strip().lower() in {"true", "1"}
        parts = ["optimizer-converged" if converged else "not optimizer-converged"]

    max_force = details.get("max_force_eV_A", np.nan)
    if pd.notna(max_force):
        parts.append(f"max force {float(max_force):.3f} eV/A")
    parts.append(f"{int(steps)} steps")
    return "; ".join(parts)


def _ranked_winner_items(
    winners_df: pd.DataFrame, pair_summary_path: Path
) -> tuple[list[dict], Path]:
    """Load winner structures from saved `.extxyz` files."""
    artifact_root = pair_summary_path.parent.parent
    relaxed_dir = artifact_root / "structures" / "relaxed_adsorption"
    if not relaxed_dir.exists():
        raise RuntimeError(
            f"Cannot load final structures; missing directory: {relaxed_dir}"
        )

    result_details = _load_result_details(artifact_root)
    items: list[dict] = []
    missing: list[Path] = []
    for _, row in winners_df.iterrows():
        host = row["host"]
        adsorbate = row["adsorbate"]
        label = row["best_label"]
        structure_path = relaxed_dir / f"{safe_artifact_label(label)}.extxyz"
        if not structure_path.exists():
            missing.append(structure_path)
            continue

        final_atoms = ase_read(structure_path)
        details = result_details.get(label, {})
        relaxation_status = _format_relaxation_status(
            details, row["best_optimizer_nsteps"]
        )
        title = f"{adsorbate}/{host}: {float(row['best_E_ads_eV']):.2f} eV"
        caption = (
            f"start {label}; final site {row['best_final_site']}; {relaxation_status}"
        )
        items.append(
            {
                "title": title,
                "caption": caption,
                "adsorbate": adsorbate,
                "host": host,
                "label": label,
                "atoms": final_atoms,
                "display_atoms": _display_cell_aligned_atoms(final_atoms, adsorbate),
                "relaxation_status": relaxation_status,
                "structure_path": structure_path.as_posix(),
            }
        )

    if missing:
        raise RuntimeError(
            "Cannot load all winning final structures; first missing file: "
            f"{missing[0]}"
        )
    return items, artifact_root


def _chunked(items: list, columns: int) -> list[list]:
    return [items[start : start + columns] for start in range(0, len(items), columns)]


def _display_widget_grid(
    items: list[dict],
    *,
    columns: int,
    width: str,
    height: str,
    header_height: str,
    show_cell: bool,
    wrap_periodic_cell: bool,
) -> None:
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "ipywidgets is required for the OVITO widget display."
        ) from exc

    rows = []
    for row_items in _chunked(items, columns):
        cards = []
        for item in row_items:
            label_html = (
                "<div style='font-weight:700; font-size:16px; margin-bottom:3px;'>"
                f"{subscript_formula_html(item['title'])}</div>"
                "<div style='font-size:12px; line-height:1.25; color:#5b6472;'>"
                f"{subscript_formula_html(item['caption'])}</div>"
            )
            header = widgets.HTML(
                label_html,
                layout=widgets.Layout(
                    height=header_height,
                    min_height=header_height,
                    overflow="hidden",
                ),
            )
            widget = create_interactive_view(
                item["display_atoms"],
                width=width,
                height=height,
                show_cell=show_cell,
                wrap_periodic_cell=wrap_periodic_cell,
            )
            if widget is None:
                widget = widgets.HTML(
                    "<div style='font-size:13px; color:#8a1f11;'>"
                    "OVITO's notebook widget is unavailable in this runtime.</div>"
                    f"<code>{item['structure_path']}</code>"
                )
            cards.append(
                widgets.VBox(
                    [header, widget],
                    layout=widgets.Layout(
                        width=width,
                        min_width=width,
                        align_items="stretch",
                    ),
                )
            )
        rows.append(
            widgets.HBox(
                cards,
                layout=widgets.Layout(
                    align_items="flex-start",
                    gap="18px",
                    overflow="visible",
                ),
            )
        )

    display(
        widgets.VBox(
            rows,
            layout=widgets.Layout(
                align_items="flex-start",
                gap="18px",
                overflow="visible",
            ),
        )
    )


def _display_winner_table(items: list[dict]) -> None:
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{subscript_formula_html(item['adsorbate'])}</td>"
            f"<td>{subscript_formula_html(item['host'])}</td>"
            f"<td>{subscript_formula_html(item['label'])}</td>"
            f"<td>{subscript_formula_html(item['relaxation_status'])}</td>"
            f"<td>{len(item['atoms'])}</td>"
            "</tr>"
        )
    display(
        HTML(
            "<table>"
            "<thead><tr><th>adsorbate</th><th>surface</th><th>winning start</th>"
            "<th>relaxation</th><th>atoms</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )
    )


def display_surface_screen_winner_widgets(
    *,
    surface_pair_summary_df: pd.DataFrame | None = None,
    surface_screen_paths: Mapping[str, object] | None = None,
    columns: int = 2,
    width: str = "460px",
    height: str = "360px",
    header_height: str = "92px",
    show_cell: bool = True,
    wrap_periodic_cell: bool = False,
    show_intro: bool = True,
    show_table: bool = True,
) -> dict[str, object]:
    """Display the best final structure per adsorbate as OVITO widgets.

    The structures are loaded from saved surface-screen artifacts rather than
    from transient notebook variables. A display-only periodic image is used so
    the slab sits inside the drawn cell while the adsorbate remains coherent.
    """
    winners_df, pair_summary_path = _load_ranked_winners(
        surface_pair_summary_df=surface_pair_summary_df,
        surface_screen_paths=surface_screen_paths,
    )
    items, artifact_root = _ranked_winner_items(winners_df, pair_summary_path)

    if show_intro:
        display(
            Markdown(
                "The heatmap ranks adsorption energies; these OVITO widgets show the "
                "optimizer-converged structures selected for each adsorbate. They are "
                "loaded from saved surface-screen artifacts. The displayed periodic image "
                "is adjusted only for visualization so the slab sits inside the drawn cell "
                "while the adsorbate molecule remains intact."
            )
        )
        if pair_summary_path.exists():
            display(Markdown(f"Loaded winner metadata from `{pair_summary_path}`."))
        else:
            display(
                Markdown(
                    "Using the live ranking table from the notebook and loading final "
                    f"structures from `{pair_summary_path.parent.parent / 'structures' / 'relaxed_adsorption'}`."
                )
            )

    if show_table:
        _display_winner_table(items)

    _display_widget_grid(
        items,
        columns=columns,
        width=width,
        height=height,
        header_height=header_height,
        show_cell=show_cell,
        wrap_periodic_cell=wrap_periodic_cell,
    )

    return {
        "ranked_winners_df": winners_df,
        "items": items,
        "pair_summary_path": pair_summary_path,
        "artifact_root": artifact_root,
    }
