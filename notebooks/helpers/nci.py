# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked NCI Atlas loading, reduction, validation, and plotting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase import units as ase_units
from nvalchemi.data import Batch

from .data_records import PROJECT_ROOT, load_science_context_record, sha256_file
from .figures import (
    MUTED,
    NVIDIA_BLUE,
    NVIDIA_GREEN,
    NVIDIA_ORANGE,
    QUIET,
    SURFACE,
    TEXT,
    TUTORIAL_FIGURE_WIDTH,
    display_figure,
    polish_axis,
)

if TYPE_CHECKING:
    import pandas as pd

_NCI_DATA_FILE = (
    PROJECT_ROOT
    / "notebooks"
    / "data"
    / "nci_atlas"
    / "nci-atlas-curves.csv.gz"
)
NCI_ATLAS_SHA256 = "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
_EV_TO_KCAL_MOL = 1.0 / (ase_units.kcal / ase_units.mol)
_NCI_SCALES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50, 2.00)
_NCI_SYSTEMS = {
    "1.041": ("phenol - N-methylacetamide", "neutral hydrogen bond"),
    "1.07.74": ("propyne - methyl azide", "dispersion-dominated"),
    "08.007": ("ammonia - benzoate", "ionic hydrogen bond"),
}
_NCI_CURVE_KEYS = (
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
)
_NCI_GRAPH_COLUMNS = (
    *_NCI_CURVE_KEYS,
    "fragment",
    "charge",
    "natoms",
    "source_gradient_block",
    "source_geometry_file",
)
_NCI_REQUIRED_COLUMNS = (
    *_NCI_GRAPH_COLUMNS,
    "symbols",
    "positions_angstrom",
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    "ccsd_t_cbs_interaction_energy_kcal_mol",
)


def _parse_nci_structure(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[str], np.ndarray, int]:
    try:
        atom_count_value = float(record["natoms"])
        symbols = str(record["symbols"]).split()
        positions = np.fromstring(str(record["positions_angstrom"]), sep=" ")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid structure fields") from error
    if not np.isfinite(atom_count_value) or not atom_count_value.is_integer():
        raise ValueError(f"{label} has a non-integer atom count")
    atom_count = int(atom_count_value)
    if atom_count <= 0 or len(symbols) != atom_count:
        raise ValueError(f"{label} atom count does not match its symbols")
    if positions.size != 3 * atom_count or not np.isfinite(positions).all():
        raise ValueError(f"{label} atom count does not match its coordinates")
    return symbols, positions.reshape(atom_count, 3), atom_count


def _validate_nci_atlas(table: pd.DataFrame) -> None:
    import pandas as pd

    missing = set(_NCI_REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"NCI Atlas table is missing {sorted(missing)!r}")
    if len(table) != 90:
        raise ValueError(f"NCI Atlas table has {len(table)} rows; expected 90")

    numeric_columns = (
        "scale",
        "charge",
        "natoms",
        "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
        "ccsd_t_cbs_interaction_energy_kcal_mol",
    )
    for column in numeric_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains a non-finite value")

    observed_systems = {
        str(row.system_id): (str(row.system_name), str(row.interaction_class))
        for row in table[["system_id", "system_name", "interaction_class"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_systems != _NCI_SYSTEMS:
        raise ValueError("NCI Atlas system identities do not match the tutorial subset")
    if table.duplicated(["system_id", "scale", "fragment"]).any():
        raise ValueError("system, scale, and fragment keys must be unique")
    if table["source_gradient_block"].nunique() != 90:
        raise ValueError("source gradient-block identifiers must be unique")

    for system_id, group in table.groupby("system_id", sort=False):
        scales = np.sort(pd.to_numeric(group["scale"]).unique())
        if scales.shape != (10,) or not np.allclose(
            scales,
            _NCI_SCALES,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{system_id} does not contain the expected ten scales")

    grouped = table.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)
    if grouped.ngroups != 30:
        raise ValueError(f"NCI Atlas table has {grouped.ngroups} curves; expected 30")
    for key, group in grouped:
        label = f"{key[1]} at scale {key[-1]:g}"
        if len(group) != 3 or set(group["fragment"]) != {"AB", "A", "B"}:
            raise ValueError(f"{label} must contain one AB, A, and B row")
        records = {str(row.fragment): row._asdict() for row in group.itertuples()}
        structures = {
            fragment: _parse_nci_structure(record, label=f"{label} {fragment}")
            for fragment, record in records.items()
        }
        ab_symbols, ab_positions, ab_count = structures["AB"]
        a_symbols, a_positions, a_count = structures["A"]
        b_symbols, b_positions, b_count = structures["B"]
        if ab_count != a_count + b_count or ab_symbols != a_symbols + b_symbols:
            raise ValueError(f"{label} monomer symbols do not reconstruct the dimer")
        if not np.array_equal(ab_positions, np.vstack((a_positions, b_positions))):
            raise ValueError(
                f"{label} monomer coordinates do not reconstruct the dimer"
            )
        charges = {
            fragment: float(record["charge"]) for fragment, record in records.items()
        }
        if any(not value.is_integer() for value in charges.values()):
            raise ValueError(f"{label} contains a non-integer charge")
        if int(charges["AB"]) != int(charges["A"] + charges["B"]):
            raise ValueError(f"{label} fragment charges do not sum to the dimer charge")
        if group["ccsd_t_cbs_interaction_energy_kcal_mol"].nunique() != 1:
            raise ValueError(f"{label} fragments do not share the CCSD(T)/CBS value")


def _nci_row_to_atoms(record: Mapping[str, Any]) -> Atoms:
    label = str(record.get("source_gradient_block", "NCI Atlas row"))
    symbols, positions, _ = _parse_nci_structure(record, label=label)
    charge = float(record["charge"])
    if not np.isfinite(charge) or not charge.is_integer():
        raise ValueError(f"{label} has a non-integer total charge")
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    for field in (
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "scale",
        "fragment",
        "source_gradient_block",
        "source_geometry_file",
    ):
        value = record[field]
        atoms.info[field] = value.item() if isinstance(value, np.generic) else value
    atoms.info["charge"] = int(charge)
    return atoms


def _nci_reference_curves(table: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    curve_order = table[list(_NCI_CURVE_KEYS)].drop_duplicates(ignore_index=True)
    curve_index = pd.MultiIndex.from_frame(curve_order)
    dft_rows = table[list(_NCI_CURVE_KEYS) + ["fragment"]].copy()
    dft_rows["total"] = pd.to_numeric(
        table["wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"]
    )
    dft_wide = dft_rows.pivot(
        index=list(_NCI_CURVE_KEYS),
        columns="fragment",
        values="total",
    ).reindex(curve_index)
    references = curve_order.copy()
    references["dft_d3_kcal_mol"] = (
        dft_wide["AB"] - dft_wide["A"] - dft_wide["B"]
    ).to_numpy(dtype=float)
    cc = table.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)[
        "ccsd_t_cbs_interaction_energy_kcal_mol"
    ].first()
    references["ccsd_t_cbs_kcal_mol"] = cc.reindex(curve_index).to_numpy(dtype=float)
    return references


def load_nci_atlas(
    path: str | Path | None = None,
    *,
    expected_sha256: str = NCI_ATLAS_SHA256,
) -> tuple[list[Atoms], pd.DataFrame, pd.DataFrame]:
    """Load the checked 90-structure NCI Atlas tutorial subset.

    Returns the ASE structures in file order, row-aligned graph metadata, and
    30 reference interaction energies. The DFT-D3 reference is calculated from
    the stored frozen-geometry AB, A, and B total energies.
    """

    import pandas as pd

    data_path = _NCI_DATA_FILE if path is None else Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"NCI Atlas subset is missing: {data_path}")
    observed = sha256_file(data_path)
    if observed != expected_sha256:
        raise ValueError(
            f"NCI Atlas subset SHA-256 mismatch: {observed}; expected {expected_sha256}"
        )
    table = pd.read_csv(
        data_path,
        dtype={
            "subset": "string",
            "system_id": "string",
            "system_name": "string",
            "interaction_class": "string",
            "fragment": "string",
            "symbols": "string",
            "positions_angstrom": "string",
            "source_gradient_block": "string",
            "source_geometry_file": "string",
        },
    )
    _validate_nci_atlas(table)
    atoms = [_nci_row_to_atoms(row) for row in table.to_dict(orient="records")]
    graph_rows = table[list(_NCI_GRAPH_COLUMNS)].copy().reset_index(drop=True)
    graph_rows.insert(0, "graph_index", np.arange(len(graph_rows), dtype=np.int64))
    return atoms, graph_rows, _nci_reference_curves(table)


def checked_nci_interaction_curves(references: pd.DataFrame) -> pd.DataFrame:
    """Return the checked full NCI curves aligned with current reference rows."""

    import pandas as pd

    record = load_science_context_record("nci_baseline")
    curve_records = record.get("curves")
    if not isinstance(curve_records, list) or not curve_records:
        raise ValueError("nci_baseline curves must be a non-empty list")
    saved = pd.DataFrame(curve_records).rename(
        columns={
            "aimnet_local_kcal_mol": "aimnet_base",
            "local_plus_d3_kcal_mol": "base_d3",
            "complete_kcal_mol": "complete",
        }
    )
    join_keys = ["system_id", "system_name", "scale"]
    required_saved = {*join_keys, "aimnet_base", "base_d3", "complete"}
    missing = required_saved - set(saved.columns)
    if missing:
        raise ValueError(f"nci_baseline curves are missing {sorted(missing)!r}")
    required_references = {*_NCI_CURVE_KEYS}
    missing = required_references - set(references.columns)
    if missing:
        raise ValueError(f"references are missing {sorted(missing)!r}")

    curve_keys = references[list(_NCI_CURVE_KEYS)].copy()
    saved_columns = [
        *join_keys,
        "aimnet_base",
        "base_d3",
        "complete",
    ]
    curves = curve_keys.merge(
        saved[saved_columns],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(curves) != len(references) or len(curves) != len(saved):
        raise ValueError("checked curves and references describe different scan points")
    values = curves[["aimnet_base", "base_d3", "complete"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("checked curves contain a non-finite energy")
    curves["d3_increment"] = curves["base_d3"] - curves["aimnet_base"]
    curves["electrostatic_increment"] = curves["complete"] - curves["base_d3"]
    return curves


def _ordered_nci_graph_rows(graph_rows: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    required = {"graph_index", *_NCI_CURVE_KEYS, "fragment"}
    missing = required - set(graph_rows.columns)
    if missing:
        raise ValueError(f"graph rows are missing {sorted(missing)!r}")
    graph_indices = pd.to_numeric(graph_rows["graph_index"], errors="coerce").to_numpy(
        dtype=float
    )
    expected = np.arange(len(graph_rows), dtype=float)
    if (
        not np.isfinite(graph_indices).all()
        or not np.equal(graph_indices, np.rint(graph_indices)).all()
        or not np.array_equal(np.sort(graph_indices), expected)
    ):
        raise ValueError("graph_index must be a permutation of 0 through N-1")
    ordered = graph_rows.assign(graph_index=graph_indices.astype(np.int64)).sort_values(
        "graph_index"
    )
    grouped = ordered.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)
    for key, group in grouped:
        if len(group) != 3 or set(group["fragment"]) != {"AB", "A", "B"}:
            raise ValueError(f"curve {key!r} must contain one AB, A, and B graph")
    return ordered.reset_index(drop=True)


def _graph_energy_array(values: Any, graph_count: int, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.shape == (graph_count, 1):
        array = array[:, 0]
    if array.shape != (graph_count,):
        raise ValueError(
            f"{name!r} must have shape ({graph_count},) or ({graph_count}, 1)"
        )
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name!r} contains a non-finite graph energy")
    return array


def interaction_components(
    graph_rows: pd.DataFrame,
    graph_energies_ev: Mapping[str, Any],
) -> pd.DataFrame:
    """Reduce named AB, A, and B graph energies to interaction curves."""

    import pandas as pd

    if not graph_energies_ev:
        raise ValueError("at least one energy component is required")
    ordered = _ordered_nci_graph_rows(graph_rows)
    curve_order = ordered[list(_NCI_CURVE_KEYS)].drop_duplicates(ignore_index=True)
    curve_index = pd.MultiIndex.from_frame(curve_order)
    result = curve_order.copy()
    for name, values in graph_energies_ev.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component names must be non-empty strings")
        rows = ordered[list(_NCI_CURVE_KEYS) + ["fragment"]].copy()
        rows["energy"] = _graph_energy_array(values, len(ordered), name)
        wide = rows.pivot(
            index=list(_NCI_CURVE_KEYS),
            columns="fragment",
            values="energy",
        ).reindex(curve_index)
        result[name] = (
            (wide["AB"] - wide["A"] - wide["B"]) * _EV_TO_KCAL_MOL
        ).to_numpy(dtype=float)
    return result


def max_system_charge_error(batch: Batch, charges: torch.Tensor) -> float:
    """Return the largest per-system charge error in a batch."""

    flat_charges = charges.reshape(-1)
    if flat_charges.numel() != batch.num_nodes:
        raise ValueError("charges must contain one value per atom")
    expected = batch.charge.reshape(-1).to(
        device=flat_charges.device,
        dtype=flat_charges.dtype,
    )
    if expected.numel() != batch.num_graphs:
        raise ValueError("batch charge must contain one value per system")
    predicted = torch.zeros_like(expected).index_add(
        0,
        batch.batch_idx.to(device=flat_charges.device, dtype=torch.long),
        flat_charges,
    )
    return float((predicted - expected).abs().max().item())


def nci_curve_deltas(
    live_curves: pd.DataFrame,
    checked_curves: pd.DataFrame,
) -> dict[str, float]:
    """Return the largest absolute live-to-checked difference for each route."""

    keys = ["subset", "system_id", "system_name", "interaction_class", "scale"]
    routes = ("aimnet_base", "base_d3", "complete")
    required = {*keys, *routes}
    for name, table in (
        ("live_curves", live_curves),
        ("checked_curves", checked_curves),
    ):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)!r}")
    comparison = live_curves.merge(
        checked_curves,
        on=keys,
        suffixes=("_live", "_checked"),
        validate="one_to_one",
    )
    if len(comparison) != len(live_curves) or len(comparison) != len(checked_curves):
        raise ValueError("live and checked curves describe different scan points")
    return {
        route: float(
            (comparison[f"{route}_live"] - comparison[f"{route}_checked"]).abs().max()
        )
        for route in routes
    }


def plot_nci_interaction_curves(
    curves: pd.DataFrame,
    references: pd.DataFrame,
    *,
    components: Sequence[str] | None = None,
    show_mae: bool = True,
) -> None:
    """Plot cumulative model routes against the CCSD(T)/CBS reference."""

    import pandas as pd

    reference_columns = {"ccsd_t_cbs_kcal_mol"}
    required_references = {*_NCI_CURVE_KEYS, *reference_columns}
    missing_reference = required_references - set(references.columns)
    if missing_reference:
        raise ValueError(f"references are missing {sorted(missing_reference)!r}")
    component_names = tuple(
        components if components is not None else ("aimnet_base", "base_d3", "complete")
    )
    if not component_names:
        raise ValueError("at least one plotted component is required")
    missing_components = set(component_names) - set(curves.columns)
    if missing_components:
        raise ValueError(f"curves are missing {sorted(missing_components)!r}")
    merged = curves.merge(
        references,
        on=list(_NCI_CURVE_KEYS),
        how="inner",
        validate="one_to_one",
    )
    plotted_columns = [*component_names, *reference_columns]
    plotted = merged[plotted_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(plotted.to_numpy(dtype=float)).all():
        raise ValueError("interaction curves contain a non-finite value")

    labels = {
        "aimnet_base": "AIMNet2",
        "base_d3": "AIMNet2 + D3",
        "aimnet_coulomb": "AIMNet2 + electrostatics",
        "complete": "AIMNet2 + D3 + electrostatics",
    }
    styles = {
        "aimnet_base": (NVIDIA_ORANGE, ":", 1.8),
        "base_d3": (NVIDIA_BLUE, "--", 2.0),
        "aimnet_coulomb": (NVIDIA_ORANGE, "--", 2.0),
        "complete": (NVIDIA_GREEN, "-", 2.8),
    }
    mae = {
        component: float(
            np.mean(
                np.abs(
                    merged[component].to_numpy(dtype=float)
                    - merged["ccsd_t_cbs_kcal_mol"].to_numpy(dtype=float)
                )
            )
        )
        for component in component_names
    }
    system_names = tuple(dict.fromkeys(merged["system_name"].astype(str)))
    figure, axes = plt.subplots(
        1,
        len(system_names),
        figsize=(TUTORIAL_FIGURE_WIDTH, 4.7),
        sharex=True,
        squeeze=False,
    )
    for axis, system_name in zip(axes.reshape(-1), system_names, strict=True):
        group = merged[merged["system_name"].astype(str) == system_name].sort_values(
            "scale"
        )
        scan_coordinate = group["scale"].to_numpy(dtype=float)
        for component in component_names:
            color, linestyle, width = styles.get(
                component,
                (MUTED, "-", 1.8),
            )
            label = labels.get(component, component.replace("_", " "))
            if show_mae:
                label = f"{label} ({mae[component]:.2f})"
            axis.plot(
                scan_coordinate,
                group[component],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                marker="o" if component == "complete" else None,
                markersize=4.0 if component == "complete" else None,
                label=label,
                zorder=4 if component == "complete" else 2,
            )
        axis.scatter(
            scan_coordinate,
            group["ccsd_t_cbs_kcal_mol"],
            s=24,
            facecolor=SURFACE,
            edgecolor=TEXT,
            linewidth=1.0,
            label="CCSD(T)/CBS",
            zorder=5,
        )
        axis.axhline(0.0, color=QUIET, linewidth=0.8, alpha=0.65)
        axis.axvline(1.0, color=QUIET, linewidth=0.8, alpha=0.65)
        axis.set_xticks((0.8, 1.0, 1.25, 1.5, 2.0))
        axis.set(
            title=system_name,
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        polish_axis(axis)
    axes[0, 0].set_ylabel("Interaction energy [kcal/mol]")
    figure.supxlabel("Intermolecular coordinate", y=0.04)
    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(6, len(handles)),
        frameon=False,
        title="MAE vs CCSD(T)/CBS [kcal/mol]" if show_mae else None,
        fontsize=9.5,
        title_fontsize=10.5,
    )
    display_figure(
        figure,
        "Interaction curves for "
        + ", ".join(
            labels.get(name, name.replace("_", " ")) for name in component_names
        )
        + " across three NCI Atlas dissociation scans, compared with CCSD(T)/CBS references.",
        layout_rect=(0.0, 0.05, 1.0, 0.74),
    )
