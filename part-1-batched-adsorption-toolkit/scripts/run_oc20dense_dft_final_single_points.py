#!/usr/bin/env python3
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
"""Run MACE single points on exact OC20Dense DFT-relaxed final geometries.

This complements ``run_oc20dense_known_examples.py``:

* initial-geometry SP checks MACE energies on released OC20Dense starting frames;
* Toolkit relaxation checks the MACE relaxation path and final ranking;
* this script checks MACE energies and forces on the official DFT-relaxed final frames.

The energies in this layer are MACE total energies. They are valid for ranking
configurations within the same OC20Dense system. A separate script adds the
explicit MACE clean-surface and gas-reference subtraction used for adsorption
energies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PART1 / "scripts"
sys.path.insert(0, str(PART1))
sys.path.insert(0, str(SCRIPT_DIR))

from helpers import ase_to_atomic_data  # noqa: E402
from oc20dense_dft_reference_checks import (  # noqa: E402
    DEFAULT_OUTDIR as DEFAULT_DFT_CHECK_DIR,
    _read_atoms_sequence,
)
from _oc20dense_common import (  # noqa: E402
    DEFAULT_SYSTEMS,
    MACE_EADS_REFERENCE_STATUS,
    MACE_RANK_BASIS,
    TOOLKIT_PROVENANCE_COLUMNS,
    require_precomputed_write_allowed,
    toolkit_cache_matches,
    toolkit_model_label,
    toolkit_provenance_from_env,
    toolkit_provenance_mismatch,
)
from run_oc20dense_known_examples import (  # noqa: E402
    DEFAULT_OUTDIR as DEFAULT_TOOLKIT_ROOT,
    _build_backend,
    _cuda_memory_snapshot,
    _max_force,
    _memory_fields,
    _memory_message,
    _model_tensor,
    _reset_cuda_peak_memory,
)


DEFAULT_OUTDIR = DEFAULT_TOOLKIT_ROOT / "dft_final_single_points"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit-root", type=Path, default=DEFAULT_TOOLKIT_ROOT)
    parser.add_argument("--dft-check-dir", type=Path, default=DEFAULT_DFT_CHECK_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="OC20Dense system_id values to score.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("OC20DENSE_SP_CHUNK_SIZE", "24")),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(os.environ.get("OC20DENSE_TOOLKIT_N_STEPS", "200")),
        help="Backend compatibility option; not used for single-point scoring.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=float(os.environ.get("TOOLKIT_FMAX", "0.05")),
        help="Backend compatibility option; not used for single-point scoring.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached chunks.")
    return parser.parse_args()


def _safe(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("*", "star")
    )


def _ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "chunks": outdir / "chunks",
        "structures": outdir / "structures" / "dft_final",
        "sp_structures": outdir / "structures" / "dft_final_mace_sp",
        "logs": outdir / "single_point_logs",
        "tables": outdir / "tables",
        "reports": outdir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _with_initial_tags(dft_final: Atoms, initial_path: Path) -> tuple[Atoms, list[bool]]:
    initial = ase_read(initial_path)
    tags = np.asarray(initial.get_tags(), dtype=int)
    atoms = dft_final.copy()
    atoms.set_tags(tags)
    atoms.set_constraint(FixAtoms(mask=(tags == 0).tolist()))
    active_mask = [bool(tag != 0) for tag in tags]
    return atoms, active_mask


def _load_items(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, Any]]:
    comparison_path = args.dft_check_dir / "dft_reference_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing DFT trajectory comparison table: {comparison_path}. "
            "Run oc20dense_dft_reference_checks.py --mode compare --scope all first."
        )

    comparison = pd.read_csv(comparison_path)
    requested = {str(system_id) for system_id in args.systems}
    comparison = comparison[comparison["system_id"].astype(str).isin(requested)]
    if comparison.empty:
        raise RuntimeError("No rows matched the requested systems.")

    items: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        label = f"{row.system_id}_{row.config_id}_sid{int(row.sid)}"
        frames = _read_atoms_sequence(Path(row.dft_trajectory_path))
        dft_final, active_mask = _with_initial_tags(
            frames[-1],
            Path(row.initial_structure_path),
        )
        dft_final.info.update(
            {
                "label": label,
                "system_id": str(row.system_id),
                "config_id": str(row.config_id),
                "sid": int(row.sid),
                "adsorbate": str(row.adsorbate),
                "adsorbate_reference_species": str(row.adsorbate_reference_species),
                "dft_rank": int(row.dft_rank),
                "dft_adsorption_energy_eV": float(
                    row.dft_adsorption_energy_target_eV
                ),
                "dft_gap_to_best_eV": float(row.dft_gap_to_system_best_eV),
            }
        )
        structure_path = paths["structures"] / f"{_safe(label)}.extxyz"
        if args.force or not structure_path.exists():
            ase_write(structure_path, dft_final, format="extxyz")
        items.append(
            {
                "label": label,
                "system_id": str(row.system_id),
                "config_id": str(row.config_id),
                "sid": int(row.sid),
                "adsorbate": str(row.adsorbate),
                "adsorbate_reference_species": str(row.adsorbate_reference_species),
                "dft_rank": int(row.dft_rank),
                "dft_adsorption_energy_eV": float(
                    row.dft_adsorption_energy_target_eV
                ),
                "dft_gap_to_best_eV": float(row.dft_gap_to_system_best_eV),
                "dft_trajectory_path": str(row.dft_trajectory_path),
                "dft_final_structure_path": str(structure_path),
                "atoms": dft_final,
                "active_mask": active_mask,
            }
        )
    return sorted(items, key=lambda item: (item["system_id"], item["dft_rank"]))


def _single_point_chunk(
    *,
    backend: Any,
    chunk: list[dict[str, Any]],
    chunk_label: str,
    paths: dict[str, Path],
    force: bool,
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    chunk_json = paths["chunks"] / f"{chunk_label}.json"
    metadata_json = paths["chunks"] / f"{chunk_label}.metadata.json"
    if chunk_json.exists() and not force:
        metadata = (
            json.loads(metadata_json.read_text(encoding="utf-8"))
            if metadata_json.exists()
            else {}
        )
        if toolkit_cache_matches(metadata, provenance):
            return json.loads(chunk_json.read_text(encoding="utf-8")), float(
                metadata.get("runtime_s", 0.0)
            ), metadata
        mismatch = toolkit_provenance_mismatch(metadata, provenance)
        print(
            f"{chunk_label}: cache provenance mismatch, recomputing: {mismatch}",
            flush=True,
        )

    from nvalchemi.neighbors import compute_neighbors

    payloads = [
        ase_to_atomic_data(
            item["atoms"],
            structure_id=f"dft_final_sid_{item['sid']}_{item['config_id']}",
            active_mask=item["active_mask"],
        )
        for item in chunk
    ]
    data_list = [backend._to_atomic_data(payload) for payload in payloads]
    batch = backend.api.Batch.from_data_list(data_list, device=backend.device)
    n_atoms = sum(len(item["atoms"]) for item in chunk)
    _reset_cuda_peak_memory(backend.device)
    before_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"{chunk_label}: single-point batch {len(chunk)} configs, {n_atoms} atoms; "
        f"{_memory_message(before_memory)}",
        flush=True,
    )

    start = time.perf_counter()
    compute_neighbors(batch, config=backend.model.model_config.neighbor_config)
    outputs = backend.model(batch)
    runtime_s = time.perf_counter() - start
    after_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"{chunk_label}: single-point done {runtime_s:.2f} s; "
        f"{_memory_message(after_memory)}",
        flush=True,
    )

    energies = _model_tensor(outputs, "energy").detach().cpu().numpy().reshape(-1)
    forces = _model_tensor(outputs, "forces").detach().cpu().numpy().reshape(-1, 3)
    batch_ptr = getattr(batch, "batch_ptr", None)
    if batch_ptr is None:
        offsets = np.cumsum([0, *[len(item["atoms"]) for item in chunk]])
    else:
        offsets = batch_ptr.detach().cpu().numpy().astype(int)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(chunk):
        force_block = forces[offsets[index]: offsets[index + 1]]
        sp_structure_path = paths["sp_structures"] / f"{_safe(item['label'])}.extxyz"
        sp_log_path = paths["logs"] / f"{_safe(item['label'])}.csv"
        sp_atoms = item["atoms"].copy()
        sp_atoms.info["structure_id"] = item["label"]
        sp_atoms.info["mace_total_energy_eV"] = float(energies[index])
        sp_atoms.info["mace_free_fmax_eV_A"] = _max_force(
            force_block.flatten().tolist(),
            active_mask=item["active_mask"],
        )
        sp_atoms.info["mace_all_atom_fmax_eV_A"] = _max_force(
            force_block.flatten().tolist()
        )
        sp_atoms.arrays["forces"] = force_block
        ase_write(sp_structure_path, sp_atoms, format="extxyz")
        sp_log_path.write_text(
            "\n".join(
                [
                    "step,structure_id,energy_eV,max_force_eV_A,free_max_force_eV_A",
                    (
                        f"0,{item['label']},{float(energies[index])},"
                        f"{sp_atoms.info['mace_all_atom_fmax_eV_A']},"
                        f"{sp_atoms.info['mace_free_fmax_eV_A']}"
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "system_id": item["system_id"],
                "config_id": item["config_id"],
                "sid": item["sid"],
                "adsorbate": item["adsorbate"],
                "adsorbate_reference_species": item["adsorbate_reference_species"],
                "dft_rank": item["dft_rank"],
                "dft_adsorption_energy_eV": item["dft_adsorption_energy_eV"],
                "dft_gap_to_best_eV": item["dft_gap_to_best_eV"],
                "mace_dft_final_sp_total_energy_eV": float(energies[index]),
                "mace_dft_final_sp_free_fmax_eV_A": _max_force(
                    force_block.flatten().tolist(),
                    active_mask=item["active_mask"],
                ),
                "mace_dft_final_sp_all_atom_fmax_eV_A": _max_force(
                    force_block.flatten().tolist()
                ),
                "dft_trajectory_path": item["dft_trajectory_path"],
                "dft_final_structure_path": item["dft_final_structure_path"],
                "mace_sp_structure_path": str(sp_structure_path),
                "mace_sp_log_path": str(sp_log_path),
                "mace_rank_basis": MACE_RANK_BASIS,
                "mace_eads_reference_status": MACE_EADS_REFERENCE_STATUS,
                **provenance,
            }
        )

    chunk_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata = {
        "chunk_label": chunk_label,
        "n_configs": len(chunk),
        "n_atoms": n_atoms,
        "runtime_s": runtime_s,
        **provenance,
        **_memory_fields("single_point_cuda_after", after_memory),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows, runtime_s, metadata


def _write_outputs(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_config = pd.DataFrame(rows)
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)
    for key, value in provenance.items():
        if key not in per_config:
            per_config[key] = value
    per_config["mace_dft_final_sp_rank"] = (
        per_config.groupby("system_id")["mace_dft_final_sp_total_energy_eV"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    per_config["dft_rank1_relative_energy_eV"] = (
        per_config["dft_adsorption_energy_eV"]
        - per_config.groupby("system_id")["dft_adsorption_energy_eV"].transform("min")
    )
    dft_rank1_mace_energy = per_config.loc[
        per_config.groupby("system_id")["dft_adsorption_energy_eV"].idxmin(),
        ["system_id", "mace_dft_final_sp_total_energy_eV"],
    ].rename(
        columns={
            "mace_dft_final_sp_total_energy_eV": "mace_total_energy_at_dft_rank1_geometry_eV"
        }
    )
    per_config = per_config.merge(
        dft_rank1_mace_energy,
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    per_config["mace_dft_rank1_relative_total_energy_eV"] = (
        per_config["mace_dft_final_sp_total_energy_eV"]
        - per_config["mace_total_energy_at_dft_rank1_geometry_eV"]
    )
    per_config["mace_dft_rank1_relative_energy_error_eV"] = (
        per_config["mace_dft_rank1_relative_total_energy_eV"]
        - per_config["dft_rank1_relative_energy_eV"]
    )

    summary_rows: list[dict[str, Any]] = []
    for system_id, group in per_config.groupby("system_id", sort=False):
        dft_best = group.loc[group["dft_adsorption_energy_eV"].idxmin()]
        sp_best = group.loc[group["mace_dft_final_sp_total_energy_eV"].idxmin()]
        top3 = group.nsmallest(min(3, len(group)), "mace_dft_final_sp_total_energy_eV")
        top5 = group.nsmallest(min(5, len(group)), "mace_dft_final_sp_total_energy_eV")
        spearman = (
            float(group["dft_rank"].corr(group["mace_dft_final_sp_rank"]))
            if len(group) > 1
            else float("nan")
        )
        summary_rows.append(
            {
                "system_id": system_id,
                "adsorbate": str(group.iloc[0]["adsorbate"]),
                "adsorbate_reference_species": str(
                    group.iloc[0]["adsorbate_reference_species"]
                ),
                "n_configs": int(len(group)),
                "dft_best_config": str(dft_best["config_id"]),
                "dft_best_energy_eV": float(dft_best["dft_adsorption_energy_eV"]),
                "dft_best_mace_dft_final_sp_rank": int(
                    dft_best["mace_dft_final_sp_rank"]
                ),
                "dft_final_sp_best_config": str(sp_best["config_id"]),
                "dft_final_sp_best_sid": int(sp_best["sid"]),
                "dft_final_sp_best_dft_rank": int(sp_best["dft_rank"]),
                "dft_final_sp_best_dft_gap_to_best_eV": float(
                    sp_best["dft_gap_to_best_eV"]
                ),
                "dft_final_sp_top1_success_0p10eV": bool(
                    sp_best["dft_gap_to_best_eV"] <= 0.1
                ),
                "dft_final_sp_top3_best_dft_gap_eV": float(
                    top3["dft_gap_to_best_eV"].min()
                ),
                "dft_final_sp_top3_success_0p10eV": bool(
                    top3["dft_gap_to_best_eV"].min() <= 0.1
                ),
                "dft_final_sp_top5_best_dft_gap_eV": float(
                    top5["dft_gap_to_best_eV"].min()
                ),
                "dft_final_sp_top5_success_0p10eV": bool(
                    top5["dft_gap_to_best_eV"].min() <= 0.1
                ),
                "dft_final_sp_spearman_rank_corr": spearman,
                "dft_rank1_anchored_relative_mae_eV": float(
                    group["mace_dft_rank1_relative_energy_error_eV"].abs().mean()
                ),
                "dft_rank1_anchored_relative_rmse_eV": float(
                    np.sqrt(
                        np.mean(group["mace_dft_rank1_relative_energy_error_eV"] ** 2)
                    )
                ),
                "dft_rank1_anchored_relative_bias_eV": float(
                    group["mace_dft_rank1_relative_energy_error_eV"].mean()
                ),
                "mace_dft_final_sp_free_fmax_median_eV_A": float(
                    group["mace_dft_final_sp_free_fmax_eV_A"].median()
                ),
                "mace_dft_final_sp_free_fmax_max_eV_A": float(
                    group["mace_dft_final_sp_free_fmax_eV_A"].max()
                ),
                "mace_rank_basis": MACE_RANK_BASIS,
                "mace_eads_reference_status": MACE_EADS_REFERENCE_STATUS,
            }
        )

    summary = pd.DataFrame(summary_rows)
    chunks = pd.DataFrame(chunk_rows)
    per_config.to_csv(paths["tables"] / "dft_final_sp_results.csv", index=False)
    summary.to_csv(paths["tables"] / "dft_final_sp_system_summary.csv", index=False)
    chunks.to_csv(paths["tables"] / "dft_final_sp_chunk_timings.csv", index=False)
    _write_report(args=args, paths=paths, summary=summary, chunks=chunks)
    return per_config, summary


def _write_report(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    summary: pd.DataFrame,
    chunks: pd.DataFrame,
) -> None:
    runtime_s = float(chunks["runtime_s"].sum()) if not chunks.empty else 0.0
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)
    model_label = toolkit_model_label(provenance)
    cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "n_configs",
        "dft_best_config",
        "dft_best_mace_dft_final_sp_rank",
        "dft_final_sp_best_config",
        "dft_final_sp_best_dft_rank",
        "dft_final_sp_best_dft_gap_to_best_eV",
        "dft_final_sp_top3_best_dft_gap_eV",
        "dft_final_sp_spearman_rank_corr",
        "dft_rank1_anchored_relative_mae_eV",
        "dft_rank1_anchored_relative_rmse_eV",
        "dft_rank1_anchored_relative_bias_eV",
        "mace_dft_final_sp_free_fmax_median_eV_A",
    ]
    lines = [
        "# OC20Dense DFT-Relaxed Final MACE Single-Point Check",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Backend: ALCHEMI Toolkit MACE single points on official OC20Dense DFT-relaxed final frames",
        f"Model: {model_label}; D3(BJ) disabled",
        "",
        "## What this checks",
        "",
        (
            "This run evaluates the selected Toolkit MACE checkpoint on the "
            "final frame of each matched OC20Dense DFT trajectory. It isolates "
            "the energy model on DFT-relaxed final geometries from the separate Toolkit "
            "relaxation path."
        ),
        "",
        (
            "The reported MACE energies are total energies and are used only to "
            "rank configurations within the same `system_id`. The separate "
            "MACE Eads layer applies the clean-surface and gas-reference "
            "subtraction."
        ),
        "",
        (
            "The DFT-rank-1 anchored relative columns compare energy gaps from "
            "the same reference geometry: the released DFT minimum. For each "
            "row, DFT uses `E_DFT(row) - E_DFT(rank1)` and MACE uses "
            "`E_MACE(row) - E_MACE(on the DFT-rank-1 geometry)`. This keeps "
            "the reference structure fixed instead of shifting MACE to its own "
            "minimum."
        ),
        "",
        "## System Summary",
        "",
        summary[cols].to_markdown(index=False),
        "",
        "## Runtime",
        "",
        f"- Systems: `{', '.join(args.systems)}`.",
        f"- Structures scored: {int(summary['n_configs'].sum())}.",
        f"- Chunk size: {args.chunk_size}.",
        f"- Single-point runtime total: {runtime_s:.2f} s.",
        "",
        "## Output Files",
        "",
        f"- Per-config results: `{paths['tables'] / 'dft_final_sp_results.csv'}`",
        f"- System summary: `{paths['tables'] / 'dft_final_sp_system_summary.csv'}`",
        f"- Chunk timings: `{paths['tables'] / 'dft_final_sp_chunk_timings.csv'}`",
        f"- DFT-relaxed final structures: `{paths['structures']}`",
    ]
    (paths["reports"] / "dft_final_single_point_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_oc20dense_dft_final_single_points(args: argparse.Namespace) -> int:
    require_precomputed_write_allowed(args.outdir)
    paths = _ensure_dirs(args.outdir)
    items = _load_items(args, paths)
    backend = _build_backend(args)
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)

    rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for start in range(0, len(items), args.chunk_size):
        chunk = items[start:start + args.chunk_size]
        chunk_index = start // args.chunk_size + 1
        chunk_label = f"dft_final_sp_chunk_{chunk_index:02d}"
        sp_rows, runtime_s, metadata = _single_point_chunk(
            backend=backend,
            chunk=chunk,
            chunk_label=chunk_label,
            paths=paths,
            force=args.force,
            provenance=provenance,
        )
        rows.extend(sp_rows)
        chunk_rows.append(
            {
                "chunk_label": chunk_label,
                "n_configs": len(chunk),
                "runtime_s": runtime_s,
                "n_atoms": int(metadata.get("n_atoms", 0)),
                **{
                    key: value
                    for key, value in metadata.items()
                    if key.startswith("single_point_cuda_after_")
                },
                **{
                    key: metadata.get(key, provenance.get(key))
                    for key in TOOLKIT_PROVENANCE_COLUMNS
                },
            }
        )
        print(f"{chunk_label}: {len(chunk)} configs, {runtime_s:.2f}s", flush=True)

    _per_config, summary = _write_outputs(
        args=args,
        paths=paths,
        rows=rows,
        chunk_rows=chunk_rows,
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "systems": list(args.systems),
        "chunk_size": int(args.chunk_size),
        "n_rows": int(len(_per_config)),
        "summary_csv": str(paths["tables"] / "dft_final_sp_system_summary.csv"),
        "per_config_csv": str(paths["tables"] / "dft_final_sp_results.csv"),
        "chunk_csv": str(paths["tables"] / "dft_final_sp_chunk_timings.csv"),
        **provenance,
    }
    (paths["reports"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(paths["tables"] / "dft_final_sp_system_summary.csv", flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_oc20dense_dft_final_single_points(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
