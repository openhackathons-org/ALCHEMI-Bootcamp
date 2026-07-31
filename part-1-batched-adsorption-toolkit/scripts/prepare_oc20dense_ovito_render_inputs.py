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
"""Prepare OC20Dense DFT-vs-Toolkit structures for OVITO Pro rendering."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read as ase_read


PART1 = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PART1
    / "outputs"
    / "oc20dense_known_examples"
    / "dft_reference_checks"
    / "selected_case_comparison.csv"
)
DEFAULT_OUTPUT = (
    PART1 / "outputs" / "ovito_dft_toolkit_pairs"
)

ELEMENT_COLORS = {
    "H": (0.92, 0.94, 0.91),
    "C": (0.08, 0.09, 0.10),
    "N": (0.20, 0.32, 0.85),
    "O": (0.86, 0.18, 0.16),
    "F": (0.45, 0.92, 0.35),
    "S": (0.95, 0.78, 0.20),
    "Cu": (0.78, 0.48, 0.25),
    "Pd": (0.58, 0.66, 0.70),
    "Hg": (0.62, 0.62, 0.72),
    "Sr": (0.22, 0.78, 0.38),
}


@dataclass(frozen=True)
class PairRow:
    system_id: str
    case: str
    config_id: str
    sid: int
    dft_rank: int
    dft_gap_to_best_eV: float
    active_rmsd_A: float
    adsorbate_rmsd_A: float
    n_dft_frames: int
    dft_trajectory_path: Path
    ml_relaxed_path: Path

    @property
    def slug(self) -> str:
        case = (
            self.case.lower()
            .replace("-", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        return f"{self.system_id}_{case}_{self.config_id}_sid{self.sid}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path, limit: int = 0) -> list[PairRow]:
    rows: list[PairRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            rows.append(
                PairRow(
                    system_id=str(item["system_id"]),
                    case=str(item["case"]),
                    config_id=str(item["config_id"]),
                    sid=int(item["sid"]),
                    dft_rank=int(item["dft_rank"]),
                    dft_gap_to_best_eV=float(item["dft_gap_to_best_eV"]),
                    active_rmsd_A=float(item["active_rmsd_A"]),
                    adsorbate_rmsd_A=float(item["adsorbate_rmsd_A"]),
                    n_dft_frames=int(item["n_dft_frames"]),
                    dft_trajectory_path=Path(item["dft_trajectory_path"]),
                    ml_relaxed_path=Path(item["ml_relaxed_path"]),
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def element_color(symbol: str) -> tuple[float, float, float]:
    if symbol in ELEMENT_COLORS:
        return ELEMENT_COLORS[symbol]
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    rgb = jmol_colors[atomic_numbers[symbol]]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def colors_and_radii(
    atoms: Atoms,
    reference_tags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    colors = []
    radii = []
    for symbol, number, tag in zip(
        atoms.get_chemical_symbols(), atoms.numbers, reference_tags, strict=True
    ):
        color = np.array(element_color(symbol), dtype=float)
        if int(tag) == 0:
            color = color * 0.62 + np.array([0.10, 0.11, 0.12]) * 0.38
        elif int(tag) == 1:
            color = color * 0.86 + np.array([0.92, 0.92, 0.85]) * 0.14
        elif int(tag) == 2:
            color = color * 0.70 + np.array([0.61, 1.00, 0.29]) * 0.30
        colors.append(np.clip(color, 0.02, 1.0))

        radius = float(covalent_radii[number]) * (0.36 if int(tag) != 2 else 0.46)
        radii.append(min(max(radius, 0.16), 0.72))

    return np.asarray(colors, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def atoms_payload(
    atoms: Atoms,
    reference_tags: np.ndarray,
    source_path: Path,
    source_frame: str,
) -> dict[str, Any]:
    colors, radii = colors_and_radii(atoms, reference_tags)
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.asarray(atoms.positions, dtype=float).tolist(),
        "tags": np.asarray(reference_tags, dtype=int).tolist(),
        "colors": colors.tolist(),
        "radii": radii.tolist(),
        "cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "pbc": [bool(x) for x in atoms.pbc],
        "source_path": str(source_path),
        "source_frame": source_frame,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input, args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.input}")

    structures_dir = args.output / "prepared" / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []

    for row in rows:
        if not row.dft_trajectory_path.exists():
            raise FileNotFoundError(f"Missing DFT trajectory: {row.dft_trajectory_path}")
        if not row.ml_relaxed_path.exists():
            raise FileNotFoundError(
                f"Missing Toolkit relaxed structure: {row.ml_relaxed_path}"
            )

        dft_frames = ase_read(row.dft_trajectory_path, ":")
        if isinstance(dft_frames, Atoms):
            dft_frames = [dft_frames]
        dft_final = list(dft_frames)[-1]
        toolkit_final = ase_read(row.ml_relaxed_path)
        reference_tags = np.asarray(toolkit_final.get_tags(), dtype=int)

        if len(dft_final) != len(toolkit_final):
            raise ValueError(
                f"Atom-count mismatch for {row.slug}: "
                f"{len(dft_final)} vs {len(toolkit_final)}"
            )
        if dft_final.get_chemical_symbols() != toolkit_final.get_chemical_symbols():
            raise ValueError(f"Species/order mismatch for {row.slug}")

        dft_json = structures_dir / f"{row.slug}_dft_final.json"
        toolkit_json = structures_dir / f"{row.slug}_toolkit_relaxed.json"
        dft_json.write_text(
            json.dumps(
                atoms_payload(
                    dft_final,
                    reference_tags,
                    row.dft_trajectory_path,
                    "last",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        toolkit_json.write_text(
            json.dumps(
                atoms_payload(
                    toolkit_final,
                    reference_tags,
                    row.ml_relaxed_path,
                    "single",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        jobs.append(
            {
                "slug": row.slug,
                "system_id": row.system_id,
                "case": row.case,
                "config_id": row.config_id,
                "sid": row.sid,
                "dft_rank": row.dft_rank,
                "dft_gap_to_best_eV": row.dft_gap_to_best_eV,
                "active_rmsd_A": row.active_rmsd_A,
                "adsorbate_rmsd_A": row.adsorbate_rmsd_A,
                "n_dft_frames": row.n_dft_frames,
                "n_atoms": len(dft_final),
                "dft_json": str(dft_json),
                "toolkit_json": str(toolkit_json),
                "dft_source_path": str(row.dft_trajectory_path),
                "toolkit_source_path": str(row.ml_relaxed_path),
            }
        )

    write_csv(args.output / "prepared" / "render_jobs.csv", jobs)
    print(f"Prepared {len(jobs)} render jobs")
    print(f"Render jobs: {args.output / 'prepared' / 'render_jobs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
