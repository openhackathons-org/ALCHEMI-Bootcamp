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
"""Compose labeled side-by-side panels from OVITO-rendered OC20Dense images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PART1 = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PART1 / "outputs" / "ovito_dft_toolkit_pairs"
)
DEFAULT_JOBS = DEFAULT_OUTPUT / "prepared" / "render_jobs.csv"

RENDER_SIZE = (900, 640)
PAIR_SIZE = (1900, 870)
CONTACT_THUMB = (610, 280)
CHARCOAL = (13, 16, 18)
GRAPHITE = (24, 29, 33)
TEXT = (238, 244, 240)
MUTED = (176, 188, 182)
NV_GREEN = (118, 185, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def read_jobs(path: Path, limit: int) -> list[dict[str, str]]:
    jobs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            jobs.append(row)
            if limit and len(jobs) >= limit:
                break
    return jobs


def image_nonblank(path: Path) -> tuple[bool, float, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    std = float(arr.std())
    return std > 2.0, std, image.size


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fill: tuple[int, int, int],
    fnt: ImageFont.ImageFont,
) -> None:
    draw.text(xy, value, fill=fill, font=fnt)


def compose_pair(row: dict[str, str], output: Path) -> Path:
    dft_png = output / "renders" / f"{row['slug']}_dft_final.png"
    toolkit_png = output / "renders" / f"{row['slug']}_toolkit_relaxed.png"
    if not dft_png.exists():
        raise FileNotFoundError(f"Missing DFT render: {dft_png}")
    if not toolkit_png.exists():
        raise FileNotFoundError(f"Missing Toolkit render: {toolkit_png}")

    canvas = Image.new("RGB", PAIR_SIZE, CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    margin = 44
    title = f"{row['system_id']}  {row['case']}  {row['config_id']}  sid{row['sid']}"
    metrics = (
        f"DFT rank {row['dft_rank']} | gap {float(row['dft_gap_to_best_eV']):.4f} eV | "
        f"active RMSD {float(row['active_rmsd_A']):.3f} A | "
        f"ads RMSD {float(row['adsorbate_rmsd_A']):.3f} A"
    )
    draw_text(draw, (margin, 28), title, TEXT, font(34, bold=True))
    draw_text(draw, (margin, 74), metrics, MUTED, font(24))

    left = Image.open(dft_png).convert("RGB")
    right = Image.open(toolkit_png).convert("RGB")
    y = 140
    x1 = margin
    x2 = margin + RENDER_SIZE[0] + 20
    for x, label, image in (
        (x1, "DFT-relaxed final frame", left),
        (x2, "Toolkit relaxed", right),
    ):
        draw.rectangle(
            (x - 2, y - 44, x + RENDER_SIZE[0] + 2, y + RENDER_SIZE[1] + 2),
            fill=GRAPHITE,
            outline=NV_GREEN,
            width=2,
        )
        draw_text(draw, (x + 18, y - 36), label, TEXT, font(24, bold=True))
        canvas.paste(image, (x, y))

    footer = "Same camera per pair; colors/radii follow the Toolkit tag groups."
    draw_text(draw, (margin, PAIR_SIZE[1] - 44), footer, MUTED, font(20))
    pair_path = output / "pairs" / f"{row['slug']}_side_by_side.png"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(pair_path)
    return pair_path


def make_contact_sheet(
    pair_paths: list[Path],
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    cols = 3
    cell_w = CONTACT_THUMB[0] + 36
    cell_h = CONTACT_THUMB[1] + 92
    width = cols * cell_w + 44
    n_rows = int(np.ceil(len(pair_paths) / cols))
    height = n_rows * cell_h + 112
    sheet = Image.new("RGB", (width, height), CHARCOAL)
    draw = ImageDraw.Draw(sheet)
    draw_text(draw, (28, 22), "DFT-relaxed final vs Toolkit relaxed", TEXT, font(34, bold=True))
    draw_text(
        draw,
        (28, 66),
        "Selected OC20Dense visual checks; fixed camera within each pair.",
        MUTED,
        font(22),
    )

    for idx, (path, row) in enumerate(zip(pair_paths, rows, strict=True)):
        col = idx % cols
        grid_row = idx // cols
        x = 28 + col * cell_w
        y = 112 + grid_row * cell_h
        image = Image.open(path).convert("RGB").resize(
            CONTACT_THUMB,
            Image.Resampling.LANCZOS,
        )
        draw.rectangle(
            (x - 2, y - 2, x + CONTACT_THUMB[0] + 2, y + CONTACT_THUMB[1] + 2),
            fill=GRAPHITE,
            outline=NV_GREEN,
            width=2,
        )
        sheet.paste(image, (x, y))
        label = f"{row['system_id']} {row['case']}"
        metric = (
            f"{row['config_id']} sid{row['sid']} | "
            f"ads {float(row['adsorbate_rmsd_A']):.3f} A"
        )
        draw_text(draw, (x, y + CONTACT_THUMB[1] + 12), label, TEXT, font(20, bold=True))
        draw_text(draw, (x, y + CONTACT_THUMB[1] + 40), metric, MUTED, font(18))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


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
    rows = read_jobs(args.jobs, args.limit)
    if not rows:
        raise ValueError(f"No jobs loaded from {args.jobs}")

    pair_paths = [compose_pair(row, args.output) for row in rows]
    contact_sheet = args.output / "dft_toolkit_selected_contact_sheet.png"
    make_contact_sheet(pair_paths, rows, contact_sheet)

    qa_rows: list[dict[str, Any]] = []
    for path in [
        *(args.output / "renders").glob("*.png"),
        *pair_paths,
        contact_sheet,
    ]:
        nonblank, std, size = image_nonblank(path)
        qa_rows.append(
            {
                "output_path": str(path),
                "nonblank": nonblank,
                "pixel_std": std,
                "width": size[0],
                "height": size[1],
            }
        )
    write_csv(args.output / "visual_qa.csv", qa_rows)
    print(f"Composed {len(pair_paths)} side-by-side pairs")
    print(f"Contact sheet: {contact_sheet}")
    print(f"Visual QA: {args.output / 'visual_qa.csv'}")
    print(f"Python: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
