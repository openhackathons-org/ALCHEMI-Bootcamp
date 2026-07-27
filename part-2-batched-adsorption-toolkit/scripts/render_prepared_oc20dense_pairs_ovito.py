#!/usr/bin/env python3
"""Render prepared OC20Dense structure pairs with OVITO Pro."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal

import numpy as np


PART1 = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PART1 / "outputs" / "ovito_dft_toolkit_pairs"
)
DEFAULT_JOBS = DEFAULT_OUTPUT / "prepared" / "render_jobs.csv"

RendererName = Literal["visrtx", "anari", "opengl"]
RENDER_SIZE = (900, 640)
BACKGROUND = (0.055, 0.062, 0.07)
CELL_COLOR = (0.56, 0.64, 0.61)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--renderer",
        choices=["visrtx", "anari", "opengl"],
        default="visrtx",
    )
    parser.add_argument("--samples-per-pixel", type=int, default=64)
    parser.add_argument(
        "--wsl-unc-root",
        default=r"\\wsl.localhost\Ubuntu",
        help="Windows UNC root used by ovitos.exe for WSL absolute paths.",
    )
    return parser.parse_args()


def path_for_runtime(value: str, wsl_unc_root: str) -> Path:
    path = Path(value)
    if path.exists() or os.name != "nt" or not value.startswith("/"):
        return path

    posix = PurePosixPath(value)
    unc_root = Path(wsl_unc_root)
    return unc_root.joinpath(*posix.parts[1:])


def read_jobs(path: Path, limit: int, wsl_unc_root: str) -> list[dict[str, str]]:
    jobs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["dft_json_runtime"] = str(path_for_runtime(row["dft_json"], wsl_unc_root))
            row["toolkit_json_runtime"] = str(
                path_for_runtime(row["toolkit_json"], wsl_unc_root)
            )
            jobs.append(row)
            if limit and len(jobs) >= limit:
                break
    return jobs


def load_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_renderer(renderer: RendererName, samples_per_pixel: int) -> tuple[Any, str]:
    if renderer in {"visrtx", "anari"}:
        from ovito.vis import AnariRenderer

        ovito_renderer = AnariRenderer()
        ovito_renderer.samples_per_pixel = int(samples_per_pixel)
        ovito_renderer.denoising_enabled = True
        ovito_renderer.dof_enabled = False
        if hasattr(ovito_renderer, "ambient_light_radiance"):
            ovito_renderer.ambient_light_radiance = 0.65
        if hasattr(ovito_renderer, "direct_light_irradiance"):
            ovito_renderer.direct_light_irradiance = 2.2
        return ovito_renderer, f"AnariRenderer({renderer}, spp={samples_per_pixel})"

    from ovito.vis import OpenGLRenderer

    return OpenGLRenderer(), "OpenGLRenderer"


def adsorbate_focus(payload: dict[str, Any]) -> np.ndarray:
    positions = np.asarray(payload["positions"], dtype=float)
    tags = np.asarray(payload.get("tags", []), dtype=int)
    if len(tags) == len(positions):
        adsorbate = positions[tags == 2]
        if len(adsorbate):
            return adsorbate.mean(axis=0)
        active = positions[tags > 0]
        if len(active):
            return active.mean(axis=0)
    return positions.mean(axis=0)


def shared_camera(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> dict[str, Any]:
    positions = np.vstack(
        [
            np.asarray(payload_a["positions"], dtype=float),
            np.asarray(payload_b["positions"], dtype=float),
        ]
    )
    focus = np.vstack([adsorbate_focus(payload_a), adsorbate_focus(payload_b)]).mean(axis=0)
    radius = np.linalg.norm(positions - focus, axis=1).max()
    fov = np.deg2rad(33.0)
    camera_unit = np.array([0.78, -1.18, 0.62], dtype=float)
    camera_unit = camera_unit / np.linalg.norm(camera_unit)
    distance = max(8.0, radius / np.sin(fov / 2.0) * 1.08)
    camera_pos = focus + camera_unit * distance
    return {
        "camera_pos": camera_pos,
        "camera_dir": focus - camera_pos,
        "fov": float(fov),
    }


def data_from_payload(payload: dict[str, Any]) -> Any:
    from ovito.data import DataCollection, ParticleType
    from ovito.vis import ParticlesVis

    symbols = payload["symbols"]
    positions = np.asarray(payload["positions"], dtype=float)
    colors = np.asarray(payload["colors"], dtype=float)

    type_map: dict[str, int] = {}
    type_ids = []
    for symbol in symbols:
        if symbol not in type_map:
            type_map[symbol] = len(type_map) + 1
        type_ids.append(type_map[symbol])

    data = DataCollection()
    cell = np.asarray(payload["cell"], dtype=float)
    pbc = [bool(x) for x in payload.get("pbc", [False, False, False])]
    if cell.shape == (3, 3):
        cell_obj = data.create_cell(matrix=cell.T, pbc=pbc)
        cell_obj.vis.enabled = True
        cell_obj.vis.render_cell = True
        cell_obj.vis.rendering_color = CELL_COLOR
        cell_obj.vis.line_width = 0.045

    particles = data.create_particles(count=len(symbols))
    particles.vis.shape = ParticlesVis.Shape.Sphere
    particles.create_property("Position", data=positions)
    particle_types = particles.create_property("Particle Type", data=type_ids)
    for symbol, type_id in type_map.items():
        particle_type = ParticleType(id=type_id, name=symbol)
        particle_type.load_defaults()
        particle_types.types.append(particle_type)
    particles.create_property("Color", data=colors)
    return data


def render_payload(
    payload: dict[str, Any],
    output_path: Path,
    camera: dict[str, Any],
    renderer: RendererName,
    samples_per_pixel: int,
) -> str:
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport

    data = data_from_payload(payload)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    viewport = Viewport(type=Viewport.Type.Perspective)
    viewport.camera_pos = tuple(float(x) for x in camera["camera_pos"])
    viewport.camera_dir = tuple(float(x) for x in camera["camera_dir"])
    viewport.fov = float(camera["fov"])
    ovito_renderer, renderer_label = make_renderer(renderer, samples_per_pixel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        viewport.render_image(
            filename=str(output_path),
            size=RENDER_SIZE,
            renderer=ovito_renderer,
            background=BACKGROUND,
        )
    finally:
        pipeline.remove_from_scene()
    return renderer_label


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ovito_version() -> str:
    import ovito

    return str(ovito.version_string)


def main() -> int:
    args = parse_args()
    jobs = read_jobs(args.jobs, args.limit, args.wsl_unc_root)
    if not jobs:
        raise ValueError(f"No jobs loaded from {args.jobs}")

    renders_dir = args.output / "renders"
    manifest_rows: list[dict[str, Any]] = []
    renderer_label = ""

    for row in jobs:
        dft_payload = load_payload(Path(row["dft_json_runtime"]))
        toolkit_payload = load_payload(Path(row["toolkit_json_runtime"]))
        camera = shared_camera(dft_payload, toolkit_payload)
        dft_png = renders_dir / f"{row['slug']}_dft_final.png"
        toolkit_png = renders_dir / f"{row['slug']}_toolkit_relaxed.png"

        renderer_label = render_payload(
            dft_payload,
            dft_png,
            camera,
            args.renderer,
            args.samples_per_pixel,
        )
        render_payload(
            toolkit_payload,
            toolkit_png,
            camera,
            args.renderer,
            args.samples_per_pixel,
        )

        for role, png_path, source_path, source_frame in (
            ("dft_final", dft_png, row["dft_source_path"], "last"),
            ("toolkit_relaxed", toolkit_png, row["toolkit_source_path"], "single"),
        ):
            manifest_rows.append(
                {
                    "output_path": str(png_path),
                    "role": role,
                    "source_path": source_path,
                    "source_frame": source_frame,
                    "system_id": row["system_id"],
                    "case": row["case"],
                    "config_id": row["config_id"],
                    "sid": row["sid"],
                    "renderer": renderer_label,
                    "ovito_version": ovito_version(),
                    "python_executable": sys.executable,
                    "camera_policy": (
                        "shared perspective camera per DFT/Toolkit pair; "
                        "camera direction targets adsorbate/tagged active atoms; "
                        "full structure fit; simulation cell rendered"
                    ),
                    "status": "draft",
                }
            )

    write_csv(args.output / "render_manifest.csv", manifest_rows)
    print(f"Rendered {len(jobs)} DFT/Toolkit pairs")
    print(f"Renderer: {renderer_label}")
    print(f"Render manifest: {args.output / 'render_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
