#!/usr/bin/env python3
"""Render the checked Part 1 molecular box with OVITO.

This is an offline preparation tool. The learner notebook loads the rendered
image and the prebuilt structure instead of running Packmol during the lesson.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_domain_box(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    renderer_name: str,
) -> dict[str, object]:
    import ovito
    import numpy as np
    from ovito.io import import_file
    from ovito.modifiers import CreateBondsModifier
    from ovito.vis import AnariRenderer, TachyonRenderer, Viewport

    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing render: {output}")
    if renderer_name not in {"anari", "tachyon"}:
        raise ValueError("renderer_name must be 'anari' or 'tachyon'")

    pipeline = import_file(str(source), multiple_frames=False)
    bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    for first, second, cutoff in (
        ("H", "C", 1.25),
        ("H", "N", 1.25),
        ("H", "O", 1.20),
        ("C", "C", 1.75),
        ("C", "N", 1.70),
        ("C", "O", 1.65),
    ):
        bonds.set_pairwise_cutoff(first, second, cutoff)
    pipeline.modifiers.append(bonds)
    pipeline.add_to_scene()
    try:
        data = pipeline.compute()
        data.particles.vis.radius = 0.45
        if data.particles.bonds is not None:
            data.particles.bonds.vis.width = 0.18

        viewport = Viewport(type=Viewport.Type.Perspective)
        viewport.camera_dir = (-1.0, -1.0, -0.8)
        viewport.fov = 45.0
        viewport.zoom_all(size=(width, height))
        particle_center = np.asarray(data.particles.positions).mean(axis=0)
        camera_position = np.asarray(viewport.camera_pos)
        viewport.camera_pos = particle_center + 1.12 * (
            camera_position - particle_center
        )
        if renderer_name == "anari":
            renderer = AnariRenderer(
                samples_per_pixel=64,
                ambient_occlusion_samples=4,
                denoising_enabled=True,
            )
            renderer_label = "OVITO ANARI renderer"
        else:
            renderer = TachyonRenderer(
                ambient_occlusion=True,
                shadows=True,
                antialiasing_samples=12,
            )
            renderer_label = "OVITO TachyonRenderer"
        output.parent.mkdir(parents=True, exist_ok=True)
        viewport.render_image(
            filename=str(output),
            size=(width, height),
            background=(0.97, 0.98, 1.0),
            alpha=False,
            renderer=renderer,
        )
        bond_count = (
            0 if data.particles.bonds is None else int(data.particles.bonds.count)
        )
        particle_count = int(data.particles.count)
    finally:
        pipeline.remove_from_scene()

    return {
        "schema": "alchemi.part1-domain-box-render.v1",
        "renderer": renderer_label,
        "ovito_version": ".".join(str(value) for value in ovito.version),
        "source": source.name,
        "source_sha256": sha256_file(source),
        "output": output.name,
        "output_sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "width_px": width,
        "height_px": height,
        "particle_count": particle_count,
        "bond_count": bond_count,
        "camera_direction": [-1.0, -1.0, -0.8],
        "camera_distance_margin": 1.12,
        "background_rgb": [0.97, 0.98, 1.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument(
        "--renderer",
        choices=("anari", "tachyon"),
        default="anari",
        help="Use ANARI/VisRTX for the release render; Tachyon is for previews.",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    result = render_domain_box(
        args.source.resolve(),
        args.output.resolve(),
        width=args.width,
        height=args.height,
        renderer_name=args.renderer,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        if args.summary.exists():
            raise FileExistsError(
                f"refusing to replace existing summary: {args.summary}"
            )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
