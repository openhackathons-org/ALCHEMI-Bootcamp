"""Render the challenge adsorption slabs (Li metal, LiF) to PNG with OVITO.

Builds the slabs with the same challenge_utils builders used by the solution
notebook, dims the frozen bottom half of each slab (frozen_surface_fraction),
and renders a perspective and a top view per surface into outputs/surface_png/.
"""

import sys
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHALLENGE_DIR))
sys.path.insert(0, str(CHALLENGE_DIR.parent / "part-1-batched-adsorption"))

import numpy as np

from challenge_utils.solution_helpers import (
    SolutionSettings,
    build_adsorption_surfaces,
    surface_active_mask,
)

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from ovito.qt_compat import QtCore
from ovito.vis import TachyonRenderer, TextLabelOverlay, Viewport

SETTINGS = SolutionSettings(frozen_surface_fraction=0.5)
OUTPUT_DIR = CHALLENGE_DIR / "outputs" / "surface_png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (1600, 1200)
FROZEN_TRANSPARENCY = 0.65

surfaces = build_adsorption_surfaces(["LiF", "Li_metal"], settings=SETTINGS)

renderer = TachyonRenderer(antialiasing_samples=32, direct_light_intensity=1.1)

for surface_id, atoms in surfaces.items():
    active = np.asarray(surface_active_mask(atoms, settings=SETTINGS), dtype=bool)

    data = ase_to_ovito(atoms)
    transparency = np.where(active, 0.0, FROZEN_TRANSPARENCY)
    data.particles_.create_property("Transparency", data=transparency)
    data.cell.vis.enabled = False

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    slab_names = {"Li_metal": "Li metal bcc(100)", "LiF": "LiF rocksalt(100)"}
    label = TextLabelOverlay(
        text=(
            f"{slab_names.get(surface_id, surface_id)} slab · {len(atoms)} atoms "
            f"· frozen half faded"
        ),
        alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter,
        font_size=0.02,
        text_color=(0.1, 0.1, 0.1),
    )

    for view_name, camera_dir, zoom_crop in (
        ("perspective", (-2.0, -1.2, -0.8), 1.0),
        ("top", (0.0, 0.0, -1.0), 1.0),
    ):
        vp = Viewport(type=Viewport.Type.Ortho, camera_dir=camera_dir)
        vp.zoom_all(size=IMAGE_SIZE)
        vp.fov *= zoom_crop
        vp.overlays.append(label)
        out_path = OUTPUT_DIR / f"{surface_id}_{view_name}.png"
        vp.render_image(
            filename=str(out_path),
            size=IMAGE_SIZE,
            renderer=renderer,
            background=(1.0, 1.0, 1.0),
        )
        print(f"Wrote {out_path}")

    pipeline.remove_from_scene()

print("Done.")
