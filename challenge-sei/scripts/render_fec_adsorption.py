"""Render the relaxed FEC adsorption structures (on Li metal and LiF) to PNG.

Loads the selected lowest-energy relaxed structures the solution notebook
exported to outputs/ovito_structures/, fades the frozen bottom half of each
slab (same style as scripts/render_surfaces.py), and renders perspective,
side, and top views into outputs/surface_png/.
"""

import sys
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHALLENGE_DIR))
sys.path.insert(0, str(CHALLENGE_DIR.parent / "part-1-batched-adsorption"))

import numpy as np
import pandas as pd
from ase.io import read

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from ovito.qt_compat import QtCore
from ovito.vis import TachyonRenderer, TextLabelOverlay, Viewport

STRUCTURE_DIR = CHALLENGE_DIR / "outputs" / "ovito_structures"
OUTPUT_DIR = CHALLENGE_DIR / "outputs" / "surface_png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (1600, 1200)
FROZEN_TRANSPARENCY = 0.65

SYSTEMS = (
    ("FEC_li_metal_Li_metal.extxyz", "Li metal bcc(100)", 90),
    ("FEC_passivating_LiF.extxyz", "LiF rocksalt(100)", 144),
)

energies = pd.read_csv(CHALLENGE_DIR / "outputs" / "raw_component_energies.csv")
energies = energies[energies["candidate_id"].eq("FEC")].set_index("interaction")

renderer = TachyonRenderer(antialiasing_samples=32, direct_light_intensity=1.1)

for filename, surface_name, n_slab in SYSTEMS:
    atoms = read(STRUCTURE_DIR / filename)
    interaction = atoms.info["interaction"]

    row = energies.loc[interaction]
    e_bind = row["E_surface_species_eV"] - row["E_surface_eV"] - row["E_species_eV"]

    # Fade the frozen bottom half of the slab; the FEC molecule (atoms appended
    # after the slab) and the mobile top layers stay opaque.
    z = atoms.positions[:, 2]
    slab_z = z[:n_slab]
    frozen = np.zeros(len(atoms), dtype=bool)
    frozen[:n_slab] = slab_z < (slab_z.min() + slab_z.max()) / 2.0

    data = ase_to_ovito(atoms)
    data.particles_.create_property(
        "Transparency", data=np.where(frozen, FROZEN_TRANSPARENCY, 0.0)
    )
    data.cell.vis.enabled = False

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    label = TextLabelOverlay(
        text=(
            f"FEC on {surface_name} · {row['selected_site_label']}, "
            f"{row['selected_start_orientation']} · E_bind = {e_bind:.2f} eV"
        ),
        alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter,
        font_size=0.02,
        text_color=(0.1, 0.1, 0.1),
    )

    for view_name, camera_dir in (
        ("perspective", (-2.0, -1.2, -0.8)),
        ("side", (0.0, -1.0, 0.0)),
        ("top", (0.0, 0.0, -1.0)),
    ):
        vp = Viewport(type=Viewport.Type.Ortho, camera_dir=camera_dir)
        vp.zoom_all(size=IMAGE_SIZE)
        vp.overlays.append(label)
        out_path = OUTPUT_DIR / f"{filename.removesuffix('.extxyz')}_{view_name}.png"
        vp.render_image(
            filename=str(out_path),
            size=IMAGE_SIZE,
            renderer=renderer,
            background=(1.0, 1.0, 1.0),
        )
        print(f"Wrote {out_path}")

    pipeline.remove_from_scene()

print("Done.")
