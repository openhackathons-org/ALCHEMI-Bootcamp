"""Visualisation utilities using OVITO Python API and matplotlib."""

from pathlib import Path

import ase
import ase.data
import pandas as pd


def render_structure_ovito(
    atoms: ase.Atoms,
    output_path: str = "structure.png",
    size: tuple[int, int] = (800, 600),
) -> str:
    """Render an ASE Atoms object to a PNG via OVITO's TachyonRenderer.

    Returns the path to the rendered image.
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.vis import TachyonRenderer, Viewport
    from ovito.pipeline import StaticSource, Pipeline
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all()

    renderer = TachyonRenderer()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vp.render_image(
        filename=output_path,
        size=size,
        renderer=renderer,
        background=(1.0, 1.0, 1.0),
    )

    pipeline.remove_from_scene()
    return output_path


def display_inline(image_path: str):
    """Display a PNG image inline in a Jupyter notebook."""
    from IPython.display import Image, display
    display(Image(filename=image_path))


def structure_summary_table(atoms: ase.Atoms) -> pd.DataFrame:
    """Return a one-row DataFrame summarising key structural properties."""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    composition = {}
    for s in symbols:
        composition[s] = composition.get(s, 0) + 1
    formula = " ".join(f"{k}{v}" for k, v in sorted(composition.items()))

    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    vol = atoms.get_volume()

    total_mass_amu = ase.data.atomic_masses[atoms.numbers].sum()
    density = total_mass_amu * 1.66054e-24 / (vol * 1e-24) if vol > 0 else 0.0

    return pd.DataFrame([{
        "Formula": formula,
        "Atoms": n_atoms,
        "a (A)": round(a, 3),
        "b (A)": round(b, 3),
        "c (A)": round(c, 3),
        "Volume (A^3)": round(vol, 2),
        "Density (g/cm^3)": round(density, 3),
    }])
