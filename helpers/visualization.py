"""Visualisation utilities using OVITO Python API and matplotlib."""

from pathlib import Path

import ase
import ase.data
import pandas as pd

from .constants import AMU_TO_G, ANGSTROM3_TO_CM3


def _clean_atoms_for_ovito(atoms: ase.Atoms) -> ase.Atoms:
    """Strip arrays with dtypes unsupported by OVITO (e.g. Unicode strings
    added by pymatgen's SlabGenerator) to avoid conversion errors."""
    _supported = {"int8", "int16", "int32", "int64", "float32", "float64"}
    clean = atoms.copy()
    for key in list(clean.arrays):
        if key in ("numbers", "positions"):
            continue
        if clean.arrays[key].dtype.name not in _supported:
            del clean.arrays[key]
    return clean


def render_structure_ovito(
    atoms: ase.Atoms,
    output_path: str = "structure.png",
    size: tuple[int, int] = (800, 600),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Render an ASE Atoms object to a PNG via OVITO's TachyonRenderer.

    Parameters
    ----------
    atoms : ase.Atoms
    output_path : str
    size : tuple[int, int]
    background : tuple of 3 floats in [0, 1]
        RGB background colour.  Use (1, 1, 1) for white (default) or
        (0.15, 0.15, 0.15) for dark charcoal (better for light-coloured
        atoms such as hydrogen).

    Returns the path to the rendered image.
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.vis import TachyonRenderer, Viewport
    from ovito.pipeline import StaticSource, Pipeline

    clean = _clean_atoms_for_ovito(atoms)

    data = ase_to_ovito(clean)
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
        background=background,
    )

    pipeline.remove_from_scene()
    return output_path


def create_interactive_view(
    atoms: ase.Atoms,
    width: str = "600px",
    height: str = "400px",
    particle_colors=None,
):
    """Create an interactive 3-D OVITO widget for Jupyter notebooks.

    Falls back to None if ipywidgets or OVITO GUI is unavailable.

    Parameters
    ----------
    atoms : ase.Atoms
    width, height : str
        CSS size strings for the widget layout.
    particle_colors : np.ndarray shape (N, 3) or None
        Per-particle RGB colours in [0, 1].  When provided, these
        override OVITO's default element colouring.

    Returns
    -------
    ipywidgets.DOMWidget or None
    """
    import numpy as np

    try:
        import ipywidgets
        from ovito.io.ase import ase_to_ovito
        from ovito.vis import Viewport
        from ovito.pipeline import StaticSource, Pipeline
        from ovito.gui import create_ipywidget
    except ImportError:
        return None

    from ovito import scene

    clean = _clean_atoms_for_ovito(atoms)
    data = ase_to_ovito(clean)

    # Apply per-particle colours if provided
    if particle_colors is not None:
        colors = np.asarray(particle_colors, dtype=np.float64)
        data.particles_.create_property("Color", data=colors)

    # Clear previous pipelines so widgets don't overlap
    while scene.pipelines:
        scene.pipelines[-1].remove_from_scene()

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all()

    widget = create_ipywidget(vp, layout=ipywidgets.Layout(width=width, height=height))
    return widget


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
    density = total_mass_amu * AMU_TO_G / (vol * ANGSTROM3_TO_CM3) if vol > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "Formula": formula,
                "Atoms": n_atoms,
                "a (A)": round(a, 3),
                "b (A)": round(b, 3),
                "c (A)": round(c, 3),
                "Volume (A^3)": round(vol, 2),
                "Density (g/cm^3)": round(density, 3),
            }
        ]
    )
