"""Visualisation utilities using OVITO Python API and matplotlib."""

from pathlib import Path
from typing import Literal

import ase
import ase.data
import pandas as pd

from .constants import AMU_TO_G, ANGSTROM3_TO_CM3

RendererName = Literal["tachyon", "visrtx", "anari", "ospray", "opengl"]


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


def _make_renderer(
    renderer: RendererName,
    samples_per_pixel: int,
):
    """Build a configured OVITO final-frame renderer."""
    renderer_key = renderer.lower()
    if renderer_key in {"visrtx", "anari"}:
        from ovito.vis import AnariRenderer

        r = AnariRenderer()
        r.samples_per_pixel = int(samples_per_pixel)
        r.denoising_enabled = True
        return r
    if renderer_key == "ospray":
        from ovito.vis import OSPRayRenderer

        r = OSPRayRenderer()
        r.samples_per_pixel = int(samples_per_pixel)
        r.denoising_enabled = True
        return r
    if renderer_key == "opengl":
        from ovito.vis import OpenGLRenderer

        return OpenGLRenderer()
    if renderer_key == "tachyon":
        from ovito.vis import TachyonRenderer

        return TachyonRenderer()
    raise ValueError(
        "renderer must be one of 'tachyon', 'visrtx', 'anari', 'ospray', or 'opengl'"
    )


def render_structure_ovito(
    atoms: ase.Atoms,
    output_path: str = "structure.png",
    size: tuple[int, int] = (800, 600),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    renderer: RendererName = "tachyon",
    samples_per_pixel: int = 64,
    show_cell: bool = True,
) -> str:
    """Render an ASE Atoms object to a PNG via OVITO.

    Parameters
    ----------
    atoms : ase.Atoms
    output_path : str
    size : tuple[int, int]
    background : tuple of 3 floats in [0, 1]
        RGB background colour.  Use (1, 1, 1) for white (default) or
        (0.15, 0.15, 0.15) for dark charcoal (better for light-coloured
        atoms such as hydrogen).
    renderer : {"tachyon", "visrtx", "anari", "ospray", "opengl"}
        Final-frame renderer.  "visrtx" and "anari" use OVITO's
        AnariRenderer, which is the Python API for NVIDIA VisRTX.
    samples_per_pixel : int
        Ray-tracing samples per pixel for VisRTX/ANARI and OSPRay.
    show_cell : bool
        If False, hide the simulation-cell wireframe.

    Returns the path to the rendered image.
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.vis import Viewport
    from ovito.pipeline import StaticSource, Pipeline

    clean = _clean_atoms_for_ovito(atoms)

    data = ase_to_ovito(clean)
    if not show_cell and data.cell_ is not None:
        data.cell_.vis.enabled = False

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all(size=size)

    ovito_renderer = _make_renderer(renderer, samples_per_pixel=samples_per_pixel)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        vp.render_image(
            filename=output_path,
            size=size,
            renderer=ovito_renderer,
            background=background,
        )
    finally:
        pipeline.remove_from_scene()
    return output_path


def create_interactive_view(
    atoms: ase.Atoms,
    width: str = "600px",
    height: str = "400px",
    particle_colors=None,
    show_cell: bool = True,
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
    show_cell : bool
        If False, hide the simulation-cell wireframe.  Useful for slab
        structures where the cell box (including vacuum) is misleading.

    Returns
    -------
    ipywidgets.DOMWidget or None
    """
    import numpy as np

    try:
        import ipywidgets
        from ovito.io.ase import ase_to_ovito
        from ovito.pipeline import StaticSource, Pipeline
        from ovito.gui import create_ipywidget
    except ImportError:
        return None

    clean = _clean_atoms_for_ovito(atoms)
    data = ase_to_ovito(clean)

    # Apply per-particle colours if provided
    if particle_colors is not None:
        colors = np.asarray(particle_colors, dtype=np.float64)
        data.particles_.create_property("Color", data=colors)

    # Optionally hide the simulation cell wireframe
    if not show_cell and data.cell_ is not None:
        data.cell_.vis.enabled = False

    pipeline = Pipeline(source=StaticSource(data=data))

    widget = create_ipywidget(
        pipeline,
        layout=ipywidgets.Layout(width=width, height=height),
    )
    return widget


def display_widgets_row(
    items: list[tuple[str, ase.Atoms]],
    width: str = "300px",
    height: str = "300px",
    particle_colors_list=None,
    show_cell: bool = True,
):
    """Display a horizontal row of labelled interactive OVITO widgets.

    Falls back to static PNG rendering if interactive widgets are
    unavailable.

    Parameters
    ----------
    items : list of (label, atoms) tuples
    width, height : str
        CSS size for each widget.
    particle_colors_list : list of np.ndarray or None
        Per-item particle colours (same length as *items*).
    show_cell : bool
        If False, hide the simulation-cell wireframe.
    """
    try:
        from ipywidgets import HBox, VBox, Layout
        from ipywidgets import HTML as HTMLWidget
        from IPython.display import display
    except ImportError:
        for label, atoms in items:
            print(f"{label}: {len(atoms)} atoms")
        return

    widgets = []
    for idx, (label, atoms) in enumerate(items):
        pc = particle_colors_list[idx] if particle_colors_list else None
        w = create_interactive_view(
            atoms, width=width, height=height,
            particle_colors=pc, show_cell=show_cell,
        )
        if w is not None:
            widgets.append(VBox([HTMLWidget(f"<b>{label}</b>"), w]))

    if widgets:
        display(HBox(widgets, layout=Layout(justify_content="center", gap="15px")))
    else:
        for label, atoms in items:
            print(f"{label}: {len(atoms)} atoms (widget unavailable)")


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
