"""Small OVITO helpers for the Part 1 surface-model example.

The notebook uses OVITO's official Jupyter widget for the interactive view.
Imports stay inside the functions so the data and table tests do not require a
working OpenGL/Qt installation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms


def _ovito_compatible_copy(atoms: Atoms) -> Atoms:
    """Copy an ASE structure and normalize display-only particle properties."""

    display_atoms = atoms.copy()
    for name, values in tuple(display_atoms.arrays.items()):
        if values.dtype == np.dtype(bool):
            # OVITO 3.15 particle properties do not accept NumPy bool arrays.
            # The int8 copy preserves the 0/1 mask without changing the source.
            del display_atoms.arrays[name]
            display_atoms.new_array(name, values.astype(np.int8, copy=True))
    if display_atoms.cell.rank:
        display_atoms.wrap()
    return display_atoms


def make_ovito_widget(
    atoms: Atoms,
    *,
    width: str = "260px",
    height: str = "220px",
    show_cell: bool = True,
) -> Any:
    """Return OVITO's official interactive widget for one ASE structure."""

    import ipywidgets
    from ovito.gui import create_ipywidget
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    display_atoms = _ovito_compatible_copy(atoms)
    data = ase_to_ovito(display_atoms)
    if data.cell_ is not None:
        data.cell_.vis.enabled = bool(show_cell)
        if hasattr(data.cell_.vis, "render_cell"):
            data.cell_.vis.render_cell = bool(show_cell)
    if data.particles is not None:
        data.particles.vis.radius = max(
            float(getattr(data.particles.vis, "radius", 0.0)), 0.42
        )
    pipeline = Pipeline(source=StaticSource(data=data))
    widget = create_ipywidget(
        pipeline,
        layout=ipywidgets.Layout(width=width, height=height),
    )
    # Keep the pipeline alive for as long as the widget is displayed.
    widget._alchemi_pipeline = pipeline
    return widget


def adsorption_widget_grid(
    structures: Sequence[tuple[str, Atoms]],
    *,
    columns: int = 2,
    width: str = "260px",
    height: str = "220px",
) -> Any:
    """Return a labelled grid of official OVITO widgets."""

    if columns <= 0:
        raise ValueError("columns must be positive")
    import ipywidgets

    cards = []
    for label, atoms in structures:
        cards.append(
            ipywidgets.VBox(
                [
                    ipywidgets.HTML(f"<b>{label}</b>"),
                    make_ovito_widget(
                        atoms,
                        width=width,
                        height=height,
                        show_cell=True,
                    ),
                ],
                layout=ipywidgets.Layout(width=width, gap="4px"),
            )
        )
    rows = [
        ipywidgets.HBox(
            cards[start : start + columns],
            layout=ipywidgets.Layout(gap="14px", align_items="flex-start"),
        )
        for start in range(0, len(cards), columns)
    ]
    return ipywidgets.VBox(rows, layout=ipywidgets.Layout(gap="14px"))
