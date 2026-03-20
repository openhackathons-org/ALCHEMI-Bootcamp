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

    # Optionally hide the simulation cell wireframe
    if not show_cell and data.cell_ is not None:
        data.cell_.vis.enabled = False

    # Clear previous pipelines so widgets don't overlap
    while scene.pipelines:
        scene.pipelines[-1].remove_from_scene()

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all()

    widget = create_ipywidget(vp, layout=ipywidgets.Layout(width=width, height=height))
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


def plot_electrolysis_diagram(output_path: str) -> str:
    """Render a schematic of PEM water electrolysis (HER + OER) and save to *output_path*.

    Returns the path to the saved image.
    """
    import os

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Electrolyte background
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.5, 0.2), 9.0, 5.5, boxstyle="round,pad=0.15",
            facecolor="#dbeafe", edgecolor="#3b82f6", linewidth=2,
        )
    )
    ax.text(5.0, 0.45, "Electrolyte (H$_2$O)", ha="center", fontsize=11, color="#1e40af")

    # Cathode (left) — HER
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (1.0, 0.9), 2.0, 3.2, boxstyle="round,pad=0.1",
            facecolor="#d1d5db", edgecolor="#374151", linewidth=2,
        )
    )
    ax.text(2.0, 3.8, "Cathode (\u2212)", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.0, 3.4, "Hydrogen Evolution\nReaction (HER)", ha="center", fontsize=8, color="#6b7280")
    ax.text(2.0, 2.5, "4H$^+$ + 4e$^-$\n\u2192 2H$_2$\u2191", ha="center", fontsize=11, color="#1f2937")
    ax.text(2.0, 1.6, r"$\mathbf{H_2}$", ha="center", fontsize=18, color="#16a34a", fontweight="bold")
    ax.text(2.0, 1.15, "produced", ha="center", fontsize=9, color="#6b7280")

    # Anode (right) — OER
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (7.0, 0.9), 2.0, 3.2, boxstyle="round,pad=0.1",
            facecolor="#fde68a", edgecolor="#d97706", linewidth=2,
        )
    )
    ax.text(8.0, 3.8, "Anode (+)", ha="center", fontsize=12, fontweight="bold")
    ax.text(8.0, 3.4, "Oxygen Evolution\nReaction (OER)", ha="center", fontsize=8, color="#6b7280")
    ax.text(8.0, 2.5, "2H$_2$O \u2192\nO$_2$\u2191 + 4H$^+$\n+ 4e$^-$", ha="center", fontsize=11, color="#92400e")
    ax.text(8.0, 1.6, r"$\mathbf{O_2}$", ha="center", fontsize=18, color="#dc2626", fontweight="bold")
    ax.text(8.0, 1.15, "produced", ha="center", fontsize=9, color="#6b7280")

    # Electron flow arrow
    ax.annotate(
        "", xy=(3.3, 4.8), xytext=(6.7, 4.8),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#ef4444"),
    )
    ax.text(5.0, 5.1, "4 e$^-$", ha="center", fontsize=12, fontweight="bold", color="#ef4444")
    ax.text(5.0, 5.4, "External circuit", ha="center", fontsize=10, color="#6b7280")

    # Ion flow arrow
    ax.annotate(
        "", xy=(6.7, 2.2), xytext=(3.3, 2.2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#3b82f6", linestyle="dashed"),
    )
    ax.text(5.0, 2.5, "H$^+$ migration", ha="center", fontsize=10, color="#1e40af")

    # Overall reaction
    ax.text(
        5.0, -0.3,
        r"Overall: 2H$_2$O $\rightarrow$ 2H$_2$ + O$_2$   ($E^0$ = 1.23 V)",
        ha="center", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#9ca3af"),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_oer_energy_ladders(ev_per_step: float, output_path: str) -> str:
    """Render a side-by-side comparison of non-ideal vs ideal OER energy ladders.

    Parameters
    ----------
    ev_per_step : float
        Ideal free-energy step (1.23 eV for OER).
    output_path : str
        Path to save the figure.

    Returns the path to the saved image.
    """
    import os

    import matplotlib.pyplot as plt

    labels = ["H$_2$O + $*$", "OH$^*$", "O$^*$", "OOH$^*$", "$*$ + O$_2$"]

    def _draw_ladder(ax, steps_vals, color, lw=2.5, alpha=1.0):
        for i in range(len(steps_vals)):
            x0 = i * 1.5
            ax.plot([x0, x0 + 1.2], [steps_vals[i], steps_vals[i]], color=color, lw=lw, alpha=alpha)
            if i < len(steps_vals) - 1:
                ax.plot(
                    [x0 + 1.2, (i + 1) * 1.5],
                    [steps_vals[i], steps_vals[i + 1]],
                    color=color, lw=1, alpha=alpha * 0.5, linestyle="--",
                )
        for i, lbl in enumerate(labels):
            ax.text(i * 1.5 + 0.6, -0.4, lbl, ha="center", fontsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Left panel: non-ideal catalyst
    ax_l = axes[0]
    non_ideal = [0.0, 0.8, 2.8, 3.2, 4.92]
    _draw_ladder(ax_l, non_ideal, "#2563eb")

    deltas = [non_ideal[i + 1] - non_ideal[i] for i in range(4)]
    rls_idx = deltas.index(max(deltas))
    x_rls = rls_idx * 1.5 + 1.35
    y_lo, y_hi = non_ideal[rls_idx], non_ideal[rls_idx + 1]
    ax_l.annotate(
        "", xy=(x_rls, y_hi - 0.05), xytext=(x_rls, y_lo + 0.05),
        arrowprops=dict(arrowstyle="<->", lw=2, color="#dc2626"),
    )
    rls_label = (
        r"$\Delta G_{" + str(rls_idx + 1) + r"}$ = "
        + f"{deltas[rls_idx]:.2f} eV" + "\n(rate-limiting)"
    )
    ax_l.text(
        x_rls + 0.15, (y_lo + y_hi) / 2, rls_label,
        fontsize=9, color="#dc2626", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#fca5a5"),
    )
    eta = max(deltas) - ev_per_step
    ax_l.text(
        0.5, 4.6, r"$\eta$ = " + f"{eta:.2f} V overpotential",
        fontsize=10, color="#dc2626", fontweight="bold",
    )
    ax_l.set_title("Non-Ideal Catalyst", fontsize=13, fontweight="bold")
    ax_l.set_ylabel("Adsorption Energy (eV)", fontsize=12)

    # Right panel: ideal catalyst
    ax_r = axes[1]
    ideal = [i * ev_per_step for i in range(5)]
    _draw_ladder(ax_r, ideal, "#16a34a")

    for i in range(4):
        x_mid = i * 1.5 + 1.35
        y_mid = (ideal[i] + ideal[i + 1]) / 2
        ax_r.text(
            x_mid + 0.1, y_mid,
            r"$\Delta G$ = " + f"{ev_per_step} eV",
            fontsize=8, color="#16a34a", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#dcfce7", edgecolor="#86efac"),
        )
    ax_r.text(
        0.5, 4.6, r"$\eta$ = 0 V (theoretical minimum)",
        fontsize=10, color="#16a34a", fontweight="bold",
    )
    ax_r.set_title("Ideal Catalyst", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_xlim(-0.3, 7)
        ax.set_ylim(-0.7, 5.5)
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    fig.suptitle("OER Adsorption Energy Diagrams", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
