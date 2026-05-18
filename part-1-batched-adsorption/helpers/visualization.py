"""Visualisation utilities using OVITO Python API and matplotlib."""

import math
import os
import re
from pathlib import Path
from typing import Literal

import ase
import ase.data
import pandas as pd

from .constants import AMU_TO_G, ANGSTROM3_TO_CM3

RendererName = Literal["tachyon", "visrtx", "anari", "ospray", "opengl"]
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FORMULA_SUBSCRIPTS = (
    ("Al2O3", "Al<sub>2</sub>O<sub>3</sub>"),
    ("CH3OH", "CH<sub>3</sub>OH"),
    ("TiO2", "TiO<sub>2</sub>"),
    ("ZrO2", "ZrO<sub>2</sub>"),
    ("SiO2", "SiO<sub>2</sub>"),
    ("CO2", "CO<sub>2</sub>"),
    ("H2O", "H<sub>2</sub>O"),
    ("NH3", "NH<sub>3</sub>"),
    ("OH2", "OH<sub>2</sub>"),
    ("C2H2", "C<sub>2</sub>H<sub>2</sub>"),
    ("N2", "N<sub>2</sub>"),
    ("O2", "O<sub>2</sub>"),
    ("H2", "H<sub>2</sub>"),
)
_FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k, _ in _FORMULA_SUBSCRIPTS) + r")(?![A-Za-z0-9])"
)
_FORMULA_HTML = dict(_FORMULA_SUBSCRIPTS)


def subscript_formula_html(text: object, *, escape_text: bool = True) -> str:
    """Render common chemical formulas with HTML subscripts for notebook output."""
    from html import escape

    rendered = escape(str(text)) if escape_text else str(text)
    return _FORMULA_RE.sub(lambda match: _FORMULA_HTML[match.group(1)], rendered)


def subscript_formula_markdown(text: object) -> str:
    """Render common formulas in Markdown while preserving code spans/fences."""
    rendered = str(text)
    parts = re.split(r"(```.*?```|`[^`]*`)", rendered, flags=re.DOTALL)
    return "".join(
        part if part.startswith("`") else subscript_formula_html(part, escape_text=False)
        for part in parts
    )


def _allow_artifact_overwrite() -> bool:
    return os.environ.get("ALCHEMI_ALLOW_ARTIFACT_OVERWRITE", "").strip().lower() in _TRUE_VALUES


def _clean_atoms_for_ovito(atoms: ase.Atoms, *, wrap_periodic_cell: bool = False) -> ase.Atoms:
    """Strip arrays with dtypes unsupported by OVITO (e.g. Unicode strings
    added by pymatgen's SlabGenerator) to avoid conversion errors."""
    _supported = {"int8", "int16", "int32", "int64", "float32", "float64"}
    clean = atoms.copy()
    for key in list(clean.arrays):
        if key in ("numbers", "positions"):
            continue
        if clean.arrays[key].dtype.name not in _supported:
            del clean.arrays[key]
    if wrap_periodic_cell and clean.cell.rank:
        # Pymatgen-generated slabs can carry valid periodic cells while atom
        # positions sit in neighboring periodic images. OVITO draws the cell at
        # the cell origin, so wrap a display-only copy before showing the
        # wireframe. The simulation structure passed by callers is unchanged.
        clean.wrap(eps=1e-7)
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

    clean = _clean_atoms_for_ovito(atoms, wrap_periodic_cell=show_cell)

    data = ase_to_ovito(clean)
    if not show_cell and data.cell_ is not None:
        data.cell_.vis.enabled = False

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all(size=size)

    ovito_renderer = _make_renderer(renderer, samples_per_pixel=samples_per_pixel)
    path = Path(output_path)
    if path.exists() and not _allow_artifact_overwrite():
        return str(path)
    if "outputs/precomputed" in path.as_posix() and not _allow_artifact_overwrite():
        raise FileExistsError(
            f"Refusing to create official saved render without refresh enabled: {path}. "
            "Use a live-run output path or set REFRESH_SAVED_RESULTS = True."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        vp.render_image(
            filename=str(path),
            size=size,
            renderer=ovito_renderer,
            background=background,
        )
    finally:
        pipeline.remove_from_scene()
    return str(path)


def create_interactive_view(
    atoms: ase.Atoms,
    width: str = "600px",
    height: str = "400px",
    particle_colors=None,
    show_cell: bool = False,
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
        If True, show the simulation-cell wireframe.  The default hides it
        because slab/vacuum cells are often visually offset from the atoms in
        compact notebook widgets.

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

    clean = _clean_atoms_for_ovito(atoms, wrap_periodic_cell=show_cell)
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


def create_trajectory_view(
    trajectory,
    width: str = "300px",
    height: str = "260px",
    show_cell: bool = False,
    frame_interval_ms: int = 160,
):
    """Create a slider/play widget for an ASE-readable trajectory.

    ``trajectory`` may be a path or a sequence of ``ase.Atoms`` frames. The
    returned widget displays one OVITO view at a time and updates it when the
    frame slider changes.
    """
    if isinstance(trajectory, (str, Path)):
        try:
            from ase.io import read as ase_read
            trajectory = ase_read(trajectory, ":")
        except Exception:
            return None

    try:
        import ipywidgets
        from IPython.display import display
    except ImportError:
        return None

    frames = list(trajectory)
    if not frames:
        return None

    output = ipywidgets.Output(layout=ipywidgets.Layout(width=width, height=height))
    widget_cache = [None] * len(frames)

    controls = _trajectory_controls(
        num_frames=len(frames),
        frame_interval_ms=frame_interval_ms,
        width=_expanded_px_width(width, min_px=360),
    )

    def _render_frame(change=None) -> None:
        frame = int(controls["slider"].value)
        with output:
            output.clear_output(wait=True)
            if widget_cache[frame] is None:
                widget_cache[frame] = create_interactive_view(
                    frames[frame],
                    width=width,
                    height=height,
                    show_cell=show_cell,
                )
            widget = widget_cache[frame]
            if widget is None:
                print(f"frame {frame + 1}/{len(frames)}: {len(frames[frame])} atoms")
            else:
                display(widget)

    controls["slider"].observe(_render_frame, names="value")
    _render_frame()
    return ipywidgets.VBox(
        [
            output,
            controls["container"],
        ],
        layout=ipywidgets.Layout(width=_expanded_px_width(width, min_px=360)),
    )


def _expanded_px_width(width: str, *, min_px: int) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*px\s*", str(width))
    if not match:
        return width
    return f"{max(int(match.group(1)), min_px)}px"


def _paired_px_width(width: str, *, min_px: int, gap_px: int = 16) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*px\s*", str(width))
    if not match:
        return width
    return f"{max(int(match.group(1)) * 2 + gap_px, min_px)}px"


def _grid_px_width(width: str, columns: int, *, gap_px: int = 15) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*px\s*", str(width))
    if not match:
        return "100%"
    columns = max(1, int(columns))
    return f"{int(match.group(1)) * columns + gap_px * (columns - 1)}px"


def _trajectory_controls(
    *,
    num_frames: int,
    frame_interval_ms: int,
    width: str,
):
    import ipywidgets
    import ovito

    slider = ipywidgets.IntSlider(
        value=0,
        min=0,
        max=max(int(num_frames) - 1, 0),
        step=1,
        description="",
        continuous_update=True,
        readout=True,
        layout=ipywidgets.Layout(flex="1 1 auto", min_width="260px"),
        style={"description_width": "0px"},
    )
    play = ipywidgets.Play(
        value=0,
        min=0,
        max=max(int(num_frames) - 1, 0),
        step=1,
        interval=frame_interval_ms,
        disabled=num_frames < 2,
        layout=ipywidgets.Layout(width="54px", flex="0 0 54px"),
    )
    frame_label = ipywidgets.HTML(
        value=f"<span style='white-space:nowrap'>1 / {num_frames}</span>",
        layout=ipywidgets.Layout(width="72px", flex="0 0 72px"),
    )
    ipywidgets.jslink((play, "value"), (slider, "value"))

    def _set_frame(change=None) -> None:
        frame = int(slider.value)
        ovito.dataset.anim.current_frame = frame
        frame_label.value = f"<span style='white-space:nowrap'>{frame + 1} / {num_frames}</span>"

    slider.observe(_set_frame, names="value")
    _set_frame()
    container = ipywidgets.HBox(
        [play, slider, frame_label],
        layout=ipywidgets.Layout(
            width=width,
            min_width="340px",
            align_items="center",
            gap="6px",
        ),
    )

    def _wrap(widget):
        return ipywidgets.VBox(
            [widget, container],
            layout=ipywidgets.Layout(width=width, min_width="340px"),
        )

    return {"container": container, "slider": slider, "wrap": _wrap}


def _create_trajectory_pipeline_widget(
    trajectory,
    *,
    width: str,
    height: str,
    show_cell: bool,
):
    try:
        import ipywidgets
        from ovito.gui import create_ipywidget
        from ovito.io import import_file
        from ovito.modifiers import PythonModifier
    except ImportError:
        return None

    pipeline = import_file(str(trajectory))
    if not show_cell:
        def _hide_cell(frame, data):
            if data.cell_ is not None:
                data.cell_.vis.enabled = False

        pipeline.modifiers.append(PythonModifier(function=_hide_cell))
    widget = create_ipywidget(
        pipeline,
        layout=ipywidgets.Layout(width=width, height=height),
    )
    return widget, int(pipeline.source.num_frames)


def _trajectory_widget_only(
    trajectory,
    *,
    width: str,
    height: str,
    show_cell: bool,
):
    if isinstance(trajectory, (str, Path)):
        return _create_trajectory_pipeline_widget(
            trajectory,
            width=width,
            height=height,
            show_cell=show_cell,
        )
    return None


def _atoms_from_widget_payload(payload) -> ase.Atoms:
    if isinstance(payload, ase.Atoms):
        return payload
    from ase.io import read as ase_read

    return ase_read(payload)


def _chunked(items: list, columns: int) -> list[list]:
    return [items[start:start + columns] for start in range(0, len(items), columns)]


def display_widgets_grid(
    rows: list[list[tuple[str, object]]],
    width: str = "220px",
    height: str = "200px",
    particle_colors_rows=None,
    show_cell: bool = False,
):
    """Display labelled OVITO widgets in a row/column grid."""
    try:
        from IPython.display import display
    except ImportError:
        for row in rows:
            print(" | ".join(str(label) for label, _payload in row))
        return

    display(
        _build_widgets_grid(
            rows,
            width=width,
            height=height,
            particle_colors_rows=particle_colors_rows,
            show_cell=show_cell,
        )
    )


def _build_widgets_grid(
    rows: list[list[tuple[str, object]]],
    *,
    width: str,
    height: str,
    particle_colors_rows=None,
    show_cell: bool,
):
    from ipywidgets import HBox, VBox, Layout
    from ipywidgets import HTML as HTMLWidget

    grid_rows = []
    for row_idx, row in enumerate(rows):
        widgets = []
        for col_idx, (label, payload) in enumerate(row):
            colors = None
            if particle_colors_rows:
                colors = particle_colors_rows[row_idx][col_idx]
            atoms = _atoms_from_widget_payload(payload)
            widget = create_interactive_view(
                atoms,
                width=width,
                height=height,
                particle_colors=colors,
                show_cell=show_cell,
            )
            if widget is None:
                widget = HTMLWidget(f"{len(atoms)} atoms")
            widgets.append(
                VBox(
                    [HTMLWidget(f"<b>{subscript_formula_html(label)}</b>"), widget],
                    layout=Layout(width=width, min_width=width, gap="4px"),
                )
            )
        grid_rows.append(
            HBox(
                widgets,
                layout=Layout(
                    justify_content="flex-start",
                    align_items="flex-start",
                    gap="15px",
                    width=_grid_px_width(width, len(row)),
                ),
            )
        )

    columns = max((len(row) for row in rows), default=1)
    return VBox(
        grid_rows,
        layout=Layout(
            align_items="flex-start",
            gap="15px",
            width=_grid_px_width(width, columns),
        ),
    )


def display_trajectory_widgets_grid(
    rows: list[list[tuple[str, object]]],
    width: str = "260px",
    height: str = "220px",
    show_cell: bool = False,
    frame_interval_ms: int = 160,
):
    """Display trajectory widgets as a left-aligned grid with per-panel controls."""
    try:
        from ipywidgets import HBox, VBox, Layout
        from ipywidgets import HTML as HTMLWidget
        from IPython.display import display
    except ImportError:
        for row in rows:
            print(" | ".join(str(label) for label, _payload in row))
        return

    rows = [row for row in rows if row]
    if not rows:
        print("No trajectories to display.")
        return

    panel_width = _expanded_px_width(width, min_px=360)
    grid_width = _grid_px_width(panel_width, max(len(row) for row in rows))

    def _trajectory_panel(label: str, trajectory):
        widget = create_trajectory_view(
            trajectory,
            width=panel_width,
            height=height,
            show_cell=show_cell,
            frame_interval_ms=frame_interval_ms,
        )
        if widget is None:
            widget = HTMLWidget("trajectory widget unavailable")
        return VBox(
            [
                HTMLWidget(f"<b>{subscript_formula_html(label)}</b>"),
                widget,
            ],
            layout=Layout(width=panel_width, min_width=panel_width, gap="4px"),
        )

    grid_rows = []
    for row in rows:
        grid_rows.append(
            HBox(
                [_trajectory_panel(label, trajectory) for label, trajectory in row],
                layout=Layout(
                    justify_content="flex-start",
                    align_items="flex-start",
                    gap="15px",
                    width=_grid_px_width(panel_width, len(row)),
                ),
            )
        )

    display(
        VBox(
            grid_rows,
            layout=Layout(
                align_items="flex-start",
                gap="15px",
                width=grid_width,
            ),
        )
    )


def display_paged_widgets_grid(
    items: list[tuple[str, object]],
    width: str = "180px",
    height: str = "160px",
    columns: int = 4,
    page_size: int = 12,
    show_cell: bool = False,
):
    """Display a larger widget collection as static left-aligned sections."""
    try:
        import ipywidgets
        from IPython.display import display
    except ImportError:
        for label, _payload in items:
            print(label)
        return

    items = list(items)
    if not items:
        print("No structures to display.")
        return

    page_size = max(1, int(page_size))
    columns = max(1, int(columns))
    grid_width = _grid_px_width(width, columns)
    sections = []
    for start in range(0, len(items), page_size):
        stop = min(start + page_size, len(items))
        sections.append(
            ipywidgets.VBox(
                [
                    ipywidgets.HTML(
                        f"<b>{start + 1}-{stop}</b> of <b>{len(items)}</b> structures"
                    ),
                    _build_widgets_grid(
                        _chunked(items[start:stop], columns),
                        width=width,
                        height=height,
                        show_cell=show_cell,
                    ),
                ],
                layout=ipywidgets.Layout(
                    align_items="flex-start",
                    gap="6px",
                    width=grid_width,
                ),
            )
        )

    display(
        ipywidgets.VBox(
            sections,
            layout=ipywidgets.Layout(
                align_items="flex-start",
                gap="18px",
                width=grid_width,
            ),
        )
    )


def display_widgets_row(
    items: list[tuple[str, ase.Atoms]],
    width: str = "300px",
    height: str = "300px",
    particle_colors_list=None,
    show_cell: bool = False,
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
        If True, show the simulation-cell wireframe.  Hidden by default because
        it is often misleading for slab/vacuum structures.
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
        atoms = _atoms_from_widget_payload(atoms)
        pc = particle_colors_list[idx] if particle_colors_list else None
        w = create_interactive_view(
            atoms, width=width, height=height,
            particle_colors=pc, show_cell=show_cell,
        )
        if w is not None:
            widgets.append(VBox([HTMLWidget(f"<b>{subscript_formula_html(label)}</b>"), w]))

    if widgets:
        display(
            HBox(
                widgets,
                layout=Layout(
                    justify_content="flex-start",
                    align_items="flex-start",
                    gap="15px",
                    width=_grid_px_width(width, len(widgets)),
                ),
            )
        )
    else:
        for label, atoms in items:
            print(f"{label}: {len(atoms)} atoms (widget unavailable)")


def display_inline(image_path: str):
    """Display a PNG image inline in a Jupyter notebook."""
    from IPython.display import Image, display

    display(Image(filename=image_path))


def format_elapsed(seconds: float) -> str:
    """Format elapsed wall time for compact notebook progress displays."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)} min {rem:04.1f} s"


def notebook_progress_html(
    *,
    title: str,
    done: int,
    total: int,
    message: str,
    elapsed_s: float,
    unit: str,
    average_label: str | None = None,
    width_px: int = 560,
):
    """Return a small HTML progress strip for VS Code and Jupyter notebooks.

    The caller owns the progress count.  This helper only renders the given
    ``done / total`` state, so counts advance only after the notebook cell has
    actually completed the corresponding batch or stage.
    """
    from IPython.display import HTML

    return HTML(_notebook_progress_html_string(
        title=title,
        done=done,
        total=total,
        message=message,
        elapsed_s=elapsed_s,
        unit=unit,
        average_label=average_label,
        width_px=width_px,
    ))


def _notebook_progress_html_string(
    *,
    title: str,
    done: int,
    total: int,
    message: str,
    elapsed_s: float,
    unit: str,
    average_label: str | None = None,
    width_px: int = 560,
) -> str:
    """Return the raw HTML string used by both widget and static fallbacks."""
    percent = 100 if total == 0 else max(0, min(100, int(round(100 * done / total))))
    avg = ""
    if done and average_label:
        avg = f" &nbsp;|&nbsp; avg {elapsed_s / done:.2f} {subscript_formula_html(average_label)}"
    return (
        f"""
        <style>
          html, body, .jp-RenderedHTMLCommon, .output_html {{
            background: transparent !important;
            margin: 0 !important;
            padding: 0 !important;
          }}
        </style>
        <div style="display:inline-flex; width:fit-content; max-width:100%; background:transparent !important; padding:0; margin:2px 0;">
          <div style="width:{int(width_px)}px; max-width:100%; background:#000; color:#f3f5f7; padding:12px 14px; border:1px solid #26313b; border-radius:10px; font-family:system-ui,-apple-system,Segoe UI,sans-serif; overflow:hidden; box-sizing:border-box;">
            <div style="display:flex; justify-content:space-between; gap:16px; font-size:13px; margin-bottom:10px;">
              <strong>{subscript_formula_html(title)}</strong>
              <span style="color:#a8b0b8;">{done}/{total} {subscript_formula_html(unit)} &nbsp;|&nbsp; elapsed {format_elapsed(elapsed_s)}{avg}</span>
            </div>
            <div style="height:8px; width:100%; background:#1f2933; border-radius:999px; overflow:hidden;">
              <div style="height:8px; width:{percent}%; background:#76b900; border-radius:999px;"></div>
            </div>
            <div style="color:#a8b0b8; font-size:12px; margin-top:10px; white-space:normal;">{subscript_formula_html(message)}</div>
          </div>
        </div>
        """
    )


class NotebookProgress:
    """Small display-handle wrapper for deterministic notebook progress updates."""

    def __init__(
        self,
        *,
        title: str,
        total: int,
        unit: str,
        message: str = "ready",
        average_label: str | None = None,
        width_px: int = 560,
    ) -> None:
        import time
        from IPython.display import display

        self.title = title
        self.total = int(total)
        self.unit = unit
        self.average_label = average_label
        self.width_px = int(width_px)
        self.started = time.perf_counter()
        self._widget = None
        self._display = None
        try:
            import ipywidgets as widgets

            self._widget = widgets.HTML(
                value=self.render_string(done=0, message=message),
                layout=widgets.Layout(
                    width=f"{self.width_px}px",
                    max_width="100%",
                    padding="0",
                    margin="0",
                ),
            )
            display(self._widget)
        except Exception:
            self._display = display(
                self.render(done=0, message=message),
                display_id=True,
            )

    def elapsed(self) -> float:
        """Return elapsed wall time since this progress display was created."""
        import time

        return time.perf_counter() - self.started

    def render(self, *, done: int, message: str):
        """Render the current progress state without mutating the display."""
        return notebook_progress_html(
            title=self.title,
            done=int(done),
            total=self.total,
            message=message,
            elapsed_s=self.elapsed(),
            unit=self.unit,
            average_label=self.average_label,
            width_px=self.width_px,
        )

    def render_string(self, *, done: int, message: str) -> str:
        """Render the current progress state as a widget-updatable HTML string."""
        return _notebook_progress_html_string(
            title=self.title,
            done=int(done),
            total=self.total,
            message=message,
            elapsed_s=self.elapsed(),
            unit=self.unit,
            average_label=self.average_label,
            width_px=self.width_px,
        )

    def update(self, *, done: int, message: str) -> None:
        """Update the progress display.

        Widget updates are used when available because VS Code and Jupyter
        refresh comm-backed widget state more reliably than display-id HTML
        replacement during a running cell.
        """
        from IPython.display import display

        if self._widget is not None:
            self._widget.value = self.render_string(done=done, message=message)
            return

        html = self.render(done=done, message=message)
        try:
            self._display.update(html)
        except Exception:
            display(html)


def make_notebook_progress(
    *,
    title: str,
    total: int,
    unit: str,
    message: str = "ready",
    average_label: str | None = None,
    width_px: int = 560,
) -> NotebookProgress:
    """Create a deterministic HTML progress strip for a notebook cell."""
    return NotebookProgress(
        title=title,
        total=total,
        unit=unit,
        message=message,
        average_label=average_label,
        width_px=width_px,
    )


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
