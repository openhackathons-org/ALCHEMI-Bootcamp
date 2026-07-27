"""Visualisation utilities using OVITO Python API and matplotlib."""

import csv
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ase
import ase.data
import pandas as pd

from .constants import AMU_TO_G, ANGSTROM3_TO_CM3

RendererName = Literal["tachyon", "visrtx", "anari", "ospray", "opengl"]
TrajectoryVideoRendererName = Literal["opengl", "visrtx", "anari"]
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
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(k) for k, _ in _FORMULA_SUBSCRIPTS)
    + r")(?![A-Za-z0-9])"
)
_FORMULA_HTML = dict(_FORMULA_SUBSCRIPTS)


@dataclass(frozen=True)
class TrajectoryRenderPair:
    """A matched DFT/Toolkit trajectory pair for video rendering."""

    label: str
    dft_path: Path
    mace_path: Path


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
        part
        if part.startswith("`")
        else subscript_formula_html(part, escape_text=False)
        for part in parts
    )


def _allow_artifact_overwrite() -> bool:
    return (
        os.environ.get("ALCHEMI_ALLOW_ARTIFACT_OVERWRITE", "").strip().lower()
        in _TRUE_VALUES
    )


def _clean_atoms_for_ovito(
    atoms: ase.Atoms,
    wrap_periodic_cell: bool,
) -> ase.Atoms:
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
        clean.wrap()
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


def _apply_particle_colors(data, particle_colors) -> None:
    """Apply explicit particle colours and mirror uniform colours onto OVITO types."""
    import numpy as np

    colors = np.asarray(particle_colors, dtype=np.float64)
    if colors.shape != (data.particles.count, 3):
        raise ValueError(
            "particle_colors must have shape (len(atoms), 3); "
            f"got {colors.shape} for {data.particles.count} particles."
        )

    if data.particles is None:
        return
    data.particles_.create_property("Color", data=colors)

    try:
        type_property = data.particles["Particle Type"]
    except KeyError:
        return

    type_ids = np.asarray(type_property, dtype=int)
    for particle_type in type_property.types:
        mask = type_ids == int(particle_type.id)
        if not bool(np.any(mask)):
            continue
        type_colors = colors[mask]
        if np.allclose(type_colors, type_colors[0], atol=1e-8):
            particle_type.color = tuple(float(channel) for channel in type_colors[0])


def _clear_ovito_scene() -> None:
    """Remove stale OVITO scene pipelines before static renders."""
    try:
        from ovito import scene
    except Exception:
        return

    for existing in list(scene.pipelines):
        try:
            existing.remove_from_scene()
        except Exception:
            pass


def _set_custom_camera(vp, camera: dict) -> None:
    vp.camera_pos = tuple(float(x) for x in camera["camera_pos"])
    vp.camera_dir = tuple(float(x) for x in camera["camera_dir"])
    if "fov" in camera:
        vp.fov = float(camera["fov"])


def render_structure_ovito(
    atoms: ase.Atoms,
    output_path: str = "structure.png",
    size: tuple[int, int] = (800, 600),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    renderer: RendererName = "tachyon",
    samples_per_pixel: int = 64,
    show_cell: bool = True,
    particle_colors=None,
    camera: dict | None = None,
    isolate_scene: bool = True,
    wrap_periodic_cell: bool | None = None,
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
    particle_colors : array-like, optional
        Per-particle RGB colours in [0, 1]. Use this when a rendered comparison
        needs unambiguous atom identity.
    camera : dict, optional
        Explicit OVITO camera settings with camera_pos, camera_dir, and
        optionally fov. When provided, zoom_all is not called.
    isolate_scene : bool
        If True, clear stale OVITO scene pipelines before rendering this image.
    wrap_periodic_cell : bool or None
        If None, wrap display positions before showing a periodic cell. Set
        False for structures that were already unwrapped for visual inspection.

    Returns the path to the rendered image.
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport

    if wrap_periodic_cell is None:
        wrap_periodic_cell = show_cell
    clean = _clean_atoms_for_ovito(atoms, wrap_periodic_cell=bool(wrap_periodic_cell))

    data = ase_to_ovito(clean)
    if particle_colors is not None:
        _apply_particle_colors(data, particle_colors)
    _style_ovito_data(data, show_cell=show_cell)

    if isolate_scene:
        _clear_ovito_scene()

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)
    if camera is None:
        _set_surface_focused_camera(vp, clean, size=size)
        vp.zoom_all(size=size)
    else:
        _set_custom_camera(vp, camera)

    ovito_renderer = _make_renderer(renderer, samples_per_pixel=samples_per_pixel)
    path = Path(output_path)
    if "outputs/precomputed" in path.as_posix() and not _allow_artifact_overwrite():
        if path.exists():
            return str(path)
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
        if isolate_scene:
            _clear_ovito_scene()
    return str(path)


def _set_surface_focused_camera(vp, atoms: ase.Atoms, *, size: tuple[int, int]) -> None:
    """Use an oblique camera aimed at the upper adsorption region when possible."""
    import numpy as np

    positions = np.asarray(atoms.positions, dtype=float)
    if positions.size == 0:
        return
    if atoms.cell.rank < 2:
        target = positions.mean(axis=0)
        span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        camera_dir = np.array([-0.48, -0.36, -0.80], dtype=float)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        distance = max(span * 2.8, 7.5)
        vp.camera_dir = tuple(camera_dir)
        vp.camera_pos = tuple(target - camera_dir * distance)
        vp.fov = math.radians(27.0 if size[0] >= size[1] else 32.0)
        return

    cell = np.asarray(atoms.cell.array, dtype=float)
    normal = np.cross(cell[0], cell[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return
    normal = normal / norm
    if np.dot(normal, positions.mean(axis=0) - positions.min(axis=0)) < 0:
        normal = -normal

    z_along_normal = positions @ normal
    top_cut = np.quantile(z_along_normal, 0.72)
    focus_positions = positions[z_along_normal >= top_cut]
    if len(focus_positions) == 0:
        focus_positions = positions
    target = focus_positions.mean(axis=0)

    lateral = cell[0]
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm > 1e-8:
        lateral = lateral / lateral_norm
    else:
        lateral = np.array([1.0, 0.0, 0.0])

    lateral_2 = np.cross(normal, lateral)
    lateral_2_norm = np.linalg.norm(lateral_2)
    if lateral_2_norm > 1e-8:
        lateral_2 = lateral_2 / lateral_2_norm
    else:
        lateral_2 = np.array([0.0, 1.0, 0.0])

    camera_dir = -(1.05 * normal + 0.52 * lateral + 0.20 * lateral_2)
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    distance = max(span * 1.55, 18.0)
    vp.camera_dir = tuple(camera_dir)
    vp.camera_pos = tuple(target - camera_dir * distance)
    vp.fov = math.radians(36.0 if size[0] >= size[1] else 42.0)


def _style_ovito_data(data, *, show_cell: bool) -> None:
    """Apply presentation defaults shared by static renders and widgets."""
    if data.cell_ is not None:
        if not show_cell:
            data.cell_.vis.enabled = False
            if hasattr(data.cell_.vis, "render_cell"):
                data.cell_.vis.render_cell = False
        else:
            data.cell_.vis.rendering_color = (0.62, 0.72, 0.82)
            data.cell_.vis.line_width = 0.16
            if hasattr(data.cell_.vis, "render_cell"):
                data.cell_.vis.render_cell = True
    if data.particles is not None:
        data.particles.vis.radius = max(
            float(getattr(data.particles.vis, "radius", 0.0)), 0.42
        )


def create_interactive_view(
    atoms: ase.Atoms,
    width: str = "600px",
    height: str = "400px",
    particle_colors=None,
    show_cell: bool = False,
    wrap_periodic_cell: bool = False,
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
    wrap_periodic_cell : bool or None
        If None, wrap display positions before showing a periodic cell. Set
        False for pre-unwrapped adsorbates so H atoms stay attached to NH3.

    Returns
    -------
    ipywidgets.DOMWidget or None
    """
    import ipywidgets
    from ovito.gui import create_ipywidget
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    clean = _clean_atoms_for_ovito(
        atoms,
        wrap_periodic_cell=wrap_periodic_cell,
    )

    data = ase_to_ovito(clean)

    if particle_colors is not None:
        _apply_particle_colors(data, particle_colors)

    _style_ovito_data(
        data,
        show_cell=show_cell,
    )

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
        rendered_widget = _trajectory_rendered_image_widget(
            trajectory,
            width=width,
            height=height,
            show_cell=show_cell,
            frame_interval_ms=frame_interval_ms,
        )
        if rendered_widget is not None:
            return rendered_widget

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
    on_frame_change=None,
):
    import ipywidgets
    import ovito

    slider = ipywidgets.IntSlider(
        value=0,
        min=0,
        max=max(int(num_frames) - 1, 0),
        step=1,
        description="",
        continuous_update=False,
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
    play_link = ipywidgets.link((play, "value"), (slider, "value"))

    def _set_frame(change=None) -> None:
        frame = int(slider.value)
        ovito.dataset.anim.current_frame = frame
        frame_label.value = (
            f"<span style='white-space:nowrap'>{frame + 1} / {num_frames}</span>"
        )
        if on_frame_change is not None:
            on_frame_change(frame)

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
    container._trajectory_play_link = play_link

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
            _style_ovito_data(data, show_cell=False)

        pipeline.modifiers.append(PythonModifier(function=_hide_cell))
    else:

        def _style_frame(frame, data):
            _style_ovito_data(data, show_cell=True)

        pipeline.modifiers.append(PythonModifier(function=_style_frame))
    viewport = _viewport_for_pipeline(pipeline, width=width, height=height)
    widget = create_ipywidget(
        viewport,
        layout=ipywidgets.Layout(width=width, height=height),
    )
    widget._ovito_pipeline = pipeline
    widget._ovito_viewport = viewport
    return widget, int(pipeline.source.num_frames)


def _viewport_for_pipeline(pipeline, *, width: str, height: str):
    """Create a zoomed OVITO viewport for a trajectory pipeline."""
    from ovito.vis import Viewport

    pipeline.add_to_scene()
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(0.7, -1.1, -0.55))
    size = (
        _css_px_int(width, default=900),
        _css_px_int(height, default=520),
    )
    if _set_pipeline_focused_camera(vp, pipeline, size=size):
        return vp
    try:
        vp.zoom_all(size=size)
    except Exception:
        vp.zoom_all()
    return vp


def _set_pipeline_focused_camera(vp, pipeline, *, size: tuple[int, int]) -> bool:
    """Aim the viewport at the adsorbate region instead of the full vacuum cell."""
    import numpy as np

    try:
        data = pipeline.compute(0)
        if data.particles is None or data.particles.count == 0:
            return False
        positions = np.asarray(data.particles["Position"], dtype=float)
        cell = (
            np.asarray(data.cell.matrix[:3, :3], dtype=float)
            if data.cell is not None
            else None
        )
        pbc = (
            tuple(bool(x) for x in getattr(data.cell, "pbc", (False, False, False)))
            if data.cell is not None
            else (False, False, False)
        )
    except Exception:
        return False
    if positions.size == 0:
        return False

    adsorbate_indices = _infer_tail_adsorbate_indices_from_numbers(
        _ovito_atomic_numbers(data)
    )
    if adsorbate_indices:
        positions = positions.copy()
        _unwrap_positions_in_place(positions, cell, pbc, adsorbate_indices)
        adsorbate_positions = positions[adsorbate_indices]
        adsorbate_focus = adsorbate_positions.mean(axis=0)
        local_mask = np.linalg.norm(positions - adsorbate_focus, axis=1) <= 7.5
        focus_positions = positions[local_mask]
        if len(focus_positions) < len(adsorbate_indices):
            focus_positions = adsorbate_positions
        target = adsorbate_focus
        span = float(
            np.linalg.norm(focus_positions.max(axis=0) - focus_positions.min(axis=0))
        )
    else:
        target = positions.mean(axis=0)
        span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))

    camera_dir = np.asarray((0.7, -1.1, -0.55), dtype=float)
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    distance = max(span * 2.25, 8.0)
    vp.camera_dir = tuple(camera_dir)
    vp.camera_pos = tuple(target - camera_dir * distance)
    vp.fov = math.radians(34.0 if size[0] >= size[1] else 42.0)
    return True


def _infer_tail_adsorbate_indices_from_numbers(numbers) -> list[int]:
    """Infer validation adsorbates written at the end of OC20Dense trajectories."""
    nums = [int(x) for x in numbers]
    known_tail_patterns = [
        [1, 1, 8],  # H2O in the validation pack
        [7, 1, 1, 1],  # NH3
        [7, 7],  # N2
        [6, 8],  # CO
        [6, 1, 1, 1, 8, 1],  # CH3OH
    ]
    for pattern in known_tail_patterns:
        if len(nums) >= len(pattern) and nums[-len(pattern) :] == pattern:
            return list(range(len(nums) - len(pattern), len(nums)))
    return []


def _ovito_atomic_numbers(data) -> list[int]:
    """Return atomic numbers from an OVITO frame, preserving particle order."""
    import numpy as np
    from ase.data import atomic_numbers

    values = np.asarray(data.particles["Particle Type"], dtype=int)
    try:
        particle_type_property = data.particles["Particle Type"]
        id_to_number = {}
        for particle_type in getattr(particle_type_property, "types", []):
            name = str(getattr(particle_type, "name", "")).strip()
            if name in atomic_numbers:
                id_to_number[int(particle_type.id)] = int(atomic_numbers[name])
        if id_to_number:
            return [id_to_number.get(int(value), int(value)) for value in values]
    except Exception:
        pass
    return [int(value) for value in values]


def _unwrap_positions_in_place(positions, cell, pbc, indices: list[int]) -> None:
    """Keep the adsorbate molecule connected across periodic boundaries."""
    import numpy as np

    if cell is None or not indices:
        return
    periodic = np.asarray(pbc, dtype=bool)
    if not periodic.any():
        return
    try:
        inv_cell_t = np.linalg.inv(np.asarray(cell, dtype=float).T)
    except np.linalg.LinAlgError:
        return
    anchor = positions[indices[0]].copy()
    for index in indices[1:]:
        delta = positions[index] - anchor
        frac = inv_cell_t @ delta
        frac[periodic] -= np.round(frac[periodic])
        positions[index] = anchor + np.asarray(cell, dtype=float).T @ frac


def _css_px_int(value: str, *, default: int) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*px\s*", str(value))
    return int(match.group(1)) if match else int(default)


def _trajectory_rendered_image_widget(
    trajectory,
    *,
    width: str,
    height: str,
    show_cell: bool,
    frame_interval_ms: int,
):
    try:
        import ipywidgets
        from ovito.io import import_file
        from ovito.modifiers import PythonModifier
    except ImportError:
        return None

    trajectory_path = Path(trajectory)
    pipeline = import_file(str(trajectory_path))

    def _style_frame(frame, data):
        if data.particles is not None:
            try:
                adsorbate_indices = _infer_tail_adsorbate_indices_from_numbers(
                    _ovito_atomic_numbers(data)
                )
                if adsorbate_indices:
                    positions = data.particles_.positions_
                    cell = data.cell.matrix[:3, :3] if data.cell is not None else None
                    pbc = (
                        getattr(data.cell, "pbc", (False, False, False))
                        if data.cell is not None
                        else (False, False, False)
                    )
                    _unwrap_positions_in_place(positions, cell, pbc, adsorbate_indices)
            except Exception:
                pass
        _style_ovito_data(data, show_cell=show_cell)

    pipeline.modifiers.append(PythonModifier(function=_style_frame))
    num_frames = int(pipeline.source.num_frames)
    pixel_width = _css_px_int(width, default=900)
    pixel_height = _css_px_int(height, default=520)
    viewport = _viewport_for_pipeline(pipeline, width=width, height=height)
    try:
        renderer = _make_renderer("opengl", samples_per_pixel=1)
    except Exception:
        from ovito.vis import TachyonRenderer

        renderer = TachyonRenderer()
    cache_dir = _trajectory_frame_cache_dir(trajectory_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image = ipywidgets.Image(
        format="png",
        layout=ipywidgets.Layout(
            width=width,
            height=height,
            object_fit="contain",
            border="1px solid #30363d",
        ),
    )
    status = ipywidgets.HTML(
        layout=ipywidgets.Layout(width=_expanded_px_width(width, min_px=360))
    )

    def _frame_png(frame: int) -> bytes:
        png_path = cache_dir / (
            f"style-v3-black-frame-{frame:04d}-{pixel_width}x{pixel_height}-cell{int(show_cell)}.png"
        )
        if not png_path.exists():
            try:
                viewport.render_image(
                    filename=str(png_path),
                    size=(pixel_width, pixel_height),
                    renderer=renderer,
                    background=(0.0, 0.0, 0.0),
                    frame=frame,
                )
            except TypeError:
                import ovito

                ovito.dataset.anim.current_frame = frame
                viewport.render_image(
                    filename=str(png_path),
                    size=(pixel_width, pixel_height),
                    renderer=renderer,
                    background=(0.0, 0.0, 0.0),
                )
        return png_path.read_bytes()

    def _render_frame(frame: int) -> None:
        try:
            image.value = _frame_png(frame)
            status.value = ""
        except Exception as exc:
            status.value = (
                f"<span style='color:#b00020'>OVITO render failed: {exc}</span>"
            )

    controls = _trajectory_controls(
        num_frames=num_frames,
        frame_interval_ms=frame_interval_ms,
        width=_expanded_px_width(width, min_px=360),
        on_frame_change=_render_frame,
    )
    _render_frame(0)
    widget = ipywidgets.VBox(
        [image, controls["container"], status],
        layout=ipywidgets.Layout(
            width=_expanded_px_width(width, min_px=360), gap="4px"
        ),
    )
    widget._ovito_pipeline = pipeline
    widget._ovito_viewport = viewport
    return widget


def _trajectory_frame_cache_dir(trajectory_path: Path) -> Path:
    safe_stem = (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", trajectory_path.stem).strip("_")
        or "trajectory"
    )
    digest = hashlib.sha1(str(trajectory_path.resolve()).encode("utf-8")).hexdigest()[
        :10
    ]
    return Path.cwd() / "outputs" / "ovito_widget_frame_cache" / f"{safe_stem}-{digest}"


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
    return [items[start : start + columns] for start in range(0, len(items), columns)]


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
    from ipywidgets import HTML as HTMLWidget
    from ipywidgets import HBox, Layout, VBox

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
    """Display trajectory widgets with one DFT/MACE pair active at a time.

    OVITO's stable trajectory path is file-backed: ``import_file()`` builds one
    multi-frame pipeline and the notebook widget displays that pipeline. Keeping
    only one validation pair live avoids overloading Jupyter front ends with
    many WebGL/Qt-backed widgets at once.
    """
    try:
        import ipywidgets
        from IPython.display import display
        from ipywidgets import HTML as HTMLWidget
        from ipywidgets import HBox, Layout, VBox
    except ImportError:
        for row in rows:
            print(" | ".join(str(label) for label, _payload in row))
        return

    rows = [row for row in rows if row]
    if not rows:
        print("No trajectories to display.")
        return

    panel_width = _expanded_px_width(width, min_px=360)
    row_width = _grid_px_width(panel_width, max(len(row) for row in rows))

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

    def _display_row(row):
        return HBox(
            [_trajectory_panel(label, trajectory) for label, trajectory in row],
            layout=Layout(
                justify_content="flex-start",
                align_items="flex-start",
                gap="15px",
                width=_grid_px_width(panel_width, len(row)),
            ),
        )

    if len(rows) == 1:
        display(_display_row(rows[0]))
        return

    options = []
    for index, row in enumerate(rows):
        first_label = row[0][0]
        label = first_label.replace("DFT trajectory | ", "")
        options.append((label, index))
    selector = ipywidgets.Dropdown(
        options=options,
        value=0,
        description="Trajectory",
        layout=Layout(width=row_width, min_width="520px"),
        style={"description_width": "80px"},
    )
    selected_panel = VBox(
        [],
        layout=Layout(width=row_width, min_width=row_width, align_items="flex-start"),
    )
    row_cache: dict[int, object] = {}

    def _render_selected(change=None):
        selected = int(selector.value)
        if selected not in row_cache:
            row_cache[selected] = _display_row(rows[selected])
        selected_panel.children = (row_cache[selected],)

    selector.observe(_render_selected, names="value")
    _render_selected()
    container = VBox(
        [selector, selected_panel],
        layout=Layout(align_items="flex-start", gap="8px", width=row_width),
    )
    display_trajectory_widgets_grid._last_container = container
    display(container)


def _trajectory_pairs_from_table(trajectory_paths) -> list[TrajectoryRenderPair]:
    rows = (
        trajectory_paths.to_dict("records")
        if hasattr(trajectory_paths, "to_dict")
        else trajectory_paths
    )
    pairs: list[TrajectoryRenderPair] = []
    for index, row in enumerate(rows):
        dft_path = row.get("dft_trajectory_widget_path") or row.get("dft_path")
        mace_path = row.get("mace_trajectory_widget_path") or row.get("mace_path")
        if not dft_path or not mace_path:
            continue
        label_parts = [
            row.get("adsorbate") or row.get("adsorbate_label"),
            row.get("system_id") or row.get("oc20dense_system_id"),
            row.get("config_id"),
        ]
        label = " | ".join(str(part) for part in label_parts if part not in (None, ""))
        if not label:
            label = f"trajectory pair {index + 1}"
        pairs.append(
            TrajectoryRenderPair(
                label=label,
                dft_path=Path(dft_path),
                mace_path=Path(mace_path),
            )
        )
    if not pairs:
        raise FileNotFoundError(
            "No DFT/Toolkit trajectory pairs were found in the provided table."
        )
    return pairs


def _trajectory_frame_count(path: Path) -> int:
    from ovito.io import import_file

    pipeline = import_file(str(path))
    return int(pipeline.source.num_frames)


def _sampled_indices(frame_count: int, output_frames: int) -> list[int]:
    import numpy as np

    if frame_count <= 0:
        raise ValueError("Trajectory has no frames.")
    if output_frames <= 1:
        return [0]
    return [int(round(x)) for x in np.linspace(0, frame_count - 1, output_frames)]


def _read_sampled_atoms(path: Path, max_samples: int = 9) -> list[ase.Atoms]:
    from ase.io import read as ase_read

    frames = ase_read(path, ":")
    if len(frames) <= max_samples:
        return frames
    return [frames[index] for index in _sampled_indices(len(frames), max_samples)]


def _trajectory_with_whole_adsorbate(source_path: Path, output_path: Path) -> Path:
    """Write a render-only trajectory with tail adsorbates whole across PBC."""
    import numpy as np
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    frames = ase_read(source_path, ":")
    fixed_frames = []
    changed = False
    for atoms in frames:
        fixed = atoms.copy()
        indices = _infer_tail_adsorbate_indices_from_numbers(fixed.get_atomic_numbers())
        if indices:
            positions = fixed.positions.copy()
            before = positions.copy()
            _unwrap_positions_in_place(positions, fixed.cell.array, fixed.pbc, indices)
            if np.any(np.abs(positions - before) > 1e-12):
                changed = True
            fixed.positions = positions
        fixed_frames.append(fixed)

    if not changed:
        return source_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(output_path, fixed_frames, format="extxyz")
    return output_path


def _shared_trajectory_camera(dft_path: Path, mace_path: Path) -> dict[str, Any]:
    import numpy as np

    frames = _read_sampled_atoms(dft_path) + _read_sampled_atoms(mace_path)
    all_positions = []
    adsorbate_positions = []
    for atoms in frames:
        positions = atoms.positions.copy()
        indices = _infer_tail_adsorbate_indices_from_numbers(atoms.get_atomic_numbers())
        if indices:
            _unwrap_positions_in_place(positions, atoms.cell.array, atoms.pbc, indices)
            adsorbate_positions.append(positions[indices])
        all_positions.append(positions)
    positions = np.vstack(all_positions)
    if adsorbate_positions:
        adsorbate_stack = np.vstack(adsorbate_positions)
        focus_xy = adsorbate_stack[:, :2].mean(axis=0)
        local_mask = np.linalg.norm(positions[:, :2] - focus_xy, axis=1) <= 10.5
        local_positions = positions[local_mask]
        if len(local_positions) < max(12, len(adsorbate_stack)):
            local_mask = np.linalg.norm(positions[:, :2] - focus_xy, axis=1) <= 13.0
            local_positions = positions[local_mask]
        if len(local_positions) < len(adsorbate_stack):
            local_positions = positions
        focus_z = float(np.percentile(local_positions[:, 2], 82))
        focus = np.array([focus_xy[0], focus_xy[1], focus_z], dtype=float)
    else:
        local_positions = positions
        focus = positions.mean(axis=0)

    camera_dir = np.array([0.035, -0.055, -1.0], dtype=float)
    camera_dir /= np.linalg.norm(camera_dir)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=float)
    camera_up = camera_up - camera_dir * np.dot(camera_up, camera_dir)
    camera_up /= np.linalg.norm(camera_up)
    camera_right = np.cross(camera_dir, camera_up)
    camera_right /= np.linalg.norm(camera_right)

    # Orthographic framing is set by the projected extents, not raw x/y bounds.
    # This avoids clipping tilted/top-view panels when an adsorbate crosses a
    # periodic boundary and is unwrapped into a visually whole molecule.
    projected_right = local_positions @ camera_right
    projected_up = local_positions @ camera_up
    center_right = 0.5 * (projected_right.min() + projected_right.max())
    center_up = 0.5 * (projected_up.min() + projected_up.max())
    focus = (
        camera_right * center_right
        + camera_up * center_up
        + camera_dir * np.dot(focus, camera_dir)
    )
    span_x = max(float(projected_right.max() - projected_right.min()), 1.0)
    span_y = max(float(projected_up.max() - projected_up.min()), 1.0)
    aspect = 960 / 720
    ortho_height = max(span_y, span_x / aspect, 7.5) * 1.28
    camera_pos = focus - camera_dir * 45.0
    return {
        "projection": "ortho",
        "camera_pos": tuple(float(x) for x in camera_pos),
        "camera_dir": tuple(float(x) for x in camera_dir),
        "camera_up": tuple(float(x) for x in camera_up),
        "ortho_height": float(ortho_height),
    }


def _style_trajectory_pipeline(pipeline, *, show_cell: bool = True) -> None:
    from ovito.modifiers import PythonModifier

    def _style(_frame, data):
        if data.particles is not None:
            try:
                adsorbate_indices = _infer_tail_adsorbate_indices_from_numbers(
                    _ovito_atomic_numbers(data)
                )
                if adsorbate_indices:
                    positions = data.particles_.positions_
                    cell = data.cell.matrix[:3, :3] if data.cell is not None else None
                    pbc = (
                        getattr(data.cell, "pbc", (False, False, False))
                        if data.cell is not None
                        else (False, False, False)
                    )
                    _unwrap_positions_in_place(positions, cell, pbc, adsorbate_indices)
            except Exception:
                pass
        if data.cell_ is not None:
            data.cell_.vis.enabled = bool(show_cell)
            data.cell_.vis.rendering_color = (0.62, 0.72, 0.82)
            data.cell_.vis.line_width = 0.16
        if data.particles is not None:
            data.particles.vis.radius = max(
                float(getattr(data.particles.vis, "radius", 0.0)),
                0.44,
            )

    pipeline.modifiers.append(PythonModifier(function=_style))


def _frame_stride_for_target(frame_count: int, target_frames: int) -> int:
    if frame_count <= 0:
        raise ValueError("Trajectory has no frames.")
    if target_frames <= 0:
        return 1
    return max(1, math.ceil(frame_count / int(target_frames)))


def _add_trajectory_label_overlay(viewport, label: str) -> None:
    from ovito.qt_compat import QtCore
    from ovito.vis import TextLabelOverlay

    overlay = TextLabelOverlay(
        text=label,
        alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
        offset_x=0.02,
        offset_y=0.02,
        font_size=0.035,
        text_color=(1.0, 1.0, 1.0),
        outline_enabled=True,
        outline_color=(0.0, 0.0, 0.0),
    )
    viewport.overlays.append(overlay)


def _render_trajectory_mp4(
    *,
    trajectory_path: Path,
    output_path: Path,
    label: str,
    camera: dict[str, Any],
    renderer,
    size: tuple[int, int],
    fps: int,
    target_frames: int,
) -> dict[str, Any]:
    import time

    from ovito.io import import_file
    from ovito.vis import Viewport

    start_time = time.perf_counter()
    pipeline = import_file(str(trajectory_path))
    frame_count = int(pipeline.source.num_frames)
    every_nth = _frame_stride_for_target(frame_count, target_frames)
    _style_trajectory_pipeline(pipeline, show_cell=True)
    pipeline.add_to_scene()
    projection = str(camera.get("projection", "perspective")).lower()
    viewport = Viewport(
        type=Viewport.Type.Ortho if projection == "ortho" else Viewport.Type.Perspective
    )
    viewport.camera_pos = camera["camera_pos"]
    viewport.camera_dir = camera["camera_dir"]
    if projection == "ortho":
        viewport.camera_up = camera.get("camera_up", (0.0, 1.0, 0.0))
        viewport.fov = camera["ortho_height"]
    else:
        viewport.fov = camera["fov"]
    _add_trajectory_label_overlay(viewport, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    try:
        viewport.render_anim(
            filename=str(output_path),
            size=size,
            fps=int(fps),
            background=(0.0, 0.0, 0.0),
            renderer=renderer,
            range=(0, frame_count - 1),
            every_nth=every_nth,
        )
    finally:
        pipeline.remove_from_scene()
    elapsed_s = time.perf_counter() - start_time
    rendered_frames = ((frame_count - 1) // every_nth) + 1
    return {
        "path": output_path.as_posix(),
        "input_frames": frame_count,
        "every_nth": every_nth,
        "rendered_frames": rendered_frames,
        "render_elapsed_s": elapsed_s,
        "render_s_per_frame": elapsed_s / max(rendered_frames, 1),
    }


def trajectory_video_links_markdown(
    render_result: dict[str, Any],
    *,
    relpath_fn=None,
) -> str:
    """Return notebook-native Markdown links for rendered trajectory MP4 files."""

    def _path(path: str) -> str:
        return str(relpath_fn(path) if relpath_fn else path)

    lines = ["**MP4 files**"]
    for row in render_result.get("video_rows", []):
        label = subscript_formula_markdown(row.get("label", "trajectory"))
        dft_path = _path(row["dft_mp4_path"])
        mace_path = _path(row["mace_mp4_path"])
        lines.append(f"- {label}: [DFT MP4]({dft_path}) | [Toolkit MP4]({mace_path})")
    return "\n".join(lines)


def render_trajectory_video_grid(
    trajectory_paths,
    output_dir: str | Path,
    *,
    renderer: TrajectoryVideoRendererName = "opengl",
    samples_per_pixel: int = 1,
    frames: int = 48,
    fps: int = 12,
    width: int = 360,
    height: int = 240,
    limit: int | None = None,
    progress_factory=None,
) -> dict[str, Any]:
    """Render matched DFT/Toolkit trajectories directly to MP4 files.

    This is intended for a Linux workstation or server with a working OVITO
    renderer. OVITO writes the MP4 files through ``Viewport.render_anim``; the
    notebook displays the resulting DFT/Toolkit videos and timing summary.
    """
    pairs = _trajectory_pairs_from_table(trajectory_paths)
    if limit is not None:
        pairs = pairs[: int(limit)]
    if not pairs:
        raise FileNotFoundError("No trajectory pairs selected for rendering.")

    output_root = Path(output_dir)
    size = (int(width), int(height))
    ovito_renderer = _make_renderer(renderer, samples_per_pixel=int(samples_per_pixel))
    renderer_label = f"{renderer} (spp={int(samples_per_pixel)})"
    progress = None
    if progress_factory is not None:
        progress = progress_factory(
            title="OVITO trajectory MP4 render",
            total=2 * len(pairs),
            unit="videos",
            message="ready to render validation trajectories",
            average_label="s/video",
            width_px=760,
        )

    manifest_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    done = 0
    for pair_index, pair in enumerate(pairs):
        pair_key = f"pair_{pair_index + 1:02d}"
        pair_dir = output_root / pair_key
        dft_render_input = _trajectory_with_whole_adsorbate(
            pair.dft_path,
            pair_dir / "render_inputs" / "dft_whole_adsorbate.extxyz",
        )
        mace_render_input = _trajectory_with_whole_adsorbate(
            pair.mace_path,
            pair_dir / "render_inputs" / "toolkit_whole_adsorbate.extxyz",
        )
        camera = _shared_trajectory_camera(dft_render_input, mace_render_input)
        dft_mp4 = output_root / pair_key / "dft.mp4"
        mace_mp4 = output_root / pair_key / "toolkit.mp4"
        if progress is not None:
            progress.update(done=done, message=f"rendering DFT video: {pair.label}")
        dft_render = _render_trajectory_mp4(
            trajectory_path=dft_render_input,
            output_path=dft_mp4,
            label=f"DFT | {pair.label}",
            camera=camera,
            renderer=ovito_renderer,
            size=size,
            fps=int(fps),
            target_frames=int(frames),
        )
        done += 1
        if progress is not None:
            progress.update(done=done, message=f"finished DFT video: {pair.label}")
            progress.update(done=done, message=f"rendering Toolkit video: {pair.label}")
        mace_render = _render_trajectory_mp4(
            trajectory_path=mace_render_input,
            output_path=mace_mp4,
            label=f"Toolkit | {pair.label}",
            camera=camera,
            renderer=ovito_renderer,
            size=size,
            fps=int(fps),
            target_frames=int(frames),
        )
        done += 1
        if progress is not None:
            progress.update(done=done, message=f"finished Toolkit video: {pair.label}")
        video_rows.append(
            {
                "label": pair.label,
                "dft_mp4_path": dft_render["path"],
                "mace_mp4_path": mace_render["path"],
            }
        )
        manifest_rows.append(
            {
                "label": pair.label,
                "dft_path": pair.dft_path.as_posix(),
                "mace_path": pair.mace_path.as_posix(),
                "dft_mp4_path": dft_render["path"],
                "mace_mp4_path": mace_render["path"],
                "dft_input_frames": dft_render["input_frames"],
                "mace_input_frames": mace_render["input_frames"],
                "dft_every_nth": dft_render["every_nth"],
                "mace_every_nth": mace_render["every_nth"],
                "dft_rendered_frames": dft_render["rendered_frames"],
                "mace_rendered_frames": mace_render["rendered_frames"],
                "dft_render_elapsed_s": dft_render["render_elapsed_s"],
                "mace_render_elapsed_s": mace_render["render_elapsed_s"],
                "total_render_elapsed_s": dft_render["render_elapsed_s"]
                + mace_render["render_elapsed_s"],
                "dft_s_per_frame": dft_render["render_s_per_frame"],
                "mace_s_per_frame": mace_render["render_s_per_frame"],
                "renderer": renderer_label,
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "camera_policy": (
                    "DFT and Toolkit panels for each system share target, "
                    "distance, field of view, and near-top camera direction"
                ),
            }
        )

    manifest_path = output_root / "render_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "mp4_paths": [
            path
            for row in video_rows
            for path in (row["dft_mp4_path"], row["mace_mp4_path"])
        ],
        "video_rows": video_rows,
        "manifest_path": manifest_path.as_posix(),
        "manifest_rows": manifest_rows,
        "renderer": renderer_label,
    }


def _ffmpeg_executable(*, install_if_missing: bool) -> str:
    import shutil
    import subprocess
    import sys

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        if not install_if_missing:
            raise RuntimeError(
                "ffmpeg is required to make OVITO MP4s playable in Jupyter. "
                "Install ffmpeg or enable the imageio-ffmpeg fallback."
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"],
            check=True,
        )
        import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_browser_safe_mp4(
    source_path: str | Path,
    *,
    install_ffmpeg_if_missing: bool = True,
) -> tuple[str, float]:
    """Transcode an OVITO MP4 to browser-compatible H.264/yuv420p."""
    import subprocess
    import time

    source = Path(source_path)
    target = source.with_name(source.stem + "_browser.mp4")
    started = time.perf_counter()
    subprocess.run(
        [
            _ffmpeg_executable(install_if_missing=install_ffmpeg_if_missing),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
    return target.as_posix(), time.perf_counter() - started


def _trajectory_video_timing_table(
    render_result: dict[str, Any],
    *,
    browser_rows: list[dict[str, Any]],
    ovito_wall_s: float,
) -> pd.DataFrame:
    render_timing = pd.DataFrame(render_result["manifest_rows"])
    transcode_timing = pd.DataFrame(browser_rows)
    timing_table = render_timing.merge(transcode_timing, on="label", how="left")
    timing_columns = [
        "label",
        "dft_render_elapsed_s",
        "mace_render_elapsed_s",
        "total_render_elapsed_s",
        "dft_rendered_frames",
        "mace_rendered_frames",
        "dft_s_per_frame",
        "mace_s_per_frame",
        "dft_transcode_s",
        "toolkit_transcode_s",
        "renderer",
        "width",
        "height",
        "fps",
    ]
    missing_columns = [
        column for column in timing_columns if column not in timing_table.columns
    ]
    if missing_columns:
        timing_table["OVITO render batch wall (s)"] = ovito_wall_s
        timing_table["estimated render per video (s)"] = ovito_wall_s / max(
            len(render_result["mp4_paths"]),
            1,
        )
        return timing_table[
            [
                "label",
                "OVITO render batch wall (s)",
                "estimated render per video (s)",
                "dft_rendered_frames",
                "mace_rendered_frames",
                "dft_transcode_s",
                "toolkit_transcode_s",
                "renderer",
                "width",
                "height",
                "fps",
            ]
        ].rename(
            columns={
                "dft_rendered_frames": "DFT frames",
                "mace_rendered_frames": "Toolkit frames",
                "dft_transcode_s": "DFT transcode (s)",
                "toolkit_transcode_s": "Toolkit transcode (s)",
            }
        )
    return timing_table[timing_columns].rename(
        columns={
            "dft_render_elapsed_s": "DFT render (s)",
            "mace_render_elapsed_s": "Toolkit render (s)",
            "total_render_elapsed_s": "OVITO render total (s)",
            "dft_rendered_frames": "DFT frames",
            "mace_rendered_frames": "Toolkit frames",
            "dft_s_per_frame": "DFT s/frame",
            "mace_s_per_frame": "Toolkit s/frame",
            "dft_transcode_s": "DFT transcode (s)",
            "toolkit_transcode_s": "Toolkit transcode (s)",
        }
    )


def render_validation_trajectory_videos(
    trajectory_paths,
    output_dir: str | Path,
    *,
    renderer: TrajectoryVideoRendererName = "visrtx",
    samples_per_pixel: int = 32,
    frames: int = 480,
    fps: int = 24,
    width: int = 960,
    height: int = 720,
    install_ffmpeg_if_missing: bool = True,
    progress_factory=None,
) -> dict[str, Any]:
    """Render centered orthographic DFT/Toolkit trajectory MP4s for the notebook."""
    import time

    started = time.perf_counter()
    render_result = render_trajectory_video_grid(
        trajectory_paths,
        output_dir=output_dir,
        renderer=renderer,
        samples_per_pixel=samples_per_pixel,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        progress_factory=progress_factory,
    )
    ovito_wall_s = time.perf_counter() - started

    browser_rows: list[dict[str, Any]] = []
    for row in render_result["video_rows"]:
        dft_browser_mp4, dft_transcode_s = make_browser_safe_mp4(
            row["dft_mp4_path"],
            install_ffmpeg_if_missing=install_ffmpeg_if_missing,
        )
        mace_browser_mp4, mace_transcode_s = make_browser_safe_mp4(
            row["mace_mp4_path"],
            install_ffmpeg_if_missing=install_ffmpeg_if_missing,
        )
        row["dft_browser_mp4_path"] = dft_browser_mp4
        row["mace_browser_mp4_path"] = mace_browser_mp4
        browser_rows.append(
            {
                "label": row["label"],
                "dft_transcode_s": dft_transcode_s,
                "toolkit_transcode_s": mace_transcode_s,
            }
        )

    render_result["ovito_wall_s"] = ovito_wall_s
    render_result["timing_table"] = _trajectory_video_timing_table(
        render_result,
        browser_rows=browser_rows,
        ovito_wall_s=ovito_wall_s,
    )
    return render_result


def display_validation_trajectory_videos(
    render_result: dict[str, Any],
    *,
    output_dir: str | Path,
    relpath_fn=None,
    width: int = 960,
) -> None:
    """Display rendered validation trajectory videos and the compact timing table."""
    from IPython.display import Markdown, Video, display

    def _path(path: str | Path) -> str:
        return str(relpath_fn(path) if relpath_fn else path)

    display(
        Markdown(
            f"Rendered {len(render_result['mp4_paths'])} centered orthographic MP4 files under "
            f"`{_path(output_dir)}` and transcoded browser-safe copies. "
            f"OVITO render wall time: {render_result['ovito_wall_s']:.1f} s."
        )
    )
    display(render_result["timing_table"].round(2))
    for row in render_result["video_rows"]:
        label = row["label"]
        display(Markdown(f"#### {label} | DFT reference trajectory"))
        display(
            Video(
                filename=row["dft_browser_mp4_path"],
                embed=True,
                width=width,
                html_attributes="controls loop muted playsinline",
            )
        )
        display(Markdown(f"[Open DFT MP4]({_path(row['dft_browser_mp4_path'])})"))

        display(Markdown(f"#### {label} | Toolkit/MACE trajectory"))
        display(
            Video(
                filename=row["mace_browser_mp4_path"],
                embed=True,
                width=width,
                html_attributes="controls loop muted playsinline",
            )
        )
        display(Markdown(f"[Open Toolkit MP4]({_path(row['mace_browser_mp4_path'])})"))


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
    wrap_periodic_cell: bool = False,
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
    wrap_periodic_cell : bool or None
        Forwarded to create_interactive_view. Use False for display-unwrapped
        adsorbates that should keep their molecule whole across PBCs.
    """
    from IPython.display import display
    from ipywidgets import HTML as HTMLWidget
    from ipywidgets import HBox, Layout, VBox

    widgets = []
    for idx, (label, atoms) in enumerate(items):
        if not isinstance(atoms, ase.Atoms):
            from ase.io import read as ase_read

            atoms = ase_read(atoms)
            assert not isinstance(atoms, list)
        pc = particle_colors_list[idx] if particle_colors_list else None
        w = create_interactive_view(
            atoms,
            width=width,
            height=height,
            particle_colors=pc,
            show_cell=show_cell,
            wrap_periodic_cell=wrap_periodic_cell,
        )
        if w is not None:
            widgets.append(
                VBox([HTMLWidget(f"<b>{subscript_formula_html(label)}</b>"), w])
            )

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

    return HTML(
        _notebook_progress_html_string(
            title=title,
            done=done,
            total=total,
            message=message,
            elapsed_s=elapsed_s,
            unit=unit,
            average_label=average_label,
            width_px=width_px,
        )
    )


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
    return f"""
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

        from time import perf_counter

        from IPython.display import display

        self.title = title
        self.total = total
        self.unit = unit
        self.average_label = average_label
        self.width_px = width_px
        self.started = perf_counter()
        self._widget = None
        self._display = None
        try:
            import ipywidgets as widgets

            self._widget = widgets.HTML(
                value=self.render_string(
                    done=0,
                    message=message,
                ),
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

        from time import perf_counter

        return perf_counter() - self.started

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
            self._widget.value = self.render_string(
                done=done,
                message=message,
            )
            return

        html = self.render(done=done, message=message)
        try:
            self._display.update(html)
        except Exception:
            display(html)


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


# ---------------------------------------------------------------------------
# Notebook callouts and table styling
#
# These keep DataFrame chrome and HTML out of the notebook code cells so the
# Toolkit / ASE / pymatgen calls stay the visible content of each cell.
# ---------------------------------------------------------------------------

_CALLOUT_STYLES = {
    "info": ("#00a3e0", "#eef7fb", "\N{INFORMATION SOURCE}", "Note"),
    "api": ("#76b900", "#f0f7e6", "\N{JIGSAW PUZZLE PIECE}", "API"),
    "choice": ("#cc7a00", "#fdf6e3", "\N{GEAR}", "Choice"),
    "warning": ("#e8590c", "#fdeee6", "\N{WARNING SIGN}", "Heads up"),
}


def callout_html(body: str, *, kind: str = "info", title: str | None = None) -> str:
    """Return the HTML string for a styled notebook callout box.

    ``kind`` is one of ``info`` / ``api`` / ``choice`` / ``warning``. The same
    visual style is reused for the static Markdown callouts in the notebook so
    every admonition looks the same regardless of how it is rendered.
    """
    accent, background, icon, default_title = _CALLOUT_STYLES.get(
        kind, _CALLOUT_STYLES["info"]
    )
    heading = title if title is not None else default_title
    return (
        f'<div style="border-left:5px solid {accent}; background:{background}; '
        'padding:12px 16px; margin:12px 0; border-radius:6px; '
        'font-family:Inter, Arial, sans-serif;">'
        f'<div style="font-weight:700; color:{accent}; margin-bottom:5px; '
        'font-size:0.9em; letter-spacing:0.03em; text-transform:uppercase;">'
        f"{icon} {heading}</div>"
        f'<div style="color:#1f2937; font-size:0.95em; line-height:1.5;">{body}</div>'
        "</div>"
    )


def callout(body: str, *, kind: str = "info", title: str | None = None):
    """Display a styled callout box in a notebook code cell.

    Returns an ``IPython.display.HTML`` handle so it can be returned as a cell
    value or passed to ``display(...)``.
    """
    from IPython.display import HTML

    return HTML(callout_html(body, kind=kind, title=title))


def styled_table(df: pd.DataFrame, *, fmt: dict | None = None, hide_index: bool = True):
    """Return a pandas Styler with the notebook's standard table styling.

    Replaces the repeated ``df.style.hide(axis="index").format({...})`` chains
    inline in the notebook. ``fmt`` is the same column-format mapping that would
    otherwise be passed to ``Styler.format``.
    """
    styler = df.style
    if hide_index:
        styler = styler.hide(axis="index")
    if fmt:
        styler = styler.format(fmt)
    return styler


# Placement convention shared by the adsorbate preview grid and the orientation
# table: "down" always means the surface-facing end after placement along the
# slab normal, never the screen-down direction in the rotatable widget.
ORIENTATION_SURFACE_FACING_END = {
    "C-down": "C atom faces the surface",
    "O-down": "O atom faces the surface",
    "H-down": "H side faces the surface",
    "N-down": "N atom faces the surface",
    "methyl-down": "methyl group faces the surface",
}


def standalone_adsorbate_preview(atoms: ase.Atoms) -> ase.Atoms:
    """Return a molecule-only display copy with no artificial periodic box."""
    preview = atoms.copy()
    preview.set_cell([0.0, 0.0, 0.0])
    preview.set_pbc(False)
    preview.translate(-preview.get_center_of_mass())
    return preview


def display_adsorbate_preview_grid(
    items: list[tuple[str, ase.Atoms]],
    *,
    width: str = "500px",
    height: str = "220px",
    columns: int = 4,
) -> None:
    """Display labelled molecule-only OVITO previews in a grid.

    Each item is a ``(label, atoms)`` tuple; labels render with chemical
    subscripts. Falls back to a bold label when the widget is unavailable.
    """
    from IPython.display import display
    import ipywidgets as widgets

    cards = []
    for label, atoms in items:
        widget = create_interactive_view(
            atoms, width=width, height=height, show_cell=False
        )
        label_html = (
            "<div style='font-size: 200%; font-weight: 700; line-height: 1.15; "
            f"margin: 0 0 8px 0;'>{subscript_formula_html(label)}</div>"
        )
        if widget is None:
            cards.append(widgets.HTML(label_html))
            continue
        cards.append(
            widgets.VBox(
                [widgets.HTML(label_html), widget],
                layout=widgets.Layout(width=width, min_width=width, gap="4px"),
            )
        )
    display(
        widgets.GridBox(
            cards,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({int(columns)}, {width})",
                grid_gap="18px 20px",
                align_items="flex-start",
                width=f"calc({int(columns)} * {width} + {(int(columns) - 1) * 20}px)",
            ),
        )
    )


def show_orientation_convention(specs) -> pd.DataFrame:
    """Display the adsorbate orientation-convention banner and table.

    ``specs`` is any iterable of objects exposing ``.name`` and ``.orientations``
    (the notebook's ``SurfaceScreenAdsorbateSpec`` list). Returns the assembled
    convention DataFrame for optional downstream reuse.
    """
    from IPython.display import HTML, display

    orientation_convention_df = pd.DataFrame(
        [
            {
                "adsorbate": subscript_formula_html(spec.name),
                "orientation label": orient,
                "meaning of down": ORIENTATION_SURFACE_FACING_END.get(orient, orient),
            }
            for spec in specs
            for orient in spec.orientations
        ]
    )
    display(
        HTML(
            "<div style='font-size: 200%; line-height: 1.2; margin: 0 0 10px 0;'>"
            "<strong>Orientation convention:</strong> in labels such as "
            "<code>N-down</code>, <strong>down means the surface-facing end after "
            "placement along the slab normal</strong>. It does not mean lower on "
            "the notebook screen; the molecule preview widget can be rotated freely."
            "</div>"
        )
    )
    display(
        orientation_convention_df.style.format(escape=None)
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "200%"),
                        ("line-height", "1.15"),
                        ("padding", "8px 12px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "200%"),
                        ("line-height", "1.15"),
                        ("padding", "8px 12px"),
                    ],
                },
            ]
        )
    )
    return orientation_convention_df
