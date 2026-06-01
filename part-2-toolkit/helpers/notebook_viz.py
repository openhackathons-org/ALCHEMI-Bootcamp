"""OVITO interactive structure views and notebook progress for the Part 2 tutorial.

Ported from the Part 3 notebook's visualization helpers so the two tutorials share
the same look: labelled rows of interactive 3-D OVITO widgets
(:func:`display_widgets_row` / :func:`create_interactive_view`) and an
NVIDIA-green progress strip (:class:`NotebookProgress`).

OVITO and ipywidgets are imported lazily inside the functions, so importing this
module host-side (without those packages) is safe — only calling the renderers
needs them. Inside the container both are present.
"""

from __future__ import annotations

import re

import ase

# ── Chemical-formula subscripting for widget / progress labels ───────────────
_FORMULA_SUBSCRIPTS = (
    ("CH3OH", "CH<sub>3</sub>OH"),
    ("C10H8", "C<sub>10</sub>H<sub>8</sub>"),
    ("CO2", "CO<sub>2</sub>"),
    ("H2O", "H<sub>2</sub>O"),
    ("NH3", "NH<sub>3</sub>"),
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


def subscript_formula_html(text: object, *, escape_text: bool = True) -> str:
    """Render common chemical formulas with HTML subscripts for notebook output."""
    from html import escape

    rendered = escape(str(text)) if escape_text else str(text)
    return _FORMULA_RE.sub(lambda match: _FORMULA_HTML[match.group(1)], rendered)


# ── OVITO interactive views ──────────────────────────────────────────────────
def _clean_atoms_for_ovito(atoms: ase.Atoms, wrap_periodic_cell: bool) -> ase.Atoms:
    """Strip arrays with dtypes unsupported by OVITO (e.g. Unicode strings added
    by pymatgen's SlabGenerator) to avoid conversion errors."""
    _supported = {"int8", "int16", "int32", "int64", "float32", "float64"}
    clean = atoms.copy()
    for key in list(clean.arrays):
        if key in ("numbers", "positions"):
            continue
        if clean.arrays[key].dtype.name not in _supported:
            del clean.arrays[key]
    if wrap_periodic_cell and clean.cell.rank:
        # OVITO draws the cell at the origin, so wrap a display-only copy before
        # showing the wireframe. The structure passed by callers is unchanged.
        clean.wrap()
    return clean


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

    Parameters
    ----------
    atoms : ase.Atoms
    width, height : str
        CSS size strings for the widget layout.
    particle_colors : np.ndarray shape (N, 3) or None
        Per-particle RGB colours in [0, 1]; override OVITO's element colouring.
    show_cell : bool
        If True, show the simulation-cell wireframe.
    wrap_periodic_cell : bool
        Wrap display positions into the cell before showing the wireframe.

    Returns
    -------
    ipywidgets.DOMWidget
    """
    import ipywidgets
    from ovito.gui import create_ipywidget
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    clean = _clean_atoms_for_ovito(atoms, wrap_periodic_cell=wrap_periodic_cell)
    data = ase_to_ovito(clean)
    if particle_colors is not None:
        _apply_particle_colors(data, particle_colors)
    _style_ovito_data(data, show_cell=show_cell)
    pipeline = Pipeline(source=StaticSource(data=data))
    return create_ipywidget(
        pipeline, layout=ipywidgets.Layout(width=width, height=height)
    )


def _grid_px_width(width: str, columns: int, *, gap_px: int = 15) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*px\s*", str(width))
    if not match:
        return "100%"
    columns = max(1, int(columns))
    return f"{int(match.group(1)) * columns + gap_px * (columns - 1)}px"


def display_widgets_row(
    items: list[tuple[str, ase.Atoms]],
    width: str = "300px",
    height: str = "300px",
    particle_colors_list=None,
    show_cell: bool = False,
    wrap_periodic_cell: bool = False,
):
    """Display a horizontal row of labelled interactive OVITO widgets.

    Falls back to a one-line text summary if interactive widgets are unavailable.

    Parameters
    ----------
    items : list of (label, atoms) tuples
    width, height : str
        CSS size for each widget.
    particle_colors_list : list of np.ndarray or None
        Per-item particle colours (same length as *items*).
    show_cell : bool
        If True, show the simulation-cell wireframe.
    wrap_periodic_cell : bool
        Forwarded to :func:`create_interactive_view`.
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


# ── Deterministic notebook progress strip (NVIDIA-green) ─────────────────────
def format_elapsed(seconds: float) -> str:
    """Format elapsed wall time for compact notebook progress displays."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)} min {rem:04.1f} s"


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

    The caller owns the progress count; this renders the given ``done / total``
    state, so counts advance only after the cell has actually completed a stage.
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
                value=self.render_string(done=0, message=message),
                layout=widgets.Layout(
                    width=f"{self.width_px}px", max_width="100%", padding="0", margin="0"
                ),
            )
            display(self._widget)
        except Exception:
            self._display = display(self.render(done=0, message=message), display_id=True)

    def elapsed(self) -> float:
        """Elapsed wall time since this progress display was created."""
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
        """Update the progress display (widget value when available)."""
        from IPython.display import display

        if self._widget is not None:
            self._widget.value = self.render_string(done=done, message=message)
            return
        html = self.render(done=done, message=message)
        try:
            self._display.update(html)
        except Exception:
            display(html)


# ── OVITO trajectory animation (MP4) ─────────────────────────────────────────
# Ported from the Part 3 visualization helpers so the two tutorials render
# trajectory videos with an identical look (cell wireframe, particle radii,
# NVIDIA-VisRTX renderer, browser-safe transcode). The Part 3 adsorbate-unwrap
# PythonModifier is dropped here — Part 2 is a bulk molecular crystal with no
# tail adsorbate to reconstruct across the periodic boundary.
def _style_trajectory_pipeline(pipeline, *, show_cell: bool = True) -> None:
    from ovito.modifiers import PythonModifier

    def _style(_frame, data):
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


def _make_renderer(
    renderer: str,
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


def _frame_stride_for_target(frame_count: int, target_frames: int) -> int:
    import math

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
    source_path,
    *,
    install_ffmpeg_if_missing: bool = True,
):
    """Transcode an OVITO MP4 to browser-compatible H.264/yuv420p."""
    import subprocess
    import time
    from pathlib import Path

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


def render_trajectory_animation(
    frames,
    *,
    output_path,
    label: str | None = None,
    target_frames: int = 12,
    fps: int = 6,
    renderer: str = "tachyon",
    samples_per_pixel: int = 8,
    size: tuple[int, int] = (480, 360),
    show_cell: bool = True,
) -> dict:
    """Render an OVITO trajectory animation to a browser-safe MP4.

    Parameters
    ----------
    frames : list[ase.Atoms] | str | pathlib.Path
        Either an in-memory list of frames (written to an extxyz file next to
        ``output_path``) or a path to an existing extxyz/xyz trajectory.
    output_path : str | pathlib.Path
        Destination MP4. ``make_browser_safe_mp4`` writes a companion
        ``*_browser.mp4`` that plays inline in Jupyter.
    label : str or None
        Optional caption drawn in the top-left of the viewport.
    target_frames : int
        Approximate number of frames to render; the stride is derived from the
        trajectory length so longer runs stay snappy.
    fps : int
        Playback frame rate of the MP4.
    renderer : str
        OVITO renderer key (default ``"tachyon"``, a CPU ray tracer that needs
        no GPU/display and is the most reliable headless option). ``"visrtx"``
        (NVIDIA VisRTX/ANARI, GPU) is faster but its device initialisation can
        fail on some headless nodes — if it does, this function falls back to
        Tachyon automatically. ``"opengl"`` needs a display and is unsuitable
        for the container.
    samples_per_pixel : int
        Ray-tracing samples per pixel for the VisRTX/OSPRay paths.
    size : tuple[int, int]
        Output ``(width, height)`` in pixels.
    show_cell : bool
        If True, draw the simulation-cell wireframe.

    Returns
    -------
    dict
        Timing and path metadata: ``path``, ``input_frames``, ``every_nth``,
        ``rendered_frames``, ``render_elapsed_s``, ``render_s_per_frame``, plus
        ``browser_path`` and ``transcode_elapsed_s`` from the H.264 transcode.
    """
    import time
    from pathlib import Path

    from ovito.io import import_file
    from ovito.vis import Viewport

    output_path = Path(output_path)
    # Front door: accept an in-memory frame list or an on-disk trajectory.
    if isinstance(frames, (str, Path)):
        trajectory_path = Path(frames)
    else:
        from ase.io import write

        trajectory_path = output_path.with_suffix(".traj.extxyz")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(trajectory_path), list(frames), format="extxyz")

    start_time = time.perf_counter()
    pipeline = import_file(str(trajectory_path))
    frame_count = int(pipeline.source.num_frames)
    every_nth = _frame_stride_for_target(frame_count, target_frames)
    _style_trajectory_pipeline(pipeline, show_cell=show_cell)
    renderer_obj = _make_renderer(renderer, samples_per_pixel)
    pipeline.add_to_scene()
    # Part 2 has no per-system camera table — let OVITO frame the scene.
    viewport = Viewport(type=Viewport.Type.Perspective)
    viewport.zoom_all(size=size)
    if label:
        _add_trajectory_label_overlay(viewport, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    def _render_with(robj):
        viewport.render_anim(
            filename=str(output_path),
            size=size,
            fps=int(fps),
            background=(0.0, 0.0, 0.0),
            renderer=robj,
            range=(0, frame_count - 1),
            every_nth=every_nth,
        )

    try:
        try:
            _render_with(renderer_obj)
        except RuntimeError:
            # GPU/ANARI renderers (e.g. visrtx) can fail to initialise a
            # rendering device on some headless nodes; fall back to the CPU
            # Tachyon renderer, which needs no GPU or display.
            if renderer.lower() == "tachyon":
                raise
            renderer = "tachyon"
            _render_with(_make_renderer("tachyon", samples_per_pixel))
    finally:
        pipeline.remove_from_scene()
    elapsed_s = time.perf_counter() - start_time
    rendered_frames = ((frame_count - 1) // every_nth) + 1
    result = {
        "path": output_path.as_posix(),
        "input_frames": frame_count,
        "every_nth": every_nth,
        "rendered_frames": rendered_frames,
        "render_elapsed_s": elapsed_s,
        "render_s_per_frame": elapsed_s / max(rendered_frames, 1),
        "renderer": renderer,
    }
    browser_path, transcode_elapsed_s = make_browser_safe_mp4(output_path)
    result["browser_path"] = browser_path
    result["transcode_elapsed_s"] = transcode_elapsed_s
    return result


def display_trajectory_animation(
    frames,
    *,
    label: str | None = None,
    target_frames: int = 12,
    fps: int = 6,
    width: int = 480,
    output_path=None,
    **render_kwargs,
) -> dict:
    """Render a trajectory MP4 and embed it inline in the notebook.

    Thin wrapper over :func:`render_trajectory_animation` that defaults the MP4
    to a transient scratch path under ``assets/images/.anim_cache/`` (not a
    shipped asset) and displays the browser-safe video with looping controls.
    Returns the same dict as :func:`render_trajectory_animation`.
    """
    from pathlib import Path

    from IPython.display import Video, display

    if output_path is None:
        output_path = Path("assets/images") / ".anim_cache" / f"{(label or 'trajectory')}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = render_trajectory_animation(
        frames,
        output_path=output_path,
        label=label,
        target_frames=target_frames,
        fps=fps,
        **render_kwargs,
    )
    display(
        Video(
            filename=result["browser_path"],
            embed=True,
            width=width,
            html_attributes="controls loop muted playsinline",
        )
    )
    return result
