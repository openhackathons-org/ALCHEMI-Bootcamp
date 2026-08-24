# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained interactive viewers for the ALCHEMI Core notebooks.

The public functions accept prepared structures and simulation callbacks. This
module owns conversion to browser data, synchronized controls, and the static
previews that remain useful when a notebook host does not run JavaScript.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from html import escape
from io import StringIO
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write

if TYPE_CHECKING:
    from nvalchemi.data import Batch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PALETTE = json.loads(
    (_PROJECT_ROOT / "shared" / "alchemi-palette.json").read_text(encoding="utf-8")
)
_VIEWER_BUNDLE = _PROJECT_ROOT / "shared" / "3dmol-2.5.5" / "3Dmol-min.js"

_BACKGROUND = _PALETTE["background"]
_SURFACE = _PALETTE["surface"]
_SURFACE_RAISED = _PALETTE["surface_raised"]
_BORDER = _PALETTE["border"]
_TEXT = _PALETTE["text"]
_MUTED = _PALETTE["muted"]
_GREEN = _PALETTE["nvidia_green"]
_BLUE = _PALETTE["nvidia_blue"]
_MAX_WIDTH = 920
_VIEWER_HEIGHT = 320
_TUTORIAL_FORMULA_PREFIX = ("C", "N", "O")
MAX_RELAXATION_SYSTEMS = 5

_ELEMENT_COLORS = {
    "H": "#E8ECEF",
    "C": "#5C6770",
    "N": "#00A3E0",
    "O": "#E05252",
    "F": "#8DC63F",
    "P": "#F0A23B",
    "S": "#E1C542",
    "Cl": "#4CB944",
    "Cu": "#C47B48",
    "Pt": "#A7B0BA",
    "Zn": "#8294AE",
}


def _html(markup: str) -> Any:
    from IPython.display import HTML

    return HTML(markup)


def runtime_required_notice(
    *,
    element_id: str | None = None,
    fill: bool = False,
    margin: str = "0 0 12px",
) -> str:
    """Return the shared static-preview notice for interactive outputs."""

    identity = f' id="{escape(element_id, quote=True)}"' if element_id else ""
    layout = (
        "height:100%;margin:0;padding:28px;display:flex;flex-direction:column;"
        "justify-content:center;align-items:center;text-align:center"
        if fill
        else f"margin:{margin};padding:14px 16px"
    )
    return f"""
<aside{identity} data-runtime-note role="note"
       style="box-sizing:border-box;{layout};gap:7px;border:1px solid {_BORDER};
              border-left:3px solid {_GREEN};border-radius:8px;
              background:{_BACKGROUND};color:{_TEXT};
              font-family:'NVIDIA Sans',Arial,sans-serif;">
  <strong style="font-size:.92rem;font-weight:700;">Runtime required</strong>
  <span style="max-width:620px;color:{_MUTED};font-size:.84rem;line-height:1.45;">
    This interactive output may not display properly on GitHub. Open the notebook
    in Jupyter or VS Code, connect to a runtime, and run this cell.
  </span>
</aside>
""".strip()


def _json_for_script(value: Any) -> str:
    """Serialize browser data without allowing a value to close its script tag."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _atoms_to_xyz(atoms: Atoms) -> str:
    stream = StringIO()
    write(stream, atoms, format="xyz")
    return stream.getvalue()


def _formula(atoms: Atoms) -> str:
    return atoms.get_chemical_formula(mode="hill")


def _viewing_rotation(positions: np.ndarray) -> np.ndarray:
    """Return a rigid rotation that turns a structure to face the default camera.

    3Dmol.js looks down z, so anything built along z is drawn end-on: ASE's
    linear `molecule("C2H2")` collapses to a single visible atom. Aligning the
    two broadest principal axes with the screen fixes that for any structure,
    and picks the same framing `plot_molecule` projects onto, so the interactive
    view and the flat projection agree.
    """

    centered = positions - positions.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=True)
    if np.linalg.det(axes) < 0:
        axes[2] *= -1.0  # keep the viewing frame right-handed
    return axes


def _to_view_frame(
    points: np.ndarray, origin: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Apply one viewing transform to atom positions and lattice corners alike."""

    return (np.atleast_2d(points) - origin) @ rotation.T


def _cell_edge_points(cell: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the twelve edges of a lattice parallelepiped anchored at the origin."""

    vectors = np.asarray(cell, dtype=float)
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for axis in range(3):
        first, second = (vectors[index] for index in range(3) if index != axis)
        for along_first, along_second in ((0, 0), (1, 0), (0, 1), (1, 1)):
            corner = along_first * first + along_second * second
            edges.append((corner, corner + vectors[axis]))
    return edges


def _viewer_point(position: np.ndarray) -> dict[str, float]:
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def _tutorial_formula(atoms: Atoms) -> str:
    """Format viewer labels with C, N, and O first and hydrogen last."""

    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1

    ordered = [symbol for symbol in _TUTORIAL_FORMULA_PREFIX if symbol in counts]
    ordered.extend(sorted(symbol for symbol in counts if symbol not in {*ordered, "H"}))
    if "H" in counts:
        ordered.append("H")

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in ordered
    )


def _zoom_factor(framed: np.ndarray, *, fill: float = 0.82) -> float:
    """Return the extra zoom that undoes 3Dmol's minimum framing distance.

    `zoomTo` never frames anything closer than its `minimumZoomToDistance`, a
    10 Å field of view that py3Dmol gives no way to reconfigure, so a four-atom
    molecule lands as a speck in the middle of the frame. `zoomTo` sizes that
    field to twice the distance from the bounding-box centre to the outermost
    point, so scaling by that ratio makes the structure span `fill` of the
    viewport's short side instead.
    """

    lower, upper = framed.min(axis=0), framed.max(axis=0)
    radius = float(np.linalg.norm(framed - (lower + upper) / 2, axis=1).max())
    if radius < 1.0e-9:
        return 1.0  # a lone atom has no extent to frame
    return max(radius, 5.0) * fill / radius


def _viewer_html(
    view: Any,
    height: int,
    *,
    formula: str,
    atom_count: int,
) -> str:
    """Wrap py3Dmol's snippet in a compact, self-contained structure widget.

    py3Dmol always emits a `<script src=...>` pointing at jsDelivr, guarded by
    `if (typeof $3Dmolpromise === 'undefined')`. Defining that promise ahead of
    the snippet satisfies the guard, so the vendored bundle inlined here is the
    only copy of 3Dmol.js the page ever sees. Inlining rather than linking is
    what keeps the view alive inside a self-contained `nbconvert` export.
    """

    bundle = _VIEWER_BUNDLE.read_text(encoding="utf-8")
    # `var define, exports, module` shadows the CommonJS and AMD hooks the UMD
    # bundle probes. nbconvert's HTML template loads RequireJS, and a defined
    # `define.amd` would park the bundle in a module registry that nothing here
    # requires, leaving `window.$3Dmol` undefined and the viewer blank.
    preamble = (
        "<script>(function(){var define,exports,module;\n"
        f"{bundle}\n"
        "$3Dmolpromise=Promise.resolve();}).call(window);</script>"
    )
    viewer_markup = view.write_html()
    warning = re.search(
        r'<p id="(3dmolwarning_[^"]+)"[^>]*>.*?</p>',
        viewer_markup,
        flags=re.DOTALL,
    )
    if warning:
        warning_id = escape(warning.group(1), quote=True)
        runtime_notice = runtime_required_notice(
            element_id=warning_id,
            fill=True,
        )
        viewer_markup = (
            viewer_markup[: warning.start()]
            + runtime_notice
            + viewer_markup[warning.end() :]
        )
    formula_text = escape(formula)
    return f"""
<div class="alchemi-structure-widget"
     aria-label="Interactive structure viewer for {formula_text}"
     style="box-sizing:border-box; width:100%; max-width:{_MAX_WIDTH}px;
            margin:0; overflow:hidden; border-radius:10px;
            background:{_BACKGROUND}; color:{_TEXT};
            font-family:'NVIDIA Sans',Arial,sans-serif;">
  <div style="display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between;
              gap:16px; padding:12px 16px 10px; background:{_SURFACE};">
    <div style="display:flex; align-items:baseline; gap:10px; min-width:0;">
      <strong style="font-size:1.05rem; font-weight:650;">{formula_text}</strong>
      <span style="color:{_MUTED}; font-size:0.84rem;">{atom_count} atoms</span>
    </div>
    <span style="color:{_MUTED}; font-size:0.78rem; text-align:right;">
      Drag to rotate · scroll to zoom · select up to 3 atoms
    </span>
  </div>
  <div class="alchemi-structure-view"
       style="width:100%; height:{height}px; overflow:hidden;
              background:{_BACKGROUND};">
    {preamble}{viewer_markup}
  </div>
  <div style="display:flex; align-items:center; justify-content:space-between;
              gap:14px; padding:10px 16px 12px; background:{_SURFACE};">
    <div data-atom-detail role="status" aria-live="polite"
         style="min-height:22px; color:{_TEXT}; font-size:0.84rem;
                font-variant-numeric:tabular-nums;">
      Select 1 atom to inspect, 2 for a distance, or 3 for an angle.
    </div>
    <button type="button" data-clear-measurement disabled
            style="flex:none; border:1px solid {_BORDER}; border-radius:6px;
                   padding:5px 9px; background:{_SURFACE_RAISED}; color:{_TEXT};
                   font:inherit; font-size:0.76rem; cursor:pointer;">
      Clear
    </button>
  </div>
</div>
"""


def _atom_click_callback(atoms: Atoms) -> str:
    """Return the 3Dmol.js callback for atom selection and measurements."""

    atomic_numbers = {
        chemical_symbols[number]: int(number)
        for number in sorted({int(number) for number in atoms.get_atomic_numbers()})
    }
    number_map = json.dumps(atomic_numbers, sort_keys=True)
    return f"""function(atom, viewer, event, container) {{
      const numbers = {number_map};
      const host = container && container.jquery ? container[0] : container;
      const root = host && host.closest
        ? host.closest('.alchemi-structure-widget')
        : null;
      const atomIndex = Number(atom.serial);
      const atomicNumber = numbers[atom.elem];
      const detail = root ? root.querySelector('[data-atom-detail]') : null;
      const clearButton = root
        ? root.querySelector('[data-clear-measurement]')
        : null;
      const state = viewer.__alchemiMeasurement || {{
        atoms: [],
        shapes: [],
        labels: []
      }};
      viewer.__alchemiMeasurement = state;

      const labelPixelRatio = Math.min(
        3,
        Math.max(2, window.devicePixelRatio||1)
      );
      const highResolutionLabel = function(text, spec) {{
        const fontSize = Number(spec.fontSize || 12);
        const padding = Number(spec.padding || 4);
        const borderThickness = Number(spec.borderThickness || 0);
        const label = viewer.addLabel(text, {{
          ...spec,
          fontSize:Math.round(fontSize*labelPixelRatio),
          padding:padding*labelPixelRatio,
          borderThickness:borderThickness*labelPixelRatio
        }});
        label.sprite.scale.set(1/labelPixelRatio,1/labelPixelRatio,1);
        return label;
      }};

      const reset = function(message) {{
        state.shapes.forEach(function(shape) {{ viewer.removeShape(shape); }});
        state.labels.forEach(function(label) {{ viewer.removeLabel(label); }});
        state.atoms = [];
        state.shapes = [];
        state.labels = [];
        if (detail) {{
          detail.textContent = message
            || 'Select 1 atom to inspect, 2 for a distance, or 3 for an angle.';
        }}
        if (clearButton) {{ clearButton.disabled = true; }}
        viewer.render();
      }};

      if (clearButton && !clearButton.__alchemiBound) {{
        clearButton.__alchemiBound = true;
        clearButton.addEventListener('click', function() {{ reset(); }});
      }}
      if (state.atoms.length === 3) {{ reset(''); }}
      if (state.atoms.some(function(selected) {{
        return Number(selected.serial) === atomIndex;
      }})) {{
        if (detail) {{
          detail.textContent = 'Row ' + atomIndex
            + ' is already selected. Choose a different atom.';
        }}
        return;
      }}

      state.atoms.push(atom);
      state.shapes.push(viewer.addSphere({{
        center: atom,
        radius: 0.34,
        color: '{_GREEN}',
        opacity: 0.30
      }}));
      state.labels.push(highResolutionLabel(
        atom.elem + ' · row ' + atomIndex,
        {{
          position: atom,
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.92,
          fontColor: '{_TEXT}',
          fontSize: 11,
          borderColor: '{_GREEN}',
          borderThickness: 1,
          inFront: true
        }}
      ));
      if (clearButton) {{ clearButton.disabled = false; }}

      const distance = function(first, second) {{
        return Math.hypot(
          first.x - second.x,
          first.y - second.y,
          first.z - second.z
        );
      }};
      const midpoint = function(first, second) {{
        return {{
          x: (first.x + second.x) / 2,
          y: (first.y + second.y) / 2,
          z: (first.z + second.z) / 2
        }};
      }};

      if (state.atoms.length === 1) {{
        if (detail) {{
          detail.textContent = 'Selected row ' + atomIndex + ' · ' + atom.elem
            + ' · atomic number ' + atomicNumber
            + '. Select a second atom for a distance.';
        }}
      }} else if (state.atoms.length === 2) {{
        const first = state.atoms[0];
        const second = state.atoms[1];
        const value = distance(first, second);
        state.shapes.push(viewer.addLine({{
          start: first,
          end: second,
          color: '{_BLUE}',
          dashed: true,
          dashLength: 0.10,
          gapLength: 0.06
        }}));
        state.labels.push(highResolutionLabel(value.toFixed(3) + ' Å', {{
          position: midpoint(first, second),
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.94,
          fontColor: '{_TEXT}',
          fontSize: 12,
          borderColor: '{_BLUE}',
          borderThickness: 1,
          inFront: true
        }}));
        if (detail) {{
          detail.textContent = 'Distance rows ' + Number(first.serial) + '–'
            + Number(second.serial) + ': ' + value.toFixed(3)
            + ' Å. Select a third atom; row ' + Number(second.serial)
            + ' will be the angle vertex.';
        }}
      }} else {{
        const first = state.atoms[0];
        const vertex = state.atoms[1];
        const third = state.atoms[2];
        const ux = first.x - vertex.x;
        const uy = first.y - vertex.y;
        const uz = first.z - vertex.z;
        const vx = third.x - vertex.x;
        const vy = third.y - vertex.y;
        const vz = third.z - vertex.z;
        const denominator = Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz);
        const cosine = Math.max(
          -1,
          Math.min(1, (ux * vx + uy * vy + uz * vz) / denominator)
        );
        const angle = Math.acos(cosine) * 180 / Math.PI;
        state.shapes.push(viewer.addLine({{
          start: vertex,
          end: third,
          color: '{_BLUE}',
          dashed: true,
          dashLength: 0.10,
          gapLength: 0.06
        }}));
        state.labels.push(highResolutionLabel(angle.toFixed(1) + '°', {{
          position: vertex,
          screenOffset: {{x: 16, y: -22}},
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.94,
          fontColor: '{_TEXT}',
          fontSize: 12,
          borderColor: '{_BLUE}',
          borderThickness: 1,
          inFront: true
        }}));
        if (detail) {{
          detail.textContent = 'Angle rows ' + Number(first.serial) + '–'
            + Number(vertex.serial) + '–' + Number(third.serial) + ': '
            + angle.toFixed(1) + '° · row ' + Number(vertex.serial)
            + ' is the vertex. Click another atom to start again.';
        }}
      }}
      viewer.render();
    }}"""


def show_molecule(
    atoms: Atoms,
    *,
    height: int = _VIEWER_HEIGHT,
    sphere_scale: float = 0.23,
) -> Any:
    """Return a compact 3Dmol.js ball-and-stick view of one ASE structure.

    3Dmol.js perceives connectivity from the coordinates alone, so no explicit
    bond list is passed in. A periodic structure also gets its lattice drawn as
    twelve edges: 3Dmol's `addUnitCell` reads crystal metadata that XYZ input
    cannot carry, so it draws nothing here.

    The structure is turned to face the camera before it is handed over. The
    rigid rotation preserves the distances and angles measured in the viewer.
    """

    import py3Dmol
    from IPython.display import HTML

    positions = atoms.get_positions()
    origin = positions.mean(axis=0)
    rotation = _viewing_rotation(positions)
    oriented = atoms.copy()
    oriented.set_positions(_to_view_frame(positions, origin, rotation))

    view = py3Dmol.view(
        width="100%",
        height=height,
        # Never fetched: `_viewer_html` pre-resolves py3Dmol's loader promise.
        # Kept as the repository-relative provenance of the inlined bundle.
        js="shared/3dmol-2.5.5/3Dmol-min.js",
    )
    view.addModel(_atoms_to_xyz(oriented), "xyz")
    # One `setStyle` per element rather than a single `colorscheme` map: 3Dmol
    # 2.5.5 tests a custom map with `scheme[element]` but reads the colour back
    # from `scheme.map[element]`, so the map never applies and carbon keeps its
    # near-white default next to hydrogen.
    for number in sorted({int(number) for number in atoms.get_atomic_numbers()}):
        color = _ELEMENT_COLORS.get(chemical_symbols[number], _GREEN)
        view.setStyle(
            {"elem": chemical_symbols[number]},
            {
                "sphere": {"scale": sphere_scale, "color": color},
                "stick": {"radius": 0.09, "color": color},
            },
        )
    view.setClickable({}, True, _atom_click_callback(atoms))
    framed = [oriented.get_positions()]
    if atoms.pbc.any():
        for edge in _cell_edge_points(atoms.cell):
            start, end = _to_view_frame(np.asarray(edge), origin, rotation)
            framed.append(np.stack([start, end]))
            view.addLine(
                {
                    "start": _viewer_point(start),
                    "end": _viewer_point(end),
                    "color": _BORDER,
                }
            )
    view.setBackgroundColor(_BACKGROUND)
    view.zoomTo()
    view.zoom(_zoom_factor(np.concatenate(framed)))
    # A fixed oblique tilt, so a linear or planar structure and a lattice box all
    # still read as three-dimensional instead of as a rod or a flat outline.
    view.rotate(-24, "y")
    view.rotate(16, "x")
    return HTML(
        _viewer_html(
            view,
            height,
            formula=_tutorial_formula(atoms),
            atom_count=len(atoms),
        )
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _graph_count(frame: Any) -> int:
    count = getattr(frame, "num_graphs", None)
    if count is not None:
        return int(count)
    batch_ptr = getattr(frame, "batch_ptr", None)
    if batch_ptr is None:
        raise TypeError("A batched frame must provide num_graphs or batch_ptr")
    return int(len(_to_numpy(batch_ptr).reshape(-1)) - 1)


def _data_to_atoms(data: Any) -> Atoms:
    numbers = _to_numpy(data.atomic_numbers).reshape(-1).astype(int)
    positions = _to_numpy(data.positions).reshape(-1, 3).astype(float)
    cell_value = getattr(data, "cell", None)
    pbc_value = getattr(data, "pbc", None)
    cell = None if cell_value is None else _to_numpy(cell_value).reshape(-1, 3)[-3:]
    pbc_array = None if pbc_value is None else _to_numpy(pbc_value).reshape(-1)
    pbc = (
        False
        if pbc_array is None
        else bool(pbc_array[0])
        if pbc_array.size == 1
        else pbc_array[-3:]
    )
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


def _graph_atoms(frame: Any, graph_index: int) -> Atoms:
    """Extract one graph while applying its row offset exactly once."""

    if isinstance(frame, Atoms):
        if graph_index != 0:
            raise IndexError("An ASE Atoms frame contains one graph")
        return frame.copy()

    get_data = getattr(frame, "get_data", None)
    if callable(get_data):
        return _data_to_atoms(get_data(graph_index))

    ptr = _to_numpy(frame.batch_ptr).reshape(-1).astype(int)
    if graph_index < 0 or graph_index + 1 >= len(ptr):
        raise IndexError(f"graph_index {graph_index} is outside the batch")
    start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
    numbers = _to_numpy(frame.atomic_numbers).reshape(-1)[start:stop]
    positions = _to_numpy(frame.positions).reshape(-1, 3)[start:stop]

    cell_value = getattr(frame, "cell", None)
    pbc_value = getattr(frame, "pbc", None)
    cell = None
    pbc: Any = False
    if cell_value is not None:
        cells = _to_numpy(cell_value)
        cell = cells.reshape(-1, 3, 3)[graph_index]
    if pbc_value is not None:
        pbcs = _to_numpy(pbc_value)
        pbc = (
            bool(pbcs.reshape(-1)[0])
            if pbcs.size == 1
            else pbcs.reshape(-1, 3)[graph_index]
        )
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


def _normalise_trajectories(
    trajectories: Mapping[str, Sequence[Any]] | Sequence[Any],
    labels: Sequence[str] | None,
) -> dict[str, list[Atoms]]:
    if isinstance(trajectories, Mapping):
        result = {
            str(label): [
                frame.copy() if isinstance(frame, Atoms) else _graph_atoms(frame, 0)
                for frame in frames
            ]
            for label, frames in trajectories.items()
        }
    else:
        frames = list(trajectories)
        if not frames:
            raise ValueError("trajectories must contain at least one frame")
        graph_count = _graph_count(frames[0])
        names = (
            list(labels)
            if labels is not None
            else [f"System {i + 1}" for i in range(graph_count)]
        )
        if len(names) != graph_count:
            raise ValueError("labels must contain one name per graph")
        if any(_graph_count(frame) != graph_count for frame in frames):
            raise ValueError("every batched frame must contain the same graph count")
        result = {
            str(name): [_graph_atoms(frame, graph_index) for frame in frames]
            for graph_index, name in enumerate(names)
        }

    if not result or any(not frames for frames in result.values()):
        raise ValueError("each trajectory must contain at least one frame")
    frame_counts = {len(frames) for frames in result.values()}
    if len(frame_counts) != 1:
        raise ValueError("synchronized trajectories must contain the same frame count")
    return result


def _normalise_atom_force_magnitudes(
    trajectories: Mapping[str, Sequence[Any]] | Sequence[Any],
    labels: Sequence[str] | None,
) -> dict[str, list[list[float]]] | None:
    """Return per-atom force magnitudes when trajectory frames store forces."""

    if isinstance(trajectories, Mapping):
        result: dict[str, list[list[float]]] = {}
        for label, frames in trajectories.items():
            magnitudes = []
            for frame in frames:
                forces = getattr(frame, "forces", None)
                if forces is None and isinstance(frame, Atoms):
                    forces = frame.arrays.get("forces")
                if forces is None:
                    return None
                values = np.linalg.norm(_to_numpy(forces).reshape(-1, 3), axis=1)
                magnitudes.append(values.astype(float).tolist())
            result[str(label)] = magnitudes
        return result

    frames = list(trajectories)
    if not frames:
        return None
    graph_count = _graph_count(frames[0])
    names = (
        list(labels)
        if labels is not None
        else [f"System {index + 1}" for index in range(graph_count)]
    )
    result = {str(name): [] for name in names}
    for frame in frames:
        for graph_index, name in enumerate(names):
            data = frame.get_data(graph_index)
            forces = getattr(data, "forces", None)
            if forces is None:
                return None
            values = np.linalg.norm(_to_numpy(forces).reshape(-1, 3), axis=1)
            result[str(name)].append(values.astype(float).tolist())
    return result


def snapshot_batches(sink: Any, num_graphs: int) -> list[Batch]:
    """Rebuild synchronized batches from graph records stored by ``HostMemory``.

    ``SnapshotHook`` writes every graph in one simulation batch before writing
    the next step. ``HostMemory.read()`` returns those records as one flat
    ``Batch``. This helper checks the record count and groups each consecutive
    set of ``num_graphs`` records back into one snapshot.
    """

    if (
        isinstance(num_graphs, bool)
        or not isinstance(num_graphs, int)
        or num_graphs < 1
    ):
        raise ValueError("num_graphs must be a positive integer")
    if not callable(getattr(sink, "read", None)):
        raise TypeError("sink must provide read()")
    if hasattr(sink, "__len__") and len(sink) == 0:
        raise ValueError("snapshot sink is empty")

    flat_batch = sink.read()
    total_graphs = int(flat_batch.num_graphs)
    if total_graphs == 0:
        raise ValueError("snapshot sink is empty")
    if total_graphs % num_graphs:
        raise ValueError(
            f"snapshot sink contains {total_graphs} graph records; expected a multiple "
            f"of {num_graphs}"
        )

    from nvalchemi.data import Batch

    records = flat_batch.to_data_list()
    return [
        Batch.from_data_list(
            records[start : start + num_graphs], device=flat_batch.device
        )
        for start in range(0, total_graphs, num_graphs)
    ]


def _cell_edges(atoms: Atoms) -> list[dict[str, list[float]]]:
    if not bool(np.asarray(atoms.pbc).any()):
        return []
    vectors = np.asarray(atoms.cell, dtype=float)
    edges: list[dict[str, list[float]]] = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for first in (0, 1):
            for second in (0, 1):
                start = first * vectors[other[0]] + second * vectors[other[1]]
                stop = start + vectors[axis]
                edges.append({"start": start.tolist(), "end": stop.tolist()})
    return edges


def _structure_payload(atoms: Atoms) -> dict[str, Any]:
    sphere_only = bool(np.asarray(atoms.pbc).any()) and bool(
        {"Cu", "Pt"}.intersection(atoms.get_chemical_symbols())
    )
    return {
        "xyz": _atoms_to_xyz(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.asarray(atoms.positions, dtype=float).tolist(),
        "formula": _formula(atoms),
        "atom_count": len(atoms),
        "periodic": bool(np.asarray(atoms.pbc).any()),
        "sphere_only": sphere_only,
        "cell_edges": _cell_edges(atoms),
    }


def _project_positions(atoms: Atoms) -> np.ndarray:
    positions = np.asarray(atoms.positions, dtype=float)
    centered = positions - positions.mean(axis=0, keepdims=True)
    if len(atoms) > 1:
        _, _, axes = np.linalg.svd(centered, full_matrices=True)
        centered = centered @ axes.T
    return centered[:, :2]


def _static_structure_svg(
    atoms: Atoms,
    *,
    title: str,
    subtitle: str = "Static preview",
    width: int = 280,
    height: int = 190,
) -> str:
    points = _project_positions(atoms)
    extent = np.ptp(points, axis=0)
    scale = min(
        (width - 44) / max(float(extent[0]), 1.0),
        (height - 68) / max(float(extent[1]), 1.0),
    )
    xy = points * scale
    xy[:, 0] += width / 2
    xy[:, 1] = height / 2 - xy[:, 1] + 10
    atoms_svg = []
    for symbol, (x_coord, y_coord) in zip(
        atoms.get_chemical_symbols(), xy, strict=True
    ):
        color = _ELEMENT_COLORS.get(symbol, _GREEN)
        atoms_svg.append(
            f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="6" fill="{color}" '
            f'stroke="{_BACKGROUND}" stroke-width="1.2"><title>{escape(symbol)}</title></circle>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Static structure preview for {escape(title, quote=True)}" '
        'style="display:block;width:100%;height:auto">'
        f'<rect width="{width}" height="{height}" rx="8" fill="{_BACKGROUND}"/>'
        f'<text x="14" y="22" fill="{_TEXT}" font-size="13" font-weight="650" '
        f'font-family="NVIDIA Sans,Arial,sans-serif">{escape(title)}</text>'
        f'<text x="14" y="40" fill="{_MUTED}" font-size="10" '
        f'font-family="NVIDIA Sans,Arial,sans-serif">{escape(subtitle)}</text>'
        + "".join(atoms_svg)
        + "</svg>"
    )


def _static_pair_svg(
    first: Atoms,
    last: Atoms,
    *,
    title: str,
    first_label: str = "start",
    last_label: str = "end",
) -> str:
    left = _static_structure_svg(first, title=first_label, width=260, height=175)
    right = _static_structure_svg(last, title=last_label, width=260, height=175)
    return (
        f'<div role="img" aria-label="{escape(title, quote=True)}" '
        'style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px">'
        f"{left}{right}</div>"
    )


def _bundle_preamble() -> str:
    bundle = _VIEWER_BUNDLE.read_text(encoding="utf-8")
    return (
        "<script>(function(){if(window.$3Dmol){window.$3Dmolpromise=Promise.resolve();return;}"
        "var define,exports,module;\n"
        f"{bundle}\n"
        "window.$3Dmolpromise=Promise.resolve();}).call(window);</script>"
    )


def _trajectory_viewers_javascript() -> str:
    """Return the shared py3Dmol setup for synchronized trajectory cards."""

    return (
        f"const elementColors={_json_for_script(_ELEMENT_COLORS)};\n"
        "const applyStyle=function(viewer){viewer.setStyle({},"
        f"{{sphere:{{scale:.23,color:'{_GREEN}'}},stick:{{radius:.09,color:'{_GREEN}'}}}});"
        "Object.keys(elementColors).forEach(function(symbol){viewer.setStyle("
        "{elem:symbol},{sphere:{scale:.23,color:elementColors[symbol]},"
        "stick:{radius:.09,color:elementColors[symbol]}});});};\n"
        "const viewers=items.map(function(item,index){const node=document.querySelector("
        "'[data-viewer=\"'+index+'\"]');const viewer=$3Dmol.createViewer(node,"
        f"{{backgroundColor:'{_BACKGROUND}',antialias:true}});"
        "viewer.addModelsAsFrames(item.trajectory_xyz,'xyz');applyStyle(viewer);"
        "viewer.zoomTo();viewer.rotate(-18,'y');viewer.rotate(12,'x');"
        "viewer.render();return viewer;});"
    )


def _player_iframe(document: str, *, title: str, height: int) -> str:
    """Wrap one self-contained player document in its notebook iframe."""

    return (
        f'<iframe title="{escape(title, quote=True)}" '
        f'srcdoc="{escape(document, quote=True)}" style="display:block;width:100%;'
        f'height:{height}px;border:0;background:{_BACKGROUND};" loading="eager"></iframe>'
    )


def _common_css(root_id: str) -> str:
    return f"""
<style>
#{root_id}{{box-sizing:border-box;width:100%;max-width:{_MAX_WIDTH}px;margin:0;
  overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND};
  color:{_TEXT};font-family:'NVIDIA Sans',Arial,sans-serif}}
#{root_id} *{{box-sizing:border-box}}
#{root_id} button,#{root_id} select,#{root_id} input{{font:inherit}}
#{root_id} button{{min-width:40px;min-height:40px;border:1px solid {_BORDER};border-radius:7px;
  background:{_SURFACE_RAISED};color:{_TEXT};cursor:pointer;touch-action:manipulation}}
#{root_id} button:hover{{border-color:{_GREEN}}}
#{root_id} button:focus-visible,#{root_id} select:focus-visible,#{root_id} input:focus-visible{{
  outline:3px solid {_GREEN};outline-offset:2px}}
#{root_id} [data-live-view]{{position:relative;isolation:isolate;min-width:0;width:100%;
  height:{_VIEWER_HEIGHT}px;overflow:hidden;background:{_BACKGROUND}}}
#{root_id} [data-live-view] canvas{{position:absolute!important;inset:0;width:100%!important;
  height:100%!important;max-width:100%!important}}
#{root_id} [data-interactive]{{display:none}}
#{root_id}.is-live [data-interactive]{{display:block}}
#{root_id}.is-live [data-interactive].alchemi-grid{{display:grid}}
#{root_id}.is-live [data-static-fallback]{{display:none}}
#{root_id} .alchemi-toolbar{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
#{root_id} .alchemi-muted{{color:{_MUTED}}}
@media (max-width:640px){{
  #{root_id} .alchemi-grid{{grid-template-columns:1fr!important}}
  #{root_id} [data-live-view]{{height:{_VIEWER_HEIGHT}px}}
}}
</style>
"""


def _metadata_from_spec(key: str, spec: Any, atoms: Atoms) -> dict[str, Any]:
    return {
        "key": key,
        "display_name": str(_field(spec, "display_name", key)),
        "domain": str(_field(spec, "domain", "Other")),
        "purpose": str(_field(spec, "purpose", "Structure exploration")),
        **_structure_payload(atoms),
    }


def _py3dmol_card_view(item: Mapping[str, Any], *, height: int = 172) -> Any:
    """Build one self-contained py3Dmol viewer for a selectable grid card."""

    import py3Dmol

    viewer = py3Dmol.view(
        width="100%",
        height=height,
        js="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js",
    )
    viewer.setBackgroundColor(_BACKGROUND)
    viewer.addModel(
        item["xyz"],
        "xyz",
        {"assignBonds": not bool(item["sphere_only"])},
    )
    atom_scale = 0.38 if item["sphere_only"] else 0.24
    base_style = {"sphere": {"scale": atom_scale, "color": _GREEN}}
    if not item["sphere_only"]:
        base_style["stick"] = {"radius": 0.09, "color": _GREEN}
    viewer.setStyle({}, base_style)
    for symbol, color in _ELEMENT_COLORS.items():
        element_style = {"sphere": {"scale": atom_scale, "color": color}}
        if not item["sphere_only"]:
            element_style["stick"] = {"radius": 0.09, "color": color}
        viewer.setStyle(
            {"elem": symbol},
            element_style,
        )
    for edge in item["cell_edges"]:
        viewer.addLine(
            {
                "start": dict(zip(("x", "y", "z"), edge["start"], strict=True)),
                "end": dict(zip(("x", "y", "z"), edge["end"], strict=True)),
                "color": _BORDER,
            }
        )
    viewer.zoomTo()
    viewer.rotate(-18, "y")
    viewer.rotate(10, "x")
    return viewer


def _py3dmol_card_markup(item: Mapping[str, Any], *, height: int = 172) -> str:
    """Return py3Dmol HTML that works inside Jupyter and VS Code webviews."""

    view = _py3dmol_card_view(item, height=height)
    return _bundle_preamble() + view.write_html()


def _viewer_iframe(markup: str, *, height: int) -> str:
    """Run viewer scripts in an isolated dark document without Output widgets."""

    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;"
        f"background:{_BACKGROUND};}}</style></head><body>"
        f"{markup}</body></html>"
    )
    return (
        f'<iframe title="Interactive 3D structure" srcdoc="{escape(document, quote=True)}" '
        f'style="display:block;width:100%;height:{height}px;border:0;'
        f'background:{_BACKGROUND};" loading="eager"></iframe>'
    )


class StructureSelection:
    """Jupyter widget that connects a molecular card grid to later Python cells."""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        items: Sequence[dict[str, Any]],
        selected_keys: Sequence[str],
    ) -> None:
        import ipywidgets as widgets

        self._catalog = dict(catalog)
        self._items = list(items)
        self._checkboxes: dict[str, Any] = {}
        self._cards: dict[str, Any] = {}
        self._status = widgets.HTML()
        widget_class = f"alchemi-picker-{uuid4().hex}"

        for item in items:
            key = item["key"]
            control = widgets.ToggleButton(
                value=key in selected_keys,
                description=(
                    "✓ In batch" if key in selected_keys else "+ Add to batch"
                ),
                layout=widgets.Layout(width="100%", height="40px"),
            )
            control.add_class("alchemi-picker-state")
            control.observe(
                lambda change, item_key=key: self._toggle(item_key, change),
                "value",
            )
            self._checkboxes[key] = control
            viewer = widgets.HTML(
                value=_viewer_iframe(
                    _py3dmol_card_markup(item, height=164),
                    height=164,
                ),
                layout=widgets.Layout(width="100%", height="164px"),
            )
            viewer.add_class("alchemi-picker-viewer")
            facts = widgets.HTML(
                value=(
                    '<div class="alchemi-picker-facts">'
                    f"<strong>{escape(item['display_name'])}</strong>"
                    f"<span>{item['atom_count']} atoms</span>"
                    f"<p>{escape(item['purpose'])} · "
                    f"{escape(item['domain'])}</p></div>"
                ),
                layout=widgets.Layout(width="100%", height="76px"),
            )
            card = widgets.VBox(
                [viewer, facts, control],
                layout=widgets.Layout(
                    width="100%",
                    height="300px",
                    padding="6px",
                    overflow="hidden",
                ),
            )
            card.add_class("alchemi-picker-card")
            if control.value:
                card.add_class("is-selected")
                control.add_class("is-selected")
            self._cards[key] = card

        title = widgets.HTML(
            value=(
                "<div><strong>Choose systems for the Batch</strong>"
                '<span class="alchemi-picker-guidance">Drag to rotate · scroll '
                f"to zoom · choose up to {MAX_RELAXATION_SYSTEMS} systems</span></div>"
            )
        )
        header = widgets.HBox(
            [title, self._status],
            layout=widgets.Layout(
                width="100%",
                align_items="flex-end",
                justify_content="space-between",
            ),
        )
        header.add_class("alchemi-picker-head")
        grid = widgets.GridBox(
            list(self._cards.values()),
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="repeat(4, minmax(0, 1fr))",
                grid_gap="12px",
            ),
        )
        grid.add_class("alchemi-picker-grid")

        styles = widgets.HTML(
            f"""
<style>
.cell-output-ipywidget-background:has(.{widget_class}),
.widget-subarea:has(> .{widget_class}),
.{widget_class}, .{widget_class}.jupyter-widgets {{
  background-color: {_BACKGROUND} !important; color: {_TEXT} !important;
}}
.{widget_class} {{
  box-sizing:border-box;width:100% !important;max-width:none !important;
  background:{_BACKGROUND} !important;color:{_TEXT} !important;
}}
.{widget_class} .widget-html,
.{widget_class} .widget-html-content {{
  width:100% !important;max-width:none !important;
  background:transparent !important;color:{_TEXT} !important;
}}
.{widget_class} .alchemi-picker-head {{padding:12px 12px 10px;border:1px solid {_BORDER};
  border-bottom:0;border-radius:10px 10px 0 0;background:{_BACKGROUND}}}
.{widget_class} .alchemi-picker-head strong{{font-size:1rem}}
.{widget_class} .alchemi-picker-guidance{{display:block;margin-top:3px;color:{_MUTED};font-size:.82rem}}
.{widget_class} .alchemi-picker-grid{{padding:0 12px 12px;border:1px solid {_BORDER};
  border-top:0;border-radius:0 0 10px 10px;background:{_BACKGROUND}}}
.{widget_class} .alchemi-picker-card{{border:2px solid {_BORDER};border-radius:10px;
  background:{_SURFACE};gap:6px}}
.{widget_class} .alchemi-picker-card.is-selected{{border-color:{_GREEN}}}
.{widget_class} .alchemi-picker-viewer{{overflow:hidden;background:{_BACKGROUND}}}
.{widget_class} .alchemi-picker-facts{{height:76px;padding:2px 4px;line-height:1.3;overflow:hidden}}
.{widget_class} .alchemi-picker-facts strong{{display:block;font-size:.92rem;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}}
.{widget_class} .alchemi-picker-facts span{{display:block;margin-top:2px;color:{_MUTED};
  font-size:.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.{widget_class} .alchemi-picker-facts p{{margin:3px 0 0;font-size:.77rem;line-height:1.25}}
.{widget_class} .alchemi-picker-state button{{width:100%;height:40px;border:2px solid {_GREEN};
  border-radius:7px;background:{_SURFACE_RAISED};color:{_GREEN};font-weight:800}}
.{widget_class} .alchemi-picker-state.is-selected button{{background:{_GREEN};color:{_BACKGROUND}}}
@media(max-width:760px){{.{widget_class} .alchemi-picker-grid{{grid-template-columns:repeat(2,minmax(0,1fr)) !important}}}}
@media(max-width:520px){{.{widget_class} .alchemi-picker-grid{{grid-template-columns:1fr !important}}}}
</style>
"""
        )
        self.widget = widgets.VBox(
            [styles, header, grid],
            layout=widgets.Layout(width="100%"),
        )
        self.widget.add_class("alchemi-structure-picker")
        self.widget.add_class(widget_class)
        self._static_fallback = self._build_static_fallback()
        self._update_status()

    @property
    def selected_keys(self) -> tuple[str, ...]:
        """Return selected catalog keys in the catalog's stable display order."""

        return tuple(
            item["key"]
            for item in self._items
            if bool(self._checkboxes[item["key"]].value)
        )

    def selected_specs(self) -> list[Any]:
        """Return the selected catalog entries for structure construction."""

        keys = self.selected_keys
        if not keys:
            raise ValueError(
                "Select at least one molecular card before building a Batch"
            )
        return [self._catalog[key] for key in keys]

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        bundle = self.widget._repr_mimebundle_(**kwargs)
        if isinstance(bundle, tuple):
            data, metadata = bundle
            data = dict(data)
            data["text/html"] = self._static_fallback
            return data, metadata
        data = {} if bundle is None else dict(bundle)
        data["text/html"] = self._static_fallback
        return data

    def _toggle(self, key: str, change: Mapping[str, Any]) -> None:
        control = self._checkboxes[key]
        card = self._cards[key]
        selected = bool(change["new"])
        if selected and len(self.selected_keys) > MAX_RELAXATION_SYSTEMS:
            control.value = False
            return
        control.description = "✓ In batch" if selected else "+ Add to batch"
        if selected:
            control.add_class("is-selected")
            card.add_class("is-selected")
        else:
            control.remove_class("is-selected")
            card.remove_class("is-selected")
        self._update_status()

    def _update_status(self) -> None:
        count = len(self.selected_keys)
        label = "system" if count == 1 else "systems"
        color = _GREEN if count else _MUTED
        message = f"{count} {label} in batch" if count else "Choose one or more systems"
        self._status.value = (
            f"<div style='font-family:NVIDIA Sans,Arial,sans-serif;"
            f"background:{_BACKGROUND};color:{color};"
            f"font-size:.86rem;font-weight:650'>{message}</div>"
        )

    def _build_static_fallback(self) -> str:
        return runtime_required_notice(margin="0")


def show_structure_explorer(
    catalog: Mapping[str, Any],
    *,
    selected: str | Sequence[str] | None = None,
) -> StructureSelection:
    """Show a selectable molecular grid linked to later Python cells.

    ``catalog`` accepts the ordered mapping returned by ``opening_performance_catalog``.
    """

    if not catalog:
        raise ValueError("catalog must contain at least one structure")
    items = []
    for key, spec in catalog.items():
        if isinstance(spec, Atoms) or not callable(getattr(spec, "build", None)):
            raise TypeError(f"catalog entry {key!r} must provide a build() method")
        atoms = spec.build()
        if not isinstance(atoms, Atoms):
            raise TypeError(f"catalog entry {key!r} did not build an ASE Atoms object")
        items.append(_metadata_from_spec(str(key), spec, atoms))

    valid_keys = {item["key"] for item in items}
    if selected is None:
        selected_keys = []
    elif isinstance(selected, str):
        selected_keys = [selected]
    else:
        selected_keys = list(selected)
    missing = [key for key in selected_keys if key not in valid_keys]
    if missing:
        raise KeyError(f"selected structures are absent from catalog: {missing}")
    if len(set(selected_keys)) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            f"select at most {MAX_RELAXATION_SYSTEMS} systems for the synchronized player"
        )
    return StructureSelection(catalog, items, selected_keys)


def _force_values(
    force_history: Mapping[str, Sequence[float]] | None,
    labels: Sequence[str],
    frame_count: int,
) -> dict[str, list[float | None]]:
    if force_history is None:
        return {label: [None] * frame_count for label in labels}
    result: dict[str, list[float | None]] = {}
    for label in labels:
        values = list(force_history.get(label, []))
        if values and len(values) != frame_count:
            raise ValueError(f"force history for {label!r} must match the trajectory")
        result[label] = (
            [float(value) for value in values] if values else [None] * frame_count
        )
    return result


def _reference_structure_map(
    reference_structures: Mapping[str, Atoms] | Sequence[Atoms] | None,
    names: Sequence[str],
) -> dict[str, Atoms]:
    """Align optional rendering metadata with the normalized trajectory names."""

    if reference_structures is None:
        return {}
    if isinstance(reference_structures, Mapping):
        missing = [name for name in names if name not in reference_structures]
        if missing:
            raise KeyError(f"reference structures are missing labels: {missing}")
        references = {name: reference_structures[name] for name in names}
    else:
        values = list(reference_structures)
        if len(values) != len(names):
            raise ValueError(
                "reference_structures must contain one structure per trajectory"
            )
        references = dict(zip(names, values, strict=True))
    if any(not isinstance(atoms, Atoms) for atoms in references.values()):
        raise TypeError("reference_structures must contain ASE Atoms objects")
    return references


def _force_sparkline_svg(item: Mapping[str, Any], index: int, frame_count: int) -> str:
    """Draw one compact maximum-force trace for a trajectory card."""

    width, height = 300, 42
    left, right, top, bottom = 4, 4, 4, 5
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite = [
        float(value)
        for value in item["forces"]
        if value is not None and np.isfinite(value)
    ]
    maximum = max(finite, default=1.0)
    maximum = maximum if maximum > 0 else 1.0

    def x_coord(index: int) -> float:
        return left + plot_width * index / max(frame_count - 1, 1)

    def y_coord(value: float) -> float:
        return top + plot_height * (1.0 - value / maximum)

    values = item["forces"]
    series = ""
    if not any(value is None for value in values):
        points = " ".join(
            f"{x_coord(frame):.2f},{y_coord(float(value)):.2f}"
            for frame, value in enumerate(values)
        )
        series = (
            f'<polyline points="{points}" fill="none" stroke="{_GREEN}" '
            'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle data-trace-dot="{index}" cx="{x_coord(0):.2f}" '
            f'cy="{y_coord(float(values[0])):.2f}" r="3" fill="{_GREEN}" '
            f'stroke="{_BACKGROUND}" stroke-width="1.5"/>'
        )

    return f"""
<svg data-force-trace="{index}" data-force-maximum="{maximum}"
  data-plot-left="{left}" data-plot-width="{plot_width}"
  data-plot-top="{top}" data-plot-height="{plot_height}"
  viewBox="0 0 {width} {height}"
  role="img" aria-label="Maximum force for {escape(str(item["name"]), quote=True)}"
  style="display:block;width:100%;height:auto">
  <rect width="{width}" height="{height}" fill="{_SURFACE}"/>
  {series}
  <line data-trace-cursor="{index}" x1="{x_coord(0):.2f}" y1="{top}" x2="{x_coord(0):.2f}"
    y2="{top + plot_height}" stroke="{_TEXT}" stroke-width="1" stroke-dasharray="3 3"/>
</svg>
"""


def _py3dmol_relaxation_player(
    items: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    frame_interval_ms: int,
    batch_multiplier: int,
) -> tuple[str, int]:
    """Build one isolated py3Dmol grid driven by a shared native frame index."""

    rows = 1
    row_height = 286
    stage_height = rows * row_height + 28
    grid_columns = f"repeat({MAX_RELAXATION_SYSTEMS},minmax(0,1fr))"
    cards = "".join(
        f'<section class="trajectory-card" data-panel="{index}" '
        f'data-atom-count="{item["atom_count"]}">'
        f'<div class="molecule-view" data-viewer="{index}" '
        f'aria-label="Trajectory for {escape(str(item["name"]), quote=True)}"></div>'
        f'<div class="card-title"><strong>{escape(str(item["name"]))}</strong>'
        f"<span>{item['atom_count']} atoms</span></div>"
        '<div class="force-panel"><span data-panel-status>'
        '<span class="metric"><b>MAX FORCE</b><em>…<i>eV/Å</i></em></span>'
        "</span>"
        f"{_force_sparkline_svg(item, index, frame_count)}</div></section>"
        for index, item in enumerate(items)
    )
    cards += "".join(
        '<section class="trajectory-card trajectory-placeholder" data-empty-slot>'
        '<div class="placeholder-message">Go back to the selector to choose '
        "another system.</div>"
        "</section>"
        for _ in range(MAX_RELAXATION_SYSTEMS - len(items))
    )
    total_graphs = len(items) * batch_multiplier
    total_atoms = sum(int(item["atom_count"]) for item in items) * batch_multiplier
    molecule_label = "molecule" if len(items) == 1 else "molecules"
    replica_label = "replica" if batch_multiplier == 1 else "replicas"
    graph_label = "graph" if total_graphs == 1 else "graphs"
    atom_label = "atom" if total_atoms == 1 else "atoms"
    batch_status = (
        f"Batch · {len(items)} {molecule_label} × {batch_multiplier} "
        f"{replica_label} = {total_graphs:,} {graph_label} · "
        f"{total_atoms:,} {atom_label}"
    )
    iframe_height = 56 + stage_height + 2

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{height:100%;overflow:hidden;margin:0;width:100%;background:{_BACKGROUND};color:{_TEXT};font-family:'NVIDIA Sans','NVIDIA Sans Regular',Arial,Helvetica,sans-serif}}
*{{box-sizing:border-box}}.player{{width:100%;height:100%;overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND}}}
.toolbar{{height:56px;padding:0 14px;display:flex;align-items:center;gap:16px;border-bottom:1px solid {_BORDER};background:{_SURFACE}}}
button{{min-width:92px;height:36px;padding:0 16px;border:0;border-radius:7px;background:{_GREEN};color:#11170D;font:inherit;font-size:.8rem;font-weight:700;cursor:pointer;transition:background .15s ease}}
button:hover{{background:#8CD400}}button:active{{background:#6AA700}}button:focus-visible,input:focus-visible{{outline:3px solid {_GREEN};outline-offset:2px}}
.batch-status{{flex:0 0 auto;padding:5px 8px;border:1px solid #3A444D;border-radius:5px;color:{_TEXT};font-size:.72rem;font-weight:650;white-space:nowrap}}
.frame-control{{flex:1;min-width:180px;display:flex;align-items:center;gap:14px;color:#8A939B;font-size:.66rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;white-space:nowrap}}
input{{accent-color:{_GREEN}}}.range-shell{{position:relative;display:flex;align-items:center;flex:1;height:18px;margin:0}}
.range-shell input{{position:absolute;z-index:2;inset:0;width:100%;height:18px;margin:0;opacity:0;cursor:pointer}}
.range-track{{position:absolute;z-index:1;left:7px;right:7px;top:7px;height:4px;border-radius:2px;background:#3A444D;pointer-events:none}}
.range-fill{{position:absolute;inset:0;border-radius:inherit;background:{_GREEN};transform:scaleX(0);transform-origin:left center;transition:transform 110ms linear}}
.range-thumb{{position:absolute;left:0;top:-5px;width:14px;height:14px;border-radius:50%;background:{_GREEN};box-shadow:0 0 0 3px {_SURFACE};transform:translate3d(-7px,0,0);transition:transform 110ms linear}}
.range-shell input:focus-visible~.range-track{{outline:3px solid {_GREEN};outline-offset:3px}}
.force-toggle{{display:flex;align-items:center;gap:7px;white-space:nowrap}}.force-toggle input{{width:16px;height:16px;margin:0}}
output{{flex:0 0 auto;text-align:right;font-size:.78rem;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}}
output .of{{color:#68737E;font-weight:500}}
.stage-grid{{display:grid;grid-template-columns:{grid_columns};grid-template-rows:repeat({rows},{row_height}px);justify-content:start;gap:12px;width:100%;height:{stage_height}px;padding:14px;overflow:hidden;background:{_BACKGROUND}}}
.trajectory-card{{position:relative;min-width:0;overflow:hidden;border:1px solid #3A444D;border-radius:8px;background:{_SURFACE}}}
.trajectory-placeholder{{display:grid;place-items:center;padding:16px;border-style:dashed;background:#1B2126}}
.placeholder-message{{max-width:120px;text-align:center;color:#8A939B;font-size:.66rem;line-height:1.4}}
.molecule-view{{position:absolute;inset:0 0 82px;margin:10px 14px 0;overflow:hidden;border-radius:5px;background:#12181A}}
.molecule-view canvas{{position:absolute!important;inset:0;width:100%!important;height:100%!important}}
.card-title{{position:absolute;z-index:3;top:12px;left:12px;display:flex;align-items:baseline;gap:7px;max-width:calc(100% - 24px);width:max-content;padding:4px 7px;border-radius:5px;overflow:hidden;white-space:nowrap;background:rgba(23,29,32,.86);pointer-events:none}}
.card-title strong{{overflow:hidden;text-overflow:ellipsis;font-size:.82rem}}.card-title span{{flex:0 0 auto;color:#A8B0B8;font-size:.68rem;font-weight:600}}
.force-panel{{position:absolute;left:0;right:0;bottom:0;height:82px;padding:0 14px 4px;border-top:1px solid #2C353B;background:{_SURFACE};pointer-events:auto;cursor:crosshair}}
[data-panel-status]{{height:36px;display:flex;align-items:flex-end;gap:20px;color:{_TEXT}}}
.metric{{display:flex;flex-direction:column;min-width:118px}}.metric b{{font-size:.6rem;font-weight:700;letter-spacing:.12em;color:#8A939B}}.metric em{{margin-top:4px;font-size:.94rem;font-weight:600;font-style:normal;font-variant-numeric:tabular-nums}}.metric i{{font-style:normal;font-size:.72rem;font-weight:500;color:#8A939B;margin-left:.25em}}
[data-trace-cursor]{{transition:x1 110ms linear,x2 110ms linear}}
[data-trace-dot]{{transition:cx 110ms linear,cy 110ms linear}}
@media(prefers-reduced-motion:reduce){{button,.range-fill,.range-thumb,[data-trace-cursor],[data-trace-dot]{{transition:none}}}}
</style></head><body>{_bundle_preamble()}
<main class="player" aria-label="Synchronized py3Dmol relaxation player" data-batch-multiplier="{batch_multiplier}" data-total-graphs="{total_graphs}" data-total-atoms="{total_atoms}">
  <div class="toolbar"><button type="button" data-play aria-label="Play relaxation">▶ Play</button>
    <span class="batch-status">{batch_status}</span>
    <label class="frame-control">Shared step<span class="range-shell" data-range-shell>
      <input data-frame aria-label="Shared relaxation step" type="range" min="0" max="{frame_count - 1}" value="0" step="1">
      <span class="range-track"><span class="range-fill" data-progress-fill></span><span class="range-thumb" data-progress-thumb></span></span>
    </span></label>
    <label class="force-toggle"><input data-force-color type="checkbox" checked>Color atoms by force</label>
    <output data-frame-label>Step 1<span class="of"> / {frame_count}</span></output></div>
  <div class="stage-grid">{cards}</div>
</main>
<script>$3Dmolpromise.then(function(){{
const items={_json_for_script(list(items))};
const viewers=items.map(function(item,index){{
  const node=document.querySelector('[data-viewer="'+index+'"]');
  const viewer=$3Dmol.createViewer(node,{{backgroundColor:'{_BACKGROUND}',antialias:true}});
  viewer.addModelsAsFrames(item.trajectory_xyz,'xyz',{{assignBonds:!item.sphere_only}});
  item.cell_edges.forEach(function(edge){{viewer.addLine({{start:{{x:edge.start[0],y:edge.start[1],z:edge.start[2]}},end:{{x:edge.end[0],y:edge.end[1],z:edge.end[2]}},color:'{_BORDER}'}});}});
  viewer.zoomTo();viewer.rotate(-18,'y');viewer.rotate(12,'x');viewer.render();return viewer;
}});
const slider=document.querySelector('[data-frame]');const play=document.querySelector('[data-play]');
const forceToggle=document.querySelector('[data-force-color]');
const rangeShell=document.querySelector('[data-range-shell]');const progressFill=document.querySelector('[data-progress-fill]');const progressThumb=document.querySelector('[data-progress-thumb]');
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const frameInterval=reduced?Math.max(240,{frame_interval_ms}):{frame_interval_ms};
const elementColors={_json_for_script(_ELEMENT_COLORS)};
const scales=items.map(function(item){{return item.atom_forces[0].length?Math.max(...item.atom_forces[0]):1;}});
let animationFrame=null;let lastFrameTime=null;let frameQueue=Promise.resolve();let rendering=false;
const lerp=function(a,b,t){{return Math.round(a+(b-a)*t);}};
const forceColor=function(value,scale){{
  const t=Math.max(0,Math.min(1,value/Math.max(scale,1e-12)));
  const a=t<.5?[118,185,0]:[242,201,76];const b=t<.5?[242,201,76]:[224,82,82];const u=t<.5?t*2:(t-.5)*2;
  return 'rgb('+lerp(a[0],b[0],u)+','+lerp(a[1],b[1],u)+','+lerp(a[2],b[2],u)+')';
}};
const colorByElement=function(viewer,item){{
  const scale=item.sphere_only ? .38 : .23;
  viewer.setStyle({{}},{{sphere:{{scale:scale,color:'{_GREEN}'}},...(item.sphere_only?{{}}:{{stick:{{radius:.09,color:'{_GREEN}'}}}})}});
  Object.keys(elementColors).forEach(function(symbol){{const color=elementColors[symbol];viewer.setStyle({{elem:symbol}},{{sphere:{{scale:scale,color:color}},...(item.sphere_only?{{}}:{{stick:{{radius:.09,color:color}}}})}});}});
}};
const colorAtoms=function(viewer,item,magnitudes,forceScale){{
  if(!forceToggle.checked||!magnitudes.length){{colorByElement(viewer,item);return;}}
  const atomScale=item.sphere_only ? .38 : .23;
  magnitudes.forEach(function(value,atom){{const color=forceColor(value,forceScale);viewer.setStyle({{serial:atom}},{{sphere:{{scale:atomScale,color:color}},...(item.sphere_only?{{}}:{{stick:{{radius:.09,color:color}}}})}});}});
}};
const updateProgress=function(index){{
  const fraction=index/Math.max({frame_count - 1},1);const travel=Math.max(0,rangeShell.clientWidth-14);
  progressFill.style.transform='scaleX('+fraction+')';progressThumb.style.transform='translate3d('+(fraction*travel-7)+'px,0,0)';
}};
const update=function(index){{
  rendering=true;frameQueue=frameQueue.then(function(){{return Promise.all(viewers.map(function(viewer){{return viewer.setFrame(index);}}));}}).then(function(){{
    viewers.forEach(function(viewer,panel){{colorAtoms(viewer,items[panel],items[panel].atom_forces[index],scales[panel]);viewer.render();}});
    items.forEach(function(item,panel){{const value=item.forces[index];
      document.querySelector('[data-panel="'+panel+'"] [data-panel-status]').innerHTML=value===null?'<span class="metric"><b>FRAME</b><em>'+(index+1)+'</em></span>':'<span class="metric"><b>MAX FORCE</b><em>'+value.toFixed(3)+'<i>eV/Å</i></em></span>';
      const trace=document.querySelector('[data-force-trace="'+panel+'"]');const cursor=document.querySelector('[data-trace-cursor="'+panel+'"]');const dot=document.querySelector('[data-trace-dot="'+panel+'"]');
      const plotLeft=Number(trace.dataset.plotLeft);const plotWidth=Number(trace.dataset.plotWidth);
      const plotTop=Number(trace.dataset.plotTop);const plotHeight=Number(trace.dataset.plotHeight);
      const x=plotLeft+plotWidth*index/Math.max({frame_count - 1},1);cursor.style.setProperty('x1',x+'px');cursor.style.setProperty('x2',x+'px');
      if(dot&&value!==null){{dot.style.setProperty('cx',x+'px');dot.style.setProperty('cy',(plotTop+plotHeight*(1-value/Number(trace.dataset.forceMaximum)))+'px');}}
    }});
    slider.value=String(index);updateProgress(index);document.querySelector('[data-frame-label]').innerHTML='Step '+(index+1)+'<span class="of"> / {frame_count}</span>';
  }}).finally(function(){{rendering=false;}});return frameQueue;
}};
const stop=function(){{if(animationFrame!==null)window.cancelAnimationFrame(animationFrame);animationFrame=null;lastFrameTime=null;play.textContent='▶ Play';}};
const tick=function(time){{if(animationFrame===null)return;if(!rendering&&(lastFrameTime===null||time-lastFrameTime>=frameInterval)){{update((Number(slider.value)+1)%{frame_count});lastFrameTime=time;}}animationFrame=window.requestAnimationFrame(tick);}};
play.addEventListener('click',function(){{if(animationFrame!==null){{stop();return;}}play.textContent='❚❚ Pause';animationFrame=window.requestAnimationFrame(tick);}});
slider.addEventListener('input',function(){{stop();update(Number(slider.value));}});
forceToggle.addEventListener('change',function(){{update(Number(slider.value));}});
window.addEventListener('resize',function(){{updateProgress(Number(slider.value));}});
document.querySelectorAll('[data-force-trace]').forEach(function(trace){{trace.addEventListener('pointerdown',function(event){{const box=trace.getBoundingClientRect();const svgX=(event.clientX-box.left)/box.width*300;const plotLeft=Number(trace.dataset.plotLeft);const plotWidth=Number(trace.dataset.plotWidth);const fraction=Math.max(0,Math.min(1,(svgX-plotLeft)/plotWidth));stop();update(Math.round(fraction*{frame_count - 1}));}});}});
update(0);
}});</script></body></html>"""
    iframe = _player_iframe(
        document,
        title="Synchronized py3Dmol relaxation player",
        height=iframe_height,
    )
    return iframe, iframe_height


def show_relaxation_player(
    trajectories: Mapping[str, Sequence[Any]] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    force_history: Mapping[str, Sequence[float]] | None = None,
    device: str = "device not recorded",
    elapsed_s: float | None = None,
    convergence: Mapping[str, str] | str | None = None,
    batch_multiplier: int = 1,
    frame_interval_ms: int = 120,
    reference_structures: Mapping[str, Atoms] | Sequence[Atoms] | None = None,
) -> Any:
    """Show synchronized relaxation frames and their maximum-force traces."""

    atom_force_history = _normalise_atom_force_magnitudes(trajectories, labels)
    normalised = _normalise_trajectories(trajectories, labels)
    if len(normalised) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            "the Core relaxation player supports at most "
            f"{MAX_RELAXATION_SYSTEMS} systems"
        )
    if (
        isinstance(batch_multiplier, bool)
        or not isinstance(batch_multiplier, int)
        or batch_multiplier < 1
    ):
        raise ValueError("batch_multiplier must be a positive integer")
    if (
        isinstance(frame_interval_ms, bool)
        or not isinstance(frame_interval_ms, int)
        or frame_interval_ms < 16
    ):
        raise ValueError("frame_interval_ms must be an integer of at least 16")
    names = list(normalised)
    total_graphs = len(names) * batch_multiplier
    atom_counts = {name: len(frames[0]) for name, frames in normalised.items()}
    total_atoms = sum(atom_counts.values()) * batch_multiplier
    molecule_label = "molecule" if len(names) == 1 else "molecules"
    replica_label = "replica" if batch_multiplier == 1 else "replicas"
    graph_label = "graph" if total_graphs == 1 else "graphs"
    atom_label = "atom" if total_atoms == 1 else "atoms"
    batch_status = (
        f"Batch · {len(names)} {molecule_label} × {batch_multiplier} "
        f"{replica_label} = {total_graphs:,} {graph_label} · "
        f"{total_atoms:,} {atom_label}"
    )
    references = _reference_structure_map(reference_structures, names)
    frame_count = len(next(iter(normalised.values())))
    forces = _force_values(force_history, names, frame_count)
    items = []
    for name, frames in normalised.items():
        rendering_reference = references.get(name, frames[0])
        rendering_payload = _structure_payload(rendering_reference)
        atom_forces = (
            atom_force_history[name]
            if atom_force_history is not None
            else [[] for _ in range(frame_count)]
        )
        items.append(
            {
                "name": name,
                "atom_count": atom_counts[name],
                "trajectory_xyz": "".join(_atoms_to_xyz(frame) for frame in frames),
                "cell_edges": rendering_payload["cell_edges"],
                "sphere_only": rendering_payload["sphere_only"],
                "forces": forces[name],
                "atom_forces": atom_forces,
            }
        )
    del device, elapsed_s, convergence
    root_id = f"alchemi-relax-{uuid4().hex}"
    static_card_parts = []
    for name, frames in normalised.items():
        values = forces[name]
        force_text = (
            "force history not recorded"
            if values[0] is None
            else f"fmax {values[0]:.3f} → {values[-1]:.3f} eV/Å"
        )
        static_card_parts.append(
            f'<section style="min-width:0;padding:0 10px 12px;overflow:hidden;'
            f'border-right:1px solid {_BORDER};border-bottom:1px solid {_BORDER}">'
            + _static_pair_svg(
                frames[0], frames[-1], title=f"{name}: initial and final"
            )
            + '<div style="padding:8px 2px 0;font-size:.78rem">'
            + f"{escape(name)} · {atom_counts[name]} atoms · "
            + f"{escape(force_text)}</div></section>"
        )
    static_cards = "".join(static_card_parts)
    player_iframe, _ = _py3dmol_relaxation_player(
        items,
        frame_count=frame_count,
        frame_interval_ms=frame_interval_ms,
        batch_multiplier=batch_multiplier,
    )
    markup = f"""
<div id="{root_id}" aria-label="Synchronized relaxation player" data-batch-multiplier="{batch_multiplier}" data-total-graphs="{total_graphs}" data-total-atoms="{total_atoms}" style="max-width:none">
  {_common_css(root_id)}
  <div data-static-fallback style="padding:12px">
    {runtime_required_notice()}
    <p style="margin:0 2px 10px;font-size:.82rem;font-weight:650">{batch_status}</p>
    <div class="alchemi-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0">{static_cards}</div>
    <p class="alchemi-muted" style="margin:10px 2px 0;font-size:.82rem">Static initial/final preview. Run this cell in Jupyter to play synchronized frames.</p>
  </div>
  <div data-interactive>{player_iframe}</div>
</div>
<script>(function(){{
const root=document.getElementById('{root_id}');
if(root&&root.querySelector('iframe'))root.classList.add('is-live');
}})();</script>
"""
    return _html(markup)


def _fragment_com_line(atoms: Atoms, fragment_a_size: int) -> dict[str, Any]:
    """Return the mass-weighted fragment centers and their separation."""

    if (
        isinstance(fragment_a_size, bool)
        or not isinstance(fragment_a_size, int)
        or not 0 < fragment_a_size < len(atoms)
    ):
        raise ValueError("fragment split must leave at least one atom in each fragment")
    masses = np.asarray(atoms.get_masses(), dtype=float)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("fragment centers require finite positive atomic masses")
    positions = np.asarray(atoms.positions, dtype=float)
    first = np.average(
        positions[:fragment_a_size], axis=0, weights=masses[:fragment_a_size]
    )
    second = np.average(
        positions[fragment_a_size:], axis=0, weights=masses[fragment_a_size:]
    )
    return {
        "start": first.tolist(),
        "end": second.tolist(),
        "distance_a": float(np.linalg.norm(second - first)),
    }


def _closest_interfragment_contact(
    atoms: Atoms, fragment_a_size: int
) -> dict[str, Any]:
    """Return the closest atom pair across two contiguous fragments."""

    if (
        isinstance(fragment_a_size, bool)
        or not isinstance(fragment_a_size, int)
        or not 0 < fragment_a_size < len(atoms)
    ):
        raise ValueError("fragment split must leave at least one atom in each fragment")
    positions = np.asarray(atoms.positions, dtype=float)
    displacements = (
        positions[:fragment_a_size, None, :] - positions[None, fragment_a_size:, :]
    )
    distances = np.linalg.norm(displacements, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("closest-contact distances must be finite")
    local_a, local_b = np.unravel_index(int(np.argmin(distances)), distances.shape)
    atom_b = int(local_b + fragment_a_size)
    return {
        "start": positions[local_a].tolist(),
        "end": positions[atom_b].tolist(),
        "distance_a": float(distances[local_a, local_b]),
        "atom_indices": [int(local_a), atom_b],
    }


def _source_coordinate_text(scale: float) -> str:
    """Label one NCI Atlas source interaction coordinate."""

    suffix = " (equilibrium)" if np.isclose(scale, 1.0) else ""
    return f"Source coordinate s = {float(scale):.2f}{suffix}"


def _interpolate_nci_display_frames(
    trajectories: Mapping[str, Sequence[Atoms]],
    scales: np.ndarray,
    *,
    maximum_scale_step: float = 0.01,
) -> tuple[dict[str, list[Atoms]], np.ndarray]:
    """Add display-only frames between the calculated NCI scan geometries."""

    segment_steps = [
        max(1, int(np.ceil((stop - start) / maximum_scale_step - 1.0e-12)))
        for start, stop in pairwise(scales)
    ]
    display_scales = [float(scales[0])]
    display_frames = {name: [frames[0].copy()] for name, frames in trajectories.items()}

    for source_index, steps in enumerate(segment_steps):
        start_scale = float(scales[source_index])
        stop_scale = float(scales[source_index + 1])
        for step in range(1, steps + 1):
            fraction = step / steps
            display_scales.append(start_scale + fraction * (stop_scale - start_scale))
            for name, frames in trajectories.items():
                start = frames[source_index]
                stop = frames[source_index + 1]
                if start.get_chemical_symbols() != stop.get_chemical_symbols():
                    raise ValueError(f"NCI scan atom order changes for {name!r}")
                if step == steps:
                    display_frames[name].append(stop.copy())
                    continue
                frame = start.copy()
                frame.positions = (1.0 - fraction) * np.asarray(
                    start.positions
                ) + fraction * np.asarray(stop.positions)
                display_frames[name].append(frame)

    return display_frames, np.asarray(display_scales, dtype=float)


def _normalise_fragment_sizes(
    fragment_sizes: Mapping[str, int] | Sequence[int], names: Sequence[str]
) -> dict[str, int]:
    """Align one fragment-A size with each named NCI scan."""

    if isinstance(fragment_sizes, Mapping):
        missing = [name for name in names if name not in fragment_sizes]
        extra = [name for name in fragment_sizes if name not in names]
        if missing or extra:
            raise ValueError(
                "fragment_sizes must contain exactly one entry per NCI system"
            )
        return {name: fragment_sizes[name] for name in names}
    values = list(fragment_sizes)
    if len(values) != len(names):
        raise ValueError("fragment_sizes must contain one value per NCI system")
    return dict(zip(names, values, strict=True))


def _static_nci_frame_svg(
    atoms: Atoms,
    line: Mapping[str, Any],
    *,
    state_label: str,
    width: int = 240,
    height: int = 170,
) -> str:
    """Draw one static NCI frame with the same COM separation as the live view."""

    positions = np.asarray(atoms.positions, dtype=float)
    origin = positions.mean(axis=0)
    rotation = _viewing_rotation(positions)
    projected_atoms = _to_view_frame(positions, origin, rotation)[:, :2]
    projected_line = _to_view_frame(
        np.asarray([line["start"], line["end"]]), origin, rotation
    )[:, :2]
    all_points = np.concatenate([projected_atoms, projected_line])
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    extent = upper - lower
    scale_factor = min(
        (width - 34) / max(float(extent[0]), 1.0),
        (height - 76) / max(float(extent[1]), 1.0),
    )

    def screen(points: np.ndarray) -> np.ndarray:
        centered = points - (lower + upper) / 2
        result = centered * scale_factor
        result[:, 0] += width / 2
        result[:, 1] = height / 2 - result[:, 1] + 17
        return result

    atom_xy = screen(projected_atoms)
    line_xy = screen(projected_line)
    atom_marks = []
    for symbol, (x_coord, y_coord) in zip(
        atoms.get_chemical_symbols(), atom_xy, strict=True
    ):
        atom_marks.append(
            f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="5.2" '
            f'fill="{_ELEMENT_COLORS.get(symbol, _GREEN)}" stroke="{_BACKGROUND}" '
            f'stroke-width="1"><title>{escape(symbol)}</title></circle>'
        )
    distance = float(line["distance_a"])
    delta_distance = float(line["delta_distance_a"])
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(state_label, quote=True)} NCI structure at closest '
        f"contact distance "
        f'{distance:.3f} angstrom" '
        'style="display:block;width:100%;height:auto">'
        f'<rect width="{width}" height="{height}" rx="7" fill="{_BACKGROUND}"/>'
        f'<text x="12" y="18" fill="{_TEXT}" font-size="11" font-weight="650">'
        f"{escape(state_label)}</text>"
        f'<text x="12" y="36" fill="{_MUTED}" font-size="9.5">'
        f"D min = {distance:.3f} Å; ΔD = {delta_distance:+.3f} Å</text>"
        f'<line x1="{line_xy[0, 0]:.1f}" y1="{line_xy[0, 1]:.1f}" '
        f'x2="{line_xy[1, 0]:.1f}" y2="{line_xy[1, 1]:.1f}" '
        f'stroke="{_BLUE}" stroke-width="2" stroke-dasharray="3 4"/>'
        + "".join(atom_marks)
        + "</svg>"
    )


def _py3dmol_nci_player(
    items: Sequence[Mapping[str, Any]],
    *,
    scales: Sequence[float],
    source_scales: Sequence[float],
    frame_interval_ms: int,
) -> str:
    """Build three synchronized NCI cards driven by one separation control."""

    frame_count = len(scales)
    cards = "".join(
        f'<section class="nci-card" data-panel="{index}">'
        f'<div class="molecule-view" data-viewer="{index}" '
        f'aria-label="NCI scan for {escape(str(item["name"]), quote=True)}"></div>'
        f'<strong class="card-title">{escape(str(item["name"]))}</strong>'
        '<div class="separation-readout" data-separation-readout>'
        '<span class="metric"><b>D MIN</b><em>…<i>Å</i></em></span>'
        '<span class="metric"><b>ΔD</b><em>…<i>Å</i></em></span>'
        "</div></section>"
        for index, item in enumerate(items)
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;background:{_BACKGROUND};color:{_TEXT};font-family:'NVIDIA Sans','NVIDIA Sans Regular',Arial,Helvetica,sans-serif}}
*{{box-sizing:border-box}}.player{{width:100%;overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND}}}
.toolbar{{height:56px;padding:0 14px;display:flex;align-items:center;gap:16px;border-bottom:1px solid {_BORDER};background:{_SURFACE}}}
button{{min-width:92px;height:36px;padding:0 16px;border:0;border-radius:7px;background:{_GREEN};color:#11170D;font:inherit;font-size:.8rem;font-weight:700;cursor:pointer;transition:background .15s ease}}
button:hover{{background:#8CD400}}button:active{{background:#6AA700}}button:focus-visible,input:focus-visible{{outline:3px solid {_GREEN};outline-offset:2px}}
label{{flex:1;min-width:180px;display:flex;align-items:center;gap:14px;color:#8A939B;font-size:.66rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;white-space:nowrap}}input{{width:100%;margin:0;accent-color:{_GREEN}}}
.stage-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(0,1fr));gap:12px;padding:14px;background:{_BACKGROUND}}}
.nci-card{{position:relative;min-width:0;height:340px;overflow:hidden;border:1px solid #3A444D;border-radius:8px;background:{_SURFACE}}}
.molecule-view{{position:absolute;inset:0 0 60px;margin:10px 14px 0;border-radius:5px;overflow:hidden;background:#12181A}}.molecule-view canvas{{position:absolute!important;inset:0;width:100%!important;height:100%!important}}
.card-title{{position:absolute;z-index:3;top:12px;left:12px;display:block;max-width:calc(100% - 24px);padding:4px 7px;border-radius:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:rgba(23,29,32,.86);font-size:.82rem;pointer-events:none}}
.separation-readout{{position:absolute;left:0;right:0;bottom:0;height:60px;padding:10px 14px;border-top:1px solid #2C353B;background:{_SURFACE};display:flex;align-items:flex-end;gap:20px;color:{_TEXT}}}
.metric{{display:flex;flex-direction:column;min-width:92px}}.metric b{{font-size:.6rem;font-weight:700;letter-spacing:.12em;color:#8A939B}}.metric em{{margin-top:4px;font-size:.94rem;font-weight:600;font-style:normal;font-variant-numeric:tabular-nums}}.metric i{{font-style:normal;font-size:.72rem;font-weight:500;color:#8A939B;margin-left:.25em}}
@media(max-width:700px){{.stage-grid{{grid-template-columns:1fr}}.nci-card{{height:340px}}}}
</style></head><body>{_bundle_preamble()}
<main class="player" aria-label="Synchronized py3Dmol NCI separation player">
  <div class="toolbar"><button type="button" data-play aria-label="Play NCI separation scan">▶ Play</button>
    <label>Shared NCI Atlas coordinate
      <input data-frame aria-label="Shared NCI Atlas scan coordinate" type="range" min="0" max="{frame_count - 1}" value="0" step="1">
    </label>
  </div>
  <div class="stage-grid">{cards}</div>
</main>
<script>$3Dmolpromise.then(function(){{
const items={_json_for_script(list(items))};const scales={_json_for_script(list(scales))};const sourceScales={_json_for_script(list(source_scales))};
{_trajectory_viewers_javascript()}
const slider=document.querySelector('[data-frame]');const play=document.querySelector('[data-play]');let timer=null;let queue=Promise.resolve();
const point=function(values){{return {{x:values[0],y:values[1],z:values[2]}};}};
const update=function(index){{queue=queue.then(function(){{return Promise.all(viewers.map(function(viewer){{return viewer.setFrame(index);}}));}}).then(function(){{viewers.forEach(function(viewer,panel){{const line=items[panel].contact_lines[index];viewer.removeAllShapes();viewer.addLine({{start:point(line.start),end:point(line.end),color:'{_BLUE}',dashed:true,dashLength:.12,gapLength:.08}});applyStyle(viewer);viewer.render();document.querySelector('[data-panel="'+panel+'"] [data-separation-readout]').innerHTML='<span class="metric"><b>D MIN</b><em>'+line.distance_a.toFixed(3)+'<i>Å</i></em></span><span class="metric"><b>ΔD</b><em>'+(line.delta_distance_a>=0?'+':'')+line.delta_distance_a.toFixed(3)+'<i>Å</i></em></span>';}});slider.value=String(index);}});return queue;}};
const stop=function(){{if(timer!==null)window.clearTimeout(timer);timer=null;play.textContent='▶ Play';}};
const advance=function(){{if(timer===null)return;update((Number(slider.value)+1)%{frame_count}).then(function(){{if(timer!==null)timer=window.setTimeout(advance,{frame_interval_ms});}});}};
play.addEventListener('click',function(){{if(timer!==null){{stop();return;}}play.textContent='❚❚ Pause';timer=window.setTimeout(advance,{frame_interval_ms});}});
slider.addEventListener('input',function(){{stop();update(Number(slider.value));}});update(0);
}});</script></body></html>"""
    return _player_iframe(
        document,
        title="Synchronized py3Dmol NCI separation player",
        height=440,
    )


def show_nci_separation_player(
    grouped_frames: Mapping[str, Sequence[Any]],
    fragment_sizes: Mapping[str, int] | Sequence[int],
    scale_values: Sequence[float],
    *,
    frame_interval_ms: int = 650,
) -> Any:
    """Show aligned NCI scans with closest contacts and one shared control.

    ``grouped_frames`` holds one ordered scan per named complex. Each
    ``fragment_sizes`` value is the number of leading atoms in fragment A.
    ``scale_values`` supplies the NCI Atlas source interaction coordinate for
    every frame. The frame at scale 1 defines each complex's reference closest
    contact.
    Linear display frames at one-percent scale increments smooth the player;
    every supplied scan geometry remains an exact frame.
    """

    normalised = _normalise_trajectories(grouped_frames, labels=None)
    if len(normalised) != 3:
        raise ValueError("the Core NCI player requires exactly three systems")
    names = list(normalised)
    sizes = _normalise_fragment_sizes(fragment_sizes, names)
    scales = np.asarray(scale_values, dtype=float).reshape(-1)
    frame_count = len(next(iter(normalised.values())))
    if scales.size != frame_count:
        raise ValueError("scale_values must contain one value per aligned frame")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("scale_values must contain finite positive values")
    if np.any(np.diff(scales) <= 0):
        raise ValueError("scale_values must be strictly increasing")
    equilibrium_indices = np.flatnonzero(np.isclose(scales, 1.0))
    if equilibrium_indices.size != 1:
        raise ValueError("scale_values must contain one reference value at scale 1")
    if (
        isinstance(frame_interval_ms, bool)
        or not isinstance(frame_interval_ms, int)
        or frame_interval_ms < 16
    ):
        raise ValueError("frame_interval_ms must be an integer of at least 16")

    source_frame_count = frame_count
    source_scales = scales.copy()
    normalised, scales = _interpolate_nci_display_frames(normalised, scales)
    frame_count = len(scales)
    equilibrium_index = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    items = []
    for name, frames in normalised.items():
        lines = [_closest_interfragment_contact(frame, sizes[name]) for frame in frames]
        reference_distance = float(lines[equilibrium_index]["distance_a"])
        for line in lines:
            line["delta_distance_a"] = float(line["distance_a"] - reference_distance)
        items.append(
            {
                "name": name,
                "trajectory_xyz": "".join(_atoms_to_xyz(frame) for frame in frames),
                "contact_lines": lines,
            }
        )
    fallback_states = (
        ("initial", 0),
        ("equilibrium", equilibrium_index),
        ("far", frame_count - 1),
    )
    static_sections = []
    for item, (name, frames) in zip(items, normalised.items(), strict=True):
        previews = "".join(
            _static_nci_frame_svg(
                frames[index],
                item["contact_lines"][index],
                state_label=state_label,
            )
            for state_label, index in fallback_states
        )
        static_sections.append(
            f'<section style="padding:10px;border-bottom:1px solid {_BORDER}">'
            f'<strong style="display:block;margin:0 2px 8px;font-size:.84rem">'
            f'{escape(name)}</strong><div style="display:grid;grid-template-columns:'
            f'repeat(3,minmax(0,1fr));gap:8px">{previews}</div></section>'
        )

    root_id = f"alchemi-nci-{uuid4().hex}"
    display_interval_ms = max(
        32,
        round(frame_interval_ms * source_frame_count / frame_count),
    )
    player = _py3dmol_nci_player(
        items,
        scales=scales.astype(float).tolist(),
        source_scales=source_scales.astype(float).tolist(),
        frame_interval_ms=display_interval_ms,
    )
    markup = f"""
<div id="{root_id}" aria-label="Synchronized NCI separation player">
  {_common_css(root_id)}
  <div data-static-fallback style="padding:2px 12px 12px">
    {runtime_required_notice()}
    {"".join(static_sections)}
  </div>
  <div data-interactive>{player}</div>
</div>
<script>(function(){{const root=document.getElementById('{root_id}');if(root&&root.querySelector('iframe'))root.classList.add('is-live');}})();</script>
"""
    return _html(markup)


def _py3dmol_bias_comparison_player(
    items: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    frame_interval_ms: int,
) -> str:
    """Build a two-condition, three-system player with one frame control."""

    cards = "".join(
        f'<section class="trajectory-card" data-panel="{index}">'
        f'<div class="molecule-view" data-viewer="{index}" '
        f'aria-label="{escape(str(item["condition"]), quote=True)} trajectory for '
        f'{escape(str(item["name"]), quote=True)}"></div>'
        f'<strong class="card-title">{escape(str(item["condition"]))}</strong>'
        f'<span class="card-sub">{escape(str(item["name"]))}</span>'
        '<div class="trajectory-readout" data-trajectory-readout>'
        '<span class="metric"><b>SEPARATION</b><em>…<i>Å</i></em></span>'
        '<span class="metric"><b>MAX BIAS FORCE</b><em>…<i>eV/Å</i></em></span>'
        "</div></section>"
        for index, item in enumerate(items)
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;background:{_BACKGROUND};color:{_TEXT};font-family:'NVIDIA Sans','NVIDIA Sans Regular',Arial,Helvetica,sans-serif}}
*{{box-sizing:border-box}}.player{{width:100%;overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND}}}
.toolbar{{height:56px;padding:0 14px;display:flex;align-items:center;gap:16px;border-bottom:1px solid {_BORDER};background:{_SURFACE}}}
button{{min-width:92px;height:36px;padding:0 16px;border:0;border-radius:7px;background:{_GREEN};color:#11170D;font:inherit;font-size:.8rem;font-weight:700;cursor:pointer;transition:background .15s ease}}
button:hover{{background:#8CD400}}button:active{{background:#6AA700}}button:focus-visible,input:focus-visible{{outline:3px solid {_GREEN};outline-offset:2px}}
label{{flex:1;min-width:180px;display:flex;align-items:center;gap:14px;color:#8A939B;font-size:.66rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;white-space:nowrap}}input{{width:100%;margin:0;accent-color:{_GREEN}}}
output{{flex:0 0 auto;text-align:right;font-size:.78rem;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}}output .of{{color:#68737E;font-weight:500}}
.stage-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(2,278px);gap:12px;padding:14px;background:{_BACKGROUND}}}
.trajectory-card{{position:relative;min-width:0;overflow:hidden;border:1px solid #3A444D;border-radius:8px;background:{_SURFACE}}}
.molecule-view{{position:absolute;inset:0 0 66px;margin:10px 14px 0;border-radius:5px;overflow:hidden;background:#12181A}}.molecule-view canvas{{position:absolute!important;inset:0;width:100%!important;height:100%!important}}
.card-title{{position:absolute;z-index:3;top:12px;left:12px;display:block;max-width:calc(100% - 24px);padding:4px 7px;border-radius:5px;background:rgba(23,29,32,.86);font-size:.8rem;pointer-events:none}}
.card-sub{{position:absolute;z-index:3;top:42px;left:12px;display:block;max-width:calc(100% - 24px);padding:2px 7px;border-radius:4px;background:rgba(23,29,32,.82);color:#8A939B;font-size:.7rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;pointer-events:none}}
.trajectory-readout{{position:absolute;left:0;right:0;bottom:0;height:66px;padding:10px 14px;border-top:1px solid #2C353B;background:{_SURFACE};display:flex;align-items:flex-end;gap:18px;color:{_TEXT}}}
.metric{{display:flex;flex-direction:column;min-width:92px}}.metric b{{font-size:.58rem;font-weight:700;letter-spacing:.1em;color:#8A939B}}.metric em{{margin-top:4px;font-size:.9rem;font-weight:600;font-style:normal;font-variant-numeric:tabular-nums}}.metric i{{font-style:normal;font-size:.68rem;font-weight:500;color:#8A939B;margin-left:.25em}}
</style></head><body>{_bundle_preamble()}
<main class="player" aria-label="Matched molecular dynamics comparison">
  <div class="toolbar"><button type="button" data-play aria-label="Play matched trajectories">▶ Play</button>
    <label>Shared trajectory frame
      <input data-frame aria-label="Shared trajectory frame" type="range" min="0" max="{frame_count - 1}" value="0" step="1">
    </label>
    <output data-frame-label>1<span class="of"> / {frame_count}</span></output>
  </div>
  <div class="stage-grid">{cards}</div>
</main>
<script>$3Dmolpromise.then(function(){{
const items={_json_for_script(list(items))};
{_trajectory_viewers_javascript()}
const slider=document.querySelector('[data-frame]');const play=document.querySelector('[data-play]');let timer=null;let queue=Promise.resolve();
const point=function(values){{return {{x:values[0],y:values[1],z:values[2]}};}};
const update=function(index){{queue=queue.then(function(){{return Promise.all(viewers.map(function(viewer){{return viewer.setFrame(index);}}));}}).then(function(){{viewers.forEach(function(viewer,panel){{const item=items[panel],line=item.com_lines[index];viewer.removeAllShapes();viewer.addLine({{start:point(line.start),end:point(line.end),color:'{_MUTED}',dashed:true,dashLength:.12,gapLength:.08}});applyStyle(viewer);viewer.render();document.querySelector('[data-panel="'+panel+'"] [data-trajectory-readout]').innerHTML='<span class="metric"><b>SEPARATION</b><em>'+line.distance_a.toFixed(3)+'<i>Å</i></em></span><span class="metric"><b>MAX BIAS FORCE</b><em>'+item.maximum_bias_force[index].toFixed(3)+'<i>eV/Å</i></em></span>';}});slider.value=String(index);document.querySelector('[data-frame-label]').innerHTML=(index+1)+'<span class="of"> / {frame_count}</span>';}});return queue;}};
const stop=function(){{if(timer!==null)window.clearInterval(timer);timer=null;play.textContent='▶ Play';}};
play.addEventListener('click',function(){{if(timer!==null){{stop();return;}}play.textContent='❚❚ Pause';timer=window.setInterval(function(){{update((Number(slider.value)+1)%{frame_count});}},{frame_interval_ms});}});
slider.addEventListener('input',function(){{stop();update(Number(slider.value));}});update(0);
}});</script></body></html>"""
    return _player_iframe(
        document,
        title="Matched molecular dynamics comparison",
        height=660,
    )


def show_bias_comparison_player(
    unrestrained_frames: Mapping[str, Sequence[Any]] | Sequence[Any],
    restrained_frames: Mapping[str, Sequence[Any]] | Sequence[Any],
    *,
    labels: Sequence[str],
    fragment_sizes: Mapping[str, int] | Sequence[int],
    reference_distances: Sequence[float],
    force_constant_ev_per_a2: float,
    frame_interval_ms: int = 250,
) -> Any:
    """Compare three matched MD trajectories with and without a COM bias."""

    names = [str(label) for label in labels]
    unrestrained = _normalise_trajectories(unrestrained_frames, names)
    restrained = _normalise_trajectories(restrained_frames, names)
    if list(unrestrained) != names or list(restrained) != names or len(names) != 3:
        raise ValueError("labels must identify the same three systems in both runs")
    frame_counts = {
        *(len(frames) for frames in unrestrained.values()),
        *(len(frames) for frames in restrained.values()),
    }
    if len(frame_counts) != 1:
        raise ValueError("both conditions must contain the same frame count")
    frame_count = frame_counts.pop()
    sizes = _normalise_fragment_sizes(fragment_sizes, names)
    references = np.asarray(reference_distances, dtype=float).reshape(-1)
    if references.size != len(names) or not np.isfinite(references).all():
        raise ValueError("reference_distances must contain one finite value per system")
    if np.any(references <= 0):
        raise ValueError("reference_distances must be positive")
    if not np.isfinite(force_constant_ev_per_a2) or force_constant_ev_per_a2 <= 0:
        raise ValueError("force_constant_ev_per_a2 must be positive and finite")
    if (
        isinstance(frame_interval_ms, bool)
        or not isinstance(frame_interval_ms, int)
        or frame_interval_ms < 16
    ):
        raise ValueError("frame_interval_ms must be an integer of at least 16")

    items = []
    for condition, trajectories, has_bias in (
        ("Without bias", unrestrained, False),
        ("With BiasedPotentialHook", restrained, True),
    ):
        for system_index, name in enumerate(names):
            frames = trajectories[name]
            lines = [_fragment_com_line(frame, sizes[name]) for frame in frames]
            if has_bias:
                force = force_constant_ev_per_a2 * np.abs(
                    np.asarray([line["distance_a"] for line in lines])
                    - references[system_index]
                )
                maximum_bias_force = np.maximum.accumulate(force).tolist()
            else:
                maximum_bias_force = [0.0] * frame_count
            items.append(
                {
                    "condition": condition,
                    "name": name,
                    "trajectory_xyz": "".join(_atoms_to_xyz(frame) for frame in frames),
                    "com_lines": lines,
                    "maximum_bias_force": maximum_bias_force,
                }
            )

    static_cards = []
    for item, condition_data in zip(
        items,
        [*unrestrained.items(), *restrained.items()],
        strict=True,
    ):
        name, frames = condition_data
        final_line = item["com_lines"][-1]
        final_force = item["maximum_bias_force"][-1]
        static_cards.append(
            f'<section style="min-width:0;padding:10px;border:1px solid {_BORDER};'
            f'border-radius:8px;background:{_SURFACE}"><strong style="display:block;'
            f'margin-bottom:8px;font-size:.78rem">{escape(str(item["condition"]))}'
            f'<br><span style="color:{_MUTED};font-weight:500">{escape(name)}</span>'
            f"</strong>{_static_pair_svg(frames[0], frames[-1], title='Initial and final', first_label='start', last_label='end')}"
            f'<div style="margin-top:7px;font-size:.75rem;line-height:1.5">'
            f"Final R = {float(final_line['distance_a']):.3f} Å<br>"
            f"max derived |F<sub>bias</sub>(R)| = {float(final_force):.3f} eV/Å</div></section>"
        )

    root_id = f"alchemi-bias-comparison-{uuid4().hex}"
    player = _py3dmol_bias_comparison_player(
        items,
        frame_count=frame_count,
        frame_interval_ms=frame_interval_ms,
    )
    markup = f"""
<div id="{root_id}" aria-label="Matched molecular dynamics comparison" style="max-width:none">
  {_common_css(root_id)}
  <div data-static-fallback style="padding:12px">
    {runtime_required_notice()}
    <div class="alchemi-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px">{"".join(static_cards)}</div>
    <p class="alchemi-muted" style="margin:10px 2px 0;font-size:.82rem">Static initial/final comparison. Run this cell in Jupyter to play all six matched trajectories.</p>
  </div>
  <div data-interactive>{player}</div>
</div>
<script>(function(){{const root=document.getElementById('{root_id}');if(root&&root.querySelector('iframe'))root.classList.add('is-live');}})();</script>
"""
    return _html(markup)
