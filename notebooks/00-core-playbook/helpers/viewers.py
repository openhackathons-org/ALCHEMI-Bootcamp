# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained interactive viewers for the ALCHEMI Core notebooks.

The public functions accept prepared structures and simulation callbacks. This
module owns conversion to browser data, synchronized controls, and the static
previews that remain useful when a notebook host does not run JavaScript.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from html import escape
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io import write

if TYPE_CHECKING:
    from nvalchemi.data import Batch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


def _viewer_js_helpers() -> str:
    colors = _json_for_script(_ELEMENT_COLORS)
    return f"""
const elementColors={colors};
const applyStyle=function(viewer,item){{
  const sphereOnly=Boolean(item&&item.sphere_only);const scale=sphereOnly ? .38 : .23;
  viewer.setStyle({{}},{{
    sphere:{{scale:scale,color:'{_GREEN}'}},
    ...(sphereOnly?{{}}:{{stick:{{radius:0.09,color:'{_GREEN}'}}}})
  }});
  Object.keys(elementColors).forEach(function(symbol){{
    viewer.setStyle({{elem:symbol}},{{
      sphere:{{scale:scale,color:elementColors[symbol]}},
      ...(sphereOnly?{{}}:{{stick:{{radius:0.09,color:elementColors[symbol]}}}})
    }});
  }});
}};
const addCell=function(viewer,edges){{
  (edges||[]).forEach(function(edge){{viewer.addLine({{
    start:{{x:edge.start[0],y:edge.start[1],z:edge.start[2]}},
    end:{{x:edge.end[0],y:edge.end[1],z:edge.end[2]}},color:'{_BORDER}'
  }});}});
}};
const drawStructure=function(viewer,item){{
  viewer.removeAllModels();viewer.removeAllShapes();viewer.removeAllLabels();
  viewer.addModel(item.xyz,'xyz',{{assignBonds:!item.sphere_only}});applyStyle(viewer,item);addCell(viewer,item.cell_edges);
  viewer.zoomTo();viewer.rotate(-18,'y');viewer.rotate(12,'x');viewer.render();
}};
const makeViewer=function(node){{
  if(!node)throw new Error('3Dmol viewer container is missing');
  node.style.position='relative';node.style.overflow='hidden';
  return window.$3Dmol.createViewer(node,{{backgroundColor:'{_BACKGROUND}',antialias:true}});
}};
"""


def _metadata_from_spec(key: str, spec: Any, atoms: Atoms) -> dict[str, Any]:
    return {
        "key": key,
        "display_name": str(_field(spec, "display_name", key)),
        "domain": str(_field(spec, "domain", "Other")),
        "purpose": str(_field(spec, "purpose", "Structure exploration")),
        "is_default": bool(_field(spec, "is_default", True)),
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

    return _py3dmol_card_view(item, height=height)._make_html()


def _viewer_iframe(markup: str, *, height: int, auto_height: bool = False) -> str:
    """Run viewer scripts in an isolated dark document without Output widgets."""

    body_size = (
        "min-height:100%;overflow:visible" if auto_height else "height:100%;overflow:hidden"
    )
    resize_script = (
        """
<script>(function(){
  const frame=window.frameElement;if(!frame||!window.ResizeObserver)return;
  const resize=function(){
    const height=Math.ceil(document.documentElement.scrollHeight);
    if(frame.style.height!==height+'px')frame.style.height=height+'px';
  };
  new ResizeObserver(resize).observe(document.body);
  window.addEventListener('load',resize);window.requestAnimationFrame(resize);
})();</script>
"""
        if auto_height
        else ""
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>html,body{{margin:0;width:100%;{body_size};"
        f"background:{_BACKGROUND};}}</style></head><body>"
        f"{markup}{resize_script}</body></html>"
    )
    return (
        f'<iframe title="Interactive 3D structure" srcdoc="{escape(document, quote=True)}" '
        f'style="display:block;width:100%;height:{height}px;border:0;'
        f'background:{_BACKGROUND};" loading="eager"></iframe>'
    )


def _picker_iframe(
    items: Sequence[Mapping[str, Any]],
    *,
    widget_class: str,
    selected_keys: Sequence[str],
) -> str:
    """Render all selectable py3Dmol views in one shared dark document."""

    selected = set(selected_keys)
    cards = []
    for index, item in enumerate(items):
        is_selected = item["key"] in selected
        cards.append(
            f'<article class="picker-card{" is-selected" if is_selected else ""}" '
            f'data-card="{index}">'
            f'<div class="viewer">{_py3dmol_card_markup(item, height=164)}</div>'
            '<div class="facts">'
            f'<strong>{escape(item["display_name"])}</strong>'
            f'<span>{item["atom_count"]} atoms · {escape(item["domain"])}</span>'
            f'<p>{escape(item["purpose"])}</p>'
            '</div>'
            f'<button type="button" data-select="{index}" aria-pressed="'
            f'{str(is_selected).lower()}">'
            f'{"✓ In batch" if is_selected else "+ Add to batch"}</button>'
            '</article>'
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;background:{_BACKGROUND};color:{_TEXT};
  font-family:'NVIDIA Sans',Arial,sans-serif}}
*{{box-sizing:border-box}}
.picker-shell{{padding:12px;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND}}}
.picker-head{{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:0 0 10px}}
.picker-head strong{{font-size:1rem}}
.picker-head span{{display:block;color:{_MUTED};font-size:.82rem;margin-top:3px}}
.picker-status{{color:{_MUTED};font-size:.84rem;font-weight:700;white-space:nowrap}}
.picker-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.picker-card{{min-width:0;height:292px;display:grid;grid-template-rows:164px 76px 40px;
  gap:6px;padding:6px;border:2px solid {_BORDER};border-radius:10px;
  overflow:hidden;background:{_SURFACE}}}
.picker-card.is-selected{{border-color:{_GREEN}}}
.viewer,.viewer>div{{min-width:0;width:100%!important;height:164px!important;
  overflow:hidden;background:{_BACKGROUND}!important}}
.facts{{min-width:0;padding:2px 4px;line-height:1.3;overflow:hidden}}
.facts strong{{display:block;font-size:.92rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.facts span{{display:block;color:{_MUTED};font-size:.75rem;margin-top:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.facts p{{margin:3px 0 0;font-size:.77rem;line-height:1.25}}
.picker-card button{{width:100%;height:40px;border:2px solid {_GREEN};border-radius:7px;
  background:{_SURFACE_RAISED};color:{_GREEN};font:inherit;font-size:.82rem;font-weight:800;
  cursor:pointer}}
.picker-card.is-selected button{{background:{_GREEN};color:{_BACKGROUND}}}
.picker-card button:focus-visible{{outline:3px solid {_TEXT};outline-offset:2px}}
@media(max-width:760px){{.picker-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:520px){{.picker-grid{{grid-template-columns:1fr}}}}
</style></head><body>
{_bundle_preamble()}
<main class="picker-shell">
  <header class="picker-head"><div><strong>Choose systems for the Batch</strong>
    <span>Drag to rotate · scroll to zoom · choose up to five systems</span></div>
    <div class="picker-status" role="status" aria-live="polite"></div>
  </header>
  <section class="picker-grid">{"".join(cards)}</section>
</main>
<script>$3Dmolpromise.then(function(){{
const selected=new Set({_json_for_script(list(selected_keys))});
const cards=Array.from(document.querySelectorAll('[data-card]'));
const buttons=Array.from(document.querySelectorAll('[data-select]'));
const status=document.querySelector('.picker-status');
const keys={_json_for_script([item['key'] for item in items])};
const parentRoot=window.parent.document.querySelector('.{widget_class}');
const parentButtons=parentRoot
  ? Array.from(parentRoot.querySelectorAll('.alchemi-picker-state')) : [];
const update=function(message){{
  cards.forEach(function(card,index){{
    const active=selected.has(keys[index]);const button=buttons[index];
    card.classList.toggle('is-selected',active);button.setAttribute('aria-pressed',String(active));
    button.textContent=active?'✓ In batch':'+ Add to batch';
  }});
  status.textContent=message || (selected.size
    ? selected.size+' '+(selected.size===1?'system':'systems')+' in batch'
    : 'Choose one or more systems');
}};
buttons.forEach(function(button,index){{button.addEventListener('click',function(){{
  const key=keys[index];
  if(!selected.has(key)&&selected.size>=5){{update('Choose up to five systems');return;}}
  if(selected.has(key))selected.delete(key);else selected.add(key);
  if(parentButtons[index])parentButtons[index].click();
  update();
}});}});
update();
}})();</script>
</body></html>"""
    return (
        f'<iframe title="Interactive py3Dmol structure picker" '
        f'srcdoc="{escape(document, quote=True)}" '
        f'style="display:block;width:100%;height:1000px;border:0;'
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
                    f'<strong>{escape(item["display_name"])}</strong>'
                    f'<span>{item["atom_count"]} atoms · '
                    f'{escape(item["domain"])}</span>'
                    f'<p>{escape(item["purpose"])}</p></div>'
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
                "to zoom · choose up to five systems</span></div>"
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
            raise ValueError("Select at least one molecular card before building a Batch")
        return [self._catalog[key] for key in keys]

    def selected_atoms(self) -> list[Atoms]:
        """Build fresh ASE structures for the current selection."""

        structures = []
        for spec in self.selected_specs():
            structure = spec.copy() if isinstance(spec, Atoms) else spec.build()
            structures.append(structure)
        return structures

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
        return (
            '<div role="note" style="box-sizing:border-box;width:100%;max-width:920px;'
            f'padding:16px 18px;border:1px solid {_BORDER};border-left:3px solid {_GREEN};'
            f'border-radius:8px;background:{_BACKGROUND};color:{_TEXT};'
            'font-family:NVIDIA Sans,Arial,sans-serif">'
            '<strong>Interactive molecular explorer</strong>'
            f'<p style="color:{_MUTED};font-size:.86rem;line-height:1.5;margin:6px 0 0">'
            'This output needs a running notebook. Open it in Jupyter or VS Code, '
            'connect a runtime, and run this cell to choose systems and explore them in 3D.'
            '</p></div>'
        )


def _single_structure_markup(item: Mapping[str, Any]) -> str:
    root_id = f"alchemi-preview-{uuid4().hex}"
    payload = _json_for_script(item)
    return f"""
<div id="{root_id}" tabindex="0" aria-label="3Dmol structure preview">
  {_common_css(root_id)}
  <div style="padding:10px 14px;background:{_SURFACE};display:flex;gap:12px;
              align-items:center;justify-content:space-between;flex-wrap:wrap">
    <div><strong data-name></strong><div data-area class="alchemi-muted" style="font-size:.82rem;margin-top:3px"></div></div>
    <div class="alchemi-toolbar" aria-label="Viewer controls">
      <button type="button" data-action="rotate-left" aria-label="Rotate left">↶</button>
      <button type="button" data-action="rotate-right" aria-label="Rotate right">↷</button>
      <button type="button" data-action="zoom-in" aria-label="Zoom in">＋</button>
      <button type="button" data-action="zoom-out" aria-label="Zoom out">−</button>
      <button type="button" data-action="reset">Reset</button>
    </div>
  </div>
  <div data-live-view aria-label="Interactive 3D structure"></div>
  <div style="padding:9px 14px;background:{_SURFACE};font-size:.82rem">
    <span data-facts></span><span aria-hidden="true"> · </span>
    <span data-measure role="status" aria-live="polite">Select two atoms to measure a distance.</span>
  </div>
</div>
{_bundle_preamble()}
<script>(function(){{
const root=document.getElementById('{root_id}');if(!root||!window.$3Dmol){{return;}}
const item={payload};{_viewer_js_helpers()}
const viewer=makeViewer(root.querySelector('[data-live-view]'));let selectedAtoms=[];
const setText=function(selector,text){{root.querySelector(selector).textContent=text;}};
const installMeasurement=function(){{viewer.setClickable({{}},true,function(atom){{
  const index=Number(atom.index===undefined?atom.serial:atom.index);
  if(selectedAtoms.length===2){{selectedAtoms=[];viewer.removeAllShapes();addCell(viewer,item.cell_edges);}}
  selectedAtoms.push(atom);viewer.addSphere({{center:atom,radius:.32,color:'{_GREEN}',opacity:.28}});
  if(selectedAtoms.length===1){{setText('[data-measure]','row '+index+' selected; choose one more atom');}}
  else{{const a=selectedAtoms[0],b=selectedAtoms[1];const distance=Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
    viewer.addLine({{start:a,end:b,color:'{_BLUE}',dashed:true,dashLength:.12,gapLength:.07}});
    setText('[data-measure]','distance '+distance.toFixed(3)+' Å · rows '
      +Number(a.index===undefined?a.serial:a.index)+'–'+index);}}
  viewer.render();
}});}};
const reset=function(){{selectedAtoms=[];drawStructure(viewer,item);installMeasurement();
  setText('[data-measure]','Select two atoms to measure a distance.');}};
setText('[data-name]',item.display_name);setText('[data-area]',item.domain+' · '+item.purpose);
setText('[data-facts]',item.atom_count+' atoms');reset();
root.querySelectorAll('[data-action]').forEach(function(button){{button.addEventListener('click',function(){{
  const action=button.dataset.action;if(action==='rotate-left')viewer.rotate(-12,'y');
  if(action==='rotate-right')viewer.rotate(12,'y');if(action==='zoom-in')viewer.zoom(1.15);
  if(action==='zoom-out')viewer.zoom(.87);if(action==='reset')reset();viewer.render();
}});}});
}})();</script>
"""


def show_structure_explorer(
    catalog: Mapping[str, Any],
    *,
    selected: str | Sequence[str] | None = None,
) -> StructureSelection:
    """Show a selectable molecular grid linked to later Python cells.

    ``catalog`` accepts the ordered mapping returned by ``opening_performance_catalog``.
    A value may also be an ASE ``Atoms`` object, which is useful for small custom
    extensions in a notebook.
    """

    if not catalog:
        raise ValueError("catalog must contain at least one structure")
    items = []
    for key, spec in catalog.items():
        atoms = spec.copy() if isinstance(spec, Atoms) else spec.build()
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
        references = {
            name: reference_structures[name]
            for name in names
        }
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


def _force_sparkline_svg(
    item: Mapping[str, Any], index: int, frame_count: int
) -> str:
    """Draw one compact maximum-force trace for a trajectory card."""

    width, height = 280, 68
    left, right, top, bottom = 32, 10, 22, 14
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite = [
        float(value) for value in item["forces"]
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
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle data-trace-dot="{index}" cx="{x_coord(0):.2f}" '
            f'cy="{y_coord(float(values[0])):.2f}" r="3.5" fill="{_GREEN}" '
            f'stroke="{_BACKGROUND}" stroke-width="1.5"/>'
        )

    return f"""
<svg data-force-trace="{index}" data-force-maximum="{maximum}" viewBox="0 0 {width} {height}"
  role="img" aria-label="Maximum force for {escape(str(item['name']), quote=True)}"
  style="display:block;width:100%;height:auto">
  <rect width="{width}" height="{height}" fill="{_SURFACE}"/>
  <text x="{left}" y="14" fill="{_TEXT}" font-size="9.5" font-weight="700">
    Maximum force (eV/Å)
  </text>
  <line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="{_BORDER}"/>
  <text x="{left - 5}" y="{top + 3}" text-anchor="end" fill="{_MUTED}" font-size="8">{maximum:.1f}</text>
  <text x="{left - 5}" y="{top + plot_height + 3}" text-anchor="end" fill="{_MUTED}" font-size="8">0</text>
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
) -> tuple[str, int]:
    """Build one isolated py3Dmol grid driven by a shared native frame index."""

    columns = 5
    rows = 1
    row_height = 286
    stage_height = rows * row_height + 20
    cards = "".join(
        f'<section class="trajectory-card" data-panel="{index}">'
        f'<div class="molecule-view" data-viewer="{index}" '
        f'aria-label="Trajectory for {escape(str(item["name"]), quote=True)}"></div>'
        f'<strong>{escape(str(item["name"]))}</strong>'
        '<div class="force-panel"><span data-panel-status></span>'
        f'{_force_sparkline_svg(item, index, frame_count)}</div></section>'
        for index, item in enumerate(items)
    )
    cards += '<section class="trajectory-card empty" aria-hidden="true"></section>' * (
        rows * columns - len(items)
    )
    iframe_height = 66 + stage_height

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;background:{_BACKGROUND};color:{_TEXT};font-family:'NVIDIA Sans',Arial,sans-serif}}
*{{box-sizing:border-box}}.player{{width:100%;overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_BACKGROUND}}}
.toolbar{{height:66px;padding:10px 12px;display:flex;align-items:center;gap:12px;border-bottom:1px solid {_BORDER};background:{_SURFACE}}}
button{{min-width:92px;height:40px;border:1px solid {_GREEN};border-radius:7px;background:{_SURFACE_RAISED};color:{_TEXT};font:inherit;font-weight:700;cursor:pointer}}
button:hover{{background:{_GREEN};color:{_BACKGROUND}}}button:focus-visible,input:focus-visible{{outline:3px solid {_GREEN};outline-offset:2px}}
label{{color:{_MUTED};font-size:.78rem}}.frame-control{{flex:1;min-width:180px}}
input{{accent-color:{_GREEN}}}.range-shell{{position:relative;display:block;height:18px;margin-top:5px}}
.range-shell input{{position:absolute;z-index:2;inset:0;width:100%;height:18px;margin:0;opacity:0;cursor:pointer}}
.range-track{{position:absolute;z-index:1;left:7px;right:7px;top:7px;height:4px;border-radius:4px;background:{_BORDER};pointer-events:none}}
.range-fill{{position:absolute;inset:0;border-radius:inherit;background:{_GREEN};transform:scaleX(0);transform-origin:left center;transition:transform 110ms linear}}
.range-thumb{{position:absolute;left:0;top:-4px;width:12px;height:12px;border:2px solid {_TEXT};border-radius:50%;background:{_GREEN};transform:translate3d(-6px,0,0);transition:transform 110ms linear}}
.range-shell input:focus-visible~.range-track{{outline:3px solid {_GREEN};outline-offset:3px}}
.force-toggle{{display:flex;align-items:center;gap:7px;white-space:nowrap}}.force-toggle input{{width:16px;height:16px;margin:0}}
output{{flex:0 0 118px;width:118px;text-align:right;font-size:.8rem;font-variant-numeric:tabular-nums}}
.stage-grid{{display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));grid-template-rows:repeat({rows},{row_height}px);gap:10px;width:100%;height:{stage_height}px;padding:10px;overflow:hidden;background:{_BACKGROUND}}}
.trajectory-card{{position:relative;min-width:0;overflow:hidden;border:1px solid {_BORDER};border-radius:10px;background:{_SURFACE}}}
.trajectory-card.empty{{border-color:transparent;background:transparent}}
.molecule-view{{position:absolute;inset:0 0 92px;overflow:hidden;background:{_BACKGROUND}}}
.molecule-view canvas{{position:absolute!important;inset:0;width:100%!important;height:100%!important}}
.trajectory-card strong{{position:absolute;z-index:3;top:12px;left:12px;display:block;max-width:calc(100% - 24px);width:max-content;padding:4px 7px;border-radius:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:rgba(23,29,32,.86);font-size:.82rem;pointer-events:none}}
.force-panel{{position:absolute;left:0;right:0;bottom:0;height:92px;padding:5px 8px 2px;border-top:1px solid {_BORDER};background:{_SURFACE};pointer-events:auto;cursor:crosshair}}
.force-panel span{{display:block;height:15px;padding:0 5px;color:{_MUTED};font-size:.7rem;line-height:15px;font-variant-numeric:tabular-nums}}
[data-trace-cursor]{{transition:x1 110ms linear,x2 110ms linear}}
[data-trace-dot]{{transition:cx 110ms linear,cy 110ms linear}}
@media(prefers-reduced-motion:reduce){{button,.range-fill,.range-thumb,[data-trace-cursor],[data-trace-dot]{{transition:none}}}}
</style></head><body>{_bundle_preamble()}
<main class="player" aria-label="Synchronized py3Dmol relaxation player">
  <div class="toolbar"><button type="button" data-play aria-label="Play relaxation">▶ Play</button>
    <label class="frame-control">Shared step<span class="range-shell" data-range-shell>
      <input data-frame aria-label="Shared relaxation step" type="range" min="0" max="{frame_count - 1}" value="0" step="1">
      <span class="range-track"><span class="range-fill" data-progress-fill></span><span class="range-thumb" data-progress-thumb></span></span>
    </span></label>
    <label class="force-toggle"><input data-force-color type="checkbox" checked>Color atoms by force</label>
    <output data-frame-label>Step 1 / {frame_count}</output></div>
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
  progressFill.style.transform='scaleX('+fraction+')';progressThumb.style.transform='translate3d('+(fraction*travel-6)+'px,0,0)';
}};
const update=function(index){{
  rendering=true;frameQueue=frameQueue.then(function(){{return Promise.all(viewers.map(function(viewer){{return viewer.setFrame(index);}}));}}).then(function(){{
    viewers.forEach(function(viewer,panel){{colorAtoms(viewer,items[panel],items[panel].atom_forces[index],scales[panel]);viewer.render();}});
    items.forEach(function(item,panel){{const value=item.forces[index];const first=item.forces[0];const last=item.forces[item.forces.length-1];
      document.querySelector('[data-panel="'+panel+'"] [data-panel-status]').textContent=value===null?'frame '+(index+1):'max force '+value.toFixed(3)+' eV/Å · '+first.toFixed(3)+' → '+last.toFixed(3);
      const trace=document.querySelector('[data-force-trace="'+panel+'"]');const cursor=document.querySelector('[data-trace-cursor="'+panel+'"]');const dot=document.querySelector('[data-trace-dot="'+panel+'"]');
      const x=32+238*index/Math.max({frame_count - 1},1);cursor.style.setProperty('x1',x+'px');cursor.style.setProperty('x2',x+'px');
      if(dot&&value!==null){{dot.style.setProperty('cx',x+'px');dot.style.setProperty('cy',(22+32*(1-value/Number(trace.dataset.forceMaximum)))+'px');}}
    }});
    slider.value=String(index);updateProgress(index);document.querySelector('[data-frame-label]').textContent='Step '+(index+1)+' / {frame_count}';
  }}).finally(function(){{rendering=false;}});return frameQueue;
}};
const stop=function(){{if(animationFrame!==null)window.cancelAnimationFrame(animationFrame);animationFrame=null;lastFrameTime=null;play.textContent='▶ Play';}};
const tick=function(time){{if(animationFrame===null)return;if(!rendering&&(lastFrameTime===null||time-lastFrameTime>=frameInterval)){{update((Number(slider.value)+1)%{frame_count});lastFrameTime=time;}}animationFrame=window.requestAnimationFrame(tick);}};
play.addEventListener('click',function(){{if(animationFrame!==null){{stop();return;}}play.textContent='❚❚ Pause';animationFrame=window.requestAnimationFrame(tick);}});
slider.addEventListener('input',function(){{stop();update(Number(slider.value));}});
forceToggle.addEventListener('change',function(){{update(Number(slider.value));}});
window.addEventListener('resize',function(){{updateProgress(Number(slider.value));}});
document.querySelectorAll('[data-force-trace]').forEach(function(trace){{trace.addEventListener('pointerdown',function(event){{const box=trace.getBoundingClientRect();const svgX=(event.clientX-box.left)/box.width*280;const fraction=Math.max(0,Math.min(1,(svgX-32)/238));stop();update(Math.round(fraction*{frame_count - 1}));}});}});
update(0);
}});</script></body></html>"""
    iframe = (
        f'<iframe title="Synchronized py3Dmol relaxation player" '
        f'srcdoc="{escape(document, quote=True)}" style="display:block;width:100%;'
        f'height:{iframe_height}px;border:0;background:{_BACKGROUND};" '
        'loading="eager"></iframe>'
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
    frame_interval_ms: int = 120,
    reference_structures: Mapping[str, Atoms] | Sequence[Atoms] | None = None,
) -> Any:
    """Show synchronized relaxation frames and their maximum-force traces."""

    atom_force_history = _normalise_atom_force_magnitudes(trajectories, labels)
    normalised = _normalise_trajectories(trajectories, labels)
    if len(normalised) > 5:
        raise ValueError("the Core relaxation player supports at most five systems")
    if (
        isinstance(frame_interval_ms, bool)
        or not isinstance(frame_interval_ms, int)
        or frame_interval_ms < 16
    ):
        raise ValueError("frame_interval_ms must be an integer of at least 16")
    names = list(normalised)
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
                "trajectory_xyz": "".join(
                    _atoms_to_xyz(frame) for frame in frames
                ),
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
            + f"{escape(name)} · {escape(force_text)}</div></section>"
        )
    static_cards = "".join(static_card_parts)
    player_iframe, _ = _py3dmol_relaxation_player(
        items,
        frame_count=frame_count,
        frame_interval_ms=frame_interval_ms,
    )
    markup = f"""
<div id="{root_id}" aria-label="Synchronized relaxation player" style="max-width:none">
  {_common_css(root_id)}
  <div data-static-fallback style="padding:12px">
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


def show_bond_slider(
    atoms: Atoms,
    bond: tuple[int, int],
    *,
    spring_constant: float = 10.0,
    activation_distance: float | None = None,
    distance_range: tuple[float, float] | None = None,
) -> Any:
    """Move one bond endpoint and calculate a one-sided Hookean response in-browser."""

    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")
    first, second = (int(index) for index in bond)
    if first == second or min(first, second) < 0 or max(first, second) >= len(atoms):
        raise ValueError("bond must contain two different valid atom indices")
    if not np.isfinite(spring_constant) or spring_constant <= 0:
        raise ValueError("spring_constant must be positive and finite")

    vector = np.asarray(atoms.positions[second] - atoms.positions[first], dtype=float)
    initial_distance = float(np.linalg.norm(vector))
    if initial_distance <= 1.0e-12:
        raise ValueError("the selected bond has zero length")
    threshold = (
        initial_distance if activation_distance is None else float(activation_distance)
    )
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("activation_distance must be positive and finite")
    limits = distance_range or (
        max(0.25, initial_distance * 0.70),
        initial_distance * 1.65,
    )
    low, high = (float(value) for value in limits)
    if not np.isfinite([low, high]).all() or low <= 0 or high <= low:
        raise ValueError(
            "distance_range must contain increasing positive finite values"
        )

    final_atoms = atoms.copy()
    final_atoms.positions[second] = (
        atoms.positions[first] + vector / initial_distance * high
    )
    payload = _structure_payload(atoms)
    bond_label = f"{atoms[first].symbol}–{atoms[second].symbol}"
    root_id = f"alchemi-bond-{uuid4().hex}"
    markup = f"""
<div id="{root_id}" aria-label="Interactive one-sided Hookean bond experiment">
  {_common_css(root_id)}
  <div style="padding:11px 14px;background:{_SURFACE}">
    <strong>{bond_label} bond experiment</strong>
    <span class="alchemi-muted" style="margin-left:8px;font-size:.8rem">atom rows {first}–{second}</span>
  </div>
  <div data-static-fallback style="padding:12px">
    {_static_pair_svg(atoms, final_atoms, title="Bond at the initial and maximum slider distances", first_label=f"start · {initial_distance:.2f} Å", last_label=f"end · {high:.2f} Å")}
    <p class="alchemi-muted" style="margin:10px 2px 0;font-size:.82rem">Static start/end preview. Run this cell in Jupyter to move the bond.</p>
  </div>
  <div data-interactive>
    <div data-live-view aria-label="Bond slider structure viewer"></div>
    <div style="padding:12px 14px;background:{_SURFACE}">
      <label style="display:block;font-size:.82rem;font-weight:650">{bond_label} distance
        <input data-distance type="range" min="{low:.8g}" max="{high:.8g}" value="{initial_distance:.8g}" step="{max((high - low) / 180, 0.001):.8g}" style="width:100%;margin-top:8px">
      </label>
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:10px;font-size:.8rem">
        <div><span class="alchemi-muted">Distance</span><br><strong data-value-distance></strong></div>
        <div><span class="alchemi-muted">Hookean energy</span><br><strong data-value-energy></strong></div>
        <div><span class="alchemi-muted">Restoring force</span><br><strong data-value-force></strong></div>
        <div><span class="alchemi-muted">Threshold</span><br><strong data-value-active></strong></div>
      </div>
      <p class="alchemi-muted" style="margin:10px 0 0;font-size:.78rem">For Δr above the threshold: E = ½kΔr² and |F| = kΔr.</p>
    </div>
  </div>
</div>
{_bundle_preamble()}
<script>(function(){{
const root=document.getElementById('{root_id}');if(!root||!window.$3Dmol)return;
const base={_json_for_script(payload)};const first={first},second={second},k={float(spring_constant):.12g},threshold={threshold:.12g};
{_viewer_js_helpers()}
const viewer=makeViewer(root.querySelector('[data-live-view]'));const slider=root.querySelector('[data-distance]');
drawStructure(viewer,base);
const origin=base.positions[first];const endpoint=base.positions[second];const length=Math.hypot(endpoint[0]-origin[0],endpoint[1]-origin[1],endpoint[2]-origin[2]);
const direction=[(endpoint[0]-origin[0])/length,(endpoint[1]-origin[1])/length,(endpoint[2]-origin[2])/length];
const xyz=function(symbols,positions){{return symbols.length+'\\nALCHEMI bond slider\\n'+symbols.map(function(symbol,index){{return symbol+' '+positions[index].join(' ');}}).join('\\n')+'\\n';}};
const update=function(){{
  const view=viewer.getView();const distance=Number(slider.value);const positions=base.positions.map(function(row){{return row.slice();}});
  positions[second]=[origin[0]+direction[0]*distance,origin[1]+direction[1]*distance,origin[2]+direction[2]*distance];
  const item=Object.assign({{}},base,{{positions:positions,xyz:xyz(base.symbols,positions)}});
  viewer.removeAllModels();viewer.removeAllShapes();viewer.removeAllLabels();viewer.addModel(item.xyz,'xyz');applyStyle(viewer);addCell(viewer,item.cell_edges);
  viewer.addLine({{start:{{x:origin[0],y:origin[1],z:origin[2]}},end:{{x:positions[second][0],y:positions[second][1],z:positions[second][2]}},color:'{_GREEN}',dashed:true}});
  viewer.setView(view);viewer.render();
  const stretch=Math.max(0,distance-threshold);const energy=.5*k*stretch*stretch;const force=k*stretch;
  root.querySelector('[data-value-distance]').textContent=distance.toFixed(3)+' Å';
  root.querySelector('[data-value-energy]').textContent=energy.toFixed(3)+' eV';
  root.querySelector('[data-value-force]').textContent=force.toFixed(3)+' eV/Å';
  root.querySelector('[data-value-active]').textContent=stretch>0?'active above '+threshold.toFixed(3)+' Å':'inactive below '+threshold.toFixed(3)+' Å';
}};
slider.addEventListener('input',update);update();root.classList.add('is-live');
}})();</script>
"""
    return _html(markup)


_BASELINE_BRANCH = "AIMNet2 + FIRE2"
_HOOKEAN_BRANCH = "AIMNet2 + Hookean hook + FIRE2"


def copy_with_bond_distance(
    atoms: Atoms, bond_indices: tuple[int, int], distance: float
) -> Atoms:
    """Copy ``atoms`` and move the second bond endpoint to ``distance`` in Å."""

    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")
    if len(bond_indices) != 2:
        raise ValueError("bond_indices must contain exactly two atom indices")
    first, second = (int(index) for index in bond_indices)
    if first == second or min(first, second) < 0 or max(first, second) >= len(atoms):
        raise ValueError("bond_indices must contain two different valid atom indices")
    value = float(distance)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("distance must be positive and finite")
    moved = atoms.copy()
    vector = np.asarray(atoms.positions[second] - atoms.positions[first], dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise ValueError("the selected bond has zero length")
    moved.positions[second] = atoms.positions[first] + vector / length * value
    return moved


def _normalise_hook_experiment_result(
    result: Mapping[str, Any],
) -> tuple[
    dict[str, Sequence[Any]],
    dict[str, Sequence[float]] | None,
    str,
    float | None,
    Mapping[str, str] | str | None,
]:
    if not isinstance(result, Mapping):
        raise TypeError("the run callback must return a mapping")
    trajectories = result.get("trajectories")
    if not isinstance(trajectories, Mapping):
        raise TypeError("callback data must contain a trajectories mapping")
    missing = {"baseline", "hookean"}.difference(trajectories)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"callback trajectories are missing: {names}")
    named_trajectories = {
        _BASELINE_BRANCH: trajectories["baseline"],
        _HOOKEAN_BRANCH: trajectories["hookean"],
    }

    force_history = result.get("force_history")
    named_forces = None
    if force_history is not None:
        if not isinstance(force_history, Mapping):
            raise TypeError("force_history must be a mapping when supplied")
        named_forces = {
            _BASELINE_BRANCH: force_history["baseline"],
            _HOOKEAN_BRANCH: force_history["hookean"],
        }

    convergence = result.get("convergence")
    if isinstance(convergence, Mapping):
        convergence = {
            _BASELINE_BRANCH: str(convergence.get("baseline", "not recorded")),
            _HOOKEAN_BRANCH: str(convergence.get("hookean", "not recorded")),
        }
    elapsed = result.get("elapsed_s")
    elapsed_s = None if elapsed is None else float(elapsed)
    return (
        named_trajectories,
        named_forces,
        str(result.get("device", "device not recorded")),
        elapsed_s,
        convergence,
    )


def _hook_experiment_result_html(
    result: Mapping[str, Any], bond: tuple[int, int]
) -> str:
    trajectories, forces, device, elapsed_s, convergence = (
        _normalise_hook_experiment_result(result)
    )
    return show_trajectory_comparison(
        trajectories,
        bond=bond,
        force_history=forces,
        device=device,
        elapsed_s=elapsed_s,
        convergence=convergence,
    ).data


def _curve_path(
    distances: np.ndarray, values: np.ndarray, maximum: float
) -> str:
    safe_maximum = max(float(maximum), 1.0e-12)
    x_values = 42.0 + 238.0 * (distances - distances[0]) / (
        distances[-1] - distances[0]
    )
    y_values = 118.0 - 82.0 * values / safe_maximum
    return " ".join(
        f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(zip(x_values, y_values, strict=True))
    )


def _hook_experiment_preview_markup(
    atoms: Atoms,
    bond: tuple[int, int],
    *,
    spring_constant: float,
    activation_distance: float,
    distance_range: tuple[float, float],
    commit_slider_class: str,
) -> str:
    first, second = bond
    bond_label = f"{atoms[first].symbol}–{atoms[second].symbol}"
    bond_ascii = f"{atoms[first].symbol}-{atoms[second].symbol}"
    low, high = distance_range
    initial_distance = atoms.get_distance(first, second)
    stretched = copy_with_bond_distance(atoms, bond, high)
    distances = np.linspace(low, high, 81)
    extension = np.maximum(0.0, distances - activation_distance)
    energies = 0.5 * spring_constant * extension**2
    forces = spring_constant * extension
    energy_max = max(float(energies.max()), 1.0e-12)
    force_max = max(float(forces.max()), 1.0e-12)
    initial_extension = max(0.0, initial_distance - activation_distance)
    initial_energy = 0.5 * spring_constant * initial_extension**2
    initial_force = spring_constant * initial_extension
    initial_x = 42.0 + 238.0 * (initial_distance - low) / (high - low)
    energy_y = 118.0 - 82.0 * initial_energy / energy_max
    force_y = 118.0 - 82.0 * initial_force / force_max
    payload = _structure_payload(atoms)
    root_id = f"alchemi-phenol-hook-preview-{uuid4().hex}"
    markup = f"""
<div id="{root_id}" aria-label="Phenol {bond_label} hook experiment preview">
  {_common_css(root_id)}
  <div style="padding:11px 14px;background:{_SURFACE}">
    <strong>Phenol {bond_label} hook experiment</strong>
    <span class="alchemi-muted" style="margin-left:8px;font-size:.8rem">atom rows {first}–{second}</span>
  </div>
  <div data-static-fallback style="padding:12px">
    {_static_pair_svg(atoms, stretched, title=f"Phenol {bond_label} preview", first_label=f"start · {initial_distance:.2f} Å", last_label=f"slider end · {high:.2f} Å")}
  </div>
  <div data-interactive>
    <div data-live-view aria-label="Phenol {bond_label} structure preview"></div>
  </div>
  <div style="padding:10px 14px 12px;background:{_SURFACE};border-top:1px solid {_BORDER}">
    <label style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;font-size:.82rem;font-weight:650">
      <span>{bond_label} distance</span>
      <strong data-preview-distance>{initial_distance:.3f} Å</strong>
    </label>
    <input data-distance type="range" min="{low:.6g}" max="{high:.6g}"
      step="{max((high - low) / 180, 0.001):.6g}" value="{initial_distance:.6g}"
      aria-label="{bond_label} distance in ångström"
      style="display:block;width:100%;margin:9px 0 4px;accent-color:{_GREEN}">
    <div class="alchemi-muted" style="display:flex;justify-content:space-between;font-size:.72rem">
      <span>{low:.2f} Å</span><span>Release to run both optimizations</span><span>{high:.2f} Å</span>
    </div>
  </div>
  <div class="alchemi-grid" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px;background:{_SURFACE}">
    <figure style="margin:0;border:1px solid {_BORDER};border-radius:8px;padding:8px">
      <figcaption style="font-size:.8rem;font-weight:650">Hookean potential energy, E (eV)</figcaption>
      <svg viewBox="0 0 300 145" role="img" aria-label="Quadratic Hookean potential energy against {bond_label} distance" style="display:block;width:100%;height:auto">
        <path d="M42 24V118H286" fill="none" stroke="{_BORDER}" stroke-width="1"/>
        <path d="{_curve_path(distances, energies, energy_max)}" fill="none" stroke="{_GREEN}" stroke-width="3"/>
        <circle data-energy-marker cx="{initial_x:.2f}" cy="{energy_y:.2f}" r="5" fill="{_GREEN}" stroke="{_TEXT}" stroke-width="1.5"/>
        <text x="42" y="137" fill="{_MUTED}" font-size="10">{low:.2f} Å</text><text x="248" y="137" fill="{_MUTED}" font-size="10">{high:.2f} Å</text>
      </svg>
      <div class="alchemi-muted" style="font-size:.76rem">E = ½kΔr² · quadratic</div>
    </figure>
    <figure style="margin:0;border:1px solid {_BORDER};border-radius:8px;padding:8px">
      <figcaption style="font-size:.8rem;font-weight:650">Restoring-force magnitude, |F| (eV/Å)</figcaption>
      <svg viewBox="0 0 300 145" role="img" aria-label="Linear restoring-force magnitude against {bond_label} distance" style="display:block;width:100%;height:auto">
        <path d="M42 24V118H286" fill="none" stroke="{_BORDER}" stroke-width="1"/>
        <path d="{_curve_path(distances, forces, force_max)}" fill="none" stroke="{_BLUE}" stroke-width="3"/>
        <circle data-force-marker cx="{initial_x:.2f}" cy="{force_y:.2f}" r="5" fill="{_BLUE}" stroke="{_TEXT}" stroke-width="1.5"/>
        <text x="42" y="137" fill="{_MUTED}" font-size="10">{low:.2f} Å</text><text x="248" y="137" fill="{_MUTED}" font-size="10">{high:.2f} Å</text>
      </svg>
      <div class="alchemi-muted" style="font-size:.76rem">|F| = kΔr · linear</div>
    </figure>
  </div>
  <div style="padding:0 14px 12px;background:{_SURFACE};font-size:.8rem">
    <span class="alchemi-muted" data-preview-values></span>
  </div>
</div>
{_bundle_preamble()}
<script>(function(){{
const root=document.getElementById('{root_id}');if(!root||!window.$3Dmol)return;
const base={_json_for_script(payload)};const first={first},second={second};
const k={spring_constant:.12g},threshold={activation_distance:.12g},low={low:.12g},high={high:.12g};
const energyMax={energy_max:.12g},forceMax={force_max:.12g};
{_viewer_js_helpers()}
const viewer=makeViewer(root.querySelector('[data-live-view]'));drawStructure(viewer,base);
const slider=root.querySelector('[data-distance]');
const origin=base.positions[first],endpoint=base.positions[second];
const length=Math.hypot(endpoint[0]-origin[0],endpoint[1]-origin[1],endpoint[2]-origin[2]);
const direction=[(endpoint[0]-origin[0])/length,(endpoint[1]-origin[1])/length,(endpoint[2]-origin[2])/length];
const xyz=function(symbols,positions){{return symbols.length+'\\nPhenol {bond_ascii} preview\\n'+symbols.map(function(symbol,index){{return symbol+' '+positions[index].join(' ');}}).join('\\n')+'\\n';}};
const updatePreview=function(event){{
  const distance=Number(event.target.value),view=viewer.getView();
  const positions=base.positions.map(function(row){{return row.slice();}});
  positions[second]=[origin[0]+direction[0]*distance,origin[1]+direction[1]*distance,origin[2]+direction[2]*distance];
  viewer.removeAllModels();viewer.removeAllShapes();viewer.removeAllLabels();viewer.addModel(xyz(base.symbols,positions),'xyz');applyStyle(viewer);
  viewer.addLine({{start:{{x:origin[0],y:origin[1],z:origin[2]}},end:{{x:positions[second][0],y:positions[second][1],z:positions[second][2]}},color:'{_GREEN}',dashed:true}});
  viewer.setView(view);viewer.render();
  const extension=Math.max(0,distance-threshold),energy=.5*k*extension*extension,force=k*extension;
  const x=42+238*(distance-low)/(high-low);
  root.querySelector('[data-energy-marker]').setAttribute('cx',String(x));
  root.querySelector('[data-energy-marker]').setAttribute('cy',String(118-82*energy/energyMax));
  root.querySelector('[data-force-marker]').setAttribute('cx',String(x));
  root.querySelector('[data-force-marker]').setAttribute('cy',String(118-82*force/forceMax));
  root.querySelector('[data-preview-distance]').textContent=distance.toFixed(3)+' Å';
  root.querySelector('[data-preview-values]').textContent='Hookean energy '+energy.toFixed(3)+' eV · restoring force '+force.toFixed(3)+' eV/Å';
}};
const commitPreview=function(){{
  const bridge=window.parent.document.querySelector('.{commit_slider_class} input[type="range"]');
  if(!bridge)return;
  const setValue=Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set;
  setValue.call(bridge,slider.value);
  bridge.dispatchEvent(new window.parent.Event('input',{{bubbles:true}}));
  bridge.dispatchEvent(new window.parent.Event('change',{{bubbles:true}}));
}};
slider.addEventListener('input',updatePreview);
slider.addEventListener('change',commitPreview);
updatePreview({{target:slider}});root.classList.add('is-live');
}})();</script>
"""
    return markup


class PhenolHookExperiment:
    """Presentation and event wiring for a notebook-owned two-branch run."""

    def __init__(
        self,
        phenol: Atoms,
        bond_indices: tuple[int, int],
        run_callback: Callable[[Atoms], Mapping[str, Any]],
        static_fallback_data: Mapping[str, Any],
        *,
        spring_constant: float,
        activation_distance: float,
        distance_range: tuple[float, float],
    ) -> None:
        self._phenol = phenol.copy()
        self.bond_indices = bond_indices
        self.bond_label = (
            f"{phenol[bond_indices[0]].symbol}–{phenol[bond_indices[1]].symbol}"
        )
        self._run_callback = run_callback
        self.spring_constant = spring_constant
        self.activation_distance = activation_distance
        self.distance_range = distance_range
        self.preview_distance = phenol.get_distance(*bond_indices)
        self.status = "ready"
        self.status_message = (
            f"Move {self.bond_label}, then release the slider to run both branches."
        )
        self.result_html = _hook_experiment_result_html(
            static_fallback_data, bond_indices
        )
        self._widget: Any = None
        self._status_widget: Any = None
        self._result_widget: Any = None
        self._commit_slider_class = f"alchemi-hook-commit-{uuid4().hex}"
        self._preview_html = _hook_experiment_preview_markup(
            self._phenol,
            bond_indices,
            spring_constant=spring_constant,
            activation_distance=activation_distance,
            distance_range=distance_range,
            commit_slider_class=self._commit_slider_class,
        )

    def preview(self, distance: float) -> Atoms:
        """Return preview coordinates without starting either optimization."""

        value = self._validate_distance(distance)
        self.preview_distance = value
        return copy_with_bond_distance(self._phenol, self.bond_indices, value)

    def commit(self, distance: float) -> Mapping[str, Any]:
        """Run both notebook-owned branches once for a committed bond distance."""

        edited = self.preview(distance)
        self._set_status("running", "Running both FIRE2 optimization branches…")
        try:
            result = self._run_callback(edited)
            result_html = _hook_experiment_result_html(result, self.bond_indices)
        except Exception as error:
            self._set_status("error", f"Run failed: {error}")
            raise
        self.result_html = result_html
        self._replace_result_output()
        self._set_status("success", "Both trajectories updated from returned coordinates.")
        return result

    def _validate_distance(self, distance: float) -> float:
        value = float(distance)
        low, high = self.distance_range
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"distance must be between {low:.3f} and {high:.3f} Å")
        return value

    def _set_status(self, state: str, message: str) -> None:
        self.status = state
        self.status_message = message
        if self._status_widget is not None:
            colors = {"ready": _MUTED, "running": _BLUE, "success": _GREEN, "error": "#E05252"}
            self._status_widget.value = (
                f'<div role="status" aria-live="polite" style="padding:9px 12px;'
                f'border-left:4px solid {colors[state]};background:{_SURFACE};color:{_TEXT}">'
                f"<strong>{escape(state.title())}</strong> · {escape(message)}</div>"
            )

    def _replace_result_output(self) -> None:
        if self._result_widget is None:
            return
        self._result_widget.value = _viewer_iframe(self.result_html, height=500)

    def _build_widget(self) -> Any:
        import ipywidgets as widgets

        preview = widgets.HTML(
            value=_viewer_iframe(self._preview_html, height=720, auto_height=True),
            layout=widgets.Layout(width="100%"),
        )
        commit_slider = widgets.FloatSlider(
            value=self.preview_distance,
            min=self.distance_range[0],
            max=self.distance_range[1],
            step=max((self.distance_range[1] - self.distance_range[0]) / 180, 0.001),
            description=f"{self.bond_label} distance",
            readout_format=".3f",
            continuous_update=False,
            layout=widgets.Layout(
                width="100%",
                height="1px",
                min_height="1px",
                margin="0",
                overflow="hidden",
                visibility="hidden",
            ),
        )
        commit_slider.add_class(self._commit_slider_class)
        self._status_widget = widgets.HTML()
        self._result_widget = widgets.HTML(
            value=_viewer_iframe(self.result_html, height=500),
            layout=widgets.Layout(width="100%", height="500px", overflow="hidden"),
        )
        self._result_widget.add_class("alchemi-phenol-hook-results")
        commit_slider.observe(lambda change: self.commit(change["new"]), names="value")
        self._widget = widgets.VBox(
            [
                commit_slider,
                preview,
                self._status_widget,
                self._result_widget,
            ],
            layout=widgets.Layout(width="100%", max_width=f"{_MAX_WIDTH}px"),
        )
        self._widget.add_class("alchemi-phenol-hook-experiment")
        self._set_status(self.status, self.status_message)
        return self._widget

    def _static_fallback(self) -> str:
        return (
            f'<section aria-label="Static fallback" style="max-width:{_MAX_WIDTH}px;'
            f'background:{_BACKGROUND};color:{_TEXT};border:1px solid {_BORDER};border-radius:10px;overflow:hidden">'
            f'<div style="padding:9px 12px;background:{_SURFACE};font-weight:650">Static fallback</div>'
            f"{self._preview_html}{self.result_html}</section>"
        )

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        try:
            widget = self._widget or self._build_widget()
        except ModuleNotFoundError as error:
            if error.name != "ipywidgets":
                raise
            return {"text/html": self._static_fallback()}
        bundle = widget._repr_mimebundle_(**kwargs)
        if isinstance(bundle, tuple):
            data, metadata = bundle
            data = dict(data)
            data["text/html"] = self._static_fallback()
            return data, metadata
        data = {} if bundle is None else dict(bundle)
        data["text/html"] = self._static_fallback()
        return data


def show_phenol_hook_experiment(
    phenol: Atoms,
    bond_indices: tuple[int, int],
    run_callback: Callable[[Atoms], Mapping[str, Any]],
    static_fallback_data: Mapping[str, Any],
    *,
    spring_constant: float = 10.0,
    activation_distance: float = 1.10,
    distance_range: tuple[float, float] = (0.95, 1.45),
) -> PhenolHookExperiment:
    """Build one phenol C–H or O–H experiment with a notebook-owned callback."""

    if not isinstance(phenol, Atoms):
        raise TypeError("phenol must be an ASE Atoms object")
    if not callable(run_callback):
        raise TypeError("run_callback must be callable")
    first, second = (int(index) for index in bond_indices)
    if first == second or min(first, second) < 0 or max(first, second) >= len(phenol):
        raise ValueError("bond_indices must contain two different valid atom indices")
    if phenol[first].symbol not in {"C", "O"} or phenol[second].symbol != "H":
        raise ValueError(
            "bond_indices must be ordered as carbon or oxygen, then hydrogen"
        )
    if not np.isfinite(spring_constant) or spring_constant <= 0:
        raise ValueError("spring_constant must be positive and finite")
    if not np.isfinite(activation_distance) or activation_distance <= 0:
        raise ValueError("activation_distance must be positive and finite")
    low, high = (float(value) for value in distance_range)
    if not np.isfinite([low, high]).all() or low <= 0 or high <= low:
        raise ValueError("distance_range must contain increasing positive finite values")
    initial = phenol.get_distance(first, second)
    if not low <= initial <= high:
        raise ValueError("the supplied bond distance must lie inside distance_range")

    return PhenolHookExperiment(
        phenol,
        (first, second),
        run_callback,
        static_fallback_data,
        spring_constant=float(spring_constant),
        activation_distance=float(activation_distance),
        distance_range=(low, high),
    )


def show_trajectory_comparison(
    trajectories: Mapping[str, Sequence[Any]] | Sequence[Any],
    *,
    bond: tuple[int, int],
    labels: Sequence[str] | None = None,
    force_history: Mapping[str, Sequence[float]] | None = None,
    device: str = "device not recorded",
    elapsed_s: float | None = None,
    convergence: Mapping[str, str] | str | None = None,
) -> Any:
    """Compare two synchronized trajectories with bond, force, and run status.

    A sequence of batched frames is split with ``batch_ptr``. ``bond`` is local
    to each graph, so the second graph reads positions after its graph offset.
    """

    normalised = _normalise_trajectories(trajectories, labels)
    if len(normalised) != 2:
        raise ValueError("trajectory comparison requires exactly two systems")
    first_index, second_index = (int(index) for index in bond)
    if first_index == second_index or min(first_index, second_index) < 0:
        raise ValueError("bond must contain two different non-negative local indices")
    for name, frames in normalised.items():
        if max(first_index, second_index) >= len(frames[0]):
            raise ValueError(f"bond indices are outside trajectory {name!r}")

    names = list(normalised)
    frame_count = len(next(iter(normalised.values())))
    forces = _force_values(force_history, names, frame_count)
    convergence_map = (
        {name: str(convergence.get(name, "not recorded")) for name in names}
        if isinstance(convergence, Mapping)
        else {name: str(convergence or "not recorded") for name in names}
    )
    elapsed_text = "not recorded" if elapsed_s is None else f"{float(elapsed_s):.3f} s"
    items = []
    for name, frames in normalised.items():
        bond_label = (
            f"{frames[0][first_index].symbol}–{frames[0][second_index].symbol}"
        )
        distances = [
            float(
                np.linalg.norm(
                    frame.positions[second_index] - frame.positions[first_index]
                )
            )
            for frame in frames
        ]
        items.append(
            {
                "name": name,
                "bond_label": bond_label,
                "frames": [_structure_payload(frame) for frame in frames],
                "distances": distances,
                "forces": forces[name],
            }
        )
    root_id = f"alchemi-compare-{uuid4().hex}"
    static_card_parts = []
    for name, frames in normalised.items():
        bond_label = (
            f"{frames[0][first_index].symbol}–{frames[0][second_index].symbol}"
        )
        values = forces[name]
        force_text = (
            "maximum force not recorded"
            if values[0] is None
            else f"fmax {values[0]:.3f} → {values[-1]:.3f} eV/Å"
        )
        static_card_parts.append(
            f'<section style="border:1px solid {_BORDER};border-radius:8px;overflow:hidden">'
            + _static_pair_svg(
                frames[0], frames[-1], title=f"{name}: initial and final"
            )
            + f'<div style="padding:8px 10px;background:{_SURFACE};font-size:.8rem">{escape(name)} · {escape(bond_label)} '
            + f"{np.linalg.norm(frames[0].positions[second_index] - frames[0].positions[first_index]):.3f} → "
            + f"{np.linalg.norm(frames[-1].positions[second_index] - frames[-1].positions[first_index]):.3f} Å · "
            + f"{escape(force_text)} · {escape(convergence_map[name])}</div></section>"
        )
    static_status = f"""
<section style="min-height:{_VIEWER_HEIGHT}px;padding:16px;border:1px solid {_BORDER};border-radius:8px;background:{_SURFACE}">
  <strong>Run status</strong>
  <dl style="display:grid;grid-template-columns:auto 1fr;gap:9px 14px;margin:16px 0 0;font-size:.82rem">
    <dt class="alchemi-muted">Device</dt><dd style="margin:0">{escape(str(device))}</dd>
    <dt class="alchemi-muted">Frames</dt><dd style="margin:0">{frame_count}</dd>
    <dt class="alchemi-muted">Elapsed</dt><dd style="margin:0">{escape(elapsed_text)}</dd>
  </dl>
</section>
"""
    static_cards = "".join(static_card_parts) + static_status
    panels = "".join(
        f'<section data-panel="{index}" style="min-width:0;border:1px solid {_BORDER};border-radius:8px;overflow:hidden">'
        f'<div style="padding:9px 11px;background:{_SURFACE};font-size:.84rem;font-weight:650">{escape(name)}</div>'
        f'<div data-live-view aria-label="Trajectory for {escape(name, quote=True)}"></div>'
        f'<div data-distance-output role="status" style="padding:8px 11px;background:{_SURFACE};font-size:.8rem"></div></section>'
        for index, name in enumerate(names)
    )
    markup = f"""
<div id="{root_id}" tabindex="0" aria-label="Synchronized trajectory comparison">
  {_common_css(root_id)}
  <div style="padding:10px 12px;background:{_SURFACE};display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <button type="button" data-play aria-label="Play trajectories">▶ Play</button>
    <label style="flex:1;min-width:220px;font-size:.78rem">Shared frame
      <input data-frame type="range" min="0" max="{frame_count - 1}" value="0" step="1" style="width:100%">
    </label>
    <output data-frame-label style="min-width:80px;text-align:right;font-size:.8rem">1 / {frame_count}</output>
  </div>
  <div data-static-fallback style="padding:12px">
    <div class="alchemi-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px">{static_cards}</div>
    <p class="alchemi-muted" style="margin:10px 2px 0;font-size:.82rem">Static initial/final comparison. Run this cell in Jupyter for synchronized playback.</p>
  </div>
  <div data-interactive class="alchemi-grid" style="padding:12px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px">
    {panels}
    <section style="min-height:{_VIEWER_HEIGHT}px;padding:16px;border:1px solid {_BORDER};border-radius:8px;background:{_SURFACE}">
      <strong>Run status</strong>
      <dl style="display:grid;grid-template-columns:auto 1fr;gap:9px 14px;margin:16px 0 0;font-size:.82rem">
        <dt class="alchemi-muted">Device</dt><dd style="margin:0">{escape(str(device))}</dd>
        <dt class="alchemi-muted">Frame</dt><dd data-metric-frame style="margin:0">1 / {frame_count}</dd>
        <dt class="alchemi-muted">Elapsed</dt><dd style="margin:0">{escape(elapsed_text)}</dd>
        <dt class="alchemi-muted">Convergence</dt><dd data-convergence style="margin:0"></dd>
      </dl>
    </section>
  </div>
</div>
{_bundle_preamble()}
<script>(function(){{
const root=document.getElementById('{root_id}');if(!root||!window.$3Dmol)return;
const items={_json_for_script(items)};const bond=[{first_index},{second_index}];const convergence={_json_for_script(convergence_map)};
{_viewer_js_helpers()}
const viewers=items.map(function(item,index){{return makeViewer(root.querySelector('[data-panel="'+index+'"] [data-live-view]'));}});
const slider=root.querySelector('[data-frame]');const play=root.querySelector('[data-play]');let timer=null;
const update=function(index){{
  items.forEach(function(item,panel){{
    drawStructure(viewers[panel],item.frames[index]);
    const positions=item.frames[index].positions,a=positions[bond[0]],b=positions[bond[1]];
    viewers[panel].addLine({{start:{{x:a[0],y:a[1],z:a[2]}},end:{{x:b[0],y:b[1],z:b[2]}},color:'{_GREEN}',dashed:true}});viewers[panel].render();
    const force=item.forces[index];const forceText=force===null?'maximum force not recorded':'max force '+force.toFixed(3)+' eV/Å';
    root.querySelector('[data-panel="'+panel+'"] [data-distance-output]').textContent=item.bond_label+' distance: '+item.distances[index].toFixed(3)+' Å · '+forceText+' · fixed-step frame '+(index+1);
  }});
  slider.value=String(index);root.querySelector('[data-frame-label]').textContent=(index+1)+' / {frame_count}';
  root.querySelector('[data-metric-frame]').textContent=(index+1)+' / {frame_count}';
  root.querySelector('[data-convergence]').textContent=items.map(function(item){{return item.name+': '+convergence[item.name];}}).join(' · ');
}};
const stop=function(){{if(timer)window.clearInterval(timer);timer=null;play.textContent='▶ Play';}};
play.addEventListener('click',function(){{if(timer){{stop();return;}}play.textContent='❚❚ Pause';timer=window.setInterval(function(){{let next=Number(slider.value)+1;if(next>Number(slider.max))next=0;update(next);}},500);}});
slider.addEventListener('input',function(){{stop();update(Number(slider.value));}});
root.addEventListener('keydown',function(event){{
  if(event.key===' '){{event.preventDefault();play.click();}}
  if(event.key==='ArrowLeft'||event.key==='ArrowRight'){{event.preventDefault();stop();update(Math.max(0,Math.min({frame_count - 1},Number(slider.value)+(event.key==='ArrowLeft'?-1:1))));}}
}});
update(0);root.classList.add('is-live');
}})();</script>
"""
    return _html(markup)


__all__ = [
    "copy_with_bond_distance",
    "show_bond_slider",
    "show_phenol_hook_experiment",
    "show_relaxation_player",
    "show_structure_explorer",
    "show_trajectory_comparison",
    "snapshot_batches",
]
