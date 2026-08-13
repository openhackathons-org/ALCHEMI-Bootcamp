#!/usr/bin/env python3
"""Generate editable Core diagrams and their deterministic SVG presentations."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parent
ASSET_INDEX_PATH = ASSET_DIR / "core-assets.json"
ASSET_INDEX_SCHEMA = "alchemi.core-assets.v1"
HASH_PREFIX_LENGTH = 16
PHASES = (
    ("Data preparation", ("AtomicData", "Batches", "Data loaders")),
    ("Models", ("Built-in models", "Wrapping your model")),
    ("Running simulations", ("Hooks", "Structure relaxation", "Molecular dynamics")),
    ("Learning and scale", ("Fine-tuning", "Domain decomposition")),
)
SLUGS = {
    "AtomicData": "atomicdata",
    "Batches": "batches",
    "Data loaders": "data-loaders",
    "Built-in models": "built-in-models",
    "Wrapping your model": "wrapping-your-model",
    "Hooks": "hooks",
    "Structure relaxation": "structure-relaxation",
    "Molecular dynamics": "molecular-dynamics",
    "Fine-tuning": "fine-tuning",
    "Domain decomposition": "domain-decomposition",
}

BACKGROUND = "#11161B"
SURFACE = "#20262C"
GREEN = "#76B900"
TEXT = "#F3F4F6"
MUTED = "#A8B0B8"
QUIET = "#68737E"
WARM = "#3A332C"
STEP_LINES = {
    "AtomicData": ("AtomicData",),
    "Batches": ("Batches",),
    "Data loaders": ("Data loaders",),
    "Built-in models": ("Built-in", "models"),
    "Wrapping your model": ("Wrapping your", "model"),
    "Hooks": ("Hooks",),
    "Structure relaxation": ("Structure", "relaxation"),
    "Molecular dynamics": ("Molecular", "dynamics"),
    "Fine-tuning": ("Fine-tuning",),
    "Domain decomposition": ("Domain", "decomposition"),
}
STEP_WIDTHS = {
    "AtomicData": 72,
    "Batches": 64,
    "Data loaders": 82,
    "Built-in models": 78,
    "Wrapping your model": 98,
    "Hooks": 58,
    "Structure relaxation": 88,
    "Molecular dynamics": 86,
    "Fine-tuning": 74,
    "Domain decomposition": 94,
}


def _steps() -> list[str]:
    return [step for _, phase_steps in PHASES for step in phase_steps]


def _step_layout(*, x0: float = 16, gap: float = 6) -> list[tuple[str, float, float]]:
    """Return label-aware step widths within the 880 px canvas."""

    layout = []
    x = x0
    for step in _steps():
        step_width = STEP_WIDTHS[step]
        layout.append((step, x, step_width))
        x += step_width + gap
    assert x - gap + x0 == 880
    return layout


def drawio_xml() -> str:
    """Return the editable two-level curriculum source."""

    width, x0, gap = 880, 16, 6
    layout = _step_layout(x0=x0, gap=gap)
    root = ET.Element(
        "mxGraphModel",
        {
            "adaptiveColors": "auto",
            "grid": "1",
            "gridSize": "10",
            "page": "0",
            "pageScale": "1",
            "pageWidth": str(width),
            "pageHeight": "118",
        },
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})

    phase_start = 0
    for phase_index, (phase, phase_steps) in enumerate(PHASES, start=1):
        phase_end = phase_start + len(phase_steps)
        phase_x = layout[phase_start][1]
        phase_last_x, phase_last_width = layout[phase_end - 1][1:]
        phase_width = phase_last_x + phase_last_width - phase_x
        phase_cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": f"phase-{phase_index}",
                "value": phase,
                "style": (
                    "rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
                    f"fillColor=#171C21;strokeColor=#3F4A54;fontColor={MUTED};"
                    "fontStyle=1;align=center;verticalAlign=middle;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            phase_cell,
            "mxGeometry",
            {
                "x": f"{phase_x:.1f}",
                "y": "12",
                "width": f"{phase_width:.1f}",
                "height": "28",
                "as": "geometry",
            },
        )
        phase_start = phase_end

    previous_id: str | None = None
    for index, (step, x, step_width) in enumerate(layout):
        cell_id = f"step-{index + 1}"
        step_cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": cell_id,
                "value": "<div>" + "<br/>".join(STEP_LINES[step]) + "</div>",
                "style": (
                    "rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
                    f"fillColor={SURFACE};strokeColor={QUIET};fontColor={TEXT};"
                    "fontStyle=1;fontSize=12;spacing=2;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            step_cell,
            "mxGeometry",
            {
                "x": f"{x:.1f}",
                "y": "58",
                "width": f"{step_width:.1f}",
                "height": "38",
                "as": "geometry",
            },
        )
        if previous_id is not None:
            edge = ET.SubElement(
                graph_root,
                "mxCell",
                {
                    "id": f"edge-{index}",
                    "style": (
                        "edgeStyle=none;rounded=0;html=1;"
                        f"strokeColor={QUIET};strokeWidth=1;"
                        "startArrow=none;endArrow=none;"
                    ),
                    "edge": "1",
                    "parent": "1",
                    "source": previous_id,
                    "target": cell_id,
                },
            )
            ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
        previous_id = cell_id

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def curriculum_svg(current: str | None = None, *, banner: bool = False) -> str:
    """Render the full journey or a compact current-section variant."""

    width = 880
    height = 92 if banner else 148
    x0, gap = 16, 6
    steps = _steps()
    layout = _step_layout(x0=x0, gap=gap)
    phase_y = 24 if banner else 65
    phase_box_y = 7 if banner else 47
    phase_box_height = 28
    step_y = 45 if banner else 91
    step_height = 38
    current_index = steps.index(current) if current in steps else None

    title = (
        f"ALCHEMI Core Playbook journey, current topic: {current}"
        if current is not None
        else "ALCHEMI Core Playbook journey"
    )
    description = "Four broad areas appear above ten concrete steps. " + (
        f"{current} and its parent area are highlighted in NVIDIA green."
        if current is not None
        else "The sequence runs from batching to domain decomposition."
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(description)}</desc>',
        "<defs><style>",
        ".phase{font:700 12px 'NVIDIA Sans',Arial,sans-serif;letter-spacing:.01em}",
        ".step{font:700 12px 'NVIDIA Sans',Arial,sans-serif}",
        ".heading{font:700 21px 'NVIDIA Sans',Arial,sans-serif}",
        "</style></defs>",
        f'<rect width="{width}" height="{height}" rx="8" fill="{BACKGROUND}"/>',
    ]
    if not banner:
        parts.append(
            f'<text x="{x0}" y="28" class="heading" fill="{TEXT}">'
            "From atomic data to distributed execution</text>"
        )

    phase_start = 0
    for phase, phase_steps in PHASES:
        phase_end = phase_start + len(phase_steps)
        start_x = layout[phase_start][1]
        phase_last_x, phase_last_width = layout[phase_end - 1][1:]
        phase_width = phase_last_x + phase_last_width - start_x
        active_phase = current in phase_steps
        phase_fill = "#213016" if active_phase else "#171C21"
        phase_stroke = GREEN if active_phase else "#3F4A54"
        phase_text = "#EDF7DC" if active_phase else MUTED
        phase_state = "active" if active_phase else "neutral"
        parts.extend(
            [
                f'<rect data-kind="area" data-label="{html.escape(phase)}" '
                f'data-state="{phase_state}" x="{start_x:.1f}" y="{phase_box_y}" '
                f'width="{phase_width:.1f}" height="{phase_box_height}" rx="4" '
                f'fill="{phase_fill}" stroke="{phase_stroke}" stroke-width="1"/>',
                f'<text x="{start_x + phase_width / 2:.1f}" y="{phase_y}" '
                f'class="phase" fill="{phase_text}" text-anchor="middle">'
                f"{html.escape(phase)}</text>",
            ]
        )
        phase_start = phase_end

    for index, (step, x, step_width) in enumerate(layout):
        active = index == current_index
        completed = current_index is not None and index < current_index
        fill = GREEN if active else (SURFACE if completed else "#171C21")
        stroke = GREEN if active else ("#4A555F" if completed else "#343D46")
        text_color = "#081005" if active else (TEXT if completed else "#B5BDC5")
        step_state = "active" if active else ("completed" if completed else "future")
        parts.append(
            f'<rect data-kind="step" data-label="{html.escape(step)}" '
            f'data-state="{step_state}" x="{x:.1f}" y="{step_y}" width="{step_width:.1f}" '
            f'height="{step_height}" rx="4" fill="{fill}" stroke="{stroke}"/>'
        )
        lines = STEP_LINES[step]
        line_gap = 12
        first_y = step_y + step_height / 2 + 4
        if len(lines) == 2:
            first_y -= line_gap / 2
        parts.append(
            f'<text x="{x + step_width / 2:.1f}" y="{first_y:.1f}" '
            f'class="step" fill="{text_color}" text-anchor="middle">'
        )
        for line_index, line in enumerate(lines):
            dy = "0" if line_index == 0 else str(line_gap)
            parts.append(
                f'<tspan x="{x + step_width / 2:.1f}" dy="{dy}">'
                f"{html.escape(line)}</tspan>"
            )
        parts.append("</text>")
        if index < len(steps) - 1:
            connector_start = x + step_width
            connector_end = x + step_width + gap
            parts.append(
                f'<line x1="{connector_start:.1f}" '
                f'y1="{step_y + step_height / 2:.1f}" x2="{connector_end:.1f}" '
                f'y2="{step_y + step_height / 2:.1f}" stroke="{QUIET}" '
                'stroke-width="1" stroke-linecap="round"/>'
            )
    parts.append("</svg>\n")
    return "".join(parts)


def data_relationship_svg() -> str:
    """Render the Part 01 molecule-to-batch mental model."""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="158" viewBox="0 0 920 158" role="img" aria-labelledby="title desc">
<title id="title">Molecular structure, AtomicData, and Batch</title>
<desc id="desc">A molecular structure becomes one AtomicData graph. Batch packs several graphs into shared tensors while batch index and batch pointer preserve graph boundaries.</desc>
<defs><marker id="data-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker></defs>
<rect width="920" height="158" rx="8" fill="{BACKGROUND}"/>
<g font-family="NVIDIA Sans,Arial,sans-serif" text-anchor="middle">
<rect x="28" y="38" width="238" height="82" rx="7" fill="{WARM}" stroke="{QUIET}"/>
<text x="147" y="70" fill="{TEXT}" font-size="17" font-weight="700">Molecule or structure</text>
<text x="147" y="96" fill="{MUTED}" font-size="13">ASE Atoms or pymatgen</text>
<rect x="341" y="38" width="238" height="82" rx="7" fill="{GREEN}" stroke="{GREEN}"/>
<text x="460" y="70" fill="#081005" font-size="17" font-weight="700">AtomicData</text>
<text x="460" y="96" fill="#15200D" font-size="13">one system, one graph</text>
<rect x="654" y="38" width="238" height="82" rx="7" fill="{SURFACE}" stroke="{GREEN}"/>
<text x="773" y="67" fill="{TEXT}" font-size="17" font-weight="700">Batch</text>
<text x="773" y="91" fill="{MUTED}" font-size="13">many graphs, packed tensors</text>
<text x="773" y="108" fill="{MUTED}" font-size="11">batch_idx maps rows; batch_ptr marks bounds</text>
<path d="M274 79H331" stroke="{QUIET}" stroke-width="1.5" marker-end="url(#data-arrow)"/>
<path d="M587 79H644" stroke="{QUIET}" stroke-width="1.5" marker-end="url(#data-arrow)"/>
<text x="302" y="63" fill="{MUTED}" font-size="10">from_atoms</text>
<text x="615" y="63" fill="{MUTED}" font-size="10">from_data_list</text>
</g>
</svg>
"""


def framework_drawio_xml() -> str:
    """Return editable Draw.io source for the verified binding topology."""

    root = ET.Element(
        "mxGraphModel",
        {"adaptiveColors": "auto", "grid": "1", "gridSize": "10", "page": "0"},
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})

    nodes = (
        (
            "toolkit",
            "ALCHEMI Toolkit<br><small>PyTorch-facing APIs</small>",
            20,
            62,
            GREEN,
            GREEN,
            "#081005",
        ),
        ("torch-binding", "Toolkit-Ops Torch bindings", 250, 62, SURFACE, GREEN, TEXT),
        ("operation", "Selected operation", 480, 62, SURFACE, GREEN, TEXT),
        (
            "warp",
            "Accelerated Warp kernels<br><small>CPU or CUDA</small>",
            700,
            62,
            SURFACE,
            GREEN,
            TEXT,
        ),
        (
            "jax",
            "JAX program<br><small>not used here</small>",
            20,
            152,
            WARM,
            QUIET,
            MUTED,
        ),
        ("jax-binding", "Toolkit-Ops JAX bindings", 250, 152, SURFACE, QUIET, MUTED),
        (
            "torch-native",
            "Torch-native operations<br><small>where documented</small>",
            700,
            152,
            SURFACE,
            QUIET,
            MUTED,
        ),
    )
    for node_id, value, x, y, fill, stroke, font in nodes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": value,
                "style": (
                    "rounded=1;arcSize=7;whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor={stroke};fontColor={font};"
                    "fontStyle=1;fontSize=12;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": "190",
                "height": "54",
                "as": "geometry",
            },
        )

    edges = (
        ("main-1", "toolkit", "torch-binding", GREEN, False),
        ("main-2", "torch-binding", "operation", GREEN, False),
        ("main-3", "operation", "warp", GREEN, False),
        ("jax-1", "jax", "jax-binding", QUIET, True),
        ("jax-2", "jax-binding", "warp", QUIET, True),
        ("native", "torch-binding", "torch-native", QUIET, False),
    )
    for edge_id, source, target, color, dashed in edges:
        edge = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": edge_id,
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                    f"strokeColor={color};strokeWidth=1.5;"
                    f"dashed={1 if dashed else 0};endArrow=block;endFill=1;"
                ),
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def framework_svg() -> str:
    """Render Toolkit's Torch path and Toolkit-Ops implementation branches."""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="246" viewBox="0 0 920 246" role="img" aria-labelledby="title desc">
<title id="title">ALCHEMI Toolkit, Toolkit-Ops bindings, and operation implementations</title>
<desc id="desc">The highlighted path runs from ALCHEMI Toolkit PyTorch-facing APIs through Toolkit-Ops Torch bindings and a selected operation to accelerated Warp kernels. A muted JAX program reaches supported Warp kernels through separate Toolkit-Ops JAX bindings. Some Toolkit-Ops Torch operations or utilities use a documented Torch-native implementation.</desc>
<defs>
<marker id="framework-arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{GREEN}"/></marker>
<marker id="framework-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker>
</defs>
<rect width="920" height="246" rx="8" fill="{BACKGROUND}"/>
<g font-family="NVIDIA Sans,Arial,sans-serif" text-anchor="middle">
<text x="460" y="25" fill="{TEXT}" font-size="18" font-weight="700">ALCHEMI Toolkit follows the Torch binding path</text>
<text x="113" y="48" fill="{MUTED}" font-size="10" font-weight="700">CALLER</text>
<text x="343" y="48" fill="{MUTED}" font-size="10" font-weight="700">TOOLKIT-OPS BINDING</text>
<text x="573" y="48" fill="{MUTED}" font-size="10" font-weight="700">OPERATION</text>
<text x="810" y="48" fill="{MUTED}" font-size="10" font-weight="700">IMPLEMENTATION</text>
<rect x="18" y="61" width="190" height="55" rx="6" fill="{GREEN}" stroke="{GREEN}"/>
<text x="113" y="84" fill="#081005" font-size="14" font-weight="700">ALCHEMI Toolkit</text>
<text x="113" y="103" fill="#15200D" font-size="11">PyTorch-facing APIs</text>
<rect x="248" y="61" width="190" height="55" rx="6" fill="{SURFACE}" stroke="{GREEN}"/>
<text x="343" y="93" fill="{TEXT}" font-size="13" font-weight="700">Toolkit-Ops Torch bindings</text>
<rect x="478" y="61" width="190" height="55" rx="6" fill="{SURFACE}" stroke="{GREEN}"/>
<text x="573" y="93" fill="{TEXT}" font-size="13" font-weight="700">Selected operation</text>
<rect x="713" y="61" width="190" height="55" rx="6" fill="{SURFACE}" stroke="{GREEN}"/>
<text x="808" y="84" fill="{TEXT}" font-size="13" font-weight="700">Accelerated Warp kernels</text>
<text x="808" y="103" fill="{MUTED}" font-size="11">CPU or CUDA</text>
<rect x="18" y="146" width="190" height="55" rx="6" fill="{WARM}" stroke="{QUIET}"/>
<text x="113" y="169" fill="{MUTED}" font-size="13" font-weight="700">JAX program</text>
<text x="113" y="188" fill="{QUIET}" font-size="11">not used here</text>
<rect x="248" y="146" width="190" height="55" rx="6" fill="{SURFACE}" stroke="{QUIET}"/>
<text x="343" y="178" fill="{MUTED}" font-size="13" font-weight="700">Toolkit-Ops JAX bindings</text>
<rect x="713" y="146" width="190" height="55" rx="6" fill="{SURFACE}" stroke="{QUIET}"/>
<text x="808" y="169" fill="{MUTED}" font-size="13" font-weight="700">Torch-native operations</text>
<text x="808" y="188" fill="{QUIET}" font-size="11">where documented</text>
<path d="M208 88H238" stroke="{GREEN}" stroke-width="2" marker-end="url(#framework-arrow-green)"/>
<path d="M438 88H468" stroke="{GREEN}" stroke-width="2" marker-end="url(#framework-arrow-green)"/>
<path d="M668 88H703" stroke="{GREEN}" stroke-width="2" marker-end="url(#framework-arrow-green)"/>
<path d="M208 173H238" stroke="{QUIET}" stroke-width="1.4" stroke-dasharray="4 4" marker-end="url(#framework-arrow-muted)"/>
<path d="M438 173C560 173 610 116 703 100" fill="none" stroke="{QUIET}" stroke-width="1.4" stroke-dasharray="4 4" marker-end="url(#framework-arrow-muted)"/>
<path d="M438 101C560 122 610 173 703 173" fill="none" stroke="{QUIET}" stroke-width="1.4" marker-end="url(#framework-arrow-muted)"/>
</g>
<text x="18" y="229" fill="{MUTED}" font-family="NVIDIA Sans,Arial,sans-serif" font-size="11">Arrows show API delegation toward an operation implementation. Results return through the same binding.</text>
</svg>
"""


def capability_drawio_xml() -> str:
    """Return editable Draw.io source for the Toolkit capability map."""

    root = ET.Element(
        "mxGraphModel",
        {"adaptiveColors": "auto", "grid": "1", "gridSize": "10", "page": "0"},
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})

    nodes = (
        (
            "toolkit",
            "ALCHEMI Toolkit<br><font style='font-size:10px'>reusable data, model, workflow, and training tools</font>",
            24,
            40,
            892,
            58,
            GREEN,
            GREEN,
            "#081005",
        ),
        (
            "ops",
            "Toolkit-Ops<br><font style='font-size:9px'>neighbors · reductions · interactions</font>",
            646,
            49,
            254,
            40,
            "#172119",
            "#43533B",
            TEXT,
        ),
        (
            "data",
            "Data and state<br><font style='font-size:9px'>01 AtomicData + Batch · 02 Zarr</font>",
            24,
            240,
            202,
            86,
            SURFACE,
            QUIET,
            TEXT,
        ),
        (
            "models",
            "Models and potentials<br><font style='font-size:9px'>03 Models + composition</font>",
            254,
            240,
            202,
            86,
            SURFACE,
            QUIET,
            TEXT,
        ),
        (
            "workflows",
            "Simulation workflows<br><font style='font-size:9px'>04 Hooks · 05 Dynamics · 06 GPU</font>",
            484,
            240,
            202,
            86,
            SURFACE,
            QUIET,
            TEXT,
        ),
        (
            "scale",
            "Training and scale<br><font style='font-size:9px'>07 Training · 08 Domain decomposition</font>",
            714,
            240,
            202,
            86,
            SURFACE,
            QUIET,
            TEXT,
        ),
    )
    for node_id, value, x, y, width, height, fill, stroke, font in nodes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": value,
                "style": (
                    "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor={stroke};fontColor={font};"
                    "fontStyle=1;fontSize=12;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )

    edges = tuple(
        ("toolkit", target) for target in ("data", "models", "workflows", "scale")
    )
    for index, (source, target) in enumerate(edges, start=1):
        edge = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": f"capability-edge-{index}",
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                    f"strokeColor={QUIET};strokeWidth=1.2;"
                    "startArrow=none;endArrow=block;endFill=1;"
                ),
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def capability_map_svg() -> str:
    """Render a compact interactive map from Toolkit areas to scientific work."""

    capabilities = (
        {
            "id": "data",
            "title": "Data and state",
            "summary": ("01 AtomicData + Batch", "02 Zarr data"),
            "meaning": (
                "Keep atomistic data as tensors",
                "on the model device.",
            ),
            "applications": "inputs · datasets · trajectories",
            "docs": (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/data.html",
                ),
            ),
            "deep_links": (
                (
                    "Part 01",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb",
                ),
                (
                    "Part 02",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/02-zarr-data-loading/zarr-data-loading.ipynb",
                ),
            ),
            "x": 24,
            "icon": '<circle cx="7" cy="8" r="1.4"/><circle cx="17" cy="8" r="1.4"/><circle cx="12" cy="17" r="1.4"/><path d="M8.5 8h7M7.8 9.2l3.4 6.4m5-6.4-3.4 6.4"/>',
        },
        {
            "id": "models",
            "title": "Models and potentials",
            "summary": ("03 Models + composition", ""),
            "meaning": (
                "Evaluate model outputs for each system:",
                "energy, forces, stress, and charges.",
            ),
            "applications": "MLIPs · custom models · composition",
            "docs": (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/models.html",
                ),
            ),
            "deep_links": (
                (
                    "Part 03",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb",
                ),
            ),
            "x": 254,
            "icon": '<path d="m4 8 8-4 8 4-8 4zM4 12l8 4 8-4M4 16l8 4 8-4"/>',
        },
        {
            "id": "simulation",
            "title": "Simulation workflows",
            "summary": ("04 Hooks · 05 Dynamics", "06 GPU pipelines"),
            "meaning": (
                "Simulate batches of chemical systems",
                "with optimization, MD, and screening.",
            ),
            "applications": "FIRE2 · NVE/NVT · hooks",
            "docs": (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/dynamics.html",
                ),
            ),
            "deep_links": (
                (
                    "Part 04",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/04-hooks/hooks.ipynb",
                ),
                (
                    "Part 05",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/05-base-dynamics/base-dynamics.ipynb",
                ),
                (
                    "Part 06",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/06-gpu-pipelines-profiling/gpu-pipelines-profiling.ipynb",
                ),
            ),
            "x": 484,
            "icon": '<path d="M19 8a8 8 0 0 0-13-2L4 8m0-4v4h4M5 16a8 8 0 0 0 13 2l2-2m0 4v-4h-4"/>',
        },
        {
            "id": "scale",
            "title": "Training and scale",
            "summary": ("07 Training", "08 Domain decomposition"),
            "meaning": (
                "Train and fine-tune models.",
                "Split one large system across GPUs.",
            ),
            "applications": "TrainingStrategy · DomainParallel",
            "docs": (
                (
                    "Training docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/training.html",
                ),
                (
                    "Scale docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/distributed.html",
                ),
            ),
            "deep_links": (
                (
                    "Part 07",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/07-training-finetuning/training-finetuning.ipynb",
                ),
                (
                    "Part 08",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/08-domain-decomposition/domain-decomposition.ipynb",
                ),
            ),
            "x": 714,
            "icon": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16M3 12h18"/><circle cx="7" cy="8" r="1"/><circle cx="17" cy="8" r="1"/><circle cx="7" cy="16" r="1"/><circle cx="17" cy="16" r="1"/>',
        },
    )
    css = f"""
.title{{font:700 19px 'NVIDIA Sans',Arial,sans-serif}}.root-title{{font:700 15px 'NVIDIA Sans',Arial,sans-serif}}.label{{font:700 14px 'NVIDIA Sans',Arial,sans-serif}}.sub{{font:500 11px 'NVIDIA Sans',Arial,sans-serif}}.micro{{font:600 9px 'NVIDIA Sans',Arial,sans-serif;letter-spacing:.02em}}.tooltip-title{{font:700 14px 'NVIDIA Sans',Arial,sans-serif}}.tooltip-copy{{font:500 11.5px 'NVIDIA Sans',Arial,sans-serif}}.doc-label,.deep-status{{font:700 11px 'NVIDIA Sans',Arial,sans-serif}}.doc-label{{fill:{GREEN};text-decoration:underline}}.deep-status{{fill:#929BA4}}.cap-card{{transition:fill 150ms cubic-bezier(.23,1,.32,1),stroke 150ms cubic-bezier(.23,1,.32,1),filter 150ms cubic-bezier(.23,1,.32,1)}}.capability:hover .cap-card{{fill:#26351D;stroke:{GREEN};filter:url(#soft-glow)}}.cap-tooltip{{visibility:hidden;opacity:0;pointer-events:none;transition:opacity 150ms ease}}.capability:hover .cap-tooltip{{visibility:visible;opacity:1;pointer-events:auto}}.icon{{fill:none;stroke:{GREEN};stroke-width:1.45;stroke-linecap:round;stroke-linejoin:round}}@media (prefers-reduced-motion:reduce){{.cap-card,.cap-tooltip{{transition:none}}}}
"""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 940 390" role="img" aria-labelledby="title desc" style="display:block;max-width:940px">',
        '<title id="title">Interactive ALCHEMI Toolkit capability map</title>',
        '<desc id="desc">ALCHEMI Toolkit and Toolkit-Ops connect to four areas: data and state, models and potentials, simulation workflows, and training and scale. Hover or focus a card to read its meaning, applications, documentation links, and course status.</desc>',
        '<defs><filter id="soft-glow" x="-20%" y="-30%" width="140%" height="160%"><feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#76B900" flood-opacity=".25"/></filter><marker id="cap-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#68737E"/></marker><style>',
        css,
        "</style></defs>",
        f'<rect width="940" height="390" rx="10" fill="{BACKGROUND}"/>',
        f'<text x="24" y="29" class="title" fill="{TEXT}">Toolkit capabilities and the work they enable</text>',
        f'<rect x="24" y="45" width="892" height="58" rx="8" fill="{GREEN}" stroke="{GREEN}"/>',
        '<text x="44" y="70" class="root-title" fill="#081005">ALCHEMI Toolkit</text>',
        '<text x="44" y="90" class="sub" fill="#15200D">reusable data · model · workflow · training tools</text>',
        '<rect x="646" y="54" width="254" height="40" rx="6" fill="#172119" stroke="#43533B"/>',
        f'<text x="660" y="72" class="micro" fill="{TEXT}">TOOLKIT-OPS</text>',
        f'<text x="660" y="87" class="sub" fill="{MUTED}">neighbor lists · grouped sums · interactions</text>',
        f'<path d="M470 103V200M125 200H815" fill="none" stroke="{QUIET}" stroke-width="1.25"/>',
        f'<path d="M125 200V266" fill="none" stroke="{QUIET}" stroke-width="1.25" marker-end="url(#cap-arrow)"/>',
        f'<path d="M355 200V266" fill="none" stroke="{QUIET}" stroke-width="1.25" marker-end="url(#cap-arrow)"/>',
        f'<path d="M585 200V266" fill="none" stroke="{QUIET}" stroke-width="1.25" marker-end="url(#cap-arrow)"/>',
        f'<path d="M815 200V266" fill="none" stroke="{QUIET}" stroke-width="1.25" marker-end="url(#cap-arrow)"/>',
    ]
    for item in capabilities:
        x = item["x"]
        tooltip_x = x - 8
        parts.extend(
            [
                f'<g id="{item["id"]}" class="capability" data-icon="{item["id"]}" aria-label="{html.escape(item["title"])}. {html.escape(" ".join(item["meaning"]))}">',
                '<g class="cap-tooltip">',
                f'<rect x="{tooltip_x}" y="110" width="218" height="160" rx="8" fill="#1B211D" stroke="{GREEN}"/>',
                f'<text x="{tooltip_x + 14}" y="133" class="tooltip-title" fill="{TEXT}">{html.escape(item["title"])}</text>',
                f'<text x="{tooltip_x + 14}" y="154" class="tooltip-copy" fill="#B9C0C6">{html.escape(item["meaning"][0])}</text>',
                f'<text x="{tooltip_x + 14}" y="169" class="tooltip-copy" fill="#B9C0C6">{html.escape(item["meaning"][1])}</text>',
                f'<text x="{tooltip_x + 14}" y="189" class="tooltip-copy" fill="{MUTED}">{html.escape(item["applications"])}</text>',
            ]
        )
        link_x = tooltip_x + 14
        for label, _ in item["docs"]:
            parts.append(f'<text x="{link_x}" y="213" class="doc-label">{label}</text>')
            link_x += 88
        parts.append(
            f'<text x="{tooltip_x + 14}" y="231" class="micro" fill="{QUIET}">DEEP DIVES · IN PROGRESS</text>'
        )
        link_x = tooltip_x + 14
        for label, _ in item["deep_links"]:
            parts.append(
                f'<text x="{link_x}" y="248" class="deep-status">{label}</text>'
            )
            link_x += 47
        parts.extend(
            [
                f'<rect x="{tooltip_x}" y="240" width="218" height="18" fill="transparent"/>',
                "</g>",
                f'<rect class="cap-card" x="{x}" y="276" width="202" height="84" rx="8" fill="{SURFACE}" stroke="#3A444D"/>',
                f'<g class="icon" transform="translate({x + 12} 288)">{item["icon"]}</g>',
                f'<text x="{x + 44}" y="303" class="label" fill="{TEXT}">{html.escape(item["title"])}</text>',
                f'<text x="{x + 14}" y="331" class="sub" fill="{MUTED}">{html.escape(item["summary"][0])}</text>',
                f'<text x="{x + 14}" y="347" class="sub" fill="{MUTED}">{html.escape(item["summary"][1])}</text>',
                "</g>",
            ]
        )
    parts.append(
        f'<text x="24" y="379" class="sub" fill="{MUTED}">Four areas, one shared data path. Docs are maintained; focused course notebooks are in progress.</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def zarr_data_flow_svg() -> str:
    """Render the pinned storage-to-model-device data path."""

    boxes = (
        ("Zarr store", "disk or CPU storage", 20, 176, WARM, GREEN, TEXT, MUTED),
        ("Reader", "CPU tensors", 220, 144, SURFACE, QUIET, TEXT, MUTED),
        ("Dataset", "target device", 388, 144, SURFACE, GREEN, TEXT, MUTED),
        ("DataLoader", "batch + prefetch", 556, 152, SURFACE, QUIET, TEXT, MUTED),
        ("Batch", "model device", 732, 168, GREEN, GREEN, "#081005", "#15200D"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="96" '
        'viewBox="0 0 920 96" role="img" aria-labelledby="title desc">',
        '<title id="title">Saved Zarr records to a model-device Batch</title>',
        '<desc id="desc">A Zarr store on disk or in CPU storage feeds a Reader '
        "that returns CPU tensors. Dataset chooses the target device, DataLoader "
        "batches and prefetches, and the result is a Batch on the model device.</desc>",
        '<defs><marker id="zarr-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto">'
        f'<path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker></defs>',
        f'<rect width="920" height="96" rx="8" fill="{BACKGROUND}"/>',
        '<g font-family="NVIDIA Sans,Arial,sans-serif" text-anchor="middle">',
    ]
    for label, detail, x, width, fill, stroke, text, detail_text in boxes:
        parts.extend(
            [
                f'<rect x="{x}" y="18" width="{width}" height="60" rx="7" '
                f'fill="{fill}" stroke="{stroke}"/>',
                f'<text x="{x + width / 2:g}" y="43" fill="{text}" '
                f'font-size="14" font-weight="700">{html.escape(label)}</text>',
                f'<text x="{x + width / 2:g}" y="63" fill="{detail_text}" '
                f'font-size="11">{html.escape(detail)}</text>',
            ]
        )
    for left, right in zip(boxes, boxes[1:]):
        left_edge = left[2] + left[3] + 7
        right_edge = right[2] - 7
        parts.append(
            f'<path d="M{left_edge} 48H{right_edge}" '
            f'stroke="{QUIET}" stroke-width="1.25" '
            'marker-end="url(#zarr-flow-arrow)"/>'
        )
    parts.extend(["</g>", "</svg>\n"])
    return "".join(parts)


def svg_payloads() -> dict[str, str]:
    """Return every logical SVG asset generated from this source."""

    payloads = {
        "core-journey.svg": curriculum_svg("AtomicData"),
        "framework-bindings.svg": framework_svg(),
        "molecule-atomicdata-batch.svg": data_relationship_svg(),
        "toolkit-capability-map.svg": capability_map_svg(),
        "zarr-data-flow.svg": zarr_data_flow_svg(),
    }
    payloads.update(
        {
            f"journey-banner-{slug}.svg": curriculum_svg(step, banner=True)
            for step, slug in SLUGS.items()
        }
    )
    return payloads


def write_content_addressed_svgs(payloads: dict[str, str]) -> dict[str, object]:
    """Write stable aliases, immutable hashed copies, and their asset index."""

    assets: dict[str, dict[str, str]] = {}
    for logical_name in sorted(payloads):
        payload = payloads[logical_name].encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        logical_path = Path(logical_name)
        versioned_name = (
            f"{logical_path.stem}-{digest[:HASH_PREFIX_LENGTH]}{logical_path.suffix}"
        )
        (ASSET_DIR / logical_name).write_bytes(payload)
        (ASSET_DIR / versioned_name).write_bytes(payload)
        assets[logical_name] = {
            "filename": versioned_name,
            "sha256": digest,
        }

    index: dict[str, object] = {
        "schema": ASSET_INDEX_SCHEMA,
        "assets": assets,
    }
    ASSET_INDEX_PATH.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    (ASSET_DIR / "core-curriculum.drawio").write_text(drawio_xml(), encoding="utf-8")
    (ASSET_DIR / "framework-bindings.drawio").write_text(
        framework_drawio_xml(), encoding="utf-8"
    )
    (ASSET_DIR / "toolkit-capability-map.drawio").write_text(
        capability_drawio_xml(), encoding="utf-8"
    )
    index = write_content_addressed_svgs(svg_payloads())
    print(
        "Generated curriculum map, framework diagram, "
        f"{len(SLUGS)} banners, and {len(index['assets'])} hashed SVGs"
    )


if __name__ == "__main__":
    main()
