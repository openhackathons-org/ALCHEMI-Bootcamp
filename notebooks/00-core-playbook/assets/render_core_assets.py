#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate editable Core diagrams and their deterministic SVG presentations."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parent
ASSET_INDEX_PATH = ASSET_DIR / "core-assets.json"
ASSET_INDEX_SCHEMA = "alchemi.core-assets.v1"
HASH_PREFIX_LENGTH = 16
PALETTE = json.loads(
    (ASSET_DIR.parents[2] / "shared" / "alchemi-palette.json").read_text(
        encoding="utf-8"
    )
)
BACKGROUND = PALETTE["background"]
SURFACE = PALETTE["surface"]
SURFACE_RAISED = PALETTE["surface_raised"]
BORDER = PALETTE["border"]
GRID = PALETTE["grid"]
GREEN = PALETTE["nvidia_green"]
BLUE = PALETTE["nvidia_blue"]
TEAL = PALETTE["nvidia_teal"]
ORANGE = PALETTE["nvidia_orange"]
TEXT = PALETTE["text"]
MUTED = PALETTE["muted"]
QUIET = PALETTE["quiet"]
INK = PALETTE["ink_on_accent"]

DIAGRAM_WIDTH = 920
DIAGRAM_RENDER_SCALE = 1.2
DIAGRAM_RENDER_WIDTH = round(DIAGRAM_WIDTH * DIAGRAM_RENDER_SCALE)
DIAGRAM_TITLE_SIZE = 17
DIAGRAM_SECTION_SIZE = 13
DIAGRAM_BODY_SIZE = 11
DIAGRAM_MICRO_SIZE = 10
DIAGRAM_RADIUS = 8
DIAGRAM_CARD_RADIUS = 6
DIAGRAM_CHIP_RADIUS = 5
DIAGRAM_STROKE_WIDTH = 1
DIAGRAM_CONNECTOR_WIDTH = 1.25
# The icons use xyzrender 0.2.0 with a 180 px canvas, atom scale 3.36, bond width
# 28, atom stroke width 8, gradient strength 0.65, visible hydrogens, and no fog.
MOLECULE_ICON_DIR = ASSET_DIR / "molecule-icons"


def _inline_molecule_icon(name: str, *, x: int, y: int, width: int, height: int) -> str:
    """Embed a deterministic xyzrender SVG inside a larger diagram."""

    source = (MOLECULE_ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    view_box_match = re.search(r'viewBox="([^"]+)"', source)
    if view_box_match is None:
        raise ValueError(f"Molecule icon has no viewBox: {name}")
    body = source.split(">", 1)[1].rsplit("</svg>", 1)[0]
    body = re.sub(r'id="([^"]+)"', lambda match: f'id="{name}-{match.group(1)}"', body)
    body = re.sub(
        r"url\(#([^\)]+)\)",
        lambda match: f"url(#{name}-{match.group(1)})",
        body,
    )
    body = re.sub(
        r'xlink:href="#([^"]+)"',
        lambda match: f'href="#{name}-{match.group(1)}"',
        body,
    )
    return (
        f'<svg id="molecule-{name}" x="{x}" y="{y}" width="{width}" '
        f'height="{height}" viewBox="{view_box_match.group(1)}" '
        f'preserveAspectRatio="xMidYMid meet" overflow="visible">'
        f'<g filter="url(#molecule-shadow)">{body}</g></svg>'
    )


def _drawio_molecule_style(name: str) -> str:
    """Return a self-contained SVG image style for an editable draw.io node."""

    payload = base64.b64encode((MOLECULE_ICON_DIR / f"{name}.svg").read_bytes()).decode(
        "ascii"
    )
    return (
        "shape=image;imageAspect=0;aspect=fixed;strokeColor=none;fillColor=none;"
        f"image=data:image/svg+xml;base64,{payload};"
    )


def capability_drawio_xml() -> str:
    """Return editable Draw.io source for the three-module Core map."""

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
            "ALCHEMI Toolkit<br><font style='font-size:10px'>data · models · hooks · dynamics · composition · distributed workflows</font>",
            24,
            40,
            872,
            58,
            GREEN,
            GREEN,
            INK,
        ),
        (
            "ops",
            "Toolkit-Ops<br><font style='font-size:9px'>batched operations · PyTorch · JAX</font>",
            650,
            49,
            230,
            40,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "data",
            "01 · Data and batching<br><font style='font-size:9px'>AtomicData · Batch<br>GPU batches · Zarr</font>",
            24,
            140,
            280,
            86,
            SURFACE,
            BLUE,
            TEXT,
        ),
        (
            "simulation",
            "02 · Models and simulation<br><font style='font-size:9px'>AIMNet2 · hooks<br>FIRE2 · saved state</font>",
            320,
            140,
            280,
            86,
            SURFACE,
            GREEN,
            TEXT,
        ),
        (
            "scale",
            "03 · Compose and scale<br><font style='font-size:9px'>interaction models<br>pipelines · distributed</font>",
            616,
            140,
            280,
            86,
            SURFACE,
            TEAL,
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
        ("toolkit", target) for target in ("data", "simulation", "scale")
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
                    "startArrow=none;endArrow=none;"
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
    """Render a compact static map of the three Core modules."""

    capabilities = (
        {
            "id": "data",
            "title": "01 · Data and batching",
            "summary": ("AtomicData · Batch", "GPU batches · Zarr"),
            "x": 24,
            "accent": BLUE,
            "icon": '<circle cx="7" cy="8" r="1.4"/><circle cx="17" cy="8" r="1.4"/><circle cx="12" cy="17" r="1.4"/><path d="M8.5 8h7M7.8 9.2l3.4 6.4m5-6.4-3.4 6.4"/>',
        },
        {
            "id": "simulation",
            "title": "02 · Models and simulation",
            "summary": ("AIMNet2 · hooks", "FIRE2 · saved state"),
            "x": 320,
            "accent": GREEN,
            "icon": '<path d="M19 8a8 8 0 0 0-13-2L4 8m0-4v4h4M5 16a8 8 0 0 0 13 2l2-2m0 4v-4h-4"/>',
        },
        {
            "id": "scale",
            "title": "03 · Compose and scale",
            "summary": ("interaction models", "pipelines · distributed"),
            "x": 616,
            "accent": TEAL,
            "icon": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16M3 12h18"/><circle cx="7" cy="8" r="1"/><circle cx="17" cy="8" r="1"/><circle cx="7" cy="16" r="1"/><circle cx="17" cy="16" r="1"/>',
        },
    )
    css = f"""
.title{{font:700 {DIAGRAM_TITLE_SIZE}px 'NVIDIA Sans',Arial,sans-serif}}.root-title{{font:700 {DIAGRAM_SECTION_SIZE}px 'NVIDIA Sans',Arial,sans-serif}}.label{{font:700 {DIAGRAM_SECTION_SIZE}px 'NVIDIA Sans',Arial,sans-serif}}.sub{{font:500 {DIAGRAM_BODY_SIZE}px 'NVIDIA Sans',Arial,sans-serif}}.micro{{font:600 {DIAGRAM_MICRO_SIZE}px 'NVIDIA Sans',Arial,sans-serif;letter-spacing:.02em}}.icon{{fill:none;stroke-width:1.45;stroke-linecap:round;stroke-linejoin:round}}
"""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {DIAGRAM_WIDTH} 250" role="img" aria-labelledby="title desc" style="display:block;max-width:{DIAGRAM_WIDTH}px">',
        '<title id="title">Three ALCHEMI Core modules</title>',
        '<desc id="desc">ALCHEMI Toolkit and Toolkit-Ops support three Core modules: data and batching, models and simulation, and compose and scale.</desc>',
        "<defs><style>",
        css,
        "</style></defs>",
        f'<rect width="{DIAGRAM_WIDTH}" height="250" rx="10" fill="{BACKGROUND}"/>',
        f'<text x="24" y="29" class="title" fill="{TEXT}">Three Core modules on one Toolkit foundation</text>',
        f'<rect x="24" y="45" width="872" height="58" rx="8" fill="{GREEN}" stroke="{GREEN}"/>',
        f'<text x="44" y="70" class="root-title" fill="{INK}">ALCHEMI Toolkit</text>',
        f'<text x="44" y="90" class="sub" fill="{INK}">data · models · hooks · dynamics · composition · distributed workflows</text>',
        f'<rect x="650" y="54" width="230" height="40" rx="6" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>',
        f'<text x="664" y="72" class="micro" fill="{TEXT}">TOOLKIT-OPS</text>',
        f'<text x="664" y="87" class="sub" fill="{MUTED}">batched operations · PyTorch · JAX</text>',
        f'<path d="M460 103V126M164 126H756" fill="none" stroke="{QUIET}" stroke-width="1.25"/>',
        f'<path d="M164 126V136" fill="none" stroke="{QUIET}" stroke-width="1.25"/>',
        f'<path d="M460 126V136" fill="none" stroke="{QUIET}" stroke-width="1.25"/>',
        f'<path d="M756 126V136" fill="none" stroke="{QUIET}" stroke-width="1.25"/>',
    ]
    for item in capabilities:
        x = item["x"]
        accent = item["accent"]
        parts.extend(
            [
                f'<g id="{item["id"]}" aria-label="{html.escape(item["title"])}. {html.escape(" ".join(item["summary"]))}">',
                f'<rect x="{x}" y="136" width="280" height="84" rx="8" fill="{SURFACE}" stroke="{accent}"/>',
                f'<g class="icon" transform="translate({x + 14} 148)" stroke="{accent}">{item["icon"]}</g>',
                f'<text x="{x + 46}" y="163" class="label" fill="{TEXT}">{html.escape(item["title"])}</text>',
                f'<text x="{x + 14}" y="191" class="sub" fill="{MUTED}">{html.escape(item["summary"][0])}</text>',
                f'<text x="{x + 14}" y="207" class="sub" fill="{MUTED}">{html.escape(item["summary"][1])}</text>',
                "</g>",
            ]
        )
    parts.append(
        f'<text x="24" y="239" class="sub" fill="{MUTED}">Core moves from data to simulation, then composition and scale.</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def zarr_data_flow_svg() -> str:
    """Render the pinned storage-to-selected-device data path."""

    boxes = (
        ("Zarr store", "disk or CPU storage", 24, SURFACE_RAISED, BORDER, TEXT, MUTED),
        ("Reader", "CPU tensors", 202, SURFACE_RAISED, BORDER, TEXT, MUTED),
        ("Dataset", "validate + move", 380, SURFACE_RAISED, BORDER, TEXT, MUTED),
        ("DataLoader", "batch + prefetch", 558, SURFACE_RAISED, BORDER, TEXT, MUTED),
        ("Batch", "selected device", 736, GREEN, GREEN, INK, INK),
    )
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{DIAGRAM_RENDER_WIDTH}" '
            f'height="{96 * DIAGRAM_RENDER_SCALE:g}" '
            f'viewBox="0 0 {DIAGRAM_WIDTH} 96" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">Saved Zarr records to a selected-device Batch</title>',
        (
            '<desc id="desc">A Zarr store on disk or in CPU storage feeds a Reader '
            "that returns CPU tensors. Dataset validates records and moves them to "
            "the configured device. DataLoader batches and prefetches, and the "
            "resulting Batch stays on that device.</desc>"
        ),
        (
            '<defs><marker id="zarr-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker></defs>'
        ),
        (
            f'<rect width="{DIAGRAM_WIDTH}" height="96" rx="{DIAGRAM_RADIUS}" '
            f'fill="{BACKGROUND}"/>'
        ),
        '<g font-family="NVIDIA Sans,Arial,sans-serif" text-anchor="middle">',
    ]
    for label, detail, x, fill, stroke, text, detail_text in boxes:
        parts.extend(
            [
                (
                    f'<rect x="{x}" y="18" width="160" height="60" '
                    f'rx="{DIAGRAM_CARD_RADIUS}" fill="{fill}" stroke="{stroke}" '
                    f'stroke-width="{DIAGRAM_STROKE_WIDTH}"/>'
                ),
                (
                    f'<text x="{x + 80:g}" y="43" fill="{text}" '
                    f'font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">'
                    f"{html.escape(label)}</text>"
                ),
                (
                    f'<text x="{x + 80:g}" y="63" fill="{detail_text}" '
                    f'font-size="{DIAGRAM_BODY_SIZE}">{html.escape(detail)}</text>'
                ),
            ]
        )
    for left, right in pairwise(boxes):
        left_edge = left[2] + 160
        right_edge = right[2] - 4
        parts.append(
            f'<path d="M{left_edge} 48H{right_edge}" '
            f'stroke="{QUIET}" stroke-width="{DIAGRAM_CONNECTOR_WIDTH}" '
            'marker-end="url(#zarr-flow-arrow)"/>'
        )
    parts.extend(["</g>", "</svg>\n"])
    return "".join(parts)


def zarr_drawio_xml() -> str:
    """Return an editable draw.io source for the Zarr data path."""

    root = ET.Element(
        "mxGraphModel",
        {
            "adaptiveColors": "auto",
            "page": "0",
            "pageWidth": str(DIAGRAM_WIDTH),
            "pageHeight": "96",
        },
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    nodes = (
        (
            "zarr",
            f"Zarr store<br><font color='{MUTED}'>disk or CPU storage</font>",
            24,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "reader",
            f"Reader<br><font color='{MUTED}'>CPU tensors</font>",
            202,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "dataset",
            f"Dataset<br><font color='{MUTED}'>validate + move</font>",
            380,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "loader",
            f"DataLoader<br><font color='{MUTED}'>batch + prefetch</font>",
            558,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "batch",
            f"Batch<br><font color='{INK}'>selected device</font>",
            736,
            GREEN,
            GREEN,
            INK,
        ),
    )
    for node_id, label, x, fill, stroke, font_color in nodes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": label,
                "style": (
                    f"rounded=1;arcSize={DIAGRAM_CARD_RADIUS};"
                    "whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor={stroke};"
                    f"strokeWidth={DIAGRAM_STROKE_WIDTH};fontColor={font_color};"
                    f"fontFamily=NVIDIA Sans;fontSize={DIAGRAM_SECTION_SIZE};fontStyle=1;"
                    "align=center;verticalAlign=middle;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": "18", "width": "160", "height": "60", "as": "geometry"},
        )
    for edge_index, (source, target) in enumerate(pairwise(nodes), start=1):
        edge = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": f"edge-{edge_index}",
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                    f"strokeColor={QUIET};strokeWidth={DIAGRAM_CONNECTOR_WIDTH};"
                    "endArrow=block;endFill=1;"
                ),
                "edge": "1",
                "parent": "1",
                "source": source[0],
                "target": target[0],
            },
        )
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def atomicdata_batch_fields_svg() -> str:
    """Show how several AtomicData objects become one field-aware Batch."""

    ammonia_icon = _inline_molecule_icon("ammonia", x=34, y=59, width=96, height=90)
    propyne_icon = _inline_molecule_icon("propyne", x=330, y=59, width=96, height=90)
    phenol_icon = _inline_molecule_icon("phenol", x=626, y=59, width=96, height=90)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{DIAGRAM_WIDTH}" height="440" viewBox="0 0 {DIAGRAM_WIDTH} 440" role="img" aria-labelledby="title desc">
<title id="title">A Batch packs unequal atomic systems without padding</title>
<desc id="desc">Ammonia, propyne, and phenol AtomicData objects become one Batch. Atom fields are concatenated by atom, system fields by system, and optional neighbor data is added in the format requested by a model.</desc>
<defs>
<marker id="batch-fields-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker>
<filter id="molecule-shadow" x="-25%" y="-25%" width="150%" height="160%"><feDropShadow dx="0" dy="1.4" stdDeviation="1.5" flood-color="#0B1013" flood-opacity="0.58"/></filter>
</defs>
<rect width="{DIAGRAM_WIDTH}" height="440" rx="{DIAGRAM_RADIUS}" fill="{BACKGROUND}"/>
<g font-family="NVIDIA Sans,Arial,sans-serif">
<text x="24" y="29" fill="{TEXT}" font-size="{DIAGRAM_TITLE_SIZE}" font-weight="700">A Batch packs unequal atomic systems without padding</text>
<g>
<rect x="24" y="48" width="280" height="112" rx="{DIAGRAM_RADIUS}" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
{ammonia_icon}
<text x="146" y="76" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">ATOMICDATA</text>
<text x="146" y="101" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">Ammonia</text>
<text x="146" y="124" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">4 atoms</text>
<rect x="320" y="48" width="280" height="112" rx="{DIAGRAM_RADIUS}" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
{propyne_icon}
<text x="442" y="76" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">ATOMICDATA</text>
<text x="442" y="101" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">Propyne</text>
<text x="442" y="124" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">7 atoms</text>
<rect x="616" y="48" width="280" height="112" rx="{DIAGRAM_RADIUS}" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
{phenol_icon}
<text x="738" y="76" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">ATOMICDATA</text>
<text x="738" y="101" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">Phenol</text>
<text x="738" y="124" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">13 atoms</text>
</g>
<path d="M460 163V193" stroke="{QUIET}" stroke-width="1.25" marker-end="url(#batch-fields-arrow)"/>
<rect x="475" y="168" width="154" height="20" rx="5" fill="{BACKGROUND}"/>
<text x="482" y="182" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}">Batch.from_data_list(...)</text>
<rect x="24" y="202" width="872" height="222" rx="{DIAGRAM_RADIUS}" fill="{SURFACE}" stroke="{GREEN}" stroke-width="1.5"/>
<text x="42" y="229" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">Batch</text>
<text x="102" y="229" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">3 systems · 24 atoms · no padding</text>
<text x="42" y="249" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">Field levels keep every tensor aligned when the systems are packed, moved, or sliced.</text>
<rect x="42" y="263" width="836" height="38" rx="6" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
<text x="58" y="287" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">Node fields</text>
<text x="190" y="287" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}">atomic_numbers [24] · positions [24, 3]</text>
<text x="862" y="287" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}" text-anchor="end">one row per atom</text>
<rect x="42" y="309" width="836" height="38" rx="6" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
<text x="58" y="333" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">System fields</text>
<text x="190" y="333" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}">charge [3, 1]</text>
<text x="862" y="333" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}" text-anchor="end">one row per system</text>
<rect x="42" y="355" width="836" height="42" rx="6" fill="{SURFACE_RAISED}" stroke="{BORDER}"/>
<text x="58" y="380" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700">Neighbor data</text>
<text x="190" y="373" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}">created by compute_neighbors(...)</text>
<text x="190" y="389" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}">default MATRIX per atom · optional COO edge list [E, 2]</text>
<text x="862" y="380" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}" text-anchor="end">absent in this fresh Batch</text>
<text x="42" y="413" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}">batch_ptr [0, 4, 11, 24] preserves the three system boundaries.</text>
</g>
</svg>
'''


def atomicdata_batch_fields_drawio_xml() -> str:
    """Return editable draw.io source for the AtomicData-to-Batch diagram."""

    root = ET.Element(
        "mxGraphModel",
        {"adaptiveColors": "auto", "page": "0", "pageWidth": str(DIAGRAM_WIDTH)},
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    nodes = (
        (
            "systems",
            "AtomicData systems",
            24,
            48,
            872,
            112,
            BACKGROUND,
            BACKGROUND,
            TEXT,
        ),
        (
            "ammonia",
            "Ammonia<br><font style='font-size:11px'>AtomicData · 4 atoms</font>",
            24,
            48,
            280,
            112,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "propyne",
            "Propyne<br><font style='font-size:11px'>AtomicData · 7 atoms</font>",
            320,
            48,
            280,
            112,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "phenol",
            "Phenol<br><font style='font-size:11px'>AtomicData · 13 atoms</font>",
            616,
            48,
            280,
            112,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "batch",
            "Batch<br><font style='font-size:11px'>3 systems · 24 atoms · no padding<br>batch_ptr [0, 4, 11, 24] preserves system boundaries</font>",
            24,
            202,
            872,
            222,
            SURFACE,
            GREEN,
            TEXT,
        ),
        (
            "node",
            "Node fields<br><font style='font-size:10px'>atomic_numbers [24] · positions [24, 3] · one row per atom</font>",
            42,
            263,
            836,
            38,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "system",
            "System fields<br><font style='font-size:10px'>charge [3, 1] · one row per system</font>",
            42,
            309,
            836,
            38,
            SURFACE_RAISED,
            BORDER,
            TEXT,
        ),
        (
            "neighbor",
            "Neighbor data<br><font style='font-size:10px'>created by compute_neighbors(...) · default MATRIX per atom · optional COO edge list [E, 2] · absent in this fresh Batch</font>",
            42,
            355,
            836,
            42,
            SURFACE_RAISED,
            BORDER,
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
                    f"fontFamily=NVIDIA Sans;fontSize={DIAGRAM_SECTION_SIZE};fontStyle=1;"
                    + (
                        "align=left;spacingLeft=120;verticalAlign=middle;"
                        if node_id in {"ammonia", "propyne", "phenol"}
                        else "align=center;verticalAlign=middle;"
                    )
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
    for name, x in (("ammonia", 34), ("propyne", 330), ("phenol", 626)):
        icon = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": f"icon-{name}",
                "value": "",
                "style": _drawio_molecule_style(name),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            icon,
            "mxGeometry",
            {"x": str(x), "y": "59", "width": "96", "height": "90", "as": "geometry"},
        )
    for edge_id, source, target in (("pack", "systems", "batch"),):
        edge = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": edge_id,
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                    f"strokeColor={QUIET};strokeWidth=1.25;endArrow=block;endFill=1;"
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


def batch_row_ownership_svg() -> str:
    """Show the atom rows, graph owners, and boundaries of the example Batch."""

    systems = (
        ("0 · Ammonia", 0, 4, GREEN),
        ("1 · Propyne", 4, 11, BLUE),
        ("2 · Phenol", 11, 24, TEAL),
    )
    symbols = (
        "N",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "O",
        "H",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
    )
    owners = (0,) * 4 + (1,) * 7 + (2,) * 13
    owner_colors = (GREEN, BLUE, TEAL)
    start_x = 150
    pitch = 29
    chip_width = 24
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{DIAGRAM_RENDER_WIDTH}" '
            f'height="{260 * DIAGRAM_RENDER_SCALE:g}" '
            f'viewBox="0 0 {DIAGRAM_WIDTH} 260" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">Atom-row ownership inside the example Batch</title>',
        (
            '<desc id="desc">The packed atomic number rows for ammonia, propyne, '
            'and phenol align with repeated batch index values. Batch pointer values '
            'zero, four, eleven, and twenty-four mark the system boundaries.</desc>'
        ),
        (
            f'<rect width="{DIAGRAM_WIDTH}" height="260" '
            f'rx="{DIAGRAM_RADIUS}" fill="{BACKGROUND}"/>'
        ),
        '<g font-family="NVIDIA Sans,Arial,sans-serif">',
        (
            f'<text x="24" y="29" fill="{TEXT}" '
            f'font-size="{DIAGRAM_TITLE_SIZE}" font-weight="700">'
            "One column per atom row</text>"
        ),
        (
            f'<text x="24" y="48" fill="{MUTED}" '
            f'font-size="{DIAGRAM_BODY_SIZE}">batch_idx assigns each row to a '
            "system; batch_ptr marks where each system starts and ends.</text>"
        ),
        (
            f'<rect x="24" y="56" width="872" height="166" '
            f'rx="{DIAGRAM_RADIUS}" fill="{SURFACE}" stroke="{BORDER}" '
            f'stroke-width="{DIAGRAM_STROKE_WIDTH}"/>'
        ),
    ]
    for label, start, stop, color in systems:
        x = start_x - 4 + start * pitch
        width = (stop - start) * pitch
        center = x + width / 2
        parts.extend(
            [
                (
                    f'<rect x="{x}" y="58" width="{width}" height="132" '
                    f'rx="{DIAGRAM_CARD_RADIUS}" fill="{SURFACE_RAISED}" '
                    f'stroke="{color}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>'
                ),
                (
                    f'<rect x="{x + 2}" y="60" width="{width - 4}" height="40" '
                    f'rx="{DIAGRAM_CHIP_RADIUS}" fill="{color}" fill-opacity="0.18"/>'
                ),
                (
                    f'<text x="{center:g}" y="77" fill="{TEXT}" '
                    f'font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700" '
                    f'text-anchor="middle">{html.escape(label)}</text>'
                ),
                (
                    f'<text x="{center:g}" y="93" fill="{MUTED}" '
                    f'font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">'
                    f"rows {start}:{stop}</text>"
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<text x="132" y="132" fill="{TEXT}" '
                f'font-size="{DIAGRAM_BODY_SIZE}" font-weight="700" '
                'text-anchor="end">atomic_numbers</text>'
            ),
            (
                f'<text x="132" y="174" fill="{TEXT}" '
                f'font-size="{DIAGRAM_BODY_SIZE}" font-weight="700" '
                'text-anchor="end">batch_idx</text>'
            ),
        ]
    )
    for row, (symbol, owner) in enumerate(zip(symbols, owners, strict=True)):
        x = start_x + row * pitch
        color = owner_colors[owner]
        parts.extend(
            [
                (
                    f'<rect x="{x}" y="111" width="{chip_width}" height="30" '
                    f'rx="{DIAGRAM_CHIP_RADIUS}" fill="{color}" fill-opacity="0.12" '
                    f'stroke="{color}" stroke-opacity="0.65" '
                    f'stroke-width="{DIAGRAM_STROKE_WIDTH}"/>'
                ),
                (
                    f'<text x="{x + chip_width / 2:g}" y="131" fill="{TEXT}" '
                    f'font-size="{DIAGRAM_BODY_SIZE}" font-weight="700" '
                    f'text-anchor="middle">{symbol}</text>'
                ),
                (
                    f'<rect x="{x}" y="153" width="{chip_width}" height="28" '
                    f'rx="{DIAGRAM_CHIP_RADIUS}" fill="{color}" fill-opacity="0.28" '
                    f'stroke="{color}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>'
                ),
                (
                    f'<text x="{x + chip_width / 2:g}" y="172" fill="{TEXT}" '
                    f'font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" '
                    f'text-anchor="middle">{owner}</text>'
                ),
            ]
        )
    for pointer in (0, 4, 11, 24):
        x = start_x - 4 + pointer * pitch
        parts.extend(
            [
                (
                    f'<path d="M{x} 104V194" stroke="{QUIET}" '
                    f'stroke-width="{DIAGRAM_STROKE_WIDTH}" stroke-dasharray="3 3"/>'
                ),
                (
                    f'<text x="{x}" y="211" fill="{MUTED}" '
                    f'font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">'
                    f"{pointer}</text>"
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<text x="24" y="241" fill="{TEXT}" '
                f'font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">'
                "batch_ptr</text>"
            ),
            (
                f'<text x="104" y="241" fill="{MUTED}" '
                f'font-size="{DIAGRAM_BODY_SIZE}">[0, 4, 11, 24]</text>'
            ),
            "</g></svg>\n",
        ]
    )
    return "".join(parts)


def batch_row_ownership_drawio_xml() -> str:
    """Return editable draw.io source for the Batch row-ownership diagram."""

    root = ET.Element(
        "mxGraphModel",
        {
            "adaptiveColors": "auto",
            "page": "0",
            "pageWidth": str(DIAGRAM_WIDTH),
            "pageHeight": "260",
        },
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    nodes = (
        (
            "title",
            "One column per atom row",
            24,
            8,
            872,
            28,
            "none",
            "none",
            TEXT,
            DIAGRAM_TITLE_SIZE,
            DIAGRAM_RADIUS,
        ),
        (
            "table-panel",
            "",
            24,
            56,
            872,
            166,
            SURFACE,
            BORDER,
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_RADIUS,
        ),
        (
            "ammonia",
            "0 · Ammonia<br><font style='font-size:10px'>rows 0:4</font>",
            146,
            60,
            116,
            40,
            SURFACE_RAISED,
            GREEN,
            TEXT,
            DIAGRAM_SECTION_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
        (
            "propyne",
            "1 · Propyne<br><font style='font-size:10px'>rows 4:11</font>",
            262,
            60,
            203,
            40,
            SURFACE_RAISED,
            BLUE,
            TEXT,
            DIAGRAM_SECTION_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
        (
            "phenol",
            "2 · Phenol<br><font style='font-size:10px'>rows 11:24</font>",
            465,
            60,
            377,
            40,
            SURFACE_RAISED,
            TEAL,
            TEXT,
            DIAGRAM_SECTION_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
        (
            "atomic-label",
            "atomic_numbers",
            24,
            116,
            108,
            30,
            "none",
            "none",
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
        (
            "atomic-ammonia",
            "N H H H",
            146,
            111,
            111,
            30,
            SURFACE_RAISED,
            GREEN,
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "atomic-propyne",
            "C C C H H H H",
            262,
            111,
            203,
            30,
            SURFACE_RAISED,
            BLUE,
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "atomic-phenol",
            "C O H C C C C C H H H H H",
            465,
            111,
            377,
            30,
            SURFACE_RAISED,
            TEAL,
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "owner-label",
            "batch_idx",
            24,
            153,
            108,
            28,
            "none",
            "none",
            TEXT,
            DIAGRAM_BODY_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
        (
            "owner-ammonia",
            "0  0  0  0",
            146,
            153,
            116,
            28,
            SURFACE_RAISED,
            GREEN,
            TEXT,
            DIAGRAM_MICRO_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "owner-propyne",
            "1  1  1  1  1  1  1",
            262,
            153,
            203,
            28,
            SURFACE_RAISED,
            BLUE,
            TEXT,
            DIAGRAM_MICRO_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "owner-phenol",
            "2  2  2  2  2  2  2  2  2  2  2  2  2",
            465,
            153,
            377,
            28,
            SURFACE_RAISED,
            TEAL,
            TEXT,
            DIAGRAM_MICRO_SIZE,
            DIAGRAM_CHIP_RADIUS,
        ),
        (
            "pointer",
            "batch_ptr  [0, 4, 11, 24]",
            24,
            224,
            818,
            26,
            "none",
            "none",
            TEXT,
            DIAGRAM_SECTION_SIZE,
            DIAGRAM_CARD_RADIUS,
        ),
    )
    for (
        node_id,
        value,
        x,
        y,
        width,
        height,
        fill,
        stroke,
        font,
        font_size,
        radius,
    ) in nodes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": value,
                "style": (
                    f"rounded=1;arcSize={radius};whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor={stroke};"
                    f"strokeWidth={DIAGRAM_STROKE_WIDTH};fontColor={font};"
                    f"fontFamily=NVIDIA Sans;fontSize={font_size};fontStyle=1;"
                    "align=center;verticalAlign=middle;"
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
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def prefetch_window_svg() -> str:
    """Compare synchronous loading with target-device prefetching."""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{DIAGRAM_RENDER_WIDTH}" height="{520 * DIAGRAM_RENDER_SCALE:g}" viewBox="0 0 {DIAGRAM_WIDTH} 520" role="img" aria-labelledby="title desc">
<title id="title">Prefetch trades device memory for less waiting</title>
<desc id="desc">Without prefetch, loading and model work alternate and one Batch is on the target device. With prefetch factor two, Batches 2 and 3 transfer while Batch 1 runs, so the current Batch and two prefetched Batches can occupy device memory.</desc>
<defs>
<marker id="prefetch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{QUIET}"/></marker>
</defs>
<rect width="{DIAGRAM_WIDTH}" height="520" rx="{DIAGRAM_RADIUS}" fill="{BACKGROUND}"/>
<g font-family="NVIDIA Sans,Arial,sans-serif">
<text x="24" y="29" fill="{TEXT}" font-size="{DIAGRAM_TITLE_SIZE}" font-weight="700">Prefetch trades device memory for less waiting</text>
<g font-size="{DIAGRAM_BODY_SIZE}" fill="{MUTED}">
<rect x="24" y="43" width="12" height="12" rx="2" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}"/><text x="42" y="54">load or transfer</text>
<rect x="168" y="43" width="12" height="12" rx="2" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}"/><text x="186" y="54">model work</text>
</g>

<rect x="24" y="72" width="872" height="176" rx="{DIAGRAM_RADIUS}" fill="{SURFACE}" stroke="{BORDER}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="42" y="99" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">prefetch_factor = 0</text>
<text x="42" y="120" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">load and model work alternate</text>
<text x="42" y="174" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}" font-weight="700">Lower device memory</text>
<text x="42" y="195" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">one Batch on device</text>
<g text-anchor="middle" font-size="{DIAGRAM_BODY_SIZE}" font-weight="700">
<rect x="220" y="118" width="66" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="253" y="144" fill="{TEXT}">load B1</text>
<rect x="290" y="118" width="86" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="333" y="144" fill="{TEXT}">run B1</text>
<rect x="380" y="118" width="66" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="413" y="144" fill="{TEXT}">load B2</text>
<rect x="450" y="118" width="86" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="493" y="144" fill="{TEXT}">run B2</text>
<rect x="540" y="118" width="66" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="573" y="144" fill="{TEXT}">load B3</text>
<rect x="610" y="118" width="66" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="643" y="144" fill="{TEXT}">run B3</text>
</g>
<path d="M220 196H676" stroke="{QUIET}" stroke-width="{DIAGRAM_CONNECTOR_WIDTH}" marker-end="url(#prefetch-arrow)"/>
<text x="448" y="217" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">elapsed time</text>
<rect x="700" y="96" width="178" height="128" rx="{DIAGRAM_RADIUS}" fill="{SURFACE_RAISED}" stroke="{BORDER}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="119" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="middle">TARGET DEVICE MEMORY</text>
<rect x="719" y="138" width="140" height="38" rx="{DIAGRAM_CHIP_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="162" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}" font-weight="700" text-anchor="middle">B1 · current</text>
<text x="789" y="202" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">1 Batch on device</text>

<rect x="24" y="264" width="872" height="232" rx="{DIAGRAM_RADIUS}" fill="{SURFACE}" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="42" y="291" fill="{TEXT}" font-size="{DIAGRAM_SECTION_SIZE}" font-weight="700">prefetch_factor = 2 · CUDA streams</text>
<text x="42" y="312" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">transfers overlap model work</text>
<text x="42" y="386" fill="{TEXT}" font-size="{DIAGRAM_BODY_SIZE}" font-weight="700">Higher device memory</text>
<text x="42" y="407" fill="{MUTED}" font-size="{DIAGRAM_BODY_SIZE}">current + two prefetched Batches</text>
<text x="210" y="353" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="end">MODEL</text>
<text x="210" y="404" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="end">PREFETCH</text>
<g text-anchor="middle" font-size="{DIAGRAM_BODY_SIZE}" font-weight="700">
<rect x="220" y="326" width="68" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="254" y="352" fill="{TEXT}">load B1</text>
<rect x="296" y="326" width="112" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="352" y="352" fill="{TEXT}">run B1</text>
<rect x="416" y="326" width="112" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="472" y="352" fill="{TEXT}">run B2</text>
<rect x="536" y="326" width="112" height="42" rx="{DIAGRAM_CARD_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="592" y="352" fill="{TEXT}">run B3</text>
<rect x="296" y="378" width="112" height="38" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="352" y="402" fill="{TEXT}">prefetch B2</text>
<rect x="416" y="378" width="112" height="38" rx="{DIAGRAM_CARD_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/><text x="472" y="402" fill="{TEXT}">prefetch B3</text>
</g>
<path d="M220 444H648" stroke="{QUIET}" stroke-width="{DIAGRAM_CONNECTOR_WIDTH}" marker-end="url(#prefetch-arrow)"/>
<text x="434" y="465" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">loading can finish while the model is busy</text>
<rect x="700" y="286" width="178" height="182" rx="{DIAGRAM_RADIUS}" fill="{SURFACE_RAISED}" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="309" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="middle">TARGET DEVICE MEMORY</text>
<rect x="719" y="324" width="140" height="30" rx="{DIAGRAM_CHIP_RADIUS}" fill="{GREEN}" fill-opacity="0.22" stroke="{GREEN}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="344" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="middle">B1 · current</text>
<rect x="719" y="362" width="140" height="30" rx="{DIAGRAM_CHIP_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="382" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="middle">B2 · prefetched</text>
<rect x="719" y="400" width="140" height="30" rx="{DIAGRAM_CHIP_RADIUS}" fill="{BLUE}" fill-opacity="0.22" stroke="{BLUE}" stroke-width="{DIAGRAM_STROKE_WIDTH}"/>
<text x="789" y="420" fill="{TEXT}" font-size="{DIAGRAM_MICRO_SIZE}" font-weight="700" text-anchor="middle">B3 · prefetched</text>
<text x="789" y="451" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}" text-anchor="middle">3 Batches on device</text>
<text x="24" y="512" fill="{MUTED}" font-size="{DIAGRAM_MICRO_SIZE}">Illustrative timing. Batch size stays fixed; prefetch changes how much data is prepared ahead.</text>
</g>
</svg>
'''


def prefetch_window_drawio_xml() -> str:
    """Return editable draw.io source for the prefetch comparison."""

    root = ET.Element(
        "mxGraphModel",
        {
            "adaptiveColors": "auto",
            "page": "0",
            "pageWidth": str(DIAGRAM_WIDTH),
            "pageHeight": "520",
        },
    )
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    nodes = (
        (
            "title",
            "Prefetch trades device memory for less waiting",
            24,
            8,
            872,
            28,
            BACKGROUND,
            BACKGROUND,
            TEXT,
            DIAGRAM_TITLE_SIZE,
        ),
        (
            "sync",
            "prefetch_factor = 0<br><font style='font-size:11px'>lower device memory · load and model work alternate</font>",
            24,
            72,
            182,
            152,
            SURFACE,
            BORDER,
            TEXT,
            DIAGRAM_SECTION_SIZE,
        ),
        ("sync-load1", "load B1", 220, 118, 66, 42, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        ("sync-compute1", "run B1", 290, 118, 86, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        ("sync-load2", "load B2", 380, 118, 66, 42, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        ("sync-compute2", "run B2", 450, 118, 86, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        ("sync-load3", "load B3", 540, 118, 66, 42, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        ("sync-compute3", "run B3", 610, 118, 66, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        (
            "sync-memory",
            "TARGET DEVICE MEMORY<br><font style='font-size:10px'>B1 · current<br>1 Batch on device</font>",
            700,
            96,
            178,
            128,
            SURFACE_RAISED,
            BORDER,
            TEXT,
            DIAGRAM_MICRO_SIZE,
        ),
        (
            "prefetch",
            "prefetch_factor = 2 · CUDA streams<br><font style='font-size:11px'>higher device memory · transfers overlap model work</font>",
            24,
            264,
            182,
            204,
            SURFACE,
            BLUE,
            TEXT,
            DIAGRAM_SECTION_SIZE,
        ),
        ("prefetch-load1", "load B1", 220, 326, 68, 42, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        ("prefetch-compute1", "run B1", 296, 326, 112, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        ("prefetch-compute2", "run B2", 416, 326, 112, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        ("prefetch-compute3", "run B3", 536, 326, 112, 42, GREEN, GREEN, TEXT, DIAGRAM_BODY_SIZE),
        ("prefetch-load2", "prefetch B2", 296, 378, 112, 38, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        ("prefetch-load3", "prefetch B3", 416, 378, 112, 38, BLUE, BLUE, TEXT, DIAGRAM_BODY_SIZE),
        (
            "prefetch-memory",
            "TARGET DEVICE MEMORY<br><font style='font-size:10px'>B1 · current<br>B2 · prefetched<br>B3 · prefetched<br>3 Batches on device</font>",
            700,
            286,
            178,
            182,
            SURFACE_RAISED,
            BLUE,
            TEXT,
            DIAGRAM_MICRO_SIZE,
        ),
    )
    for node_id, value, x, y, width, height, fill, stroke, font, font_size in nodes:
        fill_opacity = (
            22
            if node_id.startswith(("sync-load", "sync-compute", "prefetch-load", "prefetch-compute"))
            else 100
        )
        radius = (
            DIAGRAM_RADIUS
            if node_id in {"sync", "sync-memory", "prefetch", "prefetch-memory"}
            else DIAGRAM_CARD_RADIUS
        )
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": value,
                "style": (
                    f"rounded=1;arcSize={radius};whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor={stroke};fontColor={font};"
                    f"fillOpacity={fill_opacity};"
                    f"strokeWidth={DIAGRAM_STROKE_WIDTH};"
                    f"fontFamily=NVIDIA Sans;fontSize={font_size};fontStyle=1;"
                    "align=center;verticalAlign=middle;"
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
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"
def svg_payloads() -> dict[str, str]:
    """Return the SVG assets used by the Core playbook."""

    payloads = {
        "atomicdata-batch-fields.svg": atomicdata_batch_fields_svg(),
        "batch-row-ownership.svg": batch_row_ownership_svg(),
        "prefetch-window.svg": prefetch_window_svg(),
        "toolkit-capability-map.svg": capability_map_svg(),
        "zarr-data-flow.svg": zarr_data_flow_svg(),
    }
    for logical_name in ("fused-stage-flow.svg", "inflight-batching-flow.svg"):
        path = ASSET_DIR / logical_name
        if not path.is_file():
            raise FileNotFoundError(f"Required diagram is missing: {path}")
        payloads[logical_name] = path.read_text(encoding="utf-8")
    return payloads


def write_content_addressed_svgs(payloads: dict[str, str]) -> dict[str, object]:
    """Write stable aliases, current hashed copies, and their asset index."""

    assets: dict[str, dict[str, str]] = {}
    for logical_name in sorted(payloads):
        payload = payloads[logical_name].encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        logical_path = Path(logical_name)
        versioned_name = (
            f"{logical_path.stem}-{digest[:HASH_PREFIX_LENGTH]}{logical_path.suffix}"
        )
        for stale_path in ASSET_DIR.glob(f"{logical_path.stem}-*{logical_path.suffix}"):
            if stale_path.name != versioned_name:
                stale_path.unlink()
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
    (ASSET_DIR / "atomicdata-batch-fields.drawio").write_text(
        atomicdata_batch_fields_drawio_xml(), encoding="utf-8"
    )
    (ASSET_DIR / "batch-row-ownership.drawio").write_text(
        batch_row_ownership_drawio_xml(), encoding="utf-8"
    )
    (ASSET_DIR / "prefetch-window.drawio").write_text(
        prefetch_window_drawio_xml(), encoding="utf-8"
    )
    (ASSET_DIR / "toolkit-capability-map.drawio").write_text(
        capability_drawio_xml(), encoding="utf-8"
    )
    (ASSET_DIR / "zarr-data-flow.drawio").write_text(
        zarr_drawio_xml(), encoding="utf-8"
    )
    index = write_content_addressed_svgs(svg_payloads())
    print(f"Generated {len(index['assets'])} Core SVGs")


if __name__ == "__main__":
    main()
