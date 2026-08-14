#!/usr/bin/env python3
"""Generate editable Core diagrams and their deterministic SVG presentations."""

from __future__ import annotations

import hashlib
import html
import json
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parent
ASSET_INDEX_PATH = ASSET_DIR / "core-assets.json"
ASSET_INDEX_SCHEMA = "alchemi.core-assets.v1"
HASH_PREFIX_LENGTH = 16
BACKGROUND = "#11161B"
SURFACE = "#20262C"
GREEN = "#76B900"
TEXT = "#F3F4F6"
MUTED = "#A8B0B8"
QUIET = "#68737E"


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
            "Toolkit-Ops<br><font style='font-size:9px'>neighbor lists · segment operations · interactions</font>",
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
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb",
                ),
                (
                    "Part 02",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/02-zarr-data-loading/zarr-data-loading.ipynb",
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
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb",
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
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/04-hooks/hooks.ipynb",
                ),
                (
                    "Part 05",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/05-base-dynamics/base-dynamics.ipynb",
                ),
                (
                    "Part 06",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/06-gpu-pipelines-profiling/gpu-pipelines-profiling.ipynb",
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
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/07-training-finetuning/training-finetuning.ipynb",
                ),
                (
                    "Part 08",
                    "https://github.com/openhackathons-org/ALCHEMI-Bootcamp/blob/v3-deep-dives/notebooks/08-domain-decomposition/domain-decomposition.ipynb",
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
        f'<text x="660" y="87" class="sub" fill="{MUTED}">neighbor lists · segment operations · interactions</text>',
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
    """Render the pinned storage-to-selected-device data path."""

    boxes = (
        ("Zarr store", "disk or CPU storage", 20, SURFACE, TEXT, MUTED),
        ("Reader", "CPU tensors", 200, SURFACE, TEXT, MUTED),
        ("Dataset", "validate + move", 380, SURFACE, TEXT, MUTED),
        ("DataLoader", "batch + prefetch", 560, SURFACE, TEXT, MUTED),
        ("Batch", "selected device", 740, GREEN, "#081005", "#15200D"),
    )
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="96" '
            'viewBox="0 0 920 96" role="img" aria-labelledby="title desc">'
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
        f'<rect width="920" height="96" rx="8" fill="{BACKGROUND}"/>',
        '<g font-family="NVIDIA Sans,Arial,sans-serif" text-anchor="middle">',
    ]
    for label, detail, x, fill, text, detail_text in boxes:
        parts.extend(
            [
                f'<rect x="{x}" y="18" width="160" height="60" rx="8" fill="{fill}"/>',
                (
                    f'<text x="{x + 80:g}" y="43" fill="{text}" '
                    f'font-size="14" font-weight="700">{html.escape(label)}</text>'
                ),
                (
                    f'<text x="{x + 80:g}" y="63" fill="{detail_text}" '
                    f'font-size="11">{html.escape(detail)}</text>'
                ),
            ]
        )
    for left, right in pairwise(boxes):
        left_edge = left[2] + 167
        right_edge = right[2] - 7
        parts.append(
            f'<path d="M{left_edge} 48H{right_edge}" '
            f'stroke="{QUIET}" stroke-width="1.25" '
            'marker-end="url(#zarr-flow-arrow)"/>'
        )
    parts.extend(["</g>", "</svg>\n"])
    return "".join(parts)


def zarr_drawio_xml() -> str:
    """Return an editable draw.io source for the Zarr data path."""

    root = ET.Element("mxGraphModel", {"adaptiveColors": "auto", "page": "0"})
    graph_root = ET.SubElement(root, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    nodes = (
        (
            "zarr",
            "Zarr store<br><font color='#A8B0B8'>disk or CPU storage</font>",
            20,
            SURFACE,
            TEXT,
        ),
        (
            "reader",
            "Reader<br><font color='#A8B0B8'>CPU tensors</font>",
            200,
            SURFACE,
            TEXT,
        ),
        (
            "dataset",
            "Dataset<br><font color='#A8B0B8'>validate + move</font>",
            380,
            SURFACE,
            TEXT,
        ),
        (
            "loader",
            "DataLoader<br><font color='#A8B0B8'>batch + prefetch</font>",
            560,
            SURFACE,
            TEXT,
        ),
        (
            "batch",
            "Batch<br><font color='#15200D'>selected device</font>",
            740,
            GREEN,
            "#081005",
        ),
    )
    for node_id, label, x, fill, font_color in nodes:
        cell = ET.SubElement(
            graph_root,
            "mxCell",
            {
                "id": node_id,
                "value": label,
                "style": (
                    "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
                    f"fillColor={fill};strokeColor=none;fontColor={font_color};"
                    "fontFamily=NVIDIA Sans;fontSize=14;fontStyle=1;"
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
                    f"strokeColor={QUIET};strokeWidth=1.25;endArrow=block;endFill=1;"
                ),
                "edge": "1",
                "parent": "1",
                "source": source[0],
                "target": target[0],
            },
        )
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    return ET.tostring(root, encoding="unicode") + "\n"


def svg_payloads() -> dict[str, str]:
    """Return the four SVG assets used by the Core playbook."""

    payloads = {
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
        for stale_path in ASSET_DIR.glob(
            f"{logical_path.stem}-*{logical_path.suffix}"
        ):
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
