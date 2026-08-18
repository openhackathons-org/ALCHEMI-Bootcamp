# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
DIAGRAM_NAMES = (
    "atomicdata-batch-fields",
    "batch-row-ownership",
    "fused-stage-flow",
    "inflight-batching-flow",
    "prefetch-window",
)
CAPABILITY_NAME = "toolkit-capability-map"


def _parse(name: str, suffix: str) -> ET.Element:
    return ET.parse(ASSET_DIR / f"{name}.{suffix}").getroot()


def _drawio_cells(name: str) -> dict[str, ET.Element]:
    root = _parse(name, "drawio")
    return {cell.attrib["id"]: cell for cell in root.findall(".//mxCell")}


def _plain_value(cell: ET.Element) -> str:
    value = html.unescape(cell.attrib.get("value", ""))
    return re.sub(r"<[^>]+>", " ", value)


def _style_value(cell: ET.Element, key: str) -> str | None:
    for declaration in cell.attrib.get("style", "").split(";"):
        style_key, separator, value = declaration.partition("=")
        if separator and style_key == key:
            return value
    return None


def test_diagram_sources_are_well_formed_xml() -> None:
    for name in DIAGRAM_NAMES:
        assert _parse(name, "drawio").tag == "mxGraphModel"
        assert _parse(name, "svg").tag.endswith("svg")


def test_atomicdata_batch_diagram_names_field_ownership() -> None:
    cells = _drawio_cells("atomicdata-batch-fields")
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values())
    svg_text = (ASSET_DIR / "atomicdata-batch-fields.svg").read_text(
        encoding="utf-8"
    )

    for phrase in (
        "AtomicData",
        "Batch",
        "3 systems · 24 atoms · no padding",
        "Node fields",
        "System fields",
        "Neighbor data",
        "default MATRIX per atom",
        "optional COO edge list [E, 2]",
        "absent in this fresh Batch",
        "batch_ptr [0, 4, 11, 24]",
    ):
        assert phrase in drawio_text
        assert phrase in svg_text

    system_cards = (cells["ammonia"], cells["propyne"], cells["phenol"])
    assert {_style_value(cell, "fillColor") for cell in system_cards} == {"#28312B"}
    assert {_style_value(cell, "strokeColor") for cell in system_cards} == {
        "#4B565F"
    }
    for name in ("ammonia", "propyne", "phenol"):
        icon_text = (ASSET_DIR / "molecule-icons" / f"{name}.svg").read_text(
            encoding="utf-8"
        )
        assert f'id="molecule-{name}"' in svg_text
        assert _style_value(cells[f"icon-{name}"], "shape") == "image"
        geometry = cells[f"icon-{name}"].find("mxGeometry")
        assert geometry is not None
        assert geometry.attrib["width"] == "96"
        assert geometry.attrib["height"] == "90"
        radii = [float(value) for value in re.findall(r'<circle[^>]+r="([0-9.]+)"', icon_text)]
        bond_widths = [
            float(value)
            for value in re.findall(r'<line[^>]+stroke-width="([0-9.]+)"', icon_text)
        ]
        assert min(radii) >= 5.4
        assert max(radii) <= 25.0
        assert min(bond_widths) >= 3.3
        assert 'role="img"' in icon_text
        assert f'aria-label="{name.title()} molecule"' in icon_text
    assert "<radialGradient" in svg_text
    assert "<feDropShadow" in svg_text
    assert svg_text.count('filter="url(#molecule-shadow)"') == 3
    svg_ids = re.findall(r' id="([^"]+)"', svg_text)
    assert len(svg_ids) == len(set(svg_ids))
    assert not {
        "batch-node",
        "batch-system",
        "batch-edge",
    }.intersection(cells)


def test_prefetch_diagram_compares_memory_and_waiting() -> None:
    cells = _drawio_cells("prefetch-window")
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values()).lower()
    svg_text = (ASSET_DIR / "prefetch-window.svg").read_text(
        encoding="utf-8"
    ).lower()

    for phrase in (
        "Prefetch trades device memory for less waiting",
        "prefetch_factor = 0",
        "prefetch_factor = 2",
        "Lower device memory",
        "Higher device memory",
        "load B1",
        "run B1",
        "prefetch B2",
        "prefetch B3",
        "TARGET DEVICE MEMORY",
        "B1 · current",
        "B2 · prefetched",
        "B3 · prefetched",
        "1 Batch on device",
        "3 Batches on device",
    ):
        assert phrase.lower() in drawio_text
        assert phrase.lower() in svg_text

    sync_memory = cells["sync-memory"].find("mxGeometry")
    prefetch_memory = cells["prefetch-memory"].find("mxGeometry")
    assert sync_memory is not None
    assert prefetch_memory is not None
    assert sync_memory.attrib["x"] == prefetch_memory.attrib["x"] == "700"
    assert sync_memory.attrib["width"] == prefetch_memory.attrib["width"] == "178"

    for node_id in (
        "sync-load1",
        "sync-compute1",
        "sync-load2",
        "sync-compute2",
        "sync-load3",
        "sync-compute3",
        "prefetch-load1",
        "prefetch-compute1",
        "prefetch-compute2",
        "prefetch-compute3",
        "prefetch-load2",
        "prefetch-load3",
    ):
        geometry = cells[node_id].find("mxGeometry")
        assert geometry is not None
        assert int(geometry.attrib["width"]) >= 66
        assert int(geometry.attrib["height"]) >= 38




def test_new_core_diagrams_share_a_readable_type_scale() -> None:
    for name in (
        "atomicdata-batch-fields",
        "batch-row-ownership",
        "prefetch-window",
    ):
        svg_text = (ASSET_DIR / f"{name}.svg").read_text(encoding="utf-8")
        root = _parse(name, "svg")
        assert root.attrib["viewBox"].split()[2] == "920"
        sizes = {float(value) for value in re.findall(r'font-size="([0-9.]+)', svg_text)}
        assert {10.0, 11.0, 13.0, 17.0} <= sizes
        assert min(sizes) >= 10.0


def test_capability_map_uses_three_core_modules_and_neutral_connectors() -> None:
    cells = _drawio_cells(CAPABILITY_NAME)
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values())
    svg_text = (ASSET_DIR / f"{CAPABILITY_NAME}.svg").read_text(encoding="utf-8")

    for phrase in (
        "01 · Data and batching",
        "AtomicData · Batch",
        "GPU batches · Zarr",
        "02 · Models and simulation",
        "AIMNet2 · hooks",
        "FIRE2 · saved state",
        "03 · Compose and scale",
        "interaction models",
        "pipelines · distributed",
    ):
        assert phrase in drawio_text
        assert phrase in svg_text

    for index in range(1, 4):
        assert _style_value(cells[f"capability-edge-{index}"], "endArrow") == "none"
    assert "04 · Training and fine-tuning" not in drawio_text
    assert "FineTuningStrategy" not in svg_text
    assert "DEEP DIVES" not in svg_text
    assert "v3-deep-dives" not in svg_text
    assert 'id="cap-arrow"' not in svg_text
    assert 'marker-end="url(#cap-arrow)"' not in svg_text
    assert 'class="cap-tooltip"' not in svg_text
    assert ":hover" not in svg_text
    assert "transition:" not in svg_text
    assert "Adapt and scale" not in drawio_text


def test_fused_stage_flow_shows_ordered_stages_and_one_shared_evaluation() -> None:
    cells = _drawio_cells("fused-stage-flow")
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values())
    svg_text = (ASSET_DIR / "fused-stage-flow.svg").read_text(encoding="utf-8")

    for phrase in (
        "FusedStage · one fixed active Batch on one device or rank",
        "Active Batch",
        "Stage 0 · Optimization",
        "Stage 1 · Dynamics",
        "shared model evaluation after stage selection",
        "composed model",
        "model(batch)",
        "Next fused iteration",
        "updated model outputs and stage status",
    ):
        assert phrase in drawio_text
        assert phrase in svg_text
    stage_nodes = ("batch", "stage0", "stage1", "continue")
    assert {_style_value(cells[cell_id], "fillColor") for cell_id in stage_nodes} == {
        "#20262C"
    }
    assert _style_value(cells["model"], "fillColor") == "#76B900"

    expected_edges = {
        "edge-batch-stage0",
        "edge-stage0-stage1",
        "edge-stage1-model",
        "edge-model-continue",
        "edge-continue-batch",
    }
    assert expected_edges <= cells.keys()


def test_inflight_flow_uses_status_colors_and_equal_vertical_arrows() -> None:
    cells = _drawio_cells("inflight-batching-flow")
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values())
    svg_text = (ASSET_DIR / "inflight-batching-flow.svg").read_text(encoding="utf-8")
    scope_title = "Inflight batching · one refilled active Batch on one device or rank"
    assert scope_title in drawio_text
    assert scope_title in svg_text
    assert "waiting" in _plain_value(cells["d-before"])
    assert "new" not in _plain_value(cells["d-before"])

    waiting = ("d-before", "e-before", "f-before", "e-after", "f-after")
    running = ("a-before", "c-before", "a-after", "c-after", "d-after")
    complete = ("b-before", "b-after")
    assert {_style_value(cells[cell_id], "fillColor") for cell_id in waiting} == {
        "#2B3238"
    }
    assert {_style_value(cells[cell_id], "fillColor") for cell_id in running} == {
        "#314A59"
    }
    assert {_style_value(cells[cell_id], "fillColor") for cell_id in complete} == {
        "#3A332C"
    }

    endpoints = []
    for edge_id in ("edge-before-refill", "edge-refill-after"):
        geometry = cells[edge_id].find("mxGeometry")
        assert geometry is not None
        source = geometry.find("mxPoint[@as='sourcePoint']")
        target = geometry.find("mxPoint[@as='targetPoint']")
        assert source is not None and target is not None
        endpoints.append(
            (
                float(source.attrib["x"]),
                float(source.attrib["y"]),
                float(target.attrib["x"]),
                float(target.attrib["y"]),
            )
        )
    lengths = []
    for source_x, source_y, target_x, target_y in endpoints:
        assert source_x == target_x == 460.0
        lengths.append(target_y - source_y)
    assert lengths == [34.0, 34.0]


def test_asset_aliases_match_the_content_addressed_manifest() -> None:
    index = json.loads((ASSET_DIR / "core-assets.json").read_text(encoding="utf-8"))
    for logical_name, entry in index["assets"].items():
        payload = (ASSET_DIR / logical_name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert (ASSET_DIR / entry["filename"]).read_bytes() == payload

        versioned_files = ASSET_DIR.glob(f"{Path(logical_name).stem}-*.svg")
        assert {path.name for path in versioned_files} == {entry["filename"]}
