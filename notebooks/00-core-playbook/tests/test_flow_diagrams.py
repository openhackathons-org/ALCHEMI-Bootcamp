from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
FLOW_NAMES = ("fused-stage-flow", "inflight-batching-flow")


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


def test_flow_sources_are_well_formed_xml() -> None:
    for name in FLOW_NAMES:
        assert _parse(name, "drawio").tag == "mxGraphModel"
        assert _parse(name, "svg").tag.endswith("svg")


def test_fused_stage_flow_shows_ordered_stages_and_one_shared_evaluation() -> None:
    cells = _drawio_cells("fused-stage-flow")
    drawio_text = " ".join(_plain_value(cell) for cell in cells.values())
    svg_text = (ASSET_DIR / "fused-stage-flow.svg").read_text(encoding="utf-8")

    for phrase in (
        "FusedStage: ordered stage selection, then one shared MLIP call",
        "Active Batch",
        "Stage 0 · Optimization",
        "Stage 1 · Dynamics",
        "shared model evaluation after stage selection",
        "MLIP",
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
    assert "waiting" in _plain_value(cells["d-before"])
    assert "new" not in _plain_value(cells["d-before"])

    waiting = ("d-before", "e-before", "f-before", "e-after", "f-after")
    running = ("a-before", "b-before", "a-after", "b-after", "d-after")
    complete = ("c-before", "c-after")
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


def test_flow_aliases_match_the_content_addressed_manifest() -> None:
    index = json.loads((ASSET_DIR / "core-assets.json").read_text(encoding="utf-8"))
    for name in FLOW_NAMES:
        logical_name = f"{name}.svg"
        payload = (ASSET_DIR / logical_name).read_bytes()
        entry = index["assets"][logical_name]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert (ASSET_DIR / entry["filename"]).read_bytes() == payload

    for logical_name, entry in index["assets"].items():
        versioned_files = ASSET_DIR.glob(f"{Path(logical_name).stem}-*.svg")
        assert {path.name for path in versioned_files} == {entry["filename"]}
