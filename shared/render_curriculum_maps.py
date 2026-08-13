"""Generate compact, linked ALCHEMI curriculum maps."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from itertools import pairwise
from math import hypot
from pathlib import Path


@dataclass(frozen=True)
class Lesson:
    part: str
    title: str
    icon: str
    primary_capability: str
    notebook: str | None = None


@dataclass(frozen=True)
class Capability:
    key: str
    heading: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Relation:
    part: str
    capability: str
    route: str = "direct"


@dataclass(frozen=True)
class CapabilityRelation:
    source: str
    target: str
    lane: int | None = None


LESSONS = (
    Lesson(
        "01",
        "AtomicData + Batch",
        "atomic-data",
        "C1",
        "../notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb",
    ),
    Lesson(
        "02",
        "Zarr data loading",
        "database",
        "C2",
        "../notebooks/02-zarr-data-loading/zarr-data-loading.ipynb",
    ),
    Lesson(
        "03",
        "Models + composition",
        "model",
        "C3",
        "../notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb",
    ),
    Lesson("04", "Hooks", "hook", "C4", "../notebooks/04-hooks/hooks.ipynb"),
    Lesson(
        "05",
        "BaseDynamics",
        "dynamics",
        "C4",
        "../notebooks/05-base-dynamics/base-dynamics.ipynb",
    ),
    Lesson("06", "GPU pipelines + profiling", "gpu", "C4"),
    Lesson("07", "Training + fine-tuning", "training", "C5"),
    Lesson("08", "Domain decomposition", "domains", "C6"),
)

CAPABILITIES = (
    Capability("C1", "FUNDAMENTALS", ("AtomicData · Batch · GPU tensors",)),
    Capability("C2", "DATA MANAGEMENT", ("Read · stream · save", "Datasets and trajectories")),
    Capability(
        "C3",
        "MODELS + POTENTIALS",
        ("Use MLIPs · wrap custom models", "Compose physical terms"),
    ),
    Capability(
        "C4",
        "SIMULATION WORKFLOWS",
        ("Observe · control · relax", "Simulate · screen · profile"),
    ),
    Capability("C5", "MODEL DEVELOPMENT", ("Train · validate · fine-tune",)),
    Capability(
        "C6",
        "MULTI-GPU EXECUTION",
        ("Domain decomposition", "Large systems across GPUs"),
    ),
)

RELATIONS = (
    Relation("01", "C1"),
    Relation("02", "C2"),
    Relation("03", "C3"),
    Relation("04", "C3", "hook-model"),
    Relation("04", "C4", "hook-workflow"),
    Relation("05", "C4"),
    Relation("06", "C4", "pipeline-workflow"),
    Relation("07", "C5"),
    Relation("08", "C6"),
)

CAPABILITY_RELATIONS = (
    CapabilityRelation("C1", "C2"),
    CapabilityRelation("C1", "C3", lane=0),
    CapabilityRelation("C2", "C3"),
    CapabilityRelation("C2", "C4", lane=1),
    CapabilityRelation("C2", "C5", lane=2),
    CapabilityRelation("C3", "C4"),
    CapabilityRelation("C3", "C5", lane=3),
    CapabilityRelation("C3", "C6", lane=0),
    CapabilityRelation("C4", "C6", lane=1),
)

WIDTH = 900
HEIGHT = 552
LESSON_X = 24
LESSON_WIDTH = 282
LESSON_HEIGHT = 42
LESSON_Y0 = 42
LESSON_STEP = 63
CAPABILITY_X = 500
CAPABILITY_WIDTH = 300
CAPABILITY_TOP = 36
CAPABILITY_GAP = 18
CAPABILITY_HEADING_OFFSET = 18
CAPABILITY_BODY_OFFSET = 42
CAPABILITY_LINE_STEP = 18
CAPABILITY_BOTTOM_PADDING = 12
CAPABILITY_LANE_X = (820, 838, 856, 874)


ICON_PATHS = {
    "atomic-data": """
      <circle cx="7" cy="8" r="1.4"/><circle cx="17" cy="8" r="1.4"/>
      <circle cx="12" cy="17" r="1.4"/><path d="M8.5 8h7M7.8 9.2l3.4 6.4m5-6.4-3.4 6.4"/>
    """,
    "database": """
      <ellipse cx="12" cy="6" rx="6.5" ry="2.5"/>
      <path d="M5.5 6v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5V6M5.5 12v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-6"/>
    """,
    "model": """
      <path d="m4 8 8-4 8 4-8 4zM4 12l8 4 8-4M4 16l8 4 8-4"/>
    """,
    "hook": """
      <path d="M8 4v9a4 4 0 0 0 8 0v-1m-3 0 3-3 3 3"/>
    """,
    "dynamics": """
      <path d="M19 8a8 8 0 0 0-13-2L4 8m0-4v4h4M5 16a8 8 0 0 0 13 2l2-2m0 4v-4h-4"/>
    """,
    "gpu": """
      <rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9" y="9" width="6" height="6"/>
      <path d="M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m12 0h3M3 15h3m12 0h3"/>
    """,
    "training": """
      <path d="M5 5v14h14M8 15l4-4 3 2 4-6"/>
    """,
    "domains": """
      <rect x="4" y="4" width="5.5" height="5.5" rx="0.8"/>
      <rect x="10.5" y="4" width="5.5" height="5.5" rx="0.8"/>
      <rect x="4" y="10.5" width="5.5" height="5.5" rx="0.8"/>
      <rect x="15.5" y="15.5" width="5.5" height="5.5" rx="0.8"/>
      <path d="M11.5 11.5l2.8 2.8"/>
    """,
}


def _lesson(part: str) -> Lesson:
    return next(item for item in LESSONS if item.part == part)


def _capability(key: str) -> Capability:
    return next(item for item in CAPABILITIES if item.key == key)


def _capability_height(item: Capability) -> int:
    return (
        CAPABILITY_BODY_OFFSET
        + CAPABILITY_LINE_STEP * (len(item.lines) - 1)
        + CAPABILITY_BOTTOM_PADDING
    )


def _capability_y(key: str) -> int:
    item = _capability(key)
    index = CAPABILITIES.index(item)
    prior_height = sum(_capability_height(prior) for prior in CAPABILITIES[:index])
    return CAPABILITY_TOP + prior_height + CAPABILITY_GAP * index


def _lesson_y(part: str) -> float:
    index = LESSONS.index(_lesson(part))
    return LESSON_Y0 + index * LESSON_STEP


def _lesson_center(part: str) -> float:
    return _lesson_y(part) + LESSON_HEIGHT / 2


def _capability_center(key: str) -> float:
    item = _capability(key)
    return _capability_y(key) + _capability_height(item) / 2


def _capability_relation_points(
    relation: CapabilityRelation,
) -> tuple[tuple[float, float], ...]:
    source = _capability(relation.source)
    target = _capability(relation.target)
    source_bottom = _capability_y(source.key) + _capability_height(source)
    target_top = _capability_y(target.key)

    if relation.lane is None:
        center_x = CAPABILITY_X + CAPABILITY_WIDTH / 2
        return ((center_x, source_bottom), (center_x, target_top))

    lane_x = CAPABILITY_LANE_X[relation.lane]
    source_y = _capability_center(source.key)
    target_y = _capability_center(target.key)
    card_edge = CAPABILITY_X + CAPABILITY_WIDTH
    return (
        (card_edge, source_y),
        (lane_x, source_y),
        (lane_x, target_y),
        (card_edge, target_y),
    )


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _rounded_path(points: tuple[tuple[float, float], ...], radius: float = 7) -> str:
    commands = [f"M{_number(points[0][0])} {_number(points[0][1])}"]
    for previous, corner, following in zip(points, points[1:-1], points[2:]):
        incoming = (previous[0] - corner[0], previous[1] - corner[1])
        outgoing = (following[0] - corner[0], following[1] - corner[1])
        incoming_length = hypot(*incoming)
        outgoing_length = hypot(*outgoing)
        corner_radius = min(radius, incoming_length / 2, outgoing_length / 2)
        before = (
            corner[0] + incoming[0] / incoming_length * corner_radius,
            corner[1] + incoming[1] / incoming_length * corner_radius,
        )
        after = (
            corner[0] + outgoing[0] / outgoing_length * corner_radius,
            corner[1] + outgoing[1] / outgoing_length * corner_radius,
        )
        commands.append(f"L{_number(before[0])} {_number(before[1])}")
        commands.append(
            f"Q{_number(corner[0])} {_number(corner[1])} "
            f"{_number(after[0])} {_number(after[1])}"
        )
    commands.append(f"L{_number(points[-1][0])} {_number(points[-1][1])}")
    return "".join(commands)


def _relation_points(relation: Relation) -> tuple[tuple[float, float], ...]:
    start_x = LESSON_X + LESSON_WIDTH
    lesson_center = _lesson_center(relation.part)
    target_x = CAPABILITY_X
    target_center = _capability_center(relation.capability)

    if relation.route == "direct":
        return ((start_x, lesson_center), (target_x, target_center))
    if relation.route == "hook-model":
        item = _capability("C3")
        target_y = _capability_y("C3") + _capability_height(item) - 13
        return (
            (start_x, lesson_center - 8),
            (400, lesson_center - 8),
            (400, target_y),
            (target_x, target_y),
        )
    if relation.route == "hook-workflow":
        target_y = _capability_y("C4") + 21
        return (
            (start_x, lesson_center + 8),
            (430, lesson_center + 8),
            (430, target_y),
            (target_x, target_y),
        )
    if relation.route == "pipeline-workflow":
        item = _capability("C4")
        target_y = _capability_y("C4") + _capability_height(item) - 21
        return (
            (start_x, lesson_center),
            (460, lesson_center),
            (460, target_y),
            (target_x, target_y),
        )
    raise ValueError(f"Unknown relation route: {relation.route}")


def _icon(name: str, x: int, y: int, color: str) -> str:
    return f"""
    <g transform="translate({x} {y}) scale(0.72)" fill="none" stroke="{color}"
       stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">
      {ICON_PATHS[name].strip()}
    </g>"""


def _text_lines(
    lines: tuple[str, ...], *, x: int, y: int, color: str, size: int, step: int
) -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else step}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}">{spans}</text>'


def _connector(path: str, *, current: bool, kind: str) -> str:
    color = "#76B900" if current else "#66727F"
    width = "2" if current else "1.25"
    marker = "arrow-current" if current else "arrow-muted"
    return (
        f'<path data-connector="{kind}" d="{path}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#{marker})"/>'
    )


def render_map(current_part: str) -> str:
    current_lesson = _lesson(current_part)
    current_capability = current_lesson.primary_capability
    out = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'role="img" aria-labelledby="title-{current_part} desc-{current_part}">'
        ),
        f'<title id="title-{current_part}">ALCHEMI Toolkit curriculum, Part {current_part}</title>',
        (
            f'<desc id="desc-{current_part}">A compact eight-part course map. '
            "Available lesson rows open their notebooks. Capability arrows show how "
            "fundamentals and data management support model, simulation, training, "
            "and multi-GPU work. Hooks connect to models and simulation workflows. "
            f"Part {current_part} and its primary capability use NVIDIA green.</desc>"
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="16" fill="#0B0F10"/>',
        """<defs>
          <marker id="arrow-muted" markerUnits="userSpaceOnUse" markerWidth="6" markerHeight="6" refX="5.5" refY="3" orient="auto"><path d="M0 0 6 3 0 6z" fill="#66727F"/></marker>
          <marker id="arrow-current" markerUnits="userSpaceOnUse" markerWidth="6" markerHeight="6" refX="5.5" refY="3" orient="auto"><path d="M0 0 6 3 0 6z" fill="#76B900"/></marker>
        </defs>""",
        '<g font-family="NVIDIA Sans, Arial, sans-serif">',
        '<style>.lesson-link{cursor:pointer}.lesson-link:hover rect,.lesson-link:focus rect{stroke:#A7D84B;stroke-width:2}.lesson-link:focus{outline:none}</style>',
        '<text x="24" y="24" fill="#9BA4AE" font-size="11" font-weight="650" letter-spacing="1.25">COURSE</text>',
        '<text x="500" y="24" fill="#9BA4AE" font-size="11" font-weight="650" letter-spacing="1.25">CAPABILITIES</text>',
    ]

    spine_x = LESSON_X + LESSON_WIDTH / 2
    for index in range(len(LESSONS) - 1):
        start_y = LESSON_Y0 + index * LESSON_STEP + LESSON_HEIGHT
        end_y = LESSON_Y0 + (index + 1) * LESSON_STEP - 5
        out.append(
            _connector(
                f"M{_number(spine_x)} {_number(start_y)}V{_number(end_y)}",
                current=False,
                kind="course",
            )
        )

    for relation in RELATIONS:
        points = _relation_points(relation)
        current = relation.part == current_part
        out.append(
            _connector(
                _rounded_path(points),
                current=current,
                kind=f"{relation.part}-{relation.capability}",
            )
        )

    for relation in CAPABILITY_RELATIONS:
        out.append(
            _connector(
                _rounded_path(_capability_relation_points(relation), radius=6),
                current=False,
                kind=f"{relation.source}-{relation.target}",
            )
        )

    for index, lesson in enumerate(LESSONS):
        y = _lesson_y(lesson.part)
        current = lesson.part == current_part
        fill = "#76B900" if current else ("#20252B" if index < 5 else "#3A332C")
        text_color = "#050505" if current else "#F3F4F6"
        quiet_color = "#202428" if current else "#AEB6BE"
        icon_color = "#111416" if current else "#B7BEC5"
        link_start = ""
        link_end = ""
        clickable = ""
        if lesson.notebook is not None:
            label = escape(f"Open Part {lesson.part}: {lesson.title}")
            link_start = (
                f'<a class="lesson-link" href="{escape(lesson.notebook)}" '
                f'target="_top" aria-label="{label}"><title>{label}</title>'
            )
            link_end = "</a>"
            clickable = ' data-clickable="true"'
        out.extend(
            [
                link_start,
                f'<g id="part-{lesson.part}" data-icon="{lesson.icon}"{clickable}>',
                (
                    f'<rect x="{LESSON_X}" y="{_number(y)}" width="{LESSON_WIDTH}" '
                    f'height="{LESSON_HEIGHT}" rx="9" fill="{fill}" '
                    f'stroke="{fill if current else "#30363D"}"/>'
                ),
                _icon(lesson.icon, LESSON_X + 13, int(y + 11), icon_color),
                (
                    f'<text x="{LESSON_X + 42}" y="{_number(y + 27)}" fill="{quiet_color}" '
                    f'font-size="11" font-weight="650" letter-spacing="0.65">{lesson.part}</text>'
                ),
                (
                    f'<text x="{LESSON_X + 72}" y="{_number(y + 28)}" fill="{text_color}" '
                    f'font-size="15" font-weight="600">{escape(lesson.title)}</text>'
                ),
                "</g>",
                link_end,
            ]
        )

    for capability in CAPABILITIES:
        capability_y = _capability_y(capability.key)
        capability_height = _capability_height(capability)
        current = capability.key == current_capability
        fill = "#76B900" if current else "#151A1F"
        heading_color = "#15191C" if current else "#9BA4AE"
        body_color = "#050505" if current else "#F3F4F6"
        out.extend(
            [
                f'<g id="capability-{capability.key}">',
                (
                    f'<rect x="{CAPABILITY_X}" y="{capability_y}" width="{CAPABILITY_WIDTH}" '
                    f'height="{capability_height}" rx="10" fill="{fill}" '
                    f'stroke="{fill if current else "#262C33"}"/>'
                ),
                (
                    f'<text x="{CAPABILITY_X + 20}" '
                    f'y="{capability_y + CAPABILITY_HEADING_OFFSET}" fill="{heading_color}" '
                    f'font-size="10" font-weight="700" letter-spacing="0.95">{capability.heading}</text>'
                ),
                _text_lines(
                    capability.lines,
                    x=CAPABILITY_X + 20,
                    y=capability_y + CAPABILITY_BODY_OFFSET,
                    color=body_color,
                    size=14,
                    step=CAPABILITY_LINE_STEP,
                ),
                "</g>",
            ]
        )

    out.extend(["</g>", "</svg>", ""])
    return "\n".join(out)


def render_drawio() -> str:
    """Return an editable, part-neutral draw.io source using the same graph."""
    cells = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>',
    ]
    lesson_style = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#20252B;strokeColor=#30363D;"
        "fontColor=#F3F4F6;fontFamily=NVIDIA Sans;fontSize=16;align=left;spacingLeft=18;"
    )
    capability_style = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#151A1F;strokeColor=#262C33;"
        "fontColor=#F3F4F6;fontFamily=NVIDIA Sans;fontSize=14;align=left;spacingLeft=18;"
    )
    edge_style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#66727F;endArrow=block;endFill=1;"
    )

    for lesson in LESSONS:
        link = f' link="{escape(lesson.notebook)}"' if lesson.notebook else ""
        value = escape(f"{lesson.part}   {lesson.title}")
        cells.append(
            f'<mxCell id="part-{lesson.part}" value="{value}" style="{lesson_style}" '
            f'vertex="1" parent="1"{link}><mxGeometry x="{LESSON_X}" y="{_number(_lesson_y(lesson.part))}" '
            f'width="{LESSON_WIDTH}" height="{LESSON_HEIGHT}" as="geometry"/></mxCell>'
        )

    for capability in CAPABILITIES:
        capability_y = _capability_y(capability.key)
        capability_height = _capability_height(capability)
        value = escape(f"{capability.heading}\n" + "\n".join(capability.lines))
        cells.append(
            f'<mxCell id="capability-{capability.key}" value="{value}" style="{capability_style}" '
            f'vertex="1" parent="1"><mxGeometry x="{CAPABILITY_X}" y="{capability_y}" '
            f'width="{CAPABILITY_WIDTH}" height="{capability_height}" as="geometry"/></mxCell>'
        )

    for index, (source, target) in enumerate(pairwise(LESSONS), start=1):
        cells.append(
            f'<mxCell id="course-{index}" style="{edge_style}" edge="1" parent="1" '
            f'source="part-{source.part}" target="part-{target.part}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    for index, relation in enumerate(RELATIONS, start=1):
        cells.append(
            f'<mxCell id="relation-{index}" style="{edge_style}" edge="1" parent="1" '
            f'source="part-{relation.part}" target="capability-{relation.capability}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    for index, relation in enumerate(CAPABILITY_RELATIONS, start=1):
        cells.append(
            f'<mxCell id="capability-relation-{index}" style="{edge_style}" edge="1" parent="1" '
            f'source="capability-{relation.source}" target="capability-{relation.target}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    graph = "\n        ".join(cells)
    return f"""<mxfile host="app.diagrams.net" agent="Codex" version="24.7.17">
  <diagram id="alchemi-curriculum" name="Curriculum">
    <mxGraphModel dx="900" dy="552" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="900" pageHeight="552" math="0" shadow="0" adaptiveColors="auto">
      <root>
        {graph}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    for lesson in LESSONS:
        path = output_dir / f"curriculum-map-{lesson.part}.svg"
        path.write_text(render_map(lesson.part), encoding="utf-8")
    (output_dir / "curriculum-map.drawio").write_text(
        render_drawio(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
