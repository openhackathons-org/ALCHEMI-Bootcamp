#!/usr/bin/env python3
"""Build deterministic tutorial visual assets.

The assets are intentionally SVG: they are small, reviewable, and can be
regenerated without depending on a model, design tool, or binary template.
They use a restrained NVIDIA-inspired palette for notebook banners,
icons, and workflow canvases.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ICONS = ASSETS / "icons"

NV_GREEN = "#76B900"
CHARCOAL = "#111417"
GRAPHITE = "#1F2328"
MID = "#4B5563"
LIGHT = "#F4F7F5"
MUTED = "#B8C2BF"
BLUE = "#5DADEC"
AMBER = "#F5B342"
RED = "#EF6C73"


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <defs>
    <style>
      .title {{ font: 700 42px Arial, Helvetica, sans-serif; fill: {LIGHT}; letter-spacing: 0; }}
      .subtitle {{ font: 400 21px Arial, Helvetica, sans-serif; fill: {MUTED}; letter-spacing: 0; }}
      .label {{ font: 700 17px Arial, Helvetica, sans-serif; fill: {LIGHT}; letter-spacing: 0; }}
      .small {{ font: 400 13px Arial, Helvetica, sans-serif; fill: {MUTED}; letter-spacing: 0; }}
      .tiny {{ font: 400 11px Arial, Helvetica, sans-serif; fill: {MUTED}; letter-spacing: 0; }}
    </style>
  </defs>
{body}
</svg>
"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def atom(cx: float, cy: float, r: float, fill: str, stroke: str = LIGHT) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def surface(x: float, y: float, cols: int = 6, rows: int = 2, dx: float = 34, dy: float = 28) -> str:
    parts = []
    for j in range(rows):
        for i in range(cols):
            parts.append(atom(x + i * dx + (j % 2) * dx / 2, y + j * dy, 11, MID, GRAPHITE))
    return "\n".join(parts)


def icon_frame(title: str, body: str, note: str) -> str:
    return svg(
        420,
        300,
        f"""  <rect width="420" height="300" rx="0" fill="{CHARCOAL}"/>
  <rect x="18" y="18" width="384" height="264" rx="10" fill="{GRAPHITE}" stroke="{NV_GREEN}" stroke-width="2"/>
  <text x="34" y="55" class="label">{title}</text>
{body}
  <text x="34" y="258" class="small">{note}</text>""",
    )


def build_icons() -> None:
    co = "\n".join(
        [
            surface(92, 180, cols=6, rows=2),
            atom(196, 105, 13, "#3B82F6"),
            atom(196, 75, 11, "#EF4444"),
            '<line x1="196" y1="92" x2="196" y2="84" stroke="#E5E7EB" stroke-width="4"/>',
            '<path d="M112 145 C145 105, 185 105, 196 136 C210 105, 250 105, 284 145" fill="none" stroke="#76B900" stroke-width="4"/>',
            '<text x="92" y="132" class="tiny">top</text>',
            '<text x="178" y="132" class="tiny">hollow</text>',
            '<text x="270" y="132" class="tiny">bridge</text>',
        ]
    )
    write(
        ICONS / "icon_configuration_search.svg",
        icon_frame(
            "Configuration search",
            co,
            "CO/Pd(111): many starts, one lower-energy site",
        ),
    )

    water = "\n".join(
        [
            surface(96, 180, cols=6, rows=2),
            atom(205, 112, 14, "#EF4444"),
            atom(180, 92, 8, "#E5E7EB"),
            atom(230, 92, 8, "#E5E7EB"),
            '<line x1="205" y1="112" x2="180" y2="92" stroke="#E5E7EB" stroke-width="3"/>',
            '<line x1="205" y1="112" x2="230" y2="92" stroke="#E5E7EB" stroke-width="3"/>',
            '<path d="M130 150 C165 132, 245 132, 280 150" fill="none" stroke="#5DADEC" stroke-width="4"/>',
            '<text x="118" y="132" class="tiny">first binding step only</text>',
        ]
    )
    write(
        ICONS / "icon_water_first_binding.svg",
        icon_frame(
            "Water sorption motif",
            water,
            "H2O binding is not a full pore-filling model",
        ),
    )

    oer = "\n".join(
        [
            surface(96, 180, cols=6, rows=2),
            atom(195, 112, 13, "#EF4444"),
            atom(220, 92, 8, "#E5E7EB"),
            '<line x1="195" y1="112" x2="220" y2="92" stroke="#E5E7EB" stroke-width="3"/>',
            '<path d="M120 145 L195 112 L285 145" fill="none" stroke="#F5B342" stroke-width="4"/>',
            '<text x="126" y="128" class="tiny">oxide site</text>',
            '<text x="236" y="128" class="tiny">OH/H2O*</text>',
        ]
    )
    write(
        ICONS / "icon_oer_first_adsorption.svg",
        icon_frame(
            "OER adsorption motif",
            oer,
            "Adsorption geometry, not electrochemical free energy",
        ),
    )

    nh3 = "\n".join(
        [
            surface(96, 180, cols=6, rows=2),
            atom(205, 105, 13, "#60A5FA"),
            atom(178, 84, 8, "#E5E7EB"),
            atom(232, 84, 8, "#E5E7EB"),
            atom(205, 72, 8, "#E5E7EB"),
            '<line x1="205" y1="105" x2="178" y2="84" stroke="#E5E7EB" stroke-width="3"/>',
            '<line x1="205" y1="105" x2="232" y2="84" stroke="#E5E7EB" stroke-width="3"/>',
            '<line x1="205" y1="105" x2="205" y2="72" stroke="#E5E7EB" stroke-width="3"/>',
            '<path d="M156 138 C190 125, 220 125, 254 138" fill="none" stroke="#76B900" stroke-width="4"/>',
            '<text x="154" y="128" class="tiny">N-lone-pair binding</text>',
        ]
    )
    write(
        ICONS / "icon_nh3_surface_binding.svg",
        icon_frame(
            "NH3 binding motif",
            nh3,
            "Surface binding context, not N2 dissociation kinetics",
        ),
    )


def build_banner() -> None:
    body = f"""  <rect width="1400" height="420" fill="{CHARCOAL}"/>
  <rect x="0" y="0" width="1400" height="10" fill="{NV_GREEN}"/>
  <text x="72" y="115" class="title">Batched Atomistic Simulation</text>
  <text x="72" y="157" class="subtitle">Throughput adsorption search with NVIDIA ALCHEMI Toolkit and MACE-MPA-0</text>
  <g transform="translate(790,64)">
    {surface(0, 226, cols=8, rows=3)}
    {atom(140, 102, 17, '#3B82F6')}
    {atom(140, 62, 14, '#EF4444')}
    <line x1="140" y1="85" x2="140" y2="76" stroke="#E5E7EB" stroke-width="5"/>
    <path d="M22 180 C110 95, 205 95, 286 180" fill="none" stroke="{NV_GREEN}" stroke-width="5"/>
    <path d="M80 160 C140 118, 206 118, 254 160" fill="none" stroke="{BLUE}" stroke-width="3" opacity="0.9"/>
    <text x="0" y="26" class="small">surface + adsorbate</text>
    <text x="188" y="26" class="small">batch relaxation</text>
  </g>
  <rect x="72" y="250" width="260" height="74" rx="10" fill="{GRAPHITE}" stroke="{NV_GREEN}" stroke-width="2"/>
  <text x="94" y="282" class="label">many starts</text>
  <text x="94" y="306" class="small">top, bridge, hollow, orientations</text>
  <rect x="362" y="250" width="260" height="74" rx="10" fill="{GRAPHITE}" stroke="{NV_GREEN}" stroke-width="2"/>
  <text x="384" y="282" class="label">one batch</text>
  <text x="384" y="306" class="small">independent relaxations on GPU</text>
  <rect x="652" y="250" width="260" height="74" rx="10" fill="{GRAPHITE}" stroke="{NV_GREEN}" stroke-width="2"/>
  <text x="674" y="282" class="label">rank minima</text>
  <text x="674" y="306" class="small">site, energy, validation status</text>"""
    write(ASSETS / "banner_adsorbml_toolkit.svg", svg(1400, 420, body))


def workflow_box(
    x: int,
    y: int,
    width: int,
    title: str,
    line1: str,
    line2: str,
    line3: str | None = None,
    accent: str = NV_GREEN,
) -> str:
    extra = f'\n  <text x="{x + 18}" y="{y + 112}" class="small">{line3}</text>' if line3 else ""
    return f"""  <rect x="{x}" y="{y}" width="{width}" height="140" rx="10" fill="{GRAPHITE}" stroke="{accent}" stroke-width="2"/>
  <text x="{x + 18}" y="{y + 36}" class="label">{title}</text>
  <text x="{x + 18}" y="{y + 70}" class="small">{line1}</text>
  <text x="{x + 18}" y="{y + 94}" class="small">{line2}</text>{extra}"""


def workflow_arrow(x1: int, x2: int, y: int, color: str = NV_GREEN) -> str:
    return f'<path d="M{x1} {y} H{x2}" stroke="{color}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>'


def mini_surface(x: float, y: float, cols: int = 5, rows: int = 2, dx: float = 25, dy: float = 19) -> str:
    parts = []
    for j in range(rows):
        for i in range(cols):
            cx = x + i * dx + (j % 2) * dx / 2
            cy = y + j * dy
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{MID}" stroke="#262B30" stroke-width="1.4"/>')
    return "\n".join(parts)


def molecule_icon(kind: str, cx: float, cy: float, tilt: float = 0) -> str:
    transform = f'transform="rotate({tilt:.1f} {cx:.1f} {cy:.1f})"' if tilt else ""
    if kind == "co":
        return f"""<g {transform}>
      <line x1="{cx:.1f}" y1="{cy - 9:.1f}" x2="{cx:.1f}" y2="{cy - 29:.1f}" stroke="#D8DEE4" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx:.1f}" cy="{cy - 6:.1f}" r="8.5" fill="#15191D" stroke="#E5E7EB" stroke-width="1.6"/>
      <circle cx="{cx:.1f}" cy="{cy - 33:.1f}" r="9.5" fill="#D95D5D" stroke="#F2A0A0" stroke-width="1.4"/>
    </g>"""
    if kind == "h2o":
        return f"""<g {transform}>
      <line x1="{cx:.1f}" y1="{cy - 7:.1f}" x2="{cx - 18:.1f}" y2="{cy - 25:.1f}" stroke="#D8DEE4" stroke-width="2.6" stroke-linecap="round"/>
      <line x1="{cx:.1f}" y1="{cy - 7:.1f}" x2="{cx + 19:.1f}" y2="{cy - 23:.1f}" stroke="#D8DEE4" stroke-width="2.6" stroke-linecap="round"/>
      <circle cx="{cx:.1f}" cy="{cy - 6:.1f}" r="10" fill="#D95D5D" stroke="#F2A0A0" stroke-width="1.5"/>
      <circle cx="{cx - 20:.1f}" cy="{cy - 27:.1f}" r="6.2" fill="#F4F7F5" stroke="#E5E7EB" stroke-width="1.2"/>
      <circle cx="{cx + 21:.1f}" cy="{cy - 25:.1f}" r="6.2" fill="#F4F7F5" stroke="#E5E7EB" stroke-width="1.2"/>
    </g>"""
    return f"""<g {transform}>
      <line x1="{cx - 13:.1f}" y1="{cy - 7:.1f}" x2="{cx + 12:.1f}" y2="{cy - 16:.1f}" stroke="#D8DEE4" stroke-width="2.8" stroke-linecap="round"/>
      <line x1="{cx + 16:.1f}" y1="{cy - 18:.1f}" x2="{cx + 31:.1f}" y2="{cy - 31:.1f}" stroke="#D8DEE4" stroke-width="2.4" stroke-linecap="round"/>
      <circle cx="{cx - 17:.1f}" cy="{cy - 5:.1f}" r="9" fill="#15191D" stroke="#E5E7EB" stroke-width="1.5"/>
      <circle cx="{cx + 15:.1f}" cy="{cy - 17:.1f}" r="8.5" fill="#D95D5D" stroke="#F2A0A0" stroke-width="1.4"/>
      <circle cx="{cx + 34:.1f}" cy="{cy - 34:.1f}" r="5.8" fill="#F4F7F5" stroke="#E5E7EB" stroke-width="1.1"/>
      <circle cx="{cx - 30:.1f}" cy="{cy - 22:.1f}" r="5.5" fill="#F4F7F5" stroke="#E5E7EB" stroke-width="1.1"/>
      <circle cx="{cx - 31:.1f}" cy="{cy + 8:.1f}" r="5.5" fill="#F4F7F5" stroke="#E5E7EB" stroke-width="1.1"/>
    </g>"""


def batching_panel(x: int, y: int, molecule: str, site_offset: int, tilt: float = 0) -> str:
    site_x = x + 80 + site_offset
    site_y = y + 82
    mol_y = y + 58
    return f"""  <g>
    <rect x="{x}" y="{y}" width="160" height="116" rx="8" fill="#1A1E22" stroke="#394148" stroke-width="1.5"/>
    <rect x="{x + 9}" y="{y + 9}" width="142" height="98" rx="6" fill="#20252A" stroke="#2E363D" stroke-width="1"/>
    <ellipse cx="{site_x}" cy="{site_y + 8}" rx="18" ry="7" fill="none" stroke="{NV_GREEN}" stroke-width="3" opacity="0.9"/>
    {mini_surface(x + 31, y + 78)}
    {molecule_icon(molecule, site_x, mol_y, tilt)}
  </g>"""


def powerline(x1: float, y1: float, x2: float, y2: float) -> str:
    mx = (x1 + x2) / 2
    return f"""  <path d="M{x1:.1f} {y1:.1f} C{mx:.1f} {y1:.1f}, {mx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" fill="none" stroke="{NV_GREEN}" stroke-width="7" stroke-linecap="round" opacity="0.18" filter="url(#green-glow)"/>
  <path d="M{x1:.1f} {y1:.1f} C{mx:.1f} {y1:.1f}, {mx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" fill="none" stroke="#8DFF4E" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="10 12" opacity="0.88"/>"""


def gpu_chip(x: int, y: int, w: int, h: int) -> str:
    pins = []
    for i in range(8):
        px = x + 20 + i * 22
        pins.append(f'<rect x="{px}" y="{y - 10}" width="8" height="14" rx="2" fill="#56616A"/>')
        pins.append(f'<rect x="{px}" y="{y + h - 4}" width="8" height="14" rx="2" fill="#56616A"/>')
    for i in range(5):
        py = y + 22 + i * 22
        pins.append(f'<rect x="{x - 10}" y="{py}" width="14" height="8" rx="2" fill="#56616A"/>')
        pins.append(f'<rect x="{x + w - 4}" y="{py}" width="14" height="8" rx="2" fill="#56616A"/>')
    pin_markup = "\n    ".join(pins)
    return f"""  <g filter="url(#soft-shadow)">
    {pin_markup}
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#1E242A" stroke="{NV_GREEN}" stroke-width="2.5"/>
    <rect x="{x + 28}" y="{y + 24}" width="{w - 56}" height="{h - 48}" rx="9" fill="#121619" stroke="#4D5B63" stroke-width="1.4"/>
    <rect x="{x + 58}" y="{y + 48}" width="{w - 116}" height="{h - 96}" rx="7" fill="#263038" stroke="#76B900" stroke-width="2"/>
    <path d="M{x + 77} {y + 64} H{x + w - 77} M{x + 77} {y + 84} H{x + w - 77} M{x + w / 2} {y + 55} V{y + h - 55}" stroke="#76B900" stroke-width="2" opacity="0.65"/>
    <circle cx="{x + w / 2:.1f}" cy="{y + h / 2:.1f}" r="8" fill="{NV_GREEN}" opacity="0.9"/>
  </g>"""


def build_workflow() -> None:
    center_x = 800
    center_y = 280
    panels = [
        (180, 58, "co", -22, -8),
        (440, 44, "h2o", 0, 6),
        (1000, 44, "methanol", 18, -7),
        (1260, 58, "co", 21, 9),
        (70, 222, "methanol", -19, 8),
        (1370, 222, "h2o", 18, -8),
        (250, 390, "h2o", -20, 5),
        (720, 406, "co", 0, 0),
        (1190, 390, "methanol", 21, -6),
    ]
    lines = "\n".join(powerline(center_x, center_y, x + 80, y + 58) for x, y, *_ in panels)
    panel_markup = "\n".join(batching_panel(*panel) for panel in panels)
    body = f"""  <defs>
    <filter id="green-glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#000000" flood-opacity="0.35"/>
    </filter>
  </defs>
  <rect width="1600" height="560" fill="{CHARCOAL}"/>
  <rect x="52" y="34" width="1496" height="492" rx="18" fill="#15191D" stroke="#273038" stroke-width="1.2"/>
  <circle cx="{center_x}" cy="{center_y}" r="126" fill="none" stroke="{NV_GREEN}" stroke-width="1.6" stroke-dasharray="3 13" opacity="0.45"/>
  <circle cx="{center_x}" cy="{center_y}" r="171" fill="none" stroke="#3A4740" stroke-width="1.2" stroke-dasharray="2 15" opacity="0.7"/>
{lines}
{panel_markup}
{gpu_chip(690, 198, 220, 164)}"""
    write(ASSETS / "workflow_adsorbml_bgr.svg", svg(1600, 560, body))


def build_phenomenon() -> None:
    body = f"""  <rect width="1100" height="430" fill="{CHARCOAL}"/>
  <text x="54" y="70" class="title">Why One Starting Geometry Can Fail</text>
  <text x="54" y="103" class="subtitle">different initial sites may relax into different local minima</text>
  <path d="M110 330 C250 130, 330 325, 455 180 C590 24, 670 330, 800 150 C890 55, 985 205, 1030 125" fill="none" stroke="{MUTED}" stroke-width="5"/>
  <path d="M110 330 C250 130, 330 325, 455 180 C590 24, 670 330, 800 150 C890 55, 985 205, 1030 125" fill="none" stroke="{NV_GREEN}" stroke-width="2"/>
  <circle cx="455" cy="180" r="13" fill="{AMBER}"/>
  <circle cx="800" cy="150" r="15" fill="{NV_GREEN}"/>
  <circle cx="260" cy="182" r="10" fill="{RED}"/>
  <path d="M260 182 C315 215, 386 208, 455 180" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="7 7"/>
  <path d="M610 86 C672 115, 735 126, 800 150" fill="none" stroke="{NV_GREEN}" stroke-width="3" stroke-dasharray="7 7"/>
  <text x="225" y="155" class="small">single nominated start</text>
  <text x="392" y="215" class="small">local minimum</text>
  <text x="748" y="184" class="small">batch-search minimum</text>
  <text x="98" y="372" class="small">configuration coordinate</text>
  <text x="40" y="226" class="small" transform="rotate(-90 40 226)">relative energy</text>"""
    write(ASSETS / "phenomenon_local_minima.svg", svg(1100, 430, body))


def main() -> None:
    build_icons()
    build_banner()
    build_workflow()
    build_phenomenon()
    print(f"Wrote visual assets under {ASSETS}")


if __name__ == "__main__":
    main()
