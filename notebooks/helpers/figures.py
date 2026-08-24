# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared figure sizing, styling, and notebook display for Core helpers."""

from __future__ import annotations

import json
from base64 import b64encode
from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt

from .data_records import PROJECT_ROOT

COLORS = json.loads(
    (PROJECT_ROOT / "shared" / "alchemi-palette.json").read_text(encoding="utf-8")
)
SURFACE = COLORS["surface"]
BORDER = COLORS["border"]
GRID = COLORS["grid"]
TEXT = COLORS["text"]
MUTED = COLORS["muted"]
QUIET = COLORS["quiet"]
NVIDIA_GREEN = COLORS["nvidia_green"]
NVIDIA_BLUE = COLORS["nvidia_blue"]
NVIDIA_ORANGE = COLORS["nvidia_orange"]
TUTORIAL_VISUAL_MAX_WIDTH = 920
TUTORIAL_SMALL_VISUAL_MAX_WIDTH = 640
TUTORIAL_PIXELS_PER_INCH = 100
TUTORIAL_FIGURE_WIDTH = TUTORIAL_VISUAL_MAX_WIDTH / TUTORIAL_PIXELS_PER_INCH
TUTORIAL_SMALL_FIGURE_WIDTH = TUTORIAL_SMALL_VISUAL_MAX_WIDTH / TUTORIAL_PIXELS_PER_INCH
TUTORIAL_BODY_FONT_SIZE = 8.0
TUTORIAL_MICRO_FONT_SIZE = 7.2


def polish_axis(axis: Any, *, grid: str | None = "y") -> None:
    """Apply the restrained Core playbook scientific-figure treatment."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is None:
        axis.grid(False)
    else:
        axis.grid(axis=grid, color=GRID, alpha=0.58, linewidth=0.75)
    axis.tick_params(length=3.5, width=0.8)
    axis.title.set_fontweight("bold")


def figure_html(
    figure: Any,
    alt: str,
    *,
    max_width: int = TUTORIAL_VISUAL_MAX_WIDTH,
    align: str = "left",
    layout_rect: tuple[float, float, float, float] | None = None,
) -> str:
    """Render one Matplotlib figure as a self-contained notebook image."""

    if align not in {"left", "center"}:
        raise ValueError("align must be 'left' or 'center'")
    figure.tight_layout(pad=1.15, rect=layout_rect)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    payload = b64encode(buffer.getvalue()).decode("ascii")
    width_style = f"width:100%;max-width:{max_width}px;"
    margin_style = "margin:0;" if align == "left" else "margin:0 auto;"
    return (
        '<img src="data:image/png;base64,'
        f'{payload}" alt="{escape(alt, quote=True)}" '
        f'style="display:block;box-sizing:border-box;{width_style}'
        f'height:auto;{margin_style}">'
    )


def display_figure(
    figure: Any,
    alt: str,
    *,
    max_width: int = TUTORIAL_VISUAL_MAX_WIDTH,
    align: str = "left",
    layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Display one self-contained figure and release its Matplotlib state."""

    from IPython.display import HTML, display

    display(
        HTML(
            figure_html(
                figure,
                alt,
                max_width=max_width,
                align=align,
                layout_rect=layout_rect,
            )
        )
    )
    plt.close(figure)
