from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import core as helpers


def test_molecule_viewer_has_static_preview_message_and_measurements() -> None:
    html = helpers.show_molecule(molecule("NH3")).data

    assert "height:320px" in html
    assert "3Dmol.js failed to load" not in html
    assert "GitHub shows this notebook as a static preview" in html
    assert "connect to a runtime" in html
    assert "data-clear-measurement" in html
    assert "Math.hypot" in html
    assert "Math.acos" in html
    assert "viewer.addLine" in html


def test_scientific_figures_use_shared_width_and_left_alignment() -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    html = helpers._figure_html(figure, "Line from zero to one.")
    plt.close(figure)
    assert "width:100%;max-width:920px;" in html
    assert "margin:0;" in html
