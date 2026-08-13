"""Lightweight HTML checks before the manual rendered review."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "training-finetuning.ipynb"


def test_static_html_keeps_assets_labels_and_width_bounds() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)

    assert 'alt="NVIDIA ALCHEMI: AI for Chemistry and Materials Science"' in html
    assert "curriculum-map-07.svg" in html
    assert "<object" in html
    assert "max-width:100%" in html.replace(" ", "")
    assert "training mse" in html.lower()
    assert "validation mse" in html.lower()
    assert "Energy RMSE (eV)" in html
    assert "Force RMSE (eV/Å)" in html
    assert "Epsilon (eV)" in html
    assert "Sigma (Å)" in html
    assert "output_error" not in html
    assert "Traceback (most recent call last)" not in html
