from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers.figures import figure_html
from helpers.viewers import show_molecule


def test_core_asset_manifest_matches_shipped_files() -> None:
    manifest_path = _NOTEBOOK_DIR / "assets" / "core-assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for record in manifest["assets"].values():
        asset = manifest_path.parent / record["filename"]
        assert asset.is_file(), asset
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == record["sha256"]


def test_molecule_viewer_has_static_preview_message_and_measurements() -> None:
    html = show_molecule(molecule("NH3")).data

    assert "height:320px" in html
    assert "3Dmol.js failed to load" not in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert "data-clear-measurement" in html
    assert "Math.hypot" in html
    assert "Math.acos" in html
    assert "viewer.addLine" in html


def test_scientific_figures_use_shared_width_and_left_alignment() -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    html = figure_html(figure, "Line from zero to one.")
    plt.close(figure)
    assert "width:100%;max-width:920px;" in html
    assert "margin:0;" in html
