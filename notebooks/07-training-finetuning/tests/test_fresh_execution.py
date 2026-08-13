"""Opt-in fresh-kernel execution for the complete notebook."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbconvert import HTMLExporter

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = NOTEBOOK_DIR / "training-finetuning.ipynb"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_N07_NOTEBOOK") != "1",
    reason="set RUN_N07_NOTEBOOK=1 for the full fresh-kernel check",
)


def test_notebook_executes_from_a_fresh_kernel(tmp_path: Path) -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )

    executed = client.execute()
    nbformat.validate(executed)
    nbformat.write(executed, tmp_path / "training-finetuning.executed.ipynb")

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.outputs
        if output.output_type == "error"
    ]

    html, _ = HTMLExporter(template_name="lab").from_notebook_node(executed)
    for alt_text in (
        "Training MSE and validation MSE across four readout-only optimizer updates.",
        "Pair distances in generated Ar4 structures, colored by train and validation split.",
        "Generated argon training and validation loss, energy and force RMSE, and fitted epsilon and sigma traces.",
    ):
        assert f'alt="{alt_text}"' in html
    assert "No description has been provided for this image" not in html
