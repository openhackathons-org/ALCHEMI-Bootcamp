"""Fresh-kernel and rendered-output regression for Part 06."""

from __future__ import annotations

import fcntl
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = NOTEBOOK_DIR / "gpu-pipelines-profiling.ipynb"
GPU_LOCK = Path("/tmp/alchemi-v3-notebook.lock")


def test_notebook_fresh_executes_and_renders_cpu_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3")

    with GPU_LOCK.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            executed = client.execute(cwd=str(NOTEBOOK_DIR))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    nbformat.validate(executed)
    assert not any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )

    html, _ = HTMLExporter(template_name="lab").from_notebook_node(executed)
    assert "GPU measurement unavailable" in html
    assert 'alt="Synchronized first-call' in html
    assert 'alt="Per-update fused-stage' in html
    assert "Traceback (most recent call last)" not in html
