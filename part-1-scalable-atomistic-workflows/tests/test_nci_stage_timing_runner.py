"""Focused checks for the canonical Stage 3 timing runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "measure_part1_nci_stage.py"
SBATCH_PATH = ROOT / "scripts" / "slurm_part1_nci_stage3.sbatch"
NOTEBOOK_PATH = (
    ROOT
    / "part-1-scalable-atomistic-workflows"
    / "alchemi-water-ir.ipynb"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("measure_part1_nci_stage", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_notebook_has_exact_eight_cell_stage(runner) -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    start, end = runner._validate_stage_layout(notebook)

    measured_ids = tuple(
        cell["id"]
        for cell in notebook.cells[start + 1 : end]
        if cell.cell_type == "code"
    )
    assert measured_ids == runner.STAGE_CELL_IDS


def test_stage_layout_rejects_an_extra_code_cell(runner) -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    _, end = runner._validate_stage_layout(notebook)
    notebook.cells.insert(end, nbformat.v4.new_code_cell("pass", id="extra-nci-code"))

    with pytest.raises(ValueError, match="Stage 3 code-cell order changed"):
        runner._validate_stage_layout(notebook)


def test_temporary_cell_does_not_replace_canonical_cells(runner) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("kept", id="kept-cell")]
    )
    before = runner._source_identity(notebook)

    class Client:
        def execute_cell(self, cell, cell_index, execution_count, store_history):
            assert notebook.cells[cell_index] is cell
            assert store_history is False
            cell.outputs = []

    runner._execute_temporary_cell(
        Client(), notebook, "value = 1", execution_count=1
    )

    assert runner._source_identity(notebook) == before
    assert len(notebook.cells) == 1


def test_scientific_export_records_force_routes_and_each_residual(runner) -> None:
    source = runner._scientific_export_source()

    for term in (
        '"official_analytic_force_route"',
        '"toolkit_analytic_force_route"',
        '"official_total_energy_route"',
        '"official_analytic_force_eV_A"',
        '"official_finite_difference_force_eV_A"',
        '"toolkit_analytic_force_eV_A"',
        '"official_analytic_vs_official_finite_difference_abs_error_eV_A"',
        '"toolkit_analytic_vs_official_analytic_abs_error_eV_A"',
        '"official_analytic_vs_official_total_energy_finite_difference"',
        '"toolkit_analytic_vs_official_analytic"',
    ):
        assert term in source

    assert '"central_finite_difference"' not in source
    assert '"autograd_force_eV_A"' not in source


def test_slurm_readback_uses_the_exported_force_check_names() -> None:
    source = SBATCH_PATH.read_text(encoding="utf-8")

    for term in (
        '"official_analytic_vs_official_total_energy_finite_difference"',
        '"toolkit_analytic_vs_official_analytic"',
        '"official_analytic_vs_official_finite_difference_abs_error_eV_A"',
        '"toolkit_analytic_vs_official_analytic_abs_error_eV_A"',
    ):
        assert term in source

    assert '"official_analytic_vs_official_energy_finite_difference"' not in source
    assert '"toolkit_pipeline_analytic_vs_official_analytic"' not in source
