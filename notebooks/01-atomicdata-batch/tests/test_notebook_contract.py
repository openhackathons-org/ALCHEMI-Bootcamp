from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "atomicdata-and-batch.ipynb"


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_schema_and_complete_namespace_parse() -> None:
    notebook = read_notebook()
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source)
        for cell in code_cells(notebook)
    )

    ast.parse(transformed)
    assert code_cells(notebook)


def test_cell_ids_are_unique_and_sources_are_nonempty() -> None:
    notebook = read_notebook()
    cell_ids = [cell.id for cell in notebook.cells]

    assert len(cell_ids) == len(set(cell_ids))
    assert all(cell.source.strip() for cell in notebook.cells)


def test_code_cells_stay_small() -> None:
    notebook = read_notebook()
    lengths = [len(cell.source.splitlines()) for cell in code_cells(notebook)]

    assert max(lengths) <= 20
    assert sum(length <= 5 for length in lengths) > len(lengths) / 2


def test_each_code_cell_parses_after_ipython_transforms() -> None:
    notebook = read_notebook()

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = TransformerManager().transform_cell(cell.source)
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"code cell {index} does not parse") from error


def test_local_markdown_assets_exist() -> None:
    notebook = read_notebook()
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    references = re.findall(r'(?:src|data)=["\']([^"\']+)["\']', markdown)

    missing: list[str] = []
    for reference in references:
        if reference.startswith(("http://", "https://", "data:")):
            continue
        path = (NOTEBOOK_PATH.parent / reference.split("#", 1)[0]).resolve()
        if not path.is_file():
            missing.append(reference)

    assert not missing, f"missing local notebook assets: {missing}"


def test_markdown_code_fences_are_balanced() -> None:
    notebook = read_notebook()

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            assert cell.source.count("```") % 2 == 0, (
                f"markdown cell {index} has an unclosed code fence"
            )
