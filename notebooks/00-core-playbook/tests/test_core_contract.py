"""Mechanical checks for the hand-authored ALCHEMI Core Playbook.

The playbook is the only source of truth for its own content. These tests
deliberately verify nothing about lesson sequence, API coverage, prose, diagram
meaning, or styling: those belong to the LLM review and the human rendered
review. What lives here is limited to breakage a machine can detect without an
opinion about the lesson -- schema validity, Python that parses, unique cell
ids, clean saved outputs, honest execution counts, local assets that resolve,
a code-cell line ceiling, and alt text on generated figures.

Adding a content assertion here re-creates the failure mode this file was cut
down to escape: a check that must be edited whenever the notebook changes
verifies nothing.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from ase.build import molecule
from IPython.core.inputtransformer2 import TransformerManager

CORE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = CORE_DIR / "alchemi-core-playbook.ipynb"

# Ceiling on the visible source length of a single code cell. Cells that
# legitimately need more room opt out with `core_allow_long` metadata.
MAX_CODE_CELL_LINES = 20


def read_notebook_as_authored(path: Path | None = None) -> nbformat.NotebookNode:
    """Read the notebook exactly as stored, without validating or repairing it.

    Every check except the schema check reads the notebook this way, for two
    reasons. First, `nbformat.validate` repairs duplicate cell ids in place and
    only warns, so anything inspecting ids has to see the file before validation
    touches it. Second, confining validation to one test means a schema defect
    produces one focused failure instead of masking every other check.

    The path is resolved on each call rather than bound as a default argument so
    that these checks can themselves be exercised against a mutated notebook.
    """

    return nbformat.read(path or NOTEBOOK_PATH, as_version=4)


def read_notebook(path: Path | None = None) -> nbformat.NotebookNode:
    notebook = read_notebook_as_authored(path)
    nbformat.validate(notebook)
    return notebook


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_matches_the_nbformat_schema() -> None:
    read_notebook()


def test_every_code_cell_parses_as_python() -> None:
    # Cells may use IPython-only syntax (`%magic`, `!shell`), so transform each
    # one the way IPython would before handing it to the Python parser.
    transformer = TransformerManager()
    for cell in code_cells(read_notebook_as_authored()):
        try:
            ast.parse(transformer.transform_cell(cell.source))
        except SyntaxError as error:
            pytest.fail(f"cell {cell.id} does not parse: {error}")


def test_cell_ids_are_unique_and_non_empty() -> None:
    """Duplicate ids break `#cell-id` links, diffs, and per-cell tooling.

    This reads the stored JSON rather than going through nbformat, because both
    `nbformat.read` and `nbformat.validate` rename duplicate ids in place and
    only emit a warning. Routed through either of them, this check could never
    fail.
    """

    stored = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    ids = [cell.get("id") for cell in stored["cells"]]

    blank = [index for index, value in enumerate(ids) if not value]
    assert not blank, f"cells at these positions have no id: {blank}"

    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    assert not duplicates, f"cell ids used more than once: {duplicates}"


def test_execution_counts_agree_with_saved_outputs() -> None:
    """A saved output must come from a real run, and an unrun cell must say so."""

    notebook = read_notebook_as_authored()
    outputs_without_count = [
        cell.id
        for cell in code_cells(notebook)
        if cell.get("outputs") and cell.get("execution_count") is None
    ]
    assert not outputs_without_count, outputs_without_count

    count_without_outputs = [
        (cell.id, cell.get("execution_count"))
        for cell in code_cells(notebook)
        if not cell.get("outputs") and cell.get("execution_count") is not None
    ]
    assert not count_without_outputs, count_without_outputs


def test_referenced_local_assets_resolve_on_disk() -> None:
    notebook = read_notebook_as_authored()
    source = "\n\n".join(cell.source for cell in notebook.cells)

    embedded = [
        src
        for src in re.findall(r'<img[^>]*\bsrc="([^"]+)"', source)
        if not src.startswith(("http://", "https://", "data:"))
    ]
    assert embedded, "the playbook embeds its figures with <img src=...>"
    missing = [src for src in embedded if not (CORE_DIR / src).resolve().is_file()]
    assert not missing, missing

    # The asset index records which rendered file each figure currently points
    # at. Its naming scheme is the renderer's business; that the files it names
    # are actually present is not.
    index_reference = notebook.metadata.get("core", {}).get("asset_index")
    assert index_reference, "notebook metadata must record core.asset_index"
    index_path = CORE_DIR / index_reference
    assert index_path.is_file(), index_path

    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = [record["filename"] for record in index.get("assets", {}).values()]
    assert indexed, "the asset index lists no assets"
    absent = [
        filename
        for filename in indexed
        if not (CORE_DIR / "assets" / filename).is_file()
    ]
    assert not absent, absent


def test_code_cells_stay_within_the_line_ceiling() -> None:
    visible = [
        cell
        for cell in code_cells(read_notebook_as_authored())
        if "hide-input" not in cell.metadata.get("tags", [])
        and not cell.metadata.get("jupyter", {}).get("source_hidden", False)
    ]
    assert visible

    oversized = {
        cell.id: len(cell.source.splitlines())
        for cell in visible
        if len(cell.source.splitlines()) > MAX_CODE_CELL_LINES
        and not cell.metadata.get("core_allow_long")
    }
    assert not oversized, oversized


def test_plot_cells_carry_alt_text() -> None:
    """Generated figures need a text equivalent; screen readers cannot infer one."""

    missing = [
        cell.id
        for cell in code_cells(read_notebook_as_authored())
        if "helpers.plot_" in cell.source and not cell.metadata.get("alt")
    ]
    assert not missing, missing


def test_infer_bonds_returns_unique_ordered_index_pairs(monkeypatch) -> None:
    """Explicit connectivity must survive the trip into the viewer unchanged.

    MatterViz silently ignores malformed connectivity, so a duplicated or
    reversed pair would show up only as a wrong picture in a rendered review.
    """

    pytest.importorskip("torch")
    sys.path.insert(0, str(CORE_DIR))
    from helpers import core as helpers

    atoms = molecule("C2H2")
    bonds = helpers.infer_bonds(atoms)

    assert bonds, "infer_bonds found no connectivity in C2H2"
    duplicates = sorted({pair for pair in bonds if list(bonds).count(pair) > 1})
    assert not duplicates, f"duplicate bond pairs: {duplicates}"
    unordered = [pair for pair in bonds if not pair[0] < pair[1]]
    assert not unordered, f"pairs not stored lowest index first: {unordered}"
    out_of_range = [
        pair for pair in bonds if not all(0 <= index < len(atoms) for index in pair)
    ]
    assert not out_of_range, f"pairs outside 0..{len(atoms) - 1}: {out_of_range}"

    captured: dict[str, object] = {}

    def fake_widget(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(
        sys.modules, "pymatviz", SimpleNamespace(StructureWidget=fake_widget)
    )
    helpers.show_molecule(atoms, bonds=bonds, height=360)

    structure = captured["structure"]
    assert isinstance(structure, dict)
    forwarded = [
        (entry["site_idx_1"], entry["site_idx_2"])
        for entry in structure["properties"]["bonds"]
    ]
    assert forwarded == [tuple(pair) for pair in bonds], (
        f"widget received {forwarded}, expected {list(bonds)}"
    )


def flattened_output_text(output: nbformat.NotebookNode) -> str:
    """Collect every textual payload of one saved output, list or string form."""

    chunks: list[str] = []
    candidates = [output.get("text", "")]
    candidates += [
        value
        for key, value in output.get("data", {}).items()
        if key.startswith("text/")
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            chunks.append(candidate)
        elif isinstance(candidate, list):
            chunks.extend(str(line) for line in candidate)
    return "".join(chunks)


def test_saved_outputs_carry_no_errors_or_tracebacks() -> None:
    notebook = read_notebook()
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error", cell.id
            assert "ename" not in output, cell.id
            text = flattened_output_text(output)
            assert "Traceback (most recent call last)" not in text, cell.id


def test_answer_blocks_quote_the_real_atom_counts() -> None:
    """Folded answers must agree with the molecules the notebook actually builds.

    This is a consistency check, not a content assertion: the truth comes from
    building `G2_NAMES` with ASE, and the notebook is only required to quote
    those numbers somewhere in its answers. Rewording the sentences is free.
    Changing the molecule set without updating the answers is not, which is the
    failure this catches.
    """

    notebook = read_notebook_as_authored()
    source = "\n".join(cell.source for cell in code_cells(notebook))
    match = re.search(r"^G2_NAMES\s*=\s*(\([^)]*\))", source, flags=re.MULTILINE)
    assert match, "no G2_NAMES tuple found in the notebook code"

    counts = [len(molecule(name)) for name in ast.literal_eval(match.group(1))]
    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    total = boundaries[-1]

    answers = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown" and "<summary>Check" in cell.source
    )
    assert answers, "no folded answer blocks found"

    for count in counts:
        assert re.search(rf"\b{count}\b", answers), (
            f"no answer block mentions the {count}-atom molecule"
        )
    # The boundary list is quoted as a literal, so compare it as one.
    expected = "[" + ", ".join(str(edge) for edge in boundaries) + "]"
    assert expected in answers, f"answer blocks do not quote boundaries {expected}"
    assert re.search(rf"\b{total}\b", answers), (
        f"answer blocks do not quote the {total} packed atom rows"
    )
