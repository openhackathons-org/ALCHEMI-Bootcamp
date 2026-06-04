from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "sei-pareto-challenge.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_has_no_committed_outputs():
    nb = _notebook()
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None, f"cell {index} has execution_count"
            assert cell.get("outputs", []) == [], f"cell {index} has committed outputs"


def test_notebook_contains_todo_blocks_and_submission_write():
    nb = _notebook()
    sources = ["".join(cell.get("source", [])) for cell in nb["cells"]]

    assert sum("TODO" in source for source in sources) >= 5
    assert any("RAW_COMPONENT_ENERGIES_PATH" in source for source in sources)
    assert any("SUBMISSION_PATH" in source and "to_csv" in source for source in sources)


def test_notebook_uses_part1_toolkit_backend():
    nb = _notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "PART1_ROOT" in source
    assert "ToolkitRelaxationConfig" in source
    assert "get_toolkit_relaxation_engine" in source
    assert "ase_to_atomic_data" in source
