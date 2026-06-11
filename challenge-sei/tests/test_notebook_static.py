from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "sei-pareto-challenge.ipynb"
SOLUTION_NOTEBOOK = ROOT / "sei-pareto-challenge-solution.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _solution_notebook():
    return json.loads(SOLUTION_NOTEBOOK.read_text(encoding="utf-8"))


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
    assert any("custom_molecule_manifest.csv" in source for source in sources)
    assert any("from challenge_utils.pareto import" in source for source in sources)
    assert any("from challenge_utils.rewards import" in source for source in sources)
    assert not any("def hypervolume_2d" in source for source in sources)
    assert not any("def seeding_score" in source for source in sources)


def test_notebook_uses_part1_toolkit_backend():
    nb = _notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "PART1_ROOT" in source
    assert "ToolkitRelaxationConfig" in source
    assert "get_toolkit_relaxation_engine" in source
    assert "ase_to_atomic_data" in source


def test_solution_notebook_has_no_placeholders_or_outputs():
    nb = _solution_notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "TODO" not in source
    assert "NotImplementedError" not in source
    assert "challenge_submission.csv" in source
    assert "raw_component_energies.csv" in source
    assert "custom_molecule_manifest.csv" in source
    assert "from challenge_utils.pareto import" in source
    assert "from challenge_utils.rewards import" in source
    assert "def hypervolume_2d" not in source
    assert "def seeding_score" not in source
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None, f"cell {index} has execution_count"
            assert cell.get("outputs", []) == [], f"cell {index} has committed outputs"
