from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from ase.build import molecule

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))

from helpers import core as helpers
from nvalchemi.data import AtomicData, Batch


def _cells(name: str) -> list[dict[str, object]]:
    notebook = json.loads((CORE_DIR / name).read_text(encoding="utf-8"))
    return notebook["cells"]


def _source_containing(name: str, fragment: str) -> str:
    matches = [
        "".join(cell["source"])
        for cell in _cells(name)
        if fragment in "".join(cell["source"])
    ]
    assert len(matches) == 1
    return matches[0]


def _source_by_id(name: str, cell_id: str) -> str:
    matches = [
        "".join(cell["source"])
        for cell in _cells(name)
        if cell["id"] == cell_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_ase_builds_the_three_core_molecules() -> None:
    structures = [
        helpers.load_example_molecule(name)
        for name in ("Ammonia", "Propyne", "Phenol")
    ]

    assert [len(structure) for structure in structures] == [4, 7, 13]
    assert [structure.get_chemical_formula() for structure in structures] == [
        "H3N",
        "C3H4",
        "C6H6O",
    ]


def test_modules_one_and_two_construct_molecules_through_ase() -> None:
    module_one = "\n".join(
        "".join(cell["source"])
        for cell in _cells("alchemi-core-01-data-and-batching.ipynb")
    )
    module_two = "\n".join(
        "".join(cell["source"])
        for cell in _cells("alchemi-core-02-models-and-simulation.ipynb")
    )

    assert 'molecule("NH3")' in module_one
    assert 'helpers.load_example_molecule("Propyne")' in module_one
    assert 'helpers.load_example_molecule("Phenol")' in module_one
    assert "helpers.load_example_molecule(label)" in module_two

    combined = module_one + module_two
    assert "C3H4_C3v" not in combined
    assert "Phenol_dimer" not in combined
    assert "NCI Atlas subset" not in combined


def test_module_one_direct_tensors_match_ase_ammonia() -> None:
    name = "alchemi-core-01-data-and-batching.ipynb"
    namespace = {
        "AtomicData": AtomicData,
        "DTYPE": torch.float32,
        "device": torch.device("cpu"),
        "molecule": molecule,
        "torch": torch,
    }
    for cell_id in (
        "core-t10",
        "core-t16",
        "core-t19",
        "core-t20",
    ):
        exec(  # noqa: S102 - execute the exact code taught in the notebook
            _source_by_id(name, cell_id), namespace
        )

    assert namespace["ammonia_data"] == namespace["direct_ammonia_data"]
    torch.testing.assert_close(
        namespace["ammonia_data"].positions,
        namespace["direct_ammonia_data"].positions,
    )


def test_module_two_rebuilds_the_same_batch_on_cpu() -> None:
    name = "alchemi-core-02-models-and-simulation.ipynb"
    namespace = {
        "AtomicData": AtomicData,
        "Batch": Batch,
        "device": torch.device("cpu"),
        "helpers": helpers,
    }
    exec(  # noqa: S102 - execute the exact code taught in the notebook
        _source_containing(name, 'labels = ("Ammonia", "Propyne", "Phenol")'),
        namespace,
    )

    assert namespace["labels"] == ("Ammonia", "Propyne", "Phenol")
    assert namespace["example_batch"].num_nodes_per_graph.tolist() == [4, 7, 13]
    assert namespace["example_batch"].device.type == "cpu"


def test_module_two_triggers_nan_guard_from_bad_geometry() -> None:
    cells = _cells("alchemi-core-02-models-and-simulation.ipynb")
    sources = ["".join(cell["source"]) for cell in cells]

    guard_index = next(
        index for index, source in enumerate(sources) if "nan_guard = NaNDetectorHook" in source
    )
    example_index = next(
        index for index, source in enumerate(sources) if "### Trigger the NaN safety hook" in source
    )

    assert example_index == guard_index + 1
    assert "n_steps=10" in "\n".join(sources)
    assert "example_batch.index_select(0)" in "\n".join(sources)
    assert "bad_positions[1] = bad_positions[0]" in "\n".join(sources)
    assert "raises `RuntimeError` immediately after the model call" in "\n".join(sources)
    assert "InjectNaNHook" not in "\n".join(sources)


def test_module_two_tracks_progress_on_the_main_batch() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("alchemi-core-02-models-and-simulation.ipynb")
    )

    assert "REPORT_EVERY = 4" in source
    assert "force_progress_hook = ForceProgressHook(labels, frequency=REPORT_EVERY)" in source
    assert "force_progress_hook.reports" in source
    assert "a custom hook that follows Toolkit's hook interface" in source
    assert "Local storage owned by this hook instance" in source
    assert "Its `reports` attribute is a Python list containing one dictionary per report" in source
    assert "history_hook" not in source
    assert '"delta_fmax_ev_per_a": delta_fmax' in source
    assert "ConvergenceHook.from_fmax(FMAX_EV_PER_A)" in source
    assert "NEAR_RELAXED_FMAX" not in source
    assert "relaxed_batch = fire2.run(model_batch)" in source
    assert "result_writer.write(relaxed_batch)" in source
    assert 'relaxed_batch.to("cpu")' not in source
    assert "relaxation_input" not in source
    assert "final_batch" not in source


def test_module_two_keeps_wrapper_apis_visible() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("alchemi-core-02-models-and-simulation.ipynb")
    )

    assert 'name.endswith("Wrapper")' in source
    assert '"required inputs": model.input_data()' in source
    assert "sorted(model.input_data())" not in source
    assert "MACEWrapper.from_checkpoint" in source
    assert '"medium-0b2"' in source
    assert "MACE-MP-0b2 is trained for materials" in source
    assert "nvalchemi.models.mace.MACEWrapper.html" in source
    assert "FMAX_EV_PER_A" in source


def test_module_two_connects_the_workflow_stages() -> None:
    sources = [
        "".join(cell["source"])
        for cell in _cells("alchemi-core-02-models-and-simulation.ipynb")
    ]
    source = "\n".join(sources)

    assert "The relaxation below continues with `model`" in source
    assert "The safety hook protects every model call" in source
    assert "Snapshots preserve complete structures every four steps" in source

    recap_index = next(
        index for index, text in enumerate(sources) if "## Module 2 recap" in text
    )
    continue_index = next(
        index for index, text in enumerate(sources) if "## Continue to molecular dynamics" in text
    )
    assert recap_index < continue_index


def test_module_two_removes_demo_only_scaffolding() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("alchemi-core-02-models-and-simulation.ipynb")
    )

    assert "predict the returned shapes" not in source
    assert "MOLECULE_IDS" not in source
    assert "system_id" not in source
