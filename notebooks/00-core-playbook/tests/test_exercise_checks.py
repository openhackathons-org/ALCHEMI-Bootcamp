import json
from pathlib import Path

import torch
from helpers import core as helpers
from nvalchemi.data import AtomicData, Batch

CORE_DIR = Path(__file__).resolve().parents[1]


class MACEWrapper:
    """Minimal exercise-check stand-in with the expected public class name."""


def test_exercise_checks_accept_completed_answers() -> None:
    amide = AtomicData(
        atomic_numbers=torch.tensor([7, 1, 1]),
        positions=torch.zeros((3, 3)),
    )
    amide.add_system_property("charge", torch.tensor([[-1.0]]))
    assert helpers.check_amide_anion(amide)

    batch = Batch.from_data_list([amide])
    wrapper = MACEWrapper()
    outputs = {
        "energy": torch.zeros((1, 1)),
        "forces": torch.zeros((3, 3)),
    }
    assert helpers.check_model_wrapper_exercise(wrapper, batch, outputs)

    force_drop = {"Ammonia": 0.1, "Propyne": 0.2, "Phenol": 0.3}
    assert helpers.check_force_drop_exercise(
        force_drop,
        "Phenol",
        ("Ammonia", "Propyne", "Phenol"),
    )

def test_exercise_checks_handle_pending_answers() -> None:
    assert not helpers.check_amide_anion(None)
    assert not helpers.check_model_wrapper_exercise(None, None, None)
    assert not helpers.check_force_drop_exercise(None, None, ("Ammonia",))
def test_notebook_checks_delegate_to_helpers() -> None:
    notebooks = [
        CORE_DIR / f"alchemi-core-0{index}-{slug}.ipynb"
        for index, slug in (
            (1, "data-and-batching"),
            (2, "models-and-simulation"),
            (3, "adapt-and-scale"),
        )
    ]
    sources = []
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        sources.extend("".join(cell["source"]) for cell in payload["cells"])

    combined = "\n".join(sources)
    assert "globals()" not in combined
    assert "helpers.check_amide_anion" in combined
    assert "helpers.check_model_wrapper_exercise" in combined
    assert "helpers.check_force_drop_exercise" in combined
