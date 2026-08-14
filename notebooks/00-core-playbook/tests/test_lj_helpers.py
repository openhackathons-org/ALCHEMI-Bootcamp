from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import core as helpers


def _small_experiment() -> tuple[float, object, object]:
    seed_batch = helpers.build_argon_batch(torch.device("cpu"), torch.float32)
    return helpers.prepare_lj_finetuning_experiment(
        seed_batch,
        sigma=3.40,
        cutoff=8.5,
        reference_epsilon=0.0104,
        baseline_epsilon=0.0060,
        initial_fit_samples=6,
        fine_tune_samples=10,
        test_samples=7,
        batch_size=4,
    )


def test_lj_experiment_builds_requested_disjoint_splits() -> None:
    initial_epsilon, loader, test_batch = _small_experiment()

    assert torch.isfinite(torch.tensor(initial_epsilon))
    assert len(loader.dataset) == 10
    assert [batch.num_graphs for batch in loader] == [4, 4, 2]
    assert test_batch.num_graphs == 7
    assert all("energy" in record.system_properties for record in loader.dataset)
    assert not torch.equal(loader.dataset[0].positions, test_batch[0].positions)


def test_lj_experiment_is_seeded() -> None:
    first_epsilon, first_loader, first_test = _small_experiment()
    second_epsilon, second_loader, second_test = _small_experiment()

    assert first_epsilon == pytest.approx(second_epsilon)
    assert torch.equal(first_loader.dataset[0].energy, second_loader.dataset[0].energy)
    assert torch.equal(first_test.positions, second_test.positions)


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_fit_samples": 0},
        {"fine_tune_samples": 0},
        {"test_samples": 0},
        {"batch_size": 0},
        {"initial_fit_noise": -0.1},
        {"fine_tune_noise": -0.1},
    ],
)
def test_lj_experiment_rejects_invalid_sizes_and_noise(
    overrides: dict[str, float],
) -> None:
    seed_batch = helpers.build_argon_batch(torch.device("cpu"), torch.float32)
    arguments = {
        "sigma": 3.40,
        "cutoff": 8.5,
        "reference_epsilon": 0.0104,
        "baseline_epsilon": 0.0060,
        **overrides,
    }

    with pytest.raises(ValueError):
        helpers.prepare_lj_finetuning_experiment(seed_batch, **arguments)
