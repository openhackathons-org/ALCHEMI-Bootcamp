"""CPU-only checks for the rank layout selected by SpatialPartitioner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "scripts" / "part1_domain_run.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_runner_accepts_layout_from_rectangular_cell() -> None:
    assert RUNNER.validate_spatial_layout(
        (2, 2, 4),
        (1, 1, 4),
        world_size=4,
    ) == ((2, 2, 4), (1, 1, 4))


@pytest.mark.parametrize(
    ("cells_per_dim", "rank_grid", "world_size", "message"),
    (
        ((2, 2), (1, 1, 2), 2, "three positive integers"),
        ((2, 2, 2), (1, 1, 1), 2, "does not match 2 ranks"),
        ((2, 2, 2), (1, 1, 4), 4, "exceeds cells_per_dim"),
        ((3, 2, 4), (2, 2, 1), 4, "does not divide cells_per_dim"),
        ((2, 2, 2), (1, 0, 2), 2, "three positive integers"),
    ),
)
def test_runner_rejects_invalid_spatial_layout(
    cells_per_dim: tuple[int, ...],
    rank_grid: tuple[int, ...],
    world_size: int,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        RUNNER.validate_spatial_layout(
            cells_per_dim,
            rank_grid,
            world_size=world_size,
        )
