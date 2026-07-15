"""Unit tests for the step-by-step relaxation engine (challenge_utils/relaxation_engine.py).

These mirror the notebook's engine unit-test cell: a distorted water molecule must relax
back to the textbook geometry, and relaxing structures in a batch must give the same
answer as relaxing them alone. They require the native ALCHEMI Toolkit (nvalchemi +
torch) and the MACE checkpoint, so they skip cleanly on model-free environments —
the rest of the test suite stays lightweight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT.parent / "part-1-batched-adsorption"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PART1))

pytest.importorskip("torch", reason="engine tests need torch")
pytest.importorskip("nvalchemi", reason="engine tests need the ALCHEMI Toolkit")
if not (PART1 / "helpers" / "__init__.py").exists():
    pytest.skip("part-1-batched-adsorption helpers not found", allow_module_level=True)

from ase.build import molecule as g2_molecule  # noqa: E402
from helpers import ase_to_atomic_data, atomic_data_to_ase  # noqa: E402

from challenge_utils.relaxation_engine import build_step_by_step_engine  # noqa: E402

CHECKPOINT = "medium-mpa-0"
FMAX = 0.05


@pytest.fixture(scope="module")
def engine():
    try:
        return build_step_by_step_engine(
            checkpoint=CHECKPOINT,
            device="auto",
            dtype="float32",
            dt=0.005,
            n_steps=5000,
            fmax=FMAX,
            maxstep=0.04,
        )
    except Exception as exc:  # e.g. checkpoint download unavailable offline
        pytest.skip(f"could not build engine ({type(exc).__name__}: {exc})")


def _distorted_water(seed: int = 7):
    water = g2_molecule("H2O")
    water.positions[1:] *= 1.25          # stretch both O-H bonds by 25%
    water.rattle(stdev=0.02, seed=seed)  # break the symmetry slightly
    water.set_cell([20.0, 20.0, 20.0])
    water.set_pbc(True)
    water.center()
    return water


def _geometry(result):
    relaxed = atomic_data_to_ase(result)
    d_oh = sorted(relaxed.get_distances(0, [1, 2]))
    angle = float(relaxed.get_angle(1, 0, 2))
    fmax = float(np.linalg.norm(np.asarray(result.forces).reshape(-1, 3), axis=1).max())
    return d_oh, angle, fmax


def test_engine_exposes_workflow_contract(engine):
    assert engine.name == "toolkit"
    assert callable(engine.relax)
    assert {"energy", "forces"} <= set(engine.model.model_config.active_outputs)


def test_water_relaxes_to_textbook_geometry(engine):
    payload = ase_to_atomic_data(_distorted_water(), structure_id="unit_test_water")
    reply = engine.relax([payload], label="pytest_water")
    result = reply.atoms[0]
    d_oh, angle, fmax = _geometry(result)

    assert result.converged, "engine failed to converge on water"
    assert fmax <= FMAX + 1e-8
    assert 0.90 <= d_oh[0] and d_oh[1] <= 1.05, f"O-H lengths {d_oh}"
    assert abs(d_oh[1] - d_oh[0]) < 0.02, f"asymmetric O-H bonds {d_oh}"
    assert 95.0 <= angle <= 115.0, f"H-O-H angle {angle}"


def test_batch_relaxation_matches_single(engine):
    # The same structure relaxed alone and inside a 2-structure batch must land in the
    # same minimum: batching must not couple independent systems.
    a = _distorted_water(seed=7)
    b = _distorted_water(seed=11)
    solo = engine.relax(
        [ase_to_atomic_data(a, structure_id="solo_a")], label="pytest_solo"
    ).atoms[0]
    pair = engine.relax(
        [ase_to_atomic_data(a, structure_id="pair_a"),
         ase_to_atomic_data(b, structure_id="pair_b")],
        label="pytest_pair",
    ).atoms

    assert all(r.converged for r in [solo, *pair])
    assert abs(pair[0].energy - solo.energy) < 2e-3, (
        f"batched energy differs from solo relax: {pair[0].energy} vs {solo.energy}"
    )
    # both waters end at the same minimum energy regardless of the starting rattle
    assert abs(pair[0].energy - pair[1].energy) < 5e-3
