from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge_utils.pareto import dominates, hypervolume_2d, pareto_flags
from challenge_utils.rewards import binding_strength, passivation_score, seeding_score


def test_dominates_uses_two_objective_maximization():
    assert dominates((0.8, 0.7), (0.6, 0.7))
    assert dominates((0.8, 0.7), (0.8, 0.5))
    assert not dominates((0.8, 0.7), (0.8, 0.7))
    assert not dominates((0.8, 0.4), (0.6, 0.7))


def test_pareto_flags_mark_nondominated_points():
    points = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2), (0.4, 0.4)]

    assert pareto_flags(points) == [True, True, True, False]


def test_hypervolume_2d_uses_reference_origin():
    points = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

    assert abs(hypervolume_2d(points) - 0.37) < 1e-12


def test_seeding_score_rewards_moderate_li_binding():
    assert seeding_score(-0.4) == 0.0
    assert abs(seeding_score(-0.65) - 0.5) < 1e-12
    assert seeding_score(-1.0) == 1.0
    assert abs(seeding_score(-1.75) - 0.5) < 1e-12
    assert seeding_score(-2.2) == 0.0


def test_passivation_score_rewards_weak_binding():
    assert binding_strength(0.2) == 0.0
    assert passivation_score(0.2) == 1.0
    assert passivation_score(-0.3) == 1.0
    assert abs(passivation_score(-0.55) - 0.5) < 1e-12
    assert passivation_score(-0.8) == 0.0
