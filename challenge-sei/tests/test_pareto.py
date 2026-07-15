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
    # windows: weak edge 1.0, ideal plateau 1.4-1.8, strong edge 3.0 eV (strength = -E_bind)
    assert seeding_score(-0.8) == 0.0                    # strength 0.8, below weak edge
    assert abs(seeding_score(-1.2) - 0.5) < 1e-12        # lower taper midpoint
    assert seeding_score(-1.6) == 1.0                    # inside ideal plateau
    assert abs(seeding_score(-2.4) - 0.5) < 1e-12        # upper taper midpoint
    assert seeding_score(-3.5) == 0.0                    # strength 3.5, above strong edge


def test_passivation_score_rewards_weak_binding():
    # windows: full reward for strength <= 0.6, tapering to zero by 1.3 eV
    assert binding_strength(0.2) == 0.0
    assert passivation_score(0.2) == 1.0                 # positive E_bind -> strength 0
    assert passivation_score(-0.6) == 1.0                # strength 0.6 at full-reward edge
    assert abs(passivation_score(-0.95) - 0.5) < 1e-12   # taper midpoint
    assert passivation_score(-1.3) == 0.0                # strength 1.3 at zero edge
