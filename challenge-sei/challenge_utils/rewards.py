"""Literature-motivated reward functions for the SEI Pareto challenge.

The challenge still uses simplified vacuum binding energies, but the reward
shape follows two common literature ideas:

* SEI-forming additives should react preferentially at the reducing anode and
  form a passivating interphase.
* Surface reactions often follow a Sabatier-style tradeoff: useful interaction
  is neither too weak to activate the molecule nor so strong that the surface is
  effectively poisoned.

All constants below are adsorption-strength magnitudes in eV per molecule,
where ``strength = max(0, -E_bind)``.
"""

from __future__ import annotations


# Weak-to-moderate adsorption window for Li-metal seeding.
SEEDING_WEAK_EDGE_EV = 0.50
SEEDING_IDEAL_LOW_EV = 0.80
SEEDING_IDEAL_HIGH_EV = 1.50
SEEDING_STRONG_EDGE_EV = 2.00

# Weak adsorption window for compatibility with an already-passivating SEI proxy.
PASSIVATION_FULL_REWARD_EV = 0.30
PASSIVATION_ZERO_REWARD_EV = 0.80


def _clip01(value: float) -> float:
    """Clip one scalar value into the closed interval ``[0, 1]``."""
    return min(1.0, max(0.0, float(value)))


def binding_strength(e_bind_eV: float) -> float:
    """Return the exothermic adsorption-strength magnitude for a binding energy."""
    return max(0.0, -float(e_bind_eV))


def seeding_score(e_bind_li_eV: float) -> float:
    """Score Li-metal SEI seeding from the Li-surface binding energy.

    Full reward is assigned for moderate chemisorption-like strengths
    (0.8--1.5 eV), with linear tapers to zero below 0.5 eV and above 2.0 eV.
    This implements a Sabatier-style objective: the additive should interact
    strongly enough with reactive Li metal to plausibly seed interphase
    formation, but extremely strong binding is penalized.
    """
    strength = binding_strength(e_bind_li_eV)
    if strength <= SEEDING_WEAK_EDGE_EV:
        return 0.0
    if strength < SEEDING_IDEAL_LOW_EV:
        return _clip01(
            (strength - SEEDING_WEAK_EDGE_EV)
            / (SEEDING_IDEAL_LOW_EV - SEEDING_WEAK_EDGE_EV)
        )
    if strength <= SEEDING_IDEAL_HIGH_EV:
        return 1.0
    if strength < SEEDING_STRONG_EDGE_EV:
        return _clip01(
            (SEEDING_STRONG_EDGE_EV - strength)
            / (SEEDING_STRONG_EDGE_EV - SEEDING_IDEAL_HIGH_EV)
        )
    return 0.0


def passivation_score(e_bind_passivating_eV: float) -> float:
    """Score compatibility with a passivating SEI proxy surface.

    Full reward is assigned for weak or endothermic adsorption
    (strength <= 0.3 eV). The score decreases linearly to zero by 0.8 eV,
    reflecting the challenge assumption that a good passivating layer should
    discourage continued strong molecule-surface interaction.
    """
    strength = binding_strength(e_bind_passivating_eV)
    if strength <= PASSIVATION_FULL_REWARD_EV:
        return 1.0
    if strength >= PASSIVATION_ZERO_REWARD_EV:
        return 0.0
    return _clip01(
        1.0
        - (strength - PASSIVATION_FULL_REWARD_EV)
        / (PASSIVATION_ZERO_REWARD_EV - PASSIVATION_FULL_REWARD_EV)
    )
