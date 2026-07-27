"""Shared methodology settings for the Part 1 isotope and IR workflow.

These values are methodology choices, not helper defaults.  The notebook shows
their resolved values, passes them explicitly to each relevant API, and records
them in the run manifest.  The external validator imports the same check values
so it cannot silently apply a different acceptance rule.

Names remain separate when two checks currently use the same number: charge
neutrality, dipole origin invariance, and capture-time charge neutrality test
different quantities and may need to change independently.
"""

from __future__ import annotations


# H/D structures are constructed with identical coordinates.  These limits
# catch accidental changes from conversion or model evaluation without claiming
# physical isotope effects on a Born-Oppenheimer potential-energy surface.
MASS_ONLY_POSITION_RTOL = 1.0e-7
MASS_ONLY_POSITION_ATOL_A = 1.0e-7
MASS_ONLY_ENERGY_TOLERANCE_EV = 1.0e-5
MASS_ONLY_FORCE_TOLERANCE_EV_A = 2.0e-5
MASS_ONLY_CHARGE_TOLERANCE_E = 2.0e-6

# The fused NVT -> NVE workflow assigns these integer status values.  Recording
# begins only after a system enters the production status.
IR_WARMUP_STATUS = 0
IR_PRODUCTION_STATUS = 1
IR_CAPTURE_CHARGE_TOLERANCE_E = 5.0e-5

# Main relaxation and dynamics settings.  FIRE2's ``dt`` is its initial
# adaptive optimizer step, not an MD time in femtoseconds.  Langevin friction
# is expressed in inverse femtoseconds.  Paired velocity seeds deliberately
# give each H/D pair identical random draws before mass scaling; the thermostat
# uses a separate seed so those two sources of randomness remain independent.
IR_FIRE_INITIAL_DT = 0.01
IR_NVT_FRICTION_PER_FS = 0.01
IR_INITIAL_VELOCITY_RANDOM_SEEDS = (101, 101, 202, 202)
IR_NVT_RANDOM_SEED = 303

# Welch settings for the predicted-charge IR proxy.  A 5 ps segment gives a
# roughly 6.7 cm^-1 frequency grid.  Fifty-percent overlap provides multiple
# shifted windows without treating nearly identical windows as independent.
IR_WELCH_SEGMENT_TIME_FS = 5_000.0
IR_WELCH_OVERLAP_FRACTION = 0.5

# These two tests are deliberately separate even though their present limits
# are equal: one checks net predicted charge, the other translation invariance
# of the dipole assembled from those charges.
IR_CHARGE_NEUTRALITY_TOLERANCE_E = 5.0e-5
IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM = 5.0e-5

# Post-run reporting choices.  Topology cutoffs follow the simple geometric
# definitions stated in the notebook; the energy limit is an advisory rather
# than a conserved-energy acceptance criterion for float32 dynamics.
PAIR_TEMPERATURE_RELATIVE_TOLERANCE = 0.20
ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM = 1.0
OXYGEN_CONNECTIVITY_CUTOFF_A = 4.0
COVALENT_OH_CUTOFF_A = 1.25
HBOND_H_ACCEPTOR_CUTOFF_A = 2.5
HBOND_OO_CUTOFF_A = 3.5
HBOND_ANGLE_CUTOFF_DEG = 140.0


__all__ = [
    "COVALENT_OH_CUTOFF_A",
    "ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM",
    "HBOND_ANGLE_CUTOFF_DEG",
    "HBOND_H_ACCEPTOR_CUTOFF_A",
    "HBOND_OO_CUTOFF_A",
    "IR_CAPTURE_CHARGE_TOLERANCE_E",
    "IR_CHARGE_NEUTRALITY_TOLERANCE_E",
    "IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM",
    "IR_FIRE_INITIAL_DT",
    "IR_INITIAL_VELOCITY_RANDOM_SEEDS",
    "IR_NVT_FRICTION_PER_FS",
    "IR_NVT_RANDOM_SEED",
    "IR_PRODUCTION_STATUS",
    "IR_WELCH_OVERLAP_FRACTION",
    "IR_WELCH_SEGMENT_TIME_FS",
    "IR_WARMUP_STATUS",
    "MASS_ONLY_CHARGE_TOLERANCE_E",
    "MASS_ONLY_ENERGY_TOLERANCE_EV",
    "MASS_ONLY_FORCE_TOLERANCE_EV_A",
    "MASS_ONLY_POSITION_ATOL_A",
    "MASS_ONLY_POSITION_RTOL",
    "OXYGEN_CONNECTIVITY_CUTOFF_A",
    "PAIR_TEMPERATURE_RELATIVE_TOLERANCE",
]
