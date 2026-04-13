"""Physical and scientific constants used across the helpers package."""

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
KE_CONV = 103.64269667160806  # amu*A^2/fs^2 -> eV
BOLTZ_EV_K = 8.617333262145179e-05  # eV/K
P_CONV = 1.602176634e6  # eV/A^3 -> Bar
AMU_TO_G = 1.66054e-24  # atomic mass unit -> grams
ANGSTROM3_TO_CM3 = 1e-24  # A^3 -> cm^3
KCAL_MOL_TO_EV = 0.0434  # kcal/mol -> eV

# ---------------------------------------------------------------------------
# OER thermodynamics
# ---------------------------------------------------------------------------
EV_PER_OER_STEP = 1.23  # Ideal OER step free energy (eV) = 4.92 / 4
# Ref: Rossmeisl et al., J. Electroanal. Chem. 607, 83-89 (2007).
