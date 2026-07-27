"""Physical and scientific constants used across the helpers package."""

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
KE_CONV = 103.64269667160806  # amu*A^2/fs^2 -> eV
BOLTZ_EV_K = 8.617333262145179e-05  # eV/K
P_CONV = 1.602176634e6  # eV/A^3 -> Bar
AMU_TO_G = 1.66054e-24  # atomic mass unit -> grams
ANGSTROM3_TO_CM3 = 1e-24  # A^3 -> cm^3
KCAL_MOL_TO_EV = 0.04336411530  # kcal/mol -> eV
KJ_MOL_TO_EV = 0.01036427  # kJ/mol -> eV (exact to 7 sig figs)
EV_TO_KJ_MOL = 1.0 / KJ_MOL_TO_EV  # eV -> kJ/mol

# ---------------------------------------------------------------------------
# Notebook-specific
# ---------------------------------------------------------------------------
RUN_SCOPES = {"short": "short check", "full": "full grid"}
RESULT_SOURCES = {"compute": "recomputes results", "saved": "reads saved results"}
