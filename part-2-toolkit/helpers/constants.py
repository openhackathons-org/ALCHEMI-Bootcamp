"""Physical constants, unit conversions, and pipeline structural enums.

Tutorial knobs (DT, FRICTION, TEMPS, RUN_NAME, LOG_DIR, ...) stay in the
notebook. This module holds only values that are universal to any ALCHEMI
MD run using these helpers.
"""

AMU_OVER_A3_TO_G_CM3 = 1.66054  # 1 amu/A^3 in g/cm^3
P_1ATM = 101325.0 / 1.602176634e11  # 1 atm in eV/A^3

MAX_FORCE_CLAMP = 50.0  # eV/A; default cap used by make_safety_hooks

# Fused warmup pipeline: FIRE -> NVT -> NPT. Status codes match the graph
# status encoding emitted by SnapshotHook / LoggingHook.
WARMUP_STAGE_NAMES = ("fire", "nvt_200k", "npt_200k")
STATUS_BY_STAGE = {"fire": 0, "nvt_200k": 1, "npt_200k": 2}

# Plot styling used by shade_stages(), keyed by status code.
STAGE_COLORS = {0: "#fff4e6", 1: "#e6f4ff", 2: "#e8f5e9"}
STAGE_LABELS = {0: "FIRE", 1: "NVT", 2: "NPT"}

# Keys in the serialised analysis dict that need np.asarray rehydration
# by io._restore_arrays.
_ARRAY_KEYS = (
    "log_steps",
    "log_energies",
    "log_density",
    "msd_steps",
    "msd_crystal",
    "msd_melt",
)
