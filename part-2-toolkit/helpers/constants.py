"""Physical constants, unit conversions, and pipeline structural enums.

Tutorial knobs (DT, FRICTION, TEMPS, RUN_NAME, LOG_DIR, ...) stay in the
notebook. This module holds only values that are universal to any ALCHEMI
MD run using these helpers.
"""

AMU_OVER_A3_TO_G_CM3 = 1.66054  # 1 amu/A^3 in g/cm^3
P_1ATM = 101325.0 / 1.602176634e11  # 1 atm in eV/A^3

MAX_FORCE_CLAMP = 50.0  # eV/A; default cap used by make_safety_hooks


# Fused warmup pipeline: FIRE -> NVT -> NPT. Status codes match the graph
# status encoding emitted by SnapshotHook / LoggingHook. Stage names are
# parameterised by the warmup target temperature (default 200 K) so that
# warmup runs at different solid-phase temperatures sit side-by-side on
# disk via `nvt_<T>k` / `npt_<T>k` artefact stems (see `T_WARMUP_TAG` in
# the warmup driver, parallel to `DT_TAG`).
def warmup_stage_names(t_warmup: float = 200.0) -> tuple[str, str, str]:
    tag = f"{int(t_warmup)}k"
    return ("fire", f"nvt_{tag}", f"npt_{tag}")


def status_by_stage(t_warmup: float = 200.0) -> dict[str, int]:
    tag = f"{int(t_warmup)}k"
    return {"fire": 0, f"nvt_{tag}": 1, f"npt_{tag}": 2}


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
