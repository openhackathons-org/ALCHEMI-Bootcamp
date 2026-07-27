"""Physical constants, unit conversions, and pipeline structural enums.

Tutorial knobs (DT, FRICTION, TEMPS, RUN_NAME, LOG_DIR, ...) stay in the
notebook. This module holds only values that are universal to any ALCHEMI
MD run using these helpers.
"""

AMU_OVER_A3_TO_G_CM3 = 1.66054  # 1 amu/A^3 in g/cm^3
P_1ATM = 101325.0 / 1.602176634e11  # 1 atm in eV/A^3

MAX_FORCE_CLAMP = 50.0  # eV/A; default cap used by make_safety_hooks

# DFT-D3(BJ) damping parameters per training functional. Drivers select at
# init time via D3_PRESETS[args.model]. Each entry's damping params match
# the functional the model was trained against, so explicit --d3 stays in
# the same parameterization as the model's implicit dispersion. D3 on top
# of a functional that already includes D3 (aimnet2_2025, mace) is
# correction-on-correction; --d3 is exposed as a general toggle regardless.
D3_PRESETS: dict[str, dict[str, float]] = {
    # wB97M-D3(BJ) -- legacy aimnet2 (isolated-mol training).
    # DOI 10.1021/acs.jctc.8b00842 (Najibi & Goerigk 2018).
    "aimnet2": {"a1": 0.5660, "a2": 3.1280, "s8": 0.3908},
    # B97-3c-D3(BJ) -- aimnet2_2025 training functional.
    # DOI 10.1063/1.5012601 (Brandenburg et al. 2018).
    "aimnet2_2025": {"a1": 0.37, "a2": 4.10, "s8": 1.50},
    # PBE-D3(BJ) -- MACE-MP-0 v0 training functional (PBE+D3).
    # Source: dftd3/simple-dftd3 parameters.toml [parameter.pbe].d3.bj.
    "mace": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
    # wB97M-D3(BJ) -- matches ORB-v3 conservative-omol (OMol25, trained on
    # wB97M-V/def2-TZVPD; orb-models/MODELS.md:7). Same damping numbers as
    # the legacy aimnet2 preset because both target wB97M-family functionals
    # (Najibi & Goerigk 2018, DOI 10.1021/acs.jctc.8b00842). Note this is
    # correction-on-correction: wB97M-V already includes nonlocal VV10
    # dispersion. If you're running the older OMat24 checkpoint
    # (`orb_v3_conservative_inf_omat`, PBE+D3-trained), restore the
    # PBE-D3(BJ) values used by `mace` here:
    #   {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875}
    "orb": {"a1": 0.5660, "a2": 3.1280, "s8": 0.3908},
}


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
