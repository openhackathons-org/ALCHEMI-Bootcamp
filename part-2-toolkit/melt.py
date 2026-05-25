"""Consolidated melt-phase driver for the SLC pipeline.

Clones the end-of-warmup solid (NPT or NVT, picked by `--source`),
reseeds Maxwell-Boltzmann velocities at T_MELT, and runs Langevin NVT
at T_MELT to generate the liquid half used for SLC construction.
Visualisation is deliberately absent so the run has no plotting
dependencies and can execute under `docker exec` in a detached session
on the compute node -- decoupled from the Jupyter kernel, safe against
SSH / websocket drops.

The driver is material- and model-agnostic (set via `--material` and
`--model`; both must match the upstream warmup invocation so the seed
checkpoint is found). Disambiguation between runs is purely directory-
level via `--run-name`; file stems inside that dir are independent of
material/model/corrections.

Variants at a glance (choose via flags)::

    # Bare AIMNet2 (wB97M-D3) -- defaults; NPT-end seed
    python melt.py

    # Larger 200-mol supercell (warmup --supercell 5,5,4 / --run-name *_big)
    python melt.py --run-name naphthalene_long_big

    # AIMNet2-2025, melt-in-box from un-expanded NVT crystal (avoids the
    # post-NPT plastic-crystal artefact at 200 K)
    python melt.py \\
        --run-name naphthalene_long_2025 --model aimnet2_2025 --source nvt

    # AIMNet2-2025 + DFT-D3(BJ) dispersion (B97-3c-D3(BJ) preset auto-
    # selected via D3_PRESETS[args.model]). Match the warmup invocation's
    # --d3 flag and --run-name so the seed checkpoint physics line up
    # with the melt physics.
    python melt.py \\
        --run-name naphthalene_long_d3_2025 --model aimnet2_2025 \\
        --source nvt --d3

    # MACE-MP-0 (medium-0b2). --ewald is rejected with MACE since the
    # wrapper has no charges output.
    python melt.py \\
        --run-name naphthalene_long_mace --model mace --source nvt

    # ORB-v3 conservative-inf-omat (OMAT24, PBE+D3-trained). Same charges-
    # free constraint as MACE: --ewald is rejected. --d3 is permitted with
    # PBE-D3(BJ) damping.
    python melt.py \\
        --run-name naphthalene_long_orb --model orb --source nvt

Two-phase usage:
  1. Smoke test:  python melt.py --melt-ps 1.5
  2. Full extend: python melt.py
The second invocation resumes from the 3000-step checkpoint and runs the
remaining delta. NVT Langevin is memoryless, so the checkpointed
velocities are a valid continuation -- no integrator state needs to
persist.

All stage / checkpoint / artefact names carry the DT_TAG suffix
(e.g. `dt0p5fs`) plus a T_WARMUP_TAG (e.g. `200k`, `100k`, matching the
warmup driver's `--t-warmup`) and a `from_{nvt,npt}` tag identifying
which warmup endpoint seeded the melt, so runs across
(source, t_warmup, DT) variants sit side by side without overwriting.

Note on `--source nvt` (melt-in-box): the system runs at fixed volume
inheriting the un-expanded warmup-NVT crystal (rho ~= 1.2 g/cm^3). The
end-of-run diagnostic `density` is correctly invariant; it does NOT
indicate an un-equilibrated melt -- expect ~1.2 throughout.

Every stage writes its checkpoint on exit; re-running is idempotent.

Precondition: warmup.py must have produced the
`nvt_{T_WARMUP_TAG}_{DT_TAG}` or `npt_{T_WARMUP_TAG}_{DT_TAG}` checkpoint
under logs/<run-name>/checkpoints/ (`--source` picks which; pass the
matching `--t-warmup` value).
"""

import argparse
from loguru import logger
import sys
import time
from pathlib import Path

import torch

from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.aimnet2 import AIMNet2Wrapper
from nvalchemi.models.dftd3 import DFTD3ModelWrapper
from nvalchemi.models.ewald import EwaldModelWrapper
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)

from helpers import (
    D3_PRESETS,
    DYNAMICS_SCALARS,
    OrbV3Wrapper,
    checkpoint_exists,
    compute_density,
    fresh_zarr_sink,
    load_checkpoint,
    load_stage_meta,
    make_safety_hooks,
    next_part_index,
    part_paths,
    save_checkpoint,
    save_stage_meta,
    stdout_writer,
)

torch.set_float32_matmul_precision("high")


# Constants that are stable across the current variant set. Promote to CLI
# flags when a future experiment first needs to vary one.
FRICTION = 0.01  # fs^-1 (= 10 ps^-1, correlation ~100 fs)
SNAPSHOT_EVERY = 100  # steps
LOG_EVERY = 100  # steps

# MACE foundation checkpoint resolved by `--model mace`. Matches the alias
# used in nvalchemi-toolkit/examples/advanced/04_mace_nvt.py.
MACE_CHECKPOINT = "medium-0b2"
# ORB foundation checkpoint resolved by `--model orb`. Conservative head,
# OMAT24-trained, PBE+D3; precision `float32-high` is the recommended A100/H100
# MD setting per the orb-models README. Defaults match helpers/orb.py.
ORB_ALIAS = "orb_v3_conservative_omol"
ORB_PRECISION = "float32-high"


def build_base_model(model_alias: str, device: str):
    """Instantiate the base MLIP wrapper for `--model {aimnet2, aimnet2_2025, mace, orb}`."""
    if model_alias in {"aimnet2", "aimnet2_2025"}:
        return AIMNet2Wrapper.from_checkpoint(
            model_alias, device=device, compile_model=True
        )
    if model_alias == "mace":
        return MACEWrapper.from_checkpoint(
            MACE_CHECKPOINT, device=device, compile_model=True
        )
    if model_alias == "orb":
        return OrbV3Wrapper.from_checkpoint(
            ORB_ALIAS, device=device, precision=ORB_PRECISION, compile_model=True
        )
    raise SystemExit(f"unknown --model {model_alias!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--material",
        default="naphthalene",
        help="Material name; must match the upstream warmup invocation's "
        "--material so the seed checkpoint is found. Default 'naphthalene'.",
    )
    p.add_argument(
        "--model",
        default="aimnet2",
        choices=["aimnet2", "aimnet2_2025", "mace", "orb"],
        help="MLIP checkpoint alias. 'aimnet2' (default) = wB97M-D3 AIMNet2 "
        "(isolated-molecule training, legacy). 'aimnet2_2025' = AIMNet2 "
        "B97-3c+D3, retrained on intermolecular data; recommended for "
        "condensed-phase / crystal packing. 'mace' = MACE-MP-0 foundation "
        f"model (alias {MACE_CHECKPOINT!r}). 'orb' = ORB-v3 conservative "
        f"foundation model (alias {ORB_ALIAS!r}, precision={ORB_PRECISION!r}; "
        "OMAT24, PBE+D3-trained). Neither MACE nor ORB expose a charges "
        "output, so `--ewald` is rejected with --model {mace, orb}.",
    )
    p.add_argument(
        "--ewald",
        action="store_true",
        help="Wrap the base model with Ewald long-range electrostatics "
        "pipeline (use_autograd=True). Requires a model that exposes a "
        "`charges` output -- supported with --model aimnet2 / aimnet2_2025; "
        "rejected with --model mace / orb.",
    )
    p.add_argument(
        "--d3",
        action="store_true",
        help="Add DFT-D3(BJ) dispersion via DFTD3ModelWrapper, composed with "
        "the base model (and Ewald, if --ewald) inside a single "
        "PipelineModelWrapper. Damping parameters are selected at init time "
        "from helpers/constants.py::D3_PRESETS[args.model] so they match the "
        "functional the chosen model was trained against (wB97M-D3(BJ) for "
        "aimnet2, B97-3c-D3(BJ) for aimnet2_2025, PBE-D3(BJ) for mace). Must "
        "match the warmup and slc invocations' --d3 flag (and --run-name) to "
        "keep the physics consistent end-to-end.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="MD timestep in fs (default 0.5).",
    )
    p.add_argument(
        "--t-warmup",
        type=float,
        default=200.0,
        help="Warmup-phase target temperature in K (default 200). Must match "
        "the value passed to warmup.py so this melt driver finds the "
        "correct seed checkpoint -- the warmup writes stems like "
        "'<src>_<t_warmup_tag>_<dt_tag>' (e.g. 'npt_200k_dt0p5fs', "
        "'npt_100k_dt0p5fs').",
    )
    p.add_argument(
        "--melt-ps",
        type=float,
        default=15.0,
        help="Melt NVT duration at T_MELT in ps (default 15, matching warmup NVT). "
        "Use small values (e.g. 1.5 ps = 3000 steps at dt=0.5 fs) for smoke tests; "
        "a follow-up run with a larger value triggers the extend path and runs the delta.",
    )
    p.add_argument(
        "--t-melt",
        type=float,
        default=500.0,
        help="Melt-phase target temperature in K (default 500).",
    )
    p.add_argument(
        "--source",
        choices=["nvt", "npt"],
        default="npt",
        help="Which warmup checkpoint to start the melt from. "
        "'npt' (default) uses after_npt_<T_WARMUP_TAG>_<DT_TAG>.zarr -- "
        "the post-equilibration cell. 'nvt' uses "
        "after_nvt_<T_WARMUP_TAG>_<DT_TAG>.zarr -- the un-expanded "
        "crystal at rho~1.2 g/cm^3, which melts in-place at fixed V "
        "(melt-in-box). Outputs carry "
        "from_<source>_<T_WARMUP_TAG>_<DT_TAG> so (source, t_warmup, "
        "DT) variants sit side by side.",
    )
    p.add_argument(
        "--run-name",
        default="naphthalene_long",
        help="Identifies this run's artefact subdir under logs/<run-name>/ (and "
        "assets/<run-name>/figs/ via plot scripts). Must match the warmup "
        "run-name so the seed checkpoint is found. E.g. 'naphthalene_long_big', "
        "'naphthalene_long_2025'.",
    )
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="PyTorch device. Pin a specific GPU (e.g. 'cuda:2') when running "
        "alongside other GPU jobs on the same node.",
    )
    return p.parse_args()


def print_config(
    args: argparse.Namespace,
    n_nvt: int,
    dt_tag: str,
    t_warmup_tag: str,
    src_ck: str,
) -> None:
    """Echo the resolved run configuration so CLI overrides are visible."""
    bar = "=" * 72
    ewald_msg = (
        "on (base model + EwaldModelWrapper via PipelineModelWrapper)"
        if args.ewald
        else f"off (bare {args.model})"
    )
    d3_params = D3_PRESETS[args.model]
    d3_msg = (
        f"on (DFTD3ModelWrapper, D3_PRESETS[{args.model!r}]: "
        f"a1={d3_params['a1']} a2={d3_params['a2']} Bohr s8={d3_params['s8']}, "
        "cutoff=<wrapper default 15 A>)"
        if args.d3
        else "off"
    )
    print(bar)
    print(f"Run name:             {args.run_name}")
    print(f"Device:               {args.device}")
    print(f"Material:             {args.material}")
    print(f"Model alias:          {args.model}")
    print(f"Ewald:                {ewald_msg}")
    print(f"D3:                   {d3_msg}")
    print(f"Timestep:             {args.dt} fs   (DT_TAG={dt_tag})")
    print(f"Melt NVT:             {args.melt_ps} ps ({n_nvt} steps)")
    print(f"T_MELT:               {args.t_melt} K")
    print(f"Warmup target T:      {args.t_warmup} K (T_TAG={t_warmup_tag})")
    print(f"Source warmup ck:     {src_ck}")
    print(f"Friction:             {FRICTION} fs^-1")
    print(bar)


def main() -> None:
    args = parse_args()

    if args.ewald and args.model in {"mace", "orb"}:
        raise SystemExit(
            f"--ewald is not supported with --model {args.model}: the wrapper "
            "does not expose a `charges` output, so it cannot compose with "
            "EwaldModelWrapper. Drop --ewald or switch to --model aimnet2 / "
            "aimnet2_2025."
        )

    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    log_dir = Path("logs") / args.run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    n_nvt = int(args.melt_ps * 1000 / args.dt)

    # DT_TAG / T_WARMUP_TAG-tagged names: checkpoint key (<phase>_ck) drives
    # after_<stage>.{zarr,json} under CKPT_DIR; file stem (<phase>_fs)
    # drives the .csv / .zarr trajectory basenames in LOG_DIR. Melt is
    # downstream of warmup, so both stems carry `from_<source>` to
    # record which warmup endpoint seeded this run. The warmup-temperature
    # tag is only on the source-checkpoint stem (we look up the warmup that
    # was run at this T); the melt artefacts themselves stay melt-T-only,
    # so reruns from different warmup temperatures don't collide.
    src_ck = f"{args.source}_{t_warmup_tag}_{dt_tag}"  # warmup checkpoint to seed from
    melt_ck = f"meltgen_nvt_500k_from_{args.source}_{t_warmup_tag}_{dt_tag}"
    melt_fs = f"melt_nvt_500k_from_{args.source}_{t_warmup_tag}_{dt_tag}"

    print_config(args, n_nvt, dt_tag, t_warmup_tag, src_ck)

    if not checkpoint_exists(src_ck, log_dir):
        logger.error(
            "[MELT] missing warmup checkpoint at {}/after_{}.zarr -- "
            "run warmup.py first.",
            ckpt_dir,
            src_ck,
        )
        sys.exit(1)

    # Build model -- --ewald and --d3 each append a wrapper into a single
    # PipelineGroup; bare base model stays the fast path when both are off.
    base = build_base_model(args.model, args.device)
    print(
        f"Base model loaded on {args.device} (alias={args.model}), "
        f"cutoff={base.model_config.neighbor_config.cutoff} A"
    )

    ewald = None
    d3 = None

    if args.ewald:
        # Need positions/cell to estimate Ewald parameters; load the seed
        # checkpoint once here and discard. Inexpensive vs the model load.
        _seed = load_checkpoint(src_ck, log_dir, args.device)
        ewald_params = estimate_ewald_parameters(
            _seed.positions, _seed.cell, _seed.batch_idx
        )
        ewald_cutoff = ewald_params.real_space_cutoff.max().item()
        ewald = EwaldModelWrapper(
            cutoff=ewald_cutoff, accuracy=1e-6, hybrid_forces=False
        )
        print(
            f"Ewald cutoff: {ewald_cutoff:.2f} A  (accuracy=1e-6, hybrid_forces=False)"
        )
        del _seed

    if args.d3:
        d3_params = D3_PRESETS[args.model]
        d3 = DFTD3ModelWrapper(**d3_params)
        # DFTD3ModelWrapper's default active_outputs is {'energy', 'forces'} --
        # 'stress' is NOT included. Without this set_config, a `base + d3`
        # pipeline asks d3 only for energy + forces and silently drops d3's
        # stress contribution. The attractive dispersion virial then never
        # reaches downstream stress consumers. Also move buffers (rcov /
        # r4r2 / c6ab / cn_ref) onto the integration device so the per-step
        # kernel runs on GPU.
        d3.set_config("active_outputs", {"energy", "forces", "stress"})
        d3 = d3.to(args.device)
        print(f"D3 cutoff: {d3.cutoff:.2f} A  (preset D3_PRESETS[{args.model!r}])")

    if args.ewald:
        # Ewald needs (a) the base model's `charges` output wired into its
        # `node_charges` input and (b) forces/stress derived via autograd
        # over the summed (base + ewald [+ d3]) energy -- neither of which
        # the BaseModelMixin `+` operator does (it produces independent
        # single-step direct groups). Keep the explicit PipelineGroup +
        # use_autograd=True for the Ewald path; D3 piggybacks when present.
        steps = [base, ewald] + ([d3] if d3 is not None else [])
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=steps, use_autograd=True)]
        )
        pipe.set_config("active_outputs", {"energy", "forces", "stress", "charges"})
        model = pipe
    elif args.d3:
        # Match the warmup driver's D3-only composition pattern
        # (warmup.py:608-626) and the alchemi-toolkit-demo reference
        # (example/main.py:506: `model = orb_wrapper + d3_model`). The `+`
        # operator builds a PipelineModelWrapper of single-step direct
        # groups whose energies, forces, and stresses are summed element-
        # wise via sum_outputs.
        #
        # active_outputs must be set on each submodule BEFORE composition.
        # pipe.set_config('active_outputs', ...) only updates the pipeline-
        # level declaration; it does not recurse into child steps. AIMNet2's
        # default active_outputs is {energy, forces, charges} (no stress)
        # and DFTD3ModelWrapper's default is {energy, forces} (no stress),
        # so without explicit per-step config sum_outputs sees zero stress
        # from both.
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base + d3
        model.set_config("active_outputs", {"energy", "forces", "stress"})
    else:
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base
    print(f"active_outputs: {sorted(model.model_config.active_outputs)}")

    _melt_t0 = time.monotonic()

    # --- Melt NVT @ T_MELT -------------------------------------------------
    melt_meta = load_stage_meta(melt_ck, log_dir)
    melt_done = int(melt_meta["steps_completed"]) if melt_meta else 0
    melt_can_extend = checkpoint_exists(melt_ck, log_dir)

    if checkpoint_exists(melt_ck, log_dir) and melt_done >= n_nvt:
        logger.info("[MELT] skip (checkpoint covers {} >= {} steps)", melt_done, n_nvt)
        batch = load_checkpoint(melt_ck, log_dir, args.device)
    else:
        _t_stage = time.monotonic()
        if melt_can_extend:
            n_delta = n_nvt - melt_done
            logger.info(
                "[MELT] extend ({} -> {} steps, +{})", melt_done, n_nvt, n_delta
            )
            batch = load_checkpoint(melt_ck, log_dir, args.device)
            part = next_part_index(log_dir, melt_fs)
        else:
            n_delta = n_nvt
            logger.info("[MELT] start  ({} steps, from {})", n_delta, src_ck)
            batch = load_checkpoint(src_ck, log_dir, args.device)
            print(
                f"Loaded warmup {src_ck} checkpoint; density={compute_density(batch)[0]:.3f} g/cm3"
            )
            print(
                f"Cell lengths: {[f'{l:.2f}' for l in batch.cell.squeeze().norm(dim=-1).tolist()]} A"
            )
            batch.velocities = torch.zeros_like(batch.positions)
            initialize_velocities(
                batch.velocities,
                batch.atomic_masses,
                temperature=torch.tensor([args.t_melt], device=args.device),
                batch_idx=batch.batch_idx,
                random_seed=123,
                remove_com=True,
                rescale=True,
            )
            part = 1

        melt_csv_path, melt_zarr_path = part_paths(log_dir, melt_fs, part)
        melt_zarr = fresh_zarr_sink(
            melt_zarr_path,
            capacity=n_delta // SNAPSHOT_EVERY + 10,
        )
        melt_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(melt_csv_path),
            frequency=LOG_EVERY,
        )
        melt_out = LoggingHook(
            backend="custom",
            writer_fn=stdout_writer,
            custom_scalars=DYNAMICS_SCALARS,
            frequency=LOG_EVERY,
        )
        nvt_melt = NVTLangevin(
            model=model,
            dt=args.dt,
            temperature=args.t_melt,
            friction=FRICTION,
            n_steps=n_delta,
        )
        for h in [
            *make_safety_hooks(model),
            SnapshotHook(sink=melt_zarr, frequency=SNAPSHOT_EVERY),
            melt_csv,
            melt_out,
        ]:
            nvt_melt.register_hook(h)
        with melt_csv, melt_out:
            batch = nvt_melt.run(batch)
        save_checkpoint(batch, melt_ck, log_dir)
        save_stage_meta(melt_ck, log_dir, n_nvt)
        logger.info(
            "[MELT->done] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _melt_t0,
        )

    fmax_final = batch.forces.norm(dim=-1).max().item()
    density = compute_density(batch)[0]
    print(f"Melt done: fmax={fmax_final:.4f} eV/A, density={density:.3f} g/cm3")
    print(
        f"Cell lengths: {[f'{l:.2f}' for l in batch.cell.squeeze().norm(dim=-1).tolist()]} A"
    )
    print(f"Melt script complete - checkpoints under {ckpt_dir}")


if __name__ == "__main__":
    main()
