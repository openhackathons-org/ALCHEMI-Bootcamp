"""Consolidated warmup driver for the SLC pipeline.

FIRE2 -> NVT @ T_warmup -> NPT @ T_warmup @ 1 atm (anisotropic pressure
coupling). T_warmup defaults to 200 K and is configurable via `--t-warmup`;
all varying physics + bookkeeping settings are CLI flags so one driver
covers every warmup variant.

The driver is material- and model-agnostic. `--material <name>` resolves to
`data/<name>.cif`; `--model {aimnet2, aimnet2_2025, mace}` selects the
underlying MLIP. File stems inside `logs/<run-name>/` are unchanged
regardless of material/model/corrections -- disambiguation between runs is
purely directory-level (set `--run-name` per run).

Variants at a glance (choose via flags)::

    # Bare AIMNet2 (wB97M-D3), naphthalene, 200-mol (5,5,4) supercell -- defaults
    python warmup.py

    # Smaller 108-mol cell for cheap iteration (pair with a distinct
    # --run-name and thread it through melt + slc)
    python warmup.py \\
        --supercell 3,6,3 --run-name naphthalene_long_small

    # AIMNet2 + Ewald long-range electrostatics
    python warmup.py \\
        --ewald \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_ewald

    # AIMNet2-2025 (B97-3c + D3, intermolecular retraining; recommended
    # for condensed phase per aimnetcentral)
    python warmup.py \\
        --model aimnet2_2025 \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_2025

    # AIMNet2-2025 + explicit DFT-D3(BJ) dispersion. Damping params come
    # from helpers/constants.py::D3_PRESETS[args.model] (model-keyed so
    # the correction matches the functional the model was trained against).
    # Pick a fresh --run-name; thread the same name through melt + slc.
    python warmup.py \\
        --model aimnet2_2025 --d3 \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_d3_2025

    # MACE-MP-0 (medium-0b2 foundation checkpoint). MACE has no charges
    # output so --ewald is rejected at parse time; --d3 is permitted with
    # PBE-D3(BJ) damping (correction-on-correction since MACE-MP-0 is
    # PBE+D3-trained).
    python warmup.py \\
        --model mace \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_mace

    # ORB-v3 conservative-inf-omat (OMAT24, PBE+D3-trained). Same charges-
    # free constraint as MACE: --ewald is rejected. --d3 is permitted with
    # PBE-D3(BJ) damping; analytical ORB forces/stress survive the
    # autograd pipeline via OrbV3Wrapper.direct_derivative_keys.
    python warmup.py \\
        --model orb \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_orb

    # Both long-range corrections combined (Ewald electrostatics + D3
    # dispersion). The two wrappers compose inside one PipelineGroup;
    # use a distinct --run-name so artefacts don't clobber single-fix runs.
    python warmup.py \\
        --model aimnet2_2025 --ewald --d3 \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --run-name naphthalene_long_ewald_d3_2025

    # Same as the aimnet2_2025 line but warm to 100 K instead of the
    # default 200 K (the T_TAG flips from '200k' to '100k' so artefacts
    # coexist on disk under the same --run-name).
    python warmup.py \\
        --model aimnet2_2025 \\
        --nvt-ps 15 --npt-ps 50 --fmax 0.15 \\
        --t-warmup 100 --run-name naphthalene_long_2025

    # Packmol-built amorphous box (50 mol naphthalene at 1.0 g/cm^3,
    # cubic cell). Single-molecule template is auto-extracted from the
    # CIF. --supercell is ignored; pick a fresh --run-name to keep the
    # disordered start separate from crystal-derived runs.
    python warmup.py \\
        --packmol --n-molecules 50 --initial-density 1.0 \\
        --run-name naphthalene_long_packmol

Stage checkpoints + integrator state carry a DT_TAG suffix
(e.g. `dt0p5fs`) and a T_WARMUP_TAG (e.g. `200k`, `100k`) so different
timesteps and warmup temperatures sit side-by-side without overwriting.
Re-running is idempotent: existing stages skip, stages
whose budget has grown extend via the `partN` path, missing stages
start fresh. Every stage writes its checkpoint on exit.
"""

import argparse
import time
from pathlib import Path

import torch
from ase.io import read as ase_read
from helpers import (
    D3_PRESETS,
    DYNAMICS_SCALARS,
    OrbV3Wrapper,
    checkpoint_exists,
    compute_density,
    extract_single_molecule,
    fresh_zarr_sink,
    integrator_state_exists,
    load_checkpoint,
    load_integrator_state,
    load_stage_meta,
    make_safety_hooks,
    next_part_index,
    pack_liquid_box,
    part_paths,
    save_checkpoint,
    save_integrator_state,
    save_stage_meta,
    stdout_writer,
)
from loguru import logger
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.models.aimnet2 import AIMNet2Wrapper
from nvalchemi.models.dftd3 import DFTD3ModelWrapper
from nvalchemi.models.ewald import EwaldModelWrapper
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)

# torch._functorch.config.donated_buffer = False
# torch.set_float32_matmul_precision("high")


# Constants that are stable across the current variant set. Promote to CLI
# flags when a future experiment first needs to vary one.
FRICTION = 0.01  # fs^-1 (= 10 ps^-1, correlation ~100 fs)
P_1ATM = 101325.0 / 1.602176634e11  # eV/A^3 (NPT target per diagonal axis)
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


def parse_supercell(s: str) -> tuple[int, int, int]:
    try:
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--supercell expects 'a,b,c' with three ints, got {s!r}"
        ) from e
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--material",
        default="naphthalene",
        help="Material name; resolves to `data/<material>.cif`. Default "
        "'naphthalene' (the canonical Part-2 molecular crystal). The CIF is "
        "the only material-specific input; everything downstream "
        "(supercell, MD, analysis) is structure-agnostic.",
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
        "aimnet2, B97-3c-D3(BJ) for aimnet2_2025, PBE-D3(BJ) for mace). The "
        "element-wise reference cache (rcov / r4r2 / c6ab / cn_ref) "
        "auto-downloads from the Grimme group archive on first use to "
        "~/.cache/nvalchemiops/dftd3_parameters.pt. Pick a fresh --run-name "
        "to keep D3 artefacts separate from bare runs.",
    )
    p.add_argument(
        "--supercell",
        type=parse_supercell,
        default=(5, 5, 4),
        help="Supercell replication 'a,b,c' (default '5,5,4' = 200 mol, the "
        "canonical Part-2 cell; '3,6,3' = 108 mol smaller cell -- pair with "
        "a distinct `--run-name` and thread it through melt + slc to keep "
        "artefacts in sync). Ignored when --packmol is set.",
    )
    p.add_argument(
        "--packmol",
        action="store_true",
        help="Build the simulation cell with Packmol instead of replicating "
        "the CIF unit cell. Extracts one molecule from data/<material>.cif "
        "(largest connected component, PBC-stitched) and packs --n-molecules "
        "copies into a cubic periodic box sized for --initial-density. "
        "Produces an amorphous / liquid-like starting configuration -- use "
        "for glassy / liquid initial conditions, NOT the canonical "
        "crystal-phase Part-2 SLC pipeline. Pick a fresh --run-name so "
        "artefacts stay distinct from CIF-supercell runs; --supercell is "
        "ignored.",
    )
    p.add_argument(
        "--n-molecules",
        type=int,
        default=200,
        help="Number of molecules to pack when --packmol is set (default 200, "
        "matching the canonical 5,5,4 naphthalene supercell). May be "
        "increased automatically if --initial-density gives a cubic box "
        "smaller than 2 * model cutoff.",
    )
    p.add_argument(
        "--initial-density",
        type=float,
        default=1.0,
        help="Target initial density in g/cm^3 for the Packmol box (default "
        "1.0). For naphthalene the experimental 295 K density is 1.18; "
        "starting below that gives FIRE2 + NPT room to compact.",
    )
    p.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol minimum inter-atom distance in A (default 2.0). "
        "Smaller values pack tighter but risk overlap that FIRE2 has "
        "to relax out.",
    )
    p.add_argument(
        "--packmol-nloop",
        type=int,
        default=50,
        help="Packmol GENCAN iteration cap per molecule (default 50 = "
        "packmol default). At densities close to experimental crystal "
        "packing, packmol may exit 'ENDED WITHOUT PERFECT PACKING'; the "
        "wrapper accepts the best-effort output but bumping --packmol-nloop "
        "(e.g. 200) gives packmol more attempts.",
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
        help="Warmup-phase NVT + NPT target temperature in K (default 200). "
        "Tagged into checkpoint and artefact stems as f'{int(T)}k' "
        "(e.g. '200k', '100k') so runs at different solid-phase "
        "temperatures coexist on disk under the same --run-name. The "
        "matching value must be passed to melt.py and slc.py so they "
        "find the right seed checkpoint.",
    )
    p.add_argument(
        "--nvt-ps",
        type=float,
        default=30.0,
        help="NVT thermalisation duration at the warmup target T in ps (default 30). "
        "Use small values (e.g. 1.5 ps = 3000 steps at dt=0.5 fs) for smoke tests; "
        "a follow-up run with a larger value triggers the extend path and runs the delta.",
    )
    p.add_argument(
        "--npt-ps",
        type=float,
        default=75.0,
        help="NPT equilibration duration at the warmup target T in ps (default 75). "
        "Use small values (e.g. 1.5 ps = 3000 steps at dt=0.5 fs) for smoke tests; "
        "a follow-up run with a larger value triggers the extend path and runs the delta.",
    )
    p.add_argument(
        "--pressure-coupling",
        choices=["anisotropic", "isotropic"],
        default="anisotropic",
        help="NPT pressure coupling mode (default 'anisotropic'). "
        "'anisotropic' integrates each cell axis against its own diagonal "
        "stress (and allows shear) -- correct for the Part-2 SLC pipeline "
        "where the two-phase solid/liquid interface breaks cell isotropy. "
        "'isotropic' couples Tr(stress) / 3 to a uniform cubic dilation -- "
        "the physical choice for a homogeneous liquid (e.g. --packmol "
        "amorphous boxes), where per-axis stress noise can otherwise drive "
        "runaway expansion. Drives the pressure tensor shape: scalar P_1atm "
        "for isotropic, rank-2 [P,P,P] for anisotropic (per the toolkit "
        "convention -- scalar + 'anisotropic' silently falls through to "
        "isotropic).",
    )
    p.add_argument(
        "--barostat-time",
        type=float,
        default=4000.0,
        help="MTK barostat coupling τ_P in fs (default 2000 = 2 ps; ~π·τ_P ≈ 6 ps "
        "equilibration). Larger values (e.g. 5000) decouple the barostat from "
        "interface motion and reduce cell oscillations at the cost of slower "
        "volume relaxation. Tagged into the NPT checkpoint via the existing "
        "T_WARMUP/DT keys -- if you rerun with a different τ_P under the same "
        "--run-name, back up the prior NPT artefacts first.",
    )
    p.add_argument(
        "--thermostat-time",
        type=float,
        default=100.0,
        help="NHC thermostat coupling τ_T in fs (default 100 = 0.1 ps). Sets "
        "the Nose-Hoover chain response time in the NPT stage (NVT uses "
        "Langevin friction instead, so this only affects NPT). Larger values "
        "couple the thermostat more loosely with larger T fluctuations; "
        "smaller values give tighter T control. Tagged into the NPT checkpoint "
        "via the existing T_WARMUP/DT keys -- if you rerun with a different "
        "τ_T under the same --run-name, back up the prior NPT artefacts first.",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="FIRE2 convergence fmax in eV/A (default 0.05 for bare AIMNet2; "
        "use ~0.15 with --ewald where autograd stress has a higher noise floor).",
    )
    p.add_argument(
        "--fire-max-steps",
        type=int,
        default=5000,
        help="FIRE2 step budget cap (default 5000). FIRE exits earlier when "
        "fmax<--fmax via the convergence hook; this is the upper safety "
        "bound. Lower it if FIRE plateaus on a stiff landscape (e.g. SLC).",
    )
    p.add_argument(
        "--run-name",
        default="naphthalene_long",
        help="Identifies this run's artefact subdir under logs/<run-name>/ (and "
        "assets/<run-name>/figs/ via plot scripts). E.g. 'naphthalene_long_big', "
        "'naphthalene_long_ewald', 'naphthalene_long_2025'.",
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
    n_npt: int,
    dt_tag: str,
    t_warmup_tag: str,
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
    print(f"Material:             {args.material}  (data/{args.material}.cif)")
    print(f"Model alias:          {args.model}")
    print(f"Ewald:                {ewald_msg}")
    print(f"D3:                   {d3_msg}")
    if args.packmol:
        print(
            f"Cell construction:    packmol "
            f"(n_mol={args.n_molecules}, rho_init={args.initial_density} g/cm^3, "
            f"tol={args.packmol_tolerance} A)"
        )
    else:
        print(f"Supercell:            {args.supercell}")
    print(f"Timestep:             {args.dt} fs   (DT_TAG={dt_tag})")
    print(f"Thermalise NVT:       {args.nvt_ps} ps ({n_nvt} steps)")
    print(f"Equilibrate NPT:      {args.npt_ps} ps ({n_npt} steps)")
    print(f"FIRE fmax / max steps:{args.fmax} eV/A / {args.fire_max_steps}")
    print(
        f"T target / P target:  {args.t_warmup} K (T_TAG={t_warmup_tag}) / "
        f"1 atm ({P_1ATM:.3e} eV/A^3 per diag)"
    )
    print(f"Pressure coupling:    {args.pressure_coupling}")
    print(
        f"Friction / τ_T / τ_P: {FRICTION} fs^-1 / {args.thermostat_time} fs / {args.barostat_time} fs"
    )
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

    cif_path = Path("data") / f"{args.material}.cif"
    if not cif_path.exists():
        raise SystemExit(
            f"{cif_path} not found. Pass --material <name> matching a CIF "
            f"under data/ (available: "
            f"{sorted(p.stem for p in Path('data').glob('*.cif'))})."
        )

    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    log_dir = Path("logs") / args.run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    n_nvt = int(args.nvt_ps * 1000 / args.dt)
    n_npt = int(args.npt_ps * 1000 / args.dt)

    print_config(args, n_nvt, n_npt, dt_tag, t_warmup_tag)

    unit_cell = ase_read(str(cif_path))
    a_cif, b_cif, c_cif = unit_cell.cell.lengths()
    alpha, beta, gamma = unit_cell.cell.angles()
    print(
        f"Unit cell: {len(unit_cell)} atoms, "
        f"a={a_cif:.3f}, b={b_cif:.3f}, c={c_cif:.3f} A, "
        f"alpha={alpha:.1f}, beta={beta:.1f}, gamma={gamma:.1f} deg"
    )

    # Build model first so we know the neighbor cutoff before sizing the
    # cell -- --packmol needs it for the MIC guard, and the non-packmol
    # path still uses it for the post-construction assertion below.
    base = build_base_model(args.model, args.device)
    base_cutoff = base.model_config.neighbor_config.cutoff
    print(
        f"Base model loaded on {args.device} (alias={args.model}), "
        f"cutoff={base_cutoff} A"
    )

    if args.packmol:
        # D3's default 15 A cutoff dominates the stack when --d3 is on; use
        # it for MIC sizing so packmol doesn't hand FIRE a box that fails
        # the D3 wrapper's own assertion. Ewald is post-validated via
        # estimate_ewald_parameters.
        mic_cutoff = max(base_cutoff, 15.0) if args.d3 else base_cutoff
        monomer = extract_single_molecule(unit_cell)
        print(
            f"Monomer extracted from {cif_path.name}: "
            f"{len(monomer)} atoms ({monomer.get_chemical_formula()})"
        )
        supercell, n_packed = pack_liquid_box(
            monomer,
            n_molecules=args.n_molecules,
            target_density=args.initial_density,
            model_cutoff=mic_cutoff,
            tolerance=args.packmol_tolerance,
            nloop=args.packmol_nloop,
        )
        box_side = float(supercell.cell.lengths()[0])
        print(
            f"Packmol cell: {n_packed} molecules, cubic side={box_side:.2f} A, "
            f"target rho={args.initial_density} g/cm^3"
        )
    else:
        supercell = unit_cell * args.supercell
        print(f"Supercell {args.supercell}: {len(supercell)} atoms")
    n = len(supercell)

    data = AtomicData.from_atoms(supercell)
    data.forces = torch.zeros(n, 3)
    data.energy = torch.zeros(1, 1)
    data.stress = torch.zeros(1, 3, 3)
    data.add_node_property("velocities", torch.zeros(n, 3))
    data.charge = torch.zeros(1, 1)
    batch = Batch.from_data_list([data], device=args.device)

    cell_lengths = batch.cell.squeeze().norm(dim=-1)
    print(
        f"Cell lengths: {[f'{l:.2f}' for l in cell_lengths.tolist()]} A, "
        f"density: {compute_density(batch)[0]:.3f} g/cm3"
    )

    # PBC minimum-image convention requires every cell vector to exceed
    # 2 * cutoff so atoms don't interact with their own periodic image.
    # Use the actual model's cutoff rather than a hardcoded AIMNet2 5 A.
    min_cell_required = 2 * base_cutoff
    assert (cell_lengths > min_cell_required).all(), (
        f"Cell too small for {args.model!r} cutoff: cell lengths "
        f"{[round(l, 2) for l in cell_lengths.tolist()]} A include values "
        f"<= 2 * cutoff ({min_cell_required:.2f} A). Increase --supercell "
        f"or use --packmol with a larger --n-molecules / lower "
        f"--initial-density."
    )

    ewald = None
    d3 = None

    if args.ewald:
        ewald_params = estimate_ewald_parameters(
            batch.positions, batch.cell, batch.batch_idx
        )
        ewald_cutoff = ewald_params.real_space_cutoff.max().item()
        ewald = EwaldModelWrapper(
            cutoff=ewald_cutoff, accuracy=1e-6, hybrid_forces=False
        )
        print(
            f"Ewald cutoff: {ewald_cutoff:.2f} A  (accuracy=1e-6, hybrid_forces=False)"
        )

    if args.d3:
        d3_params = D3_PRESETS[args.model]
        d3 = DFTD3ModelWrapper(**d3_params)
        # DFTD3ModelWrapper's default active_outputs is {'energy', 'forces'} --
        # 'stress' is NOT included. Without this set_config, a `base + d3`
        # pipeline asks d3 only for energy + forces and silently drops d3's
        # stress contribution. The attractive dispersion virial then never
        # reaches the NPT barostat and the box expands monotonically. Also
        # move buffers (rcov / r4r2 / c6ab / cn_ref) onto the integration
        # device so the per-step kernel runs on GPU.
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
        # Match alchemi-toolkit-demo's composition pattern
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
        # from both and the NPT barostat reads ~kinetic-only pressure -- the
        # condensed-phase virial pull-back never reaches the integrator and
        # the box expands monotonically.
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base + d3
        model.set_config("active_outputs", {"energy", "forces", "stress"})
    else:
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base
    print(f"active_outputs: {sorted(model.model_config.active_outputs)}")

    nc = base.model_config.neighbor_config
    effective_cutoff = nc.cutoff + nc.skin
    min_cell_dim = cell_lengths.min().item()
    print(
        f"Effective cutoff (cutoff+skin): {effective_cutoff:.2f} A, min cell dim: {min_cell_dim:.2f} A"
    )
    if effective_cutoff >= min_cell_dim / 2:
        logger.warning(
            f"Cutoff+skin {effective_cutoff:.2f} A >= half min cell dim "
            f"{min_cell_dim / 2:.2f} A. Neighbor list does redundant work."
        )

    # DT_TAG / T_WARMUP_TAG-tagged names: checkpoint key (<phase>_ck) drives
    # after_<stage>.{zarr,json,pt} under CKPT_DIR; file stem (<phase>_fs)
    # drives the .csv / .zarr trajectory basenames in LOG_DIR. The warmup
    # temperature tag (e.g. '200k', '100k') sits between phase and DT so
    # different solid-phase temperatures sit side-by-side under the same
    # --run-name without overwriting.
    fire_ck = f"fire_{dt_tag}"
    nvt_ck = f"nvt_{t_warmup_tag}_{dt_tag}"
    npt_ck = f"npt_{t_warmup_tag}_{dt_tag}"
    fire_fs = f"warmup_fire_{dt_tag}"
    nvt_fs = f"warmup_nvt_{t_warmup_tag}_{dt_tag}"
    npt_fs = f"warmup_npt_{t_warmup_tag}_{dt_tag}"

    print(
        f"Warmup: FIRE2 (<={args.fire_max_steps}) -> NVT {n_nvt} -> NPT {n_npt} steps"
    )
    _warmup_t0 = time.monotonic()

    # --- FIRE2 minimize ----------------------------------------------------
    if checkpoint_exists(fire_ck, log_dir):
        logger.info("[FIRE] skip (checkpoint exists); loading end-of-stage batch")
        batch = load_checkpoint(fire_ck, log_dir, args.device)
    else:
        logger.info("[FIRE] start  (budget: <={} steps)", args.fire_max_steps)
        _t_stage = time.monotonic()
        fire_zarr = fresh_zarr_sink(
            log_dir / f"{fire_fs}.zarr",
            capacity=args.fire_max_steps // SNAPSHOT_EVERY + 10,
        )
        fire_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(log_dir / f"{fire_fs}.csv"),
            frequency=LOG_EVERY,
        )
        fire_out = LoggingHook(
            backend="custom",
            writer_fn=stdout_writer,
            custom_scalars=DYNAMICS_SCALARS,
            frequency=LOG_EVERY,
        )
        fire_stage = FIRE2(
            model=model,
            dt=0.01,
            n_steps=args.fire_max_steps,
            convergence_hook=ConvergenceHook.from_fmax(threshold=args.fmax),
        )
        for h in [
            *make_safety_hooks(model, track_stress=False),
            SnapshotHook(sink=fire_zarr, frequency=SNAPSHOT_EVERY),
            fire_csv,
            fire_out,
        ]:
            fire_stage.register_hook(h)
        with fire_csv, fire_out:
            batch = fire_stage.run(batch)
        save_checkpoint(batch, fire_ck, log_dir)
        save_stage_meta(fire_ck, log_dir, args.fire_max_steps)
        logger.info(
            "[FIRE->NVT] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _warmup_t0,
        )

    # --- NVT thermalize @ args.t_warmup -----------------------------------
    nvt_meta = load_stage_meta(nvt_ck, log_dir)
    nvt_done = int(nvt_meta["steps_completed"]) if nvt_meta else 0
    nvt_can_extend = checkpoint_exists(nvt_ck, log_dir)
    nvt_label = f"NVT {t_warmup_tag.upper()}"

    if checkpoint_exists(nvt_ck, log_dir) and nvt_done >= n_nvt:
        logger.info(
            "[{}] skip (checkpoint covers {} >= {} steps)", nvt_label, nvt_done, n_nvt
        )
        batch = load_checkpoint(nvt_ck, log_dir, args.device)
    else:
        _t_stage = time.monotonic()
        if nvt_can_extend:
            n_delta = n_nvt - nvt_done
            logger.info(
                "[{}] extend ({} -> {} steps, +{})", nvt_label, nvt_done, n_nvt, n_delta
            )
            batch = load_checkpoint(nvt_ck, log_dir, args.device)
            part = next_part_index(log_dir, nvt_fs)
        else:
            n_delta = n_nvt
            logger.info("[{}] start  ({} steps)", nvt_label, n_delta)
            batch.velocities = torch.zeros_like(batch.positions)
            initialize_velocities(
                batch.velocities,
                batch.atomic_masses,
                temperature=torch.tensor([args.t_warmup], device=args.device),
                batch_idx=batch.batch_idx,
                random_seed=42,
                remove_com=True,
                rescale=True,
            )
            part = 1

        nvt_csv_path, nvt_zarr_path = part_paths(log_dir, nvt_fs, part)
        nvt_zarr = fresh_zarr_sink(
            nvt_zarr_path, capacity=n_delta // SNAPSHOT_EVERY + 10
        )
        nvt_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(nvt_csv_path),
            frequency=LOG_EVERY,
        )
        nvt_out = LoggingHook(
            backend="custom",
            writer_fn=stdout_writer,
            custom_scalars=DYNAMICS_SCALARS,
            frequency=LOG_EVERY,
        )
        nvt_stage = NVTLangevin(
            model=model,
            dt=args.dt,
            temperature=args.t_warmup,
            friction=FRICTION,
            n_steps=n_delta,
        )
        for h in [
            *make_safety_hooks(model),
            SnapshotHook(sink=nvt_zarr, frequency=SNAPSHOT_EVERY),
            nvt_csv,
            nvt_out,
        ]:
            nvt_stage.register_hook(h)
        with nvt_csv, nvt_out:
            batch = nvt_stage.run(batch)
        save_checkpoint(batch, nvt_ck, log_dir)
        save_stage_meta(nvt_ck, log_dir, n_nvt)
        logger.info(
            "[NVT->NPT] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _warmup_t0,
        )

    # --- NPT equilibrate @ args.t_warmup ----------------------------------
    npt_meta = load_stage_meta(npt_ck, log_dir)
    npt_done = int(npt_meta["steps_completed"]) if npt_meta else 0
    npt_can_extend = checkpoint_exists(npt_ck, log_dir) and integrator_state_exists(
        npt_ck, log_dir
    )
    npt_label = f"NPT {t_warmup_tag.upper()}"

    if checkpoint_exists(npt_ck, log_dir) and npt_done >= n_npt:
        logger.info(
            "[{}] skip (checkpoint covers {} >= {} steps)", npt_label, npt_done, n_npt
        )
        batch = load_checkpoint(npt_ck, log_dir, args.device)
    else:
        _t_stage = time.monotonic()
        if npt_can_extend:
            n_delta = n_npt - npt_done
            logger.info(
                "[{}] extend ({} -> {} steps, +{})", npt_label, npt_done, n_npt, n_delta
            )
            batch = load_checkpoint(npt_ck, log_dir, args.device)
            preloaded_state = load_integrator_state(npt_ck, log_dir, args.device)
            part = next_part_index(log_dir, npt_fs)
        else:
            n_delta = n_npt
            logger.info("[{}] start  ({} steps)", npt_label, n_delta)
            preloaded_state = None
            part = 1

        npt_csv_path, npt_zarr_path = part_paths(log_dir, npt_fs, part)
        npt_zarr = fresh_zarr_sink(
            npt_zarr_path, capacity=n_delta // SNAPSHOT_EVERY + 10
        )
        npt_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(npt_csv_path),
            frequency=LOG_EVERY,
        )
        npt_out = LoggingHook(
            backend="custom",
            writer_fn=stdout_writer,
            custom_scalars=DYNAMICS_SCALARS,
            frequency=LOG_EVERY,
        )
        # Pressure tensor shape follows the coupling mode: rank-2 [P,P,P]
        # for anisotropic (per-axis target), scalar P_1ATM for isotropic
        # (single hydrostatic target). Mixing scalar + anisotropic silently
        # falls through to isotropic at the toolkit level -- using the
        # documented pair here keeps the integrator's runtime path obvious.
        if args.pressure_coupling == "anisotropic":
            npt_pressure: float | torch.Tensor = torch.tensor(
                [[P_1ATM, P_1ATM, P_1ATM]], dtype=torch.float32
            )
        else:
            npt_pressure = P_1ATM
        npt_stage = NPT(
            model=model,
            dt=args.dt,
            temperature=args.t_warmup,
            pressure=npt_pressure,
            barostat_time=args.barostat_time,
            thermostat_time=args.thermostat_time,
            pressure_coupling=args.pressure_coupling,
            n_steps=n_delta,
        )
        for h in [
            *make_safety_hooks(model),
            SnapshotHook(sink=npt_zarr, frequency=SNAPSHOT_EVERY),
            npt_csv,
            npt_out,
        ]:
            npt_stage.register_hook(h)
        if preloaded_state is not None:
            npt_stage._state = preloaded_state
        with npt_csv, npt_out:
            batch = npt_stage.run(batch)
        save_checkpoint(batch, npt_ck, log_dir)
        save_integrator_state(npt_stage._state, npt_ck, log_dir)
        save_stage_meta(npt_ck, log_dir, n_npt)
        logger.info(
            "[NPT->done] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _warmup_t0,
        )

    fmax_final = batch.forces.norm(dim=-1).max().item()
    density = compute_density(batch)[0]
    print(f"Warmup done: fmax={fmax_final:.4f} eV/A, density={density:.3f} g/cm3")
    print(
        f"Cell lengths: {[f'{l:.2f}' for l in batch.cell.squeeze().norm(dim=-1).tolist()]} A"
    )
    print(f"Warmup script complete - checkpoints under {ckpt_dir}")


if __name__ == "__main__":
    main()
