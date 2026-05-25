"""Consolidated SLC driver for the SLC pipeline.

Mirrors the SLC construction / pre-equilibration / production cells of
melting-point-slc.ipynb: stack the warm solid above the melt along b
(the monoclinic unique axis, so the interface is the (010) lattice
plane with a 90-degree angle at the interface -- direct c is tilted
~33.6 deg off (001) for P2_1/a naphthalene, so stacking along c would
not give a right-angle interface), then apply the Yoneya-Harada SLC
pre-equilibration (FIRE2 minimize, then 10 ps NVT to bring both halves
to a common temperature) before running anisotropic NPT across
TEMPS = [250..450 K] as a multi-graph Batch.
Visualisation is deliberately absent so the run has no plotting
dependencies and can execute under `docker exec` in a detached session
on the compute node -- decoupled from the Jupyter kernel, safe against
SSH / websocket drops.

The driver is material- and model-agnostic (set via `--material` and
`--model`; both must match the upstream warmup + melt invocations so the
seed checkpoints are found). Disambiguation between runs is purely
directory-level via `--run-name`; file stems inside that dir are
independent of material/model/corrections.

Variants at a glance (choose via flags)::

    # Bare AIMNet2 (wB97M-D3) -- defaults; auto-source select
    python slc.py

    # Larger 200-mol supercell (warmup --supercell 5,5,4 / *_big run-name)
    python slc.py --run-name naphthalene_long_big

    # AIMNet2-2025 melt-in-box (avoids post-NPT plastic-crystal artefact)
    python slc.py \\
        --run-name naphthalene_long_2025 --model aimnet2_2025 --source nvt

    # AIMNet2-2025 + DFT-D3(BJ) dispersion (preset auto-selected via
    # D3_PRESETS[args.model]). Match the warmup + melt invocations' --d3
    # flag and --run-name so the SLC stack inherits a physics-consistent
    # crystal/melt pair.
    python slc.py \\
        --run-name naphthalene_long_d3_2025 --model aimnet2_2025 \\
        --source nvt --d3

    # MACE-MP-0 (medium-0b2). --ewald is rejected with MACE since the
    # wrapper has no charges output.
    python slc.py \\
        --run-name naphthalene_long_mace --model mace --source nvt

    # ORB-v3 conservative-inf-omat (OMAT24, PBE+D3-trained). Same charges-
    # free constraint as MACE: --ewald is rejected. --d3 is permitted with
    # PBE-D3(BJ) damping.
    python slc.py \\
        --run-name naphthalene_long_orb --model orb --source nvt

Pipeline:
  1. Load warmup NPT solid + melt NVT liquid checkpoints.
  2. Construct single-graph SLC stack along b (cell b-vec doubled).
  3. Stage A -- FIRE2 minimize single-graph SLC (<= --fire-max-steps,
     fmax < --fmax) to relieve close contacts at the interface.
  4. Clone FIRE-minimized geometry into len(TEMPS) copies; reinit
     velocities at per-T Maxwell-Boltzmann targets.
  5. Stage B -- short NVT Langevin @ per-T for --nvt-ps to
     equalize the two halves to their shared target T.
  6. Stage C -- production anisotropic NPT @ per-T, 1 atm, --npt-ps
     per graph.

Two-phase usage:
  1. Smoke test:  python slc.py --npt-ps 1.5
  2. Full extend: python slc.py
The second invocation exercises the production NPT extend dispatch:
load the smoke-step checkpoint + integrator state, run the remaining
delta, write to slc_all_from_{src}_{DT_TAG}.part2.{zarr,csv}. Stages A
and B use their defaults on every run -- they are cheap relative to
production (FIRE is single-graph; Stage B is 20 000 steps vs 400 000 x
5 systems for Stage C).

`--temps <comma-list>` (default: full sweep) runs a strict subset of
TEMPS as one multi-graph Batch in the current process. Use it to
split the sweep across multiple GPUs: launch one process per GPU with
disjoint temperature subsets and `CUDA_VISIBLE_DEVICES=N` pinning
(see `slc_multi_gpu.sh` for the canonical 4-GPU 2+1+1+1 split). Every
stage artefact gains a `_t<T1>_<T2>...` suffix when a strict subset
is active, so concurrent processes do not collide on
checkpoints/CSVs/zarrs. Full-sweep runs keep the original unsuffixed
names for backwards compatibility with the notebook's diagnostics
loader.

`--source {npt,nvt,auto}` (default `auto`) picks **which warmup branch
feeds both halves** of the SLC stack so the two are cell-consistent:
  * `npt` -- crystal from after_npt_<T_WARMUP_TAG>_<DT_TAG>.zarr, melt
    from after_meltgen_nvt_500k_from_npt_<T_WARMUP_TAG>_<DT_TAG>.zarr.
    Full pipeline (FIRE -> NVT @ T_warmup -> NPT @ T_warmup -> melt -> SLC).
  * `nvt` -- crystal from after_nvt_<T_WARMUP_TAG>_<DT_TAG>.zarr, melt
    from after_meltgen_nvt_500k_from_nvt_<T_WARMUP_TAG>_<DT_TAG>.zarr.
    **NPT warmup is skipped entirely**; both halves stay in the
    unexpanded NVT cell. Useful when bare AIMNet2 can't hold the cell
    under anisotropic NPT (no Ewald -> density collapses) and you want
    the SLC to start from the pristine NVT geometry instead.
  * `auto` (default) -- tries the `npt` pair first, falls back to the
    `nvt` pair if NPT warmup is absent.
Mixing halves across branches is not allowed: the crystal and melt
cells would differ (the warmup NPT generally shifts `a`/`b`/`c` vs the
warmup NVT baseline), and stacking would cram the two halves into a
single geometry that makes no physical sense -- FIRE diverges within a
few hundred steps. The selected source is baked into every SLC stage
name as `from_{src}`, so runs with different branches (or DTs) coexist
on disk.

All stage / checkpoint / artefact names carry the DT_TAG suffix
(e.g. `dt0p5fs`), a T_WARMUP_TAG (e.g. `200k`, `100k`, matching the
warmup driver's `--t-warmup`), and a `from_{nvt,npt}` tag identifying
which melt half (and therefore which warmup endpoint) seeded this
SLC run.

Every stage writes its checkpoint on exit; re-running is idempotent. Pull
the artefacts back and open the notebook for the SLC diagnostics cell
(slc_timeseries / slc_endpoints / slc_rdf plots derive from
slc_all_from_{src}_{DT_TAG}.zarr + .csv + the final checkpoint this
script writes).

Preconditions: warmup.py and melt.py must have produced a matching
(warmup, melt) pair under logs/<run-name>/checkpoints/ at the same
`--t-warmup` value:
  * for `--source npt`: both `npt_{T_WARMUP_TAG}_{DT_TAG}` and
    `meltgen_nvt_500k_from_npt_{T_WARMUP_TAG}_{DT_TAG}`;
  * for `--source nvt`: both `nvt_{T_WARMUP_TAG}_{DT_TAG}` and
    `meltgen_nvt_500k_from_nvt_{T_WARMUP_TAG}_{DT_TAG}`.
`--source auto` (default) prefers the npt pair and falls back to the
nvt pair, matching the melt script's default (`--source npt`).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
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
    make_graph_tagged_writer,
    make_safety_hooks,
    min_pbc_distance,
    next_part_index,
    pack_with_fixed_obstacle,
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

torch._functorch.config.donated_buffer = False
torch.set_float32_matmul_precision("high")


# Constants that are stable across the current variant set. Promote to CLI
# flags when a future experiment first needs to vary one.
FRICTION = 0.01  # fs^-1 (= 10 ps^-1, correlation ~100 fs)
P_1ATM = 101325.0 / 1.602176634e11  # eV/A^3
SNAPSHOT_EVERY = 100  # steps (default; overridable via --snapshot-every)
LOG_EVERY = 100  # steps (default; overridable via --log-every)

# MACE foundation checkpoint resolved by `--model mace`. Matches the alias
# used in nvalchemi-toolkit/examples/advanced/04_mace_nvt.py.
MACE_CHECKPOINT = "medium-0b2"
# ORB foundation checkpoint resolved by `--model orb`. Conservative head,
# OMol-trained (must match warmup.py:ORB_ALIAS so the SLC dynamics use the
# same potential as the seed checkpoint -- mixing OMol and OMat24 variants
# across the warmup -> SLC chain gives inconsistent forces at the interface).
# Precision `float32-high` is the recommended A100/H100 MD setting per the
# orb-models README.
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


# Naphthalene atom layout in an nvalchemi Batch built from the (3,6,3) P2_1/a
# supercell is not a simple stride-N grouping: atoms block per unit cell as
# [C x 20, H x 16], but the 20 C atoms within a block are NOT partitioned
# into two contiguous 10-atom molecules -- C-C bonds span across [0:10] and
# [10:20]. Trying to guess the per-molecule partition analytically is
# fragile, so `_detect_naphthalene_molecules` instead runs a neighbor-list
# connected-components search once per half to discover molecules from
# bonding topology (cutoff 1.8 A catches C-C and C-H but not H-H or
# close-contact van-der-Waals). Topology is identity-preserving through
# dynamics, so the mapping computed on the unwrapped checkpoint is valid
# for the whole simulation.

TEMPS = [200, 250, 300, 350, 400, 450, 500]  # K (SLC production sweep -- canonical baseline;
# `--temps` accepts a strict subset to enable
# multi-GPU splitting without renaming artefacts)


def _detect_naphthalene_molecules(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
    c_c_cutoff: float = 1.6,
    c_h_cutoff: float = 1.4,
) -> list[torch.Tensor]:
    """Discover per-molecule atom indices via PBC-aware neighbor-list
    connected components with element-pair-aware bond cutoffs:

      - C-C bond: d < 1.6 A  (typical 1.40, max ~1.55)
      - C-H bond: d < 1.4 A  (typical 1.09, stretched up to ~1.35 at 500 K)
      - H-H: never bond (no H-H bonds in naphthalene)

    A single isotropic cutoff cannot separate the crystal (tight C-H)
    from the melt (stretched C-H): 1.5 A fragments melt C-H bonds, 1.7 A
    merges adjacent molecules via intermolecular H-H contacts. Element-
    pair filtering sidesteps both failure modes -- verified to partition
    both `nvt_200k` and `meltgen_nvt_500k` into exactly 108 C10H8 groups.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=atomic_numbers.cpu().numpy(),
        positions=positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=True,
    )
    i_list, j_list, d_list = neighbor_list(
        "ijd", atoms, cutoff=max(c_c_cutoff, c_h_cutoff)
    )
    n = positions.shape[0]
    Z = atomic_numbers.cpu().numpy()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, d in zip(i_list.tolist(), j_list.tolist(), d_list.tolist()):
        za, zb = int(Z[a]), int(Z[b])
        if za == 6 and zb == 6 and d < c_c_cutoff:
            union(a, b)
        elif {za, zb} == {1, 6} and d < c_h_cutoff:
            union(a, b)
        # H-H pairs never bond

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[torch.Tensor] = []
    for gid, members in groups.items():
        assert len(members) == 18, (
            f"molecule {gid} has {len(members)} atoms (expected 18)"
        )
        Zs = atomic_numbers[members].tolist()
        assert sorted(Zs) == [1] * 8 + [6] * 10, (
            f"molecule {gid} stoichiometry {sorted(Zs)} != C10H8"
        )
        out.append(torch.tensor(sorted(members), dtype=torch.long))
    return out


def _unwrap_molecules_inplace(
    positions: torch.Tensor,
    cell: torch.Tensor,
    mol_indices: list[torch.Tensor],
) -> None:
    """Pull every atom of each molecule to the minimum-image position relative
    to that molecule's first atom. Must be called on each SLC half *before*
    concatenation, using that half's *own* cell. Caller supplies the
    molecule partition (from ``_detect_naphthalene_molecules``) so this
    routine does not need to rediscover topology.
    """
    cell_inv = torch.linalg.inv(cell)
    for idx in mol_indices:
        idx = idx.to(positions.device)
        ref = positions[idx[0]].clone()
        dr = positions[idx] - ref
        frac = dr @ cell_inv
        frac -= frac.round()
        positions[idx] = ref + frac @ cell


def _wrap_molecules_by_com_inplace(
    positions: torch.Tensor,
    cell: torch.Tensor,
    mol_indices: list[torch.Tensor],
) -> None:
    """After stacking, fold whole molecules back into the stacked cell by
    their COM (integer lattice shifts only), so no molecule is split across
    the PBC. Uses uniform-mass COM for speed -- the shift is an integer
    number of lattice vectors so atomic-mass weighting cannot change it for
    compact molecules. Apply AFTER concatenation, using the STACKED cell.
    """
    cell_inv = torch.linalg.inv(cell)
    for idx in mol_indices:
        idx = idx.to(positions.device)
        com = positions[idx].mean(dim=0)
        com_frac = com @ cell_inv
        shift_frac = com_frac.floor()
        if shift_frac.abs().sum() > 0:
            positions[idx] = positions[idx] - shift_frac @ cell


# Interface vacuum gap for SLC stacking via `construct_slc_stack` (the
# meltgen-melt path; see its docstring). The --packmol path uses
# `build_packmol_slc_stack` with no gap concept -- Packmol's tolerance
# check against the fixed crystal half handles interface clashes during
# placement rather than after the fact.
SLC_INTERFACE_GAP = 4.0  # A


def construct_slc_stack(
    crystal_batch: Batch,
    melt_batch: Batch,
    *,
    device: str,
    gap: float = SLC_INTERFACE_GAP,
) -> tuple[Batch, list[torch.Tensor], int]:
    """Stack a crystal half above a melt half along the monoclinic b axis.

    b is the monoclinic unique axis (b perp a, b perp c), so stacking along
    direct b places the interface in the (010) lattice plane with the plane
    normal (b*) equal to the stacking direction -- the 90-degree interface
    condition. Stacking along direct c would use a c-vector tilted
    (beta - 90) = 33.6 deg off the (001) normal for P2_1/a naphthalene,
    which is why we stack along b rather than c.

    Vacuum gap at both interfaces (Morris-Song / Yoneya-Harada convention):
    placing the melt atoms flush against the crystal atoms via `+b_vec`
    introduces sub-Ang close contacts at the (010) interface (and the
    PBC-wrapped counterpart). Empirically: GAP=0 -> cross-interface min
    ~0.28 A (atom overlap, FIRE diverges); GAP=1.0 A -> 1.14 A (matches
    intra-molecular C-H floor); GAP=1.5 A -> 1.63 A. The stacked cell
    b-vector is extended by 2*gap so the PBC-wrap side gets the same
    clearance as the middle.

    GAP must absorb unwrap protrusion from BOTH halves at each interface.
    When a molecule straddles a half's PBC boundary, MIC unwrap places its
    atoms in an extended range like frac ~ [0.92, 1.06] (for naphthalene
    in the 35.84 A half cell, molecular extent ~0.14 frac = 5.2 A). Half
    that extent (~2.6 A) protrudes into the gap from each side, so the
    gap must be > ~5.2 A to guarantee no crystal-melt atom clashes at the
    interface. 4.0 A is empirically sufficient and stays below the 5 A
    AIMNet2 cutoff.

    Each half is unwrapped + COM-rewrapped in its OWN cell BEFORE
    concatenation. The melt was equilibrated with a WrapPeriodicHook in
    its own cell (|b_melt|), so molecules straddling its PBC boundary
    have atoms separated by ~|b_melt|. After stacking, |b_new| = 2|b_melt|
    + 2*gap, so MIC in the stacked cell has |dr|/|b_new| ~ 0.47 < 0.5 and
    never fires -- torn molecules survive into NVT as false "dissociated
    H"s. MIC in the melt's own cell has |dr|/|b_melt| ~ 1, correctly
    pulling wrapped atoms back onto their parent molecule.

    Returns
    -------
    slc_batch : Batch
        Single-graph Batch with the stacked geometry, combined velocities,
        zero forces/energy/stress, and the doubled-along-b cell.
    stacked_mols : list[Tensor]
        Per-molecule atom indices in the stacked frame (crystal molecules
        first, then melt molecules with indices shifted by n_crystal).
    n_crystal : int
        Number of atoms from the crystal half (first n_crystal entries
        in slc_batch correspond to the crystal half).
    """
    cell = crystal_batch.cell.squeeze()
    b_vec = cell[1, :]
    b_len = b_vec.norm()
    b_unit = b_vec / b_len

    crystal_pos = crystal_batch.positions.clone()
    melt_pos = melt_batch.positions.clone()
    melt_cell = melt_batch.cell.squeeze()

    # Discover molecule partition once per half via PBC-aware connectivity.
    # The ordering of atoms within an nvalchemi Batch built from the
    # P2_1/a supercell does NOT admit a simple stride-based molecule
    # partition -- C-C bonds span blocks that look contiguous.
    crystal_mols = _detect_naphthalene_molecules(
        crystal_pos,
        crystal_batch.atomic_numbers,
        cell,
    )
    melt_mols = _detect_naphthalene_molecules(
        melt_pos,
        melt_batch.atomic_numbers,
        melt_cell,
    )
    _unwrap_molecules_inplace(crystal_pos, cell, crystal_mols)
    _unwrap_molecules_inplace(melt_pos, melt_cell, melt_mols)
    # Fold each unwrapped molecule back into its half's own cell by integer
    # lattice shifts on the COM. Without this, a molecule whose reference
    # atom sits near frac=1 in the half's cell ends up with all atoms at
    # frac ~ 1 + eps; after stacking those eps-protrusions collide in the
    # middle vacuum gap with symmetric protrusions from the other half.
    _wrap_molecules_by_com_inplace(crystal_pos, cell, crystal_mols)
    _wrap_molecules_by_com_inplace(melt_pos, melt_cell, melt_mols)

    melt_shift = b_unit * (b_len + gap)
    slc_pos = torch.cat([crystal_pos, melt_pos + melt_shift], dim=0)
    slc_Z = torch.cat([crystal_batch.atomic_numbers, melt_batch.atomic_numbers], dim=0)
    slc_vel = torch.cat([crystal_batch.velocities, melt_batch.velocities], dim=0)

    slc_cell = cell.clone()
    slc_cell[1, :] = b_unit * (2 * b_len + 2 * gap)
    # Rewrap whole molecules (by COM) into the stacked cell so no atom sits
    # outside the PBC box. Per-atom wrap would re-tear molecules across the
    # stacked boundary; molecular COM wrap preserves whole-molecule integrity.
    n_crystal = crystal_pos.shape[0]
    stacked_mols = crystal_mols + [idx + n_crystal for idx in melt_mols]
    _wrap_molecules_by_com_inplace(slc_pos, slc_cell, stacked_mols)
    n_slc = slc_pos.shape[0]

    slc_data = AtomicData(
        positions=slc_pos,
        atomic_numbers=slc_Z,
        velocities=slc_vel,
        forces=torch.zeros(n_slc, 3, device=device),
        energy=torch.zeros(1, 1, device=device),
        stress=torch.zeros(1, 3, 3, device=device),
        cell=slc_cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    slc_data.charge = torch.zeros(1, 1, device=device)
    slc_batch = Batch.from_data_list([slc_data], device=device)

    # Geometry invariants: a, c unchanged; b doubled + 2*gap.
    assert slc_batch.num_nodes == 2 * crystal_batch.num_nodes
    slc_cl = slc_batch.cell.squeeze().norm(dim=-1)
    cryst_cl = crystal_batch.cell.squeeze().norm(dim=-1)
    assert torch.allclose(slc_cl[[0, 2]], cryst_cl[[0, 2]], atol=0.01), "a,c changed"
    expected_b = 2 * cryst_cl[1].item() + 2 * gap
    assert abs(slc_cl[1].item() - expected_b) < 0.01, (
        f"b != 2*|b| + 2*GAP (got {slc_cl[1].item():.3f}, expected {expected_b:.3f})"
    )

    return slc_batch, stacked_mols, n_crystal


def build_packmol_slc_stack(
    crystal_batch: Batch,
    monomer: Atoms,
    n_melt_molecules: int,
    *,
    device: str,
    tolerance: float = 2.0,
    nloop: int = 20,
    seed: int | None = None,
) -> Batch:
    """Construct the full SLC stack in a single Packmol invocation, with
    the crystal half held fixed as an obstacle.

    Packmol's ``tolerance`` enforces minimum inter-atom separation against
    the crystal atoms (including PBC-wrap images), so interface clashes are
    prevented at construction rather than relaxed away afterwards. No
    vacuum gap: the stacked cell is exactly (a, 2*b, c) of the crystal
    cell.

    The melt half is confined to the upper-b parallelepiped via six
    ``above plane`` / ``below plane`` constraints. Packmol's ``pbc Lx Ly Lz``
    operates over the orthorhombic bounding box of the stacked cell; we
    shift all coordinates so the bbox sits at the world origin (see the
    plan file's "Why the shift" section for the rationale), then shift
    back into the crystal's original frame on output.

    Returns a single-graph Batch with the crystal atoms first (positions
    unchanged from the warmup endpoint after unwrap + COM-wrap) and the
    packmol-placed melt atoms after.
    """
    # 1. Unwrap + COM-wrap crystal molecules in the crystal cell so any
    # WrapPeriodicHook splits from the warmup NPT endpoint are stitched
    # back together before we hand atoms to Packmol.
    crystal_pos = crystal_batch.positions.clone()
    crystal_Z = crystal_batch.atomic_numbers
    crystal_cell = crystal_batch.cell.squeeze().clone()
    crystal_mols = _detect_naphthalene_molecules(
        crystal_pos, crystal_Z, crystal_cell,
    )
    _unwrap_molecules_inplace(crystal_pos, crystal_cell, crystal_mols)
    _wrap_molecules_by_com_inplace(crystal_pos, crystal_cell, crystal_mols)

    # 2. Build the stacked cell + Packmol-frame shift.
    cell_np = crystal_cell.detach().cpu().numpy()
    a, b, c = cell_np[0], cell_np[1], cell_np[2]
    stacked_cell_np = np.stack([a, 2.0 * b, c])
    vertices = np.stack([
        np.zeros(3), a, 2.0 * b, c,
        a + 2.0 * b, a + c, 2.0 * b + c, a + 2.0 * b + c,
    ])
    bbox_lo = vertices.min(axis=0)
    bbox_hi = vertices.max(axis=0)
    shift = -bbox_lo
    pbc_box = bbox_hi - bbox_lo

    # 3. Six plane constraints carving the upper-b parallelepiped (the melt
    # half) out of the stacked-cell bbox. Origin = shift + b_vec so the
    # lower-b face sits at the crystal-half top.
    n_a = np.cross(b, c); n_a /= np.linalg.norm(n_a)
    n_b = np.cross(c, a); n_b /= np.linalg.norm(n_b)
    n_c = np.cross(a, b); n_c /= np.linalg.norm(n_c)
    margin = max(tolerance / 2.0, 0.5)
    melt_origin = shift + b
    plane_constraints = [
        ("above", n_a, float(n_a @ melt_origin) + margin),
        ("below", n_a, float(n_a @ (melt_origin + a)) - margin),
        ("above", n_b, float(n_b @ melt_origin) + margin),
        ("below", n_b, float(n_b @ (melt_origin + b)) - margin),
        ("above", n_c, float(n_c @ melt_origin) + margin),
        ("below", n_c, float(n_c @ (melt_origin + c)) - margin),
    ]

    # 4. Per-atom wrap into the STACKED cell so any unwrap-protrusion that
    # sat at e.g. y = -0.1 (just below the crystal cell's y=0 boundary)
    # lands at y = 2|b| - 0.1 (top of the stacked cell, adjacent to the
    # melt-top across the stacked-cell PBC). This puts every fixed atom
    # inside the Packmol bbox and lets Packmol's tolerance check work
    # correctly at the wrap-side interface. Molecules torn by this wrap
    # are re-stitched after Packmol returns (step 7).
    crystal_pos_np = crystal_pos.detach().cpu().numpy()
    crystal_Z_np = crystal_Z.detach().cpu().numpy()
    stacked_cell_inv = np.linalg.inv(stacked_cell_np)
    crystal_frac = crystal_pos_np @ stacked_cell_inv
    crystal_frac -= np.floor(crystal_frac)
    crystal_pos_np = crystal_frac @ stacked_cell_np
    fixed_atoms = Atoms(
        numbers=crystal_Z_np, positions=crystal_pos_np + shift,
    )

    # 5. Single Packmol run -- tolerance enforces interface separation.
    combined = pack_with_fixed_obstacle(
        monomer, n_melt_molecules, fixed_atoms,
        pbc_box=pbc_box,
        plane_constraints=plane_constraints,
        tolerance=tolerance,
        nloop=nloop,
        seed=seed,
    )

    # 6. Undo the bbox shift; combined now sits in the crystal's frame.
    combined_pos_np = combined.get_positions() - shift
    combined_Z_np = combined.get_atomic_numbers()
    n_total = len(combined_pos_np)

    # 7. Re-stitch molecules in the stacked cell. The per-atom wrap in step 4
    # may have split some crystal molecules across the stacked-cell PBC
    # (bottom-protruders wrapped to the top); molecule detection here uses
    # the stacked-cell PBC neighbour list so the split atoms are still
    # recognised as one molecule. Unwrap pulls them back together, then
    # COM-wrap folds whole molecules into the stacked cell.
    combined_pos = torch.tensor(
        combined_pos_np, dtype=torch.float32, device=device,
    )
    combined_Z = torch.tensor(combined_Z_np, dtype=torch.long, device=device)
    stacked_cell = torch.tensor(
        stacked_cell_np, dtype=torch.float32, device=device,
    )
    stacked_mols = _detect_naphthalene_molecules(
        combined_pos, combined_Z, stacked_cell,
    )
    _unwrap_molecules_inplace(combined_pos, stacked_cell, stacked_mols)
    _wrap_molecules_by_com_inplace(combined_pos, stacked_cell, stacked_mols)

    # 8. Wrap into a single-graph SLC Batch. Velocities are zero -- Stage B
    # reinitialises at per-T Maxwell-Boltzmann after multi-graph expansion.
    slc_data = AtomicData(
        positions=combined_pos,
        atomic_numbers=combined_Z,
        velocities=torch.zeros(n_total, 3, device=device),
        forces=torch.zeros(n_total, 3, device=device),
        energy=torch.zeros(1, 1, device=device),
        stress=torch.zeros(1, 3, 3, device=device),
        cell=stacked_cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    slc_data.charge = torch.zeros(1, 1, device=device)
    return Batch.from_data_list([slc_data], device=device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--material",
        default="naphthalene",
        help="Material name; must match the upstream warmup + melt "
        "invocations' --material so the seed checkpoints are found. "
        "Default 'naphthalene'.",
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
        "match the warmup and melt invocations' --d3 flag (and --run-name) "
        "so the SLC stack inherits a physics-consistent crystal/melt pair.",
    )
    p.add_argument(
        "--packmol",
        action="store_true",
        help="Generate the melt half via Packmol instead of loading a melt "
        "NVT checkpoint. Extracts one molecule from data/<material>.cif, "
        "packs N_mol_crystal copies (derived from the warmup checkpoint) "
        "into a parallelepiped matching the crystal-half cell (six "
        "above/below plane constraints + Packmol `pbc` over the orthorhombic "
        "bounding box so the in-plane boundaries get PBC neighbours). Density "
        "is identical across the interface at construction. Pairs with "
        "--source {npt,nvt,auto} (no melt checkpoint required). Pick a fresh "
        "--run-name (or accept the `packmol_melt_from_{src}` tag baked into "
        "every SLC stage name) to keep artefacts distinct from melt-NVT runs.",
    )
    p.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol minimum inter-atom distance in A (default 2.0, Packmol "
        "stock default). Smaller values pack tighter but risk overlap that "
        "FIRE2 has to relax out.",
    )
    p.add_argument(
        "--packmol-nloop",
        type=int,
        default=20,
        help="Packmol GENCAN iteration cap per molecule (default 20, Packmol "
        "stock default). Bump (e.g. 200) if Packmol exits 'ENDED WITHOUT "
        "PERFECT PACKING' at crystal density; the wrapper accepts the best-"
        "effort output regardless.",
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
        "the value passed to warmup.py / melt.py so the seed checkpoints "
        "are addressable. Tagged into the lookup as f'{int(T)}k' (e.g. "
        "'200k', '100k').",
    )
    p.add_argument(
        "--nvt-ps",
        type=float,
        default=10.0,
        help="Stage B short NVT Langevin duration per T in ps (default 10, "
        "Yoneya-Harada SLC pre-equilibration).",
    )
    p.add_argument(
        "--npt-ps",
        type=float,
        default=200.0,
        help="Stage C production anisotropic NPT duration per T in ps "
        "(default 200). Use small values (e.g. 1.5 ps = 3000 steps at "
        "dt=0.5 fs) for smoke tests; a follow-up run with a larger value "
        "triggers the extend path and runs the delta.",
    )
    p.add_argument(
        "--pressure-coupling",
        choices=["anisotropic", "isotropic"],
        default="anisotropic",
        help="Stage C NPT pressure coupling mode (default 'anisotropic'). "
        "'anisotropic' integrates each cell axis against its own diagonal "
        "stress -- the canonical choice for the SLC two-phase stack, where "
        "the interface-normal (b-axis for naphthalene's (010) interface) "
        "tracks the shifting phase fraction while in-plane axes (a, c) "
        "stay pinned by the solid's lattice. 'isotropic' couples Tr(stress) "
        "/ 3 to uniform cubic dilation -- only useful as a diagnostic for "
        "single-phase / homogeneous comparison runs. Drives the pressure "
        "tensor shape: scalar P_1atm for isotropic, rank-2 [P,P,P] for "
        "anisotropic.",
    )
    p.add_argument(
        "--barostat-time",
        type=float,
        default=4000.0,
        help="Stage C MTK barostat coupling τ_P in fs (default 2000 = 2 ps; "
        "~π·τ_P ≈ 6 ps equilibration). Larger values (e.g. 5000) decouple "
        "the barostat from the (010) interface motion and reduce cell "
        "oscillations at the cost of slower volume relaxation; match the "
        "value used for warmup NPT for consistent cell dynamics.",
    )
    p.add_argument(
        "--thermostat-time",
        type=float,
        default=100.0,
        help="Stage C NHC thermostat coupling τ_T in fs (default 100 = 0.1 ps). "
        "Sets the Nose-Hoover chain response time in the production NPT stage "
        "(FIRE and NVT stages don't use it). Match the value used for warmup "
        "NPT for consistent thermal dynamics across the warmup -> SLC chain.",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=SLC_INTERFACE_GAP,
        help=f"SLC interface vacuum gap in Å for the meltgen-melt path "
        f"(default {SLC_INTERFACE_GAP}). Absorbs molecular-unwrap protrusion "
        f"from both halves at each interface; values too small give torn/fused "
        f"molecules. Ignored when --packmol is set (the single-Packmol-run "
        f"workflow with the crystal as a fixed obstacle has no gap concept -- "
        f"tolerance is enforced by Packmol during placement). Values >= the "
        f"base model's cutoff push the gap out of the neighbor list and may "
        f"break interface forces (cutoff ~5 Å for AIMNet2 / MACE-MP-0).",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Stage A FIRE2 convergence fmax in eV/A (default 0.05; FIRE "
        "frequently plateaus higher than this on the SLC stacked "
        "landscape, so the cap usually exits via --fire-max-steps).",
    )
    p.add_argument(
        "--fire-max-steps",
        type=int,
        default=5000,
        help="Stage A FIRE2 step budget cap (default 5000, parity with "
        "warmup). FIRE exits earlier when fmax<--fmax via the convergence "
        "hook; this is the upper safety bound. Empirically FIRE plateaus "
        "at fmax ~0.5-1.0 eV/A on the SLC stacked landscape (saddles "
        "between shallow local basins) and rarely reaches the default "
        "fmax<0.05 threshold, so the cap usually controls exit. Lower it "
        "(e.g. 500) if you want Stage B Langevin to do the bulk of strain "
        "relief; use `--skip-fire` to skip Stage A entirely.",
    )
    p.add_argument(
        "--source",
        choices=["npt", "nvt", "auto"],
        default="auto",
        help="Which warmup branch feeds *both* halves of the SLC stack. "
        "'npt' uses the NPT-equilibrated warmup endpoint for the "
        "crystal half and the melt that was seeded from it (full "
        "FIRE -> NVT -> NPT pipeline). 'nvt' uses the warmup NVT "
        "endpoint for the crystal half and the melt-in-box NVT "
        "variant for the melt half -- NPT warmup is skipped entirely, "
        "so the SLC stays in the unexpanded NVT cell. 'auto' (default) "
        "tries the npt pair first, falls back to the nvt pair. The "
        "two halves must come from the same branch or their cells "
        "won't match and the stacking is geometrically broken. The "
        "chosen source is baked into every SLC stage name "
        "(from_{src}) so (source, DT) variants sit side by side.",
    )
    p.add_argument(
        "--skip-fire",
        action="store_true",
        help="Skip Stage A (FIRE2 minimisation) entirely and hand the stacked "
        "geometry directly to Stage B (NVT Langevin). FIRE on the SLC "
        "stacked landscape rarely reaches fmax<0.05 (plateau ~0.5-1.0 eV/A "
        "between shallow local basins); Stage B Langevin friction relieves "
        "any residual interface strain in ~100 fs at the per-T target. "
        "When set, the stacked batch is saved as the FIRE endpoint "
        "checkpoint so Stage B's existing load-from-checkpoint path works "
        "transparently.",
    )
    p.add_argument(
        "--temps",
        type=str,
        default=None,
        help="Comma-separated subset of TEMPS to run as one multi-graph Batch. "
        f"Default (omit flag) runs the full sweep {TEMPS} K in one process. "
        "Pass a subset (e.g. `--temps 250,300`) to split the sweep across "
        "GPUs -- launch one process per GPU with disjoint subsets and "
        "CUDA_VISIBLE_DEVICES=N pinning. Every stage artefact gains a "
        "`_t<T1>_<T2>...` suffix when a strict subset is active, so "
        "concurrent processes do not collide on checkpoints/CSVs/zarrs. "
        "Full-sweep runs keep the original unsuffixed names. The "
        "companion `slc_multi_gpu.sh` launches the canonical 2+1+1+1 "
        "split (GPU0=[250,300], GPU1=[350], GPU2=[400], GPU3=[450]).",
    )
    p.add_argument(
        "--run-name",
        default="naphthalene_long",
        help="Identifies this run's artefact subdir under logs/<run-name>/ (and "
        "assets/<run-name>/figs/ via plot scripts). Must match the warmup "
        "+ melt run-name so the seed checkpoints are found. E.g. "
        "'naphthalene_long_big', 'naphthalene_long_2025'.",
    )
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="PyTorch device. Pin a specific GPU (e.g. 'cuda:2') when running "
        "alongside other GPU jobs on the same node. The slc_multi_gpu.sh "
        "wrapper sets CUDA_VISIBLE_DEVICES per process so each sees only "
        "its own GPU and the default 'cuda' resolves correctly.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=SNAPSHOT_EVERY,
        help=f"SnapshotHook frequency in steps (default {SNAPSHOT_EVERY}). "
        "Drives how often a frame is written to the trajectory zarr at every "
        "stage (FIRE, NVT, NPT). Smaller values give finer-grained trajectories "
        "at the cost of zarr capacity + disk; for smoke tests, 10-50 is "
        "common to maintain useful resolution over short runs.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY,
        help=f"LoggingHook frequency in steps (default {LOG_EVERY}). Drives "
        "how often a row is written to the CSV log + stdout writer at every "
        "stage. Independent of --snapshot-every.",
    )
    return p.parse_args()


def print_config(
    args: argparse.Namespace,
    n_steps: int,
    n_equil: int,
    dt_tag: str,
    t_warmup_tag: str,
    melt_src: str,
    warmup_ck: str,
    melt_ck: str,
    local_temps: list[int],
) -> None:
    """Echo the resolved run configuration so CLI overrides are visible.

    Called AFTER source resolution (which may dispatch via `--source auto`)
    so the chosen warmup/melt pair is part of the echoed config.
    """
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
    full_sweep = local_temps == sorted(TEMPS)
    temps_msg = f"{local_temps} K" + ("" if full_sweep else " [strict subset]")
    print(bar)
    packmol_msg = (
        f"on (parallelepiped pack, tol={args.packmol_tolerance} A, "
        f"nloop={args.packmol_nloop})"
        if args.packmol
        else "off (melt half from checkpoint)"
    )
    if args.packmol:
        source_msg = (
            f"{melt_src} -- crystal '{warmup_ck}', melt generated via Packmol"
        )
    else:
        source_msg = (
            f"{melt_src} -- crystal '{warmup_ck}', melt '{melt_ck}'"
        )
    print(f"Run name:             {args.run_name}")
    print(f"Device:               {args.device}")
    print(f"Material:             {args.material}")
    print(f"Model alias:          {args.model}")
    print(f"Ewald:                {ewald_msg}")
    print(f"D3:                   {d3_msg}")
    print(f"Packmol melt:         {packmol_msg}")
    print(f"Timestep:             {args.dt} fs   (DT_TAG={dt_tag})")
    print(f"Warmup target T:      {args.t_warmup} K (T_TAG={t_warmup_tag})")
    print(f"Temperatures:         {temps_msg}")
    print(f"Source pair:          {source_msg}")
    print(
        f"Stage A FIRE2:        fmax<={args.fmax} eV/A, <={args.fire_max_steps} steps"
        + (" [--skip-fire]" if args.skip_fire else "")
    )
    print(f"Stage B equil NVT:    {args.nvt_ps} ps ({n_equil} steps)")
    print(f"Stage C prod NPT:     {args.npt_ps} ps ({n_steps} steps)")
    print(
        f"P target:             1 atm ({P_1ATM:.3e} eV/A^3 per diag, "
        f"{args.pressure_coupling})"
    )
    gap_msg = (
        "n/a (Packmol enforces interface tolerance directly)"
        if args.packmol
        else f"{args.gap} A"
    )
    print(f"SLC interface gap:    {gap_msg}")
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

    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    log_dir = Path("logs") / args.run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # --temps: full-sweep default, or a strict subset for multi-GPU splitting.
    # A strict subset gets a `_t<T1>_<T2>...` tag appended to every stage
    # artefact so concurrent processes don't collide. Full-sweep runs keep
    # the original unsuffixed names (backwards-compatible with existing
    # checkpoints and with melting-point-slc.ipynb's diagnostics loader).
    if args.temps is None:
        local_temps = list(TEMPS)
    else:
        try:
            local_temps = sorted(
                {int(t.strip()) for t in args.temps.split(",") if t.strip()}
            )
        except ValueError:
            raise SystemExit(
                f"--temps must be a comma-separated list of ints (got {args.temps!r})"
            )
        if not local_temps:
            raise SystemExit("--temps must list at least one temperature")
        extras = [T for T in local_temps if T not in TEMPS]
        if extras:
            raise SystemExit(f"--temps values {extras} not in full sweep {TEMPS}")
    temps_tag = (
        f"_t{'_'.join(str(T) for T in local_temps)}"
        if local_temps != sorted(TEMPS)
        else ""
    )

    # Warmup branch selection. The crystal half and the melt half must
    # come from the same warmup branch so their cells match -- `--source
    # npt` pairs after_npt_<T_WARMUP_TAG>_<DT_TAG> with
    # meltgen_...from_npt_<T_WARMUP_TAG>_<DT_TAG>, and `--source nvt`
    # pairs after_nvt_<T_WARMUP_TAG>_<DT_TAG> with the matching
    # meltgen_...from_nvt (skipping NPT warmup entirely, keeping the
    # system in the unexpanded NVT cell). Mixing halves across branches
    # gives a geometrically broken stack (atoms cram into the wrong
    # cell -> forces explode). `--source auto` tries the npt pair first
    # and falls back to the nvt pair. DT_TAG / T_WARMUP_TAG tie every
    # lookup to this SLC run's timestep + warmup target temperature, so
    # a DT or t_warmup change forces matching reruns.
    if args.source == "auto":
        source_order = ("npt", "nvt")
    else:
        source_order = (args.source,)
    melt_src = None
    warmup_ck = None
    melt_ck = None
    missing_pairs = []
    for src in source_order:
        warmup_candidate = f"{src}_{t_warmup_tag}_{dt_tag}"
        has_warmup = checkpoint_exists(warmup_candidate, log_dir)
        if args.packmol:
            # --packmol: melt half is generated, only the warmup checkpoint
            # is required to seed the crystal half.
            if has_warmup:
                melt_src = src
                warmup_ck = warmup_candidate
                melt_ck = None
                break
            missing_pairs.append((src, [f"after_{warmup_candidate}.zarr"]))
            continue
        melt_candidate = f"meltgen_nvt_500k_from_{src}_{t_warmup_tag}_{dt_tag}"
        has_melt = checkpoint_exists(melt_candidate, log_dir)
        if has_warmup and has_melt:
            melt_src = src
            warmup_ck = warmup_candidate
            melt_ck = melt_candidate
            break
        missing = []
        if not has_warmup:
            missing.append(f"after_{warmup_candidate}.zarr")
        if not has_melt:
            missing.append(f"after_{melt_candidate}.zarr")
        missing_pairs.append((src, missing))
    if melt_src is None:
        if args.packmol:
            logger.error(
                "[SLC --packmol] no warmup checkpoint under {} for "
                "T_WARMUP={}, DT={}. Missing per source: {}. "
                "Run `warmup.py --t-warmup {}` first.",
                ckpt_dir,
                t_warmup_tag,
                dt_tag,
                dict(missing_pairs),
                args.t_warmup,
            )
        elif args.source == "auto":
            logger.error(
                "[SLC] no complete (warmup, melt) pair under {} for "
                "T_WARMUP={}, DT={}. Missing per source: {}. "
                "Run `warmup.py --t-warmup {}` and "
                "`melt.py --source {{npt|nvt}} --t-warmup {}` "
                "to produce a matching pair.",
                ckpt_dir,
                t_warmup_tag,
                dt_tag,
                dict(missing_pairs),
                args.t_warmup,
                args.t_warmup,
            )
        else:
            logger.error(
                "[SLC] incomplete --source {} pair under {} for T_WARMUP={}, "
                "DT={}. Missing: {}. "
                "Run `warmup.py --t-warmup {}` (to produce "
                "after_{}_{}_{}.zarr) and `melt.py --source {} --t-warmup {}` "
                "(to produce after_meltgen_nvt_500k_from_{}_{}_{}.zarr).",
                args.source,
                ckpt_dir,
                t_warmup_tag,
                dt_tag,
                missing_pairs[0][1],
                args.t_warmup,
                args.source,
                t_warmup_tag,
                dt_tag,
                args.source,
                args.t_warmup,
                args.source,
                t_warmup_tag,
                dt_tag,
            )
        sys.exit(1)
    if args.packmol:
        logger.info(
            "[SLC --packmol] source={} -- crystal from '{}', melt generated via Packmol",
            melt_src,
            warmup_ck,
        )
    else:
        logger.info(
            "[SLC] source={} -- crystal from '{}', melt from '{}'",
            melt_src,
            warmup_ck,
            melt_ck,
        )

    n_steps = int(args.npt_ps * 1000 / args.dt)
    n_equil = int(args.nvt_ps * 1000 / args.dt)

    # Resolved -- echo the full config now.
    print_config(
        args,
        n_steps,
        n_equil,
        dt_tag,
        t_warmup_tag,
        melt_src,
        warmup_ck,
        melt_ck,
        local_temps,
    )

    # DT_TAG / T_WARMUP_TAG-tagged names for the three SLC stages. SLC is
    # downstream of both warmup and melt, so every stem carries
    # `from_{melt_src}` to record the provenance plus the warmup-T tag so
    # SLC runs seeded from differently-warmed crystals coexist. When
    # `--packmol` is set, the melt half is generated rather than loaded
    # and the `packmol_melt_` prefix discriminates from melt-NVT runs.
    # Production NPT uses `slc_all_*` as file stem (legacy, matches
    # notebook diagnostics loader) and `slc_npt_*` as checkpoint key.
    src_tag = f"packmol_melt_from_{melt_src}" if args.packmol else f"from_{melt_src}"
    fire_ck = f"slc_fire_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    fire_fs = f"slc_fire_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    equil_ck = f"slc_nvt_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    equil_fs = f"slc_nvt_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    npt_ck = f"slc_npt_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    npt_fs = f"slc_all_{src_tag}_{t_warmup_tag}_{dt_tag}{temps_tag}"

    print(
        f"SLC: {len(local_temps)} systems at {local_temps} K, FIRE2 (<={args.fire_max_steps}) "
        f"-> NVT {n_equil} -> NPT {n_steps} steps each (DT={args.dt} fs)"
    )

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
        # Need positions/cell to estimate Ewald parameters; load the warmup
        # crystal seed once here and discard. Inexpensive vs the model load.
        _seed = load_checkpoint(warmup_ck, log_dir, args.device)
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
        # 'stress' is NOT included. The `base + d3` composition path below
        # (and the use_autograd=False Ewald-pipeline path) sums each step's
        # declared outputs verbatim; without this set_config d3's dispersion
        # stress is silently dropped and the NPT barostat reads ~kinetic-only
        # pressure (the box then expands monotonically).
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
        pipe.set_config(
            "active_outputs", {"energy", "forces", "stress", "charges"}
        )
        model = pipe
    elif args.d3:
        # Match alchemi-toolkit-demo's composition pattern
        # (example/main.py:506: `model = orb_wrapper + d3_model`). The `+`
        # operator builds a PipelineModelWrapper of single-step direct
        # groups whose energies, forces, and stresses are summed element-
        # wise via sum_outputs.
        #
        # active_outputs must be set on each submodule before composition.
        # pipe.set_config('active_outputs', ...) does NOT recurse into child
        # steps -- it only updates the pipeline-level declaration. AIMNet2's
        # default active_outputs is {energy, forces, charges} (no stress)
        # and DFTD3ModelWrapper's default is {energy, forces} (no stress),
        # so without explicit per-step config sum_outputs sees zero stress
        # from both and NPT runs at ~kinetic-only pressure.
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base + d3
        model.set_config("active_outputs", {"energy", "forces", "stress"})
    else:
        base.set_config("active_outputs", {"energy", "forces", "stress"})
        model = base
    print(f"active_outputs: {sorted(model.model_config.active_outputs)}")

    _slc_t0 = time.monotonic()
    temps_tensor = torch.tensor([float(T) for T in local_temps], device=args.device)
    t_labels = [f"T={T}K" for T in local_temps]

    # --- SLC construction: stack crystal above melt along b ---------------
    # See `construct_slc_stack` for the geometry rationale (stacking-axis
    # choice, vacuum-gap sizing, per-half molecule unwrap+COM-rewrap).
    crystal_batch = load_checkpoint(warmup_ck, log_dir, args.device)
    if args.packmol:
        cif_path = Path("data") / f"{args.material}.cif"
        if not cif_path.exists():
            raise SystemExit(
                f"{cif_path} not found. --packmol needs the CIF to extract the "
                f"single-molecule template. Available materials: "
                f"{sorted(p.stem for p in Path('data').glob('*.cif'))}."
            )
        unit_cell = ase_read(str(cif_path))
        monomer = extract_single_molecule(unit_cell)
        atoms_per_mol = len(monomer)
        n_atoms_crystal = crystal_batch.num_nodes
        assert n_atoms_crystal % atoms_per_mol == 0, (
            f"crystal half has {n_atoms_crystal} atoms, not divisible by "
            f"monomer size {atoms_per_mol} ({monomer.get_chemical_formula()})"
        )
        n_mol_crystal = n_atoms_crystal // atoms_per_mol
        logger.info(
            "[SLC --packmol] monomer={} ({} atoms); packing {} copies "
            "around fixed crystal half (single Packmol run)",
            monomer.get_chemical_formula(),
            atoms_per_mol,
            n_mol_crystal,
        )
        slc_batch = build_packmol_slc_stack(
            crystal_batch,
            monomer,
            n_mol_crystal,
            device=args.device,
            tolerance=args.packmol_tolerance,
            nloop=args.packmol_nloop,
        )
    else:
        melt_batch = load_checkpoint(melt_ck, log_dir, args.device)
        # `stacked_mols` and `n_crystal` are returned for downstream verifiers
        # (see `tools/verify_slc_construction.py`); production driver only
        # needs the Batch and total atom count.
        slc_batch, _, _ = construct_slc_stack(
            crystal_batch,
            melt_batch,
            device=args.device,
            gap=args.gap,
        )
    n_slc = slc_batch.num_nodes
    slc_cl = slc_batch.cell.squeeze().norm(dim=-1)
    print(f"SLC system: {slc_batch.num_nodes} atoms (crystal half + melt half)")
    print(f"Cell lengths: {[f'{l:.2f}' for l in slc_cl.tolist()]} A")

    # --- Stage A: FIRE2 minimize (single-graph) ---------------------------
    # The stacked geometry is identical across all TEMPS at construction,
    # so a single-graph FIRE converges to the same stationary point that 5
    # copies would. Clone to multi-graph only after FIRE completes.
    if args.skip_fire and not checkpoint_exists(fire_ck, log_dir):
        logger.info(
            "[SLC FIRE] skip (--skip-fire); saving stacked geometry as FIRE endpoint"
        )
        save_checkpoint(slc_batch, fire_ck, log_dir)
        save_stage_meta(fire_ck, log_dir, 0)
    if checkpoint_exists(fire_ck, log_dir):
        logger.info("[SLC FIRE] skip (checkpoint exists); loading end-of-stage batch")
        slc_batch = load_checkpoint(fire_ck, log_dir, args.device)
    else:
        logger.info("[SLC FIRE] start  (budget: <={} steps)", args.fire_max_steps)
        _t_stage = time.monotonic()
        fire_zarr = fresh_zarr_sink(
            log_dir / f"{fire_fs}.zarr",
            capacity=args.fire_max_steps // args.snapshot_every + 10,
        )
        fire_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(log_dir / f"{fire_fs}.csv"),
            frequency=args.log_every,
        )
        fire_out = LoggingHook(
            backend="custom",
            writer_fn=stdout_writer,
            custom_scalars=DYNAMICS_SCALARS,
            frequency=args.log_every,
        )
        # tmax=0.01 caps the adaptive dt growth at 2x the initial step, so
        # FIRE can't ramp up (default tmax=0.08) and overshoot the shallow
        # basins. Paired with the shorter FIRE_MAX_STEPS, this keeps FIRE in
        # "gentle unwind" mode and lets Stage B handle residual strain.
        fire_stage = FIRE2(
            model=model,
            dt=0.005,
            tmax=0.01,
            n_steps=args.fire_max_steps,
            convergence_hook=ConvergenceHook.from_fmax(threshold=args.fmax),
        )
        for h in [
            *make_safety_hooks(model, track_stress=False),
            SnapshotHook(sink=fire_zarr, frequency=args.snapshot_every),
            fire_csv,
            fire_out,
        ]:
            fire_stage.register_hook(h)
        with fire_csv, fire_out:
            slc_batch = fire_stage.run(slc_batch)
        save_checkpoint(slc_batch, fire_ck, log_dir)
        save_stage_meta(fire_ck, log_dir, args.fire_max_steps)
        logger.info(
            "[SLC FIRE->NVT] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _slc_t0,
        )

    # Post-FIRE clash guard: FIRE's job was exactly to relieve the close
    # contacts at the interface, so the assertion belongs here rather than
    # before production NPT. Under --skip-fire we expect residual clashes
    # (Langevin friction will relieve them in Stage B), so only log.
    min_dist = min_pbc_distance(slc_batch.positions, slc_batch.cell.squeeze())
    if args.skip_fire:
        print(f"Min pairwise distance post-stack (no FIRE): {min_dist:.2f} A")
    else:
        assert min_dist > 0.5, f"Clash detected post-FIRE: min dist {min_dist:.2f} A"
        print(f"Min pairwise distance post-FIRE (PBC, all atoms): {min_dist:.2f} A")

    # --- Stage B: short NVT @ per-T (multi-graph) -------------------------
    # Bring both halves of the SLC stack to their shared target temperature
    # via Langevin NVT (one graph per T, per-graph target via tensor
    # broadcast). Multi-graph expansion happens here, not at Stage C: once
    # Stage B has checkpointed the 5-graph batch with per-T velocities,
    # Stage C loads it directly and doesn't need to clone/reinit.
    equil_meta = load_stage_meta(equil_ck, log_dir)
    equil_done = int(equil_meta["steps_completed"]) if equil_meta else 0
    equil_can_extend = checkpoint_exists(equil_ck, log_dir)

    if checkpoint_exists(equil_ck, log_dir) and equil_done >= n_equil:
        logger.info(
            "[SLC NVT] skip (checkpoint covers {} >= {} steps)",
            equil_done,
            n_equil,
        )
        slc_multi_batch = load_checkpoint(equil_ck, log_dir, args.device)
    else:
        _t_stage = time.monotonic()
        if equil_can_extend:
            n_delta = n_equil - equil_done
            logger.info(
                "[SLC NVT] extend ({} -> {} steps, +{})",
                equil_done,
                n_equil,
                n_delta,
            )
            slc_multi_batch = load_checkpoint(equil_ck, log_dir, args.device)
            part = next_part_index(log_dir, equil_fs)
        else:
            n_delta = n_equil
            logger.info(
                "[SLC NVT] start  ({} systems x {} steps)",
                len(local_temps),
                n_delta,
            )
            slc_data_list = [
                AtomicData(
                    positions=slc_batch.positions.clone(),
                    atomic_numbers=slc_batch.atomic_numbers.clone(),
                    velocities=torch.zeros_like(slc_batch.positions),
                    forces=torch.zeros(n_slc, 3, device=args.device),
                    energy=torch.zeros(1, 1, device=args.device),
                    stress=torch.zeros(1, 3, 3, device=args.device),
                    cell=slc_batch.cell.squeeze().clone().unsqueeze(0),
                    pbc=torch.tensor([[True, True, True]], device=args.device),
                )
                for _ in local_temps
            ]
            for d in slc_data_list:
                d.charge = torch.zeros(1, 1, device=args.device)
            slc_multi_batch = Batch.from_data_list(slc_data_list, device=args.device)
            initialize_velocities(
                slc_multi_batch.velocities,
                slc_multi_batch.atomic_masses,
                temperature=temps_tensor,
                batch_idx=slc_multi_batch.batch_idx,
                random_seed=42,
                remove_com=True,
                rescale=True,
            )
            part = 1

        equil_csv_path, equil_zarr_path = part_paths(log_dir, equil_fs, part)
        equil_zarr = fresh_zarr_sink(
            equil_zarr_path,
            # Multi-graph: ZarrData.write() adds num_graphs samples per
            # snapshot, so capacity must scale with the batch's graph count
            # (== len(local_temps), which may be a --temps subset).
            capacity=slc_multi_batch.num_graphs * (n_delta // args.snapshot_every) + 10,
        )
        equil_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(equil_csv_path),
            frequency=args.log_every,
        )
        equil_out = LoggingHook(
            backend="custom",
            writer_fn=make_graph_tagged_writer(t_labels),
            custom_scalars=DYNAMICS_SCALARS,
            frequency=args.log_every,
        )
        nvt_equil = NVTLangevin(
            model=model,
            dt=args.dt,
            temperature=temps_tensor,
            friction=FRICTION,
            n_steps=n_delta,
        )
        for h in [
            *make_safety_hooks(model),
            SnapshotHook(sink=equil_zarr, frequency=args.snapshot_every),
            equil_csv,
            equil_out,
        ]:
            nvt_equil.register_hook(h)
        with equil_csv, equil_out:
            slc_multi_batch = nvt_equil.run(slc_multi_batch)
        save_checkpoint(slc_multi_batch, equil_ck, log_dir)
        save_stage_meta(equil_ck, log_dir, n_equil)
        logger.info(
            "[SLC NVT->NPT] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _slc_t0,
        )

    # --- Stage C: production anisotropic NPT @ per-T ----------------------
    # Three-way dispatch mirroring the warmup NPT cell. Multi-graph Batch
    # (one SLC copy per target T); per-graph target temperature via tensor
    # broadcast; anisotropic pressure coupling (b-axis tracks the shifting
    # phase fraction across the (010) interface, a and c pinned by the
    # solid's in-plane lattice). NHC + barostat state is preserved across
    # restarts via the same _state-preload trick used for the warmup NPT.
    npt_meta = load_stage_meta(npt_ck, log_dir)
    npt_done = int(npt_meta["steps_completed"]) if npt_meta else 0
    npt_can_extend = checkpoint_exists(npt_ck, log_dir) and integrator_state_exists(
        npt_ck, log_dir
    )

    if checkpoint_exists(npt_ck, log_dir) and npt_done >= n_steps:
        logger.info(
            "[SLC NPT] skip (checkpoint covers {} >= {} steps)", npt_done, n_steps
        )
        final_batch = load_checkpoint(npt_ck, log_dir, args.device)
    else:
        _t_stage = time.monotonic()
        if npt_can_extend:
            n_delta = n_steps - npt_done
            logger.info(
                "[SLC NPT] extend ({} -> {} steps, +{})", npt_done, n_steps, n_delta
            )
            slc_multi_batch = load_checkpoint(npt_ck, log_dir, args.device)
            preloaded_state = load_integrator_state(npt_ck, log_dir, args.device)
            part = next_part_index(log_dir, npt_fs)
        else:
            n_delta = n_steps
            logger.info(
                "[SLC NPT] start  ({} systems x {} steps)", len(local_temps), n_delta
            )
            # slc_multi_batch already carries per-T velocities from Stage B.
            preloaded_state = None
            part = 1

        npt_csv_path, npt_zarr_path = part_paths(log_dir, npt_fs, part)
        # Multi-graph: ZarrData.write() adds num_graphs samples per snapshot,
        # so capacity must scale with the batch's graph count (== len(local_temps)).
        npt_zarr = fresh_zarr_sink(
            npt_zarr_path,
            capacity=slc_multi_batch.num_graphs * (n_delta // args.snapshot_every) + 10,
        )
        npt_csv = LoggingHook(
            backend="csv",
            custom_scalars=DYNAMICS_SCALARS,
            log_path=str(npt_csv_path),
            frequency=args.log_every,
        )
        npt_out = LoggingHook(
            backend="custom",
            writer_fn=make_graph_tagged_writer(t_labels),
            custom_scalars=DYNAMICS_SCALARS,
            frequency=args.log_every,
        )
        # Pressure tensor shape follows the coupling mode: rank-2 [P,P,P]
        # for anisotropic (per-axis target), scalar P_1ATM for isotropic
        # (single hydrostatic target). Mixing scalar + anisotropic silently
        # falls through to isotropic at the toolkit level, so we send the
        # documented pair.
        if args.pressure_coupling == "anisotropic":
            npt_pressure: float | torch.Tensor = torch.tensor(
                [[P_1ATM, P_1ATM, P_1ATM]], dtype=torch.float32
            )
        else:
            npt_pressure = P_1ATM
        npt_slc = NPT(
            model=model,
            dt=args.dt,
            temperature=temps_tensor,
            pressure=npt_pressure,
            barostat_time=args.barostat_time,
            thermostat_time=args.thermostat_time,
            pressure_coupling=args.pressure_coupling,
            n_steps=n_delta,
        )
        for h in [
            *make_safety_hooks(model),
            SnapshotHook(sink=npt_zarr, frequency=args.snapshot_every),
            npt_csv,
            npt_out,
        ]:
            npt_slc.register_hook(h)
        if preloaded_state is not None:
            npt_slc._state = preloaded_state
        with npt_csv, npt_out:
            final_batch = npt_slc.run(slc_multi_batch)
        save_checkpoint(final_batch, npt_ck, log_dir)
        save_integrator_state(npt_slc._state, npt_ck, log_dir)
        save_stage_meta(npt_ck, log_dir, n_steps)
        logger.info(
            "[SLC NPT->done] stage={:.2f}s  elapsed={:.2f}s",
            time.monotonic() - _t_stage,
            time.monotonic() - _slc_t0,
        )

    final_densities = compute_density(final_batch)
    for i, T in enumerate(local_temps):
        print(f"T={T}K done, density={final_densities[i]:.3f} g/cm3")
    print(f"\nAll {len(local_temps)} temperature points complete")
    print(f"SLC script complete - checkpoints under {ckpt_dir}")


if __name__ == "__main__":
    main()
