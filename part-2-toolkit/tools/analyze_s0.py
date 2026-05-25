"""Plot rotational ACF C(t) and translational COM-MSD vs MD time.

Two stacked panels per the Yoneya & Harada classifier:
  - Top: ``C_k(t) = <P_2(v_k(t) . v_k(0))>_mol`` for k = 0, 1, 2 (long, short,
    plane-normal). Sign-flip invariant via P_2. Its tail = S_0 (rotational
    order parameter).
  - Bottom: cross-molecule averaged COM MSD with an Einstein-relation linear
    fit on the trailing ``--fit-frac`` window. Reports D in cm^2/s.

Phase read:
  - D ~ 0, S_0 -> 1:  ordered crystal
  - D ~ 0, 0 < S_0 < 1:  plastic crystal (hindered rotation)
  - D >> 0, S_0 -> 0:  liquid

Atom-ordering note: the supercell extxyz has unit cells laid down sequentially
(20 C + 16 H per unit cell × 100 unit cells = 3600 atoms), so naive
"every 18 atoms = one molecule" slicing is wrong. We discover molecules via
PBC-aware connectivity on frame 0 (same approach as
visualize_warmup_trajectory.py) and reuse those indices for all frames. For
the MSD pipeline (which expects contiguous ``atoms_per_mol`` blocks), we
permute atoms once via the discovered molecule indices.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from ase.neighborlist import primitive_neighbor_list

import _path  # noqa: F401  # parent dir on sys.path for `helpers` import
from helpers.diffusion import compute_com_msd_numpy, fit_diffusion_coefficient

ROLLING_WINDOW = 20  # matches tools/plot_warmup.py


def _rolling_mean(y: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Centered rolling mean; NaN-padded edges (mirrors pandas center=True)."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size < window:
        return y.copy()
    valid = np.convolve(y, np.ones(window) / window, mode="valid")
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.concatenate(
        [np.full(pad_left, np.nan), valid, np.full(pad_right, np.nan)]
    )


def find_molecules(
    atoms, c_c_cutoff: float = 1.6, c_h_cutoff: float = 1.4
) -> list[np.ndarray]:
    """Connected components under PBC using element-pair-aware bond cutoffs.

    Mirrors ``slc._detect_naphthalene_molecules``:
      * C-C bond if d < c_c_cutoff (default 1.6 Å; aromatic ~1.40)
      * C-H bond if d < c_h_cutoff (default 1.4 Å; aromatic ~1.09, stretched
        up to ~1.35 at 500 K)
      * H-H pairs never bond
    A single isotropic cutoff (e.g. 1.55 Å) cannot simultaneously (a)
    catch stretched C-H bonds in the melt and (b) reject close
    intermolecular H-H / C-C contacts at high T; element-pair filtering
    sidesteps both failure modes.
    """
    cutoff = max(c_c_cutoff, c_h_cutoff)
    i, j, d = primitive_neighbor_list(
        "ijd",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=cutoff,
    )
    n = len(atoms)
    Z = atoms.get_atomic_numbers()
    adjacency = [[] for _ in range(n)]
    for a, b, dist in zip(i, j, d):
        za, zb = int(Z[a]), int(Z[b])
        if za == 6 and zb == 6 and dist < c_c_cutoff:
            adjacency[a].append(b)
        elif {za, zb} == {1, 6} and dist < c_h_cutoff:
            adjacency[a].append(b)
        # H-H pairs never bond
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        queue = [start]
        labels[start] = next_label
        while queue:
            u = queue.pop()
            for v in adjacency[u]:
                if labels[v] == -1:
                    labels[v] = next_label
                    queue.append(v)
        next_label += 1
    return [np.where(labels == k)[0] for k in range(next_label)]


def molecular_principal_axes(atoms, molecule_idxs: list[np.ndarray]) -> np.ndarray:
    """Per-molecule mass-weighted principal-inertia axes, sorted by eigenvalue
    ascending (long, short, normal for planar molecules). Unwraps each molecule
    via MIC relative to its first atom before forming the inertia tensor.
    Returns shape ``[n_mol, 3, 3]`` with ``axes[m, k, :]`` = axis k for mol m.
    """
    pos = atoms.positions
    cell = np.asarray(atoms.cell)
    cell_inv = np.linalg.inv(cell)
    masses = atoms.get_masses()
    axes = np.zeros((len(molecule_idxs), 3, 3))
    for k, idx in enumerate(molecule_idxs):
        mol_pos = pos[idx]
        m = masses[idx]
        # Unwrap molecule across PBC relative to its first atom (MIC).
        ref = mol_pos[0]
        dr = mol_pos - ref
        dr_frac = dr @ cell_inv
        dr_frac -= np.round(dr_frac)
        mol_unwrapped = ref + dr_frac @ cell
        # Mass-weighted COM and centered positions.
        com = (m[:, None] * mol_unwrapped).sum(axis=0) / m.sum()
        r = mol_unwrapped - com
        # Inertia tensor: I = Σ m_i (|r_i|² I_3 - r_i r_iᵀ).
        r2 = (r * r).sum(axis=1)
        inertia = (m * r2).sum() * np.eye(3) - np.einsum("i,ij,ik->jk", m, r, r)
        eigvals, eigvecs = np.linalg.eigh(inertia)
        order = np.argsort(eigvals)
        axes[k] = eigvecs[:, order].T  # row k = axis k
    return axes


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-name",
        default="naphthalene_long_2025",
        help="Run subdirectory under assets/. Default naphthalene_long_2025.",
    )
    p.add_argument(
        "--workflow",
        choices=["warmup", "melt", "slc"],
        default="warmup",
        help="Which pipeline stage's trajectory to analyse. ``warmup`` uses "
        "warmup_<stage>_<T_WARMUP>k_<DT>; ``melt`` uses "
        "melt_nvt_<MELT_T>k_from_<SOURCE>_<T_WARMUP>k_<DT> and ignores "
        "--stage (melt is always NVT); ``slc`` uses "
        "slc_all_from_<SOURCE>_<T_WARMUP>k_<DT>_t<SLC_T> and reads the "
        "SLC NPT production trajectory at one target T (set via --slc-t).",
    )
    p.add_argument(
        "--stage",
        choices=["nvt", "npt"],
        default="npt",
        help="Warmup stage to analyse (only used when --workflow=warmup).",
    )
    p.add_argument(
        "--source",
        choices=["nvt", "npt"],
        default="npt",
        help="Warmup endpoint the melt was seeded from (only used when "
        "--workflow=melt). Must match the --source passed to "
        "melt.py.",
    )
    p.add_argument(
        "--melt-t",
        type=float,
        default=500.0,
        help="Melt NVT target temperature in K (default 500). Only used "
        "when --workflow=melt.",
    )
    p.add_argument(
        "--slc-t",
        type=int,
        default=None,
        help="SLC NPT production target T in K (e.g. 350). Required when "
        "--workflow=slc; selects the per-T multi-GPU shard.",
    )
    p.add_argument(
        "--slc-stage",
        choices=["nvt", "npt"],
        default="npt",
        help="Which SLC sub-stage trajectory to analyse (only used when "
        "--workflow=slc). ``npt`` reads the NPT production trajectory "
        "(file stem ``slc_all_*``); ``nvt`` reads the per-T NVT "
        "equilibration trajectory (``slc_nvt_*``).",
    )
    p.add_argument(
        "--dt", type=float, default=0.5, help="MD timestep in fs (default 0.5)."
    )
    p.add_argument(
        "--t-warmup",
        type=float,
        default=200.0,
        help="Warmup target temperature in K (default 200). Must "
        "match the value the warmup driver was run with so the "
        "right extxyz trajectory is found.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=100,
        help="Steps between snapshots (matches warmup SNAPSHOT_EVERY=100).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for analysis (default 1 = every frame).",
    )
    p.add_argument(
        "--tail-frac",
        type=float,
        default=0.2,
        help="Trailing fraction of frames averaged into S_0 (default 0.2).",
    )
    p.add_argument(
        "--fit-frac",
        type=float,
        default=0.5,
        help="Trailing fraction of MSD curve used for the Einstein-relation "
        "linear fit (default 0.5; skips the ballistic / sub-diffusive head).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG. Default depends on --workflow: "
        "warmup -> s0_<stage>_<T_WARMUP_TAG>_<DT_TAG>.png; "
        "melt -> s0_melt_nvt_<MELT_T>k_from_<SOURCE>_<T_WARMUP_TAG>_<DT_TAG>.png. "
        "All paths are under assets/<run-name>/figs/.",
    )
    p.add_argument(
        "--packmol",
        action="store_true",
        help="The SLC stack was generated via `slc.py --packmol`. Adds the "
        "`packmol_melt_` infix to the SLC extxyz / output basenames so "
        "this matches `slc.py`'s artefact naming. Only meaningful when "
        "--workflow=slc.",
    )
    args = p.parse_args()

    HERE = Path(__file__).resolve().parent.parent  # part-2-toolkit/ root
    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    if args.workflow == "warmup":
        traj_stage = args.stage
        extxyz_basename = f"warmup_{traj_stage}_{t_warmup_tag}_{dt_tag}"
        out_basename = f"s0_{traj_stage}_{t_warmup_tag}_{dt_tag}"
        title_workflow = f"warmup {traj_stage.upper()}"
    elif args.workflow == "melt":
        traj_stage = "nvt"  # melt is always NVT in this pipeline
        melt_t_tag = f"{int(args.melt_t)}k"
        extxyz_basename = (
            f"melt_nvt_{melt_t_tag}_from_{args.source}_{t_warmup_tag}_{dt_tag}"
        )
        out_basename = (
            f"s0_melt_nvt_{melt_t_tag}_from_{args.source}_{t_warmup_tag}_{dt_tag}"
        )
        title_workflow = (
            f"melt NVT {int(args.melt_t)} K from {args.source.upper()} "
            f"{int(args.t_warmup)} K endpoint"
        )
    else:  # slc
        if args.slc_t is None:
            raise SystemExit("--workflow=slc requires --slc-t (e.g. 350)")
        slc_t_tag = f"t{args.slc_t}"
        # NPT production traj is written under the legacy `slc_all_*` stem;
        # NVT equilibration traj uses `slc_nvt_*`. Both are per-T shards.
        traj_stem = "slc_all" if args.slc_stage == "npt" else "slc_nvt"
        src_infix = (
            f"packmol_melt_from_{args.source}" if args.packmol
            else f"from_{args.source}"
        )
        extxyz_basename = (
            f"{traj_stem}_{src_infix}_{t_warmup_tag}_{dt_tag}_{slc_t_tag}"
        )
        out_basename = (
            f"s0_slc_{args.slc_stage}_{args.slc_t}k_{src_infix}_"
            f"{t_warmup_tag}_{dt_tag}"
        )
        title_workflow = (
            f"SLC {args.slc_stage.upper()} {args.slc_t} K from "
            f"{args.source.upper()} {int(args.t_warmup)} K endpoint"
            + (" (packmol melt)" if args.packmol else "")
        )

    extxyz_path = HERE / "assets" / args.run_name / "traj" / f"{extxyz_basename}.extxyz"
    if not extxyz_path.exists():
        gz_path = extxyz_path.with_suffix(".extxyz.gz")
        if gz_path.exists():
            extxyz_path = gz_path
        else:
            raise SystemExit(f"Not found: {extxyz_path} (or .gz variant)")

    out = args.out or (HERE / "assets" / args.run_name / "figs" / f"{out_basename}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {extxyz_path} (stride={args.stride}) ...")
    t0 = time.monotonic()
    frames = ase_read(str(extxyz_path), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    print(
        f"  -> {len(frames)} frames, {len(frames[0])} atoms each "
        f"({time.monotonic() - t0:.1f}s)"
    )

    print("Discovering molecules on frame 0 via PBC-aware connectivity ...")
    t0 = time.monotonic()
    raw_molecules = find_molecules(frames[0])
    raw_sizes = {len(m) for m in raw_molecules}
    print(
        f"  -> {len(raw_molecules)} raw groups, atoms/mol: {sorted(raw_sizes)} "
        f"({time.monotonic() - t0:.1f}s)"
    )
    # Filter to size-18 (C10H8) molecules. Merged-cluster groups (36, 54, ...)
    # arise where intermolecular C-C contacts dip below the bond cutoff -- a
    # signature of unrelaxed SLC interface geometry; their inertia tensors
    # don't represent a single molecule's orientation, so excluding them
    # gives a cleaner S_0 and lets the MSD pipeline run on contiguous blocks.
    naphthalene_size = 18
    molecules = [m for m in raw_molecules if len(m) == naphthalene_size]
    if not molecules:
        raise SystemExit(
            f"No size-{naphthalene_size} molecules discovered. Raw sizes: "
            f"{sorted(raw_sizes)}"
        )
    if len(molecules) != len(raw_molecules):
        n_dropped = len(raw_molecules) - len(molecules)
        n_atoms_dropped = sum(len(m) for m in raw_molecules) - len(molecules) * naphthalene_size
        print(
            f"  NOTE: filtered to {len(molecules)} canonical "
            f"{naphthalene_size}-atom molecules; dropped {n_dropped} "
            f"merged-cluster groups ({n_atoms_dropped} atoms)"
        )
    sizes = {naphthalene_size}

    print(f"Computing principal axes for {len(frames)} frames ...")
    t0 = time.monotonic()
    axes_seq = np.zeros((len(frames), len(molecules), 3, 3))
    for i, atoms in enumerate(frames):
        axes_seq[i] = molecular_principal_axes(atoms, molecules)
        if (i + 1) % 200 == 0:
            print(f"  frame {i + 1}/{len(frames)} ({time.monotonic() - t0:.1f}s)")
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    # C_k(t) = <P_2(v_k(t) · v_k(0))>_mol
    ref = axes_seq[0]  # [n_mol, 3, 3]
    n_frames = axes_seq.shape[0]
    acf = np.zeros((n_frames, 3))
    for t in range(n_frames):
        dot = (axes_seq[t] * ref).sum(axis=-1)  # [n_mol, 3]
        acf[t] = (0.5 * (3.0 * dot * dot - 1.0)).mean(axis=0)

    # Time axis in ps. snapshot_every accounts for the fact that frames are sampled
    # every N MD steps; stride further subsamples those frames.
    time_per_frame_ps = args.snapshot_every * args.stride * args.dt / 1000.0
    time_ps = np.arange(n_frames) * time_per_frame_ps

    # Tail-averaged S_0.
    n_tail = max(1, int(n_frames * args.tail_frac))
    s0_per_axis = acf[-n_tail:].mean(axis=0)
    s0 = float(s0_per_axis.mean())
    print(
        f"\nS_0 (tail-averaged over last {n_tail} frames "
        f"= {n_tail * time_per_frame_ps:.2f} ps):"
    )
    print(f"  axis 0 (long, in-plane):  {s0_per_axis[0]:+.3f}")
    print(f"  axis 1 (short, in-plane): {s0_per_axis[1]:+.3f}")
    print(f"  axis 2 (normal to plane): {s0_per_axis[2]:+.3f}")
    print(f"  mean S_0:                 {s0:+.3f}")

    # --- COM-MSD + Einstein-relation D --------------------------------------
    # `molecules` was filtered to canonical size-18 (C10H8) above, so this
    # path just consumes that list as contiguous atoms_per_mol blocks.
    atoms_per_mol = naphthalene_size
    flat_idx = np.concatenate(molecules)

    print("\nBuilding numpy snapshots/cells/masses for COM-MSD ...")
    t0 = time.monotonic()
    masses_arr = frames[0].get_masses()[flat_idx].astype(np.float64)
    positions_seq = [a.positions[flat_idx].astype(np.float64) for a in frames]
    cells_seq = [np.asarray(a.cell, dtype=np.float64) for a in frames]
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    print("Computing molecular COM MSD ...")
    t0 = time.monotonic()
    msd_per_mol = compute_com_msd_numpy(
        positions_seq, cells_seq, masses_arr, atoms_per_mol
    )
    msd_time_ps = time_ps[1:]  # compute_com_msd_numpy returns n_frames-1 entries
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    d = fit_diffusion_coefficient(msd_per_mol, msd_time_ps, fit_frac=args.fit_frac)
    n_fit = msd_per_mol.shape[0] - d["fit_start_idx"]
    fit_t0 = msd_time_ps[d["fit_start_idx"]]
    fit_t1 = msd_time_ps[-1]
    print(
        f"\nDiffusion coefficient (3D Einstein, slope/6 over last "
        f"{args.fit_frac:.0%} = {n_fit} frames, t in [{fit_t0:.2f}, "
        f"{fit_t1:.2f}] ps):"
    )
    print(f"  D = {d['D_A2_per_ps']:.3e} A^2/ps")
    print(f"    = {d['D_cm2_per_s']:.3e} cm^2/s")

    # --- Plot ---------------------------------------------------------------
    fig, (ax_acf, ax_msd) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    # Top: rotational ACF (existing logic) ----------------------------------
    axis_labels = (
        "axis 0 (long, in-plane)",
        "axis 1 (short, in-plane)",
        "axis 2 (normal to plane)",
    )
    colors = ("#c62828", "#2e7d32", "#1565c0")
    for k in range(3):
        ax_acf.plot(
            time_ps,
            acf[:, k],
            color=colors[k],
            ls="--",
            lw=0.7,
            alpha=0.25,
        )
        ax_acf.plot(
            time_ps,
            _rolling_mean(acf[:, k]),
            color=colors[k],
            lw=1.8,
            label=f"{axis_labels[k]}: S_0={s0_per_axis[k]:+.3f}",
        )
    ax_acf.axhline(0, color="black", ls="--", lw=0.6, alpha=0.6)
    ax_acf.axhline(1, color="black", ls=":", lw=0.6, alpha=0.4)
    if n_tail >= 1:
        ax_acf.axvspan(
            time_ps[-n_tail],
            time_ps[-1],
            color="grey",
            alpha=0.12,
            label=f"S_0 tail window (last {args.tail_frac:.0%})",
        )
    ax_acf.set_ylabel(
        r"$C_k(t) = \langle P_2(\hat v_k(t)\cdot\hat v_k(0)) \rangle_\mathrm{mol}$"
    )
    ax_acf.set_title(
        f"Rotational ACF + COM MSD — {args.run_name}, {title_workflow} "
        f"(DT = {args.dt} fs)\n{n_frames} frames × stride {args.stride} = "
        f"{time_ps[-1]:.1f} ps, {len(molecules)} molecules; "
        f"mean S_0 = {s0:+.3f}, D = {d['D_cm2_per_s']:.2e} cm$^2$/s"
    )
    ax_acf.set_ylim(-0.25, 1.05)
    ax_acf.grid(alpha=0.3)
    ax_acf.legend(loc="best", fontsize=9, framealpha=0.9)

    # Bottom: COM MSD + Einstein fit ----------------------------------------
    ax_msd.plot(
        msd_time_ps,
        d["msd_mean"],
        color="#37474f",
        ls="--",
        lw=0.7,
        alpha=0.25,
    )
    ax_msd.plot(
        msd_time_ps,
        _rolling_mean(d["msd_mean"]),
        color="#37474f",
        lw=1.8,
        label=f"COM MSD; D = {d['D_cm2_per_s']:.2e} cm$^2$/s",
    )
    fit_line = d["slope"] * msd_time_ps[d["fit_start_idx"] :] + d["intercept"]
    ax_msd.plot(
        msd_time_ps[d["fit_start_idx"] :],
        fit_line,
        color="#c62828",
        ls="--",
        lw=1.4,
        alpha=0.9,
        label=f"Einstein fit (slope/6 = D); slope = {d['slope']:.3e} A$^2$/ps",
    )
    ax_msd.axvspan(
        fit_t0,
        fit_t1,
        color="grey",
        alpha=0.12,
        label=f"D fit window (last {args.fit_frac:.0%})",
    )
    ax_msd.set_xlabel("Time (ps)")
    ax_msd.set_ylabel(r"COM MSD ($\mathrm{\AA}^2$)")
    ax_msd.grid(alpha=0.3)
    ax_msd.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
