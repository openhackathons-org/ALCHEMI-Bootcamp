"""Animate per-phase COM MSD with live Einstein D fits.

Companion to ``render_slc_phase.py``. Splits molecules into the
crystal half (first n_mol/2 molecules in the stacked SLC cell) and
the melt half (last n_mol/2) and animates the cross-molecule mean
COM MSD per phase, with a live Einstein-relation fit on the
trailing ``--fit-frac`` of each curve. Same default cadence as the
phase render (``--every-nth 1``, ``--fps 20``) so they stay in sync
side-by-side.

MSD math goes through ``helpers.diffusion.compute_com_msd_numpy``:
fractional-coord accumulation removes affine cell deformation and
the system-COM subtraction kills lab-frame rigid drift -- both
needed under anisotropic NPT. The two-phase mean curves and fits
are derived from the per-molecule MSD by indexing the first vs
second half of the molecule list.

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python animate_slc_phase_msd.py \\
        --input assets/<run>/traj/<traj>.extxyz
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from matplotlib.animation import FFMpegWriter

import _path  # noqa: F401  # parent dir on sys.path for `helpers` import
from analyze_s0 import find_molecules
from helpers.diffusion import compute_com_msd_numpy, fit_diffusion_coefficient

CRYSTAL_COLOR = "#1f77b4"
MELT_COLOR = "#ff7f0e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Output MP4. Default: <traj.parent.parent>/figs/<stem>_phase_msd.mp4")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--every-nth", type=int, default=1,
                   help="Render every Nth frame (default 1; match the paired phase render).")
    p.add_argument("--fit-frac", type=float, default=0.5,
                   help="Trailing fraction of frames for the Einstein-relation D fit.")
    p.add_argument("--snapshot-every", type=int, default=100,
                   help="MD steps between trajectory snapshots (matches SnapshotHook).")
    p.add_argument("--dt", type=float, default=0.5, help="MD timestep in fs.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_phase_msd.mp4"
    )

    print(f"Loading {args.input.name} (stride={args.stride}) ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_frames = len(frames)
    print(f"  -> {n_frames} frames, {len(frames[0])} atoms ({time.monotonic() - t0:.1f}s)")

    print("Discovering molecules ...")
    t0 = time.monotonic()
    molecules = find_molecules(frames[0])
    n_mol = len(molecules)
    sizes_set = {len(m) for m in molecules}
    if len(sizes_set) != 1:
        raise SystemExit(f"non-uniform mol sizes: {sorted(sizes_set)}")
    atoms_per_mol = next(iter(sizes_set))
    if n_mol % 2 != 0:
        raise SystemExit(
            f"SLC cell must have even n_mol (got {n_mol}); crystal+melt halves must match."
        )
    half = n_mol // 2

    pos0_raw = np.asarray(frames[0].positions)
    crystal_y = np.mean([pos0_raw[idx, 1].mean() for idx in molecules[:half]])
    melt_y = np.mean([pos0_raw[idx, 1].mean() for idx in molecules[half:]])
    print(
        f"  -> {n_mol} molecules; mean b at frame 0: "
        f"crystal={crystal_y:.2f} Å  melt={melt_y:.2f} Å ({time.monotonic() - t0:.1f}s)"
    )
    if crystal_y > melt_y:
        print("  !! first-half molecules sit above second-half — labels may be flipped")

    time_per_frame_ps = args.snapshot_every * args.stride * args.dt / 1000.0
    time_ps = np.arange(n_frames) * time_per_frame_ps

    print("Computing per-molecule COM MSD ...")
    t0 = time.monotonic()
    flat_idx = np.concatenate(molecules)
    masses_arr = frames[0].get_masses()[flat_idx].astype(np.float64)
    positions_seq = [a.positions[flat_idx].astype(np.float64) for a in frames]
    cells_seq = [np.asarray(a.cell, dtype=np.float64) for a in frames]
    msd_per_mol = compute_com_msd_numpy(
        positions_seq, cells_seq, masses_arr, atoms_per_mol
    )  # [n_frames - 1, n_mol]
    msd_time_ps = time_ps[1:]
    msd_crystal = msd_per_mol[:, :half].mean(axis=1)
    msd_melt = msd_per_mol[:, half:].mean(axis=1)

    d_crystal = fit_diffusion_coefficient(
        msd_per_mol[:, :half], msd_time_ps, fit_frac=args.fit_frac
    )
    d_melt = fit_diffusion_coefficient(
        msd_per_mol[:, half:], msd_time_ps, fit_frac=args.fit_frac
    )
    print(
        f"  -> final D: crystal = {d_crystal['D_cm2_per_s']:.2e} cm²/s, "
        f"melt = {d_melt['D_cm2_per_s']:.2e} cm²/s "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # --- Figure ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi
    )

    crystal_line, = ax.plot(
        [], [], color=CRYSTAL_COLOR, lw=1.8, alpha=0.95,
        label=f"crystal half ⟨MSD⟩  D = {d_crystal['D_cm2_per_s']:+.2e} cm²/s",
    )
    melt_line, = ax.plot(
        [], [], color=MELT_COLOR, lw=1.8, alpha=0.95,
        label=f"melt half ⟨MSD⟩  D = {d_melt['D_cm2_per_s']:+.2e} cm²/s",
    )
    crystal_fit, = ax.plot(
        [], [], color=CRYSTAL_COLOR, ls="--", lw=1.4, alpha=0.85,
        label=f"Einstein fit (trailing {args.fit_frac:.0%})",
    )
    melt_fit, = ax.plot(
        [], [], color=MELT_COLOR, ls="--", lw=1.4, alpha=0.85,
    )
    fit_band = ax.axvspan(
        0, 0, color="grey", alpha=0.12, label=f"D fit window (last {args.fit_frac:.0%})"
    )
    d_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=10,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.85),
    )

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"⟨COM MSD⟩  ($\mathrm{\AA}^2$)")
    ax.set_title(
        f"{args.input.stem}\n"
        f"{n_frames} frames × stride {args.stride} = {time_ps[-1]:.1f} ps, "
        f"{half} crystal + {half} melt molecules",
        fontsize=10,
    )
    ax.set_xlim(0, time_ps[-1])
    msd_pad = max(msd_crystal.max(), msd_melt.max(), 1e-6) * 1.10
    ax.set_ylim(0, msd_pad)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    # --- Animate ---------------------------------------------------------
    n_render = (n_frames + args.every_nth - 1) // args.every_nth
    print(f"Rendering {n_render} frames @ {args.fps} fps -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    t0 = time.monotonic()
    min_fit_msd = 4
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(0, n_frames, args.every_nth):
            n_msd = i  # MSD samples up to (and including) frame index i-1
            if n_msd >= 1:
                crystal_line.set_data(msd_time_ps[:n_msd], msd_crystal[:n_msd])
                melt_line.set_data(msd_time_ps[:n_msd], msd_melt[:n_msd])
            if n_msd >= min_fit_msd:
                d_c_live = fit_diffusion_coefficient(
                    msd_per_mol[:n_msd, :half], msd_time_ps[:n_msd], fit_frac=args.fit_frac,
                )
                d_m_live = fit_diffusion_coefficient(
                    msd_per_mol[:n_msd, half:], msd_time_ps[:n_msd], fit_frac=args.fit_frac,
                )
                fit_xs = msd_time_ps[d_c_live["fit_start_idx"] : n_msd]
                crystal_fit.set_data(
                    fit_xs, d_c_live["slope"] * fit_xs + d_c_live["intercept"]
                )
                melt_fit.set_data(
                    fit_xs, d_m_live["slope"] * fit_xs + d_m_live["intercept"]
                )
                band_t0 = float(msd_time_ps[d_c_live["fit_start_idx"]])
                band_t1 = float(msd_time_ps[n_msd - 1])
                fit_band.set_x(band_t0)
                fit_band.set_width(band_t1 - band_t0)
                d_text.set_text(
                    f"live D(t)\n"
                    f"  crystal = {d_c_live['D_cm2_per_s']:+.2e} cm²/s\n"
                    f"  melt    = {d_m_live['D_cm2_per_s']:+.2e} cm²/s\n"
                    f"final D\n"
                    f"  crystal = {d_crystal['D_cm2_per_s']:+.2e} cm²/s\n"
                    f"  melt    = {d_melt['D_cm2_per_s']:+.2e} cm²/s"
                )
            writer.grab_frame()
    plt.close(fig)
    elapsed = time.monotonic() - t0
    print(
        f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{elapsed:.1f}s, {n_render / max(elapsed, 1e-3):.1f} fps render rate)"
    )


if __name__ == "__main__":
    main()
