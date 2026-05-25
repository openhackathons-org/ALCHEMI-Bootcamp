"""Animate the rotational ACF + COM MSD figure (matplotlib).

Companion to ``render_warmup_s0.py``. The trajectory video and this
metrics animation are rendered with identical frame cadence + fps so
they play in sync side-by-side: at video time τ the trajectory shows
the supercell at MD time t and this animation shows the ACF / MSD
curves drawn from 0 → t.

Plot is the same 2-panel figure ``analyze_s0.py`` produces (top:
``C_k(t) = <P_2(v_k(t)·v_k(0))>_mol`` for the 3 inertia axes, with the
S0 tail window shaded; bottom: per-mol-mean COM MSD with the Einstein
fit + fit window). Final S0 / D values are pre-computed from the full
trajectory and shown in the title throughout (so they don't pop in at
the end). Curves grow with simulation time.

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python animate_s0_metrics.py \\
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
from analyze_s0 import find_molecules, molecular_principal_axes
from helpers.diffusion import compute_com_msd_numpy, fit_diffusion_coefficient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--every-nth", type=int, default=1,
                   help="Render every Nth frame (default 1; match the paired trajectory render).")
    p.add_argument("--ref-idx", type=int, default=0)
    p.add_argument("--tail-frac", type=float, default=0.2)
    p.add_argument("--fit-frac", type=float, default=0.5,
                   help="Trailing fraction of frames for the Einstein-relation D fit.")
    p.add_argument("--snapshot-every", type=int, default=100,
                   help="MD steps between trajectory snapshots (matches SnapshotHook).")
    p.add_argument("--dt", type=float, default=0.5, help="MD timestep in fs.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_s0_metrics.mp4"
    )

    print(f"Loading {args.input.name} (stride={args.stride}) ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_frames = len(frames)
    print(f"  -> {n_frames} frames, {len(frames[0])} atoms ({time.monotonic() - t0:.1f}s)")

    print("Discovering molecules + computing axes ...")
    t0 = time.monotonic()
    molecules = find_molecules(frames[0])
    n_mol = len(molecules)
    sizes = {len(m) for m in molecules}
    if len(sizes) != 1:
        raise SystemExit(f"non-uniform mol sizes: {sorted(sizes)}")
    atoms_per_mol = next(iter(sizes))

    axes_seq = np.zeros((n_frames, n_mol, 3, 3))
    for i, atoms in enumerate(frames):
        axes_seq[i] = molecular_principal_axes(atoms, molecules)

    ref = axes_seq[args.ref_idx]
    acf = np.zeros((n_frames, 3))
    for t in range(n_frames):
        dot = (axes_seq[t] * ref).sum(axis=-1)  # [n_mol, 3]
        acf[t] = (0.5 * (3.0 * dot * dot - 1.0)).mean(axis=0)
    print(f"  -> {n_mol} molecules ({time.monotonic() - t0:.1f}s)")

    n_tail = max(1, int(n_frames * args.tail_frac))
    s0_per_axis = acf[-n_tail:].mean(axis=0)
    s0 = float(s0_per_axis.mean())

    time_per_frame_ps = args.snapshot_every * args.stride * args.dt / 1000.0
    time_ps = np.arange(n_frames) * time_per_frame_ps

    print("Computing COM-MSD + Einstein D ...")
    t0 = time.monotonic()
    flat_idx = np.concatenate(molecules)
    masses_arr = frames[0].get_masses()[flat_idx].astype(np.float64)
    positions_seq = [a.positions[flat_idx].astype(np.float64) for a in frames]
    cells_seq = [np.asarray(a.cell, dtype=np.float64) for a in frames]
    msd_per_mol = compute_com_msd_numpy(positions_seq, cells_seq, masses_arr, atoms_per_mol)
    msd_time_ps = time_ps[1:]
    msd_mean = msd_per_mol.mean(axis=1)
    d = fit_diffusion_coefficient(msd_per_mol, msd_time_ps, fit_frac=args.fit_frac)
    print(
        f"  -> mean S0 = {s0:+.3f}, D = {d['D_cm2_per_s']:.2e} cm^2/s "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # --- Figure ----------------------------------------------------------
    fig, (ax_acf, ax_msd) = plt.subplots(
        2, 1,
        figsize=(args.width / args.dpi, args.height / args.dpi),
        sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}, dpi=args.dpi,
    )

    axis_labels = (
        "axis 0 (long, in-plane)",
        "axis 1 (short, in-plane)",
        "axis 2 (normal to plane)",
    )
    colors = ("#c62828", "#2e7d32", "#1565c0")

    acf_lines = []
    for k in range(3):
        ln, = ax_acf.plot(
            [], [], color=colors[k], lw=1.6, alpha=0.9,
            label=f"{axis_labels[k]}: S$_0$={s0_per_axis[k]:+.3f}",
        )
        acf_lines.append(ln)
    ax_acf.axhline(0, color="black", ls="--", lw=0.6, alpha=0.6)
    ax_acf.axhline(1, color="black", ls=":", lw=0.6, alpha=0.4)
    ax_acf.axvspan(
        time_ps[-n_tail], time_ps[-1], color="grey", alpha=0.12,
        label=f"S$_0$ tail window (last {args.tail_frac:.0%})",
    )
    ax_acf.set_ylabel(r"$C_k(t) = \langle P_2(\hat v_k(t)\cdot\hat v_k(0)) \rangle_\mathrm{mol}$")
    ax_acf.set_title(
        f"{args.input.stem}\n"
        f"{n_frames} frames × stride {args.stride} = {time_ps[-1]:.1f} ps, "
        f"{n_mol} molecules; mean S$_0$ = {s0:+.3f}, "
        f"D = {d['D_cm2_per_s']:.2e} cm$^2$/s",
        fontsize=10,
    )
    ax_acf.set_xlim(0, time_ps[-1])
    ax_acf.set_ylim(-0.25, 1.05)
    ax_acf.grid(alpha=0.3)
    ax_acf.legend(loc="lower left", fontsize=9, framealpha=0.9)

    msd_line, = ax_msd.plot(
        [], [], color="#37474f", lw=1.6, alpha=0.95, label="COM MSD",
    )
    # Live Einstein fit: line, fit-window band, and D(t) annotation all
    # update per frame using the MSD data available up to that frame.
    fit_line, = ax_msd.plot(
        [], [], color="#c62828", ls="--", lw=1.5, alpha=0.9,
        label=f"Einstein fit (slope/6 = D, last {args.fit_frac:.0%})",
    )
    fit_band = ax_msd.axvspan(
        0, 0, color="grey", alpha=0.12, label=f"D fit window (last {args.fit_frac:.0%})",
    )
    d_text = ax_msd.text(
        0.02, 0.97, "", transform=ax_msd.transAxes, fontsize=10,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.85),
    )
    ax_msd.set_xlabel("Time (ps)")
    ax_msd.set_ylabel(r"COM MSD ($\mathrm{\AA}^2$)")
    ax_msd.set_xlim(0, time_ps[-1])
    msd_pad = max(msd_mean.max(), 1e-6) * 1.10
    ax_msd.set_ylim(0, msd_pad)
    ax_msd.grid(alpha=0.3)
    ax_msd.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # --- Animate ---------------------------------------------------------
    n_render = (n_frames + args.every_nth - 1) // args.every_nth
    print(f"Rendering {n_render} frames @ {args.fps} fps -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    t0 = time.monotonic()
    # Need ≥4 MSD points (so trailing fit_frac of 0.5 leaves ≥2 points for the linear fit).
    min_fit_msd = 4
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(0, n_frames, args.every_nth):
            for k in range(3):
                acf_lines[k].set_data(time_ps[: i + 1], acf[: i + 1, k])
            # MSD has n_frames-1 entries; trajectory frame i corresponds to msd index i-1.
            n_msd = i  # MSD samples available so far
            if n_msd >= 1:
                msd_line.set_data(msd_time_ps[:n_msd], msd_mean[:n_msd])
            if n_msd >= min_fit_msd:
                d_live = fit_diffusion_coefficient(
                    msd_per_mol[:n_msd], msd_time_ps[:n_msd], fit_frac=args.fit_frac,
                )
                fit_xs_live = msd_time_ps[d_live["fit_start_idx"] : n_msd]
                fit_ys_live = d_live["slope"] * fit_xs_live + d_live["intercept"]
                fit_line.set_data(fit_xs_live, fit_ys_live)
                band_t0 = float(msd_time_ps[d_live["fit_start_idx"]])
                band_t1 = float(msd_time_ps[n_msd - 1])
                fit_band.set_x(band_t0)
                fit_band.set_width(band_t1 - band_t0)
                d_text.set_text(
                    f"live  D(t)   = {d_live['D_cm2_per_s']:+.2e} cm²/s\n"
                    f"      slope  = {d_live['slope']:+.2e} Å²/ps\n"
                    f"final D      = {d['D_cm2_per_s']:+.2e} cm²/s"
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
