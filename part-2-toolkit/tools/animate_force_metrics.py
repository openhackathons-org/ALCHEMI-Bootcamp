"""Animate |F|max and |F|mean curves alongside a FIRE force-magnitude render.

Companion to ``render_force.py``. The trajectory video and this metrics
animation share fps so they play in sync side-by-side: at video time τ
the trajectory shows the supercell at FIRE step k and this animation
shows the |F|max / |F|mean curves drawn from step 0 → step k.

Both curves are plotted on a shared log-y axis (FIRE force magnitudes
typically span ~3 decades from initial clashes to plateau). Dashed
horizontal references mark the final-frame values so the convergence
target is visible from the start. X-axis is the FIRE pseudo-step from
``atoms.info['step']`` (FIRE has no MD time — it's the optimizer's
inner iteration counter).

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python animate_force_metrics.py \\
        --input assets/<run>/traj/slc_fire_<...>.extxyz --fps 3
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from matplotlib.animation import FFMpegWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <traj.parent.parent>/figs/<stem>_force_metrics.mp4")
    p.add_argument("--every-nth", type=int, default=1,
                   help="Render every Nth frame (default 1; match the paired trajectory render).")
    p.add_argument("--fps", type=int, default=3,
                   help="Default 3 fps to match render_force.py default.")
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--linear-y", action="store_true",
                   help="Linear y-axis (default: log; FIRE forces span ~3 decades).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_force_metrics.mp4"
    )

    print(f"Loading {args.input.name} ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_frames = len(frames)
    print(f"  -> {n_frames} frames, {len(frames[0])} atoms ({time.monotonic() - t0:.1f}s)")

    try:
        _ = frames[0].get_forces()
    except Exception as exc:
        raise SystemExit(
            f"Trajectory frame 0 exposes no forces ({exc}). Re-run "
            "zarr_to_extxyz.py (forces are auto-included if present in the source zarr)."
        ) from exc

    print("Computing per-frame |F| stats ...")
    fmag_per_frame = [np.linalg.norm(a.get_forces(), axis=1) for a in frames]
    fmax = np.array([m.max() for m in fmag_per_frame])
    fmean = np.array([m.mean() for m in fmag_per_frame])
    steps = np.array([int(a.info.get("step", i)) for i, a in enumerate(frames)])

    fmax_final = float(fmax[-1])
    fmean_final = float(fmean[-1])
    print(
        f"  -> |F|max  range = [{fmax.min():.3e}, {fmax.max():.3e}] eV/Å, "
        f"final = {fmax_final:.3e}"
    )
    print(
        f"  -> |F|mean range = [{fmean.min():.3e}, {fmean.max():.3e}] eV/Å, "
        f"final = {fmean_final:.3e}"
    )

    # --- Figure ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi
    )
    fmax_color = "#c62828"   # red
    fmean_color = "#1565c0"  # blue

    fmax_line, = ax.plot(
        [], [], color=fmax_color, lw=2.0, alpha=0.95,
        marker="o", markersize=5,
        label=r"$|\mathbf{F}|_\mathrm{max}$",
    )
    fmean_line, = ax.plot(
        [], [], color=fmean_color, lw=2.0, alpha=0.95,
        marker="s", markersize=5,
        label=r"$|\mathbf{F}|_\mathrm{mean}$",
    )
    ax.axhline(
        fmax_final, color=fmax_color, ls="--", lw=0.8, alpha=0.55,
        label=rf"final $|\mathbf{{F}}|_\mathrm{{max}}$ = {fmax_final:.3f} eV/Å",
    )
    ax.axhline(
        fmean_final, color=fmean_color, ls="--", lw=0.8, alpha=0.55,
        label=rf"final $|\mathbf{{F}}|_\mathrm{{mean}}$ = {fmean_final:.4f} eV/Å",
    )

    if not args.linear_y:
        ax.set_yscale("log")
    ax.set_xlim(int(steps[0]), int(steps[-1]))
    y_lo = max(float(fmean.min()) * 0.5, 1e-5)
    y_hi = float(fmax.max()) * 1.6
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("FIRE step")
    ax.set_ylabel(r"$|\mathbf{F}|$ (eV/Å)")
    ax.set_title(
        f"{args.input.stem}\n"
        f"{n_frames} FIRE snapshots; "
        rf"final $|\mathbf{{F}}|_\mathrm{{max}}$ = {fmax_final:.3f} eV/Å, "
        rf"$|\mathbf{{F}}|_\mathrm{{mean}}$ = {fmean_final:.4f} eV/Å",
        fontsize=10,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()

    # --- Animate ---------------------------------------------------------
    n_render = (n_frames + args.every_nth - 1) // args.every_nth
    print(f"Rendering {n_render} frames @ {args.fps} fps -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    t0 = time.monotonic()
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(0, n_frames, args.every_nth):
            fmax_line.set_data(steps[: i + 1], fmax[: i + 1])
            fmean_line.set_data(steps[: i + 1], fmean[: i + 1])
            writer.grab_frame()
    plt.close(fig)
    elapsed = time.monotonic() - t0
    print(
        f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{elapsed:.1f}s, {n_render / max(elapsed, 1e-3):.1f} fps render rate)"
    )


if __name__ == "__main__":
    main()
