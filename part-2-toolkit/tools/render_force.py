"""Animate a (FIRE) trajectory with atoms colored by force magnitude.

Companion to ``render_warmup_s0.py``. Same matplotlib pipeline (per-frame
scatter + bond LineCollection + dynamic cell-box outline), but each atom's
color encodes |F_i| at that frame instead of a per-molecule S0. Designed
for FIRE relaxations where the visual story is "initially clashed → bright
forces fade as the structure relaxes".

Forces must be present in the input extxyz under the standard ``forces``
property — produced by re-running ``zarr_to_extxyz.py`` (which auto-includes
forces if the source zarr has them).

Default colormap is ``inferno`` (dark = relaxed, bright = clashing). Default
norm is log-scale because FIRE force magnitudes typically span 3+ decades:
construction clashes can give |F| ~ 50–100 eV/Å while converged values sit
around fmax ≈ 0.1–1 eV/Å.

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python render_force.py \\
        --input assets/<run>/traj/slc_fire_<...>.extxyz --fps 3
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <traj.parent.parent>/figs/<stem>_force.mp4")
    p.add_argument("--every-nth", type=int, default=1)
    p.add_argument("--cmap", default="inferno",
                   help="Sequential colormap (default 'inferno' — dark = relaxed, bright = clashing).")
    p.add_argument("--vmin", type=float, default=None,
                   help="Force-magnitude minimum (eV/Å). Default: trajectory min, clipped to 1e-3 in log mode.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Force-magnitude maximum (eV/Å). Default: trajectory max.")
    p.add_argument("--linear-norm", action="store_true",
                   help="Linear norm (default is log — forces span ~3 decades).")
    p.add_argument("--fps", type=int, default=3,
                   help="Default 3 fps — FIRE trajectories are typically short (≲ 50 frames).")
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--no-bonds", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_force.mp4"
    )

    print(f"Loading {args.input.name} ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_atoms = len(frames[0])
    n_frames = len(frames)
    print(f"  -> {n_frames} frames, {n_atoms} atoms ({time.monotonic() - t0:.1f}s)")

    # ASE's extxyz reader puts forces on a SinglePointCalculator
    # (atoms.get_forces()), not in atoms.arrays.
    try:
        _ = frames[0].get_forces()
    except Exception as exc:
        raise SystemExit(
            f"Trajectory frame 0 exposes no forces ({exc}). Re-run "
            "zarr_to_extxyz.py (which auto-includes forces if present in "
            "the source zarr)."
        ) from exc

    print("Computing per-atom force magnitudes ...")
    t0 = time.monotonic()
    fmag_per_frame = np.array(
        [np.linalg.norm(a.get_forces(), axis=1) for a in frames]
    )  # [n_frames, n_atoms]
    print(
        f"  -> |F| (eV/Å) global min/median/max = "
        f"{fmag_per_frame.min():.3e} / {np.median(fmag_per_frame):.3e} / "
        f"{fmag_per_frame.max():.3e} ({time.monotonic() - t0:.1f}s)"
    )

    cmap = mpl.colormaps[args.cmap]
    if args.linear_norm:
        vmin = args.vmin if args.vmin is not None else 0.0
        vmax = args.vmax if args.vmax is not None else float(fmag_per_frame.max())
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        print(f"  -> linear color range: [{vmin:.3e}, {vmax:.3e}] eV/Å")
    else:
        positives = fmag_per_frame[fmag_per_frame > 0]
        vmin = args.vmin if args.vmin is not None else max(float(positives.min()), 1e-3)
        vmax = args.vmax if args.vmax is not None else float(fmag_per_frame.max())
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        print(f"  -> log color range: [{vmin:.3e}, {vmax:.3e}] eV/Å")

    Z = np.asarray(frames[0].get_atomic_numbers())
    sizes = np.where(Z == 6, 55, 22).astype(float)

    # Bonds: detect at frame 0 via a simple within-cutoff pair search inside
    # each connected molecule. Reuse the same logic as render_warmup_s0 but
    # inline — keeps render_force standalone.
    if not args.no_bonds:
        print("Detecting bonds at frame 0 ...")
        from ase.neighborlist import primitive_neighbor_list
        ai, aj = primitive_neighbor_list(
            "ij",
            pbc=frames[0].pbc, cell=frames[0].cell,
            positions=frames[0].positions, cutoff=1.65,
        )
        # Keep only i<j to avoid duplicates.
        keep = ai < aj
        bonds = np.column_stack([ai[keep], aj[keep]]).astype(np.int64)
        print(f"  -> {len(bonds)} bonds")
    else:
        bonds = np.zeros((0, 2), dtype=np.int64)

    # Cell projects to a rectangle in the y-z plane (a along x → drops out).
    cells_yz = np.array(
        [[float(a.cell[1, 1]), float(a.cell[2, 2])] for a in frames]
    )  # [n_frames, 2] = (b_y, c_z)
    b_y_max, c_z_max = float(cells_yz[:, 0].max()), float(cells_yz[:, 1].max())
    b_y_0, c_z_0 = float(cells_yz[0, 0]), float(cells_yz[0, 1])
    pad = 2.0

    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi
    )
    ax.set_aspect("equal")
    ax.set_xlim(-pad, c_z_max + pad)
    ax.set_ylim(-pad, b_y_max + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(rf"$c$-axis projection (max {c_z_max:.1f} Å)")
    ax.set_ylabel(rf"$b$-axis (max {b_y_max:.1f} Å)")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.06)

    cell_box = ax.add_patch(
        Rectangle((0, 0), c_z_0, b_y_0, fill=False, edgecolor="0.3", linewidth=1.2)
    )
    title = ax.set_title("")

    n_render = (n_frames + args.every_nth - 1) // args.every_nth
    print(f"Rendering {n_render} frames @ {args.fps} fps -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    scatter_artist = None
    bond_artist = None
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    t0 = time.monotonic()
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(0, n_frames, args.every_nth):
            atoms = frames[i]
            pos = atoms.positions
            x_screen = pos[:, 2]  # z (horizontal)
            y_screen = pos[:, 1]  # y (vertical)
            fmag_now = fmag_per_frame[i]

            if scatter_artist is None:
                scatter_artist = ax.scatter(
                    x_screen, y_screen, c=fmag_now, cmap=cmap, norm=norm,
                    s=sizes, edgecolors="black", linewidths=0.35, zorder=3,
                )
            else:
                scatter_artist.set_offsets(np.column_stack([x_screen, y_screen]))
                scatter_artist.set_array(fmag_now)
            cell_box.set_width(float(cells_yz[i, 1]))
            cell_box.set_height(float(cells_yz[i, 0]))

            if len(bonds):
                segs = np.stack(
                    [np.column_stack([x_screen[bonds[:, 0]], y_screen[bonds[:, 0]]]),
                     np.column_stack([x_screen[bonds[:, 1]], y_screen[bonds[:, 1]]])],
                    axis=1,
                )
                if bond_artist is None:
                    bond_artist = LineCollection(
                        segs, colors="0.3", linewidths=0.9, alpha=0.75, zorder=2,
                    )
                    ax.add_collection(bond_artist)
                else:
                    bond_artist.set_segments(segs)

            fmax_now = float(fmag_now.max())
            fmean_now = float(fmag_now.mean())
            f_max_sym = r"$|\mathbf{F}|_\mathrm{max}$"
            f_mean_sym = r"$|\mathbf{F}|_\mathrm{mean}$"
            title.set_text(
                f"frame {i}/{n_frames - 1}    "
                f"{f_max_sym} = {fmax_now:.2f} eV/Å    "
                f"{f_mean_sym} = {fmean_now:.3f} eV/Å"
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
