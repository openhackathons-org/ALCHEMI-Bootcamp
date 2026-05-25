"""Render an SLC trajectory to MP4 with each molecule colored by which half
of the initial cell it came from (crystal = blue, melt = orange).

Visual diagnostic for solid-liquid interface dynamics: original-phase labels
are fixed at frame 0 and travel with the molecule, so a blue dot in the
melt half (or vice versa) is a molecule that has physically crossed the
interface during the run. Same projection convention as ``render_warmup_s0.py``
(view down +a, b vertical, c horizontal).

SLC construction places crystal molecules first then melt molecules (see
``slc.py``), so molecule indices ``[0, n_mol/2)`` are
crystal and ``[n_mol/2, n_mol)`` are melt. Atoms inherit their molecule's
phase color; molecule indices come from ``analyze_s0.find_molecules``.

Default output: ``<traj.parent.parent>/figs/<traj.stem>_phase.mp4``.

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python render_slc_phase.py \\
        --input assets/naphthalene_long_2025/traj/slc_all_from_npt_100k_dt0p5fs_t250.extxyz
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from analyze_s0 import find_molecules
from render_warmup_s0 import build_static_bonds, unwrap_per_molecule

CRYSTAL_COLOR = "#1f77b4"  # matplotlib tab:blue
MELT_COLOR = "#ff7f0e"     # matplotlib tab:orange


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True, help="Input extxyz trajectory.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output MP4. Default: <traj.parent.parent>/figs/<stem>_phase.mp4")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame subsample stride applied at read time.")
    p.add_argument("--every-nth", type=int, default=1,
                   help="Render every Nth frame (default 1 = all loaded frames).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--no-bonds", action="store_true",
                   help="Skip bond LineCollection.")
    p.add_argument("--no-interface-line", action="store_true",
                   help="Suppress the dashed line marking the initial b/2 boundary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_phase.mp4"
    )

    print(f"Loading {args.input.name} (stride={args.stride}) ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_atoms = len(frames[0])
    print(f"  -> {len(frames)} frames, {n_atoms} atoms ({time.monotonic() - t0:.1f}s)")

    print("Discovering molecules ...")
    t0 = time.monotonic()
    molecules = find_molecules(frames[0])
    sizes_set = {len(m) for m in molecules}
    if len(sizes_set) != 1 or 18 not in sizes_set:
        raise SystemExit(
            f"Expected uniformly 18 atoms/mol for naphthalene; got sizes={sorted(sizes_set)}"
        )
    n_mol = len(molecules)
    if n_mol % 2 != 0:
        raise SystemExit(
            f"SLC cell must have even n_mol (got {n_mol}); crystal+melt halves must match."
        )
    half = n_mol // 2
    print(f"  -> {n_mol} molecules, half = {half} ({time.monotonic() - t0:.1f}s)")

    # Phase label per molecule: 0 = crystal (first half), 1 = melt (second half).
    # Sanity-check the ordering by comparing the mean y-coordinate (b-projection)
    # of each half at frame 0; crystal sits below melt in the stacked cell.
    pos0_raw = np.asarray(frames[0].positions)
    crystal_y = np.mean([pos0_raw[idx, 1].mean() for idx in molecules[:half]])
    melt_y = np.mean([pos0_raw[idx, 1].mean() for idx in molecules[half:]])
    print(f"  -> mean b at frame 0: crystal={crystal_y:.2f} Å  melt={melt_y:.2f} Å")
    if crystal_y > melt_y:
        print("  !! first-half molecules sit above second-half — labels may be flipped")

    atom_phase = np.empty(n_atoms, dtype=np.int64)
    for k, idx in enumerate(molecules):
        atom_phase[idx] = 0 if k < half else 1
    atom_colors = np.where(atom_phase[:, None] == 0,
                           np.array(plt.matplotlib.colors.to_rgba(CRYSTAL_COLOR)),
                           np.array(plt.matplotlib.colors.to_rgba(MELT_COLOR)))

    Z = np.asarray(frames[0].get_atomic_numbers())
    sizes = np.where(Z == 6, 55, 22).astype(float)

    pos0 = unwrap_per_molecule(
        frames[0].positions, np.asarray(frames[0].cell), molecules
    )
    bonds = (
        build_static_bonds(pos0, molecules)
        if not args.no_bonds
        else np.zeros((0, 2), dtype=np.int64)
    )
    print(f"  -> {len(bonds)} static bonds ({len(bonds) / max(n_mol, 1):.1f} per mol)")

    cells_yz = np.array(
        [[float(a.cell[1, 1]), float(a.cell[2, 2])] for a in frames]
    )
    b_y_max, c_z_max = float(cells_yz[:, 0].max()), float(cells_yz[:, 1].max())
    b_y_0, c_z_0 = float(cells_yz[0, 0]), float(cells_yz[0, 1])
    interface_y0 = b_y_0 / 2.0
    pad = 2.0

    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi
    )
    ax.set_aspect("equal")
    ax.set_xlim(-pad, c_z_max + pad)
    ax.set_ylim(-pad, b_y_max + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"c-axis projection (max {c_z_max:.1f} Å)")
    ax.set_ylabel(f"b-axis (max {b_y_max:.1f} Å)")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06)

    cell_box = ax.add_patch(
        Rectangle((0, 0), c_z_0, b_y_0, fill=False, edgecolor="0.3", linewidth=1.2)
    )
    interface_line = None
    if not args.no_interface_line:
        interface_line = ax.axhline(
            interface_y0, color="0.4", linestyle="--", linewidth=1.0, zorder=1
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=CRYSTAL_COLOR, markeredgecolor="black",
               markeredgewidth=0.4, markersize=8, label="crystal half initial"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=MELT_COLOR, markeredgecolor="black",
               markeredgewidth=0.4, markersize=8, label="melt half initial"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

    title = ax.set_title("")

    n_render = (len(frames) + args.every_nth - 1) // args.every_nth
    print(f"Rendering {n_render} frames @ {args.fps} fps -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    scatter_artist = None
    bond_artist = None
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    t0 = time.monotonic()
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(0, len(frames), args.every_nth):
            atoms = frames[i]
            pos = unwrap_per_molecule(
                atoms.positions, np.asarray(atoms.cell), molecules
            )
            x_screen = pos[:, 2]
            y_screen = pos[:, 1]

            if scatter_artist is None:
                scatter_artist = ax.scatter(
                    x_screen, y_screen, c=atom_colors, s=sizes,
                    edgecolors="black", linewidths=0.35, zorder=3,
                )
            else:
                scatter_artist.set_offsets(np.column_stack([x_screen, y_screen]))
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
                        segs, colors="0.3", linewidths=0.9, alpha=0.6, zorder=2,
                    )
                    ax.add_collection(bond_artist)
                else:
                    bond_artist.set_segments(segs)

            title.set_text(f"frame {i}/{len(frames) - 1}   crystal={half} mol  melt={half} mol")
            writer.grab_frame()

    plt.close(fig)
    elapsed = time.monotonic() - t0
    print(
        f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{elapsed:.1f}s, {n_render / max(elapsed, 1e-3):.1f} fps render rate)"
    )


if __name__ == "__main__":
    main()
