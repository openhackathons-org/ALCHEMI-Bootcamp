"""Render a warmup-NPT trajectory to MP4 with each molecule's atoms colored
by its tail-averaged Yoneya–Harada rotational order parameter S₀.

ASE/matplotlib path (replaces OVITO Tachyon, which was ~12 s/frame on
this hardware): per-frame scatter of all atoms with per-molecule color +
optional bond LineCollection. Looking down the +a crystal axis so the
b-stack is vertical and c projects horizontal — matches the SLC render
convention. ~50 ms/frame on these naphthalene cells (≈ 1500x faster than
OVITO software ray-tracing); the whole 100-frame video lands in ~6 s.

S₀ is a rigid-body, per-molecule quantity: every atom of a given molecule
inherits that molecule's color. Viridis colormap auto-fits to the data
5–95 percentile by default so spatial heterogeneity is visible deep in
the solid phase (override via ``--vmin`` / ``--vmax``).

Pipeline reuses ``analyze_s0.find_molecules`` for PBC-aware molecule
discovery + ``analyze_s0.molecular_principal_axes`` for inertia
eigenvectors. The per-molecule rotational ACF and S₀ tail are inlined
here in numpy so the script stays in the host conda env (no torch).

Default output: ``<traj.parent.parent>/figs/<traj.stem>_s0.mp4``.

Usage::

    MPLCONFIGDIR=/tmp/claude/mpl python render_warmup_s0.py \\
        --input assets/naphthalene_long_2025/traj/warmup_npt_100k_dt0p5fs.extxyz
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

from analyze_s0 import find_molecules, molecular_principal_axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True, help="Input extxyz trajectory.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output MP4. Default: <traj.parent.parent>/figs/<stem>_s0.mp4")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame subsample stride for S0 computation only.")
    p.add_argument("--every-nth", type=int, default=1,
                   help="Render every Nth frame (default 1 = all frames). matplotlib "
                   "scatter is ~50 ms/frame so 1000 frames takes ~1 min. Bump to 10 "
                   "if you want a shorter, less smooth video.")
    p.add_argument("--ref-idx", type=int, default=0,
                   help="Reference frame for the rotational ACF (default 0).")
    p.add_argument("--tail-frac", type=float, default=0.2,
                   help="Trailing fraction of frames to average as the S0 tail.")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--vmin", type=float, default=None,
                   help="Colormap minimum. Default: data 5th percentile.")
    p.add_argument("--vmax", type=float, default=1.0,
                   help="Colormap maximum. Default 1.0 (perfect crystal).")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="S0 above this is treated as 'crystalline' and compressed "
                   "into a thin top slice of the colormap (default 0.8).")
    p.add_argument("--top-frac", type=float, default=0.05,
                   help="Fraction of the colormap reserved for the [threshold, vmax] "
                   "range (default 0.05). Smaller = more uniform crystal color.")
    p.add_argument("--linear-norm", action="store_true",
                   help="Disable the crystal-emphasis norm; use a plain linear map.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--no-bonds", action="store_true",
                   help="Skip bond LineCollection.")
    p.add_argument("--no-colorbar", action="store_true",
                   help="Suppress the colorbar (caller already knows the scale).")
    p.add_argument("--time-resolved", action="store_true",
                   help="Color by instantaneous C(t) = P_2(v(t)·v(ref))_axis-mean per "
                   "molecule per frame, instead of the static tail-averaged S0. "
                   "Useful for SLC-like trajectories where crystal/melt halves "
                   "blend over time as the crystal decoheres.")
    p.add_argument("--ref-fold", action="store_true",
                   help="SLC mode: molecule m's reference axes are taken from the "
                   "corresponding first-half (crystal) molecule (m mod n/2) at "
                   "frame ref_idx, so the order parameter measures alignment "
                   "with the crystal lattice at each site (not memory of own "
                   "frame-0 axes). Combine with --time-resolved.")
    return p.parse_args()


class CrystalEmphasisNorm(mpl.colors.Normalize):
    """Piecewise-linear norm that compresses the crystalline range.

    ``[threshold, vmax]`` (e.g. 0.8..1.0) maps to the top ``top_frac``
    slice of the colormap (e.g. last 5%), so any S0 in the "crystalline"
    range looks similar. ``[vmin, threshold]`` expands across the
    remaining ``1 - top_frac`` of the colormap so disorder gradations
    are vivid. The colorbar inherits this nonlinearity so ticks are
    placed at the corresponding actual S0 values.
    """

    def __init__(self, vmin, vmax, threshold=0.8, top_frac=0.05, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.threshold = threshold
        self.top_frac = top_frac

    def __call__(self, value, clip=None):
        x = np.asarray(value, dtype=float)
        thr, tf = self.threshold, self.top_frac
        below_span = max(thr - self.vmin, 1e-12)
        above_span = max(self.vmax - thr, 1e-12)
        bottom = (1.0 - tf) * (x - self.vmin) / below_span
        top = (1.0 - tf) + tf * (x - thr) / above_span
        out = np.where(x >= thr, top, bottom)
        return np.ma.masked_array(np.clip(out, 0.0, 1.0))

    def inverse(self, value):
        y = np.asarray(value, dtype=float)
        thr, tf = self.threshold, self.top_frac
        bottom = self.vmin + y / (1.0 - tf) * (thr - self.vmin)
        top = thr + (y - (1.0 - tf)) / tf * (self.vmax - thr)
        return np.where(y >= (1.0 - tf), top, bottom)


def compute_per_mol_S0(frames, ref_idx: int, tail_frac: float, ref_fold: bool = False):
    """Per-molecule S0 (mean over the 3 inertia axes) via the ASE/numpy path.

    Args:
        ref_fold: if True (SLC mode), molecule ``m``'s reference axes are
            ``axes[ref_idx][m % (n_mol // 2)]`` — the corresponding first-half
            (crystal) molecule. Encodes "alignment with the perfect crystal
            lattice at this site" rather than "memory of own frame-0 axes".
            Crystal half then starts at S0=1, melt half starts at random
            P_2 of uniform sphere ≈ 0 with spread.

    Returns:
        ``(s0_per_mol [n_mol], s0_per_mol_per_frame [n_frames, n_mol], molecules)``
    """
    molecules = find_molecules(frames[0])
    sizes = {len(m) for m in molecules}
    if len(sizes) != 1 or 18 not in sizes:
        raise SystemExit(
            f"Expected uniformly 18 atoms/mol for naphthalene; got sizes={sorted(sizes)}"
        )
    n_mol = len(molecules)
    n_frames = len(frames)
    axes_seq = np.zeros((n_frames, n_mol, 3, 3))
    for i, atoms in enumerate(frames):
        axes_seq[i] = molecular_principal_axes(atoms, molecules)

    if ref_fold:
        if n_mol % 2 != 0:
            raise SystemExit(
                f"--ref-fold requires even n_mol (got {n_mol}); SLC stacks always "
                f"have equal crystal+melt halves."
            )
        half = n_mol // 2
        ref = axes_seq[ref_idx][np.arange(n_mol) % half]  # [n_mol, 3, 3]
    else:
        ref = axes_seq[ref_idx]
    acf = np.zeros((n_frames, n_mol, 3))
    for t in range(n_frames):
        dot = (axes_seq[t] * ref).sum(axis=-1)
        acf[t] = 0.5 * (3.0 * dot * dot - 1.0)

    n_tail = max(1, int(n_frames * tail_frac))
    s0_per_mol = acf[-n_tail:].mean(axis=0).mean(axis=1)  # [n_mol]
    s0_per_mol_per_frame = acf.mean(axis=2)  # [n_frames, n_mol]
    return s0_per_mol, s0_per_mol_per_frame, molecules


def unwrap_per_molecule(positions: np.ndarray, cell: np.ndarray, molecules) -> np.ndarray:
    """MIC unwrap: each molecule's atoms relative to its first atom.

    Keeps molecules visually contiguous when one or more of their atoms
    happen to wrap across a PBC boundary mid-trajectory. Same pattern as
    ``analyze_s0.molecular_principal_axes``.
    """
    out = positions.copy()
    cell_inv = np.linalg.inv(cell)
    for idx in molecules:
        ref = out[idx[0]]
        dr = out[idx] - ref
        dr_frac = dr @ cell_inv
        dr_frac -= np.round(dr_frac)
        out[idx] = ref + dr_frac @ cell
    return out


def build_static_bonds(positions: np.ndarray, molecules, cutoff: float = 1.65):
    """All bonds within a molecule at frame 0 (already unwrapped).

    Returned as `(i, j)` index pairs. Bond list is computed once and the
    same indices are used every frame — only positions change.
    """
    bonds = []
    for idx in molecules:
        idx_arr = np.asarray(idx)
        # Pairwise distances within this molecule (no PBC since unwrapped).
        d = np.linalg.norm(positions[idx_arr][:, None] - positions[idx_arr][None, :], axis=-1)
        for a, b in zip(*np.where((d > 0) & (d < cutoff))):
            if a < b:
                bonds.append((int(idx_arr[a]), int(idx_arr[b])))
    return np.asarray(bonds, dtype=np.int64) if bonds else np.zeros((0, 2), dtype=np.int64)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    out = args.output or (
        args.input.parent.parent / "figs" / f"{args.input.stem}_s0.mp4"
    )

    print(f"Loading {args.input.name} (stride={args.stride}) ...")
    t0 = time.monotonic()
    frames = ase_read(str(args.input), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    n_atoms = len(frames[0])
    print(f"  -> {len(frames)} frames, {n_atoms} atoms ({time.monotonic() - t0:.1f}s)")

    print("Discovering molecules + computing per-mol S0 ...")
    t0 = time.monotonic()
    s0_per_mol, s0_per_mol_per_frame, molecules = compute_per_mol_S0(
        frames, args.ref_idx, args.tail_frac, ref_fold=args.ref_fold
    )
    n_mol = len(molecules)
    print(
        f"  -> {n_mol} molecules; S0 min/median/mean/max = "
        f"{s0_per_mol.min():.3f} / {np.median(s0_per_mol):.3f} / "
        f"{s0_per_mol.mean():.3f} / {s0_per_mol.max():.3f} "
        f"({time.monotonic() - t0:.1f}s)"
    )

    vmin = args.vmin if args.vmin is not None else float(np.percentile(s0_per_mol, 5))
    vmax = float(args.vmax)
    cmap = mpl.colormaps[args.cmap]
    if args.linear_norm:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        print(f"  -> linear colormap range: [{vmin:.4f}, {vmax:.4f}]")
    else:
        norm = CrystalEmphasisNorm(
            vmin=vmin, vmax=vmax, threshold=args.threshold, top_frac=args.top_frac
        )
        print(
            f"  -> crystal-emphasis norm: [{vmin:.4f}, {args.threshold:.2f}] -> "
            f"bottom {(1 - args.top_frac) * 100:.0f}% of cmap, "
            f"[{args.threshold:.2f}, {vmax:.4f}] -> top {args.top_frac * 100:.0f}%"
        )

    atom_to_mol = np.empty(n_atoms, dtype=np.int64)
    for k, idx in enumerate(molecules):
        atom_to_mol[idx] = k
    s0_per_atom = s0_per_mol[atom_to_mol]

    Z = np.asarray(frames[0].get_atomic_numbers())
    sizes = np.where(Z == 6, 55, 22).astype(float)  # C bigger than H

    pos0 = unwrap_per_molecule(
        frames[0].positions, np.asarray(frames[0].cell), molecules
    )
    bonds = (
        build_static_bonds(pos0, molecules)
        if not args.no_bonds
        else np.zeros((0, 2), dtype=np.int64)
    )
    print(f"  -> {len(bonds)} static bonds ({len(bonds) / max(n_mol, 1):.1f} per mol)")

    # Cell projects to a rectangle in the y-z plane (a along x → drops out).
    # Pre-scan all frames so the viewport fits the LARGEST cell of the
    # trajectory (NPT cells breathe; NVT cells are constant — no harm).
    cells_yz = np.array(
        [[float(a.cell[1, 1]), float(a.cell[2, 2])] for a in frames]
    )  # [n_frames, 2] = (b_y, c_z) per frame
    b_y_max, c_z_max = float(cells_yz[:, 0].max()), float(cells_yz[:, 1].max())
    b_y_0, c_z_0 = float(cells_yz[0, 0]), float(cells_yz[0, 1])
    pad = 2.0  # Å margin around the cell box

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
    if args.no_colorbar:
        fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.06)
    else:
        fig.subplots_adjust(left=0.06, right=0.86, top=0.94, bottom=0.06)

    # Cell box — kept as an artist so it updates per frame as the cell breathes.
    cell_box = ax.add_patch(
        Rectangle((0, 0), c_z_0, b_y_0, fill=False, edgecolor="0.3", linewidth=1.2)
    )

    title = ax.set_title("")
    if not args.no_colorbar:
        cbar_ax = fig.add_axes([0.89, 0.10, 0.014, 0.82])
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
        cb.set_label("S$_0$  (per molecule)")

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
            x_screen = pos[:, 2]  # z (horizontal)
            y_screen = pos[:, 1]  # y (vertical)

            if scatter_artist is None:
                scatter_artist = ax.scatter(
                    x_screen, y_screen, c=s0_per_atom, cmap=cmap, norm=norm,
                    s=sizes, edgecolors="black", linewidths=0.35, zorder=3,
                )
            else:
                scatter_artist.set_offsets(np.column_stack([x_screen, y_screen]))
            if args.time_resolved:
                s0_now = s0_per_mol_per_frame[i][atom_to_mol]
                scatter_artist.set_array(s0_now)
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

            title.set_text(f"frame {i}/{len(frames) - 1}   S₀ ∈ [{vmin:.3f}, {vmax:.3f}]")
            writer.grab_frame()

    plt.close(fig)
    elapsed = time.monotonic() - t0
    print(
        f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{elapsed:.1f}s, {n_render / max(elapsed, 1e-3):.1f} fps render rate)"
    )


if __name__ == "__main__":
    main()
