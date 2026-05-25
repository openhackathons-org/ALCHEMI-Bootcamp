"""Render a single MD frame to a PNG image via ASE's PIL writer.

Modified version of ``visualize_warmup_trajectory.py`` that writes a
still PNG of one frame rather than opening the ASE GUI. Same
trajectory-resolution flags so the same ``--stage`` / ``--source`` /
``--temps`` / ``--t-warmup`` / ``--dt`` / ``--run-name`` tuple
selects, visualises, and renders the same artefact.

Usage::

    # Frame 0 of the warmup-NVT trajectory, looking down the b-axis:
    python render_frame.py \\
        --stage warmup-nvt --run-name naphthalene_long_2025 \\
        --t-warmup 200 --dt 0.5

    # Initial packmol box (warmup-FIRE frame 0) for the orb-iso run:
    python render_frame.py \\
        --stage warmup-fire --run-name naphthalene_orb_iso --dt 0.5
"""

import argparse
import sys
from pathlib import Path

from ase.io import read, write

from visualize_warmup_trajectory import (
    find_molecules,
    resolve_trajectory,
    unwrap_frame,
)

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the extended-XYZ trajectory (accepts .xyz / .extxyz / "
        ".extxyz.gz). When omitted, resolved via --stage / --source / "
        "--t-warmup / --dt under assets/<run-name>/traj/.",
    )
    p.add_argument(
        "--stage",
        choices=[
            "warmup-fire", "warmup-nvt", "warmup-npt",
            "melt-nvt",
            "slc-fire", "slc-nvt", "slc-npt",
        ],
        default="warmup-nvt",
    )
    p.add_argument("--source", choices=["nvt", "npt"], default="npt")
    p.add_argument("--temps", default="")
    p.add_argument("--run-name", default="naphthalene_long")
    p.add_argument("--t-warmup", type=float, default=200.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to render (default 0).",
    )
    p.add_argument(
        "--rotation", default="-90x,0y,0z",
        help="ASE rotation string. Default '-90x,0y,0z' looks down the +b "
        "(y) axis -- the convention used for the crystal frame-0 view.",
    )
    p.add_argument(
        "--no-unwrap", action="store_true",
        help="Skip molecular unwrapping across PBC.",
    )
    p.add_argument(
        "--scale", type=float, default=20.0,
        help="ASE PIL scale in px/A (default 20).",
    )
    p.add_argument(
        "--maxwidth", type=int, default=1600,
        help="ASE PIL maxwidth cap in px (default 1600).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG. Default: "
        "assets/<run-name>/figs/<traj-basename>_frame<N>_view_b.png",
    )
    return p.parse_args()


_TRAJ_SUFFIXES = (".extxyz.gz", ".extxyz", ".xyz.gz", ".xyz")


def _basename_no_traj_ext(path: Path) -> str:
    name = path.name
    for s in _TRAJ_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return path.stem


def main() -> None:
    args = parse_args()
    if args.trajectory is None:
        args.trajectory = resolve_trajectory(
            args.run_name, args.stage, args.source,
            args.t_warmup, args.dt, args.temps,
        )
    if not args.trajectory.exists():
        sys.exit(f"Not found: {args.trajectory}")

    print(f"Loading frame {args.frame} from {args.trajectory} ...")
    frames = read(str(args.trajectory), index=f"{args.frame}:{args.frame + 1}",
                  format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    atoms = frames[0]
    print(f"  {len(atoms)} atoms; cell lengths = "
          f"{atoms.cell.lengths().tolist()} A")

    if not args.no_unwrap:
        molecules = find_molecules(atoms)
        sizes = sorted({len(m) for m in molecules})
        print(f"  {len(molecules)} molecules; atoms per molecule: {sizes}")
        unwrap_frame(atoms, molecules)

    if args.out is None:
        figs = HERE / "assets" / args.run_name / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        stem = _basename_no_traj_ext(args.trajectory)
        args.out = figs / f"{stem}_frame{args.frame}_view_b.png"

    print(f"Writing -> {args.out}")
    write(
        str(args.out), atoms,
        rotation=args.rotation,
        scale=args.scale,
        maxwidth=args.maxwidth,
        show_unit_cell=2,
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"Done: {args.out} ({size_kb:.1f} kB)")


if __name__ == "__main__":
    main()
