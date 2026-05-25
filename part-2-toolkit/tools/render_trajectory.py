"""Render a pulled MD trajectory to a video file.

Uses the same `--stage` / `--source` / `--temps` / `--t-warmup` / `--dt`
/ `--run-name` flag set as ``visualize_warmup_trajectory.py`` so the
same arguments select the same artefact: if you can `visualize` it,
you can `render` it.

Pipeline:

  1. Load the trajectory (extxyz / extxyz.gz / xyz).
  2. Unwrap molecules using the connectivity discovered on frame 0
     (skip with `--no-unwrap`); same logic as the viewer.
  3. Render each frame to PNG via ASE's built-in PIL renderer (camera
     `--rotation`, `--scale` pixels/Angstrom, full unit cell).
  4. Stitch the PNG sequence into MP4 with ffmpeg + libx264 (or directly
     into a GIF via PIL if `--out` ends in ``.gif``).

Default output:
    ``assets/<run-name>/figs/<trajectory-basename>.mp4``
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read, write
from ase.utils import rotate as ase_rotate

from visualize_warmup_trajectory import (
    find_molecules,
    resolve_trajectory,
    unwrap_frame,
)

HERE = Path(__file__).parent

# Stems we recognise as trajectory containers so we can produce a clean
# basename without the format extensions stacked up.
_TRAJ_SUFFIXES = (".extxyz.gz", ".extxyz", ".xyz.gz", ".xyz")


def _frame_xyextent(atoms, rot, show_cell: bool = True):
    """Projected (X1, X2) of atoms (with covalent radii) plus cell vertices,
    after applying ``rot`` (3x3). Mirrors the auto-bbox calc in
    ``ase.io.utils.PlottingVariables``; values are in Angstrom in the
    rotated 2D frame (z is irrelevant for the writer).
    """
    radii = covalent_radii[atoms.get_atomic_numbers()]
    R = atoms.get_positions() @ rot
    X1 = (R - radii[:, None]).min(axis=0)
    X2 = (R + radii[:, None]).max(axis=0)
    if show_cell:
        cell = np.asarray(atoms.get_cell())
        disp = atoms.get_celldisp().flatten()
        verts = np.array([
            np.dot([c1, c2, c3], cell) + disp
            for c1 in (0, 1) for c2 in (0, 1) for c3 in (0, 1)
        ]) @ rot
        X1 = np.minimum(X1, verts.min(axis=0))
        X2 = np.maximum(X2, verts.max(axis=0))
    return X1, X2


def compute_union_bbox(frames, rotation_str: str, margin_frac: float = 0.025):
    """Return ``(x0, y0, x1, y1)`` (Angstrom, rotated frame) covering every
    frame's atoms+cell. Pass to ``write(..., bbox=...)`` so all PNGs share
    the same canvas — eliminates per-frame size jitter under NPT (the cause
    of flickering letterbox bars in the encoded video / GIF).
    """
    rot = ase_rotate(rotation_str)
    X1_g = np.full(3, np.inf)
    X2_g = np.full(3, -np.inf)
    for atoms in frames:
        X1, X2 = _frame_xyextent(atoms, rot, show_cell=True)
        X1_g = np.minimum(X1_g, X1)
        X2_g = np.maximum(X2_g, X2)
    M = (X1_g + X2_g) / 2
    S = (1.0 + 2.0 * margin_frac) * (X2_g - X1_g)
    return (
        float(M[0] - S[0] / 2),
        float(M[1] - S[1] / 2),
        float(M[0] + S[0] / 2),
        float(M[1] + S[1] / 2),
    )


def basename_no_traj_ext(path: Path) -> str:
    name = path.name
    for s in _TRAJ_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return path.stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "trajectory", nargs="?", type=Path, default=None,
        help="Path to the trajectory (extxyz / extxyz.gz / xyz). When "
             "omitted, the path is built from --stage / --source / "
             "--temps / --t-warmup / --dt under assets/<run-name>/traj/.",
    )
    p.add_argument(
        "--stage",
        choices=[
            "warmup-fire", "warmup-nvt", "warmup-npt",
            "melt-nvt",
            "slc-fire", "slc-nvt", "slc-npt",
        ],
        default="warmup-npt",
        help="Mirrors visualize_warmup_trajectory.py --stage.",
    )
    p.add_argument(
        "--source", choices=["nvt", "npt"], default="npt",
        help="For melt-nvt or slc-* stages: which warmup endpoint seeded "
             "the melt half.",
    )
    p.add_argument(
        "--temps", default="",
        help="For slc-* stages: comma-separated subset matching "
             "slc.py --temps (e.g. '250,300').",
    )
    p.add_argument(
        "--run-name", default="naphthalene_long",
        help="Run name; drives the default-trajectory search path under "
             "assets/<run-name>/traj/ and the default --out figs/ path.",
    )
    p.add_argument(
        "--t-warmup", type=float, default=200.0,
        help="Warmup target T in K (default 200). Tags the seed lookup.",
    )
    p.add_argument(
        "--dt", type=float, default=0.5,
        help="MD timestep in fs; tags the seed lookup.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride (default 1 = every frame; use 5 to subsample).",
    )
    p.add_argument(
        "--no-unwrap", action="store_true",
        help="Skip per-frame molecular unwrapping (PBC splits will be "
             "visible in the rendered video).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output video path. Default is "
             "``assets/<run-name>/figs/<basename>.mp4``. Use a `.gif` "
             "extension to write directly via PIL (no ffmpeg needed; "
             "much larger files but viewable everywhere).",
    )
    p.add_argument(
        "--fps", type=int, default=10,
        help="Output framerate (default 10). MD trajectories with 100 "
             "frames look natural at 10 fps (10 s clip).",
    )
    p.add_argument(
        "--rotation", default="-60x,30y,0z",
        help="ASE rotation string '<x>x,<y>y,<z>z' (degrees) controlling "
             "the camera. '0x,0y,0z' = top-down; '-60x,30y,0z' = perspective. "
             "Apply once -- consistent across all frames.",
    )
    p.add_argument(
        "--scale", type=float, default=12.0,
        help="ASE PIL render scale in pixels/Angstrom (default 12). 12 -> "
             "~800 px wide for a 67-A SLC cell. Higher = more detail + "
             "larger file. NOTE: ASE caps the actual PNG width at "
             "--maxwidth (default 500), so bump --maxwidth alongside "
             "--scale for true higher-resolution renders.",
    )
    p.add_argument(
        "--no-fixed-bbox", action="store_true",
        help="Disable union-bbox stabilisation. By default the renderer "
             "computes one bbox covering all frames so PNG canvases are "
             "identical-sized; this prevents flickering letterbox/pillarbox "
             "bars in NPT trajectories where the cell breathes.",
    )
    p.add_argument(
        "--maxwidth", type=int, default=500,
        help="Cap on the rendered PNG width in pixels (ASE's PlottingVariables "
             "default = 500). Bump (e.g. 4000) to actually honour high "
             "--scale values; otherwise ffmpeg's lanczos upscale to --width "
             "produces blurry atom edges.",
    )
    p.add_argument(
        "--width", type=int, default=1280,
        help="Final video width in px (height auto from aspect; lanczos "
             "downsample). Ignored for .gif output.",
    )
    p.add_argument(
        "--height", type=int, default=None,
        help="If set, force final video to --width X --height with "
             "aspect-preserving scale + letterbox/pillarbox padding "
             "(black bars). Use to lock output to e.g. 1920x1080.",
    )
    return p.parse_args()


def render_mp4(
    frames_dir: Path, out_path: Path, fps: int, width: int,
    height: int | None = None,
) -> None:
    if height is None:
        vf = f"scale={width}:-2:flags=lanczos"
    else:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if args.trajectory is None:
        args.trajectory = resolve_trajectory(
            args.run_name, args.stage, args.source,
            args.t_warmup, args.dt, args.temps,
        )
    if not args.trajectory.exists():
        sys.exit(f"Not found: {args.trajectory}")

    out_path = args.out
    if out_path is None:
        figs = HERE / "assets" / args.run_name / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        out_path = figs / f"{basename_no_traj_ext(args.trajectory)}.mp4"

    if out_path.suffix == ".mp4" and shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH. Install via `brew install ffmpeg` "
                 "or use a `.gif` --out instead.")

    print(f"Loading {args.trajectory} (stride={args.stride}) ...")
    frames = read(str(args.trajectory), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    print(f"  {len(frames)} frames, {len(frames[0])} atoms each")

    if not args.no_unwrap:
        print("Unwrapping molecules across PBC ...")
        molecules = find_molecules(frames[0])
        for atoms in frames:
            unwrap_frame(atoms, molecules)

    bbox = None
    if not args.no_fixed_bbox:
        print("Computing union bbox across all frames "
              "(stabilises canvas under NPT cell drift) ...")
        bbox = compute_union_bbox(frames, args.rotation)
        print(f"  bbox (rotated frame, A): "
              f"x=[{bbox[0]:.2f}, {bbox[2]:.2f}], "
              f"y=[{bbox[1]:.2f}, {bbox[3]:.2f}]")

    print(f"Rendering -> {out_path}")
    if out_path.suffix == ".gif":
        # ASE writes the GIF directly via PIL; `interval` is per-frame ms.
        write(
            str(out_path), frames,
            rotation=args.rotation, scale=args.scale,
            maxwidth=args.maxwidth, bbox=bbox, show_unit_cell=2,
            interval=int(1000 / args.fps),
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for i, atoms in enumerate(frames):
                png = tmpdir / f"frame_{i:05d}.png"
                write(
                    str(png), atoms,
                    rotation=args.rotation, scale=args.scale,
                    maxwidth=args.maxwidth, bbox=bbox, show_unit_cell=2,
                )
                if (i + 1) % 25 == 0 or (i + 1) == len(frames):
                    print(f"  {i + 1}/{len(frames)} frames")
            render_mp4(tmpdir, out_path, args.fps, args.width, args.height)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
