"""Render a single MD frame to a PNG via OVITO's TachyonRenderer.

OVITO variant of ``render_frame.py``: same trajectory-resolution flags
(``--stage`` / ``--source`` / ``--temps`` / ``--t-warmup`` / ``--dt`` /
``--run-name``) but renders with ambient occlusion + direct lighting +
perspective projection so dense amorphous boxes no longer flatten into
a fully-filled silhouette. The molecule-unwrap helpers are imported
from ``visualize_warmup_trajectory`` so the same PBC-stitching applies.

Camera presets:

  ``b``        head-on along +b (cubic box → square; monoclinic crystal
               → parallelogram face). Matches the ASE PIL "view_b" view.
  ``b_tilt``   along +b with a ~17° tilt; reveals the front cell face
               receding into the scene. Default — best depth read.

Output naming: ``<traj-basename>_frame<N>_view_<preset>.png`` under
``assets/<run-name>/figs/``.

Requires ``ovito`` (conda channel https://conda.ovito.org); available
in the ``alchemi-playbook`` conda env.
"""

import argparse
import sys
from pathlib import Path

from ase.io import read

from visualize_warmup_trajectory import (
    find_molecules,
    resolve_trajectory,
    unwrap_frame,
)

HERE = Path(__file__).parent

_TRAJ_SUFFIXES = (".extxyz.gz", ".extxyz", ".xyz.gz", ".xyz")

# camera_dir vectors point in the direction the camera looks; OVITO places
# the camera on the opposite side and ``zoom_all`` frames the scene.
PRESETS = {
    "b": (0.0, 1.0, 0.0),
    "b_tilt": (-0.25, 1.0, -0.25),
}


def _basename_no_traj_ext(path: Path) -> str:
    name = path.name
    for s in _TRAJ_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return path.stem


def _clean_atoms_for_ovito(atoms):
    """Strip Unicode/object arrays OVITO can't accept (pymatgen leftovers, etc)."""
    _supported = {"int8", "int16", "int32", "int64", "float32", "float64"}
    for k in list(atoms.arrays):
        if k in ("numbers", "positions"):
            continue
        if atoms.arrays[k].dtype.name not in _supported:
            del atoms.arrays[k]
    return atoms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory", nargs="?", type=Path, default=None)
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
    p.add_argument("--run-name", default="naphthalene_long_2025")
    p.add_argument("--t-warmup", type=float, default=200.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument(
        "--preset", choices=list(PRESETS), default="b_tilt",
        help="Camera preset. 'b' = head-on along +b (matches the ASE "
             "PIL view_b convention). 'b_tilt' = +b with ~17° tilt "
             "for depth (default).",
    )
    p.add_argument(
        "--projection", choices=["perspective", "ortho"],
        default=None,
        help="Perspective adds foreshortening; ortho is a flat parallel "
             "projection. Default chosen by preset: ortho for 'b', "
             "perspective for 'b_tilt'.",
    )
    p.add_argument(
        "--fov", type=float, default=None,
        help="Perspective FOV in radians (default 0.5 ≈ 29°) or ortho "
             "field-of-view in Angstrom. None = OVITO auto-frame.",
    )
    p.add_argument("--no-unwrap", action="store_true")
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--no-ao", action="store_true")
    p.add_argument("--size", type=int, nargs=2, default=[1600, 1200])
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


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
    frames = read(
        str(args.trajectory),
        index=f"{args.frame}:{args.frame + 1}",
        format="extxyz",
    )
    if not isinstance(frames, list):
        frames = [frames]
    atoms = frames[0]
    print(
        f"  {len(atoms)} atoms; cell lengths = "
        f"{atoms.cell.lengths().tolist()} A"
    )

    if not args.no_unwrap:
        mols = find_molecules(atoms)
        sizes = sorted({len(m) for m in mols})
        print(f"  {len(mols)} molecules; atoms per molecule: {sizes}")
        unwrap_frame(atoms, mols)

    _clean_atoms_for_ovito(atoms)

    # Deferred imports so --help works without the ovito wheel installed.
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import TachyonRenderer, Viewport

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    # Head-on `b` defaults to Ortho so the herringbone pattern reads flat;
    # tilted presets default to Perspective so receding edges show depth.
    default_projection = "ortho" if args.preset == "b" else "perspective"
    projection = args.projection or default_projection
    vp = Viewport(
        type=Viewport.Type.Perspective if projection == "perspective"
        else Viewport.Type.Ortho
    )
    vp.camera_dir = PRESETS[args.preset]
    vp.camera_up = (0.0, 0.0, 1.0)
    if args.fov is not None:
        vp.fov = args.fov
    elif projection == "perspective":
        vp.fov = 0.5  # radians ≈ 29° (slightly tighter than OVITO default)
    vp.zoom_all()

    renderer = TachyonRenderer(
        ambient_occlusion=not args.no_ao,
        shadows=not args.no_shadows,
    )

    if args.out is None:
        figs = HERE / "assets" / args.run_name / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        stem = _basename_no_traj_ext(args.trajectory)
        args.out = figs / f"{stem}_frame{args.frame}_view_{args.preset}.png"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering -> {args.out}")
    vp.render_image(
        filename=str(args.out),
        size=tuple(args.size),
        renderer=renderer,
        background=(1.0, 1.0, 1.0),
    )
    pipeline.remove_from_scene()
    print(f"Done: {args.out.stat().st_size / 1024:.1f} kB")


if __name__ == "__main__":
    main()
