"""Export a warmup Zarr trajectory to extended-XYZ for ASE-GUI playback.

The warmup pipeline writes per-stage snapshots to Zarr sinks (e.g.
``logs/naphthalene_long/warmup_nvt_200k.zarr``). Reading Zarr requires
``nvalchemi-toolkit``, which in this project is container-only. This script
runs inside the Part-2 container, converts the Zarr to extxyz (optionally
gzipped), and the result can then be pulled to any host and opened by
``visualize_warmup_trajectory.py``.

Usage (inside alchemi-playbook-part2):
    python export_zarr_to_extxyz.py <zarr-path> <out-path> \\
           [--snapshot-every N] [--dt-fs DT]

If ``out-path`` ends in ``.gz`` the output is gzip-compressed via
``gzip.open``. ``atoms.info["step"]`` is stamped as ``i*snapshot_every``
so the ASE-GUI info panel shows the MD step. ``--dt-fs`` additionally
stamps ``atoms.info["time_fs"]``.
"""

import argparse
import gzip
from pathlib import Path

from ase.io import write as ase_write

import _path  # noqa: F401  # parent dir on sys.path for `helpers` import
from helpers import batch_to_ase, load_zarr_trajectory, zarr_trajectory_length


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zarr_path", type=Path, help="Input Zarr store")
    p.add_argument("out_path", type=Path, help="Output extxyz path (.extxyz or .extxyz.gz)")
    p.add_argument("--snapshot-every", type=int, default=100,
                   help="Steps between snapshots (matches warmup SNAPSHOT_EVERY)")
    p.add_argument("--dt-fs", type=float, default=None,
                   help="MD timestep in fs; if given, also stamp info['time_fs']")
    args = p.parse_args()

    if not args.zarr_path.exists():
        raise SystemExit(f"Not found: {args.zarr_path}")

    n = zarr_trajectory_length(args.zarr_path)
    print(f"Loading {args.zarr_path} ({n} frames) ...")
    batches = load_zarr_trajectory(args.zarr_path, device="cpu")

    frames = []
    for i, batch in enumerate(batches):
        atoms = batch_to_ase(batch)
        step = i * args.snapshot_every
        atoms.info["step"] = step
        if args.dt_fs is not None:
            atoms.info["time_fs"] = step * args.dt_fs
        frames.append(atoms)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(frames)} frames to {args.out_path} ...")
    if args.out_path.suffix == ".gz":
        with gzip.open(args.out_path, "wt") as f:
            ase_write(f, frames, format="extxyz")
    else:
        ase_write(str(args.out_path), frames, format="extxyz")
    print(f"Done: {args.out_path} ({args.out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
