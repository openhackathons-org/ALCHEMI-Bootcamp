"""Visualise a pulled MD trajectory with ASE GUI.

Loads extended XYZ (with per-frame cell info), unwraps each molecule across
PBC so atoms in the same molecule stay contiguous in the viewer, then opens
the ASE GUI for playback. ASE GUI draws the cell box natively and understands
PBC, so the progressive cell expansion during the bare-AIMNet2 collapse is
visible directly.

Usage:
    python visualize_warmup_trajectory.py [PATH]
        [--stage {warmup-fire,warmup-nvt,warmup-npt,
                  melt-nvt,
                  slc-fire,slc-nvt,slc-npt}]
        [--source {nvt,npt}] [--temps 250,300]
        [--run-name NAME] [--t-warmup 200] [--dt 0.5]
        [--stride N] [--no-unwrap]

Trajectory resolution:
  * If PATH is given, use it directly.
  * Otherwise build the basename from --stage / --source / --temps /
    --t-warmup / --dt using the same formula as `pull_trajectory.py`
    (so the same flags pull and visualise the same artefact), and pick
    the first extension that exists in ``assets/<run-name>/traj/`` from
    {``.extxyz``, ``.extxyz.gz``, ``.xyz``}.

`--stage` defaults to `warmup-npt` (the most-equilibrated single-phase
trajectory). `--source` is only used when `--stage` is `melt-nvt` or
`slc-*` and selects which warmup endpoint seeded the melt half. `--temps`
is only used when `--stage` is `slc-*` and selects the multi-GPU subset
suffix (e.g. `250,300` -> `_t250_300`).

Controls once the GUI is open:
    - space / arrow keys: play / frame step
    - "View -> Show bonds": toggle bond drawing
    - "View -> Repeat": tile the cell to see periodic images
    - The MD step for each frame is stored in ``atoms.info["step"]``; the GUI
      shows it in the info panel.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import primitive_neighbor_list
from ase.visualize import view

from pull_trajectory import zarr_basename

HERE = Path(__file__).parent

# Search order: prefer .extxyz (uncompressed, fastest), fall back to .gz, then
# legacy plain .xyz.
EXTS = (".extxyz", ".extxyz.gz", ".xyz")


def resolve_trajectory(
    run_name: str,
    stage: str,
    source: str,
    t_warmup: float,
    dt: float,
    temps: str,
) -> Path:
    """Locate the pulled trajectory matching the (stage, source, t_warmup,
    dt, temps) tuple under ``assets/<run-name>/traj/``. Returns the first
    existing candidate; if none exist, returns the canonical ``.extxyz``
    path so the caller's ``Not found`` message points at the most likely
    target.
    """
    assets = HERE / "assets" / run_name / "traj"
    base = zarr_basename(stage, dt, source, t_warmup, temps)
    for ext in EXTS:
        path = assets / f"{base}{ext}"
        if path.exists():
            return path
    return assets / f"{base}{EXTS[0]}"


def find_molecules(atoms, cutoff: float = 1.55):
    """Connected components under PBC using covalent-range cutoff."""
    i, j = primitive_neighbor_list(
        "ij",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=cutoff,
    )
    n = len(atoms)
    adjacency = [[] for _ in range(n)]
    for a, b in zip(i, j):
        adjacency[a].append(b)

    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        queue = [start]
        labels[start] = next_label
        while queue:
            u = queue.pop()
            for v in adjacency[u]:
                if labels[v] == -1:
                    labels[v] = next_label
                    queue.append(v)
        next_label += 1
    return [np.where(labels == k)[0] for k in range(next_label)]


def unwrap_frame(atoms, molecules):
    """Pull each molecule's atoms next to their reference atom via minimum image.

    Keeps inter-molecular layout unchanged (so the cell expansion is still
    visible), only fixes the intra-molecular split across PBC faces.
    """
    positions = atoms.positions.copy()
    cell = np.asarray(atoms.cell)
    cell_inv = np.linalg.inv(cell)
    for idx in molecules:
        ref = positions[idx[0]]
        for k in idx[1:]:
            disp = positions[k] - ref
            frac = disp @ cell_inv
            frac -= np.round(frac)
            positions[k] = ref + frac @ cell
    atoms.positions = positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the extended-XYZ trajectory (accepts .xyz / .extxyz / .extxyz.gz). "
        "When omitted, the default is built from --stage / --source / "
        "--t-warmup / --dt and resolved under assets/<run-name>/traj/.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "warmup-fire", "warmup-nvt", "warmup-npt",
            "melt-nvt",
            "slc-fire", "slc-nvt", "slc-npt",
        ],
        default="warmup-npt",
        help="Which trajectory to open (default 'warmup-npt'). Mirrors "
        "pull_trajectory.py --stage so the same value pulls and "
        "visualises the same artefact.",
    )
    parser.add_argument(
        "--source",
        choices=["nvt", "npt"],
        default="npt",
        help="For --stage melt-nvt or slc-*: which warmup endpoint seeded "
        "the melt half (matches melt/slc driver --source). Ignored for "
        "warmup-* stages.",
    )
    parser.add_argument(
        "--temps",
        type=str,
        default="",
        help="For --stage slc-*: comma-separated subset suffix matching "
        "slc.py --temps (e.g. '250,300' for GPU 0 of the canonical 4-GPU "
        "split). Empty = full-sweep zarr (no _tN_M suffix). Ignored for "
        "warmup-* and melt-nvt stages.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="naphthalene_long",
        help="Run name; drives the default-trajectory search path under assets/.",
    )
    parser.add_argument(
        "--t-warmup",
        type=float,
        default=200.0,
        help="Warmup target temperature in K (default 200). Used to build the "
        "default-trajectory basename; must match the warmup driver's value.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="MD timestep in fs (default 0.5). Used to build the "
        "default-trajectory basename; must match the driver's value.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (default: 1 = every frame; try 5 for faster playback)",
    )
    parser.add_argument(
        "--no-unwrap",
        action="store_true",
        help="Skip molecular unwrapping (PBC splits will be visible)",
    )
    args = parser.parse_args()
    if args.trajectory is None:
        args.trajectory = resolve_trajectory(
            args.run_name,
            args.stage,
            args.source,
            args.t_warmup,
            args.dt,
            args.temps,
        )

    if not args.trajectory.exists():
        sys.exit(f"Not found: {args.trajectory}")

    print(f"Loading {args.trajectory} (stride={args.stride}) ...")
    frames = read(str(args.trajectory), index=f"::{args.stride}", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    print(f"  -> {len(frames)} frames, {len(frames[0])} atoms each")

    if not args.no_unwrap:
        print("Identifying molecules on frame 0 ...")
        molecules = find_molecules(frames[0])
        sizes = {len(m) for m in molecules}
        print(f"  -> {len(molecules)} molecules; atoms per molecule: {sorted(sizes)}")
        print("Unwrapping each molecule across PBC for every frame ...")
        for a in frames:
            unwrap_frame(a, molecules)

    print("Opening ASE GUI (close the window to exit) ...")
    view(frames)


if __name__ == "__main__":
    main()
