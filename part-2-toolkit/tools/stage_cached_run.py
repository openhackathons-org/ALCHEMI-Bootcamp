"""Stage the shipped Part 2 cache from the dev canonical-run artefacts.

DEV-ONLY: this tool is stripped on merge to ``main`` (it lives under
``tools/``). It produces ``data/cached/naphthalene_orbmol/`` from the local dev
run ``assets/naphthalene_orb_crystal_aniso/`` (the run was produced before the
rename to ``naphthalene_orbmol``).

What it does:

* Thins the dev extended-XYZ trajectories and **strips per-atom forces**
  (and any velocities/momenta) so only positions + species/Z + cell + pbc
  ship — the notebook's OVITO animation and MSD/S₀ analysis need nothing else.
  This is the dominant size win (forces double the per-atom payload). The
  cleaned frames land under ``data/cached/naphthalene_orbmol/traj/`` with
  **clean stems** (the dev ``_200k_dt0p5fs`` / ``packmol_melt_from_npt`` tags
  are dropped) so the §7-§11 notebook ``cfg.cached_extxyz(stem)`` calls resolve.
  NPT-class trajectories (warmup_npt, slc_npt_t*) are thinned to every
  ``--npt-stride`` frames (default 10 ≈ one frame per 1000 MD steps at the
  100-step SnapshotHook spacing). FIRE/NVT trajectories are short and kept
  whole.
* Stages the ``LoggingHook`` CSV logs under
  ``data/cached/naphthalene_orbmol/csv/`` with clean stems (``.partN``
  preserved). Diagnostic figures are now rendered **live** in the notebook
  from these CSVs, so no PNGs are copied.

Frame counting streams via ``ase.io.iread`` so the 392 MB warmup-NPT file is
never fully loaded just to size it; the strided ``read`` then materialises only
the kept frames.

Usage::

    python tools/stage_cached_run.py --stage warmup   # default
    python tools/stage_cached_run.py --stage slc
    python tools/stage_cached_run.py --stage all
    python tools/stage_cached_run.py --npt-stride 20  # coarser NPT thinning

Run from ``part-2-toolkit/`` so the relative dest paths land correctly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ase.io import iread, read, write

# ── Run identity ─────────────────────────────────────────────────────────
RUN_DEV = "naphthalene_orb_crystal_aniso"  # source dev run name
RUN_CLEAN = "naphthalene_orbmol"  # dest under data/cached/<RUN_CLEAN>/

# ── Trajectory mappings: clean_stem -> dev filename ──────────────────────
WARMUP_TRAJ = {
    "warmup_nvt": "warmup_nvt_200k_dt0p5fs.extxyz",
    "warmup_npt": "warmup_npt_200k_dt0p5fs.extxyz",
}

# SLC trajectory staging:
#   * slc_npt_t<T> <- dev "all" file (the full SLC NPT trajectory), thinned
#     by --npt-stride (NPT-class -> ~60 frames at the 1000-step source spacing).
#   * slc_nvt_t<T> <- dev "nvt" file (short; all frames kept, forces stripped).
#   * slc_fire     <- dev "fire" t200 file ONLY (single shared FIRE endpoint;
#     the FIRE-minimised stack is T-independent, so no per-T copies).
SLC_TRAJ = {
    f"slc_{stage_clean}_t{T}": f"slc_{stage_dev}_packmol_melt_from_npt_200k_dt0p5fs_t{T}.extxyz"
    for T in (200, 300, 400, 500)
    for stage_clean, stage_dev in (("nvt", "nvt"), ("npt", "all"))
}
SLC_TRAJ["slc_fire"] = "slc_fire_packmol_melt_from_npt_200k_dt0p5fs_t200.extxyz"

# ── CSV log mappings: clean_dest -> dev filename (under logs/<RUN_DEV>/) ──
# Clean stems strip the ``_200k_dt0p5fs`` tag but PRESERVE ``.partN``.
WARMUP_CSV = {
    "warmup_fire.csv": "warmup_fire_dt0p5fs.csv",
    "warmup_nvt.csv": "warmup_nvt_200k_dt0p5fs.csv",
    "warmup_npt.csv": "warmup_npt_200k_dt0p5fs.csv",
    "warmup_npt.part2.csv": "warmup_npt_200k_dt0p5fs.part2.csv",
}

# Only the SLC NPT logs feed the §11 diagnostic figures; the multi-part
# ``.partN`` chain (per T) is preserved. slc_fire/slc_nvt CSVs are not staged.
SLC_CSV = {
    f"slc_npt_t{T}{part_clean}.csv": f"slc_all_packmol_melt_from_npt_200k_dt0p5fs_t{T}{part_dev}.csv"
    for T in (200, 300, 400, 500)
    for part_clean, part_dev in (("", ""), (".part2", ".part2"), (".part3", ".part3"), (".part4", ".part4"))
}

# NPT-class stems thinned by ``--npt-stride``; everything else kept whole
# (or thinned to ``--target-frames`` if it exceeds that).
_NPT_CLASS = ("warmup_npt", "slc_npt_t")


def _is_npt_class(clean_stem: str) -> bool:
    return any(clean_stem.startswith(prefix) for prefix in _NPT_CLASS)


def _strip_dynamics(frame) -> None:
    """Drop forces / velocities / momenta and detach any calculator in-place."""
    frame.calc = None
    frame.arrays.pop("forces", None)
    frame.arrays.pop("momenta", None)
    frame.arrays.pop("velocities", None)


def _thin_trajectory(src: Path, dst: Path, index: str, label: str) -> None:
    """Read ``src`` at ``index``, strip dynamics fields, and write to ``dst``."""
    n = sum(1 for _ in iread(str(src)))
    frames = read(str(src), index=index)
    if not isinstance(frames, list):
        frames = [frames]
    for frame in frames:
        _strip_dynamics(frame)
    write(str(dst), frames)
    print(f"  {label}: {src.name} ({n} frames) [{index}] -> {dst.name} ({len(frames)} frames)")


def stage_trajectories(
    mapping: dict[str, str],
    traj_dir: Path,
    dest_dir: Path,
    *,
    target_frames: int,
    npt_stride: int,
) -> None:
    """Stage a clean_stem->dev-filename trajectory mapping into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for clean_stem, dev_name in mapping.items():
        src = traj_dir / dev_name
        if not src.exists():
            print(f"  WARNING: missing source trajectory, skipping: {src}")
            continue
        dst = dest_dir / f"{clean_stem}.extxyz"
        if _is_npt_class(clean_stem):
            index = f"::{npt_stride}"
        else:
            n = sum(1 for _ in iread(str(src)))
            stride = max(1, n // target_frames)
            index = f"::{stride}"
        _thin_trajectory(src, dst, index, clean_stem)


def stage_csv(mapping: dict[str, str], logs_dir: Path, dest_dir: Path) -> None:
    """Copy a clean_dest->dev-filename CSV mapping into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for dest_name, dev_name in mapping.items():
        src = logs_dir / dev_name
        if not src.exists():
            print(f"  WARNING: missing source CSV, skipping: {src}")
            continue
        dst = dest_dir / dest_name
        shutil.copyfile(src, dst)
        print(f"  csv: {src.name} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("warmup", "slc", "all"), default="warmup")
    parser.add_argument(
        "--target-frames",
        type=int,
        default=120,
        help="target frames for non-NPT trajectories thinned to a frame budget",
    )
    parser.add_argument(
        "--npt-stride",
        type=int,
        default=10,
        help="keep every Nth frame of NPT-class trajectories (default 10 ≈ 1000 MD steps)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("assets") / RUN_DEV,
        help=f"source dev run dir (default assets/{RUN_DEV})",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs") / RUN_DEV,
        help=f"source dev logs dir (default logs/{RUN_DEV})",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    traj_dir = run_dir / "traj"
    logs_dir: Path = args.logs_dir

    cache_root = Path("data") / "cached" / RUN_CLEAN
    cache_traj_dir = cache_root / "traj"
    cache_csv_dir = cache_root / "csv"

    do_warmup = args.stage in ("warmup", "all")
    do_slc = args.stage in ("slc", "all")

    if do_warmup:
        print(f"[stage warmup] trajectories -> {cache_traj_dir}")
        stage_trajectories(
            WARMUP_TRAJ,
            traj_dir,
            cache_traj_dir,
            target_frames=args.target_frames,
            npt_stride=args.npt_stride,
        )
        print(f"[stage warmup] CSV logs -> {cache_csv_dir}")
        stage_csv(WARMUP_CSV, logs_dir, cache_csv_dir)

    if do_slc:
        print(f"[stage slc] trajectories -> {cache_traj_dir}")
        stage_trajectories(
            SLC_TRAJ,
            traj_dir,
            cache_traj_dir,
            target_frames=args.target_frames,
            npt_stride=args.npt_stride,
        )
        print(f"[stage slc] CSV logs -> {cache_csv_dir}")
        stage_csv(SLC_CSV, logs_dir, cache_csv_dir)

    print("done.")


if __name__ == "__main__":
    main()
