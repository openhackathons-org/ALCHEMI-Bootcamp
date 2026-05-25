"""Faster trajectory pull: tar the remote zarr directory and convert locally.

Drop-in alternative to ``pull_trajectory.py`` for cases where the in-container
``export_zarr_to_extxyz.py`` step is the wall-time bottleneck (long
trajectories or large supercells). Pulls the raw zarr directory via the
canonical ``ssh -J ... docker exec ... tar czf - | tar xzf -`` pattern
(small payload because zarr ships compressed numpy arrays), then converts
to extxyz on the host using ``zarr_to_extxyz.py`` (no nvalchemi-toolkit
dependency on the host).

CLI mirrors ``pull_trajectory.py`` so the same flags select the same
artefact. Final extxyz lands at the same path
``assets/<run-name>/traj/<stem><ext>`` as the original pull, so downstream
visualisers/analysers don't notice the difference.

Usage::

    python pull_trajectory_zarr.py \\
        --run-name naphthalene_long_2025 --stage slc-npt \\
        --source npt --t-warmup 100 --temps 250,300 --no-gzip
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from pull_trajectory import read_deploy_env, zarr_basename

HERE = Path(__file__).parent
DEPLOY_ENV = Path("/tmp/alchemi-playbook-part2-deploy.env")
CONTAINER = "alchemi-playbook-part2"
REMOTE_WORKSPACE = "/workspace"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-name", default="naphthalene_long")
    p.add_argument(
        "--stage",
        choices=[
            "warmup-fire", "warmup-nvt", "warmup-npt",
            "melt-nvt",
            "slc-fire", "slc-nvt", "slc-npt",
        ],
        default="warmup-npt",
    )
    p.add_argument("--source", choices=["nvt", "npt"], default="nvt")
    p.add_argument("--temps", default="")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--t-warmup", type=float, default=200.0)
    p.add_argument(
        "--no-gzip",
        dest="gzip",
        action="store_false",
        help="Write plain .extxyz (default: gzip to .extxyz.gz).",
    )
    p.add_argument(
        "--keep-zarr",
        action="store_true",
        help="Don't delete the local zarr after extxyz conversion. Useful "
        "for re-running the conversion with different stamps without "
        "pulling again.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = read_deploy_env(DEPLOY_ENV)
    login, node = env["LOGIN_HOST"], env["COMPUTE_NODE"]

    stem = zarr_basename(
        args.stage, args.dt, args.source, args.t_warmup, args.temps,
    )
    remote_zarr = f"{REMOTE_WORKSPACE}/logs/{args.run_name}/{stem}.zarr"
    ext = ".extxyz.gz" if args.gzip else ".extxyz"
    out_name = f"{stem}{ext}"
    local_traj_dir = HERE / "assets" / args.run_name / "traj"
    local_traj_dir.mkdir(parents=True, exist_ok=True)
    local_out = local_traj_dir / out_name
    local_zarr_dir = HERE / "logs" / args.run_name
    local_zarr_dir.mkdir(parents=True, exist_ok=True)
    local_zarr = local_zarr_dir / f"{stem}.zarr"

    # --- 1. Pull the zarr directory via tar+ssh ---------------------------
    print(f"[1/2] Pulling {remote_zarr}")
    print(f"      -> {local_zarr}")
    if local_zarr.exists():
        shutil.rmtree(local_zarr)
    t0 = time.monotonic()
    pull_cmd = (
        f'ssh -J {login} {node} "docker exec {CONTAINER} '
        f'tar czf - -C {REMOTE_WORKSPACE}/logs/{args.run_name} {stem}.zarr 2>/dev/null" '
        f'| tar xzf - -C {local_zarr_dir}'
    )
    rc = subprocess.call(pull_cmd, shell=True)
    if rc != 0:
        raise SystemExit(f"zarr pull failed (rc={rc})")
    if not local_zarr.exists():
        raise SystemExit(f"local zarr not materialised at {local_zarr}")
    pull_size_mb = sum(p.stat().st_size for p in local_zarr.rglob("*") if p.is_file()) / 1e6
    print(f"      pulled {pull_size_mb:.1f} MB in {time.monotonic() - t0:.1f}s")

    # --- 2. Convert zarr -> extxyz locally --------------------------------
    print(f"[2/2] Converting -> {local_out}")
    t1 = time.monotonic()
    convert_cmd = [
        "/Users/ryreese/miniforge3/envs/alchemi-playbook/bin/python",
        str(HERE / "zarr_to_extxyz.py"),
        str(local_zarr),
        str(local_out),
        "--dt-fs", str(args.dt),
    ]
    subprocess.check_call(convert_cmd)
    print(f"      conversion took {time.monotonic() - t1:.1f}s")

    # --- 3. Cleanup local zarr (default) ----------------------------------
    if not args.keep_zarr:
        shutil.rmtree(local_zarr)
        print(f"Removed {local_zarr} (pass --keep-zarr to retain)")

    out_size_mb = local_out.stat().st_size / 1e6
    print(f"\nDone: {local_out} ({out_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
