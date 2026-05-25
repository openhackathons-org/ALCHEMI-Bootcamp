"""Pull a warmup trajectory off the remote container as extxyz.

Local-side wrapper around ``export_zarr_to_extxyz.py`` (which runs only
inside the Part-2 container, since it imports nvalchemi-toolkit helpers).
This script:

1. Reads ``/tmp/alchemi-playbook-part2-deploy.env`` for LOGIN_HOST/COMPUTE_NODE.
2. Runs ``export_zarr_to_extxyz.py`` inside the container via
   ``ssh -J $LOGIN_HOST $COMPUTE_NODE docker exec ...`` to convert
   ``/workspace/logs/<run-name>/warmup_<stage>_...zarr`` -> ``/tmp/<name>.extxyz.gz``.
3. Pulls the resulting extxyz back to local ``assets/<run-name>/traj/``
   via the canonical ``docker exec tar czf -`` pattern.
4. Cleans up the /tmp staging file inside the container.

Usage::

    python pull_trajectory.py [--run-name naphthalene_long_2025]
                              [--stage {warmup-fire,warmup-nvt,warmup-npt,
                                        melt-nvt,
                                        slc-fire,slc-nvt,slc-npt}]
                              [--dt 0.5] [--t-warmup 200]
                              [--source {nvt,npt}]   # melt-* + slc-* stages
                              [--temps 250,300]      # slc-* multi-GPU subset
                              [--no-gzip]

After pulling, view with::

    python visualize_warmup_trajectory.py --run-name naphthalene_long_2025
"""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DEPLOY_ENV = Path("/tmp/alchemi-playbook-part2-deploy.env")
CONTAINER = "alchemi-playbook-part2"
REMOTE_WORKSPACE = "/workspace"


def read_deploy_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def zarr_basename(
    stage: str,
    dt: float,
    source: str = "nvt",
    t_warmup: float = 200.0,
    temps: str = "",
) -> str:
    """Match the warmup/melt/slc driver file-stem conventions.

    See ``warmup.py`` (FIRE/NVT/NPT),
    ``melt.py`` (melt — carries ``from_<source>`` and ``T_WARMUP_TAG``),
    and ``slc.py`` (the three SLC stages). The SLC production NPT
    writes its trajectory under the
    ``slc_all_*`` stem (legacy, matches the notebook's diagnostics
    loader); checkpoint key is ``slc_npt_*`` -- the ``slc-npt`` stage
    here resolves to the trajectory stem, not the checkpoint key.

    For ``slc-*`` stages, ``temps`` selects the multi-GPU subset suffix:
    e.g. ``"250,300"`` -> ``_t250_300`` (matches the wrapper's split).
    Empty ``temps`` means full-sweep (no suffix).
    """
    dt_tag = f"dt{str(dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(t_warmup)}k"
    temps_tag = f"_t{temps.replace(',', '_')}" if temps else ""
    if stage == "warmup-fire":
        return f"warmup_fire_{dt_tag}"
    if stage == "warmup-nvt":
        return f"warmup_nvt_{t_warmup_tag}_{dt_tag}"
    if stage == "warmup-npt":
        return f"warmup_npt_{t_warmup_tag}_{dt_tag}"
    if stage == "melt-nvt":
        return f"melt_nvt_500k_from_{source}_{t_warmup_tag}_{dt_tag}"
    if stage == "slc-fire":
        return f"slc_fire_from_{source}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    if stage == "slc-nvt":
        return f"slc_nvt_from_{source}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    if stage == "slc-npt":
        return f"slc_all_from_{source}_{t_warmup_tag}_{dt_tag}{temps_tag}"
    raise ValueError(f"Unknown stage {stage!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-name",
        default="naphthalene_long",
        help="Run subdirectory under logs/ on remote and assets/ locally "
        "(default 'naphthalene_long').",
    )
    p.add_argument(
        "--stage",
        choices=[
            "warmup-fire", "warmup-nvt", "warmup-npt",
            "melt-nvt",
            "slc-fire", "slc-nvt", "slc-npt",
        ],
        default="warmup-npt",
        help="Which trajectory to pull, named as <phase>-<integrator>: "
        "warmup-{fire,nvt,npt} = the three warmup stages; "
        "melt-nvt = melt-phase NVT @ T_MELT (requires --source); "
        "slc-{fire,nvt,npt} = the three SLC stages "
        "(use --temps to select the multi-GPU subset; require --source).",
    )
    p.add_argument(
        "--source",
        choices=["nvt", "npt"],
        default="nvt",
        help="For --stage melt-nvt or slc-*: which warmup endpoint seeded "
        "the melt half (matches melt/slc driver --source). Ignored for "
        "warmup-* stages.",
    )
    p.add_argument(
        "--temps",
        default="",
        help="For --stage slc-*: comma-separated subset of TEMPS matching "
        "slc.py --temps (e.g. '250,300' for GPU 0 of the canonical 4-GPU "
        "split). Empty = full-sweep zarr (no _tN_M suffix). Ignored for "
        "warmup/melt stages.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="MD timestep in fs (default 0.5). Used for DT_TAG and "
        "stamped into each frame's info['time_fs'].",
    )
    p.add_argument(
        "--t-warmup",
        type=float,
        default=200.0,
        help="Warmup target temperature in K (default 200). Must match the "
        "value passed to warmup.py so the right zarr is addressed.",
    )
    p.add_argument(
        "--no-gzip",
        dest="gzip",
        action="store_false",
        help="Write plain .extxyz (default: gzip to .extxyz.gz).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = read_deploy_env(DEPLOY_ENV)
    login, node = env["LOGIN_HOST"], env["COMPUTE_NODE"]

    stem = zarr_basename(
        args.stage, args.dt, args.source, args.t_warmup, args.temps,
    )
    ext = ".extxyz.gz" if args.gzip else ".extxyz"
    out_name = f"{stem}{ext}"

    remote_zarr = f"{REMOTE_WORKSPACE}/logs/{args.run_name}/{stem}.zarr"
    remote_tmp = f"/tmp/{out_name}"
    local_dir = HERE / "assets" / args.run_name / "traj"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_out = local_dir / out_name

    print(f"Converting (in container): {remote_zarr}")
    print(f"                      -> {remote_tmp}")
    subprocess.check_call(
        [
            "ssh",
            "-J",
            login,
            node,
            "docker",
            "exec",
            CONTAINER,
            "python",
            f"{REMOTE_WORKSPACE}/export_zarr_to_extxyz.py",
            remote_zarr,
            remote_tmp,
            "--dt-fs",
            str(args.dt),
        ]
    )

    print(f"Pulling -> {local_out}")
    # tar cf (uncompressed) -- the payload is already gzipped, so wrapping
    # in tar.gz adds CPU on both sides for ~zero size reduction. The CLAUDE.md
    # canonical pattern is `tar czf -` but it's tuned for zarr directories;
    # for single already-compressed files `tar cf -` is faster.
    pull = subprocess.Popen(
        [
            "ssh",
            "-J",
            login,
            node,
            "docker",
            "exec",
            CONTAINER,
            "tar",
            "cf",
            "-",
            "-C",
            "/tmp",
            out_name,
        ],
        stdout=subprocess.PIPE,
    )
    subprocess.check_call(["tar", "xf", "-", "-C", str(local_dir)], stdin=pull.stdout)
    pull.wait()
    if pull.returncode != 0:
        raise SystemExit(f"ssh/docker-exec tar failed with rc={pull.returncode}")

    if not local_out.exists():
        raise SystemExit(f"Expected {local_out} not materialised after pull")

    # Clean up remote staging file.
    subprocess.run(
        ["ssh", "-J", login, node, "docker", "exec", CONTAINER, "rm", "-f", remote_tmp],
        check=False,
    )

    size_mb = local_out.stat().st_size / 1e6
    print(f"Done: {local_out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
