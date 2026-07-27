#!/usr/bin/env python3
"""Write one failed raw record when a campaign ``srun`` exits early."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from part1_distributed_campaign_contract import (
    ATOMS_PER_SYSTEM,
    CAMPAIGN_SEED,
    CHARGE_ATOL_E,
    CORE_BRANCH,
    CORE_COMMIT,
    CORE_VERSION,
    COVALENT_OH_CUTOFF_A,
    ENERGY_ATOL_EV,
    FORCE_ATOL_EV_PER_A,
    MAX_ABS_NET_CHARGE_E,
    MIN_INTERATOMIC_DISTANCE_A,
    OPS_COMMIT,
    OPS_VERSION,
    OXYGEN_CONNECTIVITY_CUTOFF_A,
    PRODUCER_FILES,
    ROUTE_PAIR_BOUNDARIES,
    ROUTE_TOPOLOGY,
    RUN_SCHEMA,
    TIMING_BOUNDARY,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--purpose",
        choices=("campaign", "tuning"),
        default="campaign",
    )
    parser.add_argument("--route", choices=tuple(ROUTE_TOPOLOGY), required=True)
    parser.add_argument("--systems", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fire-fmax", type=float, required=True)
    parser.add_argument("--nvt-steps", type=int, required=True)
    parser.add_argument("--nve-steps", type=int, required=True)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--friction-per-fs", type=float, required=True)
    parser.add_argument("--comm-mode", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--case-log", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--slurm-producer",
        default="scripts/slurm_part1_distributed_campaign.sbatch",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gpu_name() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    )
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if not names or len(set(names)) != 1:
        raise RuntimeError(f"could not identify one GPU model: {names!r}")
    return names[0]


def _torch_version() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _producer_set(root: Path, slurm_producer: str) -> dict[str, str]:
    candidate = Path(slurm_producer)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Slurm producer must be a safe repository-relative path")
    campaign_slurm = "scripts/slurm_part1_distributed_campaign.sbatch"
    producer_files = tuple(
        slurm_producer if relative == campaign_slurm else relative
        for relative in PRODUCER_FILES
    )
    return {relative: _sha256(root / relative) for relative in producer_files}


def _workload(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    builder = root / "part-1-scalable-atomistic-workflows" / "aux" / "structures.py"
    return {
        "campaign_definition_sha256": None,
        "source_structure": "generated cyclic water hexamer",
        "structure_builder_file": builder.relative_to(root).as_posix(),
        "structure_builder_sha256": _sha256(builder),
        "systems_total": args.systems,
        "atoms_per_system": ATOMS_PER_SYSTEM,
        "batch_size": args.batch_size,
        "campaign_seed": CAMPAIGN_SEED,
        "perturbation_description": (
            "Eight O-O levels from 2.72 to 2.86 A in 0.02 A steps; one "
            "seeded generator adds isotropic 0.025 A Gaussian coordinate noise "
            "in global campaign-ID order, then removes the mean displacement"
        ),
        "fire_fmax_ev_per_a": args.fire_fmax,
        "nvt_steps": args.nvt_steps,
        "nve_steps": args.nve_steps,
        "dt_fs": args.dt_fs,
        "temperature_k": args.temperature_k,
        "friction_per_fs": args.friction_per_fs,
        "velocity_seed_rule": "910000 + campaign_id",
        "pipeline_partition_rule": (
            "4 GPUs: campaign_id % 2; 1 or 2 GPUs: all campaign IDs"
        ),
        "pipeline_pair_boundaries": {
            route: [list(pair) for pair in pairs]
            for route, pairs in ROUTE_PAIR_BOUNDARIES.items()
        },
        "comm_mode": args.comm_mode,
        "stage_names": ["FIRE2", "NVTLangevin", "NVE"],
    }


def main(
    args: argparse.Namespace | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    gpu_name: str | None = None,
    torch_version: str | None = None,
    repository_commit: str | None = None,
) -> int:
    """Write one failure record from CLI or explicitly supplied run facts."""

    args = _parse_args() if args is None else args
    environment = os.environ if environment is None else environment
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    case_log = args.case_log.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw record: {output}")
    if not case_log.is_file():
        raise FileNotFoundError(f"case log not found: {case_log}")
    if args.exit_code <= 0 or args.repeat <= 0:
        raise ValueError("exit code and repeat must be positive")

    topology = ROUTE_TOPOLOGY[args.route]
    nodes = int(environment["SLURM_NNODES"])
    if nodes != topology["world_size"]:
        raise RuntimeError("Slurm node count does not match the failed route")

    null_result_fields = {
        "systems_completed": None,
        "unique_systems_completed": None,
        "missing_systems": None,
        "duplicate_systems": None,
        "unexpected_systems": None,
        "stage_1_completions": None,
        "stage_2_completions": None,
        "max_energy_difference_ev": None,
        "max_force_difference_ev_per_a": None,
        "max_charge_difference_e": None,
        "max_abs_net_charge_e": None,
        "max_handoff_fmax_ev_per_a": None,
        "min_interatomic_distance_a": None,
        "max_relaxation_steps_observed": None,
        "nvt_steps_verified": None,
        "nve_steps_verified": None,
        "covalent_oh_gate_passed": None,
        "oxygen_connectivity_gate_passed": None,
        "rank_audits": None,
        "elapsed_s": None,
        "systems_per_s": None,
        "peak_memory_bytes_max_rank": None,
    }
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "failed",
        "success": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": (
            f"{environment['SLURM_JOB_ID']}-{args.purpose}-{args.route}-"
            f"s{args.systems}-b{args.batch_size}-{args.comm_mode}-r{args.repeat}"
        ),
        "slurm_job_id": environment["SLURM_JOB_ID"],
        "route": args.route,
        "purpose": args.purpose,
        "repeat": args.repeat,
        "error_type": "SrunExitCode",
        "error": (
            f"srun exited with status {args.exit_code}; inspect {case_log.name} "
            "and the Slurm job log"
        ),
        "nodes": nodes,
        "gpu_count": nodes,
        "rank_count": nodes,
        "pipeline_count": topology["pipeline_count"],
        "systems_requested": args.systems,
        "correctness_passed": False,
        **null_result_fields,
        "gpu_name": _gpu_name() if gpu_name is None else gpu_name,
        "backend": "nccl",
        "hostname_rank0": platform.node(),
        "torch_version": (_torch_version() if torch_version is None else torch_version),
        "python_version": platform.python_version(),
        "partition": environment.get("SLURM_JOB_PARTITION"),
        "python_hash_seed": environment.get("PYTHONHASHSEED"),
        "toolkit_core_commit": CORE_COMMIT,
        "toolkit_core_branch": CORE_BRANCH,
        "toolkit_core_clean": True,
        "toolkit_core_version": CORE_VERSION,
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_ops_version": OPS_VERSION,
        "producer_set": _producer_set(root, args.slurm_producer),
        "repository_commit": (
            _git_head(root) if repository_commit is None else repository_commit
        ),
        "model": None,
        "workload": _workload(args, root),
        "timing_boundary": TIMING_BOUNDARY,
        "correctness_checks": None,
        "correctness_tolerances": {
            "energy_atol_ev": ENERGY_ATOL_EV,
            "force_atol_ev_per_a": FORCE_ATOL_EV_PER_A,
            "charge_atol_e": CHARGE_ATOL_E,
            "max_abs_net_charge_e": MAX_ABS_NET_CHARGE_E,
            "max_handoff_fmax_ev_per_a": args.fire_fmax,
            "min_interatomic_distance_a": MIN_INTERATOMIC_DISTANCE_A,
            "covalent_oh_cutoff_a": COVALENT_OH_CUTOFF_A,
            "oxygen_connectivity_cutoff_a": OXYGEN_CONNECTIVITY_CUTOFF_A,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
