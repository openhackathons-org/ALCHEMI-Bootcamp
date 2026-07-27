"""Failure records for campaign and H100 tuning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import record_part1_campaign_failure as recorder  # noqa: E402


def test_tuning_failure_keeps_the_actual_slurm_producer(tmp_path: Path) -> None:
    case_log = tmp_path / "pipeline_2gpu-b512-r1.log"
    case_log.write_text("worker failed\n", encoding="utf-8")
    output = tmp_path / "pipeline_2gpu-b512-r1.json"
    environment = {
        "SLURM_NNODES": "2",
        "SLURM_JOB_ID": "24680",
        "SLURM_JOB_PARTITION": "h100-nvl@ts3/example/1gpu-32cpu-128gb",
        "PYTHONHASHSEED": "0",
    }
    args = argparse.Namespace(
        purpose="tuning",
        route="pipeline_2gpu",
        systems=4096,
        batch_size=512,
        fire_fmax=0.01,
        nvt_steps=10,
        nve_steps=20,
        dt_fs=0.5,
        temperature_k=75.0,
        friction_per_fs=0.01,
        comm_mode="async_recv",
        repeat=1,
        exit_code=17,
        case_log=case_log,
        repository_root=REPOSITORY_ROOT,
        slurm_producer="scripts/slurm_part1_pipeline_tuning.sbatch",
        output=output,
    )

    assert (
        recorder.main(
            args,
            environment=environment,
            gpu_name="NVIDIA H100 NVL",
            torch_version="2.12.0+cu130",
            repository_commit="a" * 40,
        )
        == 0
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["success"] is False
    assert record["purpose"] == "tuning"
    assert record["python_version"] == platform.python_version()
    assert record["systems_requested"] == 4096
    assert "-s4096-b512-async_recv-r1" in record["run_id"]
    assert "scripts/slurm_part1_pipeline_tuning.sbatch" in record["producer_set"]
    assert (
        "scripts/slurm_part1_distributed_campaign.sbatch" not in record["producer_set"]
    )
