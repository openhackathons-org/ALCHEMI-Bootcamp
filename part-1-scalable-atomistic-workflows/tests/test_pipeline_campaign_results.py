"""Tests for the prerecorded Compute Lab pipeline campaign bundle."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import sha256_file  # noqa: E402
from aux.pipeline_campaign_results import (  # noqa: E402
    BUNDLE_SCHEMA,
    FIXED_SYSTEMS_TOTAL,
    ROUTE_SPECS,
    RUN_COLUMNS,
    RUN_SCHEMA,
    PipelineCampaignError,
    canonical_json_sha256,
    correctness_record_sha256,
    load_pipeline_campaign_bundle,
    producer_set_sha256,
    runtime_identity_sha256,
    toolkit_runtime_record_sha256,
)


CORE_BRANCH = "0.2.0-rc"
CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
CORE_VERSION = "0.2.0"
OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
OPS_VERSION = "0.4.0"
REPOSITORY_COMMIT = "b" * 40
SYSTEMS_TOTAL = FIXED_SYSTEMS_TOTAL
REPEATS = 5
MODEL = {
    "checkpoint_source": "runtime cache",
    "checkpoint_sha256": "1" * 64,
    "d3_parameter_sha256": "2" * 64,
    "d3_bj_parameters": {"a1": 0.37, "a2": 4.1, "s6": 1.0, "s8": 1.5},
    "components": [
        "AIMNet2 B97-3c residual",
        "finite all-pairs Coulomb",
        "pairwise D3(BJ)",
    ],
    "dtype": (
        "float32 positions, velocities, forces, AIMNet, and Coulomb pair math; "
        "float64 Coulomb energy accumulation"
    ),
    "eager": True,
}
WORKLOAD = {
    "campaign_definition_sha256": "3" * 64,
    "source_structure": "generated cyclic water hexamer",
    "structure_builder_file": "part-1-scalable-atomistic-workflows/aux/structures.py",
    "structure_builder_sha256": "4" * 64,
    "systems_total": SYSTEMS_TOTAL,
    "atoms_per_system": 18,
    "batch_size": 16,
    "campaign_seed": 20260714,
    "perturbation_description": "Deterministic small displacements.",
    "pipeline_partition_rule": (
        "4 GPUs: campaign_id % 2; 1 or 2 GPUs: all campaign IDs"
    ),
    "pipeline_pair_boundaries": {
        "fused_1gpu": [],
        "pipeline_2gpu": [[0, 1]],
        "pipeline_4gpu": [[0, 1], [2, 3]],
    },
    "fire_fmax_ev_per_a": 0.01,
    "nvt_steps": 10,
    "nve_steps": 20,
    "dt_fs": 0.5,
    "temperature_k": 300.0,
    "friction_per_fs": 0.01,
    "velocity_seed_rule": "910000 + campaign_id",
    "comm_mode": "async_recv",
    "stage_names": ["FIRE2", "NVTLangevin", "NVE"],
}
CORRECTNESS = {
    "reference_route": "fused_1gpu",
    "required_checks": [
        "same system IDs",
        "each system completes both pipeline stages",
        "final energy, force, and charge reevaluation matches",
        "chemistry gates pass",
    ],
    "energy_atol_ev": 5.0e-5,
    "force_atol_ev_per_a": 5.0e-5,
    "charge_atol_e": 5.0e-6,
    "max_abs_net_charge_e": 5.0e-5,
    "max_handoff_fmax_ev_per_a": 0.01,
    "min_interatomic_distance_a": 0.55,
    "covalent_oh_cutoff_a": 1.25,
    "oxygen_connectivity_cutoff_a": 4.0,
}
PRODUCER_FILES = {
    "scripts/benchmark_part1_distributed_campaign.py": "5" * 64,
    "scripts/part1_distributed_campaign_contract.py": "6" * 64,
    "scripts/record_part1_campaign_failure.py": "7" * 64,
    "scripts/run_part1_distributed_torchrun.sh": "9" * 64,
    "scripts/slurm_part1_distributed_campaign.sbatch": "a" * 64,
    "part-1-scalable-atomistic-workflows/aux/artifacts.py": "b" * 64,
    "part-1-scalable-atomistic-workflows/aux/checkpoint.py": "c" * 64,
    "part-1-scalable-atomistic-workflows/aux/electrostatics.py": "d" * 64,
    "part-1-scalable-atomistic-workflows/aux/hooks.py": "f" * 64,
    "part-1-scalable-atomistic-workflows/aux/runtime.py": "e" * 64,
    "part-1-scalable-atomistic-workflows/aux/structures.py": "f" * 64,
}


def _run_rows() -> pd.DataFrame:
    elapsed_by_route = {
        "fused_1gpu": (12.0, 11.5, 11.0, 10.5, 10.0),
        "pipeline_2gpu": (6.0, 5.75, 5.5, 5.25, 5.0),
        "pipeline_4gpu": (3.0, 2.875, 2.75, 2.625, 2.5),
    }
    rows: list[dict[str, Any]] = []
    for route, spec in ROUTE_SPECS.items():
        for repeat, elapsed_s in enumerate(elapsed_by_route[route], start=1):
            rows.append(
                {
                    "schema": RUN_SCHEMA,
                    "run_id": f"job-{route}-r{repeat}",
                    "timestamp_utc": "2026-07-14T18:00:00+00:00",
                    "slurm_job_id": str(4100 + spec["gpu_count"]),
                    "source_artifact": f"raw/{route}-r{repeat}.json",
                    "route": route,
                    "repeat": repeat,
                    "success": True,
                    "status": "complete",
                    "error_type": "",
                    "error": "",
                    "nodes": spec["nodes"],
                    "gpu_count": spec["gpu_count"],
                    "rank_count": spec["rank_count"],
                    "pipeline_count": spec["pipeline_count"],
                    "systems_requested": SYSTEMS_TOTAL,
                    "systems_completed": SYSTEMS_TOTAL,
                    "unique_systems_completed": SYSTEMS_TOTAL,
                    "missing_systems": 0,
                    "duplicate_systems": 0,
                    "unexpected_systems": 0,
                    "stage_1_completions": SYSTEMS_TOTAL,
                    "stage_2_completions": SYSTEMS_TOTAL,
                    "correctness_passed": True,
                    "max_energy_difference_ev": 1.0e-7,
                    "max_force_difference_ev_per_a": 2.0e-6,
                    "max_charge_difference_e": 3.0e-8,
                    "max_abs_net_charge_e": 1.0e-8,
                    "max_handoff_fmax_ev_per_a": 0.008,
                    "min_interatomic_distance_a": 0.8,
                    "max_relaxation_steps_observed": 20,
                    "nvt_steps_verified": True,
                    "nve_steps_verified": True,
                    "covalent_oh_gate_passed": True,
                    "oxygen_connectivity_gate_passed": True,
                    "elapsed_s": elapsed_s,
                    "systems_per_s": SYSTEMS_TOTAL / elapsed_s,
                    "peak_memory_bytes_max_rank": 2 * 1024**3,
                }
            )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def _manifest() -> dict[str, Any]:
    manifest = {
        "artifact_id": "filled-when-sealed",
        "campaign": {
            "model": MODEL,
            "repeats": REPEATS,
            "routes": [
                {"id": route, **spec} for route, spec in ROUTE_SPECS.items()
            ],
            "systems_total": SYSTEMS_TOTAL,
            "timing_boundary": (
                "Start before the first system enters the workflow; stop after "
                "the final result is collected and every GPU is synchronized."
            ),
            "workload": WORKLOAD,
        },
        "correctness": CORRECTNESS,
        "data": {
            "bytes": 0,
            "columns": list(RUN_COLUMNS),
            "file": "runs.csv",
            "row_count": 0,
            "sha256": "filled-when-sealed",
        },
        "failure_policy": {
            "failed_rows_retained": True,
            "required_failure_fields": ["success", "error_type", "error"],
        },
        "integrity": {
            "checksum_index": "SHA256SUMS",
            "checksum_index_covers_manifest": True,
        },
        "provenance": {
            "backend": "nccl",
            "generated_at_utc": "2026-07-14T19:00:00+00:00",
            "gpu_name": "NVIDIA H100 NVL",
            "partition": "h100-nvl@ts3/example/1gpu-32cpu-128gb",
            "producer_files_sha256": dict(PRODUCER_FILES),
            "producer_set_sha256": producer_set_sha256(PRODUCER_FILES),
            "python_hash_seed": "0",
            "python_version": "3.12.13",
            "repository_commit": REPOSITORY_COMMIT,
            "site": "Compute Lab",
            "slurm_job_ids_by_route": {
                route: str(4100 + spec["gpu_count"])
                for route, spec in ROUTE_SPECS.items()
            },
            "toolkit_core_branch": CORE_BRANCH,
            "toolkit_core_clean": True,
            "toolkit_core_commit": CORE_COMMIT,
            "toolkit_core_version": CORE_VERSION,
            "toolkit_ops_commit": OPS_COMMIT,
            "toolkit_ops_version": OPS_VERSION,
            "torch_version": "2.12.0+cu130",
        },
        "publication": {
            "ready": True,
            "successful_repeats_required_per_route": 5,
        },
        "schema": BUNDLE_SCHEMA,
        "scope": {
            "fixed_workload_speed_benchmark": True,
            "learner_use": "Plot saved H100 results without a cluster allocation.",
            "scientific_accuracy_benchmark": False,
        },
        "status": "complete",
    }
    tolerances = {
        key: value
        for key, value in CORRECTNESS.items()
        if key not in {"reference_route", "required_checks"}
    }
    manifest["acceptance"] = {
        "performance_metrics_used": False,
        "purpose": "smoke",
        "required_successes_per_route": 1,
        "runs": [
            {
                "stock_toolkit_record_sha256": toolkit_runtime_record_sha256(
                    manifest["provenance"]
                ),
                "correctness_contract_sha256": correctness_record_sha256(
                    CORRECTNESS["required_checks"], tolerances
                ),
                "correctness_passed": True,
                "covalent_oh_gate_passed": True,
                "duplicate_systems": 0,
                "error": None,
                "error_type": None,
                "gpu_count": spec["gpu_count"],
                "max_abs_net_charge_e": 1.0e-8,
                "max_charge_difference_e": 3.0e-8,
                "max_energy_difference_ev": 1.0e-7,
                "max_force_difference_ev_per_a": 2.0e-6,
                "max_handoff_fmax_ev_per_a": 0.008,
                "max_relaxation_steps_observed": 20,
                "min_interatomic_distance_a": 0.8,
                "missing_systems": 0,
                "model_record_sha256": canonical_json_sha256(MODEL),
                "nodes": spec["nodes"],
                "nve_steps_verified": True,
                "nvt_steps_verified": True,
                "oxygen_connectivity_gate_passed": True,
                "pipeline_count": spec["pipeline_count"],
                "producer_set_sha256": producer_set_sha256(PRODUCER_FILES),
                "purpose": "smoke",
                "rank_count": spec["rank_count"],
                "raw_record_sha256": str(spec["gpu_count"]) * 64,
                "repeat": 0,
                "route": route,
                "run_id": f"job-{route}-r0",
                "runtime_identity_sha256": runtime_identity_sha256(
                    manifest["provenance"]
                ),
                "schema": RUN_SCHEMA,
                "slurm_job_id": str(4100 + spec["gpu_count"]),
                "source_artifact": f"job-{route}/raw/smoke.json",
                "stage_1_completions": SYSTEMS_TOTAL,
                "stage_2_completions": SYSTEMS_TOTAL,
                "status": "complete",
                "success": True,
                "systems_completed": SYSTEMS_TOTAL,
                "systems_requested": SYSTEMS_TOTAL,
                "timestamp_utc": "2026-07-14T17:00:00+00:00",
                "unexpected_systems": 0,
                "unique_systems_completed": SYSTEMS_TOTAL,
                "workload_record_sha256": canonical_json_sha256(WORKLOAD),
            }
            for route, spec in ROUTE_SPECS.items()
        ],
        "systems_total": SYSTEMS_TOTAL,
    }
    return manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal_bundle(
    directory: Path,
    runs: pd.DataFrame,
    manifest: dict[str, Any] | None = None,
) -> Path:
    directory.mkdir()
    runs_path = directory / "runs.csv"
    runs.to_csv(runs_path, index=False, lineterminator="\n")
    runs_sha256 = sha256_file(runs_path)
    manifest = _manifest() if manifest is None else manifest
    manifest["artifact_id"] = f"pipeline-campaign-{runs_sha256[:16]}"
    manifest["data"].update(
        bytes=runs_path.stat().st_size,
        row_count=len(runs),
        sha256=runs_sha256,
    )
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    (directory / "SHA256SUMS").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
        f"{runs_sha256}  runs.csv\n",
        encoding="utf-8",
    )
    return directory


def test_loads_campaign_and_computes_learner_metrics(tmp_path: Path) -> None:
    bundle = load_pipeline_campaign_bundle(
        _seal_bundle(tmp_path / "campaign", _run_rows())
    )

    assert len(bundle.runs) == 15
    assert bundle.failed_runs.empty
    assert len(bundle.manifest["acceptance"]["runs"]) == 3
    assert bundle.runs.loc[0, "systems_per_s"] == pytest.approx(
        SYSTEMS_TOTAL / 12
    )
    assert bundle.runs.loc[0, "gpu_seconds_per_structure"] == pytest.approx(
        12 / SYSTEMS_TOTAL
    )

    summary = bundle.summary.set_index("route")
    assert summary.loc["fused_1gpu", "median_elapsed_s"] == pytest.approx(11.0)
    assert summary.loc["pipeline_2gpu", "speedup_vs_1gpu"] == pytest.approx(2.0)
    assert summary.loc["pipeline_4gpu", "speedup_vs_1gpu"] == pytest.approx(4.0)
    assert summary.loc["fused_1gpu", "parallel_efficiency_pct"] == pytest.approx(
        100.0
    )
    assert summary.loc["pipeline_2gpu", "parallel_efficiency_pct"] == pytest.approx(
        100.0
    )
    assert summary.loc["pipeline_4gpu", "parallel_efficiency_pct"] == pytest.approx(
        100.0
    )
    assert summary.loc[
        "pipeline_4gpu", "median_gpu_seconds_per_structure"
    ] == pytest.approx(2.75 * 4 / SYSTEMS_TOTAL)


def test_summary_preserves_a_measured_distributed_slowdown(tmp_path: Path) -> None:
    runs = _run_rows()
    two_gpu = runs["route"] == "pipeline_2gpu"
    two_gpu_times = pd.Series((14.0, 15.0, 16.0, 17.0, 18.0), index=runs.index[two_gpu])
    runs.loc[two_gpu, "elapsed_s"] = two_gpu_times
    runs.loc[two_gpu, "systems_per_s"] = SYSTEMS_TOTAL / two_gpu_times

    bundle = load_pipeline_campaign_bundle(
        _seal_bundle(tmp_path / "campaign", runs)
    )
    summary = bundle.summary.set_index("route")

    assert summary.loc["fused_1gpu", "median_elapsed_s"] == pytest.approx(11.0)
    assert summary.loc["pipeline_2gpu", "median_elapsed_s"] == pytest.approx(16.0)
    assert summary.loc["pipeline_2gpu", "speedup_vs_1gpu"] == pytest.approx(
        11.0 / 16.0
    )
    assert summary.loc[
        "pipeline_2gpu", "parallel_efficiency_pct"
    ] == pytest.approx(100.0 * (11.0 / 16.0) / 2.0)
    assert summary.loc[
        "pipeline_2gpu", "median_gpu_seconds_per_structure"
    ] == pytest.approx(2.0 * 16.0 / SYSTEMS_TOTAL)


def test_rejects_missing_full_route_acceptance(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["acceptance"]["runs"].pop()
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="acceptance.*every route"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_failed_full_route_acceptance(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["acceptance"]["runs"][0]["success"] = False
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="acceptance.*successful"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_acceptance_workload_identity_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["acceptance"]["runs"][0]["workload_record_sha256"] = "0" * 64
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="acceptance.*workload"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_d3_bj_parameter_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["campaign"]["model"] = dict(MODEL)
    manifest["campaign"]["model"]["d3_bj_parameters"] = dict(
        MODEL["d3_bj_parameters"], s8=1.4
    )
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="D3.*parameters"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_missing_route_repeat_case(tmp_path: Path) -> None:
    runs = _run_rows().iloc[:-1].copy()
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="complete requested matrix"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_route_topology_drift(tmp_path: Path) -> None:
    runs = _run_rows()
    runs.loc[runs["route"] == "pipeline_2gpu", "pipeline_count"] = 2
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="pipeline_count"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_cross_pair_four_gpu_boundary(tmp_path: Path) -> None:
    manifest = _manifest()
    workload = dict(WORKLOAD)
    workload["pipeline_pair_boundaries"] = {
        "fused_1gpu": [],
        "pipeline_2gpu": [[0, 1]],
        "pipeline_4gpu": [[0, 1], [1, 2], [2, 3]],
    }
    manifest["campaign"]["workload"] = workload
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="pipeline_pair_boundaries"):
        load_pipeline_campaign_bundle(directory)


def test_retains_failed_repeat_and_excludes_it_from_medians(tmp_path: Path) -> None:
    runs = _run_rows().astype(object)
    failed = (runs["route"] == "pipeline_4gpu") & (runs["repeat"] == 2)
    runs.loc[failed, "success"] = False
    runs.loc[failed, "status"] = "failed"
    runs.loc[failed, "error_type"] = "RuntimeError"
    runs.loc[failed, "error"] = "recorded worker failure"
    runs.loc[failed, "systems_completed"] = 40
    runs.loc[failed, "unique_systems_completed"] = 40
    runs.loc[failed, "missing_systems"] = 60
    runs.loc[failed, "stage_1_completions"] = 40
    runs.loc[failed, "stage_2_completions"] = 0
    runs.loc[
        failed,
        [
            "correctness_passed",
            "max_energy_difference_ev",
            "max_force_difference_ev_per_a",
            "max_charge_difference_e",
            "max_abs_net_charge_e",
            "max_handoff_fmax_ev_per_a",
            "min_interatomic_distance_a",
            "max_relaxation_steps_observed",
            "nvt_steps_verified",
            "nve_steps_verified",
            "covalent_oh_gate_passed",
            "oxygen_connectivity_gate_passed",
            "elapsed_s",
            "systems_per_s",
            "peak_memory_bytes_max_rank",
        ],
    ] = ""
    runs.loc[failed, "correctness_passed"] = False
    manifest = _manifest()
    manifest["publication"]["ready"] = False
    directory = _seal_bundle(tmp_path / "campaign", runs, manifest)
    with pytest.raises(PipelineCampaignError, match="not ready for publication"):
        load_pipeline_campaign_bundle(directory)
    bundle = load_pipeline_campaign_bundle(
        directory, require_publishable=False
    )

    assert len(bundle.failed_runs) == 1
    assert bundle.failed_runs.loc[0, "systems_completed"] == 40
    summary = bundle.summary.set_index("route")
    assert summary.loc["pipeline_4gpu", "successful_runs"] == 4
    assert summary.loc["pipeline_4gpu", "failed_runs"] == 1
    assert summary.loc["pipeline_4gpu", "median_elapsed_s"] == pytest.approx(
        2.6875
    )
    assert summary.loc["pipeline_4gpu", "speedup_vs_1gpu"] == pytest.approx(
        11 / 2.6875
    )
    assert summary.loc[
        "pipeline_4gpu", "parallel_efficiency_pct"
    ] == pytest.approx(100 * (11 / 2.6875) / 4)


def test_rejects_failed_repeat_without_error(tmp_path: Path) -> None:
    runs = _run_rows()
    failed = (runs["route"] == "pipeline_4gpu") & (runs["repeat"] == 2)
    runs.loc[failed, "success"] = False
    runs.loc[failed, "status"] = "failed"
    runs.loc[failed, "error_type"] = "RuntimeError"
    runs.loc[failed, "error"] = ""
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="error_type and error"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_success_that_failed_correctness_checks(tmp_path: Path) -> None:
    runs = _run_rows()
    runs.loc[0, "correctness_passed"] = False
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="correctness_passed"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_modified_toolkit_core(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["provenance"]["toolkit_core_clean"] = False
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="stock Toolkit runtime"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_old_patched_bundle_schema(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["schema"] = "alchemi.pipeline-campaign-bundle.v1"
    manifest["provenance"]["compatibility_patch"] = {"file": "old.patch"}
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="unexpected.*schema"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_missing_imported_producer_helper(tmp_path: Path) -> None:
    manifest = _manifest()
    producer_files = manifest["provenance"]["producer_files_sha256"]
    producer_files.pop(
        "part-1-scalable-atomistic-workflows/aux/electrostatics.py"
    )
    manifest["provenance"]["producer_set_sha256"] = producer_set_sha256(
        producer_files
    )
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="exact campaign producer"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_run_from_the_wrong_job(tmp_path: Path) -> None:
    runs = _run_rows()
    runs.loc[0, "slurm_job_id"] = "9999"
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="slurm_job_id"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_invalid_input_identity(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["campaign"]["workload"] = dict(WORKLOAD)
    manifest["campaign"]["workload"]["campaign_definition_sha256"] = (
        "not-a-sha256"
    )
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="campaign_definition_sha256"):
        load_pipeline_campaign_bundle(directory)


@pytest.mark.parametrize("filename", ["manifest.json", "runs.csv"])
def test_rejects_checksum_tamper(tmp_path: Path, filename: str) -> None:
    directory = _seal_bundle(tmp_path / "campaign", _run_rows())
    path = directory / filename
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        PipelineCampaignError, match=f"SHA-256 mismatch for {filename}"
    ):
        load_pipeline_campaign_bundle(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("systems_completed", SYSTEMS_TOTAL - 1, "systems_completed"),
        ("unique_systems_completed", SYSTEMS_TOTAL - 1, "unique_systems_completed"),
        ("missing_systems", 1, "missing_systems"),
        ("duplicate_systems", 1, "duplicate_systems"),
        ("unexpected_systems", 1, "unexpected_systems"),
        ("stage_1_completions", SYSTEMS_TOTAL - 1, "stage_1_completions"),
        ("stage_2_completions", SYSTEMS_TOTAL - 1, "stage_2_completions"),
        ("max_energy_difference_ev", 6.0e-5, "max_energy_difference_ev"),
        (
            "max_force_difference_ev_per_a",
            2.0e-4,
            "max_force_difference_ev_per_a",
        ),
        ("max_charge_difference_e", 6.0e-6, "max_charge_difference_e"),
        ("max_abs_net_charge_e", 6.0e-5, "max_abs_net_charge_e"),
        (
            "max_handoff_fmax_ev_per_a",
            0.02,
            "max_handoff_fmax_ev_per_a",
        ),
        ("min_interatomic_distance_a", 0.5, "min_interatomic_distance_a"),
        ("nvt_steps_verified", False, "nvt_steps_verified"),
        ("nve_steps_verified", False, "nve_steps_verified"),
        ("covalent_oh_gate_passed", False, "covalent_oh_gate_passed"),
        (
            "oxygen_connectivity_gate_passed",
            False,
            "oxygen_connectivity_gate_passed",
        ),
    ],
)
def test_rejects_incomplete_or_mismatched_success(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    runs = _run_rows()
    runs.loc[0, field] = value
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match=message):
        load_pipeline_campaign_bundle(directory)


def test_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["unreviewed_claim"] = True
    directory = _seal_bundle(tmp_path / "campaign", _run_rows(), manifest)

    with pytest.raises(PipelineCampaignError, match="manifest root has the wrong keys"):
        load_pipeline_campaign_bundle(directory)


def test_rejects_invalid_partial_measurement_in_failed_row(tmp_path: Path) -> None:
    runs = _run_rows().astype(object)
    failed = (runs["route"] == "pipeline_4gpu") & (runs["repeat"] == 2)
    runs.loc[failed, "success"] = False
    runs.loc[failed, "status"] = "failed"
    runs.loc[failed, "error_type"] = "RuntimeError"
    runs.loc[failed, "error"] = "recorded worker failure"
    runs.loc[failed, "correctness_passed"] = ""
    runs.loc[failed, "elapsed_s"] = "not-a-number"
    directory = _seal_bundle(tmp_path / "campaign", runs)

    with pytest.raises(PipelineCampaignError, match="elapsed_s"):
        load_pipeline_campaign_bundle(directory)
