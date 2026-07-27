"""Tests for sealing fetched Compute Lab pipeline campaign records."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PART_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(PART_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from assemble_part1_pipeline_campaign import (  # noqa: E402
    AssemblyError,
    assemble_pipeline_campaign,
)
from aux.artifacts import sha256_file  # noqa: E402
from aux.pipeline_campaign_results import (  # noqa: E402
    PipelineCampaignError,
    ROUTE_SPECS,
    load_pipeline_campaign_bundle,
)
from part1_distributed_campaign_contract import (  # noqa: E402
    ATOMS_PER_SYSTEM,
    CAMPAIGN_SEED,
    CHARGE_ATOL_E,
    CORE_BRANCH,
    CORE_COMMIT,
    CORE_VERSION,
    COVALENT_OH_CUTOFF_A,
    DEFAULT_SYSTEMS,
    ENERGY_ATOL_EV,
    FORCE_ATOL_EV_PER_A,
    MAX_ABS_NET_CHARGE_E,
    MIN_INTERATOMIC_DISTANCE_A,
    OPS_COMMIT,
    OPS_VERSION,
    OXYGEN_CONNECTIVITY_CUTOFF_A,
    PRODUCER_FILES,
    ROUTE_PAIR_BOUNDARIES,
    RUN_SCHEMA,
    TIMING_BOUNDARY,
)


SYSTEMS = DEFAULT_SYSTEMS
NVT_STEPS = 10
NVE_STEPS = 20
FIRE_FMAX = 0.01
PARTITION = "h100-nvl@ts3/example/1gpu-32cpu-128gb"


def _producer_set() -> dict[str, str]:
    return {
        relative_path: sha256_file(REPOSITORY_ROOT / relative_path)
        for relative_path in PRODUCER_FILES
    }


def _model() -> dict[str, Any]:
    return {
        "checkpoint_source": "aimnet2-b973c-2025-d3_0",
        "checkpoint_sha256": "1" * 64,
        "d3_parameter_sha256": "2" * 64,
        "d3_bj_parameters": {"a1": 0.37, "a2": 4.1, "s6": 1.0, "s8": 1.5},
        "components": [
            "AIMNet2 B97-3c residual",
            "finite all-pairs Coulomb",
            "pairwise D3(BJ)",
        ],
        "dtype": (
            "float32 positions, velocities, forces, AIMNet, and Coulomb pair "
            "math; float64 Coulomb energy accumulation"
        ),
        "eager": True,
    }


def _workload() -> dict[str, Any]:
    builder = "part-1-scalable-atomistic-workflows/aux/structures.py"
    return {
        "campaign_definition_sha256": "3" * 64,
        "source_structure": "generated cyclic water hexamer",
        "structure_builder_file": builder,
        "structure_builder_sha256": sha256_file(REPOSITORY_ROOT / builder),
        "systems_total": SYSTEMS,
        "atoms_per_system": ATOMS_PER_SYSTEM,
        "batch_size": 16,
        "campaign_seed": CAMPAIGN_SEED,
        "perturbation_description": "Deterministic test campaign.",
        "fire_fmax_ev_per_a": FIRE_FMAX,
        "nvt_steps": NVT_STEPS,
        "nve_steps": NVE_STEPS,
        "dt_fs": 0.5,
        "temperature_k": 75.0,
        "friction_per_fs": 0.01,
        "velocity_seed_rule": "910000 + campaign_id",
        "pipeline_partition_rule": (
            "4 GPUs: campaign_id % 2; 1 or 2 GPUs: all campaign IDs"
        ),
        "pipeline_pair_boundaries": {
            route: [list(pair) for pair in pairs]
            for route, pairs in ROUTE_PAIR_BOUNDARIES.items()
        },
        "comm_mode": "async_recv",
        "stage_names": ["FIRE2", "NVTLangevin", "NVE"],
    }


def _tolerances() -> dict[str, float]:
    return {
        "energy_atol_ev": ENERGY_ATOL_EV,
        "force_atol_ev_per_a": FORCE_ATOL_EV_PER_A,
        "charge_atol_e": CHARGE_ATOL_E,
        "max_abs_net_charge_e": MAX_ABS_NET_CHARGE_E,
        "max_handoff_fmax_ev_per_a": FIRE_FMAX,
        "min_interatomic_distance_a": MIN_INTERATOMIC_DISTANCE_A,
        "covalent_oh_cutoff_a": COVALENT_OH_CUTOFF_A,
        "oxygen_connectivity_cutoff_a": OXYGEN_CONNECTIVITY_CUTOFF_A,
    }


def _record(route: str, repeat: int, *, success: bool = True) -> dict[str, Any]:
    spec = ROUTE_SPECS[route]
    elapsed_s = 10.0 / spec["gpu_count"] + 0.1 * repeat
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "complete" if success else "failed",
        "success": success,
        "timestamp_utc": "2026-07-14T20:00:00+00:00",
        "run_id": f"{5000 + spec['gpu_count']}-{route}-r{repeat}",
        "slurm_job_id": str(5000 + spec["gpu_count"]),
        "route": route,
        "purpose": "campaign",
        "repeat": repeat,
        "error_type": None if success else "SrunExitCode",
        "error": None if success else "recorded worker failure",
        "nodes": spec["nodes"],
        "gpu_count": spec["gpu_count"],
        "rank_count": spec["rank_count"],
        "pipeline_count": spec["pipeline_count"],
        "systems_requested": SYSTEMS,
        "systems_completed": SYSTEMS if success else None,
        "unique_systems_completed": SYSTEMS if success else None,
        "missing_systems": [] if success else None,
        "duplicate_systems": [] if success else None,
        "unexpected_systems": [] if success else None,
        "stage_1_completions": SYSTEMS if success else None,
        "stage_2_completions": SYSTEMS if success else None,
        "correctness_passed": success,
        "max_energy_difference_ev": 1.0e-7 if success else None,
        "max_force_difference_ev_per_a": 2.0e-7 if success else None,
        "max_charge_difference_e": 3.0e-8 if success else None,
        "max_abs_net_charge_e": 1.0e-8 if success else None,
        "max_handoff_fmax_ev_per_a": 0.008 if success else None,
        "min_interatomic_distance_a": 0.8 if success else None,
        "max_relaxation_steps_observed": 20 if success else None,
        "nvt_steps_verified": success if success else None,
        "nve_steps_verified": success if success else None,
        "covalent_oh_gate_passed": success if success else None,
        "oxygen_connectivity_gate_passed": success if success else None,
        "rank_audits": [] if success else None,
        "elapsed_s": elapsed_s if success else None,
        "systems_per_s": SYSTEMS / elapsed_s if success else None,
        "peak_memory_bytes_max_rank": 2 * 1024**3 if success else None,
        "gpu_name": "NVIDIA H100 NVL",
        "backend": "nccl",
        "hostname_rank0": f"compute-{spec['gpu_count']}",
        "torch_version": "2.12.0+cu130",
        "python_version": "3.12.13",
        "partition": PARTITION,
        "python_hash_seed": "0",
        "toolkit_core_branch": CORE_BRANCH,
        "toolkit_core_clean": True,
        "toolkit_core_commit": CORE_COMMIT,
        "toolkit_core_version": CORE_VERSION,
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_ops_version": OPS_VERSION,
        "producer_set": _producer_set(),
        "repository_commit": "a" * 40,
        "model": _model() if success else None,
        "workload": _workload(),
        "timing_boundary": TIMING_BOUNDARY,
        "correctness_checks": [
            "same campaign IDs",
            "direct final model reevaluation matches",
            "water chemistry gates pass",
        ] if success else None,
        "correctness_tolerances": _tolerances(),
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_job(
    root: Path,
    route: str,
    *,
    failed_repeat: int | None = None,
    absolute_index_paths: bool = False,
) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True)
    artifacts: list[Path] = []
    for repeat in range(1, 6):
        path = raw / f"{route}-r{repeat}.json"
        _write_json(path, _record(route, repeat, success=repeat != failed_repeat))
        artifacts.append(path)
    smoke = raw / "smoke.json"
    smoke_record = _record(route, 0)
    smoke_record["purpose"] = "smoke"
    _write_json(smoke, smoke_record)
    artifacts.append(smoke)

    def raw_index_path(path: Path) -> str:
        relative = path.relative_to(root)
        if absolute_index_paths:
            return (Path("/compute-lab/results") / root.name / relative).as_posix()
        return relative.as_posix()

    (root / "raw-SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {raw_index_path(path)}\n"
            for path in artifacts
        ),
        encoding="utf-8",
    )

    def producer_index_path(relative_path: str) -> str:
        if not absolute_index_paths:
            return relative_path
        prefix = Path("/shared/tutorials")
        return (prefix / relative_path).as_posix()

    (root / "producer-SHA256SUMS").write_text(
        "".join(
            f"{digest}  {producer_index_path(relative_path)}\n"
            for relative_path, digest in _producer_set().items()
        ),
        encoding="utf-8",
    )
    return root


def _rewrite_raw_index(root: Path) -> None:
    artifacts = sorted((root / "raw").glob("*.json")) + sorted(
        (root / "raw").glob("*.json.incomplete")
    )
    (root / "raw-SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in artifacts
        ),
        encoding="utf-8",
    )


def test_seals_publishable_campaign_and_loads_summary(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    output = tmp_path / "bundle"

    assembled = assemble_pipeline_campaign(
        inputs,
        output_dir=output,
        generated_at_utc="2026-07-14T21:00:00+00:00",
    )
    bundle = load_pipeline_campaign_bundle(assembled)

    assert assembled == output.resolve()
    assert len(bundle.runs) == 15
    assert bundle.failed_runs.empty
    assert bundle.manifest["publication"]["ready"] is True
    assert {
        run["route"] for run in bundle.manifest["acceptance"]["runs"]
    } == set(ROUTE_SPECS)
    assert set(bundle.runs["route"]) == set(ROUTE_SPECS)
    assert bundle.summary.set_index("route").loc[
        "pipeline_4gpu", "speedup_vs_1gpu"
    ] > 3.0


def test_retains_failure_but_marks_bundle_not_publishable(tmp_path: Path) -> None:
    inputs = [
        _write_job(
            tmp_path / f"job-{route}",
            route,
            failed_repeat=3 if route == "pipeline_4gpu" else None,
        )
        for route in ROUTE_SPECS
    ]
    output = assemble_pipeline_campaign(
        inputs,
        output_dir=tmp_path / "bundle",
        generated_at_utc="2026-07-14T21:00:00+00:00",
    )

    with pytest.raises(PipelineCampaignError, match="not ready for publication"):
        load_pipeline_campaign_bundle(output)
    diagnostic = load_pipeline_campaign_bundle(
        output, require_publishable=False
    )
    assert len(diagnostic.failed_runs) == 1
    assert diagnostic.failed_runs.loc[0, "error_type"] == "SrunExitCode"
    assert diagnostic.manifest["publication"]["ready"] is False


def test_accepts_compute_lab_absolute_paths_after_fetch(tmp_path: Path) -> None:
    inputs = [
        _write_job(
            tmp_path / f"job-{route}",
            route,
            absolute_index_paths=True,
        )
        for route in ROUTE_SPECS
    ]

    output = assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")

    assert load_pipeline_campaign_bundle(output).manifest["publication"][
        "ready"
    ] is True


def test_rejects_job_without_full_smoke_acceptance(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    (inputs[0] / "raw" / "smoke.json").unlink()
    _rewrite_raw_index(inputs[0])

    with pytest.raises(AssemblyError, match="exactly one.*smoke"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_failed_smoke_acceptance(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    smoke = inputs[0] / "raw" / "smoke.json"
    failed = _record("fused_1gpu", 0, success=False)
    failed["purpose"] = "smoke"
    _write_json(smoke, failed)
    _rewrite_raw_index(inputs[0])

    with pytest.raises(AssemblyError, match="smoke.*successful"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_smoke_from_wrong_route(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    smoke = inputs[0] / "raw" / "smoke.json"
    wrong_route = _record("pipeline_2gpu", 0)
    wrong_route["purpose"] = "smoke"
    _write_json(smoke, wrong_route)
    _rewrite_raw_index(inputs[0])

    with pytest.raises(AssemblyError, match="smoke.*route"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_smoke_with_different_workload_identity(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    smoke = inputs[0] / "raw" / "smoke.json"
    record = json.loads(smoke.read_text(encoding="utf-8"))
    record["workload"]["temperature_k"] = 80.0
    _write_json(smoke, record)
    _rewrite_raw_index(inputs[0])

    with pytest.raises(AssemblyError, match="field 'workload' differs"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_missing_measured_case_without_writing_bundle(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    missing_job = inputs[-1]
    missing = missing_job / "raw" / "pipeline_4gpu-r5.json"
    missing.unlink()
    index = missing_job / "raw-SHA256SUMS"
    index.write_text(
        "\n".join(
            line
            for line in index.read_text(encoding="utf-8").splitlines()
            if "pipeline_4gpu-r5.json" not in line
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "bundle"

    with pytest.raises(AssemblyError, match="requested full matrix"):
        assemble_pipeline_campaign(inputs, output_dir=output)
    assert not output.exists()


def test_rejects_raw_record_tamper(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    tampered = inputs[0] / "raw" / "fused_1gpu-r1.json"
    tampered.write_bytes(tampered.read_bytes() + b" ")

    with pytest.raises(AssemblyError, match="SHA-256 mismatch"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_raw_record_from_modified_toolkit(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    record_path = inputs[0] / "raw" / "fused_1gpu-r1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["toolkit_core_clean"] = False
    _write_json(record_path, record)
    _rewrite_raw_index(inputs[0])

    with pytest.raises(AssemblyError, match="toolkit_core_clean"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")


def test_rejects_historical_patch_in_active_producer_index(tmp_path: Path) -> None:
    inputs = [
        _write_job(tmp_path / f"job-{route}", route) for route in ROUTE_SPECS
    ]
    patch_path = (
        REPOSITORY_ROOT
        / "scripts/patches/nvalchemi-toolkit-b770ee6-sustained-pipeline-compat.patch"
    )
    index = inputs[0] / "producer-SHA256SUMS"
    index.write_text(
        index.read_text(encoding="utf-8")
        + f"{sha256_file(patch_path)}  {patch_path.as_posix()}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssemblyError, match="unexpected producer path"):
        assemble_pipeline_campaign(inputs, output_dir=tmp_path / "bundle")
