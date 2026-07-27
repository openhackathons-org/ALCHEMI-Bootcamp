"""Tests for the notebook-facing pipeline campaign lesson view."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.artifacts import sha256_file  # noqa: E402
from aux.pipeline_campaign_results import (  # noqa: E402
    CHECKSUM_INDEX_NAME,
    FIXED_SYSTEMS_TOTAL,
    MANIFEST_NAME,
    ROUTE_SPECS,
    RUNS_NAME,
    PipelineCampaignBundle,
    PipelineCampaignError,
)
from aux.pipeline_campaign_view import (  # noqa: E402
    _load_pipeline_campaign_lesson_view_with_loader,
    load_pipeline_campaign_lesson_view,
)


def _write_required_files(root: Path) -> None:
    root.mkdir()
    for index, name in enumerate((MANIFEST_NAME, RUNS_NAME, CHECKSUM_INDEX_NAME)):
        (root / name).write_text(f"test file {index}\n", encoding="utf-8")


def _fake_bundle(root: Path) -> PipelineCampaignBundle:
    systems_total = 321
    repeats = 2
    model = {
        "components": ["short range", "electrostatics", "dispersion"],
    }
    workload = {
        "batch_size": 64,
        "fire_fmax_ev_per_a": 0.02,
        "nvt_steps": 12,
        "nve_steps": 34,
        "dt_fs": 0.25,
        "temperature_k": 275.0,
    }
    manifest: dict[str, Any] = {
        "artifact_id": "pipeline-campaign-test",
        "campaign": {
            "systems_total": systems_total,
            "repeats": repeats,
            "routes": [{"id": route, **spec} for route, spec in ROUTE_SPECS.items()],
            "model": model,
            "workload": workload,
        },
        "provenance": {
            "site": "Compute Lab",
            "partition": "H100 test partition",
            "gpu_name": "NVIDIA H100 NVL",
            "torch_version": "2.test",
            "python_version": "3.12.test",
            "toolkit_core_commit": "a" * 40,
            "toolkit_ops_commit": "b" * 40,
            "producer_set_sha256": "c" * 64,
        },
    }

    elapsed_by_route = {
        "fused_1gpu": (12.0, 10.0),
        "pipeline_2gpu": (7.0, 5.0),
        "pipeline_4gpu": (4.0, 2.0),
    }
    rows: list[dict[str, Any]] = []
    for route, elapsed_values in elapsed_by_route.items():
        gpu_count = ROUTE_SPECS[route]["gpu_count"]
        for repeat, elapsed_s in enumerate(elapsed_values, start=1):
            rows.append(
                {
                    "route": route,
                    "repeat": repeat,
                    "success": True,
                    "status": "complete",
                    "error_type": "",
                    "systems_completed": systems_total,
                    "elapsed_s": elapsed_s,
                    "systems_per_s": systems_total / elapsed_s,
                    "gpu_seconds_per_structure": (
                        gpu_count * elapsed_s / systems_total
                    ),
                }
            )
    runs = pd.DataFrame(rows)
    return PipelineCampaignBundle(root=root, manifest=manifest, runs=runs)


def test_public_lesson_view_signature_hides_loader_details() -> None:
    assert "bundle_loader" not in signature(
        load_pipeline_campaign_lesson_view
    ).parameters


def test_unavailable_view_lists_files_and_supplies_downstream_defaults(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-campaign"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("present\n", encoding="utf-8")

    view = load_pipeline_campaign_lesson_view(
        root,
        planned_systems_total=FIXED_SYSTEMS_TOTAL,
    )

    assert view.availability == "unavailable"
    assert not view.available
    assert view.missing_files == (RUNS_NAME, CHECKSUM_INDEX_NAME)
    assert view.bundle is None
    assert view.bundle_record is None
    assert view.systems_total == FIXED_SYSTEMS_TOTAL
    assert view.repeats == 0
    assert view.successful_runs == 0
    assert view.failed_runs == 0
    assert view.workload_table.empty
    assert view.routes_table.empty
    assert view.route_summary.empty
    assert view.summary_table.empty
    assert view.run_details_table.empty
    assert view.failures.empty
    assert list(view.summary_table.columns) == [
        "Layout",
        "GPUs",
        "Successful runs",
        "Failed runs",
        "Median wall time (s)",
        "Structures/s",
        "Speedup",
        "Parallel efficiency (%)",
        "Timed H100-s/structure",
    ]


def test_available_view_formats_verified_bundle_without_changing_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _write_required_files(root)
    bundle = _fake_bundle(root.resolve())
    loaded_paths: list[Path] = []

    def fake_loader(path: str | Path) -> PipelineCampaignBundle:
        loaded_paths.append(Path(path))
        return bundle

    view = _load_pipeline_campaign_lesson_view_with_loader(
        root,
        planned_systems_total=9_999,
        bundle_loader=fake_loader,
    )

    assert loaded_paths == [root.resolve()]
    assert view.availability == "available"
    assert view.available
    assert view.missing_files == ()
    assert view.bundle is bundle
    assert view.bundle_record == {
        "artifact_id": "pipeline-campaign-test",
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        "runs_sha256": sha256_file(root / RUNS_NAME),
        "checksum_index_sha256": sha256_file(root / CHECKSUM_INDEX_NAME),
        "producer_set_sha256": "c" * 64,
    }
    assert view.workload_table.loc["Structures", "Recorded workload"] == (
        "321 water hexamers"
    )
    assert view.workload_table.loc["Active batch size", "Recorded workload"] == 64
    assert view.workload_table.loc["Relaxation", "Recorded workload"] == (
        "FIRE2 to 0.02 eV/Å"
    )
    assert view.workload_table.loc["Dynamics", "Recorded workload"] == (
        "12 NVT + 34 NVE steps at 0.25 fs"
    )
    assert view.workload_table.loc["Temperature", "Recorded workload"] == "275 K"
    assert list(view.routes_table.columns) == [
        "Route",
        "Nodes",
        "GPUs",
        "Ranks",
        "Pipeline pairs",
    ]
    assert view.summary_table["Layout"].tolist() == [
        "1 GPU · fused stages",
        "2 GPUs · one pipeline pair",
        "4 GPUs · two pipeline pairs",
    ]
    assert view.summary_table.loc[0, "Median wall time (s)"] == pytest.approx(11.0)
    assert view.summary_table.loc[1, "Speedup"] == pytest.approx(11.0 / 6.0)
    assert view.summary_table.loc[2, "Parallel efficiency (%)"] == pytest.approx(
        100.0 * (11.0 / 3.0) / 4.0
    )
    assert view.run_details_table.loc["GPU", "Recorded run"] == "NVIDIA H100 NVL"
    assert view.failures.empty
    assert view.systems_total == 321
    assert view.repeats == 2
    assert view.successful_runs == 6
    assert view.failed_runs == 0


def test_complete_but_invalid_bundle_keeps_strict_loader_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-campaign"
    _write_required_files(root)

    def reject_bundle(path: str | Path) -> PipelineCampaignBundle:
        raise PipelineCampaignError(f"invalid verified bundle at {path}")

    with pytest.raises(PipelineCampaignError, match="invalid verified bundle"):
        _load_pipeline_campaign_lesson_view_with_loader(
            root,
            planned_systems_total=FIXED_SYSTEMS_TOTAL,
            bundle_loader=reject_bundle,
        )
