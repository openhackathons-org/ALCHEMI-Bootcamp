"""Prepare the prerecorded pipeline campaign for the Part 1 lesson.

The strict bundle loader remains the source of truth for checksums, runtime
identity, workload settings, correctness checks, and publishability.  This
module only checks whether the three bundle files are installed and reshapes a
verified bundle into the small set of tables and counts used by the notebook.

No timing values, scientific thresholds, or pass/fail decisions are chosen
here.  Display tables retain the full values from the verified bundle; the
notebook may round a copy when it renders them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal

import pandas as pd

from .artifacts import sha256_file
from .pipeline_campaign_results import (
    CHECKSUM_INDEX_NAME,
    MANIFEST_NAME,
    RUNS_NAME,
    PipelineCampaignBundle,
    load_pipeline_campaign_bundle,
)


PipelineCampaignAvailability = Literal["available", "unavailable"]

_REQUIRED_BUNDLE_FILES = (MANIFEST_NAME, RUNS_NAME, CHECKSUM_INDEX_NAME)
_ROUTE_DISPLAY_COLUMNS = (
    "Route",
    "Nodes",
    "GPUs",
    "Ranks",
    "Pipeline pairs",
)
_SUMMARY_SOURCE_COLUMNS = (
    "route",
    "gpu_count",
    "successful_runs",
    "failed_runs",
    "median_elapsed_s",
    "median_systems_per_s",
    "speedup_vs_1gpu",
    "parallel_efficiency_pct",
    "median_gpu_seconds_per_structure",
)
_SUMMARY_DISPLAY_COLUMNS = (
    "Layout",
    "GPUs",
    "Successful runs",
    "Failed runs",
    "Median wall time (s)",
    "Structures/s",
    "Speedup",
    "Parallel efficiency (%)",
    "Timed H100-s/structure",
)
_FAILURE_COLUMNS = (
    "route",
    "repeat",
    "status",
    "error_type",
    "systems_completed",
)
_LAYOUT_NAMES = {
    "fused_1gpu": "1 GPU · fused stages",
    "pipeline_2gpu": "2 GPUs · one pipeline pair",
    "pipeline_4gpu": "4 GPUs · two pipeline pairs",
}


@dataclass(frozen=True)
class PipelineCampaignLessonView:
    """Tables and downstream values for one available or unavailable bundle.

    ``availability == "unavailable"`` means at least one required file is not
    installed.  A present but invalid bundle never becomes an unavailable
    view: :func:`load_pipeline_campaign_bundle` raises its original error.

    ``bundle`` is retained only because the existing plotting helper consumes
    the verified bundle.  The remaining fields let the notebook avoid reaching
    back into the manifest to format tables or prepare its saved run summary.
    """

    availability: PipelineCampaignAvailability
    missing_files: tuple[str, ...]
    bundle: PipelineCampaignBundle | None
    bundle_record: dict[str, str] | None
    workload_table: pd.DataFrame
    routes_table: pd.DataFrame
    route_summary: pd.DataFrame
    summary_table: pd.DataFrame
    run_details_table: pd.DataFrame
    failures: pd.DataFrame
    systems_total: int
    repeats: int
    successful_runs: int
    failed_runs: int

    @property
    def available(self) -> bool:
        """Return whether a verified, publishable bundle was loaded."""

        return self.availability == "available"


def _empty_table(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _unavailable_view(
    missing_files: tuple[str, ...],
    *,
    planned_systems_total: int,
) -> PipelineCampaignLessonView:
    return PipelineCampaignLessonView(
        availability="unavailable",
        missing_files=missing_files,
        bundle=None,
        bundle_record=None,
        workload_table=_empty_table(("Recorded workload",)),
        routes_table=_empty_table(_ROUTE_DISPLAY_COLUMNS),
        route_summary=_empty_table(_SUMMARY_SOURCE_COLUMNS),
        summary_table=_empty_table(_SUMMARY_DISPLAY_COLUMNS),
        run_details_table=_empty_table(("Recorded run",)),
        failures=_empty_table(_FAILURE_COLUMNS),
        systems_total=planned_systems_total,
        repeats=0,
        successful_runs=0,
        failed_runs=0,
    )


def _available_view(
    bundle: PipelineCampaignBundle,
    *,
    root: Path,
) -> PipelineCampaignLessonView:
    manifest = bundle.manifest
    campaign = manifest["campaign"]
    workload = campaign["workload"]
    model = campaign["model"]
    run_details = bundle.run_details

    workload_table = pd.Series(
        {
            "Structures": f"{campaign['systems_total']} water hexamers",
            "Active batch size": workload["batch_size"],
            "Model": " + ".join(model["components"]),
            "Relaxation": f"FIRE2 to {workload['fire_fmax_ev_per_a']:.2f} eV/Å",
            "Dynamics": (
                f"{workload['nvt_steps']} NVT + {workload['nve_steps']} NVE "
                f"steps at {workload['dt_fs']} fs"
            ),
            "Temperature": f"{workload['temperature_k']:.0f} K",
            "Saved timing runs": f"{campaign['repeats']} per route",
        },
        name="Recorded workload",
    ).to_frame()

    routes_table = (
        pd.DataFrame(campaign["routes"])[
            ["id", "nodes", "gpu_count", "rank_count", "pipeline_count"]
        ]
        .rename(
            columns={
                "id": "Route",
                "nodes": "Nodes",
                "gpu_count": "GPUs",
                "rank_count": "Ranks",
                "pipeline_count": "Pipeline pairs",
            }
        )
        .reset_index(drop=True)
    )

    route_summary = bundle.summary.copy()
    summary_table = route_summary[list(_SUMMARY_SOURCE_COLUMNS)].copy()
    summary_table["route"] = summary_table["route"].map(_LAYOUT_NAMES)
    summary_table = summary_table.rename(
        columns={
            "route": "Layout",
            "gpu_count": "GPUs",
            "successful_runs": "Successful runs",
            "failed_runs": "Failed runs",
            "median_elapsed_s": "Median wall time (s)",
            "median_systems_per_s": "Structures/s",
            "speedup_vs_1gpu": "Speedup",
            "parallel_efficiency_pct": "Parallel efficiency (%)",
            "median_gpu_seconds_per_structure": "Timed H100-s/structure",
        }
    )

    run_details_table = pd.Series(
        {
            "Site / partition": f"{run_details['site']} / {run_details['partition']}",
            "GPU": run_details["gpu_name"],
            "Torch / Python": (
                f"{run_details['torch_version']} / {run_details['python_version']}"
            ),
            "Toolkit Core commit": run_details["toolkit_core_commit"],
            "Toolkit-Ops commit": run_details["toolkit_ops_commit"],
            "Result set": manifest["artifact_id"],
        },
        name="Recorded run",
    ).to_frame()

    failures = bundle.failed_runs.loc[:, list(_FAILURE_COLUMNS)].copy()
    successful_runs = int(route_summary["successful_runs"].sum())
    failed_runs = int(route_summary["failed_runs"].sum())
    bundle_record = {
        "artifact_id": manifest["artifact_id"],
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        "runs_sha256": sha256_file(root / RUNS_NAME),
        "checksum_index_sha256": sha256_file(root / CHECKSUM_INDEX_NAME),
        "producer_set_sha256": run_details["producer_set_sha256"],
    }

    return PipelineCampaignLessonView(
        availability="available",
        missing_files=(),
        bundle=bundle,
        bundle_record=bundle_record,
        workload_table=workload_table,
        routes_table=routes_table,
        route_summary=route_summary,
        summary_table=summary_table,
        run_details_table=run_details_table,
        failures=failures,
        systems_total=int(campaign["systems_total"]),
        repeats=int(campaign["repeats"]),
        successful_runs=successful_runs,
        failed_runs=failed_runs,
    )


def load_pipeline_campaign_lesson_view(
    bundle_dir: str | Path,
    *,
    planned_systems_total: int,
) -> PipelineCampaignLessonView:
    """Load one lesson view or report which required files are absent.

    ``planned_systems_total`` is required rather than hidden in this helper.  It
    is used only for the unavailable-state sentence in the final results
    summary; an available view always takes the value from the verified
    manifest.

    Missing files are an expected tutorial state and return an unavailable
    view.  If every file is present, the existing strict loader runs unchanged;
    checksum, schema, correctness, or publishability errors propagate to the
    caller instead of being converted into a softer unavailable state.
    """

    return _load_pipeline_campaign_lesson_view_with_loader(
        bundle_dir,
        planned_systems_total=planned_systems_total,
        bundle_loader=load_pipeline_campaign_bundle,
    )


def _load_pipeline_campaign_lesson_view_with_loader(
    bundle_dir: str | Path,
    *,
    planned_systems_total: int,
    bundle_loader: Callable[[str | Path], PipelineCampaignBundle],
) -> PipelineCampaignLessonView:
    """Load a lesson view through an explicit strict loader for tests."""

    if (
        isinstance(planned_systems_total, bool)
        or not isinstance(planned_systems_total, int)
        or planned_systems_total <= 0
    ):
        raise ValueError("planned_systems_total must be a positive integer")

    root = Path(bundle_dir).resolve()
    missing_files = tuple(
        name for name in _REQUIRED_BUNDLE_FILES if not (root / name).is_file()
    )
    if missing_files:
        return _unavailable_view(
            missing_files,
            planned_systems_total=planned_systems_total,
        )

    bundle = bundle_loader(root)
    return _available_view(bundle, root=root)


__all__ = (
    "PipelineCampaignAvailability",
    "PipelineCampaignLessonView",
    "load_pipeline_campaign_lesson_view",
)
