"""Run-cache registry for the batched adsorption tutorial.

The notebook has three distinct storage modes:

* official precomputed artifacts under ``outputs/precomputed``;
* explicit, timestamped live runs under ``outputs/live_runs/<run_id>``;
* runtime/model caches under ``outputs/runtime_cache``.

This module keeps those paths and their completeness checks in one place so a
presentation run cannot silently read an arbitrary timestamped scratch run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
OFFICIAL_OUTPUT_ROOT = Path("outputs") / "precomputed"
LIVE_RUN_ROOT = Path("outputs") / "live_runs"
RUNTIME_CACHE_ROOT = Path("outputs") / "runtime_cache"
LATEST_COMPLETE_RUN_ID = "latest-complete"

TUTORIAL_SURFACE_SCREEN_STEM = "surface_screen_v1_mace_mpa0"
ACCURACY_TRAJECTORY_STEM = "oc20dense_closed_shell_trajectory_mace_mpa0"
ACCURACY_NH3_RANKING_STEM = "oc20dense_nh3_92_fixed_geometry_mace_mpa0"

SURFACE_SCREEN_REQUIRED_FILES = (
    "run_metadata.json",
    "tables/surface_fingerprints.csv",
    "tables/initial_geometry_audit.csv",
    "tables/clean_slab_energies.csv",
    "tables/gas_energies.csv",
    "tables/adsorption_results.csv",
    "tables/pair_summary.csv",
    "tables/batch_summary.csv",
    "tables/step_statistics.csv",
    "tables/difficult_cases.csv",
    "tables/application_heatmap.csv",
    "raw_batches",
    "structures/relaxed_adsorption",
    "reports/surface_screen_report.md",
)

ACCURACY_TRAJECTORY_REQUIRED_FILES = (
    "tables/per_config_results.csv",
    "tables/system_summary.csv",
    "dft_reference_checks/dft_reference_comparison.csv",
    "mace_adsorption_energy/tables/mace_adsorption_energies.csv",
    "mace_adsorption_energy/tables/mace_adsorption_energy_summary.csv",
    "mace_adsorption_energy/tables/mace_adsorption_reference_energies.csv",
    "reports/run_metadata.json",
)

ACCURACY_NH3_REQUIRED_FILES = (
    "dft_reference_checks/dft_reference_comparison.csv",
    "dft_final_single_points/tables/dft_final_sp_results.csv",
    "dft_final_single_points/tables/dft_final_sp_system_summary.csv",
)


@dataclass(frozen=True)
class RunRoots:
    """Resolved tutorial and accuracy roots for one notebook run."""

    run_scope: str
    live_run_id: str
    tutorial_output_dir: Path
    accuracy_output_dir: Path
    cache_dir: Path
    plots_dir: Path
    runtime_cache_dir: Path
    tutorial_source_label: str
    accuracy_source_label: str
    official_tutorial_root: Path
    official_accuracy_root: Path
    live_root: Path

    @property
    def surface_screen_root(self) -> Path:
        return (
            self.tutorial_output_dir
            / f"{TUTORIAL_SURFACE_SCREEN_STEM}_{self.run_scope}"
            / "surface_screen"
        )

    @property
    def trajectory_validation_root(self) -> Path:
        return self.accuracy_output_dir / ACCURACY_TRAJECTORY_STEM

    @property
    def nh3_ranking_root(self) -> Path:
        return self.accuracy_output_dir / ACCURACY_NH3_RANKING_STEM


@dataclass(frozen=True)
class ArtifactValidation:
    """Completeness status for a saved result root."""

    root: Path
    complete: bool
    missing: tuple[str, ...]


def make_live_run_id(now: datetime | None = None) -> str:
    """Return a sortable timestamp run id."""
    now = now or datetime.now()
    return now.strftime(RUN_ID_FORMAT)


def _rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _relative_or_absolute(path: Path, tutorial_root: Path) -> Path:
    try:
        return path.resolve().relative_to(tutorial_root.resolve())
    except ValueError:
        return path


def _live_surface_screen_root(run_root: Path, run_scope: str) -> Path:
    return run_root / "tutorial" / f"{TUTORIAL_SURFACE_SCREEN_STEM}_{run_scope}" / "surface_screen"


def _live_trajectory_accuracy_root(run_root: Path) -> Path:
    return run_root / "accuracy" / ACCURACY_TRAJECTORY_STEM


def _live_nh3_ranking_root(run_root: Path) -> Path:
    return run_root / "accuracy" / ACCURACY_NH3_RANKING_STEM


def find_latest_complete_live_run(
    tutorial_root: Path,
    *,
    run_scope: str,
    artifact_kind: str,
) -> str:
    """Return the newest live run id with a complete requested artifact set."""
    live_runs_root = tutorial_root / LIVE_RUN_ROOT
    if not live_runs_root.exists():
        raise FileNotFoundError(f"No live runs found under {live_runs_root}.")

    for run_root in sorted((p for p in live_runs_root.iterdir() if p.is_dir()), reverse=True):
        if artifact_kind == "tutorial":
            validation = validate_surface_screen(_live_surface_screen_root(run_root, run_scope))
            if validation.complete:
                return run_root.name
        elif artifact_kind == "accuracy":
            trajectory = validate_trajectory_accuracy(_live_trajectory_accuracy_root(run_root))
            nh3 = validate_nh3_ranking(_live_nh3_ranking_root(run_root))
            if trajectory.complete and nh3.complete:
                return run_root.name
        else:
            raise ValueError(f"Unknown artifact kind: {artifact_kind}")

    raise FileNotFoundError(
        f"No complete {artifact_kind} live run found under {live_runs_root} for scope '{run_scope}'."
    )


def _resolve_saved_live_run_id(
    tutorial_root: Path,
    *,
    run_scope: str,
    saved_run_id: str,
    artifact_kind: str,
) -> str:
    if saved_run_id == LATEST_COMPLETE_RUN_ID:
        return find_latest_complete_live_run(
            tutorial_root,
            run_scope=run_scope,
            artifact_kind=artifact_kind,
        )
    return saved_run_id


def resolve_run_roots(
    *,
    tutorial_root: Path,
    run_scope: str,
    use_saved_tutorial_results: bool,
    use_saved_accuracy_results: bool,
    saved_tutorial_run_id: str | None,
    saved_accuracy_run_id: str | None,
    refresh_saved_results: bool,
    live_run_id: str | None = None,
) -> RunRoots:
    """Resolve output roots without guessing from timestamped scratch runs."""
    live_run_id = live_run_id or make_live_run_id()
    official_tutorial_root = tutorial_root / OFFICIAL_OUTPUT_ROOT / "tutorial"
    official_accuracy_root = tutorial_root / OFFICIAL_OUTPUT_ROOT / "accuracy"
    live_root = tutorial_root / LIVE_RUN_ROOT / live_run_id

    if refresh_saved_results:
        tutorial_output = official_tutorial_root
        accuracy_output = official_accuracy_root
        tutorial_label = "official precomputed cache refresh"
        accuracy_label = "official precomputed cache refresh"
    else:
        if use_saved_tutorial_results:
            if saved_tutorial_run_id:
                resolved_tutorial_run_id = _resolve_saved_live_run_id(
                    tutorial_root,
                    run_scope=run_scope,
                    saved_run_id=saved_tutorial_run_id,
                    artifact_kind="tutorial",
                )
                tutorial_output = tutorial_root / LIVE_RUN_ROOT / resolved_tutorial_run_id / "tutorial"
                tutorial_label = f"selected live run {resolved_tutorial_run_id}"
            else:
                tutorial_output = official_tutorial_root
                tutorial_label = "official precomputed cache"
        else:
            tutorial_output = live_root / "tutorial"
            tutorial_label = f"new live run {live_run_id}"

        if use_saved_accuracy_results:
            if saved_accuracy_run_id:
                resolved_accuracy_run_id = _resolve_saved_live_run_id(
                    tutorial_root,
                    run_scope=run_scope,
                    saved_run_id=saved_accuracy_run_id,
                    artifact_kind="accuracy",
                )
                accuracy_output = tutorial_root / LIVE_RUN_ROOT / resolved_accuracy_run_id / "accuracy"
                accuracy_label = f"selected live run {resolved_accuracy_run_id}"
            else:
                accuracy_output = official_accuracy_root
                accuracy_label = "official precomputed cache"
        else:
            accuracy_output = live_root / "accuracy"
            accuracy_label = f"new live run {live_run_id}"

    return RunRoots(
        run_scope=run_scope,
        live_run_id=live_run_id,
        tutorial_output_dir=_relative_or_absolute(tutorial_output, tutorial_root),
        accuracy_output_dir=_relative_or_absolute(accuracy_output, tutorial_root),
        cache_dir=_relative_or_absolute(tutorial_output / "cache_tables", tutorial_root),
        plots_dir=_relative_or_absolute(tutorial_output / "plots", tutorial_root),
        runtime_cache_dir=_relative_or_absolute(tutorial_root / RUNTIME_CACHE_ROOT, tutorial_root),
        tutorial_source_label=tutorial_label,
        accuracy_source_label=accuracy_label,
        official_tutorial_root=_relative_or_absolute(official_tutorial_root, tutorial_root),
        official_accuracy_root=_relative_or_absolute(official_accuracy_root, tutorial_root),
        live_root=_relative_or_absolute(live_root, tutorial_root),
    )


def validate_artifact_set(root: Path, required_files: Iterable[str]) -> ArtifactValidation:
    """Return the missing files for a saved artifact root."""
    root = Path(root)
    missing = tuple(str(name) for name in required_files if not (root / name).exists())
    return ArtifactValidation(root=root, complete=not missing, missing=missing)


def validate_surface_screen(root: Path) -> ArtifactValidation:
    """Validate a surface-screen cache root."""
    return validate_artifact_set(root, SURFACE_SCREEN_REQUIRED_FILES)


def validate_trajectory_accuracy(root: Path) -> ArtifactValidation:
    """Validate the closed-shell trajectory-replay accuracy root."""
    return validate_artifact_set(root, ACCURACY_TRAJECTORY_REQUIRED_FILES)


def validate_nh3_ranking(root: Path) -> ArtifactValidation:
    """Validate the NH3 fixed-geometry ranking accuracy root."""
    return validate_artifact_set(root, ACCURACY_NH3_REQUIRED_FILES)


def _run_manifest_path(run_root: Path) -> Path:
    return Path(run_root) / "run_manifest.json"


def write_run_manifest(
    run_root: Path,
    *,
    run_scope: str,
    tutorial_root: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a lightweight manifest for a live or precomputed run root."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_scope": run_scope,
        "root": str(_rel(run_root, tutorial_root)),
        "metadata": metadata or {},
    }
    path = _run_manifest_path(run_root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_artifact_index(root: Path) -> Path:
    """Write a CSV index of files below *root* for audit and promotion."""
    root = Path(root)
    path = root / "artifact_index.csv"
    rows = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        if file_path == path:
            continue
        rows.append(
            {
                "relative_path": file_path.relative_to(root).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    file_path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "modified_utc"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def list_live_runs(tutorial_root: Path, *, run_scope: str | None = None) -> list[dict[str, object]]:
    """Return manifest and completeness information for timestamped live runs."""
    root = tutorial_root / LIVE_RUN_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, object]] = []
    for run_root in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        manifest_path = _run_manifest_path(run_root)
        manifest: dict[str, object] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_scope = str(manifest.get("run_scope", ""))
        if run_scope is not None and manifest_scope and manifest_scope != run_scope:
            continue
        tutorial_root_for_run = _live_surface_screen_root(run_root, run_scope or manifest_scope)
        trajectory_root = _live_trajectory_accuracy_root(run_root)
        nh3_root = _live_nh3_ranking_root(run_root)
        tutorial = validate_surface_screen(tutorial_root_for_run)
        trajectory = validate_trajectory_accuracy(trajectory_root)
        nh3 = validate_nh3_ranking(nh3_root)
        rows.append(
            {
                "run_id": run_root.name,
                "run_scope": manifest_scope or "unknown",
                "manifest": manifest_path.exists(),
                "tutorial_complete": tutorial.complete,
                "accuracy_complete": trajectory.complete and nh3.complete,
                "missing_tutorial": "; ".join(tutorial.missing[:5]),
                "missing_accuracy": "; ".join((*trajectory.missing, *nh3.missing)[:5]),
            }
        )
    return rows


def require_complete(validation: ArtifactValidation, *, label: str) -> Path:
    """Return the root or raise an error listing the missing files."""
    if validation.complete:
        return validation.root
    preview = ", ".join(validation.missing[:6])
    more = "" if len(validation.missing) <= 6 else f" and {len(validation.missing) - 6} more"
    raise FileNotFoundError(
        f"Saved {label} cache is incomplete at {validation.root}. "
        f"Missing: {preview}{more}."
    )
