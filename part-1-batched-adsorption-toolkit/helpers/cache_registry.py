# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run-cache registry for the batched adsorption tutorial.

The notebook has three distinct storage modes:

* official precomputed artifacts under ``outputs/precomputed``;
* explicit, timestamped live runs under ``outputs/live_runs/<run_id>``;
* runtime/model caches under ``outputs/runtime_cache``.

Path derivation lives in :mod:`helpers.config` on the ``Config`` Pydantic
model. This module owns the path-layout constants, artifact-completeness
checks, manifest IO, and the "latest complete run" discovery routine those
checks compose into.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# --- Path layout -----------------------------------------------------------

RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
LATEST_COMPLETE_RUN_ID = "latest-complete"

OFFICIAL_OUTPUT_ROOT = Path("outputs") / "precomputed"
LIVE_RUN_ROOT = Path("outputs") / "live_runs"
RUNTIME_CACHE_ROOT = Path("outputs") / "runtime_cache"

TUTORIAL_DIR_NAME = "tutorial"
ACCURACY_DIR_NAME = "accuracy"
CACHE_TABLES_DIR_NAME = "cache_tables"
PLOTS_DIR_NAME = "plots"
SURFACE_SCREEN_DIR_NAME = "surface_screen"

TUTORIAL_SURFACE_SCREEN_STEM = "surface_screen_v1_mace_mpa0"
ACCURACY_TRAJECTORY_STEM = "oc20dense_closed_shell_trajectory_mace_mpa0"
ACCURACY_NH3_RANKING_STEM = "oc20dense_nh3_92_fixed_geometry_mace_mpa0"

# --- Source-label templates used by Config.{TUTORIAL,ACCURACY}_SOURCE_LABEL -

LABEL_OFFICIAL_REFRESH = "official precomputed cache refresh"
LABEL_OFFICIAL = "official precomputed cache"
LABEL_NEW_LIVE_RUN = "new live run {run_id}"
LABEL_SELECTED_LIVE_RUN = "selected live run {run_id}"

# --- Required-files manifests for completeness validation ------------------

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


# --- Artifact validation ---------------------------------------------------


@dataclass(frozen=True)
class ArtifactValidation:
    """Completeness status for a saved result root."""

    root: Path
    complete: bool
    missing: tuple[str, ...]


def _relative_or_absolute(path: Path, tutorial_root: Path) -> Path:
    try:
        return path.resolve().relative_to(tutorial_root.resolve())
    except ValueError:
        return path


def _live_surface_screen_root(run_root: Path, run_scope: str) -> Path:
    return (
        run_root
        / TUTORIAL_DIR_NAME
        / f"{TUTORIAL_SURFACE_SCREEN_STEM}_{run_scope}"
        / SURFACE_SCREEN_DIR_NAME
    )


def _live_trajectory_accuracy_root(run_root: Path) -> Path:
    return run_root / ACCURACY_DIR_NAME / ACCURACY_TRAJECTORY_STEM


def _live_nh3_ranking_root(run_root: Path) -> Path:
    return run_root / ACCURACY_DIR_NAME / ACCURACY_NH3_RANKING_STEM


def validate_artifact_set(
    root: Path, required_files: Iterable[str]
) -> ArtifactValidation:
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


def require_complete(validation: ArtifactValidation, *, label: str) -> Path:
    """Return the root or raise an error listing the missing files."""
    if validation.complete:
        return validation.root
    preview = ", ".join(validation.missing[:6])
    more = (
        ""
        if len(validation.missing) <= 6
        else f" and {len(validation.missing) - 6} more"
    )
    raise FileNotFoundError(
        f"Saved {label} cache is incomplete at {validation.root}. "
        f"Missing: {preview}{more}."
    )


# --- Live-run discovery ----------------------------------------------------


def find_latest_complete_live_run(
    tutorial_root: Path,
    run_scope: str,
    artifact_kind: str,
) -> str:
    """Return the newest live run id with a complete requested artifact set."""
    live_runs_root = tutorial_root / LIVE_RUN_ROOT
    if not live_runs_root.exists():
        raise FileNotFoundError(f"No live runs found under {live_runs_root}.")

    for run_root in sorted(
        (p for p in live_runs_root.iterdir() if p.is_dir()), reverse=True
    ):
        if artifact_kind == "tutorial":
            validation = validate_surface_screen(
                _live_surface_screen_root(run_root, run_scope)
            )
            if validation.complete:
                return run_root.name
        elif artifact_kind == "accuracy":
            trajectory = validate_trajectory_accuracy(
                _live_trajectory_accuracy_root(run_root)
            )
            nh3 = validate_nh3_ranking(_live_nh3_ranking_root(run_root))
            if trajectory.complete and nh3.complete:
                return run_root.name
        else:
            raise ValueError(f"Unknown artifact kind: {artifact_kind}")

    raise FileNotFoundError(
        f"No complete {artifact_kind} live run found under {live_runs_root} for scope '{run_scope}'."
    )


# --- Manifest / artifact-index IO ------------------------------------------


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
        "root": str(_relative_or_absolute(run_root, tutorial_root)),
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
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "modified_utc"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
