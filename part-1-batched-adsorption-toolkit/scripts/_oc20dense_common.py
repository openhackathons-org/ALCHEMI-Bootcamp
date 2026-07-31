#!/usr/bin/env python3
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
"""Shared constants and provenance helpers for OC20Dense benchmark scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PART1 = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PART1 / "data" / "reference" / "oc20dense"
DEFAULT_TRAJECTORY_ARCHIVE = DEFAULT_DATA_ROOT / "raw_archives" / "oc20_dense_trajectories.tar.gz"
DEFAULT_EXTRACT_DIR = DEFAULT_DATA_ROOT / "selected_trajectories" / "adslab"
DEFAULT_SURFACE_DIR = DEFAULT_DATA_ROOT / "selected_trajectories" / "surfaces"
DEFAULT_INITIAL_STRUCTURE_DIR = DEFAULT_DATA_ROOT / "initial_structures" / "adslab"
DEFAULT_LMDB_PATH = DEFAULT_DATA_ROOT / "lmdb" / "oc20dense.lmdb"
FULL_DATA_ROOT_ENV = "OC20DENSE_FULL_DATA_ROOT"
FULL_DATA_NOTICE = (
    "This checkout keeps only the slim OC20Dense validation subset. "
    "Requests outside that subset require the full OC20Dense archives/LMDB "
    "(about 40 GB locally). Set OC20DENSE_FULL_DATA_ROOT to an extracted full "
    "OC20Dense data tree if you need additional system/config ids."
)

DEFAULT_CLOSED_SHELL_SYSTEMS = ("3_2070_48", "72_7104_115", "69_1615_2")
DEFAULT_SYSTEMS = DEFAULT_CLOSED_SHELL_SYSTEMS
CLOSED_SHELL_ADSORBATE_REFERENCES = {
    "*OH2": "H2O",
    "*NH3": "NH3",
    "*N2": "N2",
}
MACE_RANK_BASIS = "total_energy_within_fixed_system"
MACE_EADS_REFERENCE_STATUS = "defined_mace_eads_official_surface_neutral_gas_refs"
TOOLKIT_PROVENANCE_COLUMNS = (
    "toolkit_checkpoint",
    "toolkit_head",
    "toolkit_device",
    "toolkit_dtype",
    "toolkit_d3bj_enabled",
)


def toolkit_provenance_from_env(*, d3bj_enabled: bool = False) -> dict[str, Any]:
    """Return the active Toolkit/MACE settings that affect model energies."""
    return {
        "toolkit_checkpoint": os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
        "toolkit_head": os.environ.get("TOOLKIT_HEAD") or None,
        "toolkit_device": os.environ.get("TOOLKIT_DEVICE", "cuda"),
        "toolkit_dtype": os.environ.get("TOOLKIT_DTYPE", "float32"),
        "toolkit_d3bj_enabled": bool(d3bj_enabled),
    }


def toolkit_model_label(provenance: dict[str, Any]) -> str:
    """Return a compact checkpoint/head label for reports."""
    head = provenance.get("toolkit_head")
    checkpoint = provenance.get("toolkit_checkpoint", "medium-mpa-0")
    return f"{checkpoint} (head={head})" if head else str(checkpoint)


def toolkit_provenance_mismatch(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return observed/expected differences for model-defining settings."""
    return {
        key: (observed.get(key), expected.get(key))
        for key in TOOLKIT_PROVENANCE_COLUMNS
        if observed.get(key) != expected.get(key)
    }


def toolkit_cache_matches(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Return True only when cached model outputs match active Toolkit settings."""
    return not toolkit_provenance_mismatch(observed, expected)


def oc20dense_mapping_file(data_root: Path, name: str) -> Path:
    """Return a mapping/reference file path, accepting old and new layouts."""
    full_data_root = os.environ.get(FULL_DATA_ROOT_ENV)
    candidates = [
        Path(data_root) / "mappings" / name,
        Path(data_root) / name,
    ]
    if full_data_root:
        candidates.extend(
            [
                Path(full_data_root) / "mappings" / name,
                Path(full_data_root) / name,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_precomputed_write_allowed(*paths: Path) -> None:
    """Refuse accidental writes into official saved output roots."""
    precomputed_root = PART1 / "outputs" / "precomputed"
    if not any(_is_relative_to(Path(path), precomputed_root) for path in paths):
        return
    if (
        _env_flag_enabled("REFRESH_SAVED_RESULTS")
        or _env_flag_enabled("ALCHEMI_ALLOW_ARTIFACT_OVERWRITE")
        or _env_flag_enabled("ALCHEMI_ALLOW_PRECOMPUTED_WRITE")
    ):
        return
    raise PermissionError(
        "Refusing to write under outputs/precomputed without an explicit refresh "
        "guard. Use a live-run output directory, or set REFRESH_SAVED_RESULTS=1 "
        "or ALCHEMI_ALLOW_PRECOMPUTED_WRITE=1 for an intentional official refresh."
    )


def oc20dense_archive_file(data_root: Path, name: str) -> Path:
    """Return an archive path, accepting old and new layouts."""
    full_data_root = os.environ.get(FULL_DATA_ROOT_ENV)
    candidates = [
        Path(data_root) / "raw_archives" / name,
        Path(data_root) / name,
    ]
    if full_data_root:
        candidates.extend(
            [
                Path(full_data_root) / "raw_archives" / name,
                Path(full_data_root) / name,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def oc20dense_lmdb_path(data_root: Path) -> Path:
    """Return the OC20Dense LMDB path, accepting old and new layouts."""
    full_data_root = os.environ.get(FULL_DATA_ROOT_ENV)
    candidates = [
        Path(data_root) / "lmdb" / "oc20dense.lmdb",
        Path(data_root) / "oc20_dense_data" / "data" / "oc20dense.lmdb",
    ]
    if full_data_root:
        candidates.extend(
            [
                Path(full_data_root) / "lmdb" / "oc20dense.lmdb",
                Path(full_data_root) / "oc20_dense_data" / "data" / "oc20dense.lmdb",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
