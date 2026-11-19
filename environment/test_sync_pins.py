# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every repeated pin equal to environment/runtime-pins.toml."""

from __future__ import annotations

import shutil
from pathlib import Path

import sync_pins

ROOT = Path(__file__).resolve().parents[1]
SYNCED_FILES = (
    "environment/runtime-pins.toml",
    "pyproject.toml",
    "uv.lock",
    "THIRD_PARTY_NOTICES.md",
    "SOURCES_AND_LICENSES.md",
)


def _copy_repository_files(destination: Path) -> None:
    for name in SYNCED_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def test_repository_files_match_runtime_pins() -> None:
    assert sync_pins.sync(ROOT, check=True) == []


def test_sync_rewrites_a_changed_toolkit_commit(tmp_path: Path) -> None:
    _copy_repository_files(tmp_path)
    pins_path = tmp_path / "environment" / "runtime-pins.toml"
    old_commit = sync_pins.load_pins(pins_path)["toolkit"]["commit"]
    new_commit = "0" * 40
    pins_path.write_text(pins_path.read_text().replace(old_commit, new_commit))

    changes = sync_pins.sync(tmp_path, check=False)

    assert any("Toolkit requirement" in change for change in changes)
    assert f"nvalchemi-toolkit.git@{new_commit}" in (
        tmp_path / "pyproject.toml"
    ).read_text()
    assert f"nvalchemi-toolkit/blob/{new_commit}/LICENSE" in (
        tmp_path / "THIRD_PARTY_NOTICES.md"
    ).read_text()
    remaining = sync_pins.sync(tmp_path, check=True)
    assert remaining
    assert all(message.startswith("uv.lock:") for message in remaining)
