# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paths and checksum-verified data records shared by Core helper modules."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_science_context_record(name: str) -> dict[str, Any]:
    """Load one checked scientific context record from the Core data directory."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(
            "science context record names use lowercase letters, numbers, - and _"
        )
    path = (
        PROJECT_ROOT
        / "notebooks"
        / "data"
        / "science_context"
        / f"{name}.json"
    )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load science context record: {path}") from error
    if not isinstance(record, dict):
        raise TypeError(f"Science context record must be a JSON object: {path}")
    return record
