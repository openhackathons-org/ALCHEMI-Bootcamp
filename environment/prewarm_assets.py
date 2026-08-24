# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and verify model assets for the Core playbook."""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aimnet.calculators.model_registry import get_model_path

if __package__:
    from .runtime_files import require_file_sha256, sha256_file
else:
    from runtime_files import require_file_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PINS = tomllib.loads((ROOT / "environment" / "runtime-pins.toml").read_text())


def ensure_pinned_asset(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    create: Callable[[Path], None],
    publish_link: Callable[[Path, Path], None] = os.link,
) -> Path:
    """Create a missing asset in staging, verify it, and publish it atomically."""
    path = path.expanduser().resolve()
    if path.exists():
        require_file_sha256(label, path, expected_sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.stem}-", dir=path.parent
    ) as staging:
        staged_path = Path(staging) / path.name
        create(staged_path)
        require_file_sha256(label, staged_path, expected_sha256)
        try:
            # A hard link publishes the fully verified inode atomically and
            # fails with EEXIST instead of replacing a concurrent winner.
            publish_link(staged_path, path)
        except FileExistsError:
            require_file_sha256(label, path, expected_sha256)
        except OSError as error:
            raise RuntimeError(
                f"{label}: atomic no-clobber publication via hard link is "
                f"unavailable in {path.parent}: {error}"
            ) from error

    require_file_sha256(label, path, expected_sha256)
    return path


def generate_d3_parameter_file(path: Path) -> None:
    """Generate D3 parameters through Toolkit's public extraction helpers."""
    from nvalchemi.models.dftd3 import (
        extract_dftd3_parameters,
        save_dftd3_parameters,
    )

    parameters = extract_dftd3_parameters()
    saved_path = save_dftd3_parameters(parameters, param_file=path).resolve()
    if saved_path != path.resolve():
        raise RuntimeError(f"D3 parameters saved to unexpected path: {saved_path}")


def resolve_mace_checkpoint(alias: str) -> str | Path:
    """Resolve a named checkpoint through MACE's official cache helper."""

    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    return download_mace_mp_checkpoint(alias)


def main(
    *,
    pins: dict[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    model_resolver: Callable[[str], str | Path] = get_model_path,
    mace_model_resolver: Callable[[str], str | Path] = resolve_mace_checkpoint,
    d3_generator: Callable[[Path], None] = generate_d3_parameter_file,
) -> None:
    """Create and verify the fixed AIMNet, MACE, and D3 runtime assets."""
    pins = PINS if pins is None else pins
    environment = os.environ if environment is None else environment
    model = pins["model"]
    path = Path(model_resolver(model["checkpoint_alias"])).resolve()
    actual = sha256_file(path)
    expected = model["checkpoint_sha256"]
    if actual != expected:
        raise RuntimeError(f"AIMNet checkpoint: expected {expected}, found {actual}")
    print(f"AIMNet checkpoint verified: {path}")

    mace = pins["mace"]
    mace_path = Path(mace_model_resolver(mace["checkpoint_alias"])).resolve()
    require_file_sha256(
        "MACE checkpoint",
        mace_path,
        mace["checkpoint_sha256"],
    )
    print(f"MACE checkpoint verified: {mace_path}")

    configured_d3_path = environment.get("ALCHEMI_D3_PARAM_FILE")
    if not configured_d3_path:
        raise RuntimeError("ALCHEMI_D3_PARAM_FILE is not configured")
    d3_path = ensure_pinned_asset(
        Path(configured_d3_path),
        expected_sha256=pins["dispersion"]["generated_parameter_sha256"],
        label="D3 parameters",
        create=d3_generator,
    )
    print(f"D3 parameters verified: {d3_path}")


if __name__ == "__main__":
    main()
