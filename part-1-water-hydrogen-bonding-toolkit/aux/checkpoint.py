"""AIMNet2 checkpoint provenance and composition-contract checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


EXPECTED_B973C_D3_BJ = {"s6": 1.0, "s8": 1.5, "a1": 0.37, "a2": 4.1}


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a registry name or local AIMNet checkpoint path."""

    from aimnet.calculators.model_registry import get_model_path

    return Path(get_model_path(str(checkpoint)))


def checkpoint_metadata(model: Any) -> dict[str, Any]:
    """Read metadata from an AIMNet2 wrapper, including compiled models."""

    raw = model.model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    metadata = getattr(raw, "metadata", None)
    if metadata is None:
        metadata = getattr(raw, "_metadata", None)
    if metadata is None:
        raise RuntimeError("AIMNet2 checkpoint metadata is required")
    return dict(metadata)


def validate_b973c_external_components(
    metadata: Mapping[str, Any],
) -> dict[str, float]:
    """Validate the released residual's Coulomb and pairwise-D3 contract."""

    if metadata.get("needs_coulomb") is not True:
        raise ValueError("Expected an AIMNet2 checkpoint with external Coulomb")
    if metadata.get("needs_dispersion") is not True:
        raise ValueError("Expected an AIMNet2 checkpoint with external D3")
    if metadata.get("coulomb_mode") != "sr_embedded":
        raise ValueError("Expected the released sr_embedded Coulomb convention")
    if abs(float(metadata.get("coulomb_sr_rc", -1.0)) - 4.6) > 1e-5:
        raise ValueError("Unexpected embedded short-range Coulomb cutoff")

    raw_d3 = metadata.get("d3_params")
    if not isinstance(raw_d3, Mapping):
        raise ValueError("Checkpoint metadata does not contain D3 parameters")
    d3_params = {key: float(value) for key, value in raw_d3.items()}
    for key, expected in EXPECTED_B973C_D3_BJ.items():
        if key not in d3_params:
            raise ValueError(f"Checkpoint D3 parameters are missing {key}")
        if abs(d3_params[key] - expected) > 1e-7:
            raise ValueError(f"Unexpected B97-3c D3 parameter {key}={d3_params[key]!r}")
    return d3_params


def checkpoint_card(
    model: Any,
    checkpoint_source: str | Path,
    resolved_path: str | Path,
) -> dict[str, Any]:
    """Return validated metadata plus immutable checkpoint provenance."""

    metadata = checkpoint_metadata(model)
    validate_b973c_external_components(metadata)
    path = Path(resolved_path)
    metadata.update(
        checkpoint_source=str(checkpoint_source),
        resolved_checkpoint_path=str(path),
        checkpoint_sha256=sha256_file(path),
    )
    return metadata
