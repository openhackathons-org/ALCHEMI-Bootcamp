"""AIMNet2 checkpoint metadata and composition checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import sha256_file


EXPECTED_B973C_D3_BJ = {"s6": 1.0, "s8": 1.5, "a1": 0.37, "a2": 4.1}


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a registry name or local AIMNet checkpoint path."""

    from aimnet.calculators.model_registry import get_model_path

    return Path(get_model_path(str(checkpoint)))


def verify_checkpoint_identities(
    expected_identities: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Resolve checkpoint aliases and verify their fixed file identities."""

    records: dict[str, dict[str, object]] = {}
    for alias, expected in expected_identities.items():
        path = resolve_checkpoint_path(alias).resolve()
        observed = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        declared = {
            "filename": expected.get("filename"),
            "bytes": expected.get("bytes"),
            "sha256": expected.get("sha256"),
        }
        if observed != declared:
            raise RuntimeError(
                f"AIMNet checkpoint identity mismatch for {alias}: "
                f"{observed} != {declared}"
            )
        records[alias] = observed
    return records


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
    """Validate the released residual's Coulomb and pairwise-D3 settings."""

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
    """Return validated metadata plus the checkpoint path and checksum."""

    metadata = checkpoint_metadata(model)
    validate_b973c_external_components(metadata)
    path = Path(resolved_path)
    metadata.update(
        checkpoint_source=str(checkpoint_source),
        resolved_checkpoint_path=str(path),
        checkpoint_sha256=sha256_file(path),
    )
    return metadata


def aimnet_model_card_table(
    model_card: Mapping[str, Any],
    *,
    aimnet_version: str,
    cutoff_A: float,
    supports_pbc: bool,
    optional_inputs: Iterable[str],
    neighbor_convention: str,
    device: str,
) -> pd.DataFrame:
    """Format the current AIMNet2 lesson metadata as a two-column model card.

    Checkpoint loading, wrapper configuration, and model-output selection stay
    in the notebook.  The caller passes their observed values to this display
    helper after ``checkpoint_card`` has validated the checkpoint metadata.
    """

    required = {
        "checkpoint_source",
        "checkpoint_sha256",
        "needs_coulomb",
        "needs_dispersion",
    }
    missing = required - set(model_card)
    if missing:
        raise ValueError(f"model_card is missing fields: {sorted(missing)!r}")
    checkpoint_sha256 = str(model_card["checkpoint_sha256"])
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "checkpoint_sha256 must contain 64 hexadecimal characters"
        ) from exc

    rows = [
        ("checkpoint", model_card["checkpoint_source"]),
        ("checkpoint_sha256", checkpoint_sha256[:16] + "…"),
        (
            "package",
            f"aimnet {aimnet_version} through Toolkit AIMNet2Wrapper",
        ),
        ("code_license", "AIMNet software: MIT"),
        (
            "target",
            "B97-3c checkpoint base; declared Coulomb and D3 terms added later",
        ),
        (
            "domain",
            "molecular training domain; wrapper supports PBC, but "
            "condensed-phase accuracy is not established here",
        ),
        (
            "implemented_atomic_numbers",
            model_card.get("implemented_species"),
        ),
        ("weight_license", "MIT"),
        (
            "total_charge",
            "one explicit total charge per graph through Batch.charge",
        ),
        (
            "spin_multiplicity",
            "selected checkpoint is closed-shell; no multiplicity input is used",
        ),
        ("cutoff_A", cutoff_A),
        ("external_coulomb", model_card["needs_coulomb"]),
        ("external_dispersion", model_card["needs_dispersion"]),
        ("supports_pbc", supports_pbc),
        ("optional_inputs", sorted(map(str, optional_inputs))),
        ("neighbor_convention", str(neighbor_convention)),
        ("device", str(device)),
        (
            "coordinate_precision",
            "AIMNet2Wrapper evaluates positions in float32",
        ),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])
