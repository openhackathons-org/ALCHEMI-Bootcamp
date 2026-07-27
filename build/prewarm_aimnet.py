#!/usr/bin/env python3
"""Download and verify the AIMNet checkpoints used by Part 1."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
from typing import Any


PREFIX_CHECKPOINTS = ("aimnet2-b973c-2025-d3_0",)
NCI_CHECKPOINTS = tuple(f"aimnet2-wb97m-d3_{index}" for index in range(4))
CHECKPOINTS = (*PREFIX_CHECKPOINTS, *NCI_CHECKPOINTS)
CHECKPOINT_IDENTITIES = {
    "aimnet2-b973c-2025-d3_0": {
        "filename": "aimnet2_2025_b973c_d3_0.pt",
        "bytes": 8_839_102,
        "sha256": "043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5",
    },
    "aimnet2-wb97m-d3_0": {
        "filename": "aimnet2_wb97m_d3_0.pt",
        "bytes": 8_836_941,
        "sha256": "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28",
    },
    "aimnet2-wb97m-d3_1": {
        "filename": "aimnet2_wb97m_d3_1.pt",
        "bytes": 8_836_941,
        "sha256": "0505ec73a1759bd1d9885fa5f396c79f2a0cc93d20c62ea388e92b30e9351432",
    },
    "aimnet2-wb97m-d3_2": {
        "filename": "aimnet2_wb97m_d3_2.pt",
        "bytes": 8_836_941,
        "sha256": "a8c1ea27bd07e1bb1a5942f93cb7285cc7308c2a26fa4e1f82b7b5302feb7bfa",
    },
    "aimnet2-wb97m-d3_3": {
        "filename": "aimnet2_wb97m_d3_3.pt",
        "bytes": 8_836_941,
        "sha256": "0c5f26b9de15c72a9ccb1d411b3b7814816bb06cca46f784dccd1f5efb2c334b",
    },
}
SHARED_NCI_METADATA_FIELDS = (
    "cutoff",
    "needs_coulomb",
    "needs_dispersion",
    "coulomb_mode",
    "coulomb_sr_rc",
    "coulomb_sr_envelope",
    "d3_params",
    "implemented_species",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prewarm(*, device: str) -> dict[str, Any]:
    """Resolve every checkpoint, load it once, and return an identity report."""

    from aimnet.calculators.model_registry import (
        get_cache_dir,
        get_model_path,
        load_model_registry,
        resolve_registry_model_name,
    )
    from nvalchemi.models import AIMNet2Wrapper

    registry = load_model_registry()
    cache_dir = Path(get_cache_dir()).resolve()
    rows: list[dict[str, Any]] = []
    for alias in CHECKPOINTS:
        canonical_name = resolve_registry_model_name(alias)
        registry_row = registry["models"][canonical_name]
        checkpoint_path = Path(get_model_path(alias)).resolve()
        if checkpoint_path.parent != cache_dir:
            raise RuntimeError(
                f"{alias} resolved outside AIMNET_CACHE_DIR: {checkpoint_path}"
            )
        digest = sha256_file(checkpoint_path)
        expected_identity = CHECKPOINT_IDENTITIES[alias]
        if checkpoint_path.name != expected_identity["filename"]:
            raise RuntimeError(
                f"{alias} filename mismatch: "
                f"{checkpoint_path.name} != {expected_identity['filename']}"
            )
        byte_count = checkpoint_path.stat().st_size
        if byte_count != expected_identity["bytes"]:
            raise RuntimeError(
                f"{alias} byte-size mismatch: "
                f"{byte_count} != {expected_identity['bytes']}"
            )
        if digest != expected_identity["sha256"]:
            raise RuntimeError(
                f"{alias} pinned SHA-256 mismatch: "
                f"{digest} != {expected_identity['sha256']}"
            )
        expected_digest = registry_row.get("sha256")
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError(
                f"{alias} SHA-256 mismatch: {digest} != {expected_digest}"
            )

        wrapper = AIMNet2Wrapper.from_checkpoint(
            checkpoint_path,
            device=device,
            compile_model=False,
        ).eval()
        model_metadata = dict(wrapper.model.metadata)
        if not model_metadata.get("needs_coulomb"):
            raise RuntimeError(f"{alias} does not request external Coulomb")
        if not model_metadata.get("needs_dispersion"):
            raise RuntimeError(f"{alias} does not request external dispersion")

        rows.append(
            {
                "alias": alias,
                "canonical_name": canonical_name,
                "path": str(checkpoint_path),
                "filename": checkpoint_path.name,
                "bytes": byte_count,
                "sha256": digest,
                "registry_sha256": expected_digest,
                "source_url": registry_row["url"],
                "scope": "notebook prefix" if alias in PREFIX_CHECKPOINTS else "NCI stage",
                "format_version": model_metadata.get("format_version"),
                "metadata": {
                    field: model_metadata.get(field)
                    for field in SHARED_NCI_METADATA_FIELDS
                },
                "coulomb_mode": model_metadata.get("coulomb_mode"),
                "needs_coulomb": model_metadata.get("needs_coulomb"),
                "needs_dispersion": model_metadata.get("needs_dispersion"),
            }
        )
        del wrapper

    nci_rows = [row for row in rows if row["alias"] in NCI_CHECKPOINTS]
    reference_metadata = nci_rows[0]["metadata"]
    for row in nci_rows[1:]:
        if row["metadata"] != reference_metadata:
            raise RuntimeError(
                "NCI ensemble members do not share the same Coulomb and D3 "
                f"convention: {nci_rows[0]['alias']} != {row['alias']}"
            )

    return {
        "schema": "alchemi.part1-aimnet-prewarm.v1",
        "device": device,
        "cache_dir": str(cache_dir),
        "checkpoints_retained": True,
        "aimnet_version": metadata.version("aimnet"),
        "toolkit_version": metadata.version("nvalchemi-toolkit"),
        "nci_shared_metadata": reference_metadata,
        "checkpoints": rows,
    }


def remove_verified_checkpoints(report: dict[str, Any]) -> list[str]:
    """Remove only the checkpoint files verified by :func:`prewarm`."""

    cache_dir = Path(report["cache_dir"]).resolve()
    removed: list[str] = []
    for row in report["checkpoints"]:
        path = Path(row["path"]).resolve()
        if path.parent != cache_dir:
            raise RuntimeError(f"refusing to remove a file outside {cache_dir}: {path}")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"checkpoint changed after validation: {path}")
        path.unlink()
        removed.append(str(path))
    report["checkpoints_retained"] = False
    report["removed_checkpoint_paths"] = removed
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--remove-checkpoints-after-test",
        action="store_true",
        help="Delete the downloaded checkpoint files after validation.",
    )
    args = parser.parse_args()

    report = prewarm(device=args.device)
    if args.remove_checkpoints_after_test:
        remove_verified_checkpoints(report)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
