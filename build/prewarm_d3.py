#!/usr/bin/env python3
"""Create and verify the Toolkit D3 parameter cache on first use."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path


EXPECTED_SHA256 = "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cache(parameter_file: Path) -> dict[str, object]:
    """Create the cache through the public Toolkit wrapper and verify its bytes."""

    from nvalchemi.models import DFTD3ModelWrapper

    parameter_file = parameter_file.expanduser().resolve()
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    # These PBE damping values only satisfy the wrapper constructor. No energy
    # evaluation is performed; the parameter tensor itself is functional-independent.
    wrapper = DFTD3ModelWrapper(
        a1=0.4289,
        a2=4.4407,
        s8=0.7875,
        auto_download=True,
        param_file=parameter_file,
    )
    del wrapper

    digest = sha256_file(parameter_file)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"D3 parameter SHA-256 mismatch: {digest} != {EXPECTED_SHA256}"
        )
    return {
        "schema": "alchemi.part1-d3-cache.v1",
        "parameter_file": str(parameter_file),
        "bytes": parameter_file.stat().st_size,
        "sha256": digest,
        "toolkit_version": metadata.version("nvalchemi-toolkit"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = prepare_cache(args.parameter_file)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
