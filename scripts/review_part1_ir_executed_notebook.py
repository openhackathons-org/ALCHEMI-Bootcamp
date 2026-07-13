#!/usr/bin/env python3
"""Apply a Markdown-only source refresh to an executed Part 1 notebook."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import nbformat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calculation-job-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = nbformat.read(args.source, as_version=4)
    executed = nbformat.read(args.executed, as_version=4)
    if len(source.cells) != len(executed.cells):
        raise RuntimeError("source and executed notebook cell counts differ")

    reviewed = copy.deepcopy(source)
    code_count = 0
    for source_cell, executed_cell, reviewed_cell in zip(
        source.cells, executed.cells, reviewed.cells, strict=True
    ):
        identity = (source_cell.cell_type, source_cell.get("id"))
        if identity != (executed_cell.cell_type, executed_cell.get("id")):
            raise RuntimeError(f"cell identity mismatch: {identity!r}")
        if source_cell.cell_type != "code":
            continue
        code_count += 1
        if source_cell.source != executed_cell.source:
            raise RuntimeError(
                f"code source changed for cell {source_cell.get('id')!r}"
            )
        reviewed_cell.execution_count = executed_cell.execution_count
        reviewed_cell.outputs = copy.deepcopy(executed_cell.outputs)
        if reviewed_cell.execution_count is None:
            raise RuntimeError(f"code cell {source_cell.get('id')!r} was not executed")

    reviewed.metadata["alchemi_review"] = {
        "kind": "markdown-only-source-refresh",
        "calculation_job_id": str(args.calculation_job_id),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "original_executed_sha256": sha256_file(args.executed),
        "code_cell_count": code_count,
        "code_sources_unchanged": True,
        "reason": (
            "Generalized run-specific validation and interpretation prose after "
            "the definitive trajectory; numerical code and outputs are unchanged."
        ),
    }
    nbformat.validate(reviewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(reviewed, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
