#!/usr/bin/env python3
"""Repair saved notebook outputs that omit schema-required fields.

Some cell runners write the minimum an output needs to display and skip keys the
nbformat v4 schema marks required, which makes the notebook fail
``nbformat.validate``:

    stream          requires output_type, name, text
    execute_result  requires output_type, data, metadata, execution_count
    display_data    requires output_type, data, metadata

This fills in only the missing keys and leaves output payloads untouched. It also
clears execution counts on code cells whose outputs were cleared, so a cell never
claims a run it no longer shows.

Usage:
    python scripts/repair_notebook_outputs.py NOTEBOOK [NOTEBOOK ...]
    python scripts/repair_notebook_outputs.py --check NOTEBOOK

``--check`` reports without writing and exits non-zero if a repair is needed,
which makes it usable as a pre-commit gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


def repair(notebook: dict) -> list[str]:
    """Fill missing required output keys. Returns one label per repair made."""

    repairs: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue

        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            missing: list[str] = []
            if kind == "stream" and "name" not in output:
                # Assume stdout: a runner that drops the channel is far more
                # likely to be printing than to be reporting stderr.
                output["name"] = "stdout"
                missing.append("name")
            if kind in ("display_data", "execute_result") and "metadata" not in output:
                output["metadata"] = {}
                missing.append("metadata")
            if kind == "execute_result" and "execution_count" not in output:
                output["execution_count"] = cell.get("execution_count")
                missing.append("execution_count")
            if missing:
                repairs.append(f"{cell.get('id')}:{kind}:{'+'.join(missing)}")

        if not cell.get("outputs") and cell.get("execution_count") is not None:
            cell["execution_count"] = None
            repairs.append(f"{cell.get('id')}:cleared stale execution_count")

    return repairs


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    paths = [Path(arg) for arg in argv if not arg.startswith("-")]
    if not paths:
        print(__doc__)
        return 2

    needed_repair = False
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        repairs = repair(notebook)
        node = nbformat.from_dict(notebook)
        nbformat.validate(node)

        if not repairs:
            print(f"{path.name}: already valid")
            continue

        needed_repair = True
        if check_only:
            print(f"{path.name}: {len(repairs)} repair(s) needed")
        else:
            nbformat.write(node, path)
            print(f"{path.name}: repaired {len(repairs)}")
        for label in repairs:
            print(f"  {label}")

    return 1 if (check_only and needed_repair) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
