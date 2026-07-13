#!/usr/bin/env python3
"""Execute a notebook cell-by-cell with no per-cell timeout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="alchemi-main")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.reset_execution_trackers()

    code_count = 0
    with client.setup_kernel():
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            code_count += 1
            first_line = next(
                (
                    line.strip()
                    for line in cell.source.splitlines()
                    if line.strip()
                ),
                "<empty>",
            )
            print(
                f"[cell {cell_index:02d} | code {code_count:02d}] start: "
                f"{first_line[:100]}",
                flush=True,
            )
            started = time.monotonic()
            try:
                client.execute_cell(cell, cell_index, execution_count=code_count)
            except CellExecutionError:
                client.set_widgets_metadata()
                nbformat.write(notebook, output_path)
                print(
                    f"cell {cell_index} failed; partial notebook saved to "
                    f"{output_path}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            client.set_widgets_metadata()
            nbformat.write(notebook, output_path)
            print(
                f"[cell {cell_index:02d} | code {code_count:02d}] done in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    print(f"executed notebook saved to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
