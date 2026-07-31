#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Execute a notebook one code cell at a time and save outputs.

This is intentionally small: it keeps notebook execution reproducible from a
plain uv/venv Python without requiring an interactive Jupyter browser session.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    code_count = 0
    with client.setup_kernel():
        for cell_index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            code_count += 1
            first_line = next(
                (line.strip() for line in cell.source.splitlines() if line.strip()),
                "<empty>",
            )
            print(
                f"[cell {cell_index:02d} | code {code_count:02d}] start: {first_line[:90]}",
                flush=True,
            )
            started = time.time()
            try:
                client.execute_cell(cell, cell_index, execution_count=code_count)
            except CellExecutionError:
                nbformat.write(nb, output_path)
                print(
                    f"[cell {cell_index:02d} | code {code_count:02d}] failed; "
                    f"partial notebook written to {output_path}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            elapsed = time.time() - started
            print(
                f"[cell {cell_index:02d} | code {code_count:02d}] done in {elapsed:.1f}s",
                flush=True,
            )

    nbformat.write(nb, output_path)
    print(f"Executed notebook written to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
