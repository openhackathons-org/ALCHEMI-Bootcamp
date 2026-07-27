#!/usr/bin/env python3
"""Execute a notebook cell-by-cell and save a machine-readable timing report."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import nbformat
from nbclient import NotebookClient


TIMING_SCHEMA = "alchemi.part1-notebook-timing.v1"
_STAGE_ID = re.compile(r"stage-(\d+)")
_STAGE_TITLE = re.compile(r"<h2\b[^>]*>([^<]+)</h2>")
CUDA_SYNCHRONIZE_SOURCE = """\
if (
    "torch" in globals()
    and hasattr(torch, "cuda")
    and torch.cuda.is_available()
):
    torch.cuda.synchronize()
"""


class CellSynchronizationError(RuntimeError):
    """Raised when a timing-boundary synchronization cannot complete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute_temporary_cell(
    client: Any,
    notebook: Any,
    source: str,
) -> None:
    """Execute unrecorded helper code without changing learner cells."""

    temporary_cell = nbformat.v4.new_code_cell(source)
    notebook.cells.append(temporary_cell)
    temporary_index = len(notebook.cells) - 1
    try:
        client.execute_cell(
            temporary_cell,
            temporary_index,
            execution_count=None,
            store_history=False,
        )
    finally:
        notebook.cells.pop()


def stage_contexts(
    notebook: Any,
    *,
    expected_stages: tuple[int, ...] = tuple(range(1, 8)),
) -> dict[int, tuple[int, str]]:
    """Map each code-cell index to the stage that precedes it."""

    current_stage = 0
    current_title = "Setup"
    contexts: dict[int, tuple[int, str]] = {}
    observed_stages: list[int] = []
    for cell_index, cell in enumerate(notebook.cells):
        cell_id = str(cell.get("id", ""))
        match = _STAGE_ID.fullmatch(cell_id)
        if match is not None:
            current_stage = int(match.group(1))
            observed_stages.append(current_stage)
            title_match = _STAGE_TITLE.search(cell.source)
            current_title = (
                html.unescape(title_match.group(1))
                if title_match is not None
                else f"Stage {current_stage}"
            )
        if cell.cell_type == "code":
            contexts[cell_index] = (current_stage, current_title)
    if tuple(observed_stages) != expected_stages:
        raise ValueError(
            "notebook stage IDs must appear once in order: "
            f"expected {expected_stages}, found {tuple(observed_stages)}"
        )
    return contexts


def refresh_timing_summary(report: dict[str, Any]) -> None:
    """Recompute counts and stage totals from the cell records."""

    records = report["cell_timings"]
    report["code_cells_started"] = len(records)
    report["code_cells_completed"] = sum(
        record["status"] == "complete" for record in records
    )
    report["code_cells_failed"] = sum(
        record["status"] == "failed" for record in records
    )
    report["total_code_elapsed_s"] = float(
        sum(
            record["elapsed_s"]
            for record in records
            if record["elapsed_s"] is not None
        )
    )

    grouped: dict[int, dict[str, Any]] = {}
    for record in records:
        stage = int(record["stage"])
        item = grouped.setdefault(
            stage,
            {
                "stage": stage,
                "title": record["stage_title"],
                "code_cells_started": 0,
                "code_cells_completed": 0,
                "code_cells_failed": 0,
                "elapsed_s": 0.0,
            },
        )
        item["code_cells_started"] += 1
        item["code_cells_completed"] += record["status"] == "complete"
        item["code_cells_failed"] += record["status"] == "failed"
        if record["elapsed_s"] is not None:
            item["elapsed_s"] += record["elapsed_s"]
    report["stage_timings"] = [grouped[stage] for stage in sorted(grouped)]


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[..., Any] = NotebookClient,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--kernel", default="alchemi-main")
    args = parser.parse_args(argv)

    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    timing_path = args.timing_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    contexts = stage_contexts(notebook)
    expected_code_cells = sum(
        cell.cell_type == "code" for cell in notebook.cells
    )
    report: dict[str, Any] = {
        "schema": TIMING_SCHEMA,
        "status": "starting",
        "started_utc": utc_now(),
        "finished_utc": None,
        "input_notebook": args.notebook.as_posix(),
        "input_notebook_sha256": sha256_file(notebook_path),
        "executed_notebook": args.output.as_posix(),
        "executed_notebook_sha256": None,
        "kernel": args.kernel,
        "code_cell_count_expected": expected_code_cells,
        "code_cells_started": 0,
        "code_cells_completed": 0,
        "code_cells_failed": 0,
        "total_code_elapsed_s": 0.0,
        "total_wall_elapsed_s": None,
        "cell_timing_boundary": (
            "Host monotonic time starts after a conditional temporary "
            "torch.cuda.synchronize() immediately before each learner code "
            "cell and ends after a second conditional synchronization. It "
            "includes learner-cell execution, output transfer, and CUDA work "
            "queued by that cell. It excludes the pre-cell synchronization, "
            "temporary cells from learner-cell counts, and notebook and "
            "timing-report serialization."
        ),
        "wall_timing_boundary": (
            "From before kernel setup through kernel shutdown; includes cell "
            "execution and per-cell persistence."
        ),
        "runner_error_type": None,
        "runner_error_message": None,
        "stage_timings": [],
        "cell_timings": [],
    }
    atomic_write_json(timing_path, report)
    client = client_factory(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.reset_execution_trackers()

    code_count = 0
    runner_error_message_override: str | None = None
    run_started = monotonic()
    try:
        with client.setup_kernel():
            report["status"] = "running"
            atomic_write_json(timing_path, report)
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
                stage, stage_title = contexts[cell_index]
                original_source = cell.source
                try:
                    execute_temporary_cell(
                        client,
                        notebook,
                        CUDA_SYNCHRONIZE_SOURCE,
                    )
                except BaseException as exc:
                    try:
                        client.set_widgets_metadata()
                    except Exception:
                        pass
                    cell.source = original_source
                    notebook.cells[cell_index] = cell
                    nbformat.write(notebook, output_path)
                    raise CellSynchronizationError(
                        "pre-cell CUDA synchronization failed before learner "
                        f"cell {cell_index} ({cell.get('id', '')}) started: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                record = {
                    "cell_index": cell_index,
                    "code_index": code_count,
                    "cell_id": str(cell.get("id", "")),
                    "stage": stage,
                    "stage_title": stage_title,
                    "first_line": first_line[:100],
                    "status": "running",
                    "started_utc": utc_now(),
                    "elapsed_s": None,
                    "error_type": None,
                    "error_message": None,
                }
                report["cell_timings"].append(record)
                refresh_timing_summary(report)
                atomic_write_json(timing_path, report)
                print(
                    f"[cell {cell_index:02d} | code {code_count:02d}] start: "
                    f"{first_line[:100]}",
                    flush=True,
                )
                cell_started = monotonic()
                synchronization_failure: str | None = None
                try:
                    try:
                        try:
                            client.execute_cell(
                                cell,
                                cell_index,
                                execution_count=code_count,
                            )
                        except BaseException:
                            try:
                                execute_temporary_cell(
                                    client,
                                    notebook,
                                    CUDA_SYNCHRONIZE_SOURCE,
                                )
                            except BaseException as sync_exc:
                                synchronization_failure = (
                                    "post-cell CUDA synchronization also "
                                    f"failed: {type(sync_exc).__name__}: "
                                    f"{sync_exc}"
                                )
                            raise
                        try:
                            execute_temporary_cell(
                                client,
                                notebook,
                                CUDA_SYNCHRONIZE_SOURCE,
                            )
                        except BaseException as sync_exc:
                            raise CellSynchronizationError(
                                "post-cell CUDA synchronization failed after "
                                f"learner cell {cell_index} "
                                f"({cell.get('id', '')}) completed: "
                                f"{type(sync_exc).__name__}: {sync_exc}"
                            ) from sync_exc
                    finally:
                        cell.source = original_source
                        notebook.cells[cell_index] = cell
                except BaseException as exc:
                    record["elapsed_s"] = monotonic() - cell_started
                    record["status"] = "failed"
                    record["error_type"] = type(exc).__name__
                    error_message = str(exc)
                    if synchronization_failure is not None:
                        error_message = (
                            f"{error_message}\n{synchronization_failure}"
                        )
                        runner_error_message_override = error_message
                    record["error_message"] = error_message[:2000]
                    try:
                        client.set_widgets_metadata()
                    except Exception:
                        pass
                    nbformat.write(notebook, output_path)
                    refresh_timing_summary(report)
                    atomic_write_json(timing_path, report)
                    print(
                        f"cell {cell_index} failed; partial notebook saved to "
                        f"{output_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                record["elapsed_s"] = monotonic() - cell_started
                record["status"] = "complete"
                client.set_widgets_metadata()
                nbformat.write(notebook, output_path)
                refresh_timing_summary(report)
                atomic_write_json(timing_path, report)
                print(
                    f"[cell {cell_index:02d} | code {code_count:02d}] done in "
                    f"{record['elapsed_s']:.1f}s",
                    flush=True,
                )
    except BaseException as exc:
        report["status"] = "failed"
        report["finished_utc"] = utc_now()
        report["total_wall_elapsed_s"] = monotonic() - run_started
        report["runner_error_type"] = type(exc).__name__
        report["runner_error_message"] = (
            runner_error_message_override
            if runner_error_message_override is not None
            else str(exc)
        )[:2000]
        if output_path.is_file():
            report["executed_notebook_sha256"] = sha256_file(output_path)
        refresh_timing_summary(report)
        atomic_write_json(timing_path, report)
        raise

    report["status"] = "complete"
    report["finished_utc"] = utc_now()
    report["total_wall_elapsed_s"] = monotonic() - run_started
    report["executed_notebook_sha256"] = sha256_file(output_path)
    refresh_timing_summary(report)
    atomic_write_json(timing_path, report)
    print(f"executed notebook saved to {output_path}", flush=True)
    print(f"timing report saved to {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
