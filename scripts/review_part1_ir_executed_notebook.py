#!/usr/bin/env python3
"""Apply a Markdown-only source refresh to an executed Part 1 notebook."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = REPO_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))

from aux.release_links import (  # noqa: E402
    LOCAL_NOTEBOOK_REFERENCES,
    local_reference_replacements,
)


WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"
LOCAL_MARKDOWN_REFERENCES = LOCAL_NOTEBOOK_REFERENCES


def _saved_progress_html(executed: nbformat.NotebookNode) -> dict[str, str]:
    """Return final progress-card HTML keyed by the widget model ID."""

    widget_state = (
        executed.get("metadata", {})
        .get("widgets", {})
        .get(WIDGET_STATE_MIME, {})
        .get("state", {})
    )
    progress_html: dict[str, str] = {}
    for model_id, model in widget_state.items():
        state = model.get("state", {})
        value = state.get("value")
        if (
            model.get("model_name") == "HTMLModel"
            and isinstance(value, str)
            and '<section role="group"' in value
            and 'role="progressbar"' in value
        ):
            progress_html[str(model_id)] = value
    return progress_html


def flatten_saved_progress_cards(
    reviewed: nbformat.NotebookNode,
    executed: nbformat.NotebookNode,
) -> int:
    """Replace saved progress-widget views with ordinary exportable HTML."""

    progress_html = _saved_progress_html(executed)
    flattened = 0
    for cell in reviewed.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            view = data.get(WIDGET_VIEW_MIME)
            if not isinstance(view, dict):
                continue
            model_id = str(view.get("model_id", ""))
            if model_id not in progress_html:
                continue
            data.pop(WIDGET_VIEW_MIME)
            data["text/html"] = progress_html[model_id]
            data["text/plain"] = "<IPython.core.display.HTML object>"
            flattened += 1
    return flattened


def preserve_saved_widget_state(
    reviewed: nbformat.NotebookNode,
    executed: nbformat.NotebookNode,
) -> int:
    """Copy saved state for interactive outputs that remain in the review copy."""

    saved_widgets = executed.get("metadata", {}).get("widgets")
    if saved_widgets is None:
        return 0
    reviewed.metadata["widgets"] = copy.deepcopy(saved_widgets)
    state = saved_widgets.get(WIDGET_STATE_MIME, {}).get("state", {})
    return len(state)


def rebase_local_markdown_references(
    reviewed: nbformat.NotebookNode,
    *,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Keep local images and links working after the reviewed notebook is copied.

    The source notebook lives in the Part 1 directory, while release notebooks
    are saved below ``outputs/<run>``. Rebase the intentionally local
    references against the release location without touching web or attachment
    links.
    """

    replacements = local_reference_replacements(
        source_dir=source_dir,
        output_dir=output_dir,
    )
    for reference, replacement in replacements.items():
        matches = 0
        for cell in reviewed.cells:
            if cell.cell_type != "markdown":
                continue
            matches += cell.source.count(reference)
            cell.source = cell.source.replace(reference, replacement)
        if matches != 1:
            raise RuntimeError(
                f"expected one local Markdown reference {reference!r}; found {matches}"
            )
    return replacements


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

    saved_widget_models_preserved = preserve_saved_widget_state(reviewed, executed)
    progress_outputs_flattened = flatten_saved_progress_cards(reviewed, executed)
    rebased_references = rebase_local_markdown_references(
        reviewed,
        source_dir=args.source.resolve().parent,
        output_dir=args.output.resolve().parent,
    )
    reviewed.metadata["alchemi_review"] = {
        "kind": "markdown-only-source-refresh",
        "calculation_job_id": str(args.calculation_job_id),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "original_executed_sha256": sha256_file(args.executed),
        "code_cell_count": code_count,
        "code_sources_unchanged": True,
        "progress_outputs_flattened": progress_outputs_flattened,
        "saved_widget_models_preserved": saved_widget_models_preserved,
        "rebased_local_markdown_references": rebased_references,
        "reason": (
            "Refreshed learner-facing Markdown and kept its local images and links "
            "working from the release directory; numerical code and outputs are "
            "unchanged."
        ),
    }
    nbformat.validate(reviewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(reviewed, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
