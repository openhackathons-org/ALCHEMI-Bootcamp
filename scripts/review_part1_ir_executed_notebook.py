#!/usr/bin/env python3
"""Build the reviewed Part 1 notebook and package its local HTML support."""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import shutil
import sys
import sysconfig
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = REPO_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))

from aux.release_links import (
    LOCAL_NOTEBOOK_REFERENCES,
    PACKAGED_NOTEBOOK_ASSETS,
    local_reference_replacements,
)

WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"
LOCAL_MARKDOWN_REFERENCES = LOCAL_NOTEBOOK_REFERENCES
OVITO_WIDGET_MODULE = "jupyter-ovito"
OVITO_WIDGET_SCRIPT_NAME = f"{OVITO_WIDGET_MODULE}.js"
OVITO_WIDGET_LICENSE_NAME = "index.js.LICENSE.txt"
OVITO_NBEXTENSION_RELATIVE_PATH = (
    Path("share") / "jupyter" / "nbextensions" / OVITO_WIDGET_MODULE / "index.js"
)
KNOWN_UPSTREAM_WARNING_PATTERNS = tuple(
    re.compile(pattern, re.DOTALL)
    for pattern in (
        (
            r"^.*physicsnemo/utils/logging/launch[.]py:\d+: SyntaxWarning: "
            r"invalid escape sequence.*\n\s*key = re[.]sub.*$"
        ),
        (
            r"^.*nvalchemi/models/aimnet2[.]py:\d+: UserWarning: Converting a "
            r"tensor with requires_grad=True to a scalar.*\n"
            r"Consider using tensor[.]detach[(][)] first[.].*\n"
            r"\s*values =.*$"
        ),
        (
            r"^.*torch/_inductor/compile_fx[.]py:\d+: UserWarning: TensorFloat32 "
            r"tensor cores for float32 matrix multiplication available but not "
            r"enabled[.].*\n\s*warnings[.]warn[(](?:[)])?$"
        ),
        (
            r"^Sets are not currently considered sequences, but this may change "
            r"in the future, so consider avoiding using them[.]$"
        ),
    )
)


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


def remove_known_upstream_warnings(reviewed: nbformat.NotebookNode) -> int:
    """Remove exact warning-only stderr blocks from the learner review copy."""

    removed = 0
    for cell in reviewed.cells:
        if cell.cell_type != "code":
            continue
        kept_outputs = []
        for output in cell.get("outputs", []):
            text = output.get("text")
            is_known_warning = (
                output.get("output_type") == "stream"
                and output.get("name") == "stderr"
                and isinstance(text, str)
                and any(
                    pattern.fullmatch(text.strip())
                    for pattern in KNOWN_UPSTREAM_WARNING_PATTERNS
                )
            )
            if is_known_warning:
                removed += 1
            else:
                kept_outputs.append(output)
        cell.outputs = kept_outputs
    return removed


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


def stage_local_notebook_assets(
    *,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Copy notebook images into the portable learner-review directory."""

    staged: dict[str, str] = {}
    for reference in PACKAGED_NOTEBOOK_ASSETS:
        source = (source_dir.resolve() / reference).resolve()
        destination = (output_dir.resolve() / reference).resolve()
        try:
            destination.relative_to(output_dir.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"release asset escapes the output directory: {reference!r}"
            ) from error
        if not source.is_file():
            raise FileNotFoundError(f"missing notebook release asset: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_without_overwriting_different_file(source, destination)
        staged[reference] = reference
    return staged


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_ovito_nbextension_candidates() -> tuple[Path, ...]:
    """Return official OVITO widget paths in the active Python environment."""

    candidates = [
        Path(sys.prefix) / OVITO_NBEXTENSION_RELATIVE_PATH,
        Path(sysconfig.get_path("data")) / OVITO_NBEXTENSION_RELATIVE_PATH,
    ]
    spec = find_spec("ovito")
    if spec is not None and spec.submodule_search_locations is not None:
        candidates.extend(
            Path(location) / "nbextension" / "index.js"
            for location in spec.submodule_search_locations
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def validate_ovito_nbextension_source(
    script_path: Path,
    license_path: Path,
) -> None:
    """Check that the installed files are the OVITO AMD widget and its notices."""

    missing = [path.name for path in (script_path, license_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing installed OVITO widget files: {missing}")

    script = script_path.read_text(encoding="utf-8")
    required_script_markers = (
        "define(",
        "@jupyter-widgets/base",
        "OvitoViewportModel",
        "OvitoViewportView",
    )
    missing_markers = [
        marker for marker in required_script_markers if marker not in script
    ]
    if missing_markers:
        raise RuntimeError(
            "installed OVITO widget does not contain the expected AMD module "
            f"markers: {missing_markers}"
        )

    license_text = license_path.read_text(encoding="utf-8")
    if "@license" not in license_text:
        raise RuntimeError(
            "installed OVITO widget license companion has no license notice"
        )


def find_ovito_nbextension(
    nbextension_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Find and check OVITO's installed classic-notebook widget files."""

    if nbextension_dir is not None:
        candidates = (nbextension_dir.resolve() / "index.js",)
    else:
        candidates = installed_ovito_nbextension_candidates()

    for script_path in candidates:
        license_path = script_path.with_name(OVITO_WIDGET_LICENSE_NAME)
        if not script_path.is_file() or not license_path.is_file():
            continue
        validate_ovito_nbextension_source(script_path, license_path)
        return script_path, license_path

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "could not find OVITO's installed Jupyter widget and license companion; "
        f"checked: {checked}"
    )


def _copy_without_overwriting_different_file(source: Path, destination: Path) -> None:
    """Copy one release support file, preserving an identical earlier copy."""

    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"release path is not a file: {destination}")
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"refusing to replace a different release file: {destination}"
            )
        return
    shutil.copy2(source, destination)


def _read_checksum_index(path: Path) -> dict[str, str]:
    """Read a sha256sum-compatible index and reject ambiguous entries."""

    if not path.is_file():
        return {}

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line:
            continue
        fields = raw_line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed checksum line {path}:{line_number}")
        digest, relative = fields
        relative = relative.removeprefix("*").strip()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid SHA-256 at {path}:{line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe checksum path at {path}:{line_number}")
        if relative in entries:
            raise RuntimeError(f"duplicate checksum path at {path}:{line_number}")
        entries[relative] = digest
    return entries


def update_release_checksums(checksums_path: Path, paths: tuple[Path, ...]) -> None:
    """Add or refresh release-file checksums relative to the index directory."""

    base = checksums_path.resolve().parent
    entries = _read_checksum_index(checksums_path)
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"cannot checksum missing release file: {resolved}")
        try:
            relative = resolved.relative_to(base).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"release file is outside checksum directory {base}: {resolved}"
            ) from error
        entries[relative] = sha256_file(resolved)

    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text(
        "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())
        ),
        encoding="utf-8",
    )


def validate_review_html_bundle(html_path: Path, checksums_path: Path) -> None:
    """Check the reviewed HTML and all local files in its release bundle."""

    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"missing reviewed HTML: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    if OVITO_WIDGET_MODULE not in html:
        raise RuntimeError(
            f"reviewed HTML has no saved {OVITO_WIDGET_MODULE} widget state"
        )

    script_path = html_path.with_name(OVITO_WIDGET_SCRIPT_NAME)
    license_path = html_path.with_name(OVITO_WIDGET_LICENSE_NAME)
    asset_paths = tuple(
        html_path.parent / reference for reference in PACKAGED_NOTEBOOK_ASSETS
    )
    missing = [
        path.relative_to(html_path.parent).as_posix()
        for path in (script_path, license_path, *asset_paths)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"reviewed HTML bundle is missing OVITO support files: {missing}"
        )
    validate_ovito_nbextension_source(script_path, license_path)

    entries = _read_checksum_index(checksums_path)
    base = checksums_path.resolve().parent
    for path in (html_path, script_path, license_path, *asset_paths):
        relative = path.resolve().relative_to(base).as_posix()
        expected = entries.get(relative)
        if expected is None:
            raise RuntimeError(f"checksum index is missing {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"checksum does not match for {relative}")


def package_review_html_support(
    html_path: Path,
    checksums_path: Path,
    *,
    nbextension_dir: Path | None = None,
) -> dict[str, str]:
    """Copy OVITO's official AMD widget and checksum the portable HTML bundle."""

    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"missing reviewed HTML: {html_path}")
    if OVITO_WIDGET_MODULE not in html_path.read_text(encoding="utf-8"):
        raise RuntimeError(
            f"reviewed HTML has no saved {OVITO_WIDGET_MODULE} widget state"
        )

    source_script, source_license = find_ovito_nbextension(nbextension_dir)
    destination_script = html_path.with_name(OVITO_WIDGET_SCRIPT_NAME)
    destination_license = html_path.with_name(OVITO_WIDGET_LICENSE_NAME)
    _copy_without_overwriting_different_file(source_script, destination_script)
    _copy_without_overwriting_different_file(source_license, destination_license)
    asset_paths = tuple(
        html_path.parent / reference for reference in PACKAGED_NOTEBOOK_ASSETS
    )
    missing_assets = [
        path.relative_to(html_path.parent).as_posix()
        for path in asset_paths
        if not path.is_file()
    ]
    if missing_assets:
        raise FileNotFoundError(
            f"reviewed notebook bundle is missing local assets: {missing_assets}"
        )
    release_files = (
        html_path,
        destination_script,
        destination_license,
        *asset_paths,
    )
    update_release_checksums(checksums_path, release_files)
    validate_review_html_bundle(html_path, checksums_path)
    base = html_path.parent
    return {
        path.relative_to(base).as_posix(): sha256_file(path) for path in release_files
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--executed", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calculation-job-id")
    parser.add_argument(
        "--package-html",
        type=Path,
        help=(
            "copy OVITO's installed widget beside an exported reviewed HTML file "
            "and add all three release files to --checksums"
        ),
    )
    parser.add_argument("--checksums", type=Path)
    parser.add_argument(
        "--ovito-nbextension-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.package_html is not None:
        review_args = (
            args.source,
            args.executed,
            args.output,
            args.calculation_job_id,
        )
        if any(value is not None for value in review_args):
            parser.error(
                "--package-html cannot be combined with notebook review options"
            )
        if args.checksums is None:
            parser.error("--package-html requires --checksums")
        packaged = package_review_html_support(
            args.package_html,
            args.checksums,
            nbextension_dir=args.ovito_nbextension_dir,
        )
        for name, digest in packaged.items():
            print(f"{digest}  {name}")
        return 0

    required = {
        "--source": args.source,
        "--executed": args.executed,
        "--output": args.output,
        "--calculation-job-id": args.calculation_job_id,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        parser.error(f"notebook review requires: {', '.join(missing)}")
    if args.checksums is not None or args.ovito_nbextension_dir is not None:
        parser.error("--checksums and --ovito-nbextension-dir require --package-html")

    assert args.source is not None
    assert args.executed is not None
    assert args.output is not None
    assert args.calculation_job_id is not None
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
    upstream_warning_streams_removed = remove_known_upstream_warnings(reviewed)
    packaged_assets = stage_local_notebook_assets(
        source_dir=args.source.resolve().parent,
        output_dir=args.output.resolve().parent,
    )
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
        "upstream_warning_streams_removed": upstream_warning_streams_removed,
        "packaged_local_assets": packaged_assets,
        "rebased_local_markdown_references": rebased_references,
        "reason": (
            "Refreshed learner-facing Markdown, removed only exact one-time "
            "upstream warning streams, and kept local images and links working "
            "from the release directory. Numerical code and results are unchanged; "
            "the original executed notebook is preserved."
        ),
    }
    nbformat.validate(reviewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(reviewed, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
