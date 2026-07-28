from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "review_part1_ir_executed_notebook.py"
SPEC = importlib.util.spec_from_file_location("review_part1_notebook", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_fake_ovito_nbextension(root: Path) -> tuple[Path, Path]:
    nbextension = root / "jupyter-ovito"
    nbextension.mkdir(parents=True)
    script = nbextension / "index.js"
    script.write_text(
        'define(["@jupyter-widgets/base"], function(widgets) {'
        "class OvitoViewportModel {} "
        "class OvitoViewportView {} "
        "return {OvitoViewportModel, OvitoViewportView};"
        "});\n",
        encoding="utf-8",
    )
    license_path = nbextension / MODULE.OVITO_WIDGET_LICENSE_NAME
    license_path.write_text(
        "/** @license Test-only dependency notice. */\n",
        encoding="utf-8",
    )
    return script, license_path


def test_saved_progress_widgets_flatten_to_static_html() -> None:
    model_id = "progress-model"
    progress_html = (
        '<section role="group" aria-label="Test progress">'
        '<div role="progressbar">COMPLETE</div></section>'
    )
    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "answer = 42",
                id="calculation",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            MODULE.WIDGET_VIEW_MIME: {
                                "model_id": model_id,
                                "version_major": 2,
                                "version_minor": 0,
                            },
                            "text/plain": "HTML(value='...')",
                        },
                    )
                ],
            )
        ]
    )
    executed = nbformat.from_dict(reviewed)
    executed.metadata["widgets"] = {
        MODULE.WIDGET_STATE_MIME: {
            "state": {
                model_id: {
                    "model_name": "HTMLModel",
                    "state": {"value": progress_html},
                }
            }
        }
    }

    assert MODULE.flatten_saved_progress_cards(reviewed, executed) == 1
    data = reviewed.cells[0].outputs[0].data
    assert MODULE.WIDGET_VIEW_MIME not in data
    assert data["text/html"] == progress_html
    assert data["text/plain"] == "<IPython.core.display.HTML object>"


def test_non_progress_widget_views_are_left_intact() -> None:
    model_id = "ovito-view"
    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "display(viewer)",
                id="viewer",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            MODULE.WIDGET_VIEW_MIME: {
                                "model_id": model_id,
                                "version_major": 2,
                                "version_minor": 0,
                            },
                            "text/plain": "VBox(children=(...))",
                        },
                    )
                ],
            )
        ]
    )
    executed = nbformat.from_dict(reviewed)
    executed.metadata["widgets"] = {
        MODULE.WIDGET_STATE_MIME: {
            "state": {
                model_id: {
                    "model_name": "VBoxModel",
                    "state": {"children": []},
                }
            }
        }
    }

    assert MODULE.flatten_saved_progress_cards(reviewed, executed) == 0
    data = reviewed.cells[0].outputs[0].data
    assert data[MODULE.WIDGET_VIEW_MIME]["model_id"] == model_id
    assert data["text/plain"] == "VBox(children=(...))"


def test_saved_widget_state_is_preserved_for_review_copy() -> None:
    reviewed = nbformat.v4.new_notebook()
    executed = nbformat.v4.new_notebook()
    executed.metadata["widgets"] = {
        MODULE.WIDGET_STATE_MIME: {
            "state": {
                "ovito-view": {
                    "model_name": "VBoxModel",
                    "state": {"children": ["IPY_MODEL_viewport"]},
                },
                "viewport": {
                    "model_name": "OvitoViewportModel",
                    "state": {"_model_module": "jupyter-ovito"},
                },
            }
        }
    }

    assert MODULE.preserve_saved_widget_state(reviewed, executed) == 2
    assert reviewed.metadata["widgets"] == executed.metadata["widgets"]
    assert reviewed.metadata["widgets"] is not executed.metadata["widgets"]


def test_local_markdown_references_are_rebased_for_release_copy(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "part-1"
    output_dir = source_dir / "outputs" / "run-42"
    local_targets = {
        "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png": (
            source_dir
            / "assets/images/banner_candidates"
            / "water-ir-v2-04-trajectory-to-spectrum.png"
        ),
        "COMPUTE_LAB_RUNBOOK.md": (
            source_dir / "COMPUTE_LAB_RUNBOOK.md"
        ),
        "../part-2-batched-adsorption-toolkit/README.md": (
            tmp_path / "part-2-batched-adsorption-toolkit/README.md"
        ),
        "../THIRD_PARTY_NOTICES.md": (
            tmp_path / "THIRD_PARTY_NOTICES.md"
        ),
    }
    for target in local_targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                '<img src="assets/images/banner_candidates/'
                'water-ir-v2-04-trajectory-to-spectrum.png">\n'
                "[domain run](COMPUTE_LAB_RUNBOOK.md"
                "#5-build-and-check-the-recorded-result-set)\n"
                "[pipeline run](COMPUTE_LAB_RUNBOOK.md"
                "#6-check-the-separate-distributedpipeline-campaign)\n"
                "[Part 2](../part-2-batched-adsorption-toolkit/README.md)\n"
                "[notices](../THIRD_PARTY_NOTICES.md)\n"
                "![embedded](attachment:preview.png)\n"
                "[web](https://example.com)"
            )
        ]
    )

    replacements = MODULE.rebase_local_markdown_references(
        reviewed,
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert replacements == {
        "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png": (
            "../../assets/images/banner_candidates/"
            "water-ir-v2-04-trajectory-to-spectrum.png"
        ),
        "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set": (
            "../../COMPUTE_LAB_RUNBOOK.md"
            "#5-build-and-check-the-recorded-result-set"
        ),
        "COMPUTE_LAB_RUNBOOK.md#6-check-the-separate-distributedpipeline-campaign": (
            "../../COMPUTE_LAB_RUNBOOK.md"
            "#6-check-the-separate-distributedpipeline-campaign"
        ),
        "../part-2-batched-adsorption-toolkit/README.md": (
            "../../../part-2-batched-adsorption-toolkit/README.md"
        ),
        "../THIRD_PARTY_NOTICES.md": "../../../THIRD_PARTY_NOTICES.md",
    }
    source = reviewed.cells[0].source
    assert "../../assets/images/banner_candidates/" in source
    assert "../../COMPUTE_LAB_RUNBOOK.md#5-" in source
    assert "../../COMPUTE_LAB_RUNBOOK.md#6-" in source
    assert "../../../part-2-batched-adsorption-toolkit/README.md" in source
    assert "../../../THIRD_PARTY_NOTICES.md" in source
    assert "attachment:preview.png" in source
    assert "https://example.com" in source


def test_package_review_html_support_copies_and_checksums_official_files(
    tmp_path: Path,
) -> None:
    source_script, source_license = _write_fake_ovito_nbextension(
        tmp_path / "installed"
    )
    release = tmp_path / "release"
    release.mkdir()
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(
        '<script type="application/vnd.jupyter.widget-state+json">'
        '{"model_module":"jupyter-ovito"}'
        "</script>\n",
        encoding="utf-8",
    )
    existing = release / "notebook-review-validation.json"
    existing.write_text("{}\n", encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"
    checksums.write_text(
        f"{sha256(existing.read_bytes()).hexdigest()}  {existing.name}\n",
        encoding="utf-8",
    )

    packaged = MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )

    copied_script = release / MODULE.OVITO_WIDGET_SCRIPT_NAME
    copied_license = release / MODULE.OVITO_WIDGET_LICENSE_NAME
    assert copied_script.read_bytes() == source_script.read_bytes()
    assert copied_license.read_bytes() == source_license.read_bytes()
    assert packaged == {
        html.name: MODULE.sha256_file(html),
        copied_script.name: MODULE.sha256_file(copied_script),
        copied_license.name: MODULE.sha256_file(copied_license),
    }
    entries = MODULE._read_checksum_index(checksums)
    assert entries == {
        html.name: MODULE.sha256_file(html),
        copied_license.name: MODULE.sha256_file(copied_license),
        copied_script.name: MODULE.sha256_file(copied_script),
        existing.name: MODULE.sha256_file(existing),
    }
    MODULE.validate_review_html_bundle(html, checksums)


@pytest.mark.parametrize(
    "missing_name",
    [
        MODULE.OVITO_WIDGET_SCRIPT_NAME,
        MODULE.OVITO_WIDGET_LICENSE_NAME,
    ],
)
def test_review_html_validation_catches_missing_ovito_support(
    tmp_path: Path,
    missing_name: str,
) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    release = tmp_path / "release"
    release.mkdir()
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(
        '{"model_module":"jupyter-ovito"}\n',
        encoding="utf-8",
    )
    checksums = release / "SHA256SUMS-reviewed"
    MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )
    (release / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        MODULE.validate_review_html_bundle(html, checksums)
