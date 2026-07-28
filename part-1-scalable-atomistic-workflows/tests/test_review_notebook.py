from __future__ import annotations

import importlib.util
import shutil
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

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


def _write_packaged_notebook_sources(source_dir: Path) -> None:
    contents = {
        "COMPUTE_LAB_RUNBOOK.md": ("## 5. Build and check the recorded result set\n"),
        "../part-2-batched-adsorption-toolkit/README.md": (
            "[Part 1](../part-1-scalable-atomistic-workflows/)\n"
        ),
        "../THIRD_PARTY_NOTICES.md": (
            "[reference](part-1-scalable-atomistic-workflows/reference/README.md)\n"
            "[NCI data](part-1-scalable-atomistic-workflows/data/nci_atlas/README.md)\n"
        ),
        "reference/README.md": "# Reference data\n",
        "data/nci_atlas/README.md": "# NCI data\n",
    }
    for source_reference, output_reference in MODULE.PACKAGED_NOTEBOOK_FILES:
        source = (source_dir / source_reference).resolve()
        source.parent.mkdir(parents=True, exist_ok=True)
        if source_reference == output_reference and source.suffix == ".png":
            source.write_bytes(b"banner bytes")
        else:
            source.write_text(contents[source_reference], encoding="utf-8")


def _review_html_text(*, embed_banner: bool = False) -> str:
    output_references = MODULE.LOCAL_NOTEBOOK_OUTPUT_REFERENCES
    banner = output_references[
        "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png"
    ]
    runbook = output_references[
        "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set"
    ]
    part_2 = output_references["../part-2-batched-adsorption-toolkit/README.md"]
    notices = output_references["../THIRD_PARTY_NOTICES.md"]
    banner_source = "data:image/png;base64,YmFubmVy" if embed_banner else banner
    return (
        '<script type="application/vnd.jupyter.widget-state+json">'
        '{"model_module":"jupyter-ovito"}'
        "</script>\n"
        f'<img src="{banner_source}" alt="review banner">\n'
        f'<a href="{runbook}">runbook</a>\n'
        f'<a href="{part_2}">Part 2</a>\n'
        f'<a href="{notices}">notices</a>\n'
    )


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


def test_only_exact_known_upstream_warning_streams_are_removed() -> None:
    known_warnings = [
        (
            "/env/lib/python3.12/site-packages/physicsnemo/utils/logging/"
            "launch.py:327: SyntaxWarning: invalid escape sequence '\\.'\n"
            '  key = re.sub("[^a-zA-Z0-9\\.\\-\\s\\/\\_]+", "", key)\n'
        ),
        (
            "/env/lib/python3.12/site-packages/nvalchemi/models/aimnet2.py:342: "
            "UserWarning: Converting a tensor with requires_grad=True to a scalar "
            "may lead to unexpected behavior.\n"
            "Consider using tensor.detach() first. (Triggered internally.)\n"
            "  values = [float(v) for v in (rc_s, rc_v) if v is not None]\n"
        ),
        (
            "/env/lib/python3.12/site-packages/torch/_inductor/compile_fx.py:320: "
            "UserWarning: TensorFloat32 tensor cores for float32 matrix "
            "multiplication available but not enabled. Consider setting the "
            "matmul precision.\n"
            "  warnings.warn(\n"
        ),
        (
            "Sets are not currently considered sequences, but this may change in "
            "the future, so consider avoiding using them.\n"
        ),
    ]
    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run()",
                id="calculation",
                execution_count=1,
                outputs=[
                    *[
                        nbformat.v4.new_output(
                            "stream",
                            name="stderr",
                            text=warning,
                        )
                        for warning in known_warnings
                    ],
                    nbformat.v4.new_output(
                        "stream",
                        name="stderr",
                        text="unexpected warning stays visible\n",
                    ),
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text="calculation complete\n",
                    ),
                ],
            )
        ]
    )

    assert MODULE.remove_known_upstream_warnings(reviewed) == 4
    assert [output.text for output in reviewed.cells[0].outputs] == [
        "unexpected warning stays visible\n",
        "calculation complete\n",
    ]


def test_known_warning_is_not_removed_when_stderr_contains_other_text() -> None:
    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run()",
                id="calculation",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stderr",
                        text=(
                            "Sets are not currently considered sequences, but this "
                            "may change in the future, so consider avoiding using "
                            "them.\nreal error follows\n"
                        ),
                    )
                ],
            )
        ]
    )

    assert MODULE.remove_known_upstream_warnings(reviewed) == 0
    assert len(reviewed.cells[0].outputs) == 1


def test_local_markdown_references_are_rebased_for_release_copy(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "part-1"
    output_dir = source_dir / "outputs" / "run-42"
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=output_dir,
    )

    reviewed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                '<img src="assets/images/banner_candidates/'
                'water-ir-v2-04-trajectory-to-spectrum.png">\n'
                "[domain run](COMPUTE_LAB_RUNBOOK.md"
                "#5-build-and-check-the-recorded-result-set)\n"
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
            "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png"
        ),
        "COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set": (
            "docs/part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md"
            "#5-build-and-check-the-recorded-result-set"
        ),
        "../part-2-batched-adsorption-toolkit/README.md": (
            "docs/part-2-batched-adsorption-toolkit/README.md"
        ),
        "../THIRD_PARTY_NOTICES.md": "docs/THIRD_PARTY_NOTICES.md",
    }
    source = reviewed.cells[0].source
    assert "assets/images/banner_candidates/" in source
    assert (
        "docs/part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md#5-" in source
    )
    assert "docs/part-2-batched-adsorption-toolkit/README.md" in source
    assert "docs/THIRD_PARTY_NOTICES.md" in source
    assert "../../stage/repo" not in source
    assert "attachment:preview.png" in source
    assert "https://example.com" in source


def test_local_notebook_files_are_copied_into_release_directory(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "part-1"
    output_dir = tmp_path / "release"
    _write_packaged_notebook_sources(source_dir)

    staged = MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert staged == dict(MODULE.PACKAGED_NOTEBOOK_FILES)
    for source_reference, output_reference in MODULE.PACKAGED_NOTEBOOK_FILES:
        source = (source_dir / source_reference).resolve()
        copied = output_dir / output_reference
        replacements = MODULE.PACKAGED_DOCUMENT_LINK_REPLACEMENTS.get(
            output_reference,
            {},
        )
        if replacements:
            expected = source.read_text(encoding="utf-8")
            for original, replacement in replacements.items():
                expected = expected.replace(original, replacement)
            assert copied.read_text(encoding="utf-8") == expected
        else:
            assert copied.read_bytes() == source.read_bytes()


def test_package_review_html_support_copies_and_checksums_official_files(
    tmp_path: Path,
) -> None:
    source_script, source_license = _write_fake_ovito_nbextension(
        tmp_path / "installed"
    )
    release = tmp_path / "release"
    release.mkdir()
    source_dir = tmp_path / "part-1"
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(), encoding="utf-8")
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
    expected_packaged = {
        html.name: MODULE.sha256_file(html),
        copied_script.name: MODULE.sha256_file(copied_script),
        copied_license.name: MODULE.sha256_file(copied_license),
    }
    expected_packaged.update(
        {
            output_reference: MODULE.sha256_file(release / output_reference)
            for _, output_reference in MODULE.PACKAGED_NOTEBOOK_FILES
        }
    )
    assert packaged == expected_packaged
    entries = MODULE._read_checksum_index(checksums)
    assert entries == {
        **expected_packaged,
        existing.name: MODULE.sha256_file(existing),
    }
    MODULE.validate_review_html_bundle(html, checksums)


@pytest.mark.parametrize(
    "missing_name",
    [
        MODULE.OVITO_WIDGET_SCRIPT_NAME,
        MODULE.OVITO_WIDGET_LICENSE_NAME,
        *(output for _, output in MODULE.PACKAGED_NOTEBOOK_FILES),
    ],
)
def test_review_html_validation_catches_missing_release_support(
    tmp_path: Path,
    missing_name: str,
) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    release = tmp_path / "release"
    release.mkdir()
    source_dir = tmp_path / "part-1"
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(), encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"
    MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )
    (release / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        MODULE.validate_review_html_bundle(html, checksums)


def test_review_bundle_remains_valid_after_relocation(tmp_path: Path) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    source_dir = tmp_path / "source" / "part-1"
    release = tmp_path / "build" / "review"
    release.mkdir(parents=True)
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(), encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"
    MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )

    relocated = tmp_path / "unrelated" / "portable-review"
    relocated.parent.mkdir()
    shutil.move(release, relocated)
    shutil.rmtree(tmp_path / "source")
    shutil.rmtree(tmp_path / "installed")
    MODULE.validate_review_html_bundle(
        relocated / html.name,
        relocated / checksums.name,
    )
    for reference in MODULE.LOCAL_NOTEBOOK_OUTPUT_REFERENCES.values():
        assert (relocated / urlsplit(reference).path).is_file()
    for relative, expected in MODULE._read_checksum_index(
        relocated / checksums.name
    ).items():
        path = relocated / relative
        assert path.is_file()
        assert MODULE.sha256_file(path) == expected


def test_repository_documents_form_a_portable_review_bundle(tmp_path: Path) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    source_dir = ROOT / "part-1-scalable-atomistic-workflows"
    release = tmp_path / "review"
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(embed_banner=True), encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"

    MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )

    relocated = tmp_path / "relocated"
    shutil.move(release, relocated)
    MODULE.validate_review_html_bundle(
        relocated / html.name,
        relocated / checksums.name,
    )


def test_review_validation_checks_every_indexed_file(tmp_path: Path) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    source_dir = tmp_path / "part-1"
    release = tmp_path / "release"
    release.mkdir()
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(), encoding="utf-8")
    calculation = release / "calculation.csv"
    calculation.write_text("energy\n-1.0\n", encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"
    checksums.write_text(
        f"{MODULE.sha256_file(calculation)}  {calculation.name}\n",
        encoding="utf-8",
    )
    MODULE.package_review_html_support(
        html,
        checksums,
        nbextension_dir=source_script.parent,
    )
    calculation.unlink()

    with pytest.raises(FileNotFoundError, match=calculation.name):
        MODULE.validate_review_html_bundle(html, checksums)


def test_review_validation_checks_every_indexed_digest(tmp_path: Path) -> None:
    source_script, _ = _write_fake_ovito_nbextension(tmp_path / "installed")
    source_dir = tmp_path / "part-1"
    release = tmp_path / "release"
    release.mkdir()
    _write_packaged_notebook_sources(source_dir)
    MODULE.stage_local_notebook_files(
        source_dir=source_dir,
        output_dir=release,
    )
    html = release / "alchemi-water-ir-reviewed.html"
    html.write_text(_review_html_text(), encoding="utf-8")
    calculation = release / "calculation.csv"
    calculation.write_text("energy\n-1.0\n", encoding="utf-8")
    checksums = release / "SHA256SUMS-reviewed"
    checksums.write_text(
        f"{'0' * 64}  {calculation.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match=f"checksum does not match for {calculation.name}"
    ):
        MODULE.package_review_html_support(
            html,
            checksums,
            nbextension_dir=source_script.parent,
        )
