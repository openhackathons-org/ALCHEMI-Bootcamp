"""Static contract checks for the training and fine-tuning notebook."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "training-finetuning.ipynb"
REPO_ROOT = NOTEBOOK_PATH.parents[2]


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def joined_source(
    notebook: nbformat.NotebookNode,
    cell_type: str | None = None,
) -> str:
    return "\n\n".join(
        cell.source
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


def test_notebook_exists_validates_and_parses_as_one_namespace() -> None:
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source) for cell in code_cells
    )
    ast.parse(transformed)

    assert 60 <= len(notebook.cells) <= 110
    assert 25 <= len(code_cells) <= 75
    lengths = [len(cell.source.splitlines()) for cell in code_cells]
    assert sum(length <= 5 for length in lengths) > len(lengths) / 2
    assert max(lengths) <= 20


def test_banner_course_map_and_opening_have_clear_hierarchy() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    banner = (REPO_ROOT / "shared" / "banner.md").read_text().strip()

    assert notebook.cells[0].source.strip() == banner
    assert "# 07 · Training and fine-tuning" in markdown
    assert "## Course map" in markdown
    assert "../../shared/curriculum-map-07.svg" in markdown
    assert "<object" in markdown
    assert "Green marks Part 07" in markdown
    assert "two separate examples" in markdown


def test_only_setup_and_presentation_plumbing_are_hidden() -> None:
    notebook = read_notebook()
    hidden = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.metadata.get("jupyter", {}).get("source_hidden") is True
        and "hide-input" in cell.metadata.get("tags", [])
    ]

    assert hidden == [2, 4]
    assert "device =" in notebook.cells[3].source


def test_required_training_and_model_apis_remain_visible() -> None:
    source = joined_source(read_notebook())
    for token in (
        "FineTuningStrategy(",
        "trainable_patterns=",
        'freeze_mode="requires_grad"',
        "OptimizerConfig(",
        "EnergyMSELoss(",
        "ForceMSELoss(",
        "ComposedLossFunction(",
        "default_training_fn",
        "ValidationConfig(",
        "BatchValidationCallback",
        "TrainingStage.BEFORE_FORWARD",
        "ReportingOrchestrator(",
        "RichReporter(",
        "RichReporter.preview(",
        "CheckpointHook(",
        ".save_checkpoint(",
        "FineTuningStrategy.load_checkpoint(",
        "BaseModelMixin",
        "ModelConfig",
        "NeighborConfig",
        "compute_neighbors(",
        ".checkpoint_spec()",
        "LennardJonesModelWrapper(",
    ):
        assert token in source, token


def test_examples_progress_from_toy_ownership_to_argon_restart() -> None:
    source = joined_source(read_notebook())

    toy = source.index("## Level 1: inspect four fine-tuning updates")
    argon = source.index("## Level 2: fit a generated argon potential")
    checkpoint = source.index("CheckpointHook(")
    resume = source.index("FineTuningStrategy.load_checkpoint(")
    transfer = source.index("LennardJonesModelWrapper(")
    assert toy < argon < checkpoint < resume < transfer

    for token in (
        "parameter_ownership",
        "toy_training_rows",
        "toy_validation_rows",
        "split_frame",
        "energy target shape",
        "force target shape",
        "epsilon",
        "sigma",
        "cutoff",
        "completed optimizer updates",
        "validation",
    ):
        assert token in source


def test_scientific_scope_and_non_claims_are_explicit() -> None:
    markdown = joined_source(read_notebook(), "markdown").lower()
    for phrase in (
        "dimensionless synthetic score",
        "does not represent a physical energy",
        "generated labels come from the stated 12–6 lennard-jones equation",
        "not electronic-structure data",
        "does not establish transferability",
        "training and validation share the same generated potential family",
        "a lower training loss is not evidence of broader scientific accuracy",
        "report the held-out result whether it improves or not",
        "the combined loss is an optimization scalar",
        "isolated, non-periodic ar4 structures",
    ):
        assert phrase in markdown, phrase


def test_aimnet_and_optional_scale_boundaries_are_honest() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    code = joined_source(notebook, "code")

    assert "AIMNet2" in markdown
    assert "from_pretrained_checkpoint" in markdown
    assert "pickle-free reconstruction specification" in markdown
    assert "MACE" in markdown and "UMA" in markdown
    assert "not runnable in this frozen environment" in markdown
    assert "DDP" in markdown and "does not launch" in markdown
    assert "from_pretrained_checkpoint(" not in code
    assert not re.search(r"from_pretrained_checkpoint\([^)]*\.pt", code, re.DOTALL)
    assert "DemoModelWrapper" not in joined_source(notebook)


def test_outputs_units_splits_and_parameter_ownership_are_visible() -> None:
    source = joined_source(read_notebook())
    for token in (
        "eV",
        "eV/Å",
        "Å",
        "torch.float64",
        "device.type",
        "sample_id",
        "train",
        "validation",
        "(4, 3)",
        "(1, 1)",
        "requires_grad",
        "optimizer_owned",
        "energy MSE (score²)",
        "0.010188",
        "0.001028 eV",
        "0.000617 eV/Å",
    ):
        assert token in source, token
    assert "force MSE (not used)" not in source


def test_visuals_have_questions_interpretation_and_exportable_alt_text() -> None:
    notebook = read_notebook()
    visual_tokens = (
        "plot_toy_history(",
        "plot_argon_split(",
        "plot_argon_training(",
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not any(
            token in cell.source for token in visual_tokens
        ):
            continue
        before = next(
            item.source
            for item in reversed(notebook.cells[:index])
            if item.cell_type == "markdown"
        )
        after = next(
            item.source
            for item in notebook.cells[index + 1 :]
            if item.cell_type == "markdown"
        )
        assert "?" in before
        assert after.strip()
        assert "helpers.render_figure(" in cell.source
        assert "alt_text=" in cell.source
        assert cell.metadata.get("alt")


def test_only_shared_highlight_and_api_callouts_are_styled() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    shared = (REPO_ROOT / "shared" / "callouts.md").read_text()
    shared_styles = re.findall(r'^<div style="([^"]+)">', shared, re.MULTILINE)
    styled_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown" and '<div style="' in cell.source
    ]

    assert len(styled_cells) == 2
    assert markdown.count("💡 Highlight") == 1
    assert markdown.count("ALCHEMI TOOLKIT API") == 1
    for style in shared_styles:
        assert markdown.count(f'<div style="{style}">') == 1


def test_source_is_clean_and_avoids_private_or_internal_language() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    source = joined_source(notebook)
    forbidden = (
        "object.__setattr__",
        "._models",
        "._groups",
        "core_action",
        "core_mode",
        "canonical",
        "pedagogical",
        "internal curriculum",
        "timing band",
        "human review stage",
        "provenance",
    )
    for token in forbidden:
        assert token.lower() not in markdown.lower()
    assert "—" not in markdown
    assert not re.search(r"(?m)^(?:Now|Next|In this section)\b", markdown)
    assert not re.search(r"\bDemoModelWrapper\b", source)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


def test_computation_display_plotting_and_interpretation_are_separate() -> None:
    notebook = read_notebook()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if ".run(" in source:
            assert "display(" not in source and "plot_" not in source
        if "pd.DataFrame(" in source:
            assert "display(" not in source
        if "display(" in source:
            assert "pd.DataFrame(" not in source
        if "plot_" in source:
            assert ".run(" not in source and "pd.DataFrame(" not in source


def test_recap_links_core_models_dynamics_and_domain() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    for target in (
        "../00-core-playbook/alchemi-core-playbook.ipynb",
        "../03-model-interfaces-composition/model-interfaces-composition.ipynb",
        "../05-base-dynamics/base-dynamics.ipynb",
        "../08-domain-decomposition/domain-decomposition.ipynb",
    ):
        assert target in markdown
    assert "in progress" in markdown
    assert "## Try it:" in markdown
    assert "## Recap" in markdown
    assert "### What you learned" in markdown
    assert "### How we will use this" in markdown


def test_local_links_exist_or_are_marked_in_progress() -> None:
    notebook = read_notebook()
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        targets = re.findall(r"\]\((\.\./[^)#]+\.ipynb)(?:#[^)]*)?\)", cell.source)
        for target in targets:
            if (NOTEBOOK_PATH.parent / target).resolve().is_file():
                continue
            assert "in progress" in cell.source, target
