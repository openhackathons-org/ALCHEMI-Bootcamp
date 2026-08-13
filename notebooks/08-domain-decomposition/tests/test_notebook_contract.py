"""Structural and pedagogical checks for the domain-decomposition notebook."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "domain-decomposition.ipynb"
REPO_ROOT = NOTEBOOK_PATH.parents[2]


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def joined_source(
    notebook: nbformat.NotebookNode, cell_type: str | None = None
) -> str:
    return "\n\n".join(
        cell.source
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


def test_notebook_schema_code_and_cell_ids_are_stable() -> None:
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source) for cell in code_cells
    )
    ast.parse(transformed)

    cell_ids = [cell.id for cell in notebook.cells]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    lengths = [len(cell.source.splitlines()) for cell in code_cells]
    assert sum(length <= 8 for length in lengths) > len(lengths) / 2
    assert max(lengths) <= 18


def test_opening_banner_status_course_map_and_product_recap() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    banner = (REPO_ROOT / "shared" / "banner.md").read_text().strip()

    assert notebook.cells[0].source.strip() == banner
    assert "# 08 · Domain decomposition" in markdown
    assert "TECHNICALLY VALIDATED DRAFT — HUMAN CELL REVIEW REQUIRED" in markdown
    assert "**Goal:**" in markdown and "**Core concepts:**" in markdown
    assert "<summary>Where NVIDIA ALCHEMI fits (recap)</summary>" in markdown
    assert "## Course map" in markdown
    assert "../../shared/curriculum-map-08.svg" in markdown
    assert "<object" in markdown
    assert "Green marks Part 08" in markdown


def test_required_public_path_is_visible_and_in_order() -> None:
    notebook = read_notebook()
    code = joined_source(notebook, "code")
    tokens = [
        "DistributedManager.initialize()",
        "manager = DistributedManager()",
        "manager.initialize_mesh(",
        "domain_config = DomainConfig(",
        "domain = DomainParallel(",
        "owned_batch = domain.partition(",
        "domain_result = domain.run(",
        "gathered_result = domain.gather(",
        "DistributedManager.cleanup()",
    ]
    for token in tokens:
        assert token in code
    indices = [code.index(token) for token in tokens]
    assert indices == sorted(indices)

    for token in (
        "AtomicData",
        "Batch",
        "LennardJonesModelWrapper(",
        "BaseDynamics(",
        "model.make_neighbor_hooks()",
        "model.distribution_spec()",
    ):
        assert token in code


def test_world_size_one_claim_is_unmistakable() -> None:
    source = joined_source(read_notebook())
    required_phrases = (
        "one process does not partition the system",
        "all 32 atoms remain owned by rank 0",
        "no communication occurred",
        "not a scaling result",
        "one-process fallback",
        "mesh is `None`",
    )
    for phrase in required_phrases:
        assert phrase.lower() in source.lower()


def test_halos_ownership_gather_and_model_specs_are_taught() -> None:
    source = joined_source(read_notebook())
    for token in (
        "owned atom",
        "ghost atom",
        "cutoff + skin",
        "halo exchange",
        "migration",
        "source_atom_id",
        "`gather(..., dst=0)`",
        "returns `None` on every other rank",
        "`MLIPSpec`",
        "`distribution_spec(strategy)`",
        "`HookScope.LOCAL`",
        "`HookScope.GLOBAL`",
        "`HookScope.RANK_ZERO`",
        "energy",
        "forces",
        "per-graph",
        "per-node",
    ):
        assert token.lower() in source.lower()


def test_domain_and_pipeline_parallelism_are_separated() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    assert "Domain decomposition" in markdown
    assert "Distributed pipeline parallelism" in markdown
    assert "one large periodic system" in markdown
    assert "different model stages" in markdown
    assert "They solve different bottlenecks" in markdown


def test_current_pin_evidence_gate_and_archived_scope_are_honest() -> None:
    source = joined_source(read_notebook())
    for token in (
        "8c2c307c1c0c76baee6f7a68eb75a45da83ffd18",
        "c1e23460859a784e1d78043bcd1c8af0d1095fa2",
        "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28",
        "5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da",
        "56b9d1c71c9c392a2e12ad8149f3ca0cb0ab816fd4926af42fd264e8874d9a36",
        "NOT REPORTED",
        "archived methodology",
        "not current evidence",
        "1, 2, and 4",
        "51,200",
        "rank ownership",
        "energy parity",
        "force parity",
        "helpers.validate_campaign(",
        "if campaign_report.ready:",
    ):
        assert token.lower() in source.lower()
    assert "plot_campaign(campaign_report)" in source


def test_notebook_code_cannot_launch_or_prompt() -> None:
    code = joined_source(read_notebook(), "code")
    banned = (
        "subprocess",
        "os.system",
        "Popen",
        "input(",
        "getpass(",
        "torchrun",
        "mpirun",
        "srun",
        "ssh ",
        "!",
        "%run",
        "%bash",
        "%%bash",
    )
    for token in banned:
        assert token not in code


def test_only_setup_plumbing_is_hidden_and_core_calls_stay_visible() -> None:
    notebook = read_notebook()
    hidden = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.metadata.get("jupyter", {}).get("source_hidden") is True
        and "hide-input" in cell.metadata.get("tags", [])
    ]
    assert len(hidden) == 2
    for index in hidden:
        assert not any(
            token in notebook.cells[index].source
            for token in ("DomainConfig(", "DomainParallel(", ".partition(", ".gather(")
        )


def test_visuals_are_interpretable_and_scaling_plot_is_gated() -> None:
    notebook = read_notebook()
    visual_tokens = (
        "plot_domain_ownership(",
        "plot_control_parity(",
        "plot_campaign(",
    )
    visual_cells = [
        (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(token in cell.source for token in visual_tokens)
    ]
    assert len(visual_cells) == 3
    for index, cell in visual_cells[:2]:
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
        assert 'metadata={"image/png": {"alt":' in cell.source
    assert "if campaign_report.ready:" in visual_cells[-1][1].source


def test_styled_callouts_are_restrained_and_match_shared_templates() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    shared = (REPO_ROOT / "shared" / "callouts.md").read_text()
    shared_styles = re.findall(r'^<div style="([^"]+)">', shared, flags=re.MULTILINE)
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


def test_links_cover_prerequisites_scale_out_and_advanced_route() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    for token in (
        "../00-core-playbook/alchemi-core-playbook.ipynb",
        "../03-model-interfaces-composition/model-interfaces-composition.ipynb",
        "../05-base-dynamics/base-dynamics.ipynb",
        "GPU pipeline",
        "R&D",
        "Universal Models for Atoms",
        "separate environment",
        "external/README.md",
    ):
        assert token in markdown

    targets = re.findall(r"\]\((\.\./[^)#]+\.ipynb)(?:#[^)]*)?\)", markdown)
    assert targets
    for target in targets:
        assert (NOTEBOOK_PATH.parent / target).resolve().is_file(), target


def test_try_it_recap_and_human_review_close_the_lesson() -> None:
    source = joined_source(read_notebook())
    assert "## Try it:" in source
    assert "try_skin" in source
    assert "ghost width" in source
    assert "## Recap" in source
    assert "### What you learned" in source
    assert "### What the evidence does not show" in source
    assert "### Human review required" in source
    assert "scientific copy" in source
    assert "external 1/2/4-GPU campaign" in source


def test_fresh_execution_is_complete_clean_and_accessible() -> None:
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(isinstance(cell.execution_count, int) for cell in code_cells)
    assert code_cells[0].outputs == []
    assert not [
        output
        for cell in code_cells
        for output in cell.outputs
        if output.output_type == "error"
    ]
    image_outputs = [
        output
        for cell in code_cells
        for output in cell.outputs
        if output.output_type in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    assert len(image_outputs) == 2
    assert all(
        output.get("metadata", {}).get("image/png", {}).get("alt")
        for output in image_outputs
    )
    streams = "\n".join(
        output.get("text", "")
        for cell in code_cells
        for output in cell.outputs
        if output.output_type == "stream"
    )
    assert "NOT REPORTED — no scaling or parity plot is drawn." in streams
    assert "Warp CUDA error" not in streams
    assert "Traceback" not in streams
