"""Learner-facing and protocol checks for the hooks lesson."""

from __future__ import annotations

import ast
import hashlib
import re
import struct
from pathlib import Path

import helpers
import nbformat
import pandas as pd
import pytest
import torch
from IPython.core.inputtransformer2 import TransformerManager
from nvalchemi.dynamics import DynamicsStage

NOTEBOOK = Path(__file__).resolve().parents[1] / "hooks.ipynb"
ROOT = NOTEBOOK.parents[2]
LABELS = ("Ethyne", "Phenol", "2,3-dimethylbutane")
COURSE_MAP = ROOT / "shared" / "curriculum-map-04.svg"
CALLOUTS = ROOT / "shared" / "callouts.md"
BANNER = ROOT / "shared" / "banner.md"
BANNER_IMAGE = ROOT / "shared" / "alchemi-banner-left.png"
N01_NOTEBOOK = (
    ROOT / "notebooks" / "01-atomicdata-batch" / "atomicdata-and-batch.ipynb"
)
RECAP_SUMMARY = "<summary>Where NVIDIA ALCHEMI fits (recap)</summary>"


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return notebook


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n\n".join(cell.source for cell in code_cells(notebook))


def markdown_source(notebook: nbformat.NotebookNode) -> str:
    return "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )


def normalized_product_copy(source: str) -> str:
    """Compare learner copy while ignoring its heading/disclosure wrapper."""

    content = source.strip()
    content = re.sub(
        r"\A## Where NVIDIA ALCHEMI fits\s*",
        "",
        content,
        count=1,
    )
    content = re.sub(
        rf"\A<details>\s*{re.escape(RECAP_SUMMARY)}\s*",
        "",
        content,
        count=1,
    )
    content = re.sub(r"\s*</details>\Z", "", content, count=1)
    return " ".join(content.split())


def n01_product_copy() -> str:
    notebook = nbformat.read(N01_NOTEBOOK, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and cell.source.startswith("## Where NVIDIA ALCHEMI fits")
    )
    return normalized_product_copy(source)


def test_selected_molecules_match_the_course_batch() -> None:
    molecules, frame = helpers.load_molecule_selection(LABELS)

    assert frame["label"].tolist() == list(LABELS)
    assert frame["formula"].tolist() == ["C2H2", "C6H6O", "C6H14"]
    assert frame["atoms"].tolist() == [4, 13, 20]
    assert frame["system_id"].tolist() == [0, 1, 2]
    assert len(molecules) == 3
    assert sum(map(len, molecules)) == 37
    assert frame["charge"].eq(0).all()
    assert [atoms.info["charge"] for atoms in molecules] == [0, 0, 0]


def test_helper_surface_hides_setup_and_owns_plot_presentation() -> None:
    expected = {
        "configure_presentation",
        "freeze_model",
        "load_molecule_selection",
        "model_checkpoint",
        "plot_energy_history",
    }
    missing = {name for name in expected if not callable(getattr(helpers, name, None))}
    assert not missing, f"missing helper functions: {sorted(missing)}"

    model = torch.nn.Linear(2, 1)
    assert helpers.freeze_model(model) is model
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_energy_history_plot_uses_schedule_and_nvidia_green() -> None:
    history = pd.DataFrame(
        {
            "step": [0, 2, 0, 2],
            "system_id": [0, 0, 1, 1],
            "molecule": ["A", "A", "B", "B"],
            "energy change (meV)": [0.0, -1.0, 0.0, -2.0],
        }
    )

    figure = helpers.plot_energy_history(history)

    axis = figure.axes[0]
    assert axis.get_xlabel() == "FIRE2 step"
    assert axis.get_ylabel() == "Energy change from first record [meV]"
    assert axis.lines[-1].get_label() == "Batch mean"
    assert axis.lines[-1].get_color().upper() == "#76B900"


def test_notebook_is_valid_and_complete_namespace_parses() -> None:
    notebook = read_notebook()
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source)
        for cell in code_cells(notebook)
    )

    ast.parse(transformed)
    assert notebook.cells


def test_shared_banner_is_first_and_course_map_stays_live() -> None:
    notebook = read_notebook()
    markdown = markdown_source(notebook)
    expected_banner = BANNER.read_text(encoding="utf-8").strip()

    assert notebook.cells[0].source.strip() == expected_banner
    image = BANNER_IMAGE.read_bytes()
    width, height = struct.unpack(">II", image[16:24])
    assert (width, height) == (2880, 450)
    assert hashlib.sha256(image).hexdigest() == (
        "016f3840bb97e61a3950bd70e587305fe9477831db9763c3d081db0b8a5bbf19"
    )

    assert COURSE_MAP.is_file()
    assert '<object data="../../shared/curriculum-map-04.svg"' in markdown
    assert '<img src="../../shared/curriculum-map-04.svg"' in markdown
    assert 'aria-label="ALCHEMI Toolkit curriculum.' in markdown
    assert "width:100%;max-width:100%;height:auto;" in markdown
    assert "Later lessons cover" in markdown
    assert "Green marks Part 04" in markdown
    map_index = markdown.index("curriculum-map-04.svg")
    assert "Hooks add behavior at named workflow stages." in markdown[map_index:]
    assert "```mermaid" not in markdown[:map_index]


def test_product_overview_is_folded_but_map_is_not() -> None:
    markdown = markdown_source(read_notebook())
    disclosure = re.search(r"<details>.*?</details>", markdown, flags=re.DOTALL)
    assert disclosure is not None
    folded = disclosure.group()
    map_index = markdown.index("curriculum-map-04.svg")

    assert RECAP_SUMMARY in folded
    assert normalized_product_copy(folded) == n01_product_copy()
    assert map_index > disclosure.end()
    assert "<details>" not in markdown[disclosure.end() : map_index]


def test_opening_links_prior_capabilities_and_names_one_new_capability() -> None:
    markdown = markdown_source(read_notebook())

    assert "# 04 · Hooks" in markdown
    assert "**Goal:**" in markdown
    assert "**Core concepts:**" in markdown
    assert "../00-core-playbook/alchemi-core-playbook.ipynb" in markdown
    assert "../01-atomicdata-batch/atomicdata-and-batch.ipynb" in markdown
    assert (
        "../03-model-interfaces-composition/model-interfaces-composition.ipynb"
        in markdown
    )
    assert "four-step argon host" in markdown
    assert "checked molecular batch" in markdown


def test_setup_is_compact_and_imports_stay_together() -> None:
    notebook = read_notebook()
    imports = [
        cell
        for cell in code_cells(notebook)
        if any(
            line.startswith(("import ", "from ")) for line in cell.source.splitlines()
        )
    ]

    assert len(imports) == 1
    assert len(imports[0].source.splitlines()) <= 15
    assert "import helpers" in imports[0].source
    assert "helpers.configure_presentation()" in imports[0].source
    assert imports[0].metadata["jupyter"]["source_hidden"] is True
    assert "hide-input" in imports[0].metadata["tags"]


def test_most_code_cells_are_small_and_only_hook_class_is_long() -> None:
    notebook = read_notebook()
    lengths = [len(cell.source.splitlines()) for cell in code_cells(notebook)]

    assert sum(length <= 5 for length in lengths) > len(lengths) / 2
    oversized = {
        index: len(cell.source.splitlines())
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and len(cell.source.splitlines()) > 20
        and "class EnergyHistoryHook:" not in cell.source
    }
    assert not oversized, f"unexpected long code cells: {oversized}"


def test_public_hook_path_and_lifecycle_stay_visible() -> None:
    source = code_source(read_notebook())
    required = {
        "AtomicData.from_atoms(",
        "Batch.from_data_list(",
        'batch.add_key("system_id"',
        "LennardJonesModelWrapper(",
        "model.make_neighbor_hooks()",
        "MaxForceClampHook(",
        'LoggingHook(\n    backend="custom"',
        "SnapshotHook(sink=snapshot_sink, frequency=2)",
        "lj_host.run(argon_batch)",
        "failure_host.run(failure_batch)",
        "ConvergenceHook.from_fmax(",
        "convergence_hook=evaluator",
        "hooks=[*lj_model.make_neighbor_hooks(), status_hook]",
        "AIMNet2Wrapper.from_checkpoint(",
        'model.set_config("active_outputs", {"energy", "forces"})',
        "class EnergyHistoryHook:",
        "DynamicsContext",
        "DynamicsStage.AFTER_COMPUTE",
        "isinstance(history_hook, Hook)",
        "def on_register(",
        "def __enter__(",
        "def __exit__(",
        "last_stage",
        "NaNDetectorHook(frequency=1)",
        "FIRE2(",
        "hooks=molecular_hooks",
        "molecular_batch = molecular_host.run(molecular_batch)",
    }
    missing = {token for token in required if token not in source}
    assert not missing, f"missing visible Toolkit operations: {sorted(missing)}"
    assert "_runs_on_stage" not in source
    assert "succeeded" not in source


def test_all_dynamics_stages_and_exact_schedule_are_explained() -> None:
    notebook = read_notebook()
    source = f"{markdown_source(notebook)}\n{code_source(notebook)}"

    for stage in DynamicsStage:
        assert stage.name in source
    assert "step_count % frequency == 0" in source
    assert "steps 0 and 2" in source
    assert "registration order" in source


def test_responsibilities_are_taught_separately() -> None:
    source = "\n".join(cell.source for cell in read_notebook().cells)

    quick = source.index("## Start with a four-step argon host")
    builtins = source.index("## Register safety, logging, and snapshots")
    failure = source.index("## Watch a real safety failure")
    convergence = source.index("## Separate convergence from status migration")
    molecular = source.index("## Observe real molecules")
    observer = source.index("## Write one small observer")
    assert quick < builtins < failure < convergence < molecular < observer
    assert "BEFORE_COMPUTE" in source[quick:builtins]
    assert "MaxForceClampHook" in source[builtins:failure]
    assert "NaNDetectorHook" in source[builtins:failure]
    assert "LoggingHook" in source[builtins:failure]
    assert "SnapshotHook" in source[builtins:failure]
    assert "RuntimeError" in source[failure:convergence]
    assert "source_status" in source[convergence:molecular]
    assert "AFTER_COMPUTE" in source[observer:]


def test_runtime_context_is_bounded_and_stage_is_separate() -> None:
    source = "\n".join(cell.source for cell in read_notebook().cells)
    context_start = source.index('"workflow": type(ctx.workflow).__name__')
    context_end = source.index("def __exit__", context_start)
    context = source[context_start:context_end]

    for token in (
        "ctx.batch",
        "ctx.model",
        "ctx.step_count",
        "ctx.global_rank",
        "ctx.converged_mask",
    ):
        assert token in context
    assert "last_stage = stage.name" in source
    assert "separate callback argument" in source
    assert (
        "because the molecular host was constructed with `convergence_hook=None`"
        in source
    )
    assert "host builds this context" in source


def test_run_lifecycle_copy_is_per_run_and_never_success_copy() -> None:
    markdown = markdown_source(read_notebook())

    assert "once per run" in markdown
    assert "workflow success" in markdown
    assert "does not mean the workflow succeeded" in markdown


def test_recap_previews_convergence_evaluation_without_conflation() -> None:
    markdown = markdown_source(read_notebook())
    recap = markdown[markdown.index("## Recap") :]

    assert "object passed through `convergence_hook=`" in recap
    assert "`evaluate(batch)`" in recap
    assert "after `AFTER_STEP`" in recap
    assert "registered `ConvergenceHook` in `hooks=[...]`" in recap
    assert "`source_status`/`target_status`" in recap
    assert "does not drive the host's early exit" in recap


def test_logging_and_profiling_are_referenced_without_a_catalogue() -> None:
    markdown = markdown_source(read_notebook())

    assert "LoggingHook" in markdown
    assert "ReportingOrchestrator" in markdown
    assert "RichReporter" in markdown
    assert markdown.count("StageTimingHook") <= 1
    assert "Part 06" in markdown


def test_exactly_the_two_approved_callout_families_are_used() -> None:
    notebook = read_notebook()
    markdown = markdown_source(notebook)
    shared = CALLOUTS.read_text(encoding="utf-8")
    styles = re.findall(r'^<div style="([^"]+)">', shared, flags=re.MULTILINE)
    styled_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown" and '<div style="' in cell.source
    ]

    assert len(styles) == 2
    assert len(styled_cells) == 2
    assert markdown.count("💡 Highlight") == 1
    assert markdown.count("ALCHEMI TOOLKIT API") == 1
    for style in styles:
        assert markdown.count(f'<div style="{style}">') == 1


def test_computation_shaping_display_plot_and_interpretation_are_separate() -> None:
    notebook = read_notebook()
    sources = [cell.source for cell in notebook.cells]

    run_index = next(
        i for i, source in enumerate(sources) if "molecular_host.run(molecular_batch)" in source
    )
    shape_index = next(
        i for i, source in enumerate(sources) if "history = pd.DataFrame" in source
    )
    display_index = next(
        i
        for i, source in enumerate(sources)
        if "history_summary" in source and source.strip().endswith("history_summary")
    )
    plot_index = next(
        i for i, source in enumerate(sources) if "plot_energy_history(" in source
    )
    assert run_index < shape_index < display_index < plot_index
    assert "plt." not in code_source(notebook)

    preceding = next(
        cell.source
        for cell in reversed(notebook.cells[:plot_index])
        if cell.cell_type == "markdown"
    )
    following = next(
        cell.source
        for cell in notebook.cells[plot_index + 1 :]
        if cell.cell_type == "markdown"
    )
    assert "?" in preceding
    assert "The scheduled records" in following


def test_try_it_is_bounded_and_recap_hands_off_to_part_five() -> None:
    markdown = markdown_source(read_notebook())
    source = code_source(read_notebook())

    assert "## Try it: change the observation schedule" in markdown
    assert "try_frequency = 3" in source
    assert "expected_try_steps = [0, 3]" in source
    assert "Try it passed" in source
    assert "## Recap" in markdown
    assert "### What you learned" in markdown
    assert "### How we will use this" in markdown
    assert "../05-base-dynamics/base-dynamics.ipynb" in markdown
    assert "does not establish geometry convergence" in markdown


def test_quick_argon_host_precedes_checked_molecular_host() -> None:
    source = "\n".join(cell.source for cell in read_notebook().cells)

    argon = source.index("## Start with a four-step argon host")
    molecular = source.index("## Observe real molecules")
    assert argon < molecular
    assert "LennardJonesModelWrapper" in source[argon:molecular]
    assert "AIMNet2Wrapper" not in source[argon:molecular]
    assert "Ethyne" in source[molecular:]


def test_learner_path_uses_registry_dispatch_only() -> None:
    source = code_source(read_notebook())

    assert "SimpleNamespace" not in source
    assert "DynamicsContext(" not in source
    assert "_call_hooks" not in source
    assert "_runs_on_stage" not in source
    assert not re.search(r"\b(?:history_hook|nan_guard)\s*\(", source)
    assert not re.search(r"\bwith\s+\w*hook\b", source)


def test_one_custom_observer_and_one_lifecycle_diagram() -> None:
    notebook = read_notebook()
    source = code_source(notebook)
    markdown = markdown_source(notebook)

    classes = re.findall(r"^class\s+(\w+Hook)\s*:", source, flags=re.MULTILINE)
    assert classes == ["EnergyHistoryHook"]
    assert markdown.count("```mermaid") == 1
    assert "BEFORE_STEP" in markdown
    assert "ON_CONVERGE" in markdown


def test_official_follow_on_examples_are_linked() -> None:
    markdown = markdown_source(read_notebook())

    required = {
        "examples/advanced/02_custom_hook.html",
        "examples/intermediate/05_safety_and_monitoring.html",
        "examples/intermediate/07_rich_training_reporting.html",
        "examples/distributed/02_distributed_monitoring.html",
    }
    missing = {target for target in required if target not in markdown}
    assert not missing, f"missing official follow-on links: {sorted(missing)}"


def test_repository_and_presentation_plumbing_stays_hidden() -> None:
    source = code_source(read_notebook())
    hidden = {
        "repo_root",
        "Path.cwd",
        "sys.path",
        "sha256",
        "MODEL_SHA256",
        "EXPECTED_",
        "runtime-pins.toml",
        "plt.",
    }
    present = {token for token in hidden if token in source}
    assert not present, f"internal plumbing leaked into learner code: {sorted(present)}"


def test_source_outputs_are_clean_and_plot_has_alt_text() -> None:
    notebook = read_notebook()
    assert all(not cell.get("outputs") for cell in code_cells(notebook))
    assert all("execution" not in cell.metadata for cell in notebook.cells)

    plot_cell = next(
        cell
        for cell in code_cells(notebook)
        if "helpers.plot_energy_history(history)" in cell.source
    )
    assert plot_cell.metadata["alt"].startswith("Per-molecule and batch-mean")
