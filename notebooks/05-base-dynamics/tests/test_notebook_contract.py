from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "base-dynamics.ipynb"
REPO_ROOT = NOTEBOOK_PATH.parents[2]
N01_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "01-atomicdata-batch" / "atomicdata-and-batch.ipynb"
)
RECAP_SUMMARY = "<summary>Where NVIDIA ALCHEMI fits (recap)</summary>"


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def joined_source(
    notebook: nbformat.NotebookNode, cell_type: str | None = None
) -> str:
    return "\n\n".join(
        cell.source
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
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
    notebook = nbformat.read(N01_NOTEBOOK_PATH, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        and cell.source.startswith("## Where NVIDIA ALCHEMI fits")
    )
    return normalized_product_copy(source)


def test_notebook_parses_and_uses_small_progressive_cells() -> None:
    notebook = read_notebook()
    cells = code_cells(notebook)
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source) for cell in cells
    )
    ast.parse(transformed)

    lengths = [len(cell.source.splitlines()) for cell in cells]
    assert sum(length <= 5 for length in lengths) > len(lengths) / 2
    assert max(lengths) <= 20


def test_banner_and_live_part05_course_map_have_visible_hierarchy() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    banner = (REPO_ROOT / "shared" / "banner.md").read_text().strip()

    assert notebook.cells[0].source.strip() == banner
    assert "## Course map" in markdown
    assert "../../shared/curriculum-map-05.svg" in markdown
    assert "<object" in markdown
    assert "Green marks Part 05" in markdown
    assert "The map places `BaseDynamics`" in markdown

    map_heading = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and "## Course map" in cell.source
    )
    map_asset = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and "../../shared/curriculum-map-05.svg" in cell.source
    )
    assert map_heading < map_asset


def test_product_overview_is_folded_but_map_is_not() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    details = re.findall(r"<details>.*?</details>", markdown, flags=re.DOTALL)

    assert len(details) == 1
    assert RECAP_SUMMARY in details[0]
    assert normalized_product_copy(details[0]) == n01_product_copy()
    assert "curriculum-map-05.svg" not in details[0]


def test_only_setup_plumbing_cells_are_hidden() -> None:
    notebook = read_notebook()
    hidden = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.metadata.get("jupyter", {}).get("source_hidden") is True
        and "hide-input" in cell.metadata.get("tags", [])
    ]

    assert hidden == [2, 4]
    assert "FMAX_TARGET_EV_A" in notebook.cells[3].source
    assert not notebook.cells[3].metadata.get("jupyter", {}).get("source_hidden")


def test_opening_reuses_prior_parts_and_executes_contracted_dynamics() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    code = joined_source(notebook, "code")

    assert "[Part 01" in markdown and "[Part 03" in markdown and "[Part 04" in markdown
    assert "one new capability" in markdown.lower()
    for token in (
        "AtomicData.from_atoms(",
        ".use_default_velocities()",
        '.add_node_property("forces"',
        '.add_system_property("energy"',
        '.add_system_property("status"',
        "Batch.from_data_list(",
        "helpers.configure_presentation(LABELS)",
        "LennardJonesModelWrapper(",
        "initialize_velocities(",
        "NVE(",
        "NVTLangevin(",
        "nvt_stage + nve_stage",
        "AIMNet2Wrapper.from_checkpoint(",
        '.set_config("active_outputs"',
        "model.make_neighbor_hooks()",
        "ConvergenceHook.from_fmax(",
        "FIRE2(",
        "hooks=[*neighbor_hooks, status_hook, monitor]",
        "convergence_hook=convergence_check",
        "fire2.run(",
    ):
        assert token in code

    assert "BatchedSteepestDescent" not in code


def test_fast_argon_mechanics_precede_supported_molecular_interpretation() -> None:
    source = joined_source(read_notebook())

    argon = source.index("## Fast mechanics: Lennard-Jones argon")
    nve = source.index("nve_result = nve.run(")
    molecule = source.index("## Molecular interpretation: supported AIMNet2 relaxation")
    fire2 = source.index("relaxed_batch = fire2.run(")

    assert argon < nve < molecule < fire2
    for token in (
        "epsilon = 0.0104 eV",
        "sigma = 3.40 Å",
        "cutoff = 8.5 Å",
        "27 periodic argon atoms",
        "simple-cubic",
        "five-update",
        "200-update",
        "not equilibration",
        "not a production trajectory",
    ):
        assert token in source


def test_md_ensembles_and_part06_boundary_are_taught_without_overclaiming() -> None:
    source = joined_source(read_notebook())

    for token in (
        "Velocity Verlet",
        "BAOAB",
        "nonperiodic simple-cubic cluster",
        "friction",
        "random_seed",
        "FusedStage",
        "one active `Batch`",
        "one shared model evaluation",
        "status 0",
        "status 1",
        "same model",
        "MTK barostat",
        "Nosé-Hoover chains",
        "requires `stress`",
        "melting-point",
        "Part 06",
        "inflight",
        "profiling",
    ):
        assert token in source

    assert "melting point =" not in source.lower()


def test_lifecycle_and_field_contract_are_taught_exactly() -> None:
    notebook = read_notebook()
    source = joined_source(notebook)

    for token in (
        "BaseDynamics.step",
        "pre_update",
        "compute",
        "post_update",
        "FIRE2.__needs_keys__",
        "FIRE2.__provides_keys__",
        '{"forces"}',
        '{"positions", "velocities"}',
        "model output",
        "existing writable storage",
        "one_step.step(",
        "override `pre_update` and `post_update`",
        "leave `compute`, `step`, and `run`",
        "in-place",
    ):
        assert token in source

    assert source.index("fire2.run(") < source.index(
        "## Selected stages inside one `BaseDynamics.step`"
    )


def test_dynamics_units_and_drift_metric_match_public_conventions() -> None:
    source = joined_source(read_notebook())

    assert source.count("sqrt(eV/amu)") >= 3
    assert "maximum |ΔE| per atom per update (eV/atom/update)" in source
    assert "DRIFT_LIMIT_EV_ATOM_UPDATE" in source


def test_real_batch_hooks_and_workflow_properties_are_inspected() -> None:
    source = joined_source(read_notebook())

    for token in (
        "starting_batch.keys.items()",
        "batch_field_table",
        "registered_hook_table",
        "detector_table",
        "status_hook",
        "convergence_check",
        "fire2.n_steps",
        "fire2.maxstep",
        "fire2.delaystep",
        "fire2.exit_status",
        "fire2.hooks",
        '"status per system": relaxed_batch.status',
        "torch.equal(relaxed_batch.positions, post_run_positions)",
        'coherent_outputs["energy"].shape',
        'coherent_outputs["forces"].shape',
    ):
        assert token in source


def test_registered_hooks_and_host_detector_have_distinct_ownership() -> None:
    notebook = read_notebook()
    code = joined_source(notebook, "code")
    markdown = joined_source(notebook, "markdown")

    assert (
        "registered_hook_objects = [*neighbor_hooks, status_hook, monitor]" in code
    )
    assert "registered_hook_objects = [*neighbor_hooks, status_hook, convergence_check" not in code
    assert '"ownership": "hooks= registry"' in code
    assert '"ownership": "convergence_hook= host detector"' in code
    assert '"fields schedule host evaluation": False' in code
    assert "convergence_check` is not registered in `hooks=`" in markdown
    assert "do not schedule this host convergence evaluation" in markdown


def test_result_numbering_is_one_based_completed_updates() -> None:
    source = joined_source(read_notebook())

    for label in (
        "completed update (1-based)",
        "first converged completed update (1-based)",
        "completed updates (1-based count)",
    ):
        assert label in source

    assert "zero-based callback `ctx.step_count`" in source
    assert '"FIRE2 steps completed"' not in source
    assert '"step count"' not in source
    assert "first converged step" not in source
    assert "FIRE2 step" not in source
    assert "step limit" not in source.lower()


def test_model_identity_scope_and_wrapper_configuration_are_visible() -> None:
    source = joined_source(read_notebook())

    for token in (
        "aimnet2-wb97m-d3_0",
        "AIMNetCentral / MIT model weights",
        "finite neutral H/C/N/O molecules",
        "torch.float32",
        "parameter.dtype",
        "parameter.device",
        "config.active_outputs",
        "neighbor.cutoff",
        "neighbor.format.value",
        "full list",
        "wrapper Coulomb / dispersion",
        "disabled / disabled",
    ):
        assert token in source


def test_final_outputs_are_recomputed_before_all_analysis() -> None:
    source = joined_source(read_notebook())

    run = source.index("relaxed_batch = fire2.run(")
    neighbors = source.index("compute_neighbors(\n    relaxed_batch")
    recompute = source.index("coherent_outputs = fire2.compute(relaxed_batch)")
    history = source.index("final_batch=relaxed_batch")
    summary = source.index("summarize_relaxation(")
    plot = source.index("plot_force_history(")
    selection = source.index("relaxed_batch.index_select(")
    exercise = source.index("try_hook.evaluate(relaxed_batch)")

    assert run < neighbors < recompute < history < summary < plot < selection < exercise
    assert "positions preserved" in source and "status preserved" in source
    assert "truncate_history_at_convergence(" in source


def test_convergence_selection_recovery_and_before_after_stay_visible() -> None:
    notebook = read_notebook()
    source = joined_source(notebook)

    for token in (
        "source_status=0",
        "target_status=1",
        "per_system_results",
        ".index_select(",
        ".get_data(",
        ".to_data_list()",
        "initial_positions",
        "final_positions",
        "plot_structure_change(",
    ):
        assert token in source

    assert "status 1" in source.lower()
    assert "update limit" in source.lower()
    assert "first converged status" in source


def test_only_shared_highlight_and_api_callouts_are_styled() -> None:
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


def test_computation_shaping_display_plotting_and_interpretation_are_separate() -> None:
    notebook = read_notebook()
    cells = code_cells(notebook)
    for cell in cells:
        source = cell.source
        if "fire2.run(" in source:
            assert "display(" not in source and "plot_" not in source
        if "nve.run(" in source or "nvt.run(" in source:
            assert "display(" not in source and "plot_" not in source
        if "summarize_relaxation(" in source:
            assert "display(" not in source
        if "display(per_system_results" in source:
            assert "summarize_relaxation(" not in source
        if "plot_force_history(" in source or "plot_structure_change(" in source:
            assert "fire2.run(" not in source and "pd.DataFrame(" not in source


def test_visuals_have_questions_takeaways_and_exportable_alt_text() -> None:
    notebook = read_notebook()
    visual_tokens = (
        "plot_force_history(",
        "plot_structure_change(",
        "plot_energy_conservation(",
        "plot_temperature_history(",
        "plot_argon_trajectory(",
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
        assert "metadata={\"image/png\": {\"alt\":" in cell.source
        assert "plt.close(" in cell.source
        assert cell.metadata.get("alt")


def test_try_it_and_recap_close_the_lesson() -> None:
    notebook = read_notebook()
    source = joined_source(notebook)
    recap = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown" and "## Recap" in cell.source
    )

    assert "## Try it:" in source
    assert "try_fmax" in source
    assert "ConvergenceHook.from_fmax(try_fmax)" in source
    assert "selected systems satisfy the chosen target" in source
    assert "## Recap" in source
    assert "### What you learned" in source
    assert "### How we will use this" in source
    assert "Course foundation: [Part 01" in recap
    assert "Prerequisite review: [Part 04" in recap
    assert "Planned next lesson: **Part 06" in recap
    assert not re.search(r"^Next:", recap, flags=re.MULTILINE)


def test_all_local_notebook_links_exist() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    targets = re.findall(r"\]\((\.\./[^)#]+\.ipynb)(?:#[^)]*)?\)", markdown)

    assert targets
    for target in targets:
        assert (NOTEBOOK_PATH.parent / target).resolve().is_file(), target


def test_notebook_hides_repository_and_checkpoint_plumbing() -> None:
    code = joined_source(read_notebook(), "code")
    for hidden in (
        "MODEL_SHA256",
        "EXPECTED_",
        "runtime-pins.toml",
        "sha256",
        "repo_root",
        "sys.path",
    ):
        assert hidden not in code
