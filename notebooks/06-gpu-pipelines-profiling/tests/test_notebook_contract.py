"""Static contract checks for the GPU pipelines and profiling deep dive."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "gpu-pipelines-profiling.ipynb"
REPO_ROOT = NOTEBOOK_PATH.parents[2]
RECAP_SUMMARY = "<summary>Where NVIDIA ALCHEMI fits (recap)</summary>"


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


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_parses_validates_and_has_stable_unique_cell_ids() -> None:
    notebook = read_notebook()
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source) for cell in code_cells(notebook)
    )

    ast.parse(transformed)
    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    assert all(ids)
    assert notebook.nbformat == 4 and notebook.nbformat_minor >= 5
    assert notebook.metadata.kernelspec.name == "python3"


def test_code_cells_are_small_and_progressive() -> None:
    cells = code_cells(read_notebook())
    lengths = [len(cell.source.splitlines()) for cell in cells]
    visible_lengths = [
        len(cell.source.splitlines())
        for cell in cells
        if "hide-input" not in cell.metadata.get("tags", [])
    ]
    hidden_lengths = [
        len(cell.source.splitlines())
        for cell in cells
        if "hide-input" in cell.metadata.get("tags", [])
    ]

    assert sum(length <= 5 for length in lengths) > len(lengths) / 2
    assert max(visible_lengths) <= 20
    assert max(hidden_lengths) <= 24


def test_banner_course_map_and_folded_product_recap_match_shared_assets() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    banner = (REPO_ROOT / "shared" / "banner.md").read_text().strip()

    assert notebook.cells[0].source.strip() == banner
    assert "## Course map" in markdown
    assert "../../shared/curriculum-map-06.svg" in markdown
    assert "<object" in markdown
    assert "Green marks Part 06" in markdown
    details = re.findall(r"<details>.*?</details>", markdown, flags=re.DOTALL)
    assert len(details) == 1
    assert RECAP_SUMMARY in details[0]
    assert "curriculum-map-06.svg" not in details[0]


def test_only_two_setup_plumbing_cells_are_hidden() -> None:
    notebook = read_notebook()
    hidden = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.metadata.get("jupyter", {}).get("source_hidden") is True
        and "hide-input" in cell.metadata.get("tags", [])
    ]

    assert hidden == [2, 4]
    assert "import helpers" in notebook.cells[2].source
    assert "configure_presentation" in notebook.cells[4].source


def test_opening_names_outcomes_prerequisites_and_measurement_boundary() -> None:
    source = joined_source(read_notebook())

    for token in (
        "# GPU pipelines and profiling",
        "Part 06",
        "[Core playbook]",
        "[Part 01",
        "[Part 04",
        "[Part 05",
        "synthetic",
        "molecular",
        "warm-up",
        "synchronization",
        "timing scope",
        "memory",
        "reproducibility",
        "no hardware-independent performance claim",
    ):
        assert token in source


def test_ownership_diagram_separates_three_parallelism_routes() -> None:
    markdown = joined_source(read_notebook(), "markdown")

    for token in (
        "```mermaid",
        "FusedStage",
        "one GPU",
        "DistributedPipeline",
        "stage pipeline",
        "DomainParallel",
        "one large system",
        "halo",
        "Part 08",
    ):
        assert token in markdown
    assert "domain decomposition is not a faster `FusedStage`" in markdown


def test_synthetic_device_transfer_and_compile_path_precede_molecules() -> None:
    source = joined_source(read_notebook())
    host = source.index("host_batch = Batch.from_data_list(")
    transfer = source.index("device_batch = host_batch.to(device)")
    compile_call = source.index("torch.compile(")
    molecular = source.index("## Molecular inflight pipeline")

    assert host < transfer < compile_call < molecular
    for token in (
        "fixed-shape",
        "compiled_stage = torch.compile(",
        "torch.testing.assert_close(",
        "helpers.time_callable(",
        "warmup=0",
        "warmup=WARMUP_CALLS",
        "torch.compiler.reset()",
        "device_memory_snapshot(",
        "plot_measurement_panels(",
    ):
        assert token in source


def test_device_is_concrete_and_synthetic_randomness_is_seeded() -> None:
    code = joined_source(read_notebook(), "code")

    assert 'torch.device("cuda", torch.cuda.current_device())' in code
    assert "SYNTHETIC_SEED = 19" in code
    assert "torch.manual_seed(SYNTHETIC_SEED)" in code
    assert "seed=SYNTHETIC_SEED" in code
    assert "assert device_batch.positions.device == device" in code


def test_inflight_pipeline_keeps_public_construction_and_run_visible() -> None:
    code = joined_source(read_notebook(), "code")

    for token in (
        "SizeAwareSampler(",
        "FusedStage(",
        "HostMemory(",
        "DemoDynamics(",
        "FIRE2(",
        "NVE(",
        "StageTimingHook(",
        "register_fused_hook(",
        "model.make_neighbor_hooks()",
        "pipeline.run(batch=None",
        "sink.read()",
        "system_id",
        "source_index",
    ):
        assert token in code
    assert code.index("DemoDynamics(") < code.index("FIRE2(")


def test_downstream_fixed_step_stages_run_before_graduation() -> None:
    source = joined_source(read_notebook())
    code = joined_source(read_notebook(), "code")

    for token in (
        "stage_1 = DemoDynamics(\n    model=demo_model, n_steps=2",
        "warmup_nve = NVE(\n    model=model, dt=0.1, n_steps=2",
        "nve_stage = NVE(\n    model=model, dt=0.1, n_steps=2",
        'assert pipeline_occupancy["status 1"].max() > 0',
        'assert molecular_occupancy["status 1"].max() > 0',
        "downstream fixed-step counter",
    ):
        assert token in source
    assert code.count("n_steps=2") >= 3


def test_transfer_and_pipeline_ownership_are_taught_explicitly() -> None:
    source = joined_source(read_notebook())

    for token in (
        "The learner owns this transfer",
        "`MoleculeDataset` owns each admission transfer",
        "`SizeAwareSampler` owns admission",
        "`FusedStage` owns stage routing",
        "`HostMemory` owns collection",
        "one active `Batch`",
        "one shared model evaluation",
        "setup is outside",
        "inside `pipeline.run(...)`",
        "initial batch",
        "prime model evaluation",
        "refill",
        "sink writes",
    ):
        assert token in source


def test_molecular_scope_model_identity_and_checks_are_honest() -> None:
    source = joined_source(read_notebook())

    for token in (
        "NCI Atlas",
        "CC BY 4.0",
        "aimnet2-wb97m-d3_0",
        "AIMNetCentral / MIT model weights",
        "finite neutral H/C/N/O molecules",
        "torch.float32",
        "energy (eV)",
        "maximum force (eV/Å)",
        "routing and identity check",
        "not a relaxation result",
        "not an equilibrated trajectory",
        "not an accuracy validation",
        "not a throughput comparison",
    ):
        assert token in source


def test_timing_memory_and_profiler_have_distinct_scopes() -> None:
    source = joined_source(read_notebook())
    timed_run = source.index("pipeline_result = pipeline.run(batch=None")
    profile_heading = source.index("## Profile a separate run")
    profiler = source.index("TorchProfilerHook(")

    assert timed_run < profile_heading < profiler
    for token in (
        "helpers.synchronize(device)",
        "perf_counter()",
        "reset_peak_memory_stats(",
        "StageTimingHook",
        "CUDA events",
        "synchronized wall-clock",
        "Profiler overhead",
        "per-rank",
        "rank_subdirs=True",
        "TemporaryDirectory",
        "profile_artifacts(",
        "instrumented complete run (ms)",
        "instrumented complete pipeline.run (ms)",
        "observer hooks",
        "with_stack=False",
        "stack traces are disabled",
    ):
        assert token in source


def test_distributed_plan_is_public_but_never_launched_from_notebook() -> None:
    code = joined_source(read_notebook(), "code")
    markdown = joined_source(read_notebook(), "markdown")

    for token in (
        "BufferConfig(",
        "rank_0_inner = DemoDynamics(",
        "rank_0_stage = FusedStage(",
        "sub_stages=[(0, rank_0_inner)]",
        "DistributedPipeline(",
        "stages={0:",
        'comm_mode="async_recv"',
        "synchronized=False",
    ):
        assert token in code
    assert "torchrun" not in code
    assert "torchrun --nproc_per_node=2" in markdown
    assert "one process per stage rank" in markdown
    assert 'os.environ["LOCAL_RANK"]' in markdown
    assert "torch.cuda.set_device(local_rank)" in markdown
    assert "rank-local CUDA device" in markdown
    assert "construction-only" in markdown
    assert "normal mixed-dtype `AtomicData` execution is blocked" in markdown
    assert "send and receive buffers call `Batch.put`" in markdown
    assert "raise RuntimeError(" in markdown
    init_process_group = markdown.index('dist.init_process_group(backend="nccl")')
    get_rank = markdown.index("dist.get_rank()")
    assert init_process_group < get_rank


def test_gpu_buffer_limitation_is_advanced_explicit_and_test_backed() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    code = joined_source(notebook, "code")
    behavior_tests = (
        NOTEBOOK_PATH.parent / "tests" / "test_pipeline_behavior.py"
    ).read_text()

    assert "## Advanced limitation: mixed-dtype `GPUBuffer`" in markdown
    assert "float32 fields" in markdown
    assert "integer fields may be skipped" in markdown
    assert (
        "def test_gpu_buffer_probe_exposes_integer_field_limitation_at_this_pin"
        in behavior_tests
    )
    assert "test_gpu_buffer_probe_exposes_integer_field_limitation_at_this_pin" not in markdown
    assert "helpers.probe_gpu_buffer_mixed_dtype(device)" in code
    assert "Batch.put(" not in code
    assert "Batch.defrag(" not in code
    assert "GPUBuffer(" not in code
    assert "safe general mixed-dtype route" in markdown


def test_runtime_identity_and_results_never_embed_gpu_numbers() -> None:
    notebook = read_notebook()
    source = joined_source(notebook)

    for token in (
        "runtime_identity(device)",
        "Toolkit commit",
        "Toolkit-Ops commit",
        "GPU name",
        "CPU name",
        "input identity",
        "measurement status",
        "GPU measurement unavailable",
        "Rerun on the target GPU",
    ):
        assert token in source
    assert "speedup" not in source.lower()
    assert "cached result" not in source.lower()


def test_native_diagnostics_filter_only_known_cpu_fallback_noise() -> None:
    source = joined_source(read_notebook())
    helper_source = (
        NOTEBOOK_PATH.parent / "helpers" / "lesson.py"
    ).read_text(encoding="utf-8")

    assert "helpers.filter_known_native_stderr(device)" in source
    assert "native_stderr_to_devnull" not in source
    assert "os.devnull" not in source
    assert "unexpected_lines" in helper_source
    assert "sys.stderr.writelines(unexpected_lines)" in helper_source


def test_only_public_apis_and_no_stale_or_private_routes_are_taught() -> None:
    code = joined_source(read_notebook(), "code")
    source = joined_source(read_notebook())

    for forbidden in (
        "loguru",
        "nvalchemi._",
        "private logger",
        "v2 result",
        "stale result",
        "BatchDynamicBuffer",
    ):
        assert forbidden not in source
    assert not re.search(r"\.\_[A-Za-z_]+", code)


def test_only_shared_highlight_and_api_callouts_are_styled() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    shared = (REPO_ROOT / "shared" / "callouts.md").read_text()
    shared_styles = re.findall(r'^<div style="([^"]+)">', shared, flags=re.MULTILINE)
    styled = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown" and '<div style="' in cell.source
    ]

    assert len(styled) == 2
    assert markdown.count("💡 Highlight") == 1
    assert markdown.count("ALCHEMI TOOLKIT API") == 1
    for style in shared_styles:
        assert markdown.count(f'<div style="{style}">') == 1


def test_computation_shaping_display_plotting_and_interpretation_are_separate() -> None:
    for cell in code_cells(read_notebook()):
        source = cell.source
        if ".run(batch=None" in source:
            assert "display(" not in source and "plot_" not in source
        if "sink.read()" in source:
            assert "display(" not in source
        if "summarize_collected(" in source:
            assert "display(" not in source
        if "display(" in source:
            assert "pd.DataFrame(" not in source
        if "plot_" in source:
            assert ".run(" not in source and "pd.DataFrame(" not in source


def test_visuals_have_questions_takeaways_and_exportable_alt_text() -> None:
    notebook = read_notebook()
    visual_tokens = ("plot_measurement_panels(", "plot_pipeline_diagnostics(")
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


def test_try_it_and_recap_close_with_bounded_claims_and_links() -> None:
    source = joined_source(read_notebook())
    recap = next(
        cell.source
        for cell in read_notebook().cells
        if cell.cell_type == "markdown" and "## Recap" in cell.source
    )

    for token in (
        "## Try it:",
        "TRY_WARMUP_CALLS",
        "TRY_REPEATS",
        "assert len(try_timings) == TRY_REPEATS",
        "## Recap",
        "### What you learned",
        "### How we will use this",
        "Course foundation: [Core playbook]",
        "Prerequisite review: [Part 05",
        "Next deep dive: **Part 07, training and finetuning**",
        "Domain decomposition: **Part 08",
    ):
        assert token in source
    assert "speedup" not in recap.lower()


def test_all_local_notebook_links_exist() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    targets = re.findall(r"\]\((\.\./[^)#]+\.ipynb)(?:#[^)]*)?\)", markdown)

    assert targets
    for target in targets:
        assert (NOTEBOOK_PATH.parent / target).resolve().is_file(), target


def test_source_is_clean_and_outputs_have_no_errors() -> None:
    notebook = read_notebook()
    code = joined_source(notebook, "code")

    assert "print(" not in code
    assert "\\n\\n\\n" not in joined_source(notebook)
    for cell in code_cells(notebook):
        assert len(cell.metadata.get("tags", [])) == len(
            set(cell.metadata.get("tags", []))
        )
        assert not any(
            output.output_type == "error" for output in cell.get("outputs", [])
        )
