"""Static learner-facing checks for the Part 1 notebook.

These checks deliberately inspect the rendered notebook source, rather than the
generator, because the ``.ipynb`` file is the artifact learners open.  Runtime
and scientific validation remain separate concerns.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import subprocess
import symtable
import sys
from pathlib import Path
from typing import Any


PART_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PART_DIR / "alchemi-water-ir.ipynb"
BOOTCAMP_ROOT = PART_DIR.parent
GENERATOR_PATH = BOOTCAMP_ROOT / "scripts" / "rebuild_part1_ir_notebook.py"
COMPUTE_LAB_RUNBOOK = PART_DIR / "COMPUTE_LAB_RUNBOOK.md"


def _notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _code_source() -> str:
    return "\n\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def _parse_code(source: str) -> ast.Module:
    """Parse notebook code after applying IPython's transform when available."""

    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except ModuleNotFoundError:
        transformed = source
    else:
        transformed = TransformerManager().transform_cell(source)
    return ast.parse(transformed)


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _single_call(source: str, name: str) -> ast.Call:
    """Return one rendered call by its public name."""

    matches = [
        node
        for node in ast.walk(_parse_code(source))
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]
    assert len(matches) == 1, f"expected one {name}(...) call; found {len(matches)}"
    return matches[0]


def _call_keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"{_call_name(call)}(...) does not pass {name}=")


def _resolved_integer(source: str, expression: ast.expr) -> int:
    """Resolve a literal or a notebook-level named integer setting."""

    if isinstance(expression, ast.Constant) and isinstance(expression.value, int):
        return expression.value
    if isinstance(expression, ast.Name):
        resolved: set[int] = set()
        for candidate_source in (source, _code_source()):
            for node in ast.walk(_parse_code(candidate_source)):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if not any(
                    isinstance(target, ast.Name) and target.id == expression.id
                    for target in targets
                ):
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, int):
                    resolved.add(value)
        assert len(resolved) <= 1, (
            f"{expression.id} has conflicting integer definitions: {sorted(resolved)}"
        )
        if resolved:
            return resolved.pop()
    raise AssertionError("repeat count must resolve to one notebook integer setting")


def test_checked_in_notebook_matches_deterministic_generator(tmp_path: Path) -> None:
    first = tmp_path / "part1-first.ipynb"
    second = tmp_path / "part1-second.ipynb"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(GENERATOR_PATH), "--output", str(output)],
            cwd=BOOTCAMP_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    checked_in = NOTEBOOK_PATH.read_bytes()
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == checked_in


def test_compute_lab_html_export_is_learner_ready() -> None:
    runbook = COMPUTE_LAB_RUNBOOK.read_text(encoding="utf-8")
    assert 'jupyter" nbconvert' in runbook
    assert "--HTMLExporter.embed_images=True" in runbook
    assert "--TagRemovePreprocessor.enabled=True" in runbook
    assert """--TagRemovePreprocessor.remove_input_tags='{"remove-input"}'""" in runbook


def test_notebook_has_seven_stylized_sequential_stage_cards() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    expected_ids = [f"stage-{stage}" for stage in range(1, 8)]
    missing = [cell_id for cell_id in expected_ids if cell_id not in by_id]
    assert not missing, f"missing learner stage card cells: {missing}"
    rendered_stage_ids = [
        cell.get("id") for cell in cells if cell.get("id") in expected_ids
    ]
    assert rendered_stage_ids == expected_ids, (
        "the seven learner stages must appear once and in numerical order"
    )

    stage_card_markers = ('role="progressbar"', 'aria-label="Notebook stage"')
    rendered_stage_cards = sum(
        all(marker in _source(cell) for marker in stage_card_markers)
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )
    assert rendered_stage_cards == 7, (
        "the learner notebook must render exactly seven top-level stage cards; "
        f"found {rendered_stage_cards}"
    )

    expected_stage_text = {
        1: ("One structure, one result", "energy, force, and charge result"),
        2: ("Same calculation, one batch", "homogeneous/heterogeneous batch behavior"),
        3: ("Complete and check the potential", "90 NCI Atlas graphs"),
        4: (
            "Bring a model for a new domain",
            "five periodic and four finite structures",
        ),
        5: ("Prepare dynamics and IR", "charge-predicting molecular model"),
        6: ("Run and inspect the trajectory", "5,000-step NVT + 20,000-step NVE"),
        7: (
            "Choose a scaling path by workload shape",
            "one checked periodic box for the domain-parallel path",
        ),
    }

    for stage, cell_id in enumerate(expected_ids, start=1):
        cell = by_id[cell_id]
        source = _source(cell)
        assert cell.get("cell_type") == "markdown"
        required_fragments = (
            'role="region"',
            f'aria-labelledby="alchemi-stage-{stage}-heading"',
            *stage_card_markers,
            'aria-valuemin="1"',
            'aria-valuemax="7"',
            f'aria-valuenow="{stage}"',
            f"STAGE {stage} OF 7",
            "Outcome:",
            "Compute time:",
            "background:#76B900",
            f'<h2 id="alchemi-stage-{stage}-heading"',
            *expected_stage_text[stage],
        )
        absent = [fragment for fragment in required_fragments if fragment not in source]
        assert not absent, (
            f"{cell_id} is not a complete stylized progress card: {absent}"
        )
        assert source.count("<h2 ") == 1, f"{cell_id} duplicates its stage heading"
        assert f"## Stage {stage}" not in source

    stage_3 = _source(by_id["stage-3"])
    assert "earlier six-cell form using the current Toolkit versions" in stage_3
    assert "current eight-cell stage not timed" in stage_3
    roadmap = _source(by_id["roadmap"])
    assert "22.643 s" in roadmap
    assert "earlier six-cell form" in roadmap
    assert "current eight-cell Stage 3 have not been timed" in roadmap
    assert "Stage 7: scaling paths" in roadmap
    assert "**Not measured**" in roadmap
    assert "**Choose a scaling path:**" in roadmap
    assert "exercise `DomainParallel` on one GPU without decomposition" in roadmap
    assert "`DomainParallel`" in roadmap
    for depth in (
            "work directly with `AtomicData`, `Batch`, model composition",
            "PME and `DomainParallel` are",
            "walked through live on one GPU without decomposition",
            "`DistributedPipeline` is an API preview",
            "reported correctness or timing result",
    ):
        assert depth in roadmap


def test_learner_visible_cells_are_short() -> None:
    oversized = []
    for cell in _notebook()["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is True:
            continue
        line_count = len(_source(cell).splitlines())
        limit = 60
        if line_count > limit:
            oversized.append(
                f"{cell.get('id', '<missing id>')}: {line_count} lines (limit {limit})"
            )

    assert not oversized, "learner-visible code cells must stay focused:\n" + "\n".join(
        oversized
    )


def test_learner_path_stays_compact_and_starts_with_a_real_toolkit_result() -> None:
    cells = _notebook()["cells"]
    order = [cell.get("id") for cell in cells]
    visible_code = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is not True
    ]
    markdown_text = "\n".join(
        _source(cell) for cell in cells if cell.get("cell_type") == "markdown"
    )
    markdown_without_html = re.sub(r"<[^>]+>", " ", markdown_text)

    assert order.index("stage-1") < order.index("framework-primer")
    assert order.index("hello-world") < order.index("framework-primer")
    # Splitting the checked-box conversion and DomainConfig setup keeps each
    # Toolkit concept short even though it adds two visible cells. The Markdown
    # budget includes the neutral-charge and gather limitations, plus the
    # BufferConfig sketch needed for the Toolkit 0.2 distributed lesson and the
    # compact multitask model sweep.
    assert len(visible_code) <= 55
    assert sum(len(_source(cell).splitlines()) for cell in visible_code) <= 1_800
    assert len(re.findall(r"\b[\w-]+\b", markdown_without_html)) <= 6_500


def test_setup_and_persistence_plumbing_are_collapsed_and_export_hidden() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    for cell_id in (
        "setup",
        "tutorial-settings",
        "helper-imports",
        "results-summary",
        "save",
    ):
        metadata = by_id[cell_id].get("metadata", {})
        assert metadata.get("jupyter", {}).get("source_hidden") is True
        assert "remove-input" in metadata.get("tags", [])

    imports_metadata = by_id["imports"].get("metadata", {})
    assert imports_metadata.get("jupyter", {}).get("source_hidden") is not True

    for cell_id in (
        "validate-nci-graph-order",
        "domain-parallel-results",
        "display-domain-parallel-results",
        "pipeline-campaign-results",
    ):
        metadata = by_id[cell_id].get("metadata", {})
        assert metadata.get("jupyter", {}).get("source_hidden") is True
        assert "remove-input" in metadata.get("tags", [])


def test_hidden_setup_locates_the_part_directory_before_importing_aux() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    setup = _source(by_id["setup"])

    for bootstrap_step in (
        "def locate_part_directory() -> Path:",
        'os.environ.get("ALCHEMI_BOOTCAMP_ROOT")',
        "starts.extend((cwd, *cwd.parents))",
        '(candidate / "aux").is_dir()',
        'candidate / "alchemi-water-ir.ipynb"',
        "PART_DIR = locate_part_directory()",
    ):
        assert bootstrap_step in setup
    assert setup.index("PART_DIR = locate_part_directory()") < setup.index(
        "from aux.ui import"
    )
    assert setup.index("PART_DIR = locate_part_directory()") < setup.index(
        "from aux.domain.config import"
    )


def test_notebook_hero_and_presentation_blocks_share_one_visual_system() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    title = _source(by_id["title"])
    banner_relative = (
        "assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png"
    )
    banner = PART_DIR / banner_relative

    assert '<h1 id="alchemi-notebook-title"' in title
    assert 'aria-label="Lesson summary"' in title
    assert 'aria-label="NOTE"' in title
    assert "WHAT TO CHECK" in title
    assert "BOUNDARY" not in title
    assert "periodic PME" in title
    assert "DomainParallel" in title
    assert "DistributedPipeline" in title
    assert (
        "The merged notebook and Stage 7 have not been measured "
        "with the current Toolkit versions."
    ) in title
    assert banner_relative in title
    assert banner.is_file()
    png = banner.read_bytes()[:24]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(png[16:20], "big") == 2880
    assert int.from_bytes(png[20:24], "big") == 1440

    presentation_markers = (
        'aria-label="Lesson summary"',
        'aria-label="Process diagram:',
        'aria-label="Notebook stage"',
        'role="note"',
    )
    styled_blocks = [
        _source(cell)
        for cell in cells
        if any(marker in _source(cell) for marker in presentation_markers)
    ]
    assert styled_blocks
    assert all("max-width:880px" in source for source in styled_blocks)
    assert all("max-width:720px" not in source for source in styled_blocks)
    assert all("max-width:840px" not in source for source in styled_blocks)


def test_notebook_uses_plain_scientific_computing_language() -> None:
    source = "\n".join(_source(cell) for cell in _notebook()["cells"])
    unwanted = (
        "BOUNDARY",
        'kind="boundary"',
        "RESULT — WITHHELD",
        "Claim ledger",
        "Single-trajectory boundary",
        "reporting gate",
        "force gate",
        "workflow contract",
        "model contract",
        "Recorded provenance",
        "acceptance records",
        "evidence_state",
        "live_evidence",
        "pinned ALCHEMI environment",
        "source pin",
        "target-level workflow consistency",
    )
    present = [phrase for phrase in unwanted if phrase in source]
    assert not present, f"learner-facing governance language remains: {present}"
    assert "provenance=" not in source
    assert "gates=" not in source
    assert '["provenance"]' not in source
    assert "—" not in source

    for expected in (
        'aria-label="NOTE"',
        'aria-label="WHAT TO CHECK"',
        "NOT REPORTED",
        "### Results summary",
        "DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON",
        "nci_metrics",
    ):
        assert expected in source


def test_important_compute_cells_have_visible_progress_cards() -> None:
    """Keep progress visible around model loading, evaluation, and long work."""

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    expected = {
        "setup": "Check the tutorial environment",
        "tutorial-settings": "Show the tutorial settings",
        "imports": "Load public Toolkit APIs",
        "helper-imports": "Load tutorial helpers",
        "framework-primer-example": "Run the PyTorch and JAX bindings",
        "framework-primer-warp": "Call the raw Warp operation",
        "load-aimnet": "Load the verified AIMNet2 checkpoint",
        "inspect-float-precision": "Inspect floating-point precision",
        "first-prediction": "one water dimer",
        "serial-batch-agreement": "Compare individual and batched calls",
        "cpu-gpu-crossover": "Measure first and warm CPU/GPU calls",
        "cpu-gpu-sweep": "Measure the warm CPU/GPU crossover",
        "display-cpu-gpu-sweep": "Show the CPU/GPU crossover",
        "build-batch-layouts": "Build one mixed batch and three size buckets",
        "batch-layouts": "Compare mixed and bucketed batches",
        "load-nci-atlas": "Build the 90-graph NCI Atlas batch",
        "configure-nci-model": "Configure AIMNet, Coulomb, and D3",
        "evaluate-nci-components": "Evaluate the NCI set in nine batched calls",
        "compose-nci-pipeline": "Compose the complete model",
        "validate-nci-graph-order": "Check NCI graph ordering",
        "check-nci-force": "Check one complete-model force",
        "analyze-nci-curves": "Compare interaction curves with two references",
        "display-nci-curves": "Show the NCI comparison",
        "component-ablation": "Evaluate model components",
        "official-composition-agreement": "Verify the composed model",
        "define-sevennet-config": "Declare the SevenNet Toolkit interface",
        "define-sevennet-wrapper": "Define the SevenNet-Omni Toolkit adapter",
        "load-sevennet-wrapper": "Load the SevenNet-Omni adapter",
        "compose-sevennet-surface-model": "Compose SevenNet-Omni with pairwise PBE-D3(BJ)",
        "build-adsorption-panel": "Load the adsorption starting structures",
        "pack-adsorption-batches": "Pack periodic and finite adsorption batches",
        "compare-sevennet-tasks": "Inspect and switch SevenNet tasks",
        "view-adsorption-panel": "Inspect the four surface inputs in OVITO",
        "run-sevennet-wrapper": "Evaluate nine structures in two batches",
        "validate-sevennet-wrapper": "Validate the custom adapter",
        "analyze-adsorption-panel": "Show energies and forces",
        "build-compiled-ir-model": "Compile the fixed IR model",
        "compile-fixed-ir-model": "Check the compiled model against eager execution",
        "relax": "Batched FIRE2 relaxation",
        "validate-relaxation": "Validate the relaxed structures",
        "save-relaxed-structures": "Save and replay the relaxed batch",
        "harmonic-minimum": "Tighten the water-monomer minimum",
        "harmonic-finite-difference": "Evaluate all harmonic displacements in three batches",
        "harmonic-comparison": "Map AIMNet + Coulomb + D3 and B97-3c modes",
        "run-dynamics": "Run the full NVT → NVE trajectory",
        "inflight-example": "Build the inflight queue",
        "configure-inflight-stage": "Configure the inflight fused stage",
        "run-inflight-example": "Run and inspect the inflight queue",
        "build-domain-box": "Load the checked periodic base box",
        "compose-domain-model": "Compose AIMNet2, PME, and D3",
        "run-domain-single-gpu": "Walk through DomainParallel on one GPU",
        "inspect-domain-molecule-charges": "Resolve predicted charge by molecule",
        "domain-parallel-results": "Load recorded H100 domain results",
        "display-domain-parallel-results": "Display recorded domain results",
        "pipeline-campaign-results": "Report the pipeline status",
    }
    for cell_id, title in expected.items():
        source = _source(by_id[cell_id])
        assert "NotebookProgress(" in source, cell_id
        assert title in source, cell_id

    run_source = _source(by_id["run-dynamics"])
    assert run_source.index("NotebookProgress(") < run_source.index(
        "torch.cuda.synchronize()"
    )


def test_key_results_remain_visible_and_the_final_inventory_is_complete() -> None:
    """Hide long analysis code without hiding the outputs learners need."""

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    for cell_id in (
        "component-ablation",
        "official-composition-agreement",
        "full-pipeline-agreement",
        "dimer-ablation-plot",
    ):
        jupyter = by_id[cell_id].get("metadata", {}).get("jupyter", {})
        assert jupyter.get("source_hidden") is True
        assert jupyter.get("outputs_hidden") is not True

    adsorption = _source(by_id["analyze-adsorption-panel"])
    assert 'label="Toolkit-to-SevenNet graph mapping"' in adsorption
    assert 'label="Adapter and pipeline numerical checks"' in adsorption

    save = _source(by_id["save"])
    assert "saved_run.relative_files" in save
    assert 'label="Files written by the final report"' in save
    assert "path.name.startswith" not in save
    assert "run manifest also inventories the" in save


def test_every_executable_cell_has_a_stylized_progress_card() -> None:
    """Keep one consistent progress treatment across the complete lesson."""

    missing = [
        cell.get("id", "<missing id>")
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code" and "NotebookProgress(" not in _source(cell)
    ]
    assert not missing, f"code cells without NotebookProgress: {missing}"


def test_notebook_exposes_core_toolkit_apis_in_executable_cells() -> None:
    source = _code_source()
    required_calls = {
        "ASE to Toolkit conversion": "AtomicData.from_atoms(",
        "Toolkit batch construction": "Batch.from_data_list(",
        "single-graph recovery": ".get_data(",
        "batch round trip": ".to_data_list(",
        "graph selection": ".index_select(",
        "neighbor construction": "compute_neighbors(",
        "AIMNet2 checkpoint loading": "AIMNet2Wrapper.from_checkpoint(",
        "AIMNet2 wrapper construction": "AIMNet2Wrapper(",
        "custom model capability declaration": "ModelConfig(",
        "custom model neighbor declaration": "NeighborConfig(",
        "raw SevenNet-Omni loading": "load_raw_sevennet_omni(",
        "custom SevenNet-Omni adapter": "SevenNetOmniWrapper(",
        "explicit D3 component": "DFTD3ModelWrapper(",
        "explicit electrostatics component": "DirectCoulombWrapper(",
        "explicit pipeline step": "PipelineStep(",
        "pipeline grouping": "PipelineGroup(",
        "pipeline construction": "PipelineModelWrapper(",
        "segmented reduction": "segmented_sum(",
        "FIRE2 relaxation": "FIRE2(",
        "force convergence hook": "ConvergenceHook.from_fmax(",
        "Langevin dynamics": "NVTLangevin(",
        "NVE dynamics": "NVE(",
        "velocity initialization": "initialize_velocities(",
        "fused dynamics": "FusedStage",
        "size-aware inflight sampler": "SizeAwareSampler",
        "inflight dataset": "InMemoryDataset(",
        "inflight result sink": "HostMemory(",
        "inflight source preparation": "prepare_inflight_dimer_source(",
        "periodic electrostatics": "PMEModelWrapper(",
        "domain configuration": "DomainConfig(",
        "single-system domain execution": "DomainParallel(",
        "hook protocol": ": Hook =",
        "NaN safety hook": "NaNDetectorHook(",
        "logging hook": "LoggingHook(",
        "Toolkit Zarr sink": "ZarrData(",
        "Toolkit neighbor hooks": ".make_neighbor_hooks(",
        "hook registration": ".register_hook(",
    }
    missing = [label for label, token in required_calls.items() if token not in source]
    assert not missing, (
        f"core Toolkit calls hidden from learner-facing cells: {missing}"
    )
    assert "DynamicsContext" in _source(
        next(cell for cell in _notebook()["cells"] if cell.get("id") == "ir-mechanism")
    )
    assert 'converge_after_steps("nvt_steps_done", WARMUP_STEPS)' in source
    assert "register_fused_hook(" in source
    assert "DistributedPipeline(" not in source
    assert "BufferConfig(" not in source
    assert "ConvergedSnapshotHook(" not in source
    assert "init_process_group(" not in source
    assert "pipeline.run()" not in source


def test_stage_2_timings_use_repeated_blocks_and_report_median_iqr() -> None:
    """Show variability for both device and batch-layout comparisons."""

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    sweep = _source(by_id["cpu-gpu-sweep"])
    sweep_display = _source(by_id["display-cpu-gpu-sweep"])
    layouts = _source(by_id["batch-layouts"])

    sweep_call = _single_call(sweep, "benchmark_device_sweep")
    sweep_repeats = _resolved_integer(
        sweep, _call_keyword(sweep_call, "measured_repeats")
    )
    assert sweep_repeats >= 3
    _call_keyword(sweep_call, "energy_atol")
    _call_keyword(sweep_call, "energy_rtol")

    layout_call = _single_call(layouts, "compare_mixed_and_bucketed")
    layout_repeats = _resolved_integer(
        layouts, _call_keyword(layout_call, "measured_repeats")
    )
    assert layout_repeats >= 3

    for rendered_table in (sweep_display, layouts):
        assert "median_structures_per_s" in rendered_table
        assert "median_atoms_per_s" in rendered_table
        assert "relative_iqr" in rendered_table
    assert "plot_device_sweep(crossover)" in sweep_display
    assert "interquartile range" in sweep_display
    assert "max_abs_energy_difference" in sweep_display + layouts


def test_notebook_explains_hooks_before_registering_them() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]
    lesson = _source(by_id["hooks-quick-note"])
    registration = _source(by_id["attach-dynamics-hooks"])

    for term in (
        "chosen point and frequency",
        "current `Batch`",
        "`NaN` means *not a number*",
        "`Inf` means a value overflowed to infinity",
        "later trajectory steps unreliable",
        "`NaNDetectorHook` checks energy and forces",
        "rebuild neighbors",
        "detect convergence",
        "record dipoles",
        "write a log",
        "`register_hook(...)`",
        "`register_fused_hook(...)`",
    ):
        assert term in lesson

    assert order.index("check-ir-dipoles") < order.index("hooks-quick-note")
    assert order.index("hooks-quick-note") < order.index("relax")
    assert 'NaNDetectorHook(frequency=100, extra_keys=["velocities"])' in registration


def test_notebook_wires_raw_sevennet_into_toolkit_for_surface_single_points() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]

    intro = _source(by_id["surface-model-switch"])
    model_config = _source(by_id["sevennet-model-config"])
    input_map = _source(by_id["sevennet-input-map"])
    output_map = _source(by_id["sevennet-output-map"])
    explanation = "\n".join((intro, model_config, input_map, output_map))
    config_code = _source(by_id["define-sevennet-config"])
    define = _source(by_id["define-sevennet-wrapper"])
    helper_imports = _source(by_id["helper-imports"])
    load = "\n".join(
        (
            _source(by_id["load-sevennet-wrapper"]),
            _source(by_id["compose-sevennet-surface-model"]),
        )
    )
    build = "\n".join(
        (
            _source(by_id["build-adsorption-panel"]),
            _source(by_id["pack-adsorption-batches"]),
            (PART_DIR / "aux" / "adsorption.py").read_text(encoding="utf-8"),
        )
    )
    view = _source(by_id["view-adsorption-panel"])
    run = "\n".join(
        (
            _source(by_id["run-sevennet-wrapper"]),
            _source(by_id["validate-sevennet-wrapper"]),
        )
    )
    analyze = _source(by_id["analyze-adsorption-panel"])
    save = _source(by_id["save"])
    save_plumbing = "\n".join((_source(by_id["results-summary"]), save))

    for term in (
        "Change the model when the chemistry changes",
        "cannot represent Cu",
        "model limit, not a Toolkit limit",
        "materials, surfaces, and adsorption tests",
        "`mpa` task targets PBE(+U) energies and forces and does not include D3",
        "CO, CO₂, NH₃, and CH₃OH",
        "fixed starting geometries",
        "not relaxed adsorption energies",
        "`AtomicData`, `Batch`",
        "`ModelConfig` tells Toolkit",
        "energy and forces",
        "periodic support",
        "full COO neighbors",
        "`skin=0`",
        "direct forces",
        "does not attach AIMNet charges",
        "COO neighbor list",
        "periodic shifts and cells",
        "one `mpa` label per graph",
        "inferred_total_energy",
        "inferred_force",
        'PipelineModelWrapper(..., neighbor_adaptation="always")',
    ):
        assert term in explanation

    assert order.index("surface-model-switch") < order.index("sevennet-model-config")
    assert order.index("sevennet-model-config") < order.index("sevennet-input-map")
    assert order.index("sevennet-input-map") < order.index("sevennet-output-map")
    assert order.index("sevennet-output-map") < order.index("define-sevennet-config")
    assert order.index("define-sevennet-config") < order.index(
        "define-sevennet-wrapper"
    )

    for term in (
        "def make_sevennet_model_config(cutoff: float) -> ModelConfig:",
        "return ModelConfig(",
        'outputs=frozenset({"energy", "forces"})',
        "supports_pbc=True",
        "needs_pbc=False",
        "NeighborConfig(",
        "format=NeighborListFormat.COO",
        "half_list=False",
        "skin=0.0",
    ):
        assert term in config_code

    for term in (
        "class SevenNetOmniWrapper(_SevenNetAdapterBase, BaseModelMixin):",
        "self.model_config = make_sevennet_model_config(self.cutoff)",
        "def direct_derivative_keys(self) -> set[str]:",
        "def adapt_input(self, data: Batch",
        "_toolkit_batch_to_sevennet_graph(",
        "def adapt_output(self, raw: Any, data: Batch)",
        "_map_sevennet_outputs(raw, data)",
        "def forward(self, data: Batch",
        "raw = self.model(graph)",
        "super().adapt_output(mapped, data)",
    ):
        assert term in define

    wrapper_module = (PART_DIR / "aux" / "models" / "sevennet.py").read_text(
        encoding="utf-8"
    )
    maintained_config_source = (
        wrapper_module.split("# BEGIN NOTEBOOK MODEL CONFIG", 1)[1]
        .split("# END NOTEBOOK MODEL CONFIG", 1)[0]
        .strip()
    )
    maintained_wrapper_source = (
        wrapper_module.split("# BEGIN NOTEBOOK WRAPPER", 1)[1]
        .split("# END NOTEBOOK WRAPPER", 1)[0]
        .strip()
    )
    assert maintained_config_source in config_code
    assert maintained_wrapper_source in define
    assert len(maintained_config_source.splitlines()) <= 20
    assert len(maintained_wrapper_source.splitlines()) <= 40
    for private_name in (
        "_SevenNetAdapterBase",
        "_model_device_and_dtype",
        "_map_sevennet_outputs",
        "_toolkit_batch_to_sevennet_graph",
    ):
        assert private_name in helper_imports
        assert "from aux.models.sevennet import" not in define
    for base_term in (
        "class _SevenNetAdapterBase(nn.Module):",
        "_validated_sevennet_metadata(",
        "self.register_buffer(",
        "def cutoff(self) -> float:",
        "def embedding_shapes(self)",
        "def compute_embeddings(",
    ):
        assert base_term in wrapper_module
    for term in (
        "def _map_sevennet_outputs(raw: Any, data: Batch)",
        "energy = raw[_PREDICTED_ENERGY]",
        "forces = raw[_PREDICTED_FORCE]",
        "energy.numel() != data.num_graphs",
        "forces.shape != data.positions.shape",
        '"energy": energy.detach().reshape(data.num_graphs, 1)',
        '"forces": forces.detach()',
    ):
        assert term in wrapper_module

    for term in (
        "resolve_sevennet_checkpoint()",
        "load_raw_sevennet_omni(",
        "SevenNetOmniWrapper(",
        "modality=SEVENNET_MODALITY",
        "sevennet_model.model_config",
        "sevennet_config.supports_pbc",
        "NeighborListFormat.COO",
        "DFTD3ModelWrapper(",
        "PBE_D3_BJ_A1",
        "cutoff=SURFACE_D3_CUTOFF_A",
        "smoothing_fraction=D3_REFERENCE_SMOOTHING_FRACTION",
        "PipelineModelWrapper(",
        "PipelineStep(model=sevennet_model)",
        "PipelineStep(model=surface_d3)",
    ):
        assert term in load

    for term in (
        "load_initial_structure_set()",
        "load_adsorption_methodology()",
        "split_for_batches(",
        "AtomicData.from_atoms(",
        "Batch.from_data_list(",
        "periodic_surface_batch",
        "finite_molecule_batch",
        '"geometry": atoms.info["geometry_status"]',
        'label="Four fixed starting placements"',
    ):
        assert term in build
    assert "charge" not in build
    assert "spin" not in build
    assert "adsorption_widget_grid(" in view
    assert "ADSLAB_KEYS[name]" in view

    for term in (
        "config=sevennet_config.neighbor_config",
        "periodic_model_outputs = sevennet_model(periodic_surface_batch)",
        "finite_model_outputs = sevennet_model(finite_molecule_batch)",
        "config=surface_d3.model_config.neighbor_config",
        "periodic_d3_outputs = surface_d3(periodic_surface_batch)",
        "finite_d3_outputs = surface_d3(finite_molecule_batch)",
        "periodic_pipeline_outputs = surface_model(periodic_surface_batch)",
        "finite_pipeline_outputs = surface_model(finite_molecule_batch)",
        "finalize_sevennet_lesson(",
        "wrapper=sevennet_model",
        "raw_model=raw_sevennet",
        "periodic_model_outputs=periodic_model_outputs",
        "finite_model_outputs=finite_model_outputs",
        "periodic_d3_outputs=periodic_d3_outputs",
        "finite_d3_outputs=finite_d3_outputs",
        "periodic_pipeline_outputs=periodic_pipeline_outputs",
        "finite_pipeline_outputs=finite_pipeline_outputs",
        "energy_tolerance_eV_per_atom=SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM",
        "force_tolerance_eV_A=SEVENNET_REPEAT_FORCE_TOL_EV_A",
        "sevennet_lesson.graph_mapping",
        "sevennet_lesson.official_calculator_check",
        "sevennet_lesson.numerical_agreement",
        "surface_combined_energies",
        "surface_combined_forces",
    ):
        assert term in run

    lesson_helper = (PART_DIR / "aux" / "models" / "sevennet_lesson.py").read_text(
        encoding="utf-8"
    )
    for term in (
        "build_sevennet_mapping_table(wrapper, periodic_batch)",
        "build_sevennet_repeat_table(",
        "SevenNetCalculator(",
        '"custom adapter vs official SevenNetCalculator"',
        '"pipeline output vs explicit component sum"',
        "split_model_outputs(",
        'agreement["energy_difference_eV_per_atom"].max()',
        'agreement["max_force_component_difference_eV_A"].max()',
    ):
        assert term in lesson_helper

    for term in (
        "surface_energy_table = pd.DataFrame",
        "assemble_adsorption_results(",
        "model_adsorption_energy_eV",
        "d3_adsorption_energy_eV",
        "adsorption_energy_eV",
        "build_full_force_table(",
        '"SevenNet-Omni mpa"',
        '"pairwise PBE-D3(BJ)"',
        '"SevenNet-Omni mpa + pairwise PBE-D3(BJ)"',
        "surface_force_summary",
        "summarize_adslab_force_regions(",
        'result_state="not_reported"',
        "sevennet_graph_mapping",
        "sevennet_numerical_agreement",
        "single points are not adsorption minima",
        "E(adslab) - E(clean slab) - E(gas)",
        "lateral periodic-image interactions",
        "zero-point, thermal, and entropy terms",
    ):
        assert term in analyze

    for term in (
        "results=water_run_results",
        "**manifest_input.as_save_arguments()",
        "save_water_run_outputs(",
    ):
        assert term in save_plumbing

    surface_cells = explanation + define + load + build + view + run + analyze
    for forbidden in (
        "OrbMol",
        "orbmol",
        "MACEWrapper",
        "water or pure H2",
        "relaxer.run(",
        "FIRE2(",
    ):
        assert forbidden not in surface_cells

    assert order.index("display-nci-curves") < order.index("stage-4")
    assert order.index("stage-4") < order.index("surface-model-switch")
    assert order.index("surface-model-switch") < order.index("define-sevennet-config")
    assert order.index("define-sevennet-config") < order.index(
        "define-sevennet-wrapper"
    )
    assert order.index("define-sevennet-wrapper") < order.index("load-sevennet-wrapper")
    assert order.index("load-sevennet-wrapper") < order.index(
        "compose-sevennet-surface-model"
    )
    assert order.index("compose-sevennet-surface-model") < order.index(
        "build-adsorption-panel"
    )
    assert order.index("build-adsorption-panel") < order.index(
        "pack-adsorption-batches"
    )
    assert order.index("pack-adsorption-batches") < order.index("view-adsorption-panel")
    assert order.index("view-adsorption-panel") < order.index("run-sevennet-wrapper")
    assert order.index("run-sevennet-wrapper") < order.index(
        "validate-sevennet-wrapper"
    )
    assert order.index("validate-sevennet-wrapper") < order.index(
        "analyze-adsorption-panel"
    )
    assert order.index("analyze-adsorption-panel") < order.index("stage-5")

    compile_cell = _source(by_id["compile-fixed-ir-model"])
    for term in (
        "fixed_ir_batch = batch.clone()",
        "compute_neighbors(fixed_ir_batch",
        'IR_COMPARE_OUTPUTS = ("energy", "forces", "charges")',
        "snapshot_tensor_fields(",
        "compiled_ir_raw = model(fixed_ir_batch)",
        "compiled_repeat_raw = model(fixed_ir_batch)",
        "max_absolute_differences(",
        "COMPILED_EAGER_ENERGY_TOLERANCE_EV",
        "COMPILED_REPEAT_FORCE_TOLERANCE_EV_A",
    ):
        assert term in compile_cell


def test_notebook_discovers_and_switches_sevennet_tasks() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]
    lesson = _source(by_id["sevennet-tasks"])
    run = _source(by_id["compare-sevennet-tasks"])
    helper_imports = _source(by_id["helper-imports"])

    for term in (
        "One checkpoint, several tasks",
        "tasks** or **modalities**, not separate output",
        "loaded checkpoint",
        "`mpa`",
        "`oc20`",
        "model sweep",
        "one task per call",
        "Do not compare raw total energies across tasks",
        "same task and the same correction or composite-model scheme",
        "does not add D3",
        "Batch.index_select(...)",
        "does not automatically send different graphs to different model objects",
        "routing different terms of one energy cycle to different full models is invalid",
    ):
        assert term in lesson

    for term in (
        "sorted(str(name) for name in raw_sevennet.modal_map)",
        '"mpa": "PBE(+U)-level MPtrj/sAlex + cross-domain data"',
        '"oc20": "RPBE OC20 metal-catalyst adsorption data"',
        "available_sevennet_tasks",
        "Tasks reported by the loaded checkpoint",
    ):
        assert term in run

    for term in (
        "periodic_surface_batch.index_select(task_probe_indices)",
        "SevenNetOmniWrapper(",
        "raw_sevennet, modality=task",
        "task_probe_batch.clone()",
        "compute_neighbors(task_batch",
        "task_outputs = task_model(task_batch)",
        'task_outputs["energy"]',
        'task_outputs["forces"]',
        "torch.isfinite(energies).all()",
        "torch.isfinite(forces).all()",
        "sevennet_task_outputs",
            "summarize_sevennet_task_outputs(",
            "sevennet_task_summary",
            "method sensitivity, not accuracy",
            "energies remain available",
            "main calculation uses mpa",
    ):
        assert term in run

    assert "summarize_sevennet_task_outputs" in helper_imports
    assert "raw total energies / eV" not in run
    assert "surface_d3(" not in run
    assert "PipelineModelWrapper(" not in run
    assert order.index("pack-adsorption-batches") < order.index(
        "sevennet-tasks"
    )
    assert order.index("sevennet-tasks") < order.index(
        "compare-sevennet-tasks"
    )
    assert order.index("compare-sevennet-tasks") < order.index(
        "view-adsorption-panel"
    )


def test_notebook_builds_and_displays_a_live_sevennet_model_card() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    helper_imports = _source(by_id["helper-imports"])
    compose = _source(by_id["compose-sevennet-surface-model"])

    assert "build_sevennet_model_card" in helper_imports
    card_call = _single_call(compose, "build_sevennet_model_card")
    model = _call_keyword(card_call, "model")
    wrapper = _call_keyword(card_call, "wrapper")
    assert isinstance(model, ast.Name) and model.id == "raw_sevennet"
    assert isinstance(wrapper, ast.Name) and wrapper.id == "sevennet_model"

    assert "sevennet_model_card = build_sevennet_model_card(" in compose
    assert "readable_table(" in compose
    assert "sevennet_model_card" in compose
    assert "SevenNet-Omni" in compose
    assert compose.index("build_sevennet_model_card(") < compose.index(
        "readable_table(\n    sevennet_model_card"
    )


def test_notebook_uses_the_core_model_aware_neighbor_path() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]
    lesson = _source(by_id["core-neighbor-path"])

    for term in (
        "several neighbor-search implementations",
        "small, nonperiodic molecular batch",
        "Toolkit chooses a compatible implementation",
        "model.model_config.neighbor_config",
        "declared cutoff, output format, and full- or half-list setting",
        "model.make_neighbor_hooks()",
        "configured skin",
        "Direct kernel selection belongs in a separate performance study",
    ):
        assert term in lesson
    assert "ops_neighbor_list(" not in _code_source()
    assert "ops-neighbor-dispatch" not in by_id
    assert order.index("serial-batch-agreement") < order.index("core-neighbor-path")
    assert order.index("core-neighbor-path") < order.index("cpu-gpu-crossover")


def test_composition_force_check_uses_the_independent_energy_route() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    source = _source(by_id["official-composition-agreement"])

    for term in (
        "FD_STEP_A = COMPOSITION_FD_STEP_A",
        "official_displaced = official(",
        'official_displaced["energy"]',
        "official_fd_force_error",
        "COMPOSITION_FD_FORCE_TOLERANCE_EV_A",
        "large constant atomic",
        "finite differences of the resulting ~keV totals",
    ):
        assert term in source
    assert 'model(fd_batch)["energy"]' not in source
    assert "best_fd_row" not in source
    assert "FD_STEPS_A" not in source


def test_notebook_orients_new_alchemi_users_before_the_first_model_call() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]

    orientation = _source(by_id["alchemi-orientation"])
    for term in (
        "Toolkit Core (`nvalchemi`)",
        "Toolkit-Ops (`nvalchemiops`)",
        "AIMNet2",
        "PyTorch",
        "JAX",
        "Warp",
        "**`aux/`** contains tutorial-only",
    ):
        assert term in orientation
    assert order.index("alchemi-orientation") < order.index("setup")

    atomistic_loop = _source(by_id["atomistic-loop"])
    assert "each independent atomistic system in a `Batch`" in atomistic_loop
    assert (
        "does not mean a chemical bond diagram or a PyTorch computational graph"
        in atomistic_loop
    )
    assert order.index("atomistic-loop") < order.index("hello-world")

    hello = _source(by_id["hello-world"])
    for term in (
        'label="First model outputs by atom"',
        '"charge (e)"',
        '"Fx (eV/Å)"',
        '"|F| (eV/Å)"',
        'print("energy :"',
        'print("total predicted charge:"',
    ):
        assert term in hello

    batch_model = _source(by_id["batch-mental-model"])
    for term in (
        "E(AB) - E(A) - E(B)",
        "num_nodes_per_graph",
        "batch_idx",
        "batch_ptr",
        "segmented_sum(values, batch_idx, num_graphs)",
        "differentiation path",
    ):
        assert term in batch_model
    assert "not a chemical bond diagram" in batch_model
    assert order.index("batch-mental-model") < order.index("first-prediction")


def test_notebook_explains_pytorch_jax_and_warp_after_the_first_result() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]

    primer = _source(by_id["framework-primer"])
    for term in (
        "segmented sum",
        "PyTorch binding",
        "PyTorch tensors and PyTorch gradients",
        "JAX binding",
        "JAX arrays and JAX gradients",
        "raw Warp operation",
        "typed GPU arrays and explicit output storage",
        "array and automatic-differentiation front ends",
        "GPU-kernel layer",
        "Toolkit Core",
        "not a performance benchmark",
    ):
        assert term in primer

    bindings = _source(by_id["framework-primer-example"])
    warp = _source(by_id["framework-primer-warp"])
    helper = (PART_DIR / "aux" / "framework_comparison.py").read_text(encoding="utf-8")
    example = "\n".join((bindings, warp, helper))
    for term in (
        "import jax",
        "import jax.numpy as jnp",
        "import warp as wp",
        "from nvalchemiops.torch.segment_ops import segmented_sum as torch_segmented_sum",
        "from nvalchemiops.jax.segment_ops import segmented_sum as jax_segmented_sum",
        "from nvalchemiops.segment_ops import segmented_sum as warp_segmented_sum",
        "torch_segmented_sum(",
        "jax_segmented_sum(jax_values, jax_graph_idx, num_segments=2)",
        "warp_segmented_sum(warp_values, warp_graph_idx, warp_totals)",
        "torch.autograd.grad(",
        "jax.grad(",
        "jax_totals.block_until_ready()",
        "torch.cuda.synchronize(DEVICE)",
        "wp.from_torch(torch_values.detach(), dtype=wp.float32)",
        "wp.zeros(2, dtype=wp.float32",
        "np.testing.assert_allclose(",
        "readable_table(",
        'columns=["Path", "Array", "Dtype", "Device", "Totals", "Output"]',
        "all three routes returned [3, 7]",
        'kind="result"',
        'result_state="pass"',
    ):
        assert term in example
    assert "@wp.kernel" not in example
    assert "wp.Tape" not in example
    assert "three interfaces" not in example
    assert len(bindings.splitlines()) <= 60
    assert len(warp.splitlines()) <= 60

    setup = _source(by_id["setup"])
    assert 'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"' in setup
    assert "import jax" not in setup

    assert order.index("imports") < order.index("stage-1")
    assert order.index("hello-world") < order.index("framework-primer")
    assert order.index("framework-primer") < order.index("framework-primer-example")
    assert order.index("framework-primer-example") < order.index(
        "framework-primer-warp"
    )
    assert order.index("framework-primer-warp") < order.index("precision-note")


def test_declared_runtime_supports_the_torch_jax_warp_primer() -> None:
    requirements = (BOOTCAMP_ROOT / "build" / "requirements.txt").read_text(
        encoding="utf-8"
    )
    dockerfile = (BOOTCAMP_ROOT / "build" / "Dockerfile").read_text(encoding="utf-8")
    runtime_check = (BOOTCAMP_ROOT / "build" / "verify_part1_runtime.py").read_text(
        encoding="utf-8"
    )

    assert "nvalchemi-toolkit-ops[torch-cu13,jax-cu13]" in requirements
    assert "jax[cuda13]==0.9.0.1" in requirements
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in dockerfile
    assert "import torch, jax" in dockerfile
    for term in (
        '"jax": "0.9.0.1"',
        'device.platform == "gpu"',
        "jax_segmented_sum(",
        "warp_segmented_sum(",
        "segmented_sum(values, graph_index, 1)",
    ):
        assert term in runtime_check


def test_precision_lesson_distinguishes_tensor_storage_from_model_math() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    order = [cell.get("id") for cell in cells]

    note = _source(by_id["precision-note"])
    for term in (
        "`float32` uses 4 bytes per value",
        "`float64` uses 8 bytes",
        "Atomic numbers and neighbor indices remain integer tensors",
        "`Tensor.to(dtype)` returns a converted tensor",
        "`Module.to(dtype)` changes the module's floating parameters and buffers in place",
        "A float64 coordinate tensor does not turn it into a float64 model",
        "widening saved float32 weights cannot recover information",
        "torch.set_float32_matmul_precision",
        "highest",
        "not model weights or input dtypes",
    ):
        assert term in note

    lesson = _source(by_id["inspect-float-precision"])
    precision_helper = (PART_DIR / "aux" / "precision.py").read_text(encoding="utf-8")
    implementation = lesson + "\n" + precision_helper
    for term in (
        "model.parameters()",
        "parameter.element_size()",
        "dtype=torch.float64",
        "aimnet.adapt_input(precision_probe_batch)",
        'precision_model_input["coord"].dtype == torch.float32',
        "torch.nextafter(",
        "torch.get_float32_matmul_precision()",
        "precision_summary.widening_preserves_stored_values",
        "A force may return",
        "float32 model calculation",
        "numerical resolution, not model error",
    ):
        assert term in implementation
    assert "aimnet.double()" not in lesson
    assert "aimnet.to(dtype=torch.float64)" not in lesson
    assert order.index("hello-world") < order.index("precision-note")
    assert order.index("precision-note") < order.index("inspect-float-precision")
    assert order.index("inspect-float-precision") < order.index("batch-mental-model")


def test_stage_7_keeps_each_scaling_api_with_the_right_workload_shape() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    stage_7 = _source(by_id["stage-7"])

    for workload, api in (
        ("independent systems that fit together", "Batch"),
        ("one active batch at different workflow stages", "FusedStage"),
        ("a queue larger than the active batch", "FusedStage"),
        ("one large periodic system", "DomainParallel"),
        ("independent batches moving through different stages", "DistributedPipeline"),
    ):
        assert workload in stage_7
        assert api in stage_7
    assert "one-GPU walkthrough with no decomposition" in stage_7
    assert "checked H100 capacity, OOM recovery, and speed" in stage_7
    assert "otherwise `NOT REPORTED`" in stage_7
    assert "API sketch only" in stage_7

    milestones = (
        "stage-7",
        "inflight-intro",
        "run-inflight-example",
        "domain-decomposition-intro",
        "run-domain-single-gpu",
        "inspect-domain-molecule-charges",
        "domain-parallel-api",
        "domain-scaling-plan",
        "domain-parallel-results",
        "distributed-pipeline-intro",
        "pipeline-campaign-results",
        "results-summary",
    )
    positions = [
        next(i for i, cell in enumerate(cells) if cell.get("id") == cell_id)
        for cell_id in milestones
    ]
    assert positions == sorted(positions)


def test_aimnet_card_and_fused_stage_state_the_important_limits() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    model_card = _source(by_id["load-aimnet"])
    fused_intro = _source(by_id["fused-stage-intro"])

    for detail in (
        "metadata.version('aimnet')",
        '"code_license": "AIMNet software: MIT"',
        "molecular training domain; wrapper supports PBC",
        "condensed-phase accuracy is not established here",
        "one explicit total charge per graph through Batch.charge",
        "selected checkpoint is closed-shell; no multiplicity input is used",
    ):
        assert detail in model_card
    assert "uses the model attached to its first sub-stage" in fused_intro
    assert "`nvt + nve` changes the update rule, not the model" in fused_intro


def test_inflight_lesson_registers_and_validates_an_actual_refill_trace() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    intro = _source(by_id["inflight-intro"])
    setup = "\n".join(
        (
            _source(by_id["inflight-example"]),
            _source(by_id["configure-inflight-stage"]),
        )
    )
    run = _source(by_id["run-inflight-example"])

    for public_api in (
        "InMemoryDataset(",
        "SizeAwareSampler(",
        "HostMemory(",
        "FusedStage(",
    ):
        assert public_api in setup
    for setting in (
        "INFLIGHT_SYSTEMS = 2_048",
        "INFLIGHT_ACTIVE_SYSTEMS = 256",
        "INFLIGHT_NVT_STEPS = 2",
        "INFLIGHT_NVE_STEPS = 3",
        "refill_frequency=1",
        "max_edges=None",
    ):
        assert setting in setup

    trace_call = _single_call(setup, "register_inflight_trace")
    assert len(trace_call.args) == 1
    assert isinstance(trace_call.args[0], ast.Name)
    assert trace_call.args[0].id == "inflight"
    assert setup.index("inflight = FusedStage(") < setup.index(
        "inflight_trace = register_inflight_trace(inflight)"
    )

    run_call = _single_call(run, "inflight.run")
    batch_argument = _call_keyword(run_call, "batch")
    assert isinstance(batch_argument, ast.Constant) and batch_argument.value is None
    finalize_call = _single_call(run, "inflight_trace.finalize")
    completed_ids = _call_keyword(finalize_call, "completed_system_ids")
    assert isinstance(completed_ids, ast.Name) and completed_ids.id == "system_ids"
    failure_count = _call_keyword(finalize_call, "failure_count")
    assert isinstance(failure_count, ast.Name)
    assert failure_count.id == "inflight_failure_count"
    table_call = _single_call(run, "inflight_trace_table")
    assert len(table_call.args) == 1
    assert isinstance(table_call.args[0], ast.Name)
    assert table_call.args[0].id == "inflight_trace"

    for validation in (
        "torch.unique(system_ids, sorted=True, return_counts=True)",
        "torch.arange(INFLIGHT_SYSTEMS",
        "torch.equal(counts, torch.ones_like(counts))",
        'inflight_trace_rows.iloc[-1]["Active"]',
        'inflight_trace_rows.iloc[-1]["Completed"]',
        'inflight_trace_rows.iloc[-1]["Failures"]',
        "inflight_failure_count = 0",
        "== INFLIGHT_SYSTEMS",
        "readable_table(\n    inflight_trace_rows",
    ):
        assert validation in run
    assert (
        run.index("inflight.run(batch=None)")
        < run.index("inflight_trace.finalize(")
        < run.index("inflight_trace_table(")
    )

    assert "atom and structure limits only" in intro
    assert "chosen so refills are visible" in intro
    assert "not a\nmeasured GPU capacity" in intro
    assert "result hook observes the run without changing it" in " ".join(
        intro.split()
    )
    assert "actual active count" in " ".join(intro.split())
    assert "`NaNDetectorHook` stops this teaching run" in intro
    assert "zero observed failures" in intro
    assert "too short for scientific MD" in intro


def test_domain_parallel_lesson_preserves_periodic_science_and_public_api() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    intro = _source(by_id["domain-decomposition-intro"])
    model_intro = _source(by_id["domain-model-intro"])
    build = _source(by_id["build-domain-box"])
    convert = _source(by_id["convert-domain-box"])
    compose = (
        _source(by_id["configure-domain-pme"])
        + _source(by_id["compose-domain-model"])
    )
    configure = _source(by_id["configure-domain-parallel"])
    run = _source(by_id["run-domain-single-gpu"])
    molecule_charges = _source(by_id["inspect-domain-molecule-charges"])
    inspect = _source(by_id["inspect-domain-single-gpu"])

    intro_text = " ".join(intro.split())
    for scientific_boundary in (
        "phenol and N-methylacetamide",
        "independent species",
        "neutral 3,200-atom periodic box",
        "not a liquid-property calculation",
        "many independent graphs into one crowded periodic graph",
        "128 rigid copies of each",
        "Packmol placed them once",
        "it does not run here",
        "neither relaxed nor equilibrated",
        "static OVITO render",
        "loading it does not start OVITO or change coordinates",
        "integer supercells",
        "without giant live packing",
        "NCI curves do not validate the dense mixture",
        "1:1 composition and 1.0 g cm⁻³ construction density are teaching inputs",
        "AIMNet2-predicted charges with particle mesh Ewald (PME)",
        "tapered D3(BJ)",
        "halo",
        "boundary interactions are retained",
        "not a material-property result",
    ):
        assert scientific_boundary in intro_text
    model_text = " ".join(model_intro.split())
    for model_boundary in (
        "E_base = E_NN - E_Coulomb^SR",
        "E_composed = E_base + E_PME(q(R)) + E_D3",
        "Adding full PME gives `E_NN + E_Coulomb^LR` without double counting",
        "`Batch.charge = 0` sets the total charge",
        "geometry-dependent atomic charges",
        "`PMEModelWrapper` consumes those charges",
        "fixed-charge PME forces",
        "response through the predicted charges",
        "`DomainParallel` then rebuilds neighbors for each GPU region",
        "composition includes every declared energy term",
        "does not establish accuracy for this mixture",
    ):
        assert model_boundary in model_text

    for construction in (
        "PREBUILT_DOMAIN_BOX_DIR",
        "load_prebuilt_domain_box(",
        "domain_plan = domain_box.plan",
        "domain_atoms = domain_box.atoms",
        "box_summary_table(domain_plan, domain_atoms)",
        "plt.imread(domain_box.preview_path)",
        "figure_with_alt(",
        "AtomicData.from_atoms(",
        "Batch.from_data_list([domain_data], device=DEVICE)",
        "domain_batch.num_graphs == 1",
        "float(domain_batch.charge.item()) == 0.0",
    ):
        assert construction in build + convert
    for forbidden_live_setup in (
        "which(\"packmol\")",
        "plan_nci_molecular_box(",
        "build_nci_molecular_box(",
        "make_ovito_widget(",
    ):
        assert forbidden_live_setup not in build

    for model_call in (
        "estimate_pme_parameters(",
        "PMEModelWrapper(",
        "hybrid_forces=True",
        "PipelineStep(model=nci_aimnet)",
        "PipelineStep(model=periodic_pme)",
        "use_autograd=True",
        "PipelineStep(model=nci_d3)",
        "use_autograd=False",
        'neighbor_adaptation="never"',
        'periodic_model.set_config("active_outputs", {"energy", "forces", "charges"})',
    ):
        assert model_call in compose
    assert "DomainConfig(" in configure
    assert "mesh=None" in configure
    assert "compile=False" in configure

    public_sequence = (
        "DomainParallel(",
        "with domain_dynamics as domain_run:",
        "domain_run.partition(domain_batch)",
        "domain_run.run(domain_local)",
        "domain_run.gather(domain_local, dst=0)",
    )
    offsets = [run.index(token) for token in public_sequence]
    assert offsets == sorted(offsets)
    assert "periodic_model.make_neighbor_hooks()" in run
    assert "NaNDetectorHook(frequency=1)" in run
    molecule_charge_text = " ".join(molecule_charges.split())
    for charge_diagnostic in (
        'domain_atoms.arrays["molecule_id"]',
        "segmented_sum(",
        "domain_plan.molecule_count",
        "molecule_charge_tables(",
        "Predicted molecular charge sums",
        "Most negative and positive molecular charge sums",
        "Only the total box charge is constrained",
        "model-dependent sums",
        "not validated intermolecular charge transfer",
        "all 256",
        "values are saved",
        "dtype=torch.int32",
    ):
        assert charge_diagnostic in molecule_charge_text

    api_preview = " ".join(_source(by_id["domain-parallel-api"]).split())
    for boundary in (
        "total energy is summed across GPUs",
        "`gather` collects atom-level fields on rank 0",
        "process mesh | assigns one worker process to each GPU",
        "spatial grid | assigns atoms to GPU regions",
        "PME grid | defines the electrostatics FFT repeated on every GPU",
        "restricts this example to a neutral box",
        "`gather` reconstructs declared atom fields such as forces",
        "does not emit predicted atomic charges",
        "rebuilds neighbors inside each region",
        "one-GPU path uses the model's ordinary neighbor hooks",
    ):
        assert boundary in api_preview

    inspect_text = " ".join(inspect.split())
    for result_check in (
        "np.isfinite([domain_energy_ev, domain_fmax_ev_a, domain_charge_sum])",
        "abs(domain_charge_sum) <= DOMAIN_CHARGE_SUM_TOLERANCE_E",
        "assert domain_live_api_passed",
        '("spatially decomposed", False)',
        "raw model energy / atom for this fixed input",
        "not domain decomposition, a speedup, or a capacity measurement",
        "without changing its coordinates",
        "not an interaction, cohesive, or liquid-property",
        "would need relaxation or",
        "equilibration before dynamics",
    ):
        assert result_check in inspect_text


def test_domain_results_require_same_input_agreement_and_stable_timings() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    methodology = (
        _source(by_id["display-domain-methodology"])
        + _source(by_id["display-domain-scaling-methodology"])
    )
    plan = _source(by_id["domain-scaling-plan"])
    loader = _source(by_id["domain-parallel-results"])
    display_results = _source(by_id["display-domain-parallel-results"])

    for methodology_field in (
        "steady_timing_warmup_count",
        "steady_timing_sample_count",
        "steady_timing_model_evaluations_per_workflow",
        "steady_timing_max_relative_iqr",
        "charge_sum_tolerance_e",
        "parity_energy_tolerance_ev_per_atom",
        "parity_force_atol_ev_a",
        "parity_force_rtol",
    ):
        assert methodology_field in methodology

    plan_text = " ".join(plan.split())
    for safeguard in (
        "integer supercells",
        "Packmol is not rerun",
        "three different questions",
        "notebook never triggers an out-of-memory failure live",
        "first natural CUDA OOM",
        "exact first-OOM input, unchanged",
        "separate input that already fits one H100",
        "Nodes = ranks = GPUs",
        "Four GPUs means four nodes, each with one worker and one H100",
        "1-GPU forces against 2/4 GPUs",
        "2-GPU distributed energy against 4",
        "raw 1-to-multi energy offset is diagnostic",
        "slowest-rank median and interquartile range (IQR)",
        "first OOM and its unchanged retries measure capacity",
        "separate one-GPU-fit input measures speed",
        "Missing files produce `NOT REPORTED`, never an estimate",
    ):
        assert safeguard in plan_text

    for loader_check in (
        "load_domain_lesson_view(",
        "planned_atom_counts=DOMAIN_PLANNED_ATOM_COUNTS",
        "expected_parity_atom_count=DOMAIN_PARITY_ATOM_COUNT",
        "if not domain_view.available:",
        "NOT REPORTED:",
    ):
        assert loader_check in loader
    assert "if domain_view.available:" in display_results
    for split_check in (
        'domain_takeaway["all_one_gpu_force_checks_passed"]',
        'domain_takeaway["all_distributed_energy_checks_passed"]',
        'domain_takeaway["timed_one_gpu_force_checks_passed"]',
        'domain_takeaway["timed_distributed_energy_checks_passed"]',
    ):
        assert split_check in display_results
    assert "only as a diagnostic" in display_results
    assert "speedup_by_gpu" in display_results
    for lesson_result in (
        "One-H100 capacity, including the first natural OOM",
        '"measurement_role"].eq("rescue")',
        "The same first-OOM input on 2 and 4 H100s",
        '"measurement_role"].eq("steady_timing")',
        "A separate one-H100-fit input on 1, 2, and 4 H100s",
    ):
        assert lesson_result in display_results
    assert 'result_state="not_reported"' in display_results


def test_distributed_pipeline_stays_an_unlaunched_preview_with_safe_reporting() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    preview = _source(by_id["distributed-pipeline-intro"])
    result_cell = _source(by_id["pipeline-campaign-results"])

    for api_token in (
        "DistributedPipeline",
        "BufferConfig(",
        "fixed_optimization = FIRE2(",
        "n_steps=optimization_steps",
        "sub_stages=[(0, fixed_optimization)]",
        "stages={0: optimization, 1: dynamics}",
        "synchronized=False",
        'backend="nccl"',
        "with pipeline:",
        "pipeline.run()",
    ):
        assert api_token in preview
    assert "API preview, not a performance result" in preview
    assert "Toolkit 0.2 does not yet preserve every field" in preview
    assert "A **rank** is one worker process" in preview
    assert '`comm_mode="async_recv"`' in preview
    assert "A plain `FIRE2(n_steps=...)` pipeline stage" in preview
    assert "pipeline correctness, overlap, and speed" in preview
    assert "NOT REPORTED" in preview

    executable_source = _code_source()
    for forbidden_launcher in (
        "DistributedPipeline(",
        "BufferConfig(",
        "init_process_group(",
        "pipeline.run()",
        "torchrun",
        "sbatch",
    ):
        assert forbidden_launcher not in executable_source

    for reporting_check in (
        "DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON",
        "NOT REPORTED",
        'result_state="not_reported"',
    ):
        assert reporting_check in result_cell
    for forbidden_recorded_path in (
        "load_pipeline_campaign_lesson_view(",
        "campaign_view",
        "plot_pipeline_campaign(",
        "RECORDED:",
    ):
        assert forbidden_recorded_path not in result_cell
    assert result_cell.count("NOT REPORTED") == 1


def test_ir_endpoint_separates_md_dft_and_observed_fundamentals() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    reference_note = _source(by_id["reference-note"])
    plot_source = _source(by_id["plot"])
    all_source = "\n".join(_source(cell) for cell in cells)

    assert "three comparisons" in reference_note.casefold()
    assert "selected gas-phase band positions" in reference_note
    assert "no experimental intensity curve" in reference_note
    assert "plot_monomer_ir_comparison(" in plot_source
    assert "experimental_fundamentals" in plot_source
    assert "MD_minus_DFT" not in all_source
    references = _source(by_id["references"])
    assert "NCI Atlas" in references
    assert "Dinu et al., Table 1" in references
    assert "CC BY 4.0" in references
    assert "10.1006/jmsp.1998.7771" in references
    assert "10.1006/jmsp.1998.7611" in references
    assert "10.1006/jmsp.1999.7815" in references
    assert "[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)" in references
    source_links = _source(by_id["observed-source-links"])
    assert "Dinu et al., Table 1" in source_links
    assert "H₂-¹⁶O and D₂-¹⁶O" in source_links
    assert "Toth's" in source_links
    assert "Positions only" in source_links

    harmonic_intro = _source(by_id["harmonic-intro"])
    reference_preview = _source(by_id["reference-preview"])
    interpretation = _source(by_id["interpretation"])
    comparison_text = "\n".join((harmonic_intro, reference_preview, interpretation))
    assert "separately optimized minima" in comparison_text
    assert "trained toward B97-3c data" in comparison_text
    assert "not an independent validation of the training domain" in comparison_text
    assert "in-domain numerical comparison" not in all_source
    assert "full B97-3c endpoint on the same geometries" not in all_source


def test_final_cell_asks_the_learner_to_transfer_the_workflow() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    interpretation = _source(by_id["interpretation"])

    for term in (
        "Try it: decide whether a new workload is many independent systems",
        "choose batching or domain decomposition",
        "Prepare and check any new base box offline",
        "Charged periodic systems and multi-stage multi-GPU execution are left for later",
        "The Toolkit 0.2 API shape is introduced here",
        "correctness, overlap, and speed remain `NOT REPORTED`",
        "[Part 2: batched adsorption]",
        "../part-2-batched-adsorption-toolkit/README.md",
    ):
        assert term in interpretation


def test_harmonic_frequencies_name_the_complete_water_model() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    comparison = _source(by_id["harmonic-comparison"])
    intro = _source(by_id["harmonic-intro"])
    finite_difference = _source(by_id["harmonic-finite-difference"])
    workflow_helper = (PART_DIR / "aux" / "harmonic_workflow.py").read_text(
        encoding="utf-8"
    )

    assert "build_harmonic_mode_comparison_table(" in comparison
    assert "empty_harmonic_mode_comparison_table()" in comparison
    assert "harmonic_frequency_mae_cm1 = None" in comparison
    assert comparison.index("if harmonic_comparison_reported:") < comparison.index(
        "build_harmonic_mode_comparison_table("
    )
    assert comparison.index("if harmonic_comparison_reported:") < comparison.index(
        "display(readable_table("
    )
    assert "AIMNet+Coulomb+D3_harmonic_cm-1" in workflow_helper
    assert "AIMNet+Coulomb+D3_minus_B97-3c_cm-1" in comparison
    assert "AIMNet_harmonic_cm-1" not in comparison + workflow_helper
    assert "AIMNet_minus_B97-3c_cm-1" not in comparison + workflow_helper
    assert "complete AIMNet + Coulomb + D3 model" in intro
    assert "predicted point-charge dipole response" in intro
    assert "collect_harmonic_displacement_result(" in finite_difference
    assert "dipole_origin_atom_index=0" in finite_difference
    assert "HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E" in finite_difference
    assert "plus_positions = stored_positions[:n_coordinates]" in workflow_helper
    assert "minus_positions = stored_positions[n_coordinates:]" in workflow_helper


def test_stage_3_runs_the_90_graph_nci_composition_and_reference_check() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    stage = _source(by_id["stage-3"])
    load = _source(by_id["load-nci-atlas"])
    configure = _source(by_id["configure-nci-model"])
    evaluate = _source(by_id["evaluate-nci-components"])
    composition_context = _source(by_id["nci-composition-context"])
    compose = _source(by_id["compose-nci-pipeline"])
    graph_order = _source(by_id["validate-nci-graph-order"])
    force = _source(by_id["check-nci-force"])
    reference_context = _source(by_id["nci-reference-context"])
    analyze = _source(by_id["analyze-nci-curves"])
    display_analysis = _source(by_id["display-nci-curves"])

    for term in (
        "**NCI** means noncovalent interaction",
        "E(AB) - E(A) - E(B)",
        "90-graph batch",
        "`R/Rₑ`",
        "four combinations",
        "**checkpoint base:**",
        "**base + D3:**",
        "**base + full Coulomb:**",
        "**complete model:**",
        "E_base = E_NN - E_Coulomb^SR",
        "E_complete = E_base + E_Coulomb^full + E_D3",
    ):
        assert term in stage

    for term in (
        "`use_autograd=True`",
        "direct all-pairs Coulomb interactions",
        "Ewald and PME are periodic methods",
        "checks, not separate production models",
        "neighbor_adaptation=",
        "always",
    ):
        assert term in composition_context

    for term in (
        "`aimnet2-wb97m-d3_*` ensemble",
        "training level is ωB97M-D3/def2-TZVPP",
        "ωB97M-D3(BJ)/def2-TZVPPD",
        "ensemble mean and member-to-member spread",
        "not calibrated uncertainty",
        "DFT-D3 and CCSD(T)/CBS",
        "0.5 kcal/mol check",
        "not a general accuracy guarantee",
    ):
        assert term in reference_context

    for term in (
        "load_nci_atlas_subset(NCI_DATA_FILE)",
        "rows_to_atoms(nci_reference_data)",
        "build_graph_index(nci_reference_data)",
        "AtomicData.from_atoms(",
        "Batch.from_data_list(",
        "nci_batch.num_graphs == 90",
        "3 systems × 10 separations × AB/A/B",
    ):
        assert term in load

    for term in (
        'NCI_CHECKPOINTS = [f"aimnet2-wb97m-d3_{index}" for index in range(4)]',
        'nci_metadata["needs_coulomb"] is True',
        'nci_metadata["needs_dispersion"] is True',
        'nci_metadata["coulomb_mode"] == "sr_embedded"',
        ("NCI_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction"),
        "DFTD3ModelWrapper(",
        "param_file=D3_PARAMETER_FILE",
        "auto_download=True",
        "D3_PARAMETER_SHA256 = sha256_file(D3_PARAMETER_FILE)",
        "assert D3_PARAMETER_SHA256 == EXPECTED_D3_PARAMETER_SHA256",
        "DirectCoulombWrapper()",
        "nci_validation_settings_table(NCI_VALIDATION)",
    ):
        assert term in configure

    for term in (
        'title="Evaluate the NCI set in nine batched calls", total=9',
        "one shared D3 pass over 90 graphs",
        "compute_neighbors(",
        "segmented_sum(",
        "member_batch.charge.reshape(-1),",
        "NCI_VALIDATION.charge_atol_e",
        "for member_index, checkpoint in enumerate(NCI_CHECKPOINTS)",
        'message=f"AIMNet ensemble member {member_index}"',
        'message=f"Coulomb from member {member_index} charges"',
        "nci_member_residual_eV.shape == nci_member_coulomb_eV.shape == (4, 90)",
        "four AIMNet, four Coulomb, and one D3 call complete",
    ):
        assert term in evaluate

    for term in (
        "PipelineModelWrapper(",
        "PipelineGroup(",
        "PipelineStep(model=nci_aimnet)",
        "PipelineStep(model=nci_coulomb)",
        "PipelineStep(model=nci_d3)",
        'neighbor_adaptation="always"',
        "nci_member_residual_eV[0] + nci_member_coulomb_eV[0]",
    ):
        assert term in compose

    for term in (
        "nci_batch_template.index_select(reverse_order)",
        "nci_reversed_energy",
        "nci_graph_order_max_abs_eV",
    ):
        assert term in graph_order

    force_helper = (PART_DIR / "aux" / "nci_validation.py").read_text(encoding="utf-8")
    force_implementation = force + "\n" + force_helper
    for term in (
        "check_nci_force(",
        "net_force = toolkit_forces.sum(dim=0)",
        "AIMNet2Calculator(",
        'nci_official.set_lrcoulomb_method("simple")',
        "smoothing_fraction=NCI_D3_SMOOTHING_FRACTION",
        'base["forces"].abs().reshape(-1).argmax()',
        "dtype=torch.float64",
        "finite_difference_force = -(",
        "toolkit_force",
        "official_force",
        "settings.toolkit_official_force_atol_eV_A",
        "build_nci_force_check_table(",
    ):
        assert term in force_implementation
    assert "NCI_FD_STEP_A" not in force
    assert "displaced_output" not in force

    analysis_helper = (PART_DIR / "aux" / "nci_atlas.py").read_text(encoding="utf-8")
    analysis_implementation = analyze + "\n" + display_analysis + "\n" + analysis_helper
    for term in (
        "reduce_fragment_energies(",
        '"core_plus_d3"',
        '"core_plus_coulomb"',
        'dft["dft_no_d3"] = dft["dft_full"] -',
        '"complete vs CC": ("full", "ccsd_t_cbs")',
        '"complete vs DFT-D3": ("full", "dft_full")',
        '"same-D3 bookkeeping identity"',
        'nci_metrics["complete vs CC"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL',
        'nci_metrics["complete vs DFT-D3"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL',
        "plot_nci_interaction_curves(nci_curves)",
    ):
        assert term in analysis_implementation

    order = [cell.get("id") for cell in _notebook()["cells"]]
    assert order.index("evaluate-nci-components") < order.index(
        "nci-composition-context"
    )
    assert order.index("nci-composition-context") < order.index("compose-nci-pipeline")
    assert order.index("check-nci-force") < order.index("nci-reference-context")
    assert order.index("nci-reference-context") < order.index("analyze-nci-curves")
    assert order.index("analyze-nci-curves") < order.index("display-nci-curves")


def test_nci_accuracy_limit_is_imported_once_and_applied_to_both_references() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    helper_imports = _source(by_id["helper-imports"])
    configure = _source(by_id["configure-nci-model"])
    analyze = _source(by_id["analyze-nci-curves"])

    nci_import = re.search(
        r"from aux\.nci_config import \((?P<names>.*?)\)",
        helper_imports,
        flags=re.DOTALL,
    )
    assert nci_import is not None
    assert "NCI_COMPLETE_MAE_LIMIT_KCAL_MOL" in nci_import.group("names")
    assert "NCI_COMPLETE_MAE_LIMIT_KCAL_MOL =" not in configure
    assert analyze.count("NCI_COMPLETE_MAE_LIMIT_KCAL_MOL") == 2
    for comparison in ("complete vs CC", "complete vs DFT-D3"):
        assert (
            f'nci_metrics["{comparison}"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL' in analyze
        )


def test_all_generated_plot_outputs_have_descriptive_alt_text() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    expected = {
        "build-domain-box": "Static OVITO rendering of the checked periodic base box",
        "display-cpu-gpu-sweep": "CPU and GPU throughput versus batch size",
        "display-nci-curves": "Interaction-energy curves for neutral hydrogen bonding",
        "display-domain-parallel-results": "Two-panel H100 domain-decomposition result",
        "dimer-ablation-plot": "Water-dimer interaction energy versus O-O distance",
        "harmonic-comparison": "Side-by-side H2O and D2O harmonic frequency comparisons",
        "topology-timeline": "Two stacked 10 ps time series",
        "plot": "Side-by-side H2O and D2O IR panels",
    }
    source = _code_source()
    assert "plt.show(" not in source
    assert source.count("figure_with_alt(") == len(expected)
    for cell_id, description in expected.items():
        cell_source = _source(by_id[cell_id])
        assert cell_source.count("figure_with_alt(") == 1
        assert description in cell_source

    preview = _source(by_id["reference-preview"])
    assert (
        "![B97-3c harmonic IR reference for H2O, D2O, cyclic (H2O)6, "
        "and cyclic (D2O)6.]"
    ) in preview


def test_learner_facing_dataframes_use_the_shared_table_style() -> None:
    source = _code_source()
    direct_dataframe_displays = (
        "display(pd.Series(",
        "display(pd.DataFrame(",
        "display(composition_check_table)",
        "display(dimer_table",
        "display(ablation_mae",
        "display(compiled_ir_checks)",
        "display(harmonic_fd_table)",
        "display(harmonic_validation_table)",
        "display(harmonic_convergence_table",
        "display(harmonic_comparison_table",
        "display(experimental_fundamentals",
        "display(diagnostic_table",
        "display(integrity_table)",
        "display(metrics.",
        "display(reference_metrics.",
        "display(monomer_mode_map",
        "display(results_summary)",
    )
    assert not [token for token in direct_dataframe_displays if token in source]
    assert source.count("display(readable_table(") >= 30


def test_nci_partial_models_and_reference_levels_stay_distinct() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    stage = _source(by_id["stage-3"])
    composition = _source(by_id["nci-composition-context"])
    reference = _source(by_id["nci-reference-context"])
    analyze = _source(by_id["analyze-nci-curves"])

    assert "partial combinations expose omitted terms" in composition
    assert "not separate production models" in composition
    assert "one prescribed net charge per graph in `Batch.charge`" in stage
    assert "per-atom `charges`" in stage
    assert "their sum is checked" in stage
    assert "ensemble mean and member-to-member spread" in reference
    assert "not calibrated uncertainty" in reference
    assert "training level is ωB97M-D3/def2-TZVPP" in reference
    assert "training level is ωB97M-D3/def2-TZVP;" not in reference
    assert '"same-D3 bookkeeping identity"' in analyze
    assert '"complete vs DFT-D3": ("full", "dft_full")' in analyze
    assert '"complete vs CC": ("full", "ccsd_t_cbs")' in analyze
    assert "bookkeeping, not accuracy" in _source(
        by_id["display-nci-curves"]
    )
    all_source = "\n".join(_source(cell) for cell in _notebook()["cells"])
    assert "learned short-range output" not in all_source
    assert "AIMNet residual" not in all_source


def test_notebook_connects_nci_custom_model_and_dynamics() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}

    stage_3 = _source(by_id["stage-3"])
    reference_context = _source(by_id["nci-reference-context"])
    for term in (
        "NCI** means noncovalent interaction",
        "neutral hydrogen bonding",
        "dispersion-dominated binding",
        "ionic hydrogen bond",
        "one 90-graph batch",
        "one shared D3 pass",
    ):
        assert term in stage_3
    assert "four-member `aimnet2-wb97m-d3_*` ensemble" in reference_context

    stage_4 = "\n".join(
        (
            _source(by_id["stage-4"]),
            _source(by_id["surface-model-switch"]),
        )
    )
    for term in (
        "Bring a model for a new domain",
        "AIMNet2 checkpoint cannot represent Cu",
        "model limit, not a Toolkit limit",
        "AtomicData",
        "Batch",
        "`BaseModelMixin` is Toolkit's base class for a custom model adapter",
        "energies, and forces",
    ):
        assert term in stage_4

    stage_5 = _source(by_id["stage-5"])
    for term in (
        "charge-predicting molecular model",
        "Reusable Toolkit dynamics path",
        "complete model",
        "four-system Batch",
        "FIRE2 relaxation",
        "NVT + NVE",
        "hooks + saved state",
    ):
        assert term in stage_5

    ir_batch = _source(by_id["build-ir-batch"])
    assert (
        "AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)" in ir_batch
    )
    assert "torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)" in ir_batch
    assert "for tensor in (batch.positions, batch.velocities, batch.forces)" in ir_batch
    assert "dtype=torch.float64" not in ir_batch


def test_notebook_uses_one_compact_reporting_helper_and_safe_display() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    helper_imports = _source(by_id["helper-imports"])
    summary = _source(by_id["results-summary"])
    save = _source(by_id["save"])

    assert "from aux.notebook_reporting import build_part1_notebook_report" in (
        helper_imports
    )
    report_call = _single_call(summary, "build_part1_notebook_report")
    assert len(report_call.args) == 1
    assert isinstance(report_call.args[0], ast.Call)
    assert _call_name(report_call.args[0]) == "globals"
    assert summary.count("build_part1_notebook_report(globals())") == 1
    for field in (
        "results_summary",
        "not_reported_count",
        "water_run_results",
        "manifest_input",
    ):
        assert f"{field} = notebook_report.{field}" in summary
    assert "NotebookProgress(" in summary

    reporting_helper = (PART_DIR / "aux" / "notebook_reporting.py").read_text(
        encoding="utf-8"
    )
    for derivation in (
        'values["serial_batch_error"] < values["RESIDUAL_SERIAL_BATCH_TOLERANCE_EV"]',
        "campaign_available=False",
        'comparisons.loc["H2O_over_D2O_centroid", "reported"]',
        'nci_metrics["complete vs DFT-D3"].max()',
        'nci_metrics["complete vs CC"].max()',
        "domain_view.successful_case_count",
    ):
        assert derivation in reporting_helper

    assert (
        by_id["results-summary"]
        .get("metadata", {})
        .get("jupyter", {})
        .get("source_hidden")
        is True
    )

    executable_source = _code_source()
    for expanded_constructor in (
        "build_results_summary(",
        "WaterRunResults(",
        "WaterRunManifestInput.from_sections(",
    ):
        assert expanded_constructor not in executable_source
    for save_call in (
        "save_water_run_outputs(",
        "results=water_run_results",
        "**manifest_input.as_save_arguments()",
    ):
        assert save_call in save

    summary_display = _source(by_id["display-results-summary"])
    assert "NotebookProgress(" in summary_display
    assert "readable_table(" in summary_display
    assert 'label="Part 1 results summary"' in summary_display
    assert 'missing="NOT REPORTED"' in summary_display
    assert "display(results_summary)" not in summary
    assert "NO RESULTS" not in summary
    assert "claim_ledger" not in summary
    assert "evidence_state" not in summary
    assert '"boundary"' not in summary

    spectrum = _source(by_id["spectrum"])
    assert "readable_table(" in spectrum
    assert 'label="IR comparison availability"' in spectrum
    assert 'missing="NOT REPORTED"' in spectrum
    assert "display(comparison_display_table" not in spectrum
    assert "Other values are labeled NOT REPORTED" in spectrum

    summary_note = _source(by_id["results-summary-note"])
    assert "live one-GPU `DomainParallel` row records the API call" in summary_note
    assert "one GPU means one domain" in summary_note
    assert "it does not claim spatial decomposition" in summary_note
    assert "recovery of the exact first-OOM input" in summary_note
    assert "separate one-GPU-fit input are **RECORDED**" in summary_note
    assert (
        "`DistributedPipeline`\ncorrectness, overlap, and timing are "
        "**NOT REPORTED**" in summary_note
    )
    assert "**RECORDED**" in summary_note
    assert "**NOT REPORTED**" in summary_note

    mode_mapping = _source(by_id["mode-mapping"])
    assert "monomer_mode_mapping_display_table(mode_mapping)" in mode_mapping
    assert "readable_table(" in mode_mapping
    assert 'label="H2O to D2O monomer mode mapping"' in mode_mapping
    assert "display(mode_map_table.round(" not in mode_mapping


def test_aux_does_not_hide_toolkit_batch_or_pipeline_construction() -> None:
    forbidden_call_tails = {
        "AtomicData.from_atoms",
        "Batch.from_data_list",
        "compute_neighbors",
        "PipelineGroup",
        "PipelineModelWrapper",
    }
    violations: list[str] = []
    allowed_aux_calls = {
        ("aux/runtime.py", "Batch.from_data_list"),
        # The inflight helper prepares repeated starting structures. The notebook
        # teaches both calls directly in build-ir-batch before using this helper.
        ("aux/inflight.py", "AtomicData.from_atoms"),
        ("aux/inflight.py", "Batch.from_data_list"),
        # The NCI helper owns an independent calculator validation route;
        # the primary Toolkit model conversion remains visible in Stage 3.
        ("aux/nci_validation.py", "AtomicData.from_atoms"),
        ("aux/nci_validation.py", "Batch.from_data_list"),
    }

    for path in sorted((PART_DIR / "aux").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "nvalchemi"
            ):
                imported_aliases.update(
                    {alias.asname or alias.name: alias.name for alias in node.names}
                )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            root, _, remainder = name.partition(".")
            canonical = ".".join(
                part for part in (imported_aliases.get(root, root), remainder) if part
            )
            if any(
                canonical == tail or canonical.endswith(f".{tail}")
                for tail in forbidden_call_tails
            ):
                relative_path = path.relative_to(PART_DIR).as_posix()
                if (relative_path, canonical) in allowed_aux_calls:
                    continue
                violations.append(f"{relative_path}:{node.lineno}: {name}")

    assert not violations, (
        "Toolkit conversion, batching, neighbors, and pipeline assembly must be "
        "visible in the notebook, not hidden in aux:\n" + "\n".join(violations)
    )


def test_aux_package_root_has_no_reexport_api() -> None:
    tree = ast.parse((PART_DIR / "aux" / "__init__.py").read_text(encoding="utf-8"))
    exported = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "__all__" and node.value is not None:
            exported = ast.literal_eval(node.value)

    assert exported == (), "aux/__init__.py must not become a competing public API"


def test_notebook_declares_full_h100_workload_without_short_run_controls() -> None:
    source = _code_source()
    tree = _parse_code(source)
    assigned_literals: dict[str, object] = {}
    identifiers: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned_literals[target.id] = literal

    assert assigned_literals.get("WARMUP_STEPS") == 5_000
    assert assigned_literals.get("PRODUCTION_STEPS") == 20_000
    assert assigned_literals.get("DT_FS") == 0.5
    assert assigned_literals.get("AIMNET_MATMUL_PRECISION") == "highest"
    assert "torch.set_float32_matmul_precision(AIMNET_MATMUL_PRECISION)" in source
    assert "torch.get_float32_matmul_precision() == AIMNET_MATMUL_PRECISION" in source
    assert "final_batch = dynamics.run(batch)" in source
    assert "dynamics.run(batch, n_steps=" not in source
    assert 'converge_after_steps("nvt_steps_done", WARMUP_STEPS)' in source
    assert 'converge_after_steps("nve_steps_done", PRODUCTION_STEPS)' in source
    assert "dynamics.step_count == TOTAL_DYNAMICS_STEPS" in source
    assert "shape == (PRODUCTION_STEPS, 4, 3)" in source
    assert "shape == (PRODUCTION_STEPS, 42, 3)" in source

    production_steps = int(assigned_literals["PRODUCTION_STEPS"])
    dt_fs = float(assigned_literals["DT_FS"])
    current_frames = production_steps - 1
    frames_per_window = int(round(5_000.0 / dt_fs))
    window_step = frames_per_window // 2
    window_starts = range(
        0,
        current_frames - frames_per_window + 1,
        window_step,
    )
    assert production_steps * dt_fs / 1_000 == 10.0
    assert len(window_starts) == 2
    stage_six = next(
        _source(cell) for cell in _notebook()["cells"] if cell.get("id") == "stage-6"
    )
    assert "two overlapping 5 ps windows" in " ".join(stage_six.casefold().split())
    assert "not a trajectory-length convergence study" in stage_six

    short_run_control = re.compile(
        r"(?:fast|quick|smoke|reduced|short|demo|local|ci)_"
        r"(?:mode|steps?|run|limit|cutoff|batch)|"
        r"(?:frame|step|memory|gpu)_(?:limit|guard)",
        flags=re.IGNORECASE,
    )
    forbidden = sorted(
        name for name in identifiers if short_run_control.fullmatch(name)
    )
    assert not forbidden, f"artificial reduced-run controls found: {forbidden}"


def test_notebook_helpers_are_moved_to_aux() -> None:
    violations: list[str] = []

    class DefinitionVisitor(ast.NodeVisitor):
        def __init__(self, cell_id: str) -> None:
            self.cell_id = cell_id

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            if (
                self.cell_id == "define-sevennet-wrapper"
                and node.name == "SevenNetOmniWrapper"
            ):
                return
            violations.append(f"{self.cell_id}:{node.lineno} class {node.name}")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if (
                self.cell_id == "define-sevennet-config"
                and node.name == "make_sevennet_model_config"
            ):
                return
            if self.cell_id == "setup" and node.name == "locate_part_directory":
                # This hidden bootstrap must find aux before aux can be imported.
                return
            line_count = (node.end_lineno or node.lineno) - node.lineno + 1
            if line_count > 8:
                violations.append(
                    f"{self.cell_id}:{node.lineno} {node.name} ({line_count} lines)"
                )
            self.generic_visit(node)

    for cell in _notebook()["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell_id = str(cell.get("id", "<no-id>"))
        DefinitionVisitor(cell_id).visit(_parse_code(_source(cell)))

    assert not violations, (
        "move substantial notebook helpers into a focused aux module; keep "
        "learner-facing cells focused on public Toolkit calls:\n"
        + "\n".join(violations)
    )


def test_rendered_sevennet_wrapper_has_no_unbound_private_module_names() -> None:
    """Private adapter machinery must come from the collapsed helper cell."""

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    tree = _parse_code(_source(by_id["define-sevennet-wrapper"]))
    helper_tree = _parse_code(_source(by_id["helper-imports"]))
    helper_imported: set[str] = set()
    wrapper: ast.ClassDef | None = None

    for node in helper_tree.body:
        if isinstance(node, ast.ImportFrom):
            helper_imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            helper_imported.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SevenNetOmniWrapper":
            wrapper = node

    assert wrapper is not None
    private_loads = {
        node.id
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id.startswith("_")
        and not node.id.startswith("__")
    }
    assert private_loads <= helper_imported, (
        "the rendered SevenNet wrapper refers to private module names that the "
        f"collapsed helper cell does not import: {sorted(private_loads - helper_imported)}"
    )
    assert (
        by_id["helper-imports"]
        .get("metadata", {})
        .get("jupyter", {})
        .get("source_hidden")
        is True
    )


def test_rendered_sevennet_wrapper_cell_executes_in_fresh_notebook_scope() -> None:
    """Execute the exact copied class cell with its preceding public imports."""

    from collections.abc import Mapping
    from typing import Any

    import torch
    from nvalchemi.data import Batch
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
        NeighborConfig,
        NeighborListFormat,
    )
    from torch import nn

    class ProgressStub:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def advance(self, **_kwargs: Any) -> None:
            pass

        def complete(self, _message: str) -> None:
            pass

    if str(PART_DIR) not in sys.path:
        sys.path.insert(0, str(PART_DIR))
    from aux.models.sevennet import (
        _SevenNetAdapterBase,
        _map_sevennet_outputs,
        _model_device_and_dtype,
        _toolkit_batch_to_sevennet_graph,
    )

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    namespace = {
        "Any": Any,
        "BaseModelMixin": BaseModelMixin,
        "Batch": Batch,
        "Mapping": Mapping,
        "ModelConfig": ModelConfig,
        "NeighborConfig": NeighborConfig,
        "NeighborListFormat": NeighborListFormat,
        "NotebookProgress": ProgressStub,
        "_SevenNetAdapterBase": _SevenNetAdapterBase,
        "_map_sevennet_outputs": _map_sevennet_outputs,
        "_model_device_and_dtype": _model_device_and_dtype,
        "_toolkit_batch_to_sevennet_graph": _toolkit_batch_to_sevennet_graph,
        "callout": lambda message, **_kwargs: message,
        "display": lambda *_args, **_kwargs: None,
        "nn": nn,
        "torch": torch,
    }

    exec(
        compile(
            _source(by_id["define-sevennet-config"]),
            "define-sevennet-config",
            "exec",
        ),
        namespace,
    )
    exec(
        compile(
            _source(by_id["define-sevennet-wrapper"]),
            "define-sevennet-wrapper",
            "exec",
        ),
        namespace,
    )

    assert issubclass(namespace["SevenNetOmniWrapper"], nn.Module)


def test_fresh_kernel_has_no_undefined_global_names() -> None:
    """Model top-to-bottom execution rather than a stateful authoring kernel."""

    transformed = "\n\n".join(
        ast.unparse(_parse_code(_source(cell)))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )
    root = symtable.symtable(transformed, str(NOTEBOOK_PATH), "exec")
    module_definitions = {
        name
        for name in root.get_identifiers()
        if (
            root.lookup(name).is_assigned()
            or root.lookup(name).is_imported()
            or root.lookup(name).is_namespace()
        )
    }
    allowed = set(dir(builtins)) | {"get_ipython"}
    unresolved: set[str] = set()
    pending = [root]
    while pending:
        table = pending.pop()
        pending.extend(table.get_children())
        for name in table.get_identifiers():
            symbol = table.lookup(name)
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and name not in module_definitions
                and name not in allowed
            ):
                unresolved.add(name)

    assert not unresolved, f"undefined names in a fresh kernel: {sorted(unresolved)}"
