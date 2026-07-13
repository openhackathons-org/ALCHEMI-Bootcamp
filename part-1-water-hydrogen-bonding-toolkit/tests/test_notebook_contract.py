"""Static learner-facing contract for the Part 1 notebook.

These checks deliberately inspect the rendered notebook source, rather than the
generator, because the ``.ipynb`` file is the artifact learners open.  Runtime
and scientific validation remain separate concerns.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


PART_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PART_DIR / "alchemi-water-ir.ipynb"


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


def test_notebook_has_six_stylized_sequential_stage_cards() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    expected_ids = [f"stage-{stage}" for stage in range(1, 7)]
    missing = [cell_id for cell_id in expected_ids if cell_id not in by_id]
    assert not missing, f"missing learner stage card cells: {missing}"

    stage_card_markers = ('role="progressbar"', 'aria-label="Notebook stage"')
    rendered_stage_cards = sum(
        all(marker in _source(cell) for marker in stage_card_markers)
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )
    assert rendered_stage_cards == 6, (
        "the learner notebook must render exactly six top-level stage cards; "
        f"found {rendered_stage_cards}"
    )

    for stage, cell_id in enumerate(expected_ids, start=1):
        cell = by_id[cell_id]
        source = _source(cell)
        assert cell.get("cell_type") == "markdown"
        required_fragments = (
            'role="region"',
            f'aria-labelledby="alchemi-stage-{stage}-heading"',
            *stage_card_markers,
            'aria-valuemin="1"',
            'aria-valuemax="6"',
            f'aria-valuenow="{stage}"',
            f"STAGE {stage} OF 6",
            "Outcome:",
            "background:#76B900",
            f'<h2 id="alchemi-stage-{stage}-heading"',
        )
        absent = [
            fragment for fragment in required_fragments if fragment not in source
        ]
        assert not absent, (
            f"{cell_id} is not a complete stylized progress card: {absent}"
        )
        assert source.count("<h2 ") == 1, f"{cell_id} duplicates its stage heading"
        assert f"## Stage {stage}" not in source


def test_notebook_hero_and_presentation_blocks_share_one_visual_system() -> None:
    cells = _notebook()["cells"]
    by_id = {cell.get("id"): cell for cell in cells}
    title = _source(by_id["title"])
    banner_relative = (
        "assets/images/banner_candidates/"
        "water-ir-v2-04-trajectory-to-spectrum.png"
    )
    banner = PART_DIR / banner_relative

    assert '<h1 id="alchemi-notebook-title"' in title
    assert 'aria-label="Lesson summary"' in title
    assert "BOUNDARY" in title
    assert "CHECK" in title
    assert banner_relative in title
    assert banner.is_file()
    png = banner.read_bytes()[:24]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(png[16:20], "big") == 2880
    assert int.from_bytes(png[20:24], "big") == 1440

    presentation_markers = (
        'aria-label="Lesson summary"',
        "ILLUSTRATION SLOT · VISUAL REVIEW",
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


def test_measured_slow_cells_have_visible_progress_cards() -> None:
    """Keep progress visible around every cell that took at least 5 s on H100."""

    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    expected = {
        "setup": "Verify the pinned runtime",
        "imports": "Load simulation APIs",
        "load-aimnet": "Load pinned AIMNet2 checkpoint",
        "first-prediction": "Evaluate the first water interaction",
        "serial-batch-parity": "Serial loop vs one Toolkit batch",
        "cpu-gpu-crossover": "CPU / GPU fixed-workload sweep",
        "component-ablation": "Evaluate model components",
        "official-composition-parity": "Composition parity and force gate",
        "compile-fixed-ir-model": "Compile the fixed 42-atom IR workload",
        "relax": "Batched FIRE2 relaxation",
        "configure-dynamics": "Prepare and run the full NVT → NVE trajectory",
    }
    for cell_id, title in expected.items():
        source = _source(by_id[cell_id])
        assert "NotebookProgress(" in source, cell_id
        assert title in source, cell_id

    configure_source = _source(by_id["configure-dynamics"])
    assert configure_source.index("NotebookProgress(") < configure_source.index(
        "initialize_velocities("
    )


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
        "explicit D3 component": "DFTD3ModelWrapper(",
        "explicit electrostatics component": "DirectCoulombWrapper(",
        "pipeline grouping": "PipelineGroup(",
        "pipeline construction": "PipelineModelWrapper(",
        "segmented reduction": "segmented_sum(",
        "FIRE2 relaxation": "FIRE2(",
        "force convergence hook": "ConvergenceHook.from_fmax(",
        "Langevin dynamics": "NVTLangevin(",
        "NVE dynamics": "NVE(",
        "velocity initialization": "initialize_velocities(",
        "fused dynamics": "FusedStage",
        "hook protocol": ": Hook =",
        "NaN safety hook": "NaNDetectorHook(",
        "logging hook": "LoggingHook(",
        "Toolkit Zarr sink": "ZarrData(",
        "Toolkit neighbor hooks": ".make_neighbor_hooks(",
        "hook registration": ".register_hook(",
    }
    missing = [label for label, token in required_calls.items() if token not in source]
    assert not missing, f"core Toolkit calls hidden from learner-facing cells: {missing}"
    assert "DynamicsContext" in _source(
        next(cell for cell in _notebook()["cells"] if cell.get("id") == "ir-mechanism")
    )
    assert "ConvergenceHook.from_fmax(threshold=-1.0)" in source


def test_dimer_ablation_and_reference_claims_stay_distinct() -> None:
    by_id = {cell.get("id"): cell for cell in _notebook()["cells"]}
    boundary = _source(by_id["composition-boundary"])

    assert "All four curves are shown against full B97-3c" in boundary
    assert "only the complete AIMNet + all-pairs Coulomb + D3 model" in boundary
    assert "is interpreted as the endpoint comparison" in boundary
    assert "ablation distances that mix omitted physics with ML error" in boundary
    assert "not matched-level accuracy estimates" in boundary


def test_aux_does_not_hide_toolkit_batch_or_pipeline_construction() -> None:
    forbidden_call_tails = {
        "AtomicData.from_atoms",
        "Batch.from_data_list",
        "compute_neighbors",
        "PipelineGroup",
        "PipelineModelWrapper",
    }
    violations: list[str] = []

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
                part
                for part in (imported_aliases.get(root, root), remainder)
                if part
            )
            if any(
                canonical == tail or canonical.endswith(f".{tail}")
                for tail in forbidden_call_tails
            ):
                violations.append(
                    f"{path.relative_to(PART_DIR)}:{node.lineno}: {name}"
                )

    assert not violations, (
        "Toolkit conversion, batching, neighbors, and pipeline assembly must be "
        "visible in the notebook, not hidden in aux:\n" + "\n".join(violations)
    )


def test_aux_package_root_has_no_reexport_api() -> None:
    tree = ast.parse(
        (PART_DIR / "aux" / "__init__.py").read_text(encoding="utf-8")
    )
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
    assert assigned_literals.get("PRODUCTION_STEPS") == 50_000
    assert assigned_literals.get("DT_FS") == 0.5
    assert "n_steps=TOTAL_DYNAMICS_STEPS" in source
    assert "shape == (PRODUCTION_STEPS, 4, 3)" in source
    assert "shape == (PRODUCTION_STEPS, 42, 3)" in source

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
            violations.append(f"{self.cell_id}:{node.lineno} class {node.name}")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(node)

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
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
        "move substantial notebook helpers into a focused aux module:\n"
        + "\n".join(violations)
    )
