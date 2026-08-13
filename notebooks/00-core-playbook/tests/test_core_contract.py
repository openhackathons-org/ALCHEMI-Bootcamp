from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager

CORE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CORE_DIR.parents[1]
NOTEBOOK_PATH = CORE_DIR / "alchemi-core-playbook.ipynb"
HELPER_PATH = CORE_DIR / "helpers" / "core.py"
ASSET_INDEX_PATH = CORE_DIR / "assets" / "core-assets.json"
JOURNEY_PATH = CORE_DIR / "assets" / "core-journey.svg"
FRAMEWORK_PATH = CORE_DIR / "assets" / "framework-bindings.svg"
FRAMEWORK_DRAWIO_PATH = CORE_DIR / "assets" / "framework-bindings.drawio"
FRAMEWORK_ARCHIVE_PATH = CORE_DIR / "reference" / "framework-kernel-topology-archive.md"
DATA_RELATIONSHIP_PATH = CORE_DIR / "assets" / "molecule-atomicdata-batch.svg"
ZARR_FLOW_PATH = CORE_DIR / "assets" / "zarr-data-flow.svg"
ZARR_FLOW_DRAWIO_PATH = CORE_DIR / "assets" / "zarr-data-flow.drawio"
CAPABILITY_PATH = CORE_DIR / "assets" / "toolkit-capability-map.svg"
CAPABILITY_DRAWIO_PATH = CORE_DIR / "assets" / "toolkit-capability-map.drawio"
CURRICULUM_DRAWIO_PATH = CORE_DIR / "assets" / "core-curriculum.drawio"
CURRICULUM_RENDERER_PATH = CORE_DIR / "assets" / "render_core_journey.py"
HANDOFF_PATH = CORE_DIR / "DEEP_DIVE_CONTRACT.md"


def read_notebook(path: Path = NOTEBOOK_PATH) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def joined_source(notebook: nbformat.NotebookNode, cell_type: str | None = None) -> str:
    return "\n\n".join(
        cell.source
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


def read_asset_index() -> dict[str, object]:
    return json.loads(ASSET_INDEX_PATH.read_text(encoding="utf-8"))


def current_asset_name(logical_name: str) -> str:
    index = read_asset_index()
    return index["assets"][logical_name]["filename"]  # type: ignore[index]


def current_asset_path(logical_name: str) -> Path:
    return CORE_DIR / "assets" / current_asset_name(logical_name)


def test_contract_artifacts_exist() -> None:
    for path in (
        NOTEBOOK_PATH,
        HELPER_PATH,
        ASSET_INDEX_PATH,
        JOURNEY_PATH,
        FRAMEWORK_PATH,
        FRAMEWORK_DRAWIO_PATH,
        DATA_RELATIONSHIP_PATH,
        ZARR_FLOW_PATH,
        ZARR_FLOW_DRAWIO_PATH,
        CAPABILITY_PATH,
        CAPABILITY_DRAWIO_PATH,
        CURRICULUM_DRAWIO_PATH,
        CURRICULUM_RENDERER_PATH,
        HANDOFF_PATH,
    ):
        assert path.is_file(), path


def test_notebook_svg_references_are_content_addressed() -> None:
    assert ASSET_INDEX_PATH.is_file(), ASSET_INDEX_PATH
    index = read_asset_index()
    assert index["schema"] == "alchemi.core-assets.v1"

    markdown = joined_source(read_notebook(), "markdown")
    for logical_name, record in index["assets"].items():
        digest = record["sha256"]
        versioned_name = record["filename"]
        logical_path = CORE_DIR / "assets" / logical_name
        versioned_path = CORE_DIR / "assets" / versioned_name
        assert re.fullmatch(
            rf"{re.escape(Path(logical_name).stem)}-[0-9a-f]{{16}}\.svg",
            versioned_name,
        )
        assert versioned_name.endswith(f"-{digest[:16]}.svg")
        assert hashlib.sha256(versioned_path.read_bytes()).hexdigest() == digest
        assert logical_path.read_bytes() == versioned_path.read_bytes()
        assert f'assets/{logical_name}"' not in markdown

    referenced = set(re.findall(r'assets/([^"\s>]+\.svg)', markdown))
    available = {record["filename"] for record in index["assets"].values()}
    assert referenced <= available
    assert current_asset_path("toolkit-capability-map.svg").is_file()


def test_notebook_is_valid_and_parses() -> None:
    notebook = read_notebook()
    assert notebook.metadata["core"]["asset_index"] == "assets/core-assets.json"
    assert len(notebook.cells) >= 145
    assert sum(cell.cell_type == "code" for cell in notebook.cells) >= 85

    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    ast.parse(transformed)


def test_learner_copy_uses_the_core_title_and_plain_duration() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    assert "# 00 · ALCHEMI Core Playbook" in markdown
    assert "**Time to complete: about 90 minutes.**" in markdown
    assert "Toolkit Core" not in markdown
    assert not (CORE_DIR.parent / "00-workflow-playbook").exists()


def test_major_sections_have_one_shared_divider() -> None:
    notebook = read_notebook()
    section_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown" and "alchemi-section-divider" in cell.source
    ]
    assert len(section_cells) == 10
    divider = '<hr class="alchemi-section-divider"'
    assert sum(cell.source.count(divider) for cell in notebook.cells) == 10
    assert all(cell.source.count(divider) == 1 for cell in section_cells)
    assert not any(
        divider in left.source and divider in right.source
        for left, right in zip(notebook.cells, notebook.cells[1:])
    )
    assert all(
        "border-top:1px solid #D6D9D4" in cell.source
        and "margin:2.4rem 0 1rem" in cell.source
        for cell in section_cells
    )
    markdown = joined_source(notebook, "markdown")
    assert "journey-banner-" not in markdown
    # Section headings carry no lesson number. The deep-dive part each section
    # maps to is named in that section's closing "Go deeper" link instead.
    for heading in (
        "## Build the molecule with ASE",
        "## Pack several molecules into one batch",
        "## Save records and recover the next model input",
        "## Evaluate a pretrained model",
        "## Combine model components",
        "## Observe and protect a simulation with hooks",
        "## Attempt a bounded relaxation with FIRE2",
        "## Run a short molecular-dynamics stage",
        "## Take a few training steps",
        "## Preview domain parallelism in one process",
    ):
        assert heading in markdown
    assert not re.search(r"^## \d\d · ", markdown, flags=re.MULTILINE)


def test_zarr_round_trip_teaches_the_supported_public_path() -> None:
    notebook = read_notebook()
    start = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "Save records and recover the next model input" in cell.source
    )
    stop = next(
        index
        for index, cell in enumerate(notebook.cells[start + 1 :], start + 1)
        if cell.cell_type == "markdown" and "alchemi-section-divider" in cell.source
    )
    section = notebook.cells[start:stop]
    source = "\n".join(cell.source for cell in section)
    assert len(section) >= 25
    assert sum(cell.cell_type == "code" for cell in section) >= 17
    ordered_calls = (
        "AtomicDataZarrWriter(STORE)",
        "writer.write(source_batch)",
        "source_objects_released",
        "AtomicDataZarrReader(STORE)",
        "reader.read_many(",
        "reader.read(1)",
        'Dataset(reader, device="cpu", num_workers=2)',
        "dataset[1]",
        "DataLoader(",
        "next(iter(loader))",
    )
    positions = [source.index(call) for call in ordered_calls]
    assert positions == sorted(positions)
    for keyword in (
        "batch_size=3",
        "shuffle=False",
        "prefetch_factor=1",
        "use_streams=False",
    ):
        assert keyword in source
    for evidence in (
        '"record_id"',
        "reader.field_levels",
        "batch_ptr.tolist()",
        "atomic numbers preserved",
        "positions preserved",
        "temperatures preserved",
        "shutil.rmtree(STORE.parent)",
        "assets/zarr-data-flow-",
        "records_to_save",
        "loaded_records",
        "Bring another data source",
        "Dataset → DataLoader → Batch",
        "another notebook or training job",
    ):
        assert evidence in source
    assert source.index("assets/zarr-data-flow-") < source.index("records_to_save = [")
    assert "persisted_records" not in source
    assert "raw_records" not in source
    assert "DataLoader(device=" not in source
    assert "**What to notice:**" not in source
    assert "disk or in CPU storage" in source
    # The reader hands back saved arrays; either wording of that contrast is fine.
    assert re.search(r"(?:not|rather than) live Python objects", source)
    assert "pinned CPU memory" in source
    assert "pinned memory is VRAM" not in source
    assert "Records are saved once, then loaded and batched for model use." in source
    assert "logical length (<code>__len__</code>)" in source
    assert "field ownership (<code>field_levels</code>)" in source
    assert "ordered reads through <code>read_many(...)</code>" in source
    assert "InMemoryDataset" in source
    assert "malformed" not in source.lower()


def test_zarr_flow_asset_matches_the_pinned_device_path() -> None:
    svg_root = ET.parse(ZARR_FLOW_PATH).getroot()
    assert svg_root.attrib["width"] == "920"
    assert svg_root.attrib["height"] == "96"
    text = " ".join("".join(svg_root.itertext()).split())
    for label in (
        "Zarr store",
        "disk or CPU storage",
        "Reader",
        "CPU tensors",
        "Dataset",
        "target device",
        "DataLoader",
        "batch + prefetch",
        "Batch",
        "model device",
    ):
        assert label in text
    assert "VRAM" not in text
    assert "GPU" not in text

    drawio_root = ET.parse(ZARR_FLOW_DRAWIO_PATH).getroot()
    cells = {cell.attrib["id"]: cell for cell in drawio_root.findall(".//mxCell")}
    assert {"zarr", "reader", "dataset", "loader", "batch"} <= cells.keys()
    assert cells["zarr"].find("mxGeometry").attrib["x"] == "20"
    assert cells["batch"].find("mxGeometry").attrib["x"] == "732"
    assert all(
        "edgeStyle=none" in cells[edge_id].attrib["style"]
        for edge_id in (
            "edge-zarr-reader",
            "edge-reader-dataset",
            "edge-dataset-loader",
            "edge-loader-batch",
        )
    )


def test_matterviz_widgets_receive_explicit_checked_bonds(monkeypatch) -> None:
    pytest.importorskip("torch")
    sys.path.insert(0, str(CORE_DIR))
    from helpers import core as helpers

    learner_source = joined_source(read_notebook())
    assert "helpers.bond_rows" not in learner_source
    assert "Explicit bonds:" not in learner_source
    assert "benzoate_bonds" not in learner_source
    assert (
        "helpers.show_molecule(ethyne, bonds=helpers.infer_bonds(ethyne), height=360)"
        in learner_source
    )
    assert (
        "helpers.show_molecule(benzoate, bonds=helpers.infer_bonds(benzoate), height=360)"
        in learner_source
    )

    ethyne = helpers.load_molecules(("Ethyne",))[0][0]
    benzoate, record = helpers.load_benzoate_anion()
    assert record["charge"] == -1
    assert benzoate.info["charge"] == -1

    captured: dict[str, object] = {}

    def fake_widget(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(
        sys.modules, "pymatviz", SimpleNamespace(StructureWidget=fake_widget)
    )
    for atoms in (ethyne, benzoate):
        bonds = helpers.infer_bonds(atoms)
        assert bonds
        assert len(bonds) == len(set(bonds))
        assert all(left < right for left, right in bonds)
        helpers.show_molecule(atoms, bonds=bonds, height=360)

        structure = captured["structure"]
        assert isinstance(structure, dict)
        explicit_bonds = structure["properties"]["bonds"]
        assert len(explicit_bonds) == len(bonds)
        assert explicit_bonds[0] == {
            "site_idx_1": bonds[0][0],
            "site_idx_2": bonds[0][1],
            "order": 1,
        }
        assert "show_bonds" not in captured
        assert captured["bond_thickness"] == 0.14
        assert captured["bond_color"] == "#D7DEE5"
        assert captured["atom_radius"] == 0.72


def test_visible_code_cells_remain_small_and_focused() -> None:
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    visible = [
        cell
        for cell in code_cells
        if "hide-input" not in cell.metadata.get("tags", [])
        and not cell.metadata.get("jupyter", {}).get("source_hidden", False)
    ]
    assert visible

    lengths = [len(cell.source.splitlines()) for cell in visible]
    assert sum(length <= 8 for length in lengths) > len(lengths) / 2
    oversized = {
        cell.id: len(cell.source.splitlines())
        for cell in visible
        if len(cell.source.splitlines()) > 20
        and not cell.metadata.get("core_allow_long")
    }
    assert not oversized, oversized


def test_required_public_apis_remain_visible() -> None:
    source = joined_source(read_notebook())
    required = {
        "AtomicData.from_atoms(",
        ".system_properties",
        "Batch.from_data_list(",
        ".num_nodes_per_graph",
        ".batch_idx",
        ".batch_ptr",
        ".get_data(",
        ".add_key(",
        "AtomicDataZarrReader(",
        "Dataset(",
        "DataLoader(",
        "AIMNet2Wrapper.from_checkpoint(",
        ".model_config",
        ".input_data()",
        '.set_config("active_outputs"',
        "compute_neighbors(",
        "model(model_batch)",
        ".make_neighbor_hooks()",
        "NaNDetectorHook(",
        "SnapshotHook(",
        "DynamicsStage.AFTER_COMPUTE",
        "ConvergenceHook.from_fmax(",
        "FIRE2(",
        "fire2.run(",
        "LennardJonesModelWrapper(",
        "NVE(",
        "WrapPeriodicHook(",
        "BaseModelMixin",
        "ModelConfig",
        "PipelineModelWrapper",
        "lj_model + correction_wrapper",
        "FineTuningStrategy(",
        "OptimizerConfig(",
        "EnergyMSELoss(",
        "default_training_fn",
        "DistributedManager",
        "DomainConfig(",
        "DomainParallel(",
        ".partition(",
        "domain.run(",
        "domain.gather(",
        "DistributedManager.cleanup",
    }
    missing = {token for token in required if token not in source}
    assert not missing, sorted(missing)


def test_teaching_order_places_fire2_before_md_and_scale_last() -> None:
    source = joined_source(read_notebook())
    assert source.index("FIRE2(") < source.index("NVE(")
    assert source.index("NVE(") < source.index("FineTuningStrategy(")
    assert source.index("FineTuningStrategy(") < source.index("DomainParallel(")


def test_scientific_boundaries_are_explicit() -> None:
    source = joined_source(read_notebook()).lower()
    required_claims = {
        "each section rebuilds the inputs needed for its own api",
        "these cells verify adapter execution, output shapes, dtypes, and labeled units",
        "they do not establish equilibration",
        "they do not measure chemical accuracy",
        "with a world size of one",
        "no domain decomposition or scaling measurement occurs",
    }
    missing = {claim for claim in required_claims if claim not in source}
    assert not missing, sorted(missing)


def test_source_is_clean_and_avoids_forbidden_patterns() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    forbidden = {
        "object.__setattr__",
        "._models",
        "._groups",
        "fake context",
        "learner can state",
        "pedagogical",
        "ai-powered",
        "alchemi-stage-card",
        "taught",
        "glimpsed",
        "real chemistry lane",
        "fast physics lane",
        "protected buffer",
        "owner_block",
        "canonical",
        "pinned evidence",
        "reference material here",
        "intentionally not taught",
    }
    present = {token for token in forbidden if token.lower() in markdown.lower()}
    assert not present, sorted(present)
    assert not re.search(r"(?m)^#{2,4}\s+A\d{2}\b", markdown)
    assert not re.search(r"(?m)^##\s+\d{2}[–-]\d{2}\b", markdown)
    assert not re.search(r"\baction(?:s)?\b", markdown, flags=re.IGNORECASE)
    assert not re.search(
        r"(?m)^(?:Now|Next|In this section)\b", markdown, flags=re.IGNORECASE
    )
    # Saved outputs are the exception, so every one of them must come from a real
    # execution rather than a hand-pasted result.
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.outputs:
            assert cell.execution_count is not None, cell.id


def test_every_section_ends_with_a_go_deeper_link() -> None:
    notebook = read_notebook()
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    deeper = [cell for cell in markdown_cells if "**Go deeper:**" in cell.source]
    assert len(deeper) >= 10
    expected_targets = {
        "../01-atomicdata-batch/atomicdata-and-batch.ipynb",
        "../02-zarr-data-loading/zarr-data-loading.ipynb",
        "../03-model-interfaces-composition/model-interfaces-composition.ipynb",
        "../04-hooks/hooks.ipynb",
        "../05-base-dynamics/base-dynamics.ipynb",
        "../06-gpu-pipelines-profiling/gpu-pipelines-profiling.ipynb",
        "../07-training-finetuning/training-finetuning.ipynb",
        "../08-domain-decomposition/domain-decomposition.ipynb",
    }
    source = "\n".join(cell.source for cell in deeper)
    assert expected_targets <= set(re.findall(r"\]\(([^)#]+)", source))
    for cell in deeper:
        for target in re.findall(r"\]\(([^)#]+)", cell.source):
            if target.startswith(("https://", "http://")):
                continue
            # Deep-dive notebooks live on the v3-deep-dives branch and are not
            # checked out here, so a sibling lesson link is verified by shape.
            # Any other relative target must still resolve on disk.
            if re.fullmatch(r"\.\./\d\d-[a-z0-9-]+/[a-z0-9-]+\.ipynb", target):
                continue
            assert (CORE_DIR / target).resolve().exists(), target


def test_visuals_are_bounded_and_accessible() -> None:
    notebook = read_notebook()
    source = joined_source(notebook)
    assert "helpers.show_capability_map()" in source
    assert "helpers.show_molecule(" in source
    assert "helpers.plot_batch_ownership(" in source
    assert "helpers.plot_fire2_evidence(" in source
    assert "helpers.plot_structure_change(" in source
    assert "helpers.plot_nve_trace(" in source
    assert "helpers.plot_wrapper_flow(" in source
    assert "helpers.plot_training_loss(" in source
    assert "helpers.plot_domain_control(" in source
    for cell in notebook.cells:
        if cell.cell_type == "code" and "helpers.plot_" in cell.source:
            assert cell.metadata.get("alt"), cell.id


def test_alchemi_context_and_archived_framework_reference_are_separate() -> None:
    core = joined_source(read_notebook(), "markdown")
    assert "## Where NVIDIA ALCHEMI fits" in core
    for required_link in (
        "https://github.com/NVIDIA/nvalchemi-toolkit",
        "https://github.com/NVIDIA/nvalchemi-toolkit-ops",
        "https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest/index.html",
    ):
        assert required_link in core
    assert "## Explore the Toolkit capability map" in core
    assert "helpers.show_capability_map()" in joined_source(read_notebook(), "code")
    assert FRAMEWORK_ARCHIVE_PATH.exists()
    archive = FRAMEWORK_ARCHIVE_PATH.read_text(encoding="utf-8")
    assert "Part 06" in archive
    assert "framework-bindings.drawio" in archive
    assert FRAMEWORK_PATH.exists()
    assert ET.parse(FRAMEWORK_DRAWIO_PATH).getroot().tag == "mxGraphModel"


def test_capability_map_is_interactive_accessible_and_task_focused() -> None:
    source = CAPABILITY_PATH.read_text(encoding="utf-8")
    root = ET.parse(CAPABILITY_PATH).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None
    for label in (
        "Data and state",
        "Models and potentials",
        "Simulation workflows",
        "Training and scale",
    ):
        assert label in source
    for application in (
        "inputs · datasets · trajectories",
        "MLIPs · custom models · composition",
        "FIRE2 · NVE/NVT · hooks",
        "TrainingStrategy · DomainParallel",
    ):
        assert application in source
    for numbered_area in (
        "01 AtomicData + Batch",
        "02 Zarr data",
        "03 Models + composition",
        "04 Hooks · 05 Dynamics",
        "06 GPU pipelines",
        "07 Training",
        "08 Domain decomposition",
    ):
        assert numbered_area in source
    assert source.count('class="capability"') == 4
    assert source.count('class="cap-tooltip"') == 4
    assert "Toolkit-Ops" in source
    assert "neighbor lists · segment operations · interactions" in source
    assert "prefers-reduced-motion" in source
    assert "soft-glow" in source
    assert "DEEP DIVES · IN PROGRESS" in source
    assert ".doc-label" in source
    assert ".deep-status" in source
    assert ".deep-status{fill:#929BA4}" in source
    anchors = root.findall(f".//{namespace}a")
    assert anchors == []
    assert len(root.findall(".//*[@class='doc-label']")) == 5
    assert len(root.findall(".//*[@class='deep-status']")) == 8
    assert "ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/" not in source
    assert ET.parse(CAPABILITY_DRAWIO_PATH).getroot().tag == "mxGraphModel"


def test_capability_map_uses_clickable_html_and_plain_in_progress_status() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import core as helpers

    html = helpers.show_capability_map().data
    assert html.count("<a href=") == 5
    assert html.count('class="alchemi-cap-region"') == 4
    assert html.count('class="alchemi-cap-status-label"') == 4
    assert ":focus-within" not in html
    assert ".alchemi-cap-region:hover" in html
    assert "min-height:160px" in html
    assert "overflow:hidden" not in html
    detail_rule = html.split(".alchemi-cap-detail{", 1)[1].split("}", 1)[0]
    assert "background-color:#151A17" in detail_rule
    assert "opacity:" not in detail_rule
    assert "z-index:10" in detail_rule
    for guide in ("data", "models", "dynamics", "training", "distributed"):
        assert f'userguide/{guide}.html" target="_blank"' in html
    assert "ALCHEMI-Bootcamp/blob/v3-api-first/notebooks/" not in html

    markdown = joined_source(read_notebook(), "markdown")
    assert markdown.count('style="color:#7C8794;"') >= 8
    assert 'href="../01-atomicdata-batch/' not in markdown


def test_small_exercises_keep_attempt_feedback_and_answer_together() -> None:
    notebook = read_notebook()
    markdown = joined_source(notebook, "markdown")
    assert "### Try it:" in markdown
    assert "<details>" in markdown
    assert "<summary>Check" in markdown or "<summary>Show" in markdown
    code = joined_source(notebook, "code")
    assert "helpers.load_benzoate_anion()" in code
    assert "benzoate_data.charge.item()" in code
    assert "graph_index = 1" in code
    assert "SNAPSHOT_EVERY = 4" in code


def flattened_output_text(output: nbformat.NotebookNode) -> str:
    """Collect every textual payload of one saved output, list or string form."""

    chunks: list[str] = []
    candidates = [output.get("text", "")]
    candidates += [
        value
        for key, value in output.get("data", {}).items()
        if key.startswith("text/")
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            chunks.append(candidate)
        elif isinstance(candidate, list):
            chunks.extend(str(line) for line in candidate)
    return "".join(chunks)


def test_saved_outputs_carry_no_errors_or_tracebacks() -> None:
    notebook = read_notebook()
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error", cell.id
            assert "ename" not in output, cell.id
            text = flattened_output_text(output)
            assert "Traceback (most recent call last)" not in text, cell.id


def test_archived_framework_primer_is_absent_from_learner_notebook() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    for forbidden in (
        "How ALCHEMI Toolkit reaches accelerated kernels",
        "framework-bindings",
        "ALCHEMI Toolkit calls Toolkit-Ops Torch bindings",
        "Toolkit-Ops also provides separate JAX bindings",
        "muted JAX branch is ecosystem context",
    ):
        assert forbidden not in markdown


def test_visual_system_matches_part01_without_competing_cards() -> None:
    markdown = joined_source(read_notebook(), "markdown")
    assert "background:#F2F3F1" in markdown
    assert "background:#151A1F" in markdown
    assert ">💡 Highlight</div>" in markdown
    assert "alchemi-go-deeper-label" not in markdown
    assert "alchemi-process" not in markdown
    assert "STEP 1" not in markdown
    assert "box-shadow:" not in markdown
    assert "linear-gradient(" not in markdown
    assert markdown.count("**Go deeper:**") >= 10


def test_journey_variants_keep_two_level_accessible_structure() -> None:
    expected_mapping = {
        "Data preparation": {"AtomicData", "Batches", "Data loaders"},
        "Models": {"Built-in models", "Wrapping your model"},
        "Running simulations": {
            "Hooks",
            "Structure relaxation",
            "Molecular dynamics",
        },
        "Learning and scale": {"Fine-tuning", "Domain decomposition"},
    }
    expected_phase_labels = set(expected_mapping)
    for path in [JOURNEY_PATH, *(CORE_DIR / "assets").glob("journey-banner-*.svg")]:
        root = ET.parse(path).getroot()
        namespace = "{http://www.w3.org/2000/svg}"
        assert root.find(f"{namespace}title") is not None
        assert root.find(f"{namespace}desc") is not None
        text = " ".join(node.text or "" for node in root.iter())
        assert expected_phase_labels <= {
            label for label in expected_phase_labels if label in text
        }
        assert not re.search(r"\b0[1-9]\b", text)
        view_box = [float(value) for value in root.attrib["viewBox"].split()]
        assert view_box[2] == 880
        assert "marker-end=" not in path.read_text(encoding="utf-8")
        assert len(root.findall(f"{namespace}line")) == 9
        if path.name.startswith("journey-banner-"):
            assert view_box[3] < 100
            active_area = root.find(
                f"{namespace}rect[@data-kind='area'][@data-state='active']"
            )
            active_step = root.find(
                f"{namespace}rect[@data-kind='step'][@data-state='active']"
            )
            assert active_area is not None
            assert active_step is not None
            assert active_area.attrib["fill"] == "#213016"
            assert active_area.attrib["stroke"] == "#76B900"
            assert active_step.attrib["fill"] == "#76B900"

    drawio_root = ET.parse(CURRICULUM_DRAWIO_PATH).getroot()
    assert drawio_root.attrib["pageWidth"] == "880"
    cells = {cell.attrib["id"]: cell for cell in drawio_root.findall(".//mxCell")}
    observed: dict[str, set[str]] = {}
    step_cells = [cells[f"step-{index}"] for index in range(1, 11)]
    assert {float(cell.find("mxGeometry").attrib["height"]) for cell in step_cells} == {
        38.0
    }
    for index in range(1, 10):
        edge_style = cells[f"edge-{index}"].attrib["style"]
        assert "startArrow=none" in edge_style
        assert "endArrow=none" in edge_style
    for phase_index, phase in enumerate(expected_mapping, start=1):
        phase_cell = cells[f"phase-{phase_index}"]
        phase_geometry = phase_cell.find("mxGeometry")
        assert phase_geometry is not None
        left = float(phase_geometry.attrib["x"])
        right = left + float(phase_geometry.attrib["width"])
        children = set()
        for step_cell in step_cells:
            geometry = step_cell.find("mxGeometry")
            assert geometry is not None
            center = float(geometry.attrib["x"]) + float(geometry.attrib["width"]) / 2
            if left <= center <= right:
                label = re.sub(r"<[^>]+>", " ", step_cell.attrib["value"])
                children.add(" ".join(label.split()))
        observed[phase] = children
    assert observed == expected_mapping


def test_curriculum_assets_remain_editable_without_repeating_in_sections() -> None:
    slugs = (
        "atomicdata",
        "batches",
        "data-loaders",
        "built-in-models",
        "wrapping-your-model",
        "hooks",
        "structure-relaxation",
        "molecular-dynamics",
        "fine-tuning",
        "domain-decomposition",
    )
    markdown = joined_source(read_notebook(), "markdown")
    assert ET.parse(CURRICULUM_DRAWIO_PATH).getroot().tag == "mxGraphModel"
    for slug in slugs:
        logical_name = f"journey-banner-{slug}.svg"
        asset = current_asset_path(logical_name)
        assert asset.is_file()
        assert f"assets/{asset.name}" not in markdown
        assert f'assets/{logical_name}"' not in markdown
        svg = ET.parse(asset).getroot()
        assert svg.tag.endswith("svg")
        assert svg.find("{http://www.w3.org/2000/svg}title") is not None
        assert svg.find("{http://www.w3.org/2000/svg}desc") is not None

    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (CORE_DIR / "assets").glob("*")
        if path.suffix in {".drawio", ".svg"}
    }
    subprocess.run(
        [sys.executable, str(CURRICULUM_RENDERER_PATH)],
        cwd=REPO_ROOT,
        check=True,
    )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (CORE_DIR / "assets").glob("*")
        if path.suffix in {".drawio", ".svg"}
    }
    assert before == after


def test_code_comments_avoid_python_narration() -> None:
    code = joined_source(read_notebook(), "code")
    forbidden = {
        "# create a list",
        "# loop over",
        "# print the result",
        "# import torch",
    }
    assert not {comment for comment in forbidden if comment.lower() in code.lower()}


def test_named_scientific_constants_feed_the_public_configuration() -> None:
    tree = ast.parse(joined_source(read_notebook(), "code"))
    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    required = {
        "FMAX_EV_PER_A",
        "MAX_STEPS",
        "LJ_EPSILON_EV",
        "LJ_SIGMA_A",
        "LJ_CUTOFF_A",
        "ARGON_INITIAL_TEMPERATURE_K",
        "NVE_DT_FS",
        "NVE_STEPS",
        "LEARNING_RATE",
        "TRAINING_UPDATES",
        "DOMAIN_STEPS",
    }
    assert required <= assignments

    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    assert all(names.count(name) >= 2 for name in required)


def test_deep_dive_handoff_contract_is_complete() -> None:
    source = HANDOFF_PATH.read_text(encoding="utf-8")
    for token in (
        "Core excerpt IDs and provenance",
        "Synthetic quick example + real-world example",
        "Required public APIs",
        "Language, visual, and cell-size rules",
        "Human-review stages",
        "Placeholder paths to replace",
        "Required tests",
    ):
        assert token in source
