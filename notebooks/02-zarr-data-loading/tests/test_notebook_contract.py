from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "zarr-data-loading.ipynb"
N01_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "01-atomicdata-batch"
    / "atomicdata-and-batch.ipynb"
)


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def normalize_markdown(source: str) -> str:
    """Ignore incidental Markdown whitespace while preserving wording and links."""
    return " ".join(source.split())


def test_notebook_is_valid_and_complete_namespace_parses() -> None:
    notebook = read_notebook()
    transformer = TransformerManager()
    code = "\n\n".join(
        transformer.transform_cell(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )

    ast.parse(code)
    assert 68 <= len(notebook.cells) <= 84


def test_opening_uses_shared_part_two_orientation() -> None:
    notebook = read_notebook()

    assert "alchemi-banner-left.png" in notebook.cells[0].source
    assert notebook.cells[1].source.startswith("# 02 · Data loading with Zarr")
    assert "**Goal:**" in notebook.cells[1].source
    assert "**Core concepts:**" in notebook.cells[1].source
    assert "**Prerequisites:**" in notebook.cells[1].source
    assert "alchemi-core-playbook.ipynb" in notebook.cells[1].source

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "<details" in source
    assert "curriculum-map-02.svg" in source
    assert "../01-atomicdata-batch/atomicdata-and-batch.ipynb#from-one-molecule-to-a-batch" in source

    orientation = notebook.cells[3].source
    assert '<details aria-label="New to ALCHEMI Toolkit?">' in orientation
    assert orientation.index("</details>") < orientation.index("<object")
    assert '<object data="../../shared/curriculum-map-02.svg"' in orientation
    assert 'aria-label="Interactive ALCHEMI Toolkit curriculum.' in orientation
    assert "multi-GPU workflows" in orientation
    assert "aspect-ratio:900/552" in orientation
    assert '<img src="../../shared/curriculum-map-02.svg"' not in orientation
    assert orientation.index("</object>") < orientation.index(
        "[Open the evolving Part 02 curriculum map directly]"
    )
    assert "Part 02" in orientation
    assert "evolving" in orientation.lower()


def test_cpu_cuda_environment_boundary_is_plain_and_unsuppressed() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "driver-entry-point errors" in markdown
    assert "CPU-only" in markdown
    assert "If you expected GPU" in markdown
    for suppression in (
        "redirect_stderr",
        "filterwarnings",
        "WARP_DISABLE_CUDA",
        "os.devnull",
    ):
        assert suppression not in source


def test_alchemi_recap_tracks_current_n01_product_text() -> None:
    notebook = read_notebook()
    n01 = nbformat.read(N01_NOTEBOOK_PATH, as_version=4)
    n01_cell = next(
        cell
        for cell in n01.cells
        if cell.cell_type == "markdown"
        and cell.source.startswith("## Where NVIDIA ALCHEMI fits")
    )
    n01_body = n01_cell.source.split("\n", 1)[1]

    recap = notebook.cells[3].source
    summary = "<summary>Where NVIDIA ALCHEMI fits (recap)</summary>"
    recap_body = recap.split(summary, 1)[1].split("</details>", 1)[0]

    assert summary in recap
    assert normalize_markdown(recap_body) == normalize_markdown(n01_body)
    assert recap.index("</details>") < recap.index("**Course orientation:**")
    assert recap.index("**Course orientation:**") < recap.index("<object")
    assert recap.index("</object>") < recap.index("**Takeaway:**")


def test_setup_cells_use_shared_hidden_input_metadata() -> None:
    notebook = read_notebook()

    for index in (2, 4):
        metadata = notebook.cells[index].metadata
        assert "hide-input" in metadata.get("tags", [])
        assert metadata.get("jupyter", {}).get("source_hidden") is True


def test_notebook_keeps_required_toolkit_calls_visible() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    required = {
        "AtomicData(",
        "add_system_property",
        "AtomicDataZarrWriter",
        "writer.write(source_batch)",
        "AtomicDataZarrReader",
        "reader.read(1)",
        "class ExtXYZReader(Reader)",
        "extxyz_reader.read(23)",
        "nci_dataset[23]",
        "nci_writer.write(batch)",
        "nci_writer.append(batch)",
        "nci_reader.read_many(EXAMPLE_IDS)",
        "Batch.from_data_list",
        "Dataset(",
        "InMemoryDataset(",
        "DataLoader(",
        "class MissingPositionReader(Reader)",
        "batch_ptr",
        "dataset.close()",
        "nci_stored_dataset.close()",
    }

    missing = {token for token in required if token not in source}
    assert not missing, f"missing visible Toolkit calls: {sorted(missing)}"


def test_notebook_follows_the_persistence_to_loading_order() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert source.index("## First: three small records") < source.index(
        "## Then: stream the NCI collection"
    )
    assert source.index("writer.write(source_batch)") < source.index(
        "reader = AtomicDataZarrReader(STORE)"
    )
    assert source.index("reader = AtomicDataZarrReader(STORE)") < source.index(
        "dataset = Dataset(reader"
    )
    assert source.index("dataset = Dataset(reader") < source.index(
        "loader = DataLoader("
    )
    assert source.index("loader = DataLoader(") < source.index(
        "first_batch = next(iter(loader))"
    )
    assert source.index("first_batch = next(iter(loader))") < source.index(
        "class ExtXYZReader(Reader)"
    )
    assert source.index("class ExtXYZReader(Reader)") < source.index(
        "nci_dataset = Dataset(extxyz_reader"
    )
    assert source.index("nci_dataset = Dataset(extxyz_reader") < source.index(
        "nci_writer = AtomicDataZarrWriter(NCI_STORE)"
    )
    assert source.index("nci_writer.append(batch)") < source.index(
        "nci_reader = AtomicDataZarrReader(NCI_STORE)"
    )
    assert "list(stored_loader)" not in source


def test_real_objects_are_inspected_across_public_boundaries() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert '"record_id", torch.tensor' in source
    assert source.index("add_system_property(") < source.index(
        "writer.write(source_batch)"
    )
    for boundary in (
        '"boundary": "AtomicData before write"',
        '"boundary": "Reader after reopen"',
        '"boundary": "Dataset indexing"',
        '"boundary": "DataLoader batching"',
    ):
        assert boundary in source
    assert "payload_comparison = pd.DataFrame(" in source
    assert '"metadata": "not emitted; record_id remains in the payload"' in source
    assert source.count('"ownership evidence":') == 4
    assert '"AtomicData system view": sorted(validated_record.system_properties)' in source
    assert "does not rebuild the custom `AtomicData.system_properties` view" in source
    assert source.count("class MissingPositionReader(Reader)") == 1


def test_custom_extxyz_reader_teaches_the_public_extension_hook() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    reader_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "class ExtXYZReader(Reader)" in cell.source
    )

    assert "def __len__(self)" in reader_cell
    assert "def _load_sample(self, index)" in reader_cell
    assert "def _get_sample_metadata(self, index)" in reader_cell
    assert "def field_levels(self)" in reader_cell
    assert (
        '"atomic_numbers": "atom", "positions": "atom", "record_id": "system"'
        in reader_cell
    )
    assert 'self.table.loc[index, ["label", "formula", "source"]]' in reader_cell
    assert "read_extxyz(self.path, index=index)" in reader_cell
    assert "AtomicData.from_atoms" not in reader_cell
    assert source.index("extxyz_reader.read(23)") < source.index("nci_dataset[23]")


def test_core_selected_writer_and_reader_cells_remain_stable() -> None:
    notebook = read_notebook()
    by_id = {cell.id: cell for cell in notebook.cells}

    assert by_id["3dd33a3f"].source == (
        "writer = AtomicDataZarrWriter(STORE)\nwriter.write(source_batch)"
    )
    assert by_id["5e4d2132"].source == (
        "reader = AtomicDataZarrReader(STORE)\nlen(reader)"
    )


def test_public_signatures_and_resident_ownership_are_exact() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "DataLoader(dataset, batch_size=...) -&gt; Iterator[Batch]" in source
    assert "DataLoader(dataset, batch_size) " not in source
    assert "reader-backed cache is built on CPU" in source
    assert "prebuilt GPU `in_memory_batch`" in source
    assert "num_workers` creates threads" in source


def test_resident_loader_actions_are_split() -> None:
    notebook = read_notebook()
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    construction = next(source for source in code_cells if "resident_loader = DataLoader(" in source)
    iteration = next(source for source in code_cells if "resident_first = next(" in source)
    inspection = next(source for source in code_cells if "resident_parity = {" in source)

    assert "resident_first" not in construction
    assert "resident_parity" not in construction
    assert "resident_loader = DataLoader(" not in iteration
    assert "resident_parity" not in iteration
    assert "resident_first = next(" not in inspection


def test_streamed_loader_actions_are_split() -> None:
    notebook = read_notebook()
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    construction = next(source for source in code_cells if "loader = DataLoader(" in source)
    iteration = next(source for source in code_cells if "first_batch = next(" in source)
    inspection = next(source for source in code_cells if "first_batch_summary = {" in source)
    comparison = next(source for source in code_cells if "payload_comparison = " in source)

    assert "first_batch" not in construction
    assert "first_batch_summary" not in iteration
    assert "first_batch = next(" not in inspection
    assert "payload_comparison" not in inspection
    assert "first_batch_summary = {" not in comparison


def test_compute_and_dataframe_display_are_separate_cells() -> None:
    notebook = read_notebook()
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    field_compute = next(source for source in code_cells if "field_table = " in source)
    field_display = next(
        source for source in code_cells if 'field_table.set_index("field")' in source
    )
    stream_compute = next(source for source in code_cells if "stream_rows = [" in source)
    stream_display = next(
        source for source in code_cells if 'stream_table.set_index("batch")' in source
    )

    assert "field_table.set_index" not in field_compute
    assert "field_table =" not in field_display
    assert "stream_table.set_index" not in stream_compute
    assert "stream_rows =" not in stream_display


def test_notebook_uses_approved_visual_and_callout_patterns() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert source.count("💡 Highlight") == 1
    assert source.count("ALCHEMI TOOLKIT API") == 1
    assert "## Where does a stored record become graph data?" in source
    assert "classDef current fill:#76B900" in source
    assert "accTitle: Stored records become graph batches" in source
    assert "accDescr: Zarr arrays pass through Reader and Dataset" in source
    assert "The reader owns storage I/O" in source
    assert "helpers.plot_record_layout(atom_counts, atoms_ptr)" in source
    assert "helpers.figure_to_html(figure, plot_alt)" in source

    assert (
        "**Figure description:** NCI record sizes and cumulative Zarr "
        "atom-row boundaries."
    ) in source
    assert source.index("helpers.figure_to_html(figure, plot_alt)") < source.index(
        "**Figure description:**"
    )
    assert source.index("## Stream the reopened store") < source.index(
        "nci_stored_dataset = Dataset("
    )


def test_notebook_has_bounded_try_it_and_current_recap() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "## Try it: request three records" in source
    assert "assert loaded_ids == requested_ids" in source
    assert "assert loaded_labels == requested_labels" in source
    assert "Success: requested order is preserved" in source
    assert "## Recap" in source
    assert "### What you learned" in source
    assert "### How we will use this" in source
    assert "../03-model-interfaces-composition/model-interfaces-composition.ipynb" in source
    assert "Results summary" not in source
    assert "homework" not in source


def test_validation_claim_matches_the_demonstrated_invariant() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert 'sample["positions"] = sample["positions"][:-1]' in source
    assert 'error.errors()[0]["msg"]' in source
    assert "expected 13, got 12" in source
    assert "positions` row count agrees with the graph's atom count" in source


def test_code_cell_pacing_stays_compact() -> None:
    notebook = read_notebook()
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    visible_code_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and not cell.metadata.get("jupyter", {}).get("source_hidden")
    ]
    line_counts = [len(source.splitlines()) for source in code_cells]

    assert max(line_counts) <= 26
    assert sum(count <= 6 for count in line_counts) > len(line_counts) / 2
    assert max(
        len(line)
        for source in visible_code_cells
        for line in source.splitlines()
    ) <= 80


def test_notebook_hides_repository_and_validation_plumbing() -> None:
    notebook = read_notebook()
    source = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type != "code"
        or "hide-input" not in cell.metadata.get("tags", [])
    )
    hidden_terms = {
        "if ROOT is None",
        "FileNotFoundError",
        "runtime-pins.toml",
        "sha256",
        "checksum",
        "TemporaryDirectory",
        "molecule_source_path",
    }

    present = {term for term in hidden_terms if term in source}
    assert not present, f"internal setup leaked into notebook: {sorted(present)}"


def test_learner_text_has_no_internal_stage_language_or_inflight_scope() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    for internal_term in (
        "core_action",
        "action ID",
        "timing band",
        "validation gate",
        "TECHNICALLY VALIDATED",
        "stage card",
    ):
        assert internal_term.lower() not in markdown.lower()
    assert "inflight" not in markdown.lower()
    assert "profiling" not in markdown.lower()
    assert "effective_read_window" not in source
    assert '"reader window"' not in source


def test_prerequisite_and_handoff_links_are_present() -> None:
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "../00-core-playbook/alchemi-core-playbook.ipynb" in source
    assert "../01-atomicdata-batch/atomicdata-and-batch.ipynb" in source
    assert (
        "https://nvidia.github.io/nvalchemi-toolkit/userguide/datapipes.html"
        in source
    )
    assert (
        "https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/"
        "02_trajectory_zarr_io.html"
    ) in source
    assert "../03-model-interfaces-composition/model-interfaces-composition.ipynb" in source


def test_source_notebook_is_clean() -> None:
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(not cell.get("outputs") for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
