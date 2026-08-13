from __future__ import annotations

import ast
import sys
from collections import OrderedDict
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import torch
from ase import Atoms, units
from IPython.core.inputtransformer2 import TransformerManager
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_FILE = NOTEBOOK_DIR / "model-interfaces-composition.ipynb"
sys.path.insert(0, str(NOTEBOOK_DIR))

import helpers

AIMNET_CHECKPOINT_SOURCE = (
    "aimnet = AIMNet2Wrapper.from_checkpoint(\n"
    "    helpers.model_checkpoint(), device=device, compile_model=False\n"
    ").eval()\n"
    "aimnet = helpers.freeze_model(aimnet)"
)


def read_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_FILE, as_version=4)
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


def notebook_types(*names: str) -> tuple[type, ...]:
    notebook = read_notebook()
    indices = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(f"class {name}" in cell.source for name in names)
    ]
    if len(indices) != len(names):
        raise AssertionError(f"could not find notebook classes {names}")
    sources = [
        cell.source
        for cell in notebook.cells[min(indices) : max(indices) + 1]
        if cell.cell_type == "code"
    ]
    namespace = {
        "torch": torch,
        "units": units,
        "Batch": Batch,
        "BaseModelMixin": BaseModelMixin,
        "ModelConfig": ModelConfig,
    }
    from nvalchemi.models.base import NeighborConfig, NeighborListFormat

    namespace.update(
        NeighborConfig=NeighborConfig,
        NeighborListFormat=NeighborListFormat,
    )
    exec(compile("\n\n".join(sources), NOTEBOOK_FILE, "exec"), namespace)  # noqa: S102
    return tuple(namespace[name] for name in names)


def one_graph_batch(
    positions: list[list[float]], *, dtype: torch.dtype = torch.float64
) -> Batch:
    atoms = Atoms("H" * len(positions), positions=positions)
    graph = AtomicData.from_atoms(atoms, device="cpu", dtype=dtype)
    return Batch.from_data_list([graph], device="cpu")


class QuadraticDirectModel(torch.nn.Module, BaseModelMixin):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs):
        atom_energy = self.scale * data.positions.square().sum(dim=1)
        energy = torch.zeros(
            data.num_graphs, dtype=data.positions.dtype, device=data.device
        ).index_add(0, data.batch_idx.long(), atom_energy)
        return OrderedDict(
            energy=energy.unsqueeze(-1),
            forces=-2.0 * self.scale * data.positions,
        )


class ChargeProducer(torch.nn.Module, BaseModelMixin):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "charges"}),
            active_outputs={"energy", "charges"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs):
        self.calls.append("producer")
        charges = data.positions[:, 0]
        energy = charges.sum().reshape(1, 1) * 0.0
        return OrderedDict(energy=energy, charges=charges)


class ChargeConsumer(torch.nn.Module, BaseModelMixin):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.seen_charges: torch.Tensor | None = None
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            active_outputs={"energy"},
            required_inputs=frozenset({"partial_charges"}),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs):
        self.calls.append("consumer")
        self.seen_charges = data.partial_charges
        atom_energy = data.partial_charges.square()
        energy = torch.zeros(
            data.num_graphs, dtype=data.positions.dtype, device=data.device
        ).index_add(0, data.batch_idx.long(), atom_energy)
        return OrderedDict(energy=energy.unsqueeze(-1))


def test_checked_nci_triplet_and_public_batch_fields() -> None:
    records = helpers.load_nci_records(ROOT)
    selected = records[
        records["system_id"].eq("1.041") & records["scale"].eq(1.0)
    ].reset_index(drop=True)
    batch = helpers.build_batch(selected, device="cpu")

    assert selected["fragment"].tolist() == ["AB", "A", "B"]
    assert selected["system_name"].nunique() == 1
    assert selected["system_name"].iat[0] == "phenol - N-methylacetamide"
    assert selected["interaction_class"].iat[0] == "neutral hydrogen bond"
    assert selected["natoms"].tolist() == [25, 13, 12]
    assert batch.num_graphs == 3
    assert batch.num_nodes == 50
    assert batch.charge.reshape(-1).tolist() == [0, 0, 0]


def test_ab_minus_a_minus_b_is_order_aware() -> None:
    values = torch.tensor([2.0, 10.0, 3.0])
    fragments = ["A", "AB", "B"]

    assert helpers.ab_minus_a_minus_b(values, fragments).item() == pytest.approx(5.0)

    with pytest.raises(ValueError, match="one AB, A, and B"):
        helpers.ab_minus_a_minus_b(values[:2], fragments[:2])


def test_model_and_output_tables_report_live_public_contracts() -> None:
    model = torch.nn.Linear(3, 1)
    from nvalchemi.models.base import NeighborConfig, NeighborListFormat

    model.model_config = ModelConfig(
        outputs=frozenset({"energy", "forces"}),
        active_outputs={"energy"},
        required_inputs=frozenset({"partial_charges"}),
        neighbor_config=NeighborConfig(
            cutoff=8.0,
            format=NeighborListFormat.COO,
            half_list=False,
        ),
    )
    model.input_data = lambda: {
        "positions",
        "atomic_numbers",
        "neighbor_list",
        "partial_charges",
    }
    contract = helpers.model_contract_table(model)
    outputs = helpers.output_contract_table(
        {
            "energy": torch.zeros(2, 1),
            "forces": torch.zeros(5, 3),
            "charges": torch.zeros(5),
        },
        num_graphs=2,
        num_nodes=5,
    )

    assert contract.loc["active outputs", "value"] == "energy"
    assert contract.loc["neighbor format", "value"] == "coo"
    assert contract.loc["neighbor convention", "value"] == "full list"
    assert "partial_charges" in contract.loc["required inputs", "value"]
    assert outputs.loc["energy", ["shape", "level", "unit"]].tolist() == [
        (2, 1),
        "graph",
        "eV",
    ]
    assert outputs.loc["forces", "unit"] == "eV/Å"
    assert outputs.loc["charges", "expected rows"] == 5


def test_component_plot_is_bounded_and_accessible() -> None:
    values = pd.Series(
        {
            "AIMNet2 checkpoint base": -8.0,
            "finite Coulomb": -1.0,
            "D3(BJ)": -2.0,
            "complete model": -11.0,
        }
    )

    html = helpers.component_plot_html(values, reference_kcal_mol=-11.9)

    assert 'src="data:image/png;base64,' in html
    assert 'alt="Bar chart of AB minus A minus B interaction energies' in html
    assert "kcal/mol" in html
    assert "CCSD(T)/CBS" in html


def test_runtime_assets_are_verified_without_lesson_cache_files() -> None:
    checkpoint = helpers.model_checkpoint()
    d3_file = helpers.d3_parameter_file()

    assert helpers.MODEL_ALIAS == "aimnet2-wb97m-d3_0"
    assert checkpoint.is_file()
    assert d3_file.is_file()
    assert ROOT not in d3_file.parents
    assert not list(NOTEBOOK_DIR.rglob("*.pt"))


def test_missing_d3_asset_fails_without_creating_a_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "dftd3_parameters.pt"
    monkeypatch.setenv("ALCHEMI_D3_PARAM_FILE", str(missing))

    with pytest.raises(RuntimeError, match="D3 parameter"):
        helpers.d3_parameter_file()

    assert not missing.exists()


def test_notebook_quadratic_wrapper_matches_native_energy_and_forces() -> None:
    native_type, wrapper_type = notebook_types("NativeQuadratic", "QuadraticWrapper")
    native = native_type(scale=0.25)
    wrapper = wrapper_type(native).eval()
    batch = one_graph_batch([[1.0, 0.0, 0.0], [-2.0, 0.5, 0.0]])
    native_positions = batch.positions.detach().clone().requires_grad_(True)
    native_energy = native(
        coordinates=native_positions,
        graph_index=batch.batch_idx,
        graph_count=batch.num_graphs,
    )["native_energy"]
    native_forces = -torch.autograd.grad(native_energy.sum(), native_positions)[0]

    wrapped = wrapper(batch)

    torch.testing.assert_close(wrapped["energy"], native_energy)
    torch.testing.assert_close(wrapped["forces"], native_forces)


def test_notebook_coulomb_adapter_uses_public_field_addition_and_matches_native() -> (
    None
):
    native_type, adapter_type = notebook_types("NativeCoulomb", "DirectCoulombAdapter")
    from nvalchemi.neighbors import compute_neighbors

    adapter = adapter_type(native_type(), cutoff=5.0).eval()
    charges = torch.tensor([0.4, -0.4], dtype=torch.float64)
    native_batch = one_graph_batch([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    native_batch.add_key("partial_charges", [charges.clone()], level="node")
    native_batch.positions.requires_grad_(True)
    compute_neighbors(native_batch, config=adapter.model_config.neighbor_config)
    native = adapter.model(
        positions=native_batch.positions,
        partial_charges=native_batch.partial_charges,
        neighbor_pairs=native_batch.neighbor_list,
        batch_idx=native_batch.batch_idx,
        num_graphs=native_batch.num_graphs,
    )
    native_forces = -torch.autograd.grad(
        native["native_energy"].sum(), native_batch.positions
    )[0]

    adapter_batch = one_graph_batch([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    adapter_batch.add_key("partial_charges", [charges.clone()], level="node")
    compute_neighbors(adapter_batch, config=adapter.model_config.neighbor_config)
    adapted = adapter(adapter_batch)

    torch.testing.assert_close(adapted["energy"], native["native_energy"])
    torch.testing.assert_close(adapted["forces"], native_forces)


def test_real_addition_and_wiring_behaviors_match_documented_semantics() -> None:
    first = QuadraticDirectModel(1.5)
    second = QuadraticDirectModel(0.5)
    combined = first + second
    batch = one_graph_batch([[1.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])

    combined_output = combined(batch)
    first_output = first(batch)
    second_output = second(batch)

    assert [group.use_autograd for group in combined.groups] == [False, False]
    torch.testing.assert_close(
        combined_output["energy"],
        first_output["energy"] + second_output["energy"],
    )
    torch.testing.assert_close(
        combined_output["forces"],
        first_output["forces"] + second_output["forces"],
    )

    calls: list[str] = []
    producer = ChargeProducer(calls)
    consumer = ChargeConsumer(calls)
    pipeline = PipelineModelWrapper(
        groups=[
            PipelineGroup(
                steps=[
                    PipelineStep(producer, wire={"charges": "partial_charges"}),
                    consumer,
                ],
                use_autograd=True,
            )
        ]
    ).eval()
    pipeline.set_config("active_outputs", {"energy", "forces", "charges"})
    wired = pipeline(one_graph_batch([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]))

    assert calls == ["producer", "consumer"]
    torch.testing.assert_close(
        consumer.seen_charges,
        torch.tensor([1.0, -2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        wired["forces"],
        torch.tensor([[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
    )


def test_notebook_schema_namespace_and_cell_hygiene() -> None:
    notebook = read_notebook()
    transformed = "\n\n".join(
        TransformerManager().transform_cell(cell.source)
        for cell in code_cells(notebook)
    )
    ast.parse(transformed)

    assert len(notebook.cells) <= 65
    assert all(not cell.outputs for cell in code_cells(notebook))
    assert all(cell.execution_count is None for cell in code_cells(notebook))
    cell_ids = [cell.id for cell in notebook.cells]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    assert notebook.cells[2].metadata["jupyter"]["source_hidden"] is True
    assert notebook.cells[2].metadata["jupyter"]["outputs_hidden"] is True
    assert "hide-input" in notebook.cells[2].metadata["tags"]
    assert "remove-output" in notebook.cells[2].metadata["tags"]
    assert AIMNET_CHECKPOINT_SOURCE in code_source(notebook)


def test_notebook_is_synthetic_first_and_uses_public_model_paths() -> None:
    source = code_source(read_notebook())
    required = {
        "AtomicData.from_atoms(",
        "Batch.from_data_list(",
        ".add_key(",
        "AIMNet2Wrapper.from_checkpoint(",
        ".model_config",
        ".set_config(",
        "compute_neighbors(",
        ".make_neighbor_hooks(",
        "BaseModelMixin",
        "ModelConfig(",
        "NeighborConfig(",
        "NeighborListFormat.COO",
        "adapt_input(",
        "adapt_output(",
        " + toy_correction",
        "PipelineStep(",
        "PipelineGroup(",
        "PipelineModelWrapper(",
    }
    missing = {token for token in required if token not in source}
    assert not missing, f"missing visible public APIs: {sorted(missing)}"

    synthetic = source.index("class NativeQuadratic")
    supported = source.index("aimnet = AIMNet2Wrapper.from_checkpoint(")
    real_adapter = source.index("class NativeCoulomb")
    composition = source.index("full_model = PipelineModelWrapper(")
    assert synthetic < supported < real_adapter < composition
    assert 'wire={"charges": "partial_charges"}' in source
    assert 'neighbor_adaptation="always"' in source
    assert "auto_download=False" in source


def test_notebook_avoids_private_or_fabricated_paths() -> None:
    notebook = read_notebook()
    source = code_source(notebook)
    forbidden = {
        "object.__setattr__",
        "._models",
        "._neighbor",
        "._storage",
        "nvalchemi._",
        "HookContext",
        "SimpleNamespace",
        "get_model_path",
    }
    present = {token for token in forbidden if token in source}

    assert not present, f"forbidden implementation paths: {sorted(present)}"
    assert "helpers.prepare_model_neighbors" not in source


def test_cell_pacing_and_learner_prose_are_direct() -> None:
    notebook = read_notebook()
    oversized = {
        index: len(cell.source.splitlines())
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and len(cell.source.splitlines()) > 24
        and cell.metadata.get("jupyter", {}).get("source_hidden") is not True
        and not any(
            token in cell.source
            for token in (
                "class NativeQuadratic",
                "class QuadraticWrapper",
                "class NativeCoulomb",
                "class DirectCoulombAdapter",
            )
        )
    }
    markdown = markdown_source(notebook)
    prose_without_asset_name = markdown.replace("curriculum-map-03.svg", "")
    banned = {
        "N03-REQ",
        "capability level",
        "timing band",
        "provenance",
        "action ID",
        "let's dive",
        "at its core",
        "game changer",
        "robust",
        "seamless",
        "As you can see",
        "—",
    }

    assert not oversized, f"non-wrapper cells exceed pacing limit: {oversized}"
    assert not {
        phrase
        for phrase in banned
        if phrase.lower() in prose_without_asset_name.lower()
    }


def test_scientific_scope_charge_and_component_closure_are_explicit() -> None:
    notebook = read_notebook()
    source = code_source(notebook)
    markdown = markdown_source(notebook)

    for token in (
        "predicted_graph_charge",
        "charge_residual_e",
        "assert charge_residual_e <=",
        "component_closure_eV",
        "assert component_closure_eV <=",
        "ab_minus_a_minus_b",
    ):
        assert token in source
    for phrase in (
        "finite, nonperiodic",
        "architectural model terms",
        "one fixed geometry",
        "does not establish accuracy",
        "do not support a condensed-phase claim",
    ):
        assert phrase in markdown
    assert "quantum energy decomposition" not in markdown.lower()


def test_visuals_links_exercise_and_recap_complete_the_lesson() -> None:
    markdown = markdown_source(read_notebook())
    source = code_source(read_notebook())

    assert "../../shared/alchemi-banner-left.png" in markdown
    assert "../../shared/curriculum-map-03.svg" in markdown
    assert "flowchart LR" in markdown
    assert "AIMNet2 charges" in markdown
    assert "component_plot_html(" in source
    assert "## Try it" in markdown
    assert "exercise_outputs" in source
    assert "## Recap" in markdown
    assert "../00-core-playbook/alchemi-core-playbook.ipynb" in markdown
    assert "../04-hooks/hooks.ipynb" in markdown
    assert "../05-base-dynamics/base-dynamics.ipynb" in markdown
    assert "../07-training-finetuning/training-finetuning.ipynb" in markdown
    assert "in progress" in markdown
    for url in (
        "https://nvidia.github.io/nvalchemi-toolkit/models/index.html",
        "https://nvidia.github.io/nvalchemi-toolkit/userguide/models.html",
        "07_composable_model_composition.html",
        "08_aimnet2_ewald_pipeline.html",
    ):
        assert url in markdown
