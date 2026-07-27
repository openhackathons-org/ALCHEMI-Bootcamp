"""Focused tests for SevenNet graph and output checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.models.sevennet import SevenNetOmniWrapper  # noqa: E402
from aux.models.sevennet_checks import (  # noqa: E402
    build_sevennet_model_card,
    build_sevennet_settings_table,
    build_sevennet_mapping_table,
    build_sevennet_repeat_table,
    split_model_outputs,
    summarize_sevennet_task_outputs,
)


class _ModelCardModel(torch.nn.Module):
    cutoff = 4.5
    modal_map = {"mpa": 0}
    eval_type_map = True
    eval_modal_map = True
    key_grad = "edge_vec"

    def __init__(
        self,
        *,
        type_map=None,
        dtype=torch.float32,
        buffer_dtype=None,
    ) -> None:
        super().__init__()
        self.type_map = (
            {29: 4, 8: 3, 1: 0, 7: 2, 6: 1}
            if type_map is None
            else type_map
        )
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        if buffer_dtype is not None:
            self.register_buffer(
                "floating_buffer",
                torch.ones(1, dtype=buffer_dtype),
            )

    def set_is_batch_data(self, enabled: bool) -> None:
        self.batch_mode = enabled


def _model_card_wrapper(model, **config_overrides):
    config_values = {
        "required_inputs": frozenset(),
        "optional_inputs": frozenset(
            {"cell", "pbc", "neighbor_list_shifts"}
        ),
        "outputs": frozenset({"energy", "forces"}),
    }
    config_values.update(config_overrides)
    return SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(**config_values),
    )


def test_model_card_reads_live_model_and_declared_wrapper_facts() -> None:
    model = _ModelCardModel()
    wrapper = SevenNetOmniWrapper(model, modality="mpa")

    table = build_sevennet_model_card(model=model, wrapper=wrapper)

    assert model.batch_mode is True
    assert table.columns.tolist() == ["Field", "Value"]
    card = table.set_index("Field")["Value"]
    assert card["Supported elements"] == (
        "5 elements: H (Z=1), C (Z=6), N (Z=7), O (Z=8), Cu (Z=29)"
    )
    assert card["Model dtype"] == "float32"
    assert card["Charge behavior"] == "not consumed as input; not returned"
    assert card["Spin behavior"] == "not consumed as input; not returned"
    assert card["Multiplicity behavior"] == (
        "not consumed as input; not returned"
    )


def test_model_card_reports_required_optional_and_returned_fields() -> None:
    model = _ModelCardModel()
    wrapper = _model_card_wrapper(
        model,
        required_inputs=frozenset({"charge"}),
        optional_inputs=frozenset({"spin"}),
        outputs=frozenset({"energy", "charges", "spin_multiplicity"}),
    )

    card = build_sevennet_model_card(
        model=model,
        wrapper=wrapper,
    ).set_index("Field")["Value"]

    assert card["Charge behavior"] == "required input (charge); returned (charges)"
    assert card["Spin behavior"] == "optional input (spin); not returned"
    assert card["Multiplicity behavior"] == (
        "not consumed as input; returned (spin_multiplicity)"
    )


def test_model_card_fails_when_required_facts_are_unavailable() -> None:
    missing_elements = _ModelCardModel(type_map={})
    with pytest.raises(ValueError, match="type_map.*supported elements"):
        build_sevennet_model_card(
            model=missing_elements,
            wrapper=_model_card_wrapper(missing_elements),
        )

    no_parameters = torch.nn.Module()
    no_parameters.type_map = {1: 0}
    with pytest.raises(ValueError, match="no parameters.*dtype"):
        build_sevennet_model_card(
            model=no_parameters,
            wrapper=_model_card_wrapper(no_parameters),
        )

    model = _ModelCardModel()
    with pytest.raises(ValueError, match="must declare outputs"):
        build_sevennet_model_card(
            model=model,
            wrapper=SimpleNamespace(
                model=model,
                model_config=SimpleNamespace(
                    required_inputs=frozenset(),
                    optional_inputs=frozenset(),
                ),
            ),
        )


def test_model_card_rejects_ambiguous_model_or_wrapper_metadata() -> None:
    mixed_dtype = _ModelCardModel(buffer_dtype=torch.float64)
    with pytest.raises(ValueError, match="one floating-point dtype"):
        build_sevennet_model_card(
            model=mixed_dtype,
            wrapper=_model_card_wrapper(mixed_dtype),
        )

    model = _ModelCardModel()
    with pytest.raises(ValueError, match="both required and optional"):
        build_sevennet_model_card(
            model=model,
            wrapper=_model_card_wrapper(
                model,
                required_inputs=frozenset({"charge"}),
                optional_inputs=frozenset({"charge"}),
            ),
        )

    with pytest.raises(ValueError, match="wrapper.model"):
        build_sevennet_model_card(
            model=model,
            wrapper=_model_card_wrapper(_ModelCardModel()),
        )


def test_settings_table_records_model_d3_and_license_choices() -> None:
    table = build_sevennet_settings_table(
        model_name="SevenNet-Omni",
        modality="mpa",
        reference_method="PBE(+U)",
        package_version="0.11.0",
        checkpoint_sha256="a" * 64,
        checkpoint_record="10.0000/example",
        model_cutoff_A=6.0,
        supports_pbc=True,
        outputs=("forces", "energy"),
        d3_reference_cutoff_bohr=95.0,
        d3_cutoff_A=50.27,
        d3_smoothing_fraction=0.1,
    ).set_index("Setting")["Value"]

    assert table["Neighbor list"] == "COO, full directed"
    assert table["D3 parameters"] == "pairwise PBE-D3(BJ)"
    assert table["D3 cutoff"] == "95 bohr / 50.270 Å"
    assert table["SevenNet code license"] == "MIT"
    assert table["Checkpoint record license"] == "CC BY 4.0"
    assert table["Exact checkpoint file license"] == (
        "not separately stated; runtime download only"
    )


def _mapping_inputs():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=torch.float32
    )
    batch = SimpleNamespace(
        atomic_numbers=torch.tensor([29, 6], dtype=torch.int32),
        positions=positions,
        batch_idx=torch.tensor([0, 0], dtype=torch.int32),
        num_nodes_list=[2],
        num_nodes=2,
        num_graphs=1,
        neighbor_list=torch.tensor(
            [[0, 1], [1, 0], [2, 2]], dtype=torch.int32
        ),
        neighbor_list_shifts=torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.int32
        ),
        cell=torch.tensor(
            [[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]],
            dtype=torch.float32,
        ),
    )
    graph = {
        "atomic_numbers": batch.atomic_numbers.long(),
        "pos": positions.clone(),
        "batch": batch.batch_idx.long(),
        "num_atoms": torch.tensor([2], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_vec": torch.tensor(
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32
        ),
        "cell_volume": torch.tensor([120.0], dtype=torch.float32),
        "data_modality": ["mpa"],
    }
    wrapper = SimpleNamespace(
        modality="mpa",
        adapt_input=lambda _batch: graph,
    )
    return wrapper, batch, graph


def test_mapping_table_checks_all_visible_graph_fields() -> None:
    wrapper, batch, _ = _mapping_inputs()

    table = build_sevennet_mapping_table(wrapper, batch)

    assert table["component"].tolist() == [
        "atomic numbers",
        "positions",
        "graph ownership",
        "atoms per graph",
        "directed COO edges",
        "periodic edge vectors",
        "cell volumes",
        "model task",
    ]
    assert table["exact_match"].tolist() == [True] * 8
    assert table["max_abs_difference"].tolist() == pytest.approx([0.0] * 8)
    assert table.loc[
        table["component"] == "periodic edge vectors", "units"
    ].item() == "Å"


def test_mapping_table_reports_numerical_and_task_mismatches() -> None:
    wrapper, batch, graph = _mapping_inputs()
    graph["edge_vec"][0, 0] += 2.0e-6
    graph["data_modality"] = ["oc20"]

    table = build_sevennet_mapping_table(wrapper, batch).set_index("component")

    assert not bool(table.loc["periodic edge vectors", "exact_match"])
    assert table.loc[
        "periodic edge vectors", "max_abs_difference"
    ] == pytest.approx(2.0e-6, abs=1.0e-8)
    assert not bool(table.loc["model task", "exact_match"])


def test_mapping_table_rejects_missing_or_nontensor_fields() -> None:
    wrapper, batch, graph = _mapping_inputs()
    del graph["edge_vec"]

    with pytest.raises(KeyError, match="edge_vec"):
        build_sevennet_mapping_table(wrapper, batch)

    wrapper, batch, graph = _mapping_inputs()
    graph["pos"] = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="pos must be a torch.Tensor"):
        build_sevennet_mapping_table(wrapper, batch)


def test_repeat_table_reports_each_graph_separately() -> None:
    first = {
        "energy": torch.tensor([[-1.0], [-3.0]]),
        "forces": torch.zeros((3, 3)),
    }
    second = {
        "energy": torch.tensor([[-0.98], [-2.97]]),
        "forces": torch.tensor(
            [[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        ),
    }

    table = build_sevennet_repeat_table(
        first,
        second,
        labels=["adslab", "gas"],
        atom_counts=[2, 1],
    )

    assert table["structure"].tolist() == ["adslab", "gas"]
    assert table["atoms"].tolist() == [2, 1]
    assert table["energy_difference_eV"].tolist() == pytest.approx([0.02, 0.03])
    assert table["energy_difference_eV_per_atom"].tolist() == pytest.approx(
        [0.01, 0.03]
    )
    assert table[
        "max_force_component_difference_eV_A"
    ].tolist() == pytest.approx([0.2, 0.3])


def test_repeat_table_validates_labels_energies_and_tensor_outputs() -> None:
    outputs = {
        "energy": torch.zeros(1),
        "forces": torch.zeros((1, 3)),
    }
    with pytest.raises(ValueError, match="same length"):
        build_sevennet_repeat_table(
            outputs,
            outputs,
            labels=["one", "two"],
            atom_counts=[1],
        )
    with pytest.raises(ValueError, match="energy outputs"):
        build_sevennet_repeat_table(
            outputs,
            outputs,
            labels=["one", "two"],
            atom_counts=[1, 0],
        )
    with pytest.raises(TypeError, match="first energy must be a torch.Tensor"):
        build_sevennet_repeat_table(
            {"energy": [0.0], "forces": torch.zeros((1, 3))},
            outputs,
            labels=["one"],
            atom_counts=[1],
        )


def test_task_output_summary_labels_each_graph_without_exposing_energy_zeros() -> None:
    batch = SimpleNamespace(
        num_graphs=2,
        num_nodes=3,
        batch_idx=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    outputs = {
        "energy": torch.tensor([[-10.0], [-2.5]]),
        "forces": torch.tensor(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        ),
    }

    rows = summarize_sevennet_task_outputs(
        task="oc20",
        target="RPBE OC20 adsorption data",
        structure_keys=("clean_cu111", "co_on_cu111"),
        batch=batch,
        outputs=outputs,
    )

    assert [row["structure"] for row in rows] == [
        "clean_cu111",
        "co_on_cu111",
    ]
    assert [row["energy output"] for row in rows] == [
        "finite scalar",
        "finite scalar",
    ]
    assert [row["force output"] for row in rows] == ["(2, 3)", "(1, 3)"]
    assert [row["max |F| / eV Å⁻¹"] for row in rows] == pytest.approx([5.0, 2.0])
    assert all("energy / eV" not in row for row in rows)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (
            {"energy": torch.zeros(1), "forces": torch.zeros((3, 3))},
            "number of graphs",
        ),
        (
            {"energy": torch.zeros((2, 1)), "forces": torch.zeros((2, 3))},
            "batched atoms",
        ),
        (
            {
                "energy": torch.tensor([[float("nan")], [0.0]]),
                "forces": torch.zeros((3, 3)),
            },
            "non-finite energy",
        ),
    ],
)
def test_task_output_summary_rejects_invalid_outputs(outputs, message: str) -> None:
    batch = SimpleNamespace(
        num_graphs=2,
        num_nodes=3,
        batch_idx=torch.tensor([0, 0, 1], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match=message):
        summarize_sevennet_task_outputs(
            task="mpa",
            target="PBE(+U)-level data",
            structure_keys=("clean_cu111", "co_on_cu111"),
            batch=batch,
            outputs=outputs,
        )


def test_split_model_outputs_preserves_order_and_copies_force_arrays() -> None:
    outputs = {
        "energy": torch.tensor([[-10.0], [-2.5]]),
        "forces": torch.arange(9, dtype=torch.float32).reshape(3, 3),
    }

    energies, forces = split_model_outputs(
        ["surface", "gas"], [2, 1], outputs
    )

    assert list(energies) == ["surface", "gas"]
    assert energies == {"surface": -10.0, "gas": -2.5}
    np.testing.assert_array_equal(forces["surface"], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(forces["gas"], np.arange(6, 9).reshape(1, 3))
    forces["surface"][0, 0] = -99.0
    assert outputs["forces"][0, 0].item() == 0.0


@pytest.mark.parametrize(
    ("keys", "atom_counts", "outputs", "message"),
    [
        (
            ["one", "two"],
            [1],
            {"energy": torch.zeros(2), "forces": torch.zeros((1, 3))},
            "same length",
        ),
        (
            ["one", "two"],
            [1, 1],
            {"energy": torch.zeros(1), "forces": torch.zeros((2, 3))},
            "graph layout",
        ),
        (
            ["one", "two"],
            [1, 1],
            {"energy": torch.zeros(2), "forces": torch.zeros((3, 3))},
            "graph layout",
        ),
    ],
)
def test_split_model_outputs_validates_graph_layout(
    keys,
    atom_counts,
    outputs,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split_model_outputs(keys, atom_counts, outputs)
