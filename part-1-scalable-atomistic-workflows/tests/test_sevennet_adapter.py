"""Focused tests for the raw SevenNet-Omni Toolkit adapter."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")
ase = pytest.importorskip("ase")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from torch import nn  # noqa: E402

from aux.models.sevennet import (  # noqa: E402
    SevenNetOmniWrapper,
    _model_device_and_dtype,
    _toolkit_batch_to_sevennet_graph,
)


class _FakeAtomGraphData(dict):
    """Small mapping with the fields used from SevenNet's AtomGraphData."""

    def __init__(self, x=None, edge_index=None, pos=None, edge_attr=None, **kwargs):
        super().__init__(
            x=x,
            edge_index=edge_index,
            pos=pos,
            edge_attr=edge_attr,
            **kwargs,
        )

    @property
    def x(self):
        return self["x"]


class _FakeSevenNetModel(nn.Module):
    """Raw-model stand-in that uses SevenNet 0.13 field names."""

    cutoff = 4.5
    type_map = {1: 0, 6: 1, 7: 2, 8: 3, 29: 4}
    modal_map = {"mpa": 0, "oc20": 1}
    eval_type_map = True
    eval_modal_map = True
    key_grad = "edge_vec"

    def __init__(
        self,
        *,
        cutoff=4.5,
        type_map=None,
        modal_map=None,
        eval_type_map=True,
        eval_modal_map=True,
        key_grad="edge_vec",
        malformed_field: str | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.type_map = (
            {1: 0, 6: 1, 7: 2, 8: 3, 29: 4}
            if type_map is None
            else type_map
        )
        self.modal_map = (
            {"mpa": 0, "oc20": 1} if modal_map is None else modal_map
        )
        self.eval_type_map = eval_type_map
        self.eval_modal_map = eval_modal_map
        self.key_grad = key_grad
        self.malformed_field = malformed_field
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.batch_mode_calls: list[bool] = []
        self.last_graph = None
        self.forward_calls = 0

    def set_is_batch_data(self, enabled: bool) -> None:
        self.batch_mode_calls.append(enabled)

    def forward(self, graph):
        self.forward_calls += 1
        self.last_graph = graph
        graph["edge_vec"].requires_grad_(True)
        n_graphs = graph["num_atoms"].numel()
        node_energy = graph["pos"].square().sum(dim=1)
        energy = torch.zeros(
            n_graphs,
            dtype=node_energy.dtype,
            device=node_energy.device,
        ).index_add_(0, graph["batch"], node_energy)
        if self.malformed_field == "energy":
            energy = torch.zeros(1, dtype=energy.dtype, device=energy.device)
        forces = -2.0 * graph["pos"]
        if self.malformed_field == "forces":
            forces = forces[:-1]
        graph["inferred_total_energy"] = energy
        graph["inferred_force"] = forces
        return graph


class _TestSevenNetOmniWrapper(SevenNetOmniWrapper):
    """Use the pure converter with a small in-memory graph container."""

    def adapt_input(self, data: Batch, **kwargs):
        del kwargs
        device, dtype = _model_device_and_dtype(self.model)
        return _toolkit_batch_to_sevennet_graph(
            data,
            device=device,
            dtype=dtype,
            modality=self.modality,
            supported_atomic_numbers=self._supported_atomic_numbers,
            atom_graph_factory=_FakeAtomGraphData,
        )


def _atomic_data(
    numbers: list[int],
    positions: list[list[float]],
    *,
    cell: list[list[float]] | None = None,
    pbc: list[bool] | None = None,
    edges: list[list[int]],
    shifts: list[list[int]] | None,
) -> AtomicData:
    atoms = ase.Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc if pbc is not None else False,
    )
    data = AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
    data.add_edge_property(
        "neighbor_list",
        torch.tensor(edges, dtype=torch.int32),
    )
    if shifts is not None:
        data.add_edge_property(
            "neighbor_list_shifts",
            torch.tensor(shifts, dtype=torch.int32),
        )
    return data


def _finite_batch() -> Batch:
    co = _atomic_data(
        [6, 8],
        [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]],
        edges=[[0, 1], [1, 0]],
        shifts=None,
    )
    ammonia = _atomic_data(
        [7, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.94, 0.0, 0.0],
            [-0.31, 0.89, 0.0],
            [-0.31, -0.45, 0.78],
        ],
        edges=[
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
            [0, 3],
            [3, 0],
        ],
        shifts=None,
    )
    return Batch.from_data_list([co, ammonia], device="cpu")


def _periodic_batch() -> Batch:
    first = _atomic_data(
        [29, 6, 8],
        [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [3.5, 1.2, 0.0]],
        cell=[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        pbc=[True, True, False],
        edges=[[0, 1], [1, 0], [1, 2], [2, 1]],
        shifts=[[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    second = _atomic_data(
        [29, 7, 1],
        [[0.0, 0.0, 0.0], [0.0, 2.7, 0.0], [0.0, 3.6, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        pbc=[True, True, False],
        edges=[[0, 1], [1, 0], [1, 2], [2, 1]],
        shifts=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    return Batch.from_data_list([first, second], device="cpu")


def _wrapper(**model_options) -> tuple[SevenNetOmniWrapper, _FakeSevenNetModel]:
    model = _FakeSevenNetModel(**model_options)
    return (
        _TestSevenNetOmniWrapper(
            model,
            modality="oc20",
        ),
        model,
    )


def test_constructor_exposes_only_production_dependencies() -> None:
    parameters = inspect.signature(SevenNetOmniWrapper.__init__).parameters

    assert tuple(parameters) == ("self", "model", "modality")


def test_config_declares_sevennet_energy_force_and_periodic_requirements() -> None:
    wrapper, raw_model = _wrapper()
    config = wrapper.model_config

    assert raw_model.batch_mode_calls == [True]
    assert not raw_model.training
    assert wrapper.cutoff == pytest.approx(4.5)
    assert wrapper.modality == "oc20"
    assert config.outputs == frozenset({"energy", "forces"})
    assert config.active_outputs == {"energy", "forces"}
    assert config.autograd_outputs == frozenset()
    assert config.autograd_inputs == frozenset()
    assert config.supports_pbc
    assert not config.needs_pbc
    assert config.optional_inputs == frozenset(
        {"cell", "pbc", "neighbor_list_shifts"}
    )
    assert config.neighbor_config is not None
    assert config.neighbor_config.cutoff == pytest.approx(4.5)
    assert config.neighbor_config.format is NeighborListFormat.COO
    assert not config.neighbor_config.half_list
    assert config.neighbor_config.skin == pytest.approx(0.0)
    assert wrapper.direct_derivative_keys() == {"forces"}
    assert wrapper.input_data() == {
        "positions",
        "atomic_numbers",
        "neighbor_list",
    }


def test_periodic_ragged_batch_maps_edges_vectors_volume_and_modality() -> None:
    batch = _periodic_batch()
    wrapper, _ = _wrapper()

    graph = wrapper.adapt_input(batch)

    assert graph["edge_index"].shape == (2, 8)
    assert graph["edge_index"].dtype is torch.long
    assert graph["batch"].tolist() == [0, 0, 0, 1, 1, 1]
    assert graph["num_atoms"].tolist() == [3, 3]
    assert graph["data_modality"] == ["oc20", "oc20"]
    torch.testing.assert_close(
        graph["cell_volume"],
        torch.tensor([120.0, 210.0]),
    )

    # First edge crosses the x boundary: 3.5 - 0.0 - 4.0 = -0.5 A.
    torch.testing.assert_close(
        graph["edge_vec"][0],
        torch.tensor([-0.5, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        graph["edge_vec"][1],
        torch.tensor([0.5, 0.0, 0.0]),
    )
    torch.testing.assert_close(graph["pos"], batch.positions)
    assert graph["pos"].data_ptr() != batch.positions.data_ptr()


def test_finite_batch_uses_zero_volume_without_fabricating_periodicity() -> None:
    batch = _finite_batch()
    wrapper, _ = _wrapper()

    graph = wrapper.adapt_input(batch)

    torch.testing.assert_close(graph["cell_volume"], torch.zeros(2))
    expected = graph["pos"][graph["edge_index"][1]] - graph["pos"][
        graph["edge_index"][0]
    ]
    torch.testing.assert_close(graph["edge_vec"], expected)


def test_forward_calls_raw_model_once_and_maps_shapes() -> None:
    batch = _periodic_batch()
    wrapper, raw_model = _wrapper()

    outputs = wrapper(batch)

    assert raw_model.forward_calls == 1
    assert outputs["energy"].shape == (2, 1)
    assert outputs["forces"].shape == batch.positions.shape
    assert not outputs["energy"].requires_grad
    assert not outputs["forces"].requires_grad
    assert not batch.positions.requires_grad
    torch.testing.assert_close(outputs["forces"], -2.0 * batch.positions)


def test_sequential_wrappers_share_weights_without_leaking_task_selection() -> None:
    batch = _periodic_batch()
    raw_model = _FakeSevenNetModel()
    mpa = _TestSevenNetOmniWrapper(raw_model, modality="mpa")
    oc20 = _TestSevenNetOmniWrapper(raw_model, modality="oc20")

    first_mpa = mpa(batch.clone())
    assert raw_model.last_graph["data_modality"] == ["mpa", "mpa"]

    oc20(batch.clone())
    assert raw_model.last_graph["data_modality"] == ["oc20", "oc20"]

    second_mpa = mpa(batch.clone())
    assert raw_model.last_graph["data_modality"] == ["mpa", "mpa"]
    assert raw_model.forward_calls == 3
    torch.testing.assert_close(second_mpa["energy"], first_mpa["energy"])
    torch.testing.assert_close(second_mpa["forces"], first_mpa["forces"])


def test_wrapper_to_dtype_controls_converted_graph_dtype() -> None:
    batch = _periodic_batch()
    wrapper, _ = _wrapper()

    wrapper.to(dtype=torch.float64)
    graph = wrapper.adapt_input(batch)

    assert graph["pos"].dtype is torch.float64
    assert graph["edge_vec"].dtype is torch.float64
    assert graph["cell_volume"].dtype is torch.float64
    assert graph["atomic_numbers"].dtype is torch.long


@pytest.mark.parametrize("modality", ["", " oc20", "OC20", "unknown"])
def test_modality_must_be_explicit_and_match_the_checkpoint(modality: str) -> None:
    with pytest.raises(ValueError, match="modality"):
        SevenNetOmniWrapper(_FakeSevenNetModel(), modality=modality)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("cutoff", None, "cutoff"),
        ("type_map", {}, "type_map"),
        ("modal_map", {}, "modal_map"),
        ("eval_type_map", False, "eval_type_map"),
        ("eval_modal_map", False, "eval_modal_map"),
        ("key_grad", "pos", "edge_vec"),
    ],
)
def test_incompatible_raw_model_is_rejected(
    attribute: str,
    value,
    message: str,
) -> None:
    model = _FakeSevenNetModel(**{attribute: value})

    with pytest.raises((TypeError, ValueError), match=message):
        SevenNetOmniWrapper(model, modality="oc20")


def test_unsupported_element_is_rejected() -> None:
    batch = _finite_batch()
    batch.atomic_numbers[0] = 79
    wrapper, _ = _wrapper()

    with pytest.raises(ValueError, match="atomic numbers|does not support"):
        wrapper.adapt_input(batch)


def test_periodic_batch_requires_neighbor_shifts() -> None:
    batch = _periodic_batch()
    del batch["neighbor_list_shifts"]
    wrapper, _ = _wrapper()

    with pytest.raises(KeyError, match="neighbor_list_shifts"):
        wrapper.adapt_input(batch)


def test_periodic_batch_requires_a_valid_cell() -> None:
    batch = _periodic_batch()
    batch.cell[0].zero_()
    wrapper, _ = _wrapper()

    with pytest.raises(ValueError, match="right-handed nonzero cell"):
        wrapper.adapt_input(batch)


def test_padding_sentinel_is_removed_before_sevennet_gathers() -> None:
    batch = _periodic_batch()
    batch.neighbor_list[-1] = batch.num_nodes
    wrapper, _ = _wrapper()

    graph = wrapper.adapt_input(batch)

    assert graph["edge_index"].shape == (2, 7)
    assert int(graph["edge_index"].max()) < batch.num_nodes


def test_cross_graph_edge_is_rejected() -> None:
    batch = _periodic_batch()
    batch.neighbor_list[0] = torch.tensor([0, 3], dtype=torch.int32)
    wrapper, _ = _wrapper()

    with pytest.raises(ValueError, match="different structures"):
        wrapper.adapt_input(batch)


@pytest.mark.parametrize("bad_field", ["energy", "forces"])
def test_malformed_raw_output_is_rejected(
    bad_field: str,
) -> None:
    batch = _periodic_batch()
    wrapper, _ = _wrapper(malformed_field=bad_field)

    with pytest.raises(ValueError, match="wrong number|wrong shape"):
        wrapper(batch)
