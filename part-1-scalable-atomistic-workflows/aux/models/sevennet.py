"""Toolkit adapter for a raw SevenNet-Omni 0.13 model.

The notebook loads and configures the SevenNet checkpoint itself.  This module
only handles the model boundary: it declares the Toolkit capabilities, converts
Toolkit COO neighbor data to SevenNet's graph fields, calls the raw model, and
maps SevenNet's energy and force keys back to Toolkit names.

SevenNet is imported only when :meth:`SevenNetOmniWrapper.adapt_input` runs.
Keeping that dependency lazy lets the adapter tests use a small fake graph
container without downloading a checkpoint or installing SevenNet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
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


# SevenNet 0.13 public data keys from ``sevenn._keys``.  Keeping these strings
# here avoids importing the optional package when this module is imported.
_ATOMIC_NUMBERS = "atomic_numbers"
_BATCH = "batch"
_CELL_VOLUME = "cell_volume"
_DATA_MODALITY = "data_modality"
_EDGE_INDEX = "edge_index"
_EDGE_VECTOR = "edge_vec"
_NUM_ATOMS = "num_atoms"
_POSITION = "pos"
_PREDICTED_ENERGY = "inferred_total_energy"
_PREDICTED_FORCE = "inferred_force"


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return the raw model's live device and floating-point dtype."""

    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError(
            "SevenNetOmniWrapper requires a raw model with parameters so its "
            "device and dtype are unambiguous."
        ) from exc
    if not parameter.dtype.is_floating_point:
        raise TypeError("SevenNet model parameters must use a floating-point dtype")
    return parameter.device, parameter.dtype


def _periodic_fields(
    data: Batch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize PBC/cell data and return ``(pbc, cell, volume)``.

    A missing cell is valid for a finite batch.  It is an error for a periodic
    batch because silently fabricating a cell would change the atomic system.
    """

    n_graphs = data.num_graphs
    raw_pbc = getattr(data, "pbc", None)
    if raw_pbc is None:
        pbc = torch.zeros((n_graphs, 3), dtype=torch.bool, device=device)
    else:
        if raw_pbc.numel() != n_graphs * 3:
            raise ValueError("Toolkit pbc must contain three flags per structure")
        pbc = raw_pbc.to(device=device, dtype=torch.bool).reshape(n_graphs, 3)

    raw_cell = getattr(data, "cell", None)
    if raw_cell is None:
        if bool(pbc.any()):
            raise ValueError("Periodic SevenNet input requires a cell for every structure")
        cell = torch.zeros((n_graphs, 3, 3), dtype=dtype, device=device)
    else:
        if raw_cell.numel() != n_graphs * 9:
            raise ValueError("Toolkit cell must have shape (structures, 3, 3)")
        cell = raw_cell.to(device=device, dtype=dtype).reshape(n_graphs, 3, 3)
        if not bool(torch.isfinite(cell).all()):
            raise ValueError("SevenNet cells must contain finite values")

    signed_volume = torch.linalg.det(cell)
    periodic_graph = pbc.any(dim=1)
    if bool((signed_volume[periodic_graph] <= 0.0).any()):
        raise ValueError("Every periodic structure needs a right-handed nonzero cell")
    return pbc, cell, signed_volume.abs()


def _toolkit_batch_to_sevennet_graph(
    data: Batch,
    *,
    device: torch.device,
    dtype: torch.dtype,
    modality: str,
    supported_atomic_numbers: torch.Tensor,
    atom_graph_factory: Callable[..., Any] | None = None,
) -> Any:
    """Convert one Toolkit batch to the graph consumed by SevenNet 0.13.

    Toolkit stores COO pairs as ``(edge, [source, target])`` plus integer
    lattice shifts.  SevenNet stores the transposed edge index and Cartesian
    source-to-target vectors.  Full neighbor lists are required: both
    directions of each interaction must already be present.
    """

    if atom_graph_factory is None:
        try:
            from sevenn.atom_graph_data import AtomGraphData
        except ImportError as exc:
            raise ImportError(
                "SevenNetOmniWrapper requires sevenn==0.13.0 at runtime"
            ) from exc
        atom_graph_factory = AtomGraphData

    if not isinstance(data, Batch):
        raise TypeError(
            "SevenNetOmniWrapper expects a Toolkit Batch. Build finite and "
            "periodic batches explicitly before evaluation."
        )
    if data.num_graphs < 1 or data.num_nodes < 1:
        raise ValueError("SevenNet cannot evaluate an empty Toolkit Batch")

    positions = data.positions.to(device=device, dtype=dtype).clone().contiguous()
    if positions.ndim != 2 or positions.shape != (data.num_nodes, 3):
        raise ValueError("Toolkit positions must have shape (atoms, 3)")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("SevenNet positions must contain finite values")

    atomic_numbers = data.atomic_numbers.to(device=device, dtype=torch.long)
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() != data.num_nodes:
        raise ValueError("Toolkit atomic_numbers must contain one value per atom")
    if atomic_numbers.numel():
        smallest = int(atomic_numbers.min())
        largest = int(atomic_numbers.max())
        if smallest < 0 or largest >= supported_atomic_numbers.numel():
            raise ValueError(
                f"atomic numbers [{smallest}, {largest}] fall outside the loaded "
                "SevenNet type map"
            )
        supported = supported_atomic_numbers.index_select(0, atomic_numbers)
        if not bool(supported.all()):
            missing = sorted(set(atomic_numbers[~supported].detach().cpu().tolist()))
            raise ValueError(
                f"loaded SevenNet checkpoint does not support atomic numbers {missing}"
            )

    batch_index = data.batch_idx.to(device=device, dtype=torch.long)
    if batch_index.ndim != 1 or batch_index.numel() != data.num_nodes:
        raise ValueError("Toolkit batch_idx must contain one graph index per atom")
    atom_counts = torch.as_tensor(
        data.num_nodes_list,
        dtype=torch.long,
        device=device,
    )
    if atom_counts.numel() != data.num_graphs or bool((atom_counts <= 0).any()):
        raise ValueError("Every SevenNet structure must contain at least one atom")

    pbc, cell, cell_volume = _periodic_fields(data, device=device, dtype=dtype)

    raw_neighbors = getattr(data, "neighbor_list", None)
    if raw_neighbors is None:
        raise KeyError(
            "Toolkit COO neighbors are missing; call compute_neighbors(...) with "
            "the wrapper's NeighborConfig before SevenNet evaluation."
        )
    neighbors = raw_neighbors.to(device=device)
    if neighbors.ndim != 2 or neighbors.shape[1] != 2:
        raise ValueError("SevenNet expects Toolkit COO neighbors with shape (edges, 2)")

    raw_shifts = getattr(data, "neighbor_list_shifts", None)
    if raw_shifts is None:
        if bool(pbc.any()):
            raise KeyError(
                "Periodic SevenNet input requires Toolkit neighbor_list_shifts"
            )
        shifts = torch.zeros((neighbors.shape[0], 3), dtype=dtype, device=device)
    else:
        if raw_shifts.ndim != 2 or raw_shifts.shape != (neighbors.shape[0], 3):
            raise ValueError(
                "SevenNet expects neighbor_list_shifts with shape (edges, 3)"
            )
        shifts = raw_shifts.to(device=device, dtype=dtype)

    sources = neighbors[:, 0].to(torch.long)
    targets = neighbors[:, 1].to(torch.long)
    n_nodes = data.num_nodes
    if sources.numel():
        out_of_bounds = (
            (sources < 0)
            | (targets < 0)
            | (sources > n_nodes)
            | (targets > n_nodes)
        )
        if bool(out_of_bounds.any()):
            raise ValueError("neighbor-list indices fall outside the Toolkit batch")

        # Toolkit COO buffers can contain N as an unused padding sentinel.  It
        # cannot be passed to SevenNet because SevenNet gathers atom tensors.
        valid = (sources < n_nodes) & (targets < n_nodes)
        sources = sources[valid]
        targets = targets[valid]
        shifts = shifts[valid]

    source_graph = batch_index.index_select(0, sources)
    target_graph = batch_index.index_select(0, targets)
    if not torch.equal(source_graph, target_graph):
        raise ValueError("neighbor-list edges may not connect different structures")

    cartesian_shifts = torch.einsum(
        "ei,eij->ej",
        shifts,
        cell.index_select(0, source_graph),
    )
    edge_vectors = (
        positions.index_select(0, targets)
        - positions.index_select(0, sources)
        + cartesian_shifts
    ).contiguous()
    edge_index = torch.stack((sources, targets), dim=0).contiguous()

    # ``x`` is only a placeholder here.  The raw checkpoint's verified
    # ``eval_type_map=True`` preprocessing maps ``atomic_numbers`` to its own
    # contiguous species indices before the first embedding layer.
    return atom_graph_factory(
        x=atomic_numbers,
        edge_index=edge_index,
        pos=positions,
        **{
            _ATOMIC_NUMBERS: atomic_numbers,
            _EDGE_VECTOR: edge_vectors,
            _CELL_VOLUME: cell_volume,
            _NUM_ATOMS: atom_counts,
            _BATCH: batch_index,
            _DATA_MODALITY: [modality] * data.num_graphs,
        },
    )


def _validated_sevennet_metadata(
    model: nn.Module,
    *,
    modality: str,
) -> tuple[float, torch.Tensor, Callable[[bool], Any]]:
    """Validate checkpoint metadata used to configure the Toolkit adapter."""

    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("SevenNet modality must be a non-empty explicit string")
    if modality != modality.strip():
        raise ValueError("SevenNet modality may not contain surrounding whitespace")

    cutoff_value = getattr(model, "cutoff", None)
    if cutoff_value is None:
        raise ValueError("raw SevenNet model does not expose its checkpoint cutoff")
    cutoff = float(cutoff_value)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("raw SevenNet model cutoff must be finite and positive")

    type_map = getattr(model, "type_map", None)
    if not isinstance(type_map, Mapping) or not type_map:
        raise ValueError("raw SevenNet model does not expose a non-empty type_map")
    atomic_numbers = sorted({int(number) for number in type_map})
    if atomic_numbers[0] < 0:
        raise ValueError("SevenNet type_map contains a negative atomic number")

    modal_map = getattr(model, "modal_map", None)
    if not isinstance(modal_map, Mapping) or not modal_map:
        raise ValueError("SevenNet-Omni model does not expose its modal_map")
    if modality not in modal_map:
        available = ", ".join(sorted(str(name) for name in modal_map))
        raise ValueError(
            f"SevenNet modality {modality!r} is unavailable; choose one of: "
            f"{available}"
        )
    if getattr(model, "eval_type_map", None) is not True:
        raise ValueError(
            "raw SevenNet model must have eval_type_map=True for atomic-number input"
        )
    if getattr(model, "eval_modal_map", None) is not True:
        raise ValueError("raw SevenNet-Omni model must have eval_modal_map=True")
    if getattr(model, "key_grad", None) != _EDGE_VECTOR:
        raise ValueError(
            "raw SevenNet model must differentiate edge_vec to produce forces"
        )

    set_batch_mode = getattr(model, "set_is_batch_data", None)
    if not callable(set_batch_mode):
        raise TypeError("raw SevenNet model does not support batched graph input")

    model_device, _ = _model_device_and_dtype(model)
    supported = torch.zeros(
        max(atomic_numbers) + 1,
        dtype=torch.bool,
        device=model_device,
    )
    supported[atomic_numbers] = True
    return cutoff, supported, set_batch_mode


def _map_sevennet_outputs(raw: Any, data: Batch) -> dict[str, torch.Tensor]:
    """Validate the raw result and return Toolkit energy/force fields."""

    if not isinstance(raw, Mapping):
        try:
            from sevenn.atom_graph_data import AtomGraphData
        except ImportError as exc:
            raise ImportError(
                "SevenNetOmniWrapper requires sevenn==0.13.0 at runtime"
            ) from exc
        if not isinstance(raw, AtomGraphData):
            raise TypeError("raw SevenNet output must be a mapping or AtomGraphData")
    try:
        energy = raw[_PREDICTED_ENERGY]
        forces = raw[_PREDICTED_FORCE]
    except KeyError as exc:
        raise KeyError(
            "raw SevenNet output must contain inferred_total_energy and inferred_force"
        ) from exc
    if not isinstance(energy, torch.Tensor) or not isinstance(forces, torch.Tensor):
        raise TypeError("raw SevenNet energy and forces must be tensors")
    if energy.numel() != data.num_graphs:
        raise ValueError("SevenNet returned the wrong number of graph energies")
    if forces.shape != data.positions.shape:
        raise ValueError("SevenNet returned forces with the wrong shape")
    return {
        "energy": energy.detach().reshape(data.num_graphs, 1),
        "forces": forces.detach(),
    }


class _SevenNetAdapterBase(nn.Module):
    """Handle checkpoint-specific setup not central to the adapter lesson.

    The learner-facing subclass declares the Toolkit configuration and shows
    the input, output, and forward methods.  This base keeps checkpoint metadata
    validation, the supported-element buffer, the cutoff property, and the
    unused embedding interface in one maintained place.
    """

    def __init__(self, model: nn.Module, *, modality: str) -> None:
        super().__init__()
        cutoff, supported, set_batch_mode = _validated_sevennet_metadata(
            model, modality=modality
        )
        self.model = model
        self.modality = modality
        self._cutoff = cutoff
        self.register_buffer(
            "_supported_atomic_numbers", supported, persistent=False
        )
        set_batch_mode(True)
        self.model.eval()

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:
        del data, kwargs
        raise NotImplementedError("This adapter exposes only energy and forces")


# BEGIN NOTEBOOK MODEL CONFIG
def make_sevennet_model_config(cutoff: float) -> ModelConfig:
    """Declare the energy, force, periodic, and neighbor interface to Toolkit."""

    return ModelConfig(
        outputs=frozenset({"energy", "forces"}),
        active_outputs={"energy", "forces"},
        autograd_outputs=frozenset(),
        autograd_inputs=frozenset(),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"cell", "pbc", "neighbor_list_shifts"}),
        supports_pbc=True,
        needs_pbc=False,
        neighbor_config=NeighborConfig(
            cutoff=cutoff,
            format=NeighborListFormat.COO,
            half_list=False,
            skin=0.0,
        ),
    )
# END NOTEBOOK MODEL CONFIG


# BEGIN NOTEBOOK WRAPPER
class SevenNetOmniWrapper(_SevenNetAdapterBase, BaseModelMixin):
    """Connect a raw SevenNet-Omni energy/force model to Toolkit."""

    def __init__(self, model: nn.Module, *, modality: str) -> None:
        super().__init__(model, modality=modality)
        self.model_config = make_sevennet_model_config(self.cutoff)

    def direct_derivative_keys(self) -> set[str]:
        # SevenNet already differentiates edge vectors to produce forces.
        return {"forces"}

    def adapt_input(self, data: Batch, **kwargs: Any) -> Any:
        del kwargs
        device, dtype = _model_device_and_dtype(self.model)
        if self._supported_atomic_numbers.device != device:
            raise RuntimeError("move the complete wrapper with wrapper.to(device)")
        return _toolkit_batch_to_sevennet_graph(
            data,
            device=device,
            dtype=dtype,
            modality=self.modality,
            supported_atomic_numbers=self._supported_atomic_numbers,
        )

    def adapt_output(self, raw: Any, data: Batch):
        mapped = _map_sevennet_outputs(raw, data)
        return super().adapt_output(mapped, data)

    def forward(self, data: Batch, **kwargs: Any):
        graph = self.adapt_input(data, **kwargs)
        raw = self.model(graph)  # one call for the complete ragged batch
        return self.adapt_output(raw, data)
# END NOTEBOOK WRAPPER


__all__ = ["SevenNetOmniWrapper"]
