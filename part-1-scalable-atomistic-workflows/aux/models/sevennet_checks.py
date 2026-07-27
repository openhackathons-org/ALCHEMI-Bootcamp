"""Focused checks and table assembly for the SevenNet teaching adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase.data import chemical_symbols
from nvalchemi.data import Batch


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _comparison_row(
    component: str,
    toolkit: torch.Tensor,
    sevennet: torch.Tensor,
    *,
    units: str,
    note: str,
    exact: bool = False,
) -> dict[str, Any]:
    toolkit = toolkit.detach().to(device="cpu")
    sevennet = sevennet.detach().to(device="cpu")
    same_shape = toolkit.shape == sevennet.shape
    if same_shape and toolkit.numel():
        if toolkit.dtype == torch.bool or not toolkit.dtype.is_floating_point:
            difference = float((toolkit != sevennet.to(toolkit.dtype)).any())
        else:
            difference = float(
                torch.max(torch.abs(toolkit - sevennet.to(toolkit.dtype))).item()
            )
    elif same_shape:
        difference = 0.0
    else:
        difference = float("inf")
    matched = same_shape and (
        difference == 0.0 if exact else difference <= 1.0e-6
    )
    return {
        "component": component,
        "toolkit_shape": str(tuple(toolkit.shape)),
        "sevennet_shape": str(tuple(sevennet.shape)),
        "exact_match": bool(matched),
        "max_abs_difference": difference,
        "units": units,
        "note": note,
    }


def build_sevennet_mapping_table(wrapper: Any, batch: Batch) -> pd.DataFrame:
    """Check the visible Toolkit fields against the adapted SevenNet graph."""

    graph = wrapper.adapt_input(batch)
    required = {
        "atomic_numbers",
        "pos",
        "batch",
        "num_atoms",
        "edge_index",
        "edge_vec",
        "cell_volume",
        "data_modality",
    }
    missing = sorted(required - set(graph.keys()))
    if missing:
        raise KeyError("adapted SevenNet graph is missing: " + ", ".join(missing))

    neighbors = batch.neighbor_list
    valid = (neighbors[:, 0] < batch.num_nodes) & (
        neighbors[:, 1] < batch.num_nodes
    )
    neighbors = neighbors[valid].to(device=graph["edge_index"].device)
    shifts = batch.neighbor_list_shifts[valid].to(
        device=graph["edge_vec"].device,
        dtype=graph["edge_vec"].dtype,
    )
    source = neighbors[:, 0].long()
    target = neighbors[:, 1].long()
    source_graph = batch.batch_idx.to(source.device).index_select(0, source)
    cell = batch.cell.to(graph["edge_vec"]).reshape(batch.num_graphs, 3, 3)
    expected_vectors = (
        batch.positions.to(graph["edge_vec"]).index_select(0, target)
        - batch.positions.to(graph["edge_vec"]).index_select(0, source)
        + torch.einsum("ei,eij->ej", shifts, cell.index_select(0, source_graph))
    )
    expected_volume = torch.linalg.det(cell).abs()

    rows = [
        _comparison_row(
            "atomic numbers",
            batch.atomic_numbers.long(),
            _tensor(graph["atomic_numbers"], name="atomic_numbers"),
            units="Z",
            note="one value per atom",
            exact=True,
        ),
        _comparison_row(
            "positions",
            batch.positions,
            _tensor(graph["pos"], name="pos"),
            units="Å",
            note="copied without a unit conversion",
        ),
        _comparison_row(
            "graph ownership",
            batch.batch_idx.long(),
            _tensor(graph["batch"], name="batch"),
            units="index",
            note="one graph index per atom",
            exact=True,
        ),
        _comparison_row(
            "atoms per graph",
            torch.as_tensor(batch.num_nodes_list, dtype=torch.long),
            _tensor(graph["num_atoms"], name="num_atoms"),
            units="atoms",
            note="ragged reduction sizes",
            exact=True,
        ),
        _comparison_row(
            "directed COO edges",
            neighbors.T.long(),
            _tensor(graph["edge_index"], name="edge_index"),
            units="index",
            note="Toolkit (edge, source/target) becomes SevenNet (2, edge)",
            exact=True,
        ),
        _comparison_row(
            "periodic edge vectors",
            expected_vectors,
            _tensor(graph["edge_vec"], name="edge_vec"),
            units="Å",
            note="target - source + integer shift × cell",
        ),
        _comparison_row(
            "cell volumes",
            expected_volume,
            _tensor(graph["cell_volume"], name="cell_volume"),
            units="Å³",
            note="one value per graph",
        ),
    ]
    modality = graph["data_modality"]
    rows.append(
        {
            "component": "model task",
            "toolkit_shape": str((batch.num_graphs,)),
            "sevennet_shape": str((len(modality),)),
            "exact_match": bool(
                modality == [wrapper.modality] * batch.num_graphs
            ),
            "max_abs_difference": 0.0,
            "units": "name",
            "note": f"explicit {wrapper.modality!r} energy task",
        }
    )
    return pd.DataFrame(rows)


def build_sevennet_repeat_table(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    *,
    labels: Sequence[str],
    atom_counts: Sequence[int],
) -> pd.DataFrame:
    """Measure repeat-call energy and force differences for each graph."""

    if len(labels) != len(atom_counts):
        raise ValueError("labels and atom_counts must have the same length")
    first_energy = _tensor(first["energy"], name="first energy").reshape(-1)
    second_energy = _tensor(second["energy"], name="second energy").reshape(-1)
    if first_energy.numel() != len(labels) or second_energy.numel() != len(labels):
        raise ValueError("energy outputs do not match the graph labels")
    first_forces = torch.split(
        _tensor(first["forces"], name="first forces"), tuple(atom_counts)
    )
    second_forces = torch.split(
        _tensor(second["forces"], name="second forces"), tuple(atom_counts)
    )
    rows = []
    for index, label in enumerate(labels):
        energy_difference = abs(
            float((first_energy[index] - second_energy[index]).detach().cpu())
        )
        force_difference = float(
            torch.max(
                torch.abs(first_forces[index] - second_forces[index])
            ).detach().cpu()
        )
        rows.append(
            {
                "structure": label,
                "atoms": int(atom_counts[index]),
                "energy_difference_eV": energy_difference,
                "energy_difference_eV_per_atom": energy_difference
                / int(atom_counts[index]),
                "max_force_component_difference_eV_A": force_difference,
            }
        )
    return pd.DataFrame(rows)


def summarize_sevennet_task_outputs(
    *,
    task: str,
    target: str,
    structure_keys: Sequence[str],
    batch: Batch,
    outputs: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Summarize one task call without comparing task-specific energy zeros."""

    if not isinstance(task, str) or not task:
        raise ValueError("task must be non-empty text")
    if not isinstance(target, str) or not target:
        raise ValueError("target must be non-empty text")
    if len(structure_keys) != batch.num_graphs:
        raise ValueError("structure_keys must match the number of graphs")

    energies = _tensor(outputs["energy"], name="energy").detach().reshape(-1)
    forces = _tensor(outputs["forces"], name="forces").detach()
    if energies.numel() != batch.num_graphs:
        raise ValueError("energy output does not match the number of graphs")
    if forces.shape != (batch.num_nodes, 3):
        raise ValueError("force output does not match the batched atoms")

    graph_index = batch.batch_idx.to(device=forces.device)
    rows = []
    for index, structure in enumerate(structure_keys):
        graph_forces = forces[graph_index == index]
        if graph_forces.shape[0] == 0:
            raise ValueError(f"graph {index} contains no force vectors")
        if not bool(torch.isfinite(energies[index])):
            raise ValueError(f"{task} returned a non-finite energy for {structure}")
        if not bool(torch.isfinite(graph_forces).all()):
            raise ValueError(f"{task} returned non-finite forces for {structure}")
        rows.append(
            {
                "task": task,
                "target": target,
                "structure": structure,
                "energy output": "finite scalar",
                "force output": str(tuple(graph_forces.shape)),
                "max |F| / eV Å⁻¹": float(
                    torch.linalg.vector_norm(graph_forces, dim=1).max().cpu()
                ),
            }
        )
    return rows


def split_model_outputs(
    keys: Sequence[str],
    atom_counts: Sequence[int],
    outputs: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Split a batched energy/force result into ordered per-structure maps."""

    if len(keys) != len(atom_counts):
        raise ValueError("keys and atom_counts must have the same length")
    energy = _tensor(outputs["energy"], name="energy").detach().cpu().reshape(-1)
    forces = _tensor(outputs["forces"], name="forces").detach().cpu()
    if energy.numel() != len(keys) or sum(atom_counts) != forces.shape[0]:
        raise ValueError("model outputs do not match the supplied graph layout")
    force_parts = torch.split(forces, tuple(atom_counts))
    return (
        {key: float(energy[index]) for index, key in enumerate(keys)},
        {
            key: force_parts[index].numpy().copy()
            for index, key in enumerate(keys)
        },
    )


def _supported_element_value(model: Any) -> str:
    """Format the exact element domain exposed by the loaded model."""

    type_map = getattr(model, "type_map", None)
    if not isinstance(type_map, Mapping) or not type_map:
        raise ValueError(
            "raw SevenNet model must expose a non-empty type_map to report "
            "supported elements"
        )

    atomic_numbers = []
    for value in type_map:
        if isinstance(value, bool):
            raise ValueError("SevenNet type_map keys must be atomic numbers")
        try:
            atomic_number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SevenNet type_map keys must be atomic numbers"
            ) from exc
        if atomic_number != value:
            raise ValueError("SevenNet type_map keys must be integer atomic numbers")
        if not 1 <= atomic_number < len(chemical_symbols):
            raise ValueError(
                "SevenNet type_map contains an unsupported atomic number: "
                f"{atomic_number}"
            )
        atomic_numbers.append(atomic_number)

    atomic_numbers = sorted(atomic_numbers)
    if len(set(atomic_numbers)) != len(atomic_numbers):
        raise ValueError("SevenNet type_map contains duplicate atomic numbers")
    elements = ", ".join(
        f"{chemical_symbols[number]} (Z={number})" for number in atomic_numbers
    )
    return f"{len(atomic_numbers)} elements: {elements}"


def _model_dtype_value(model: Any) -> str:
    """Return one unambiguous floating-point dtype for the live model."""

    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise TypeError(
            "raw SevenNet model must expose parameters() and buffers() to "
            "report its dtype"
        )

    parameter_values = tuple(parameters())
    if not parameter_values:
        raise ValueError(
            "raw SevenNet model has no parameters; its model dtype is unavailable"
        )
    if not all(isinstance(value, torch.Tensor) for value in parameter_values):
        raise TypeError("raw SevenNet model parameters must be torch tensors")
    if not parameter_values[0].dtype.is_floating_point:
        raise TypeError("raw SevenNet model parameters must use a floating-point dtype")

    buffer_values = tuple(buffers())
    if not all(isinstance(value, torch.Tensor) for value in buffer_values):
        raise TypeError("raw SevenNet model buffers must be torch tensors")
    floating_dtypes = {
        value.dtype
        for value in (*parameter_values, *buffer_values)
        if value.dtype.is_floating_point
    }
    if len(floating_dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in floating_dtypes))
        raise ValueError(
            "raw SevenNet model must use one floating-point dtype; "
            f"found {names or 'none'}"
        )
    return str(next(iter(floating_dtypes))).removeprefix("torch.")


def _config_names(config: Any, attribute: str) -> frozenset[str]:
    """Read one declared ModelConfig name set without inventing defaults."""

    values = getattr(config, attribute, None)
    if values is None:
        raise ValueError(
            f"SevenNet wrapper ModelConfig must declare {attribute}"
        )
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"SevenNet wrapper ModelConfig {attribute} must be a collection of names"
        )
    try:
        names = frozenset(values)
    except TypeError as exc:
        raise TypeError(
            f"SevenNet wrapper ModelConfig {attribute} must be a collection of names"
        ) from exc
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError(
            f"SevenNet wrapper ModelConfig {attribute} must contain non-empty strings"
        )
    return names


def _field_behavior(
    *,
    label: str,
    input_names: frozenset[str],
    output_names: frozenset[str],
    required_inputs: frozenset[str],
    optional_inputs: frozenset[str],
    outputs: frozenset[str],
) -> str:
    """Describe one field using only the wrapper's declared interface."""

    required_matches = sorted(input_names & required_inputs)
    optional_matches = sorted(input_names & optional_inputs)
    if required_matches and optional_matches:
        raise ValueError(
            f"SevenNet wrapper declares ambiguous {label} input behavior"
        )
    if required_matches:
        input_behavior = (
            "required input (" + ", ".join(required_matches) + ")"
        )
    elif optional_matches:
        input_behavior = (
            "optional input (" + ", ".join(optional_matches) + ")"
        )
    else:
        input_behavior = "not consumed as input"

    returned_matches = sorted(output_names & outputs)
    output_behavior = (
        "returned (" + ", ".join(returned_matches) + ")"
        if returned_matches
        else "not returned"
    )
    return f"{input_behavior}; {output_behavior}"


def build_sevennet_model_card(*, model: Any, wrapper: Any) -> pd.DataFrame:
    """Return checked learner-facing facts for the loaded SevenNet adapter.

    Supported elements and precision come from the live raw model. Charge,
    spin, and multiplicity behavior comes from the wrapper's declared
    ``ModelConfig`` rather than assumptions about those inputs.
    """

    wrapped_model = getattr(wrapper, "model", model)
    if wrapped_model is not model:
        raise ValueError("wrapper.model must be the supplied raw SevenNet model")
    config = getattr(wrapper, "model_config", None)
    if config is None:
        raise ValueError(
            "SevenNet wrapper must expose model_config to report input behavior"
        )

    required_inputs = _config_names(config, "required_inputs")
    optional_inputs = _config_names(config, "optional_inputs")
    outputs = _config_names(config, "outputs")
    overlap = sorted(required_inputs & optional_inputs)
    if overlap:
        raise ValueError(
            "SevenNet wrapper ModelConfig declares inputs as both required and "
            "optional: " + ", ".join(overlap)
        )

    rows = [
        {
            "Field": "Supported elements",
            "Value": _supported_element_value(model),
        },
        {
            "Field": "Model dtype",
            "Value": _model_dtype_value(model),
        },
        {
            "Field": "Charge behavior",
            "Value": _field_behavior(
                label="charge",
                input_names=frozenset({"charge"}),
                output_names=frozenset({"charge", "charges"}),
                required_inputs=required_inputs,
                optional_inputs=optional_inputs,
                outputs=outputs,
            ),
        },
        {
            "Field": "Spin behavior",
            "Value": _field_behavior(
                label="spin",
                input_names=frozenset({"spin"}),
                output_names=frozenset({"spin", "spins"}),
                required_inputs=required_inputs,
                optional_inputs=optional_inputs,
                outputs=outputs,
            ),
        },
        {
            "Field": "Multiplicity behavior",
            "Value": _field_behavior(
                label="multiplicity",
                input_names=frozenset({"multiplicity", "spin_multiplicity"}),
                output_names=frozenset({"multiplicity", "spin_multiplicity"}),
                required_inputs=required_inputs,
                optional_inputs=optional_inputs,
                outputs=outputs,
            ),
        },
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def build_sevennet_settings_table(
    *,
    model_name: str,
    modality: str,
    reference_method: str,
    package_version: str,
    checkpoint_sha256: str,
    checkpoint_record: str,
    model_cutoff_A: float,
    supports_pbc: bool,
    outputs: Sequence[str],
    d3_reference_cutoff_bohr: float,
    d3_cutoff_A: float,
    d3_smoothing_fraction: float,
) -> pd.DataFrame:
    """Format the model, neighbor, D3, and license settings shown in Part 1."""

    return pd.Series(
        {
            "Raw model": model_name,
            "Task": modality,
            "Training target": reference_method,
            "sevenn": package_version,
            "Checkpoint SHA-256": checkpoint_sha256,
            "Checkpoint record": checkpoint_record,
            "Model cutoff / Å": float(model_cutoff_A),
            "Neighbor list": "COO, full directed",
            "Periodic systems": bool(supports_pbc),
            "Outputs": sorted(outputs),
            "D3 parameters": "pairwise PBE-D3(BJ)",
            "D3 cutoff": (
                f"{float(d3_reference_cutoff_bohr):.0f} bohr / "
                f"{float(d3_cutoff_A):.3f} Å"
            ),
            "D3 taper": float(d3_smoothing_fraction),
            "SevenNet code license": "MIT",
            "Checkpoint record license": "CC BY 4.0",
            "Exact checkpoint file license": (
                "not separately stated; runtime download only"
            ),
        },
        name="Value",
    ).rename_axis("Setting").reset_index()


__all__ = [
    "build_sevennet_model_card",
    "build_sevennet_settings_table",
    "build_sevennet_mapping_table",
    "build_sevennet_repeat_table",
    "summarize_sevennet_task_outputs",
    "split_model_outputs",
]
