"""Validation and result assembly for the Part 1 SevenNet adapter lesson.

The notebook keeps Toolkit neighbor construction and every model call visible.
This module handles the longer comparison, splitting, and table-building work
after those calls have returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import os
from pathlib import Path
from typing import Any

from ase import Atoms
import numpy as np
import pandas as pd
import torch
from nvalchemi.data import Batch

from .sevennet_checks import (
    build_sevennet_mapping_table,
    build_sevennet_repeat_table,
    split_model_outputs,
)


@dataclass(frozen=True)
class SevenNetLessonResults:
    """Checked tables and graph-aligned energy/force dictionaries."""

    graph_mapping: pd.DataFrame
    graph_mapping_passed: bool
    official_calculator_check: pd.DataFrame
    numerical_agreement: pd.DataFrame
    repeat_max_energy_difference_eV_per_atom: float
    repeat_max_force_difference_eV_A: float
    model_energies: dict[str, float]
    d3_energies: dict[str, float]
    combined_energies: dict[str, float]
    model_forces: dict[str, np.ndarray]
    d3_forces: dict[str, np.ndarray]
    combined_forces: dict[str, np.ndarray]


def _raw_repeat(
    wrapper: Any,
    raw_model: Any,
    batch: Batch,
    outputs: Mapping[str, torch.Tensor],
    *,
    labels: Sequence[str],
) -> pd.DataFrame:
    raw_graph = wrapper.adapt_input(batch)
    raw_outputs = wrapper.adapt_output(raw_model(raw_graph), batch)
    table = build_sevennet_repeat_table(
        outputs,
        raw_outputs,
        labels=labels,
        atom_counts=batch.num_nodes_list,
    )
    table.insert(0, "comparison", "adapter output vs direct raw call")
    return table


def _component_sum_repeat(
    component_a: Mapping[str, torch.Tensor],
    component_b: Mapping[str, torch.Tensor],
    pipeline: Mapping[str, torch.Tensor],
    *,
    labels: Sequence[str],
    atom_counts: Sequence[int],
) -> pd.DataFrame:
    expected = {
        "energy": component_a["energy"] + component_b["energy"],
        "forces": component_a["forces"] + component_b["forces"],
    }
    table = build_sevennet_repeat_table(
        expected,
        pipeline,
        labels=labels,
        atom_counts=atom_counts,
    )
    table.insert(0, "comparison", "pipeline output vs explicit component sum")
    return table


def _official_calculator_comparison(
    *,
    atoms: Atoms,
    structure_key: str,
    checkpoint_path: Path,
    modality: str,
    device: torch.device,
    adapter_energy_eV: float,
    adapter_forces_eV_A: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enabled = [
        name
        for name in (
            "SEVENNET_ENABLE_CUEQ",
            "SEVENNET_ENABLE_FLASH",
            "SEVENNET_ENABLE_OEQ",
        )
        if os.environ.get(name) == "1"
    ]
    if enabled:
        raise RuntimeError(
            "Official comparison must use the same e3nn backend; unset: "
            + ", ".join(enabled)
        )

    from sevenn.calculator import SevenNetCalculator

    calculator = SevenNetCalculator(
        model=checkpoint_path,
        file_type="checkpoint",
        device=str(device),
        modal=modality,
        enable_cueq=False,
        enable_flash=False,
        enable_oeq=False,
        compute_atomic_virial=False,
    )
    checked_atoms = atoms.copy()
    try:
        checked_atoms.calc = calculator
        official_energy = float(checked_atoms.get_potential_energy())
        official_forces = np.asarray(
            checked_atoms.get_forces(apply_constraint=False), dtype=np.float64
        ).copy()
        official_edges = int(calculator.results["num_edges"])
    finally:
        checked_atoms.calc = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del calculator
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if adapter_forces_eV_A.shape != official_forces.shape:
        raise RuntimeError("Official and Toolkit force arrays have different shapes")
    energy_difference = abs(adapter_energy_eV - official_energy)
    force_difference = float(
        np.max(np.abs(adapter_forces_eV_A - official_forces))
    )
    row = {
        "structure": structure_key,
        "atoms": len(checked_atoms),
        "official_directed_edges": official_edges,
        "adapter_energy_eV": adapter_energy_eV,
        "official_energy_eV": official_energy,
        "energy_difference_eV": energy_difference,
        "energy_difference_eV_per_atom": energy_difference / len(checked_atoms),
        "max_force_component_difference_eV_A": force_difference,
    }
    display_table = pd.DataFrame([row])
    agreement = display_table[[
        "structure",
        "atoms",
        "energy_difference_eV",
        "energy_difference_eV_per_atom",
        "max_force_component_difference_eV_A",
    ]].copy()
    agreement.insert(0, "comparison", "custom adapter vs official SevenNetCalculator")
    return display_table, agreement


def finalize_sevennet_lesson(
    *,
    wrapper: Any,
    raw_model: Any,
    checkpoint_path: str | Path,
    modality: str,
    device: torch.device,
    periodic_structures: Mapping[str, Atoms],
    finite_structures: Mapping[str, Atoms],
    periodic_batch: Batch,
    finite_batch: Batch,
    periodic_model_outputs: Mapping[str, torch.Tensor],
    finite_model_outputs: Mapping[str, torch.Tensor],
    periodic_d3_outputs: Mapping[str, torch.Tensor],
    finite_d3_outputs: Mapping[str, torch.Tensor],
    periodic_pipeline_outputs: Mapping[str, torch.Tensor],
    finite_pipeline_outputs: Mapping[str, torch.Tensor],
    official_structure_key: str,
    energy_tolerance_eV_per_atom: float,
    force_tolerance_eV_A: float,
) -> SevenNetLessonResults:
    """Check adapter parity and split six visible batched outputs by graph."""

    periodic_labels = list(periodic_structures)
    finite_labels = list(finite_structures)
    mapping = build_sevennet_mapping_table(wrapper, periodic_batch)
    mapping_passed = bool(mapping["exact_match"].all())
    if not mapping_passed:
        raise RuntimeError("SevenNet graph mapping check failed")

    periodic_repeat = _raw_repeat(
        wrapper,
        raw_model,
        periodic_batch,
        periodic_model_outputs,
        labels=periodic_labels,
    )
    finite_repeat = _raw_repeat(
        wrapper,
        raw_model,
        finite_batch,
        finite_model_outputs,
        labels=finite_labels,
    )

    official_index = periodic_labels.index(official_structure_key)
    force_blocks = torch.split(
        periodic_model_outputs["forces"],
        tuple(int(count) for count in periodic_batch.num_nodes_list),
    )
    official_table, official_agreement = _official_calculator_comparison(
        atoms=periodic_structures[official_structure_key],
        structure_key=official_structure_key,
        checkpoint_path=Path(checkpoint_path),
        modality=modality,
        device=device,
        adapter_energy_eV=float(
            periodic_model_outputs["energy"]
            .reshape(-1)[official_index]
            .detach()
            .cpu()
        ),
        adapter_forces_eV_A=(
            force_blocks[official_index].detach().cpu().numpy().copy()
        ),
    )

    periodic_pipeline_repeat = _component_sum_repeat(
        periodic_model_outputs,
        periodic_d3_outputs,
        periodic_pipeline_outputs,
        labels=periodic_labels,
        atom_counts=periodic_batch.num_nodes_list,
    )
    finite_pipeline_repeat = _component_sum_repeat(
        finite_model_outputs,
        finite_d3_outputs,
        finite_pipeline_outputs,
        labels=finite_labels,
        atom_counts=finite_batch.num_nodes_list,
    )
    agreement = pd.concat(
        (
            periodic_repeat,
            finite_repeat,
            official_agreement,
            periodic_pipeline_repeat,
            finite_pipeline_repeat,
        ),
        ignore_index=True,
    )
    max_energy = float(agreement["energy_difference_eV_per_atom"].max())
    max_force = float(agreement["max_force_component_difference_eV_A"].max())
    if max_energy >= float(energy_tolerance_eV_per_atom):
        raise RuntimeError("SevenNet energy difference exceeded its numerical check")
    if max_force >= float(force_tolerance_eV_A):
        raise RuntimeError("SevenNet force difference exceeded its numerical check")

    def split_pair(
        periodic: Mapping[str, torch.Tensor], finite: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        periodic_energy, periodic_forces = split_model_outputs(
            periodic_labels,
            periodic_batch.num_nodes_list,
            periodic,
        )
        finite_energy, finite_forces = split_model_outputs(
            finite_labels,
            finite_batch.num_nodes_list,
            finite,
        )
        return periodic_energy | finite_energy, periodic_forces | finite_forces

    model_energy, model_forces = split_pair(
        periodic_model_outputs, finite_model_outputs
    )
    d3_energy, d3_forces = split_pair(periodic_d3_outputs, finite_d3_outputs)
    combined_energy, combined_forces = split_pair(
        periodic_pipeline_outputs, finite_pipeline_outputs
    )
    if not all(np.isfinite(value) for value in combined_energy.values()):
        raise RuntimeError("SevenNet pipeline returned a non-finite energy")
    if not all(np.isfinite(value).all() for value in combined_forces.values()):
        raise RuntimeError("SevenNet pipeline returned a non-finite force")

    return SevenNetLessonResults(
        graph_mapping=mapping,
        graph_mapping_passed=mapping_passed,
        official_calculator_check=official_table,
        numerical_agreement=agreement,
        repeat_max_energy_difference_eV_per_atom=max_energy,
        repeat_max_force_difference_eV_A=max_force,
        model_energies=model_energy,
        d3_energies=d3_energy,
        combined_energies=combined_energy,
        model_forces=model_forces,
        d3_forces=d3_forces,
        combined_forces=combined_forces,
    )


__all__ = ["SevenNetLessonResults", "finalize_sevennet_lesson"]
