# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
from ase import Atoms

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))

from helpers import module02 as module02_helpers
from helpers.model_support import suppress_known_mace_load_noise
from helpers.relaxation import maximum_force_history, prepare_dynamics_batch
from nvalchemi.data import AtomicData, Batch


def _cells(name: str) -> list[dict[str, object]]:
    notebook = json.loads((CORE_DIR / name).read_text(encoding="utf-8"))
    return notebook["cells"]


def _tagged_power_user_cells(name: str) -> list[dict[str, object]]:
    return [
        cell
        for cell in _cells(name)
        if "power-user" in cell.get("metadata", {}).get("tags", [])
    ]


def _cells_in_power_user_sections(name: str) -> list[dict[str, object]]:
    """Return complete tagged optional sections through the next H2."""
    optional_cells: list[dict[str, object]] = []
    inside_power_user = False

    for cell in _cells(name):
        tags = cell.get("metadata", {}).get("tags", [])
        source = "".join(cell["source"]).lstrip()
        has_h2 = any(line.startswith("## ") for line in source.splitlines())

        if (
            not inside_power_user
            and cell["cell_type"] == "markdown"
            and "power-user" in tags
        ):
            inside_power_user = True
        elif inside_power_user and cell["cell_type"] == "markdown" and has_h2:
            inside_power_user = False

        if inside_power_user:
            optional_cells.append(cell)

    return optional_cells


def _two_hydrogen_batch(distance: float) -> Batch:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    first = AtomicData.from_atoms(atoms)
    second = AtomicData.from_atoms(atoms)
    for record in (first, second):
        record.add_system_property("energy", torch.zeros((1, 1)))
        record.add_node_property("forces", torch.zeros((2, 3)))
    return Batch.from_data_list([first, second])


def test_mace_noise_filter_is_narrow() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_known_mace_load_noise():
            warnings.warn("invalid escape sequence '\\.'", SyntaxWarning)
            warnings.warn(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, details",
                UserWarning,
            )
            warnings.warn("scientific syntax warning", SyntaxWarning)
            warnings.warn("scientific warning", UserWarning)

    assert [str(item.message) for item in caught] == [
        "scientific syntax warning",
        "scientific warning",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "alchemi-core-01-data-and-batching.ipynb",
        "alchemi-core-02-models-and-simulation.ipynb",
    ],
)
def test_power_user_code_is_folded(name: str) -> None:
    tagged_cells = _tagged_power_user_cells(name)

    tagged_code = [cell for cell in tagged_cells if cell["cell_type"] == "code"]
    assert tagged_code
    assert all(
        cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is True
        for cell in tagged_code
    )


@pytest.mark.parametrize(
    "name",
    [
        "alchemi-core-01-data-and-batching.ipynb",
        "alchemi-core-02-models-and-simulation.ipynb",
    ],
)
def test_power_user_sections_start_collapsed(name: str) -> None:
    """Keep each complete optional section folded when JupyterLab opens it."""
    headings = [
        cell
        for cell in _tagged_power_user_cells(name)
        if cell["cell_type"] == "markdown"
        and "".join(cell["source"]).lstrip().startswith("### ⚡ Power User:")
    ]

    assert headings
    assert all(
        cell.get("metadata", {}).get("jp-MarkdownHeadingCollapsed") is True
        for cell in headings
    )


def test_module_one_power_user_sections_are_fully_folded() -> None:
    """Prevent an untagged code cell from escaping an optional section fold."""
    optional_cells = _cells_in_power_user_sections(
        "alchemi-core-01-data-and-batching.ipynb"
    )
    optional_code = [cell for cell in optional_cells if cell["cell_type"] == "code"]

    assert optional_code
    assert all(
        "power-user" in cell.get("metadata", {}).get("tags", [])
        and cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is True
        for cell in optional_code
    )


def test_module_two_power_user_sections_include_all_cells() -> None:
    """Keep complete optional blocks tagged and their code sources hidden."""
    optional_cells = _cells_in_power_user_sections(
        "alchemi-core-02-models-and-simulation.ipynb"
    )
    optional_code = [cell for cell in optional_cells if cell["cell_type"] == "code"]

    assert optional_cells
    assert optional_code
    assert all(
        "power-user" in cell.get("metadata", {}).get("tags", [])
        for cell in optional_cells
    )
    assert all(
        cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is True
        for cell in optional_code
    )


def test_prepare_dynamics_batch_creates_missing_system_storage() -> None:
    record = AtomicData.from_atoms(Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    batch = Batch.from_data_list([record])

    prepared = prepare_dynamics_batch(batch)

    assert prepared.energy.shape == (1, 1)
    assert prepared.forces.shape == (1, 3)
    assert prepared.velocities.shape == (1, 3)


def test_maximum_force_history_reads_registered_node_properties() -> None:
    batch = _two_hydrogen_batch(1.0)
    batch.forces[0, 0] = 2.0
    batch.forces[2, 1] = 3.0

    history = maximum_force_history([batch], ("first", "second"))

    assert history == {"first": [2.0], "second": [3.0]}


def test_fragment_com_restraint_is_zero_at_reference_and_balanced() -> None:
    atoms, graph_rows, _ = module02_helpers.load_nci_atlas()
    _, structures, fragment_a_sizes = module02_helpers.equilibrium_nci_complexes(
        atoms, graph_rows
    )
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(structure) for structure in structures]
    )
    batch = prepare_dynamics_batch(batch)
    reference = module02_helpers.fragment_com_separations(batch, fragment_a_sizes)

    energy, forces = module02_helpers.fragment_com_harmonic_bias(
        batch, fragment_a_sizes, reference, 10.0
    )

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    torch.testing.assert_close(forces, torch.zeros_like(forces), atol=1.0e-6, rtol=0.0)

    graph = 0
    start = int(batch.batch_ptr[graph])
    stop = int(batch.batch_ptr[graph + 1])
    split = start + fragment_a_sizes[graph]
    data = batch.get_data(graph)
    local_split = fragment_a_sizes[graph]
    masses = data.atomic_masses.reshape(-1, 1)
    com_a = (masses[:local_split] * data.positions[:local_split]).sum(0) / masses[
        :local_split
    ].sum()
    com_b = (masses[local_split:] * data.positions[local_split:]).sum(0) / masses[
        local_split:
    ].sum()
    direction = (com_b - com_a) / torch.linalg.vector_norm(com_b - com_a)
    batch.positions[split:stop] += 0.1 * direction

    energy, forces = module02_helpers.fragment_com_harmonic_bias(
        batch, fragment_a_sizes, reference, 10.0
    )

    assert energy[graph].item() == pytest.approx(0.05, abs=1.0e-5)
    torch.testing.assert_close(
        forces[batch.batch_idx != graph],
        torch.zeros_like(forces[batch.batch_idx != graph]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        forces[start:stop].sum(0),
        torch.zeros(3),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_nci_scan_player_data_groups_aligned_complex_frames() -> None:
    atoms, graph_rows, _ = module02_helpers.load_nci_atlas()

    trajectories, fragment_a_sizes, scales = module02_helpers.nci_scan_player_data(
        atoms, graph_rows
    )

    assert list(trajectories) == [
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    ]
    assert scales == pytest.approx(
        [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0]
    )
    assert all(len(frames) == len(scales) for frames in trajectories.values())
    for label, frames in trajectories.items():
        assert fragment_a_sizes[label] > 0
        assert fragment_a_sizes[label] < len(frames[0])
