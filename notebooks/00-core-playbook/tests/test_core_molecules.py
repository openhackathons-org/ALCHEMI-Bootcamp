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

from helpers import core as helpers
from nvalchemi.data import AtomicData, Batch


def _cells(name: str) -> list[dict[str, object]]:
    notebook = json.loads((CORE_DIR / name).read_text(encoding="utf-8"))
    return notebook["cells"]


def _source(name: str) -> str:
    return "\n".join("".join(cell["source"]) for cell in _cells(name))


def _cell_source(name: str, fragment: str) -> str:
    matches = [
        "".join(cell["source"])
        for cell in _cells(name)
        if fragment in "".join(cell["source"])
    ]
    assert len(matches) == 1
    return matches[0]


def _power_user_cells(name: str) -> list[dict[str, object]]:
    return [
        cell
        for cell in _cells(name)
        if "⚡ Power User:" in "".join(cell["source"])
    ]


def _tagged_power_user_cells(name: str) -> list[dict[str, object]]:
    return [
        cell
        for cell in _cells(name)
        if "power-user" in cell.get("metadata", {}).get("tags", [])
    ]


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
        with helpers.suppress_known_mace_load_noise():
            warnings.warn(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, details",
                UserWarning,
            )
            warnings.warn("scientific warning", UserWarning)

    assert [str(item.message) for item in caught] == ["scientific warning"]


def test_module_one_reserves_advanced_data_paths_for_power_users() -> None:
    source = _source("alchemi-core-01-data-and-batching.ipynb")
    direct_construction = _cell_source(
        "alchemi-core-01-data-and-batching.ipynb",
        "Direct construction still runs `AtomicData` validation",
    )

    cells = _cells("alchemi-core-01-data-and-batching.ipynb")
    zarr_power_index = next(
        index
        for index, cell in enumerate(cells)
        if "⚡ Power User: save, load, and prefetch with Zarr"
        in "".join(cell["source"])
    )
    required_zarr_code = "\n".join(
        "".join(cell["source"])
        for cell in cells[:zarr_power_index]
        if cell["cell_type"] == "code"
    )

    assert source.count("⚡ Power User:") == 3
    assert "⚡ Power User: direct tensor construction and validation" in source
    assert "⚡ Power User: Batch fields, indexing, and slicing" in source
    assert "⚡ Power User: save, load, and prefetch with Zarr" in source
    assert "AtomicDataZarrWriter" not in required_zarr_code
    assert "AtomicDataZarrReader" not in required_zarr_code
    assert "DataLoader(" not in required_zarr_code
    assert "Dataset(" not in required_zarr_code
    assert "AtomicDataZarrWriter" in source
    assert "AtomicDataZarrReader" in source
    assert "DataLoader" in source
    assert "Dataset" in source
    assert "prefetch_factor" in source
    assert "module-01-data-and-loading-power-user.ipynb" not in source
    assert "Direct construction still runs `AtomicData` validation" in (
        direct_construction
    )
    assert "InMemoryDataset(..., skip_validation=True)" in direct_construction


def test_module_one_power_user_topics_are_three_compact_blocks() -> None:
    cells = _cells("alchemi-core-01-data-and-batching.ipynb")
    header_indices = [
        index
        for index, cell in enumerate(cells)
        if "⚡ Power User:" in "".join(cell["source"])
    ]

    assert len(header_indices) == 3
    for index in header_indices:
        header_source = "".join(cells[index]["source"])
        explanation_cell = cells[index + 1]
        code_cell = cells[index + 2]
        assert header_source.startswith("### ⚡ Power User:")
        assert "<details>" not in header_source
        assert explanation_cell["cell_type"] == "markdown"
        explanation_source = "".join(explanation_cell["source"])
        assert "optional" in explanation_source.lower()
        assert "**Return to the required path:**" in explanation_source
        assert code_cell["cell_type"] == "code"
        assert "power-user" in code_cell.get("metadata", {}).get("tags", [])
        assert code_cell.get("metadata", {}).get("jupyter", {}).get(
            "source_hidden"
        ) is True
        assert code_cell.get("outputs", []) == []


@pytest.mark.parametrize(
    "name",
    [
        "alchemi-core-02-models-and-simulation.ipynb",
    ],
)
def test_power_user_units_are_explained_and_folded(name: str) -> None:
    power_cells = _power_user_cells(name)
    tagged_cells = _tagged_power_user_cells(name)

    assert power_cells
    for cell in power_cells:
        cell_source = "".join(cell["source"])
        assert "<details>" in cell_source
        assert "<summary><strong>⚡ Power User:" in cell_source
        assert "optional" in cell_source.lower()

    tagged_code = [
        cell
        for cell in tagged_cells
        if cell["cell_type"] == "code"
    ]
    assert tagged_code
    assert all(
        cell.get("metadata", {}).get("jupyter", {}).get("source_hidden") is True
        for cell in tagged_code
    )


def test_packed_layout_matches_batch_ownership() -> None:
    structures = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        Atoms("H3", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    ]
    batch = Batch.from_data_list([AtomicData.from_atoms(item) for item in structures])

    layout = helpers.packed_layout(batch)

    assert layout["counts"] == [1, 2, 3]
    assert layout["batch_ptr"] == [0, 1, 3, 6]
    assert layout["batch_idx"] == [0, 1, 1, 2, 2, 2]
    assert layout["packed_rows"] == 6
    assert layout["padded_rows"] == 9
    assert layout["padding_rows"] == 3


def test_graph_local_pairs_apply_the_exact_row_offsets() -> None:
    batch = _two_hydrogen_batch(1.0)

    assert helpers.graph_local_pairs(batch, (0, 1)) == ((0, 1), (2, 3))


def test_prepare_dynamics_batch_creates_missing_system_storage() -> None:
    record = AtomicData.from_atoms(Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    batch = Batch.from_data_list([record])

    prepared = helpers.prepare_dynamics_batch(batch)

    assert prepared.energy.shape == (1, 1)
    assert prepared.forces.shape == (1, 3)
    assert prepared.velocities.shape == (1, 3)


def test_maximum_force_history_reads_registered_node_properties() -> None:
    batch = _two_hydrogen_batch(1.0)
    batch.forces[0, 0] = 2.0
    batch.forces[2, 1] = 3.0

    history = helpers.maximum_force_history([batch], ("first", "second"))

    assert history == {"first": [2.0], "second": [3.0]}


def test_hydrogen_bond_table_finds_phenol_ch_and_oh_bonds() -> None:
    phenol = helpers.load_example_molecule("Phenol")

    bonds = helpers.hydrogen_bond_table(phenol)

    assert len(bonds) == 6
    assert bonds["bond"].value_counts().to_dict() == {"C-H": 5, "O-H": 1}
    assert bonds["hydrogen_index"].is_unique
    assert bonds["initial_distance_a"].between(0.9, 1.2).all()


def test_hydrogen_retention_history_tracks_each_graph_independently() -> None:
    phenol = helpers.load_example_molecule("Phenol")
    bonds = helpers.hydrogen_bond_table(phenol)
    first = AtomicData.from_atoms(phenol)
    second = AtomicData.from_atoms(phenol)
    frame = Batch.from_data_list([first, second])
    broken_hydrogen = int(bonds.iloc[0]["hydrogen_index"])
    graph_offset = int(frame.batch_ptr[1])
    frame.positions[graph_offset + broken_hydrogen, 0] += 4.0

    history = helpers.hydrogen_retention_history(
        [frame], bonds, ("baseline", "protected")
    )

    assert history.loc[history["branch"] == "baseline", "intact_bonds"].item() == 6
    assert history.loc[history["branch"] == "protected", "intact_bonds"].item() == 5


def test_module_two_restraint_is_zero_below_threshold_and_selected_by_graph() -> None:
    name = "alchemi-core-02-models-and-simulation.ipynb"
    phenol = helpers.load_example_molecule("Phenol")
    phenol_bonds = helpers.hydrogen_bond_table(phenol)
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(phenol), AtomicData.from_atoms(phenol)]
    )
    batch = helpers.prepare_dynamics_batch(batch)
    namespace = {
        "torch": torch,
        "device": torch.device("cpu"),
        "md_batch": batch,
        "phenol_bonds": phenol_bonds,
    }
    exec(_cell_source(name, "SPRING_EV_PER_A2 = 200.0"), namespace)  # noqa: S102
    exec(  # noqa: S102
        _cell_source(name, "def hookean_hydrogen_protection"),
        namespace,
    )

    energy, forces = namespace["hookean_hydrogen_protection"](batch)

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    torch.testing.assert_close(forces, torch.zeros_like(forces))

    heavy, hydrogen = namespace["protected_pairs"][0]
    vector = batch.positions[hydrogen] - batch.positions[heavy]
    batch.positions[hydrogen] += 0.1 * vector / torch.linalg.vector_norm(vector)
    energy, forces = namespace["hookean_hydrogen_protection"](batch)
    assert energy[0].item() == 0.0
    assert energy[1].item() == pytest.approx(1.0, abs=1.0e-4)
    torch.testing.assert_close(
        forces[batch.batch_idx == 0],
        torch.zeros_like(forces[batch.batch_idx == 0]),
    )
    assert torch.linalg.vector_norm(forces[hydrogen]).item() == pytest.approx(
        20.0, abs=1.0e-3
    )
    torch.testing.assert_close(forces[heavy], -forces[hydrogen])


def test_module_two_keeps_bias_and_composition_apis_visible() -> None:
    source = _source("alchemi-core-02-models-and-simulation.ipynb")

    assert "BiasedPotentialHook" in source
    assert "DynamicsStage.AFTER_COMPUTE" in source
    assert "NVTLangevin" in source
    assert "initialize_velocities" in source
    assert "show_trajectory_comparison" in source
    assert "PipelineModelWrapper" in source
    assert 'wire={"charges": "partial_charges"}' in source
    assert "plot_nci_interaction_curves" in source
    assert "Compose the model through single forward calls" in source
    assert "class VelocityRescalingThermostat(BaseDynamics)" in source
    assert source.count("⚡ Power User:") == 1
    assert "Zarr" not in source
    assert "show_bond_slider" not in source
    assert "phenol_lesson" not in source
    assert "module-02-models-hooks-and-persistence-power-user.ipynb" not in source


def test_focused_nci_curves_live_in_module_two() -> None:
    module_two = _source("alchemi-core-02-models-and-simulation.ipynb")

    assert "Predict before inspecting the curves" in module_two
    assert "checked_nci_interaction_curves" in module_two
    assert "helpers.summarize_nci_interaction_curves" in module_two
