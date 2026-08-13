"""Unit checks for BaseDynamics lesson inputs and presentation support."""

from __future__ import annotations

from pathlib import Path

import helpers
import pandas as pd
import pytest
import torch
from nvalchemi.data import AtomicData, Batch

LABELS = (
    "Ethyne",
    "Acetonitrile",
    "Methanol",
    "Acetaldehyde",
    "Acetamide",
    "Pyridine",
    "Phenol",
    "2,3-dimethylbutane",
)
UPDATE_COLUMN = "completed update (1-based)"
FIRST_CONVERGED_UPDATE_COLUMN = "first converged completed update (1-based)"


def test_repo_root_finds_shared_runtime_files() -> None:
    root = helpers.repo_root(Path(__file__))
    assert (root / "environment" / "runtime-pins.toml").is_file()
    assert (root / "shared" / "alchemi-dark.mplstyle").is_file()
    assert callable(helpers.configure_presentation)
    assert not hasattr(helpers, "start_lesson")


def test_selected_molecules_have_expected_identity() -> None:
    atoms, frame = helpers.load_molecule_selection(LABELS)
    assert frame["label"].tolist() == list(LABELS)
    assert frame["formula"].tolist() == [
        "C2H2",
        "C2H3N",
        "CH4O",
        "C2H4O",
        "C2H5NO",
        "C5H5N",
        "C6H6O",
        "C6H14",
    ]
    assert frame["atoms"].tolist() == [4, 6, 6, 7, 9, 11, 13, 20]
    assert len(atoms) == 8
    assert sum(len(item) for item in atoms) == 76
    assert frame["charge"].eq(0).all()
    assert [item.info["charge"] for item in atoms] == frame["charge"].tolist()


def test_checked_model_and_freezing_support() -> None:
    checkpoint = helpers.model_checkpoint()
    assert checkpoint.is_file()
    model = torch.nn.Linear(3, 1)
    assert helpers.freeze_model(model) is model
    assert all(not parameter.requires_grad for parameter in model.parameters())


def _example_batch() -> Batch:
    first = AtomicData(
        atomic_numbers=torch.tensor([1, 6]),
        atomic_masses=torch.tensor([1.008, 12.011]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocities=torch.zeros(2, 3),
        forces=torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        energy=torch.tensor([[-2.0]]),
    )
    first.add_system_property("status", torch.tensor([[1]]))
    second = AtomicData(
        atomic_numbers=torch.tensor([8]),
        atomic_masses=torch.tensor([15.999]),
        positions=torch.tensor([[0.0, 1.0, 0.0]]),
        velocities=torch.zeros(1, 3),
        forces=torch.tensor([[0.0, 0.0, 2.0]]),
        energy=torch.tensor([[-1.0]]),
    )
    second.add_system_property("status", torch.tensor([[0]]))
    return Batch.from_data_list([first, second])


def test_relaxation_monitor_records_per_system_force_and_status() -> None:
    monitor_type = getattr(helpers, "RelaxationMonitor", None)
    assert monitor_type is not None, "RelaxationMonitor must be public local support"
    monitor = monitor_type(("first", "second"), total_steps=10)

    monitor.record(_example_batch(), completed_update=3)
    history = monitor.history_frame()

    assert history["molecule"].tolist() == ["first", "second"]
    assert history[UPDATE_COLUMN].tolist() == [3, 3]
    assert history["fmax (eV/Å)"].tolist() == pytest.approx([5.0, 2.0])
    assert history["energy (eV)"].tolist() == pytest.approx([-2.0, -1.0])
    assert history["status"].tolist() == [1, 0]


def test_per_system_fmax_matches_graph_boundaries() -> None:
    fmax = helpers.per_system_fmax(_example_batch())

    assert fmax.tolist() == pytest.approx([5.0, 2.0])


def test_relaxation_summary_preserves_source_order_and_outcomes() -> None:
    summarize = getattr(helpers, "summarize_relaxation", None)
    assert summarize is not None, "summarize_relaxation must be public local support"
    batch = _example_batch()
    initial_positions = batch.positions.clone()
    batch.positions[0, 0] += 0.2
    metadata = pd.DataFrame(
        {"label": ["first", "second"], "formula": ["CH", "O"], "atoms": [2, 1]}
    )
    history = pd.DataFrame(
        [
            {UPDATE_COLUMN: 1, "molecule": "first", "energy (eV)": -1.8, "fmax (eV/Å)": 5.0, "status": 0},
            {UPDATE_COLUMN: 1, "molecule": "second", "energy (eV)": -0.9, "fmax (eV/Å)": 2.0, "status": 0},
            {UPDATE_COLUMN: 3, "molecule": "first", "energy (eV)": -2.0, "fmax (eV/Å)": 0.1, "status": 1},
            {UPDATE_COLUMN: 3, "molecule": "second", "energy (eV)": -1.0, "fmax (eV/Å)": 1.0, "status": 0},
        ]
    )

    summary = summarize(metadata, history, initial_positions, batch)

    assert summary["molecule"].tolist() == ["first", "second"]
    assert summary["outcome"].tolist() == ["converged", "update limit"]
    assert summary[FIRST_CONVERGED_UPDATE_COLUMN].tolist()[0] == 3
    assert pd.isna(summary[FIRST_CONVERGED_UPDATE_COLUMN].tolist()[1])
    assert summary["final fmax (eV/Å)"].tolist() == pytest.approx([5.0, 2.0])
    assert summary["energy change (eV)"].tolist() == pytest.approx([-0.2, -0.1])
    assert summary["maximum displacement (Å)"].tolist() == pytest.approx([0.2, 0.0])


def test_history_stops_each_system_at_first_convergence() -> None:
    history = pd.DataFrame(
        {
            UPDATE_COLUMN: [1, 2, 3, 1, 2, 3],
            "molecule": ["first"] * 3 + ["second"] * 3,
            "energy (eV)": [9.0] * 6,
            "fmax (eV/Å)": [9.0] * 6,
            "status": [0, 1, 1, 0, 0, 0],
        }
    )

    truncated = helpers.truncate_history_at_convergence(history)

    assert truncated.groupby("molecule")[UPDATE_COLUMN].apply(list).to_dict() == {
        "first": [1, 2],
        "second": [1, 2, 3],
    }
    coherent = helpers.truncate_history_at_convergence(
        history, final_batch=_example_batch()
    ).groupby("molecule", sort=False).last()
    assert coherent["energy (eV)"].tolist() == pytest.approx([-2.0, -1.0])
    assert coherent["fmax (eV/Å)"].tolist() == pytest.approx([5.0, 2.0])
    assert coherent["status"].tolist() == [1, 0]


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_relaxation_visuals_use_green_for_results_and_show_units() -> None:
    plot_history = getattr(helpers, "plot_force_history", None)
    plot_change = getattr(helpers, "plot_structure_change", None)
    assert plot_history is not None and plot_change is not None
    history = pd.DataFrame(
        {
            UPDATE_COLUMN: [1, 2, 1, 2],
            "molecule": ["first", "first", "second", "second"],
            "fmax (eV/Å)": [1.0, 0.1, 2.0, 0.5],
        }
    )

    force_figure = plot_history(history, target=0.2)
    change_figure = plot_change(
        torch.tensor([1, 6]),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]]),
        label="first",
    )

    assert force_figure.axes[0].get_ylabel() == "Maximum force (eV/Å)"
    assert force_figure.axes[0].get_xlabel() == "Completed update (1-based)"
    assert force_figure.axes[0].lines[-1].get_color() == "#76B900"
    assert change_figure.axes[0].collections[-1].get_facecolor()[0][:3] == pytest.approx(
        (118 / 255, 185 / 255, 0.0)
    )


def test_periodic_argon_geometry_matches_official_mechanics_route() -> None:
    positions, cell = helpers.periodic_argon_geometry(
        n_side=3,
        spacing=2 ** (1 / 6) * 3.40,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert positions.shape == (27, 3)
    assert cell.shape == (1, 3, 3)
    assert positions.dtype == torch.float32
    assert torch.allclose(cell[0].diag(), torch.full((3,), 3 * 2 ** (1 / 6) * 3.40))
    assert torch.equal(cell[0] - torch.diag(cell[0].diag()), torch.zeros(3, 3))


def test_dynamics_trace_records_energy_temperature_status_and_positions() -> None:
    batch = _example_batch()
    trace = helpers.DynamicsTrace(("first", "second"), store_positions=True)

    trace.record(batch, completed_update=0)
    history = trace.history_frame()

    assert history["system"].tolist() == ["first", "second"]
    assert history["completed update (1-based)"].tolist() == [0, 0]
    assert history["potential energy (eV)"].tolist() == pytest.approx([-2.0, -1.0])
    assert history["kinetic energy (eV)"].tolist() == pytest.approx([0.0, 0.0])
    assert history["total energy (eV)"].tolist() == pytest.approx([-2.0, -1.0])
    assert history["temperature (K)"].tolist() == pytest.approx([0.0, 0.0])
    assert history["status"].tolist() == [1, 0]
    assert len(trace.position_frames) == 1
    assert torch.equal(trace.position_frames[0], batch.positions.cpu())


def test_temperature_matches_toolkit_three_n_convention() -> None:
    batch = _example_batch()
    batch.velocities.fill_(1.0)

    kinetic = helpers.kinetic_energy_per_system(batch)
    temperature = helpers.temperature_per_system(batch)
    counts = batch.num_nodes_per_graph.to(dtype=kinetic.dtype)
    expected = 2 * kinetic / (3 * counts * 8.617333262e-5)

    torch.testing.assert_close(temperature, expected)


def test_energy_conservation_summary_uses_per_atom_drift() -> None:
    history = pd.DataFrame(
        {
            "completed update (1-based)": [0, 1, 2],
            "system": ["argon"] * 3,
            "potential energy (eV)": [-1.0, -0.9, -0.8],
            "kinetic energy (eV)": [0.5, 0.4, 0.3],
            "total energy (eV)": [-0.5, -0.5, -0.5 + 2.7e-5],
            "temperature (K)": [50.0, 40.0, 30.0],
            "status": [0, 0, 0],
        }
    )

    summary = helpers.energy_conservation_summary(history, atoms_per_system=27)

    assert summary["updates"] == 2
    assert summary["maximum |ΔE| (eV)"] == pytest.approx(2.7e-5)
    assert summary["maximum |ΔE| per atom (meV/atom)"] == pytest.approx(0.001)
    assert summary[
        "maximum |ΔE| per atom per update (eV/atom/update)"
    ] == pytest.approx(5e-7)
    assert summary["final temperature (K)"] == pytest.approx(30.0)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_md_visuals_have_units_green_results_and_matched_frames() -> None:
    history = pd.DataFrame(
        {
            "completed update (1-based)": [0, 1, 2],
            "system": ["argon"] * 3,
            "potential energy (eV)": [-1.0, -0.9, -0.8],
            "kinetic energy (eV)": [0.5, 0.4, 0.3],
            "total energy (eV)": [-0.5, -0.5, -0.4999],
            "temperature (K)": [5.0, 25.0, 45.0],
            "status": [0, 0, 0],
        }
    )
    frames = [
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0]]),
        torch.tensor([[0.2, 0.0, 0.0], [1.0, 1.2, 1.0]]),
    ]

    energy = helpers.plot_energy_conservation(history, atoms_per_system=2)
    temperature = helpers.plot_temperature_history(history, target=50.0)
    trajectory = helpers.plot_argon_trajectory(frames, box_size=4.0)

    assert energy.axes[0].get_ylabel() == "Energy (eV)"
    assert energy.axes[1].get_ylabel() == "ΔE per atom (meV/atom)"
    assert energy.axes[0].lines[-1].get_color() == "#76B900"
    assert temperature.axes[0].get_ylabel() == "Instantaneous temperature (K)"
    assert temperature.axes[0].lines[0].get_color() == "#76B900"
    assert len(trajectory.axes) == 3
    assert all(axis.get_xlabel() == "x (Å)" for axis in trajectory.axes)
