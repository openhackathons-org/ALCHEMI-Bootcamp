from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ase.build import molecule
from ase.io import read
from helpers.lesson import (
    atoms_to_xyz,
    compare_device_calls,
    compare_warm_calls,
    freeze_model,
    load_molecule_collection,
    maximum_force_table,
    measure_call,
    model_checkpoint,
    molecule_result_table,
    plot_batch_ownership,
    plot_device_comparison,
    runtime_identity,
    summarize_samples,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import AIMNet2Wrapper
from nvalchemi.neighbors import compute_neighbors

ROOT = Path(__file__).resolve().parents[3]


def test_atoms_to_xyz_preserves_nonperiodic_molecule_geometry() -> None:
    water = molecule("H2O")

    restored = read(StringIO(atoms_to_xyz(water)), format="xyz")

    assert restored.get_chemical_symbols() == water.get_chemical_symbols()
    np.testing.assert_allclose(restored.positions, water.positions)
    assert not restored.pbc.any()


def test_pinned_molecule_collection_identity() -> None:
    atoms, frame = load_molecule_collection(ROOT)

    assert len(atoms) == len(frame) == 32
    assert sum(map(len, atoms)) == 322
    assert frame.loc[frame["label"] == "Ethyne", "atoms"].item() == 4
    assert frame.loc[frame["label"] == "Phenol", "atoms"].item() == 13
    assert frame.loc[frame["label"] == "2,3-dimethylbutane", "atoms"].item() == 20
    assert [structure.info["charge"] for structure in atoms] == frame["charge"].tolist()


def test_manifest_charge_flows_into_atomic_data() -> None:
    atoms, _ = load_molecule_collection(ROOT)

    graph = AtomicData.from_atoms(atoms[0], dtype=torch.float32)

    assert graph.charge.shape == (1, 1)
    assert graph.charge.item() == 0.0


def test_measure_call_returns_result_and_elapsed_time() -> None:
    elapsed_s, result = measure_call(lambda: 6 * 7, torch.device("cpu"))

    assert result == 42
    assert elapsed_s >= 0.0


def test_warm_call_comparison_preserves_order_and_uses_first_route_as_baseline() -> (
    None
):
    calls = {
        "individual calls": lambda: torch.arange(128).sum(),
        "one Batch": lambda: torch.arange(128).sum(),
    }

    frame = compare_warm_calls(
        calls, device=torch.device("cpu"), structures=4, atoms=16, repeats=2
    )

    assert frame["mode"].tolist() == list(calls)
    assert frame["execution device"].tolist() == ["CPU", "CPU"]
    assert frame.loc[0, "speedup vs individual calls"] == pytest.approx(1.0)
    assert frame["warm call median (ms)"].gt(0).all()
    assert frame["structures/s"].gt(0).all()


def test_warm_call_comparison_rejects_incomplete_studies() -> None:
    with pytest.raises(ValueError, match="at least two"):
        compare_warm_calls(
            {"one route": lambda: None},
            device=torch.device("cpu"),
            structures=1,
            atoms=1,
        )


def test_device_call_comparison_reports_cpu_timing_and_hardware() -> None:
    frame = compare_device_calls(
        {"CPU": (torch.device("cpu"), lambda: torch.arange(128).sum())},
        structures=4,
        atoms=16,
        repeats=2,
    )

    assert frame.loc[0, "mode"] == "CPU"
    assert frame.loc[0, "hardware"] == "CPU"
    assert frame.loc[0, "speedup vs CPU"] == pytest.approx(1.0)
    assert np.isnan(frame.loc[0, "peak CUDA memory (MiB)"])


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_device_plot_uses_nvidia_green_for_gpu() -> None:
    timing = np.array([20.0, 2.0])
    frame = pd.DataFrame(
        {
            "mode": ["CPU", "GPU"],
            "warm call median (ms)": timing,
            "molecules/s": [100.0, 1000.0],
        }
    )

    figure = plot_device_comparison(frame, structures=2048, atoms=20796)

    assert figure.axes[0].patches[1].get_facecolor()[:3] == pytest.approx(
        (118 / 255, 185 / 255, 0.0)
    )
    assert figure.axes[0].get_title() == "Evaluation time"
    assert figure.axes[0].get_ylabel() == "Elapsed time [s]"
    assert figure.axes[0].get_yscale() == "linear"
    assert [label.get_text() for label in figure.axes[0].texts] == [
        "0.020 s",
        "0.002 s",
    ]
    assert figure.axes[1].get_title() == "Throughput"
    assert figure.axes[1].get_ylabel() == "Throughput [molecules/s]"


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_batch_ownership_plot_uses_batch_boundaries() -> None:
    atomic_numbers = torch.tensor([1, 6, 8, 1, 1])
    batch_idx = torch.tensor([0, 0, 1, 1, 1])
    batch_ptr = torch.tensor([0, 2, 5])

    figure = plot_batch_ownership(atomic_numbers, batch_idx, batch_ptr, ["A", "B"])

    axis = figure.axes[0]
    assert len(axis.patches) == 7
    assert len(axis.lines) == 1
    labels = [text.get_text() for text in axis.texts]
    assert "A · 2 atoms" in labels
    assert "B · 3 atoms" in labels
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["0", "2", "5"]


def test_batch_ownership_plot_requires_one_label_per_molecule() -> None:
    with pytest.raises(ValueError, match="one name per molecule"):
        plot_batch_ownership(
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1, 2]),
            ["A"],
        )


def test_maximum_force_table_tracks_sampled_batch_positions() -> None:
    _, metadata = load_molecule_collection(ROOT)
    sampled_indices = [0, 1]
    forces = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

    frame = maximum_force_table(
        metadata, sampled_indices, forces, [2, 1], positions=[0, 1]
    )

    assert frame["batch position"].tolist() == [0, 1]
    assert frame["molecule"].tolist() == metadata.iloc[:2]["label"].tolist()
    assert frame["maximum force (eV/Å)"].tolist() == pytest.approx([5.0, 2.0])

    with pytest.raises(ValueError, match="positive"):
        compare_warm_calls(
            {"one": lambda: None, "two": lambda: None},
            device=torch.device("cpu"),
            structures=1,
            atoms=1,
            repeats=0,
        )


def test_molecule_result_table_selects_metadata_and_matching_energies() -> None:
    _, metadata_frame = load_molecule_collection(ROOT)
    energies = torch.arange(32, dtype=torch.float64).reshape(-1, 1)

    frame = molecule_result_table(metadata_frame, [0, 23, 31], energies)

    assert frame["molecule"].tolist() == [
        "Ethyne",
        "Phenol",
        "2,3-dimethylbutane",
    ]
    assert frame["energy (eV)"].tolist() == [0.0, 23.0, 31.0]


def test_freeze_model_preserves_module_and_freezes_parameters() -> None:
    model = torch.nn.Linear(2, 1)

    assert freeze_model(model) is model
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_runtime_identity_reports_every_execution_device() -> None:
    frame = runtime_identity({"CPU": torch.device("cpu")}).set_index("component")

    assert frame.loc["CPU device", "value"] == "CPU"
    assert {"PyTorch", "Toolkit", "Toolkit-Ops"} <= set(frame.index)


def test_timing_summary_uses_median_equal_work_time() -> None:
    row = summarize_samples("batch", [0.01, 0.02, 0.03], structures=32, atoms=322)

    assert row["warm call median (ms)"] == pytest.approx(20.0)
    assert row["spread p25–p75 (ms)"] == pytest.approx(10.0)
    assert row["structures/s"] == pytest.approx(1600.0)
    assert row["atoms/s"] == pytest.approx(16100.0)


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [[0.1]]])
def test_timing_summary_rejects_invalid_samples(samples: list[object]) -> None:
    with pytest.raises(ValueError):
        summarize_samples("bad", samples, structures=1, atoms=1)  # type: ignore[arg-type]


DEVICES = [pytest.param(torch.device("cpu"), id="cpu")]
if torch.cuda.is_available():
    DEVICES.append(pytest.param(torch.device("cuda"), id="cuda"))


@pytest.mark.parametrize("device", DEVICES)
def test_serial_and_batched_toolkit_paths_agree(device: torch.device) -> None:
    structures, _ = load_molecule_collection(ROOT)
    structures = [structures[index] for index in (0, 23, 31)]
    graphs = [AtomicData.from_atoms(atoms, dtype=torch.float32) for atoms in structures]

    serial_batches = [Batch.from_data_list([graph], device=device) for graph in graphs]
    batch = Batch.from_data_list(graphs, device=device)
    model = AIMNet2Wrapper.from_checkpoint(model_checkpoint(), device=device).eval()
    model = freeze_model(model)
    model.set_config("active_outputs", {"energy", "forces"})

    for serial_batch in serial_batches:
        compute_neighbors(serial_batch, config=model.model_config.neighbor_config)
    compute_neighbors(batch, config=model.model_config.neighbor_config)

    serial_outputs = [model(serial_batch) for serial_batch in serial_batches]
    batch_outputs = model(batch)
    serial_energy = torch.cat([output["energy"] for output in serial_outputs])
    serial_forces = torch.cat([output["forces"] for output in serial_outputs])

    torch.testing.assert_close(
        batch_outputs["energy"], serial_energy, rtol=0.0, atol=1.5e-3
    )
    torch.testing.assert_close(
        batch_outputs["forces"], serial_forces, rtol=0.0, atol=3.0e-5
    )

    node_counts = batch.num_nodes_per_graph.detach().cpu().tolist()
    batch.add_key(
        "energy",
        list(batch_outputs["energy"].detach().to(batch.positions.dtype).split(1)),
        level="system",
    )
    batch.add_key(
        "forces",
        list(batch_outputs["forces"].detach().split(node_counts)),
        level="node",
    )
    recovered = batch.to_data_list()

    assert [graph.num_nodes for graph in recovered] == [4, 13, 20]
    assert all(
        graph.energy is not None and graph.forces is not None for graph in recovered
    )
