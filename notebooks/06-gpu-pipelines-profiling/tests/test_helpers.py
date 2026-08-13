"""Unit checks for Part 06 inputs, measurement, and presentation support."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import helpers
import pandas as pd
import pytest
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.hooks import StageTimingHook

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


def test_repo_root_and_runtime_identity_find_pinned_inputs() -> None:
    root = helpers.repo_root(Path(__file__))
    identity = helpers.runtime_identity(torch.device("cpu"))

    assert (root / "environment" / "runtime-pins.toml").is_file()
    manifest = root / "data" / "nci_atlas" / "ir-molecule-library-manifest.json"
    assert manifest.is_file()
    assert identity["execution device"] == "cpu"
    assert identity["CUDA available"] is False
    assert identity["PyTorch CUDA build"] == (torch.version.cuda or "unavailable")
    assert "CUDA runtime" not in identity
    assert identity["PyTorch"]
    assert identity["CPU name"]
    assert identity["CPU name"] != "unavailable"
    assert callable(helpers.configure_presentation)


def test_synthetic_graphs_are_deterministic_and_mixed_dtype() -> None:
    first = helpers.make_synthetic_graphs((3, 5, 4), seed=17)
    second = helpers.make_synthetic_graphs((3, 5, 4), seed=17)

    assert [graph.num_nodes for graph in first] == [3, 5, 4]
    assert torch.equal(first[1].positions, second[1].positions)
    assert first[0].positions.dtype == torch.float32
    assert first[0].atomic_numbers.dtype == torch.int64
    assert first[0].positions.device.type == "cpu"
    assert first[0].source_index.item() == 0


def test_timing_helper_warms_then_returns_requested_samples() -> None:
    device = torch.device("cpu")
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    result, timings = helpers.time_callable(
        lambda tensor: tensor.square().sum(dim=1),
        values,
        device=device,
        warmup=2,
        repeats=4,
    )

    assert result.shape == (4,)
    assert timings.columns.tolist() == ["sample", "elapsed (ms)"]
    assert timings["sample"].tolist() == [1, 2, 3, 4]
    assert (timings["elapsed (ms)"] >= 0.0).all()


def test_cpu_memory_snapshot_labels_cuda_values_unavailable() -> None:
    snapshot = helpers.device_memory_snapshot(torch.device("cpu"))

    assert snapshot["CUDA memory available"] is False
    assert snapshot["allocated (MiB)"] is None
    assert snapshot["reserved (MiB)"] is None
    assert snapshot["peak allocated (MiB)"] is None


def test_native_stderr_filter_keeps_unexpected_diagnostics(
    capfd: pytest.CaptureFixture,
) -> None:
    with helpers.filter_known_native_stderr(torch.device("cpu")):
        os.write(
            2,
            b"Warp CUDA error: Failed to get driver entry point "
            b"'cuInit' (CUDA error 100)\n",
        )
        os.write(2, b"unexpected native diagnostic\n")
    os.write(2, b"visible diagnostic\n")

    captured = capfd.readouterr()
    assert "Failed to get driver entry point" not in captured.err
    assert "unexpected native diagnostic" in captured.err
    assert "visible diagnostic" in captured.err


def test_native_stderr_filter_suppresses_nothing_on_cuda(
    capfd: pytest.CaptureFixture,
) -> None:
    with helpers.filter_known_native_stderr(torch.device("cuda")):
        os.write(
            2,
            b"Warp CUDA error: Failed to get driver entry point "
            b"'cuInit' (CUDA error 100)\n",
        )

    assert "Failed to get driver entry point" in capfd.readouterr().err


def test_selected_molecules_and_dataset_preserve_source_identity() -> None:
    atoms, metadata = helpers.load_molecule_selection(LABELS)
    dataset = helpers.MoleculeDataset(atoms, metadata)
    first, sample_metadata = dataset[0]

    assert metadata["label"].tolist() == list(LABELS)
    assert metadata["formula"].tolist() == [
        "C2H2",
        "C2H3N",
        "CH4O",
        "C2H4O",
        "C2H5NO",
        "C5H5N",
        "C6H6O",
        "C6H14",
    ]
    assert metadata["atoms"].tolist() == [4, 6, 6, 7, 9, 11, 13, 20]
    assert dataset.get_metadata(0) == (4, 0)
    assert first.source_index.item() == 0
    assert first.atomic_masses.shape == (4,)
    assert first.velocities.shape == (4, 3)
    assert first.forces.shape == (4, 3)
    assert first.energy.shape == (1, 1)
    assert first.charge.shape == (1, 1)
    assert sample_metadata == {"source_index": 0, "label": "Ethyne"}


def test_checked_checkpoint_and_model_freezing_support() -> None:
    checkpoint = helpers.model_checkpoint()
    model = torch.nn.Linear(3, 1)

    assert checkpoint.is_file()
    assert helpers.freeze_model(model) is model
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_pipeline_monitor_records_public_context_fields() -> None:
    graphs = helpers.make_synthetic_graphs((2, 3), seed=4)
    graphs[0].add_system_property("status", torch.tensor([[0]]))
    graphs[1].add_system_property("status", torch.tensor([[1]]))
    batch = Batch.from_data_list(graphs)
    context = SimpleNamespace(
        batch=batch,
        stage=DynamicsStage.AFTER_STEP,
        step_count=2,
    )
    monitor = helpers.PipelineMonitor(device=torch.device("cpu"))

    monitor(context)
    history = monitor.frame()

    assert history["completed update (1-based)"].tolist() == [3]
    assert history["active systems"].tolist() == [2]
    assert history["active atoms"].tolist() == [5]
    assert history["status 0"].tolist() == [1]
    assert history["status 1"].tolist() == [1]
    assert history["status 2"].tolist() == [0]
    assert history["allocated (MiB)"].isna().all()


def test_stage_timing_frame_uses_public_timings_mapping() -> None:
    hook = StageTimingHook("step", enable_nvtx=False)
    hook.timings[DynamicsStage.AFTER_STEP] = [0.001, 0.0025]
    frame = helpers.stage_timing_frame(hook)

    assert frame["completed update (1-based)"].tolist() == [1, 2]
    assert frame["step time (ms)"].tolist() == pytest.approx([1.0, 2.5])


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_measurement_plots_label_time_and_cuda_memory() -> None:
    import matplotlib.pyplot as plt

    timings = pd.DataFrame(
        {
            "implementation": ["eager warm", "eager warm", "compiled warm"],
            "elapsed (ms)": [1.0, 1.2, 0.8],
        }
    )
    memory = helpers.device_memory_snapshot(torch.device("cpu"))
    stage = pd.DataFrame(
        {
            "completed update (1-based)": [1, 2],
            "step time (ms)": [1.1, 0.9],
        }
    )
    occupancy = pd.DataFrame(
        {
            "completed update (1-based)": [1, 2],
            "active systems": [2, 2],
            "active atoms": [7, 6],
        }
    )

    measurement = helpers.plot_measurement_panels(timings, memory)
    pipeline = helpers.plot_pipeline_diagnostics(stage, occupancy)
    rendered = helpers.render_figure(
        measurement,
        alt_text='Timing & memory "evidence"',
    )

    assert measurement.axes[0].get_ylabel() == "Synchronized wall time (ms, log scale)"
    assert measurement.axes[0].get_yscale() == "log"
    assert measurement.axes[1].get_ylabel() == "CUDA memory (MiB)"
    assert len(measurement.axes[1].get_yticks()) == 0
    assert "unavailable" in measurement.axes[1].texts[0].get_text().lower()
    assert pipeline.axes[0].get_ylabel() == "Stage time (ms)"
    assert pipeline.axes[1].get_ylabel() == "Active count"
    assert pipeline.axes[0].lines[0].get_color() == "#76B900"
    assert not plt.fignum_exists(measurement.number)
    assert not plt.fignum_exists(pipeline.number)
    assert 'alt="Timing &amp; memory &quot;evidence&quot;"' in rendered.data
    assert 'src="data:image/png;base64,' in rendered.data


def test_profile_artifacts_lists_rank_relative_paths(tmp_path: Path) -> None:
    trace = tmp_path / "rank_0" / "trace.json"
    trace.parent.mkdir()
    trace.write_text('{"traceEvents": []}')

    artifacts = helpers.profile_artifacts(tmp_path)

    assert artifacts["relative path"].tolist() == ["rank_0/trace.json"]
    assert artifacts["bytes"].iloc[0] > 0


def test_mixed_dtype_buffer_probe_is_explicitly_unavailable_without_cuda() -> None:
    report = helpers.probe_gpu_buffer_mixed_dtype(torch.device("cpu"))

    assert report["probe ran"] is False
    assert report["reason"] == "GPUBuffer requires CUDA"
    assert report["float positions preserved"] is None
    assert report["integer atomic numbers preserved"] is None
    assert report["integer source indices preserved"] is None
    assert "integer system IDs preserved" not in report
