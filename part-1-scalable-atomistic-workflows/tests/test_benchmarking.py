"""Focused CPU tests for transparent Part 1 benchmarking mechanics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from pathlib import Path
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import benchmarking  # noqa: E402


@dataclass(frozen=True)
class _Device:
    type: str
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


@dataclass
class _Batch:
    name: str
    num_graphs: int
    num_nodes: int
    batch_ptr: np.ndarray
    atomic_numbers: np.ndarray
    positions: np.ndarray
    device: object = "cpu"


class _Model:
    def __init__(self, energies: dict[str, list[float]]) -> None:
        self.energies = energies
        self.calls: list[str] = []

    def __call__(self, batch: _Batch) -> dict[str, np.ndarray]:
        self.calls.append(batch.name)
        return {"energy": np.asarray(self.energies[batch.name])[:, None]}


@dataclass
class _Atoms:
    graph_size: int
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _AtomicDatum:
    atoms: _Atoms
    device: object
    dtype: object


class _AtomicDataFactory:
    def __init__(self) -> None:
        self.calls: list[_AtomicDatum] = []

    def from_atoms(
        self, atoms: _Atoms, *, device: object, dtype: object
    ) -> _AtomicDatum:
        datum = _AtomicDatum(atoms=atoms, device=device, dtype=dtype)
        self.calls.append(datum)
        return datum


class _BatchFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[list[_AtomicDatum], object]] = []

    def from_data_list(self, data: list[_AtomicDatum], *, device: object) -> _Batch:
        self.calls.append((list(data), device))
        return _batch(
            str(device),
            [datum.atoms.graph_size for datum in data],
            device=device,
        )


class _SweepModel:
    def __init__(self, energy_offset: float = 0.0) -> None:
        self.energy_offset = energy_offset
        self.calls: list[int] = []

    def __call__(self, batch: _Batch) -> dict[str, np.ndarray]:
        self.calls.append(batch.num_graphs)
        energies = np.arange(batch.num_graphs, dtype=float) + self.energy_offset
        return {"energy": energies[:, None]}


def _batch(
    name: str,
    graph_sizes: list[int],
    *,
    device: object = "cpu",
    position_offset: float = 0.0,
) -> _Batch:
    ptr = np.cumsum([0, *graph_sizes])
    atoms = int(ptr[-1])
    numbers = np.resize(np.array([8, 1, 1]), atoms)
    positions = np.arange(atoms * 3, dtype=float).reshape(atoms, 3)
    positions += position_offset
    return _Batch(
        name=name,
        num_graphs=len(graph_sizes),
        num_nodes=atoms,
        batch_ptr=ptr,
        atomic_numbers=numbers,
        positions=positions,
        device=device,
    )


def _clock(values: list[float]) -> Callable[[], float]:
    times = iter(values)
    return lambda: next(times)


def _block_clock(durations: list[float]) -> Callable[[], float]:
    """Return start/end clock values for consecutive measured blocks."""

    values = []
    start = 0.0
    for duration in durations:
        values.extend((start, start + duration))
        start += duration + 10.0
    return _clock(values)


def _no_synchronize(batch: object) -> None:
    del batch


def test_public_benchmark_signatures_hide_timing_internals() -> None:
    for helper in (
        benchmarking.benchmark_device_sweep,
        benchmarking.build_benchmark_batch,
        benchmarking.first_and_warm_call_rows,
        benchmarking.compare_fixed_workload_devices,
        benchmarking.compare_mixed_and_bucketed,
    ):
        parameters = signature(helper).parameters
        assert "clock" not in parameters
        assert "synchronize" not in parameters


def test_device_sweep_keeps_benchmark_choices_explicit() -> None:
    parameters = signature(benchmarking.benchmark_device_sweep).parameters

    for name in (
        "batch_sizes",
        "structure_factory",
        "atoms_info",
        "routes",
        "dtype",
        "atomic_data_type",
        "batch_type",
        "compute_neighbors",
        "warmup_calls",
        "measured_calls",
        "measured_repeats",
        "energy_key",
        "energy_atol",
        "energy_rtol",
    ):
        assert parameters[name].default is Parameter.empty
    assert (
        signature(benchmarking.compare_mixed_and_bucketed)
        .parameters["measured_repeats"]
        .default
        is Parameter.empty
    )
    assert (
        signature(benchmarking.compare_fixed_workload_devices)
        .parameters["measured_repeats"]
        .default
        is Parameter.empty
    )


def test_build_benchmark_batch_uses_supplied_toolkit_types_and_dtype() -> None:
    atomic_data = _AtomicDataFactory()
    batches = _BatchFactory()
    atoms = [_Atoms(3), _Atoms(6)]

    batch = benchmarking.build_benchmark_batch(
        atoms,
        device="cuda:2",
        dtype="float64",
        atomic_data_type=atomic_data,
        batch_type=batches,
    )

    assert batch.num_graphs == 2
    assert batch.num_nodes == 9
    assert batch.device == "cuda:2"
    assert [call.atoms for call in atomic_data.calls] == atoms
    assert {call.device for call in atomic_data.calls} == {"cuda:2"}
    assert {call.dtype for call in atomic_data.calls} == {"float64"}
    assert len(batches.calls) == 1


def test_device_sweep_builds_matches_times_and_reports_each_size() -> None:
    atomic_data = _AtomicDataFactory()
    batches = _BatchFactory()
    gpu_model = _SweepModel()
    cpu_model = _SweepModel(energy_offset=1.0e-5)
    structures_built: list[_Atoms] = []
    neighbor_calls: list[tuple[str, object]] = []
    completed: list[tuple[int, int]] = []

    def make_structure() -> _Atoms:
        atoms = _Atoms(3)
        structures_built.append(atoms)
        return atoms

    results = benchmarking._benchmark_device_sweep_with_dependencies(
        batch_sizes=(1, 3),
        structure_factory=make_structure,
        atoms_info={"charge": 0},
        routes={
            "GPU": (gpu_model, "cuda:0", "gpu-neighbors"),
            "CPU": (cpu_model, "cpu", "cpu-neighbors"),
        },
        dtype="float64",
        atomic_data_type=atomic_data,
        batch_type=batches,
        compute_neighbors=lambda batch, *, config: neighbor_calls.append(
            (batch.name, config)
        ),
        warmup_calls=1,
        measured_calls=2,
        measured_repeats=3,
        energy_key="energy",
        energy_atol=2.0e-5,
        energy_rtol=0.0,
        on_batch_complete=lambda done, size: completed.append((done, size)),
        clock=_block_clock(
            [2.0, 1.0, 3.0, 4.0, 2.0, 6.0, 3.0, 2.0, 4.0, 6.0, 4.0, 8.0]
        ),
        synchronize=_no_synchronize,
    )

    assert len(structures_built) == 4
    assert all(atoms.info == {"charge": 0} for atoms in structures_built)
    assert len(atomic_data.calls) == 8
    assert len(batches.calls) == 4
    assert neighbor_calls == [
        ("cuda:0", "gpu-neighbors"),
        ("cpu", "cpu-neighbors"),
        ("cuda:0", "gpu-neighbors"),
        ("cpu", "cpu-neighbors"),
    ]
    assert gpu_model.calls == [1] * 8 + [3] * 8
    assert cpu_model.calls == [1] * 8 + [3] * 8
    assert completed == [(1, 1), (2, 3)]
    assert results["batch_size"].tolist() == [1, 1, 3, 3]
    assert results["route"].tolist() == ["GPU", "CPU", "GPU", "CPU"]
    assert set(results["dtype"]) == {"float64"}
    assert set(results["energy_key"]) == {"energy"}
    assert set(results["energy_atol"]) == {2.0e-5}
    assert set(results["energy_rtol"]) == {0.0}
    assert set(results["reference_route"]) == {"GPU"}
    assert set(results["passes_per_repeat"]) == {2}
    assert set(results["passes"]) == {6}
    assert set(results["measured_repeats"]) == {3}
    assert set(results["calls"]) == {6}
    assert set(results["warmup_calls"]) == {1}
    assert set(results["validation_calls"]) == {1}
    assert set(results["total_calls_executed"]) == {8}
    assert results.loc[0, "elapsed_samples_s"] == (2.0, 1.0, 3.0)
    assert results.loc[0, "elapsed_median_s"] == pytest.approx(2.0)
    assert results.loc[0, "elapsed_q1_s"] == pytest.approx(1.5)
    assert results.loc[0, "elapsed_q3_s"] == pytest.approx(2.5)
    assert results.loc[0, "elapsed_iqr_s"] == pytest.approx(1.0)
    assert results.loc[0, "relative_iqr"] == pytest.approx(0.5)
    assert results.loc[0, "median_structures_per_s"] == pytest.approx(1.0)
    assert results.loc[0, "structures_per_s_q1"] == pytest.approx(5.0 / 6.0)
    assert results.loc[0, "structures_per_s_q3"] == pytest.approx(1.5)
    assert results.loc[0, "structures_per_s_iqr"] == pytest.approx(2.0 / 3.0)
    assert results.loc[0, "median_atoms_per_s"] == pytest.approx(3.0)
    assert results.loc[0, "atoms_per_s_q1"] == pytest.approx(2.5)
    assert results.loc[0, "atoms_per_s_q3"] == pytest.approx(4.5)
    assert results.loc[0, "atoms_per_s_iqr"] == pytest.approx(2.0)
    assert results.loc[0, "structures_per_s"] == pytest.approx(1.0)
    assert results.loc[0, "atoms_per_s"] == pytest.approx(3.0)
    np.testing.assert_allclose(results["max_abs_energy_difference"], 1.0e-5)


def test_device_sweep_stops_before_timing_when_energies_disagree() -> None:
    atomic_data = _AtomicDataFactory()
    batches = _BatchFactory()
    gpu_model = _SweepModel()
    cpu_model = _SweepModel(energy_offset=0.2)

    with pytest.raises(AssertionError, match="CPU energies differ from GPU"):
        benchmarking._benchmark_device_sweep_with_dependencies(
            batch_sizes=(2,),
            structure_factory=lambda: _Atoms(3),
            atoms_info={"charge": 0},
            routes={
                "GPU": (gpu_model, "cuda:0", "gpu-neighbors"),
                "CPU": (cpu_model, "cpu", "cpu-neighbors"),
            },
            dtype="float64",
            atomic_data_type=atomic_data,
            batch_type=batches,
            compute_neighbors=lambda batch, *, config: None,
            warmup_calls=1,
            measured_calls=2,
            measured_repeats=3,
            energy_key="energy",
            energy_atol=1.0e-4,
            energy_rtol=0.0,
            on_batch_complete=None,
            clock=lambda: pytest.fail("timing must not start after failed parity"),
            synchronize=_no_synchronize,
        )

    assert gpu_model.calls == [2]
    assert cpu_model.calls == [2]


def test_plot_device_sweep_draws_one_sorted_line_per_route() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    results = pd.DataFrame(
        {
            "batch_size": [8, 1, 8, 1],
            "route": ["GPU", "GPU", "CPU", "CPU"],
            "median_structures_per_s": [800.0, 20.0, 120.0, 40.0],
            "structures_per_s_q1": [700.0, 15.0, 100.0, 30.0],
            "structures_per_s_q3": [900.0, 25.0, 140.0, 50.0],
        }
    )

    figure, axis = benchmarking.plot_device_sweep(results)

    assert figure.axes == [axis]
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert axis.get_xlabel() == "batch size (structures)"
    assert axis.get_ylabel() == "median throughput (structures/s)"
    assert [line.get_label() for line in axis.lines] == ["GPU", "CPU"]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 8])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [20.0, 800.0])
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), [1, 8])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), [40.0, 120.0])
    assert len(axis.collections) == 2
    np.testing.assert_array_equal(
        axis.collections[0].get_segments()[0],
        [[1.0, 15.0], [1.0, 25.0]],
    )
    plt.close(figure)


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (
            {"batch_size": [1], "route": ["GPU"]},
            "missing median_structures_per_s",
        ),
        (
            {
                "batch_size": [1],
                "route": ["GPU"],
                "median_structures_per_s": [0.0],
                "structures_per_s_q1": [1.0],
                "structures_per_s_q3": [1.0],
            },
            "median_structures_per_s values must be finite and positive",
        ),
        (
            {
                "batch_size": [float("inf")],
                "route": ["GPU"],
                "median_structures_per_s": [1.0],
                "structures_per_s_q1": [1.0],
                "structures_per_s_q3": [1.0],
            },
            "batch_size values must be finite and positive",
        ),
        (
            {
                "batch_size": [1],
                "route": ["GPU"],
                "median_structures_per_s": [1.0],
                "structures_per_s_q1": [1.2],
                "structures_per_s_q3": [1.4],
            },
            "quartiles must satisfy",
        ),
    ],
)
def test_plot_device_sweep_rejects_incomplete_or_non_positive_results(
    results: dict[str, list[object]], message: str
) -> None:
    import pandas as pd

    with pytest.raises(ValueError, match=message):
        benchmarking.plot_device_sweep(pd.DataFrame(results))


def test_first_and_warm_calls_report_aggregate_timing() -> None:
    batch = _batch("mixed", [3, 3])
    model = _Model({"mixed": [1.0, 2.0]})

    first, warm = benchmarking._first_and_warm_call_rows_with_dependencies(
        model,
        batch,
        warm_calls=4,
        route="AIMNet2 pipeline",
        clock=_clock([0.0, 2.0, 10.0, 14.0]),
        synchronize=_no_synchronize,
    )

    assert model.calls == ["mixed"] * 5
    assert first["phase"] == "first call"
    assert first["calls"] == 1
    assert first["graphs"] == 2
    assert first["atoms"] == 6
    assert first["elapsed_s"] == pytest.approx(2.0)
    assert first["structures_per_s"] == pytest.approx(1.0)
    assert first["atoms_per_s"] == pytest.approx(3.0)
    assert warm["phase"] == "warm calls"
    assert warm["calls"] == 4
    assert warm["elapsed_s"] == pytest.approx(4.0)
    assert warm["structures_per_s"] == pytest.approx(2.0)
    assert warm["atoms_per_s"] == pytest.approx(6.0)


def test_cuda_calls_are_synchronized_around_each_measured_block() -> None:
    device = _Device("cuda", 1)
    batch = _batch("gpu", [3], device=device)
    model = _Model({"gpu": [1.0]})
    synchronized: list[object] = []

    benchmarking._first_and_warm_call_rows_with_dependencies(
        model,
        batch,
        warm_calls=2,
        clock=_clock([0.0, 1.0, 5.0, 7.0]),
        synchronize=lambda selected: synchronized.append(selected.device),
    )

    assert synchronized == [device, device, device, device]


def test_fixed_workload_device_comparison_times_caller_built_batches() -> None:
    cpu_batch = _batch("cpu", [3, 6], device=_Device("cpu"))
    gpu_batch = _batch("gpu", [3, 6], device=_Device("cuda"))
    cpu_model = _Model({"cpu": [1.0, 2.0]})
    gpu_model = _Model({"gpu": [1.0, 2.0]})
    synchronized: list[str] = []

    rows = benchmarking._compare_fixed_workload_devices_with_dependencies(
        {"CPU": (cpu_model, cpu_batch), "GPU": (gpu_model, gpu_batch)},
        warmup_calls=2,
        measured_calls=3,
        measured_repeats=3,
        clock=_block_clock([3.0, 6.0, 9.0, 1.0, 2.0, 4.0]),
        synchronize=lambda batch: synchronized.append(batch.name),
    )

    assert cpu_model.calls == ["cpu"] * 11
    assert gpu_model.calls == ["gpu"] * 11
    assert synchronized == ["cpu"] * 5 + ["gpu"] * 5
    assert [row["passes_per_repeat"] for row in rows] == [3, 3]
    assert [row["passes"] for row in rows] == [9, 9]
    assert [row["measured_repeats"] for row in rows] == [3, 3]
    assert [row["calls"] for row in rows] == [9, 9]
    assert [row["warmup_calls"] for row in rows] == [2, 2]
    assert [row["total_calls_executed"] for row in rows] == [11, 11]
    assert all(row["graphs"] == 2 and row["atoms"] == 9 for row in rows)
    assert rows[0]["elapsed_samples_s"] == (3.0, 6.0, 9.0)
    assert rows[0]["median_structures_per_s"] == pytest.approx(1.0)
    assert rows[1]["median_structures_per_s"] == pytest.approx(3.0)
    assert rows[0]["median_atoms_per_s"] == pytest.approx(4.5)
    assert rows[1]["median_atoms_per_s"] == pytest.approx(13.5)


def test_fixed_workload_mismatch_fails_before_model_calls() -> None:
    reference = _batch("cpu", [3, 3])
    changed = _batch("gpu", [3, 3], position_offset=0.25)
    first = _Model({"cpu": [1.0, 2.0]})
    second = _Model({"gpu": [1.0, 2.0]})

    with pytest.raises(ValueError, match="positions differ"):
        benchmarking.compare_fixed_workload_devices(
            {"CPU": (first, reference), "GPU": (second, changed)},
            measured_calls=1,
            measured_repeats=3,
        )

    assert first.calls == []
    assert second.calls == []


def test_mixed_and_bucketed_comparison_restores_energy_order() -> None:
    mixed = _batch("mixed", [2, 4, 2, 4])
    small = _batch("small", [2, 2])
    large = _batch("large", [4, 4])
    model = _Model(
        {
            "mixed": [10.0, 20.0, 30.0, 40.0],
            "small": [10.0, 30.0],
            "large": [20.0, 40.0],
        }
    )
    result = benchmarking._compare_mixed_and_bucketed_with_dependencies(
        model,
        mixed,
        [small, large],
        [[0, 2], [1, 3]],
        warmup_passes=1,
        measured_passes=2,
        measured_repeats=3,
        clock=_block_clock([2.0, 4.0, 3.0, 4.0, 8.0, 6.0]),
        synchronize=_no_synchronize,
    )

    np.testing.assert_array_equal(
        result["bucketed_energies_in_mixed_order"],
        [10.0, 20.0, 30.0, 40.0],
    )
    assert result["max_abs_energy_difference"] == 0.0
    mixed_row, bucket_row = result["timings"]
    assert mixed_row["passes_per_repeat"] == 2
    assert mixed_row["passes"] == 6
    assert mixed_row["measured_repeats"] == 3
    assert mixed_row["calls_per_pass"] == 1
    assert mixed_row["calls"] == 6
    assert bucket_row["calls_per_pass"] == 2
    assert bucket_row["calls"] == 12
    assert mixed_row["graphs"] == bucket_row["graphs"] == 4
    assert mixed_row["atoms"] == bucket_row["atoms"] == 12
    assert mixed_row["elapsed_median_s"] == pytest.approx(3.0)
    assert mixed_row["elapsed_q1_s"] == pytest.approx(2.5)
    assert mixed_row["elapsed_q3_s"] == pytest.approx(3.5)
    assert mixed_row["elapsed_iqr_s"] == pytest.approx(1.0)
    assert mixed_row["relative_iqr"] == pytest.approx(1.0 / 3.0)
    assert mixed_row["median_structures_per_s"] == pytest.approx(8.0 / 3.0)
    assert bucket_row["median_structures_per_s"] == pytest.approx(4.0 / 3.0)
    assert mixed_row["median_atoms_per_s"] == pytest.approx(8.0)
    assert bucket_row["median_atoms_per_s"] == pytest.approx(4.0)


def test_bucket_energy_mismatch_is_not_timed() -> None:
    mixed = _batch("mixed", [2, 4])
    small = _batch("small", [2])
    large = _batch("large", [4])
    model = _Model(
        {
            "mixed": [10.0, 20.0],
            "small": [10.0],
            "large": [21.0],
        }
    )
    with pytest.raises(AssertionError):
        benchmarking._compare_mixed_and_bucketed_with_dependencies(
            model,
            mixed,
            [small, large],
            [[0], [1]],
            measured_passes=1,
            measured_repeats=3,
            clock=lambda: pytest.fail("timing must not start after failed parity"),
            synchronize=_no_synchronize,
        )

    assert model.calls == ["mixed", "small", "large"]


def test_bucket_route_rejects_batches_on_different_cuda_devices() -> None:
    mixed = _batch("mixed", [2, 2], device=_Device("cuda", 0))
    first = _batch("first", [2], device=_Device("cuda", 0))
    second = _batch("second", [2], device=_Device("cuda", 1))
    model = _Model({"mixed": [1.0, 2.0], "first": [1.0], "second": [2.0]})

    with pytest.raises(ValueError, match="same device"):
        benchmarking._compare_mixed_and_bucketed_with_dependencies(
            model,
            mixed,
            [first, second],
            [[0], [1]],
            measured_passes=1,
            measured_repeats=3,
            clock=lambda: pytest.fail("timing must not start for invalid devices"),
            synchronize=_no_synchronize,
        )

    assert model.calls == []


def test_benchmark_helpers_do_not_choose_structures_or_reconfigure_models() -> None:
    source = (PART_DIR / "aux" / "benchmarking.py").read_text()

    assert "AtomicData.from_atoms(" not in source
    assert "Batch.from_data_list(" not in source
    assert "make_water" not in source
    assert "AIMNet" not in source
    assert "from nvalchemi" not in source
    assert "set_config(" not in source
    assert "cutoff=" not in source
