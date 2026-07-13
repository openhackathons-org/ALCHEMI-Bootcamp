"""Focused CPU tests for transparent Part 1 benchmarking mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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


def _clock(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    times = iter(values)
    monkeypatch.setattr(benchmarking, "perf_counter", lambda: next(times))


def test_first_and_warm_calls_report_aggregate_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch("mixed", [3, 3])
    model = _Model({"mixed": [1.0, 2.0]})
    _clock(monkeypatch, [0.0, 2.0, 10.0, 14.0])

    first, warm = benchmarking.first_and_warm_call_rows(
        model,
        batch,
        warm_calls=4,
        route="AIMNet2 pipeline",
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


def test_cuda_calls_are_synchronized_around_each_measured_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _Device("cuda", 1)
    batch = _batch("gpu", [3], device=device)
    model = _Model({"gpu": [1.0]})
    synchronized: list[object] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda selected: synchronized.append(selected))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _clock(monkeypatch, [0.0, 1.0, 5.0, 7.0])

    benchmarking.first_and_warm_call_rows(model, batch, warm_calls=2)

    assert synchronized == [device, device, device, device]


def test_fixed_workload_device_comparison_times_caller_built_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_batch = _batch("cpu", [3, 6], device=_Device("cpu"))
    gpu_batch = _batch("gpu", [3, 6], device=_Device("cuda"))
    cpu_model = _Model({"cpu": [1.0, 2.0]})
    gpu_model = _Model({"gpu": [1.0, 2.0]})
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _clock(monkeypatch, [0.0, 3.0, 10.0, 11.0])

    rows = benchmarking.compare_fixed_workload_devices(
        {"CPU": (cpu_model, cpu_batch), "GPU": (gpu_model, gpu_batch)},
        warmup_calls=2,
        measured_calls=3,
    )

    assert cpu_model.calls == ["cpu"] * 5
    assert gpu_model.calls == ["gpu"] * 5
    assert [row["calls"] for row in rows] == [3, 3]
    assert [row["warmup_calls"] for row in rows] == [2, 2]
    assert [row["total_calls_executed"] for row in rows] == [5, 5]
    assert all(row["graphs"] == 2 and row["atoms"] == 9 for row in rows)
    assert rows[0]["structures_per_s"] == pytest.approx(2.0)
    assert rows[1]["structures_per_s"] == pytest.approx(6.0)


def test_fixed_workload_mismatch_fails_before_model_calls() -> None:
    reference = _batch("cpu", [3, 3])
    changed = _batch("gpu", [3, 3], position_offset=0.25)
    first = _Model({"cpu": [1.0, 2.0]})
    second = _Model({"gpu": [1.0, 2.0]})

    with pytest.raises(ValueError, match="positions differ"):
        benchmarking.compare_fixed_workload_devices(
            {"CPU": (first, reference), "GPU": (second, changed)},
            measured_calls=1,
        )

    assert first.calls == []
    assert second.calls == []


def test_mixed_and_bucketed_comparison_restores_energy_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    _clock(monkeypatch, [0.0, 2.0, 10.0, 14.0])

    result = benchmarking.compare_mixed_and_bucketed(
        model,
        mixed,
        [small, large],
        [[0, 2], [1, 3]],
        warmup_passes=1,
        measured_passes=2,
    )

    np.testing.assert_array_equal(
        result["bucketed_energies_in_mixed_order"],
        [10.0, 20.0, 30.0, 40.0],
    )
    assert result["max_abs_energy_difference"] == 0.0
    mixed_row, bucket_row = result["timings"]
    assert mixed_row["calls_per_pass"] == 1
    assert mixed_row["calls"] == 2
    assert bucket_row["calls_per_pass"] == 2
    assert bucket_row["calls"] == 4
    assert mixed_row["graphs"] == bucket_row["graphs"] == 4
    assert mixed_row["atoms"] == bucket_row["atoms"] == 12
    assert mixed_row["structures_per_s"] == pytest.approx(4.0)
    assert bucket_row["structures_per_s"] == pytest.approx(2.0)


def test_bucket_energy_mismatch_is_not_timed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        benchmarking,
        "perf_counter",
        lambda: pytest.fail("timing must not start after failed parity"),
    )

    with pytest.raises(AssertionError):
        benchmarking.compare_mixed_and_bucketed(
            model,
            mixed,
            [small, large],
            [[0], [1]],
            measured_passes=1,
        )

    assert model.calls == ["mixed", "small", "large"]


def test_bucket_route_rejects_batches_on_different_cuda_devices() -> None:
    mixed = _batch("mixed", [2, 2], device=_Device("cuda", 0))
    first = _batch("first", [2], device=_Device("cuda", 0))
    second = _batch("second", [2], device=_Device("cuda", 1))
    model = _Model(
        {"mixed": [1.0, 2.0], "first": [1.0], "second": [2.0]}
    )

    with pytest.raises(ValueError, match="same device"):
        benchmarking.compare_mixed_and_bucketed(
            model,
            mixed,
            [first, second],
            [[0], [1]],
            measured_passes=1,
        )

    assert model.calls == []


def test_benchmark_helpers_do_not_construct_or_reconfigure_workloads() -> None:
    source = (PART_DIR / "aux" / "benchmarking.py").read_text()

    assert "AtomicData.from_atoms(" not in source
    assert "Batch.from_data_list(" not in source
    assert "compute_neighbors(" not in source
    assert "set_config(" not in source
    assert "cutoff=" not in source
