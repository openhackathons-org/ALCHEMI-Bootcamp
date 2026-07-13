"""Transparent model-call timing for already-constructed Toolkit batches.

This module deliberately owns only timing, synchronization, workload checks,
and output ordering.  Notebook cells remain responsible for constructing each
batch, computing its neighbors, configuring the model, and choosing the number
of calls.  Nothing here changes structures, cutoffs, or model configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np


def _positive_int(value: int, *, name: str, allow_zero: bool = False) -> int:
    value = int(value)
    lower = 0 if allow_zero else 1
    if value < lower:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _integer(value: Any, *, name: str) -> int:
    """Convert a Python or scalar tensor-like value to ``int``."""

    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc


def _to_numpy(value: Any) -> np.ndarray:
    """Move a tensor-like value to host memory without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _batch_shape(batch: Any) -> tuple[int, int]:
    """Return graph and atom counts from a Toolkit-compatible batch."""

    try:
        graphs = _integer(batch.num_graphs, name="batch.num_graphs")
        atoms = _integer(batch.num_nodes, name="batch.num_nodes")
    except AttributeError as exc:
        raise AttributeError(
            "benchmark batches must expose num_graphs and num_nodes"
        ) from exc
    if graphs <= 0 or atoms <= 0:
        raise ValueError("benchmark batches must contain graphs and atoms")
    return graphs, atoms


def _batch_device(batch: Any) -> Any:
    device = getattr(batch, "device", None)
    if device is None and hasattr(batch, "positions"):
        device = getattr(batch.positions, "device", None)
    if device is None:
        return "cpu"
    return device


def _device_type(device: Any) -> str:
    kind = getattr(device, "type", None)
    if kind is None:
        kind = str(device).split(":", 1)[0]
    return str(kind).lower()


def _synchronize(batch: Any) -> None:
    """Synchronize the batch's CUDA device; CPU calls are synchronous."""

    device = _batch_device(batch)
    if _device_type(device) != "cuda":
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - impossible in a real CUDA run
        raise RuntimeError("CUDA timing requires PyTorch") from exc
    torch.cuda.synchronize(device)


def _assert_one_device(batches: Sequence[Any]) -> None:
    devices = {str(_batch_device(batch)) for batch in batches}
    if len(devices) != 1:
        raise ValueError("all batches in one timed route must use the same device")


def _run_route(model: Any, batches: Sequence[Any]) -> list[Any]:
    return [model(batch) for batch in batches]


def _time_route(
    model: Any,
    batches: Sequence[Any],
    *,
    warmup_passes: int,
    timed_passes: int,
) -> float:
    """Time complete route passes with synchronization outside the boundary."""

    if not batches:
        raise ValueError("a timed route must contain at least one batch")
    _assert_one_device(batches)
    warmup_passes = _positive_int(
        warmup_passes, name="warmup_passes", allow_zero=True
    )
    timed_passes = _positive_int(timed_passes, name="timed_passes")

    _synchronize(batches[0])
    if warmup_passes:
        for _ in range(warmup_passes):
            _run_route(model, batches)
        _synchronize(batches[0])

    start = perf_counter()
    for _ in range(timed_passes):
        _run_route(model, batches)
    _synchronize(batches[0])
    elapsed_s = perf_counter() - start
    if elapsed_s <= 0.0:
        raise RuntimeError("non-positive elapsed time from perf_counter")
    return elapsed_s


def _timing_row(
    *,
    route: str,
    phase: str,
    batch: Any,
    calls_per_pass: int,
    timed_passes: int,
    warmup_passes: int,
    elapsed_s: float,
    validation_calls: int = 0,
) -> dict[str, Any]:
    graphs, atoms = _batch_shape(batch)
    calls_per_pass = _positive_int(calls_per_pass, name="calls_per_pass")
    timed_passes = _positive_int(timed_passes, name="timed_passes")
    warmup_passes = _positive_int(
        warmup_passes, name="warmup_passes", allow_zero=True
    )
    validation_calls = _positive_int(
        validation_calls, name="validation_calls", allow_zero=True
    )
    elapsed_s = float(elapsed_s)
    if elapsed_s <= 0.0:
        raise ValueError("elapsed_s must be positive")

    calls = calls_per_pass * timed_passes
    warmup_calls = calls_per_pass * warmup_passes
    structures = graphs * timed_passes
    atom_evaluations = atoms * timed_passes
    return {
        "route": str(route),
        "phase": str(phase),
        "device": str(_batch_device(batch)),
        "passes": timed_passes,
        "calls_per_pass": calls_per_pass,
        "calls": calls,
        "warmup_calls": warmup_calls,
        "validation_calls": validation_calls,
        "total_calls_executed": calls + warmup_calls + validation_calls,
        "graphs": graphs,
        "atoms": atoms,
        "elapsed_s": elapsed_s,
        "wall_ms_per_pass": 1.0e3 * elapsed_s / timed_passes,
        "structures_per_s": structures / elapsed_s,
        "atoms_per_s": atom_evaluations / elapsed_s,
    }


def first_and_warm_call_rows(
    model: Any,
    batch: Any,
    *,
    warm_calls: int = 20,
    route: str = "model",
) -> list[dict[str, Any]]:
    """Measure one first call, then a caller-sized block of warm calls.

    The first call is not reused as a hidden warm-up.  It is reported as its
    own synchronized measurement.  The following row reports the aggregate
    time for ``warm_calls`` calls, so synchronization overhead is paid once for
    the measured block rather than once per call.
    """

    warm_calls = _positive_int(warm_calls, name="warm_calls")
    first_elapsed_s = _time_route(
        model,
        [batch],
        warmup_passes=0,
        timed_passes=1,
    )
    warm_elapsed_s = _time_route(
        model,
        [batch],
        warmup_passes=0,
        timed_passes=warm_calls,
    )
    return [
        _timing_row(
            route=route,
            phase="first call",
            batch=batch,
            calls_per_pass=1,
            timed_passes=1,
            warmup_passes=0,
            elapsed_s=first_elapsed_s,
        ),
        _timing_row(
            route=route,
            phase="warm calls",
            batch=batch,
            calls_per_pass=1,
            timed_passes=warm_calls,
            warmup_passes=0,
            elapsed_s=warm_elapsed_s,
        ),
    ]


def _require_workload_field(batch: Any, field: str, *, label: str) -> np.ndarray:
    if not hasattr(batch, field):
        raise AttributeError(
            f"{label} batch must expose {field!r} for fixed-workload validation"
        )
    return _to_numpy(getattr(batch, field))


def _assert_same_fixed_workload(
    reference: Any,
    candidate: Any,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    reference_shape = _batch_shape(reference)
    candidate_shape = _batch_shape(candidate)
    if candidate_shape != reference_shape:
        raise ValueError(
            f"{candidate_label} workload has {candidate_shape}, expected "
            f"{reference_shape} from {reference_label}"
        )

    for field in ("batch_ptr", "atomic_numbers"):
        expected = _require_workload_field(reference, field, label=reference_label)
        actual = _require_workload_field(candidate, field, label=candidate_label)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{candidate_label} {field} differs from {reference_label}"
            )

    expected_positions = _require_workload_field(
        reference, "positions", label=reference_label
    )
    actual_positions = _require_workload_field(
        candidate, "positions", label=candidate_label
    )
    try:
        np.testing.assert_allclose(
            actual_positions,
            expected_positions,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    except AssertionError as exc:
        raise ValueError(
            f"{candidate_label} positions differ from {reference_label}"
        ) from exc


def compare_fixed_workload_devices(
    routes: Mapping[str, tuple[Any, Any]],
    *,
    warmup_calls: int = 2,
    measured_calls: int = 20,
) -> list[dict[str, Any]]:
    """Compare caller-built CPU/GPU routes for the same fixed workload.

    ``routes`` maps a display label to ``(model, batch)``.  Batch construction,
    transfer, neighbor construction, and model configuration therefore remain
    visible in the notebook and outside the timing boundary.  Graph partition,
    atomic numbers, and positions are checked before any model call.
    """

    if len(routes) < 2:
        raise ValueError("provide at least two device routes")
    warmup_calls = _positive_int(
        warmup_calls, name="warmup_calls", allow_zero=True
    )
    measured_calls = _positive_int(measured_calls, name="measured_calls")

    route_items = list(routes.items())
    reference_label, (_, reference_batch) = route_items[0]
    for label, (_, batch) in route_items[1:]:
        _assert_same_fixed_workload(
            reference_batch,
            batch,
            reference_label=str(reference_label),
            candidate_label=str(label),
        )

    rows = []
    for label, (model, batch) in route_items:
        elapsed_s = _time_route(
            model,
            [batch],
            warmup_passes=warmup_calls,
            timed_passes=measured_calls,
        )
        rows.append(
            _timing_row(
                route=str(label),
                phase="warm calls",
                batch=batch,
                calls_per_pass=1,
                timed_passes=measured_calls,
                warmup_passes=warmup_calls,
                elapsed_s=elapsed_s,
            )
        )
    return rows


def _energy_vector(outputs: Any, *, energy_key: str, graphs: int) -> np.ndarray:
    if isinstance(outputs, Mapping):
        if energy_key not in outputs:
            raise KeyError(f"model outputs do not contain {energy_key!r}")
        value = outputs[energy_key]
    elif hasattr(outputs, energy_key):
        value = getattr(outputs, energy_key)
    else:
        raise KeyError(f"model outputs do not contain {energy_key!r}")
    energies = _to_numpy(value).reshape(-1)
    if energies.size != graphs:
        raise ValueError(
            f"{energy_key!r} contains {energies.size} values for {graphs} graphs"
        )
    return energies.astype(np.float64, copy=False)


def _validate_bucket_indices(
    bucket_batches: Sequence[Any],
    bucket_graph_indices: Sequence[Sequence[int]],
    *,
    mixed_graphs: int,
) -> list[np.ndarray]:
    if len(bucket_batches) != len(bucket_graph_indices):
        raise ValueError("provide one graph-index sequence for every bucket")
    normalized = []
    for bucket_number, (batch, raw_indices) in enumerate(
        zip(bucket_batches, bucket_graph_indices, strict=True)
    ):
        bucket_graphs, _ = _batch_shape(batch)
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or indices.size != bucket_graphs:
            raise ValueError(
                f"bucket {bucket_number} needs {bucket_graphs} graph indices"
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("bucket graph indices must be integers")
        normalized.append(indices.astype(np.int64, copy=False))

    if normalized:
        combined = np.concatenate(normalized)
    else:
        combined = np.empty(0, dtype=np.int64)
    if not np.array_equal(np.sort(combined), np.arange(mixed_graphs)):
        raise ValueError(
            "bucket graph indices must cover each mixed-batch graph exactly once"
        )
    return normalized


def compare_mixed_and_bucketed(
    model: Any,
    mixed_batch: Any,
    bucket_batches: Sequence[Any],
    bucket_graph_indices: Sequence[Sequence[int]],
    *,
    warmup_passes: int = 2,
    measured_passes: int = 20,
    energy_key: str = "energy",
    atol: float = 2.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare one mixed call with caller-provided homogeneous bucket calls.

    ``bucket_graph_indices[i]`` gives the positions of bucket ``i`` in the
    mixed batch.  The function reconstructs that original ordering and asserts
    energy parity before timing either route.  Both routes must contain exactly
    the same number of graphs and atoms; no structures are created or removed.
    """

    if not bucket_batches:
        raise ValueError("provide at least one homogeneous bucket batch")
    warmup_passes = _positive_int(
        warmup_passes, name="warmup_passes", allow_zero=True
    )
    measured_passes = _positive_int(measured_passes, name="measured_passes")
    mixed_graphs, mixed_atoms = _batch_shape(mixed_batch)
    bucket_graphs_atoms = [_batch_shape(batch) for batch in bucket_batches]
    if sum(graphs for graphs, _ in bucket_graphs_atoms) != mixed_graphs:
        raise ValueError("bucket routes do not contain the mixed-batch graph count")
    if sum(atoms for _, atoms in bucket_graphs_atoms) != mixed_atoms:
        raise ValueError("bucket routes do not contain the mixed-batch atom count")
    _assert_one_device([mixed_batch, *bucket_batches])
    indices = _validate_bucket_indices(
        bucket_batches,
        bucket_graph_indices,
        mixed_graphs=mixed_graphs,
    )

    _synchronize(mixed_batch)
    mixed_outputs = model(mixed_batch)
    bucket_outputs = [model(batch) for batch in bucket_batches]
    _synchronize(mixed_batch)
    mixed_energies = _energy_vector(
        mixed_outputs,
        energy_key=energy_key,
        graphs=mixed_graphs,
    )
    ordered_bucket_energies = np.empty(mixed_graphs, dtype=np.float64)
    for output, batch, graph_indices in zip(
        bucket_outputs, bucket_batches, indices, strict=True
    ):
        graphs, _ = _batch_shape(batch)
        ordered_bucket_energies[graph_indices] = _energy_vector(
            output,
            energy_key=energy_key,
            graphs=graphs,
        )
    np.testing.assert_allclose(
        ordered_bucket_energies,
        mixed_energies,
        atol=float(atol),
        rtol=float(rtol),
    )

    mixed_elapsed_s = _time_route(
        model,
        [mixed_batch],
        warmup_passes=warmup_passes,
        timed_passes=measured_passes,
    )
    bucket_elapsed_s = _time_route(
        model,
        list(bucket_batches),
        warmup_passes=warmup_passes,
        timed_passes=measured_passes,
    )
    bucket_calls = len(bucket_batches)
    rows = [
        _timing_row(
            route="one heterogeneous batch",
            phase="warm calls",
            batch=mixed_batch,
            calls_per_pass=1,
            timed_passes=measured_passes,
            warmup_passes=warmup_passes,
            elapsed_s=mixed_elapsed_s,
            validation_calls=1,
        ),
        _timing_row(
            route="homogeneous buckets",
            phase="warm calls",
            batch=mixed_batch,
            calls_per_pass=bucket_calls,
            timed_passes=measured_passes,
            warmup_passes=warmup_passes,
            elapsed_s=bucket_elapsed_s,
            validation_calls=bucket_calls,
        ),
    ]
    return {
        "timings": rows,
        "mixed_energies": mixed_energies,
        "bucketed_energies_in_mixed_order": ordered_bucket_energies,
        "max_abs_energy_difference": float(
            np.max(np.abs(mixed_energies - ordered_bucket_energies))
        ),
    }


__all__ = [
    "compare_fixed_workload_devices",
    "compare_mixed_and_bucketed",
    "first_and_warm_call_rows",
]
