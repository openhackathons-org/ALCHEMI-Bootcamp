"""Small benchmark helpers for the Part 1 learner notebook.

This module owns repeated batch assembly, timing, synchronization, workload
checks, and output ordering. Notebook cells still choose the structures,
models, devices, dtype, neighbor settings, call counts, and comparison
tolerances. Nothing here changes cutoffs or model configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .plotting import DFT_COLOR, FIGURE_SIZE, MD_COLOR, style_axis


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


def build_benchmark_batch(
    atoms_sequence: Sequence[Any],
    *,
    device: Any,
    dtype: Any,
    atomic_data_type: Any,
    batch_type: Any,
) -> Any:
    """Build one Toolkit batch without hiding its device or numeric dtype.

    ``atomic_data_type`` and ``batch_type`` are supplied by the notebook so the
    Toolkit classes used for conversion remain explicit at the call site. This
    helper only removes the repeated list construction from learner-facing
    cells.
    """

    atoms_list = list(atoms_sequence)
    if not atoms_list:
        raise ValueError("atoms_sequence must contain at least one structure")
    data = [
        atomic_data_type.from_atoms(atoms, device=device, dtype=dtype)
        for atoms in atoms_list
    ]
    return batch_type.from_data_list(data, device=device)


def _time_route(
    model: Any,
    batches: Sequence[Any],
    *,
    warmup_passes: int,
    timed_passes: int,
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> float:
    """Time one complete route block with synchronization at its boundaries."""

    return _time_route_blocks(
        model,
        batches,
        warmup_passes=warmup_passes,
        timed_passes=timed_passes,
        measured_repeats=1,
        clock=clock,
        synchronize=synchronize,
    )[0]


def _time_route_blocks(
    model: Any,
    batches: Sequence[Any],
    *,
    warmup_passes: int,
    timed_passes: int,
    measured_repeats: int,
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> tuple[float, ...]:
    """Time repeated route blocks after one caller-sized warm-up phase."""

    if not batches:
        raise ValueError("a timed route must contain at least one batch")
    _assert_one_device(batches)
    warmup_passes = _positive_int(warmup_passes, name="warmup_passes", allow_zero=True)
    timed_passes = _positive_int(timed_passes, name="timed_passes")
    measured_repeats = _positive_int(measured_repeats, name="measured_repeats")

    synchronize(batches[0])
    if warmup_passes:
        for _ in range(warmup_passes):
            _run_route(model, batches)
        synchronize(batches[0])

    elapsed_samples_s = []
    for _ in range(measured_repeats):
        start = clock()
        for _ in range(timed_passes):
            _run_route(model, batches)
        synchronize(batches[0])
        elapsed_s = clock() - start
        if elapsed_s <= 0.0:
            raise RuntimeError("clock returned a non-positive elapsed time")
        elapsed_samples_s.append(elapsed_s)
    return tuple(elapsed_samples_s)


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
    warmup_passes = _positive_int(warmup_passes, name="warmup_passes", allow_zero=True)
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


def _repeated_timing_row(
    *,
    route: str,
    phase: str,
    batch: Any,
    calls_per_pass: int,
    timed_passes: int,
    warmup_passes: int,
    elapsed_samples_s: Sequence[float],
    validation_calls: int = 0,
) -> dict[str, Any]:
    """Summarize repeated measured blocks without discarding raw samples."""

    samples = np.asarray(elapsed_samples_s, dtype=float)
    if (
        samples.ndim != 1
        or samples.size == 0
        or not np.isfinite(samples).all()
        or np.any(samples <= 0.0)
    ):
        raise ValueError("elapsed_samples_s must contain positive finite measurements")
    timed_passes = _positive_int(timed_passes, name="timed_passes")
    measured_repeats = int(samples.size)
    elapsed_q1_s, elapsed_median_s, elapsed_q3_s = (
        float(value)
        for value in np.quantile(
            samples,
            (0.25, 0.5, 0.75),
            method="linear",
        )
    )
    row = _timing_row(
        route=route,
        phase=phase,
        batch=batch,
        calls_per_pass=calls_per_pass,
        timed_passes=timed_passes,
        warmup_passes=warmup_passes,
        elapsed_s=elapsed_median_s,
        validation_calls=validation_calls,
    )

    graphs, atoms = _batch_shape(batch)
    structures_per_repeat = graphs * timed_passes
    atoms_per_repeat = atoms * timed_passes
    structures_per_s_samples = structures_per_repeat / samples
    atoms_per_s_samples = atoms_per_repeat / samples
    structures_q1, structures_median, structures_q3 = (
        float(value)
        for value in np.quantile(
            structures_per_s_samples,
            (0.25, 0.5, 0.75),
            method="linear",
        )
    )
    atoms_q1, atoms_median, atoms_q3 = (
        float(value)
        for value in np.quantile(
            atoms_per_s_samples,
            (0.25, 0.5, 0.75),
            method="linear",
        )
    )
    elapsed_iqr_s = elapsed_q3_s - elapsed_q1_s
    row.update(
        {
            "passes_per_repeat": timed_passes,
            "passes": timed_passes * measured_repeats,
            "measured_repeats": measured_repeats,
            "calls": row["calls"] * measured_repeats,
            "total_calls_executed": (
                row["calls"] * measured_repeats + row["warmup_calls"] + validation_calls
            ),
            "elapsed_samples_s": tuple(float(value) for value in samples),
            "elapsed_median_s": elapsed_median_s,
            "elapsed_q1_s": elapsed_q1_s,
            "elapsed_q3_s": elapsed_q3_s,
            "elapsed_iqr_s": elapsed_iqr_s,
            "relative_iqr": elapsed_iqr_s / elapsed_median_s,
            "median_structures_per_s": structures_median,
            "structures_per_s_q1": structures_q1,
            "structures_per_s_q3": structures_q3,
            "structures_per_s_iqr": structures_q3 - structures_q1,
            "median_atoms_per_s": atoms_median,
            "atoms_per_s_q1": atoms_q1,
            "atoms_per_s_q3": atoms_q3,
            "atoms_per_s_iqr": atoms_q3 - atoms_q1,
        }
    )
    # Preserve the original convenience columns, now defined as repeat medians.
    row["structures_per_s"] = structures_median
    row["atoms_per_s"] = atoms_median
    return row


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

    return _first_and_warm_call_rows_with_dependencies(
        model,
        batch,
        warm_calls=warm_calls,
        route=route,
        clock=perf_counter,
        synchronize=_synchronize,
    )


def _first_and_warm_call_rows_with_dependencies(
    model: Any,
    batch: Any,
    *,
    warm_calls: int = 20,
    route: str = "model",
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> list[dict[str, Any]]:
    """Run the first/warm timing calculation with explicit test dependencies."""

    warm_calls = _positive_int(warm_calls, name="warm_calls")
    first_elapsed_s = _time_route(
        model,
        [batch],
        warmup_passes=0,
        timed_passes=1,
        clock=clock,
        synchronize=synchronize,
    )
    warm_elapsed_s = _time_route(
        model,
        [batch],
        warmup_passes=0,
        timed_passes=warm_calls,
        clock=clock,
        synchronize=synchronize,
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
    measured_repeats: int,
) -> list[dict[str, Any]]:
    """Compare caller-built CPU/GPU routes for the same fixed workload.

    ``routes`` maps a display label to ``(model, batch)``.  Batch construction,
    transfer, neighbor construction, and model configuration therefore remain
    visible in the notebook and are not timed. Graph partition, atomic numbers,
    and positions are checked before any model call. Warm-up runs once per route,
    followed by ``measured_repeats`` separately synchronized blocks of
    ``measured_calls`` calls.
    """

    return _compare_fixed_workload_devices_with_dependencies(
        routes,
        warmup_calls=warmup_calls,
        measured_calls=measured_calls,
        measured_repeats=measured_repeats,
        clock=perf_counter,
        synchronize=_synchronize,
    )


def _compare_fixed_workload_devices_with_dependencies(
    routes: Mapping[str, tuple[Any, Any]],
    *,
    warmup_calls: int = 2,
    measured_calls: int = 20,
    measured_repeats: int,
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> list[dict[str, Any]]:
    """Run fixed-workload timing with explicit test dependencies."""

    if len(routes) < 2:
        raise ValueError("provide at least two device routes")
    warmup_calls = _positive_int(warmup_calls, name="warmup_calls", allow_zero=True)
    measured_calls = _positive_int(measured_calls, name="measured_calls")
    measured_repeats = _positive_int(measured_repeats, name="measured_repeats")

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
        elapsed_samples_s = _time_route_blocks(
            model,
            [batch],
            warmup_passes=warmup_calls,
            timed_passes=measured_calls,
            measured_repeats=measured_repeats,
            clock=clock,
            synchronize=synchronize,
        )
        rows.append(
            _repeated_timing_row(
                route=str(label),
                phase="warm calls",
                batch=batch,
                calls_per_pass=1,
                timed_passes=measured_calls,
                warmup_passes=warmup_calls,
                elapsed_samples_s=elapsed_samples_s,
            )
        )
    return rows


def benchmark_device_sweep(
    *,
    batch_sizes: Sequence[int],
    structure_factory: Callable[[], Any],
    atoms_info: Mapping[str, Any],
    routes: Mapping[str, tuple[Any, Any, Any]],
    dtype: Any,
    atomic_data_type: Any,
    batch_type: Any,
    compute_neighbors: Callable[..., Any],
    warmup_calls: int,
    measured_calls: int,
    measured_repeats: int,
    energy_key: str,
    energy_atol: float,
    energy_rtol: float,
    on_batch_complete: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Measure matched device routes over several caller-chosen batch sizes.

    ``routes`` maps each display label to ``(model, device, neighbor_config)``.
    For every requested size, the helper builds the same fresh structures on
    each device, computes model-aware neighbors, checks the selected energy
    output, and then times repeated warm model-call blocks. The returned table
    retains every elapsed sample and reports elapsed-time and throughput
    quartiles for each route.

    The notebook must pass the batch sizes, model routes, dtype, warm-up count,
    calls per measured block, measured repeat count, and energy tolerances
    explicitly. The optional callback receives ``(completed_size_count,
    batch_size)`` so a notebook progress card can be updated without placing the
    sweep loop in a learner-facing cell.
    """

    return _benchmark_device_sweep_with_dependencies(
        batch_sizes=batch_sizes,
        structure_factory=structure_factory,
        atoms_info=atoms_info,
        routes=routes,
        dtype=dtype,
        atomic_data_type=atomic_data_type,
        batch_type=batch_type,
        compute_neighbors=compute_neighbors,
        warmup_calls=warmup_calls,
        measured_calls=measured_calls,
        measured_repeats=measured_repeats,
        energy_key=energy_key,
        energy_atol=energy_atol,
        energy_rtol=energy_rtol,
        on_batch_complete=on_batch_complete,
        clock=perf_counter,
        synchronize=_synchronize,
    )


def _benchmark_device_sweep_with_dependencies(
    *,
    batch_sizes: Sequence[int],
    structure_factory: Callable[[], Any],
    atoms_info: Mapping[str, Any],
    routes: Mapping[str, tuple[Any, Any, Any]],
    dtype: Any,
    atomic_data_type: Any,
    batch_type: Any,
    compute_neighbors: Callable[..., Any],
    warmup_calls: int,
    measured_calls: int,
    measured_repeats: int,
    energy_key: str,
    energy_atol: float,
    energy_rtol: float,
    on_batch_complete: Callable[[int, int], None] | None,
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> pd.DataFrame:
    """Run the device sweep with explicit timing dependencies for CPU tests."""

    normalized_sizes = [
        _positive_int(size, name="batch_sizes entry") for size in batch_sizes
    ]
    if not normalized_sizes:
        raise ValueError("batch_sizes must contain at least one size")
    if len(routes) < 2:
        raise ValueError("provide at least two device routes")
    warmup_calls = _positive_int(warmup_calls, name="warmup_calls", allow_zero=True)
    measured_calls = _positive_int(measured_calls, name="measured_calls")
    measured_repeats = _positive_int(measured_repeats, name="measured_repeats")
    energy_atol = float(energy_atol)
    energy_rtol = float(energy_rtol)
    if energy_atol < 0.0 or energy_rtol < 0.0:
        raise ValueError("energy tolerances must be non-negative")

    route_items: list[tuple[str, Any, Any, Any]] = []
    for raw_label, route in routes.items():
        try:
            model, device, neighbor_config = route
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each route must be (model, device, neighbor_config)"
            ) from exc
        route_items.append((str(raw_label), model, device, neighbor_config))

    # The notebook supplies Toolkit's public compute_neighbors function. Keep
    # the benchmark loop generic so this helper does not own that API choice.
    neighbor_builder = compute_neighbors
    rows: list[dict[str, Any]] = []
    for completed, batch_size in enumerate(normalized_sizes, start=1):
        structures = [structure_factory() for _ in range(batch_size)]
        for atoms in structures:
            if not hasattr(atoms, "info"):
                raise AttributeError("benchmark structures must expose an info mapping")
            atoms.info.update(atoms_info)

        timed_routes: dict[str, tuple[Any, Any]] = {}
        outputs_by_route: dict[str, np.ndarray] = {}
        for label, model, device, neighbor_config in route_items:
            batch = build_benchmark_batch(
                structures,
                device=device,
                dtype=dtype,
                atomic_data_type=atomic_data_type,
                batch_type=batch_type,
            )
            neighbor_builder(batch, config=neighbor_config)
            timed_routes[label] = (model, batch)
            outputs_by_route[label] = _energy_vector(
                model(batch), energy_key=energy_key, graphs=batch_size
            )

        reference_label = route_items[0][0]
        reference_energy = outputs_by_route[reference_label]
        max_error = 0.0
        for label, energies in list(outputs_by_route.items())[1:]:
            try:
                np.testing.assert_allclose(
                    energies,
                    reference_energy,
                    atol=energy_atol,
                    rtol=energy_rtol,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"{label} energies differ from {reference_label} at "
                    f"batch size {batch_size}"
                ) from exc
            max_error = max(
                max_error,
                float(np.max(np.abs(energies - reference_energy))),
            )

        timing_rows = _compare_fixed_workload_devices_with_dependencies(
            timed_routes,
            warmup_calls=warmup_calls,
            measured_calls=measured_calls,
            measured_repeats=measured_repeats,
            clock=clock,
            synchronize=synchronize,
        )
        for row in timing_rows:
            row["validation_calls"] += 1
            row["total_calls_executed"] += 1
            row.update(
                {
                    "batch_size": batch_size,
                    "dtype": str(dtype),
                    "energy_key": str(energy_key),
                    "energy_atol": energy_atol,
                    "energy_rtol": energy_rtol,
                    "reference_route": reference_label,
                    "max_abs_energy_difference": max_error,
                }
            )
        rows.extend(timing_rows)
        if on_batch_complete is not None:
            on_batch_complete(completed, batch_size)

    return pd.DataFrame(rows)


def plot_device_sweep(results: pd.DataFrame) -> tuple[Any, Any]:
    """Plot median warm-call throughput and IQR for every device route.

    Both axes use logarithmic scales because tutorial sweeps usually span
    several batch sizes and throughput ranges. The function returns the figure
    and axis, leaving display and saving to the notebook.
    """

    required_columns = {
        "batch_size",
        "route",
        "median_structures_per_s",
        "structures_per_s_q1",
        "structures_per_s_q3",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            "device sweep results are missing " + ", ".join(sorted(missing_columns))
        )
    if results.empty:
        raise ValueError("device sweep results must contain at least one row")

    plot_data = results.loc[
        :,
        [
            "batch_size",
            "route",
            "median_structures_per_s",
            "structures_per_s_q1",
            "structures_per_s_q3",
        ],
    ].copy()
    numeric_columns = (
        "batch_size",
        "median_structures_per_s",
        "structures_per_s_q1",
        "structures_per_s_q3",
    )
    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
        values = plot_data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{column} values must be finite and positive")
    if (
        plot_data["structures_per_s_q1"] > plot_data["median_structures_per_s"]
    ).any() or (
        plot_data["median_structures_per_s"] > plot_data["structures_per_s_q3"]
    ).any():
        raise ValueError("throughput quartiles must satisfy Q1 <= median <= Q3")
    batch_sizes = plot_data["batch_size"].to_numpy(dtype=float)
    if not np.equal(batch_sizes, np.floor(batch_sizes)).all():
        raise ValueError("batch_size values must be whole numbers")
    if plot_data["route"].isna().any():
        raise ValueError("route values must not be empty")
    plot_data["route"] = plot_data["route"].astype(str).str.strip()
    if (plot_data["route"] == "").any():
        raise ValueError("route values must not be empty")
    if plot_data.duplicated(["route", "batch_size"]).any():
        raise ValueError("each route must have one row per batch size")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency message
        raise ImportError("Plotting benchmark results requires matplotlib") from exc

    colors = (MD_COLOR, DFT_COLOR, "#4D7C0F", "#7C3AED")
    markers = ("o", "s", "^", "D")
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for index, (route, route_rows) in enumerate(plot_data.groupby("route", sort=False)):
        route_rows = route_rows.sort_values("batch_size")
        median = route_rows["median_structures_per_s"].to_numpy(dtype=float)
        lower = route_rows["structures_per_s_q1"].to_numpy(dtype=float)
        upper = route_rows["structures_per_s_q3"].to_numpy(dtype=float)
        color = colors[index % len(colors)]
        axis.plot(
            route_rows["batch_size"],
            median,
            color=color,
            marker=markers[index % len(markers)],
            linewidth=1.8,
            markersize=6,
            label=route,
        )
        axis.errorbar(
            route_rows["batch_size"],
            median,
            yerr=np.vstack((median - lower, upper - median)),
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    visible_sizes = sorted(plot_data["batch_size"].astype(int).unique())
    axis.set_xticks(visible_sizes, [str(size) for size in visible_sizes])
    axis.set_xlabel("batch size (structures)")
    axis.set_ylabel("median throughput (structures/s)")
    axis.set_title("Warm model-call throughput by batch size (median and IQR)")
    style_axis(axis)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


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
    measured_repeats: int,
    energy_key: str = "energy",
    atol: float = 2.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare one mixed call with caller-provided homogeneous bucket calls.

    ``bucket_graph_indices[i]`` gives the positions of bucket ``i`` in the
    mixed batch.  The function reconstructs that original ordering and asserts
    energy agreement before timing either route. Both routes must contain
    exactly the same number of graphs and atoms; no structures are created or
    removed. Warm-up runs once per route, followed by ``measured_repeats``
    separately synchronized blocks of ``measured_passes`` passes.
    """

    return _compare_mixed_and_bucketed_with_dependencies(
        model,
        mixed_batch,
        bucket_batches,
        bucket_graph_indices,
        warmup_passes=warmup_passes,
        measured_passes=measured_passes,
        measured_repeats=measured_repeats,
        energy_key=energy_key,
        atol=atol,
        rtol=rtol,
        clock=perf_counter,
        synchronize=_synchronize,
    )


def neighbor_storage_table(
    routes: Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    """Summarize valid and allocated matrix-neighbor slots by batch route."""

    if not routes:
        raise ValueError("routes must not be empty")
    rows = []
    for route, batches in routes.items():
        if not batches:
            raise ValueError(f"route {route!r} has no batches")
        valid = sum(int(batch.num_neighbors.sum().detach().cpu()) for batch in batches)
        allocated = sum(int(batch.neighbor_matrix.numel()) for batch in batches)
        if allocated <= 0:
            raise ValueError(f"route {route!r} has no allocated neighbor slots")
        rows.append(
            {
                "route": route,
                "valid_neighbor_slots": valid,
                "allocated_neighbor_slots": allocated,
                "neighbor_slot_utilization": valid / allocated,
            }
        )
    return pd.DataFrame(rows)


def _compare_mixed_and_bucketed_with_dependencies(
    model: Any,
    mixed_batch: Any,
    bucket_batches: Sequence[Any],
    bucket_graph_indices: Sequence[Sequence[int]],
    *,
    warmup_passes: int = 2,
    measured_passes: int = 20,
    measured_repeats: int,
    energy_key: str = "energy",
    atol: float = 2.0e-6,
    rtol: float = 1.0e-5,
    clock: Callable[[], float],
    synchronize: Callable[[Any], None],
) -> dict[str, Any]:
    """Run layout comparison with explicit test dependencies."""

    if not bucket_batches:
        raise ValueError("provide at least one homogeneous bucket batch")
    warmup_passes = _positive_int(warmup_passes, name="warmup_passes", allow_zero=True)
    measured_passes = _positive_int(measured_passes, name="measured_passes")
    measured_repeats = _positive_int(measured_repeats, name="measured_repeats")
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

    synchronize(mixed_batch)
    mixed_outputs = model(mixed_batch)
    bucket_outputs = [model(batch) for batch in bucket_batches]
    synchronize(mixed_batch)
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

    mixed_elapsed_samples_s = _time_route_blocks(
        model,
        [mixed_batch],
        warmup_passes=warmup_passes,
        timed_passes=measured_passes,
        measured_repeats=measured_repeats,
        clock=clock,
        synchronize=synchronize,
    )
    bucket_elapsed_samples_s = _time_route_blocks(
        model,
        list(bucket_batches),
        warmup_passes=warmup_passes,
        timed_passes=measured_passes,
        measured_repeats=measured_repeats,
        clock=clock,
        synchronize=synchronize,
    )
    bucket_calls = len(bucket_batches)
    rows = [
        _repeated_timing_row(
            route="one heterogeneous batch",
            phase="warm calls",
            batch=mixed_batch,
            calls_per_pass=1,
            timed_passes=measured_passes,
            warmup_passes=warmup_passes,
            elapsed_samples_s=mixed_elapsed_samples_s,
            validation_calls=1,
        ),
        _repeated_timing_row(
            route="homogeneous buckets",
            phase="warm calls",
            batch=mixed_batch,
            calls_per_pass=bucket_calls,
            timed_passes=measured_passes,
            warmup_passes=warmup_passes,
            elapsed_samples_s=bucket_elapsed_samples_s,
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
    "benchmark_device_sweep",
    "build_benchmark_batch",
    "compare_fixed_workload_devices",
    "compare_mixed_and_bucketed",
    "first_and_warm_call_rows",
    "neighbor_storage_table",
    "plot_device_sweep",
]
