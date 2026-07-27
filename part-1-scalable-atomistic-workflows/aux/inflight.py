"""Prepare the repeated water-dimer source used by the inflight lesson."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any
import weakref

from ase import Atoms
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage, initialize_velocities


@dataclass(frozen=True)
class InflightTraceEvent:
    """One observed change in the active inflight batch."""

    refill: int
    fused_step: int
    active: int
    completed: int
    entered_system_ids: tuple[int, ...]
    leaving_system_ids: tuple[int, ...]
    failures: int | None


def _integer_ids(values: Any, *, name: str) -> tuple[int, ...]:
    """Copy one-dimensional integer IDs to a validated Python tuple."""

    if isinstance(values, torch.Tensor):
        if torch.is_floating_point(values) or values.dtype == torch.bool:
            raise ValueError(f"{name} must use an integer dtype")
        raw_values = values.detach().reshape(-1).cpu().tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_values = list(values)
    else:
        raise TypeError(f"{name} must be an integer tensor or sequence")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_values
    ):
        raise ValueError(f"{name} must contain only integers")
    system_ids = tuple(int(value) for value in raw_values)
    if len(set(system_ids)) != len(system_ids):
        raise ValueError(f"{name} values must be unique")
    return system_ids


def _system_ids(batch: Batch) -> tuple[int, ...]:
    """Read stable per-system identifiers without modifying the batch."""

    values = getattr(batch, "system_id", None)
    system_ids = _integer_ids(values, name="batch.system_id")
    if len(system_ids) != batch.num_graphs:
        raise ValueError("batch.system_id must contain one value per system")
    return system_ids


def _short_id_ranges(values: Sequence[int], *, max_ranges: int = 4) -> str:
    """Compress IDs into bounded consecutive ranges for notebook cells."""

    if not values:
        return "none"
    ordered = sorted(set(int(value) for value in values))
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))

    def label(bounds: tuple[int, int]) -> str:
        first, last = bounds
        return str(first) if first == last else f"{first}-{last}"

    if len(ranges) <= max_ranges:
        return ", ".join(label(bounds) for bounds in ranges)
    kept = [label(bounds) for bounds in ranges[:2]]
    kept.extend(("...", label(ranges[-1])))
    return f"{', '.join(kept)} ({len(ordered)} IDs)"


class InflightTraceCollector:
    """Collect compact membership changes from a fused ``BEFORE_STEP`` hook.

    The hook runs only at the stage's refill frequency. It copies stable
    ``system_id`` values to the CPU for reporting and never writes to the
    dynamics batch.
    """

    stage = DynamicsStage.BEFORE_STEP

    def __init__(self, *, frequency: int) -> None:
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError("frequency must be an integer")
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.frequency = frequency
        self._snapshots: list[tuple[int, tuple[int, ...]]] = []
        self._retired_ids: set[int] = set()
        self._inflight_ref: weakref.ReferenceType[Any] | None = None
        self._finalized: tuple[int, tuple[int, ...], int | None] | None = None

    def record(self, batch: Batch, *, fused_step: int) -> None:
        """Record an actual active-batch membership snapshot."""

        if self._finalized is not None:
            raise RuntimeError("cannot record inflight events after finalize()")
        if isinstance(fused_step, bool) or not isinstance(fused_step, int):
            raise TypeError("fused_step must be an integer")
        if fused_step < 0:
            raise ValueError("fused_step must be non-negative")
        system_ids = _system_ids(batch)
        if self._snapshots:
            previous_ids = self._snapshots[-1][1]
            if set(system_ids) == set(previous_ids):
                return
            leaving_ids = set(previous_ids).difference(system_ids)
            entered_ids = set(system_ids).difference(previous_ids)
            reused_ids = entered_ids.intersection(self._retired_ids)
            if reused_ids:
                reused = ", ".join(str(value) for value in sorted(reused_ids))
                raise ValueError(
                    f"system_id values cannot re-enter after leaving: {reused}"
                )
            self._retired_ids.update(leaving_ids)
        self._snapshots.append((fused_step, system_ids))

    def __call__(self, context: Any, stage: DynamicsStage) -> None:
        """Consume the public fused-hook context."""

        if stage is not DynamicsStage.BEFORE_STEP:
            return
        batch = getattr(context, "batch", None)
        if batch is None:
            return
        self.record(batch, fused_step=int(context.step_count))

    def finalize(
        self,
        *,
        completed_system_ids: Sequence[int] | torch.Tensor,
        fused_step: int | None = None,
        failure_count: int | None = None,
    ) -> None:
        """Close the trace after draining the sink and add ``active = 0``.

        ``completed_system_ids`` must exactly match every stable ID observed by
        the hook. This prevents a partial sink drain or a late hook registration
        from being displayed as a complete inflight run.
        """

        completed_ids = _integer_ids(
            completed_system_ids,
            name="completed_system_ids",
        )
        if not self._snapshots:
            raise RuntimeError("cannot finalize an empty inflight trace")
        observed_ids = {
            system_id for _, system_ids in self._snapshots for system_id in system_ids
        }
        completed_set = set(completed_ids)
        missing = observed_ids.difference(completed_set)
        unknown = completed_set.difference(observed_ids)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing {_short_id_ranges(tuple(missing))}")
            if unknown:
                details.append(f"unobserved {_short_id_ranges(tuple(unknown))}")
            raise ValueError(
                "completed_system_ids do not match the trace: " + "; ".join(details)
            )

        inflight = self._inflight_ref() if self._inflight_ref is not None else None
        if fused_step is None:
            fused_step = getattr(inflight, "step_count", None)
        if isinstance(fused_step, bool) or not isinstance(fused_step, int):
            raise TypeError(
                "fused_step must be supplied when no registered stage is available"
            )
        if fused_step < self._snapshots[-1][0]:
            raise ValueError("fused_step cannot precede the last trace event")

        if failure_count is None and inflight is not None:
            for field in ("failure_count", "failed_count", "num_failures"):
                value = getattr(inflight, field, None)
                if isinstance(value, Integral) and not isinstance(value, bool):
                    failure_count = int(value)
                    break
        if failure_count is not None:
            if isinstance(failure_count, bool) or not isinstance(
                failure_count, Integral
            ):
                raise TypeError("failure_count must be an integer or None")
            failure_count = int(failure_count)
            if not 0 <= failure_count <= len(completed_ids):
                raise ValueError(
                    "failure_count must be between zero and the completed count"
                )

        final_state = (fused_step, completed_ids, failure_count)
        if self._finalized is not None and self._finalized != final_state:
            raise RuntimeError("inflight trace was already finalized differently")
        self._finalized = final_state

    def events(self) -> tuple[InflightTraceEvent, ...]:
        """Return immutable events derived from observed runtime snapshots."""

        snapshots = list(self._snapshots)
        terminal_failure_count: int | None = None
        if self._finalized is not None:
            fused_step, _completed_ids, terminal_failure_count = self._finalized
            snapshots.append((fused_step, ()))

        events: list[InflightTraceEvent] = []
        previous_ids: tuple[int, ...] = ()
        completed_ids: set[int] = set()
        for event_index, (fused_step, system_ids) in enumerate(snapshots):
            current = set(system_ids)
            previous = set(previous_ids)
            entered = tuple(value for value in system_ids if value not in previous)
            leaving = tuple(value for value in previous_ids if value not in current)
            completed_ids.update(leaving)
            events.append(
                InflightTraceEvent(
                    refill=len(events),
                    fused_step=fused_step,
                    active=len(system_ids),
                    completed=len(completed_ids),
                    entered_system_ids=entered,
                    leaving_system_ids=leaving,
                    failures=(
                        terminal_failure_count
                        if self._finalized is not None
                        and event_index == len(snapshots) - 1
                        else None
                    ),
                )
            )
            previous_ids = system_ids
        if self._finalized is not None:
            expected_completed = len(self._finalized[1])
            if not events or events[-1].completed != expected_completed:
                raise RuntimeError(
                    "finalized trace completed count does not match drained IDs"
                )
        return tuple(events)

    def table(self) -> Any:
        """Return :func:`inflight_trace_table` for this collector."""

        return inflight_trace_table(self)


def inflight_trace_table(
    trace_or_events: InflightTraceCollector | Sequence[InflightTraceEvent],
) -> Any:
    """Build a compact learner-facing DataFrame from real trace events."""

    import pandas as pd

    if isinstance(trace_or_events, InflightTraceCollector):
        events = trace_or_events.events()
    else:
        events = tuple(trace_or_events)
        if any(not isinstance(event, InflightTraceEvent) for event in events):
            raise TypeError("trace_or_events must contain InflightTraceEvent values")

    rows = [
        {
            "Refill": event.refill,
            "Fused step": event.fused_step,
            "Active": event.active,
            "Completed": event.completed,
            "Entered": len(event.entered_system_ids),
            "Entered IDs": _short_id_ranges(event.entered_system_ids),
            "Leaving": len(event.leaving_system_ids),
            "Leaving IDs": _short_id_ranges(event.leaving_system_ids),
            "Failures": ("NOT REPORTED" if event.failures is None else event.failures),
        }
        for event in events
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "Refill",
            "Fused step",
            "Active",
            "Completed",
            "Entered",
            "Entered IDs",
            "Leaving",
            "Leaving IDs",
            "Failures",
        ),
    )


def register_inflight_trace(inflight: Any) -> InflightTraceCollector:
    """Register and return a compact, read-only inflight trace collector."""

    register = getattr(inflight, "register_fused_hook", None)
    if not callable(register):
        raise TypeError("inflight must provide register_fused_hook(hook)")
    frequency = getattr(inflight, "refill_frequency", None)
    collector = InflightTraceCollector(frequency=frequency)
    collector._inflight_ref = weakref.ref(inflight)
    register(collector)
    return collector


def _checked_output(
    outputs: Mapping[str, torch.Tensor],
    name: str,
    *,
    first_dimension: int,
) -> torch.Tensor:
    """Return one finite output tensor with the expected leading dimension."""

    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"full_outputs must contain tensor output {name!r}")
    if value.ndim == 0 or value.shape[0] != first_dimension:
        raise ValueError(
            f"full_outputs[{name!r}] must have leading dimension "
            f"{first_dimension}, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"full_outputs[{name!r}] contains NaN or infinity")
    return value


def prepare_inflight_dimer_source(
    *,
    scan_dimers: Sequence[Atoms],
    scan_batch: Batch,
    full_outputs: Mapping[str, torch.Tensor],
    num_systems: int,
    temperature_k: float,
    velocity_seed: int,
) -> Batch:
    """Build the CPU source batch consumed by ``InMemoryDataset``.

    ``scan_batch`` must use the tutorial's repeated ``AB, A, B`` ordering. The
    full-model values for each ``AB`` graph seed the corresponding replacement
    template. The pinned inflight path can inspect these fields before its first
    fresh model result, so energy, forces, and charges are intentionally copied
    rather than initialized with placeholders.
    """

    if isinstance(num_systems, bool) or not isinstance(num_systems, int):
        raise TypeError("num_systems must be an integer")
    if num_systems <= 0:
        raise ValueError("num_systems must be positive")
    temperature_k = float(temperature_k)
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and positive")

    dimers = tuple(scan_dimers)
    if not dimers:
        raise ValueError("scan_dimers must contain at least one dimer")
    atom_counts = [len(atoms) for atoms in dimers]
    if min(atom_counts) <= 0 or len(set(atom_counts)) != 1:
        raise ValueError("scan_dimers must be homogeneous nonempty structures")
    if scan_batch.num_graphs != 3 * len(dimers):
        raise ValueError("scan_batch must contain one AB, A, B triplet per dimer")

    energy = _checked_output(
        full_outputs, "energy", first_dimension=scan_batch.num_graphs
    )
    forces = _checked_output(
        full_outputs, "forces", first_dimension=scan_batch.num_nodes
    )
    charges = _checked_output(
        full_outputs, "charges", first_dimension=scan_batch.num_nodes
    )
    if charges.ndim not in (1, 2) or (charges.ndim == 2 and charges.shape[1] != 1):
        raise ValueError("charges must have shape [n_atoms] or [n_atoms, 1]")
    charges = charges.reshape(-1)
    scan_ptr = scan_batch.batch_ptr.detach().cpu().tolist()

    templates: list[AtomicData] = []
    for dimer_index, atoms in enumerate(dimers):
        graph_index = 3 * dimer_index
        ab_start, ab_stop = scan_ptr[graph_index : graph_index + 2]
        a_start, a_stop = scan_ptr[graph_index + 1 : graph_index + 3]
        b_start, b_stop = scan_ptr[graph_index + 2 : graph_index + 4]
        if ab_stop - ab_start != len(atoms):
            raise ValueError("AB graph atom count does not match its scan dimer")
        if (a_stop - a_start) + (b_stop - b_start) != len(atoms):
            raise ValueError("A and B graph atom counts do not reconstruct AB")

        atoms_copy = atoms.copy()
        atoms_copy.info["charge"] = 0
        data = AtomicData.from_atoms(atoms_copy, device="cpu", dtype=torch.float32)
        ab_numbers = scan_batch.atomic_numbers[ab_start:ab_stop].detach().cpu()
        monomer_numbers = torch.cat(
            (
                scan_batch.atomic_numbers[a_start:a_stop].detach().cpu(),
                scan_batch.atomic_numbers[b_start:b_stop].detach().cpu(),
            )
        )
        if not torch.equal(ab_numbers, data.atomic_numbers.cpu()):
            raise ValueError("AB graph elements do not match its scan dimer")
        if not torch.equal(monomer_numbers, ab_numbers):
            raise ValueError("scan_batch graphs are not ordered as AB, A, B")

        data.add_system_property(
            "energy",
            energy[graph_index : graph_index + 1]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone(),
        )
        data.add_node_property(
            "forces",
            forces[ab_start:ab_stop]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone(),
        )
        data.add_node_property(
            "charges",
            charges[ab_start:ab_stop]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone(),
        )
        data.add_node_property("velocities", torch.zeros_like(data.positions))
        data.add_system_property("nvt_steps_done", torch.zeros(1, 1, dtype=torch.long))
        data.add_system_property("nve_steps_done", torch.zeros(1, 1, dtype=torch.long))
        templates.append(data)

    source = Batch.from_data_list(
        [templates[index % len(templates)] for index in range(num_systems)],
        device="cpu",
    )
    initialize_velocities(
        source.velocities,
        source.atomic_masses,
        torch.full((num_systems,), temperature_k, dtype=source.positions.dtype),
        source.batch_idx.to(torch.int32),
        random_seed=int(velocity_seed),
        remove_com=True,
        remove_rotations=True,
        rescale=True,
        positions=source.positions,
    )
    return source


__all__ = [
    "InflightTraceCollector",
    "InflightTraceEvent",
    "inflight_trace_table",
    "prepare_inflight_dimer_source",
    "register_inflight_trace",
]
