"""Small adapters for Toolkit dynamics hooks used by the notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.hooks import DynamicsContext

if TYPE_CHECKING:
    from .ui import FusedStageStatusCard, NotebookProgress


@dataclass(frozen=True)
class FusedStageStatusEvent:
    """One changed state observed by a fused-stage hook."""

    fused_step: int
    queued: int
    active: int
    status_counts: tuple[tuple[str, int], ...]
    completed: int
    entered: int | None
    leaving: int | None
    active_system_ids: tuple[int, ...] | None


class FusedStageStatusHook:
    """Trace stage occupancy without changing the dynamics batch.

    Register this object with ``FusedStage.register_fused_hook``. Toolkit
    applies ``frequency`` before calling the hook; the defensive check in
    :meth:`__call__` also prevents a device-to-host copy when the hook is
    invoked directly at an intermediate step.
    """

    stage = DynamicsStage.BEFORE_STEP

    def __init__(
        self,
        *,
        status_labels: Mapping[int, str],
        total_systems: int | None = None,
        frequency: int = 1,
        track_system_ids: bool = False,
        card: FusedStageStatusCard | None = None,
    ) -> None:
        if isinstance(frequency, bool) or not isinstance(frequency, Integral):
            raise TypeError("frequency must be an integer")
        if int(frequency) <= 0:
            raise ValueError("frequency must be positive")
        if not status_labels:
            raise ValueError("status_labels must not be empty")
        if not isinstance(track_system_ids, bool):
            raise TypeError("track_system_ids must be a boolean")

        labels: dict[int, str] = {}
        for status, label in status_labels.items():
            if isinstance(status, bool) or not isinstance(status, Integral):
                raise TypeError("status codes must be integers")
            clean_label = str(label).strip()
            if not clean_label:
                raise ValueError("status labels must not be empty")
            labels[int(status)] = clean_label
        if len(set(labels.values())) != len(labels):
            raise ValueError("status labels must be unique")

        if total_systems is not None:
            if isinstance(total_systems, bool) or not isinstance(
                total_systems, Integral
            ):
                raise TypeError("total_systems must be an integer or None")
            if int(total_systems) <= 0:
                raise ValueError("total_systems must be positive")
            total_systems = int(total_systems)

        self.status_labels = labels
        self.total_systems = total_systems
        self.frequency = int(frequency)
        self.track_system_ids = track_system_ids
        self.card = card
        self._events: list[FusedStageStatusEvent] = []
        self._last_state: tuple[Any, ...] | None = None
        self._last_system_ids: tuple[int, ...] | None = None
        self._tracks_system_ids = False
        self._finalized = False

    @property
    def events(self) -> tuple[FusedStageStatusEvent, ...]:
        """Return the changed states retained for static notebook output."""

        return tuple(self._events)

    def _read_batch(
        self, batch: Batch
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        status = getattr(batch, "status", None)
        if not isinstance(status, torch.Tensor):
            raise ValueError("batch.status must be a tensor")
        status = status.detach().reshape(-1)
        if status.numel() != batch.num_graphs:
            raise ValueError("batch.status must contain one value per system")
        if torch.is_floating_point(status) or status.dtype == torch.bool:
            raise ValueError("batch.status must use an integer dtype")
        if not self.track_system_ids:
            return tuple(int(value) for value in status.cpu().tolist()), None

        system_id = getattr(batch, "system_id", None)
        if system_id is None:
            return tuple(int(value) for value in status.cpu().tolist()), None
        if not isinstance(system_id, torch.Tensor):
            raise ValueError("batch.system_id must be a tensor when present")
        system_id = system_id.detach().reshape(-1)
        if system_id.numel() != batch.num_graphs:
            raise ValueError("batch.system_id must contain one value per system")
        if torch.is_floating_point(system_id) or system_id.dtype == torch.bool:
            raise ValueError("batch.system_id must use an integer dtype")

        # One transfer covers both arrays. It occurs only at the configured
        # hook frequency, rather than on every dynamics step.
        values = torch.stack(
            (status.to(dtype=torch.int64), system_id.to(dtype=torch.int64))
        ).cpu()
        status_values = tuple(int(value) for value in values[0].tolist())
        system_ids = tuple(int(value) for value in values[1].tolist())
        if system_ids and all(value == -1 for value in system_ids):
            # Toolkit creates this sentinel field for a fixed batch that has
            # not been assigned stable inflight IDs.
            return status_values, None
        if any(value == -1 for value in system_ids):
            raise ValueError(
                "batch.system_id cannot mix unset and assigned values"
            )
        if len(set(system_ids)) != len(system_ids):
            raise ValueError("batch.system_id values must be unique")
        return status_values, system_ids

    @staticmethod
    def _queued_count(workflow: Any) -> int:
        sampler = getattr(workflow, "sampler", None)
        if sampler is None:
            return 0
        queued = len(sampler)
        if isinstance(queued, bool) or not isinstance(queued, Integral):
            raise TypeError("len(workflow.sampler) must return an integer")
        if int(queued) < 0:
            raise ValueError("queued system count cannot be negative")
        return int(queued)

    def _record(
        self,
        *,
        batch: Batch,
        workflow: Any,
        fused_step: int,
    ) -> bool:
        status_values, system_ids = self._read_batch(batch)
        exit_status = getattr(
            workflow,
            "exit_status",
            max(self.status_labels) + 1,
        )
        if isinstance(exit_status, bool) or not isinstance(exit_status, Integral):
            raise TypeError("workflow.exit_status must be an integer")
        exit_status = int(exit_status)

        counts = {label: 0 for label in self.status_labels.values()}
        completed_in_batch = 0
        for status in status_values:
            if status >= exit_status:
                completed_in_batch += 1
                continue
            try:
                label = self.status_labels[status]
            except KeyError as exc:
                raise ValueError(
                    f"active status {status} is missing from status_labels"
                ) from exc
            counts[label] += 1

        queued = self._queued_count(workflow)
        batch_size = len(status_values)
        if self.total_systems is None:
            self.total_systems = queued + batch_size
        retired = self.total_systems - queued - batch_size
        if retired < 0:
            raise ValueError(
                "total_systems is smaller than the queued and batched systems"
            )
        completed = retired + completed_in_batch
        active = batch_size - completed_in_batch
        if queued + active + completed != self.total_systems:
            raise RuntimeError("fused-stage status counts do not add up")

        ordered_counts = tuple(
            (label, counts[label]) for label in self.status_labels.values()
        )
        state = (queued, active, ordered_counts, completed, system_ids)
        changed = state != self._last_state
        if changed:
            if system_ids is None:
                entered = leaving = None
            else:
                self._tracks_system_ids = True
                previous = set(self._last_system_ids or ())
                current = set(system_ids)
                entered = len(current - previous)
                leaving = len(previous - current)
            event = FusedStageStatusEvent(
                fused_step=int(fused_step),
                queued=queued,
                active=active,
                status_counts=ordered_counts,
                completed=completed,
                entered=entered,
                leaving=leaving,
                active_system_ids=system_ids,
            )
            self._events.append(event)
            self._last_state = state
            self._last_system_ids = system_ids
            if self.card is not None:
                self.card.update(
                    fused_step=event.fused_step,
                    queued=event.queued,
                    active=event.active,
                    status_counts=dict(event.status_counts),
                    completed=event.completed,
                )

        return changed

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Read one full-batch snapshot from the public hook context."""

        if self._finalized or stage is not DynamicsStage.BEFORE_STEP:
            return
        if int(ctx.step_count) % self.frequency:
            return
        if ctx.batch is None:
            return
        self._record(
            batch=ctx.batch,
            workflow=ctx.workflow,
            fused_step=int(ctx.step_count),
        )

    def finalize(
        self,
        *,
        batch: Batch | None = None,
        completed_count: int | None = None,
        fused_step: int,
        workflow: Any = None,
    ) -> None:
        """Close the trace explicitly after ``FusedStage.run`` returns.

        Pass the returned ``batch`` for a fixed batch. For a fully drained
        inflight run, pass ``completed_count`` from the sink instead.
        """

        if self._finalized:
            raise RuntimeError("fused-stage status hook is already finalized")
        if (batch is None) == (completed_count is None):
            raise ValueError("pass exactly one of batch or completed_count")
        if isinstance(fused_step, bool) or not isinstance(fused_step, Integral):
            raise TypeError("fused_step must be an integer")
        if int(fused_step) < 0:
            raise ValueError("fused_step must be non-negative")

        if batch is not None:
            self._record(
                batch=batch,
                workflow=workflow,
                fused_step=int(fused_step),
            )
        else:
            if self.total_systems is None:
                raise RuntimeError(
                    "total_systems is required before count-only finalization"
                )
            if isinstance(completed_count, bool) or not isinstance(
                completed_count, Integral
            ):
                raise TypeError("completed_count must be an integer")
            if int(completed_count) != self.total_systems:
                raise ValueError(
                    "count-only finalization requires a fully drained run"
                )
            ordered_counts = tuple(
                (label, 0) for label in self.status_labels.values()
            )
            system_ids = () if self._tracks_system_ids else None
            state = (0, 0, ordered_counts, self.total_systems, system_ids)
            if state != self._last_state:
                previous = set(self._last_system_ids or ())
                event = FusedStageStatusEvent(
                    fused_step=int(fused_step),
                    queued=0,
                    active=0,
                    status_counts=ordered_counts,
                    completed=self.total_systems,
                    entered=0 if self._tracks_system_ids else None,
                    leaving=len(previous) if self._tracks_system_ids else None,
                    active_system_ids=system_ids,
                )
                self._events.append(event)
                self._last_state = state
                self._last_system_ids = system_ids
        latest = self._events[-1]
        if (
            latest.queued != 0
            or latest.active != 0
            or latest.completed != self.total_systems
        ):
            raise ValueError("cannot mark an incomplete fused-stage run complete")
        if self.card is not None:
            self.card.finalize(
                fused_step=latest.fused_step,
                queued=latest.queued,
                active=latest.active,
                status_counts=dict(latest.status_counts),
                completed=latest.completed,
            )
        self._finalized = True

    def table(self) -> Any:
        """Return a compact DataFrame containing changed hook snapshots."""

        import pandas as pd

        rows = []
        for event in self._events:
            row: dict[str, Any] = {
                "Fused step": event.fused_step,
                "Queued": event.queued,
                "Active": event.active,
            }
            row.update(dict(event.status_counts))
            row["Completed"] = event.completed
            if self._tracks_system_ids:
                row["Entered"] = event.entered
                row["Leaving"] = event.leaving
            rows.append(row)
        columns = [
            "Fused step",
            "Queued",
            "Active",
            *self.status_labels.values(),
            "Completed",
        ]
        if self._tracks_system_ids:
            columns.extend(("Entered", "Leaving"))
        return pd.DataFrame(rows, columns=columns)


class NotebookStageProgressHook:
    """Adapt Toolkit dynamics progress to one in-place notebook card."""

    stage = DynamicsStage.AFTER_STEP

    def __init__(
        self,
        progress: "NotebookProgress",
        *,
        frequency: int,
        label: str,
    ) -> None:
        self.progress = progress
        self.frequency = int(frequency)
        self.label = str(label)

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        done = min(int(ctx.step_count) + 1, self.progress.total)
        self.progress.update(done=done, message=f"{self.label}: {done:,} steps")


class StageStepCounterHook:
    """Count actual updates from each system's status at step entry.

    ``FusedStage`` calls fused ``BEFORE_STEP`` hooks before any integrator
    update or status migration.  The counter therefore advances only for the
    stage that will update a system in the current fused step.
    """

    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(self, status_to_field: Mapping[int, str]) -> None:
        if not status_to_field:
            raise ValueError("status_to_field must not be empty")
        self.status_to_field = {
            int(status): str(field) for status, field in status_to_field.items()
        }
        if len(set(self.status_to_field.values())) != len(self.status_to_field):
            raise ValueError("each status must use a distinct counter field")

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        status = ctx.batch.status.reshape(-1)
        for status_code, field in self.status_to_field.items():
            counter = getattr(ctx.batch, field).reshape(-1)
            counter[status == status_code] += 1


class _AtLeast:
    """Device-side comparison used by :func:`converge_after_steps`."""

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, counter: torch.Tensor) -> torch.Tensor:
        if counter.ndim == 1:
            values = counter
        elif counter.ndim == 2 and counter.shape[1] == 1:
            values = counter[:, 0]
        else:
            raise ValueError("a stage-step counter must have shape (B,) or (B, 1)")
        return values >= self.value


def converge_after_steps(field: str, n_steps: int) -> ConvergenceHook:
    """Return a public Toolkit convergence hook for an exact step budget.

    Pair this hook with :class:`StageStepCounterHook` and leave the dynamics
    stage's built-in ``n_steps`` unset.  Migration then happens after exactly
    ``n_steps`` actual updates of that stage for each system.
    """

    if not field:
        raise ValueError("field must not be empty")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return ConvergenceHook(
        criteria={
            "key": str(field),
            "threshold": 0.0,
            "custom_op": _AtLeast(n_steps),
        }
    )


def add_stage_step_counters(batch: Batch, fields: Sequence[str]) -> None:
    """Add zeroed per-system counters to an existing Toolkit batch."""

    if not fields:
        raise ValueError("fields must not be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("counter field names must be unique")
    if batch.keys is None:
        raise ValueError("batch field levels are unavailable")
    for field in fields:
        if not field:
            raise ValueError("counter field names must not be empty")
        if field in batch:
            raise ValueError(f"batch already contains {field!r}")
        batch[field] = torch.zeros(
            batch.num_graphs,
            1,
            dtype=torch.long,
            device=batch.device,
        )
        batch.keys.setdefault("system", set()).add(field)


__all__ = [
    "FusedStageStatusEvent",
    "FusedStageStatusHook",
    "NotebookStageProgressHook",
    "StageStepCounterHook",
    "add_stage_step_counters",
    "converge_after_steps",
]
