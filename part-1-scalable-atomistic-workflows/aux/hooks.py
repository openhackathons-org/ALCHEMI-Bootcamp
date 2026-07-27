"""Small adapters for Toolkit dynamics hooks used by the notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.hooks import DynamicsContext

if TYPE_CHECKING:
    from .ui import NotebookProgress


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
    "NotebookStageProgressHook",
    "StageStepCounterHook",
    "add_stage_step_counters",
    "converge_after_steps",
]
