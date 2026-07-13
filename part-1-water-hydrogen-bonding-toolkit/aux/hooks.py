"""Notebook presentation hooks that do not change simulation state."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


__all__ = ["NotebookStageProgressHook"]
