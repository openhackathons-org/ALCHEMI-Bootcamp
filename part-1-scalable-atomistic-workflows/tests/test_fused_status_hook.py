"""Focused checks for the learner-facing fused-stage status hook."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(
    os.environ.get(
        "ALCHEMI_REPOSITORY_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))
TOOLKIT_ROOT = os.environ.get("ALCHEMI_TOOLKIT_ROOT")
if TOOLKIT_ROOT:
    sys.path.insert(0, str(Path(TOOLKIT_ROOT).expanduser().resolve()))

torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

from nvalchemi.dynamics import DynamicsStage  # noqa: E402
from nvalchemi.hooks import DynamicsContext  # noqa: E402
from aux.hooks import FusedStageStatusHook  # noqa: E402
from aux.ui import FusedStageStatusCard  # noqa: E402


class _Batch:
    def __init__(
        self,
        status: list[int],
        system_ids: list[int] | None = None,
    ) -> None:
        self.status = torch.tensor(status, dtype=torch.int64).reshape(-1, 1)
        self.num_graphs = len(status)
        if system_ids is not None:
            self.system_id = torch.tensor(
                system_ids, dtype=torch.int64
            ).reshape(-1, 1)


class _Sampler:
    def __init__(self, queued: int) -> None:
        self.queued = queued

    def __len__(self) -> int:
        return self.queued


class _FusedStageProbe:
    def __init__(self, *, queued: int | None = None) -> None:
        self.exit_status = 2
        self.sampler = None if queued is None else _Sampler(queued)
        self.fused_hooks: list[object] = []

    def register_fused_hook(self, hook: object) -> None:
        self.fused_hooks.append(hook)


def _call(
    hook: FusedStageStatusHook,
    workflow: _FusedStageProbe,
    batch: _Batch,
    step: int,
) -> None:
    hook(
        DynamicsContext(
            batch=batch,
            workflow=workflow,
            step_count=step,
        ),
        DynamicsStage.BEFORE_STEP,
    )


def test_fixed_batch_without_system_ids_tracks_only_changed_stage_counts() -> None:
    card = FusedStageStatusCard(
        title="Fixed NVT + NVE",
        total=2,
        auto_display=False,
    )
    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        card=card,
    )
    stage = _FusedStageProbe()
    stage.register_fused_hook(hook)

    nvt = _Batch([0, 0])
    _call(hook, stage, nvt, 0)
    _call(hook, stage, nvt, 1)
    _call(hook, stage, _Batch([1, 1]), 2)
    final = _Batch([2, 2])
    hook.finalize(batch=final, fused_step=3, workflow=stage)

    assert stage.fused_hooks == [hook]
    assert len(hook.events) == 3
    assert hook.events[0].status_counts == (("NVT", 2), ("NVE", 0))
    assert hook.events[1].status_counts == (("NVT", 0), ("NVE", 2))
    assert hook.events[-1].completed == 2
    assert hook.events[-1].active_system_ids is None
    assert list(hook.table().columns) == [
        "Fused step",
        "Queued",
        "Active",
        "NVT",
        "NVE",
        "Completed",
    ]
    assert "COMPLETE" in card.render_string()


def test_fixed_batch_treats_toolkit_unset_system_ids_as_absent() -> None:
    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        track_system_ids=True,
    )
    stage = _FusedStageProbe()

    batch = _Batch([0, 0], [-1, -1])
    _call(hook, stage, batch, 0)

    assert hook.events[0].active_system_ids is None
    assert "Entered" not in hook.table().columns


def test_system_ids_cannot_mix_unset_and_assigned_values() -> None:
    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        track_system_ids=True,
    )
    with pytest.raises(ValueError, match="cannot mix"):
        _call(hook, _FusedStageProbe(), _Batch([0, 0], [-1, 7]), 0)


def test_inflight_batch_uses_stable_ids_and_queue_length() -> None:
    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        total_systems=6,
        track_system_ids=True,
    )
    stage = _FusedStageProbe(queued=4)

    first = _Batch([0, 0], [100, 101])
    first_status = first.status.clone()
    first_ids = first.system_id.clone()
    _call(hook, stage, first, 0)
    _call(hook, stage, _Batch([1, 0], [100, 101]), 1)

    assert stage.sampler is not None
    stage.sampler.queued = 3
    _call(hook, stage, _Batch([1, 0], [101, 102]), 2)
    _call(hook, stage, _Batch([1, 0], [101, 102]), 3)
    hook.finalize(completed_count=6, fused_step=8)

    assert torch.equal(first.status, first_status)
    assert torch.equal(first.system_id, first_ids)
    assert len(hook.events) == 4
    replacement = hook.events[2]
    assert replacement.queued == 3
    assert replacement.active == 2
    assert replacement.completed == 1
    assert replacement.entered == 1
    assert replacement.leaving == 1
    assert hook.events[-1].leaving == 2
    assert list(hook.table().columns) == [
        "Fused step",
        "Queued",
        "Active",
        "NVT",
        "NVE",
        "Completed",
        "Entered",
        "Leaving",
    ]


def test_frequency_check_precedes_any_batch_read() -> None:
    class _UnreadableBatch:
        num_graphs = 1

        @property
        def status(self):
            raise AssertionError("status should not be read between hook updates")

    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        total_systems=1,
        frequency=5,
    )
    hook(
        DynamicsContext(
            batch=_UnreadableBatch(),
            workflow=_FusedStageProbe(),
            step_count=3,
        ),
        DynamicsStage.BEFORE_STEP,
    )

    assert hook.events == ()


def test_count_only_finalize_requires_a_fully_drained_run() -> None:
    hook = FusedStageStatusHook(
        status_labels={0: "NVT", 1: "NVE"},
        total_systems=4,
    )
    with pytest.raises(ValueError, match="fully drained"):
        hook.finalize(completed_count=3, fused_step=5)
