"""MD hook classes and factories used by the dynamics pipelines."""

from loguru import logger
import time

import torch

from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import MaxForceClampHook, NaNDetectorHook
from nvalchemi.hooks import WrapPeriodicHook

from .constants import MAX_FORCE_CLAMP

try:
    # The toolkit's own temperature reduction (3N-DOF equipartition, correct
    # amu·Å²/fs² -> eV unit handling) so the live readout matches the CSV log.
    from nvalchemi.dynamics.hooks._utils import temperature_per_graph
except ImportError:  # pragma: no cover - tolerate a future toolkit layout change
    temperature_per_graph = None


def make_safety_hooks(model_or_pipe, track_stress=True, max_force=MAX_FORCE_CLAMP):
    """Defensive MD hooks (order: neighbor -> wrap -> clamp -> NaN).

    ``model_or_pipe`` is any object exposing ``make_neighbor_hooks()`` --
    both :class:`AIMNet2Wrapper` and :class:`PipelineModelWrapper` qualify.
    ``track_stress=False`` drops stress from ``NaNDetectorHook.extra_keys``
    for runs without a stress-producing pipeline (e.g. AIMNet2 without Ewald
    configured for stress).
    """
    extra = ["stress"] if track_stress else []
    return [
        *model_or_pipe.make_neighbor_hooks(),
        WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE),
        MaxForceClampHook(max_force=max_force),
        NaNDetectorHook(extra_keys=extra),
    ]


class InitVelocitiesOnConverge:
    """ON_CONVERGE hook that re-initializes Maxwell-Boltzmann velocities.

    Used at the FIRE -> NVT transition in a FusedStage pipeline to replace
    FIRE's optimization-velocities with thermal velocities at T.
    """

    stage = DynamicsStage.ON_CONVERGE

    def __init__(self, temperature, seed=42):
        self.temperature = temperature
        self.seed = seed
        self.frequency = 1

    def __call__(self, ctx, stage_):
        batch = ctx.batch
        batch.velocities = torch.zeros_like(batch.positions)
        initialize_velocities(
            batch.velocities,
            batch.atomic_masses,
            temperature=torch.tensor([self.temperature], device=batch.positions.device),
            batch_idx=batch.batch_idx,
            random_seed=self.seed,
            remove_com=True,
            rescale=True,
        )


class StatusTransitionLogger:
    """Fused BEFORE_STEP hook that logs every batch.status transition.

    Unlike ON_CONVERGE hooks (which only fire for stages that have a
    convergence_hook), this observes batch.status directly and catches
    n_steps-based migrations (NVT -> NPT, NPT -> done) as well as
    convergence-based ones (FIRE -> NVT). Register via
    ``fused.register_fused_hook(...)``.
    """

    stage = DynamicsStage.BEFORE_STEP

    def __init__(self, labels, frequency=1):
        self.labels = labels
        self.frequency = frequency
        self._prev = None
        self._t0 = time.monotonic()

    def __call__(self, ctx, stage_):
        status = ctx.batch.status.view(-1).tolist()
        if self._prev is None:
            self._prev = list(status)
            return
        elapsed = time.monotonic() - self._t0
        for prev, curr in zip(self._prev, status):
            if prev != curr:
                src = self.labels.get(prev, f"s{prev}")
                dst = self.labels.get(curr, f"s{curr}")
                logger.info(
                    "[{}->{}] graduated at step={}  elapsed={:.2f}s",
                    src,
                    dst,
                    ctx.step_count,
                    elapsed,
                )
        self._prev = list(status)

    def finalize(self, batch):
        """Flush any transition that happened on the last executed step.

        ``fused.run`` exits as soon as all systems reach exit_status, before
        the next ``BEFORE_STEP`` fires. Call this manually after the run to
        catch the final graduation (e.g. NPT -> done).
        """
        if self._prev is None:
            return
        status = batch.status.view(-1).tolist()
        elapsed = time.monotonic() - self._t0
        for prev, curr in zip(self._prev, status):
            if prev != curr:
                src = self.labels.get(prev, f"s{prev}")
                dst = self.labels.get(curr, f"s{curr}")
                logger.info(
                    "[{}->{}] graduated at end of run  elapsed={:.2f}s",
                    src,
                    dst,
                    elapsed,
                )
        self._prev = list(status)


def stdout_writer(step, rows):
    """Print log rows to stdout for Jupyter cell output."""
    for row in rows:
        parts = [
            f"{k}={v:.4g}" for k, v in row.items() if k not in ("graph_idx", "status")
        ]
        print(f"  [{int(step):>6d}] {' | '.join(parts)}")


def make_graph_tagged_writer(labels):
    """Stdout writer that prefixes each row with ``labels[graph_idx]``."""

    def writer(step, rows):
        for row in rows:
            gi = int(row.get("graph_idx", 0))
            tag = labels[gi] if gi < len(labels) else f"g{gi}"
            parts = [
                f"{k}={v:.4g}"
                for k, v in row.items()
                if k not in ("graph_idx", "status")
            ]
            print(f"  [{int(step):>6d}] {tag} | {' | '.join(parts)}")

    return writer


class ProgressHook:
    """Synchronous ``AFTER_STEP`` hook that advances a ``NotebookProgress`` bar.

    A ``LoggingHook(backend="custom")`` writer dispatches its work on a background
    thread, so the rapid widget writes a live bar needs are coalesced and the bar
    only catches up once ``run()`` returns -- it appears to jump in large strides
    rather than tick steadily. This hook instead updates the bar inline on the
    integrator's main thread (the pattern ``tqdm`` uses), so it advances smoothly
    every ``frequency`` steps; ``LoggingHook(backend="csv")`` records the full
    scalar set to disk in parallel.

    Per the toolkit hook convention the dynamics loop reads ``stage`` and
    ``frequency`` (``frequency`` gates how often ``__call__`` fires). The message
    reports the instantaneous temperature -- per graph for a multi-temperature
    batch -- via the toolkit's own ``temperature_per_graph`` so the units match
    the CSV log.
    """

    stage = DynamicsStage.AFTER_STEP

    def __init__(self, progress, frequency=1):
        self.progress = progress
        self.frequency = frequency

    def __call__(self, ctx, stage_):
        batch = ctx.batch
        done = int(ctx.step_count)
        message = f"step {done}"
        velocities = getattr(batch, "velocities", None)
        if velocities is not None and temperature_per_graph is not None:
            temps = temperature_per_graph(
                velocities,
                batch.atomic_masses,
                batch.batch_idx,
                batch.num_graphs,
                batch.num_nodes_per_graph,
            ).tolist()
            if len(temps) == 1:
                message += f" · temperature={temps[0]:.3g} K"
            else:
                message += " · T = [" + ", ".join(f"{t:.0f}" for t in temps) + "] K"
        self.progress.update(done=done, message=message)
