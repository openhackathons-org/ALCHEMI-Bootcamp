"""GPU-resident trajectory capture for predicted-charge IR dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.hooks import DynamicsContext
from nvalchemiops.torch import segmented_sum


def _charge_and_dipole_tensors(
    charges: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    graph_idx: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce charge, centered dipole, and kinetic energy for each graph."""

    charges = charges.reshape(-1).to(positions.dtype)
    graph_idx = graph_idx.to(torch.int32)
    charge_sum = segmented_sum(charges, graph_idx, num_graphs)
    counts = segmented_sum(torch.ones_like(charges), graph_idx, num_graphs).clamp_min(
        1.0
    )
    centers = segmented_sum(positions, graph_idx, num_graphs) / counts[:, None]
    centered = positions - centers[graph_idx.long()]
    dipoles = segmented_sum(
        charges[:, None] * centered,
        graph_idx,
        num_graphs,
    )
    kinetic = segmented_sum(
        0.5 * masses * torch.sum(velocities * velocities, dim=-1),
        graph_idx,
        num_graphs,
    )
    return charge_sum, dipoles, kinetic


@dataclass(frozen=True)
class IRTrajectory:
    """CPU result copied once from :class:`PredictedChargeIRHook`."""

    dipoles_e_angstrom: np.ndarray
    charge_sums_e: np.ndarray
    kinetic_energies_eV: np.ndarray
    total_energies_eV: np.ndarray
    positions_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    atomic_masses_u: np.ndarray
    batch_idx: np.ndarray
    batch_ptr: np.ndarray
    dt_fs: float


class PredictedChargeIRHook:
    """Capture total predicted-charge dipoles after each NVE step.

    ``FusedStage`` copies non-standard model outputs such as ``charges`` to the
    batch after its one shared forward pass. ``AFTER_STEP`` is therefore the
    first safe hook point, and positions/charges still refer to the same time.
    """

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(
        self,
        warmup_steps: int,
        n_steps: int,
        dt_fs: float,
        *,
        production_status: int = 1,
        charge_tolerance: float = 5e-5,
        compile_reducer: bool = True,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if dt_fs <= 0.0:
            raise ValueError("dt_fs must be positive")
        if charge_tolerance < 0.0:
            raise ValueError("charge_tolerance must be non-negative")
        self.warmup_steps = int(warmup_steps)
        self.n_steps = int(n_steps)
        self.dt_fs = float(dt_fs)
        self.production_status = int(production_status)
        self.charge_tolerance = float(charge_tolerance)
        self._sample_index = 0
        self._warmup_calls = 0
        self._production_calls = 0
        self._reduce = (
            torch.compile(
                _charge_and_dipole_tensors,
                fullgraph=True,
                dynamic=False,
            )
            if compile_reducer
            else _charge_and_dipole_tensors
        )

    @property
    def frames_captured(self) -> int:
        """Number of production frames currently stored on the device."""

        return self._sample_index

    @property
    def stage_counts(self) -> dict[str, int]:
        """Observed fused-stage hook calls, split by the declared boundary."""

        return {
            "status_0_warmup_steps": self._warmup_calls,
            f"status_{self.production_status}_production_steps": (
                self._production_calls
            ),
        }

    def _record_stage_route(
        self, batch: Batch, expected_status: int
    ) -> torch.Tensor:
        """Accumulate route mismatches on-device without per-step sync."""

        if not hasattr(self, "_stage_route_mismatch"):
            self._stage_route_mismatch = torch.zeros(
                (), dtype=torch.bool, device=batch.device
            )
        mismatch = torch.any(batch.status.reshape(-1) != expected_status)
        self._stage_route_mismatch.logical_or_(mismatch)
        return mismatch

    def _allocate(self, batch: Batch) -> None:
        device = batch.device
        graphs = batch.num_graphs
        self._dipoles = torch.empty(
            self.n_steps, graphs, 3, dtype=torch.float64, device=device
        )
        self._charge_sums = torch.empty(
            self.n_steps, graphs, dtype=torch.float64, device=device
        )
        self._total_energies = torch.empty(
            self.n_steps, graphs, dtype=torch.float64, device=device
        )
        self._kinetic_energies = torch.empty(
            self.n_steps, graphs, dtype=torch.float64, device=device
        )
        self._positions = torch.empty(
            self.n_steps,
            batch.num_nodes,
            3,
            dtype=batch.positions.dtype,
            device=device,
        )
        self._atomic_numbers = batch.atomic_numbers.detach().clone()
        self._atomic_masses = batch.atomic_masses.detach().clone()
        self._batch_idx = batch.batch_idx.detach().clone()
        self._batch_ptr = batch.batch_ptr.detach().clone()

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        batch = ctx.batch
        if batch is None:
            return
        # AFTER_STEP fires before FusedStage advances its stage counter. The
        # first NVE frame therefore has ctx.step_count == warmup_steps.
        if ctx.step_count < self.warmup_steps:
            self._warmup_calls += 1
            mismatch = self._record_stage_route(batch, expected_status=0)
            if self._warmup_calls == 2 and bool(mismatch.cpu()):
                raise RuntimeError(
                    "Fused dynamics left NVT before the fixed warmup completed"
                )
            return
        self._production_calls += 1
        mismatch = self._record_stage_route(
            batch, expected_status=self.production_status
        )
        if self._production_calls == 1 and bool(mismatch.cpu()):
            raise RuntimeError(
                "All four isotopologues must enter NVE together after warmup"
            )
        if not hasattr(self, "_dipoles"):
            self._allocate(batch)
        if self._sample_index >= self.n_steps:
            raise RuntimeError("IR hook received more NVE steps than allocated")
        if getattr(batch, "charges", None) is None:
            raise RuntimeError(
                "batch.charges is missing: request charges from the model and "
                "register this hook on the fused workflow at AFTER_STEP"
            )

        charge_sum, dipoles, kinetic = self._reduce(
            batch.charges,
            batch.positions,
            batch.velocities,
            batch.atomic_masses,
            batch.batch_idx,
            batch.num_graphs,
        )
        potential = batch.energy.reshape(-1).to(kinetic.dtype)
        self._dipoles[self._sample_index].copy_(dipoles.to(torch.float64))
        self._charge_sums[self._sample_index].copy_(charge_sum.to(torch.float64))
        self._total_energies[self._sample_index].copy_(
            (potential + kinetic).to(torch.float64)
        )
        self._kinetic_energies[self._sample_index].copy_(kinetic.to(torch.float64))
        self._positions[self._sample_index].copy_(batch.positions.detach())
        self._sample_index += 1

    def result(self) -> IRTrajectory:
        """Validate and copy the complete production trajectory to the CPU."""

        if self._sample_index != self.n_steps:
            raise RuntimeError(
                f"Expected {self.n_steps} NVE frames, captured {self._sample_index}"
            )
        if self._warmup_calls != self.warmup_steps:
            raise RuntimeError(
                f"Expected {self.warmup_steps} NVT hook calls, observed "
                f"{self._warmup_calls}"
            )
        if self._production_calls != self.n_steps:
            raise RuntimeError(
                f"Expected {self.n_steps} NVE hook calls, observed "
                f"{self._production_calls}"
            )
        if bool(self._stage_route_mismatch.cpu()):
            raise RuntimeError(
                "Fused dynamics status did not follow the declared NVT then NVE route"
            )
        for name, tensor in (
            ("dipoles", self._dipoles),
            ("charge sums", self._charge_sums),
            ("kinetic energies", self._kinetic_energies),
            ("total energies", self._total_energies),
            ("positions", self._positions),
        ):
            if not bool(torch.isfinite(tensor).all().cpu()):
                raise RuntimeError(f"Non-finite {name} detected after NVE")
        max_charge_error = float(self._charge_sums.abs().max().cpu())
        if max_charge_error > self.charge_tolerance:
            raise RuntimeError(
                f"Predicted graph charge drifted by {max_charge_error:.3e} e"
            )
        return IRTrajectory(
            dipoles_e_angstrom=self._dipoles.cpu().numpy(),
            charge_sums_e=self._charge_sums.cpu().numpy(),
            kinetic_energies_eV=self._kinetic_energies.cpu().numpy(),
            total_energies_eV=self._total_energies.cpu().numpy(),
            positions_angstrom=self._positions.cpu().numpy(),
            atomic_numbers=self._atomic_numbers.cpu().numpy(),
            atomic_masses_u=self._atomic_masses.cpu().numpy(),
            batch_idx=self._batch_idx.cpu().numpy(),
            batch_ptr=self._batch_ptr.cpu().numpy(),
            dt_fs=self.dt_fs,
        )
