"""Observation hooks and callbacks for the training examples."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from nvalchemi.training import TrainingStage


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _component(
    values: dict[str, Any],
    prefix: str,
) -> float:
    for name, value in values.items():
        if name.lower().startswith(prefix):
            return _as_float(value)
    return float("nan")


class ParameterOwnershipRecorder:
    """Record actual gradient and optimizer ownership before training."""

    stage = TrainingStage.BEFORE_TRAINING
    frequency = 1

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        del stage
        optimizer_parameters = {
            id(parameter)
            for optimizer in ctx.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.rows = [
            {
                "parameter": f"{model_name}.{parameter_name}",
                "shape": tuple(parameter.shape),
                "requires_grad": parameter.requires_grad,
                "trainable": parameter.requires_grad,
                "optimizer_owned": id(parameter) in optimizer_parameters,
            }
            for model_name, model in ctx.models.items()
            for parameter_name, parameter in model.named_parameters()
        ]

    def frame(self) -> pd.DataFrame:
        """Return recorded parameter ownership in model order."""

        return pd.DataFrame(self.rows)


class TrainingHistory:
    """Collect training loss, validation loss, and fitted LJ parameters."""

    stage = None
    frequency = 1

    def __init__(
        self,
        model: Any | None = None,
        *,
        energy_mse_label: str | None = "energy MSE (eV²)",
        force_mse_label: str | None = "force MSE (eV²/Å²)",
    ) -> None:
        self.model = model
        self.energy_mse_label = energy_mse_label
        self.force_mse_label = force_mse_label
        self.training_rows: list[dict[str, float]] = []
        self.validation_rows: list[dict[str, float]] = []

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        return stage in {
            TrainingStage.AFTER_BATCH,
            TrainingStage.AFTER_VALIDATION,
        }

    def _parameter_values(self, ctx: Any) -> dict[str, float]:
        model = self.model if self.model is not None else ctx.model
        if model is None or not hasattr(model, "epsilon_eV"):
            return {}
        return {
            "epsilon (eV)": _as_float(model.epsilon_eV),
            "sigma (Å)": _as_float(model.sigma_A),
        }

    def _component_values(
        self,
        components: dict[str, Any],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.energy_mse_label is not None:
            values[self.energy_mse_label] = _component(components, "energymse")
        if self.force_mse_label is not None:
            values[self.force_mse_label] = _component(components, "forcemse")
        return values

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        if stage is TrainingStage.AFTER_BATCH:
            components = (
                dict(ctx.losses["per_component_unweighted"])
                if ctx.losses is not None
                else {}
            )
            self.training_rows.append(
                {
                    "completed optimizer updates": int(ctx.step_count),
                    "total loss": _as_float(ctx.loss),
                    **self._component_values(components),
                    **self._parameter_values(ctx),
                }
            )
            return

        summary = ctx.validation
        if summary is None:
            raise RuntimeError("validation history requires ctx.validation")
        components = dict(summary["per_component_unweighted"])
        self.validation_rows.append(
            {
                "completed optimizer updates": int(ctx.step_count),
                "total loss": _as_float(summary["total_loss"]),
                **self._component_values(components),
                **self._parameter_values(ctx),
            }
        )

    def training_frame(self) -> pd.DataFrame:
        """Return one row per completed optimizer update."""

        return pd.DataFrame(self.training_rows)

    def validation_frame(self) -> pd.DataFrame:
        """Return one row per completed validation pass."""

        return pd.DataFrame(self.validation_rows)


class ValidationBatchRecorder:
    """Record validation identities and output shapes per batch."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        batch: Any,
        predictions: dict[str, torch.Tensor],
        loss: dict[str, Any],
        batch_count: int,
        step_count: int,
        epoch: int,
    ) -> None:
        sample_ids = getattr(batch, "sample_id", None)
        energy = predictions.get("energy")
        if energy is None:
            energy = predictions["predicted_energy"]
        forces = predictions.get("forces")
        if forces is None:
            forces = predictions.get("predicted_forces")
        self.rows.append(
            {
                "completed optimizer updates": int(step_count),
                "epoch": int(epoch),
                "validation batch": int(batch_count),
                "sample_ids": (
                    tuple(int(item) for item in sample_ids.flatten().cpu())
                    if sample_ids is not None
                    else ()
                ),
                "energy_shape": tuple(energy.shape),
                "force_shape": (tuple(forces.shape) if forces is not None else None),
                "loss": _as_float(loss["total_loss"]),
            }
        )

    def frame(self) -> pd.DataFrame:
        """Return the callback observations."""

        return pd.DataFrame(self.rows)
