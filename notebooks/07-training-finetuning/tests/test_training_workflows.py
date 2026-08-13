"""Workflow checks for fine-tuning, validation, and restart."""

from __future__ import annotations

from pathlib import Path

import helpers
import torch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import (
    CheckpointHook,
    ComposedLossFunction,
    EnergyMSELoss,
    FineTuningStrategy,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
    default_training_fn,
)


def test_toy_finetuning_updates_only_the_selected_readout() -> None:
    model, train_loader, validation_loader, split_frame = helpers.prepare_toy_transfer(
        device=torch.device("cpu")
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    ownership = helpers.ParameterOwnershipRecorder()
    history = helpers.TrainingHistory()
    validation_batches = helpers.ValidationBatchRecorder()
    strategy = FineTuningStrategy(
        models=model,
        trainable_patterns=("main.readout.*",),
        freeze_mode="requires_grad",
        optimizer_configs=OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 2.0e-2},
        ),
        training_fn=default_training_fn,
        loss_fn=EnergyMSELoss(dtype_policy="prediction_to_target"),
        validation_config=ValidationConfig(
            validation_data=validation_loader,
            every_n_steps=2,
            batch_callback=validation_batches,
            name="toy-validation",
        ),
        num_steps=4,
        devices=[torch.device("cpu")],
        hooks=[ownership, history],
    )

    strategy.run(train_loader)

    assert split_frame.groupby("split").size().to_dict() == {
        "train": 12,
        "validation": 4,
    }
    assert ownership.frame()["trainable"].tolist() == [False, False, True, True]
    assert torch.equal(model.backbone[0].weight, before["backbone.0.weight"])
    assert torch.equal(model.backbone[0].bias, before["backbone.0.bias"])
    assert not torch.equal(model.readout.weight, before["readout.weight"])
    assert not torch.equal(model.readout.bias, before["readout.bias"])
    assert len(history.training_rows) == 4
    assert len(history.validation_rows) >= 2
    assert validation_batches.rows
    assert validation_batches.rows[0]["energy_shape"][-1] == 1


def _argon_loss() -> ComposedLossFunction:
    return ComposedLossFunction(
        [EnergyMSELoss(), ForceMSELoss(normalize_by_atom_count=True)],
        weights=[1.0 / 0.02**2, 1.0 / 0.02**2],
        normalize_weights=False,
    )


def _neighbor_hook(model: helpers.TrainableLennardJones) -> NeighborListHook:
    return NeighborListHook(
        model.model_config.neighbor_config,
        stage=TrainingStage.BEFORE_FORWARD,
    )


def test_checkpoint_resume_restores_model_optimizer_and_counters(
    tmp_path: Path,
) -> None:
    records = helpers.generate_argon_records(
        count=12,
        seed=84,
        epsilon_eV=0.0104,
        sigma_A=3.40,
        cutoff_A=7.0,
        dtype=torch.float64,
        device="cpu",
    )
    train_records, validation_records, _ = helpers.split_argon_records(
        records,
        validation_count=4,
        seed=17,
    )
    train_loader = helpers.make_loader(train_records, batch_size=4)
    validation_loader = helpers.make_loader(validation_records, batch_size=4)
    model = helpers.TrainableLennardJones(
        epsilon_eV=0.007,
        sigma_A=3.10,
        cutoff_A=7.0,
    ).double()
    uninterrupted_model = helpers.TrainableLennardJones(
        epsilon_eV=0.007,
        sigma_A=3.10,
        cutoff_A=7.0,
    ).double()
    checkpoint_dir = tmp_path / "checkpoints"
    history = helpers.TrainingHistory(model)
    checkpoint_hook = CheckpointHook(
        checkpoint_dir,
        step_interval=2,
        async_save=False,
    )
    strategy = FineTuningStrategy(
        models=model,
        trainable_patterns=("main.log_epsilon", "main.log_sigma"),
        optimizer_configs=OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 2.0e-2},
        ),
        training_fn=default_training_fn,
        loss_fn=_argon_loss(),
        validation_config=ValidationConfig(
            validation_data=validation_loader,
            every_n_steps=2,
            name="argon-validation",
        ),
        num_steps=4,
        devices=[torch.device("cpu")],
        hooks=[_neighbor_hook(model), history, checkpoint_hook],
    )

    strategy.run(train_loader)
    saved_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    assert checkpoint_hook.last_checkpoint_index == 1

    resumed_history = helpers.TrainingHistory()
    resumed = FineTuningStrategy.load_checkpoint(
        checkpoint_dir,
        map_location="cpu",
        hooks=[resumed_history],
    )
    resumed_model = resumed.models["main"]

    assert resumed.step_count == 4
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, saved_parameters[name])

    resumed.hooks.extend([_neighbor_hook(resumed_model)])
    resumed.validation_config = ValidationConfig(
        validation_data=validation_loader,
        every_n_steps=2,
        name="argon-validation",
    )
    resumed.num_steps = 6
    resumed.run(train_loader)

    assert resumed.step_count == 6
    assert len(resumed_history.training_rows) == 2
    assert resumed_history.validation_rows

    uninterrupted = FineTuningStrategy(
        models=uninterrupted_model,
        trainable_patterns=("main.log_epsilon", "main.log_sigma"),
        optimizer_configs=OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 2.0e-2},
        ),
        training_fn=default_training_fn,
        loss_fn=_argon_loss(),
        num_steps=6,
        devices=[torch.device("cpu")],
        hooks=[_neighbor_hook(uninterrupted_model)],
    )
    uninterrupted.run(train_loader)
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, uninterrupted_model.state_dict()[name])
