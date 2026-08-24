# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified model assets and Toolkit adapters shared by Core modules."""

from __future__ import annotations

import os
import sys
import tomllib
import warnings
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

from .data_records import PROJECT_ROOT, sha256_file

COULOMB_CONSTANT_EV_ANGSTROM = 14.399645351950548
_MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_RUNTIME_PINS = PROJECT_ROOT / "environment" / "runtime-pins.toml"


def model_checkpoint() -> Path:
    """Resolve and verify the pinned AIMNet2 checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(_MODEL_ALIAS)).resolve()
    digest = sha256_file(path)
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"AIMNet2 checkpoint checksum mismatch: {digest}")
    return path


@contextmanager
def suppress_known_mace_load_noise():
    """Hide pinned dependency notices that do not describe the learner's run."""

    with warnings.catch_warnings(), redirect_stdout(StringIO()):
        warnings.filterwarnings(
            "ignore",
            message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
            category=UserWarning,
        )
        yield


def d3_parameter_file() -> Path:
    """Resolve and verify the D3 element-reference tables prepared by ``scripts/setup``."""

    configured = os.environ.get("ALCHEMI_D3_PARAM_FILE")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = Path(sys.prefix).resolve().parent / "dftd3" / "dftd3_parameters.pt"
        if not path.is_file():
            raise RuntimeError(
                "ALCHEMI_D3_PARAM_FILE is not configured and the current Python "
                "runtime has no prepared D3 parameter file"
            )
    pins = tomllib.loads(_RUNTIME_PINS.read_text(encoding="utf-8"))
    expected = str(pins["dispersion"]["generated_parameter_sha256"])
    if not path.is_file():
        raise RuntimeError(f"D3 parameter file is missing: {path}")
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(
            f"D3 parameter file checksum mismatch: {digest}; expected {expected}"
        )
    return path


class _NativeCoulomb(torch.nn.Module):
    """Finite 1/r electrostatic energy expressed as a native PyTorch module."""

    def __init__(self, coulomb_constant: float) -> None:
        super().__init__()
        self.coulomb_constant = float(coulomb_constant)

    def forward(
        self,
        positions: torch.Tensor,
        partial_charges: torch.Tensor,
        neighbor_pairs: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        source, target = neighbor_pairs.unbind(dim=1)
        source = source.to(torch.long)
        target = target.to(torch.long)
        charges = partial_charges.reshape(-1)
        distances = torch.linalg.vector_norm(
            positions[source] - positions[target], dim=-1
        )
        pair_energy = (
            0.5 * self.coulomb_constant * charges[source] * charges[target] / distances
        )
        energy = torch.zeros(
            num_graphs,
            dtype=positions.dtype,
            device=positions.device,
        )
        energy.index_add_(0, batch_idx[source].to(torch.long), pair_energy)
        return energy.unsqueeze(-1)


class DirectCoulombAdapter(torch.nn.Module, BaseModelMixin):
    """Adapt finite 1/r electrostatics to Toolkit model composition.

    ``partial_charges`` comes from an earlier model step. A full COO neighbor
    list includes both pair directions, and the native module assigns half of
    each ordered-pair energy to avoid double counting.
    """

    def __init__(
        self,
        cutoff: float,
        coulomb_constant: float = COULOMB_CONSTANT_EV_ANGSTROM,
    ) -> None:
        super().__init__()
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be finite and positive")
        self.cutoff = float(cutoff)
        self.native = _NativeCoulomb(coulomb_constant)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"partial_charges"}),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        del data, kwargs
        raise NotImplementedError("Direct Coulomb has no learned embeddings")

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data], device=data.positions.device)
        inputs = super().adapt_input(data, **kwargs)
        inputs.pop("atomic_numbers", None)
        inputs["neighbor_pairs"] = inputs.pop("neighbor_list")
        inputs["batch_idx"] = data.batch_idx
        inputs["num_graphs"] = data.num_graphs
        return inputs

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> OrderedDict[str, torch.Tensor]:
        inputs = self.adapt_input(data, **kwargs)
        energy = self.native(**inputs)
        output: dict[str, torch.Tensor] = {"energy": energy}
        if "forces" in self.model_config.active_outputs:
            forces = -torch.autograd.grad(
                energy.sum(),
                inputs["positions"],
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            output["forces"] = forces
        return self.adapt_output(output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        del path, as_state_dict
        raise NotImplementedError("This tutorial wrapper has no export format")


def prepare_model_batch(batch: Batch, model: Any) -> Batch:
    """Apply a model's neighbor hooks before one standalone forward call."""

    from nvalchemi.hooks import DynamicsContext

    context = DynamicsContext(batch=batch, model=model)
    for hook in model.make_neighbor_hooks():
        hook(context, DynamicsStage.BEFORE_COMPUTE)
    return batch
