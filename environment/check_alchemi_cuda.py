# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one small Toolkit model calculation on the assigned CUDA device."""

from __future__ import annotations

import tomllib
from pathlib import Path

import torch
from aimnet.calculators.model_registry import get_model_path
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import AIMNet2Wrapper
from nvalchemi.neighbors import compute_neighbors

ROOT = Path(__file__).resolve().parents[1]
PINS = tomllib.loads((ROOT / "environment" / "runtime-pins.toml").read_text())


def main() -> None:
    """Require one finite AIMNet2 energy and charge prediction on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the ALCHEMI model check")

    device = torch.device("cuda")
    structure = molecule("H2O")
    structure.info["charge"] = 0
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(structure, device=device)],
        device=device,
    )

    model_pin = PINS["model"]
    model = AIMNet2Wrapper.from_checkpoint(
        get_model_path(model_pin["checkpoint_alias"]),
        device=device,
        dtype=torch.float32,
    ).eval()
    model.set_config("active_outputs", {"energy", "charges"})
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    output = model(batch)
    torch.cuda.synchronize(device)

    energy = output["energy"]
    charges = output["charges"]
    if energy.device.type != "cuda" or charges.device.type != "cuda":
        raise RuntimeError("ALCHEMI model output left the assigned CUDA device")
    if energy.numel() != 1 or charges.numel() != len(structure):
        raise RuntimeError("ALCHEMI model output has unexpected tensor shapes")
    if not torch.isfinite(energy).all() or not torch.isfinite(charges).all():
        raise RuntimeError("ALCHEMI model output contains a non-finite value")

    print("ALCHEMI CUDA model path verified")


if __name__ == "__main__":
    main()
