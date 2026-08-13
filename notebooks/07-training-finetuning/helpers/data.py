"""Deterministic datasets for the training and fine-tuning examples."""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from nvalchemi.data import AtomicData, Batch
from torch.utils.data import DataLoader


def lj_energy_forces(
    positions: torch.Tensor,
    *,
    epsilon_eV: float | torch.Tensor,
    sigma_A: float | torch.Tensor,
    cutoff_A: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an unshifted 12-6 LJ potential for one structure.

    Each pair contributes once. The hard cutoff is safe for the generated
    records because every included pair remains well inside ``cutoff_A``.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cutoff_A <= 0:
        raise ValueError("cutoff_A must be positive")

    source, target = torch.triu_indices(
        positions.shape[0],
        positions.shape[0],
        offset=1,
        device=positions.device,
    )
    displacement = positions[target] - positions[source]
    squared_distance = displacement.square().sum(dim=1)
    inside = squared_distance < cutoff_A**2
    if not bool(inside.any()):
        return (
            positions.new_zeros((1, 1)),
            torch.zeros_like(positions),
        )

    source = source[inside]
    target = target[inside]
    displacement = displacement[inside]
    squared_distance = squared_distance[inside]
    epsilon = torch.as_tensor(
        epsilon_eV,
        dtype=positions.dtype,
        device=positions.device,
    )
    sigma = torch.as_tensor(
        sigma_A,
        dtype=positions.dtype,
        device=positions.device,
    )

    sigma_over_r_squared = sigma.square() / squared_distance
    sigma_over_r_six = sigma_over_r_squared.pow(3)
    sigma_over_r_twelve = sigma_over_r_six.square()
    pair_energy = 4.0 * epsilon * (sigma_over_r_twelve - sigma_over_r_six)

    # With displacement r_j-r_i, this is the force on atom i.
    force_scale = (
        24.0
        * epsilon
        * (sigma_over_r_six - 2.0 * sigma_over_r_twelve)
        / squared_distance
    )
    pair_force = force_scale.unsqueeze(1) * displacement
    forces = torch.zeros_like(positions)
    forces = forces.index_add(0, source, pair_force)
    forces = forces.index_add(0, target, -pair_force)
    return pair_energy.sum().reshape(1, 1), forces


def _regular_tetrahedron(
    pair_distance_A: torch.Tensor,
) -> torch.Tensor:
    vertices = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=pair_distance_A.dtype,
        device=pair_distance_A.device,
    )
    return vertices * (pair_distance_A / (2.0 * math.sqrt(2.0)))


def generate_argon_records(
    *,
    count: int,
    seed: int,
    epsilon_eV: float,
    sigma_A: float,
    cutoff_A: float,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> list[AtomicData]:
    """Generate distorted isolated Ar4 structures and exact LJ labels."""

    if count < 1:
        raise ValueError("count must be positive")
    if epsilon_eV <= 0 or sigma_A <= 0:
        raise ValueError("epsilon_eV and sigma_A must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    equilibrium_A = (2.0 ** (1.0 / 6.0)) * sigma_A
    records: list[AtomicData] = []

    for sample_id in range(count):
        stretch = 0.94 + 0.18 * torch.rand(
            (),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        positions = _regular_tetrahedron(stretch * equilibrium_A)
        distortion = 0.07 * torch.randn(
            4,
            3,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        positions = positions + distortion
        positions = positions - positions.mean(dim=0, keepdim=True)
        energy, forces = lj_energy_forces(
            positions,
            epsilon_eV=epsilon_eV,
            sigma_A=sigma_A,
            cutoff_A=cutoff_A,
        )
        record = AtomicData(
            positions=positions,
            atomic_numbers=torch.full(
                (4,),
                18,
                dtype=torch.long,
                device=device,
            ),
            energy=energy,
            forces=forces,
        )
        record.add_system_property(
            "sample_id",
            torch.tensor(
                [[sample_id]],
                dtype=torch.long,
                device=device,
            ),
        )
        records.append(record)
    return records


def split_argon_records(
    records: Sequence[AtomicData],
    *,
    validation_count: int,
    seed: int,
) -> tuple[list[AtomicData], list[AtomicData], pd.DataFrame]:
    """Create a deterministic held-out split while preserving source order."""

    if not 0 < validation_count < len(records):
        raise ValueError("validation_count must be between zero and dataset size")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    validation_indices = set(permutation[:validation_count])
    training = [
        record
        for index, record in enumerate(records)
        if index not in validation_indices
    ]
    validation = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    rows = [
        {
            "sample_id": int(record.sample_id.item()),
            "split": "validation" if index in validation_indices else "train",
            "atoms": record.num_nodes,
            "energy (eV)": float(record.energy.detach().cpu().item()),
            "force RMS (eV/Å)": float(
                record.forces.detach().cpu().square().mean().sqrt().item()
            ),
        }
        for index, record in enumerate(records)
    ]
    return training, validation, pd.DataFrame(rows)


def collate_records(records: Sequence[AtomicData]) -> Batch:
    """Pack records into one graph-aware Toolkit batch."""

    return Batch.from_data_list(list(records))


def make_loader(
    records: Sequence[AtomicData],
    *,
    batch_size: int,
) -> DataLoader:
    """Build a deterministic, single-process loader."""

    return DataLoader(
        list(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_records,
    )


def reset_checkpoint_directory(path: str | Path) -> Path:
    """Recreate only the notebook-local ``artifacts/checkpoints`` directory."""

    target = Path(path)
    if target.name != "checkpoints" or target.parent.name != "artifacts":
        raise ValueError("checkpoint path must end in artifacts/checkpoints")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


class _ToyDataset:
    """Small fixed-four-atom regression dataset."""

    def __init__(
        self,
        *,
        count: int,
        seed: int,
        target_shift: float,
        device: torch.device,
    ) -> None:
        self.records: list[AtomicData] = []
        generator = torch.Generator(device=device).manual_seed(seed)
        for sample_id in range(count):
            positions = torch.randn(
                4,
                3,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            score = (
                0.45 * positions.square().sum()
                + 0.15 * positions[:, 0].sum()
                + target_shift
            ).reshape(1, 1)
            record = AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(
                    4,
                    dtype=torch.long,
                    device=device,
                ),
                energy=score,
            )
            record.add_system_property(
                "sample_id",
                torch.tensor(
                    [[sample_id]],
                    dtype=torch.long,
                    device=device,
                ),
            )
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AtomicData:
        return self.records[index]


def toy_records(
    *,
    count: int,
    seed: int,
    target_shift: float,
    device: torch.device,
) -> list[AtomicData]:
    """Return the records from a deterministic toy dataset."""

    return _ToyDataset(
        count=count,
        seed=seed,
        target_shift=target_shift,
        device=device,
    ).records
