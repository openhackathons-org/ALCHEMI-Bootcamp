"""Helpers that operate on a dynamics ``ctx`` or a :class:`Batch`."""

import torch
from ase import Atoms

from .constants import AMU_OVER_A3_TO_G_CM3


def batch_to_ase(batch):
    """Convert a single-graph Batch to an ASE Atoms object (CPU, PBC=True)."""
    return Atoms(
        numbers=batch.atomic_numbers.cpu().numpy(),
        positions=batch.positions.detach().cpu().numpy(),
        cell=batch.cell.squeeze().detach().cpu().numpy(),
        pbc=True,
    )


def pressure_scalar(ctx):
    """LoggingHook custom_scalar: mean diagonal pressure per graph (eV/A^3)."""
    stress = ctx.batch.stress  # [B, 3, 3]
    return -stress.diagonal(dim1=-2, dim2=-1).mean(dim=-1)


def volume_scalar(ctx):
    """LoggingHook custom_scalar: cell volume per graph (A^3)."""
    return torch.linalg.det(ctx.batch.cell).abs().view(-1)


def density_scalar(ctx):
    """LoggingHook custom_scalar: density per graph (g/cm^3)."""
    vol = torch.linalg.det(ctx.batch.cell).abs().view(-1)
    mass_per_graph = torch.zeros(ctx.batch.num_graphs, device=vol.device)
    mass_per_graph.scatter_add_(0, ctx.batch.batch_idx, ctx.batch.atomic_masses)
    return mass_per_graph * AMU_OVER_A3_TO_G_CM3 / vol


DYNAMICS_SCALARS = {
    "pressure_eV_A3": pressure_scalar,
    "volume_A3": volume_scalar,
    "density_g_cm3": density_scalar,
}


def compute_density_per_graph(batch):
    """Per-graph densities in g/cm^3. Returns ``list[float]`` of length ``num_graphs``."""
    vol = torch.linalg.det(batch.cell).abs()
    mass = torch.zeros(batch.num_graphs, device=vol.device)
    mass.scatter_add_(0, batch.batch_idx, batch.atomic_masses)
    return (mass * AMU_OVER_A3_TO_G_CM3 / vol).tolist()


def compute_density(batch):
    """Scalar density (g/cm^3) for a single-graph Batch."""
    vol = torch.linalg.det(batch.cell).abs().item()
    return batch.atomic_masses.sum().item() * AMU_OVER_A3_TO_G_CM3 / vol
