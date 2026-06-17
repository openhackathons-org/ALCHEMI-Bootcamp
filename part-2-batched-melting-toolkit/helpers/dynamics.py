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
    """LoggingHook custom_scalar: instantaneous scalar pressure per graph (eV/A^3).

    P = 2*KE/(3V) - Tr(stress)/3, matching the toolkit's NPT kernel chain:
    ``nvalchemi/dynamics/integrators/npt.py`` does ``virial = -batch.stress * V``
    (its own comment: "batch.stress is tensile-positive Cauchy stress -W/V"),
    then ``nvalchemiops`` computes ``P_tensor = (KE_tensor + virial) / V`` and
    ``compute_scalar_pressure`` takes ``Tr(P_tensor)/3``. Substituting gives
    ``P = (2*KE - V*Tr(stress)) / (3V)``. The minus on the stress term is real.
    """
    batch = ctx.batch
    stress_trace = batch.stress.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    V = torch.linalg.det(batch.cell).abs().view(-1)
    ke_per_atom = 0.5 * batch.atomic_masses * (batch.velocities**2).sum(dim=-1)
    ke_per_graph = torch.zeros(batch.num_graphs, device=V.device, dtype=V.dtype)
    ke_per_graph.scatter_add_(0, batch.batch_idx, ke_per_atom.to(V.dtype))
    return (2.0 / 3.0) * ke_per_graph / V - stress_trace


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


def compute_density(batch):
    """Per-graph densities in g/cm^3. Returns ``list[float]`` of length ``num_graphs``."""
    vol = torch.linalg.det(batch.cell).abs()
    mass = torch.zeros(batch.num_graphs, device=vol.device)
    mass.scatter_add_(0, batch.batch_idx, batch.atomic_masses)
    return (mass * AMU_OVER_A3_TO_G_CM3 / vol).tolist()
