"""Trajectory analysis: MSD, RDF, rotational ACF / S0.

All functions are pure (no module-level state) and run on the device of the
input tensors. MSD variants use affine-cell-deformation removal so they are
NPT-safe.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


def compute_msd(snapshots, cells, n_atoms_total, ref_cell=None):
    """MSD with affine cell deformation removed.

    Each position is converted to fractional (scaled) coordinates using its
    own cell, so an atom pinned to a fixed fractional site contributes zero
    to MSD even when the cell is drifting under NPT. The accumulated
    fractional displacement is mapped back to A via ``ref_cell`` (default:
    mean cell across the window) to give a drift-neutral A^2 metric.
    """
    if ref_cell is None:
        ref_cell = torch.stack(list(cells)).mean(dim=0)
    cumulative_df = torch.zeros(n_atoms_total, 3, device=snapshots[0].device)
    msd_per_frame = []
    s_prev = snapshots[0] @ torch.linalg.inv(cells[0])
    for i in tqdm(range(1, len(snapshots)), desc="MSD frames", leave=False):
        s_i = snapshots[i] @ torch.linalg.inv(cells[i])
        df = s_i - s_prev
        df -= torch.round(df)
        cumulative_df += df
        disp_cart = cumulative_df @ ref_cell
        msd_per_frame.append((disp_cart**2).sum(dim=-1).cpu())
        s_prev = s_i
    return (
        torch.stack(msd_per_frame) if msd_per_frame else torch.zeros(1, n_atoms_total)
    )


def compute_com_msd(
    snapshots, cells, masses, atoms_per_mol, ref_cell=None, subtract_system_com=True
):
    """Molecular-COM MSD with affine cell deformation removed.

    Each frame's atoms are collapsed to per-molecule mass-weighted COMs
    (PBC-unwrapped relative to the first atom of each molecule). The
    fractional-coord MSD of those COMs is then accumulated. Removes
    intramolecular vibration (C-H stretches, torsions) and molecular
    rotation (atoms moving around a pinned COM) from the signal so only
    inter-molecular translational drift contributes.

    ``subtract_system_com=True`` (default) additionally subtracts the
    per-frame mass-weighted system COM from each molecular COM before
    accumulating, removing whole-system rigid translation in the lab
    frame. Under anisotropic NPT this lab-COM motion couples to the cell
    expansion and inflates per-molecule MSD even when molecules are
    individually pinned to their lattice sites. Affine cell deformation
    alone is handled regardless (fractional pinning => zero MSD).

    For a stable crystal this plateaus within ~few hundred fs at
    ~0.01-0.1 A^2; a linearly growing COM MSD is the signature of genuine
    translational diffusion. The physically-meaningful D from the 3D
    Einstein relation MSD = 6 D t is fit from this curve.
    """
    if ref_cell is None:
        ref_cell = torch.stack(list(cells)).mean(dim=0)
    n_atoms = snapshots[0].shape[0]
    n_mol = n_atoms // atoms_per_mol
    mol_mass = masses.view(n_mol, atoms_per_mol)
    mol_total_mass = mol_mass.sum(dim=-1)
    system_total_mass = mol_total_mass.sum()

    com_series = []
    for snap, cell in zip(snapshots, cells):
        cell_inv = torch.linalg.inv(cell)
        mol_pos = snap.view(n_mol, atoms_per_mol, 3)
        ref = mol_pos[:, 0:1, :]
        dr_frac = (mol_pos - ref) @ cell_inv
        dr_frac = dr_frac - torch.round(dr_frac)
        mol_unwrapped = ref + dr_frac @ cell
        com = (mol_mass.unsqueeze(-1) * mol_unwrapped).sum(
            dim=1
        ) / mol_total_mass.unsqueeze(-1)
        if subtract_system_com:
            # Lab-frame system COM (from raw atom positions). Using the sum of
            # per-mol unwrapped COMs would inherit per-molecule atom-0 PBC
            # wraps and corrupt every molecule's MSD via the subtraction.
            sys_com_lab = (masses.unsqueeze(-1) * snap).sum(dim=0) / system_total_mass
            com = com - sys_com_lab
        com_series.append(com)

    cumulative_df = torch.zeros(n_mol, 3, device=snapshots[0].device)
    msd_per_frame = []
    s_prev = com_series[0] @ torch.linalg.inv(cells[0])
    for i in range(1, len(com_series)):
        s_i = com_series[i] @ torch.linalg.inv(cells[i])
        df = s_i - s_prev
        df -= torch.round(df)
        cumulative_df += df
        disp_cart = cumulative_df @ ref_cell
        msd_per_frame.append((disp_cart**2).sum(dim=-1).cpu())
        s_prev = s_i
    return torch.stack(msd_per_frame) if msd_per_frame else torch.zeros(1, n_mol)


def compute_rdf(positions, cell, n_bins=200, r_max=10.0, chunk_size=500):
    """RDF with chunked pair computation to avoid O(N^2) memory.

    MIC wrapping is valid only when ``r_max <= min(|cell_vector|) / 2``; a
    larger ``r_max`` silently folds pair distances back on themselves and
    produces spurious g(r) tails. The guard raises before that happens.
    """
    half_min_cell = 0.5 * cell.norm(dim=-1).min().item()
    if r_max > half_min_cell:
        raise ValueError(
            f"r_max={r_max:.2f} A exceeds half the shortest cell vector "
            f"({half_min_cell:.2f} A); MIC would wrap pair distances. "
            "Reduce r_max or use a larger supercell."
        )
    n_atoms = positions.shape[0]
    vol = torch.linalg.det(cell).abs().item()
    rho = n_atoms / vol
    dr = r_max / n_bins
    r_centers = torch.linspace(dr / 2, r_max - dr / 2, n_bins, device=positions.device)
    cell_inv = torch.linalg.inv(cell)
    hist = torch.zeros(n_bins, device=positions.device)

    for start in tqdm(range(0, n_atoms, chunk_size), desc="RDF chunks", leave=False):
        end = min(start + chunk_size, n_atoms)
        chunk = positions[start:end]  # [C, 3]
        diff = chunk.unsqueeze(1) - positions.unsqueeze(0)  # [C, N, 3]
        diff_frac = diff @ cell_inv
        diff_frac -= torch.round(diff_frac)
        diff_cart = diff_frac @ cell
        dists = diff_cart.norm(dim=-1)  # [C, N]
        chunk_idx = torch.arange(end - start, device=positions.device)
        dists[chunk_idx, chunk_idx + start] = 0.0
        dists_flat = dists.reshape(-1)
        valid = dists_flat > 1e-6
        hist += torch.histc(dists_flat[valid], bins=n_bins, min=0.0, max=r_max)

    shell_vol = 4 * np.pi * r_centers**2 * dr
    g_r = hist / (shell_vol * rho * n_atoms)
    return r_centers.cpu().numpy(), g_r.cpu().numpy()


def _mol_inertia_eigvecs(positions, masses, atoms_per_mol, cell):
    """Per-molecule mass-weighted inertia-tensor eigenvectors.

    Unwraps each molecule via MIC relative to its first atom, builds the
    mass-weighted inertia tensor ``I = sum_i m_i (|r_i|^2 I_3 - r_i r_i^T)``
    around the molecular COM, and returns the eigenvectors sorted by
    ascending eigenvalue. Shape ``[n_mol, 3, 3]`` with ``out[m, k, :]``
    = axis k for molecule m.

    For planar elongated molecules (e.g. naphthalene) the ascending order is:
    k=0 long in-plane axis (smallest I), k=1 short in-plane axis (middle I),
    k=2 normal to molecular plane (largest I).

    Eigenvectors have ``+-`` sign ambiguity; downstream uses must be P_2
    (l=2 Legendre) or other sign-invariant forms.
    """
    n_atoms = positions.shape[0]
    assert n_atoms % atoms_per_mol == 0, (
        f"Atom count {n_atoms} not divisible by {atoms_per_mol}"
    )
    n_mol = n_atoms // atoms_per_mol
    cell_inv = torch.linalg.inv(cell)
    eye3 = torch.eye(3, device=positions.device, dtype=positions.dtype)
    axes = torch.zeros(n_mol, 3, 3, device=positions.device, dtype=positions.dtype)
    for m in range(n_mol):
        sl = slice(m * atoms_per_mol, (m + 1) * atoms_per_mol)
        mol_pos = positions[sl]
        mol_mass = masses[sl].unsqueeze(-1)  # [A, 1]
        ref = mol_pos[0]
        dr = mol_pos - ref
        dr_frac = dr @ cell_inv
        dr_frac -= torch.round(dr_frac)
        mol_unwrapped = ref + dr_frac @ cell
        com = (mol_mass * mol_unwrapped).sum(dim=0) / mol_mass.sum()
        centered = mol_unwrapped - com
        r2 = (centered**2).sum(dim=-1, keepdim=True)
        inertia = (mol_mass * r2).sum() * eye3 - (centered * mol_mass).T @ centered
        _, eigvecs = torch.linalg.eigh(inertia)
        axes[m] = eigvecs.T
    return axes


def compute_molecule_axes(positions, masses, atoms_per_mol, cell):
    """Return ``[n_mol, 3, 3]`` orthonormal axes per molecule, sorted by inertia.

    Vectorised (no Python loop over molecules) version of
    :func:`_mol_inertia_eigvecs` for use inside the rACF pipeline.
    """
    n_mol = positions.shape[0] // atoms_per_mol
    masses = masses.to(positions.device)
    cell = cell.to(positions.device)
    cell_inv = torch.linalg.inv(cell)
    pos = positions.view(n_mol, atoms_per_mol, 3)
    mass = masses.view(n_mol, atoms_per_mol)
    ref = pos[:, 0:1, :]
    dr_frac = (pos - ref) @ cell_inv
    dr_frac = dr_frac - torch.round(dr_frac)
    unwrapped = ref + dr_frac @ cell
    total_mass = mass.sum(dim=-1, keepdim=True)
    com = (mass.unsqueeze(-1) * unwrapped).sum(
        dim=1, keepdim=True
    ) / total_mass.unsqueeze(-1)
    centered = unwrapped - com
    r2 = (centered**2).sum(dim=-1)
    I_trace = (mass * r2).sum(dim=-1)
    weighted = centered * mass.unsqueeze(-1)
    I_outer = torch.einsum("mai,maj->mij", weighted, centered)
    I_tensor = (
        I_trace.view(n_mol, 1, 1) * torch.eye(3, device=positions.device) - I_outer
    )
    _, eigvecs = torch.linalg.eigh(I_tensor)
    return eigvecs.transpose(-2, -1)


def compute_rACF(snapshots, cells, masses, atoms_per_mol, tail_frac=0.2):
    """Rank-2 (P_2) rotational autocorrelation of molecular principal axes.

    For each axis k (long, short, normal) per molecule:

        C_k(lag) = < P_2(v_k(t0) . v_k(t0 + lag)) >_(t0, mol)

    where ``P_2(x) = (3 x^2 - 1) / 2`` is sign-invariant so the arbitrary
    sign of inertia eigenvectors doesn't matter.

    Returns a list of 3 floats (long, short, normal), each the mean of
    ``C_k(lag)`` over the last ``tail_frac`` of lags:

        1.0  = orientation perfectly preserved (crystal)
        0.0  = isotropic free rotation (liquid)

    Paper alignment: Niethammer 2024 / Yoneya-Harada rACF-tail S0.
    """
    axes_series = torch.stack(
        [
            compute_molecule_axes(snap, masses, atoms_per_mol, cell)
            for snap, cell in zip(snapshots, cells)
        ]
    )
    n_frames = axes_series.shape[0]
    if n_frames < 2:
        return [float("nan")] * 3
    n_tail = max(1, int(n_frames * tail_frac))
    tail_lags = range(n_frames - n_tail, n_frames)
    tail_means = []
    for k in range(3):
        v = axes_series[:, :, k, :]
        vals = []
        for lag in tail_lags:
            if lag <= 0 or lag >= n_frames:
                continue
            v0 = v[: n_frames - lag]
            vt = v[lag:]
            dot = (v0 * vt).sum(dim=-1)
            p2 = 0.5 * (3 * dot**2 - 1)
            vals.append(p2.mean().item())
        tail_means.append(sum(vals) / len(vals) if vals else float("nan"))
    return tail_means


def compute_mol_axes(positions, cell, masses, atoms_per_mol):
    """Principal inertia axes per molecule, sorted by eigenvalue ascending.

    Thin wrapper over :func:`_mol_inertia_eigvecs` that preserves the legacy
    argument order ``(positions, cell, masses, atoms_per_mol)`` used by the
    rotational-ACF pipeline in the warmup diagnostics cell.
    """
    return _mol_inertia_eigvecs(positions, masses, atoms_per_mol, cell)


def compute_rotational_acf(axes_per_frame, ref_idx=0):
    """Per-molecule, per-axis second-rank rotational ACF.

    ``C_k(t)[m] = P_2(v_k(t)[m] . v_k(ref)[m])`` where ``P_2(x) = (3 x^2 - 1) / 2``
    makes this sign-flip-invariant (inertia eigenvectors have no fixed sign).
    Paper reference: Yoneya & Harada -- the tail of this ACF is the rotational
    order parameter S0 used to classify plastic crystals. The molecule axis is
    preserved so callers can inspect spatial heterogeneity; take
    ``acf.mean(axis=1)`` for the legacy cross-molecule aggregate.

    Args:
        axes_per_frame: list of ``[n_mol, 3, 3]`` tensors (one per frame,
            from :func:`compute_mol_axes`).
        ref_idx: frame index to use as the rotational reference.

    Returns:
        ``np.ndarray [n_frames, n_mol, 3]`` -- one ACF curve per (molecule, axis).
    """
    ref = axes_per_frame[ref_idx]  # [n_mol, 3, 3]
    n_frames = len(axes_per_frame)
    n_mol = ref.shape[0]
    acf = torch.zeros(n_frames, n_mol, 3, device=ref.device, dtype=ref.dtype)
    for i, axes in enumerate(axes_per_frame):
        dot = (axes * ref).sum(dim=-1)  # [n_mol, 3]
        acf[i] = 0.5 * (3.0 * dot**2 - 1.0)
    return acf.cpu().numpy()


def compute_S0_tail(acf, tail_frac=0.2):
    """Per-molecule rotational order parameter S0 (Yoneya-Harada).

    Phase classification (with diffusion coefficient D):

      * D ~ 0, S0 -> 1        : ordered crystal (no translation, no rotation)
      * D ~ 0, 0 < S0 < 1     : plastic crystal (no translation, hindered rotation)
      * D >> 0, S0 -> 0       : liquid (free translation and rotation)

    Args:
        acf: ``[n_frames, n_mol, 3]`` per-mol rotational ACF from
            :func:`compute_rotational_acf`.
        tail_frac: trailing fraction of frames to average as the "tail".

    Returns:
        ``(s0_per_mol [n_mol], s0_per_mol_per_axis [n_mol, 3])`` -- per-molecule
        scalar (mean over the 3 inertia axes) plus the unaveraged per-axis
        tail values. For the system-wide aggregate scalar take
        ``s0_per_mol.mean()``; for the legacy ``[3]`` per-axis aggregate take
        ``s0_per_mol_per_axis.mean(axis=0)``.
    """
    n_tail = max(1, int(len(acf) * tail_frac))
    tail = acf[-n_tail:].mean(axis=0)  # [n_mol, 3]
    return tail.mean(axis=1), tail


def compute_S0_from_frames(
    frames, atoms_per_mol, ref_idx=0, tail_frac=0.2, atom_slice=None
):
    """Paper-aligned per-molecule rotational S0 for a sequence of Batch frames.

    Chains :func:`compute_mol_axes` -> :func:`compute_rotational_acf` ->
    :func:`compute_S0_tail` so callers don't re-implement the pipeline.
    ``frames`` is any iterable of single-graph Batch objects (as produced
    by :func:`load_zarr_trajectory` or :func:`load_warmup_trajectory`);
    masses are read from the first frame.

    ``atom_slice`` optionally restricts the computation to a contiguous
    subset of atoms (e.g. ``slice(0, n_half)`` for the crystal half of an
    SLC system). The slice length must be a multiple of ``atoms_per_mol``.

    Returns ``(s0_per_mol [n_mol], s0_per_mol_per_axis [n_mol, 3], acf
    [n_frames, n_mol, 3])`` -- the ACF is returned alongside the summary so
    callers that want to plot it (e.g. warmup-diagnostics S0-evolution
    figure) don't need a second pass. For the legacy aggregate scalar take
    ``s0_per_mol.mean()``.
    """
    n = len(frames)
    if n < 2:
        return np.array([]), np.zeros((0, 3)), np.zeros((n, 0, 3))
    masses = frames[0].atomic_masses.cpu()
    if atom_slice is not None:
        masses = masses[atom_slice]
        assert masses.shape[0] % atoms_per_mol == 0, (
            f"atom_slice length {masses.shape[0]} not a multiple of "
            f"atoms_per_mol={atoms_per_mol}"
        )
    axes = []
    for b in frames:
        pos = b.positions if atom_slice is None else b.positions[atom_slice]
        axes.append(compute_mol_axes(pos, b.cell.squeeze(), masses, atoms_per_mol))
    acf = compute_rotational_acf(axes, ref_idx=ref_idx)
    s0_per_mol, s0_per_mol_per_axis = compute_S0_tail(acf, tail_frac=tail_frac)
    return s0_per_mol, s0_per_mol_per_axis, acf


def min_pbc_distance(positions, cell, chunk_size=500):
    """Deterministic chunked PBC min-distance over all atom pairs."""
    n = positions.shape[0]
    cell_inv = torch.linalg.inv(cell)
    global_min = float("inf")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = positions[start:end]
        diff = chunk.unsqueeze(1) - positions.unsqueeze(0)  # [C, N, 3]
        diff_frac = diff @ cell_inv
        diff_frac -= torch.round(diff_frac)
        diff_cart = diff_frac @ cell
        dists = diff_cart.norm(dim=-1)  # [C, N]
        idx = torch.arange(end - start, device=positions.device)
        dists[idx, idx + start] = float("inf")
        chunk_min = dists.min().item()
        if chunk_min < global_min:
            global_min = chunk_min
    return global_min
