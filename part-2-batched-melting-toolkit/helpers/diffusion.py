# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numpy-only translational-diffusion helpers (host-side analysis scripts).

A torch counterpart of ``compute_com_msd`` already exists in ``analysis.py``
for container/notebook workflows. This module mirrors its math in numpy so
host-side tooling (e.g. ``analyze_s0.py`` running in the alchemi-playbook
conda env) can compute COM-MSD and fit an Einstein-relation diffusion
coefficient without pulling in torch or nvalchemi.
"""

import numpy as np


def compute_com_msd_numpy(
    positions_seq,
    cells_seq,
    masses,
    atoms_per_mol,
    ref_cell=None,
    subtract_system_com=True,
):
    """Molecular-COM MSD with affine cell deformation removed (numpy port).

    Mirrors :func:`helpers.analysis.compute_com_msd`. Inputs:

    - ``positions_seq``: iterable of per-frame ``[N, 3]`` numpy arrays. Atoms
      MUST be ordered as contiguous ``atoms_per_mol``-sized blocks, one block
      per molecule.
    - ``cells_seq``: iterable of per-frame ``[3, 3]`` cell matrices (numpy).
    - ``masses``: ``[N]`` numpy array of atomic masses (same atom order).
    - ``atoms_per_mol``: int.
    - ``ref_cell``: ``[3, 3]`` numpy reference cell used to map accumulated
      fractional displacements back to A. Default = mean cell across the
      window.
    - ``subtract_system_com``: at each step, subtract the mass-weighted mean
      of the (MIC-corrected) per-molecule fractional displacement from every
      molecule's df before accumulating. Default ``True``. Removes rigid
      system translation -- including Langevin / thermostat-induced COM
      drift, which for a small periodic system accumulates over hundreds of
      frames and otherwise inflates per-molecule MSD by ~1-2 orders of
      magnitude even when every molecule is individually pinned to its
      lattice site. Affine cell deformation is removed regardless (each
      step's df uses that step's cell).

    Each frame's molecules are PBC-unwrapped relative to their first atom
    before forming a mass-weighted COM, removing intramolecular vibration
    and rotation from the signal. The fractional-coord MSD of those COMs is
    then accumulated using each frame's own cell, so an atom (or molecule)
    pinned to a fixed fractional site contributes zero MSD even when the
    cell is drifting under NPT.

    The drift subtraction is done on the MIC-corrected per-step df rather
    than on the per-frame COMs because (a) under PBC the lab-frame COM of
    wrapped raw positions is near-invariant under uniform rigid translation
    -- atoms wrapping in and out of the cell cancel -- and so subtracting
    it cannot see the drift it is meant to remove; (b) subtracting an
    unwrapped per-mol-averaged COM in lab coordinates is fooled by single
    atom-0 wraps that inject cell-vector jumps into the average. Removing
    drift on the MIC-corrected df sidesteps both: per-molecule wraps are
    already cancelled by MIC round, and the residual is the true uniform
    fractional drift.

    Returns ``[n_frames-1, n_mol]`` numpy array; element ``[t, m]`` is the
    cumulative squared displacement of molecule ``m`` between frame 0 and
    frame ``t+1``.
    """
    positions_seq = list(positions_seq)
    cells_seq = list(cells_seq)
    masses = np.asarray(masses, dtype=np.float64)
    if ref_cell is None:
        ref_cell = np.stack(cells_seq).mean(axis=0)
    else:
        ref_cell = np.asarray(ref_cell, dtype=np.float64)

    n_atoms = positions_seq[0].shape[0]
    n_mol = n_atoms // atoms_per_mol

    com_series = []
    mol_mass = masses.reshape(n_mol, atoms_per_mol)
    mol_total_mass = mol_mass.sum(axis=-1)
    system_total_mass = mol_total_mass.sum()
    for snap, cell in zip(positions_seq, cells_seq):
        cell_arr = np.asarray(cell, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_arr)
        snap_arr = np.asarray(snap, dtype=np.float64)
        mol_pos = snap_arr.reshape(n_mol, atoms_per_mol, 3)
        ref = mol_pos[:, 0:1, :]
        dr_frac = (mol_pos - ref) @ cell_inv
        dr_frac -= np.round(dr_frac)
        mol_unwrapped = ref + dr_frac @ cell_arr
        com = (mol_mass[..., None] * mol_unwrapped).sum(axis=1) / mol_total_mass[
            :, None
        ]
        com_series.append(com)

    cumulative_df = np.zeros((n_mol, 3), dtype=np.float64)
    msd_per_frame = []
    s_prev = com_series[0] @ np.linalg.inv(np.asarray(cells_seq[0], dtype=np.float64))
    for i in range(1, len(com_series)):
        s_i = com_series[i] @ np.linalg.inv(np.asarray(cells_seq[i], dtype=np.float64))
        df = s_i - s_prev
        df -= np.round(df)
        if subtract_system_com:
            mean_df = (mol_total_mass[:, None] * df).sum(axis=0) / system_total_mass
            df = df - mean_df
        cumulative_df += df
        disp_cart = cumulative_df @ ref_cell
        msd_per_frame.append((disp_cart**2).sum(axis=-1))
        s_prev = s_i
    if not msd_per_frame:
        return np.zeros((1, n_mol), dtype=np.float64)
    return np.stack(msd_per_frame)


def fit_diffusion_coefficient(msd_per_mol, time_ps, fit_frac=0.5):
    """Einstein-relation D from the long-time slope of the mean COM MSD.

    Averages ``msd_per_mol`` (shape ``[n_frames, n_mol]`` from
    :func:`compute_com_msd_numpy` -- or any equivalent producer) across
    molecules, fits a line over the trailing ``fit_frac`` of the curve to
    skip the ballistic / sub-diffusive head, and returns ``D = slope/6``
    from the 3D Einstein relation ``MSD = 6 D t``. Unit conversion:
    ``1 A^2/ps = 1e-4 cm^2/s``.

    Accepts either a 2-D ``[n_frames, n_mol]`` array (averaged here) or a
    pre-averaged 1-D ``[n_frames]`` curve. Returns a dict with
    ``D_A2_per_ps``, ``D_cm2_per_s``, ``slope``, ``intercept``,
    ``fit_start_idx`` and the cross-molecule averaged ``msd_mean`` curve.
    """
    msd_arr = np.asarray(msd_per_mol, dtype=np.float64)
    msd_mean = msd_arr.mean(axis=-1) if msd_arr.ndim == 2 else msd_arr
    time_ps = np.asarray(time_ps, dtype=np.float64)
    if msd_mean.shape[0] != time_ps.shape[0]:
        raise ValueError(
            f"msd_per_mol has {msd_mean.shape[0]} frames but time_ps has "
            f"{time_ps.shape[0]}; expected equal length"
        )
    n_fit = max(2, int(msd_mean.shape[0] * fit_frac))
    fit_start_idx = msd_mean.shape[0] - n_fit
    slope, intercept = np.polyfit(time_ps[fit_start_idx:], msd_mean[fit_start_idx:], 1)
    d_A2_per_ps = float(slope) / 6.0
    return {
        "D_A2_per_ps": d_A2_per_ps,
        "D_cm2_per_s": d_A2_per_ps * 1e-4,
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_start_idx": int(fit_start_idx),
        "msd_mean": msd_mean,
    }
