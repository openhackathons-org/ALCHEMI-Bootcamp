"""Matplotlib plotting for dynamics structures and diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from .constants import STAGE_COLORS, STAGE_LABELS
from .io import load_zarr_frames, zarr_trajectory_length

# Standard CPK-ish render palette: dark for heavies, light for H so the
# H halo doesn't dominate. New atomic numbers can be added as needed.
_ATOM_COLORS = {1: "#cfd8dc", 6: "#37474f", 7: "#1e88e5", 8: "#e53935"}
_DEFAULT_ATOM_COLOR = "#9e9e9e"
_ATOM_SIZES = {1: 22.0, 6: 55.0, 7: 60.0, 8: 60.0}
_DEFAULT_ATOM_SIZE = 40.0


def _scatter_view(ax, pos, c, proj=(0, 1), label=""):
    """Scatter-plot projection of atoms with convex-hull cell outline.

    Each atom is a marker; the third (out-of-plane) coordinate drives the
    marker colour via the viridis colormap, giving a depth cue without
    obscuring low-density regions. Fast for ~10^4 atoms -- no binning, and
    every atom is rendered individually so sparse regions stay legible.
    """
    i, j = proj
    k = ({0, 1, 2} - {i, j}).pop()  # the out-of-plane axis
    axis_labels = ["x", "y", "z"]
    ax.scatter(
        pos[:, i],
        pos[:, j],
        s=6,
        c=pos[:, k],
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    o = np.zeros(3)
    va, vb, vc = c[0], c[1], c[2]
    verts_3d = [o, va, vb, vc, va + vb, va + vc, vb + vc, va + vb + vc]
    pts = np.array([[v[i], v[j]] for v in verts_3d])
    unique = [pts[0]]
    for p in pts[1:]:
        if not any(np.allclose(p, u, atol=0.01) for u in unique):
            unique.append(p)
    pts = np.array(unique)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    hull = pts[order]
    hull = np.vstack([hull, hull[0]])
    ax.plot(hull[:, 0], hull[:, 1], "k-", lw=0.6)
    ax.set_xlabel(f"{axis_labels[i]} (A)")
    ax.set_ylabel(f"{axis_labels[j]} (A)")
    ax.set_aspect("equal")
    if label:
        ax.set_title(label)


def _find_molecules(atoms: Atoms, cutoff: float = 1.55) -> list[np.ndarray]:
    """Connected components under PBC at a covalent-range cutoff. Same
    union-find approach as ``analyze_s0.find_molecules`` and the inline
    ``_detect_naphthalene_molecules`` in the SLC stack cell — kept here so
    the renderer has no script dependency."""
    i, j = primitive_neighbor_list(
        "ij",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=cutoff,
    )
    n = len(atoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i, j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return [np.asarray(sorted(g), dtype=np.int64) for g in groups.values()]


def _unwrap_per_molecule(positions: np.ndarray, cell: np.ndarray, molecules) -> np.ndarray:
    """MIC unwrap each molecule's atoms relative to its first atom, so a
    molecule wrapped across a PBC face renders contiguously."""
    out = positions.copy()
    cell_inv = np.linalg.inv(cell)
    for idx in molecules:
        ref = out[idx[0]]
        dr = out[idx] - ref
        dr_frac = dr @ cell_inv
        dr_frac -= np.round(dr_frac)
        out[idx] = ref + dr_frac @ cell
    return out


def _intramol_bond_pairs(
    positions: np.ndarray, molecules, cutoff: float = 1.65
) -> np.ndarray:
    """All bonds within each molecule (positions assumed already unwrapped),
    returned as `(i, j)` index pairs with i < j."""
    bonds = []
    for idx in molecules:
        d = np.linalg.norm(
            positions[idx][:, None] - positions[idx][None, :], axis=-1
        )
        ai, aj = np.where((d > 0) & (d < cutoff))
        for a, b in zip(ai, aj):
            if a < b:
                bonds.append((int(idx[a]), int(idx[b])))
    return (
        np.asarray(bonds, dtype=np.int64) if bonds else np.zeros((0, 2), dtype=np.int64)
    )


def _batch_to_atoms(batch) -> Atoms:
    """Single-graph Batch → ASE Atoms, mirroring the toolkit's
    ``data_to_atoms`` pattern (basic/03_ase_integration.py)."""
    return Atoms(
        numbers=batch.atomic_numbers.detach().cpu().numpy(),
        positions=batch.positions.detach().cpu().numpy(),
        cell=batch.cell.squeeze(0).detach().cpu().numpy(),
        pbc=batch.pbc.squeeze(0).detach().cpu().numpy(),
    )


def visualize_structure(structure, title="", save_path=None, ax=None, bond_cutoff=1.65):
    """One-frame matplotlib render of a molecular system, looking down the
    +a crystal axis (b vertical, c horizontal — same projection convention
    as ``render_warmup_s0.py``).

    Atoms are coloured by element (C dark grey, H light grey, N blue, O
    red — extend ``_ATOM_COLORS``/``_ATOM_SIZES`` for new species) and
    intramolecular bonds are drawn as line segments. PBC-aware
    connectivity discovers molecules and unwraps each one before render
    so none span a periodic image.

    Accepts either a single-graph nvalchemi ``Batch`` or an ASE ``Atoms``.
    """
    atoms = structure if isinstance(structure, Atoms) else _batch_to_atoms(structure)
    pos = atoms.positions
    cell = np.asarray(atoms.cell)
    Z = np.asarray(atoms.get_atomic_numbers())

    molecules = _find_molecules(atoms)
    pos_uw = _unwrap_per_molecule(pos, cell, molecules)
    bonds = _intramol_bond_pairs(pos_uw, molecules, cutoff=bond_cutoff)

    colors = [_ATOM_COLORS.get(int(z), _DEFAULT_ATOM_COLOR) for z in Z]
    sizes = np.asarray([_ATOM_SIZES.get(int(z), _DEFAULT_ATOM_SIZE) for z in Z])

    # Project to the (b, c) face: c on x-axis, b on y-axis (a drops out).
    # Cell is row-major in ASE (rows = lattice vectors), so cell[1] = b vec
    # and cell[2] = c vec. For an orthorhombic-ish view we plot the y/z
    # components, which match the projected axis lengths along b and c.
    x_screen = pos_uw[:, 2]
    y_screen = pos_uw[:, 1]
    b_y = float(np.linalg.norm(cell[1]))
    c_z = float(np.linalg.norm(cell[2]))
    pad = 2.0

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.set_aspect("equal")
    ax.set_xlim(-pad, c_z + pad)
    ax.set_ylim(-pad, b_y + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"c-axis projection ({c_z:.1f} Å)")
    ax.set_ylabel(f"b-axis ({b_y:.1f} Å)")
    ax.add_patch(
        Rectangle((0, 0), c_z, b_y, fill=False, edgecolor="0.3", linewidth=1.0)
    )

    if len(bonds):
        segs = np.stack(
            [
                np.column_stack([x_screen[bonds[:, 0]], y_screen[bonds[:, 0]]]),
                np.column_stack([x_screen[bonds[:, 1]], y_screen[bonds[:, 1]]]),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(segs, colors="0.4", linewidths=0.8, alpha=0.85, zorder=2)
        )

    ax.scatter(
        x_screen,
        y_screen,
        c=colors,
        s=sizes,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )

    if title:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.show()


def plot_trajectory_frames(source, title="", n_frames=4, save_path=None, indices=None):
    """Plot evenly-spaced xy-projection frames.

    ``source`` is either a Zarr path or a pre-loaded iterable of single-graph
    Batch objects. When given a path, only the ``n_frames`` actually plotted
    are decoded (via :func:`load_zarr_frames`).

    Pass ``indices`` to override the default evenly-spaced selection with an
    explicit list of frame indices.
    """
    if isinstance(source, (str, Path)):
        n_total = zarr_trajectory_length(source)
        if n_total == 0:
            print("No frames in trajectory")
            return
        if indices is None:
            indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
        frames = load_zarr_frames(source, indices)
    else:
        frames = list(source)
        n_total = len(frames)
        if n_total == 0:
            print("No frames in trajectory")
            return
        if indices is None:
            indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
        frames = [frames[i] for i in indices]
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx, b in zip(axes, indices, frames):
        pos = b.positions.detach().cpu().numpy()
        cell = b.cell.squeeze().detach().cpu().numpy()
        _scatter_view(ax, pos, cell, proj=(0, 1), label=f"Frame {idx}/{n_total}")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def shade_stages(ax, steps, status):
    """Shade FIRE/NVT/NPT regions by contiguous status runs."""
    if len(steps) == 0:
        return
    status = np.asarray(status)
    changes = np.where(np.diff(status) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(status)]))
    for s, e in zip(starts, ends):
        st = int(status[s])
        if st not in STAGE_COLORS:
            continue
        ax.axvspan(
            steps[s],
            steps[e - 1],
            alpha=0.35,
            color=STAGE_COLORS[st],
            label=STAGE_LABELS[st],
            zorder=0,
        )


def dedup_legend(ax, **kwargs):
    """Render ``ax``'s legend with duplicate labels collapsed to one entry."""
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    ax.legend(seen.values(), seen.keys(), **kwargs)
