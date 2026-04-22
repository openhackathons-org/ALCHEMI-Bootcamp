"""Matplotlib plotting for dynamics structures and diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import STAGE_COLORS, STAGE_LABELS
from .io import load_zarr_frames, zarr_trajectory_length


def _scatter_view(ax, positions, cell, proj=(0, 1), label=""):
    """Scatter-plot projection of atoms with convex-hull cell outline.

    Each atom is a marker; the third (out-of-plane) coordinate drives the
    marker colour via the viridis colormap, giving a depth cue without
    obscuring low-density regions. Fast for ~10^4 atoms -- no binning, and
    every atom is rendered individually so sparse regions stay legible.
    """
    i, j = proj
    k = ({0, 1, 2} - {i, j}).pop()  # the out-of-plane axis
    axis_labels = ["x", "y", "z"]
    pos = (
        positions
        if isinstance(positions, np.ndarray)
        else positions.detach().cpu().numpy()
    )
    c = cell if isinstance(cell, np.ndarray) else cell.detach().cpu().numpy()
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


def visualize_structure(batch, title="", save_path=None):
    """Three-panel (xy, xz, yz) hexbin view of a single-graph Batch."""
    pos = batch.positions.detach().cpu().numpy()
    cell = batch.cell.squeeze().detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, proj, label in zip(axes, [(0, 1), (0, 2), (1, 2)], ["xy", "xz", "yz"]):
        _scatter_view(ax, pos, cell, proj=proj, label=label)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_trajectory_frames(source, title="", n_frames=4, save_path=None, indices=None):
    """Plot evenly-spaced xy-projection frames.

    ``source`` is either a Zarr path or a pre-loaded iterable of single-graph
    Batch objects. When given a path, only the ``n_frames`` actually plotted
    are decoded (via :func:`load_zarr_frames`) -- loading 100+ frames just
    to render 4 used to dominate this function's runtime.

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
