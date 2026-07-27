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


def plot_batch_speedup(
    sizes,
    speedups,
    *,
    system_label="systems",
    title="Batching amortizes fixed work",
    suptitle="ALCHEMI Toolkit batched single-point throughput",
    output_path=None,
):
    """Measured batch-speedup vs an ideal-linear reference, NVIDIA dark style.

    Adapted from the Part 3 ``plot_h2o_batch_speedup`` figure (kept pandas-free).
    ``sizes`` / ``speedups`` are equal-length sequences: batch size and the
    speedup relative to running the same systems one at a time. Shows the figure
    inline and optionally writes a PNG.
    """
    nv_green, nv_blue = "#76B900", "#00A3E0"
    dark, light, muted = "#000000", "#F3F5F7", "#A8B0B8"
    sizes = [int(s) for s in sizes]
    speedups = [float(x) for x in speedups]

    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor=dark)
    ax.set_facecolor(dark)
    ax.plot(sizes, speedups, color=nv_green, marker="o", linewidth=3.0, markersize=7, label="measured")
    ax.plot(sizes, sizes, color=muted, linestyle="--", linewidth=1.5, label="ideal linear")
    ax.fill_between(sizes, speedups, color=nv_green, alpha=0.16)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel(f"{system_label} in one Toolkit batch", color=light, fontsize=12)
    ax.set_ylabel("speedup vs one-at-a-time", color=light, fontsize=12)
    ax.set_title(title, color=light, pad=12, fontsize=13)
    ax.tick_params(colors=light, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
    ax.grid(True, color="#2F3A44", linewidth=0.9, alpha=0.75)
    ax.legend(facecolor=dark, edgecolor="#4B5563", labelcolor=light, loc="upper left")

    ax.scatter(
        [sizes[-1]], [speedups[-1]], s=150, facecolors="none",
        edgecolors=nv_blue, linewidths=1.7, zorder=4,
    )
    ax.text(sizes[-1], speedups[-1], f"  {speedups[-1]:.1f}x", color=nv_blue, fontsize=12, va="center")

    fig.suptitle(suptitle, color=light, fontsize=16, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    return output_path


# ---------------------------------------------------------------------------
# Warmup equilibration / thermalization figure (live from shipped CSV logs)
#
# Adapted from the dev script ``tools/plot_warmup.py`` so the Part 2 notebook
# can render the warmup diagnostic LIVE from the cached LoggingHook CSVs
# instead of embedding a pre-rendered PNG. Pandas / gzip / re are lazy-imported
# inside the function bodies so importing this module host-side (without pandas
# installed) does not break — only ``numpy`` and ``matplotlib`` are required at
# module top, both of which are already imported above.
# ---------------------------------------------------------------------------

# NVIDIA dark palette (matches plot_batch_speedup above).
_NV_DARK = "#000000"
_NV_LIGHT = "#F3F5F7"
_NV_MUTED = "#A8B0B8"
_NV_GRID = "#2F3A44"
_NV_SPINE = "#4B5563"

_WARMUP_LOG_EVERY = 100
_WARMUP_WINDOW = 20
_WARMUP_P_1ATM = 101325.0 / 1.602176634e11

# Per-panel accent colours drawn from / extending the NVIDIA palette, chosen
# to stay distinguishable on the dark background.
_WARMUP_PANEL_SPECS = {
    "temperature": (
        "temperature",
        "Temperature (K)",
        "#76B900",  # nv_green
        "T_TARGET",
        "target = {T:.0f} K",
    ),
    "density": (
        "density_g_cm3",
        "Density (g/cm$^3$)",
        "#00A3E0",  # nv_blue
        "RHO_EXP",
        "experimental = {RHO_EXP} g/cm$^3$",
    ),
    "pressure": (
        "pressure_eV_A3",
        "Pressure (eV/Å$^3$)",
        "#FFB000",  # amber
        "_P_1ATM",
        "1 atm = {P_1ATM:.2e} eV/Å$^3$",
    ),
    "energy": (
        "energy",
        "Potential energy (eV)",
        "#C77DFF",  # violet
        None,
        None,
    ),
    "fmax": (
        "fmax",
        "f$_{\\mathrm{max}}$ (eV/Å)",
        "#FF6B6B",  # coral
        None,
        None,
    ),
}

_WARMUP_STAGE_CONFIG = {
    "nvt": {
        "input_stem": "warmup_nvt",
        "title_phrase": "thermalization",
        "title_target": "{T:.0f} K",
        "panels": ("temperature", "energy", "fmax", "pressure"),
        "include_lattice": False,
    },
    "npt": {
        "input_stem": "warmup_npt",
        "title_phrase": "equilibration",
        "title_target": "{T:.0f} K / 1 atm",
        "panels": ("temperature", "density", "pressure", "energy", "fmax"),
        "include_lattice": True,
    },
}

# Three distinguishable colours for |a|, |b|, |c| on the lattice panel.
_WARMUP_LATTICE_COLORS = ("#FF8C42", "#4ECDC4", "#FF5DA2")


def _warmup_load_multipart_csv(base_csv):
    """Concatenate ``<stem>.csv`` + any ``<stem>.part*.csv`` siblings.

    ``base_csv`` is the path to the base warmup CSV; sibling extend-run parts
    in the same directory are discovered, ordered by their ``.partN`` index,
    and stitched with per-part *global-step* offsets (each part's local step
    counter restarts at the driver's first logged step). Ported from
    ``tools/plot_warmup.py::load_multipart_csv``.
    """
    import re

    import pandas as pd

    base_csv = Path(base_csv)
    log_dir = base_csv.parent
    basename = base_csv.stem  # e.g. "warmup_npt_200k_dt0p5fs"

    extras = []
    for p in log_dir.glob(f"{basename}.part*.csv"):
        m = re.search(r"\.part(\d+)$", p.stem)
        if m:
            extras.append((int(m.group(1)), p))
    extras.sort(key=lambda t: t[0])
    parts = ([base_csv] if base_csv.exists() else []) + [p for _, p in extras]
    if not parts:
        raise FileNotFoundError(f"no {basename}.csv* in {log_dir}")

    frames = []
    offset = 0
    for p in parts:
        df = pd.read_csv(p)
        df["global_step"] = df["step"] + offset
        frames.append(df)
        last_step = int(df["step"].max())
        offset += last_step + _WARMUP_LOG_EVERY
    return pd.concat(frames, ignore_index=True)


def _warmup_read_lattice_from_extxyz(path):
    """Stream per-frame cell-vector norms from the extxyz ``Lattice="..."``
    header (no ASE full-atom load). Ported from
    ``tools/plot_warmup.py::read_lattice_from_extxyz``.

    Captures the 9-float row-major lattice per frame plus the ``step=N`` info
    stamp; falls back to ``frame_idx * LOG_EVERY`` if the stamp is absent.
    """
    import gzip
    import re

    import pandas as pd

    path = Path(path)
    lattice_re = re.compile(r'Lattice="([^"]+)"')
    step_re = re.compile(r"\bstep=(\d+)")
    opener = gzip.open if path.suffix == ".gz" else open

    rows = []
    frame_idx = 0
    with opener(path, "rt") as f:
        for line in f:
            m = lattice_re.search(line)
            if m is None:
                continue
            cell = np.fromstring(m.group(1), sep=" ").reshape(3, 3)
            a, b, c = np.linalg.norm(cell, axis=1)
            step_match = step_re.search(line)
            step = int(step_match.group(1)) if step_match else frame_idx * _WARMUP_LOG_EVERY
            rows.append({"step": step, "a": a, "b": b, "c": c})
            frame_idx += 1
    if not rows:
        return pd.DataFrame(columns=["step", "a", "b", "c"])
    return pd.DataFrame(rows)


def _warmup_style_axis(ax):
    """Apply the NVIDIA dark palette to a single diagnostic panel."""
    ax.set_facecolor(_NV_DARK)
    ax.tick_params(colors=_NV_LIGHT, labelsize=10)
    ax.yaxis.label.set_color(_NV_LIGHT)
    ax.xaxis.label.set_color(_NV_LIGHT)
    for spine in ax.spines.values():
        spine.set_color(_NV_SPINE)
    ax.grid(True, color=_NV_GRID, linewidth=0.9, alpha=0.6)


def _warmup_plot_panel(ax, df, spec, t_target, rho_exp):
    """Raw trace (alpha 0.35) + centred rolling-mean (window 20) overlay for
    one scalar column, with an optional reference line."""
    col, ylabel, color, ref, ref_label = spec
    ax.plot(df["global_step"], df[col], color=color, alpha=0.35, lw=1.1)
    rolling = df[col].rolling(_WARMUP_WINDOW, center=True).mean()
    ax.plot(df["global_step"], rolling, color=color, lw=3.0)
    if ref is not None:
        ref_value = (
            t_target if ref == "T_TARGET"
            else rho_exp if ref == "RHO_EXP"
            else _WARMUP_P_1ATM if ref == "_P_1ATM"
            else ref
        )
        label = ref_label.format(T=t_target, RHO_EXP=rho_exp, P_1ATM=_WARMUP_P_1ATM)
        ax.axhline(ref_value, color=_NV_MUTED, ls=":", lw=1.6, label=label)
        ax.legend(
            loc="best", fontsize=9, framealpha=0.85,
            facecolor=_NV_DARK, edgecolor=_NV_SPINE, labelcolor=_NV_LIGHT,
        )
    ax.set_ylabel(ylabel)
    _warmup_style_axis(ax)


def plot_warmup_stage(
    csv_path,
    *,
    stage,
    t_warmup,
    dt=0.5,
    rho_exp=1.18,
    lattice_extxyz=None,
    title=None,
):
    """Render the warmup NVT/NPT diagnostic figure live from shipped CSV logs.

    Notebook-friendly adaptation of ``tools/plot_warmup.py``: builds a
    stacked-panel matplotlib figure on the NVIDIA dark palette and RETURNS it
    (no ``plt.show()`` / savefig / close — the caller displays then closes).

    Every panel draws the raw trace at ``alpha=0.35`` with a bold centred
    rolling mean (window 20) overlaid.

    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to the BASE warmup CSV (e.g.
        ``data/cached/naphthalene_orbmol/csv/warmup_npt.csv``). Any
        ``<stem>.part*.csv`` siblings in the same directory are auto-discovered
        and concatenated with per-part global-step offsets.
    stage : {"nvt", "npt"}
        Selects the panel set and title. NVT = temperature / energy / fmax /
        pressure (fixed cell). NPT = temperature / density / pressure / energy /
        fmax, plus an optional lattice panel.
    t_warmup : float
        Target temperature (K) — drives the temperature reference line + title.
    dt : float, optional
        MD timestep (fs); carried for the title. Default 0.5.
    rho_exp : float, optional
        Experimental density reference line on the NPT density panel. Default
        1.18 (naphthalene, Brock & Dunitz 1982, 295 K).
    lattice_extxyz : str | pathlib.Path | None, optional
        If given AND ``stage == "npt"``, append a |a| / |b| / |c| cell-vector
        panel parsed by streaming the trajectory's ``Lattice="..."`` headers.
        If None (or stage is NVT), the panel is silently omitted.
    title : str | None, optional
        Title override; if None a stage-aware title is built.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (not shown, not saved, not closed).
    """
    if stage not in _WARMUP_STAGE_CONFIG:
        raise ValueError(f"stage must be 'nvt' or 'npt', got {stage!r}")
    cfg = _WARMUP_STAGE_CONFIG[stage]

    df = _warmup_load_multipart_csv(csv_path)
    t_target = float(t_warmup)

    # NPT only: optionally parse per-frame lattice from a (thinned) trajectory.
    lattice_df = None
    if cfg["include_lattice"] and lattice_extxyz is not None:
        lattice_df = _warmup_read_lattice_from_extxyz(lattice_extxyz)
        if lattice_df.empty:
            lattice_df = None
        else:
            # The cached trajectory carries part-LOCAL step stamps for the main
            # (last) run-part, while the CSV is concatenated onto a global step
            # axis (part offsets > 0). Shift the lattice steps by that part's
            # offset so the lattice panel aligns with the CSV-derived panels.
            part_offsets = sorted(set(df["global_step"] - df["step"]))
            lattice_df = lattice_df.copy()
            lattice_df["step"] = lattice_df["step"] + part_offsets[-1]

    panel_keys = list(cfg["panels"])
    if lattice_df is not None:
        panel_keys.append("lattice")
    n_panels = len(panel_keys)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.2 * n_rows + 1.5), sharex=True,
        facecolor=_NV_DARK,
    )
    axes = np.atleast_1d(axes_grid).flatten().tolist()

    for ax, panel_key in zip(axes, panel_keys):
        if panel_key == "lattice":
            for col, color in zip(("a", "b", "c"), _WARMUP_LATTICE_COLORS):
                ax.plot(
                    lattice_df["step"], lattice_df[col],
                    color=color, alpha=0.35, lw=1.1,
                )
                rolling = lattice_df[col].rolling(_WARMUP_WINDOW, center=True).mean()
                ax.plot(
                    lattice_df["step"], rolling,
                    color=color, lw=3.0, label=f"|{col}|",
                )
            ax.set_ylabel("Cell-vector norm (Å)")
            _warmup_style_axis(ax)
            ax.legend(
                loc="best", fontsize=9, framealpha=0.85,
                facecolor=_NV_DARK, edgecolor=_NV_SPINE, labelcolor=_NV_LIGHT,
            )
        else:
            _warmup_plot_panel(
                ax, df, _WARMUP_PANEL_SPECS[panel_key], t_target, rho_exp
            )

    # Hide leftover grid cells when n_panels is odd.
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Mark each extend-run boundary (per-part offset) with a faint vertical rule.
    offsets = sorted(set(df["global_step"] - df["step"]))
    for off in offsets:
        if off == 0:
            continue
        for ax in axes[:n_panels]:
            ax.axvline(off, color=_NV_MUTED, ls="--", lw=0.7, alpha=0.4)

    if title is None:
        title_target = cfg["title_target"].format(T=t_target)
        title = (
            f"{stage.upper()} {title_target} {cfg['title_phrase']} "
            f"(DT = {dt} fs)"
        )
    fig.suptitle(title, color=_NV_LIGHT, fontsize=14, y=0.99)

    # sharex hides x labels on non-bottom rows; restore the label on the
    # bottom-most visible cell of each column (handles odd n_panels).
    bottom_row_indices = set()
    for col in range(n_cols):
        for row in range(n_rows - 1, -1, -1):
            idx = row * n_cols + col
            if idx < n_panels:
                bottom_row_indices.add(idx)
                break
    for idx in bottom_row_indices:
        axes[idx].set_xlabel("Global step")
        axes[idx].tick_params(axis="x", labelbottom=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# ---------------------------------------------------------------------------
# SLC production-NPT multi-temperature figure (live from shipped per-T CSVs)
#
# Adapted from the dev script ``tools/plot_slc.py`` so the Part 2 notebook (§11)
# can render the SLC production-NPT time-series LIVE from the shipped per-T
# multi-part CSV logs instead of embedding a pre-rendered PNG. Reuses
# ``_warmup_load_multipart_csv`` for each base CSV's ``.part*`` concatenation
# (one curve per temperature; the staging tool has already de-sharded the
# multi-GPU ``_t<subset>`` shards into per-T files, so no ``graph_idx`` remap is
# needed). Pandas is lazy-imported inside the function body.
# ---------------------------------------------------------------------------

# Physical bands each column should stay within when the system is behaving
# (ported from ``tools/plot_slc.py::HEALTHY_BOUNDS``). A temperature whose
# rolling-median lands outside its column's band is treated as diverged and
# excluded from that panel's ylim fence, so one blown-up trajectory doesn't
# crush the healthy ones. Energy/density are system-size dependent (no band)
# and fall through to a plain Tukey fence on all temperatures.
_SLC_HEALTHY_BOUNDS = {
    "temperature": (50.0, 1000.0),  # K
    "fmax": (0.0, 20.0),  # eV/Å
    "pressure_eV_A3": (-0.1, 0.1),  # eV/Å³ (condensed phase at 1 atm ~ 0)
}

# Panel registry: panel_key -> (csv_col, ylabel, accent_color). The production
# NPT stacks temperature / density / pressure / energy. fmax is omitted by
# default (clean four-panel layout); energy is the optional 4th panel.
_SLC_PANEL_SPECS = {
    "temperature": ("temperature", "Temperature (K)", "#76B900"),  # nv_green
    "density": ("density_g_cm3", "Density (g/cm$^3$)", "#00A3E0"),  # nv_blue
    "pressure": ("pressure_eV_A3", "Pressure (eV/Å$^3$)", "#FFB000"),  # amber
    "energy": ("energy", "Potential energy (eV)", "#C77DFF"),  # violet
}
_SLC_PANELS = ("temperature", "density", "pressure", "energy")


def _slc_median_in_band(rolling, band):
    """True if the finite-sample median of ``rolling`` lies in ``band``."""
    vals = np.asarray(rolling, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    return band[0] <= float(np.median(vals)) <= band[1]


def _slc_clip_ylim(per_temp_series, col, *, refs=None, k=1.5, pad=0.08):
    """Physical-band-filtered Tukey ylim (ported from
    ``tools/plot_slc.py::_clip_ylim``). Drops temperatures whose
    rolling-median falls outside ``_SLC_HEALTHY_BOUNDS[col]`` (when defined),
    then fences the surviving pooled samples with a Tukey rule. Falls back to
    all temperatures if the band rejects everything. Returns ``(lo, hi)`` or
    ``(None, None)`` when no finite samples exist.
    """
    band = _SLC_HEALTHY_BOUNDS.get(col)
    if band is not None:
        healthy = [g for g in per_temp_series if _slc_median_in_band(g, band)]
        if not healthy:
            healthy = per_temp_series
    else:
        healthy = per_temp_series
    if not healthy:
        return None, None
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in healthy])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None, None
    q25, q75 = np.nanpercentile(pooled, [25, 75])
    iqr = q75 - q25
    data_lo, data_hi = float(pooled.min()), float(pooled.max())
    if iqr > 0:
        lo = max(data_lo, q25 - k * iqr)
        hi = min(data_hi, q75 + k * iqr)
    else:
        lo, hi = data_lo, data_hi
    if refs:
        lo = min(lo, min(refs))
        hi = max(hi, max(refs))
    span = hi - lo
    if span == 0:
        span = abs(hi) * 0.1 if hi != 0 else 1.0
    return lo - pad * span, hi + pad * span


def plot_slc_stage(csv_by_temp, *, dt=0.5, title=None):
    """Render the SLC production-NPT multi-temperature diagnostic live from
    the shipped per-T CSV logs.

    Notebook-friendly adaptation of ``tools/plot_slc.py``: overlays one curve
    per temperature on a stacked-panel matplotlib figure (NVIDIA dark palette)
    and RETURNS the figure (no ``plt.show()`` / savefig / close — the caller
    displays then closes). Each temperature's base CSV is loaded via
    :func:`_warmup_load_multipart_csv`, auto-concatenating its ``.part*.csv``
    siblings with per-part global-step offsets.

    Every panel draws the raw trace at ``alpha=0.25`` with a bold centred
    rolling mean (window 20) overlaid. The per-column ylim uses the
    physical-band Tukey fence from ``tools/plot_slc.py`` so a single diverged
    temperature doesn't crush the healthy ones.

    Parameters
    ----------
    csv_by_temp : dict[int, str | pathlib.Path]
        Maps each temperature (K) to its BASE production-NPT CSV (e.g.
        ``{200: ".../csv/slc_npt_t200.csv", 300: ...}``). Each file is a single
        temperature already de-sharded from the multi-GPU ``_t<subset>`` shards,
        so any ``graph_idx`` column is single-valued and ignored.
    dt : float, optional
        MD timestep (fs); carried for the title. Default 0.5.
    title : str | None, optional
        Title override; if None a stage-aware title is built.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (not shown, not saved, not closed).
    """
    panel_keys = list(_SLC_PANELS)

    # Sort temperatures and assign a viridis colour per T.
    temps = sorted(int(t) for t in csv_by_temp)
    if not temps:
        raise ValueError("csv_by_temp is empty")
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(temps)))

    # Load each temperature's multipart CSV once; sort each on the global step.
    df_by_temp = {}
    for T in temps:
        df = _warmup_load_multipart_csv(csv_by_temp[T])
        df_by_temp[T] = df.sort_values("global_step")

    n_panels = len(panel_keys)
    fig, axes_grid = plt.subplots(
        n_panels, 1, figsize=(12, 2.6 * n_panels + 1.5), sharex=True,
        facecolor=_NV_DARK,
    )
    axes = np.atleast_1d(axes_grid).flatten().tolist()

    for ax, panel_key in zip(axes, panel_keys):
        col, ylabel, _ = _SLC_PANEL_SPECS[panel_key]
        per_temp_raw = []
        per_temp_rolling = []
        for T, color in zip(temps, colors):
            df = df_by_temp[T]
            if col not in df.columns:
                continue
            raw = df[col].to_numpy()
            rolling = df[col].rolling(_WARMUP_WINDOW, center=True).mean().to_numpy()
            per_temp_raw.append(raw)
            per_temp_rolling.append(rolling)
            ax.plot(df["global_step"], raw, color=color, alpha=0.25, lw=0.7)
            # Label only on the temperature panel; the figure legend reuses it.
            label = f"{T} K" if panel_key == "temperature" else None
            ax.plot(df["global_step"], rolling, color=color, lw=1.6, label=label)

        refs_for_ylim = None
        if panel_key == "temperature":
            # Per-T dotted target lines.
            for T, color in zip(temps, colors):
                ax.axhline(T, color=color, ls=":", lw=0.7, alpha=0.55)
            refs_for_ylim = list(temps)
        elif panel_key == "pressure":
            ax.axhline(_WARMUP_P_1ATM, color=_NV_MUTED, ls=":", lw=1.0)
            refs_for_ylim = [_WARMUP_P_1ATM]

        ax.set_ylabel(ylabel)
        _warmup_style_axis(ax)
        if per_temp_rolling:
            # Pressure raw cloud is far noisier than its rolling mean; fence the
            # raw values with a wider k=3 so the cloud stays visible. Other
            # panels fence the (tighter) rolling mean.
            if col == "pressure_eV_A3":
                lo, hi = _slc_clip_ylim(per_temp_raw, col, refs=refs_for_ylim, k=3.0)
            else:
                lo, hi = _slc_clip_ylim(per_temp_rolling, col, refs=refs_for_ylim)
            if lo is not None:
                ax.set_ylim(lo, hi)

    # Faint legend keyed by temperature on the top panel.
    axes[0].legend(
        loc="center right", fontsize=9, framealpha=0.85, title="Target T",
        ncol=1, facecolor=_NV_DARK, edgecolor=_NV_SPINE, labelcolor=_NV_LIGHT,
    )
    if axes[0].get_legend() is not None:
        axes[0].get_legend().get_title().set_color(_NV_LIGHT)

    # Mark each extend-run boundary (per-part offset) with a faint vertical rule.
    offsets = set()
    for df in df_by_temp.values():
        offsets.update(set(df["global_step"] - df["step"]))
    for off in sorted(offsets):
        if off == 0:
            continue
        for ax in axes:
            ax.axvline(off, color=_NV_MUTED, ls="--", lw=0.7, alpha=0.4)

    if title is None:
        title = f"SLC production NPT sweep — per-temperature (DT = {dt} fs)"
    fig.suptitle(title, color=_NV_LIGHT, fontsize=14, y=0.99)

    axes[-1].set_xlabel("Global step")
    axes[-1].tick_params(axis="x", labelbottom=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def plot_tm_bracket(
    per_T,
    bracket_temps,
    *,
    tm_lo,
    tm_hi,
    tm_exp,
    save_path="assets/images/tm_bracket.png",
):
    """Three-panel melting-point bracket figure (density, crystal-half S0, |D|).

    ``per_T`` maps each temperature in ``bracket_temps`` to a dict with keys
    ``rho`` / ``S0_c`` / ``D_c_cm2_s``. Draws the dark-theme figure, saves a PNG
    to ``save_path`` (unless None), and returns the Matplotlib Figure so the
    notebook can ``display`` then ``close`` it.
    """
    nv_green, nv_blue, nv_amber = "#76B900", "#00A3E0", "#F5A623"
    dark, light, grid = "#000000", "#F3F5F7", "#2F3A44"

    T_arr = np.array(bracket_temps, dtype=float)
    rho = np.array([per_T[T]["rho"] for T in bracket_temps])
    S0_c = np.array([per_T[T]["S0_c"] for T in bracket_temps])
    D_c = np.abs(np.array([per_T[T]["D_c_cm2_s"] for T in bracket_temps]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=dark)
    panels = [
        (axes[0], rho, "Density (g/cm³)", "Steady-state density", nv_green, "o-", False),
        (axes[1], S0_c, "S₀ (crystal half)", "Rotational order", nv_blue, "s-", False),
        (axes[2], D_c, "|D| crystal half (cm²/s)", "Translational diffusion", nv_amber, "^-", True),
    ]
    for ax, y, ylabel, title, color, style, logy in panels:
        ax.set_facecolor(dark)
        ax.plot(T_arr, y, style, color=color, markersize=10, lw=2.4)
        ax.axvspan(tm_lo, tm_hi, color=nv_green, alpha=0.10)
        ax.axvline(tm_exp, color=light, ls="--", lw=1.4, label=f"exp {int(tm_exp)} K")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Temperature (K)", color=light, fontsize=12)
        ax.set_ylabel(ylabel, color=light, fontsize=12)
        ax.set_title(title, color=light, fontsize=12, pad=8)
        ax.set_xlim(tm_lo - 30, tm_hi + 30)
        ax.set_xticks(list(T_arr))
        ax.tick_params(colors=light, labelsize=11)
        for s in ax.spines.values():
            s.set_color("#4B5563")
        ax.grid(True, color=grid, lw=0.8, alpha=0.7)
        ax.legend(facecolor=dark, edgecolor="#4B5563", labelcolor=light, fontsize=9, loc="best")

    fig.suptitle(
        f"Naphthalene melting-point screen — coarse bracket {tm_lo}–{tm_hi} K "
        f"(exp {int(tm_exp)} K)",
        color=light, fontsize=15, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
