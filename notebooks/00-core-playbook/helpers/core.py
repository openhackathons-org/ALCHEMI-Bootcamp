"""Checked data, tiny API sandboxes, and bounded visuals for the Core playbook."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, neighbor_list
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training import TrainingStage
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXTXYZ_SHA256 = "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
_MANIFEST_SHA256 = "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
_BENZOATE_SHA256 = "deb718970a5eb750955ab98963b089e9eab93370092fd6c69a3088fe7da7fb2d"
_MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_ELEMENT_COLORS = {1: "#E8ECEF", 6: "#5C6770", 7: "#00A3E0", 8: "#E05252"}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_molecules(labels: Sequence[str]) -> tuple[list[Atoms], list[dict[str, Any]]]:
    """Load an ordered selection after checking the shared data artifacts."""

    data_dir = _PROJECT_ROOT / "data" / "nci_atlas"
    xyz_path = data_dir / "ir-molecule-library.extxyz"
    manifest_path = data_dir / "ir-molecule-library-manifest.json"
    if _sha256_file(xyz_path) != _EXTXYZ_SHA256:
        raise RuntimeError(f"Unexpected molecule-library checksum: {xyz_path}")
    if _sha256_file(manifest_path) != _MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected molecule-manifest checksum: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_label = {record["label"]: record for record in manifest["molecules"]}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise KeyError(f"Labels absent from the pinned manifest: {missing}")

    all_atoms = list(read(xyz_path, index=":"))
    selected: list[Atoms] = []
    records: list[dict[str, Any]] = []
    for label in labels:
        record = by_label[label]
        atoms = all_atoms[int(record["extxyz_index"])].copy()
        atoms.info["charge"] = int(record["formal_charge"])
        if len(atoms) != int(record["atom_count"]):
            raise RuntimeError(f"Atom-count mismatch for {label}")
        selected.append(atoms)
        records.append(
            {
                "label": label,
                "formula": record["formula"],
                "atoms": record["atom_count"],
                "charge": record["formal_charge"],
                "source": f"{record['dataset']} / {record['system_id']}",
            }
        )
    return selected, records


def load_benzoate_anion() -> tuple[Atoms, dict[str, Any]]:
    """Load the checked benzoate fragment from NCI Atlas system 08.007."""

    path = (
        _PROJECT_ROOT
        / "notebooks"
        / "00-core-playbook"
        / "data"
        / "nci-benzoate-08.007_100.xyz"
    )
    if _sha256_file(path) != _BENZOATE_SHA256:
        raise RuntimeError(f"Unexpected benzoate checksum: {path}")

    atoms = read(path)
    charge = int(atoms.info["charge"])
    if atoms.get_chemical_formula() != "C7H5O2" or charge != -1:
        raise RuntimeError("Checked benzoate identity or charge drifted")
    return atoms, {
        "label": "Benzoate",
        "formula": "C7H5O2",
        "atoms": 14,
        "charge": charge,
        "source": "NCI Atlas IHB100x10 / 08.007 / scale 1.00 / fragment B",
        "source_revision": "1816bfc72609d7deb1d4f93ab9e27eb13bb44bec",
    }


def model_checkpoint() -> Path:
    """Resolve and verify the pinned AIMNet2 checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(_MODEL_ALIAS)).resolve()
    digest = _sha256_file(path)
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"AIMNet2 checkpoint checksum mismatch: {digest}")
    return path


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze parameters while preserving gradients with respect to positions."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def configure_presentation() -> None:
    """Apply the shared plotting style."""

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")


def atoms_to_xyz(atoms: Atoms) -> str:
    """Serialize one ASE structure for the pinned MatterViz widget."""

    stream = StringIO()
    write(stream, atoms, format="xyz")
    return stream.getvalue()


def infer_bonds(
    atoms: Atoms, *, cutoff_scale: float = 1.1
) -> tuple[tuple[int, int], ...]:
    """Infer unique molecular bonds from ASE covalent radii."""

    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    left_indices, right_indices = neighbor_list(
        "ij", atoms, cutoffs, self_interaction=False
    )
    return tuple(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for left, right in zip(left_indices, right_indices, strict=True)
                if left != right
            }
        )
    )


def bond_rows(
    atoms: Atoms, bonds: Sequence[tuple[int, int]]
) -> list[dict[str, int | str]]:
    """Return compact learner-facing rows for explicit connectivity."""

    return [
        {
            "atom 1": left,
            "element 1": atoms[left].symbol,
            "atom 2": right,
            "element 2": atoms[right].symbol,
        }
        for left, right in bonds
    ]


def _matterviz_structure(
    atoms: Atoms, bonds: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Build MatterViz's public structure dictionary with explicit bonds."""

    explicit_bonds = [
        {"site_idx_1": left, "site_idx_2": right, "order": 1} for left, right in bonds
    ]
    return {
        "sites": [
            {
                "species": [
                    {
                        "element": atom.symbol,
                        "occu": 1,
                        "oxidation_state": 0,
                    }
                ],
                "abc": [0, 0, 0],
                "xyz": [float(value) for value in atom.position],
                "label": f"{atom.symbol}{index + 1}",
                "properties": {},
            }
            for index, atom in enumerate(atoms)
        ],
        "charge": float(atoms.info.get("charge", 0)),
        "properties": {"bonds": explicit_bonds},
    }


def show_molecule(
    atoms: Atoms,
    *,
    bonds: Sequence[tuple[int, int]] | None = None,
    height: int = 440,
    show_controls: bool = True,
) -> Any:
    """Return a focused MatterViz view with explicit molecular connectivity."""

    from pymatviz import StructureWidget

    bonds = infer_bonds(atoms) if bonds is None else tuple(bonds)
    return StructureWidget(
        structure=_matterviz_structure(atoms, bonds),
        show_controls=show_controls,
        bond_thickness=0.14,
        bond_color="#D7DEE5",
        atom_radius=0.72,
        same_size_atoms=False,
        background_color="#111619",
        background_opacity=1.0,
        style=f"width:100%; height:{height}px; border-radius:8px;",
    )


def charge_badge(data: AtomicData) -> Any:
    """Return an accessible net-charge badge populated from Toolkit metadata."""

    from IPython.display import HTML

    charge = float(data.charge.item())
    formatted = f"{charge:+g}".replace("-", "−")
    return HTML(
        '<div role="note" aria-label="Net charge from Toolkit metadata" '
        'style="display:inline-flex;align-items:center;gap:0.45rem;'
        "background:#F2F3F1;color:#1B1E20;border:1px solid #D6D9D4;"
        'border-radius:999px;padding:0.35rem 0.72rem;font-size:0.92rem;">'
        f"<strong>Net charge: {formatted} e</strong>"
        '<span style="color:#555E63;">Toolkit system field</span></div>'
    )


def show_capability_map() -> Any:
    """Display the Toolkit map with native HTML hover details and links."""

    from IPython.display import HTML

    asset = _PROJECT_ROOT / "notebooks" / "00-core-playbook" / "assets"
    svg = (asset / "toolkit-capability-map.svg").read_text(encoding="utf-8")
    regions = (
        (
            1.702,
            "Data and state",
            (
                "Keep atomistic data as tensors",
                "on the model device.",
            ),
            "inputs · datasets · trajectories",
            (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/data.html",
                ),
            ),
            "Part 01 · Part 02",
        ),
        (
            26.170,
            "Models and potentials",
            (
                "Evaluate model outputs for each system:",
                "energy, forces, stress, and charges.",
            ),
            "MLIPs · custom models · composition",
            (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/models.html",
                ),
            ),
            "Part 03",
        ),
        (
            50.638,
            "Simulation workflows",
            (
                "Simulate batches of chemical systems",
                "with optimization, MD, and screening.",
            ),
            "FIRE2 · NVE/NVT · hooks",
            (
                (
                    "Docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/dynamics.html",
                ),
            ),
            "Part 04 · Part 05 · Part 06",
        ),
        (
            75.106,
            "Training and scale",
            (
                "Train and fine-tune models.",
                "Split one large system across GPUs.",
            ),
            "TrainingStrategy · DomainParallel",
            (
                (
                    "Training docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/training.html",
                ),
                (
                    "Scale docs ↗",
                    "https://nvidia.github.io/nvalchemi-toolkit/userguide/distributed.html",
                ),
            ),
            "Part 07 · Part 08",
        ),
    )
    overlays: list[str] = []
    for left, title, meaning, applications, docs, parts in regions:
        meaning_text = " ".join(meaning)
        meaning_html = "<br>".join(meaning)
        links = "".join(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
            for label, url in docs
        )
        overlays.append(
            f'<div class="alchemi-cap-region" style="left:{left}%" tabindex="0" '
            f'aria-label="{title}. {meaning_text}">'
            '<div class="alchemi-cap-detail">'
            f'<div class="alchemi-cap-title">{title}</div>'
            f'<div class="alchemi-cap-copy">{meaning_html}</div>'
            f'<div class="alchemi-cap-apps">{applications}</div>'
            f'<div class="alchemi-cap-docs">{links}</div>'
            '<div class="alchemi-cap-status-label">DEEP DIVES · IN PROGRESS</div>'
            f'<div class="alchemi-cap-status">{parts}</div>'
            "</div>"
            '<div class="alchemi-cap-outline"></div>'
            "</div>"
        )
    return HTML(
        """
<style>
.alchemi-cap-map{position:relative;isolation:isolate;width:100%;max-width:940px;aspect-ratio:94/39;font-family:'NVIDIA Sans',Arial,sans-serif}
.alchemi-cap-map>svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.alchemi-cap-map .cap-tooltip{display:none}
.alchemi-cap-region{position:absolute;top:28.205%;width:23.191%;height:64.103%;outline:none}
.alchemi-cap-detail{position:absolute;z-index:10;inset:0 0 auto 0;display:none;min-height:160px;box-sizing:border-box;padding:12px 14px 10px;background-color:#151A17;border:1px solid #76B900;border-radius:8px;box-shadow:0 10px 24px rgba(0,0,0,.52);color:#F3F4F6;visibility:hidden;pointer-events:none}
.alchemi-cap-region:hover .alchemi-cap-detail,.alchemi-cap-region:focus-visible .alchemi-cap-detail,.alchemi-cap-region:has(a:focus-visible) .alchemi-cap-detail{display:block;visibility:visible;pointer-events:auto}
.alchemi-cap-outline{position:absolute;left:3.67%;bottom:0;width:92.66%;height:33.6%;box-sizing:border-box;border:1px solid transparent;border-radius:8px;pointer-events:none;transition:border-color 130ms ease,filter 130ms ease}
.alchemi-cap-region:hover .alchemi-cap-outline,.alchemi-cap-region:focus-visible .alchemi-cap-outline,.alchemi-cap-region:has(a:focus-visible) .alchemi-cap-outline{border-color:#76B900;filter:drop-shadow(0 0 3px rgba(118,185,0,.25))}
.alchemi-cap-title{font-size:14px;font-weight:700;line-height:1.2;margin-bottom:5px}
.alchemi-cap-copy{font-size:11.5px;line-height:1.28;color:#B9C0C6}
.alchemi-cap-apps{margin-top:4px;font-size:11.25px;line-height:1.25;color:#A8B0B8}
.alchemi-cap-docs{display:flex;gap:18px;margin-top:6px;font-size:11px;font-weight:700;line-height:1.2}
.alchemi-cap-docs a{color:#76B900;text-decoration:underline;cursor:pointer}
.alchemi-cap-docs a:hover,.alchemi-cap-docs a:focus-visible{color:#A7D95B}
.alchemi-cap-status-label{margin-top:4px;color:#7C8794;font-size:9.5px;font-weight:600;letter-spacing:.02em}
.alchemi-cap-status{margin-top:1px;color:#929BA4;font-size:11px;font-weight:700;line-height:1.15}
@media (prefers-reduced-motion:reduce){.alchemi-cap-outline{transition:none}}
</style>
<div class="alchemi-cap-map">
"""
        + svg
        + "".join(overlays)
        + "</div>"
    )


def _polish_axis(axis: Any, *, grid: str | None = "y") -> None:
    """Apply the restrained Part 01 scientific-figure treatment."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is None:
        axis.grid(False)
    else:
        axis.grid(axis=grid, color="#2F3A44", alpha=0.58, linewidth=0.75)
    axis.tick_params(length=3.5, width=0.8)
    axis.title.set_fontweight("semibold")


def _display_figure(figure: Any, alt: str) -> None:
    from IPython.display import display

    figure.tight_layout(pad=1.15)
    display(figure, metadata={"alt": alt})
    plt.close(figure)


def plot_molecule(graph: AtomicData, label: str) -> Any:
    """Draw one compact molecular projection from the position tensor."""

    positions = graph.positions.detach().cpu().numpy()
    numbers = graph.atomic_numbers.detach().cpu().numpy()
    centered = positions - positions.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ basis[:2].T
    figure, axis = plt.subplots(figsize=(7.4, 3.4))
    for left in range(len(numbers)):
        for right in range(left + 1, len(numbers)):
            distance = np.linalg.norm(positions[left] - positions[right])
            limit = 1.2 * (
                covalent_radii[numbers[left]] + covalent_radii[numbers[right]]
            )
            if distance <= limit:
                axis.plot(
                    projection[[left, right], 0],
                    projection[[left, right], 1],
                    color="#8E969E",
                    linewidth=2.2,
                    zorder=1,
                )
    for atomic_number in sorted(set(numbers)):
        mask = numbers == atomic_number
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=190,
            color=_ELEMENT_COLORS.get(int(atomic_number), "#76B900"),
            edgecolor="#111315",
            linewidth=1.2,
            label=chemical_symbols[int(atomic_number)],
            zorder=2,
        )
    axis.set_title(f"{label} · {len(numbers)} atoms", pad=10)
    axis.set_aspect("equal")
    axis.margins(0.22)
    axis.axis("off")
    axis.legend(frameon=False, ncol=4, loc="upper right")
    return _display_figure(
        figure,
        f"Principal-axis projection of {label}; atoms are colored by element and connecting lines are covalent-radius visual guides.",
    )


def plot_batch_ownership(batch: Batch, labels: Sequence[str]) -> Any:
    """Show packed node rows, graph ownership, and pointer boundaries."""

    from matplotlib.patches import FancyBboxPatch, Rectangle

    numbers = batch.atomic_numbers.detach().cpu().reshape(-1).tolist()
    owners = batch.batch_idx.detach().cpu().numpy()
    pointers = batch.batch_ptr.detach().cpu().numpy()
    palette = ("#76B900", "#00A3E0", "#F5B642")
    figure, axis = plt.subplots(figsize=(11.5, 3.0), dpi=140)
    for row, (number, owner) in enumerate(zip(numbers, owners, strict=True)):
        color = palette[int(owner) % len(palette)]
        axis.add_patch(
            FancyBboxPatch(
                (row - 0.42, 0.48),
                0.84,
                0.44,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            row,
            0.70,
            chemical_symbols[int(number)],
            color="#0B0F10",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    for pointer in pointers[1:-1]:
        axis.axvline(pointer - 0.5, color="#CDD2D8", linestyle="--", linewidth=1.0)
    for graph, (label, start, stop) in enumerate(
        zip(labels, pointers[:-1], pointers[1:], strict=True)
    ):
        color = palette[graph % len(palette)]
        axis.add_patch(
            Rectangle(
                (start - 0.5, 0.08),
                stop - start,
                0.25,
                facecolor=color,
                edgecolor="none",
                alpha=0.78,
            )
        )
        center = (start + stop - 1) / 2
        axis.text(center, 1.15, f"{label} · {stop - start} atoms", ha="center")
        axis.text(
            center,
            0.205,
            f"rows {start}:{stop}  |  batch_idx {graph}",
            color="#F3F4F6",
            ha="center",
            va="center",
            fontsize=7.2,
        )
    axis.set_xlim(-0.75, len(owners) + 0.25)
    axis.set_ylim(-0.02, 1.36)
    axis.set_xticks(pointers)
    axis.set_xlabel("Atom-row offsets stored in batch_ptr")
    axis.set_yticks([])
    _polish_axis(axis, grid=None)
    axis.spines[["left", "right", "top"]].set_visible(False)
    return _display_figure(
        figure,
        "Packed atomic-number tiles for three molecules. Colored bands show coordinate slices and batch indexes; vertical boundaries are batch pointer offsets.",
    )


def plot_fire2_evidence(
    rows: Sequence[dict[str, float]],
    threshold: float,
    labels: Sequence[str],
) -> Any:
    """Plot per-system FIRE2 energy changes and maximum-force histories."""

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    colors = ("#76B900", "#00A3E0", "#F5B642")
    for graph, label in enumerate(labels):
        graph_rows = [row for row in rows if int(row["graph"]) == graph]
        steps = [int(row["step"]) for row in graph_rows]
        energies = np.asarray([float(row["energy_ev"]) for row in graph_rows])
        forces = [float(row["fmax_ev_per_a"]) for row in graph_rows]
        axes[0].plot(
            steps,
            energies - energies[0],
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            color=colors[graph % len(colors)],
            label=label,
        )
        axes[1].plot(
            steps,
            forces,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            color=colors[graph % len(colors)],
            label=label,
        )
    axes[0].set(
        title="Relative energy",
        xlabel="FIRE2 step",
        ylabel="Energy change [eV]",
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=1)
    axes[1].axhline(threshold, color="#F5B642", linestyle="--", linewidth=1.4)
    axes[1].set(
        title="Maximum force",
        xlabel="FIRE2 step",
        ylabel="Maximum force [eV/Å]",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(threshold * 0.72, None)
    axes[1].annotate(
        f"target {threshold:g} eV/Å",
        xy=(0.99, threshold),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 5),
        textcoords="offset points",
        color="#F5B642",
        fontsize=8,
        ha="right",
    )
    for axis in axes:
        _polish_axis(axis)
    return _display_figure(
        figure,
        "FIRE2 evidence: per-molecule energy change and maximum force versus optimizer step.",
    )


def plot_structure_change(before: AtomicData, after: AtomicData, label: str) -> Any:
    """Compare initial and returned coordinates with identical plot limits."""

    initial = before.positions.detach().cpu().numpy()
    final = after.positions.detach().cpu().numpy()
    numbers = before.atomic_numbers.detach().cpu().numpy()
    colors = [_ELEMENT_COLORS.get(int(number), "#76B900") for number in numbers]
    combined = np.concatenate([initial[:, :2], final[:, :2]])
    low, high = combined.min(axis=0), combined.max(axis=0)
    center = (low + high) / 2
    span = max(float((high - low).max()), 1.0) * 0.62
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), sharex=True, sharey=True)
    for axis, coordinates, title in zip(
        axes, (initial, final), ("Initial", "Returned coordinates"), strict=True
    ):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=70,
            c=colors,
            edgecolors="#111315",
            linewidth=1.0,
        )
        axis.set(
            title=title,
            xlabel="x [Å]",
            ylabel="y [Å]",
            aspect="equal",
            xlim=(center[0] - span, center[0] + span),
            ylim=(center[1] - span, center[1] + span),
        )
        _polish_axis(axis)
    maximum_displacement = np.linalg.norm(final - initial, axis=1).max()
    figure.suptitle(
        f"{label}: bounded FIRE2 update, max displacement {maximum_displacement:.3f} Å",
        fontsize=12,
        fontweight="semibold",
    )
    return _display_figure(
        figure,
        f"Side-by-side initial and returned-coordinate projections of {label} after bounded FIRE2, using identical axes.",
    )


def build_argon_batch(
    device: torch.device,
    dtype: torch.dtype,
    *,
    temperature_k: float = 50.0,
) -> Batch:
    """Build the periodic 27-atom argon sandbox used by the official NVE example."""

    sigma = 3.40
    spacing = 2 ** (1 / 6) * sigma
    coords = torch.arange(3, device=device, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    generator = torch.Generator(device=device).manual_seed(42)
    velocity_scale = (8.617333262e-5 * temperature_k / 39.948) ** 0.5
    velocities = torch.randn(
        positions.shape, generator=generator, device=device, dtype=dtype
    )
    velocities = velocity_scale * (velocities - velocities.mean(dim=0, keepdim=True))
    graph = AtomicData(
        positions=positions,
        atomic_numbers=torch.full(
            (len(positions),), 18, device=device, dtype=torch.long
        ),
        atomic_masses=torch.full((len(positions),), 39.948, device=device, dtype=dtype),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), device=device, dtype=dtype),
        cell=torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * (3 * spacing),
        pbc=torch.ones((1, 3), device=device, dtype=torch.bool),
    )
    batch = Batch.from_data_list([graph], device=device)
    batch.add_key("velocities", [velocities], level="node", overwrite=True)
    return batch


class NVETraceHook:
    """Record potential, kinetic, and total energy after each short NVE step."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
        if ctx.batch is None:
            raise RuntimeError("NVETraceHook requires a populated batch")
        batch = ctx.batch
        kinetic = 0.5 * (batch.atomic_masses[:, None] * batch.velocities.square()).sum()
        potential = batch.energy.sum()
        self.rows.append(
            {
                "step": float(ctx.step_count),
                "potential_ev": float(potential.detach().cpu()),
                "kinetic_ev": float(kinetic.detach().cpu()),
                "total_ev": float((potential + kinetic).detach().cpu()),
            }
        )


def plot_nve_trace(rows: Sequence[dict[str, float]]) -> Any:
    """Plot five recorded NVE updates as control-flow evidence."""

    steps = [row["step"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    for key, label, style in (
        ("potential_ev", "potential", "--"),
        ("kinetic_ev", "kinetic", ":"),
        ("total_ev", "total", "-"),
    ):
        axis.plot(
            steps,
            [row[key] for row in rows],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            linestyle=style,
            label=label,
        )
    axis.set(
        title="Energy components over five NVE updates",
        xlabel="Post-update index",
        ylabel="Energy [eV]",
        xticks=steps,
    )
    _polish_axis(axis)
    axis.legend(frameon=False, ncol=3, loc="center right")
    return _display_figure(
        figure,
        "Potential, kinetic, and total energy recorded over five Lennard-Jones NVE updates.",
    )


class NativeQuadraticCorrection(torch.nn.Module):
    """Deterministic native tensor term used only to demonstrate adaptation."""

    def __init__(self, scale: float = 1.0e-4) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(
        self,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        node_energy = self.scale * positions.square().sum(dim=1)
        graph_energy = torch.zeros(
            graph_count,
            device=positions.device,
            dtype=positions.dtype,
        )
        return graph_energy.index_add(0, batch_idx, node_energy).unsqueeze(-1)


def plot_wrapper_flow(parity_delta: float) -> Any:
    """Draw the native-to-wrapper-to-composition boundary and parity check."""

    figure, axis = plt.subplots(figsize=(8.6, 2.2))
    axis.set(xlim=(-0.25, 2.25), ylim=(-0.55, 0.55))
    axis.axis("off")
    labels = (
        ("Native PyTorch term", "positions + batch_idx"),
        ("Toolkit model contract", "Batch → energy"),
        ("Additive pipeline", "LJ energy + toy energy"),
    )
    for x, (label, detail) in enumerate(labels):
        axis.text(
            x,
            0.18,
            label,
            ha="center",
            va="center",
            color="#F3F5F7",
            fontweight="semibold",
        )
        axis.text(
            x, -0.02, detail, ha="center", va="center", color="#A8B0B8", fontsize=9
        )
    for start in (0, 1):
        axis.annotate(
            "",
            xy=(start + 0.68, 0.18),
            xytext=(start + 0.32, 0.18),
            arrowprops={"arrowstyle": "->", "color": "#76B900", "linewidth": 1.8},
        )
    axis.text(
        1.0,
        -0.40,
        f"max |pipeline - manual sum| = {parity_delta:.2e} eV",
        ha="center",
        color="#F5B642",
        fontsize=9.5,
    )
    return _display_figure(
        figure,
        "A native quadratic toy term passes through a Toolkit wrapper and is added to Lennard-Jones; the displayed delta checks implementation identity.",
    )


class SyntheticEnergyDataset(TorchDataset[AtomicData]):
    """Deterministic fixed-size systems with a smooth synthetic energy target."""

    def __init__(self, count: int, atoms: int, seed: int, scale: float = 1.0) -> None:
        self.count, self.atoms, self.seed, self.scale = count, atoms, seed, scale

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> AtomicData:
        generator = torch.Generator().manual_seed(self.seed + index)
        positions = torch.randn(self.atoms, 3, generator=generator)
        energy = (self.scale * positions.square().sum()).reshape(1, 1)
        return AtomicData(
            positions=positions,
            atomic_numbers=torch.ones(self.atoms, dtype=torch.long),
            energy=energy,
        )


class SimpleEnergyMLP(torch.nn.Module, BaseModelMixin):
    """Small fixed-size MLP adapted from the pinned official DDP example."""

    def __init__(self, num_atoms: int = 4, hidden_dim: int = 24) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.network = torch.nn.Sequential(
            torch.nn.Linear(num_atoms * 3, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"positions"}),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        return data

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        graph_count = data.batch_size if isinstance(data, Batch) else 1
        features = data.positions.reshape(graph_count, self.num_atoms * 3)
        return {"energy": self.network(features)}


def _synthetic_loader(
    dataset: SyntheticEnergyDataset, batch_size: int
) -> TorchDataLoader:
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: Batch.from_data_list(list(samples)),
    )


def prepare_synthetic_transfer(
    device: torch.device,
) -> tuple[SimpleEnergyMLP, TorchDataLoader, float]:
    """Fit source-task weights briefly, then return a shifted fine-tuning loader."""

    torch.manual_seed(7)
    model = SimpleEnergyMLP().to(device)
    source_loader = _synthetic_loader(SyntheticEnergyDataset(32, 4, 100), batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-2)
    last_minibatch_loss = float("nan")
    model.train()
    for _ in range(8):
        for batch in source_loader:
            batch = batch.to(device)
            loss = torch.nn.functional.mse_loss(model(batch)["energy"], batch.energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_minibatch_loss = float(loss.detach().cpu())
    shifted_loader = _synthetic_loader(
        SyntheticEnergyDataset(16, 4, 500, scale=0.8),
        batch_size=4,
    )
    return model, shifted_loader, last_minibatch_loss


class LossHistoryHook:
    """Record scalar loss after each completed training batch."""

    stage = TrainingStage.AFTER_BATCH
    frequency = 1

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        if ctx.loss is None:
            raise RuntimeError("LossHistoryHook requires a completed batch")
        self.rows.append(
            {"step": float(ctx.step_count), "loss": float(ctx.loss.detach().cpu())}
        )


def plot_training_loss(rows: Sequence[dict[str, float]]) -> Any:
    """Plot the deliberately tiny fine-tuning loss trace."""

    steps = [int(row["step"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.6, 3.4))
    axis.scatter(
        steps,
        [row["loss"] for row in rows],
        color="#76B900",
        s=52,
        edgecolor="#111315",
        linewidth=0.8,
    )
    axis.set(
        title="Synthetic minibatch loss",
        xlabel="Optimizer update",
        ylabel="MSE [synthetic units²]",
        xticks=steps,
    )
    _polish_axis(axis)
    return _display_figure(
        figure,
        "Four unconnected points show MSE on four different synthetic minibatches.",
    )


def plot_domain_control(world_size: int, atom_count: int) -> Any:
    """Draw the ownership result for a world-size-one domain walkthrough."""

    columns = 5
    rows = int(np.ceil(atom_count / columns))
    indices = np.arange(atom_count)
    x = indices % columns
    y = rows - 1 - indices // columns
    figure, axis = plt.subplots(figsize=(6.8, 3.2))
    axis.scatter(
        x,
        y,
        s=130,
        color="#76B900",
        edgecolor="#111315",
        linewidth=0.9,
    )
    axis.add_patch(
        plt.Rectangle(
            (-0.65, -0.65),
            columns - 1 + 1.3,
            rows - 1 + 1.3,
            fill=False,
            edgecolor="#CDD2D8",
            linewidth=1.1,
        )
    )
    axis.set_title(f"World-size-{world_size} ownership")
    axis.text(
        columns + 0.15,
        (rows - 1) / 2,
        f"rank 0\nowns all\n{atom_count} atoms",
        color="#F3F5F7",
        va="center",
        fontsize=11,
        linespacing=1.35,
    )
    axis.set_aspect("equal")
    axis.set_xlim(-0.9, columns + 1.45)
    axis.set_ylim(-0.9, rows - 0.1)
    axis.axis("off")
    return _display_figure(
        figure,
        f"World-size-{world_size} domain ownership schematic; rank zero owns all {atom_count} atoms.",
    )
