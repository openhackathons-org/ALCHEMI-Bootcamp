"""Checked inputs and presentation support for the BaseDynamics lesson."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Self

import pandas as pd
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read
from nvalchemi.data import Batch
from nvalchemi.dynamics import DynamicsStage
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

EXPECTED_EXTXYZ_SHA256 = (
    "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
)
EXPECTED_MANIFEST_SHA256 = (
    "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
)
MODEL_ALIAS = "aimnet2-wb97m-d3_0"
MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
COMPLETED_UPDATE_COLUMN = "completed update (1-based)"
FIRST_CONVERGED_UPDATE_COLUMN = "first converged completed update (1-based)"
KB_EV_K = 8.617333262e-5
REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root(start: Path | None = None) -> Path:
    """Find the tutorials v3 root from a notebook or repository directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "environment" / "runtime-pins.toml").is_file()
            and (candidate / "shared" / "alchemi-dark.mplstyle").is_file()
            and (candidate / "data" / "nci_atlas").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Run this notebook from inside the tutorials v3 checkout.")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_molecule_selection(
    labels: tuple[str, ...],
    root: Path = REPO_ROOT,
) -> tuple[list[Atoms], pd.DataFrame]:
    """Load and verify an ordered molecule selection from the shared collection."""

    data_dir = root / "data" / "nci_atlas"
    extxyz_path = data_dir / "ir-molecule-library.extxyz"
    manifest_path = data_dir / "ir-molecule-library-manifest.json"
    if _sha256_file(extxyz_path) != EXPECTED_EXTXYZ_SHA256:
        raise RuntimeError(f"Unexpected molecule-library checksum: {extxyz_path}")
    if _sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected molecule-manifest checksum: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = manifest["molecules"]
    record_by_label = {str(record["label"]): record for record in records}
    missing = [label for label in labels if label not in record_by_label]
    if missing:
        raise KeyError(f"Molecule labels missing from the pinned manifest: {missing}")

    all_atoms = list(read(extxyz_path, index=":"))
    selected_atoms: list[Atoms] = []
    rows: list[dict[str, Any]] = []
    for label in labels:
        record = record_by_label[label]
        atoms = all_atoms[int(record["extxyz_index"])]
        if len(atoms) != int(record["atom_count"]):
            raise RuntimeError(f"Atom-count mismatch for {label}.")
        if atoms.get_chemical_formula() != str(record["formula"]):
            raise RuntimeError(f"Formula mismatch for {label}.")
        atoms.info["charge"] = int(record["formal_charge"])
        selected_atoms.append(atoms)
        rows.append(
            {
                "label": label,
                "formula": str(record["formula"]),
                "atoms": int(record["atom_count"]),
                "charge": int(record["formal_charge"]),
                "source": f"{record['dataset']} / {record['system_id']}",
            }
        )

    frame = pd.DataFrame(rows)
    if frame["charge"].ne(0).any():
        raise RuntimeError("The BaseDynamics lesson expects neutral molecules.")
    return selected_atoms, frame


def periodic_argon_geometry(
    *,
    n_side: int,
    spacing: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build simple-cubic coordinates and one periodic cubic cell."""

    if n_side < 2:
        raise ValueError("n_side must be at least 2")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    coordinates = torch.arange(n_side, device=device, dtype=dtype) * spacing
    grid_x, grid_y, grid_z = torch.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing="ij",
    )
    positions = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()],
        dim=-1,
    )
    cell = (
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        * (n_side * spacing)
    )
    return positions, cell


def resolve_verified_checkpoint(alias: str, expected_sha256: str) -> Path:
    """Resolve one AIMNet registry alias and verify its downloaded checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(alias)).resolve()
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"Checkpoint checksum mismatch: {digest}")
    return path


def model_checkpoint() -> Path:
    """Return the verified AIMNet2 checkpoint selected for this lesson."""

    return resolve_verified_checkpoint(MODEL_ALIAS, MODEL_SHA256)


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze model parameters while preserving input-coordinate gradients."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def dynamics_progress() -> Progress:
    """Create the shared progress display for repeated dynamics steps."""

    return Progress(
        SpinnerColumn("dots", style="#76B900"),
        TextColumn("{task.description}"),
        BarColumn(
            bar_width=32,
            style="#1F2933",
            complete_style="#76B900",
            finished_style="#76B900",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


class RelaxationMonitor:
    """Record per-system FIRE2 results and show progress up to the step cap."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, labels: Sequence[str], *, total_steps: int) -> None:
        if not labels:
            raise ValueError("labels must contain at least one system")
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.labels = tuple(labels)
        self.total_steps = total_steps
        self._rows: list[dict[str, float | int | str]] = []
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def __enter__(self) -> Self:
        self._progress = dynamics_progress()
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            "Batched FIRE2 relaxation", total=self.total_steps
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(*exc_info)
        self._progress = None
        self._task_id = None

    def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
        """Record a one-based completed update from zero-based callback state."""

        self.record(ctx.batch, completed_update=ctx.step_count + 1)
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1, refresh=True)

    def record(self, batch: Batch, *, completed_update: int) -> None:
        """Append one row per system from the current batch state."""

        if batch.num_graphs != len(self.labels):
            raise ValueError("labels must contain one name per batch system")
        fmax = per_system_fmax(batch)
        energies = batch.energy.detach().reshape(-1)
        statuses = batch.status.detach().reshape(-1)
        for graph, label in enumerate(self.labels):
            self._rows.append(
                {
                    COMPLETED_UPDATE_COLUMN: completed_update,
                    "graph": graph,
                    "molecule": label,
                    "energy (eV)": float(energies[graph].cpu()),
                    "fmax (eV/Å)": float(fmax[graph].cpu()),
                    "status": int(statuses[graph].cpu()),
                }
            )

    def history_frame(self) -> pd.DataFrame:
        """Return recorded rows in stable column order."""

        return pd.DataFrame(
            self._rows,
            columns=[
                COMPLETED_UPDATE_COLUMN,
                "graph",
                "molecule",
                "energy (eV)",
                "fmax (eV/Å)",
                "status",
            ],
        )


def kinetic_energy_per_system(batch: Batch) -> torch.Tensor:
    """Return kinetic energy in eV for every graph."""

    masses = batch.atomic_masses.detach().reshape(-1)
    speed_squared = batch.velocities.detach().square().sum(dim=1)
    node_energy = 0.5 * masses * speed_squared
    energy = torch.zeros(
        batch.num_graphs,
        dtype=node_energy.dtype,
        device=node_energy.device,
    )
    energy.scatter_add_(0, batch.batch_idx.long(), node_energy)
    return energy


def temperature_per_system(batch: Batch) -> torch.Tensor:
    """Return Toolkit's unconstrained 3N kinetic-temperature estimate."""

    kinetic = kinetic_energy_per_system(batch)
    counts = batch.num_nodes_per_graph.to(
        device=kinetic.device,
        dtype=kinetic.dtype,
    )
    degrees_of_freedom = (3 * counts).clamp_min(1)
    return 2 * kinetic / (degrees_of_freedom * KB_EV_K)


class DynamicsTrace:
    """Record small per-system MD traces and optional wrapped coordinates."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(
        self,
        labels: Sequence[str],
        *,
        store_positions: bool = False,
        position_frequency: int = 1,
    ) -> None:
        if not labels:
            raise ValueError("labels must contain at least one system")
        if position_frequency < 1:
            raise ValueError("position_frequency must be positive")
        self.labels = tuple(labels)
        self.store_positions = store_positions
        self.position_frequency = position_frequency
        self._rows: list[dict[str, float | int | str]] = []
        self.position_frames: list[torch.Tensor] = []

    def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
        """Record the state after one completed update."""

        self.record(ctx.batch, completed_update=ctx.step_count + 1)

    def record(self, batch: Batch, *, completed_update: int) -> None:
        """Append energy, temperature, status, and optional positions."""

        if batch.num_graphs != len(self.labels):
            raise ValueError("labels must contain one name per batch system")
        kinetic = kinetic_energy_per_system(batch)
        potential = batch.energy.detach().reshape(-1)
        temperature = temperature_per_system(batch)
        status = batch.status.detach().reshape(-1)
        for graph, label in enumerate(self.labels):
            self._rows.append(
                {
                    COMPLETED_UPDATE_COLUMN: completed_update,
                    "system": label,
                    "potential energy (eV)": float(potential[graph].cpu()),
                    "kinetic energy (eV)": float(kinetic[graph].cpu()),
                    "total energy (eV)": float(
                        (potential[graph] + kinetic[graph]).cpu()
                    ),
                    "temperature (K)": float(temperature[graph].cpu()),
                    "status": int(status[graph].cpu()),
                }
            )
        if self.store_positions and completed_update % self.position_frequency == 0:
            self.position_frames.append(batch.positions.detach().cpu().clone())

    def history_frame(self) -> pd.DataFrame:
        """Return observations in stable column order."""

        return pd.DataFrame(
            self._rows,
            columns=[
                COMPLETED_UPDATE_COLUMN,
                "system",
                "potential energy (eV)",
                "kinetic energy (eV)",
                "total energy (eV)",
                "temperature (K)",
                "status",
            ],
        )


def per_system_fmax(batch: Batch) -> torch.Tensor:
    """Return the maximum atomic force norm for each system in a batch."""

    force_norms = torch.linalg.vector_norm(batch.forces.detach(), dim=1)
    fmax = torch.zeros(
        batch.num_graphs,
        dtype=force_norms.dtype,
        device=force_norms.device,
    )
    fmax.scatter_reduce_(
        0,
        batch.batch_idx.long(),
        force_norms,
        reduce="amax",
        include_self=True,
    )
    return fmax


def truncate_history_at_convergence(
    history: pd.DataFrame,
    *,
    final_batch: Batch | None = None,
) -> pd.DataFrame:
    """Drop post-convergence rows and optionally replace each final observation."""

    kept: list[pd.DataFrame] = []
    final_fmax = per_system_fmax(final_batch) if final_batch is not None else None
    for graph, (_, rows) in enumerate(history.groupby("molecule", sort=False)):
        rows = rows.sort_values(COMPLETED_UPDATE_COLUMN).copy()
        converged = rows.index[rows["status"].ge(1)]
        if len(converged):
            rows = rows.loc[: converged[0]]
        if final_batch is not None:
            final_index = rows.index[-1]
            rows.loc[final_index, "energy (eV)"] = final_batch.energy[
                graph
            ].detach().item()
            rows.loc[final_index, "fmax (eV/Å)"] = final_fmax[
                graph
            ].detach().item()
            rows.loc[final_index, "status"] = final_batch.status[
                graph
            ].detach().item()
        kept.append(rows)
    return pd.concat(kept, ignore_index=True)


def summarize_relaxation(
    metadata: pd.DataFrame,
    history: pd.DataFrame,
    initial_positions: torch.Tensor,
    final_batch: Batch,
) -> pd.DataFrame:
    """Shape first/final values and displacements in source-system order."""

    labels = metadata["label"].astype(str).tolist()
    if labels != list(dict.fromkeys(history["molecule"].astype(str))):
        raise ValueError("history molecule order must match metadata")
    sort_columns = (
        [COMPLETED_UPDATE_COLUMN, "graph"]
        if "graph" in history
        else [COMPLETED_UPDATE_COLUMN]
    )
    ordered = history.sort_values(sort_columns)
    first = ordered.groupby("molecule", sort=False).first()
    first_converged = (
        ordered.loc[ordered["status"].ge(1)]
        .groupby("molecule", sort=False)[COMPLETED_UPDATE_COLUMN]
        .min()
        .to_dict()
    )
    final_energies = final_batch.energy.detach().reshape(-1)
    final_fmax = per_system_fmax(final_batch)
    final_statuses = final_batch.status.detach().reshape(-1)

    rows: list[dict[str, Any]] = []
    for graph, record in enumerate(metadata.itertuples(index=False)):
        label = str(record.label)
        start = int(final_batch.batch_ptr[graph])
        stop = int(final_batch.batch_ptr[graph + 1])
        displacement = torch.linalg.vector_norm(
            final_batch.positions[start:stop] - initial_positions[start:stop],
            dim=1,
        )
        status = int(final_statuses[graph].cpu())
        rows.append(
            {
                "molecule": label,
                "formula": str(record.formula),
                "atoms": int(record.atoms),
                "outcome": "converged" if status >= 1 else "update limit",
                FIRST_CONVERGED_UPDATE_COLUMN: first_converged.get(label, pd.NA),
                "initial fmax (eV/Å)": float(first.loc[label, "fmax (eV/Å)"]),
                "final fmax (eV/Å)": float(final_fmax[graph].cpu()),
                "energy change (eV)": float(
                    final_energies[graph].cpu()
                    - first.loc[label, "energy (eV)"]
                ),
                "maximum displacement (Å)": float(displacement.max().cpu()),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def energy_conservation_summary(
    history: pd.DataFrame,
    *,
    atoms_per_system: int,
) -> pd.Series:
    """Summarize total-energy change for one sampled trajectory."""

    if atoms_per_system < 1:
        raise ValueError("atoms_per_system must be positive")
    if history.empty or history["system"].nunique() != 1:
        raise ValueError("history must contain one non-empty system trace")
    ordered = history.sort_values(COMPLETED_UPDATE_COLUMN)
    total = ordered["total energy (eV)"]
    delta = total - total.iloc[0]
    maximum = float(delta.abs().max())
    updates = int(
        ordered[COMPLETED_UPDATE_COLUMN].iloc[-1]
        - ordered[COMPLETED_UPDATE_COLUMN].iloc[0]
    )
    normalized_drift = (
        maximum / (atoms_per_system * updates) if updates > 0 else 0.0
    )
    return pd.Series(
        {
            "updates": updates,
            "maximum |ΔE| (eV)": maximum,
            "maximum |ΔE| per atom (meV/atom)": 1000
            * maximum
            / atoms_per_system,
            "maximum |ΔE| per atom per update (eV/atom/update)": normalized_drift,
            "final temperature (K)": float(ordered["temperature (K)"].iloc[-1]),
        },
        name="value",
    )


def plot_force_history(history: pd.DataFrame, *, target: float) -> Any:
    """Plot per-system and batch-maximum force histories."""

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.axhline(
        target,
        color="#F3F4F6",
        linestyle="--",
        linewidth=1.2,
        label=f"target = {target:.2f} eV/Å",
    )
    for _, rows in history.groupby("molecule", sort=False):
        axis.plot(
            rows[COMPLETED_UPDATE_COLUMN],
            rows["fmax (eV/Å)"],
            color="#7C8794",
            alpha=0.38,
            linewidth=1.0,
        )
    batch_max = history.groupby(COMPLETED_UPDATE_COLUMN, as_index=False)[
        "fmax (eV/Å)"
    ].max()
    axis.plot(
        batch_max[COMPLETED_UPDATE_COLUMN],
        batch_max["fmax (eV/Å)"],
        color="#76B900",
        linewidth=2.4,
        label="maximum among active trajectories",
    )
    axis.set(
        xlabel="Completed update (1-based)",
        ylabel="Maximum force (eV/Å)",
        title="Per-system FIRE2 convergence",
    )
    axis.set_yscale("log")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_energy_conservation(
    history: pd.DataFrame,
    *,
    atoms_per_system: int,
) -> Any:
    """Plot energy components and total-energy change per atom."""

    import matplotlib.pyplot as plt

    ordered = history.sort_values(COMPLETED_UPDATE_COLUMN)
    updates = ordered[COMPLETED_UPDATE_COLUMN]
    total = ordered["total energy (eV)"]
    delta_per_atom = 1000 * (total - total.iloc[0]) / atoms_per_system
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].plot(
        updates,
        ordered["potential energy (eV)"],
        color="#7C8794",
        label="potential",
    )
    axes[0].plot(
        updates,
        ordered["kinetic energy (eV)"],
        color="#00A3E0",
        label="kinetic",
    )
    axes[0].plot(
        updates,
        total,
        color="#76B900",
        linewidth=2.1,
        label="total",
    )
    axes[0].set(
        xlabel="Completed update (1-based)",
        ylabel="Energy (eV)",
        title="NVE energy components",
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="#7C8794", linewidth=1.0)
    axes[1].plot(updates, delta_per_atom, color="#76B900", linewidth=2.1)
    axes[1].set(
        xlabel="Completed update (1-based)",
        ylabel="ΔE per atom (meV/atom)",
        title="Total-energy change from the initial state",
    )
    figure.tight_layout()
    return figure


def plot_temperature_history(history: pd.DataFrame, *, target: float) -> Any:
    """Plot an instantaneous kinetic-temperature trace."""

    import matplotlib.pyplot as plt

    ordered = history.sort_values(COMPLETED_UPDATE_COLUMN)
    figure, axis = plt.subplots(figsize=(9.2, 4.2))
    axis.plot(
        ordered[COMPLETED_UPDATE_COLUMN],
        ordered["temperature (K)"],
        color="#76B900",
        linewidth=2.0,
        label="instantaneous temperature",
    )
    axis.axhline(
        target,
        color="#F3F4F6",
        linestyle="--",
        linewidth=1.2,
        label=f"target = {target:g} K",
    )
    axis.set(
        xlabel="Completed update (1-based)",
        ylabel="Instantaneous temperature (K)",
        title="Short NVT Langevin trace",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_argon_trajectory(
    frames: Sequence[torch.Tensor],
    *,
    box_size: float,
) -> Any:
    """Show matched wrapped coordinates at initial, middle, and final times."""

    import matplotlib.pyplot as plt

    if len(frames) < 3:
        raise ValueError("at least three trajectory frames are required")
    if box_size <= 0:
        raise ValueError("box_size must be positive")
    selected = (0, len(frames) // 2, len(frames) - 1)
    titles = ("initial", "middle", "final")
    figure = plt.figure(figsize=(11.4, 3.9))
    for panel, (frame_index, title) in enumerate(zip(selected, titles, strict=True)):
        positions = frames[frame_index].detach().cpu()
        axis = figure.add_subplot(1, 3, panel + 1, projection="3d")
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=28,
            color="#76B900",
            edgecolor="#111315",
            linewidth=0.35,
        )
        axis.set(
            title=f"{title} · frame {frame_index}",
            xlabel="x (Å)",
            ylabel="y (Å)",
            zlabel="z (Å)",
            xlim=(0, box_size),
            ylim=(0, box_size),
            zlim=(0, box_size),
        )
        axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    return figure


def plot_structure_change(
    atomic_numbers: torch.Tensor,
    initial_positions: torch.Tensor,
    final_positions: torch.Tensor,
    *,
    label: str,
) -> Any:
    """Overlay matched atom positions before and after relaxation."""

    import matplotlib.pyplot as plt

    numbers = atomic_numbers.detach().cpu().reshape(-1)
    before = initial_positions.detach().cpu()
    after = final_positions.detach().cpu()
    if before.shape != after.shape or before.shape != (len(numbers), 3):
        raise ValueError("positions must have shape [atoms, 3] before and after")

    figure = plt.figure(figsize=(8.4, 5.6))
    axis = figure.add_subplot(111, projection="3d")
    for atom in range(len(numbers)):
        axis.plot(
            [before[atom, 0], after[atom, 0]],
            [before[atom, 1], after[atom, 1]],
            [before[atom, 2], after[atom, 2]],
            color="#CDD2D8",
            linewidth=1.1,
            alpha=0.7,
        )
    sizes = 24.0 + 7.0 * numbers.numpy()
    axis.scatter(
        before[:, 0],
        before[:, 1],
        before[:, 2],
        s=sizes,
        color="#00A3E0",
        alpha=0.72,
        label="before",
    )
    axis.scatter(
        after[:, 0],
        after[:, 1],
        after[:, 2],
        s=sizes,
        color="#76B900",
        label="after",
    )
    for atom, number in enumerate(numbers.tolist()):
        axis.text(
            after[atom, 0],
            after[atom, 1],
            after[atom, 2],
            chemical_symbols[number],
            fontsize=8,
        )
    axis.set(
        title=f"{label}: matched atom positions",
        xlabel="x (Å)",
        ylabel="y (Å)",
        zlabel="z (Å)",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def start_tutorial() -> None:
    """Apply the shared plot style and suppress one known wrapper warning."""

    import matplotlib.pyplot as plt

    plt.style.use(REPO_ROOT / "shared" / "alchemi-dark.mplstyle")
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True",
        category=UserWarning,
        module="nvalchemi.models.aimnet2",
    )


def configure_presentation(
    labels: tuple[str, ...],
) -> tuple[list[Atoms], pd.DataFrame]:
    """Apply shared style and load the lesson's checked molecule selection."""

    start_tutorial()
    return load_molecule_selection(labels)
