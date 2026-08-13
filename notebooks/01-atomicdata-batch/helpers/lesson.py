"""Data identity and timing support for the AtomicData and Batch lesson."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from importlib import metadata
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read, write
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

_EXPECTED_EXTXYZ_SHA256 = (
    "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
)
_EXPECTED_MANIFEST_SHA256 = (
    "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
)
MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROGRESS_UPDATE_STRUCTURES = 100


class ActiveSpinnerColumn(SpinnerColumn):
    """Spin for the active task, mark completed tasks, and quiet pending tasks."""

    def __init__(self) -> None:
        super().__init__(
            "dots",
            style="#76B900",
            finished_text=Text("✓", style="#76B900"),
        )
        self.pending_text = Text("·", style="#59636F")

    def render(self, task: Task) -> Any:
        if task.finished:
            return self.finished_text
        if task.fields.get("active", True):
            return self.spinner.render(task.get_time())
        return self.pending_text


class AlignedCountColumn(ProgressColumn):
    """Right-align completed and total work with readable separators."""

    def __init__(self) -> None:
        super().__init__(table_column=Column(justify="right", no_wrap=True))

    def render(self, task: Task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else 0
        return Text(f"{completed:,} / {total:,}", style="progress.download")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_molecule_collection(
    root: Path | None = None,
) -> tuple[list[Atoms], pd.DataFrame]:
    """Load the pinned molecule library and verify its manifest against ASE data."""

    root = _PROJECT_ROOT if root is None else root
    data_dir = root / "data" / "nci_atlas"
    extxyz_path = data_dir / "ir-molecule-library.extxyz"
    manifest_path = data_dir / "ir-molecule-library-manifest.json"

    if _sha256_file(extxyz_path) != _EXPECTED_EXTXYZ_SHA256:
        raise RuntimeError(f"Unexpected molecule-library checksum: {extxyz_path}")
    if _sha256_file(manifest_path) != _EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected molecule-manifest checksum: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["molecules"]
    atoms = list(read(extxyz_path, index=":"))
    if len(atoms) != len(records):
        raise RuntimeError("The manifest and extxyz record counts differ.")

    rows: list[dict[str, Any]] = []
    for structure, record in zip(atoms, records, strict=True):
        if len(structure) != record["atom_count"]:
            raise RuntimeError(f"Atom-count mismatch for {record['label']}.")
        if structure.get_chemical_formula() != record["formula"]:
            raise RuntimeError(f"Formula mismatch for {record['label']}.")
        structure.info["charge"] = int(record["formal_charge"])
        rows.append(
            {
                "order": int(record["order"]),
                "label": str(record["label"]),
                "formula": str(record["formula"]),
                "atoms": int(record["atom_count"]),
                "charge": int(record["formal_charge"]),
                "source": f"{record['dataset']} / {record['system_id']}",
            }
        )

    frame = pd.DataFrame(rows).sort_values("order", ignore_index=True)
    if len(frame) != 32 or int(frame["atoms"].sum()) != 322:
        raise RuntimeError("The pinned 32-molecule / 322-atom identity check failed.")
    if frame["charge"].ne(0).any():
        raise RuntimeError("This lesson expects neutral molecules only.")
    return atoms, frame


def resolve_verified_checkpoint(alias: str, expected_sha256: str) -> Path:
    """Resolve one AIMNet registry alias and verify the downloaded checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(alias)).resolve()
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"Checkpoint checksum mismatch: {digest}")
    return path


def model_checkpoint() -> Path:
    """Return the verified checkpoint selected for this lesson."""

    return resolve_verified_checkpoint(MODEL_ALIAS, _MODEL_SHA256)


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze model parameters while leaving input gradients available."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def timing_progress(*, show_elapsed: bool = True) -> Progress:
    """Create the shared notebook progress display for repeated model calls."""

    columns = [
        ActiveSpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(
            bar_width=32,
            style="#1F2933",
            complete_style="#76B900",
            finished_style="#76B900",
        ),
        AlignedCountColumn(),
    ]
    if show_elapsed:
        columns.append(TimeElapsedColumn())
    return Progress(*columns)


def use_plot_style() -> None:
    """Load the shared ALCHEMI Matplotlib style."""

    import matplotlib.pyplot as plt

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")


def start_tutorial() -> None:
    """Apply presentation settings used by this notebook."""

    use_plot_style()
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True",
        category=UserWarning,
        module="nvalchemi.models.aimnet2",
    )


def atoms_to_xyz(atoms: Atoms) -> str:
    """Serialize one ASE structure for MatterViz without adding a periodic cell."""

    stream = StringIO()
    write(stream, atoms, format="xyz")
    return stream.getvalue()


def runtime_identity(devices: Mapping[str, torch.device]) -> pd.DataFrame:
    """Return software and hardware identities for a timing result."""

    rows = [
        {"component": "PyTorch", "value": torch.__version__},
        {"component": "Toolkit", "value": metadata.version("nvalchemi-toolkit")},
        {
            "component": "Toolkit-Ops",
            "value": metadata.version("nvalchemi-toolkit-ops"),
        },
    ]
    for mode, device in devices.items():
        rows.append({"component": f"{mode} device", "value": _device_name(device)})
    return pd.DataFrame(rows)


def _device_name(device: torch.device) -> str:
    """Return a compact hardware label for a measured execution device."""

    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when timing a public model call."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_seconds(value: float) -> str:
    """Format a bar label in ordinary decimal seconds."""

    if value >= 1:
        return f"{value:.1f} s"
    if value >= 0.1:
        return f"{value:.2f} s"
    return f"{value:.3f} s"


def measure_call[T](call: Callable[[], T], device: torch.device) -> tuple[float, T]:
    """Measure one callable with synchronization outside the timed operation."""

    synchronize(device)
    started = perf_counter()
    result = call()
    synchronize(device)
    return perf_counter() - started, result


def compare_warm_calls(
    calls: Mapping[str, Callable[[], Any]],
    *,
    device: torch.device,
    structures: int,
    atoms: int,
    repeats: int = 7,
) -> pd.DataFrame:
    """Warm, time, and summarize equal-work model callables in insertion order."""

    if len(calls) < 2:
        raise ValueError("calls must contain at least two equal-work routes")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    samples: dict[str, list[float]] = {name: [] for name in calls}
    progress = timing_progress(show_elapsed=False)
    with progress:
        task = progress.add_task(
            "Warm and measure model calls", total=len(calls) * (repeats + 1)
        )
        for call in calls.values():
            call()
            progress.update(task, advance=1, refresh=True)
        for name, call in calls.items():
            for _ in range(repeats):
                elapsed, _ = measure_call(call, device)
                samples[name].append(elapsed)
                progress.update(task, advance=1, refresh=True)

    frame = pd.DataFrame(
        [
            summarize_samples(name, values, structures=structures, atoms=atoms)
            for name, values in samples.items()
        ]
    )
    frame.insert(1, "execution device", _device_name(device))
    baseline_ms = float(frame.loc[0, "warm call median (ms)"])
    frame["speedup vs individual calls"] = baseline_ms / frame["warm call median (ms)"]
    return frame


def compare_device_calls(
    calls: Mapping[str, tuple[torch.device, Callable[[], Any]]],
    *,
    structures: int,
    atoms: int,
    repeats: int = 3,
) -> pd.DataFrame:
    """Measure equal model calls on each device and report CUDA memory."""

    if not calls:
        raise ValueError("calls must contain at least one measured route")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    rows: list[dict[str, float | str]] = []
    progress = timing_progress()
    with progress:
        task = progress.add_task(
            "Warm and measure CPU/GPU calls", total=len(calls) * (repeats + 1)
        )
        for mode, (device, call) in calls.items():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            measure_call(call, device)
            progress.update(task, advance=1, refresh=True)
            samples = []
            for _ in range(repeats):
                elapsed, _ = measure_call(call, device)
                samples.append(elapsed)
                progress.update(task, advance=1, refresh=True)

            row = summarize_samples(mode, samples, structures=structures, atoms=atoms)
            row["hardware"] = _device_name(device)
            row["peak CUDA memory (MiB)"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else float("nan")
            )
            row["CUDA memory used (%)"] = (
                100
                * torch.cuda.max_memory_allocated(device)
                / torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else float("nan")
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    cpu_ms = float(frame.loc[frame["mode"] == "CPU", "warm call median (ms)"].iloc[0])
    frame["speedup vs CPU"] = cpu_ms / frame["warm call median (ms)"]
    return frame


@torch.enable_grad()
def show_batched_speedup(
    molecules: Sequence[Atoms],
    *,
    batch_size: int = 2_048,
    seed: int = 7,
    repeats: int = 1,
) -> pd.DataFrame:
    """Benchmark individual and batched model calls on CPU and GPU."""

    import matplotlib.pyplot as plt
    from IPython.display import display
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models import AIMNet2Wrapper
    from nvalchemi.neighbors import compute_neighbors

    if not torch.cuda.is_available():
        raise RuntimeError("This opening benchmark requires a CUDA device.")
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and repeats must be positive")

    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    sample_indices = np.random.default_rng(seed).integers(
        len(molecules), size=batch_size
    )
    graphs = [AtomicData.from_atoms(molecule, device=cpu) for molecule in molecules]
    sampled_graphs = [graphs[int(index)] for index in sample_indices]
    cpu_batch = Batch.from_data_list(sampled_graphs, device=cpu)
    gpu_batch = cpu_batch.to(gpu)
    cpu_singles = [Batch.from_data_list([graph], device=cpu) for graph in graphs]
    gpu_singles = [Batch.from_data_list([graph], device=gpu) for graph in graphs]

    models = {}
    for name, device in (("CPU", cpu), ("GPU", gpu)):
        model = AIMNet2Wrapper.from_checkpoint(model_checkpoint(), device=device).eval()
        freeze_model(model).set_config("active_outputs", {"energy", "forces"})
        models[name] = model

    compute_neighbors(cpu_batch, config=models["CPU"].model_config.neighbor_config)
    compute_neighbors(gpu_batch, config=models["GPU"].model_config.neighbor_config)
    for batch in cpu_singles:
        compute_neighbors(batch, config=models["CPU"].model_config.neighbor_config)
    for batch in gpu_singles:
        compute_neighbors(batch, config=models["GPU"].model_config.neighbor_config)

    for name, batch in (
        ("CPU", cpu_singles[0]),
        ("GPU", gpu_singles[0]),
        ("CPU", cpu_batch),
        ("GPU", gpu_batch),
    ):
        models[name](batch)

    progress = timing_progress()
    progress_heading = f"Energy + forces · {batch_size:,} molecules"
    if repeats > 1:
        progress_heading += f" · {repeats} repeats"
    task_descriptions = {
        "CPU · serial": f"CPU · {batch_size:,} single-molecule calls",
        "GPU · serial": f"GPU · {batch_size:,} single-molecule calls",
        "CPU · batch": f"CPU · 1 batch of {batch_size:,} molecules",
        "GPU · batch": f"GPU · 1 batch of {batch_size:,} molecules",
    }
    task_totals = {
        "CPU · serial": batch_size * repeats,
        "GPU · serial": batch_size * repeats,
        "CPU · batch": repeats,
        "GPU · batch": repeats,
    }
    progress_tasks: dict[str, int] = {}

    def serial_call(
        mode: str, model: torch.nn.Module, singles: Sequence[Batch]
    ) -> Any:
        result = None
        pending = 0
        update_every = min(_PROGRESS_UPDATE_STRUCTURES, batch_size)
        for position, index in enumerate(sample_indices, start=1):
            result = model(singles[int(index)])
            pending += 1
            if pending == update_every or position == batch_size:
                progress.update(
                    progress_tasks[mode], advance=pending, refresh=True
                )
                pending = 0
        return result

    routes = {
        "CPU · serial": (
            cpu,
            lambda: serial_call("CPU · serial", models["CPU"], cpu_singles),
        ),
        "GPU · serial": (
            gpu,
            lambda: serial_call("GPU · serial", models["GPU"], gpu_singles),
        ),
        "CPU · batch": (cpu, lambda: models["CPU"](cpu_batch)),
        "GPU · batch": (gpu, lambda: models["GPU"](gpu_batch)),
    }
    rows = []
    overall = Progress(
        TextColumn("[bold]{task.description}"),
        TimeElapsedColumn(),
        console=progress.console,
    )
    overall_task = overall.add_task(progress_heading, total=None)
    with Live(
        Group(overall, progress),
        console=progress.console,
        refresh_per_second=10,
    ) as live:
        for mode in routes:
            progress_tasks[mode] = progress.add_task(
                task_descriptions[mode],
                total=task_totals[mode],
                active=False,
                start=False,
            )
        for mode, (device, call) in routes.items():
            progress.start_task(progress_tasks[mode])
            progress.update(progress_tasks[mode], active=True, refresh=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                elapsed, _ = measure_call(call, device)
                samples.append(elapsed)
                if "serial" not in mode:
                    progress.update(
                        progress_tasks[mode], advance=1, refresh=True
                    )
            progress.update(
                progress_tasks[mode],
                completed=task_totals[mode],
                active=False,
                refresh=True,
            )
            row = summarize_samples(
                mode, samples, structures=batch_size, atoms=cpu_batch.num_nodes
            )
            row["hardware"] = _device_name(device)
            row["peak CUDA memory (MiB)"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else float("nan")
            )
            rows.append(row)
        overall.stop_task(overall_task)
        live.refresh()

    timing = pd.DataFrame(rows).rename(columns={"structures/s": "molecules/s"})
    baseline = float(
        timing.loc[
            timing["mode"] == "GPU · serial", "warm call median (ms)"
        ].iloc[0]
    )
    timing["speedup vs GPU serial"] = baseline / timing["warm call median (ms)"]

    plot_data = timing.set_index("mode")
    plot_data["elapsed time (s)"] = plot_data["warm call median (ms)"] / 1000
    route_labels = [f"{batch_size:,} individual calls", "1 Batch call"]
    positions = np.arange(len(route_labels))
    width = 0.36
    cpu_modes = ["CPU · serial", "CPU · batch"]
    gpu_modes = ["GPU · serial", "GPU · batch"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cpu_seconds = plot_data.loc[cpu_modes, "elapsed time (s)"]
    gpu_seconds = plot_data.loc[gpu_modes, "elapsed time (s)"]
    cpu_time_bars = axes[0].bar(
        positions - width / 2,
        cpu_seconds,
        width,
        label="CPU",
        color="#00A3E0",
    )
    gpu_time_bars = axes[0].bar(
        positions + width / 2,
        gpu_seconds,
        width,
        label="GPU",
        color="#76B900",
    )
    axes[0].set(
        title="Evaluation time",
        ylabel="Elapsed time [s]",
        xticks=positions,
        xticklabels=route_labels,
    )
    axes[0].set_ylim(0, max(cpu_seconds.max(), gpu_seconds.max()) * 1.16)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].bar_label(
        cpu_time_bars,
        labels=[_format_seconds(value) for value in cpu_seconds],
        padding=3,
    )
    axes[0].bar_label(
        gpu_time_bars,
        labels=[_format_seconds(value) for value in gpu_seconds],
        padding=3,
    )
    axes[0].legend(frameon=False)
    cpu_rate_bars = axes[1].bar(
        positions - width / 2,
        plot_data.loc[cpu_modes, "molecules/s"],
        width,
        label="CPU",
        color="#00A3E0",
    )
    gpu_rate_bars = axes[1].bar(
        positions + width / 2,
        plot_data.loc[gpu_modes, "molecules/s"],
        width,
        label="GPU",
        color="#76B900",
    )
    axes[1].set(
        title="Throughput",
        ylabel="Throughput [molecules/s]",
        xticks=positions,
        xticklabels=route_labels,
    )
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].bar_label(cpu_rate_bars, fmt="%.0f", padding=3)
    axes[1].bar_label(gpu_rate_bars, fmt="%.0f", padding=3)
    gpu_memory = timing.loc[
        timing["mode"] == "GPU · batch", "peak CUDA memory (MiB)"
    ].iloc[0]
    figure.suptitle(
        f"Same {batch_size:,} molecules · {cpu_batch.num_nodes:,} atoms · "
        f"GPU peak {gpu_memory:,.0f} MiB"
    )
    figure.tight_layout()
    display(
        figure,
        metadata={
            "image/png": {
                "alt": "Linear-scale evaluation time in seconds and throughput for individual and "
                "batched CPU and GPU model calls over the same "
                f"{batch_size:,}-molecule workload."
            }
        },
    )
    plt.close(figure)
    return timing


def maximum_force_table(
    metadata_frame: pd.DataFrame,
    sampled_indices: Sequence[int],
    forces: torch.Tensor,
    atoms_per_molecule: Sequence[int],
    positions: Sequence[int],
) -> pd.DataFrame:
    """Summarize model-predicted maximum force for selected batch positions."""

    counts = [int(value) for value in atoms_per_molecule]
    force_chunks = forces.detach().cpu().split(counts)
    rows = []
    for position in positions:
        source_index = int(sampled_indices[position])
        metadata_row = metadata_frame.iloc[source_index]
        rows.append(
            {
                "batch position": int(position),
                "molecule": metadata_row["label"],
                "formula": metadata_row["formula"],
                "atoms": int(metadata_row["atoms"]),
                "maximum force (eV/Å)": float(
                    torch.linalg.vector_norm(force_chunks[position], dim=1).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_device_comparison(
    timing: pd.DataFrame,
    *,
    structures: int,
    atoms: int,
) -> Any:
    """Plot response time and throughput for equal CPU/GPU model calls."""

    import matplotlib.pyplot as plt

    plot_data = timing.set_index("mode")
    plot_data["elapsed time (s)"] = plot_data["warm call median (ms)"] / 1000
    colors = ["#76B900" if name == "GPU" else "#00A3E0" for name in plot_data.index]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    time_bars = axes[0].bar(
        plot_data.index, plot_data["elapsed time (s)"], color=colors
    )
    axes[0].set(
        title="Evaluation time",
        xlabel="Execution device",
        ylabel="Elapsed time [s]",
    )
    axes[0].set_ylim(0, plot_data["elapsed time (s)"].max() * 1.16)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].bar_label(
        time_bars,
        labels=[_format_seconds(value) for value in plot_data["elapsed time (s)"]],
        padding=3,
    )
    rate_bars = axes[1].bar(plot_data.index, plot_data["molecules/s"], color=colors)
    axes[1].set(
        title="Throughput",
        xlabel="Execution device",
        ylabel="Throughput [molecules/s]",
    )
    axes[1].bar_label(rate_bars, fmt="%.0f", padding=3)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    figure.suptitle(f"Fixed seed · {structures:,} molecules · {atoms:,} atoms")
    figure.tight_layout()
    display_alt = (
        "CPU and GPU response time and throughput for the same "
        f"{structures:,}-molecule batch."
    )
    from IPython.display import display

    display(figure, metadata={"image/png": {"alt": display_alt}})
    plt.close(figure)
    return figure


def plot_batch_ownership(
    atomic_numbers: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    labels: Sequence[str],
) -> Any:
    """Plot atom fields, coordinate slices, and molecule boundaries."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    numbers = atomic_numbers.detach().cpu().reshape(-1).tolist()
    graph_ids = batch_idx.detach().cpu().tolist()
    boundaries = batch_ptr.detach().cpu().tolist()
    if len(labels) != len(boundaries) - 1:
        raise ValueError("labels must contain one name per molecule")
    if len(numbers) != len(graph_ids):
        raise ValueError("atomic_numbers and batch_idx must have one row per atom")

    palette = ["#76B900", "#00A3E0", "#D6A94A"]
    figure, axis = plt.subplots(figsize=(14, 4), dpi=160)
    for atom_row, (number, graph_id) in enumerate(zip(numbers, graph_ids, strict=True)):
        color = palette[int(graph_id) % len(palette)]
        tile = FancyBboxPatch(
            (atom_row - 0.42, 0.48),
            0.84,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color,
            edgecolor="none",
        )
        axis.add_patch(tile)
        axis.text(
            atom_row,
            0.70,
            chemical_symbols[int(number)],
            color="#0B0F10",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    for boundary in boundaries[1:-1]:
        axis.axvline(boundary - 0.5, color="#CDD2D8", ls="--", lw=1.2)
    for graph_id, (label, start, stop) in enumerate(
        zip(labels, boundaries[:-1], boundaries[1:], strict=True)
    ):
        color = palette[graph_id % len(palette)]
        axis.add_patch(
            Rectangle(
                (start - 0.5, 0.08),
                stop - start,
                0.28,
                facecolor=color,
                edgecolor="none",
                alpha=0.75,
            )
        )
        center = (start + stop - 1) / 2
        axis.text(center, 1.18, f"{label} · {stop - start} atoms", ha="center")
        axis.text(
            center,
            0.22,
            f"positions[{start}:{stop}, :]\nbatch_idx = {graph_id}",
            color="#F3F4F6",
            ha="center",
            va="center",
            fontsize=7.5,
        )

    axis.set_xlim(-0.75, len(graph_ids) + 0.25)
    axis.set_ylim(-0.05, 1.42)
    axis.set_xticks(boundaries)
    axis.set_xlabel("Atom-row offsets stored in batch_ptr")
    axis.set_yticks([])
    axis.grid(False)
    axis.spines[["left", "right", "top"]].set_visible(False)
    figure.tight_layout()

    joined_labels = ", ".join(str(label) for label in labels)
    boundary_text = ", ".join(map(str, boundaries))
    alt = (
        f"Packed atomic-number tiles for {joined_labels}. Colored bands show "
        f"coordinate slices and batch indexes; batch_ptr is {boundary_text}."
    )
    from IPython.display import display

    display(figure, metadata={"image/png": {"alt": alt}})
    plt.close(figure)
    return figure


def molecule_result_table(
    metadata_frame: pd.DataFrame,
    indices: Sequence[int],
    energies: torch.Tensor,
) -> pd.DataFrame:
    """Format selected molecule metadata and energies for the opening result."""

    selected = metadata_frame.loc[list(indices), ["label", "formula", "atoms"]].copy()
    values = energies.detach().cpu().reshape(-1)
    selected["energy (eV)"] = values[list(indices)].numpy()
    return selected.rename(columns={"label": "molecule"})


def summarize_samples(
    mode: str,
    samples_s: Sequence[float],
    *,
    structures: int,
    atoms: int,
) -> dict[str, float | str]:
    """Summarize repeated equal-work timings for notebook display."""

    values = np.asarray(samples_s, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("samples_s must contain positive one-dimensional timings")
    median_s = float(np.median(values))
    p25_s, p75_s = np.quantile(values, [0.25, 0.75])
    return {
        "mode": mode,
        "warm call median (ms)": 1000.0 * median_s,
        "spread p25–p75 (ms)": 1000.0 * float(p75_s - p25_s),
        "structures/s": structures / median_s,
        "atoms/s": atoms / median_s,
    }
