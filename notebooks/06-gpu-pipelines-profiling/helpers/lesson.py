"""Checked inputs, measurement utilities, and visuals for Part 06."""

from __future__ import annotations

import base64
import html
import io
import json
import os
import platform
import sys
import tempfile
import tomllib
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from ase import Atoms
from ase.io import read
from IPython.display import HTML
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage, GPUBuffer
from nvalchemi.hooks import StageTimingHook

EXPECTED_EXTXYZ_SHA256 = (
    "331b087fc7ced6eaec25fae9dccba767fc47e4ae5e0523fcaddfa4fbf49c455f"
)
EXPECTED_MANIFEST_SHA256 = (
    "c3e189fb8e96a7aec8e587631659f6424fae51cf42243d9f4c54d64e5ed3207d"
)
MODEL_ALIAS = "aimnet2-wb97m-d3_0"
MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
COMPLETED_UPDATE_COLUMN = "completed update (1-based)"
REPO_ROOT = Path(__file__).resolve().parents[3]
_KNOWN_CPU_FALLBACK_NATIVE_PREFIXES = (
    "Warp CUDA error: Failed to get driver entry point",
    "Warp CUDA warning: Unable to determine CUDA driver version",
)


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


def _cpu_name() -> str:
    """Return a useful CPU identity on Linux, with portable fallbacks."""

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                return value.strip()
    return platform.processor().strip() or platform.machine()


def runtime_identity(device: torch.device) -> dict[str, Any]:
    """Return the software and hardware identity needed beside measurements."""

    pins = tomllib.loads(
        (REPO_ROOT / "environment" / "runtime-pins.toml").read_text(
            encoding="utf-8"
        )
    )
    cuda_available = torch.cuda.is_available() and device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if cuda_available else "unavailable"
    return {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "Toolkit": version("nvalchemi-toolkit"),
        "Toolkit commit": pins["toolkit"]["commit"],
        "Toolkit-Ops": version("nvalchemi-toolkit-ops"),
        "Toolkit-Ops commit": pins["toolkit_ops"]["commit"],
        "execution device": str(device),
        "CUDA available": cuda_available,
        "PyTorch CUDA build": torch.version.cuda or "unavailable",
        "GPU name": gpu_name,
        "CPU name": _cpu_name(),
    }


def start_tutorial() -> None:
    """Apply the shared plot style and suppress one known wrapper warning."""

    import matplotlib.pyplot as plt

    plt.style.use(REPO_ROOT / "shared" / "alchemi-dark.mplstyle")
    pd.set_option("display.max_colwidth", None)
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True",
        category=UserWarning,
        module="nvalchemi.models.aimnet2",
    )
    warnings.filterwarnings(
        "ignore",
        message="Can't initialize NVML",
        category=UserWarning,
        module="torch.cuda",
    )


def configure_presentation(
    labels: tuple[str, ...],
) -> tuple[list[Atoms], pd.DataFrame]:
    """Apply shared styling and load the checked molecular selection."""

    start_tutorial()
    return load_molecule_selection(labels)


def render_figure(figure: Any, *, alt_text: str) -> HTML:
    """Return bounded HTML whose embedded PNG retains explicit alt text."""

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    alt = html.escape(alt_text, quote=True)
    return HTML(
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}" '
        'style="display:block;max-width:100%;height:auto;">'
    )


def make_synthetic_graphs(
    counts: Sequence[int],
    *,
    seed: int,
) -> list[AtomicData]:
    """Build deterministic host-resident graphs with float and integer fields."""

    if not counts or any(count < 1 for count in counts):
        raise ValueError("counts must contain positive atom counts")
    graphs: list[AtomicData] = []
    for source_index, count in enumerate(counts):
        generator = torch.Generator().manual_seed(seed + source_index)
        graph = AtomicData(
            positions=torch.randn(count, 3, generator=generator),
            atomic_numbers=torch.randint(
                1,
                9,
                (count,),
                generator=generator,
                dtype=torch.long,
            ),
            atomic_masses=torch.ones(count),
            velocities=torch.zeros(count, 3),
            forces=torch.zeros(count, 3),
            energy=torch.zeros(1, 1),
        )
        graph.add_system_property(
            "source_index",
            torch.tensor([[source_index]], dtype=torch.long),
        )
        graphs.append(graph)
    return graphs


class AtomicDataset:
    """Adapt an in-memory graph sequence to the sampler dataset contract."""

    def __init__(
        self,
        records: Sequence[AtomicData],
        *,
        device: torch.device,
    ) -> None:
        if not records:
            raise ValueError("records must contain at least one graph")
        self.records = tuple(records)
        self.device = device

    def __len__(self) -> int:
        return len(self.records)

    def get_metadata(self, index: int) -> tuple[int, int]:
        record = self.records[index]
        return int(record.num_nodes), int(record.num_edges)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, int]]:
        graph = self.records[index].clone().to(self.device)
        return graph, {"source_index": index}


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when the selected device requires it."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def filter_known_native_stderr(device: torch.device) -> Iterator[None]:
    """Suppress known CPU-only Warp noise while preserving other diagnostics."""

    sys.stderr.flush()
    saved_stderr = os.dup(2)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as captured:
        os.dup2(captured.fileno(), 2)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Warning: Profiler clears events",
                    category=UserWarning,
                )
                yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            captured.seek(0)
            lines = captured.readlines()
            unexpected_lines = (
                lines
                if device.type == "cuda"
                else [
                    line
                    for line in lines
                    if not line.startswith(_KNOWN_CPU_FALLBACK_NATIVE_PREFIXES)
                ]
            )
            if unexpected_lines:
                sys.stderr.writelines(unexpected_lines)
                sys.stderr.flush()


def time_callable[T](
    function: Callable[..., T],
    *args: Any,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[T, pd.DataFrame]:
    """Measure synchronized wall time after a caller-selected warm-up."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    result: T
    for _ in range(warmup):
        result = function(*args)
        synchronize(device)
    rows: list[dict[str, float | int]] = []
    for sample in range(1, repeats + 1):
        synchronize(device)
        started = perf_counter()
        result = function(*args)
        synchronize(device)
        rows.append(
            {
                "sample": sample,
                "elapsed (ms)": 1000.0 * (perf_counter() - started),
            }
        )
    return result, pd.DataFrame(rows)


def device_memory_snapshot(device: torch.device) -> dict[str, float | bool | None]:
    """Read current and peak CUDA allocator counters in MiB."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "CUDA memory available": False,
            "allocated (MiB)": None,
            "reserved (MiB)": None,
            "peak allocated (MiB)": None,
        }
    mib = 1024.0**2
    return {
        "CUDA memory available": True,
        "allocated (MiB)": torch.cuda.memory_allocated(device) / mib,
        "reserved (MiB)": torch.cuda.memory_reserved(device) / mib,
        "peak allocated (MiB)": torch.cuda.max_memory_allocated(device) / mib,
    }


class PipelineMonitor:
    """Record public fields at AFTER_STEP, before fixed-step status migration."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.rows: list[dict[str, float | int | None]] = []

    def __call__(
        self,
        context: Any,
        stage: DynamicsStage | None = None,
    ) -> None:
        observed_stage = stage if stage is not None else getattr(context, "stage", None)
        if observed_stage is not DynamicsStage.AFTER_STEP:
            return
        batch = context.batch
        status = batch.status.detach().reshape(-1)
        memory = device_memory_snapshot(self.device)
        self.rows.append(
            {
                COMPLETED_UPDATE_COLUMN: int(context.step_count) + 1,
                "active systems": int(batch.num_graphs),
                "active atoms": int(batch.num_nodes),
                "status 0": int(status.eq(0).sum().cpu()),
                "status 1": int(status.eq(1).sum().cpu()),
                "status 2": int(status.eq(2).sum().cpu()),
                "allocated (MiB)": memory["allocated (MiB)"],
                "reserved (MiB)": memory["reserved (MiB)"],
            }
        )

    def frame(self) -> pd.DataFrame:
        """Return observations in stable column order."""

        return pd.DataFrame(
            self.rows,
            columns=[
                COMPLETED_UPDATE_COLUMN,
                "active systems",
                "active atoms",
                "status 0",
                "status 1",
                "status 2",
                "allocated (MiB)",
                "reserved (MiB)",
            ],
        )


def stage_timing_frame(hook: StageTimingHook) -> pd.DataFrame:
    """Shape the public AFTER_STEP timing samples for display and plotting."""

    values = hook.timings[DynamicsStage.AFTER_STEP]
    return pd.DataFrame(
        {
            COMPLETED_UPDATE_COLUMN: range(1, len(values) + 1),
            "step time (ms)": [1000.0 * value for value in values],
        }
    )


def _plot_groups(axis: Any, timings: pd.DataFrame) -> None:
    labels = list(dict.fromkeys(timings["implementation"].astype(str)))
    colors = ("#7C8794", "#00A3E0", "#76B900")
    for position, label in enumerate(labels):
        values = timings.loc[
            timings["implementation"].eq(label), "elapsed (ms)"
        ].to_numpy()
        axis.scatter(
            [position] * len(values),
            values,
            color=colors[position % len(colors)],
            s=38,
            alpha=0.85,
        )
        axis.hlines(
            float(pd.Series(values).median()),
            position - 0.22,
            position + 0.22,
            color="#F3F4F6",
            linewidth=1.4,
        )
    axis.set_xticks(range(len(labels)), labels, rotation=12, ha="right")


def plot_measurement_panels(
    timings: pd.DataFrame,
    memory: dict[str, float | bool | None],
) -> Any:
    """Plot synchronized call timing and CUDA allocator evidence."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    _plot_groups(axes[0], timings)
    axes[0].set(
        ylabel="Synchronized wall time (ms, log scale)",
        title="First-call and warm samples",
    )
    axes[0].set_yscale("log")
    axes[1].set(ylabel="CUDA memory (MiB)", title="PyTorch CUDA allocator")
    if memory["CUDA memory available"]:
        names = ("allocated", "reserved", "peak allocated")
        values = [memory[f"{name} (MiB)"] for name in names]
        axes[1].bar(names, values, color=("#7C8794", "#00A3E0", "#76B900"))
        axes[1].tick_params(axis="x", rotation=12)
    else:
        axes[1].text(
            0.5,
            0.5,
            "GPU measurement unavailable\nCPU fallback executed",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_pipeline_diagnostics(
    stage_timings: pd.DataFrame,
    occupancy: pd.DataFrame,
) -> Any:
    """Plot stage timing beside active-system and active-atom occupancy."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].plot(
        stage_timings[COMPLETED_UPDATE_COLUMN],
        stage_timings["step time (ms)"],
        color="#76B900",
        marker="o",
        linewidth=1.8,
    )
    axes[0].set(
        xlabel="Completed update (1-based)",
        ylabel="Stage time (ms)",
        title="BEFORE_STEP → AFTER_STEP",
    )
    axes[1].plot(
        occupancy[COMPLETED_UPDATE_COLUMN],
        occupancy["active systems"],
        color="#76B900",
        marker="o",
        label="systems",
    )
    axes[1].plot(
        occupancy[COMPLETED_UPDATE_COLUMN],
        occupancy["active atoms"],
        color="#00A3E0",
        marker="s",
        label="atoms",
    )
    axes[1].set(
        xlabel="Completed update (1-based)",
        ylabel="Active count",
        title="Batch occupancy before refill",
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    plt.close(figure)
    return figure


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
        raise RuntimeError("The Part 06 molecular selection expects neutral systems.")
    return selected_atoms, frame


class MoleculeDataset:
    """Build checked molecular records lazily on the pipeline execution device."""

    def __init__(
        self,
        atoms: Sequence[Atoms],
        metadata: pd.DataFrame,
        *,
        device: torch.device | None = None,
    ) -> None:
        if len(atoms) != len(metadata) or not atoms:
            raise ValueError("atoms and metadata must have the same non-zero length")
        self.atoms = tuple(atoms)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.device = torch.device("cpu") if device is None else device

    def __len__(self) -> int:
        return len(self.atoms)

    def get_metadata(self, index: int) -> tuple[int, int]:
        return len(self.atoms[index]), 0

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, int | str]]:
        graph = AtomicData.from_atoms(
            self.atoms[index],
            device=self.device,
            dtype=torch.float32,
        )
        graph.add_system_property(
            "charge",
            torch.zeros(1, 1, device=self.device),
        )
        graph.add_system_property(
            "energy",
            torch.zeros(1, 1, device=self.device),
        )
        graph.add_node_property(
            "forces",
            torch.zeros_like(graph.positions),
        )
        graph.add_system_property(
            "source_index",
            torch.tensor([[index]], dtype=torch.long, device=self.device),
        )
        graph.use_default_velocities()
        return graph, {
            "source_index": index,
            "label": str(self.metadata.iloc[index]["label"]),
        }


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


def per_system_fmax(batch: Batch) -> torch.Tensor:
    """Return the maximum force norm for each graph in a batch."""

    norms = torch.linalg.vector_norm(batch.forces.detach(), dim=1)
    maxima = torch.zeros(
        batch.num_graphs,
        dtype=norms.dtype,
        device=norms.device,
    )
    maxima.scatter_reduce_(
        0,
        batch.batch_idx.long(),
        norms,
        reduce="amax",
        include_self=True,
    )
    return maxima


def summarize_collected(batch: Batch, metadata: pd.DataFrame) -> pd.DataFrame:
    """Shape collected systems in source order without scientific interpretation."""

    source_indices = batch.source_index.detach().reshape(-1).cpu().tolist()
    system_ids = batch.system_id.detach().reshape(-1).cpu().tolist()
    statuses = batch.status.detach().reshape(-1).cpu().tolist()
    energies = batch.energy.detach().reshape(-1).cpu().tolist()
    maxima = per_system_fmax(batch).cpu().tolist()
    rows: list[dict[str, Any]] = []
    for graph, source_index in enumerate(source_indices):
        record = metadata.iloc[int(source_index)]
        rows.append(
            {
                "molecule": str(record["label"]),
                "formula": str(record["formula"]),
                "atoms": int(record["atoms"]),
                "system ID": int(system_ids[graph]),
                "final status": int(statuses[graph]),
                "energy (eV)": float(energies[graph]),
                "maximum force (eV/Å)": float(maxima[graph]),
            }
        )
    return pd.DataFrame(rows).sort_values("molecule", key=lambda values: values.map(
        {label: index for index, label in enumerate(metadata["label"])}
    )).reset_index(drop=True)


def profile_artifacts(directory: Path) -> pd.DataFrame:
    """List generated profiler files relative to their requested output root."""

    rows = [
        {
            "relative path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return pd.DataFrame(rows, columns=["relative path", "bytes"])


def probe_gpu_buffer_mixed_dtype(device: torch.device) -> dict[str, Any]:
    """Probe the pin-specific float/integer preservation boundary of GPUBuffer."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "probe ran": False,
            "reason": "GPUBuffer requires CUDA",
            "float positions preserved": None,
            "integer atomic numbers preserved": None,
            "integer source indices preserved": None,
        }
    source = Batch.from_data_list(make_synthetic_graphs((2,), seed=61)).to(device)
    # A nonzero marker prevents a skipped, zero-filled integer field from passing.
    source.source_index.fill_(17)
    buffer = GPUBuffer(capacity=1, max_atoms=2, max_edges=0, device=device)
    buffer.write(source)
    collected = buffer.read()
    return {
        "probe ran": True,
        "reason": "executed at the pinned Toolkit commit",
        "float positions preserved": torch.equal(
            source.positions,
            collected.positions,
        ),
        "integer atomic numbers preserved": torch.equal(
            source.atomic_numbers,
            collected.atomic_numbers,
        ),
        "integer source indices preserved": torch.equal(
            source.source_index,
            collected.source_index,
        ),
    }
