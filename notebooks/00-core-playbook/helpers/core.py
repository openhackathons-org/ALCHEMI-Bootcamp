# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked data, tiny API sandboxes, and bounded visuals for the Core playbook."""

from __future__ import annotations

import json
import os
import platform
import re
import tomllib
import warnings
from base64 import b64encode
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import BadZipFile, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase import units as ase_units
from ase.build import molecule as ase_molecule
from ase.collections import s22
from ase.data import chemical_symbols
from ase.io import write
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COLORS = json.loads(
    (_PROJECT_ROOT / "shared" / "alchemi-palette.json").read_text(encoding="utf-8")
)
_BACKGROUND = _COLORS["background"]
_SURFACE = _COLORS["surface"]
_SURFACE_RAISED = _COLORS["surface_raised"]
_BORDER = _COLORS["border"]
_GRID = _COLORS["grid"]
_TEXT = _COLORS["text"]
_MUTED = _COLORS["muted"]
_QUIET = _COLORS["quiet"]
_INK_ON_ACCENT = _COLORS["ink_on_accent"]
_NVIDIA_GREEN = _COLORS["nvidia_green"]
_NVIDIA_BLUE = _COLORS["nvidia_blue"]
_NVIDIA_TEAL = _COLORS["nvidia_teal"]
_NVIDIA_ORANGE = _COLORS["nvidia_orange"]
_TUTORIAL_VISUAL_MAX_WIDTH = 920
_TUTORIAL_COMPACT_VISUAL_MAX_WIDTH = 720
_TUTORIAL_SMALL_VISUAL_MAX_WIDTH = 640
_TUTORIAL_PIXELS_PER_INCH = 100
_TUTORIAL_FIGURE_WIDTH = _TUTORIAL_VISUAL_MAX_WIDTH / _TUTORIAL_PIXELS_PER_INCH
_TUTORIAL_COMPACT_FIGURE_WIDTH = (
    _TUTORIAL_COMPACT_VISUAL_MAX_WIDTH / _TUTORIAL_PIXELS_PER_INCH
)
_TUTORIAL_SMALL_FIGURE_WIDTH = (
    _TUTORIAL_SMALL_VISUAL_MAX_WIDTH / _TUTORIAL_PIXELS_PER_INCH
)
_TUTORIAL_BODY_FONT_SIZE = 8.0
_TUTORIAL_MICRO_FONT_SIZE = 7.2
_TUTORIAL_VIEWER_HEIGHT = 320
_MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_RUNTIME_PINS = _PROJECT_ROOT / "environment" / "runtime-pins.toml"
_NCI_DATA_FILE = (
    _PROJECT_ROOT
    / "notebooks"
    / "00-core-playbook"
    / "data"
    / "nci_atlas"
    / "nci-atlas-curves.csv.gz"
)
NCI_ATLAS_SHA256 = "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
_NCIA250_ARCHIVE = (
    _PROJECT_ROOT
    / "notebooks"
    / "00-core-playbook"
    / "data"
    / "nci_atlas"
    / "NCIA250.zip"
)
NCIA250_SHA256 = "34e3c2cec763344dd9be41aa008672c7d052e50db57abe1abc59873d3935c433"
COULOMB_CONSTANT_EV_ANGSTROM = 14.399645351950548
_EV_TO_KCAL_MOL = 1.0 / (ase_units.kcal / ase_units.mol)
_NCI_SCALES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50, 2.00)
_NCI_SYSTEMS = {
    "1.041": ("phenol - N-methylacetamide", "neutral hydrogen bond"),
    "1.07.74": ("propyne - methyl azide", "dispersion-dominated"),
    "08.007": ("ammonia - benzoate", "ionic hydrogen bond"),
}
_RECURRING_NCI_COMPLEXES = (
    ("Propyne–methyl azide", "propyne - methyl azide"),
    ("Ammonia–benzoate", "ammonia - benzoate"),
    ("Phenol–N-methylacetamide", "phenol - N-methylacetamide"),
)
_NCI_CURVE_KEYS = (
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
)
_NCI_GRAPH_COLUMNS = (
    *_NCI_CURVE_KEYS,
    "fragment",
    "charge",
    "natoms",
    "source_gradient_block",
    "source_geometry_file",
)
_NCI_REQUIRED_COLUMNS = (
    *_NCI_GRAPH_COLUMNS,
    "symbols",
    "positions_angstrom",
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    "ccsd_t_cbs_interaction_energy_kcal_mol",
)
_ELEMENT_COLORS = {
    1: "#E8ECEF",
    6: "#5C6770",
    7: "#00A3E0",
    8: "#E05252",
    18: "#80D1E3",
}
_TUTORIAL_FORMULA_PREFIX = ("C", "N", "O")


def load_example_molecule(name: str) -> Atoms:
    """Return a fresh ASE molecule used by the Core modules."""

    if name == "Ammonia":
        structure = ase_molecule("NH3")
    elif name == "Propyne":
        structure = ase_molecule("C3H4_C3v")
    elif name == "Phenol":
        structure = s22["Phenol_dimer"][:13]
    else:
        expected = "Ammonia, Propyne, or Phenol"
        raise ValueError(f"Unknown example molecule {name!r}; expected {expected}")
    return structure.copy()


def _exercise_is_pending(*values: Any) -> bool:
    """Print the shared learner prompt when an exercise still has placeholders."""

    if any(value is None for value in values):
        print("Complete the exercise cell, then run this check again.")
        return True
    return False


def check_amide_anion(amide_anion: AtomicData | None) -> bool:
    """Check the Module 1 charged-structure exercise."""

    if _exercise_is_pending(amide_anion):
        return False
    assert amide_anion is not None

    atom_count = amide_anion.num_nodes
    charge = getattr(amide_anion, "charge", None)
    if charge is None:
        print("Keep editing: add the system charge with add_system_property(...).")
        return False

    net_charge = float(charge.item())
    print(f"{atom_count} atoms | charge {net_charge:+.0f} e")
    passed = atom_count == 3 and net_charge == -1.0
    print("Check passed." if passed else "Keep editing: expected 3 atoms and charge -1 e.")
    return passed


def check_model_wrapper_exercise(
    wrapper: BaseModelMixin | None,
    batch: Batch | None,
    outputs: Mapping[str, torch.Tensor] | None,
) -> bool:
    """Check that Module 2 evaluated a batch with MACEWrapper."""

    if _exercise_is_pending(wrapper, batch, outputs):
        return False
    assert wrapper is not None and batch is not None and outputs is not None

    wrapper_name = type(wrapper).__name__
    energy = outputs.get("energy")
    forces = outputs.get("forces")
    energy_shape = tuple(energy.shape) if energy is not None else None
    force_shape = tuple(forces.shape) if forces is not None else None
    print(f"{wrapper_name} | energy {energy_shape} | forces {force_shape}")

    passed = (
        wrapper_name == "MACEWrapper"
        and energy_shape == (batch.num_graphs, 1)
        and force_shape == (batch.num_nodes, 3)
    )
    message = (
        "Check passed."
        if passed
        else "Keep editing: use MACEWrapper and return one energy per system and one force row per atom."
    )
    print(message)
    return passed


def check_force_drop_exercise(
    force_drop: Mapping[str, float] | None,
    largest_drop: str | None,
    labels: Sequence[str],
) -> bool:
    """Check the Module 2 force-history exercise."""

    if _exercise_is_pending(force_drop, largest_drop):
        return False
    assert force_drop is not None and largest_drop is not None

    passed = set(force_drop) == set(labels) and largest_drop in labels
    if not passed:
        print("Keep editing: calculate one force decrease per molecule and name the largest.")
        return False

    print("Force decrease [eV/Å]")
    for label, decrease in force_drop.items():
        print(f"  {label:10} {decrease:+.4f}")
    print(f"Largest decrease: {largest_drop}")
    print("Check passed.")
    return True




_ELEMENT_FALLBACK_COLOR = _NVIDIA_GREEN
_VIEWER_BUNDLE = _PROJECT_ROOT / "shared" / "3dmol-2.5.5" / "3Dmol-min.js"
_VIEWER_BACKGROUND = _BACKGROUND
_VIEWER_CELL_COLOR = _BORDER
_PROGRESS_UPDATE_STRUCTURES = 100


class _ActiveSpinnerColumn(SpinnerColumn):
    """Spin for the current route and keep the other routes quiet."""

    def __init__(self) -> None:
        super().__init__(
            "dots",
            style=_NVIDIA_GREEN,
            finished_text=Text("✓", style=_NVIDIA_GREEN),
        )
        self.pending_text = Text("·", style="#59636F")

    def render(self, task: Task) -> Any:
        if task.finished:
            return self.finished_text
        if task.fields.get("active", False):
            return self.spinner.render(task.get_time())
        return self.pending_text


class _AlignedCountColumn(ProgressColumn):
    """Right-align completed and total work."""

    def __init__(self) -> None:
        super().__init__(table_column=Column(justify="right", no_wrap=True))

    def render(self, task: Task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else 0
        return Text(f"{completed:,} / {total:,}", style="progress.download")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_checkpoint() -> Path:
    """Resolve and verify the pinned AIMNet2 checkpoint."""

    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(_MODEL_ALIAS)).resolve()
    digest = _sha256_file(path)
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"AIMNet2 checkpoint checksum mismatch: {digest}")
    return path


def aimnet_checkpoint_species() -> tuple[int, ...]:
    """Read the implemented atomic numbers from the verified checkpoint."""

    checkpoint = torch.load(model_checkpoint(), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("AIMNet2 checkpoint does not contain model metadata")
    values = checkpoint.get("implemented_species")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("AIMNet2 checkpoint does not list implemented species")
    species = tuple(int(value) for value in values)
    if (
        not species
        or len(species) != len(set(species))
        or any(value <= 0 or value >= len(chemical_symbols) for value in species)
    ):
        raise RuntimeError("AIMNet2 checkpoint contains invalid implemented species")
    return species


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze parameters while preserving gradients with respect to positions."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@contextmanager
def suppress_known_aimnet_load_warning():
    """Hide the one-time cutoff conversion warning in the pinned AIMNet wrapper."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting a tensor with requires_grad=True to a scalar.*",
            category=UserWarning,
        )
        yield


def d3_parameter_file() -> Path:
    """Resolve and verify the D3 element-reference tables prepared by ``scripts/setup``."""

    configured = os.environ.get("ALCHEMI_D3_PARAM_FILE")
    if not configured:
        raise RuntimeError("ALCHEMI_D3_PARAM_FILE is not configured")
    path = Path(configured).expanduser().resolve()
    pins = tomllib.loads(_RUNTIME_PINS.read_text(encoding="utf-8"))
    expected = str(pins["dispersion"]["generated_parameter_sha256"])
    if not path.is_file():
        raise RuntimeError(f"D3 parameter file is missing: {path}")
    digest = _sha256_file(path)
    if digest != expected:
        raise RuntimeError(
            f"D3 parameter file checksum mismatch: {digest}; expected {expected}"
        )
    return path


class _NativeCoulomb(torch.nn.Module):
    """Finite 1/r electrostatic energy expressed as a native PyTorch module."""

    def __init__(self, coulomb_constant: float) -> None:
        super().__init__()
        self.coulomb_constant = float(coulomb_constant)

    def forward(
        self,
        positions: torch.Tensor,
        partial_charges: torch.Tensor,
        neighbor_pairs: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        source, target = neighbor_pairs.unbind(dim=1)
        source = source.to(torch.long)
        target = target.to(torch.long)
        charges = partial_charges.reshape(-1)
        distances = torch.linalg.vector_norm(
            positions[source] - positions[target], dim=-1
        )
        pair_energy = (
            0.5 * self.coulomb_constant * charges[source] * charges[target] / distances
        )
        energy = torch.zeros(
            num_graphs,
            dtype=positions.dtype,
            device=positions.device,
        )
        energy.index_add_(0, batch_idx[source].to(torch.long), pair_energy)
        return energy.unsqueeze(-1)


class DirectCoulombAdapter(torch.nn.Module, BaseModelMixin):
    """Adapt finite 1/r electrostatics to Toolkit model composition.

    ``partial_charges`` comes from an earlier model step. A full COO neighbor
    list includes both pair directions, and the native module assigns half of
    each ordered-pair energy to avoid double counting.
    """

    def __init__(
        self,
        cutoff: float,
        coulomb_constant: float = COULOMB_CONSTANT_EV_ANGSTROM,
    ) -> None:
        super().__init__()
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be finite and positive")
        self.cutoff = float(cutoff)
        self.native = _NativeCoulomb(coulomb_constant)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"partial_charges"}),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        del data, kwargs
        raise NotImplementedError("Direct Coulomb has no learned embeddings")

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data], device=data.positions.device)
        inputs = super().adapt_input(data, **kwargs)
        inputs.pop("atomic_numbers", None)
        inputs["neighbor_pairs"] = inputs.pop("neighbor_list")
        inputs["batch_idx"] = data.batch_idx
        inputs["num_graphs"] = data.num_graphs
        return inputs

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> OrderedDict[str, torch.Tensor]:
        inputs = self.adapt_input(data, **kwargs)
        energy = self.native(**inputs)
        output: dict[str, torch.Tensor] = {"energy": energy}
        if "forces" in self.model_config.active_outputs:
            forces = -torch.autograd.grad(
                energy.sum(),
                inputs["positions"],
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            output["forces"] = forces
        return self.adapt_output(output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        del path, as_state_dict
        raise NotImplementedError("This tutorial wrapper has no export format")


def prepare_dynamics_batch(batch: Batch) -> Batch:
    """Clone a batch and allocate the fields updated by dynamics workflows."""

    prepared = batch.clone()
    # Dynamics owns a fresh autograd graph for each compute step.
    prepared.positions = prepared.positions.detach().clone()
    dtype = prepared.positions.dtype
    device = prepared.device
    counts = prepared.num_nodes_per_graph.tolist()
    if "energy" not in prepared:
        prepared.add_key(
            "energy",
            [torch.zeros((1, 1), dtype=dtype, device=device) for _ in counts],
            level="system",
        )
    if "forces" not in prepared:
        prepared.add_key(
            "forces",
            [torch.zeros((count, 3), dtype=dtype, device=device) for count in counts],
            level="node",
        )
    if "velocities" not in prepared:
        prepared.add_key(
            "velocities",
            [torch.zeros((count, 3), dtype=dtype, device=device) for count in counts],
            level="node",
        )
    return prepared


class FusedStatusTrace:
    """Record per-system stage codes before each fused iteration."""

    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(self) -> None:
        self.rows: list[tuple[int, list[int]]] = []

    def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
        codes = ctx.batch.status.reshape(-1).detach().cpu().tolist()
        self.rows.append((ctx.step_count, [int(code) for code in codes]))


class InflightStatusPrinter:
    """Print live membership and status counts for a fused workflow."""

    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(
        self,
        sink: Any,
        stage_labels: dict[int, str],
        *,
        sampler: Any | None = None,
        system_labels: dict[int, str] | None = None,
    ) -> None:
        self.sink = sink
        self.stage_labels = stage_labels
        self.sampler = sampler
        self.system_labels = system_labels or {}
        self._header_printed = False
        self._previous_ids: tuple[int, ...] | None = None

    def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
        if not self._header_printed:
            print(
                f"{'step':>4}  {'active molecules':<30} "
                f"{'waiting':>7} {'done':>5}  event"
            )
            self._header_printed = True

        status = ctx.batch.status.reshape(-1)
        stage_codes = [int(value) for value in status.detach().cpu().tolist()]
        active_ids = tuple(
            int(value)
            for value in ctx.batch.system_id.reshape(-1).detach().cpu().tolist()
        )
        active_text = " ".join(
            f"{self.system_labels.get(system_id, str(system_id))}:"
            f"{self.stage_labels.get(code, str(code))}"
            for system_id, code in zip(active_ids, stage_codes, strict=True)
        )
        waiting = len(self.sampler) if self.sampler is not None else 0

        if self._previous_ids is None:
            event = "initial fill"
        else:
            entered = [value for value in active_ids if value not in self._previous_ids]
            left = [value for value in self._previous_ids if value not in active_ids]
            events = []
            if entered:
                entered_text = ",".join(
                    self.system_labels.get(value, str(value)) for value in entered
                )
                events.append(f"refill +[{entered_text}]")
            if left:
                left_text = ",".join(
                    self.system_labels.get(value, str(value)) for value in left
                )
                events.append(f"finished [{left_text}]")
            event = " | ".join(events)

        print(
            f"{ctx.step_count:>4}  {active_text:<30} "
            f"{waiting:>7} {len(self.sink):>5}  {event}"
        )
        self._previous_ids = active_ids


@contextmanager
def measure_torch_execution(
    device: torch.device | str,
) -> Iterator[dict[str, float | None]]:
    """Measure synchronized elapsed time and peak allocated CUDA memory."""

    run_device = torch.device(device)
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
        torch.cuda.reset_peak_memory_stats(run_device)

    measurement: dict[str, float | None] = {
        "seconds": None,
        "peak_memory_mib": None,
    }
    start = perf_counter()
    try:
        yield measurement
    finally:
        if run_device.type == "cuda":
            torch.cuda.synchronize(run_device)
        measurement["seconds"] = perf_counter() - start
        if run_device.type == "cuda":
            measurement["peak_memory_mib"] = (
                torch.cuda.max_memory_allocated(run_device) / 1024**2
            )


def benchmark_progress(total: int, *, description: str) -> Iterator[int]:
    """Yield benchmark steps with one updateable notebook progress row."""

    if total < 1:
        raise ValueError("benchmark progress needs at least one step")

    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        shell = get_ipython()
        in_notebook = (
            shell is not None
            and shell.__class__.__name__ == "ZMQInteractiveShell"
        )
    except ImportError:
        in_notebook = False

    if in_notebook:
        def progress_html(completed: int) -> Any:
            finished = completed == total
            fraction = completed / total
            marker = "✓" if finished else "◌"
            animation = "" if finished else "animation:alchemi-spin 0.9s linear infinite;"
            return HTML(
                "<style>@keyframes alchemi-spin{to{transform:rotate(360deg)}}</style>"
                '<div style="display:flex;align-items:center;gap:10px;max-width:760px;'
                'font:13px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;'
                f'color:{_TEXT};background:{_BACKGROUND};padding:7px 10px;">'
                f'<span style="display:inline-block;color:{_NVIDIA_GREEN};{animation}">'
                f"{marker}</span>"
                f"<span style=\"min-width:250px\">{escape(description)}</span>"
                f'<span style="height:6px;flex:1;background:{_SURFACE_RAISED};'
                'border-radius:999px;overflow:hidden">'
                f'<span style="display:block;height:100%;width:{fraction:.3%};'
                f'background:{_NVIDIA_GREEN}"></span></span>'
                f'<span style="min-width:110px;text-align:right">'
                f"{completed:,} / {total:,}</span></div>"
            )

        handle = display(progress_html(0), display_id=True)
        for index in range(total):
            yield index
            completed = index + 1
            if (
                completed % _PROGRESS_UPDATE_STRUCTURES == 0 or completed == total
            ) and handle is not None:
                handle.update(progress_html(completed))
        return

    progress = Progress(
        _ActiveSpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(
            bar_width=28,
            style=_SURFACE_RAISED,
            complete_style=_NVIDIA_GREEN,
            finished_style=_NVIDIA_GREEN,
        ),
        _AlignedCountColumn(),
        TimeElapsedColumn(),
        refresh_per_second=10,
    )
    task = progress.add_task(description, total=total)
    with progress:
        for index in range(total):
            yield index
            completed = index + 1
            if completed % _PROGRESS_UPDATE_STRUCTURES == 0 or completed == total:
                progress.update(task, completed=completed, refresh=True)


def summarize_batching_benchmark(
    *,
    device: torch.device | str,
    molecule: str,
    molecules: int,
    atoms: int,
    individual: Mapping[str, float | None],
    batch: Mapping[str, float | None],
    batch_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Shape and print the two-route batching benchmark result."""

    individual_seconds = float(individual["seconds"])
    batch_seconds = float(batch["seconds"])
    route_device = "GPU" if torch.device(device).type == "cuda" else "CPU"
    results = {
        "device": route_device,
        "molecules": int(molecules),
        "atoms": int(atoms),
        "molecule": molecule,
        "individual_seconds": individual_seconds,
        "batch_seconds": batch_seconds,
        "individual_peak_mib": individual["peak_memory_mib"],
        "batch_peak_mib": batch["peak_memory_mib"],
    }

    print(
        f"Energy + forces | {molecules:,} {molecule} molecules | "
        f"{atoms:,} atoms | {route_device}"
    )
    for label, seconds, peak_memory in (
        ("individual calls", individual_seconds, individual["peak_memory_mib"]),
        ("one Batch call", batch_seconds, batch["peak_memory_mib"]),
    ):
        memory_text = (
            f" | peak allocated {float(peak_memory):,.0f} MiB"
            if peak_memory is not None
            else ""
        )
        print(
            f"{label:18} {seconds:8.2f} s | "
            f"{molecules / seconds:,.0f} molecules/s{memory_text}"
        )
    print(
        f"Batch outputs | energy {batch_output['energy'].shape} | "
        f"forces {batch_output['forces'].shape}"
    )
    return results


def benchmark_repeated_molecule(
    atoms: Atoms,
    *,
    batch_size: int = 2048,
    device: torch.device | str = "cuda",
    molecule_name: str | None = None,
) -> list[dict[str, Any]]:
    """Compare individual and batched model calls on CPU and CUDA."""

    from nvalchemi.models import AIMNet2Wrapper
    from nvalchemi.neighbors import compute_neighbors

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")

    def load_model(run_device: torch.device) -> AIMNet2Wrapper:
        with suppress_known_aimnet_load_warning():
            model = AIMNet2Wrapper.from_checkpoint(
                model_checkpoint(), device=run_device, compile_model=False
            ).eval()
        freeze_model(model)
        model.set_config("active_outputs", {"energy", "forces"})
        return model

    route_devices = {"CPU": torch.device("cpu")}
    if target.type == "cuda":
        route_devices["GPU"] = target

    models: dict[str, AIMNet2Wrapper] = {}
    singles: dict[str, Batch] = {}
    batches: dict[str, Batch] = {}
    for label, run_device in route_devices.items():
        model = load_model(run_device)
        graph = AtomicData.from_atoms(atoms, device=run_device)
        single = Batch.from_data_list([graph], device=run_device)
        batch = Batch.from_data_list([graph] * batch_size, device=run_device)
        compute_neighbors(single, config=model.model_config.neighbor_config)
        compute_neighbors(batch, config=model.model_config.neighbor_config)
        _ = model(single)
        _ = model(batch)
        models[label], singles[label], batches[label] = model, single, batch

    routes = []
    for label in route_devices:
        routes.append((f"{label} · individual", label, "individual", batch_size))
    for label in route_devices:
        routes.append((f"{label} · Batch", label, "Batch", 1))

    progress = Progress(
        _ActiveSpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(
            bar_width=28,
            style=_SURFACE_RAISED,
            complete_style=_NVIDIA_GREEN,
            finished_style=_NVIDIA_GREEN,
        ),
        _AlignedCountColumn(),
        TimeElapsedColumn(),
        refresh_per_second=10,
    )
    overall = Progress(
        TextColumn("[bold]{task.description}"),
        TimeElapsedColumn(),
        console=progress.console,
    )
    overall_task = overall.add_task(
        f"Energy + forces · {batch_size:,} molecules", total=None
    )
    tasks = {
        mode: progress.add_task(
            f"{label} · "
            + (
                f"{batch_size:,} individual calls"
                if route == "individual"
                else f"1 Batch call · {batch_size:,} molecules"
            ),
            total=total,
            active=False,
            start=False,
        )
        for mode, label, route, total in routes
    }

    atoms_per_molecule = len(atoms)
    display_name = molecule_name or atoms.get_chemical_formula()
    rows: list[dict[str, Any]] = []
    with Live(
        Group(overall, progress),
        console=progress.console,
        refresh_per_second=10,
        transient=True,
    ):
        for mode, label, route, total in routes:
            run_device = route_devices[label]
            progress.start_task(tasks[mode])
            progress.update(tasks[mode], active=True, refresh=True)
            if run_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(run_device)
            start = perf_counter()
            if route == "individual":
                output = None
                pending = 0
                for position in range(1, batch_size + 1):
                    output = models[label](singles[label])
                    pending += 1
                    if pending == _PROGRESS_UPDATE_STRUCTURES or position == batch_size:
                        progress.update(tasks[mode], advance=pending, refresh=True)
                        pending = 0
            else:
                output = models[label](batches[label])
                progress.update(tasks[mode], advance=1, refresh=True)
            if run_device.type == "cuda":
                torch.cuda.synchronize(run_device)
            seconds = perf_counter() - start
            progress.update(tasks[mode], completed=total, active=False, refresh=True)
            peak_memory = (
                torch.cuda.max_memory_allocated(run_device) / 1024**2
                if run_device.type == "cuda"
                else None
            )
            rows.append(
                {
                    "mode": mode,
                    "route": route,
                    "device": label,
                    "molecules": batch_size,
                    "atoms": batch_size * atoms_per_molecule,
                    "molecule": display_name,
                    "seconds": seconds,
                    "molecules/s": batch_size / seconds,
                    "peak memory [MiB]": peak_memory,
                    "energy shape": tuple(output["energy"].shape),
                    "forces shape": tuple(output["forces"].shape),
                }
            )
        overall.stop_task(overall_task)
    return rows


def plot_batching_benchmark(
    results: Mapping[str, Any] | Sequence[dict[str, Any]],
) -> Any:
    """Plot elapsed time and throughput for the batching comparison."""

    if isinstance(results, Mapping):
        batch_size = int(results["molecules"])
        device = str(results["device"])
        rows = [
            {
                "route": route,
                "device": device,
                "molecules": batch_size,
                "atoms": int(results["atoms"]),
                "molecule": str(results["molecule"]),
                "seconds": float(results[seconds_key]),
                "molecules/s": batch_size / float(results[seconds_key]),
                "peak memory [MiB]": (
                    results["batch_peak_mib"] if route == "Batch" else None
                ),
            }
            for route, seconds_key in (
                ("individual", "individual_seconds"),
                ("Batch", "batch_seconds"),
            )
        ]
    else:
        rows = list(results)

    batch_size = int(rows[0]["molecules"])
    positions = np.arange(2)
    width = 0.34
    route_labels = [f"{batch_size:,} individual calls", "1 Batch call"]
    devices = [
        name for name in ("CPU", "GPU") if any(r["device"] == name for r in rows)
    ]
    colors = {"CPU": _NVIDIA_BLUE, "GPU": _NVIDIA_GREEN}
    offsets = (
        {"CPU": -width / 2, "GPU": width / 2}
        if len(devices) == 2
        else {devices[0]: 0.0}
    )
    figure, axes = plt.subplots(1, 2, figsize=(_TUTORIAL_FIGURE_WIDTH, 3.8))
    for axis, key, title, ylabel in (
        (axes[0], "seconds", "Evaluation time", "Elapsed time [s]"),
        (axes[1], "molecules/s", "Throughput", "Molecules/s"),
    ):
        for label in devices:
            selected = {row["route"]: row for row in rows if row["device"] == label}
            values = [float(selected[route][key]) for route in ("individual", "Batch")]
            bars = axis.bar(
                positions + offsets[label],
                values,
                width,
                label=label,
                color=colors[label],
            )
            value_labels = (
                [f"{value:,.2f} s" for value in values]
                if key == "seconds"
                else [f"{value:,.0f}" for value in values]
            )
            axis.bar_label(bars, labels=value_labels, padding=3)
        axis.set(
            title=title,
            ylabel=ylabel,
            xticks=positions,
            xticklabels=route_labels,
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.legend(frameon=False)
        _polish_axis(axis, grid="y")
        axis.margins(y=0.18)
    gpu_batch = next(
        (row for row in rows if row["device"] == "GPU" and row["route"] == "Batch"),
        None,
    )
    memory_text = (
        f" · GPU allocated peak {gpu_batch['peak memory [MiB]']:,.0f} MiB"
        if gpu_batch is not None
        else ""
    )
    molecule = str(rows[0].get("molecule", "molecules"))
    figure.suptitle(
        f"Same {batch_size:,} {molecule} molecules · "
        f"{int(rows[0]['atoms']):,} atoms{memory_text}"
    )
    device_text = " and ".join(devices)
    return _display_figure(
        figure,
        "Linear-scale evaluation time and throughput for individual and batched "
        f"model calls on {device_text} over the same repeated-molecule workload.",
    )


def configure_tutorial() -> None:
    """Apply the shared tutorial style and silence a one-time library notice."""

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")
    _prime_tree_set_notice()


def _prime_tree_set_notice() -> None:
    """Consume dm-tree's one-time set notice outside learner-facing cells."""

    import tree

    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            tree.flatten({"energy", "forces"})
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def atoms_to_xyz(atoms: Atoms) -> str:
    """Serialize one ASE structure as XYZ text for the 3Dmol.js viewer."""

    stream = StringIO()
    write(stream, atoms, format="xyz")
    return stream.getvalue()


def _viewing_rotation(positions: np.ndarray) -> np.ndarray:
    """Return a rigid rotation that turns a structure to face the default camera.

    3Dmol.js looks down z, so anything built along z is drawn end-on: ASE's
    linear `molecule("C2H2")` collapses to a single visible atom. Aligning the
    two broadest principal axes with the screen fixes that for any structure,
    and picks the same framing `plot_molecule` projects onto, so the interactive
    view and the flat projection agree.
    """

    centered = positions - positions.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=True)
    if np.linalg.det(axes) < 0:
        axes[2] *= -1.0  # keep the viewing frame right-handed
    return axes


def _to_view_frame(
    points: np.ndarray, origin: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Apply one viewing transform to atom positions and lattice corners alike."""

    return (np.atleast_2d(points) - origin) @ rotation.T


def _cell_edge_points(cell: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the twelve edges of a lattice parallelepiped anchored at the origin."""

    vectors = np.asarray(cell, dtype=float)
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for axis in range(3):
        first, second = (vectors[index] for index in range(3) if index != axis)
        for along_first, along_second in ((0, 0), (1, 0), (0, 1), (1, 1)):
            corner = along_first * first + along_second * second
            edges.append((corner, corner + vectors[axis]))
    return edges


def _viewer_point(position: np.ndarray) -> dict[str, float]:
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def _tutorial_formula(atoms: Atoms) -> str:
    """Format viewer labels with C, N, and O first and hydrogen last."""

    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1

    ordered = [symbol for symbol in _TUTORIAL_FORMULA_PREFIX if symbol in counts]
    ordered.extend(
        sorted(symbol for symbol in counts if symbol not in {*ordered, "H"})
    )
    if "H" in counts:
        ordered.append("H")

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in ordered
    )


def _zoom_factor(framed: np.ndarray, *, fill: float = 0.82) -> float:
    """Return the extra zoom that undoes 3Dmol's minimum framing distance.

    `zoomTo` never frames anything closer than its `minimumZoomToDistance`, a
    10 Å field of view that py3Dmol gives no way to reconfigure, so a four-atom
    molecule lands as a speck in the middle of the frame. `zoomTo` sizes that
    field to twice the distance from the bounding-box centre to the outermost
    point, so scaling by that ratio makes the structure span `fill` of the
    viewport's short side instead.
    """

    lower, upper = framed.min(axis=0), framed.max(axis=0)
    radius = float(np.linalg.norm(framed - (lower + upper) / 2, axis=1).max())
    if radius < 1.0e-9:
        return 1.0  # a lone atom has no extent to frame
    return max(radius, 5.0) * fill / radius


def _viewer_html(
    view: Any,
    height: int,
    *,
    formula: str,
    atom_count: int,
) -> str:
    """Wrap py3Dmol's snippet in a compact, self-contained structure widget.

    py3Dmol always emits a `<script src=...>` pointing at jsDelivr, guarded by
    `if (typeof $3Dmolpromise === 'undefined')`. Defining that promise ahead of
    the snippet satisfies the guard, so the vendored bundle inlined here is the
    only copy of 3Dmol.js the page ever sees. Inlining rather than linking is
    what keeps the view alive inside a self-contained `nbconvert` export.
    """

    bundle = _VIEWER_BUNDLE.read_text(encoding="utf-8")
    # `var define, exports, module` shadows the CommonJS and AMD hooks the UMD
    # bundle probes. nbconvert's HTML template loads RequireJS, and a defined
    # `define.amd` would park the bundle in a module registry that nothing here
    # requires, leaving `window.$3Dmol` undefined and the viewer blank.
    preamble = (
        "<script>(function(){var define,exports,module;\n"
        f"{bundle}\n"
        "$3Dmolpromise=Promise.resolve();}).call(window);</script>"
    )
    viewer_markup = view.write_html()
    warning = re.search(
        r'<p id="(3dmolwarning_[^"]+)"[^>]*>.*?</p>',
        viewer_markup,
        flags=re.DOTALL,
    )
    if warning:
        warning_id = escape(warning.group(1), quote=True)
        runtime_notice = f"""
<p id="{warning_id}" data-runtime-note
   style="box-sizing:border-box; height:100%; margin:0; padding:28px;
          display:flex; flex-direction:column; justify-content:center;
          align-items:center; gap:10px; text-align:center;
          background:{_VIEWER_BACKGROUND}; color:{_TEXT};">
  <strong style="font-size:1rem;">Interactive viewer</strong>
  <span style="max-width:560px; color:{_MUTED}; font-size:0.88rem; line-height:1.5;">
    GitHub shows this notebook as a static preview. Open it in Jupyter,
    connect to a runtime, and run this cell.
  </span>
  <a href="../../README.md#install-and-run"
     style="color:{_NVIDIA_GREEN}; font-size:0.84rem; font-weight:650;">
    Runtime setup
  </a>
</p>
""".strip()
        viewer_markup = (
            viewer_markup[: warning.start()]
            + runtime_notice
            + viewer_markup[warning.end() :]
        )
    formula_text = escape(formula)
    return f"""
<div class="alchemi-structure-widget"
     aria-label="Interactive structure viewer for {formula_text}"
     style="box-sizing:border-box; width:100%; max-width:{_TUTORIAL_VISUAL_MAX_WIDTH}px;
            margin:0; overflow:hidden; border-radius:10px;
            background:{_BACKGROUND}; color:{_TEXT};
            font-family:'NVIDIA Sans',Arial,sans-serif;">
  <div style="display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between;
              gap:16px; padding:12px 16px 10px; background:{_SURFACE};">
    <div style="display:flex; align-items:baseline; gap:10px; min-width:0;">
      <strong style="font-size:1.05rem; font-weight:650;">{formula_text}</strong>
      <span style="color:{_MUTED}; font-size:0.84rem;">{atom_count} atoms</span>
    </div>
    <span style="color:{_MUTED}; font-size:0.78rem; text-align:right;">
      Drag to rotate · scroll to zoom · select up to 3 atoms
    </span>
  </div>
  <div class="alchemi-structure-view"
       style="width:100%; height:{height}px; overflow:hidden;
              background:{_VIEWER_BACKGROUND};">
    {preamble}{viewer_markup}
  </div>
  <div style="display:flex; align-items:center; justify-content:space-between;
              gap:14px; padding:10px 16px 12px; background:{_SURFACE};">
    <div data-atom-detail role="status" aria-live="polite"
         style="min-height:22px; color:{_TEXT}; font-size:0.84rem;
                font-variant-numeric:tabular-nums;">
      Select 1 atom to inspect, 2 for a distance, or 3 for an angle.
    </div>
    <button type="button" data-clear-measurement disabled
            style="flex:none; border:1px solid {_BORDER}; border-radius:6px;
                   padding:5px 9px; background:{_SURFACE_RAISED}; color:{_TEXT};
                   font:inherit; font-size:0.76rem; cursor:pointer;">
      Clear
    </button>
  </div>
</div>
"""


def _atom_click_callback(atoms: Atoms) -> str:
    """Return the 3Dmol.js callback for atom selection and measurements."""

    atomic_numbers = {
        chemical_symbols[number]: int(number)
        for number in sorted({int(number) for number in atoms.get_atomic_numbers()})
    }
    number_map = json.dumps(atomic_numbers, sort_keys=True)
    return f"""function(atom, viewer, event, container) {{
      const numbers = {number_map};
      const host = container && container.jquery ? container[0] : container;
      const root = host && host.closest
        ? host.closest('.alchemi-structure-widget')
        : null;
      const atomIndex = Number(atom.serial);
      const atomicNumber = numbers[atom.elem];
      const detail = root ? root.querySelector('[data-atom-detail]') : null;
      const clearButton = root
        ? root.querySelector('[data-clear-measurement]')
        : null;
      const state = viewer.__alchemiMeasurement || {{
        atoms: [],
        shapes: [],
        labels: []
      }};
      viewer.__alchemiMeasurement = state;

      const reset = function(message) {{
        state.shapes.forEach(function(shape) {{ viewer.removeShape(shape); }});
        state.labels.forEach(function(label) {{ viewer.removeLabel(label); }});
        state.atoms = [];
        state.shapes = [];
        state.labels = [];
        if (detail) {{
          detail.textContent = message
            || 'Select 1 atom to inspect, 2 for a distance, or 3 for an angle.';
        }}
        if (clearButton) {{ clearButton.disabled = true; }}
        viewer.render();
      }};

      if (clearButton && !clearButton.__alchemiBound) {{
        clearButton.__alchemiBound = true;
        clearButton.addEventListener('click', function() {{ reset(); }});
      }}
      if (state.atoms.length === 3) {{ reset(''); }}
      if (state.atoms.some(function(selected) {{
        return Number(selected.serial) === atomIndex;
      }})) {{
        if (detail) {{
          detail.textContent = 'Row ' + atomIndex
            + ' is already selected. Choose a different atom.';
        }}
        return;
      }}

      state.atoms.push(atom);
      state.shapes.push(viewer.addSphere({{
        center: atom,
        radius: 0.34,
        color: '{_NVIDIA_GREEN}',
        opacity: 0.30
      }}));
      state.labels.push(viewer.addLabel(
        atom.elem + ' · row ' + atomIndex,
        {{
          position: atom,
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.92,
          fontColor: '{_TEXT}',
          fontSize: 11,
          borderColor: '{_NVIDIA_GREEN}',
          borderThickness: 1,
          inFront: true
        }}
      ));
      if (clearButton) {{ clearButton.disabled = false; }}

      const distance = function(first, second) {{
        return Math.hypot(
          first.x - second.x,
          first.y - second.y,
          first.z - second.z
        );
      }};
      const midpoint = function(first, second) {{
        return {{
          x: (first.x + second.x) / 2,
          y: (first.y + second.y) / 2,
          z: (first.z + second.z) / 2
        }};
      }};

      if (state.atoms.length === 1) {{
        if (detail) {{
          detail.textContent = 'Selected row ' + atomIndex + ' · ' + atom.elem
            + ' · atomic number ' + atomicNumber
            + '. Select a second atom for a distance.';
        }}
      }} else if (state.atoms.length === 2) {{
        const first = state.atoms[0];
        const second = state.atoms[1];
        const value = distance(first, second);
        state.shapes.push(viewer.addLine({{
          start: first,
          end: second,
          color: '{_NVIDIA_BLUE}',
          dashed: true,
          dashLength: 0.10,
          gapLength: 0.06
        }}));
        state.labels.push(viewer.addLabel(value.toFixed(3) + ' Å', {{
          position: midpoint(first, second),
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.94,
          fontColor: '{_TEXT}',
          fontSize: 12,
          borderColor: '{_NVIDIA_BLUE}',
          borderThickness: 1,
          inFront: true
        }}));
        if (detail) {{
          detail.textContent = 'Distance rows ' + Number(first.serial) + '–'
            + Number(second.serial) + ': ' + value.toFixed(3)
            + ' Å. Select a third atom; row ' + Number(second.serial)
            + ' will be the angle vertex.';
        }}
      }} else {{
        const first = state.atoms[0];
        const vertex = state.atoms[1];
        const third = state.atoms[2];
        const ux = first.x - vertex.x;
        const uy = first.y - vertex.y;
        const uz = first.z - vertex.z;
        const vx = third.x - vertex.x;
        const vy = third.y - vertex.y;
        const vz = third.z - vertex.z;
        const denominator = Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz);
        const cosine = Math.max(
          -1,
          Math.min(1, (ux * vx + uy * vy + uz * vz) / denominator)
        );
        const angle = Math.acos(cosine) * 180 / Math.PI;
        state.shapes.push(viewer.addLine({{
          start: vertex,
          end: third,
          color: '{_NVIDIA_BLUE}',
          dashed: true,
          dashLength: 0.10,
          gapLength: 0.06
        }}));
        state.labels.push(viewer.addLabel(angle.toFixed(1) + '°', {{
          position: vertex,
          screenOffset: {{x: 16, y: -22}},
          backgroundColor: '{_SURFACE}',
          backgroundOpacity: 0.94,
          fontColor: '{_TEXT}',
          fontSize: 12,
          borderColor: '{_NVIDIA_BLUE}',
          borderThickness: 1,
          inFront: true
        }}));
        if (detail) {{
          detail.textContent = 'Angle rows ' + Number(first.serial) + '–'
            + Number(vertex.serial) + '–' + Number(third.serial) + ': '
            + angle.toFixed(1) + '° · row ' + Number(vertex.serial)
            + ' is the vertex. Click another atom to start again.';
        }}
      }}
      viewer.render();
    }}"""


def show_molecule(
    atoms: Atoms,
    *,
    height: int = _TUTORIAL_VIEWER_HEIGHT,
    sphere_scale: float = 0.23,
) -> Any:
    """Return a compact 3Dmol.js ball-and-stick view of one ASE structure.

    3Dmol.js perceives connectivity from the coordinates alone, so no explicit
    bond list is passed in. A periodic structure also gets its lattice drawn as
    twelve edges: 3Dmol's `addUnitCell` reads crystal metadata that XYZ input
    cannot carry, so it draws nothing here.

    The structure is turned to face the camera before it is handed over. The
    rigid rotation preserves the distances and angles measured in the viewer.
    """

    import py3Dmol
    from IPython.display import HTML

    positions = atoms.get_positions()
    origin = positions.mean(axis=0)
    rotation = _viewing_rotation(positions)
    oriented = atoms.copy()
    oriented.set_positions(_to_view_frame(positions, origin, rotation))

    view = py3Dmol.view(
        width="100%",
        height=height,
        # Never fetched: `_viewer_html` pre-resolves py3Dmol's loader promise.
        # Kept as the repository-relative provenance of the inlined bundle.
        js="shared/3dmol-2.5.5/3Dmol-min.js",
    )
    view.addModel(atoms_to_xyz(oriented), "xyz")
    # One `setStyle` per element rather than a single `colorscheme` map: 3Dmol
    # 2.5.5 tests a custom map with `scheme[element]` but reads the colour back
    # from `scheme.map[element]`, so the map never applies and carbon keeps its
    # near-white default next to hydrogen.
    for number in sorted({int(number) for number in atoms.get_atomic_numbers()}):
        color = _ELEMENT_COLORS.get(number, _ELEMENT_FALLBACK_COLOR)
        view.setStyle(
            {"elem": chemical_symbols[number]},
            {
                "sphere": {"scale": sphere_scale, "color": color},
                "stick": {"radius": 0.09, "color": color},
            },
        )
    view.setClickable({}, True, _atom_click_callback(atoms))
    framed = [oriented.get_positions()]
    if atoms.pbc.any():
        for edge in _cell_edge_points(atoms.cell):
            start, end = _to_view_frame(np.asarray(edge), origin, rotation)
            framed.append(np.stack([start, end]))
            view.addLine(
                {
                    "start": _viewer_point(start),
                    "end": _viewer_point(end),
                    "color": _VIEWER_CELL_COLOR,
                }
            )
    view.setBackgroundColor(_VIEWER_BACKGROUND)
    view.zoomTo()
    view.zoom(_zoom_factor(np.concatenate(framed)))
    # A fixed oblique tilt, so a linear or planar structure and a lattice box all
    # still read as three-dimensional instead of as a rod or a flat outline.
    view.rotate(-24, "y")
    view.rotate(16, "x")
    return HTML(
        _viewer_html(
            view,
            height,
            formula=_tutorial_formula(atoms),
            atom_count=len(atoms),
        )
    )


def show_argon_batch(
    batch: Batch, *, height: int = _TUTORIAL_VIEWER_HEIGHT
) -> Any:
    """Show the first periodic argon system in a compact interactive viewer."""

    data = batch.get_data(0)
    atoms = Atoms(
        numbers=data.atomic_numbers.detach().cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
        cell=data.cell.detach().cpu().numpy().reshape(3, 3),
        pbc=data.pbc.detach().cpu().numpy().reshape(3),
    )
    return show_molecule(atoms, height=height, sphere_scale=0.42)


def _polish_axis(axis: Any, *, grid: str | None = "y") -> None:
    """Apply the restrained Core playbook scientific-figure treatment."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is None:
        axis.grid(False)
    else:
        axis.grid(axis=grid, color=_GRID, alpha=0.58, linewidth=0.75)
    axis.tick_params(length=3.5, width=0.8)
    axis.title.set_fontweight("bold")


def _figure_html(
    figure: Any,
    alt: str,
    *,
    max_width: int = _TUTORIAL_VISUAL_MAX_WIDTH,
    align: str = "left",
) -> str:
    if align not in {"left", "center"}:
        raise ValueError("align must be 'left' or 'center'")
    figure.tight_layout(pad=1.15)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    payload = b64encode(buffer.getvalue()).decode("ascii")
    width_style = f"width:100%;max-width:{max_width}px;"
    margin_style = "margin:0;" if align == "left" else "margin:0 auto;"
    return (
        '<img src="data:image/png;base64,'
        f'{payload}" alt="{escape(alt, quote=True)}" '
        f'style="display:block;box-sizing:border-box;{width_style}'
        f'height:auto;{margin_style}">'
    )


def _display_figure(
    figure: Any,
    alt: str,
    *,
    max_width: int = _TUTORIAL_VISUAL_MAX_WIDTH,
    align: str = "left",
) -> None:
    from IPython.display import HTML, display

    display(HTML(_figure_html(figure, alt, max_width=max_width, align=align)))
    plt.close(figure)


def build_argon_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> Batch:
    """Build the periodic 27-atom argon sandbox used by the official NVE example."""

    sigma = 3.40
    spacing = 2 ** (1 / 6) * sigma
    coords = (torch.arange(3, device=device, dtype=dtype) + 0.5) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    graph = AtomicData(
        positions=positions,
        atomic_numbers=torch.full(
            (len(positions),), 18, device=device, dtype=torch.long
        ),
        atomic_masses=torch.full((len(positions),), 39.948, device=device, dtype=dtype),
        velocities=torch.zeros_like(positions),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), device=device, dtype=dtype),
        cell=torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * (3 * spacing),
        pbc=torch.ones((1, 3), device=device, dtype=torch.bool),
    )
    return Batch.from_data_list([graph], device=device)


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




def _parse_nci_structure(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[str], np.ndarray, int]:
    try:
        atom_count_value = float(record["natoms"])
        symbols = str(record["symbols"]).split()
        positions = np.fromstring(str(record["positions_angstrom"]), sep=" ")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid structure fields") from error
    if not np.isfinite(atom_count_value) or not atom_count_value.is_integer():
        raise ValueError(f"{label} has a non-integer atom count")
    atom_count = int(atom_count_value)
    if atom_count <= 0 or len(symbols) != atom_count:
        raise ValueError(f"{label} atom count does not match its symbols")
    if positions.size != 3 * atom_count or not np.isfinite(positions).all():
        raise ValueError(f"{label} atom count does not match its coordinates")
    return symbols, positions.reshape(atom_count, 3), atom_count


def _validate_nci_atlas(table: pd.DataFrame) -> None:
    missing = set(_NCI_REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"NCI Atlas table is missing {sorted(missing)!r}")
    if len(table) != 90:
        raise ValueError(f"NCI Atlas table has {len(table)} rows; expected 90")

    numeric_columns = (
        "scale",
        "charge",
        "natoms",
        "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
        "ccsd_t_cbs_interaction_energy_kcal_mol",
    )
    for column in numeric_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains a non-finite value")

    observed_systems = {
        str(row.system_id): (str(row.system_name), str(row.interaction_class))
        for row in table[["system_id", "system_name", "interaction_class"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_systems != _NCI_SYSTEMS:
        raise ValueError("NCI Atlas system identities do not match the tutorial subset")
    if table.duplicated(["system_id", "scale", "fragment"]).any():
        raise ValueError("system, scale, and fragment keys must be unique")
    if table["source_gradient_block"].nunique() != 90:
        raise ValueError("source gradient-block identifiers must be unique")

    for system_id, group in table.groupby("system_id", sort=False):
        scales = np.sort(pd.to_numeric(group["scale"]).unique())
        if scales.shape != (10,) or not np.allclose(
            scales,
            _NCI_SCALES,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{system_id} does not contain the expected ten scales")

    grouped = table.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)
    if grouped.ngroups != 30:
        raise ValueError(f"NCI Atlas table has {grouped.ngroups} curves; expected 30")
    for key, group in grouped:
        label = f"{key[1]} at scale {key[-1]:g}"
        if len(group) != 3 or set(group["fragment"]) != {"AB", "A", "B"}:
            raise ValueError(f"{label} must contain one AB, A, and B row")
        records = {str(row.fragment): row._asdict() for row in group.itertuples()}
        structures = {
            fragment: _parse_nci_structure(record, label=f"{label} {fragment}")
            for fragment, record in records.items()
        }
        ab_symbols, ab_positions, ab_count = structures["AB"]
        a_symbols, a_positions, a_count = structures["A"]
        b_symbols, b_positions, b_count = structures["B"]
        if ab_count != a_count + b_count or ab_symbols != a_symbols + b_symbols:
            raise ValueError(f"{label} monomer symbols do not reconstruct the dimer")
        if not np.array_equal(ab_positions, np.vstack((a_positions, b_positions))):
            raise ValueError(
                f"{label} monomer coordinates do not reconstruct the dimer"
            )
        charges = {
            fragment: float(record["charge"]) for fragment, record in records.items()
        }
        if any(not value.is_integer() for value in charges.values()):
            raise ValueError(f"{label} contains a non-integer charge")
        if int(charges["AB"]) != int(charges["A"] + charges["B"]):
            raise ValueError(f"{label} fragment charges do not sum to the dimer charge")
        if group["ccsd_t_cbs_interaction_energy_kcal_mol"].nunique() != 1:
            raise ValueError(f"{label} fragments do not share the CCSD(T)/CBS value")


def _nci_row_to_atoms(record: Mapping[str, Any]) -> Atoms:
    label = str(record.get("source_gradient_block", "NCI Atlas row"))
    symbols, positions, _ = _parse_nci_structure(record, label=label)
    charge = float(record["charge"])
    if not np.isfinite(charge) or not charge.is_integer():
        raise ValueError(f"{label} has a non-integer total charge")
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    for field in (
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "scale",
        "fragment",
        "source_gradient_block",
        "source_geometry_file",
    ):
        value = record[field]
        atoms.info[field] = value.item() if isinstance(value, np.generic) else value
    atoms.info["charge"] = int(charge)
    return atoms


def _nci_reference_curves(table: pd.DataFrame) -> pd.DataFrame:
    curve_order = table[list(_NCI_CURVE_KEYS)].drop_duplicates(ignore_index=True)
    curve_index = pd.MultiIndex.from_frame(curve_order)
    dft_rows = table[list(_NCI_CURVE_KEYS) + ["fragment"]].copy()
    dft_rows["total"] = pd.to_numeric(
        table["wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"]
    )
    dft_wide = dft_rows.pivot(
        index=list(_NCI_CURVE_KEYS),
        columns="fragment",
        values="total",
    ).reindex(curve_index)
    references = curve_order.copy()
    references["dft_d3_kcal_mol"] = (
        dft_wide["AB"] - dft_wide["A"] - dft_wide["B"]
    ).to_numpy(dtype=float)
    cc = table.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)[
        "ccsd_t_cbs_interaction_energy_kcal_mol"
    ].first()
    references["ccsd_t_cbs_kcal_mol"] = cc.reindex(curve_index).to_numpy(dtype=float)
    return references


def load_nci_atlas(
    path: str | Path | None = None,
    *,
    expected_sha256: str = NCI_ATLAS_SHA256,
) -> tuple[list[Atoms], pd.DataFrame, pd.DataFrame]:
    """Load the checked 90-structure NCI Atlas tutorial subset.

    Returns the ASE structures in file order, row-aligned graph metadata, and
    30 reference interaction energies. The DFT-D3 reference is calculated from
    the stored frozen-geometry AB, A, and B total energies.
    """

    data_path = _NCI_DATA_FILE if path is None else Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"NCI Atlas subset is missing: {data_path}")
    observed = _sha256_file(data_path)
    if observed != expected_sha256:
        raise ValueError(
            f"NCI Atlas subset SHA-256 mismatch: {observed}; expected {expected_sha256}"
        )
    table = pd.read_csv(
        data_path,
        dtype={
            "subset": "string",
            "system_id": "string",
            "system_name": "string",
            "interaction_class": "string",
            "fragment": "string",
            "symbols": "string",
            "positions_angstrom": "string",
            "source_gradient_block": "string",
            "source_geometry_file": "string",
        },
    )
    _validate_nci_atlas(table)
    atoms = [_nci_row_to_atoms(row) for row in table.to_dict(orient="records")]
    graph_rows = table[list(_NCI_GRAPH_COLUMNS)].copy().reset_index(drop=True)
    graph_rows.insert(0, "graph_index", np.arange(len(graph_rows), dtype=np.int64))
    return atoms, graph_rows, _nci_reference_curves(table)


def nci_equilibrium_references(
    references: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Return the equilibrium reference energies for each tutorial curve."""

    required = {
        "system_name",
        "interaction_class",
        "scale",
        "dft_d3_kcal_mol",
        "ccsd_t_cbs_kcal_mol",
    }
    missing = required - set(references.columns)
    if missing:
        raise ValueError(f"references are missing {sorted(missing)!r}")
    equilibrium = references[np.isclose(references["scale"], 1.0)]
    if len(equilibrium) != references["system_name"].nunique():
        raise ValueError("each NCI curve must have one equilibrium reference")
    return [
        {
            "system_name": str(row.system_name),
            "interaction_class": str(row.interaction_class),
            "dft_d3_kcal_mol": float(row.dft_d3_kcal_mol),
            "ccsd_t_cbs_kcal_mol": float(row.ccsd_t_cbs_kcal_mol),
        }
        for row in equilibrium.itertuples(index=False)
    ]


def select_nci_equilibrium_complex(
    atoms: Sequence[Atoms],
    graph_rows: pd.DataFrame,
    system_name: str,
) -> Atoms:
    """Return one equilibrium AB complex selected by its NCI Atlas name."""

    required = {"system_name", "scale", "fragment", "graph_index"}
    missing = required - set(graph_rows.columns)
    if missing:
        raise ValueError(f"NCI graph rows are missing {sorted(missing)!r}")
    mask = (
        graph_rows["system_name"].eq(system_name)
        & graph_rows["scale"].eq(1.0)
        & graph_rows["fragment"].eq("AB")
    )
    matches = graph_rows.loc[mask, "graph_index"]
    if len(matches) != 1:
        raise ValueError(f"Expected one equilibrium AB record for {system_name!r}")
    graph_index = int(matches.iloc[0])
    if graph_index < 0 or graph_index >= len(atoms):
        raise ValueError(f"graph_index {graph_index} is outside the structure list")
    return atoms[graph_index].copy()


def load_recurring_nci_complexes() -> OrderedDict[str, Atoms]:
    """Return three equilibrium complexes reused for composition and scaling."""

    atoms, graph_rows, _ = load_nci_atlas()
    selected: OrderedDict[str, Atoms] = OrderedDict()
    for label, system_name in _RECURRING_NCI_COMPLEXES:
        selected[label] = select_nci_equilibrium_complex(
            atoms, graph_rows, system_name
        )
    return selected


def _ncia250_metadata(text: str, *, label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            raise ValueError(f"{label} contains malformed metadata {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError(f"{label} contains malformed metadata {token!r}")
        fields[key] = value
    required = {
        "charge",
        "charge_a",
        "charge_b",
        "selection_a",
        "selection_b",
        "scaling",
        "benchmark_Eint",
        "benchmark_unit",
        "group",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label} metadata is missing {sorted(missing)!r}")
    return fields


def _ncia250_selection(value: str, natoms: int, *, label: str) -> tuple[int, ...]:
    parts = value.split("-")
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2:
        raise ValueError(f"{label} selection must be an index or inclusive range")
    try:
        first, last = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"{label} selection must contain integer atom indices") from error
    if first < 1 or last < first or last > natoms:
        raise ValueError(f"{label} selection lies outside 1 through {natoms}")
    return tuple(range(first - 1, last))


def _ncia250_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer") from error
    return parsed


def _read_ncia250_complex(
    archive: ZipFile,
    member: str,
) -> tuple[Atoms, Atoms, Atoms, dict[str, Any]]:
    label = Path(member).name
    try:
        lines = archive.read(member).decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if len(lines) < 3:
        raise ValueError(f"{label} is missing XYZ content")
    natoms = _ncia250_int(lines[0].strip(), label=f"{label} atom count")
    if natoms <= 0 or len(lines) != natoms + 2:
        raise ValueError(f"{label} does not contain its declared atom count")
    metadata = _ncia250_metadata(lines[1], label=label)

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line_number, line in enumerate(lines[2:], start=3):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{label}:{line_number} must contain symbol, x, y, z")
        symbols.append(fields[0])
        try:
            positions.append([float(value) for value in fields[1:]])
        except ValueError as error:
            raise ValueError(f"{label}:{line_number} has a non-numeric coordinate") from error
    coordinates = np.asarray(positions, dtype=float)
    if coordinates.shape != (natoms, 3) or not np.isfinite(coordinates).all():
        raise ValueError(f"{label} contains invalid Cartesian coordinates")
    try:
        complex_atoms = Atoms(symbols=symbols, positions=coordinates, pbc=False)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{label} contains an unknown element") from error

    selection_a = _ncia250_selection(
        metadata["selection_a"], natoms, label=f"{label} fragment A"
    )
    selection_b = _ncia250_selection(
        metadata["selection_b"], natoms, label=f"{label} fragment B"
    )
    if sorted((*selection_a, *selection_b)) != list(range(natoms)):
        raise ValueError(f"{label} fragment selections must cover each atom once")

    charge = _ncia250_int(metadata["charge"], label=f"{label} charge")
    charge_a = _ncia250_int(metadata["charge_a"], label=f"{label} charge_a")
    charge_b = _ncia250_int(metadata["charge_b"], label=f"{label} charge_b")
    if charge != charge_a + charge_b:
        raise ValueError(f"{label} fragment charges do not sum to the complex charge")
    try:
        scale = float(metadata["scaling"])
        reference = float(metadata["benchmark_Eint"])
    except ValueError as error:
        raise ValueError(f"{label} contains non-numeric energy metadata") from error
    if not np.isfinite([scale, reference]).all() or scale <= 0.0:
        raise ValueError(f"{label} contains invalid energy metadata")
    if metadata["benchmark_unit"] != "kcal/mol":
        raise ValueError(f"{label} benchmark energy must use kcal/mol")

    name_parts = Path(member).stem.split("_")
    if len(name_parts) != 4 or name_parts[0] != "NCIA" or name_parts[-1] != "100":
        raise ValueError(f"{label} does not follow the NCIA250 file naming scheme")
    subset, system_id = name_parts[1:3]
    common_info = {
        "subset": subset,
        "system_id": system_id,
        "system_name": f"{subset} {system_id}",
        "interaction_class": metadata["group"],
        "scale": scale,
        "source_geometry_file": member,
    }
    fragment_a = complex_atoms[list(selection_a)]
    fragment_b = complex_atoms[list(selection_b)]
    for fragment, atoms, fragment_charge in (
        ("AB", complex_atoms, charge),
        ("A", fragment_a, charge_a),
        ("B", fragment_b, charge_b),
    ):
        atoms.info.update(common_info)
        atoms.info["fragment"] = fragment
        atoms.info["charge"] = fragment_charge

    atomic_numbers = tuple(int(value) for value in complex_atoms.numbers)
    record = {
        **common_info,
        "charge": charge,
        "charge_a": charge_a,
        "charge_b": charge_b,
        "natoms": natoms,
        "natoms_a": len(fragment_a),
        "natoms_b": len(fragment_b),
        "atomic_numbers": atomic_numbers,
        "elements": " ".join(
            chemical_symbols[value] for value in sorted(set(atomic_numbers))
        ),
        "ccsd_t_cbs_kcal_mol": reference,
        "contact": metadata.get("contact", ""),
    }
    return complex_atoms, fragment_a, fragment_b, record


def _ncia250_stats(
    inventory: pd.DataFrame,
    supported_atomic_numbers: tuple[int, ...],
) -> dict[str, Any]:
    compatible = inventory[inventory["model_compatible"]]
    all_sizes = inventory["natoms"].to_numpy(dtype=int)
    compatible_sizes = compatible["natoms"].to_numpy(dtype=int)
    element_counts: dict[str, int] = {}
    for values in inventory["atomic_numbers"]:
        for value in values:
            symbol = chemical_symbols[int(value)]
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
    unsupported_numbers = sorted(
        {
            int(value)
            for values in inventory.loc[
                ~inventory["model_compatible"], "unsupported_atomic_numbers"
            ]
            for value in values
        }
    )
    return {
        "total_complexes": len(inventory),
        "compatible_complexes": len(compatible),
        "excluded_complexes": int(len(inventory) - len(compatible)),
        "compatible_graphs": int(3 * len(compatible)),
        "total_atom_rows": int(2 * all_sizes.sum()),
        "compatible_atom_rows": int(2 * compatible_sizes.sum()),
        "dimer_atoms_min": int(all_sizes.min()),
        "dimer_atoms_median": float(np.median(all_sizes)),
        "dimer_atoms_max": int(all_sizes.max()),
        "compatible_dimer_atoms_min": int(compatible_sizes.min()),
        "compatible_dimer_atoms_median": float(np.median(compatible_sizes)),
        "compatible_dimer_atoms_max": int(compatible_sizes.max()),
        "source_counts": {
            str(key): int(value)
            for key, value in inventory["subset"].value_counts().sort_index().items()
        },
        "compatible_source_counts": {
            str(key): int(value)
            for key, value in compatible["subset"].value_counts().sort_index().items()
        },
        "element_counts": dict(sorted(element_counts.items())),
        "supported_elements": tuple(
            chemical_symbols[value] for value in supported_atomic_numbers
        ),
        "excluded_elements": tuple(
            chemical_symbols[value] for value in unsupported_numbers
        ),
    }


def load_ncia250(
    path: str | Path | None = None,
    *,
    supported_atomic_numbers: Sequence[int] | None = None,
    expected_sha256: str = NCIA250_SHA256,
) -> tuple[list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load NCIA250 and select the complexes supported by the AIMNet checkpoint.

    The returned ASE list contains adjacent ``AB``, ``A``, and ``B`` structures
    for each compatible complex. The inventory and statistics retain all 250
    source complexes, including entries outside the checkpoint's element set.
    """

    archive_path = _NCIA250_ARCHIVE if path is None else Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"NCIA250 archive is missing: {archive_path}")
    observed = _sha256_file(archive_path)
    if observed != expected_sha256:
        raise ValueError(
            f"NCIA250 archive SHA-256 mismatch: {observed}; expected {expected_sha256}"
        )
    if supported_atomic_numbers is None:
        supported_atomic_numbers = aimnet_checkpoint_species()
    supported = tuple(int(value) for value in supported_atomic_numbers)
    if (
        not supported
        or len(supported) != len(set(supported))
        or any(value <= 0 or value >= len(chemical_symbols) for value in supported)
    ):
        raise ValueError("supported_atomic_numbers must contain unique valid elements")
    supported_set = set(supported)

    compatible_atoms: list[Atoms] = []
    graph_records: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    try:
        with ZipFile(archive_path) as archive:
            members = sorted(
                member for member in archive.namelist() if member.endswith(".xyz")
            )
            if len(members) != 250 or len({Path(item).name for item in members}) != 250:
                raise ValueError("NCIA250 archive must contain 250 unique XYZ files")
            for member in members:
                complex_atoms, fragment_a, fragment_b, record = _read_ncia250_complex(
                    archive, member
                )
                unsupported = tuple(
                    sorted(set(record["atomic_numbers"]) - supported_set)
                )
                inventory_record = {
                    "complex_index": len(inventory_records),
                    **record,
                    "model_compatible": not unsupported,
                    "unsupported_atomic_numbers": unsupported,
                    "unsupported_elements": " ".join(
                        chemical_symbols[value] for value in unsupported
                    ),
                }
                inventory_records.append(inventory_record)
                if unsupported:
                    continue

                for fragment, atoms, fragment_charge in (
                    ("AB", complex_atoms, int(record["charge"])),
                    ("A", fragment_a, int(record["charge_a"])),
                    ("B", fragment_b, int(record["charge_b"])),
                ):
                    graph_index = len(compatible_atoms)
                    compatible_atoms.append(atoms)
                    graph_records.append(
                        {
                            "graph_index": graph_index,
                            **{key: record[key] for key in _NCI_CURVE_KEYS},
                            "fragment": fragment,
                            "charge": fragment_charge,
                            "natoms": len(atoms),
                            "source_geometry_file": record["source_geometry_file"],
                        }
                    )
                reference_records.append(
                    {
                        **{key: record[key] for key in _NCI_CURVE_KEYS},
                        "ccsd_t_cbs_kcal_mol": record["ccsd_t_cbs_kcal_mol"],
                    }
                )
    except BadZipFile as error:
        raise ValueError(f"NCIA250 archive is not a valid ZIP file: {archive_path}") from error

    inventory = pd.DataFrame(inventory_records)
    if inventory.duplicated(["subset", "system_id"]).any():
        raise ValueError("NCIA250 contains duplicate complex identifiers")
    expected_sources = {
        "D1200": 50,
        "HB300SPXx10": 50,
        "HB375x10": 50,
        "R739x5": 50,
        "SH250x10": 50,
    }
    source_counts = {
        str(key): int(value)
        for key, value in inventory["subset"].value_counts().sort_index().items()
    }
    if source_counts != expected_sources:
        raise ValueError(f"NCIA250 source counts differ from {expected_sources!r}")
    if set(inventory["charge"].astype(int)) != {0}:
        raise ValueError("NCIA250 should contain neutral complexes")

    graph_rows = pd.DataFrame(graph_records)
    references = pd.DataFrame(reference_records)
    expected_graphs = 3 * int(inventory["model_compatible"].sum())
    if len(compatible_atoms) != len(graph_rows) or len(graph_rows) != expected_graphs:
        raise ValueError("NCIA250 compatible graph records are incomplete")
    if len(references) != int(inventory["model_compatible"].sum()):
        raise ValueError("NCIA250 compatible reference records are incomplete")
    stats = _ncia250_stats(inventory, supported)
    return compatible_atoms, graph_rows, references, inventory, stats


def _ordered_nci_graph_rows(graph_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"graph_index", *_NCI_CURVE_KEYS, "fragment"}
    missing = required - set(graph_rows.columns)
    if missing:
        raise ValueError(f"graph rows are missing {sorted(missing)!r}")
    graph_indices = pd.to_numeric(graph_rows["graph_index"], errors="coerce").to_numpy(
        dtype=float
    )
    expected = np.arange(len(graph_rows), dtype=float)
    if (
        not np.isfinite(graph_indices).all()
        or not np.equal(graph_indices, np.rint(graph_indices)).all()
        or not np.array_equal(np.sort(graph_indices), expected)
    ):
        raise ValueError("graph_index must be a permutation of 0 through N-1")
    ordered = graph_rows.assign(graph_index=graph_indices.astype(np.int64)).sort_values(
        "graph_index"
    )
    grouped = ordered.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False)
    for key, group in grouped:
        if len(group) != 3 or set(group["fragment"]) != {"AB", "A", "B"}:
            raise ValueError(f"curve {key!r} must contain one AB, A, and B graph")
    return ordered.reset_index(drop=True)


def _graph_energy_array(values: Any, graph_count: int, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.shape == (graph_count, 1):
        array = array[:, 0]
    if array.shape != (graph_count,):
        raise ValueError(
            f"{name!r} must have shape ({graph_count},) or ({graph_count}, 1)"
        )
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name!r} contains a non-finite graph energy")
    return array


def interaction_components(
    graph_rows: pd.DataFrame,
    graph_energies_ev: Mapping[str, Any],
) -> pd.DataFrame:
    """Reduce named AB, A, and B graph energies to interaction curves."""

    if not graph_energies_ev:
        raise ValueError("at least one energy component is required")
    ordered = _ordered_nci_graph_rows(graph_rows)
    curve_order = ordered[list(_NCI_CURVE_KEYS)].drop_duplicates(ignore_index=True)
    curve_index = pd.MultiIndex.from_frame(curve_order)
    result = curve_order.copy()
    for name, values in graph_energies_ev.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component names must be non-empty strings")
        rows = ordered[list(_NCI_CURVE_KEYS) + ["fragment"]].copy()
        rows["energy"] = _graph_energy_array(values, len(ordered), name)
        wide = rows.pivot(
            index=list(_NCI_CURVE_KEYS),
            columns="fragment",
            values="energy",
        ).reindex(curve_index)
        result[name] = (
            (wide["AB"] - wide["A"] - wide["B"]) * _EV_TO_KCAL_MOL
        ).to_numpy(dtype=float)
    return result


def max_system_charge_error(batch: Batch, charges: torch.Tensor) -> float:
    """Return the largest per-system charge error in a batch."""

    flat_charges = charges.reshape(-1)
    if flat_charges.numel() != batch.num_nodes:
        raise ValueError("charges must contain one value per atom")
    expected = batch.charge.reshape(-1).to(
        device=flat_charges.device,
        dtype=flat_charges.dtype,
    )
    if expected.numel() != batch.num_graphs:
        raise ValueError("batch charge must contain one value per system")
    predicted = torch.zeros_like(expected).index_add(
        0,
        batch.batch_idx.to(device=flat_charges.device, dtype=torch.long),
        flat_charges,
    )
    return float((predicted - expected).abs().max().item())


def summarize_nci_interaction_curves(
    curves: pd.DataFrame,
    references: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Summarize sequential model additions on the three NCI curves."""

    components = {"aimnet_base", "base_d3", "complete"}
    missing = components - set(curves.columns)
    if missing:
        raise ValueError(f"curves are missing {sorted(missing)!r}")
    reference_columns = {"dft_d3_kcal_mol", "ccsd_t_cbs_kcal_mol"}
    missing = {*_NCI_CURVE_KEYS, *reference_columns} - set(references.columns)
    if missing:
        raise ValueError(f"references are missing {sorted(missing)!r}")

    merged = curves.merge(
        references,
        on=list(_NCI_CURVE_KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(curves) or len(merged) != len(references):
        raise ValueError("curves and references must describe the same scan points")

    summary: list[dict[str, str | float]] = []
    for system_name, group in merged.groupby("system_name", sort=False):
        equilibrium = group[np.isclose(group["scale"], 1.0)]
        if len(equilibrium) != 1:
            raise ValueError(f"{system_name!r} must have one equilibrium point")
        eq = equilibrium.iloc[0]
        summary.append(
            {
                "system": str(system_name),
                "d3_shift": float(eq["base_d3"] - eq["aimnet_base"]),
                "electrostatic_shift": float(eq["complete"] - eq["base_d3"]),
                "dft_mae": float(
                    (group["complete"] - group["dft_d3_kcal_mol"]).abs().mean()
                ),
                "cc_mae": float(
                    (group["complete"] - group["ccsd_t_cbs_kcal_mol"])
                    .abs()
                    .mean()
                ),
            }
        )
    return summary


def interaction_accuracy_summary(
    curves: pd.DataFrame,
    references: pd.DataFrame,
    *,
    components: Sequence[str] | None = None,
    reference_column: str = "ccsd_t_cbs_kcal_mol",
    tolerance_kcal_mol: float = 0.5,
) -> pd.DataFrame:
    """Summarize interaction-energy errors against one named reference."""

    if not np.isfinite(tolerance_kcal_mol) or tolerance_kcal_mol < 0.0:
        raise ValueError("tolerance_kcal_mol must be finite and non-negative")
    required_reference = {*_NCI_CURVE_KEYS, reference_column}
    missing_reference = required_reference - set(references.columns)
    if missing_reference:
        raise ValueError(f"references are missing {sorted(missing_reference)!r}")
    component_names = tuple(
        components
        if components is not None
        else [name for name in curves.columns if name not in _NCI_CURVE_KEYS]
    )
    if not component_names:
        raise ValueError("at least one component is required")
    missing_components = set(component_names) - set(curves.columns)
    if missing_components:
        raise ValueError(f"curves are missing {sorted(missing_components)!r}")
    merged = curves.merge(
        references[list(_NCI_CURVE_KEYS) + [reference_column]],
        on=list(_NCI_CURVE_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(merged["_merge"]) != {"both"}:
        raise ValueError("curves and references must describe the same complexes")
    numeric = merged[[*component_names, reference_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("curves or references contain a non-finite energy")
    errors = numeric[list(component_names)].sub(
        numeric[reference_column], axis="index"
    )
    absolute = errors.abs()
    winners = absolute.idxmin(axis="columns")
    rows = []
    for component in component_names:
        values = absolute[component].to_numpy(dtype=float)
        rows.append(
            {
                "component": component,
                "complexes": len(values),
                "mae_kcal_mol": float(values.mean()),
                "median_absolute_error_kcal_mol": float(np.median(values)),
                "p90_absolute_error_kcal_mol": float(np.quantile(values, 0.9)),
                "fraction_within_tolerance": float(
                    np.count_nonzero(values <= tolerance_kcal_mol) / len(values)
                ),
                "tolerance_kcal_mol": float(tolerance_kcal_mol),
                "best_for_complexes": int((winners == component).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_ncia250_survey(
    inventory: pd.DataFrame,
    accuracy_summary: pd.DataFrame | None = None,
) -> None:
    """Plot NCIA250 source coverage, dimer sizes, and optional model MAE."""

    required_inventory = {"subset", "natoms", "model_compatible"}
    missing_inventory = required_inventory - set(inventory.columns)
    if missing_inventory:
        raise ValueError(f"inventory is missing {sorted(missing_inventory)!r}")
    sizes = pd.to_numeric(inventory["natoms"], errors="coerce")
    if not np.isfinite(sizes.to_numpy(dtype=float)).all() or (sizes <= 0).any():
        raise ValueError("inventory contains an invalid dimer atom count")
    compatible = inventory["model_compatible"].astype(bool)
    subsets = sorted(str(value) for value in inventory["subset"].unique())
    supported_counts = np.asarray(
        [
            int(((inventory["subset"] == subset) & compatible).sum())
            for subset in subsets
        ],
        dtype=int,
    )
    excluded_counts = np.asarray(
        [
            int(((inventory["subset"] == subset) & ~compatible).sum())
            for subset in subsets
        ],
        dtype=int,
    )

    has_accuracy = accuracy_summary is not None
    figure, axes = plt.subplots(
        1,
        3 if has_accuracy else 2,
        figsize=(_TUTORIAL_FIGURE_WIDTH, 3.8),
        gridspec_kw={"width_ratios": (1.15, 1.0, 1.0)} if has_accuracy else {},
    )
    source_axis, size_axis = axes[:2]
    positions = np.arange(len(subsets))
    source_axis.bar(
        positions,
        supported_counts,
        color=_NVIDIA_GREEN,
        width=0.68,
        label="Evaluated",
    )
    source_axis.bar(
        positions,
        excluded_counts,
        bottom=supported_counts,
        color=_QUIET,
        width=0.68,
        label="Outside checkpoint elements",
    )
    source_axis.set(
        title="Complexes by source",
        ylabel="Complexes",
        xticks=positions,
        xticklabels=subsets,
    )
    source_axis.tick_params(axis="x", rotation=35)
    source_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    source_axis.legend(frameon=False)
    _polish_axis(source_axis)

    bins = np.arange(int(sizes.min()) - 0.5, int(sizes.max()) + 2.5, 2.0)
    size_axis.hist(
        sizes,
        bins=bins,
        color=_SURFACE_RAISED,
        edgecolor=_BORDER,
        linewidth=0.7,
        label=f"All {len(inventory)}",
    )
    size_axis.hist(
        sizes[compatible],
        bins=bins,
        color=_NVIDIA_GREEN,
        alpha=0.78,
        label=f"Evaluated {int(compatible.sum())}",
    )
    size_axis.set(
        title="Dimer size distribution",
        xlabel="Atoms in AB",
        ylabel="Complexes",
    )
    size_axis.ticklabel_format(axis="both", style="plain", useOffset=False)
    size_axis.legend(frameon=False)
    _polish_axis(size_axis)

    if has_accuracy:
        assert accuracy_summary is not None
        required_accuracy = {"component", "mae_kcal_mol"}
        missing_accuracy = required_accuracy - set(accuracy_summary.columns)
        if missing_accuracy:
            raise ValueError(
                f"accuracy summary is missing {sorted(missing_accuracy)!r}"
            )
        model_names = accuracy_summary["component"].astype(str).tolist()
        mae = pd.to_numeric(
            accuracy_summary["mae_kcal_mol"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(mae).all() or (mae < 0.0).any():
            raise ValueError("accuracy summary contains an invalid MAE")
        colors = {
            "aimnet_base": _NVIDIA_ORANGE,
            "base_d3": _NVIDIA_BLUE,
            "complete": _NVIDIA_GREEN,
        }
        labels = {
            "aimnet_base": "AIMNet\nbase",
            "base_d3": "base +\nD3",
            "complete": "complete\nmodel",
        }
        accuracy_axis = axes[2]
        bars = accuracy_axis.bar(
            np.arange(len(model_names)),
            mae,
            color=[colors.get(name, _MUTED) for name in model_names],
            width=0.68,
        )
        accuracy_axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in mae],
            padding=3,
            fontsize=_TUTORIAL_BODY_FONT_SIZE,
        )
        accuracy_axis.set(
            title="MAE vs CCSD(T)/CBS",
            ylabel="MAE [kcal/mol]",
            xticks=np.arange(len(model_names)),
            xticklabels=[labels.get(name, name) for name in model_names],
        )
        accuracy_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        accuracy_axis.margins(y=0.18)
        _polish_axis(accuracy_axis)

    figure.subplots_adjust(wspace=0.34, bottom=0.24)
    _display_figure(
        figure,
        "NCIA250 source counts and dimer atom-count distribution, with the compatible AIMNet2 subset highlighted"
        + (" and three model-composition MAEs compared" if has_accuracy else "")
        + ".",
    )


def plot_nci_interaction_curves(
    curves: pd.DataFrame,
    references: pd.DataFrame,
    *,
    components: Sequence[str] | None = None,
) -> None:
    """Plot three model constructions with DFT-D3 and CCSD(T)/CBS references."""

    reference_columns = {"dft_d3_kcal_mol", "ccsd_t_cbs_kcal_mol"}
    required_references = {*_NCI_CURVE_KEYS, *reference_columns}
    missing_reference = required_references - set(references.columns)
    if missing_reference:
        raise ValueError(f"references are missing {sorted(missing_reference)!r}")
    component_names = tuple(
        components
        if components is not None
        else [name for name in curves.columns if name not in _NCI_CURVE_KEYS]
    )
    if not component_names:
        raise ValueError("at least one plotted component is required")
    missing_components = set(component_names) - set(curves.columns)
    if missing_components:
        raise ValueError(f"curves are missing {sorted(missing_components)!r}")
    merged = curves.merge(
        references,
        on=list(_NCI_CURVE_KEYS),
        how="inner",
        validate="one_to_one",
    )
    plotted_columns = [*component_names, *reference_columns]
    plotted = merged[plotted_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(plotted.to_numpy(dtype=float)).all():
        raise ValueError("interaction curves contain a non-finite value")

    labels = {
        "aimnet_base": "AIMNet base",
        "base_d3": "base + D3",
        "complete": "complete model",
    }
    styles = {
        "aimnet_base": (_NVIDIA_ORANGE, ":", 1.8),
        "base_d3": (_NVIDIA_BLUE, "--", 2.0),
        "complete": (_NVIDIA_GREEN, "-", 2.8),
    }
    system_names = tuple(dict.fromkeys(merged["system_name"].astype(str)))
    figure, axes = plt.subplots(
        1,
        len(system_names),
        figsize=(_TUTORIAL_FIGURE_WIDTH, 3.9),
        sharex=True,
        squeeze=False,
    )
    for axis, system_name in zip(axes.reshape(-1), system_names, strict=True):
        group = merged[merged["system_name"].astype(str) == system_name].sort_values(
            "scale"
        )
        for component in component_names:
            color, linestyle, width = styles.get(
                component,
                (_MUTED, "-", 1.8),
            )
            axis.plot(
                group["scale"],
                group[component],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                marker="o" if component == "complete" else None,
                markersize=3.8,
                label=labels.get(component, component.replace("_", " ")),
                zorder=4 if component == "complete" else 2,
            )
        axis.plot(
            group["scale"],
            group["dft_d3_kcal_mol"],
            color=_TEXT,
            linestyle=(0, (3, 2)),
            linewidth=1.4,
            label="DFT-D3",
            zorder=3,
        )
        axis.scatter(
            group["scale"],
            group["ccsd_t_cbs_kcal_mol"],
            s=24,
            facecolor=_SURFACE,
            edgecolor=_TEXT,
            linewidth=1.0,
            label="CCSD(T)/CBS",
            zorder=5,
        )
        axis.axhline(0.0, color=_QUIET, linewidth=0.8, alpha=0.65)
        axis.set(
            title=system_name,
            xlabel=r"Relative separation $R/R_e$",
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        _polish_axis(axis)
    axes[0, 0].set_ylabel("Interaction energy [kcal/mol]")
    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(6, len(handles)),
        frameon=False,
    )
    figure.subplots_adjust(top=0.76, wspace=0.24)
    _display_figure(
        figure,
        "AIMNet base, base plus D3, and complete interaction curves for three NCI Atlas dissociation scans, compared with DFT-D3 and CCSD(T)/CBS references.",
    )


def _benchmark_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    name = platform.processor().strip()
    return name or "CPU"


def _sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nci_triplet_indices(graph_rows: pd.DataFrame) -> list[tuple[int, int, int]]:
    ordered = _ordered_nci_graph_rows(graph_rows)
    triplets = []
    for _, group in ordered.groupby(list(_NCI_CURVE_KEYS), sort=False, dropna=False):
        by_fragment = {
            str(row.fragment): int(row.graph_index) for row in group.itertuples()
        }
        triplets.append(tuple(by_fragment[name] for name in ("AB", "A", "B")))
    flattened = [index for triplet in triplets for index in triplet]
    if flattened != list(range(len(ordered))):
        raise ValueError("graph rows must keep each AB, A, and B triplet adjacent")
    return triplets


def benchmark_interaction_batching(
    graphs: Sequence[AtomicData],
    graph_rows: pd.DataFrame,
    combined_batch: Batch,
    model: BaseModelMixin,
    *,
    warmups: int = 1,
    repeats: int = 3,
    cpu_complexes: int = 24,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare serial complex triplets with one combined ``Batch`` model call.

    Batch and neighbor construction happen before timing. CUDA runs use every
    supplied complex. CPU runs use the first ``cpu_complexes`` and report that
    smaller workload explicitly.
    """

    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    if cpu_complexes < 1:
        raise ValueError("cpu_complexes must be positive")
    if not np.isfinite([rtol, atol]).all() or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    if len(graphs) != len(graph_rows):
        raise ValueError("graphs and graph_rows must have the same length")
    if int(combined_batch.num_graphs) != len(graphs):
        raise ValueError("combined_batch must contain every supplied graph once")
    triplets = _nci_triplet_indices(graph_rows)
    if not triplets:
        raise ValueError("at least one AB, A, and B triplet is required")

    run_device = torch.device(combined_batch.device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested, but CUDA is unavailable")
    if run_device.type == "cuda" and run_device.index is None:
        run_device = torch.device("cuda", torch.cuda.current_device())
    graph_devices = {torch.device(graph.positions.device) for graph in graphs}
    if graph_devices != {run_device}:
        raise ValueError("graphs and combined_batch must use the same device")

    dataset_complexes = len(triplets)
    if run_device.type == "cuda":
        selected_triplets = triplets
    else:
        selected_triplets = triplets[: min(cpu_complexes, dataset_complexes)]
    selected_indices = [index for triplet in selected_triplets for index in triplet]
    selected_graphs = [graphs[index] for index in selected_indices]
    full_dataset = len(selected_triplets) == dataset_complexes
    if full_dataset:
        benchmark_batch = combined_batch
    else:
        benchmark_batch = Batch.from_data_list(selected_graphs, device=run_device)

    expected_counts = [len(graph.atomic_numbers) for graph in selected_graphs]
    observed_counts = [
        int(value) for value in benchmark_batch.num_nodes_per_graph.tolist()
    ]
    if observed_counts != expected_counts:
        raise ValueError("combined Batch atom boundaries do not match the source graphs")

    serial_batches = [
        Batch.from_data_list(
            [graphs[index] for index in triplet],
            device=run_device,
        )
        for triplet in selected_triplets
    ]
    from nvalchemi.neighbors import compute_neighbors

    neighbor_config = model.model_config.neighbor_config
    if neighbor_config is not None:
        compute_neighbors(benchmark_batch, config=neighbor_config)
        for batch in serial_batches:
            compute_neighbors(batch, config=neighbor_config)

    saved_outputs = set(model.model_config.active_outputs)
    model.model_config.active_outputs = {"energy", "forces"}

    def serial_output() -> tuple[torch.Tensor, torch.Tensor]:
        energies = []
        forces = []
        for batch in serial_batches:
            output = model(batch)
            energies.append(output["energy"].reshape(-1))
            forces.append(output["forces"])
        return torch.cat(energies), torch.cat(forces)

    def call_serial() -> None:
        for batch in serial_batches:
            _ = model(batch)

    def call_batch() -> None:
        _ = model(benchmark_batch)

    hardware = _benchmark_device_name(run_device)
    device_label = "CUDA GPU" if run_device.type == "cuda" else "CPU"
    try:
        serial_energy, serial_forces = serial_output()
        batch_output = model(benchmark_batch)
        batch_energy = batch_output["energy"].reshape(-1)
        batch_forces = batch_output["forces"]
        _sync_torch_device(run_device)
        torch.testing.assert_close(
            serial_energy,
            batch_energy,
            rtol=rtol,
            atol=atol,
            msg="serial and combined Batch energies differ",
        )
        torch.testing.assert_close(
            serial_forces,
            batch_forces,
            rtol=rtol,
            atol=atol,
            msg="serial and combined Batch forces differ",
        )
        max_energy_delta = float(
            (serial_energy - batch_energy).detach().abs().max().cpu()
        )
        max_force_delta = float(
            (serial_forces - batch_forces).detach().abs().max().cpu()
        )

        structures = len(selected_graphs)
        atoms = int(benchmark_batch.num_nodes)
        routes = (
            {
                "route": "serial_triplets",
                "label": f"{len(selected_triplets):,} × Batch[3]",
                "model_calls": len(selected_triplets),
                "call": call_serial,
            },
            {
                "route": "combined_batch",
                "label": f"1 × Batch[{structures:,}]",
                "model_calls": 1,
                "call": call_batch,
            },
        )
        progress = Progress(
            _ActiveSpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(
                bar_width=22,
                style=_SURFACE_RAISED,
                complete_style=_NVIDIA_GREEN,
                finished_style=_NVIDIA_GREEN,
            ),
            _AlignedCountColumn(),
            TimeElapsedColumn(),
            refresh_per_second=10,
        )
        task_ids = {
            route["route"]: progress.add_task(
                f"Energy + forces · {route['label']} · {device_label}",
                total=(warmups + repeats) * int(route["model_calls"]),
                active=False,
                start=False,
            )
            for route in routes
        }
        timing_rows = []
        with Live(
            progress, console=progress.console, refresh_per_second=10, transient=True
        ):
            for route in routes:
                task_id = task_ids[str(route["route"])]
                progress.start_task(task_id)
                progress.update(task_id, active=True, refresh=True)
                call = route["call"]
                advance = int(route["model_calls"])
                for _ in range(warmups):
                    if route["route"] == "serial_triplets":
                        reported = 0
                        for completed, batch in enumerate(serial_batches, start=1):
                            _ = model(batch)
                            if completed % 25 == 0 or completed == len(serial_batches):
                                progress.update(
                                    task_id,
                                    advance=completed - reported,
                                    refresh=True,
                                )
                                reported = completed
                    else:
                        call()
                        progress.update(task_id, advance=advance, refresh=True)
                    _sync_torch_device(run_device)
                if run_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(run_device)
                    allocated_before = torch.cuda.memory_allocated(run_device)
                else:
                    allocated_before = 0
                repeat_seconds = []
                for _ in range(repeats):
                    _sync_torch_device(run_device)
                    start = perf_counter()
                    call()
                    _sync_torch_device(run_device)
                    repeat_seconds.append(perf_counter() - start)
                    progress.update(task_id, advance=advance, refresh=True)
                progress.update(task_id, active=False, refresh=True)
                values = np.asarray(repeat_seconds, dtype=float)
                if run_device.type == "cuda":
                    peak_bytes = torch.cuda.max_memory_allocated(run_device)
                    peak_memory = peak_bytes / 1024**2
                    incremental_peak = (peak_bytes - allocated_before) / 1024**2
                else:
                    peak_memory = None
                    incremental_peak = None
                timing_rows.append(
                    {
                        "route": route["route"],
                        "label": route["label"],
                        "model_calls": route["model_calls"],
                        "complexes": len(selected_triplets),
                        "structures": structures,
                        "atoms": atoms,
                        "repeat_seconds": repeat_seconds,
                        "median_seconds": float(np.median(values)),
                        "spread_seconds": float(values.max() - values.min()),
                        "min_seconds": float(values.min()),
                        "max_seconds": float(values.max()),
                        "structures_per_second": float(
                            structures / np.median(values)
                        ),
                        "peak_memory_mib": peak_memory,
                        "incremental_peak_memory_mib": incremental_peak,
                    }
                )
    finally:
        model.model_config.active_outputs = saved_outputs

    route_by_name = {row["route"]: row for row in timing_rows}
    speedup = (
        route_by_name["serial_triplets"]["median_seconds"]
        / route_by_name["combined_batch"]["median_seconds"]
    )
    return {
        "routes": timing_rows,
        "speedup": float(speedup),
        "dataset_complexes": dataset_complexes,
        "benchmarked_complexes": len(selected_triplets),
        "full_dataset": full_dataset,
        "structures": structures,
        "atoms": atoms,
        "device": str(run_device),
        "device_label": device_label,
        "hardware": hardware,
        "warmups": warmups,
        "repeats": repeats,
        "timing_scope": (
            "energy and force model calls with Batch and neighbor construction excluded"
        ),
        "correctness": {
            "rtol": float(rtol),
            "atol": float(atol),
            "max_energy_delta_ev": max_energy_delta,
            "max_force_delta_ev_per_angstrom": max_force_delta,
        },
    }


def plot_interaction_batching(results: Mapping[str, Any]) -> None:
    """Plot serial-triplet and combined-Batch times on a linear seconds axis."""

    routes = pd.DataFrame(results.get("routes", []))
    required = {"route", "label", "median_seconds", "structures", "atoms"}
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(f"batching results are missing {sorted(missing)!r}")
    expected_routes = ("serial_triplets", "combined_batch")
    if set(routes["route"]) != set(expected_routes):
        raise ValueError("batching results must contain serial and combined routes")
    routes = routes.set_index("route").loc[list(expected_routes)].reset_index()
    seconds = pd.to_numeric(routes["median_seconds"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(seconds).all() or (seconds <= 0.0).any():
        raise ValueError("batching results contain an invalid time")
    speedup = float(results.get("speedup", np.nan))
    if not np.isfinite(speedup) or speedup <= 0.0:
        raise ValueError("batching results contain an invalid speedup")

    if np.isclose(speedup, 1.0, rtol=0.02, atol=0.0):
        comparison = "same median time"
    elif speedup > 1.0:
        comparison = f"{speedup:.1f}× faster as one Batch"
    else:
        comparison = f"{1.0 / speedup:.1f}× faster as triplet calls"

    figure, axis = plt.subplots(
        figsize=(_TUTORIAL_SMALL_FIGURE_WIDTH, 3.2)
    )
    bars = axis.barh(
        routes["label"].astype(str),
        seconds,
        color=[_MUTED, _NVIDIA_GREEN],
        height=0.56,
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.3f} s" for value in seconds],
        padding=5,
        fontsize=_TUTORIAL_BODY_FONT_SIZE,
    )
    axis.invert_yaxis()
    axis.set(
        title=f"{int(routes['structures'].iloc[0]):,} structures · {comparison}",
        xlabel="Median energy + force time [s]",
    )
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.margins(x=0.2)
    _polish_axis(axis, grid="x")
    hardware = str(results.get("hardware", "measured device"))
    figure.subplots_adjust(left=0.28, right=0.94, top=0.78, bottom=0.22)
    _display_figure(
        figure,
        f"Median energy and force time on {hardware}: repeated three-structure calls compared with one combined Toolkit Batch call.",
        max_width=_TUTORIAL_SMALL_VISUAL_MAX_WIDTH,
    )


def _prepare_component_batch(
    graphs: Sequence[AtomicData],
    count: int,
    model: BaseModelMixin,
    device: torch.device,
) -> Batch:
    from nvalchemi.neighbors import compute_neighbors

    selected = [graphs[index % len(graphs)] for index in range(count)]
    batch = Batch.from_data_list(selected, device=device)
    config = model.model_config.neighbor_config
    if config is not None:
        compute_neighbors(batch, config=config)
    return batch


def benchmark_nci_components(
    atoms: Sequence[Atoms],
    aimnet: BaseModelMixin,
    d3: BaseModelMixin,
    coulomb: BaseModelMixin,
    *,
    full_model: BaseModelMixin | None = None,
    device: torch.device | str,
    graph_counts: Sequence[int] | None = None,
    warmups: int = 2,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    """Time warm component and complete-pipeline forward calls.

    Batch construction, neighbor construction, and isolated charge preparation
    happen before timing. Isolated Coulomb treats the stored charges as inputs.
    The complete-pipeline row measures one public pipeline call on a batch whose
    source neighbor list is already present, including charge-response autograd.
    """

    if not atoms:
        raise ValueError("atoms must contain at least one structure")
    counts = tuple(
        int(value)
        for value in (
            graph_counts
            if graph_counts is not None
            else (len(atoms), 2 * len(atoms), 3 * len(atoms))
        )
    )
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("graph_counts must contain positive integers")
    if any(value % len(atoms) for value in counts):
        raise ValueError("graph_counts must contain whole repeats of the input set")
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested, but CUDA is unavailable")

    model_outputs: list[tuple[BaseModelMixin, set[str]]] = [
        (aimnet, {"energy", "forces", "charges"}),
        (d3, {"energy", "forces"}),
        (coulomb, {"energy", "forces"}),
    ]
    if full_model is not None:
        model_outputs.append((full_model, {"energy", "forces"}))
    saved_outputs = [
        set(model.model_config.active_outputs) for model, _ in model_outputs
    ]
    for model, outputs in model_outputs:
        model.model_config.active_outputs = outputs

    hardware = _benchmark_device_name(run_device)
    device_label = "CUDA GPU" if run_device.type == "cuda" else "CPU"
    source_graphs = [AtomicData.from_atoms(item, device=run_device) for item in atoms]
    rows: list[dict[str, Any]] = []
    try:
        prepared: dict[int, tuple[int, dict[str, Any]]] = {}
        for graph_count in counts:
            aimnet_batch = _prepare_component_batch(
                source_graphs, graph_count, aimnet, run_device
            )
            d3_batch = _prepare_component_batch(
                source_graphs, graph_count, d3, run_device
            )
            coulomb_batch = _prepare_component_batch(
                source_graphs, graph_count, coulomb, run_device
            )
            aimnet.model_config.active_outputs = {"charges"}
            charges = aimnet(aimnet_batch)["charges"].detach()
            aimnet.model_config.active_outputs = {"energy", "forces", "charges"}
            charge_chunks = list(
                charges.split(aimnet_batch.num_nodes_per_graph.tolist(), dim=0)
            )
            coulomb_batch.add_key(
                "partial_charges",
                charge_chunks,
                level="node",
            )
            calls: dict[str, Any] = {
                "AIMNet base": lambda batch=aimnet_batch: aimnet(batch),
                "D3": lambda batch=d3_batch: d3(batch),
                "Electrostatics": lambda batch=coulomb_batch: coulomb(batch),
            }
            if full_model is not None:
                full_batch = _prepare_component_batch(
                    source_graphs,
                    graph_count,
                    full_model,
                    run_device,
                )
                calls["Complete pipeline"] = lambda batch=full_batch: full_model(batch)
            prepared[graph_count] = (int(aimnet_batch.num_nodes), calls)

        progress = Progress(
            _ActiveSpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(
                bar_width=22,
                style=_SURFACE_RAISED,
                complete_style=_NVIDIA_GREEN,
                finished_style=_NVIDIA_GREEN,
            ),
            _AlignedCountColumn(),
            TimeElapsedColumn(),
            refresh_per_second=10,
        )
        task_ids: dict[tuple[int, str], Any] = {}
        for graph_count, (_, calls) in prepared.items():
            for component in calls:
                task_ids[(graph_count, component)] = progress.add_task(
                    f"{component} · {graph_count:,} structures · {device_label}",
                    total=warmups + repeats,
                    active=False,
                    start=False,
                )

        with Live(
            progress, console=progress.console, refresh_per_second=10, transient=True
        ):
            for graph_count, (atom_count, calls) in prepared.items():
                for component, call in calls.items():
                    task_id = task_ids[(graph_count, component)]
                    progress.start_task(task_id)
                    progress.update(task_id, active=True, refresh=True)
                    for _ in range(warmups):
                        _ = call()
                        if run_device.type == "cuda":
                            torch.cuda.synchronize(run_device)
                        progress.update(task_id, advance=1, refresh=True)
                    if run_device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(run_device)
                        allocated_before = torch.cuda.memory_allocated(run_device)
                    else:
                        allocated_before = 0
                    repeat_seconds = []
                    for _ in range(repeats):
                        if run_device.type == "cuda":
                            torch.cuda.synchronize(run_device)
                        start = perf_counter()
                        _ = call()
                        if run_device.type == "cuda":
                            torch.cuda.synchronize(run_device)
                        repeat_seconds.append(perf_counter() - start)
                        progress.update(task_id, advance=1, refresh=True)
                    progress.update(task_id, active=False, refresh=True)
                    timings = np.asarray(repeat_seconds, dtype=float)
                    if run_device.type == "cuda":
                        peak_bytes = torch.cuda.max_memory_allocated(run_device)
                        peak_memory = peak_bytes / 1024**2
                        incremental_peak = (peak_bytes - allocated_before) / 1024**2
                    else:
                        peak_memory = None
                        incremental_peak = None
                    rows.append(
                        {
                            "structures": graph_count,
                            "dataset_repeats": graph_count // len(atoms),
                            "atoms": atom_count,
                            "component": component,
                            "repeat_seconds": repeat_seconds,
                            "median_seconds": float(np.median(timings)),
                            "spread_seconds": float(timings.max() - timings.min()),
                            "min_seconds": float(timings.min()),
                            "max_seconds": float(timings.max()),
                            "peak_memory_mib": peak_memory,
                            "incremental_peak_memory_mib": incremental_peak,
                            "device": device_label,
                            "hardware": hardware,
                            "warmups": warmups,
                            "repeats": repeats,
                            "timing_scope": (
                                "warm energy and force call with prepared neighbors and stored charges"
                                if component == "Electrostatics"
                                else (
                                    "warm energy and force pipeline call with prepared source neighbors and charge-response autograd"
                                    if component == "Complete pipeline"
                                    else "warm energy and force call with prepared neighbors"
                                )
                            ),
                        }
                    )
    finally:
        for (model, _), outputs in zip(model_outputs, saved_outputs, strict=True):
            model.model_config.active_outputs = outputs
    return rows


def plot_nci_component_timings(rows: Sequence[Mapping[str, Any]]) -> None:
    """Plot scaling and the four measured calls at the largest workload."""

    if not rows:
        raise ValueError("timing rows must not be empty")
    table = pd.DataFrame(rows)
    required = {
        "structures",
        "atoms",
        "component",
        "median_seconds",
        "hardware",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"timing rows are missing {sorted(missing)!r}")
    numeric = table[["structures", "atoms", "median_seconds"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("timing rows contain a non-finite value")
    if table.duplicated(["structures", "component"]).any():
        raise ValueError("timing rows contain duplicate structure/component entries")

    colors = {
        "AIMNet base": _NVIDIA_ORANGE,
        "D3": _NVIDIA_BLUE,
        "Electrostatics": _NVIDIA_TEAL,
        "Complete pipeline": _NVIDIA_GREEN,
    }
    component_order = tuple(
        component
        for component in (
            "AIMNet base",
            "D3",
            "Electrostatics",
            "Complete pipeline",
        )
        if component in set(table["component"])
    )
    if len(component_order) != 4:
        missing_components = set(colors) - set(component_order)
        raise ValueError(f"timing rows are missing {sorted(missing_components)!r}")
    largest = int(table["structures"].max())
    largest_rows = table[table["structures"] == largest].set_index("component")
    if largest_rows["atoms"].nunique() != 1:
        raise ValueError("largest-workload rows must have one shared atom count")

    figure, (scaling_axis, comparison_axis) = plt.subplots(
        1,
        2,
        figsize=(_TUTORIAL_FIGURE_WIDTH, 4.2),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )

    for component in component_order:
        selected = table[table["component"] == component].sort_values("atoms")
        scaling_axis.plot(
            selected["atoms"],
            selected["median_seconds"],
            color=colors[component],
            linewidth=2.6 if component == "Complete pipeline" else 2.0,
            marker="o",
            markersize=5.0,
            label=component,
        )
    atom_ticks = sorted(int(value) for value in table["atoms"].unique())
    scaling_axis.set(
        title="Warm energy and force time",
        xlabel="Total atoms in Batch",
        ylabel="Median time [s]",
        xticks=atom_ticks,
    )
    scaling_axis.ticklabel_format(axis="both", style="plain", useOffset=False)
    scaling_axis.legend(frameon=False)
    _polish_axis(scaling_axis)

    values = largest_rows.loc[list(component_order), "median_seconds"].to_numpy(
        dtype=float
    )
    largest_atoms = int(largest_rows["atoms"].iloc[0])
    short_labels = ("AIMNet\nbase", "D3", "Electrostatics", "Complete\npipeline")
    bars = comparison_axis.bar(
        short_labels,
        values,
        color=[colors[component] for component in component_order],
        width=0.68,
    )
    comparison_axis.bar_label(
        bars,
        labels=[f"{value:.3f} s" for value in values],
        padding=3,
        fontsize=_TUTORIAL_BODY_FONT_SIZE,
    )
    comparison_axis.set(
        title=f"Largest workload · {largest:,} structures · {largest_atoms:,} atoms",
        ylabel="Median time [s]",
    )
    comparison_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    comparison_axis.margins(y=0.2)
    _polish_axis(comparison_axis)

    hardware = str(table["hardware"].iloc[0])
    figure.suptitle(f"Interaction-model timing on {hardware}", y=1.01)
    _display_figure(
        figure,
        "Median warm energy and force time versus total atoms, followed by separate AIMNet base, D3, electrostatic, and complete-pipeline measurements at the largest workload.",
    )
