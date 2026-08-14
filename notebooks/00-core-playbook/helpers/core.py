"""Checked data, tiny API sandboxes, and bounded visuals for the Core playbook."""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Sequence
from hashlib import sha256
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.training import TrainingStage
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
from torch.utils.data import DataLoader as TorchDataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ALIAS = "aimnet2-wb97m-d3_0"
_MODEL_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
_ELEMENT_COLORS = {
    1: "#E8ECEF",
    6: "#5C6770",
    7: "#00A3E0",
    8: "#E05252",
    18: "#80D1E3",
}
_ELEMENT_FALLBACK_COLOR = "#76B900"
_VIEWER_BUNDLE = _PROJECT_ROOT / "shared" / "3dmol-2.5.5" / "3Dmol-min.js"
_VIEWER_BACKGROUND = "#111619"
_VIEWER_CELL_COLOR = "#59656E"
_PROGRESS_UPDATE_STRUCTURES = 100


class _ActiveSpinnerColumn(SpinnerColumn):
    """Spin for the current route and keep the other routes quiet."""

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


def freeze_model[T: torch.nn.Module](model: T) -> T:
    """Freeze parameters while preserving gradients with respect to positions."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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


def probe_nan_detector(model: torch.nn.Module, batch: Batch, detector: Any) -> str:
    """Exercise a NaN detector through one real dynamics step on a clone."""

    from nvalchemi.dynamics import BaseDynamics, DynamicsStage

    class InjectNonFiniteForce:
        stage = DynamicsStage.AFTER_COMPUTE
        frequency = 1

        def __call__(self, ctx: Any, stage: DynamicsStage) -> None:
            ctx.batch.forces[0, 0] = torch.nan

    probe = prepare_dynamics_batch(batch)
    hooks = [*model.make_neighbor_hooks(), InjectNonFiniteForce(), detector]
    workflow = BaseDynamics(model=model, n_steps=1, hooks=hooks)
    try:
        workflow.run(probe)
    except RuntimeError as error:
        if "Non-finite" not in str(error):
            raise
        return str(error)
    raise RuntimeError("NaNDetectorHook did not report the injected non-finite force")


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


def benchmark_repeated_molecule(
    atoms: Atoms,
    *,
    batch_size: int = 2048,
    device: torch.device | str = "cuda",
) -> list[dict[str, Any]]:
    """Compare individual and batched model calls on CPU and CUDA."""

    import warnings

    from nvalchemi.models import AIMNet2Wrapper
    from nvalchemi.neighbors import compute_neighbors

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")

    def load_model(run_device: torch.device) -> AIMNet2Wrapper:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting a tensor with requires_grad=True to a scalar.*",
                category=UserWarning,
            )
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
            style="#25303A",
            complete_style="#76B900",
            finished_style="#76B900",
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
                    "seconds": seconds,
                    "molecules/s": batch_size / seconds,
                    "peak memory [MiB]": peak_memory,
                    "energy shape": tuple(output["energy"].shape),
                    "forces shape": tuple(output["forces"].shape),
                }
            )
        overall.stop_task(overall_task)
    return rows


def plot_batching_benchmark(rows: Sequence[dict[str, Any]]) -> Any:
    """Plot elapsed time and throughput for the batching comparison."""

    batch_size = int(rows[0]["molecules"])
    positions = np.arange(2)
    width = 0.34
    route_labels = [f"{batch_size:,} individual calls", "1 Batch call"]
    devices = [
        name for name in ("CPU", "GPU") if any(r["device"] == name for r in rows)
    ]
    colors = {"CPU": "#00A3E0", "GPU": "#76B900"}
    offsets = (
        {"CPU": -width / 2, "GPU": width / 2}
        if len(devices) == 2
        else {devices[0]: 0.0}
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
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
    figure.suptitle(
        f"Same {batch_size:,} ethyne molecules · {int(rows[0]['atoms']):,} atoms{memory_text}"
    )
    return _display_figure(
        figure,
        "Linear-scale evaluation time and throughput for individual and batched CPU and GPU model calls over the same repeated-molecule workload.",
    )


def configure_presentation() -> None:
    """Apply the shared plotting style."""

    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-dark.mplstyle")


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
    formula_text = escape(formula)
    return f"""
<div class="alchemi-structure-widget"
     aria-label="Interactive structure viewer for {formula_text}"
     style="width:100%; overflow:hidden; border-radius:10px;
            background:#111619; color:#F3F4F6;
            font-family:'NVIDIA Sans',Arial,sans-serif;">
  <div style="display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between;
              gap:16px; padding:12px 16px 10px; background:#171D20;">
    <div style="display:flex; align-items:baseline; gap:10px; min-width:0;">
      <strong style="font-size:1.05rem; font-weight:650;">{formula_text}</strong>
      <span style="color:#B8C0C6; font-size:0.84rem;">{atom_count} atoms</span>
    </div>
    <span style="color:#A9B3B9; font-size:0.78rem; text-align:right;">
      Drag to rotate · scroll to zoom · click an atom
    </span>
  </div>
  <div class="alchemi-structure-view"
       style="width:100%; height:{height}px; overflow:hidden;
              background:{_VIEWER_BACKGROUND};">
    {preamble}{viewer_markup}
  </div>
  <div data-atom-detail role="status" aria-live="polite"
       style="min-height:22px; padding:10px 16px 12px; background:#171D20;
              color:#DCE2E5; font-size:0.84rem; font-variant-numeric:tabular-nums;">
    Click an atom to inspect it.
  </div>
</div>
"""


def _atom_click_callback(atoms: Atoms) -> str:
    """Return the 3Dmol.js callback used by the structure widget."""

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
      if (detail) {{
        detail.textContent = 'Atom row ' + atomIndex + ' · ' + atom.elem
          + ' · atomic number ' + atomicNumber;
      }}
      if (viewer.__alchemiAtomLabel) {{
        viewer.removeLabel(viewer.__alchemiAtomLabel);
      }}
      viewer.__alchemiAtomLabel = viewer.addLabel(
        atom.elem + ' · row ' + atomIndex + ' · Z=' + atomicNumber,
        {{
          position: atom,
          backgroundColor: '#171D20',
          backgroundOpacity: 0.92,
          fontColor: '#F3F4F6',
          fontSize: 12,
          borderColor: '#76B900',
          borderThickness: 1,
          inFront: true
        }}
      );
      viewer.render();
    }}"""


def show_molecule(
    atoms: Atoms,
    *,
    height: int = 360,
    sphere_scale: float = 0.23,
) -> Any:
    """Return a compact 3Dmol.js ball-and-stick view of one ASE structure.

    3Dmol.js perceives connectivity from the coordinates alone, so no explicit
    bond list is passed in. A periodic structure also gets its lattice drawn as
    twelve edges: 3Dmol's `addUnitCell` reads crystal metadata that XYZ input
    cannot carry, so it draws nothing here.

    The structure is turned to face the camera before it is handed over. That is
    a rigid rotation, so every bond length and angle a learner reads off the
    printed `Atoms` object is unchanged.
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
            formula=atoms.get_chemical_formula(mode="hill"),
            atom_count=len(atoms),
        )
    )


def show_argon_batch(batch: Batch, *, height: int = 280) -> Any:
    """Show the first periodic argon system in a compact interactive viewer."""

    data = batch.get_data(0)
    atoms = Atoms(
        numbers=data.atomic_numbers.detach().cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
        cell=data.cell.detach().cpu().numpy().reshape(3, 3),
        pbc=data.pbc.detach().cpu().numpy().reshape(3),
    )
    return show_molecule(atoms, height=height, sphere_scale=0.42)


def show_capability_map() -> Any:
    """Display the Toolkit map with native HTML hover details and links."""

    from IPython.display import HTML

    asset = _PROJECT_ROOT / "notebooks" / "00-core-playbook" / "assets"
    svg = (asset / "toolkit-capability-map.svg").read_text(encoding="utf-8")
    regions = (
        (
            1.702,
            "Data and state",
            ("Keep atomistic data as tensors", "on the model device."),
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
            ("Train and fine-tune models.", "Split one large system across GPUs."),
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
    """Apply the restrained Core playbook scientific-figure treatment."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is None:
        axis.grid(False)
    else:
        axis.grid(axis=grid, color="#2F3A44", alpha=0.58, linewidth=0.75)
    axis.tick_params(length=3.5, width=0.8)
    axis.title.set_fontweight("bold")


def _display_figure(
    figure: Any,
    alt: str,
    *,
    max_width: int | None = None,
) -> None:
    from IPython.display import HTML, display

    figure.tight_layout(pad=1.15)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    payload = b64encode(buffer.getvalue()).decode("ascii")
    width_style = (
        f"width:100%;max-width:{max_width}px;" if max_width else "max-width:100%;"
    )
    display(
        HTML(
            '<img src="data:image/png;base64,'
            f'{payload}" alt="{escape(alt, quote=True)}" '
            f'style="display:block;{width_style}height:auto;margin:0 auto;">'
        )
    )
    plt.close(figure)


def plot_batch_ownership(batch: Batch, labels: Sequence[str]) -> Any:
    """Show packed node rows, graph ownership, and pointer boundaries."""

    from matplotlib.patches import FancyBboxPatch, Rectangle

    numbers = batch.atomic_numbers.detach().cpu().reshape(-1).tolist()
    owners = batch.batch_idx.detach().cpu().numpy()
    pointers = batch.batch_ptr.detach().cpu().numpy()
    palette = ("#76B900", "#00A3E0", "#F5B642")
    figure, axis = plt.subplots(figsize=(11.5, 3.2), dpi=160)
    for row, (number, owner) in enumerate(zip(numbers, owners, strict=True)):
        color = palette[int(owner) % len(palette)]
        axis.add_patch(
            FancyBboxPatch(
                (row + 0.06, 0.54),
                0.88,
                0.40,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            row + 0.5,
            0.74,
            chemical_symbols[int(number)],
            color="#0B0F10",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        axis.add_patch(
            FancyBboxPatch(
                (row + 0.06, 0.10),
                0.88,
                0.27,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor=color,
                edgecolor="none",
                alpha=0.76,
            )
        )
        axis.text(
            row + 0.5,
            0.235,
            str(int(owner)),
            color="#0B0F10",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )

    for pointer in pointers:
        axis.vlines(
            pointer,
            ymin=0.02,
            ymax=1.02,
            color="#CDD2D8",
            linestyle="--",
            linewidth=1.0,
            zorder=0,
        )
    for graph, (label, start, stop) in enumerate(
        zip(labels, pointers[:-1], pointers[1:], strict=True)
    ):
        color = palette[graph % len(palette)]
        axis.add_patch(
            Rectangle(
                (start, 0.02),
                stop - start,
                1.0,
                facecolor=color,
                edgecolor="none",
                alpha=0.055,
                zorder=-1,
            )
        )
        center = (start + stop) / 2
        axis.text(
            center,
            1.22,
            f"{graph} · {label}\nrows {start}:{stop}",
            color="#F3F4F6",
            ha="center",
            va="center",
            fontsize=8.2,
        )
    axis.text(-0.35, 0.74, "atomic_numbers", ha="right", va="center", fontsize=8)
    axis.text(-0.35, 0.235, "batch_idx", ha="right", va="center", fontsize=8)
    axis.set_xlim(-2.25, len(owners) + 0.25)
    axis.set_ylim(-0.02, 1.42)
    axis.set_xticks(pointers)
    axis.set_xlabel(f"batch_ptr boundaries: {pointers.tolist()}")
    axis.set_yticks([])
    _polish_axis(axis, grid=None)
    axis.spines[["left", "right", "top"]].set_visible(False)
    return _display_figure(
        figure,
        "Packed atom rows for three molecules. The repeated batch_idx row assigns each atom to a molecule, and vertical lines align with the batch_ptr boundaries.",
    )


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


def prepare_lj_finetuning_experiment(
    seed_batch: Batch,
    *,
    sigma: float,
    cutoff: float,
    reference_epsilon: float,
    baseline_epsilon: float,
    displacement_std: float = 0.08,
    initial_fit_noise: float = 2.0,
    fine_tune_noise: float = 0.005,
    initial_fit_samples: int = 32,
    fine_tune_samples: int = 128,
    test_samples: int = 64,
    batch_size: int = 16,
) -> tuple[float, TorchDataLoader, Batch]:
    """Create a noisy initial fit, cleaner fine-tuning data, and an LJ test set."""

    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    sample_counts = (initial_fit_samples, fine_tune_samples, test_samples)
    if any(count < 1 for count in sample_counts) or batch_size < 1:
        raise ValueError("Sample counts and batch_size must be positive")
    if initial_fit_noise < 0.0 or fine_tune_noise < 0.0:
        raise ValueError("Label-noise scales must be non-negative")

    seed_data = seed_batch.get_data(0).to(torch.device("cpu"))
    split_specs = (
        (initial_fit_samples, 1000),
        (fine_tune_samples, 2000),
        (test_samples, 3000),
    )
    split_records: list[list[AtomicData]] = []
    for count, seed_start in split_specs:
        records = []
        for index in range(count):
            generator = torch.Generator().manual_seed(seed_start + index)
            offset = torch.randn(
                seed_data.positions.shape,
                generator=generator,
                dtype=seed_data.positions.dtype,
            )
            records.append(
                AtomicData(
                    positions=seed_data.positions + displacement_std * offset,
                    atomic_numbers=seed_data.atomic_numbers,
                    cell=seed_data.cell,
                    pbc=seed_data.pbc,
                )
            )
        split_records.append(records)

    all_records = [record for records in split_records for record in records]
    all_batch = Batch.from_data_list(all_records)
    unit_lj = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=sigma,
        cutoff=cutoff,
    )
    unit_lj.set_config("active_outputs", {"energy"})
    compute_neighbors(all_batch, config=unit_lj.model_config.neighbor_config)
    unit_energy = unit_lj(all_batch)["energy"].detach().cpu()

    initial_stop = initial_fit_samples
    fine_tune_stop = initial_stop + fine_tune_samples
    initial_unit = unit_energy[:initial_stop]
    fine_tune_unit = unit_energy[initial_stop:fine_tune_stop]
    true_correction = reference_epsilon - baseline_epsilon
    initial_generator = torch.Generator().manual_seed(21)
    initial_residual = true_correction * initial_unit + initial_fit_noise * torch.randn(
        initial_unit.shape,
        generator=initial_generator,
        dtype=initial_unit.dtype,
    )
    initial_epsilon = float(
        (initial_unit.flatten() @ initial_residual.flatten())
        / (initial_unit.flatten() @ initial_unit.flatten())
    )

    fine_tune_generator = torch.Generator().manual_seed(99)
    fine_tune_residual = (
        true_correction * fine_tune_unit
        + fine_tune_noise
        * torch.randn(
            fine_tune_unit.shape,
            generator=fine_tune_generator,
            dtype=fine_tune_unit.dtype,
        )
    )
    for record, target in zip(split_records[1], fine_tune_residual, strict=True):
        record.add_system_property("energy", target.reshape(1, 1))

    finetune_loader = TorchDataLoader(
        split_records[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: Batch.from_data_list(list(samples)),
    )
    test_batch = Batch.from_data_list(split_records[2])
    return initial_epsilon, finetune_loader, test_batch


def build_argon_pair_curve(distances: torch.Tensor) -> Batch:
    """Build nonperiodic Ar2 systems at the requested pair distances."""

    values = torch.as_tensor(distances)
    records = []
    for distance in values:
        positions = torch.zeros((2, 3), device=values.device, dtype=values.dtype)
        positions[1, 0] = distance
        records.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.full(
                    (2,), 18, device=values.device, dtype=torch.long
                ),
            )
        )
    return Batch.from_data_list(records, device=values.device)


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


def plot_lj_transfer_curve(
    pair_distance: torch.Tensor,
    reference: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
) -> Any:
    """Plot exact, initial, and fine-tuned LJ pair energies."""

    distance = pair_distance.detach().cpu().flatten().numpy()
    truth = reference.detach().cpu().flatten().numpy()
    initial = before.detach().cpu().flatten().numpy()
    tuned = after.detach().cpu().flatten().numpy()
    marker_stride = max(1, len(distance) // 12)
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    axis.plot(
        distance,
        initial,
        color="#D8A657",
        linewidth=2.0,
        linestyle="--",
        zorder=1,
        label="Initial noisy fit",
    )
    axis.plot(
        distance,
        tuned,
        color="#76B900",
        linewidth=3.2,
        zorder=2,
        label="Fine-tuned",
    )
    axis.plot(
        distance,
        truth,
        color="#F3F5F7",
        linewidth=1.2,
        marker="o",
        markersize=4.2,
        markevery=marker_stride,
        markeredgecolor="#11161B",
        markeredgewidth=0.6,
        zorder=3,
        label="Exact LJ",
    )
    axis.axhline(0.0, color="#7C8794", linewidth=0.8, alpha=0.55)
    axis.set(
        title="Ar₂ potential after fine-tuning",
        xlabel="Ar-Ar distance [Å]",
        ylabel="Pair energy [eV]",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        [handles[2], handles[0], handles[1]],
        [labels[2], labels[0], labels[1]],
        frameon=False,
        ncol=3,
    )
    _polish_axis(axis)
    return _display_figure(
        figure,
        "Exact Lennard-Jones Ar2 pair energy, a fit to noisy labels, and the result after fine-tuning on cleaner labels.",
    )


def plot_lj_finetuning_loss(rows: Sequence[dict[str, float]]) -> Any:
    """Plot the residual-energy loss recorded after each optimizer update."""

    steps = [int(row["step"]) for row in rows]
    losses = [row["loss"] for row in rows]
    if not steps or any(loss <= 0.0 for loss in losses):
        raise ValueError("Loss history must contain positive values")

    tick_stride = max(1, len(steps) // 6)
    tick_steps = steps[::tick_stride]
    if tick_steps[-1] != steps[-1]:
        tick_steps.append(steps[-1])

    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.plot(
        steps,
        losses,
        color="#76B900",
        linewidth=2.6,
        marker="o",
        markersize=4.2,
    )
    axis.set_ylim(bottom=0.0)
    axis.set(
        title="Fine-tuning loss",
        xlabel="Optimizer update",
        ylabel="Residual MSE [eV²]",
        xticks=tick_steps,
    )
    _polish_axis(axis)
    return _display_figure(
        figure,
        "Residual-energy mean-squared error recorded after each fine-tuning update.",
        max_width=760,
    )
