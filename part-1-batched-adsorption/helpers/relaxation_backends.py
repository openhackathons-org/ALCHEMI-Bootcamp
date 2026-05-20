"""Toolkit relaxation engine for adsorption geometry relaxations.

The tutorial path is native ALCHEMI Toolkit:
``AtomicData`` -> ``Batch.from_data_list`` -> ``FIRE2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import types
from typing import Literal, Protocol

import numpy as np

from .cache import cache_exists, load_cache, save_cache
from .models import (
    AtomicStructurePayload,
    OptimizationResult,
    RelaxationBatchResult,
    atomic_data_to_ase,
)

ToolkitExecutionName = Literal["toolkit"]


class BackendUnavailableError(RuntimeError):
    """Raised when the Toolkit engine cannot satisfy its native contract."""


@dataclass(frozen=True)
class ToolkitD3BJConfig:
    """Explicit DFT-D3(BJ) parameters for Toolkit-side parity runs.

    The engine intentionally has no hidden D3 defaults. D3 parameters must be
    supplied from verified workflow metadata or documentation.
    """

    a1: float
    a2: float
    s8: float
    s6: float = 1.0
    cutoff: float = 15.0
    k1: float = 16.0
    k3: float = -4.0
    smoothing_fraction: float = 0.2
    auto_download: bool = True
    param_file: str | None = None


@dataclass(frozen=True)
class RelaxationBackendConfig:
    """Configuration for the native Toolkit relaxation engine."""

    name: ToolkitExecutionName = "toolkit"
    cache_dir: str = "outputs/live_runs/manual/cache_json"
    use_cached_responses: bool = False

    # Toolkit route.
    toolkit_checkpoint: str = "medium-mpa-0"
    toolkit_device: str = "cuda"
    toolkit_dtype: str = "float32"
    toolkit_enable_cueq: bool = False
    toolkit_compile_model: bool = False
    toolkit_head: str | None = None
    toolkit_dt: float = 0.01
    toolkit_n_steps: int = 5000
    toolkit_fmax: float = 0.05
    toolkit_d3bj: ToolkitD3BJConfig | None = None
    toolkit_require_d3bj: bool = True


class RelaxationBackend(Protocol):
    """Common relaxation interface used by the notebook."""

    name: ToolkitExecutionName

    def relax(
        self,
        atoms_list: list[AtomicStructurePayload],
        label: str,
        *,
        cellopt: bool = False,
    ) -> RelaxationBatchResult:
        """Relax a batch of structures and return Toolkit relaxation results."""

    async def async_relax(
        self,
        atoms_list: list[AtomicStructurePayload],
        label: str,
        *,
        cellopt: bool = False,
        session: object | None = None,
    ) -> RelaxationBatchResult:
        """Async-compatible variant used by notebook batch orchestration."""


@dataclass(frozen=True)
class _ToolkitApi:
    torch: object
    AtomicData: object
    Batch: object
    FIRE2: object
    FIRE2VariableCell: object
    ConvergenceHook: object
    DynamicsStage: object
    FreezeAtomsHook: object
    NaNDetectorHook: object
    MACEWrapper: object
    DFTD3ModelWrapper: object
    PipelineGroup: object
    PipelineModelWrapper: object


def _import_required(module_name: str, *, context: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or module_name
        raise BackendUnavailableError(
            f"Toolkit execution selected, but the native ALCHEMI "
            f"Toolkit API is not importable while loading {context}. Missing "
            f"module: {missing!r}. Install "
            f"`nvalchemi-toolkit[ase,mace]` and `nvalchemi-toolkit-ops` in "
            f"this kernel/environment."
        ) from exc


def _require_attr(module: object, attr: str, *, module_name: str) -> object:
    if not hasattr(module, attr):
        raise BackendUnavailableError(
            f"Toolkit execution selected, but installed "
            f"{module_name} does not expose required native API `{attr}`. "
            f"The tutorial requires Toolkit batching/relaxation via "
            f"AtomicData, Batch.from_data_list, MACEWrapper, and FIRE2."
        )
    return getattr(module, attr)


def require_toolkit_api() -> _ToolkitApi:
    """Import and validate the native Toolkit symbols used by the engine."""

    torch = _import_required("torch", context="PyTorch tensors")
    data = _import_required("nvalchemi.data", context="AtomicData/Batch")
    fire2 = _import_required(
        "nvalchemi.dynamics.optimizers.fire2",
        context="FIRE2 optimizer",
    )
    dynamics_base = _import_required(
        "nvalchemi.dynamics.base",
        context="ConvergenceHook/DynamicsStage",
    )
    dynamics_hooks = _import_required(
        "nvalchemi.dynamics.hooks",
        context="FreezeAtomsHook/NaNDetectorHook",
    )
    mace = _import_required("nvalchemi.models.mace", context="MACEWrapper")
    dftd3 = _import_required(
        "nvalchemi.models.dftd3",
        context="DFTD3ModelWrapper",
    )
    pipeline = _import_required(
        "nvalchemi.models.pipeline",
        context="PipelineModelWrapper",
    )

    AtomicData = _require_attr(data, "AtomicData", module_name="nvalchemi.data")
    Batch = _require_attr(data, "Batch", module_name="nvalchemi.data")
    if not hasattr(Batch, "from_data_list") or not hasattr(Batch, "get_data"):
        raise BackendUnavailableError(
            "Toolkit execution selected, but nvalchemi.data.Batch "
            "does not provide both `from_data_list` and `get_data`. Native "
            "batched geometry relaxation cannot be constructed."
        )

    return _ToolkitApi(
        torch=torch,
        AtomicData=AtomicData,
        Batch=Batch,
        FIRE2=_require_attr(fire2, "FIRE2", module_name=fire2.__name__),
        FIRE2VariableCell=_require_attr(
            fire2,
            "FIRE2VariableCell",
            module_name=fire2.__name__,
        ),
        ConvergenceHook=_require_attr(
            dynamics_base,
            "ConvergenceHook",
            module_name=dynamics_base.__name__,
        ),
        DynamicsStage=_require_attr(
            dynamics_base,
            "DynamicsStage",
            module_name=dynamics_base.__name__,
        ),
        FreezeAtomsHook=_require_attr(
            dynamics_hooks,
            "FreezeAtomsHook",
            module_name=dynamics_hooks.__name__,
        ),
        NaNDetectorHook=_require_attr(
            dynamics_hooks,
            "NaNDetectorHook",
            module_name=dynamics_hooks.__name__,
        ),
        MACEWrapper=_require_attr(mace, "MACEWrapper", module_name=mace.__name__),
        DFTD3ModelWrapper=_require_attr(
            dftd3,
            "DFTD3ModelWrapper",
            module_name=dftd3.__name__,
        ),
        PipelineGroup=_require_attr(
            pipeline,
            "PipelineGroup",
            module_name=pipeline.__name__,
        ),
        PipelineModelWrapper=_require_attr(
            pipeline,
            "PipelineModelWrapper",
            module_name=pipeline.__name__,
        ),
    )


def check_toolkit_native_api() -> dict[str, str | bool]:
    """Return a notebook-friendly Toolkit API availability report."""

    try:
        require_toolkit_api()
    except BackendUnavailableError as exc:
        return {"available": False, "message": str(exc)}
    return {
        "available": True,
        "message": (
            "Native Toolkit API available: AtomicData, Batch.from_data_list, "
            "MACEWrapper, DFTD3ModelWrapper, PipelineModelWrapper, and FIRE2."
        ),
    }


class ToolkitBackend:
    """Native ALCHEMI Toolkit relaxation backend."""

    name: ToolkitExecutionName = "toolkit"

    def __init__(self, config: RelaxationBackendConfig) -> None:
        self.config = config
        if config.use_cached_responses:
            self.api = None
            self.device = None
            self.dtype = None
            self.model = None
            return
        self.api = require_toolkit_api()
        if config.toolkit_require_d3bj and config.toolkit_d3bj is None:
            raise BackendUnavailableError(
                "Toolkit execution selected, but DFT-D3(BJ) "
                "parameters were not supplied. D3-enabled Toolkit runs require "
                "an explicit ToolkitD3BJConfig(a1=..., a2=..., s8=...) from "
                "verified workflow metadata. The default tutorial sets "
                "toolkit_require_d3bj=False to match the non-D3 OC20Dense "
                "reference convention."
            )
        self.device = self._resolve_device(config.toolkit_device)
        self.dtype = self._resolve_dtype(config.toolkit_dtype)
        self.model = self._build_model()

    def _resolve_device(self, device_name: str) -> object:
        torch = self.api.torch
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailableError(
                f"Toolkit execution selected with "
                f"toolkit_device={device_name!r}, but PyTorch reports CUDA is "
                f"unavailable. Choose toolkit_device='cpu' for a CPU validation "
                f"run or run on a CUDA-capable Toolkit environment."
            )
        return torch.device(device_name)

    def _resolve_dtype(self, dtype_name: str) -> object:
        torch = self.api.torch
        if not hasattr(torch, dtype_name):
            raise BackendUnavailableError(
                f"Toolkit execution selected with unsupported "
                f"toolkit_dtype={dtype_name!r}. Expected a torch dtype name "
                f"such as 'float32' or 'float64'."
            )
        dtype = getattr(torch, dtype_name)
        if not getattr(dtype, "is_floating_point", False):
            raise BackendUnavailableError(
                f"Toolkit execution selected with non-floating "
                f"toolkit_dtype={dtype_name!r}."
            )
        return dtype

    def _load_cached_reply(self, label: str) -> RelaxationBatchResult:
        if cache_exists(self.config.cache_dir, label):
            print(f"  Loading cached response: {label}")
            return load_cache(self.config.cache_dir, label, RelaxationBatchResult)
        raise RuntimeError(
            f"No precomputed Toolkit response for '{label}'. In the notebook, "
            f"set USE_SAVED_TUTORIAL_RESULTS = False to compute it on a "
            f"GPU-capable cluster environment, or provide the cached JSON under "
            f"{self.config.cache_dir}/."
        )

    def _build_model(self) -> object:
        try:
            mace = self.api.MACEWrapper.from_checkpoint(
                self.config.toolkit_checkpoint,
                device=self.device,
                dtype=self.dtype,
                enable_cueq=self.config.toolkit_enable_cueq,
                compile_model=self.config.toolkit_compile_model,
            )
            if self.config.toolkit_head:
                self._select_mace_head(mace, self.config.toolkit_head)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Toolkit execution selected, but "
                f"MACEWrapper.from_checkpoint({self.config.toolkit_checkpoint!r}) "
                f"failed: {type(exc).__name__}: {exc}. This is a native Toolkit "
                f"or MACE checkpoint/API failure."
            ) from exc

        groups = [self.api.PipelineGroup(steps=[mace])]
        if self.config.toolkit_d3bj is not None:
            d3cfg = self.config.toolkit_d3bj
            try:
                d3 = self.api.DFTD3ModelWrapper(
                    a1=d3cfg.a1,
                    a2=d3cfg.a2,
                    s8=d3cfg.s8,
                    s6=d3cfg.s6,
                    cutoff=d3cfg.cutoff,
                    k1=d3cfg.k1,
                    k3=d3cfg.k3,
                    smoothing_fraction=d3cfg.smoothing_fraction,
                    auto_download=d3cfg.auto_download,
                    param_file=d3cfg.param_file,
                ).to(self.device)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Toolkit execution selected with explicit "
                    f"DFT-D3(BJ), but DFTD3ModelWrapper construction failed: "
                    f"{type(exc).__name__}: {exc}."
                ) from exc
            groups.append(self.api.PipelineGroup(steps=[d3]))

        model = self.api.PipelineModelWrapper(groups=groups)
        model.model_config.active_outputs = {"energy", "forces"}
        return model

    def _select_mace_head(self, mace: object, head: str) -> None:
        """Select a named head for multi-head MACE checkpoints.

        The Toolkit MACE wrapper accepts named multi-head checkpoints, but the
        wrapped model only sees a head choice if the input dictionary contains a
        ``head`` tensor. We inject that tensor here so head selection is explicit
        and reproducible for OC20/MH-style checkpoints.
        """

        torch = self.api.torch
        heads = list(getattr(getattr(mace, "model", None), "heads", []))
        if not heads:
            raise BackendUnavailableError(
                f"toolkit_head={head!r} was requested, but checkpoint "
                f"{self.config.toolkit_checkpoint!r} does not expose named heads."
            )
        if head not in heads:
            raise BackendUnavailableError(
                f"toolkit_head={head!r} was requested, but checkpoint "
                f"{self.config.toolkit_checkpoint!r} exposes heads: {heads}."
            )
        head_index = heads.index(head)
        original_adapt_input = mace.adapt_input

        def adapt_input_with_head(wrapper_self: object, data: object, **kwargs: object) -> dict[str, object]:
            inputs = original_adapt_input(data, **kwargs)
            ptr = inputs.get("ptr")
            if ptr is None:
                num_graphs = 1
            else:
                num_graphs = int(ptr.numel() - 1)
            device = inputs["positions"].device
            inputs["head"] = torch.full(
                (num_graphs,),
                head_index,
                dtype=torch.long,
                device=device,
            )
            return inputs

        mace.adapt_input = types.MethodType(adapt_input_with_head, mace)
        setattr(mace, "selected_head", head)
        setattr(mace, "available_heads", heads)

    def _to_atomic_data(self, payload: AtomicStructurePayload) -> object:
        torch = self.api.torch
        coord = torch.tensor(
            np.asarray(payload.coord, dtype=float).reshape(-1, 3),
            dtype=self.dtype,
            device=self.device,
        )
        numbers = torch.tensor(payload.numbers, dtype=torch.long, device=self.device)
        fields = {
            "positions": coord,
            "atomic_numbers": numbers,
            "atomic_masses": torch.tensor(
                self._atomic_masses(payload.numbers),
                dtype=self.dtype,
                device=self.device,
            ),
            "forces": torch.zeros_like(coord),
            "energy": torch.zeros(1, 1, dtype=self.dtype, device=self.device),
            "velocities": torch.zeros_like(coord),
        }
        if payload.cell is not None:
            fields["cell"] = torch.tensor(
                np.asarray(payload.cell, dtype=float).reshape(1, 3, 3),
                dtype=self.dtype,
                device=self.device,
            )
        if payload.pbc is not None:
            fields["pbc"] = torch.tensor(
                [payload.pbc],
                dtype=torch.bool,
                device=self.device,
            )
        if payload.charge is not None:
            fields["charge"] = torch.tensor(
                [[payload.charge]],
                dtype=self.dtype,
                device=self.device,
            )
        if payload.active_mask is not None:
            if len(payload.active_mask) != len(payload.numbers):
                raise ValueError(
                    f"active_mask length mismatch for {payload.structure_id!r}: "
                    f"{len(payload.active_mask)} mask entries for "
                    f"{len(payload.numbers)} atoms."
                )
            active = torch.tensor(
                payload.active_mask,
                dtype=torch.bool,
                device=self.device,
            )
            categories = torch.zeros(len(payload.numbers), dtype=torch.long, device=self.device)
            categories[~active] = -1  # FreezeAtomsHook default AtomCategory.SPECIAL.
            fields["atom_categories"] = categories
        return self.api.AtomicData(**fields)

    @staticmethod
    def _atomic_masses(numbers: list[int]) -> list[float]:
        from ase.data import atomic_masses

        return [float(atomic_masses[number]) for number in numbers]

    def _to_result(
        self,
        data: object,
        original: AtomicStructurePayload,
        *,
        nsteps: int,
    ) -> OptimizationResult:
        positions = self._required_tensor(data, "positions").reshape(-1, 3)
        forces = self._required_tensor(data, "forces").reshape(-1, 3)
        energy = float(self._required_tensor(data, "energy").reshape(-1)[0])
        fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0

        cell_tensor = self._optional_tensor(data, "cell")
        pbc_tensor = self._optional_tensor(data, "pbc")
        stress_tensor = self._optional_tensor(data, "stress")
        charges_tensor = self._optional_tensor(data, "charges")

        cell = (
            cell_tensor.reshape(3, 3).flatten().tolist()
            if cell_tensor is not None
            else original.cell
        )
        pbc = (
            [bool(x) for x in pbc_tensor.reshape(-1).tolist()]
            if pbc_tensor is not None
            else original.pbc
        )

        return OptimizationResult(
            coord=positions.flatten().tolist(),
            numbers=list(original.numbers),
            charge=original.charge,
            mult=original.mult,
            cell=cell,
            pbc=pbc,
            structure_id=original.structure_id,
            active_mask=original.active_mask,
            converged=fmax <= self.config.toolkit_fmax,
            optimizer_nsteps=nsteps,
            energy=energy,
            forces=forces.flatten().tolist(),
            stress=stress_tensor.flatten().tolist() if stress_tensor is not None else None,
            charges=charges_tensor.flatten().tolist()
            if charges_tensor is not None
            else None,
        )

    @staticmethod
    def _required_tensor(data: object, name: str) -> np.ndarray:
        tensor = getattr(data, name, None)
        if tensor is None:
            raise BackendUnavailableError(
                f"Toolkit relaxation completed, but result is missing required "
                f"`{name}` tensor. Native Toolkit API contract changed or the "
                f"selected model did not produce required outputs."
            )
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _optional_tensor(data: object, name: str) -> np.ndarray | None:
        tensor = getattr(data, name, None)
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy()

    def relax(
        self,
        atoms_list: list[AtomicStructurePayload],
        label: str,
        *,
        cellopt: bool = False,
    ) -> RelaxationBatchResult:
        if self.config.use_cached_responses:
            return self._load_cached_reply(label)

        if not atoms_list:
            raise ValueError("Cannot relax an empty batch.")

        data_list = [self._to_atomic_data(payload) for payload in atoms_list]
        batch = self.api.Batch.from_data_list(data_list, device=self.device)
        optimizer_cls = self.api.FIRE2VariableCell if cellopt else self.api.FIRE2
        optimizer = optimizer_cls(
            model=self.model,
            dt=self.config.toolkit_dt,
            n_steps=self.config.toolkit_n_steps,
            convergence_hook=self.api.ConvergenceHook.from_fmax(
                threshold=self.config.toolkit_fmax,
                source_status=0,
                target_status=1,
            ),
        )
        for hook in self.model.make_neighbor_hooks():
            optimizer.register_hook(hook)
        if any(payload.active_mask is not None for payload in atoms_list):
            optimizer.register_hook(self.api.FreezeAtomsHook())
        optimizer.register_hook(self.api.NaNDetectorHook())

        batch = optimizer.run(batch)
        step_count = int(getattr(optimizer, "step_count", self.config.toolkit_n_steps))
        results = [
            self._to_result(batch.get_data(i), original, nsteps=step_count)
            for i, original in enumerate(atoms_list)
        ]
        reply = RelaxationBatchResult(
            atoms=results,
            status="Success",
            info=(
                f"toolkit native FIRE2 batch relaxation; label={label}; "
                f"checkpoint={self.config.toolkit_checkpoint}; "
                f"head={self.config.toolkit_head or 'default'}; "
                f"device={self.device}; d3bj={self.config.toolkit_d3bj is not None}"
            ),
        )
        save_cache(self.config.cache_dir, label, reply)
        print(f"  Cached response saved: {label}")
        return reply

    async def async_relax(
        self,
        atoms_list: list[AtomicStructurePayload],
        label: str,
        *,
        cellopt: bool = False,
        session: object | None = None,
    ) -> RelaxationBatchResult:
        return self.relax(atoms_list, label=label, cellopt=cellopt)


def run_toolkit_relaxation_with_trajectory(
    backend: ToolkitBackend,
    atoms_list: list[AtomicStructurePayload],
    label: str,
    *,
    trajectory_paths: list[str | Path],
    log_paths: list[str | Path],
    cellopt: bool = False,
    snapshot_frequency: int = 1,
) -> RelaxationBatchResult:
    """Run native Toolkit relaxation and persist per-structure trajectories.

    This is intentionally Toolkit-specific. It mirrors ``ToolkitBackend.relax``
    but registers Toolkit's ``SnapshotHook`` so human reviewers can inspect the
    actual relaxation path, not only initial/final structures.
    """

    if not isinstance(backend, ToolkitBackend):
        raise TypeError("Trajectory capture is implemented for ToolkitBackend only.")
    if backend.config.use_cached_responses:
        raise RuntimeError("Trajectory capture requires a fresh Toolkit run, not cache replay.")
    if not atoms_list:
        raise ValueError("Cannot relax an empty batch.")
    if len(trajectory_paths) != len(atoms_list):
        raise ValueError("trajectory_paths length must match atoms_list length.")
    if len(log_paths) != len(atoms_list):
        raise ValueError("log_paths length must match atoms_list length.")
    if snapshot_frequency < 1:
        raise ValueError("snapshot_frequency must be >= 1.")

    from ase.io import write as ase_write
    from nvalchemi.dynamics.hooks import SnapshotHook
    from nvalchemi.dynamics.sinks import HostMemory

    data_list = [backend._to_atomic_data(payload) for payload in atoms_list]
    batch = backend.api.Batch.from_data_list(data_list, device=backend.device)
    optimizer_cls = backend.api.FIRE2VariableCell if cellopt else backend.api.FIRE2
    optimizer = optimizer_cls(
        model=backend.model,
        dt=backend.config.toolkit_dt,
        n_steps=backend.config.toolkit_n_steps,
        convergence_hook=backend.api.ConvergenceHook.from_fmax(
            threshold=backend.config.toolkit_fmax,
            source_status=0,
            target_status=1,
        ),
    )
    for hook in backend.model.make_neighbor_hooks():
        optimizer.register_hook(hook)
    if any(payload.active_mask is not None for payload in atoms_list):
        optimizer.register_hook(backend.api.FreezeAtomsHook())
    optimizer.register_hook(backend.api.NaNDetectorHook())

    sink = HostMemory(
        capacity=max(1, backend.config.toolkit_n_steps) * len(atoms_list)
    )
    optimizer.register_hook(SnapshotHook(sink=sink, frequency=snapshot_frequency))

    batch = optimizer.run(batch)
    step_count = int(getattr(optimizer, "step_count", backend.config.toolkit_n_steps))
    results = [
        backend._to_result(batch.get_data(i), original, nsteps=step_count)
        for i, original in enumerate(atoms_list)
    ]

    if len(sink) > 0:
        snapshots = sink.read().to_data_list()
        expected = len(atoms_list)
        for structure_index, original in enumerate(atoms_list):
            frames = []
            log_rows = [
                "step,structure_id,energy_eV,max_force_eV_A,"
                "free_max_force_eV_A,converged"
            ]
            active_mask = (
                np.asarray(original.active_mask, dtype=bool)
                if original.active_mask is not None
                else None
            )
            for frame_index in range(structure_index, len(snapshots), expected):
                step = (frame_index // expected) * snapshot_frequency
                result = backend._to_result(
                    snapshots[frame_index],
                    original,
                    nsteps=step,
                )
                atoms = atomic_data_to_ase(result)
                forces = np.asarray(result.forces, dtype=float).reshape(-1, 3)
                force_norm = np.linalg.norm(forces, axis=1) if len(forces) else np.array([])
                max_force = float(force_norm.max()) if len(force_norm) else 0.0
                if active_mask is not None and active_mask.any():
                    free_max_force = float(force_norm[active_mask].max())
                else:
                    free_max_force = max_force
                converged = free_max_force <= backend.config.toolkit_fmax
                atoms.info["structure_id"] = original.structure_id or ""
                atoms.info["optimizer_step"] = int(step)
                atoms.info["energy_eV"] = float(result.energy)
                atoms.info["max_force_eV_A"] = max_force
                atoms.info["free_max_force_eV_A"] = free_max_force
                atoms.info["converged"] = bool(converged)
                atoms.arrays["forces"] = forces
                frames.append(atoms)
                log_rows.append(
                    f"{step},{original.structure_id or ''},{float(result.energy)},"
                    f"{max_force},{free_max_force},{bool(converged)}"
                )

            trajectory_path = Path(trajectory_paths[structure_index])
            log_path = Path(log_paths[structure_index])
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            ase_write(trajectory_path, frames, format="extxyz")
            log_path.write_text("\n".join(log_rows) + "\n", encoding="utf-8")

    reply = RelaxationBatchResult(
        atoms=results,
        status="Success",
        info=(
            f"toolkit native FIRE2 batch relaxation with trajectory capture; "
            f"label={label}; checkpoint={backend.config.toolkit_checkpoint}; "
            f"head={backend.config.toolkit_head or 'default'}; "
            f"device={backend.device}; d3bj={backend.config.toolkit_d3bj is not None}; "
            f"snapshot_frequency={snapshot_frequency}"
        ),
    )
    save_cache(backend.config.cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


def get_relaxation_backend(config: RelaxationBackendConfig) -> RelaxationBackend:
    """Construct the Toolkit relaxation engine used by the tutorial."""

    if config.name == "toolkit":
        return ToolkitBackend(config)
    raise ValueError(
        f"Unsupported relaxation engine {config.name!r}. "
        "This notebook exposes the native Toolkit path only."
    )
