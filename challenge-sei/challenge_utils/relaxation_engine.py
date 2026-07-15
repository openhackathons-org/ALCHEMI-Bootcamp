"""Step-by-step Toolkit relaxation engine for the SEI Pareto challenge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------
# Step 1 - device and precision
# --------------------------------------------------------------------------
def resolve_device_and_dtype(device_name: str = "auto", dtype_name: str = "float32"):
    """Pick the torch device ('auto' -> cuda when available) and float dtype."""
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device_name!r} requested but CUDA is unavailable; use 'cpu' or 'auto'."
        )
    dtype = getattr(torch, dtype_name, None)
    if dtype is None or not getattr(dtype, "is_floating_point", False):
        raise RuntimeError(f"dtype must be a floating torch dtype name, got {dtype_name!r}.")
    return torch.device(device_name), dtype


# --------------------------------------------------------------------------
# Step 2 - the machine-learned interatomic potential
# --------------------------------------------------------------------------
def load_mlip(checkpoint: str, device, dtype, *, enable_cueq: bool = False,
              compile_model: bool = False):
    """Load the MACE MLIP wrapper for ``checkpoint`` (e.g. 'medium-mpa-0').
    """
    import importlib.util

    from nvalchemi.models.mace import MACEWrapper

    if enable_cueq and importlib.util.find_spec("cuequivariance_ops_torch") is None:
        print("load_mlip: cuequivariance kernels not installed "
              "(nvalchemi-toolkit[cu13]); falling back to enable_cueq=False.")
        enable_cueq = False

    return MACEWrapper.from_checkpoint(
        checkpoint, device=device, dtype=dtype,
        enable_cueq=enable_cueq, compile_model=compile_model,
    )


# --------------------------------------------------------------------------
# Step 3 - the model pipeline the optimizer will call
# --------------------------------------------------------------------------
def assemble_pipeline(mlip, device, *, d3bj=None):
    """Wrap the MLIP in a PipelineModelWrapper (optionally adding DFT-D3(BJ))
    and request exactly the outputs a relaxation needs: energy + forces."""
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    groups = [PipelineGroup(steps=[mlip])]
    if d3bj is not None:
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        d3 = DFTD3ModelWrapper(
            a1=d3bj.a1, a2=d3bj.a2, s8=d3bj.s8, s6=d3bj.s6, cutoff=d3bj.cutoff,
            k1=d3bj.k1, k3=d3bj.k3, smoothing_fraction=d3bj.smoothing_fraction,
            auto_download=d3bj.auto_download, param_file=d3bj.param_file,
        ).to(device)
        groups.append(PipelineGroup(steps=[d3]))
    model = PipelineModelWrapper(groups=groups)
    model.model_config.active_outputs = {"energy", "forces"}
    return model


# --------------------------------------------------------------------------
# Step 4 - one structure payload -> Toolkit AtomicData
# --------------------------------------------------------------------------
def payload_to_atomic_data(payload, device, dtype):
    """Convert an AtomicStructurePayload into a Toolkit AtomicData object.

    Frozen atoms (active_mask False) become atom_categories == -1, which the
    FreezeAtomsHook pins in place during the relaxation.
    """
    import torch
    from ase.data import atomic_masses
    from nvalchemi.data import AtomicData

    coord = torch.tensor(np.asarray(payload.coord, dtype=float).reshape(-1, 3),
                         dtype=dtype, device=device)
    numbers = torch.tensor(payload.numbers, dtype=torch.long, device=device)
    fields = {
        "positions": coord,
        "atomic_numbers": numbers,
        "atomic_masses": torch.tensor([float(atomic_masses[n]) for n in payload.numbers],
                                      dtype=dtype, device=device),
        "forces": torch.zeros_like(coord),
        "energy": torch.zeros(1, 1, dtype=dtype, device=device),
        "velocities": torch.zeros_like(coord),
    }
    if payload.cell is not None:
        fields["cell"] = torch.tensor(np.asarray(payload.cell, dtype=float).reshape(1, 3, 3),
                                      dtype=dtype, device=device)
    if payload.pbc is not None:
        fields["pbc"] = torch.tensor([payload.pbc], dtype=torch.bool, device=device)
    if payload.charge is not None:
        fields["charge"] = torch.tensor([[payload.charge]], dtype=dtype, device=device)
    if payload.active_mask is not None:
        if len(payload.active_mask) != len(payload.numbers):
            raise ValueError(
                f"active_mask length mismatch for {payload.structure_id!r}: "
                f"{len(payload.active_mask)} entries for {len(payload.numbers)} atoms."
            )
        active = torch.tensor(payload.active_mask, dtype=torch.bool, device=device)
        categories = torch.zeros(len(payload.numbers), dtype=torch.long, device=device)
        categories[~active] = -1  # FreezeAtomsHook pins category -1 (SPECIAL) atoms
        fields["atom_categories"] = categories
    return AtomicData(**fields)


# --------------------------------------------------------------------------
# Step 5 - batched FIRE2 relaxation
# --------------------------------------------------------------------------
def relax_batch(model, payloads, device, dtype, *, dt: float = 0.005,
                n_steps: int = 5000, fmax: float = 0.05, maxstep: float | None = 0.04):
    """Relax a list of payloads together in ONE Toolkit batch.

    AtomicData list -> Batch.from_data_list -> FIRE2 with a force-convergence
    hook, the model's neighbor-list hooks, FreezeAtomsHook (frozen slab atoms),
    and NaNDetectorHook -> per-structure OptimizationResult list.
    """
    from helpers.models import OptimizationResult  # part-1 helpers (sys.path)
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.dynamics.hooks import FreezeAtomsHook, NaNDetectorHook
    from nvalchemi.dynamics.optimizers.fire2 import FIRE2

    if not payloads:
        raise ValueError("Cannot relax an empty batch.")

    data_list = [payload_to_atomic_data(p, device, dtype) for p in payloads]
    batch = Batch.from_data_list(data_list, device=device)

    optimizer_kwargs = {} if maxstep is None else {"maxstep": maxstep}
    optimizer = FIRE2(
        model=model, dt=dt, n_steps=n_steps,
        convergence_hook=ConvergenceHook.from_fmax(threshold=fmax,
                                                   source_status=0, target_status=1),
        **optimizer_kwargs,
    )
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook)
    if any(p.active_mask is not None for p in payloads):
        optimizer.register_hook(FreezeAtomsHook())
    optimizer.register_hook(NaNDetectorHook())

    batch = optimizer.run(batch)
    nsteps = int(getattr(optimizer, "step_count", n_steps))

    results = []
    for i, original in enumerate(payloads):
        data = batch.get_data(i)
        positions = data.positions.detach().cpu().numpy().reshape(-1, 3)
        forces = data.forces.detach().cpu().numpy().reshape(-1, 3)
        energy = float(data.energy.detach().cpu().numpy().reshape(-1)[0])
        result_fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        cell = getattr(data, "cell", None)
        pbc = getattr(data, "pbc", None)
        results.append(OptimizationResult(
            coord=positions.flatten().tolist(),
            numbers=list(original.numbers),
            charge=original.charge,
            mult=original.mult,
            cell=(cell.detach().cpu().numpy().reshape(3, 3).flatten().tolist()
                  if cell is not None else original.cell),
            pbc=([bool(x) for x in pbc.detach().cpu().numpy().reshape(-1).tolist()]
                 if pbc is not None else original.pbc),
            structure_id=original.structure_id,
            active_mask=original.active_mask,
            converged=result_fmax <= fmax,
            optimizer_nsteps=nsteps,
            energy=energy,
            forces=forces.flatten().tolist(),
        ))
    return results


# --------------------------------------------------------------------------
# Step 6 - the engine object the challenge workflow consumes
# --------------------------------------------------------------------------
@dataclass
class StepByStepRelaxationEngine:
    """The assembled engine: `.relax(payloads, label)` -> RelaxationBatchResult.
    """

    model: object
    device: object
    dtype: object
    checkpoint: str
    cache_dir: str = "outputs/cache_json"
    use_cached_responses: bool = False
    dt: float = 0.005
    n_steps: int = 5000
    fmax: float = 0.05
    maxstep: float | None = 0.04
    name: str = field(default="toolkit", init=False)

    def relax(self, atoms_list, label: str, *, cellopt: bool = False):
        from helpers.cache import cache_exists, load_cache, save_cache
        from helpers.models import RelaxationBatchResult

        if cellopt:
            raise NotImplementedError(
                "The step-by-step challenge engine relaxes at fixed cell; "
                "see Part 1's FIRE2VariableCell for cell optimisation."
            )
        if self.use_cached_responses:
            if cache_exists(self.cache_dir, label):
                print(f"  Loading cached response: {label}")
                return load_cache(self.cache_dir, label, RelaxationBatchResult)
            raise RuntimeError(f"No cached Toolkit response for '{label}' in {self.cache_dir}.")

        results = relax_batch(
            self.model, list(atoms_list), self.device, self.dtype,
            dt=self.dt, n_steps=self.n_steps, fmax=self.fmax, maxstep=self.maxstep,
        )
        reply = RelaxationBatchResult(
            atoms=results,
            status="Success",
            info=(f"step-by-step FIRE2 batch relaxation; label={label}; "
                  f"checkpoint={self.checkpoint}; device={self.device}"),
        )
        save_cache(self.cache_dir, label, reply)
        print(f"  Cached response saved: {label}")
        return reply

    async def async_relax(self, atoms_list, label: str, *, cellopt: bool = False,
                          session: object | None = None):
        return self.relax(atoms_list, label=label, cellopt=cellopt)


def build_step_by_step_engine(*, checkpoint: str, device: str = "auto",
                              dtype: str = "float32", enable_cueq: bool = False,
                              compile_model: bool = False, d3bj=None,
                              cache_dir: str = "outputs/cache_json",
                              use_cached_responses: bool = False, dt: float = 0.005,
                              n_steps: int = 5000, fmax: float = 0.05,
                              maxstep: float | None = 0.04) -> StepByStepRelaxationEngine:

    device_obj, dtype_obj = resolve_device_and_dtype(device, dtype)          # Step 1
    mlip = load_mlip(checkpoint, device_obj, dtype_obj,
                     enable_cueq=enable_cueq, compile_model=compile_model)   # Step 2
    model = assemble_pipeline(mlip, device_obj, d3bj=d3bj)                   # Step 3
    return StepByStepRelaxationEngine(                                       # Step 6
        model=model, device=device_obj, dtype=dtype_obj, checkpoint=checkpoint,
        cache_dir=cache_dir, use_cached_responses=use_cached_responses,
        dt=dt, n_steps=n_steps, fmax=fmax, maxstep=maxstep,
    )
