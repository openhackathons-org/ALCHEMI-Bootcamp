"""CPU-vs-GPU throughput helpers for the batched-relaxation tutorial.

These keep the timing machinery out of the notebook so the H2O "hello world"
section can make one point cleanly: the GPU's per-call overhead is roughly
fixed, so batching amortizes it. GPU throughput climbs with batch size while
the CPU stays comparatively flat; the crossover is hardware- and model-
dependent and is *measured*, never asserted.

Fairness notes baked in here:
  * Both devices are warmed up once before timing (CPU has its own MKL/thread
    first-call cost, the GPU has kernel/compile warmup).
  * cuEquivariance is CUDA-only, so the CPU model is always built with
    ``enable_cueq=False`` and ``compile_model=False``.
  * CPU thread count dominates the CPU number; it is pinned and reported.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def _resolve_dtype(dtype: Any):
    """Accept a torch.dtype or a string like ``"float32"`` and return a dtype."""
    import torch

    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def _output_tensor(outputs, name: str):
    """Read a named field from a MACE output dict or object."""
    value = outputs.get(name) if isinstance(outputs, dict) else getattr(outputs, name, None)
    if value is None:
        raise RuntimeError(f"MACE output is missing `{name}`.")
    return value


def load_mace_model(
    checkpoint: str,
    *,
    device,
    dtype,
    enable_cueq: bool,
    compile_model: bool,
):
    """Load a MACE checkpoint through Toolkit's ``MACEWrapper`` for one device.

    cuEquivariance and ``torch.compile`` are silently disabled on CPU because
    those acceleration paths require CUDA.
    """
    from nvalchemi.models.mace import MACEWrapper

    resolved_dtype = _resolve_dtype(dtype)
    is_cuda = str(device).startswith("cuda")
    model = MACEWrapper.from_checkpoint(
        checkpoint,
        device=device,
        dtype=resolved_dtype,
        enable_cueq=enable_cueq if is_cuda else False,
        compile_model=compile_model if is_cuda else False,
    )
    model.model_config.active_outputs = {"energy", "forces"}
    return model


def time_relaxation_batch(
    model,
    atoms_list,
    *,
    device,
    dtype,
    n_steps: int,
    dt: float = 0.01,
    fmax: float = 0.05,
) -> dict[str, Any]:
    """Time one batched FIRE2 relaxation of ``atoms_list`` on ``device``.

    Mirrors the explicit Step-4 H2O sequence in the notebook (AtomicData ->
    Batch -> FIRE2.run) and returns throughput/timing for one batch.
    """
    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import ConvergenceHook
    from nvalchemi.dynamics.hooks import NaNDetectorHook
    from nvalchemi.dynamics.optimizers import FIRE2

    resolved_dtype = _resolve_dtype(dtype)
    is_cuda = str(device).startswith("cuda")
    batch_size = len(atoms_list)

    atomic_data = []
    for atoms in atoms_list:
        data = AtomicData.from_atoms(atoms, device="cpu", dtype=resolved_dtype)
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=resolved_dtype)
        atomic_data.append(data)
    batch = Batch.from_data_list(atomic_data, device=device)

    optimizer = FIRE2(
        model,
        dt=dt,
        n_steps=n_steps,
        convergence_hook=ConvergenceHook.from_fmax(
            threshold=fmax,
            source_status=0,
            target_status=1,
        ),
    )
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook)
    optimizer.register_hook(NaNDetectorHook())

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = perf_counter()
    relaxed = optimizer.run(batch)
    if is_cuda:
        torch.cuda.synchronize(device)
    wall_time_s = perf_counter() - start

    energies = [
        float(relaxed.get_data(idx).energy.detach().cpu().reshape(-1)[0])
        for idx in range(batch_size)
    ]
    total_atoms = sum(len(atoms) for atoms in atoms_list)
    return {
        "device": "cuda" if is_cuda else "cpu",
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        "wall_time_s": wall_time_s,
        "structures_per_s": batch_size / wall_time_s,
        "atoms_per_s": total_atoms / wall_time_s,
        "energy_mean_eV": float(np.mean(energies)),
        "energy_std_eV": float(np.std(energies)),
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else float("nan")
        ),
    }


def run_h2o_batch_timing(
    model,
    batch_size: int,
    *,
    device,
    dtype,
    n_steps: int,
    dt: float = 0.01,
    fmax: float = 0.05,
) -> dict[str, Any]:
    """Time one batch of ``batch_size`` independent H2O molecules on ``device``.

    Extracted from the notebook's H2O speedup loop so the same code drives both
    the single-device speedup figure and the CPU-vs-GPU comparison.
    """
    from ase.build import molecule as ase_molecule

    atoms_list = []
    for index in range(batch_size):
        atoms = ase_molecule("H2O")
        atoms.info["structure_id"] = f"H2O_{index}"
        atoms_list.append(atoms)
    return time_relaxation_batch(
        model,
        atoms_list,
        device=device,
        dtype=dtype,
        n_steps=n_steps,
        dt=dt,
        fmax=fmax,
    )


def score_structures_single_batch(
    atoms_list,
    model,
    *,
    device,
    dtype,
) -> dict[str, Any]:
    """Score a list of structures with one neighbor build and one MACE call.

    Returns energies, per-atom forces, the per-structure offsets into the force
    array, and timing/memory. This is a single-point evaluation (no geometry
    relaxation) and is the core of the NH3 92-configuration ranking checkpoint.
    """
    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.neighbors import compute_neighbors

    resolved_dtype = _resolve_dtype(dtype)
    is_cuda = str(device).startswith("cuda")

    atomic_data = []
    for atoms in atoms_list:
        data = AtomicData.from_atoms(atoms, device="cpu", dtype=resolved_dtype)
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=resolved_dtype)
        atomic_data.append(data)
    batch = Batch.from_data_list(atomic_data, device=device)

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = perf_counter()
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    outputs = model(batch)
    if is_cuda:
        torch.cuda.synchronize(device)
    wall_time_s = perf_counter() - start

    energies = _output_tensor(outputs, "energy").detach().cpu().numpy().reshape(-1)
    forces = _output_tensor(outputs, "forces").detach().cpu().numpy().reshape(-1, 3)
    batch_ptr = getattr(batch, "batch_ptr", None)
    if batch_ptr is None:
        offsets = np.cumsum([0, *[len(atoms) for atoms in atoms_list]])
    else:
        offsets = batch_ptr.detach().cpu().numpy().astype(int)

    return {
        "energies": energies,
        "forces": forces,
        "offsets": offsets,
        "n_structures": len(atoms_list),
        "total_atoms": int(sum(len(atoms) for atoms in atoms_list)),
        "wall_time_s": wall_time_s,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else float("nan")
        ),
        "device": "cuda" if is_cuda else "cpu",
    }


def per_structure_force_max(scored: dict[str, Any]) -> list[float]:
    """Largest per-atom force magnitude for each structure in a scored batch."""
    offsets = scored["offsets"]
    forces = scored["forces"]
    return [
        float(np.linalg.norm(forces[offsets[i] : offsets[i + 1]], axis=1).max())
        for i in range(len(offsets) - 1)
    ]


def _available_devices(gpu_available: bool) -> list[str]:
    import torch

    devices = ["cpu"]
    if gpu_available and torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _system_atoms(system: dict, n: int):
    """Build ``n`` ASE structures for one device-comparison system spec.

    ``kind="h2o"`` returns independent water molecules (3 atoms each);
    ``kind="slab"`` cycles the AdsorbML-style adsorbate/slab pool from
    ``build_config_grid`` so the comparison also covers a realistically large
    periodic system, not just a tiny molecule.
    """
    kind = system["kind"]
    if kind == "h2o":
        from ase.build import molecule as ase_molecule

        out = []
        for i in range(n):
            atoms = ase_molecule("H2O")
            atoms.info["structure_id"] = f"H2O_{i}"
            out.append(atoms)
        return out
    if kind == "slab":
        from .config_search import build_config_grid

        pool = [
            cfg.atoms
            for cfg in build_config_grid(
                host_name=system["host"],
                adsorbate_name=system["adsorbate"],
                rotations_deg=(0.0, 60.0, 120.0),
                heights_A=(2.2,),
                frozen_fraction=0.5,
            )
        ]
        if not pool:
            raise ValueError(f"empty slab pool for system {system!r}")
        return [pool[i % len(pool)].copy() for i in range(n)]
    raise ValueError(f"unknown system kind: {kind!r}")


def load_or_run_device_throughput_comparison(
    *,
    systems,
    use_precomputed: bool,
    checkpoint: str,
    dtype,
    cache_path,
    dt: float = 0.01,
    fmax: float = 0.05,
    enable_cueq: bool = True,
    compile_model: bool = False,
    cpu_threads: int | None = None,
    gpu_available: bool = True,
    progress_factory=None,
    relpath_fn=None,
):
    """CPU-vs-GPU throughput across one or more system sizes (hybrid live + cached).

    ``systems`` is a list of dicts, each describing one system size to sweep, e.g.::

        {"label": "H2O molecule", "kind": "h2o",
         "live": [1, 2, 4, 8], "cached": [16, 32, 64, 128, 256], "n_steps": 15}
        {"label": "CO on Cu(111)", "kind": "slab", "host": "Cu(111)",
         "adsorbate": "CO", "live": [1, 2], "cached": [4, 8, 16], "n_steps": 8}

    Each system's ``live`` sizes are always timed live on every available
    device; ``cached`` sizes are read from ``cache_path`` in saved mode (raising
    if the shipped CSV is missing, matching ``load_or_run_adsorption_batch_sweep``)
    or computed live and persisted in compute mode. CPU on a large slab batch is
    slow, which is exactly the point - so those sizes are cached. Returns
    ``(results, meta)`` where ``results`` maps each system label to
    ``{"cpu": df, "gpu": df_or_None, "atoms_per_structure": int}``.
    """
    import torch

    cache_path = Path(cache_path)
    if cpu_threads is not None:
        torch.set_num_threads(int(cpu_threads))
    actual_threads = torch.get_num_threads()

    devices = _available_devices(gpu_available)
    models = {
        device: load_mace_model(
            checkpoint,
            device=device,
            dtype=dtype,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )
        for device in devices
    }
    # Symmetric warmup: one discarded batch-of-1 per device PER system shape, so
    # each panel's first live point does not absorb one-time allocation/first-call cost.
    for device in devices:
        for system in systems:
            time_relaxation_batch(
                models[device], _system_atoms(system, 1), device=device,
                dtype=dtype, n_steps=5, dt=dt, fmax=fmax,
            )

    cached_lookup = None
    if use_precomputed:
        if not cache_path.exists():
            shown = relpath_fn(cache_path) if relpath_fn else str(cache_path)
            raise RuntimeError(
                "Precomputed CPU-vs-GPU throughput table not found at "
                f"{shown}. Set TUTORIAL_RESULT_SOURCE = 'compute' "
                "(USE_SAVED_TUTORIAL_RESULTS = False) to recompute it."
            )
        cached_lookup = pd.read_csv(cache_path)

    total_live = sum(len(devices) * len(set(s["live"])) for s in systems)
    progress = None
    if progress_factory is not None:
        progress = progress_factory(
            title="CPU vs GPU throughput",
            total=max(1, total_live),
            unit="live batches",
            message="warmup complete; timing live batches",
            average_label="s/batch",
            width_px=620,
        )

    live_rows: list[dict[str, Any]] = []
    cached_frames: list[pd.DataFrame] = []
    done = 0
    for system in systems:
        label = system["label"]
        live = sorted(set(int(b) for b in system["live"]))
        cached = sorted(set(int(b) for b in system["cached"]))
        nsteps = int(system["n_steps"])
        for device in devices:
            for bs in live:
                if progress is not None:
                    progress.update(done=done, message=f"{label} | {device}: batch {bs}")
                row = time_relaxation_batch(
                    models[device], _system_atoms(system, bs), device=device,
                    dtype=dtype, n_steps=nsteps, dt=dt, fmax=fmax,
                )
                row["system"] = label
                row["source"] = "live"
                live_rows.append(row)
                done += 1
                if progress is not None:
                    progress.update(done=done, message=f"{label} | {device}: batch {bs} done")
        if use_precomputed:
            part = cached_lookup[
                (cached_lookup["system"].astype(str) == label)
                & (cached_lookup["batch_size"].astype(int).isin(cached))
            ].copy()
            part["source"] = "cached"
            cached_frames.append(part)
        else:
            for device in devices:
                for bs in cached:
                    row = time_relaxation_batch(
                        models[device], _system_atoms(system, bs), device=device,
                        dtype=dtype, n_steps=nsteps, dt=dt, fmax=fmax,
                    )
                    row["system"] = label
                    row["source"] = "cached"
                    cached_frames.append(pd.DataFrame([row]))

    live_df = pd.DataFrame(live_rows)
    cached_df = (
        pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()
    )
    if not use_precomputed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached_df.to_csv(cache_path, index=False)

    combined = pd.concat([live_df, cached_df], ignore_index=True)

    results: dict[str, Any] = {}
    for system in systems:
        label = system["label"]
        sub = combined[combined["system"] == label]
        cpu_df = sub[sub["device"] == "cpu"].sort_values("batch_size").reset_index(drop=True)
        gpu_df = sub[sub["device"] == "cuda"].sort_values("batch_size").reset_index(drop=True)
        results[label] = {
            "cpu": cpu_df,
            "gpu": (None if gpu_df.empty else gpu_df),
            "atoms_per_structure": len(_system_atoms(system, 1)[0]),
        }

    has_gpu = any(r["gpu"] is not None for r in results.values())
    gpu_name = (
        torch.cuda.get_device_name(0)
        if (has_gpu and torch.cuda.is_available())
        else "no GPU"
    )
    meta = {
        "checkpoint": checkpoint,
        "gpu_name": gpu_name,
        "cpu_threads": int(actual_threads),
        "enable_cueq_gpu": bool(enable_cueq),
        "cache_source": str(cache_path),
        "systems": [s["label"] for s in systems],
    }
    return results, meta


def build_nh3_single_batch_tables(scored, nh3_92_table):
    """Build per-structure and summary tables for the NH3 single-batch scoring.

    ``scored`` is the dict returned by :func:`score_structures_single_batch`.
    """
    offsets = scored["offsets"]
    results_df = pd.DataFrame({
        "reference rank": nh3_92_table["dft_rank"].astype(int).to_numpy(),
        "config_id": nh3_92_table["config_id"].to_numpy(),
        "sid": nh3_92_table["sid"].astype(int).to_numpy(),
        "atoms": [int(offsets[i + 1] - offsets[i]) for i in range(scored["n_structures"])],
        "MACE single-point total energy (eV)": scored["energies"],
        "max force (eV/A)": per_structure_force_max(scored),
    })
    results_df["MACE single-point rank"] = (
        results_df["MACE single-point total energy (eV)"]
        .rank(method="first", ascending=True).astype(int))
    summary_df = pd.DataFrame([{
        "ASE structures": scored["n_structures"], "Toolkit Batch objects": 1,
        "MACE model calls": 1, "energies returned": len(scored["energies"]),
        "device": scored["device"], "total atoms": scored["total_atoms"],
        "wall time (s)": scored["wall_time_s"], "peak CUDA memory (GB)": scored["peak_mem_gb"],
    }])
    return results_df, summary_df
