"""Benchmark Toolkit batch sizes for real adsorption structures.

The H2O saturation benchmark measures a clean molecule-only throughput curve.
This script probes the adsorption workflow itself: slab + adsorbate graphs,
active masks, neighbor hooks, and FIRE2 batching. It keeps the same Toolkit
API used by the notebook and builds all chemistry through helper functions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

PART1 = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PART1))

from helpers import (  # noqa: E402
    ADSORBATE_ORIENTATIONS,
    RelaxationBackendConfig,
    ase_to_atomic_data,
    build_alpha_alumina_0001_slab,
    build_config_grid,
    build_cu111_slab,
    build_tio2_110_slab,
    get_relaxation_backend,
)


HOST_BUILDERS = {
    "Cu(111)": lambda: build_cu111_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(3, 3, 1),
    ),
    "Al2O3(0001)": lambda: build_alpha_alumina_0001_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(2, 2, 1),
    ),
    "TiO2(110)": lambda: build_tio2_110_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(2, 2, 1),
    ),
}


PROFILES = {
    "metal_h2o": {
        "host": "Cu(111)",
        "adsorbate": "H2O",
        "sites": None,
        "orientations": None,
        "rotations": (0.0, 60.0, 120.0),
    },
    "oxide_h2o": {
        "host": "Al2O3(0001)",
        "adsorbate": "H2O",
        "sites": None,
        "orientations": None,
        "rotations": (0.0, 60.0, 120.0),
    },
    "tio2_h2o": {
        "host": "TiO2(110)",
        "adsorbate": "H2O",
        "sites": None,
        "orientations": None,
        "rotations": (0.0, 60.0, 120.0),
    },
    "oxide_live": {
        "host": "Al2O3(0001)",
        "adsorbate": "H2O",
        "sites": ["al-top", "o-top", "bridge", "hollow"],
        "orientations": ["O-down", "flat"],
        "rotations": (0.0,),
    },
    "mixed_h2o": {
        "host": None,
        "adsorbate": "H2O",
        "sites": None,
        "orientations": None,
        "rotations": (0.0, 60.0, 120.0),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="oxide_h2o")
    parser.add_argument("--batch-sizes", default="4,8,12,18,24,36")
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--fmax", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--checkpoint", default="medium-mpa-0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-at-memory-fraction", type=float, default=0.80)
    parser.add_argument("--warmup-batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TOOLKIT_COMPILE_MODEL", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Enable model compilation in MACEWrapper.from_checkpoint.",
    )
    parser.add_argument(
        "--enable-cueq",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TOOLKIT_ENABLE_CUEQ", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Enable cuEquivariance kernels in MACEWrapper.from_checkpoint.",
    )
    return parser.parse_args()


def _build_profile_configs(profile_name: str):
    profile = PROFILES[profile_name]
    if profile_name == "mixed_h2o":
        groups = []
        for host in ("Cu(111)", "Al2O3(0001)"):
            slab = HOST_BUILDERS[host]()
            groups.append(
                build_config_grid(
                    host_name=host,
                    slab=slab,
                    adsorbate_name="H2O",
                    sites_filter=profile["sites"],
                    orientations_filter=profile["orientations"],
                    rotations_deg=profile["rotations"],
                    heights_A=(2.2,),
                    frozen_fraction=0.5,
                )
            )
        configs = []
        for idx in range(max(len(group) for group in groups)):
            for group in groups:
                if idx < len(group):
                    configs.append(group[idx])
        return configs

    host = profile["host"]
    slab = HOST_BUILDERS[host]()
    return build_config_grid(
        host_name=host,
        slab=slab,
        adsorbate_name=profile["adsorbate"],
        sites_filter=profile["sites"],
        orientations_filter=profile["orientations"],
        rotations_deg=profile["rotations"],
        heights_A=(2.2,),
        frozen_fraction=0.5,
    )


def _payloads_for_batch(configs, batch_size: int):
    payloads = []
    for idx in range(batch_size):
        config = configs[idx % len(configs)]
        payload = ase_to_atomic_data(
            config.atoms,
            structure_id=f"{config.label}_bench{idx:04d}",
            active_mask=config.active_mask,
        )
        payloads.append(payload)
    return payloads


def _max_force(result) -> float:
    forces = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = output_path.with_suffix(".csv")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    memory_limit_gb = total_memory_gb * args.stop_at_memory_fraction

    print(f"device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"visible_memory_gb: {total_memory_gb:.2f}", flush=True)
    print(f"profile: {args.profile}", flush=True)
    print(
        f"compile_model: {args.compile_model}; enable_cueq: {args.enable_cueq}",
        flush=True,
    )

    configs = _build_profile_configs(args.profile)
    if not configs:
        raise ValueError(f"Profile {args.profile!r} produced no configurations.")

    natoms = [len(config.atoms) for config in configs]
    active_atoms = [sum(config.active_mask) for config in configs]
    print(
        f"config_pool: {len(configs)} configs; atoms {min(natoms)}-{max(natoms)}; "
        f"active {min(active_atoms)}-{max(active_atoms)}",
        flush=True,
    )

    cache_dir = output_path.parent / "_adsorption_batch_cache"
    backend_start = perf_counter()
    backend = get_relaxation_backend(
        RelaxationBackendConfig(
            name="toolkit",
            cache_dir=str(cache_dir),
            use_cached_responses=False,
            toolkit_checkpoint=args.checkpoint,
            toolkit_device="cuda",
            toolkit_dtype=args.dtype,
            toolkit_compile_model=args.compile_model,
            toolkit_enable_cueq=args.enable_cueq,
            toolkit_dt=args.dt,
            toolkit_n_steps=args.n_steps,
            toolkit_fmax=args.fmax,
            toolkit_require_d3bj=False,
            toolkit_d3bj=None,
        )
    )
    backend_init_time_s = perf_counter() - backend_start
    print(f"backend_init_time_s: {backend_init_time_s:.3f}", flush=True)

    if args.warmup_batch_size > 0:
        print(f"warmup: {args.warmup_batch_size} structures", flush=True)
        warmup_payloads = _payloads_for_batch(configs, args.warmup_batch_size)
        _ = backend.relax(
            warmup_payloads,
            label=f"bench_{args.profile}_warmup_n{args.warmup_batch_size}",
            cellopt=False,
        )
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile",
                "compile_model",
                "enable_cueq",
                "backend_init_time_s",
                "batch_size",
                "status",
                "wall_time_s",
                "structures_per_s",
                "optimizer_steps",
                "n_converged",
                "peak_memory_gb",
                "total_atoms",
                "atoms_per_structure_min",
                "atoms_per_structure_max",
                "active_atoms_per_structure_min",
                "active_atoms_per_structure_max",
                "max_force_eV_A_max",
                "error",
            ],
        )
        writer.writeheader()

        for batch_size in batch_sizes:
            print(f"batch {batch_size}: start", flush=True)
            samples: list[dict[str, object]] = []
            row: dict[str, object] | None = None
            for repeat in range(args.repeats):
                payloads = _payloads_for_batch(configs, batch_size)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = perf_counter()
                try:
                    reply = backend.relax(
                        payloads,
                        label=(
                            f"bench_{args.profile}_n{batch_size}_"
                            f"steps{args.n_steps}_rep{repeat}"
                        ),
                        cellopt=False,
                    )
                    torch.cuda.synchronize(device)
                    wall_time_s = perf_counter() - started
                    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    max_forces = [_max_force(result) for result in reply.atoms]
                    sample = {
                        "repeat": repeat,
                        "wall_time_s": wall_time_s,
                        "structures_per_s": batch_size / wall_time_s,
                        "optimizer_steps": max(int(result.optimizer_nsteps) for result in reply.atoms),
                        "n_converged": sum(bool(result.converged) for result in reply.atoms),
                        "peak_memory_gb": peak_memory_gb,
                        "max_force_eV_A_max": max(max_forces),
                    }
                    samples.append(sample)
                    print(json.dumps({"batch_size": batch_size, **sample}), flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = {
                        "profile": args.profile,
                        "compile_model": args.compile_model,
                        "enable_cueq": args.enable_cueq,
                        "backend_init_time_s": backend_init_time_s,
                        "batch_size": batch_size,
                        "status": "oom",
                        "wall_time_s": "",
                        "structures_per_s": "",
                        "optimizer_steps": "",
                        "n_converged": "",
                        "peak_memory_gb": "",
                        "total_atoms": "",
                        "atoms_per_structure_min": "",
                        "atoms_per_structure_max": "",
                        "active_atoms_per_structure_min": "",
                        "active_atoms_per_structure_max": "",
                        "max_force_eV_A_max": "",
                        "error": str(exc),
                        "samples": samples,
                    }
                    break

            if row is None:
                row = {
                    "profile": args.profile,
                    "compile_model": args.compile_model,
                    "enable_cueq": args.enable_cueq,
                    "backend_init_time_s": backend_init_time_s,
                    "batch_size": batch_size,
                    "status": "ok",
                    "wall_time_s": median(float(sample["wall_time_s"]) for sample in samples),
                    "structures_per_s": median(float(sample["structures_per_s"]) for sample in samples),
                    "optimizer_steps": median(int(sample["optimizer_steps"]) for sample in samples),
                    "n_converged": median(int(sample["n_converged"]) for sample in samples),
                    "peak_memory_gb": median(float(sample["peak_memory_gb"]) for sample in samples),
                    "total_atoms": sum(len(configs[i % len(configs)].atoms) for i in range(batch_size)),
                    "atoms_per_structure_min": min(len(configs[i % len(configs)].atoms) for i in range(batch_size)),
                    "atoms_per_structure_max": max(len(configs[i % len(configs)].atoms) for i in range(batch_size)),
                    "active_atoms_per_structure_min": min(sum(configs[i % len(configs)].active_mask) for i in range(batch_size)),
                    "active_atoms_per_structure_max": max(sum(configs[i % len(configs)].active_mask) for i in range(batch_size)),
                    "max_force_eV_A_max": max(float(sample["max_force_eV_A_max"]) for sample in samples),
                    "error": "",
                    "samples": samples,
                }
            rows.append(row)
            writer.writerow({key: value for key, value in row.items() if key != "samples"})
            handle.flush()
            output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                "summary "
                + json.dumps({key: value for key, value in row.items() if key != "samples"}),
                flush=True,
            )

            if row["status"] == "oom":
                return 2
            if float(row["peak_memory_gb"]) >= memory_limit_gb:
                print(
                    f"stopping: peak memory {row['peak_memory_gb']:.2f} GB reached "
                    f"{args.stop_at_memory_fraction:.0%} of visible memory",
                    flush=True,
                )
                break
            gc.collect()
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
