# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark H2O batch throughput for the Toolkit hello-world cell.

This is intentionally separate from the adsorption benchmark. It measures the
small-molecule batching curve used to choose a pedagogical max batch size for
the notebook's H2O demo.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule as ase_molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics.hooks import NaNDetectorHook
from nvalchemi.dynamics.optimizers import FIRE2
from nvalchemi.models.mace import MACEWrapper


def gas_phase_h2o_atoms(vacuum: float = 7.5) -> Atoms:
    """Build one isolated H2O molecule in a nonperiodic vacuum box."""
    h2o = ase_molecule("H2O")
    h2o.center(vacuum=vacuum)
    return h2o


def to_atomic_data(atoms: Atoms, dtype: torch.dtype) -> AtomicData:
    """Convert ASE atoms into Toolkit data with initialized outputs."""
    data = AtomicData.from_atoms(atoms, device="cpu", dtype=dtype)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, dtype=dtype)
    return data


def run_batch(
    batch_size: int,
    model: MACEWrapper,
    device: torch.device,
    dtype: torch.dtype,
    *,
    n_steps: int,
    dt: float,
    fmax: float,
) -> dict[str, object]:
    """Relax one H2O batch and return timing, memory, and sample-energy stats."""
    atoms_list = [gas_phase_h2o_atoms() for _ in range(batch_size)]
    atomic_data = [to_atomic_data(atoms, dtype) for atoms in atoms_list]
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

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = perf_counter()
    relaxed_batch = optimizer.run(batch)
    torch.cuda.synchronize(device)
    wall_time_s = perf_counter() - start
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    energies = []
    for idx in range(min(batch_size, 8)):
        data = relaxed_batch.get_data(idx)
        energies.append(float(data.energy.detach().cpu().reshape(-1)[0]))
    step_count = getattr(optimizer, "step_count", None)
    if step_count is None:
        raise AttributeError("FIRE2 optimizer did not expose step_count; benchmark cannot report optimizer steps.")
    return {
        "batch_size": batch_size,
        "wall_time_s": wall_time_s,
        "structures_per_s": batch_size / wall_time_s,
        "optimizer_steps": int(step_count),
        "peak_memory_gb": peak_memory_gb,
        "sample_energy_mean_eV": float(np.mean(energies)) if energies else math.nan,
        "sample_energy_std_eV": float(np.std(energies)) if energies else math.nan,
    }


def parse_args() -> argparse.Namespace:
    default_output = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "h2o_saturation.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated batch sizes to test.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--checkpoint", default="medium-mpa-0")
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


def main() -> int:
    args = parse_args()
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float32

    print(f"device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"batch_sizes: {batch_sizes}", flush=True)
    print(f"repeats: {args.repeats}; n_steps: {args.n_steps}", flush=True)
    print(
        f"compile_model: {args.compile_model}; enable_cueq: {args.enable_cueq}",
        flush=True,
    )
    model_start = perf_counter()
    model = MACEWrapper.from_checkpoint(
        args.checkpoint,
        device=device,
        dtype=dtype,
        enable_cueq=args.enable_cueq,
        compile_model=args.compile_model,
    )
    model_load_time_s = perf_counter() - model_start
    print(f"model_load_time_s: {model_load_time_s:.3f}", flush=True)
    model.model_config.active_outputs = {"energy", "forces"}

    print("warmup: 16 structures x 2", flush=True)
    for _ in range(2):
        run_batch(
            16,
            model,
            device,
            dtype,
            n_steps=args.n_steps,
            dt=args.dt,
            fmax=args.fmax,
        )

    rows = []
    for batch_size in batch_sizes:
        samples = []
        print(f"batch {batch_size}: start", flush=True)
        for repeat in range(args.repeats):
            try:
                sample = run_batch(
                    batch_size,
                    model,
                    device,
                    dtype,
                    n_steps=args.n_steps,
                    dt=args.dt,
                    fmax=args.fmax,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": str(exc),
                }
                rows.append(row)
                output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)
                return 2
            samples.append(sample)
            print(json.dumps({"repeat": repeat, **sample}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()

        row = {
            "batch_size": batch_size,
            "status": "ok",
            "compile_model": args.compile_model,
            "enable_cueq": args.enable_cueq,
            "model_load_time_s": model_load_time_s,
            "median_wall_time_s": median(sample["wall_time_s"] for sample in samples),
            "median_structures_per_s": median(
                sample["structures_per_s"] for sample in samples
            ),
            "median_peak_memory_gb": median(
                sample["peak_memory_gb"] for sample in samples
            ),
            "median_optimizer_steps": median(
                sample["optimizer_steps"] for sample in samples
            ),
            "samples": samples,
        }
        rows.append(row)
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(
            "summary "
            + json.dumps({k: v for k, v in row.items() if k != "samples"}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
