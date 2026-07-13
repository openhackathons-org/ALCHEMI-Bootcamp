#!/usr/bin/env python3
"""Validate and checksum one executed Part 1 IR notebook run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

import nbformat
import numpy as np
import pandas as pd


RUN_MANIFEST_NAME = "water_run_manifest.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5"
)
EXPECTED_D3_PARAMETER_SHA256 = (
    "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
)
EXPECTED_TOOLKIT_CORE_COMMIT = "b770ee6963fd2f6137891e408c370012751918e2"
EXPECTED_TOOLKIT_OPS_COMMIT = "c6fbe652315e0cebd4f57a6a25f626258f0dbbfd"
EXPECTED_AIMNET_VERSION = "0.2.0"
EXPECTED_TORCH_VERSION_PREFIX = "2.12.0"

REQUIRED_FILES = (
    RUN_MANIFEST_NAME,
    "water_batch_cpu_gpu_crossover.csv",
    "water_batch_first_warm_calls.csv",
    "water_batch_layouts.csv",
    "water_dimer_ablation.csv",
    "water_dimer_ablation.png",
    "water_dimer_ablation_mae.csv",
    "water_ir_d6_topology_timeline.csv",
    "water_ir_diagnostics.csv",
    "water_ir_dynamics_log.csv",
    "water_ir_h6_topology_timeline.csv",
    "water_ir_trajectory.npz",
    "water_ir_metrics.csv",
    "water_ir_topology.csv",
    "water_ir_comparisons.csv",
    "water_ir_dft_comparison.csv",
    "water_ir_h_to_d_mode_map.csv",
    "water_ir_spectra.csv",
    "water_ir_dft_mapping.png",
    "water_ir_relaxed_start.extxyz",
    "water_ir_topology_timeline.png",
    "water_hexamer_seed.extxyz",
    "water_hexamer_relaxed.extxyz",
    "water_hexamer_trajectory_stride100.extxyz",
)

REQUIRED_DIRECTORIES = ("water_ir_relaxed.zarr",)
REQUIRED_OUTPUTS = REQUIRED_FILES + REQUIRED_DIRECTORIES
BUNDLE_SOURCE_FILES = (
    "alchemi-water-ir-source.ipynb",
    "run_notebook_no_timeout.py",
)

SOURCE_PATHS = (
    "build/Dockerfile",
    "build/overrides.txt",
    "build/requirements.txt",
    "scripts/rebuild_part1_ir_notebook.py",
    "scripts/run_notebook_no_timeout.py",
    "scripts/review_part1_ir_executed_notebook.py",
    "scripts/slurm_part1_dimer_reference.sbatch",
    "scripts/slurm_part1_remaster_h100.sbatch",
    "scripts/validate_part1_ir_run.py",
    "part-1-water-hydrogen-bonding-toolkit/alchemi-water-ir.ipynb",
    "part-1-water-hydrogen-bonding-toolkit/aux/__init__.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/README.md",
    "part-1-water-hydrogen-bonding-toolkit/aux/analysis.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/artifacts.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/benchmarking.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/capture.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/checkpoint.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/diagnostics.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/electrostatics.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/hooks.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/plotting.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/reference/__init__.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/reference/core.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/reference_data.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/runtime.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/spectra.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/structures.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/topology.py",
    "part-1-water-hydrogen-bonding-toolkit/aux/ui.py",
    "part-1-water-hydrogen-bonding-toolkit/reference/b97_3c_ir.py",
    "part-1-water-hydrogen-bonding-toolkit/reference/environment.yml",
    "part-1-water-hydrogen-bonding-toolkit/reference/plot_artifacts.py",
    "part-1-water-hydrogen-bonding-toolkit/reference/water_dimer_b97_3c.py",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/SHA256SUMS",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/h2o/manifest.json",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/h2o/ir_arrays.npz",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/d2o/manifest.json",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/d2o/ir_arrays.npz",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/h6/manifest.json",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/h6/ir_arrays.npz",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/d6/manifest.json",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/d6/ir_arrays.npz",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/water_dimer_b97_3c/SHA256SUMS",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/water_dimer_b97_3c/interaction_curve.csv",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/water_dimer_b97_3c/manifest.json",
    "part-1-water-hydrogen-bonding-toolkit/reference/artifacts/water_dimer_b97_3c/structures.extxyz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_run_manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid run manifest: {path}") from exc
    manifest = require_mapping(manifest, "run manifest")
    if manifest.get("schema") != "alchemi.water-ir-run.v1":
        raise ValueError(f"unexpected run manifest schema: {manifest.get('schema')!r}")
    require_mapping(manifest.get("provenance"), "run manifest provenance")
    require_mapping(manifest.get("settings"), "run manifest settings")
    require_mapping(manifest.get("gates"), "run manifest gates")
    if not isinstance(manifest.get("files"), list):
        raise ValueError("run manifest files must be a JSON array")
    return manifest


def validate_manifest_inventory(
    output_dir: Path, manifest: Mapping[str, object]
) -> dict[str, int]:
    """Verify the manifest against every regular output file recursively."""

    manifest_path = output_dir / RUN_MANIFEST_NAME
    actual_paths = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    actual = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in actual_paths
    }

    declared: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(manifest["files"]):
        record = require_mapping(raw_record, f"run manifest files[{index}]")
        relative_name = record.get("path")
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError(f"run manifest files[{index}].path must be nonempty text")
        relative_path = Path(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
        ):
            raise ValueError(f"non-canonical run manifest path: {relative_name!r}")
        if relative_name == RUN_MANIFEST_NAME:
            raise ValueError("run manifest must not inventory itself")
        if relative_name in declared:
            raise ValueError(f"duplicate run manifest file: {relative_name}")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ValueError(f"invalid byte size for {relative_name}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 for {relative_name}")
        declared[relative_name] = {"bytes": byte_count, "sha256": digest}

    missing = sorted(set(actual).difference(declared))
    unexpected = sorted(set(declared).difference(actual))
    if missing or unexpected:
        raise RuntimeError(
            "run manifest inventory differs from output directory: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative_name, observed in actual.items():
        record = declared[relative_name]
        if record["bytes"] != observed["bytes"]:
            raise RuntimeError(f"run manifest byte size mismatch: {relative_name}")
        if record["sha256"] != observed["sha256"]:
            raise RuntimeError(f"run manifest SHA-256 mismatch: {relative_name}")
    return {
        "file_count": len(actual),
        "total_bytes": sum(int(record["bytes"]) for record in actual.values()),
    }


def numeric_setting(
    settings: Mapping[str, object],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = settings.get(name)
    if isinstance(value, bool):
        raise ValueError(f"run manifest setting {name!r} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run manifest setting {name!r} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"run manifest setting {name!r} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"run manifest setting {name!r} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"run manifest setting {name!r} must be <= {maximum}")
    return number


def integer_setting(
    settings: Mapping[str, object], name: str, *, minimum: int = 1
) -> int:
    value = settings.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"run manifest setting {name!r} must be an integer >= {minimum}"
        )
    return value


def positive_numeric_setting(settings: Mapping[str, object], name: str) -> float:
    value = numeric_setting(settings, name, minimum=0.0)
    if value == 0.0:
        raise ValueError(f"run manifest setting {name!r} must be positive")
    return value


def validate_run_provenance(
    run_manifest: Mapping[str, object], source_notebook: Path
) -> dict[str, object]:
    provenance = require_mapping(
        run_manifest["provenance"], "run manifest provenance"
    )
    expected_exact = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "d3_parameter_file_sha256": EXPECTED_D3_PARAMETER_SHA256,
        "toolkit_core_commit": EXPECTED_TOOLKIT_CORE_COMMIT,
        "toolkit_ops_commit": EXPECTED_TOOLKIT_OPS_COMMIT,
        "aimnet": EXPECTED_AIMNET_VERSION,
    }
    for name, expected in expected_exact.items():
        if provenance.get(name) != expected:
            raise RuntimeError(
                f"run manifest provenance {name!r} does not match the pinned value"
            )
    if provenance.get("checkpoint_override") is not False:
        raise RuntimeError(
            "run manifest must use the pinned checkpoint, not an override"
        )

    torch_version = provenance.get("torch")
    if not isinstance(torch_version, str) or not torch_version.startswith(
        EXPECTED_TORCH_VERSION_PREFIX
    ):
        raise RuntimeError(
            "run manifest Torch version does not match the pinned 2.12.0 series"
        )

    live_notebook_sha256 = sha256_file(source_notebook)
    if provenance.get("notebook_sha256") != live_notebook_sha256:
        raise RuntimeError(
            "run manifest notebook SHA-256 does not match the live source notebook"
        )
    return {
        **expected_exact,
        "checkpoint_override": False,
        "torch": torch_version,
        "notebook_sha256": live_notebook_sha256,
    }


def validate_fused_stage_route(
    run_manifest: Mapping[str, object], *, warmup_steps: int, production_steps: int
) -> dict[str, int]:
    gates = require_mapping(run_manifest["gates"], "run manifest gates")
    route = require_mapping(
        gates.get("fused_stage_route_counts"),
        "run manifest gates.fused_stage_route_counts",
    )
    expected = {
        "status_0_warmup_steps": warmup_steps,
        "status_1_production_steps": production_steps,
    }
    for name, value in route.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"fused stage route count {name!r} must be an integer")
    if dict(route) != expected:
        raise RuntimeError(
            "fused stage route counts do not exactly match the declared workloads"
        )
    return expected


def validate_composition_gates(
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Reapply every saved model-composition acceptance threshold."""

    gates = require_mapping(run_manifest["gates"], "run manifest gates")
    scalar_limits = {
        "residual_serial_batch_max_abs_eV": 1e-5,
        "full_serial_batch_max_abs_eV": 2e-5,
        "component_closure_max_abs_eV": 2e-5,
    }
    scalar_results = {}
    for name, limit in scalar_limits.items():
        value = numeric_setting(gates, name, minimum=0.0)
        if value >= limit:
            raise RuntimeError(f"composition gate failed: {name} >= {limit}")
        scalar_results[name] = value

    nested_limits = {
        "official_calculator_parity": {
            "energy_eV": 3e-6,
            "forces_eV_A": 2e-6,
            "charges_e": 1e-7,
        },
        "analytic_coulomb": {
            "energy_eV": 2e-6,
            "forces_eV_A": 2e-6,
        },
        "compiled_ir_eager_parity": {
            "energy": 5e-6,
            "forces": 5e-6,
            "charges": 2e-7,
        },
        "compiled_ir_repeat_parity": {
            "energy": 2e-6,
            "forces": 2e-6,
            "charges": 1e-7,
        },
    }
    nested_results = {}
    for gate_name, limits in nested_limits.items():
        values = require_mapping(
            gates.get(gate_name), f"run manifest gates.{gate_name}"
        )
        if set(values) != set(limits):
            raise RuntimeError(
                f"composition gate {gate_name!r} has unexpected fields"
            )
        checked = {}
        for name, limit in limits.items():
            value = numeric_setting(values, name, minimum=0.0)
            if value >= limit:
                raise RuntimeError(
                    f"composition gate failed: {gate_name}.{name} >= {limit}"
                )
            checked[name] = value
        nested_results[gate_name] = checked

    reference_force = numeric_setting(
        gates, "finite_difference_force_reference_eV_A"
    )
    pipeline_force = numeric_setting(
        gates, "finite_difference_force_pipeline_eV_A"
    )
    force_error = numeric_setting(
        gates, "finite_difference_force_abs_error_eV_A", minimum=0.0
    )
    reproduced_error = abs(reference_force - pipeline_force)
    if not np.isclose(force_error, reproduced_error, rtol=1e-12, atol=1e-12):
        raise RuntimeError("finite-difference force error was not reproduced")
    force_tolerance = 2e-3 + 2e-2 * abs(reference_force)
    if force_error >= force_tolerance:
        raise RuntimeError("finite-difference force gate failed")

    return {
        **scalar_results,
        **nested_results,
        "finite_difference_force": {
            "reference_eV_A": reference_force,
            "pipeline_eV_A": pipeline_force,
            "abs_error_eV_A": force_error,
            "abs_tolerance_eV_A": force_tolerance,
        },
    }


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"cannot parse boolean values in {series.name!r}")
    return normalized == "true"


def distribution_record(name: str) -> dict[str, object]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None, "direct_url": None}
    direct_url = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "direct_url": json.loads(direct_url) if direct_url else None,
    }


def runtime_provenance() -> dict[str, object]:
    record: dict[str, object] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: distribution_record(name)
            for name in ("nvalchemi-toolkit", "nvalchemi-toolkit-ops", "aimnet")
        },
    }
    try:
        import torch

        record["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        record["torch"] = None

    checkpoint = (
        Path.home() / ".cache" / "aimnet" / "aimnet2_2025_b973c_d3_0.pt"
    )
    record["aimnet_checkpoint"] = {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }
    return record


def comparable_cell(cell: object) -> dict[str, object]:
    return {
        "cell_type": cell.get("cell_type"),
        "id": cell.get("id"),
        "source": cell.get("source"),
        "attachments": cell.get("attachments", {}),
    }


def validate_notebook(executed_path: Path, source_path: Path) -> int:
    notebook = nbformat.read(executed_path, as_version=4)
    source_notebook = nbformat.read(source_path, as_version=4)
    if [comparable_cell(cell) for cell in notebook.cells] != [
        comparable_cell(cell) for cell in source_notebook.cells
    ]:
        raise RuntimeError(
            "executed notebook cells or attachments do not match the source notebook"
        )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"executed notebook contains {len(errors)} error outputs")
    if any(cell.get("execution_count") is None for cell in code_cells):
        raise RuntimeError("one or more code cells were not executed")
    return len(code_cells)


def validate_trajectory(
    path: Path, *, expected_frames: int, expected_dt_fs: float
) -> dict[str, list[int]]:
    expected = {
        "dipoles_e_angstrom": (expected_frames, 4, 3),
        "charge_sums_e": (expected_frames, 4),
        "kinetic_energies_eV": (expected_frames, 4),
        "total_energies_eV": (expected_frames, 4),
        "positions_angstrom": (expected_frames, 42, 3),
    }
    with np.load(path, allow_pickle=False) as arrays:
        observed = {name: tuple(arrays[name].shape) for name in expected}
        if observed != expected:
            raise RuntimeError(f"trajectory shape mismatch: {observed!r}")
        for name in expected:
            if not np.isfinite(arrays[name]).all():
                raise RuntimeError(f"non-finite trajectory array: {name}")
        observed_dt_fs = float(np.asarray(arrays["dt_fs"]).reshape(()))
        if observed_dt_fs != expected_dt_fs:
            raise RuntimeError(
                f"trajectory timestep {observed_dt_fs} does not match "
                f"run manifest setting {expected_dt_fs}"
            )
    return {name: list(shape) for name, shape in observed.items()}


def cluster_gate_from_trajectory(
    arrays: object,
    graph_index: int,
    *,
    oxygen_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> dict[str, object]:
    batch_ptr = arrays["batch_ptr"]
    start, stop = map(int, batch_ptr[graph_index : graph_index + 2])
    numbers = arrays["atomic_numbers"][start:stop]
    frames = arrays["positions_angstrom"][:, start:stop]
    oxygen = np.flatnonzero(numbers == 8)
    hydrogen = np.flatnonzero(numbers == 1)
    oxygen_frames = frames[:, oxygen]
    hydrogen_frames = frames[:, hydrogen]
    assignment = np.argmin(
        np.linalg.norm(
            frames[0, hydrogen, None] - frames[0, oxygen][None, :], axis=-1
        ),
        axis=1,
    )
    oh_distance = np.linalg.norm(
        hydrogen_frames - oxygen_frames[:, assignment], axis=-1
    )

    oxygen_distance = np.linalg.norm(
        oxygen_frames[:, :, None] - oxygen_frames[:, None, :], axis=-1
    )
    connected = oxygen_distance < oxygen_cutoff_angstrom
    reachability = connected.copy()
    for _ in range(len(oxygen)):
        reachability |= (
            np.matmul(
                reachability.astype(np.uint8), connected.astype(np.uint8)
            )
            > 0
        )
    component_count = np.ones(frames.shape[0], dtype=int)
    for node in range(1, len(oxygen)):
        component_count += ~reachability[:, node, :node].any(axis=1)
    max_oxygen_components = int(component_count.max())

    donor_oxygen = oxygen_frames[:, assignment]
    h_to_donor = donor_oxygen - hydrogen_frames
    h_to_acceptor = oxygen_frames[:, None] - hydrogen_frames[:, :, None]
    h_acceptor_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    cosine = np.sum(h_to_donor[:, :, None] * h_to_acceptor, axis=-1)
    cosine /= np.linalg.norm(h_to_donor, axis=-1)[:, :, None]
    cosine /= np.linalg.norm(h_to_acceptor, axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    oo_distance = np.linalg.norm(
        donor_oxygen[:, :, None] - oxygen_frames[:, None], axis=-1
    )
    is_donor = (
        np.arange(len(oxygen))[None, None, :] == assignment[None, :, None]
    )
    hbond = (
        (h_acceptor_distance <= h_acceptor_cutoff_angstrom)
        & (oo_distance <= oo_cutoff_angstrom)
        & (angle >= hbond_angle_cutoff_deg)
        & ~is_donor
    )
    adjacency = np.zeros(
        (frames.shape[0], len(oxygen), len(oxygen)), dtype=bool
    )
    for hydrogen_index, donor_index in enumerate(assignment):
        adjacency[:, donor_index] |= hbond[:, hydrogen_index]

    initial_cycle = np.zeros(frames.shape[0], dtype=bool)
    import itertools

    for tail in itertools.permutations(range(1, len(oxygen))):
        nodes = (0, *tail)
        sources = np.asarray(nodes)
        targets = np.asarray((*tail, 0))
        if adjacency[0, sources, targets].all():
            initial_cycle = adjacency[:, sources, targets].all(axis=1)
            break
    return {
        "max_OH_angstrom": float(oh_distance.max()),
        "max_oxygen_components": max_oxygen_components,
        "all_frames_connected": max_oxygen_components == 1,
        "initial_ring_fraction": float(initial_cycle.mean()),
        "all_frames_initial_ring": bool(initial_cycle.all()),
    }


def validate_scientific_gates(
    output_dir: Path,
    trajectory_path: Path,
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    recorded_gates = require_mapping(run_manifest["gates"], "run manifest gates")
    oxygen_cutoff_angstrom = positive_numeric_setting(
        settings, "oxygen_connectivity_cutoff_A"
    )
    covalent_oh_cutoff_angstrom = positive_numeric_setting(
        settings, "covalent_OH_cutoff_A"
    )
    h_acceptor_cutoff_angstrom = positive_numeric_setting(
        settings, "hbond_H_acceptor_cutoff_A"
    )
    oo_cutoff_angstrom = positive_numeric_setting(settings, "hbond_OO_cutoff_A")
    hbond_angle_cutoff_deg = numeric_setting(
        settings,
        "hbond_angle_cutoff_deg",
        minimum=0.0,
        maximum=180.0,
    )
    pair_temperature_relative_tolerance = numeric_setting(
        settings, "pair_temperature_relative_tolerance", minimum=0.0
    )
    energy_excursion_advisory = numeric_setting(
        settings, "energy_excursion_advisory_meV_atom", minimum=0.0
    )

    with np.load(trajectory_path, allow_pickle=False) as arrays:
        batch_ptr = arrays["batch_ptr"]
        atoms_per_graph = np.diff(batch_ptr)
        temperature = (
            2.0
            * arrays["kinetic_energies_eV"]
            / (3.0 * atoms_per_graph[None, :] * 8.617333262145e-5)
        )
        mean_temperature = temperature.mean(axis=0)
        recomputed_topology = {
            label: cluster_gate_from_trajectory(
                arrays,
                graph,
                oxygen_cutoff_angstrom=oxygen_cutoff_angstrom,
                h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
                oo_cutoff_angstrom=oo_cutoff_angstrom,
                hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
            )
            for label, graph in (("(H2O)6", 2), ("(D2O)6", 3))
        }
        charge_error = np.max(np.abs(arrays["charge_sums_e"]), axis=0)
        energy_delta = (
            arrays["total_energies_eV"] - arrays["total_energies_eV"][0]
        )
        energy_excursion = (
            1000.0 * np.max(np.abs(energy_delta), axis=0) / atoms_per_graph
        )

    topology = pd.read_csv(output_dir / "water_ir_topology.csv", index_col=0)
    csv_ring_gate = as_bool(topology["all_frames_initial_ring"])
    for label, recomputed in recomputed_topology.items():
        if not np.isclose(
            topology.loc[label, "max_OH_angstrom"],
            recomputed["max_OH_angstrom"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(f"topology CSV max O-H does not match {label}")
        if bool(csv_ring_gate.loc[label]) != bool(
            recomputed["all_frames_initial_ring"]
        ):
            raise RuntimeError(f"topology CSV ring gate does not match {label}")
        if int(topology.loc[label, "max_oxygen_components"]) != int(
            recomputed["max_oxygen_components"]
        ):
            raise RuntimeError(
                f"topology CSV oxygen components do not match {label}"
            )

    intact = bool(
        all(item["all_frames_connected"] for item in recomputed_topology.values())
        and all(
            item["max_OH_angstrom"] < covalent_oh_cutoff_angstrom
            for item in recomputed_topology.values()
        )
    )
    if not intact:
        raise RuntimeError("saved trajectory failed cluster-integrity gates")
    ring_gate = bool(
        all(
            item["all_frames_initial_ring"]
            for item in recomputed_topology.values()
        )
    )
    energy_within_advisory = bool(
        np.max(energy_excursion) <= energy_excursion_advisory
    )

    comparisons = pd.read_csv(
        output_dir / "water_ir_comparisons.csv", index_col="comparison"
    )
    required_columns = {
        "value",
        "reported",
        "thermal_gate_passed",
        "topology_gate_passed",
        "status",
    }
    if not required_columns <= set(comparisons.columns):
        raise RuntimeError("comparison table is missing gate columns")
    expected_rows = {
        "H2O_over_D2O_centroid",
        "H6_over_D6_centroid",
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
    }
    if set(comparisons.index) != expected_rows:
        raise RuntimeError("comparison table does not contain the four declared rows")

    relative_temperature_difference = {
        "H2O_over_D2O_centroid": abs(mean_temperature[0] - mean_temperature[1])
        / (0.5 * (mean_temperature[0] + mean_temperature[1])),
        "H6_over_D6_centroid": abs(mean_temperature[2] - mean_temperature[3])
        / (0.5 * (mean_temperature[2] + mean_temperature[3])),
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": abs(
            mean_temperature[0] - mean_temperature[2]
        )
        / (0.5 * (mean_temperature[0] + mean_temperature[2])),
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": abs(
            mean_temperature[1] - mean_temperature[3]
        )
        / (0.5 * (mean_temperature[1] + mean_temperature[3])),
    }
    expected_thermal = {
        name: difference <= pair_temperature_relative_tolerance
        for name, difference in relative_temperature_difference.items()
    }
    expected_topology = {
        "H2O_over_D2O_centroid": True,
        "H6_over_D6_centroid": ring_gate,
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": ring_gate,
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": ring_gate,
    }
    reported = as_bool(comparisons["reported"])
    thermal = as_bool(comparisons["thermal_gate_passed"])
    topology_valid = as_bool(comparisons["topology_gate_passed"])
    metrics = pd.read_csv(output_dir / "water_ir_metrics.csv", index_col="system")
    candidate_values = {
        "H2O_over_D2O_centroid": (
            metrics.loc["H2O", "OH_OD_region_centroid_cm-1"]
            / metrics.loc["D2O", "OH_OD_region_centroid_cm-1"]
        ),
        "H6_over_D6_centroid": (
            metrics.loc["(H2O)6", "OH_OD_region_centroid_cm-1"]
            / metrics.loc["(D2O)6", "OH_OD_region_centroid_cm-1"]
        ),
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": (
            metrics.loc["(H2O)6", "OH_OD_region_centroid_cm-1"]
            - metrics.loc["H2O", "OH_OD_region_centroid_cm-1"]
        ),
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": (
            metrics.loc["(D2O)6", "OH_OD_region_centroid_cm-1"]
            - metrics.loc["D2O", "OH_OD_region_centroid_cm-1"]
        ),
    }
    for name in expected_rows:
        expected_reported = bool(expected_thermal[name] and expected_topology[name])
        if bool(thermal.loc[name]) != bool(expected_thermal[name]):
            raise RuntimeError(f"thermal gate was not independently reproduced: {name}")
        if bool(topology_valid.loc[name]) != bool(expected_topology[name]):
            raise RuntimeError(
                f"topology gate was not independently reproduced: {name}"
            )
        if bool(reported.loc[name]) != expected_reported:
            raise RuntimeError(f"reported state does not match gates: {name}")
        value = comparisons.loc[name, "value"]
        if expected_reported and not np.isclose(value, candidate_values[name]):
            raise RuntimeError(
                f"reported comparison value does not match metrics: {name}"
            )
        if not expected_reported and not pd.isna(value):
            raise RuntimeError(f"withheld comparison retained a numeric value: {name}")

    labels = ("H2O", "D2O", "(H2O)6", "(D2O)6")
    diagnostics = pd.read_csv(
        output_dir / "water_ir_diagnostics.csv", index_col="system"
    )
    if set(diagnostics.index) != set(labels):
        raise RuntimeError("diagnostics table does not contain the four systems")
    diagnostic_expectations = {
        "NVE_start_T_3N_K": temperature[0],
        "NVE_mean_T_3N_K": mean_temperature,
        "max_charge_error_e": charge_error,
        "max_energy_excursion_meV_atom": energy_excursion,
    }
    for column, expected_values in diagnostic_expectations.items():
        if column not in diagnostics.columns:
            raise RuntimeError(f"diagnostics table is missing {column}")
        observed_values = diagnostics.loc[list(labels), column].to_numpy(dtype=float)
        if not np.allclose(
            observed_values,
            np.asarray(expected_values, dtype=float),
            rtol=1e-12,
            atol=1e-10,
        ):
            raise RuntimeError(f"diagnostics table does not reproduce {column}")

    independently_recomputed_gates = {
        "cluster_integrity_passed": intact,
        "initial_ring_persisted_all_frames": ring_gate,
        "energy_excursion_within_advisory": energy_within_advisory,
    }
    for name, expected_value in independently_recomputed_gates.items():
        recorded_value = recorded_gates.get(name)
        if not isinstance(recorded_value, bool):
            raise ValueError(f"run manifest gate {name!r} must be boolean")
        if recorded_value != expected_value:
            raise RuntimeError(f"run manifest gate was not reproduced: {name}")

    recorded_reported = require_mapping(
        recorded_gates.get("reported_comparisons"),
        "run manifest gates.reported_comparisons",
    )
    if set(recorded_reported) != expected_rows:
        raise RuntimeError(
            "run manifest reported comparisons do not contain the four declared rows"
        )
    for name in expected_rows:
        recorded_value = recorded_reported[name]
        if not isinstance(recorded_value, bool):
            raise ValueError(
                f"run manifest reported comparison {name!r} must be boolean"
            )
        if recorded_value != bool(reported.loc[name]):
            raise RuntimeError(
                f"run manifest reported comparison was not reproduced: {name}"
            )

    return {
        "cluster_integrity_passed": intact,
        "cyclic_dft_overlay_gate_passed": ring_gate,
        "energy_excursion_within_advisory": energy_within_advisory,
        "comparison_reporting_valid": True,
        "manifest_thresholds": {
            "oxygen_connectivity_cutoff_A": oxygen_cutoff_angstrom,
            "covalent_OH_cutoff_A": covalent_oh_cutoff_angstrom,
            "hbond_H_acceptor_cutoff_A": h_acceptor_cutoff_angstrom,
            "hbond_OO_cutoff_A": oo_cutoff_angstrom,
            "hbond_angle_cutoff_deg": hbond_angle_cutoff_deg,
            "pair_temperature_relative_tolerance": (
                pair_temperature_relative_tolerance
            ),
            "energy_excursion_advisory_meV_atom": energy_excursion_advisory,
        },
        "max_energy_excursion_meV_atom": {
            label: float(value)
            for label, value in zip(labels, energy_excursion, strict=True)
        },
        "temperature_3N_mean_K": {
            label: float(value)
            for label, value in zip(
                labels,
                mean_temperature,
                strict=True,
            )
        },
        "recomputed_cluster_topology": recomputed_topology,
        "comparisons": {
            index: {
                "reported": bool(reported.loc[index]),
                "thermal_gate_passed": bool(thermal.loc[index]),
                "topology_gate_passed": bool(topology_valid.loc[index]),
                "temperature_relative_difference": float(
                    relative_temperature_difference[index]
                ),
                "status": str(row["status"]),
            }
            for index, row in comparisons.iterrows()
        },
    }


def write_portable_checksums(
    *,
    base: Path,
    paths: list[Path],
    destination: Path,
) -> None:
    base = base.resolve()
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    lines = []
    for path in unique:
        relative = path.relative_to(base)
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executed-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--calculation-validation", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    executed = args.executed_notebook.resolve()
    output_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve()
    summary_path = args.summary.resolve()
    checksums_path = args.checksums.resolve()
    bundle_root = summary_path.parent
    missing_bundle_sources = [
        name for name in BUNDLE_SOURCE_FILES if not (bundle_root / name).is_file()
    ]
    if missing_bundle_sources:
        raise FileNotFoundError(
            f"missing bundled execution sources: {missing_bundle_sources}"
        )
    missing_files = [
        name for name in REQUIRED_FILES if not (output_dir / name).is_file()
    ]
    missing_directories = [
        name for name in REQUIRED_DIRECTORIES if not (output_dir / name).is_dir()
    ]
    if missing_files or missing_directories:
        raise FileNotFoundError(
            "missing notebook outputs: "
            f"files={missing_files}, directories={missing_directories}"
        )
    for name in REQUIRED_DIRECTORIES:
        if not any(path.is_file() for path in (output_dir / name).rglob("*")):
            raise RuntimeError(f"required output directory is empty: {name}")

    run_manifest_path = output_dir / RUN_MANIFEST_NAME
    run_manifest = load_run_manifest(run_manifest_path)
    manifest_inventory = validate_manifest_inventory(output_dir, run_manifest)
    run_settings = require_mapping(
        run_manifest["settings"], "run manifest settings"
    )
    warmup_steps = integer_setting(run_settings, "warmup_steps")
    production_steps = integer_setting(run_settings, "production_steps")
    dt_fs = positive_numeric_setting(run_settings, "dt_fs")
    if warmup_steps != 5_000 or production_steps != 50_000:
        raise RuntimeError(
            "Part 1 requires the complete 5,000-step warmup and "
            "50,000-step production workloads"
        )
    if dt_fs != 0.5:
        raise RuntimeError("Part 1 requires the declared 0.5 fs timestep")
    if run_settings.get("compile_mode") != (
        "default Torch compile on the fixed 42-atom IR batch"
    ):
        raise RuntimeError("Part 1 requires fixed-workload default compilation")
    fused_stage_route = validate_fused_stage_route(
        run_manifest,
        warmup_steps=warmup_steps,
        production_steps=production_steps,
    )

    source_hashes = {}
    for relative in SOURCE_PATHS:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source file: {relative}")
        source_hashes[relative] = sha256_file(path)

    reference_artifacts = {}
    artifact_root = (
        source_root
        / "part-1-water-hydrogen-bonding-toolkit"
        / "reference"
        / "artifacts"
    )
    for label in ("h2o", "d2o", "h6", "d6"):
        reference_manifest_path = artifact_root / label / "manifest.json"
        arrays_path = artifact_root / label / "ir_arrays.npz"
        manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        reference_artifacts[label] = {
            "artifact_id": manifest["artifact_id"],
            "manifest_sha256": sha256_file(reference_manifest_path),
            "arrays_sha256": sha256_file(arrays_path),
        }

    source_notebook = (
        source_root
        / "part-1-water-hydrogen-bonding-toolkit"
        / "alchemi-water-ir.ipynb"
    )
    provenance_contract = validate_run_provenance(run_manifest, source_notebook)
    trajectory_path = output_dir / "water_ir_trajectory.npz"
    review_metadata = nbformat.read(executed, as_version=4).metadata.get(
        "alchemi_review"
    )
    validation_runtime = runtime_provenance()
    calculation_runtime = validation_runtime
    if args.calculation_validation is not None:
        prior = json.loads(
            args.calculation_validation.resolve().read_text(encoding="utf-8")
        )
        calculation_runtime = prior["runtime"]

    summary = {
        "code_cells_executed": validate_notebook(executed, source_notebook),
        "error_outputs": 0,
        "trajectory_shapes": validate_trajectory(
            trajectory_path,
            expected_frames=production_steps,
            expected_dt_fs=dt_fs,
        ),
        "scientific_gates": validate_scientific_gates(
            output_dir, trajectory_path, run_manifest
        ),
        "run_manifest": {
            "schema": run_manifest["schema"],
            "sha256": sha256_file(run_manifest_path),
            "inventory": manifest_inventory,
            "provenance_contract": provenance_contract,
            "fused_stage_route_counts": fused_stage_route,
            "composition_gates": validate_composition_gates(run_manifest),
        },
        "reference_artifacts": reference_artifacts,
        "runtime": calculation_runtime,
        "review_validation_runtime": (
            validation_runtime if args.calculation_validation is not None else None
        ),
        "notebook_review": review_metadata,
        "source_sha256": source_hashes,
        "executed_notebook_sha256": sha256_file(executed),
        "trajectory_sha256": sha256_file(trajectory_path),
        "required_outputs": list(REQUIRED_OUTPUTS),
        "required_output_types": {
            "files": list(REQUIRED_FILES),
            "directories": list(REQUIRED_DIRECTORIES),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    original_executed = executed.with_name(
        executed.stem + "-original" + executed.suffix
    )
    checksum_inputs = [
        executed,
        original_executed,
        summary_path,
        *(bundle_root / name for name in BUNDLE_SOURCE_FILES),
        *output_dir.rglob("*"),
    ]
    if args.calculation_validation is not None:
        checksum_inputs.append(args.calculation_validation.resolve())
        calculation_checksums = args.calculation_validation.with_name(
            "SHA256SUMS-calculation"
        ).resolve()
        checksum_inputs.append(calculation_checksums)
    write_portable_checksums(
        base=summary_path.parent,
        paths=checksum_inputs,
        destination=checksums_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"portable checksums: {checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
