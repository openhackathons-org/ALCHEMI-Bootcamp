#!/usr/bin/env python3
"""Run the 9-facet adsorption surface screen with native Toolkit."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART1))

# Keep Toolkit and GPU-kernel caches under a dedicated ignored runtime-cache tree.
RUNTIME_CACHE = PART1 / "outputs" / "runtime_cache"
os.environ.setdefault("WARP_CACHE_PATH", str(RUNTIME_CACHE / "warp"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))
os.environ.setdefault("HF_HOME", str(RUNTIME_CACHE / "hf"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("TRITON_CACHE_DIR", str(RUNTIME_CACHE / "triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(RUNTIME_CACHE / "torchinductor"),
)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_cuda_ld_path_for_cli() -> None:
    """Re-exec once so Toolkit/NVRTC can find CUDA wheel libraries by name."""
    cu13_lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
        / "lib"
    )
    if not cu13_lib.exists():
        return
    ld_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if str(cu13_lib) in ld_paths:
        return
    if os.environ.get("SURFACE_SCREEN_LD_REEXEC") == "1":
        print(
            "SURFACE_SCREEN_LD_REEXEC=1 is already set, but the CUDA wheel "
            f"library path is still missing from LD_LIBRARY_PATH: {cu13_lib}",
            file=sys.stderr,
        )
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([str(cu13_lib), *ld_paths])
    os.environ["SURFACE_SCREEN_LD_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_ensure_cuda_ld_path_for_cli()

from helpers import (  # noqa: E402
    OptimizationResult,
    RelaxationBackendConfig,
    SURFACE_SCREEN_GRID_VERSION,
    TUTORIAL_SURFACE_SCREEN_STEM,
    SurfaceScreenAdsorbateSpec,
    SurfaceScreenSlabSpec,
    ase_to_atomic_data,
    atomic_data_to_ase,
    audit_initial_configs,
    build_application_heatmap,
    build_co,
    build_cu100_slab,
    build_cu111_slab,
    build_cu110_slab,
    build_difficult_cases,
    build_h2o,
    build_methanol,
    build_nh3,
    build_pair_results_table,
    build_config_grid,
    build_step_statistics,
    build_tin_001_slab,
    build_tin_110_slab,
    build_tin_210_slab,
    build_tio2_100_slab,
    build_tio2_101_slab,
    build_tio2_110_slab,
    get_relaxation_backend,
    make_active_mask,
    run_toolkit_relaxation_with_trajectory,
    summarize_surface_screen_pairs,
    surface_screen_expected_counts,
    surface_screen_output_paths,
    write_surface_screen_audit_tables,
)


DEFAULT_OUTPUT_ROOT = (
    PART1
    / "outputs"
    / "precomputed"
    / "tutorial"
    / f"{TUTORIAL_SURFACE_SCREEN_STEM}_full"
    / "surface_screen"
)

# Headless execution mirrors the visible notebook cells. Keep these definitions
# explicit here so the script can refresh artifacts without importing a hidden
# panel definition from helpers.
SCREEN_SLABS: tuple[SurfaceScreenSlabSpec, ...] = (
    SurfaceScreenSlabSpec(
        "Cu(111)", "fcc metal", "close-packed terrace", (1, 1, 1),
        "build_cu111_slab",
        lambda: build_cu111_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
        (3, 3, 1),
        "close-packed low-index Cu surface",
    ),
    SurfaceScreenSlabSpec(
        "Cu(100)", "fcc metal", "square terrace", (1, 0, 0),
        "build_cu100_slab",
        lambda: build_cu100_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
        (3, 3, 1),
        "square low-index Cu surface",
    ),
    SurfaceScreenSlabSpec(
        "Cu(110)", "fcc metal", "open row surface", (1, 1, 0),
        "build_cu110_slab",
        lambda: build_cu110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(3, 3, 1)),
        (3, 3, 1),
        "more open low-index Cu surface",
    ),
    SurfaceScreenSlabSpec(
        "TiO2(110)", "oxide", "rutile bridge-row surface", (1, 1, 0),
        "build_tio2_110_slab",
        lambda: build_tio2_110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "canonical rutile surface, distinct from the Cu terraces",
    ),
    SurfaceScreenSlabSpec(
        "TiO2(100)", "oxide", "rutile side surface", (1, 0, 0),
        "build_tio2_100_slab",
        lambda: build_tio2_100_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "second low-index rutile cut",
    ),
    SurfaceScreenSlabSpec(
        "TiO2(101)", "oxide", "rutile oblique surface", (1, 0, 1),
        "build_tio2_101_slab",
        lambda: build_tio2_101_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "oblique rutile cut; useful contrast to (110) and (100)",
    ),
    SurfaceScreenSlabSpec(
        "TiN(001)", "nitride ceramic", "rocksalt nonpolar terrace", (0, 0, 1),
        "build_tin_001_slab",
        lambda: build_tin_001_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "nonpolar rocksalt TiN terrace",
    ),
    SurfaceScreenSlabSpec(
        "TiN(110)", "nitride ceramic", "rocksalt rectangular terrace", (1, 1, 0),
        "build_tin_110_slab",
        lambda: build_tin_110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "second nonpolar TiN low-index cut",
    ),
    SurfaceScreenSlabSpec(
        "TiN(210)", "nitride ceramic", "rocksalt stepped surface", (2, 1, 0),
        "build_tin_210_slab",
        lambda: build_tin_210_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
        (2, 2, 1),
        "stepped/high-index TiN cut; TiN(111) is avoided here because it is polar",
    ),
)

SCREEN_ADSORBATES: tuple[SurfaceScreenAdsorbateSpec, ...] = (
    SurfaceScreenAdsorbateSpec("CO", ("C-down", "O-down"), "CO binding and catalysis probe", "linear probe molecule with two obvious contact atoms"),
    SurfaceScreenAdsorbateSpec("H2O", ("O-down", "H-down"), "water adsorption and hydrophilicity", "closed-shell water adsorption example"),
    SurfaceScreenAdsorbateSpec("NH3", ("N-down", "H-down"), "ammonia binding and nitrogen chemistry", "closed-shell lone-pair donor"),
    SurfaceScreenAdsorbateSpec("CH3OH", ("O-down", "methyl-down"), "alcohol adsorption and separations proxy", "larger polar molecule that tests orientation sensitivity"),
)

SCREEN_SITES_BY_HOST: dict[str, tuple[str, ...]] = {
    "Cu(111)": ("top", "bridge", "fcc"),
    "Cu(100)": ("top", "bridge", "hollow"),
    "Cu(110)": ("top", "bridge", "hollow"),
    "TiO2(110)": ("ti-top", "o-top", "bridge"),
    "TiO2(100)": ("ti-top", "o-top", "bridge"),
    "TiO2(101)": ("ti-top", "o-top", "bridge"),
    "TiN(001)": ("ti-top", "n-top", "bridge"),
    "TiN(110)": ("ti-top", "n-top", "bridge"),
    "TiN(210)": ("ti-top", "n-top", "bridge"),
}
SCREEN_ROTATIONS_DEG = (0.0,)
SCREEN_START_HEIGHT_A = 2.5
SCREEN_FROZEN_SLAB_FRACTION = 0.5
GREEN_OPTIMIZER_STEP_MAX = 200
YELLOW_OPTIMIZER_STEP_MAX = 500
STEP_FORCE_THRESHOLD_EV_A = 0.05
EXCLUDE_RED_STEP_STATUS_FROM_PAIR_MINIMUM = True


def screen_expected_counts() -> dict[str, int]:
    return surface_screen_expected_counts(
        n_slabs=len(SCREEN_SLABS),
        n_adsorbates=len(SCREEN_ADSORBATES),
        starts_per_pair=6,
    )


def build_screen_slabs() -> dict[str, object]:
    return {spec.name: spec.builder() for spec in SCREEN_SLABS}


def build_screen_configs(
    slabs: dict[str, object],
    *,
    height_A: float = SCREEN_START_HEIGHT_A,
    frozen_fraction: float = SCREEN_FROZEN_SLAB_FRACTION,
) -> list[object]:
    configs: list[object] = []
    for slab_spec in SCREEN_SLABS:
        for adsorbate in SCREEN_ADSORBATES:
            pair_configs = build_config_grid(
                host_name=slab_spec.name,
                slab=slabs[slab_spec.name],
                adsorbate_name=adsorbate.name,
                sites_filter=list(SCREEN_SITES_BY_HOST[slab_spec.name]),
                orientations_filter=list(adsorbate.orientations),
                rotations_deg=SCREEN_ROTATIONS_DEG,
                heights_A=(height_A,),
                frozen_fraction=frozen_fraction,
            )
            if len(pair_configs) != 6:
                raise ValueError(
                    f"Expected 6 starts for {adsorbate.name}/{slab_spec.name}: "
                    "3 site classes x 2 orientation classes x 1 rotation x 1 height, "
                    f"got {len(pair_configs)}."
                )
            configs.extend(pair_configs)
    return configs


ADSORBATE_BUILDERS = {
    "CO": lambda: build_co("C-down"),
    "H2O": lambda: build_h2o("O-down"),
    "NH3": lambda: build_nh3("N-down"),
    "CH3OH": lambda: build_methanol("O-down"),
}
TRUE_VALUES = {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--n-steps", type=int, default=int(os.environ.get("TOOLKIT_N_STEPS", "500")))
    parser.add_argument("--fmax", type=float, default=float(os.environ.get("TOOLKIT_FMAX", "0.05")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ADSORPTION_BATCH_SIZE", "12")))
    parser.add_argument("--clean-batch-size", type=int, default=3)
    parser.add_argument("--gas-batch-size", type=int, default=4)
    parser.add_argument("--fallback-batch-size", type=int, default=6)
    parser.add_argument("--snapshot-frequency", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-trajectories", action="store_true")
    return parser.parse_args()


def _safe(name: str) -> str:
    return (
        name.replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _allow_overwrite() -> bool:
    return os.environ.get("ALCHEMI_ALLOW_ARTIFACT_OVERWRITE", "").strip().lower() in TRUE_VALUES


def _ensure_force_is_intentional(force: bool) -> None:
    if force and not _allow_overwrite():
        raise RuntimeError(
            "--force would refresh official surface-screen artifacts. Set "
            "ALCHEMI_ALLOW_ARTIFACT_OVERWRITE=1 for an intentional refresh."
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _ensure_precomputed_write_is_intentional(output_root: Path) -> None:
    precomputed_root = PART1 / "outputs" / "precomputed"
    if _is_relative_to(output_root, precomputed_root) and not _allow_overwrite():
        raise RuntimeError(
            "Refusing to write under outputs/precomputed without an explicit "
            "refresh guard. Use an outputs/live_runs/<run_id>/tutorial path, "
            "or set ALCHEMI_ALLOW_ARTIFACT_OVERWRITE=1 for an intentional "
            "official refresh."
        )


def _ensure_dirs(paths: dict[str, Path]) -> None:
    for key in (
        "tables",
        "chunks",
        "raw",
        "initial_structures",
        "relaxed_structures",
        "clean_structures",
        "gas_structures",
        "trajectories",
        "trajectory_logs",
        "reports",
        "figures",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    for subdir in ("adsorption", "clean", "gas"):
        (paths["trajectories"] / subdir).mkdir(parents=True, exist_ok=True)
        (paths["trajectory_logs"] / subdir).mkdir(parents=True, exist_ok=True)


def _build_backend(args: argparse.Namespace, paths: dict[str, Path]):
    device = os.environ.get("TOOLKIT_DEVICE", "cuda")
    if device == "auto":
        device = "cuda"
    return get_relaxation_backend(
        RelaxationBackendConfig(
            name="toolkit",
            cache_dir=str(paths["root"] / "toolkit_response_cache"),
            use_cached_responses=False,
            toolkit_checkpoint=os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
            toolkit_device=device,
            toolkit_dtype=os.environ.get("TOOLKIT_DTYPE", "float32"),
            toolkit_enable_cueq=_env_bool("TOOLKIT_ENABLE_CUEQ", False),
            toolkit_compile_model=_env_bool("TOOLKIT_COMPILE_MODEL", False),
            toolkit_head=os.environ.get("TOOLKIT_HEAD", "") or None,
            toolkit_dt=float(os.environ.get("TOOLKIT_DT", "0.01")),
            toolkit_n_steps=args.n_steps,
            toolkit_fmax=args.fmax,
            toolkit_require_d3bj=False,
            toolkit_d3bj=None,
        )
    )


def _gpu_before(backend) -> dict[str, float | None]:
    try:
        torch = backend.api.torch
        if not torch.cuda.is_available():
            return {}
        torch.cuda.reset_peak_memory_stats(backend.device)
        free, total = torch.cuda.mem_get_info(backend.device)
        return {
            "gpu_free_before_gb": free / 1024**3,
            "gpu_total_gb": total / 1024**3,
        }
    except Exception:
        return {}


def _gpu_after(backend) -> dict[str, float | None]:
    try:
        torch = backend.api.torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info(backend.device)
        return {
            "gpu_free_after_gb": free / 1024**3,
            "gpu_allocated_after_gb": torch.cuda.memory_allocated(backend.device) / 1024**3,
            "gpu_reserved_after_gb": torch.cuda.memory_reserved(backend.device) / 1024**3,
            "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(backend.device) / 1024**3,
            "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(backend.device) / 1024**3,
            "gpu_total_gb": total / 1024**3,
        }
    except Exception:
        return {}


def _max_force(result: OptimizationResult) -> float:
    forces = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def _write_initial_atoms(path: Path, atoms, *, label: str) -> None:
    if path.exists() and not _allow_overwrite():
        return
    initial = atoms.copy()
    initial.info["label"] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(path, initial, format="extxyz")


def _write_result_atoms(path: Path, result: OptimizationResult, *, label: str) -> None:
    if path.exists() and not _allow_overwrite():
        return
    atoms = atomic_data_to_ase(result)
    atoms.info["label"] = label
    atoms.info["energy_eV"] = float(result.energy)
    atoms.info["converged"] = bool(result.converged)
    atoms.info["optimizer_nsteps"] = int(result.optimizer_nsteps)
    atoms.arrays["forces"] = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(path, atoms, format="extxyz")


def _result_to_json(result: OptimizationResult) -> dict[str, object]:
    return result.model_dump(mode="json")


def _load_results(path: Path) -> list[OptimizationResult]:
    return [OptimizationResult.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _write_results(path: Path, results: list[OptimizationResult]) -> None:
    if path.exists() and not _allow_overwrite():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_result_to_json(result) for result in results], indent=2),
        encoding="utf-8",
    )


def _write_csv_guarded(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _allow_overwrite():
        return
    df.to_csv(path, index=False)


def _write_json_guarded(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _allow_overwrite():
        return
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_audit_tables_guarded(
    paths: dict[str, Path],
    slabs: dict[str, object],
    configs: list[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        paths["surface_fingerprints_csv"].exists()
        and paths["initial_geometry_audit_csv"].exists()
        and not _allow_overwrite()
    ):
        return (
            pd.read_csv(paths["surface_fingerprints_csv"]),
            pd.read_csv(paths["initial_geometry_audit_csv"]),
        )
    return write_surface_screen_audit_tables(
        paths["root"],
        slabs,
        configs,
        slab_specs=SCREEN_SLABS,
        expected_counts=screen_expected_counts(),
        validate=True,
    )


def _atoms_per_batch(items: list[dict[str, object]]) -> tuple[int, int]:
    total_atoms = sum(len(item["atoms"]) for item in items)
    active_atoms = 0
    for item in items:
        mask = item.get("active_mask")
        if mask is None:
            active_atoms += len(item["atoms"])
        else:
            active_atoms += int(sum(mask))
    return total_atoms, active_atoms


def _paths_for_item(paths: dict[str, Path], label: str, kind: str) -> dict[str, Path]:
    safe = _safe(label)
    if kind == "clean":
        return {
            "initial": paths["clean_structures"] / f"{safe}_initial.extxyz",
            "relaxed": paths["clean_structures"] / f"{safe}.extxyz",
            "trajectory": paths["trajectories"] / "clean" / f"{safe}.extxyz",
            "log": paths["trajectory_logs"] / "clean" / f"{safe}.csv",
        }
    if kind == "gas":
        return {
            "initial": paths["gas_structures"] / f"{safe}_initial.extxyz",
            "relaxed": paths["gas_structures"] / f"{safe}.extxyz",
            "trajectory": paths["trajectories"] / "gas" / f"{safe}.extxyz",
            "log": paths["trajectory_logs"] / "gas" / f"{safe}.csv",
        }
    return {
        "initial": paths["initial_structures"] / f"{safe}.extxyz",
        "relaxed": paths["relaxed_structures"] / f"{safe}.extxyz",
        "trajectory": paths["trajectories"] / "adsorption" / f"{safe}.extxyz",
        "log": paths["trajectory_logs"] / "adsorption" / f"{safe}.csv",
    }


def _cached_batch_ready(
    raw_path: Path,
    items: list[dict[str, object]],
    paths: dict[str, Path],
    *,
    write_trajectories: bool,
) -> bool:
    if not raw_path.exists():
        return False
    if not write_trajectories:
        return True
    for item in items:
        item_paths = _paths_for_item(paths, str(item["label"]), str(item["kind"]))
        if not item_paths["trajectory"].exists() or not item_paths["log"].exists():
            return False
    return True


def _run_relaxation_batch(
    backend,
    items: list[dict[str, object]],
    *,
    label: str,
    batch_type: str,
    paths: dict[str, Path],
    force: bool,
    write_trajectories: bool,
    snapshot_frequency: int,
) -> tuple[list[OptimizationResult], dict[str, object]]:
    raw_path = paths["raw"] / f"{label}.json"
    metadata_path = paths["chunks"] / f"{label}.metadata.json"
    if _cached_batch_ready(raw_path, items, paths, write_trajectories=write_trajectories) and not force:
        results = _load_results(raw_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        runtime_s = float(metadata.get("runtime_s", 0.0))
        loaded_from_cache = True
        print(f"{label}: loaded cached batch ({len(items)} structures)", flush=True)
    else:
        if raw_path.exists() and not force and not _allow_overwrite():
            raise RuntimeError(
                f"{label} has cached raw results but missing trajectories/logs. "
                "Use a new output root for exploratory recompute, or set "
                "ALCHEMI_ALLOW_ARTIFACT_OVERWRITE=1 with --force for an "
                "intentional official refresh."
            )
        data_list = [
            ase_to_atomic_data(
                item["atoms"],
                structure_id=str(item["label"]),
                active_mask=item.get("active_mask"),
            )
            for item in items
        ]
        trajectory_paths = [
            _paths_for_item(paths, str(item["label"]), str(item["kind"]))["trajectory"]
            for item in items
        ]
        log_paths = [
            _paths_for_item(paths, str(item["label"]), str(item["kind"]))["log"]
            for item in items
        ]
        memory_before = _gpu_before(backend)
        started = time.perf_counter()
        if write_trajectories:
            reply = run_toolkit_relaxation_with_trajectory(
                backend,
                data_list,
                label=label,
                trajectory_paths=trajectory_paths,
                log_paths=log_paths,
                cellopt=False,
                snapshot_frequency=snapshot_frequency,
            )
        else:
            reply = backend.relax(data_list, label=label, cellopt=False)
        runtime_s = time.perf_counter() - started
        loaded_from_cache = False
        results = reply.atoms
        _write_results(raw_path, results)
        memory_after = _gpu_after(backend)
        metadata = {**memory_before, **memory_after}

    for item, result in zip(items, results):
        item_paths = _paths_for_item(paths, str(item["label"]), str(item["kind"]))
        _write_initial_atoms(item_paths["initial"], item["atoms"], label=str(item["label"]))
        _write_result_atoms(item_paths["relaxed"], result, label=str(item["label"]))
        item_raw = paths["raw"] / f"{_safe(str(item['label']))}.json"
        if not item_raw.exists() or _allow_overwrite():
            item_raw.write_text(json.dumps(_result_to_json(result), indent=2), encoding="utf-8")

    total_atoms, active_atoms = _atoms_per_batch(items)
    batch_row = {
        "batch_label": label,
        "batch_type": batch_type,
        "n_structures": len(items),
        "n_atoms_total": total_atoms,
        "n_active_atoms": active_atoms,
        "runtime_s": runtime_s,
        "loaded_from_cache": loaded_from_cache,
        "structures_per_s": len(items) / runtime_s if runtime_s > 0 else np.nan,
        "atoms_per_s": total_atoms / runtime_s if runtime_s > 0 else np.nan,
        "active_atoms_per_s": active_atoms / runtime_s if runtime_s > 0 else np.nan,
        "optimizer_nsteps": max(int(result.optimizer_nsteps) for result in results),
        "n_converged": sum(bool(result.converged) for result in results),
        "raw_json": str(raw_path),
        "metadata_json": str(metadata_path),
        "labels": ";".join(str(item["label"]) for item in items),
        "pairs": ";".join(sorted({str(item.get("pair", "")) for item in items if item.get("pair")})),
        **metadata,
    }
    _write_json_guarded(batch_row, metadata_path)
    print(
        f"{label}: {len(items)} structures, {batch_row['n_converged']} converged, "
        f"{batch_row['optimizer_nsteps']} batch steps, {runtime_s:.1f}s",
        flush=True,
    )
    return results, batch_row


def _run_batch_with_fallback(
    backend,
    items: list[dict[str, object]],
    *,
    label: str,
    batch_type: str,
    paths: dict[str, Path],
    force: bool,
    write_trajectories: bool,
    snapshot_frequency: int,
    fallback_batch_size: int,
) -> tuple[list[OptimizationResult], list[dict[str, object]]]:
    try:
        results, row = _run_relaxation_batch(
            backend,
            items,
            label=label,
            batch_type=batch_type,
            paths=paths,
            force=force,
            write_trajectories=write_trajectories,
            snapshot_frequency=snapshot_frequency,
        )
        return results, [row]
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()
        if not is_oom or len(items) <= fallback_batch_size:
            raise
        print(
            f"{label}: OOM at {len(items)} structures; retrying in batches of "
            f"{fallback_batch_size}",
            flush=True,
        )
        all_results: list[OptimizationResult] = []
        rows: list[dict[str, object]] = []
        for start in range(0, len(items), fallback_batch_size):
            sub_items = items[start : start + fallback_batch_size]
            sub_label = f"{label}_fallback_{start // fallback_batch_size + 1:02d}"
            sub_results, sub_rows = _run_batch_with_fallback(
                backend,
                sub_items,
                label=sub_label,
                batch_type=batch_type,
                paths=paths,
                force=force,
                write_trajectories=write_trajectories,
                snapshot_frequency=snapshot_frequency,
                fallback_batch_size=fallback_batch_size,
            )
            for row in sub_rows:
                row["fallback_from_batch_size"] = len(items)
            all_results.extend(sub_results)
            rows.extend(sub_rows)
        return all_results, rows


def _chunk_items(items: list[dict[str, object]], chunk_size: int) -> list[list[dict[str, object]]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def _pack_pair_groups(
    pair_groups: list[list[dict[str, object]]],
    *,
    batch_size: int,
) -> list[list[dict[str, object]]]:
    packed: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for group in pair_groups:
        if current and len(current) + len(group) > batch_size:
            packed.append(current)
            current = []
        current.extend(group)
    if current:
        packed.append(current)
    return packed


def _gas_atoms(name: str):
    atoms = ADSORBATE_BUILDERS[name]()
    atoms.set_cell([15.0, 15.0, 15.0])
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def _reference_items(slabs: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clean_items = [
        {
            "label": f"clean_{_safe(host)}",
            "kind": "clean",
            "host": host,
            "atoms": slab,
            "active_mask": make_active_mask(slab, bottom_fraction=0.5),
        }
        for host, slab in slabs.items()
    ]
    gas_items = [
        {
            "label": f"gas_{spec.name}",
            "kind": "gas",
            "adsorbate": spec.name,
            "atoms": _gas_atoms(spec.name),
            "active_mask": None,
        }
        for spec in SCREEN_ADSORBATES
    ]
    return clean_items, gas_items


def _clean_rows(
    items: list[dict[str, object]],
    results: list[OptimizationResult],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, OptimizationResult], dict[str, object]]:
    rows: list[dict[str, object]] = []
    results_by_host: dict[str, OptimizationResult] = {}
    atoms_by_host: dict[str, object] = {}
    for item, result in zip(items, results):
        host = str(item["host"])
        item_paths = _paths_for_item(paths, str(item["label"]), "clean")
        results_by_host[host] = result
        atoms_by_host[host] = atomic_data_to_ase(result)
        rows.append(
            {
                "host": host,
                "label": item["label"],
                "energy_eV": float(result.energy),
                "converged": bool(result.converged),
                "optimizer_nsteps": int(result.optimizer_nsteps),
                "max_force_eV_A": _max_force(result),
                "structure_path": str(item_paths["relaxed"]),
                "trajectory_path": str(item_paths["trajectory"]),
                "trajectory_log_path": str(item_paths["log"]),
            }
        )
    return pd.DataFrame(rows), results_by_host, atoms_by_host


def _gas_rows(
    items: list[dict[str, object]],
    results: list[OptimizationResult],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, OptimizationResult]]:
    rows: list[dict[str, object]] = []
    results_by_adsorbate: dict[str, OptimizationResult] = {}
    for item, result in zip(items, results):
        adsorbate = str(item["adsorbate"])
        item_paths = _paths_for_item(paths, str(item["label"]), "gas")
        results_by_adsorbate[adsorbate] = result
        rows.append(
            {
                "adsorbate": adsorbate,
                "label": item["label"],
                "energy_eV": float(result.energy),
                "converged": bool(result.converged),
                "optimizer_nsteps": int(result.optimizer_nsteps),
                "max_force_eV_A": _max_force(result),
                "structure_path": str(item_paths["relaxed"]),
                "trajectory_path": str(item_paths["trajectory"]),
                "trajectory_log_path": str(item_paths["log"]),
            }
        )
    return pd.DataFrame(rows), results_by_adsorbate


def _adsorption_items(configs) -> list[dict[str, object]]:
    return [
        {
            "label": config.label,
            "kind": "adsorption",
            "host": config.host,
            "adsorbate": config.adsorbate,
            "pair": f"{config.adsorbate}/{config.host}",
            "config": config,
            "atoms": config.atoms,
            "active_mask": config.active_mask,
        }
        for config in configs
    ]


def _write_report(
    *,
    paths: dict[str, Path],
    metadata: dict[str, object],
    pair_summary_df: pd.DataFrame,
    batch_summary_df: pd.DataFrame,
    difficult_df: pd.DataFrame,
) -> None:
    counts = screen_expected_counts()
    max_steps = int(pair_summary_df["max_optimizer_nsteps"].max()) if len(pair_summary_df) else 0
    mean_steps = float(pair_summary_df["median_optimizer_nsteps"].mean()) if len(pair_summary_df) else 0.0
    slow = pair_summary_df.sort_values("max_optimizer_nsteps", ascending=False).head(8)
    lines = [
        "# Surface-Screen Toolkit Run",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Grid version: `{SURFACE_SCREEN_GRID_VERSION}`",
        f"Model: `{metadata['toolkit_checkpoint']}` head `{metadata['toolkit_head']}`; D3(BJ) disabled.",
        "",
        "## Scope",
        "",
        (
            f"This run covers {counts['slabs']} slabs, {counts['adsorbates']} "
            f"adsorbates, {counts['adsorbate_surface_pairs']} adsorbate-surface "
            f"pairs, and {counts['adsorption_relaxations']} adsorption starting "
            "geometries. Clean-slab and gas-reference relaxations are included "
            f"for {counts['core_relaxations']} core relaxations."
        ),
        "",
        "The heatmap ranks the best converged molecular adsorption structure "
        "found within the six-start search for each pair. It is a first-pass "
        "screen, not a DFT or experimental claim.",
        "",
        "## Runtime",
        "",
        f"- Total wall time: {metadata['runtime_s'] / 60.0:.1f} min.",
        f"- Recorded Toolkit batch time: {metadata.get('recorded_batch_runtime_s', metadata['runtime_s']) / 60.0:.1f} min.",
        f"- Adsorption batch size requested: {metadata['adsorption_batch_size']}.",
        f"- Toolkit step cap: {metadata['toolkit_n_steps_cap']} steps.",
        f"- Force threshold: {metadata['toolkit_fmax']} eV/A.",
        f"- Max batch optimizer steps: {max_steps}.",
        f"- Mean pair median optimizer steps: {mean_steps:.1f}.",
        "",
        "Optimizer steps are batch-level steps. A single hard structure can keep "
        "a batch running after easier structures have already crossed the force "
        "threshold; per-structure `first_converged_step` is written in "
        "`step_statistics.csv`.",
        "",
        "## Slowest Pairs",
        "",
        slow[
            [
                "pair",
                "best_E_ads_eV",
                "n_green",
                "n_yellow",
                "n_red",
                "median_optimizer_nsteps",
                "max_optimizer_nsteps",
            ]
        ].to_markdown(index=False),
        "",
        "## Difficult Cases",
        "",
        (
            "No red/non-adsorbed configurations were detected."
            if difficult_df.empty
            else difficult_df[
                [
                    "pair",
                    "label",
                    "optimizer_nsteps",
                    "geometry_status",
                    "E_ads_eV",
                    "trajectory_path",
                ]
            ].head(20).to_markdown(index=False)
        ),
        "",
        "## Output Files",
        "",
        f"- Full results: `{paths['adsorption_results_csv']}`",
        f"- Pair summary: `{paths['pair_summary_csv']}`",
        f"- Batch summary: `{paths['batch_summary_csv']}`",
        f"- Step statistics: `{paths['step_statistics_csv']}`",
        f"- Difficult cases: `{paths['difficult_cases_csv']}`",
        f"- Metadata: `{paths['metadata']}`",
    ]
    if paths["report_md"].exists() and not _allow_overwrite():
        return
    paths["report_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _ensure_force_is_intentional(args.force)
    paths = surface_screen_output_paths(args.output_root)
    _ensure_precomputed_write_is_intentional(paths["root"])
    _ensure_dirs(paths)
    started_total = time.perf_counter()

    slabs = build_screen_slabs()
    preflight_configs = build_screen_configs(slabs)
    expected = screen_expected_counts()
    preflight_audit = audit_initial_configs(
        preflight_configs,
        {host: len(slab) for host, slab in slabs.items()},
    )
    if len(preflight_configs) != expected["adsorption_relaxations"]:
        raise RuntimeError(
            f"Expected {expected['adsorption_relaxations']} adsorption starts, "
            f"built {len(preflight_configs)}."
        )
    if "review" in set(preflight_audit["audit_status"]):
        n_review = int((preflight_audit["audit_status"] == "review").sum())
        raise RuntimeError(f"Preflight geometry audit found {n_review} short contacts.")
    if args.audit_only:
        _write_audit_tables_guarded(paths, slabs, preflight_configs)
    print(
        "Geometry audit passed: "
        f"{len(slabs)} slabs, {len(preflight_configs)} adsorption starts.",
        flush=True,
    )
    if args.audit_only:
        return 0

    backend = _build_backend(args, paths)
    write_trajectories = not args.no_trajectories
    batch_rows: list[dict[str, object]] = []

    clean_items, gas_items = _reference_items(slabs)
    clean_results: list[OptimizationResult] = []
    for index, chunk in enumerate(_chunk_items(clean_items, args.clean_batch_size), start=1):
        results, rows = _run_batch_with_fallback(
            backend,
            chunk,
            label=f"surface_screen_clean_batch_{index:02d}",
            batch_type="clean_slab",
            paths=paths,
            force=args.force,
            write_trajectories=write_trajectories,
            snapshot_frequency=args.snapshot_frequency,
            fallback_batch_size=max(1, args.fallback_batch_size),
        )
        clean_results.extend(results)
        batch_rows.extend(rows)
    clean_df, clean_by_host, relaxed_hosts = _clean_rows(clean_items, clean_results, paths)
    _write_csv_guarded(clean_df, paths["clean_slab_energies_csv"])

    gas_results: list[OptimizationResult] = []
    for index, chunk in enumerate(_chunk_items(gas_items, args.gas_batch_size), start=1):
        results, rows = _run_batch_with_fallback(
            backend,
            chunk,
            label=f"surface_screen_gas_batch_{index:02d}",
            batch_type="gas_reference",
            paths=paths,
            force=args.force,
            write_trajectories=write_trajectories,
            snapshot_frequency=args.snapshot_frequency,
            fallback_batch_size=max(1, args.fallback_batch_size),
        )
        gas_results.extend(results)
        batch_rows.extend(rows)
    gas_df, gas_by_adsorbate = _gas_rows(gas_items, gas_results, paths)
    _write_csv_guarded(gas_df, paths["gas_energies_csv"])

    configs = build_screen_configs(relaxed_hosts)
    _write_audit_tables_guarded(paths, relaxed_hosts, configs)

    pair_to_items: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in _adsorption_items(configs):
        pair_to_items[(str(item["host"]), str(item["adsorbate"]))].append(item)
    pair_groups = list(pair_to_items.values())
    work_batches = _pack_pair_groups(pair_groups, batch_size=args.batch_size)

    all_pair_tables: list[pd.DataFrame] = []
    for index, work_batch in enumerate(work_batches, start=1):
        results, rows = _run_batch_with_fallback(
            backend,
            work_batch,
            label=f"surface_screen_adsorption_batch_{index:02d}",
            batch_type="adsorption",
            paths=paths,
            force=args.force,
            write_trajectories=write_trajectories,
            snapshot_frequency=args.snapshot_frequency,
            fallback_batch_size=max(1, args.fallback_batch_size),
        )
        batch_rows.extend(rows)
        grouped_configs: dict[tuple[str, str], list[object]] = defaultdict(list)
        grouped_results: dict[tuple[str, str], list[OptimizationResult]] = defaultdict(list)
        for item, result in zip(work_batch, results):
            key = (str(item["host"]), str(item["adsorbate"]))
            grouped_configs[key].append(item["config"])
            grouped_results[key].append(result)
        for (host, adsorbate), group_configs in grouped_configs.items():
            group_results = grouped_results[(host, adsorbate)]
            pair_df = build_pair_results_table(
                host=host,
                adsorbate=adsorbate,
                configs=group_configs,
                opt_results=group_results,
                clean_slab_atoms=relaxed_hosts[host],
                e_clean_slab_ev=float(clean_by_host[host].energy),
                e_gas_ads_ev=float(gas_by_adsorbate[adsorbate].energy),
                execution_path=(
                    f"toolkit:{os.environ.get('TOOLKIT_CHECKPOINT', 'medium-mpa-0')}"
                    + (
                        f":{os.environ.get('TOOLKIT_HEAD')}"
                        if os.environ.get("TOOLKIT_HEAD")
                        else ""
                    )
                ),
            )
            item_paths_by_label = {
                str(item["label"]): _paths_for_item(paths, str(item["label"]), "adsorption")
                for item in work_batch
            }
            pair_df["initial_structure_path"] = [
                str(item_paths_by_label[label]["initial"]) for label in pair_df["label"]
            ]
            pair_df["relaxed_structure_path"] = [
                str(item_paths_by_label[label]["relaxed"]) for label in pair_df["label"]
            ]
            pair_df["trajectory_path"] = [
                str(item_paths_by_label[label]["trajectory"]) for label in pair_df["label"]
            ]
            pair_df["trajectory_log_path"] = [
                str(item_paths_by_label[label]["log"]) for label in pair_df["label"]
            ]
            pair_df["batch_label"] = f"surface_screen_adsorption_batch_{index:02d}"
            pair_csv = paths["tables"] / f"pair_results_{_safe(adsorbate)}_{_safe(host)}.csv"
            _write_csv_guarded(pair_df, pair_csv)
            all_pair_tables.append(pair_df)

    results_df = pd.concat(all_pair_tables, ignore_index=True)
    batch_summary_df = pd.DataFrame(batch_rows)
    step_statistics_df = build_step_statistics(
        results_df,
        green_step_max=GREEN_OPTIMIZER_STEP_MAX,
        yellow_step_max=YELLOW_OPTIMIZER_STEP_MAX,
        force_threshold_eV_A=STEP_FORCE_THRESHOLD_EV_A,
    )
    pair_summary_df = summarize_surface_screen_pairs(
        results_df,
        step_statistics_df,
        exclude_red_step_status=EXCLUDE_RED_STEP_STATUS_FROM_PAIR_MINIMUM,
    )
    heatmap_df = build_application_heatmap(
        pair_summary_df,
        adsorbate_hints={spec.name: spec.application_hint for spec in SCREEN_ADSORBATES},
    )
    difficult_df = build_difficult_cases(results_df, step_statistics_df)

    _write_csv_guarded(results_df, paths["adsorption_results_csv"])
    _write_csv_guarded(batch_summary_df, paths["batch_summary_csv"])
    _write_csv_guarded(step_statistics_df, paths["step_statistics_csv"])
    _write_csv_guarded(pair_summary_df, paths["pair_summary_csv"])
    _write_csv_guarded(heatmap_df, paths["application_heatmap_csv"])
    _write_csv_guarded(difficult_df, paths["difficult_cases_csv"])

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grid_version": SURFACE_SCREEN_GRID_VERSION,
        "output_root": str(paths["root"]),
        "backend": "toolkit",
        "toolkit_checkpoint": os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
        "toolkit_head": os.environ.get("TOOLKIT_HEAD", "") or None,
        "toolkit_device": str(getattr(backend, "device", os.environ.get("TOOLKIT_DEVICE", "cuda"))),
        "toolkit_dtype": os.environ.get("TOOLKIT_DTYPE", "float32"),
        "toolkit_enable_cueq": _env_bool("TOOLKIT_ENABLE_CUEQ", False),
        "toolkit_compile_model": _env_bool("TOOLKIT_COMPILE_MODEL", False),
        "toolkit_n_steps_cap": args.n_steps,
        "toolkit_fmax": args.fmax,
        "toolkit_dt": float(os.environ.get("TOOLKIT_DT", "0.01")),
        "toolkit_d3bj_enabled": False,
        "adsorption_batch_size": args.batch_size,
        "clean_batch_size": args.clean_batch_size,
        "gas_batch_size": args.gas_batch_size,
        "fallback_batch_size": args.fallback_batch_size,
        "write_trajectories": write_trajectories,
        "snapshot_frequency": args.snapshot_frequency,
        "ranking_policy": {
            "green_optimizer_step_max": GREEN_OPTIMIZER_STEP_MAX,
            "yellow_optimizer_step_max": YELLOW_OPTIMIZER_STEP_MAX,
            "step_force_threshold_eV_A": STEP_FORCE_THRESHOLD_EV_A,
            "exclude_red_step_status_from_pair_minimum": EXCLUDE_RED_STEP_STATUS_FROM_PAIR_MINIMUM,
        },
        "grid_policy": {
            "sites_by_host": {key: list(value) for key, value in SCREEN_SITES_BY_HOST.items()},
            "orientations_by_adsorbate": {
                spec.name: list(spec.orientations) for spec in SCREEN_ADSORBATES
            },
            "rotations_deg": list(SCREEN_ROTATIONS_DEG),
            "start_height_A": SCREEN_START_HEIGHT_A,
            "frozen_slab_fraction": SCREEN_FROZEN_SLAB_FRACTION,
        },
        "expected_counts": screen_expected_counts(),
        "actual_adsorption_relaxations": int(len(results_df)),
        "actual_core_relaxations": int(len(results_df) + len(clean_df) + len(gas_df)),
        "n_converged_adsorption": int(results_df["converged"].sum()),
        "n_reliable_adsorption": int(results_df["reliable_for_minimum"].sum()),
        "n_difficult_adsorption": int(len(difficult_df)),
        "runtime_s": time.perf_counter() - started_total,
        "recorded_batch_runtime_s": float(batch_summary_df["runtime_s"].sum()),
        "loaded_batches_from_cache": int(batch_summary_df.get("loaded_from_cache", pd.Series(dtype=bool)).sum()),
        "total_batches": int(len(batch_summary_df)),
        "paths": {key: str(path) for key, path in paths.items() if isinstance(path, Path)},
    }
    _write_json_guarded(metadata, paths["metadata"])
    _write_report(
        paths=paths,
        metadata=metadata,
        pair_summary_df=pair_summary_df,
        batch_summary_df=batch_summary_df,
        difficult_df=difficult_df,
    )

    print(f"Wrote {paths['adsorption_results_csv']}", flush=True)
    print(f"Wrote {paths['pair_summary_csv']}", flush=True)
    print(f"Wrote {paths['batch_summary_csv']}", flush=True)
    print(f"Wrote {paths['step_statistics_csv']}", flush=True)
    print(f"Wrote {paths['metadata']}", flush=True)
    print(f"Total invocation wall time: {metadata['runtime_s'] / 60.0:.1f} min", flush=True)
    print(f"Recorded Toolkit batch time: {metadata['recorded_batch_runtime_s'] / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
