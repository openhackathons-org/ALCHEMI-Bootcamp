"""Adsorption-specific Toolkit batch-size calibration helpers."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config_search import Configuration, build_config_grid
from .models import ase_to_atomic_data
from .oxide_slabs import build_alpha_alumina_0001_slab, build_tio2_110_slab
from .metal_slabs import build_cu111_slab
from .relaxation_backends import RelaxationBackendConfig, ToolkitBackend, get_relaxation_backend


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


def build_adsorption_calibration_pool(
    *,
    host_name: str = "TiO2(110)",
    adsorbate_name: str = "H2O",
    sites_filter: list[str] | None = None,
    orientations_filter: list[str] | None = None,
    rotations_deg: tuple[float, ...] = (0.0, 60.0, 120.0),
    heights_A: tuple[float, ...] = (2.2,),
    frozen_fraction: float = 0.5,
) -> list[Configuration]:
    """Build a small adsorption-configuration pool for batch calibration."""
    try:
        slab = HOST_BUILDERS[host_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown calibration host {host_name!r}") from exc
    return build_config_grid(
        host_name=host_name,
        slab=slab,
        adsorbate_name=adsorbate_name,
        sites_filter=sites_filter,
        orientations_filter=orientations_filter,
        rotations_deg=rotations_deg,
        heights_A=heights_A,
        frozen_fraction=frozen_fraction,
    )


def _cuda_snapshot(torch, device) -> dict[str, float | str]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {
            "gpu_total_gb": np.nan,
            "gpu_free_gb": np.nan,
            "process_allocated_gb": np.nan,
            "process_reserved_gb": np.nan,
            "process_peak_allocated_gb": np.nan,
            "process_peak_reserved_gb": np.nan,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "gpu_total_gb": total_bytes / 1024**3,
        "gpu_free_gb": free_bytes / 1024**3,
        "process_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "process_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "process_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "process_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _payloads_for_batch(configs: list[Configuration], batch_size: int):
    payloads = []
    selected = []
    for idx in range(batch_size):
        config = configs[idx % len(configs)]
        selected.append(config)
        payloads.append(
            ase_to_atomic_data(
                config.atoms,
                structure_id=f"{config.label}_calibration{idx:04d}",
                active_mask=config.active_mask,
            )
        )
    return payloads, selected


def _disable_fragile_torch_fusers(torch) -> None:
    """Avoid TorchScript/NVRTC fusion paths that can be fragile in notebooks."""
    for name in (
        "_jit_set_nvfuser_enabled",
        "_jit_set_texpr_fuser_enabled",
        "_jit_override_can_fuse_on_gpu",
    ):
        if hasattr(torch._C, name):
            try:
                getattr(torch._C, name)(False)
            except TypeError:
                pass


def _run_one_calibration_batch(
    *,
    backend: ToolkitBackend,
    configs: list[Configuration],
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    import torch

    _disable_fragile_torch_fusers(torch)
    device = backend.device
    payloads, selected_configs = _payloads_for_batch(configs, batch_size)
    total_atoms = sum(len(config.atoms) for config in selected_configs)
    active_atoms = [int(np.asarray(config.active_mask, dtype=bool).sum()) for config in selected_configs]
    atom_counts = [len(config.atoms) for config in selected_configs]

    data_list = [backend._to_atomic_data(payload) for payload in payloads]
    batch = backend.api.Batch.from_data_list(data_list, device=device)
    optimizer = backend.api.FIRE2(
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
    optimizer.register_hook(backend.api.FreezeAtomsHook())
    optimizer.register_hook(backend.api.NaNDetectorHook())

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    before = _cuda_snapshot(torch, device)
    started = perf_counter()
    try:
        batch = optimizer.run(batch)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        wall_time_s = perf_counter() - started
        after = _cuda_snapshot(torch, device)
        step_count = int(getattr(optimizer, "step_count", backend.config.toolkit_n_steps))
        return {
            "label": label,
            "batch_size": batch_size,
            "status": "ok",
            "wall_time_s": wall_time_s,
            "structures_per_s": batch_size / wall_time_s,
            "atoms_per_s": total_atoms / wall_time_s,
            "total_atoms": total_atoms,
            "atoms_per_structure_min": min(atom_counts),
            "atoms_per_structure_max": max(atom_counts),
            "active_atoms_per_structure_min": min(active_atoms),
            "active_atoms_per_structure_max": max(active_atoms),
            "optimizer_steps": step_count,
            "gpu_total_gb": before["gpu_total_gb"],
            "gpu_free_before_gb": before["gpu_free_gb"],
            "gpu_free_after_gb": after["gpu_free_gb"],
            "gpu_free_drop_gb": max(0.0, before["gpu_free_gb"] - after["gpu_free_gb"]),
            "process_peak_allocated_gb": after["process_peak_allocated_gb"],
            "process_peak_reserved_gb": after["process_peak_reserved_gb"],
            "error": "",
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
            raise
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        after = _cuda_snapshot(torch, device)
        return {
            "label": label,
            "batch_size": batch_size,
            "status": "oom",
            "wall_time_s": np.nan,
            "structures_per_s": np.nan,
            "atoms_per_s": np.nan,
            "total_atoms": total_atoms,
            "atoms_per_structure_min": min(atom_counts),
            "atoms_per_structure_max": max(atom_counts),
            "active_atoms_per_structure_min": min(active_atoms),
            "active_atoms_per_structure_max": max(active_atoms),
            "optimizer_steps": np.nan,
            "gpu_total_gb": before["gpu_total_gb"],
            "gpu_free_before_gb": before["gpu_free_gb"],
            "gpu_free_after_gb": after["gpu_free_gb"],
            "gpu_free_drop_gb": max(0.0, before["gpu_free_gb"] - after["gpu_free_gb"]),
            "process_peak_allocated_gb": after["process_peak_allocated_gb"],
            "process_peak_reserved_gb": after["process_peak_reserved_gb"],
            "error": f"CUDA out of memory: {exc}",
        }


def run_adsorption_batch_calibration(
    *,
    configs: list[Configuration],
    model_specs: list[dict[str, str | None]],
    batch_sizes: list[int],
    output_csv: str | Path,
    cache_dir: str | Path,
    toolkit_device: str = "cuda",
    toolkit_dtype: str = "float32",
    toolkit_dt: float = 0.01,
    n_steps: int = 40,
    fmax: float = 0.0,
    compile_model: bool = False,
    enable_cueq: bool = False,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Run short adsorption relaxations to calibrate batch size by model."""
    if not configs:
        raise ValueError("Calibration config pool is empty.")

    rows: list[dict[str, Any]] = []
    total = len(model_specs) * len(batch_sizes)
    done = 0
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)

    for spec in model_specs:
        label = str(spec["label"])
        backend = get_relaxation_backend(
            RelaxationBackendConfig(
                name="toolkit",
                cache_dir=str(cache_dir / label.replace(" ", "_").replace("+", "plus")),
                use_cached_responses=False,
                toolkit_checkpoint=str(spec["checkpoint"]),
                toolkit_head=spec.get("head"),
                toolkit_device=toolkit_device,
                toolkit_dtype=toolkit_dtype,
                toolkit_compile_model=compile_model,
                toolkit_enable_cueq=enable_cueq,
                toolkit_dt=toolkit_dt,
                toolkit_n_steps=n_steps,
                toolkit_fmax=fmax,
                toolkit_require_d3bj=False,
                toolkit_d3bj=None,
            )
        )
        for batch_size in batch_sizes:
            if progress is not None:
                progress.update(done=done, message=f"{label}: batch size {batch_size}")
            row = _run_one_calibration_batch(
                backend=backend,
                configs=configs,
                batch_size=batch_size,
                label=label,
            )
            row["checkpoint"] = spec["checkpoint"]
            row["head"] = spec.get("head") or ""
            rows.append(row)
            done += 1
            if progress is not None:
                progress.update(done=done, message=f"{label}: batch size {batch_size} {row['status']}")
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            if row["status"] == "oom":
                if progress is not None:
                    progress.total = done
                    progress.update(done=done, message=f"{label}: stopped after batch size {batch_size} reached memory limit")
                break

    return pd.DataFrame(rows)


def summarize_adsorption_batch_calibration(
    calibration_df: pd.DataFrame,
    *,
    memory_fraction: float = 0.80,
    near_best_fraction: float = 0.90,
) -> pd.DataFrame:
    """Recommend a conservative batch size from calibration results."""
    rows = []
    for label, df in calibration_df.groupby("label", sort=False):
        ok = df[df["status"] == "ok"].copy()
        if ok.empty:
            rows.append({
                "model": label,
                "recommended_batch_size": np.nan,
                "best_measured_batch_size": np.nan,
                "best_structures_per_s": np.nan,
                "recommended_structures_per_s": np.nan,
                "recommended_atoms_per_s": np.nan,
                "peak_allocated_gb": np.nan,
                "peak_reserved_gb": np.nan,
                "observed_free_drop_gb": np.nan,
                "gpu_free_before_gb": np.nan,
                "reserved_fraction_of_free": np.nan,
                "headroom_rule": f"peak reserved <= {memory_fraction:.0%} of free GPU memory before the run",
            })
            continue
        ok["within_memory_headroom"] = (
            ok["process_peak_reserved_gb"] <= memory_fraction * ok["gpu_free_before_gb"]
        )
        if "gpu_free_drop_gb" not in ok:
            ok["gpu_free_drop_gb"] = (
                ok["gpu_free_before_gb"] - ok["gpu_free_after_gb"]
            ).clip(lower=0.0)
        headroom = ok[ok["within_memory_headroom"]]
        if headroom.empty:
            headroom = ok
        best_rate = headroom["structures_per_s"].max()
        near_best = headroom[headroom["structures_per_s"] >= near_best_fraction * best_rate]
        recommendation = near_best.sort_values(["batch_size"]).iloc[0]
        best = ok.loc[ok["structures_per_s"].idxmax()]
        rows.append({
            "model": label,
            "recommended_batch_size": int(recommendation["batch_size"]),
            "best_measured_batch_size": int(best["batch_size"]),
            "best_structures_per_s": best["structures_per_s"],
            "recommended_structures_per_s": recommendation["structures_per_s"],
            "recommended_atoms_per_s": recommendation["atoms_per_s"],
            "peak_allocated_gb": recommendation["process_peak_allocated_gb"],
            "peak_reserved_gb": recommendation["process_peak_reserved_gb"],
            "observed_free_drop_gb": recommendation["gpu_free_drop_gb"],
            "gpu_free_before_gb": recommendation["gpu_free_before_gb"],
            "reserved_fraction_of_free": (
                recommendation["process_peak_reserved_gb"] / recommendation["gpu_free_before_gb"]
            ),
            "headroom_rule": (
                f"smallest batch within {near_best_fraction:.0%} of best throughput "
                f"and below {memory_fraction:.0%} of free VRAM using peak reserved memory"
            ),
        })
    return pd.DataFrame(rows)


def _validate_precomputed_calibration(
    calibration_df: pd.DataFrame,
    *,
    model_specs: list[dict[str, str | None]],
    batch_sizes: list[int],
    source: Path,
) -> pd.DataFrame:
    """Return the requested calibration slice, or fail on stale provenance."""
    required_columns = {"label", "batch_size", "status"}
    missing_columns = sorted(required_columns - set(calibration_df.columns))
    if missing_columns:
        raise RuntimeError(
            "Precomputed adsorption batch-calibration table is missing "
            f"required columns {missing_columns}: {source}"
        )

    filtered_parts = []
    for spec in model_specs:
        label = str(spec["label"])
        expected_checkpoint = str(spec["checkpoint"])
        expected_head = spec.get("head") or ""
        label_mask = calibration_df["label"].astype(str) == label
        metadata_mask = pd.Series(False, index=calibration_df.index)
        if {"checkpoint", "head"}.issubset(calibration_df.columns):
            metadata_mask = (
                calibration_df["checkpoint"].astype(str).eq(expected_checkpoint)
                & calibration_df["head"].fillna("").astype(str).eq(expected_head)
            )
        df_label = calibration_df[label_mask | metadata_mask].copy()
        if df_label.empty:
            raise RuntimeError(
                "Precomputed adsorption batch-calibration table does not include "
                f"model {label!r} with checkpoint={expected_checkpoint!r}, "
                f"head={expected_head!r}. Set USE_SAVED_TUTORIAL_RESULTS = False "
                "to regenerate it."
            )
        df_label["label"] = label

        available_batches = set(df_label["batch_size"].astype(int))
        missing_batches = [
            size for size in batch_sizes if size not in available_batches
        ]
        if missing_batches:
            raise RuntimeError(
                "Precomputed adsorption batch-calibration table does not "
                f"include batch sizes {missing_batches} for {label!r}. Set "
                "USE_SAVED_TUTORIAL_RESULTS = False to regenerate it."
            )

        if "checkpoint" in df_label.columns:
            observed_checkpoints = set(df_label["checkpoint"].astype(str))
            if expected_checkpoint not in observed_checkpoints:
                raise RuntimeError(
                    "Precomputed adsorption batch-calibration table has stale "
                    f"checkpoint metadata for {label!r}. Expected "
                    f"{expected_checkpoint!r}, found {sorted(observed_checkpoints)}."
                )

        if "head" in df_label.columns:
            observed_heads = set(df_label["head"].fillna("").astype(str))
            if expected_head not in observed_heads:
                raise RuntimeError(
                    "Precomputed adsorption batch-calibration table has stale "
                    f"head metadata for {label!r}. Expected {expected_head!r}, "
                    f"found {sorted(observed_heads)}."
                )

        df_label = df_label[df_label["batch_size"].astype(int).isin(batch_sizes)]
        filtered_parts.append(df_label)

    return pd.concat(filtered_parts, ignore_index=True)


def load_or_run_adsorption_batch_calibration(
    *,
    use_precomputed: bool,
    run_scope: str,
    batch_sizes: list[int],
    cache_path: str | Path,
    full_cache_path: str | Path,
    tutorial_relpath,
    host_name: str,
    adsorbate_name: str,
    model_specs: list[dict[str, str | None]],
    output_dir: str | Path,
    toolkit_device: str,
    toolkit_dtype: str,
    toolkit_dt: float,
    n_steps: int,
    memory_fraction: float,
    compile_model: bool,
    enable_cueq: bool,
    progress_factory=None,
    display_fn=None,
    markdown_cls=None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load or run the notebook's adsorption batch-size calibration."""
    cache_path = Path(cache_path)
    full_cache_path = Path(full_cache_path)
    cache_source = cache_path
    if (
        use_precomputed
        and not cache_source.exists()
        and run_scope == "short"
        and full_cache_path.exists()
    ):
        cache_source = full_cache_path

    if use_precomputed:
        if not cache_source.exists():
            raise RuntimeError(
                "Precomputed adsorption batch-calibration table not found. "
                f"Expected {tutorial_relpath(cache_path)}. "
                "Set USE_SAVED_TUTORIAL_RESULTS = False to recompute it."
            )
        calibration_df = _validate_precomputed_calibration(
            pd.read_csv(cache_source),
            model_specs=model_specs,
            batch_sizes=batch_sizes,
            source=cache_source,
        )
        if display_fn is not None and markdown_cls is not None:
            display_fn(
                markdown_cls(
                    "Using saved adsorption batch calibration from "
                    f"`{tutorial_relpath(cache_source)}`."
                )
            )
    else:
        calibration_pool = build_adsorption_calibration_pool(
            host_name=host_name,
            adsorbate_name=adsorbate_name,
            rotations_deg=(0.0, 60.0, 120.0),
            heights_A=(2.2,),
            frozen_fraction=0.5,
        )
        print(
            f"Calibration pool: {len(calibration_pool)} starting structures; "
            "larger batches cycle through this pool to measure GPU throughput."
        )
        progress = None
        if progress_factory is not None:
            progress = progress_factory(
                title="Adsorption batch calibration",
                total=len(model_specs) * len(batch_sizes),
                unit="model-batch tests",
                message="building short relaxation batches",
                average_label="s/test",
                width_px=680,
            )
        calibration_df = run_adsorption_batch_calibration(
            configs=calibration_pool,
            model_specs=model_specs,
            batch_sizes=batch_sizes,
            output_csv=cache_path,
            cache_dir=Path(output_dir) / "adsorption_batch_calibration_work",
            toolkit_device=toolkit_device,
            toolkit_dtype=toolkit_dtype,
            toolkit_dt=toolkit_dt,
            n_steps=n_steps,
            fmax=0.0,
            compile_model=compile_model,
            enable_cueq=enable_cueq,
            progress=progress,
        )

    summary = summarize_adsorption_batch_calibration(
        calibration_df,
        memory_fraction=memory_fraction,
    )
    return calibration_df, summary, cache_source


def display_adsorption_batch_calibration_results(
    calibration_df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    display_fn,
    markdown_cls,
) -> None:
    """Print a compact notebook-facing calibration summary."""
    _ = calibration_df
    _ = display_fn, markdown_cls

    print("Recommended adsorption batch sizes:")
    for _, row in summary.iterrows():
        if pd.isna(row.get("recommended_batch_size")):
            print(f"  {row.get('model', 'model')}: no successful calibration point")
            continue
        print(
            f"  {row['model']}: batch {int(row['recommended_batch_size'])} "
            f"({row['recommended_structures_per_s']:.2f} structures/s, "
            f"{row['recommended_atoms_per_s']:.0f} atoms/s, "
            f"{row['peak_reserved_gb']:.2f} GB reserved footprint, "
            f"{row['observed_free_drop_gb']:.2f} GB observed free-VRAM drop)"
        )
    print("Full timing rows are saved to CSV; the figure shows throughput saturation and measured VRAM footprint.")
