#!/usr/bin/env python3
"""Run a small record-backed OC20Dense model check with native Toolkit.

This script uses the official OC20Dense/AdsorbML metadata and input structures:

* oc20dense_mapping.pkl maps integer sid values to system/config metadata.
* oc20dense_targets.pkl stores released DFT adsorption energies by system/config.
* A slim tutorial checkout stores the selected initial adsorbate-slab structures
  as extxyz. A full OC20Dense checkout may instead provide oc20dense.lmdb.

The default benchmark systems are a closed-shell set: adsorbed water, ammonia,
and nitrogen. They avoid CH3-containing adsorbates and radical-like gas
references so the same systems can be used for single-point checks, relaxation
checks, and the later MACE adsorption-energy calculation.

The current comparison is intentionally rank-based. MACE-MPA-0 total energies
are used to rank configurations within the same OC20Dense system in two ways:

1. single-point scoring on the exact OC20Dense input coordinates; and
2. batched Toolkit relaxation followed by final-energy ranking.

The selected configuration is then checked against the released DFT
adsorption-energy ranking for that system. The companion
run_oc20dense_dft_final_single_points.py and
run_oc20dense_mace_adsorption_energies.py scripts add the DFT-relaxed final single-point
and explicit MACE adsorption-energy layers.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART1))

RUNTIME_CACHE = PART1 / "outputs" / "runtime_cache"

# Warp compiles Toolkit kernels on first import. Keep runtime caches together
# instead of mixing them with scientific result artifacts.
os.environ.setdefault("WARP_CACHE_PATH", str(RUNTIME_CACHE / "warp"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))
os.environ.setdefault(
    "CUEQ_TRITON_CACHE_DIR",
    str(RUNTIME_CACHE / "cueq_triton"),
)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("TRITON_CACHE_DIR", str(RUNTIME_CACHE / "triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(RUNTIME_CACHE / "torchinductor"),
)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_CU13_LIB = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
    / "cu13"
    / "lib"
)
if _CU13_LIB.exists() and os.environ.get("OC20DENSE_LD_REEXEC") != "1":
    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if str(_CU13_LIB) not in ld_paths:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{_CU13_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        ).rstrip(":")
        os.environ["OC20DENSE_LD_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])
elif _CU13_LIB.exists() and os.environ.get("OC20DENSE_LD_REEXEC") == "1":
    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if str(_CU13_LIB) not in ld_paths:
        print(
            "OC20DENSE_LD_REEXEC=1 is already set, so the script will not re-exec "
            f"to prepend {_CU13_LIB} to LD_LIBRARY_PATH. If Toolkit CUDA imports fail, "
            "unset OC20DENSE_LD_REEXEC or set LD_LIBRARY_PATH before launching.",
            file=sys.stderr,
        )

from helpers import (  # noqa: E402
    OptimizationResult,
    RelaxationBackendConfig,
    ase_to_atomic_data,
    atomic_data_to_ase,
    get_relaxation_backend,
    run_toolkit_relaxation_with_trajectory,
)
from _oc20dense_common import (  # noqa: E402
    CLOSED_SHELL_ADSORBATE_REFERENCES,
    DEFAULT_DATA_ROOT,
    DEFAULT_CLOSED_SHELL_SYSTEMS,
    DEFAULT_INITIAL_STRUCTURE_DIR,
    DEFAULT_SYSTEMS,
    FULL_DATA_NOTICE,
    MACE_EADS_REFERENCE_STATUS,
    MACE_RANK_BASIS,
    TOOLKIT_PROVENANCE_COLUMNS,
    oc20dense_archive_file,
    oc20dense_lmdb_path,
    oc20dense_mapping_file,
    require_precomputed_write_allowed,
    toolkit_cache_matches,
    toolkit_model_label,
    toolkit_provenance_from_env,
    toolkit_provenance_mismatch,
)


DEFAULT_OUTDIR = (
    PART1
    / "outputs"
    / "precomputed"
    / "accuracy"
    / "oc20dense_closed_shell_trajectory_mace_mpa0"
)

OFFICIAL_SOURCES = {
    "oc20dense_docs": "https://fair-chem.github.io/catalysts/datasets/oc20dense.html",
    "adsorbml_readme": "https://github.com/Open-Catalyst-Project/AdsorbML",
    "mapping_archive": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/"
        "oc20_dense_mappings.tar.gz"
    ),
    "lmdb_archive": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/"
        "oc20_dense_data.tar.gz"
    ),
}


@dataclass(frozen=True)
class SelectedConfig:
    sid: int
    system_id: str
    config_id: str
    mpid: str
    miller_idx: tuple[int, int, int]
    top: bool
    adsorbate: str
    dft_adsorption_energy_eV: float
    dft_rank: int


class _FakePyGObject:
    """Small pickle shim for torch_geometric.data.Data/GlobalStorage."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _install_pyg_pickle_shim() -> None:
    modules = {
        "torch_geometric": types.ModuleType("torch_geometric"),
        "torch_geometric.data": types.ModuleType("torch_geometric.data"),
        "torch_geometric.data.data": types.ModuleType("torch_geometric.data.data"),
        "torch_geometric.data.storage": types.ModuleType("torch_geometric.data.storage"),
    }
    modules["torch_geometric.data.data"].Data = _FakePyGObject
    modules["torch_geometric.data.storage"].GlobalStorage = _FakePyGObject
    sys.modules.update(modules)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native Toolkit on a few official OC20Dense systems.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help=(
            "OC20Dense system_id values to run. The default set is closed-shell: "
            "one *OH2, one *NH3, and one *N2 system."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing OC20Dense mapping pickles and selected source structures.",
    )
    parser.add_argument(
        "--initial-structure-dir",
        type=Path,
        default=DEFAULT_INITIAL_STRUCTURE_DIR,
        help=(
            "Directory of selected OC20Dense initial structures as extxyz. "
            "Used when the full OC20Dense LMDB is not present."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for local reproducibility artifacts.",
    )
    parser.add_argument(
        "--max-configs-per-system",
        type=int,
        default=0,
        help="0 means all DFT-targeted configs; positive values select ranks evenly.",
    )
    parser.add_argument(
        "--config-ids",
        nargs="*",
        default=None,
        help=(
            "Optional explicit OC20Dense config_id values to run for each "
            "selected system. Useful for exact subset recomputation."
        ),
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with system_id, config_id, and sid columns. When set, "
            "the script runs exactly those OC20Dense rows instead of applying "
            "--systems/--max-configs-per-system/--config-ids selection."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("OC20DENSE_CHUNK_SIZE", "12")),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(os.environ.get("OC20DENSE_TOOLKIT_N_STEPS", "200")),
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=float(os.environ.get("TOOLKIT_FMAX", "0.05")),
    )
    parser.add_argument(
        "--allow-unpinned-adsorbates",
        action="store_true",
        help=(
            "Allow systems outside the closed-shell reference map. Use only for "
            "rank/geometry diagnostics, not MACE adsorption-energy parity."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached chunks.")
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable per-step Toolkit trajectory and energy/force log output.",
    )
    return parser.parse_args()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _safe(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("*", "star")
    )


def _read_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required OC20Dense file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _lmdb_path(data_root: Path) -> Path:
    path = oc20dense_lmdb_path(data_root)
    if not path.exists():
        raise FileNotFoundError(
            "Missing extracted OC20Dense LMDB. Expected "
            f"{path}. {FULL_DATA_NOTICE}"
        )
    return path


def _open_lmdb(data_root: Path) -> lmdb.Environment:
    return lmdb.open(
        str(_lmdb_path(data_root)),
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )


def _load_oc20dense_atom_data(txn: lmdb.Transaction, sid: int) -> dict[str, Any]:
    raw = txn.get(str(sid).encode("ascii"))
    if raw is None:
        raise KeyError(f"LMDB entry for sid={sid} is missing")
    obj = pickle.loads(raw)
    return dict(obj._store._mapping)


def _initial_structure_path(initial_structure_dir: Path, label: str) -> Path:
    return Path(initial_structure_dir) / f"{_safe(label)}.extxyz"


def _atoms_from_slim_initial_structure(
    initial_structure_dir: Path,
    *,
    config: "SelectedConfig",
    label: str,
) -> tuple[Atoms, list[bool]]:
    path = _initial_structure_path(initial_structure_dir, label)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing slim OC20Dense initial structure for {label}: {path}. "
            f"{FULL_DATA_NOTICE}"
        )
    atoms = ase_read(path)
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if not tags.size or not np.any(tags == 2):
        raise ValueError(
            f"Slim OC20Dense initial structure lacks OC20 tags needed for {label}: {path}"
        )
    active_mask = [bool(tag != 0) for tag in tags]
    fixed = tags == 0
    atoms.set_constraint(FixAtoms(mask=fixed.tolist()))
    atoms.info.update(
        {
            "label": label,
            "sid": int(config.sid),
            "config": config.config_id,
            "system_id": config.system_id,
        }
    )
    return atoms, active_mask


def _atoms_for_selected_config(
    config: "SelectedConfig",
    *,
    txn: lmdb.Transaction | None,
    initial_structure_dir: Path,
) -> tuple[str, Atoms, list[bool]]:
    label = f"{config.system_id}_{config.config_id}_sid{config.sid}"
    if txn is None:
        atoms, active_mask = _atoms_from_slim_initial_structure(
            initial_structure_dir,
            config=config,
            label=label,
        )
    else:
        atom_data = _load_oc20dense_atom_data(txn, config.sid)
        atoms, active_mask = _atoms_from_oc20dense(atom_data, label=label)
    return label, atoms, active_mask


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _atoms_from_oc20dense(data: dict[str, Any], *, label: str) -> tuple[Atoms, list[bool]]:
    positions = _tensor_to_numpy(data["pos"]).reshape(-1, 3)
    numbers = _tensor_to_numpy(data["atomic_numbers"]).astype(int).reshape(-1)
    cell = _tensor_to_numpy(data["cell"]).reshape(3, 3)
    tags = _tensor_to_numpy(data["tags"]).astype(int).reshape(-1)
    fixed = tags == 0
    active_mask = [bool(tag != 0) for tag in tags]

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[True, True, True])
    atoms.set_tags(tags)
    atoms.info["label"] = label
    atoms.info["sid"] = int(data.get("sid", -1))
    atoms.info["config"] = int(data.get("config", -1))
    atoms.set_constraint(FixAtoms(mask=fixed.tolist()))
    return atoms, active_mask


def _max_force(forces: list[float], active_mask: list[bool] | None = None) -> float:
    arr = np.asarray(forces, dtype=float).reshape(-1, 3)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape[0] == arr.shape[0] and mask.any():
            arr = arr[mask]
    if arr.size == 0:
        return float("nan")
    return float(np.linalg.norm(arr, axis=1).max())


def _result_to_json(result: OptimizationResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _load_result(path: Path) -> OptimizationResult:
    return OptimizationResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _write_result(path: Path, result: OptimizationResult) -> None:
    path.write_text(json.dumps(_result_to_json(result), indent=2), encoding="utf-8")


def _select_system_configs(
    *,
    system_id: str,
    mapping_by_system: dict[str, list[tuple[int, dict[str, Any]]]],
    targets: dict[str, list[tuple[str, float]]],
    max_configs: int,
    config_ids: set[str] | None = None,
) -> list[SelectedConfig]:
    if system_id not in mapping_by_system:
        raise KeyError(f"Unknown system_id in mapping: {system_id}")
    if system_id not in targets:
        raise KeyError(f"Unknown system_id in targets: {system_id}")

    meta_by_config = {
        str(meta["config_id"]): (sid, meta)
        for sid, meta in mapping_by_system[system_id]
    }
    ranked_target_rows = [
        (rank, str(config_id), float(energy))
        for rank, (config_id, energy) in enumerate(
            sorted(targets[system_id], key=lambda item: item[1]),
            start=1,
        )
    ]
    if config_ids is not None:
        ranked_target_rows = [
            row for row in ranked_target_rows if row[1] in config_ids
        ]
    elif max_configs > 0 and max_configs < len(ranked_target_rows):
        ranks = np.linspace(0, len(ranked_target_rows) - 1, max_configs)
        keep = sorted({int(round(rank)) for rank in ranks})
        ranked_target_rows = [ranked_target_rows[index] for index in keep]

    selected: list[SelectedConfig] = []
    for rank, config_id, energy in ranked_target_rows:
        if config_id not in meta_by_config:
            continue
        sid, meta = meta_by_config[config_id]
        selected.append(
            SelectedConfig(
                sid=int(sid),
                system_id=system_id,
                config_id=str(config_id),
                mpid=str(meta["mpid"]),
                miller_idx=tuple(int(x) for x in meta["miller_idx"]),
                top=bool(meta["top"]),
                adsorbate=str(meta["adsorbate"]),
                dft_adsorption_energy_eV=float(energy),
                dft_rank=rank,
            )
        )
    return selected


def _read_selection_csv(path: Path) -> dict[str, list[tuple[str, int]]]:
    required = {"system_id", "config_id", "sid"}
    selection = pd.read_csv(path, dtype={"system_id": str, "config_id": str})
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(f"Selection CSV is missing columns: {missing}")

    selected: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for row in selection.itertuples(index=False):
        system_id = str(getattr(row, "system_id"))
        config_id = str(getattr(row, "config_id"))
        sid = int(getattr(row, "sid"))
        key = (system_id, config_id, sid)
        if key in seen:
            raise ValueError(f"Duplicate selection row: {key}")
        seen.add(key)
        selected[system_id].append((config_id, sid))
    if not selected:
        raise ValueError(f"Selection CSV has no rows: {path}")
    return selected


def _select_from_exact_rows(
    *,
    selection_by_system: dict[str, list[tuple[str, int]]],
    mapping_by_system: dict[str, list[tuple[int, dict[str, Any]]]],
    targets: dict[str, list[tuple[str, float]]],
) -> dict[str, list[SelectedConfig]]:
    selected_by_system: dict[str, list[SelectedConfig]] = {}
    for system_id, exact_keys in selection_by_system.items():
        all_configs = _select_system_configs(
            system_id=system_id,
            mapping_by_system=mapping_by_system,
            targets=targets,
            max_configs=0,
            config_ids=None,
        )
        by_key = {(config.config_id, int(config.sid)): config for config in all_configs}
        missing = [key for key in exact_keys if key not in by_key]
        if missing:
            raise KeyError(f"Selection CSV rows were not found for {system_id}: {missing}")
        selected_by_system[system_id] = [by_key[key] for key in exact_keys]
    return selected_by_system


def _reference_species_for_adsorbate(adsorbate: str) -> str:
    return CLOSED_SHELL_ADSORBATE_REFERENCES.get(str(adsorbate), "unpinned")


def _validate_closed_shell_reference_set(
    selected_by_system: dict[str, list[SelectedConfig]],
    *,
    allow_unpinned_adsorbates: bool,
) -> None:
    unpinned: list[str] = []
    empty: list[str] = []
    for system_id, selected in selected_by_system.items():
        if not selected:
            empty.append(system_id)
            continue
        adsorbates = {config.adsorbate for config in selected}
        if len(adsorbates) != 1:
            unpinned.append(f"{system_id}: mixed adsorbates {sorted(adsorbates)}")
            continue
        adsorbate = next(iter(adsorbates))
        if adsorbate not in CLOSED_SHELL_ADSORBATE_REFERENCES:
            unpinned.append(f"{system_id}: {adsorbate}")

    if empty:
        raise ValueError(
            "No target-backed configurations were selected for: "
            + ", ".join(sorted(empty))
        )
    if unpinned and not allow_unpinned_adsorbates:
        allowed = ", ".join(
            f"{ads}->{ref}" for ads, ref in CLOSED_SHELL_ADSORBATE_REFERENCES.items()
        )
        raise ValueError(
            "The closed-shell benchmark slice is limited to adsorbates with "
            f"pinned neutral gas references ({allowed}). Off-scope systems: "
            + "; ".join(unpinned)
            + ". Pass --allow-unpinned-adsorbates only for rank/geometry diagnostics."
        )


def _build_backend(args: argparse.Namespace):
    device = os.environ.get("TOOLKIT_DEVICE", "cuda")
    if device == "auto":
        device = "cuda"
    return get_relaxation_backend(
        RelaxationBackendConfig(
            name="toolkit",
            cache_dir=str(args.outdir / "backend_cache"),
            toolkit_checkpoint=os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
            toolkit_device=device,
            toolkit_dtype=os.environ.get("TOOLKIT_DTYPE", "float32"),
            toolkit_enable_cueq=_env_bool("TOOLKIT_ENABLE_CUEQ", False),
            toolkit_compile_model=_env_bool("TOOLKIT_COMPILE_MODEL", False),
            toolkit_head=os.environ.get("TOOLKIT_HEAD") or None,
            toolkit_n_steps=args.n_steps,
            toolkit_fmax=args.fmax,
            toolkit_require_d3bj=False,
            toolkit_d3bj=None,
        )
    )


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - metadata helper
        return f"{type(exc).__name__}: {exc}"
    text = (completed.stdout or completed.stderr).strip()
    return text[:4000]


def _cuda_memory_snapshot(device: str | Any) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {}
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    device_index = torch.cuda.current_device()
    return {
        "cuda_device": torch.cuda.get_device_name(device_index),
        "cuda_allocated_mb": torch.cuda.memory_allocated(device_index) / 1024**2,
        "cuda_reserved_mb": torch.cuda.memory_reserved(device_index) / 1024**2,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device_index) / 1024**2,
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device_index) / 1024**2,
    }


def _reset_cuda_peak_memory(device: str | Any) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _memory_fields(prefix: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in snapshot.items()}


def _memory_message(snapshot: dict[str, Any]) -> str:
    if not snapshot:
        return "CUDA memory: unavailable"
    return (
        "CUDA memory: "
        f"allocated {snapshot['cuda_allocated_mb']:.0f} MiB, "
        f"reserved {snapshot['cuda_reserved_mb']:.0f} MiB, "
        f"peak allocated {snapshot['cuda_peak_allocated_mb']:.0f} MiB"
    )


def _file_md5(path: Path) -> str | None:
    if not path.exists():
        return None
    return _command_output(["md5sum", str(path)]).split()[0]


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "ase", "torch", "nvalchemi", "lmdb"):
        try:
            module = __import__(name)
        except Exception as exc:  # pragma: no cover - metadata helper
            versions[name] = f"unavailable: {type(exc).__name__}"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "raw": outdir / "raw_results",
        "initial": outdir / "structures" / "initial",
        "relaxed": outdir / "structures" / "relaxed",
        "trajectories": outdir / "structures" / "toolkit_trajectories",
        "initial_single_point": outdir / "structures" / "initial_mace_sp",
        "initial_single_point_logs": outdir / "single_point_logs" / "initial",
        "trajectory_logs": outdir / "trajectory_logs",
        "single_point": outdir / "single_point",
        "chunks": outdir / "chunks",
        "tables": outdir / "tables",
        "reports": outdir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _relax_chunk(
    *,
    backend: Any,
    chunk: list[tuple[SelectedConfig, Atoms, list[bool]]],
    chunk_label: str,
    paths: dict[str, Path],
    force: bool,
    write_trajectory: bool,
    provenance: dict[str, Any],
) -> tuple[list[OptimizationResult], float, dict[str, Any]]:
    chunk_json = paths["chunks"] / f"{chunk_label}.json"
    chunk_metadata = paths["chunks"] / f"{chunk_label}.metadata.json"
    trajectory_paths = [
        paths["trajectories"]
        / f"{_safe(config.system_id + '_' + config.config_id + '_sid' + str(config.sid))}.extxyz"
        for config, _atoms, _active_mask in chunk
    ]
    log_paths = [
        paths["trajectory_logs"]
        / f"{_safe(config.system_id + '_' + config.config_id + '_sid' + str(config.sid))}.csv"
        for config, _atoms, _active_mask in chunk
    ]
    trajectories_available = (
        not write_trajectory
        or all(path.exists() for path in trajectory_paths + log_paths)
    )
    if chunk_json.exists() and trajectories_available and not force:
        raw = json.loads(chunk_json.read_text(encoding="utf-8"))
        metadata = (
            json.loads(chunk_metadata.read_text(encoding="utf-8"))
            if chunk_metadata.exists()
            else {}
        )
        if toolkit_cache_matches(metadata, provenance):
            return [
                OptimizationResult.model_validate(item) for item in raw
            ], float(metadata.get("runtime_s", 0.0)), metadata
        mismatch = toolkit_provenance_mismatch(metadata, provenance)
        print(
            f"    {chunk_label} relaxation cache provenance mismatch, "
            f"recomputing: {mismatch}",
            flush=True,
        )

    payloads = [
        ase_to_atomic_data(
            atoms,
            structure_id=f"oc20dense_sid_{config.sid}_{config.config_id}",
            active_mask=active_mask,
        )
        for config, atoms, active_mask in chunk
    ]
    n_atoms = sum(len(atoms) for _config, atoms, _active_mask in chunk)
    _reset_cuda_peak_memory(backend.device)
    before_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"    {chunk_label} relaxation batch: {len(chunk)} configs, "
        f"{n_atoms} atoms; {_memory_message(before_memory)}",
        flush=True,
    )
    start = time.perf_counter()
    if write_trajectory:
        reply = run_toolkit_relaxation_with_trajectory(
            backend,
            payloads,
            label=chunk_label,
            trajectory_paths=trajectory_paths,
            log_paths=log_paths,
            cellopt=False,
        )
    else:
        reply = backend.relax(payloads, label=chunk_label, cellopt=False)
    runtime_s = time.perf_counter() - start
    after_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"    {chunk_label} relaxation done: {runtime_s:.2f} s; "
        f"{_memory_message(after_memory)}",
        flush=True,
    )
    chunk_json.write_text(
        json.dumps([_result_to_json(result) for result in reply.atoms], indent=2),
        encoding="utf-8",
    )
    metadata = {
        "chunk_label": chunk_label,
        "n_configs": len(chunk),
        "n_atoms": n_atoms,
        "runtime_s": runtime_s,
        "max_optimizer_nsteps": max(int(r.optimizer_nsteps) for r in reply.atoms),
        "n_backend_converged": sum(bool(r.converged) for r in reply.atoms),
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "trajectory_log_paths": [str(path) for path in log_paths],
        **provenance,
        **_memory_fields("relax_cuda_after", after_memory),
    }
    chunk_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return reply.atoms, runtime_s, metadata


def _model_tensor(outputs: Any, name: str) -> Any:
    if isinstance(outputs, dict):
        value = outputs.get(name)
    else:
        value = getattr(outputs, name, None)
    if value is None:
        raise RuntimeError(f"Toolkit single-point output is missing `{name}`.")
    return value


def _single_point_chunk(
    *,
    backend: Any,
    chunk: list[tuple[SelectedConfig, Atoms, list[bool]]],
    chunk_label: str,
    paths: dict[str, Path],
    force: bool,
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    sp_json = paths["single_point"] / f"{chunk_label}.json"
    sp_metadata = paths["single_point"] / f"{chunk_label}.metadata.json"
    if sp_json.exists() and not force:
        rows = json.loads(sp_json.read_text(encoding="utf-8"))
        metadata = (
            json.loads(sp_metadata.read_text(encoding="utf-8"))
            if sp_metadata.exists()
            else {}
        )
        if toolkit_cache_matches(metadata, provenance):
            return rows, float(metadata.get("runtime_s", 0.0)), metadata
        mismatch = toolkit_provenance_mismatch(metadata, provenance)
        print(
            f"    {chunk_label} single-point cache provenance mismatch, "
            f"recomputing: {mismatch}",
            flush=True,
        )

    from nvalchemi.neighbors import compute_neighbors

    payloads = [
        ase_to_atomic_data(
            atoms,
            structure_id=f"oc20dense_sid_{config.sid}_{config.config_id}",
            active_mask=active_mask,
        )
        for config, atoms, active_mask in chunk
    ]
    data_list = [backend._to_atomic_data(payload) for payload in payloads]
    batch = backend.api.Batch.from_data_list(data_list, device=backend.device)
    n_atoms = sum(len(atoms) for _config, atoms, _active_mask in chunk)
    _reset_cuda_peak_memory(backend.device)
    before_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"    {chunk_label} single-point batch: {len(chunk)} configs, "
        f"{n_atoms} atoms; {_memory_message(before_memory)}",
        flush=True,
    )

    start = time.perf_counter()
    compute_neighbors(batch, config=backend.model.model_config.neighbor_config)
    outputs = backend.model(batch)
    runtime_s = time.perf_counter() - start
    after_memory = _cuda_memory_snapshot(backend.device)
    print(
        f"    {chunk_label} single-point done: {runtime_s:.2f} s; "
        f"{_memory_message(after_memory)}",
        flush=True,
    )

    energies = _model_tensor(outputs, "energy").detach().cpu().numpy().reshape(-1)
    forces = _model_tensor(outputs, "forces").detach().cpu().numpy().reshape(-1, 3)
    batch_ptr = getattr(batch, "batch_ptr", None)
    if batch_ptr is None:
        offsets = np.cumsum([0, *[len(atoms) for _config, atoms, _mask in chunk]])
    else:
        offsets = batch_ptr.detach().cpu().numpy().astype(int)

    rows: list[dict[str, Any]] = []
    for index, (config, _atoms, active_mask) in enumerate(chunk):
        force_block = forces[offsets[index]: offsets[index + 1]]
        label = f"{config.system_id}_{config.config_id}_sid{config.sid}"
        sp_structure_path = paths["initial_single_point"] / f"{_safe(label)}.extxyz"
        sp_log_path = paths["initial_single_point_logs"] / f"{_safe(label)}.csv"
        sp_atoms = _atoms.copy()
        free_fmax = _max_force(force_block.flatten().tolist(), active_mask=active_mask)
        all_fmax = _max_force(force_block.flatten().tolist())
        sp_atoms.info["structure_id"] = label
        sp_atoms.info["mace_total_energy_eV"] = float(energies[index])
        sp_atoms.info["mace_free_fmax_eV_A"] = free_fmax
        sp_atoms.info["mace_all_atom_fmax_eV_A"] = all_fmax
        sp_atoms.arrays["forces"] = force_block
        ase_write(sp_structure_path, sp_atoms, format="extxyz")
        sp_log_path.write_text(
            "\n".join(
                [
                    "step,structure_id,energy_eV,max_force_eV_A,free_max_force_eV_A",
                    f"0,{label},{float(energies[index])},{all_fmax},{free_fmax}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "sid": int(config.sid),
                "system_id": str(config.system_id),
                "config_id": str(config.config_id),
                "ml_initial_sp_total_energy_eV": float(energies[index]),
                "ml_initial_sp_free_fmax_eV_A": free_fmax,
                "ml_initial_sp_all_atom_fmax_eV_A": all_fmax,
                "ml_initial_sp_structure_path": str(sp_structure_path),
                "ml_initial_sp_log_path": str(sp_log_path),
                **provenance,
            }
        )

    sp_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata = {
        "chunk_label": chunk_label,
        "n_configs": len(chunk),
        "n_atoms": n_atoms,
        "runtime_s": runtime_s,
        "initial_single_point_structures": str(paths["initial_single_point"]),
        "initial_single_point_logs": str(paths["initial_single_point_logs"]),
        **provenance,
        **_memory_fields("single_point_cuda_after", after_memory),
    }
    sp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows, runtime_s, metadata


def _formula_from_tags(atoms: Atoms) -> tuple[str, str]:
    tags = atoms.get_tags()
    symbols = atoms.get_chemical_symbols()
    slab = Counter(symbol for symbol, tag in zip(symbols, tags) if int(tag) != 2)
    ads = Counter(symbol for symbol, tag in zip(symbols, tags) if int(tag) == 2)

    def fmt(counter: Counter[str]) -> str:
        return "".join(f"{symbol}{counter[symbol]}" for symbol in sorted(counter))

    return fmt(slab), fmt(ads)


def _write_summary_tables(
    *,
    rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_config = pd.DataFrame(rows)
    if per_config.empty:
        raise RuntimeError("No OC20Dense rows were generated.")
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)
    for key, value in provenance.items():
        if key not in per_config:
            per_config[key] = value

    chunks = pd.DataFrame(chunk_rows)
    chunk_runtime_by_system: dict[str, float] = {}
    if not chunks.empty and "runtime_s" in chunks:
        chunk_runtime_by_system = (
            chunks.groupby("system_id")["runtime_s"].sum().astype(float).to_dict()
        )
    sp_runtime_by_system: dict[str, float] = {}
    if not chunks.empty and "single_point_runtime_s" in chunks:
        sp_runtime_by_system = (
            chunks.groupby("system_id")["single_point_runtime_s"]
            .sum()
            .astype(float)
            .to_dict()
        )

    per_config["ml_relaxed_rank"] = (
        per_config.groupby("system_id")["ml_total_energy_eV"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    per_config["ml_rank"] = per_config["ml_relaxed_rank"]
    per_config["ml_initial_sp_rank"] = (
        per_config.groupby("system_id")["ml_initial_sp_total_energy_eV"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    per_config["dft_gap_to_best_eV"] = (
        per_config["dft_adsorption_energy_eV"]
        - per_config.groupby("system_id")["dft_adsorption_energy_eV"].transform("min")
    )

    summary_rows: list[dict[str, Any]] = []
    for system_id, group in per_config.groupby("system_id", sort=False):
        dft_best = group.loc[group["dft_adsorption_energy_eV"].idxmin()]
        sp_best = group.loc[group["ml_initial_sp_total_energy_eV"].idxmin()]
        ml_best = group.loc[group["ml_total_energy_eV"].idxmin()]
        sp_top3 = group.nsmallest(min(3, len(group)), "ml_initial_sp_total_energy_eV")
        sp_top5 = group.nsmallest(min(5, len(group)), "ml_initial_sp_total_energy_eV")
        top3 = group.nsmallest(min(3, len(group)), "ml_total_energy_eV")
        top5 = group.nsmallest(min(5, len(group)), "ml_total_energy_eV")
        dft_ranks = group["dft_rank"]
        sp_spearman = (
            float(dft_ranks.corr(group["ml_initial_sp_rank"]))
            if len(group) > 1
            else float("nan")
        )
        relaxed_spearman = (
            float(dft_ranks.corr(group["ml_relaxed_rank"]))
            if len(group) > 1
            else float("nan")
        )
        summary_rows.append(
            {
                "system_id": system_id,
                "adsorbate": str(group.iloc[0]["adsorbate"]),
                "adsorbate_reference_species": str(
                    group.iloc[0]["adsorbate_reference_species"]
                ),
                "mpid": str(group.iloc[0]["mpid"]),
                "miller_idx": str(group.iloc[0]["miller_idx"]),
                "slab_formula": str(group.iloc[0]["slab_formula"]),
                "n_configs": int(len(group)),
                "dft_best_config": str(dft_best["config_id"]),
                "dft_best_sid": int(dft_best["sid"]),
                "dft_best_energy_eV": float(dft_best["dft_adsorption_energy_eV"]),
                "dft_best_ml_initial_sp_rank": int(dft_best["ml_initial_sp_rank"]),
                "dft_best_ml_relaxed_rank": int(dft_best["ml_relaxed_rank"]),
                "dft_best_ml_rank": int(dft_best["ml_relaxed_rank"]),
                "sp_best_config": str(sp_best["config_id"]),
                "sp_best_sid": int(sp_best["sid"]),
                "sp_best_total_energy_eV": float(
                    sp_best["ml_initial_sp_total_energy_eV"]
                ),
                "sp_best_dft_rank": int(sp_best["dft_rank"]),
                "sp_best_dft_energy_eV": float(sp_best["dft_adsorption_energy_eV"]),
                "sp_best_dft_gap_to_best_eV": float(sp_best["dft_gap_to_best_eV"]),
                "sp_top1_success_0p10eV": bool(sp_best["dft_gap_to_best_eV"] <= 0.1),
                "sp_top3_best_dft_gap_eV": float(sp_top3["dft_gap_to_best_eV"].min()),
                "sp_top3_success_0p10eV": bool(
                    sp_top3["dft_gap_to_best_eV"].min() <= 0.1
                ),
                "sp_top5_best_dft_gap_eV": float(sp_top5["dft_gap_to_best_eV"].min()),
                "sp_top5_success_0p10eV": bool(
                    sp_top5["dft_gap_to_best_eV"].min() <= 0.1
                ),
                "sp_spearman_rank_corr": sp_spearman,
                "ml_best_config": str(ml_best["config_id"]),
                "ml_best_sid": int(ml_best["sid"]),
                "ml_best_total_energy_eV": float(ml_best["ml_total_energy_eV"]),
                "ml_best_dft_rank": int(ml_best["dft_rank"]),
                "ml_best_dft_energy_eV": float(ml_best["dft_adsorption_energy_eV"]),
                "ml_best_dft_gap_to_best_eV": float(ml_best["dft_gap_to_best_eV"]),
                "ml_top1_success_0p10eV": bool(ml_best["dft_gap_to_best_eV"] <= 0.1),
                "ml_top3_best_dft_gap_eV": float(top3["dft_gap_to_best_eV"].min()),
                "ml_top3_success_0p10eV": bool(top3["dft_gap_to_best_eV"].min() <= 0.1),
                "ml_top5_best_dft_gap_eV": float(top5["dft_gap_to_best_eV"].min()),
                "ml_top5_success_0p10eV": bool(top5["dft_gap_to_best_eV"].min() <= 0.1),
                "relaxed_spearman_rank_corr": relaxed_spearman,
                "spearman_rank_corr": relaxed_spearman,
                "n_converged_backend": int(group["backend_converged"].sum()),
                "n_converged_free_fmax": int(group["free_fmax_converged"].sum()),
                "max_optimizer_steps": int(group["optimizer_nsteps"].max()),
                "mace_rank_basis": MACE_RANK_BASIS,
                "mace_eads_reference_status": MACE_EADS_REFERENCE_STATUS,
                "single_point_chunk_runtime_s": float(
                    sp_runtime_by_system.get(system_id, 0.0)
                ),
                "total_chunk_runtime_s": float(
                    chunk_runtime_by_system.get(system_id, 0.0)
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    per_config.to_csv(paths["tables"] / "per_config_results.csv", index=False)
    summary.to_csv(paths["tables"] / "system_summary.csv", index=False)
    chunks.to_csv(paths["tables"] / "chunk_timings.csv", index=False)
    return per_config, summary, chunks


def _metadata(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    data_root = args.data_root
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _command_output(["git", "status", "--short"]),
        "python": sys.version,
        "package_versions": _package_versions(),
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ),
        "official_sources": OFFICIAL_SOURCES,
        "source_checksums": {
            "oc20_dense_data.tar.gz": _file_md5(oc20dense_archive_file(data_root, "oc20_dense_data.tar.gz")),
            "oc20_dense_mappings.tar.gz": _file_md5(oc20dense_archive_file(data_root, "oc20_dense_mappings.tar.gz")),
            "oc20dense_mapping.pkl": _file_md5(oc20dense_mapping_file(data_root, "oc20dense_mapping.pkl")),
            "oc20dense_targets.pkl": _file_md5(oc20dense_mapping_file(data_root, "oc20dense_targets.pkl")),
            "oc20dense_compute.pkl": _file_md5(oc20dense_mapping_file(data_root, "oc20dense_compute.pkl")),
            "oc20dense_ref_energies.pkl": _file_md5(oc20dense_mapping_file(data_root, "oc20dense_ref_energies.pkl")),
            "oc20dense_tags.pkl": _file_md5(oc20dense_mapping_file(data_root, "oc20dense_tags.pkl")),
        },
        "default_closed_shell_systems": list(DEFAULT_CLOSED_SHELL_SYSTEMS),
        "closed_shell_adsorbate_references": CLOSED_SHELL_ADSORBATE_REFERENCES,
        "allow_unpinned_adsorbates": bool(args.allow_unpinned_adsorbates),
        "systems": list(args.systems),
        "config_ids": list(args.config_ids) if args.config_ids is not None else None,
        "selection_csv": str(args.selection_csv) if args.selection_csv else None,
        "max_configs_per_system": args.max_configs_per_system,
        "chunk_size": args.chunk_size,
        "toolkit_checkpoint": os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
        "toolkit_head": os.environ.get("TOOLKIT_HEAD") or None,
        "toolkit_device": os.environ.get("TOOLKIT_DEVICE", "cuda"),
        "toolkit_dtype": os.environ.get("TOOLKIT_DTYPE", "float32"),
        "toolkit_n_steps": args.n_steps,
        "toolkit_fmax": args.fmax,
        "toolkit_d3bj_enabled": False,
        "write_trajectories": not args.no_trajectories,
        "output_dirs": {key: str(value) for key, value in paths.items()},
    }


def _write_report(
    *,
    path: Path,
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    chunks: pd.DataFrame,
) -> None:
    sp_cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "mpid",
        "miller_idx",
        "n_configs",
        "dft_best_config",
        "dft_best_energy_eV",
        "sp_best_config",
        "sp_best_dft_rank",
        "sp_best_dft_gap_to_best_eV",
        "sp_top1_success_0p10eV",
        "sp_top3_best_dft_gap_eV",
        "sp_top3_success_0p10eV",
        "sp_spearman_rank_corr",
    ]
    relaxed_cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "mpid",
        "miller_idx",
        "n_configs",
        "dft_best_config",
        "dft_best_energy_eV",
        "ml_best_config",
        "ml_best_dft_rank",
        "ml_best_dft_gap_to_best_eV",
        "ml_top1_success_0p10eV",
        "ml_top3_best_dft_gap_eV",
        "ml_top3_success_0p10eV",
        "relaxed_spearman_rank_corr",
    ]
    sp_runtime_s = (
        float(chunks["single_point_runtime_s"].sum())
        if "single_point_runtime_s" in chunks
        else 0.0
    )
    relax_runtime_s = float(chunks["runtime_s"].sum()) if "runtime_s" in chunks else 0.0
    model_label = toolkit_model_label(metadata)
    lines = [
        "# OC20Dense Known-Example Toolkit Check",
        "",
        f"Generated: {metadata['generated_utc']}",
        "Backend: ALCHEMI Toolkit on official OC20Dense inputs",
        f"Model: {model_label}; D3(BJ) disabled",
        "",
        "## What this checks",
        "",
        (
            "This run uses official OC20Dense initial structures and released DFT "
            "adsorption-energy targets. Each row is keyed by the official "
            "`system_id`, `config_id`, and `sid`: the starting geometry is read "
            "from the OC20Dense LMDB, the DFT adsorption energy is read from "
            "`oc20dense_targets.pkl`, and the metadata is read from "
            "`oc20dense_mapping.pkl`."
        ),
        "",
        (
            "For each selected system, the selected Toolkit MACE checkpoint "
            "first scores every input configuration at the exact OC20Dense "
            "starting coordinates, then relaxes the same configurations in "
            "Toolkit batches and ranks the relaxed structures by final Toolkit "
            "total energy."
        ),
        "",
        (
            "The default selected systems are closed-shell adsorbate examples: "
            "`*OH2` referenced to H2O, `*NH3` referenced to NH3, and `*N2` "
            "referenced to N2. CH3-containing systems are out of scope for this "
            "closed-shell benchmark slice."
        ),
        "",
        (
            "The DFT value reported beside each selected configuration is the "
            "released OC20Dense adsorption energy for that configuration. A "
            "separate trajectory-reference check validates the DFT target by "
            "reading the official DFT trajectory final frame and "
            "verifying `final DFT total energy - oc20dense_ref_energies[system_id]` "
            "against the released target."
        ),
        "",
        (
            "Energy columns in this report have different meanings: "
            "`dft_*_energy_eV` values are released DFT adsorption energies; "
            "`sp_best_total_energy_eV` and `ml_best_total_energy_eV` are "
            "selected-checkpoint MACE total energies. Within one system the "
            "compared structures have the same composition, so those ML total "
            "energies can rank candidate geometries. Across different systems, "
            "use adsorption energies or another consistent reference."
        ),
        "",
        (
            "MACE adsorption energies are intentionally not reported by this "
            "script yet. The gas-reference species are now pinned for the "
            "closed-shell set, but a valid MACE adsorption-energy comparison "
            "still requires MACE-scale clean-slab references generated with the "
            "same model settings. The DFT `oc20dense_ref_energies.pkl` values "
            "must not be subtracted from MACE totals."
        ),
        "",
        "## Single-Point Screening on Exact Input Coordinates",
        "",
        summary[sp_cols].to_markdown(index=False),
        "",
        "## Relaxed-Structure Screening",
        "",
        summary[relaxed_cols].to_markdown(index=False),
        "",
        "## Runtime",
        "",
        f"- Host: `{metadata['hostname']}`.",
        f"- GPU: `{metadata['gpu']}`.",
        f"- Toolkit checkpoint: `{metadata['toolkit_checkpoint']}`.",
        f"- Toolkit n_steps cap: {metadata['toolkit_n_steps']}.",
        f"- Chunk size: {metadata['chunk_size']}.",
        f"- Structures single-point scored: {int(summary['n_configs'].sum())}.",
        f"- Structures relaxed: {int(summary['n_configs'].sum())}.",
        f"- Single-point chunk runtime total: {sp_runtime_s / 60.0:.2f} min.",
        f"- Relaxation chunk runtime total: {relax_runtime_s / 60.0:.2f} min.",
        "",
        "## Output Files",
        "",
        f"- Per-config table: `{metadata['output_dirs']['tables']}/per_config_results.csv`",
        f"- System summary: `{metadata['output_dirs']['tables']}/system_summary.csv`",
        f"- Chunk timings: `{metadata['output_dirs']['tables']}/chunk_timings.csv`",
        f"- Single-point chunk JSON: `{metadata['output_dirs']['single_point']}`",
        f"- Initial structures: `{metadata['output_dirs']['initial']}`",
        f"- Relaxed structures: `{metadata['output_dirs']['relaxed']}`",
        f"- Toolkit relaxation trajectories: `{metadata['output_dirs']['trajectories']}`",
        f"- Toolkit energy/force logs: `{metadata['output_dirs']['trajectory_logs']}`",
        f"- Raw optimization JSON: `{metadata['output_dirs']['raw']}`",
        f"- Metadata: `{metadata['output_dirs']['reports']}/run_metadata.json`",
        "",
        "## Source Anchors",
        "",
        f"- OC20Dense docs: {OFFICIAL_SOURCES['oc20dense_docs']}",
        f"- AdsorbML repository: {OFFICIAL_SOURCES['adsorbml_readme']}",
        f"- Mapping archive MD5: {metadata['source_checksums']['oc20_dense_mappings.tar.gz']}",
        f"- LMDB archive MD5: {metadata['source_checksums']['oc20_dense_data.tar.gz']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oc20dense_known_examples(args: argparse.Namespace) -> int:
    write_trajectory = not args.no_trajectories
    _install_pyg_pickle_shim()
    require_precomputed_write_allowed(args.outdir)
    paths = _ensure_dirs(args.outdir)

    mapping: dict[int, dict[str, Any]] = _read_pickle(
        oc20dense_mapping_file(args.data_root, "oc20dense_mapping.pkl")
    )
    targets: dict[str, list[tuple[str, float]]] = _read_pickle(
        oc20dense_mapping_file(args.data_root, "oc20dense_targets.pkl")
    )
    mapping_by_system: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for sid, meta in mapping.items():
        mapping_by_system[str(meta["system_id"])].append((int(sid), meta))

    if args.selection_csv is not None:
        selection_by_system = _read_selection_csv(args.selection_csv)
        selected_by_system = _select_from_exact_rows(
            selection_by_system=selection_by_system,
            mapping_by_system=mapping_by_system,
            targets=targets,
        )
        args.systems = list(selected_by_system)
        args.max_configs_per_system = 0
        args.config_ids = None
    else:
        selected_by_system = {
            system_id: _select_system_configs(
                system_id=system_id,
                mapping_by_system=mapping_by_system,
                targets=targets,
                max_configs=args.max_configs_per_system,
                config_ids={str(config_id) for config_id in args.config_ids}
                if args.config_ids is not None
                else None,
            )
            for system_id in args.systems
        }
    _validate_closed_shell_reference_set(
        selected_by_system,
        allow_unpinned_adsorbates=bool(args.allow_unpinned_adsorbates),
    )

    backend = _build_backend(args)
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)
    lmdb_path = oc20dense_lmdb_path(args.data_root)
    env = _open_lmdb(args.data_root) if lmdb_path.exists() else None
    if env is None:
        print(
            "OC20Dense LMDB not found; using slim selected initial extxyz structures. "
            + FULL_DATA_NOTICE,
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    if args.selection_csv is not None:
        run_groups = [
            (
                "exact_selection",
                [
                    config
                    for selected in selected_by_system.values()
                    for config in selected
                ],
            )
        ]
    else:
        run_groups = list(selected_by_system.items())

    txn_context = env.begin() if env is not None else nullcontext(None)
    with txn_context as txn:
        for group_label, selected in run_groups:
            group_systems = sorted({config.system_id for config in selected})
            print(
                f"Selection {group_label}: {len(selected)} configs across "
                f"{', '.join(group_systems)}",
                flush=True,
            )
            loaded: list[tuple[SelectedConfig, Atoms, list[bool]]] = []
            for config in selected:
                label, atoms, active_mask = _atoms_for_selected_config(
                    config,
                    txn=txn,
                    initial_structure_dir=args.initial_structure_dir,
                )
                slab_formula, ads_formula = _formula_from_tags(atoms)
                atoms.info.update(
                    {
                        "system_id": config.system_id,
                        "config_id": config.config_id,
                        "mpid": config.mpid,
                        "miller_idx": str(config.miller_idx),
                        "adsorbate": config.adsorbate,
                        "dft_adsorption_energy_eV": config.dft_adsorption_energy_eV,
                        "dft_rank": config.dft_rank,
                    }
                )
                initial_path = paths["initial"] / f"{_safe(label)}.extxyz"
                if args.force or not initial_path.exists():
                    ase_write(initial_path, atoms, format="extxyz")
                loaded.append((config, atoms, active_mask))

            for start in range(0, len(loaded), args.chunk_size):
                chunk = loaded[start:start + args.chunk_size]
                chunk_index = start // args.chunk_size + 1
                chunk_label = f"oc20dense_{_safe(group_label)}_chunk_{chunk_index:02d}"
                print(f"  {chunk_label}: {len(chunk)} configs", flush=True)
                sp_rows, sp_runtime_s, sp_metadata = _single_point_chunk(
                    backend=backend,
                    chunk=chunk,
                    chunk_label=chunk_label,
                    paths=paths,
                    force=args.force,
                    provenance=provenance,
                )
                sp_by_sid = {int(row["sid"]): row for row in sp_rows}
                results, runtime_s, relax_metadata = _relax_chunk(
                    backend=backend,
                    chunk=chunk,
                    chunk_label=chunk_label,
                    paths=paths,
                    force=args.force,
                    write_trajectory=write_trajectory,
                    provenance=provenance,
                )
                chunk_rows.append(
                    {
                        "system_id": group_label,
                        "chunk_systems": ",".join(group_systems),
                        "chunk_label": chunk_label,
                        "n_configs": len(chunk),
                        "single_point_runtime_s": sp_runtime_s,
                        "runtime_s": runtime_s,
                        "max_optimizer_nsteps": max(int(r.optimizer_nsteps) for r in results),
                        "n_backend_converged": sum(bool(r.converged) for r in results),
                        "trajectory_dir": str(paths["trajectories"]),
                        "trajectory_log_dir": str(paths["trajectory_logs"]),
                        "n_atoms": int(relax_metadata.get("n_atoms", 0)),
                        **{
                            key: value
                            for key, value in sp_metadata.items()
                            if key.startswith("single_point_cuda_after_")
                        },
                        **{
                            key: value
                            for key, value in relax_metadata.items()
                            if key.startswith("relax_cuda_after_")
                        },
                        **{
                            key: relax_metadata.get(
                                key,
                                sp_metadata.get(key, provenance.get(key)),
                            )
                            for key in TOOLKIT_PROVENANCE_COLUMNS
                        },
                    }
                )

                for (config, atoms, active_mask), result in zip(chunk, results):
                    label = f"{config.system_id}_{config.config_id}_sid{config.sid}"
                    raw_path = paths["raw"] / f"{_safe(label)}.json"
                    relaxed_path = paths["relaxed"] / f"{_safe(label)}.extxyz"
                    trajectory_path = paths["trajectories"] / f"{_safe(label)}.extxyz"
                    trajectory_log_path = paths["trajectory_logs"] / f"{_safe(label)}.csv"
                    _write_result(raw_path, result)
                    final_atoms = atomic_data_to_ase(result)
                    final_atoms.set_tags(atoms.get_tags())
                    final_atoms.info.update(atoms.info)
                    final_atoms.info["ml_total_energy_eV"] = float(result.energy)
                    final_atoms.info["optimizer_nsteps"] = int(result.optimizer_nsteps)
                    final_atoms.arrays["forces"] = np.asarray(
                        result.forces,
                        dtype=float,
                    ).reshape(-1, 3)
                    ase_write(relaxed_path, final_atoms, format="extxyz")

                    slab_formula, ads_formula = _formula_from_tags(atoms)
                    free_fmax = _max_force(result.forces, active_mask=active_mask)
                    sp_row = sp_by_sid[int(config.sid)]
                    rows.append(
                        {
                            "system_id": config.system_id,
                            "sid": config.sid,
                            "config_id": config.config_id,
                            "mpid": config.mpid,
                            "miller_idx": str(config.miller_idx),
                        "top": config.top,
                        "adsorbate": config.adsorbate,
                        "adsorbate_reference_species": _reference_species_for_adsorbate(
                            config.adsorbate
                        ),
                        "slab_formula": slab_formula,
                            "adsorbate_formula": ads_formula,
                            "natoms": len(atoms),
                            "n_active_atoms": int(sum(active_mask)),
                            "dft_adsorption_energy_eV": config.dft_adsorption_energy_eV,
                            "dft_rank": config.dft_rank,
                            "ml_initial_sp_total_energy_eV": sp_row[
                                "ml_initial_sp_total_energy_eV"
                            ],
                            "ml_initial_sp_free_fmax_eV_A": sp_row[
                                "ml_initial_sp_free_fmax_eV_A"
                            ],
                            "ml_initial_sp_all_atom_fmax_eV_A": sp_row[
                                "ml_initial_sp_all_atom_fmax_eV_A"
                            ],
                            "ml_initial_sp_structure_path": sp_row[
                                "ml_initial_sp_structure_path"
                            ],
                            "ml_initial_sp_log_path": sp_row["ml_initial_sp_log_path"],
                            "ml_total_energy_eV": float(result.energy),
                            "optimizer_nsteps": int(result.optimizer_nsteps),
                            "backend_converged": bool(result.converged),
                            "free_fmax_eV_A": free_fmax,
                            "all_atom_fmax_eV_A": _max_force(result.forces),
                            "free_fmax_converged": bool(free_fmax <= args.fmax),
                            "chunk_label": chunk_label,
                            "chunk_runtime_s": runtime_s,
                            "initial_structure": str(paths["initial"] / f"{_safe(label)}.extxyz"),
                            "relaxed_structure": str(relaxed_path),
                            "toolkit_trajectory": str(trajectory_path),
                            "toolkit_trajectory_log": str(trajectory_log_path),
                            "raw_json": str(raw_path),
                            "mace_rank_basis": MACE_RANK_BASIS,
                            "mace_eads_reference_status": MACE_EADS_REFERENCE_STATUS,
                            **provenance,
                        }
                    )
                print(
                    f"    SP {sp_runtime_s:.1f}s, relax {runtime_s:.1f}s; "
                    f"{chunk_rows[-1]['n_backend_converged']}/{len(chunk)} backend-converged",
                    flush=True,
                )

    per_config, summary, chunks = _write_summary_tables(
        rows=rows,
        chunk_rows=chunk_rows,
        paths=paths,
    )
    metadata = _metadata(args, paths)
    metadata.update(
        {
            "n_rows": int(len(per_config)),
            "n_systems": int(len(summary)),
            "summary_csv": str(paths["tables"] / "system_summary.csv"),
            "per_config_csv": str(paths["tables"] / "per_config_results.csv"),
            "chunk_csv": str(paths["tables"] / "chunk_timings.csv"),
            "report_md": str(paths["reports"] / "oc20dense_known_examples_report.md"),
        }
    )
    (paths["reports"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_report(
        path=paths["reports"] / "oc20dense_known_examples_report.md",
        metadata=metadata,
        summary=summary,
        chunks=chunks,
    )
    print(f"Wrote {metadata['summary_csv']}", flush=True)
    print(f"Wrote {metadata['per_config_csv']}", flush=True)
    print(f"Wrote {metadata['report_md']}", flush=True)
    return 0


def main() -> int:
    return run_oc20dense_known_examples(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
