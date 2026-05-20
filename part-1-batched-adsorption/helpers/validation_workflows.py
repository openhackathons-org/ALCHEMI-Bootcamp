"""In-process OC20Dense validation workflows for the tutorial notebook.

The notebook exposes the scientific choices and calls Python functions directly.
The command-line scripts remain available for reproducible batch runs, but the
tutorial notebook should not shell out to them.
"""

from __future__ import annotations

import os
import sys
import time
import io
import importlib
import pickle
import tarfile
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import gcd
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.io import write as ase_write


@dataclass(frozen=True)
class ValidationWorkflowStep:
    """One notebook-visible validation step backed by in-process Python code."""

    label: str
    message: str
    action: Callable[[], int]


@dataclass(frozen=True)
class OC20DenseValidationContext:
    """Notebook-facing settings and paths for the compact OC20Dense checks."""

    tutorial_root: Path
    accuracy_output_dir: Path
    result_source: str
    checkpoint: str
    head: str
    model_label: str
    trajectory_selection: tuple[dict, ...]
    nh3_system: str
    preview_ranks: tuple[int, ...]
    show_all_nh3_ranking: bool
    relax_batch_size: int
    single_point_batch_size: int
    n_steps: int
    fmax: float
    oc20dense_data_root: Path
    trajectory_root: Path
    nh3_ranking_root: Path
    nh3_reference_source_root: Path

    @property
    def compute_live(self) -> bool:
        return self.result_source == "compute"


def make_oc20dense_validation_context(
    *,
    tutorial_root: Path,
    accuracy_output_dir: str | Path,
    result_source: str,
    checkpoint: str,
    head: str,
    trajectory_selection: Sequence[dict],
    nh3_system: str,
    preview_ranks: Sequence[int],
    show_all_nh3_ranking: bool,
    relax_batch_size: int | None = None,
    single_point_batch_size: int = 12,
    n_steps: int = 200,
    fmax: float = 0.05,
) -> OC20DenseValidationContext:
    """Create the compact validation context from notebook-visible choices."""
    if result_source not in {"compute", "saved"}:
        raise ValueError("result_source must be 'compute' or 'saved'.")

    tutorial_root = Path(tutorial_root).resolve()
    accuracy_output_dir = Path(accuracy_output_dir)
    if not accuracy_output_dir.is_absolute():
        accuracy_output_dir = tutorial_root / accuracy_output_dir

    model_label = f"{checkpoint} (head={head})" if head else checkpoint
    trajectory_root = (
        accuracy_output_dir
        / "oc20dense_closed_shell_trajectory_mace_mpa0"
    )
    nh3_ranking_root = (
        accuracy_output_dir
        / "oc20dense_nh3_92_fixed_geometry_mace_mpa0"
    )

    # Keep the validation Toolkit path explicit and local to this section.
    os.environ["TOOLKIT_CHECKPOINT"] = checkpoint
    if head:
        os.environ["TOOLKIT_HEAD"] = head
    else:
        os.environ.pop("TOOLKIT_HEAD", None)

    selection = tuple(dict(row) for row in trajectory_selection)
    return OC20DenseValidationContext(
        tutorial_root=tutorial_root,
        accuracy_output_dir=accuracy_output_dir,
        result_source=result_source,
        checkpoint=checkpoint,
        head=head,
        model_label=model_label,
        trajectory_selection=selection,
        nh3_system=str(nh3_system),
        preview_ranks=tuple(int(rank) for rank in preview_ranks),
        show_all_nh3_ranking=bool(show_all_nh3_ranking),
        relax_batch_size=int(relax_batch_size or len(selection)),
        single_point_batch_size=int(single_point_batch_size),
        n_steps=int(n_steps),
        fmax=float(fmax),
        oc20dense_data_root=_reference_data_root(tutorial_root),
        trajectory_root=trajectory_root,
        nh3_ranking_root=nh3_ranking_root,
        nh3_reference_source_root=nh3_ranking_root / "reference_source",
    )


def validation_context_table(
    context: OC20DenseValidationContext,
    *,
    relpath_fn: Callable[[Path], str] | None = None,
) -> pd.DataFrame:
    """Return a compact table of the validation choices readers can change."""
    relpath = relpath_fn or (lambda path: Path(path).as_posix())
    adsorbates = ", ".join(row["adsorbate"] for row in context.trajectory_selection)
    nh3_surface = describe_oc20dense_system(context, context.nh3_system)
    return pd.DataFrame(
        [
            {"choice": "result source", "value": context.result_source},
            {"choice": "model", "value": context.model_label},
            {
                "choice": "relaxation check",
                "value": (
                    f"{len(context.trajectory_selection)} OC20Dense starts "
                    f"({adsorbates}), Toolkit batch size {context.relax_batch_size}"
                ),
            },
            {
                "choice": "NH3 ranking",
                "value": (
                    f"{nh3_surface['surface_label']}, 92 DFT-relaxed geometries, "
                    f"single-point batch size {context.single_point_batch_size}"
                ),
            },
            {
                "choice": "relaxation stop",
                "value": f"{context.n_steps} step cap, fmax <= {context.fmax} eV/A",
            },
            {
                "choice": "reference data",
                "value": relpath(context.oc20dense_data_root),
            },
            {
                "choice": "trajectory output",
                "value": relpath(context.trajectory_root),
            },
            {
                "choice": "NH3 ranking output",
                "value": relpath(context.nh3_ranking_root),
            },
        ]
    )


def describe_oc20dense_system(
    context: OC20DenseValidationContext,
    system_id: str,
) -> dict[str, object]:
    """Return the notebook-facing OC20Dense surface label for one system."""
    data_root = ensure_oc20dense_reference_data(context.tutorial_root)
    mapping = _read_pickle(_mapping_file(data_root, "oc20dense_mapping.pkl"))
    matches = [
        (sid, meta)
        for sid, meta in mapping.items()
        if str(meta.get("system_id")) == str(system_id)
    ]
    if not matches:
        return {
            "system_id": str(system_id),
            "surface_label": f"OC20Dense system {system_id}",
            "mpid": "",
            "miller_idx": "",
            "side": "",
            "slab_formula": "",
        }

    sid, meta = sorted(matches, key=lambda item: str(item[1].get("config_id", "")))[0]
    miller = tuple(meta.get("miller_idx", ()))
    miller_label = "".join(str(index) for index in miller) if miller else "surface"
    side = "top" if bool(meta.get("top", False)) else "bottom"
    slab_formula = _slab_formula_for_system(data_root, system_id=str(system_id), sid=sid, meta=meta)
    material_label = slab_formula or str(meta.get("mpid", "OC20Dense slab"))
    material_label_html = _formula_html_subscripts(material_label) if slab_formula else material_label
    surface_label_plain = (
        f"{material_label}({miller_label})"
        f" [{meta.get('mpid', '')}, system {system_id}, {side} side]"
    )
    surface_label = (
        f"{material_label_html}({miller_label})"
        f" [{meta.get('mpid', '')}, system {system_id}, {side} side]"
    )
    return {
        "system_id": str(system_id),
        "surface_label": surface_label,
        "surface_label_plain": surface_label_plain,
        "mpid": str(meta.get("mpid", "")),
        "miller_idx": str(miller),
        "side": side,
        "slab_formula": slab_formula,
    }


def _slab_formula_for_system(
    data_root: Path,
    *,
    system_id: str,
    sid: object,
    meta: dict,
) -> str:
    config_id = str(meta.get("config_id", ""))
    initial_dir = data_root / "initial_structures" / "adslab"
    candidates = [
        data_root
        / "initial_structures"
        / "adslab"
        / f"{system_id}_{config_id}_sid{int(sid)}.extxyz"
    ]
    candidates.extend(sorted(initial_dir.glob(f"{system_id}_*_sid*.extxyz")))
    candidates.append(
        data_root
        / "selected_trajectories"
        / "surfaces"
        / f"{system_id}_surface.traj"
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return ""
    atoms = ase_read(path)
    adsorbate_mask = _adsorbate_atom_mask(atoms)
    symbols = [
        symbol
        for symbol, is_adsorbate in zip(atoms.get_chemical_symbols(), adsorbate_mask)
        if not is_adsorbate
    ]
    return _reduced_formula(symbols)


def _reduced_formula(symbols: Sequence[str]) -> str:
    counts = Counter(str(symbol) for symbol in symbols if str(symbol))
    if not counts:
        return ""
    divisor = 0
    for count in counts.values():
        divisor = count if divisor == 0 else gcd(divisor, count)
    divisor = max(divisor, 1)
    parts = []
    for symbol in sorted(counts):
        count = counts[symbol] // divisor
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _formula_html_subscripts(formula: str) -> str:
    import re

    def replace(match):
        element, count = match.groups()
        return f"{element}<sub>{count}</sub>" if count else element

    return re.sub(r"([A-Z][a-z]?)([0-9]*)", replace, str(formula))


def find_oc20dense_source_root(tutorial_root: Path) -> Path:
    """Return the first local OC20Dense result root with a per-config table."""
    candidates = [
        tutorial_root
        / "outputs"
        / "precomputed"
        / "accuracy"
        / "oc20dense_closed_shell_trajectory_mace_mpa0",
        tutorial_root / "outputs" / "oc20dense_known_examples",
    ]
    return next(
        (
            path for path in candidates
            if (path / "tables" / "per_config_results.csv").exists()
        ),
        candidates[0],
    )


def write_exact_selection_csv(path: Path, rows: Sequence[dict]) -> Path:
    """Write the exact OC20Dense rows used for the small replay benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[
        ["system_id", "config_id", "sid", "adsorbate", "dft_rank"]
    ].to_csv(path, index=False)
    return path


def _prepare_script_imports(tutorial_root: Path) -> None:
    """Make tutorial scripts importable without triggering a kernel re-exec."""
    scripts_dir = tutorial_root / "scripts"
    for path in (tutorial_root, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # The CLI script can re-exec itself to adjust LD_LIBRARY_PATH before Toolkit
    # imports. A notebook kernel should never re-exec; the setup cell already
    # loads the CUDA runtime libraries explicitly.
    cu13_lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
        / "lib"
    )
    if cu13_lib.exists():
        ld_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
        if str(cu13_lib) not in ld_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join([str(cu13_lib), *ld_paths])
    os.environ.setdefault("OC20DENSE_LD_REEXEC", "1")


def _fresh_import(module_name: str):
    """Import or reload a validation script after notebook-side file edits."""
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _reference_data_root(tutorial_root: Path) -> Path:
    return tutorial_root / "data" / "reference" / "oc20dense"


def _validation_pack_path(tutorial_root: Path) -> Path:
    return tutorial_root / "data" / "reference" / "oc20dense-validation-pack.tgz"


def ensure_oc20dense_reference_data(tutorial_root: Path) -> Path:
    """Return the OC20Dense reference folder, unpacking the bundled pack if needed."""
    root = _reference_data_root(tutorial_root)
    required = [
        root / "mappings" / "oc20dense_mapping.pkl",
        root / "mappings" / "oc20dense_targets.pkl",
        root / "mappings" / "oc20dense_ref_energies.pkl",
        root / "selected_trajectories" / "adslab",
    ]
    if all(path.exists() for path in required):
        return root

    pack = _validation_pack_path(tutorial_root)
    if not pack.exists():
        raise FileNotFoundError(
            "OC20Dense validation data is missing. Expected either the expanded "
            f"reference folder at `{root}` or the bundled validation pack at `{pack}`."
        )

    target_dir = root.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with tarfile.open(pack, mode="r:gz") as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            try:
                destination.relative_to(target_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsafe path in OC20Dense validation pack: {member.name}"
                ) from exc
        archive.extractall(target_dir)

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "OC20Dense validation pack was unpacked, but required files are still "
            "missing: " + ", ".join(str(path) for path in missing)
        )
    return root


def _read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _safe_label(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("*", "star")
    )


def _mapping_file(data_root: Path, name: str) -> Path:
    for candidate in (data_root / "mappings" / name, data_root / name):
        if candidate.exists():
            return candidate
    return data_root / "mappings" / name


def prepare_nh3_ranking_reference_source(
    *,
    tutorial_root: Path,
    source_root: Path,
    system_id: str,
    force: bool = False,
) -> Path:
    """Build the lightweight Toolkit-root expected by the NH3 ranking scripts.

    The fixed-geometry ranking check does not need a 92-trajectory MACE
    relaxation. It needs a reference index, the DFT starting frames for tags,
    and the DFT final frames used for single-point scoring. The slim OC20Dense
    subset already ships those trajectories, so this function creates the small
    table/structure scaffold locally from released DFT data.
    """

    data_root = ensure_oc20dense_reference_data(tutorial_root)
    per_config_path = source_root / "tables" / "per_config_results.csv"
    if per_config_path.exists() and not force:
        existing = pd.read_csv(per_config_path)
        if len(existing[existing["system_id"].astype(str).eq(str(system_id))]) >= 90:
            return source_root

    mapping = _read_pickle(_mapping_file(data_root, "oc20dense_mapping.pkl"))
    targets = _read_pickle(_mapping_file(data_root, "oc20dense_targets.pkl"))
    target_by_config = {
        str(config_id): float(energy)
        for config_id, energy in targets[str(system_id)]
    }
    best_energy = min(target_by_config.values())
    ranked_configs = {
        config_id: rank
        for rank, (config_id, _energy) in enumerate(
            sorted(target_by_config.items(), key=lambda item: item[1]),
            start=1,
        )
    }

    paths = {
        "tables": source_root / "tables",
        "initial": source_root / "structures" / "initial",
        "relaxed": source_root / "structures" / "relaxed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, meta in sorted(mapping.items(), key=lambda item: str(item[1]["config_id"])):
        if str(meta["system_id"]) != str(system_id):
            continue
        config_id = str(meta["config_id"])
        if config_id not in target_by_config:
            continue
        label = f"{system_id}_{config_id}_sid{int(sid)}"
        trajectory_path = (
            data_root
            / "selected_trajectories"
            / "adslab"
            / f"{label}.traj"
        )
        if not trajectory_path.exists():
            raise FileNotFoundError(
                f"Missing selected OC20Dense DFT trajectory: {trajectory_path}"
            )

        safe = _safe_label(label)
        initial_path = paths["initial"] / f"{safe}.extxyz"
        relaxed_path = paths["relaxed"] / f"{safe}.extxyz"
        if force or not initial_path.exists() or not relaxed_path.exists():
            frames = ase_read(trajectory_path, ":")
            if not frames:
                raise RuntimeError(f"No frames found in {trajectory_path}")
            ase_write(initial_path, frames[0], format="extxyz")
            # For fixed-geometry ranking, the "relaxed" scaffold is the DFT final
            # frame. This lets the DFT reference checker create a consistent index
            # without running a 92-configuration MACE relaxation.
            ase_write(relaxed_path, frames[-1], format="extxyz")

        dft_energy = target_by_config[config_id]
        dft_rank = ranked_configs[config_id]
        rows.append(
            {
                "system_id": str(system_id),
                "sid": int(sid),
                "config_id": config_id,
                "mpid": str(meta.get("mpid", "")),
                "miller_idx": str(tuple(meta.get("miller_idx", ()))),
                "top": bool(meta.get("top", False)),
                "adsorbate": str(meta.get("adsorbate", "*NH3")),
                "adsorbate_reference_species": "NH3",
                "slab_formula": "",
                "adsorbate_formula": "H3N1",
                "natoms": np.nan,
                "n_active_atoms": np.nan,
                "dft_adsorption_energy_eV": dft_energy,
                "dft_rank": int(dft_rank),
                "ml_initial_sp_total_energy_eV": 0.0,
                "ml_total_energy_eV": 0.0,
                "ml_relaxed_rank": int(dft_rank),
                "ml_rank": int(dft_rank),
                "ml_initial_sp_rank": int(dft_rank),
                "dft_gap_to_best_eV": dft_energy - best_energy,
                "initial_structure": str(initial_path),
                "relaxed_structure": str(relaxed_path),
                "mace_rank_basis": "fixed_geometry_reference_scaffold",
                "mace_eads_reference_status": "dft_final_geometry_source",
            }
        )

    if not rows:
        raise RuntimeError(f"No NH3 ranking rows found for system {system_id}")

    pd.DataFrame(rows).sort_values(["dft_rank", "config_id"]).to_csv(
        per_config_path,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "system_id": str(system_id),
                "n_configs": len(rows),
                "dft_best_config": min(rows, key=lambda row: row["dft_rank"])["config_id"],
                "sp_best_config": min(rows, key=lambda row: row["dft_rank"])["config_id"],
                "ml_best_config": min(rows, key=lambda row: row["dft_rank"])["config_id"],
            }
        ]
    ).to_csv(source_root / "tables" / "system_summary.csv", index=False)
    return source_root


def run_validation_step(step: ValidationWorkflowStep) -> float:
    """Run one in-process validation step and return elapsed minutes."""
    print(f"Running: {step.label}")
    print(step.message)
    started = time.perf_counter()
    result = step.action()
    if result not in (None, 0):
        raise RuntimeError(f"{step.label} returned non-zero status: {result}")
    elapsed_minutes = (time.perf_counter() - started) / 60
    print(f"Finished {step.label} in {elapsed_minutes:.2f} min")
    return elapsed_minutes


def run_or_load_oc20dense_validation(
    context: OC20DenseValidationContext,
    *,
    progress_factory=None,
    display_fn=None,
    relpath_fn: Callable[[Path], str] | None = None,
    force: bool | None = None,
) -> pd.DataFrame:
    """Run the compact validation or show which saved roots will be read."""
    relpath = relpath_fn or (lambda path: Path(path).as_posix())
    if not context.compute_live:
        table = pd.DataFrame(
            [
                {
                    "validation check": "relaxation check",
                    "source": relpath(context.trajectory_root),
                    "action": "read saved tables",
                },
                {
                    "validation check": "NH3 fixed-geometry ranking",
                    "source": relpath(context.nh3_ranking_root),
                    "action": "read saved tables",
                },
            ]
        )
        if display_fn is not None:
            display_fn(table.style.hide(axis="index"))
        return table

    recompute = True if force is None else bool(force)
    trajectory_steps = build_trajectory_stage_plan(
        tutorial_root=context.tutorial_root,
        selection=context.trajectory_selection,
        output_root=context.trajectory_root,
        relax_batch_size=context.relax_batch_size,
        n_steps=context.n_steps,
        fmax=context.fmax,
        force=recompute,
        write_selection=True,
    )
    nh3_ranking_steps = build_nh3_ranking_stage_plan(
        tutorial_root=context.tutorial_root,
        source_root=context.nh3_reference_source_root,
        output_root=context.nh3_ranking_root,
        system_id=context.nh3_system,
        single_point_batch_size=context.single_point_batch_size,
        n_steps=context.n_steps,
        fmax=context.fmax,
        force=recompute,
    )
    all_steps = [
        ("relaxation check", step) for step in trajectory_steps
    ] + [
        ("NH3 fixed-geometry ranking", step) for step in nh3_ranking_steps
    ]

    progress = None
    if progress_factory is not None:
        progress = progress_factory(
            title="OC20Dense validation",
            total=len(all_steps),
            unit="steps",
            message="ready to start",
            width_px=760,
        )

    old_cache_overwrite = os.environ.get("ALCHEMI_ALLOW_CACHE_OVERWRITE")
    live_root = context.tutorial_root / "outputs" / "live_runs"
    if _is_relative_to(context.accuracy_output_dir, live_root):
        # Live notebook runs should be rerunnable in the same timestamped folder.
        # Official saved/precomputed roots remain protected by the separate
        # REFRESH_SAVED_RESULTS / ALCHEMI_ALLOW_ARTIFACT_OVERWRITE path.
        os.environ["ALCHEMI_ALLOW_CACHE_OVERWRITE"] = "1"

    log_dir = context.accuracy_output_dir / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "notebook_validation_run.log"

    rows = []
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            for index, (piece, step) in enumerate(all_steps, start=1):
                if progress is not None:
                    progress.update(done=index - 1, message=f"{piece}: {step.label}")

                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        elapsed_minutes = run_validation_step(step)
                finally:
                    log_handle.write(f"\n=== {piece}: {step.label} ===\n")
                    stdout_text = stdout_buffer.getvalue()
                    stderr_text = stderr_buffer.getvalue()
                    if stdout_text:
                        log_handle.write(stdout_text)
                    if stderr_text:
                        log_handle.write("\n--- stderr ---\n")
                        log_handle.write(stderr_text)
                    log_handle.flush()

                rows.append(
                    {
                        "validation check": piece,
                        "timed action": step.label,
                        "elapsed_min": elapsed_minutes,
                        "log": relpath(log_path),
                    }
                )
                if progress is not None:
                    progress.update(done=index, message=f"{piece}: {step.label} complete")
    finally:
        if old_cache_overwrite is None:
            os.environ.pop("ALCHEMI_ALLOW_CACHE_OVERWRITE", None)
        else:
            os.environ["ALCHEMI_ALLOW_CACHE_OVERWRITE"] = old_cache_overwrite

    table = pd.DataFrame(rows)
    if display_fn is not None:
        display_fn(table.style.hide(axis="index"))
    return table


def run_nh3_single_point_batches(
    context: OC20DenseValidationContext,
    *,
    progress_factory=None,
    display_fn=None,
    markdown_cls=None,
    relpath_fn: Callable[[Path], str] | None = None,
    force: bool = True,
) -> pd.DataFrame:
    """Construct and run the NH3 DFT-final single-point batches."""
    relpath = relpath_fn or (lambda path: Path(path).as_posix())
    steps = build_nh3_ranking_stage_plan(
        tutorial_root=context.tutorial_root,
        source_root=context.nh3_reference_source_root,
        output_root=context.nh3_ranking_root,
        system_id=context.nh3_system,
        single_point_batch_size=context.single_point_batch_size,
        n_steps=context.n_steps,
        fmax=context.fmax,
        force=bool(force),
    )

    if display_fn is not None and markdown_cls is not None:
        display_fn(
            markdown_cls(
                "Constructing the NH<sub>3</sub> validation workload from the released "
                "DFT-final frames, then scoring it as Toolkit single-point batches. "
                f"Configured batch size: `{context.single_point_batch_size}`."
            )
        )

    progress = None
    if progress_factory is not None:
        progress = progress_factory(
            title="NH3 single-point batches",
            total=len(steps),
            unit="steps",
            message="ready to construct validation batches",
            width_px=760,
        )

    old_cache_overwrite = os.environ.get("ALCHEMI_ALLOW_CACHE_OVERWRITE")
    live_root = context.tutorial_root / "outputs" / "live_runs"
    if _is_relative_to(context.accuracy_output_dir, live_root):
        os.environ["ALCHEMI_ALLOW_CACHE_OVERWRITE"] = "1"

    rows = []
    try:
        for index, step in enumerate(steps, start=1):
            if progress is not None:
                progress.update(done=index - 1, message=step.label)
            elapsed_minutes = run_validation_step(step)
            rows.append(
                {
                    "timed action": step.label,
                    "elapsed_min": elapsed_minutes,
                }
            )
            if progress is not None:
                progress.update(done=index, message=f"{step.label} complete")
    finally:
        if old_cache_overwrite is None:
            os.environ.pop("ALCHEMI_ALLOW_CACHE_OVERWRITE", None)
        else:
            os.environ["ALCHEMI_ALLOW_CACHE_OVERWRITE"] = old_cache_overwrite

    chunk_timing_path = (
        context.nh3_ranking_root
        / "dft_final_single_points"
        / "tables"
        / "dft_final_sp_chunk_timings.csv"
    )
    if chunk_timing_path.exists():
        chunk_timings = pd.read_csv(chunk_timing_path)
        total_runtime_s = float(chunk_timings["runtime_s"].astype(float).sum())
        n_structures = int(chunk_timings["n_configs"].astype(int).sum())
        n_batches = int(len(chunk_timings))
        occupancy = ", ".join(str(value) for value in chunk_timings["n_configs"].astype(int))
        if display_fn is not None and markdown_cls is not None:
            display_fn(
                markdown_cls(
                    f"Ran `{n_structures}` NH<sub>3</sub> DFT-final single-point "
                    f"calculations as `{n_batches}` Toolkit batches "
                    f"(`{occupancy}` structures per batch). Total single-point "
                    f"batch runtime: `{total_runtime_s:.3f} s`. Chunk timings: "
                    f"`{relpath(chunk_timing_path)}`."
                )
            )

    return pd.DataFrame(rows)


def build_trajectory_stage_plan(
    *,
    tutorial_root: Path,
    selection: Sequence[dict],
    output_root: Path,
    relax_batch_size: int,
    n_steps: int = 200,
    fmax: float = 0.05,
    force: bool = False,
    write_selection: bool = True,
) -> list[ValidationWorkflowStep]:
    """Build the closed-shell trajectory-replay workflow from visible settings."""
    _prepare_script_imports(tutorial_root)
    dft_checks = _fresh_import("oc20dense_dft_reference_checks")
    known_examples = _fresh_import("run_oc20dense_known_examples")
    mace_eads = _fresh_import("run_oc20dense_mace_adsorption_energies")

    selection_csv = output_root / "tables" / "trajectory_selection.csv"
    if write_selection:
        selection_csv = write_exact_selection_csv(selection_csv, selection)
    system_args = [str(row["system_id"]) for row in selection]
    adsorbate_list = ", ".join(str(row["adsorbate"]) for row in selection)
    data_root = ensure_oc20dense_reference_data(tutorial_root)
    archive = data_root / "raw_archives" / "oc20_dense_trajectories.tar.gz"

    return [
        ValidationWorkflowStep(
            label="Toolkit relaxation batch",
            message=(
                f"Relax {relax_batch_size} OC20Dense starts together with the "
                f"native Toolkit path: {adsorbate_list}."
            ),
            action=lambda: known_examples.run_oc20dense_known_examples(
                Namespace(
                    systems=system_args,
                    data_root=data_root,
                    initial_structure_dir=data_root / "initial_structures" / "adslab",
                    outdir=output_root,
                    max_configs_per_system=0,
                    config_ids=None,
                    selection_csv=selection_csv,
                    chunk_size=relax_batch_size,
                    n_steps=n_steps,
                    fmax=fmax,
                    allow_unpinned_adsorbates=False,
                    force=force,
                    no_trajectories=False,
                )
            ),
        ),
        ValidationWorkflowStep(
            label="Released DFT reference lookup and RMSD check",
            message=(
                "Read matching released DFT trajectories, convert them to extxyz, "
                "and compute RMSD/reference-target consistency. No DFT is run here."
            ),
            action=lambda: dft_checks.run_oc20dense_dft_reference_checks(
                Namespace(
                    data_root=data_root,
                    toolkit_root=output_root,
                    archive=archive,
                    extract_dir=data_root / "selected_trajectories" / "adslab",
                    outdir=output_root / "dft_reference_checks",
                    systems=system_args,
                    mode="compare",
                    scope="all",
                    max_members=0,
                )
            ),
        ),
        ValidationWorkflowStep(
            label="Toolkit adsorption-energy recomputation",
            message=(
                "Relax neutral gas molecules with Toolkit, evaluate clean surfaces, "
                "then compute Eads for the MACE-relaxed adslabs."
            ),
            action=lambda: mace_eads.run_oc20dense_mace_adsorption_energies(
                Namespace(
                    toolkit_root=output_root,
                    dft_check_dir=output_root / "dft_reference_checks",
                    dft_final_sp_dir=None,
                    archive=archive,
                    surface_dir=data_root / "selected_trajectories" / "surfaces",
                    outdir=output_root / "mace_adsorption_energy",
                    systems=system_args,
                    n_steps=n_steps,
                    fmax=fmax,
                    force=force,
                    skip_relaxed_adslab=False,
                    skip_dft_final_adslab=True,
                )
            ),
        ),
    ]


def build_nh3_ranking_stage_plan(
    *,
    tutorial_root: Path,
    source_root: Path,
    output_root: Path,
    system_id: str,
    single_point_batch_size: int,
    n_steps: int = 200,
    fmax: float = 0.05,
    force: bool = False,
) -> list[ValidationWorkflowStep]:
    """Build the fixed-geometry NH3 ranking workflow from visible settings."""
    source_root = prepare_nh3_ranking_reference_source(
        tutorial_root=tutorial_root,
        source_root=source_root,
        system_id=system_id,
        force=force,
    )
    if not (source_root / "tables" / "per_config_results.csv").exists():
        raise FileNotFoundError(
            f"Missing full OC20Dense per-config table at `{source_root}`."
        )

    _prepare_script_imports(tutorial_root)
    dft_checks = _fresh_import("oc20dense_dft_reference_checks")
    dft_final_sp = _fresh_import("run_oc20dense_dft_final_single_points")

    system_args = [system_id]
    data_root = ensure_oc20dense_reference_data(tutorial_root)
    archive = data_root / "raw_archives" / "oc20_dense_trajectories.tar.gz"

    return [
        ValidationWorkflowStep(
            label="Released DFT ranking lookup for all 92 NH3 structures",
            message=(
                "Read the released DFT trajectory records for one fixed NH3 "
                "surface system. No DFT is run here."
            ),
            action=lambda: dft_checks.run_oc20dense_dft_reference_checks(
                Namespace(
                    data_root=data_root,
                    toolkit_root=source_root,
                    archive=archive,
                    extract_dir=data_root / "selected_trajectories" / "adslab",
                    outdir=output_root / "dft_reference_checks",
                    systems=system_args,
                    mode="compare",
                    scope="all",
                    max_members=0,
                )
            ),
        ),
        ValidationWorkflowStep(
            label="Toolkit single-point scoring on 92 DFT-final geometries",
            message=(
                "Score all NH3 DFT-relaxed final geometries with Toolkit batch size "
                f"{single_point_batch_size}."
            ),
            action=lambda: dft_final_sp.run_oc20dense_dft_final_single_points(
                Namespace(
                    toolkit_root=source_root,
                    dft_check_dir=output_root / "dft_reference_checks",
                    outdir=output_root / "dft_final_single_points",
                    systems=system_args,
                    chunk_size=single_point_batch_size,
                    n_steps=n_steps,
                    fmax=fmax,
                    force=force,
                )
            ),
        ),
    ]


def _require_artifacts(files: dict[str, Path], *, label: str) -> None:
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} artifacts: " + ", ".join(missing)
        )


def _artifact_stem(row) -> str:
    return f"{row.system_id}_{row.config_id}_sid{int(row.sid)}"


def _path_or_none(value) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value)
    if not text:
        return None
    return Path(text)


def _resolve_validation_artifact(
    value,
    *,
    root: Path,
    relative_dir: str,
    fallback_name: str,
) -> Path:
    """Resolve live paths and stale saved-cache absolute paths to this context."""
    raw = _path_or_none(value)
    candidates = []
    if raw is not None:
        candidates.append(raw)
    candidates.append(root / relative_dir / fallback_name)
    if raw is not None:
        candidates.append(root / relative_dir / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_nh3_dft_final_structure(row, *, root: Path) -> Path:
    stem = _artifact_stem(row)
    primary = _resolve_validation_artifact(
        getattr(row, "dft_final_structure_path", None),
        root=root,
        relative_dir="dft_final_single_points/structures/dft_final",
        fallback_name=f"{stem}.extxyz",
    )
    if primary.exists():
        return primary
    alternate = _resolve_validation_artifact(
        getattr(row, "dft_final_structure_path", None),
        root=root,
        relative_dir="structures/dft_final",
        fallback_name=f"{stem}.extxyz",
    )
    return alternate if alternate.exists() else primary


def _adsorbate_atom_mask(atoms) -> np.ndarray:
    tags = atoms.arrays.get("tags")
    if tags is not None:
        tagged_adsorbate = np.asarray(tags) == 2
        if bool(np.any(tagged_adsorbate)):
            return tagged_adsorbate
    symbols = np.asarray(atoms.get_chemical_symbols())
    return np.isin(symbols, ["N", "H"])


def _adsorbate_anchor_index(atoms, adsorbate_mask: np.ndarray) -> int | None:
    symbols = np.asarray(atoms.get_chemical_symbols())
    nitrogen = np.flatnonzero(adsorbate_mask & (symbols == "N"))
    if nitrogen.size:
        return int(nitrogen[0])
    adsorbate = np.flatnonzero(adsorbate_mask)
    if adsorbate.size:
        return int(adsorbate[0])
    return None


def _adsorbate_unwrapped_local_positions(atoms, adsorbate_mask: np.ndarray) -> np.ndarray:
    """Return display positions with the adsorbate kept whole across PBCs."""
    from ase.geometry import find_mic

    positions = np.asarray(atoms.positions, dtype=float)
    anchor_index = _adsorbate_anchor_index(atoms, adsorbate_mask)
    if anchor_index is None:
        return positions.copy()

    anchor = positions[anchor_index]
    vectors, _distances = find_mic(
        positions - anchor,
        atoms.cell,
        atoms.pbc,
    )
    adsorbate_center = vectors[adsorbate_mask].mean(axis=0)
    return vectors - adsorbate_center


def _adsorbate_unwrapped_positions_in_cell(atoms, adsorbate_mask: np.ndarray) -> np.ndarray:
    """Return positions with only the adsorbate unwrapped around its anchor atom."""
    from ase.geometry import find_mic

    positions = np.asarray(atoms.positions, dtype=float)
    anchor_index = _adsorbate_anchor_index(atoms, adsorbate_mask)
    if anchor_index is None:
        return positions.copy()

    anchor = positions[anchor_index]
    adsorbate_indices = np.flatnonzero(adsorbate_mask)
    adsorbate_vectors, _distances = find_mic(
        positions[adsorbate_indices] - anchor,
        atoms.cell,
        atoms.pbc,
    )
    unwrapped = positions.copy()
    unwrapped[adsorbate_indices] = anchor + adsorbate_vectors
    return unwrapped


def _crop_adsorbate_local_environment(
    atoms,
    *,
    radius: float = 5.0,
):
    """Return a display-only local site crop around the tagged adsorbate."""
    adsorbate_mask = _adsorbate_atom_mask(atoms)
    if not bool(np.any(adsorbate_mask)):
        return atoms

    local_positions_all = _adsorbate_unwrapped_local_positions(atoms, adsorbate_mask)
    adsorbate_vectors = local_positions_all[adsorbate_mask]
    distances_to_adsorbate = np.linalg.norm(
        local_positions_all[:, None, :] - adsorbate_vectors[None, :, :],
        axis=2,
    ).min(axis=1)
    selected = adsorbate_mask | (distances_to_adsorbate <= float(radius))
    cropped = atoms[selected].copy()
    local_positions = local_positions_all[selected]
    cropped.positions = local_positions
    cropped.set_cell([0.0, 0.0, 0.0])
    cropped.set_pbc(False)
    camera = _top_down_adsorbate_site_camera(
        atoms,
        local_positions=local_positions,
        selected_mask=selected,
        adsorbate_mask=adsorbate_mask,
    )
    if camera is not None:
        cropped.info["render_camera"] = camera
    return cropped


def _unwrap_full_adsorbate_structure(atoms):
    """Return all atoms in a display frame where NH3 is not split by PBCs."""
    adsorbate_mask = _adsorbate_atom_mask(atoms)
    if not bool(np.any(adsorbate_mask)):
        return atoms

    local_positions = _adsorbate_unwrapped_positions_in_cell(atoms, adsorbate_mask)
    unwrapped = atoms.copy()
    unwrapped.positions = local_positions
    selected = np.ones(len(atoms), dtype=bool)
    camera = _top_down_adsorbate_site_camera(
        atoms,
        local_positions=local_positions,
        selected_mask=selected,
        adsorbate_mask=adsorbate_mask,
        target="extent",
        margin_factor=1.35,
    )
    if camera is not None:
        unwrapped.info["render_camera"] = camera
    return unwrapped


def _top_down_adsorbate_site_camera(
    atoms,
    *,
    local_positions: np.ndarray,
    selected_mask: np.ndarray,
    adsorbate_mask: np.ndarray,
    target: str = "adsorbate",
    margin_factor: float = 1.9,
) -> dict[str, object] | None:
    """Build a top-down camera from the original slab normal for local crops."""
    if atoms.cell.rank < 2 or len(local_positions) == 0:
        return None

    cell = np.asarray(atoms.cell.array, dtype=float)
    normal = np.cross(cell[0], cell[1])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm

    selected_adsorbate = np.asarray(adsorbate_mask[selected_mask], dtype=bool)
    selected_surface = ~selected_adsorbate
    if bool(np.any(selected_surface)):
        adsorbate_mean = local_positions[selected_adsorbate].mean(axis=0)
        surface_mean = local_positions[selected_surface].mean(axis=0)
        if float(np.dot(adsorbate_mean - surface_mean, normal)) < 0:
            normal = -normal

    lateral_1 = cell[0] / np.linalg.norm(cell[0])
    lateral_2 = np.cross(normal, lateral_1)
    lateral_2_norm = float(np.linalg.norm(lateral_2))
    if lateral_2_norm < 1e-8:
        lateral_2 = np.array([0.0, 1.0, 0.0])
    else:
        lateral_2 = lateral_2 / lateral_2_norm

    adsorbate_center = local_positions[selected_adsorbate].mean(axis=0)
    if target == "extent":
        lateral_1_values = local_positions @ lateral_1
        lateral_2_values = local_positions @ lateral_2
        center = adsorbate_center.copy()
        center += lateral_1 * float((lateral_1_values.min() + lateral_1_values.max()) / 2.0)
        center += lateral_2 * float((lateral_2_values.min() + lateral_2_values.max()) / 2.0)
        center -= lateral_1 * float(adsorbate_center @ lateral_1)
        center -= lateral_2 * float(adsorbate_center @ lateral_2)
    else:
        center = adsorbate_center

    relative = local_positions - center
    lateral_extent = max(
        float(np.ptp(relative @ lateral_1)),
        float(np.ptp(relative @ lateral_2)),
        3.8,
    )
    fov = float(np.deg2rad(24.0))
    distance = max(
        lateral_extent / (2.0 * np.tan(fov / 2.0)) * float(margin_factor),
        6.0,
    )
    camera_pos = center + normal * distance
    return {
        "camera_pos": tuple(float(x) for x in camera_pos),
        "camera_dir": tuple(float(x) for x in -normal),
        "fov": fov,
    }


def _assert_single_nh3_structure(atoms, *, source: str) -> None:
    symbols = np.asarray(atoms.get_chemical_symbols())
    adsorbate_mask = _adsorbate_atom_mask(atoms)
    counts = Counter(symbols[adsorbate_mask])
    if counts.get("O", 0):
        raise ValueError(f"Expected NH3-only adsorbate, found adsorbate O atoms in {source}.")
    if counts.get("N", 0) != 1 or counts.get("H", 0) != 3:
        raise ValueError(
            "Expected exactly one NH3 adsorbate in "
            f"{source}; found N={counts.get('N', 0)} and H={counts.get('H', 0)}."
        )


def _nh3_site_particle_colors(atoms) -> np.ndarray:
    colors = []
    for symbol in atoms.get_chemical_symbols():
        if symbol == "N":
            colors.append((0.05, 0.16, 0.85))
        elif symbol == "H":
            colors.append((0.96, 0.96, 0.96))
        else:
            colors.append((0.56, 0.58, 0.60))
    return np.asarray(colors, dtype=float)


def _normalize_rgb_float(values: Sequence[float]) -> tuple[float, float, float]:
    rgb = tuple(float(value) for value in values)
    if len(rgb) != 3:
        raise ValueError("RGB float colour must contain exactly three values")
    if any(value < 0.0 or value > 1.0 for value in rgb):
        raise ValueError("RGB float colour values must be in the [0, 1] range")
    return rgb


def _normalize_rgb_int(values: Sequence[int]) -> tuple[int, int, int]:
    rgb = tuple(int(value) for value in values)
    if len(rgb) != 3:
        raise ValueError("RGB integer colour must contain exactly three values")
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError("RGB integer colour values must be in the [0, 255] range")
    return rgb


def _rgb_float_cache_token(
    prefix: str,
    values: tuple[float, float, float],
    *,
    default: tuple[float, float, float],
) -> str:
    if all(abs(value - reference) < 1e-9 for value, reference in zip(values, default)):
        return ""
    return "_" + prefix + "".join(f"{int(round(value * 100)):03d}" for value in values)


def _rgb_int_cache_token(
    prefix: str,
    values: tuple[int, int, int],
    *,
    default: tuple[int, int, int],
) -> str:
    if values == default:
        return ""
    return "_" + prefix + "".join(f"{value:03d}" for value in values)


def _nh3_render_focus_token(
    focus: str,
    *,
    local_environment_radius: float,
) -> str:
    if focus == "full_slab":
        return "fullslab_top"
    if focus != "adsorbate_site":
        raise ValueError("focus must be 'adsorbate_site' or 'full_slab'")
    radius_token = f"{float(local_environment_radius):.1f}".replace(".", "p")
    return f"nh3site_r{radius_token}"


def load_trajectory_validation_results(
    *,
    root: Path,
    validation_model_label: str,
    display_fn=None,
    markdown_cls=None,
) -> dict[str, object]:
    """Load and display the three-start relaxation validation tables."""
    files = {
        "per_config": root / "tables" / "per_config_results.csv",
        "dft_check": root / "dft_reference_checks" / "dft_reference_comparison.csv",
        "eads_summary": root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_energy_summary.csv",
        "eads_per_config": root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_energies.csv",
        "eads_refs": root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_reference_energies.csv",
        "metadata": root / "reports" / "run_metadata.json",
    }
    _require_artifacts(files, label="trajectory-replay")

    per_config = pd.read_csv(files["per_config"])
    dft_reference = pd.read_csv(files["dft_check"])
    mace_results = pd.read_csv(files["eads_per_config"])
    mace_refs = pd.read_csv(files["eads_refs"])
    max_target_delta = dft_reference["dft_traj_minus_target_eV"].abs().max()
    if max_target_delta > 1e-9:
        raise AssertionError(f"DFT target mismatch: {max_target_delta:.3e} eV")

    relaxation_errors = mace_results.merge(
        dft_reference[
            [
                "system_id",
                "config_id",
                "sid",
                "start_adsorbate_rmsd_A",
                "mic_all_atom_rmsd_A",
                "mic_adsorbate_rmsd_A",
                "mic_active_atom_rmsd_A",
            ]
        ],
        on=["system_id", "config_id", "sid"],
        how="left",
        validate="one_to_one",
    ).sort_values(["system_id", "dft_rank"])

    if display_fn is not None and markdown_cls is not None:
        display_fn(
            markdown_cls(
                f"Loaded validated trajectory-replay artifacts for "
                f"`{len(dft_reference)}` OC20Dense records "
                f"with `{validation_model_label}`."
            )
        )
        trajectory_columns = [
            "system_id",
            "adsorbate",
            "config_id",
            "sid",
            "dft_rank",
            "dft_adsorption_energy_target_eV",
            "mace_relaxed_eads_eV",
            "mace_relaxed_eads_error_eV",
            "start_adsorbate_rmsd_A",
            "mic_adsorbate_rmsd_A",
            "mic_active_atom_rmsd_A",
        ]
        display_fn(
            relaxation_errors[trajectory_columns].rename(
                columns={
                    "dft_adsorption_energy_target_eV": "DFT Eads (eV)",
                    "mace_relaxed_eads_eV": "MACE relaxed Eads (eV)",
                    "mace_relaxed_eads_error_eV": "relaxed Eads error (eV)",
            "start_adsorbate_rmsd_A": "start adsorbate RMSD (A)",
            "mic_all_atom_rmsd_A": "final all-atom RMSD, MIC (A)",
            "mic_adsorbate_rmsd_A": "final tagged-adsorbate RMSD, MIC (A)",
            "mic_active_atom_rmsd_A": "final tagged-active RMSD, MIC (A)",
                }
            )
        )
        display_fn(
            markdown_cls(
                "This table compares the **MACE-relaxed trajectory endpoint** "
                "against the released OC20Dense DFT reference. Energy "
                "differences are `MACE relaxed Eads - released OC20Dense "
                "DFT-level Eads` in eV; positive values mean the model "
                "predicts weaker binding. RMSD values are reported in A and "
                "use the exact OC20Dense atom order plus tags."
            )
        )

    return {
        "trajectory_per_config": per_config,
        "trajectory_dft_reference": dft_reference,
        "trajectory_mace_results": mace_results,
        "trajectory_mace_refs": mace_refs,
        "trajectory_relaxation_errors": relaxation_errors,
        "max_trajectory_dft_target_delta_eV": max_target_delta,
    }


def load_nh3_ranking_results(
    *,
    root: Path,
    validation_model_label: str,
    preview_ranks: Sequence[int],
    show_all: bool,
    display_fn=None,
    markdown_cls=None,
) -> dict[str, object]:
    """Load and display the fixed-geometry NH3 ranking validation tables."""
    files = {
        "dft_check": root / "dft_reference_checks" / "dft_reference_comparison.csv",
        "dft_sp": root
        / "dft_final_single_points"
        / "tables"
        / "dft_final_sp_results.csv",
    }
    _require_artifacts(files, label="NH3 ranking")

    dft_reference = pd.read_csv(files["dft_check"])
    dft_final_sp = (
        pd.read_csv(files["dft_sp"]).sort_values("dft_rank").reset_index(drop=True)
    )
    ranking_all = dft_final_sp.copy()

    required_rank1_columns = {
        "dft_rank1_relative_energy_eV",
        "mace_dft_rank1_relative_total_energy_eV",
        "mace_dft_rank1_relative_energy_error_eV",
    }
    missing_rank1_columns = required_rank1_columns - set(dft_final_sp.columns)
    if missing_rank1_columns:
        raise RuntimeError(
            "The NH3 single-point table predates the DFT-rank-1 anchored "
            "columns. Rerun the NH3 ranking workflow with "
            "VALIDATION_FORCE_RECOMPUTE = True. "
            f"Missing: {sorted(missing_rank1_columns)}"
        )

    rank1_error = dft_final_sp["mace_dft_rank1_relative_energy_error_eV"]
    dft_rank1_row = dft_final_sp.loc[dft_final_sp["dft_rank"].idxmin()]
    mace_top_row = dft_final_sp.loc[
        dft_final_sp["mace_dft_final_sp_total_energy_eV"].idxmin()
    ]
    rank1_summary = pd.DataFrame(
        [
            {
                "validation model": validation_model_label,
                "n_configs": len(dft_final_sp),
                "DFT rank-1 config": dft_rank1_row["config_id"],
                "MACE top config": mace_top_row["config_id"],
                "MACE top DFT rank": int(mace_top_row["dft_rank"]),
                "MACE top DFT gap (eV)": mace_top_row["dft_gap_to_best_eV"],
                "DFT-best anchored MAE (eV)": rank1_error.abs().mean(),
                "DFT-best anchored RMSE (eV)": np.sqrt((rank1_error**2).mean()),
                "DFT-best anchored bias (eV)": rank1_error.mean(),
                "Spearman rank correlation": dft_final_sp["dft_rank"].corr(
                    dft_final_sp["mace_dft_final_sp_rank"]
                ),
            }
        ]
    )

    max_target_delta = dft_reference["dft_traj_minus_target_eV"].abs().max()
    max_start_adsorbate_rmsd = dft_reference["start_adsorbate_rmsd_A"].max()
    if max_target_delta > 1e-9:
        raise AssertionError(f"NH3 DFT target mismatch: {max_target_delta:.3e} eV")

    if show_all:
        display_rows = dft_final_sp.copy()
    else:
        first_ten = dft_final_sp.nsmallest(10, "dft_rank").copy()
        last_ten = dft_final_sp.nlargest(10, "dft_rank").sort_values("dft_rank").copy()
        first_ten.insert(0, "DFT rank window", "DFT top 10")
        last_ten.insert(0, "DFT rank window", "DFT bottom 10")
        display_rows = pd.concat([first_ten, last_ten], ignore_index=True)

    if display_fn is not None and markdown_cls is not None:
        display_fn(
            markdown_cls(
                f"#### NH<sub>3</sub> fixed-geometry ranking: `{len(dft_final_sp)}` "
                "DFT-relaxed final configurations"
            )
        )
        display_fn(
            markdown_cls(
                f"Max start-frame adsorbate RMSD: `{max_start_adsorbate_rmsd:.2e} A`."
            )
        )
        display_fn(
            markdown_cls(
                "The energy gaps below are anchored to the **DFT-best "
                "geometry**: DFT uses `E_DFT(row) - E_DFT(DFT-best)`, and "
                "MACE uses `E_MACE(row) - E_MACE(the same DFT-best geometry)`. "
                "The relative-energy error is `MACE gap - DFT gap`. Because "
                "all 92 rows share the same slab and adsorbate, constant "
                "gas/surface offsets cancel."
            )
        )
        ranking_columns = [
            "DFT rank window",
            "dft_rank",
            "config_id",
            "sid",
            "dft_adsorption_energy_eV",
            "dft_rank1_relative_energy_eV",
            "mace_dft_rank1_relative_total_energy_eV",
            "mace_dft_rank1_relative_energy_error_eV",
            "mace_dft_final_sp_rank",
        ]
        if show_all:
            ranking_columns = ranking_columns[1:]
        display_fn(
            display_rows[ranking_columns].rename(
                columns={
                    "dft_rank": "DFT rank",
                    "dft_adsorption_energy_eV": "DFT Eads target (eV)",
                    "dft_rank1_relative_energy_eV": "DFT gap from DFT-best (eV)",
                    "mace_dft_rank1_relative_total_energy_eV": "MACE gap from DFT-best geometry (eV)",
                    "mace_dft_rank1_relative_energy_error_eV": "relative-energy error, MACE - DFT (eV)",
                    "mace_dft_final_sp_rank": "MACE total-energy rank",
                }
            )
        )
        if not show_all:
            display_fn(
                markdown_cls(
                    "Showing the first 10 and last 10 configurations by released "
                    "DFT adsorption-energy rank. Set `SHOW_ALL_NH3_RANKING = True` "
                    "above to display all 92 rows. The full table is loaded as "
                    "`ranking_dft_final_sp`."
                )
            )
        summary = rank1_summary.iloc[0]
        display_fn(
            markdown_cls(
                "Across all 92 fixed geometries, the DFT-best anchored "
                f"relative-energy error is RMSE `{summary['DFT-best anchored RMSE (eV)']:.4f} eV`, "
                f"MAE `{summary['DFT-best anchored MAE (eV)']:.4f} eV`, "
                f"bias `{summary['DFT-best anchored bias (eV)']:.4f} eV`, and "
                f"Spearman rank correlation `{summary['Spearman rank correlation']:.3f}`."
            )
        )
        display_fn(
            markdown_cls(
                "This fixed-geometry ranking check does not need a separate "
                "clean-surface or gas reference: all 92 structures share the "
                "same surface and adsorbate, so the comparison uses relative "
                "gaps from the same DFT rank-1 geometry."
            )
        )

    return {
        "ranking_dft_reference": dft_reference,
        "ranking_dft_final_sp": dft_final_sp,
        "nh3_ranking_all": ranking_all,
        "rank1_summary": rank1_summary,
        "max_ranking_dft_target_delta_eV": max_target_delta,
        "max_ranking_start_adsorbate_rmsd_A": max_start_adsorbate_rmsd,
    }


def show_trajectory_validation_results(
    context: OC20DenseValidationContext,
    *,
    display_fn,
    markdown_cls,
) -> dict[str, object]:
    """Display the relaxation replay results with the notebook's compact columns."""
    results = load_trajectory_validation_results(
        root=context.trajectory_root,
        validation_model_label=context.model_label,
    )
    relaxation_errors = results["trajectory_relaxation_errors"]
    mace_refs = results["trajectory_mace_refs"]

    trajectory_table = relaxation_errors[
        [
            "adsorbate",
            "system_id",
            "config_id",
            "dft_adsorption_energy_target_eV",
            "mace_relaxed_eads_eV",
            "mace_relaxed_eads_error_eV",
            "mic_all_atom_rmsd_A",
        ]
    ].rename(
        columns={
            "dft_adsorption_energy_target_eV": "DFT Eads (eV)",
            "mace_relaxed_eads_eV": "MACE relaxed Eads (eV)",
            "mace_relaxed_eads_error_eV": "Eads error (eV)",
            "mic_all_atom_rmsd_A": "all-atom RMSD, MIC (A)",
        }
    )

    display_fn(markdown_cls("#### Relaxation check: Toolkit trajectories against OC20Dense DFT"))
    display_fn(trajectory_table.style.hide(axis="index"))
    display_fn(
        markdown_cls(
            "RMSD is reported for all atoms using the exact OC20Dense atom order "
            "and the minimum-image convention for periodic boundaries."
        )
    )

    worst_row = (
        relaxation_errors.assign(
            abs_eads_error=lambda df: df["mace_relaxed_eads_error_eV"].abs()
        )
        .sort_values("abs_eads_error", ascending=False)
        .iloc[0]
    )
    display_fn(
        markdown_cls(
            f"Largest MACE-vs-DFT relaxed-endpoint miss in this compact slice: "
            f"`{worst_row['adsorbate']}`: Eads error "
            f"`{worst_row['mace_relaxed_eads_error_eV']:.3f} eV`, "
            f"all-atom RMSD `{worst_row['mic_all_atom_rmsd_A']:.3f} A`."
        )
    )

    reference_table = mace_refs[
        [
            "system_id",
            "adsorbate_reference_species",
            "mace_gas_energy_eV",
            "gas_converged",
            "gas_optimizer_nsteps",
            "mace_surface_relaxed_energy_eV",
            "surface_relaxed_fmax_eV_A",
        ]
    ].rename(
        columns={
            "adsorbate_reference_species": "isolated gas molecule",
            "mace_gas_energy_eV": "Toolkit/MACE gas energy (eV)",
            "gas_optimizer_nsteps": "gas steps",
            "mace_surface_relaxed_energy_eV": "Toolkit/MACE clean-surface energy (eV)",
            "surface_relaxed_fmax_eV_A": "surface fmax after Toolkit/MACE relaxation (eV/A)",
        }
    )
    display_fn(markdown_cls("#### Toolkit/MACE reference terms used for relaxed Eads"))
    display_fn(
        markdown_cls(
            f"These clean-surface and gas terms are recomputed with "
            f"`{context.model_label}` so the MACE-relaxed adsorption energy uses "
            "one consistent model. The DFT comparison column above comes from "
            "the released OC20Dense adsorption-energy target."
        )
    )
    display_fn(reference_table.style.hide(axis="index"))
    return results


def validation_trajectory_artifact_paths(
    context: OC20DenseValidationContext,
    trajectory_results: dict[str, object],
) -> pd.DataFrame:
    """Return resolved DFT and Toolkit trajectory paths for validation videos."""
    per_config = trajectory_results["trajectory_per_config"]
    dft_reference = trajectory_results["trajectory_dft_reference"]
    keys = ["system_id", "config_id", "sid"]
    merged = dft_reference.merge(
        per_config[keys + ["toolkit_trajectory"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    order = {
        (row["system_id"], row["config_id"], int(row["sid"])): idx
        for idx, row in enumerate(context.trajectory_selection)
    }
    merged = (
        merged.assign(
            _display_order=lambda df: [
                order.get((r.system_id, r.config_id, int(r.sid)), len(order))
                for r in df.itertuples(index=False)
            ]
        )
        .sort_values("_display_order")
        .drop(columns="_display_order")
    )

    artifact_rows = []
    missing = []
    for row in merged.itertuples(index=False):
        stem = _artifact_stem(row)
        dft_path = _resolve_validation_artifact(
            getattr(row, "dft_trajectory_extxyz_path", None),
            root=context.trajectory_root,
            relative_dir="dft_reference_checks/dft_trajectories_extxyz",
            fallback_name=f"{stem}.extxyz",
        )
        mace_path = _resolve_validation_artifact(
            getattr(row, "toolkit_trajectory", None),
            root=context.trajectory_root,
            relative_dir="structures/toolkit_trajectories",
            fallback_name=f"{stem}.extxyz",
        )
        if not dft_path.exists():
            missing.append(f"DFT trajectory: {dft_path}")
        if not mace_path.exists():
            missing.append(f"MACE trajectory: {mace_path}")
        artifact_rows.append(
            {
                "system_id": row.system_id,
                "config_id": row.config_id,
                "sid": int(row.sid),
                "adsorbate": row.adsorbate,
                "dft_trajectory_path": dft_path.as_posix(),
                "toolkit_trajectory_path": mace_path.as_posix(),
                # Keep legacy keys so older render helpers/notebooks still work.
                "dft_trajectory_widget_path": dft_path.as_posix(),
                "mace_trajectory_widget_path": mace_path.as_posix(),
            }
        )
    if missing:
        raise FileNotFoundError("Missing validation trajectory artifacts: " + "; ".join(missing))

    return pd.DataFrame(artifact_rows)


def show_trajectory_validation_widget_grid(
    context: OC20DenseValidationContext,
    trajectory_results: dict[str, object],
    *,
    display_fn,
    markdown_cls,
    trajectory_grid_fn,
    width: str = "260px",
    height: str = "220px",
    show_cell: bool = False,
) -> pd.DataFrame:
    """Deprecated widget display retained for older notebooks."""
    artifact_df = validation_trajectory_artifact_paths(context, trajectory_results)
    widget_rows = []
    for row in artifact_df.itertuples(index=False):
        label_core = f"{row.adsorbate} | {row.system_id} {row.config_id}"
        widget_rows.append(
            [
                (f"DFT trajectory | {label_core}", Path(row.dft_trajectory_path)),
                (f"MACE trajectory | {label_core}", Path(row.toolkit_trajectory_path)),
            ]
        )

    display_fn(markdown_cls("#### DFT and MACE trajectory widgets"))
    try:
        trajectory_grid_fn(
            widget_rows,
            width=width,
            height=height,
            show_cell=show_cell,
        )
    except Exception as exc:
        display_fn(
            markdown_cls(
                "Interactive OVITO trajectory widgets are unavailable in this "
                f"frontend/kernel session: `{type(exc).__name__}: {exc}`. "
                "The trajectory files were still resolved successfully and can "
                "be opened in OVITO or tested in `ovito-trajectory-widget-smoke.ipynb`."
            )
        )
        display_fn(artifact_df.style.hide(axis="index"))
    return artifact_df


def show_nh3_ranking_results(
    context: OC20DenseValidationContext,
    *,
    display_fn,
    markdown_cls,
) -> dict[str, object]:
    """Display the fixed-geometry NH3 ranking check with compact columns."""
    results = load_nh3_ranking_results(
        root=context.nh3_ranking_root,
        validation_model_label=context.model_label,
        preview_ranks=context.preview_ranks,
        show_all=context.show_all_nh3_ranking,
    )
    dft_final_sp = results["ranking_dft_final_sp"]
    rank1_summary = results["rank1_summary"]

    if context.show_all_nh3_ranking:
        rows = dft_final_sp.copy()
    else:
        first_ten = dft_final_sp.nsmallest(10, "dft_rank").copy()
        last_ten = dft_final_sp.nlargest(10, "dft_rank").sort_values("dft_rank").copy()
        rows = pd.concat([first_ten, last_ten], ignore_index=True)
    ranking_columns = [
        "dft_rank",
        "config_id",
        "sid",
        "dft_rank1_relative_energy_eV",
        "mace_dft_rank1_relative_total_energy_eV",
        "mace_dft_rank1_relative_energy_error_eV",
        "mace_dft_final_sp_rank",
    ]
    ranking_table = rows[
        ranking_columns
    ].rename(
        columns={
            "dft_rank": "DFT rank",
            "dft_rank1_relative_energy_eV": "DFT gap from DFT-best (eV)",
            "mace_dft_rank1_relative_total_energy_eV": "MACE gap from DFT-best geometry (eV)",
            "mace_dft_rank1_relative_energy_error_eV": "relative-energy error, MACE - DFT (eV)",
            "mace_dft_final_sp_rank": "MACE rank",
        }
    )

    display_fn(markdown_cls("#### Fixed-geometry NH<sub>3</sub> ranking: 92 DFT-relaxed structures"))
    display_fn(ranking_table.style.hide(axis="index"))
    if not context.show_all_nh3_ranking:
        display_fn(
            markdown_cls(
                "Showing the first 10 and last 10 configurations by released "
                "DFT adsorption-energy rank. Set `SHOW_ALL_NH3_RANKING = True` "
                "in the validation settings cell to display all 92 rows."
            )
        )
    summary = rank1_summary.iloc[0]
    display_fn(
        markdown_cls(
            "Across all 92 fixed geometries, the DFT-best anchored "
            f"relative-energy error is RMSE `{summary['DFT-best anchored RMSE (eV)']:.4f} eV`, "
            f"MAE `{summary['DFT-best anchored MAE (eV)']:.4f} eV`, "
            f"bias `{summary['DFT-best anchored bias (eV)']:.4f} eV`, and "
            f"Spearman rank correlation `{summary['Spearman rank correlation']:.3f}`."
        )
    )
    display_fn(
        markdown_cls(
            "The ranking metric compares relative gaps from the same DFT-best "
            "geometry: `E(row) - E(DFT-best)` for DFT and for MACE. Because all "
            "rows share the same surface and adsorbate, constant gas/surface "
            "offsets cancel. The relative-energy error is `MACE gap - DFT gap`."
        )
    )
    return results


def show_nh3_all_geometry_grid(
    context: OC20DenseValidationContext,
    ranking_results: dict[str, object],
    *,
    display_fn,
    markdown_cls,
    paged_grid_fn,
    width: str = "180px",
    height: str = "160px",
    columns: int = 4,
    page_size: int = 12,
    show_cell: bool = False,
) -> pd.DataFrame:
    """Display all 92 NH3 DFT-final geometries as a static widget scan."""
    dft_final_sp = (
        ranking_results["ranking_dft_final_sp"]
        .sort_values("dft_rank")
        .reset_index(drop=True)
    )
    items = []
    rows = []
    missing = []
    for row in dft_final_sp.itertuples(index=False):
        structure_path = _resolve_nh3_dft_final_structure(
            row,
            root=context.nh3_ranking_root,
        )
        if not structure_path.exists():
            missing.append(structure_path.as_posix())
        label = (
            f"DFT rank {int(row.dft_rank)} | "
            f"MACE rank {int(row.mace_dft_final_sp_rank)} | {row.config_id}"
        )
        items.append((label, structure_path))
        rows.append(
            {
                "dft_rank": int(row.dft_rank),
                "mace_rank": int(row.mace_dft_final_sp_rank),
                "config_id": row.config_id,
                "sid": int(row.sid),
                "widget_structure_path": structure_path.as_posix(),
            }
        )
    if missing:
        raise FileNotFoundError(
            "Missing NH3 geometry widget structures: " + "; ".join(missing[:5])
            + (f"; ... {len(missing)} total" if len(missing) > 5 else "")
        )

    display_fn(markdown_cls("#### NH<sub>3</sub> 92-configuration visual scan"))
    paged_grid_fn(
        items,
        width=width,
        height=height,
        columns=columns,
        page_size=page_size,
        show_cell=show_cell,
    )
    return pd.DataFrame(rows)


def show_nh3_all_geometry_render_grid(
    context: OC20DenseValidationContext,
    ranking_results: dict[str, object],
    *,
    display_fn,
    markdown_cls,
    renderer: str = "visrtx",
    samples_per_pixel: int = 24,
    tile_size: tuple[int, int] = (360, 260),
    columns: int = 6,
    show_cell: bool = False,
    focus: str = "adsorbate_site",
    local_environment_radius: float = 5.0,
    force: bool = False,
    display_width: int = 1600,
    annotation_scale: float = 1.0,
    render_background: Sequence[float] = (0.97, 0.98, 0.99),
    overview_background: Sequence[int] = (255, 255, 255),
) -> pd.DataFrame:
    """Batch-render all 92 NH3 DFT-final geometries and show one grid PNG."""
    from IPython.display import Image as IPythonImage

    from .visualization import render_structure_ovito

    dft_final_sp = (
        ranking_results["ranking_dft_final_sp"]
        .sort_values("dft_rank")
        .reset_index(drop=True)
    )
    tile_width, tile_height = (int(tile_size[0]), int(tile_size[1]))
    columns = max(1, int(columns))
    annotation_scale = max(1.0, float(annotation_scale))
    render_background = _normalize_rgb_float(render_background)
    overview_background = _normalize_rgb_int(overview_background)
    visual_root = context.nh3_ranking_root / "visuals" / "nh3_92_geometry_scan"
    focus = str(focus)
    focus_token = _nh3_render_focus_token(
        focus,
        local_environment_radius=local_environment_radius,
    )
    if focus == "full_slab" and show_cell:
        focus_token = f"{focus_token}_cell"
    render_background_token = _rgb_float_cache_token(
        "bg",
        render_background,
        default=(0.97, 0.98, 0.99),
    )
    overview_background_token = _rgb_int_cache_token(
        "canvas",
        overview_background,
        default=(255, 255, 255),
    )
    tile_root = visual_root / (
        f"tiles_{focus_token}{render_background_token}_{renderer}_"
        f"{tile_width}x{tile_height}_spp{samples_per_pixel}"
    )
    annotation_token = (
        ""
        if abs(annotation_scale - 1.0) < 1e-9
        else f"_text{annotation_scale:.1f}".replace(".", "p")
    )
    overview_path = visual_root / (
        f"nh3_92_geometry_scan_{focus_token}{annotation_token}"
        f"{render_background_token}{overview_background_token}_{renderer}_{columns}col_"
        f"{tile_width}x{tile_height}_spp{samples_per_pixel}.png"
    )
    tile_root.mkdir(parents=True, exist_ok=True)
    surface_summary = describe_oc20dense_system(context, context.nh3_system)

    rows = []
    missing = []
    for row in dft_final_sp.itertuples(index=False):
        stem = _artifact_stem(row)
        structure_path = _resolve_nh3_dft_final_structure(
            row,
            root=context.nh3_ranking_root,
        )
        if not structure_path.exists():
            missing.append(structure_path.as_posix())
        tile_path = tile_root / f"rank_{int(row.dft_rank):03d}_{stem}.png"
        rows.append(
            {
                "dft_rank": int(row.dft_rank),
                "mace_rank": int(row.mace_dft_final_sp_rank),
                "config_id": row.config_id,
                "sid": int(row.sid),
                "dft_adsorption_energy_eV": float(row.dft_adsorption_energy_eV),
                "dft_gap_from_best_eV": float(row.dft_rank1_relative_energy_eV),
                "mace_gap_error_eV": float(row.mace_dft_rank1_relative_energy_error_eV),
                "structure_path": structure_path.as_posix(),
                "render_tile_path": tile_path.as_posix(),
                "render_focus": focus_token,
            }
        )
    if missing:
        raise FileNotFoundError(
            "Missing NH3 geometry render structures: " + "; ".join(missing[:5])
            + (f"; ... {len(missing)} total" if len(missing) > 5 else "")
        )

    if focus == "full_slab":
        cell_text = (
            "shows the simulation-cell boundary"
            if show_cell
            else "hides the simulation-cell boundary"
        )
        render_scope_text = (
            "The render keeps all atoms from each DFT-final structure, unwraps "
            f"NH<sub>3</sub> around the N atom, {cell_text}, and uses the same "
            "top-down surface-normal orientation for every tile. "
        )
        focus_label = (
            "all atoms, top-down NH3 orientation, cell shown"
            if show_cell
            else "all atoms, top-down NH3 orientation"
        )
    else:
        render_scope_text = (
            f"The render uses a `{local_environment_radius:.1f}` A local crop "
            "around NH<sub>3</sub> and its nearby surface atoms. "
        )
        focus_label = f"NH3/site crop radius {local_environment_radius:.1f} A"

    display_fn(markdown_cls("#### NH<sub>3</sub> 92-configuration visual scan"))
    display_fn(
        markdown_cls(
            f"Rendering `{len(rows)}` DFT-final NH<sub>3</sub> geometries with "
            f"`{renderer}` at `{tile_width}x{tile_height}` px per tile on "
            f"{surface_summary['surface_label']}. "
            f"{render_scope_text}"
            "Each tile reports the released reference rank, MACE single-point "
            "rank, configuration id, and sid for visual inspection."
        )
    )

    for idx, item in enumerate(rows, start=1):
        tile_path = Path(item["render_tile_path"])
        if force or not tile_path.exists():
            atoms = ase_read(item["structure_path"])
            _assert_single_nh3_structure(atoms, source=item["structure_path"])
            render_camera = None
            if focus == "adsorbate_site":
                atoms = _crop_adsorbate_local_environment(
                    atoms,
                    radius=float(local_environment_radius),
                )
                render_camera = atoms.info.get("render_camera")
                wrap_periodic_cell = show_cell
            else:
                atoms = _unwrap_full_adsorbate_structure(atoms)
                render_camera = atoms.info.get("render_camera")
                wrap_periodic_cell = False
            item["rendered_atoms"] = len(atoms)
            render_structure_ovito(
                atoms,
                output_path=tile_path.as_posix(),
                size=(tile_width, tile_height),
                background=render_background,
                renderer=renderer,
                samples_per_pixel=samples_per_pixel,
                show_cell=show_cell,
                camera=render_camera,
                wrap_periodic_cell=wrap_periodic_cell,
            )
        if idx == 1 or idx % 10 == 0 or idx == len(rows):
            print(f"NH3 render progress: {idx}/{len(rows)} tiles ready")

    _compose_nh3_render_overview(
        rows,
        overview_path=overview_path,
        columns=columns,
        tile_size=(tile_width, tile_height),
        renderer=renderer,
        samples_per_pixel=samples_per_pixel,
        surface_label=str(surface_summary["surface_label_plain"]),
        focus_label=focus_label,
        annotation_scale=annotation_scale,
        overview_background=overview_background,
        force=force,
    )
    display_fn(IPythonImage(filename=str(overview_path), width=int(display_width)))
    display_fn(
        pd.DataFrame(
            [
                {
                    "molecule": "NH3",
                    "surface": surface_summary["surface_label"],
                    "mpid": surface_summary["mpid"],
                    "miller_idx": surface_summary["miller_idx"],
                    "side": surface_summary["side"],
                    "system_id": context.nh3_system,
                    "structures": len(rows),
                    "render_focus": focus_token,
                    "renderer": renderer,
                    "samples_per_pixel": int(samples_per_pixel),
                    "annotation_scale": annotation_scale,
                    "render_background": render_background,
                    "overview_background": overview_background,
                    "grid_png": overview_path.as_posix(),
                    "tile_dir": tile_root.as_posix(),
                }
            ]
        ).style.hide(axis="index")
    )
    if focus == "adsorbate_site":
        display_fn(
            markdown_cls(
                "Adsorbate-site tiles use OVITO's default element colours and a "
                "top-down camera centred on NH<sub>3</sub>. Each source structure "
                "is checked to contain exactly one NH<sub>3</sub> adsorbate and no "
                "adsorbate O atoms before rendering."
            )
        )
    else:
        cell_note = (
            " The simulation-cell boundary is shown."
            if show_cell
            else " The simulation-cell boundary is hidden."
        )
        display_fn(
            markdown_cls(
                "Full-structure tiles use OVITO's default element colours and a "
                "shared top-down surface-normal orientation. The display copy is "
                "unwrapped around NH<sub>3</sub> so no H atom is split across a "
                "periodic boundary; the source structure is checked to contain "
                "exactly one NH<sub>3</sub> adsorbate and no adsorbate O atoms."
                f"{cell_note}"
            )
        )
    return pd.DataFrame(rows)


def _compose_nh3_render_overview(
    rows: list[dict],
    *,
    overview_path: Path,
    columns: int,
    tile_size: tuple[int, int],
    renderer: str,
    samples_per_pixel: int,
    surface_label: str,
    focus_label: str,
    annotation_scale: float,
    overview_background: tuple[int, int, int],
    force: bool,
) -> None:
    """Compose cached NH3 render tiles into one annotated PNG."""
    if overview_path.exists() and not force:
        return

    from PIL import Image, ImageDraw

    tile_width, tile_height = tile_size
    scale = max(1.0, float(annotation_scale))
    gap = 16
    margin = 24
    title_height = int(round(92 * scale))
    label_height = int(round(96 * scale))
    title_offset = int(round(38 * scale))
    subtitle_offset = int(round(62 * scale))
    label_pad_x = int(round(10 * scale))
    label_pad_y = int(round(8 * scale))
    label_line_step = int(round(20 * scale))
    rows_count = int(np.ceil(len(rows) / columns))
    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = (
        margin * 2
        + title_height
        + rows_count * (tile_height + label_height)
        + (rows_count - 1) * gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), overview_background)
    draw = ImageDraw.Draw(canvas)
    font_title = _pil_font(int(round(28 * scale)), bold=True)
    font_subtitle = _pil_font(int(round(16 * scale)))
    font_label = _pil_font(int(round(13 * scale)))
    font_label_bold = _pil_font(int(round(14 * scale)), bold=True)

    draw.text((margin, margin), "NH3 92-configuration visual scan", fill=(20, 24, 31), font=font_title)
    draw.text(
        (margin, margin + title_offset),
        f"{surface_label}; OVITO {renderer}, {samples_per_pixel} spp. "
        "Tiles sorted by released reference rank.",
        fill=(65, 72, 84),
        font=font_subtitle,
    )
    draw.text(
        (margin, margin + subtitle_offset),
        focus_label,
        fill=(65, 72, 84),
        font=font_subtitle,
    )

    for index, item in enumerate(rows):
        row_idx, col_idx = divmod(index, columns)
        x = margin + col_idx * (tile_width + gap)
        y = margin + title_height + row_idx * (tile_height + label_height + gap)
        tile = Image.open(item["render_tile_path"]).convert("RGB")
        if tile.size != (tile_width, tile_height):
            tile = tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        draw.rectangle(
            [x, y, x + tile_width - 1, y + tile_height + label_height - 1],
            outline=(202, 208, 216),
            width=1,
        )
        label_y = y + tile_height + label_pad_y
        draw.text(
            (x + label_pad_x, label_y),
            f"Reference #{item['dft_rank']:02d} | MACE SP #{item['mace_rank']:02d}",
            fill=(20, 24, 31),
            font=font_label_bold,
        )
        label_lines = [
            f"config {item['config_id']}",
            f"sid {item['sid']}",
            "released DFT-final geometry",
        ]
        for offset, text in enumerate(label_lines, start=1):
            draw.text(
                (x + label_pad_x, label_y + label_line_step * offset),
                text,
                fill=(65, 72, 84),
                font=font_label,
            )

    overview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(overview_path, optimize=True)


def _pil_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def show_nh3_geometry_widgets(
    context: OC20DenseValidationContext,
    ranking_results: dict[str, object],
    *,
    display_fn,
    markdown_cls,
    widgets_row_fn,
    width: str = "240px",
    height: str = "220px",
    show_cell: bool = False,
    local_environment_radius: float = 5.0,
) -> pd.DataFrame:
    """Display highlighted NH3 DFT-final structures and pairwise geometry spread."""
    ranking_all = ranking_results["nh3_ranking_all"]
    dft_final_sp = ranking_results["ranking_dft_final_sp"]
    if "dft_final_structure_path" in ranking_all.columns:
        highlight_paths = ranking_all.copy()
    else:
        highlight_paths = ranking_all.merge(
            dft_final_sp[["system_id", "config_id", "sid", "dft_final_structure_path"]],
            on=["system_id", "config_id", "sid"],
            how="left",
            validate="one_to_one",
        )
    highlight_paths = highlight_paths[
        highlight_paths["dft_rank"].isin(context.preview_ranks)
    ].copy()
    highlight_paths["resolved_dft_final_structure_path"] = [
        _resolve_nh3_dft_final_structure(
            row,
            root=context.nh3_ranking_root,
        )
        for row in highlight_paths.itertuples(index=False)
    ]

    def adsorbate_geometry(path: str) -> tuple[np.ndarray, object, np.ndarray]:
        atoms = ase_read(path)
        adsorbate = _adsorbate_atom_mask(atoms)
        if not bool(np.any(adsorbate)):
            raise ValueError(f"Could not identify NH3 adsorbate atoms in {path}")
        local_positions = _adsorbate_unwrapped_local_positions(atoms, adsorbate)
        positions = local_positions[adsorbate]
        return positions, atoms.cell, atoms.pbc

    rows = []
    for left, right in combinations(highlight_paths.itertuples(index=False), 2):
        from ase.geometry import find_mic

        left_pos, left_cell, left_pbc = adsorbate_geometry(left.resolved_dft_final_structure_path)
        right_pos, _right_cell, _right_pbc = adsorbate_geometry(
            right.resolved_dft_final_structure_path
        )
        left_com = left_pos.mean(axis=0)
        right_com = right_pos.mean(axis=0)
        _mic_vec, mic_com_distance = find_mic(
            right_com - left_com,
            cell=left_cell,
            pbc=left_pbc,
        )
        internal_delta = (left_pos - left_com) - (right_pos - right_com)
        rows.append(
            {
                "rank pair": f"{int(left.dft_rank)} vs {int(right.dft_rank)}",
                "config pair": f"{left.config_id} vs {right.config_id}",
                "adsorbate COM distance, MIC (A)": float(mic_com_distance),
                "internal adsorbate RMSD (A)": float(
                    np.sqrt(np.mean(np.sum(internal_delta**2, axis=1)))
                ),
            }
        )
    pairwise_rmsd = pd.DataFrame(rows)

    widget_items = []
    for row in highlight_paths.sort_values("dft_rank").itertuples(index=False):
        label = f"DFT rank {int(row.dft_rank)}: {row.config_id}"
        atoms = ase_read(row.resolved_dft_final_structure_path)
        _assert_single_nh3_structure(atoms, source=row.resolved_dft_final_structure_path)
        if show_cell:
            widget_atoms = _unwrap_full_adsorbate_structure(atoms)
        else:
            widget_atoms = _crop_adsorbate_local_environment(
                atoms,
                radius=float(local_environment_radius),
            )
        widget_items.append((
            label,
            widget_atoms,
        ))

    display_fn(markdown_cls("#### Highlighted NH<sub>3</sub> geometries"))
    widgets_row_fn(
        widget_items,
        width=width,
        height=height,
        show_cell=show_cell,
        wrap_periodic_cell=False if show_cell else None,
    )
    display_fn(markdown_cls("#### Geometry spread among highlighted NH<sub>3</sub> ranks"))
    display_fn(pairwise_rmsd.style.hide(axis="index"))
    display_fn(
        markdown_cls(
            "The pairwise table uses DFT-relaxed final structures and is "
            "reported in A. COM distances use the minimum-image convention; "
            "internal RMSD compares the adsorbate after subtracting its COM. "
            "This is a visual-diversity check, not a model-error metric."
        )
    )
    return pairwise_rmsd


def show_validation_model_tradeoff(
    *,
    display_fn,
    markdown_cls,
) -> dict[str, pd.DataFrame]:
    """Display the model-choice policy and the measured open-model baseline."""
    model_policy = pd.DataFrame(
        [
            {
                "model": "MACE-MP-0 small",
                "tutorial use": "active batch-size sweep option",
                "license note": "MIT-listed MACE-MP-0 family",
                "why include it": "fast open baseline for batch-size and memory trade-off",
            },
            {
                "model": "MACE-MP-0 large",
                "tutorial use": "active batch-size sweep option",
                "license note": "MIT-listed MACE-MP-0 family",
                "why include it": "larger open checkpoint for throughput comparison",
            },
            {
                "model": "MACE-MPA-0 medium",
                "tutorial use": "active default",
                "license note": "MIT-listed MACE-MPA-0 model",
                "why include it": "more recent open materials baseline for validation and screen",
            },
        ]
    )
    open_model_baseline = pd.DataFrame(
        [
            {
                "model": "MACE-MPA-0 medium",
                "workload": "92 NH3 fixed-geometry single-point energies",
                "batch size": 12,
                "wall time (s)": 14.90,
                "energy evaluation (s)": 1.48,
                "peak GPU alloc (GB)": 2.85,
                "relative-energy RMSE (eV)": 0.178,
                "DFT rank selected by MACE": 3,
            },
        ]
    )

    display_fn(markdown_cls("#### Model choice is a license, cost, and accuracy trade-off"))
    display_fn(model_policy.style.hide(axis="index"))
    display_fn(markdown_cls("#### Measured open-model baseline for this validation slice"))
    display_fn(open_model_baseline.style.hide(axis="index"))
    display_fn(
        markdown_cls(
            "The runnable notebook uses MACE-MP/MACE-MPA checkpoints only. "
            "For adsorption and surface chemistry, MACE-MH-1 with the OC20 "
            "surface head is the strongest follow-up candidate to test: the "
            "model card reports an OC20/surface head and surface-adsorption "
            "benchmarks, and the MACE foundation-model registry lists it as a "
            "multihead model for materials, molecules, and surfaces. It is "
            "beyond the scope of this runnable NVIDIA tutorial because the "
            "upstream model is ASL-licensed, so no shipped outputs in this "
            "notebook are produced with MH-1. Reference: "
            "[MACE-MH-1 model card](https://huggingface.co/mace-foundations/mace-mh-1) "
            "and [MACE foundation-model repository](https://github.com/ACEsuit/mace-foundations)."
        )
    )
    return {
        "model_policy": model_policy,
        "open_model_baseline": open_model_baseline,
    }
