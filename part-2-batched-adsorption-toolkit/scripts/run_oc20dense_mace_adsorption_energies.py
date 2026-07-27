#!/usr/bin/env python3
"""Compute defined MACE adsorption energies for the closed-shell OC20Dense set.

This script uses official OC20Dense DFT-relaxed final adslab and clean-surface
geometries when available:

E_ads^MACE = E_MACE(adslab) - E_MACE(clean surface) - E_MACE(gas molecule)

The clean-surface geometries are extracted from the official
``*_surface.traj`` members in the OC20Dense trajectory archive. Gas references
are neutral closed-shell molecules relaxed with the same MACE backend.

This is a useful MACE-vs-DFT comparison layer, but it is still a defined local
MACE convention. It should not be confused with subtracting the DFT
``oc20dense_ref_energies.pkl`` offsets from MACE totals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PART1 / "scripts"
sys.path.insert(0, str(PART1))
sys.path.insert(0, str(SCRIPT_DIR))

from helpers import (  # noqa: E402
    ase_to_atomic_data,
    atomic_data_to_ase,
    run_toolkit_relaxation_with_trajectory,
)
from oc20dense_dft_reference_checks import (  # noqa: E402
    DEFAULT_ARCHIVE,
    _read_atoms_sequence,
)
from _oc20dense_common import (  # noqa: E402
    CLOSED_SHELL_ADSORBATE_REFERENCES,
    DEFAULT_SURFACE_DIR,
    DEFAULT_SYSTEMS,
    FULL_DATA_NOTICE,
    MACE_EADS_REFERENCE_STATUS,
    MACE_RANK_BASIS,
    TOOLKIT_PROVENANCE_COLUMNS,
    require_precomputed_write_allowed,
    toolkit_cache_matches,
    toolkit_model_label,
    toolkit_provenance_from_env,
    toolkit_provenance_mismatch,
)
from run_oc20dense_known_examples import (  # noqa: E402
    DEFAULT_OUTDIR as DEFAULT_TOOLKIT_ROOT,
    _build_backend,
    _load_result,
    _max_force,
    _model_tensor,
    _result_to_json,
)


DEFAULT_OUTDIR = DEFAULT_TOOLKIT_ROOT / "mace_adsorption_energy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit-root", type=Path, default=DEFAULT_TOOLKIT_ROOT)
    parser.add_argument(
        "--dft-check-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing dft_reference_comparison.csv. Defaults to "
            "<toolkit-root>/dft_reference_checks."
        ),
    )
    parser.add_argument(
        "--dft-final-sp-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing tables/dft_final_sp_results.csv. Defaults to "
            "<toolkit-root>/dft_final_single_points."
        ),
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="OC20Dense system_id values to evaluate.",
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
    parser.add_argument("--force", action="store_true", help="Recompute cached refs.")
    parser.add_argument(
        "--skip-relaxed-adslab",
        action="store_true",
        help=(
            "Only compute Eads for fixed DFT-relaxed final adslab geometries. Use this "
            "when the per-config relaxation table was not produced with the "
            "same Toolkit checkpoint/head as the current single-point run."
        ),
    )
    parser.add_argument(
        "--skip-dft-final-adslab",
        action="store_true",
        help=(
            "Only compute Eads for Toolkit-relaxed adslab geometries. Use this "
            "for trajectory replay checks where the comparison is MACE relaxation "
            "from the OC20Dense start versus the DFT-relaxed reference."
        ),
    )
    return parser.parse_args()


def _safe(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("*", "star")
    )


def _ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "raw": outdir / "raw_refs",
        "structures": outdir / "structures",
        "trajectories": outdir / "trajectories",
        "trajectory_logs": outdir / "trajectory_logs",
        "single_point_logs": outdir / "single_point_logs",
        "tables": outdir / "tables",
        "reports": outdir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for name in (
        "surface_dft_final",
        "surface_dft_final_mace_sp",
        "surface_mace_relaxed",
        "gas",
    ):
        (dirs["structures"] / name).mkdir(parents=True, exist_ok=True)
    for name in ("surface_mace_relaxed", "gas"):
        (dirs["trajectories"] / name).mkdir(parents=True, exist_ok=True)
        (dirs["trajectory_logs"] / name).mkdir(parents=True, exist_ok=True)
    for name in ("surface_dft_final",):
        (dirs["single_point_logs"] / name).mkdir(parents=True, exist_ok=True)
    return dirs


def _raw_metadata_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(".metadata.json")


def _load_matching_raw_result(
    *,
    raw_path: Path,
    provenance: dict[str, Any],
    force: bool,
) -> Any | None:
    if force or not raw_path.exists():
        return None
    metadata_path = _raw_metadata_path(raw_path)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    if toolkit_cache_matches(metadata, provenance):
        return _load_result(raw_path)
    mismatch = toolkit_provenance_mismatch(metadata, provenance)
    print(f"  Reference cache mismatch for {raw_path.name}, recomputing: {mismatch}")
    return None


def _write_raw_result(
    *,
    raw_path: Path,
    result: Any,
    provenance: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    raw_path.write_text(
        json.dumps(_result_to_json(result), indent=2),
        encoding="utf-8",
    )
    metadata = {
        "raw_result": str(raw_path),
        **provenance,
        **(extra_metadata or {}),
    }
    _raw_metadata_path(raw_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _extract_surface_trajectories(
    *,
    archive: Path,
    systems: list[str],
    surface_dir: Path,
) -> dict[str, Path]:
    surface_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        f"trajs/{system_id}/{system_id}_surface.traj": system_id
        for system_id in systems
    }
    found: dict[str, Path] = {}
    for system_id in systems:
        existing = surface_dir / f"{system_id}_surface.traj"
        if existing.exists():
            found[system_id] = existing

    missing = set(systems) - set(found)
    if not missing:
        return found

    if not archive.exists():
        raise FileNotFoundError(
            "Missing selected clean-surface trajectories for "
            + ", ".join(sorted(missing))
            + f", and full archive is not available at {archive}. "
            + FULL_DATA_NOTICE
        )

    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            system_id = expected.get(member.name)
            if system_id is None or system_id not in missing:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            out_path = surface_dir / f"{system_id}_surface.traj"
            out_path.write_bytes(source.read())
            found[system_id] = out_path
            missing.remove(system_id)
            if not missing:
                break

    if missing:
        raise FileNotFoundError(
            "Could not find official surface trajectories for: "
            + ", ".join(sorted(missing))
        )
    return found


def _gas_atoms(species: str) -> Atoms:
    atoms = molecule(species)
    atoms.set_cell([15.0, 15.0, 15.0])
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def _surface_tags(initial_structure_path: Path) -> np.ndarray:
    initial = ase_read(initial_structure_path)
    tags = np.asarray(initial.get_tags(), dtype=int)
    return tags[tags != 2]


def _surface_atoms(surface_path: Path, tags: np.ndarray) -> tuple[Atoms, list[bool]]:
    frames = _read_atoms_sequence(surface_path)
    atoms = frames[-1].copy()
    if len(tags) > len(atoms):
        # OC20Dense adslab structures store the slab atoms first and append the
        # adsorbate atoms. Clean-surface trajectories only contain the slab, so
        # use the matching slab prefix when the adslab tag vector includes the
        # appended adsorbate.
        tags = tags[: len(atoms)]
    if len(atoms) != len(tags):
        raise ValueError(
            f"Surface atom count mismatch for {surface_path}: "
            f"{len(atoms)} atoms vs {len(tags)} tags"
        )
    atoms.set_tags(tags)
    atoms.set_constraint(FixAtoms(mask=(tags == 0).tolist()))
    active_mask = [bool(tag != 0) for tag in tags]
    return atoms, active_mask


def _single_point_energy(
    *,
    backend: Any,
    atoms: Atoms,
    active_mask: list[bool] | None,
    structure_id: str,
    structure_path: Path | None = None,
    log_path: Path | None = None,
) -> tuple[float, float]:
    from nvalchemi.neighbors import compute_neighbors

    payload = ase_to_atomic_data(
        atoms,
        structure_id=structure_id,
        active_mask=active_mask,
    )
    data = backend._to_atomic_data(payload)
    batch = backend.api.Batch.from_data_list([data], device=backend.device)
    compute_neighbors(batch, config=backend.model.model_config.neighbor_config)
    outputs = backend.model(batch)
    energy = float(_model_tensor(outputs, "energy").detach().cpu().numpy().reshape(-1)[0])
    forces = _model_tensor(outputs, "forces").detach().cpu().numpy().reshape(-1, 3)
    free_fmax = _max_force(forces.flatten().tolist(), active_mask=active_mask)
    all_fmax = _max_force(forces.flatten().tolist())
    if structure_path is not None:
        sp_atoms = atoms.copy()
        sp_atoms.info["structure_id"] = structure_id
        sp_atoms.info["mace_total_energy_eV"] = energy
        sp_atoms.info["mace_free_fmax_eV_A"] = free_fmax
        sp_atoms.info["mace_all_atom_fmax_eV_A"] = all_fmax
        sp_atoms.arrays["forces"] = forces
        ase_write(structure_path, sp_atoms, format="extxyz")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    "step,structure_id,energy_eV,max_force_eV_A,free_max_force_eV_A",
                    f"0,{structure_id},{energy},{all_fmax},{free_fmax}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return energy, free_fmax


def _relax_one(
    *,
    backend: Any,
    atoms: Atoms,
    active_mask: list[bool] | None,
    label: str,
    trajectory_path: Path | None = None,
    log_path: Path | None = None,
):
    start = time.perf_counter()
    payload = ase_to_atomic_data(atoms, structure_id=label, active_mask=active_mask)
    if trajectory_path is not None and log_path is not None:
        reply = run_toolkit_relaxation_with_trajectory(
            backend,
            [payload],
            label=label,
            trajectory_paths=[trajectory_path],
            log_paths=[log_path],
            cellopt=False,
        )
    else:
        reply = backend.relax([payload], label=label, cellopt=False)
    return reply.atoms[0], time.perf_counter() - start


def _gas_references(
    *,
    backend: Any,
    paths: dict[str, Path],
    species: list[str],
    force: bool,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in sorted(set(species)):
        raw_path = paths["raw"] / f"gas_{spec}.json"
        trajectory_path = paths["trajectories"] / "gas" / f"gas_{spec}.extxyz"
        trajectory_log_path = paths["trajectory_logs"] / "gas" / f"gas_{spec}.csv"
        trajectory_ready = trajectory_path.exists() and trajectory_log_path.exists()
        result = (
            _load_matching_raw_result(
                raw_path=raw_path,
                provenance=provenance,
                force=force,
            )
            if trajectory_ready
            else None
        )
        if result is not None:
            runtime_s = 0.0
        else:
            result, runtime_s = _relax_one(
                backend=backend,
                atoms=_gas_atoms(spec),
                active_mask=None,
                label=f"mace_gas_{spec}",
                trajectory_path=trajectory_path,
                log_path=trajectory_log_path,
            )
            _write_raw_result(
                raw_path=raw_path,
                result=result,
                provenance=provenance,
                extra_metadata={"reference_kind": "gas", "species": spec},
            )
        final_atoms = atomic_data_to_ase(result)
        ase_write(paths["structures"] / "gas" / f"gas_{spec}.extxyz", final_atoms)
        rows.append(
            {
                "adsorbate_reference_species": spec,
                "mace_gas_energy_eV": float(result.energy),
                "gas_converged": bool(result.converged),
                "gas_optimizer_nsteps": int(result.optimizer_nsteps),
                "gas_fmax_eV_A": _max_force(result.forces),
                "gas_runtime_s": runtime_s,
                "gas_trajectory_path": str(trajectory_path),
                "gas_trajectory_log_path": str(trajectory_log_path),
            }
        )
    return pd.DataFrame(rows)


def _surface_references(
    *,
    args: argparse.Namespace,
    backend: Any,
    paths: dict[str, Path],
    dft_reference: pd.DataFrame,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    surface_paths = _extract_surface_trajectories(
        archive=args.archive,
        systems=[str(system_id) for system_id in args.systems],
        surface_dir=args.surface_dir,
    )
    rows: list[dict[str, Any]] = []
    for system_id in args.systems:
        group = dft_reference[dft_reference["system_id"].astype(str) == str(system_id)]
        if group.empty:
            raise RuntimeError(f"No DFT reference rows for system {system_id}")
        first = group.iloc[0]
        tags = _surface_tags(Path(first["initial_structure_path"]))
        surface, active_mask = _surface_atoms(surface_paths[str(system_id)], tags)
        dft_final_path = (
            paths["structures"] / "surface_dft_final" / f"{system_id}_surface.extxyz"
        )
        ase_write(dft_final_path, surface, format="extxyz")
        surface_sp_path = (
            paths["structures"]
            / "surface_dft_final_mace_sp"
            / f"{system_id}_surface.extxyz"
        )
        surface_sp_log_path = (
            paths["single_point_logs"]
            / "surface_dft_final"
            / f"{system_id}_surface.csv"
        )

        sp_energy, sp_fmax = _single_point_energy(
            backend=backend,
            atoms=surface,
            active_mask=active_mask,
            structure_id=f"{system_id}_surface_dft_final",
            structure_path=surface_sp_path,
            log_path=surface_sp_log_path,
        )

        raw_path = paths["raw"] / f"surface_{system_id}_relaxed.json"
        trajectory_path = (
            paths["trajectories"]
            / "surface_mace_relaxed"
            / f"{system_id}_surface.extxyz"
        )
        trajectory_log_path = (
            paths["trajectory_logs"]
            / "surface_mace_relaxed"
            / f"{system_id}_surface.csv"
        )
        trajectory_ready = trajectory_path.exists() and trajectory_log_path.exists()
        relaxed = (
            _load_matching_raw_result(
                raw_path=raw_path,
                provenance=provenance,
                force=args.force,
            )
            if trajectory_ready
            else None
        )
        if relaxed is not None:
            relax_runtime_s = 0.0
        else:
            relaxed, relax_runtime_s = _relax_one(
                backend=backend,
                atoms=surface,
                active_mask=active_mask,
                label=f"mace_surface_{system_id}",
                trajectory_path=trajectory_path,
                log_path=trajectory_log_path,
            )
            _write_raw_result(
                raw_path=raw_path,
                result=relaxed,
                provenance=provenance,
                extra_metadata={"reference_kind": "surface", "system_id": system_id},
            )
        relaxed_atoms = atomic_data_to_ase(relaxed)
        relaxed_atoms.set_tags(tags)
        ase_write(
            paths["structures"]
            / "surface_mace_relaxed"
            / f"{system_id}_surface.extxyz",
            relaxed_atoms,
            format="extxyz",
        )

        rows.append(
            {
                "system_id": str(system_id),
                "surface_trajectory_path": str(surface_paths[str(system_id)]),
                "surface_dft_final_structure_path": str(dft_final_path),
                "surface_dft_final_mace_sp_structure_path": str(surface_sp_path),
                "surface_dft_final_mace_sp_log_path": str(surface_sp_log_path),
                "mace_surface_dft_final_sp_energy_eV": sp_energy,
                "mace_surface_dft_final_sp_fmax_eV_A": sp_fmax,
                "mace_surface_relaxed_energy_eV": float(relaxed.energy),
                "surface_relaxed_converged": bool(relaxed.converged),
                "surface_relaxed_optimizer_nsteps": int(relaxed.optimizer_nsteps),
                "surface_relaxed_fmax_eV_A": _max_force(
                    relaxed.forces,
                    active_mask=active_mask,
                ),
                "surface_relax_runtime_s": relax_runtime_s,
                "surface_relax_trajectory_path": str(trajectory_path),
                "surface_relax_trajectory_log_path": str(trajectory_log_path),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    *,
    paths: dict[str, Path],
    per_config: pd.DataFrame,
    summary: pd.DataFrame,
    refs: pd.DataFrame,
    skip_relaxed_adslab: bool,
    skip_dft_final_adslab: bool,
) -> None:
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)
    model_label = toolkit_model_label(provenance)
    cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "n_configs",
        "dft_final_eads_mae_eV",
        "dft_final_eads_rmse_eV",
        "dft_final_eads_bias_eV",
        "dft_final_relative_eads_mae_eV",
        "dft_final_relative_eads_rmse_eV",
        "dft_final_relative_eads_bias_eV",
        "dft_rank1_anchored_relative_eads_mae_eV",
        "dft_rank1_anchored_relative_eads_rmse_eV",
        "dft_rank1_anchored_relative_eads_bias_eV",
        "relaxed_eads_mae_eV",
        "relaxed_eads_rmse_eV",
        "relaxed_eads_bias_eV",
        "relaxed_relative_eads_mae_eV",
        "relaxed_relative_eads_rmse_eV",
        "relaxed_relative_eads_bias_eV",
        "relaxed_dft_rank1_anchored_relative_eads_mae_eV",
        "relaxed_dft_rank1_anchored_relative_eads_rmse_eV",
        "relaxed_dft_rank1_anchored_relative_eads_bias_eV",
        "dft_final_eads_top1_gap_eV",
        "relaxed_eads_top1_gap_eV",
    ]
    lines = [
        "# OC20Dense Defined MACE Adsorption-Energy Layer",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Backend: ALCHEMI Toolkit local MACE evaluations and relaxations",
        f"Model: {model_label}; D3(BJ) disabled",
        "",
        "## Scope",
        "",
        (
            "This report computes MACE adsorption energies with a defined local "
            "MACE convention: adslab energy minus official clean-surface final "
            "geometry energy minus neutral gas-molecule energy."
        ),
        "",
        (
            "The DFT comparison target is the released OC20Dense adsorption "
            "energy. Do not subtract `oc20dense_ref_energies.pkl` from MACE "
            "totals; those offsets are DFT-level reference offsets."
        ),
        "",
        (
            "Because the gas references are MACE-relaxed neutral molecules and "
            "the surface references are MACE evaluations of official DFT-relaxed final "
            "surface trajectories, these values are best read as a defined "
            "model-level Eads comparison rather than a claim that every OC20 "
            "reference convention has been exactly reproduced."
        ),
        "",
        (
            "DFT-relaxed final adslab single-point Eads columns were intentionally "
            "omitted for this run; the trajectory replay compares Toolkit/MACE "
            "relaxations from OC20Dense starting frames against the released DFT "
            "relaxation references."
            if skip_dft_final_adslab
            else "DFT-relaxed final adslab Eads columns use Toolkit/MACE single points "
            "on the released DFT-relaxed final adslab geometries."
        ),
        "",
        (
            "The `relative` columns shift both DFT and MACE energies to zero at "
            "their own minimum within the same fixed system before comparing "
            "rows. They test the shape of the adsorption-energy landscape and "
            "remove constant gas/surface reference offsets."
        ),
        "",
        (
            "The DFT-rank-1 anchored columns use a stricter reference: both DFT "
            "and MACE energy gaps are measured from the released DFT minimum "
            "geometry. This does not let MACE choose a different zero-energy "
            "structure before errors are calculated."
        ),
        "",
        (
            "Relaxed-adslab Eads columns were intentionally omitted for this "
            "run because the source relaxation table may come from a different "
            "Toolkit checkpoint/head."
            if skip_relaxed_adslab
            else "Relaxed-adslab Eads columns use the Toolkit relaxation table "
            "from the requested `--toolkit-root`; only compare them when that "
            "root was generated with the same checkpoint/head as this run."
        ),
        "",
        "## System Summary",
        "",
        summary[cols].to_markdown(index=False),
        "",
        "## Reference Energies",
        "",
        refs.to_markdown(index=False),
        "",
        "## Output Tables",
        "",
        f"- Per-config Eads: `{paths['tables'] / 'mace_adsorption_energies.csv'}`",
        f"- System summary: `{paths['tables'] / 'mace_adsorption_energy_summary.csv'}`",
        f"- Reference energies: `{paths['tables'] / 'mace_adsorption_reference_energies.csv'}`",
        f"- Relaxation trajectories: `{paths['trajectories']}`",
        f"- Relaxation energy/force logs: `{paths['trajectory_logs']}`",
        f"- Single-point energy/force logs: `{paths['single_point_logs']}`",
    ]
    (paths["reports"] / "mace_adsorption_energy_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_oc20dense_mace_adsorption_energies(args: argparse.Namespace) -> int:
    require_precomputed_write_allowed(args.outdir)
    paths = _ensure_dirs(args.outdir)
    backend = _build_backend(args)
    skip_relaxed_adslab = bool(getattr(args, "skip_relaxed_adslab", False))
    skip_dft_final_adslab = bool(getattr(args, "skip_dft_final_adslab", False))
    if skip_relaxed_adslab and skip_dft_final_adslab:
        raise ValueError(
            "At least one adslab energy layer must be enabled: fixed DFT-relaxed final "
            "single points, Toolkit-relaxed adslabs, or both."
        )
    provenance = toolkit_provenance_from_env(d3bj_enabled=False)

    dft_check_dir = args.dft_check_dir or (args.toolkit_root / "dft_reference_checks")
    dft_final_sp_dir = args.dft_final_sp_dir or (
        args.toolkit_root / "dft_final_single_points"
    )
    dft_reference = pd.read_csv(dft_check_dir / "dft_reference_comparison.csv")
    dft_final_sp = (
        None
        if skip_dft_final_adslab
        else pd.read_csv(dft_final_sp_dir / "tables" / "dft_final_sp_results.csv")
    )
    if skip_relaxed_adslab:
        toolkit = dft_reference[["system_id", "config_id", "sid"]].copy()
        toolkit["ml_total_energy_eV"] = np.nan
    else:
        toolkit = pd.read_csv(args.toolkit_root / "tables" / "per_config_results.csv")
        missing_provenance = [
            key for key in TOOLKIT_PROVENANCE_COLUMNS if key not in toolkit.columns
        ]
        if missing_provenance:
            raise RuntimeError(
                "The relaxed-adslab Toolkit table has no model provenance columns. "
                "Rerun the Toolkit relaxation table with the current scripts, or "
                "pass --skip-relaxed-adslab for fixed-geometry Eads only. Missing: "
                f"{missing_provenance}"
            )
        observed = {
            key: toolkit[key].dropna().iloc[0]
            if not toolkit[key].dropna().empty
            else None
            for key in TOOLKIT_PROVENANCE_COLUMNS
        }
        mismatch = toolkit_provenance_mismatch(observed, provenance)
        if mismatch:
            raise RuntimeError(
                "The relaxed-adslab Toolkit table was generated with different "
                f"model settings. Refusing to form mixed-model Eads: {mismatch}. "
                "Use a matching toolkit root or pass --skip-relaxed-adslab."
            )

    requested = {str(system_id) for system_id in args.systems}
    dft_reference = dft_reference[
        dft_reference["system_id"].astype(str).isin(requested)
    ]
    if dft_final_sp is not None:
        dft_final_sp = dft_final_sp[
            dft_final_sp["system_id"].astype(str).isin(requested)
        ]
    toolkit = toolkit[toolkit["system_id"].astype(str).isin(requested)]

    gas_species = [
        CLOSED_SHELL_ADSORBATE_REFERENCES[str(adsorbate)]
        for adsorbate in sorted(dft_reference["adsorbate"].unique())
    ]
    gas_refs = _gas_references(
        backend=backend,
        paths=paths,
        species=gas_species,
        force=args.force,
        provenance=provenance,
    )
    surface_refs = _surface_references(
        args=args,
        backend=backend,
        paths=paths,
        dft_reference=dft_reference,
        provenance=provenance,
    )

    refs = surface_refs.merge(
        dft_reference[
            ["system_id", "adsorbate", "adsorbate_reference_species"]
        ].drop_duplicates(),
        on="system_id",
        how="left",
        validate="one_to_one",
    ).merge(
        gas_refs,
        on="adsorbate_reference_species",
        how="left",
        validate="many_to_one",
    )
    for key, value in provenance.items():
        refs[key] = value

    per_config = dft_reference[
        [
            "system_id",
            "config_id",
            "sid",
            "adsorbate",
            "adsorbate_reference_species",
            "dft_rank",
            "dft_adsorption_energy_target_eV",
            "dft_gap_to_system_best_eV",
        ]
    ]
    if dft_final_sp is None:
        per_config["mace_dft_final_sp_total_energy_eV"] = np.nan
    else:
        per_config = per_config.merge(
            dft_final_sp[
                [
                    "system_id",
                    "config_id",
                    "sid",
                    "mace_dft_final_sp_total_energy_eV",
                ]
            ],
            on=["system_id", "config_id", "sid"],
            how="left",
            validate="one_to_one",
        )
    per_config = per_config.merge(
        toolkit[
            [
                "system_id",
                "config_id",
                "sid",
                "ml_total_energy_eV",
            ]
        ],
        on=["system_id", "config_id", "sid"],
        how="left",
        validate="one_to_one",
    ).merge(
        refs[
            [
                "system_id",
                "mace_surface_dft_final_sp_energy_eV",
                "mace_surface_relaxed_energy_eV",
                "mace_gas_energy_eV",
            ]
        ],
        on="system_id",
        how="left",
        validate="many_to_one",
    )

    per_config["mace_dft_final_eads_eV"] = (
        per_config["mace_dft_final_sp_total_energy_eV"]
        - per_config["mace_surface_dft_final_sp_energy_eV"]
        - per_config["mace_gas_energy_eV"]
    )
    per_config["mace_relaxed_eads_eV"] = (
        per_config["ml_total_energy_eV"]
        - per_config["mace_surface_relaxed_energy_eV"]
        - per_config["mace_gas_energy_eV"]
    )
    per_config["mace_dft_final_eads_error_eV"] = (
        per_config["mace_dft_final_eads_eV"]
        - per_config["dft_adsorption_energy_target_eV"]
    )
    per_config["mace_relaxed_eads_error_eV"] = (
        per_config["mace_relaxed_eads_eV"]
        - per_config["dft_adsorption_energy_target_eV"]
    )
    per_config["dft_relative_eads_eV"] = (
        per_config["dft_adsorption_energy_target_eV"]
        - per_config.groupby("system_id")["dft_adsorption_energy_target_eV"].transform("min")
    )
    per_config["mace_dft_final_relative_eads_eV"] = (
        per_config["mace_dft_final_eads_eV"]
        - per_config.groupby("system_id")["mace_dft_final_eads_eV"].transform("min")
    )
    per_config["mace_relaxed_relative_eads_eV"] = (
        per_config["mace_relaxed_eads_eV"]
        - per_config.groupby("system_id")["mace_relaxed_eads_eV"].transform("min")
    )
    per_config["mace_dft_final_relative_eads_error_eV"] = (
        per_config["mace_dft_final_relative_eads_eV"]
        - per_config["dft_relative_eads_eV"]
    )
    per_config["mace_relaxed_relative_eads_error_eV"] = (
        per_config["mace_relaxed_relative_eads_eV"]
        - per_config["dft_relative_eads_eV"]
    )
    dft_rank1_anchor = per_config.loc[
        per_config.groupby("system_id")["dft_adsorption_energy_target_eV"].idxmin(),
        ["system_id", "mace_dft_final_eads_eV", "mace_relaxed_eads_eV"],
    ].rename(
        columns={
            "mace_dft_final_eads_eV": "mace_dft_final_eads_at_dft_rank1_geometry_eV",
            "mace_relaxed_eads_eV": "mace_relaxed_eads_at_dft_rank1_start_eV",
        }
    )
    per_config = per_config.merge(
        dft_rank1_anchor,
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    per_config["mace_dft_rank1_relative_eads_eV"] = (
        per_config["mace_dft_final_eads_eV"]
        - per_config["mace_dft_final_eads_at_dft_rank1_geometry_eV"]
    )
    per_config["mace_relaxed_dft_rank1_relative_eads_eV"] = (
        per_config["mace_relaxed_eads_eV"]
        - per_config["mace_relaxed_eads_at_dft_rank1_start_eV"]
    )
    per_config["mace_dft_rank1_relative_eads_error_eV"] = (
        per_config["mace_dft_rank1_relative_eads_eV"]
        - per_config["dft_relative_eads_eV"]
    )
    per_config["mace_relaxed_dft_rank1_relative_eads_error_eV"] = (
        per_config["mace_relaxed_dft_rank1_relative_eads_eV"]
        - per_config["dft_relative_eads_eV"]
    )
    per_config["mace_dft_final_eads_rank"] = (
        per_config.groupby("system_id")["mace_dft_final_eads_eV"]
        .rank(method="first", ascending=True)
        .astype("Int64")
    )
    per_config["mace_relaxed_eads_rank"] = (
        per_config.groupby("system_id")["mace_relaxed_eads_eV"]
        .rank(method="first", ascending=True)
        .astype("Int64")
    )
    per_config["mace_rank_basis"] = MACE_RANK_BASIS
    per_config["mace_eads_reference_status"] = MACE_EADS_REFERENCE_STATUS
    for key, value in provenance.items():
        per_config[key] = value

    summary_rows: list[dict[str, Any]] = []
    for system_id, group in per_config.groupby("system_id", sort=False):
        dft_final_has_values = group["mace_dft_final_eads_eV"].notna().any()
        dft_final_best = (
            group.loc[group["mace_dft_final_eads_eV"].idxmin()]
            if dft_final_has_values
            else None
        )
        relaxed_has_values = group["mace_relaxed_eads_eV"].notna().any()
        relaxed_best = (
            group.loc[group["mace_relaxed_eads_eV"].idxmin()]
            if relaxed_has_values
            else None
        )
        summary_rows.append(
            {
                "system_id": system_id,
                "adsorbate": str(group.iloc[0]["adsorbate"]),
                "adsorbate_reference_species": str(
                    group.iloc[0]["adsorbate_reference_species"]
                ),
                "n_configs": int(len(group)),
                "dft_final_eads_mae_eV": (
                    float(group["mace_dft_final_eads_error_eV"].abs().mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_final_eads_rmse_eV": (
                    float(np.sqrt(np.mean(group["mace_dft_final_eads_error_eV"] ** 2)))
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_final_eads_bias_eV": (
                    float(group["mace_dft_final_eads_error_eV"].mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_final_relative_eads_mae_eV": (
                    float(group["mace_dft_final_relative_eads_error_eV"].abs().mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_final_relative_eads_rmse_eV": (
                    float(
                        np.sqrt(
                            np.mean(group["mace_dft_final_relative_eads_error_eV"] ** 2)
                        )
                    )
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_final_relative_eads_bias_eV": (
                    float(group["mace_dft_final_relative_eads_error_eV"].mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_rank1_anchored_relative_eads_mae_eV": (
                    float(group["mace_dft_rank1_relative_eads_error_eV"].abs().mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_rank1_anchored_relative_eads_rmse_eV": (
                    float(
                        np.sqrt(
                            np.mean(group["mace_dft_rank1_relative_eads_error_eV"] ** 2)
                        )
                    )
                    if dft_final_has_values
                    else np.nan
                ),
                "dft_rank1_anchored_relative_eads_bias_eV": (
                    float(group["mace_dft_rank1_relative_eads_error_eV"].mean())
                    if dft_final_has_values
                    else np.nan
                ),
                "relaxed_eads_mae_eV": (
                    float(group["mace_relaxed_eads_error_eV"].abs().mean())
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_eads_rmse_eV": (
                    float(np.sqrt(np.mean(group["mace_relaxed_eads_error_eV"] ** 2)))
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_eads_bias_eV": (
                    float(group["mace_relaxed_eads_error_eV"].mean())
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_relative_eads_mae_eV": (
                    float(group["mace_relaxed_relative_eads_error_eV"].abs().mean())
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_relative_eads_rmse_eV": (
                    float(
                        np.sqrt(
                            np.mean(group["mace_relaxed_relative_eads_error_eV"] ** 2)
                        )
                    )
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_relative_eads_bias_eV": (
                    float(group["mace_relaxed_relative_eads_error_eV"].mean())
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_dft_rank1_anchored_relative_eads_mae_eV": (
                    float(
                        group["mace_relaxed_dft_rank1_relative_eads_error_eV"]
                        .abs()
                        .mean()
                    )
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_dft_rank1_anchored_relative_eads_rmse_eV": (
                    float(
                        np.sqrt(
                            np.mean(
                                group["mace_relaxed_dft_rank1_relative_eads_error_eV"]
                                ** 2
                            )
                        )
                    )
                    if relaxed_has_values
                    else np.nan
                ),
                "relaxed_dft_rank1_anchored_relative_eads_bias_eV": (
                    float(group["mace_relaxed_dft_rank1_relative_eads_error_eV"].mean())
                    if relaxed_has_values
                    else np.nan
                ),
                "dft_final_eads_best_config": (
                    str(dft_final_best["config_id"]) if dft_final_best is not None else ""
                ),
                "dft_final_eads_best_dft_rank": (
                    int(dft_final_best["dft_rank"]) if dft_final_best is not None else np.nan
                ),
                "dft_final_eads_top1_gap_eV": float(
                    dft_final_best["dft_gap_to_system_best_eV"]
                )
                if dft_final_best is not None
                else np.nan,
                "relaxed_eads_best_config": (
                    str(relaxed_best["config_id"]) if relaxed_best is not None else ""
                ),
                "relaxed_eads_best_dft_rank": (
                    int(relaxed_best["dft_rank"]) if relaxed_best is not None else np.nan
                ),
                "relaxed_eads_top1_gap_eV": (
                    float(relaxed_best["dft_gap_to_system_best_eV"])
                    if relaxed_best is not None
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    refs.to_csv(paths["tables"] / "mace_adsorption_reference_energies.csv", index=False)
    per_config.to_csv(paths["tables"] / "mace_adsorption_energies.csv", index=False)
    summary.to_csv(
        paths["tables"] / "mace_adsorption_energy_summary.csv",
        index=False,
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "systems": list(args.systems),
        "skip_relaxed_adslab": skip_relaxed_adslab,
        "skip_dft_final_adslab": skip_dft_final_adslab,
        "n_rows": int(len(per_config)),
        "summary_csv": str(paths["tables"] / "mace_adsorption_energy_summary.csv"),
        "per_config_csv": str(paths["tables"] / "mace_adsorption_energies.csv"),
        "reference_csv": str(paths["tables"] / "mace_adsorption_reference_energies.csv"),
        **provenance,
    }
    (paths["reports"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_report(
        paths=paths,
        per_config=per_config,
        summary=summary,
        refs=refs,
        skip_relaxed_adslab=skip_relaxed_adslab,
        skip_dft_final_adslab=skip_dft_final_adslab,
    )

    print(paths["tables"] / "mace_adsorption_energy_summary.csv")
    print(summary.to_string(index=False))
    return 0


def main() -> int:
    return run_oc20dense_mace_adsorption_energies(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
