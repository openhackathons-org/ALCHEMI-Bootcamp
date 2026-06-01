"""Surface-screen helpers for the batched adsorption tutorial.

The public notebook should show the scientific choices: which surfaces, which
adsorbates, how many starting structures, and why. This module deliberately
keeps only repeatable bookkeeping: artifact paths, audit tables, convergence
statistics, and result summaries.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

from .config_search import Configuration, sites_for_host

if TYPE_CHECKING:
    import ase


SURFACE_SCREEN_OUTPUT_DIR_NAME = "surface_screen_v1_mace_mpa0"
SURFACE_SCREEN_GRID_VERSION = "surface-screen-v1-six-start-9x4"


@dataclass(frozen=True)
class SurfaceScreenSlabSpec:
    """One slab in the tutorial surface screen."""

    name: str
    material_class: str
    facet: str
    miller_index: tuple[int, int, int]
    builder_name: str
    builder: Callable[[], "ase.Atoms"]
    default_supercell: tuple[int, int, int]
    note: str


@dataclass(frozen=True)
class SurfaceScreenAdsorbateSpec:
    """One adsorbate in the tutorial surface screen."""

    name: str
    orientations: tuple[str, str]
    application_hint: str
    note: str


def _safe(name: str) -> str:
    return (
        name.replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def safe_artifact_label(name: str) -> str:
    """Return the filesystem-safe label used for per-structure artifacts."""
    return _safe(name)


def surface_screen_plan_table(
    specs: list[SurfaceScreenSlabSpec] | tuple[SurfaceScreenSlabSpec, ...],
) -> pd.DataFrame:
    """Format the visible slab plan for notebook display."""
    return pd.DataFrame(
        [
            {
                "surface": spec.name,
                "class": spec.material_class,
                "Miller index": str(spec.miller_index),
                "facet model": spec.facet,
                "builder": spec.builder_name,
                "supercell": str(spec.default_supercell),
                "why included": spec.note,
            }
            for spec in specs
        ]
    )


def require_surface_screen_artifact(
    paths: dict[str, Path],
    path_key: str,
    *,
    relpath=lambda path: str(path),
) -> Path:
    """Return a required surface-screen artifact path or raise a tutorial-facing error."""
    path = paths[path_key]
    if not path.exists():
        raise RuntimeError(
            f"Precomputed surface-screen artifact is missing: {relpath(path)}. "
            "Run scripts/run_surface_screen.py on the Toolkit GPU environment, select a complete "
            'SAVED_TUTORIAL_RUN_ID, or set TUTORIAL_RESULT_SOURCE = "compute" for a live notebook recompute.'
        )
    return path


def surface_screen_result_json_path(paths: dict[str, Path], label: str) -> Path:
    """Return the raw-result JSON path for one generated adsorption configuration."""
    return paths["raw"] / f"{safe_artifact_label(label)}.json"


def surface_screen_result_artifact_paths(
    output_root: str | Path,
    pair_results: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, object]:
    """Return the result artifact paths written by the notebook."""
    paths = surface_screen_output_paths(output_root)
    pair_result_paths = [
        paths["tables"] / f"pair_results_{_safe(adsorbate)}_{_safe(host)}.csv"
        for (host, adsorbate) in pair_results
    ]
    return {
        "pair_result_paths": pair_result_paths,
        "summary_validation_csv": paths["tables"] / "summary_validation.csv",
        "adsorbml_bias_csv": paths["tables"] / "adsorbml_bias.csv",
        "metadata": paths["metadata"],
        "adsorption_results_csv": paths["adsorption_results_csv"],
        "report_md": paths["report_md"],
    }


def _formula(atoms: ase.Atoms) -> str:
    counts = Counter(atoms.get_chemical_symbols())
    return "".join(
        f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}"
        for symbol in sorted(counts)
    )


def surface_screen_expected_counts(
    *,
    n_slabs: int,
    n_adsorbates: int,
    starts_per_pair: int = 6,
) -> dict[str, int]:
    """Return the planned simulation counts for the surface screen."""
    n_pairs = n_slabs * n_adsorbates
    return {
        "slabs": n_slabs,
        "adsorbates": n_adsorbates,
        "adsorbate_surface_pairs": n_pairs,
        "starts_per_pair": starts_per_pair,
        "adsorption_relaxations": n_pairs * starts_per_pair,
        "clean_slab_relaxations": n_slabs,
        "gas_reference_relaxations": n_adsorbates,
        "core_relaxations": n_pairs * starts_per_pair + n_slabs + n_adsorbates,
    }


def surface_screen_output_paths(output_root: str | Path) -> dict[str, Path]:
    """Return canonical artifact paths for the 9-facet surface screen."""
    root = Path(output_root)
    return {
        "root": root,
        "tables": root / "tables",
        "chunks": root / "chunks",
        "raw": root / "raw_batches",
        "structures": root / "structures",
        "initial_structures": root / "structures" / "initial_adsorption",
        "relaxed_structures": root / "structures" / "relaxed_adsorption",
        "clean_structures": root / "structures" / "clean_slabs",
        "gas_structures": root / "structures" / "gas_adsorbates",
        "trajectories": root / "trajectories",
        "trajectory_logs": root / "logs",
        "reports": root / "reports",
        "figures": root / "figures",
        "metadata": root / "run_metadata.json",
        "surface_fingerprints_csv": root / "tables" / "surface_fingerprints.csv",
        "initial_geometry_audit_csv": root / "tables" / "initial_geometry_audit.csv",
        "adsorption_results_csv": root / "tables" / "adsorption_results.csv",
        "pair_summary_csv": root / "tables" / "pair_summary.csv",
        "batch_summary_csv": root / "tables" / "batch_summary.csv",
        "step_statistics_csv": root / "tables" / "step_statistics.csv",
        "difficult_cases_csv": root / "tables" / "difficult_cases.csv",
        "application_heatmap_csv": root / "tables" / "application_heatmap.csv",
        "clean_slab_energies_csv": root / "tables" / "clean_slab_energies.csv",
        "gas_energies_csv": root / "tables" / "gas_energies.csv",
        "report_md": root / "reports" / "surface_screen_report.md",
    }


def _surface_site_names(host_name: str, slab: "ase.Atoms") -> str:
    try:
        return ", ".join(sorted(sites_for_host(host_name, slab)))
    except Exception as exc:  # pragma: no cover - retained for audit output
        return f"site-finder-error: {type(exc).__name__}: {exc}"


def surface_fingerprint_table(
    slabs: dict[str, "ase.Atoms"],
    slab_specs: list[SurfaceScreenSlabSpec]
    | tuple[SurfaceScreenSlabSpec, ...]
    | None = None,
) -> pd.DataFrame:
    """Describe the slabs before relaxation."""
    spec_by_name = {spec.name: spec for spec in slab_specs or ()}
    rows: list[dict[str, object]] = []
    for name, atoms in slabs.items():
        spec = spec_by_name.get(name)
        z = atoms.positions[:, 2]
        z_span = float(np.ptp(z))
        cellpar = atoms.cell.cellpar()
        z_cut = float(z.max() - 0.15 * z_span)
        top_symbols = sorted(set(np.asarray(atoms.get_chemical_symbols())[z >= z_cut]))
        rows.append(
            {
                "host": name,
                "material_class": spec.material_class if spec else "",
                "facet": spec.facet if spec else "",
                "miller_index": str(spec.miller_index) if spec else "",
                "builder": spec.builder_name if spec else "",
                "default_supercell": str(spec.default_supercell) if spec else "",
                "formula": _formula(atoms),
                "n_atoms": len(atoms),
                "cell_a_A": float(cellpar[0]),
                "cell_b_A": float(cellpar[1]),
                "cell_c_A": float(cellpar[2]),
                "slab_thickness_A": z_span,
                "approx_vacuum_A": float(cellpar[2] - z_span),
                "top_layer_species": ", ".join(top_symbols),
                "site_classes": _surface_site_names(name, atoms),
                "note": spec.note if spec else "",
            }
        )
    return pd.DataFrame(rows)


def _minimum_adsorbate_slab_distance(config: Configuration, slab_n: int) -> float:
    slab = config.atoms[:slab_n]
    ads = config.atoms[slab_n:]
    distances = np.linalg.norm(
        ads.positions[:, None, :] - slab.positions[None, :, :],
        axis=2,
    )
    return float(distances.min())


def audit_initial_configs(
    configs: list[Configuration],
    slab_atom_counts: dict[str, int],
    *,
    min_distance_warning_A: float = 1.0,
) -> pd.DataFrame:
    """Audit starting structures before any relaxation is trusted."""
    rows: list[dict[str, object]] = []
    for config in configs:
        slab_n = slab_atom_counts[config.host]
        slab = config.atoms[:slab_n]
        ads = config.atoms[slab_n:]
        top_z = float(slab.positions[:, 2].max())
        min_ads_z = float(ads.positions[:, 2].min())
        min_distance = _minimum_adsorbate_slab_distance(config, slab_n)
        active_atoms = int(sum(config.active_mask))
        status = "ok" if min_distance >= min_distance_warning_A else "review"
        rows.append(
            {
                "label": config.label,
                "pair": f"{config.adsorbate}/{config.host}",
                "host": config.host,
                "adsorbate": config.adsorbate,
                "start_site": config.site,
                "start_orientation": config.orientation,
                "rot_deg": config.rot_deg,
                "height_A_start": config.height,
                "n_atoms_total": len(config.atoms),
                "n_slab_atoms": slab_n,
                "n_adsorbate_atoms": len(ads),
                "n_active_atoms": active_atoms,
                "n_frozen_atoms": len(config.atoms) - active_atoms,
                "min_adsorbate_slab_distance_A": min_distance,
                "initial_adsorbate_clearance_above_top_z_A": min_ads_z - top_z,
                "audit_status": status,
            }
        )
    return pd.DataFrame(rows)


def _validate_audit(
    fingerprint_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    expected_counts: dict[str, int],
) -> None:
    expected = expected_counts
    errors: list[str] = []
    if len(fingerprint_df) != expected["slabs"]:
        errors.append(f"expected {expected['slabs']} slabs, got {len(fingerprint_df)}")
    if len(audit_df) != expected["adsorption_relaxations"]:
        errors.append(
            "expected "
            f"{expected['adsorption_relaxations']} starting structures, got {len(audit_df)}"
        )
    if audit_df["label"].duplicated().any():
        errors.append("starting-structure labels are not unique")
    pair_counts = audit_df.groupby(["host", "adsorbate"]).size()
    bad_pairs = pair_counts[pair_counts != expected["starts_per_pair"]]
    if len(bad_pairs):
        errors.append(f"pairs without exactly six starts: {bad_pairs.to_dict()}")
    if "review" in set(audit_df["audit_status"]):
        n_review = int((audit_df["audit_status"] == "review").sum())
        errors.append(f"{n_review} starting structures have very short contacts")
    if errors:
        raise ValueError("Surface-screen geometry audit failed: " + "; ".join(errors))


def write_surface_screen_audit_tables(
    output_root: str | Path,
    slabs: dict[str, "ase.Atoms"],
    configs: list[Configuration],
    *,
    slab_specs: list[SurfaceScreenSlabSpec]
    | tuple[SurfaceScreenSlabSpec, ...]
    | None = None,
    expected_counts: dict[str, int] | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write slab fingerprints and initial-geometry audit tables."""
    paths = surface_screen_output_paths(output_root)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    fingerprint_df = surface_fingerprint_table(slabs, slab_specs=slab_specs)
    slab_atom_counts = {host: len(slab) for host, slab in slabs.items()}
    audit_df = audit_initial_configs(configs, slab_atom_counts)
    if validate:
        if expected_counts is None:
            pair_counts = audit_df.groupby(["host", "adsorbate"]).size()
            starts_per_pair = int(pair_counts.iloc[0]) if len(pair_counts) else 0
            expected_counts = surface_screen_expected_counts(
                n_slabs=len(slabs),
                n_adsorbates=int(audit_df["adsorbate"].nunique()),
                starts_per_pair=starts_per_pair,
            )
        _validate_audit(fingerprint_df, audit_df, expected_counts)
    fingerprint_df.to_csv(paths["surface_fingerprints_csv"], index=False)
    audit_df.to_csv(paths["initial_geometry_audit_csv"], index=False)
    return fingerprint_df, audit_df


def first_converged_step_from_log(log_path: str | Path) -> dict[str, object]:
    """Return convergence statistics from a trajectory log CSV."""
    path = Path(log_path)
    if not str(log_path).strip() or not path.is_file():
        return {
            "trajectory_log_exists": False,
            "n_logged_frames": 0,
            "first_converged_step": np.nan,
            "final_logged_step": np.nan,
            "final_logged_energy_eV": np.nan,
            "final_logged_max_force_eV_A": np.nan,
            "final_logged_free_max_force_eV_A": np.nan,
        }
    df = pd.read_csv(path)
    if df.empty:
        return {
            "trajectory_log_exists": True,
            "n_logged_frames": 0,
            "first_converged_step": np.nan,
            "final_logged_step": np.nan,
            "final_logged_energy_eV": np.nan,
            "final_logged_max_force_eV_A": np.nan,
            "final_logged_free_max_force_eV_A": np.nan,
        }
    final = df.iloc[-1]
    converged_rows = df[df["converged"].astype(str).str.lower().isin(["true", "1"])]
    first_step = (
        float(converged_rows.iloc[0]["step"]) if len(converged_rows) else np.nan
    )
    return {
        "trajectory_log_exists": True,
        "n_logged_frames": int(len(df)),
        "first_converged_step": first_step,
        "final_logged_step": float(final["step"]),
        "final_logged_energy_eV": float(final["energy_eV"]),
        "final_logged_max_force_eV_A": float(final["max_force_eV_A"]),
        "final_logged_free_max_force_eV_A": float(final["free_max_force_eV_A"]),
    }


def build_step_statistics(
    results_df: pd.DataFrame,
    *,
    green_step_max: int = 200,
    yellow_step_max: int = 500,
    force_threshold_eV_A: float = 0.05,
) -> pd.DataFrame:
    """Attach trajectory-log convergence statistics to result rows."""
    rows: list[dict[str, object]] = []
    for _, row in results_df.iterrows():
        stats = first_converged_step_from_log(row.get("trajectory_log_path", ""))
        optimizer_steps = int(row.get("optimizer_nsteps", 0))
        final_result_force = row.get("max_force_eV_A", np.nan)
        final_force_over_threshold = (
            pd.notna(final_result_force)
            and float(final_result_force) > force_threshold_eV_A
        )
        if not bool(row.get("converged", False)) or final_force_over_threshold:
            step_status = "red"
        elif optimizer_steps <= green_step_max:
            step_status = "green"
        elif optimizer_steps <= yellow_step_max:
            step_status = "yellow"
        else:
            step_status = "red"
        rows.append(
            {
                "label": row["label"],
                "pair": row["pair"],
                "host": row["host"],
                "adsorbate": row["adsorbate"],
                "optimizer_nsteps": optimizer_steps,
                "step_status": step_status,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def summarize_surface_screen_pairs(
    results_df: pd.DataFrame,
    step_statistics_df: pd.DataFrame | None = None,
    *,
    exclude_red_step_status: bool = True,
) -> pd.DataFrame:
    """Summarize each adsorbate/surface search into one row."""
    df = results_df.copy()
    if step_statistics_df is not None and not step_statistics_df.empty:
        df = df.merge(
            step_statistics_df[["label", "step_status", "first_converged_step"]],
            on="label",
            how="left",
        )
    else:
        df["step_status"] = np.where(df["converged"], "unknown", "red")
        df["first_converged_step"] = np.nan

    rows: list[dict[str, object]] = []
    for (host, adsorbate), group in df.groupby(["host", "adsorbate"], sort=False):
        reliable = group[group["reliable_for_minimum"].astype(bool)]
        eligible = reliable
        if exclude_red_step_status:
            eligible = reliable[~reliable["step_status"].eq("red")]
        if eligible.empty:
            eligible = reliable
        winner = (
            eligible.loc[eligible["E_ads_eV"].idxmin()]
            if len(eligible)
            else group.loc[group["E_ads_eV"].idxmin()]
        )
        rows.append(
            {
                "pair": f"{adsorbate}/{host}",
                "host": host,
                "adsorbate": adsorbate,
                "n_starting_geometries": int(len(group)),
                "n_converged": int(group["converged"].sum()),
                "n_reliable": int(group["reliable_for_minimum"].sum()),
                "n_green": int(group["step_status"].eq("green").sum()),
                "n_yellow": int(group["step_status"].eq("yellow").sum()),
                "n_red": int(group["step_status"].eq("red").sum()),
                "best_label": winner["label"],
                "best_start_site": winner["start_site"],
                "best_start_orientation": winner["start_orientation"],
                "best_final_site": winner["final_site"],
                "best_binding_atom_symbol": winner["binding_atom_symbol"],
                "best_E_ads_eV": float(winner["E_ads_eV"]),
                "best_optimizer_nsteps": int(winner["optimizer_nsteps"]),
                "best_step_status": winner["step_status"],
                "best_trajectory_path": winner.get("trajectory_path", ""),
                "best_trajectory_log_path": winner.get("trajectory_log_path", ""),
                "min_optimizer_nsteps": int(group["optimizer_nsteps"].min()),
                "median_optimizer_nsteps": float(group["optimizer_nsteps"].median()),
                "max_optimizer_nsteps": int(group["optimizer_nsteps"].max()),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["rank_within_adsorbate"] = (
            summary.groupby("adsorbate")["best_E_ads_eV"]
            .rank(method="min", ascending=True)
            .astype(int)
        )
    return summary


def build_application_heatmap(
    pair_summary_df: pd.DataFrame,
    *,
    adsorbate_hints: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a compact heatmap-ready table from pair summaries."""
    hints = adsorbate_hints or {}
    rows: list[dict[str, object]] = []
    water_by_host = {
        row["host"]: float(row["best_E_ads_eV"])
        for _, row in pair_summary_df[pair_summary_df["adsorbate"].eq("H2O")].iterrows()
    }
    methanol_by_host = {
        row["host"]: float(row["best_E_ads_eV"])
        for _, row in pair_summary_df[
            pair_summary_df["adsorbate"].eq("CH3OH")
        ].iterrows()
    }
    for _, row in pair_summary_df.iterrows():
        host = row["host"]
        methanol_minus_water = np.nan
        if host in water_by_host and host in methanol_by_host:
            methanol_minus_water = methanol_by_host[host] - water_by_host[host]
        rows.append(
            {
                "host": host,
                "adsorbate": row["adsorbate"],
                "application_hint": hints.get(row["adsorbate"], ""),
                "best_E_ads_eV": float(row["best_E_ads_eV"]),
                "rank_within_adsorbate": int(row["rank_within_adsorbate"]),
                "best_final_site": row["best_final_site"],
                "n_starting_geometries": int(row["n_starting_geometries"]),
                "n_red": int(row["n_red"]),
                "methanol_minus_water_Eads_eV": methanol_minus_water,
            }
        )
    return pd.DataFrame(rows)


def build_difficult_cases(
    results_df: pd.DataFrame,
    step_statistics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Collect configurations that need manual review."""
    merged = results_df.merge(
        step_statistics_df[["label", "step_status", "first_converged_step"]],
        on="label",
        how="left",
    )
    difficult = merged[
        merged["step_status"].eq("red")
        | ~merged["converged"].astype(bool)
        | ~merged["geometry_status"].eq("adsorbed")
    ].copy()
    if difficult.empty:
        return difficult
    return difficult.sort_values(
        ["step_status", "optimizer_nsteps", "E_ads_eV"],
        ascending=[False, False, True],
    )


def load_surface_screen_tables(output_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load generated surface-screen tables for notebook display."""
    paths = surface_screen_output_paths(output_root)
    table_keys = [
        "surface_fingerprints_csv",
        "initial_geometry_audit_csv",
        "adsorption_results_csv",
        "pair_summary_csv",
        "batch_summary_csv",
        "step_statistics_csv",
        "difficult_cases_csv",
        "application_heatmap_csv",
        "clean_slab_energies_csv",
        "gas_energies_csv",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for key in table_keys:
        path = paths[key]
        if path.exists():
            tables[key.removesuffix("_csv")] = pd.read_csv(path)
    return tables


def _max_force_from_result(result: object | None) -> float:
    if result is None or not hasattr(result, "forces"):
        return np.nan
    forces = np.asarray(getattr(result, "forces"), dtype=float).reshape(-1, 3)
    return float(np.max(np.linalg.norm(forces, axis=1)))


def _result_steps(result: object | None) -> int | None:
    if result is None:
        return None
    if hasattr(result, "optimizer_nsteps"):
        return int(getattr(result, "optimizer_nsteps"))
    if hasattr(result, "num_optimization_steps"):
        return int(getattr(result, "num_optimization_steps"))
    return None


def write_surface_screen_result_artifacts(
    *,
    output_root: str | Path,
    pair_results: dict[tuple[str, str], pd.DataFrame],
    adsorption_results_df: pd.DataFrame,
    initial_geometry_audit: pd.DataFrame,
    summary_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    batch_summary_df: pd.DataFrame,
    step_statistics_df: pd.DataFrame,
    surface_pair_summary_df: pd.DataFrame,
    application_heatmap_df: pd.DataFrame,
    difficult_cases_df: pd.DataFrame,
    host_names: list[str],
    host_relaxed: dict[str, "ase.Atoms"],
    host_energies_eV: dict[str, float],
    host_compositions: dict[str, str],
    host_miller_indices: dict[str, tuple[int, int, int]],
    adsorbates: list[str],
    gas_energies_eV: dict[str, float],
    gas_reference_atoms: Callable[[str], "ase.Atoms"],
    metadata: dict[str, object],
    clean_results: list[object] | None = None,
    gas_results: list[object] | None = None,
    gas_relaxed: dict[str, "ase.Atoms"] | None = None,
) -> list[Path]:
    """Write result tables, structure snapshots, report, and run metadata.

    The notebook owns the chemistry and analysis. This helper only handles the
    repetitive file-writing work so the final tutorial cells stay readable.
    """
    from ase.io import write as ase_write

    paths = surface_screen_output_paths(output_root)
    for key in [
        "tables",
        "reports",
        "clean_structures",
        "gas_structures",
        "initial_structures",
        "relaxed_structures",
        "raw",
        "figures",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)

    artifact_paths = surface_screen_result_artifact_paths(output_root, pair_results)
    pair_result_paths = artifact_paths["pair_result_paths"]
    for path, ((_host, _adsorbate), df) in zip(pair_result_paths, pair_results.items()):
        df.to_csv(path, index=False)

    summary_path = artifact_paths["summary_validation_csv"]
    bias_path = artifact_paths["adsorbml_bias_csv"]
    summary_df.to_csv(summary_path, index=False)
    bias_df.to_csv(bias_path, index=False)
    adsorption_results_df.to_csv(paths["adsorption_results_csv"], index=False)
    initial_geometry_audit.to_csv(paths["initial_geometry_audit_csv"], index=False)
    batch_summary_df.to_csv(paths["batch_summary_csv"], index=False)
    step_statistics_df.to_csv(paths["step_statistics_csv"], index=False)
    surface_pair_summary_df.to_csv(paths["pair_summary_csv"], index=False)
    application_heatmap_df.to_csv(paths["application_heatmap_csv"], index=False)
    difficult_cases_df.to_csv(paths["difficult_cases_csv"], index=False)

    clean_by_host = dict(zip(host_names, clean_results or []))
    clean_rows: list[dict[str, object]] = []
    for host in host_names:
        atoms = host_relaxed[host]
        result = clean_by_host.get(host)
        structure_path = paths["clean_structures"] / f"clean_{_safe(host)}.extxyz"
        ase_write(structure_path, atoms)
        clean_rows.append(
            {
                "host": host,
                "energy_eV": host_energies_eV[host],
                "converged": bool(getattr(result, "converged", False))
                if result is not None
                else None,
                "optimizer_nsteps": _result_steps(result),
                "max_force_eV_A": _max_force_from_result(result),
                "structure_path": str(structure_path),
            }
        )
    pd.DataFrame(clean_rows).to_csv(paths["clean_slab_energies_csv"], index=False)

    gas_by_adsorbate = dict(zip(adsorbates, gas_results or []))
    gas_relaxed = gas_relaxed or {}
    gas_rows: list[dict[str, object]] = []
    for adsorbate in adsorbates:
        result = gas_by_adsorbate.get(adsorbate)
        atoms = gas_relaxed.get(adsorbate, gas_reference_atoms(adsorbate))
        structure_path = paths["gas_structures"] / f"gas_{_safe(adsorbate)}.extxyz"
        ase_write(structure_path, atoms)
        gas_rows.append(
            {
                "adsorbate": adsorbate,
                "energy_eV": gas_energies_eV[adsorbate],
                "converged": bool(getattr(result, "converged", False))
                if result is not None
                else None,
                "optimizer_nsteps": _result_steps(result),
                "max_force_eV_A": _max_force_from_result(result),
                "structure_path": str(structure_path),
            }
        )
    pd.DataFrame(gas_rows).to_csv(paths["gas_energies_csv"], index=False)

    surface_fingerprint_rows = [
        {
            "host": host,
            "composition": host_compositions[host],
            "miller_index": str(host_miller_indices[host]),
            "n_atoms": len(host_relaxed[host]),
        }
        for host in host_names
    ]
    pd.DataFrame(surface_fingerprint_rows).to_csv(
        paths["surface_fingerprints_csv"], index=False
    )

    report_path = paths["report_md"]
    report_path.write_text(
        (
            "# Surface-screen run\n\n"
            f"Run scope: `{metadata.get('run_scope', '')}`\n\n"
            f"Adsorption structures: `{len(adsorption_results_df)}`\n\n"
            f"Toolkit model: `{metadata.get('toolkit_model_label', '')}`\n"
        ),
        encoding="utf-8",
    )

    metadata_path = paths["metadata"]
    metadata_out = dict(metadata)
    metadata_out.update(
        {
            "surface_screen_root": str(output_root),
            "pair_result_paths": [str(path) for path in pair_result_paths],
            "summary_path": str(summary_path),
            "bias_path": str(bias_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    saved_paths = [
        *pair_result_paths,
        summary_path,
        bias_path,
        metadata_path,
        paths["adsorption_results_csv"],
        paths["report_md"],
        paths["surface_fingerprints_csv"],
        paths["clean_slab_energies_csv"],
        paths["gas_energies_csv"],
        paths["batch_summary_csv"],
        paths["step_statistics_csv"],
        paths["pair_summary_csv"],
        paths["application_heatmap_csv"],
        paths["difficult_cases_csv"],
    ]
    return [Path(path) for path in saved_paths]
