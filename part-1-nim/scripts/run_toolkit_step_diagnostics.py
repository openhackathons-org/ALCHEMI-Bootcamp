#!/usr/bin/env python3
"""Run CO/Cu(111) Toolkit convergence diagnostics with a 200-step cap."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import time

import nbformat
import numpy as np
import pandas as pd


PART1 = Path(__file__).resolve().parents[1]
ROOT = PART1.parent
sys.path.insert(0, str(PART1))

from helpers import (  # noqa: E402
    ADSORBATE_ORIENTATIONS,
    ADSORBML_REFERENCES,
    MACE_MPA0_OC157_MAD_EV,
    RelaxationBackendConfig,
    ase_to_atomic_data,
    atomic_data_to_ase,
    build_co,
    build_config_grid,
    build_cu111_slab,
    build_pair_results_table,
    get_relaxation_backend,
    make_active_mask,
)


HOST = "Cu(111)"
ADSORBATE = "CO"
SITES = ["top", "bridge", "fcc", "hcp"]
ROTATIONS = (0.0, 60.0)
HEIGHTS = (1.6, 1.8, 2.0, 2.2, 2.4)
DIAGNOSTIC_N_STEPS = int(os.environ.get("DIAGNOSTIC_N_STEPS", "200"))
FAILURE_RERUN_N_STEPS = int(os.environ.get("FAILURE_RERUN_N_STEPS", "5000"))
FMAX = float(os.environ.get("TOOLKIT_FMAX", "0.05"))

REPORT_DIR = PART1 / "outputs" / "reports"
TABLE_DIR = PART1 / "outputs" / "tables"
NOTEBOOK_OUTPUTS = [
    PART1 / "outputs" / "alchemi-toolkit-corrected-co-cu-200.executed.ipynb",
    PART1 / "outputs" / "alchemi-toolkit-expanded-co-cu.executed.ipynb",
]


def _safe(name: str) -> str:
    return name.replace("(", "_").replace(")", "").replace(",", "_").replace("/", "_")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _nearest_ads_slab_distance(config, slab_n: int) -> float:
    slab = config.atoms[:slab_n]
    ads = config.atoms[slab_n:]
    distances = np.linalg.norm(
        ads.positions[:, None, :] - slab.positions[None, :, :],
        axis=2,
    )
    return float(distances.min())


def _adsorbate_bond_length(atoms, slab_n: int) -> float | None:
    ads = atoms[slab_n:]
    if len(ads) != 2:
        return None
    return float(np.linalg.norm(ads.positions[0] - ads.positions[1]))


def _minimum_image_xy_delta(p_xy: np.ndarray, q_xy: np.ndarray, cell: np.ndarray) -> float:
    delta = np.asarray(p_xy, dtype=float) - np.asarray(q_xy, dtype=float)
    basis = np.vstack([cell[0, :2], cell[1, :2]]).T
    try:
        frac = np.linalg.solve(basis, delta)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(delta))
    frac -= np.round(frac)
    return float(np.linalg.norm(basis @ frac))


def _lateral_adsorbate_drift(initial_atoms, final_atoms, slab_n: int) -> float:
    initial_ads_xy = initial_atoms[slab_n:].positions[:, :2].mean(axis=0)
    final_ads_xy = final_atoms[slab_n:].positions[:, :2].mean(axis=0)
    return _minimum_image_xy_delta(final_ads_xy, initial_ads_xy, final_atoms.cell.array)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cell_text(cell) -> str:
    texts: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            texts.append(output.get("text", ""))
        elif output.get("output_type") in {"execute_result", "display_data"}:
            data = output.get("data", {})
            text = data.get("text/plain")
            if isinstance(text, str):
                texts.append(text)
            elif isinstance(text, list):
                texts.append("".join(text))
    return "\n".join(texts)


def _notebook_timing_summary(path: Path) -> dict[str, float | str | bool | None]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    nb = nbformat.read(path, as_version=4)
    first_start = None
    last_end = None
    relaxation_s = None
    clean_slab_s = None
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        execution = cell.get("metadata", {}).get("execution", {})
        start = execution.get("iopub.execute_input")
        end = execution.get("shell.execute_reply") or execution.get("iopub.status.idle")
        if not start or not end:
            continue
        start_dt = _parse_time(start)
        end_dt = _parse_time(end)
        first_start = start_dt if first_start is None else min(first_start, start_dt)
        last_end = end_dt if last_end is None else max(last_end, end_dt)
        seconds = (end_dt - start_dt).total_seconds()
        text = _cell_text(cell)
        if "Relaxed 80 configurations" in text:
            relaxation_s = seconds
        if "E_host (eV)" in cell.source and "HOST_RELAXED" in cell.source:
            clean_slab_s = seconds
    total_s = None
    if first_start is not None and last_end is not None:
        total_s = (last_end - first_start).total_seconds()
    return {
        "path": str(path),
        "exists": True,
        "total_runtime_s": total_s,
        "batch_relaxation_cell_s": relaxation_s,
        "clean_slab_cell_s": clean_slab_s,
    }


def _select_notebook_output() -> Path:
    for path in NOTEBOOK_OUTPUTS:
        if path.exists():
            return path
    return NOTEBOOK_OUTPUTS[-1]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _relax_one(backend, payload, label: str):
    start = time.perf_counter()
    reply = backend.relax([payload], label=label, cellopt=False)
    elapsed_s = time.perf_counter() - start
    return reply.atoms[0], elapsed_s


def _build_backend(n_steps: int):
    device = os.environ.get("TOOLKIT_DEVICE", "cuda")
    if device == "auto":
        device = "cuda"
    return get_relaxation_backend(
        RelaxationBackendConfig(
            name="toolkit",
            toolkit_checkpoint=os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
            toolkit_device=device,
            toolkit_dtype=os.environ.get("TOOLKIT_DTYPE", "float32"),
            toolkit_enable_cueq=_env_bool("TOOLKIT_ENABLE_CUEQ", False),
            toolkit_compile_model=_env_bool("TOOLKIT_COMPILE_MODEL", False),
            toolkit_n_steps=n_steps,
            toolkit_fmax=FMAX,
            toolkit_require_d3bj=False,
            toolkit_d3bj=None,
        )
    )


def _series_value(row: pd.Series, name: str, default=None):
    return row[name] if name in row.index else default


def _format_seconds(seconds: float | int | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "not recorded"
    seconds_f = float(seconds)
    if seconds_f < 60:
        return f"{seconds_f:.1f} s"
    return f"{seconds_f / 60.0:.1f} min"


def _write_report(
    *,
    report_path: Path,
    diagnostic_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    clean_row: dict[str, object],
    gas_row: dict[str, object],
    metadata: dict[str, object],
    notebook_timing: dict[str, object],
    notebook_run_metadata: dict[str, object],
    prior_batch_df: pd.DataFrame | None,
    failure_rerun_df: pd.DataFrame | None,
) -> None:
    ref = ADSORBML_REFERENCES[(HOST, ADSORBATE)]
    reliable = diagnostic_df[diagnostic_df["reliable_for_minimum"]]
    winner = reliable.loc[reliable["E_bind (eV)"].idxmin()]
    top_c = reliable[
        (reliable["final_site"] == "top")
        & (reliable["binding_atom_symbol"] == "C")
    ]
    top_best = top_c.loc[top_c["E_bind (eV)"].idxmin()] if not top_c.empty else None
    failed = diagnostic_df[~diagnostic_df["converged"]]
    desorbed = diagnostic_df[diagnostic_df["geometry_status"] != "adsorbed"]
    drifted = diagnostic_df[diagnostic_df["lateral_adsorbate_drift_A"] > 2.5]
    max_steps = int(diagnostic_df["optimizer_nsteps"].max())
    median_steps = float(diagnostic_df["optimizer_nsteps"].median())
    batch_min = None
    if prior_batch_df is not None and not prior_batch_df.empty:
        prior_reliable = prior_batch_df[prior_batch_df["reliable_for_minimum"]]
        if not prior_reliable.empty:
            batch_min = prior_reliable.loc[prior_reliable["E_bind (eV)"].idxmin()]
    notebook_steps = notebook_run_metadata.get("toolkit_n_steps")
    notebook_steps_text = (
        f"n_steps cap {notebook_steps}"
        if notebook_steps is not None
        else "step cap not recorded"
    )

    lines = [
        "# Toolkit CO/Cu(111) Step Diagnostics",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        (
            "This report checks whether the CO/Cu(111) Toolkit run needs "
            "`TOOLKIT_N_STEPS=5000`, using the same 3x3 four-layer Cu(111) slab, "
            "bottom-half frozen atoms, four named fcc(111) sites, two CO orientations, "
            "two rotations, and five initial heights used by the expanded notebook."
        ),
        "",
        "The diagnostic is explicit native Toolkit MACE-only. D3(BJ) is disabled "
        "because verified BGR parity damping parameters are not yet available.",
        "",
        "During this investigation the fcc(111) hollow-site generator was fixed. "
        "The current fcc and hcp starts are enumerated from the periodic top layer; "
        "hcp is the hollow with a second-layer atom below it. Tables generated "
        "before this fix are superseded by the outputs listed here.",
        "",
        "## Runtime",
        "",
        f"- Corrected notebook total runtime: {_format_seconds(notebook_timing.get('total_runtime_s'))}.",
        f"- Corrected notebook 80-configuration batch cell: {_format_seconds(notebook_timing.get('batch_relaxation_cell_s'))}.",
        f"- This 200-step individual diagnostic total runtime: {_format_seconds(metadata['diagnostic_runtime_s'])}.",
        f"- Clean Cu(111) relaxation: {_format_seconds(clean_row['runtime_s'])}, "
        f"{clean_row['optimizer_nsteps']} steps, converged={clean_row['converged']}.",
        f"- Gas CO relaxation: {_format_seconds(gas_row['runtime_s'])}, "
        f"{gas_row['optimizer_nsteps']} steps, converged={gas_row['converged']}.",
        "",
        "## Convergence and Geometry",
        "",
        f"- Configurations tested: {len(diagnostic_df)}.",
        f"- Converged within {DIAGNOSTIC_N_STEPS} steps: {int(diagnostic_df['converged'].sum())}/{len(diagnostic_df)}.",
        f"- Maximum individual optimizer steps: {max_steps}; median: {median_steps:.1f}.",
        f"- Non-adsorbed or dissociated outcomes: {len(desorbed)}.",
        f"- Lateral drift greater than 2.5 A: {len(drifted)}.",
        f"- Initial nearest adsorbate-slab clearance range: "
        f"{diagnostic_df['initial_nearest_ads_slab_distance_A'].min():.3f} to "
        f"{diagnostic_df['initial_nearest_ads_slab_distance_A'].max():.3f} A.",
        f"- Final nearest surface distance range: "
        f"{diagnostic_df['nearest_surface_distance_A'].min():.3f} to "
        f"{diagnostic_df['nearest_surface_distance_A'].max():.3f} A.",
        "",
        "## Energies",
        "",
        f"- E_clean(Cu slab): {float(clean_row['energy_eV']):.6f} eV.",
        f"- E_gas(CO): {float(gas_row['energy_eV']):.6f} eV.",
        f"- 200-step MACE-only minimum: {float(winner['E_bind (eV)']):.6f} eV, "
        f"start={winner['start_site']}/{winner['start_orientation']}/h={winner['height_A_start']}, "
        f"final={winner['final_site']}({winner['binding_atom_symbol']}-down), "
        f"steps={int(winner['optimizer_nsteps'])}.",
    ]
    if top_best is not None:
        lines.append(
            f"- Best final top C-down structure: {float(top_best['E_bind (eV)']):.6f} eV, "
            f"steps={int(top_best['optimizer_nsteps'])}."
        )
    if batch_min is not None:
        lines.append(
            f"- Corrected notebook batch minimum ({notebook_steps_text}): "
            f"{float(batch_min['E_bind (eV)']):.6f} eV, "
            f"final={batch_min['final_site']}({batch_min['binding_atom_symbol']}-down)."
        )
    lines.extend([
        "",
        "## Literature and Apples-to-Apples Status",
        "",
        f"- Context reference in the tutorial: {ref.E_bind_eV:.3f} eV, site={ref.binding_site}, "
        f"scope={ref.reference_scope}, source={ref.ref}.",
        f"- Difference from context reference for the 200-step minimum: "
        f"{float(winner['E_bind (eV)']) - float(ref.E_bind_eV):+.3f} eV "
        f"({abs(float(winner['E_bind (eV)']) - float(ref.E_bind_eV)) / MACE_MPA0_OC157_MAD_EV:.2f} x "
        f"the MACE-MPA-0 OC157 relative-energy MAD of {MACE_MPA0_OC157_MAD_EV:.2f} eV).",
        "- Site agreement: failed for the MACE-only minimum because it relaxes to an fcc hollow; "
        "the high-level surface-science checkpoint for CO/Cu(111) is top-site adsorption.",
        "- This is not a strict DFT parity comparison. Strict parity still requires the exact "
        "reference row with matching slab size, coverage, frozen layers, exchange-correlation "
        "functional, dispersion, and BGR D3(BJ) settings.",
        "",
        "The discrepancy is scientifically plausible as a model/functional warning, not a "
        "validated discovery: semilocal DFT is known to exhibit the CO adsorption puzzle on "
        "Cu(111), where hollow sites can be over-stabilized relative to the experimentally "
        "preferred top site. The current MACE-only result reproduces that qualitative warning.",
        "",
        "## Per-Start Summary",
        "",
        summary_df.to_markdown(index=False),
    ])
    if not failed.empty:
        lines.extend([
            "",
            "## Failed 200-Step Structures",
            "",
            failed[
                [
                    "label",
                    "start_site",
                    "start_orientation",
                    "height_A_start",
                    "optimizer_nsteps",
                    "max_force_eV_A",
                    "geometry_status",
                    "E_bind (eV)",
                ]
            ].to_markdown(index=False),
        ])
    if failure_rerun_df is not None and not failure_rerun_df.empty:
        lines.extend([
            "",
            "## 5000-Step Failure Reruns",
            "",
            failure_rerun_df.to_markdown(index=False),
        ])
    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Diagnostic table: `{metadata['diagnostic_csv']}`",
        f"- Per-start summary: `{metadata['summary_csv']}`",
        f"- Metadata: `{metadata['metadata_json']}`",
        "",
        "## Sources for Domain Review",
        "",
        "- MACE foundation-model SI/PDF: https://database.ouro.foundation/storage/v1/object/public/public-files/847f4445-78ee-41b1-913b-5bd155c71b13/fbf2ab55-1086-4988-ae8e-4e3d44462764.pdf",
        "- CO/Cu(111) GGA vs experiment/top-site puzzle: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.195440",
        "- QMC/DMC CO adsorption context: https://labs.iams.sinica.edu.tw/project/cmw/publications/quantum-monte-carlo-studies-co-adsorption-transition-metal-surfaces",
        "- OC20 DFT reference context: https://doi.org/10.1021/acscatal.0c04525",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_rows(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def main() -> int:
    start_total = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    notebook_timing = _notebook_timing_summary(_select_notebook_output())
    notebook_run_metadata = _read_json(TABLE_DIR / "run_metadata.json")
    backend = _build_backend(DIAGNOSTIC_N_STEPS)

    slab = build_cu111_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(3, 3, 1),
    )
    slab_mask = make_active_mask(slab, bottom_fraction=0.5)
    clean_result, clean_runtime = _relax_one(
        backend,
        ase_to_atomic_data(
            slab,
            structure_id=f"clean_{_safe(HOST)}_diagnostic",
            active_mask=slab_mask,
        ),
        label=f"diagnostic_clean_{_safe(HOST)}",
    )
    clean_atoms = atomic_data_to_ase(clean_result)
    clean_row = {
        "energy_eV": float(clean_result.energy),
        "converged": bool(clean_result.converged),
        "optimizer_nsteps": int(clean_result.optimizer_nsteps),
        "runtime_s": clean_runtime,
    }

    gas = build_co(ADSORBATE_ORIENTATIONS[ADSORBATE][0])
    gas.set_cell([15.0, 15.0, 15.0])
    gas.set_pbc(True)
    gas.center()
    gas_result, gas_runtime = _relax_one(
        backend,
        ase_to_atomic_data(gas, structure_id=f"gas_{ADSORBATE}_diagnostic"),
        label=f"diagnostic_gas_{ADSORBATE}",
    )
    gas_row = {
        "energy_eV": float(gas_result.energy),
        "converged": bool(gas_result.converged),
        "optimizer_nsteps": int(gas_result.optimizer_nsteps),
        "runtime_s": gas_runtime,
    }

    configs = build_config_grid(
        host_name=HOST,
        slab=clean_atoms,
        adsorbate_name=ADSORBATE,
        sites_filter=SITES,
        orientations_filter=None,
        rotations_deg=ROTATIONS,
        heights_A=HEIGHTS,
        frozen_fraction=0.5,
    )

    opt_results = []
    runtime_rows = []
    for idx, config in enumerate(configs, start=1):
        print(f"[{idx:02d}/{len(configs)}] {config.label}", flush=True)
        result, runtime_s = _relax_one(
            backend,
            ase_to_atomic_data(
                config.atoms,
                structure_id=f"diagnostic_{config.label}",
                active_mask=config.active_mask,
            ),
            label=f"diagnostic_{idx:03d}_{_safe(config.label)}",
        )
        opt_results.append(result)
        final_atoms = atomic_data_to_ase(result)
        runtime_rows.append(
            {
                "label": config.label,
                "runtime_s": runtime_s,
                "initial_nearest_ads_slab_distance_A": _nearest_ads_slab_distance(
                    config,
                    len(clean_atoms),
                ),
                "initial_adsorbate_bond_A": _adsorbate_bond_length(
                    config.atoms,
                    len(clean_atoms),
                ),
                "final_adsorbate_bond_A": _adsorbate_bond_length(
                    final_atoms,
                    len(clean_atoms),
                ),
                "lateral_adsorbate_drift_A": _lateral_adsorbate_drift(
                    config.atoms,
                    final_atoms,
                    len(clean_atoms),
                ),
            }
        )

    pair_df = build_pair_results_table(
        host=HOST,
        adsorbate=ADSORBATE,
        configs=configs,
        opt_results=opt_results,
        clean_slab_atoms=clean_atoms,
        e_clean_slab_ev=float(clean_result.energy),
        e_gas_ads_ev=float(gas_result.energy),
        backend="toolkit",
    )
    runtime_df = pd.DataFrame(runtime_rows)
    diagnostic_df = pair_df.merge(runtime_df, on="label", how="left")
    diagnostic_df.insert(0, "diagnostic_n_steps_cap", DIAGNOSTIC_N_STEPS)

    summary_df = (
        diagnostic_df.groupby(["start_site", "start_orientation"], as_index=False)
        .agg(
            n_configs=("label", "count"),
            n_converged=("converged", "sum"),
            max_optimizer_nsteps=("optimizer_nsteps", "max"),
            median_optimizer_nsteps=("optimizer_nsteps", "median"),
            min_E_bind_eV=("E_bind (eV)", "min"),
            median_E_bind_eV=("E_bind (eV)", "median"),
            max_lateral_drift_A=("lateral_adsorbate_drift_A", "max"),
            n_non_adsorbed=("geometry_status", lambda s: int((s != "adsorbed").sum())),
        )
        .sort_values(["start_site", "start_orientation"])
    )

    rerun_df = pd.DataFrame()
    failures = diagnostic_df[~diagnostic_df["converged"]]
    if not failures.empty:
        print(
            f"Rerunning {len(failures)} non-converged structures with "
            f"{FAILURE_RERUN_N_STEPS} steps.",
            flush=True,
        )
        rerun_backend = _build_backend(FAILURE_RERUN_N_STEPS)
        rerun_rows = []
        config_by_label = {config.label: config for config in configs}
        for _, failure in failures.iterrows():
            config = config_by_label[str(failure["label"])]
            result, runtime_s = _relax_one(
                rerun_backend,
                ase_to_atomic_data(
                    config.atoms,
                    structure_id=f"diagnostic_rerun_{config.label}",
                    active_mask=config.active_mask,
                ),
                label=f"diagnostic_rerun_{_safe(config.label)}",
            )
            rerun_pair_df = build_pair_results_table(
                host=HOST,
                adsorbate=ADSORBATE,
                configs=[config],
                opt_results=[result],
                clean_slab_atoms=clean_atoms,
                e_clean_slab_ev=float(clean_result.energy),
                e_gas_ads_ev=float(gas_result.energy),
                backend="toolkit",
            )
            row = rerun_pair_df.iloc[0].to_dict()
            row["rerun_runtime_s"] = runtime_s
            row["rerun_n_steps_cap"] = FAILURE_RERUN_N_STEPS
            rerun_rows.append(row)
        rerun_df = pd.DataFrame(rerun_rows)

    prior_batch_path = TABLE_DIR / "pair_results_CO_Cu_111.csv"
    prior_batch_df = pd.read_csv(prior_batch_path) if prior_batch_path.exists() else None

    diagnostic_csv = REPORT_DIR / "toolkit_co_cu111_step_diagnostics.csv"
    summary_csv = REPORT_DIR / "toolkit_co_cu111_step_summary.csv"
    rerun_csv = REPORT_DIR / "toolkit_co_cu111_step_failure_reruns.csv"
    metadata_json = REPORT_DIR / "toolkit_co_cu111_step_metadata.json"
    report_path = REPORT_DIR / "toolkit_co_cu111_step_report.md"

    diagnostic_df.to_csv(diagnostic_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    if not rerun_df.empty:
        rerun_df.to_csv(rerun_csv, index=False)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "toolkit",
        "toolkit_checkpoint": os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
        "toolkit_device": str(getattr(backend, "device", os.environ.get("TOOLKIT_DEVICE", "cuda"))),
        "toolkit_compile_model": _env_bool("TOOLKIT_COMPILE_MODEL", False),
        "toolkit_n_steps_cap": DIAGNOSTIC_N_STEPS,
        "toolkit_fmax": FMAX,
        "toolkit_d3bj_enabled": False,
        "toolkit_require_d3bj": False,
        "host": HOST,
        "adsorbate": ADSORBATE,
        "sites": SITES,
        "rotations_deg": ROTATIONS,
        "heights_A": HEIGHTS,
        "n_configurations": len(configs),
        "n_converged": int(diagnostic_df["converged"].sum()),
        "n_non_adsorbed": int((diagnostic_df["geometry_status"] != "adsorbed").sum()),
        "clean_slab": clean_row,
        "gas_adsorbate": gas_row,
        "notebook_timing": notebook_timing,
        "notebook_run_metadata": notebook_run_metadata,
        "diagnostic_runtime_s": time.perf_counter() - start_total,
        "diagnostic_csv": str(diagnostic_csv),
        "summary_csv": str(summary_csv),
        "failure_rerun_csv": str(rerun_csv) if not rerun_df.empty else None,
        "metadata_json": str(metadata_json),
        "report_path": str(report_path),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _write_report(
        report_path=report_path,
        diagnostic_df=diagnostic_df,
        summary_df=summary_df,
        clean_row=clean_row,
        gas_row=gas_row,
        metadata=metadata,
        notebook_timing=notebook_timing,
        notebook_run_metadata=notebook_run_metadata,
        prior_batch_df=prior_batch_df,
        failure_rerun_df=rerun_df if not rerun_df.empty else None,
    )

    print(f"Wrote {diagnostic_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
