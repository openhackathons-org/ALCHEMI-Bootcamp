#!/usr/bin/env python3
"""Run the notebook's full AdsorbML panel with native Toolkit, resumably."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from ase.build import molecule as ase_molecule
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART1))

from helpers import (  # noqa: E402
    ADSORBATE_ORIENTATIONS,
    ADSORBML_REFERENCES,
    MACE_MPA0_OC157_MAD_EV,
    OptimizationResult,
    RelaxationBackendConfig,
    ase_to_atomic_data,
    atomic_data_to_ase,
    build_alpha_alumina_0001_slab,
    build_config_grid,
    build_co,
    build_cu111_slab,
    build_h2o,
    build_methanol,
    build_pair_results_table,
    build_pd111_slab,
    get_relaxation_backend,
    make_active_mask,
    summarize_pair_validation,
)


HOST_BUILDERS = {
    "Cu(111)": lambda: build_cu111_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(3, 3, 1),
    ),
    "Pd(111)": lambda: build_pd111_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(3, 3, 1),
    ),
    "Al2O3(0001)": lambda: build_alpha_alumina_0001_slab(
        min_slab_size=8.0,
        min_vacuum_size=15.0,
        supercell=(2, 2, 1),
    ),
}
ADSORBATE_BUILDERS = {
    "CO": build_co,
    "H2O": build_h2o,
    "CH3OH": build_methanol,
}
ROTATIONS = (0.0, 60.0, 120.0)
HEIGHTS = (2.2,)
REPORT_DIR = PART1 / "outputs" / "reports"
OUTPUT_DIR = PART1 / "outputs" / "full_panel_toolkit"
CHUNK_DIR = OUTPUT_DIR / "chunks"
RAW_DIR = OUTPUT_DIR / "raw_results"
STRUCTURE_DIR = OUTPUT_DIR / "structures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=int(os.environ.get("TOOLKIT_N_STEPS", "200")))
    parser.add_argument("--failure-n-steps", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("FULL_PANEL_CHUNK_SIZE", "12")))
    parser.add_argument("--fmax", type=float, default=float(os.environ.get("TOOLKIT_FMAX", "0.05")))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe(name: str) -> str:
    return name.replace("(", "_").replace(")", "").replace(",", "_").replace("/", "_")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _build_backend(n_steps: int, fmax: float):
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
            toolkit_fmax=fmax,
            toolkit_require_d3bj=False,
            toolkit_d3bj=None,
        )
    )


def _relax_one(backend, atoms, *, label: str, active_mask: list[bool] | None = None):
    start = time.perf_counter()
    reply = backend.relax(
        [ase_to_atomic_data(atoms, structure_id=label, active_mask=active_mask)],
        label=label,
        cellopt=False,
    )
    return reply.atoms[0], time.perf_counter() - start


def _gas_atoms(name: str):
    if name == "H2O":
        atoms = ase_molecule("H2O")
    else:
        atoms = ADSORBATE_BUILDERS[name](ADSORBATE_ORIENTATIONS[name][0])
    atoms.set_cell([15.0, 15.0, 15.0])
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def _max_force(result) -> float:
    forces = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(forces, axis=1).max())


def _ensure_output_dirs() -> None:
    for directory in (
        OUTPUT_DIR,
        REPORT_DIR,
        CHUNK_DIR,
        RAW_DIR,
        STRUCTURE_DIR / "initial",
        STRUCTURE_DIR / "relaxed",
        STRUCTURE_DIR / "clean",
        STRUCTURE_DIR / "gas",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _result_to_json(result: OptimizationResult) -> dict[str, object]:
    return result.model_dump(mode="json")


def _load_result(path: Path) -> OptimizationResult:
    return OptimizationResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _write_result(path: Path, result: OptimizationResult) -> None:
    path.write_text(json.dumps(_result_to_json(result), indent=2), encoding="utf-8")


def _write_result_atoms(path: Path, result: OptimizationResult, *, label: str) -> None:
    atoms = atomic_data_to_ase(result)
    atoms.info["label"] = label
    atoms.info["energy_eV"] = float(result.energy)
    atoms.info["converged"] = bool(result.converged)
    atoms.info["optimizer_nsteps"] = int(result.optimizer_nsteps)
    atoms.arrays["forces"] = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    ase_write(path, atoms, format="extxyz")


def _write_initial_atoms(path: Path, atoms, *, label: str) -> None:
    initial = atoms.copy()
    initial.info["label"] = label
    ase_write(path, initial, format="extxyz")


def _chunk_cache_paths(chunk_label: str) -> dict[str, Path]:
    return {
        "raw": RAW_DIR / f"{chunk_label}.json",
        "csv": CHUNK_DIR / f"{chunk_label}.csv",
        "metadata": CHUNK_DIR / f"{chunk_label}.metadata.json",
    }


def _relax_configs_in_chunks(
    backend,
    configs,
    *,
    host: str,
    adsorbate: str,
    clean_slab_atoms,
    e_clean_slab_ev: float,
    e_gas_ads_ev: float,
    pair_label: str,
    chunk_size: int,
    force: bool,
) -> tuple[list[object], list[dict[str, object]]]:
    results: list[object] = []
    chunk_rows: list[dict[str, object]] = []
    for start_idx in range(0, len(configs), chunk_size):
        chunk = configs[start_idx:start_idx + chunk_size]
        chunk_index = start_idx // chunk_size + 1
        chunk_label = f"{pair_label}_chunk_{chunk_index:02d}"
        paths = _chunk_cache_paths(chunk_label)
        if paths["raw"].exists() and paths["csv"].exists() and not force:
            raw_results = json.loads(paths["raw"].read_text(encoding="utf-8"))
            chunk_results = [
                OptimizationResult.model_validate(item)
                for item in raw_results
            ]
            metadata = (
                json.loads(paths["metadata"].read_text(encoding="utf-8"))
                if paths["metadata"].exists()
                else {}
            )
            runtime_s = float(metadata.get("runtime_s", 0.0))
            print(f"  {chunk_label}: loaded cached chunk", flush=True)
        else:
            data_list = [
                ase_to_atomic_data(config.atoms, structure_id=config.label, active_mask=config.active_mask)
                for config in chunk
            ]
            started = time.perf_counter()
            reply = backend.relax(data_list, label=chunk_label, cellopt=False)
            runtime_s = time.perf_counter() - started
            chunk_results = reply.atoms
            paths["raw"].write_text(
                json.dumps([_result_to_json(result) for result in chunk_results], indent=2),
                encoding="utf-8",
            )
            chunk_df = build_pair_results_table(
                host=host,
                adsorbate=adsorbate,
                configs=chunk,
                opt_results=chunk_results,
                clean_slab_atoms=clean_slab_atoms,
                e_clean_slab_ev=e_clean_slab_ev,
                e_gas_ads_ev=e_gas_ads_ev,
                backend="toolkit",
            )
            chunk_df.insert(0, "chunk_label", chunk_label)
            chunk_df.to_csv(paths["csv"], index=False)
            paths["metadata"].write_text(
                json.dumps(
                    {
                        "chunk_label": chunk_label,
                        "pair": f"{adsorbate}/{host}",
                        "chunk_index": chunk_index,
                        "n_configs": len(chunk),
                        "runtime_s": runtime_s,
                        "optimizer_nsteps": max(int(result.optimizer_nsteps) for result in chunk_results),
                        "n_converged": sum(bool(result.converged) for result in chunk_results),
                        "raw_json": str(paths["raw"]),
                        "chunk_csv": str(paths["csv"]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            for config, result in zip(chunk, chunk_results):
                safe_label = _safe(config.label)
                _write_initial_atoms(
                    STRUCTURE_DIR / "initial" / f"{safe_label}.extxyz",
                    config.atoms,
                    label=config.label,
                )
                _write_result(
                    RAW_DIR / f"{safe_label}.json",
                    result,
                )
                _write_result_atoms(
                    STRUCTURE_DIR / "relaxed" / f"{safe_label}.extxyz",
                    result,
                    label=config.label,
                )
        results.extend(chunk_results)
        chunk_rows.append(
            {
                "pair": f"{adsorbate}/{host}",
                "chunk_label": chunk_label,
                "n_configs": len(chunk),
                "runtime_s": runtime_s,
                "optimizer_nsteps": max(int(result.optimizer_nsteps) for result in chunk_results),
                "n_converged": sum(bool(result.converged) for result in chunk_results),
                "chunk_csv": str(paths["csv"]),
                "raw_json": str(paths["raw"]),
            }
        )
        print(
            f"  {chunk_label}: {len(chunk)} configs, "
            f"{chunk_rows[-1]['n_converged']} converged, "
            f"{chunk_rows[-1]['optimizer_nsteps']} steps, {runtime_s:.1f}s",
            flush=True,
        )
    return results, chunk_rows


def _reference_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        ref = ADSORBML_REFERENCES.get((row["host"], row["adsorbate"]))
        ref_dict = asdict(ref) if ref is not None else {}
        rows.append({
            **row.to_dict(),
            "reference_method": ref_dict.get("method"),
            "reference_ref": ref_dict.get("ref"),
            "reference_doi": ref_dict.get("doi"),
            "reference_url": ref_dict.get("source_url"),
            "reference_functional": ref_dict.get("functional"),
            "reference_dispersion": ref_dict.get("dispersion"),
            "reference_coverage": ref_dict.get("coverage"),
            "reference_notes": ref_dict.get("notes"),
        })
    return pd.DataFrame(rows)


def _write_report(
    *,
    path: Path,
    metadata: dict[str, object],
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    full_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> None:
    reliable = full_df[full_df["reliable_for_minimum"]]
    minima = (
        reliable.sort_values("E_bind (eV)")
        .groupby(["host", "adsorbate"], as_index=False)
        .first()
    )
    display_cols = [
        "pair",
        "status",
        "MACE_site",
        "reference_site",
        "E_MACE_eV",
        "E_ref_eV",
        "delta_E_eV",
        "abs_delta_over_MAD",
        "reference_ref",
    ]
    lines = [
        "# Toolkit Full-Panel Adsorption Energies",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "This run covers the notebook full panel: Cu(111), Pd(111), and "
        "Al2O3(0001) crossed with CO, H2O, and CH3OH. The grid uses all "
        "notebook sites, all notebook orientations, rotations 0/60/120 deg, "
        "and an initial height of 2.2 A.",
        "",
        "The backend is native Toolkit MACE-MPA-0 only. D3(BJ) is disabled, "
        "so these numbers are not strict BGR/D3 or DFT parity values.",
        "",
        "## Runtime and Convergence",
        "",
        f"- Total runtime: {metadata['runtime_s'] / 60.0:.1f} min.",
        f"- Structures relaxed: {len(full_df)}.",
        f"- Reliable structures: {len(reliable)}/{len(full_df)}.",
        f"- Non-converged structures after final reruns: {int((~full_df['converged']).sum())}.",
        f"- Non-adsorbed/dissociated structures: {int((full_df['geometry_status'] != 'adsorbed').sum())}.",
        f"- Max optimizer steps in final table: {int(full_df['optimizer_nsteps'].max())}.",
        "",
        "## Pair Minima",
        "",
        minima[
            [
                "pair",
                "label",
                "E_bind (eV)",
                "final_site",
                "binding_atom_symbol",
                "optimizer_nsteps",
                "max_force_eV_A",
                "geometry_status",
            ]
        ].to_markdown(index=False),
        "",
        "## MACE vs Listed DFT/Literature References",
        "",
        comparison_df[display_cols].to_markdown(index=False),
        "",
        "All listed reference rows are currently `context`, not strict parity. "
        f"The MACE-MPA-0 OC157 relative-energy MAD used for scaling is {MACE_MPA0_OC157_MAD_EV:.2f} eV. "
        "Strict comparison still requires exact row-level DFT metadata: slab, "
        "coverage, frozen layers, functional, dispersion, and sign convention.",
        "",
        "## Chunk Timings",
        "",
        chunk_df.to_markdown(index=False),
        "",
        "## Output Files",
        "",
        f"- Full pair table: `{metadata['full_results_csv']}`",
        f"- Validation summary: `{metadata['summary_csv']}`",
        f"- Reference comparison: `{metadata['comparison_csv']}`",
        f"- Metadata: `{metadata['metadata_json']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    start_total = time.perf_counter()
    _ensure_output_dirs()

    backend = _build_backend(args.n_steps, args.fmax)
    rerun_backend = None

    hosts = {name: builder() for name, builder in HOST_BUILDERS.items()}
    clean_results = {}
    relaxed_hosts = {}
    clean_rows = []
    for host, slab in hosts.items():
        safe_host = _safe(host)
        raw_path = RAW_DIR / f"full_clean_{safe_host}.json"
        if raw_path.exists() and not args.force:
            print(f"Load clean slab cache: {host}", flush=True)
            result = _load_result(raw_path)
            runtime_s = 0.0
        else:
            print(f"Relax clean slab: {host}", flush=True)
            result, runtime_s = _relax_one(
                backend,
                slab,
                label=f"full_clean_{safe_host}",
                active_mask=make_active_mask(slab, bottom_fraction=0.5),
            )
            _write_result(raw_path, result)
            _write_result_atoms(
                STRUCTURE_DIR / "clean" / f"full_clean_{safe_host}.extxyz",
                result,
                label=f"full_clean_{host}",
            )
        clean_results[host] = result
        relaxed_hosts[host] = atomic_data_to_ase(result)
        clean_rows.append({
            "host": host,
            "energy_eV": float(result.energy),
            "converged": bool(result.converged),
            "optimizer_nsteps": int(result.optimizer_nsteps),
            "max_force_eV_A": _max_force(result),
            "runtime_s": runtime_s,
        })

    gas_results = {}
    gas_rows = []
    for adsorbate in ADSORBATE_BUILDERS:
        raw_path = RAW_DIR / f"full_gas_{adsorbate}.json"
        if raw_path.exists() and not args.force:
            print(f"Load gas cache: {adsorbate}", flush=True)
            result = _load_result(raw_path)
            runtime_s = 0.0
        else:
            print(f"Relax gas molecule: {adsorbate}", flush=True)
            result, runtime_s = _relax_one(
                backend,
                _gas_atoms(adsorbate),
                label=f"full_gas_{adsorbate}",
            )
            _write_result(raw_path, result)
            _write_result_atoms(
                STRUCTURE_DIR / "gas" / f"full_gas_{adsorbate}.extxyz",
                result,
                label=f"full_gas_{adsorbate}",
            )
        gas_results[adsorbate] = result
        gas_rows.append({
            "adsorbate": adsorbate,
            "energy_eV": float(result.energy),
            "converged": bool(result.converged),
            "optimizer_nsteps": int(result.optimizer_nsteps),
            "max_force_eV_A": _max_force(result),
            "runtime_s": runtime_s,
        })

    all_pair_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    chunk_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for host in HOST_BUILDERS:
        for adsorbate in ADSORBATE_BUILDERS:
            pair = (host, adsorbate)
            pair_label = f"{adsorbate}_{_safe(host)}"
            pair_csv = OUTPUT_DIR / f"pair_results_{_safe(adsorbate)}_{_safe(host)}.csv"
            if pair_csv.exists() and args.force:
                pair_csv.unlink()
            if pair_csv.exists() and not args.force:
                print(f"Skip existing pair: {adsorbate}/{host}", flush=True)
                all_pair_dfs[pair] = pd.read_csv(pair_csv)
                continue

            configs = build_config_grid(
                host_name=host,
                slab=relaxed_hosts[host],
                adsorbate_name=adsorbate,
                sites_filter=None,
                orientations_filter=None,
                rotations_deg=ROTATIONS,
                heights_A=HEIGHTS,
                frozen_fraction=0.5,
            )
            print(f"Relax pair: {adsorbate}/{host} ({len(configs)} configs)", flush=True)
            started = time.perf_counter()
            results, pair_chunk_rows = _relax_configs_in_chunks(
                backend,
                configs,
                host=host,
                adsorbate=adsorbate,
                clean_slab_atoms=relaxed_hosts[host],
                e_clean_slab_ev=float(clean_results[host].energy),
                e_gas_ads_ev=float(gas_results[adsorbate].energy),
                pair_label=pair_label,
                chunk_size=args.chunk_size,
                force=args.force,
            )
            pair_df = build_pair_results_table(
                host=host,
                adsorbate=adsorbate,
                configs=configs,
                opt_results=results,
                clean_slab_atoms=relaxed_hosts[host],
                e_clean_slab_ev=float(clean_results[host].energy),
                e_gas_ads_ev=float(gas_results[adsorbate].energy),
                backend="toolkit",
            )

            failures = pair_df[~pair_df["converged"]]
            if not failures.empty:
                print(
                    f"  Rerun {len(failures)} non-converged configs at "
                    f"{args.failure_n_steps} steps",
                    flush=True,
                )
                if rerun_backend is None:
                    rerun_backend = _build_backend(args.failure_n_steps, args.fmax)
                config_by_label = {config.label: config for config in configs}
                for idx, failure in failures.iterrows():
                    config = config_by_label[str(failure["label"])]
                    result, runtime_s = _relax_one(
                        rerun_backend,
                        config.atoms,
                        label=f"full_rerun_{_safe(config.label)}",
                        active_mask=config.active_mask,
                    )
                    rerun_df = build_pair_results_table(
                        host=host,
                        adsorbate=adsorbate,
                        configs=[config],
                        opt_results=[result],
                        clean_slab_atoms=relaxed_hosts[host],
                        e_clean_slab_ev=float(clean_results[host].energy),
                        e_gas_ads_ev=float(gas_results[adsorbate].energy),
                        backend="toolkit",
                    )
                    for col, value in rerun_df.iloc[0].items():
                        pair_df.loc[idx, col] = value
                    pair_df.loc[idx, "rerun_runtime_s"] = runtime_s
                    pair_df.loc[idx, "rerun_n_steps_cap"] = args.failure_n_steps

            pair_df.to_csv(pair_csv, index=False)
            all_pair_dfs[pair] = pair_df
            pair_runtime_s = time.perf_counter() - started
            chunk_rows.extend(pair_chunk_rows)
            pair_rows.append({
                "pair": f"{adsorbate}/{host}",
                "n_configs": len(configs),
                "n_converged": int(pair_df["converged"].sum()),
                "n_reliable": int(pair_df["reliable_for_minimum"].sum()),
                "runtime_s": pair_runtime_s,
                "pair_csv": str(pair_csv),
            })
            print(f"Done pair: {adsorbate}/{host} in {pair_runtime_s / 60.0:.1f} min", flush=True)

    full_df = pd.concat(all_pair_dfs.values(), ignore_index=True)
    summary_df = summarize_pair_validation(all_pair_dfs, ADSORBML_REFERENCES)
    comparison_df = _reference_rows(summary_df)
    chunk_df = pd.DataFrame(chunk_rows)
    clean_df = pd.DataFrame(clean_rows)
    gas_df = pd.DataFrame(gas_rows)
    pair_runtime_df = pd.DataFrame(pair_rows)

    full_results_csv = OUTPUT_DIR / "full_panel_pair_results.csv"
    summary_csv = OUTPUT_DIR / "summary_validation.csv"
    comparison_csv = OUTPUT_DIR / "reference_comparison.csv"
    chunk_csv = OUTPUT_DIR / "chunk_timings.csv"
    clean_csv = OUTPUT_DIR / "clean_slab_energies.csv"
    gas_csv = OUTPUT_DIR / "gas_energies.csv"
    pair_runtime_csv = OUTPUT_DIR / "pair_runtime_summary.csv"
    metadata_json = OUTPUT_DIR / "run_metadata.json"
    report_path = REPORT_DIR / "toolkit_full_panel_report.md"

    full_df.to_csv(full_results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    chunk_df.to_csv(chunk_csv, index=False)
    clean_df.to_csv(clean_csv, index=False)
    gas_df.to_csv(gas_csv, index=False)
    pair_runtime_df.to_csv(pair_runtime_csv, index=False)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "toolkit",
        "toolkit_checkpoint": os.environ.get("TOOLKIT_CHECKPOINT", "medium-mpa-0"),
        "toolkit_device": str(getattr(backend, "device", os.environ.get("TOOLKIT_DEVICE", "cuda"))),
        "toolkit_n_steps_cap": args.n_steps,
        "failure_n_steps_cap": args.failure_n_steps,
        "toolkit_fmax": args.fmax,
        "toolkit_d3bj_enabled": False,
        "toolkit_require_d3bj": False,
        "chunk_size": args.chunk_size,
        "hosts": list(HOST_BUILDERS),
        "adsorbates": list(ADSORBATE_BUILDERS),
        "rotations_deg": ROTATIONS,
        "heights_A": HEIGHTS,
        "n_structures": len(full_df),
        "n_converged": int(full_df["converged"].sum()),
        "n_reliable": int(full_df["reliable_for_minimum"].sum()),
        "runtime_s": time.perf_counter() - start_total,
        "full_results_csv": str(full_results_csv),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comparison_csv),
        "chunk_csv": str(chunk_csv),
        "clean_csv": str(clean_csv),
        "gas_csv": str(gas_csv),
        "pair_runtime_csv": str(pair_runtime_csv),
        "metadata_json": str(metadata_json),
        "report_path": str(report_path),
        "chunk_dir": str(CHUNK_DIR),
        "raw_results_dir": str(RAW_DIR),
        "structure_dir": str(STRUCTURE_DIR),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(
        path=report_path,
        metadata=metadata,
        summary_df=summary_df,
        comparison_df=comparison_df,
        full_df=full_df,
        chunk_df=chunk_df,
    )
    print(f"Wrote {full_results_csv}", flush=True)
    print(f"Wrote {summary_csv}", flush=True)
    print(f"Wrote {comparison_csv}", flush=True)
    print(f"Wrote {metadata_json}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
