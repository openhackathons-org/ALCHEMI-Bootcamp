#!/usr/bin/env python3
"""Create one OC20Dense accuracy report from the generated benchmark layers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PART1 = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PART1 / "outputs" / "oc20dense_known_examples"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required benchmark table: {path}")
    return pd.read_csv(path)


def _success_count(frame: pd.DataFrame, column: str) -> str:
    return f"{int(frame[column].sum())}/{len(frame)}"


def _median_or_nan(values: pd.Series) -> float:
    clean = pd.Series(pd.to_numeric(values, errors="coerce")).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.median())


def _toolkit_model_label(root: Path) -> str:
    metadata_path = root / "reports" / "run_metadata.json"
    checkpoint = "medium-mpa-0"
    head = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = str(metadata.get("toolkit_checkpoint", checkpoint))
        head = metadata.get("toolkit_head")
    return f"{checkpoint} (head={head})" if head else checkpoint


def _write_report(
    *,
    root: Path,
    layer_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    eads_summary: pd.DataFrame,
    selected_cases: pd.DataFrame,
    dft_reference: pd.DataFrame,
) -> Path:
    report_path = root / "reports" / "oc20dense_accuracy_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    max_target_delta = float(np.abs(dft_reference["dft_traj_minus_target_eV"]).max())
    max_start_active = float(dft_reference["start_active_atom_rmsd_A"].max())
    max_start_ads = float(dft_reference["start_adsorbate_rmsd_A"].max())
    max_mic_active = float(dft_reference["mic_active_atom_rmsd_A"].max())
    max_mic_ads = float(dft_reference["mic_adsorbate_rmsd_A"].max())
    adsorbates = ", ".join(sorted(dft_reference["adsorbate"].unique()))
    model_label = _toolkit_model_label(root)

    layer_cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "n_configs",
        "initial_sp_top1_gap_eV",
        "dft_final_sp_top1_gap_eV",
        "relaxed_top1_gap_eV",
        "initial_sp_spearman",
        "dft_final_sp_spearman",
        "relaxed_spearman",
        "raw_active_rmsd_median_A",
        "mic_active_rmsd_median_A",
        "raw_adsorbate_rmsd_median_A",
        "mic_adsorbate_rmsd_median_A",
        "relaxed_backend_converged",
    ]
    aggregate_cols = [
        "layer",
        "top1_success_0p10eV",
        "top3_success_0p10eV",
        "median_top1_gap_eV",
        "max_top1_gap_eV",
        "median_spearman",
    ]
    selected_cols = [
        "system_id",
        "case",
        "config_id",
        "adsorbate",
        "adsorbate_reference_species",
        "dft_rank",
        "dft_gap_to_best_eV",
        "active_rmsd_A",
        "adsorbate_rmsd_A",
        "mic_active_rmsd_A",
        "mic_adsorbate_rmsd_A",
    ]
    eads_cols = [
        "system_id",
        "adsorbate",
        "adsorbate_reference_species",
        "n_configs",
        "dft_final_eads_mae_eV",
        "dft_final_eads_bias_eV",
        "relaxed_eads_mae_eV",
        "relaxed_eads_bias_eV",
        "dft_final_eads_top1_gap_eV",
        "relaxed_eads_top1_gap_eV",
    ]

    lines = [
        "# OC20Dense Closed-Shell Accuracy Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Backend: post-processing summary of OC20Dense Toolkit/MACE outputs",
        f"Model: {model_label} Toolkit layers with D3(BJ) disabled",
        "",
        "## Scope",
        "",
        (
            "This report compares the same closed-shell OC20Dense systems across "
            "all generated benchmark layers: `*OH2`/H2O, `*NH3`/NH3, and "
            "`*N2`/N2. CH3-containing and radical-like adsorbates are outside "
            "this closed-shell benchmark slice."
        ),
        "",
        (
            f"The comparison covers {len(dft_reference)} exact "
            f"`system_id`/`config_id`/`sid` records and the active adsorbates are: "
            f"{adsorbates}."
        ),
        "",
        "## Layer Definitions",
        "",
        (
            "- Initial-coordinate SP: selected-checkpoint MACE single point on "
            "the official OC20Dense starting geometries."
        ),
        (
            "- DFT-relaxed final SP: selected-checkpoint MACE single point on the final "
            "frame of the official OC20Dense DFT trajectory."
        ),
        (
            "- Toolkit relaxation: selected-checkpoint MACE relaxation from "
            "the official OC20Dense starting geometry, ranked by final MACE "
            "total energy."
        ),
        (
            "- DFT trajectory reference: independent check that "
            "`DFT-relaxed final total energy - oc20dense_ref_energies[system_id]` "
            "exactly reproduces the released OC20Dense adsorption-energy target."
        ),
        "",
        (
            "The three ranking layers above use MACE total energies within a "
            "fixed system and fixed composition. The separate Eads section below "
            "uses an explicit MACE-scale clean-surface and gas-reference "
            "subtraction."
        ),
        "",
        "## Aggregate Accuracy",
        "",
        aggregate[aggregate_cols].to_markdown(index=False),
        "",
        "## Per-System Accuracy",
        "",
        layer_summary[layer_cols].to_markdown(index=False),
        "",
        "## DFT Reference Integrity",
        "",
        f"- Exact trajectory records compared: {len(dft_reference)}.",
        f"- Max absolute DFT trajectory-target difference: {max_target_delta:.6g} eV.",
        f"- Max starting-frame active-atom RMSD: {max_start_active:.6g} A.",
        f"- Max starting-frame adsorbate RMSD: {max_start_ads:.6g} A.",
        f"- Max final-frame minimum-image active-atom RMSD: {max_mic_active:.6g} A.",
        f"- Max final-frame minimum-image adsorbate RMSD: {max_mic_ads:.6g} A.",
        "",
        "## Defined MACE Adsorption Energies",
        "",
        (
            "This layer uses official clean-surface trajectories plus neutral "
            "MACE-relaxed gas molecules: `E_ads^MACE = E(adslab) - E(surface) "
            "- E(gas)`. It is useful for model-level comparison, but remains a "
            "defined local MACE convention rather than a claim that every OC20 "
            "reference detail has been reproduced exactly."
        ),
        "",
        eads_summary[eads_cols].to_markdown(index=False),
        "",
        "## Selected Geometry Cases",
        "",
        selected_cases[selected_cols].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "The DFT-relaxed final SP layer isolates the model energy on known DFT minima. "
            "It gives the clearest energy-model comparison because relaxation "
            "path differences are removed."
        ),
        "",
        (
            "The relaxation layer is stricter: it combines the energy model, "
            "forces, optimizer, active/frozen atom convention, and the finite "
            "step cap. A relaxed top-1 miss with a successful top-3 result means "
            "the workflow still places a near-DFT-best candidate in the shortlist, "
            "but the final single reported winner is less reliable for that system."
        ),
        "",
        (
            "Large RMSD outliers should be inspected visually before making a "
            "chemistry claim. They can reflect different local minima, periodic "
            "image choices, or adsorbate reorientation even when the ranked "
            "energy gap remains small."
        ),
        "",
        "## Output Tables",
        "",
        f"- Layer summary: `{root / 'tables' / 'accuracy_layer_summary.csv'}`",
        f"- Aggregate summary: `{root / 'tables' / 'accuracy_aggregate_summary.csv'}`",
        f"- DFT-relaxed final SP summary: `{root / 'dft_final_single_points' / 'tables' / 'dft_final_sp_system_summary.csv'}`",
        f"- Defined MACE Eads summary: `{root / 'mace_adsorption_energy' / 'tables' / 'mace_adsorption_energy_summary.csv'}`",
        f"- DFT trajectory comparison: `{root / 'dft_reference_checks' / 'dft_reference_comparison.csv'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = _parse_args()
    root = args.root
    system = _read_required(root / "tables" / "system_summary.csv")
    dft_final = _read_required(
        root
        / "dft_final_single_points"
        / "tables"
        / "dft_final_sp_system_summary.csv"
    )
    rmsd = _read_required(root / "dft_reference_checks" / "system_rmsd_summary.csv")
    selected = _read_required(root / "dft_reference_checks" / "selected_case_comparison.csv")
    dft_reference = _read_required(
        root / "dft_reference_checks" / "dft_reference_comparison.csv"
    )
    eads_summary = _read_required(
        root
        / "mace_adsorption_energy"
        / "tables"
        / "mace_adsorption_energy_summary.csv"
    )

    layer_summary = system[
        [
            "system_id",
            "adsorbate",
            "adsorbate_reference_species",
            "n_configs",
            "sp_best_dft_gap_to_best_eV",
            "sp_top3_best_dft_gap_eV",
            "sp_top1_success_0p10eV",
            "sp_top3_success_0p10eV",
            "sp_spearman_rank_corr",
            "ml_best_dft_gap_to_best_eV",
            "ml_top3_best_dft_gap_eV",
            "ml_top1_success_0p10eV",
            "ml_top3_success_0p10eV",
            "relaxed_spearman_rank_corr",
            "n_converged_backend",
        ]
    ].merge(
        dft_final[
            [
                "system_id",
                "dft_final_sp_best_dft_gap_to_best_eV",
                "dft_final_sp_top3_best_dft_gap_eV",
                "dft_final_sp_top1_success_0p10eV",
                "dft_final_sp_top3_success_0p10eV",
                "dft_final_sp_spearman_rank_corr",
                "mace_dft_final_sp_free_fmax_median_eV_A",
            ]
        ],
        on="system_id",
        how="left",
        validate="one_to_one",
    ).merge(
        rmsd[
            [
                "system_id",
                "active_rmsd_median_A",
                "adsorbate_rmsd_median_A",
                "mic_active_rmsd_median_A",
                "mic_adsorbate_rmsd_median_A",
                "active_rmsd_max_A",
                "adsorbate_rmsd_max_A",
                "mic_active_rmsd_max_A",
                "mic_adsorbate_rmsd_max_A",
            ]
        ],
        on="system_id",
        how="left",
        validate="one_to_one",
    )

    layer_summary = layer_summary.rename(
        columns={
            "sp_best_dft_gap_to_best_eV": "initial_sp_top1_gap_eV",
            "sp_top3_best_dft_gap_eV": "initial_sp_top3_gap_eV",
            "sp_spearman_rank_corr": "initial_sp_spearman",
            "dft_final_sp_best_dft_gap_to_best_eV": "dft_final_sp_top1_gap_eV",
            "dft_final_sp_top3_best_dft_gap_eV": "dft_final_sp_top3_gap_eV",
            "dft_final_sp_spearman_rank_corr": "dft_final_sp_spearman",
            "ml_best_dft_gap_to_best_eV": "relaxed_top1_gap_eV",
            "ml_top3_best_dft_gap_eV": "relaxed_top3_gap_eV",
            "relaxed_spearman_rank_corr": "relaxed_spearman",
            "active_rmsd_median_A": "raw_active_rmsd_median_A",
            "adsorbate_rmsd_median_A": "raw_adsorbate_rmsd_median_A",
            "active_rmsd_max_A": "raw_active_rmsd_max_A",
            "adsorbate_rmsd_max_A": "raw_adsorbate_rmsd_max_A",
        }
    )
    layer_summary["relaxed_backend_converged"] = (
        layer_summary["n_converged_backend"].astype(str)
        + "/"
        + layer_summary["n_configs"].astype(str)
    )

    aggregate = pd.DataFrame(
        [
            {
                "layer": "initial-coordinate SP",
                "top1_success_0p10eV": _success_count(
                    system, "sp_top1_success_0p10eV"
                ),
                "top3_success_0p10eV": _success_count(
                    system, "sp_top3_success_0p10eV"
                ),
                "median_top1_gap_eV": float(
                    system["sp_best_dft_gap_to_best_eV"].median()
                ),
                "max_top1_gap_eV": float(system["sp_best_dft_gap_to_best_eV"].max()),
                "median_spearman": _median_or_nan(system["sp_spearman_rank_corr"]),
            },
            {
                "layer": "DFT-relaxed final SP",
                "top1_success_0p10eV": _success_count(
                    dft_final, "dft_final_sp_top1_success_0p10eV"
                ),
                "top3_success_0p10eV": _success_count(
                    dft_final, "dft_final_sp_top3_success_0p10eV"
                ),
                "median_top1_gap_eV": float(
                    dft_final["dft_final_sp_best_dft_gap_to_best_eV"].median()
                ),
                "max_top1_gap_eV": float(
                    dft_final["dft_final_sp_best_dft_gap_to_best_eV"].max()
                ),
                "median_spearman": _median_or_nan(
                    dft_final["dft_final_sp_spearman_rank_corr"]
                ),
            },
            {
                "layer": "Toolkit relaxation",
                "top1_success_0p10eV": _success_count(
                    system, "ml_top1_success_0p10eV"
                ),
                "top3_success_0p10eV": _success_count(
                    system, "ml_top3_success_0p10eV"
                ),
                "median_top1_gap_eV": float(
                    system["ml_best_dft_gap_to_best_eV"].median()
                ),
                "max_top1_gap_eV": float(system["ml_best_dft_gap_to_best_eV"].max()),
                "median_spearman": _median_or_nan(
                    system["relaxed_spearman_rank_corr"]
                ),
            },
        ]
    )

    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(tables / "accuracy_layer_summary.csv", index=False)
    aggregate.to_csv(tables / "accuracy_aggregate_summary.csv", index=False)
    report_path = _write_report(
        root=root,
        layer_summary=layer_summary,
        aggregate=aggregate,
        eads_summary=eads_summary,
        selected_cases=selected,
        dft_reference=dft_reference,
    )
    print(report_path)
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
