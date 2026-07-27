#!/usr/bin/env python3
"""Execute the real Part 1 prefix and measure its eight Stage 3 code cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


STAGE_CELL_IDS = (
    "load-nci-atlas",
    "configure-nci-model",
    "evaluate-nci-components",
    "compose-nci-pipeline",
    "validate-nci-graph-order",
    "check-nci-force",
    "analyze-nci-curves",
    "display-nci-curves",
)
STAGE_START_ID = "stage-3"
STAGE_END_ID = "stage-4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_index_by_id(notebook: Any) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, cell in enumerate(notebook.cells):
        cell_id = str(cell.get("id", ""))
        if not cell_id:
            raise ValueError(f"notebook cell {index} has no ID")
        if cell_id in indices:
            raise ValueError(f"duplicate notebook cell ID: {cell_id}")
        indices[cell_id] = index
    return indices


def _validate_stage_layout(notebook: Any) -> tuple[int, int]:
    indices = _cell_index_by_id(notebook)
    required = (STAGE_START_ID, *STAGE_CELL_IDS, STAGE_END_ID)
    missing = [cell_id for cell_id in required if cell_id not in indices]
    if missing:
        raise ValueError(f"missing required notebook cells: {missing}")

    start = indices[STAGE_START_ID]
    end = indices[STAGE_END_ID]
    actual = tuple(
        str(cell.get("id"))
        for cell in notebook.cells[start + 1 : end]
        if cell.cell_type == "code"
    )
    if actual != STAGE_CELL_IDS:
        raise ValueError(
            "Stage 3 code-cell order changed; expected "
            f"{STAGE_CELL_IDS}, found {actual}"
        )
    return start, end


def _source_identity(notebook: Any) -> tuple[tuple[str, str, str], ...]:
    """Return the notebook fields that a timing run must not change."""

    return tuple(
        (str(cell.get("id")), str(cell.cell_type), str(cell.source))
        for cell in notebook.cells
    )


def _execute_temporary_cell(
    client: NotebookClient,
    notebook: Any,
    source: str,
    *,
    execution_count: int,
) -> None:
    """Execute setup/export code without replacing a canonical notebook cell."""

    temporary_cell = nbformat.v4.new_code_cell(source)
    notebook.cells.append(temporary_cell)
    temporary_index = len(notebook.cells) - 1
    try:
        client.execute_cell(
            temporary_cell,
            temporary_index,
            execution_count=execution_count,
            store_history=False,
        )
    finally:
        notebook.cells.pop()


def _verify_checkpoint_report(path: Path) -> dict[str, Any]:
    """Check that all four checkpoint files were resolved before kernel start."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "alchemi.part1-aimnet-prewarm.v1":
        raise ValueError(f"unexpected checkpoint report schema in {path}")
    rows = report.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError(f"checkpoint report has no checkpoint list: {path}")
    aliases = tuple(str(row.get("alias")) for row in rows)
    expected = (
        "aimnet2-b973c-2025-d3_0",
        *(f"aimnet2-wb97m-d3_{index}" for index in range(4)),
    )
    if aliases != expected:
        raise ValueError(f"checkpoint report lists {aliases}; expected {expected}")
    live_cache_dir = Path(os.environ["AIMNET_CACHE_DIR"]).resolve()
    if Path(str(report.get("cache_dir"))).resolve() != live_cache_dir:
        raise ValueError(
            "checkpoint report does not describe the live AIMNET_CACHE_DIR"
        )
    for row in rows:
        checkpoint_path = Path(str(row["path"]))
        if checkpoint_path.parent != live_cache_dir:
            raise ValueError(
                f"checkpoint is outside the live AIMNET_CACHE_DIR: {checkpoint_path}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"prewarmed checkpoint is missing: {checkpoint_path}"
            )
        observed = _sha256(checkpoint_path)
        if observed != row.get("sha256"):
            raise ValueError(f"checkpoint changed after prewarm: {checkpoint_path}")
    return report


def _scientific_export_source() -> str:
    return r"""
import json as _json
import os as _os
from pathlib import Path as _Path

_result_dir = _Path(_os.environ["ALCHEMI_NCI_RESULT_DIR"])
_result_dir.mkdir(parents=True, exist_ok=True)
nci_metrics.reset_index().to_csv(_result_dir / "nci_metrics.csv", index=False)
nci_curves.to_csv(_result_dir / "nci_curves.csv", index=False)
nci_graph_index.to_csv(_result_dir / "nci_graph_index.csv", index=False)
np.savez_compressed(
    _result_dir / "nci_component_energies.npz",
    checkpoint_residual_graph_energy_eV=nci_member_core_eV.numpy(),
    coulomb_graph_energy_eV=nci_member_coulomb_eV.numpy(),
    d3_graph_energy_eV=nci_d3_graph_eV.numpy(),
    complete_member0_graph_energy_eV=nci_pipeline_energy_cpu.numpy(),
    ensemble_predicted_charge_e=nci_member_charges_e.numpy(),
    complete_member0_force_eV_per_A=(
        nci_full_outputs["forces"].detach().cpu().numpy()
    ),
    batch_ptr=nci_full_batch.batch_ptr.detach().cpu().numpy(),
)

_nci_result = {
    "graph_count": int(nci_batch.num_graphs),
    "atom_count": int(nci_batch.num_nodes),
    "geometry_group_count": int(nci_batch.num_graphs // 3),
    "ensemble_member_count": len(NCI_CHECKPOINTS),
    "checkpoints": list(NCI_CHECKPOINTS),
    "nci_subset_sha256": sha256_file(NCI_DATA_FILE),
    "d3_parameter_sha256": D3_PARAMETER_SHA256,
    "core_commit": installed_pins["Core"],
    "ops_commit": installed_pins["Ops"],
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(DEVICE),
    "array_file": "nci_component_energies.npz",
    "array_order": (
        "Graph arrays follow nci_graph_index.csv; atom arrays are concatenated "
        "in that graph order and batch_ptr gives graph boundaries."
    ),
    "curve_energy_unit": "kcal/mol",
    "graph_energy_unit": "eV",
    "charge_unit": "elementary charge",
    "force_unit": "eV/angstrom",
    "method_settings": {
        "checkpoint_family": "aimnet2-wb97m-d3",
        "ensemble_members": list(NCI_CHECKPOINTS),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "d3_parameters": dict(nci_d3_params),
        "d3_cutoff_A": D3_CUTOFF_A,
        "d3_smoothing_fraction": NCI_D3_SMOOTHING_FRACTION,
        "coulomb_method": "direct nonperiodic all-pairs 1/r",
        "coulomb_constant_eV_A_per_e2": nci_coulomb.coulomb_constant,
        "force_check": {
            "component_selection": (
                "largest-magnitude component of the official analytic force"
            ),
            "finite_difference_step_A": NCI_FD_STEP_A,
            "official_total_energy_route": (
                "official AIMNet2Calculator complete total energy with "
                "simple Coulomb and D3"
            ),
            "official_analytic_force_route": (
                "official AIMNet2Calculator complete-model force"
            ),
            "toolkit_analytic_force_route": (
                "Toolkit PipelineModelWrapper complete-model force"
            ),
        },
    },
    "force_check": {
        "atom_index": nci_fd_atom_index,
        "cartesian_axis": nci_fd_axis,
        "official_analytic_force_eV_A": nci_official_analytic_force_eV_A,
        "official_finite_difference_force_eV_A": nci_official_fd_force_eV_A,
        "toolkit_analytic_force_eV_A": (
            nci_toolkit_analytic_force_eV_A
        ),
        "official_analytic_vs_official_finite_difference_abs_error_eV_A": (
            nci_official_fd_error_eV_A
        ),
        "toolkit_analytic_vs_official_analytic_abs_error_eV_A": (
            nci_toolkit_official_error_eV_A
        ),
    },
    "charge_conservation_max_abs_e": nci_charge_conservation_max_abs_e,
    "component_sum_max_abs_eV": nci_component_sum_max_abs_eV,
    "graph_order_max_abs_eV": nci_graph_order_max_abs_eV,
    "net_force_vector_eV_A": nci_net_force_eV_A.detach().cpu().tolist(),
    "net_force_max_abs_eV_A": nci_net_force_max_abs_eV_A,
    "max_complete_mae_vs_dft_d3_kcal_mol": float(
        nci_metrics["complete vs DFT-D3"].max()
    ),
    "max_complete_mae_vs_ccsd_t_cbs_kcal_mol": float(
        nci_metrics["complete vs CC"].max()
    ),
    "metrics": _json.loads(nci_metrics.reset_index().to_json(orient="records")),
    "checks": {
        "charge_conservation": {
            "status": "passed", "atol_e": NCI_CHARGE_ATOL_E, "rtol": 0.0
        },
        "pipeline_component_sum": {
            "status": "passed",
            "atol_eV": NCI_ENERGY_ATOL_EV,
            "rtol": NCI_ENERGY_RTOL,
        },
        "graph_order_invariance": {
            "status": "passed",
            "atol_eV": NCI_ENERGY_ATOL_EV,
            "rtol": NCI_ENERGY_RTOL,
        },
        "net_force": {
            "status": "passed",
            "component_atol_eV_A": NCI_NET_FORCE_ATOL_EV_A,
            "rtol": 0.0,
        },
        "official_analytic_vs_official_total_energy_finite_difference": {
            "status": "passed",
            "atol_eV_A": NCI_FD_ATOL_EV_A,
            "rtol": NCI_FD_RTOL,
            "official_analytic_vs_official_finite_difference_abs_error_eV_A": (
                nci_official_fd_error_eV_A
            ),
            "official_analytic_force_route": (
                "official AIMNet2Calculator complete-model force"
            ),
            "official_total_energy_route": (
                "official AIMNet2Calculator complete total energy with "
                "simple Coulomb and D3"
            ),
        },
        "toolkit_analytic_vs_official_analytic": {
            "status": "passed",
            "atol_eV_A": NCI_PIPELINE_OFFICIAL_FORCE_ATOL_EV_A,
            "rtol": 0.0,
            "toolkit_analytic_vs_official_analytic_abs_error_eV_A": (
                nci_toolkit_official_error_eV_A
            ),
            "toolkit_analytic_force_route": (
                "Toolkit PipelineModelWrapper complete-model force"
            ),
            "official_analytic_force_route": (
                "official AIMNet2Calculator complete-model force"
            ),
        },
        "complete_mae": {
            "status": "passed",
            "per_curve_limit_kcal_mol": NCI_COMPLETE_MAE_LIMIT_KCAL_MOL,
        },
    },
}
(_result_dir / "nci_scientific_results.json").write_text(
    _json.dumps(_nci_result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
"""


def _write_cell_times(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cell_index", "code_index", "cell_id", "scope", "wall_s"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--executed-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cell-times", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--kernel", default="alchemi-main")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    executed_path = args.executed_prefix.resolve()
    report_path = args.report.resolve()
    cell_times_path = args.cell_times.resolve()
    result_dir = Path(os.environ["ALCHEMI_NCI_RESULT_DIR"]).resolve()
    for path in (executed_path, report_path, cell_times_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_report_path = args.checkpoint_report.resolve()
    distinct_paths = {
        notebook_path,
        executed_path,
        report_path,
        cell_times_path,
        checkpoint_report_path,
    }
    if len(distinct_paths) != 5:
        raise ValueError(
            "notebook, executed notebook, timing report, cell timings, and "
            "checkpoint report must use distinct paths"
        )
    checkpoint_report = _verify_checkpoint_report(checkpoint_report_path)

    notebook = nbformat.read(notebook_path, as_version=4)
    stage_start, stage_end = _validate_stage_layout(notebook)
    canonical_source_identity = _source_identity(notebook)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.reset_execution_trackers()

    started_at = datetime.now(UTC)
    all_rows: list[dict[str, Any]] = []
    code_index = 0
    stage_outer_started: float | None = None
    stage_outer_elapsed: float | None = None
    success = False
    failure: str | None = None
    kernel_started = False
    post_run_error: Exception | None = None

    try:
        with client.setup_kernel():
            kernel_started = True
            for cell_index, cell in enumerate(notebook.cells[:stage_end]):
                if cell.cell_type != "code":
                    continue
                code_index += 1
                cell_id = str(cell.get("id"))
                in_stage = stage_start < cell_index < stage_end
                if in_stage and stage_outer_started is None:
                    _execute_temporary_cell(
                        client,
                        notebook,
                        "torch.cuda.synchronize(DEVICE)",
                        execution_count=code_index,
                    )
                    stage_outer_started = perf_counter()

                original_source = cell.source
                if in_stage:
                    cell.source = (
                        original_source + "\n\ntorch.cuda.synchronize(DEVICE)\n"
                    )
                cell_started = perf_counter()
                try:
                    client.execute_cell(
                        cell,
                        cell_index,
                        execution_count=code_index,
                    )
                finally:
                    cell.source = original_source
                elapsed = perf_counter() - cell_started
                all_rows.append(
                    {
                        "cell_index": cell_index,
                        "code_index": code_index,
                        "cell_id": cell_id,
                        "scope": "nci_stage_3" if in_stage else "dependency_prefix",
                        "wall_s": round(elapsed, 6),
                    }
                )
                client.set_widgets_metadata()
                nbformat.write(notebook, executed_path)
                print(
                    f"[{cell_id}] {elapsed:.3f} s "
                    f"({'measured' if in_stage else 'setup'})",
                    flush=True,
                )

            if stage_outer_started is None:
                raise RuntimeError("Stage 3 did not execute")
            stage_outer_elapsed = perf_counter() - stage_outer_started

            stage_rows = [row for row in all_rows if row["scope"] == "nci_stage_3"]
            measured_ids = tuple(str(row["cell_id"]) for row in stage_rows)
            if measured_ids != STAGE_CELL_IDS:
                raise RuntimeError(
                    "Stage 3 timing did not execute the required eight-cell sequence"
                )

            _execute_temporary_cell(
                client,
                notebook,
                _scientific_export_source(),
                execution_count=code_index + 1,
            )
            success = True
    except CellExecutionError as error:
        failure = str(error)
        raise
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if kernel_started:
            try:
                client.set_widgets_metadata()
            except Exception as error:
                post_run_error = error
                success = False
                if failure is None:
                    failure = f"{type(error).__name__}: {error}"
        source_identity_preserved = (
            _source_identity(notebook) == canonical_source_identity
        )
        if not source_identity_preserved:
            post_run_error = RuntimeError("timing run changed canonical notebook cells")
            success = False
            if failure is None:
                failure = str(post_run_error)
        nbformat.write(notebook, executed_path)
        _write_cell_times(cell_times_path, all_rows)
        stage_rows = [row for row in all_rows if row["scope"] == "nci_stage_3"]
        dependency_rows = [
            row for row in all_rows if row["scope"] == "dependency_prefix"
        ]
        scientific_path = result_dir / "nci_scientific_results.json"
        scientific = (
            json.loads(scientific_path.read_text(encoding="utf-8"))
            if success and scientific_path.is_file()
            else None
        )
        report = {
            "schema": "alchemi.part1-nci-stage-timing.v1",
            "success": success,
            "failure": failure,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "notebook": str(notebook_path),
            "notebook_sha256": _sha256(notebook_path),
            "stage_cell_ids": list(STAGE_CELL_IDS),
            "stage_cell_count": len(stage_rows),
            "canonical_source_identity_preserved": source_identity_preserved,
            "checkpoint_report": str(checkpoint_report_path),
            "checkpoint_report_sha256": _sha256(checkpoint_report_path),
            "checkpoint_aliases": [
                row["alias"] for row in checkpoint_report["checkpoints"]
            ],
            "timing_method": (
                "one torch.cuda.synchronize(DEVICE) before Stage 3, then "
                "perf_counter wall time around each canonical code cell with "
                "torch.cuda.synchronize(DEVICE) appended after that cell"
            ),
            "timing_scope": (
                "Stage 3 data loading, four checkpoint loads and AIMNet passes, "
                "one shared D3 pass, composed-pipeline checks, force check, "
                "reference reductions, table construction, and plotting. Kernel "
                "startup, dependency cells, checkpoint downloads, and result "
                "packaging are excluded."
            ),
            "run_conditions": {
                "checkpoint_cache": "warm; all four files were resolved before kernel start",
                "cuda_context": "warm from the real Stage 1 and Stage 2 prefix",
                "nci_model_objects": "cold in this fresh notebook kernel",
                "first_nci_model_calls": "included",
            },
            "nci_stage_cell_wall_s": round(
                sum(float(row["wall_s"]) for row in stage_rows), 6
            ),
            "nci_stage_outer_wall_s": (
                round(stage_outer_elapsed, 6)
                if stage_outer_elapsed is not None
                else None
            ),
            "dependency_prefix_cell_wall_s": round(
                sum(float(row["wall_s"]) for row in dependency_rows), 6
            ),
            "cell_times_csv": str(cell_times_path),
            "executed_prefix": str(executed_path),
            "scientific_results": scientific,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if post_run_error is not None:
        raise post_run_error
    print(f"NCI_STAGE_CELL_WALL_S={report['nci_stage_cell_wall_s']:.6f}")
    print(f"NCI_STAGE_OUTER_WALL_S={report['nci_stage_outer_wall_s']:.6f}")
    print(f"NCI_TIMING_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
