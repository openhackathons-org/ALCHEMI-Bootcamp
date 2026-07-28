#!/usr/bin/env python3
"""Validate and checksum one executed Part 1 IR notebook run."""

# ruff: noqa: E402 -- the validator adds the staged repo roots before imports.

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

import nbformat
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PART_ROOT = REPO_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))

from build.prewarm_aimnet import CHECKPOINT_IDENTITIES
from build.prewarm_sevennet import resolve_checkpoint as resolve_sevennet_checkpoint
from build.verify_part1_runtime import (
    EXPECTED_PACKMOL_VERSION,
    EXPECTED_VERSIONS as EXPECTED_RUNTIME_VERSIONS,
    RECORDED_SCIENTIFIC_VERSIONS,
    REQUIRED_TRACKED_SOURCE_PATHS,
    SOURCE_MANIFEST_RELATIVE_PATH,
    git_source_revision,
    load_source_paths,
)
from aux.adsorption import (
    ADSLAB_KEYS,
    ADSORBATES,
    GEOMETRY_STATUS,
    STRUCTURE_KEYS,
    load_initial_structure_set,
)
from aux.analysis import (
    h_to_d_mode_mapping_table,
    ir_comparison_table,
    ir_spectrum_metrics,
    reference_comparison_metrics,
    topology_time_series,
)
from aux.artifacts import (
    WATER_RUN_MANIFEST_SCHEMA,
    load_ir_trajectory,
)
from aux.checkpoint import (
    resolve_checkpoint_path,
    verify_checkpoint_identities,
)
from aux.composition_config import (
    COMPILED_EAGER_CHARGE_TOLERANCE_E,
    COMPILED_EAGER_ENERGY_TOLERANCE_EV,
    COMPILED_EAGER_FORCE_TOLERANCE_EV_A,
    COMPILED_REPEAT_CHARGE_TOLERANCE_E,
    COMPILED_REPEAT_ENERGY_TOLERANCE_EV,
    COMPILED_REPEAT_FORCE_TOLERANCE_EV_A,
    COMPONENT_CLOSURE_TOLERANCE_EV,
    COMPOSITION_ANALYTIC_COULOMB_ENERGY_TOLERANCE_EV,
    COMPOSITION_ANALYTIC_COULOMB_FORCE_TOLERANCE_EV_A,
    COMPOSITION_CHARGE_AGREEMENT_TOLERANCE_E,
    COMPOSITION_ENERGY_AGREEMENT_TOLERANCE_EV,
    COMPOSITION_FD_ENERGY_ROUTE,
    COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
    COMPOSITION_FD_STEP_A,
    COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
    COMPOSITION_INTERACTION_AGREEMENT_TOLERANCE_EV,
    FULL_SERIAL_BATCH_TOLERANCE_EV,
    RESIDUAL_SERIAL_BATCH_TOLERANCE_EV,
)
from aux.experimental_reference import load_experimental_water_fundamentals
from aux.domain.config import DOMAIN_METHODOLOGY
from aux.domain.results import load_domain_lesson_view
from aux.harmonic_config import (
    HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E,
    HARMONIC_DISPLACEMENT_STEPS_BOHR,
    HARMONIC_FMAX_EV_A,
    HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1,
    HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX,
    HARMONIC_IMAGINARY_FLOOR_CM1,
    HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL,
    HARMONIC_INTENSITY_STEP_REL_TOLERANCE,
    HARMONIC_MODE_OVERLAP_MIN,
    HARMONIC_SELECTED_STEP_BOHR,
)
from aux.harmonic_ir import (
    ANGSTROM_PER_BOHR,
    analyze_harmonic_ir,
    assemble_harmonic_ir_finite_difference,
    molecular_dipoles_from_atomic_predictions,
    summarize_harmonic_ir_convergence,
    symmetric_cartesian_displacements,
)
from aux.models.sevennet_config import (
    SEVENNET_CHECKPOINT_BYTES as EXPECTED_SEVENNET_CHECKPOINT_BYTES,
    SEVENNET_CHECKPOINT_DOI,
    SEVENNET_CHECKPOINT_SHA256 as EXPECTED_SEVENNET_CHECKPOINT_SHA256,
    SEVENNET_CHECKPOINT_URL,
    SEVENNET_MODALITY,
    SEVENNET_PACKAGE_VERSION as EXPECTED_SEVENNET_VERSION,
    SEVENNET_REFERENCE_METHOD,
    SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
    SEVENNET_REPEAT_FORCE_TOL_EV_A,
)
from aux.nci_atlas import (
    CURVE_KEY_COLUMNS,
    EXPECTED_SYSTEMS,
    NCI_ATLAS_SUBSET_SHA256,
    build_graph_index,
    extract_repeated_interaction_reference,
    interaction_metrics,
    load_nci_atlas_subset,
    mean_member_curves,
    reduce_fragment_energies,
)
from aux.nci_config import NCI_VALIDATION
from aux.reference import (
    label_water_monomer_modes,
    load_psi4_b973c_ir_artifact,
    reference_water_monomer_mode_labels,
)
from aux.release_links import (
    LOCAL_NOTEBOOK_REFERENCES,
    local_reference_replacements,
)
from aux.workflow_config import IR_PRODUCTION_STATUS, IR_WARMUP_STATUS

RUN_MANIFEST_NAME = "water_run_manifest.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5"
)
EXPECTED_D3_PARAMETER_SHA256 = (
    "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
)
EXPECTED_TOOLKIT_CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
EXPECTED_TOOLKIT_OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
EXPECTED_AIMNET_VERSION = "0.2.0"
EXPECTED_TORCH_VERSION_PREFIX = "2.12.0"
SEVENNET_MAPPING_TOLERANCE = 1.0e-6
EXPECTED_NCI_CHECKPOINTS = tuple(f"aimnet2-wb97m-d3_{member}" for member in range(4))
EXPECTED_NCI_REFERENCE_LEVELS = (
    "ωB97M-D3(BJ)/def2-TZVPPD",
    "CCSD(T)/CBS interaction energies",
)
NCI_COMPONENT_COLUMNS = (
    "core",
    "core_plus_d3",
    "core_plus_coulomb",
    "full",
)
NCI_COMPARISONS = {
    "residual vs CC": ("core", "ccsd_t_cbs"),
    "residual + Coulomb vs CC": ("core_plus_coulomb", "ccsd_t_cbs"),
    "complete vs CC": ("full", "ccsd_t_cbs"),
    "same-D3 bookkeeping identity": (
        "core_plus_coulomb",
        "dft_no_d3",
    ),
    "complete vs DFT-D3": ("full", "dft_full"),
    "DFT-D3 vs CC": ("dft_full", "ccsd_t_cbs"),
}
NCI_CURVE_COLUMNS = (
    *CURVE_KEY_COLUMNS,
    *NCI_COMPONENT_COLUMNS,
    "full_std",
    "dft_full",
    "dft_no_d3",
    "ccsd_t_cbs",
)
NCI_ENSEMBLE_COLUMNS = (
    *CURVE_KEY_COLUMNS,
    "member",
    *NCI_COMPONENT_COLUMNS,
)
NCI_METRIC_COLUMNS = (
    "system_name",
    *NCI_COMPARISONS,
    "ensemble spread",
)

REQUIRED_FILES = (
    RUN_MANIFEST_NAME,
    "domain_box_evaluated.extxyz",
    "domain_single_gpu_result.csv",
    "inflight_queue_summary.csv",
    "nci_ensemble_curves.csv",
    "nci_interaction_curves.csv",
    "nci_interaction_metrics.csv",
    "part1_results_summary.csv",
    "surface_adsorption_energies.csv",
    "surface_adsorption_forces.csv",
    "sevennet_adapter_graph_mapping.csv",
    "sevennet_adapter_numerical_agreement.csv",
    "water_batch_cpu_gpu_crossover.csv",
    "water_batch_first_warm_calls.csv",
    "water_batch_layouts.csv",
    "water_dimer_ablation.csv",
    "water_dimer_ablation.png",
    "water_dimer_ablation_mae.csv",
    "water_ir_d6_topology_timeline.csv",
    "water_ir_diagnostics.csv",
    "water_ir_dynamics_log.csv",
    "water_ir_h6_topology_timeline.csv",
    "water_ir_trajectory.npz",
    "water_ir_metrics.csv",
    "water_ir_topology.csv",
    "water_ir_comparisons.csv",
    "water_ir_dft_comparison.csv",
    "water_ir_h_to_d_mode_map.csv",
    "water_ir_spectra.csv",
    "water_ir_dft_mapping.png",
    "water_ir_relaxed_start.extxyz",
    "water_ir_topology_timeline.png",
    "water_monomer_aimnet_harmonic_ir.npz",
    "water_monomer_harmonic_checks.csv",
    "water_monomer_harmonic_comparison.csv",
    "water_monomer_harmonic_convergence.csv",
    "water_monomer_harmonic_displacements.csv",
    "water_monomer_harmonic_minimum.extxyz",
    "water_hexamer_seed.extxyz",
    "water_hexamer_relaxed.extxyz",
    "water_hexamer_trajectory_stride100.extxyz",
)

HARMONIC_COMPARISON_PLOT = "water_monomer_harmonic_comparison.png"
DERIVED_FREQUENCY_RTOL = 1.0e-12
DERIVED_FREQUENCY_ATOL = 1.0e-9
DERIVED_PSD_RTOL = 1.0e-11
DERIVED_PSD_ATOL = 1.0e-14
DERIVED_TOPOLOGY_MODE_RTOL = 1.0e-12
DERIVED_TOPOLOGY_MODE_ATOL = 1.0e-10

REQUIRED_DIRECTORIES = ("domain_box", "water_ir_relaxed.zarr")
REQUIRED_OUTPUTS = REQUIRED_FILES + REQUIRED_DIRECTORIES
BUNDLE_SOURCE_FILES = (
    "alchemi-water-ir-source.ipynb",
    "run_notebook_no_timeout.py",
)
RUNTIME_CHECK_NAME = "part1-runtime.json"
D3_CACHE_REPORT_NAME = "part1-d3-cache.json"
TIMING_REPORT_NAME = "notebook-timings.json"
TIMING_REPORT_SCHEMA = "alchemi.part1-notebook-timing.v1"
BUNDLE_REQUIRED_FILES = (
    *BUNDLE_SOURCE_FILES,
    RUNTIME_CHECK_NAME,
    D3_CACHE_REPORT_NAME,
    TIMING_REPORT_NAME,
)

LOCAL_MARKDOWN_REFERENCES = LOCAL_NOTEBOOK_REFERENCES

SOURCE_PATHS = REQUIRED_TRACKED_SOURCE_PATHS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_run_manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid run manifest: {path}") from exc
    manifest = require_mapping(manifest, "run manifest")
    if manifest.get("schema") != WATER_RUN_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected run manifest schema: {manifest.get('schema')!r}")
    expected_fields = {"schema", "run_details", "settings", "checks", "files"}
    if set(manifest) != expected_fields:
        missing = sorted(expected_fields - set(manifest))
        unexpected = sorted(set(manifest) - expected_fields)
        raise ValueError(
            "run manifest has incorrect top-level fields: "
            f"missing={missing}, unexpected={unexpected}"
        )
    require_mapping(manifest.get("run_details"), "run manifest run details")
    require_mapping(manifest.get("settings"), "run manifest settings")
    require_mapping(manifest.get("checks"), "run manifest checks")
    if not isinstance(manifest.get("files"), list):
        raise ValueError("run manifest files must be a JSON array")
    return manifest


def validate_manifest_inventory(
    output_dir: Path, manifest: Mapping[str, object]
) -> dict[str, int]:
    """Verify the manifest against every regular output file recursively."""

    manifest_path = output_dir / RUN_MANIFEST_NAME
    actual_paths = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    actual = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in actual_paths
    }

    declared: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(manifest["files"]):
        record = require_mapping(raw_record, f"run manifest files[{index}]")
        relative_name = record.get("path")
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError(f"run manifest files[{index}].path must be nonempty text")
        relative_path = Path(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
        ):
            raise ValueError(f"non-canonical run manifest path: {relative_name!r}")
        if relative_name == RUN_MANIFEST_NAME:
            raise ValueError("run manifest must not inventory itself")
        if relative_name in declared:
            raise ValueError(f"duplicate run manifest file: {relative_name}")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ValueError(f"invalid byte size for {relative_name}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 for {relative_name}")
        declared[relative_name] = {"bytes": byte_count, "sha256": digest}

    missing = sorted(set(actual).difference(declared))
    unexpected = sorted(set(declared).difference(actual))
    if missing or unexpected:
        raise RuntimeError(
            "run manifest inventory differs from output directory: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative_name, observed in actual.items():
        record = declared[relative_name]
        if record["bytes"] != observed["bytes"]:
            raise RuntimeError(f"run manifest byte size mismatch: {relative_name}")
        if record["sha256"] != observed["sha256"]:
            raise RuntimeError(f"run manifest SHA-256 mismatch: {relative_name}")
    return {
        "file_count": len(actual),
        "total_bytes": sum(int(record["bytes"]) for record in actual.values()),
    }


def numeric_setting(
    settings: Mapping[str, object],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = settings.get(name)
    if isinstance(value, bool):
        raise ValueError(f"run manifest setting {name!r} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run manifest setting {name!r} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"run manifest setting {name!r} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"run manifest setting {name!r} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"run manifest setting {name!r} must be <= {maximum}")
    return number


def integer_setting(
    settings: Mapping[str, object], name: str, *, minimum: int = 1
) -> int:
    value = settings.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"run manifest setting {name!r} must be an integer >= {minimum}"
        )
    return value


def positive_numeric_setting(settings: Mapping[str, object], name: str) -> float:
    value = numeric_setting(settings, name, minimum=0.0)
    if value == 0.0:
        raise ValueError(f"run manifest setting {name!r} must be positive")
    return value


def validate_domain_producer_hashes(
    value: object,
    *,
    source_root: Path,
) -> dict[str, str]:
    """Match a saved domain campaign to its current repo files and D3 cache."""

    observed = dict(
        require_mapping(
            value,
            "domain bundle producer file SHA-256 values",
        )
    )
    expected_paths = (
        "scripts/part1_domain_plan.py",
        "scripts/part1_domain_run.py",
        "scripts/run_part1_domain_decomposition.sh",
        "scripts/slurm_part1_domain_decomposition.sbatch",
        "part-1-scalable-atomistic-workflows/aux/domain/packing.py",
        "part-1-scalable-atomistic-workflows/aux/domain/config.py",
        "part-1-scalable-atomistic-workflows/data/nci_atlas/nci-atlas-curves.csv.gz",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/manifest.json",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/structure.extxyz",
        "part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/SHA256SUMS",
    )
    expected_repo_files = {
        Path(relative).name: sha256_file(source_root / relative)
        for relative in expected_paths
    }
    remaining = dict(observed)
    for name, expected_sha256 in expected_repo_files.items():
        if remaining.pop(name, None) != expected_sha256:
            raise RuntimeError(
                f"domain bundle producer does not match the current tutorial: {name}"
            )
    if (
        len(remaining) != 1
        or next(iter(remaining.values()), None) != EXPECTED_D3_PARAMETER_SHA256
    ):
        raise RuntimeError(
            "domain bundle must identify one D3 parameter file with the pinned SHA-256"
        )
    return dict(sorted(observed.items()))


def expected_reference_bundle_details(
    source_root: Path,
) -> dict[str, dict[str, object] | None]:
    """Return exact identities for installed read-only Part 1 bundles."""

    part_root = source_root / "part-1-scalable-atomistic-workflows"
    experimental_dir = part_root / "reference" / "experimental_water_fundamentals"
    campaign_dir = part_root / "data" / "compute_lab_pipeline_campaign"
    domain_dir = part_root / "data" / "domain_decomposition" / "recorded"

    experimental_manifest = json.loads(
        (experimental_dir / "manifest.json").read_text(encoding="utf-8")
    )
    expected: dict[str, dict[str, object] | None] = {
        "experimental_reference_bundle": {
            "artifact_id": experimental_manifest.get("artifact_id"),
            "manifest_sha256": sha256_file(experimental_dir / "manifest.json"),
            "data_sha256": sha256_file(
                experimental_dir / "water_gas_phase_fundamentals.csv"
            ),
            "checksum_index_sha256": sha256_file(experimental_dir / "SHA256SUMS"),
        },
        "pipeline_campaign_bundle": None,
        "domain_decomposition_bundle": None,
    }

    campaign_files = {
        "manifest.json",
        "runs.csv",
        "SHA256SUMS",
    }
    present_campaign_files = {
        name for name in campaign_files if (campaign_dir / name).is_file()
    }
    if present_campaign_files and present_campaign_files != campaign_files:
        missing = sorted(campaign_files - present_campaign_files)
        raise FileNotFoundError(
            f"incomplete pipeline campaign bundle; missing {missing}"
        )
    if present_campaign_files:
        campaign_manifest = json.loads(
            (campaign_dir / "manifest.json").read_text(encoding="utf-8")
        )
        campaign_provenance = require_mapping(
            campaign_manifest.get("provenance"),
            "pipeline campaign manifest provenance",
        )
        expected["pipeline_campaign_bundle"] = {
            "artifact_id": campaign_manifest.get("artifact_id"),
            "manifest_sha256": sha256_file(campaign_dir / "manifest.json"),
            "runs_sha256": sha256_file(campaign_dir / "runs.csv"),
            "checksum_index_sha256": sha256_file(campaign_dir / "SHA256SUMS"),
            "producer_set_sha256": campaign_provenance.get("producer_set_sha256"),
        }

    fixed_atom_count = (
        DOMAIN_METHODOLOGY.fixed_molecules_per_species
        * DOMAIN_METHODOLOGY.atoms_per_composition_unit
    )
    domain_view = load_domain_lesson_view(
        domain_dir,
        expected_atom_count=fixed_atom_count,
        expected_world_sizes=DOMAIN_METHODOLOGY.campaign_world_sizes,
    )
    if domain_view.available:
        domain_job_records = require_mapping(
            domain_view.manifest.get("job_records"),
            "domain bundle job records",
        )
        reference_world_size = str(DOMAIN_METHODOLOGY.campaign_world_sizes[0])
        reference_job = require_mapping(
            domain_job_records.get(reference_world_size),
            f"domain bundle {reference_world_size}-GPU job record",
        )
        validate_domain_producer_hashes(
            reference_job.get("producer_files"),
            source_root=source_root,
        )
        expected["domain_decomposition_bundle"] = dict(
            require_mapping(domain_view.bundle_record, "domain bundle record")
        )
    return expected


def validate_run_details(
    run_manifest: Mapping[str, object],
    source_notebook: Path,
    source_root: Path,
) -> dict[str, object]:
    run_details = require_mapping(
        run_manifest["run_details"], "run manifest run details"
    )
    adsorption_manifest = (
        source_root
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "adsorption"
        / "cu111-important-molecules-v1"
        / "manifest.json"
    )
    nci_subset = (
        source_root
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "nci_atlas"
        / "nci-atlas-curves.csv.gz"
    )
    nci_subset_sha256 = sha256_file(nci_subset)
    if nci_subset_sha256 != NCI_ATLAS_SUBSET_SHA256:
        raise RuntimeError("the packaged NCI Atlas subset does not match its checksum")
    expected_exact = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "d3_parameter_file_sha256": EXPECTED_D3_PARAMETER_SHA256,
        "toolkit_core_commit": EXPECTED_TOOLKIT_CORE_COMMIT,
        "toolkit_ops_commit": EXPECTED_TOOLKIT_OPS_COMMIT,
        "aimnet": EXPECTED_AIMNET_VERSION,
        "sevennet": EXPECTED_SEVENNET_VERSION,
        "sevennet_checkpoint_source": SEVENNET_CHECKPOINT_URL,
        "sevennet_checkpoint_sha256": EXPECTED_SEVENNET_CHECKPOINT_SHA256,
        "sevennet_checkpoint_doi": SEVENNET_CHECKPOINT_DOI,
        "sevennet_task": SEVENNET_MODALITY,
        "sevennet_reference_method": SEVENNET_REFERENCE_METHOD,
        "adsorption_structure_manifest_sha256": sha256_file(adsorption_manifest),
        "nci_subset_sha256": nci_subset_sha256,
        "aimnet_checkpoint_identities": CHECKPOINT_IDENTITIES,
    }
    for name, expected in expected_exact.items():
        if run_details.get(name) != expected:
            raise RuntimeError(f"run detail {name!r} does not match the pinned value")
    if run_details.get("checkpoint_override") is not False:
        raise RuntimeError(
            "run manifest must use the pinned checkpoint, not an override"
        )
    if run_details.get("nci_checkpoints") != list(EXPECTED_NCI_CHECKPOINTS):
        raise RuntimeError(
            "run detail 'nci_checkpoints' does not match the four members"
        )

    torch_version = run_details.get("torch")
    if not isinstance(torch_version, str) or not torch_version.startswith(
        EXPECTED_TORCH_VERSION_PREFIX
    ):
        raise RuntimeError(
            "run manifest Torch version does not match the pinned 2.12.0 series"
        )

    live_notebook_sha256 = sha256_file(source_notebook)
    if run_details.get("notebook_sha256") != live_notebook_sha256:
        raise RuntimeError(
            "run manifest notebook SHA-256 does not match the live source notebook"
        )
    bundle_details = expected_reference_bundle_details(source_root)
    for name, expected in bundle_details.items():
        if expected is None:
            if run_details.get(name) is not None:
                raise RuntimeError(
                    f"run detail {name!r} must be null when no "
                    "validated bundle is installed"
                )
            continue
        observed = require_mapping(
            run_details.get(name), f"run manifest run detail {name}"
        )
        if dict(observed) != expected:
            raise RuntimeError(f"run detail {name!r} does not match the live bundle")
    return {
        **expected_exact,
        "checkpoint_override": False,
        "nci_checkpoints": list(EXPECTED_NCI_CHECKPOINTS),
        "torch": torch_version,
        "notebook_sha256": live_notebook_sha256,
        **bundle_details,
    }


def _nci_table(
    path: Path,
    *,
    columns: tuple[str, ...],
    text_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    try:
        table = pd.read_csv(
            path,
            dtype={column: "string" for column in text_columns},
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    observed_columns = tuple(table.columns)
    if observed_columns != columns:
        raise ValueError(
            f"{label} columns differ from the notebook schema: "
            f"observed={observed_columns}, expected={columns}"
        )
    if table.empty:
        raise ValueError(f"{label} must not be empty")
    return table


def _assert_nci_curve_identity(
    table: pd.DataFrame,
    expected_keys: pd.DataFrame,
    *,
    label: str,
) -> None:
    actual_keys = table[list(CURVE_KEY_COLUMNS)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual_keys,
            expected_keys,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            f"{label} does not contain the checked NCI curve identities in source order"
        ) from exc


def _assert_finite_nci_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    numeric = table[list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains a non-finite or non-numeric value")


def validate_nci_outputs(
    curves_path: Path,
    metrics_path: Path,
    ensemble_path: Path,
    run_manifest: Mapping[str, object],
    *,
    source_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Tie the three saved NCI tables to the checked 90-row source subset."""

    data_path = (
        source_root
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "nci_atlas"
        / "nci-atlas-curves.csv.gz"
    )
    source_data = load_nci_atlas_subset(data_path)
    source_sha256 = sha256_file(data_path)
    run_details = require_mapping(
        run_manifest["run_details"], "run manifest run details"
    )
    if run_details.get("nci_subset_sha256") != source_sha256:
        raise RuntimeError(
            "run detail 'nci_subset_sha256' does not match the checked NCI input"
        )
    if run_details.get("nci_checkpoints") != list(EXPECTED_NCI_CHECKPOINTS):
        raise RuntimeError(
            "run detail 'nci_checkpoints' does not match the four members"
        )

    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    expected_curve_keys = source_data[list(CURVE_KEY_COLUMNS)].drop_duplicates(
        ignore_index=True
    )
    expected_graph_count = len(source_data)
    expected_curve_count = len(expected_curve_keys)
    if settings.get("nci_graphs") != expected_graph_count:
        raise RuntimeError("run setting 'nci_graphs' does not match the checked input")
    if settings.get("nci_interaction_geometries") != expected_curve_count:
        raise RuntimeError(
            "run setting 'nci_interaction_geometries' does not match the checked input"
        )
    if settings.get("nci_reference_levels") != list(EXPECTED_NCI_REFERENCE_LEVELS):
        raise RuntimeError(
            "run setting 'nci_reference_levels' is not the expected pair"
        )
    validation_settings = require_mapping(
        settings.get("nci_validation"), "run manifest settings.nci_validation"
    )
    if dict(validation_settings) != NCI_VALIDATION.as_record():
        raise RuntimeError(
            "run setting 'nci_validation' does not match the checked method"
        )

    curve_text_columns = tuple(
        column for column in CURVE_KEY_COLUMNS if column != "scale"
    )
    curves = _nci_table(
        curves_path,
        columns=NCI_CURVE_COLUMNS,
        text_columns=curve_text_columns,
        label="NCI interaction curves",
    )
    if len(curves) != expected_curve_count:
        raise ValueError(
            f"NCI interaction curves have {len(curves)} rows; "
            f"expected {expected_curve_count}"
        )
    _assert_nci_curve_identity(
        curves,
        expected_curve_keys,
        label="NCI interaction curves",
    )
    _assert_finite_nci_columns(
        curves,
        (*NCI_COMPONENT_COLUMNS, "full_std", "dft_full", "dft_no_d3", "ccsd_t_cbs"),
        label="NCI interaction curves",
    )

    ensemble = _nci_table(
        ensemble_path,
        columns=NCI_ENSEMBLE_COLUMNS,
        text_columns=curve_text_columns,
        label="NCI ensemble curves",
    )
    expected_ensemble_rows = expected_curve_count * len(EXPECTED_NCI_CHECKPOINTS)
    if len(ensemble) != expected_ensemble_rows:
        raise ValueError(
            f"NCI ensemble curves have {len(ensemble)} rows; "
            f"expected {expected_ensemble_rows}"
        )
    member_values = pd.to_numeric(ensemble["member"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(member_values).all()
        or not np.equal(member_values, np.rint(member_values)).all()
    ):
        raise ValueError("NCI ensemble member values must be finite integers")
    ensemble = ensemble.assign(member=member_values.astype(np.int64))
    if tuple(sorted(ensemble["member"].unique())) != tuple(
        range(len(EXPECTED_NCI_CHECKPOINTS))
    ):
        raise ValueError("NCI ensemble curves do not contain members 0 through 3")
    for member in range(len(EXPECTED_NCI_CHECKPOINTS)):
        member_table = ensemble.loc[ensemble["member"] == member]
        _assert_nci_curve_identity(
            member_table,
            expected_curve_keys,
            label=f"NCI ensemble member {member}",
        )
    _assert_finite_nci_columns(
        ensemble,
        NCI_COMPONENT_COLUMNS,
        label="NCI ensemble curves",
    )

    averaged = mean_member_curves(
        ensemble,
        NCI_COMPONENT_COLUMNS,
        spread_component="full",
    )
    try:
        pd.testing.assert_frame_equal(
            curves[
                [
                    *CURVE_KEY_COLUMNS,
                    *NCI_COMPONENT_COLUMNS,
                    "full_std",
                ]
            ],
            averaged,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "NCI interaction curves do not reproduce the saved ensemble mean and spread"
        ) from exc

    graph_index = build_graph_index(source_data)
    expected_dft = reduce_fragment_energies(
        graph_index,
        {"dft_full": source_data["wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"]},
    )
    expected_cc = extract_repeated_interaction_reference(
        source_data,
        "ccsd_t_cbs_interaction_energy_kcal_mol",
        output_column="ccsd_t_cbs",
    )
    for column, expected in (
        ("dft_full", expected_dft["dft_full"]),
        ("ccsd_t_cbs", expected_cc["ccsd_t_cbs"]),
    ):
        try:
            np.testing.assert_allclose(
                curves[column].to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-10,
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"NCI curve reference column {column!r} does not match the checked input"
            ) from exc

    metrics = _nci_table(
        metrics_path,
        columns=NCI_METRIC_COLUMNS,
        text_columns=("system_name",),
        label="NCI interaction metrics",
    )
    expected_system_names = source_data["system_name"].drop_duplicates(
        ignore_index=True
    )
    if not metrics["system_name"].reset_index(drop=True).equals(expected_system_names):
        raise RuntimeError(
            "NCI interaction metrics do not contain the checked systems in source order"
        )
    metric_value_columns = tuple(NCI_METRIC_COLUMNS[1:])
    _assert_finite_nci_columns(
        metrics,
        metric_value_columns,
        label="NCI interaction metrics",
    )
    recomputed_metrics = interaction_metrics(
        curves,
        NCI_COMPARISONS,
        mean_columns={"ensemble spread": "full_std"},
    ).reset_index()
    try:
        pd.testing.assert_frame_equal(
            metrics,
            recomputed_metrics,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "NCI interaction metrics do not reproduce the saved curve errors"
        ) from exc

    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    recorded_maxima = {
        "nci_complete_max_MAE_vs_DFT_D3_kcal_mol": float(
            metrics["complete vs DFT-D3"].max()
        ),
        "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol": float(
            metrics["complete vs CC"].max()
        ),
    }
    for name, expected in recorded_maxima.items():
        observed = checks.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not np.isfinite(observed)
            or not np.isclose(observed, expected, rtol=0.0, atol=1.0e-12)
        ):
            raise RuntimeError(f"run check {name!r} does not match the NCI metrics")

    force_check = require_mapping(
        checks.get("nci_force_check"), "run manifest checks.nci_force_check"
    )
    expected_force_fields = {
        "atom_index",
        "cartesian_axis",
        "finite_difference_step_A",
        "official_total_energy_route",
        "official_analytic_force_route",
        "toolkit_analytic_force_route",
        "official_analytic_force_eV_A",
        "official_finite_difference_force_eV_A",
        "toolkit_analytic_force_eV_A",
        "official_analytic_vs_official_finite_difference_abs_error_eV_A",
        "toolkit_analytic_vs_official_analytic_abs_error_eV_A",
    }
    if set(force_check) != expected_force_fields:
        raise ValueError("run manifest NCI force-check fields differ from the schema")
    atom_index = force_check["atom_index"]
    cartesian_axis = force_check["cartesian_axis"]
    if (
        not isinstance(atom_index, int)
        or isinstance(atom_index, bool)
        or atom_index < 0
    ):
        raise ValueError("NCI force-check atom_index must be a non-negative integer")
    if (
        not isinstance(cartesian_axis, int)
        or isinstance(cartesian_axis, bool)
        or cartesian_axis not in (0, 1, 2)
    ):
        raise ValueError("NCI force-check cartesian_axis must be 0, 1, or 2")
    expected_routes = {
        "official_total_energy_route": (
            "official AIMNet2Calculator complete total energy with simple Coulomb "
            "and D3"
        ),
        "official_analytic_force_route": (
            "official AIMNet2Calculator complete-model force"
        ),
        "toolkit_analytic_force_route": (
            "Toolkit PipelineModelWrapper complete-model force"
        ),
    }
    for name, expected in expected_routes.items():
        if force_check.get(name) != expected:
            raise RuntimeError(
                f"NCI force-check route {name!r} is not the expected one"
            )
    force_numbers = {}
    for name in expected_force_fields - {
        "atom_index",
        "cartesian_axis",
        *expected_routes,
    }:
        value = force_check.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"NCI force-check value {name!r} must be finite")
        force_numbers[name] = float(value)
    if force_numbers["finite_difference_step_A"] != (
        NCI_VALIDATION.finite_difference_step_A
    ):
        raise RuntimeError("NCI force-check finite-difference step does not match")
    finite_difference_limit = NCI_VALIDATION.finite_difference_atol_eV_A + (
        NCI_VALIDATION.finite_difference_rtol
        * abs(force_numbers["official_finite_difference_force_eV_A"])
    )
    if (
        force_numbers["official_analytic_vs_official_finite_difference_abs_error_eV_A"]
        > finite_difference_limit
    ):
        raise RuntimeError("NCI official force does not match its energy derivative")
    if (
        force_numbers["toolkit_analytic_vs_official_analytic_abs_error_eV_A"]
        > NCI_VALIDATION.toolkit_official_force_atol_eV_A
    ):
        raise RuntimeError("NCI Toolkit force does not match the official force")

    return {
        "input_sha256": source_sha256,
        "graph_rows": expected_graph_count,
        "curve_rows": expected_curve_count,
        "ensemble_rows": expected_ensemble_rows,
        "systems": list(EXPECTED_SYSTEMS),
        "members": list(EXPECTED_NCI_CHECKPOINTS),
        "curve_columns": list(NCI_CURVE_COLUMNS),
        "ensemble_columns": list(NCI_ENSEMBLE_COLUMNS),
        "metric_columns": list(NCI_METRIC_COLUMNS),
    }


def validate_fused_stage_route(
    run_manifest: Mapping[str, object], *, warmup_steps: int, production_steps: int
) -> dict[str, int]:
    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    route = require_mapping(
        checks.get("fused_stage_route_counts"),
        "run manifest checks.fused_stage_route_counts",
    )
    expected = {
        f"status_{IR_WARMUP_STATUS}_warmup_steps": warmup_steps,
        f"status_{IR_PRODUCTION_STATUS}_production_steps": production_steps,
    }
    for name, value in route.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"fused stage route count {name!r} must be an integer")
    if dict(route) != expected:
        raise RuntimeError(
            "fused stage route counts do not exactly match the declared workloads"
        )
    return expected


def validate_composition_checks(
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Reapply every saved model-composition acceptance threshold."""

    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    scalar_limits = {
        "residual_serial_batch_max_abs_eV": RESIDUAL_SERIAL_BATCH_TOLERANCE_EV,
        "full_serial_batch_max_abs_eV": FULL_SERIAL_BATCH_TOLERANCE_EV,
        "component_closure_max_abs_eV": COMPONENT_CLOSURE_TOLERANCE_EV,
    }
    scalar_results = {}
    for name, limit in scalar_limits.items():
        value = numeric_setting(checks, name, minimum=0.0)
        if value >= limit:
            raise RuntimeError(f"composition check failed: {name} >= {limit}")
        scalar_results[name] = value

    nested_limits = {
        "official_calculator_agreement": {
            "energy_eV": COMPOSITION_ENERGY_AGREEMENT_TOLERANCE_EV,
            "interaction_energy_eV": (COMPOSITION_INTERACTION_AGREEMENT_TOLERANCE_EV),
            "forces_eV_A": COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
            "charges_e": COMPOSITION_CHARGE_AGREEMENT_TOLERANCE_E,
        },
        "analytic_coulomb": {
            "energy_eV": COMPOSITION_ANALYTIC_COULOMB_ENERGY_TOLERANCE_EV,
            "forces_eV_A": COMPOSITION_ANALYTIC_COULOMB_FORCE_TOLERANCE_EV_A,
        },
        "compiled_ir_eager_agreement": {
            "energy": COMPILED_EAGER_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_EAGER_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_EAGER_CHARGE_TOLERANCE_E,
        },
        "compiled_ir_repeat_agreement": {
            "energy": COMPILED_REPEAT_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_REPEAT_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_REPEAT_CHARGE_TOLERANCE_E,
        },
    }
    nested_results = {}
    for check_name, limits in nested_limits.items():
        values = require_mapping(
            checks.get(check_name), f"run manifest checks.{check_name}"
        )
        if set(values) != set(limits):
            raise RuntimeError(
                f"composition check {check_name!r} has unexpected fields"
            )
        checked = {}
        for name, limit in limits.items():
            value = numeric_setting(values, name, minimum=0.0)
            if value >= limit:
                raise RuntimeError(
                    f"composition check failed: {check_name}.{name} >= {limit}"
                )
            checked[name] = value
        nested_results[check_name] = checked

    force_route = checks.get("finite_difference_force_energy_route")
    if force_route != COMPOSITION_FD_ENERGY_ROUTE:
        raise RuntimeError("finite-difference force energy route is not recognized")
    force_step = numeric_setting(checks, "finite_difference_force_step_A", minimum=0.0)
    if not np.isclose(force_step, COMPOSITION_FD_STEP_A, rtol=0.0, atol=1e-15):
        raise RuntimeError("finite-difference force step does not match the tutorial")
    reference_force = numeric_setting(checks, "finite_difference_force_reference_eV_A")
    official_force = numeric_setting(
        checks, "finite_difference_force_official_analytic_eV_A"
    )
    pipeline_force = numeric_setting(checks, "finite_difference_force_pipeline_eV_A")
    official_force_error = numeric_setting(
        checks, "finite_difference_force_official_abs_error_eV_A", minimum=0.0
    )
    pipeline_force_error = numeric_setting(
        checks, "finite_difference_force_pipeline_abs_error_eV_A", minimum=0.0
    )
    reproduced_official_error = abs(reference_force - official_force)
    if not np.isclose(
        official_force_error, reproduced_official_error, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("official finite-difference force error was not reproduced")
    reproduced_pipeline_error = abs(reference_force - pipeline_force)
    if not np.isclose(
        pipeline_force_error, reproduced_pipeline_error, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("pipeline finite-difference force error was not reproduced")
    force_tolerance = COMPOSITION_FD_FORCE_TOLERANCE_EV_A
    if official_force_error >= force_tolerance:
        raise RuntimeError("official finite-difference force check failed")
    if pipeline_force_error >= force_tolerance:
        raise RuntimeError("pipeline finite-difference force check failed")

    return {
        **scalar_results,
        **nested_results,
        "finite_difference_force": {
            "energy_route": force_route,
            "step_A": force_step,
            "reference_eV_A": reference_force,
            "official_analytic_eV_A": official_force,
            "pipeline_eV_A": pipeline_force,
            "official_abs_error_eV_A": official_force_error,
            "pipeline_abs_error_eV_A": pipeline_force_error,
            "abs_tolerance_eV_A": force_tolerance,
        },
    }


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"cannot parse boolean values in {series.name!r}")
    return normalized == "true"


def distribution_record(name: str) -> dict[str, object]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None, "direct_url": None}
    direct_url = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "direct_url": json.loads(direct_url) if direct_url else None,
    }


def validate_packaged_source_identity(
    value: object,
    *,
    source_root: Path,
) -> dict[str, object]:
    """Match the runtime report to the exact tutorial checkout being validated."""

    source = require_mapping(value, "packaged runtime check source")
    expected_fields = {
        "clean_checkout",
        "repository_commit",
        "repository_tree",
        "manifest_path",
        "manifest_sha256",
        "files_sha256",
    }
    if set(source) != expected_fields:
        raise ValueError("packaged runtime source fields are incomplete")
    if source.get("clean_checkout") is not True:
        raise RuntimeError("packaged runtime did not use a clean tutorial checkout")
    if source.get("manifest_path") != SOURCE_MANIFEST_RELATIVE_PATH:
        raise RuntimeError(
            "packaged runtime source manifest path does not match Part 1"
        )

    source_root = source_root.resolve()
    source_paths = load_source_paths(source_root)
    manifest_sha256 = sha256_file(source_root / SOURCE_MANIFEST_RELATIVE_PATH)
    if source.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("packaged runtime source manifest SHA-256 does not match")

    files_sha256 = require_mapping(
        source.get("files_sha256"),
        "packaged runtime source file SHA-256 values",
    )
    if set(files_sha256) != set(source_paths):
        missing = sorted(set(source_paths).difference(files_sha256))
        unexpected = sorted(set(files_sha256).difference(source_paths))
        raise ValueError(
            "packaged runtime source file list differs from the source manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative in source_paths:
        observed = files_sha256.get(relative)
        expected = sha256_file(source_root / relative)
        if observed != expected:
            raise RuntimeError(
                f"packaged runtime source SHA-256 does not match: {relative}"
            )

    live_revision = git_source_revision(source_root)
    for name, expected in live_revision.items():
        observed = source.get(name)
        if (
            not isinstance(observed, str)
            or len(observed) != 40
            or any(character not in "0123456789abcdef" for character in observed)
        ):
            raise ValueError(f"packaged runtime source {name} is not a full Git SHA")
        if observed != expected:
            raise RuntimeError(
                f"packaged runtime source {name} does not match the live checkout"
            )

    return {
        "clean_checkout": True,
        **live_revision,
        "manifest_path": SOURCE_MANIFEST_RELATIVE_PATH,
        "manifest_sha256": manifest_sha256,
        "files_sha256": dict(files_sha256),
    }


def validate_packaged_runtime_check(
    path: Path,
    *,
    source_root: Path,
) -> dict[str, object]:
    """Validate the preflight report stored beside the packaged notebook run."""

    try:
        raw_report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid packaged runtime check: {path}") from exc
    report = require_mapping(raw_report, "packaged runtime check")
    expected_fields = {
        "schema",
        "source",
        "python",
        "python_executable",
        "base_environment",
        "versions",
        "resolved_scientific_versions",
        "commits",
        "cuda_available",
        "cuda_device",
        "jax_cuda_device",
        "packmol_binary",
        "packmol_check",
        "ovito_ase_check",
        "toolkit_ops_cuda_check",
        "module_files",
    }
    if set(report) != expected_fields:
        raise ValueError(
            "packaged runtime check fields differ from the expected schema"
        )
    if report.get("schema") != "alchemi.part1-runtime-check.v2":
        raise ValueError("packaged runtime check has an unexpected schema")
    source = validate_packaged_source_identity(
        report.get("source"),
        source_root=source_root,
    )

    versions = require_mapping(
        report.get("versions"), "packaged runtime check versions"
    )
    expected_version_names = {
        *EXPECTED_RUNTIME_VERSIONS,
        "packmol",
        "torch",
        "uv",
    }
    if set(versions) != expected_version_names:
        raise ValueError("packaged runtime version fields are incomplete")
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        if versions.get(name) != expected:
            raise RuntimeError(
                f"packaged runtime version for {name!r} does not match Part 1"
            )
    if versions.get("packmol") != EXPECTED_PACKMOL_VERSION:
        raise RuntimeError("packaged Packmol version does not match Part 1")
    torch_version = versions.get("torch")
    if not isinstance(torch_version, str) or not torch_version.startswith(
        EXPECTED_TORCH_VERSION_PREFIX
    ):
        raise RuntimeError("packaged Torch version does not match Part 1")
    uv_version = versions.get("uv")
    if not isinstance(uv_version, str):
        raise ValueError("packaged uv version must be text")
    try:
        uv_version_tuple = tuple(int(part) for part in uv_version.split(".")[:3])
    except ValueError as exc:
        raise ValueError("packaged uv version is invalid") from exc
    if len(uv_version_tuple) != 3 or uv_version_tuple < (0, 9, 26):
        raise RuntimeError("packaged uv version is older than 0.9.26")

    base_environment = report.get("base_environment")
    if (
        not isinstance(base_environment, str)
        or not base_environment.strip()
        or not Path(base_environment).is_absolute()
    ):
        raise ValueError(
            "packaged runtime base environment must be a non-empty absolute path"
        )

    scientific_versions = require_mapping(
        report.get("resolved_scientific_versions"),
        "packaged runtime scientific versions",
    )
    if set(scientific_versions) != set(RECORDED_SCIENTIFIC_VERSIONS) or any(
        not isinstance(value, str) or not value.strip()
        for value in scientific_versions.values()
    ):
        raise ValueError("packaged runtime scientific version fields are incomplete")

    commits = require_mapping(
        report.get("commits"), "packaged runtime check Toolkit commits"
    )
    expected_commits = {
        "nvalchemi-toolkit": EXPECTED_TOOLKIT_CORE_COMMIT,
        "nvalchemi-toolkit-ops": EXPECTED_TOOLKIT_OPS_COMMIT,
    }
    if dict(commits) != expected_commits:
        raise RuntimeError("packaged Toolkit commits do not match Part 1")

    if report.get("cuda_available") is not True:
        raise RuntimeError("packaged runtime check did not run with CUDA")
    for name in ("cuda_device", "jax_cuda_device", "packmol_binary"):
        value = report.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"packaged runtime check {name!r} must be non-empty text")

    ops_check = require_mapping(
        report.get("toolkit_ops_cuda_check"),
        "packaged runtime check Toolkit-Ops CUDA result",
    )
    expected_ops_fields = {
        "directed_edges",
        "jax_segments",
        "segments",
        "warp_segments",
    }
    if set(ops_check) != expected_ops_fields:
        raise ValueError("packaged Toolkit-Ops CUDA result fields are incomplete")
    for name, value in ops_check.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"packaged Toolkit-Ops CUDA result {name!r} must be positive"
            )

    packmol_check = require_mapping(
        report.get("packmol_check"), "packaged runtime check Packmol result"
    )
    expected_packmol_fields = {
        "atoms",
        "molecules",
        "net_charge_e",
        "packmol_precision_a",
        "density_from_mass_and_cell_g_cm3",
        "periodic_min_distance_lower_bound_a",
    }
    if set(packmol_check) != expected_packmol_fields:
        raise ValueError("packaged Packmol result fields are incomplete")
    for name in ("atoms", "molecules"):
        value = packmol_check.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"packaged Packmol result {name!r} must be positive")
    for name in expected_packmol_fields - {"atoms", "molecules"}:
        value = packmol_check.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"packaged Packmol result {name!r} must be finite")
    if not np.isclose(
        packmol_check["packmol_precision_a"],
        1.0e-3,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("packaged Packmol precision does not match the preflight")

    ovito_check = require_mapping(
        report.get("ovito_ase_check"), "packaged runtime check OVITO result"
    )
    if set(ovito_check) != {"structures", "particle_counts"}:
        raise ValueError("packaged OVITO result fields are incomplete")
    particle_counts = require_mapping(
        ovito_check.get("particle_counts"),
        "packaged runtime check OVITO particle counts",
    )
    if (
        not isinstance(ovito_check.get("structures"), int)
        or isinstance(ovito_check.get("structures"), bool)
        or ovito_check["structures"] != len(particle_counts)
        or not particle_counts
    ):
        raise ValueError("packaged OVITO structure count is inconsistent")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in particle_counts.values()
    ):
        raise ValueError("packaged OVITO particle counts must be positive integers")

    module_files = require_mapping(
        report.get("module_files"), "packaged runtime check module files"
    )
    expected_modules = {
        "aimnet",
        "e3nn",
        "jax",
        "nvalchemi",
        "nvalchemiops",
        "ovito",
        "physicsnemo",
        "sevenn",
        "torch",
        "warp",
    }
    if set(module_files) != expected_modules or any(
        not isinstance(value, str) or not value.strip()
        for value in module_files.values()
    ):
        raise ValueError("packaged runtime module files are incomplete")

    return {
        "schema": report["schema"],
        "sha256": sha256_file(path),
        "source": source,
        "base_environment": base_environment,
        "versions": dict(versions),
        "resolved_scientific_versions": dict(scientific_versions),
        "commits": dict(commits),
        "cuda_device": report["cuda_device"],
        "jax_cuda_device": report["jax_cuda_device"],
        "packmol_check": dict(packmol_check),
        "toolkit_ops_cuda_check": dict(ops_check),
    }


def validate_d3_cache_report(path: Path) -> dict[str, object]:
    """Validate the first-use D3 cache report stored with the run."""

    try:
        raw_report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid D3 cache report: {path}") from exc
    report = require_mapping(raw_report, "D3 cache report")
    expected_fields = {
        "schema",
        "parameter_file",
        "bytes",
        "sha256",
        "toolkit_version",
    }
    if set(report) != expected_fields:
        raise ValueError("D3 cache report fields differ from the expected schema")
    if report.get("schema") != "alchemi.part1-d3-cache.v1":
        raise ValueError("D3 cache report has an unexpected schema")
    parameter_file = report.get("parameter_file")
    if not isinstance(parameter_file, str) or not parameter_file.strip():
        raise ValueError("D3 cache report parameter_file must be non-empty text")
    byte_count = report.get("bytes")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 1
    ):
        raise ValueError("D3 cache report bytes must be a positive integer")
    if report.get("sha256") != EXPECTED_D3_PARAMETER_SHA256:
        raise RuntimeError("D3 cache report SHA-256 does not match Part 1")
    if report.get("toolkit_version") != EXPECTED_RUNTIME_VERSIONS["nvalchemi-toolkit"]:
        raise RuntimeError("D3 cache report Toolkit version does not match Part 1")
    return dict(report)


def _nonnegative_finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{label} must be a non-negative finite number")
    return float(value)


def validate_notebook_timing_report(
    path: Path,
    *,
    source_notebook: Path,
    executed_notebook: Path,
    expected_source_sha256: str | None = None,
    expected_executed_sha256: str | None = None,
) -> dict[str, object]:
    """Check the cell and stage timings written by the notebook runner."""

    try:
        raw_report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid notebook timing report: {path}") from exc
    report = require_mapping(raw_report, "notebook timing report")
    expected_fields = {
        "schema",
        "status",
        "started_utc",
        "finished_utc",
        "input_notebook",
        "input_notebook_sha256",
        "executed_notebook",
        "executed_notebook_sha256",
        "kernel",
        "code_cell_count_expected",
        "code_cells_started",
        "code_cells_completed",
        "code_cells_failed",
        "total_code_elapsed_s",
        "total_wall_elapsed_s",
        "cell_timing_boundary",
        "wall_timing_boundary",
        "runner_error_type",
        "runner_error_message",
        "stage_timings",
        "cell_timings",
    }
    if set(report) != expected_fields:
        raise ValueError(
            "notebook timing report fields differ from the expected schema"
        )
    if report.get("schema") != TIMING_REPORT_SCHEMA:
        raise ValueError("notebook timing report has an unexpected schema")
    if report.get("status") != "complete":
        raise RuntimeError("notebook timing report is not complete")
    if (
        report.get("runner_error_type") is not None
        or report.get("runner_error_message") is not None
    ):
        raise RuntimeError("complete notebook timing report contains a runner error")
    for field in (
        "started_utc",
        "finished_utc",
        "input_notebook",
        "executed_notebook",
        "kernel",
        "cell_timing_boundary",
        "wall_timing_boundary",
    ):
        value = report.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"notebook timing report {field} must be non-empty text")
    for field in ("started_utc", "finished_utc"):
        try:
            timestamp = datetime.fromisoformat(str(report[field]))
        except ValueError as exc:
            raise ValueError(
                f"notebook timing report {field} is not an ISO timestamp"
            ) from exc
        if timestamp.utcoffset() is None:
            raise ValueError(
                f"notebook timing report {field} must include a UTC offset"
            )
    source_sha256 = (
        sha256_file(source_notebook)
        if expected_source_sha256 is None
        else expected_source_sha256
    )
    if expected_executed_sha256 is None:
        executed = nbformat.read(executed_notebook, as_version=4)
        review = executed.metadata.get("alchemi_review")
        original_sha256 = (
            review.get("original_executed_sha256")
            if isinstance(review, Mapping)
            and review.get("kind") == "markdown-only-source-refresh"
            else None
        )
        executed_sha256 = (
            original_sha256
            if isinstance(original_sha256, str) and original_sha256
            else sha256_file(executed_notebook)
        )
    else:
        executed_sha256 = expected_executed_sha256
    if report.get("input_notebook_sha256") != source_sha256:
        raise RuntimeError("notebook timing report source SHA-256 does not match")
    if report.get("executed_notebook_sha256") != executed_sha256:
        raise RuntimeError("notebook timing report executed SHA-256 does not match")

    source = nbformat.read(source_notebook, as_version=4)
    expected_cells: list[tuple[int, int, str, int]] = []
    stage = 0
    code_index = 0
    observed_stages: list[int] = []
    for cell_index, cell in enumerate(source.cells):
        cell_id = str(cell.get("id", ""))
        if cell_id.startswith("stage-") and cell_id[6:].isdigit():
            stage = int(cell_id[6:])
            observed_stages.append(stage)
        if cell.cell_type == "code":
            code_index += 1
            expected_cells.append((cell_index, code_index, cell_id, stage))
    if observed_stages != list(range(1, 8)):
        raise RuntimeError(
            "source notebook stage IDs are not exactly stage-1 through stage-7"
        )

    cell_timings = report.get("cell_timings")
    if not isinstance(cell_timings, list) or len(cell_timings) != len(expected_cells):
        raise ValueError("notebook timing report does not contain every code cell")
    elapsed_by_stage: dict[int, float] = {}
    cells_by_stage: dict[int, int] = {}
    for expected, raw_record in zip(expected_cells, cell_timings, strict=True):
        record = require_mapping(raw_record, "notebook cell timing")
        record_fields = {
            "cell_index",
            "code_index",
            "cell_id",
            "stage",
            "stage_title",
            "first_line",
            "status",
            "started_utc",
            "elapsed_s",
            "error_type",
            "error_message",
        }
        if set(record) != record_fields:
            raise ValueError("notebook cell timing fields differ from the schema")
        observed_identity = (
            record.get("cell_index"),
            record.get("code_index"),
            record.get("cell_id"),
            record.get("stage"),
        )
        if observed_identity != expected:
            raise RuntimeError("notebook timing cell identity or stage does not match")
        if record.get("status") != "complete":
            raise RuntimeError("notebook timing report contains an incomplete cell")
        if (
            record.get("error_type") is not None
            or record.get("error_message") is not None
        ):
            raise RuntimeError("completed notebook timing cell contains an error")
        for field in ("stage_title", "first_line", "started_utc"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"notebook cell timing {field} must be non-empty text")
        try:
            cell_started = datetime.fromisoformat(str(record["started_utc"]))
        except ValueError as exc:
            raise ValueError(
                "notebook cell timing started_utc is not an ISO timestamp"
            ) from exc
        if cell_started.utcoffset() is None:
            raise ValueError(
                "notebook cell timing started_utc must include a UTC offset"
            )
        elapsed = _nonnegative_finite_number(
            record.get("elapsed_s"),
            "notebook cell elapsed_s",
        )
        elapsed_by_stage[expected[3]] = elapsed_by_stage.get(expected[3], 0.0) + elapsed
        cells_by_stage[expected[3]] = cells_by_stage.get(expected[3], 0) + 1

    count_fields = {
        "code_cell_count_expected": len(expected_cells),
        "code_cells_started": len(expected_cells),
        "code_cells_completed": len(expected_cells),
        "code_cells_failed": 0,
    }
    for field, expected in count_fields.items():
        value = report.get(field)
        if isinstance(value, bool) or value != expected:
            raise RuntimeError(f"notebook timing report {field} does not match")

    total_code_elapsed_s = _nonnegative_finite_number(
        report.get("total_code_elapsed_s"),
        "notebook total code elapsed_s",
    )
    recomputed_total = float(sum(elapsed_by_stage.values()))
    if not np.isclose(total_code_elapsed_s, recomputed_total, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("notebook timing total does not equal the cell timings")
    total_wall_elapsed_s = _nonnegative_finite_number(
        report.get("total_wall_elapsed_s"),
        "notebook total wall elapsed_s",
    )
    if total_wall_elapsed_s + 1.0e-9 < total_code_elapsed_s:
        raise RuntimeError("notebook wall time is shorter than its code-cell total")

    stage_timings = report.get("stage_timings")
    expected_stages = sorted(cells_by_stage)
    if not isinstance(stage_timings, list) or len(stage_timings) != len(
        expected_stages
    ):
        raise ValueError("notebook timing report has incomplete stage totals")
    checked_stages: list[dict[str, object]] = []
    for expected_stage, raw_stage in zip(expected_stages, stage_timings, strict=True):
        stage_record = require_mapping(raw_stage, "notebook stage timing")
        if set(stage_record) != {
            "stage",
            "title",
            "code_cells_started",
            "code_cells_completed",
            "code_cells_failed",
            "elapsed_s",
        }:
            raise ValueError("notebook stage timing fields differ from the schema")
        if stage_record.get("stage") != expected_stage:
            raise RuntimeError("notebook stage timing order does not match")
        title = stage_record.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("notebook stage timing title must be non-empty text")
        expected_count = cells_by_stage[expected_stage]
        for field, expected_count_value in (
            ("code_cells_started", expected_count),
            ("code_cells_completed", expected_count),
            ("code_cells_failed", 0),
        ):
            value = stage_record.get(field)
            if isinstance(value, bool) or value != expected_count_value:
                raise RuntimeError(f"notebook stage timing {field} does not match")
        elapsed = _nonnegative_finite_number(
            stage_record.get("elapsed_s"),
            "notebook stage elapsed_s",
        )
        if not np.isclose(
            elapsed,
            elapsed_by_stage[expected_stage],
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError("notebook stage total does not equal its cell timings")
        checked_stages.append(
            {
                "stage": expected_stage,
                "title": title,
                "code_cells": expected_count,
                "elapsed_s": elapsed,
            }
        )

    return {
        "schema": report["schema"],
        "sha256": sha256_file(path),
        "total_code_elapsed_s": total_code_elapsed_s,
        "total_wall_elapsed_s": total_wall_elapsed_s,
        "stage_timings": checked_stages,
    }


def runtime_details() -> dict[str, object]:
    record: dict[str, object] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: distribution_record(name)
            for name in (
                "nvalchemi-toolkit",
                "nvalchemi-toolkit-ops",
                "aimnet",
                "sevenn",
            )
        },
    }
    try:
        import torch

        record["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        record["torch"] = None

    aimnet_identities = verify_checkpoint_identities(CHECKPOINT_IDENTITIES)
    checkpoint_alias = "aimnet2-b973c-2025-d3_0"
    checkpoint = resolve_checkpoint_path(checkpoint_alias).resolve()
    record["aimnet_checkpoint"] = {
        "alias": checkpoint_alias,
        "path": str(checkpoint),
        **aimnet_identities[checkpoint_alias],
    }
    record["aimnet_checkpoints"] = aimnet_identities
    sevennet_checkpoint, sevennet_digest = resolve_sevennet_checkpoint()
    record["sevennet_checkpoint"] = {
        "source": SEVENNET_CHECKPOINT_URL,
        "path": str(sevennet_checkpoint),
        "bytes": sevennet_checkpoint.stat().st_size,
        "sha256": sevennet_digest,
    }
    return record


def comparable_cell(cell: object) -> dict[str, object]:
    return {
        "cell_type": cell.get("cell_type"),
        "id": cell.get("id"),
        "source": cell.get("source"),
        "attachments": cell.get("attachments", {}),
    }


def _reviewed_cells_match_source(
    notebook: nbformat.NotebookNode,
    source_notebook: nbformat.NotebookNode,
    *,
    executed_path: Path,
    source_path: Path,
) -> bool:
    """Accept only the declared path rebasing in a reviewed release copy."""

    review = notebook.metadata.get("alchemi_review")
    if not isinstance(review, Mapping):
        return False
    if review.get("kind") != "markdown-only-source-refresh":
        return False
    if review.get("code_sources_unchanged") is not True:
        return False
    observed_replacements = review.get("rebased_local_markdown_references")
    if not isinstance(observed_replacements, Mapping):
        return False
    expected_replacements = local_reference_replacements(
        source_dir=source_path.parent,
        output_dir=executed_path.parent,
    )
    if dict(observed_replacements) != expected_replacements:
        return False
    if len(notebook.cells) != len(source_notebook.cells):
        return False

    for reviewed_cell, source_cell in zip(
        notebook.cells, source_notebook.cells, strict=True
    ):
        if (
            reviewed_cell.get("cell_type") != source_cell.get("cell_type")
            or reviewed_cell.get("id") != source_cell.get("id")
            or reviewed_cell.get("attachments", {})
            != source_cell.get("attachments", {})
        ):
            return False
        reviewed_source = reviewed_cell.get("source")
        if reviewed_cell.get("cell_type") == "markdown":
            for original, replacement in expected_replacements.items():
                reviewed_source = reviewed_source.replace(replacement, original)
        if reviewed_source != source_cell.get("source"):
            return False
    return True


def validate_notebook(executed_path: Path, source_path: Path) -> int:
    notebook = nbformat.read(executed_path, as_version=4)
    source_notebook = nbformat.read(source_path, as_version=4)
    cells_match_exactly = [comparable_cell(cell) for cell in notebook.cells] == [
        comparable_cell(cell) for cell in source_notebook.cells
    ]
    if not cells_match_exactly and not _reviewed_cells_match_source(
        notebook,
        source_notebook,
        executed_path=executed_path,
        source_path=source_path,
    ):
        raise RuntimeError(
            "executed notebook cells or attachments do not match the source notebook"
        )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"executed notebook contains {len(errors)} error outputs")
    if any(cell.get("execution_count") is None for cell in code_cells):
        raise RuntimeError("one or more code cells were not executed")
    return len(code_cells)


SEVENNET_MAPPING_COLUMNS = (
    "component",
    "toolkit_shape",
    "sevennet_shape",
    "exact_match",
    "max_abs_difference",
    "units",
    "note",
)
SEVENNET_MAPPING_COMPONENTS = (
    "atomic numbers",
    "positions",
    "graph ownership",
    "atoms per graph",
    "directed COO edges",
    "periodic edge vectors",
    "cell volumes",
    "model task",
)
SEVENNET_AGREEMENT_COLUMNS = (
    "comparison",
    "structure",
    "atoms",
    "energy_difference_eV",
    "energy_difference_eV_per_atom",
    "max_force_component_difference_eV_A",
)
ADSORPTION_ENERGY_COLUMNS = (
    "molecule",
    "model_adslab_energy_eV",
    "model_clean_slab_energy_eV",
    "model_gas_energy_eV",
    "model_adsorption_energy_eV",
    "d3_adslab_energy_eV",
    "d3_clean_slab_energy_eV",
    "d3_gas_energy_eV",
    "d3_adsorption_energy_eV",
    "combined_adslab_energy_eV",
    "combined_clean_slab_energy_eV",
    "combined_gas_energy_eV",
    "adsorption_energy_eV",
    "fmax_eV_A",
    "force_rms_eV_A",
    "force_atoms",
)
ADSORPTION_FORCE_COLUMNS = (
    "structure",
    "role",
    "atom_index",
    "element",
    "is_adsorbate",
    "x_angstrom",
    "y_angstrom",
    "z_angstrom",
    "fx_eV_A",
    "fy_eV_A",
    "fz_eV_A",
    "force_norm_eV_A",
)


def validate_sevennet_graph_mapping(path: Path) -> dict[str, object]:
    """Validate the visible Toolkit-to-SevenNet graph-field mapping."""

    mapping = pd.read_csv(path)
    if tuple(mapping.columns) != SEVENNET_MAPPING_COLUMNS:
        raise RuntimeError(
            f"unexpected SevenNet graph-mapping columns: {list(mapping.columns)}"
        )
    if tuple(mapping["component"]) != SEVENNET_MAPPING_COMPONENTS:
        raise RuntimeError(
            "SevenNet graph mapping must contain each taught field once in lesson order"
        )
    matches = as_bool(mapping["exact_match"])
    differences = pd.to_numeric(mapping["max_abs_difference"], errors="coerce")
    difference_values = differences.to_numpy(dtype=float)
    if not np.isfinite(difference_values).all() or np.any(difference_values < 0.0):
        raise RuntimeError(
            "SevenNet graph mapping must report finite, nonnegative differences"
        )
    if not matches.all() or np.any(difference_values > SEVENNET_MAPPING_TOLERANCE):
        raise RuntimeError("one or more SevenNet graph fields do not match")
    if (mapping["toolkit_shape"].astype(str).str.len() == 0).any() or (
        mapping["sevennet_shape"].astype(str).str.len() == 0
    ).any():
        raise RuntimeError("SevenNet graph mapping contains a blank tensor shape")
    return {
        "mapping_passed": True,
        "rows": len(mapping),
        "max_abs_difference": float(difference_values.max(initial=0.0)),
    }


def _validate_sevennet_agreement(
    path: Path,
    *,
    structures: Mapping[str, object],
) -> dict[str, float | int]:
    agreement = pd.read_csv(path)
    if tuple(agreement.columns) != SEVENNET_AGREEMENT_COLUMNS:
        raise RuntimeError(
            "unexpected SevenNet numerical-agreement columns: "
            f"{list(agreement.columns)}"
        )
    expected_comparisons = (
        ("adapter output vs direct raw call",) * len(STRUCTURE_KEYS)
        + ("custom adapter vs official SevenNetCalculator",)
        + ("pipeline output vs explicit component sum",) * len(STRUCTURE_KEYS)
    )
    expected_structures = STRUCTURE_KEYS + (ADSLAB_KEYS["CO"],) + STRUCTURE_KEYS
    if tuple(agreement["comparison"]) != expected_comparisons:
        raise RuntimeError(
            "SevenNet numerical agreement must contain the raw-model, "
            "official-calculator, and Toolkit-pipeline comparisons in lesson order"
        )
    if tuple(agreement["structure"]) != expected_structures:
        raise RuntimeError(
            "SevenNet numerical agreement must contain all nine structures "
            "for both full-panel comparisons and the one official check"
        )
    expected_atoms = np.asarray(
        [len(structures[key]) for key in expected_structures], dtype=int
    )
    observed_atoms = pd.to_numeric(agreement["atoms"], errors="coerce").to_numpy()
    if not np.array_equal(observed_atoms, expected_atoms):
        raise RuntimeError("SevenNet numerical-agreement atom counts are incorrect")
    values = agreement[list(SEVENNET_AGREEMENT_COLUMNS[3:])].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise RuntimeError(
            "SevenNet numerical agreement must contain finite, nonnegative values"
        )
    if not np.allclose(
        values[:, 0] / expected_atoms,
        values[:, 1],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "SevenNet energy-per-atom differences do not match total differences"
        )
    max_energy = float(values[:, 1].max(initial=0.0))
    max_force = float(values[:, 2].max(initial=0.0))
    if max_energy >= SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM:
        raise RuntimeError(
            "SevenNet energy-per-atom agreement exceeds the lesson threshold"
        )
    if max_force >= SEVENNET_REPEAT_FORCE_TOL_EV_A:
        raise RuntimeError("SevenNet force agreement exceeds the lesson threshold")
    return {
        "rows": len(agreement),
        "max_energy_difference_eV_per_atom": max_energy,
        "max_force_difference_eV_A": max_force,
    }


def _validate_adsorption_forces(
    path: Path,
    *,
    structures: Mapping[str, object],
) -> dict[str, object]:
    forces = pd.read_csv(path)
    if tuple(forces.columns) != ADSORPTION_FORCE_COLUMNS:
        raise RuntimeError(f"unexpected surface-force columns: {list(forces.columns)}")
    expected_structure_column = tuple(
        key for key in STRUCTURE_KEYS for _ in range(len(structures[key]))
    )
    if tuple(forces["structure"]) != expected_structure_column:
        raise RuntimeError(
            "surface-force table must contain every atom of all nine structures "
            "in batch order"
        )
    adsorbate_flags = as_bool(forces["is_adsorbate"]).to_numpy(dtype=bool)
    numeric_columns = (
        "atom_index",
        "x_angstrom",
        "y_angstrom",
        "z_angstrom",
        "fx_eV_A",
        "fy_eV_A",
        "fz_eV_A",
        "force_norm_eV_A",
    )
    numeric = forces[list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("surface-force table contains a non-finite value")

    force_statistics: dict[str, dict[str, float | int]] = {}
    offset = 0
    for key in STRUCTURE_KEYS:
        atoms = structures[key]
        stop = offset + len(atoms)
        rows = forces.iloc[offset:stop]
        if not np.array_equal(
            rows["atom_index"].to_numpy(dtype=float),
            np.arange(len(atoms), dtype=float),
        ):
            raise RuntimeError(f"surface-force atom indices are incorrect for {key}")
        if not (rows["role"] == atoms.info["role"]).all():
            raise RuntimeError(f"surface-force roles are incorrect for {key}")
        if tuple(rows["element"]) != tuple(atoms.get_chemical_symbols()):
            raise RuntimeError(f"surface-force elements are incorrect for {key}")
        expected_adsorbate = np.asarray(atoms.arrays["is_adsorbate"], dtype=bool)
        if not np.array_equal(adsorbate_flags[offset:stop], expected_adsorbate):
            raise RuntimeError(f"surface-force adsorbate flags are incorrect for {key}")
        saved_positions = rows[["x_angstrom", "y_angstrom", "z_angstrom"]].to_numpy(
            dtype=float
        )
        if not np.allclose(
            saved_positions,
            atoms.positions,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError(
                f"surface-force coordinates do not match the pinned initial {key}"
            )
        vectors = rows[["fx_eV_A", "fy_eV_A", "fz_eV_A"]].to_numpy(dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(
            rows["force_norm_eV_A"].to_numpy(dtype=float),
            norms,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise RuntimeError(f"surface-force norms are inconsistent for {key}")
        force_statistics[key] = {
            "atoms": len(atoms),
            "fmax_eV_A": float(norms.max(initial=0.0)),
            "force_rms_eV_A": float(np.sqrt(np.mean(norms**2))),
        }
        offset = stop
    return {
        "rows": len(forces),
        "structures": len(STRUCTURE_KEYS),
        "max_fmax_eV_A": max(
            float(statistics["fmax_eV_A"]) for statistics in force_statistics.values()
        ),
        "per_structure": force_statistics,
    }


def _validate_adsorption_energies(
    path: Path,
    *,
    structures: Mapping[str, object],
    force_check: Mapping[str, object],
) -> dict[str, object]:
    energies = pd.read_csv(path)
    if tuple(energies.columns) != ADSORPTION_ENERGY_COLUMNS:
        raise RuntimeError(
            f"unexpected adsorption-energy columns: {list(energies.columns)}"
        )
    if tuple(energies["molecule"]) != ADSORBATES:
        raise RuntimeError(
            "adsorption-energy table must contain CO, CO2, NH3, and CH3OH in "
            "lesson order"
        )
    numeric = energies[list(ADSORPTION_ENERGY_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("adsorption-energy table contains a non-finite value")
    if np.any(energies[["fmax_eV_A", "force_rms_eV_A"]].to_numpy() < 0.0):
        raise RuntimeError("adsorption force statistics must be nonnegative")

    def require_formula(label: str, observed: object, expected: object) -> None:
        if not np.allclose(observed, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(f"adsorption energy formula is inconsistent for {label}")

    require_formula(
        "SevenNet component",
        energies["model_adsorption_energy_eV"],
        energies["model_adslab_energy_eV"]
        - energies["model_clean_slab_energy_eV"]
        - energies["model_gas_energy_eV"],
    )
    require_formula(
        "D3 component",
        energies["d3_adsorption_energy_eV"],
        energies["d3_adslab_energy_eV"]
        - energies["d3_clean_slab_energy_eV"]
        - energies["d3_gas_energy_eV"],
    )
    for role in ("adslab", "clean_slab", "gas"):
        require_formula(
            f"combined {role}",
            energies[f"combined_{role}_energy_eV"],
            energies[f"model_{role}_energy_eV"] + energies[f"d3_{role}_energy_eV"],
        )
    require_formula(
        "combined adsorption",
        energies["adsorption_energy_eV"],
        energies["combined_adslab_energy_eV"]
        - energies["combined_clean_slab_energy_eV"]
        - energies["combined_gas_energy_eV"],
    )
    require_formula(
        "component closure",
        energies["adsorption_energy_eV"],
        energies["model_adsorption_energy_eV"] + energies["d3_adsorption_energy_eV"],
    )

    per_structure = require_mapping(
        force_check["per_structure"], "surface force statistics"
    )
    for row in energies.itertuples(index=False):
        key = ADSLAB_KEYS[row.molecule]
        expected = require_mapping(per_structure[key], f"forces for {key}")
        if row.force_atoms != len(structures[key]):
            raise RuntimeError(f"adsorption force atom count is incorrect for {key}")
        if not np.isclose(
            row.fmax_eV_A,
            expected["fmax_eV_A"],
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise RuntimeError(f"adsorption fmax does not match full forces for {key}")
        if not np.isclose(
            row.force_rms_eV_A,
            expected["force_rms_eV_A"],
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                f"adsorption force RMS does not match full forces for {key}"
            )
    return {
        "molecules": len(energies),
        "max_adslab_fmax_eV_A": float(energies["fmax_eV_A"].max()),
        "adsorption_energy_eV": dict(
            zip(
                energies["molecule"],
                energies["adsorption_energy_eV"],
                strict=True,
            )
        ),
    }


def validate_sevennet_adsorption_outputs(
    energies_path: Path,
    forces_path: Path,
    mapping_path: Path,
    agreement_path: Path,
    run_manifest: Mapping[str, object],
    *,
    source_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Recompute the SevenNet adapter and fixed-geometry adsorption checks."""

    structure_dir = (
        source_root
        / "part-1-scalable-atomistic-workflows"
        / "data"
        / "adsorption"
        / "cu111-important-molecules-v1"
    )
    structures = load_initial_structure_set(structure_dir, verify_hashes=True)
    mapping_check = validate_sevennet_graph_mapping(mapping_path)
    agreement_check = _validate_sevennet_agreement(
        agreement_path, structures=structures
    )
    force_check = _validate_adsorption_forces(forces_path, structures=structures)
    energy_check = _validate_adsorption_energies(
        energies_path,
        structures=structures,
        force_check=force_check,
    )

    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    energy_tolerance = positive_numeric_setting(
        settings, "custom_adapter_energy_repeat_tolerance_eV_per_atom"
    )
    force_tolerance = positive_numeric_setting(
        settings, "custom_adapter_force_repeat_tolerance_eV_A"
    )
    if energy_tolerance != SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM:
        raise RuntimeError(
            "SevenNet energy repeat tolerance does not match the fixed lesson value"
        )
    if force_tolerance != SEVENNET_REPEAT_FORCE_TOL_EV_A:
        raise RuntimeError(
            "SevenNet force repeat tolerance does not match the fixed lesson value"
        )

    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    recorded = require_mapping(
        checks.get("sevennet_adapter"), "run manifest checks.sevennet_adapter"
    )
    expected_check_keys = {
        "graph_mapping_passed",
        "structures",
        "batches",
        "finite_outputs",
        "numerical_max_abs_energy_eV_per_atom",
        "numerical_max_abs_forces_eV_A",
        "max_combined_fmax_eV_A",
        "geometry_status",
        "molecules",
        "periodic_pbc",
    }
    if set(recorded) != expected_check_keys:
        raise RuntimeError("saved SevenNet adapter check has unexpected fields")
    expected_exact = {
        "graph_mapping_passed": mapping_check["mapping_passed"],
        "structures": len(STRUCTURE_KEYS),
        "batches": 2,
        "finite_outputs": True,
        "geometry_status": GEOMETRY_STATUS,
        "molecules": list(ADSORBATES),
        "periodic_pbc": [True, True, False],
    }
    for name, expected in expected_exact.items():
        if recorded.get(name) != expected:
            raise RuntimeError(
                f"saved SevenNet adapter {name} does not match validated outputs"
            )
    recorded_numeric = {
        "numerical_max_abs_energy_eV_per_atom": agreement_check[
            "max_energy_difference_eV_per_atom"
        ],
        "numerical_max_abs_forces_eV_A": agreement_check["max_force_difference_eV_A"],
        "max_combined_fmax_eV_A": force_check["max_fmax_eV_A"],
    }
    for name, expected in recorded_numeric.items():
        value = recorded.get(name)
        if isinstance(value, bool):
            raise RuntimeError(f"saved SevenNet adapter {name} must be numeric")
        try:
            observed = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"saved SevenNet adapter {name} must be numeric"
            ) from exc
        if not np.isfinite(observed) or observed < 0.0:
            raise RuntimeError(f"saved SevenNet adapter {name} must be finite")
        if not np.isclose(observed, expected, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError(f"saved SevenNet adapter {name} does not match the CSV")

    return {
        "graph_mapping": mapping_check,
        "numerical_agreement": agreement_check,
        "structures": len(STRUCTURE_KEYS),
        "batches": 2,
        "molecules": list(ADSORBATES),
        "geometry_status": GEOMETRY_STATUS,
        "forces": force_check,
        "adsorption": energy_check,
    }


def require_allclose(
    label: str,
    observed: object,
    expected: object,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-10,
) -> None:
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array,
        expected_array,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        raise RuntimeError(f"harmonic output mismatch: {label}")


def _validate_derived_csv(
    path: Path,
    expected: pd.DataFrame,
    *,
    save_index: bool,
    default_rtol: float,
    default_atol: float,
    column_tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, int | float]:
    """Compare one saved derived table with its recomputed DataFrame.

    CSV values are read as text so schema, row order, strings, booleans,
    integer spelling, and blank NaN fields are checked before numeric values
    receive a tolerance.
    """

    expected_csv = (
        expected.reset_index() if save_index else expected.reset_index(drop=True)
    )
    if save_index and expected.index.name is None:
        raise ValueError(f"recomputed table for {path.name} has an unnamed index")
    observed = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(observed.columns) != list(expected_csv.columns):
        raise RuntimeError(
            f"derived IR output mismatch: {path.name} has unexpected columns"
        )
    if len(observed) != len(expected_csv):
        raise RuntimeError(
            f"derived IR output mismatch: {path.name} has {len(observed)} rows; "
            f"expected {len(expected_csv)}"
        )

    tolerances = dict(column_tolerances or {})
    max_abs_difference = 0.0
    for column in expected_csv.columns:
        saved_tokens = observed[column].to_numpy(dtype=str)
        expected_series = expected_csv[column]
        if pd.api.types.is_bool_dtype(expected_series.dtype):
            expected_tokens = np.where(
                expected_series.to_numpy(dtype=bool), "True", "False"
            )
            if not np.array_equal(saved_tokens, expected_tokens):
                raise RuntimeError(
                    f"derived IR output mismatch: {path.name} boolean column "
                    f"{column!r} was not reproduced"
                )
            continue
        if pd.api.types.is_integer_dtype(expected_series.dtype):
            expected_tokens = np.asarray(
                [str(int(value)) for value in expected_series], dtype=str
            )
            if not np.array_equal(saved_tokens, expected_tokens):
                raise RuntimeError(
                    f"derived IR output mismatch: {path.name} integer column "
                    f"{column!r} was not reproduced"
                )
            continue
        if not pd.api.types.is_numeric_dtype(expected_series.dtype):
            expected_tokens = np.asarray(
                ["" if pd.isna(value) else str(value) for value in expected_series],
                dtype=str,
            )
            if not np.array_equal(saved_tokens, expected_tokens):
                raise RuntimeError(
                    f"derived IR output mismatch: {path.name} text column "
                    f"{column!r} was not reproduced"
                )
            continue

        expected_values = expected_series.to_numpy(dtype=float)
        if np.isinf(expected_values).any():
            raise RuntimeError(
                f"recomputed derived table contains infinity: {path.name} "
                f"column {column!r}"
            )
        expected_blank = np.isnan(expected_values)
        saved_blank = saved_tokens == ""
        if not np.array_equal(saved_blank, expected_blank):
            raise RuntimeError(
                f"derived IR output mismatch: {path.name} blank/NaN pattern in "
                f"column {column!r} was not reproduced"
            )
        finite = ~expected_blank
        if not finite.any():
            continue
        try:
            saved_values = saved_tokens[finite].astype(float)
        except ValueError as exc:
            raise RuntimeError(
                f"derived IR output mismatch: {path.name} numeric column "
                f"{column!r} contains non-numeric text"
            ) from exc
        if not np.isfinite(saved_values).all():
            raise RuntimeError(
                f"derived IR output mismatch: {path.name} numeric column "
                f"{column!r} contains a non-finite value"
            )
        expected_finite = expected_values[finite]
        rtol, atol = tolerances.get(column, (default_rtol, default_atol))
        differences = np.abs(saved_values - expected_finite)
        max_abs_difference = max(
            max_abs_difference,
            float(np.max(differences, initial=0.0)),
        )
        if not np.allclose(
            saved_values,
            expected_finite,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            raise RuntimeError(
                f"derived IR output mismatch: {path.name} numeric column "
                f"{column!r} was not reproduced"
            )

    return {
        "rows": len(expected_csv),
        "max_abs_difference": max_abs_difference,
    }


def _spectrum_windows_setting(
    settings: Mapping[str, object],
) -> dict[str, tuple[float, float]]:
    raw_windows = require_mapping(
        settings.get("spectrum_windows_cm1"),
        "run manifest settings.spectrum_windows_cm1",
    )
    if set(raw_windows) != {"H", "D"}:
        raise ValueError(
            "run manifest spectrum_windows_cm1 must contain exactly H and D"
        )
    windows: dict[str, tuple[float, float]] = {}
    for isotope in ("H", "D"):
        raw_window = raw_windows[isotope]
        if not isinstance(raw_window, list) or len(raw_window) != 2:
            raise ValueError(
                f"run manifest spectrum window {isotope!r} must be a two-value list"
            )
        if any(isinstance(value, bool) for value in raw_window):
            raise ValueError(
                f"run manifest spectrum window {isotope!r} must be numeric"
            )
        try:
            low, high = map(float, raw_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"run manifest spectrum window {isotope!r} must be numeric"
            ) from exc
        if not np.isfinite((low, high)).all() or not low < high:
            raise ValueError(f"run manifest spectrum window {isotope!r} must increase")
        windows[isotope] = (low, high)
    return windows


def validate_derived_ir_outputs(
    output_dir: Path,
    trajectory_path: Path,
    source_root: Path,
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute every saved MD/DFT-derived IR table from raw inputs."""

    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    dt_fs = positive_numeric_setting(settings, "dt_fs")
    segment_time_fs = positive_numeric_setting(settings, "spectrum_segment_time_fs")
    overlap = numeric_setting(settings, "spectrum_overlap", minimum=0.0)
    if overlap >= 1.0:
        raise ValueError("run manifest spectrum_overlap must be less than 1")
    windows_cm1 = _spectrum_windows_setting(settings)
    pair_temperature_tolerance = numeric_setting(
        settings, "pair_temperature_relative_tolerance", minimum=0.0
    )
    covalent_oh_cutoff = positive_numeric_setting(settings, "covalent_OH_cutoff_A")
    h_acceptor_cutoff = positive_numeric_setting(settings, "hbond_H_acceptor_cutoff_A")
    oo_cutoff = positive_numeric_setting(settings, "hbond_OO_cutoff_A")
    hbond_angle_cutoff = numeric_setting(
        settings,
        "hbond_angle_cutoff_deg",
        minimum=0.0,
        maximum=180.0,
    )
    coarse_mass_steps = integer_setting(
        settings, "h_to_d_coarse_mass_path_steps", minimum=2
    )
    fine_mass_steps = integer_setting(
        settings, "h_to_d_fine_mass_path_steps", minimum=2
    )
    if fine_mass_steps <= coarse_mass_steps:
        raise ValueError(
            "run manifest h_to_d_fine_mass_path_steps must exceed the coarse grid"
        )
    degeneracy_tolerance = positive_numeric_setting(
        settings, "h_to_d_degeneracy_tolerance_cm1"
    )
    cluster_reference_allowed = checks.get("initial_ring_persisted_all_frames")
    if not isinstance(cluster_reference_allowed, bool):
        raise ValueError(
            "run manifest check 'initial_ring_persisted_all_frames' must be boolean"
        )

    trajectory, labels = load_ir_trajectory(trajectory_path)
    expected_labels = ("H2O", "D2O", "(H2O)6", "(D2O)6")
    if tuple(labels) != expected_labels:
        raise RuntimeError(
            "derived IR output mismatch: trajectory labels are not the four "
            "declared systems in notebook order"
        )
    if float(trajectory.dt_fs) != dt_fs:
        raise RuntimeError(
            "derived IR output mismatch: trajectory timestep does not match "
            "the run manifest"
        )

    spectrum_analysis = ir_spectrum_metrics(
        trajectory.dipoles_e_angstrom,
        labels,
        dt_fs=dt_fs,
        segment_time_fs=segment_time_fs,
        overlap=overlap,
        region_windows_cm1=windows_cm1,
    )
    spectra = spectrum_analysis.spectra
    metrics = spectrum_analysis.metrics
    spectrum_table = pd.DataFrame({"wavenumber_cm-1": spectra[labels[0]][0]})
    for label in labels:
        wavenumber, intensity = spectra[label]
        if not np.array_equal(wavenumber, spectra[labels[0]][0]):
            raise RuntimeError("recomputed IR systems have different frequency grids")
        spectrum_table[f"{label}_PSD_arb"] = intensity

    atoms_per_graph = np.diff(np.asarray(trajectory.batch_ptr, dtype=int))
    nve_temperature = (
        2.0
        * np.asarray(trajectory.kinetic_energies_eV)
        / (3.0 * atoms_per_graph[None, :] * 8.617333262145e-5)
    )
    comparisons = ir_comparison_table(
        metrics,
        nve_temperature,
        labels,
        pair_temperature_relative_tolerance=pair_temperature_tolerance,
        cluster_reference_allowed=cluster_reference_allowed,
    ).table

    reference_root = (
        source_root / "part-1-scalable-atomistic-workflows" / "reference" / "artifacts"
    )
    reference_directories = {
        "H2O": "h2o",
        "D2O": "d2o",
        "(H2O)6": "h6",
        "(D2O)6": "d6",
    }
    references = {
        label: load_psi4_b973c_ir_artifact(
            reference_root / reference_directories[label]
        )
        for label in labels
    }
    dft_summary = reference_comparison_metrics(
        spectra,
        references,
        labels,
        dt_fs=dt_fs,
        segment_time_fs=segment_time_fs,
        region_windows_cm1=windows_cm1,
        cluster_reference_allowed=cluster_reference_allowed,
    ).metrics
    mode_map = h_to_d_mode_mapping_table(
        references,
        coarse_mass_path_steps=coarse_mass_steps,
        fine_mass_path_steps=fine_mass_steps,
        degeneracy_tolerance_cm1=degeneracy_tolerance,
        covalent_oh_cutoff_angstrom=covalent_oh_cutoff,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff,
        oo_cutoff_angstrom=oo_cutoff,
        hbond_angle_cutoff_deg=hbond_angle_cutoff,
    ).table
    topology_tables = {
        "water_ir_h6_topology_timeline.csv": topology_time_series(
            trajectory,
            2,
            h_acceptor_cutoff_angstrom=h_acceptor_cutoff,
            oo_cutoff_angstrom=oo_cutoff,
            hbond_angle_cutoff_deg=hbond_angle_cutoff,
        ),
        "water_ir_d6_topology_timeline.csv": topology_time_series(
            trajectory,
            3,
            h_acceptor_cutoff_angstrom=h_acceptor_cutoff,
            oo_cutoff_angstrom=oo_cutoff,
            hbond_angle_cutoff_deg=hbond_angle_cutoff,
        ),
    }

    table_results: dict[str, dict[str, int | float]] = {}
    spectrum_tolerances = {
        column: (DERIVED_PSD_RTOL, DERIVED_PSD_ATOL)
        for column in spectrum_table.columns
        if column.endswith("_PSD_arb")
    }
    table_results["water_ir_spectra.csv"] = _validate_derived_csv(
        output_dir / "water_ir_spectra.csv",
        spectrum_table,
        save_index=False,
        default_rtol=DERIVED_FREQUENCY_RTOL,
        default_atol=DERIVED_FREQUENCY_ATOL,
        column_tolerances=spectrum_tolerances,
    )
    for filename, expected, save_index in (
        ("water_ir_metrics.csv", metrics, True),
        ("water_ir_comparisons.csv", comparisons, True),
        ("water_ir_dft_comparison.csv", dft_summary, True),
    ):
        table_results[filename] = _validate_derived_csv(
            output_dir / filename,
            expected,
            save_index=save_index,
            default_rtol=DERIVED_FREQUENCY_RTOL,
            default_atol=DERIVED_FREQUENCY_ATOL,
        )
    for filename, expected in topology_tables.items():
        table_results[filename] = _validate_derived_csv(
            output_dir / filename,
            expected,
            save_index=False,
            default_rtol=DERIVED_TOPOLOGY_MODE_RTOL,
            default_atol=DERIVED_TOPOLOGY_MODE_ATOL,
        )
    table_results["water_ir_h_to_d_mode_map.csv"] = _validate_derived_csv(
        output_dir / "water_ir_h_to_d_mode_map.csv",
        mode_map,
        save_index=False,
        default_rtol=DERIVED_TOPOLOGY_MODE_RTOL,
        default_atol=DERIVED_TOPOLOGY_MODE_ATOL,
    )

    return {
        "tables": table_results,
        "cluster_reference_allowed": cluster_reference_allowed,
        "labels": list(labels),
        "spectrum_segment_time_fs": segment_time_fs,
        "spectrum_overlap": overlap,
        "h_to_d_mass_path_steps": {
            "coarse": coarse_mass_steps,
            "fine": fine_mass_steps,
        },
    }


def validate_harmonic_outputs(
    output_dir: Path,
    source_root: Path,
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the saved harmonic result from raw displaced structures."""

    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    checks = require_mapping(run_manifest["checks"], "run manifest checks")
    run_details = require_mapping(
        run_manifest["run_details"], "run manifest run details"
    )

    raw_steps = settings.get("harmonic_displacement_steps_bohr")
    if not isinstance(raw_steps, list) or len(raw_steps) < 2:
        raise ValueError("run manifest harmonic_displacement_steps_bohr must be a list")
    try:
        steps_bohr = np.asarray(raw_steps, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("harmonic displacement steps must be numeric") from exc
    if (
        not np.isfinite(steps_bohr).all()
        or np.any(steps_bohr <= 0.0)
        or np.any(np.diff(steps_bohr) >= 0.0)
        or np.unique(steps_bohr).size != steps_bohr.size
    ):
        raise ValueError(
            "harmonic displacement steps must be unique, positive, and coarse-to-fine"
        )
    require_allclose(
        "fixed harmonic displacement grid",
        steps_bohr,
        HARMONIC_DISPLACEMENT_STEPS_BOHR,
        rtol=0.0,
        atol=0.0,
    )
    selected_step_bohr = positive_numeric_setting(
        settings, "harmonic_selected_step_bohr"
    )
    require_allclose(
        "fixed selected harmonic step",
        selected_step_bohr,
        HARMONIC_SELECTED_STEP_BOHR,
        rtol=0.0,
        atol=0.0,
    )
    selected_matches = np.flatnonzero(
        np.isclose(steps_bohr, selected_step_bohr, rtol=0.0, atol=1.0e-12)
    )
    if selected_matches.size != 1:
        raise RuntimeError("selected harmonic step is not in the displacement grid")

    archive_path = output_dir / "water_monomer_aimnet_harmonic_ir.npz"
    archive_record = require_mapping(
        run_details.get("aimnet_harmonic_archive"),
        "run manifest run_details.aimnet_harmonic_archive",
    )
    if archive_record.get("path") != archive_path.name:
        raise RuntimeError("harmonic archive path does not match the saved file")
    if archive_record.get("sha256") != sha256_file(archive_path):
        raise RuntimeError("harmonic archive SHA-256 does not match the saved file")

    core_keys = {
        "geometry_angstrom",
        "atomic_numbers",
        "H2O_masses_u",
        "D2O_masses_u",
        "minimum_forces_eV_per_angstrom",
        "selected_step_bohr",
        "hessian_raw_eV_per_angstrom2",
        "hessian_eV_per_angstrom2",
        "hessian_hartree_per_bohr2",
        "dipole_derivative_3n_by_3_au",
        "H2O_frequencies_cm1",
        "D2O_frequencies_cm1",
        "H2O_ir_intensities_km_mol",
        "D2O_ir_intensities_km_mol",
        "H2O_mass_weighted_modes",
        "D2O_mass_weighted_modes",
    }
    sample_names = (
        "positions_angstrom",
        "forces_eV_per_angstrom",
        "charges_e",
        "dipoles_e_angstrom",
    )
    step_keys = {
        f"step_{step:.3f}".replace(".", "p") + f"_{name}"
        for step in steps_bohr
        for name in sample_names
    }
    with np.load(archive_path, allow_pickle=False) as loaded:
        expected_keys = core_keys | step_keys
        if set(loaded.files) != expected_keys:
            missing = sorted(expected_keys - set(loaded.files))
            unexpected = sorted(set(loaded.files) - expected_keys)
            raise RuntimeError(
                "harmonic archive fields differ from the declared format: "
                f"missing={missing}, unexpected={unexpected}"
            )
        archive = {name: np.array(loaded[name], copy=True) for name in loaded.files}

    geometry = np.asarray(archive["geometry_angstrom"], dtype=np.float64)
    raw_atomic_numbers = np.asarray(archive["atomic_numbers"])
    if not np.issubdtype(raw_atomic_numbers.dtype, np.integer):
        raise RuntimeError("harmonic archive atomic numbers must use an integer dtype")
    atomic_numbers = raw_atomic_numbers.astype(np.int64, copy=False)
    masses = {
        "H2O": np.asarray(archive["H2O_masses_u"], dtype=np.float64),
        "D2O": np.asarray(archive["D2O_masses_u"], dtype=np.float64),
    }
    if geometry.shape != (3, 3) or not np.isfinite(geometry).all():
        raise RuntimeError("harmonic geometry must be one finite water monomer")
    if not np.array_equal(atomic_numbers, np.array([8, 1, 1])):
        raise RuntimeError("harmonic archive atomic numbers are not H2O ordered O-H-H")
    for label, mass_values in masses.items():
        if mass_values.shape != (3,) or not np.isfinite(mass_values).all():
            raise RuntimeError(f"harmonic {label} masses are invalid")
    require_allclose(
        "oxygen isotope mass",
        masses["H2O"][:1],
        masses["D2O"][:1],
        rtol=0.0,
        atol=0.0,
    )
    if not np.all(masses["D2O"][1:] > masses["H2O"][1:]):
        raise RuntimeError("D2O hydrogen-site masses are not heavier than H2O")

    from ase.io import read as ase_read

    minimum_atoms = ase_read(
        output_dir / "water_monomer_harmonic_minimum.extxyz", index=0
    )
    if not np.array_equal(minimum_atoms.numbers, atomic_numbers):
        raise RuntimeError("saved harmonic minimum atomic numbers do not match")
    require_allclose(
        "saved harmonic minimum geometry",
        minimum_atoms.positions,
        geometry,
        rtol=0.0,
        atol=1.0e-8,
    )

    minimum_forces = np.asarray(
        archive["minimum_forces_eV_per_angstrom"], dtype=np.float64
    )
    if minimum_forces.shape != (3, 3) or not np.isfinite(minimum_forces).all():
        raise RuntimeError("harmonic minimum forces have the wrong shape")
    final_fmax = float(np.max(np.linalg.norm(minimum_forces, axis=1)))
    fmax_limit = positive_numeric_setting(settings, "harmonic_fmax_eV_A")
    require_allclose(
        "fixed harmonic minimum force target",
        fmax_limit,
        HARMONIC_FMAX_EV_A,
        rtol=0.0,
        atol=0.0,
    )
    charge_neutrality_tolerance = positive_numeric_setting(
        settings, "harmonic_charge_neutrality_tolerance_e"
    )
    require_allclose(
        "fixed harmonic charge-neutrality tolerance",
        charge_neutrality_tolerance,
        HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E,
        rtol=0.0,
        atol=0.0,
    )
    imaginary_floor = numeric_setting(
        settings, "harmonic_imaginary_floor_cm1", maximum=0.0
    )
    require_allclose(
        "fixed harmonic imaginary-frequency floor",
        imaginary_floor,
        HARMONIC_IMAGINARY_FLOOR_CM1,
        rtol=0.0,
        atol=0.0,
    )

    displacement_table = pd.read_csv(
        output_dir / "water_monomer_harmonic_displacements.csv"
    )
    required_displacement_columns = {
        "step_bohr",
        "step_angstrom",
        "structures_in_call",
        "max_realized_step_relative_error",
        "raw_H_max_antisymmetry_relative",
    }
    if set(displacement_table.columns) != required_displacement_columns:
        raise RuntimeError("harmonic displacement table has unexpected columns")
    if len(displacement_table) != len(steps_bohr):
        raise RuntimeError("harmonic displacement table has the wrong row count")
    require_allclose(
        "displacement step grid",
        displacement_table["step_bohr"].to_numpy(dtype=float),
        steps_bohr,
        rtol=0.0,
        atol=1.0e-12,
    )

    estimates = []
    for row_index, step_bohr in enumerate(steps_bohr):
        prefix = f"step_{step_bohr:.3f}".replace(".", "p")
        positions = np.asarray(
            archive[f"{prefix}_positions_angstrom"], dtype=np.float64
        )
        forces = np.asarray(
            archive[f"{prefix}_forces_eV_per_angstrom"], dtype=np.float64
        )
        charges = np.asarray(archive[f"{prefix}_charges_e"], dtype=np.float64)
        dipoles = np.asarray(archive[f"{prefix}_dipoles_e_angstrom"], dtype=np.float64)
        if positions.shape != (18, 3, 3) or forces.shape != (18, 3, 3):
            raise RuntimeError(
                "harmonic displaced positions or forces have wrong shape"
            )
        if charges.shape != (18, 3) or dipoles.shape != (18, 3):
            raise RuntimeError("harmonic displaced charges or dipoles have wrong shape")
        if not all(
            np.isfinite(values).all()
            for values in (positions, forces, charges, dipoles)
        ):
            raise RuntimeError("harmonic displaced arrays contain non-finite values")

        step_angstrom = float(step_bohr * ANGSTROM_PER_BOHR)
        expected_displacements = symmetric_cartesian_displacements(
            geometry, step_angstrom
        )
        expected_positions = np.concatenate(
            (
                expected_displacements.plus_angstrom,
                expected_displacements.minus_angstrom,
            ),
            axis=0,
        )
        require_allclose(
            f"{step_bohr:.3f} bohr displaced positions",
            positions,
            expected_positions,
            rtol=0.0,
            atol=2.0e-6,
        )
        recomputed_dipoles = molecular_dipoles_from_atomic_predictions(
            positions,
            charges,
            origin_angstrom=positions[:, 0, :],
            neutral_tolerance_e=charge_neutrality_tolerance,
        )
        require_allclose(
            f"{step_bohr:.3f} bohr dipoles",
            dipoles,
            recomputed_dipoles,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        n_coordinates = 9
        estimate = assemble_harmonic_ir_finite_difference(
            forces_plus_eV_per_angstrom=forces[:n_coordinates],
            forces_minus_eV_per_angstrom=forces[n_coordinates:],
            dipoles_plus_e_angstrom=recomputed_dipoles[:n_coordinates],
            dipoles_minus_e_angstrom=recomputed_dipoles[n_coordinates:],
            step_angstrom=step_angstrom,
        )
        estimates.append(estimate)

        plus_flat = positions[:n_coordinates].reshape(n_coordinates, n_coordinates)
        minus_flat = positions[n_coordinates:].reshape(n_coordinates, n_coordinates)
        coordinate = np.arange(n_coordinates)
        realized_steps = 0.5 * (
            plus_flat[coordinate, coordinate] - minus_flat[coordinate, coordinate]
        )
        realized_error = float(
            np.max(np.abs(realized_steps - step_angstrom)) / step_angstrom
        )
        row = displacement_table.iloc[row_index]
        if int(row["structures_in_call"]) != 18:
            raise RuntimeError(
                "each harmonic displacement pass must contain 18 systems"
            )
        require_allclose(
            "displacement step in angstrom",
            row["step_angstrom"],
            step_angstrom,
            rtol=0.0,
            atol=1.0e-12,
        )
        require_allclose(
            "realized displacement error",
            row["max_realized_step_relative_error"],
            realized_error,
            rtol=1.0e-7,
            atol=1.0e-12,
        )
        require_allclose(
            "raw Hessian antisymmetry",
            row["raw_H_max_antisymmetry_relative"],
            estimate.hessian.max_relative_antisymmetry,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    selected_index = int(selected_matches[0])
    selected_estimate = estimates[selected_index]
    require_allclose(
        "selected harmonic step",
        archive["selected_step_bohr"],
        np.asarray(selected_step_bohr),
        rtol=0.0,
        atol=1.0e-12,
    )
    selected_arrays = {
        "hessian_raw_eV_per_angstrom2": (
            selected_estimate.hessian.raw_hessian_eV_per_angstrom2
        ),
        "hessian_eV_per_angstrom2": (
            selected_estimate.hessian.hessian_eV_per_angstrom2
        ),
        "hessian_hartree_per_bohr2": (selected_estimate.hessian_hartree_per_bohr2),
        "dipole_derivative_3n_by_3_au": (
            selected_estimate.dipole_derivative_3n_by_3_au
        ),
    }
    for name, expected in selected_arrays.items():
        require_allclose(name, archive[name], expected, rtol=1.0e-11, atol=1.0e-11)

    analyses = {
        label: analyze_harmonic_ir(
            selected_estimate.hessian_hartree_per_bohr2,
            selected_estimate.dipole_derivative_3n_by_3_au,
            geometry,
            mass_values,
        )
        for label, mass_values in masses.items()
    }
    convergence = {
        label: summarize_harmonic_ir_convergence(estimates, geometry, mass_values)
        for label, mass_values in masses.items()
    }
    for label, analysis in analyses.items():
        require_allclose(
            f"{label} frequencies",
            archive[f"{label}_frequencies_cm1"],
            analysis.frequencies_cm1,
        )
        require_allclose(
            f"{label} intensities",
            archive[f"{label}_ir_intensities_km_mol"],
            analysis.ir_intensities_km_mol,
        )
        saved_modes = np.asarray(archive[f"{label}_mass_weighted_modes"])
        recomputed_modes = np.asarray(analysis.mass_weighted_modes)
        if saved_modes.shape != recomputed_modes.shape:
            raise RuntimeError(f"harmonic {label} mode array has the wrong shape")
        mode_overlaps = np.abs(
            np.sum(
                saved_modes.reshape(saved_modes.shape[0], -1)
                * recomputed_modes.reshape(recomputed_modes.shape[0], -1),
                axis=1,
            )
        )
        if not np.allclose(mode_overlaps, 1.0, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"harmonic {label} saved modes were not reproduced")

    convergence_table = pd.read_csv(
        output_dir / "water_monomer_harmonic_convergence.csv"
    ).set_index("isotopologue")
    if set(convergence_table.index) != {"H2O", "D2O"}:
        raise RuntimeError("harmonic convergence table must contain H2O and D2O")
    frequency_column = "max |frequency change|, 0.010→0.005 bohr (cm-1)"
    intensity_column = "max |intensity change|, 0.010→0.005 bohr (km/mol)"
    overlap_column = "minimum same-mode overlap"
    expected_convergence_columns = {
        frequency_column,
        intensity_column,
        overlap_column,
    }
    if set(convergence_table.columns) != expected_convergence_columns:
        raise RuntimeError("harmonic convergence table has unexpected columns")
    for label, summary in convergence.items():
        require_allclose(
            f"{label} final frequency change",
            convergence_table.loc[label, frequency_column],
            summary.frequency_max_abs_change_cm1[-1],
        )
        require_allclose(
            f"{label} final intensity change",
            convergence_table.loc[label, intensity_column],
            summary.ir_intensity_max_abs_change_km_mol[-1],
        )
        require_allclose(
            f"{label} final mode overlap",
            convergence_table.loc[label, overlap_column],
            summary.minimum_same_index_mode_squared_overlap[-1],
        )

    frequency_tolerance = positive_numeric_setting(
        settings, "harmonic_frequency_step_tolerance_cm1"
    )
    intensity_absolute_tolerance = positive_numeric_setting(
        settings, "harmonic_intensity_step_abs_tolerance_km_mol"
    )
    intensity_relative_tolerance = positive_numeric_setting(
        settings, "harmonic_intensity_step_relative_tolerance"
    )
    mode_overlap_minimum = numeric_setting(
        settings, "harmonic_mode_overlap_min", minimum=0.0, maximum=1.0
    )
    antisymmetry_limit = positive_numeric_setting(
        settings, "harmonic_hessian_antisymmetry_relative_max"
    )
    fixed_tolerances = {
        "harmonic frequency tolerance": (
            frequency_tolerance,
            HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1,
        ),
        "harmonic intensity absolute tolerance": (
            intensity_absolute_tolerance,
            HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL,
        ),
        "harmonic intensity relative tolerance": (
            intensity_relative_tolerance,
            HARMONIC_INTENSITY_STEP_REL_TOLERANCE,
        ),
        "harmonic mode overlap minimum": (
            mode_overlap_minimum,
            HARMONIC_MODE_OVERLAP_MIN,
        ),
        "harmonic Hessian antisymmetry limit": (
            antisymmetry_limit,
            HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX,
        ),
    }
    for label, (observed_value, expected_value) in fixed_tolerances.items():
        require_allclose(
            f"fixed {label}",
            observed_value,
            expected_value,
            rtol=0.0,
            atol=0.0,
        )

    def intensity_is_stable(label: str) -> bool:
        summary = convergence[label]
        absolute = summary.ir_intensity_abs_change_km_mol[-1]
        relative = summary.ir_intensity_relative_change[-1]
        return bool(
            np.all(
                (absolute <= intensity_absolute_tolerance)
                | (relative <= intensity_relative_tolerance)
            )
        )

    recomputed_checks = {
        "tight minimum": final_fmax <= fmax_limit,
        "H2O frequency step stability": bool(
            convergence["H2O"].frequency_max_abs_change_cm1[-1] <= frequency_tolerance
        ),
        "D2O frequency step stability": bool(
            convergence["D2O"].frequency_max_abs_change_cm1[-1] <= frequency_tolerance
        ),
        "H2O intensity step stability": intensity_is_stable("H2O"),
        "D2O intensity step stability": intensity_is_stable("D2O"),
        "mode continuity": bool(
            min(
                convergence["H2O"].minimum_same_index_mode_squared_overlap[-1],
                convergence["D2O"].minimum_same_index_mode_squared_overlap[-1],
            )
            >= mode_overlap_minimum
        ),
        "Hessian symmetry": bool(
            selected_estimate.hessian.max_relative_antisymmetry <= antisymmetry_limit
        ),
        "no significant imaginary modes": bool(
            all(
                np.all(analysis.frequencies_cm1 >= imaginary_floor)
                for analysis in analyses.values()
            )
        ),
    }
    recorded_checks = require_mapping(
        checks.get("harmonic_checks"), "run manifest checks.harmonic_checks"
    )
    if dict(recorded_checks) != recomputed_checks:
        raise RuntimeError("run manifest harmonic checks were not reproduced")

    checks_table = pd.read_csv(
        output_dir / "water_monomer_harmonic_checks.csv"
    ).set_index("check")
    if set(checks_table.index) != set(recomputed_checks):
        raise RuntimeError("harmonic checks table has unexpected rows")
    if set(checks_table.columns) != {"passed"}:
        raise RuntimeError("harmonic checks table has unexpected columns")
    saved_check_values = as_bool(checks_table["passed"])
    for name, expected in recomputed_checks.items():
        if bool(saved_check_values.loc[name]) != expected:
            raise RuntimeError(f"harmonic check table was not reproduced: {name}")

    comparison_reported = bool(all(recomputed_checks.values()))
    recorded_reported = checks.get("harmonic_comparison_reported")
    if (
        not isinstance(recorded_reported, bool)
        or recorded_reported != comparison_reported
    ):
        raise RuntimeError("harmonic comparison reporting state was not reproduced")
    plot_exists = (output_dir / HARMONIC_COMPARISON_PLOT).is_file()
    if plot_exists != comparison_reported:
        expected_state = "present" if comparison_reported else "absent"
        raise RuntimeError(
            f"harmonic comparison plot must be {expected_state} for this run"
        )
    if plot_exists:
        plot_bytes = (output_dir / HARMONIC_COMPARISON_PLOT).read_bytes()
        png_signature = b"\x89PNG\r\n\x1a\n"
        if (
            len(plot_bytes) < 33
            or not plot_bytes.startswith(png_signature)
            or plot_bytes[12:16] != b"IHDR"
            or plot_bytes[-8:-4] != b"IEND"
        ):
            raise RuntimeError("harmonic comparison plot is not a valid PNG file")

    require_allclose(
        "harmonic final fmax check",
        numeric_setting(checks, "harmonic_final_fmax_eV_A", minimum=0.0),
        final_fmax,
    )
    require_allclose(
        "selected Hessian antisymmetry check",
        numeric_setting(
            checks,
            "harmonic_selected_Hessian_antisymmetry_relative",
            minimum=0.0,
        ),
        selected_estimate.hessian.max_relative_antisymmetry,
    )
    recorded_frequency_change = require_mapping(
        checks.get("harmonic_final_frequency_step_change_cm1"),
        "run manifest harmonic final frequency changes",
    )
    recorded_intensity_change = require_mapping(
        checks.get("harmonic_final_intensity_step_change_km_mol"),
        "run manifest harmonic final intensity changes",
    )
    expected_isotopologues = {"H2O", "D2O"}
    if set(recorded_frequency_change) != expected_isotopologues:
        raise RuntimeError("harmonic final frequency change keys are invalid")
    if set(recorded_intensity_change) != expected_isotopologues:
        raise RuntimeError("harmonic final intensity change keys are invalid")
    for label in ("H2O", "D2O"):
        require_allclose(
            f"{label} manifest final frequency change",
            recorded_frequency_change.get(label),
            convergence[label].frequency_max_abs_change_cm1[-1],
        )
        require_allclose(
            f"{label} manifest final intensity change",
            recorded_intensity_change.get(label),
            convergence[label].ir_intensity_max_abs_change_km_mol[-1],
        )

    reference_root = source_root / "part-1-scalable-atomistic-workflows" / "reference"
    references = {
        label: load_psi4_b973c_ir_artifact(reference_root / "artifacts" / directory)
        for label, directory in (("H2O", "h2o"), ("D2O", "d2o"))
    }
    observed = load_experimental_water_fundamentals(
        reference_root / "experimental_water_fundamentals"
    ).set_index(["isotopologue", "mode"])
    mode_order = ("symmetric_stretch", "bend", "antisymmetric_stretch")
    mode_number = {
        "symmetric_stretch": 1,
        "bend": 2,
        "antisymmetric_stretch": 3,
    }
    expected_rows = []
    for label in ("H2O", "D2O"):
        aimnet_labels = label_water_monomer_modes(
            geometry,
            atomic_numbers,
            masses[label],
            analyses[label].mass_weighted_modes,
        )
        dft_labels = reference_water_monomer_mode_labels(references[label])
        for mode in mode_order:
            aimnet_index = aimnet_labels.index(mode)
            dft_index = dft_labels.index(mode)
            aimnet_frequency = float(analyses[label].frequencies_cm1[aimnet_index])
            dft_frequency = float(references[label].frequencies_cm1[dft_index])
            observed_frequency = float(observed.loc[(label, mode), "wavenumber_cm1"])
            expected_rows.append(
                {
                    "system": label,
                    "mode": f"ν{mode_number[mode]} {mode.replace('_', ' ')}",
                    "AIMNet+Coulomb+D3_harmonic_cm-1": aimnet_frequency,
                    "AIMNet_point_charge_IR_km_mol": float(
                        analyses[label].ir_intensities_km_mol[aimnet_index]
                    ),
                    "B97-3c_harmonic_cm-1": dft_frequency,
                    "B97-3c_IR_intensity_km_mol": float(
                        references[label].ir_intensities_km_mol[dft_index]
                    ),
                    "observed_gas_cm-1": observed_frequency,
                    "AIMNet+Coulomb+D3_minus_B97-3c_cm-1": (
                        aimnet_frequency - dft_frequency
                    ),
                    "B97-3c_minus_observed_cm-1": (dft_frequency - observed_frequency),
                }
            )
    expected_comparison = pd.DataFrame(expected_rows)
    saved_comparison = pd.read_csv(output_dir / "water_monomer_harmonic_comparison.csv")
    if list(saved_comparison.columns) != list(expected_comparison.columns):
        raise RuntimeError("harmonic comparison table has unexpected columns")
    frequency_mae = float(
        expected_comparison["AIMNet+Coulomb+D3_minus_B97-3c_cm-1"].abs().mean()
    )
    recorded_frequency_mae = checks.get("harmonic_frequency_MAE_vs_B97_3c_cm1")
    if comparison_reported:
        if not saved_comparison[["system", "mode"]].equals(
            expected_comparison[["system", "mode"]]
        ):
            raise RuntimeError(
                "harmonic comparison mode assignments were not reproduced"
            )
        numeric_columns = [
            name
            for name in expected_comparison.columns
            if name not in {"system", "mode"}
        ]
        require_allclose(
            "harmonic comparison numeric table",
            saved_comparison[numeric_columns].to_numpy(dtype=float),
            expected_comparison[numeric_columns].to_numpy(dtype=float),
            rtol=1.0e-10,
            atol=1.0e-8,
        )
        require_allclose(
            "harmonic frequency MAE check",
            numeric_setting(
                checks,
                "harmonic_frequency_MAE_vs_B97_3c_cm1",
                minimum=0.0,
            ),
            frequency_mae,
        )
    else:
        if not saved_comparison.empty:
            raise RuntimeError(
                "unreported harmonic comparison table must contain no data rows"
            )
        if recorded_frequency_mae is not None:
            raise RuntimeError("unreported harmonic frequency MAE must be null")

    return {
        "comparison_reported": comparison_reported,
        "comparison_plot_present": plot_exists,
        "final_fmax_eV_A": final_fmax,
        "selected_step_bohr": selected_step_bohr,
        "selected_hessian_antisymmetry_relative": (
            selected_estimate.hessian.max_relative_antisymmetry
        ),
        "frequency_MAE_vs_B97_3c_cm1": (frequency_mae if comparison_reported else None),
        "candidate_frequency_MAE_vs_B97_3c_cm1": frequency_mae,
        "checks": recomputed_checks,
    }


def validate_trajectory(
    path: Path, *, expected_frames: int, expected_dt_fs: float
) -> dict[str, list[int]]:
    expected = {
        "dipoles_e_angstrom": (expected_frames, 4, 3),
        "charge_sums_e": (expected_frames, 4),
        "kinetic_energies_eV": (expected_frames, 4),
        "total_energies_eV": (expected_frames, 4),
        "positions_angstrom": (expected_frames, 42, 3),
    }
    with np.load(path, allow_pickle=False) as arrays:
        observed = {name: tuple(arrays[name].shape) for name in expected}
        if observed != expected:
            raise RuntimeError(f"trajectory shape mismatch: {observed!r}")
        for name in expected:
            if not np.isfinite(arrays[name]).all():
                raise RuntimeError(f"non-finite trajectory array: {name}")
        observed_dt_fs = float(np.asarray(arrays["dt_fs"]).reshape(()))
        if observed_dt_fs != expected_dt_fs:
            raise RuntimeError(
                f"trajectory timestep {observed_dt_fs} does not match "
                f"run manifest setting {expected_dt_fs}"
            )
    return {name: list(shape) for name, shape in observed.items()}


def cluster_check_from_trajectory(
    arrays: object,
    graph_index: int,
    *,
    oxygen_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> dict[str, object]:
    batch_ptr = arrays["batch_ptr"]
    start, stop = map(int, batch_ptr[graph_index : graph_index + 2])
    numbers = arrays["atomic_numbers"][start:stop]
    # The recorder stores positions as float32. The notebook topology helper
    # promotes them to float64 before calculating distances, so the independent
    # recomputation must do the same rather than compare float32 and float64
    # norms under a near-bitwise tolerance.
    frames = np.asarray(arrays["positions_angstrom"][:, start:stop], dtype=np.float64)
    oxygen = np.flatnonzero(numbers == 8)
    hydrogen = np.flatnonzero(numbers == 1)
    oxygen_frames = frames[:, oxygen]
    hydrogen_frames = frames[:, hydrogen]
    assignment = np.argmin(
        np.linalg.norm(frames[0, hydrogen, None] - frames[0, oxygen][None, :], axis=-1),
        axis=1,
    )
    oh_distance = np.linalg.norm(
        hydrogen_frames - oxygen_frames[:, assignment], axis=-1
    )

    oxygen_distance = np.linalg.norm(
        oxygen_frames[:, :, None] - oxygen_frames[:, None, :], axis=-1
    )
    connected = oxygen_distance < oxygen_cutoff_angstrom
    reachability = connected.copy()
    for _ in range(len(oxygen)):
        reachability |= (
            np.matmul(reachability.astype(np.uint8), connected.astype(np.uint8)) > 0
        )
    component_count = np.ones(frames.shape[0], dtype=int)
    for node in range(1, len(oxygen)):
        component_count += ~reachability[:, node, :node].any(axis=1)
    max_oxygen_components = int(component_count.max())

    donor_oxygen = oxygen_frames[:, assignment]
    h_to_donor = donor_oxygen - hydrogen_frames
    h_to_acceptor = oxygen_frames[:, None] - hydrogen_frames[:, :, None]
    h_acceptor_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    cosine = np.sum(h_to_donor[:, :, None] * h_to_acceptor, axis=-1)
    cosine /= np.linalg.norm(h_to_donor, axis=-1)[:, :, None]
    cosine /= np.linalg.norm(h_to_acceptor, axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    oo_distance = np.linalg.norm(
        donor_oxygen[:, :, None] - oxygen_frames[:, None], axis=-1
    )
    is_donor = np.arange(len(oxygen))[None, None, :] == assignment[None, :, None]
    hbond = (
        (h_acceptor_distance <= h_acceptor_cutoff_angstrom)
        & (oo_distance <= oo_cutoff_angstrom)
        & (angle >= hbond_angle_cutoff_deg)
        & ~is_donor
    )
    adjacency = np.zeros((frames.shape[0], len(oxygen), len(oxygen)), dtype=bool)
    for hydrogen_index, donor_index in enumerate(assignment):
        adjacency[:, donor_index] |= hbond[:, hydrogen_index]

    initial_cycle = np.zeros(frames.shape[0], dtype=bool)
    import itertools

    for tail in itertools.permutations(range(1, len(oxygen))):
        nodes = (0, *tail)
        sources = np.asarray(nodes)
        targets = np.asarray((*tail, 0))
        if adjacency[0, sources, targets].all():
            initial_cycle = adjacency[:, sources, targets].all(axis=1)
            break
    return {
        "max_OH_angstrom": float(oh_distance.max()),
        "max_oxygen_components": max_oxygen_components,
        "all_frames_connected": max_oxygen_components == 1,
        "initial_ring_fraction": float(initial_cycle.mean()),
        "all_frames_initial_ring": bool(initial_cycle.all()),
    }


def validate_scientific_checks(
    output_dir: Path,
    trajectory_path: Path,
    run_manifest: Mapping[str, object],
) -> dict[str, object]:
    settings = require_mapping(run_manifest["settings"], "run manifest settings")
    recorded_checks = require_mapping(run_manifest["checks"], "run manifest checks")
    oxygen_cutoff_angstrom = positive_numeric_setting(
        settings, "oxygen_connectivity_cutoff_A"
    )
    covalent_oh_cutoff_angstrom = positive_numeric_setting(
        settings, "covalent_OH_cutoff_A"
    )
    h_acceptor_cutoff_angstrom = positive_numeric_setting(
        settings, "hbond_H_acceptor_cutoff_A"
    )
    oo_cutoff_angstrom = positive_numeric_setting(settings, "hbond_OO_cutoff_A")
    hbond_angle_cutoff_deg = numeric_setting(
        settings,
        "hbond_angle_cutoff_deg",
        minimum=0.0,
        maximum=180.0,
    )
    pair_temperature_relative_tolerance = numeric_setting(
        settings, "pair_temperature_relative_tolerance", minimum=0.0
    )
    energy_excursion_advisory = numeric_setting(
        settings, "energy_excursion_advisory_meV_atom", minimum=0.0
    )

    with np.load(trajectory_path, allow_pickle=False) as arrays:
        batch_ptr = arrays["batch_ptr"]
        atoms_per_graph = np.diff(batch_ptr)
        temperature = (
            2.0
            * arrays["kinetic_energies_eV"]
            / (3.0 * atoms_per_graph[None, :] * 8.617333262145e-5)
        )
        mean_temperature = temperature.mean(axis=0)
        recomputed_topology = {
            label: cluster_check_from_trajectory(
                arrays,
                graph,
                oxygen_cutoff_angstrom=oxygen_cutoff_angstrom,
                h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
                oo_cutoff_angstrom=oo_cutoff_angstrom,
                hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
            )
            for label, graph in (("(H2O)6", 2), ("(D2O)6", 3))
        }
        charge_error = np.max(np.abs(arrays["charge_sums_e"]), axis=0)
        energy_delta = arrays["total_energies_eV"] - arrays["total_energies_eV"][0]
        energy_excursion = (
            1000.0 * np.max(np.abs(energy_delta), axis=0) / atoms_per_graph
        )

    topology = pd.read_csv(output_dir / "water_ir_topology.csv", index_col=0)
    csv_ring_check = as_bool(topology["all_frames_initial_ring"])
    for label, recomputed in recomputed_topology.items():
        if not np.isclose(
            topology.loc[label, "max_OH_angstrom"],
            recomputed["max_OH_angstrom"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(f"topology CSV max O-H does not match {label}")
        if bool(csv_ring_check.loc[label]) != bool(
            recomputed["all_frames_initial_ring"]
        ):
            raise RuntimeError(f"topology CSV ring check does not match {label}")
        if int(topology.loc[label, "max_oxygen_components"]) != int(
            recomputed["max_oxygen_components"]
        ):
            raise RuntimeError(f"topology CSV oxygen components do not match {label}")

    intact = bool(
        all(item["all_frames_connected"] for item in recomputed_topology.values())
        and all(
            item["max_OH_angstrom"] < covalent_oh_cutoff_angstrom
            for item in recomputed_topology.values()
        )
    )
    if not intact:
        raise RuntimeError("saved trajectory failed the cluster-integrity check")
    ring_check = bool(
        all(item["all_frames_initial_ring"] for item in recomputed_topology.values())
    )
    energy_within_advisory = bool(np.max(energy_excursion) <= energy_excursion_advisory)

    comparisons = pd.read_csv(
        output_dir / "water_ir_comparisons.csv", index_col="comparison"
    )
    required_columns = {
        "value",
        "reported",
        "thermal_gate_passed",
        "topology_gate_passed",
        "status",
    }
    if not required_columns <= set(comparisons.columns):
        raise RuntimeError("comparison table is missing temperature/topology columns")
    expected_rows = {
        "H2O_over_D2O_centroid",
        "H6_over_D6_centroid",
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
    }
    if set(comparisons.index) != expected_rows:
        raise RuntimeError("comparison table does not contain the four declared rows")

    relative_temperature_difference = {
        "H2O_over_D2O_centroid": abs(mean_temperature[0] - mean_temperature[1])
        / (0.5 * (mean_temperature[0] + mean_temperature[1])),
        "H6_over_D6_centroid": abs(mean_temperature[2] - mean_temperature[3])
        / (0.5 * (mean_temperature[2] + mean_temperature[3])),
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": abs(
            mean_temperature[0] - mean_temperature[2]
        )
        / (0.5 * (mean_temperature[0] + mean_temperature[2])),
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": abs(
            mean_temperature[1] - mean_temperature[3]
        )
        / (0.5 * (mean_temperature[1] + mean_temperature[3])),
    }
    expected_thermal = {
        name: difference <= pair_temperature_relative_tolerance
        for name, difference in relative_temperature_difference.items()
    }
    expected_topology = {
        "H2O_over_D2O_centroid": True,
        "H6_over_D6_centroid": ring_check,
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": ring_check,
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": ring_check,
    }
    reported = as_bool(comparisons["reported"])
    thermal = as_bool(comparisons["thermal_gate_passed"])
    topology_valid = as_bool(comparisons["topology_gate_passed"])
    metrics = pd.read_csv(output_dir / "water_ir_metrics.csv", index_col="system")
    candidate_values = {
        "H2O_over_D2O_centroid": (
            metrics.loc["H2O", "OH_OD_region_centroid_cm-1"]
            / metrics.loc["D2O", "OH_OD_region_centroid_cm-1"]
        ),
        "H6_over_D6_centroid": (
            metrics.loc["(H2O)6", "OH_OD_region_centroid_cm-1"]
            / metrics.loc["(D2O)6", "OH_OD_region_centroid_cm-1"]
        ),
        "H_cluster_minus_monomer_OH_region_centroid_cm-1": (
            metrics.loc["(H2O)6", "OH_OD_region_centroid_cm-1"]
            - metrics.loc["H2O", "OH_OD_region_centroid_cm-1"]
        ),
        "D_cluster_minus_monomer_OD_region_centroid_cm-1": (
            metrics.loc["(D2O)6", "OH_OD_region_centroid_cm-1"]
            - metrics.loc["D2O", "OH_OD_region_centroid_cm-1"]
        ),
    }
    for name in expected_rows:
        expected_reported = bool(expected_thermal[name] and expected_topology[name])
        if bool(thermal.loc[name]) != bool(expected_thermal[name]):
            raise RuntimeError(
                f"temperature check was not independently reproduced: {name}"
            )
        if bool(topology_valid.loc[name]) != bool(expected_topology[name]):
            raise RuntimeError(
                f"topology check was not independently reproduced: {name}"
            )
        if bool(reported.loc[name]) != expected_reported:
            raise RuntimeError(f"reported state does not match the checks: {name}")
        value = comparisons.loc[name, "value"]
        if expected_reported and not np.isclose(value, candidate_values[name]):
            raise RuntimeError(
                f"reported comparison value does not match metrics: {name}"
            )
        if not expected_reported and not pd.isna(value):
            raise RuntimeError(f"withheld comparison retained a numeric value: {name}")

    labels = ("H2O", "D2O", "(H2O)6", "(D2O)6")
    diagnostics = pd.read_csv(
        output_dir / "water_ir_diagnostics.csv", index_col="system"
    )
    if set(diagnostics.index) != set(labels):
        raise RuntimeError("diagnostics table does not contain the four systems")
    diagnostic_expectations = {
        "NVE_start_T_3N_K": temperature[0],
        "NVE_mean_T_3N_K": mean_temperature,
        "max_charge_error_e": charge_error,
        "max_energy_excursion_meV_atom": energy_excursion,
    }
    for column, expected_values in diagnostic_expectations.items():
        if column not in diagnostics.columns:
            raise RuntimeError(f"diagnostics table is missing {column}")
        observed_values = diagnostics.loc[list(labels), column].to_numpy(dtype=float)
        if not np.allclose(
            observed_values,
            np.asarray(expected_values, dtype=float),
            rtol=1e-12,
            atol=1e-10,
        ):
            raise RuntimeError(f"diagnostics table does not reproduce {column}")

    independently_recomputed_checks = {
        "cluster_integrity_passed": intact,
        "initial_ring_persisted_all_frames": ring_check,
        "energy_excursion_within_advisory": energy_within_advisory,
    }
    for name, expected_value in independently_recomputed_checks.items():
        recorded_value = recorded_checks.get(name)
        if not isinstance(recorded_value, bool):
            raise ValueError(f"run manifest check {name!r} must be boolean")
        if recorded_value != expected_value:
            raise RuntimeError(f"run manifest check was not reproduced: {name}")

    recorded_reported = require_mapping(
        recorded_checks.get("reported_comparisons"),
        "run manifest checks.reported_comparisons",
    )
    if set(recorded_reported) != expected_rows:
        raise RuntimeError(
            "run manifest reported comparisons do not contain the four declared rows"
        )
    for name in expected_rows:
        recorded_value = recorded_reported[name]
        if not isinstance(recorded_value, bool):
            raise ValueError(
                f"run manifest reported comparison {name!r} must be boolean"
            )
        if recorded_value != bool(reported.loc[name]):
            raise RuntimeError(
                f"run manifest reported comparison was not reproduced: {name}"
            )

    return {
        "cluster_integrity_passed": intact,
        "cyclic_dft_overlay_check_passed": ring_check,
        "energy_excursion_within_advisory": energy_within_advisory,
        "comparison_reporting_valid": True,
        "configured_limits": {
            "oxygen_connectivity_cutoff_A": oxygen_cutoff_angstrom,
            "covalent_OH_cutoff_A": covalent_oh_cutoff_angstrom,
            "hbond_H_acceptor_cutoff_A": h_acceptor_cutoff_angstrom,
            "hbond_OO_cutoff_A": oo_cutoff_angstrom,
            "hbond_angle_cutoff_deg": hbond_angle_cutoff_deg,
            "pair_temperature_relative_tolerance": (
                pair_temperature_relative_tolerance
            ),
            "energy_excursion_advisory_meV_atom": energy_excursion_advisory,
        },
        "max_energy_excursion_meV_atom": {
            label: float(value)
            for label, value in zip(labels, energy_excursion, strict=True)
        },
        "temperature_3N_mean_K": {
            label: float(value)
            for label, value in zip(
                labels,
                mean_temperature,
                strict=True,
            )
        },
        "recomputed_cluster_topology": recomputed_topology,
        "comparisons": {
            index: {
                "reported": bool(reported.loc[index]),
                "thermal_gate_passed": bool(thermal.loc[index]),
                "topology_gate_passed": bool(topology_valid.loc[index]),
                "temperature_relative_difference": float(
                    relative_temperature_difference[index]
                ),
                "status": str(row["status"]),
            }
            for index, row in comparisons.iterrows()
        },
    }


def write_portable_checksums(
    *,
    base: Path,
    paths: list[Path],
    destination: Path,
) -> None:
    base = base.resolve()
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    lines = []
    for path in unique:
        relative = path.relative_to(base)
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executed-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--calculation-validation", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    executed = args.executed_notebook.resolve()
    output_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve()
    summary_path = args.summary.resolve()
    checksums_path = args.checksums.resolve()
    bundle_root = summary_path.parent
    missing_bundle_files = [
        name for name in BUNDLE_REQUIRED_FILES if not (bundle_root / name).is_file()
    ]
    if missing_bundle_files:
        raise FileNotFoundError(f"missing packaged run files: {missing_bundle_files}")
    packaged_runtime_check = validate_packaged_runtime_check(
        bundle_root / RUNTIME_CHECK_NAME,
        source_root=source_root,
    )
    d3_cache_report = validate_d3_cache_report(bundle_root / D3_CACHE_REPORT_NAME)
    missing_files = [
        name for name in REQUIRED_FILES if not (output_dir / name).is_file()
    ]
    missing_directories = [
        name for name in REQUIRED_DIRECTORIES if not (output_dir / name).is_dir()
    ]
    if missing_files or missing_directories:
        raise FileNotFoundError(
            "missing notebook outputs: "
            f"files={missing_files}, directories={missing_directories}"
        )
    for name in REQUIRED_DIRECTORIES:
        if not any(path.is_file() for path in (output_dir / name).rglob("*")):
            raise RuntimeError(f"required output directory is empty: {name}")

    run_manifest_path = output_dir / RUN_MANIFEST_NAME
    run_manifest = load_run_manifest(run_manifest_path)
    manifest_inventory = validate_manifest_inventory(output_dir, run_manifest)
    run_settings = require_mapping(run_manifest["settings"], "run manifest settings")
    warmup_steps = integer_setting(run_settings, "warmup_steps")
    production_steps = integer_setting(run_settings, "production_steps")
    dt_fs = positive_numeric_setting(run_settings, "dt_fs")
    if warmup_steps != 5_000 or production_steps != 20_000:
        raise RuntimeError(
            "Part 1 requires the complete 5,000-step warmup and "
            "20,000-step production workloads"
        )
    if dt_fs != 0.5:
        raise RuntimeError("Part 1 requires the declared 0.5 fs timestep")
    if run_settings.get("compile_mode") != (
        "default Torch compile on the fixed 42-atom IR batch"
    ):
        raise RuntimeError("Part 1 requires fixed-workload default compilation")
    fused_stage_route = validate_fused_stage_route(
        run_manifest,
        warmup_steps=warmup_steps,
        production_steps=production_steps,
    )

    source_hashes = dict(
        require_mapping(
            require_mapping(
                packaged_runtime_check.get("source"),
                "packaged runtime source",
            ).get("files_sha256"),
            "packaged runtime source file SHA-256 values",
        )
    )

    reference_artifacts = {}
    artifact_root = (
        source_root / "part-1-scalable-atomistic-workflows" / "reference" / "artifacts"
    )
    for label in ("h2o", "d2o", "h6", "d6"):
        reference_manifest_path = artifact_root / label / "manifest.json"
        arrays_path = artifact_root / label / "ir_arrays.npz"
        manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        reference_artifacts[label] = {
            "artifact_id": manifest["artifact_id"],
            "manifest_sha256": sha256_file(reference_manifest_path),
            "arrays_sha256": sha256_file(arrays_path),
        }

    source_notebook = (
        source_root / "part-1-scalable-atomistic-workflows" / "alchemi-water-ir.ipynb"
    )
    calculation_summary: Mapping[str, object] | None = None
    timing_source_sha256: str | None = None
    timing_executed_sha256: str | None = None
    if args.calculation_validation is not None:
        calculation_summary = require_mapping(
            json.loads(
                args.calculation_validation.resolve().read_text(encoding="utf-8")
            ),
            "calculation validation summary",
        )
        prior_source_hashes = require_mapping(
            calculation_summary.get("source_sha256"),
            "calculation validation source SHA-256 values",
        )
        timing_source_sha256 = prior_source_hashes.get(
            "part-1-scalable-atomistic-workflows/alchemi-water-ir.ipynb"
        )
        timing_executed_sha256 = calculation_summary.get("executed_notebook_sha256")
        if (
            not isinstance(timing_source_sha256, str)
            or not timing_source_sha256
            or not isinstance(timing_executed_sha256, str)
            or not timing_executed_sha256
        ):
            raise ValueError(
                "calculation validation does not identify the timed notebooks"
            )
    notebook_timing = validate_notebook_timing_report(
        bundle_root / TIMING_REPORT_NAME,
        source_notebook=source_notebook,
        executed_notebook=executed,
        expected_source_sha256=timing_source_sha256,
        expected_executed_sha256=timing_executed_sha256,
    )
    verified_run_details = validate_run_details(
        run_manifest, source_notebook, source_root
    )
    trajectory_path = output_dir / "water_ir_trajectory.npz"
    review_metadata = nbformat.read(executed, as_version=4).metadata.get(
        "alchemi_review"
    )
    validation_runtime = runtime_details()
    calculation_runtime = validation_runtime
    if calculation_summary is not None:
        calculation_runtime = require_mapping(
            calculation_summary.get("runtime"),
            "calculation validation runtime",
        )
    calculation_sevennet = require_mapping(
        calculation_runtime.get("sevennet_checkpoint"),
        "calculation runtime SevenNet-Omni checkpoint",
    )
    expected_sevennet_runtime = {
        "source": SEVENNET_CHECKPOINT_URL,
        "bytes": EXPECTED_SEVENNET_CHECKPOINT_BYTES,
        "sha256": EXPECTED_SEVENNET_CHECKPOINT_SHA256,
    }
    for name, expected in expected_sevennet_runtime.items():
        if calculation_sevennet.get(name) != expected:
            raise RuntimeError(
                f"calculation runtime SevenNet-Omni checkpoint {name!r} is not pinned"
            )
    harmonic_validation = validate_harmonic_outputs(
        output_dir,
        source_root,
        run_manifest,
    )
    required_output_files = list(REQUIRED_FILES)
    if harmonic_validation["comparison_reported"]:
        required_output_files.append(HARMONIC_COMPARISON_PLOT)
    required_outputs = required_output_files + list(REQUIRED_DIRECTORIES)

    summary = {
        "code_cells_executed": validate_notebook(executed, source_notebook),
        "error_outputs": 0,
        "trajectory_shapes": validate_trajectory(
            trajectory_path,
            expected_frames=production_steps,
            expected_dt_fs=dt_fs,
        ),
        "nci_outputs": validate_nci_outputs(
            output_dir / "nci_interaction_curves.csv",
            output_dir / "nci_interaction_metrics.csv",
            output_dir / "nci_ensemble_curves.csv",
            run_manifest,
            source_root=source_root,
        ),
        "sevennet_adapter": validate_sevennet_adsorption_outputs(
            output_dir / "surface_adsorption_energies.csv",
            output_dir / "surface_adsorption_forces.csv",
            output_dir / "sevennet_adapter_graph_mapping.csv",
            output_dir / "sevennet_adapter_numerical_agreement.csv",
            run_manifest,
            source_root=source_root,
        ),
        "harmonic_validation": harmonic_validation,
        "derived_ir_outputs": validate_derived_ir_outputs(
            output_dir,
            trajectory_path,
            source_root,
            run_manifest,
        ),
        "scientific_checks": validate_scientific_checks(
            output_dir, trajectory_path, run_manifest
        ),
        "run_manifest": {
            "schema": run_manifest["schema"],
            "sha256": sha256_file(run_manifest_path),
            "inventory": manifest_inventory,
            "verified_run_details": verified_run_details,
            "fused_stage_route_counts": fused_stage_route,
            "composition_checks": validate_composition_checks(run_manifest),
        },
        "reference_artifacts": reference_artifacts,
        "runtime": calculation_runtime,
        "packaged_runtime_check": packaged_runtime_check,
        "d3_cache_report": d3_cache_report,
        "notebook_timing": notebook_timing,
        "review_validation_runtime": (
            validation_runtime if args.calculation_validation is not None else None
        ),
        "notebook_review": review_metadata,
        "source_sha256": source_hashes,
        "executed_notebook_sha256": sha256_file(executed),
        "trajectory_sha256": sha256_file(trajectory_path),
        "required_outputs": required_outputs,
        "required_output_types": {
            "files": required_output_files,
            "directories": list(REQUIRED_DIRECTORIES),
        },
        "required_packaged_run_files": list(BUNDLE_REQUIRED_FILES),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    original_executed = executed.with_name(
        executed.stem + "-original" + executed.suffix
    )
    checksum_inputs = [
        executed,
        original_executed,
        summary_path,
        *(bundle_root / name for name in BUNDLE_REQUIRED_FILES),
        *output_dir.rglob("*"),
    ]
    if args.calculation_validation is not None:
        checksum_inputs.append(args.calculation_validation.resolve())
        calculation_checksums = args.calculation_validation.with_name(
            "SHA256SUMS-calculation"
        ).resolve()
        checksum_inputs.append(calculation_checksums)
    write_portable_checksums(
        base=summary_path.parent,
        paths=checksum_inputs,
        destination=checksums_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"portable checksums: {checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
