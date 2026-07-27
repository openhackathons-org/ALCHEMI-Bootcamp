#!/usr/bin/env python3
"""Run one fresh Part 1 domain-decomposition or electrostatics case.

Launch ``capacity`` with ``torchrun``. The capacity path uses only the public
Toolkit construction:

``DistributedManager -> DeviceMesh -> DomainConfig -> DomainParallel ->
partition -> run -> gather``.

The Toolkit 0.2 version used here returns energy and forces but not
the charge field from its AIMNet2-to-PME group. One-GPU references record the
predicted charge sum; the separate ``electrostatics-validation`` mode also
checks PME against Ewald on the same fixed charges.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import median, quantiles
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from aux.domain.config import DOMAIN_METHODOLOGY  # noqa: E402


DOMAIN_METHODOLOGY_CONFIG_PATH = (
    PART_DIR / "aux" / "domain" / "config.py"
).resolve()
RESULT_SCHEMA = "alchemi.part1-domain-case.v2"
RANK_SCHEMA = "alchemi.part1-domain-rank.v1"
CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
CHECKPOINT_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
D3_PARAMETER_SHA256 = "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
EXPECTED_D3 = {"a1": 0.566, "a2": 3.128, "s6": 1.0, "s8": 0.3908}
DEFAULT_PME_CUTOFF_A = DOMAIN_METHODOLOGY.pme_realspace_cutoff_a
DEFAULT_PME_MESH_SAFETY_FACTOR = DOMAIN_METHODOLOGY.pme_mesh_safety_factor
DEFAULT_PME_SPLINE_ORDER = DOMAIN_METHODOLOGY.pme_spline_order
DEFAULT_PME_ACCURACY = DOMAIN_METHODOLOGY.pme_accuracy
DEFAULT_EWALD_REFERENCE_ACCURACY = DOMAIN_METHODOLOGY.ewald_reference_accuracy
DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM = (
    DOMAIN_METHODOLOGY.pme_ewald_energy_tolerance_ev_per_atom
)
DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A = (
    DOMAIN_METHODOLOGY.pme_ewald_force_max_tolerance_ev_a
)
DEFAULT_CHARGE_SUM_TOL_E = DOMAIN_METHODOLOGY.charge_sum_tolerance_e
DEFAULT_D3_CUTOFF_A = DOMAIN_METHODOLOGY.d3_cutoff_a
DEFAULT_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction
DEFAULT_DOMAIN_SKIN_A = DOMAIN_METHODOLOGY.domain_halo_skin_a
EXPECTED_AIMNET_NEIGHBOR_CUTOFF_A = DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a
ATOMS_PER_COMPOSITION_UNIT = DOMAIN_METHODOLOGY.atoms_per_composition_unit
DEFAULT_STEADY_TIMING_WARMUP_COUNT = (
    DOMAIN_METHODOLOGY.steady_timing_warmup_count
)
DEFAULT_STEADY_TIMING_SAMPLE_COUNT = DOMAIN_METHODOLOGY.steady_timing_sample_count
EXPECTED_PYTHON_MAJOR_MINOR = (3, 12)
EXPECTED_TORCH_VERSION = "2.12.0+cu130"
EXPECTED_TORCH_CUDA_VERSION = "13.0"
EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "aimnet": "0.2.0",
    "nvidia-physicsnemo": "2.1.1",
    "nvalchemi-toolkit": "0.2.0",
    "nvalchemi-toolkit-ops": "0.4.0",
}

COLD_MEASUREMENT_ROLES = frozenset({"capacity", "parity", "rescue"})
MEASUREMENT_ROLE_BY_MODE = {
    "capacity": "capacity",
    "parity": "parity",
    "distributed": "rescue",
    "steady-timing": "steady_timing",
    "electrostatics-validation": "electrostatics_validation",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_timing_samples(samples_s: list[float]) -> dict[str, float]:
    """Return median and inclusive quartiles for max-rank workflow samples."""

    if not samples_s or any(
        not math.isfinite(value) or value <= 0.0 for value in samples_s
    ):
        raise ValueError("timing samples must be nonempty, positive, and finite")
    sample_median = float(median(samples_s))
    if len(samples_s) == 1:
        q1 = q3 = sample_median
    else:
        q1, _, q3 = (
            float(value)
            for value in quantiles(samples_s, n=4, method="inclusive")
        )
    return {
        "median_s": sample_median,
        "q1_s": q1,
        "q3_s": q3,
        "iqr_s": q3 - q1,
    }


def validate_spatial_layout(
    cells_per_dim: tuple[int, ...],
    rank_grid: tuple[int, ...],
    *,
    world_size: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Check the layout selected by SpatialPartitioner for the current cell."""

    if (
        len(cells_per_dim) != 3
        or len(rank_grid) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (*cells_per_dim, *rank_grid)
        )
    ):
        raise RuntimeError(
            "Toolkit cells_per_dim and rank_grid must each contain three "
            "positive integers"
        )
    if math.prod(rank_grid) != world_size:
        raise RuntimeError(
            f"Toolkit rank_grid {rank_grid} does not match {world_size} ranks"
        )
    if any(
        ranks > cells
        for ranks, cells in zip(rank_grid, cells_per_dim, strict=True)
    ):
        raise RuntimeError(
            f"Toolkit rank_grid {rank_grid} exceeds cells_per_dim {cells_per_dim}"
        )
    if any(
        cells % ranks != 0
        for ranks, cells in zip(rank_grid, cells_per_dim, strict=True)
    ):
        raise RuntimeError(
            f"Toolkit rank_grid {rank_grid} does not divide cells_per_dim "
            f"{cells_per_dim}"
        )
    return (
        (cells_per_dim[0], cells_per_dim[1], cells_per_dim[2]),
        (rank_grid[0], rank_grid[1], rank_grid[2]),
    )


def validate_measurement_args(args: argparse.Namespace) -> None:
    """Resolve timing counts and reject a mode/role mismatch before imports."""

    expected_role = MEASUREMENT_ROLE_BY_MODE[args.mode]
    if args.measurement_role != expected_role:
        raise ValueError(
            f"mode {args.mode!r} requires measurement role {expected_role!r}"
        )
    if args.warmup_count is None:
        args.warmup_count = (
            DEFAULT_STEADY_TIMING_WARMUP_COUNT
            if expected_role == "steady_timing"
            else 0
        )
    if args.sample_count is None:
        args.sample_count = (
            DEFAULT_STEADY_TIMING_SAMPLE_COUNT
            if expected_role == "steady_timing"
            else 1
        )
    if expected_role == "steady_timing":
        if args.warmup_count < 1:
            raise ValueError("steady_timing requires at least one warmup")
        if args.sample_count < 5:
            raise ValueError("steady_timing requires at least five samples")
    elif expected_role in COLD_MEASUREMENT_ROLES or expected_role == (
        "electrostatics_validation"
    ):
        if (args.warmup_count, args.sample_count) != (0, 1):
            raise ValueError(
                f"{expected_role} requires one cold sample and no warmup"
            )


def file_identity(path: Path) -> dict[str, Any]:
    """Record enough information to verify and relocate one referenced file."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"referenced file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def resolved_methodology_record(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Record the config identity and exact values applied by this process."""

    values = DOMAIN_METHODOLOGY.resolved_values(json_compatible=True)
    values.update(
        {
            "pme_realspace_cutoff_a": args.pme_cutoff_a,
            "pme_mesh_safety_factor": args.pme_mesh_safety_factor,
            "pme_spline_order": args.pme_spline_order,
            "pme_accuracy": args.pme_accuracy,
            "ewald_reference_accuracy": args.ewald_reference_accuracy,
            "pme_ewald_energy_tolerance_ev_per_atom": (
                args.pme_ewald_energy_tol_ev_per_atom
            ),
            "pme_ewald_force_max_tolerance_ev_a": (
                args.pme_ewald_force_max_tol_ev_a
            ),
            "charge_sum_tolerance_e": args.charge_sum_tol_e,
            "d3_cutoff_a": args.d3_cutoff_a,
            "d3_smoothing_fraction": args.d3_smoothing_fraction,
            "domain_halo_skin_a": args.domain_skin_a,
            "steady_timing_warmup_count": (
                args.warmup_count
                if args.measurement_role == "steady_timing"
                else DOMAIN_METHODOLOGY.steady_timing_warmup_count
            ),
            "steady_timing_sample_count": (
                args.sample_count
                if args.measurement_role == "steady_timing"
                else DOMAIN_METHODOLOGY.steady_timing_sample_count
            ),
        }
    )
    return {
        "source": DOMAIN_METHODOLOGY.as_record(),
        "source_file": file_identity(DOMAIN_METHODOLOGY_CONFIG_PATH),
        "resolved_values": values,
        "case_molecules_per_species": args.pair_count,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_npy(path: Path, array: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(path)


def direct_url_commit(distribution_name: str) -> str | None:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if text is None:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verify_clean_git_checkout(
    root_text: str | None,
    *,
    expected_commit: str,
    label: str,
) -> dict[str, str]:
    if not root_text:
        raise RuntimeError(f"{label} source root is not set")
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise RuntimeError(f"{label} source root does not exist: {root}")
    top_level = git_value(root, "rev-parse", "--show-toplevel")
    if top_level is None or Path(top_level).resolve() != root:
        raise RuntimeError(f"{label} source root is not the Git checkout root: {root}")
    commit = git_value(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            f"{label} source is at {commit!r}; expected commit {expected_commit}"
        )
    status = git_value(root, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        raise RuntimeError(f"could not read {label} source status: {root}")
    if status:
        raise RuntimeError(f"{label} source checkout is not clean: {root}")
    ignored = git_value(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
    )
    if ignored is None:
        raise RuntimeError(f"could not inspect ignored files in {label} source: {root}")
    if ignored:
        raise RuntimeError(f"{label} source checkout contains ignored files: {root}")
    return {"root": str(root), "commit": commit}


def _verify_clean_repository(
    root: Path,
    *,
    required_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Require reportable tutorial code to come from one clean Git revision."""

    resolved_root = root.resolve()
    commit = git_value(resolved_root, "rev-parse", "HEAD")
    tree = git_value(resolved_root, "rev-parse", "HEAD^{tree}")
    if commit is None or tree is None:
        raise RuntimeError(f"tutorial source is not a Git checkout: {resolved_root}")
    status = git_value(
        resolved_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status is None:
        raise RuntimeError(f"could not read tutorial source status: {resolved_root}")
    if status:
        raise RuntimeError(f"tutorial source checkout is not clean: {resolved_root}")

    tracked_paths: list[str] = []
    for path in required_paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"required tutorial file is outside the checkout: {resolved}"
            ) from exc
        tracked = git_value(
            resolved_root,
            "ls-files",
            "--error-unmatch",
            relative,
        )
        if tracked != relative:
            raise RuntimeError(
                f"required tutorial file is not tracked by Git: {relative}"
            )
        tracked_paths.append(relative)

    branch = git_value(resolved_root, "symbolic-ref", "--short", "-q", "HEAD")
    return {
        "root": str(resolved_root),
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "clean": True,
        "tracked_required_paths": tracked_paths,
    }


def _verify_optional_direct_url_commit(
    observed_commit: str | None,
    *,
    expected_commit: str,
    label: str,
) -> None:
    if observed_commit is not None and observed_commit != expected_commit:
        raise RuntimeError(
            f"{label} installed-package metadata reports commit "
            f"{observed_commit!r}; expected {expected_commit}"
        )


def _module_source_in_checkout(module: Any, root: Path, *, label: str) -> Path:
    source_text = getattr(module, "__file__", None)
    if not source_text:
        raise RuntimeError(f"{label} module has no source file")
    source = Path(source_text).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} imported from {source}, outside expected source root {root}"
        ) from exc
    return source


def validate_runtime_versions(
    *,
    python_major_minor: tuple[int, int],
    torch_version: str,
    torch_cuda_version: str | None,
    distribution_versions: dict[str, str],
) -> None:
    """Check the software versions declared for the recorded H100 run."""

    if python_major_minor != EXPECTED_PYTHON_MAJOR_MINOR:
        expected = ".".join(str(value) for value in EXPECTED_PYTHON_MAJOR_MINOR)
        observed = ".".join(str(value) for value in python_major_minor)
        raise RuntimeError(f"Python {observed} does not match required {expected}.x")
    if torch_version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"Torch {torch_version!r} does not match required "
            f"{EXPECTED_TORCH_VERSION!r}"
        )
    if torch_cuda_version != EXPECTED_TORCH_CUDA_VERSION:
        raise RuntimeError(
            f"Torch CUDA {torch_cuda_version!r} does not match required "
            f"{EXPECTED_TORCH_CUDA_VERSION!r}"
        )
    for name, expected in EXPECTED_RUNTIME_DISTRIBUTIONS.items():
        observed = distribution_versions.get(name)
        if observed != expected:
            raise RuntimeError(
                f"{name} version {observed!r} does not match required {expected!r}"
            )


def runtime_software_identity() -> dict[str, Any]:
    """Return and validate the software identity shared by every rank."""

    import torch

    distribution_versions = {
        name: metadata.version(name) for name in EXPECTED_RUNTIME_DISTRIBUTIONS
    }
    validate_runtime_versions(
        python_major_minor=(sys.version_info.major, sys.version_info.minor),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        distribution_versions=distribution_versions,
    )
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "distribution_versions": distribution_versions,
    }


def validate_runtime_rows(
    rows: list[dict[str, Any]],
    *,
    expected_software: dict[str, Any] | None = None,
) -> None:
    """Reject a case assembled from ranks with different software or GPUs."""

    if not rows:
        raise RuntimeError("no per-rank runtime records were collected")
    comparable_keys = (
        "gpu_name",
        "gpu_total_memory_bytes",
        "compute_capability",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "nccl_version",
        "driver_version",
        "python_version",
        "python_executable",
        "python_prefix",
        "distribution_versions",
    )
    reference = {key: rows[0].get(key) for key in comparable_keys}
    for row in rows[1:]:
        observed = {key: row.get(key) for key in comparable_keys}
        if observed != reference:
            raise RuntimeError(
                "software or GPU identity differs between distributed ranks"
            )
    if expected_software is not None:
        observed_software = {key: rows[0].get(key) for key in expected_software}
        if observed_software != expected_software:
            raise RuntimeError(
                "per-rank software identity differs from the checked source record"
            )


def tensor_checksum(tensor: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_bundle_checksum(fields: dict[str, Any]) -> str:
    digest = sha256()
    for name, tensor in sorted(fields.items()):
        digest.update(name.encode())
        digest.update(tensor_checksum(tensor).encode())
    return digest.hexdigest()


def source_atom_ids_tensor(atoms: Any, device: Any) -> Any:
    """Copy stable input atom IDs without adding them to a Toolkit Batch."""

    import torch

    return torch.as_tensor(
        atoms.arrays["source_atom_id"],
        dtype=torch.int64,
        device=device,
    )


def source_input_checksum(batch: Any, source_atom_ids: Any) -> str:
    """Hash the scientific Batch fields together with external stable IDs."""

    return tensor_bundle_checksum(
        {
            "atomic_numbers": batch.atomic_numbers,
            "positions": batch.positions,
            "cell": batch.cell,
            "pbc": batch.pbc,
            "source_atom_id": source_atom_ids,
        }
    )


def predict_gathered_source_ids(
    *,
    source_atom_ids: Any,
    positions: Any,
    partitioner: Any,
    world_size: int,
) -> Any:
    """Reproduce Toolkit's stable, rank-contiguous scatter and gather order."""

    import torch

    source_ids = source_atom_ids.to(torch.int64).reshape(-1)
    if source_ids.numel() != positions.shape[0]:
        raise ValueError("source_atom_ids and positions must have the same length")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return source_ids.clone()

    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    rank_assignment = rank_assignment.to(torch.int64).reshape(-1)
    if rank_assignment.shape != source_ids.shape:
        raise RuntimeError("spatial rank assignment has the wrong shape")
    if bool((rank_assignment < 0).any()) or bool(
        (rank_assignment >= world_size).any()
    ):
        raise RuntimeError("spatial rank assignment is outside the rank mesh")
    scatter_order = torch.argsort(rank_assignment, stable=True)
    return source_ids[scatter_order]


def source_order_from_gathered_ids(
    gathered_source_ids: Any,
    *,
    expected_atom_count: int,
) -> Any:
    """Return the row order that restores the stable 0..N-1 atom identity."""

    import torch

    source_ids = gathered_source_ids.to(torch.int64).reshape(-1)
    if source_ids.numel() != expected_atom_count:
        raise RuntimeError("predicted gathered source_atom_id has the wrong shape")
    order = torch.argsort(source_ids, stable=True)
    expected_ids = torch.arange(
        expected_atom_count,
        dtype=torch.int64,
        device=source_ids.device,
    )
    if not torch.equal(source_ids[order], expected_ids):
        raise RuntimeError(
            "predicted gathered source_atom_id is not an exact 0..N-1 permutation"
        )
    return order


def resolve_checkpoint(alias_or_path: str) -> Path:
    from aimnet.calculators.model_registry import get_model_path

    path = Path(get_model_path(alias_or_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AIMNet2 checkpoint is missing: {path}")
    observed = sha256_file(path)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"AIMNet2 checkpoint SHA-256 {observed} != {CHECKPOINT_SHA256}"
        )
    return path


def validate_model_metadata(raw: dict[str, Any]) -> dict[str, float]:
    if raw.get("needs_coulomb") is not True:
        raise RuntimeError("AIMNet2 checkpoint does not request external Coulomb")
    if raw.get("needs_dispersion") is not True:
        raise RuntimeError("AIMNet2 checkpoint does not request external D3")
    if raw.get("coulomb_mode") != "sr_embedded":
        raise RuntimeError("unexpected AIMNet2 Coulomb convention")
    if abs(float(raw.get("coulomb_sr_rc", -1.0)) - 4.6) > 1.0e-5:
        raise RuntimeError("unexpected AIMNet2 short-range Coulomb cutoff")
    values = {name: float(raw["d3_params"][name]) for name in EXPECTED_D3}
    for name, expected in EXPECTED_D3.items():
        if abs(values[name] - expected) > 1.0e-7:
            raise RuntimeError(
                f"unexpected checkpoint D3 parameter {name}={values[name]}"
            )
    return values


def verify_runtime_source(repository_root: Path) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    repository = _verify_clean_repository(
        repository_root,
        required_paths=(runner_path, DOMAIN_METHODOLOGY_CONFIG_PATH),
    )
    core_checkout = _verify_clean_git_checkout(
        os.environ.get("ALCHEMI_TOOLKIT_CORE_ROOT"),
        expected_commit=CORE_COMMIT,
        label="Toolkit Core",
    )
    ops_checkout = _verify_clean_git_checkout(
        os.environ.get("ALCHEMI_TOOLKIT_OPS_ROOT"),
        expected_commit=OPS_COMMIT,
        label="Toolkit-Ops",
    )
    import nvalchemi
    import nvalchemiops

    core_direct = direct_url_commit("nvalchemi-toolkit")
    ops_direct = direct_url_commit("nvalchemi-toolkit-ops")
    core_root = Path(core_checkout["root"])
    ops_root = Path(ops_checkout["root"])
    core_git = core_checkout["commit"]
    ops_git = ops_checkout["commit"]
    _verify_optional_direct_url_commit(
        core_direct,
        expected_commit=CORE_COMMIT,
        label="Toolkit Core",
    )
    _verify_optional_direct_url_commit(
        ops_direct,
        expected_commit=OPS_COMMIT,
        label="Toolkit-Ops",
    )
    version = getattr(nvalchemi, "version", getattr(nvalchemi, "__version__", None))
    if version != "0.2.0":
        raise RuntimeError(f"Toolkit version {version!r} != '0.2.0'")
    ops_version = getattr(nvalchemiops, "__version__", None)
    if ops_version != "0.4.0":
        raise RuntimeError(f"Toolkit-Ops version {ops_version!r} != '0.4.0'")
    core_source = _module_source_in_checkout(
        nvalchemi,
        core_root,
        label="Toolkit Core",
    )
    ops_source = _module_source_in_checkout(
        nvalchemiops,
        ops_root,
        label="Toolkit-Ops",
    )
    software = runtime_software_identity()
    return {
        "toolkit_core_commit": CORE_COMMIT,
        "toolkit_core_direct_url_commit": core_direct,
        "toolkit_core_git_commit": core_git,
        "toolkit_core_source_root": str(core_root),
        "toolkit_core_source_file": str(core_source),
        "toolkit_core_source_file_sha256": sha256_file(core_source),
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_ops_direct_url_commit": ops_direct,
        "toolkit_ops_git_commit": ops_git,
        "toolkit_ops_source_root": str(ops_root),
        "toolkit_ops_source_file": str(ops_source),
        "toolkit_ops_source_file_sha256": sha256_file(ops_source),
        "toolkit_version": version,
        "toolkit_ops_version": ops_version,
        "repository_root": repository["root"],
        "repository_commit": repository["commit"],
        "repository_tree": repository["tree"],
        "repository_branch": repository["branch"],
        "repository_dirty": False,
        "repository_required_paths": repository["tracked_required_paths"],
        "runtime_software": software,
        "domain_methodology_name": DOMAIN_METHODOLOGY.name,
        "domain_methodology_version": DOMAIN_METHODOLOGY.version,
        "domain_methodology_config_file": str(DOMAIN_METHODOLOGY_CONFIG_PATH),
        "domain_methodology_config_sha256": sha256_file(
            DOMAIN_METHODOLOGY_CONFIG_PATH
        ),
        "domain_methodology_record": DOMAIN_METHODOLOGY.as_record(),
    }


def load_atoms_and_manifest(
    input_path: Path,
    manifest_path: Path | None,
) -> tuple[Any, dict[str, Any] | None]:
    from ase.io import read as ase_read

    atoms = ase_read(input_path)
    if not bool(atoms.pbc.all()):
        raise ValueError("domain-decomposition input must be periodic in x, y, z")
    cell = atoms.cell.array
    if cell.shape != (3, 3) or abs(float(atoms.get_volume())) <= 0.0:
        raise ValueError("input must have a nonzero 3D cell")
    if int(atoms.info.get("charge", 0)) != 0:
        raise ValueError("the phenol and N-methylacetamide box must be neutral")
    if "source_atom_id" not in atoms.arrays:
        raise ValueError("input is missing stable source_atom_id values")
    source_ids = atoms.arrays["source_atom_id"]
    if sorted(int(value) for value in source_ids) != list(range(len(atoms))):
        raise ValueError("source_atom_id must contain each integer from 0 to N-1")

    manifest = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["structure"]["sha256"]
        if sha256_file(input_path) != expected:
            raise ValueError("input structure checksum does not match its manifest")
        if int(manifest["atom_count"]) != len(atoms):
            raise ValueError("input atom count does not match its manifest")
        manifest_source = manifest.get("source", {})
        if (
            manifest_source.get("domain_methodology_config_sha256")
            != sha256_file(DOMAIN_METHODOLOGY_CONFIG_PATH)
            or manifest_source.get("domain_methodology_name")
            != DOMAIN_METHODOLOGY.name
            or manifest_source.get("domain_methodology_version")
            != DOMAIN_METHODOLOGY.version
        ):
            raise ValueError(
                "input manifest was built with a different domain methodology"
            )
    return atoms, manifest


def build_input_record(
    *,
    input_path: Path,
    tensor_sha256: str | None,
    manifest_path: Path | None,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Record the structure and optional Packmol manifest as verifiable files."""

    structure = file_identity(input_path)
    manifest_file = file_identity(manifest_path) if manifest_path is not None else None
    return {
        # Keep the original fields for the campaign reader.
        "path": structure["path"],
        "file_sha256": structure["sha256"],
        "file_size_bytes": structure["size_bytes"],
        "tensor_sha256": tensor_sha256,
        "manifest": manifest,
        "manifest_file": manifest_file,
    }


def make_batch(atoms: Any, device: Any) -> Any:
    import torch
    from nvalchemi.data import AtomicData, Batch

    data = AtomicData.from_atoms(atoms, device=device, dtype=torch.float32)
    data.forces = torch.zeros(len(atoms), 3, device=device, dtype=torch.float32)
    data.energy = torch.zeros(1, 1, device=device, dtype=torch.float32)
    data.charge = torch.zeros(1, 1, device=device, dtype=torch.float32)
    return Batch.from_data_list([data], device=device)


def estimate_pme_setup(
    *,
    positions: Any,
    cell: Any,
    batch_idx: Any,
    real_space_cutoff_a: float,
    accuracy: float,
    mesh_safety_factor: float,
) -> dict[str, Any]:
    """Couple PME alpha and mesh to the real-space neighbor cutoff."""

    from nvalchemiops.torch.interactions.electrostatics import (
        estimate_pme_parameters,
    )

    parameters = estimate_pme_parameters(
        positions,
        cell,
        batch_idx=batch_idx,
        accuracy=accuracy,
        real_space_cutoff=real_space_cutoff_a,
        mesh_safety_factor=mesh_safety_factor,
    )
    if parameters.alpha.numel() != 1 or parameters.real_space_cutoff.numel() != 1:
        raise ValueError("the domain lesson expects one periodic system")
    resolved_cutoff = float(parameters.real_space_cutoff.item())
    if not math.isclose(
        resolved_cutoff,
        real_space_cutoff_a,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError("PME parameter estimation changed the requested cutoff")
    mesh_dimensions = tuple(int(value) for value in parameters.mesh_dimensions)
    if len(mesh_dimensions) != 3 or any(value <= 0 for value in mesh_dimensions):
        raise RuntimeError("PME parameter estimation returned an invalid mesh")
    return {
        "real_space_cutoff_a": resolved_cutoff,
        "alpha_a_inverse": float(parameters.alpha.item()),
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing_a": [
            float(value) for value in parameters.mesh_spacing[0].detach().cpu()
        ],
        "accuracy": float(accuracy),
        "mesh_safety_factor": float(mesh_safety_factor),
        "parameter_rule": (
            "estimate_pme_parameters(accuracy, real_space_cutoff, "
            "mesh_safety_factor)"
        ),
    }


def estimate_ewald_reference_setup(
    *,
    positions: Any,
    cell: Any,
    batch_idx: Any,
    accuracy: float,
) -> dict[str, Any]:
    """Resolve the direct Ewald reference cutoff from the official estimator."""

    from nvalchemiops.torch.interactions.electrostatics import (
        estimate_ewald_parameters,
    )

    parameters = estimate_ewald_parameters(
        positions,
        cell,
        batch_idx=batch_idx,
        accuracy=accuracy,
    )
    if (
        parameters.alpha.numel() != 1
        or parameters.real_space_cutoff.numel() != 1
        or parameters.reciprocal_space_cutoff.numel() != 1
    ):
        raise ValueError("the Ewald reference expects one periodic system")
    return {
        "real_space_cutoff_a": float(parameters.real_space_cutoff.item()),
        "reciprocal_space_cutoff_a_inverse": float(
            parameters.reciprocal_space_cutoff.item()
        ),
        "alpha_a_inverse": float(parameters.alpha.item()),
        "accuracy": float(accuracy),
        "parameter_rule": "estimate_ewald_parameters(accuracy)",
    }


def build_aimnet(checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    from nvalchemi.models import AIMNet2Wrapper

    model = AIMNet2Wrapper.from_checkpoint(
        checkpoint,
        device=device,
        compile_model=False,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw_metadata = dict(model.model.metadata)
    d3_parameters = validate_model_metadata(raw_metadata)
    model.set_config("active_outputs", {"energy", "charges"})
    return model, {
        "raw": raw_metadata,
        "d3_parameters": d3_parameters,
        "cutoff_a": float(model.model_config.neighbor_config.cutoff),
    }


def build_complete_pipeline(
    *,
    checkpoint: Path,
    d3_parameter_file: Path,
    device: Any,
    args: argparse.Namespace,
    pme_setup: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from nvalchemi.models import (
        DFTD3ModelWrapper,
        PMEModelWrapper,
        PipelineGroup,
        PipelineModelWrapper,
        PipelineStep,
    )

    aimnet, aimnet_info = build_aimnet(checkpoint, device)
    d3p = aimnet_info["d3_parameters"]
    pme = (
        PMEModelWrapper(
            cutoff=args.pme_cutoff_a,
            mesh_dimensions=pme_setup["mesh_dimensions"],
            spline_order=args.pme_spline_order,
            alpha=pme_setup["alpha_a_inverse"],
            accuracy=args.pme_accuracy,
            hybrid_forces=True,
        )
        .to(device)
        .eval()
    )
    d3 = (
        DFTD3ModelWrapper(
            a1=d3p["a1"],
            a2=d3p["a2"],
            s6=d3p["s6"],
            s8=d3p["s8"],
            cutoff=args.d3_cutoff_a,
            smoothing_fraction=args.d3_smoothing_fraction,
            param_file=d3_parameter_file,
            auto_download=False,
        )
        .to(device)
        .eval()
    )
    d3.set_config("active_outputs", {"energy", "forces"})
    pipeline = (
        PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[PipelineStep(model=aimnet), PipelineStep(model=pme)],
                    use_autograd=True,
                ),
                PipelineGroup(
                    steps=[PipelineStep(model=d3)],
                    use_autograd=False,
                ),
            ],
            neighbor_adaptation="never",
        )
        .to(device)
        .eval()
    )
    pipeline.set_config("active_outputs", {"energy", "forces", "charges"})
    return pipeline, {
        "aimnet": aimnet_info,
        "pme": {
            "cutoff_a": args.pme_cutoff_a,
            "alpha_a_inverse": pme_setup["alpha_a_inverse"],
            "mesh_dimensions": list(pme_setup["mesh_dimensions"]),
            "mesh_spacing_a": pme_setup["mesh_spacing_a"],
            "mesh_safety_factor": args.pme_mesh_safety_factor,
            "parameter_rule": pme_setup["parameter_rule"],
            "spline_order": args.pme_spline_order,
            "accuracy": args.pme_accuracy,
            "hybrid_forces": True,
        },
        "d3": {
            **d3p,
            "cutoff_a": args.d3_cutoff_a,
            "smoothing_fraction": args.d3_smoothing_fraction,
            "parameter_file": str(d3_parameter_file),
            "parameter_file_sha256": sha256_file(d3_parameter_file),
            "parameter_file_identity": file_identity(d3_parameter_file),
        },
        "groups": [
            {
                "steps": ["AIMNet2Wrapper", "PMEModelWrapper"],
                "use_autograd": True,
            },
            {"steps": ["DFTD3ModelWrapper"], "use_autograd": False},
        ],
        "neighbor_adaptation": "never",
    }


def runtime_row(rank: int, local_rank: int, device: Any) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    gpu_uuid = getattr(properties, "uuid", None)
    software = runtime_software_identity()
    try:
        raw_nccl_version = torch.cuda.nccl.version()
        nccl_version = (
            list(raw_nccl_version)
            if isinstance(raw_nccl_version, tuple)
            else raw_nccl_version
        )
    except (AttributeError, RuntimeError):
        nccl_version = None
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                f"--id={local_rank}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver_version = None
    return {
        "rank": rank,
        "local_rank": local_rank,
        "host": platform.node(),
        "gpu_name": properties.name,
        # Torch 2.11 exposes a private _CUuuid object here. Convert it before
        # writing JSON so an early run failure still leaves a readable record.
        "gpu_uuid": None if gpu_uuid is None else str(gpu_uuid),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        **software,
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "driver_version": driver_version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "slurm_process_id": os.environ.get("SLURM_PROCID"),
    }


def force_summary(forces: Any) -> dict[str, Any]:
    import torch

    values = forces.detach().to(torch.float64)
    magnitudes = torch.linalg.vector_norm(values, dim=1)
    return {
        "shape": list(values.shape),
        "dtype": str(forces.dtype),
        "sha256": tensor_checksum(forces),
        "sum_vector_ev_a": [float(v) for v in values.sum(dim=0).cpu()],
        "sum_abs_ev_a": float(values.abs().sum().item()),
        "sum_squares_ev2_a2": float((values * values).sum().item()),
        "rms_ev_a": float(torch.sqrt((values * values).mean()).item()),
        "max_norm_ev_a": float(magnitudes.max().item()),
        "finite": bool(torch.isfinite(values).all()),
    }


def run_electrostatics_validation(
    args: argparse.Namespace,
    *,
    device: Any,
    checkpoint: Path,
    atoms: Any,
    source: dict[str, Any],
    input_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch
    from nvalchemi.models import EwaldModelWrapper, PMEModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("electrostatics-validation requires exactly one rank")
    batch = make_batch(atoms, device)
    source_atom_ids = source_atom_ids_tensor(atoms, device)
    input_hash = source_input_checksum(
        batch,
        source_atom_ids,
    )
    aimnet, aimnet_info = build_aimnet(checkpoint, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = perf_counter()
    compute_neighbors(batch, config=aimnet.model_config.neighbor_config)
    with torch.no_grad():
        aimnet_output = aimnet(batch)
    charges = aimnet_output["charges"].detach()
    batch.charges = charges

    pme_setup = estimate_pme_setup(
        positions=batch.positions,
        cell=batch.cell,
        batch_idx=batch.batch_idx,
        real_space_cutoff_a=args.pme_cutoff_a,
        accuracy=args.pme_accuracy,
        mesh_safety_factor=args.pme_mesh_safety_factor,
    )
    ewald_setup = estimate_ewald_reference_setup(
        positions=batch.positions,
        cell=batch.cell,
        batch_idx=batch.batch_idx,
        accuracy=args.ewald_reference_accuracy,
    )
    minimum_cell_length_a = float(
        torch.linalg.vector_norm(batch.cell[0], dim=1).min().item()
    )
    if 2.0 * ewald_setup["real_space_cutoff_a"] >= minimum_cell_length_a:
        raise RuntimeError(
            "the electrostatics validation box is too small for the estimated "
            "Ewald real-space cutoff"
        )
    pme = (
        PMEModelWrapper(
            cutoff=args.pme_cutoff_a,
            mesh_dimensions=pme_setup["mesh_dimensions"],
            spline_order=args.pme_spline_order,
            alpha=pme_setup["alpha_a_inverse"],
            accuracy=args.pme_accuracy,
            hybrid_forces=True,
        )
        .to(device)
        .eval()
    )
    ewald = (
        EwaldModelWrapper(
            cutoff=ewald_setup["real_space_cutoff_a"],
            accuracy=args.ewald_reference_accuracy,
            hybrid_forces=True,
        )
        .to(device)
        .eval()
    )
    compute_neighbors(batch, config=pme.model_config.neighbor_config)
    with torch.no_grad():
        pme_output = pme(batch)
    compute_neighbors(batch, config=ewald.model_config.neighbor_config)
    with torch.no_grad():
        ewald_output = ewald(batch)
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    pme_forces = pme_output["forces"].detach()
    ewald_forces = ewald_output["forces"].detach()
    force_difference = pme_forces - ewald_forces
    pme_energy = float(pme_output["energy"].sum().item())
    ewald_energy = float(ewald_output["energy"].sum().item())
    energy_difference = pme_energy - ewald_energy
    energy_difference_per_atom = abs(energy_difference) / len(atoms)
    force_difference_values = force_difference.detach().to(torch.float64)
    force_difference_rms = float(
        torch.sqrt((force_difference_values * force_difference_values).mean()).item()
    )
    force_difference_max_norm = float(
        torch.linalg.vector_norm(force_difference_values, dim=1).max().item()
    )
    charge_sum = float(charges.to(torch.float64).sum().item())
    acceptance = {
        "declared_before_measurement": True,
        "absolute_energy_difference_ev_per_atom_max": (
            args.pme_ewald_energy_tol_ev_per_atom
        ),
        "force_difference_max_norm_ev_a_max": args.pme_ewald_force_max_tol_ev_a,
        "absolute_charge_sum_e_max": args.charge_sum_tol_e,
    }
    passed = (
        energy_difference_per_atom
        <= acceptance["absolute_energy_difference_ev_per_atom_max"]
        and force_difference_max_norm
        <= acceptance["force_difference_max_norm_ev_a_max"]
        and abs(charge_sum) <= acceptance["absolute_charge_sum_e_max"]
    )
    row = {
        "schema": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "run_id": args.run_id,
        "case_id": args.case_id,
        "mode": "electrostatics-validation",
        "measurement_role": args.measurement_role,
        "status": "complete",
        "success": True,
        "world_size": 1,
        "pair_count": args.pair_count,
        "molecules_per_species": args.pair_count,
        "atom_count": len(atoms),
        "source": source,
        "methodology": resolved_methodology_record(args),
        "runtime": [runtime_row(0, 0, device)],
        "input": build_input_record(
            input_path=args.input_extxyz.resolve(),
            tensor_sha256=input_hash,
            manifest_path=(
                args.input_manifest.resolve() if args.input_manifest else None
            ),
            manifest=input_manifest,
        ),
        "settings": {
            "aimnet": aimnet_info,
            "pme": {
                **pme_setup,
                "mesh_dimensions": list(pme_setup["mesh_dimensions"]),
                "spline_order": args.pme_spline_order,
            },
            "ewald_reference": ewald_setup,
            "minimum_cell_length_a": minimum_cell_length_a,
            "compile_model": False,
        },
        "charges": {
            "available": True,
            "shape": list(charges.shape),
            "sha256": tensor_checksum(charges),
            "sum_e": charge_sum,
            "sum_abs_e": float(charges.to(torch.float64).abs().sum().item()),
            "max_abs_e": float(charges.to(torch.float64).abs().max().item()),
            "finite": bool(torch.isfinite(charges).all()),
        },
        "pme": {
            "energy_ev": pme_energy,
            "forces": force_summary(pme_forces),
        },
        "ewald": {
            "energy_ev": ewald_energy,
            "forces": force_summary(ewald_forces),
        },
        "comparison": {
            "energy_difference_ev": energy_difference,
            "absolute_energy_difference_ev": abs(energy_difference),
            "absolute_energy_difference_ev_per_atom": energy_difference_per_atom,
            "force_difference_rms_ev_a": force_difference_rms,
            "force_difference_max_norm_ev_a": force_difference_max_norm,
            "force_difference": force_summary(force_difference),
            "acceptance": acceptance,
            "passed": passed,
            "assessment": "Measured once with declared limits; settings are not tuned.",
        },
        "timing": {
            "wall_s": elapsed,
            "boundary": (
                "AIMNet2 charge forward, PME forward, and Ewald forward after "
                "model loading; includes neighbor construction."
            ),
        },
        "memory": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    return row


def run_capacity(
    args: argparse.Namespace,
    *,
    device: Any,
    checkpoint: Path,
    atoms: Any | None,
    source: dict[str, Any],
    input_manifest: dict[str, Any] | None,
    stage_tracker: dict[str, str],
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from nvalchemi.data import Batch
    from nvalchemi.distributed import (
        DistributedManager,
        DomainConfig,
        DomainParallel,
        SpatialPartitioner,
    )
    from nvalchemi.dynamics import BaseDynamics

    stage_tracker["stage"] = "distributed_initialize"
    DistributedManager.initialize()
    manager = DistributedManager()
    rank = int(manager.rank)
    world_size = int(manager.world_size)
    device = torch.device(manager.device)
    if world_size != args.world_size:
        raise ValueError(
            f"torchrun world size {world_size} != declared {args.world_size}"
        )
    mesh = manager.initialize_mesh(
        mesh_shape=(world_size,),
        mesh_dim_names=("domain",),
    )
    d3_path = args.d3_parameter_file.resolve()
    if not d3_path.is_file() or sha256_file(d3_path) != D3_PARAMETER_SHA256:
        raise RuntimeError("D3 parameter file is missing or has the wrong SHA-256")

    torch.cuda.reset_peak_memory_stats(device)
    setup_start = perf_counter()
    stage_tracker["stage"] = "input_transfer"
    full_batch = make_batch(atoms, device) if rank == 0 else None
    source_atom_ids = (
        source_atom_ids_tensor(atoms, device) if rank == 0 else None
    )
    stage_tracker["stage"] = "model_setup"
    pme_values = torch.zeros(7, dtype=torch.float64, device=device)
    if rank == 0:
        rank_zero_pme_setup = estimate_pme_setup(
            positions=full_batch.positions,
            cell=full_batch.cell,
            batch_idx=full_batch.batch_idx,
            real_space_cutoff_a=args.pme_cutoff_a,
            accuracy=args.pme_accuracy,
            mesh_safety_factor=args.pme_mesh_safety_factor,
        )
        pme_values.copy_(
            torch.tensor(
                (
                    rank_zero_pme_setup["alpha_a_inverse"],
                    *rank_zero_pme_setup["mesh_dimensions"],
                    *rank_zero_pme_setup["mesh_spacing_a"],
                ),
                dtype=torch.float64,
                device=device,
            )
        )
    dist.broadcast(pme_values, src=0)
    pme_values_list = [float(value) for value in pme_values.cpu().tolist()]
    resolved_mesh_dimensions = tuple(
        int(round(value)) for value in pme_values_list[1:4]
    )
    if (
        pme_values_list[0] <= 0.0
        or any(value <= 0 for value in resolved_mesh_dimensions)
        or any(value <= 0.0 for value in pme_values_list[4:7])
    ):
        raise RuntimeError("rank 0 did not provide valid PME parameters")
    resolved_pme_setup = {
        "real_space_cutoff_a": args.pme_cutoff_a,
        "alpha_a_inverse": pme_values_list[0],
        "mesh_dimensions": resolved_mesh_dimensions,
        "mesh_spacing_a": pme_values_list[4:7],
        "accuracy": args.pme_accuracy,
        "mesh_safety_factor": args.pme_mesh_safety_factor,
        "parameter_rule": (
            "estimate_pme_parameters(accuracy, real_space_cutoff, "
            "mesh_safety_factor)"
        ),
    }
    pipeline, model_info = build_complete_pipeline(
        checkpoint=checkpoint,
        d3_parameter_file=d3_path,
        device=device,
        args=args,
        pme_setup=resolved_pme_setup,
    )
    aimnet_cutoff_a = float(model_info["aimnet"]["cutoff_a"])
    if not math.isclose(
        aimnet_cutoff_a,
        EXPECTED_AIMNET_NEIGHBOR_CUTOFF_A,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "AIMNet2 checkpoint neighbor cutoff changed from the declared "
            f"{EXPECTED_AIMNET_NEIGHBOR_CUTOFF_A:g} Å to "
            f"{aimnet_cutoff_a:g} Å"
        )
    input_tensor_hash = None
    if full_batch is not None:
        assert source_atom_ids is not None
        input_tensor_hash = source_input_checksum(
            full_batch,
            source_atom_ids,
        )
    domain_config = DomainConfig(
        cutoff=max(
            aimnet_cutoff_a,
            args.pme_cutoff_a,
            args.d3_cutoff_a,
        ),
        skin=args.domain_skin_a,
        mesh=mesh,
        grid_dims=DOMAIN_METHODOLOGY.domain_grid_dims,
        compile=False,
        require_nondegenerate=world_size > 1,
    )
    layout_tensor = torch.zeros(6, dtype=torch.int64, device=device)
    gathered_source_ids = None
    if rank == 0:
        derived_layout = SpatialPartitioner(
            config=domain_config,
            cell_matrix=full_batch.cell,
            pbc=full_batch.pbc,
        )
        assert source_atom_ids is not None
        gathered_source_ids = predict_gathered_source_ids(
            source_atom_ids=source_atom_ids,
            positions=full_batch.positions,
            partitioner=derived_layout,
            world_size=world_size,
        )
        layout_tensor.copy_(
            torch.tensor(
                (*derived_layout.cells_per_dim, *derived_layout.rank_grid),
                dtype=torch.int64,
                device=device,
            )
        )
    dist.broadcast(layout_tensor, src=0)
    layout_values = tuple(int(value) for value in layout_tensor.cpu().tolist())
    cells_per_dim, rank_grid = validate_spatial_layout(
        layout_values[:3],
        layout_values[3:],
        world_size=world_size,
    )
    atomic_write_json(
        args.rank_output_dir.resolve() / f"rank-{rank:02d}.json",
        {
            "schema": RANK_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "rank": rank,
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "status": "running",
            "success": False,
            "stage": "layout_derived",
            "cells_per_dim": list(cells_per_dim),
            "rank_grid": list(rank_grid),
            "source": source,
        },
    )
    setup_s = perf_counter() - setup_start

    local_samples_s: list[float] = []
    samples_s_max_rank: list[float] = []
    workflow_max_allocated_bytes: list[int] = []
    workflow_max_reserved_bytes: list[int] = []
    source_input_hashes: list[str] = []
    owned: Batch | None = None
    result_owned: Batch | None = None
    gathered: Batch | None = None
    # Multi-rank DomainParallel primes forces once before its first requested
    # step. The one-rank pass-through does not. Equalize only the reportable
    # timing series so every GPU count performs two model evaluations.
    run_steps = (
        DOMAIN_METHODOLOGY.steady_timing_run_steps(world_size)
        if args.measurement_role == "steady_timing"
        else 1
    )
    if world_size > 1 and run_steps != 1:
        raise RuntimeError(
            "multi-rank source-order reconstruction requires one requested step"
        )
    automatic_initial_evaluations = (
        DOMAIN_METHODOLOGY.domain_parallel_multi_rank_initial_force_evaluations
        if world_size > 1
        else 0
    )
    model_evaluations_per_workflow = run_steps + automatic_initial_evaluations
    if (
        args.measurement_role == "steady_timing"
        and model_evaluations_per_workflow
        != DOMAIN_METHODOLOGY.steady_timing_model_evaluations_per_workflow
    ):
        raise RuntimeError("steady timing must perform two model evaluations")

    def run_public_workflow(*, measured: bool, index: int) -> None:
        """Run one synchronized workflow with a fresh public wrapper."""

        nonlocal owned, result_owned, gathered

        # Multi-rank DistributedPipelineModel builds its neighbor lists
        # internally. The normal one-rank pipeline receives fresh hooks with
        # each fresh BaseDynamics wrapper.
        hooks = pipeline.make_neighbor_hooks() if world_size == 1 else []
        inner = BaseDynamics(model=pipeline, n_steps=1, hooks=hooks)
        with DomainParallel(
            dynamics=inner,
            config=domain_config,
            n_steps=1,
        ) as domain:
            # Context construction/entry, reset, barrier, and initial CUDA
            # synchronization are deliberately outside the timer.
            torch.cuda.reset_peak_memory_stats(device)
            dist.barrier()
            torch.cuda.synchronize(device)
            if measured:
                start = perf_counter()
            stage_tracker["stage"] = (
                f"sample_{index}_partition" if measured else f"warmup_{index}_partition"
            )
            owned = domain.partition(full_batch if rank == 0 else None)
            stage_tracker["stage"] = (
                f"sample_{index}_run" if measured else f"warmup_{index}_run"
            )
            result_owned = domain.run(owned, n_steps=run_steps)
            stage_tracker["stage"] = (
                f"sample_{index}_gather" if measured else f"warmup_{index}_gather"
            )
            gathered = domain.gather(result_owned, dst=0)
            torch.cuda.synchronize(device)
            if measured:
                local_elapsed_s = perf_counter() - start
            workflow_max_allocated_bytes.append(
                int(torch.cuda.max_memory_allocated(device))
            )
            workflow_max_reserved_bytes.append(
                int(torch.cuda.max_memory_reserved(device))
            )

        # DomainParallel exit and the unchanged-input check remain outside the
        # timed public partition -> run -> gather boundary.
        input_unchanged = True
        if rank == 0:
            assert full_batch is not None
            assert source_atom_ids is not None
            observed_hash = source_input_checksum(
                full_batch,
                source_atom_ids,
            )
            source_input_hashes.append(observed_hash)
            input_unchanged = observed_hash == input_tensor_hash
        input_unchanged_tensor = torch.tensor(
            [int(input_unchanged)],
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(input_unchanged_tensor, src=0)
        if int(input_unchanged_tensor.item()) != 1:
            raise RuntimeError("the public workflow mutated the source input Batch")

        if measured:
            local_samples_s.append(local_elapsed_s)
            max_elapsed = torch.tensor(
                [local_elapsed_s],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
            samples_s_max_rank.append(float(max_elapsed.item()))

    for warmup_index in range(args.warmup_count):
        run_public_workflow(measured=False, index=warmup_index)
    for sample_index in range(args.sample_count):
        run_public_workflow(measured=True, index=sample_index)

    timing_summary = summarize_timing_samples(samples_s_max_rank)
    local_timing_summary = summarize_timing_samples(local_samples_s)
    timed_max_allocated_bytes = max(workflow_max_allocated_bytes)
    timed_max_reserved_bytes = max(workflow_max_reserved_bytes)
    if owned is None or result_owned is None:
        raise RuntimeError("no DomainParallel workflow completed")
    owned_count = int(owned.num_nodes)
    stage_tracker["stage"] = "result_collection"

    # Shared-filesystem writes happen only after the measured CUDA work.
    atomic_write_json(
        args.rank_output_dir.resolve() / f"rank-{rank:02d}.json",
        {
            "schema": RANK_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "rank": rank,
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "status": "running",
            "success": False,
            "stage": "partition_run_gather_complete",
            "owned_atom_count": owned_count,
            "halo_atom_count": None,
            "halo_atom_count_reason": "not_exposed_by_public_api",
            "cells_per_dim": list(cells_per_dim),
            "rank_grid": list(rank_grid),
            "wall_s": local_timing_summary["median_s"],
            "samples_s_local_rank": local_samples_s,
            "samples_s_max_rank": samples_s_max_rank,
            "max_allocated_bytes": timed_max_allocated_bytes,
            "max_reserved_bytes": timed_max_reserved_bytes,
        },
    )

    local_outputs_valid = (
        result_owned.energy is not None
        and result_owned.energy.numel() == 1
        and bool(torch.isfinite(result_owned.energy).all())
        and result_owned.forces is not None
        and result_owned.forces.ndim == 2
        and result_owned.forces.shape[1] == 3
        and bool(torch.isfinite(result_owned.forces).all())
    )
    all_outputs_valid = torch.tensor(
        [int(local_outputs_valid)], dtype=torch.int32, device=device
    )
    dist.all_reduce(all_outputs_valid, op=dist.ReduceOp.MIN)
    if int(all_outputs_valid.item()) != 1:
        raise RuntimeError(
            "DomainParallel returned a missing, non-finite, or malformed "
            "energy/force result on at least one rank"
        )
    replicated_energy = result_owned.energy.detach().clone()
    energies_by_rank = [torch.empty_like(replicated_energy) for _ in range(world_size)]
    dist.all_gather(energies_by_rank, replicated_energy)
    if not all(
        torch.equal(energy, energies_by_rank[0]) for energy in energies_by_rank[1:]
    ):
        raise RuntimeError("the globally reduced energy differs between ranks")

    local = {
        **runtime_row(rank, int(os.environ["LOCAL_RANK"]), device),
        "owned_atom_count": owned_count,
        "halo_atom_count": None,
        "halo_atom_count_reason": "not_exposed_by_public_api",
        "wall_s": local_timing_summary["median_s"],
        "samples_s_local_rank": local_samples_s,
        "max_allocated_bytes": timed_max_allocated_bytes,
        "max_reserved_bytes": timed_max_reserved_bytes,
    }
    atomic_write_json(
        args.rank_output_dir.resolve() / f"rank-{rank:02d}.json",
        {
            "schema": RANK_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "status": "running",
            "success": False,
            "stage": "runtime_identity_complete",
            **local,
        },
    )
    per_rank: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(per_rank, local)
    runtime_rows = [item for item in per_rank if item is not None]
    if len(runtime_rows) != world_size:
        raise RuntimeError("one or more ranks did not return runtime identity")
    validate_runtime_rows(
        runtime_rows,
        expected_software=source["runtime_software"],
    )
    input_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(input_hashes, input_tensor_hash)

    row: dict[str, Any] = {}
    if rank == 0:
        assert gathered is not None
        assert atoms is not None
        expected_atom_count = len(atoms)
        if int(gathered.num_nodes) != expected_atom_count:
            raise RuntimeError(
                f"gather returned {gathered.num_nodes} atoms; "
                f"expected {expected_atom_count}"
            )
        if gathered.positions.shape != (expected_atom_count, 3):
            raise RuntimeError("gathered positions have the wrong shape")
        if gathered.forces.shape != (expected_atom_count, 3):
            raise RuntimeError("gathered forces have the wrong shape")
        if not bool(torch.isfinite(gathered.positions).all()):
            raise RuntimeError("gathered positions contain a non-finite value")
        if not bool(torch.isfinite(gathered.forces).all()):
            raise RuntimeError("gathered forces contain a non-finite value")

        assert gathered_source_ids is not None
        assert source_atom_ids is not None
        order = source_order_from_gathered_ids(
            gathered_source_ids,
            expected_atom_count=expected_atom_count,
        )
        sorted_forces = gathered.forces[order]
        sorted_positions = gathered.positions[order]
        sorted_atomic_numbers = gathered.atomic_numbers.reshape(-1)[order]
        source_file_order = torch.argsort(source_atom_ids, stable=True)
        expected_atomic_numbers = torch.as_tensor(
            atoms.numbers, dtype=sorted_atomic_numbers.dtype, device=device
        )[source_file_order]
        if not torch.equal(sorted_atomic_numbers, expected_atomic_numbers):
            raise RuntimeError("gather changed the source-ordered atomic numbers")
        # Toolkit 0.2 gathers atom-level fields. The globally reduced total
        # energy is already replicated on each rank before gather.
        energy_ev = float(replicated_energy.reshape(-1)[0].item())
        row = {
            "schema": RESULT_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "mode": args.mode,
            "status": "complete",
            "success": True,
            "world_size": world_size,
            "pair_count": args.pair_count,
            "molecules_per_species": args.pair_count,
            "atom_count": int(gathered.num_nodes),
            "source": source,
            "methodology": resolved_methodology_record(args),
            "runtime": runtime_rows,
            "input": build_input_record(
                input_path=args.input_extxyz.resolve(),
                tensor_sha256=input_hashes[0],
                manifest_path=(
                    args.input_manifest.resolve() if args.input_manifest else None
                ),
                manifest=input_manifest,
            ),
            "model": model_info,
            "distributed": {
                "api": "DomainParallel",
                "mesh_shape": [world_size],
                "mesh_dim_names": ["domain"],
                "grid_dims": domain_config.grid_dims,
                "cells_per_dim": list(cells_per_dim),
                "rank_grid": list(rank_grid),
                "domain_cutoff_a": domain_config.cutoff,
                "domain_skin_a": domain_config.skin,
                "compile": False,
                "require_nondegenerate": world_size > 1,
                "owned_atom_counts": [
                    int(item["owned_atom_count"]) for item in runtime_rows
                ],
                "halo_atom_counts": None,
                "halo_atom_counts_reason": "not_exposed_by_public_api",
                "pme_reciprocal_mesh": (
                    "replicated on every rank in the Toolkit 0.2 version used here"
                ),
                "gathered_atom_order": (
                    "source_atom_id kept outside Batch; rank-contiguous gather "
                    "order reproduced with "
                    "SpatialPartitioner.assign_atoms_to_ranks"
                ),
            },
            "output": {
                "energy_ev": energy_ev,
                "energy_ev_per_atom": energy_ev / int(gathered.num_nodes),
                "forces_source_atom_order": force_summary(sorted_forces),
                "positions_source_atom_order_sha256": tensor_checksum(sorted_positions),
                "atomic_numbers_source_atom_order_sha256": tensor_checksum(
                    sorted_atomic_numbers
                ),
                "source_atom_id_sha256": tensor_checksum(
                    gathered_source_ids[order]
                ),
            },
            "charges": (
                {
                    "available": True,
                    "values": None,
                    "sum_e": float(gathered.charges.to(torch.float64).sum().item()),
                    "sha256": tensor_checksum(gathered.charges),
                    "finite": bool(torch.isfinite(gathered.charges).all()),
                    "reason": "Values are summarized rather than copied into JSON.",
                }
                if world_size == 1
                else {
                    "available": False,
                    "values": None,
                    "sum_e": None,
                    "reason": (
                        "The Toolkit 0.2 DistributedPipelineModel AIMNet2-to-PME "
                        "group returns energy and forces only. The same input's "
                        "one-GPU reference records charge neutrality."
                    ),
                }
            ),
            "timing": {
                "setup_s_rank0": setup_s,
                "wall_s_max_rank": timing_summary["median_s"],
                "samples_s_max_rank": samples_s_max_rank,
                **timing_summary,
                "measurement_kind": (
                    "steady_partition_run_gather"
                    if args.measurement_role == "steady_timing"
                    else "cold_one_shot_partition_run_gather"
                ),
                "measurement_role": args.measurement_role,
                "warmup_count": args.warmup_count,
                "sample_count": args.sample_count,
                "run_steps": run_steps,
                "automatic_multi_rank_force_prime": world_size > 1,
                "automatic_initial_force_evaluations": (
                    automatic_initial_evaluations
                ),
                "model_evaluations_per_workflow": model_evaluations_per_workflow,
                "publishable_benchmark": False,
                "elapsed_reduction": "maximum across ranks via all_reduce",
                "quartile_method": "inclusive linear interpolation",
                "source_input_sha256_before": input_tensor_hash,
                "source_input_sha256_after_each_workflow": source_input_hashes,
                "boundary": (
                    "Each workflow uses a fresh BaseDynamics and DomainParallel "
                    "context. Context entry, the rank barrier, initial CUDA "
                    "synchronization, max-rank reduction, context exit, input and "
                    "output checks, statistics, and file writes are outside the "
                    "timer. The timer covers exactly the public partition, run, "
                    "and gather calls plus final CUDA synchronization. In steady "
                    "timing, one rank requests two BaseDynamics steps; several "
                    "ranks request one step after DomainParallel's automatic "
                    "initial force evaluation, giving two model evaluations in "
                    "both paths."
                ),
                "interpretation": (
                    "Speedup is reportable only for the complete dedicated "
                    "steady_timing 1/2/4-GPU series on the identical input."
                ),
            },
            "memory": {
                "max_allocated_bytes_per_rank": [
                    int(item["max_allocated_bytes"]) for item in runtime_rows
                ],
                "max_reserved_bytes_per_rank": [
                    int(item["max_reserved_bytes"]) for item in runtime_rows
                ],
                "max_allocated_bytes": max(
                    int(item["max_allocated_bytes"]) for item in runtime_rows
                ),
                "max_reserved_bytes": max(
                    int(item["max_reserved_bytes"]) for item in runtime_rows
                ),
                "boundary": (
                    "Peaks reset after model setup and input transfer for each "
                    "workflow; the reported value is the maximum over workflows. "
                    "Validation and file writes are excluded."
                ),
            },
        }
        if args.force_output_npy is None:
            raise ValueError(
                "--force-output-npy is required for capacity, parity, rescue, "
                "and steady-timing measurements"
            )
        force_path = args.force_output_npy.resolve()
        atomic_write_npy(force_path, sorted_forces.detach().cpu().numpy())
        force_file = file_identity(force_path)
        row["output"]["forces_source_atom_order_npy"] = {
            **force_file,
            "dtype": str(sorted_forces.dtype),
            "shape": list(sorted_forces.shape),
        }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "capacity",
            "parity",
            "distributed",
            "steady-timing",
            "electrostatics-validation",
        ),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--measurement-role",
        choices=(
            "capacity",
            "parity",
            "rescue",
            "steady_timing",
            "electrostatics_validation",
        ),
        required=True,
    )
    parser.add_argument("--warmup-count", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--input-extxyz", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rank-output-dir", type=Path, required=True)
    parser.add_argument("--force-output-npy", type=Path)
    parser.add_argument("--checkpoint", default="aimnet2-wb97m-d3_0")
    parser.add_argument("--d3-parameter-file", type=Path, required=True)
    parser.add_argument(
        "--pme-cutoff-a",
        type=float,
        default=DEFAULT_PME_CUTOFF_A,
    )
    parser.add_argument(
        "--pme-mesh-safety-factor",
        type=float,
        default=DEFAULT_PME_MESH_SAFETY_FACTOR,
        help="Multiplier used by Toolkit-Ops' PME mesh estimator.",
    )
    parser.add_argument(
        "--pme-spline-order",
        type=int,
        default=DEFAULT_PME_SPLINE_ORDER,
    )
    parser.add_argument(
        "--pme-accuracy",
        type=float,
        default=DEFAULT_PME_ACCURACY,
    )
    parser.add_argument(
        "--ewald-reference-accuracy",
        type=float,
        default=DEFAULT_EWALD_REFERENCE_ACCURACY,
    )
    parser.add_argument(
        "--pme-ewald-energy-tol-ev-per-atom",
        type=float,
        default=DEFAULT_PME_EWAL_ENERGY_TOL_EV_PER_ATOM,
    )
    parser.add_argument(
        "--pme-ewald-force-max-tol-ev-a",
        type=float,
        default=DEFAULT_PME_EWAL_FORCE_MAX_TOL_EV_A,
    )
    parser.add_argument(
        "--charge-sum-tol-e",
        type=float,
        default=DEFAULT_CHARGE_SUM_TOL_E,
    )
    parser.add_argument("--d3-cutoff-a", type=float, default=DEFAULT_D3_CUTOFF_A)
    parser.add_argument(
        "--d3-smoothing-fraction",
        type=float,
        default=DEFAULT_D3_SMOOTHING_FRACTION,
    )
    parser.add_argument(
        "--domain-skin-a",
        type=float,
        default=DEFAULT_DOMAIN_SKIN_A,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_measurement_args(args)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_output = args.rank_output_dir.resolve() / f"rank-{rank:02d}.json"
    stage_tracker = {"stage": "imports"}
    distributed_manager_initialized = False
    source_context: dict[str, Any] | None = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("this runner requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        repository_root = Path(__file__).resolve().parents[1]
        source = verify_runtime_source(repository_root)
        source_context = source
        stage_tracker["stage"] = "checkpoint"
        checkpoint = resolve_checkpoint(args.checkpoint)
        checkpoint_file = file_identity(checkpoint)
        runner_file = file_identity(Path(__file__).resolve())
        source["aimnet_checkpoint"] = checkpoint_file["path"]
        source["aimnet_checkpoint_sha256"] = checkpoint_file["sha256"]
        source["aimnet_checkpoint_file"] = checkpoint_file
        source["runner"] = runner_file["path"]
        source["runner_sha256"] = runner_file["sha256"]
        source["runner_file"] = runner_file

        stage_tracker["stage"] = "input"
        atoms = None
        input_manifest = None
        if args.mode == "electrostatics-validation" or rank == 0:
            atoms, input_manifest = load_atoms_and_manifest(
                args.input_extxyz.resolve(),
                args.input_manifest.resolve() if args.input_manifest else None,
            )
            expected_atoms = args.pair_count * ATOMS_PER_COMPOSITION_UNIT
            if len(atoms) != expected_atoms:
                raise ValueError(
                    f"input has {len(atoms)} atoms; expected {expected_atoms}"
                )

        stage_tracker["stage"] = args.mode
        if args.mode == "electrostatics-validation":
            row = run_electrostatics_validation(
                args,
                device=device,
                checkpoint=checkpoint,
                atoms=atoms,
                source=source,
                input_manifest=input_manifest,
            )
        else:
            distributed_manager_initialized = True
            row = run_capacity(
                args,
                device=device,
                checkpoint=checkpoint,
                atoms=atoms,
                source=source,
                input_manifest=input_manifest,
                stage_tracker=stage_tracker,
            )
        previous_rank_record: dict[str, Any] = {}
        if rank_output.is_file():
            try:
                previous_rank_record = json.loads(
                    rank_output.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                previous_rank_record = {}
        rank_record = {
            **previous_rank_record,
            "schema": RANK_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "rank": rank,
            "local_rank": local_rank,
            "status": "complete",
            "success": True,
            "stage": "complete",
            "source": source,
        }
        atomic_write_json(rank_output, rank_record)
        if rank == 0:
            atomic_write_json(args.output_json.resolve(), row)
        return 0
    except Exception as exc:
        previous: dict[str, Any] = {}
        if rank_output.is_file():
            try:
                previous = json.loads(rank_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        failure = {
            "schema": RANK_SCHEMA,
            "created_utc": utc_now(),
            "run_id": args.run_id,
            "case_id": args.case_id,
            "measurement_role": args.measurement_role,
            "rank": rank,
            "local_rank": local_rank,
            "status": "failed",
            "success": False,
            "failure": {
                "type": type(exc).__name__,
                "stage": stage_tracker["stage"],
                "is_cuda_oom": type(exc).__name__ == "OutOfMemoryError"
                or "out of memory" in str(exc).lower(),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "owned_atom_count": previous.get("owned_atom_count"),
            "cells_per_dim": previous.get("cells_per_dim"),
            "rank_grid": previous.get("rank_grid"),
            "halo_atom_count": None,
            "halo_atom_count_reason": "not_exposed_by_public_api",
            "source": source_context,
        }
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda", local_rank)
                failure["runtime"] = runtime_row(rank, local_rank, device)
                failure["memory"] = {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
        except Exception:
            pass
        atomic_write_json(rank_output, failure)
        raise
    finally:
        if distributed_manager_initialized:
            active_exception = sys.exc_info()[0] is not None
            try:
                from nvalchemi.distributed import DistributedManager

                DistributedManager.cleanup()
            except Exception as cleanup_error:
                if not active_exception:
                    raise
                print(
                    f"DistributedManager cleanup also failed: {cleanup_error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
