#!/usr/bin/env python3
"""Check the package versions used by the Part 1 notebook."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any


EXPECTED_CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
EXPECTED_OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
EXPECTED_PACKMOL_VERSION = "21.2.1"
EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_TORCH_VERSION = "2.12.0+cu130"
EXPECTED_CUDA_VERSION = "13.0"
EXPECTED_VERSIONS = {
    "aimnet": "0.2.0",
    "e3nn": "0.5.9",
    "jax": "0.9.0.1",
    "nvalchemi-toolkit": "0.2.0",
    "nvalchemi-toolkit-ops": "0.4.0",
    "nvidia-physicsnemo": "2.1.1",
    "ovito": "3.15.4",
    "sevenn": "0.13.0",
    "warp-lang": "1.13.0",
}
RECORDED_SCIENTIFIC_VERSIONS = (
    "ase",
    "matscipy",
    "numpy",
    "pandas",
    "pydantic",
    "pymatgen",
    "scipy",
    "torch-geometric",
    "triton",
)
SOURCE_MANIFEST_RELATIVE_PATH = "build/part1-source-files.txt"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_base_environment(
    python_prefix: str | Path,
    configured_base: str | Path | None,
) -> Path:
    """Resolve the Conda prefix that supplies Packmol and OVITO.

    A normal image installs Python packages and Conda tools in one prefix. The
    Compute Lab path uses a Python overlay on top of a smaller Conda base, so
    ``sys.prefix`` identifies the overlay while ``ALCHEMI_MAIN_ENV`` identifies
    the Conda prefix.
    """

    if configured_base is None or not str(configured_base).strip():
        return Path(python_prefix).resolve()
    base = Path(configured_base).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"configured base environment does not exist: {base}")
    return base


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_paths(source_root: Path) -> tuple[str, ...]:
    """Load the versioned list of files that can affect a Part 1 run."""

    manifest_path = source_root.resolve() / SOURCE_MANIFEST_RELATIVE_PATH
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"could not read Part 1 source manifest: {manifest_path}"
        ) from exc

    paths: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        relative = raw_line.strip()
        if not relative or relative.startswith("#"):
            continue
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative != pure.as_posix():
            raise RuntimeError(
                "invalid repository-relative path in Part 1 source manifest "
                f"at line {line_number}: {relative!r}"
            )
        if relative in seen:
            raise RuntimeError(f"duplicate path in Part 1 source manifest: {relative}")
        seen.add(relative)
        paths.append(relative)

    if SOURCE_MANIFEST_RELATIVE_PATH not in seen:
        raise RuntimeError("Part 1 source manifest must include itself")
    if not paths:
        raise RuntimeError("Part 1 source manifest is empty")
    return tuple(paths)


REQUIRED_TRACKED_SOURCE_PATHS = load_source_paths(_REPOSITORY_ROOT)


def installed_git_commit(distribution_name: str) -> str:
    """Return the commit recorded for a package installed from Git."""

    direct_url_text = metadata.distribution(distribution_name).read_text(
        "direct_url.json"
    )
    if direct_url_text is None:
        raise RuntimeError(f"{distribution_name} has no direct_url.json record")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise RuntimeError(f"{distribution_name} is not installed from a Git commit")
    return str(commit)


def _git_output(source_root: Path, *arguments: str) -> str:
    """Run a read-only Git command in the tutorial checkout."""

    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git could not inspect the tutorial source: {message}")
    return completed.stdout.strip()


def git_source_revision(source_root: Path) -> dict[str, str]:
    """Return the commit and tree for one tutorial checkout."""

    source_root = source_root.resolve()
    return {
        "repository_commit": _git_output(source_root, "rev-parse", "HEAD"),
        "repository_tree": _git_output(source_root, "rev-parse", "HEAD^{tree}"),
    }


def _source_identity(
    source_root: Path,
    *,
    clean_checkout: bool,
) -> dict[str, Any]:
    """Record the exact tutorial revision and file bytes used by a run."""

    source_root = source_root.resolve()
    source_paths = load_source_paths(source_root)
    missing = [
        relative for relative in source_paths if not (source_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing Part 1 source files: {missing}")
    files_sha256 = {
        relative: sha256_file(source_root / relative) for relative in source_paths
    }
    return {
        "clean_checkout": clean_checkout,
        **git_source_revision(source_root),
        "manifest_path": SOURCE_MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_file(source_root / SOURCE_MANIFEST_RELATIVE_PATH),
        "files_sha256": files_sha256,
    }


def verify_clean_tracked_source(source_root: Path) -> dict[str, Any]:
    """Require one clean Git checkout containing the files used by this run."""

    source_root = source_root.resolve()
    checkout_root = Path(
        _git_output(source_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if checkout_root != source_root:
        raise RuntimeError(
            "tutorial source root does not match the Git checkout root: "
            f"{source_root} != {checkout_root}"
        )
    _git_output(source_root, "rev-parse", "--verify", "HEAD")
    source_changes = _git_output(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if source_changes:
        raise RuntimeError(
            "tutorial source has tracked or untracked files; stage one clean "
            "revision before running the H100 notebook"
        )
    ignored_files = _git_output(
        source_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
    )
    if ignored_files:
        raise RuntimeError(
            "tutorial source contains ignored files; use a fresh checkout "
            "of the requested revision"
        )
    source_paths = load_source_paths(source_root)
    _git_output(
        source_root,
        "ls-files",
        "--error-unmatch",
        "--",
        *source_paths,
    )
    return _source_identity(source_root, clean_checkout=True)


def source_report(
    source_root: Path,
    *,
    require_clean_source: bool,
    skip_source_check: bool,
) -> dict[str, Any]:
    """Describe the source check used for this runtime report."""

    if require_clean_source and skip_source_check:
        raise ValueError(
            "require_clean_source and skip_source_check are mutually exclusive"
        )
    if skip_source_check:
        return {
            "checked": False,
            "clean_checkout": None,
            "reason": (
                "source identity is checked separately; this invocation checks "
                "only the baked runtime"
            ),
        }
    if require_clean_source:
        return {"checked": True, **verify_clean_tracked_source(source_root)}
    return {
        "checked": True,
        **_source_identity(source_root, clean_checkout=False),
    }


def verify_runtime(
    source_root: Path,
    *,
    require_cuda: bool,
    require_clean_source: bool = False,
    skip_source_check: bool = False,
) -> dict[str, Any]:
    """Import the notebook stack and return a compact environment report."""

    source_root = source_root.resolve()
    part1_root = source_root / "part-1-scalable-atomistic-workflows"
    if not part1_root.is_dir():
        raise FileNotFoundError(f"Part 1 source directory not found: {part1_root}")
    source_identity = source_report(
        source_root,
        require_clean_source=require_clean_source,
        skip_source_check=skip_source_check,
    )
    if sys.version_info[:3] != EXPECTED_PYTHON:
        found = ".".join(map(str, sys.version_info[:3]))
        expected = ".".join(map(str, EXPECTED_PYTHON))
        raise RuntimeError(f"expected Python {expected}, found {found}")
    sys.path.insert(0, str(part1_root))

    versions = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
    resolved_scientific_versions = {
        name: metadata.version(name) for name in RECORDED_SCIENTIFIC_VERSIONS
    }
    for name, expected in EXPECTED_VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(f"expected {name} {expected}, found {versions[name]}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import aimnet
    import e3nn
    import jax
    import jax.numpy as jnp
    import nvalchemi
    import nvalchemiops
    import ovito
    import physicsnemo
    import sevenn
    import torch
    import warp
    from aux.adsorption import load_initial_structure_set
    from aux.adsorption_visualization import _ovito_compatible_copy
    from aux.domain.packing import (
        build_nci_molecular_box,
        plan_nci_molecular_box,
        validate_molecular_box,
    )
    from aux.models.sevennet import SevenNetOmniWrapper
    from aux.nci_atlas import load_nci_atlas_subset
    from nvalchemi.distributed import DomainConfig, DomainParallel
    from nvalchemi.models import (
        AIMNet2Wrapper,
        DFTD3ModelWrapper,
        PMEModelWrapper,
        PipelineModelWrapper,
    )
    from ovito.io.ase import ase_to_ovito

    del (
        AIMNet2Wrapper,
        DFTD3ModelWrapper,
        DomainConfig,
        DomainParallel,
        PMEModelWrapper,
        PipelineModelWrapper,
        SevenNetOmniWrapper,
    )

    packmol_command = shutil.which("packmol")
    if packmol_command is None:
        raise RuntimeError("Packmol executable not found on PATH")
    packmol_binary = Path(packmol_command).resolve()
    base_environment = resolve_base_environment(
        sys.prefix,
        os.environ.get("ALCHEMI_MAIN_ENV"),
    )
    uv_output = subprocess.run(
        [str(base_environment / "bin" / "uv"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    uv_fields = uv_output.split()
    if len(uv_fields) < 2 or uv_fields[0] != "uv":
        raise RuntimeError(f"could not parse uv version output: {uv_output!r}")
    uv_version = uv_fields[1]
    try:
        uv_version_tuple = tuple(int(part) for part in uv_version.split(".")[:3])
    except ValueError as error:
        raise RuntimeError(f"could not parse uv version: {uv_version!r}") from error
    if len(uv_version_tuple) != 3 or uv_version_tuple < (0, 9, 26):
        raise RuntimeError(f"uv 0.9.26 or newer is required, found {uv_version!r}")
    if not packmol_binary.is_relative_to(base_environment):
        raise RuntimeError(
            "Packmol executable is outside the configured Conda environment: "
            f"{packmol_binary}"
        )
    conda_records = sorted((base_environment / "conda-meta").glob("packmol-*.json"))
    if len(conda_records) != 1:
        raise RuntimeError(
            "expected one Packmol record in the configured Conda environment"
        )
    packmol_record = json.loads(conda_records[0].read_text(encoding="utf-8"))
    if (
        packmol_record.get("name") != "packmol"
        or packmol_record.get("version") != EXPECTED_PACKMOL_VERSION
    ):
        raise RuntimeError(
            "expected Packmol "
            f"{EXPECTED_PACKMOL_VERSION}, found {packmol_record.get('version')!r}"
        )

    nci_data = load_nci_atlas_subset(
        part1_root / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
    )
    packmol_plan = plan_nci_molecular_box(
        nci_data,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=20260723,
    )
    with TemporaryDirectory(prefix="part1-packmol-check-") as temporary:
        packed = build_nci_molecular_box(
            packmol_plan,
            temporary,
            packmol_binary=packmol_binary,
        )
        packmol_check = validate_molecular_box(packmol_plan, packed)

    if nvalchemi.version != EXPECTED_VERSIONS["nvalchemi-toolkit"]:
        raise RuntimeError(
            "nvalchemi.version does not match the installed Toolkit package: "
            f"{nvalchemi.version}"
        )
    torch_distribution_version = metadata.version("torch")
    if torch_distribution_version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH_VERSION}, "
            f"found {torch_distribution_version}"
        )
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            "imported Torch does not match its package record: "
            f"{torch.__version__} != {torch_distribution_version}"
        )
    if torch.version.cuda != EXPECTED_CUDA_VERSION:
        raise RuntimeError(
            f"expected Torch CUDA {EXPECTED_CUDA_VERSION}, found {torch.version.cuda}"
        )

    commits = {
        "nvalchemi-toolkit": installed_git_commit("nvalchemi-toolkit"),
        "nvalchemi-toolkit-ops": installed_git_commit("nvalchemi-toolkit-ops"),
    }
    expected_commits = {
        "nvalchemi-toolkit": EXPECTED_CORE_COMMIT,
        "nvalchemi-toolkit-ops": EXPECTED_OPS_COMMIT,
    }
    if commits != expected_commits:
        raise RuntimeError(
            "Toolkit Git revisions do not match Part 1: "
            f"Core={commits['nvalchemi-toolkit']}, "
            f"Ops={commits['nvalchemi-toolkit-ops']}"
        )

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("Torch cannot see a CUDA device")
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    if require_cuda and (cuda_device is None or "H100" not in cuda_device.upper()):
        raise RuntimeError(f"this run requires an H100 GPU; found {cuda_device}")
    jax_cuda_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if require_cuda and not jax_cuda_devices:
        raise RuntimeError("JAX cannot see a CUDA device")

    ovito_particle_counts: dict[str, int] = {}
    for name, atoms in load_initial_structure_set().items():
        data = ase_to_ovito(_ovito_compatible_copy(atoms))
        particle_count = int(data.particles.count)
        if particle_count != len(atoms):
            raise RuntimeError(
                f"OVITO converted {name} to {particle_count} particles; "
                f"expected {len(atoms)}"
            )
        ovito_particle_counts[name] = particle_count

    ops_cuda_check: dict[str, int] | None = None
    if cuda_available:
        from nvalchemiops.jax.segment_ops import (
            segmented_sum as jax_segmented_sum,
        )
        from nvalchemiops.segment_ops import (
            segmented_sum as warp_segmented_sum,
        )
        from nvalchemiops.torch import segmented_sum
        from nvalchemiops.torch.neighbors import neighbor_list

        device = torch.device("cuda")
        values = torch.tensor([1.0, 2.0], device=device)
        graph_index = torch.tensor([0, 0], dtype=torch.int32, device=device)
        segment_total = segmented_sum(values, graph_index, 1)
        if not torch.equal(segment_total, torch.tensor([3.0], device=device)):
            raise RuntimeError("Toolkit-Ops segmented_sum failed on CUDA")

        if not jax_cuda_devices:
            raise RuntimeError("Torch sees CUDA but JAX does not")
        with jax.default_device(jax_cuda_devices[0]):
            jax_values = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
            jax_graph_index = jnp.asarray([0, 0], dtype=jnp.int32)
        jax_segment_total = jax_segmented_sum(jax_values, jax_graph_index, 1)
        jax_segment_total.block_until_ready()
        if float(jax_segment_total[0]) != 3.0:
            raise RuntimeError("Toolkit-Ops JAX segmented_sum failed on CUDA")

        torch.cuda.synchronize(device)
        warp_values = warp.from_torch(values, dtype=warp.float32)
        warp_graph_index = warp.from_torch(graph_index, dtype=warp.int32)
        warp_segment_total = warp.zeros(
            1, dtype=warp.float32, device=warp_values.device
        )
        warp_segmented_sum(warp_values, warp_graph_index, warp_segment_total)
        warp.synchronize_device(warp_values.device)
        if float(warp_segment_total.numpy()[0]) != 3.0:
            raise RuntimeError("Toolkit-Ops raw Warp segmented_sum failed on CUDA")

        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        batch_ptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
        edge_index, edge_ptr = neighbor_list(
            positions=positions,
            cutoff=1.5,
            batch_idx=graph_index,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )
        if edge_index.shape != (2, 2) or int(edge_ptr[-1].item()) != 2:
            raise RuntimeError(
                "Toolkit-Ops CUDA neighbor search returned an unexpected shape: "
                f"edges={tuple(edge_index.shape)}, edge_ptr={edge_ptr.tolist()}"
            )
        torch.cuda.synchronize()
        ops_cuda_check = {
            "directed_edges": int(edge_index.shape[1]),
            "jax_segments": int(jax_segment_total.size),
            "segments": int(segment_total.numel()),
            "warp_segments": int(warp_segment_total.shape[0]),
        }

    modules = {
        name: str(Path(module.__file__).resolve())
        for name, module in {
            "aimnet": aimnet,
            "e3nn": e3nn,
            "jax": jax,
            "nvalchemi": nvalchemi,
            "nvalchemiops": nvalchemiops,
            "ovito": ovito,
            "physicsnemo": physicsnemo,
            "sevenn": sevenn,
            "torch": torch,
            "warp": warp,
        }.items()
    }
    return {
        "schema": "alchemi.part1-runtime-check.v2",
        "source": source_identity,
        "python": sys.version,
        "python_executable": sys.executable,
        "base_environment": str(base_environment),
        "versions": {
            **versions,
            "packmol": EXPECTED_PACKMOL_VERSION,
            "torch": torch.__version__,
            "uv": uv_version,
        },
        "resolved_scientific_versions": resolved_scientific_versions,
        "commits": commits,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "jax_cuda_device": str(jax_cuda_devices[0]) if jax_cuda_devices else None,
        "packmol_binary": str(packmol_binary),
        "packmol_check": {
            "atoms": packmol_check.atom_count,
            "molecules": packmol_check.molecule_count,
            "net_charge_e": packmol_check.net_charge_e,
            "packmol_precision_a": packmol_check.packmol_precision_a,
            "density_from_mass_and_cell_g_cm3": (
                packmol_check.density_from_mass_and_cell_g_cm3
            ),
            "periodic_min_distance_lower_bound_a": (
                packmol_check.periodic_min_distance_lower_bound_a
            ),
        },
        "ovito_ase_check": {
            "structures": len(ovito_particle_counts),
            "particle_counts": ovito_particle_counts,
        },
        "toolkit_ops_cuda_check": ops_cuda_check,
        "module_files": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    source_options = parser.add_mutually_exclusive_group()
    source_options.add_argument("--require-clean-source", action="store_true")
    source_options.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_runtime(
        args.source_root,
        require_cuda=args.require_cuda,
        require_clean_source=args.require_clean_source,
        skip_source_check=args.skip_source_check,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
