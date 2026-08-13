#!/usr/bin/env python3
"""Record one already-launched world-size case for the Part 08 campaign."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import socket
import sys
import time
from importlib.metadata import distribution
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from ase.io import read
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DistributedManager, DomainConfig, DomainParallel
from nvalchemi.dynamics import BaseDynamics
from nvalchemi.models import AIMNet2Wrapper

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NOTEBOOK_DIR.parents[1]
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from helpers import (
    array_sha256,
    sha256_file,
    validate_campaign,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record one current-pin DomainParallel case. Start this script with "
            "one process per visible GPU using an external distributed launcher."
        )
    )
    parser.add_argument(
        "--campaign-spec",
        type=Path,
        default=Path(__file__).with_name("campaign-spec.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional pre-downloaded AIMNet2 checkpoint; its checksum is mandatory.",
    )
    return parser.parse_args()


def load_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema") != "alchemi.part08-domain-campaign.v1":
        raise RuntimeError("Unexpected campaign schema.")
    return campaign


def installed_pin(name: str) -> dict[str, str]:
    package = distribution(name)
    direct_url_text = package.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{name} has no direct_url.json VCS provenance.")
    direct_url = json.loads(direct_url_text)
    commit = str(direct_url.get("vcs_info", {}).get("commit_id", ""))
    if len(commit) != 40:
        raise RuntimeError(f"{name} has no full installed VCS commit.")
    return {
        "distribution": name,
        "version": package.version,
        "commit": commit,
    }


def verify_installed_pins(campaign: dict[str, Any]) -> dict[str, Any]:
    installed = {
        "toolkit": installed_pin("nvalchemi-toolkit"),
        "toolkit_ops": installed_pin("nvalchemi-toolkit-ops"),
    }
    if installed != campaign["current_pins"]:
        raise RuntimeError(
            "Installed Toolkit/Toolkit-Ops provenance does not match the campaign."
        )
    return installed


def resolve_checkpoint(
    campaign: dict[str, Any],
    requested: Path | None,
) -> Path:
    model = campaign["workload"]["model"]
    if requested is None:
        from aimnet.calculators.model_registry import get_model_path

        path = Path(get_model_path(model["alias"])).resolve()
    else:
        path = requested.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != model["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint checksum mismatch: {digest}")
    return path


def _input_tensor_sha(batch: Batch) -> str:
    identities = {
        "atomic_numbers": array_sha256(
            batch.atomic_numbers.detach().cpu().numpy()
        ),
        "positions": array_sha256(batch.positions.detach().cpu().numpy()),
        "cell": array_sha256(batch.cell.detach().cpu().numpy()),
        "pbc": array_sha256(batch.pbc.detach().cpu().numpy()),
        "charge": array_sha256(batch.charge.detach().cpu().numpy()),
        "source_atom_id": array_sha256(
            batch.source_atom_id.detach().cpu().numpy()
        ),
    }
    import hashlib

    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_campaign_batch(
    campaign: dict[str, Any],
    *,
    device: torch.device,
) -> Batch:
    workload = campaign["workload"]
    base_path = (REPO_ROOT / workload["base_structure"]).resolve()
    if sha256_file(base_path) != workload["base_structure_sha256"]:
        raise RuntimeError("Base phenol/NMA structure checksum mismatch.")

    atoms = read(base_path)
    atoms.wrap()
    atoms = atoms.repeat(tuple(map(int, workload["repeat_factors_xyz"])))
    atoms.wrap()
    if len(atoms) != int(workload["atom_count"]):
        raise RuntimeError("Generated campaign atom count is wrong.")
    if not bool(np.asarray(atoms.pbc).all()):
        raise RuntimeError("Campaign input must be periodic in all dimensions.")

    graph = AtomicData.from_atoms(atoms, device=device)
    graph.add_system_property(
        "charge",
        torch.zeros((1, 1), dtype=graph.positions.dtype, device=device),
    )
    graph.add_system_property(
        "energy",
        torch.zeros((1, 1), dtype=graph.positions.dtype, device=device),
    )
    graph.add_node_property("forces", torch.zeros_like(graph.positions))
    graph_fields = graph.model_dump(exclude_none=True)
    graph_fields["source_atom_id"] = torch.arange(
        len(atoms), dtype=torch.int64, device=device
    )
    graph = AtomicData(**graph_fields)
    batch = Batch.from_data_list(
        [graph],
        device=device,
        field_levels={"source_atom_id": "atom"},
    )
    return batch


def input_identity(batch: Batch, campaign: dict[str, Any]) -> dict[str, Any]:
    """Record the exact tensor identity shared by every world-size case."""

    identity = {
        "atom_count": int(batch.num_nodes),
        "base_structure_sha256": campaign["workload"]["base_structure_sha256"],
        "tensor_sha256": _input_tensor_sha(batch),
        "cell_a": batch.cell.detach().cpu().numpy().tolist(),
        "pbc": batch.pbc.detach().cpu().numpy().tolist()[0],
    }
    if identity["tensor_sha256"] != campaign["workload"]["input_tensor_sha256"]:
        raise RuntimeError("Generated campaign input tensor checksum mismatch.")
    return identity


def build_model(checkpoint: Path, device: torch.device) -> AIMNet2Wrapper:
    model = AIMNet2Wrapper.from_checkpoint(
        checkpoint,
        device=device,
        compile_model=False,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.set_config("active_outputs", {"energy", "forces"})
    return model


def maximum_mic_displacement(
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> float:
    delta = final_positions - initial_positions
    fractional = delta @ np.linalg.inv(cell)
    fractional[:, np.asarray(pbc, dtype=bool)] -= np.rint(
        fractional[:, np.asarray(pbc, dtype=bool)]
    )
    mic = fractional @ cell
    return float(np.linalg.norm(mic, axis=1).max(initial=0.0))


def cuda_driver_version() -> str:
    """Read the loaded CUDA driver version through the public driver API."""

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return "unavailable"
    encoded = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(encoded)) != 0:
        return "unavailable"
    major = encoded.value // 1000
    minor = (encoded.value % 1000) // 10
    return f"{major}.{minor}"


def rank_runtime(rank: int, local_rank: int, owned_atoms: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "owned_atom_count": owned_atoms,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "driver_version": cuda_driver_version(),
    }


def write_case(
    *,
    campaign: dict[str, Any],
    output_root: Path,
    world_size: int,
    current_pins: dict[str, Any],
    input_identity: dict[str, Any],
    checkpoint: Path,
    gathered: Batch,
    owned_atom_counts: list[int],
    runtimes: list[dict[str, Any]],
    pass_times_s: list[float],
    initial_positions: np.ndarray,
) -> Path:
    case_dir = output_root / f"gpus-{world_size:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    source_ids = gathered.source_atom_id.detach().cpu().numpy()
    forces = gathered.forces.detach().cpu().numpy()
    positions = gathered.positions.detach().cpu().numpy()
    source_order_positions = positions[np.argsort(source_ids, kind="stable")]
    cell = gathered.cell[0].detach().cpu().numpy()
    pbc = gathered.pbc[0].detach().cpu().numpy()

    artifact = case_dir / "result.npz"
    np.savez(artifact, forces=forces, source_atom_id=source_ids)
    case_record = {
        "schema": "alchemi.part08-domain-case.v1",
        "status": "complete",
        "world_size": world_size,
        "current_pins": current_pins,
        "input": input_identity,
        "model": {
            "adapter": "AIMNet2Wrapper",
            "alias": campaign["workload"]["model"]["alias"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "scope": campaign["workload"]["model"]["scope"],
        },
        "distributed": {
            "api": "DomainParallel",
            "mesh_shape": [world_size],
            "mesh_dim_names": ["domain"],
            "grid_dims": campaign["workload"]["domain"]["grid_dims"],
            "cutoff_a": campaign["workload"]["domain"]["cutoff_a"],
            "skin_a": campaign["workload"]["domain"]["skin_a"],
            "owned_atom_counts": owned_atom_counts,
            "halo_atom_counts": None,
            "halo_atom_counts_reason": "not exposed by the public API",
        },
        "output": {
            "energy_ev": float(gathered.energy.detach().reshape(-1)[0].cpu()),
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "forces_sha256": array_sha256(forces),
            "source_atom_id_sha256": array_sha256(source_ids),
            "maximum_mic_displacement_a": maximum_mic_displacement(
                initial_positions,
                source_order_positions,
                cell,
                pbc,
            ),
        },
        "timing": {
            "warmup_count": campaign["measurement"]["warmup_count"],
            "pass_times_s": pass_times_s,
            "median_s": median(pass_times_s),
            "elapsed_reduction": "maximum across ranks",
            "ranks_synchronized": True,
            "publishable_benchmark": False,
        },
        "runtime": {
            "gpu_names": [record["gpu_name"] for record in runtimes],
            "python_version": runtimes[0]["python_version"],
            "torch_version": runtimes[0]["torch_version"],
            "torch_cuda_version": runtimes[0]["torch_cuda_version"],
            "driver_version": runtimes[0]["driver_version"],
            "ranks": runtimes,
        },
        "provenance": {
            "runner": Path(__file__).name,
            "runner_sha256": sha256_file(Path(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    case_path = case_dir / "case.json"
    case_path.write_text(
        json.dumps(case_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return case_path


def update_campaign(
    template: dict[str, Any],
    *,
    output_root: Path,
    world_size: int,
    case_path: Path,
) -> None:
    manifest_path = output_root / "campaign.json"
    if manifest_path.is_file():
        campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "current_pins", "workload", "acceptance"):
            if campaign.get(key) != template.get(key):
                raise RuntimeError(f"Existing campaign changed immutable field: {key}")
    else:
        campaign = json.loads(json.dumps(template))

    campaign["cases"][str(world_size)] = str(case_path.relative_to(output_root))
    missing = [
        value
        for value in campaign["required_world_sizes"]
        if str(value) not in campaign["cases"]
    ]
    campaign["status"] = "partial" if missing else "complete"
    campaign["status_reason"] = (
        f"Waiting for serial world-size cases: {missing}."
        if missing
        else "All required case files are present; validation is in progress."
    )

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)

    if not missing:
        report = validate_campaign(campaign, root=output_root)
        if not report.ready:
            campaign["status"] = "INVALID"
            campaign["status_reason"] = report.reason
            manifest_path.write_text(
                json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(report.reason)
        campaign["status_reason"] = report.reason
        manifest_path.write_text(
            json.dumps(campaign, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_case(args: argparse.Namespace) -> None:
    campaign = load_campaign(args.campaign_spec.resolve())
    if args.world_size not in campaign["required_world_sizes"]:
        raise ValueError("World size is outside the declared 1/2/4-GPU campaign.")
    if not torch.cuda.is_available():
        raise RuntimeError("This external campaign requires one process per visible GPU.")

    DistributedManager.initialize()
    manager = DistributedManager()
    try:
        if manager.world_size != args.world_size:
            raise RuntimeError(
                f"Distributed world size {manager.world_size} != {args.world_size}."
            )
        device = torch.device(manager.device)
        torch.cuda.set_device(device)
        mesh = manager.initialize_mesh(
            mesh_shape=(manager.world_size,),
            mesh_dim_names=("domain",),
        )
        current_pins = verify_installed_pins(campaign)

        checkpoint_holder: list[str | None] = [None]
        if manager.rank == 0:
            checkpoint_holder[0] = str(
                resolve_checkpoint(campaign, args.checkpoint)
            )
        dist.broadcast_object_list(checkpoint_holder, src=0)
        checkpoint = Path(str(checkpoint_holder[0]))
        if sha256_file(checkpoint) != campaign["workload"]["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs across ranks.")

        full_batch = build_campaign_batch(
            campaign,
            device=device,
        ) if manager.rank == 0 else None
        input_holder = [
            input_identity(full_batch, campaign) if full_batch is not None else None
        ]
        dist.broadcast_object_list(input_holder, src=0)
        input_record = input_holder[0]
        if not isinstance(input_record, dict):
            raise TypeError("Rank 0 did not broadcast the input identity.")
        initial_positions = (
            full_batch.positions.detach().cpu().numpy().copy()
            if manager.rank == 0
            else None
        )

        model = build_model(checkpoint, device)
        configured_cutoff = float(model.model_config.neighbor_config.cutoff)
        if configured_cutoff != float(campaign["workload"]["domain"]["cutoff_a"]):
            raise RuntimeError("Model cutoff differs from the campaign.")
        domain_config = DomainConfig(
            cutoff=configured_cutoff,
            skin=float(campaign["workload"]["domain"]["skin_a"]),
            mesh=mesh,
            grid_dims=campaign["workload"]["domain"]["grid_dims"],
            compile=bool(campaign["workload"]["domain"]["compile"]),
        )
        evaluator = BaseDynamics(
            model=model,
            n_steps=1,
            hooks=model.make_neighbor_hooks(),
        )

        with DomainParallel(
            dynamics=evaluator,
            config=domain_config,
            n_steps=1,
            device_type="cuda",
        ) as domain:
            owned_batch = domain.partition(full_batch)
            owned_count = int(owned_batch.positions.shape[0])
            owned_atom_counts: list[int] = [0] * manager.world_size
            dist.all_gather_object(owned_atom_counts, owned_count)

            domain_result = domain.run(owned_batch, n_steps=1)
            torch.cuda.synchronize()
            pass_times_s: list[float] = []
            for _ in range(int(campaign["measurement"]["measured_pass_count"])):
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                domain_result = domain.run(domain_result, n_steps=1)
                torch.cuda.synchronize()
                elapsed = torch.tensor(
                    time.perf_counter() - started,
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                pass_times_s.append(float(elapsed.cpu()))
            gathered = domain.gather(domain_result, dst=0)

        runtime = rank_runtime(manager.rank, manager.local_rank, owned_count)
        runtimes: list[dict[str, Any] | None] = [None] * manager.world_size
        dist.all_gather_object(runtimes, runtime)

        write_error: list[str | None] = [None]
        if manager.rank == 0:
            try:
                if gathered is None or initial_positions is None:
                    raise RuntimeError("Rank 0 did not receive the gathered batch.")
                output_root = args.output_root.resolve()
                output_root.mkdir(parents=True, exist_ok=True)
                case_path = write_case(
                    campaign=campaign,
                    output_root=output_root,
                    world_size=manager.world_size,
                    current_pins=current_pins,
                    input_identity=input_record,
                    checkpoint=checkpoint,
                    gathered=gathered,
                    owned_atom_counts=owned_atom_counts,
                    runtimes=[item for item in runtimes if item is not None],
                    pass_times_s=pass_times_s,
                    initial_positions=initial_positions,
                )
                update_campaign(
                    campaign,
                    output_root=output_root,
                    world_size=manager.world_size,
                    case_path=case_path,
                )
            except Exception as error:  # noqa: BLE001
                # Propagate rank-0 file and validation failures to every rank.
                write_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(write_error, src=0)
        if write_error[0] is not None:
            raise RuntimeError(write_error[0])
    finally:
        DistributedManager.cleanup(barrier=False)


def main() -> None:
    run_case(parse_args())


if __name__ == "__main__":
    main()
