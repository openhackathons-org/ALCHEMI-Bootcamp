#!/usr/bin/env python3
"""Run the recorded Part 1 water campaign on one, two, or four H100 GPUs.

The publishable routes all process the same deterministic set of water
hexamers with the same eager molecular potential:

* ``fused_1gpu`` keeps relaxation, NVT, and NVE work in one ``FusedStage``;
* ``pipeline_2gpu`` sends relaxed batches from FIRE2 to fused NVT + NVE;
* ``pipeline_4gpu`` runs two independent copies of that two-stage pipeline.

Each distributed route constructs its stages once. Its upstream
``SizeAwareSampler`` owns the complete campaign partition and keeps the same
pipeline running until that partition is exhausted.

Launch this file with ``torchrun`` through the repository launcher. The
one-GPU route is the Toolkit fused-workflow baseline, not a serial Python loop.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
PART_DIR = REPOSITORY_ROOT / "part-1-scalable-atomistic-workflows"
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

import nvalchemi  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics import (  # noqa: E402
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FIRE2,
    FusedStage,
    HostMemory,
    NVE,
    NVTLangevin,
    SizeAwareSampler,
    initialize_velocities,
)
from nvalchemi.dynamics.base import BufferConfig  # noqa: E402
from nvalchemi.dynamics.hooks import (  # noqa: E402
    ConvergedSnapshotHook,
    NaNDetectorHook,
)
from nvalchemi.hooks import DynamicsContext  # noqa: E402
from nvalchemi.models import (  # noqa: E402
    AIMNet2Wrapper,
    DFTD3ModelWrapper,
    PipelineGroup,
    PipelineModelWrapper,
)
from nvalchemi.neighbors import compute_neighbors  # noqa: E402

from aux.artifacts import sha256_file  # noqa: E402
from aux.checkpoint import checkpoint_card, resolve_checkpoint_path  # noqa: E402
from aux.electrostatics import DirectCoulombWrapper  # noqa: E402
from aux.hooks import (  # noqa: E402
    StageStepCounterHook,
    converge_after_steps,
)
from aux.runtime import check_batch_buffer_transfer, verify_toolkit_pins  # noqa: E402
from aux.structures import make_cyclic_water_hexamer  # noqa: E402
from part1_distributed_campaign_contract import (  # noqa: E402
    AIMNET_CHECKPOINT,
    ATOMS_PER_SYSTEM,
    BALANCE_PROBE_BATCHES,
    BALANCE_PROBE_SCHEMA,
    CAMPAIGN_SEED,
    CHARGE_ATOL_E,
    CORE_BRANCH,
    CORE_COMMIT,
    CORE_VERSION,
    CORRECTNESS_CHECKS,
    CURRENT_RC_TIMING_STATUS,
    COVALENT_OH_CUTOFF_A,
    D3_CUTOFF_A,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMM_MODE,
    DEFAULT_DT_FS,
    DEFAULT_FIRE_FMAX_EV_PER_A,
    DEFAULT_FRICTION_PER_FS,
    DEFAULT_SYSTEMS,
    DEFAULT_TEMPERATURE_K,
    ENERGY_ATOL_EV,
    EXPECTED_AIMNET_CHECKPOINT_SHA256,
    EXPECTED_D3_PARAMETER_SHA256,
    FORCE_ATOL_EV_PER_A,
    MAX_ABS_NET_CHARGE_E,
    MIN_INTERATOMIC_DISTANCE_A,
    NEIGHBOR_SKIN_A,
    OPS_COMMIT,
    OPS_VERSION,
    OXYGEN_CONNECTIVITY_CUTOFF_A,
    PRODUCER_FILES,
    ROUTE_PAIR_BOUNDARIES,
    ROUTE_TOPOLOGY,
    RUN_SCHEMA,
    TIMING_BOUNDARY,
    VELOCITY_SEED,
    pair_boundaries_for_world_size,
)


_NEIGHBOR_KEYS = frozenset(
    {
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_list",
        "neighbor_list_shifts",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_once(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_id(args: argparse.Namespace) -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "interactive")
    if args.purpose == "tuning":
        return (
            f"{job_id}-tuning-{args.route}-s{args.systems}-b{args.batch_size}-"
            f"{args.comm_mode}-r{args.repeat}"
        )
    return f"{job_id}-{args.route}-r{args.repeat}"


def _partition_campaign_ids(total: int, partitions: int) -> list[tuple[int, ...]]:
    """Split the full campaign evenly across one or two upstream samplers."""

    if partitions not in (1, 2):
        raise ValueError("campaign supports one or two pipelines")
    if total < partitions:
        raise ValueError("systems must be at least the number of pipelines")
    if total % partitions != 0:
        raise ValueError("systems must divide evenly across pipeline pairs")
    if partitions == 1:
        return [tuple(range(total))]
    return [
        tuple(campaign_id for campaign_id in range(total) if campaign_id % 2 == pair)
        for pair in range(2)
    ]


def _make_campaign_structures(total: int) -> tuple[list[Atoms], str]:
    """Build deterministic, gently perturbed cyclic water hexamers."""

    structures: list[Atoms] = []
    digest = hashlib.sha256()
    definition = {
        "schema": "alchemi.water-hexamer-campaign-input.v1",
        "seed": CAMPAIGN_SEED,
        "systems": total,
        "oo_distance_levels_a": [2.72 + 0.02 * level for level in range(8)],
        "level_rule": "(campaign_id // 2) % 8",
        "coordinate_noise_sigma_a": 0.025,
        "rng_rule": "one PCG64 generator advanced in global campaign ID order",
    }
    digest.update(json.dumps(definition, sort_keys=True).encode("utf-8"))
    rng = np.random.default_rng(CAMPAIGN_SEED)

    for campaign_id in range(total):
        level = (campaign_id // 2) % 8
        atoms = make_cyclic_water_hexamer(oo_distance=2.72 + 0.02 * level)
        displacement = rng.normal(
            loc=0.0,
            scale=0.025,
            size=atoms.positions.shape,
        )
        displacement -= displacement.mean(axis=0, keepdims=True)
        atoms.positions += displacement
        atoms.center(about=(0.0, 0.0, 0.0))
        atoms.set_pbc(False)
        atoms.info["charge"] = 0
        atoms.info["campaign_id"] = campaign_id
        structures.append(atoms)

        digest.update(np.asarray([campaign_id], dtype="<i8").tobytes())
        digest.update(np.asarray(atoms.numbers, dtype="<i8").tobytes())
        digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())

    return structures, digest.hexdigest()


class WaterHexamerDataset:
    """Dataset interface used by Toolkit ``SizeAwareSampler``."""

    def __init__(
        self,
        structures: list[Atoms],
        campaign_ids: Sequence[int],
        device: torch.device,
    ) -> None:
        self._structures = structures
        self._campaign_ids = tuple(campaign_ids)
        self._device = device

    def __len__(self) -> int:
        return len(self._campaign_ids)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, int]]:
        campaign_id = self._campaign_ids[index]
        data = AtomicData.from_atoms(
            self._structures[campaign_id],
            device=self._device,
            # AIMNet2Wrapper evaluates coordinates in float32. Start the
            # complete dynamics state in that dtype so FIRE2, velocity
            # initialization, positions, velocities, and forces agree.
            dtype=torch.float32,
        )
        zeros_system_long = torch.zeros(
            1, 1, dtype=torch.long, device=self._device
        )
        zeros_system_float = torch.zeros(
            1, 1, dtype=torch.float32, device=self._device
        )
        data.add_system_property(
            "campaign_id",
            torch.tensor([[campaign_id]], dtype=torch.long, device=self._device),
        )
        data.add_system_property("energy", zeros_system_float.clone())
        data.add_system_property("relaxation_steps", zeros_system_long.clone())
        data.add_system_property("handoff_fmax", zeros_system_float.clone())
        data.add_system_property("velocity_initialized", zeros_system_long.clone())
        data.add_system_property("nvt_steps_done", zeros_system_long.clone())
        data.add_system_property("nve_steps_done", zeros_system_long.clone())
        data.add_node_property("forces", torch.zeros_like(data.positions))
        data.add_node_property("velocities", torch.zeros_like(data.positions))
        data.add_node_property(
            "charges",
            torch.zeros(data.num_nodes, dtype=torch.float32, device=self._device),
        )
        return data, {"num_atoms": data.num_nodes, "num_edges": data.num_edges}

    def get_metadata(self, index: int) -> tuple[int, int]:
        del index
        return ATOMS_PER_SYSTEM, 0


def _per_graph_fmax(batch: Batch) -> torch.Tensor:
    force_norm = torch.linalg.vector_norm(batch.forces, dim=1)
    result = torch.zeros(
        batch.num_graphs, dtype=force_norm.dtype, device=force_norm.device
    )
    for graph in range(batch.num_graphs):
        mask = batch.batch_idx == graph
        result[graph] = force_norm[mask].max()
    return result


class RelaxationStepCounterHook:
    """Count FIRE2 updates for each active structure without imposing a cap."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, source_status: int) -> None:
        self.source_status = source_status

    def __call__(self, context: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        status = context.batch.status.reshape(-1)
        active = status == self.source_status
        context.batch.relaxation_steps.reshape(-1)[active] += 1


class VelocityHandoffHook:
    """Verify the FIRE2 force gate and initialize velocities before NVT."""

    stage = DynamicsStage.ON_CONVERGE
    frequency = 1

    def __init__(self, fmax_threshold: float, temperature_k: float) -> None:
        self.fmax_threshold = float(fmax_threshold)
        self.temperature_k = float(temperature_k)

    def __call__(self, context: DynamicsContext, stage: DynamicsStage) -> None:
        del stage
        mask = context.converged_mask
        if mask is None or not mask.any():
            return

        batch = context.batch
        fmax = _per_graph_fmax(batch)
        selected = torch.where(mask)[0]
        if bool((fmax[selected] > self.fmax_threshold + 1.0e-10).any()):
            raise RuntimeError("velocity handoff received a structure above the force gate")

        ptr = batch.batch_ptr.detach().cpu().tolist()
        campaign_ids = batch.campaign_id.reshape(-1).detach().cpu().tolist()
        for graph in selected.detach().cpu().tolist():
            start, stop = ptr[graph], ptr[graph + 1]
            local_batch_idx = torch.zeros(
                stop - start, dtype=torch.int32, device=batch.device
            )
            initialize_velocities(
                batch.velocities[start:stop],
                batch.atomic_masses[start:stop],
                torch.tensor(
                    [self.temperature_k],
                    dtype=batch.positions.dtype,
                    device=batch.device,
                ),
                local_batch_idx,
                random_seed=VELOCITY_SEED + int(campaign_ids[graph]),
                remove_com=True,
                remove_rotations=True,
                rescale=True,
                positions=batch.positions[start:stop],
            )

        batch.handoff_fmax.reshape(-1)[selected] = fmax[selected]
        batch.velocity_initialized.reshape(-1)[selected] = 1


@dataclass
class ModelBuild:
    model: PipelineModelWrapper
    checkpoint_sha256: str
    d3_parameter_sha256: str
    d3_bj_parameters: dict[str, float]


def _build_model(device: torch.device, d3_parameter_file: Path) -> ModelBuild:
    checkpoint_source = os.environ.get("ALCHEMI_AIMNET_CHECKPOINT", AIMNET_CHECKPOINT)
    checkpoint_path = resolve_checkpoint_path(checkpoint_source)
    aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device=device)
    aimnet.eval()
    for parameter in aimnet.parameters():
        parameter.requires_grad_(False)
    aimnet.set_config("active_outputs", {"energy", "charges"})
    card = checkpoint_card(aimnet, checkpoint_source, checkpoint_path)
    if card["checkpoint_sha256"] != EXPECTED_AIMNET_CHECKPOINT_SHA256:
        raise RuntimeError("AIMNet2 checkpoint hash does not match the campaign pin")

    if not d3_parameter_file.is_file():
        raise FileNotFoundError(f"D3 parameter file not found: {d3_parameter_file}")
    d3_sha256 = sha256_file(d3_parameter_file)
    if d3_sha256 != EXPECTED_D3_PARAMETER_SHA256:
        raise RuntimeError("D3 parameter file hash does not match")

    d3_params = {
        key: float(card["d3_params"][key])
        for key in ("a1", "a2", "s6", "s8")
    }
    coulomb = DirectCoulombWrapper().to(device)
    d3 = DFTD3ModelWrapper(
        a1=d3_params["a1"],
        a2=d3_params["a2"],
        s8=d3_params["s8"],
        s6=d3_params.get("s6", 1.0),
        cutoff=D3_CUTOFF_A,
        param_file=d3_parameter_file,
        auto_download=False,
    ).to(device)
    for component in (aimnet, d3):
        if component.model_config.neighbor_config is not None:
            component.model_config.neighbor_config.skin = NEIGHBOR_SKIN_A
    d3.set_config("active_outputs", {"energy", "forces"})

    model = PipelineModelWrapper(
        groups=[
            PipelineGroup(steps=[aimnet, coulomb], use_autograd=True),
            PipelineGroup(steps=[d3], use_autograd=False),
        ],
        neighbor_adaptation="always",
    ).to(device)
    model.set_config("active_outputs", {"energy", "forces", "charges"})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return ModelBuild(
        model=model,
        checkpoint_sha256=card["checkpoint_sha256"],
        d3_parameter_sha256=d3_sha256,
        d3_bj_parameters=d3_params,
    )


def _make_sampler(
    structures: list[Atoms],
    campaign_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> SizeAwareSampler:
    return SizeAwareSampler(
        dataset=WaterHexamerDataset(structures, campaign_ids, device),
        max_atoms=batch_size * ATOMS_PER_SYSTEM,
        # Neighbor lists are rebuilt by model hooks and remain on active
        # batches. max_edges=0 would make RC refill_check see a negative edge
        # budget after the first model call and stop admitting replacements.
        # Atom and graph counts bound this fixed-size molecular workload.
        max_edges=None,
        max_batch_size=batch_size,
        bin_width=1,
        shuffle=False,
    )


def _attach_model_hooks(stage: Any, model: PipelineModelWrapper) -> None:
    for hook in model.make_neighbor_hooks():
        stage.register_hook(hook)
    stage.register_hook(NaNDetectorHook(frequency=10, extra_keys=["velocities"]))


def _make_fire(
    model: PipelineModelWrapper,
    *,
    fmax_threshold: float,
    temperature_k: float,
    source_status: int = 0,
    **kwargs: Any,
) -> FIRE2:
    fire = FIRE2(
        model=model,
        dt=0.01,
        convergence_hook=ConvergenceHook.from_fmax(threshold=fmax_threshold),
        **kwargs,
    )
    fire.register_hook(RelaxationStepCounterHook(source_status=source_status))
    fire.register_hook(
        VelocityHandoffHook(
            fmax_threshold=fmax_threshold,
            temperature_k=temperature_k,
        )
    )
    return fire


def _make_nvt(
    model: PipelineModelWrapper,
    *,
    n_steps: int,
    dt_fs: float,
    temperature_k: float,
    friction_per_fs: float,
) -> NVTLangevin:
    return NVTLangevin(
        model=model,
        dt=dt_fs,
        temperature=temperature_k,
        friction=friction_per_fs,
        random_seed=CAMPAIGN_SEED,
        convergence_hook=converge_after_steps("nvt_steps_done", n_steps),
    )


def _make_nve(model: PipelineModelWrapper, *, n_steps: int, dt_fs: float) -> NVE:
    return NVE(
        model=model,
        dt=dt_fs,
        convergence_hook=converge_after_steps("nve_steps_done", n_steps),
    )


@dataclass
class WorkflowBuild:
    workflow: FusedStage | DistributedPipeline
    sinks: dict[int, HostMemory]
    initial_batch: Batch | None
    pipeline_count: int


def _build_fused_workflow(
    *,
    model: PipelineModelWrapper,
    structures: list[Atoms],
    systems: int,
    batch_size: int,
    fmax_threshold: float,
    nvt_steps: int,
    nve_steps: int,
    dt_fs: float,
    temperature_k: float,
    friction_per_fs: float,
    device: torch.device,
) -> WorkflowBuild:
    sampler = _make_sampler(structures, range(systems), device, batch_size)
    sink = HostMemory(capacity=systems)
    fire = _make_fire(
        model,
        fmax_threshold=fmax_threshold,
        temperature_k=temperature_k,
        source_status=0,
    )
    nvt = _make_nvt(
        model,
        n_steps=nvt_steps,
        dt_fs=dt_fs,
        temperature_k=temperature_k,
        friction_per_fs=friction_per_fs,
    )
    nve = _make_nve(model, n_steps=nve_steps, dt_fs=dt_fs)
    nve.register_hook(ConvergedSnapshotHook(sink=sink))
    fused = FusedStage(
        sub_stages=[(0, fire), (1, nvt), (2, nve)],
        sampler=sampler,
        refill_frequency=1,
        max_batch_size=batch_size,
        device_type="cuda",
    )
    fused.register_fused_hook(
        StageStepCounterHook({1: "nvt_steps_done", 2: "nve_steps_done"})
    )
    _attach_model_hooks(fused, model)
    initial_batch = sampler.build_initial_batch()
    fused.active_batch = initial_batch
    return WorkflowBuild(
        workflow=fused,
        sinks={0: sink},
        initial_batch=initial_batch,
        pipeline_count=0,
    )


def _build_distributed_sinks(
    systems: int,
    world_size: int,
) -> dict[int, HostMemory]:
    pair_boundaries = pair_boundaries_for_world_size(world_size)
    campaign_partitions = _partition_campaign_ids(systems, len(pair_boundaries))
    return {
        downstream_rank: HostMemory(capacity=len(campaign_ids))
        for (_, downstream_rank), campaign_ids in zip(
            pair_boundaries, campaign_partitions, strict=True
        )
    }


def _build_distributed_workflow(
    *,
    model: PipelineModelWrapper,
    structures: list[Atoms],
    campaign_ids_by_pair: Sequence[Sequence[int]],
    sinks: dict[int, HostMemory],
    batch_size: int,
    fmax_threshold: float,
    nvt_steps: int,
    nve_steps: int,
    dt_fs: float,
    temperature_k: float,
    friction_per_fs: float,
    comm_mode: str,
    world_size: int,
    device: torch.device,
) -> WorkflowBuild:
    """Build one pipeline run whose samplers own the complete partitions."""

    if world_size not in (2, 4):
        raise ValueError("distributed campaign requires two or four ranks")
    pair_boundaries = pair_boundaries_for_world_size(world_size)
    pair_count = len(pair_boundaries)
    if len(campaign_ids_by_pair) != pair_count:
        raise ValueError("each pipeline pair needs one campaign-ID partition")
    if any(not campaign_ids for campaign_ids in campaign_ids_by_pair):
        raise ValueError("each pipeline pair needs a non-empty campaign partition")
    buffer = BufferConfig(
        num_systems=batch_size,
        num_nodes=batch_size * ATOMS_PER_SYSTEM,
        num_edges=0,
    )
    stages: dict[int, Any] = {}

    for (upstream_rank, downstream_rank), campaign_ids in zip(
        pair_boundaries, campaign_ids_by_pair, strict=True
    ):
        sampler = _make_sampler(structures, campaign_ids, device, batch_size)
        try:
            sink = sinks[downstream_rank]
        except KeyError as exc:
            raise ValueError(
                f"missing final HostMemory sink for rank {downstream_rank}"
            ) from exc

        fire = _make_fire(
            model,
            fmax_threshold=fmax_threshold,
            temperature_k=temperature_k,
            source_status=0,
            sampler=sampler,
            refill_frequency=1,
            prior_rank=None,
            next_rank=downstream_rank,
            max_batch_size=batch_size,
            buffer_config=buffer,
            comm_mode=comm_mode,
            device_type="cuda",
        )
        _attach_model_hooks(fire, model)

        nvt = _make_nvt(
            model,
            n_steps=nvt_steps,
            dt_fs=dt_fs,
            temperature_k=temperature_k,
            friction_per_fs=friction_per_fs,
        )
        nve = _make_nve(model, n_steps=nve_steps, dt_fs=dt_fs)
        nve.register_hook(ConvergedSnapshotHook(sink=sink))
        dynamics = FusedStage(
            sub_stages=[(0, nvt), (1, nve)],
            prior_rank=upstream_rank,
            next_rank=None,
            max_batch_size=batch_size,
            buffer_config=buffer,
            comm_mode=comm_mode,
            device_type="cuda",
        )
        dynamics.register_fused_hook(
            StageStepCounterHook({0: "nvt_steps_done", 1: "nve_steps_done"})
        )
        _attach_model_hooks(dynamics, model)

        stages[upstream_rank] = fire
        stages[downstream_rank] = dynamics

    pipeline = DistributedPipeline(
        stages=stages,
        synchronized=False,
        backend="nccl",
        device_id=device,
    )
    return WorkflowBuild(
        workflow=pipeline,
        sinks=sinks,
        initial_batch=None,
        pipeline_count=pair_count,
    )


def _warm_model(
    model: PipelineModelWrapper,
    structures: list[Atoms],
    batch_size: int,
    device: torch.device,
) -> None:
    dataset = WaterHexamerDataset(
        structures,
        range(min(batch_size, len(structures))),
        device,
    )
    batch = Batch.from_data_list(
        [dataset[index][0] for index in range(len(dataset))],
        device=device,
    )
    for _ in range(2):
        compute_neighbors(batch, config=model.model_config.neighbor_config)
        outputs = model(batch)
        if not all(
            torch.isfinite(outputs[key]).all()
            for key in ("energy", "forces", "charges")
        ):
            raise RuntimeError("model warm-up produced a non-finite value")
    torch.cuda.synchronize(device)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def validate_batch_put_roundtrip() -> None:
    """Require exact fields and safe repeated use of the transfer buffer."""

    report = check_batch_buffer_transfer("cpu")
    if report["passed"]:
        return
    failures = [
        f"{case['float_dtype']}: {', '.join(case['mismatches'])}"
        for case in report["cases"]
        if not case["passed"]
    ]
    raise RuntimeError(
        f"{CURRENT_RC_TIMING_STATUS} Batch.put failed the CPU round-trip: "
        f"{'; '.join(failures)}. No GPU campaign was launched."
    )


def _validate_runtime(core_root: Path) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required for stable transfer schemas")

    imported_core_root = Path(nvalchemi.__file__).resolve().parents[1]
    if imported_core_root != core_root:
        raise RuntimeError(
            "Imported Toolkit Core does not come from ALCHEMI_TOOLKIT_CORE_ROOT"
        )
    if _git_output(core_root, "rev-parse", "HEAD") != CORE_COMMIT:
        raise RuntimeError("Toolkit Core commit does not match 0.2.0-rc")
    if _git_output(core_root, "branch", "--show-current") != CORE_BRANCH:
        raise RuntimeError("Toolkit Core checkout is not on the 0.2.0-rc branch")
    if _git_output(core_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Toolkit Core checkout must be clean and unmodified")
    if nvalchemi.version != CORE_VERSION:
        raise RuntimeError(
            f"Toolkit Core version must be {CORE_VERSION}, found {nvalchemi.version}"
        )
    if metadata.version("nvalchemi-toolkit-ops") != OPS_VERSION:
        raise RuntimeError(
            "Toolkit Ops package version does not match the pinned 0.4.0 release"
        )

    verify_toolkit_pins(CORE_COMMIT, OPS_COMMIT)
    validate_batch_put_roundtrip()


def _drop_neighbor_fields(batch: Batch) -> None:
    for key in _NEIGHBOR_KEYS:
        try:
            del batch[key]
        except (KeyError, IndexError):
            pass


def _water_chemistry_metrics(batch: Batch) -> tuple[float, bool, bool]:
    expected_numbers = torch.tensor(
        [8, 1, 1] * 6,
        dtype=batch.atomic_numbers.dtype,
        device=batch.atomic_numbers.device,
    )
    minimum_distance = float("inf")
    covalent_ok = True
    oxygen_network_ok = True
    ptr = batch.batch_ptr.detach().cpu().tolist()

    for graph in range(batch.num_graphs):
        start, stop = ptr[graph], ptr[graph + 1]
        numbers = batch.atomic_numbers[start:stop].reshape(-1)
        if not torch.equal(numbers, expected_numbers):
            raise RuntimeError(f"campaign graph {graph} has unexpected atomic numbers")
        positions = batch.positions[start:stop]
        distances = torch.cdist(positions, positions)
        non_diagonal = distances + torch.eye(
            stop - start, dtype=distances.dtype, device=distances.device
        ) * 1.0e6
        minimum_distance = min(minimum_distance, float(non_diagonal.min().item()))

        oxygen = torch.where(numbers == 8)[0]
        hydrogen = torch.where(numbers == 1)[0]
        oh_distances = distances[oxygen][:, hydrogen]
        covalent_ok &= bool(
            ((oh_distances <= COVALENT_OH_CUTOFF_A).sum(dim=1) == 2).all()
            and ((oh_distances <= COVALENT_OH_CUTOFF_A).sum(dim=0) == 1).all()
        )

        oo_distances = distances[oxygen][:, oxygen]
        adjacency = (oo_distances <= OXYGEN_CONNECTIVITY_CUTOFF_A) & (
            oo_distances > 0.0
        )
        reached = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            neighbors = torch.where(adjacency[current])[0].detach().cpu().tolist()
            for neighbor in neighbors:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        oxygen_network_ok &= len(reached) == 6

    return minimum_distance, covalent_ok, oxygen_network_ok


def _empty_rank_audit(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "systems_completed": 0,
        "unique_systems_completed": 0,
        "stage_1_completions": 0,
        "stage_2_completions": 0,
        "correctness_passed": True,
        "error": None,
        "max_energy_difference_ev": 0.0,
        "max_force_difference_ev_per_a": 0.0,
        "max_charge_difference_e": 0.0,
        "max_abs_net_charge_e": 0.0,
        "max_handoff_fmax_ev_per_a": 0.0,
        "min_interatomic_distance_a": None,
        "total_relaxation_steps_observed": 0,
        "max_relaxation_steps_observed": 0,
        "nvt_steps_verified": True,
        "nve_steps_verified": True,
        "covalent_oh_gate_passed": True,
        "oxygen_connectivity_gate_passed": True,
        "model_parity_batch_size": None,
        "model_parity_batches": 0,
        "campaign_ids_min": None,
        "campaign_ids_max": None,
        "_campaign_ids": [],
    }


def _audit_stored_outputs_in_chunks(
    *,
    completed_cpu: Batch,
    model: PipelineModelWrapper,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Re-evaluate stored outputs without creating a campaign-sized batch."""

    if batch_size <= 0:
        raise ValueError("audit batch size must be positive")

    max_energy_difference_ev = 0.0
    max_force_difference_ev_per_a = 0.0
    max_charge_difference_e = 0.0
    max_abs_net_charge_e = 0.0
    parity_batches = 0

    for start in range(0, completed_cpu.num_graphs, batch_size):
        stop = min(start + batch_size, completed_cpu.num_graphs)
        eval_batch = completed_cpu.index_select(slice(start, stop)).to(device)
        stored_energy = eval_batch.energy.clone()
        stored_forces = eval_batch.forces.clone()
        stored_charges = eval_batch.charges.clone()
        _drop_neighbor_fields(eval_batch)
        compute_neighbors(eval_batch, config=model.model_config.neighbor_config)
        outputs = model(eval_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        max_energy_difference_ev = max(
            max_energy_difference_ev,
            float(torch.max(torch.abs(outputs["energy"] - stored_energy)).item()),
        )
        max_force_difference_ev_per_a = max(
            max_force_difference_ev_per_a,
            float(torch.max(torch.abs(outputs["forces"] - stored_forces)).item()),
        )
        max_charge_difference_e = max(
            max_charge_difference_e,
            float(torch.max(torch.abs(outputs["charges"] - stored_charges)).item()),
        )

        graph_idx = eval_batch.batch_idx.to(torch.long)
        net_charge = torch.zeros(
            eval_batch.num_graphs,
            dtype=outputs["charges"].dtype,
            device=device,
        )
        net_charge.index_add_(0, graph_idx, outputs["charges"].reshape(-1))
        max_abs_net_charge_e = max(
            max_abs_net_charge_e,
            float(net_charge.abs().max().item()),
        )
        parity_batches += 1

    return {
        "max_energy_difference_ev": max_energy_difference_ev,
        "max_force_difference_ev_per_a": max_force_difference_ev_per_a,
        "max_charge_difference_e": max_charge_difference_e,
        "max_abs_net_charge_e": max_abs_net_charge_e,
        "model_parity_batch_size": batch_size,
        "model_parity_batches": parity_batches,
    }


def _audit_sink(
    *,
    rank: int,
    sink: HostMemory | None,
    model: PipelineModelWrapper,
    device: torch.device,
    nvt_steps: int,
    nve_steps: int,
    fmax_threshold: float,
    model_parity_batch_size: int,
) -> dict[str, Any]:
    audit = _empty_rank_audit(rank)
    if sink is None:
        return audit

    try:
        completed_cpu = sink.read()
        audit["systems_completed"] = completed_cpu.num_graphs
        campaign_ids = [
            int(value) for value in completed_cpu.campaign_id.reshape(-1).tolist()
        ]
        audit["_campaign_ids"] = campaign_ids
        audit["unique_systems_completed"] = len(set(campaign_ids))
        if campaign_ids:
            audit["campaign_ids_min"] = min(campaign_ids)
            audit["campaign_ids_max"] = max(campaign_ids)

        if not bool(
            torch.isfinite(completed_cpu.positions).all()
            and torch.isfinite(completed_cpu.velocities).all()
            and torch.isfinite(completed_cpu.energy).all()
            and torch.isfinite(completed_cpu.forces).all()
            and torch.isfinite(completed_cpu.charges).all()
        ):
            raise RuntimeError("final HostMemory data contain a non-finite value")
        if not bool((completed_cpu.num_nodes_per_graph == ATOMS_PER_SYSTEM).all()):
            raise RuntimeError("a completed result is not an 18-atom hexamer")

        velocity_initialized = completed_cpu.velocity_initialized.reshape(-1)
        audit["stage_1_completions"] = int((velocity_initialized == 1).sum().item())
        nvt_done = completed_cpu.nvt_steps_done.reshape(-1)
        nve_done = completed_cpu.nve_steps_done.reshape(-1)
        audit["nvt_steps_verified"] = bool((nvt_done == nvt_steps).all())
        audit["nve_steps_verified"] = bool((nve_done == nve_steps).all())
        audit["stage_2_completions"] = int(
            ((nvt_done == nvt_steps) & (nve_done == nve_steps)).sum().item()
        )
        relaxation_steps = completed_cpu.relaxation_steps.reshape(-1)
        audit["total_relaxation_steps_observed"] = int(
            relaxation_steps.sum().item()
        )
        audit["max_relaxation_steps_observed"] = int(relaxation_steps.max().item())
        audit["max_handoff_fmax_ev_per_a"] = float(
            completed_cpu.handoff_fmax.max().item()
        )
        if audit["max_handoff_fmax_ev_per_a"] > fmax_threshold + 1.0e-10:
            raise RuntimeError("a completed structure exceeded the FIRE2 force gate")

        minimum_distance, covalent_ok, oxygen_network_ok = _water_chemistry_metrics(
            completed_cpu
        )
        audit["min_interatomic_distance_a"] = minimum_distance
        audit["covalent_oh_gate_passed"] = covalent_ok
        audit["oxygen_connectivity_gate_passed"] = oxygen_network_ok
        if minimum_distance < MIN_INTERATOMIC_DISTANCE_A:
            raise RuntimeError("final structure contains overlapping atoms")
        if not covalent_ok:
            raise RuntimeError("final structure failed the covalent O-H gate")
        if not oxygen_network_ok:
            raise RuntimeError("final structure failed the oxygen-network gate")

        audit.update(
            _audit_stored_outputs_in_chunks(
                completed_cpu=completed_cpu,
                model=model,
                device=device,
                batch_size=model_parity_batch_size,
            )
        )

        if audit["max_energy_difference_ev"] > ENERGY_ATOL_EV:
            raise RuntimeError("stored energy does not match direct reevaluation")
        if audit["max_force_difference_ev_per_a"] > FORCE_ATOL_EV_PER_A:
            raise RuntimeError("stored forces do not match direct reevaluation")
        if audit["max_charge_difference_e"] > CHARGE_ATOL_E:
            raise RuntimeError("stored charges do not match direct reevaluation")
        if audit["max_abs_net_charge_e"] > MAX_ABS_NET_CHARGE_E:
            raise RuntimeError("predicted net charge exceeds the neutral-system gate")
        if audit["stage_1_completions"] != completed_cpu.num_graphs:
            raise RuntimeError("a result reached dynamics without velocity handoff")
        if not audit["nvt_steps_verified"] or not audit["nve_steps_verified"]:
            observed_nvt = sorted(set(int(value) for value in nvt_done.tolist()))
            observed_nve = sorted(set(int(value) for value in nve_done.tolist()))
            raise RuntimeError(
                "a result has an incorrect NVT or NVE step count: "
                f"NVT requested={nvt_steps}, observed={observed_nvt}; "
                f"NVE requested={nve_steps}, observed={observed_nve}"
            )
    except Exception as exc:
        audit["correctness_passed"] = False
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return audit


def _producer_set(
    slurm_producer: str = "scripts/slurm_part1_distributed_campaign.sbatch",
) -> dict[str, str]:
    candidate = Path(slurm_producer)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Slurm producer must be a safe repository-relative path")
    campaign_slurm = "scripts/slurm_part1_distributed_campaign.sbatch"
    producer_files = tuple(
        slurm_producer if relative_path == campaign_slurm else relative_path
        for relative_path in PRODUCER_FILES
    )
    result: dict[str, str] = {}
    for relative_path in producer_files:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"producer file not found: {relative_path}")
        result[relative_path] = _sha256(path)
    return result


def _workload_record(
    args: argparse.Namespace,
    campaign_sha256: str,
) -> dict[str, Any]:
    structure_builder = PART_DIR / "aux" / "structures.py"
    return {
        "campaign_definition_sha256": campaign_sha256,
        "source_structure": "generated cyclic water hexamer",
        "structure_builder_file": structure_builder.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "structure_builder_sha256": _sha256(structure_builder),
        "systems_total": args.systems,
        "atoms_per_system": ATOMS_PER_SYSTEM,
        "batch_size": args.batch_size,
        "campaign_seed": CAMPAIGN_SEED,
        "perturbation_description": (
            "Eight O-O levels from 2.72 to 2.86 A in 0.02 A steps; one "
            "seeded generator adds isotropic 0.025 A Gaussian coordinate noise "
            "in global campaign-ID order, then removes the mean displacement"
        ),
        "fire_fmax_ev_per_a": args.fire_fmax,
        "nvt_steps": args.nvt_steps,
        "nve_steps": args.nve_steps,
        "dt_fs": args.dt_fs,
        "temperature_k": args.temperature_k,
        "friction_per_fs": args.friction_per_fs,
        "velocity_seed_rule": "910000 + campaign_id",
        "pipeline_partition_rule": (
            "4 GPUs: campaign_id % 2; 1 or 2 GPUs: all campaign IDs"
        ),
        "pipeline_pair_boundaries": {
            route: [list(pair) for pair in pairs]
            for route, pairs in ROUTE_PAIR_BOUNDARIES.items()
        },
        "comm_mode": args.comm_mode,
        "stage_names": ["FIRE2", "NVTLangevin", "NVE"],
    }


def _correctness_tolerances(args: argparse.Namespace) -> dict[str, float]:
    return {
        "energy_atol_ev": ENERGY_ATOL_EV,
        "force_atol_ev_per_a": FORCE_ATOL_EV_PER_A,
        "charge_atol_e": CHARGE_ATOL_E,
        "max_abs_net_charge_e": MAX_ABS_NET_CHARGE_E,
        "max_handoff_fmax_ev_per_a": args.fire_fmax,
        "min_interatomic_distance_a": MIN_INTERATOMIC_DISTANCE_A,
        "covalent_oh_cutoff_a": COVALENT_OH_CUTOFF_A,
        "oxygen_connectivity_cutoff_a": OXYGEN_CONNECTIVITY_CUTOFF_A,
    }


def _model_record(model_build: ModelBuild) -> dict[str, Any]:
    return {
        "checkpoint_source": os.environ.get(
            "ALCHEMI_AIMNET_CHECKPOINT", AIMNET_CHECKPOINT
        ),
        "checkpoint_sha256": model_build.checkpoint_sha256,
        "d3_parameter_sha256": model_build.d3_parameter_sha256,
        "d3_bj_parameters": dict(model_build.d3_bj_parameters),
        "components": [
            "AIMNet2 B97-3c residual",
            "finite all-pairs Coulomb",
            "pairwise D3(BJ)",
        ],
        "dtype": (
            "float32 positions, velocities, forces, AIMNet, and Coulomb pair "
            "math; float64 Coulomb energy accumulation"
        ),
        "eager": True,
    }


def _combine_rank_audits(
    audits: list[dict[str, Any]],
    systems: int,
) -> dict[str, Any]:
    all_ids = [
        campaign_id
        for audit in audits
        for campaign_id in audit.get("_campaign_ids", [])
    ]
    counts = Counter(all_ids)
    expected = set(range(systems))
    observed = set(all_ids)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(
        campaign_id for campaign_id, count in counts.items() if count > 1
    )
    completed = sum(int(audit["systems_completed"]) for audit in audits)

    sink_audits = [audit for audit in audits if audit["systems_completed"] > 0]
    minimum_distances = [
        float(audit["min_interatomic_distance_a"])
        for audit in sink_audits
        if audit["min_interatomic_distance_a"] is not None
    ]
    correctness_passed = (
        all(bool(audit["correctness_passed"]) for audit in audits)
        and completed == systems
        and len(observed) == systems
        and not missing
        and not unexpected
        and not duplicates
    )
    public_audits = [
        {key: value for key, value in audit.items() if not key.startswith("_")}
        for audit in audits
    ]
    return {
        "systems_completed": completed,
        "unique_systems_completed": len(observed),
        "missing_systems": missing,
        "duplicate_systems": duplicates,
        "unexpected_systems": unexpected,
        "stage_1_completions": sum(
            int(audit["stage_1_completions"]) for audit in audits
        ),
        "stage_2_completions": sum(
            int(audit["stage_2_completions"]) for audit in audits
        ),
        "correctness_passed": correctness_passed,
        "max_energy_difference_ev": max(
            float(audit["max_energy_difference_ev"]) for audit in audits
        ),
        "max_force_difference_ev_per_a": max(
            float(audit["max_force_difference_ev_per_a"]) for audit in audits
        ),
        "max_charge_difference_e": max(
            float(audit["max_charge_difference_e"]) for audit in audits
        ),
        "max_abs_net_charge_e": max(
            float(audit["max_abs_net_charge_e"]) for audit in audits
        ),
        "max_handoff_fmax_ev_per_a": max(
            float(audit["max_handoff_fmax_ev_per_a"]) for audit in audits
        ),
        "min_interatomic_distance_a": (
            min(minimum_distances) if minimum_distances else None
        ),
        "total_relaxation_steps_observed": sum(
            int(audit["total_relaxation_steps_observed"]) for audit in audits
        ),
        "max_relaxation_steps_observed": max(
            int(audit["max_relaxation_steps_observed"]) for audit in audits
        ),
        "nvt_steps_verified": all(
            bool(audit["nvt_steps_verified"]) for audit in audits
        ),
        "nve_steps_verified": all(
            bool(audit["nve_steps_verified"]) for audit in audits
        ),
        "covalent_oh_gate_passed": all(
            bool(audit["covalent_oh_gate_passed"]) for audit in audits
        ),
        "oxygen_connectivity_gate_passed": all(
            bool(audit["oxygen_connectivity_gate_passed"]) for audit in audits
        ),
        "rank_audits": public_audits,
    }


def _aggregate_failure_error(
    summary: Mapping[str, Any], systems: int
) -> str:
    """Explain a failed combined audit even when every rank-local audit passed."""

    errors: list[str] = []
    for audit in summary["rank_audits"]:
        message = audit.get("error")
        if isinstance(message, str) and message.strip() and message not in errors:
            errors.append(message)
    completed = int(summary["systems_completed"])
    unique = int(summary["unique_systems_completed"])
    if completed != systems:
        errors.append(f"completed {completed} of {systems} systems")
    if unique != systems:
        errors.append(f"received {unique} unique campaign IDs; expected {systems}")
    for field, label in (
        ("missing_systems", "missing campaign IDs"),
        ("duplicate_systems", "duplicate campaign IDs"),
        ("unexpected_systems", "unexpected campaign IDs"),
    ):
        values = summary[field]
        if values:
            errors.append(f"{label}: {values}")
    if not errors:
        errors.append("campaign correctness audit failed")
    return "; ".join(errors)


def _run_balance_probe(
    *,
    args: argparse.Namespace,
    model_build: ModelBuild,
    structures: list[Atoms],
    campaign_sha256: str,
    device: torch.device,
    runtime_identity: tuple[str, str, str],
) -> None:
    if dist.get_world_size() != 1:
        raise ValueError("balance probe requires one GPU")
    expected_systems = BALANCE_PROBE_BATCHES * args.batch_size
    if args.systems != expected_systems:
        raise ValueError(
            "balance probe requires four full batches: "
            f"systems={expected_systems} for batch_size={args.batch_size}"
        )
    if args.batch_size % 16:
        raise ValueError(
            "balance-probe batch size must be a multiple of 16 so every "
            "O-O level is represented evenly"
        )

    batch_timings: list[dict[str, Any]] = []
    relaxation_times: list[float] = []
    dynamics_times: list[float] = []

    for batch_index in range(BALANCE_PROBE_BATCHES):
        start_id = batch_index * args.batch_size
        stop_id = start_id + args.batch_size
        campaign_ids = tuple(range(start_id, stop_id))
        level_counts = Counter((campaign_id // 2) % 8 for campaign_id in campaign_ids)
        counts_by_level = [level_counts[level] for level in range(8)]
        expected_level_count = args.batch_size // 8
        if counts_by_level != [expected_level_count] * 8:
            raise RuntimeError("balance-probe batch does not balance the O-O levels")

        dataset = WaterHexamerDataset(structures, campaign_ids, device)
        batch = Batch.from_data_list(
            [dataset[index][0] for index in range(args.batch_size)],
            device=device,
        )

        fire = _make_fire(
            model_build.model,
            fmax_threshold=args.fire_fmax,
            temperature_k=args.temperature_k,
            source_status=0,
        )
        relaxation = FusedStage(
            sub_stages=[(0, fire)],
            device_type="cuda",
        )
        _attach_model_hooks(relaxation, model_build.model)
        with relaxation:
            torch.cuda.synchronize(device)
            start = perf_counter()
            batch = relaxation.run(batch)
            torch.cuda.synchronize(device)
            relaxation_elapsed_s = perf_counter() - start
        if batch is None:
            raise RuntimeError("balance-probe relaxation returned no batch")
        if not bool((batch.velocity_initialized.reshape(-1) == 1).all()):
            raise RuntimeError("balance probe did not initialize every velocity")
        if float(batch.handoff_fmax.max().item()) > args.fire_fmax + 1.0e-10:
            raise RuntimeError("balance probe crossed the FIRE2 force gate")
        relaxation_step_total = int(batch.relaxation_steps.sum().item())
        relaxation_step_max = int(batch.relaxation_steps.max().item())

        batch.status.zero_()
        batch.nvt_steps_done.zero_()
        batch.nve_steps_done.zero_()
        nvt = _make_nvt(
            model_build.model,
            n_steps=args.nvt_steps,
            dt_fs=args.dt_fs,
            temperature_k=args.temperature_k,
            friction_per_fs=args.friction_per_fs,
        )
        nve = _make_nve(
            model_build.model,
            n_steps=args.nve_steps,
            dt_fs=args.dt_fs,
        )
        dynamics = FusedStage(
            sub_stages=[(0, nvt), (1, nve)],
            device_type="cuda",
        )
        dynamics.register_fused_hook(
            StageStepCounterHook({0: "nvt_steps_done", 1: "nve_steps_done"})
        )
        _attach_model_hooks(dynamics, model_build.model)
        with dynamics:
            torch.cuda.synchronize(device)
            start = perf_counter()
            # The relaxed batch already carries current forces, as it does at
            # the pipeline handoff, so step directly without an extra forward.
            while not dynamics.all_complete(batch, dynamics.exit_status):
                batch, _ = dynamics.step(batch)
            torch.cuda.synchronize(device)
            dynamics_elapsed_s = perf_counter() - start
        if not bool((batch.nvt_steps_done == args.nvt_steps).all()):
            observed = sorted(
                set(batch.nvt_steps_done.reshape(-1).detach().cpu().tolist())
            )
            raise RuntimeError(
                "balance probe did not complete the requested NVT steps: "
                f"requested={args.nvt_steps}, observed={observed}"
            )
        if not bool((batch.nve_steps_done == args.nve_steps).all()):
            observed = sorted(
                set(batch.nve_steps_done.reshape(-1).detach().cpu().tolist())
            )
            raise RuntimeError(
                "balance probe did not complete the requested NVE steps: "
                f"requested={args.nve_steps}, observed={observed}"
            )

        relaxation_times.append(relaxation_elapsed_s)
        dynamics_times.append(dynamics_elapsed_s)
        batch_timings.append(
            {
                "batch_index": batch_index,
                "campaign_id_start": start_id,
                "campaign_id_stop_exclusive": stop_id,
                "systems": args.batch_size,
                "oo_level_counts": counts_by_level,
                "relaxation_step_total": relaxation_step_total,
                "relaxation_step_max": relaxation_step_max,
                "relaxation_elapsed_s": relaxation_elapsed_s,
                "dynamics_elapsed_s": dynamics_elapsed_s,
            }
        )

    relaxation_median_s = float(np.median(relaxation_times))
    dynamics_median_s = float(np.median(dynamics_times))
    steady_relaxation_median_s = float(np.median(relaxation_times[1:]))
    steady_dynamics_median_s = float(np.median(dynamics_times[1:]))
    slower_stage = max(relaxation_median_s, dynamics_median_s)
    record = {
        "schema": BALANCE_PROBE_SCHEMA,
        "status": "complete",
        "success": True,
        "timestamp_utc": _utc_now(),
        "run_id": f"{os.environ.get('SLURM_JOB_ID', 'interactive')}-balance-probe",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "route": "balance_probe",
        "systems": args.systems,
        "batch_count": BALANCE_PROBE_BATCHES,
        "batch_size": args.batch_size,
        "campaign_id_start": 0,
        "campaign_id_stop_exclusive": args.systems,
        "batch_timings": batch_timings,
        "relaxation_median_s": relaxation_median_s,
        "dynamics_median_s": dynamics_median_s,
        "steady_relaxation_median_s": steady_relaxation_median_s,
        "steady_dynamics_median_s": steady_dynamics_median_s,
        "projected_two_stage_speedup_ceiling": (
            (relaxation_median_s + dynamics_median_s) / slower_stage
        ),
        "gpu_name": runtime_identity[0],
        "torch_version": runtime_identity[1],
        "python_version": runtime_identity[2],
        "toolkit_core_commit": CORE_COMMIT,
        "toolkit_core_branch": CORE_BRANCH,
        "toolkit_core_clean": True,
        "toolkit_core_version": CORE_VERSION,
        "toolkit_ops_commit": OPS_COMMIT,
        "toolkit_ops_version": OPS_VERSION,
        "producer_set": _producer_set(args.slurm_producer),
        "repository_commit": _git_head(REPOSITORY_ROOT),
        "model": _model_record(model_build),
        "workload": _workload_record(args, campaign_sha256),
        "timing_boundary": (
            f"Four independent {args.batch_size}-system batches covering global "
            f"campaign IDs 0 through {args.systems - 1} are timed separately "
            "through FIRE2 and then fused NVT + NVE. Reported stage times include "
            "the median of all four batches and a steady-state median that excludes "
            "the first batch's one-time compiler/setup cost. "
            "Model loading, warm-up, input construction, stage construction, "
            "context setup, and validation are outside the measurements."
        ),
    }
    _write_json_once(args.output, record)
    print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=tuple(ROUTE_TOPOLOGY), required=True)
    parser.add_argument(
        "--purpose",
        choices=("campaign", "smoke", "tuning", "balance-probe"),
        default="campaign",
    )
    parser.add_argument("--systems", type=int, default=DEFAULT_SYSTEMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--fire-fmax", type=float, default=DEFAULT_FIRE_FMAX_EV_PER_A)
    parser.add_argument("--nvt-steps", type=int, required=True)
    parser.add_argument("--nve-steps", type=int, required=True)
    parser.add_argument("--dt-fs", type=float, default=DEFAULT_DT_FS)
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument(
        "--friction-per-fs", type=float, default=DEFAULT_FRICTION_PER_FS
    )
    parser.add_argument(
        "--comm-mode",
        choices=("sync", "async_recv", "fully_async"),
        default=DEFAULT_COMM_MODE,
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--slurm-producer",
        default="scripts/slurm_part1_distributed_campaign.sbatch",
        help="Repository-relative Slurm script that launched this record.",
    )
    parser.add_argument(
        "--d3-parameter-file",
        type=Path,
        default=Path(
            os.environ.get(
                "ALCHEMI_D3_PARAM_FILE",
                Path.home() / ".cache" / "nvalchemiops" / "dftd3_parameters.pt",
            )
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    expected_world_size = int(ROUTE_TOPOLOGY[args.route]["world_size"])
    if world_size != expected_world_size:
        raise ValueError(
            f"route {args.route} requires {expected_world_size} ranks, got {world_size}"
        )
    if args.systems <= 0 or args.batch_size <= 0:
        raise ValueError("systems and batch size must be positive")
    if args.nvt_steps <= 0 or args.nve_steps <= 0:
        raise ValueError("NVT and NVE step counts must be explicitly positive")
    if args.fire_fmax <= 0.0 or args.dt_fs <= 0.0:
        raise ValueError("FIRE force threshold and timestep must be positive")
    if args.temperature_k <= 0.0 or args.friction_per_fs <= 0.0:
        raise ValueError("temperature and friction must be positive")
    if args.repeat < 0:
        raise ValueError("repeat must be non-negative")
    if args.purpose == "campaign":
        fixed = {
            "systems": (args.systems, DEFAULT_SYSTEMS),
            "batch size": (args.batch_size, DEFAULT_BATCH_SIZE),
            "FIRE force threshold": (
                args.fire_fmax,
                DEFAULT_FIRE_FMAX_EV_PER_A,
            ),
            "timestep": (args.dt_fs, DEFAULT_DT_FS),
            "temperature": (args.temperature_k, DEFAULT_TEMPERATURE_K),
            "friction": (args.friction_per_fs, DEFAULT_FRICTION_PER_FS),
        }
        changed = [name for name, (observed, expected) in fixed.items() if observed != expected]
        if changed:
            raise ValueError(
                "publishable campaign changed fixed inputs: " + ", ".join(changed)
            )


def main() -> int:
    args = _parse_args()
    core_root = Path(os.environ["ALCHEMI_TOOLKIT_CORE_ROOT"]).resolve()
    _validate_runtime(core_root)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("campaign requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        _validate_args(args, world_size)
        runtime_identity = (
            torch.cuda.get_device_name(local_rank),
            torch.__version__,
            platform.python_version(),
        )
        runtime_identities: list[tuple[str, str, str] | None] = [None] * world_size
        dist.all_gather_object(runtime_identities, runtime_identity)
        if any(identity != runtime_identity for identity in runtime_identities):
            raise RuntimeError(
                "ranks disagree on GPU model, Torch version, or Python version"
            )
        if "H100" not in runtime_identity[0].upper():
            raise RuntimeError(f"campaign requires H100 GPUs, found {runtime_identity[0]}")

        torch.manual_seed(CAMPAIGN_SEED)
        np.random.seed(CAMPAIGN_SEED)
        structures, campaign_sha256 = _make_campaign_structures(args.systems)
        model_build = _build_model(device, args.d3_parameter_file.resolve())
        _warm_model(model_build.model, structures, args.batch_size, device)

        if args.purpose == "balance-probe":
            if rank == 0:
                _run_balance_probe(
                    args=args,
                    model_build=model_build,
                    structures=structures,
                    campaign_sha256=campaign_sha256,
                    device=device,
                    runtime_identity=runtime_identity,
                )
            return 0

        if args.route == "fused_1gpu":
            build = _build_fused_workflow(
                model=model_build.model,
                structures=structures,
                systems=args.systems,
                batch_size=args.batch_size,
                fmax_threshold=args.fire_fmax,
                nvt_steps=args.nvt_steps,
                nve_steps=args.nve_steps,
                dt_fs=args.dt_fs,
                temperature_k=args.temperature_k,
                friction_per_fs=args.friction_per_fs,
                device=device,
            )
            sinks = build.sinks
            pipeline_count = build.pipeline_count
            torch.cuda.reset_peak_memory_stats(device)
            # The context creates the stage stream. The public run includes
            # hook setup and the initial force evaluation in the measurement.
            with build.workflow:
                torch.cuda.synchronize(device)
                dist.barrier()
                start = perf_counter()
                build.workflow.run(build.initial_batch)
                torch.cuda.synchronize(device)
                local_elapsed_s = perf_counter() - start
            local_stage_type = type(build.workflow).__name__
            local_stage_step_count = int(build.workflow.step_count)
        else:
            pair_count = len(pair_boundaries_for_world_size(world_size))
            campaign_ids_by_pair = _partition_campaign_ids(args.systems, pair_count)
            sinks = _build_distributed_sinks(args.systems, world_size)
            pipeline_count = pair_count
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            dist.barrier()
            start = perf_counter()
            distributed_build = _build_distributed_workflow(
                model=model_build.model,
                structures=structures,
                campaign_ids_by_pair=campaign_ids_by_pair,
                sinks=sinks,
                batch_size=args.batch_size,
                fmax_threshold=args.fire_fmax,
                nvt_steps=args.nvt_steps,
                nve_steps=args.nve_steps,
                dt_fs=args.dt_fs,
                temperature_k=args.temperature_k,
                friction_per_fs=args.friction_per_fs,
                comm_mode=args.comm_mode,
                world_size=world_size,
                device=device,
            )
            with distributed_build.workflow:
                distributed_build.workflow.run()
            torch.cuda.synchronize(device)
            local_elapsed_s = perf_counter() - start
            local_stage = distributed_build.workflow.local_stage
            local_stage_type = type(local_stage).__name__
            local_stage_step_count = int(local_stage.step_count)

        local_runtime = {
            "rank": rank,
            "stage_type": local_stage_type,
            "stage_step_count": local_stage_step_count,
            "elapsed_s": local_elapsed_s,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        rank_runtime_metrics: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_runtime_metrics, local_runtime)

        elapsed = torch.tensor(local_elapsed_s, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(device),
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

        local_audit = _audit_sink(
            rank=rank,
            sink=sinks.get(rank),
            model=model_build.model,
            device=device,
            nvt_steps=args.nvt_steps,
            nve_steps=args.nve_steps,
            fmax_threshold=args.fire_fmax,
            model_parity_batch_size=args.batch_size,
        )
        rank_audits: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_audits, local_audit)

        if rank == 0:
            complete_audits = [audit for audit in rank_audits if audit is not None]
            summary = _combine_rank_audits(complete_audits, args.systems)
            elapsed_s = float(elapsed.item())
            success = bool(summary["correctness_passed"])
            failure_error = (
                None
                if success
                else _aggregate_failure_error(summary, args.systems)
            )
            record: dict[str, Any] = {
                "schema": RUN_SCHEMA,
                "status": "complete" if success else "failed",
                "success": success,
                "timestamp_utc": _utc_now(),
                "run_id": _run_id(args),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "route": args.route,
                "purpose": args.purpose,
                "repeat": args.repeat,
                "error_type": None if success else "CorrectnessAuditError",
                "error": failure_error,
                "nodes": int(os.environ.get("SLURM_NNODES", world_size)),
                "gpu_count": world_size,
                "rank_count": world_size,
                "pipeline_count": pipeline_count,
                "systems_requested": args.systems,
                **summary,
                "elapsed_s": elapsed_s,
                "systems_per_s": summary["systems_completed"] / elapsed_s,
                "peak_memory_bytes_max_rank": int(peak_memory.item()),
                "rank_runtime_metrics": [
                    metric
                    for metric in rank_runtime_metrics
                    if metric is not None
                ],
                "gpu_name": runtime_identity[0],
                "backend": dist.get_backend(),
                "hostname_rank0": platform.node(),
                "torch_version": runtime_identity[1],
                "python_version": runtime_identity[2],
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "toolkit_core_commit": CORE_COMMIT,
                "toolkit_core_branch": CORE_BRANCH,
                "toolkit_core_clean": True,
                "toolkit_core_version": CORE_VERSION,
                "toolkit_ops_commit": OPS_COMMIT,
                "toolkit_ops_version": OPS_VERSION,
                "producer_set": _producer_set(args.slurm_producer),
                "repository_commit": _git_head(REPOSITORY_ROOT),
                "model": _model_record(model_build),
                "workload": _workload_record(args, campaign_sha256),
                "timing_boundary": TIMING_BOUNDARY,
                "correctness_checks": (
                    list(CORRECTNESS_CHECKS)
                    if args.systems == DEFAULT_SYSTEMS
                    else [
                        f"all {args.systems} campaign IDs appear exactly once",
                        *CORRECTNESS_CHECKS[1:],
                    ]
                ),
                "correctness_tolerances": _correctness_tolerances(args),
            }
            _write_json_once(args.output, record)
            print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)

        failure = torch.tensor(
            int(not local_audit["correctness_passed"]),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if int(failure.item()):
            raise RuntimeError("campaign correctness audit failed")
        if rank == 0:
            global_success = _combine_rank_audits(
                [audit for audit in rank_audits if audit is not None], args.systems
            )["correctness_passed"]
        else:
            global_success = None
        result = [global_success]
        dist.broadcast_object_list(result, src=0)
        if not bool(result[0]):
            raise RuntimeError("campaign identity or completion audit failed")
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
