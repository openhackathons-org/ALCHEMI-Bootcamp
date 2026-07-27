"""Immutable inputs shared by the Part 1 campaign producers."""

from __future__ import annotations


RUN_SCHEMA = "alchemi.pipeline-campaign-run.v2"
BALANCE_PROBE_SCHEMA = "alchemi.pipeline-campaign-balance-probe.v1"

CORE_BRANCH = "0.2.0-rc"
CORE_VERSION = "0.2.0"
CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
OPS_VERSION = "0.4.0"
OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"

CURRENT_RC_TIMING_STATUS = (
    "NOT REPORTED: Toolkit Core 0.2.0-rc at commit "
    f"{CORE_COMMIT} fixes reusable-buffer capacity and waits for an asynchronous "
    "send before reusing its storage. Batch.put still copies only float32 "
    "segmented fields, so integer fields including atomic_numbers are not "
    "preserved. The classic DistributedPipeline also makes every rank join a "
    "global completion check (all_reduce) after every iteration. The campaign "
    "remains blocked until the complete transfer and stage-overlap checks pass."
)

# The prerecorded run keeps one SizeAwareSampler and one DistributedPipeline
# alive for each complete route. The two-GPU route gives all 8192 systems to one
# sampler. The four-GPU route gives 4096 systems to each upstream sampler. Each
# sampler admits at most 512 systems at a time while the same stages, transfer
# buffers, and pipeline remain active until the full partition is complete.
# Repeated transfers make the H100 buffer-reuse preflight mandatory.
# Run the tuning job on H100 against this exact commit before the campaign.
DEFAULT_BATCH_SIZE = 512
BATCHES_PER_PIPELINE_AT_MAX_SCALE = 8
MAX_PIPELINES = 2
DEFAULT_SYSTEMS = (
    DEFAULT_BATCH_SIZE * BATCHES_PER_PIPELINE_AT_MAX_SCALE * MAX_PIPELINES
)
ATOMS_PER_SYSTEM = 18
BALANCE_PROBE_BATCHES = 4
BALANCE_PROBE_SYSTEMS = BALANCE_PROBE_BATCHES * DEFAULT_BATCH_SIZE
CAMPAIGN_REPEATS = 5
CAMPAIGN_SEED = 20260714
VELOCITY_SEED = 910000
DEFAULT_FIRE_FMAX_EV_PER_A = 0.01
DEFAULT_DT_FS = 0.5
DEFAULT_TEMPERATURE_K = 75.0
DEFAULT_FRICTION_PER_FS = 0.01
DEFAULT_COMM_MODE = "async_recv"

ENERGY_ATOL_EV = 5.0e-5
FORCE_ATOL_EV_PER_A = 5.0e-5
CHARGE_ATOL_E = 5.0e-6
MAX_ABS_NET_CHARGE_E = 5.0e-5
MIN_INTERATOMIC_DISTANCE_A = 0.55
COVALENT_OH_CUTOFF_A = 1.25
OXYGEN_CONNECTIVITY_CUTOFF_A = 4.0

AIMNET_CHECKPOINT = "aimnet2-b973c-2025-d3_0"
EXPECTED_AIMNET_CHECKPOINT_SHA256 = (
    "043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5"
)
EXPECTED_D3_PARAMETER_SHA256 = (
    "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
)
D3_CUTOFF_A = 15.0
NEIGHBOR_SKIN_A = 0.5

ROUTE_TOPOLOGY = {
    "fused_1gpu": {
        "label": "1 GPU - fused stages",
        "world_size": 1,
        "pipeline_count": 0,
    },
    "pipeline_2gpu": {
        "label": "2 GPUs - one pipeline",
        "world_size": 2,
        "pipeline_count": 1,
    },
    "pipeline_4gpu": {
        "label": "4 GPUs - two pipelines",
        "world_size": 4,
        "pipeline_count": 2,
    },
}

# ``DistributedPipeline`` wires a stage only when ``prior_rank`` or
# ``next_rank`` names that neighbor.  Keep the two four-rank streams explicit:
# rank 1 must never hand a batch to rank 2.
ROUTE_PAIR_BOUNDARIES = {
    "fused_1gpu": (),
    "pipeline_2gpu": ((0, 1),),
    "pipeline_4gpu": ((0, 1), (2, 3)),
}

TIMING_BOUNDARY = (
    "Each rank synchronizes its CUDA work, then all ranks meet at a barrier "
    "immediately before the campaign timer starts. Each rank synchronizes its "
    "CUDA work and stops its local timer immediately after the campaign. The "
    "reported wall time is the largest rank-local elapsed time; the reduction "
    "used to select it is outside the measurement. "
    "The measurement includes model evaluations, "
    "relaxation, velocity initialization, NVT and NVE updates, batch replacement, "
    "transfers between stages, coordination, and the final transfer to HostMemory. "
    "For distributed routes it includes constructing the stages and one "
    "DistributedPipeline, one public run, and one setup and transfer-template "
    "exchange. The same pipeline remains active while SizeAwareSampler admits "
    "each route's complete campaign partition in batches of at most 512 systems. "
    "It excludes process startup, model and checkpoint loading, input generation, "
    "model warm-up, the one-GPU FusedStage construction and context setup, and the "
    "correctness audit."
)

CORRECTNESS_CHECKS = (
    f"all {DEFAULT_SYSTEMS} campaign IDs appear exactly once",
    "every result is an 18-atom neutral water hexamer with the expected elements",
    "positions, velocities, energies, forces, and predicted charges are finite",
    "the stored energy, forces, and charges match a direct final model evaluation",
    "every structure met the FIRE2 force threshold before velocity initialization",
    "every structure completed the requested NVT and NVE step counts",
    "covalent O-H connectivity remains intact and no atoms overlap",
)

PRODUCER_FILES = (
    "scripts/benchmark_part1_distributed_campaign.py",
    "scripts/part1_distributed_campaign_contract.py",
    "scripts/record_part1_campaign_failure.py",
    "scripts/run_part1_distributed_torchrun.sh",
    "scripts/slurm_part1_distributed_campaign.sbatch",
    "part-1-scalable-atomistic-workflows/aux/artifacts.py",
    "part-1-scalable-atomistic-workflows/aux/checkpoint.py",
    "part-1-scalable-atomistic-workflows/aux/electrostatics.py",
    "part-1-scalable-atomistic-workflows/aux/hooks.py",
    "part-1-scalable-atomistic-workflows/aux/runtime.py",
    "part-1-scalable-atomistic-workflows/aux/structures.py",
)


def route_for_world_size(world_size: int) -> str:
    """Return the publishable route ID for a supported allocation."""

    matches = [
        route
        for route, values in ROUTE_TOPOLOGY.items()
        if values["world_size"] == world_size
    ]
    if len(matches) != 1:
        raise ValueError("world size must be 1, 2, or 4")
    return matches[0]


def pair_boundaries_for_world_size(world_size: int) -> tuple[tuple[int, int], ...]:
    """Return the explicit upstream/downstream rank pairs for a route."""

    return ROUTE_PAIR_BOUNDARIES[route_for_world_size(world_size)]


__all__ = [
    "AIMNET_CHECKPOINT",
    "ATOMS_PER_SYSTEM",
    "BALANCE_PROBE_BATCHES",
    "BALANCE_PROBE_SCHEMA",
    "BALANCE_PROBE_SYSTEMS",
    "CAMPAIGN_SEED",
    "CAMPAIGN_REPEATS",
    "CORE_BRANCH",
    "CORE_COMMIT",
    "CORE_VERSION",
    "CORRECTNESS_CHECKS",
    "CURRENT_RC_TIMING_STATUS",
    "D3_CUTOFF_A",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_COMM_MODE",
    "DEFAULT_DT_FS",
    "DEFAULT_FIRE_FMAX_EV_PER_A",
    "DEFAULT_FRICTION_PER_FS",
    "DEFAULT_SYSTEMS",
    "DEFAULT_TEMPERATURE_K",
    "EXPECTED_AIMNET_CHECKPOINT_SHA256",
    "EXPECTED_D3_PARAMETER_SHA256",
    "ENERGY_ATOL_EV",
    "FORCE_ATOL_EV_PER_A",
    "BATCHES_PER_PIPELINE_AT_MAX_SCALE",
    "CHARGE_ATOL_E",
    "MAX_ABS_NET_CHARGE_E",
    "MAX_PIPELINES",
    "MIN_INTERATOMIC_DISTANCE_A",
    "COVALENT_OH_CUTOFF_A",
    "OXYGEN_CONNECTIVITY_CUTOFF_A",
    "NEIGHBOR_SKIN_A",
    "OPS_COMMIT",
    "OPS_VERSION",
    "PRODUCER_FILES",
    "ROUTE_PAIR_BOUNDARIES",
    "ROUTE_TOPOLOGY",
    "RUN_SCHEMA",
    "TIMING_BOUNDARY",
    "VELOCITY_SEED",
    "pair_boundaries_for_world_size",
    "route_for_world_size",
]
