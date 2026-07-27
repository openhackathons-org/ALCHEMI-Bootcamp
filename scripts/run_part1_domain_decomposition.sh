#!/usr/bin/env bash
set -euo pipefail

# Do not write import bytecode into the staged tutorial or Toolkit checkouts.
export PYTHONDONTWRITEBYTECODE=1

# Start one fresh torchrun process for a Part 1 domain-decomposition case.
#
# Slurm mode expects one task and one GPU per node. Local mode expects
# ALCHEMI_DOMAIN_GPUS to be the number of visible GPUs on one machine.

: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to the verified Conda base}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to the verified Python package layer}"
: "${ALCHEMI_TOOLKIT_CORE_ROOT:?set ALCHEMI_TOOLKIT_CORE_ROOT to the exact Toolkit Core checkout}"
: "${ALCHEMI_TOOLKIT_OPS_ROOT:?set ALCHEMI_TOOLKIT_OPS_ROOT to the exact Toolkit-Ops checkout}"

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
CORE_ROOT="$(realpath "$ALCHEMI_TOOLKIT_CORE_ROOT")"
OPS_ROOT="$(realpath "$ALCHEMI_TOOLKIT_OPS_ROOT")"
test -d "$CORE_ROOT"
test -d "$OPS_ROOT"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_TOOLKIT_OPS_ROOT="$OPS_ROOT"
export PYTHONPATH="$CORE_ROOT:$OPS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

READY_FILE="$ALCHEMI_PYTHON_OVERLAY/.part1-ready.json"
TORCHRUN="$ALCHEMI_PYTHON_OVERLAY/bin/torchrun"
test -s "$READY_FILE"
test -x "$TORCHRUN"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${SLURM_NNODES:?Slurm did not provide SLURM_NNODES}"
  : "${SLURM_NODEID:?Slurm did not provide SLURM_NODEID}"
  : "${ALCHEMI_MASTER_ADDR:?set ALCHEMI_MASTER_ADDR on the allocation}"
  : "${ALCHEMI_MASTER_PORT:?set ALCHEMI_MASTER_PORT on the allocation}"

  exec "$TORCHRUN" \
    --nnodes "$SLURM_NNODES" \
    --nproc-per-node 1 \
    --node-rank "$SLURM_NODEID" \
    --master-addr "$ALCHEMI_MASTER_ADDR" \
    --master-port "$ALCHEMI_MASTER_PORT" \
    "$@"
fi

: "${ALCHEMI_DOMAIN_GPUS:?set ALCHEMI_DOMAIN_GPUS for the local launch}"

exec "$TORCHRUN" \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "$ALCHEMI_DOMAIN_GPUS" \
  "$@"
