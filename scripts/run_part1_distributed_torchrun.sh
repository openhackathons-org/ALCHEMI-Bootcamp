#!/usr/bin/env bash
set -euo pipefail

: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV}"
: "${ALCHEMI_MASTER_ADDR:?set ALCHEMI_MASTER_ADDR}"
: "${ALCHEMI_MASTER_PORT:?set ALCHEMI_MASTER_PORT}"
: "${ALCHEMI_TOOLKIT_CORE_ROOT:?set ALCHEMI_TOOLKIT_CORE_ROOT}"

# Keep Python-level ordering deterministic in the campaign record and any
# third-party model code used by all ranks.
export PYTHONHASHSEED=0
export PYTHONPATH="$ALCHEMI_TOOLKIT_CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ALCHEMI_MAIN_ENV/bin/torchrun" \
  --nnodes "$SLURM_NNODES" \
  --nproc-per-node 1 \
  --node-rank "$SLURM_NODEID" \
  --master-addr "$ALCHEMI_MASTER_ADDR" \
  --master-port "$ALCHEMI_MASTER_PORT" \
  "$@"
