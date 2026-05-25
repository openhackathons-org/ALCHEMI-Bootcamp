#!/usr/bin/env bash
# Fan out the SLC temperature sweep across 4 GPUs.
#
# Launches four `slc.py` processes in parallel, each pinned to one GPU via
# CUDA_VISIBLE_DEVICES and running a disjoint subset of
# TEMPS = [250, 300, 350, 400, 450] K. The 2+1+1+1 split below is the best
# balance possible for 5 temps on 4 GPUs; the GPU running 2 temps is the
# wall-clock bottleneck.
#
# Each process has its own `_t<T1>_<T2>...` stage-artefact suffix (enforced by
# slc.py's `--temps` handling) so the four runs do not collide on
# checkpoints, CSVs, or zarrs. FIRE is replicated per GPU -- with all four
# GPUs running FIRE in parallel the wall time is still ~244 s; the "waste"
# is only in total GPU-seconds, not turnaround.
#
# Any extra args are forwarded to every python invocation, so pass
# `--material <name>`, `--source {npt,nvt,auto}`, `--run-name <name>`,
# `--model <alias>`, `--npt-ps N` etc here:
#   ./slc_multi_gpu.sh --source nvt                                    # full production
#   ./slc_multi_gpu.sh --source nvt --npt-ps 1.5                       # smoke test
#   ./slc_multi_gpu.sh --run-name naphthalene_long_2025 \
#       --model aimnet2_2025 --source nvt                              # 2025 run
#
# `--run-name` is parsed locally so the wrapper's stdout-log directory
# (logs/<run-name>/mgpu_stdout_*.log) tracks the same root the python
# processes write their CSVs/zarrs under. Every other arg is forwarded
# verbatim. Both `--run-name foo` and `--run-name=foo` forms are accepted,
# matching argparse.
#
# `--split` overrides the per-GPU temperature partition. Pipe-separated
# GPU subsets, commas within a subset:
#   ./slc_multi_gpu.sh --split "200|300|400|500" --packmol \
#       --run-name naphthalene_orb_crystal_aniso --source npt          # 1 T per GPU
# Length must match the GPU count (4); each subset is forwarded to that
# GPU's slc.py as `--temps <subset>`. If omitted, the canonical
# 2+1+1+1 split (250,300 | 350 | 400 | 450) applies.
#
# Run from /workspace inside the alchemi-playbook-part2 container, typically
# detached via `docker exec -d` so the processes survive client disconnects:
#   srun --jobid=<J> --overlap docker exec -d -w /workspace \
#        alchemi-playbook-part2 bash -c 'exec ./slc_multi_gpu.sh --source nvt'
set -euo pipefail

# Parse --run-name and --split out of "$@" (rest is forwarded verbatim to
# python). Defaults match slc.py's --run-name default and the canonical
# 2+1+1+1 partition so a no-arg invocation preserves prior behaviour.
RUN_NAME="naphthalene_long"
SPLIT_STR=""
PASS_THROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      [[ $# -ge 2 ]] || { echo "--run-name requires a value" >&2; exit 2; }
      RUN_NAME="$2"
      PASS_THROUGH+=("$1" "$2")
      shift 2
      ;;
    --run-name=*)
      RUN_NAME="${1#--run-name=}"
      PASS_THROUGH+=("$1")
      shift
      ;;
    --split)
      [[ $# -ge 2 ]] || { echo "--split requires a value" >&2; exit 2; }
      SPLIT_STR="$2"
      shift 2
      ;;
    --split=*)
      SPLIT_STR="${1#--split=}"
      shift
      ;;
    *)
      PASS_THROUGH+=("$1")
      shift
      ;;
  esac
done

LOG_DIR="logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

if [[ -n "${SPLIT_STR}" ]]; then
  # Pipe-separated GPU subsets (e.g. "200|300|400|500"); commas allowed
  # inside a subset to give one GPU multiple temperatures (e.g. "200,250").
  IFS='|' read -r -a SPLIT <<< "${SPLIT_STR}"
  if [[ ${#SPLIT[@]} -ne 4 ]]; then
    echo "--split must have 4 pipe-separated subsets (got ${#SPLIT[@]}: ${SPLIT[*]})" >&2
    exit 2
  fi
else
  declare -a SPLIT=("250,300" "350" "400" "450")
fi

pids=()
for gpu in 0 1 2 3; do
  subset="${SPLIT[$gpu]}"
  tag="t${subset//,/_}"
  log="${LOG_DIR}/mgpu_stdout_${tag}.log"
  echo "GPU ${gpu} -> --temps ${subset}  (log: ${log})"
  # `${arr[@]+"${arr[@]}"}` = bash-portable empty-array expansion that survives
  # `set -u` (a bare `"${arr[@]}"` errors with "unbound variable" when arr is
  # empty on bash 4.x).
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python -u slc.py --temps "${subset}" "${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

echo
echo "Launched 4 processes with PIDs: ${pids[*]}"
echo "Tail all four:"
echo "  tail -f ${LOG_DIR}/mgpu_stdout_*.log"
echo "Per GPU:"
for gpu in 0 1 2 3; do
  subset="${SPLIT[$gpu]}"
  tag="t${subset//,/_}"
  echo "  tail -f ${LOG_DIR}/mgpu_stdout_${tag}.log   # GPU ${gpu}"
done
echo

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
    echo "PID ${pid} exited non-zero"
  fi
done
echo "All 4 processes finished (overall exit=${rc})"
exit "${rc}"
