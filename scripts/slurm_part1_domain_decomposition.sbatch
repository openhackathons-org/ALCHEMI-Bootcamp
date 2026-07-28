#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-domain
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue

set -euo pipefail

# Submit three independent jobs. They evaluate the same 51,200-atom input and
# do not wait for one another:
#
#   DOMAIN_JOB="$ALCHEMI_SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch"
#   for nodes in 1 2 4; do
#     sbatch --nodes="$nodes" \
#       --chdir="$ALCHEMI_RUN_ROOT/logs" \
#       --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-${nodes}gpu-%j.out" \
#       "$DOMAIN_JOB"
#   done
#
# Each job performs one untimed initialization pass followed by three measured
# fixed-structure energy-and-force passes. The one-GPU job also checks PME
# against direct Ewald on the smaller 3,200-atom input.

export PYTHONDONTWRITEBYTECODE=1

: "${ALCHEMI_RUN_ROOT:?set ALCHEMI_RUN_ROOT to shared persistent storage}"
: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to the verified Conda base}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to the verified Python package layer}"
: "${ALCHEMI_TUTORIAL_COMMIT:?set the full tutorial Git commit to run}"

RUN_ROOT="$ALCHEMI_RUN_ROOT"
SHARED_REPO="${ALCHEMI_SHARED_REPO:-$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp}"
TUTORIAL_COMMIT="$ALCHEMI_TUTORIAL_COMMIT"
MAIN_ENV="$ALCHEMI_MAIN_ENV"
PYTHON_OVERLAY="$ALCHEMI_PYTHON_OVERLAY"
CORE_COMMIT=331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409
OPS_COMMIT=e8e7a7464f6745277a156a3d6f433d06b58c60e3
CORE_ROOT="${ALCHEMI_TOOLKIT_CORE_ROOT:-$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-$CORE_COMMIT}"
OPS_ROOT="${ALCHEMI_TOOLKIT_OPS_ROOT:-$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-ops-$OPS_COMMIT}"

WORLD_SIZE="$SLURM_NNODES"
RUN_ID="domain-fixed-${SLURM_JOB_ID}-gpus-${WORLD_SIZE}"
FINAL_DIR="$RUN_ROOT/results/$RUN_ID"
INPUT_ROOT="$FINAL_DIR/inputs"
RESULT_ROOT="$FINAL_DIR/results"
RANK_ROOT="$FINAL_DIR/ranks"
LOG_ROOT="$FINAL_DIR/logs"

PLAN_SCRIPT="$SHARED_REPO/scripts/part1_domain_plan.py"
RUNNER="$SHARED_REPO/scripts/part1_domain_run.py"
LAUNCHER="$SHARED_REPO/scripts/run_part1_domain_decomposition.sh"
NCI_DATA="$SHARED_REPO/part-1-scalable-atomistic-workflows/data/nci_atlas/nci-atlas-curves.csv.gz"
PACKING_HELPER="$SHARED_REPO/part-1-scalable-atomistic-workflows/aux/domain/packing.py"
METHODOLOGY_CONFIG="$SHARED_REPO/part-1-scalable-atomistic-workflows/aux/domain/config.py"
BASE_BOX_DIR="$SHARED_REPO/part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box"
BASE_BOX_MANIFEST="$BASE_BOX_DIR/manifest.json"
BASE_BOX_STRUCTURE="$BASE_BOX_DIR/structure.extxyz"
BASE_BOX_CHECKSUMS="$BASE_BOX_DIR/SHA256SUMS"

READY_FILE="$PYTHON_OVERLAY/.part1-ready.json"
OVERLAY_SITE="$PYTHON_OVERLAY/lib/python3.12/site-packages"
PYTHON="$PYTHON_OVERLAY/bin/python"
D3_PARAMETER_FILE="${ALCHEMI_D3_PARAM_FILE:-$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt}"
SHARED_AIMNET_CACHE="${ALCHEMI_SHARED_AIMNET_CACHE:-$RUN_ROOT/cache/aimnet}"
NODE_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/alchemi-part1-domain-${SLURM_JOB_ID}"
LOCAL_AIMNET_CACHE="$NODE_CACHE_ROOT/aimnet"

if [[ ! "$TUTORIAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ALCHEMI_TUTORIAL_COMMIT must be a full lowercase 40-character Git SHA" >&2
  exit 2
fi
test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$SHARED_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard
)"
test -s "$READY_FILE"
test -d "$MAIN_ENV/conda-meta"
test -d "$OVERLAY_SITE"
test -x "$PYTHON"
test -x "$PYTHON_OVERLAY/bin/torchrun"

mapfile -t DOMAIN_METHODOLOGY_DEFAULTS < <(
  "$PYTHON" -c \
    'import importlib.util,sys; spec=importlib.util.spec_from_file_location("part1_domain_methodology", sys.argv[1]); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); config=module.DOMAIN_METHODOLOGY; print(config.fixed_molecules_per_species); print(config.electrostatics_validation_molecules_per_species); print(" ".join(map(str, config.campaign_world_sizes))); print(config.evaluation_warmup_count); print(config.evaluation_pass_count)' \
    "$METHODOLOGY_CONFIG"
)
test "${#DOMAIN_METHODOLOGY_DEFAULTS[@]}" -eq 5
DOMAIN_DEFAULTS=("${DOMAIN_METHODOLOGY_DEFAULTS[@]}")
FIXED_PAIRS="${DOMAIN_DEFAULTS[0]}"
VALIDATION_PAIRS="${DOMAIN_DEFAULTS[1]}"
SUPPORTED_WORLD_SIZES="${DOMAIN_DEFAULTS[2]}"
WARMUP_COUNT="${DOMAIN_DEFAULTS[3]}"
PASS_COUNT="${DOMAIN_DEFAULTS[4]}"
if [[ " $SUPPORTED_WORLD_SIZES " != *" $WORLD_SIZE "* ]]; then
  echo "This example supports --nodes 1, 2, or 4; got $WORLD_SIZE" >&2
  exit 2
fi
if [[ "$WARMUP_COUNT" -ne 1 || "$PASS_COUNT" -ne 3 ]]; then
  echo "The recorded example requires one warmup and three measured passes" >&2
  exit 2
fi

export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_PYTHON_OVERLAY="$PYTHON_OVERLAY"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_TOOLKIT_OPS_ROOT="$OPS_ROOT"
MASTER_NODE="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_NODE
export ALCHEMI_MASTER_ADDR
ALCHEMI_MASTER_ADDR="$(
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$MASTER_NODE" \
    "$LAUNCHER" --print-master-address
)"
if [[ -z "$ALCHEMI_MASTER_ADDR" ]]; then
  echo "Could not resolve a routed global IPv4 address on $MASTER_NODE" >&2
  exit 2
fi
export ALCHEMI_MASTER_PORT="$((22000 + SLURM_JOB_ID % 20000))"
export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export XDG_DATA_HOME="$RUN_ROOT/cache/xdg-data"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export AIMNET_CACHE_DIR="$SHARED_AIMNET_CACHE"
export ALCHEMI_D3_PARAM_FILE="$D3_PARAMETER_FILE"
export NODE_CACHE_ROOT
export WARP_CACHE_PATH="$NODE_CACHE_ROOT/warp"
export CUDA_CACHE_PATH="$NODE_CACHE_ROOT/cuda"
export LOCAL_AIMNET_CACHE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="$OVERLAY_SITE/nvidia/cu13/lib:$MAIN_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MAIN_ENV/bin:$PATH"

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" \
  "$SHARED_AIMNET_CACHE" "$INPUT_ROOT" "$RESULT_ROOT" "$RANK_ROOT" "$LOG_ROOT"
srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
  "$LAUNCHER" --print-network-record \
  | LC_ALL=C sort > "$FINAL_DIR/network-interfaces.txt"
srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
  bash -c 'set -euo pipefail; mkdir -p "$NODE_CACHE_ROOT/tmp" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$LOCAL_AIMNET_CACHE"'
export TMPDIR="$NODE_CACHE_ROOT/tmp"

test -f "$PLAN_SCRIPT"
test -f "$RUNNER"
test -x "$LAUNCHER"
test -f "$NCI_DATA"
test -f "$PACKING_HELPER"
test -f "$METHODOLOGY_CONFIG"
test -s "$BASE_BOX_MANIFEST"
test -s "$BASE_BOX_STRUCTURE"
test -s "$BASE_BOX_CHECKSUMS"
test -f "$SHARED_REPO/build/prewarm_d3.py"
test -d "$CORE_ROOT"
test -d "$OPS_ROOT"

"$PYTHON" "$SHARED_REPO/build/verify_part1_runtime.py" \
  --source-root "$SHARED_REPO" \
  --require-cuda \
  --require-clean-source \
  --output "$FINAL_DIR/part1-runtime.json" >/dev/null
"$PYTHON" "$SHARED_REPO/build/prewarm_d3.py" \
  --parameter-file "$D3_PARAMETER_FILE" \
  --output "$FINAL_DIR/d3-cache.json" >/dev/null
test "$(git -C "$CORE_ROOT" rev-parse HEAD)" = "$CORE_COMMIT"
test -z "$(git -C "$CORE_ROOT" status --porcelain --untracked-files=all)"
test -z "$(
  git -C "$CORE_ROOT" ls-files --others --ignored --exclude-standard
)"
test "$(git -C "$OPS_ROOT" rev-parse HEAD)" = "$OPS_COMMIT"
test -z "$(git -C "$OPS_ROOT" status --porcelain --untracked-files=all)"
test -z "$(
  git -C "$OPS_ROOT" ls-files --others --ignored --exclude-standard
)"

D3_PARAMETER_SHA256=b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84
NCI_SUBSET_SHA256=7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7
AIMNET_CHECKPOINT=aimnet2-wb97m-d3_0
AIMNET_CHECKPOINT_SHA256=f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28
export AIMNET_CHECKPOINT_SHA256
test "$(sha256sum "$D3_PARAMETER_FILE" | cut -d ' ' -f 1)" = "$D3_PARAMETER_SHA256"
test "$(sha256sum "$NCI_DATA" | cut -d ' ' -f 1)" = "$NCI_SUBSET_SHA256"

sha256sum "$PLAN_SCRIPT" "$RUNNER" "$LAUNCHER" \
  "$SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch" \
  "$PACKING_HELPER" "$METHODOLOGY_CONFIG" "$NCI_DATA" \
  "$BASE_BOX_MANIFEST" "$BASE_BOX_STRUCTURE" "$BASE_BOX_CHECKSUMS" \
  "$D3_PARAMETER_FILE" \
  > "$FINAL_DIR/producer-SHA256SUMS"

"$PYTHON" "$PLAN_SCRIPT" checkpoint-preflight \
  --checkpoint "$AIMNET_CHECKPOINT" \
  --output "$FINAL_DIR/aimnet-checkpoint-preflight.json" >/dev/null
SHARED_CHECKPOINT="$(
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["path"])' \
    "$FINAL_DIR/aimnet-checkpoint-preflight.json"
)"
test -f "$SHARED_CHECKPOINT"
test "$(sha256sum "$SHARED_CHECKPOINT" | cut -d ' ' -f 1)" = \
  "$AIMNET_CHECKPOINT_SHA256"
export SHARED_CHECKPOINT
srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
  bash -c 'set -euo pipefail; cp "$SHARED_CHECKPOINT" "$LOCAL_AIMNET_CACHE/$(basename "$SHARED_CHECKPOINT")"; test "$(sha256sum "$LOCAL_AIMNET_CACHE/$(basename "$SHARED_CHECKPOINT")" | cut -d " " -f 1)" = "$AIMNET_CHECKPOINT_SHA256"'
export AIMNET_CACHE_DIR="$LOCAL_AIMNET_CACHE"

mapfile -t GPU_NAMES < <(
  srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
    nvidia-smi --query-gpu=name --format=csv,noheader
)
if [[ "${#GPU_NAMES[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "Expected one GPU report per node" >&2
  exit 2
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  if [[ "$gpu_name" != *H100* ]]; then
    echo "This run requires H100 GPUs; found: $gpu_name" >&2
    exit 2
  fi
done
printf '%s\n' "${GPU_NAMES[@]}" > "$FINAL_DIR/gpu-names.txt"
srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
  nvidia-smi topo -m > "$FINAL_DIR/gpu-topology.txt"

"$PYTHON" "$PLAN_SCRIPT" plan \
  --run-id "$RUN_ID" \
  --tutorial-commit "$TUTORIAL_COMMIT" \
  --world-size "$WORLD_SIZE" \
  --output "$FINAL_DIR/plan.json" >/dev/null

prepare_input() {
  local pair_count="$1"
  local input_dir="$INPUT_ROOT/$(printf 'phenol-nma-pairs-%06d' "$pair_count")"
  "$PYTHON" "$PLAN_SCRIPT" prepare \
    --pair-count "$pair_count" \
    --base-box-dir "$BASE_BOX_DIR" \
    --nci-data "$NCI_DATA" \
    --output-dir "$input_dir" >/dev/null
}

prepare_input "$FIXED_PAIRS"
if [[ "$WORLD_SIZE" -eq 1 && "$VALIDATION_PAIRS" -ne "$FIXED_PAIRS" ]]; then
  prepare_input "$VALIDATION_PAIRS"
fi

CURRENT_MODE=""
CURRENT_CASE_ID=""
CURRENT_INPUT=""
CURRENT_RANK_DIR=""
CURRENT_LOG=""
CURRENT_OUTPUT=""

record_failure() {
  local exit_code="$1"
  if [[ -e "$CURRENT_OUTPUT" ]]; then
    mv "$CURRENT_OUTPUT" "${CURRENT_OUTPUT}.incomplete"
  fi
  "$PYTHON" "$PLAN_SCRIPT" record-failure \
    --run-id "$RUN_ID" \
    --case-id "$CURRENT_CASE_ID" \
    --mode "$CURRENT_MODE" \
    --world-size "$WORLD_SIZE" \
    --input-extxyz "$CURRENT_INPUT" \
    --rank-output-dir "$CURRENT_RANK_DIR" \
    --case-log "$CURRENT_LOG" \
    --exit-code "$exit_code" \
    --output "$CURRENT_OUTPUT" >/dev/null
}

record_active_timeout() {
  trap - TERM
  set +e
  if [[ -n "$CURRENT_CASE_ID" && ! -f "$CURRENT_OUTPUT" ]]; then
    record_failure 124
  fi
  exit 124
}
trap record_active_timeout TERM

run_case() {
  local mode="$1"
  local pair_count="$2"
  local case_id="$3"
  local input_dir="$INPUT_ROOT/$(printf 'phenol-nma-pairs-%06d' "$pair_count")"

  CURRENT_MODE="$mode"
  CURRENT_CASE_ID="$case_id"
  CURRENT_INPUT="$input_dir/structure.extxyz"
  CURRENT_RANK_DIR="$RANK_ROOT/$case_id"
  CURRENT_LOG="$LOG_ROOT/$case_id.log"
  CURRENT_OUTPUT="$RESULT_ROOT/$case_id.json"
  mkdir -p "$CURRENT_RANK_DIR"

  local force_args=()
  if [[ "$mode" == distributed ]]; then
    force_args=(
      --force-output-npy "$RESULT_ROOT/${case_id}-forces.npy"
    )
  fi

  set +e
  srun --kill-on-bad-exit=1 \
    --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
    "$LAUNCHER" "$RUNNER" \
      --mode "$mode" \
      --run-id "$RUN_ID" \
      --case-id "$case_id" \
      --world-size "$WORLD_SIZE" \
      --pair-count "$pair_count" \
      --input-extxyz "$CURRENT_INPUT" \
      --input-manifest "$input_dir/manifest.json" \
      --output-json "$CURRENT_OUTPUT" \
      --rank-output-dir "$CURRENT_RANK_DIR" \
      --checkpoint "$AIMNET_CHECKPOINT" \
      --d3-parameter-file "$D3_PARAMETER_FILE" \
      "${force_args[@]}" \
      >"$CURRENT_LOG" 2>&1
  local exit_code="$?"
  set -e
  if [[ "$exit_code" -ne 0 ]]; then
    record_failure "$exit_code"
  elif [[ ! -s "$CURRENT_OUTPUT" ]]; then
    echo "The launcher exited successfully but wrote no result JSON." \
      >>"$CURRENT_LOG"
    record_failure 66
  fi
}

if [[ "$WORLD_SIZE" -eq 1 ]]; then
  validation_case_id="$(
    printf 'electrostatics-validation-pairs-%06d-gpus-01' "$VALIDATION_PAIRS"
  )"
  run_case electrostatics-validation "$VALIDATION_PAIRS" "$validation_case_id"
fi

fixed_case_id="$(
  printf 'fixed-evaluation-pairs-%06d-gpus-%02d' "$FIXED_PAIRS" "$WORLD_SIZE"
)"
run_case distributed "$FIXED_PAIRS" "$fixed_case_id"

"$PYTHON" "$PLAN_SCRIPT" assemble \
  --plan "$FINAL_DIR/plan.json" \
  --result-dir "$RESULT_ROOT" \
  --output-jsonl "$FINAL_DIR/results.jsonl" \
  --output-summary "$FINAL_DIR/collection-summary.json" >/dev/null
"$PYTHON" "$PLAN_SCRIPT" phase-summary \
  --phase-dir "$FINAL_DIR" \
  --output "$FINAL_DIR/phase-summary.json" >/dev/null

test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$SHARED_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard
)"
sha256sum -c "$FINAL_DIR/producer-SHA256SUMS"

artifact_files=(
  "$FINAL_DIR/plan.json"
  "$FINAL_DIR/part1-runtime.json"
  "$FINAL_DIR/d3-cache.json"
  "$FINAL_DIR/aimnet-checkpoint-preflight.json"
  "$FINAL_DIR/gpu-names.txt"
  "$FINAL_DIR/gpu-topology.txt"
  "$FINAL_DIR/network-interfaces.txt"
  "$FINAL_DIR/producer-SHA256SUMS"
  "$FINAL_DIR/results.jsonl"
  "$FINAL_DIR/collection-summary.json"
  "$FINAL_DIR/phase-summary.json"
)
for result_file in "$RESULT_ROOT"/*; do
  test -f "$result_file"
  artifact_files+=("$result_file")
done
for rank_dir in "$RANK_ROOT"/*; do
  test -d "$rank_dir"
  for rank_file in "$rank_dir"/*; do
    test -f "$rank_file"
    artifact_files+=("$rank_file")
  done
done
for log_file in "$LOG_ROOT"/*; do
  test -f "$log_file"
  artifact_files+=("$log_file")
done
for input_dir in "$INPUT_ROOT"/*; do
  test -d "$input_dir"
  artifact_files+=(
    "$input_dir/manifest.json"
    "$input_dir/structure.extxyz"
  )
done
sha256sum "${artifact_files[@]}" > "$FINAL_DIR/artifact-SHA256SUMS"

echo "Completed $WORLD_SIZE-GPU fixed-input job: $FINAL_DIR"
echo "After all three jobs finish, run:"
echo "$PYTHON $PLAN_SCRIPT bundle \\"
echo "  --job-dir DIR_1 --job-dir DIR_2 --job-dir DIR_4 \\"
echo "  --site COMPUTE_LAB --interconnect INTERCONNECT \\"
echo "  --output-dir $RUN_ROOT/results/domain-fixed-bundle"
