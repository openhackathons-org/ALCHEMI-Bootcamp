#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-domain
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue

set -euo pipefail

# Do not write import bytecode into the staged tutorial or Toolkit checkouts.
export PYTHONDONTWRITEBYTECODE=1

# Submit from the login node with absolute source and log paths:
#
#   export ALCHEMI_RUN_ROOT=/shared/alchemi
#   export ALCHEMI_SHARED_REPO="$ALCHEMI_RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
#   export ALCHEMI_TUTORIAL_COMMIT=<full-40-character-Git-SHA>
#   export ALCHEMI_MAIN_ENV="$ALCHEMI_RUN_ROOT/envs/alchemi-playbook"
#   export ALCHEMI_PYTHON_OVERLAY="$ALCHEMI_RUN_ROOT/envs/part1-current"
#   export ALCHEMI_TOOLKIT_CORE_ROOT="$ALCHEMI_RUN_ROOT/stage/toolkit/nvalchemi-toolkit-331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
#   export ALCHEMI_TOOLKIT_OPS_ROOT="$ALCHEMI_RUN_ROOT/stage/toolkit/nvalchemi-toolkit-ops-e8e7a7464f6745277a156a3d6f433d06b58c60e3"
#   mkdir -p "$ALCHEMI_RUN_ROOT/logs"
#   DOMAIN_JOB="$ALCHEMI_SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch"
#
# Phase 1: one H100 measures the PME-versus-Ewald check, capacity prefix,
# largest successful box, first natural CUDA OOM, and the dedicated one-GPU
# steady timing for the largest successful box.
#
#   capacity_job=$(
#     ALCHEMI_DOMAIN_PHASE=capacity \
#       sbatch --parsable --nodes=1 \
#       --chdir="$ALCHEMI_RUN_ROOT/logs" \
#       --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-capacity-%j.out" \
#       "$DOMAIN_JOB"
#   )
#   capacity_job=${capacity_job%%;*}
#
# Phase 2: submit the selected, identical inputs on 2/4 H100s. Slurm holds
# these jobs until the capacity job has completed successfully.
#
#   CAPACITY_DIR="$ALCHEMI_RUN_ROOT/results/domain-capacity-${capacity_job}-gpus-1"
#   for nodes in 2 4; do
#     ALCHEMI_DOMAIN_PHASE=distributed \
#     ALCHEMI_DOMAIN_CAPACITY_DIR="$CAPACITY_DIR" \
#       sbatch --dependency=afterok:${capacity_job} --nodes=${nodes} \
#       --chdir="$ALCHEMI_RUN_ROOT/logs" \
#       --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-${nodes}gpu-%j.out" \
#       "$DOMAIN_JOB"
#   done
#
# The `bundle` command shown by the completed jobs combines the three result
# directories. No GPU count runs the complete large-size ladder unnecessarily.

: "${ALCHEMI_RUN_ROOT:?set ALCHEMI_RUN_ROOT to shared persistent storage}"
: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to the verified Conda base}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to the verified Python package layer}"

PHASE="${ALCHEMI_DOMAIN_PHASE:-capacity}"
if [[ "$PHASE" != capacity && "$PHASE" != distributed ]]; then
  echo "ALCHEMI_DOMAIN_PHASE must be capacity or distributed" >&2
  exit 2
fi
if [[ "$PHASE" == capacity && "$SLURM_NNODES" -ne 1 ]]; then
  echo "The capacity phase requires --nodes=1" >&2
  exit 2
fi
if [[ "$PHASE" == distributed ]]; then
  : "${ALCHEMI_DOMAIN_CAPACITY_DIR:?set the completed capacity result directory}"
fi

RUN_ROOT="$ALCHEMI_RUN_ROOT"
SHARED_REPO="${ALCHEMI_SHARED_REPO:-$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp}"
: "${ALCHEMI_TUTORIAL_COMMIT:?set the full tutorial Git commit to run}"
TUTORIAL_COMMIT="$ALCHEMI_TUTORIAL_COMMIT"
MAIN_ENV="$ALCHEMI_MAIN_ENV"
PYTHON_OVERLAY="$ALCHEMI_PYTHON_OVERLAY"
CORE_COMMIT=331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409
OPS_COMMIT=e8e7a7464f6745277a156a3d6f433d06b58c60e3
CORE_ROOT="${ALCHEMI_TOOLKIT_CORE_ROOT:-$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-$CORE_COMMIT}"
OPS_ROOT="${ALCHEMI_TOOLKIT_OPS_ROOT:-$RUN_ROOT/stage/toolkit/nvalchemi-toolkit-ops-$OPS_COMMIT}"
FINAL_DIR="$RUN_ROOT/results/domain-${PHASE}-${SLURM_JOB_ID}-gpus-${SLURM_NNODES}"
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
LOCAL_CHECKPOINT=""

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

D3_PARAMETER_SHA256=b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84
NCI_SUBSET_SHA256=7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7
AIMNET_CHECKPOINT=aimnet2-wb97m-d3_0
AIMNET_CHECKPOINT_SHA256=f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28
test -f "$METHODOLOGY_CONFIG"
mapfile -t DOMAIN_METHODOLOGY_DEFAULTS < <(
  "$PYTHON" -c \
    'import importlib.util,sys; spec=importlib.util.spec_from_file_location("part1_domain_methodology", sys.argv[1]); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); config=module.DOMAIN_METHODOLOGY; print(" ".join(map(str, config.capacity_molecules_per_species))); print(config.electrostatics_validation_molecules_per_species); print(config.parity_molecules_per_species); print(" ".join(map(str, config.distributed_world_sizes)))' \
    "$METHODOLOGY_CONFIG"
)
test "${#DOMAIN_METHODOLOGY_DEFAULTS[@]}" -eq 4
PAIR_COUNTS_TEXT="${ALCHEMI_DOMAIN_PAIR_COUNTS:-${DOMAIN_METHODOLOGY_DEFAULTS[0]}}"
VALIDATION_PAIRS="${ALCHEMI_DOMAIN_VALIDATION_PAIRS:-${DOMAIN_METHODOLOGY_DEFAULTS[1]}}"
PARITY_PAIRS="${ALCHEMI_DOMAIN_PARITY_PAIRS:-${DOMAIN_METHODOLOGY_DEFAULTS[2]}}"
DISTRIBUTED_WORLD_SIZES_TEXT="${DOMAIN_METHODOLOGY_DEFAULTS[3]}"
read -r -a DEFAULT_PAIR_COUNTS <<< "${DOMAIN_METHODOLOGY_DEFAULTS[0]}"
read -r -a PAIR_COUNTS <<< "$PAIR_COUNTS_TEXT"
if [[ "${PAIR_COUNTS[*]}" != "${DEFAULT_PAIR_COUNTS[*]}" ]] || \
   [[ "$VALIDATION_PAIRS" != "${DOMAIN_METHODOLOGY_DEFAULTS[1]}" ]] || \
   [[ "$PARITY_PAIRS" != "${DOMAIN_METHODOLOGY_DEFAULTS[2]}" ]]; then
  echo "This reportable tutorial job only accepts the versioned methodology sizes." >&2
  echo "Use part1_domain_plan.py and part1_domain_run.py directly for diagnostic size overrides." >&2
  exit 2
fi
if [[ "$PHASE" == distributed ]] && \
   [[ " $DISTRIBUTED_WORLD_SIZES_TEXT " != *" $SLURM_NNODES "* ]]; then
  echo "The distributed phase requires --nodes from: $DISTRIBUTED_WORLD_SIZES_TEXT" >&2
  exit 2
fi
export AIMNET_CHECKPOINT_SHA256

export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_PYTHON_OVERLAY="$PYTHON_OVERLAY"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_TOOLKIT_OPS_ROOT="$OPS_ROOT"
export ALCHEMI_MASTER_ADDR
ALCHEMI_MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
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
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c 'set -euo pipefail; mkdir -p "$NODE_CACHE_ROOT/tmp" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$LOCAL_AIMNET_CACHE"'
export TMPDIR="$NODE_CACHE_ROOT/tmp"

test -x "$PYTHON"
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
test "$(sha256sum "$D3_PARAMETER_FILE" | cut -d ' ' -f 1)" = \
  "$D3_PARAMETER_SHA256"
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
LOCAL_CHECKPOINT="$LOCAL_AIMNET_CACHE/$(basename "$SHARED_CHECKPOINT")"
export LOCAL_AIMNET_CACHE SHARED_CHECKPOINT LOCAL_CHECKPOINT
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c 'set -euo pipefail; cp "$SHARED_CHECKPOINT" "$LOCAL_CHECKPOINT"; test "$(sha256sum "$LOCAL_CHECKPOINT" | cut -d " " -f 1)" = "$AIMNET_CHECKPOINT_SHA256"'
export AIMNET_CACHE_DIR="$LOCAL_AIMNET_CACHE"

mapfile -t GPU_NAMES < <(
  srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    nvidia-smi --query-gpu=name --format=csv,noheader
)
if [[ "${#GPU_NAMES[@]}" -ne "$SLURM_NNODES" ]]; then
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
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  nvidia-smi topo -m > "$FINAL_DIR/gpu-topology.txt"

CURRENT_MODE=""
CURRENT_CASE_ID=""
CURRENT_PAIR_COUNT=""
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
    --world-size "$SLURM_NNODES" \
    --pair-count "$CURRENT_PAIR_COUNT" \
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
  local measurement_role="$2"
  local warmup_count="$3"
  local sample_count="$4"
  local pair_count="$5"
  local case_id="$6"
  local input_dir="$7"

  CURRENT_MODE="$mode"
  CURRENT_PAIR_COUNT="$pair_count"
  CURRENT_CASE_ID="$case_id"
  CURRENT_INPUT="$input_dir/structure.extxyz"
  CURRENT_RANK_DIR="$RANK_ROOT/$case_id"
  CURRENT_LOG="$LOG_ROOT/$case_id.log"
  CURRENT_OUTPUT="$RESULT_ROOT/$case_id.json"
  mkdir -p "$CURRENT_RANK_DIR"

  set +e
  srun --kill-on-bad-exit=1 \
    --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    "$LAUNCHER" "$RUNNER" \
    --mode "$mode" \
    --measurement-role "$measurement_role" \
    --warmup-count "$warmup_count" \
    --sample-count "$sample_count" \
    --run-id "$RUN_ID" \
    --case-id "$case_id" \
    --world-size "$SLURM_NNODES" \
    --pair-count "$pair_count" \
    --input-extxyz "$CURRENT_INPUT" \
    --input-manifest "$input_dir/manifest.json" \
    --output-json "$CURRENT_OUTPUT" \
    --rank-output-dir "$CURRENT_RANK_DIR" \
    --force-output-npy "$RESULT_ROOT/$case_id-forces.npy" \
    --checkpoint "$AIMNET_CHECKPOINT" \
    --d3-parameter-file "$D3_PARAMETER_FILE" \
    2>&1 | tee "$CURRENT_LOG"
  local launch_status="${PIPESTATUS[0]}"
  set -e

  if [[ "$launch_status" -ne 0 ]]; then
    record_failure "$launch_status"
  elif [[ ! -s "$CURRENT_OUTPUT" ]]; then
    echo "The launcher exited successfully but wrote no result JSON." >&2
    record_failure 66
  fi
  CURRENT_MODE=""
  CURRENT_CASE_ID=""
  CURRENT_PAIR_COUNT=""
  CURRENT_INPUT=""
  CURRENT_RANK_DIR=""
  CURRENT_LOG=""
  CURRENT_OUTPUT=""
}

if [[ "$PHASE" == capacity ]]; then
  RUN_ID="domain-capacity-${SLURM_JOB_ID}"
  PLAN="$FINAL_DIR/plan.json"
  "$PYTHON" "$PLAN_SCRIPT" plan \
    --run-id "$RUN_ID" \
    --world-sizes 1 \
    --capacity-pair-counts "${PAIR_COUNTS[@]}" \
    --validation-pairs "$VALIDATION_PAIRS" \
    --output "$PLAN" >/dev/null

  prepare_input() {
    local pair_count="$1"
    local input_dir="$INPUT_ROOT/phenol-nma-pairs-$(printf '%06d' "$pair_count")"
    "$PYTHON" "$PLAN_SCRIPT" prepare \
      --pair-count "$pair_count" \
      --base-box-dir "$BASE_BOX_DIR" \
      --nci-data "$NCI_DATA" \
      --output-dir "$input_dir" >/dev/null
  }

  prepare_input "$VALIDATION_PAIRS"
  validation_case_id="$(
    printf 'electrostatics-validation-pairs-%06d-gpus-01' "$VALIDATION_PAIRS"
  )"
  run_case \
    electrostatics-validation electrostatics_validation 0 1 \
    "$VALIDATION_PAIRS" "$validation_case_id" \
    "$INPUT_ROOT/phenol-nma-pairs-$(printf '%06d' "$VALIDATION_PAIRS")"
  "$PYTHON" "$PLAN_SCRIPT" phase-summary \
    --phase electrostatics \
    --phase-dir "$FINAL_DIR" \
    --output "$FINAL_DIR/phase-summary.json" >/dev/null

  reached_oom=false
  for pair_count in "${PAIR_COUNTS[@]}"; do
    input_dir="$INPUT_ROOT/phenol-nma-pairs-$(printf '%06d' "$pair_count")"
    if [[ ! -f "$input_dir/manifest.json" ]]; then
      prepare_input "$pair_count"
    fi
    capacity_case_id="$(
      printf 'capacity-pairs-%06d-gpus-01' "$pair_count"
    )"
    run_case capacity capacity 0 1 \
      "$pair_count" "$capacity_case_id" "$input_dir"
    read -r success is_oom < <(
      "$PYTHON" -c \
        'import json,sys; r=json.load(open(sys.argv[1])); print(int(bool(r["success"])), int(bool(r.get("failure",{}).get("is_cuda_oom"))))' \
        "$RESULT_ROOT/$capacity_case_id.json"
    )
    if [[ "$success" -eq 1 ]]; then
      continue
    fi
    if [[ "$is_oom" -eq 1 ]]; then
      reached_oom=true
      break
    fi
    echo "Capacity failed without a CUDA OOM; the failed row was retained" >&2
    exit 1
  done
  if [[ "$reached_oom" != true ]]; then
    echo "Capacity ladder completed without a CUDA OOM; extend the declared ladder" >&2
    exit 1
  fi
  "$PYTHON" "$PLAN_SCRIPT" select \
    --plan "$PLAN" \
    --result-dir "$RESULT_ROOT" \
    --input-root "$INPUT_ROOT" \
    --parity-pairs "$PARITY_PAIRS" \
    --output "$FINAL_DIR/selection.json" >/dev/null
  IFS='|' read -r timing_pair_count timing_case_id timing_input_dir \
      timing_warmup_count timing_sample_count < <(
    "$PYTHON" -c \
      'import json,sys; c=json.load(open(sys.argv[1]))["steady_timing_case"]; print(c["pair_count"], c["case_id"], c["input"]["directory"], c["warmup_count"], c["sample_count"], sep="|")' \
      "$FINAL_DIR/selection.json"
  )
  run_case steady-timing steady_timing \
    "$timing_warmup_count" "$timing_sample_count" \
    "$timing_pair_count" "$timing_case_id" "$timing_input_dir"
  "$PYTHON" "$PLAN_SCRIPT" phase-summary \
    --phase capacity \
    --phase-dir "$FINAL_DIR" \
    --output "$FINAL_DIR/phase-summary.json" >/dev/null

  echo "Capacity result directory: $FINAL_DIR"
else
  CAPACITY_DIR="$(realpath "$ALCHEMI_DOMAIN_CAPACITY_DIR")"
  CAPACITY_DIR_NAME="$(basename "$CAPACITY_DIR")"
  if [[ ! "$CAPACITY_DIR_NAME" =~ ^domain-capacity-([0-9]+)-gpus-1$ ]]; then
    echo "Unexpected capacity directory name: $CAPACITY_DIR_NAME" >&2
    exit 2
  fi
  CAPACITY_JOB_ID="${BASH_REMATCH[1]}"
  SELECTION="$CAPACITY_DIR/selection.json"
  test -f "$SELECTION"
  RUN_ID="domain-distributed-${SLURM_JOB_ID}-gpus-${SLURM_NNODES}"
  DERIVED_PLAN="$FINAL_DIR/derived-plan.json"
  "$PYTHON" "$PLAN_SCRIPT" derive \
    --selection "$SELECTION" \
    --world-size "$SLURM_NNODES" \
    --output "$DERIVED_PLAN" >/dev/null

  mapfile -t CASE_ROWS < <(
    "$PYTHON" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); [print(c["mode"], c["measurement_role"], c["warmup_count"], c["sample_count"], c["pair_count"], c["case_id"], c["input"]["directory"], sep="|") for c in p["cases"]]' \
      "$DERIVED_PLAN"
  )
  for case_row in "${CASE_ROWS[@]}"; do
    IFS='|' read -r mode measurement_role warmup_count sample_count \
      pair_count case_id input_dir <<< "$case_row"
    run_case "$mode" "$measurement_role" "$warmup_count" "$sample_count" \
      "$pair_count" "$case_id" "$input_dir"
  done
  "$PYTHON" "$PLAN_SCRIPT" phase-summary \
    --phase distributed \
    --phase-dir "$FINAL_DIR" \
    --output "$FINAL_DIR/phase-summary.json" >/dev/null

  echo "Distributed result directory: $FINAL_DIR"
  echo "After both 2/4-GPU jobs finish, run:"
  echo "$PYTHON $PLAN_SCRIPT bundle --capacity-dir $CAPACITY_DIR \\"
  echo "  --distributed-dir DIR_2 --distributed-dir DIR_4 \\"
  echo "  --site SITE --interconnect INTERCONNECT \\"
  echo "  --producer-file $LAUNCHER --producer-file $SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch \\"
  echo "  --producer-file $PACKING_HELPER --producer-file $METHODOLOGY_CONFIG \\"
  echo "  --producer-file $BASE_BOX_MANIFEST --producer-file $BASE_BOX_STRUCTURE \\"
  echo "  --producer-file $BASE_BOX_CHECKSUMS \\"
  echo "  --output-dir $RUN_ROOT/results/domain-bundle-$CAPACITY_JOB_ID"
  echo "Validate that bundle, commit it under data/domain_decomposition/recorded,"
  echo "then run the notebook from a fresh checkout of the result commit."
fi

test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$SHARED_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard
)"
sha256sum -c "$FINAL_DIR/producer-SHA256SUMS"
find "$RESULT_ROOT" "$RANK_ROOT" "$LOG_ROOT" \
  "$INPUT_ROOT" -type f -print0 > "$FINAL_DIR/artifact-paths.bin"
for campaign_file in \
  "$FINAL_DIR/plan.json" \
  "$FINAL_DIR/selection.json" \
  "$FINAL_DIR/derived-plan.json" \
  "$FINAL_DIR/phase-summary.json" \
  "$FINAL_DIR/aimnet-checkpoint-preflight.json" \
  "$FINAL_DIR/part1-runtime.json" \
  "$FINAL_DIR/d3-cache.json" \
  "$FINAL_DIR/gpu-names.txt" \
  "$FINAL_DIR/gpu-topology.txt"; do
  if [[ -f "$campaign_file" ]]; then
    printf '%s\0' "$campaign_file" >> "$FINAL_DIR/artifact-paths.bin"
  fi
done
sort -zu "$FINAL_DIR/artifact-paths.bin" \
  | xargs -0 sha256sum > "$FINAL_DIR/artifact-SHA256SUMS"
rm "$FINAL_DIR/artifact-paths.bin"
