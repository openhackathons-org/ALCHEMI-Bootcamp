#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-campaign
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# The pinned RC fixes reusable-buffer capacity but still skips non-float32
# segmented fields during Batch.put. Keep the exact unmodified pin and run the
# preflight below before attempting the 1/2/4-H100 campaign. Once it passes,
# first use the one-GPU balance probe to choose the NVT and NVE lengths:
#
#   ALCHEMI_CAMPAIGN_PURPOSE=balance-probe \
#   ALCHEMI_CAMPAIGN_NVT_STEPS=<pilot value> \
#   ALCHEMI_CAMPAIGN_NVE_STEPS=<pilot value> \
#   sbatch --nodes=1 scripts/slurm_part1_distributed_campaign.sbatch

: "${ALCHEMI_RUN_ROOT:?set ALCHEMI_RUN_ROOT to the shared Compute Lab run root}"
: "${ALCHEMI_CAMPAIGN_NVT_STEPS:?set the NVT length selected by the balance probe}"
: "${ALCHEMI_CAMPAIGN_NVE_STEPS:?set the NVE length selected by the balance probe}"

RUN_ROOT="$ALCHEMI_RUN_ROOT"
SHARED_REPO="$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
MAIN_ENV="$RUN_ROOT/envs/alchemi-playbook"
CORE_ROOT="${ALCHEMI_TOOLKIT_CORE_ROOT:-$RUN_ROOT/stage/nvalchemi-toolkit-0.2.0-rc}"
NODE_TMP="/tmp/alchemi-part1-campaign-${SLURM_JOB_ID}"
FINAL_DIR="$RUN_ROOT/results/pipeline-campaign-${SLURM_JOB_ID}"
DRIVER="$SHARED_REPO/scripts/benchmark_part1_distributed_campaign.py"
CONTRACT="$SHARED_REPO/scripts/part1_distributed_campaign_contract.py"
LAUNCHER="$SHARED_REPO/scripts/run_part1_distributed_torchrun.sh"
FAILURE_RECORDER="$SHARED_REPO/scripts/record_part1_campaign_failure.py"
SLURM_PRODUCER="$SHARED_REPO/scripts/slurm_part1_distributed_campaign.sbatch"
PYTHON="$MAIN_ENV/bin/python"

mapfile -t CAMPAIGN_CONTRACT < <(
  PYTHONPATH="$SHARED_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
    'import part1_distributed_campaign_contract as c; values=(c.CORE_BRANCH,c.CORE_COMMIT,c.CORE_VERSION,c.OPS_COMMIT,c.OPS_VERSION,c.DEFAULT_SYSTEMS,c.DEFAULT_BATCH_SIZE,c.BALANCE_PROBE_SYSTEMS,c.DEFAULT_FIRE_FMAX_EV_PER_A,c.DEFAULT_DT_FS,c.DEFAULT_TEMPERATURE_K,c.DEFAULT_FRICTION_PER_FS,c.DEFAULT_COMM_MODE,c.EXPECTED_D3_PARAMETER_SHA256,c.CAMPAIGN_REPEATS); print(*values, sep="\n")'
)
if [[ "${#CAMPAIGN_CONTRACT[@]}" -ne 15 ]]; then
  echo "Could not read the campaign contract" >&2
  exit 2
fi
CORE_BRANCH="${CAMPAIGN_CONTRACT[0]}"
CORE_COMMIT="${CAMPAIGN_CONTRACT[1]}"
CORE_VERSION="${CAMPAIGN_CONTRACT[2]}"
OPS_COMMIT="${CAMPAIGN_CONTRACT[3]}"
OPS_VERSION="${CAMPAIGN_CONTRACT[4]}"
SYSTEMS="${CAMPAIGN_CONTRACT[5]}"
BATCH_SIZE="${CAMPAIGN_CONTRACT[6]}"
BALANCE_SYSTEMS="${CAMPAIGN_CONTRACT[7]}"
FIRE_FMAX="${CAMPAIGN_CONTRACT[8]}"
DT_FS="${CAMPAIGN_CONTRACT[9]}"
TEMPERATURE_K="${CAMPAIGN_CONTRACT[10]}"
FRICTION_PER_FS="${CAMPAIGN_CONTRACT[11]}"
COMM_MODE="${CAMPAIGN_CONTRACT[12]}"
EXPECTED_D3_PARAMETER_SHA256="${CAMPAIGN_CONTRACT[13]}"
CONTRACT_REPEATS="${CAMPAIGN_CONTRACT[14]}"
REPEATS="${ALCHEMI_CAMPAIGN_REPEATS:-$CONTRACT_REPEATS}"
PURPOSE="${ALCHEMI_CAMPAIGN_PURPOSE:-campaign}"
PRODUCER_RELATIVE_PATHS=(
  "scripts/benchmark_part1_distributed_campaign.py"
  "scripts/part1_distributed_campaign_contract.py"
  "scripts/record_part1_campaign_failure.py"
  "scripts/run_part1_distributed_torchrun.sh"
  "scripts/slurm_part1_distributed_campaign.sbatch"
  "part-1-scalable-atomistic-workflows/aux/artifacts.py"
  "part-1-scalable-atomistic-workflows/aux/checkpoint.py"
  "part-1-scalable-atomistic-workflows/aux/electrostatics.py"
  "part-1-scalable-atomistic-workflows/aux/hooks.py"
  "part-1-scalable-atomistic-workflows/aux/runtime.py"
  "part-1-scalable-atomistic-workflows/aux/structures.py"
)

case "$SLURM_NNODES" in
  1) ROUTE=fused_1gpu ;;
  2) ROUTE=pipeline_2gpu ;;
  4) ROUTE=pipeline_4gpu ;;
  *) echo "Submit with --nodes=1, --nodes=2, or --nodes=4" >&2; exit 2 ;;
esac
if [[ "$PURPOSE" != campaign && "$PURPOSE" != balance-probe ]]; then
  echo "ALCHEMI_CAMPAIGN_PURPOSE must be campaign or balance-probe" >&2
  exit 2
fi
if [[ "$PURPOSE" == balance-probe && "$SLURM_NNODES" -ne 1 ]]; then
  echo "The balance probe requires --nodes=1" >&2
  exit 2
fi
if [[ "$REPEATS" -ne 5 && "$PURPOSE" == campaign ]]; then
  echo "Publishable campaign jobs require exactly five repeats" >&2
  exit 2
fi

export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_MASTER_ADDR
ALCHEMI_MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export ALCHEMI_MASTER_PORT="$((22000 + SLURM_JOB_ID % 20000))"
export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export XDG_DATA_HOME="$RUN_ROOT/cache/xdg-data"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export AIMNET_CACHE_DIR="$RUN_ROOT/cache/aimnet"
export ALCHEMI_D3_PARAM_FILE="$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="$MAIN_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PATH="$MAIN_ENV/bin:$PATH"

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" \
  "$AIMNET_CACHE_DIR" "$NODE_TMP" "$FINAL_DIR/raw" "$FINAL_DIR/probe"
exec > >(tee -a "$FINAL_DIR/slurm-${SLURM_JOB_ID}.log") 2>&1

test "$(git -C "$CORE_ROOT" rev-parse HEAD)" = "$CORE_COMMIT"
test "$(git -C "$CORE_ROOT" branch --show-current)" = "$CORE_BRANCH"
test -z "$(git -C "$CORE_ROOT" status --porcelain --untracked-files=all)"
test -s "$ALCHEMI_D3_PARAM_FILE"
test "$(sha256sum "$ALCHEMI_D3_PARAM_FILE" | cut -d ' ' -f 1)" = \
  "$EXPECTED_D3_PARAMETER_SHA256"
PYTHONPATH="$CORE_ROOT:$SHARED_REPO/scripts:$SHARED_REPO/part-1-scalable-atomistic-workflows${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
  'from importlib import metadata; from aux.runtime import verify_toolkit_pins; import nvalchemi; import part1_distributed_campaign_contract as c; assert nvalchemi.version == c.CORE_VERSION; assert metadata.version("nvalchemi-toolkit-ops") == c.OPS_VERSION; verify_toolkit_pins(c.CORE_COMMIT, c.OPS_COMMIT)'

# This CPU check exercises the same Batch.empty/Batch.put path used by the
# distributed stage buffers: first put, repeated put with a distinct payload,
# and zero then put with another payload. At the pinned RC, the reuse checks
# pass but integer fields still fail. The job stops before any srun or torchrun
# process starts, so invalid timings cannot be recorded.
PYTHONPATH="$CORE_ROOT:$SHARED_REPO/scripts:$SHARED_REPO/part-1-scalable-atomistic-workflows${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
  'from benchmark_part1_distributed_campaign import validate_batch_put_roundtrip; validate_batch_put_roundtrip()'

printf 'job=%s route=%s nodes=%s master=%s:%s purpose=%s\n' \
  "$SLURM_JOB_ID" "$ROUTE" "$SLURM_NNODES" \
  "$ALCHEMI_MASTER_ADDR" "$ALCHEMI_MASTER_PORT" "$PURPOSE"
printf 'NVT steps=%s NVE steps=%s\n' \
  "$ALCHEMI_CAMPAIGN_NVT_STEPS" "$ALCHEMI_CAMPAIGN_NVE_STEPS"
date -Is
srun --ntasks="$SLURM_NNODES" \
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader

common_args=(
  --route "$ROUTE"
  --batch-size "$BATCH_SIZE"
  --fire-fmax "$FIRE_FMAX"
  --nvt-steps "$ALCHEMI_CAMPAIGN_NVT_STEPS"
  --nve-steps "$ALCHEMI_CAMPAIGN_NVE_STEPS"
  --dt-fs "$DT_FS"
  --temperature-k "$TEMPERATURE_K"
  --friction-per-fs "$FRICTION_PER_FS"
  --comm-mode "$COMM_MODE"
)

CURRENT_REPEAT=""
CURRENT_OUTPUT=""
CURRENT_CASE_LOG=""

record_failed_case() {
  local exit_code="$1"
  local repeat="$2"
  local output="$3"
  local case_log="$4"

  "$PYTHON" "$FAILURE_RECORDER" \
    --route "$ROUTE" --systems "$SYSTEMS" --batch-size "$BATCH_SIZE" \
    --fire-fmax "$FIRE_FMAX" \
    --nvt-steps "$ALCHEMI_CAMPAIGN_NVT_STEPS" \
    --nve-steps "$ALCHEMI_CAMPAIGN_NVE_STEPS" \
    --dt-fs "$DT_FS" --temperature-k "$TEMPERATURE_K" \
    --friction-per-fs "$FRICTION_PER_FS" --comm-mode "$COMM_MODE" \
    --repeat "$repeat" --exit-code "$exit_code" --case-log "$case_log" \
    --repository-root "$SHARED_REPO" --output "$output"
}

write_checksum_indexes() {
  local producer_paths=()
  local relative_path
  for relative_path in "${PRODUCER_RELATIVE_PATHS[@]}"; do
    producer_paths+=("$SHARED_REPO/$relative_path")
  done

  sha256sum "${producer_paths[@]}" > "$FINAL_DIR/producer-SHA256SUMS"
  find "$FINAL_DIR/raw" -type f \( -name '*.json' -o -name '*.json.incomplete' \) -print0 \
    | sort -z | xargs -0 -r sha256sum > "$FINAL_DIR/raw-SHA256SUMS"
  find "$FINAL_DIR/raw" -type f -name '*.log' -print0 \
    | sort -z | xargs -0 -r sha256sum > "$FINAL_DIR/case-log-SHA256SUMS"
  find "$FINAL_DIR/probe" -type f -name '*.json' -print0 \
    | sort -z | xargs -0 -r sha256sum > "$FINAL_DIR/probe-SHA256SUMS"
}

record_timeout_failure() {
  trap - TERM
  set +e
  echo "Slurm sent TERM before the job time limit; recording the active case."
  if [[ -n "$CURRENT_REPEAT" && -n "$CURRENT_OUTPUT" && -n "$CURRENT_CASE_LOG" ]]; then
    "$PYTHON" -c \
      'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if isinstance(value, dict) and value.get("schema") == "alchemi.pipeline-campaign-run.v2" else 1)' \
      "$CURRENT_OUTPUT" >/dev/null 2>&1
    local record_status="$?"
    if [[ "$record_status" -ne 0 ]]; then
      if [[ -e "$CURRENT_OUTPUT" ]]; then
        mv -f "$CURRENT_OUTPUT" "${CURRENT_OUTPUT}.incomplete"
      fi
      if [[ ! -f "$CURRENT_CASE_LOG" ]]; then
        : > "$CURRENT_CASE_LOG"
      fi
      record_failed_case \
        124 "$CURRENT_REPEAT" "$CURRENT_OUTPUT" "$CURRENT_CASE_LOG" || true
    fi
  fi
  write_checksum_indexes
  exit 124
}

trap record_timeout_failure TERM

if [[ "$PURPOSE" == balance-probe ]]; then
  srun --kill-on-bad-exit=1 "$LAUNCHER" "$DRIVER" \
    "${common_args[@]}" \
    --purpose balance-probe --systems "$BALANCE_SYSTEMS" --repeat 0 \
    --output "$FINAL_DIR/probe/balance-probe.json"
else
  # Complete untimed acceptance run. It exercises the same 8192-system route
  # before any of the five records used by the tutorial are measured.
  srun --kill-on-bad-exit=1 "$LAUNCHER" "$DRIVER" \
    "${common_args[@]}" \
    --purpose smoke --systems "$SYSTEMS" --repeat 0 \
    --output "$FINAL_DIR/raw/smoke.json"

  run_measured_case() {
    local repeat="$1"
    local output="$FINAL_DIR/raw/${ROUTE}-r${repeat}.json"
    local case_log="$FINAL_DIR/raw/${ROUTE}-r${repeat}.log"
    local saved_success

    CURRENT_REPEAT="$repeat"
    CURRENT_OUTPUT="$output"
    CURRENT_CASE_LOG="$case_log"

    set +e
    srun --kill-on-bad-exit=1 "$LAUNCHER" "$DRIVER" \
      "${common_args[@]}" \
      --purpose campaign --systems "$SYSTEMS" --repeat "$repeat" \
      --output "$output" 2>&1 | tee "$case_log"
    local srun_status="${PIPESTATUS[0]}"
    set -e

    if [[ "$srun_status" -eq 0 ]]; then
      test -s "$output"
      CURRENT_REPEAT=""
      CURRENT_OUTPUT=""
      CURRENT_CASE_LOG=""
      return
    fi
    if [[ -s "$output" ]]; then
      saved_success="$("$PYTHON" -c \
        'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["success"])).lower())' \
        "$output" 2>/dev/null || echo unreadable)"
      if [[ "$saved_success" == false ]]; then
        echo "The driver saved a failed correctness record: $output"
        CURRENT_REPEAT=""
        CURRENT_OUTPUT=""
        CURRENT_CASE_LOG=""
        return
      fi
      mv "$output" "${output}.incomplete"
    fi
    record_failed_case "$srun_status" "$repeat" "$output" "$case_log"
    CURRENT_REPEAT=""
    CURRENT_OUTPUT=""
    CURRENT_CASE_LOG=""
  }

  for repeat in $(seq 1 "$REPEATS"); do
    export ALCHEMI_MASTER_PORT="$((ALCHEMI_MASTER_PORT + 1))"
    run_measured_case "$repeat"
  done
fi

write_checksum_indexes
date -Is
echo PART1_PIPELINE_CAMPAIGN_JOB_COMPLETE
