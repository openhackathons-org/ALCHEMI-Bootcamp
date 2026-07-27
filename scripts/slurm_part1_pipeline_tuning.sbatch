#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-pipeline-tuning
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${ALCHEMI_RUN_ROOT:?set ALCHEMI_RUN_ROOT to the shared Compute Lab run root}"
: "${ALCHEMI_PROBE_NVT_STEPS:?set ALCHEMI_PROBE_NVT_STEPS}"
: "${ALCHEMI_PROBE_NVE_STEPS:?set ALCHEMI_PROBE_NVE_STEPS}"
: "${ALCHEMI_PROBE_BATCH_SIZES:?set ALCHEMI_PROBE_BATCH_SIZES}"
: "${ALCHEMI_PROBE_COMM_MODES:?set ALCHEMI_PROBE_COMM_MODES}"
: "${ALCHEMI_PROBE_REPEATS:?set ALCHEMI_PROBE_REPEATS}"
: "${ALCHEMI_PROBE_BATCHES_PER_PIPELINE:?set ALCHEMI_PROBE_BATCHES_PER_PIPELINE}"

RUN_ROOT="$ALCHEMI_RUN_ROOT"
SHARED_REPO="$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
MAIN_ENV="$RUN_ROOT/envs/alchemi-playbook"
CORE_ROOT="${ALCHEMI_TOOLKIT_CORE_ROOT:-$RUN_ROOT/stage/nvalchemi-toolkit-0.2.0-rc}"
DRIVER="$SHARED_REPO/scripts/benchmark_part1_distributed_campaign.py"
CONTRACT="$SHARED_REPO/scripts/part1_distributed_campaign_contract.py"
LAUNCHER="$SHARED_REPO/scripts/run_part1_distributed_torchrun.sh"
PROBE_SCRIPT="$SHARED_REPO/scripts/slurm_part1_pipeline_tuning.sbatch"
FAILURE_RECORDER="$SHARED_REPO/scripts/record_part1_campaign_failure.py"

BATCH_SIZES="$ALCHEMI_PROBE_BATCH_SIZES"
COMM_MODES="$ALCHEMI_PROBE_COMM_MODES"
REPEATS="$ALCHEMI_PROBE_REPEATS"
BATCHES_PER_PIPELINE="$ALCHEMI_PROBE_BATCHES_PER_PIPELINE"
mapfile -t TUNING_CONTRACT < <(
  PYTHONPATH="$SHARED_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$MAIN_ENV/bin/python" -c \
    'import part1_distributed_campaign_contract as c; values=(c.CORE_BRANCH,c.CORE_COMMIT,c.CORE_VERSION,c.OPS_COMMIT,c.OPS_VERSION,c.DEFAULT_FIRE_FMAX_EV_PER_A,c.DEFAULT_DT_FS,c.DEFAULT_TEMPERATURE_K,c.DEFAULT_FRICTION_PER_FS,c.EXPECTED_D3_PARAMETER_SHA256); print(*values, sep="\n")'
)
if [[ "${#TUNING_CONTRACT[@]}" -ne 10 ]]; then
  echo "Could not read the campaign tuning contract" >&2
  exit 2
fi
CORE_BRANCH="${TUNING_CONTRACT[0]}"
CORE_COMMIT="${TUNING_CONTRACT[1]}"
CORE_VERSION="${TUNING_CONTRACT[2]}"
OPS_COMMIT="${TUNING_CONTRACT[3]}"
OPS_VERSION="${TUNING_CONTRACT[4]}"
FIRE_FMAX="${ALCHEMI_PROBE_FIRE_FMAX:-${TUNING_CONTRACT[5]}}"
DT_FS="${ALCHEMI_PROBE_DT_FS:-${TUNING_CONTRACT[6]}}"
TEMPERATURE_K="${ALCHEMI_PROBE_TEMPERATURE_K:-${TUNING_CONTRACT[7]}}"
FRICTION_PER_FS="${ALCHEMI_PROBE_FRICTION_PER_FS:-${TUNING_CONTRACT[8]}}"
EXPECTED_D3_PARAMETER_SHA256="${TUNING_CONTRACT[9]}"

case "$SLURM_NNODES" in
  1) ROUTE=fused_1gpu; PIPELINE_COUNT=1 ;;
  2) ROUTE=pipeline_2gpu; PIPELINE_COUNT=1 ;;
  4) ROUTE=pipeline_4gpu; PIPELINE_COUNT=2 ;;
  *) echo "Submit with --nodes=1, --nodes=2, or --nodes=4" >&2; exit 2 ;;
esac

FINAL_DIR="$RUN_ROOT/results/pipeline-tuning-${SLURM_JOB_ID}"
mkdir -p "$FINAL_DIR/raw"
exec > >(tee -a "$FINAL_DIR/slurm-${SLURM_JOB_ID}.log") 2>&1

export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_MASTER_ADDR
ALCHEMI_MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export ALCHEMI_MASTER_PORT="$((24000 + SLURM_JOB_ID % 18000))"
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

test "$(git -C "$CORE_ROOT" rev-parse HEAD)" = \
  "$CORE_COMMIT"
test "$(git -C "$CORE_ROOT" branch --show-current)" = "$CORE_BRANCH"
test -z "$(git -C "$CORE_ROOT" status --porcelain --untracked-files=all)"
test -s "$ALCHEMI_D3_PARAM_FILE"
test "$(sha256sum "$ALCHEMI_D3_PARAM_FILE" | cut -d ' ' -f 1)" = \
  "$EXPECTED_D3_PARAMETER_SHA256"
PYTHONPATH="$CORE_ROOT:$SHARED_REPO/scripts:$SHARED_REPO/part-1-scalable-atomistic-workflows${PYTHONPATH:+:$PYTHONPATH}" \
  "$MAIN_ENV/bin/python" -c \
  'from importlib import metadata; from aux.runtime import verify_toolkit_pins; import nvalchemi; import part1_distributed_campaign_contract as c; assert nvalchemi.version == c.CORE_VERSION; assert metadata.version("nvalchemi-toolkit-ops") == c.OPS_VERSION; verify_toolkit_pins(c.CORE_COMMIT, c.OPS_COMMIT)'

printf 'job=%s route=%s nodes=%s batch_sizes=%s batches_per_pipeline=%s comm_modes=%s repeats=%s\n' \
  "$SLURM_JOB_ID" "$ROUTE" "$SLURM_NNODES" "$BATCH_SIZES" \
  "$BATCHES_PER_PIPELINE" "$COMM_MODES" "$REPEATS"
printf 'NVT steps=%s NVE steps=%s\n' \
  "$ALCHEMI_PROBE_NVT_STEPS" "$ALCHEMI_PROBE_NVE_STEPS"
date -Is
srun --ntasks="$SLURM_NNODES" \
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

for repeat in $(seq 1 "$REPEATS"); do
  for comm_mode in $COMM_MODES; do
    for batch_size in $BATCH_SIZES; do
      if (( batch_size <= 0 || BATCHES_PER_PIPELINE <= 0 )); then
        echo "batch sizes and batches per pipeline must be positive" >&2
        exit 2
      fi
      systems="$((batch_size * BATCHES_PER_PIPELINE * PIPELINE_COUNT))"
      export ALCHEMI_MASTER_PORT="$((ALCHEMI_MASTER_PORT + 1))"
      label="${ROUTE}-b${batch_size}-${comm_mode}-r${repeat}"
      output="$FINAL_DIR/raw/${label}.json"
      case_log="$FINAL_DIR/raw/${label}.log"
      echo "START $label $(date -Is)"
      set +e
      srun --kill-on-bad-exit=1 "$LAUNCHER" "$DRIVER" \
        --route "$ROUTE" \
        --purpose tuning \
        --slurm-producer scripts/slurm_part1_pipeline_tuning.sbatch \
        --systems "$systems" \
        --batch-size "$batch_size" \
        --fire-fmax "$FIRE_FMAX" \
        --nvt-steps "$ALCHEMI_PROBE_NVT_STEPS" \
        --nve-steps "$ALCHEMI_PROBE_NVE_STEPS" \
        --dt-fs "$DT_FS" \
        --temperature-k "$TEMPERATURE_K" \
        --friction-per-fs "$FRICTION_PER_FS" \
        --comm-mode "$comm_mode" \
        --repeat "$repeat" \
        --output "$output" 2>&1 | tee "$case_log"
      srun_status="${PIPESTATUS[0]}"
      set -e

      if [[ "$srun_status" -eq 0 ]]; then
        test -s "$output"
        "$MAIN_ENV/bin/python" -c \
          'import json,sys; r=json.load(open(sys.argv[1])); assert r["success"] and r["correctness_passed"]' \
          "$output"
        echo "DONE $label $(date -Is)"
        continue
      fi

      if [[ -s "$output" ]]; then
        saved_success="$("$MAIN_ENV/bin/python" -c \
          'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["success"])).lower())' \
          "$output" 2>/dev/null || echo unreadable)"
        if [[ "$saved_success" == false ]]; then
          echo "FAILED $label; the driver saved its correctness audit $(date -Is)"
          continue
        fi
        mv "$output" "${output}.incomplete"
      fi
      "$MAIN_ENV/bin/python" "$FAILURE_RECORDER" \
        --purpose tuning --route "$ROUTE" --systems "$systems" \
        --batch-size "$batch_size" --fire-fmax "$FIRE_FMAX" \
        --nvt-steps "$ALCHEMI_PROBE_NVT_STEPS" \
        --nve-steps "$ALCHEMI_PROBE_NVE_STEPS" --dt-fs "$DT_FS" \
        --temperature-k "$TEMPERATURE_K" \
        --friction-per-fs "$FRICTION_PER_FS" --comm-mode "$comm_mode" \
        --repeat "$repeat" --exit-code "$srun_status" \
        --case-log "$case_log" --repository-root "$SHARED_REPO" \
        --slurm-producer scripts/slurm_part1_pipeline_tuning.sbatch \
        --output "$output"
      echo "FAILED $label status=$srun_status $(date -Is)"
    done
  done
done

sha256sum \
  "$DRIVER" "$CONTRACT" "$LAUNCHER" "$FAILURE_RECORDER" "$PROBE_SCRIPT" \
  > "$FINAL_DIR/producer-SHA256SUMS"
find "$FINAL_DIR/raw" -type f -name '*.json' -print0 \
  | sort -z | xargs -0 -r sha256sum > "$FINAL_DIR/raw-SHA256SUMS"
find "$FINAL_DIR/raw" -type f -name '*.log' -print0 \
  | sort -z | xargs -0 -r sha256sum > "$FINAL_DIR/case-log-SHA256SUMS"

date -Is
echo PART1_PIPELINE_TUNING_JOB_COMPLETE
