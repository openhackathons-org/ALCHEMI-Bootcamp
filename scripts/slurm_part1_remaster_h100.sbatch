#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-remaster
#SBATCH --partition=h100-80gb-hbm3@ts6/h12dsgocpu/1gpu-48cpu-256gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Set ALCHEMI_RUN_ROOT for a site-specific persistent scratch location.
RUN_ROOT="${ALCHEMI_RUN_ROOT:-${SCRATCH:-$HOME}/alchemi-ir}"
SHARED_REPO="$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
MAIN_ENV="$RUN_ROOT/envs/alchemi-playbook"
NODE_TMP="/tmp/alchemi-part1-remaster-${SLURM_JOB_ID}"
LOCAL_REPO="$NODE_TMP/ALCHEMI-Bootcamp"
PART_REL=part-1-water-hydrogen-bonding-toolkit
NOTEBOOK_REL="$PART_REL/alchemi-water-ir.ipynb"
RUN_OUTPUT_REL="$PART_REL/outputs/run-${SLURM_JOB_ID}"
LOCAL_EXECUTED="$NODE_TMP/alchemi-water-ir-executed.ipynb"
FINAL_DIR="$RUN_ROOT/results/notebook-remaster-${SLURM_JOB_ID}"
TRANSCRIPT_PATH="$FINAL_DIR/notebook-transcript-${SLURM_JOB_ID}.log"

export ALCHEMI_RUN_ID="$SLURM_JOB_ID"
export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export XDG_DATA_HOME="$RUN_ROOT/cache/xdg-data"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export AIMNET_CACHE_DIR="$RUN_ROOT/cache/aimnet"
export ALCHEMI_D3_PARAM_FILE="$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt"
export TMPDIR="$NODE_TMP/t"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MPLCONFIGDIR="$NODE_TMP/mpl"
export WARP_CACHE_PATH="$NODE_TMP/wp"
export CUDA_CACHE_PATH="$NODE_TMP/cu"
export TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/ti"
export TRITON_CACHE_DIR="$NODE_TMP/tr"
export JUPYTER_DATA_DIR="$NODE_TMP/jd"
export JUPYTER_CONFIG_DIR="$NODE_TMP/jc"
export OVITO_GUI_MODE=headless
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$MAIN_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PATH="$MAIN_ENV/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" \
  "$AIMNET_CACHE_DIR" \
  "$TMPDIR" "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$JUPYTER_DATA_DIR" \
  "$JUPYTER_CONFIG_DIR" "$FINAL_DIR/outputs/run-${SLURM_JOB_ID}"

exec 3>&1
exec > >(tee -a "$TRANSCRIPT_PATH" >&3) 2>&1
TRANSCRIPT_TEE_PID=$!

preserve_run() {
  status=$?
  set +e
  if [[ -d "$LOCAL_REPO/$RUN_OUTPUT_REL" ]]; then
    rsync -a \
      "$LOCAL_REPO/$RUN_OUTPUT_REL/" \
      "$FINAL_DIR/outputs/run-${SLURM_JOB_ID}/"
  fi
  if [[ -f "$LOCAL_EXECUTED" ]]; then
    rsync -a "$LOCAL_EXECUTED" "$FINAL_DIR/"
  fi
  if [[ -f "$LOCAL_REPO/$NOTEBOOK_REL" ]]; then
    rsync -a \
      "$LOCAL_REPO/$NOTEBOOK_REL" \
      "$FINAL_DIR/alchemi-water-ir-source.ipynb"
  fi
  if [[ -f "$LOCAL_REPO/scripts/run_notebook_no_timeout.py" ]]; then
    rsync -a \
      "$LOCAL_REPO/scripts/run_notebook_no_timeout.py" \
      "$FINAL_DIR/run_notebook_no_timeout.py"
  fi
  date -Is
  echo "NOTEBOOK_EXIT_STATUS=$status"
  exec 1>&3 2>&3
  wait "$TRANSCRIPT_TEE_PID"
  if [[ -f "$FINAL_DIR/SHA256SUMS" ]]; then
    cd "$FINAL_DIR"
    sha256sum "$(basename "$TRANSCRIPT_PATH")" >> SHA256SUMS
    sha256sum -c SHA256SUMS
  fi
  trap - EXIT
  exit "$status"
}
trap preserve_run EXIT

echo "== allocation =="
date -Is
hostname
printf 'job=%s cpus=%s tmp=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$TMPDIR"
nvidia-smi

echo "== stage immutable execution copy =="
rsync -a "$SHARED_REPO/" "$LOCAL_REPO/"

KERNEL_PREFIX="$NODE_TMP/kernel"
"$MAIN_ENV/bin/python" -m ipykernel install \
  --prefix "$KERNEL_PREFIX" \
  --name alchemi-main \
  --display-name "ALCHEMI Main"
export JUPYTER_PATH="$KERNEL_PREFIX/share/jupyter"

echo "== source identity =="
cd "$LOCAL_REPO"
sha256sum "$NOTEBOOK_REL"
sha256sum \
  "$PART_REL/reference/artifacts/water_dimer_b97_3c/interaction_curve.csv" \
  "$PART_REL/reference/artifacts/water_dimer_b97_3c/manifest.json"
test -s "$ALCHEMI_D3_PARAM_FILE"
D3_PARAMETER_SHA256=$(sha256sum "$ALCHEMI_D3_PARAM_FILE")
test "${D3_PARAMETER_SHA256%% *}" = \
  "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
printf 'D3 parameter cache: %s\n' "$D3_PARAMETER_SHA256"

echo "== complete Part 1 notebook: 5,000 NVT + 50,000 NVE =="
"$MAIN_ENV/bin/python" "$LOCAL_REPO/scripts/run_notebook_no_timeout.py" \
  "$NOTEBOOK_REL" \
  --output "$LOCAL_EXECUTED" \
  --kernel alchemi-main

echo "== package and validate =="
rsync -a \
  "$LOCAL_REPO/$RUN_OUTPUT_REL/" \
  "$FINAL_DIR/outputs/run-${SLURM_JOB_ID}/"
rsync -a "$LOCAL_EXECUTED" "$FINAL_DIR/"
rsync -a "$LOCAL_REPO/$NOTEBOOK_REL" "$FINAL_DIR/alchemi-water-ir-source.ipynb"
rsync -a \
  "$LOCAL_REPO/scripts/run_notebook_no_timeout.py" \
  "$FINAL_DIR/run_notebook_no_timeout.py"

"$MAIN_ENV/bin/python" "$LOCAL_REPO/scripts/validate_part1_ir_run.py" \
  --executed-notebook "$FINAL_DIR/alchemi-water-ir-executed.ipynb" \
  --output-dir "$FINAL_DIR/outputs/run-${SLURM_JOB_ID}" \
  --source-root "$LOCAL_REPO" \
  --summary "$FINAL_DIR/notebook-validation.json" \
  --checksums "$FINAL_DIR/SHA256SUMS"

cd "$FINAL_DIR"
sha256sum -c SHA256SUMS
nvidia-smi
echo "NOTEBOOK_COMPLETE"
