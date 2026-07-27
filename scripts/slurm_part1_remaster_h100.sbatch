#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-remaster
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --no-requeue

set -euo pipefail

# Do not write import bytecode into the staged tutorial checkout.
export PYTHONDONTWRITEBYTECODE=1

# Submit from the login node with absolute source and log paths:
#
#   export ALCHEMI_RUN_ROOT=/shared/alchemi
#   export ALCHEMI_SHARED_REPO="$ALCHEMI_RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
#   export ALCHEMI_TUTORIAL_COMMIT=<full-40-character-Git-SHA>
#   export ALCHEMI_MAIN_ENV="$ALCHEMI_RUN_ROOT/envs/alchemi-playbook"
#   export ALCHEMI_PYTHON_OVERLAY="$ALCHEMI_RUN_ROOT/envs/part1-current"
#   mkdir -p "$ALCHEMI_RUN_ROOT/logs"
#   sbatch \
#     --chdir="$ALCHEMI_RUN_ROOT/logs" \
#     --output="$ALCHEMI_RUN_ROOT/logs/part1-remaster-%j.out" \
#     "$ALCHEMI_SHARED_REPO/scripts/slurm_part1_remaster_h100.sbatch"
#
# Set ALCHEMI_RUN_ROOT for a site-specific persistent scratch location.
# ALCHEMI_SHARED_REPO can point to an isolated staged copy for validation runs.
RUN_ROOT="${ALCHEMI_RUN_ROOT:-${SCRATCH:-$HOME}/alchemi-ir}"
SHARED_REPO="${ALCHEMI_SHARED_REPO:-$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp}"
: "${ALCHEMI_TUTORIAL_COMMIT:?set the full tutorial Git commit to run}"
: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to the verified Conda base}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to the verified Python package layer}"
TUTORIAL_COMMIT="$ALCHEMI_TUTORIAL_COMMIT"
MAIN_ENV="$ALCHEMI_MAIN_ENV"
PYTHON_OVERLAY="$ALCHEMI_PYTHON_OVERLAY"
NODE_TMP="/tmp/alchemi-part1-remaster-${SLURM_JOB_ID}"
LOCAL_REPO="$NODE_TMP/ALCHEMI-Bootcamp"
PART_REL=part-1-scalable-atomistic-workflows
NOTEBOOK_REL="$PART_REL/alchemi-water-ir.ipynb"
RUN_OUTPUT_REL="$PART_REL/outputs/run-${SLURM_JOB_ID}"
LOCAL_EXECUTED="$NODE_TMP/alchemi-water-ir-executed.ipynb"
FINAL_DIR="$RUN_ROOT/results/notebook-remaster-${SLURM_JOB_ID}"
TRANSCRIPT_PATH="$FINAL_DIR/notebook-transcript-${SLURM_JOB_ID}.log"
TIMING_PATH="$FINAL_DIR/notebook-timings.json"
READY_FILE="$PYTHON_OVERLAY/.part1-ready.json"
OVERLAY_SITE="$PYTHON_OVERLAY/lib/python3.12/site-packages"
PYTHON="$PYTHON_OVERLAY/bin/python"
JUPYTER="$PYTHON_OVERLAY/bin/jupyter"

export ALCHEMI_RUN_ID="$SLURM_JOB_ID"
export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_PYTHON_OVERLAY="$PYTHON_OVERLAY"
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
export LD_LIBRARY_PATH="$OVERLAY_SITE/nvidia/cu13/lib:$MAIN_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MAIN_ENV/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"

test -s "$READY_FILE"
test -d "$MAIN_ENV/conda-meta"
test -d "$OVERLAY_SITE"
test -x "$PYTHON"
test -x "$JUPYTER"

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
  checksum_status=0
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
    checksum_status=$?
    if [[ "$checksum_status" -eq 0 ]]; then
      sha256sum -c SHA256SUMS
      checksum_status=$?
    fi
  elif [[ "$status" -eq 0 ]]; then
    echo "Successful notebook run did not create SHA256SUMS" >&2
    checksum_status=1
  fi
  if [[ "$status" -eq 0 && "$checksum_status" -ne 0 ]]; then
    status="$checksum_status"
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

echo "== stage the requested committed source =="
if [[ ! "$TUTORIAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ALCHEMI_TUTORIAL_COMMIT must be a full lowercase 40-character Git SHA" >&2
  exit 2
fi
test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$SHARED_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard
)"
git clone --no-checkout --no-hardlinks "$SHARED_REPO" "$LOCAL_REPO"
git -C "$LOCAL_REPO" checkout --detach "$TUTORIAL_COMMIT"
test "$(git -C "$LOCAL_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$LOCAL_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$LOCAL_REPO" ls-files --others --ignored --exclude-standard
)"

KERNEL_PREFIX="$NODE_TMP/kernel"
"$PYTHON" -m ipykernel install \
  --prefix "$KERNEL_PREFIX" \
  --name alchemi-main \
  --display-name "ALCHEMI Main"
export JUPYTER_PATH="$KERNEL_PREFIX/share/jupyter"
"$JUPYTER" kernelspec list

echo "== files and package versions =="
cd "$LOCAL_REPO"
sha256sum "$NOTEBOOK_REL"
sha256sum \
  "$PART_REL/reference/artifacts/water_dimer_b97_3c/interaction_curve.csv" \
  "$PART_REL/reference/artifacts/water_dimer_b97_3c/manifest.json"
(
  cd "$PART_REL/reference/experimental_water_fundamentals"
  sha256sum -c SHA256SUMS
)
test -d "$PART_REL/data/domain_decomposition/recorded"
(
  cd "$PART_REL/data/domain_decomposition/recorded"
  sha256sum -c SHA256SUMS
)
"$PYTHON" "$LOCAL_REPO/build/verify_part1_runtime.py" \
  --source-root "$LOCAL_REPO" \
  --require-cuda \
  --require-clean-source \
  --output "$FINAL_DIR/part1-runtime.json"

"$PYTHON" "$LOCAL_REPO/build/prewarm_d3.py" \
  --parameter-file "$ALCHEMI_D3_PARAM_FILE" \
  --output "$FINAL_DIR/part1-d3-cache.json"
test -s "$ALCHEMI_D3_PARAM_FILE"
D3_PARAMETER_SHA256=$(sha256sum "$ALCHEMI_D3_PARAM_FILE")
test "${D3_PARAMETER_SHA256%% *}" = \
  "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
printf 'D3 parameter cache: %s\n' "$D3_PARAMETER_SHA256"

echo "== prewarm and verify SevenNet-Omni =="
"$PYTHON" "$LOCAL_REPO/build/prewarm_sevennet.py" --device cuda

echo "== complete Part 1 notebook: 5,000 NVT + 20,000 NVE =="
"$PYTHON" "$LOCAL_REPO/scripts/run_notebook_no_timeout.py" \
  "$NOTEBOOK_REL" \
  --output "$LOCAL_EXECUTED" \
  --timing-output "$TIMING_PATH" \
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

"$PYTHON" "$LOCAL_REPO/scripts/validate_part1_ir_run.py" \
  --executed-notebook "$FINAL_DIR/alchemi-water-ir-executed.ipynb" \
  --output-dir "$FINAL_DIR/outputs/run-${SLURM_JOB_ID}" \
  --source-root "$LOCAL_REPO" \
  --summary "$FINAL_DIR/notebook-validation.json" \
  --checksums "$FINAL_DIR/SHA256SUMS"

cd "$FINAL_DIR"
sha256sum -c SHA256SUMS
nvidia-smi
echo "NOTEBOOK_COMPLETE"
