#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-nci
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN_ROOT="${ALCHEMI_RUN_ROOT:-${SCRATCH:-$HOME}/alchemi-ir}"
SHARED_REPO="${ALCHEMI_SHARED_REPO:-$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp}"
MAIN_ENV="${ALCHEMI_MAIN_ENV:-$RUN_ROOT/envs/alchemi-playbook}"
BASE_OVERLAY="${ALCHEMI_BASE_OVERLAY:-$RUN_ROOT/envs/part1-sevennet-v2-20260720-r3}"
NODE_TMP="/tmp/alchemi-nci-${SLURM_JOB_ID}"
LOCAL_REPO="$NODE_TMP/ALCHEMI-Bootcamp"
CURRENT_CORE="$NODE_TMP/current-core"
KERNEL_PREFIX="$NODE_TMP/kernel"
PART_REL="part-1-scalable-atomistic-workflows"
NOTEBOOK_REL="$PART_REL/alchemi-water-ir.ipynb"
RUN_ID="nci-${SLURM_JOB_ID}"
LOCAL_OUTPUT="$LOCAL_REPO/$PART_REL/outputs/run-${RUN_ID}"
FINAL_DIR="$RUN_ROOT/results/nci-stage3-${SLURM_JOB_ID}"
TRANSCRIPT="$FINAL_DIR/slurm-${SLURM_JOB_ID}.log"
SHARED_AIMNET_CACHE="$RUN_ROOT/cache/aimnet"
LOCAL_AIMNET_CACHE="$NODE_TMP/aimnet-cache"
SHARED_D3_PARAMETER="$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt"
LOCAL_D3_PARAMETER="$NODE_TMP/dftd3_parameters.pt"
PYTHON="$MAIN_ENV/bin/python"
UV="$MAIN_ENV/bin/uv"

CORE_COMMIT="331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
CORE_REQUIREMENT="nvalchemi-toolkit[ase,aimnet] @ git+https://github.com/NVIDIA/nvalchemi-toolkit.git@${CORE_COMMIT}"
BASE_OVERLAY_SITE="$BASE_OVERLAY/lib/python3.12/site-packages"

export ALCHEMI_RUN_ID="$RUN_ID"
export ALCHEMI_NCI_RESULT_DIR="$FINAL_DIR"
export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export XDG_DATA_HOME="$RUN_ROOT/cache/xdg-data"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export AIMNET_CACHE_DIR="$SHARED_AIMNET_CACHE"
export ALCHEMI_D3_PARAM_FILE="$SHARED_D3_PARAMETER"
export TMPDIR="$NODE_TMP/t"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MPLCONFIGDIR="$NODE_TMP/mpl"
export WARP_CACHE_PATH="$NODE_TMP/wp"
export CUDA_CACHE_PATH="$NODE_TMP/cu"
export TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/ti"
export TRITON_CACHE_DIR="$NODE_TMP/tr"
export UV_CACHE_DIR="$NODE_TMP/uv"
export JUPYTER_DATA_DIR="$NODE_TMP/jd"
export JUPYTER_CONFIG_DIR="$NODE_TMP/jc"
export OVITO_GUI_MODE=headless
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
export PATH="$MAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$MAIN_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

test -x "$PYTHON"
test -x "$UV"
test -d "$BASE_OVERLAY_SITE"
test -s "$BASE_OVERLAY/.part1-ready.json"
test -d "$SHARED_REPO"

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" \
  "$TORCH_HOME" "$SHARED_AIMNET_CACHE" "$LOCAL_AIMNET_CACHE" \
  "$TMPDIR" "$MPLCONFIGDIR" \
  "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$UV_CACHE_DIR" "$JUPYTER_DATA_DIR" \
  "$JUPYTER_CONFIG_DIR" "$FINAL_DIR"

exec 3>&1
exec > >(tee -a "$TRANSCRIPT" >&3) 2>&1
TRANSCRIPT_TEE_PID=$!

preserve_run() {
  status=$?
  set +e
  if [[ -f "$LOCAL_OUTPUT/nci_interaction_curves.png" ]]; then
    cp "$LOCAL_OUTPUT/nci_interaction_curves.png" "$FINAL_DIR/"
  fi
  if [[ -f "$LOCAL_REPO/$NOTEBOOK_REL" ]]; then
    cp "$LOCAL_REPO/$NOTEBOOK_REL" "$FINAL_DIR/alchemi-water-ir-source.ipynb"
  fi
  if [[ -f "$LOCAL_REPO/scripts/measure_part1_nci_stage.py" ]]; then
    cp "$LOCAL_REPO/scripts/measure_part1_nci_stage.py" "$FINAL_DIR/"
  fi
  printf '%s\n' "$status" > "$FINAL_DIR/exit-status.txt"
  date -Is
  printf 'NCI_JOB_EXIT_STATUS=%s\n' "$status"
  exec 1>&3 2>&3
  wait "$TRANSCRIPT_TEE_PID"
  (
    cd "$FINAL_DIR" || exit 1
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
  trap - EXIT
  exit "$status"
}
trap preserve_run EXIT

echo "== allocation =="
date -Is
hostname
printf 'job=%s cpus=%s tmp=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$NODE_TMP"
nvidia-smi
nvidia-smi -q -x > "$FINAL_DIR/gpu-before.xml"

echo "== copy the exact source to the compute node =="
rsync -a "$SHARED_REPO/" "$LOCAL_REPO/"

echo "== install the declared current Toolkit Core pin in a node-local layer =="
mkdir -p "$CURRENT_CORE"
"$UV" pip install --target "$CURRENT_CORE" --no-deps "$CORE_REQUIREMENT"
export PYTHONPATH="$CURRENT_CORE:$BASE_OVERLAY_SITE"

echo "== install the notebook kernel =="
"$PYTHON" -m ipykernel install \
  --prefix "$KERNEL_PREFIX" \
  --name alchemi-nci \
  --display-name "ALCHEMI NCI timing"
export JUPYTER_PATH="$KERNEL_PREFIX/share/jupyter"

echo "== record source identities =="
cd "$LOCAL_REPO"
(
  find "$PART_REL" -type f ! -path "*/outputs/*" -print0
  printf '%s\0' \
    "scripts/rebuild_part1_ir_notebook.py" \
    "scripts/measure_part1_nci_stage.py" \
    "scripts/slurm_part1_nci_stage3.sbatch" \
    "build/requirements.txt" \
    "build/verify_part1_runtime.py" \
    "build/prewarm_aimnet.py"
) | sort -z | xargs -0 sha256sum > "$FINAL_DIR/source-files.sha256"

test -s "$ALCHEMI_D3_PARAM_FILE"
test "$(sha256sum "$ALCHEMI_D3_PARAM_FILE" | cut -d ' ' -f 1)" = \
  "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"

echo "== verify current package pins and CUDA =="
"$PYTHON" build/verify_part1_runtime.py \
  --source-root "$LOCAL_REPO" \
  --require-cuda \
  --output "$FINAL_DIR/part1-runtime.json"

echo "== resolve the prefix checkpoint and four NCI checkpoints before timing =="
"$PYTHON" build/prewarm_aimnet.py \
  --device cpu \
  --output "$FINAL_DIR/aimnet-shared-checkpoints.json"

echo "== move checkpoint and D3 reads to node-local storage =="
CHECKPOINT_FILES=(
  "aimnet2_2025_b973c_d3_0.pt"
  "aimnet2_wb97m_d3_0.pt"
  "aimnet2_wb97m_d3_1.pt"
  "aimnet2_wb97m_d3_2.pt"
  "aimnet2_wb97m_d3_3.pt"
)
for checkpoint_file in "${CHECKPOINT_FILES[@]}"; do
  test -s "$SHARED_AIMNET_CACHE/$checkpoint_file"
  cp "$SHARED_AIMNET_CACHE/$checkpoint_file" "$LOCAL_AIMNET_CACHE/"
done
test -s "$SHARED_D3_PARAMETER"
cp "$SHARED_D3_PARAMETER" "$LOCAL_D3_PARAMETER"
export AIMNET_CACHE_DIR="$LOCAL_AIMNET_CACHE"
export ALCHEMI_D3_PARAM_FILE="$LOCAL_D3_PARAMETER"

test "$(sha256sum "$ALCHEMI_D3_PARAM_FILE" | cut -d ' ' -f 1)" = \
  "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
"$PYTHON" build/prewarm_aimnet.py \
  --device cpu \
  --output "$FINAL_DIR/aimnet-checkpoints.json"

echo "== execute the real notebook prefix and measure Stage 3 =="
"$PYTHON" scripts/measure_part1_nci_stage.py \
  "$NOTEBOOK_REL" \
  --executed-prefix "$FINAL_DIR/nci-stage3-executed-prefix.ipynb" \
  --report "$FINAL_DIR/nci-stage3-timing.json" \
  --cell-times "$FINAL_DIR/nci-stage3-cell-times.csv" \
  --checkpoint-report "$FINAL_DIR/aimnet-checkpoints.json" \
  --kernel alchemi-nci

echo "== final checks =="
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["ALCHEMI_NCI_RESULT_DIR"])
report = json.loads((result_dir / "nci-stage3-timing.json").read_text())
science = report["scientific_results"]
assert report["success"] is True
assert report["stage_cell_count"] == 6
assert science["graph_count"] == 90
assert science["ensemble_member_count"] == 4
assert science["max_complete_mae_vs_dft_d3_kcal_mol"] < 0.5
assert science["max_complete_mae_vs_ccsd_t_cbs_kcal_mol"] < 0.5
force_check = science["force_check"]
checks = science["checks"]
assert science["method_settings"]["force_check"]["finite_difference_step_A"] == 0.003
assert (
    force_check[
        "official_analytic_vs_official_finite_difference_abs_error_eV_A"
    ]
    < checks[
        "official_analytic_vs_official_total_energy_finite_difference"
    ]["atol_eV_A"]
)
assert (
    force_check["toolkit_analytic_vs_official_analytic_abs_error_eV_A"]
    < checks["toolkit_analytic_vs_official_analytic"]["atol_eV_A"]
)
print(json.dumps({
    "nci_stage_cell_wall_s": report["nci_stage_cell_wall_s"],
    "nci_stage_outer_wall_s": report["nci_stage_outer_wall_s"],
    "graph_count": science["graph_count"],
    "atom_count": science["atom_count"],
}, indent=2))
PY
nvidia-smi
nvidia-smi -q -x > "$FINAL_DIR/gpu-after.xml"
echo "NCI_STAGE3_COMPLETE"
