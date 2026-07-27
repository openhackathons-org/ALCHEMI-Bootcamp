#!/usr/bin/env bash
#SBATCH --job-name=alchemi-nci-force
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
NODE_TMP="/tmp/alchemi-nci-force-${SLURM_JOB_ID}"
LOCAL_REPO="$NODE_TMP/ALCHEMI-Bootcamp"
CURRENT_CORE="$NODE_TMP/current-core"
KERNEL_PREFIX="$NODE_TMP/kernel"
LOCAL_AIMNET_CACHE="$NODE_TMP/aimnet-cache"
LOCAL_D3_PARAMETER="$NODE_TMP/dftd3_parameters.pt"
PART_REL="part-1-scalable-atomistic-workflows"
NOTEBOOK_REL="$PART_REL/alchemi-water-ir.ipynb"
FINAL_DIR="$RUN_ROOT/results/nci-force-diagnostic-${SLURM_JOB_ID}"
PYTHON="$MAIN_ENV/bin/python"
UV="$MAIN_ENV/bin/uv"
CORE_COMMIT="331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
BASE_OVERLAY_SITE="$BASE_OVERLAY/lib/python3.12/site-packages"

export HOME="$RUN_ROOT/home"
export AIMNET_CACHE_DIR="$LOCAL_AIMNET_CACHE"
export ALCHEMI_D3_PARAM_FILE="$LOCAL_D3_PARAMETER"
export ALCHEMI_RUN_ID="nci-force-${SLURM_JOB_ID}"
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
export PATH="$MAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$MAIN_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
  "$FINAL_DIR" "$LOCAL_AIMNET_CACHE" "$TMPDIR" "$MPLCONFIGDIR" \
  "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$UV_CACHE_DIR" "$JUPYTER_DATA_DIR" \
  "$JUPYTER_CONFIG_DIR"
exec > >(tee -a "$FINAL_DIR/slurm-${SLURM_JOB_ID}.log") 2>&1

echo "== allocation =="
date -Is
hostname
nvidia-smi

echo "== node-local source and current Core =="
rsync -a "$SHARED_REPO/" "$LOCAL_REPO/"
"$UV" pip install \
  --target "$CURRENT_CORE" \
  --no-deps \
  "nvalchemi-toolkit[ase,aimnet] @ git+https://github.com/NVIDIA/nvalchemi-toolkit.git@${CORE_COMMIT}"
export PYTHONPATH="$CURRENT_CORE:$BASE_OVERLAY_SITE"

echo "== node-local model data =="
for checkpoint_file in \
  aimnet2_2025_b973c_d3_0.pt \
  aimnet2_wb97m_d3_0.pt \
  aimnet2_wb97m_d3_1.pt \
  aimnet2_wb97m_d3_2.pt \
  aimnet2_wb97m_d3_3.pt
do
  cp "$RUN_ROOT/cache/aimnet/$checkpoint_file" "$LOCAL_AIMNET_CACHE/"
done
cp "$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt" "$LOCAL_D3_PARAMETER"

"$PYTHON" -m ipykernel install \
  --prefix "$KERNEL_PREFIX" \
  --name alchemi-nci-force \
  --display-name "ALCHEMI NCI force diagnostic"
export JUPYTER_PATH="$KERNEL_PREFIX/share/jupyter"

cd "$LOCAL_REPO"
"$PYTHON" build/verify_part1_runtime.py \
  --source-root "$LOCAL_REPO" \
  --require-cuda \
  --output "$FINAL_DIR/part1-runtime.json"
"$PYTHON" build/prewarm_aimnet.py \
  --device cpu \
  --output "$FINAL_DIR/aimnet-checkpoints.json"

echo "== force component and step sweep =="
"$PYTHON" scripts/diagnose_part1_nci_force.py \
  "$NOTEBOOK_REL" \
  --output "$FINAL_DIR/nci-force-diagnostic.json" \
  --executed-prefix "$FINAL_DIR/nci-force-executed-prefix.ipynb" \
  --kernel alchemi-nci-force

cp "$LOCAL_REPO/$NOTEBOOK_REL" "$FINAL_DIR/alchemi-water-ir-source.ipynb"
cp scripts/diagnose_part1_nci_force.py "$FINAL_DIR/"
cd "$FINAL_DIR"
find . -type f ! -name SHA256SUMS ! -name 'slurm-*.log' -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
nvidia-smi
echo "NCI_FORCE_DIAGNOSTIC_COMPLETE"
