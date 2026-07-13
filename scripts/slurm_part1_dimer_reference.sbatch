#!/usr/bin/env bash
#SBATCH --job-name=alchemi-dimer-b973c
#SBATCH --partition=h100-80gb-hbm3@ts6/h12dsgocpu/1gpu-48cpu-256gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Set ALCHEMI_RUN_ROOT for a site-specific persistent scratch location.
RUN_ROOT="${ALCHEMI_RUN_ROOT:-${SCRATCH:-$HOME}/alchemi-ir}"
REPO_ROOT="$RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
PART_DIR="$REPO_ROOT/part-1-water-hydrogen-bonding-toolkit"
REF_ENV="$RUN_ROOT/envs/alchemi-b973c-ir"
NODE_TMP="/tmp/alchemi-dimer-${SLURM_JOB_ID}"
LOCAL_OUT="$NODE_TMP/water_dimer_b97_3c"
FINAL_OUT="$RUN_ROOT/results/reference/water-dimer-b97-3c-${SLURM_JOB_ID}"

export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TMPDIR="$NODE_TMP/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export PATH="$REF_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

mkdir -p "$TMPDIR" "$LOCAL_OUT" "$FINAL_OUT"
exec > >(tee -a "$RUN_ROOT/results/reference/dimer-reference-${SLURM_JOB_ID}.log") 2>&1

preserve_reference() {
  status=$?
  set +e
  if [[ -d "$LOCAL_OUT" ]]; then
    rsync -a "$LOCAL_OUT/" "$FINAL_OUT/"
  fi
  date -Is
  echo "REFERENCE_EXIT_STATUS=$status"
  trap - EXIT
  exit "$status"
}
trap preserve_reference EXIT

date -Is
hostname
printf 'job=%s cpus=%s tmp=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$TMPDIR"
"$REF_ENV/bin/python" -c 'import ase, psi4; print("ASE", ase.__version__, "Psi4", psi4.__version__)'

"$REF_ENV/bin/python" \
  "$PART_DIR/reference/water_dimer_b97_3c.py" \
  --output "$LOCAL_OUT" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --memory "160 GB"

rsync -a "$LOCAL_OUT/" "$FINAL_OUT/"
PUBLISHED="$PART_DIR/reference/artifacts/water_dimer_b97_3c"
mkdir -p "$PUBLISHED"
rsync -a "$LOCAL_OUT/" "$PUBLISHED/"

"$REF_ENV/bin/python" -m pytest -q \
  "$PART_DIR/reference/tests/test_water_dimer_b97_3c.py"

echo "DIMER_REFERENCE_COMPLETE"
