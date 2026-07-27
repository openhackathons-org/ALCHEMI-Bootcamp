#!/usr/bin/env bash
#SBATCH --job-name=alchemi-part1-setup
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

# Do not write import bytecode into the staged tutorial checkout.
export PYTHONDONTWRITEBYTECODE=1

# Submit from the login node with absolute source and log paths. The package
# layer must name a new directory because this job will not replace one:
#
#   export ALCHEMI_RUN_ROOT=/shared/alchemi
#   export ALCHEMI_SHARED_REPO="$ALCHEMI_RUN_ROOT/stage/repo/ALCHEMI-Bootcamp"
#   export ALCHEMI_TUTORIAL_COMMIT=<full-40-character-Git-SHA>
#   export ALCHEMI_CONDA_EXE=/absolute/path/to/conda
#   export ALCHEMI_MAIN_ENV="$ALCHEMI_RUN_ROOT/envs/alchemi-playbook"
#   export ALCHEMI_PYTHON_OVERLAY="$ALCHEMI_RUN_ROOT/envs/part1-current"
#   mkdir -p "$ALCHEMI_RUN_ROOT/logs"
#   sbatch \
#     --chdir="$ALCHEMI_RUN_ROOT/logs" \
#     --output="$ALCHEMI_RUN_ROOT/logs/part1-setup-%j.out" \
#     "$ALCHEMI_SHARED_REPO/scripts/slurm_part1_sevennet_setup.sbatch"
#
: "${ALCHEMI_RUN_ROOT:?set ALCHEMI_RUN_ROOT to the Compute Lab run directory}"
: "${ALCHEMI_SHARED_REPO:?set ALCHEMI_SHARED_REPO to the staged source directory}"
: "${ALCHEMI_TUTORIAL_COMMIT:?set the full tutorial Git commit to run}"
: "${ALCHEMI_CONDA_EXE:?set ALCHEMI_CONDA_EXE to an absolute Conda executable}"
: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to a new or valid immutable Conda environment}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to a new directory}"

RUN_ROOT="$ALCHEMI_RUN_ROOT"
SHARED_REPO="$ALCHEMI_SHARED_REPO"
TUTORIAL_COMMIT="$ALCHEMI_TUTORIAL_COMMIT"
CONDA_EXE="$ALCHEMI_CONDA_EXE"
MAIN_ENV_REQUESTED="$ALCHEMI_MAIN_ENV"
PYTHON_OVERLAY_REQUESTED="$ALCHEMI_PYTHON_OVERLAY"
MAIN_ENV="$(realpath -m "$MAIN_ENV_REQUESTED")"
PYTHON_OVERLAY="$(realpath -m "$PYTHON_OVERLAY_REQUESTED")"
export ALCHEMI_MAIN_ENV="$MAIN_ENV"
export ALCHEMI_PYTHON_OVERLAY="$PYTHON_OVERLAY"
RESULT_DIR="$RUN_ROOT/results/part1-environment-${SLURM_JOB_ID}"
TRANSCRIPT_PATH="$RESULT_DIR/setup-${SLURM_JOB_ID}.log"
READY_FILE="$PYTHON_OVERLAY/.part1-ready.json"
RUNTIME_REPORT="$RESULT_DIR/runtime-check.json"
ENVIRONMENT_SPEC="$SHARED_REPO/build/environment.yml"
REQUIREMENTS="$SHARED_REPO/build/requirements.txt"
BASE_SPEC_MARKER="$MAIN_ENV/.part1-environment.sha256"

export HOME="$RUN_ROOT/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export XDG_DATA_HOME="$RUN_ROOT/cache/xdg-data"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export AIMNET_CACHE_DIR="$RUN_ROOT/cache/aimnet"
export ALCHEMI_D3_PARAM_FILE="$RUN_ROOT/home/.cache/nvalchemiops/dftd3_parameters.pt"
export OVITO_GUI_MODE=headless
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" \
  "$TORCH_HOME" "$AIMNET_CACHE_DIR" "$RESULT_DIR"

exec > >(tee -a "$TRANSCRIPT_PATH") 2>&1

echo "== allocation =="
date -Is
hostname
nvidia-smi

echo "== input directories =="
test -d "$SHARED_REPO"
if [[ ! "$TUTORIAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ALCHEMI_TUTORIAL_COMMIT must be a full lowercase 40-character Git SHA" >&2
  exit 2
fi
test "$(git -C "$SHARED_REPO" rev-parse HEAD)" = "$TUTORIAL_COMMIT"
test -z "$(git -C "$SHARED_REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(
  git -C "$SHARED_REPO" ls-files --others --ignored --exclude-standard
)"
test -f "$ENVIRONMENT_SPEC"
test -f "$REQUIREMENTS"
ENVIRONMENT_SPEC_SHA256="$(sha256sum "$ENVIRONMENT_SPEC" | cut -d ' ' -f 1)"
if [[ "$CONDA_EXE" != /* || ! -x "$CONDA_EXE" ]]; then
  echo "ALCHEMI_CONDA_EXE must be an absolute executable path: $CONDA_EXE" >&2
  exit 2
fi
CONDA_EXE="$(realpath "$CONDA_EXE")"
if [[ -L "$MAIN_ENV_REQUESTED" && ! -e "$MAIN_ENV_REQUESTED" ]]; then
  echo "Refusing broken Conda-base symlink: $MAIN_ENV_REQUESTED" >&2
  exit 2
fi
if [[ "$PYTHON_OVERLAY/" == "$MAIN_ENV/"* ]] || \
   [[ "$MAIN_ENV/" == "$PYTHON_OVERLAY/"* ]]; then
  echo "The Conda base and Python package layer must use separate paths" >&2
  exit 2
fi
if [[ -e "$PYTHON_OVERLAY_REQUESTED" ]] || \
   [[ -L "$PYTHON_OVERLAY_REQUESTED" ]] || \
   [[ -e "$PYTHON_OVERLAY" ]] || \
   [[ -L "$PYTHON_OVERLAY" ]]; then
  echo "Refusing to replace existing package layer: $PYTHON_OVERLAY_REQUESTED" >&2
  exit 2
fi

echo "== create or verify the immutable Conda base =="
BASE_CREATED=false
if [[ -e "$MAIN_ENV" || -L "$MAIN_ENV" ]]; then
  if [[ ! -d "$MAIN_ENV/conda-meta" ]] || \
     [[ ! -f "$MAIN_ENV/conda-meta/history" ]] || \
     [[ ! -x "$MAIN_ENV/bin/python" ]] || \
     [[ ! -x "$MAIN_ENV/bin/uv" ]] || \
     [[ ! -x "$MAIN_ENV/bin/packmol" ]] || \
     [[ ! -x "$MAIN_ENV/bin/dot" ]] || \
     [[ ! -s "$BASE_SPEC_MARKER" ]] || \
     [[ "$(< "$BASE_SPEC_MARKER")" != "$ENVIRONMENT_SPEC_SHA256" ]]; then
    echo "Refusing existing partial or invalid Conda base: $MAIN_ENV" >&2
    exit 2
  fi
else
  BASE_CREATED=true
  mkdir -p "$(dirname "$MAIN_ENV")"
  "$CONDA_EXE" env create \
    --prefix "$MAIN_ENV" \
    --file "$ENVIRONMENT_SPEC"
fi

test -d "$MAIN_ENV/conda-meta"
test -f "$MAIN_ENV/conda-meta/history"
test -x "$MAIN_ENV/bin/python"
test -x "$MAIN_ENV/bin/uv"
test -x "$MAIN_ENV/bin/packmol"
test -x "$MAIN_ENV/bin/dot"
"$MAIN_ENV/bin/python" - "$MAIN_ENV/conda-meta" <<'PY'
import json
from pathlib import Path
import sys

records = {}
for path in Path(sys.argv[1]).glob("*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    name = record.get("name")
    if name:
        records[name] = record.get("version")

expected_exact = {
    "ovito": "3.15.4",
    "packmol": "21.2.1",
    "python": "3.12.13",
}
for name, expected in expected_exact.items():
    if records.get(name) != expected:
        raise SystemExit(
            f"invalid Conda base: expected {name} {expected}, "
            f"found {records.get(name)!r}"
        )
uv_text = str(records.get("uv", ""))
try:
    uv_version = tuple(int(part) for part in uv_text.split(".")[:3])
except ValueError as error:
    raise SystemExit(
        f"invalid Conda base: could not parse uv version {uv_text!r}"
    ) from error
if len(uv_version) != 3 or uv_version < (0, 9, 26):
    raise SystemExit(
        "invalid Conda base: uv 0.9.26 or newer is required, "
        f"found {uv_text!r}"
    )
if "graphviz" not in records:
    raise SystemExit("invalid Conda base: missing graphviz")
PY
if [[ "$BASE_CREATED" == true ]]; then
  BASE_SPEC_MARKER_TMP="$BASE_SPEC_MARKER.tmp-${SLURM_JOB_ID}"
  printf '%s\n' "$ENVIRONMENT_SPEC_SHA256" > "$BASE_SPEC_MARKER_TMP"
  mv "$BASE_SPEC_MARKER_TMP" "$BASE_SPEC_MARKER"
fi

export PATH="$MAIN_ENV/bin:$PATH"
printf 'source=%s\nconda=%s\nbase=%s\npackage_layer=%s\n' \
  "$SHARED_REPO" "$CONDA_EXE" "$MAIN_ENV" "$PYTHON_OVERLAY"
"$MAIN_ENV/bin/uv" --version | tee "$RESULT_DIR/uv-version.txt"

echo "== create the Part 1 Python package layer =="
"$MAIN_ENV/bin/uv" venv \
  --no-project \
  --no-managed-python \
  --system-site-packages \
  --python "$MAIN_ENV/bin/python" \
  "$PYTHON_OVERLAY"

"$MAIN_ENV/bin/uv" pip install \
  --python "$PYTHON_OVERLAY/bin/python" \
  --no-managed-python \
  --no-sources-package nvalchemi-toolkit-ops \
  --torch-backend cu130 \
  --requirements "$REQUIREMENTS"

"$MAIN_ENV/bin/uv" pip check \
  --python "$PYTHON_OVERLAY/bin/python"

OVERLAY_SITE="$PYTHON_OVERLAY/lib/python3.12/site-packages"
test -d "$OVERLAY_SITE"
PYTHON="$PYTHON_OVERLAY/bin/python"
TORCHRUN="$PYTHON_OVERLAY/bin/torchrun"
JUPYTER="$PYTHON_OVERLAY/bin/jupyter"
test -x "$PYTHON"
test -x "$TORCHRUN"
test -x "$JUPYTER"
export LD_LIBRARY_PATH="$OVERLAY_SITE/nvidia/cu13/lib:$MAIN_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== verify imports, versions, Git revisions, and CUDA =="
"$PYTHON" "$SHARED_REPO/build/verify_part1_runtime.py" \
  --source-root "$SHARED_REPO" \
  --require-cuda \
  --require-clean-source \
  --output "$RUNTIME_REPORT"

echo "== download and check AIMNet checkpoints =="
"$PYTHON" "$SHARED_REPO/build/prewarm_aimnet.py" \
  --device cuda \
  --output "$RESULT_DIR/aimnet-cache.json"

echo "== create and check the shared D3 parameter cache =="
"$PYTHON" "$SHARED_REPO/build/prewarm_d3.py" \
  --parameter-file "$ALCHEMI_D3_PARAM_FILE" \
  --output "$RESULT_DIR/d3-cache.json"

echo "== download and check SevenNet-Omni =="
"$PYTHON" "$SHARED_REPO/build/prewarm_sevennet.py" --device cuda

echo "== run maintained Part 1 and reference tests, excluding retired OrbMol tests =="
export WARP_CACHE_PATH="$RESULT_DIR/warp-cache"
"$PYTHON" -m pytest -q \
  -o "cache_dir=$RESULT_DIR/pytest-cache" \
  --ignore="$SHARED_REPO/part-1-scalable-atomistic-workflows/tests/test_orbmol_adapter.py" \
  --ignore="$SHARED_REPO/part-1-scalable-atomistic-workflows/tests/test_orbmol_checks.py" \
  --ignore="$SHARED_REPO/part-1-scalable-atomistic-workflows/tests/test_orbmol_config.py" \
  "$SHARED_REPO/part-1-scalable-atomistic-workflows/tests" \
  "$SHARED_REPO/part-1-scalable-atomistic-workflows/reference/tests" \
  | tee "$RESULT_DIR/pytest.txt"

echo "== installed packages =="
"$CONDA_EXE" list --prefix "$MAIN_ENV" --explicit \
  > "$RESULT_DIR/conda-explicit.txt"
"$MAIN_ENV/bin/uv" pip freeze \
  --python "$PYTHON_OVERLAY/bin/python" \
  > "$RESULT_DIR/python-packages.txt"
sha256sum \
  "$RUNTIME_REPORT" \
  "$RESULT_DIR/aimnet-cache.json" \
  "$RESULT_DIR/d3-cache.json" \
  "$RESULT_DIR/pytest.txt" \
  "$RESULT_DIR/conda-explicit.txt" \
  "$RESULT_DIR/python-packages.txt" \
  "$RESULT_DIR/uv-version.txt" \
  "$ENVIRONMENT_SPEC" \
  "$SHARED_REPO/build/requirements.txt" \
  > "$RESULT_DIR/SHA256SUMS"

nvidia-smi
READY_TMP="$PYTHON_OVERLAY/.part1-ready.json.tmp-${SLURM_JOB_ID}"
cp "$RUNTIME_REPORT" "$READY_TMP"
mv "$READY_TMP" "$READY_FILE"
echo "PART1_ENVIRONMENT_READY=$PYTHON_OVERLAY"
