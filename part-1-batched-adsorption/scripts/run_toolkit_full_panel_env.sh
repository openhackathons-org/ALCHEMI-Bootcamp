#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

python_bin="${TOOLKIT_PYTHON:-${repo_root}/.venv-toolkit/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Toolkit runner requires ${python_bin}; create .venv-toolkit and install tutorial dependencies first." >&2
  exit 2
fi

export PYTHONPATH="${repo_root}/part-1-batched-adsorption${PYTHONPATH:+:${PYTHONPATH}}"
export TOOLKIT_REQUIRE_D3BJ="${TOOLKIT_REQUIRE_D3BJ:-0}"
export TOOLKIT_COMPILE_MODEL="${TOOLKIT_COMPILE_MODEL:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TOOLKIT_N_STEPS="${TOOLKIT_N_STEPS:-200}"
export FULL_PANEL_CHUNK_SIZE="${FULL_PANEL_CHUNK_SIZE:-12}"
runtime_cache="${repo_root}/part-1-batched-adsorption/outputs/runtime_cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${runtime_cache}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${runtime_cache}/torch}"
export HF_HOME="${HF_HOME:-${runtime_cache}/hf}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${runtime_cache}/warp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${runtime_cache}/matplotlib}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

opengl_lib="${repo_root}/.local-debs/libopengl0/usr/lib/x86_64-linux-gnu"
cuda_lib="${repo_root}/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "${opengl_lib}" && -d "${cuda_lib}" ]]; then
  export LD_LIBRARY_PATH="${opengl_lib}:${cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${HF_HOME}" "${WARP_CACHE_PATH}" "${MPLCONFIGDIR}"

exec "${python_bin}" part-1-batched-adsorption/scripts/run_toolkit_full_panel.py "$@"
