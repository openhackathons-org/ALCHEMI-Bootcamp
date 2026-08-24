#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

setup_started_epoch="$(date +%s)"
workspace_root="${ALCHEMI_BREV_WORKSPACE_ROOT:-${HOME}/workspace}"
course_root="${ALCHEMI_BREV_COURSE_ROOT:-${workspace_root}/ALCHEMI-Bootcamp}"
runtime_root="${ALCHEMI_RUNTIME_ROOT:-${workspace_root}/.alchemi-runtime}"
bootstrap_root="${runtime_root}/bootstrap"
uv_version="0.12.1"
expected_revision="${ALCHEMI_BREV_EXPECTED_REVISION:-}"

if [[ ! "${expected_revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ALCHEMI_BREV_EXPECTED_REVISION must be the 40-character release commit." >&2
    exit 2
fi

script_parent="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_candidates=(
    "${ALCHEMI_BREV_SOURCE_ROOT:-}"
    "${script_parent}"
    "${PWD}"
    "${HOME}/ALCHEMI-Bootcamp"
    "${workspace_root}/ALCHEMI-Bootcamp"
)
source_root=""
for candidate in "${source_candidates[@]}"; do
    if [[ -n "${candidate}" && -f "${candidate}/uv.lock" ]]; then
        source_root="$(cd "${candidate}" && pwd)"
        break
    fi
done

if [[ -z "${source_root}" ]]; then
    echo "Could not find the cloned ALCHEMI-Bootcamp repository." >&2
    echo "Set ALCHEMI_BREV_SOURCE_ROOT to its absolute path." >&2
    exit 2
fi

source_revision="$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if [[ ! -e "${course_root}" && "${source_revision}" != "${expected_revision}" ]]; then
    echo "Course revision mismatch: expected ${expected_revision}, found ${source_revision}" >&2
    echo "Set the Launchable revision to the attached repository commit and deploy a new instance." >&2
    exit 2
fi

mkdir -p "${workspace_root}" "${runtime_root}"

if [[ "${source_root}" != "${course_root}" ]]; then
    if [[ ! -e "${course_root}" ]]; then
        echo "Copying the course into persistent Brev workspace storage."
        cp -a "${source_root}" "${course_root}"
    elif [[ ! -f "${course_root}/uv.lock" ]]; then
        echo "Course path exists without uv.lock: ${course_root}" >&2
        exit 2
    fi
fi

revision="$(git -C "${course_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if [[ "${revision}" != "${expected_revision}" ]]; then
    echo "Course revision mismatch: expected ${expected_revision}, found ${revision}" >&2
    echo "This instance keeps its existing course files. Deploy a new instance for a new release." >&2
    exit 2
fi

ready_file="${runtime_root}/brev-ready.txt"
rm -f "${ready_file}"

uv_bin="${ALCHEMI_UV_BIN:-${bootstrap_root}/bin/uv}"
if [[ ! -x "${uv_bin}" ]]; then
    echo "Installing uv ${uv_version} for the Brev runtime."
    python3 -m venv "${bootstrap_root}"
    "${bootstrap_root}/bin/python" -m pip install \
        --disable-pip-version-check \
        "uv==${uv_version}"
    uv_bin="${bootstrap_root}/bin/uv"
fi

actual_uv_version="$("${uv_bin}" --version | awk '{print $2}')"
if [[ "${actual_uv_version}" != "${uv_version}" ]]; then
    echo "uv version mismatch: expected ${uv_version}, found ${actual_uv_version}" >&2
    exit 2
fi

export ALCHEMI_RUNTIME_ROOT="${runtime_root}"
export ALCHEMI_INSTALL_PROFILE="brev"
export ALCHEMI_UV_BIN="${uv_bin}"
export UV_PYTHON_INSTALL_DIR="${runtime_root}/python"

echo "Installing the frozen ALCHEMI environment."
"${course_root}/scripts/setup" --sync-only

host_jupyter_data_dir="${ALCHEMI_BREV_HOST_JUPYTER_DATA_DIR:-${HOME}/.local/share/jupyter}"
kernel_root="${ALCHEMI_BREV_KERNEL_ROOT:-${host_jupyter_data_dir}/kernels}"
kernel_directory="${kernel_root}/alchemi-core"
kernel_json="${kernel_directory}/kernel.json"
mkdir -p "${kernel_directory}"

job_root="$(mktemp -d "${runtime_root}/brev-setup-logs.XXXXXX")"
cuda_device_file="${job_root}/cuda-device.txt"
job_names=()
job_pids=()
job_logs=()

cleanup_jobs() {
    local pid
    for pid in "${job_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup_jobs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting asset prewarming and runtime verification."
(
    cd "${course_root}"
    "${course_root}/scripts/run" python - <<'PY'
from environment import check_runtime, prewarm_assets

prewarm_assets.main()
check_runtime.main()
PY
) > "${job_root}/assets-runtime.log" 2>&1 &
job_names+=("assets-runtime")
job_pids+=("$!")
job_logs+=("${job_root}/assets-runtime.log")

echo "Starting widget frontend installation."
"${course_root}/scripts/run" \
    python "${course_root}/environment/install_widget_frontend.py" \
    --jupyter-data-dir "${host_jupyter_data_dir}" \
    > "${job_root}/widgets.log" 2>&1 &
job_names+=("widgets")
job_pids+=("$!")
job_logs+=("${job_root}/widgets.log")

echo "Starting CUDA access and compiler verification."
(
    cd "${course_root}"
    "${course_root}/scripts/run" python - "${cuda_device_file}" <<'PY'
import sys
from pathlib import Path

import torch

from environment import check_cuda_compile

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the Brev tutorial runtime")

cuda_device = torch.cuda.get_device_name(0)
check_cuda_compile.main()
Path(sys.argv[1]).write_text(cuda_device + "\n")
print(f"CUDA device: {cuda_device}")
PY
) > "${job_root}/cuda.log" 2>&1 &
job_names+=("cuda")
job_pids+=("$!")
job_logs+=("${job_root}/cuda.log")

echo "Starting Jupyter kernel registration."
python3 - \
    "${kernel_json}" \
    "${course_root}/scripts/run" \
    "${runtime_root}" \
    "${uv_bin}" > "${job_root}/kernel.log" 2>&1 <<'PY' &
import json
import os
import sys
from pathlib import Path

kernel_json = Path(sys.argv[1])
runner = str(Path(sys.argv[2]).resolve())
runtime_root = str(Path(sys.argv[3]).resolve())
uv_bin = str(Path(sys.argv[4]).resolve())

specification = {
    "argv": [
        runner,
        "python",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ],
    "display_name": "Python 3 (ALCHEMI Core)",
    "language": "python",
    "metadata": {"debugger": True},
    "env": {
        "ALCHEMI_RUNTIME_ROOT": runtime_root,
        "ALCHEMI_UV_BIN": uv_bin,
    },
}

temporary_json = kernel_json.with_suffix(".json.tmp")
temporary_json.write_text(json.dumps(specification, indent=2) + "\n")
os.replace(temporary_json, kernel_json)
PY
job_names+=("kernel")
job_pids+=("$!")
job_logs+=("${job_root}/kernel.log")

job_failed=false
for index in "${!job_pids[@]}"; do
    name="${job_names[$index]}"
    pid="${job_pids[$index]}"
    log="${job_logs[$index]}"
    if wait "${pid}"; then
        echo "Brev setup job completed: ${name}"
    else
        status="$?"
        echo "Brev setup job failed: ${name} (exit ${status})" >&2
        job_failed=true
    fi
    cat "${log}"
done

trap - EXIT INT TERM
if [[ "${job_failed}" == true ]]; then
    exit 1
fi

cuda_device="$(<"${cuda_device_file}")"

setup_finished_epoch="$(date +%s)"
setup_seconds="$((setup_finished_epoch - setup_started_epoch))"
runtime_kib="$(du -sk "${runtime_root}" | awk '{print $1}')"
ready_temporary="${ready_file}.tmp.$$"
{
    printf 'status=ready\n'
    printf 'revision=%s\n' "${revision}"
    printf 'course_root=%s\n' "${course_root}"
    printf 'runtime_root=%s\n' "${runtime_root}"
    printf 'kernel=%s\n' "${kernel_json}"
    printf 'widget_frontend=%s\n' \
        "${host_jupyter_data_dir}/labextensions/@jupyter-widgets/jupyterlab-manager"
    printf 'gpu=%s\n' "${cuda_device}"
    printf 'setup_seconds=%s\n' "${setup_seconds}"
    printf 'runtime_kib=%s\n' "${runtime_kib}"
} > "${ready_temporary}"
mv "${ready_temporary}" "${ready_file}"

echo "ALCHEMI is ready."
echo "Course: ${course_root}"
echo "Runtime: ${runtime_root}"
echo "Jupyter kernel: Python 3 (ALCHEMI Core)"
echo "Widget frontend: ${host_jupyter_data_dir}/labextensions/@jupyter-widgets/jupyterlab-manager"
echo "Setup time: ${setup_seconds} seconds"
echo "Runtime disk use: ${runtime_kib} KiB"
echo "Ready record: ${ready_file}"
