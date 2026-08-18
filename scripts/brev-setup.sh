#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

workspace_root="${ALCHEMI_BREV_WORKSPACE_ROOT:-${HOME}/workspace}"
course_root="${ALCHEMI_BREV_COURSE_ROOT:-${workspace_root}/ALCHEMI-Bootcamp}"
runtime_root="${ALCHEMI_RUNTIME_ROOT:-${workspace_root}/.alchemi-runtime}"
bootstrap_root="${runtime_root}/bootstrap"
uv_version="0.11.1"

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
expected_revision="${ALCHEMI_BREV_EXPECTED_REVISION:-}"
if [[ -n "${expected_revision}" && "${revision}" != "${expected_revision}" ]]; then
    echo "Course revision mismatch: expected ${expected_revision}, found ${revision}" >&2
    exit 2
fi

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
export ALCHEMI_UV_BIN="${uv_bin}"
export UV_PYTHON_INSTALL_DIR="${runtime_root}/python"

echo "Installing the frozen ALCHEMI environment."
"${course_root}/scripts/setup"

echo "Checking package pins and prewarmed assets."
"${course_root}/scripts/run" \
    python "${course_root}/environment/check_runtime.py"

echo "Checking CUDA access."
"${course_root}/scripts/run" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the Brev tutorial runtime")

print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

kernel_root="${ALCHEMI_BREV_KERNEL_ROOT:-${HOME}/.local/share/jupyter/kernels}"
kernel_directory="${kernel_root}/python3"
kernel_json="${kernel_directory}/kernel.json"
mkdir -p "${kernel_directory}"

python3 - \
    "${kernel_json}" \
    "${course_root}/scripts/run" \
    "${runtime_root}" \
    "${uv_bin}" <<'PY'
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
    "display_name": "Python 3 (ALCHEMI)",
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

ready_file="${runtime_root}/brev-ready.txt"
{
    printf 'status=ready\n'
    printf 'revision=%s\n' "${revision}"
    printf 'course_root=%s\n' "${course_root}"
    printf 'runtime_root=%s\n' "${runtime_root}"
    printf 'kernel=%s\n' "${kernel_json}"
} > "${ready_file}"

echo "ALCHEMI is ready."
echo "Course: ${course_root}"
echo "Runtime: ${runtime_root}"
echo "Jupyter kernel: Python 3 (ALCHEMI)"
echo "Ready record: ${ready_file}"
