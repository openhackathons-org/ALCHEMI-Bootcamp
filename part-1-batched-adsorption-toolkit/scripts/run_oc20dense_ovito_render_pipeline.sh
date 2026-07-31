#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

OUTPUT="${1:-part-1-batched-adsorption/outputs/ovito_dft_toolkit_pairs}"
SPP="${2:-64}"
PYTHON_BIN="${PYTHON_BIN:-.venv-toolkit/bin/python}"
OVITOS_BIN="${OVITOS_BIN:-/mnt/c/Program Files/OVITO Pro/ovitos.exe}"

case "${OUTPUT}" in
  part-1-batched-adsorption/outputs/*|./part-1-batched-adsorption/outputs/*|/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/*) ;;
  *)
    printf 'Refusing to write outside part-1-batched-adsorption/outputs/: %s\n' "${OUTPUT}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" part-1-batched-adsorption/scripts/prepare_oc20dense_ovito_render_inputs.py \
  --output "${OUTPUT}"

JOBS_WIN="$(wslpath -w "$(pwd)/${OUTPUT}/prepared/render_jobs.csv")"
OUTPUT_WIN="$(wslpath -w "$(pwd)/${OUTPUT}")"

"${OVITOS_BIN}" part-1-batched-adsorption/scripts/render_prepared_oc20dense_pairs_ovito.py \
  --jobs "${JOBS_WIN}" \
  --output "${OUTPUT_WIN}" \
  --renderer visrtx \
  --samples-per-pixel "${SPP}"

"${PYTHON_BIN}" part-1-batched-adsorption/scripts/compose_oc20dense_ovito_render_panels.py \
  --jobs "${OUTPUT}/prepared/render_jobs.csv" \
  --output "${OUTPUT}"

mv "${OUTPUT}/prepared/render_jobs.csv" "${OUTPUT}/render_jobs.csv"
rm -rf "${OUTPUT}/prepared"

printf 'Rendered improved OVITO panels: %s\n' "${OUTPUT}/dft_toolkit_selected_contact_sheet.png"
