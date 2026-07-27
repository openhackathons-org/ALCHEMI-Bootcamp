#!/usr/bin/env bash
set -euo pipefail

PART_DIR="part-2-batched-adsorption-toolkit"
OUTPUT="${1:-${PART_DIR}/outputs/ovito_dft_toolkit_pairs}"
SPP="${2:-64}"
PYTHON_BIN="${PYTHON_BIN:-.venv-toolkit/bin/python}"
OVITOS_BIN="${OVITOS_BIN:-/mnt/c/Program Files/OVITO Pro/ovitos.exe}"

case "${OUTPUT}" in
  "${PART_DIR}"/outputs/*|./"${PART_DIR}"/outputs/*|"$(pwd)/${PART_DIR}"/outputs/*) ;;
  *)
    printf 'Refusing to write outside %s/outputs/: %s\n' "${PART_DIR}" "${OUTPUT}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" \
  "${PART_DIR}/scripts/prepare_oc20dense_ovito_render_inputs.py" \
  --output "${OUTPUT}"

JOBS_WIN="$(wslpath -w "$(pwd)/${OUTPUT}/prepared/render_jobs.csv")"
OUTPUT_WIN="$(wslpath -w "$(pwd)/${OUTPUT}")"

"${OVITOS_BIN}" \
  "${PART_DIR}/scripts/render_prepared_oc20dense_pairs_ovito.py" \
  --jobs "${JOBS_WIN}" \
  --output "${OUTPUT_WIN}" \
  --renderer visrtx \
  --samples-per-pixel "${SPP}"

"${PYTHON_BIN}" \
  "${PART_DIR}/scripts/compose_oc20dense_ovito_render_panels.py" \
  --jobs "${OUTPUT}/prepared/render_jobs.csv" \
  --output "${OUTPUT}"

mv "${OUTPUT}/prepared/render_jobs.csv" "${OUTPUT}/render_jobs.csv"
rm -rf "${OUTPUT}/prepared"

printf 'Rendered improved OVITO panels: %s\n' "${OUTPUT}/dft_toolkit_selected_contact_sheet.png"
