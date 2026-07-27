#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART2="$ROOT/part-2-batched-adsorption-toolkit"
OUTDIR="${1:-$PART2/outputs/toolkit_acceleration_matrix}"
PYTHON="${PYTHON:-$ROOT/.venv-toolkit/bin/python}"
RUNTIME_CACHE_DIR="${PART2}/outputs/runtime_cache"

mkdir -p "$OUTDIR"/logs "$RUNTIME_CACHE_DIR"/warp "$RUNTIME_CACHE_DIR"/cueq_triton

export WARP_CACHE_DIR="${WARP_CACHE_DIR:-$RUNTIME_CACHE_DIR/warp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$RUNTIME_CACHE_DIR/warp}"
export CUEQ_TRITON_CACHE_DIR="${CUEQ_TRITON_CACHE_DIR:-$RUNTIME_CACHE_DIR/cueq_triton}"
export LD_LIBRARY_PATH="$ROOT/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

run_case() {
  local compile_model="$1"
  local enable_cueq="$2"
  local tag="compile${compile_model}_cueq${enable_cueq}"

  echo "=== $tag: H2O fixed-step throughput ==="
  "$PYTHON" "$PART2/scripts/benchmark_h2o_saturation.py" \
    --batch-sizes 256,1024 \
    --repeats 1 \
    --n-steps 20 \
    --fmax 0.0 \
    --output "$OUTDIR/h2o_${tag}.json" \
    "$compile_model" \
    "$enable_cueq" \
    2>&1 | tee "$OUTDIR/logs/h2o_${tag}.log"

  echo "=== $tag: adsorption fixed-step throughput ==="
  "$PYTHON" "$PART2/scripts/benchmark_adsorption_batching.py" \
    --profile oxide_h2o \
    --batch-sizes 24 \
    --repeats 1 \
    --n-steps 20 \
    --fmax 0.0 \
    --output "$OUTDIR/adsorption_${tag}.json" \
    "$compile_model" \
    "$enable_cueq" \
    2>&1 | tee "$OUTDIR/logs/adsorption_${tag}.log"
}

run_case --no-compile-model --no-enable-cueq
run_case --compile-model --no-enable-cueq
run_case --no-compile-model --enable-cueq
run_case --compile-model --enable-cueq

"$PYTHON" - <<'PY' "$OUTDIR"
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
rows = []
for path in sorted(outdir.glob("h2o_compile*_cueq*.json")):
    tag = path.stem.removeprefix("h2o_")
    data = json.loads(path.read_text())
    for row in data:
        rows.append({
            "benchmark": "h2o",
            "tag": tag,
            "batch_size": row.get("batch_size"),
            "compile_model": row.get("compile_model"),
            "enable_cueq": row.get("enable_cueq"),
            "init_time_s": row.get("model_load_time_s"),
            "wall_time_s": row.get("median_wall_time_s"),
            "structures_per_s": row.get("median_structures_per_s"),
            "peak_memory_gb": row.get("median_peak_memory_gb"),
            "optimizer_steps": row.get("median_optimizer_steps"),
            "status": row.get("status"),
        })
for path in sorted(outdir.glob("adsorption_compile*_cueq*.json")):
    tag = path.stem.removeprefix("adsorption_")
    data = json.loads(path.read_text())
    for row in data:
        rows.append({
            "benchmark": "adsorption_oxide_h2o",
            "tag": tag,
            "batch_size": row.get("batch_size"),
            "compile_model": row.get("compile_model"),
            "enable_cueq": row.get("enable_cueq"),
            "init_time_s": row.get("backend_init_time_s"),
            "wall_time_s": row.get("wall_time_s"),
            "structures_per_s": row.get("structures_per_s"),
            "peak_memory_gb": row.get("peak_memory_gb"),
            "optimizer_steps": row.get("optimizer_steps"),
            "status": row.get("status"),
        })

summary = outdir / "summary.csv"
with summary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "benchmark", "tag", "batch_size", "compile_model", "enable_cueq",
        "init_time_s", "wall_time_s", "structures_per_s", "peak_memory_gb",
        "optimizer_steps", "status",
    ])
    writer.writeheader()
    writer.writerows(rows)
print(summary)
PY
