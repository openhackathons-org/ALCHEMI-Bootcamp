#!/usr/bin/env bash
#
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
#
# Reproduce the uv-locked Python environment and generate the committed
# third-party package license inventory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
DEFAULT_ENV_DIR="${TMPDIR:-/tmp}/alchemi-license-inventory"
ENV_DIR="${ENV_DIR:-$DEFAULT_ENV_DIR}"
OUT_DIR="$REPO_ROOT/.licenses"

if [[ "$ENV_DIR" != /* ]]; then
  ENV_DIR="$REPO_ROOT/$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
MARKER="${ENV_DIR}.managed-by-alchemi-license-inventory"

if [[ "${REUSE_ENV:-0}" == "1" ]]; then
  if [[ ! -x "$PY" ]]; then
    echo "Inventory environment is missing $PY" >&2
    exit 2
  fi
  echo ">>> checking reused environment against pyproject.toml and uv.lock"
  UV_PROJECT_ENVIRONMENT="$ENV_DIR" "$UV_BIN" sync \
    --locked \
    --all-groups \
    --no-install-project \
    --check
else
  case "$(basename "$ENV_DIR")" in
    *license*) ;;
    *)
      echo "Refusing to synchronize a non-license environment: $ENV_DIR" >&2
      exit 2
      ;;
  esac
  if [[ -e "$ENV_DIR" && ! -f "$MARKER" ]]; then
    echo "Refusing to modify an unmarked environment: $ENV_DIR" >&2
    echo "Use REUSE_ENV=1 to inspect an existing locked environment." >&2
    exit 2
  fi

  printf '%s\n' "managed by .licenses/generate_licenses.sh" > "$MARKER"
  echo ">>> synchronizing disposable inventory environment at $ENV_DIR"
  UV_PROJECT_ENVIRONMENT="$ENV_DIR" "$UV_BIN" sync \
    --locked \
    --all-groups \
    --no-install-project \
    --python "$PYTHON_VERSION"
fi

echo ">>> checking installed dependency compatibility"
"$UV_BIN" pip check --python "$PY"

echo ">>> rendering direct-dependencies.md, summary.md, details.json, and Third_party_attr.txt"
"$PY" "$SCRIPT_DIR/generate_licenses.py" \
  --repo-root "$REPO_ROOT" \
  --out-dir "$OUT_DIR"

echo ">>> inventory complete"
