# Runtime Snapshot

This document records what goes into the updated main Docker environment. It is a reproducibility snapshot, not a floating "latest" statement.

## Image base

- Base image: `nvidia/cuda:13.2.0-runtime-ubuntu24.04`
- Conda environment: `/opt/conda/envs/alchemi-playbook`
- Python: `3.12.13`
- Torch wheel source: `https://download.pytorch.org/whl/cu130`
- Jupyter kernel shipped by the image: `alchemi-main` (`ALCHEMI Main`)
- Jupyter server default kernel: `alchemi-main`

The image intentionally removes the default `python3`, `ovito-pro`, and old `alchemi-barebones` kernelspecs so notebooks bind to one playbook kernel. It also sets `c.KernelSpecManager.ensure_native_kernel = False` and `c.MappingKernelManager.default_kernel_name = "alchemi-main"` in the environment-level Jupyter config; otherwise Jupyter can synthesize a second native `python3` kernel and report `python3` as the API default even after the installed kernelspec is removed.

## Pinned Git snapshots

- `nvalchemi-toolkit`: `01c99d5cde6f63d6f662b071a9f408d3bfc12b0a`
- `nvalchemi-toolkit-ops`: `2b7c3c3adfb1ca84b886eecbf14bc60ff6ba1dc2`

`build/requirements.txt` is the source of truth for these pins. The Dockerfile installs with `uv pip install --no-sources-package nvalchemi-toolkit-ops` because the current Toolkit main snapshot still carries an older internal `tool.uv.sources` pin for Toolkit-Ops; the direct URL in `requirements.txt` is the intended resolver input.

Part 1 applies explicit neighbor-list dispatch for these pinned Toolkit and Toolkit-Ops snapshots. The helper layer selects `batch_naive` or `batch_cell_list` to match the pre-allocated neighbor-list buffers used by the Toolkit relaxation path.

## Core package versions

- `torch`: `2.12.0+cu130`
- `nvalchemi-toolkit`: `0.1.0` from the pinned Git commit above
- `nvalchemi-toolkit-ops`: `0.3.1` from the pinned Git commit above
- `ovito`: `3.15.4`
- `mace-torch`: `0.3.15`
- `cuequivariance`: `0.10.0`
- `cuequivariance-torch`: `0.10.0`
- `cuequivariance-ops-cu13`: `0.10.0`
- `cuequivariance-ops-torch-cu13`: `0.10.0`
- `ase`: `3.28.0`
- `pymatgen`: `2026.5.4`
- `pydantic`: `2.13.4`
- `jupyterlab`: `4.5.8`
- `ipykernel`: `7.3.0`

## Build entrypoints

- Conda package spec: `build/environment.yml`
- Python/runtime package spec: `build/requirements.txt`
- Docker recipe: `build/Dockerfile`
- Compose service: `build/docker-compose.yml`
