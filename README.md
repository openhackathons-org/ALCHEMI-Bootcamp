# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

- **[Part 1: OER Catalyst Screening with ALCHEMI NIMs](part-1-nim/)** — screen rutile oxide catalyst surfaces (IrO₂, RuO₂, TiO₂) for oxygen-evolution activity using the ALCHEMI BGR NIM and MACE-MP-0.
- **[Part 2: ALCHEMI Toolkit Sandbox](part-2-toolkit/)** — melting-point validation of naphthalene via the Solid-Liquid Coexistence pipeline with the ALCHEMI Toolkit Python library (AIMNet2).

Both parts run from a single unified Docker container with Jupyter at port **8888**. All build artefacts (Dockerfile, docker-compose.yml, environment.yml, requirements files, scripts, monitoring config) live under [`build/`](build/) — the repo root contains only what tutorial attendees should see.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| NGC API key | Required for the BGR NIM sidecar (Part 1 live mode). Get one at [build.nvidia.com](https://build.nvidia.com). |
| Docker + Compose | `docker compose` v2 plugin. |
| GPU + host arch | **x86_64 NVIDIA GPU host.** OVITO ships only x86_64 builds on its conda channel, so aarch64 nodes (Grace+Hopper, Grace+Blackwell) are not supported. Tested on A100, H100, B200, L40S, RTX 6000 Ada. The `docker-dev.sh` helper aborts with a clear error if invoked on aarch64. |

## Quick Start (local Docker host)

```bash
cd dev/build
cp .env.example .env       # then edit .env and set NGC_API_KEY
docker compose up          # builds the unified image and starts all services
```

| Service    | Port | URL                              |
|------------|------|----------------------------------|
| Jupyter    | 8888 | http://localhost:8888/lab        |
| BGR NIM    | 8000 | http://localhost:8000/v1/...     |
| Grafana    | 3000 | http://localhost:3000 (admin/admin) |
| Prometheus | 9090 | http://localhost:9090            |

Both notebooks live in the same Jupyter:

- `part-1-nim/alchemi-oer-catalyst-screening.ipynb`
- `part-2-toolkit/melting-point-slc.ipynb`

The bind-mount is selective — students see only `part-1-nim/`, `part-2-toolkit/`, `README.md`, and `LICENSE` under `/workspace`. The `build/` dir and root-level helpers are hidden.

Both services reserve `count: all` GPUs. BGR NIM auto-balances inference across visible GPUs; Part 2 drivers take whichever GPUs are free. The two parts aren't expected to run concurrent GPU workloads on the same compute node.

If you only need Part 2 (no NGC, no BGR), launch a subset:

```bash
docker compose up jupyter prometheus grafana
```

## HPC / SLURM Quick Start

Allocate an x86_64 GPU node on a remote cluster (the `start-*` scripts at this repo root are dev-only SLURM helpers for specific clusters; copy the one that matches your cluster onto your login host and run it there):

```bash
ssh <login-host>
./start-b200          # or ./start-h200 (computelab x86_64); ./start-arch (aarch64, NOT compatible)
# note the compute node hostname
```

From your local machine, deploy onto the compute node via `build/scripts/deploy.sh`:

```bash
cd dev
cp build/.env.example build/.env       # then edit build/.env and set NGC_API_KEY
build/scripts/deploy.sh setup <login-host> <compute-node>
```

The script copies the repo to `/tmp/alchemi-playbook` on the compute node, builds the unified image, starts the compose stack, and opens SSH tunnels for Jupyter (8888) and Grafana (3000). Optional ports can be forwarded via env vars:

```bash
PROMETHEUS_LOCAL_PORT=9090 BGR_LOCAL_PORT=8000 build/scripts/deploy.sh setup ...
```

State is saved to `/tmp/alchemi-playbook-deploy.env`. Subsequent commands need no arguments:

```bash
build/scripts/deploy.sh status         # show URLs, BGR health, GPU visibility
build/scripts/deploy.sh restart        # sync local edits, rebuild stack (re-tars + recreates)
build/scripts/deploy.sh pull-changes   # pull remote notebook/helper edits back locally
build/scripts/deploy.sh stop           # tear down stack, close tunnels
```

## FAST_DEMO mode (no BGR, fully offline)

Part 1 supports `FAST_DEMO = True` (set in the control-panel cell) which reads pre-cached responses from `part-1-nim/cached_responses/oer-catalyst-screening/` instead of calling the BGR endpoint. Useful when no NGC entitlement is available. Part 2 is standalone and never needs BGR.

## Distribution as a single-tarball image

To bundle the unified Jupyter image (and optionally the BGR NIM image) for offline distribution:

```bash
cd dev/build
docker compose pull bgr
docker compose build jupyter
docker save \
  alchemi-playbook:latest \
  nvcr.io/nim/nvidia/alchemi-bgr:1.0.0 \
  | gzip > ../alchemi-bundle.tar.gz
```

Attendees load with `gunzip -c alchemi-bundle.tar.gz | docker load`, then run `docker compose up` from a separately-distributed copy of `build/` + `part-1-nim/` + `part-2-toolkit/` + `README.md` + `LICENSE`. They do not need to `docker login nvcr.io` — the locally-loaded BGR image takes precedence over the registry tag.

> **Open question for offline distribution:** the BGR NIM container receives `NGC_API_KEY` at runtime. Whether it phones home to NVIDIA license services with that key (vs. just using it for the initial pull) is empirically unknown. Verify by running `docker compose up bgr` on a machine with no NGC login and `NGC_API_KEY=invalid` before relying on the offline-bundle flow for an event.

## Repo layout

```
dev/
├── README.md                    # this file
├── LICENSE
├── part-1-nim/                  # tutorial content — NIM-based OER screening
├── part-2-toolkit/              # tutorial content — toolkit-based SLC
└── build/                       # everything used to build & deploy the container
    ├── Dockerfile
    ├── docker-compose.yml
    ├── environment.yml          # conda manifest (python 3.12, ovito, uv)
    ├── requirements.txt         # uv-installed runtime deps (toolkit, torch cu130, JupyterLab, ...)
    ├── requirements-dev.txt     # dev extras (toolkit-ops, pytest, ruff)
    ├── .env.example
    ├── monitoring/              # Prometheus + Grafana config
    └── scripts/
        ├── deploy.sh            # local orchestrator (SSH tunnels, tar+scp)
        └── docker-dev.sh        # remote compose driver
```

### Host venv setup (for `part-2-toolkit/tools/*.py`)

The host-side trajectory/plotting tools in `part-2-toolkit/tools/` run outside the container, so they need a matching Python env on the host. Uses the same `environment.yml` + `requirements.txt` pair as the Dockerfile:

```bash
cd dev/build
conda env create -f environment.yml         # python 3.12 + ovito + uv
conda activate alchemi-playbook
export UV_SYSTEM_PYTHON=true                # uv installs into the conda env's Python
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt -r requirements-dev.txt
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
