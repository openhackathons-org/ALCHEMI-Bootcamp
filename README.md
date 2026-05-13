# ALCHEMI Playbook

This playbook provides researchers hands-on approaches for GPU-accelerated computational chemistry and materials discovery with NVIDIA ALCHEMI. Across two tutorials, participants screen oxide catalysts for the oxygen-evolution reaction and predict the melting point of a molecular crystal — both running entirely from a single Docker container with JupyterLab at port 8888.

## Playbook contents

The content is structured in two parts:

- **[Part 1: OER Catalyst Screening with ALCHEMI NIMs](part-1-nim/)** — screen rutile oxide catalyst surfaces (IrO₂, RuO₂, TiO₂) for oxygen-evolution activity using the ALCHEMI BGR (Batch Geometry Relaxation) NIM and the MACE-MP-0 foundation model.
- **[Part 2: OLED Melting Point predictions with ALCHEMI Toolkit](part-2-toolkit/)** — predict molecular-crystal melting points via the Solid-Liquid Coexistence (SLC) pipeline using the ALCHEMI Toolkit Python library and the AIMNet2 neural network potential, with naphthalene as a model OLED material.

## Tools and frameworks

The tools and frameworks used in this playbook:

- [NVIDIA ALCHEMI BGR NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/alchemi-bgr?version=1.0.0) — batch geometry-relaxation microservice
- [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) — Python library for atomistic dynamics
- [MACE-MP-0](https://github.com/ACEsuit/mace) — materials foundation model used by the BGR NIM
- [AIMNet2](https://github.com/isayevlab/AIMNet2) — neural network potential for molecular dynamics
- [OVITO](https://www.ovito.org/) — atomistic visualization
- [JupyterLab](https://jupyterlab.readthedocs.io/) — interactive notebook environment

## Playbook duration

Approximately **90–120 minutes per part** (~3–4 hours total).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Background | Python proficiency; basic familiarity with computational chemistry / atomistic simulation. |
| GPU host | NVIDIA x86_64 GPU. Tested on A100, H100, B200, L40S, RTX 6000 Ada. |
| Docker | Latest [Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and the Docker Compose v2 plugin. |
| NGC API key | Required for Part 1's live BGR NIM service. Generate one at [build.nvidia.com](https://build.nvidia.com). Part 2 needs no NGC entitlement. |
| Internet | Needed during initial container build and image pulls. |

## Deploying the Playbook

Both parts run from a single unified Docker container alongside the BGR NIM service, Prometheus, and Grafana — all orchestrated via Docker Compose.

```bash
cd build
cp .env.example .env       # then edit .env and set NGC_API_KEY
docker compose up          # builds the unified image and starts all services
```

Once running, the services are reachable at:

| Service    | URL                                  |
|------------|--------------------------------------|
| Jupyter    | http://localhost:8888/lab            |
| BGR NIM    | http://localhost:8000/v1/...         |
| Grafana    | http://localhost:3000 (admin/admin)  |
| Prometheus | http://localhost:9090                |

Open the Jupyter URL in your browser and launch either notebook:

- `part-1-nim/alchemi-oer-catalyst-screening.ipynb`
- `part-2-toolkit/melting-point-slc.ipynb`

### Running Part 2 only (no NGC entitlement)

If you only want to work through Part 2, skip the BGR sidecar:

```bash
docker compose up jupyter prometheus grafana
```

### FAST_DEMO mode (offline, no live BGR)

Part 1's control-panel cell exposes a `FAST_DEMO = True` toggle. When enabled, the notebook reads pre-cached BGR responses from `part-1-nim/cached_responses/oer-catalyst-screening/` instead of calling the live BGR NIM. Useful for workshop settings without an NGC entitlement or with limited GPU availability.

## License

Apache 2.0 — see [LICENSE](LICENSE).
