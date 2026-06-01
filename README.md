# ALCHEMI Playbook

This playbook gives researchers hands-on, GPU-accelerated workflows for computational chemistry and materials discovery with NVIDIA ALCHEMI. Across two tutorials, participants run a batched adsorption-site search over catalyst surfaces and predict the melting point of a molecular crystal — both running entirely from a single Docker container with JupyterLab at port 8888.

## Playbook contents

The content is structured in two parts:

- **[Part 1: Batched Adsorption-Site Search with ALCHEMI Toolkit](part-1-batched-adsorption/)** — run an AdsorbML-style adsorption-energy workflow on the GPU: enumerate many plausible adsorbate–surface configurations, then relax them together as a single batch with the ALCHEMI Toolkit and the MACE-MPA-0 (`medium-mpa-0`) foundation model, using FIRE2 geometry optimization. The notebook builds up from a batched H₂O "hello world", checks the model against released OC20Dense adsorption data, then searches sites and orientations across the active surface panel.
- **[Part 2: OLED Melting Point Predictions with ALCHEMI Toolkit](part-2-toolkit/)** — predict molecular-crystal melting points via the Solid–Liquid Coexistence (SLC) pipeline using the ALCHEMI Toolkit Python library and the AIMNet2 neural network potential, with naphthalene as a model OLED material.

## Tools and frameworks

The tools and frameworks used in this playbook:

- [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) — Python library for batched, GPU-native atomistic relaxation and dynamics
- [MACE-MPA-0](https://github.com/ACEsuit/mace) — materials foundation model (machine-learned interatomic potential) used for the adsorption search in Part 1
- [AIMNet2](https://github.com/isayevlab/AIMNet2) — neural network potential used for molecular dynamics in Part 2
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
| Internet | Needed during the initial container build, image pulls, and the first download of the MACE-MPA-0 checkpoint. |

## Deploying the Playbook

Both parts run from a single unified Docker container orchestrated via Docker Compose.

```bash
cd build
docker compose up          # builds the unified image and starts JupyterLab
```

Once running, the service is reachable at:

| Service    | URL                                  |
|------------|--------------------------------------|
| Jupyter    | http://localhost:8888/lab            |

Open the Jupyter URL in your browser and launch either notebook:

- `part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`
- `part-2-toolkit/melting-point-slc.ipynb`

### Browsing without live GPU work

The Part 1 notebook's run-configuration cell exposes a `RUN_SCOPE` toggle — `"short"` runs one representative adsorption example with six starting structures, `"full"` runs the complete adsorption grid — and a result-source toggle, where `"saved"` reads pre-computed results so you can step through the tutorial without waiting on the GPU. These are useful in workshop settings with limited GPU availability.

## License

Apache 2.0 — see [LICENSE](LICENSE).
