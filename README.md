# ALCHEMI Playbook

This playbook gives researchers hands-on, GPU-accelerated workflows for computational chemistry and materials discovery with NVIDIA ALCHEMI. Across two tutorials, participants run a batched adsorption-site search over catalyst surfaces and predict the melting point of a molecular crystal — both running entirely from a single Docker container with JupyterLab at port 8888.

## Playbook contents

The content is structured in two parts:

- **[Part 1: Batched Adsorption-Site Search with ALCHEMI Toolkit](part-1-batched-adsorption-toolkit/)** — run an AdsorbML-style adsorption-energy workflow on the GPU: enumerate many plausible adsorbate–surface configurations, then relax them together as a single batch with the ALCHEMI Toolkit and the MACE-MPA-0 (`medium-mpa-0`) foundation model, using FIRE2 geometry optimization. The notebook builds up from a batched H₂O "hello world", checks the model against released OC20Dense adsorption data, then searches sites and orientations across the active surface panel.
- **[Part 2: OLED Melting Point Predictions with ALCHEMI Toolkit](part-2-batched-melting-toolkit/)** — predict molecular-crystal melting points via the Solid–Liquid Coexistence (SLC) pipeline using the ALCHEMI Toolkit Python library and the Orb-v3 (OMol) machine-learned interatomic potential, with naphthalene as a model OLED material.

## Tools and frameworks

The tools and frameworks used in this playbook:

- [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) — Python library for batched, GPU-native atomistic relaxation and dynamics
- [NVIDIA ALCHEMI Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) — GPU kernels (neighbor lists, DFT-D3, long-range electrostatics) under the Toolkit
- [MACE-MPA-0](https://github.com/ACEsuit/mace) — materials foundation model (machine-learned interatomic potential) used for the adsorption search in Part 1
- [Orb-v3](https://github.com/orbital-materials/orb-models) — machine-learned interatomic potential used for molecular dynamics in Part 2
- [OVITO](https://www.ovito.org/) — atomistic visualization
- [JupyterLab](https://jupyterlab.readthedocs.io/) — interactive notebook environment

## Resources

- **ALCHEMI:** [developer hub](https://developer.nvidia.com/cuda/cuda-x-libraries/alchemi) · [Toolkit docs](https://nvidia.github.io/nvalchemi-toolkit/) · [Toolkit-Ops docs](https://nvidia.github.io/nvalchemi-toolkit-ops/)
- **Source (GitHub):** [nvalchemi-toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) · [nvalchemi-toolkit-ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
- **pip:** `pip install nvalchemi-toolkit` ([PyPI](https://pypi.org/project/nvalchemi-toolkit/)) — see the [Toolkit docs](https://nvidia.github.io/nvalchemi-toolkit/) for GPU wheels and optional extras (`[ase,mace,aimnet]`). This playbook's Docker image installs the pinned build for you (see [RUNTIME_SNAPSHOT.md](RUNTIME_SNAPSHOT.md)).

## Runtime snapshot

The updated main environment is pinned for reproducible Linux x86_64 rebuilds.
The CUDA base image is fixed by digest, the Miniforge installer is fixed by
version and SHA-256, and all Conda, Python, and Git dependencies are recorded
in checked-in lock files. The locally built Docker image exposes a single
Jupyter kernel, `alchemi-main` (`ALCHEMI Main`), backed by the
`/opt/conda/envs/alchemi-playbook` Python environment. The complete build
snapshot is recorded in [RUNTIME_SNAPSHOT.md](RUNTIME_SNAPSHOT.md).

## Playbook duration

Approximately **90–120 minutes per part** (~3–4 hours total).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Background | Python proficiency; basic familiarity with computational chemistry / atomistic simulation. |
| GPU host | NVIDIA x86_64 GPU. Tested on A100, H100, B200, L40S, RTX 6000 Ada. |
| Docker | Latest [Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and the Docker Compose v2 plugin. |
| Internet | Needed during the initial container build, image pulls, and the first download of the model checkpoints (MACE-MPA-0 for Part 1, Orb-v3 for Part 2). |

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

- `part-1-batched-adsorption-toolkit/alchemi-mace-adsorption-search.ipynb`
- `part-2-batched-melting-toolkit/melting-point-slc.ipynb`

### Browsing without live GPU work

The Part 1 notebook's run-configuration cell exposes a `RUN_SCOPE` toggle —
`"short"` runs one representative adsorption example with six starting
structures, `"full"` runs the complete adsorption grid — and a result-source
toggle, where `"saved"` reads pre-computed results so you can step through the
tutorial without waiting on the GPU.

NOTE: Part 2's pre-computed results have been removed while the workflow is
being regenerated from the COD crystal input. Its saved-result mode is
temporarily unavailable.

## Developers who shaped the tutorials

- **[Ryan Reese](https://github.com/Ryan-Reese)** — creator of the melting-point study using the NVIDIA ALCHEMI Toolkit.
- **[Anoushka Bhutani](https://github.com/anoushka2000)** — developer of companion challenges for the tutorials. These challenges will be integrated in a future update.

## License

Except where otherwise noted, the original code and documentation in this
repository are licensed under the [Apache License 2.0](LICENSE).

This repository distributes tutorial source files and a Docker build recipe.
It does not distribute a prebuilt container image or model checkpoints.
Building or running the tutorial downloads third-party software and models
from their official sources under their respective terms.

The NVIDIA ALCHEMI Toolkit and NVIDIA ALCHEMI Toolkit-Ops are separate projects
governed by their own licenses. Third-party software and other materials remain
subject to the terms supplied by their official sources or distributions. See
[Sources and licenses](SOURCES_AND_LICENSES.md) for distributed scientific data
and visual assets, the
[direct dependency license inventory](.licenses/direct-dependencies.md), and
[complete package attributions](.licenses/Third_party_attr.txt) generated for
this tutorial environment.

No rights are granted to NVIDIA trademarks or branding.
