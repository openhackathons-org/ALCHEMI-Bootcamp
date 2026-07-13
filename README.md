# ALCHEMI Playbook

This v2 branch retains the existing ALCHEMI tutorials and adds the implemented
water-based Part 1 rebuild. The Part 3 notebook remains available as a
research and runtime-validation harness. Everything runs from one Docker
container with JupyterLab at port 8888.

## Playbook contents

The checkout currently contains these tutorial notebooks:

- **[Part 1 rebuild: One Water Dimer to a Batched IR Trajectory](part-1-water-hydrogen-bonding-toolkit/)** — compose the AIMNet2-2025 B97-3c residual with pairwise D3(BJ) and its official finite-system all-pairs `simple` Coulomb convention, then compute live predicted-charge spectra for H₂O/D₂O monomers and cyclic-hexamer seeds in one NVT→NVE batch, with topology and thermal-state gates on every comparison.
- **[Legacy adsorption tutorial — former Part 1](part-1-batched-adsorption-toolkit/)** — enumerate and relax adsorbate–surface candidates with MACE-MPA-0 and Toolkit FIRE2.
- **[Part 2: OLED Melting Point Predictions with ALCHEMI Toolkit](part-2-batched-melting-toolkit/)** — study solid–liquid coexistence for naphthalene with the Orb-v3 molecular potential.
- **[Retained research/validation prototype: Atoms to Batched GPU Workflows](part-3-toolkit-foundations/)** — compare CPU/GPU and homogeneous/heterogeneous batching, inspect neighbor-buffer tradeoffs, implement a custom Toolkit model, and compose AIMNet2 ωB97M-D3 with pairwise D3(BJ) and finite nonperiodic electrostatics. The complete model is evaluated live on three ten-point NCI Atlas interaction curves against near-matched DFT-D3 and independent CCSD(T)/CBS references. This is not the polished learner-facing Part 1 story.

## Tools and frameworks

The tools and frameworks used in this playbook:

- [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) — Python library for batched, GPU-native atomistic relaxation and dynamics
- [NVIDIA ALCHEMI Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) — GPU kernels (neighbor lists, DFT-D3, long-range electrostatics) under the Toolkit
- [MACE-MPA-0](https://github.com/ACEsuit/mace) — materials foundation model used for the Part 1 adsorption search
- [Orb-v3](https://github.com/orbital-materials/orb-models) — molecular potential used by the Part 2 melting workflow
- [AIMNet2-2025 B97-3c](https://huggingface.co/isayevlab/aimnet2-2025) — molecular potential used for the Part 1 predicted-charge IR trajectory
- [AIMNet2 ωB97M-D3](https://huggingface.co/isayevlab/aimnet2-wb97m-d3) — molecular potential used for the Part 3 component-composition example
- [OVITO](https://www.ovito.org/) — atomistic visualization
- [JupyterLab](https://jupyterlab.readthedocs.io/) — interactive notebook environment

## Resources

- **ALCHEMI:** [developer hub](https://developer.nvidia.com/cuda/cuda-x-libraries/alchemi) · [Toolkit docs](https://nvidia.github.io/nvalchemi-toolkit/) · [Toolkit-Ops docs](https://nvidia.github.io/nvalchemi-toolkit-ops/)
- **Source (GitHub):** [nvalchemi-toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) · [nvalchemi-toolkit-ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
- **pip:** `pip install nvalchemi-toolkit` ([PyPI](https://pypi.org/project/nvalchemi-toolkit/)) — see the [Toolkit docs](https://nvidia.github.io/nvalchemi-toolkit/) for GPU wheels and optional extras (`[ase,mace,aimnet]`). This playbook's Docker image installs the pinned build for you (see [RUNTIME_SNAPSHOT.md](RUNTIME_SNAPSHOT.md)).

## Runtime snapshot

The updated main environment is pinned for reproducible rebuilds. The Docker image exposes a single Jupyter kernel, `alchemi-main` (`ALCHEMI Main`), backed by the `/opt/conda/envs/alchemi-playbook` Python environment. The full package/commit snapshot is recorded in [RUNTIME_SNAPSHOT.md](RUNTIME_SNAPSHOT.md).

The Part 1 scientific path is acceptance-tested in the exact pinned CL H100
environment: scientific execution source SHA `5403dfcd…` passed in job
`3087665`. On the validation host, its intentionally gitignored local bundle at
`part-1-water-hydrogen-bonding-toolkit/outputs/h100-remaster-3087665/` contains
the source, executed notebook, validator report, full trajectory, and figures.
The current `v2` learner notebook SHA is `81124de…`: it adds the new
banner and presentation system, changes one CPU/GPU callout from `CHECK` to the
truthful `RESULT — OBSERVED`, and leaves the scientific code path unchanged. It
has not been rerun on H100. A clean build of the distributable Docker image
remains a separate release-image gate.

Part 1 keeps the variable-size dimer scan in eager mode. It applies default
`torch.compile` only after building the fixed 42-atom IR batch, then requires
the compiled energy, forces, and charges to match both an eager evaluation and
a second synchronized compiled evaluation before relaxation or dynamics.

## Playbook duration

Parts 1 and 2 retain their existing workshop scope. The retained Part 3
research harness does not yet have a validated workshop-duration target.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Background | Python proficiency; basic familiarity with computational chemistry / atomistic simulation. |
| GPU host | NVIDIA x86_64 GPU. Tested on A100, H100, B200, L40S, RTX 6000 Ada. |
| Docker | Latest [Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and the Docker Compose v2 plugin. |
| Internet | Needed during environment setup for image pulls, model checkpoints, and prewarming the Toolkit D3 parameter cache. The Part 1 notebook does not download D3 data while it runs. |

## Deploying the Playbook

All listed notebooks run from a single unified Docker container orchestrated
via Docker Compose.

```bash
cd build
docker compose up          # builds the unified image and starts JupyterLab
```

Once running, the service is reachable at:

| Service    | URL                                  |
|------------|--------------------------------------|
| Jupyter    | http://localhost:8888/lab            |

Open the Jupyter URL in your browser and launch any notebook:

- `part-1-water-hydrogen-bonding-toolkit/alchemi-water-ir.ipynb`
- `part-1-batched-adsorption-toolkit/alchemi-mace-adsorption-search.ipynb`
- `part-2-batched-melting-toolkit/melting-point-slc.ipynb`
- `part-3-toolkit-foundations/alchemi-toolkit-foundations.ipynb`

### Browsing without live GPU work

The legacy adsorption notebook exposes its original short/full and saved/compute controls. The new water-IR notebook always runs its full 5,000-step NVT + 50,000-step NVE live workload. Part 2 can replay the long production trajectories from shipped results. Part 3 is intentionally bounded live compute; its scientific outputs are not replayed from saved results.

## License

Apache 2.0 — see [LICENSE](LICENSE).

The Toolkit D3 parameter tensor is a separate runtime asset. Part 1 passes its
path explicitly with `param_file`, sets `auto_download=False`, and requires
SHA-256
`b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.
The tensor is not bundled in this repository while redistribution rights remain
unconfirmed.
