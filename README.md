# ALCHEMI Playbook

Hands-on tutorials for building batched atomistic workflows with NVIDIA
ALCHEMI Toolkit. The playbook focuses on the Toolkit data model, model adapters,
GPU execution, relaxation and dynamics, hooks, model composition, and saved
results.

The curriculum is Toolkit-first. Scientific examples provide meaningful inputs
and outputs, but they do not limit which reusable ALCHEMI capabilities the
series teaches.

## Tutorials

| Tutorial | Focus |
|---|---|
| [Part 1: From one structure to scalable atomistic workflows](part-1-scalable-atomistic-workflows/) | Follow one seven-stage path from a single result through NCI Atlas batching, model composition, a custom materials-model adapter, dynamics, infrared analysis, inflight queues, and spatial domain decomposition. |
| [Part 2: Batched adsorption](part-2-batched-adsorption-toolkit/) | Enumerate and relax adsorbate-surface candidates with a materials model. |
| [Part 3: OLED melting-point prediction](part-3-batched-melting-toolkit/) | Use a molecular potential in a solid-liquid coexistence workflow. |

These part numbers and directory names are the permanent curriculum order. The
archived `research-toolkit-foundations` notebook contains the research version
of the NCI Atlas lesson now incorporated into Part 1. It is not included in the
Part 1 image, is not an active tutorial, and is not a source of part numbering.

The v2 image targets the remastered Part 1. The retained adsorption notebook,
now Part 2, requires its separate historical MACE environment. The retained
OLED notebook, now Part 3 and historically Part 2, requires its separate
historical Orb environment. The Part 1 image intentionally does not install
`orb-models` or the legacy-only `loguru` dependency. The retained notebooks
still need updated environments and validation before learner use.

## What the playbook teaches

- Convert ASE or pymatgen structures to `AtomicData`.
- Combine variable-sized systems in a `Batch` and recover individual results.
- Inspect model inputs, outputs, precision, and neighbor requirements.
- Use public Toolkit model adapters and connect a new model through the adapter
  interface.
- Build neighbors through Toolkit Core and understand the accelerated
  Toolkit-Ops implementations underneath.
- Compose independent and dependent model contributions without hiding the
  data flow.
- Run batched relaxation and dynamics with hooks for neighbors, checks,
  logging, and saved snapshots.
- Explain single-call batching, inflight processing, spatial domain
  decomposition, and distributed stage pipelines as different execution
  patterns.
- Measure CPU and GPU behavior with equal work, teach the intended multi-GPU
  workflow, and analyze verified multi-GPU results when they are available.

## Requirements

| Requirement | Details |
|---|---|
| Background | Python, plus basic computational chemistry, atomistic simulation, or machine-learning experience |
| Host | x86_64 Linux host with an NVIDIA GPU |
| Container runtime | Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit |
| Storage and network | Enough space for the image and model checkpoints; network access during image build and the first D3 setup, unless a checked D3 cache is supplied |

The main remastered notebook is paced for an H100-class GPU. It can be opened
on weaker hardware, but runtimes will differ. The notebook does not silently
shorten the declared scientific workload to fit a smaller device.

## Start JupyterLab

From the repository root:

```bash
cd build
docker compose up
```

Open `http://localhost:8888/lab`, then launch
`part-1-scalable-atomistic-workflows/alchemi-water-ir.ipynb`. Other tutorial
folders remain visible for reference, but they need the separate environments
listed above.

Compose bind-mounts the local tutorial directories so edits appear immediately;
it is the development path, not the clean-image release check. Rebuild after
changing package pins. To inspect the files baked into the image with no host
source mounted over them, stop Compose and run the command below. The clean
v2 image contains the remastered Part 1 plus the Part 2/3 status READMEs; it
does not package their historical notebooks or scientific data.

```bash
docker run --rm --gpus all --shm-size=8g \
  -p 8888:8888 alchemi-playbook:latest
```

The image definition and package pins live in [build/Dockerfile](build/Dockerfile),
[build/environment.yml](build/environment.yml), and
[build/requirements.txt](build/requirements.txt).

## Documentation

| Document | Purpose |
|---|---|
| [Fundamental tutorial design principles](TUTORIAL_DESIGN_PRINCIPLES.md) | General rules derived from strong PyTorch, TorchSim, Warp, BioNeMo, and large-scale computing tutorials |
| [ALCHEMI tutorial principles and visual style](ALCHEMI_TUTORIAL_PRINCIPLES.md) | Toolkit-first curriculum, public API exposure, live compute, helper-code, and notebook-style rules |
| [Toolkit API curriculum](TOOLKIT_API_CURRICULUM.md) | Separate list of the Toolkit capabilities and public APIs the playbook should cover |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Software, model, checkpoint, and redistribution notes |
| [Changelog](CHANGELOG.md) | User-visible changes to the playbook |

Exact calculation records, hardware measurements, checksums, and failed-run
details belong beside the saved results they describe. They are not part of the
root project overview or the tutorial-design guides.

## Official resources

- [ALCHEMI developer hub](https://developer.nvidia.com/cuda/cuda-x-libraries/alchemi)
- [Toolkit documentation](https://nvidia.github.io/nvalchemi-toolkit/)
- [Toolkit source](https://github.com/NVIDIA/nvalchemi-toolkit)
- [Toolkit-Ops documentation](https://nvidia.github.io/nvalchemi-toolkit-ops/)
- [Toolkit-Ops source](https://github.com/NVIDIA/nvalchemi-toolkit-ops)

## License

This repository is licensed under Apache 2.0. See [LICENSE](LICENSE).

Models, checkpoints, datasets, figures, and downloaded runtime files may have
separate terms. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
README beside each input before redistribution.
