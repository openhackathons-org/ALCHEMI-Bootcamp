# ALCHEMI Toolkit tutorials v3

Hands-on tutorials for building batched atomistic workflows with NVIDIA
ALCHEMI Toolkit. The playbook focuses on the Toolkit data model, model adapters,
GPU execution, relaxation and dynamics, hooks, model composition, and saved
results.

The curriculum is Toolkit-first. Scientific examples provide meaningful inputs
and outputs, but they do not limit which reusable ALCHEMI capabilities the
series teaches.

## Core session

The ACS 2026 Chicago fundamentals session is paced for 90 minutes, including
transitions and discussion.

| Notebook | Live target | Focus |
|---|---:|---|
| 01 — AtomicData and Batch | 20 min | Build, validate, batch, inspect, and recover molecular graphs. |
| 02 — Data loading with Zarr | 12 min | Read individual records and stream Toolkit batches. |
| 03 — Model interfaces and composite potentials | 18 min | Wrap a model and combine learned, electrostatic, and dispersion terms. |
| 04 — Hooks | 12 min | Add reusable observation and behavior through the Hook protocol. |
| 05 — BaseDynamics | 18 min | Build a small batched optimizer from Toolkit dynamics pieces. |

The five notebooks are under active development in `notebooks/`. See the
[tutorial guide](TUTORIAL_GUIDE.md) for the curriculum and design system, and
[WORKLOG.md](WORKLOG.md) for ownership and current work.

GPU pipelines, training, and domain decomposition are planned as advanced
follow-up notebooks. Adsorption and melting notebooks are maintained on other
branches and are outside this release.

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
| Environment tool | `uv` |
| Storage and network | Scratch space for the locked environment and model caches; network access for first setup |

The live performance path is paced for an H100-class GPU. Runtimes on other
hardware will differ.

## Start JupyterLab

From the repository root, create the saved environment once:

```bash
./scripts/v3-sync
```

Then launch JupyterLab through the frozen environment:

```bash
./scripts/v3-run jupyter lab
```

The exact Python resolution is saved in [`uv.lock`](uv.lock). Immutable Toolkit,
Toolkit-Ops, model, and data identities are recorded in
[`environment/runtime-pins.toml`](environment/runtime-pins.toml).

## Documentation

| Document | Purpose |
|---|---|
| [ALCHEMI tutorial guide](TUTORIAL_GUIDE.md) | The single teaching standard, curriculum, visual system, helper boundary, and review process |
| [Toolkit API reference](TOOLKIT_API_REFERENCE.md) | Exact public Toolkit capabilities, names, shapes, relationships, and release-specific constraints |
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
