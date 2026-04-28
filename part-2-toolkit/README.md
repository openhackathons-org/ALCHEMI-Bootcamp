# Part 2: Predicting Melting Points with the ALCHEMI Toolkit

Direct-coexistence (solid–liquid) molecular dynamics on naphthalene using the AIMNet2-2025 foundation model and the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit).

The notebook walks through the full SLC pipeline — CIF → warmup → melt generation → SLC stack → temperature sweep → $T_\mathrm{m}$ extraction — and ships in `FAST_DEMO=True` mode by default so the entire walkthrough runs in minutes from cached MD outputs. Flip to `FAST_DEMO=False` to reproduce the cache live (~6 GPU-hours on A100/H100).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Docker | Docker Engine with NVIDIA GPU support (`nvidia-container-toolkit`) |
| GPU | One NVIDIA GPU is sufficient for `FAST_DEMO=True` (analysis-only). `FAST_DEMO=False` benefits from A100/H100. |

No NGC API key or enterprise licence required.

## Quick Start

Build the container and launch JupyterLab:

```bash
docker build -t alchemi-toolkit .
docker run --rm -it --gpus all -p 8889:8889 -v "$PWD":/workspace alchemi-toolkit
```

Open the JupyterLab URL printed in the terminal and run `melting-point-slc.ipynb` top to bottom.

## Trajectory cache (FAST_DEMO mode)

`FAST_DEMO=True` (cell 11, default) reads cached MD trajectories from `assets/naphthalene_long_2025/traj/`. The small per-stage progress logs that drive the in-cell replay output (`assets/naphthalene_long_2025/logs/*.csv`) ship in this repository. The trajectory archive itself (~3.8 GB) is hosted externally:

```bash
mkdir -p assets/naphthalene_long_2025/traj
curl -fL <ASSET_URL> | tar -C assets/naphthalene_long_2025/traj -xzf -
```

> **Note**: `<ASSET_URL>` will be populated when the cache is published. Until then, set `FAST_DEMO = False` in cell 11 to regenerate the trajectories live (~6 GPU-hours end-to-end).

## What's inside the notebook

The notebook is organised in 10 sections:

1. **Environment setup** — control panel + library imports
2. **Crystal structure** — read the CIF, build a supercell, introduce `AtomicData` + `Batch`
3. **Foundation model** — load AIMNet2-2025 with autograd-stress enabled
4. **Warmup MD** — FIRE2 → NVT → anisotropic NPT
5. **Warmup diagnostics** — density, COM-MSD diffusion coefficient, rotational $S_0$
6. **Liquid-half generation** — reseed velocities at $T_\mathrm{melt}$, run NVT
7. **SLC construction** — stack solid + melt, vacuum gap, per-half PBC unwrap
8. **SLC pre-equilibration** — FIRE + multi-temperature NVT in one batch
9. **SLC production NPT** — anisotropic NPT across the temperature sweep
10. **$T_\mathrm{m}$ extraction** — per-temperature $D + S_0$ + density classifier

The notebook ends with *Discussion*, *Extensions*, *Discussion Prompts*, and *References* sections that point at the natural next steps (Ewald electrostatics, larger supercells, multi-compound benchmarks).

## License

Apache 2.0 — see the top-level [LICENSE](../LICENSE).
