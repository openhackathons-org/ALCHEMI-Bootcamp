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

- `nvalchemi-toolkit`: `b770ee6963fd2f6137891e408c370012751918e2`
- `nvalchemi-toolkit-ops`: `c6fbe652315e0cebd4f57a6a25f626258f0dbbfd`

`build/requirements.txt` is the source of truth for these pins. The Dockerfile installs with `uv pip install --no-sources-package nvalchemi-toolkit-ops` so Toolkit's internal `tool.uv.sources` revision cannot override the directly tested Toolkit-Ops commit. It also supplies `build/overrides.txt`: Orb 0.7.0 still declares Toolkit-Ops `<0.4`, while this playbook intentionally pins and validates Toolkit-Ops 0.4.0.

Part 1 applies explicit neighbor-list dispatch for these pinned Toolkit and Toolkit-Ops snapshots. The helper layer selects `batch_naive` or `batch_cell_list` to match the pre-allocated neighbor-list buffers used by the Toolkit relaxation path.

## Core package versions

- `torch`: `2.12.0+cu130`
- `nvalchemi-toolkit`: `0.1.0` from the pinned Git commit above
- `nvalchemi-toolkit-ops`: `0.4.0` from the pinned Git commit above
- `aimnet`: `0.2.0`
- `nvidia-physicsnemo`: `2.1.1`
- `warp-lang`: `1.13.0`
- `orb-models`: `0.7.0`
- `ovito`: `3.15.4`
- `mace-torch`: `0.3.15`
- `cuequivariance`: `0.10.0`
- `cuequivariance-torch`: `0.10.0`
- `cuequivariance-ops-cu13`: `0.10.0`
- `cuequivariance-ops-torch-cu13`: `0.10.0`
- `ase`: `3.29.0`
- `pymatgen`: `2026.5.4`
- `pydantic`: `2.13.4`
- `jupyterlab`: `4.6.1`
- `ipykernel`: `7.3.0`

## Build entrypoints

- Conda package spec: `build/environment.yml`
- Python/runtime package spec: `build/requirements.txt`
- Docker recipe: `build/Dockerfile`
- Compose service: `build/docker-compose.yml`

## Part 1 IR implementation validation

The predicted-charge IR path was tested on the local RTX 4000 SFF Ada using
the exact pinned Core/Ops source trees and the released
`aimnet2-b973c-2025-d3_0` checkpoint.

- The custom Toolkit residual + DSF + D3 pipeline matched AIMNet's official
  DSF+D3 calculator on `{H2O, D2O, (H2O)6, (D2O)6}` to `1.8e-6 eV` in energy,
  `7.2e-7 eV/Å` in force, and `3.0e-8 e` in charge.
- A 15-step fused NVT→NVE control-flow test captured charges from
  `batch.charges` at `AFTER_STEP` with no second model call. Maximum
  graph-charge error was `9e-8 e`. This is validation only; the notebook has
  no reduced execution mode.
- The integrated float64 path converged the generated hexamer in 181 FIRE2
  steps, then reproduced exactly 5 NVT + 10 NVE steps on a fresh rerun. A
  separate 100 NVT + 1,000 NVE development check kept maximum total-energy
  excursion below `0.085 meV/atom` across all four graphs.
- The custom H/D masses were consumed by the integrators; the model inputs and
  potential predictions remained isotope-invariant.
- The deterministic cyclic hexamer seed reached `fmax < 0.01 eV/Å` in 181
  FIRE2 steps.
- A synthetic 3657 cm⁻¹ dipole recovered its peak within 1.2 cm⁻¹ using the
  5 ps Welch estimator (Fourier-bin spacing: 6.7 cm⁻¹; the Hann response is
  broader).

The local environment could not exercise the target compiled neighbor-hook
path because it lacks Python development headers. Its uncompiled RTX timing is
not a production estimate and does not change the notebook's 5,000-step NVT +
50,000-step NVE workload.

The full workflow was subsequently validated from a new scratch installation
on one NVIDIA H100 80 GB HBM3 (CL job `3064655`). All 14 code cells
completed without an error output. Compilation/first evaluation took 182.3 s,
FIRE2 relaxation took 39.4 s, the complete 5,000-step NVT + 50,000-step NVE
notebook cell took 1268.6 s, and the end-to-end Slurm job took 25:33. The raw
trajectory contains
50,000 dipole, charge, kinetic-energy, total-energy, and position samples for
all four graphs. The maximum NVE energy excursion was `0.0909 meV/atom`, below
the notebook's `1 meV/atom` reporting advisory.

A post-run review generalized run-specific validation and interpretation prose
across two Markdown cells without changing code, outputs, or execution counts.
A separate H100 job independently verified the reviewed notebook against the
original calculation artifact; both hashes and the review reason are recorded
in the final validation JSON.

Both hexamers remained connected with intact O–H bonds. Their initial directed
hydrogen-bond rings were not present in every frame, so the notebook correctly
withheld cyclic-DFT overlays and cluster isotope/cluster-minus-monomer
interpretations instead of treating different sampled topologies as
like-for-like.

The monomer Toolkit 3N-reported mean NVE temperatures were 66.41 K (H₂O) and
27.24 K (D₂O). The final notebook therefore also withholds their quantitative
MD centroid ratio; the harmonic mass-only mapping remains the controlled
isotope comparison.
