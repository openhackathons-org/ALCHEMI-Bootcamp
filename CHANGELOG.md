# Changelog

High-level history of the ALCHEMI Playbook. Dates are first-creation dates from
the Git history.

## v2 — 2026-07-09

**Part 1 rebuild slice — predicted-charge water IR**

- Added the full live 5,000-step NVT + 50,000-step NVE workflow for batched
  H₂O/D₂O monomers and cyclic-hexamer seeds using AIMNet2-2025 B97-3c residual,
  explicit pairwise D3(BJ), and the official finite-molecule `simple`
  all-pairs 1/r electrostatics convention.
- Added an `AFTER_STEP` predicted-charge dipole hook with one shared model call
  per fused step, mass-only isotope substitution, 5 ps Welch spectra, and
  charge/energy/cluster-integrity diagnostics.
- Added checksummed H₂O/D₂O/cyclic-H₆/D₆ B97-3c harmonic references and exact
  source/artifact acceptance on an H100 in CL job `3087665`. Topology and
  thermal-state gates withhold comparisons that the production trajectories do
  not support.
- Added six accessible stage cards and 14 live, persisted progress cards around
  every target-H100 wait of at least five seconds. All 14 accepted widget states
  finish as `COMPLETE`; early FIRE convergence reports steps used against the
  declared limit without a partially filled completion rail.
- Added four 2880×1440 Water IR banner candidates and one shared visual system
  for the hero, lesson summary, semantic headings, callouts, placeholders,
  progress states, and plots. The default hero uses real HTML text over art-only
  imagery for sharp rendering and accessibility.
- Moved tutorial mechanics into focused `aux/` modules with no package-level
  re-export surface, while keeping Toolkit conversion, batching, neighbors,
  pipeline composition, FIRE2, fused dynamics, hooks, reductions, and Zarr
  persistence visible in learner cells.
- Pinned both AIMNet and D3 runtime assets by SHA-256, disabled implicit D3
  download, and kept the D3 tensor outside the distributable repository pending
  confirmation of its redistribution rights.

**Part 3 prototype — Toolkit foundations**

- Added a bounded live-compute notebook covering `AtomicData`, homogeneous and
  heterogeneous `Batch` objects, CPU/GPU throughput, neighbor-buffer capacity,
  Toolkit-Ops segmented reduction, and a custom model wrapper.
- Added official Warp Tape computational graphs comparing one heterogeneous
  model call with three homogeneous size-bucket calls, plus NVTX labels for
  optional NVIDIA Nsight Systems profiling.
- Added a charge-aware AIMNet2 ωB97M-D3 composition example: AIMNet core +
  pairwise D3(BJ) + full nonperiodic Coulomb, with explicit pipeline wiring,
  charge conservation, batch/order parity, and force finite-difference checks.
- Added a checksummed, CC BY-attributed 90-structure NCI Atlas subset: three
  ten-point frozen-monomer interaction curves evaluated by all four official
  ensemble members against near-matched ωB97M-D3(BJ)/def2-TZVPPD totals and
  independent CCSD(T)/CBS interaction energies.
- Kept adsorption and periodic Ewald/PME out of Part 3; they require separate
  model-domain and reference validation.

**Runtime**

- Advanced Toolkit Core and Toolkit-Ops to the exact commits required by the
  new pipeline API; pinned AIMNet and PhysicsNeMo explicitly.
- Updated the Docker image, Compose mounts, README, and build-time import gate
  to expose Part 3 while retaining Parts 1 and 2.

## v1 — 2026-06-17

Major changes since initial creation, including a bug fix.

**Bug fix**
- Fixed a Part 1 OC20Dense import failure: on externally-managed ALCHEMI
  environments the OC20Dense validation scripts (e.g.
  `oc20dense_dft_reference_checks.py`) were missing, so importing them raised.
  Restored the scripts and bake Part 1 into the Docker image so the validation
  imports resolve with or without compose bind mounts.

**Runtime & build**
- Modernized the Docker image — current ALCHEMI Toolkit + Toolkit-Ops main
  commits with explicit Git pins, OVITO 3.15.0 → 3.15.4, and the CUDA 13
  cuEquivariance Torch ops so MACE uses the compiled cuEq path.
- Added explicit neighbor-list dispatch for the pinned Toolkit / Toolkit-Ops
  stack (both parts) so the model forward does not hit the host-only
  `neighbor_list(method=None)` path.
- Standardized on a single `alchemi-main` Jupyter kernel (removed `python3`,
  `ovito-pro`, `alchemi-barebones`); JupyterLab now lands at the repo root
  (`/workspace`) showing both parts; both tutorials are baked into the image so
  they work with or without compose bind mounts.
- Added a runtime snapshot document (base image, pinned commits, versions).

**Part 1 — batched adsorption search**
- Added a CPU-vs-GPU batched-throughput demonstration.
- Moved non-Toolkit plumbing into helpers while keeping the Toolkit API and
  concepts visible in the narrative.

**Part 2 — melting-point SLC**
- Trimmed the narrative and extracted the melting-bracket figure into a helper.
- Reworked OVITO rendering: VisRTX (GPU) with a graceful opengl → tachyon
  fallback, browser-safe H.264/yuv420p MP4 output, an oblique camera, and
  optional trajectory smoothing; longer melting and NPT-warmup animations.
- Finalized notebook animation styling for the cached NPT warmup and 500 K
  SLC renders: covalent-radius bonds, trajectory unwrap, bond-based cluster
  unwrap, smoothing after the OVITO modifiers without minimum-image wrapping,
  a horizon-aligned camera, 60-frame / 20 fps MP4 output, and adjacent NPT
  plot/animation cells.
- Improved cached-log diagnostic readability with thicker raw, rolling-mean,
  and reference lines.
- Corrected documentation to reflect the Orb-v3 (OMol) potential.

**Repo**
- Renamed tutorial directories to `*-toolkit`; refreshed README links
  (GitHub, pip, ALCHEMI hub).

## v0 — Initial creation

- **Part 2 — melting-point SLC (naphthalene)** — Ryan — 2026-04-17
- **Part 1 — batched adsorption-site search** — Nikita Fedik — 2026-05-30
