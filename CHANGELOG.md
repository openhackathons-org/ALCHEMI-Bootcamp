# Changelog

High-level history of the ALCHEMI Playbook. Dates are first-creation dates from
the Git history.

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
