# Changelog

User-visible history of the ALCHEMI Playbook. Exact job records, checksums, and
development failures belong beside the saved results they describe.

## Unreleased - v2 remaster

### Documentation

- Split the author guidance into general product-tutorial principles, an
  ALCHEMI-specific curriculum and visual guide, and a separate Toolkit API map.
- Reworked the root and tutorial READMEs around current setup, learning
  outcomes, public APIs, expected outputs, and known limits.
- Added a single third-party notice for software, models, checkpoints, and
  datasets used across the tutorial tree.

### Part 1: Toolkit workflows from water interactions to IR

- Rebuilt Part 1 as a Toolkit-first lesson covering `AtomicData`, `Batch`,
  neighbors, model configuration, serial and batched evaluation, composition,
  relaxation, dynamics, hooks, inflight work, distributed stages, and saved
  results.
- Added a short PyTorch, JAX, and Warp primer using the same Toolkit-Ops
  segmented reduction through both framework bindings and the lower-level Warp
  path.
- Added a 90-graph NCI Atlas example that restores the AIMNet2 checkpoint's
  intended finite-molecule Coulomb and pairwise D3(BJ) contributions, then
  compares the completed interaction curves with DFT-D3 and CCSD(T)/CBS.
- Check component closure and graph-order invariance after the `AB − A − B`
  reduction, using an absolute interaction-energy tolerance instead of a
  relative tolerance on large molecular total energies. The lesson also keeps
  the source formal charges for the ionic NCI example.
- Added compatible B97-3c harmonic references and selected observed H2O/D2O
  band positions. Electronic-structure, finite-temperature simulation, and
  experiment remain visibly separate comparisons.
- Added a custom Toolkit adapter for SevenNet-Omni on a small Cu(111) molecular
  adsorption panel. The lesson checks energy and force mapping against the
  model's native call and labels the structures as unrelaxed initial
  placements.
- Added CPU/GPU, homogeneous/heterogeneous batch, and inflight measurements.
  The public `DistributedPipeline` layout is shown, but valid multi-GPU timing
  remains not reported.
- Corrected the offline `DistributedPipeline` example so its fixed-step FIRE2
  work is wrapped in `FusedStage` and can graduate systems to the next worker.
- Added a periodic Packmol box with equal numbers of the phenol and
  N-methylacetamide molecules from the neutral NCI dimer lesson, completed the
  periodic model with PME, and
  added the public `DomainConfig`/`DomainParallel` sequence. The live
  single-GPU call is labeled as an API demonstration without decomposition;
  checked fixed-input 1/2/4-H100 energy/force passes are loaded separately when
  available.
- Shortened the live molecular-dynamics segment for workshop pacing while
  keeping its qualitative limits explicit.
- Applied one notebook visual system for the hero, lesson summary, stage cards,
  progress cards, callouts, plots, and accessible labels.
- Moved data preparation, analysis, plotting, reference loading, and notebook UI
  code into focused `aux/` modules while keeping reusable Toolkit calls visible.

### Archived development notebook

- Incorporated the maintained NCI Atlas work into Part 1 and marked the former
  Toolkit-foundations notebook as an unnumbered development record.
- Retained the separate DESS66 research files and redistribution notice; they
  are not part of the learner notebook.

### Runtime

- Moved the main environment to pinned Toolkit Core 0.2 and Toolkit-Ops 0.4
  source revisions with an H100-class CUDA 13 runtime.
- Fixed Python at 3.12.13, required a `uv` version that supports the install
  options used by the build, and recorded the resolved Conda and Python package
  lists for the later exact-lock step.
- Kept the setup test cache outside the staged source so the domain and notebook
  jobs can enforce a clean checkout after setup.
- Stored inflight predicted charges as one value per atom, matching Toolkit
  `AtomicData` and `Batch` field shapes.
- Made NCI subset regeneration require a clean checkout at the stated upstream
  revision before reading source data.
- Added the Toolkit-Ops JAX CUDA binding and disabled JAX bulk memory
  preallocation so the JAX primer can share one notebook process with PyTorch.
- Added build-time imports and checkpoint setup for the active tutorial model
  paths.
- Measured the six-cell NCI Atlas stage at 22.6 seconds on one H100 NVL and
  checked its Toolkit force against the official AIMNet2 analytic and
  total-energy finite-difference routes.

## v1 - 2026-06-17

- Added CPU/GPU batching to the original adsorption lesson.
- Simplified the melting-point notebook and improved its OVITO animations and
  cached-log plots.
- Standardized the container on one Jupyter kernel and made both tutorial parts
  available from the repository root.
- Corrected the Part 1 OC20Dense reference import path in externally managed
  environments.

## v0 - initial version

- Added the batched adsorption-site tutorial.
- Added the naphthalene solid-liquid coexistence tutorial.
