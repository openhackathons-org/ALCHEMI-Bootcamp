# Current Tutorial Goal, Flow, and Status

Date: 2026-05-18

This is the current source-of-truth plan for the Part 1 tutorial. Older review
reports remain useful audit history, but this file reflects the current
Toolkit-first, open-model, D3-disabled workflow and the broader batching
narrative.

## License Guardrail

- Active tutorial execution uses MACE-MP/MACE-MPA checkpoints that the MACE
  foundation-model registry lists under MIT.
- MACE-MH-1 and the OC20 surface head are not used in the runnable NVIDIA
  tutorial path because the model is ASL-listed. They can be mentioned only as
  an optional user-side experiment, subject to that user's license review.
- The bundled OC20Dense validation pack is source data for reproducibility, not
  generated model output. Keep the full OC20Dense archives local-only; ship only
  the slim validation subset used by the notebook.

## Goal

Show domain experts how batched, GPU-native atomistic simulation changes the
practical scale of exploratory computational chemistry. NVIDIA ALCHEMI is the
enabling layer that connects familiar structure workflows to GPU throughput:
batching, model wrappers, optimizers, constraints, metadata, accelerated
kernels.

Adsorption configuration search is the worked example. The tutorial's concrete
research question is:

> How many starting geometries do we need before we can trust the
> lowest-energy relaxed structure found within a defined adsorption search space?

The scientific promise is intentionally scoped around the throughput-screening
stage of a discovery pipeline. This tutorial does not run DFT or experiment.
The result should help readers rank, inspect, and document candidate structures
before whatever project-specific review or validation path comes next.

## Audience

The reader is expected to know Python, atomistic structures, MLIPs, and basic
chemistry or materials modeling. The reader is not expected to know ALCHEMI,
this notebook's helper modules, or the OC20Dense validation setup.

## Current Execution Contract

- Default execution path: `toolkit`.
- Public notebook/helper surface: native Toolkit only. Historical service-client
  code has been removed from the active helper package, dependency file,
  deployment files, and focused tests.
- Default notebook scope for production/presentation refreshes: `RUN_SCOPE =
  "full"`.
- Saved tutorial artifacts are read-only unless `REFRESH_SAVED_RESULTS = True`.
  Exploratory live recomputes write under `outputs/live_runs/<timestamp>/` so
  presentation caches are not overwritten by accident.
- Current GitHub-clean state: generated outputs, runtime/model caches, PDFs,
  and review-candidate images are not bundled in the working tree. The
  OC20Dense validation source data is bundled as
  `data/reference/oc20dense-validation-pack.tgz` so live validation can be
  rerun without committing the expanded reference folder.
- OC20Dense reference-data policy: the full OC20Dense archives/LMDB must remain
  local-only outside the repo. The bundled validation pack is the slim public
  subset used here: full released DFT trajectories for the three replay cases,
  all 92 NH3 DFT trajectories for the fixed-geometry ranking slice, clean-slab
  references, and mapping/target files.
- Saved-output loading is explicit. `SAVED_TUTORIAL_RUN_ID` and
  `SAVED_ACCURACY_RUN_ID` reopen one named live run. Set either value to
  `"latest-complete"` only when you want the newest live run that passes the
  required-file checks; the notebook does not silently load arbitrary timestamp
  folders.
- Default surface-screen teaching model: open MACE checkpoint
  `medium-mpa-0` through the ALCHEMI Toolkit.
- Default full-grid adsorption batch size in the notebook control cell:
  `12` structures per Toolkit batch. The artifact runner keeps each six-start
  adsorbate/surface question together as the scientific unit, so the generated
  surface-screen artifacts use 36 adsorption batches of 6 structures each.
- Batch-size calibration models: open MACE checkpoints, currently framed as a
  small-vs-large comparison plus the `medium-mpa-0` default, so readers can see
  that throughput and VRAM headroom depend on the selected model as well as the
  chemistry.
- Default device: CUDA GPU on `ws-loc` or a comparable cluster/workstation GPU.
- D3(BJ): available in Toolkit workflows, but disabled here to match the
  non-D3 OC20Dense reference convention.
- Keep the current notebook Toolkit-only. Service/API variants should be
  separate future routes that reuse the same scientific result schema.
- Adsorption energy convention:
  `E_ads = E(adslab) - E(clean slab) - E(gas adsorbate)`.
- Sign convention: negative `E_ads` means exothermic binding.
- Result tables publish `E_ads` as the canonical field. Legacy `E_bind` aliases
  may remain in generated CSVs only for compatibility with older cached outputs.
- Structure generation: programmatic ASE/pymatgen/helper code only; no
  hand-authored token-written XYZ blocks.
- Reference interpretation: contextual literature rows are not strict DFT
  parity unless slab, coverage, functional, dispersion convention, frozen
  layers, and sign convention match.

## Reader Flow

1. **Frame the combinatorial problem.**
   The notebook starts from the general problem: chemical discovery often
   creates many plausible structures, not one obvious structure.

2. **Narrow the question.**
   Adsorption gives the problem a concrete form: one molecule, one surface, many
   possible sites, orientations, heights, and final local minima.

3. **Introduce NVIDIA and ALCHEMI's role.**
   NVIDIA provides the accelerated computing layer. ALCHEMI makes existing
   atomistic workflows batched, customizable, GPU-native, and deployable. The
   MLIP is the final computational ingredient inside that broader workflow, not
   the whole story.

4. **Run the smallest batch-speedup example.**
   The H2O example shows how independent structures become `AtomicData`, then a
   `Batch`, then a GPU calculation.

5. **Calibrate adsorption batch size on real chemistry.**
   A short H2O/TiO2(110) relaxation sweep compares open MACE checkpoints. The
   section reports structures/s, atoms/s, and GPU memory so readers can see why
   batch size depends on both chemistry and model choice.

6. **Build scientific structures with established tools.**
   The notebook builds slabs, adsorbates, clean-slab references, and gas-phase
   references programmatically with ASE/pymatgen/helper code, then accelerates
   the relaxation bottleneck with ALCHEMI.

7. **Run a two-start hello-world.**
   The reader sees why one plausible starting point can miss a lower-energy
   relaxed structure.

8. **Design and run the worked surface screen.**
   The active panel is now explicit in the notebook, not hidden in a helper:
   Cu(111), Cu(100), Cu(110); rutile TiO2(110), TiO2(100), TiO2(101);
   and TiN(001), TiN(110), TiN(210), crossed with CO, H2O, NH3, and
   CH3OH. Each adsorbate/surface pair uses three site classes and two
   orientations, for six starting geometries per pair. `RUN_SCOPE = "short"`
   runs one six-start example. `RUN_SCOPE = "full"` runs the 216-start panel.

9. **Rank and inspect results.**
   The notebook ranks final structures by `E_ads`, classifies final sites from
   relaxed geometry, filters unreliable structures, and visualizes the batch
   minima.

10. **Compare with reference context.**
   The notebook keeps contextual literature values separate from strict
   apples-to-apples validation.

11. **Use the OC20Dense checkpoint for reproducibility checks.**
    The tutorial uses closed-shell H2O, NH3, and N2 OC20Dense records to check
    DFT trajectory arithmetic, DFT-relaxed final single-point energies, Toolkit relaxations,
    RMSD, and explicit MACE adsorption-energy subtraction.

## Documentation Coverage

- Current source-of-truth plan:
  `part-1-batched-adsorption/docs/current_tutorial_status_and_flow.md`
- Running project changelog:
  `part-1-batched-adsorption/docs/CHANGELOG.md`
- Main reader tutorial:
  `part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`
- Reproducibility companion:
  `part-1-batched-adsorption/oc20dense-accuracy-reproducibility-check.ipynb`
- Part 1 overview and run instructions:
  `part-1-batched-adsorption/README.md`
- Shared execution/science contract:
  `shared/adsorption_tutorial/README.md`,
  `shared/adsorption_tutorial/backends.md`,
  `shared/adsorption_tutorial/contract.py`,
  `shared/adsorption_tutorial/panel.yml`
- Reference provenance and review status:
  `part-1-batched-adsorption/references/manifest.yml`,
  `part-1-batched-adsorption/references/manual_checks.md`,
  `part-1-batched-adsorption/references/domain_expert_fact_check.md`
- Image manifest:
  `part-1-batched-adsorption/assets/images/manifest.md`
- Benchmark and validation scripts:
  `part-1-batched-adsorption/scripts/run_oc20dense_known_examples.py`,
  `part-1-batched-adsorption/scripts/oc20dense_dft_reference_checks.py`,
  `part-1-batched-adsorption/scripts/run_oc20dense_dft_final_single_points.py`,
  `part-1-batched-adsorption/scripts/run_oc20dense_mace_adsorption_energies.py`,
  `part-1-batched-adsorption/scripts/summarize_oc20dense_accuracy.py`,
  `part-1-batched-adsorption/scripts/benchmark_h2o_saturation.py`,
  `part-1-batched-adsorption/scripts/benchmark_adsorption_batching.py`,
  `part-1-batched-adsorption/scripts/run_oc20dense_ovito_render_pipeline.sh`
- Auto-generated local evidence reports:
  `part-1-batched-adsorption/outputs/reports/toolkit_full_panel_report.md`,
  `part-1-batched-adsorption/outputs/reports/toolkit_co_cu111_step_report.md`,
  `part-1-batched-adsorption/outputs/oc20dense_known_examples/reports/oc20dense_known_examples_report.md`,
  `part-1-batched-adsorption/outputs/oc20dense_known_examples/reports/oc20dense_accuracy_comparison_report.md`,
  `part-1-batched-adsorption/outputs/oc20dense_known_examples/dft_reference_checks/dft_reference_check_report.md`,
  `part-1-batched-adsorption/outputs/oc20dense_known_examples/dft_final_single_points/reports/dft_final_single_point_report.md`,
  `part-1-batched-adsorption/outputs/oc20dense_known_examples/mace_adsorption_energy/reports/mace_adsorption_energy_report.md`
- Batching-grid decision:
  `part-1-batched-adsorption/docs/batching_grid_2026-05-12.md`
- Audit history:
  `part-1-batched-adsorption/docs/tutorial_review_2026-05-07.md`,
  `part-1-batched-adsorption/docs/tutorial_review_2026-05-11.md`,
  `part-1-batched-adsorption/docs/tutorial_review_2026-05-11-delta.md`,
  `part-1-batched-adsorption/docs/tutorial_review_2026-05-11-style.md`

## Current Computed Evidence

Current presentation notebook state:

- Current active panel: 9 slabs x 4 adsorbates x 6 starts = 216 adsorption
  relaxations, plus 9 clean-slab and 4 gas-reference relaxations.
- Teaching path: ALCHEMI Toolkit with the open `medium-mpa-0` checkpoint for
  the surface-screen adsorption chemistry, D3(BJ) disabled.
- The surface, Miller-index, adsorbate, site-class, orientation, and starting
  height choices are visible in notebook cells. Surface construction is paced
  by family: Cu facets, rutile TiO2 facets, TiN facets, then panel assembly and
  counts. `helpers/surface_screen.py` is bookkeeping/statistics only; the
  scientific grid and ranking policy are visible in the notebook and recorded
  in runner metadata.
- Validation path: open MACE checkpoint `medium-mpa-0` for selected OC20Dense
  surface-chemistry checks, D3(BJ) disabled.
- Read-only presentation execution means cached/precomputed outputs were loaded
  and checked, not recomputed. The latest read-only notebook run on `ws-loc`
  executed all 54 code cells successfully.
- Live recompute timings are separate from read-only execution timings. The
  latest full surface-screen artifact refresh on `ws-loc` used 9 clean-slab
  relaxations, 4 gas-reference relaxations, and 216 adsorption relaxations.
  Recorded Toolkit batch time was 4.86 min.

Historical full-panel evidence retained for context:

- Earlier raw full-panel script run on `ws-loc` relaxed 252 structures for the
  old Cu/Pd/Al2O3 panel in about 3481 s, or 58 min. This is not the current
  active panel timing; it remains useful only as historical throughput context.
- Earlier notebook execution artifacts include a 415.8 s full-notebook run and
  several 7-8 s read-only cached-output checks. These should not be compared as
  identical workloads.

Batching evidence:

- Host: `aad51f7-lcedt` (`ws-loc`).
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 94.97 GiB
  visible VRAM.
- H2O throughput plateaued around 1024-4096 structures; 12288 H2O used
  58.33 GB without increasing throughput, and 16384 OOMed.
- Real Al2O3/H2O adsorption throughput plateaued around batch 24:
  24 configs used 19.20 GB at 4.64 structures/s; 96 configs used 76.63 GB
  at 4.46 structures/s.
- Notebook full-grid adsorption batches currently default to 12 structures for
  shared-GPU reliability, not maximum-VRAM batching. The production artifact
  runner kept each adsorbate/surface question as one six-start batch; peak
  reserved GPU memory in the recorded batch metadata was about 5.62 GB above
  the already-loaded model/session state.
- New notebook calibration section: H2O/TiO2(110), 27 starting structures, 75
  atoms each, 31 active atoms each, 40 FIRE2 steps. The current direction is to
  compare open MACE checkpoints so the batch-size recommendation is tied to
  both model size and available VRAM. Earlier MH-1 timings are historical only
  and should not appear as active tutorial defaults.

OC20Dense validation checks:

- Uses closed-shell H2O, NH3, and N2 systems, intentionally separate from the
  CO/H2O/CH3OH teaching panel.
- The main tutorial includes the selected validation cells directly: one
  trajectory-replay record per closed-shell adsorbate, then one fixed-system
  NH3 ranking check across all 92 released DFT-relaxed final geometries.
- The trajectory replay verifies the data path and exposes model limitations.
  The current three-record slice includes one clear NH3 miss: relaxed Eads error
  about 0.995 eV and final adsorbate RMSD about 1.566 A.
- The stronger quantitative validation story is the fixed-geometry NH3 ranking
  check, where DFT and MACE energies are both shifted relative to the released
  DFT rank-1 geometry for the same fixed system.
- DFT trajectory arithmetic reproduces released adsorption-energy targets for
  the selected records; the 92-geometry NH3 ranking check has max DFT
  trajectory-target difference `0.00e+00 eV`.
- Current open-model fixed-geometry baseline: `medium-mpa-0` gives
  DFT-rank-1 anchored RMSE `0.178 eV`, MAE `0.150 eV`, bias `-0.145 eV`,
  Spearman `0.786`, and selects a structure only `0.0128 eV` above the released
  DFT minimum in this 92-geometry NH3 slice. The stronger MH-1/OC20-head result
  from earlier exploration is retained only as a license-gated note, not as the
  active NVIDIA tutorial path.
- Neutral-gas Eads subtraction is retained only as a reference-convention
  control. It is not the primary accuracy metric because it does not reproduce
  the OC20Dense DFT reference convention.

## Current Verification State

Latest GitHub-clean check on 2026-05-18:

- Notebook edits were applied through the local VS Code notebook MCP bridge and
  saved through that bridge.
- The post-validation section was restructured so long cells now separate
  visible scientific choices, starting-geometry assembly, analysis tables, and
  artifact writing.
- Generated outputs, runtime/model caches, local OC20Dense validation data,
  PDFs, review-candidate images, and local spillover files were archived
  outside the repo at
  `/home/nfedik/projects/tutorials-local-archive/part-1-batched-adsorption-20260518/`.
- Main notebook JSON parses, and the saved notebook has zero stored error
  outputs.
- Helper modules compile with `.venv-toolkit/bin/python -m py_compile
  part-1-batched-adsorption/helpers/*.py`.
- Full local test suite passed:
  `pytest -q part-1-batched-adsorption/tests`
  (`90 passed, 9 skipped`). The additional skips are expected in the
  GitHub-clean checkout because large OC20Dense source artifacts and optional
  rendering/runtime assets are not bundled.

Passed after the residual cleanup on 2026-05-16:

- Active source scan passed for root README, Part 1 README, shared adsorption
  contract docs, active helpers, active tests, and active scripts: no stale
  service-client dependency, old folder path, or retired DFT wording remains in
  those active surfaces.
- Main notebook JSON parses, has 83 cells and 50 code cells, preserves the
  three intended TODO anchors, and has zero stored error outputs.
- Local Python compile checks passed for touched helpers and scripts.
- Local Toolkit venv focused tests passed except for the known WSL OVITO
  `libOpenGL.so.0` import limitation: 39 passed, 1 environment failure.
- Focused remote tests on `ws-loc` passed:
  `pytest -q tests/test_cache.py tests/test_models.py tests/test_imports.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`40 passed`).
- The cleaned notebook executed read-only on `ws-loc`; all `50` code cells
  passed. The executed artifact is
  `outputs/executed_notebooks/alchemi-mace-adsorption-search-clean-check.ipynb`,
  and the local main notebook was refreshed from that artifact.

Passed after the Toolkit-only public-surface cleanup:

- Main notebook and active helper/test/script/shared-doc surfaces are clean for
  the retired service-client path, old folder paths, and retired DFT wording.
- Python compile checks passed for the touched helpers and active tests.
- Focused remote tests on `ws-loc` passed:
  `pytest -q tests/test_cache.py tests/test_models.py tests/test_imports.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`46 passed`).
- The read-only presentation notebook executed on `ws-loc`; all `50` code
  cells passed. The executed artifact is
  `outputs/executed_notebooks/alchemi-mace-adsorption-search-toolkit-only-check.ipynb`,
  and the main notebook was refreshed from that artifact.
- The notebook execution caught and fixed one stale service-era config field:
  `timeout=1800` was removed from `ToolkitRelaxationConfig(...)`.

Passed after the latest adsorption-calibration update:

- Main notebook JSON parses.
- Python compile checks passed for the touched helper files.
- The notebook was executed in read-only presentation mode on `ws-loc`; all 50
  code cells passed in `7.7 s`.
- Focused remote tests passed:
  `pytest -q tests/test_cache.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`21 passed`).

Passed after the language/positioning cleanup:

- Source-only terminology scan passed for the targeted presentation issues:
  no retired DFT wording, raw service/backend wording, chunk wording, or
  "global minimum" claims remain in notebook cell source.

Passed earlier in the validation/provenance cleanup:

- Focused tests:
  `part-1-batched-adsorption/tests/test_relaxation_backends.py` and
  `part-1-batched-adsorption/tests/test_oc20dense_benchmark.py`: 15 passed.
- Main notebook source parsed with `jq empty`.
- Selected validation cells were executed through the live notebook MCP bridge.

Latest production check on 2026-05-16:

- Notebook-bridge edits were used for notebook cells, and the live notebook
  buffer was saved clean (`isDirty: false`).
- Surface-screen artifacts were previously regenerated on `ws-loc` with
  MH-1/OC20-head, D3 disabled. Those results are now historical and should be
  regenerated with the open-model default before publishing saved tutorial
  outputs.
- Output counts: 1093 files under
  `outputs/precomputed/tutorial/surface_screen_v1_mace_mpa0_full/surface_screen/`,
  including 229 trajectory files and 229 trajectory-log CSVs. That full
  surface-screen cache is not present in the current local checkout after the
  archive cleanup; regenerate it or promote one complete live run before using
  official saved tutorial mode for the surface-screen section.
- Adsorption results: 216/216 converged, 216/216 reliable for the minimum
  search, 216/216 still adsorbed. Step status: 186 green, 30 yellow, 0 red.
  Adsorption optimizer steps: min 117, median 171.5, max 247.
- Focused remote tests on `ws-loc` passed:
  `pytest -q tests/test_geometry_validation.py tests/test_imports.py`
  (`44 passed`).
- Full notebook execution through `scripts/run_notebook_cell_by_cell.py` passed
  all 54 code cells and wrote
  `outputs/executed_notebooks/alchemi-mace-adsorption-search.executed.latest.ipynb`.

Current presentation caveat:

- The notebook stores intentional TODO review anchors. These are human review
  markers, not execution blockers.
- Saved/precomputed outputs and the slim OC20Dense reference subset are not
  bundled in the current GitHub-clean working tree. They are archived locally
  under
  `/home/nfedik/projects/tutorials-local-archive/part-1-batched-adsorption-20260518/`.
  Restore them or regenerate them before using saved-result mode or live
  OC20Dense validation.

Latest local cache-layout validation on 2026-05-18:

- Superseded by the GitHub-clean check above. The older cache-validation
  numbers remain in this document as execution history, not current repo state.

Known local test limitation:

- OVITO import/render tests skip on this WSL venv when the optional OpenGL/OVITO
  runtime is unavailable. This is an environment/rendering limitation, not an
  adsorption-workflow failure. VisRTX rendering should be run through the
  documented Windows/PowerShell or GPU-capable route.

## Review Anchors To Keep

Keep visible review markers where they represent real human work still needed.
Do not strip them just to make the notebook look finished.

- `<mark>TODO - REFERENCE REVIEW</mark>`: use for claims that need human
  reference/domain review before promotion from context to strict comparison.
- `<mark>TODO - VISUAL REVIEW</mark>`: use for visuals that need manual layout
  or rendering review.
- `<mark>TODO - HUMAN REVIEW</mark>`: use for final judgment calls that cannot
  be decided by tests alone.

## Review Anchors Still Intended

These are intentional human-facing anchors, not cleanup leftovers:

1. `TODO - VISUAL REVIEW`: workflow graphics and selected renders.
2. `TODO - REFERENCE REVIEW`: contextual literature values before promotion to
   strict quantitative claims.
3. `TODO - HUMAN REVIEW`: final presentation choices that cannot be decided by
   tests alone.

## Definition Of Done For Presentation

- The first screen states the broad throughput problem, the adsorption example,
  and the ALCHEMI value clearly.
- NVIDIA/ALCHEMI is framed as an enabling layer that respects established
  atomistic tools, invites integration, and accelerates the bottleneck.
- The default teaching path is consistently Toolkit-first. The surface-screen
  adsorption chemistry uses open MACE checkpoints, D3 disabled; the batch-size
  calibration compares open model sizes so readers can choose a batch size for
  their model and GPU.
- Service/API variants are treated as separate routes that reuse the same
  scientific result schema.
- Every plotted energy is labeled as `E_ads` or otherwise clearly tied to the
  adsorption-energy convention.
- Literature MAD guides are described as orientation guides, not run-specific
  error bars.
- The notebook can execute on `ws-loc` with the `alchemi-toolkit` kernel.
- The main notebook can recompute the selected OC20Dense checks directly; any
  companion notebook should be treated as an auxiliary audit surface.
- Human/reference/visual review markers are either intentionally retained or
  explicitly resolved.
