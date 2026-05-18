# Part 1 Tutorial Changelog

This is the running change log for the ALCHEMI Toolkit adsorption-search
tutorial. Use it to track what changed, what was verified, and what still needs
execution. Keep detailed audit notes and raw run logs in their dated documents;
keep this file short enough that a new agent can scan it first.

## 2026-05-18

### License-Gated Model Cleanup

- Removed MACE-MH-1 / `oc20_usemppbe` from the active runnable tutorial path
  because the model is ASL-listed and this NVIDIA tutorial must use open MACE
  checkpoints.
- Switched active helper/script/cache defaults to open `medium-mpa-0` output
  stems: `surface_screen_v1_mace_mpa0`,
  `oc20dense_closed_shell_trajectory_mace_mpa0`, and
  `oc20dense_nh3_92_fixed_geometry_mace_mpa0`.
- Reframed MH-1 as an optional user-side model to test only when license review
  permits. The open-model validation baseline remains `medium-mpa-0`; the batch
  calibration should compare open MACE sizes such as small versus large.
- Dependency review against `origin/dev`: no new Python/package dependency
  manifests were added by this branch. The only dependency-file delta remains
  deletion of the old `part-1-nim/environment.yml`.

### Bundled OC20Dense Validation Pack

- Added `data/reference/oc20dense-validation-pack.tgz` as the tracked
  reference-data artifact for live validation. It contains the full DFT
  trajectories used by the notebook widgets and checks: three closed-shell
  replay trajectories, all 92 NH3 fixed-geometry ranking trajectories, clean
  surface references, and OC20Dense mapping/target files.
- Kept the expanded `data/reference/oc20dense/` folder gitignored. The
  validation helper now unpacks the bundled tarball when live validation needs
  the reference folder and it is not already present.
- Updated run documentation to distinguish the validation source-data pack from
  saved/precomputed output caches.

### GitHub-Clean Notebook Restructure

- Restructured the post-validation half of the notebook so the surface-screen
  section now separates: visible science choices, starting-geometry assembly,
  reference relaxation/loading, batched relaxation, result interpretation, and
  artifact writing.
- Moved final result file-writing into `helpers.surface_screen` while keeping
  the chemistry, reference convention, and analysis steps visible in notebook
  cells.
- Removed hidden `globals()` state checks from the final artifact path.
- Cleaned stale active-reference wording for the current Toolkit-only path.
- Archived large/local-only artifacts outside the repo at
  `/home/nfedik/projects/tutorials-local-archive/part-1-batched-adsorption-20260518/`.
  This includes generated outputs, runtime/model caches, local OC20Dense
  validation data, review candidate images, PDFs, root slide screenshots, and
  old local background-tool spillover. The old AWH/OER pivot archive was also
  moved out of the GitHub-minimal tutorial tree.
- Pruned duplicate/stale visual drafts from the shipped assets folder; the
  notebook keeps only the four referenced tutorial figures plus logos and the
  image manifest.
- Adjusted the OC20Dense slim-data test to skip cleanly in a minimum GitHub
  checkout when local validation structures are not restored.
- Verification: notebook saved through the local notebook MCP bridge; notebook
  JSON parses; saved notebook has zero stored error outputs; helper modules
  compile; `pytest -q part-1-batched-adsorption/tests` passed locally
  (`90 passed, 9 skipped`).

### Triple-Review Cache And Teaching-Clarity Cleanup

- Tightened official saved-output guards: OC20Dense benchmark scripts now refuse
  accidental writes under `outputs/precomputed/` unless an explicit refresh
  guard is set.
- Extended OC20Dense path discovery so slim local data remains the default, but
  mapping/archive/LMDB lookups can use `OC20DENSE_FULL_DATA_ROOT` when a full
  local OC20Dense tree is intentionally provided.
- Updated saved/live cache wording in the notebook and docs so loaded artifacts
  are described as saved validation artifacts, not as newly rerun calculations.
- Kept the main notebook's scientific grid visible and left only formatting,
  plotting, cache, and validation bookkeeping in helper modules.
- Corrected the root README to match the current 9-slab x 4-adsorbate teaching
  panel and the current surface-chemistry MACE path.
- Verification: script compile checks passed; both notebooks parse as JSON;
  `pytest -q part-1-batched-adsorption/tests` passed locally
  (`93 passed, 3 skipped`).

## 2026-05-16

### Cache Layout Cleanup For Interactive Reruns

- Added an explicit saved/live cache model. Official saved artifacts now live
  under `outputs/precomputed/`; interactive reruns write under
  `outputs/live_runs/<run_id>/`; runtime/model caches live under
  `outputs/runtime_cache/`.
- Added explicit run-id selection for reopening prior live runs:
  `SAVED_TUTORIAL_RUN_ID` and `SAVED_ACCURACY_RUN_ID`. The notebook lists recent
  live runs but does not auto-load the latest timestamp.
- Added the explicit `"latest-complete"` run-id selector. It chooses the newest
  timestamped live run only after required tutorial or accuracy files pass the
  completeness checks.
- Moved local OC20Dense source/reference data out of generated outputs and into
  a local workspace `data/reference/oc20dense/` tree, with raw archives, LMDB,
  mapping pickles, and selected DFT trajectories separated by role.
- Slimmed `data/reference/oc20dense/` to the validation subset actually used by
  the notebook: 94 DFT adslab trajectories, 3 clean-slab reference trajectories, 3
  initial structures for closed-shell live replay, and slim mapping/target
  pickles. The full 40 GB OC20Dense source tree was moved outside the repo to
  `/home/nfedik/projects/tutorials-local-data/oc20dense/full-20260518`.
- Updated OC20Dense scripts to use the slim initial-structure extxyz files when
  the full LMDB is absent, and to warn that unsupported system/config ids need
  the full OC20Dense archives/LMDB via `OC20DENSE_FULL_DATA_ROOT`.
- Archived stale/superseded generated artifacts under
  `_archive/runs/2026-05-18-pre-cache-cleanup/` with an
  `archive_manifest.csv`.
- Promoted existing OC20Dense validation/ranking outputs into
  `outputs/precomputed/accuracy/` and moved current H2O/calibration tables into
  `outputs/precomputed/tutorial/`.
- Set the current local notebook default to recompute the main tutorial
  workflow into a timestamped live run while keeping the complete OC20Dense
  accuracy caches in saved mode.
- Current local caveat: the full active 216-start surface-screen cache is still
  absent locally. Official saved tutorial mode will stop at the surface-screen
  section until that cache is regenerated or a complete live run is promoted.
- Verification after cleanup: notebook JSON parses; all 54 code cells pass
  Python syntax parsing; `pytest -q part-1-batched-adsorption/tests` passes
  locally (`93 passed, 3 skipped`). The skipped tests are optional
  OVITO/rendering paths in the local WSL runtime.

### Surface-Screen Production Rerun and Notebook Blocker Cleanup

- Kept the scientific grid visible in the notebook: slabs, Miller indices,
  site classes, adsorbate orientations, starting height, rotations, frozen
  slab fraction, reliability thresholds, and nominated single-start sites are
  defined in notebook cells. Helpers now carry low-level structure primitives,
  plotting/formatting, artifact paths, and table bookkeeping.
- Removed the hidden fixed `build_six_start_grid` recipe and its `SIX_START_*`
  constants from the active helper API. The headless surface-screen runner and
  geometry tests now pass explicit site/orientation choices into the low-level
  grid builder.
- Fixed the notebook ordering bug where site coordinates were computed before
  precomputed clean-slab structures were loaded. Site candidates are now tied
  to the active clean-slab geometries before starting structures are audited.
- Fixed the hello-world cell in precomputed mode: it now loads the two CO/Cu(111)
  results from the generated surface-screen cache instead of requiring a
  separate `hello_world_co_cu111_two_starts` cache entry.
- Corrected step-status summaries to use the final optimization result force,
  not the last saved trajectory snapshot. This matters because
  `snapshot_frequency=5` can leave the last logged snapshot behind the final
  converged optimizer state.
- Added explicit surface-screen metadata for grid policy, ranking policy,
  recorded Toolkit batch time, invocation wall time, and whether batches were
  loaded from cache during a table-regeneration pass.
- Reran the current full surface screen on `ws-loc` with `mh-1` and
  `TOOLKIT_HEAD=oc20_usemppbe`: 9 clean-slab relaxations, 4 gas references,
  and 216 adsorption relaxations. Recorded Toolkit batch time was `4.86 min`;
  all 216 adsorption structures converged, all 216 were reliable for the
  minimum search, all 216 remained adsorbed, and step status was 186 green /
  30 yellow / 0 red. Adsorption optimizer steps ranged from 117 to 247.
- Generated 229 trajectory files and 229 trajectory-log CSVs under the old
  `outputs/surface_screen_mh1_oc20_usemppbe/` path on `ws-loc`. After the cache
  cleanup, the corresponding official local target is
  `outputs/precomputed/tutorial/surface_screen_v1_mh1_oc20_usemppbe_full/surface_screen/`;
  that local cache still needs to be regenerated or promoted from a complete
  live run.
- Verification: notebook JSON parses; notebook-bridge saved state is clean;
  focused remote tests passed on `ws-loc` (`44 passed`); the full notebook
  executed cell by cell with all `54` code cells passing and wrote
  `outputs/executed_notebooks/alchemi-mace-adsorption-search.executed.latest.ipynb`.

### Residual Cleanup After Double Review

- Kept the visible notebook review anchors:
  `TODO - VISUAL REVIEW`, `TODO - REFERENCE REVIEW`, and
  `TODO - HUMAN REVIEW`.
- Removed stale active service-client files from the Toolkit tutorial:
  `helpers/api_client.py`, `helpers/throughput.py`, the Part 1 Docker/NIM
  stack files, and the old monitoring setup.
- Removed BMD/BGR model classes and compatibility aliases from
  `helpers/models.py`. Active cache/tests now use only
  `RelaxationBatchResult`, `OptimizationResult`, and
  `AtomicStructurePayload`.
- Removed service-client dependencies (`requests`, `aiohttp`) from
  `environment.yml`.
- Rewrote the shared adapter note so the current route is Toolkit-only and any
  future service/API version is described as a separate route that must prove
  schema and scientific equivalence.
- Replaced old `part-1-nim` paths in active scripts/shared config with
  `part-1-batched-adsorption`, and updated the resumable full-panel script to
  use the current Cu(111), Al2O3(0001), TiO2(110) panel.
- Tightened the notebook mini-guide guard so precomputed mode clears stale
  demo variables instead of showing a previous live-run result.
- Added provenance checks for precomputed adsorption batch calibration tables:
  cached rows must match the requested model labels, checkpoints, heads, and
  batch sizes before they are accepted.
- Softened notebook validation wording to a limited OC20Dense
  reproducibility/model-sanity check and clarified the final reference-check
  paragraph so context-only rows are not presented as direct model-error
  statistics.
- Added a visible trajectory-replay interpretation note in the main notebook:
  the NH3 case is a clear relaxation miss, while the stronger quantitative
  validation result is the DFT-rank-1 anchored fixed-geometry ranking check.
- Reworked the status document's evidence section so historical raw recompute
  timings, current live recompute timings, and read-only cached-output notebook
  executions are not mixed as equivalent workloads.
- Addressed read-only reviewer cleanup leaks: removed the unused
  `ADSORPTION_BACKEND` env var from the Toolkit runner, renamed stale
  service-response wording to cached JSON responses, changed the rejected
  backend test value to neutral `unsupported`, and normalized the generated
  OC20Dense report title to DFT-relaxed final wording.
- Verification: active source scan passed for root README, Part 1 README,
  shared adsorption docs, active helpers, active tests, and active scripts;
  notebook JSON parses with zero stored error outputs and the three intended
  TODO anchors; local `py_compile` passed; local Toolkit tests were 39 passed
  with the known WSL OVITO `libOpenGL.so.0` environment failure; remote
  `ws-loc` focused tests passed (`40 passed`); the cleaned notebook executed
  read-only on `ws-loc` with all `50` code cells passing and was copied back to
  the local presentation notebook.

### Toolkit-Only Public Surface and Notebook Pacing

- Removed the old service-style route from the active helper package exports.
  The notebook-facing helper surface now exposes the native Toolkit relaxation
  engine, neutral relaxation result models, calibration helpers, plotting,
  validation loaders, and visualization utilities. The main notebook source has
  no BGR/API-client/aiohttp/request-path references.
- Replaced public `BGR*` model imports in active tests with neutral
  `AtomicStructurePayload`, `RelaxationRequest`, and `RelaxationBatchResult`
  names. Deleted the stale live-endpoint test from the active test suite.
- Moved low-value notebook mechanics into helpers:
  `load_or_run_adsorption_batch_calibration(...)`,
  `display_adsorption_batch_calibration_results(...)`,
  `load_trajectory_validation_results(...)`, and
  `load_nh3_ranking_results(...)`. The notebook cells now keep the chemistry,
  model choices, and run settings visible while leaving cache selection,
  table formatting, and artifact checks in the helper layer.
- Removed the stale `timeout` argument from the Toolkit config cell after the
  notebook execution caught it as a production error.
- Updated the package-version cell to report Toolkit-relevant packages instead
  of service-client dependencies.
- Verification on `ws-loc`: focused remote tests passed,
  `pytest -q tests/test_cache.py tests/test_models.py tests/test_imports.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`46 passed`). The notebook executed in read-only presentation mode with all
  `50` code cells passing, and the refreshed executed notebook was copied back
  to `alchemi-mace-adsorption-search.ipynb`.

## 2026-05-15

### Adsorption Batch-Size Calibration

- Added a user-facing calibration section after the H2O batching example. It
  runs short capped adsorption relaxations on an H2O/TiO2(110) pool before the
  full adsorption search, so readers can choose batch size from actual
  slab-plus-adsorbate chemistry rather than molecule-only intuition.
- Compared two Toolkit model choices in the calibration: default
  `medium-mpa-0` and surface-specialized `mh-1 + oc20_usemppbe`. The notebook
  now makes the trade-off explicit: specialized surface validation can cost
  more VRAM, so the best batch size is model dependent.
- Added `helpers/batch_calibration.py` and
  `plot_adsorption_batch_calibration(...)`. The notebook-visible cells keep the
  chemistry, model list, batch sizes, and summary table explicit; timing,
  memory collection, OOM handling, and plot formatting live in helpers.
- Ran the calibration on `ws-loc` for 27 H2O/TiO2(110) starting configurations
  with 75 atoms each, 31 active atoms each, and a 40-step FIRE2 cap. During the
  run another Toolkit Python process held about 73 GB, so the memory-headroom
  values reflect a shared-GPU state with about 21.6 GB free.
- Measured full-scope calibration results:
  `medium-mpa-0` plateaued around `8-24` structures per batch
  (`4.61-4.65 structures/s`, about `345-349 atoms/s`);
  `mh-1 + oc20_usemppbe` plateaued around `8-24` structures per batch
  (`3.02-3.06 structures/s`, about `227-229 atoms/s`) and used more memory.
  The conservative recommendation rule picked batch `8` for `medium-mpa-0`
  and batch `4` for `mh-1 + oc20_usemppbe` under the current free-memory state.
- Wrote both full and short precomputed calibration artifacts:
  `cached_responses/adsorption-search/adsorption_batch_calibration_h2o_tio2_110_full.csv`,
  `cached_responses/adsorption-search/adsorption_batch_calibration_h2o_tio2_110_short.csv`,
  `assets/images/plots/adsorption_batch_calibration_h2o_tio2_110_full.png`,
  and `assets/images/plots/adsorption_batch_calibration_h2o_tio2_110_short.png`.
- Executed the notebook in read-only presentation mode on `ws-loc` with the
  refreshed artifacts: all 50 code cells passed in `7.7 s`. The executed audit
  copy is
  `outputs/executed_notebooks/alchemi-mace-adsorption-search-precomputed-check.ipynb`,
  and the main notebook now contains the refreshed outputs.
- Verification: helper `py_compile` passed, notebook code cells parse after
  ignoring notebook magics, and focused remote tests passed:
  `pytest -q tests/test_cache.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`21 passed`).

### Production Refresh and Artifact Safeguards

- Set the notebook back to the full presentation scope and added an explicit
  artifact policy: saved tutorial outputs are read-only by default; live
  exploratory recomputes write under `outputs/live_runs/<timestamp>/`; only
  `REFRESH_PRECOMPUTED_RESULTS = True` can refresh official cached artifacts.
- Added cache overwrite protection in `helpers/cache.py`. Existing cached JSON
  files now raise unless the caller passes `overwrite=True` or the notebook sets
  the intentional refresh environment flag.
- Added artifact overwrite guards to plotting, throughput, and OVITO rendering
  helpers. Presentation-mode plot/render cells display existing artifacts
  without replacing them.
- Changed OC20Dense validation plan construction so read-only plan cells do not
  rewrite `trajectory_selection.csv`; validation workflow defaults no longer
  force recomputation unless the notebook explicitly requests a refresh.
- Added a notebook-local `tutorial_relpath(...)` display helper so validation
  and result-table messages show transferable paths relative to the tutorial
  directory instead of workstation-specific absolute paths.
- First full refresh attempt reached the adsorption-grid relaxation cell and
  failed under the current shared-GPU state: another Toolkit kernel held about
  71.5 GB, leaving too little headroom for the `BATCH_SIZE = 24` adsorption
  batches. The notebook now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  before Torch import and uses `BATCH_SIZE = 12` for the full adsorption grid
  by default. Increase it only after checking free VRAM on the target GPU.
- Short mode now reuses the full H2O speedup cache by taking the requested
  smaller batch-size subset. This keeps the read-only precomputed path to two
  main toggles plus the explicit refresh switch, without creating another
  canonical cache variant.
- Reran the full refresh on `ws-loc` with `RUN_SCOPE = "full"`,
  `REFRESH_PRECOMPUTED_RESULTS = True`, and `BATCH_SIZE = 12`. The full
  adsorption relaxation cell completed in `431.9 s`; the notebook run exited
  with code 0 and wrote
  `outputs/full_notebook_execution/alchemi-mace-adsorption-search.refresh-executed.ipynb`.
- Reran the default read-only presentation notebook after pruning stale caches.
  It executed all 47 code cells in about 8 seconds with no error outputs and
  was copied back to `alchemi-mace-adsorption-search.ipynb`.
- Final focused remote verification passed:
  `pytest -q tests/test_cache.py tests/test_relaxation_backends.py tests/test_oc20dense_benchmark.py`
  (`21 passed`).

### Tutorial Language and Positioning Pass

- Polished the visible notebook language without changing the user's narrative
  order: ALCHEMI is framed as the enabling Toolkit/Toolkit-Ops/NIM stack, while
  this notebook stays on the direct Toolkit path.
- Replaced reader-facing internal labels with computational-chemistry wording:
  DFT-relaxed final geometries, DFT-level adsorption energies, single-point
  energies, batch size, execution path, and relaxation engine.
- Clarified the model story: the teaching adsorption search uses
  `medium-mpa-0`; the OC20Dense validation cells use the surface-specialized
  `mh-1 + oc20_usemppbe` model.
- Removed reader-facing `BGRReply`, `RELAXATION_BACKEND`, and raw backend
  wording from the main notebook source; hidden helper compatibility remains.

### Provenance-Guarded OC20Dense Rerun

- Reframed Test A as a trajectory replay check: MACE-usemppbe relaxes the
  same OC20Dense starting structures, then the notebook compares the
  MACE-relaxed endpoint against the released DFT trajectory endpoint by relaxed
  Eads and RMSD. Single-point energies on DFT-relaxed final adslab geometries
  now belong only to the separate fixed-geometry ranking section.
- Added a quick `medium-mpa-0` vs `mh-1:oc20_usemppbe` cost check on `ws-loc`.
  For 92 NH3 fixed-geometry single-point energies at batch size 12, energy-evaluation time
  was effectively the same (`1.48 s` vs `1.42 s`), while peak allocated GPU
  memory roughly doubled (`2.85 GB` vs `6.12 GB`). For the three-structure
  H2O/NH3/N2 relaxation replay, MH-1 was modestly slower (`6.47 s` vs
  `5.11 s` relaxation kernel time) and again used about twice the memory.
- Added Toolkit model provenance to OC20Dense benchmark batches, rows, and
  run metadata: checkpoint, head, device, dtype, and D3 state. Cached batches
  now recompute when these fields do not match the active model settings.
- Added a fixed-geometry-only Eads mode for the NH3 ranking check. This keeps
  the neutral-gas Eads control from accidentally combining current
  `mh-1:oc20_usemppbe` single points with relaxed-adslab energies produced by
  another checkpoint.
- Updated the visible validation cells through the notebook MCP bridge against
  the live Toolkit kernel. The saved notebook still needs one rerun/save pass
  once the bridge is live again so outputs match the current source.
- Current NH3 fixed-geometry result for `system_id=72_7104_115`, 92 released
  DFT-relaxed final geometries, `mh-1` with `TOOLKIT_HEAD=oc20_usemppbe`: DFT-rank-1
  anchored RMSE `0.0716 eV`, MAE `0.0598 eV`, bias `-0.0458 eV`, Spearman
  `0.943`; MACE top geometry is the released DFT rank-1 geometry.
- The neutral-gas Eads control remains a negative/reference-convention control:
  absolute defined Eads RMSE is about `0.878 eV` for the 92 NH3 fixed
  geometries, while the rank-1 anchored fixed-system relative metric remains
  `0.0716 eV` because the shared gas/surface offsets cancel.
- Reran Test A, the three-record trajectory-backed check, through the live
  notebook kernel with `mh-1` and `TOOLKIT_HEAD=oc20_usemppbe`. Defined
  neutral-gas Eads errors on MACE-relaxed trajectory endpoints are now the
  displayed trajectory metric; DFT-relaxed final single-point errors are no longer
  shown in this section.
- Reduced the live notebook NH3 validation batch size to `12` so it runs
  reliably when the current Jupyter kernel already holds GPU memory. This is
  a runtime reliability setting, not a change to the scientific metric.
- Verification: `jq empty` passed for the notebook; focused tests
  `test_relaxation_backends.py` and `test_oc20dense_benchmark.py` passed
  (`15 passed`).

### MACE Checkpoint Sweep for OC20Dense Ranking

- Added a DFT-rank-1 anchored relative-energy metric to the OC20Dense
  DFT-relaxed final single-point script. For each fixed system, both DFT and MACE gaps
  are measured from the released DFT minimum geometry, so MACE is not allowed to
  choose its own zero-energy reference before errors are calculated.
- Re-evaluated all 92 released `*NH3` DFT-relaxed final geometries for
  `system_id=72_7104_115` across MACE-MP, MACE-OMAT, MACE-MATPES, and MACE-MH
  variants. Best result found so far: `mh-1` with explicit
  `oc20_usemppbe` head, RMSE `0.0716 eV`, MAE `0.0598 eV`, Spearman `0.943`,
  and the DFT rank-1 geometry recovered as MACE rank 1.
- Best simpler single-head/default Toolkit checkpoint in this sweep:
  `medium-mpa-0`, RMSE `0.178 eV`, MAE `0.150 eV`; its top MACE geometry is
  DFT rank 3 with a DFT gap of `0.0128 eV`.
- Added `TOOLKIT_HEAD` support to the Toolkit execution path so multi-head MACE
  checkpoints can be selected reproducibly. Without this, `mh-1` uses its
  default first head, which is not the OC20 surface-catalysis head.
- Updated the tutorial notebook through the notebook MCP bridge. The visible
  settings now expose `TOOLKIT_HEAD`, the validation section uses
  `mh-1` with `TOOLKIT_HEAD=oc20_usemppbe` in model-specific output
  directories, and the NH3 ranking display now reports the DFT-rank-1 anchored
  relative landscape metric instead of the older model-shifted Eads table.
- Ran an absolute neutral-gas Eads control for `mh-1:oc20_usemppbe`. It failed
  as an accuracy story: neutral-gas subtraction gives RMSE `0.869 eV` against
  released OC20Dense adsorption energies. Keep the tutorial validation focused
  on fixed-system relative landscapes/ranking and use DFT follow-up for
  absolute adsorption-energy claims.
- Tracked execution note:
  `docs/mace_checkpoint_sweep_2026-05-15.md`. Ignored raw artifacts:
  `outputs/explorations/mace_checkpoint_sweep/`.
- Verification: `python -m py_compile` passed for touched Toolkit/OC20Dense
  scripts; `pytest -q tests/test_relaxation_backends.py
  tests/test_oc20dense_benchmark.py` passed (`15 passed`).

### OC20Dense Benchmark Cells

- Moved OC20Dense validation workflow assembly into
  `helpers/validation_workflows.py`. The notebook now calls Python functions
  in-process instead of shelling out to CLI scripts.
- Reduced the notebook validation settings cell from orchestration code to the
  reader-facing choices: three closed-shell trajectory-replay records, the NH3
  fixed-system ranking target, preview ranks, output roots, batch sizes,
  validation step cap, and force-recompute toggle.
- Kept the benchmark run cells explicit: they call
  `build_trajectory_stage_plan(...)` or `build_nh3_ranking_stage_plan(...)`
  with visible arguments, then run each workflow step with
  `run_validation_step(step)`.
- Added callable entry points to the four reproducibility scripts so the same
  OC20Dense logic can be used from the notebook without subprocess wrappers:
  trajectory replay, DFT trajectory checks, DFT-relaxed final Toolkit single-point energies,
  and Toolkit gas/surface-reference Eads.
- Removed `globals()` state checks from the notebook. Cells now either rely on
  the normal top-to-bottom notebook contract or raise a clear setup message.
- Verified through the live notebook MCP bridge with `%autoreload 2`: cells
  11 and 24-32 executed successfully against the remote `ws-loc` Toolkit
  kernel. The direct in-process path took about 12 seconds for trajectory replay
  and about 25 seconds for NH3 ranking.
- Replaced display-id HTML progress updates with an `ipywidgets.HTML`-backed
  progress display when widgets are available. This fixes the VS Code/Jupyter
  case where the HTML card appeared but did not repaint while cell 26 or 30 was
  running. Static HTML remains the fallback for non-widget environments.
- Added shifted relative-energy metrics for OC20Dense Eads tables. For each
  fixed `system_id`, DFT and MACE Eads values are independently shifted so the
  best structure is zero, then the same geometries are compared row by row. This
  separates landscape/ranking accuracy from absolute gas/surface reference
  offsets.
- Superseded by the provenance-guarded rerun above: for the selected
  `mh-1:oc20_usemppbe` validation path, use the DFT-rank-1 anchored
  fixed-geometry RMSE `0.0716 eV`; treat neutral-gas Eads as a
  reference-convention control, not as the primary accuracy metric.

## 2026-05-14

### Notebook Progress UI

- Moved the reusable notebook progress strip into
  `helpers/visualization.py` and exported it as `make_notebook_progress`.
- H2O speedup and OC20Dense validation cells now use the helper instead of
  carrying inline progress-card code.
- The actual adsorption relaxation cell now uses the same helper progress strip
  and advances by completed structures after each Toolkit batch returns.
- Rule for future long-running cells: update progress only after the completed
  Toolkit batch, validation stage, or recomputation unit returns; keep batching
  details in the message and UI rendering in helpers.
- Simplified the H2O batching hello-world molecule builder: no artificial
  rotations and no periodic boundary flag for the isolated gas molecule. The
  notebook and standalone H2O saturation script now use the same clear pattern.

## 2026-05-13

### Tracking

- Added this changelog as the lightweight project-tracking surface. It is not a
  full handoff document; it is the first place to summarize meaningful changes
  after notebook edits, benchmark runs, validation work, or workflow decisions.
- Reporting rule for future entries: include run scope, structure count,
  surfaces or Miller indices, batch size, step cap, rerun cap, convergence
  count, artifact location, and whether trajectories/logs were written.

### Current State

- Main tutorial source:
  `part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`.
- Current narrative direction: batched, GPU-native atomistic simulation changes
  the practical scale of computational-chemistry search; adsorption is the
  worked example.
- Current execution direction: Toolkit-only reader path for now. Service/API
  variants should be separate and reuse the same result schema.
- Current notebook control surface: two compute scopes only,
  `RUN_SCOPE = "short"` and `RUN_SCOPE = "full"`. Precomputed outputs are
  controlled independently with `USE_PRECOMPUTED_TUTORIAL_RESULTS` and
  `USE_PRECOMPUTED_ACCURACY_BENCHMARK`.

### Open Work

- Add the missing material-specific Miller-index sweep. The current fixed-panel
  run covers Cu(111), Pd(111), and Al2O3(0001); it does not yet answer how the
  best adsorption structure changes across multiple Miller-index surfaces of
  one material.
- Proposed next sweep: rutile TiO2 across `(110)`, `(100)`, `(101)`, and
  `(001)` if slab generation, atom counts, tags, and site detection are stable.
- Add a visual-review table that points to a few clean trajectories, difficult
  or rerun trajectories, and cross-Miller-index winners once the Miller sweep
  exists.

## 2026-05-12

### Validation and Artifacts

- Regenerated the full validation/artifact pipeline on `ws-loc` without syncing
  the full 29 GB OC20Dense trajectory archive.
- Minimal remote reference-data sync used `rsync --ignore-existing --relative`
  and moved 11.08 GB: the LMDB, small pickles, selected DFT trajectories, and
  selected clean-surface trajectories.
- Full pipeline wall time on `ws-loc`: about 17.5 minutes.
- Stage scope:
  - OC20Dense Toolkit relaxation and initial single-point validation: 222
    records, 6 minutes.
  - DFT trajectory comparison/conversion, DFT-relaxed final MACE single-point energies, MACE
    adsorption-energy references, and summary aggregation: each under 1 minute.
  - Fixed-surface full panel: 252 structures, 9 minutes.
- Full-panel settings for this run: batch size 24, 300-step cap, 1000-step
  rerun cap, trajectories/logs written.
- Full-panel outcome: 252/252 converged and reliable; 4 rows used the rerun
  path and report rerun trajectories as the current trajectory paths.

### Scientific Checks

- OC20Dense DFT trajectory arithmetic matched released targets exactly:
  max absolute trajectory-target difference was 0.0 eV.
- Exact starting-frame parity held: max active-atom starting RMSD was
  `6.71e-7 A`.
- Accuracy summary over the closed-shell validation systems:
  initial-coordinate single-point top-1 within 0.10 eV was 3/3; DFT-relaxed final single-point top-1 within
  0.10 eV was 3/3; Toolkit relaxation top-1 within 0.10 eV was 2/3 and top-3
  within 0.10 eV was 3/3.

### Artifact Locations

- Remote run note:
  `part-1-batched-adsorption/docs/full_artifact_run_2026-05-12.md`.
- Remote validation artifacts:
  `ws-loc:/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/oc20dense_validation_run`.
- Remote full-panel artifacts:
  `ws-loc:/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/full_panel_toolkit`.
- Remote log:
  `ws-loc:/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/logs/ws_loc_full_artifacts_20260512.log`.

### Sync Guard

- Do not sync `part-1-batched-adsorption/data/reference/oc20dense` wholesale.
  It is about 40 GB locally, mostly from `oc20_dense_trajectories.tar.gz`.
- Routine source syncs should exclude `outputs/`, archives, `.traj`,
  `.extxyz`, `.venv*`, and cache directories unless a specific artifact is
  intentionally being moved.

## 2026-05-11

### Tutorial Direction

- Reframed the tutorial around batched atomistic simulation as the headline and
  adsorption configuration search as the worked example.
- Clarified ALCHEMI's role as the enabling GPU workflow layer: batching, model
  wrappers, optimizers, constraints, metadata, accelerated kernels, and service
  deployment.
- Kept the reader path Toolkit-first and D3-disabled to match the non-D3
  OC20Dense reference convention.

### Notebook Structure

- Added the H2O batch-speedup warm-up before the adsorption search.
- Added validation as an early checkpoint in the main tutorial flow: choose a
  model/tooling path, then check it against reference-backed surface chemistry.
- Kept review markers where human, reference, or visual review is still needed.

### Benchmark Planning

- Recorded the batching decision for teaching-session runs:
  H2O molecule throughput saturates around the 1024-4096 region, while real
  adsorption throughput plateaus around batch 24 for the tested oxide cases.
- Kept homogeneous pair batches for adsorption runs because convergence is
  batch-level and one difficult structure can hold a batch open.

### Surface-Screen Panel Refactor

- Reworked the new 9-facet adsorption screen so the scientific choices are visible in the notebook, not hidden inside a helper module. The notebook now defines the surface list, Miller indices, slab builders, adsorbates, orientations, site classes, and six-start rule directly in tutorial cells.
- Split the surface construction into tutorial-paced cells: Cu facets, rutile TiO2 facets, TiN facets, then a separate panel assembly/count step. Each family has an immediate table/widget visual check before the full grid is built.
- Reduced `helpers/surface_screen.py` to bookkeeping: output paths, geometry-audit tables, step statistics, pair summaries, difficult-case tables, and heatmap-ready result tables.
- Added a headless `scripts/run_surface_screen.py` path for artifact refreshes. Its panel definition mirrors the visible notebook cells and is used only so long runs can be executed outside the interactive notebook.
- Added the same CUDA wheel library re-exec guard to `scripts/run_surface_screen.py` that the validation scripts use, so NVRTC builtins from the Toolkit environment are visible during long headless refreshes.
- Updated the surface-screen scope to 9 slabs x 4 adsorbates x 6 starts = 216 adsorption relaxations, plus 9 clean-slab and 4 gas-reference relaxations. The screen uses the `mh-1` checkpoint with the `oc20_usemppbe` head and D3(BJ) disabled.
- Remote verification on `ws-loc`: focused tests passed (`47 passed`), and the surface-screen geometry audit passed for all 216 starting structures. The minimum initial adsorbate-slab distance after the height adjustment is above 1.0 A.
