# Part 1 Tutorial Changelog

This is the running change log for the ALCHEMI Toolkit adsorption-search
tutorial. Keep this file safe to ship: do not include tables, figures, cached
outputs, or metric summaries produced by license-gated models.

## 2026-05-18

### License-Safe Open-MACE Path

- Removed MACE-MH-1 / OC20-head execution from the active runnable tutorial
  path. That model is license-gated and must not appear in shipped generated
  artifacts or reported metrics for this NVIDIA tutorial.
- Switched active helper/script/cache defaults to open `medium-mpa-0` output
  stems: `surface_screen_v1_mace_mpa0`,
  `oc20dense_closed_shell_trajectory_mace_mpa0`, and
  `oc20dense_nh3_92_fixed_geometry_mace_mpa0`.
- Updated notebook source through the local notebook MCP bridge. Stale stored
  outputs from license-gated model runs were cleared through the bridge rather
  than by direct `.ipynb` JSON editing.
- Current reported validation baseline is the open `medium-mpa-0`
  fixed-geometry NH3 ranking check: RMSE `0.178 eV`, MAE `0.150 eV`,
  Spearman `0.786`; the MACE top geometry is DFT rank 3, only `0.0128 eV`
  above the released DFT minimum.
- Batch-size calibration now compares open MACE sizes rather than a
  license-gated surface-specific checkpoint.
- Dependency review against `origin/dev`: no new Python/package dependency
  manifests were added by this branch. The only dependency-file delta remains
  deletion of the old `part-1-nim/environment.yml`.

### Bundled OC20Dense Validation Pack

- Added `data/reference/oc20dense-validation-pack.tgz` as the tracked
  reference-data artifact for live validation. It contains the released DFT
  trajectories used by the notebook widgets and checks: three closed-shell
  replay trajectories, all 92 NH3 fixed-geometry ranking trajectories, clean
  surface references, and OC20Dense mapping/target files.
- Kept the expanded `data/reference/oc20dense/` folder gitignored. The
  validation helper unpacks the bundled tarball when live validation needs the
  reference folder and it is not already present.
- The validation pack is source/reference data, not saved model output.

### GitHub-Clean Notebook Restructure

- Restructured the post-validation half of the notebook so the surface-screen
  section separates visible science choices, starting-geometry assembly,
  reference loading/relaxation, batched relaxation, result interpretation, and
  artifact writing.
- Moved final result file-writing into `helpers.surface_screen` while keeping
  chemistry, reference convention, and analysis steps visible in notebook
  cells.
- Kept the main notebook's scientific grid visible and left only formatting,
  plotting, cache, validation bookkeeping, and artifact utilities in helper
  modules.
- Tightened saved-output guards: benchmark scripts refuse accidental writes
  under `outputs/precomputed/` unless an explicit refresh guard is set.
- Archived generated outputs, runtime/model caches, local expanded OC20Dense
  data, review-candidate images, PDFs, and local spillover outside the
  GitHub-minimal tutorial tree.
- Verification before the license cleanup: notebook JSON parsed, helper modules
  compiled, and focused tests passed locally.

## Earlier History

Detailed pre-cleanup development history lives in git history and local audit
notes. It is intentionally not duplicated here because some earlier exploration
used license-gated models that are no longer shipped as tutorial artifacts.
