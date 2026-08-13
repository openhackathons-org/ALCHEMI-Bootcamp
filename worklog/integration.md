# Integration worklog

Use this file only for bounded setup and integration passes that edit root or shared files.

## 2026-08-11 — v3 worktree and coordination files

Status: complete

Observed:
- Clean worktree created at `/home/nfedik/projects/tutorials/v3`.
- Branch `v3-api-first` starts at `d09454bf4034417f94f8f0da57db524c40c1ddc2`.
- Dirty v2 work remains in `/home/nfedik/projects/tutorials/v2` as read-only source material.

Changed:
- `HANDOFF.md`
- `WORKLOG.md`
- `worklog/*.md`
- `.python-version`
- `pyproject.toml`
- `environment/README.md`
- `environment/runtime-pins.toml`
- `environment/check_runtime.py`
- `environment/prewarm_assets.py`
- `scripts/v3-sync`
- `scripts/v3-run`
- `data/nci_atlas/*`
- `shared/alchemi-dark.mplstyle`
- `shared/README.md`
- `uv.lock`

Validation:
- `uv lock`: resolved 206 packages from the fixed sources.
- `./scripts/v3-sync`: installed 201 packages under `/tmp/alchemi-v3-runtime`.
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- shared Matplotlib style and Rich progress column pattern: passed smoke checks.

## 2026-08-11 — remove deferred application notebooks

Status: complete

Changed:
- removed `part-2-batched-adsorption-toolkit/` from v3;
- removed `part-3-batched-melting-toolkit/` from v3;
- removed their root navigation, Docker copy entries, source-list entries, and ignore rules.

Recovery:
- both tutorials remain available on the existing v2 and reworked branches.

## 2026-08-12 — consolidate curriculum and authoring system

Status: in progress

Observed:
- Delivery target: 90-minute fundamentals session at ACS 2026 Chicago.
- The earlier handoff duplicated curriculum, design, ownership, environment,
  and validation rules across several active-looking documents.
- The v2 organic IR and PALIRS work remains read-only source material for a
  later application capstone.

Changed:
- made `ALCHEMI_TUTORIAL_PRINCIPLES.md` the single authoring and curriculum
  guide;
- converted `TOOLKIT_API_CURRICULUM.md` to a technical API inventory;
- moved ownership and integration protocol to `WORKLOG.md`;
- added synchronized callout and curriculum-map assets under `shared/`;
- installed and validated the `$alchemi-tutorial-authoring` skill;
- routed historical design files to the canonical guide;
- retired `HANDOFF.md` after transferring its current facts.

Validation:
- skill metadata and structure: passed `quick_validate.py`;
- skill checker: Ruff and formatting checks passed;
- shared and skill curriculum-map/callout assets: identical;
- Markdown patch whitespace: pending final consolidation check.

Next:
- rebuild Part 01 through the notebook bridge using the new skill;
- refine the skill after fresh-kernel and rendered learner review.

## 2026-08-12 - add a voice-preserving writing and feedback loop

Status: complete

Observed:
- The canonical guide covered pacing and visual design but did not state how to
  preserve an educator's voice or turn notebook feedback into a general rule.
- Peter Yang's `no-ai-slop` skill is MIT licensed, active, and narrowly scoped
  to editing. The STE Plain Writing repository offers useful technical-sentence
  rules but has a short public history, so it was used as a reference rather
  than installed. Anthropic's doc-coauthoring skill supplied the fresh-reader
  review pattern.

Changed:
- expanded `ALCHEMI_TUTORIAL_PRINCIPLES.md` with voice-preserving edits, stable
  terminology, plausible validation examples, equal-work performance openings,
  and a reusable three-part feedback delta;
- updated the installed `$alchemi-tutorial-authoring` skill and checker with the
  same general workflow;
- installed the global `$no-ai-slop` skill for future editing passes;
- replaced the internal `primer` checker terminology with `opening result`.

Validation:
- authoring skill structure: passed `quick_validate.py`;
- Part 01 authoring checker: zero errors and zero warnings;
- checker source parse: passed;
- fresh-reader review assigned with only learner-facing notebook context.

Final reusable deltas:
- save reviewed outputs when prose reports an observed result;
- identify hardware and use parallel operation labels in performance results;
- define graph nodes and the later neighbor-edge construction separately;
- reserve identity claims for explicit identifiers or metadata checks;
- state the exact invariant a validation example proves and preserve relevant
  system fields; and
- bound learner edits so the advertised success signal remains valid.

## 2026-08-12 - consolidate tutorial policy and API reference

Status: complete

Changed:
- replaced the overlapping design documents with `TUTORIAL_GUIDE.md`, the one
  source for curriculum, teaching, writing, visuals, scientific care, and
  review;
- renamed the technical inventory to `TOOLKIT_API_REFERENCE.md` and kept exact
  namespaces, APIs, shapes, relationships, release constraints, and course
  placement there;
- removed the retired design handoff, learner-gap audit, remaster plan, and
  redirect documents after their durable guidance was transferred;
- updated active repository, build, shared-asset, and historical-v2 links; and
- reduced the installed `$alchemi-tutorial-authoring` skill to an operating
  workflow that reads the two project authorities.

Validation:
- frozen v3 runtime check passed;
- authoring skill structure passed `quick_validate.py`;
- authoring checker reported zero errors and zero warnings for Part 01;
- Part 01 scoped tests passed: 46 tests;
- active links resolve, and remaining old filenames occur only in this
  historical integration log.

## 2026-08-12 - curriculum icon scale and alignment

Status: implemented; rendered review required

Educator delta:
- Icons needed about ten percent more visual weight at notebook width.
- Domain decomposition needed one tile visibly separating from a compact group.
- Capability cards aligned exactly with lesson cards and read as a second
  row-by-row curriculum.

Changed:
- Increased all curriculum icons from scale `0.82` to `0.90`.
- Rebuilt the domain-decomposition icon as three grouped tiles and one detached tile.
- Offset capability cards above or below their source lesson row and connected
  them with short angled routes.
- Added the reusable alignment rule to `TUTORIAL_GUIDE.md`.

Validation:
- regenerated all eight SVGs from the shared generator;
- all eight SVGs parsed and contained eight `0.90` icons, six offset capability
  routes, and the detached decomposition tile;

## 2026-08-12: semantic course-map routing

User feedback:

- Connector corners needed visible rounding on angled routes.
- The detached domain-decomposition tile needed to match the grouped tiles.
- Capability links needed to follow Toolkit API reuse across lessons.
- Route crossings needed a flowchart-style separation mark.
- The right column needed consistent visual spacing.

Changes:

- Replaced row-to-row capability links with direct API reuse bundles grounded
  in `TOOLKIT_API_REFERENCE.md`.
- Added dedicated rails, explicit junction dots, and background-width crossing
  gaps so route intersections do not read as joins.
- Generated angled output routes with quadratic corner curves.
- Computed right-column positions from card heights and one equal inter-card
  gap.
- Made the detached domain-decomposition tile the same size as the group tiles.

Validation:

- `ruff` passed for the generator and focused contract test.
- The two course-map contract tests passed.
- All eight generated SVGs parsed successfully.
- `git diff --check` passed.
- The full Part 01 contract file reported 26 passing tests and one current
  notebook-content failure: the notebook does not contain the architecture
  image reference required by
  `test_ecosystem_orientation_links_and_acknowledgments_are_present`.

## 2026-08-12: compact interactive course map

The semantic-rail revision was too dense for a course orientation graphic.

Changes:

- Installed the official jgraph `drawio` skill at commit `14b318b` after
  reviewing its Apache-2.0 license and confirming that the skill contains only
  instructions.
- Returned the map to one course spine, six capability routes, and one local
  bracket for Parts 04–06.
- Kept rounded angled connectors and equal right-column spacing.
- Added SVG links for the five notebooks that currently exist. Planned Parts
  06–08 remain plain nodes until their notebook files exist.
- Documented the HTML `object` embed required for clickable SVG links.
- Ruff passed for the generator and curriculum contract test;
- focused curriculum-map contract tests passed: 2 tests;
- `git diff --check` passed.

User review required:
- inspect `shared/curriculum-map-01.svg` and `shared/curriculum-map-08.svg` at
  notebook width and confirm the icon scale, detached tile, and offset right
  column feel balanced.

## 2026-08-12 — separate convergence detection from hook dispatch

Status: shared API fact resolved; N04/N05 notebook fixes pending

Release constraint:
- Verified against installed `nvalchemi-toolkit==0.2.0`, pinned to
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`. Do not generalize this behavior
  to another release without rechecking its source.

Installed-source evidence:
- `BaseDynamics.__init__` stores `convergence_hook` separately from
  `HookRegistryMixin._init_hooks(hooks)`.
- `BaseDynamics.step(...)` dispatches registered
  `DynamicsStage.AFTER_STEP` hooks, then `_check_convergence(...)` directly
  calls `self.convergence_hook.evaluate(batch)` on every step. Returned graph
  indices drive the step result, `DynamicsStage.ON_CONVERGE`, and the
  all-graphs early exit in `BaseDynamics.run(...)`.
- `ConvergenceHook.evaluate(...)` returns converged graph indices and does not
  inspect `stage`, `frequency`, `source_status`, or `target_status`; it does not
  migrate `batch.status`.
- `HookRegistryMixin._call_hooks(...)` dispatches registered hooks at their
  matching stage when `step_count % hook.frequency == 0`, including step 0.
  `ConvergenceHook` defaults to `DynamicsStage.AFTER_STEP`, and its
  `__call__(ctx, stage)` performs status migration only when both
  `source_status` and `target_status` are set and `batch.status` is present.
  Registry dispatch discards its convergence indices, so registration alone
  does not drive the host's `ON_CONVERGE` path or early exit.
- `FusedStage.__init__` creates separate frequency-1, `AFTER_STEP`
  `ConvergenceHook` instances with source/target status codes for sub-stage
  transitions, copying configured detector criteria when present. Those
  registered hooks own fused status migration; each sub-stage's
  `convergence_hook=` remains the separately evaluated detector.

Official NVIDIA documentation inspected:
- [Convergence Criteria](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/convergence.html)
  explicitly shows the `convergence_hook=` detector and registered migration
  hook as two attachment paths.
- [Hooks — Core Framework](https://nvidia.github.io/nvalchemi-toolkit/modules/hooks.html)
  documents `HookRegistryMixin`, stage matching, zero-based frequency dispatch,
  and registration through `hooks=` / `register_hook(...)`.
- [`BaseDynamics`](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/_generated/nvalchemi.dynamics.BaseDynamics.html)
  documents the `AFTER_STEP` → convergence check → `ON_CONVERGE` order.
- [`ConvergenceHook`](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/_generated/nvalchemi.dynamics.ConvergenceHook.html)
  documents `evaluate(...)`, `stage`, `frequency`, and optional
  `source_status` / `target_status`.

Documentation caveat:
- The broader [Hooks user guide](https://nvidia.github.io/nvalchemi-toolkit/userguide/hooks.html)
  also places a no-status `ConvergenceHook` in `hooks=[...]` and says
  single-stage convergence stops updates. That sentence conflates the two
  paths in the pinned implementation: only `convergence_hook=` feeds
  `BaseDynamics` convergence, while registered status migration requires both
  status arguments. The installed pinned source is unambiguous.

Affected requests:
- `N04-REQ-001`: Part 04 should preview that `hooks=` owns registry
  stage/frequency/lifecycle dispatch and hand detector timing to Part 05.
  Status: open pending the N04-owned notebook fix.
- `N05-REQ-001`: Part 05 should teach `convergence_hook=` as host-driven direct
  `evaluate(...)`, and identify status migration as a separately registered
  `ConvergenceHook` behavior used by `FusedStage`.
  Status: open pending the N05-owned notebook fix.

Changed:
- `TOOLKIT_API_REFERENCE.md`
- `worklog/integration.md`

Validation:
- Re-read both modified sections; terminology, wrapping, and Markdown link
  syntax are consistent.
- All five added official NVIDIA documentation links resolved during review.
- Edited-file diagnostics reported no errors.
- Whitespace validation passed.

## 2026-08-12 — close clean-runtime D3 prewarm request

Status: complete; `N03-REQ-001` closed

Root cause:
- `scripts/v3-sync` already exported
  `ALCHEMI_D3_PARAM_FILE=$ALCHEMI_V3_RUNTIME_ROOT/dftd3/dftd3_parameters.pt`
  and invoked `environment/prewarm_assets.py`, but prewarm resolved and
  verified only the AIMNet alias. No D3 extraction or save API ran, so a newly
  synchronized runtime had only an empty `dftd3/` directory. N03 succeeded
  locally only because a manually generated file remained under the default
  runtime root.

Pinned source and official documentation evidence:
- Installed Toolkit `0.2.0` at
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` matches the
  [pinned `nvalchemi/models/dftd3.py` source](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18/nvalchemi/models/dftd3.py).
  `DFTD3ModelWrapper` delegates parameter loading to
  `load_dftd3_params(...)`; its missing-file path calls the public
  `extract_dftd3_parameters(...)` and `save_dftd3_parameters(...)` helpers.
  Extraction downloads the official Bonn `dftd3.tgz`, requires archive MD5
  `a76c752e587422c239c99109547516d2`, parses `dftd3.f` and `pars.f` in memory,
  and produces the four float32 tensors `rcov`, `r4r2`, `c6ab`, and `cn_ref`.
- Installed Toolkit-Ops `0.4.1` at
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2` matches the
  [pinned `D3Parameters` source](https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/c1e23460859a784e1d78043bcd1c8af0d1095fa2/nvalchemiops/torch/interactions/dispersion/_dftd3.py),
  which validates tensor types and the `[95]`, `[95]`,
  `[95, 95, 5, 5]`, `[95, 95, 5, 5]` shapes consumed by DFT-D3.
- The official NVIDIA Toolkit-Ops 0.4.1
  [molecular DFT-D3 example](https://nvidia.github.io/nvalchemi-toolkit-ops/main/examples/dispersion/01_dftd3_molecule.html)
  documents the same public extraction/save sequence and identifies the first
  download as roughly 500 KB. No opaque cache was copied.

Resolution:
- `environment/prewarm_assets.py` now keeps the existing AIMNet path intact,
  rejects an existing D3 file whose SHA-256 differs, and creates a missing file
  through Toolkit's public extraction/save helpers.
- Generation occurs in a private directory beside the destination with the
  exact target basename. The generated bytes must match the immutable
  `dispersion.generated_parameter_sha256` before an atomic `os.replace(...)`;
  the published file is checked again. Failed or mismatched generation never
  publishes a target, and a mismatched existing target is preserved for
  diagnosis.
- `environment/check_runtime.py` now requires the actual file named by
  `ALCHEMI_D3_PARAM_FILE` and validates its pinned SHA-256 in addition to
  checking that its path is inside the selected runtime root.
- `environment/test_runtime_assets.py` covers a clean custom runtime path,
  verified cache reuse, preservation of a mismatched cache, non-publication of
  mismatched generated data, and runtime-check failures for missing or
  mismatched files.
- `environment/README.md` records the one-time external source requirement.
  `scripts/v3-sync`, `scripts/v3-run`, and `runtime-pins.toml` required no
  change.

Clean D3-only portability command:

```bash
./scripts/v3-run python - <<'PY'
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from environment import check_runtime, prewarm_assets

runtime_root = Path(tempfile.mkdtemp(prefix="alchemi-v3-d3-final-"))
parameter_file = runtime_root / "dftd3" / "dftd3_parameters.pt"
os.environ["ALCHEMI_D3_PARAM_FILE"] = str(parameter_file)
prewarm_assets.main()
check_runtime.require_file_sha256(
    "D3 parameter file",
    parameter_file,
    prewarm_assets.PINS["dispersion"]["generated_parameter_sha256"],
)
print(f"clean_runtime_root={runtime_root}")
print(f"d3_bytes={parameter_file.stat().st_size}")
print(f"d3_sha256={prewarm_assets.sha256_file(parameter_file)}")
print("clean_d3_runtime_check=passed")
PY
```

Result:
- Clean custom root:
  `/tmp/alchemi-v3-d3-final-qhf5weu9`.
- Generated file size: `1,808,183` bytes.
- Generated SHA-256:
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.
- Toolkit's source-MD5 download, extraction, save, prewarm pin check, atomic
  publication, and independent runtime-check helper all passed.
- The existing `/tmp/alchemi-v3-runtime` cache was neither deleted nor changed
  by the clean-path test.

Additional validation:
- `./scripts/v3-sync`: passed; uv checked 201 packages, then AIMNet and D3
  reported their configured paths as verified. The current no-driver session
  emitted Warp CUDA discovery diagnostics already associated with runtime
  imports; they did not affect the exit status or either identity check.
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python
  `3.12.13`, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`, and the shared environment at
  `/tmp/alchemi-v3-runtime/venv`.
- `./scripts/v3-run pytest -q -p no:cacheprovider
  environment/test_runtime_assets.py`: 6 passed; 14 warnings are pinned
  TorchScript deprecations.
- `./scripts/v3-run ruff check environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python -m py_compile environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- Edited-file IDE diagnostics: no errors.

Closure and remaining risk:
- `N03-REQ-001` can close: a missing configured D3 table is now generated and
  verified before N03, while N03 continues to load it with
  `auto_download=False`.
- A first D3 generation in a new root still needs the official Bonn endpoint.
  Network failure, an upstream archive change, serializer drift, a missing
  file, or any identity mismatch fails synchronization loudly.
- A complete second 201-package runtime was not installed solely for this
  check because that would duplicate the large locked Torch/CUDA environment.
  The production D3-only path was exercised from a clean custom root, while
  the normal existing-root `v3-sync` integration and full runtime check were
  exercised separately.

## 2026-08-12 19:20 EDT — final N04/N05 integration verdict

Status: PASS

Closure date and context:
- 2026-08-12 19:20 EDT — independent coordinator integration review of the
  current N04 handoff, N05 ownership and numbering, and the shared pinned API
  contract.

Closed requests:
- `N04-REQ-001`: CLOSED — verified that N04 hands relaxation, convergence, and
  `BaseDynamics` workflow behavior to N05; N05 owns completed-update numbering
  while explicitly relating it to N04's zero-based `ctx.step_count`; and both
  lessons match the shared API contract separating `hooks=` registry dispatch,
  `convergence_hook=` host detection, and separately registered status
  migration.
- `N05-REQ-001`: CLOSED — verified the same N04 handoff, N05 ownership and
  numbering, and shared API-contract consistency.

Remaining independent gates only:
- N04 fresh CUDA execution and rendered learner review.
- N05 opt-in AIMNet2 CUDA test, fresh execution, and rendered HTML review.

## 2026-08-12 19:33 EDT — reopen N03 D3 publication request

Status: `N03-REQ-001` REOPENED; independent review returned changes required

Review finding:
- The prior staging design verified the generated file and then called
  `os.replace(staged_path, destination)`. A destination could appear after the
  initial `exists()` check but before `os.replace()`. In that race, prewarm
  would overwrite the winning destination even when its bytes did not match
  the immutable pin.

Required correction:
- Publish with a same-filesystem atomic no-clobber primitive.
- If another publisher wins, validate and accept matching bytes or fail while
  preserving mismatched bytes.
- Add deterministic exception, visibility, cleanup, winner, and concurrent
  publisher coverage before closing the request again.
- Clarify that a custom `ALCHEMI_V3_RUNTIME_ROOT` must remain exported for both
  synchronization and every later `v3-run`.

## 2026-08-12 19:34 EDT — close N03 no-clobber remediation

Status: complete; `N03-REQ-001` CLOSED after new evidence

Finding-by-finding resolution:
- Replaced `os.replace(...)` with `os.link(staged_path, destination)`. Staging
  is created beside the destination, so the link is same-filesystem. Linux
  `link(2)` creates the destination directory entry atomically and explicitly
  does not overwrite `newpath`; `EEXIST` identifies a concurrent winner. See
  the [Linux `link(2)` manual](https://man7.org/linux/man-pages/man2/link.2.html).
- On `FileExistsError`, prewarm hashes the winning destination. Matching bytes
  are accepted without replacing the winner; mismatched bytes raise while the
  winner remains byte-for-byte intact.
- Any other hard-link failure raises a clear
  `atomic no-clobber publication via hard link is unavailable` error naming
  the destination filesystem path and preserving the original `OSError`.
- The private `TemporaryDirectory` remains in the destination directory.
  Generator exceptions, generated-hash failures, successful publication,
  concurrent winners, and unsupported-link failures all clean staging.
- `environment/README.md` now uses
  `export ALCHEMI_V3_RUNTIME_ROOT=/path/in/scratch` followed by both
  `./scripts/v3-sync` and
  `./scripts/v3-run python environment/check_runtime.py`. It explicitly warns
  that a one-command assignment does not persist to later `v3-run` commands.
- No change was needed in `scripts/v3-sync`, `scripts/v3-run`, or
  `environment/runtime-pins.toml`.

Deterministic regression coverage:
- `environment/test_runtime_assets.py` now has 14 tests covering clean
  publication, verified-cache reuse, generated mismatch, generator exception
  after partial staging, a matching concurrent winner with preserved inode, a
  mismatched concurrent winner with preserved bytes, two synchronized
  publishers, destination invisibility during partial generation, staging
  cleanup on success and failure, a clear unsupported-hard-link error, runtime
  missing/mismatch checks, and unchanged AIMNet resolution/checksum ordering.
- Red evidence before implementation: 2 failed and 12 passed. The mismatched
  concurrent winner was overwritten and the hard-link-unavailable test did not
  raise while `os.replace(...)` remained.
- Green command:
  `./scripts/v3-run pytest -q -p no:cacheprovider
  environment/test_runtime_assets.py`.
  Result: 14 passed; 14 pinned TorchScript deprecation warnings.

Concurrent mismatch reproduction:
- The reproduction created verified staging bytes, then created a mismatched
  destination before publication. Prewarm raised with expected and actual
  SHA-256 values, left winner bytes
  `b'concurrent mismatched winner'`, and reported zero staging entries.
- Required invariant: `concurrent_mismatch_preserved=True`.
- Reproduction root:
  `/tmp/alchemi-v3-d3-race-2l6ggkqt`.

Linux/WSL publication evidence:
- `uname -srmo`:
  `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 GNU/Linux`.
- `stat -f -c 'tmp_filesystem=%T' /tmp`:
  `tmp_filesystem=ext2/ext3`.
- All normal, winner, visibility, cleanup, and two-publisher tests exercised
  the real `os.link(...)` syscall on that WSL filesystem. The unsupported
  mechanism path was separately forced with `EOPNOTSUPP` and produced the
  required clear failure without publishing a destination.

Deterministic generation and custom-root evidence:
- A single `./scripts/v3-run python - <<'PY' ... PY` production harness created
  two independent runtime roots, set each root's
  `ALCHEMI_D3_PARAM_FILE`, called `prewarm_assets.main()`, ran
  `check_runtime.require_file_sha256(...)`, and compared size and digest.
- Generation 1:
  `/tmp/alchemi-v3-d3-determinism-1-k9gcr34v/dftd3/dftd3_parameters.pt`.
- Generation 2:
  `/tmp/alchemi-v3-d3-determinism-2-ee5mw9td/dftd3/dftd3_parameters.pt`.
- Both files are `1,808,183` bytes with SHA-256
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`;
  both roots had zero staging entries after publication.
- Harness invariants:
  `deterministic_generation_twice=True` and
  `clean_custom_root_d3_only=True`.
- The existing `/tmp/alchemi-v3-runtime` D3 file was not deleted, replaced, or
  used as the source for either generation.

Final commands and results:
- `./scripts/v3-run ruff check environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python -m py_compile environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python
  `3.12.13`, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`, and
  `/tmp/alchemi-v3-runtime/venv`.
- `./scripts/v3-sync`: exit 0; 201 packages checked, AIMNet verified at its
  existing pinned path, and D3 verified at its configured pinned path. The
  no-driver session emitted the existing Warp CUDA discovery diagnostics.

Closure recommendation and residual boundary:
- Close `N03-REQ-001`. The review race is covered by deterministic tests and
  direct reproduction, and clean generation now uses an atomic no-clobber
  primitive verified on the supported WSL/Linux filesystem.
- Filesystems that do not support hard links fail explicitly rather than
  falling back to a clobbering or non-atomic mechanism. First clean generation
  still requires the fixed-MD5 official Bonn download; those external
  availability and upstream-identity risks remain unchanged.

## 2026-08-12 — revise generated playbook pedagogy

Status: plan updated; implementation approval pending

Architecture retained:
- Deep lesson notebooks remain canonical.
- The working title from this planning pass was superseded by
  **ALCHEMI Core Playbook**.
- The playbook is generated from reviewed tagged blocks plus authored
  `transitions.ipynb`.
- A manifest declares block order, teaching depth, and standalone targets; a
  lockfile detects source drift.

Core pacing decision:
- Keep 80 minutes of planned learner content plus a 10-minute recovery buffer.
- Use 19 visible actions: orientation 1; AtomicData/Batch 4; prepared Zarr
  glimpse 2; supplied model 3; hook 3; FIRE2 3; coherent final-state recovery
  3.
- Replace the prior 12-minute, three-action Zarr block with a five-minute,
  two-action glimpse: open a prepared reader and inspect one model-ready
  `Batch`.
- Allocate the seven reclaimed minutes to Batch boundaries (+2), the supplied
  model contract (+1), hook dispatch (+1), FIRE2 convergence/run (+2), and
  coherent recovery (+1).
- Give profiling zero core actions. At most retain a one- or two-line pointer
  to the in-progress Part 07; do not teach `GPUBuffer`, `Batch.put`, or pipeline
  profiling.
- Keep the workshop exercise-free.

Navigation and status rule:
- Append one `Go deeper — Standalone tutorial` cell after every substantive
  playbook section.
- Label unfinished targets exactly as `Standalone tutorial (in progress)`.
- Treat Parts 01–05 as existing targets, while noting that Part 02 still has a
  validation-review gate and current Part 03 still combines supplied and
  custom adapter material.
- Name Parts 06–09 honestly as in-progress targets; do not present them as
  finished pages.

Sequencing retained:
- FIRE2 precedes interpreted MD.
- A complete potential precedes scientific MD.
- Fine-tuning remains an optional bounded extension and does not automatically
  supply a domain-parallel custom model.
- Optional extensions remain custom wrapping, short live fine-tuning, and
  exact-pin cached domain decomposition.

Changed:
- updated the existing interactive curriculum canvas at
  `/home/nfedik/.cursor/projects/home-nfedik-projects-tutorials-v3/canvases/alchemi-curriculum-plan.canvas.tsx`;
- updated this integration worklog only; no notebook or tutorial-guide source
  was changed.

Validation:
- Canvas TypeScript check: no errors.
- Existing Parts 01–05 targets were reconciled against current notebook paths.
- Planned Parts 06–09 are explicitly marked in progress.

Next:
- Keep the approval gate pending until the user explicitly authorizes
  implementation of the tagged-block generator, transitions, manifest, and
  lockfile.

## 2026-08-12 — curate the molecular interaction backbone

Status: plan updated; implementation approval pending

Scientific route:
- Use **NCI Atlas → AIMNet2 → phenol/N-methylacetamide** as the continuous
  playbook backbone.
- Preserve the neutral hydrogen-bond, dispersion-dominated, and ionic
  hydrogen-bond classes.
- Preserve
  `E_int = E(AB) - E(A) - E(B)` with ordered `AB/A/B` triplets.
- The selected evidence contains 30 interaction points: three systems at ten
  separations each, represented by 90 fragment graphs.
- Retain NCI Atlas source revision, checksum, paper links, Jan Řezáč/NCI Atlas
  contributor attribution, and CC BY 4.0 license.

Model and scaling route:
- Use the current Part 03 public adapters and composition path for the complete
  AIMNet2 + Coulomb + D3 potential; do not restore historical direct-wrapper
  code to the playbook.
- Reuse the periodic 1:1 phenol/N-methylacetamide box for domain decomposition.
- Treat the existing 1/2/4-H100 records as archived methodology. They are not
  current evidence until the complete campaign is rerun at the playbook's exact
  pins.
- Fine-tuning routing is superseded by the pinned AIMNet-glimpse /
  standalone-LJ split recorded below.

Removed from the main route:
- Ar/LJ and NaCl teaching toys;
- giant validation cells and legacy wrapper implementations; and
- SevenNet/Cu(111), except as a linked external case study.

Fine-tuning evidence boundary:
- No legacy Toolkit fine-tuning notebook exists.
- No current-pin end-to-end fine-tuning/domain campaign exists.
- The preferred continuous experiment would train on the 30 ordered `AB/A/B`
  interaction points, but those points have no force labels and transfer to the
  dense periodic mixture is unvalidated.
- Until current-pin evidence exists, keep the core and domain route on
  pretrained AIMNet2/NCI. The detailed optional-extension and standalone
  training split is recorded below.
- Do not claim a scientifically improved domain-decomposition model.

Playbook budget guard:
- The curation changes examples and transitions only.
- The core remains 19 visible actions across 80 planned minutes plus a
  10-minute recovery buffer.
- Zarr remains a two-action glimpse at minutes 20–25.
- Profiling remains reference-only with zero actions.
- Every substantive section retains its `Go deeper` target, and unfinished
  Parts 06–09 remain labeled in progress.
- Standalone lesson action budgets must not be imported into the playbook.

Changed:
- updated the existing
  `/home/nfedik/.cursor/projects/home-nfedik-projects-tutorials-v3/canvases/alchemi-curriculum-plan.canvas.tsx`;
- updated this integration plan only; no notebook source was changed.

Validation:
- Confirmed the three-class, 30-point, 90-graph `AB/A/B` structure in
  `research-toolkit-foundations/alchemi-toolkit-foundations.ipynb`.
- Confirmed the periodic 1:1 phenol/N-methylacetamide box and archived H100
  records under `part-1-scalable-atomistic-workflows/`.
- Confirmed the NCI Atlas revision, checksums, attribution, papers, and
  CC BY 4.0 terms in `data/nci_atlas/README.md`.
- Canvas TypeScript check: no errors.

Next:
- Keep the approval gate closed. Do not implement or assign notebook work until
  the user explicitly approves the generated playbook plan.

## 2026-08-12 — pin the AIMNet glimpse / standalone LJ split

Status: detailed plan recorded; implementation approval pending

Superseding decision:
- This entry replaces the earlier vague water/MACE fallback wording.
- The optional playbook extension and the complete standalone Part 08 have
  different models, evidence claims, and checkpoint boundaries.

Optional playbook extension — live AIMNet2 glimpse only:
- Keep it outside the 90-minute core and do not change the 19-action budget.
- Load through
  `AIMNet2Wrapper.from_checkpoint(..., compile_model=False)`, then construct
  `FineTuningStrategy(models=model, ...)`.
- Train only the 129 parameters matching
  `main.model.outputs.energy_mlp.mlp.4.*`.
- Run four full-batch updates with `num_epochs=4` and a one-batch loader.
- Use force loss only.
- Define the residual target as
  `F_base,target = F_DFT - F_Coulomb - F_D3`.
- Never fit bare AIMNet to full DFT forces.
- Assert that predicted charges do not drift.
- Do not pass the raw AIMNet `.pt` to
  `FineTuningStrategy.from_pretrained_checkpoint`.

Pre-implementation data gate:
- Extend `scripts/prepare_nci_subset.py` to retain gradients from
  `gradient/wB97M-D3BJ_def2-TZVPPD`; the current CSV writer retains positions
  and energies but excludes gradients.
- Split 36 labeled graphs as 27/6/3 train/validation/test.
- Do not implement this builder change until the playbook approval gate opens.

Hard-pin AIMNet checkpoint boundary:
- `AIMNet2Wrapper` has no `checkpoint_spec()` at the current pin.
- `AIMNet2Wrapper.to_spec_dict()` raises `NameError: Mapping`.
- Therefore the optional AIMNet glimpse has no `CheckpointHook`, no resume
  claim, no monkey-patch, no pickle, and no production-model promotion.
- Domain decomposition continues to load the original pretrained AIMNet2
  checkpoint. Live fine-tuned weights are not a promoted production model.

Rejected alternatives:
- Reject `DemoModel` as a scientific Lennard-Jones fit because it is not
  SE(3)-invariant.
- Keep the official 9M-parameter MACE example as reference only.
- MACE is not present in the current lock.

Complete standalone Part 08 — restartable LJ training:
- Status remains **in progress**.
- Generate labeled argon data.
- Implement `TrainableLJWrapper` with trainable `log_epsilon` and `log_sigma`.
- Implement explicit `checkpoint_spec()`.
- Demonstrate a true strategy checkpoint and resume, including model,
  optimizer, and counter continuity.
- Recover fitted epsilon/sigma and copy them into the built-in
  `LennardJonesModelWrapper` for `DomainParallel`.
- Preserve the generated-playbook link label:
  `Part 08 · Training/fine-tuning (in progress)`.

Unchanged playbook guards:
- Core: 19 visible actions over 80 planned minutes plus a 10-minute recovery
  buffer.
- Zarr: two-action glimpse at minutes 20–25.
- Profiling: reference-only, zero actions.
- Navigation: one `Go deeper` link after every substantive section; unfinished
  Parts remain labeled in progress.
- Approval: closed; no notebook or builder implementation authorized.

Evidence checked:
- Official Toolkit 0.2.0 `FineTuningStrategy` documentation confirms that
  `trainable_patterns` is a fully qualified allow-list and that models are
  supplied through `models=`.
- Official Toolkit checkpoint documentation requires a reconstructible model
  spec for restartable strategy checkpoints.
- `scripts/prepare_nci_subset.py` currently reads the gradient source tree but
  writes positions and total energies without gradient columns.

Changed:
- updated the existing
  `/home/nfedik/.cursor/projects/home-nfedik-projects-tutorials-v3/canvases/alchemi-curriculum-plan.canvas.tsx`;
- updated this integration plan only; no notebook, data builder, lockfile, or
  runtime source was changed.

Validation:
- Canvas TypeScript check: no errors.
- Edited-file diagnostics: no errors.

Next:
- Keep the approval gate closed until the user explicitly authorizes
  implementation.

## 2026-08-12 — harvest official Toolkit 0.2 examples for the playbook

Status: planning and existing-canvas update complete; implementation approval
pending

Scope:
- Read the complete
  [examples hub](https://nvidia.github.io/nvalchemi-toolkit/examples/).
- Read every Basic example:
  - [AtomicData and Batch](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/01_data_structures.html);
  - [FIRE Geometry Optimization with Lennard-Jones Argon](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/02_geometry_optimization.html);
  - [ASE Integration](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/03_ase_integration.html);
  - [NVE Molecular Dynamics](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/04_nve_energy_conservation.html); and
  - [NVT Langevin](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/05_nvt_langevin.html).
- Read every Intermediate example:
  - [Multi-Stage FusedStage](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/01_multistage_pipeline.html);
  - [Zarr Trajectory I/O](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/02_trajectory_zarr_io.html);
  - [NPT Barostat Validation](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/03_npt_barostat_validation.html);
  - [Inflight Batching](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/04_inflight_batching.html);
  - [Defensive MD](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/05_safety_and_monitoring.html);
  - [DDPHook with a Dummy MLP](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/06_ddp_mlp_training.html); and
  - [Rich Training Reporting](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/07_rich_training_reporting.html).
- Read matching official user guides for
  [data](https://nvidia.github.io/nvalchemi-toolkit/userguide/data.html),
  [data pipes](https://nvidia.github.io/nvalchemi-toolkit/userguide/datapipes.html),
  [models and neighbors](https://nvidia.github.io/nvalchemi-toolkit/userguide/models.html),
  [dynamics](https://nvidia.github.io/nvalchemi-toolkit/userguide/dynamics.html),
  [integrators](https://nvidia.github.io/nvalchemi-toolkit/userguide/dynamics_simulations.html),
  [hooks](https://nvidia.github.io/nvalchemi-toolkit/userguide/hooks.html),
  [sinks](https://nvidia.github.io/nvalchemi-toolkit/userguide/dynamics_sinks.html),
  [training](https://nvidia.github.io/nvalchemi-toolkit/userguide/training.html),
  [fine-tuning](https://nvidia.github.io/nvalchemi-toolkit/userguide/finetuning.html),
  [reporting](https://nvidia.github.io/nvalchemi-toolkit/userguide/reporting.html), and
  [distributed training](https://nvidia.github.io/nvalchemi-toolkit/userguide/distributed_training.html).
- Every gallery page exposes Python and notebook downloads. Matching Python
  sources were located in the official
  [Basic source tree](https://github.com/NVIDIA/nvalchemi-toolkit/tree/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18/examples/basic)
  and
  [Intermediate source tree](https://github.com/NVIDIA/nvalchemi-toolkit/tree/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18/examples/intermediate)
  at the frozen Toolkit commit.

Destination verdict:
- Basic 01 → playbook core and Part 01. Reuse construction, boundaries,
  selection, and recovery; omit the full mutation catalogue and buffer calls.
- Basic 02 → playbook FIRE2 sequence and Part 05. Reuse model hooks,
  convergence, bounded run, and fmax summary; substitute approved `FIRE2` for
  the example's `FIRE`.
- Basic 03 → playbook ASE conversion only; Parts 01/04 and future Part 07 own
  conversion, freezing, and fused stages. Do not interpret DemoModel output.
- Basic 04 → Part 07 or linked staged-dynamics material; no core NVE action.
- Basic 05 → Part 07; no core NVT action.
- Intermediate 01 → Part 07; link or syntax preview only in the playbook.
- Intermediate 02 → the two-action playbook Zarr glimpse and canonical Part 02.
- Intermediate 03 → Part 07 or NPT appendix; omit from the playbook.
- Intermediate 04 → linked Part 07 only until identity and integer-field
  limitations have a complete public path.
- Intermediate 05 → playbook built-in hook stack and Parts 04/07; timing stays
  out of the core.
- Intermediate 06 → optional synthetic fine-tuning glimpse plus Part 08.
  Reuse the dummy dataset/model and strategy shape; omit DDP from the playbook.
- Intermediate 07 → Part 08 reporting with a real strategy host. Reuse only
  reporter construction/registration in the optional extension.

Revised 90-minute playbook:
- Preserved 80 minutes of planned learner work, a 10-minute recovery buffer,
  and 19 learner-visible core actions.
- Preserved the two-action Zarr glimpse and zero profiling actions.
- Replaced the playbook's custom `EnergyHistoryHook` exercise with the cleaner
  official built-in stack: model-provided neighbor hooks,
  `NaNDetectorHook`, and bounded logging/snapshot behavior. The complete custom
  hook stays in standalone Part 04.
- Preserved the current NCI → supplied AIMNet2 → FIRE2 path for inference and
  relaxation. Official LJ/Ar remains an allowed fallback and the canonical
  system for the future dynamics/training standalones.
- Preserved one `Go deeper` transition after every substantive section and
  honest `in progress` labels for unfinished numbered notebooks.

Synthetic fine-tuning decision:
- The Basic/Intermediate gallery does not contain an LJ training example or an
  end-to-end fine-tuning example. It contains from-scratch synthetic DDP
  training with `TrainingStrategy` and a separate manual reporting demo.
- The optional playbook extension now combines only documented pieces:
  `DummyEnergyDataset` and `SimpleEnergyMLP` from Intermediate 06, prepared
  synthetic weights, and `FineTuningStrategy`, `trainable_patterns`,
  `OptimizerConfig`, `EnergyMSELoss`, `default_training_fn`, and `run` from the
  fine-tuning guide.
- Learner-visible work is two to four updates and one tiny loss plot. Dataset,
  model, and prepared-weight setup stays in a cited helper so visible cells
  remain N01-sized.
- The extension makes no SE(3)-invariance, chemistry, model-quality,
  checkpoint/resume, DDP, or production handoff claim.
- Standalone Part 08 remains the honest restartable generated-Ar/LJ lesson
  with an explicit model reconstruction spec. AIMNet2 remains the supplied
  inference/FIRE2 and domain model.
- The earlier 129-parameter AIMNet residual-force plan, NCI gradient packaging,
  and AIMNet checkpoint defect are no longer playbook blockers.

Pinned-runtime findings:
- A frozen-runtime import audit passed for every public class/function used by
  the 12 examples, including `FIRE`, `FIRE2`, NVE/NVT/NPT, Zarr, FusedStage,
  inflight, safety, DDP, reporting, training, and fine-tuning APIs.
- A one-process CPU adaptation of the official dummy MLP through
  `FineTuningStrategy` completed two optimizer steps at Toolkit 0.2.0.
- The session had no CUDA driver, so this pass did not claim GPU execution of
  the gallery examples.
- Basic 01's custom `AtomicData.add_node_property("custom_node_feat", ...)`
  creates an accessible attribute at this pin but does not register it in
  `node_properties` or `model_dump`. Do not copy that exact custom-node-field
  claim into learner material without a supported fix.
- The known fixed-capacity `Batch.put/defrag` path can skip integer fields, so
  it remains outside mixed-dtype core teaching.
- Basic 05 executes with `friction=0.5` but labels its optional plot as
  `gamma=0.05`; correct the title before reuse.
- Intermediate 03 hardcodes `cuda:0`, takes about 18 seconds in the gallery,
  and emits `torch.compile` graph-break warnings.
- Intermediate 04 uses private `HostMemory._data_list` and sampler `_bins` to
  work around `system_id` ownership loss. It is not a public core pattern.
- Intermediate 06 performs its full run only under `torchrun`; the docs build
  skips training.
- Intermediate 07 manually fabricates `TrainContext` around
  `SimpleNamespace`. That conflicts with the course rule against fake hook
  hosts; only real-strategy registration is reusable.

Current Parts 01–05 comparison:
- Part 01 has stronger real-ASE chemistry and a useful packed-ownership visual,
  but the 2,048-system performance opening and AIMNet inference detour make it
  less focused than the official data sequence. Its structure viewer is still
  pending.
- Part 02 is stronger than the official Zarr example on stable IDs, Reader
  ownership, validation, and resident/on-demand parity. Keep it canonical and
  use only the official read tail in the playbook.
- Part 03 is too broad at 109 cells and still contains forbidden
  `object.__setattr__` mutation in native/adapter parity setup. Excerpt only the
  supplied-wrapper path; split and repair custom composition before reuse.
- Part 04 is stronger on real host lifecycle and context. Keep its full custom
  hook in the deep lesson, not the playbook.
- Part 05 is stronger than Basic 02 on FIRE2, direct detector versus registered
  status migration, coherent final refresh, selection, and recovery.

Visual dressing selected:
- structures: MatterViz for three real inputs;
- batching: one packed-ownership strip;
- Zarr: one compact store → reader → Dataset → DataLoader → Batch flow;
- supplied model: one contract table and restrained component-energy figure;
- hooks: one stage-order strip and bounded NaN failure;
- FIRE2: per-system fmax traces and one before/after structure;
- recovery: one status summary and recovered-structure thumbnail; and
- optional fine-tuning: one two-to-four-point synthetic loss plot.

Changed:
- updated the existing
  `/home/nfedik/.cursor/projects/home-nfedik-projects-tutorials-v3/canvases/alchemi-curriculum-plan.canvas.tsx`
  in place with the 12-row harvest, provenance, revised beat sheet, shortcut
  boundary, current-lesson comparison, and visual plan;
- updated this integration worklog only;
- did not edit notebooks, create a competing canvas, commit, or launch lesson
  agents.

Validation:
- Canvas TypeScript check: no errors after both edits.
- Edited canvas diagnostics: no errors.
- Approval gate remains closed.

Next:
- Review the existing canvas's **Official examples harvest** and
  **90-minute playbook** views.
- Do not implement the generated playbook or Parts 06–09 until explicit
  approval.

## 2026-08-12 — CORE playbook and Part 01 approval plan

Status:
- CORE canvas: `TECHNICALLY VALIDATED DRAFT — HUMAN REVIEW REQUIRED`.
- Part 01 augmentation: `PLAN ONLY — HUMAN APPROVAL REQUIRED`.
- This entry supersedes the earlier 19-action playbook boundary and the
  earlier decision to leave NVE and synthetic fine-tuning outside the
  80-minute content window.
- The generated core notebook, manifest, lockfile, and transition cells remain
  unimplemented.

CORE operating plan:
- Rebuilt the existing canvas in place as the dominant 90-minute CORE
  artifact: 80 content minutes, 10 recovery minutes, and 23
  learner-visible actions.
- Action split: 16 `TAUGHT`, 7 `GLIMPSED`, and zero actions in the buffer.
- Minute sequence:
  - 00–04 orientation (1);
  - 04–12 AtomicData (2);
  - 12–20 Batch (3);
  - 20–25 prepared Zarr / Dataset / DataLoader glimpse (2);
  - 25–34 supplied AIMNet2 contract, neighbors, and inference (3);
  - 34–42 built-in hooks plus one tiny custom-hook contract (2);
  - 42–55 bounded FIRE2 relaxation and convergence evidence (3);
  - 55–60 five-step LJ-Ar NVE glimpse (1);
  - 60–65 supplied custom-wrapper/composition glimpse (1);
  - 65–71 official-style dummy-MLP fine-tuning glimpse (2);
  - 71–76 world-size-one domain control plus checked multi-rank evidence (1);
  - 76–80 recap/deep-link transfer (2); and
  - 80–90 recovery/discussion (0).
- Profiling, inflight batching, NPT, compilation, long MD, DDP, checkpointing,
  and notebook-launched distributed work remain zero-action pointers or
  explicit exclusions.

Scientific lanes:
- Real chemistry:
  NCI/ASE molecules → AtomicData/Batch → prepared Zarr → supplied AIMNet2 →
  hooks → FIRE2 → periodic phenol/N-methylacetamide domain endpoint.
- Synthetic API sandbox:
  27-atom LJ Ar for five-step NVE and composition mechanics; generated
  four-atom tensors with `energy = sum(positions²)` for two–four
  dummy-MLP updates.
- The sandbox ends before domain decomposition. The dummy MLP and LJ model
  never become the domain model; the domain beat returns to the original
  supplied AIMNet2 checkpoint.
- Partial AIMNet2 output is not presented as a complete scientific MD
  potential. The only live MD glimpse uses the complete LJ toy potential.

Generation and review architecture:
- Deep lessons remain canonical. A future
  `core-playbook.manifest.yaml` selects stable cell IDs and separately owned
  transition cells; `core-playbook.lock.json` records pins, source hashes,
  data/model hashes, and cached-evidence manifests.
- Generation must fail on source drift, action-count drift, API-token drift,
  output-policy drift, or visual-contract drift. It must not silently merge
  changed source cells.
- Review state is per cell and generated build:
  `Draft → technically validated → cell-reviewed → rendered-reviewed`.
- Technical validation is not human approval. Every generated and canonical
  notebook cell still requires human review.

Part 01 proposed alignment — not implemented:
- Preserve the real NCI molecules, course map, structure-input explanation,
  packed-boundary sequence, ownership visual, recovery actions, full
  collection, and transfer exercise.
- Replace the learner-facing 2,048-molecule benchmark and AIMNet2 inference
  detours only after human approval; model teaching belongs in the supplied-model
  deep dive.
- Proposed additions from official Basic 01 are selective:
  `model_dump` / `AtomicData.model_validate`, clone/equality/chemical hash,
  dictionary access, `exclude_keys`, mapping behavior, registered
  node/system `Batch.add_key`, Batch serialization, round trip, and
  cutoff-based neighbor preparation.
- Keep `append` / `append_data`, device, and memory-layout helpers as bounded
  references. Keep fixed-capacity `put` / `defrag` out of learner execution
  because the pinned path can skip integer fields.
- Proposed excerpt sources are AtomicData conversion, Batch construction and
  boundaries, recovery, and neighbor preparation. Stable source IDs and hashes
  enter the manifest only after cell-level review.
- Baseline planning inventory: 80 cells / 40 code cells. Proposed target:
  approximately 83 cells / 43 code cells, with an estimated 65–75 learner
  minutes after removing the model/performance detours.

Planning-artifact validation:
- Canvas TypeScript check: no errors.
- Canvas CLI rendering command was unavailable in this environment
  (`canvas: command not found`); the managed canvas TypeScript check remains
  the available renderer diagnostic.
- No notebook execution or tests are part of the approval-only plan.
- After approval, validation must cover schema/source cleanliness, stable IDs,
  public-API contracts, pinned behavior, focused tests, fresh execution to
  `/tmp`, and cell-by-cell rendered human review.

Rollback disclosure:
- One agent-added notebook cell (`5a8cab56`) was removed during the requested
  rollback.
- Further notebook rollback was blocked by the guarded notebook editor, so the
  remaining implementation delta was left untouched rather than bypassing the
  guard. The exact notebook and test paths are reported in the handoff.
- `worklog/01-atomicdata-batch.md` and Part 01 helpers were not changed in this
  planning correction.

Open human decisions:
- Approve, reject, or reorder the Part 01 delta rows in the canvas before any
  notebook implementation or validation.
- Approve and pin `pymatviz` / `anywidget` before replacing the MatterViz
  review marker with an executable widget.
- Approve the 23-action timing and which single action is cut if pilot pacing
  exceeds 80 minutes.
- Approve the future generator/manifest/lock design before implementing the
  core notebook.
- Approve a current-pin domain cache generator and evidence manifest before
  showing multi-GPU timing or scaling.

## 2026-08-13 — Core learner draft editorial reset

Status: learner-facing rewrite in progress; this is not Revision Pass 1 or
Revision Pass 2

The first generated Core draft failed editorial review. Internal planning
language leaked into the notebook through minute ranges, action IDs,
`TAUGHT` / `GLIMPSED` labels, lane names, action-budget notes, provenance
phrasing, and compliance-style explanations. The result read like a
transcription of the curriculum plan instead of an instructor guiding a
scientist through working APIs.

The replacement draft will:
- use the title **ALCHEMI Core Playbook** and the path
  `notebooks/00-core-playbook/alchemi-core-playbook.ipynb`;
- present the duration once as “Time to complete: about 90 minutes”;
- keep planning IDs, depth labels, timing metadata, provenance, and generation
  details outside learner-visible cells;
- rebuild section headings around scientific and API tasks;
- expand each section with construction, inspection, interpretation, a useful
  callout, and a natural deep-dive link;
- keep the separate Lennard-Jones dynamics and synthetic training examples
  explicit without curriculum jargon; and
- run both required complete revision passes only after the replacement draft
  executes and renders.

General rule: curriculum metadata may drive generation and tests, but it must
not appear in learner prose. Technical limitations belong next to the API or
scientific decision they affect, written in ordinary language.

## 2026-08-13 — Core Revision Pass 1: plan, API, and pedagogy

Status: complete; Revision Pass 2 still required

Review performed:
- reviewed all 95 generated cells against the approved Core sequence,
  `TUTORIAL_GUIDE.md`, `TOOLKIT_API_REFERENCE.md`, current Part 01, pinned
  Toolkit 0.2.0 behavior, and the relevant official Toolkit examples;
- used independent API/docs and pedagogy/scientific reviewers, then verified
  each finding against the current source rather than applying it mechanically;
- checked section depth, prerequisite order, public API visibility, result
  interpretation, scientific scope, and deterministic generation.

Applied findings:
- regenerated the stale learner artifact and provenance lock after moving the
  wrapper implementation into the visible notebook;
- corrected the `Batch.add_key` inspection so the registered system property
  reports `True`;
- recomputed FIRE2 final per-system maximum forces from the returned
  coordinates, separated those values from in-run status, and clarified the two
  convergence-hook responsibilities;
- documented the exact AIMNet2 ensemble member, supported elements,
  closed-shell molecular scope, source license, and the absence of an added
  long-range electrostatics or dispersion component;
- replaced adapter claims in the toy wrapper lesson with the actual
  `BaseModelMixin` / `ModelConfig` input-output contract and stated the
  quadratic coefficient units;
- exposed LJ epsilon, sigma, cutoff, argon mass, initialization temperature,
  velocity units, and the five-row NVE energy trace;
- evaluated the synthetic model on one fixed probe before and after training,
  recorded a parameter delta, relabeled the source loss as the last minibatch
  loss, and explained why the four minibatch losses need not be monotonic;
- renamed the domain section as a one-process preview, added the public
  `gather(...)` boundary, inspected the gathered result, and removed the false
  claim that ownership changed across ranks.

Findings not applied:
- source notebooks remain deliberately output-free; reviewed execution evidence
  lives under `/tmp` and generated source identity remains deterministic;
- the notebook keeps the user-approved `Why this matters`, `What to notice`,
  and `Scientific limit` callouts instead of reducing the design to two callout
  names;
- the live domain example intentionally keeps the verified world-size-one
  fallback instead of pretending to launch a multi-rank mesh. Multi-rank setup
  belongs in the distributed deep dive;
- missing Parts 06–09 remain explicit `in progress` placeholders by design.

Verification:
- generation check: passed;
- focused contract suite: `11 passed, 2 deselected`; the two deferred tests
  require the post-Pass-2 deep-dive handoff contract;
- fresh-kernel execution:
  `/tmp/core-pass1-final-executed.ipynb`, 95 cells / 50 code cells, passed with
  no cell errors;
- key evidence: `Batch.add_key=True`; FIRE2 stopped at the 16-update bound and
  all three recomputed force maxima remained above `0.05 eV/Å`; NVE produced
  five trace rows; the fixed synthetic probe MSE changed from `37.8617` to
  `35.6470`; domain partition/run/gather returned all 25 atoms at world size 1.
- the CPU validation environment emitted third-party CUDA/NVML initialization,
  AIMNet2 scalar-conversion, PyTorch set-sequence, and single-process
  distributed fallback warnings. These were not suppressed.

## 2026-08-13 — Core Revision Pass 2: learner and rendered review

Status: complete; technically validated and rendered-reviewed draft, human cell
review still required

Review performed:
- read the executed notebook from first cell to last as a first-time learner;
- applied the `no-ai-slop` rules to all 45 learner-facing Markdown cells;
- used independent learner/editorial and adversarial scientific reviewers;
- inspected the final HTML structure and every extracted plot;
- checked title, hidden setup, tables, outputs, warning affordances, image alt
  text, width, local links, placeholders, and source/executed identity.

Applied findings:
- displayed populated `AtomicData` fields and clarified that molecular drawing
  lines are visual guides rather than Toolkit neighbor edges;
- changed the Zarr language from a general round trip to a structural reload,
  stated what `Dataset` validates, hid the temporary directory, and compared
  selected retained tensors;
- narrowed AIMNet2 evidence to adapter execution, shapes, dtypes, and units for
  one ensemble member, without reference accuracy or ensemble-uncertainty
  claims;
- renamed the FIRE2 section as a bounded relaxation attempt, stopped calling
  the returned structures relaxed, and stated the observed non-convergence;
- relabeled hook and NVE tables with readable names and bracketed units;
- plotted different training minibatches as unconnected points, renamed the
  before/after check as a reused training minibatch, and stated that it is
  neither validation nor a generalization result;
- strengthened the toy-composition limit and changed the visual labels to
  `toy term`;
- warned that the one-rank domain cell is not a multi-rank launch recipe;
- corrected the deep-dive sequence to Part 06 GPU pipelines, Part 07 training,
  and Part 08 domain decomposition; composition now links to existing Part 03;
- added a deterministic HTML renderer that removes hidden setup inputs, embeds
  local assets, restores reviewed output alt text around an nbconvert export
  limitation, sets the page title, and constrains teaching width.

Findings not applied:
- the per-section Go deeper links remain because the Core contract requires a
  transfer point at the end of every substantive section;
- useful scientific-limit and interpretation callouts remain under the
  user-approved callout vocabulary;
- canonical Part 01 ownership output and the required FIRE2, NVE, training, and
  one-rank domain visuals remain because each supplies distinct evidence;
- the rendered artifact is a review artifact, not a standalone packaged course
  site. The renderer embeds visual assets; notebook navigation remains relative
  to the repository lesson structure.

Editorial and visual results:
- forbidden curriculum language: zero matches;
- minute ranges, repeated `Now` / `Next` / `In this section` openings, em
  dashes, banned AI-writing words, and empty stock phrases: zero matches;
- final count: 95 cells, 50 code cells, 45 Markdown cells;
- all eight generated plots plus the shared banner have useful alt text in the
  final HTML; the journey SVG is embedded with its ARIA label;
- extracted plot review found readable units, legends, and dimensions. The
  FIRE2 threshold remains visible near the force-axis floor and the
  before/after displacement is intentionally subtle; the final table prevents
  either image from implying convergence.

Final verification:
- full contract suite: `13 passed`;
- deterministic generation check: passed;
- fresh supported-runtime execution:
  `/tmp/alchemi-core-final-executed.ipynb`, 95 cells / 50 code cells, no cell
  errors, source identity equal to the generated notebook;
- execution retained three known third-party stderr cells: AIMNet2
  tensor-to-scalar conversion, PyTorch set/sequence behavior during FIRE2, and
  the expected single-process distributed fallback. The opening explains how
  to distinguish CPU fallback from a GPU setup failure;
- final render:
  `/tmp/alchemi-core-pass2-final/alchemi-core-playbook.html`; title correct,
  hidden setup source absent, 9/9 images embedded, 0 missing alt descriptions,
  journey object embedded, responsive width style present;
- browser screenshot automation was unavailable because the installed
  headless Chromium lacks `libnspr4.so`. This did not block review: the HTML DOM
  and all eight extracted plot images were inspected directly;
- the obsolete pre-reset directory is absent. One negative contract assertion
  guards against its return.

Deep-dive handoff:
- added `notebooks/00-core-playbook/DEEP_DIVE_CONTRACT.md`;
- placeholders are Parts 06–08 only and are marked `in progress`;
- no deep-dive agents were launched.

Final state:
`TECHNICALLY VALIDATED, RENDERED-REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED`.

## 2026-08-13 — Capability-map link status

- Kept all documentation and course links active in the capability SVG.
- Styled maintained documentation links in NVIDIA green.
- Styled Part 01–08 links in muted gray and labelled the group and fallback
  links as in progress.
- Updated the no-hover fallback in `transitions.ipynb` through the live notebook
  bridge.
- Added mechanical checks for non-empty SVG `href` values, five documentation
  links, eight deep-dive links, and the muted in-progress treatment.
- Checks: Ruff passed; the focused capability-map test passed; the saved
  fallback contains eight clickable in-progress notebook links.
- User review required: hover or focus each capability card and open one green
  documentation link and one gray course link. The live kernel was busy during
  the display-cell refresh, so the source and generated SVG are saved while the
  cell output remains to be rerun.

## 2026-08-13 — Compact interactive capability map

- Moved the single Toolkit capability map into the opening, directly after the
  ALCHEMI context.
- Replaced the three-column tree with two levels: Toolkit with Toolkit-Ops,
  followed by data, models, simulation, and training/scale.
- Each capability opens a nearby hover or keyboard-focus panel with its meaning,
  applications, maintained docs, and focused course links.
- Reused the restrained line icons from Part 01 and numbered the map with the
  same 01–08 deep-dive sequence used by the course.
- Removed the repeated horizontal journey image from each Core section. Section
  dividers remain, and major headings carry the matching course number.
- Preserved the editable Draw.io source and the static link list for touch,
  exported, and no-hover notebook views.

User review required:
- Open `notebooks/00-core-playbook/alchemi-core-playbook.ipynb` and hover each
  of the four cards in the opening map.
- Confirm every panel appears immediately above its card, text stays inside the
  panel, and Docs / Part links open the intended page.
- Confirm the numbered section headings are easier to scan without the repeated
  horizontal maps.

## 2026-08-13 — Core Zarr naming, device path, and Reader handoff

Status: implementation draft; human review and the combined Core/MatterViz
execution/render pass are still required

Learner-facing corrections:
- pre-write CPU objects are now named `records_to_save`; entries returned by
  `reader.read_many(...)` are `loaded_records`;
- the section states that Zarr saves supported atom- and system-level fields,
  including the explicitly added `record_id`, while `Batch` graph boundaries
  are reconstructed during collation in another notebook or training job;
- the compact, editable `zarr-data-flow.drawio` and content-addressed SVG now
  show `Zarr store (disk or CPU storage) → Reader (CPU tensors) → Dataset
  (target device) → DataLoader (batch + prefetch) → Batch (model device)`;
- the visual does not imply GPU-backed Zarr, `DataLoader(device=...)`, or that
  pinned CPU memory is VRAM. The Core code keeps `Dataset(reader,
  device="cpu", num_workers=2)` explicit and gives `DataLoader` only its
  supported batching/prefetch keywords;
- one concise **Bring another data source** callout records the pinned custom
  `Reader` contract: logical length, `field_levels`, and ordered raw CPU tensor
  reads through `read_many(...)`, backed by `_load_sample(...)` or
  `_load_many_samples(...)`. It links to Part 02's real extxyz implementation.

Pinned Toolkit 0.2.0 references:
- `AtomicDataZarrWriter`:
  https://nvidia.github.io/nvalchemi-toolkit/modules/generated/nvalchemi.data.AtomicDataZarrWriter.html
- `AtomicDataZarrReader`:
  https://nvidia.github.io/nvalchemi-toolkit/modules/generated/nvalchemi.data.AtomicDataZarrReader.html
- data loading pipeline and custom `Reader` contract:
  https://nvidia.github.io/nvalchemi-toolkit/userguide/datapipes.html
- official trajectory-to-Zarr example:
  https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/02_trajectory_zarr_io.html

Pinned-device verdict:
- filesystem/object stores persist outside accelerator memory; `MemoryStore`
  and dict-backed stores use CPU RAM;
- writer serialization crosses through detached CPU/NumPy arrays, so autograd
  history is not stored;
- the reader returns CPU tensors and `pin_memory=True` means page-locked CPU
  RAM;
- `Dataset(device=...)` owns emitted-device placement; `DataLoader` has no
  device argument and uses the dataset target while coordinating batching,
  prefetch, and optional stream transfers.

## 2026-08-13 — Core device wording and capability-map output

Status: complete; user visual review required

Changes:
- routine setup now reports `Compute device: CUDA GPU` or `Compute device: CPU`;
  exact hardware names remain part of performance results;
- the opening scope states that CUDA is the course target and CPU supports a
  quick API walkthrough;
- cell 8 now displays the capability map as an inline SVG notebook output with
  accessible text. This replaces the local Markdown object that VS Code did not
  render reliably;
- the HTML renderer now handles reviewed PNG and SVG output descriptions;
- the notebook builder can write a clean execution copy before reviewed outputs
  are refreshed.

Checks:
- full fresh-kernel execution with CUDA hidden completed with no cell errors;
  Warp printed one no-CUDA initialization line and the CPU workflow continued;
- fresh CUDA execution completed and refreshed 18 selected saved outputs;
- Core tests: `26 passed`;
- Ruff, deterministic generation, the authoring-skill validator, and the
  notebook design check passed;
- final HTML render succeeded at
  `/tmp/alchemi-core-implementation/alchemi-core-playbook-final.html`;
- the final notebook cell 8 contains one `image/svg+xml` output and its
  accessibility description.

User review required:
- open cell 8 in
  `notebooks/00-core-playbook/alchemi-core-playbook.ipynb`;
- confirm the map is visible, its hover/focus details respond, and its notebook
  links open as expected in the current VS Code/Jupyter interface.

## 2026-08-13 — Compact molecular viewers

- cell 6 now ends with a 260 px bonded MatterViz view of ethyne. The setup
  source remains folded and the compact viewer hides its controls;
- the later molecular viewer now shows phenol at 360 px, so it extends the
  example to a larger molecule instead of repeating ethyne;
- the generated notebook retains the cell 6 widget state and output;
- runtime pin check, 26 Core tests, Ruff, deterministic generation, and HTML
  rendering passed;
- a separate fresh-kernel process could not start under the current filesystem
  sandbox because Jupyter could not open its local communication socket. The
  changed cell executed successfully through the live CUDA notebook kernel.

User review required:
- confirm the ethyne molecule and bonds remain legible at the compact height in
  cell 6;
- confirm the controls appear for the larger phenol viewer later in the lesson.

## 2026-08-13 — Official AtomicData field inspection

- replaced learner-facing `model_dump(...)` field checks with the documented
  `node_properties` and `system_properties` views;
- explained that `AtomicData.from_atoms(...)` maps `atoms.info["charge"]` to
  `AtomicData.charge` and removed the duplicate charge insertion for individual
  molecules and the later molecule list;
- kept `add_system_property(...)` for genuinely added workflow fields such as
  energy and status;
- changed the answer text from CUDA-specific wording to the selected device;
- live-kernel inspection confirmed `charge` is a populated system field with
  shape `(1, 1)` and value `0.0` for ethyne;
- Core tests: `26 passed`; Ruff and deterministic generation passed.

## 2026-08-13 — Field growth and anion exercise

- replaced the generic field-name exercise with a runnable ethynyl-anion task:
  copy ethyne, remove one hydrogen, set total charge to `-1`, and convert the
  result with `AtomicData.from_atoms(...)`;
- added a folded solution with the checked result `("C2H", 3, -1.0)`;
- the model section now builds a fresh three-molecule batch, prepares neighbors,
  calls AIMNet2, registers energy and forces through `Batch.add_key(...)`, and
  reinspects one recovered `AtomicData` object;
- the live result adds `neighbor_matrix`, `num_neighbors`, and `forces` to node
  fields and `energy` to system fields; the dense model path has no populated
  edge fields;
- documented the general before/after state-inspection principle in
  `TUTORIAL_GUIDE.md` and the installed ALCHEMI authoring skill;
- live CUDA checks passed for the anion construction and model-output
  registration; Core tests: `26 passed`; Ruff, deterministic generation,
  design checks, and HTML rendering passed.

User review required:
- edit the TODO cell to produce `C2H` and charge `-1`, then compare it with the
  folded solution;
- run the model attachment cell and confirm the field groups make the state
  transition clear without extra explanation.

## 2026-08-13 — Core pedagogy and interactive navigation design record

Status: complete; technically validated and rendered-reviewed draft, human cell
review still required

- lesson outcome: help a computational chemistry developer follow the common
  Toolkit objects through data, models, hooks, dynamics, training, and
  distributed execution, then choose a focused notebook for deeper work;
- proposed cell sequence: short ecosystem orientation, interactive capability
  map, one worked data path, small predict-run-inspect exercises with folded
  answers, model and workflow applications, then recap and next-notebook links;
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `Batch.from_data_list`,
  Zarr reader/writer, model configuration and calls, model neighbor hooks,
  `FIRE2`, `NVE`, training strategy, and `DomainParallel` boundaries;
- molecules and models: the existing ASE molecule collection with AIMNet2 for
  molecular examples, periodic Lennard-Jones argon for MD, and the existing
  small synthetic training systems;
- helper boundaries: helpers own presentation, interactive navigation,
  MatterViz setup, repeated plotting, fixtures, and cleanup; learner cells own
  the public Toolkit operations and small editable questions;
- expected runtime: about 25 seconds on the shared GPU runtime, with the
  learner-facing workshop paced to about 90 minutes;
- validation plan: source and generator checks, focused unit tests, frozen
  runtime check, fresh-kernel execution, HTML export, local-link and alt-text
  review, responsive browser capture, and one final human notebook pass.

Implemented:
- added a compact interactive Toolkit capability tree with keyboard-focusable
  leaves, subtle hover treatment, public API details, application labels, and
  links to the eight focused notebooks; kept an editable Draw.io source and a
  deterministic, content-addressed SVG;
- replaced the static molecule projection with the pinned MatterViz structure
  widget used by Part 01; the learner views the original ASE structure while
  the converted Toolkit tensors remain on CUDA;
- added five native folded answers around AtomicData fields, packed boundaries,
  graph recovery, model-output shapes, and hook frequency;
- simplified the opening outcome and API cards, made the course runtime
  explicitly CUDA-based, and kept public Toolkit calls in learner cells;
- added reviewed-output capture keyed by generated cell ID and source hash;
  the generated notebook now opens with 17 selected tables and figures while
  the remaining cells stay unexecuted and editable;
- extended `TUTORIAL_GUIDE.md` and the installed
  `alchemi-tutorial-authoring` skill with ADDIE, Diátaxis, Jupyter for
  Education, portable folded answers, exercise anatomy, and assessment-tool
  selection.

Checks and evidence:
- frozen runtime check: passed;
- final fresh-kernel execution: passed, with the generated notebook written to
  `/tmp/alchemi-core-implementation/executed-final.ipynb`;
- Core contract and deterministic-generation tests: `26 passed`;
- scoped Ruff checks: passed; skill validation and notebook mechanical check:
  passed with zero errors;
- exported HTML: 2.55 MB, five folded answers, one interactive object, 21
  images, and zero images without alternative text;
- headless Edge review at 1280 and 720 pixels confirmed the opening hierarchy,
  framework diagram, GPU identity, and responsive capability map. The first
  narrow render exposed a fixed minimum-width clip; the final map scales to the
  notebook column;
- Playwright is absent from the frozen environment, so Edge supplied the
  browser capture without changing dependencies.

Blockers: none. Final release still needs one human pass through the live
notebook, including hover, keyboard focus, links, MatterViz controls, and the
folded-answer sequence.

## 2026-08-13 — Core framework topology and curriculum visual correction

Status: complete; technically validated and rendered-reviewed draft, human cell
review still required

Authoritative framework correction:
- verified the pinned Toolkit import path in `nvalchemi/neighbors.py`:
  Toolkit calls `nvalchemiops.torch.neighbors`; no Toolkit import enters the
  `nvalchemiops.jax` namespace;
- verified that `nvalchemiops.jax` is a separate optional binding namespace for
  Warp-backed primitives and that Toolkit-Ops also contains documented
  Torch-native operations or utilities;
- replaced the earlier Part 01-style merged framework input diagram. That
  visual placed PyTorch and JAX inputs before a shared operation and did not
  show that ALCHEMI Toolkit remains on the Torch binding branch;
- added editable `assets/framework-bindings.drawio` and regenerated
  `assets/framework-bindings.svg` at `920 × 246`. The highlighted path is
  `ALCHEMI Toolkit → Toolkit-Ops Torch bindings → selected operation →
  accelerated Warp kernels`; the JAX path is separate and marked not used
  here, and a muted Torch-native implementation branch records the verified
  alternative;
- revised the learner prose, caption, alt text, and contract tests. `JAX` and
  `Warp` first appear in the framework explanation, and `Toolkit Core` is
  forbidden.

Curriculum correction:
- reduced the Draw.io and SVG canvas from `1000` to `880` pixels wide (12%);
- retained `880 × 148` for the opening map and `880 × 92` for every section
  banner;
- increased top area boxes to `28` pixels and bottom step boxes to `38` pixels;
- retained `12 px` bold area and step labels, with label-aware step widths so
  `Wrapping your model` and `Domain decomposition` do not clip;
- replaced arrowheads and orthogonal elbows with nine straight, `1 px`,
  round-capped neutral connectors between adjacent steps;
- active steps use solid NVIDIA green (`#76B900`) with dark text. Their parent
  areas use a quieter green surface (`#213016`), green border, and light text.
  Other areas remain neutral;
- added SVG state metadata and contract checks for all ten active
  area-to-step mappings, dimensions, box heights, Draw.io geometry, and
  connector style;
- inspected the opening map and all ten variants at their `880 px` notebook /
  1280 px HTML presentation and at the `780 px` minimum used by 720 px HTML.
  Browser captures at 1280 and 720 pixels showed no clipping; the narrow
  notebook presentation scrolls horizontally rather than shrinking labels.

Provenance repair found during fresh execution:
- concurrent Part 01 edits preserved old cell IDs while changing the examples
  to water, methane, and ethanol. Three copied cells therefore introduced
  undefined or misleading Core variables;
- moved the Core-specific `AtomicData.from_atoms`, molecule-list conversion,
  and `Batch.get_data` calls into transition provenance and removed the stale
  Part 01 selectors;
- updated `DEEP_DIVE_CONTRACT.md`, manifest indices and hashes, generated
  notebook, and lock.

Final verification:
- generated notebook: 129 cells, 80 code and 49 Markdown;
- Core contract suite: `20 passed`;
- deterministic generation check: passed;
- fresh-kernel execution:
  `/tmp/alchemi-core-final/alchemi-core-playbook.executed.ipynb`, no cell
  errors, with source / cell ID / order identity equal to the generated
  notebook;
- final render:
  `/tmp/alchemi-core-final/alchemi-core-playbook.html`, 21 images and one
  object, all assets embedded and all image alt descriptions present;
- Edge captures:
  `/tmp/alchemi-core-final/alchemi-core-1280-final.png` and
  `/tmp/alchemi-core-final/alchemi-core-720-final.png`;
- forbidden learner-language search: zero matches; edited Python files have no
  IDE lint findings.

Final state:
`TECHNICALLY VALIDATED, RENDERED-REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED`.

## 2026-08-13 — Core human-feedback cohesion revision

Status: complete; technically validated and rendered-reviewed draft, human cell
review still required

Human feedback applied:
- removed learner-facing setup and process narration about dependency warnings,
  message suppression, validation mechanics, and the distributed fallback
  message; warnings remain in execution output without maintainer commentary;
- copied the current Part 01 `Where NVIDIA ALCHEMI fits` cell verbatim and
  added a parity test, then adapted the framework visual from
  `scripts/rebuild_part1_ir_notebook.py` (`framework-primer`) and
  `part-1-scalable-atomistic-workflows/aux/ui.py`
  (`process_diagram_html`);
- adapted Part 01 cell `3174653f` into a render-stable SVG that defines the
  molecule / `AtomicData` / `Batch` relationship before `batch_idx` and
  `batch_ptr` appear;
- replaced generic blockquotes with the Part 01 light and dark callout
  treatments for API notes, interpretation, motivation, scientific limits,
  and all 12 Go deeper links;
- generated an editable Draw.io curriculum source, a full curriculum SVG, and
  nine linked progress banners from one deterministic asset script;
- split the former generated learner cell 11 into `core-t12` and
  `core-t14` through `core-t19`: inspect one `AtomicData` object, add and
  inspect one system-level field, construct a second record, add and inspect
  its charge, and compare both records;
- split the dense hook, FIRE2 final-state, NVE, composition, fine-tuning, and
  domain sequences at conceptual boundaries without recombining earlier
  splits. The final FIRE2 evidence now progresses through `core-t60` to
  `core-t66`: prepare neighbors, evaluate, reduce forces per graph, inspect the
  table, plot history, recover one structure, and inspect snapshots;
- assigned deterministic IDs to every transition source cell and updated
  current Part 05 canonical indices after its concurrent notebook revision.

Toolkit-behavior comment audit:
- AtomicData / Batch comments identify system-, graph-, and node-level fields,
  and state what `batch_idx` preserves;
- model and dynamics comments distinguish explicit in-place neighbor
  preparation from model neighbor hooks inside a dynamics host;
- hook comments explain stage and frequency; FIRE2 comments separate position
  updates from model evaluation, hooks, and coherent final recomputation;
- wrapper comments identify the Toolkit-to-native tensor mapping and active
  output selection;
- fine-tuning comments identify the responsibilities of
  `FineTuningStrategy`;
- domain comments define `partition`, `run`, and `gather` and state why the
  one-rank grid does not partition;
- generic Python narration such as comments about imports, list creation,
  loops, printing, or line-by-line mechanics is absent.

Generated assets:
- `notebooks/00-core-playbook/assets/render_core_journey.py`;
- `notebooks/00-core-playbook/assets/core-curriculum.drawio`;
- `notebooks/00-core-playbook/assets/core-journey.svg`;
- `notebooks/00-core-playbook/assets/molecule-atomicdata-batch.svg`;
- `notebooks/00-core-playbook/assets/journey-banner-*.svg` (nine files).

Final verification:
- generated notebook: 129 cells, 80 code and 49 Markdown; the increased count
  comes from the requested conceptual splits and nine banners while retaining
  every later Core topic;
- contract, source-parity, comment, link, asset, deterministic-render, and
  generation suite: `16 passed`;
- deterministic generation check: passed;
- fresh-kernel execution:
  `/tmp/alchemi-core-human-review/alchemi-core-playbook.executed.ipynb`, no
  cell errors, source / cell ID / order identity equal across all 129 cells;
- execution used the supported CPU path. Warp reported unavailable CUDA driver
  entry points, and third-party code emitted the existing set/sequence warning;
  neither changed the calculations or required learner action;
- final render:
  `/tmp/alchemi-core-human-review/alchemi-core-playbook.html`, 2.63 MB,
  19 images and one object, all local assets embedded, all alt descriptions
  present, nine progress banners, 12 styled Go deeper callouts, correct title,
  responsive width, and no unrendered Mermaid source;
- forbidden learner-language search across Markdown and text output: zero
  matches; local links: 27 unique targets, zero missing; stale
  `workflow-playbook` labels: zero;
- generated-cell density audit found no remaining construct + mutate + run +
  inspect cells. Longer visible cells are bounded classes, configurations,
  result dictionaries, or the atomic `DomainParallel` context lifecycle;
- browser screenshot automation remains unavailable because the installed
  headless Chromium lacks `libnspr4.so`; HTML DOM, embedded SVG structure,
  callout hierarchy, alt text, outputs, and plot metadata were inspected
  directly.

Final state:
`TECHNICALLY VALIDATED, RENDERED-REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED`.

Latest visual-QA addendum:
- Windows Edge headless capture subsequently succeeded at 1280 and 720 pixels;
  the current artifacts and results are recorded in the framework topology and
  curriculum visual correction section above.

## 2026-08-13 — Core section dividers and Zarr expansion

Status: complete; technically validated and rendered-reviewed draft, human cell
review still required

Section-divider implementation:
- `build_core.py` now owns one semantic divider template:
  `<hr class="alchemi-section-divider" ...>`. The build inserts it before each
  journey banner, so the reading order is divider, compact journey, then major
  heading;
- the shared treatment is a one-pixel `#D6D9D4` top rule with
  `2.4rem 0 1rem` spacing. It uses no card, gradient, shadow, or decorative
  marker;
- all ten substantive sections contain exactly one divider. Contract tests
  reject missing, duplicate, or consecutive dividers.

Zarr teaching expansion:
- expanded the generated section from 8 cells / 5 code cells to 29 cells /
  19 code cells;
- added stable system-level `record_id` values for the three existing graphs,
  a CPU persistence batch, pre-write graph and boundary inspection, a genuine
  writer lifecycle, source-object release, reader reopen, public
  `read_many(...)` schema/ownership inspection, one-record comparison,
  `Dataset` validation and indexing, keyword-only `DataLoader` configuration,
  one model-ready `Batch`, tensor/identity round-trip checks, and safe temporary
  store cleanup;
- visible public APIs are `AtomicData.add_system_property(...)`,
  `Batch.from_data_list(...)`, `AtomicDataZarrWriter(...)`,
  `writer.write(...)`, `AtomicDataZarrReader(...)`, `reader.read_many(...)`,
  `reader.read(...)`, `reader.field_levels`, `Dataset(...)`, `dataset[...]`,
  `DataLoader(...)`, and `next(iter(loader))`;
- retained exact Part 02 source cells `5e4d2132`, `47272271`, `f4f62072`,
  `7a7e835d`, `c58c66d3`, and `66a3ca22`. The writer call was adapted into
  separate construction and write cells because Part 02 combines those actions;
- adapted Part 02 Mermaid cell `ba9aa971` into the deterministic,
  content-addressed `zarr-data-flow.svg`. The compact SVG shows
  `Batch → Zarr store → Reader → Dataset → DataLoader`;
- omitted Part 02's custom-reader, malformed-record, validation-failure,
  cache-comparison, and larger NCI sequences.

Measured execution and checks:
- fresh supported-runtime execution took 24.92 seconds wall time;
- generated notebook: 150 cells, 94 code and 56 Markdown;
- Zarr section output recovered IDs `[101, 102, 103]`, boundaries
  `[0, 4, 17, 37]`, and matching atomic numbers, positions, and temperatures;
- the temporary store was absent after section cleanup;
- contract and deterministic-generation suite: `23 passed`;
- source, cell ID, and cell order identity matched between generated and
  executed notebooks;
- responsive section renders:
  `/tmp/alchemi-core-final/alchemi-zarr-1280.png` and
  `/tmp/alchemi-core-final/alchemi-zarr-720.png`;
- the learner-facing duration remains “Time to complete: about 90 minutes.”

Final state:
`TECHNICALLY VALIDATED, RENDERED-REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED`.

## 2026-08-13 — Hover documentation links

- Changed the capability-map output from the SVG image MIME type to inline
  HTML. VS Code had treated the previous output as one image and exposed the
  SVG filename instead of activating its inner links.
- Kept five maintained documentation links inside the hover panels.
- Rendered Parts 01–08 as muted in-progress text with no `href` because those
  notebooks are not ready for public navigation.
- Updated the no-hover fallback to repeat the documentation links and the same
  muted notebook status.
- Updated the authoring guide and installed skill with the general rule: link
  ready documentation, and show unpublished lessons as plain status text.
- The live map cell now returns `text/html`; focused Ruff and map tests pass.
- The full deterministic build check remains pending while other authors are
  changing the transition notebook and its recorded source hash.

Follow-up:
- Replaced SVG-native hover links with native HTML overlays after rendered
  review found that VS Code exposed the SVG filename and retained focus.
- Mouse hover now closes on pointer exit. Keyboard access uses `:focus-visible`.
- Documentation links are ordinary HTML anchors. The generated SVG contains no
  anchors, and Parts 01–08 remain muted status text.
- Captured the corrected HTML output, regenerated the Core, and passed the
  deterministic build, Ruff, and two focused interaction tests.

## 2026-08-13 — Core Playbook build machinery removed

The Core Playbook is no longer a build artifact. `alchemi-core-playbook.ipynb`
is hand-authored and is the only source of truth, so the compiler, its manifest,
its lockfile, and its reviewed-output overlay were deleted along with the
transition notebook they assembled. Nothing regenerates the notebook now.

Deleted files:
- `notebooks/00-core-playbook/build_core.py`;
- `notebooks/00-core-playbook/render_core.py`;
- `notebooks/00-core-playbook/core-playbook.manifest.yaml`;
- `notebooks/00-core-playbook/core-playbook.lock.json`;
- `notebooks/00-core-playbook/core-playbook.outputs.json`;
- `notebooks/00-core-playbook/transitions.ipynb`.

Test changes:
- removed `test_provenance_manifest_and_lock_cover_every_cell`,
  `test_manifest_addresses_transition_cells_by_stable_id`, and Part 05's
  `test_public_fire2_run_keeps_core_provenance_cell_id`;
- replaced `test_reviewed_output_overlay_matches_generated_sources` with
  `test_saved_outputs_carry_no_errors_or_tracebacks`, which keeps the real
  intent that only clean reviewed outputs ship;
- renamed `test_notebook_is_generated_valid_and_parses` to
  `test_notebook_is_valid_and_parses` and dropped its generation-marker
  assertions; schema, cell-count, and syntax validation stay;
- dropped the manifest lookups inside the zarr round-trip and clean-source
  tests, and the pinned Core-excerpt hash in Part 03, keeping every surrounding
  teaching and hygiene assertion;
- suites after the change: Core `27 passed`, Part 03 `15 passed`, Part 05
  `39 passed, 1 skipped` (the skip is the CUDA-gated FIRE2 test).

Follow-up:
- `notebooks/00-core-playbook/DEEP_DIVE_CONTRACT.md` still describes the
  manifest and lock as authoritative for reusing Core excerpts. It needs a
  rewrite that is not tied to a generator.
- The notebook still carries `core_provenance` cell metadata and
  `metadata.core.generated`, `do_not_edit`, and `manifest` keys. Those are
  generator residue and are being stripped separately.
