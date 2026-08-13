# N05 worklog — BaseDynamics workflow construction

No entries yet.

## 2026-08-11 20:19 EDT — lesson design

Owner: N05
Status: in progress

Observed:
- The owned notebook directory was absent and has been created. The frozen runtime check passes with Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- At the pinned Toolkit revision, subclasses declare `__needs_keys__` and `__provides_keys__`; `BaseDynamics.step` runs `pre_update -> compute -> post_update` and returns graph indices selected by its `ConvergenceHook`.

Design:
- Lesson outcome: subclass `BaseDynamics` to build a small batched steepest-descent optimizer, declare its data requirements, run eight molecular graphs with force-based convergence, inspect per-graph behavior, and compare the construction interface and force histories with Toolkit `FIRE2`.
- Proposed cell sequence: title and outcome; workflow diagram; imports and editable settings; load and display eight selected molecules; build `AtomicData` records and one `Batch`; load/configure AIMNet2; define `BatchedSteepestDescent`; inspect required/provided keys; construct a `ConvergenceHook` and neighbor hooks; clone and prime equal starting batches; run steepest descent with Rich progress; run `FIRE2` with Rich progress; assemble convergence results; display per-graph and optimizer summaries; plot maximum force by model evaluation; inspect recovered final structures; advanced extension task; API recap.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `AtomicData.add_system_property`, `Batch.from_data_list`, `Batch.clone`, `AIMNet2Wrapper.from_checkpoint`, `set_config("active_outputs", ...)`, `make_neighbor_hooks`, `BaseDynamics`, `__needs_keys__`, `__provides_keys__`, `pre_update`, `post_update`, `compute`, `step`, `ConvergenceHook.from_fmax`, `FIRE2`, and `Batch.to_data_list`.
- Molecules and model: ethyne, acetonitrile, methanol, acetaldehyde, acetamide, pyridine, phenol, and 2,3-dimethylbutane from the pinned shared collection; `aimnet2-wb97m-d3_0` with the recorded SHA-256; requested outputs `{"energy", "forces"}`.
- Helper boundaries: notebook-local helpers verify the repository root, shared molecule/checkpoint checksums, exact eight-molecule identity, and construct display rows from optimizer histories. Toolkit data construction, model setup, optimizer subclass, hooks, optimization calls, DataFrames, and Matplotlib stay visible in cells.
- Expected runtime: loading and data construction under 10 seconds on CPU; model setup and first force calls about 5–15 seconds on the local RTX 4000 SFF Ada GPU; two short 30-update optimizer runs about 15–45 seconds total. CPU-only model execution may take several minutes. These are planning estimates; H100 timing is a separate later check.
- Validation plan: notebook JSON validity; IPython-aware static and complete-namespace checks; notebook-local helper and source-contract tests through `./scripts/v3-run`; fresh-kernel top-to-bottom execution through the live bridge; exact eight-graph identity; required/provided-key checks; finite energies, forces, and positions; per-graph convergence accounting; equal initial coordinates and equal step cap for the two optimizers; diagnostics scan; manual review of the dark-style NVIDIA-green plot and rendered notebook.

Next:
- Implement notebook-local helpers and tests, then author and execute `notebooks/05-base-dynamics/base-dynamics.ipynb` through the live notebook bridge.

## 2026-08-11 20:41 EDT — implementation and validation

Owner: N05
Status: ready for integration

Observed:
- The complete notebook has 37 cells: 15 Markdown cells and 22 code cells. Central Toolkit construction, stepping, convergence, table display, and plotting remain visible.
- A final fresh-kernel run completed all 22 code cells without an exception in 14.5 seconds on the local NVIDIA RTX 4000 SFF Ada GPU through the frozen v3 environment.
- After 30 updates, the custom `BaseDynamics` steepest-descent example met the `0.10 eV/Å` target for 2/8 graphs; `FIRE2` met it for 0/8. The notebook states that 30 updates end before FIRE2's default 60-step adaptation delay, so this result supports an interface and short-run behavior comparison. All 16 recovered final graphs had finite coordinates.

Changed:
- `notebooks/05-base-dynamics/base-dynamics.ipynb`
- `notebooks/05-base-dynamics/helpers/__init__.py`
- `notebooks/05-base-dynamics/helpers/lesson.py`
- `notebooks/05-base-dynamics/tests/conftest.py`
- `notebooks/05-base-dynamics/tests/test_lesson.py`
- `worklog/05-base-dynamics.md`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed; Toolkit and Toolkit-Ops pins match the handoff.
- `./scripts/v3-run pytest -q notebooks/05-base-dynamics/tests`: 3 passed; includes exact molecule identity and IPython-aware full-notebook parsing/source checks.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests`: passed.
- `./scripts/v3-run python -m json.tool notebooks/05-base-dynamics/base-dynamics.ipynb`: passed.
- `git diff --check -- notebooks/05-base-dynamics worklog/05-base-dynamics.md`: passed.
- Fresh-kernel execution: `./scripts/v3-run jupyter nbconvert --to notebook --execute ...`; 22/22 code cells executed, 0 errors; executed validation copy at `/tmp/base-dynamics-executed.ipynb`.
- Numerical checks: equal starting coordinates, energies, and forces; finite final positions; exact 8-graph/76-atom identity; source-order recovery; per-graph convergence table produced.
- Plot review: `shared/alchemi-dark.mplstyle` is loaded; custom Toolkit workflow uses NVIDIA green, `FIRE2` uses blue, target is dashed, markers distinguish the series, and axes include units. Reviewed image: `/tmp/base-dynamics-force-history.png`.
- VS Code diagnostics report nine unresolved-import warnings because Pylance is not attached to the scratch v3 interpreter. The same imports and complete notebook passed in the frozen fresh kernel.

Next:
- Integration should review notebook pacing and terminology with notebooks 01–04.
- User review required: open `notebooks/05-base-dynamics/base-dynamics.ipynb`; confirm the Mermaid flow, both DataFrames, Rich progress displays, dark force-history plot, and 18-minute lesson pacing.
- H100 timing remains a separate target-hardware check.

## 2026-08-11 23:38 EDT — revised design after N01 consolidation

Owner: N05
Status: in progress

Observed:
- Current v3 guidance and N01 establish a concept-first lesson order: start from one concrete object or result, explain the public interface, then scale the same path.
- The existing N05 numerical comparison stops before `FIRE2` reaches its default 60-step adaptation delay. The handoff calls for an interface comparison, so a short optimizer ranking does not serve the lesson.
- `BaseDynamics.step` runs hooks around `pre_update -> compute -> post_update`, evaluates its `ConvergenceHook`, and can preserve graphs whose `status` reaches the configured exit value.

Revised design:
- Lesson outcome: explain the `BaseDynamics` step lifecycle, implement a small force-driven optimizer, validate one molecular update, scale it to eight graphs, and inspect graph-level hook convergence while comparing the declared interface with `FIRE2`.
- Proposed cell sequence: title, outcomes, and lifecycle map; one import cell; editable numerical settings; load the checked molecule selection; start with ethyne; create and display one `AtomicData`/`Batch`; load and configure AIMNet2; explain and define the subclass; display its required/provided keys beside `FIRE2`; construct neighbor and convergence hooks; prime one graph; execute and inspect one step; build the eight-graph batch; construct the same workflow with per-graph status migration; run with Rich progress; shape and display convergence history; plot batch force history; recover and inspect final graphs; advanced extension; results summary.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `AtomicData.model_validate`, `add_system_property`, `Batch.from_data_list`, `Batch.clone`, `AIMNet2Wrapper.from_checkpoint`, `set_config("active_outputs", ...)`, `make_neighbor_hooks`, `BaseDynamics`, `__needs_keys__`, `__provides_keys__`, `pre_update`, `post_update`, `compute`, `step`, `ConvergenceHook.from_fmax`, `FIRE2`, and `Batch.to_data_list`.
- Molecules and model: ethyne first, followed by ethyne, acetonitrile, methanol, acetaldehyde, acetamide, pyridine, phenol, and 2,3-dimethylbutane; pinned `aimnet2-wb97m-d3_0`; requested outputs `{"energy", "forces"}`.
- Helper boundaries: the owned helper verifies shared files and the checkpoint, loads the selected molecules, applies the shared Matplotlib style, and supplies the shared Rich columns. Toolkit data construction, model configuration, subclass code, hook construction, stepping, DataFrames, and plotting remain visible.
- Expected runtime: data cells under 10 seconds on CPU; model load and the complete local RTX 4000 calculation about 10–30 seconds. CPU model execution may take several minutes. These are planning estimates; no H100 or performance claim is made.
- Validation plan: helper and lesson-contract tests; notebook JSON and complete IPython-transformed namespace parsing; Ruff; fresh-kernel execution through `./scripts/v3-run`; one-step displacement/force checks; exact eight-graph/76-atom identity; per-graph status and convergence accounting; finite recovered positions; diagnostics review; extracted dark-style NVIDIA-green plot review; user rendered review recorded separately.

Next:
- Rework the helper, tests, and notebook within the N05-owned paths, then rerun all scoped checks and fresh-kernel execution.

## 2026-08-11 23:48 EDT — concept-first rework complete

Owner: N05
Status: ready for integration

Observed:
- The current handoff, every repository Markdown file, all notebook worklogs, the routed notebook/science/benchmark rules, and N01's implemented lesson were reviewed before editing.
- The reworked notebook has 38 cells, including 23 code cells. It begins with one ethyne graph and one complete `BaseDynamics.step`, then scales the same subclass to eight graphs.
- The `FIRE2` comparison now covers the shared `BaseDynamics` inheritance and declared required/provided keys. The earlier short numerical race was removed.
- The fresh run's ethyne step changed energy from `-2104.466309 eV` to `-2104.478271 eV` and fmax from `3.623806 eV/Å` to `3.436126 eV/Å`.
- After 30 batched updates, pyridine converged at step 23 and 2,3-dimethylbutane at step 10. All eight recovered graphs had finite coordinates; the other six remain status 0 and are reported with their final forces.

Changed:
- `notebooks/05-base-dynamics/base-dynamics.ipynb`
- `notebooks/05-base-dynamics/helpers/__init__.py`
- `notebooks/05-base-dynamics/helpers/lesson.py`
- `notebooks/05-base-dynamics/tests/test_lesson.py`
- `worklog/05-base-dynamics.md`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed with the frozen Toolkit and Toolkit-Ops commits.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests notebooks/05-base-dynamics/base-dynamics.ipynb`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests`: 4 passed; 14 pinned Torch JIT deprecation warnings.
- notebook JSON and complete IPython-transformed namespace checks: passed for 38 cells and 23 code cells.
- fresh-kernel `./scripts/v3-run jupyter nbconvert --execute ...`: 23/23 code cells executed with zero errors on the NVIDIA RTX 4000 SFF Ada; validation copy at `/tmp/base-dynamics-reworked-final-executed.ipynb`.
- source cleanliness: zero saved outputs; helper/checkpoint plumbing is absent from learner cells.
- numerical checks: one-step displacement is positive; all energies, forces, and positions are finite; exact 8-graph/76-atom identity and source-order recovery pass; converged statuses remain fixed.
- plot review: shared black style loaded; the custom Toolkit result is NVIDIA green; force target, units, markers, and log scale are readable at notebook width.
- VS Code diagnostics contain eight unresolved-import warnings because Pylance is not attached to the frozen scratch interpreter. Ruff, tests, and the fresh kernel resolve and execute the same imports.
- `git diff --check -- notebooks/05-base-dynamics worklog/05-base-dynamics.md`: passed.

Blockers:
- User rendered review remains required for the Mermaid map, tables, Rich progress display, and force-history plot.
- H100 timing remains a separate target-hardware check.

Next:
- Integration can include the reworked N05 lesson in the combined navigation, execution, and rendered-bundle review.

## 2026-08-12 18:35 EDT — FIRE2-first revision brief

Owner: N05
Status: in progress

Observed:
- The current notebook teaches the `BaseDynamics` interface through a
  tutorial-local steepest-descent subclass, but it never executes the assigned
  public `FIRE2` workflow.
- Pinned source confirms that `BaseDynamics.step(...)` runs the hook-wrapped
  `pre_update -> compute -> post_update` lifecycle, validates model outputs
  declared by `__needs_keys__`, writes known model outputs into existing batch
  storage, evaluates `convergence_hook`, and returns converged graph indices.
- In the pinned release, `FIRE2.__needs_keys__ == {"forces"}` and
  `FIRE2.__provides_keys__ == {"positions", "velocities"}`. The provides set
  documents fields the optimizer updates; it does not allocate missing
  `velocities`, `forces`, or `energy` storage.
- `ConvergenceHook.from_fmax(...)` computes each system's maximum atomic force
  norm. As a registered hook with `source_status` and `target_status`, it can
  migrate per-system status; as `convergence_hook=`, it supplies the indices
  returned by `step(...)` and enables `run(...)` to stop when all systems meet
  the target.
- Official pinned docs and `examples/basic/02_geometry_optimization.py` keep
  model-provided neighbor hooks and workflow registration visible. The example
  uses `FIRE`; this lesson uses the assigned, pinned `FIRE2` implementation and
  its documented fixed-cell update.

Design:
- Lesson outcome: rebuild eight fresh molecular inputs, run one live batched
  `FIRE2` relaxation through the public API, explain the reusable
  `BaseDynamics` lifecycle and field contract, inspect per-system convergence
  and status, select/recover results, and compare one molecule before and after
  optimization.
- Prior capability reused: Part 01's `AtomicData`, `Batch`, selection, and
  recovery; Part 03's configured AIMNet2 model; Part 04's hook registration.
  New capability: one `BaseDynamics` workflow repeatedly updates a batch while
  coordinating model evaluation, neighbors, hooks, and convergence.
- Cell sequence: shared banner and short Goal/Core concepts; compact prior/new
  orientation; one small setup cell; checked molecule loading; visible
  `AtomicData`/writable-field/`Batch` construction; visible model configuration;
  visible neighbor, status, convergence, and monitor hook construction; visible
  `FIRE2` construction and `run(...)`; result shaping and display; Part 05
  course map; lifecycle diagram and one-step inspection; required/provided
  field tables; per-system convergence table and force-history visual;
  converged/pending selection and recovery; before/after structure question;
  bounded fmax exercise; recap and course handoff.
- Visual questions: the lifecycle diagram asks what happens inside one
  `BaseDynamics.step(...)`; the force-history plot asks when different systems
  cross the same force target. Each receives a one-sentence takeaway. A compact
  before/after displacement view asks where the selected molecule changed.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `use_default_velocities`, `add_node_property`, `add_system_property`,
  `Batch.from_data_list`, `AIMNet2Wrapper.from_checkpoint`, `model_config`,
  `set_config`, `make_neighbor_hooks`, `BaseDynamics`, `FIRE2`,
  `__needs_keys__`, `__provides_keys__`, `ConvergenceHook.from_fmax`,
  constructor `hooks=`, `convergence_hook=`, `step`, `run`, `index_select`,
  `get_data`, and `to_data_list`.
- Helper boundaries: checked asset/checkpoint loading, plot style, model
  freezing, Rich progress/history hook implementation, DataFrame shaping, and
  plot presentation live in tested local helpers. Learner cells retain input
  construction, model/neighbor setup, hook registration, workflow
  construction, execution, field inspection, selection, and recovery.
- Scientific system and scope: eight neutral H/C/N/O molecules from the pinned
  NCI Atlas fragment collection, 4–20 atoms and 76 atoms total; fixed-cell
  molecular geometry optimization with pinned
  `aimnet2-wb97m-d3_0`; requested energy `[B, 1]` in eV and forces `[V, 3]`
  in eV/Å; positions and velocities are `[V, 3]` in Å and Å/fs respectively;
  status is `[B, 1]` integer workflow state. The example demonstrates API
  behavior for this model and input scope, not general scientific suitability.
- Planned controls: `FIRE2(dt=0.01 fs, maxstep=0.04 Å)` with a bounded step cap
  and an explicitly displayed fmax target. Final values and convergence counts
  remain unclaimed until fresh execution on the coordinator's CUDA runtime.
- Expected runtime: checked data setup under 10 seconds; model load and first
  CUDA call about 5–15 seconds; the batched relaxation about 20–90 seconds on
  the local RTX 4000 depending on the final measured step cap. No performance
  claim or H100 result is part of this lesson.
- Try it: change a bounded `ConvergenceHook.from_fmax(...)` threshold applied
  to the final batch, select the matching systems, and verify that every
  selected system satisfies the chosen target.
- Validation plan: scoped helper and notebook-contract tests, Ruff, notebook
  design checker, `nbformat` validation, and complete transformed-namespace
  parsing through `./scripts/v3-run`. The coordinator must run the exact
  fresh-kernel CUDA command, record runtime/numerical outputs, export HTML, and
  review the banner, course map, callouts, Rich progress, tables, plots, links,
  and before/after comparison at teaching width.

Shared request:
- ID: N05-REQ-001
- For: N04 / integration
- Need: reconcile the Part 04 handoff wording with the exact pinned roles used
  here: constructor `hooks=` registers model neighbor and monitoring hooks at
  their declared stages; `convergence_hook=` is the dynamics convergence
  detector; a status-migrating `ConvergenceHook` is a separately registered
  hook.
- Why: the same public class supports detection and optional status migration,
  and N05 must not imply that passing `convergence_hook=` alone changes
  `batch.status`.
- Status: open

Next:
- Revise N05 local helpers and tests, then rebuild the notebook through guarded
  notebook-bridge edits and run only scoped static/unit checks.

## 2026-08-12 18:45 EDT — public FIRE2 rewrite completed statically

Owner: N05
Status: implementation complete; fresh CUDA execution and rendered review deferred

Handoff and baseline:
- The handoff reported cells 0–23 revised and cells 24–37 still legacy,
  6 passing helper tests, and 8 notebook-contract failures.
- Re-reading the current disk state reproduced 6 passing helper tests and
  7 failing / 4 passing contract tests. The difference from the handoff is
  recorded rather than overwriting intervening owned-path work.
- N05 was not open in the live notebook bridge, so it had no live dirty buffer
  to reconcile; the complete disk notebook was the authoritative state.
- The pinned Toolkit 0.2 docs, hooks guide, installed `BaseDynamics` and
  `FIRE2` source, and pinned basic/advanced geometry-optimization examples were
  checked before completing the rewrite.

Coordinator ruling:
- Prefer direct exploration of real public Toolkit objects over synthetic or
  replacement workflows. N05 now inspects `Batch.keys`, writable field shapes,
  hook stages/frequencies/status transitions, public FIRE2 configuration,
  step count, and final status/positions/energy/forces.
- The tutorial-local steepest-descent workflow and legacy advanced subclass
  exercise were removed. The only executed workflow in the learner path is
  public `FIRE2`; a two-step inspection reuses the same public object to expose
  the `BaseDynamics.step(...)` lifecycle.

Learner design:
- Rebuild eight neutral H/C/N/O molecules from fresh ASE inputs, allocate the
  real writable fields FIRE2 needs, and pack one 8-system / 76-atom CUDA
  `Batch`.
- Configure pinned AIMNet2 for energy `[B, 1]` in eV and forces `[V, 3]` in
  eV/Å, inspect model-provided neighbor hooks, and attach separate
  status-migrating and detector-only `ConvergenceHook.from_fmax(...)` objects.
- Inspect the constructed `FIRE2` object, execute `fire2.run(...)`, display
  per-system status, fmax, energy change, and displacement, then inspect one
  public `step(...)` lifecycle.
- Select converged and step-limited systems with `index_select(...)`, recover
  one/all systems with `get_data(...)` and `to_data_list()`, compare the
  largest-displacement structure before/after, and finish with a bounded
  convergence-threshold exercise.
- The shared Part 05 SVG remains visible and unfolded. The compact product
  overview is the only folded section. Computation, shaping, display, plotting,
  and interpretation remain separate.

Changed:
- `notebooks/05-base-dynamics/base-dynamics.ipynb`
- `notebooks/05-base-dynamics/helpers/__init__.py`
- `notebooks/05-base-dynamics/helpers/lesson.py`
- `notebooks/05-base-dynamics/tests/conftest.py`
- `notebooks/05-base-dynamics/tests/test_lesson.py`
- `notebooks/05-base-dynamics/tests/test_notebook_contract.py`
- `worklog/05-base-dynamics.md`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed; Python
  3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`,
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests`:
  19 passed; 14 pinned TorchScript deprecation warnings.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/base-dynamics.ipynb notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests`:
  passed.
- `./scripts/v3-run python /home/nfedik/.codex/skills/alchemi-tutorial-authoring/scripts/check_notebook_design.py notebooks/05-base-dynamics/base-dynamics.ipynb --part 05`:
  zero errors and zero warnings.
- `nbformat.validate`, complete IPython-transformed namespace parse, and pacing
  check: passed for 67 cells / 40 code cells; 25 code cells are at most five
  lines, the longest is 14 lines, and the notebook has zero saved outputs.
- Input/helper evidence: exact eight labels, formulas, atom counts
  `[4, 6, 6, 7, 9, 11, 13, 20]`, 76 atoms total, neutral charges, stable source
  order, and graph-level fmax reduction passed.

Fresh and rendered status:
- No fresh CUDA kernel execution was run in this guarded pass. No new energy,
  convergence-count, displacement, or runtime claim is made for the rewritten
  public FIRE2 workflow.
- Deferred fresh command:
  `./scripts/v3-run jupyter nbconvert --to notebook --execute notebooks/05-base-dynamics/base-dynamics.ipynb --output n05-base-dynamics-executed.ipynb --output-dir /tmp --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800`
- Deferred render command:
  `./scripts/v3-run jupyter nbconvert --to html /tmp/n05-base-dynamics-executed.ipynb --output n05-base-dynamics.html --output-dir /tmp`
- Review the banner, unfolded course map, both callouts, hook/field/result tables,
  Rich progress, lifecycle Mermaid diagram, convergence plot, before/after
  structure plot, links, and text density at teaching width after that run.

Shared request:
- ID: N05-REQ-001
- For: N04 / integration
- Need: reconcile Part 04 wording with constructor `hooks=` registration,
  detector-only `convergence_hook=`, and separately registered status migration.
- Status: open; N04 was not edited.

Next:
- Run the deferred fresh CUDA and HTML commands when the shared GPU is free,
  record measured per-system FIRE2 results, inspect every saved output, and
  perform the rendered learner review.

## 2026-08-12 19:15 EDT — independent-review findings resolved

Owner: N05
Status: CPU/static complete; exact AIMNet2 CUDA check and rendered review deferred

Pinned behavior verification:
- `BaseDynamics.step(...)` computes at the updated coordinates, then restores
  only mutable fields declared by the dynamics implementation for systems whose
  status has reached `exit_status`.
- In the pinned source, `FIRE2.__provides_keys__` is exactly `positions` and
  `velocities`; `BaseDynamics.step(...)` saves those fields at lines 1906–1923,
  computes at line 1929, and restores them at lines 1934–1941. Energy and forces
  are therefore not restored with protected positions.
- A deterministic CPU probe reproduced the inconsistency after three public
  FIRE2 steps: the protected coordinate remained `0.05`, while its stored force
  was `-0.049997188...` instead of the coherent `-0.05`. A direct model call at
  the restored coordinate exposed the mismatch.

Finding resolutions:

| Finding | Resolution | Evidence | Status |
| --- | --- | --- | --- |
| 1. Coherent final state | After `run(...)`, the notebook preserves returned positions/status, visibly calls public `compute_neighbors(...)`, then public `fire2.compute(...)` before any summary, plot, selection, recovery, structure comparison, or Try it operation. `summarize_relaxation(...)` now reads final energy, force, and status from that coherent batch. | CPU FIRE2 test reproduces stale outputs and verifies in-place refresh; notebook contract enforces operation order. | Resolved |
| 1a. Histories after convergence | `truncate_history_at_convergence(...)` removes every per-system row after first status 1 and replaces each retained final endpoint with the coherent post-run energy, fmax, and status. | Unit test checks truncation and endpoint replacement; plot label and interpretation describe active trajectories. | Resolved |
| 2. Model identity/configuration | Pre-run inspection names exact `aimnet2-wb97m-d3_0`, AIMNetCentral, MIT model-weight license, 14 supported elements, selected finite neutral H/C/N/O scope, live float32/device, active outputs, cutoff, MATRIX/full-list convention, and disabled wrapper Coulomb/dispersion. | Official AIMNetCentral model page/model card and Toolkit 0.2 AIMNet2Wrapper docs/source; structural contract. | Resolved |
| 3. Behavioral tests | Added deterministic CPU tests around real public `FIRE2`, `BaseDynamics.compute`, and `ConvergenceHook`: in-place ownership, two-step displacement/output parity, stale-output reproduction/repair, status migration, and all-system early stopping. | Scoped suite: 25 passed, one intentional CUDA skip. | Resolved |
| 3a. Exact CUDA validation | Added opt-in `test_cuda_fire2.py` for the pinned AIMNet2-neighbor-FIRE2 path. It is skipped unless explicitly enabled. | Deferred command and assertions below. | Deferred by instruction |
| 4. Setup collapse | `helpers.configure_presentation(LABELS)` combines style initialization with checked molecule loading. Only imports/setup cell 2 and checked-loading cell 4 carry N01's `jupyter.source_hidden` plus `hide-input` metadata; editable labels, fmax, step cap, and device remain visible in cell 3, and public Toolkit construction remains visible. | Notebook metadata assertion reports hidden code cells exactly `[2, 4]`; helper and contract tests enforce naming and visibility. | Resolved |
| 5. Lifecycle completeness | The heading and prose now state that the diagram shows selected lesson stages, and the diagram includes protected-field restoration before `AFTER_STEP`. | Notebook contract checks the revised heading and lifecycle tokens. | Resolved |
| 6. Recap navigation | Part 01 is labeled only as the course foundation, Part 04 is prerequisite review, and Part 06 is the plain-text planned next lesson. No earlier part is labeled `Next`, and no unpublished Part 06 path is linked. | The focused contract rejects any recap line beginning `Next:`, checks all three role labels, and verifies every relative `.ipynb` target exists. | Resolved |

Changed in this pass:
- `notebooks/05-base-dynamics/base-dynamics.ipynb`
- `notebooks/05-base-dynamics/helpers/__init__.py`
- `notebooks/05-base-dynamics/helpers/lesson.py`
- `notebooks/05-base-dynamics/tests/test_cuda_fire2.py`
- `notebooks/05-base-dynamics/tests/test_dynamics_behavior.py`
- `notebooks/05-base-dynamics/tests/test_lesson.py`
- `notebooks/05-base-dynamics/tests/test_notebook_contract.py`
- `worklog/05-base-dynamics.md`

Safe validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed; Python
  3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`,
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests`:
  27 passed, one intentional CUDA skip, 14 pinned TorchScript deprecation
  warnings.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/base-dynamics.ipynb notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests`:
  passed.
- `./scripts/v3-run python /home/nfedik/.codex/skills/alchemi-tutorial-authoring/scripts/check_notebook_design.py notebooks/05-base-dynamics/base-dynamics.ipynb --part 05`:
  zero errors and zero warnings.
- `nbformat.validate`, complete IPython-transformed namespace AST parse,
  metadata check, and local-link check passed for 78 cells. Hidden code cells
  are exactly `[2, 4]`, the editable-threshold cell 3 is visible, and all five
  relative notebook links resolve.

Exact deferred CUDA command:
- `RUN_N05_CUDA=1 ./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests/test_cuda_fire2.py`

The opt-in CUDA test asserts:
- positions, energy, and forces are finite;
- actual FIRE2 steps are between 1 and the 120-step cap;
- status values remain in `{0, 1}` and at least one real system migrates to 1;
- every migrated system has exactly one retained first-convergence row;
- post-run neighbor/model recomputation preserves final positions and status;
- writable batch energy/forces match both `BaseDynamics.compute(...)` outputs
  and an independent direct model evaluation at those final positions;
- the public two-step probe has finite, nonzero displacement and coherent
  one-step energy/force ownership after refresh; and
- selected/recovered system counts, atom counts, energy, and forces remain
  finite and source-aligned.

Fresh and rendered status:
- No CUDA/full fresh execution was started in this pass.
- Exact fresh notebook command remains:
  `./scripts/v3-run jupyter nbconvert --to notebook --execute notebooks/05-base-dynamics/base-dynamics.ipynb --output n05-base-dynamics-executed.ipynb --output-dir /tmp --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800`
- Exact render command remains:
  `./scripts/v3-run jupyter nbconvert --to html /tmp/n05-base-dynamics-executed.ipynb --output n05-base-dynamics.html --output-dir /tmp`

Remaining blocker:
- N05-REQ-001 remains open for N04/integration terminology reconciliation.
  N04 was not edited.

Next:
- Independent re-review can proceed on source, CPU behavior, and static design.
  CUDA numerical acceptance and rendered learner review remain the final
  execution gates.

## 2026-08-12 19:35 EDT — hook terminology and update numbering integrated

Owner: N05
Status: N05 terminology integrated; N05-REQ-001 remains open for coordinator confirmation

Pinned behavior verification:
- `BaseDynamics.__init__(...)` stores `convergence_hook` separately, then sends
  only `hooks` through `_init_hooks(...)`.
- `BaseDynamics.step(...)` dispatches registered `AFTER_STEP` hooks before
  `_check_convergence(...)` directly calls
  `self.convergence_hook.evaluate(batch)`.
- Therefore a `ConvergenceHook` supplied through `convergence_hook=` is a host
  detector, not a registered callback. Its `stage` and `frequency` fields
  support registry use but do not schedule this direct host evaluation.

N05-owned resolutions:
- The notebook now inspects `hooks=` registry objects and the
  `convergence_hook=` host detector in separate tables with explicit ownership
  columns. The FIRE2 inspection confirms that the detector is absent from
  `fire2.hooks`.
- Learner-facing history, summary, run-result, one-update result, plot axis,
  caption, and interpretation labels now use `completed update (1-based)`.
  The notebook connects this display convention once to Part 04's zero-based
  callback `ctx.step_count`; the monitor stores `ctx.step_count + 1`.
- Focused structural contracts reject detector/registry conflation and generic
  step-result labels. The CPU behavior test uses a detector frequency of 999
  and still confirms first-update host convergence, proving that registry
  frequency does not gate detector evaluation in this role.

Changed in this pass:
- `notebooks/05-base-dynamics/base-dynamics.ipynb`
- `notebooks/05-base-dynamics/helpers/lesson.py`
- `notebooks/05-base-dynamics/tests/test_dynamics_behavior.py`
- `notebooks/05-base-dynamics/tests/test_lesson.py`
- `notebooks/05-base-dynamics/tests/test_notebook_contract.py`
- `worklog/05-base-dynamics.md`

Shared request:
- ID: N05-REQ-001
- For: N04 / integration
- Status: N05 side implemented and pinned behavior verified; keep open until
  coordinator integration confirms the corresponding N04/shared terminology
  and numbering handoff.

Validation:
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests`:
  29 passed, one intentional CUDA skip, 14 pinned TorchScript deprecation
  warnings.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/base-dynamics.ipynb notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests`:
  passed.
- CUDA was not run.

## 2026-08-12 19:20 EDT — coordinator integration closure

Owner: integration
Status: final independent integration verdict PASS

Shared request:
- `N05-REQ-001`: CLOSED — verified that N04 hands relaxation, convergence, and
  `BaseDynamics` workflow behavior to N05; N05 owns completed-update numbering
  while explicitly relating it to N04's zero-based `ctx.step_count`; and both
  lessons match the shared API contract separating `hooks=` registry dispatch,
  `convergence_hook=` host detection, and separately registered status
  migration.

Remaining independent gates:
- N05 opt-in AIMNet2 CUDA test.
- N05 fresh execution.
- N05 rendered HTML review.

## 2026-08-12 20:44 EDT — N01 recap-copy parity

- Re-read the live N01 `Where NVIDIA ALCHEMI fits` cell, then copied its complete
  body—including links and the ecosystem image—without editing N01.
- The copied body is the only content inside a disclosure headed exactly
  `Where NVIDIA ALCHEMI fits (recap)`. The Part 05 curriculum orientation, map,
  and takeaway remain visible and unchanged.
- The focused contract locates N01 by its Markdown heading with `nbformat`,
  removes only the N01 heading or recap disclosure wrapper, collapses incidental
  whitespace, and compares the remaining content.
- Focused parity contracts: 3 passed across N03–N05. Scoped Ruff and N05 schema,
  cell-ID, metadata, clean-output, and source-parity validation passed.
- The design checker was run and reported 6 existing-policy errors: its obsolete
  summary and `Next:` requirements plus four current map-embed rules. CUDA and
  full notebook execution were not run.

## 2026-08-13 10:55 EDT — complete dynamics deep-dive brief

Owner: N05
Status: implementation in progress

Baseline and sources:
- The current 81-cell notebook preserves the public AIMNet2/FIRE2 relaxation,
  separate registry and host-convergence roles, coherent final recomputation,
  graph selection/recovery, force history, and before/after molecular view.
- The scoped baseline has 28 passing tests, one intentional CUDA skip, and one
  stale N01 recap-parity failure because N01 removed its ecosystem image after
  the last N05 copy. The notebook design checker reports zero errors.
- Reviewed the final Core FIRE2 and five-update NVE sections, the deep-dive
  contract, the course guide, the pinned API reference, installed Toolkit 0.2
  signatures, and the official FIRE/LJ argon, ASE integration, NVE, NVT
  Langevin, FusedStage, NPT, BaseDynamics, convergence, and custom-integrator
  material. ASE and CP2K MD guidance were checked for ensemble and conservation
  language.

Scope decision:
- Part 05 owns `BaseDynamics`, `DynamicsStage`, FIRE2, velocity initialization,
  standalone NVE and NVT Langevin mechanics, and the status/masking/shared-model
  mental model for `FusedStage`.
- Part 06 owns inflight refill, buffers, compilation, profiling, and throughput.
  N05 will link that boundary without teaching those operations.
- NPT remains a linked mechanics subtopic in this lesson: it evolves positions,
  velocities, and cell with an MTK barostat and Nosé-Hoover chains, requires
  model stress, and is not executed here. Melting-point inference belongs in a
  future R&D application.

Design:
- Start with the official Lennard-Jones argon parameters as the fast mechanics
  route. Build one periodic 27-atom simple-cubic graph, initialize seeded
  mass-aware velocities through the public custom op, inspect one five-update
  NVE trace, then run the official 200-update conservation diagnostic.
- Use separate energy-component, drift, temperature, and wrapped-trajectory
  views. State that the five- and 200-update traces are integration diagnostics,
  not equilibration or production trajectories; simple-cubic LJ argon is also
  not the stable crystal for an NPT materials study.
- Run a bounded `NVTLangevin` smoke stage from a cold seeded copy to expose
  target temperature, friction, stochastic seed, and energy exchange with the
  bath. Construct and inspect `NVT + NVE` as `FusedStage`; keep stage masking,
  statuses, one active Batch, one shared model evaluation per fused step, and
  the same-model/device constraint visible.
- Retain the supported neutral H/C/N/O AIMNet2/FIRE2 calculation as the
  interpretation route. Preserve convergence accounting, coherent final
  recomputation, recovery, force history, and matched-coordinate structure
  comparison.
- Extend the lifecycle explanation with the custom-integrator contract:
  declare `__needs_keys__` / `__provides_keys__`, override `pre_update` and
  `post_update`, modify storage in place, and leave `compute`, `step`, and `run`
  to `BaseDynamics`.

Validation plan:
- Add failing contracts and deterministic CPU behavior tests before helper or
  notebook implementation. Preserve every existing numerical and source
  contract unless the expanded public scope makes it obsolete.
- Run scoped tests, Ruff, notebook schema/AST/design checks, runtime validation,
  the opt-in CUDA FIRE2 test, and a fresh-kernel notebook execution. Export HTML
  and inspect source visibility, warnings, tables, plots, trajectory frames,
  alt text, links, width, and prose density.
- Perform two full learner-facing revision passes: first for scientific/API
  correctness and sequence, then for direct prose, cell size, and rendered
  design. Human cell approval remains required after automated and rendered
  review.

## 2026-08-13 13:48 EDT — complete deep-dive execution and review

Owner: N05
Status: implementation and automated verification complete; human cell approval required

Implemented scope:
- The first mechanics route now uses the official Lennard-Jones argon
  parameters, public seeded velocity initialization, a five-update NVE
  lifecycle trace, a 200-update conservation diagnostic, wrapped trajectory
  frames, a bounded NVT Langevin stage, and the public FusedStage construction
  contract.
- NPT remains a linked mechanics explanation: MTK cell/barostat motion,
  Nose-Hoover-chain temperature control, required model stress, periodic cell
  storage, and the explicit boundary that melting-point inference belongs in a
  future R&D application.
- The supported molecular route retains the pinned AIMNet2/FIRE2 relaxation,
  separate registered-hook and host-detector roles, coherent final
  recomputation, convergence histories, selection/recovery, and matched
  before/after coordinates.
- The notebook states repeatedly that short trajectories are API and
  integration diagnostics, not equilibration, production dynamics, or
  production scientific evidence.

Revision passes:
- Scientific/API pass: checked ensemble mechanics, Toolkit's unconstrained 3N
  temperature convention, velocity units, per-atom-per-update energy drift,
  FusedStage's shared-model/status contract, FIRE2 final-state coherence, and
  the NPT/R&D boundary against the pinned public sources and deterministic
  tests.
- Learner/design pass: tightened prose, separated computation from display,
  kept plots in dedicated cells, closed displayed Matplotlib figures, added
  exportable alt text, corrected the course-map HTML wrapper, and reconciled
  both callout styles with the current shared callout contract.
- Render pass: reviewed the complete exported notebook as a 28-page teaching-
  width PDF, including the banner, energy/conservation and temperature plots,
  wrapped trajectory, FusedStage/NPT transition, molecular tables, course map,
  lifecycle, force histories, recovery, structure comparison, and recap. The
  corrected course-map wrapper no longer prints raw HTML.

Fresh execution:
- The saved notebook contains 123 cells and 77 code cells. All 77 code cells
  have fresh execution counts and zero saved error outputs from the serialized
  CUDA run.
- The molecular result contains 8 systems / 76 atoms; all eight reached the
  displayed 0.15 eV/Å target within the 120-update cap, and the coherent final
  recomputation precedes every result summary and recovery operation.

Final validation:
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/05-base-dynamics/tests`:
  40 passed, one intentional opt-in CUDA skip, 14 pinned TorchScript
  deprecation warnings.
- `./scripts/v3-run ruff check notebooks/05-base-dynamics/base-dynamics.ipynb notebooks/05-base-dynamics/helpers notebooks/05-base-dynamics/tests`:
  passed.
- Part 05 notebook design checker: zero errors.
- `nbformat.validate`, complete transformed-namespace AST parse, execution-
  count check, and saved-output error scan: passed for 123 cells / 77 executed
  code cells / zero error outputs.
- Frozen runtime check: Python 3.12.13, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Final classic-HTML, WeasyPrint, and PyMuPDF review artifacts are under
  `/tmp/f90cb3f3-2739-42f6-a338-105874faf4c4/`.

Changed-cell review index:
- Cells 0-5: lesson framing, dependencies, settings, and checked presentation
  setup.
- Cells 6-30: Lennard-Jones argon construction, thermal velocities, NVE
  lifecycle, conservation metrics/plots, and wrapped trajectory.
- Cells 31-40: nonperiodic NVT Langevin setup, trace, endpoint table, and
  temperature plot.
- Cells 41-46: FusedStage inspection, Part 06 execution boundary, and linked NPT
  mechanics.
- Cells 47-83: supported molecular construction, field/model/hook inspection,
  FIRE2 execution, API callout, and coherent final recomputation.
- Cells 84-98: course map, BaseDynamics lifecycle, custom-integrator contract,
  and one-step public inspection.
- Cells 99-117: convergence history/plot, status selection, recovery, and
  matched-coordinate structure view.
- Cells 118-122: bounded learner exercise and recap.
- Final targeted corrections are in cells 2, 3, 15-17, 20, 23, 26, 29, 31-33,
  39, 43-44, 54, 73, 85, 88, 101, and 115 (current zero-based indices).

Temporary artifact:
- `notebooks/05-base-dynamics/tests/render_review_tmp.css` is intentionally
  retained for the rendered review. No repository cleanup was performed after
  the no-delete instruction.

Human review:
- Required. Review the current cell blocks above in order, with particular
  attention to the density of the mechanics-to-molecular transition and the
  visual scale of the API callout at normal notebook width.
