# N04 worklog — Hooks

## 2026-08-11 20:21 EDT — design record

Owner: N04
Status: in progress

Observed:
- `notebooks/04-hooks/` is absent and will be created within the owned path.
- The frozen runtime check passes with Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.

Design:
- Lesson outcome: implement a structural `Hook`, choose its `DynamicsStage` and `frequency`, receive `DynamicsContext`, and pass observation and safety hooks through a workflow constructor.
- Proposed cell sequence: outcome and hook-flow map; imports and settings; build a three-graph batch; inspect graph identity; define `EnergyHistoryHook`; inspect protocol fields; construct `FIRE2` with model, neighbor, safety, and history hooks; run; build a DataFrame; display summary; plot per-graph energy history; advanced extension.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `Batch.from_data_list`, `Batch.add_key`, `Hook`, `DynamicsContext`, `DynamicsStage`, `LennardJonesModelWrapper.make_neighbor_hooks`, `NaNDetectorHook`, `FIRE2`, and hook registration through the `hooks=` constructor argument.
- Molecules and model: Ar2, Ar3, and Ar4 clusters with Toolkit's Warp-backed Lennard-Jones wrapper using argon parameters (`epsilon=0.0104 eV`, `sigma=3.40 angstrom`, `cutoff=8.5 angstrom`).
- Helper boundaries: keep the hook class and workflow constructor visible in the notebook; use no helper module. Keep only repeated row-to-DataFrame shaping in a small local function if needed.
- Expected runtime: first CUDA run about 10-15 seconds including Warp compilation on the local RTX 4000 SFF Ada; warm GPU run below 2 seconds. CPU execution is expected to take under one minute for 16 steps and 9 atoms. No H100 timing claim.
- Validation plan: notebook JSON and static namespace checks; protocol and hook-frequency assertions; fresh-kernel execution through the frozen environment; verify 3 graphs, 9 atoms, expected history-row count, finite energies, stable `system_id` mapping, and non-increasing final energy for each cluster; inspect the shared-style Matplotlib output and record user visual review separately.

Next:
- Create the owned notebook directory and author the notebook through the live bridge.

## 2026-08-11 20:27 EDT — implementation and validation

Owner: N04
Status: ready for integration

Observed:
- The Notebook MCP bridge was available. Its cell tools required an existing file, so an empty notebook container was created once with `nbformat`; all 18 cells were then added and saved through the live bridge.
- The fresh execution used `NVIDIA RTX 4000 SFF Ada Generation`, CUDA, Torch `2.12.0+cu130`, Toolkit `0.2.0`, Toolkit-Ops `0.4.1`, Pandas `2.3.3`, Matplotlib `3.11.1`, and Rich `14.1.0`.
- The completed calculation uses 24 FIRE2 steps for 3 graphs and 9 atoms. Its visible compute cell took 2.95 seconds, including progress rendering in that fresh kernel.

Changed:
- `notebooks/04-hooks/hooks.ipynb`
- `notebooks/04-hooks/tests/test_notebook.py`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- `./scripts/v3-run pytest -q notebooks/04-hooks/tests`: 3 passed.
- notebook JSON validation and IPython-transformed parsing: 18 cells passed.
- fresh-kernel execution through `NotebookClient` with repository-root working directory: 12 code cells executed; zero cell errors.
- runtime checks: `EnergyHistoryHook` satisfies `Hook`; 24 records per graph; graph IDs `{0, 1, 2}` stayed aligned; all energies were finite; every final graph energy was below its step-0 energy.
- result summary: energy changes were `-0.020 meV` for Ar2, `-0.141 meV` for Ar3, and `-0.379 meV` for Ar4.
- text review: short expert-facing transitions; central Toolkit calls and registration remain visible; optional extension has the required advanced label.
- plot review: `shared/alchemi-dark.mplstyle` loaded; Ar4 is NVIDIA green; graph series also use distinct markers; labels and units are visible at notebook width.
- progress review: the shared Rich columns advance after each completed FIRE2 step.

Blocker:
- Automated and author visual checks are complete. User review of the rendered notebook remains required.
- H100 timing was outside this hooks lesson and was not run.

Next:
- Integration can add notebook 04 to root navigation/build outputs and include it in the combined fresh-kernel and rendered-bundle review.

## 2026-08-11 22:05 EDT — revised design record

Owner: N04
Status: in progress

Observed:
- Consolidated v3 guidance now uses agent 1's checked 32-molecule collection and AIMNet2 lesson path as the course baseline.
- Notebook 04 currently teaches the right hook interfaces, but its separate argon/Lennard-Jones example breaks the molecular and model continuity established by notebooks 01 and 05.
- The frozen runtime preflight still passes with Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.

Revised design:
- Lesson outcome: implement a structural `Hook`, select its `DynamicsStage` and `frequency`, read graph-level values from `DynamicsContext.batch`, and register observation and safety behavior with `FIRE2`.
- Proposed cell sequence: outcome and step-lifecycle map; one import cell; settings; load the checked molecule selection; display its identity; build `AtomicData` graphs and one `Batch`; load AIMNet2; explain the hook protocol; define `EnergyHistoryHook`; inspect its public fields and predicted schedule; create the complete hook list; construct `FIRE2` with `hooks=` visible; display the hook registry; run with the shared Rich pattern; shape and validate the records; display the summary; plot per-graph energy change; advanced second-stage extension; results summary.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `AtomicData.model_validate`, `Batch.from_data_list`, `Batch.add_key`, `AIMNet2Wrapper.from_checkpoint`, `model.set_config`, `model.make_neighbor_hooks`, `Hook`, `DynamicsContext`, `DynamicsStage`, `NaNDetectorHook`, `FIRE2`, the `hooks=` constructor argument, and `relaxer.step`.
- Molecules and model: Ethyne, Phenol, and 2,3-dimethylbutane from the checked NCI Atlas-derived shared molecule library; pinned `aimnet2-wb97m-d3_0`; energy and forces requested. These match agent 1's three-graph teaching batch.
- Helper boundaries: one owned helper verifies and loads the shared files and checkpoint, finds the repository root, and applies the shared Matplotlib style. The notebook keeps all Toolkit data conversion, model configuration, hook implementation, registration, execution, table construction, and plotting visible.
- Expected runtime: about 5-15 seconds on the local RTX 4000 SFF Ada for model loading plus 12 batched FIRE2 steps; CPU execution may take several minutes. No H100 timing claim.
- Validation plan: helper unit tests; notebook JSON and complete transformed-namespace parsing; static checks for the visible Toolkit path, lesson ordering, one-action cell boundaries, plot style, NVIDIA green, Rich usage, and advanced label; fresh-kernel execution through `./scripts/v3-run`; verify three graph identities, the exact frequency schedule, finite records, record counts, and non-increasing final energy; extract and inspect the rendered plot.

Next:
- Rework the helper, tests, and notebook within the N04-owned paths.

## 2026-08-11 23:43 EDT — revised implementation and validation

Owner: N04
Status: ready for integration

Changed:
- Reworked `notebooks/04-hooks/hooks.ipynb` from the isolated argon example into a 24-cell molecular lesson that continues agent 1's checked Ethyne, Phenol, and 2,3-dimethylbutane batch and pinned AIMNet2 model.
- Added `notebooks/04-hooks/helpers/__init__.py` and `notebooks/04-hooks/helpers/lesson.py` for repository discovery, checked molecule loading, checkpoint verification, shared plot setup, and suppression of one known AIMNet setup warning.
- Expanded `notebooks/04-hooks/tests/test_notebook.py` and added `notebooks/04-hooks/tests/conftest.py`.

Notebook result:
- The visible Toolkit path now covers `AtomicData.from_atoms`, `AtomicData.model_validate`, `Batch.from_data_list`, `Batch.add_key`, `AIMNet2Wrapper.from_checkpoint`, `model.set_config`, `model.make_neighbor_hooks`, `Hook`, `DynamicsContext`, `DynamicsStage`, `NaNDetectorHook`, `FIRE2(..., hooks=workflow_hooks)`, and `relaxer.step`.
- `EnergyHistoryHook(frequency=2)` recorded steps `[0, 2, 4, 6, 8, 10]` for each of the three graphs.
- Net changes from the first hook record were `-42.725 meV` for Ethyne, `-187.500 meV` for Phenol, and `-28.320 meV` for 2,3-dimethylbutane.
- The plot uses the shared dark style, distinct marker and line patterns for the graph histories, and NVIDIA green for the batch mean.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- `./scripts/v3-run ruff check notebooks/04-hooks/hooks.ipynb notebooks/04-hooks/helpers notebooks/04-hooks/tests`: passed.
- `./scripts/v3-run pytest -q notebooks/04-hooks/tests`: 7 passed.
- Notebook JSON validation and complete IPython-transformed namespace parsing: passed for 24 cells and 17 code cells.
- Fresh-kernel `jupyter nbconvert --execute` through `./scripts/v3-run`: completed in 13.8 seconds of command wall time on `NVIDIA RTX 4000 SFF Ada Generation`; all 17 code cells executed; zero cell errors; zero warning stream output.
- Runtime assertions passed for the structural `Hook` check, exact dispatched steps, six records per graph, stable IDs, and finite energies.
- Author plot review passed for labels, units, line and marker separation, legend placement, shared theme, and notebook-width readability. The extracted review image is `/tmp/hooks-N04-energy-history-reworked.png`.
- VS Code diagnostics report unresolved scientific-package imports because its selected analysis environment differs from the frozen runtime. Ruff, Pytest, and the fresh-kernel execution resolve and run those imports successfully through `./scripts/v3-run`.

Blockers:
- User visual review of the rendered plot and notebook remains required.
- H100 timing is outside this 12-minute hooks lesson and remains unreported.

Next:
- Integration can add notebook 04 to course navigation and run the combined notebook suite without changing the N04 lesson internals.

## 2026-08-12 18:16 EDT — canonical lesson redesign brief

Owner: N04
Status: in progress

Observed:
- The live 24-cell notebook is saved and clean in VS Code; the bridge reports
  revision 0 with no user or agent cell edits. The shared worktree is broadly
  dirty, while the N04 paths have no pre-existing Git diff.
- The current lesson predates the canonical banner, Part 04 course map, approved
  callouts, short Goal/Core concepts opening, bounded Try it, and two-part recap.
  Its setup leaks repository discovery, six code cells exceed 20 lines, and the
  final analysis and plot cells combine several actions.
- Pinned Toolkit docs, the exact commit example, and installed source agree that
  `Hook` is a runtime-checkable structural protocol with `stage`, positive
  integer `frequency`, and `__call__(ctx, stage)`. Dispatch occurs when
  `step_count % frequency == 0`, so a new run starts at step 0.
- The installed registry calls optional `on_register(workflow)` once after
  stage/frequency validation. `BaseDynamics.run(...)` opens and closes optional
  hook context managers around the run. Hooks registered at the same stage run
  in user-owned registration order.
- `DynamicsContext` exposes `batch`, `model`, `global_rank`, `workflow`,
  `step_count`, and `converged_mask`. `DynamicsStage` contains nine named
  boundaries from `BEFORE_STEP` through `ON_CONVERGE`.
- `model.make_neighbor_hooks()` is the supported public construction path. For
  AIMNet2 it returns a `NeighborListHook` at `BEFORE_COMPUTE`; the built-in
  `NaNDetectorHook` defaults to `AFTER_COMPUTE`.

Design:
- Lesson outcome: reuse the N01 `AtomicData`/`Batch` ownership model and the N03
  configured model boundary, then add reusable neighbor maintenance, finite-value
  safety, and graph-energy observation without editing the host workflow.
- New capability: implement a stateful `EnergyHistoryHook`, inspect
  `DynamicsContext`, select a `DynamicsStage`, predict `frequency` dispatch,
  register the complete hook list through `FIRE2(..., hooks=...)`, and observe
  registration, run-resource lifecycle, and recorded outputs.
- Cell sequence: shared banner; short Goal/Core concepts and prior/new
  orientation; compact hidden setup; load the checked three-molecule selection;
  visibly rebuild `AtomicData`, `Batch`, stable `system_id`, AIMNet2, requested
  outputs, and model-provided neighbor hooks; define the complete custom hook;
  construct safety and observation hooks; register them in a short six-step
  FIRE2 host; run once; inspect lifecycle/context/dispatch results; show the
  Part 04 course map; explain the nine-stage lifecycle with one focused diagram;
  inspect neighbor, safety, and observation responsibilities separately;
  demonstrate `NaNDetectorHook` on an isolated cloned batch; shape, display, and
  plot history in separate cells; complete a bounded frequency Try it; recap
  the reusable protocol and hand off to Part 05.
- Visible Toolkit APIs: `AtomicData.from_atoms`, `Batch.from_data_list`,
  `Batch.add_key`, `AIMNet2Wrapper.from_checkpoint`, `set_config`,
  `make_neighbor_hooks`, `Hook`, `DynamicsContext`, `DynamicsStage`,
  `NaNDetectorHook`, `FIRE2(..., hooks=...)`, `run`, and direct hook invocation
  for isolated safety testing.
- Custom hook surface kept visible: `stage`, validated `frequency`, output
  state, `on_register`, `__enter__`, `__call__`, and `__exit__`. Learners will
  see `ctx.batch`, `ctx.step_count`, `ctx.model`, `ctx.workflow`,
  `ctx.global_rank`, and the triggering stage.
- Helper boundary: local helpers own repository/checksum/checkpoint discovery,
  checked asset loading, warning/plot setup, model freezing, and Matplotlib
  layout/alternative text. Toolkit construction, configuration, hook protocol,
  scheduling, registration, execution, context inspection, result shaping, and
  safety failure remain visible.
- Scientific system and outputs: Ethyne, Phenol, and
  2,3-dimethylbutane (4, 13, and 20 atoms; 3 systems and 37 atoms total);
  pinned `aimnet2-wb97m-d3_0`; graph energy `[B, 1]` in eV and atom forces
  `[V, 3]` in eV/Å. The six-step FIRE2 run demonstrates hook semantics only and
  makes no convergence, relaxation-quality, or performance claim.
- Expected runtime: checked input and protocol-only cells below one second;
  checkpoint load plus six model evaluations about 5–15 seconds on the local
  RTX 4000 SFF Ada. CPU execution may take several minutes. No H100 claim.
- Visuals: the shared Part 04 course map answers where hooks fit; one local
  stage diagram answers when each hook can run; one energy-history plot answers
  whether the configured schedule produced graph-level records. Each receives a
  one-sentence takeaway, and the plot uses NVIDIA green for the batch mean.
- Bounded exercise: change `history_frequency` from 2 to 3 for six steps, predict
  `[0, 3]`, rerun the small workflow, and verify the observed steps and row count.
- Validation plan: local helper and protocol unit tests with a synthetic
  `DynamicsContext`; notebook JSON/nbformat and complete transformed-namespace
  parse; structural checks for the banner, map, callouts, pacing, visible APIs,
  stage names, lifecycle, helper boundary, Try it, and recap; scoped Ruff and
  shared design checker; coordinator-owned fresh CUDA execution and rendered
  review of map, diagram, callouts, tables, failure output, and plot.

Shared request:
- ID: N04-REQ-001
- For: N05 | integration
- Need: use these pinned protocol terms consistently: constructor
  `hooks=[...]`; `DynamicsStage` as named workflow boundaries;
  `DynamicsContext` as the hook snapshot; `frequency=N` dispatch when
  `step_count % N == 0`, beginning at step 0; registration order for hooks at
  the same stage; and `BaseDynamics.run(...)` as the owner of optional
  `__enter__`/`__exit__` hook lifecycle. Keep N04's six-step FIRE2 run labeled as
  a host for hook semantics; N05 owns relaxation, convergence, and
  `BaseDynamics` workflow behavior.
- Why: N05 currently teaches the same step loop concurrently and needs one
  terminology contract without duplicating or contradicting the hook lesson.
- Status: N04 preview implemented; open pending coordinator confirmation of
  N05/shared integration.

Next:
- Revise the owned helper and tests, then rebuild the notebook through guarded
  bridge edits without touching the active N01 cell or any shared path.

## 2026-08-12 19:11 EDT — API-first rebuild and teaching correction

Owner: N04
Status: implementation complete; fresh CUDA/rendered review deferred

Coordinator ruling applied:
- Prioritize direct exploration of real public Toolkit objects over a synthetic
  schedule preview. The lesson now displays all nine members of
  `DynamicsStage`, inspects `inspect.signature(DynamicsContext)`, constructs and
  inspects AIMNet2's model-provided neighbor hooks, and reads the actual host
  registry.
- `EnergyHistoryHook` is inspected before registration, after constructor
  registration, and after execution. Its visible state includes output rows,
  registry owner, context-manager lifecycle, captured workflow/batch/step
  context, and lifecycle closure count without a workflow-success claim.
- The bounded Try it changes the supported `frequency` from 2 to 3, reruns the
  same six-step host path from the preserved initial batch, and verifies the
  observable schedule changes from `[0, 2, 4]` to `[0, 3]`.
- The only isolated failure example is the documented `NaNDetectorHook`
  boundary on a cloned real batch. `FreezeAtomsHook` is constructed and
  inspected, but not applied because the molecular inputs do not provide a
  scientifically meaningful freeze category.

Changed:
- Rebuilt `notebooks/04-hooks/hooks.ipynb` through guarded bridge cell edits.
- Finalized owned helper boundaries in
  `notebooks/04-hooks/helpers/__init__.py` and
  `notebooks/04-hooks/helpers/lesson.py`.
- Finalized API, pacing, callout, lifecycle, scheduling, responsibility,
  plotting, exercise, and protocol tests in
  `notebooks/04-hooks/tests/test_notebook.py`.
- Updated `worklog/04-hooks.md`; no shared, root, environment, N05, or other
  lesson path was edited.

Learner design:
- 50 cells with one compact import cell; most learner code cells contain one
  observable action and at most five visible lines. The complete custom hook is
  the intentional long exception.
- The opening uses the canonical banner, short Goal/Core concepts, folded
  product overview, direct public-object inspection, and visible evolving Part
  04 map.
- Responsibilities remain ordered and separate: model-provided neighbor
  maintenance, constraint/control inspection, custom observation, non-finite
  safety, and complete-state snapshots.
- The six-step `FIRE2` path is explicitly a hook host and does not establish
  geometry convergence. Part 05 retains ownership of relaxation and
  `BaseDynamics`.
- Reporting is referenced once through `LoggingHook` and `StageTimingHook`,
  with the performance workflow handed to Part 06.

Validation:
- `./scripts/v3-run pytest -q notebooks/04-hooks/tests/test_notebook.py`:
  18 passed with 14 upstream Torch JIT deprecation warnings.
- `./scripts/v3-run ruff check notebooks/04-hooks/helpers notebooks/04-hooks/tests`:
  passed.
- `./scripts/v3-run ruff check notebooks/04-hooks/hooks.ipynb notebooks/04-hooks/helpers notebooks/04-hooks/tests`:
  passed after the final notebook inspection.
- Live notebook diagnostics: zero warnings or errors.
- Contract execution evidence: the custom hook registered with the test host,
  dispatched at steps `[0, 2, 4]`, recorded three calls for each of two real
  molecular graphs, captured `Registry` / `Batch` / step 4 context, and closed
  its context-manager lifecycle successfully.
- Static scientific contract: three systems, 37 atoms, six host steps,
  frequency-2 records at `[0, 2, 4]`, frequency-3 records at `[0, 3]`, and two
  full-batch snapshots scheduled at steps 0 and 3.

Deferred:
- Fresh-kernel full notebook execution and rendered learner review were not run
  in this pass. The available WSL runtime reports CUDA error 100 / no usable
  driver, and the handoff explicitly prohibited a concurrent GPU-heavy full
  execution.
- Coordinator command when CUDA is available:
  `./scripts/v3-run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/04-hooks/hooks.ipynb`
  followed by the repository's rendered-notebook review flow.

Shared request:
- `N04-REQ-001` remains open. N04 now previews the additional pinned distinction:
  the object passed through `convergence_hook=` supplies the host's
  post-`AFTER_STEP` `evaluate(batch)` convergence result, while a
  `ConvergenceHook` separately registered in `hooks=[...]` runs as an
  `AFTER_STEP` callback and can migrate matching statuses when
  `source_status`/`target_status` are configured. Coordinator confirmation of
  N05/shared integration is still required before closure.

Next:
- Coordinator runs the deferred fresh CUDA execution and rendered learner
  review, then resolves `N04-REQ-001` during N04/N05 integration.

## 2026-08-12 18:54 EDT — independent review resolution

Owner: N04
Status: ready for independent re-review; fresh CUDA/rendered review deferred

Pinned evidence:
- `BaseDynamics.run(...)` calls `_open_hooks()` before its step loop and
  `_close_hooks()` in `finally`.
- `_close_hooks()` calls every context-manager hook as
  `__exit__(None, None, None)`, including after a failed step. Hook exit proves
  resource cleanup only; it does not receive the run exception and cannot
  establish workflow success.
- `FreezeAtomsHook.stage` is the protocol's primary
  `BEFORE_PRE_UPDATE` stage. Its source-backed `_runs_on_stage(...)` predicate
  dispatches at both `BEFORE_PRE_UPDATE` and `AFTER_POST_UPDATE`; the first
  callback snapshots positions and the second restores frozen atoms and zeros
  velocities and, by default, forces.

Finding resolutions:
- False success state: removed `EnergyHistoryHook.succeeded`. Added the
  accurately named `closed_runs` counter and explicit copy that exit is cleanup,
  not workflow success. Corrected lifecycle wording to enter and exit once per
  run. CPU tests cover normal and exception paths.
- Lifecycle and freeze semantics: added the compact nine-stage sequence around
  `pre_update`, `compute`, `post_update`, and the convergence check. The lesson
  distinguishes the primary `stage` from the two documented dispatch stages in
  source-backed prose. Learner code now inspects only supported public
  `FreezeAtomsHook` configuration; the private `_runs_on_stage(...)` call was
  removed from the reusable API path.
- Runtime context: the real callback now captures and displays bounded values
  for workflow, batch, model, `step_count`, `global_rank`, and
  `converged_mask`. The triggering stage is stored separately from the context.
  Copy now states that this lesson sees `converged_mask=None` because its
  `FIRE2` host uses `convergence_hook=None`; a host that evaluates convergence
  may retain its latest mask in later callbacks.
- Real lifecycle acceptance: added
  `notebooks/04-hooks/tests/test_dynamics_lifecycle.py` with a CPU
  `ProbeDynamics(BaseDynamics)` host and public model contract. Constructor
  registration plus `run()` now own enter/exit in the tests. Coverage includes
  normal and failed cleanup, all nine callback boundaries, two-stage freeze
  snapshot/restore, registered non-finite detection, snapshots, and frequency.
  The former manual `with hook` / private registry-dispatch acceptance test was
  removed.
- Curriculum map: replaced the local embedding with the current shared
  accessible pattern: `aria-label`, `box-sizing`, `width:100%`,
  `max-width:100%`, and a progression-aware fallback alt description. The
  evolving `../../shared/curriculum-map-04.svg` remains visible and unfolded.
- Setup presentation: renamed `start_tutorial` to
  `configure_presentation`, kept public hook construction visible, and marked
  the compact import/presentation cell with N01's `source_hidden` /
  `hide-input` metadata.
- Worklog hygiene: removed the stale `No entries yet.` line and corrected the
  earlier success-signal description.

Design-checker reconciliation:
- Actionable map accessibility errors and the missing API-callout `Returns`
  token were fixed.
- Folded heading wording is adjudicated, not changed. The checker requires the
  literal `New to ALCHEMI Toolkit?`; the authoritative guide requires a short
  folded orientation but no literal heading, and the lesson's
  `Where NVIDIA ALCHEMI fits` accurately names its content.
- The 55-line custom hook is adjudicated, not split. Guide lines 170–173 allow a
  complete hook class to exceed 20 lines, and splitting it would hide the
  protocol shape the lesson teaches. The scoped contract explicitly allowlists
  this one complete class.
- The literal `Next:` label is adjudicated, not added. Guide lines 113–121
  require a two-part recap and link to the next notebook; `How we will use this`
  already links Part 05 without adding a redundant checker-only label.
- Final checker result:
  `3 error(s), 0 warning(s)`, consisting only of those three recorded
  guide/checker conflicts.

Safe validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed; Python
  3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`,
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run ruff check notebooks/04-hooks/hooks.ipynb notebooks/04-hooks/helpers notebooks/04-hooks/tests`:
  passed.
- `./scripts/v3-run pytest -q notebooks/04-hooks/tests`: 24 passed with 14
  upstream Torch JIT deprecation warnings.
- Explicit `nbformat.validate(...)` plus complete IPython-transformed
  `ast.parse(...)`: passed for 51 cells.
- Design checker: three adjudicated errors and zero warnings, as recorded above.

Deferred numerical and rendered assertions:
- Do not start a full notebook run in this CUDA-unavailable WSL session.
- On the fresh CUDA run, assert 3 systems / 37 atoms; frequency-2 history steps
  `[0, 2, 4]` and 9 rows; `FIRE2` / `Batch` / `AIMNet2Wrapper` runtime context
  at step 4, rank 0, `converged_mask=None`, and triggering stage
  `AFTER_COMPUTE`; lifecycle `registered → entered → exited` with
  `closed_runs == 1`; two complete three-graph snapshots from steps 0 and 3;
  finite main-path energies; and Try it steps `[0, 3]` with 6 rows.
- Confirm the isolated cloned-batch safety cell prints the pinned non-finite
  `RuntimeError` without interrupting later cells.
- At normal teaching width, inspect the interactive map and fallback, folded
  orientation, both approved callouts, nine-stage sequence, two freeze-stage
  values, bounded runtime-context output, summary table, plot labels/units, Try
  it success message, and Part 05 handoff.

Changed in this resolution:
- `notebooks/04-hooks/hooks.ipynb`
- `notebooks/04-hooks/helpers/__init__.py`
- `notebooks/04-hooks/helpers/lesson.py`
- `notebooks/04-hooks/tests/test_notebook.py`
- `notebooks/04-hooks/tests/test_dynamics_lifecycle.py`
- `worklog/04-hooks.md`

Remaining blocker:
- Fresh CUDA execution and rendered learner review require the coordinator's
  GPU-capable session. No N04 code or contract blocker remains.

## 2026-08-12 19:20 EDT — coordinator integration closure

Owner: integration
Status: final independent integration verdict PASS

Shared request:
- `N04-REQ-001`: CLOSED — verified that N04 hands relaxation, convergence, and
  `BaseDynamics` workflow behavior to N05; N05 owns completed-update numbering
  while explicitly relating it to N04's zero-based `ctx.step_count`; and both
  lessons match the shared API contract separating `hooks=` registry dispatch,
  `convergence_hook=` host detection, and separately registered status
  migration.

Remaining independent gate:
- N04 fresh CUDA execution and rendered learner review.

## 2026-08-12 20:44 EDT — N01 recap-copy parity

- Re-read the live N01 `Where NVIDIA ALCHEMI fits` cell, then copied its complete
  body—including links and the ecosystem image—without editing N01.
- The copied body is the only content inside a disclosure headed exactly
  `Where NVIDIA ALCHEMI fits (recap)`. The Part 04 orientation, map, and takeaway
  remain visible and unchanged.
- The focused contract locates N01 by its Markdown heading with `nbformat`,
  removes only the N01 heading or recap disclosure wrapper, collapses incidental
  whitespace, and compares the remaining content.
- Focused parity contracts: 3 passed across N03–N05. Scoped Ruff and N04 schema,
  cell-ID, metadata, clean-output, and source-parity validation passed.
- The design checker was run and reported 7 known-policy errors: its obsolete
  summary and `Next:` requirements, four current map-embed rules, and the
  previously adjudicated complete-hook length. CUDA and full notebook execution
  were not run.

## 2026-08-13 10:31 EDT — Core-following deep-dive redesign

Owner: N04
Status: implementation in progress

Sources checked:
- Final Core hook cells `core-t35` through `core-t41`, the deep-dive contract,
  the tutorial guide, the pinned API reference, and Toolkit commit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`.
- Official hook guide and source for `Hook`, `DynamicsContext`, the registry,
  `BaseDynamics.run`, `NaNDetectorHook`, `LoggingHook`, and `SnapshotHook`.
- Official custom-RDF, safety/monitoring, Rich reporting, and distributed
  monitoring examples at the pinned Toolkit commit.

Design:
- Outcome: start from the Core built-in stack, then explain how a real host
  owns registration, stage dispatch, context construction, run resources,
  convergence evaluation, and batch status.
- Sequence: a four-step CPU LJ/Ar `FIRE2` host with model-provided neighbors,
  force clamping, NaN detection, custom-backend logging, and in-memory
  snapshots; one real-host NaN failure; the nine-stage lifecycle diagram;
  direct convergence evaluator versus registered status-migrating hook; then a
  checked three-molecule AIMNet2 host with one small energy observer and a
  registered NaN guard.
- Visible results: the built-in registry, per-step log rows, snapshot count,
  real failure diagnostic, evaluator/registry comparison, custom-hook
  registration and enter/callback/exit events, `DynamicsContext` summary,
  graph-level history table, and energy-history plot.
- Ownership: host construction registers hooks; `run()` enters and exits hook
  resources; the host creates `DynamicsContext`; the active `Batch` owns
  simulation fields including `status`; the custom observer owns copied Python
  records and does not mutate `Batch`.
- Scientific scope: the LJ/Ar host is a deterministic API probe, not an MD or
  relaxation result. The four-step molecular `FIRE2` host demonstrates
  observer and safety wiring only; it does not establish convergence,
  relaxation quality, model accuracy, or performance.
- Exercise: change observer frequency on the quick LJ/Ar host, predict the
  zero-based schedule, and verify it through another ordinary `run()` call.
- Validation: test-first static and CPU lifecycle contracts; runtime preflight;
  scoped Ruff, Pytest, notebook schema and transformed namespace; design
  checker; fresh-kernel execution under the GPU lock; HTML export and rendered
  source/output/link/accessibility review; two written revision passes.

Constraints:
- No fabricated contexts, `SimpleNamespace`, private methods, direct hook
  invocation, or manual context-manager shortcuts appear in the learner path.
- Only `notebooks/04-hooks/` and this worklog are edited.

## 2026-08-13 — final Core-aligned implementation and review

Owner: N04
Status: TECHNICALLY VALIDATED, STATIC-RENDER REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED

Implemented:
- Rebuilt the lesson as 70 small cells. A four-step CPU LJ/Ar `FIRE2` host
  registers model-owned neighbor maintenance, force clamping, non-finite
  detection, custom-backend logging, and in-memory snapshots before the
  molecular example begins.
- Added a one-step pathological Ar host that raises through a registered
  `NaNDetectorHook`; a single nine-stage Mermaid lifecycle; and an ordinary-host
  comparison between `convergence_hook=` evaluation and registered
  `ConvergenceHook` status migration.
- Added the checked three-molecule AIMNet2 batch, one structural
  `EnergyHistoryHook`, registered molecular safety, visible registration /
  enter / callback / exit events, bounded `DynamicsContext` output, graph-aware
  history, an accessible energy-change plot, and a frequency-prediction
  exercise.
- Kept all central Toolkit calls in learner cells. Checks reject fabricated
  contexts, direct callback dispatch, private hook helpers, and manual hook
  context management.

Revision pass 1 — API, pedagogy, and science:
- Rechecked `Hook`, `DynamicsContext`, registry ordering, the nine
  `DynamicsStage` values, `BaseDynamics.run`, `ConvergenceHook`,
  `LoggingHook`, `SnapshotHook`, safety hooks, model-owned neighbor hooks,
  `FIRE2`, the LJ wrapper, and the AIMNet2 wrapper against the pinned source.
- Confirmed the direct evaluator runs after `AFTER_STEP` independently of its
  hook frequency, while a registered status hook follows registry dispatch and
  does not drive host early exit.
- Confirmed the active `Batch` owns `status`; the host owns registration,
  contexts, and run resources; and the observer owns only detached CPU-native
  records.
- Kept both four-step runs explicitly scoped as API demonstrations, not
  convergence, relaxation-quality, accuracy, or performance results.

Revision pass 2 — rendered learner and prose:
- Removed internal phrasing, duplicate plot rendering, stale output metadata,
  and known CPU-only dependency notices. The intentional NaN diagnostic remains
  visible as ordinary stdout.
- The accessible classic HTML export contains the closed product recap, live
  curriculum-map object and fallback, one Mermaid lifecycle, two approved
  callouts, all result tables, one plot with explicit alternative text and
  units, the successful exercise, and Core / prerequisite / Part 05 / official
  follow-on links.
- Static HTML inspection found the required lifecycle, safety, convergence,
  observer, history, and exercise outputs. Pixel-level Chromium review could
  not start because this WSL image lacks `libnspr4.so`; this is one reason the
  draft remains human-review required.

Fresh execution evidence:
- The frozen runtime passed: Python 3.12.13, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- A serialized fresh-kernel run completed all 43 code cells on the available
  CPU fallback. The host emitted four log rows and two snapshots; its final
  force maximum was `0.0029999998` eV/Å. The isolated failure reported
  non-finite force and energy fields for graph 0.
- The evaluator completed one step and left status 0; the registered migrator
  completed three steps and wrote status 1.
- The molecular observer recorded calls at steps 0 and 2, six graph rows,
  `FIRE2` / `Batch` / `AIMNet2Wrapper` context at step 2, rank 0,
  `converged_mask=None`, and status `[0, 0, 0]`. The exercise passed with
  steps `[0, 3]`.
- The executed notebook contains no cell errors and no incidental stderr. On
  this CPU-only host, the lesson helper suppresses only Warp's known
  driver-discovery probe during Toolkit initialization. A CUDA execution
  remains a human gate.

Final automated checks:
- Scoped Ruff: passed.
- Scoped Pytest: 31 passed; only 14 upstream Torch JIT deprecation warnings.
- Notebook schema, complete transformed namespace, clean source outputs, plot
  alternative text, and design checker: passed; design checker reported zero
  errors.
- Accessible classic HTML export: passed without missing-alt warnings.

Cell-level review index:
- `0`: shared banner.
- `1–2`: goal, prerequisites, folded product recap, and live course map.
- `3–10`: hidden setup and deterministic LJ/Ar host construction.
- `11–26`: built-in safety, logging, snapshots, registration, run, and visible
  counts / finite bounded values.
- `27–29`: real-host non-finite failure path.
- `30–33`: nine-stage lifecycle, `DynamicsStage`, `DynamicsContext`, and
  structural `Hook` check.
- `34–38`: direct convergence evaluation, registered status migration, and
  state ownership.
- `39–49`: molecular identities, `AtomicData` / `Batch`, AIMNet2, and
  model-owned neighbors.
- `50–58`: one custom observer, registration, ordinary run, lifecycle events,
  and bounded context.
- `59–63`: graph-level history shaping, display, accessible plot, and bounded
  interpretation.
- `64–67`: frequency prediction and verified ordinary-host exercise.
- `68–69`: recap, Part 05 handoff, and official follow-on examples.
