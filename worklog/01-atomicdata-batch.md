## 2026-08-13 — focused AtomicData teaching pass

Owner: N01
Status: ready for human review

Design:
- Lesson outcome: move from a familiar molecule to one validated `AtomicData`
  object, then pack three unequal molecules into a `Batch` and explain how
  atom rows, system boundaries, selection, and field levels work.
- Cell sequence: inspect three real molecules; convert ethyne from ASE;
  inspect required node and system fields; validate one realistic reader
  mapping; build a three-molecule `Batch`; inspect `num_nodes_per_graph`,
  `batch_idx`, and `batch_ptr`; recover and reorder molecules; add one
  meaningful system field and one meaningful atom field; complete a bounded
  selection exercise; link the full data API gallery for further exploration.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `AtomicData.model_validate`, `Batch.from_data_list`, `num_graphs`,
  `num_nodes`, `num_nodes_per_graph`, `batch_idx`, `batch_ptr`, `get_data`,
  `index_select`, `to_data_list`, and `add_key`.
- Molecules and model: ethyne, phenol, and 2,3-dimethylbutane from the pinned
  NCI collection. This is a data-structure lesson and does not run a model.
- Helper boundaries: helpers keep checked data loading, notebook setup, and the
  packed-layout graphic. Molecule conversion, validation, batching, field
  inspection, selection, recovery, and extension remain visible.
- Expected runtime: under 15 seconds on the course GPU; no CPU benchmark or
  GPU performance claim.
- Validation plan: frozen runtime check, notebook schema and full-namespace
  parse, scoped Ruff and tests, one fresh-kernel run, diagnostics, then rendered
  review of the molecule and packed-layout visuals.

Scope decision:
- Keep the gallery examples that explain the central data model.
- Move cloning, chemical hashes, mapping iteration, `exclude_keys`, batch
  serialization, full-collection scaling, and standalone neighbor construction
  to the official reference links or their owning later lessons.

Changed:
- Reduced the notebook from 83 to 63 cells and from 43 to 29 code cells.
- Kept the structure-input table, `AtomicData` field levels, a realistic reader
  mapping, variable-size `Batch` construction, packed boundaries, recovery,
  meaningful system/node fields, and a bounded reordering exercise.
- Replaced the arbitrary `atom_weight` example with `molecule_id` and an
  `is_carbon` atom mask.
- Added the built-in `node_properties`, `edge_properties`, and
  `system_properties` views so learners see the complete field groups produced
  by `from_atoms(...)`.
- Moved cloning, hashing, mapping protocol, `exclude_keys`, serialization,
  full-collection scaling, and standalone neighbor construction to official
  gallery links.
- Replaced semantic notebook tests with mechanical schema, syntax, cell-size,
  cell-ID, fence, and local-asset checks.
- Added the general gallery-selection and meaningful-field rules to
  `TUTORIAL_GUIDE.md` and the installed authoring skill.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- Ruff on the notebook, helpers, and tests: passed.
- Scoped Part 01 tests: 26 passed; 15 warnings come from the pinned TorchScript
  and AIMNet helper paths.
- Authoring-skill notebook checker: 0 errors.
- Authoring skill `quick_validate.py`: passed.
- Fresh-kernel execution: 29/29 code cells passed on NVIDIA RTX 4000 SFF Ada;
  output saved to `/tmp/atomicdata-and-batch-executed.ipynb`.
- Fresh outputs confirm CUDA tensors, field groups, `[4, 13, 20]` atom counts,
  `[0, 4, 17, 37]` boundaries, registered system/node fields, and the reordered
  `[20, 4]` exercise result.
- VS Code diagnostics: 0 warnings or errors. The live notebook is saved and
  clean at bridge revision 356.
- Rendered batch-layout plot reviewed at 2048 × 576; labels, atom symbols,
  boundaries, and packed tensor slices are legible.

Blocker:
- N01-REQ-003 remains open. The course environment still lacks the approved
  notebook-native MatterViz dependency, so the ethyne viewer cell remains a
  `REVISE — GRAPHIC` marker.

Next:
- Human review of the 63-cell pacing and rendered Markdown.
- Resolve N01-REQ-003, add the native structure viewer, rerun the fresh kernel,
  and review the interactive output in VS Code and exported HTML.

## 2026-08-13 — approved AtomicData and Batch deep-dive augmentation

Owner: N01
Status: implementation authorized; technically validated draft pending checks

Design brief:
- Outcome: convert one real ASE molecule to `AtomicData`, explain node/system
  ownership and units, pack three unequal molecules into one `Batch`, recover
  and select graphs, add registered fields at the correct level, round-trip the
  batch, and prepare one explicit neighbor representation.
- Prior knowledge: Python, ASE structures, and basic tensor shapes. This lesson
  defines Toolkit graph language before using it.
- Sequence: three-molecule question; ethyne conversion and field table;
  `model_dump` / `model_validate`; clone, equality, and chemical hash; 4/13/20
  atom Batch; `batch_idx` / `batch_ptr`; recovery and selection; mapping and
  `exclude_keys`; system/node `add_key`; serialization and round trip; explicit
  cutoff neighbors; bounded advanced-operation notes; Try it; transfer recap.
- Visible APIs: `AtomicData.from_atoms`, `model_dump`, `model_validate`,
  `chemical_hash`, `clone`, mapping access, `Batch.from_data_list`,
  `exclude_keys`, `num_nodes_per_graph`, `batch_idx`, `batch_ptr`, `get_data`,
  `index_select`, `to_data_list`, `add_key`, containment, iteration, and
  `compute_neighbors`.
- Helpers: retain checked NCI collection loading, presentation setup, and the
  existing Batch-ownership plot. Remove learner dependence on AIMNet and timing
  helpers; do not expand helper scope for central Toolkit actions.
- Structures and outputs: ethyne, phenol, and 2,3-dimethylbutane from the pinned
  NCI collection; 4, 13, and 20 atoms; 37 packed node rows; positions in Å;
  charge as a system field; custom system/node fields; matrix-format neighbor
  storage for a demonstrative 4.5 Å cutoff.
- Scientific boundary: the neighbor cutoff demonstrates data preparation, not
  a model requirement. No energy, force, accuracy, or performance claim.
  `Batch.put` / `defrag` remains non-executable because the pinned mixed-dtype
  path can skip integer fields.
- Exercise: select valid graph indices from the 32-molecule collection and
  inspect `num_nodes_per_graph` plus `batch_ptr`.
- Estimated learner time: 65–75 minutes. Target inventory: approximately
  83 cells / 43 code cells, source-clean.
- Validation: frozen runtime check; pinned `add_node_property` behavior probe;
  notebook schema and complete namespace parse; focused Part 01 tests; Ruff and
  format checks; shared design checker; fresh execution to `/tmp`; diagnostics;
  then cell-by-cell and rendered human review.

# N01 worklog — AtomicData and Batch

## 2026-08-11 20:16 EDT — lesson design

Owner: N01
Status: in progress

Observed:
- The frozen runtime check passes with Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`, and the local NVIDIA RTX 4000 SFF Ada GPU.
- The shared library contains 32 distinct neutral H/C/N/O molecules, 4–20 atoms and 322 atoms total. Ethyne, phenol, and 2,3-dimethylbutane contain 4, 13, and 20 atoms.
- The pinned Toolkit uses `AtomicData.model_validate(...)` for current Pydantic validation. The inherited `validate(...)` spelling emits a deprecation warning.
- `AIMNet2Wrapper` requires a neighbor matrix for this path; `compute_neighbors(...)` prepares it from the wrapper's public `neighbor_config`.

Design:
- Lesson outcome: build and validate molecular `AtomicData`, assemble a variable-size `Batch`, inspect graph membership and boundaries, evaluate the same requested outputs through serial and batched Toolkit calls, verify numerical agreement, and recover source-ordered systems and results.
- Proposed cell sequence: title/outcome/data-flow map; imports and editable settings; load and verify the 32-molecule collection; display the collection; convert to `AtomicData` and validate; build the ragged `Batch`; load/configure AIMNet2 and prepare neighbors; run serial/batched correctness calls; measure warm calls with Rich progress; display timing results; plot serial/batch timing and throughput; inspect one `AtomicData`; inspect `batch_idx`, `batch_ptr`, and `num_nodes_per_graph`; recover graphs with `get_data`, `index_select`, and `to_data_list`; concise reuse task and summary.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `AtomicData.model_validate`, `Batch.from_data_list`, `batch_idx`, `batch_ptr`, `num_graphs`, `num_nodes_per_graph`, `get_data`, `index_select`, `to_data_list`, `AIMNet2Wrapper.from_checkpoint`, `set_config("active_outputs", ...)`, and `compute_neighbors`.
- Molecules and model: all 32 packaged NCI Atlas fragments; highlight ethyne, phenol, and 2,3-dimethylbutane; use `aimnet2-wb97m-d3_0` with the checksum in `environment/runtime-pins.toml` and request `{"energy", "forces"}` in both execution paths.
- Helper boundaries: notebook-local file/checksum and collection loading; CUDA synchronization and one-call timing; conversion of measured samples to rows. Toolkit data construction, batching, model configuration, model calls, result attachment, recovery, DataFrame display, and Matplotlib remain visible in cells.
- Expected runtime: data-only cells under 10 seconds on CPU; model load and first call about 5–15 seconds on the local RTX 4000; warm comparison about 10–30 seconds for the planned repetitions. CPU-only model execution may take several minutes. These are execution estimates, not reportable benchmark results. H100 timing is a separate target-hardware check.
- Validation plan: notebook JSON and static top-to-bottom namespace checks; notebook-local helper tests through `./scripts/v3-run`; fresh-kernel execution through the live bridge; serial/batch energy and force tolerances; 32-graph/322-atom identity and recovery checks; diagnostics scan; manual review of the styled plot and rendered notebook; separate H100 timing later.

Next:
- Implement notebook-local helpers and tests, then create and execute `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb` through the live notebook bridge.

## 2026-08-11 20:23 EDT - helpers and Toolkit path validated

Owner: N01
Status: in progress

Observed:
- The current model path needs `compute_neighbors(...)` before `AIMNet2Wrapper(...)` evaluation.
- On the local RTX 4000, a three-repeat smoke measurement gave a 411.1 ms median for 32 serial calls and 12.2 ms for one 32-graph batch. This is a development smoke result, not the required H100 measurement.
- Maximum serial/batch differences in that smoke run were `9.75e-4 eV` for graph energy and `1.73e-5 eV/angstrom` for atomic force. The notebook will show the measured values and use explicit absolute tolerances.
- The live notebook bridge is active, but it has no create-notebook action. It currently exposes only notebooks owned by N02 and N05. The target N01 notebook must exist and be open before cell edits can begin.

Changed:
- `notebooks/01-atomicdata-batch/.gitignore`
- `notebooks/01-atomicdata-batch/helpers/__init__.py`
- `notebooks/01-atomicdata-batch/helpers/lesson.py`
- `notebooks/01-atomicdata-batch/tests/conftest.py`
- `notebooks/01-atomicdata-batch/tests/test_lesson.py`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- `./scripts/v3-run ruff check notebooks/01-atomicdata-batch/helpers notebooks/01-atomicdata-batch/tests`: passed.
- `./scripts/v3-run pytest notebooks/01-atomicdata-batch/tests -q`: 8 passed on the local CUDA runtime.
- Full 32-molecule Toolkit path: serial and batched energy/force checks passed; output recovery through `Batch.index_select(...).to_data_list()` passed.

Blocker:
- Create and open a blank `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb` in VS Code. All notebook content and execution will then use the live bridge.

Next:
- Populate the open notebook through guarded bridge edits, run it from a fresh frozen-environment kernel, inspect diagnostics and plot output, and add notebook contract checks.

## 2026-08-11 20:37 EDT - notebook implemented and fresh-kernel validated

Owner: N01
Status: ready for integration

Observed:
- The notebook contains 32 cells, including 20 code cells. It uses short Markdown, one main action per code cell, plain Pandas and Matplotlib, the shared dark style, NVIDIA green for the batched result, and the shared Rich progress columns for repeated timing.
- The first fresh-kernel run completed all code cells and exposed a zero-based graph-index to one-based source-order display error. Cell 25 was corrected through the live bridge. The second fresh-kernel run completed all 20 code cells with zero errors.
- Corrected fresh run on the local RTX 4000 SFF Ada: maximum serial/batch difference was `4.88e-4 eV` for graph energy and `1.8e-5 eV/angstrom` for atomic force. The displayed absolute tolerances are `1.5e-3 eV` and `3.0e-5 eV/angstrom`.
- The seven-repeat local warm measurement reported 514.4 ms for 32 serial calls and 14.9 ms for one 32-graph batch, or 34.5x on that run. This is local development evidence. H100 timing remains unmeasured.
- VS Code has not started a live kernel for this notebook. Bridge execution timed out with all code cells pending. Pylance therefore reports unresolved environment imports even though the frozen Jupyter run and scoped tests pass.

Changed:
- `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb`
- `notebooks/01-atomicdata-batch/helpers/__init__.py`
- `notebooks/01-atomicdata-batch/helpers/lesson.py`
- `notebooks/01-atomicdata-batch/tests/conftest.py`
- `notebooks/01-atomicdata-batch/tests/test_lesson.py`
- `notebooks/01-atomicdata-batch/tests/test_notebook_contract.py`
- `notebooks/01-atomicdata-batch/.gitignore`

Validation:
- Notebook bridge guarded changes: 34 agent changes, zero user-cell changes; notebook saved after each correction.
- `./scripts/v3-run ruff check notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb notebooks/01-atomicdata-batch/helpers notebooks/01-atomicdata-batch/tests`: passed.
- notebook JSON, `nbformat.validate`, IPython-transformed full-namespace parse: passed for 32 cells and 20 code cells.
- `./scripts/v3-run pytest notebooks/01-atomicdata-batch/tests -q`: 12 passed on the local CUDA runtime. The warnings come from pinned Torch JIT deprecations and AIMNet cutoff extraction.
- Fresh-kernel execution: `/tmp/atomicdata-and-batch-executed-v2.ipynb`, 20/20 code cells, zero errors.
- Plot review: shared black style loaded; serial is blue with diagonal hatching; the main 32-graph batch is NVIDIA green with dotted hatching; titles, labels, units, and values are readable at notebook width.

Blockers:
- H100 timing and profiling are pending target hardware.
- User review is required: open `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb`, select the frozen v3 Python kernel, run all cells, and confirm the Mermaid diagram, tables, Rich progress, and two-panel timing plot render at notebook width.

Next:
- Integration can add notebook 01 to root navigation and combined checks after reading all notebook worklogs.

## 2026-08-11 23:36 EDT — learner-first redesign

Owner: N01
Status: in progress

Observed:
- The live notebook has 36 cells and no saved outputs. Eight visible code cells exceed 20 lines; the longest is 73 lines. Imports and setup are visible, there is no ALCHEMI orientation or molecular viewer, and the performance section compares individual calls with a batch on one selected device instead of teaching CPU and GPU explicitly.
- The notebook bridge is healthy at version 0.10.0. Its change log contains 74 agent edits and zero user edits for this notebook; no cell is protected or actively focused.
- The frozen runtime passes at the selected Toolkit and Toolkit-Ops commits. PyTorch and Warp are installed. JAX is outside this frozen environment. Toolkit Core uses PyTorch tensors; Toolkit-Ops supplies PyTorch and optional JAX bindings, and many Toolkit-Ops implementations use Warp.
- The official MatterViz notebook route is `pymatviz.StructureWidget`, which accepts ASE `Atoms`. Neither `pymatviz` nor a MatterViz Python package is installed in the frozen v3 environment.

Design:
- Lesson outcome: understand ALCHEMI as reusable pieces for building atomistic simulations; run one useful molecular batch; rebuild that result from ASE `Atoms`, `AtomicData`, and `Batch`; inspect ragged graph identity and recover systems; verify individual and batched model results; move the same 32-graph calculation between CPU and GPU and interpret response time and throughput.
- Proposed cell sequence: title and outcomes; full ALCHEMI role and reusable-pieces framing; synchronized curriculum/runtime Mermaid; PyTorch, optional JAX, Toolkit-Ops, and Warp diagram; one collapsed setup cell; three-molecule batch primer and energy result; one central warm `HIGHLIGHT`; create, print, check, and view ethyne with MatterViz; convert it to `AtomicData`; inspect atom and system fields plus tensor dtype/device; rebuild three `AtomicData` objects; construct one ragged `Batch`; inspect and plot `batch_idx`/`batch_ptr`; recover with `get_data`, `index_select`, and `to_data_list`; request energy and forces; run the individual and batched paths and compare them; build the 32-graph batch; move equal work to CPU and GPU; verify device agreement; time warm public model calls; display a DataFrame; plot response time and throughput; transfer exercise; API recap and Part 02 link.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, important tensor fields, `add_system_property`, `Batch.from_data_list`, `num_graphs`, `num_nodes_per_graph`, `batch_idx`, `batch_ptr`, `get_data`, `index_select`, `to_data_list`, `Batch.to`, `AIMNet2Wrapper.from_checkpoint`, `model_config`, `set_config("active_outputs", ...)`, `compute_neighbors`, and direct `model(batch)` calls.
- Molecules and model: ethyne, phenol, and 2,3-dimethylbutane form the primer and ragged-layout example; all 32 neutral H/C/N/O molecules form the device comparison; use the verified `aimnet2-wb97m-d3_0` checkpoint and the same `{"energy", "forces"}` request for correctness and device comparisons.
- Helper boundaries: helpers own checked data/checkpoint loading, plot-style and warning setup, MatterViz serialization/display plumbing, repeated synchronization/timing, model freezing/configuration used only for repeated device setup, and compact display-row construction. Learner cells retain every central Toolkit data, batch, neighbor, model, device-movement, recovery, DataFrame-display, and Matplotlib action.
- Expected runtime: data and inspection cells under 10 seconds on CPU; first model load and first CUDA neighbor/model path about 5–15 seconds on the local RTX 4000; three-system individual/batch check under 10 seconds; CPU/GPU 32-graph warm comparison about 10–30 seconds total including both model loads. CPU-only completion may take several minutes. These are planning estimates; H100 timing remains separate.
- Validation plan: notebook JSON and IPython-transformed full-namespace parse; source checks for the approved callout set, synchronized diagrams, MatterViz use, collapsed setup, visible APIs, code-cell length, plot style, and absence of learner-facing repository guards; helper unit tests; exact 32-graph/322-atom identity; batch boundary/recovery checks; individual/batch and CPU/GPU numerical agreement; fresh-kernel execution through `./scripts/v3-run`; VS Code diagnostics; rendered inspection of MatterViz, both Mermaid diagrams, batch strip, tables, and CPU/GPU plot.

Shared request:
- ID: N01-REQ-001
- For: integration
- Need: add and pin the official `pymatviz.StructureWidget` MatterViz notebook route, including any widget asset/export requirements, in the frozen environment and notices.
- Why: molecule lessons now require MatterViz, while the owned notebook cannot change the shared lock. The source notebook will use the official hosted MatterViz viewer bridge until this request is resolved.
- Status: open

Shared request:
- ID: N01-REQ-002
- For: integration
- Need: replace the old notebook-card guidance with one shared learner-first authoring guide. It must carry the complete teaching system below, not only a code-cell limit.
  - Course framing: Part 01 introduces ALCHEMI as reusable data, model, hook, dynamics, and execution tools for constructing atomistic simulations. Later parts use a folded first-time introduction, link the earlier API lesson, state what they reuse, and name the one new capability.
  - Lesson progression: start with one useful result; rewind to the smallest familiar scientific object; introduce each Toolkit abstraction when the learner needs it; rebuild the result; then change one dimension at a time such as one structure to a batch or CPU to GPU. Follow the pacing and concrete-object style of strong ASE, NumPy, and SciPy tutorials.
  - Cell pacing: one observable action per code cell, normally 1–5 lines and at most 15–20 visible lines. Split computation, DataFrame display, plotting, and interpretation. A longer cell needs a genuine educational reason. Setup, repository plumbing, checksums, warning handling, and repeated timing mechanics stay hidden or in tested helpers.
  - API visibility: learners directly see the central public Toolkit construction, configuration, execution, inspection, and recovery calls. Helpers may prepare assets and presentation mechanics; they do not return an opaque ready-to-run Toolkit workflow.
  - Explanations: short expert-facing Markdown defines the scientific object, its tensor or graph meaning, and the output immediately around the cell that uses it. Remove repository-internal instructions and motivational or promotional preambles.
  - Visual teaching: use MatterViz whenever learners inspect a molecule or atomic structure. Use ordinary Matplotlib for quantitative figures and `shared/alchemi-dark.mplstyle` for every plot. Every graphic answers a learner question introduced immediately before it; NVIDIA green marks the main Toolkit result.
  - Callouts: use only two styled elements. `💡 HIGHLIGHT` is a light warm-neutral box for the one mental model worth remembering. `NVIDIA ALCHEMI TOOLKIT · API` is an NVIDIA-styled API reference with a prominent signature and compact input/result meaning. Results, notes, limitations, transitions, and exercises remain ordinary Markdown.
  - Diagrams: keep one synchronized curriculum/runtime Mermaid map across the series and highlight the current lesson/path. Use solid fills for hierarchy, NVIDIA green for the active path, charcoal or warm-neutral fills elsewhere, NVIDIA Sans with Arial/system fallback, rounded clean silhouettes, visually quiet borders, and muted connectors. Use no gradients, shadows, badges, or outline-first styling.
  - Framework framing: Toolkit Core uses PyTorch tensors and workflows. Toolkit-Ops provides PyTorch bindings and optional JAX bindings for selected operations; many implementations use Warp kernels on CPU or CUDA. A diagram is sufficient in Part 01 unless an operation directly advances the AtomicData/Batch lesson.
  - Continuity: every later notebook rebuilds fresh-kernel state, links the exact prior API lesson, and explains how its new module connects to the same `AtomicData`/`Batch`/model workflow. Optional depth is labeled `Advanced — if time permits / homework.`
  - Author review: leave precise source comments such as `REVISE — GRAPHIC`, `REVISE — CALLOUT`, `REVISE — WORDING`, and `REVISE — CROSS-LINK` where educator judgment remains. Acceptance includes structural tests, fresh-kernel execution, and rendered learner review of pacing, diagrams, callouts, MatterViz, tables, and plots.
- Why: these decisions must guide every notebook without copying a long start prompt into each agent chat.
- Status: open

Next:
- Refactor the owned helper and tests, rebuild the notebook entirely through guarded live-bridge edits, and execute it from a fresh frozen-environment kernel.

## 2026-08-12 — reference build under the canonical authoring skill

Owner: N01
Status: in progress

Design:
- Lesson outcome: use ALCHEMI as a collection of reusable simulation tools;
  build one useful molecular batch, then explain and reuse `AtomicData`,
  `Batch`, graph identity, recovery, model outputs, and device movement.
- Cell sequence: compact ALCHEMI orientation; synchronized course map; small
  three-molecule model primer; ethyne print/check/MatterViz; `AtomicData`
  conversion and focused validation; PyTorch/JAX/Toolkit-Ops/Warp map; rebuild
  the ragged batch; inspect counts, membership, and boundaries; batch strip;
  recovery APIs; serial/batch agreement; attach graph and atom results through
  `Batch`; build 32 graphs; run equal work on CPU and GPU; table, plot, transfer
  task, API recap, and Part 02 link.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `AtomicData.model_validate`, `Batch.from_data_list`, `add_key`, `batch_idx`,
  `batch_ptr`, `num_nodes_per_graph`, `get_data`, `index_select`,
  `to_data_list`, `Batch.to`, `AIMNet2Wrapper.from_checkpoint`,
  `model_config`, `set_config`, `compute_neighbors`, and `model(batch)`.
- Molecules and model: ethyne, phenol, and 2,3-dimethylbutane for the primer;
  all 32 NCI Atlas molecules for device comparison; verified
  `aimnet2-wb97m-d3_0`; energy first, then energy and forces.
- Helper boundary: checked data/checkpoint loading, MatterViz serialization,
  plot setup, model freezing, synchronization, timing, and repeated row
  formatting. Learner cells keep the public Toolkit path and ordinary tables
  and plots.
- Expected runtime: first model load and primer 5–15 seconds on the local GPU;
  data cells under 10 seconds; equal-work CPU/GPU comparison 10–30 seconds.
  H100 measurement remains separate.
- Validation: canonical skill check, local Ruff and Pytest, namespace parse,
  batch identity/recovery, serial/batch and CPU/GPU agreement, fresh-kernel
  execution, bridge diagnostics, and rendered inspection of both Mermaid maps,
  MatterViz, the batch strip, callouts, tables, and timing plot.

Observed:
- N01-REQ-002 is resolved by the canonical guide, shared assets, and installed
  `$alchemi-tutorial-authoring` skill.
- The bridge supports guarded cell edits and execution. It does not expose cell
  metadata editing, so the setup cell will be compact; no non-bridge notebook
  edit will be used to force collapsed metadata.

Next:
- implement every notebook cell through the live bridge and validate the
  complete reference build.

## 2026-08-12 — learner-facing reference revision

Owner: N01
Status: ready for user review

Changed:
- Reframed the opening around the NVIDIA ALCHEMI ecosystem, with the shared
  Toolkit / Toolkit-Ops / NIM graphic, short official links, and a closing
  acknowledgment and contact section.
- Replaced the dependency-heavy map with one linear curriculum column and one
  simulation-capability column. Part 01 and simulation-ready data form the
  single green row.
- Combined the opening Toolkit path into one 19-line primer with a compact
  three-molecule energy table and a light molecular gallery. Every public call
  is then rebuilt in small cells.
- Reversed the framework diagram into its implementation direction: many Warp
  kernel implementations feed Toolkit-Ops, which exposes PyTorch and optional
  JAX bindings; Toolkit Core follows PyTorch.
- Refined the warm `Highlight` and dark `ALCHEMI TOOLKIT API` callouts to use
  quiet one-pixel borders, flat fills, and signature-led hierarchy.
- Renamed the learner exercise to `Try it: filter the batch` and gave it an
  explicit `batch_ptr` success check.
- Expanded validation bypass into a complete advanced mini-lesson: verify
  field and boundary parity, measure both construction paths, display the
  result, and limit the interpretation to the current input.
- Added a hosted MatterViz view with an expandable local static fallback and
  kept every Matplotlib figure on the shared dark style.
- Updated helper and notebook tests for the molecule strip, pacing, exact
  shared map/callouts, ecosystem orientation, framework direction, visual
  question/takeaway pattern, advanced lesson completeness, and display/plot
  separation.

Observed:
- The notebook contains 103 cells: 64 code and 39 Markdown. The longest code
  cell is the 19-line primer; 45 code cells are five lines or shorter.
- Final fresh execution completed all 64 code cells with zero errors on the
  local RTX 4000 SFF Ada.
- Fresh results: serial/batch maximum differences were about `1e-6 eV` and
  `2e-6 eV/Å`; CPU/GPU maximum differences were `9.77e-4 eV` and
  `1.4e-5 eV/Å`, within the displayed tolerances.
- The optional seven-repeat warm measurement reported about `40.0 ms` CPU and
  `10.2 ms` GPU for the 32-graph, 322-atom model call on this machine.
- The optional construction study reported about `0.205 ms` validated and
  `0.150 ms` trusted construction for this in-memory input. These are local
  tutorial observations, not general performance claims.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- Notebook JSON, `nbformat.validate`, and complete transformed namespace parse:
  passed.
- `./scripts/v3-run ruff check` for the notebook, helper, and tests: passed.
- `./scripts/v3-run pytest -q notebooks/01-atomicdata-batch/tests`: 34 passed.
  The 15 warnings are upstream TorchScript deprecations and one pinned AIMNet
  tensor-conversion warning.
- Fresh-kernel execution: 64/64 code cells, zero errors. Generated three
  Matplotlib figures plus MatterViz HTML and a static fallback. Every
  Matplotlib output carries exportable alternative text; a classic-template
  HTML export retained all three descriptions.
- Manual image inspection: the ecosystem image, primer molecule gallery, batch
  ownership strip, and CPU/GPU timing plot are legible with clear hierarchy and
  units. The primer gallery received extra lower spacing after inspection.
- Official ALCHEMI, Toolkit, Toolkit-Ops, BGR NIM, and MatterViz links resolve.

Blockers and review:
- User review remains required for the two Mermaid layouts, the exact callout
  feel inside VS Code, and hosted MatterViz rotation/fullscreen on the workshop
  network. Source, structural, numerical, static fallback, and local image
  checks pass.
- H100 profiling remains Part 06 work. Part 01 explains the boundary and does
  not claim target-hardware performance.

Next:
- Use the refined `$alchemi-tutorial-authoring` skill to revise the remaining
  notebooks from their current content, preserving the parts Kelvin's feedback
  already improved.

## 2026-08-12 — compact vertical course map

Owner: N01
Status: ready for user review

Changed:
- Replaced the wide left-to-right course map with a top-to-bottom curriculum
  spine and diagonal branches to the simulation capability enabled by each
  part.
- Selected Mermaid's linear edge routing for straight, angled connectors and
  removed the large subgraph frames.
- Synchronized the notebook, shared template, authoring guide, installed skill,
  checker, and Part 01 tests.

Validation:
- Part 01 design checker: zero errors and zero warnings.
- Ruff: passed.
- Part 01 tests: 35 passed; 15 upstream Torch and AIMNet warnings remain.
- Skill validation: passed.

Next:
- Review the vertical map at notebook width in VS Code.

## 2026-08-12 - molecule-first input and batching revision

Owner: N01
Status: in progress

Design:
- Lesson outcome: load molecular structures through a normal chemistry reader,
  convert them to `AtomicData`, show that one 32-molecule `Batch` evaluates the
  same work faster than 32 separate model calls, then inspect how Toolkit keeps
  each molecule's atom rows and boundaries separate.
- Cell sequence: one polished serial-versus-batched result; select and inspect
  one molecule loaded from extended XYZ; explain ASE and optional pymatgen input
  paths; view the molecule; convert it to `AtomicData`; validate a realistic
  active-region selection; relate molecules to atomic graphs; build and inspect
  the unequal-size `Batch`; retain recovery, model-output, CPU/GPU, `Try it`, and
  optional construction lessons.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `AtomicData.from_structure` as a documented optional path,
  `AtomicData.model_validate`, `Batch.from_data_list`, `batch_idx`, `batch_ptr`,
  `num_nodes_per_graph`, `get_data`, `index_select`, `to_data_list`, `Batch.to`,
  `set_config`, `compute_neighbors`, and `model(batch)`.
- Molecules and model: the full 32-molecule neutral H/C/N/O collection for the
  opening speed comparison; ethyne, phenol, and 2,3-dimethylbutane for the
  visual and ragged-batch examples; pinned AIMNet2 energy for the opening and
  energy plus forces for later checks.
- Helper boundaries: helpers own synchronization, repeated warm timing,
  timing-table shaping, checked input loading, checkpoint checks, MatterViz,
  and plot setup. The notebook keeps both the 32-call expression and the
  one-batch model call visible.
- Expected runtime: opening warm timing about 2-10 seconds on the local GPU and
  about 1-3 minutes on CPU; the complete GPU notebook about 1-2 minutes. These
  are planning estimates until the fresh-kernel run records current values.
- Validation plan: frozen runtime check; helper and notebook tests; exact
  serial/batch energy agreement before the speed claim; 32 molecules and 322
  atoms; complete transformed-namespace parse; fresh-kernel execution; and
  rendered review of timing output, molecule views, diagrams, and wording.

Writing delta:
- Observed draft: it opened with a named "primer," constructed ethyne from
  borrowed coordinates, invented a broken mapping, and used "graph" before
  defining its chemistry meaning.
- Educator direction: open with a useful measured result, follow the normal
  structure-reader path, make validation catch a believable project error, and
  lead with molecule language before introducing the graph representation.
- Reusable rule: preserve the audience's familiar scientific noun until the
  library abstraction is defined; every error example needs a plausible source;
  performance claims compare equal work and show the timed boundary; tutorial
  prose uses one stable term per concept and concrete actors and actions.

## 2026-08-12 - molecule-first revision executed

Owner: N01
Status: ready for user review

Changed:
- `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb`
- `notebooks/01-atomicdata-batch/helpers/__init__.py`
- `notebooks/01-atomicdata-batch/helpers/lesson.py`
- `notebooks/01-atomicdata-batch/tests/test_lesson.py`
- `notebooks/01-atomicdata-batch/tests/test_notebook_contract.py`

Observed:
- The opening compares the same 32 AIMNet2 energy requests as separate calls
  and as one `Batch`. Energy agreement passes before timing.
- Fresh local RTX 4000 execution measured `235.3 ms` for 32 one-molecule model
  calls and `7.3 ms` for one batched model call, a `32.3x` warm model-call
  speedup.
  Conversion, transfer, neighbor construction, and warm-up were outside the
  timed region.
- The input lesson now states the two public converters: ASE `Atoms` through
  `AtomicData.from_atoms(...)`, and optional pymatgen `Structure` or `Molecule`
  through `AtomicData.from_structure(...)`. ASE or pymatgen owns file parsing.
- Ethyne now comes from the loaded extended XYZ collection. The validation
  example applies an active-region mask to positions and shows the mismatch
  created when the same mask misses atomic numbers.
- Learner prose introduces molecules first and defines each molecule as one
  atomic graph at the `AtomicData` boundary. Learner-facing `primer` wording is
  gone.
- A fresh-reader pass led to five final corrections: the opening names the
  measured GPU and uses parallel model-call labels; the atomic-graph text states
  when neighbor edges appear; validation claims only atom-row alignment and
  preserves charge; recovery is described by graph position; and the `Try it`
  range stays within the 20-atom collection maximum.
- Reviewed outputs are saved in the learner notebook, including the opening
  performance table and molecule strip.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed at the frozen
  Python, Toolkit, and Toolkit-Ops pins.
- Ruff for notebook, helper, and tests: passed.
- Part 01 tests: `42 passed`; 15 warnings come from pinned TorchScript and
  AIMNet code.
- Shared design checker: zero errors and zero warnings.
- Fresh-kernel execution: 64/64 code cells, zero errors, 13.6 seconds across the
  executed cell interval. The masked record raised the expected 4-versus-3
  atom-count error; the aligned record produced shapes `[3]` and `[3, 3]`.
- Image review: the three-molecule opening, molecule-ownership strip, CPU/GPU
  plot, and local ethyne fallback are legible with units and NVIDIA green on
  the main Toolkit result.
- Final HTML export completed. Nbconvert warned that it did not map three Matplotlib
  descriptions into HTML `alt` attributes; equivalent descriptions remain in
  output metadata and adjacent learner text.

User review required:
- Open `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb` with the
  `/tmp/alchemi-v3-runtime/venv/bin/python` kernel. Run all cells and review the
  two Mermaid diagrams, callouts, and hosted MatterViz rotation/fullscreen.

## 2026-08-12 - fixed-seed CPU/GPU opening and computation graph

Owner: N01
Status: in progress

Changed:
- Replaced the 32-call opening with a fixed-seed 2,048-molecule heterogeneous
  `Batch` evaluated on CPU and GPU with energy and forces.
- Added numerical agreement before timing, three warm `model(batch)` repeats,
  throughput, hardware identity, and peak PyTorch CUDA allocation.
- Replaced Matplotlib molecule panels with a three-structure MatterViz gallery
  and model-predicted maximum-force values.
- Added a Toolkit-visible computation diagram from molecules through
  `AtomicData`, `Batch`, neighbor construction, energy, autograd, and forces.
- Removed the duplicate optional CPU/GPU timing section. The optional section
  now contains one complete validated-versus-trusted construction study.
- Moved opening plot layout and MatterViz caption assembly into tested helpers;
  sampling, public Toolkit calls, device movement, correctness, and the measured
  call remain visible.

Measured planning probe:
- RTX 4000 Ada, 2,048 molecules / 20,796 atoms: CPU median 2,718 ms; GPU median
  132 ms; GPU peak allocated memory 1,292 MiB of 19,195 MiB.
- The probe used three warm energy-and-force calls and excludes data conversion,
  transfer, neighbor construction, and model loading.

Cross-notebook request:
- `N01-REQ-001`: Part 05 should run FIRE2 to a stated maximum-force target,
  report convergence per molecule, and compare before/after structures with
  MatterViz. Part 04 keeps its short FIRE2 run as the host workflow for teaching
  hook stage and frequency.

Validation so far:
- Frozen runtime check: passed.
- Helper/test Ruff check: passed.
- Part 01 tests after the first rewrite: 45 passed; 15 upstream warnings.
- Fresh-kernel execution was interrupted when the opening presentation was
  moved into helpers. A new complete run is still required.

## 2026-08-12 - docs-led scope and visual-system correction

Owner: N01
Status: in progress

Design:
- Lesson outcome: measure one fixed-seed heterogeneous `Batch` on CPU and GPU,
  then rebuild the same GPU-oriented Toolkit path from a molecule through
  `AtomicData`, `Batch`, neighbors, model outputs, selection, and recovery.
- Cell sequence: compact ALCHEMI orientation; equal-work CPU/GPU opening;
  four-capability course map; inspect one loaded molecule; convert and validate
  `AtomicData`; explain framework layers; build a heterogeneous `Batch`; inspect
  ownership and boundaries; recover molecules by position; run the model;
  compare one-molecule and batched results; attach outputs; bounded `Try it`.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `AtomicData.from_structure`, `AtomicData.model_validate`,
  `Batch.from_data_list`, `batch_idx`, `batch_ptr`,
  `num_nodes_per_graph`, `get_data`, `index_select`, `to_data_list`,
  `Batch.to`, `set_config`, `compute_neighbors`, `model(batch)`, and
  `Batch.add_key`.
- Molecules and model: a fixed-seed sample of 2,048 molecules from the pinned
  32-molecule collection for the opening; ethyne, phenol, and
  2,3-dimethylbutane for the heterogeneous-batch explanation; pinned AIMNet2
  energy and force outputs.
- Helper boundaries: helpers own checked data and checkpoint loading, repeated
  timing and synchronization, memory reporting, plot construction, labels,
  captions, and alternative text. Learner cells own scientific choices and the
  public Toolkit path.
- Expected runtime: the complete fresh GPU run should stay under one minute on
  the local RTX 4000; the 2,048-molecule CPU comparison is expected to take a
  few seconds per measured call. These are planning estimates until the clean
  run records current values.
- Validation plan: frozen-runtime check; notebook/helper Ruff; scoped tests;
  transformed full-namespace parse; CPU/GPU and serial/batch agreement;
  fresh-kernel execution; saved-output review; and rendered review of the
  ecosystem image, course map, framework map, callouts, batch strip, and timing
  plot.

Changed:
- Replaced the one-to-one capability column with four project capabilities:
  fundamentals, data management, using models, and simulation workflows.
- Shortened the course-part and capability labels. The vertical course spine
  remains linear; straight diagonal branches connect related parts to shared
  capabilities.
- Removed the duplicate CPU/GPU device lesson. Device movement remains visible
  only where the small example batch enters the GPU model path.
- Removed the isolated `skip_validation=True` section. A trusted-reader fast
  path belongs in the data-loading or profiling lesson if that lesson can show
  its complete invariant and measurement.
- Removed the full hosted MatterViz application and its custom viewer helpers.
  The notebook-native widget is pending a shared dependency.
- Replaced the generic advanced/homework heading rule with specific
  `Optional: <topic>` headings for complete mini-lessons.

Shared request:
- ID: N01-REQ-003
- For: integration / shared environment
- Need: pin `pymatviz` and `anywidget`, then verify
  `pymatviz.StructureWidget(structure=atoms)` with ASE `Atoms` in VS Code,
  Jupyter, and the published HTML path.
- Why: the official notebook-native widget supplies the focused MatterViz view.
  The current frozen environment has neither dependency, and the hosted
  MatterViz application includes unrelated page interface.
- Status: open

Next:
- Add the precise `REVISE — GRAPHIC` source marker, clear stale viewer output,
  run static and fresh-kernel validation, and review the compact course map at
  notebook width.

## 2026-08-12 - docs-led revision validated

Owner: N01
Status: ready for user review

Observed:
- The frozen Toolkit docs and maintained examples organize the product around
  batch-first data, data loading, model interfaces and composition, hooks,
  dynamics, GPU pipelines, training, and distributed execution. The revised
  course map keeps that linear order and groups it into four project
  capabilities.
- Final fresh execution completed 51/51 code cells with zero errors in 25.3
  seconds of nbconvert wall time on the local RTX 4000 SFF Ada runtime.
- The fixed 2,048-molecule / 20,796-atom opening measured a 2,267.2 ms CPU
  median and a 124.7 ms GPU median across three warm model calls. GPU throughput
  was 16,417 molecules/s, or 18.2x the CPU result for this workload.
- Peak PyTorch CUDA allocation was 1,291.4 MiB, 6.7% of the device memory.
  Maximum CPU/GPU differences were about `1e-6 eV` for energy and
  `2e-6 eV/Å` for forces, within the displayed tolerances.
- The opening timing plot and the molecule-ownership strip remain legible at
  notebook width. The course map still needs rendered VS Code review because a
  Mermaid renderer is unavailable in the frozen command environment.
- VS Code has no live kernel attached to this notebook. The stale hosted-viewer
  output was removed. Nearby prose now describes the opening conditionally
  until the user runs it with the frozen kernel.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- Ruff for Notebook 01 helpers/tests and the shared design checker: passed.
- `./scripts/v3-run pytest -q notebooks/01-atomicdata-batch/tests`: 45 passed;
  15 warnings come from pinned TorchScript and AIMNet code.
- Shared design checker: zero errors and zero warnings.
- Authoring skill validation: passed.
- Shared and installed-skill course-map templates: byte-identical.
- Final fresh notebook: `/tmp/n01-executed-final.ipynb`, 51/51 code cells,
  zero errors.
- Bridge changes since revision 1951: eight agent changes, zero user changes.
  The protected acknowledgments cell is unchanged.

User review required:
- Open `notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb` with
  `/tmp/alchemi-v3-runtime/venv/bin/python` and run all cells. Confirm the
  course spine stays vertical, the four capability boxes fit, and the angled
  branches remain easy to follow.
- Confirm the two callouts and both quantitative plots at teaching width.
- After N01-REQ-003 is resolved, confirm the native StructureWidget rotates,
  zooms, and publishes through the final HTML path.

## 2026-08-12 - CUDA-resident data path and batch packing visual

Owner: N01
Status: implemented and validated; rendered review remains

Changed:
- Kept CPU construction and execution inside the opening CPU/GPU benchmark.
- The main lesson now converts ASE `Atoms` directly with
  `AtomicData.from_atoms(..., device="cuda")`; later `AtomicData`, `Batch`,
  neighbor data, model parameters, outputs, selections, recovered records, and
  attached results remain on CUDA.
- Removed the later `Batch.to(device)` teaching step. Device fields are shown
  in the `AtomicData`, recovery, and output summaries.
- Reworked the course map into a vertical lesson spine plus a separately
  aligned vertical capability column with four quiet diagonal links.
- Changed the Highlight to a neutral warm surface and gave inline API names
  explicit colors for light and dark notebook themes.
- Replaced the low-resolution atom-dot strip with a 160-dpi packing diagram.
  It uses the actual `atomic_numbers`, `batch_idx`, and `batch_ptr` values to
  show element rows, coordinate slices, membership values, and boundaries.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- Ruff for Notebook 01 helpers/tests and the shared checker: passed.
- `./scripts/v3-run pytest -q notebooks/01-atomicdata-batch/tests`: 45 passed;
  15 warnings come from pinned TorchScript and AIMNet code.
- Shared design checker: zero errors and zero warnings.
- Authoring skill validation: passed.
- Impeccable detector on the changed callout, map, and visual helper: no issues.
- Fresh execution before the final label-wrap adjustment:
  `/tmp/n01-executed-cuda-layout.ipynb`; 51 code cells completed with zero
  execution errors. The final adjustment changes presentation text layout only.

User review required:
- Run the notebook with `/tmp/alchemi-v3-runtime/venv/bin/python` and confirm
  the packing-diagram labels remain sharp and fully contained at teaching width.
- Check the course map and Highlight in both light and dark VS Code themes.

## 2026-08-12 - local support namespace

Owner: N01
Status: implemented; full parse blocked by an active user draft cell

Changed:
- Replaced the learner-facing `lesson.*` namespace with `helpers.*`. The module
  is notebook-local support; Toolkit calls retain their public namespaces.
- Renamed the setup entry point from `helpers.start_lesson()` to
  `helpers.start_tutorial()`.
- Updated the shared authoring guide and installed skill to name local support
  by responsibility, such as `helpers`, `plotting`, or `benchmarking`.
- Preserved the user's revised GPU-first opening heading and all user-authored
  notebook text.

Validation:
- Ruff passed.
- The shared design checker reports zero errors and zero warnings.
- Scoped pytest reached 43 passed and 2 failed. Both failures come from a new
  user-authored code cell containing the unfinished expression `a =`; the cell
  was preserved.

## 2026-08-12 - data terminology ladder and contained callouts

Owner: N01
Status: implemented; rendered review remains

Changed:
- Added a vertical molecule or structure -> `AtomicData` -> `Batch` diagram.
  Its arrows are the public constructors taught in the notebook.
- Defined `AtomicData` as one atomic system stored as one graph and `Batch` as
  concatenated storage for several independent graphs. Learner prose now uses
  molecule, structure, atom row, and tensor slice wherever those terms are more
  direct.
- Replaced the recap phrase “track membership and boundaries” with concrete
  actions: inspect the packed layout, recover one molecule, select several, or
  unpack all.
- Finished both shared callout layouts with border-box containment, narrow-width
  wrapping, clipped outer overflow, compact radii, and separate API input/output
  rows. Synced the installed authoring-skill asset.
- Added the transferable terminology-ladder rule to the canonical guide and
  authoring skill.

Validation:
- Ruff passed for the Notebook 01 contract tests and shared design checker.
- Focused terminology, ladder, recovery, callout, and diagram tests: 4 passed.
- Shared design checker: zero errors and zero warnings.
- Authoring skill validation: passed; the shared and installed callout assets
  are byte-identical.
- Impeccable detector found no issues in the changed shared design files.
- Full Notebook 01 contract run: 23 passed, 2 failed. Both failures parse the
  preserved user-authored `a =` cell.

User review required:
- At normal and narrow notebook widths, confirm the Highlight and API card stay
  inside the Markdown column and all signatures wrap cleanly.
- Confirm the three ladder boxes read top to bottom and the constructor labels
  remain legible in light and dark themes.

## 2026-08-12 - shared ALCHEMI course banner

Owner: N01
Status: implemented; rendered review remains

Changed:
- Copied the supplied 2880 x 450 NVIDIA ALCHEMI banner unchanged to
  `shared/alchemi-banner-left.png`.
- Added the shared banner as the dedicated first Markdown cell in Part 01; the
  lesson title follows immediately.
- Added `shared/banner.md` as the exact reusable notebook snippet and documented
  it in the shared presentation README and canonical authoring guide.
- Bundled the same banner image and snippet with the installed ALCHEMI tutorial
  skill. The shared and skill copies have SHA-256
  `016f3840bb97e61a3950bd70e587305fe9477831db9763c3d081db0b8a5bbf19`.
- Extended the design checker and Part 01 tests to require the banner, full-width
  sizing, aspect-ratio preservation, alternative text, and exact source image.

Validation:
- Ruff passed for the changed tests and design checker.
- Focused banner, callout, and terminology tests: 3 passed.
- Shared design checker: zero errors and zero warnings.
- Authoring skill validation passed; shared and skill banner snippets are
  byte-identical.
- Impeccable layout detector found no issues in the changed shared files.
- The notebook-relative image path resolves to the checked 2880 x 450 PNG.

User review required:
- Confirm the banner fills the notebook content width, remains sharp, and keeps
  its 6.4:1 aspect ratio in VS Code and exported HTML.

## 2026-08-12 - explicit framework boundary and rounded map routes

Owner: N01
Status: implemented; rendered notebook review remains

Educator delta:
- Draft choice: the framework diagram labeled the Warp-to-Toolkit-Ops edge
  `many implementations`, which named a quantity without explaining the work.
- Learner need: see that selected Toolkit-Ops operations call low-level Warp
  kernels, while the PyTorch and optional JAX bindings expose framework-native
  arrays.
- Reusable rule: diagram edges name the actual action or data passed between
  components. Avoid quantity labels such as `many` when the mechanism is the
  lesson.
- Draft choice: the course spine had arrowheads while capability connectors
  ended as plain lines with sharp corners.
- Learner need: read one consistent direction system across the whole map.
- Reusable rule: every directed connector uses the same arrow geometry; elbow
  connectors use rounded joins and explicit routes when automatic layout makes
  direction or alignment inconsistent.

Changed:
- Reworded the framework diagram and adjacent explanation around selected
  Toolkit-Ops operations, low-level Warp kernels, and framework bindings.
- Added muted and NVIDIA-green arrow markers to the curriculum-map generator.
- Replaced sharp capability polylines with explicit rounded elbow routes and
  regenerated all eight part-specific SVG maps.
- Updated the canonical guide, installed authoring skill, skill checker, and
  Notebook 01 tests with the transferable wording and connector rules.

Validation:
- Runtime pin check: passed.
- Ruff: passed for the generator, Notebook 01 tests, and shared skill checker.
- Notebook 01 suite: 46 passed; 15 upstream Torch/AIMNet warnings remain.
- Authoring skill validation: passed.
- Shared design checker: zero errors and zero warnings.
- Shared and installed generators plus all eight SVG maps are byte-identical.
- Rendered `curriculum-map-01.svg` inspection: equal lesson boxes, consistent
  line icons, rounded routes, and arrowheads on the lesson spine and capability
  branches.

User review required:
- Confirm the rounded routes and arrowheads remain clear at normal notebook
  width and in the chosen light and dark notebook themes.

## 2026-08-12 - compact opening benchmark and validation provenance

Owner: N01
Status: implemented; saved-output refresh awaits a live VS Code kernel

Educator delta:
- The opening had expanded into several setup, timing, table, and plotting
  cells. It now uses one tested helper call to make the motivating performance
  result, then rebuilds every central Toolkit operation in small visible cells.
- The lesson repeated serial-versus-batch agreement after the opening. That
  second evaluation was removed; the rebuild now explains objects, layout,
  neighbors, and output levels.
- The active-region mask failure was local tutorial invention. Part 01 now
  explains automatic construction validation beside the field table. Raw
  dictionary validation is handed to Part 02's Reader/Dataset boundary.
- Reusable rule: establish a performance claim once, then teach the abstraction
  that produced it. Place validation examples at the documented public boundary
  that owns the data.

Changed:
- Replaced the opening sequence with `helpers.show_batched_speedup(...)` for a
  fixed 2,048-molecule sample. The plot compares GPU serial calls, one CPU
  batch, and one GPU batch for the same energy-and-force work.
- Moved sampling, model setup, synchronization, timing, agreement checks,
  memory accounting, and plotting into the tested notebook helper.
- Removed the later serial/batched evaluation and the four-cell atom-mask
  validation story.
- Kept the main lesson path on CUDA and removed stale CPU outputs from the
  learner-facing conversion and field table.
- Updated the canonical guide, installed authoring skill, checker, and scoped
  notebook contract tests with the general opening and validation rules.

Validation:
- Runtime pin check: passed.
- Helper smoke test on 16 molecules: passed on CUDA.
- Ruff: passed for helpers, tests, and the authoring checker.
- Notebook 01 suite: 46 passed; 15 upstream Torch/AIMNet warnings remain.
- Authoring checker: zero errors and zero warnings.
- Fresh-kernel execution in a temporary copy: 35 code cells completed with no
  errors. The RTX 4000 reference run measured 36.19 s for 2,048 GPU serial
  calls, 3.12 s for one CPU batch, and 0.267 s for one GPU batch. The GPU batch
  processed 7,675 molecules/s and reached 1,292 MiB peak allocated CUDA memory.
- Rendered opening plot review: labels, units, log response-time scale, colors,
  and bar annotations are legible at 1244 x 474 pixels.

Blocker:
- The notebook bridge reported no live VS Code kernel and waited indefinitely
  for kernel selection. Stale CPU outputs were cleared through the bridge. Run
  All with the frozen v3 kernel will save the reviewed CUDA outputs and opening
  plot into the notebook.

## 2026-08-12 - Batch interface exercise

Owner: N01
Status: implemented; saved outputs await a live VS Code kernel

Design:
- Outcome: read a `Batch` through its public size, field-level, and indexing
  interfaces, then select a valid sub-batch.
- Sequence: build the 32-molecule batch; inspect counts, edges, largest graph,
  and device; inspect `keys`; compare string, integer, and list indexing; change
  a bounded list of molecule indices and verify the selected layout.
- Visible APIs: `Batch.from_data_list`, `num_graphs`, `num_nodes`, `num_edges`,
  `max_num_nodes`, `device`, `keys`, bracket indexing,
  `num_nodes_per_graph`, and `batch_ptr`.
- Helpers: collection loading remains in `helpers`; all Batch operations stay
  visible.

- Runtime: tensor inspection only after the existing CUDA data conversion;
  expected below one second.
- Validation: run the notebook checker and scoped tests; execute the revised
  cells when a live frozen kernel is available.

Changed:
- Replaced the Pandas-driven atom-count filter with a `Batch` interface
  playground.
- Added compact inspection of graph, node, edge, largest-graph, device, and
  field-level properties.
- Added the documented string, integer, and list indexing forms.
- Kept the editable task bounded to three valid molecule indices and made the
  selected counts and boundaries its success signal.
- Added the complete pinned `Batch` public interface to
  `TOOLKIT_API_REFERENCE.md` and the product-centered exercise rule to
  `TUTORIAL_GUIDE.md`.
- Replaced the flat `Reuse` API list with a `Recap` that separates what the
  learner learned from how later notebooks carry these objects forward.
- Added that two-part recap and next-notebook link to the shared tutorial guide
  and authoring checker.

Validation:
- Runtime pin check: passed.
- Exact revised operations on CUDA: 32 graphs, 322 nodes, zero pre-neighbor
  edges, maximum 20 nodes; selection `[0, 23, 31]` produced counts
  `[4, 13, 20]` and boundaries `[0, 4, 17, 37]` on CUDA.
- Notebook 01 suite: 47 passed; 15 upstream Torch/AIMNet warnings remain.
- Authoring checker: zero errors and zero warnings.
- Ruff: passed for scoped tests.

## 2026-08-12 - quiet curriculum icons and connector routes

Owner: integration visual system
Status: implemented; user review required

Educator delta:
- The eight icons were visually heavy and read as generated decoration.
- Eight separately routed lesson-to-capability lines intertwined in the narrow
  channel between columns.
- Reusable rule: an orientation map needs one dominant reading path. Icons
  should support labels with the least geometry needed, and related lessons
  should join one local bracket before entering a shared capability.

Changed:
- Reduced lesson cards to one line and reduced icons to a 1.25 px shared
  stroke, smaller footprint, and simpler metaphors.
- Replaced eight capability routes with six direct routes. Hooks, BaseDynamics,
  and GPU pipelines join one local simulation-workflow bracket.
- Added Model Development as a separate capability for training and
  fine-tuning.
- Removed dashed connectors and all route crossings.
- Shortened the map from 1040 to 880 SVG units and regenerated all eight parts.
- Synchronized the generator and SVG assets with the installed authoring skill.

Validation:
- Ruff: passed for both generators, the checker, and scoped tests.
- All eight SVG files parse and match the installed skill assets.
- Visual detector: no findings.
- Part 01 map rendered and inspected at 1100 x 880.

User review required:
- Open `shared/curriculum-map-01.svg` at notebook width and confirm the quieter
  icons and simulation-workflow bracket feel appropriately restrained.

## 2026-08-12 - opening benchmark progress and complete CPU/GPU comparison

Owner: N01
Status: helper implemented and validated; notebook Markdown/output refresh pending

Educator delta:
- The old progress display counted three routes. Its first step contained 2,048
  individual GPU calls, so the bar appeared to stop at `1/3` before finishing.
- The opening now shows four independent progress rows. The CPU and GPU serial
  rows advance with the individual molecule calls; each batch row tracks its one
  model call.
- Each active row names the outputs, device, and call shape, for example
  `Evaluating energy + forces · GPU · Batch(2,048)`. Completed rows switch to
  `Evaluated`.
- The benchmark is now a complete two-by-two comparison: CPU and GPU each run
  2,048 individual calls and one `Batch` call over the same molecules.
- The grouped plot labels the two call shapes directly. Titles are
  `Evaluation time` and `Throughput`; axes use `Elapsed time [ms]` and
  `Throughput [molecules/s]`.

Measured locally:
- Hardware: Intel Core i9-14900 and NVIDIA RTX 4000 SFF Ada Generation.
- Full live helper runtime: 49 seconds, including all four timed routes after
  model/data setup and agreement checks.
- CPU serial: 17.262 s; GPU serial: 28.717 s; CPU batch: 3.244 s; GPU batch:
  0.325 s.
- GPU batch throughput: 6,292 molecules/s for the fixed 2,048-molecule sample.

Validation:
- Frozen runtime preflight: passed.
- Ruff: passed for the changed helper and scoped tests.
- Helper tests: 20 passed; 15 upstream Torch/AIMNet warnings remain.
- Complete Part 01 suite: 46 passed and one existing notebook-contract check
  failed because the current notebook no longer embeds
  `alchemi-toolkit-architecture.png`.
- 64-molecule smoke run: all four progress rows reached their exact totals.
- Full 2,048-molecule smoke run: completed with all four progress rows at their
  exact totals.
- Rendered grouped-plot check: CPU blue, GPU NVIDIA green, readable grouped
  labels, log-scaled timing panel, and square-bracket units.

Pending:
- The opening Markdown still describes three routes. Update it to four routes
  through the live notebook bridge, then run the notebook top to bottom and save
  the new progress output and plot.

## 2026-08-12 - compact interactive curriculum map

Owner: integration visual system
Status: implemented; user review required

Educator delta:
- The shared workflow rail made Hooks, BaseDynamics, and GPU pipelines look
  connected to one another.
- The course map occupied too much vertical space for repeated use.
- Reusable rule: give every lesson-to-capability relationship its own arrow.
  Embed linked SVG maps as objects so lesson navigation remains interactive.

Changed:
- Rebuilt the map at 960 x 570 with eight compact lesson rows and six capability
  cards.
- Removed the workflow rail. Hooks now has separate arrows to model input
  preparation and simulation workflow control.
- Added clickable links for the five available notebooks, hover and keyboard
  focus styling, and a static image fallback.
- Added `shared/curriculum-map.drawio` as an editable source generated from the
  same lesson and relationship data.
- Updated Part 01 to use the interactive SVG object through the notebook bridge.
- Updated the tutorial guide, shared documentation, Part 01 checks, and the
  installed authoring skill.

Checks deferred at the educator's request while visual edits accumulate.

User review required:
- Open the Part 01 course map at notebook width. Confirm that every arrow reads
  as a separate relationship and that Parts 01 through 05 open their notebooks.

## 2026-08-12 - opening benchmark progress detail

Owner: N01
Status: helper updated; visible output refresh blocked by live-kernel restart

Educator delta:
- The opening benchmark needs to show the CPU and GPU work as separate subtasks.
- Long one-molecule routes need visible molecule-count progress.
- Reusable rule: progress labels name the operation, workload shape, and device.
  Repeated structure work refreshes every 100 completions. One large batched
  model call stays one progress unit because it is one measured operation.

Changed:
- Added all four tasks before timing begins: CPU and GPU one-molecule calls,
  followed by CPU and GPU `Batch[2,048 molecules]` calls.
- Changed learner-facing descriptions to `Evaluating energy + forces · ... ·
  CPU/GPU`.
- Added an explicit Jupyter refresh after every 100 one-molecule evaluations.
- Updated the shared progress guidance and canonical tutorial guide.

Observed:
- The notebook bridge kernel stopped responding during a helper reload. An
  interrupt succeeded; the subsequent restart command did not return through
  the bridge. The helper and documentation edits are saved.

Next:
- Once the live kernel is responsive, rerun cells 3 through 6 and save the new
  four-task progress output and benchmark figure.

## 2026-08-12 - normalized capability column

Owner: integration visual system
Status: implemented; user review required

Educator delta:
- Capability-card gaps and internal text positions varied with each block.
- Reusable rule: a mixed-height card column uses one outer gap and derives card
  height from the number of body lines. Heading, body, line spacing, and bottom
  padding use the same offsets in every card.

Changed:
- Set every right-column gap to 24 px.
- Derived card heights as 54, 72, or 90 px for one, two, or three body lines.
- Aligned every heading at 18 px, first body line at 42 px, subsequent lines at
  18 px, and bottom padding at 12 px.
- Regenerated all eight SVGs and the editable draw.io source, then synchronized
  the installed authoring skill and its checker.

Checks deferred while visual edits accumulate. The Part 01 SVG was rendered for
the targeted spacing review.

User review required:
- Confirm the right column reads as one consistent stack at notebook width.

## 2026-08-12 - stable benchmark progress wording

Owner: N01
Status: implemented and rerun in the live notebook kernel

Educator delta:
- Repeating `Evaluating energy + forces` on every row forced horizontal
  scrolling.
- Changing completed rows from `Evaluating` to `Evaluated` made the labels
  inconsistent while the benchmark ran.
- Reusable rule: put shared work in one heading and keep progress-row labels
  stable. Let the spinner, bar, and count show task state.

Changed:
- Added the heading `Energy + forces · 2,048 molecules`.
- Shortened the four rows to CPU/GPU plus the exact call shape:
  `2,048 × Batch[1]` or `1 × Batch[2,048]`.
- Removed task-description changes at completion.
- Updated the canonical guide and shared Rich example.

Observed:
- The live benchmark cell completed successfully in about 48 seconds of kernel
  wall time and saved four outputs.
- Current result: 2,048 molecules, 20,796 atoms, and 1,305 MiB peak GPU memory.
- Measured calls: CPU serial 18.811 s, GPU serial 30.250 s, CPU batch 2.873 s,
  and GPU batch 0.139 s.

User review required:
- Confirm the four progress rows fit the notebook width without horizontal
  scrolling while a rerun is in progress.

## 2026-08-12 - plain-language benchmark call shapes

Owner: N01
Status: helper updated; output refresh pending

Educator delta:
- `1 × Batch[2,048]` introduces object notation before the opening result has
  explained `Batch`.
- Reusable rule: opening progress labels describe work in familiar scientific
  language. Introduce exact object notation when the lesson teaches the object.

Changed:
- Serial rows now say `2,048 single-molecule calls`.
- Batched rows now say `1 batch of 2,048 molecules`.
- Updated the shared Rich guidance with the same wording.

Next:
- Rerun the opening benchmark when the next output refresh is requested.

## 2026-08-12 - active-task spinner

Owner: N01
Status: helper updated; output refresh pending

Educator delta:
- Four simultaneous spinners implied that all benchmark routes were executing.
- Reusable rule: animate only active work. Pending tasks use a quiet mark and
  completed tasks use a check.

Changed:
- Added an `ActiveSpinnerColumn` to the notebook-local progress helper.
- Created the four benchmark tasks in a pending state, activated one immediately
  before its measured route, and marked it complete afterward.
- Updated the canonical guide and shared Rich guidance.

Next:
- Rerun the benchmark with the next requested output refresh.

## 2026-08-12 - separate elapsed timer and linear seconds figure

Owner: N01
Status: implemented and rerun in the live notebook kernel

Educator delta:
- Per-row elapsed columns repeated timing beside each progress task.
- The logarithmic millisecond plot used scientific tick labels and made the
  learner translate units.
- Reusable rule: report shared elapsed time once above related progress rows.
  Tutorial timing figures use decimal seconds on a linear scale and print the
  elapsed value above each bar.

Changed:
- Moved elapsed time to one live line above the four task rows.
- Kept the rows focused on device, call shape, progress bar, and completed work.
- Converted both timing plot helpers from milliseconds to seconds for display.
- Removed the logarithmic scale, disabled scientific tick notation, and added
  decimal second labels above each timing bar.
- Updated the canonical guide, shared progress guidance, and focused plot test.

Observed:
- The visible benchmark cell completed successfully in about 52 seconds of
  kernel wall time and was saved.
- Current measured call times are 15.2 s CPU serial, 24.9 s GPU serial, 2.4 s
  CPU batch, and 0.29 s GPU batch.
- The plot reports 2,048 molecules, 20,796 atoms, and 1,305 MiB peak GPU memory.

Checks:
- Ruff passed for the helper and focused test file.
- The focused device-plot test passed; 14 upstream TorchScript warnings remain.

User review required:
- During a rerun, confirm the overall elapsed timer stays above the rows and only
  the active route spins.
- Confirm the linear seconds plot remains readable at notebook width.

## 2026-08-12 - aligned progress counts

Owner: N01
Status: helper updated; output refresh pending

Educator delta:
- Counts such as `2,048 / 2,048` and `1 / 1` ended at different positions.
- Reusable rule: mixed-size task totals share one right-aligned count column,
  sized to its widest formatted value.

Changed:
- Added an aligned Rich count column with thousands separators.
- Right-aligned every value in a column sized by its widest count.
- Updated the canonical guide and shared progress guidance.

Next:
- Confirm alignment during the next benchmark rerun; no additional compute was
  launched for this display-only edit.

## 2026-08-12 - curriculum capability dependencies

Owner: integration visual system
Status: implemented; rendered notebook review required

Educator delta:
- The interactive object included a nested image fallback, which some notebook
  renderers displayed as a second map.
- The lesson and capability cards were wider than their labels required.
- Capability cards looked independent even though data management feeds model,
  simulation, and training work, and the fundamentals support every later path.
- Reusable rule: keep one interactive SVG embed; use compact fixed-width cards;
  show real capability dependencies with separate rounded arrows.

Changed:
- Removed the nested image fallback from the Part 01 map cell.
- Reduced lesson cards from 350 to 282 px and capability cards from 360 to
  300 px; normalized the capability column to 18 px gaps.
- Added capability routes from fundamentals into the downstream path, from data
  management into models, simulation, and model development, and from model and
  simulation capabilities into multi-GPU execution.
- Kept the separate Hooks routes to models and simulation workflows.
- Regenerated all eight SVG maps and the editable draw.io source.
- Updated the canonical guide, shared documentation, Part 01 checks, and the
  installed ALCHEMI tutorial skill.

User review required:
- Confirm Part 01 shows one map, the shorter cards fit their text, every arrow
  has a clear direction, and Parts 01–05 remain clickable.

Focused checks:
- Ruff passed for the generator and Part 01 notebook contract.
- Both course-map contract tests passed.
- Kept the educator's removal of the Part 01 ecosystem architecture image. The
  concise Toolkit, Toolkit-Ops, and NIM explanation and links remain.
- Removed the stale image requirement from the guide, notebook test, and
  installed authoring checker.
- The shared skill checker reports zero errors and zero warnings.

## 2026-08-12 — official Basic 01 Batch augmentation

Owner: current orchestration pass
Status: design approved by the user; technically validated draft in progress

Lesson brief:
- Outcome: convert real ASE molecules to `AtomicData`, inspect ownership,
  validation, and serialization, pack unequal systems into `Batch`, inspect
  boundaries and mapping behavior, select/recover systems, add level-aware
  properties, round-trip the current state, and prepare neighbors.
- Prior capability: Python, basic molecular structures, and ASE `Atoms`.
- Sequence: orientation and one clear batching question; one real molecule;
  field levels/shapes/units; three unequal molecules; `batch_idx` /
  `batch_ptr`; selection/recovery; `model_dump` / `model_validate`; clone,
  chemical hash, and mapping access; `exclude_keys`; `add_key` at system and
  node levels; `Batch.model_dump`; `to_data_list` → `from_data_list`
  round-trip; cutoff-based neighbor preparation; full-collection inspection;
  bounded selection task; recap.
- Visible public APIs: `AtomicData.from_atoms`,
  `AtomicData.from_structure` (documented optional path),
  `AtomicData.model_validate`, `AtomicData.model_dump`, `chemical_hash`,
  `clone`, dict-style access, `Batch.from_data_list`, `exclude_keys`,
  `num_graphs`, `num_nodes`, `num_edges`, `max_num_nodes`,
  `num_nodes_per_graph`, `batch_idx`, `batch_ptr`, `keys`, containment,
  `len`, iteration, bracket indexing, `get_data`, `index_select`,
  `to_data_list`, `add_key`, `Batch.model_dump`, and
  `compute_neighbors`.
- Helper boundary: checked NCI Atlas loading, shared plot styling, and the
  existing packed-ownership visual remain helper-owned. Model loading,
  inference, timing, and performance plots leave the learner path.
- Systems and outputs: real ethyne, phenol, and 2,3-dimethylbutane for the
  main 4/13/20-atom example; the 32-molecule collection for selection;
  system-level integer record IDs and node-level float weights as controlled
  custom properties; a demonstrative 4.5 Å cutoff for neighbor preparation,
  explicitly not a model contract.
- Device: choose CUDA when available and otherwise CPU, then keep the complete
  main path on that one device.
- Scope: `append` / `append_data`, memory-layout helpers, and preallocated
  buffers are referenced rather than executed. `put` / `defrag` remain
  excluded because Toolkit 0.2 copies only float32 fields on that path and can
  skip required integer fields.
- Expected learner time: 45–55 minutes, excluding orientation and rendered
  review discussion.
- Exercise: choose three distinct records and verify graph count, atom counts,
  boundaries, order, and device.
- Validation: pinned-runtime behavior probes; notebook schema and transformed
  namespace parse; focused contract and behavior tests; design checker; Ruff;
  fresh-kernel execution to `/tmp`; source-output cleanliness; diagnostics;
  and human cell-by-cell plus rendered review.

Official-source audit:
- Include: construction and system fields; property levels; dict access;
  chemical hash/equality; clone; `AtomicData` serialization; Batch
  construction, `exclude_keys`, sizes, boundaries, recovery, indexing,
  mapping behavior, level-aware `add_key`, Batch serialization, neighbors,
  and round trip.
- Reference: `append` / `append_data`, `.to()` / `.cpu()` / `.cuda()`,
  `contiguous`, `pin_memory`, and edge-level custom fields.
- Omit from learner execution: custom `AtomicData.add_node_property` because
  the pinned object exposes the attribute but does not register it in
  `node_properties` or `model_dump`; fixed-capacity `empty` / `put` /
  `defrag` / `zero`; distributed send/receive.
- Replace: the 2,048-molecule AIMNet2 benchmark and supplied-model section.
  Their APIs and performance claims belong to model and GPU-execution lessons.

Human-review rule:
- Every changed cell remains a technically validated draft.
- Required status text:
  `TECHNICALLY VALIDATED DRAFT — HUMAN CELL REVIEW REQUIRED`.

Implementation and validation:
- Reconciled the interrupted notebook against the approved canvas delta:
  83 cells / 43 code cells, all unique stable IDs, no saved outputs or
  execution counts.
- Added validation cells `f9571b25`, `37944cfd`, and `bf581a72`.
- Removed AIMNet2 inference and timing from the learner path. The notebook now
  ends its data lesson with registered custom fields, a Batch round trip, and
  explicit matrix-format neighbor preparation.
- Pinned `AtomicData.add_node_property` probe:
  the custom attribute exists, but the field is absent from
  `node_properties` and `model_dump(exclude_none=True)`. Installed source adds
  the key to `__node_keys__` while the public views read a different registry.
  The notebook therefore demonstrates registered `Batch.add_key` behavior and
  states the limitation plainly.
- Frozen runtime check passed at Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` and Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Focused Part 01 suite: 50 passed; 16 warnings are upstream TorchScript,
  AIMNet tensor-conversion, and unavailable-NVML warnings from retained helper
  behavior tests.
- Ruff lint passed for notebook, helpers, and tests.
- Ruff format check passed for the changed notebook and tests. The broader
  helper check still reports four pre-existing formatting differences inside
  the unused opening-benchmark helper; no behavior or learner path depends on
  those lines.
- Notebook schema, unique IDs, transformed namespace, and source-clean
  contracts passed.
- Fresh execution artifact:
  `/tmp/atomicdata-and-batch-draft-executed.ipynb`; 43/43 code cells executed
  with zero cell errors on the CPU fallback. Warp emitted expected unavailable
  CUDA-driver diagnostics in the sandbox.
- The shared design checker still reports seven legacy requirements for the
  removed opening benchmark, hard-coded CUDA, framework diagram, and old recap
  heading. Its global source is outside this implementation scope; the focused
  notebook contract encodes the approved replacement.

Status:
- `TECHNICALLY VALIDATED DRAFT — HUMAN CELL REVIEW REQUIRED`.
- Cell-by-cell wording review and rendered review at teaching width remain
  required. MatterViz remains conditional on approved dependencies and
  publication behavior.
## 2026-08-13 — official structure bridges and direct construction

Owner: N01
Status: in progress

Design:
- Lesson outcome: build familiar molecules with ASE, convert one molecule with
  `AtomicData.from_atoms`, construct the same required fields directly with
  `AtomicData(...)`, and pack three unequal molecules into a CUDA `Batch`.
- Cell sequence: ASE H2O/CH4/ethanol builders; inspect water; show the ASE and
  optional pymatgen entry points; convert water; inspect field levels; construct
  water from tensors; compare required fields; batch the three molecules;
  inspect boundaries, recovery, and custom fields; recap and gallery links.
- Toolkit APIs kept visible: `AtomicData.from_atoms`,
  `AtomicData.from_structure`, direct `AtomicData(...)`,
  `Batch.from_data_list`, `num_nodes_per_graph`, `batch_idx`, `batch_ptr`,
  `get_data`, `index_select`, `to_data_list`, and `add_key`.
- Structures and model: ASE G2 water, methane, and ethanol; no model in this
  data-structure lesson. pymatgen remains a short optional code mapping because
  it is not installed in the frozen environment.
- Helper boundaries: setup and presentation only. ASE construction, Toolkit
  conversion, direct tensor construction, batching, and inspection stay visible.
- Expected runtime: under 15 seconds on the course GPU.
- Validation plan: frozen-runtime API smoke already passed; after the edits
  accumulate, run notebook syntax and scoped tests, then a fresh-kernel pass and
  rendered review.

Changed:
- Replaced the custom NCI trio with the official ASE example's H2O, CH4, and
  CH3CH2OH builders.
- Kept `AtomicData.from_atoms(...)` as the main executable bridge, added the
  optional pymatgen `from_structure(...)` mapping, and replaced the reader
  validation detour with direct `AtomicData(...)` tensor construction.
- Updated field tables, batch counts, boundaries, labels, recovery examples,
  field-level diagrams, gallery links, and recap for the 3/5/9-atom batch.
- Added the general documented-bridge/direct-construction teaching rule to the
  canonical guide and installed authoring skill.

Checks:
- Frozen runtime check: passed.
- Ruff on notebook, helpers, and tests: passed.
- Scoped Part 01 tests: 26 passed; 15 warnings are from pinned TorchScript and
  AIMNet paths.

Next:
- Run a fresh kernel and inspect the rendered optional pymatgen snippet, field
  table, ladder diagram, batch layout, and gallery links during the next full
  notebook review.

## 2026-08-13 — validation and visual clarity pass

Owner: N01
Status: technically checked; rendered review pending

Changed:
- Reduced the Toolkit API callout to one signature with one input and one
  result description.
- Replaced the two local Mermaid teaching diagrams with notebook-local SVGs for
  stable rendering in VS Code and exports.
- Added a realistic conversion error: an atom mask is applied to coordinates
  and omitted from atomic numbers. The notebook shows the validation error,
  repairs the source data, and constructs the fragment again.
- Updated the canonical guide and authoring skill with general rules for simple
  API cards, diagram renderer checks, SVG fallback, and validation examples.

Checks:
- Frozen runtime check: passed.
- Validation example: raised the expected inconsistent atom-count error.
- SVG XML parse: passed for both new assets.
- Ruff: passed for helpers and tests.
- Scoped Part 01 tests: 26 passed; 15 warnings are from pinned TorchScript and
  AIMNet paths.
- No Mermaid or interactive SVG object remains in the notebook.
- The notebook Ruff pass found one protected ASE cell with an unused `Any`
  import and a typed `zip[...]` expression. The bridge cannot overwrite
  user-authored protected text, so two editor suggestions are outstanding.

Next:
- Review the API card, diagrams, and validation output at normal notebook width.
- Apply the two suggestions in the ASE construction cell, then rerun notebook
  Ruff and the next fresh-kernel pass.

## 2026-08-13 — restored opening batching result

Changed:
- Restored the one-cell CPU/GPU batching comparison near the start of Part 01.
- Sampled 2,048 molecules with seed 7 from six ASE G2 and S22 structures that
  span 3 to 30 atoms, ending with the adenine–thymine Watson–Crick complex.
- Kept the later water, methane, and ethanol trio for the smaller data-structure
  walkthrough.
- Enabled gradients inside the benchmark helper so AIMNet2 can compute forces
  after the notebook disables global gradients.
- Replaced numerical agreement checks with one warm-up call per measured route.

Observed:
- The sampled workload contained 25,209 atoms.
- On the NVIDIA RTX 4000 SFF Ada Generation, the saved run measured 0.31 s and
  6,634 molecules/s for one GPU batch, with 1,596 MiB peak GPU memory.
- The same plot retains CPU and GPU results for individual calls and one batch
  call on linear axes.

Next:
- Apply the two outstanding suggestions in the protected ASE construction cell
  before the next fresh-kernel run.

## 2026-08-13 - notebook-native MatterViz viewer

Changed:
- Added `pymatviz 0.18.0` and the pinned `matterviz-anywidget 0.4.0` frontend to
  the shared frozen environment.
- Bundled the MatterViz JavaScript, CSS, and MIT license under `shared/` and
  added runtime hash checks so the viewer opens without downloading frontend
  code during the lesson.
- Replaced the Part 01 viewer placeholder with a focused
  `pymatviz.StructureWidget` cell. ASE molecule coordinates pass through an XYZ
  string because nonperiodic ASE molecules have a zero cell.
- Kept XYZ serialization in the notebook helper while leaving the learner's
  `StructureWidget(...)` call visible.

Checks:
- Frozen runtime and bundled MatterViz asset hashes: passed.
- Ruff on the runtime check, Part 01 helpers, and Part 01 tests: passed.
- Scoped Part 01 tests: 27 passed. The existing 15 warnings come from pinned
  TorchScript and AIMNet paths.
- The new widget cell executed in the live VS Code kernel and saved a Jupyter
  widget view with the water structure.

Next:
- Human review of rotation, zoom, controls, notebook width, and exported HTML.
- Apply the two outstanding suggestions in the protected ASE construction cell
  before the next full fresh-kernel run.
