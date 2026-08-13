# N03 worklog — Model interfaces and composite potentials

No entries yet.

## 2026-08-11 20:18 EDT — design and runtime preflight

Owner: N03
Status: blocked

Observed:
- The assigned notebook directory was absent and has been created empty.
- The frozen runtime passes at Toolkit `8c2c307c1c0` and Toolkit-Ops `c1e23460859a`.
- The core selection is nine equilibrium AB/A/B graphs and 114 atoms.
- The notebook bridge is healthy, with no notebook open. Its cell insertion call cannot initialize a nonexistent `.ipynb` file.

Design:
- Lesson outcome: implement a Toolkit model wrapper, read its `ModelConfig`, and compose learned, charge-dependent Coulomb, and D3(BJ) energy terms.
- Proposed cell sequence: outcome and pipeline diagram; imports and runtime paths; checked NCI subset load; select and display the nine equilibrium records; build ASE structures; create `AtomicData` and one `Batch`; define the finite-system Coulomb wrapper; inspect `ModelConfig` and `NeighborConfig`; load AIMNet2 and D3(BJ); evaluate learned energy and predicted charges; evaluate Coulomb and D3 components; construct `PipelineStep`, `PipelineGroup`, and `PipelineModelWrapper`; calculate interaction energies; display a Pandas result table; plot the component accounting; run component-sum, official-calculator, serial/batch, and graph-order checks; optional ten-point curves.
- Toolkit APIs kept visible: `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `adapt_input`, `adapt_output`, `ModelOutputs`, `AtomicData.from_atoms`, `Batch.from_data_list`, `compute_neighbors`, `PipelineStep`, `PipelineGroup`, and `PipelineModelWrapper`.
- Molecules and model: phenol–N-methylacetamide, propyne–methyl azide, and ammonia–benzoate at `R/R_e = 1.0`; `aimnet2-wb97m-d3_0`, predicted-charge direct Coulomb, and pairwise D3(BJ) from checkpoint metadata.
- Helper boundaries: notebook cells keep every model, adapter, neighbor, and pipeline call visible; notebook-local helpers handle checked CSV loading, AB-A-B reduction, and comparison table assembly.
- Expected runtime: about 1–3 minutes for a first local RTX 4000 GPU run and 3–6 minutes on CPU; warm reruns should be shorter. H100 timing is outside this notebook-authoring check.
- Validation plan: JSON and transformed-AST checks; scoped helper tests; clean top-to-bottom namespace review; fresh-kernel execution; charge conservation; component closure; agreement with the official AIMNet2 calculator; serial/batch parity; graph-order invariance; plot style and NVIDIA-green review.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- equilibrium subset inspection through `./scripts/v3-run python`: 9 graphs, 114 atoms, expected fragments and charges.
- notebook bridge capability check: passed; notebook initialization failed because the file does not exist.

Blocker:
- The bridge cannot create a new notebook file. User approval is required before a one-time `nbformat` fallback creates the blank notebook; all subsequent `.ipynb` edits will use the live bridge.

Next:
- Create or open the blank notebook after user direction, then implement and validate the complete lesson.

## 2026-08-11 21:43 EDT - implementation and validation complete

Owner: N03
Status: ready for integration and user review

Changes:
- Created the blank notebook once with `nbformat` after the user directed work to continue. Every notebook content edit then used the live notebook bridge.
- Implemented the full 43-cell lesson in `notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb` with 31 code cells and one main action per code cell.
- Kept the central Toolkit interfaces visible: `AtomicData.from_atoms`, `Batch.from_data_list`, `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `adapt_input`, `adapt_output`, `ModelOutputs`, `PipelineStep`, `PipelineGroup`, and `PipelineModelWrapper`.
- Added a finite-system direct Coulomb wrapper driven by AIMNet2 predicted charges, an independent D3(BJ) component, pipeline composition, official `AIMNet2Calculator` comparison, route-equivalence checks, and the optional ten-point interaction curves.
- Added focused data and reduction helpers under `helpers/` and four tests under `tests/`.
- Used plain Pandas and Matplotlib, `shared/alchemi-dark.mplstyle`, NVIDIA green for the complete Toolkit result, and the shared Rich progress pattern for the model evaluations with visible wait time.

Runtime preparation:
- The shared runtime initially lacked `/tmp/alchemi-v3-runtime/dftd3/dftd3_parameters.pt`.
- Generated the Toolkit D3 parameter cache in the shared runtime. Its SHA256 is `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec`.
- The notebook verifies the NCI subset, AIMNet2 checkpoint, and D3 parameter cache checksums before numerical work.

Validation:
- `./scripts/v3-run ruff check notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb notebooks/03-model-interfaces-composition/helpers notebooks/03-model-interfaces-composition/tests`: passed.
- `./scripts/v3-run pytest notebooks/03-model-interfaces-composition/tests -q`: 4 passed.
- Notebook JSON validation and transformed-cell AST parsing: passed for all 43 cells.
- Fresh-kernel execution through the frozen environment: 31 of 31 code cells completed in order with zero errors on an NVIDIA RTX 4000 SFF Ada Generation GPU.
- HTML export: passed with accessible image alt text and no missing-alt warning.
- Component closure: `3.800691e-06 eV` maximum absolute error, limit `2.0e-05 eV`, passed.
- Official calculator comparison: energy `7.809019e-04 eV`, interaction energy `6.233875e-04 eV`, force component `2.145767e-06 eV/angstrom`, and predicted charge `1.713634e-07 e`; all passed their visible limits.
- Serial/batch comparison: energy `2.082834e-06 eV`, force component `3.397465e-06 eV/angstrom`, and charge `2.086163e-07 e`; all passed.
- Reversed graph-order comparison: energy `2.432631e-06 eV`, force component `1.639128e-06 eV/angstrom`, and charge `1.192093e-07 e`; all passed.
- Optional ten-point curve MAE against CCSD(T)/CBS: phenol/N-methylacetamide `0.30 kcal/mol`, propyne/methyl azide `0.32 kcal/mol`, and ammonia/benzoate `0.41 kcal/mol` for this selected slice.
- Reviewed both rendered figures for shared dark styling, readable labels, component identity, and NVIDIA-green complete-model encoding.

Tolerance note:
- The route-equivalence limits cover float32 GPU reduction ordering. Repeated local executions varied by a few float32 increments, so the visible limits are `5e-06 eV/angstrom` for force components and `3e-07 e` for serial/batch charges.

Shared request:
- `N03-REQ-001` - For integration. Need: make `v3-sync` prewarm and verify the configured D3 parameter cache, or document the required cache-generation step. Why: the first fresh N03 scientific run stopped because the configured D3 cache file was absent. Status: open.

User review required:
- Open the source notebook at normal notebook width and review the Mermaid flow, table density, two dark-background figures, and expert-session pacing.
- The source notebook remains clean of saved execution outputs. The successful executed notebook and HTML export are temporary validation artifacts under `/tmp`.

Next:
- Integration can consume the finished source notebook, helper module, tests, and this worklog. Resolve `N03-REQ-001` in the shared environment before release validation from a newly synchronized runtime.

## 2026-08-11 23:37 EDT - revised design after consolidated guidance

Owner: N03
Status: in progress

Observed:
- Reread the current global rules, all repository Markdown files, every notebook worklog, every owned file, and the complete live notebook through bridge revision 62.
- The notebook has no user-authored cell changes. The current active editor is the consolidated Part 1 notebook, so no notebook 03 cell is active.
- The current N03 D3 setup uses the 15 angstrom tapered periodic-box lesson setting. The finite NCI reference calculation uses the 95 bohr untapered pairwise D3(BJ) convention.
- The consolidated guidance also requires an early real Toolkit result, a compact model card, explicit capability levels, output shape/level/unit inspection, visible methodology choices, a transfer task, and partial-model labels as architectural ablations.

Revised design:
- Lesson outcome: inspect a supplied Toolkit adapter, implement one complete `BaseModelMixin` adapter, and compose dependent AIMNet2-to-Coulomb flow with an independent D3(BJ) contribution.
- Proposed cell sequence: title, outcomes, capability levels, and data-flow map; one import/setup cell; checked NCI selection; visible `AtomicData` and `Batch`; checked AIMNet2/D3 inputs; one early AIMNet2 result with requested-output shapes and units; custom direct-Coulomb adapter; compact model/config card; explicit component evaluation; equilibrium table and plot; pipeline construction; component, official-calculator, serial/batch, and graph-order checks; transfer task; advanced ten-point curves; results and scope.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `Batch.from_data_list`, `AIMNet2Wrapper.from_checkpoint`, `set_config`, `compute_neighbors`, `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `adapt_input`, `adapt_output`, `ModelOutputs`, `PipelineStep`, `PipelineGroup`, and `PipelineModelWrapper`.
- Molecules and model: the same three NCI Atlas systems with nine equilibrium AB/A/B graphs in the core lesson and 90 graphs in the advanced curves; `aimnet2-wb97m-d3_0`, predicted-charge direct Coulomb, and checkpoint-matched D3(BJ) at 95 bohr with no taper.
- Helper boundaries: checked data loading, ASE conversion, identity-aware AB-A-B reduction, and comparison assembly stay in `helpers/`; every Toolkit adapter, configuration, neighbor, model, and pipeline call stays in notebook cells.
- Expected runtime: about 20-40 seconds on the local RTX 4000 SFF Ada after the shared caches are warm; CPU execution may take several minutes. H100 timing remains a separate target-hardware check.
- Validation plan: frozen runtime check; scoped Ruff and pytest; notebook JSON, IPython-transformed parse, and complete namespace checks; source-contract checks for capability text, visible APIs, D3 methodology, and transfer task; fresh-kernel execution; charge, component, official-calculator, serial/batch, graph-order, equilibrium-value, and full-curve checks; HTML export; notebook-width plot review.

Next:
- Apply targeted guarded cell edits through the live bridge, update scoped tests, and run the complete validation again.

## 2026-08-11 23:50 EDT - consolidated-guidance rework complete

Owner: N03
Status: ready for integration and user review

Guidance review:
- Read the current global and user rules, every repository Markdown file, every notebook worklog, the complete owned notebook, and the consolidated Part 1 opening used as the current style reference.
- Reworked the lesson around the revised capability levels, early-result sequence, compact model card, output semantics, visible methodology choices, architectural-ablation labels, two reference lanes, transfer task, and user-review checklist.
- Corrected the finite-system D3(BJ) setup from the periodic-box lesson setting to the NCI reference convention: 95 bohr pair cutoff and zero smoothing.

Changes:
- Expanded the source to 47 cells with 33 code cells and kept one main action in each code cell.
- Moved the supplied AIMNet2 adapter and one real Toolkit result ahead of the custom adapter implementation.
- Added an output table covering shape, graph or atom level, unit, and observed value, with an explicit explanation of the order-free `active_outputs` set.
- Added a compact AIMNet2 model card and a separate three-component configuration table.
- Kept `AtomicData.from_atoms`, `Batch.from_data_list`, `AIMNet2Wrapper.from_checkpoint`, `set_config`, `compute_neighbors`, `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `adapt_input`, `adapt_output`, `ModelOutputs`, `PipelineStep`, `PipelineGroup`, and `PipelineModelWrapper` visible in the teaching path.
- Labeled partial models as architectural ablations and compared the complete endpoint with near-matched DFT-D3 and independent CCSD(T)/CBS values.
- Added a small reuse task before the exact `Advanced — if time permits / homework` section.
- Updated `helpers/composition.py` for the DFT-D3 reference column and expanded the scoped source/data tests to seven cases.
- Live bridge history reached revision 100. It reports 38 agent edits since revision 62 and a user change count of zero. The source notebook was saved through the bridge and remains clean of execution outputs.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python 3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run ruff check notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb notebooks/03-model-interfaces-composition/helpers notebooks/03-model-interfaces-composition/tests`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/03-model-interfaces-composition/tests`: 7 passed in 1.51 seconds.
- Notebook schema validation, IPython-transformed AST parsing, and clean-source assertions: passed for all 47 cells and 33 code cells.
- Fresh-kernel execution: 33 of 33 code cells completed in exact order with zero errors in 12.78 seconds on an NVIDIA RTX 4000 SFF Ada Generation GPU.
- HTML export: passed; the file is self-contained and contains the expected ablation and complete-result labels.
- Predicted graph-charge residual: `2.384186e-07 e`, limit `2e-06 e`, passed.
- Component closure: `1.833738e-06 eV`, limit `2e-05 eV`, passed.
- Official calculator comparison: absolute energy `7.828689e-04 eV`, interaction energy `6.212262e-04 eV`, force component `2.205372e-06 eV/angstrom`, and predicted charge `1.713634e-07 e`; all passed their visible limits.
- Serial/batch comparison: energy `4.315658e-06 eV`, force component `1.966953e-06 eV/angstrom`, and charge `2.607703e-07 e`; all passed.
- Reordered-graph comparison: energy `3.039837e-06 eV`, force component `1.847744e-06 eV/angstrom`, and charge `1.192093e-07 e`; all passed.
- Advanced ten-point MAE versus DFT-D3 / CCSD(T)/CBS: phenol/N-methylacetamide `0.29 / 0.31 kcal/mol`; propyne/methyl azide `0.34 / 0.32 kcal/mol`; ammonia/benzoate `0.47 / 0.41 kcal/mol`.
- Reviewed both rendered figures at notebook width. Labels, legend placement, hatches, markers, dark styling, and NVIDIA-green complete-result encoding are readable.
- VS Code reports 15 import-resolution warnings because Pylance is attached outside the frozen scratch environment. Frozen-environment imports, tests, and execution pass.

Shared request:
- `N03-REQ-001` remains open. Integration should prewarm and verify the configured D3 parameter cache or document the cache-generation step.

User review required:
- Open `notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb` at normal notebook width.
- Confirm the five stated outcomes match the intended session scope.
- Review the Mermaid flow, model-card density, equilibrium tables, two figures, transfer task, and expert-session pacing.
- Confirm the distinction between the DFT-D3 methodology lane and the independent CCSD(T)/CBS benchmark lane is clear.

Next:
- Integrate the finished source notebook, helper, tests, and worklog, then resolve `N03-REQ-001` before release validation from a newly synchronized runtime.

## 2026-08-12 18:16 EDT — canonical-guide revision brief

Owner: N03
Status: in progress

Observed:
- The live N03 notebook is saved and clean at bridge revision 0; it has 47
  cells, 33 code cells, no saved outputs, and no user-authored bridge changes.
  The active editor is N01, so no N03 cell is being edited concurrently.
- The current opening predates the shared banner, generated Part 03 course map,
  short Goal/Core concepts hierarchy, exact N01/N02 links, approved callouts,
  bounded Try it, and two-part Recap.
- The current 70-line setup cell exposes repository discovery, checksums,
  display encoding, plot setup, and runtime paths. Several later learner cells
  combine computation, table shaping, display, plotting, and interpretation.
- The current `DirectCoulombWrapper` has no `NeighborConfig` or
  `NeighborListFormat`, and the lesson does not compare the adapter with its
  native model before composition. It also constructs only the explicit
  pipeline, so independent `model_a + model_b` composition is not taught.
- Pinned Toolkit source at `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`
  confirms: `AIMNet2Wrapper.from_checkpoint(...)` enforces float32 and declares
  a full MATRIX neighbor list; `BaseModelMixin.set_config(...)` mutates runtime
  fields; `make_neighbor_hooks()` is the iterative-workflow entry point; bare
  pipeline steps auto-wire matching keys; and `PipelineStep(..., wire=...)` is
  needed only when producer and consumer keys differ.

Design:
- Lesson outcome: load one supported AIMNet2 checkpoint, inspect rather than
  assume its model contract, wrap one native finite-system Coulomb model through
  the complete `BaseModelMixin` boundary, prove native/wrapper energy-and-force
  parity, then distinguish independent additive composition from a dependent
  charge-to-energy pipeline with shared autograd.
- Prior and new capability: link Part 01 for structure -> `AtomicData` -> graph
  -> `Batch` and Part 02 for rebuilding batches from persisted records. Reuse
  those objects without reteaching them. The one new capability is treating
  models as inspectable, configurable, composable Toolkit interfaces.
- Cell and visual sequence: shared banner; short Goal/Core concepts; folded
  ALCHEMI ecosystem/product overview; compact prior-lesson orientation;
  generated Part 03 course map kept visible and unfolded; collapsed imports and
  tested local setup; checked NCI equilibrium triplets; one AB molecule ->
  `AtomicData` -> graph -> one-graph `Batch`; supported AIMNet2 constructor;
  model-contract inspection; `set_config`; one-shot `compute_neighbors`; first
  AIMNet2 energy/charge result; complete native Coulomb model and adapter;
  adapter config inspection; native/wrapper parity; independent `aimnet + d3`
  construction; dependent `PipelineStep`/`PipelineGroup`/
  `PipelineModelWrapper`; motivated neighbor adaptation and
  `make_neighbor_hooks`; composed result inspection; component closure and
  official-route checks; bounded Try it; Recap and Part 04 link.
- Visuals: the shared Part 03 course map answers where this lesson sits; one
  compact Mermaid flow answers how values move through the dependent pipeline;
  one component plot, only if the D3 prerequisite is available, answers how the
  complete energy is assembled. Every visual receives a preceding question and
  a one-sentence takeaway.
- Visible public APIs: `AtomicData.from_atoms`, `Batch.from_data_list`,
  `AIMNet2Wrapper.from_checkpoint`, `model_config`, `set_config`,
  `compute_neighbors`, `make_neighbor_hooks`, `BaseModelMixin`, `ModelConfig`,
  `NeighborConfig`, `NeighborListFormat`, `adapt_input`, `adapt_output`,
  `model_a + model_b`, `PipelineStep`, `PipelineGroup`, and
  `PipelineModelWrapper`.
- Helper boundary: notebook-local helpers own repository/data/checkpoint/D3
  discovery and checksums, ASE record loading, repeated setup, progress,
  table-row shaping, and plot/display plumbing. Learner cells retain scientific
  selections and public construction, configuration, execution, inspection,
  composition, and parity calls.
- Scientific system and scope: three NCI Atlas equilibrium complexes represented
  as nine finite gas-phase AB/A/B graphs (114 atoms total). The useful-result
  path starts with one AB graph. The complete endpoint uses
  `aimnet2-wb97m-d3_0`, predicted-charge direct Coulomb, and checkpoint-matched
  D3(BJ). This is an interface and composition example, not a model catalogue or
  broad accuracy claim.
- Outputs and semantics: energy `[B, 1]` in eV; predicted charges `[V]` in
  elementary-charge units; forces `[V, 3]` in eV/Å. Inspect model parameter and
  input dtype/device, required and optional inputs, active versus available
  outputs, periodic support, cutoff, neighbor format, and full-list convention
  before interpretation.
- Custom adapter: a native PyTorch Coulomb module consumes positions, predicted
  partial charges, graph membership, and a full COO neighbor list. Its wrapper
  declares `NeighborConfig(..., format=NeighborListFormat.COO,
  half_list=False)`, maps the native partial-charge and energy names, exposes
  autograd forces, and is compared with a direct native call before downstream
  use.
- Composition: `aimnet + d3` demonstrates independent additive groups. The
  dependent pipeline wires AIMNet2 `charges` to the adapter's differently named
  `partial_charges`, places AIMNet2 and Coulomb in
  `PipelineGroup(..., use_autograd=True)`, and keeps D3 in a direct-derivative
  group. Shared autograd differentiates the base-plus-Coulomb energy through the
  predicted charges; D3 contributes its direct forces separately.
- Expected runtime: static/data-only cells should stay below 10 seconds. A local
  warm GPU run was previously about 13 seconds for the 33-cell draft, but the
  revised fresh-kernel runtime is unmeasured. CPU execution may take minutes.
  H100 timing and profiling remain outside Part 03.
- Bounded Try it: change one requested-output set on the supplied adapter,
  rerun one graph, and verify the returned keys plus energy/charge shapes. The
  task does not alter the scientific model or shared runtime.
- Validation plan: frozen runtime preflight; scoped Ruff, Pytest, notebook JSON,
  `nbformat`, transformed full-namespace parse, and shared design checker;
  helper tests for pinned inputs and presentation shaping; CPU-safe native
  adapter energy/force parity; GPU-only component closure and maintained
  calculator agreement when the prerequisite is available; coordinator-run
  fresh-kernel execution and rendered review at normal teaching width.

Shared request:
- ID: N03-REQ-001
- For: integration / shared runtime
- Need: make `environment/prewarm_assets.py` generate and verify
  `$ALCHEMI_D3_PARAM_FILE` against
  `dispersion.generated_parameter_sha256`, or provide an equally deterministic
  synchronized-runtime preparation step.
- Why: `scripts/v3-sync` calls `prewarm_assets.py`, but that script currently
  resolves and verifies only the AIMNet checkpoint. The current local file
  `/tmp/alchemi-v3-runtime/dftd3/dftd3_parameters.pt` exists, is 1,808,183
  bytes, and has the pinned SHA-256
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`;
  the earlier worklog records that N03 generated it manually after a fresh run
  failed. Its presence therefore does not prove that a newly synchronized
  runtime can execute the D3 path.
- Status: open; refreshed with current evidence. N03 will not generate, replace,
  or patch around this shared runtime asset.

Next:
- Refactor notebook-local support and tests, then rebuild N03 through guarded
  notebook-cell edits. Run only scoped static/unit checks; leave D3 numerical,
  complete fresh-kernel, and rendered validation to the coordinator after
  N03-REQ-001 is resolved.

Coordinator style correction:
- Fold only the compact ALCHEMI ecosystem/product overview in a disclosure.
  Keep the Part 03 curriculum-map orientation sentence, direct reference to
  `../../shared/curriculum-map-03.svg`, and takeaway visible. Do not copy the
  SVG into N03 or freeze its markup. Verify disclosure/map hierarchy in the
  static HTML render and again in the coordinator's executed rendered review.

## 2026-08-12 19:42 EDT — public-object-first rebuild complete

Owner: N03
Status: source complete; fresh execution and rendered review deferred

Handoff and concurrency:
- Took over the 47-cell, 33-code-cell source with no saved outputs and the
  notebook-local helper/test/worklog changes listed in the prior handoff.
- Rechecked the live notebook immediately before rebuilding it. Bridge revision
  0 reported no user-authored or structural changes, and N01—not N03—was the
  active editor.
- Rebuilt in place through the notebook bridge. Source replacements used cell
  preconditions, and inserted cells used exact expected cell counts.

Learner design:
- The finished source has 81 cells and 63 code cells. More than half of the
  code cells are at most five lines; only the two complete taught classes
  exceed 20 lines.
- Added the shared banner, compact folded product overview, visible generated
  Part 03 curriculum map, exact N01/N02 links, short Goal/Core concepts,
  one approved Highlight, one approved Toolkit API callout, bounded Try it,
  and the two-part Recap.
- Starts with a real supported `AIMNet2Wrapper`, then directly inspects its
  available/active outputs, required/optional inputs, precision, device,
  periodic support, cutoff, neighbor format, and list convention.
- Uses `set_config(...)`, inspects `model_config.active_outputs`, executes the
  next public neighbor/model operations, and inspects returned key shapes,
  level, dtype, and device before introducing composition.
- The custom `DirectCoulombAdapter` exposes reusable `ModelConfig`,
  `NeighborConfig`, `NeighborListFormat`, required `partial_charges`, explicit
  Toolkit/native key mappings, and autograd force ownership. A direct native
  energy/force route is compared with the adapter before downstream use.
- Separates independent `aimnet + d3` composition from the charge-dependent
  `PipelineStep`/`PipelineGroup`/`PipelineModelWrapper` relationship. The only
  explicit wire is `charges -> partial_charges`, where names genuinely differ.
- Inspects both real pipeline structures and generated neighbor hooks. Explains
  why the small finite example uses one maximum-cutoff source, why the
  AIMNet2-plus-Coulomb group shares autograd, and why D3 remains in a direct
  derivative group.
- Removed the model-catalogue, large-curve, official-calculator, serial/batch,
  graph-order, and broad benchmark lanes from the core lesson. Retained only
  bounded native-adapter parity and one-graph component closure.

Owned changed paths:
- `notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb`
- `notebooks/03-model-interfaces-composition/helpers/composition.py`
- `notebooks/03-model-interfaces-composition/helpers/__init__.py`
- `notebooks/03-model-interfaces-composition/tests/test_composition.py`
- `worklog/03-model-interfaces-composition.md`

Validation completed:
- `./scripts/v3-run python -m py_compile
  notebooks/03-model-interfaces-composition/helpers/composition.py
  notebooks/03-model-interfaces-composition/tests/test_composition.py`:
  passed.
- Full code-cell namespace parse with Python `ast`: passed for 63 code cells
  across 81 cells.
- `nbformat.validate(...)` plus clean-output assertion: passed for 81 cells;
  no code cell has saved outputs.
- `./scripts/v3-run ruff check
  notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb
  notebooks/03-model-interfaces-composition/helpers
  notebooks/03-model-interfaces-composition/tests`: passed.
- `./scripts/v3-run python -m pytest
  notebooks/03-model-interfaces-composition/tests/test_composition.py -q`:
  19 passed in 4.01 seconds.
- Notebook bridge diagnostics and IDE lints: no warnings or errors.

Numerical and rendered status:
- No new numerical result is claimed for the rebuilt source. Native-adapter
  parity uses `atol=1e-6` for energy and force components; component closure
  uses a visible `2e-5 eV` limit. Both remain to be measured in the coordinator
  fresh-kernel run.
- Fresh-kernel execution, HTML export, Mermaid rendering, disclosure/map
  hierarchy, and normal-width learner review were intentionally not run while
  `N03-REQ-001` remains open.
- Deferred fresh execution:
  `./scripts/v3-run jupyter nbconvert --to notebook --execute
  notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb
  --output /tmp/model-interfaces-composition.executed.ipynb
  --ExecutePreprocessor.timeout=600`
- Deferred render:
  `./scripts/v3-run jupyter nbconvert --to html
  /tmp/model-interfaces-composition.executed.ipynb
  --output /tmp/model-interfaces-composition.html`

Shared request and coordinator ruling:
- `N03-REQ-001` remains open unchanged. N03 did not generate, replace, or
  patch around the shared D3 parameter asset.
- Coordinator ruling applied: direct exploration of real public Toolkit
  objects precedes custom code; every configuration change is followed by
  state/output inspection; contrived validation and catalogue material was
  removed.

Next:
- Integration resolves `N03-REQ-001`, then runs the two deferred commands,
  records current parity/closure values, and reviews the rendered hierarchy at
  normal notebook width.

## 2026-08-12 19:18 EDT — independent-review remediation

Owner: N03
Status: ready for independent re-review; one clean-runtime portability request
and one pixel-level review remain

Review findings and resolutions:
1. **Independent additive model was only constructed.** Resolved. The notebook
   now constructs `independent_model = aimnet + d3`, inspects its two direct
   groups, prepares the model's public neighbor hooks on a fresh batch, executes
   the real model, inspects output keys/shapes/levels/dtypes/devices, separately
   executes AIMNet2 and D3, and hard-asserts energy and force addition.
2. **Central tests were mainly token contracts.** Resolved. Added CPU-safe
   behavioral tests using the installed Toolkit APIs for:
   - direct native versus notebook `DirectCoulombAdapter` energy/force parity;
   - real `PipelineStep` charge wiring and `PipelineGroup` execution order;
   - shared-autograd force propagation and temporary derivative ownership;
   - `neighbor_adaptation="always"` cutoff filtering plus MATRIX-to-COO
     conversion; and
   - independent `model_a + model_b` topology and numeric addition with two
     lightweight real `BaseModelMixin` wrappers.
   Real AIMNet2+D3 addition and full closure remain notebook integration
   coverage because they require the locally cached model/D3 assets.
3. **Displayed closure was not enforced.** Resolved. The notebook now stops on
   predicted-charge, independent-energy, independent-force, and full
   component-closure failures before displaying result tables.
4. **Scientific input and charge contract were opaque.** Resolved. Before model
   interpretation, the notebook displays the verified selected record:
   HB375x10 system `1.041`, phenol–N-methylacetamide, neutral hydrogen bond,
   25 atoms, graph charge `0 e`, finite/nonperiodic. It attributes the original
   NCI Atlas project and official repository and states the data's CC BY 4.0
   license. Predicted atomic charges are reduced by graph and checked against
   the declared charge with a visible `2e-6 e` tolerance before Coulomb use.
5. **Setup hierarchy and ordering were inconsistent with N01.** Resolved. The
   import/setup cell is tagged `hide-input` with
   `jupyter.source_hidden=true`. The supported-adapter heading now precedes
   checkpoint construction and contract shaping. The taught adapter
   implementation is visible and split by responsibility; only its import
   plumbing remains collapsed.
6. **Folded overview and map fallback were noncanonical/incomplete.** Resolved.
   The disclosure summary is exactly `New to ALCHEMI Toolkit?`. The curriculum
   map remains visible and references the evolving
   `../../shared/curriculum-map-03.svg`; its object label and image fallback
   now cover the full path through multi-GPU execution and very large systems.
7. **Stale helper support remained exported.** Resolved. Removed
   `interaction_table`, `_numeric_array`, `CompleteModelAgreement`,
   `compare_complete_outputs`, related constants/imports/exports, and their
   obsolete tests after confirming no current N03 notebook use.
8. **Blanket D3 deferral was inaccurate.** Resolved. The current machine's
   `/tmp/alchemi-v3-runtime/dftd3/dftd3_parameters.pt` passes the pinned SHA-256
   check and supports local cached execution. `N03-REQ-001` is now described
   only as a newly synchronized clean-runtime portability blocker because
   prewarm does not create and verify that file. It does not block native
   adapter tests, design/schema checks, static HTML, or local cached fresh
   execution.

Local cached execution evidence:
- Runtime: Python 3.12.13; Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`; Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Fresh CPU namespace:
  `./scripts/v3-run jupyter nbconvert --to notebook --execute
  notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb
  --output /tmp/n03-review-executed.ipynb
  --ExecutePreprocessor.timeout=600`.
- Result: 84/84 code cells completed in 12.897 seconds; no error outputs.
  The environment exposed no usable CUDA driver, so the notebook correctly
  selected CPU. Inspected stderr contains Warp CUDA-driver discovery messages
  and one NVML warning; neither affected execution.
- Selected input: phenol–N-methylacetamide; neutral hydrogen bond; 25 atoms;
  declared graph charge `0 e`; finite/nonperiodic.
- Predicted-charge residual: `1.266599e-07 e` against `2e-6 e`; passed.
- Native/adapter parity: exactly `0.0 eV` energy and `0.0 eV/Å` maximum force
  component difference.
- Independent additive closure: `0.0 eV` energy and `8.766074e-08 eV/Å`
  maximum force component difference; both below `2e-5`.
- Full dependent-plus-independent component closure:
  `3.24925531e-06 eV` against `2e-5 eV`; passed. This independently reproduces
  the reviewer's rounded `3.2493e-06 eV` observation.

Static/rendered evidence:
- Executed HTML export succeeded at `/tmp/n03-review-executed.html`
  (432,854 bytes).
- Inspected HTML contains the closed canonical disclosure, external evolving
  map object, complete fallback alt text, and dependent Mermaid source with
  `charges -> partial_charges` and the shared-autograd group.
- Pixel-level notebook-width review remains open because Playwright is not
  installed in the frozen environment. This is not a scientific or execution
  blocker.

Clean-runtime request:
- `N03-REQ-001` remains open and unchanged in scope: make synchronized prewarm
  generate and verify the pinned D3 parameter table. Current local cached
  success is not evidence that a newly synchronized runtime is reproducible.

Changed paths for this remediation:
- `notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb`
- `notebooks/03-model-interfaces-composition/helpers/composition.py`
- `notebooks/03-model-interfaces-composition/helpers/__init__.py`
- `notebooks/03-model-interfaces-composition/tests/test_composition.py`
- `worklog/03-model-interfaces-composition.md`

Next:
- Return the owned changes and evidence for independent re-review.

Final validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed with the
  pinned Python, Toolkit, and Toolkit-Ops revisions listed above.
- `./scripts/v3-run ruff check
  notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb
  notebooks/03-model-interfaces-composition/helpers
  notebooks/03-model-interfaces-composition/tests`: passed.
- `./scripts/v3-run python -m pytest
  notebooks/03-model-interfaces-composition/tests -q`: 26 passed in
  7.48 seconds. The 15 inspected warnings are 14 pinned TorchScript
  deprecations plus one NVML-unavailable warning; there are no test failures.
- `./scripts/v3-run python
  /home/nfedik/.codex/skills/alchemi-tutorial-authoring/scripts/check_notebook_design.py
  notebooks/03-model-interfaces-composition/model-interfaces-composition.ipynb
  --part 03`: zero errors and zero warnings.
- Notebook `nbformat` validation, transformed full-namespace AST parse, and
  clean-source assertion: passed for 104 cells, 84 code cells, and no saved
  source outputs.
- Scoped `py_compile`: passed for `helpers/composition.py`,
  `helpers/__init__.py`, and `tests/test_composition.py`.

## 2026-08-12 — final adapter-visibility correction

The complete `DirectCoulombAdapter` contract is now taught visibly rather than
hidden as setup. It is split into small progressive code cells for
`ModelConfig`, native-input mapping, native-output/force mapping, the required
no-embedding interface, and the final wrapper. This preserves the full
implementation without duplicating it in Markdown. Cell 36 now contains only
imports and remains collapsed with `jupyter.source_hidden=true` and
`hide-input`; taught implementation cells 37, 38, 39, 40, and 41 are visible
by default.

Focused verification:
- Adapter visibility plus native energy/force parity:
  `2 passed, 24 deselected` (15 inspected pinned-environment warnings).
- Ruff on the changed notebook and focused contract test: passed.
- N03 design checker: zero errors and zero warnings.
- `nbformat` schema plus explicit metadata assertions: passed for 109 cells;
  required taught contract cells 37, 38, 39, and 41 are visible and setup cell
  36 remains hidden. Cell 40 visibly supplies the required no-embedding
  methods.
- IDE diagnostics for the focused contract test: none.

No full notebook execution was rerun because this correction only reorganizes
the already parity-tested adapter implementation without changing its
behavior.

## 2026-08-12 — N03-REQ-001 closed by integration

Owner: integration / shared runtime
Status: closed

Root cause and resolution:
- `scripts/v3-sync` configured `ALCHEMI_D3_PARAM_FILE` but its prewarm script
  previously handled only AIMNet. The local N03 success therefore depended on
  a manually generated file left under `/tmp/alchemi-v3-runtime`.
- Integration now invokes Toolkit's public
  `extract_dftd3_parameters(...)` / `save_dftd3_parameters(...)` sequence when
  the configured D3 file is absent, verifies the generated bytes before atomic
  publication, and rejects an existing mismatch without overwriting it.
- `environment/check_runtime.py` now verifies the actual configured file
  against `dispersion.generated_parameter_sha256`, rather than checking only
  that the environment path lies under the runtime root.
- The existing AIMNet preparation remains unchanged. `scripts/v3-sync`,
  `scripts/v3-run`, and `environment/runtime-pins.toml` required no edits.

Source evidence:
- Pinned Toolkit
  [`nvalchemi/models/dftd3.py`](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18/nvalchemi/models/dftd3.py)
  shows that `DFTD3ModelWrapper` uses `load_dftd3_params(...)`, whose
  missing-file path calls those public extraction/save helpers. Toolkit checks
  the official Bonn archive against MD5
  `a76c752e587422c239c99109547516d2` before parsing.
- Pinned Toolkit-Ops
  [`D3Parameters`](https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/c1e23460859a784e1d78043bcd1c8af0d1095fa2/nvalchemiops/torch/interactions/dispersion/_dftd3.py)
  owns the D3 tensor shape/type contract.
- The official NVIDIA Toolkit-Ops 0.4.1
  [DFT-D3 molecule example](https://nvidia.github.io/nvalchemi-toolkit-ops/main/examples/dispersion/01_dftd3_molecule.html)
  uses the same extraction/save mechanism and documents the roughly 500 KB
  first-use download.

Clean-path and identity evidence:
- The exact D3-only production harness recorded in `worklog/integration.md`
  created
  `/tmp/alchemi-v3-d3-final-qhf5weu9/dftd3/dftd3_parameters.pt`
  without reading, deleting, or replacing the existing shared cache.
- Generated size: `1,808,183` bytes.
- Generated SHA-256:
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.
- The prewarm check and the runtime-check helper both accepted the clean file.

Validation:
- `./scripts/v3-sync`: passed; 201 locked packages checked, AIMNet verified, D3
  verified.
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider
  environment/test_runtime_assets.py`: 6 passed.
- `./scripts/v3-run ruff check environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python -m py_compile environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.

Closure:
- `N03-REQ-001` is closed. A newly synchronized runtime now prepares the exact
  D3 table that N03 opens with `auto_download=False`.
- Remaining external requirement: the first generation in a clean runtime
  needs the official Bonn endpoint. Download failure, archive drift, serializer
  drift, missing data, or a SHA-256 mismatch stops synchronization loudly.
- A complete duplicate clean runtime was not installed because that would
  redownload the large locked Torch/CUDA environment; the D3-only production
  path was tested from a clean custom root and the normal `v3-sync` path was
  tested against the existing runtime.

## 2026-08-12 19:33 EDT — N03-REQ-001 reopened after review

Owner: integration / shared runtime
Status: reopened; changes required

Reason:
- Independent review identified a no-clobber violation between the initial
  destination existence check and `os.replace(...)`. A concurrent publisher
  could create a mismatched destination during generation and have that file
  overwritten by the staged D3 table.
- The prior clean-path evidence established deterministic generation and
  identity, but it did not establish preservation of a concurrent winner.

Closure gate:
- Integration must use an atomic same-filesystem no-clobber publication
  primitive, validate any winning destination, preserve mismatched bytes, and
  add deterministic concurrency, partial-generation, visibility, cleanup, and
  AIMNet regression tests before this request can close again.

## 2026-08-12 19:34 EDT — N03-REQ-001 reclosed after remediation

Owner: integration / shared runtime
Status: closed

Resolution evidence:
- Verified staging now publishes through same-filesystem `os.link(...)`, whose
  Linux `EEXIST` behavior cannot replace a concurrent destination.
- A matching winner is accepted without replacing its inode. A mismatched
  winner raises and remains byte-for-byte intact.
- Direct reproduction reported
  `concurrent_mismatch_preserved=True`, retained
  `b'concurrent mismatched winner'`, and left zero staging entries.
- The real hard-link path passed on
  `Linux 6.6.87.2-microsoft-standard-WSL2` with `/tmp` reported as
  `ext2/ext3`. A forced `EOPNOTSUPP` path failed clearly without publication.
- Two clean production generations under
  `/tmp/alchemi-v3-d3-determinism-1-k9gcr34v` and
  `/tmp/alchemi-v3-d3-determinism-2-ee5mw9td` each produced the pinned
  `1,808,183`-byte table with SHA-256
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`
  and zero remaining staging entries.
- `environment/README.md` now keeps a custom runtime root persistent with
  `export ALCHEMI_V3_RUNTIME_ROOT=...` for both synchronization and later
  `v3-run` commands.

Validation:
- `./scripts/v3-run pytest -q -p no:cacheprovider
  environment/test_runtime_assets.py`: 14 passed.
- `./scripts/v3-run ruff check environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python -m py_compile environment/prewarm_assets.py
  environment/check_runtime.py environment/test_runtime_assets.py`: passed.
- `./scripts/v3-run python environment/check_runtime.py`: passed.
- `./scripts/v3-sync`: passed; AIMNet and D3 identities verified.

Closure:
- `N03-REQ-001` is closed again. N03 can rely on a complete pinned D3 asset
  without a concurrent publisher being overwritten.
- A filesystem without hard-link support now fails explicitly. The first clean
  generation still requires the official fixed-MD5 Bonn archive.

## 2026-08-12 20:44 EDT — N01 recap-copy parity

- Re-read the live N01 `Where NVIDIA ALCHEMI fits` cell, then copied its complete
  body—including links and the ecosystem image—without editing N01.
- The copied body is the only content inside a disclosure headed exactly
  `Where NVIDIA ALCHEMI fits (recap)`. The Part 03 orientation, map, and takeaway
  remain visible and unchanged.
- The focused contract locates N01 by its Markdown heading with `nbformat`,
  removes only the N01 heading or recap disclosure wrapper, collapses incidental
  whitespace, and compares the remaining content.
- Focused parity contracts: 3 passed across N03–N05. Scoped Ruff and N03 schema,
  cell-ID, metadata, clean-output, and source-parity validation passed.
- The design checker was run and reported 5 existing-policy errors: its obsolete
  required summary plus four current map-embed rules. No approved map content was
  changed. CUDA and full notebook execution were not run.

## 2026-08-13 — focused deep-dive rebuild brief

Owner: N03
Status: implementation complete; human cell review required

Outcome and prior knowledge:
- Preserve Part 03 as the model-interface lesson. Reuse Part 01's
  structure-to-`AtomicData`-to-`Batch` path and Part 02's data-loading context
  without reteaching either lesson.
- Begin with a deterministic quadratic PyTorch model and its Toolkit wrapper.
  Prove native/wrapper energy-and-force parity, then execute an independent
  additive composition with the public `+` operator.
- Transfer the same interface to one real NCI Atlas AB/A/B triplet. Load the
  supported AIMNet2 checkpoint, inspect and change `model_config`, prepare its
  declared neighbors, wrap a finite-system Coulomb term, and compose AIMNet2,
  Coulomb, and D3(BJ) with explicit charge wiring and derivative ownership.

Cell and visual sequence:
- Shared banner, Part 03 title, short prior-lesson links, and the generated Part
  03 course map.
- Collapsed imports and tested local setup.
- Synthetic two-graph `Batch`; native quadratic model; one complete wrapper;
  config and input/output mapping; native parity; independent `model_a +
  model_b` closure.
- Checked NCI Atlas phenol/N-methylacetamide AB/A/B records; public
  `AtomicData.from_atoms(...)` and `Batch.from_data_list(...)`; supported
  AIMNet2 loading; live model-contract and output inspection; graph-charge
  closure.
- Native finite Coulomb model and adapter; `NeighborConfig` plus one-shot
  `compute_neighbors(...)`; native/wrapper parity without private assignment.
- Verified D3(BJ) setup; `PipelineStep` charge mapping; shared-autograd
  AIMNet2/Coulomb group; independent direct-force D3 group; public
  `make_neighbor_hooks()` inspection; full execution and component closure.
- One compact composition diagram and one interaction-energy component plot.
  The interpretation calls the bars architectural model terms, not a quantum
  energy decomposition, and makes no condensed-phase claim.
- Bounded output-selection exercise, recap, and links to Hooks, BaseDynamics,
  training, supported models, and the official composition examples.

Visible public APIs:
- `AtomicData.from_atoms`, `Batch.from_data_list`, `Batch.add_key`,
  `AIMNet2Wrapper.from_checkpoint`, `model_config`, `set_config`,
  `ModelConfig`, `NeighborConfig`, `NeighborListFormat`, `BaseModelMixin`,
  `adapt_input`, `adapt_output`, `compute_neighbors`, `model_a + model_b`,
  `PipelineStep`, `PipelineGroup`, `PipelineModelWrapper`, and
  `make_neighbor_hooks`.

Helper boundary and assets:
- Notebook-local helpers own immutable data/model/D3 identity checks, record
  parsing, repeated batch construction, compact inspection tables, AB-A-B
  reduction, and plot styling.
- Learner cells own every model configuration, wrapper, neighbor, composition,
  execution, parity, and closure call.
- The D3 tensor table remains a synchronized runtime asset. The notebook opens
  it with `auto_download=False`; no Bonn-derived archive or generated tensor
  cache is added beneath the owned lesson directory.

Scientific and execution contract:
- Synthetic energy is `[B, 1]` in arbitrary tutorial units and forces are
  `[V, 3]`; it demonstrates interface identity only.
- Real energy is `[B, 1]` in eV, forces are `[V, 3]` in eV/Å, and predicted
  charges are `[V]` in elementary-charge units. Each AB/A/B graph is finite and
  neutral.
- The real plot reports AB-A-B interaction energies at one fixed NCI geometry.
  Its CCSD(T)/CBS point is context for that geometry, not an accuracy or
  transferability result.
- Expected warm runtime is under one minute on the available local device;
  CPU execution may emit pinned Warp CUDA-discovery warnings.

Validation plan:
- Add behavioral and source-contract tests first, observe the old 109-cell
  draft fail the new sequence/private-API requirements, then rebuild.
- Run scoped Ruff, Pytest, Python compilation, notebook schema and transformed
  namespace checks, the design checker, frozen-runtime preflight, a serialized
  fresh-kernel execution, and executed-output assertions.
- Export HTML and review headings, source visibility, diagram, tables, plot,
  links, warnings, widths, alt text, and prose in two passes:
  API/pedagogy/science, then rendered learner/no-AI-slop.

Sources checked:
- Toolkit 0.2 supported-model table, model-wrapping guide, AIMNet2 wrapper
  reference, additive LJ+Ewald example, and AIMNet2+Ewald pipeline example.
- Frozen Toolkit source at
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` for `BaseModelMixin`,
  `AIMNet2Wrapper.from_checkpoint`, `Batch.add_key`, `PipelineStep`, additive
  composition, and pipeline neighbor hooks.
- The public examples and pinned source agree that explicit `wire` mappings
  are for mismatched names, shared autograd is required for charge-dependent
  energy, and composed iterative workflows should register
  `make_neighbor_hooks()`.

## 2026-08-13 — focused rebuild completion and review

Status: draft; implementation and automated review are complete, but the
notebook still requires human cell review.

Implementation result:
- Rebuilt the lesson as 62 source-clean cells. The sequence is synthetic
  wrapper and native parity first, then the finite NCI Atlas AB/A/B example,
  supported AIMNet2 loading and inspection, finite Coulomb adaptation, and
  AIMNet2 + Coulomb + D3(BJ) pipeline composition.
- Kept the Core-derived AIMNet2 loading cell byte-for-byte stable while moving
  the surrounding lesson into the focused Part 03 sequence.
- Kept product calls visible and moved only immutable checks, repeated batch
  construction, compact tables, AB-A-B reduction, and plot styling into local
  helpers.
- Removed private assignment paths. Custom fields use `Batch.add_key(...)`,
  mismatched pipeline names use an explicit `PipelineStep.wire`, and iterative
  neighbor preparation is exposed through `make_neighbor_hooks()`.
- The D3 wrapper uses the verified runtime parameter file with
  `auto_download=False`. No Bonn archive, generated D3 cache, or model tensor
  file exists under the lesson directory.

API, pedagogy, and science revision pass:
- Confirmed the visible path from `ModelConfig` and adapter mappings through
  neighbor requirements, active outputs, native-wrapper parity, dependent and
  independent groups, charge closure, and component closure.
- Confirmed that the synthetic quadratic energy is presented only as an
  interface check.
- Confirmed that the interaction bars are described as architectural model
  terms at one fixed geometry, not a quantum energy decomposition or evidence
  of accuracy, transferability, a validated interaction curve, or
  condensed-phase behavior.
- Confirmed direct links back to Core and forward to Hooks, BaseDynamics, and
  training, plus official supported-model, wrapping, composition, and D3
  references.

Rendered learner and no-AI-slop revision pass:
- Exported the fresh execution to learner HTML and rendered it in Edge at
  1440 px width. The course map, live Mermaid composition diagram, tables, and
  bounded interaction plot render without horizontal clipping.
- The hidden setup cell and its host CUDA-discovery stderr are absent from the
  learner export. No traceback, NVML warning, tensor-conversion warning, or
  Warp warning remains visible.
- Reviewed the full rendered page for heading rhythm, cell size, repeated
  explanation, callout usefulness, plot labels, alt text, link placement, and
  recap length. The final prose is direct and contains no curriculum IDs,
  decorative em dashes, or generic tutorial filler.

Final validation evidence:
- `./scripts/v3-run python -m pytest
  notebooks/03-model-interfaces-composition/tests -q`: 15 passed. The warning
  summary contains upstream TorchScript deprecations and the host's missing
  NVML library during a direct test; neither appears in the learner render.
- Scoped `ruff check`: passed.
- Scoped `ruff format --check`: passed after formatting the owned helper and
  test modules.
- `check_notebook_design.py ... --part 03`: 0 errors.
- `nbformat.validate`: passed with 62 unique cell IDs, no source outputs, and
  no source execution counts.
- `environment/check_runtime.py`: passed for Python 3.12.13, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Serialized fresh-kernel execution and HTML export: passed. Static HTML
  inspection found no warning or error strings, and the final browser render
  confirmed the diagram and plot.

Changed-cell review index (zero-based):
- Cells 0-3: banner, lesson goal, hidden setup, ecosystem recap, and course
  orientation.
- Cells 4-10: synthetic batch, native quadratic model, Toolkit wrapper,
  `model_config`, native energy/force parity, and public additive composition.
- Cells 11-27: supported AIMNet2 loading, NCI Atlas AB/A/B batch, live model
  and output contracts, neighbor preparation, output selection, and charge
  closure.
- Cells 28-39: finite Coulomb native model and adapter, input/output mapping,
  public field registration, and native-wrapper parity.
- Cells 40-50: D3(BJ) construction, explicit charge wiring, derivative groups,
  pipeline inspection, neighbor hooks, and complete-model execution.
- Cells 51-58: component closure, AB-A-B reduction, bounded component plot,
  and scientific interpretation.
- Cells 59-61: output-selection exercise, recap, and next-lesson links.

Human-review gate:
- Review the seven ranges above in order, with extra attention to the complete
  wrapper cells 7 and 30, the pipeline definition in cells 40-45, and the
  scientific interpretation in cells 51-58.
- Keep status as draft until that cell review is recorded.
