# N02 worklog — Zarr data loading

No entries yet.

## 2026-08-11 20:16 EDT — lesson design

Owner: N02
Status: in progress

Observed:
- The owned notebook directory does not exist yet; the shared runtime check passes at the pinned Toolkit and Toolkit-Ops commits.
- The lesson uses the 32 distinct neutral H/C/N/O molecules in `data/nci_atlas/ir-molecule-library.extxyz` (322 atoms total).

Design:
- Lesson outcome: write Toolkit graphs to Zarr, inspect and retrieve records, stream graph batches, and connect an extxyz source through the public `Reader` interface.
- Proposed cell sequence: outcome and data-flow map; imports and paths; load the source molecules; create `AtomicData` records with stable integer IDs; write one Zarr store; inspect one raw record; build a `Dataset`; retrieve selected records; build and iterate a `DataLoader`; display batch summaries; plot batch atom counts; define an ASE/extxyz `Reader`; run that reader through `Dataset` and `DataLoader`; round-trip checks; advanced compression/loading notes; reuse exercise and API recap.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `add_system_property`, `Batch.from_data_list`, `AtomicDataZarrWriter.write`, `AtomicDataZarrReader.read` and `read_many`, `Reader`, `Dataset`, and `DataLoader`.
- Molecules and model: the shared 32-molecule NCI Atlas collection; no model call is needed because this lesson changes the storage and loading path while holding the chemistry fixed.
- Helper boundaries: core Toolkit calls and the small `ExtXYZReader` class stay in notebook cells; only compact display-table construction is local support code.
- Expected runtime: about 5–15 seconds on CPU for 32 small records, including the fresh Zarr write and read; GPU is unused in the core lesson.
- Validation plan: JSON and per-cell AST checks, a scoped notebook contract test, fresh-kernel top-to-bottom execution, exact 32-record/322-atom checks, selected-record identity checks, and position/atomic-number equality after Zarr and extxyz-reader round trips.

Next:
- Create the owned notebook directory, implement the complete notebook through the live bridge, then run the scoped checks.

## 2026-08-11 20:25 EDT — notebook implemented and executed

Owner: N02
Status: ready for integration

Observed:
- The complete lesson has 36 cells, including 21 code cells. The core run stays on CPU and finishes within the 12-minute teaching budget.
- The bridge could edit an existing notebook but could not create the missing file. A minimal empty notebook shell was created once; all 36 cells and every later notebook change used guarded live-bridge edits.
- Zarr stores 32 graphs and 322 atoms with `record_id` as a system field. Four eight-graph loader batches contain 46, 66, 91, and 119 atoms.
- The custom `ExtXYZReader` reaches the same `Dataset` and `DataLoader` path as the Zarr reader.

Changed:
- `notebooks/02-zarr-data-loading/zarr-data-loading.ipynb`
- `notebooks/02-zarr-data-loading/tests/test_notebook_contract.py`
- `worklog/02-zarr-data-loading.md`

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python 3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/02-zarr-data-loading/tests`: 3 passed.
- `./scripts/v3-run ruff check notebooks/02-zarr-data-loading`: passed.
- Fresh-kernel `./scripts/v3-run jupyter nbconvert --execute ...`: 21/21 code cells executed in order with zero errors; total command wall time was 9.6 seconds on CPU.
- Runtime checks passed for 32 records, 322 atoms, selected-record order and IDs, exact atomic-number and position equality, and total atoms across emitted batches.
- Plot review: `shared/alchemi-dark.mplstyle` loaded; the primary batch bars use NVIDIA green; labels and units are legible at notebook width.
- Progress review: no cell waits long enough to need a Rich progress display.
- VS Code diagnostics show unresolved-import warnings because Pylance is not using the frozen scratch environment. Imports and execution pass inside `v3-run`.

Shared request:
- ID: N02-REQ-001
- For: integration
- Need: add notebook 02 to final root navigation and the built tutorial bundle after the notebook wave lands.
- Why: notebook authors cannot edit root or build files during parallel work.
- Status: open

Next:
- Integration should resolve N02-REQ-001 and run the combined notebook checks.
- User review required: open the notebook at normal width, inspect the Mermaid data-flow diagram, Zarr field tables, NVIDIA-green batch plot, and the transition into `ExtXYZReader`.

## 2026-08-11 22:15 EDT — revised lesson design after N01 consolidation

Owner: N02
Status: in progress

Design:
- Lesson outcome: trace one molecular graph from `AtomicData` into Zarr and back, then scale the same public path to a collection, loader batches, and a custom extxyz reader.
- Proposed cell sequence: lesson map and imports; checked molecule load; inspect ethyne; create one `AtomicData`; write one record; inspect Zarr groups and atom pointers; read and inspect one raw record; create all 32 records; write one `Batch`; retrieve three familiar examples; index a `Dataset`; iterate a `DataLoader`; summarize and plot ragged batches; implement `ExtXYZReader`; reuse `Dataset` and `DataLoader`; exact round-trip checks; advanced I/O study; results summary.
- Toolkit APIs kept visible: `AtomicData.from_atoms`, `add_system_property`, `model_validate`, `AtomicDataZarrWriter.write`, `AtomicDataZarrReader.read` and `read_many`, `Batch.from_data_list`, `Dataset`, `DataLoader`, and `Reader`.
- Molecules and model: the shared 32-molecule NCI Atlas collection, with ethyne, phenol, and 2,3-dimethylbutane as concrete examples; no model is used because the lesson isolates storage and loading behavior.
- Helper boundaries: checked paths, source checksums, temporary workspace setup, and plot-style setup live in a local helper; every central Toolkit storage, reader, dataset, and batching call stays in the notebook.
- Expected runtime: about 5–15 seconds on CPU for 32 small records; GPU is unused.
- Validation plan: full-notebook IPython-transformed AST parsing, lesson contract tests, helper tests, Ruff, fresh-kernel execution, exact 32-record/322-atom totals, field-level and dtype checks, selected-record identity, Zarr/extxyz equality, clean source outputs, and visual review of the dark-style NVIDIA-green batch plot.

Changed guidance applied:
- Follow N01's concept-first order, checked local helper boundary, compact settings, literal pointer and batch-boundary displays, and `Results summary` close.
- Keep compression, chunking, and loading-performance experiments in the advanced section.

Next:
- Rework the notebook and scoped tests, then rerun the frozen-runtime validation from a fresh kernel.

## 2026-08-11 23:43 EDT — revised lesson implemented and validated

Owner: N02
Status: ready for integration

Changed:
- Reworked `zarr-data-loading.ipynb` to 39 cells with a concept-first path: one ethyne graph, one Zarr record, the 32-record collection, selected reads, `Dataset`, `DataLoader`, ragged `Batch` inspection, an extxyz `Reader`, exact round-trip checks, advanced work, and a results summary.
- Added `helpers/lesson.py` and `helpers/__init__.py` for pinned-data verification, source-path resolution, temporary workspace ownership, and shared plot-style setup.
- Added `.gitignore`, `tests/conftest.py`, and `tests/test_lesson.py`; strengthened `tests/test_notebook_contract.py` with full-namespace parsing, concept-order, visible-API, separation, source-cleanliness, and helper-boundary checks.
- Kept all notebook edits in the live VS Code bridge. Revisions 40–83 contain only agent changes; the active VS Code editor was on another notebook.

Checks:
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python 3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/02-zarr-data-loading/tests`: 9 passed in 2.10 seconds.
- `./scripts/v3-run ruff check notebooks/02-zarr-data-loading`: passed.
- Fresh-kernel nbconvert execution: all 24 code cells executed with zero errors in 6.3 seconds on CPU.
- Exact execution checks passed for 32 records, 322 atoms, ordered examples at indices 0/23/31, four batches with 46/66/91/119 atoms, and value/shape/dtype equality for atomic numbers, positions, masses, categories, velocities, charge, and record ID through Zarr and extxyz paths.
- Source notebook remains clean: 39 cells, no execution counts, no outputs, and zero VS Code error diagnostics.
- Visual review passed: the dark style is active; the single plot uses NVIDIA green plus white hatching, direct values, readable labels, and explicit axes.
- Rich progress is omitted because the full CPU execution has no visible wait.

Blockers:
- The first nbconvert attempt could not allocate a local kernel socket inside the restricted sandbox. The approved external rerun passed. No active blocker remains.

Shared requests:
- N02-REQ-001 remains open for final root navigation and build integration.
- No new shared or cross-notebook change is required for the revised lesson.

Next:
- Integration should resolve N02-REQ-001 and run the combined notebook suite.
- User review: inspect the opening Mermaid flow, one-record Zarr tree and pointer explanation, 4/13/20-atom selected-record table, first-batch boundaries, ragged-batch plot, custom `Reader`, and exact round-trip table at normal notebook width.

## 2026-08-12 18:00 EDT — guide-aligned revision brief

Owner: N02
Status: in progress

Observed:
- The current source notebook is saved and idle: 39 cells, no outputs, no
  execution counts, and no live focused notebook cell. The owned directory and
  worklog are untracked in this shared worktree, so every existing owned file
  must be preserved through guarded edits.
- The draft already demonstrates a correct CPU-only 32-record / 322-atom Zarr
  round trip, but it predates the shared banner, folded Part 02 orientation,
  part-specific course map, approved callouts, current `Try it` / `Recap`
  structure, and `InMemoryDataset`.
- Official Toolkit 0.2 documentation and the pinned installed API agree on the
  main boundary: `Reader.read` / `read_many` return raw CPU tensor dictionaries
  plus metadata; `Dataset` validates them into `AtomicData`; `DataLoader`
  yields graph-aware `Batch` objects. `InMemoryDataset` implements the same
  loader-facing contract after one resident materialization.

Design:
- Lesson outcome: persist a collection of molecular `AtomicData` records with
  explicit stable record IDs, reopen it through the supported Toolkit reader,
  validate records at the `Dataset` boundary, and iterate model-ready `Batch`
  objects without keeping the complete collection in learner-managed memory.
- Prior capability: Part 01's
  [`AtomicData` and `Batch`](../notebooks/01-atomicdata-batch/atomicdata-and-batch.ipynb#from-one-molecule-to-a-batch)
  path turns structures into graphs and packs unequal systems.
- New capability: Part 02 makes those graph records durable and replaces
  hand-managed collection batching with the interchangeable
  `Reader -> Dataset / InMemoryDataset -> DataLoader -> Batch` path.
- Cell and visual sequence: shared banner and short Goal/Core concepts; folded
  orientation with `curriculum-map-02.svg`; compact hidden setup; rebuild the
  pinned 32-molecule input and attach stable integer `record_id` fields; write
  one bulk `Batch`; reopen the store and stream the first useful loader batch;
  ask where validation occurs and show one compact flow diagram; inspect Zarr
  pointer layout; use ordered `read_many`; show a realistic malformed-reader
  record rejected at the `Dataset` boundary; iterate all streamed batches and
  inspect graph/atom totals and CPU device; substitute `InMemoryDataset` behind
  the same `DataLoader`; bounded record-order `Try it`; two-part `Recap` with
  the Part 03 handoff.
- Visuals: the shared Part 02 course map answers where persistence sits in the
  course; one local flow diagram answers where raw storage tensors become
  validated `AtomicData`. Each receives a one-sentence takeaway. The old atom
  count bar chart is removed because Part 01 already teaches ragged batches.
- Visible public APIs: `AtomicData.from_atoms`, `add_system_property`,
  `Batch.from_data_list`, `AtomicDataZarrWriter.write`,
  `AtomicDataZarrReader.read`, `AtomicDataZarrReader.read_many`, `Reader`,
  `Dataset`, `InMemoryDataset`, `DataLoader`, dataset indexing, loader
  iteration, and dataset `close`. `ZarrData` is linked to the official
  trajectory example as the workflow sink that delegates this storage path;
  no dynamics workflow is duplicated here.
- Helper boundaries: local helpers own pinned path and checksum checks,
  collection loading, temporary workspace lifetime, compact source-record
  preparation that is not the new lesson, and presentation setup. Stable IDs,
  writer/reader construction, persistence, validation, dataset choice, loader
  configuration, iteration, and inspection remain visible.
- Structures and data: the pinned 32 distinct neutral H/C/N/O NCI Atlas
  molecules, 322 atoms total; ethyne (record 0, 4 atoms), phenol (record 23,
  13 atoms), and 2,3-dimethylbutane (record 31, 20 atoms) remain concrete
  continuity examples.
- Outputs, shapes, units, device, and scope: raw `atomic_numbers [V]` and
  `positions [V, 3]` (angstrom) remain CPU tensors at the reader boundary;
  system-level `record_id [1]` is a stable integer carried in the record;
  loader output is CPU `Batch` data with eight graphs per full batch and
  variable atom totals. This lesson makes no model, GPU, scientific, or
  performance claim.
- Expected runtime: about 5–15 seconds on CPU for source verification, one
  32-record bulk write, reader checks, resident materialization, and two loader
  passes. GPU is intentionally unused.
- Try it: edit a bounded non-sorted list of three valid record IDs passed to
  `read_many`; the visible success signal checks that returned `record_id`
  values exactly preserve the requested order.
- Validation plan: frozen runtime check; complete notebook namespace parse;
  notebook-design checker for Part 02; scoped Ruff and Pytest; exact
  32-record/322-atom, selected-order, pointer, shape, dtype, stable-ID,
  malformed-record, streamed/resident parity, and CPU-device checks; guarded
  notebook save; fresh-kernel top-to-bottom execution; then rendered review of
  the banner, folded map, flow diagram, API callout, tables, pacing, links, and
  success signal at normal teaching width.

Shared requests:
- N02-REQ-001 remains open for final root navigation and build integration.

Next:
- Confirm the exact pinned validator failure and resident-dataset behavior,
  then revise the local helper, tests, and notebook through guarded bridge
  operations.

## 2026-08-12 18:45 EDT — learner revision executed

Owner: N02
Status: ready for coordinator review

Changed:
- `notebooks/02-zarr-data-loading/zarr-data-loading.ipynb`
- `notebooks/02-zarr-data-loading/helpers/__init__.py`
- `notebooks/02-zarr-data-loading/helpers/lesson.py`
- `notebooks/02-zarr-data-loading/tests/test_lesson.py`
- `notebooks/02-zarr-data-loading/tests/test_notebook_contract.py`
- `worklog/02-zarr-data-loading.md`

Implemented:
- Added the shared ALCHEMI banner, exact Goal/Core concepts opening, folded
  Part 02 orientation, exact Part 01 capability link, and
  `curriculum-map-02.svg`.
- Reframed the lesson around one useful persistence question: save the
  32-molecule graph collection once, then recover model-ready CPU `Batch`
  objects through `AtomicDataZarrReader -> Dataset -> DataLoader`.
- Kept stable integer `record_id` construction, bulk writer call, raw reader
  calls, validating dataset, loader configuration, `InMemoryDataset`
  substitution, inspection, and lifecycle calls visible.
- Added the approved Highlight and ALCHEMI Toolkit API callouts plus one focused
  Mermaid diagram showing the raw-reader / validated-dataset / batched-loader
  boundary. Removed the old ragged atom-count plot, which duplicated Part 01.
- Replaced the full extxyz adapter with a compact malformed-reader test double.
  The demonstrated validator claim is limited to the observed invariant:
  `positions` must have the same atom-row count as `atomic_numbers`.
- Added the bounded non-sorted `read_many` exercise and visible stable-ID
  success message, current two-part `Recap`, Part 03 link, and official NVIDIA
  `ZarrData` trajectory example.
- Added `InMemoryDataset` round-trip coverage and updated structural tests for
  the current teaching and visual system.

Validation:
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python
  3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`,
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run ruff check notebooks/02-zarr-data-loading`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/02-zarr-data-loading/tests`:
  14 passed in 3.98 seconds.
- `./scripts/v3-run python
  /home/nfedik/.codex/skills/alchemi-tutorial-authoring/scripts/check_notebook_design.py
  notebooks/02-zarr-data-loading/zarr-data-loading.ipynb --part 02`: zero
  errors and zero warnings.
- Fresh kernel:
  `CUDA_VISIBLE_DEVICES='' ./scripts/v3-run jupyter nbconvert --to notebook
  --execute --ExecutePreprocessor.timeout=120 --output-dir /tmp --output
  n02-zarr-data-loading-executed.ipynb
  notebooks/02-zarr-data-loading/zarr-data-loading.ipynb`: 25/25 code cells
  executed, zero cell errors; command elapsed 7.2 seconds. Warp emitted expected
  no-CUDA-driver warnings in this CPU-only sandbox.
- Fresh numerical checks: 32 records / 322 atoms; four eight-graph batches with
  46, 66, 91, and 119 atoms; first reader window 16; raw phenol fields
  `atomic_numbers [13] int32`, `positions [13, 3] float32` in angstrom, and
  `record_id [1] int64`, all on CPU; selected IDs `[31, 0, 23]` preserved;
  malformed positions reported `expected 13, got 12`; resident and on-demand
  first batches had identical IDs and `batch_ptr`.
- HTML export:
  `./scripts/v3-run jupyter nbconvert --to html --output-dir /tmp --output
  n02-zarr-data-loading-executed.html
  /tmp/n02-zarr-data-loading-executed.ipynb`: passed, 345,930-byte HTML.
  Static inspection confirmed the banner and course-map references, folded
  orientation, both approved callouts, Mermaid source, and Try it success.
- Source notebook remains clean: 41 cells, 25 code cells, no saved outputs or
  execution counts. No GPU calculation or performance claim was added.

Rendered review:
- HTML export and content-marker checks passed.
- Pixel-level VS Code/browser review remains pending. Classic nbconvert retains
  the Mermaid source but does not establish its rendered geometry, and this
  session has no live notebook renderer for checking the folded map and
  callouts at teaching width.

Shared requests:
- N02-REQ-001 remains open for final root navigation and build integration.

Blockers:
- No implementation or execution blocker.
- Coordinator/user visual review is still required for the banner, folded
  course map, Mermaid flow, and both callouts at normal and narrow notebook
  widths.
- The notebook bridge exposes guarded source edits but no cell-metadata edit.
  The 14-line import cell is compact but remains visible. Coordinator ruling:
  accept it for review or apply the shared `hide-input` metadata during
  integration.

Next:
- Open the source notebook with the frozen v3 kernel, run all cells, and perform
  the pending rendered learner review. Then resolve N02-REQ-001 in the bounded
  integration pass.

## 2026-08-12 19:05 EDT — independent-review correction brief

Owner: N02
Status: in progress

Review findings and planned fixes:
1. Correct the learner-facing keyword-only loader signature to
   `DataLoader(dataset, batch_size=...) -> Iterator[Batch]`.
2. Demonstrate the on-demand boundary by deleting `molecules`, `records`, the
   bulk `source_batch`, and the writer after persistence, visibly confirming
   that the source graph collections are no longer in the learner namespace,
   then opening a new reader from the store.
3. Add a successful `dataset[index] -> (AtomicData, metadata)` step before the
   malformed-reader example. Inspect its concrete type, stable ID, atom/system
   field ownership, shapes, dtype, device, and metadata.
4. Replace ID-only runtime coverage with scientific round-trip assertions for
   atomic numbers, positions, periodic cell/PBC, charge when present, field
   levels, dtype/device, successful Dataset conversion, malformed-record
   rejection, and resident/on-demand payload parity.
5. Diagnose the CUDA stream before changing presentation. Minimal probes show
   clean `python`, `torch`, `helpers`, `warp`, and `nvalchemi` imports, but
   `from nvalchemi.data import AtomicData` emits the CUDA-driver stream.
   Installed pinned source identifies the exact trigger:
   `nvalchemi/data/buffer_kernels.py:40-43` imports Warp, sets quiet mode, and
   calls `wp.init()` unconditionally. `CUDA_VISIBLE_DEVICES=''` and setting
   Warp quiet mode before the Toolkit import do not prevent the native
   driver-entry-point errors. Official NVIDIA Warp 1.16 installation guidance
   says PyPI wheels include CUDA support, CUDA requires a compatible NVIDIA
   driver, and CPU-only operation requires the CPU conda variant or a
   `-DWARP_ENABLE_CUDA=OFF` build. N02 cannot change the frozen environment or
   installed Toolkit source.
6. Clarify that reader-backed `InMemoryDataset` materializes its resident cache
   on CPU; `device` controls emitted samples/batches. A GPU-resident cache must
   be supplied as a prebuilt GPU `in_memory_batch`.
7. Apply the shared `hide-input` plus `jupyter.source_hidden` metadata pattern
   to setup cells 2 and 4.
8. Move the evolving Part 02 map outside the collapsed orientation so it is
   immediately visible. Embed the live shared
   `../../shared/curriculum-map-02.svg` with the shared README's `<object>`
   pattern, nested visible fallback, accessible Part 02 label, and direct link;
   do not copy or hash the asset.
9. Split resident loader construction, first iteration, and parity inspection
   into separate observable cells.

Teaching sequence:
- Briefly reactivate Part 01's one-graph `AtomicData`, packed-system `Batch`,
  and field-ownership ideas, then progress explicitly through
  `AtomicData -> Zarr record -> Reader -> Dataset -> DataLoader -> Batch`.
- Keep all public boundaries visible, interpretation after inspection, the
  bounded `read_many` exercise, and the current recap.

Validation plan:
- Add behavioral tests first and confirm they fail for missing scientific
  coverage/notebook behavior before correction.
- Run the frozen runtime check, scoped Ruff, all N02 tests, Part 02 design
  checker, clean-source/metadata checks, fresh CPU execution, output and stream
  inspection, HTML export, and a 1280-pixel Playwright rendered check that the
  shared Part 02 SVG has nonzero visible geometry.
- N02-REQ-002 (coordinator-owned): provide either a compatible NVIDIA-driver
  execution host or a frozen CPU-only Warp build. Until then, current WSL
  fresh-kernel runs can verify computation but cannot meet a clean import-stream
  acceptance gate. The earlier description of these native lines as harmless
  warnings is withdrawn; they are diagnosed driver-initialization errors even
  though this CPU calculation subsequently succeeds.

## 2026-08-12 22:30 EDT — fresh final verification

Owner: N02
Status: source/spec ready; external clean-stream and pixel-review gates remain open

Latest series decision:
- The current Part 02 design checker supersedes the original nested-image
  fallback detail. Cell 3 now keeps the verbatim current Part 01 product body
  inside the folded `Where NVIDIA ALCHEMI fits (recap)` disclosure, exposes
  `New to ALCHEMI Toolkit?` as its accessible label, and places the live map
  object after the disclosure with the required interactive aria label,
  `aspect-ratio:900/552`, and `multi-GPU workflows` wording. The nested
  `<img>` was removed; the direct SVG link remains immediately after the
  object. The strengthened local contract mirrors this current series rule.
- No lesson redesign or scope expansion was made. The changes are limited to
  the current opening/map contract and a source-accuracy correction exposed by
  fresh execution.

Current-source evidence:
- The source notebook validates as nbformat 4 and remains clean at 56 cells:
  37 code and 19 Markdown, 56 unique cell IDs, zero saved outputs, and zero
  execution counts. Setup cells 2 and 4 retain both `hide-input` and
  `jupyter.source_hidden`.
- The progression is explicit and gradual:
  source `AtomicData` construction and inspection; one bulk Zarr write;
  release of `molecules`, `records`, `source_record`, and `source_batch`;
  new `AtomicDataZarrReader`; raw `read(23)` inspection; successful
  `dataset[23]`; `DataLoader` construction; first `Batch`; then the four-stage
  comparison.
- The four-stage table directly compares the stable `record_id`, shapes,
  atom/system ownership, ownership source, device, and metadata behavior for
  `AtomicData before write`, `Reader after reopen`, `Dataset indexing`, and
  `DataLoader batching`.
- Fresh output exposed a pinned-release nuance that the previous source hid:
  direct `Dataset` indexing preserves the `record_id` value but
  `AtomicData.model_validate(...)` does not rebuild the custom
  `AtomicData.system_properties` registry. The notebook now shows the empty
  direct view, uses durable `reader.field_levels` as the Dataset-stage
  ownership evidence, and shows `Batch.keys["system"]` restoring
  `record_id` at the loader boundary. Installed source confirms
  `Dataset._to_atomic_samples` calls `AtomicData.model_validate(data_dict)`
  without applying reader field levels; the validated DataLoader batch path
  does preserve them.
- Fresh values: source/raw/validated phenol all carry `record_id [23]`,
  `atomic_numbers [13]`, `positions [13, 3]`, and CPU placement. The first
  loader result is a CPU `Batch` with IDs 0–7, 8 graphs, 46 atoms, and
  `record_id` in its system keys. The four streamed batches contain
  46/66/91/119 atoms for 32 graphs and 322 atoms total.
- Exactly one learner-facing malformed-reader boundary remains:
  `MissingPositionReader` truncates phenol positions from 13 to 12 rows,
  `Dataset` raises the observed Pydantic validation error, and the claim is
  limited to that atom-row invariant.

Independent-review resolution:
1. Resolved: the callout uses
   `DataLoader(dataset, batch_size=...) -> Iterator[Batch]`.
2. Resolved: the learner releases the source graph collection and writer,
   confirms the names are absent, and opens a fresh reader.
3. Resolved: successful `dataset[23]` appears before the malformed boundary and
   inspects concrete type, stable ID, shapes, dtype, device, metadata, stored
   field levels, and the direct custom-property view.
4. Resolved: tests cover atomic numbers, positions, cell, PBC, charge, stable
   IDs, field levels, dtype/device, successful Dataset conversion, malformed
   rejection, and complete streamed/resident payload parity.
5. Resolved as diagnosis, not suppression: Toolkit data import still triggers
   Warp CUDA driver initialization. N02-REQ-002 remains open.
6. Resolved: the text and output distinguish reader-backed CPU cache placement,
   emitted target device, and a prebuilt GPU `in_memory_batch`.
7. Resolved: setup cells 2 and 4 use the shared hidden-input metadata.
8. Resolved against the latest series contract: the folded recap precedes the
   visible interactive map, required accessibility/geometry text is present,
   and a direct SVG link follows the object. The obsolete nested `<img>`
   requirement is intentionally not retained.
9. Resolved: resident loader construction, first iteration, and parity
   inspection are separate cells.
- Later API-exploration requirement: resolved by direct inspection of the real
  source `AtomicData`, raw reader dictionary/metadata, successful Dataset item,
  and emitted DataLoader `Batch`, followed by one four-stage table.
- Later recap-parity requirement: resolved by a contract that extracts the
  current Part 01 product cell and compares normalized body text with the N02
  folded recap.

Changed in this verification pass:
- `notebooks/02-zarr-data-loading/zarr-data-loading.ipynb`
- `notebooks/02-zarr-data-loading/tests/test_notebook_contract.py`
- `worklog/02-zarr-data-loading.md`
- Removed
  `notebooks/02-zarr-data-loading/n02-source-review.html` with the safe file
  deletion tool.

Commands and results:
- `./scripts/v3-run python environment/check_runtime.py`: passed; Python
  3.12.13, Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`,
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run ruff check notebooks/02-zarr-data-loading`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider notebooks/02-zarr-data-loading/tests`:
  22 passed in 4.72 seconds.
- `./scripts/v3-run python
  /home/nfedik/.codex/skills/alchemi-tutorial-authoring/scripts/check_notebook_design.py
  notebooks/02-zarr-data-loading/zarr-data-loading.ipynb --part 02`: zero
  errors and zero warnings.
- The explicit nbformat/IPython-transform/AST/metadata/output validation
  passed: valid nbformat 4, complete transformed namespace parses, 56 total /
  37 code cells, unique IDs, required hidden setup metadata, zero source
  outputs, and zero source execution counts.
- `/usr/bin/time -f 'elapsed=%e seconds' ./scripts/v3-run jupyter nbconvert
  --to notebook --execute --ExecutePreprocessor.timeout=120 --output-dir /tmp
  --output n02-zarr-data-loading-executed.ipynb
  notebooks/02-zarr-data-loading/zarr-data-loading.ipynb >
  /tmp/n02-zarr-data-loading-execute.stdout 2>
  /tmp/n02-zarr-data-loading-execute.stderr`: exit 0 in 6.46 seconds; 37/37
  code cells executed, zero cell errors.
- Captured execution stderr contains 79 Warp CUDA driver-entry-point errors,
  one Warp CUDA driver warning, and one Jupyter TCP transport warning. The
  saved notebook has one 80-line stderr stream: first
  `cuDriverGetVersion`, last `cuMipmappedArrayGetLevel`. These are not
  suppressed or relabeled.
- `/usr/bin/time -f 'elapsed=%e seconds' ./scripts/v3-run jupyter nbconvert
  --to html --output-dir /tmp --output n02-zarr-data-loading-executed.html
  /tmp/n02-zarr-data-loading-executed.ipynb`: passed in 1.65 seconds;
  379,263-byte HTML.
- Static HTML review passed for the folded product recap, unfolded interactive
  map and direct link, one Highlight, one Toolkit API callout, four tables,
  stable-ID exercise success, recap, ownership disclosure, eight well-formed
  HTTPS links, and all four local linked targets.

Remaining gates:
- N02-REQ-002 remains open. A compatible NVIDIA-driver host or frozen CPU-only
  Warp build is required for a clean import/execution stream. CPU notebook
  computation itself succeeds.
- Pixel-level review at normal and narrow teaching widths remains pending
  because no browser/notebook renderer was available in this verification
  session. Static HTML structure is complete.
- No remaining required N02 source fix was found. N02-REQ-001 remains an
  integration-owned navigation/build task outside strict N02 ownership.

## 2026-08-13 10:45 EDT — Core-aligned deep-dive revision brief

Owner: N02
Status: in progress

Observed:
- The current 56-cell draft has strong ownership, validation, indexing, and
  resident-versus-streamed material. It starts with the complete NCI collection,
  so it does not yet satisfy the Core handoff contract's small deterministic
  example followed by a larger scientific source.
- The pinned Toolkit 0.2 data path is
  `Reader -> Dataset / InMemoryDataset -> DataLoader -> Batch`.
  `Reader.read_many(...)` must preserve requested order, `Dataset` validates
  raw CPU dictionaries, and `DataLoader` restores explicit field levels when
  it collates a `Batch`.
- Importing `nvalchemi.data` in this frozen CUDA-enabled environment still
  invokes `wp.init()` from `buffer_kernels.py`. On this CPU-only WSL host that
  produces CUDA driver-entry-point errors before the CPU lesson runs. The
  stream must remain visible and be described as an environment boundary.

Design:
- Lesson outcome: write and reopen a deterministic three-record store, then
  apply the same public interfaces to the pinned 32-molecule NCI source through
  a real extxyz `Reader`, an on-demand `Dataset`, a resident
  `InMemoryDataset`, and graph-aware `DataLoader` batches.
- Prior capability: the Core and Part 01 already introduce `AtomicData`,
  per-atom/per-system ownership, `Batch`, and graph boundaries. Part 02 extends
  that path across persistent storage and replaceable readers.
- Sequence: prerequisites and CPU/CUDA boundary; three synthetic records;
  public writer and reader; raw record, validated dataset item, and first
  loader batch; field-ownership table; custom `ExtXYZReader`; NCI indexing and
  ordered `read_many`; incremental NCI-to-Zarr write; reopened NCI loader;
  pointer/count visual; one malformed-reader validation boundary; on-demand
  versus resident cache/device comparison; bounded exercise; Core, official
  trajectory, and Part 03 handoffs.
- Visible public APIs: `AtomicData`, `add_system_property`,
  `Batch.from_data_list`, `AtomicDataZarrWriter.write` and `append`,
  `AtomicDataZarrReader.read` and `read_many`, `Reader`, `Dataset`,
  `InMemoryDataset`, `DataLoader`, dataset/reader/batch indexing, and `close`.
- Helper boundary: checksums, manifest parsing, temporary paths, and plot
  styling live in the owned helper. Reader implementation and every central
  Toolkit call remain visible.
- Scientific source: the pinned NCI Atlas extxyz/manifest pair contains 32
  neutral H/C/N/O molecules and 322 atoms. Ethyne, phenol, and
  2,3-dimethylbutane anchor ordered selection and field checks.
- Outputs: positions are float32 `[V, 3]` in angstrom; atomic numbers are int32
  `[V]`; `record_id` is int64 `[1]` at system level; readers return CPU tensors;
  the dataset chooses the emitted device; each loader result is a `Batch`.
- Validation boundary: one reader record has a shortened `positions` tensor,
  and `Dataset` rejects only the demonstrated atom-row mismatch.
- Exercise: request three non-sorted NCI indices and verify that both metadata
  and system-level IDs preserve request order.
- Verification: red-green helper and notebook-contract tests; frozen runtime;
  Ruff; notebook design/schema/source checks; fresh-kernel CPU execution with
  stderr retained; output assertions; HTML export; 1280 px and narrow rendered
  review; then two complete API/science and learner-prose revision passes.

## 2026-08-13 — Final Core-aligned implementation and rendered review

Owner: N02
Status: TECHNICALLY VALIDATED, RENDERED-REVIEWED DRAFT — HUMAN CELL REVIEW REQUIRED

Implemented:
- Rebuilt the lesson around a deterministic hydrogen/water/methane write and
  genuine reopen before the 32-molecule NCI path.
- Kept the public writer, reader, `Reader` extension, `Dataset`,
  `InMemoryDataset`, `DataLoader`, `Batch`, indexing, lifecycle, field-level,
  cache, and emitted-device APIs visible.
- Added manifest-only loading, a tested count/pointer plot, explicit HTML
  alternative text, and Mermaid `accTitle` / `accDescr`.
- Kept one validation boundary: a shortened `positions` tensor is rejected
  because it has 12 rows for a 13-atom record.
- Removed `effective_read_window` from the lesson so inflight-buffer tuning and
  profiling remain out of scope.
- Moved the figure interpretation and reopened-store introduction before
  loader construction. Learner-visible code lines now stay at or below 80
  characters.
- Made the curriculum `<object>` a valid inline HTML tag. The exported notebook
  now renders the shared Part 02 map instead of escaped tag text.

Revision pass 1 — API, plan, pedagogy, and science:
- Rechecked the pinned `Reader`, `Dataset`, `InMemoryDataset`, and `DataLoader`
  contracts against Toolkit commit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` and the current official Dataset,
  DataLoader, and trajectory Zarr guides.
- Confirmed ordered `read_many`, CPU reader tensors, Dataset validation and
  target-device ownership, reader-backed CPU residence, graph-aware `Batch`
  output, and NCI identities: 32 neutral H/C/N/O molecules and 322 atoms.
- Fresh outputs confirm four eight-graph batches, 322 total atoms, phenol's
  `[13, 3]` float32 positions, the `expected 13, got 12` rejection, resident /
  on-demand parity, and non-sorted labels
  `Phenol`, `2,3-dimethylbutane`, `Ethyne`.

Revision pass 2 — rendered first-time-learner and prose:
- Removed the remaining internal read-window output, generic transition
  wording, ambiguous resident-cache recap, and two product-recap spacing
  errors.
- Reviewed the full HTML at 1280 px and 720 px. Both widths have page
  `scrollWidth == clientWidth`, no console errors, no failed requests, no
  broken images, and no missing image alternative text.
- The render contains the labeled curriculum map, one accessible Mermaid
  figure, the labeled NCI count/pointer plot, seven tables, and one folded
  recap. At 720 px, the only measured internal overflow is the intentionally
  scrollable resident-choice table container; the page itself does not
  overflow.

Final verification:
- `./scripts/v3-run python environment/check_runtime.py`: passed with Python
  3.12.13, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, and Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- `./scripts/v3-run ruff check notebooks/02-zarr-data-loading`: passed.
- `./scripts/v3-run pytest -q -p no:cacheprovider
  notebooks/02-zarr-data-loading/tests`: 29 passed.
- Notebook design/schema checker: zero errors.
- Fresh kernel: all 47 code cells executed; zero cell errors.
- Final execution emitted the Jupyter TCP warning but no Warp/CUDA stderr.
  No warning or stderr suppression was added. Earlier CPU-host Warp failures
  remain documented above, and the learner-facing environment note remains
  conditional.
- HTML export: 478,629 bytes; Chromium render reviewed at both requested
  widths.

Complete cell-review index:
- 00 banner; 01 goal, prerequisites, and CPU/CUDA boundary; 02 hidden imports;
  03 product recap and curriculum map; 04 hidden checked paths/workspace.
- 05 synthetic orientation; 06 three `AtomicData` records; 07 system IDs;
  08 source summary construction; 09 source summary display; 10 ownership
  interpretation; 11 source `Batch`.
- 12 writer contract; 13 initial Zarr write; 14 release source objects;
  15 reopen reader; 16 genuine-reopen interpretation; 17 raw indexed read;
  18 field table construction; 19 field table display; 20 raw reader summary.
- 21 Dataset/DataLoader responsibilities; 22 Dataset construction;
  23 validated indexing; 24 validated summary; 25 pinned ownership nuance;
  26 DataLoader construction; 27 first `Batch`; 28 batch summary;
  29 boundary-comparison construction; 30 comparison display;
  31 boundary interpretation.
- 32 highlight; 33 public API callout; 34 accessible ownership/data-flow
  diagram; 35 synthetic-path cleanup.
- 36 NCI orientation; 37 collection identity; 38 anchor molecules;
  39 custom `ExtXYZReader`; 40 raw extxyz read; 41 reader/dataset/loader roles;
  42 NCI Dataset; 43 validated phenol.
- 44 validation question; 45 malformed reader; 46 validation-boundary
  explanation; 47 rejected Dataset access; 48 exact invariant and write
  transition; 49 incremental write/append; 50 close source and reopen Zarr.
- 51 non-sorted `read_many`; 52 order contract; 53 selected-record table;
  54 count/pointer plot; 55 figure description and stream introduction;
  56 stored Dataset/DataLoader; 57 iterator; 58 batch-summary instructions;
  59 complete streaming assertions; 60 batch table; 61 stream summary.
- 62 on-demand/resident ownership table; 63 resident materialization;
  64 resident DataLoader; 65 resident first batch; 66 payload parity.
- 67 bounded exercise; 68 ordered-read solution and success signal;
  69 lifecycle cleanup; 70 official Dataset/trajectory handoff;
  71 recap and Part 03 handoff.

Remaining coordination:
- N02-REQ-001 remains integration-owned: add Part 02 to final root navigation
  and the built bundle after the notebook wave lands.
- N02-REQ-002 remains open for hosts that still reproduce the Warp CPU import
  stderr: provide a compatible NVIDIA-driver host or a frozen CPU-only Warp
  build. The lesson does not hide that boundary.
