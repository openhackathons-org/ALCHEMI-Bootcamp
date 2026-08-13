# ALCHEMI Core deep-dive handoff contract

This coordinator-owned file defines how later lessons extend the Core without
changing its scientific claims or copying its learner narrative. The generated
Core remains a technically validated draft until a human reviews every cell.

## Core excerpt IDs and provenance

The build manifest and lock are authoritative. A deep dive may reuse a Core
excerpt only by stable generated ID and recorded source hash. If a source hash
changes, review the excerpt again before updating the lock.

The current canonical excerpts are:

- Batch: `core-c-p01-batch-build`, `core-c-p01-graph-counts`,
  `core-c-p01-batch-pointer`, and `core-c-p01-batch-owner` from Part 01.
- Zarr: `core-c-p02-zarr-write` and `core-c-p02-zarr-reader` from Part 02.
- Supplied model: `core-c-p03-aimnet-load` from Part 03 and
  `core-c-p05-model-config` from Part 05.
- Hooks: `core-c-p04-neighbor-hooks` from Part 04.
- FIRE2: `core-c-p05-fire2-run` from Part 05.

The Core `AtomicData.from_atoms(...)`, molecule-list conversion, and
`Batch.get_data(...)` cells are intentionally Core-authored. Part 01 now uses
water, methane, and ethanol; copying those cells into the Core molecular story
would introduce undefined or misleading variables. Their transition
provenance is recorded in the lock instead of claiming Part 01 source parity.

Core-authored blocks have transition provenance, not a claimed deep-dive
origin. A later lesson may replace one only after its new source cell has a
stable ID, a reviewed hash, and an explicit manifest update.

## Synthetic quick example + real-world example

Each lesson should begin with a small deterministic example, then show why the
same API matters in a scientifically meaningful calculation.

- Part 02 pairs the three-molecule Zarr reload with a trajectory or dataset
  large enough to motivate streaming, selection, and schema checks.
- Part 03 pairs the quadratic toy wrapper with a physically defined composition
  such as Lennard-Jones plus Ewald, including unit and parity checks.
- Part 04 pairs the one-step observation hook with safety, snapshots, cleanup,
  and failure behavior in a longer workflow.
- Part 05 pairs the bounded, non-converged molecular attempt and five-step argon
  trace with validated convergence and energy-drift protocols.
- Part 06 pairs a tiny parity-safe compiled call with a real GPU pipeline and
  profiling workflow. Never present warm-up or compilation time as throughput.
- Part 07 pairs the fixed-four-atom optimizer sandbox with a documented
  train/validation split and a scientifically appropriate model.
- Part 08 pairs the world-size-one partition/run/gather sequence with a
  separately launched multi-GPU parity example before any scaling result.

## Required public APIs

Use documented interfaces only. Preserve the Core prerequisite chain:
`AtomicData` → `Batch` → storage/model interfaces → hooks → dynamics →
training or distributed execution.

- Data: `AtomicData.from_atoms`, `node_properties`, `system_properties`,
  `add_node_property`, `add_system_property`, `Batch.from_data_list`,
  `Batch.add_key`, and `Batch.get_data`.
- Storage: `AtomicDataZarrWriter`, `AtomicDataZarrReader`, `Dataset`, and
  `DataLoader`.
- Models: `ModelConfig`, `set_config`, `compute_neighbors`,
  `BaseModelMixin`, `PipelineStep`, `PipelineGroup`, and
  `PipelineModelWrapper`.
- Workflows: `BaseDynamics`, `DynamicsStage`, documented built-in hooks,
  `FIRE2`, `NVE`, and supported thermostats where the lesson requires them.
- Training: `FineTuningStrategy`, `OptimizerConfig`, `EnergyMSELoss`, and
  `default_training_fn`.
- Distributed: `DistributedManager`, `DomainConfig`, `DomainParallel`,
  `partition`, `run`, and `gather`.

Do not use private attributes, fabricated contexts, `object.__setattr__`,
notebook-launched SSH, or notebook-launched `torchrun`.

## Language, visual, and cell-size rules

- Write as an instructor guiding a scientist. Do not expose curriculum labels,
  action IDs, timing bands, provenance, tests, or implementation status.
- State when the molecule, model, dataset, or task changes. Separate examples
  must not be described as one scientific pipeline.
- Prefer one meaningful operation per code cell. Most visible cells should fit
  on one screen; split cells that combine setup, execution, plotting, and
  interpretation.
- Show shapes, units, dtypes, devices, ownership, and a bounded real result.
- Use callouts only for information that changes interpretation or safe use.
- Every plot needs readable labels, units, a useful caption or nearby
  interpretation, and alt text that survives HTML export.
- Never infer equilibration, convergence, transferability, model improvement,
  scaling, or scientific accuracy from a smoke test.
- End each lesson section with a plain Go deeper link to an existing notebook,
  official source, or an explicitly in-progress placeholder.

## Human-review stages

1. Author review: run cells individually and check the scientific story.
2. Technical validation: run contract tests and a fresh-kernel execution in the
   pinned environment.
3. Rendered review: inspect the full HTML for source visibility, warnings,
   tables, plots, width, alt text, and links.
4. Scientific review: challenge every limitation and evidence claim.
5. Human cell review: approve or revise each learner-visible cell.

Automated checks do not replace stage 5.

## Placeholder paths to replace

- `../06-gpu-pipelines-profiling/gpu-pipelines-profiling.ipynb`
- `../07-training-finetuning/training-finetuning.ipynb`
- `../08-domain-decomposition/domain-decomposition.ipynb`

Replace `in progress` text only when the target exists, executes, renders, and
has completed the human-review stages above. Composition already links to
`../03-model-interfaces-composition/model-interfaces-composition.ipynb`.

## Required tests

Every deep dive must provide:

- notebook schema and fresh-kernel execution tests;
- source-cleanliness checks for private APIs, stale outputs, and internal
  curriculum language;
- public-API token and prerequisite-order checks;
- shape, unit, dtype, device, and ownership assertions for key results;
- scientific-boundary checks specific to the lesson;
- local-link and in-progress-placeholder checks;
- deterministic data/model identity checks where external artifacts are used;
- bounded cell-size and hidden-setup checks;
- rendered image-alt, table-width, plot-label, and error-output checks; and
- provenance tests for every excerpt selected into the Core.
