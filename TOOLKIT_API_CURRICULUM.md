# ALCHEMI v2 Toolkit API curriculum

Status: current learner-facing API contract, updated 2026-07-13.

Working repository allocation: AIMNet2, explicit pairwise D3, finite
electrostatics, GPU batching, FIRE2, fused dynamics, and predicted-charge IR
form the rebuilt Part 1. The older adsorption notebook remains as legacy
material but is not part of this molecular-model story. Part 2 is parked; the
Part 3 foundations notebook remains a broader research and runtime-validation
harness; it is not the polished learner-facing sequence.

This is the standalone list of Toolkit capabilities and APIs that the v2
tutorials should expose. It is a curriculum, not a dump of every public symbol:
the core list contains APIs learners should understand and use directly; later
sections capture useful extensions and ecosystem surfaces that should not crowd
the main notebooks.

## Source baseline and labels

The list was checked against four exact source states:

- Existing tutorial Toolkit pin:
  [`01c99d5`](https://github.com/NVIDIA/nvalchemi-toolkit/tree/01c99d5cde6f63d6f662b071a9f408d3bfc12b0a).
- Current v2 Toolkit Core pin:
  [`b770ee6`](https://github.com/NVIDIA/nvalchemi-toolkit/tree/b770ee6963fd2f6137891e408c370012751918e2),
  current `main` on 2026-07-08.
- Existing tutorial Toolkit-Ops pin:
  [`2b7c3c3`](https://github.com/NVIDIA/nvalchemi-toolkit-ops/tree/2b7c3c3adfb1ca84b886eecbf14bc60ff6ba1dc2),
  reporting version 0.3.1.
- Current v2 Toolkit-Ops pin:
  [`c6fbe65`](https://github.com/NVIDIA/nvalchemi-toolkit-ops/tree/c6fbe652315e0cebd4f57a6a25f626258f0dbbfd),
  `0.4.0-rc` on 2026-07-08.

Availability labels:

- **P+T** — present at both the earlier tutorial pin and current v2 pin.
- **T** — current v2 pin only.
- **P → T rename** — the capability remains, but its public name changed.

The release image must still freeze exact passing commits. A symbol being
present in source does not prove that its model/dependency combination works in
the workshop image.

Validation completed for this document:

- The canonical imports for the existing Core/Ops pins resolve in the current
  tutorial virtual environment.
- The current v2 symbols and public export lists were checked against the
  exact `b770ee6` / `c6fbe65` source trees.
- Scientific execution source SHA-256
  `5403dfcd42bb707e15527a443e76edaec38fe38a8888ab8d527433b1dbf8efc8`
  was accepted on an H100 in CL job `3087665` with the exact pinned Core/Ops
  commits. It exercised `AtomicData`, `Batch`, neighbors, AIMNet2 and D3
  wrappers, a dependent pipeline, segmented reductions, compiled/eager
  parity, FIRE2, fused NVT→NVE dynamics, hooks, `ZarrData`, and replay. All 31
  code cells completed without error, all 14 progress cards persisted as
  `COMPLETE`, and the exact 5,000 + 50,000 stage route passed.
- The current learner-facing SHA-256
  `81124de2e95e709a527522d026288a2c98d7e41b90ce7c4dd93e17a557b5a667`
  is a presentation-only revision apart from one callout-state correction; it
  has not been rerun on H100.
- A clean Docker rebuild remains a release-image gate; the H100 run used the
  pinned CL environment recorded in its manifest.

## Canonical roster at a glance

These are the learner-facing essentials. The detailed sections below define
why, where, and under which version each belongs.

- **Data:** `AtomicData`, `AtomicData.from_atoms`, `Batch`,
  `Batch.from_data_list`, `batch_idx`, `batch_ptr`, `get_data`,
  `to_data_list`, `index_select`.
- **Model contract:** `MACEWrapper`, `AIMNet2Wrapper`,
  `LennardJonesModelWrapper`, `model_config`, `active_outputs`, `set_config`,
  `from_checkpoint`, `make_neighbor_hooks`.
- **Neighbors:** `compute_neighbors`, `NeighborListHook`, and one direct
  `nvalchemiops.torch.neighbors.neighbor_list` call.
- **Relaxation/dynamics:** `FIRE2`, `ConvergenceHook.from_fmax`,
  `initialize_velocities`, `NVTLangevin`, `NVE`, `FusedStage`, `run`.
- **Hooks:** `FreezeAtomsHook`, `WrapPeriodicHook`, `NaNDetectorHook`,
  `EnergyDriftMonitorHook`, `LoggingHook`, `SnapshotHook`,
  `ConvergedSnapshotHook`, `BiasedPotentialHook`, `StageTimingHook`.
- **Composition:** `DFTD3ModelWrapper`, `EwaldModelWrapper`,
  `PMEModelWrapper`, `PipelineModelWrapper`, `PipelineGroup`, `PipelineStep`,
  and additive model `+`.
- **Persistence/replay:** `ZarrData`, `AtomicDataZarrWriter`,
  `AtomicDataZarrReader`, `Dataset`, `DataLoader`.
- **Demonstrate once:** `FIRE2VariableCell`, `NPT`, `SizeAwareSampler`,
  `GPUBuffer`, `HostMemory`, stress/Hessian/embedding outputs,
  `segmented_sum`.
- **Map only:** JAX mirrors, `DistributedPipeline`, training/fine-tuning,
  `UMAWrapper`, raw Warp/Ops dynamics, multipoles, and compilation tuning.

## The minimum learner contract

Every learner completing v2 should be able to explain and perform these seven
operations:

1. Convert an ASE structure into `AtomicData` and assemble a heterogeneous
   `Batch`.
2. Inspect a model adapter's declared inputs, outputs, and neighbor-list
   requirements.
3. Run one real batched model call and one batched relaxation or dynamics stage.
4. Attach public hooks for neighbors, physical constraints, safety,
   observability, and snapshots.
5. Compose independent or dependent physical/model contributions correctly.
6. Persist results to Zarr/CSV/`.extxyz` and replay them through the data
   pipeline.
7. Identify when Core, Toolkit-Ops, or the external model ecosystem is doing
   the work.

## 1. Data and batching — must teach directly

Canonical imports:

```python
from nvalchemi.data import AtomicData, Batch
```

| API | Availability | Learner outcome |
|---|---|---|
| `AtomicData.from_atoms(...)` | P+T | Convert an ASE `Atoms` object without handwritten coordinate blocks. |
| `AtomicData.from_structure(...)` | P+T | Convert a pymatgen structure when that is the natural generated input. |
| `AtomicData.use_default_masses()` | P+T | Populate masses before velocity initialization or dynamics. |
| `AtomicData.use_default_velocities()` | P+T | Create an explicit velocity field before initialization. |
| `AtomicData.use_default_categories()` | P+T | Prepare category-based constraints without private storage edits. |
| `AtomicData.add_node_property(...)` | P+T | Add per-atom fields such as velocities or categories. |
| `AtomicData.add_system_property(...)` | P+T | Add graph-level charge, multiplicity, identity, or conditions. |
| `AtomicData.chemical_hash()` | P+T | Give generated inputs a reproducible identity. |
| `AtomicData.to(...)` | P+T | Move data explicitly across device/dtype boundaries. |
| `Batch.from_data_list(...)` | P+T | Combine independent systems into one model call. |
| `Batch.batch_idx` / `Batch.batch_ptr` | P+T | Understand atom-to-system membership and segment boundaries. |
| `Batch.num_graphs` / `Batch.num_nodes_per_graph` | P+T | Inspect actual batch shape and heterogeneous system sizes. |
| `Batch.get_data(i)` / `Batch.to_data_list()` | P+T | Recover inspectable per-system results. |
| `Batch.index_select(...)` | P+T | Select failed, converged, or scientifically interesting systems. |

Fields learners must see at least once:

- `positions`, `atomic_numbers`, `atomic_masses`
- `cell`, `pbc`
- `energy`, `forces`, `stress`
- `charge` and, where relevant, `mult`
- `velocities`, categories, and a stable `system_id`

Avoid presenting batching as a detached speed benchmark. The batch dimension
must correspond to the scientific comparison: structures, conditions,
restraint windows, temperatures, timesteps, or model replicas.

Homogeneous and heterogeneous workloads use the same public `Batch` API. Teach
the performance tradeoff with one fixed set of structures evaluated as one
mixed batch and as size-homogeneous buckets. Report model calls, atoms, valid
neighbors, allocated neighbor slots, wall time, atoms/s, and structures/s;
there is no universal fastest layout.

## 2. Model adapter contract — must teach directly

Canonical imports:

```python
from nvalchemi.models import AIMNet2Wrapper, LennardJonesModelWrapper, MACEWrapper
from nvalchemi.models.base import ModelConfig, NeighborConfig, NeighborListFormat
```

| API | Availability | Learner outcome |
|---|---|---|
| `MACEWrapper.from_checkpoint(...)` | P+T | Load a materials/surface model through a public Toolkit adapter. |
| `AIMNet2Wrapper.from_checkpoint(...)` | P+T | Load a molecular, charged, or spin-aware model through the same contract. |
| `LennardJonesModelWrapper(...)` | P+T | Use a transparent analytical model for composition and dynamics validation. |
| `model.model_config` | P+T | Inspect declared outputs, required inputs, PBC support, and neighbors. |
| `model.model_config.active_outputs` | P+T | Request only the properties required by the calculation. |
| `model.set_config("active_outputs", {...})` | P+T | Change energy/force/stress/Hessian selection explicitly. |
| `model.make_neighbor_hooks()` | P+T | Let an adapter or composed pipeline provide its compatible neighbor hooks. |
| `model.compute_embeddings(...)` | P+T | Inspect learned representations as an optional interpretability exercise. |
| `model.embedding_shapes` | P+T | Discover embedding layout before allocating downstream analysis. |

The notebook must show, not hide:

- checkpoint name, exact revision, hash, license, device, and dtype;
- `outputs`, `active_outputs`, `required_inputs`, `optional_inputs`;
- neighbor cutoff, matrix/COO format, half/full-list convention;
- the model's applicability domain and unsupported chemistry.

MACE exposes `energy`, `forces`, `stress`, and `hessian` through the current
wrapper. AIMNet2 exposes `energy`, `forces`, `stress`, predicted `charges`, and
`spin_charges` for NSE checkpoints. Those are model capabilities, not a license
to make claims outside the checkpoint's training domain.

Use the exact output keys above: singular `stress` and `hessian`, plural
`forces`, `charges`, and `spin_charges`. AIMNet2-NSE consumes the system field
`mult` for multiplicity; do not invent a `spin` input.

## 3. Neighbor lists — must teach Core; show Ops once

For a one-shot model evaluation:

```python
from nvalchemi.neighbors import compute_neighbors

compute_neighbors(batch, config=model.model_config.neighbor_config)
```

For iterative relaxation or dynamics, prefer the model-generated hooks:

```python
for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)
```

The explicit Core hook remains important when teaching the mechanism:

```python
from nvalchemi.hooks import NeighborListHook
```

One short Toolkit-Ops cell should expose the accelerated primitive:

```python
from nvalchemiops.torch.neighbors import neighbor_list
```

Teach these arguments and outputs:

- `positions`, `cutoff`, `cell`, `pbc`
- `batch_idx`, `batch_ptr`
- matrix output versus COO with `return_neighbor_list=True`
- periodic shifts and fill values
- automatic method selection versus an explicitly requested method

Do not make learners choose low-level naive, cell-list, or cluster-tile kernels
in the main path. That is an implementation/performance topic, not the normal
workflow contract.

## 4. Relaxation and dynamics — must teach directly

Canonical imports:

```python
from nvalchemi.dynamics import (
    ConvergenceHook,
    FIRE2,
    FusedStage,
    NVE,
    NVTLangevin,
    initialize_velocities,
)
```

| API | Availability | Required use |
|---|---|---|
| `FIRE2(...)` | P+T | Live batched geometry relaxation. |
| `ConvergenceHook.from_fmax(...)` | P+T | Per-system force convergence with explicit status transitions. |
| `BaseDynamics.register_hook(...)` | P+T | Attach behavior without wrapping the engine in tutorial-specific code. |
| `BaseDynamics.run(...)` | P+T | Execute a bounded, visible stage. |
| `initialize_velocities(...)` | P+T | Reproducible, mass-aware velocity initialization. |
| `NVTLangevin(...)` | P+T | Short controlled-temperature dynamics. |
| `NVE(...)` | P+T | Energy-conservation and timestep validation. |
| `stage_a + stage_b` / `FusedStage` | P+T | Compose multi-stage workflows with per-system progression. |
| per-graph temperature/timestep/conditions | P+T | Make the batch answer a scientific or numerical question. |

The core notebook should execute relaxation followed by at least one live MD
stage. If NVE is measured, force clamping must be disabled because it destroys
the conservation test.

`initialize_velocities(...)` receives velocity, mass, temperature, and batch
metadata tensors; it does not receive a `Batch` as one opaque argument. Show
that data flow once rather than hiding it in a helper.

## 5. Hooks, safety, and observability — must teach directly

Shared hooks:

```python
from nvalchemi.hooks import (
    BiasedPotentialHook,
    DynamicsContext,
    Hook,
    NeighborListHook,
    WrapPeriodicHook,
)
```

Dynamics hooks:

```python
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    EnergyDriftMonitorHook,
    FreezeAtomsHook,
    LoggingHook,
    MaxForceClampHook,
    NaNDetectorHook,
    SnapshotHook,
)
```

| API | Availability | Curriculum role |
|---|---|---|
| `NeighborListHook` | P+T | Rebuild/update model inputs before compute. |
| `FreezeAtomsHook` | P+T | Enforce a physically stated frozen-region constraint. |
| `WrapPeriodicHook` | P+T | Keep periodic trajectories consistent. |
| `NaNDetectorHook` | P+T | Fail visibly on numerical corruption. |
| `MaxForceClampHook` | P+T | Defensive/recovery mode only; never silently alter the headline result. |
| `EnergyDriftMonitorHook` | P+T | Quantify NVE integration quality. |
| `LoggingHook` | P+T | Produce inspectable per-system CSV observables. |
| `SnapshotHook` | P+T | Save full trajectories or periodic checkpoints. |
| `ConvergedSnapshotHook` | P+T | Save systems as they leave an inflight workflow. |
| `BiasedPotentialHook` | P+T | Add one readable restraint or collective-variable bias. |
| `Hook` + `DynamicsContext` | P+T | Define one small structural hook protocol implementation without subclassing dynamics. |

Profiling migration:

```python
# Existing tutorial pin only
from nvalchemi.dynamics.hooks import ProfilerHook

# Current v2 pin
from nvalchemi.dynamics.hooks import StageTimingHook, TorchProfilerHook
```

`ProfilerHook` is a **P → T rename/split**. V2 should use
`StageTimingHook` for readable per-stage timing and reserve
`TorchProfilerHook` for trace capture. Do not add a compatibility alias to the
learner-facing notebook.

The current v2 pin declares `nvidia-physicsnemo>=2.0.0` and imports
`TorchProfilerHook` through the public hooks package. The pinned H100
environment resolves that dependency. A clean release-image import smoke is
still required; source presence alone is insufficient.

`run()` automatically enters and exits registered context-manager hooks, so a
registered `LoggingHook` is flushed and closed even if the loop raises. Use an
explicit `with LoggingHook(...)` only around a manual `step()` loop; wrapping a
hook that is also registered with `run()` would enter it twice.

## 6. Model composition — must teach directly in Part 1

Canonical imports:

```python
from nvalchemi.models import (
    DFTD3ModelWrapper,
    EwaldModelWrapper,
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
    PMEModelWrapper,
)
```

Two distinct patterns must be named correctly:

1. Independent additive contributions:

   ```python
   combined = short_range_model + dispersion_model
   ```

2. Dependent wiring, such as predicted charges feeding electrostatics:

   ```python
   pipeline = PipelineModelWrapper(
       groups=[
           PipelineGroup(
               steps=[aimnet2, coulomb],
               use_autograd=True,
           )
       ]
   )
   ```

Part 1 uses a focused tutorial `DirectCoulombWrapper` because the public Core
pin exposes periodic Ewald/PME wrappers but not AIMNet's finite-molecule
`simple` all-pairs convention. The wrapper implementation stays in `aux`; its
construction, `charges` dependency, `PipelineGroup(use_autograd=True)`, and
official-calculator parity remain visible. Use
`PipelineStep(model, wire={"output": "required_input"})` only when source and
destination keys genuinely differ.

Learners should understand:

- why dependent charge wiring is different from adding fixed-energy terms;
- how `PipelineGroup(use_autograd=True)` preserves derivative paths;
- why each component may require a different neighbor format/cutoff;
- why D3 parameters must match the reference functional;
- why adding D3 to a checkpoint whose reference already contains nonlocal
  dispersion is double counting;
- when Ewald/PME periodic-image and net-charge conventions affect the claim;
- `EwaldModelWrapper.invalidate_cache()` and
  `PMEModelWrapper.invalidate_cache()` as explicit cache-reset APIs when cell
  state changes outside normal automatic detection.

## 7. Artifacts and replay — must teach directly

Dataset imports:

```python
from nvalchemi.data import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    DataLoader,
    Dataset,
)
from nvalchemi.dynamics import ZarrData
```

| API | Availability | Learner outcome |
|---|---|---|
| `AtomicDataZarrWriter` | P+T | Persist `AtomicData`/`Batch` datasets outside a running dynamics stage. |
| `AtomicDataZarrReader` | P+T | Read structured samples and metadata back. |
| `Dataset` | P+T | Convert stored tensors back into `AtomicData` with device handling. |
| `DataLoader` | P+T | Collate replayed structures into batches. |
| `ZarrData` | P+T | Use a dynamics `DataSink` for snapshots and stage output. |

Every headline result must carry:

- exact input and final structures;
- full short trajectory where dynamics are involved;
- CSV observables and failure/status fields;
- checkpoint and package revisions;
- deterministic seed and calculation settings;
- a manifest linking summary rows to `.extxyz`, Zarr, and plots.

Cache is a classroom recovery path, not the unannounced source of the result.

## APIs to demonstrate once

These matter to the ecosystem but should not all become core exercises.

### Variable-cell and extended ensembles

```python
from nvalchemi.dynamics import (
    FIRE2VariableCell,
    NPH,
    NPT,
    NVTNoseHoover,
)
from nvalchemi.dynamics.hooks import AlignCellHook
```

- Use `FIRE2VariableCell` for a bounded 0 K energy/stress problem.
- Use `NPT` only for a short barostat-response demonstration unless sampling is
  long enough to support a thermodynamic claim.
- Keep `NPH` and `NVTNoseHoover` as discoverable alternatives unless the chosen
  science requires them.

### Inflight processing and sinks

```python
from nvalchemi.dynamics import GPUBuffer, HostMemory, SizeAwareSampler
```

- `SizeAwareSampler.build_initial_batch()` and replacement requests explain how
  large campaigns keep a bounded live batch.
- `GPUBuffer` illustrates device-resident stage handoff.
- `HostMemory` is a simple result sink, not a trajectory-file format.
- Inflight behavior must preserve stable `system_id` values through refill and
  sink round trips before it is used in the core tutorial.

### Additional model outputs

- `MACEWrapper` Hessians for a small vibrational example.
- `compute_embeddings()` for a representation/disagreement atlas.
- stress output for equation-of-state or variable-cell work.
- AIMNet2 `charges`/`spin_charges` for scientifically matched checkpoints.

### Pipeline neighbor adaptation — current v2 pin

```python
PipelineModelWrapper(
    groups=...,
    neighbor_adaptation="auto",  # or "always" / "never"
    max_cutoff_ratio=1.5,
)
```

This **T** policy controls whether composed models share and filter a larger
source neighbor list or build exact lists for different cutoffs/formats. Show
it once in an advanced lesson; do not expose private pipeline internals.

### One direct Toolkit-Ops reduction

```python
from nvalchemiops.torch import segmented_sum
```

Use one transparent example to reduce per-atom values to per-system values.
This explains a recurring batched primitive without turning the tutorial into a
kernel course.

The exact pin also exports `segmented_sum` from
`nvalchemiops.torch.segment_ops`; both paths are public. The shorter top-level
path matches the learner notebook.

## Toolkit-Ops reference surface

The main workflow should normally use Core wrappers. This section identifies
the important accelerated primitives underneath them.

### Neighbors

```python
from nvalchemiops.torch.neighbors import (
    estimate_neighbor_list_costs,
    neighbor_list,
    suggest_neighbor_list_method,
)
```

Explicit `naive_neighbor_list`, `cell_list`, `cluster_tile_neighbor_list`, and
their `batch_*` variants are appendix material.

### Dispersion

```python
from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
```

The public Core `DFTD3ModelWrapper` belongs in the learner workflow; the Ops
call is useful for explaining what is accelerated and for validation.

### Electrostatics

```python
from nvalchemiops.torch.interactions.electrostatics import (
    EwaldParameters,
    PMEParameters,
    compute_slab_correction,
    dsf_coulomb,
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_summation,
    particle_mesh_ewald,
)
```

- Ewald is the transparent small-periodic-system validation path.
- PME is the scalable periodic path.
- DSF is a finite-cutoff alternative, not a drop-in replacement for every
  checkpoint's trained long-range convention.
- Slab correction is conditional on a two-dimensional periodic problem.
- Low-level real/reciprocal helpers, k-vector builders, PME mesh internals, and
  multipole APIs belong in reference material.

### Segmented operations

```python
from nvalchemiops.torch.segment_ops import (
    segmented_dot,
    segmented_matvec,
    segmented_mean,
    segmented_mul,
    segmented_rms_norm,
    segmented_sum,
)
```

Only `segmented_sum` needs a learner-facing example. The rest should be named
as differentiable building blocks. Do not list a `segmented_max_norm`; it is
not exported by the audited Torch or JAX public modules.

## Ecosystem map — mention, do not teach in the core

### JAX parity

The principal Ops imports have JAX mirrors:

```python
from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.interactions.dispersion import D3Parameters, dftd3
from nvalchemiops.jax.interactions.electrostatics import (
    ewald_summation,
    particle_mesh_ewald,
)
from nvalchemiops.jax.segment_ops import segmented_sum
```

Keep JAX in a portability appendix. The workshop core should use one framework
and one environment. JAX compilation requires static shape controls and
`block_until_ready()` for honest timing.

### Distributed execution

```python
from nvalchemi.dynamics import DistributedPipeline
```

The `stage_a | stage_b` operator, fixed communication buffers, `torchrun`, and
multi-rank sinks are ecosystem capabilities, not single-GPU workshop
requirements.

### Training and fine-tuning — target only

Current Toolkit main adds `nvalchemi.training`, including:

- `TrainingStrategy`, `TrainingStage`, and `FineTuningStrategy`
- `ComposedLossFunction` and energy/force/stress loss terms
- `OptimizerConfig`
- `CheckpointHook`, `DDPHook`, `EMAHook`, and `MixedPrecisionHook`
- checkpoint save/load/validation and training CLI surfaces

It also adds `InMemoryDataset`, multi-dataset/sampler features, reporting, and
`UMAWrapper`. These are **T** capabilities. Mention them on the ecosystem map,
but do not make the core tutorial depend on them until the release API,
environment, model access, and license story are stable.

### Raw Toolkit-Ops dynamics and kernels

Raw Warp integrators, FIRE/FIRE2 step functions, explicit neighbor algorithms,
PME internals, multipole kernels, and custom `BaseDynamics` implementations are
developer/reference material. Learners should use Core dynamics classes and
hooks.

### Compilation

`torch.compile`, JAX `jit`, cuEquivariance tuning, and CUDA graph capture are
performance extensions. They are not required for the scientific result, and
cold compilation time must be reported separately from warm execution.

## Public APIs to avoid hiding or misnaming

The v2 notebooks should remove these sources of confusion:

- Do not present local `ToolkitRelaxationConfig`,
  `get_toolkit_relaxation_engine`, `RelaxationEngine`, `.relax()`, or
  `.async_relax()` as Toolkit APIs. They are tutorial helpers.
- Do not import from private modules containing a leading underscore, including
  `nvalchemi.dynamics._ops.*`, `nvalchemi._typing`, or private hook utilities.
- Do not inspect private attributes such as `pipeline._models`, batch storage
  groups, or underscore-prefixed neighbor state.
- Do not retain the current Part 2 monkey-patch of neighbor-list behavior.
- Do not teach deprecated `nvalchemiops.neighborlist` or unqualified
  `nvalchemiops.neighbors` imports. Use the framework namespace:
  `nvalchemiops.torch.*` or `nvalchemiops.jax.*`.
- Do not invoke raw Warp kernels when a public Core wrapper answers the learner
  question.
- Do not use a compatibility shim to make a removed name appear current. Show
  the chosen release API directly.
- Do not use additive `model_a + model_b` when a predicted intermediate such as
  charge must remain on the autograd path; use an explicit dependent pipeline.

## Current notebook coverage

### Part 1 — water interactions to predicted-charge IR

Must expose:

- `AtomicData.from_atoms`, `AtomicData.add_node_property`,
  `AtomicData.add_system_property`, `Batch.from_data_list`, `get_data`,
  `to_data_list`, `index_select`, and batch metadata;
- `AIMNet2Wrapper`, `DFTD3ModelWrapper`, `model_config`, `active_outputs`,
  `set_config`, and explicit checkpoint/D3 hashes;
- `compute_neighbors` for one-shot inference and `make_neighbor_hooks()` for
  relaxation and dynamics;
- `PipelineGroup(use_autograd=True)`, the resulting `PipelineStep` objects,
  `PipelineModelWrapper`, and `neighbor_adaptation="always"` for predicted
  charges → finite Coulomb, followed by pairwise D3;
- `FIRE2`, `ConvergenceHook.from_fmax`, `initialize_velocities`,
  `NVTLangevin`, `NVE`, and `FusedStage`;
- public hook registration, `NaNDetectorHook`, `LoggingHook`, one tutorial
  recorder that structurally implements `Hook` and receives
  `DynamicsContext`, and `segmented_sum`; the compact recorder implementation
  stays in `aux`, while its registration and data path remain visible;
- `ZarrData.zero`, `ZarrData.write`, `ZarrData.read`, raw trajectory
  persistence, replay, per-system extraction, and a checksummed manifest.

Optional extension:

- `SizeAwareSampler` for a larger molecular campaign;
- a VDOS contrast cell to separate density of states from IR activity;
- stress/variable-cell relaxation only in a different, scientifically matched
  model lesson.

### Part 2 — live Toolkit laboratory

Regardless of the final scientific system, it should add:

- a different model/physical wrapper or explicit model composition
- `PipelineModelWrapper` for dependent wiring when appropriate
- `initialize_velocities`, NVT and/or NVE, and `FusedStage`
- `EnergyDriftMonitorHook`, a custom diagnostic or `BiasedPotentialHook`
- `ZarrData`, reader/dataset/dataloader replay
- one direct `nvalchemiops.torch` cell
- a bounded live result whose scientific route is not shortened implicitly

If the ionic-crystal concept is chosen, Part 2 should additionally show
`LennardJonesModelWrapper + EwaldModelWrapper`, variable-cell relaxation, and
Ewald/PME validation. If the coherent-vibration concept is chosen, it should
show MACE Hessians, NVE, mode projection, and timestep/energy-drift comparison.

## Release acceptance checklist

Before calling this API curriculum implemented:

- [ ] Every learner-facing import resolves in the clean release image.
- [x] The current v2 H100 environment imports its declared PhysicsNeMo
      dependency and exact Toolkit pins.
- [x] No Part 1 notebook import uses a private `_` module.
- [x] The Core/Toolkit-Ops pair passes the model, composition, dynamics, and
      Zarr smoke matrix.
- [x] Every Part 1 `active_outputs` choice matches the downstream stage.
- [x] Every Part 1 model supplies its required neighbor list automatically or through
      a visible public hook.
- [x] Predicted-charge electrostatics preserves the full force derivative path.
- [x] D3 functional parameters and long-range conventions match the checkpoint.
- [x] NVE validation runs without force clamping.
- [x] Cold load/compile and warm compute are timed separately with device
      synchronization.
- [ ] Failed/OOM systems retain status, error, and partial artifacts.
- [x] All used model/reference revisions, hashes, licenses, and citations are
      recorded; the D3 tensor stays unbundled pending redistribution review.
- [x] Cached replay is labeled and never substitutes silently for the live
      headline calculation.

## Primary references

- [Toolkit user guide](https://nvidia.github.io/nvalchemi-toolkit/userguide/index.html)
- [Toolkit supported-model matrix](https://nvidia.github.io/nvalchemi-toolkit/models/index.html)
- [Toolkit example gallery](https://nvidia.github.io/nvalchemi-toolkit/examples/index.html)
- [Toolkit API reference](https://nvidia.github.io/nvalchemi-toolkit/modules/index.html)
- [Toolkit-Ops user guide](https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/index.html)
- [Toolkit-Ops API reference](https://nvidia.github.io/nvalchemi-toolkit-ops/modules/index.html)
