# ALCHEMI Toolkit API curriculum

This is the separate list of Toolkit capabilities that the playbook should
teach. It is not a notebook plan and does not record compute-run history. The
owner table below records where deferred APIs belong and whether that lesson is
learner-ready.

The current build targets Toolkit Core 0.2 and Toolkit-Ops 0.4. Verify every
import against the versions pinned in `build/` before release.

## Coverage levels

- **Teach directly:** learners construct or call the API and inspect its output.
- **Show once:** learners see the API in a focused example, but it is not the
  main subject.
- **Reference:** name the capability and link to its documentation without
  expanding the main tutorial.

## Product API map

This map protects broad ecosystem coverage without making every notebook an API
catalogue.

| Product area | Main public namespace | Role in the playbook |
|---|---|---|
| Core data, batching, loading, and persistence | `nvalchemi.data` | Teach directly across the series |
| Built-in model adapters and composition | `nvalchemi.models` | Teach selected paths; reference the rest |
| Hooks, reporting, timing, and profiling | `nvalchemi.hooks` | Teach hooks; show diagnostics once |
| Relaxation, dynamics, inflight work, data sinks, and distributed stage pipelines | `nvalchemi.dynamics` | Teach selected workflows across the series |
| Spatial and domain decomposition | `nvalchemi.distributed` | Teach the high-level path once; reference lower-level adapters |
| Training, fine-tuning, validation, and checkpoints | `nvalchemi.training` | Reference, then teach in a training tutorial |
| Accelerated neighbors and rebuild detection | `nvalchemiops.torch.neighbors`, `nvalchemiops.jax.neighbors` | Teach one framework path; reference variants |
| Differentiable segmented operations | `nvalchemiops.torch`, `nvalchemiops.jax.segment_ops` | Show one reduction; reference the family |
| Dispersion and electrostatics kernels | `nvalchemiops.interactions` and framework bindings | Teach when a model or application needs them |
| Low-level Warp dynamics and optimizers | `nvalchemiops.dynamics` | Reference |
| GTO, harmonic, and B-spline helpers | `nvalchemiops.math` | Reference |

## Minimum learner outcomes

A learner completing the playbook should be able to:

1. convert an atomistic structure to `AtomicData`;
2. assemble and inspect a variable-size `Batch`;
3. inspect a model adapter's inputs, outputs, precision, and neighbor needs;
4. run one batched model evaluation and recover per-system results;
5. attach neighbor, safety, logging, and snapshot hooks;
6. run batched relaxation or dynamics;
7. compose independent or dependent model contributions correctly;
8. connect one external model through Toolkit's public adapter interface;
9. distinguish ordinary batching, inflight processing, domain decomposition,
   and distributed stages; and
10. save results and reload them through the Toolkit data path.

## Essential APIs at a glance

| Capability | Public APIs | Coverage |
|---|---|---|
| One system | `AtomicData`, `AtomicData.from_atoms`, `AtomicData.from_structure` | Teach directly |
| Variable-size batches | `Batch.from_data_list`, `batch_idx`, `batch_ptr`, `num_graphs`, `num_nodes_per_graph` | Teach directly |
| Recover systems | `get_data`, `to_data_list`, `index_select` | Teach directly |
| Data loading | `Dataset`, `InMemoryDataset`, `DataLoader` | Show once |
| Model adapters | `AIMNet2Wrapper`, `MACEWrapper`, `LennardJonesModelWrapper`; `UMAWrapper` in reference | Teach selected wrappers across the playbook |
| Model loading | wrapper `from_checkpoint(...)` methods | Teach directly |
| Model configuration | `model_config`, `active_outputs`, `set_config`, `make_neighbor_hooks` | Teach directly |
| Custom adapters | `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `NeighborListFormat` | Teach directly once |
| Core neighbors | `compute_neighbors`, `NeighborListHook` | Teach directly |
| Ops neighbors | `nvalchemiops.torch.neighbors.neighbor_list` and named algorithms | Part 2 shows the dispatcher once; keep variants in reference |
| Model composition | additive model `+`, `PipelineModelWrapper`, `PipelineGroup`, `PipelineStep` | Teach directly |
| Physical contributions | `DFTD3ModelWrapper`, `EwaldModelWrapper`, `PMEModelWrapper` | Teach when scientifically appropriate |
| Relaxation | `FIRE2`, `ConvergenceHook.from_fmax` | Teach directly |
| Dynamics | `initialize_velocities`, `NVTLangevin`, `NVE`, `FusedStage`, `stage.run(...)` | Teach directly |
| Hooks | `FreezeAtomsHook`, `WrapPeriodicHook`, `NaNDetectorHook`, `EnergyDriftMonitorHook`, `LoggingHook`, `SnapshotHook`, `ConvergedSnapshotHook` | Teach selected hooks across the playbook; Part 3 owns snapshot hooks |
| Reporting and profiling | `StageTimingHook`, `TorchProfilerHook`, `Reporter`, `RichReporter`, `TensorBoardReporter` | Show once or reference |
| Inflight work | `SizeAwareSampler`, `GPUBuffer`, `HostMemory`, `FusedStage` | Part 1 teaches the queue; Part 3 owns `GPUBuffer` |
| Distributed stages | `DistributedPipeline`, `BufferConfig` | Show public construction once |
| Domain parallelism | `DistributedManager.initialize`, manager `rank` / `world_size` / `device`, `manager.initialize_mesh`, `DomainConfig`, optional `SpatialPartitioner` layout preview, `DomainParallel`, `partition`, `run`, `gather`, `DistributedManager.cleanup` | Teach the high-level path once |
| Domain internals | `DistributedModel`, `DistributedPipelineModel`, `ShardedBatch`, halo and storage helpers | Reference |
| Saved results | `ZarrData`, `AtomicDataZarrWriter`, `AtomicDataZarrReader` | Part 1 teaches `ZarrData`; Part 3 owns the reader and writer |
| Segmented reductions | `segmented_sum` | Show once |
| Training and fine-tuning | `TrainingStrategy`, `TrainingStage`, `FineTuningStrategy`, loss, optimizer, validation, and checkpoint APIs | Reference |

## Owners for deferred essential APIs

Part 1 remains focused on its current seven-stage progression. The following
APIs belong to later active tutorials. A name appearing in a reference example
does not make Part 1 the teaching owner.

| Essential API | Curriculum owner | Current status | Required learner-facing use |
|---|---|---|---|
| `SnapshotHook`, `ConvergedSnapshotHook` | Part 3: OLED melting-point workflow | `SnapshotHook` is present in the current notebook. The Part 3 remaster and validation with the current Toolkit versions are still pending; `ConvergedSnapshotHook` remains planned. | Register the hook visibly, save selected dynamics or completed-system states, and inspect what was written. |
| `nvalchemiops.torch.neighbors.neighbor_list(...)` | Part 2: batched adsorption | Planned for the Part 2 remaster; not yet learner-ready or validated with the current Toolkit versions. | Call the automatic dispatcher on a materials batch, inspect the returned neighbor representation, and explain why the dispatcher was selected instead of a named low-level algorithm. |
| `GPUBuffer` | Part 3: OLED melting-point workflow | Planned for the Part 3 remaster; not yet learner-ready or validated with the current Toolkit versions. | Use it in a bounded inflight workflow and compare its device-resident role with `HostMemory` and distributed-stage transfer buffers. |
| `AtomicDataZarrReader`, `AtomicDataZarrWriter` | Part 3: OLED melting-point workflow | `AtomicDataZarrReader` is already used by a Part 3 helper. A direct writer-and-reader lesson and validation with the current Toolkit versions are still pending. | Write complete systems with stable IDs, reload one system and a batch, and repeat analysis without rerunning the expensive simulation. |

## 1. Data and batching

Core imports:

```python
from nvalchemi.data import AtomicData, Batch
```

Teach these operations directly:

| API | Learner outcome |
|---|---|
| `AtomicData.from_atoms(...)` | Convert ASE structures without handwritten coordinate blocks. |
| `AtomicData.from_structure(...)` | Convert pymatgen structures through the public path. |
| `add_node_property(...)` | Add per-atom values such as masses, velocities, or categories. |
| `add_system_property(...)` | Add graph-level values such as charge, multiplicity, or stable ID. |
| `AtomicData.to(...)` | Move data across device or dtype boundaries explicitly. |
| `data.use_default_masses()`, `data.use_default_categories()` | Add standard atom properties without tutorial-local lookup tables. |
| `data.use_default_velocities()` | Add zero velocities. This does not sample a thermal distribution; use `initialize_velocities(...)` for temperature-based initialization. |
| `data.chemical_hash` | Fingerprint atomic numbers, geometry, PBC, and cell for one structure state; it is not stable across relaxation or dynamics. |
| `Batch.from_data_list(...)` | Combine independent systems in one model call. |
| `batch_idx` and `batch_ptr` | Map atoms to systems and identify segment boundaries. |
| `num_graphs` and `num_nodes_per_graph` | Inspect the actual batch shape. |
| `get_data(i)` and `to_data_list()` | Recover individual systems and outputs. |
| `index_select(...)` | Select converged, failed, or scientifically interesting systems. |

Show important fields when they matter: `positions`, `atomic_numbers`, `cell`,
`pbc`, `energy`, `forces`, `stress`, `charge`, `mult`, and `velocities`.
Workflow code may add status, labels, or stable IDs as system properties; do
not present them as fixed built-in schema fields.

Batching must answer a real question. Do not teach it only as a tensor-shape or
speed exercise.

### Fixed-capacity batch operations in the pinned release

Core 0.2 exposes `Batch.empty(...)`, `batch.add_key(...)`, `batch.put(...)`,
`batch.defrag()`, and `batch.zero()` for fixed-capacity active batches. Keep
these calls in the reference for the pinned build: the current segmented
`put`/`defrag` path handles float32 fields but can skip required integer fields
such as `atomic_numbers` and `status`. Do not present that path as a complete
learner workflow and do not patch around the limitation. Re-evaluate it against
a later public release before teaching it directly.

## 2. Model adapters and configuration

Selected built-in adapters:

```python
from nvalchemi.models import (
    AIMNet2Wrapper,
    LennardJonesModelWrapper,
    MACEWrapper,
)
```

Load supported published checkpoints through the wrapper's public constructor,
for example `AIMNet2Wrapper.from_checkpoint(...)` or
`MACEWrapper.from_checkpoint(...)`, rather than rebuilding a wrapper around a
private model object.

For adapters around third-party learned models, Toolkit owns the wrapper API.
The model implementation, checkpoint, training data, and their license terms
come from the external model project. Present those sources separately.

The learner should inspect rather than assume:

- `model.model_config`;
- available and active outputs;
- required and optional inputs;
- precision and device;
- cutoff, neighbor format, and half-list or full-list convention;
- charge, spin, cell, and periodic support.

Use `set_config("active_outputs", {...})` to request only the values required by
the calculation. Use `make_neighbor_hooks()` for iterative workflows when the
adapter or composed model should supply its compatible neighbor hooks.

A custom adapter lesson should expose:

```python
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
```

The adapter must declare its capabilities, translate Toolkit inputs, make one
native model call, and map supported outputs back to Toolkit names. Compare its
energy and forces with the native model before using it downstream.

A concrete `BaseModelMixin` subclass provides `model_config`,
`embedding_shapes`, and `compute_embeddings(...)`. An external adapter normally
implements `adapt_input(...)` and `adapt_output(...)` for its data mapping. Use
`direct_derivative_keys()` only when the model itself returns analytical forces
or stress inside an autograd composition group.

## 3. Neighbor lists

### Core workflow

Use the Core dispatcher for a one-shot evaluation:

```python
from nvalchemi.neighbors import compute_neighbors

compute_neighbors(batch, config=model.model_config.neighbor_config)
```

Use model-provided or explicit hooks for relaxation and dynamics:

```python
for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)
```

```python
from nvalchemi.hooks import NeighborListHook
```

### Toolkit-Ops variants

The ordinary learner path should use the high-level dispatcher:

```python
from nvalchemiops.torch.neighbors import neighbor_list
```

Keep the complete variant map in this reference:

| Workload | Public Toolkit-Ops API | Main distinction |
|---|---|---|
| Automatic selection | `neighbor_list(...)` | Selects a supported implementation from geometry and options. |
| One-system naive | `naive_neighbor_list(...)` | Direct all-pairs search. |
| Batched naive | `batch_naive_neighbor_list(...)` | Independent contiguous systems with batch metadata. |
| One-system cell list | `cell_list(...)` | Spatial binning for larger systems. |
| Batched cell list | `batch_cell_list(...)` | Cell-list search across independent systems. |
| One-system cluster tile | `cluster_tile_neighbor_list(...)` | CUDA-oriented tiled search with stricter layout and option requirements. |
| Batched cluster tile | `batch_cluster_tile_neighbor_list(...)` | Tiled search for contiguous batched systems. |
| One-system dual cutoff | `naive_neighbor_list_dual_cutoff(...)` | Produces naive lists at two cutoffs. |
| Batched dual cutoff | `batch_naive_neighbor_list_dual_cutoff(...)` | Batched two-cutoff naive search. |
| Method planning | `suggest_neighbor_list_method(...)`, `estimate_neighbor_list_costs(...)` | Explains or estimates dispatcher choices. |
| Capacity estimation | `estimate_cell_list_sizes(...)`, `estimate_batch_cell_list_sizes(...)` | Estimates fixed capacities for cell-list execution. |
| Rebuild detection | `neighbor_list_needs_rebuild(...)`, `batch_neighbor_list_needs_rebuild(...)`, `cell_list_needs_rebuild(...)`, `batch_cell_list_needs_rebuild(...)` | Checks whether moved atoms require neighbor reconstruction. |
| Inline pair function | `CompiledPairFn`, `compile_pair_fn(...)` | Compiles a Warp pair function for fixed-shape pair energy or force evaluation during neighbor enumeration; these pair outputs are forward-only. |

Teach these inputs and outputs once: positions, cell, PBC, cutoff, batch
metadata, matrix versus COO output, periodic shifts, fill values, and the
dispatcher options `target_indices`, `return_distances`, `return_vectors`,
`rebuild_flags`, and `pair_fn` when they are relevant.

The rebuild checks are available from
`nvalchemiops.torch.neighbors.rebuild_detection`. Keep them in the reference
unless an iterative workflow needs to explain when neighbor data is reused.

Do not imply that one algorithm is universally fastest. Compare methods only
with equal geometry, cutoff, output format, precision, and requested work.

## 4. Model composition

Selected public APIs:

```python
from nvalchemi.models import (
    DFTD3ModelWrapper,
    EwaldModelWrapper,
    PMEModelWrapper,
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
)
```

Teach two distinct patterns:

```python
combined = short_range_model + independent_contribution
```

```python
pipeline = PipelineModelWrapper(
    groups=[PipelineGroup(steps=[producer, consumer], use_autograd=True)]
)
```

The first adds independent contributions. In the second,
`PipelineStep(..., wire=...)` and step order describe data dependencies.
`PipelineGroup(..., use_autograd=True)` sums the group energies and derives
forces or stress from that sum.

Show `PipelineStep(..., wire=...)` only when source and destination keys differ.
Introduce `neighbor_adaptation` only after learners understand why composed
models may require different cutoffs or neighbor formats.

## 5. Relaxation, dynamics, and hooks

Core workflow APIs:

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

Teach:

- live batched `FIRE2` relaxation;
- per-system convergence through `ConvergenceHook.from_fmax(...)`;
- mass-aware, seeded velocity initialization;
- at least one controlled-temperature stage and one conservation-oriented
  stage when the science requires both;
- `FusedStage` as compatible dynamics methods sharing one active `Batch` and
  one model evaluation per step, with each system's status selecting its
  current method;
- hook registration through the public dynamics interface.

`FusedStage` takes the shared model from its first sub-stage. The `+` operator
does not switch models; fused sub-stages must be intended to use the same model
and device path.

Important hooks include:

```python
from nvalchemi.hooks import BiasedPotentialHook, Hook, NeighborListHook, WrapPeriodicHook
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    EnergyDriftMonitorHook,
    FreezeAtomsHook,
    LoggingHook,
    NaNDetectorHook,
    SnapshotHook,
)
```

Use these public APIs when teaching a custom hook, reporting, or profiling:

```python
from nvalchemi.hooks import (
    DynamicsContext,
    Hook,
    Reporter,
    RichReporter,
    StageTimingHook,
    TensorBoardReporter,
    TorchProfilerHook,
)
```

Show the reporting and profiling APIs once in a diagnostics or performance
lesson rather than requiring them in every dynamics example.

Use hooks to teach one responsibility at a time: neighbors, constraints,
periodic wrapping, failure detection, conservation checks, logging, or saved
states. Do not hide the registration point.

## 6. Inflight and distributed execution

Inflight APIs:

```python
from nvalchemi.dynamics import GPUBuffer, HostMemory, SizeAwareSampler
```

`SizeAwareSampler` builds and refills active batches within graph, atom, and
edge limits. Learners should track stable IDs as systems converge, leave the
active batch, and are replaced.

Distributed APIs:

```python
from nvalchemi.dynamics import DistributedPipeline
from nvalchemi.dynamics.base import BufferConfig
```

Keep the distinction explicit:

- `FusedStage` shares a batch and model evaluation across compatible methods;
  systems select their current method through status and may advance at
  different times;
- `stage_a | stage_b` creates distributed stages with data transfer between
  workers;
- `GPUBuffer` and `HostMemory` collect results; they are not the communication
  buffers between distributed stages.

Teach the public construction with a small example. Scaling claims require a
separate equal-work benchmark that includes setup, fill, drain, and collection.

For the classic stage pipeline, teach these behaviors explicitly:

- each dynamics stage runs on one rank;
- the launch world size matches the stage count;
- converged systems move between adjacent stages;
- `pipeline.run()` takes no input batch;
- the normal sequence is construct the stages, enter the context, call
  `pipeline.run()`, and close; it has no `partition` or `gather` call;
- `BufferConfig` configures communicating stages;
- `synchronized=True` is a pipeline option that adds a per-step barrier for
  debugging; and
- each communicating dynamics stage sets its own `comm_mode`: `sync`,
  `async_recv`, or `fully_async`. The Toolkit 0.2 default is `async_recv`; do not
  pass `comm_mode` to `DistributedPipeline`.

This is pipeline parallelism. It does not automatically split one model
evaluation across GPUs.

Do not confuse a distributed stage pipeline with domain parallelism. `DistributedPipeline`
moves independent batches through workflow stages. `DomainParallel` partitions
one large periodic system across devices.

Teach the high-level domain path once:

```python
from nvalchemi.distributed import DistributedManager, DomainConfig, DomainParallel

DistributedManager.initialize()
manager = DistributedManager()
mesh = manager.initialize_mesh(
    mesh_shape=(manager.world_size,),
    mesh_dim_names=("domain",),
)
full_batch = build_batch(manager.device) if manager.rank == 0 else None

config = DomainConfig(cutoff=cutoff, skin=skin, mesh=mesh)
with DomainParallel(dynamics=inner, config=config, n_steps=n_steps) as run:
    local_batch = run.partition(full_batch)
    local_batch = run.run(local_batch)
    total_energy = local_batch.energy.detach().clone()
    full_result = run.gather(local_batch, dst=0)

DistributedManager.cleanup()
```

`DomainParallel` selects the distributed model adapter, spatial partitioner,
sharded storage, halo exchange, and output consolidation needed by supported
built-in wrappers. Learners should not reproduce that work with a custom
distributed loop. A world size of one follows the same high-level calls but
does not partition the system. In Toolkit 0.2, total energy is globally
reduced and remains on each local result; `gather` reconstructs atom fields,
including forces, on the destination rank. The Stage 7 example is neutral
because the input system charge is not copied into each GPU region. Charged
periodic systems need explicit supported charge handling.

`DomainConfig.grid_dims` is a spatial cell-grid override, not a rank layout.
Leave it as `None` to let the public `SpatialPartitioner` derive
`cells_per_dim` and `rank_grid` from the cell, cutoff, and rank count.

For a composed AIMNet2 checkpoint base, periodic PME, and D3 calculation in the
Toolkit 0.2 source used by the lesson:

- put AIMNet2 and PME in one two-step `PipelineGroup` with
  `use_autograd=True`;
- put D3 in a separate direct-force group;
- keep the same compile setting across the one- and multi-GPU checks; the
  current lesson uses `compile=False`;
- pass the PME mesh dimensions returned for the requested accuracy,
  real-space cutoff, safety factor, and cell, then record the dimensions and
  resulting spacing;
- use separate, right-sized neighbor lists when component cutoffs differ
  substantially;
- use the maximum component cutoff for `DomainConfig`;
- add the Toolkit-tested D3 coordination-number margin to the halo, then
  accept it only after the force-parity checks pass;
- verify one- versus multi-GPU energy and force agreement on a smaller system;
- record natural out-of-memory failures without manufacturing them; and
- state that the Toolkit 0.2 version used here replicates the full charge mesh
  and FFT work on each rank.

Core 0.2 can also combine pipeline and domain dimensions with a two-dimensional
device mesh. Keep that combined layout in the reference unless a later lesson
needs both forms of parallelism at once.

## 7. Saved results and replay

Teach at least one Toolkit-native save and reload path:

```python
from nvalchemi.data import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    DataLoader,
    Dataset,
    InMemoryDataset,
)
from nvalchemi.dynamics import ZarrData
```

The learner should be able to:

- save complete per-system outputs with stable IDs;
- reload one system and a batch;
- rerun analysis without repeating the expensive calculation;
- connect saved data to `Dataset`, `InMemoryDataset`, and `DataLoader` where
  appropriate.

Also save ordinary inspectable outputs such as CSV tables and `.extxyz`
structures when they are the most useful interchange formats.

## 8. One direct Toolkit-Ops reduction

Show the common batched reduction explicitly:

```python
from nvalchemiops.torch import segmented_sum
```

Use it to reduce per-atom values to one value per system. Mention, but do not
teach in the main path, the related public operations:

- `segmented_dot`
- `segmented_matvec`
- `segmented_mean`
- `segmented_mul`
- `segmented_rms_norm`

For the pinned PyTorch API, `idx` is a one-dimensional `int32` segment map. The
bindings cover scalar, 3-vector, and, where the operation requires it, 3-by-3
matrix data. They support first- and second-order PyTorch gradients and work
with `torch.compile`.

## 9. Reference-only capabilities

Keep these discoverable without crowding the core tutorials:

- JAX mirrors of neighbors, dispersion, electrostatics, and segmented
  operations;
- `UMAWrapper` and model-specific construction or secondary outputs;
- additional ensembles and variable-cell dynamics;
- reporting and profiling through `StageTimingHook`, `TorchProfilerHook`, and
  the `Reporter` implementations;
- low-level Ewald, PME, DSF, multipole, and slab-correction kernels;
- raw Warp integrators and custom dynamics implementations;
- compilation tuning, CUDA graphs, and detailed profiler integration;
- lower-level domain adapters and two-dimensional pipeline/domain meshes; and
- training and fine-tuning APIs.

### Domain parallelism

```python
from nvalchemi.distributed import (
    DomainConfig,
    DomainParallel,
    DistributedModel,
    DistributedPipelineModel,
    ShardedBatch,
    SpatialPartitioner,
)
```

`DomainConfig` and `DomainParallel` are taught through the public Part 1 call
sequence. `SpatialPartitioner` appears only as an optional read-only layout
preview. `DistributedModel`, `DistributedPipelineModel`, and `ShardedBatch`
remain reference material for custom distributed model work. They are distinct
from `nvalchemi.dynamics.DistributedPipeline`, which streams work through
stages.

### Training and fine-tuning

```python
from nvalchemi.training import (
    CheckpointHook,
    ComposedLossFunction,
    FineTuningStrategy,
    OptimizerConfig,
    TrainingStage,
    TrainingStrategy,
    ValidationLoop,
    load_checkpoint,
    save_checkpoint,
)
```

The product map should name this capability group even when the current
workshop focuses on inference and simulation. Teach it in a separate training
tutorial rather than reducing it to one unexplained call.

Move one of these into a tutorial when it serves a stated product-learning
outcome and can be checked on the intended runtime.

## API rules for tutorial authors

- Use public Core APIs for the main workflow and direct Toolkit-Ops calls only
  when they teach an accelerated primitive.
- Do not import private modules or inspect underscore-prefixed fields.
- Do not present tutorial-local classes as Toolkit APIs.
- Do not use compatibility shims or runtime patches to make removed names look
  current.
- Do not replace an intended Toolkit pipeline with an ad-hoc multiprocessing or
  distributed loop.
- Use the framework namespace, such as `nvalchemiops.torch.*` or
  `nvalchemiops.jax.*`.
- Keep model outputs singular or plural exactly as declared by the API.
- Verify every learner-facing import and example in the pinned clean image.

## Official references

- [Toolkit user guide](https://nvidia.github.io/nvalchemi-toolkit/userguide/index.html)
- [Toolkit supported models](https://nvidia.github.io/nvalchemi-toolkit/models/index.html)
- [Toolkit examples](https://nvidia.github.io/nvalchemi-toolkit/examples/index.html)
- [Toolkit API reference](https://nvidia.github.io/nvalchemi-toolkit/modules/index.html)
- [Toolkit-Ops user guide](https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/index.html)
- [Toolkit-Ops API reference](https://nvidia.github.io/nvalchemi-toolkit-ops/modules/index.html)
