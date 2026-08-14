# ALCHEMI Toolkit API reference

This document records the public Toolkit and Toolkit-Ops capabilities relevant
to the tutorial series. The [ALCHEMI tutorial guide](TUTORIAL_GUIDE.md) controls
teaching order, lesson design, visuals, and review. Review issues or pull
requests record implementation decisions and compute history.

Verify every learner-facing import and example against the immutable versions
in [`environment/runtime-pins.toml`](environment/runtime-pins.toml) and the
frozen environment described in [`environment/README.md`](environment/README.md).

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

### Framework relationship

Toolkit Core stores `AtomicData`, `Batch`, model inputs, and workflow state as
PyTorch tensors. Toolkit-Ops supplies selected atomistic operations through
PyTorch bindings and an optional JAX binding. Many Toolkit-Ops implementations
dispatch Warp kernels on CPU or CUDA while preserving the calling framework's
array and differentiation interface.

The frozen v3 environment installs the Toolkit-Ops Torch extra. Part 01 shows
this relationship as a diagram. Executable JAX examples belong in a separately
pinned JAX environment.

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
| Ops neighbors | `nvalchemiops.torch.neighbors.neighbor_list` and named algorithms | Reference in the fundamentals; teach with performance work when useful |
| Model composition | additive model `+`, `PipelineModelWrapper`, `PipelineGroup`, `PipelineStep` | Teach directly |
| Physical contributions | `DFTD3ModelWrapper`, `EwaldModelWrapper`, `PMEModelWrapper` | Teach when scientifically appropriate |
| Relaxation | `FIRE2`, `ConvergenceHook.from_fmax` | Teach directly |
| Dynamics | `BaseDynamics`, `initialize_velocities`, `NVTLangevin`, `NVE`, `FusedStage`, `stage.run(...)` | Teach `BaseDynamics` in Part 05; extend to staged MD in Part 06 |
| Hooks | `Hook`, `DynamicsContext`, `DynamicsStage`, `FreezeAtomsHook`, `WrapPeriodicHook`, `NaNDetectorHook`, `EnergyDriftMonitorHook`, `LoggingHook`, `SnapshotHook`, `ConvergedSnapshotHook` | Teach the protocol in Part 04; reuse selected hooks later |
| Reporting and profiling | `StageTimingHook`, `TorchProfilerHook`, `Reporter`, `RichReporter`, `TensorBoardReporter` | Show in Part 06 or reference |
| Inflight work | `SizeAwareSampler`, `GPUBuffer`, `HostMemory`, `FusedStage` | Teach in Part 06 |
| Distributed stages | `DistributedPipeline`, `BufferConfig` | Show public construction once |
| Domain parallelism | `DistributedManager.initialize`, manager `rank` / `world_size` / `device`, `manager.initialize_mesh`, `DomainConfig`, optional `SpatialPartitioner` layout preview, `DomainParallel`, `partition`, `run`, `gather`, `DistributedManager.cleanup` | Teach the high-level path once |
| Domain internals | `DistributedModel`, `DistributedPipelineModel`, `ShardedBatch`, halo and storage helpers | Reference |
| Saved results | `ZarrData`, `AtomicDataZarrWriter`, `AtomicDataZarrReader` | Teach record persistence in Part 02; reuse workflow sinks in Part 06 |
| Segmented reductions | `segmented_sum` | Reference from the Part 01 framework diagram; teach directly when the reduction supports a lesson question |
| Training and fine-tuning | `TrainingStrategy`, `TrainingStage`, `FineTuningStrategy`, loss, optimizer, validation, and checkpoint APIs | Teach in Part 07 |

Hooks also connect models to workflows. `model.make_neighbor_hooks()` returns
the neighbor preparation required by a single or composed model, and hook
contexts expose the active model around evaluation. Part 04 therefore connects
to both model use and simulation workflows.

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
| `add_node_property(...)` | Do not teach at this pin; see the mutation note below. Use `Batch.add_key(..., level="node")` for custom per-atom values. |
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

`AtomicData.__eq__` returns `chemical_hash` equality. Two properties of the
pinned implementation decide when that is the right check:

- The comparison is exact, not numerical. `chemical_hash` formats sorted atomic
  numbers and positions as text and applies BLAKE2s, so any change the tensor
  actually stores produces a different hash. The resolution is the dtype's, not
  the formatter's: at `float32` with coordinates near 2 Å the spacing between
  representable values is 2.38e-7 Å, so a 1e-7 Å shift is discarded on write and
  the hash is unchanged. It changes from roughly 3e-7 Å upward at `float32`, and
  from 1e-8 Å at `float64`. Use `==` for structural identity such as a save and
  reload round trip, and `torch.allclose(...)` whenever the question is whether
  values agree within tolerance.
- The hash covers absolute coordinates, with no translation or rotation
  invariance. A rigid 1 Å shift of an otherwise identical structure hashes
  unequal while every interatomic distance is preserved.
- `chemical_hash` is a plain property recomputed from the current tensors on
  every read. There is no cache and no invalidation call, so an in-place edit to
  `positions` changes the next comparison immediately.
- The docstring claims invariance to atom ordering, but the implementation sorts
  by atomic number alone. Atoms that share an atomic number keep their input
  order, so the same molecule listed in a different order hashes unequal.
  Verified against Toolkit commit `8c2c307c…`; do not teach order invariance.

`chemical_hash` moves tensors with `.cpu()` before hashing, so a CPU and CUDA
pair of the same structure hashes equal and `==` will not detect a device
difference.

Mutating an `AtomicData` object in place behaves differently per route at commit
`8c2c307c…`. Verified by execution on CPU:

- Assigning a declared field such as `positions` re-runs the field and model
  validators. A shape or dtype mismatch is caught at the field level and the
  receiver is left unchanged, which makes it safe to demonstrate. A transposed
  coordinate block is the clearest example.
- A **row-count** mismatch is not transactional. The validator raises, but the
  bad tensor has already been written, so `chemical_hash` then raises
  `IndexError` and the object stays unusable until a correctly sized tensor is
  assigned. Never demonstrate this on a live object a later cell depends on.
  The message can also name a field the caller never touched, because the
  validator walks an unordered set.
- `add_system_property(...)` is the sound path for graph-level values. The key
  lands in the model, appears in `model_dump()` and `system_properties`, survives
  `clone()` and `to()`, and `Batch.from_data_list` packs one row per graph. Shape
  is unchecked for a new key, and a repeated key overwrites without warning.
- `add_node_property(...)` writes through `object.__setattr__`, so it validates
  nothing and its `node_dim` argument is ignored. The key is invisible to
  `model_dump()`, `node_properties`, `clone()`, `to()`, and
  `Batch.from_data_list`, so every copy drops it.
- `add_edge_property(...)` does not enforce an edge count in practice, because
  the key joins the edge keys only after the validated assignment.
- `del data.charge` succeeds silently and leaves the field unreadable while the
  system keys still list it.
- At `float64` on CPU the tensor returned by `from_atoms` can share memory with
  the source ASE positions, so an in-place edit moves the ASE object too. At
  `float32` the dtype conversion forces a copy and the ASE object is untouched.

Show important fields when they matter: `positions`, `atomic_numbers`, `cell`,
`pbc`, `energy`, `forces`, `stress`, `charge`, `mult`, and `velocities`.
Workflow code may add status, labels, or stable IDs as system properties; do
not present them as fixed built-in schema fields.

Batching must answer a real question. Do not teach it only as a tensor-shape or
speed exercise.

### Complete `Batch` interface in the pinned release

The pinned `Batch` class exposes the following public interface. The fundamentals
lesson teaches the inspection and selection rows. Later lessons own preallocated
buffers and distributed communication.

| Area | Public interface | Purpose |
|---|---|---|
| Storage constructor | `Batch(device=..., storage=..., keys=...)` | Initialize graph-aware storage directly; ordinary data paths use the class methods below. |
| Construct from objects | `Batch.from_data_list(...)` | Pack validated `AtomicData` graphs. |
| Construct from tensors | `Batch.from_raw_dicts(...)` | Build directly from tensor dictionaries in a data pipeline. |
| Allocate storage | `Batch.empty(...)`, `Batch.empty_like(...)` | Create fixed-capacity storage for inflight work. |
| Basic size | `num_graphs`, `batch_size`, `num_nodes`, `num_edges`, `max_num_nodes` | Report system, atom, and edge counts. |
| Per-graph size | `num_nodes_list`, `num_edges_list`, `num_nodes_per_graph`, `num_edges_per_graph` | Report each graph's segment length as Python lists or tensors. |
| Packed layout | `batch_idx`, `batch_ptr`, `edge_ptr` | Map atom rows to graphs and locate node or edge segments. |
| Stored schema | `keys`, `device`, `system_capacity` | Report field levels, tensor device, and allocated system capacity. |
| Python indexing | `batch["positions"]`, `batch[0]`, `batch[1:4]`, `batch[[0, 2]]` | Return a stored tensor, one `AtomicData`, or a selected `Batch`. |
| Explicit recovery | `get_data(i)`, `to_data_list()` | Reconstruct one or all `AtomicData` graphs. |
| Explicit selection | `index_select(...)` | Build a batch from selected graph indices. |
| Add or combine data | `add_key(...)`, `append(...)`, `append_data(...)` | Add level-aware fields or concatenate compatible data. |
| Mapping behavior | `len(batch)`, `repr(batch)`, `key in batch`, iteration, string assignment and deletion | Inspect or update stored fields through Python protocols. |
| Device and memory | `to(...)`, `cpu()`, `cuda()`, `clone()`, `contiguous()`, `pin_memory()` | Move, copy, or prepare tensor storage. |
| Fixed-capacity buffers | `put(...)`, `defrag(...)`, `trim(...)`, `zero()` | Fill, compact, extract, or reset preallocated batches. |
| Serialization | `model_dump(...)` | Return tensors and batch metadata as a flat dictionary. |
| Distributed transfer | `send(...)`, `recv(...)`, `isend(...)`, `irecv(...)` | Transfer batches between distributed ranks. |

`Batch` also supports attribute access to stored tensors, such as
`batch.positions`. Part 01 should print a compact subset of the public
properties and use bracket indexing directly. Part 06 introduces buffer
operations; Part 08 introduces distributed transfer.

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
hooks = [*model.make_neighbor_hooks(), history_hook]
dynamics = FIRE2(model, dt=0.01, n_steps=20, hooks=hooks)
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
    BaseDynamics,
    ConvergenceHook,
    FIRE2,
    FusedStage,
    NVE,
    NVTLangevin,
    initialize_velocities,
)
```

Teach:

- the `BaseDynamics` lifecycle and its required and provided Batch fields;
- live batched `FIRE2` relaxation;
- per-system convergence through `ConvergenceHook.from_fmax(...)`;
- mass-aware, seeded velocity initialization;
- at least one controlled-temperature stage and one conservation-oriented
  stage when the science requires both;
- `FusedStage` as compatible dynamics methods sharing one active `Batch` and
  one model evaluation per step, with each system's status selecting its
  current method;
- hook registration through workflow constructors.

`FusedStage` takes the shared model from its first sub-stage. The `+` operator
does not switch models; fused sub-stages must be intended to use the same model
and device path.

One fused iteration follows an ordered loop:

1. visit each sub-stage in order and apply its masked `pre_update`;
2. call the shared model once on the full active `Batch`;
3. visit each sub-stage in order and apply its masked `post_update`; and
4. evaluate stage completion and update each molecule's status for the next
   iteration.

A molecule remains in the shared model call after it advances to a later stage.
Its current status selects one sub-stage update and skips the stages it already
completed. A molecule at `exit_status` receives no further sub-stage update.
With an inflight sampler, the next refill writes completed molecules to a sink,
keeps unfinished molecules, and inserts new dataset samples within the available
graph, atom, and edge limits.

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

### Convergence detector versus registered hook in Toolkit 0.2

These facts are release-constrained to `nvalchemi-toolkit==0.2.0` at commit
`8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`:

- `BaseDynamics(..., convergence_hook=detector)` stores `detector` separately
  from the `HookRegistryMixin` registry. After registered
  `DynamicsStage.AFTER_STEP` hooks run, `BaseDynamics.step(...)` calls
  `detector.evaluate(batch)` directly on every step and uses the returned graph
  indices for its convergence result, `DynamicsStage.ON_CONVERGE` dispatch, and
  all-graphs early exit in `run(...)`. This direct path does not use the
  detector's `stage` or `frequency`, and `evaluate(...)` does not read
  `source_status` / `target_status` or migrate `batch.status`.
- `BaseDynamics(..., hooks=[hook])` and `register_hook(hook)` attach `hook` to
  registry lifecycle dispatch. A `ConvergenceHook` declares
  `stage=DynamicsStage.AFTER_STEP`; the registry calls its `__call__(ctx, stage)`
  when `step_count % hook.frequency == 0`, including step 0.
  `ConvergenceHook.__call__` evaluates the criteria and migrates only converged
  graphs whose `batch.status` equals `source_status`, and only when both
  `source_status` and `target_status` are set. Its result is not returned to the
  host convergence path, so a registered `ConvergenceHook` alone does not drive
  `ON_CONVERGE` dispatch or `BaseDynamics.run(...)` early exit.
- One `ConvergenceHook` object implements both surfaces and can be supplied in
  either or both constructor arguments, but the roles remain independent. If
  the same object is supplied to both, registry dispatch and direct
  `evaluate(...)` are separate evaluations on matching steps. `FusedStage`
  auto-registers a separate, frequency-1 status-migrating `ConvergenceHook` at
  `AFTER_STEP` for sub-stage transitions, copying configured detector criteria
  when present.

Part 04 previews the two attachment points while teaching registry stages,
frequency, context, and lifecycle. Part 05 owns detector timing and convergence
inside the relaxation workflow. See the official
[convergence guide](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/convergence.html),
[core hooks guide](https://nvidia.github.io/nvalchemi-toolkit/modules/hooks.html),
[`BaseDynamics` reference](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/_generated/nvalchemi.dynamics.BaseDynamics.html),
and
[`ConvergenceHook` reference](https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/_generated/nvalchemi.dynamics.ConvergenceHook.html).

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
including forces, on the destination rank. System-level `charge` and `spin`
are replicated into local regions by the pinned sharded-batch implementation.
The Stage 7 example remains neutral because it does not validate
model-specific charged periodic semantics. Charged systems require explicit
single-/multi-rank parity validation for the complete composed model.

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

`DomainConfig` and `DomainParallel` are taught through the public Part 08 call
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

## Authoring use

Use the [tutorial guide](TUTORIAL_GUIDE.md) when turning this reference into a
lesson. The guide defines API visibility, helper boundaries, pacing, visuals,
and validation. This reference supplies technical names and relationships.

## Official references

- [Toolkit user guide](https://nvidia.github.io/nvalchemi-toolkit/userguide/index.html)
- [Toolkit supported models](https://nvidia.github.io/nvalchemi-toolkit/models/index.html)
- [Toolkit examples](https://nvidia.github.io/nvalchemi-toolkit/examples/index.html)
- [Toolkit API reference](https://nvidia.github.io/nvalchemi-toolkit/modules/index.html)
- [Toolkit-Ops user guide](https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/index.html)
- [Toolkit-Ops API reference](https://nvidia.github.io/nvalchemi-toolkit-ops/modules/index.html)
