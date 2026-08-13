# N06 worklog — GPU pipelines and profiling

## 2026-08-12 — initial curriculum brief

Status: planned

- Outcome: build a multi-stage workflow on one GPU, follow data between stages,
  compile compatible work, and profile the path with reusable hooks.
- Core APIs: `FusedStage`, `SizeAwareSampler`, `HostMemory`, `GPUBuffer`,
  `torch.compile`, and profiler hooks.
- Measurement: separate data movement, compilation, warm execution,
  throughput, and memory. Preserve stable graph identity through refill and
  collection.
- Continuity: reuse the data-loading path from Part 02 and the hook and
  `BaseDynamics` patterns from Parts 04 and 05.

## 2026-08-13 — approved implementation brief

Status: tests first

The user supplied the detailed lesson contract and requested implementation, so
that contract is the approved design. Work is restricted to
`notebooks/06-gpu-pipelines-profiling/` and this worklog.

### Learner outcome and prior knowledge

- Reuse `AtomicData`, `Batch`, model configuration, hooks, and `BaseDynamics`
  from Core and Parts 01–05.
- Keep one active `Batch` on one execution device, identify which object owns
  each transfer and stage transition, and distinguish `FusedStage` from a
  distributed stage pipeline and `DomainParallel`.
- Measure first-call setup, warm execution, complete `run(...)` time, and CUDA
  memory as separate quantities. Treat profiler traces as diagnostic evidence,
  not benchmark samples.

### Cell and visual sequence

1. Short orientation, shared Part 06 course map, and a three-route ownership
   diagram for fused, distributed-stage, and domain-parallel execution.
2. Deterministic synthetic `AtomicData` graphs built on the host, one explicit
   `Batch.to(device)` transfer, a fixed-shape tensor function, eager/compiled
   parity, and synchronized first-call versus warm timing.
3. A timing distribution plot and CUDA-memory panel. A CPU-only run labels
   CUDA memory as unavailable instead of plotting zero or cached GPU values.
4. A two-stage synthetic inflight `FusedStage` using `SizeAwareSampler`,
   `HostMemory`, stable `system_id`, and `StageTimingHook`.
5. A pinned NCI Atlas molecular dataset and verified AIMNet2 checkpoint in a
   bounded FIRE2-to-NVE inflight workflow. The calculation checks routing,
   shapes, units, identity, and device ownership; it makes no convergence,
   equilibration, accuracy, or throughput claim.
6. A separate `TorchProfilerHook` run that writes a short trace to a temporary
   per-rank directory. Timing and profiling runs remain separate.
7. Public `DistributedPipeline` and `BufferConfig` construction without a
   notebook-launched distributed job, followed by a clear handoff to Part 08.
8. An advanced, pin-specific `GPUBuffer` mixed-dtype probe. The lesson never
   presents `Batch.put` or `Batch.defrag` as a safe general route.
9. A bounded measurement exercise and recap.

### Visible public APIs

`AtomicData.from_atoms`, `Batch.from_data_list`, `Batch.to`,
`Batch.get_data`, `FusedStage`, `SizeAwareSampler`, `HostMemory`, `GPUBuffer`,
`FIRE2`, `NVE`, `StageTimingHook`, `TorchProfilerHook`,
`DistributedPipeline`, `BufferConfig`, model `make_neighbor_hooks`,
`torch.compile`, CUDA events/synchronization through a tested timing helper,
and public CUDA memory counters.

### Helper boundaries

- Deterministic synthetic data and checked NCI Atlas/checkpoint loading.
- Synchronization, repeated timing, runtime identity, memory snapshots,
  result shaping, trace-file listing, and plots.
- One small monitoring hook that records public `DynamicsContext` fields.
- The mixed-dtype buffer probe, because the unsafe low-level sequence should
  remain out of the main learner path.
- Public construction, configuration, execution, and result inspection stay
  visible in the notebook.

### Inputs, scope, and expected runtime

- Synthetic graphs use fixed seeds and have no scientific interpretation.
- Molecular inputs are distinct neutral H/C/N/O fragments from the pinned
  NCI Atlas CC BY 4.0 subset. AIMNet2 uses the pinned MIT checkpoint
  `aimnet2-wb97m-d3_0`, float32, energy in eV, forces in eV/Å, and its external
  matrix neighbor requirement.
- The local authoring environment verifies Toolkit commit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, Toolkit-Ops commit
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`, and PyTorch
  `2.12.0+cu130`.
- CUDA is unavailable in the current sandbox. All GPU work will use
  `/tmp/alchemi-v3-notebook.lock`. Fresh execution must therefore exercise the
  CPU fallback and label GPU measurements unavailable. No GPU numbers will be
  invented or copied from another machine.
- Target fresh-kernel runtime: under five minutes on the available CPU; the
  CUDA path uses the same cells when hardware is present.

### Bounded exercise and checks

- Exercise: change warm-up/repeat counts, rerun the synthetic measurement, and
  verify that compile setup remains outside the reported warm distribution.
- Static checks cover schema, syntax, cell IDs, private/stale API language,
  public token order, links, source cleanliness, cell size, callouts, diagrams,
  plot alt text, table width, and output errors.
- Behavioral checks cover timing sample counts, parity, device/dtype/shape
  ownership, sampler limits, stable IDs through `HostMemory`, timing-hook
  summaries, plot labels, profiler output layout, and the mixed-dtype buffer
  limitation.
- Final review requires scoped tests, lint, two editorial passes, fresh-kernel
  execution under the lock, HTML rendering, and a cell-by-cell review index.

### Primary references checked

- Official inflight batching and multi-stage examples:
  https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/04_inflight_batching.html
  and
  https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/01_multistage_pipeline.html
- Official distributed pipeline, buffer lifecycle, and per-rank monitoring:
  https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/distributed_pipeline.html,
  https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/buffers_and_data_flow.html,
  and
  https://nvidia.github.io/nvalchemi-toolkit/examples/distributed/02_distributed_monitoring.html
- Official profiling APIs:
  https://nvidia.github.io/nvalchemi-toolkit/modules/generated/nvalchemi.hooks.StageTimingHook.html
  and
  https://nvidia.github.io/nvalchemi-toolkit/modules/generated/nvalchemi.hooks.TorchProfilerHook.html
- PyTorch compiler and profiler tutorials:
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  and
  https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Pinned Toolkit source and tests at commit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, especially
  `nvalchemi/dynamics/base.py`, `sampler.py`, `sinks.py`,
  `nvalchemi/data/batch.py`, `nvalchemi/hooks/stage_timing.py`, and
  `nvalchemi/hooks/physicsnemo_profiling.py`.

## 2026-08-13 — implementation and review

Status: technically validated CPU-fallback deep dive; target-GPU measurements
and external human cell approval remain pending.

### Revision pass 1 — technical and scientific

- Verified every taught Toolkit/Toolkit-Ops route against the pinned public
  API, generated docs, examples, and source. Corrected model configuration
  access, fixed-duration distributed-stage construction, per-rank launch
  guidance, hook timing scope, and memory-scope wording.
- Kept warm-up, synchronized wall time, `StageTimingHook`, and profiler traces
  as separate evidence. The molecular run remains a bounded routing and
  identity check, not a benchmark, convergence result, or accuracy claim.
- Kept `Batch.put` and `Batch.defrag` out of learner code. The advanced
  `GPUBuffer` probe now uses a nonzero integer `source_index`, preventing a
  skipped zero-filled field from producing a false pass.

### Revision pass 2 — editorial and rendered

- Reviewed all 101 cells for pacing, ownership language, claim boundaries,
  small-cell progression, public API visibility, links, and direct prose.
- Corrected the Part 07 title, added rank-local CUDA binding to the distributed
  launch sketch, and retained explicit links back to Core/Part 05 and forward
  to distributed/domain-decomposition material.
- Rendered both plots as bounded embedded HTML so informative alt text survives
  the default `nbconvert` lab export. The timing/memory and pipeline-diagnostic
  images were visually checked for labels, units, clipping, and readability.

### Defect-review remediation

- Reproduced the pinned `FusedStage` fixed-step counter behavior: adjacent
  `n_steps=1` stages can migrate status 0 → 1 → 2 after only the first masked
  update. The downstream synthetic and NVE stages now use the pin-specific
  counter budget of 2, and both behavior tests and fresh-execution assertions
  require an observed status-1 update before collection.
- Resolved CUDA to a concrete indexed device, seeded synthetic model
  initialization, and added CPU model identity for fallback timing evidence.
- Renamed complete pipeline timings as instrumented because the timer and
  occupancy observer hooks run inside the measured call. Profiler stack capture
  is explicitly disabled and no longer claimed.
- Replaced blanket native-stderr suppression with a filter that removes only
  the known CPU-only Warp driver-lookup noise and re-emits every unexpected
  diagnostic. Import-time diagnostics are left visible.
- Marked the distributed object construction-only and added a launch-script hard
  stop: the preallocated send/receive route reaches `Batch.put`, so normal
  mixed-dtype `AtomicData` execution is blocked at this pin. The guarded launch
  order now binds the rank-local device, initializes NCCL, then reads the global
  rank.
- Added a locked fresh-kernel CPU-fallback execution and lab-HTML render
  regression so execution errors and missing plot alt text fail the test suite.

### Final validation evidence

- `pytest`: 43 passed, 1 skipped. The skipped test is the CUDA-only
  mixed-dtype `GPUBuffer` execution probe; no CUDA device is available here.
- `ruff`: all Part 06 notebook, helper, and test checks passed.
- Notebook design checker: 0 errors.
- Pinned runtime checker: Python 3.12.13, Toolkit
  `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`, Toolkit-Ops
  `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Fresh-kernel in-place execution under `/tmp/alchemi-v3-notebook.lock`
  succeeded; the automated execution/render regression also passed.
- Default HTML export: succeeded without missing-alt or execution-error
  warnings. Three rendered images have informative alt text.
- Expected non-blocking test warnings remain in pinned dependencies: Torch JIT
  deprecations and one NVML-unavailable warning.

### Cell-review index (0-based)

- 0: shared banner.
- 1: outcome, prerequisites, claim boundary, and folded product recap.
- 2–6: setup, pinned runtime identity, and unavailable-GPU policy.
- 7–10: course map, three execution routes, and transfer ownership callout.
- 11–31: synthetic host/device transfer, fixed-shape compile path, synchronized
  timing, allocator scope, and first plot.
- 32–48: synthetic inflight admission/routing/collection, pin-specific
  downstream counter compensation, instrumented run timing, occupancy, stage
  timing, and interpretation.
- 49–75: checked molecular inputs, AIMNet2 identity, equivalent warm-up,
  verified FIRE2-to-NVE routing, instrumented timing, memory evidence, result
  checks, and limitations.
- 76–83: separate profiler run with stack capture disabled, rank-scoped
  artifacts, and trace interpretation.
- 84–91: public distributed-stage construction, fixed-duration routing,
  construction-only mixed-dtype hard stop, distributed initialization order,
  rank-local device binding, monitoring, and domain-decomposition boundary.
- 92–96: advanced mixed-dtype `GPUBuffer` and distributed-buffer limitation
  with a hardware-gated probe.
- 97–100: bounded timing exercise and recap.

Automated and author-agent review is complete for every range above. Per the
deep-dive contract, an external human still needs to approve or revise each
learner-visible cell. No GPU measurements were produced, cached, or inferred,
and no commit was created.
