# ALCHEMI Playbook v2 remaster plan

This is a historical working plan. It does not define the current tutorial
order, model selection, or package plan. It remains because it records earlier
design work that may still be useful.

## Current order decided after this plan

1. **Part 1:** the water-centered, seven-stage Toolkit foundations notebook;
2. **Part 2:** the original adsorption tutorial; and
3. **Part 3:** the OLED/melting tutorial.

Some directory names retain earlier numbers. `README.md` and the learner
tutorial titles define the permanent order. The focused NCI Atlas composition
lesson below is now integrated into Stage 3 of Part 1; the broader DESS work
remains historical research.

Current guidance lives in `README.md`,
`part-1-scalable-atomistic-workflows/README.md`,
`ALCHEMI_TUTORIAL_PRINCIPLES.md`, and `TOOLKIT_API_CURRICULUM.md`.

## Historical 2026-07-10 NCI placement proposal

At that point in development, the proposal preserved the existing tutorials
and placed a new molecular notebook as Part 3. This numbering is obsolete:

1. **Then Part 1 — existing MACE adsorption tutorial.** A lean rebuild remained
   future work during the prototype.
2. **Then Part 2 — existing Orb melting tutorial.** Its live-compute replacement
   and data-license review remained future work.
3. **Then Part 3 — AIMNet2 molecular composition and GPU batching.** Teach CPU/GPU
   execution, homogeneous and heterogeneous batching, checkpoint-matched
   D3(BJ), predicted charges, and finite nonperiodic Coulomb.
4. **Later — periodic long-range physics.** Teach Ewald/PME only with a
   charge-aware model and a genuinely periodic system.

This placement was provisional and is now superseded. Earlier proposals below
are also retained as design history, not the current repository layout.

The prototype's primary accuracy comparison was not each partial model versus
S66 CCSD(T)/CBS. The selected checkpoint and reference family were
`aimnet2-wb97m-d3` and NCI Atlas:

```text
AIMNet core + full Coulomb       vs NCI ωB97M-D3(BJ) with matched D3 removed
AIMNet core + Coulomb + D3       vs full NCI ωB97M-D3(BJ)
complete model                   vs NCI CCSD(T)/CBS interaction curves
```

Use identical dimer and frozen-monomer geometries and the same two-body D3(BJ)
parameters on both sides. The NCI DFT basis is diffuse-augmented relative to
the training target, so this is near-matched rather than identical; NCI
CCSD(T)/CBS is the independent completed-model comparison. See
[`REFERENCE_DATA_PLAN.md`](REFERENCE_DATA_PLAN.md) for the measured selection
and [`TUTORIAL_DESIGN_PRINCIPLES.md`](TUTORIAL_DESIGN_PRINCIPLES.md) for the
tutorial patterns. B97-3c, DESS66, and older part-number assignments below are
retained as design history.

The prototype was designed to display all four component combinations: residual,
residual + Coulomb, residual + D3, and the complete model. The incomplete
combinations are controlled ablations that expose the roles of explicit
electrostatics and dispersion. They are not standalone DFT approximations, and
only residual + Coulomb and the complete model receive matched DFT accuracy
comparisons.

## Historical prototype implementation

The research prototype was
[`research-toolkit-foundations/alchemi-toolkit-foundations.ipynb`](research-toolkit-foundations/alchemi-toolkit-foundations.ipynb).
Its AIMNet2/D3/Coulomb and Cu(111)/CO paths had both passed runtime smoke
tests. At the time, the molecular NCI path was kept separate from the adsorption
and melting tutorials. The complete nonvisual notebook path and the replacement
Warp Tape graph passed separate local runtime smokes; the exact rebuilt workshop
image was not tested. The directory name does not define a current part number.

## Archived earlier decision summary

The sections below preserve the broader remaster analysis that preceded the
working Part 3 placement. They inform future Parts 1/2 redesigns but do not
define the current numbering.

The remaster should teach ALCHEMI as a composable simulation ecosystem rather
than use it as an implementation detail inside two long scientific stories.

- Keep Part 1, but reduce it to one visible, native Toolkit workflow:
  structures → `AtomicData` → `Batch` → model → hooks → FIRE2 → ranked results.
- Replace Part 2. Its headline 300 ps melting calculation is always replayed
  from cache, so the live path does not produce the result learners interpret.
- Make the new Part 2 a bounded live Toolkit-foundations tutorial using
  heterogeneous batches, AIMNet2/D3/electrostatic composition, explicit
  component/reference checks, and a native batched FIRE2 screen.
- Prototype a combustion elementary-step profile as an optional capstone only.
  Do not call a picosecond atomistic calculation a flame, ignition simulation,
  combustion rate, or mechanism.
- Resolve package pins and third-party data licensing before notebook edits.
  The then-current Part 2 contained CSD-derived files that should not be externally
  redistributed without permission.

## What the tutorials looked like when this plan was written

### Part 1

Measured at the time:

- 72 cells: 36 markdown and 36 code.
- Approximately 5,250 markdown words.
- 16.5 MB notebook and approximately 369 MB of tutorial content.
- A 216-start full screen: 9 surfaces × 4 adsorbates × 3 sites × 2
  orientations.

The clearest Toolkit lesson is cells 12–15: ASE H2O → `AtomicData` → `Batch`
→ `MACEWrapper` → FIRE2. It is followed by three different batching or
performance demonstrations. The actual adsorption calculation then switches
to local abstractions named `ToolkitRelaxationConfig`,
`get_toolkit_relaxation_engine`, and `.async_relax()`. Those names look like
official Toolkit APIs but are project helpers; the native Toolkit execution is
hidden inside `helpers/relaxation_backends.py`.

The scientific safeguards are valuable and should remain: matched references,
multiple starting structures, frozen-layer conventions, geometry audits,
convergence checks, desorption filtering, and inspectable trajectory files.
The breadth of the surface panel, repeated timing studies, and deep OC20Dense
render/trajectory review do not all belong in the core learner path.

### Part 2

Measured at the time:

- 85 cells: 47 markdown and 38 code.
- Approximately 7,980 markdown words.
- 3.8 MB notebook and approximately 73 MB of shipped cached results.
- A custom Orb adapter of more than 500 lines plus private Toolkit access and a
  neighbor-list compatibility monkey-patch.
- A 7,200-atom coexistence system, 50 ps warmup, and four 300 ps production
  trajectories—approximately 2.5 million integration steps at 0.5 fs.

`RESULT_SOURCE="compute"` runs only preparatory FIRE2 and short 1 ps NVT
stages. The decisive 300 ps NPT trajectories used for the melting-point verdict
are always loaded from cache. This is useful as a scientific case study but it
does not satisfy the v2 requirement that the headline result be computed live.

## Teaching thesis

The two parts should answer two different questions.

1. Part 1 — **How do I turn a scientific search into a batched GPU workflow?**
2. Part 2 — **How do I compose models, physics, dynamics, hooks, and data flow
   into a simulation I can inspect and extend?**

Every core calculation should expose the same recognizable pipeline:

```text
ASE/pymatgen structure
  → AtomicData
  → Batch
  → model or composed model
  → Toolkit-Ops-backed neighbors/interactions
  → optimizer or integrator
  → hooks, logging, and snapshots
  → live scientific observable and inspectable result
```

## Toolkit capability and API curriculum

The standalone, symbol-level curriculum is maintained in
[`TOOLKIT_API_CURRICULUM.md`](TOOLKIT_API_CURRICULUM.md). This section records
the shorter design summary.

ALCHEMI has three distinct layers. The notebooks should name them correctly.

- **Toolkit Core (`nvalchemi`)** is the workflow layer: data, model adapters,
  dynamics, hooks, model composition, sinks, and pipelines.
- **Toolkit-Ops (`nvalchemiops`)** is the accelerated primitive layer:
  neighbor lists, DFT-D3, electrostatics, segmented operations, and low-level
  dynamics kernels for Torch and JAX.
- **The model ecosystem** supplies MACE, AIMNet2, Orb, UMA, checkpoints, and
  domain-specific data. Those models are not Toolkit itself.

### Must show directly in learner code

1. **Batch-first data**
   - `AtomicData.from_atoms`
   - `Batch.from_data_list`
   - `batch_idx`, `batch_ptr`, `num_graphs`, `get_data`
   - Device, dtype, cell, PBC, charge/spin, and node/system-level fields

2. **The model adapter interface**
   - A built-in `MACEWrapper` or `AIMNet2Wrapper`
   - `ModelConfig`: supported outputs, required inputs, neighbor format
   - `model_config.active_outputs`
   - Checkpoint identity and applicability domain

3. **A batched live scientific calculation**
   - One model call across independent graphs
   - `FIRE2` and `ConvergenceHook.from_fmax`
   - A batch dimension that answers the scientific question—not a detached
     performance-only benchmark

4. **Hooks as the extension mechanism**
   - Neighbor-list hook
   - `FreezeAtomsHook` or `WrapPeriodicHook`
   - `NaNDetectorHook` and a force/energy safety check
   - `LoggingHook`, target `StageTimingHook`, and `SnapshotHook`
   - `ProfilerHook` only when describing the existing pin; it is removed on
     the proposed target
   - One small, readable domain hook or biased-potential example

5. **Model composition**
   - MLIP + `DFTD3ModelWrapper`, or
   - AIMNet2 charges wired into `EwaldModelWrapper`/PME with
     `PipelineModelWrapper`, `PipelineGroup`, and `PipelineStep`

6. **Multi-stage dynamics**
   - Relaxation followed by `NVTLangevin`
   - A fused stage using `+` where the selected pin supports it cleanly
   - Per-graph conditions such as different temperatures

7. **Saved results and replay**
   - `ZarrData` or the public writer/reader surface
   - CSV logs and short `.extxyz` trajectories
   - Reload through `Dataset`/`DataLoader`
   - Method, model, and source details next to every result

### Should show once or as a short extension

- `SizeAwareSampler` and the idea of inflight replacement.
- `GPUBuffer`/`HostMemory` and why device-resident flow matters.
- One direct high-level `nvalchemiops.torch.neighbors` call showing COO versus
  matrix formats and automatic dispatch.
- `FIREVariableCell` or `FIRE2VariableCell` for cell-aware work.
- Energy/force/stress output selection.
- A concise comparison of MACE, AIMNet2, Orb, and UMA by domain and runtime
  requirements.

### Appendix or preview only

- `torch.compile`, cuEquivariance tuning, and JAX `jit`.
- `DistributedPipeline` and the `|` operator; the workshop core must not require
  multiple GPUs.
- Raw Warp kernels and segmented reductions.
- Custom `BaseDynamics` implementations.
- Toolkit training/fine-tuning, distributed training, losses, EMA/checkpoint
  hooks, and the training CLI. These are important current-main capabilities,
  but they are not in the existing tutorial pin or stable public release.
- UMA. Current Toolkit requires a separate incompatible dependency environment,
  and access to the weights is restricted under the FAIR Chemistry License.

## Part 1: lean batched adsorption search

Target: 35–45 core cells, 45–60 minutes, one 24-start live screen. Preserve the
216-start panel as an optional scale-up exercise.

### Proposed flow

1. **Three-cell orientation**
   - One ecosystem diagram: Toolkit, Toolkit-Ops, models, ASE/pymatgen, and
     outputs.
   - One concrete learner goal and expected output.
   - One bounded run configuration, defaulting to live compute.

2. **Native API vertical slice**
   - Keep the current `AtomicData`/`Batch`/MACE/FIRE2 sequence.
   - Prefer a small heterogeneous adsorption batch that feeds the later screen;
     otherwise retain the four-water hello world and stop after one short run.

3. **One performance lesson**
   - Keep a small adsorption-specific batch-size sweep tied to the real
     workload.
   - Move the H2O saturation sweep, CPU/GPU crossover, and three-model matrix to
     an optional performance appendix.

4. **Compact model sanity check**
   - Keep two or three representative OC20Dense comparisons.
   - Show a concise ranking/geometry result, not a 92-video/image review.
   - State explicitly that this is a bounded sanity check, not validation over
     all tutorial surfaces or adsorption chemistry.

5. **Small live search**
   - Two surfaces × two adsorbates × six starts = 24 relaxations.
   - Show one slab builder and one site/orientation generator explicitly.
   - Express the remaining systems as compact data specifications.

6. **Run the real workflow with native APIs**
   - Helpers may build structures, format tables, and render plots.
   - Learners should see `AtomicData` → `Batch` → model/pipeline → FIRE2 → hooks
     → `get_data()` for the actual screen.
   - Rename or remove helper aliases that look like official Toolkit classes.

7. **Rank and audit**
   - Compute matched adsorption energies.
   - Filter failed, unconverged, or desorbed structures.
   - Save initial/final `.extxyz`, the trajectory, a result table, and a
     manifest that maps every row to its saved files.

### Keep these scientific guardrails

- Matched clean-slab and gas references.
- Explicit dispersion convention and model head/checkpoint.
- Frozen-layer convention.
- Multiple starting geometries and a pre-run collision audit.
- Convergence, final force, and desorption checks.
- “Screening result, not literature parity” language.
- Inspectable source structures and trajectories.

### Move out of the core

- Full 9 × 4 × 6 screen.
- Deep OC20Dense trajectory/render review.
- Repeated batching benchmarks.
- Wide surface-science literature tour.
- Hardware-specific optimization matrices.

The OC20Dense material then accounted for most of Part 1’s data footprint.
Keep only the small attributed subset needed by the core; publish any extended
validation pack as an optional, versioned asset with its own license and
source manifest.

## Part 2 recommendation: composable charge-aware water clusters

Working title: **Composable Molecular Simulation with ALCHEMI Toolkit: Charges,
Electrostatics, Dynamics, and GPU-Native Trajectories**

### Scientific question

How do a learned short-range molecular potential, long-range electrostatics,
and dispersion jointly relax distorted finite water-cluster hydrogen-bond
networks, and how do those clusters respond during a short finite-temperature
trajectory?

This deliberately makes finite-cluster claims. It is not a bulk-water model,
phase-equilibrium calculation, or melting-point prediction.

### Proposed live workflow

1. Generate 10–20 five-water clusters with deterministic perturbations using
   ASE; do not import a coordinate dataset.
2. Convert to `AtomicData`, add charge/cell/PBC fields, and assemble one
   heterogeneous `Batch`.
3. Load `AIMNet2Wrapper`; inspect `ModelConfig`, active outputs, predicted
   charges, and neighbor requirements.
4. Compose AIMNet2, Ewald, and DFT-D3 with the public pipeline API. Display the
   short-range, electrostatic, dispersion, and total-energy contributions.
5. Relax the full cluster batch with FIRE2 and visible convergence, neighbor,
   NaN, profiling, logging, and snapshot hooks.
6. Select one cluster, clone it across four temperatures, initialize velocities,
   and run a bounded 0.25–0.5 ps `NVTLangevin` stage live.
7. Write the trajectories and logs to Zarr/CSV, reload through the public data
   pipeline, and render hydrogen-bond counts, partial charges, energy
   components, temperature traces, and a short animation.
8. End with a five-minute ecosystem map: size-aware/inflight batching,
   distributed execution, JAX Ops, and the current-main training preview.

### Why this is the best core replacement

- The headline result is computed live.
- It adds model composition, predicted-property wiring, dynamics, hooks,
  profiling, and data persistence instead of repeating Part 1’s screening
  story.
- It uses documented public wrappers rather than a large custom Orb adapter.
- The input structures are generated, avoiding third-party coordinate
  redistribution.
- It has a small deterministic compute envelope and a meaningful reduced mode.
- NVIDIA already publishes an AIMNet2 + Ewald pipeline example, reducing API
  invention risk.

Official anchors:

- [Toolkit supported models](https://nvidia.github.io/nvalchemi-toolkit/models/index.html)
- [Toolkit examples](https://nvidia.github.io/nvalchemi-toolkit/examples/index.html)
- [AIMNet2 + Ewald example](https://nvidia.github.io/nvalchemi-toolkit/examples/advanced/08_aimnet2_ewald_pipeline.html)
- [AIMNet model-selection guide](https://isayevlab.github.io/aimnetcentral/models/guide/)

## Four live-compute candidates

The runtime and memory values below are planning estimates, not measurements.
Each candidate must be benchmarked from a cold and warm start on the slowest
supported workshop GPU.

### 1. Composable charge-aware water clusters — recommended core

- **System:** 10–20 water pentamers; four short-temperature replicas.
- **Model:** standard AIMNet2 + Ewald + matching DFT-D3(BJ).
- **APIs:** `AtomicData`, `Batch`, `AIMNet2Wrapper`, model pipeline/wiring,
  FIRE2, NVT, hooks, profiler, Zarr, reader/dataloader.
- **Estimated workload:** 150–300 relaxation atoms; about 60 MD atoms;
  30–120 s and approximately 2–4 GB after model load.
- **Output:** component energies, charge-colored structures, hydrogen-bond
  network changes, temperature traces, and live trajectory replay.
- **Scientific scope:** finite clusters only; verify large-box Ewald
  convergence and do not infer bulk-liquid behavior.
- **Risk:** low–medium science risk, high live reliability, very high Toolkit
  breadth.

### 2. OH + CH4 hydrogen abstraction — optional combustion capstone

Scientific question: what is the constrained doublet potential-energy profile
for the elementary reaction OH + CH4 → H2O + CH3?

- **System:** 24–32 neutral, multiplicity-2 seven-atom restraint windows using
  ξ = r(C–H) − r(O–H).
- **Model:** AIMNet2-NSE, with the physically matching external corrections.
- **APIs:** `AIMNet2Wrapper`, charge/spin fields, `PipelineModelWrapper`, Ewald,
  DFT-D3, `BiasedPotentialHook`, FIRE2, convergence/safety/snapshot hooks.
- **Estimated workload:** 168–224 atoms, 150–300 FIRE2 steps; 20–90 s warm,
  up to roughly three minutes cold, and approximately 2–4 GB.
- **Output:** animated H transfer, component-energy profile, charge/spin-charge
  evolution, and ensemble spread.
- **Scientific scope:** this is a restrained PES profile, not an ignition
  simulation, rate constant, free-energy barrier, IRC, or certified transition
  state. Validate endpoint connectivity, box-size convergence, and the profile
  against an independent reference before inclusion.
- **Risk:** medium–high science risk and medium live reliability, but very high
  instructional value if it passes preflight.

References:

- [AIMNet2-NSE model card and MIT license](https://huggingface.co/isayevlab/aimnet2-nse)
- [AIMNet2-NSE paper](https://chemrxiv.org/engage/chemrxiv/article-details/688ae42f728bf9025e345e87)
- [Independent reaction-profile benchmark](https://pubs.rsc.org/en/content/articlelanding/2020/cp/d0cp02560g)

Do not reproduce figures from a source whose license forbids derivatives. Use
only our own computed plots and cite the numerical/reference source.

### 3. High-throughput conformer race

- **System:** 64–128 generated conformers across a few small/flexible organic
  molecules.
- **Model:** AIMNet2 general or a specifically pinned AIMNet2-2025 checkpoint.
- **APIs:** Zarr writer/reader, `Dataset`, `DataLoader`, `SizeAwareSampler`,
  `AtomicData`, `Batch`, model composition, FIRE2, logging/snapshots.
- **Estimated workload:** 1,200–2,500 atoms total, 100–300 optimization steps;
  about 1–5 minutes and 3–8 GB depending on the atom budget.
- **Output:** energy/torsion maps, convergence timeline, before/after gallery,
  and model-ensemble uncertainty for finalists.
- **Scientific scope:** electronic-energy rankings are not solution-phase
  populations or free energies; preserve stereochemistry and deduplicate with
  symmetry-aware RMSD.
- **Risk:** low science risk and high reliability, but it overlaps Part 1’s
  batched FIRE2 pattern.
- **Extra dependency:** RDKit, BSD-3-Clause. Generate structures from SMILES;
  do not import a coordinate dataset.

### 4. Parallel Lennard-Jones argon order-loss map — robust fallback

- **System:** four 108-atom FCC boxes at approximately 60, 85, 110, and 150 K.
- **Model:** Toolkit’s `LennardJonesModelWrapper`; no checkpoint.
- **APIs:** per-graph temperatures, NVT → NVE `FusedStage`, neighbor/wrapping
  and safety hooks, profiler, Zarr, RDF/MSD/Q6 analysis.
- **Estimated workload:** 432 atoms and 2,000–5,000 steps; approximately
  0.5–3 minutes and under 2 GB.
- **Output:** synchronized animations, RDF/Q6 curves, thermostat behavior, and
  NVE energy drift.
- **Scientific scope:** call the result short-time order loss, not a melting
  point. It depends on finite size, density, cutoff, thermostat, and duration.
- **Risk:** low science risk and very high live reliability, but it shows less
  of the external model ecosystem.

### Why full free-burning combustion is not the core

Combustion is radical/open-shell, multichannel, rare-event chemistry. A generic
MACE-MPA or Orb trajectory does not automatically validate products, kinetics,
or mechanisms. Published MACE hydrogen-combustion work used a 100 ps extreme-
condition simulation and described the result as qualitative; representative
gas-phase ignition delays can be many orders of magnitude longer than workshop
atomistic trajectories.

The elementary OH + CH4 profile is the scientifically honest short form: it
isolates one spin-defined combustion step and computes a bounded observable
live. A free-burning H2/O2 or CH4/O2 nanoreactor may be a research demo or
cached comparison, but not the v2 headline.

References:

- [MACE foundation-model applications, including 100 ps H2 combustion](https://doi.org/10.1063/5.0297006)
- [Cantera ignition-delay example](https://cantera.org/stable/examples/python/reactors/non_ideal_shock_tube.html)

## Proposed workshop compute plan

These are v2 design targets and must be replaced by measured numbers before
release.

- One x86_64 NVIDIA GPU; no multi-GPU requirement.
- Target 16 GB or more GPU memory; validate explicitly on the slowest supported
  event GPU and the local 19 GB RTX 4000 SFF Ada.
- Core live stage under two minutes warm and under five minutes cold.
- Total GPU work per part under eight minutes, excluding learner discussion and
  visualization.
- No result depends on `torch.compile`; compilation is an optional extension.
- Model checkpoints are prefetched for offline delivery, with exact revision,
  checksum, license, and cache location recorded.
- Deterministic structure generation and seeds.
- A visible reduced mode—not a hidden change in scientific protocol.
- Cache is a recovery path after a classroom failure, never the silent source
  of the headline result.
- Synchronize CUDA before/after timed regions and report cold load separately
  from warm compute.
- Save partial results on OOM/failure with an explicit error and run details.

## Version and compatibility checks

The tutorial pins at the time were:

- Toolkit core `01c99d5cde6f63d6f662b071a9f408d3bfc12b0a` (2026-06-15).
- Toolkit-Ops `2b7c3c3adfb1ca84b886eecbf14bc60ff6ba1dc2`
  (2026-06-10, reports 0.3.1).

As of 2026-07-09:

- Toolkit core `main` is `b770ee6963fd2f6137891e408c370012751918e2`,
  10 commits ahead.
- Toolkit-Ops `main` still points to the tutorial pin, while `0.4.0-rc` is
  `c6fbe652315e0cebd4f57a6a25f626258f0dbbfd`, 25 commits ahead.
- Toolkit core at the time explicitly targeted Toolkit-Ops `0.4.0-rc`.

Current core adds training/fine-tuning, an in-memory data pipe, UMA, richer
reporting, and model-pipeline neighbor adaptation. The existing pin already
contains the major MTK NPT/NPH and Nosé-Hoover corrections. A direct child of
the pin adds preservation of inactive cells during masked updates.

Recommended approach:

1. Prototype against current core `b770ee6` + Ops `c6fbe65` in a separate image
   layer because that is the pair current core declares.
2. Run the minimal smoke matrix below before touching notebooks.
3. Freeze exact passing commits; do not use floating branch names in the
   released image.
4. Keep training/fine-tuning and UMA out of the core learner path until their
   API/dependency/license story is release-ready.
5. Remove the old private imports and neighbor-list monkey-patches rather than
   carrying compatibility shims into v2.

Smoke matrix:

- MACE single point, batch, FIRE2, frozen atoms, and snapshot.
- AIMNet2 single point and predicted charges.
- AIMNet2 + DFT-D3 + Ewald pipeline, forward and gradients.
- FIRE2 → NVT fused stage with per-graph temperatures.
- Zarr write/read and `DataLoader` replay.
- High-level Torch Toolkit-Ops neighbor list in matrix and COO forms.
- Reduced Part 1 and proposed Part 2 on the 19 GB local GPU.
- Clean container rebuild with no host cache, then offline replay from the
  staged model cache.

## Licensing and redistribution review

The root Apache-2.0 license covers repository code, not every embedded dataset,
model, binary, image, or derived trajectory.

Confirmed:

- Toolkit and Toolkit-Ops: Apache-2.0.
- MACE code and MACE-MPA-0 weights: MIT.
- Orb code and model weights: Apache-2.0.
- AIMNet2 code and current referenced model cards: MIT.
- ASE: LGPL-2.1-or-later.
- RDKit, if used: BSD-3-Clause.
- OC20/OC20Dense data: CC BY 4.0; attribution is required.
- UMA code: MIT, but model-weight access is restricted by the FAIR Chemistry License and
  require a separate review. Do not bake them into the workshop image.

Release blockers:

1. Part 2 includes CIFs downloaded from the Cambridge Structural Database,
   including `data/naphthalene.cif` and `data/1428146.cif`. CCDC states that
   original CSD data generally cannot be shared externally. Remove/replace
   these files unless explicit permission covers this distribution.
2. Treat trajectories and caches derived from CSD structures as requiring the
   same review; do not assume they are automatically redistributable.
3. The OC20Dense subset records its source but has no explicit data-license
   notice in the repository. Add CC BY 4.0 attribution and source/checksum
   metadata.
4. Add `THIRD_PARTY_NOTICES.md` plus a machine-readable asset manifest with:
   source, author/owner, license, citation, version/revision, checksum,
   transformation, and redistribution status.
5. Prefer generated inputs or CC0 sources such as the Crystallography Open
   Database for any replacement structures.

Primary policy links:

- [CCDC redistribution guidance](https://support.ccdc.cam.ac.uk/support/solutions/articles/103000339607-can-i-redistribute-data-from-the-csd-)
- [OC20 terms: CC BY 4.0](https://opencatalystproject.org/open_catalyst_project_tos.pdf)
- [Crystallography Open Database](https://crystallography.net/cod/)
- [Toolkit license](https://github.com/NVIDIA/nvalchemi-toolkit/blob/main/LICENSE)
- [Toolkit-Ops license](https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/main/LICENSE)

## Implementation sequence

### Phase 0 — required checks

- Freeze and test the Toolkit/Toolkit-Ops pair.
- Complete the third-party asset/license inventory.
- Remove or replace non-redistributable CSD inputs and review derived caches.
- Define the actual slowest workshop GPU and time budget.

### Phase 1 — executable spikes

- Build a tiny public-API smoke script for each row of the smoke matrix.
- Measure cold load, warm compute, synchronized wall time, and peak memory.
- Spike the water-cluster core in a standalone script before making a notebook.
- Spike the OH + CH4 profile separately; discard it if any scientific or
  runtime check fails.

### Phase 2 — lean Part 1

- Preserve that notebook as an archive/reference.
- Build the 24-start native-API core path.
- Reduce the core validation assets and move the broad panel to an optional
  extension.
- Keep every scientific safeguard and inspectable saved result.

### Phase 3 — replacement Part 2

- Build the water-cluster vertical slice with generated inputs.
- Make `compute` the default.
- Add model composition, hooks, profiler, Zarr replay, and one direct Ops cell.
- Add a clearly labeled reduced mode with the same scientific protocol.

### Phase 4 — optional capstone

- Add the combustion elementary-step profile only if it passes endpoint,
  reference, box-size, ensemble-spread, offline, runtime, and memory checks.
- Label it as a constrained elementary-step profile everywhere.

### Phase 5 — release validation

- Run both notebooks top-to-bottom in a fresh container with no pre-existing
  output directory.
- Recompute headline results from source structures; do not validate by merely
  rereading shipped CSVs.
- Verify every summary row maps to initial/final `.extxyz`, trajectory, log, and
  source and method metadata.
- Test the recovery cache separately and label it visibly.
- Rebuild the image from exact pins and test offline model access.
- Update `README.md`, `RUNTIME_SNAPSHOT.md`, `CHANGELOG.md`, and third-party
  notices.
- Perform scientific, API, licensing, and visual review before release.

## Acceptance criteria

- Part 1 core has one batching lesson and one live 24-start application.
- Part 2’s headline observable is generated live by default.
- Learner-visible code uses documented public APIs; no private imports or
  monkey-patches are required.
- Each part visibly teaches at least four distinct Toolkit capabilities.
- Part 2 visibly teaches model composition, dynamics, hooks, and persistence.
- At least one Toolkit-Ops-backed operation is inspected directly.
- All timing is synchronized and split into cold versus warm measurements.
- Both parts pass the agreed time/memory envelope on the slowest target GPU.
- Every scientific claim has an explicit scope, model domain, and remaining
  unknowns.
- Every shipped model/data/asset has a redistribution decision and attribution.
- All live results produce inspectable structures, trajectories, logs, and a
  manifest.

## Repository reconnaissance

- History: 92 commits from 2026-03-09 through 2026-06-17; four local/remote
  branches in this single-branch clone.
- Contributors: 75 non-merge commits from the primary contributor and one each
  from two others. Two of three contributors were active in the last three
  months; knowledge is still concentrated.
- Monthly commits: 38 in March, 27 in April, 15 in May, and 12 in June. This
  looks like tapering after initial construction, not evidence by itself of a
  project-health problem.
- Historical hotspots and bug magnets overlap in `README.md` and the removed
  OER notebook. The recommendation was to start with the Part 1/Part 2 notebooks and their
  helper structure, not the historical OER surface.
- One revert-style fix appears in 92 commits; there is no broad pattern of
  emergency rollback commits.
- The clean v2 clone started at `bfb8ab94aa7b25ad05d270d275121baf3da8c693`
  on local branch `v2`.

## Immediate next decision

Approve or revise the proposed Part 2 core:

1. **Recommended:** charge-aware water clusters as the guaranteed live core,
   with OH + CH4 as a conditional capstone.
2. **More chemistry-forward:** make OH + CH4 the core only after the spike
   passes every scientific and runtime check; keep water clusters as fallback.
3. **Lowest risk:** use the LJ argon order-loss map as the core and keep all
   MLIP/reaction examples optional.

The recommended option best matches the feedback: it shifts time away from a
long science narrative and into the Toolkit/Toolkit-Ops composition, dynamics,
hook, profiling, and data-flow surfaces that the current playbook underexposes.
