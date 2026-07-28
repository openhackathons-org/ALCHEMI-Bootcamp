# Part 1: From one structure to scalable atomistic workflows

This tutorial starts with one ALCHEMI Toolkit model call and grows through
batched intermolecular interactions, a custom materials-model adapter, and a
predicted-charge IR workflow. The main subject is Toolkit data, batching,
model composition, custom adapters, dynamics, larger queues, spatial domain
decomposition, and saved results.

After the first Toolkit result, a compact primer compares PyTorch, JAX, and
Warp with the same Toolkit-Ops reduction. It makes clear that Toolkit Core
follows PyTorch here, JAX is a peer Toolkit-Ops interface, and Warp is the
lower-level kernel layer used by selected operations.

Open [alchemi-water-ir.ipynb](alchemi-water-ir.ipynb) in the playbook container
and run it from top to bottom.

## Seven-stage path

| Stage | What you will do |
|---|---|
| 1. One structure, one result | Convert an ASE water structure to Toolkit data, run AIMNet2, and inspect energy, forces, the formal total charge, predicted atomic charges, and the model configuration. |
| 2. The same physics in a batch | Reproduce the result with variable-size batches, compare CPU and GPU execution, inspect batch layouts, and use the Core neighbor path selected from the model configuration. |
| 3. Complete and check the molecular model | Evaluate 90 NCI Atlas graphs with four AIMNet2 checkpoint-base calls, four direct-Coulomb calls, and one shared D3 call; then compare the complete interaction curves with near-matched DFT-D3 and independent CCSD(T)/CBS references. |
| 4. Bring a materials model into Toolkit | Explain why the molecular checkpoint does not cover Cu, connect SevenNet through a custom energy/force adapter, inspect its tasks, and reuse the loaded checkpoint for a small `mpa`/`oc20` model sweep before continuing with the `mpa` surface workflow. |
| 5. Prepare the IR calculation | Check eager and compiled results on one fixed batch, relax the structures with Toolkit's `FIRE2` geometry optimizer, and calculate the harmonic reference. |
| 6. Run, save, and inspect the trajectory | Run exact NVT and NVE stages, record predicted-charge dipoles, validate the route, and compare the resulting qualitative IR spectrum with separate DFT and experimental references. |
| 7. Scale queues and single systems | Process a larger queue with inflight batching, load a checked periodic phenol/N-methylacetamide box, walk through `DomainParallel` on one GPU, and compare three fixed-structure energy/force passes for the same 51,200-atom input on 1, 2, and 4 H100s. `DistributedPipeline` remains a separate API preview with correctness and timing left `NOT REPORTED`. |

## Hardware and runtime

- Target hardware: one NVIDIA H100-class GPU for the complete live notebook.
- Checked H100 time: **12 min 51 s** for all code cells and **13 min 1 s** of
  notebook wall time. The scheduler elapsed time was **15 min 49 s**.
- Other CUDA GPUs run the same declared calculation and may take longer.
- The notebook does not silently shorten trajectories or add atom-count limits
  for weaker hardware.
- Learners do not need a multi-GPU or multi-node allocation. The live notebook
  uses one GPU. Checked multi-GPU `DomainParallel` results are loaded from a
  recorded result set. A development checkout without that result set displays
  `NOT REPORTED`; the release requires the checked bundle.
  `DistributedPipeline` remains a short API sketch and has no reported timing.
- The repository and image do not redistribute Toolkit's generated D3
  parameter cache. On first D3 use, Toolkit creates it from its official
  fixed-checksum source unless `ALCHEMI_D3_PARAM_FILE` points to an existing
  copy. The notebook verifies the generated file before accepting results.
- The repository and image do not redistribute AIMNet checkpoints. The image
  build verifies all five files and removes them in the same layer. A mounted
  cache or the first model call supplies the checked files at runtime.
- The SevenNet-Omni checkpoint is also fetched into the user's runtime cache,
  checked by file size and SHA-256, and not baked into the tutorial image.

The notebook labels short calculations as demonstrations when they are not
long enough to support a convergence or accuracy claim.

### Checked H100 pacing

Complete-notebook job `3317215` ran commit
`1eca73058c5bae4a164f3b07c3e12fa944030086` on one NVIDIA H100 PCIe with
Toolkit Core `331d6b2` and Toolkit-Ops `e8e7a74`. All 92 code cells completed;
none failed. The measured section times were:

| Current source section | Code time |
|---|---:|
| Setup and imports | 23.476 s |
| Stage 1 | 18.211 s |
| Stage 2 | 19.250 s |
| Stage 3, ten code cells | 22.064 s |
| Stage 4 | 17.649 s |
| Stage 5 | 72.690 s |
| Stage 6 | 567.180 s |
| Stage 7 | 30.238 s |
| **Total code time** | **770.760 s (12:51)** |

Notebook wall time was `781.144 s` (`13:01`), and scheduler elapsed time was
`15:49` with exit `0:0`. These values are useful for classroom pacing. They
are not an inference benchmark. The final notebook changes only the displayed
timing text; its 92 code cells are byte-for-byte the same as the notebook that
ran.

### Historical H100 timing from earlier Toolkit versions

The previous six-stage source completed on one H100 NVL with Toolkit Core
`80aab5c` and Toolkit-Ops `e8e7a74`. These measurements help with classroom
pacing, but they are not timings for the merged seven-stage notebook.

| Previous source section | Code time |
|---|---:|
| Setup and imports | 22.2 s |
| First model result | 18.0 s |
| Batching and performance | 10.9 s |
| Molecular composition and SevenNet adapter | 38.7 s |
| IR preparation | 80.2 s |
| Dynamics and analysis | 603.2 s |
| Inflight/distributed status and save | 22.5 s |
| **Total code time** | **795.7 s (13:15.7)** |

The scheduler elapsed time was `15:37`. The NCI Atlas stage was not present in
that historical run. Current job `3317215` above supersedes these values for
classroom pacing.

## Main Toolkit APIs

| Task | Public APIs shown |
|---|---|
| Atomistic data | `AtomicData.from_atoms`, `Batch.from_data_list`, `get_data`, `to_data_list`, `index_select` |
| Model configuration | `model_config`, `active_outputs`, `set_config`, `make_neighbor_hooks` |
| Built-in adapters | `AIMNet2Wrapper`, `DFTD3ModelWrapper` |
| Custom adapter | `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `NeighborListFormat` |
| Neighbor lists | Core `compute_neighbors`, `NeighborConfig`, and model-generated neighbor hooks |
| Model composition | `PipelineStep`, `PipelineGroup`, `PipelineModelWrapper` |
| Batched reductions | Toolkit-Ops `segmented_sum` |
| Framework layers | PyTorch and JAX Toolkit-Ops bindings, with a raw Warp array call |
| Relaxation and dynamics | `FIRE2`, `NVTLangevin`, `NVE`, `FusedStage`, `initialize_velocities` |
| Hooks | `Hook`, `ConvergenceHook`, `NaNDetectorHook`, `LoggingHook` |
| Inflight work | `InMemoryDataset`, `SizeAwareSampler`, `HostMemory` |
| Periodic electrostatics | `PMEModelWrapper`, `estimate_pme_parameters` |
| One large distributed system | `DistributedManager`, `initialize_mesh`, `DomainConfig`, `SpatialPartitioner`, `DomainParallel`, and its `partition`, `run`, and `gather` path |
| Distributed stages | `DistributedPipeline` and `BufferConfig` in a short API sketch, with no timing claim |
| Saved results | `ZarrData` plus ordinary CSV, NPZ, JSON, and `.extxyz` files |

The notebook keeps these calls visible. Code in [aux](aux/) handles structure
generation, file checks, plotting, signal processing, and notebook
presentation. Those helpers are not Toolkit APIs.

The complete API map is in
[TOOLKIT_API_CURRICULUM.md](../TOOLKIT_API_CURRICULUM.md).

## Scientific scope

### Intermolecular model composition

The checkpoint metadata declares `coulomb_mode="sr_embedded"`. Its output is
the checkpoint base

```text
E_base = E_NN - E_Coulomb^SR
```

The subtraction prevents double counting when the full Coulomb energy is
added. For the finite NCI systems, the complete model is

```text
E_complete^finite = E_base + E_Coulomb^full + E_D3
```

The full Coulomb term uses AIMNet2's predicted atomic charges. The partial
combinations are ablations: they show what changes when full Coulomb or D3 is
omitted. They are not separate quantum-chemistry methods and are not expected
to improve in a fixed order.

`AtomicData.charge` supplies one formal total charge for each whole system. It
is a model input, not a set of atom-by-atom formal charges. AIMNet2 predicts
geometry-dependent float32 atomic charges, and its internal charge correction
targets that input total with a float32 reduction. Re-summing the returned
charges in float64 can expose a small residual for a large system. The tutorial
records that residual instead of treating it as a physical net charge. The
predicted charges supply the Coulomb energy and the point-charge dipoles used
later in the tutorial. The NCI graphs keep their source formal charges,
including the charge −1 ammonia–benzoate complex and benzoate fragment. The
Stage 7 phenol/N-methylacetamide periodic box has formal charge zero.

All three interaction classes use the same frozen-monomer definition:

```text
E_int = E(AB) - E(A) - E(B)
```

The complete Toolkit result is compared with NCI Atlas absolute
ωB97M-D3(BJ)/def2-TZVPPD energies reduced with the same AB/A/B definition and
with independent CCSD(T)/CBS interaction energies. The DFT level is close to,
but not identical to, the checkpoint's ωB97M-D3/def2-TZVPP training level.
Partial curves are omission tests, not a quantum-mechanical energy
decomposition.

The NCI structures are finite and nonperiodic. Direct all-pairs Coulomb is
used for that setup. Ewald and PME are periodic methods and are not substituted
into this example.

### Periodic molecular-box scaling

Stage 7 reuses the phenol and N-methylacetamide molecules from the neutral NCI
dimer lesson in one larger periodic box. The live notebook loads a checked
3,200-atom base box and a static OVITO preview. Packmol placed equal numbers
of the two molecules independently when that bundle was prepared; it does not
run in the learner notebook or preserve bound dimers. This is a deliberate
boundary-condition change: finite all-pairs Coulomb is replaced by
`PMEModelWrapper`. The composed periodic model is

```text
E_composed^periodic = E_base + E_Coulomb^PME + E_D3
```

The checkpoint base still has short-range Coulomb subtracted, PME supplies the
full periodic Coulomb term from predicted atomic charges, and D3(BJ) remains
explicit. The calculation reuses NCI ensemble member 0,
`aimnet2-wb97m-d3_0`, and its checkpoint-declared D3(BJ) damping parameters.
The tutorial term is finite-cutoff and tapered over 12–15 Å; it is not
identical to the much longer, untapered reference D3 calculation.

The checked bundle supplies a non-overlapping, unequilibrated initial placement
at a declared construction density. The recorded 51,200-atom input is one
integer supercell of that base box, with its repeat factors saved. This
preserves composition and construction density without rerunning Packmol.
The notebook does not describe the box as a liquid or infer density, structure,
phase, or thermodynamic properties from it. AIMNet2 software support for
periodic cells is also not a validation of this large condensed-phase system;
the box is used to teach the scaling API.

The AIMNet2-to-PME group uses autograd so the forces include the response of the
predicted charges to geometry. D3 remains a separate direct-force contribution.
The one-GPU check also uses Toolkit `segmented_sum` to report the predicted
charge on every molecule in the checked base box. Only the total box charge is
constrained, so the molecular sums are saved and summarized as a model
diagnostic, not treated as validated intermolecular charge transfer.

The fixed 3,200-atom PME-versus-Ewald check requires the same predicted charge
array in both solvers and applies the declared
`|Σq − Qtarget| ≤ 1e-4 e` residual limit. The 51,200-atom one-GPU run saves
the float32 charge dtype, requested total, observed total, residual, and charge
magnitudes as diagnostics. It does not reuse that small-box absolute limit.
PME consumes the recorded predicted charges without a hidden charge adjustment.

The notebook estimates PME parameters and walks through the public one-GPU
`DomainParallel` call. One GPU has one domain, so this checks the API and finite
outputs without spatial decomposition or a speed claim. The checked offline
run keeps one 51,200-atom structure fixed on 1, 2, and 4 H100s. Each GPU count
partitions the structure once, performs one untimed initialization and warm-up,
then performs three measured energy/force evaluations before gathering the
atom-level output once. `BaseDynamics` supplies the evaluator interface;
its base update methods do not integrate atomic motion. Periodic wrapping may
change the stored image of an atom without changing the physical structure.

The offline run also checks fixed-charge PME against Ewald and compares
distributed forces and energies before showing the three raw pass times and
their median. Each multi-GPU count means the same number of nodes, ranks, and
H100s. Toolkit 0.2 repeats the PME mesh and FFT workspace on every GPU, so
domain decomposition does not divide all memory. The input box has formal
charge zero. Toolkit 0.2 does not expose the intermediate multi-rank predicted
charges, so the recorded multi-GPU checks compare the supported energy and
force outputs. These comparisons do not independently verify the global charge
residual of the distributed prediction. The short times apply only to this
input, model, software, and hardware; they are not a trajectory or a general
scaling claim. The first measured one-GPU pass contains remaining first-use
work even after the warm-up, so the notebook shows every raw pass and treats
the timing as instructional.

Exact cutoffs, halo depth, tolerances, energy-reduction handling, and launch
commands are in the
[Compute Lab runbook](COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set).

### Custom materials-model adapter

The molecular checkpoint is not used for a copper surface. The tutorial makes
that domain change explicit, then connects a materials model to Toolkit through
a custom adapter.

The example evaluates a clean Cu(111) slab, four molecule-on-surface starting
structures, and the corresponding isolated molecules in two batches. It shows
energies, compact force summaries, graph mapping, and adapter-versus-native
model agreement. Every atom-wise force is saved for later inspection.

These are fixed starting geometries. They are not equilibrium adsorption
energies, a site search, or a DFT accuracy benchmark. The section teaches model
adaptation and output mapping.

### Infrared comparisons

The tutorial keeps three kinds of result separate:

- a finite-temperature predicted-charge MD spectrum;
- a 0 K double-harmonic B97-3c reference; and
- selected observed gas-phase band positions.

Harmonic model and B97-3c frequencies use the same finite-difference and mode
analysis. Their intensity models differ. The MD spectrum has its own finite
trajectory and temperature interpretation. The notebook does not put these
three results on an invented common intensity or accuracy scale.

The full B97-3c reference method and raw outputs are documented in
[reference/README.md](reference/README.md).

## Batching and performance

The notebook first verifies that individual and batched calculations preserve
the same per-system answers. It then measures:

- first-call and warm CPU/GPU time;
- response time and throughput across batch sizes;
- structures per second and atoms per second;
- one heterogeneous batch versus homogeneous buckets;
- the location of the CPU/GPU crossover on the current machine.

The same structures, model, precision, and requested outputs are used in each
comparison. A CPU win or a bucketed-layout loss is kept and explained rather
than treated as a failed lesson.

## Inflight and distributed work

Inflight batching keeps a bounded active set on one GPU while completed systems
leave and new systems enter. Stable system IDs connect the changing active
batch to the original input queue and final results.

`DomainParallel` solves a different problem: one periodic system is divided
into spatial regions. The same high-level API can execute on one GPU, but that
one-GPU walkthrough has one domain and does not partition atoms or measure
multi-GPU scaling. The saved run uses the same 51,200-atom input on 1, 2, and
4 H100s. It partitions once, performs one untimed warm-up, measures three
energy/force passes, and gathers once. The gathered coordinates must remain
PBC-equivalent to the input, with a maximum minimum-image displacement no
larger than `1e-4 Å`.

Before the times are interpreted, every multi-GPU force component must agree
with the one-GPU result. Model tensors, coordinates, and forces remain
`torch.float32`. The one-GPU path also returns a `torch.float32` total energy.
The pinned multi-GPU path returns `torch.float64` after Toolkit's distributed
energy reduction. Each GPU layout is represented by the median of its three
measured energies. The 2- and 4-GPU energy ranges must each remain within
`1e-4 eV/atom`, and the 4-GPU median must agree with the 2-GPU median within
`1e-4 eV/atom`. The raw one-GPU-to-multi-GPU offset and one-GPU pass range are
diagnostic because Toolkit 0.2 reduces the one- and multi-GPU paths
differently. The result shows all three pass times and their median for every
GPU count. The 4-GPU row uses four nodes, each with one worker process and one
H100.

The checked `DomainParallel` release bundle belongs in
`data/domain_decomposition/recorded/`. During development, an absent bundle is
shown as `NOT REPORTED`; a present but invalid bundle stops the run. The
release requires all one-, two-, and four-GPU rows and their checks. It does
not require or deliberately trigger an out-of-memory failure.

`DistributedPipeline` places workflow stages on different ranks, where one
rank is one worker process. It does not split one model call across GPUs. The
notebook shows its public construction but no patched, estimated, or
placeholder timing. Retained development code may later write
`data/compute_lab_pipeline_campaign/`, but the release notebook does not read
that directory. Correctness and timing remain `NOT REPORTED`.

## User review required

Open [alchemi-water-ir.ipynb](alchemi-water-ir.ipynb) in the target JupyterLab
theme and check the following before release:

- Scroll through all seven stages. The progress cards, headings, callouts, and
  emphasis should use one consistent style without horizontal overflow.
- Confirm that helper-heavy inputs stay hidden while the public Toolkit calls,
  model configuration, numerical results, and conclusions remain visible.
- Rotate the first molecular structure and the water-cluster batch in their X3D
  views. Atom colors, bonds, periodic boundaries, and labels should be legible.
- Inspect the four adsorption structures together. Each molecule should be
  visibly separated from the Cu slab, without clipped controls or overlapping
  panels.
- Inspect the static OVITO preview of the checked base box. It should show one
  periodic box containing separate phenol and N-methylacetamide molecules,
  with no live packing or OVITO work in the notebook.
- Check the NCI interaction curves, harmonic comparison, IR spectrum, and
  DomainParallel plots. Legends, units, reference lines, and captions should be
  readable and should match the displayed result tables.
- Inspect every energy, force, charge, timing, and comparison table in the
  reviewed output from H100 job `3317215`. Rows and units should remain
  readable without clipped labels or hidden structures.
- Before the checked multi-GPU result directory is installed, Stage 7 should
  say `NOT REPORTED`. After installation, it should show all one-, two-, and
  four-GPU rows, all three measured passes, and every output-agreement check.

## Files written by a complete run

Interactive runs write to `outputs/run-interactive` by default. Set
`ALCHEMI_RUN_ID` before starting the kernel to use a separate
`outputs/run-<id>` directory; scheduled runs use the Slurm job ID. The saved
files include:

- energy, force, charge, batching, and performance tables;
- NCI Atlas ensemble curves, completed-model curves, and reference errors;
- structures before and after relaxation;
- the complete production trajectory;
- harmonic displacement arrays and mode tables;
- IR spectra and diagnostic tables;
- figures with descriptive alternative text;
- a JSON summary containing settings, software and hardware details, checks,
  file sizes, and SHA-256 values.

During the completed notebook run, the analysis section reloads the saved
trajectory and verifies its checksum before post-processing. This checks the
saved-data route without repeating the dynamics calculation; the cell is not a
standalone fresh-kernel entry point.

## Runtime files and licenses

Model checkpoints, D3 reference data, scientific datasets, and observed band
positions have terms separate from this repository's Apache 2.0 license. The
D3 cache is generated in the user's runtime and is not baked into the image.
Do not commit or redistribute downloaded files without review.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the current software,
checkpoint, data, and redistribution notes.

## Rebuild and check the notebook

The notebook is generated from the maintained Python source:

Run these commands from the repository root:

```bash
python scripts/rebuild_part1_ir_notebook.py
python -m pytest -q part-1-scalable-atomistic-workflows/tests
```

The second command runs the full development-checkout suite. The clean learner
image keeps and runs four CPU-capable build checks; target-GPU validation runs
separately on the complete source checkout.

Automated tests, a target-GPU run, and a human review of the rendered notebook
check different things. A release requires all three.

The exact two-revision H100 and multi-GPU release sequence is in
[COMPUTE_LAB_RUNBOOK.md](COMPUTE_LAB_RUNBOOK.md).

Tutorial-writing and visual rules are in
[ALCHEMI_TUTORIAL_PRINCIPLES.md](../ALCHEMI_TUTORIAL_PRINCIPLES.md).
