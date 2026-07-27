# Part 1 learner-readiness plan

This is the working plan for the new Part 1 notebook. It is written for the
authors, not for attendees.

Audience: researchers who know basic computational chemistry or atomistic ML,
but have not used NVIDIA ALCHEMI Toolkit.

Notebook:
`part-1-scalable-atomistic-workflows/alchemi-water-ir.ipynb`

Generator:
`scripts/rebuild_part1_ir_notebook.py`

## Teaching goal

Teach the reusable Toolkit workflow through one progression:

```text
one structure
  -> one Toolkit result
  -> one batch
  -> measured batching choices
  -> complete molecular model
  -> model adapter for a new domain
  -> relaxation and dynamics
  -> saved and checked observables
  -> inflight queues
  -> one periodic system across spatial domains
  -> distributed workflow stages
```

The scientific examples make each Toolkit decision meaningful. Toolkit API
exposure takes priority over a complete lesson in intermolecular interactions,
surface science, or spectroscopy.

## Point-by-point plan

| Learner need | Implementation | Ready check |
|---|---|---|
| Know what ALCHEMI owns | Start with a compact map of ASE, Toolkit Core, Toolkit-Ops, model packages, PyTorch, JAX, Warp, and tutorial `aux/` code. | A learner can identify which names are public Toolkit APIs before the first model call. |
| Get a useful result early | Convert one water monomer to `AtomicData`, create a one-graph `Batch`, call the supplied AIMNet2 checkpoint, and inspect the numerical energy, forces, and charges. Then move to a water dimer for the first interaction energy. | `hello-world` starts from one monomer and shows actual model outputs before the dimer and framework comparison. |
| Understand PyTorch, JAX, and Warp without a detour | Use one segmented sum with identical values through the PyTorch binding, JAX binding, and raw Warp operation. State that this is an API comparison, not a speed test. | One short explanation and two short code cells return the same `[3, 7]` result. |
| Understand precision choices | Show checkpoint parameter dtype, input dtype, output dtype, spacing between nearby float32 numbers, and the effect of requesting float64 inputs. | The cell explains that widening inputs does not recover precision absent from float32 weights. |
| Understand batch identity | Compare individual calls with one batch, then show `get_data`, `to_data_list`, and `index_select`. Explain `batch_idx`, `batch_ptr`, and per-atom versus per-system outputs. | The same graph energies are recovered in the same source order. |
| Know when GPU batching helps | Measure first-call and warm-call time, then plot CPU and GPU throughput over batch sizes. Compare mixed and size-bucketed batches on the same structures. | Results report response time, structures per second, atoms per second, and numerical agreement. |
| Complete a supplied molecular model correctly | Read checkpoint metadata, expose the checkpoint base (`E_NN - E_Coulomb^SR`), add predicted-charge Coulomb and D3, and compose the dependent and independent stages with `PipelineModelWrapper`. | No D3 or electrostatics term is double counted; the complete result matches the component sum and an independent force route. |
| See why incomplete models are useful | Plot residual, residual + D3, residual + Coulomb, and complete curves for three NCI Atlas interaction types. Treat partial curves as diagnostics, not production models. | The DFT comparison with D3 removed is labeled explicitly; complete-model comparisons use full DFT-D3 and CCSD(T)/CBS references. |
| Understand the NCI reference comparison | Use 30 AB/A/B groups, apply `E(AB) - E(A) - E(B)`, and evaluate 90 graphs in four AIMNet, four direct-Coulomb, and one shared D3 call. Define `R/R_e`, ensemble spread, and the purpose of the 0.5 kcal/mol check. | The component names match the plotting and analysis helpers; the check is described as a composition check, not a general accuracy claim. |
| Change models when the domain changes | State that the molecular AIMNet2 checkpoint cannot represent Cu. Connect SevenNet-Omni through a visible `BaseModelMixin` adapter while retaining Toolkit data, neighbors, composition, and result handling. Read the checkpoint's available tasks and reuse one two-graph batch for an `mpa`/`oc20` model sweep. | The notebook names each selected target, keeps one task per call, shows energy and force outputs, forbids cross-task energy comparison, and continues the reported surface workflow with `mpa` plus its separate D3 term. |
| Learn the custom-wrapper interface | Keep `ModelConfig`, `NeighborConfig`, `adapt_input`, one raw model call, `adapt_output`, and direct force declaration visible. Move repetitive graph conversion and validation to `part-1-scalable-atomistic-workflows/aux/models/`. | The adapter agrees with the official SevenNet calculator and the composed pipeline agrees with the explicit component sum. |
| Avoid overstating the adsorption example | Evaluate CO, CO2, NH3, and CH3OH on fixed Cu(111) starting structures. Return all energies and compact force summaries. | The notebook states that these are fixed-geometry, finite-coverage electronic energy differences, not relaxed adsorption energies or a ranking. |
| Build dynamics from public APIs | Return to the charge-predicting AIMNet2 water checkpoint, build the four-system isotope batch, compose the complete model, relax with `FIRE2`, initialize velocities, and connect NVT and NVE with `FusedStage`. | The public Toolkit construction and hook registration remain visible. |
| Understand hooks | Define a hook, NaN, and Inf in plain language. Show regular and fused hook registration, neighbor updates, convergence, counters, logging, progress, and one charge-based IR recorder. | The recorder reuses charges from the model call and does not run a second inference pass. |
| Compare IR results honestly | Save the trajectory immediately after dynamics and before post-run analysis, then check temperature, energy, and topology before comparing finite-temperature MD, B97-3c harmonic calculations, and selected experimental gas-phase positions in separate lanes. | Route or shape failures before persistence are hard failures. Later temperature or topology failures keep the saved trajectory and mark affected comparisons as `NOT REPORTED`; no shared intensity scale or combined IR error is reported. |
| Explain scaling beyond one batch | Demonstrate inflight replacement live with `InMemoryDataset`, `SizeAwareSampler`, `HostMemory`, stable `system_id`, and bounded active work. Load a checked 3,200-atom phenol/N-methylacetamide base box and static OVITO preview, replace finite Coulomb with PME, walk through the `DomainParallel` API on one GPU with no decomposition, and show the separate `DistributedPipeline` stage layout. | The learner can say which API fits many independent systems and which fits one oversized periodic system. Multi-GPU measurements remain `NOT REPORTED` until complete H100 result sets pass their checks. |
| Teach the intended domain API without exposing internals | Keep `DomainConfig`, `SpatialPartitioner`, `DomainParallel`, `partition`, `run`, and `gather` visible. Explain that `DistributedManager` creates the rank mesh, `grid_dims` controls spatial cells rather than the rank layout, each GPU owns one region plus a halo, the globally reduced energy stays on the local result, and `gather` reconstructs atom fields. | No private fields, hand-written collectives, ad-hoc multiprocessing, live giant-box packing, or live OVITO rendering appears in the learner path. The single-GPU walkthrough is labeled as one domain with no decomposition, and the Toolkit 0.2 example is limited explicitly to a neutral system. |
| Keep the large-box claim honest | Build and check the small base box once offline, then create larger inputs as recorded integer supercell repeats so composition and density stay fixed without rerunning Packmol. Retain every one-H100 size attempt and the first natural OOM; retry that exact failed input on 2/4 GPUs; measure speed on a separate input that already fits one H100. Compare 2/4-GPU force components with one GPU, compare the 4-GPU energy with the 2-GPU distributed result at `1e-4 eV/atom`, and keep the raw one-to-multi-GPU energy offset as a diagnostic. | The lesson keeps capacity, OOM recovery, and speed as three different questions. It does not claim generic 1/2/4 energy parity, call the box equilibrated, claim bulk AIMNet2 accuracy, or say all memory divides by GPU count because each GPU runs the full reciprocal PME FFT and holds its workspace. Every timing row must pass `IQR / median <= 0.10`, and all GPU counts are reported together. |
| Keep supporting code out of the lesson | Put structure generation, parsing, plotting, repeated numerical reductions, reference formatting, and output-file mechanics in documented `aux/` modules. | Public Toolkit choices stay in the notebook; `aux/__init__.py` exports no competing tutorial API. |
| Keep the notebook readable | Use seven consistent stage cards, short explanations, visible compute progress, compact tables, result callouts, and descriptive figure text. | Every learner-visible code cell stays at or below 60 lines; the adapter keeps only its essential interface visible and moves repetitive support code to `aux/`. |

## Current learner path

1. One structure, one result.
2. The same calculation as a batch, including CPU/GPU and batch-layout
   measurements.
3. A 90-graph NCI model-composition and reference check.
4. A custom SevenNet-Omni adapter and fixed Cu(111) single points.
5. Molecular model composition, relaxation, harmonic preparation, and dynamics
   setup.
6. The full NVT and NVE trajectory, saved results, checks, and qualitative IR.
7. Inflight replacement on one GPU, a checked periodic base-box
   `DomainParallel` walkthrough, recorded multi-GPU capacity/OOM/speed results,
   and an offline `DistributedPipeline` contrast.

## Current size and pacing checks

- 129 notebook cells, including 34 hidden code cells.
- 54 learner-visible code cells and 1,791 visible source lines.
- 6,397 learner-facing Markdown words after HTML tags are removed.
- Stage 1 begins at cell 9; the first model result is at cell 13.
- The longest learner-visible code cell is 59 lines.
- An older six-stage run on one H100 recorded about 13 minutes of code time,
  including a 10-minute dynamics calculation. It is a pacing reference only;
  the current merged notebook remains unmeasured.

## Verification before release

- Rebuild the notebook twice and require byte-identical output.
- Validate notebook format and parse every code cell after IPython transforms.
- Run the learner-facing notebook checks.
- Run all helper and scientific-analysis tests in the declared environment.
- Run the complete notebook on one H100 without runtime patches.
- Run the single-GPU size sweep in fresh H100 processes, then require
  one-versus-multi-GPU componentwise force agreement and 2-versus-4-GPU
  energy agreement at `1e-4 eV/atom` for the exact AIMNet2 + PME + D3
  composition before plotting it. Keep the raw one-to-multi-GPU energy offset
  as a diagnostic rather than an acceptance check.
- Confirm that `DistributedPipeline` correctness and timing remain
  **NOT REPORTED** while stock Toolkit does not transfer every required batch
  field. Record the separate stage-pipeline campaign as a later update only
  after that stock path passes its maintained checks.
- Review the rendered notebook manually for pacing, table width, figure size,
  collapsed cells, and callout consistency.
- Complete the dataset, checkpoint, and redistribution review recorded in
  `THIRD_PARTY_NOTICES.md`.

## Current release status

The source notebook is ready for source and content inspection only. It is not
yet ready for an executed learner review or release.

The results below were observed on 2026-07-27 in the `v2` checkout on
`P3-Ultra`.

| Check | Current result |
|---|---|
| Source notebook | 129 cells, 54 learner-visible code cells, 34 hidden helper cells, and no visible code cell longer than 59 lines |
| Notebook identity | SHA-256 `bca20fc3436232b3282322b3c2aafc175c8fc39cc0e52288fcea09c5d8a8ba32` |
| Deterministic generation | Two independent rebuilds produced the same notebook bytes |
| Executed notebook | Not available for the current source; all 88 code cells have null execution counts and no saved outputs |
| Current focused tests | The 1/2/4-GPU campaign conversion passed 192 domain, result, launch, summary, and notebook checks. |
| Broad local compatible tests | 806 passed with 2 subtests against pinned Toolkit Core `331d6b2` and Toolkit-Ops `e8e7a74`. Retired patched-pipeline and OrbMol-only tests were excluded because those files are not part of the learner source. This does not replace the target H100 run. |
| Source checks | All shell and Slurm launch files pass `bash -n`; all Python sources compile; active Python files pass Ruff's syntax and undefined-name checks; notebook JSON, the base-box checksums, the reference-data checksums, and `git diff --check` pass. Existing formatting outside the remaster was not rewritten. |
| Static learner review | The HTML render contains seven stage cards, seven progress bars, 32 callouts, and ten process diagrams. Scientific review led to interaction-level NCI route checks, corrected ionic-system charge wording, and an enforced upstream NCI revision check. Exact Toolkit 0.2 API review led to a fixed-step `FusedStage` in the offline pipeline sketch. No other medium or larger source issue remains from those reviews. |
| Rendered human review | Pending for the executed notebook in the target theme, including pacing, table width, figures, widgets, hidden inputs, and callout consistency |
| Current H100 execution | Not run for this source revision. The clean 1/2/4-GPU campaign is the next scheduled step after the signed-off source commit. |
| Recorded scaling results | `DomainParallel` 1/2/4-GPU capacity, agreement, and timing plus `DistributedPipeline` correctness and timing are **NOT REPORTED** |
| Container build | Not run because no Docker-compatible container engine is installed locally |
| Git publication | Use the source commit recorded by Git. No remote `v2` branch is required when the exact commit is transferred to Compute Lab as a checked Git bundle. |

The setup job now places the pytest cache under its result directory, so its
test run does not make the staged source fail the later clean-checkout check.
The documented Toolkit checkout fallbacks match the staged paths, and the
runbook explicitly transfers the domain result directory from Compute Lab
before installing it locally. The first successful target build must still be
turned into exact Conda and Python lock inputs and reproduced once before the
environment can be called fully reproducible.

The H100 run must execute the complete notebook with the declared Core 0.2 and
Toolkit-Ops 0.4 environment. The domain runner must then read the distributed
energy before gather, reconstruct atom-level forces on rank 0, and check finite
values, atom identity, output shape, force agreement, and distributed-energy
agreement before the notebook may display recorded multi-GPU results.

The current checkout also retains historical Part 2 and Part 3 scientific data.
Those files must be excluded from this release, replaced with redistributable
data, or cleared for redistribution before the branch is staged.
