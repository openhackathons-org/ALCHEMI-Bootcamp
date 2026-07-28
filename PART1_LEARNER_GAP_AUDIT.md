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
| Explain scaling beyond one batch | Demonstrate inflight replacement live with `InMemoryDataset`, `SizeAwareSampler`, `HostMemory`, stable `system_id`, and bounded active work. Load a checked 3,200-atom phenol/N-methylacetamide base box and static OVITO preview, replace finite Coulomb with PME, walk through the `DomainParallel` API on one GPU with no decomposition, and show the separate `DistributedPipeline` stage layout. | The learner can say which API fits many independent systems and which fits one oversized periodic system. Multi-GPU measurements appear only when the complete 1/2/4-H100 result set passes its input, output, and checksum checks. |
| Teach the intended domain API without exposing internals | Keep `DomainConfig`, `SpatialPartitioner`, `DomainParallel`, `partition`, `run`, and `gather` visible. Explain that `DistributedManager` creates the rank mesh, `grid_dims` controls spatial cells rather than the rank layout, each GPU owns one region plus a halo, the globally reduced energy stays on the local result, and `gather` reconstructs atom fields. | No private fields, hand-written collectives, ad-hoc multiprocessing, live giant-box packing, or live OVITO rendering appears in the learner path. The single-GPU walkthrough is labeled as one domain with no decomposition, and the Toolkit 0.2 example is limited explicitly to a neutral system. |
| Keep the large-box claim honest | Build and check the small base box once offline, then create one recorded 51,200-atom integer supercell so composition and density stay fixed without rerunning Packmol. Run that same input on 1/2/4 H100s with one warm-up and three measured fixed-structure energy/force passes. Compare 2/4-GPU force components with one GPU, require repeatable 2/4-GPU energies, compare the 4-GPU median energy with the 2-GPU median at `1e-4 eV/atom`, and keep the raw one-to-multi-GPU energy offset and one-GPU pass range as diagnostics. | The lesson reports every raw pass time and its median. It does not search for an OOM, claim generic 1/2/4 energy parity, call the box equilibrated, claim bulk AIMNet2 accuracy, or say all memory divides by GPU count because each GPU runs the full reciprocal PME FFT and holds its workspace. |
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
   `DomainParallel` walkthrough, three recorded fixed-structure passes for one
   51,200-atom input on 1/2/4 H100s, and an offline `DistributedPipeline`
   contrast.

## Current size and pacing checks

- 133 notebook cells, including 38 hidden code cells.
- 54 learner-visible code cells and 1,779 visible source lines.
- 6,304 learner-facing Markdown words after HTML tags are removed.
- Stage 1 begins at cell 9; the first model result is at cell 13.
- The longest learner-visible code cell is 59 lines.
- Exact job `3317215` recorded 12 min 51 s of code time and 13 min 1 s of
  notebook wall time on one H100 PCIe. Stage 6 took 9 min 27 s; all other
  setup and stage work took 3 min 24 s.

## Verification before release

- Rebuild the notebook twice and require byte-identical output.
- Validate notebook format and parse every code cell after IPython transforms.
- Run the learner-facing notebook checks.
- Run all helper and scientific-analysis tests in the declared environment.
- Run the complete notebook on one H100 without runtime patches.
- Run the same fixed 51,200-atom input in fresh 1/2/4-H100 jobs, then require
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

The source, exact H100 execution, reviewed notebook, and reviewed HTML are ready
for learner inspection. The remaining release check is the presenter's manual
review in the target notebook theme.

The local checks below were observed on 2026-07-27 in the `v2` checkout on
`P3-Ultra`. The exact GPU checks ran on Compute Lab H100 PCIe nodes.

| Check | Current result |
|---|---|
| Source notebook | 133 cells, 92 code cells, 54 learner-visible code cells, 38 hidden helper cells, and no visible code cell longer than 59 lines |
| Notebook identity | SHA-256 `39128b84f6e99d9ffdb5c561853335d4179c12b30776f0e8e6a5d68385d8270b` |
| Deterministic generation | Two independent rebuilds produced the same notebook bytes |
| Executed notebook | Job `3317215` completed all 92 code cells with no failed cell or error output. Code time was 12 min 51 s and notebook wall time was 13 min 1 s. |
| Exact pinned tests | Job `3317136` passed 814 tests and 2 subtests in the declared H100 environment. The scheduler job completed with exit `0:0`. |
| Post-run packaging tests | The portable-banner, reviewed-notebook, validator-link, and runbook checks passed 14 focused local tests. |
| Source checks | All shell and Slurm launch files pass `bash -n`; all Python sources compile and pass Ruff checks and formatting; notebook JSON, deterministic regeneration, the base-box checksums, the reference-data checksums, and `git diff --check` pass. |
| Automated browser review | Headless Microsoft Edge loaded the portable HTML with seven stage cards, 99 saved progress bars, all 92 code outputs, six canvases, and four interactive OVITO viewports. The banner and local OVITO bundle loaded without HTTP failures. No notebook error, failed request, body overflow, or missing-widget message was found. |
| Rendered human review | Pending for the executed notebook in the target theme, including pacing, table width, figures, widgets, hidden inputs, and callout consistency |
| Current H100 execution | Jobs `3311164`, `3311328`, and `3311123` completed on one, two, and four H100 PCIe nodes with exit `0:0` in `5:53`, `3:49`, and `3:54` of scheduler wall time. All source and result checksum indexes pass. |
| Recorded scaling results | The same 51,200-atom input passed every required output check. Median fixed energy/force times were `0.268238 s`, `0.273560 s`, and `0.228844 s`, or `1.00×`, `0.98×`, and `1.17×`. These three-pass tutorial timings are installed; `DistributedPipeline` correctness and timing remain **NOT REPORTED**. |
| Container build | Not run because no Docker-compatible container engine is installed locally |
| Git publication | Use the source commit recorded by Git. No remote `v2` branch is required when the exact commit is transferred to Compute Lab as a checked Git bundle. |

The setup job now places the pytest cache under its result directory, so its
test run does not make the staged source fail the later clean-checkout check.
The documented Toolkit checkout fallbacks match the staged paths, and the
runbook explicitly transfers the domain result directory from Compute Lab
before installing it locally. The first successful target build must still be
turned into exact Conda and Python lock inputs and reproduced once before the
environment can be called fully reproducible.

The complete notebook ran against Core commit
`331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409` and Toolkit-Ops commit
`e8e7a7464f6745277a156a3d6f433d06b58c60e3`. The domain runner read the
distributed energy before gather, reconstructed atom-level forces on rank 0,
and checked finite values, atom identity, output shape, force agreement, and
distributed-energy agreement before the notebook displayed the recorded
multi-GPU results.

The current checkout also retains historical Part 2 and Part 3 scientific data.
Those files must be excluded from this release, replaced with redistributable
data, or cleared for redistribution before the branch is staged.
