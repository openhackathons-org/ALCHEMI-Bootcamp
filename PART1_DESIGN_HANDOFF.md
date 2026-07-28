# ALCHEMI v2: Part 1 design handoff

This is an internal implementation handoff. It records exact run results and
open development work. It is not part of the learner path or the general
tutorial principles. Use the Part 1 README for learner instructions.

## Current curriculum and development status

The permanent tutorial order is:

1. **Part 1:** the water-centered Toolkit foundations notebook in
   `part-1-scalable-atomistic-workflows/`;
2. **Part 2:** the original adsorption tutorial in
   `part-2-batched-adsorption-toolkit/`; and
3. **Part 3:** the OLED/melting tutorial in
   `part-3-batched-melting-toolkit/`.

These directory names now match the permanent curriculum order. The active
Part 1 follows the seven-stage plan below. The focused NCI Atlas
study is now Stage 3 of the learner notebook; the broader DESS study remains a
research prototype outside the tutorial.

The current source selects Toolkit Core `331d6b2` and Toolkit-Ops `e8e7a74`.
Complete-notebook job `3317215` ran commit
`1eca73058c5bae4a164f3b07c3e12fa944030086` at those pins on one NVIDIA H100
PCIe. All 92 code cells completed and none failed. The code took `770.760 s`
(`12:51`), notebook wall time was `781.144 s` (`13:01`), and scheduler elapsed
time was `15:49` with exit `0:0`.

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

The final notebook changes only the displayed timing text. Its 92 code cells
are byte-for-byte the same as the notebook that ran.

## Historical focused H100 run: older six-cell Stage 3 source

Compute Lab job `3222436` completed with exit `0:0` on one NVIDIA H100 NVL.
The CUDA-synchronized sum of the six Stage 3 cells was `22.643220 s`; scheduler
elapsed time was `3:15` and is not used as the tutorial compute time. The run
included 90 graphs and 1,140 atoms, four cold AIMNet ensemble-member loads and
first calls, one shared D3 pass, the composed-model checks, the independent
force check, reference analysis, and plotting. Checkpoint downloads, kernel
startup, earlier cells, and result packaging were outside the measured range.

The current Stage 3 has ten code cells. Exact-source job `3317215` measured
them at `22.064 s`. That current result supersedes the historical focused
timing for classroom pacing.

The complete-model maximum MAE across the three curves was
`0.368655 kcal/mol` against DFT-D3 and `0.351350 kcal/mol` against
CCSD(T)/CBS. The official analytic force differed from the official 0.003 Å
total-energy finite difference by `2.545e-4 eV/Å`; the Toolkit analytic force
differed from the official analytic force by `2.980e-7 eV/Å`. All saved-file
checksums passed.

The timed source notebook SHA-256 is
`9cbb4c88a1483a83d04c82d9fb144fe992f7a0cc4f96ceed3ca51210d976fb11`.
The saved timed bundle covers only that historical source; it does not time the
current code. The ignored local result bundle is
`part-1-scalable-atomistic-workflows/outputs/h100-nci-stage3-3222436/`.

## Historical accepted H100 run with earlier Toolkit versions

The previous Part 1 source replaced the OrbMol-v2 custom-wrapper lesson with a
SevenNet-Omni `mpa` surface example. CL job `3189534` completed with exit
`0:0` in `15:37` on one NVIDIA H100 NVL. All 47 code cells ran with no error
outputs, and the external validator and portable checksum check passed. The
dynamics cell took `598.6 s` (9.98 min) for exactly 5,000 NVT steps followed by
20,000 NVE steps. The saved production trajectory contains 20,000 frames over
10 ps. Differencing the dipole produces 19,999 current samples, which give
exactly two complete 5 ps Hann windows with 50% overlap and a `6.6713 cm⁻¹`
frequency grid. This short trajectory is for a qualitative teaching spectrum;
it is not a trajectory-length convergence result.

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

The scheduler elapsed time was `15:37`. These measurements used Toolkit Core
`80aab5c` with Toolkit-Ops `e8e7a74`; they are pacing references, not
benchmarks for the current Toolkit versions. NCI/DESS work was not included or
timed.

The complete Toolkit model's eight-point water-dimer MAE against full B97-3c
was `0.632837 kJ/mol`. The harmonic calculation passed all eight numerical
checks and gave a six-mode frequency MAE of `22.2530 cm⁻¹` against B97-3c.
The SevenNet graph mapping passed. The largest direct differences were
`8.4771e-7 eV/atom` in energy and `9.3937e-5 eV/Å` in a force component.
Every saved cluster frame remained oxygen-connected. The H₆ and D₆
initial-ring fractions were `0.99685` and `0.96810`, and the maximum energy
excursion was `0.379774 meV/atom`. H₂O/D₂O missed the 20% temperature-pair
check with a relative difference of `0.227301`. H₆/D₆ passed that check
at `0.136539` but did not retain the initial ring in every frame. All four
MD-derived quantitative comparisons remain **NOT REPORTED**.

The run used the installed pinned Toolkit Core `80aab5c` and Toolkit-Ops
`e8e7a74` directly. It did not replace runtime attributes or patch source files
on the compute node. The accepted bundle is
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3189534/`. It
contains the executed notebook, reviewed notebook, standalone HTML, validator
report, and checksummed calculation outputs. Human review of the rendered
notebook remains pending.

CL job `3175650` remains the accepted historical run for the immediately
preceding OrbMol-v2 source. That job completed with exit `0:0` in `24:52` on one
NVIDIA H100 NVL, executed all 45 code cells with no error outputs, followed the
exact 5,000-step NVT + 50,000-step NVE route, and passed the external validator
and portable checksum check. Its dynamics cell took `1208.5 s`. Its
2,048-system inflight cell took `15.2 s`; this single execution is not a
benchmark. The complete Toolkit model's eight-point water-dimer MAE against
full B97-3c was `0.633 kJ/mol`.

That previous source, executed notebook, trajectory, and run-manifest SHA-256
values are respectively
`f32b9f253f41c6b8f1d614bf4bc03aa188328be26a136afa8c5601c7e0a8cfad`,
`d6fd94aded92d1e19f15f4fb6294551a6fed9322b52faac5b4738eece7aa1e58`,
`43d8b06b762d3b442c604fc3202b1a08f4943228d8fe64a96a715022d069ae51`,
and `d83534f7f70839bc534d3ea4a6d7a85054300f5978bac2b53191db7d5d9dc53e`.
The calculation-validator report has SHA-256
`3a0e3d50c183532664625d86b6511199c006fef8faa978b9b64a89fa6feef3c7`.
The manifest inventories 64 output files totaling 36,830,973 bytes. The
Markdown-reviewed notebook has SHA-256
`2830333cfa00dc566b2e206b18214133f5ac80eefc0083580359d7303cb09cf7`,
and its validator report has SHA-256
`41849f9a7a23fa04605e6642f477a9955e52fd9ba31feddb21e5fd3b1cfd01b9`.
Its portable checksum index has SHA-256
`698e9b6aafe9f058de9ff7f20ae8167243082b5f30aeb9b35faeac3fce880de5`.
The standalone reviewed HTML has SHA-256
`cd024829d3f675fb5b6a8d1aad38fe83b6c142016f1eba8e8cdc96945f841e56`.
Export checks found all six images embedded, all 45 progress cards marked
`COMPLETE`, all six top-level stages, two readable tables with no ellipses or
`NaN` values, and both relative links resolved. Browser automation was
unavailable, so human visual review remains pending.

For that historical source, the OrbMol-v2 table contains 16 rows across three graphs: 14 required exact
mapping checks passed, while two rows are informational. The largest
Toolkit-versus-native differences were `0.0001830657323201 eV/atom` in energy
and `0.0005995631217956 eV/Å` in a force component. The same wrapper then ran
FIRE2 on two graphs for 99 steps and finished at
`0.0495131158989658 eV/Å` maximum force. The harmonic calculation passed all
eight numerical checks, reached `1.1697242689681489e-5 eV/Å`, and gave a
six-mode frequency MAE of `22.245965851892645 cm⁻¹` against B97-3c.

Every saved cluster frame remained oxygen-connected. The initial-ring fractions
were `0.61552` for H₆ and `0.36392` for D₆, and the maximum energy excursion was
`0.3255208333 meV/atom`. All four MD-derived quantitative comparisons remain
**NOT REPORTED**: H₂O/D₂O failed the temperature-pair check, H₆/D₆ failed the
initial-ring check, and both cluster-minus-monomer comparisons failed their
temperature and topology checks.

For the previous source, CL test job `3176022` independently passed 451 tests and two subtests in
`62.90 s` of pytest time. It completed with exit `0:0` in `1:10`, with no skips
or failures; its 16 warnings were upstream deprecation warnings. It supersedes
the 449-test pre-release-copy run `3175651`. Stock Core
`80aab5c` still fails the distributed transfer and overlap checks, so multi-GPU
performance remains **NOT REPORTED**. At that point, D3 tensor rights, a clean
Docker build, an upstream distributed fix followed by a stock 1/2/4-H100
campaign, and human review of the rendered notebook were still listed as
release checks.

Job `3174963` is retained as an earlier display-audit run. It
completed all 45 cells and passed its scientific checks, but its rendered
tables contained truncated rows. Its bundle remains at
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3174963/`; job
`3175650` replaced it as the saved OrbMol-source display record.

Jobs `3149917` and `3150048` are retained as historical pre-OrbMol records.
Job `3149917` completed the former 43-cell MACE source in `24:47`; its source,
executed notebook, trajectory, and run-manifest SHA-256 values were
`e603c629208d7ba325e4de203a0efc73a2b601513ffdcf4a3cc0327c6cfe7a13`,
`0bae4aa14e31225194d2080f4ad720ff044f0bf29b880e3b42bdabf48f159e1a`,
`41ac428b3a6a025be44cb681628769d685f3f069bd969623fc3b4840fe6ec0df`,
and `204afb9219d02942162d04e20724975a52a0f16575245f4d283f50ae85f153a7`.
Its Markdown-reviewed notebook was
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3149917/alchemi-water-ir-reviewed-current.ipynb`,
with SHA-256
`ec65ae1d63857e4de715920226d679c217c0655a68bdd658f0e078572c440a76`.

Historical job `3150048` independently completed a stricter staged source in
`24:00`. Its source, executed notebook, trajectory, run-manifest, and
validator-report SHA-256 values were
`eb1f2d4400b00e038d5d4986242214ceaea90f1fbcb2275c401d11243900173c`,
`ab4e58a9565e63845bd28b4990d9b62896408399401ed3c835cf67cecdb36b5f`,
`f4cd8be861876244dd547d8dd74abd24b3a1174215f40e295e14939bc95131a9`,
`0ee3a2f6e6c74fde90f78fb8978b7c7b628515df2aef2d250c883508c1e52122`,
and `c7f6e1f0fe246d55a8b973779aa4ca9a518b70c41634926499e642ea098433d9`.
Its validator and portable checksums covered 57 files totaling 36,904,592
bytes. Neither historical job validates the current SevenNet-Omni adapter.

Development job `3173975` completed the 45-cell scientific workload but failed
the old validator because the notebook and validator compared topology values
at different floating-point precision. Jobs `3174716` and `3174859` then
stopped during setup while the Orb stress path and `cuda`/`cuda:0` device-name
handling were corrected. Regression tests cover both fixes; these jobs are
diagnostic history, not accepted results.

The 2026-07-10 trajectory remains invalid as IR science because its nominal NVT
warm-up ended almost immediately. This document supersedes the VACF framing and
the former “Part 3 only” placement in earlier notes. Part 2 is explicitly out
of scope here (parked).

## Superseded 2026-07-10 H100 run — diagnostic only

Pinned `FusedStage` supplied its default force-based convergence when the old
NVT stage did not override it. Because the input batch was already relaxed,
that criterion passed after roughly one step and the workflow migrated to NVE
instead of completing 5,000 NVT updates. Capturing 50,000 later frames and
finishing every notebook cell did not repair the phase schedule.

Consequences:

- CL job `3064655` is **not** a valid 5,000-step NVT + 50,000-step NVE result;
- its temperatures, topology history, spectra, isotope comparisons, and
  cluster comparisons are not scientific evidence and must not be cited;
- its 1268.6 s dynamics-cell time, 25:33 job time, and saved-file hashes may be
  retained only as historical records and rough performance diagnostics for the
  defective DSF-era execution;
- the historical DSF + D3 calculator-agreement numbers remain an implementation
  diagnostic for that old composition, not validation of the current
  `simple` all-pairs composition or its dynamics.

The separately computed checksummed B97-3c harmonic reference files are not
derived from that trajectory and remain separate reference calculations.

## Current fused-stage route check

The current source stores NVT and NVE step counters on every system.
`StageStepCounterHook` increments only the counter selected by the current
status, and `converge_after_steps(...)` advances that system only when its
counter reaches the requested value. The recorder checks the live fused-stage
status and counters; the notebook plus external validator require exactly:

```text
status_0_warmup_steps       5000
status_1_production_steps  20000
```

Any early migration, mixed graph status, missing warm-up call, or missing
production frame is a hard failure before spectra or physical checks are
interpreted. Historical SevenNet-source job `3189534`, which used earlier
Toolkit versions, passed this route check with exactly 5,000 status-0 calls and
20,000 status-1 calls. Its dynamics
cell took `598.6 s` (9.98 min), and the separate scheduler wall time was
`15:37`.

Historical OrbMol-source job `3175650` passed its earlier 5,000-status-0 plus
50,000-status-1 route. Its dynamics cell took `1208.5 s` (20.14 min), and its
separate scheduler wall time was `24:52`. Historical jobs `3149917` and
`3150048` also passed that 50,000-step production route for their saved
pre-OrbMol sources; their dynamics cells took `1195.6 s` and `1194.2 s`.

## Compilation scope and H100 diagnostics

The eight-point dimer scan, variable-size performance batches, component
ablation, and editable trial remain eager. Compiling those changing graph
shapes would pay shape-specific compilation costs in the teaching path. Default
`torch.compile(..., fullgraph=False, dynamic=False)` is applied only to a fresh
AIMNet backbone for the fixed 42-atom {H₂O, D₂O, (H₂O)₆, (D₂O)₆} batch. The
compiled backbone is then placed into the same Coulomb + D3 Toolkit pipeline
used by FIRE2 and dynamics.

Compilation is a correctness check, not an assumed optimization. On the exact
fixed batch, one synchronized compiled call must match an eager call, and a
second synchronized compiled call must repeat the first:

```text
compiled vs eager   energy < 5e-6 eV; forces < 5e-6 eV/Å; charges < 2e-7 e
compiled repeat     energy < 2e-6 eV; forces < 2e-6 eV/Å; charges < 1e-7 e
```

The H100 failure and isolation history explains why the default compiler mode
is now deliberately narrow:

- job `3086356` used `max-autotune-no-cudagraphs` and failed with a CUDA
  misaligned-address error while AOTAutograd/Inductor compiled the force
  backward. It produced no acceptable scientific result;
- job `3086516` tested each path in a fresh process. Standalone Coulomb,
  standalone D3, eager full with a fresh batch, and eager full with a reused
  batch passed. The compiled no-Coulomb, compiled no-D3, and compiled full
  variants all failed with the same misaligned-address class during the
  AOTAutograd/Inductor backward. This is a compiler-path diagnostic, not
  evidence that either physical component is wrong;
- job `3086643` then passed two synchronized calls for the full 24-graph scan
  with default compilation;
- job `3086742` passed the exact fixed-batch check. Compiled-minus-eager maxima
  were `2.8201048e-6 eV`, `2.5629997e-6 eV/Å`, and `8.9406967e-8 e` for energy,
  forces, and charges. Compiled-repeat maxima were `1.4088873e-6 eV`,
  `9.2349785e-7 eV/Å`, and `8.9406967e-8 e`;
- job `3086698` stopped at an earlier `2e-6 eV/Å` premeasurement force check;
  it was not a compiler failure;
- earlier job `3086805`, retained here as compiler history, passed the
  fixed-batch check. Compiled-minus-eager
  maxima were `2.8201048e-6 eV`, `2.6822090e-6 eV/Å`, and `8.9406967e-8 e`;
  compiled-repeat maxima were `2.5342160e-8 eV`, `5.3644180e-7 eV/Å`, and
  exactly `0 e`;
- historical OrbMol-source job `3175650` passed the same check. Compiled-minus-eager
  maxima were `1.6280e-6 eV`, `2.9802e-6 eV/Å`, and `8.9407e-8 e`;
  compiled-repeat maxima were `5.9605e-8 eV`, `3.5763e-7 eV/Å`, and `0 e`.

The compiler investigations are implementation diagnostics. Historical job `3175650`
also passed the route, saved-file, charge, covalent-integrity,
cluster-connectivity, and energy-excursion checks. Passing those checks does not
authorize every comparison: the thermal-pair and initial-ring-persistence
checks below still control which IR differences may be reported.

## Pinned D3 runtime asset

The first Stage 3 `DFTD3ModelWrapper` receives an explicit `param_file` with
`auto_download=True`, allowing Toolkit to create the official cache when it is
initially absent. The notebook then verifies that file's SHA-256 before
accepting its results. Later wrappers reuse the same explicit path with
`auto_download=False`. The notebook reads `ALCHEMI_D3_PARAM_FILE` when set and
otherwise uses `~/.cache/nvalchemiops/dftd3_parameters.pt`. The Slurm preflight
and notebook both require SHA-256:

```text
b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84
```

The source tree does not contain that tensor. The runtime setup may prewarm it,
or the first Stage 3 wrapper may create it. It remains outside the files
distributed with the tutorial until its redistribution rights are confirmed.
The exact path and hash are recorded in the run manifest; a different file
fails verification before model composition, and later wrappers do not
download a replacement.

The SevenNet surface composition uses the visible PBE-D3(BJ) parameters
`a1=0.4289`, `a2=4.4407 bohr`, `s6=1.0`, and `s8=0.7875`. Its pair cutoff is
the D3 reference value of 95 bohr (50.2718 Å) with no smoothing. This is a
method choice recorded in
`part-1-scalable-atomistic-workflows/aux/models/sevennet_config.py`, not a
small-system performance cutoff.

## Second MLIP wrapper lesson

Part 1 briefly leaves the AIMNet calculation because the next question contains
a Cu surface. AIMNet2's architecture can process periodic data, but the
pretrained checkpoint used in the water lesson supports 14 elements and was
trained primarily on molecular and intermolecular-complex data. Cu is not one
of the supported elements. The model change is therefore about the
checkpoint's chemical domain, not about whether its architecture can represent
periodic boundaries.

The replacement example adapts raw SevenNet-Omni 0.13 to Toolkit. It uses the
explicit `7net-omni` checkpoint and `mpa` task. SevenNet's documentation
identifies `mpa` as its recommended PBE(+U)-level task for broad use across
molecules, crystals, surfaces, and interfaces. The SevenNet software is MIT;
the official checkpoint record is CC BY 4.0 (Figshare DOI
10.6084/m9.figshare.30399814). Toolkit does not supply a
`SevenNetOmniWrapper`, so this is a real custom adapter rather than a second
download through an existing Toolkit wrapper.

After loading the checkpoint, the notebook reads its available tasks from
`modal_map` and sends the same clean Cu(111) and CO/Cu(111) two-graph batch
through `mpa` and the RPBE `oc20` task in separate calls. This shows how one
loaded checkpoint can switch targets. The table reports finite output shapes
and per-structure maximum forces. The full tensors remain inspectable,
but raw totals are not printed or compared because the tasks use different
atomic-energy references. The main surface workflow remains `mpa` plus the
separately configured pairwise PBE-D3(BJ) correction.

The visible `SevenNetOmniWrapper(nn.Module, BaseModelMixin)` shows the choices
that make the external model usable in Toolkit:

- `ModelConfig` declares only energy and forces and records periodic support;
- `NeighborConfig` requests full directed COO neighbors at the cutoff read from
  the loaded SevenNet model;
- `adapt_input` maps Toolkit atomic numbers, positions, graph indices, cells,
  two-dimensional periodic shifts, and neighbors to SevenNet graph fields;
- `forward` contains the raw SevenNet model call; and
- `adapt_output` maps SevenNet's total energy and force arrays back to Toolkit
  names and shapes.

Checkpoint loading and hash checks live in
`part-1-scalable-atomistic-workflows/aux/models/sevennet_checkpoint.py`. The
complete adapter class lives in
`part-1-scalable-atomistic-workflows/aux/models/sevennet.py`, and the notebook
generator inserts its marked learner-facing source. Focused helpers compare the
Toolkit graph mapping and the native SevenNet results, while the learner keeps
the public `AtomicData`, `Batch`, `compute_neighbors`, model composition, and
output inspection in view.

The scientific panel contains nine structures built with ASE:

- one clean 3×3×4 Cu(111) slab;
- the same slab with one initial CO, CO2, NH3, or CH3OH placement; and
- one isolated reference for each of those four molecules.

The clean slab and four adsorbate slabs form one batch with
`pbc=(True, True, False)`. The four gas molecules form one nonperiodic batch.
The adapter evaluation uses one model call per batch; one direct SevenNet call
per batch checks repeatability and slicing. One CO/Cu(111) result is also
compared with SevenNet's official ASE calculator, loaded independently from the
same verified checkpoint. After the separate SevenNet and D3 calls, the
notebook runs the composed `PipelineModelWrapper` on each batch and checks that
its energy and forces match the explicit component sum. With
`neighbor_adaptation="always"`, Toolkit satisfies SevenNet's full-COO request
and D3's separate 95-bohr neighbor request. It then computes

```text
E_ads = E_surface+molecule - E_clean_surface - E_molecule
```

No standalone Coulomb term is added: the `mpa` task predicts a total
PBE(+U)-level energy, exposes no charges through this wrapper, and does not
define the AIMNet checkpoint's predicted-charge decomposition. The SevenNet
study likewise adds D3 separately for D3-level benchmarks when the selected
task was not trained with D3. Because `mpa` combines datasets with different
PBE(+U) protocols, the result is described as a practical PBE-family + D3 model,
not one uniformly matched DFT setup.

All nine structures are labeled **ASE-generated initial placements; not
model-relaxed**. Part 1 performs fixed-starting-geometry single points only. The
four energy differences are fixed-geometry adsorption-energy estimates for the
API lesson, not equilibrium adsorption energies or a DFT accuracy benchmark.
The notebook explicitly labels adsorption accuracy and molecule ranking **NOT
REPORTED** because there is no matched DFT or experimental result, relaxation,
or site search. Forces are displayed for Cu and adsorbate atoms separately so
the learner can see that relaxation would be the next calculation; relaxation
and site search remain later-workshop material.

The numerical table compares Toolkit-adapter and raw SevenNet outputs, checks
one periodic graph against the official SevenNet ASE calculator, then compares
the Toolkit pipeline with the explicit SevenNet + D3 sum. These checks verify
the implementation, not the model's scientific accuracy. In historical H100
job `3189534`, which used earlier Toolkit versions, the graph mapping passed and
the largest direct differences
were `8.4771e-7 eV/atom` in energy and `9.3937e-5 eV/Å` in a force component. The
saved tables are `surface_adsorption_energies.csv`,
`surface_adsorption_forces.csv`, `sevennet_adapter_graph_mapping.csv`, and
`sevennet_adapter_numerical_agreement.csv`.

## One-line summary

Part 1 is one Toolkit-first workflow: inspect one molecular result, scale the
same calculation to batches, complete and check the molecular model on three
NCI Atlas interaction classes, connect a surface-capable SevenNet model through
a custom adapter, return to AIMNet2 for batched relaxation and IR dynamics, then
show inflight batching, spatial domain decomposition, and the separate stage
pipeline layout. Water provides the first structure and the dynamics example,
but the curriculum is organized around reusable Toolkit APIs rather than one
hydrogen-bonding problem.

**Session question:** *How does one supplied model call grow into a checked and
scalable Toolkit workflow?* The notebook answers this by batching independent
structures, composing missing physical terms, changing models when the
chemistry changes, running dynamics and IR analysis, keeping a larger queue
active, and finally separating the two multi-GPU cases: one periodic system
divided with `DomainParallel`, and many systems streamed through
`DistributedPipeline`.

The second-model question is narrower: *When Cu leaves the AIMNet2 checkpoint's
supported chemistry, how do we expose a raw surface-capable model through the
same Toolkit data, neighbor, batching, composition, and output APIs?*

If FIRE2 and dynamics overlap on different batches after an upstream fix, the
ideal steady-state two-stage speedup approaches
`(t_FIRE + t_dynamics) / max(t_FIRE, t_dynamics)` and cannot exceed 2×.
Transfers, setup, pipeline fill, and drain reduce the measured value. The
current blocking loop has not demonstrated that overlap, so this is a teaching
formula rather than a performance result.

The four component curves are controlled ablations, and all four may be shown
against full canonical B97-3c to make the omitted terms visible. Only the
checkpoint base + all-pairs Coulomb + pairwise-D3 curve is interpreted as an
endpoint/reference comparison. Distances from B97-3c for the three partial
curves mix the ablation effect with ML error; they are not accuracy measures for
matched electronic-structure levels. Canonical B97-3c also contains ATM and
gCP, and public checkpoint metadata does not establish an identical separable
partition after subtracting pairwise D3.

For these small nonperiodic water systems, the current composition uses the
official AIMNet calculator's default `simple` electrostatics: direct all-pairs
1/r Coulomb with no spatial cutoff and no periodic images. DSF is a scalable
finite-cutoff alternative for larger finite systems. Ewald and PME are
periodic methods and are not used for this vacuum-cluster endpoint.

Stage 7 then changes the boundary condition deliberately. Equal numbers of the
neutral phenol and N-methylacetamide molecules from NCI Atlas system `1.041`
are placed in a periodic box with Packmol. The AIMNet2 checkpoint base now feeds
predicted charges into `PMEModelWrapper`, while D3 remains an independent
direct-force group. This reuses NCI ensemble member 0,
`aimnet2-wb97m-d3_0`, and its checkpoint-declared D3(BJ) damping parameters.
The tutorial uses a finite 15 Å cutoff tapered over 12–15 Å, not reference
D3's much longer untapered range. The box is an unequilibrated scaling input,
not a liquid or formulation prediction.

PME setup uses the public `estimate_pme_parameters` function with a fixed
12 Å real-space cutoff, accuracy `1e-4`, and mesh safety factor `1.0`. The
estimator-derived `alpha` and power-of-two mesh dimensions are passed
explicitly to `PMEModelWrapper`. A fixed-charge check on the 3,200-atom
validation box builds its direct Ewald reference independently with
`estimate_ewald_parameters` at accuracy `2e-5`. The composed one-GPU model uses
separate, right-sized neighbor lists. The domain halo keeps the 4 Å D3
coordination-number margin exercised by the Toolkit 0.2 test, and the saved
campaign must pass force checks on 2 and 4 GPUs before the notebook displays it.

`DomainParallel` is the learner-facing API for dividing that one periodic
system across GPUs. A one-GPU call follows the same sequence but does not
partition atoms. In Toolkit 0.2, the globally reduced energy remains on the
local result while `gather` reconstructs atom fields such as positions and
forces on rank 0. A GPU region does not carry the input system charge, so
this lesson is deliberately limited to the neutral box; it is not a template
for charged periodic systems. The separate `DistributedPipeline` lesson still
covers many independent systems moving through different workflow stages.

The domain run has its own clean H100 path:
`scripts/part1_domain_plan.py` creates and checks one integer supercell of the
saved Packmol base box, `scripts/part1_domain_run.py` runs the public
`partition` → `run` → `gather` path, and
`scripts/slurm_part1_domain_decomposition.sbatch` evaluates that same
51,200-atom input independently on 1, 2, and 4 GPUs.
The strict notebook loader accepts only a complete checksummed result set. The
fixed input uses separate force and energy checks because Toolkit 0.2
reduces energy differently in the ordinary one-GPU and multi-GPU
`DomainParallel` paths. Every force component on 2 and 4 GPUs is checked
against one GPU. The 4-GPU energy is checked against the 2-GPU distributed
result using the median of three measured energies. The 2- and 4-GPU energy
ranges and the 4-versus-2 median difference must each remain within
`1e-4 eV/atom`. The raw one-GPU-to-multi-GPU energy offset and one-GPU pass
range are saved as diagnostics only. Every GPU count reports one untimed
warm-up, all three measured energy/force pass times, and their median.
The checked result set is now installed at
`part-1-scalable-atomistic-workflows/data/domain_decomposition/recorded/`.
The observed medians were `0.268238 s`, `0.273560 s`, and `0.228844 s` on
one, two, and four H100 PCIe nodes. That corresponds to `0.98×` on two GPUs
and `1.17×` on four GPUs relative to one GPU for this input. These are three
short tutorial passes, not a general performance benchmark.

## The key change this session: MD centerpiece is IR, not VACF/VDOS

Earlier drafts used a velocity-autocorrelation power spectrum (VDOS). We switched
the centerpiece to a **predicted-charge infrared spectrum**. Rationale:

- VACF→VDOS is a *density of states*: it finds every mode at its frequency but has
  no selection rules and no intensities. For a molecule you could only compare
  peak *positions* to experiment — the spectrum shape is not the recognizable IR
  one (e.g. the strongly IR-active ~1595 cm⁻¹ bend is weak in VDOS).
- IR *is* dipole dynamics: I(ω) ∝ FT⟨μ̇(0)·μ̇(t)⟩ (dipole-velocity form; flatter
  baseline, no explicit ω² multiply). All it needs is the dipole trajectory μ(t).
- AIMNet2 is charge-aware and predicts **geometry-dependent** per-atom charges, so
  μ(t) = Σᵢ qᵢ(t)·rᵢ(t) includes the charge-flux contribution (dq/dt·r)
  within this point-charge dipole. That makes a predicted-charge IR observable
  possible; it does not validate relative or absolute intensities.
- Narrative closure: the same predicted charges that built the finite-Coulomb term
  in the composition block now produce the IR spectrum. The charges do double
  duty; no unrelated method is introduced.
- Same or lower compute: charges fall out of the same forward pass (≈ free);
  the run stores one dipole 3-vector per graph per step, then evaluates a 5 ps
  Hann-window Welch spectrum of the differenced dipole.

Optional teaching beat: keep a ~30 s **VDOS contrast cell** beside the IR (VDOS =
every mode; IR = only modes that shake a dipole). Nearly free since velocities are
already in hand. IR is primary.

## Resolved implementation blocker: charges at every dynamics step

The pinned Toolkit path was verified and exercised end to end. Requesting
`charges` through `active_outputs` leaves the most recent prediction on
`batch.charges`. `PredictedChargeIRHook` reads it at `AFTER_STEP`; the priming
force evaluation does not emit that stage, so it cannot create a false first
frame. The current workflow uses one shared forward per step and does not
re-run the model for charges.

## Active seven-stage Part 1 plan

The learner sees one continuous path. Toolkit calls and the decisions that
change a calculation stay in the notebook; structure factories, table
formatting, plotting, signal processing, file checks, and presentation code
stay in focused `aux/` modules.

Before Stage 1, a short framework primer places Toolkit Core, Toolkit-Ops,
PyTorch, JAX, and Warp in the correct layers. One GPU cell runs the same
`segmented_sum` through the PyTorch and JAX bindings and the lower-level Warp
API, checks matching values, and shows PyTorch and JAX gradients. The remainder
of Toolkit Core, model execution, and dynamics stays on the PyTorch path.

| Stage | Purpose and learner result | Main visible APIs |
|---|---|---|
| 1. One structure, one result | Convert one water structure, run AIMNet2, and inspect energy, forces, charges, precision, cutoff, supported elements, and active outputs. | `AtomicData.from_atoms`, `Batch.from_data_list`, `AIMNet2Wrapper.from_checkpoint`, `model_config`, `active_outputs`, `compute_neighbors` |
| 2. The same physics in a batch | Reproduce a serial dimer curve in one variable-size batch, recover each graph, measure cold and warm CPU/GPU behavior, compare same-size and mixed-size layouts, and use Core to choose a model-compatible neighbor implementation. | `Batch`, `batch_idx`, `batch_ptr`, `get_data`, `to_data_list`, `index_select`, `.to()`, `NeighborConfig`, `compute_neighbors`, `make_neighbor_hooks` |
| 3. Complete and validate the molecular model | Pack 90 NCI Atlas AB/A/B graphs, evaluate four ensemble members and one shared D3 pass, inspect all four model combinations, check charges, component sums, graph order, and one force, then compare the complete curves with near-matched DFT-D3 and independent CCSD(T)/CBS references. | `AtomicData`, `Batch`, `AIMNet2Wrapper`, `DFTD3ModelWrapper`, `PipelineStep`, `PipelineGroup`, `PipelineModelWrapper`, `set_config`, `segmented_sum` |
| 4. Bring a materials model into Toolkit | Explain why the molecular checkpoint does not cover Cu, implement the SevenNet energy/force adapter, inspect the checkpoint's tasks, reuse one two-graph batch for an `mpa`/`oc20` model sweep, check Toolkit/native agreement, and evaluate the small fixed-geometry Cu(111) panel. | `BaseModelMixin`, `ModelConfig`, `NeighborConfig`, `NeighborListFormat`, `adapt_input`, `adapt_output`, `direct_derivative_keys`, `Batch`, `index_select`, `compute_neighbors` |
| 5. Prepare the IR calculation | Check eager, compiled, and repeated results on one fixed batch; relax four systems with `FIRE2`; restore matched isotope coordinates; and calculate the harmonic reference. | `torch.compile`, `FIRE2`, `ConvergenceHook`, `make_neighbor_hooks`, `NaNDetectorHook`, `ZarrData` |
| 6. Run, save, and inspect the trajectory | Run exactly 5,000 NVT and 20,000 NVE updates, save the raw trajectory before analysis, validate route and physical checks, build predicted-charge IR spectra, and keep the MD, B97-3c harmonic, and observed-position comparisons separate. | `initialize_velocities`, `NVTLangevin`, `NVE`, `FusedStage`, `Hook`, `DynamicsContext`, `segmented_sum`, `ZarrData` |
| 7. Scale queues and single systems | Process a larger queue with bounded inflight batching; load a checked periodic phenol/N-methylacetamide box; replace finite Coulomb with PME; run the `DomainParallel` API on one GPU; then inspect three checked fixed-structure passes for the same 51,200-atom input on 1, 2, and 4 H100s when present. Contrast this with the separate `DistributedPipeline` construction. | `InMemoryDataset`, `SizeAwareSampler`, `HostMemory`, `PMEModelWrapper`, `estimate_pme_parameters`, `DomainConfig`, `SpatialPartitioner`, `DomainParallel`, `partition`, `run`, `gather`, `DistributedPipeline`, `BufferConfig` |

The historical 2026-07-10 job spent 1268.6 s in its dynamics cell, but that is
a diagnostic timing for the wrong phase route. Old-pin job `3189534` spent
`598.6 s` on the exact 25,000-update route. The 20,000-step production record
is an explicit teaching choice: it keeps the live demonstration short while
retaining two complete analysis windows. It is not presented as a converged IR
spectrum. Current-source job `3317215` measured the complete Stage 6 section at
`567.180 s` on one H100 PCIe.

## Stage 6 detail: the IR centerpiece

Three phases run on one batch of {H₂O, D₂O, (H₂O)₆, (D₂O)₆} (~42 atoms):

| Phase | Required route | Old-pin H100 reference | Notes |
|---|---|---|---|
| Warm-up `NVTLangevin` | exactly 5,000 status-0 calls | included in the 9.98 min combined dynamics cell | a per-system counter and `converge_after_steps` prevent early migration; no dipoles enter the production trajectory |
| Production `NVE` | exactly 20,000 status-1 calls | included in the 9.98 min combined dynamics cell | μ(t) = Σ qᵢ(t)rᵢ(t) per graph via `segmented_sum`; every dipole and position frame is retained |
| Spectrum and reporting | only after route validation | completed within the separate 15:37 scheduler wall time | ⟨μ̇(0)·μ̇(t)⟩ → I(ω); physical reporting checks run after raw persistence |

- **Check the route first:** the recorder rejects early stage migration and
  mixed graph status. Its result requires exactly 5,000 warm-up calls and
  20,000 production calls. The run manifest records those counts and the
  external validator requires the same exact dictionary.
- **Step count:** 20k @ 0.5 fs = 10 ps. The estimator differences the 20,000
  stored dipole frames into 19,999 current samples, removes each segment mean,
  and averages exactly two complete 5 ps Hann periodograms with 50% overlap.
  Fourier-bin spacing is `6.6713 cm⁻¹`; the known Hann response is broader.
  dt = 0.5 fs is safer than 1 fs for X–H stretches.
- **Sampling:** O–H stretch ~3757 cm⁻¹ (period ~8.9 fs) ⇒ sample dipole every ≤4
  fs (~every 8 steps at 0.5 fs). Nyquist ≈ 33,000 cm⁻¹ ≫ any band; dt does not
  limit this planned band range.
- **Reporting rules:** after the route check passes, thermal-state, topology,
  charge, and energy checks decide which spectra or comparisons may be
  interpreted. Job `3189534` passed charge, energy-excursion, covalent, and
  connectivity checks. Every cluster frame stayed oxygen-connected, and the
  maximum energy excursion was `0.379774 meV/atom`. It did not preserve the
  initial cyclic ring in every cluster frame: H₆ retained the initial ring for
  `0.99685` of frames and D₆ retained it for `0.96810`. The monomer isotope
  pair's thermal relative difference was `0.227301`, outside the 20% tolerance;
  the cluster pair passed that check at `0.136539` but failed ring persistence.
  All four comparative headline values were not reported.
- **Accepted claim:** "Approximate IR spectra from the
  model's own predicted-charge dipole (classical nuclei, 10 ps): compare band
  regions. Interpret isotope or cluster shifts only when the paired trajectories
  pass the declared thermal-state and topology checks. Absolute intensities are
  not benchmarked. This is a qualitative live demonstration on the workshop
  model, not a converged trajectory or validated IR benchmark." In this run, all four comparative headline
  values are **NOT REPORTED**: H₂O/D₂O misses the thermal-pair check; H₆/D₆ passes that
  check but fails initial-ring persistence; both cluster-minus-monomer
  comparisons fail thermal matching and ring persistence. Neutral clusters
  make the dipole origin-independent.

## Audit decisions carried into the active Part 1

1. **Interaction references:** Stage 3 loads the checksummed 90-row NCI Atlas
   subset: three interaction classes, ten separations, and AB/A/B for every
   geometry. Absolute DFT energies use ωB97M-D3(BJ)/def2-TZVPPD; CCSD(T)/CBS
   supplies an independent interaction-energy reference.
2. **Reference interpretation:** the complete core + predicted-charge
   all-pairs Coulomb + pairwise-D3 model is the final prediction. Core,
   core + D3, and core + Coulomb are omissions that expose model composition;
   they are not standalone electronic-structure methods. The DFT level is
   near-matched rather than identical, and ensemble spread is model
   disagreement rather than calibrated uncertainty.
3. **Spectral resolution:** the current text reports the known 5 ps Hann
   response and Fourier-bin spacing, not a 3–7 cm⁻¹ resolving-power claim.
4. **Sampling rule:** per-system step counters and `converge_after_steps`
   advance `FusedStage` only after 5,000 NVT updates and 20,000 NVE updates for
   each system. The recorder and external validator require the same counts;
   only the NVE updates enter the dipole trajectory.
5. **Physical checks:** job `3189534` passed the fused-stage route before any
   scientific interpretation. Isotope coordinates and predicted properties
   matched before dynamics, masses fed `initialize_velocities`, and every
   cluster frame remained covalently intact and oxygen-connected. The original
   ring did not persist in every cluster frame, and the H₂O/D₂O thermal pair
   missed its tolerance; those downstream checks correctly left the affected
   comparisons **NOT REPORTED**.
6. **Toolkit API behavior:** the dependent `PipelineGroup`, automatic neighbor
   hooks, `FIRE2`, dynamics hooks, replay, and official AIMNet `simple` + D3
   numerical checks use the pinned Toolkit APIs. Composition forces are also
   checked against a 0.003 Å central difference of the independent official
   calculator's total energy. The pinned Toolkit loader casts constant atomic
   reference energies to float32, so finite differences of its roughly 4 keV
   absolute totals are ill-conditioned even though those constants have zero
   force. Standalone FIRE does not attach a convergence snapshot writer; after
   final force and isotope checks, the validated batch is written once through
   `ZarrData` and replayed.
7. **Notebook code:** the learner-facing `SevenNetOmniWrapper` class is
   intentionally visible because adapting a raw model is the lesson. Its single
   maintained source lives in
   `part-1-scalable-atomistic-workflows/aux/models/sevennet.py` and is inserted
   by the notebook generator. Checkpoint handling, graph checks, ASE structure
   generation, table construction, validation, analysis, plotting, timing,
   persistence, and presentation mechanics live in focused `aux/` modules;
   `aux/__init__.py` exports nothing. `AtomicData`, `Batch`, neighbors, wrapper
   configuration, PBE-D3(BJ) composition, pipeline construction, FIRE2,
   dynamics, hooks, reductions, and Zarr remain visible Toolkit calls.
8. **Waiting states:** the active notebook uses seven top-level stage cards and
   live progress cards for every code cell. Historical job `3189534`, which
   used earlier Toolkit versions, used the preceding six-stage layout, ran all
   47 code cells, and produced a
   reviewed notebook and standalone HTML. Automated checks are not a human
   visual review, which remains pending. That run established a 15:37 H100
   scheduler wall time and 9.98-minute dynamics time. Current-pin job `3317215`
   has now run the seven-stage, 92-code-cell source with no failed cells. Early
   FIRE convergence is reported as
   steps used against a limit rather than a partially filled "complete" task.
   The source also replaces illustration placeholders with accessible process
   diagrams and builds its final results summary from live errors, accepted
   campaign rows when available, and scientific checks.
9. **Stage-pipeline status:** the notebook shows the intended public wiring, then
   reports **NOT REPORTED**. The selected Core `331d6b2` fixes reusable-buffer
   capacity and waits for an asynchronous send before reusing its storage, but
   `Batch.put` still skips non-float32 segmented fields. Integer atom or system
   fields, including `atomic_numbers`, therefore fail the full-dtype transfer
   preflight. Its classic loop also performs a global completion check each
   iteration, so FIRE2/dynamics overlap has not been demonstrated. No local
   Toolkit patch is accepted. The retained producer also lacks a versioned rule
   for selecting fixed NVT/NVE work and does not record stage intervals needed
   to prove overlap. Those pieces must be added after the stock transfer fix.
   Then keep one pipeline active for the complete route. The two-GPU sampler
   owns all 8,192 systems; the two upstream samplers in the four-GPU route own
   4,096 systems each, with at most 512 active per pair. Collect final systems
   through `ConvergedSnapshotHook`, and include setup plus fill and drain in
   every measured repeat. This is collection throughput, not acceleration of
   one trajectory or model call.
10. **IR comparison rules:** finite-temperature predicted-charge MD,
    unscaled 0 K double-harmonic B97-3c, and selected observed gas-phase
    fundamentals occupy separate lanes. MD and DFT are independently
    normalized; experiment supplies positions only. No shared intensity scale,
    IR MAE, or independent ML-generalization claim is made.

## Data / license status

- **Water-dimer endpoint:** the repository contains a checksummed canonical
  B97-3c/def2-mTZVP curve for eight separations, computed from 24 AB/A/B
  single-point calculations. The calculation keeps the full method, including
  its ATM/gCP convention; no partial B97-3c target is presented as an exact match
  with the Toolkit components.
- **NCI Atlas:** the active Part 1 packages a 90-row CC BY 4.0 subset with
  source identifiers, method fields, and a SHA-256 check. The complete model is
  evaluated against both near-matched DFT-D3 and independent CCSD(T)/CBS
  curves. DESS66x8 remains outside the learner notebook.
- **Water structures:** the dimer scan and H₂O/D₂O/cyclic-H₆/cyclic-D₆ IR batch
  are generated through the tested `aux` structure layer and are saved as
  inspectable files. Part 1 does not claim a CCSD(T) hexamer-isomer ranking.
- **Surface panel:** ASE builds one 3×3×4 Cu(111) slab, four copies with initial
  CO, CO2, NH3, or CH3OH placements, and four isolated molecule references. The
  structure manifest records `pbc=(True, True, False)` for slab systems and no
  PBC for gas systems. Every structure is labeled **ASE-generated initial
  placements; not model-relaxed**.
- **Checkpoint:** AIMNet2-2025 `aimnet2-b973c-2025-d3_0` ensemble member 0,
  isayevlab/aimnet2-2025 (HF), MIT, B97-3c, cutoff 5.0 Å; cite Anstine/Zubatyuk/
  Isayev, Chem. Sci. 2025, DOI 10.1039/D4SC08572H.
- **Surface checkpoint:** SevenNet-Omni 0.13, keyword `7net-omni`, explicit
  `mpa` task. The SevenNet software is MIT and the official checkpoint record
  is CC BY 4.0 (Figshare DOI 10.6084/m9.figshare.30399814). The official
  103,162,838-byte file is pinned at SHA-256
  `ca81bd3aac9fc2696c93dd386615f5a0fe41b92ab9ed7f69fa9526baaa9bab64`.
  The task is PBE(+U)-level and the tutorial adds Toolkit PBE-D3(BJ)
  separately. Cite the SevenNet-Omni pretrained-model documentation and the
  cross-domain study, *Nature Communications* (2026), DOI
  10.1038/s41467-026-70195-8.
- **Toolkit D3 parameter tensor:** prewarmed runtime cache only, pinned by
  explicit path and SHA-256
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`;
  not bundled pending confirmation of redistribution rights.
- **Vibrational references:** the computed bundle contains canonical
  B97-3c/def2-mTZVP frequencies and dipole-derivative intensities. The separate
  experimental bundle contains six observed gas-phase H₂-¹⁶O/D₂-¹⁶O positions
  from Dinu et al. Table 1 under CC BY 4.0. That table cites Toth's H₂-¹⁶O
  stretch, H₂-¹⁶O bend, and D₂-¹⁶O studies. No experimental spectrum or
  intensities are included. Its record ID is
  `experimental-water-fundamentals-0169d7538d437008`, and its data SHA-256 is
  `0169d7538d437008dcfd790b3501ab3aa446515c19a5402fc31151f275c5c103`.
- **Domain-decomposition evidence:** selected Core `331d6b2` exposes the
  `DomainParallel` path used by Stage 7, including the AIMNet2 → PME group and
  the independent D3 group. The Packmol input and local loader checks pass. The
  runner reads replicated energy from the local result and gathered forces
  from the reconstructed atom fields, with finite-value, shape, atom-order, and
  exact-input and minimum-image position checks before saving success. The
  replacement H100 run
  keeps one 51,200-atom input fixed, performs one warm-up and three measured
  energy/force passes on 1/2/4 GPUs, compares 2/4-GPU forces with one GPU, and
  requires repeatable distributed energies before comparing the 4-GPU median
  with the 2-GPU median at `1e-4 eV/atom`. The raw one-to-multi-GPU energy
  offset and one-GPU pass range remain diagnostic. Compute Lab jobs `3311164`,
  `3311328`, and `3311123` completed with exit `0:0` in `5:53`, `3:49`, and
  `3:54` of scheduler wall time. The strict loader accepts the installed
  result set: all fixed evaluations, periodic-position checks, distributed
  energy checks, force checks, and the one-GPU PME-versus-Ewald check pass.
- **Stage-pipeline evidence:** the separate `DistributedPipeline` path still
  fails the full-dtype transfer preflight because `Batch.put` skips integer
  segmented fields. Its retained producer also lacks the fixed-work selection
  rule and stage-overlap records required above. No valid stage-pipeline timing
  result is ready. After those fixes, its strict loader must retain failed rows
  and accept only the complete planned repeat set with matching inputs and
  settings.

## Settled decisions

- Model: AIMNet2-2025 `aimnet2-b973c-2025-d3_0` (MIT), member 0 (not the ensemble —
  ensemble would multiply per-step cost).
- Main path: one water result and dimer scan introduce Toolkit data and
  batching; three NCI Atlas curves broaden the model-composition check; the
  {H₂O, D₂O, cyclic H₆, cyclic D₆} batch supports relaxation and IR. A separate
  nine-structure Cu(111) panel motivates the custom SevenNet adapter and shows
  fixed-geometry energy/force batching. It does not add relaxation, site search,
  or a DFT adsorption benchmark to Part 1. Broad DESS66 benchmarking remains
  out of scope.
- Composition interpretation: residual, residual + all-pairs Coulomb, and
  residual + pairwise D3 are incomplete ablations. The full composition is
  checked against the official AIMNet calculator configured for `simple`
  Coulomb + D3. All four curves may be shown against full canonical B97-3c for
  context, but only the full composition is interpreted as an
  endpoint/reference comparison.
- Compilation: changing-size scans and editable trials stay eager. Default
  `torch.compile` is accepted only on the fixed 42-atom production batch after
  eager/compiled and compiled/repeat energy, force, and charge checks pass.
- MD centerpiece: predicted-charge IR spectrum; batch = {H₂O, D₂O, (H₂O)₆, (D₂O)₆}.
- Surface adapter: SevenNet-Omni `7net-omni`, `mpa`, energy and forces only;
  periodic slab batch plus finite molecule batch; Toolkit PBE-D3(BJ); no
  external Coulomb or charge output.
- MD route: exactly **5,000 status-0 NVT steps**, then **20,000 status-1 NVE
  steps @ 0.5 fs (10 ps)**. Per-system counters and
  `converge_after_steps(...)` control both transitions. Recorder and validator
  counts must match exactly before analysis.
- Seven stages: (1) one structure and one result, (2) the same physics in a
  batch, including layout and performance, (3) molecular model composition and
  endpoint validation, (4) a custom materials-model adapter, (5) fixed-batch
  checks, relaxation, and IR preparation, (6) staged dynamics and IR analysis,
  and (7) inflight work, a periodic-box domain-decomposition lesson, the
  separate stage-pipeline layout, persistence, and the results summary. The
  B97-3c endpoint comparison and CPU/GPU crossover are applications of these
  stages; neither changes the underlying scientific workload.

## Remaining scientific and curriculum questions

Computed B97-3c dipole-derivative intensities and observed monomer positions are
now present. They do not create a common intensity scale with finite-temperature
classical MD. These questions remain:

- Add matched thermal ensembles and replicas only if a future revision needs a
  quantitative MD frequency or isotope-shift metric.
- Decide whether a VDOS contrast cell earns its space; it must remain secondary
  to predicted-charge IR.

## Remaining whole-Part-1 decisions

1. Add matched thermal ensembles and replicas before restoring quantitative MD
   isotope or cluster–monomer centroid differences.
2. Retain the explicit no-common-intensity/no-IR-MAE rule unless a matched
   experimental/computational protocol and metric are added.
3. Decide whether the nearly free VDOS contrast earns its notebook space; it
   must remain secondary to predicted-charge IR if retained.

## Remaining release checks

- Before staging the branch, exclude the retained historical Part 2/3
  scientific data, replace it with redistributable data, or confirm its
  redistribution terms as described in `THIRD_PARTY_NOTICES.md`.
- Keep the D3 parameter tensor external unless its redistribution rights are
  confirmed.
- Run the complete test suite inside the exact declared image rather than
  substituting the older local environment.
- Build the clean distributable Docker image and repeat its import/runtime
  smoke checks.
- Keep `DistributedPipeline` correctness and timing **NOT REPORTED** for this
  release. After an upstream stock revision transfers every required batch
  field correctly, record the separate 1/2/4-H100 campaign as a later tutorial
  update.
- Perform a human rendered review of the current notebook's progress cards,
  callouts, figures, headings, fixed-geometry wording, full energy/force tables,
  and not-reported states.

## Reference paths

- Current guidance: `README.md`,
  `part-1-scalable-atomistic-workflows/README.md`,
  `ALCHEMI_TUTORIAL_PRINCIPLES.md`, `TUTORIAL_DESIGN_PRINCIPLES.md`, and
  `TOOLKIT_API_CURRICULUM.md`. `REMASTER_PLAN.md` and
  `REFERENCE_DATA_PLAN.md` are historical research records.
- Target path for the replacement checksummed campaign bundle:
  `part-1-scalable-atomistic-workflows/data/compute_lab_pipeline_campaign/`.
  No valid bundle is present. No result or speedup should be cited from this
  path until the stock transfer issue is fixed and the strict loader accepts
  the complete planned repeat set.
- Installed `DomainParallel` result set:
  `part-1-scalable-atomistic-workflows/data/domain_decomposition/recorded/`.
  Its manifest SHA-256 is
  `af5d7461808491bfd38d7e6be0645842b551bfacd6bdfa694deeb7f845b4bd7c`.
- Experimental position bundle:
  `part-1-scalable-atomistic-workflows/reference/experimental_water_fundamentals/`.
- Accepted SevenNet-source H100 result set from earlier Toolkit versions,
  including the executed and reviewed notebooks, standalone HTML, validator
  report, checksum indexes, and
  complete output bundle:
  `part-1-scalable-atomistic-workflows/outputs/h100-remaster-3189534/`.
  Human review of the rendered notebook remains pending.
- Accepted historical OrbMol-source H100 bundle saved in this checkout and
  intentionally gitignored:
  `part-1-scalable-atomistic-workflows/outputs/h100-remaster-3175650/`.
  It contains the executed notebook, reviewed notebook and HTML, independent
  validator reports, portable checksum indexes, and the complete output bundle.
  It does not validate the current SevenNet adapter or Cu(111) panel.
- Immediate historical display-audit bundle:
  `part-1-scalable-atomistic-workflows/outputs/h100-remaster-3174963/`.
  Its calculations passed, but its rendered tables contained truncated rows.
- Historical pre-OrbMol H100 bundle:
  `part-1-scalable-atomistic-workflows/outputs/h100-remaster-3149917/`.
  It contains the immutable calculation bundle and the then-current
  Markdown-refreshed `alchemi-water-ir-reviewed-current.ipynb`. It does not
  validate the current SevenNet-Omni adapter. The earlier `3095065` bundle is
  retained only as previous-source history.
- Independent pre-OrbMol staged-source follow-up:
  `part-1-scalable-atomistic-workflows/outputs/h100-remaster-3150048/`.
  It validates the stricter distributed preflight source and historical MACE
  scan but does not validate the SevenNet-Omni replacement.
- Historical working prototype:
  `research-toolkit-foundations/alchemi-toolkit-foundations.ipynb`. The current
  Part 1 implementation reduces its direct-Coulomb teaching path to a tested
  `aux` all-pairs wrapper while keeping public Toolkit construction,
  configuration, batching, hooks, and persistence visible in the notebook.
