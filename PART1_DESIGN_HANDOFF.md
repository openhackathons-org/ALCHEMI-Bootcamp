# ALCHEMI v2 — Part 1 Design Handoff

Status: integrated Part 1 source implementation accepted on an NVIDIA H100
80 GB in CL job `3087665` with exit 0, updated 2026-07-13. The accepted run
executed 31/31 code cells with no error outputs, completed the exact 5,000-step
NVT + 50,000-step NVE route and all implementation and artifact gates, and
finished with all 14/14 live progress cards at `COMPLETE`. Every comparative IR
claim remained withheld under its declared thermal or topology gate. The accepted
source notebook SHA-256 is
`5403dfcd42bb707e15527a443e76edaec38fe38a8888ab8d527433b1dbf8efc8`;
the complete trajectory SHA-256 is
`ca2251061694e067f317fdb01d044897c8d913aff67e90e8f88ea5aaa6597f88`.
The current learner-facing presentation revision is
`81124de2e95e709a527522d026288a2c98d7e41b90ce7c4dd93e17a557b5a667`.
It changes visual presentation only, except for truthfully relabeling one
CPU/GPU observation callout; it has not been rerun on H100. The accepted source
remains preserved in the job `3087665` bundle.
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
- its 1268.6 s dynamics-cell time, 25:33 job time, and artifact hashes may be
  retained only as provenance and rough performance diagnostics for the
  defective DSF-era execution;
- the historical DSF + D3 calculator-parity numbers remain an implementation
  diagnostic for that old composition, not validation of the current
  `simple` all-pairs composition or its dynamics.

The independent checksummed B97-3c harmonic reference artifacts are not
derived from that trajectory and remain separate reference calculations.

## Current fused-stage validity gate

The NVT stage now receives a never-passing force criterion
(`ConvergenceHook.from_fmax(threshold=-1.0)`), so its declared step count is the
only NVT→NVE transition condition. The recorder checks the live fused-stage
status and the notebook plus external validator require exactly:

```text
status_0_warmup_steps       5000
status_1_production_steps  50000
```

Any early migration, mixed graph status, missing warm-up call, or missing
production frame is a hard failure before spectra or physical gates are
interpreted. CL job `3087665` passed this gate with exactly 5,000 status-0 calls
and 50,000 status-1 calls. Its dynamics cell took 1,232.4 s (20.54 min). The
separate scheduler elapsed time for the complete job was 23:02.

## Compilation boundary and H100 diagnostics

The eight-point dimer scan, variable-size performance batches, component
ablation, and editable trial remain eager. Compiling those changing graph
shapes would pay shape-specific compilation costs in the teaching path. Default
`torch.compile(..., fullgraph=False, dynamic=False)` is applied only to a fresh
AIMNet backbone for the fixed 42-atom {H₂O, D₂O, (H₂O)₆, (D₂O)₆} batch. The
compiled backbone is then placed into the same Coulomb + D3 Toolkit pipeline
used by FIRE2 and dynamics.

Compilation is a correctness gate, not an assumed optimization. On the exact
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
- job `3086742` passed the exact fixed-batch gate. Compiled-minus-eager maxima
  were `2.8201048e-6 eV`, `2.5629997e-6 eV/Å`, and `8.9406967e-8 e` for energy,
  forces, and charges. Compiled-repeat maxima were `1.4088873e-6 eV`,
  `9.2349785e-7 eV/Å`, and `8.9406967e-8 e`;
- job `3086698` stopped at an earlier `2e-6 eV/Å` premeasurement force gate;
  it was not a compiler failure;
- earlier job `3086805`, retained here as compiler history, passed the
  fixed-batch gate. Compiled-minus-eager
  maxima were `2.8201048e-6 eV`, `2.6822090e-6 eV/Å`, and `8.9406967e-8 e`;
  compiled-repeat maxima were `2.5342160e-8 eV`, `5.3644180e-7 eV/Å`, and
  exactly `0 e`;
- final accepted job `3087665` passed the same gate. Compiled-minus-eager
  maxima were `2.7947626e-6 eV`, `2.5928020e-6 eV/Å`, and `8.9406967e-8 e`;
  compiled-repeat maxima were exactly `0 eV`, `4.7683716e-7 eV/Å`, and
  exactly `0 e`.

The compiler investigations are implementation diagnostics. Job `3087665`
also passed the independent route, artifact, charge, covalent-integrity,
cluster-connectivity, and energy-excursion gates. Passing those gates does not
authorize every comparison: the thermal-pair and initial-ring-persistence
gates below still control which IR differences may be reported.

## Pinned D3 runtime asset

Both Toolkit `DFTD3ModelWrapper` instances receive the same explicit
`param_file` and `auto_download=False`. The notebook reads
`ALCHEMI_D3_PARAM_FILE` when set and otherwise uses
`~/.cache/nvalchemiops/dftd3_parameters.pt`. The Slurm preflight and notebook
both require SHA-256:

```text
b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84
```

The source tree does not contain that tensor. It must be prewarmed by the
runtime setup, and it remains outside distributable tutorial artifacts until
its redistribution rights are confirmed. The exact path and hash are recorded
in the run manifest; a missing or different file fails before model
composition rather than starting an implicit download.

## One-line summary

Part 1 is a single **water hydrogen-bonding** tutorial on AIMNet2-2025 (MIT):
one dimer energy → serial curve → batched-equivalence proof → CPU/GPU crossover →
model build-up (residual + predicted-charge all-pairs Coulomb + pairwise
D3(BJ)) with a four-way ablation and a full canonical B97-3c endpoint
comparison → an isotope × cluster batch → live batched FIRE2 →
**live batched IR spectrum from AIMNet2 predicted charges** → persistence. One
model, one structure family, first result inside 10 minutes.

**Session question:** *Starting from one water-dimer interaction energy, can we
assemble the checkpoint-declared AIMNet residual, predicted-charge all-pairs
Coulomb, and pairwise D3(BJ), prove the complete curve evaluates identically in
one GPU batch, and compare its endpoint with full canonical B97-3c — then reuse
the same predicted charges to compute, live and batched, the infrared spectra
of water monomers and clusters?*

The four component curves are controlled ablations, and all four may be shown
against full canonical B97-3c to make the omitted terms visible. Only the
complete residual + all-pairs Coulomb + pairwise-D3 curve is interpreted as an
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

## Part 1 block plan (time / GPU / what it shows / APIs)

> The table started as the pre-implementation plan and now describes the
> integrated, H100-accepted source path. Measured GPU times below are from CL
> job `3087665`, not classroom estimates.

| # | Block | Class time | GPU (est) | Shows | APIs |
|---|---|---|---|---|---|
| 1 | First energy + model card | ~8 min | cold load ≤60 s + ~5 s | one residual dimer interaction energy + max force; model-card discipline (hash, MIT, B97-3c-derived residual, 5.0 Å cutoff, external pairwise D3 and charge-based electrostatics) | `AtomicData.from_atoms`, `AIMNet2Wrapper.from_checkpoint`, `model_config`, `active_outputs`, `compute_neighbors` |
| 2 | Serial dissociation curve | ~3 min (tail of B1) | ~10 s | eight frozen-monomer water-dimer separations from 2.5→5.0 Å, computed the obvious serial way | plain loop over `AtomicData` (no batching yet — on purpose) |
| 3 | Batch = the curve (equivalence) | ~7 min | ~10 s | same curve in one call; `max|E_serial−E_batch| < 1e-5 eV` — the trust anchor | `Batch.from_data_list`, `batch_idx`, `batch_ptr`, `num_graphs`, `to_data_list`/`get_data`, `index_select` |
| 4 | CPU/GPU crossover | ~9 min | measured live | first-call and 20-call warm timing at batch sizes 1, 8, 32, and 128; GPU crossover is measured rather than promised | `Batch` at scale, `.to()`, explicit synchronization, warm/cold timing discipline |
| 5 | Composition build-up | ~10 min | ~20–30 s | restore checkpoint-declared physics: predicted charges → finite-system all-pairs 1/r Coulomb with no cutoff, plus pairwise D3(BJ) from an explicit checksummed parameter file, in a dependent pipeline | `DFTD3ModelWrapper(param_file=…, auto_download=False)`, `PipelineModelWrapper`/`PipelineGroup(use_autograd=True)`/`PipelineStep(wire=…)`, `DirectCoulombWrapper` |
| 6 | 4-way ablation + B97-3c endpoint comparison | ~8 min (+3 min exercise) | ~30–40 s | {residual, +D3, +Coulomb, full} overlaid for one water curve; complete model vs separately computed full canonical B97-3c; partial curves remain incomplete ablations | `set_config`, batched calls, `index_select`, `segmented_sum` (charge conservation) |
| 7 | Fixed-batch compile gate + FIRE2 | ~11 min | 30.2 s compile/parity + 20.4 s relaxation | establish eager energy/force/charge outputs, default-compile only the fixed 42-atom batch, require compiled/eager/repeat parity, relax it, reset isotope-pair coordinates, re-evaluate, apply the final force/mass gates, then persist the validated batch once | `torch.compile`, `FIRE2`, `ConvergenceHook.from_fmax`, `register_hook`/`run`, `make_neighbor_hooks`, `NaNDetectorHook`, `ZarrData` |
| 8 | **IR spectrum (centerpiece)** | ≥22 min live compute | 20.54 min for all 55,000 updates | exact 5,000 status-0 NVT + 50,000 status-1 NVE route; only after that hard gate may topology, thermal state, and spectra be interpreted | `add_node_property` (H/D masses), `initialize_velocities`, `NVTLangevin` with never-pass convergence → `NVE` via `FusedStage`, per-graph T, `set_config(active_outputs={energy,forces,charges})`, custom `Hook`+`DynamicsContext` route/dipole recorder, `segmented_sum` |
| 9 | Persistence + manifest | ~5 min | none | save the relaxed batch to Zarr, persist the complete raw trajectory before analysis, reload artifacts, and emit a checksummed run inventory | `ZarrData`, source/API-visible replay, run manifest |

The defective 2026-07-10 job spent 1268.6 s in its dynamics cell, but that is a
diagnostic timing for the wrong phase route. The accepted corrected job spent
1,232.4 s on the exact route. The scientific workload remains 55,000 updates;
any classroom restructuring must be an explicit curriculum decision, not an
automatic cutoff.

## Block 8 detail (the IR centerpiece)

Three phases run on one batch of {H₂O, D₂O, (H₂O)₆, (D₂O)₆} (~42 atoms):

| Phase | Required route | Current H100 time | Notes |
|---|---|---|---|
| Warm-up `NVTLangevin` | exactly 5,000 status-0 calls | included in the 20.54 min combined dynamics cell | never-pass force convergence prevents early migration; no dipoles enter the production trajectory |
| Production `NVE` | exactly 50,000 status-1 calls | included in the 20.54 min combined dynamics cell | μ(t) = Σ qᵢ(t)rᵢ(t) per graph via `segmented_sum`; every dipole and position frame is retained |
| Spectrum and reporting | only after route validation | completed within the separate 23:02 scheduler elapsed time | ⟨μ̇(0)·μ̇(t)⟩ → I(ω); physical reporting gates run after raw persistence |

- **Route is the first gate:** the recorder rejects early stage migration and
  mixed graph status. Its result requires exactly 5,000 warm-up calls and
  50,000 production calls. The run manifest records those counts and the
  external validator requires the same exact dictionary.
- **Step count:** 50k @ 0.5 fs = 25 ps. The estimator differences the stored
  total dipole, removes each segment mean, and averages 5 ps Hann periodograms
  with 50% overlap. Fourier-bin spacing is ~6.7 cm⁻¹; the known Hann response is
  broader. dt = 0.5 fs is safer than 1 fs for X–H stretches.
- **Sampling:** O–H stretch ~3757 cm⁻¹ (period ~8.9 fs) ⇒ sample dipole every ≤4
  fs (~every 8 steps at 0.5 fs). Nyquist ≈ 33,000 cm⁻¹ ≫ any band; dt does not
  limit this planned band range.
- **Reporting boundary:** after the route gate passes, thermal-state, topology,
  charge, and energy gates decide which spectra or comparisons may be
  interpreted. Job `3087665` passed charge, energy-excursion, covalent, and
  connectivity checks. It did not preserve the initial cyclic ring in every
  cluster frame: H₆ retained the initial ring for `0.85718` of frames and first
  lost it at `1.2325 ps`; D₆ retained it for `0.97788` of frames and first lost
  it at `5.2825 ps`. The monomer isotope pair did not satisfy the 20% thermal
  matching tolerance, and all four comparative headline values were withheld.
- **Accepted claim:** "Approximate IR spectra from the
  model's own predicted-charge dipole (classical nuclei, ~25 ps): compare band
  regions. Interpret isotope or cluster shifts only when the paired trajectories
  pass the declared thermal-state and topology gates. Absolute intensities are
  not benchmarked. This is a live property demonstration on the workshop model,
  not a validated IR benchmark." In this run, all four comparative headline
  values are withheld: H₂O/D₂O fails the thermal-pair gate; H₆/D₆ passes that
  gate but fails initial-ring persistence; both cluster-minus-monomer
  comparisons fail thermal matching and ring persistence. Neutral clusters
  make the dipole origin-independent.

## Audit decisions incorporated into the current source

1. **Endpoint reference:** Block 6 loads separately computed, checksummed
   canonical B97-3c/def2-mTZVP values on the exact eight water AB/A/B geometry
   triplets. It does not substitute DESS66 CCSD(T) values or construct a
   “no-D3 B97-3c” target by subtracting only the pairwise term.
2. **Reference interpretation:** all four curves may be plotted against full
   B97-3c to expose the omitted terms, but only the full residual +
   predicted-charge all-pairs Coulomb + pairwise-D3 model is interpreted as an
   endpoint/reference comparison. The other three are incomplete ablations,
   not matched reference levels.
3. **Spectral resolution:** the current text reports the known 5 ps Hann
   response and Fourier-bin spacing, not a 3–7 cm⁻¹ resolving-power claim.
4. **Sampling boundary:** NVT uses a never-pass force criterion so only the
   declared 5,000-step limit can advance `FusedStage`. The recorder and external
   validator require exactly 5,000 status-0 warm-up calls followed by 50,000
   status-1 production calls; only the latter enter the dipole trajectory.
5. **Physical gates:** job `3087665` passed the fused-stage route before any
   scientific interpretation. Isotope coordinates and predicted properties
   matched before dynamics, masses fed `initialize_velocities`, and every
   cluster frame remained covalently intact and oxygen-connected. The original
   ring did not persist in every cluster frame, and the H₂O/D₂O thermal pair
   missed its tolerance; those downstream gates correctly withheld the
   affected comparisons.
6. **Toolkit contract:** the dependent `PipelineGroup`, automatic neighbor
   hooks, `FIRE2`, dynamics hooks, replay, and official AIMNet `simple` + D3
   parity checks use the pinned Toolkit APIs. Composition forces are also
   checked by finite difference. Standalone FIRE does not attach a convergence
   snapshot writer; after final force and isotope checks, the validated batch
   is written once through `ZarrData` and replayed.
7. **Notebook boundary:** learner cells contain no function or class
   definitions. Focused `aux/` modules hold structures, analysis, plotting,
   timing, persistence, and presentation mechanics; `aux/__init__.py` exports
   nothing. `AtomicData`, `Batch`, neighbors, wrappers, pipeline construction,
   FIRE2, dynamics, hooks, reductions, and Zarr remain visible Toolkit calls.
8. **Waiting states:** six static stage cards and fourteen live progress cards
   cover every cell measured at five seconds or longer on H100. In job
   `3087665`, all 14/14 live cards entered `RUNNING`, reached `COMPLETE`, and
   persisted their final widget state. Early FIRE convergence is reported as
   steps used against a limit rather than a partially filled "complete" task.

## Data / license status

- **Water-dimer endpoint:** the repository contains a checksummed canonical
  B97-3c/def2-mTZVP curve for eight separations, computed from 24 AB/A/B
  single-point calculations. The calculation keeps the full method, including
  its ATM/gCP convention; no partial B97-3c target is presented as exact parity
  with the Toolkit components.
- **DESS66x8:** acquisition and a broader benchmark were part of the superseded
  Part-3 proposal. DESS66 is not the current Part 1 endpoint dependency.
- **Water structures:** the dimer scan and H₂O/D₂O/cyclic-H₆/cyclic-D₆ IR batch
  are generated through the tested `aux` structure layer and are saved as
  inspectable artifacts. Part 1 does not claim a CCSD(T) hexamer-isomer ranking.
- **Checkpoint:** AIMNet2-2025 `aimnet2-b973c-2025-d3_0` ensemble member 0,
  isayevlab/aimnet2-2025 (HF), MIT, B97-3c, cutoff 5.0 Å; cite Anstine/Zubatyuk/
  Isayev, Chem. Sci. 2025, DOI 10.1039/D4SC08572H.
- **Toolkit D3 parameter tensor:** prewarmed runtime cache only, pinned by
  explicit path and SHA-256
  `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`;
  not bundled pending confirmation of redistribution rights.
- **Vibrational references:** gas-phase H₂O ν1 3657 / ν3 3756 / ν2 1595 cm⁻¹;
  D₂O ν1 2671 / ν3 2788 / ν2 1178 cm⁻¹ (NIST/HITRAN); isotope factor ~1.35–1.37.

## Settled decisions

- Model: AIMNet2-2025 `aimnet2-b973c-2025-d3_0` (MIT), member 0 (not the ensemble —
  ensemble would multiply per-step cost).
- Structure family: water hydrogen bonding throughout—one eight-point dimer
  separation curve, then {H₂O, D₂O, cyclic H₆, cyclic D₆}. Adsorption, benzene,
  and broad DESS66 benchmarking are not in Part 1.
- Composition interpretation: residual, residual + all-pairs Coulomb, and
  residual + pairwise D3 are incomplete ablations. The full composition is
  checked against the official AIMNet calculator configured for `simple`
  Coulomb + D3. All four curves may be shown against full canonical B97-3c for
  context, but only the full composition is interpreted as an
  endpoint/reference comparison.
- Compilation: changing-size scans and editable trials stay eager. Default
  `torch.compile` is accepted only on the fixed 42-atom production batch after
  eager/compiled and compiled/repeat energy, force, and charge gates pass.
- MD centerpiece: predicted-charge IR spectrum; batch = {H₂O, D₂O, (H₂O)₆, (D₂O)₆}.
- MD route: exactly **5,000 status-0 NVT steps**, then **50,000 status-1 NVE
  steps @ 0.5 fs (~25 ps)**. NVT uses a never-pass force criterion so only its
  step limit advances the fused workflow. Recorder and validator counts must
  match exactly before analysis.
- 7 concepts: (1) single-structure call + model card, (2) Batch = same physics,
  (3) batching economics/crossover, (4) composition + ablation, (5) FIRE2 relaxation,
  (6) MD → predicted-charge IR via Hook+DynamicsContext + segmented_sum,
  (7) persistence. The B97-3c endpoint comparison and CPU/GPU crossover are
  applications of these; neither changes the underlying scientific workload.

## Remaining scientific and curriculum questions

The passing run settled the batch composition and showed that the initial
cyclic hydrogen-bond ring is intermittent rather than persistent. These
questions remain:

- **Intensity and frequency validation:** decide whether to add reference
  dipole-derivative / IR-intensity values and matched trajectory ensembles.
  Until then, normalized spectra and raw band centroids remain inspectable
  observations rather than quantitative position, shift, or intensity claims.
- **Keep the VDOS contrast cell** (teaches selection rules, ~free) or cut for time?

## Remaining whole-Part-1 decisions

1. Add matched thermal ensembles and replicas before restoring quantitative MD
   isotope or cluster–monomer centroid differences.
2. Decide whether monomer dipole-derivative/intensity references are sufficient
   to support semi-quantitative intensity language; otherwise retain only
   inspectable normalized spectra and raw band centroids, without a quantitative
   frequency or intensity claim.
3. Fit the measured 23:02 scheduler elapsed time into the classroom
   flow without hidden short modes or hardware-dependent scientific cutoffs.
4. Decide whether the nearly free VDOS contrast earns its notebook space; it
   must remain secondary to predicted-charge IR if retained.

## Reference paths

- Governing docs: `REMASTER_PLAN.md`, `TUTORIAL_DESIGN_PRINCIPLES.md`,
  `TOOLKIT_API_CURRICULUM.md`, `RUNTIME_SNAPSHOT.md` (this directory).
- Accepted H100 bundle on the validation host, intentionally gitignored:
  `part-1-water-hydrogen-bonding-toolkit/outputs/h100-remaster-3087665/`.
- Historical working prototype:
  `part-3-toolkit-foundations/alchemi-toolkit-foundations.ipynb`. The current
  Part 1 implementation reduces its direct-Coulomb teaching path to a tested
  `aux` all-pairs wrapper while keeping public Toolkit construction,
  configuration, batching, hooks, and persistence visible in the notebook.
