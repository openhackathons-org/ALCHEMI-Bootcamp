# ALCHEMI v2 tutorial design principles

Status: current Part 1 teaching contract plus retained design-review history,
2026-07-13.

## Current Part 1 implementation — 2026-07-13

The molecular-interaction, batching, composition, relaxation, and live-IR story
is now the learner-facing **Part 1** rebuild. This placement supersedes the
2026-07-10 proposal to reserve that cohesive lesson for Part 3. The
`part-3-toolkit-foundations` research and runtime-validation harness remains in
the repository; later headings that retain “Part 3” record the superseded
learner story, not the removal of that harness.

The implemented scientific contract is:

- one pinned `aimnet2-b973c-2025-d3_0` ensemble member in its molecular domain;
- one water structure family, from a dimer separation curve through batched
  water-cluster relaxation and predicted-charge IR dynamics;
- four deliberately incomplete/completed views: residual, residual +
  finite-system all-pairs Coulomb, residual + pairwise D3(BJ), and residual +
  all-pairs Coulomb + pairwise D3(BJ);
- implementation parity between the Toolkit composition and the official
  AIMNet calculator configured for `simple` Coulomb plus the same pairwise-D3
  choice;
- one endpoint/reference interpretation for the **complete** Toolkit model
  against separately calculated canonical B97-3c/def2-mTZVP interaction
  energies, with the partial curves shown only as ablations;
- eager execution for changing graph shapes, with default `torch.compile`
  reserved for the fixed 42-atom IR batch and accepted only after
  eager/compiled/repeat parity;
- one explicit, checksummed D3 parameter file with automatic download disabled.

The four curves are an ablation, not a correction ladder: neither monotonic
improvement nor a one-to-one DFT component interpretation is claimed. Canonical
B97-3c includes D3(BJ)-ATM and gCP, while Toolkit's explicit D3 layer is the
pairwise C6/C8 contribution and the Coulomb term is defined by predicted AIMNet
charges. Public checkpoint metadata does not establish an identical separable
ATM/gCP partition. Therefore:

```text
residual                              incomplete ablation
residual + all-pairs Coulomb          incomplete ablation
residual + pairwise D3(BJ)            incomplete ablation
residual + Coulomb + pairwise D3(BJ)  complete Toolkit model
complete Toolkit model vs B97-3c      endpoint/reference comparison
```

The final line is not exact term parity and is not described as a matched
decomposition. The partial curves may be plotted against the full endpoint to
make omissions visible, but they are not independent electronic-structure
levels. DFT electrostatics is likewise not identified with an atom-centered
predicted-charge Coulomb term.

### The fused-stage route is a scientific gate

The 2026-07-10 H100 job is a superseded diagnostic run, not IR validation.
Pinned `FusedStage` supplied a default force-convergence condition to its NVT
stage; because the batch was already relaxed, that condition passed after
roughly one update and the workflow migrated to NVE instead of performing the
declared 5,000-step warm-up. Its temperatures, topology history, spectra, and
derived comparisons are not evidence. Old wall times and hashes may be retained
only as provenance for the defective execution.

The current NVT stage uses a never-passing force criterion, leaving its exact
step limit as the sole transition condition. The live recorder, notebook, run
manifest, and external validator all agree on:

```text
status_0_warmup_steps       5000
status_1_production_steps  50000
```

Any early migration or missing call invalidates the run before physical
interpretation. CL job `3087665` completed with exit 0 on an NVIDIA H100 80 GB
and passed the exact route and artifact gates: 31/31 code cells executed with no
error outputs, the dynamics cell completed in 1,232.4 s (20.54 min), and all
14/14 live progress cards reached `COMPLETE`. The separate scheduler elapsed
time was 23:02. The accepted source notebook SHA-256 is
`5403dfcd42bb707e15527a443e76edaec38fe38a8888ab8d527433b1dbf8efc8`;
the complete trajectory SHA-256 is
`ca2251061694e067f317fdb01d044897c8d913aff67e90e8f88ea5aaa6597f88`.
The current learner-facing presentation revision is `81124de…`; it has not
been rerun on H100, and the accepted source remains preserved in the job
bundle.
That execution success does not override downstream reporting gates. All four
comparative isotope/cluster IR values were correctly withheld: the monomer
isotope pair failed thermal matching, the cluster isotope pair failed
initial-ring persistence, and both cluster-minus-monomer comparisons failed
thermal and topology requirements. H₆ retained the initial ring for `0.85718`
of frames and first lost it at `1.2325 ps`; D₆ retained it for `0.97788` of
frames and first lost it at `5.2825 ps`.

### Compilation and downloaded assets are explicit contracts

Compilation should follow the stable scientific workload, not precede it.
Variable-size scans, component demonstrations, and learner-edited structures
stay eager. A fixed production batch may use default `torch.compile`, but only
after one synchronized compiled call matches an eager call and a second
synchronized compiled call repeats it for every reported model output. For the
current Part 1 batch, the gates are:

```text
compiled vs eager   energy < 5e-6 eV; forces < 5e-6 eV/Å; charges < 2e-7 e
compiled repeat     energy < 2e-6 eV; forces < 2e-6 eV/Å; charges < 1e-7 e
```

Final job `3087665` observed compiled-minus-eager maxima of
`2.7947626e-6 eV`, `2.5928020e-6 eV/Å`, and `8.9406967e-8 e` for energy,
forces, and charges. Compiled-repeat maxima were exactly `0 eV`,
`4.7683716e-7 eV/Å`, and exactly `0 e`.

A package's code license does not automatically establish redistribution
rights for a downloaded parameter tensor. Part 1 therefore passes
`param_file` explicitly to `DFTD3ModelWrapper`, sets `auto_download=False`, and
requires SHA-256
`b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.
The D3 tensor is provisioned by the runtime and is not bundled while its
redistribution rights remain unconfirmed. The path, hash, and download behavior
belong in the visible model contract and the run manifest.

## Superseded learner-facing implementation proposal — 2026-07-10

The following NCI-Atlas/ensemble/Part-3 plan is retained to show how the design
evolved. It was not the contract implemented in the Part 1 rebuild. The
corresponding notebook remains in the Part 3 research harness; “superseded”
here refers to the proposed learner-facing placement and narrative.

That proposal selected:

- four-member `aimnet2-wb97m-d3` ensemble;
- three ten-point NCI Atlas frozen-monomer curves spanning a neutral hydrogen
  bond, dispersion-dominated interaction, and ionic hydrogen bond;
- one 90-graph heterogeneous evaluation: `3 systems × 10 scales × (AB, A, B)`;
- explicit core, D3(BJ), and finite nonperiodic Coulomb ablations;
- near-matched ωB97M-D3(BJ)/def2-TZVPPD totals and independent CCSD(T)/CBS
  interaction curves;
- official Warp Tape computational graphs for one mixed model call versus
  three size-bucket calls, paired with the same fixed-workload timing comparison.

The proposed repository placement was **Part 3**, with the former adsorption
and melting notebooks left in Parts 1 and 2. The current Part 1 implementation
above supersedes that learner-facing placement; it does not erase the retained
Part 3 harness.

The proposal considered these comparisons:

```text
core + Coulomb       vs NCI DFT-D3 minus the same two-body D3(BJ)
core + Coulomb + D3  vs full NCI DFT-D3
complete model       vs NCI CCSD(T)/CBS
```

Even in that proposal, the NCI DFT basis was diffuse-augmented relative to
AIMNet training, so the teacher comparison was near-matched, not identical.
`core` and `core + D3` were incomplete ablations, not independent DFT levels.
The proposed structures, measurements, and publication gates live in
[`REFERENCE_DATA_PLAN.md`](REFERENCE_DATA_PLAN.md).

Older B97-3c, DESS66x8, S66-stage-plot, and Cu/CO references below document how
the design decision was reached. They are not the current implementation spec.
The general teaching principles and review rubric remain applicable.

## Retained Part 3 harness; superseded learner-facing decision

At the time of this review, `part-3-toolkit-foundations` was a useful research
and runtime-validation harness rather than a cohesive learner-facing tutorial.
It remains that broader harness. The proposal to promote it into the cohesive
molecular-interaction lesson was superseded: those teaching moves were carried
into the current Part 1 rebuild, while the Part 3 prototype remains available
for research and runtime validation.

The proposed tutorial question was:

> How can I evaluate many related atomistic structures in one GPU call without
> changing the scientific calculation?

The learner should use one family of structures from the first energy to the
final batch. The notebook should produce a real result in its first five to ten
minutes, prove that batching preserves the individual answers, measure the
CPU/GPU crossover on the learner's hardware, and leave behind a labeled result
table and inspectable structures.

There is one non-negotiable scientific constraint:

- AIMNet2 supports the explicit residual + D3 + charge-based Coulomb story, but
  it is a molecular model and cannot carry the tutorial into materials or
  adsorption.
- The MIT-licensed MACE-MP and MACE-MPA checkpoints support materials and a
  later adsorption prescreen, but they predict total PBE-like energies and do
  not provide partial charges. Adding a standalone Coulomb energy would double
  count physics.
- MACE-MH-1 covers molecules, surfaces, and bulk materials, but its ASL license
  permits academic non-commercial use only. It is not a safe default for
  NVIDIA training material without a separate commercial license.

No currently identified model gives us all three of these at once:

1. a defensible explicit D3 + electrostatics decomposition;
2. useful molecular, materials, and surface coverage;
3. permissive terms suitable for public NVIDIA material.

The proposed license-safe direction was specialization rather than one
nominally universal model. The molecular specialization is now Part 1:

- Part 1: AIMNet2 molecular interactions, explicit D3, finite-system
  electrostatics, batching, and GPU execution;
- Adsorption remains outside this notebook and requires a separately selected
  materials model and reference-validation plan.

## What the retained prototype actually showed

### The original stage plot used the wrong primary reference

The prototype compared every partial AIMNet stage to the stored S66
CCSD(T)/CBS interaction energy. That value is a high-level **total** interaction
energy. It contains physical electrostatics and London dispersion through the
electronic-structure calculation, but it does not contain an empirical D3
term.

CCSD(T)/CBS is useful as external scientific context for the completed model.
It is not the matched target that explains the AIMNet assembly. The old plot
should be removed from the teaching path. Its nonmonotonic behavior—only 27 of
66 systems improved at every residual → D3 → Coulomb stage—is an ablation
result, not a failed correction ladder.

An intermediate proposal attempted to make two matched checks by subtracting a
two-body D3(BJ) value from canonical B97-3c and comparing the remainder with
the AIMNet residual plus point-charge Coulomb. That partition is not
established by the public checkpoint contract: canonical B97-3c also contains
ATM and gCP, and its electronic electrostatics is not a unique atom-centered
predicted-charge Coulomb term. The implemented tutorial therefore does **not**
label the dispersion-subtracted curve as a reference level or parity target.

Interaction energies still use the same frozen-monomer subtraction on both
sides: `E(AB) - E(A) - E(B)`. The separately calculated B97-3c energies retain
the method's full canonical convention. All four Toolkit curves may be shown
against that endpoint for context, but only the complete model is interpreted
as an endpoint/reference comparison rather than exact term parity.

Retain the incomplete outputs as a **four-way component ablation**:

```text
residual only                external full Coulomb and D3 omitted
residual + all-pairs Coulomb  external pairwise D3 omitted
residual + pairwise D3       external Coulomb omitted
residual + Coulomb + D3      complete Toolkit model
```

This makes the roles of explicit electrostatics and dispersion visible without
making their order in a sequential ladder seem fundamental. Plot all four
against the full B97-3c curve for context, but mark only the complete-model
endpoint comparison as a reference comparison. The other three are
deliberately incomplete models. Their distance from the reference is an
ablation effect mixed with ML error, and monotonic improvement is neither
required nor claimed.

A hydrogen-bonded dimer and a dispersion-dominated dimer provide a useful
contrast: the Coulomb ablation should be conspicuous in the former and the D3
ablation in the latter. The conclusion must come from the computed curves, not
from the category label alone.

Do not compare AIMNet's point-charge Coulomb energy to a quantity labeled "DFT
electrostatics." DFT contains Coulomb physics, but it does not define the same
unique atom-centered point-charge decomposition. Validate the Coulomb
implementation against an analytic result and the native AIMNet calculator;
validate the **composed total** against DFT.

The historical proposal also considered S66 CCSD(T)/CBS as secondary context.
That comparison is not part of the current water-dimer endpoint block. If it is
restored later, the absence of public evidence that DESS66/S66 was excluded
from AIMNet2-2025 training means it must be called a reference comparison or
smoke benchmark—not an independent held-out test.

### Why the finite-cluster example does not use Ewald or PME

The historical prototype and the current Part 1 both use differentiable,
direct, nonperiodic all-pairs point-charge Coulomb. In the current notebook this
is AIMNet's official `simple` convention: no spatial cutoff and no periodic
images. The Toolkit composition is checked against the official AIMNet
calculator configured identically.

DSF is a scalable, force-continuous finite-cutoff alternative for larger finite
systems; it is not substituted into the current vacuum-water endpoint. Ewald
and PME are periodic electrostatics methods and would add periodic images, so
they are not used for these isolated clusters.

Use this boundary-condition rule in future material:

- small isolated finite cluster: direct all-pairs Coulomb is the reference
  finite-system convention used here;
- larger isolated finite system: a documented, converged finite-cutoff
  alternative such as DSF can be appropriate;
- neutral three-dimensional periodic system: Ewald or PME is appropriate;
- charged periodic cell: state the neutralizing convention and finite-size
  interpretation;
- periodic slab: ordinary 3D Ewald/PME can couple slab images, so use a
  slab-aware treatment or demonstrate vacuum/correction convergence.

Choosing Ewald or PME does not make an incompatible model decomposition valid.
It changes how a defined electrostatic term is evaluated; it does not justify
adding that term to a model already trained on the total energy.

### The reviewed prototype was a harness, not a story

The saved prototype has 34 cells, 859 lines of code, 128 lines of Markdown, no
saved outputs, and 17 Toolkit or Toolkit-Ops symbols. It moves through four
different scientific objects:

1. Lennard-Jones argon for CPU/GPU timing;
2. a custom all-pairs Coulomb implementation;
3. AIMNet2 and DESS66 molecular dimers;
4. MACE and CO/Cu(111) adsorption candidates.

The first clean single-point model evaluation never appears as its own learner
step. The custom wrapper arrives before the learner has obtained a useful
scientific result. The final adsorption section also depends on helpers from
the existing Part 1, so the new notebook is not self-contained.

The prototype should remain available as evidence that the APIs and numerical
checks work. The learner notebook should be rewritten around a smaller and
more deliberate path.

## Model and license decision

Model names are part of the scientific method. Do not write "MACE medium" or
rely on the package default. Print the exact checkpoint identity, checksum,
code version, target theory, intended domain, supported elements, dtype, and
weight license beside the loading cell.

| Checkpoint | Training target and domain | Weight license | Decision |
|---|---|---|---|
| `medium` | Original MACE-MP medium; MPTrj PBE(+U) materials | MIT | Avoid the ambiguous legacy alias |
| `medium-0b3` | MPTrj PBE(+U) materials; improved stability and reference behavior | MIT | Leading candidate for Part 2 prescreening |
| `medium-mpa-0` | MPTrj + sAlex PBE(+U) materials | MIT | Valid Part 2 candidate; currently runtime-tested |
| MACE-MH-1 | Multi-head molecules, surfaces, and bulk | ASL | Do not use without an NVIDIA commercial license |
| MACE-OFF / OMAT / MATPES / OMOL | Domain-specific newer checkpoints | ASL | Do not make a required NVIDIA path without legal approval |
| AIMNet2-2025 | Molecular residual + D3 + charge-based Coulomb convention | MIT | Current Part 1 model; not materials |

Current MACE documentation says that the default changed from `medium` to
`medium-mpa-0` in MACE 0.3.10. Explicit selection is required for a
reproducible tutorial. The official model registry lists both MACE-MP-0b3 and
MACE-MPA-0 as MIT-licensed materials checkpoints. It lists MACE-MH-1 as ASL.

The [ASL terms](https://raw.githubusercontent.com/gabor1/ASL/main/ASL.md) say
that the license is not open source, restrict use to academic non-commercial
work, and require a separate license for commercial use. A missing downloader
warning is not permission. Do not download, cache, bundle, or publish an
ASL-licensed checkpoint as part of the NVIDIA tutorial until NVIDIA legal and
the model licensor confirm the intended use in writing.

Relevant primary sources:

- [MACE foundation-model table](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)
- [MACE foundation-model registry](https://github.com/ACEsuit/mace-foundations#latest-recommended-foundation-models)
- [MACE-MH-1 model card](https://huggingface.co/mace-foundations/mace-mh-1)
- [MACE-MP foundation-model study](https://doi.org/10.1063/5.0297006)
- [AIMNet2-2025 model card](https://huggingface.co/isayevlab/aimnet2-2025)
- [AIMNet2 architecture and training](https://doi.org/10.1039/D4SC08572H)

### Composition rule

An API that can add two energies is not evidence that they should be added.
Composition is valid only when the base target excluded the added term or used
an explicit range-separated convention that the composition restores.

For the current candidates:

- AIMNet2 residual + its checkpoint-declared pairwise-D3 + charge-based
  Coulomb terms is the intended model inside its molecular domain.
- Treat residual, residual + all-pairs Coulomb, and residual + pairwise D3 as
  controlled incomplete ablations. They may be shown against full canonical
  B97-3c to expose omitted terms, but only the complete residual + all-pairs
  Coulomb + pairwise-D3 model is interpreted as an endpoint/reference
  comparison rather than exact component parity.
- MACE-MP/MACE-MPA + matched PBE-D3 can be scientifically defensible for a
  later materials example. MACE's public calculator supports this pattern.
- MACE-MP/MACE-MPA + standalone point-charge Coulomb is not defensible. These
  checkpoints predict total PBE-like energies and expose no partial charges.
- Ewald or PME cannot repair that double counting.

The precise MACE statement is not "MACE has no electrostatics." Standard
MACE-MP/MACE-MPA can learn local and screened electrostatic effects from total
DFT labels, but it has no explicit asymptotic electrostatic term, predicted
partial charges, or Ewald/PME solver. PBC support means cutoff-based periodic
neighbor images, not Ewald summation. Explicit electrostatic MACE variants
exist, but the identified weights are non-commercial and are not supported by
the pinned Toolkit wrapper.

MACE-MP can be useful for zero-shot screening, including preliminary surface
work, but it is not an adsorption authority. The foundation-model study treats
surfaces and adsorbates as transfer beyond the bulk training distribution and
reports that some reaction profiles require fine-tuning for quantitative
accuracy. Future wording should be concrete:

> Use this model ranking to select candidates for a reference calculation. Do
> not report it as a validated adsorption energy or site ordering.

## What effective tutorials consistently do

This synthesis uses established, project-maintained tutorials as evidence, not
as text or assets to copy.

### Start with the whole task

The [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
shows a complete workflow and a concrete prediction before sending readers
into deeper topic pages. The rebuilt notebook should likewise show structure →
model → energy in a few cells, then reopen each layer.

### Grow one authentic example

The [Software Carpentry Python lesson](https://swcarpentry.github.io/python-novice-inflammation/)
keeps returning to one scientific investigation while the program gains new
capabilities. ALCHEMI should keep the same structure family through serial
evaluation, batching, timing, and output analysis.

### Alternate action and interpretation

The [ASE molecular-dynamics tutorial](https://ase-lib.org/examples_generated/tutorials/md.html)
uses short run → plot → physical check → parameter-change loops. The strongest
check is physical, such as conservation or convergence, rather than merely
confirming that a cell executed.

### Teach sharp edges with executable contrasts

[Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
shows a tempting incorrect operation, its real failure, and the corrected
form. Its [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
separates compilation, transfer, and synchronized execution and allows the GPU
to lose on small problems. ALCHEMI should expose an unsynchronized timer and
then fix it, or compare loop and batch outputs before discussing speed.

### State the reader contract

The [NumPy tutorial style guide](https://numpy.org/numpy-tutorials/tutorial-style-guide/)
separates what the learner will do, learn, and need. It recommends concrete,
verb-led sections and moving exceptions out of the main path. Each ALCHEMI
notebook should open with those three short contracts and an expected runtime.

### Separate a tutorial from a recipe

The [OpenMM getting-started tutorial](https://openmm.github.io/openmm-cookbook/dev/notebooks/tutorials/getting_started.html)
explains the workflow and model boundaries; its short
[energy-contribution recipe](https://openmm.github.io/openmm-cookbook/latest/notebooks/cookbook/Analyzing%20Energy%20Contributions.html)
answers one narrow question. Advanced electrostatics, alternate models, and
custom wrappers should become small recipes when they interrupt the main Part
1 narrative.

### Carry an artifact into a real downstream task

The [NequIP tutorial](https://github.com/mir-group/nequip-tutorial/blob/main/NequIP_Tutorial.ipynb)
carries a model through testing, packaging, and a physical energy curve. This
tutorial should end with structures and a table a downstream tutorial can load.

### Explain batching as data representation

The [PhysicsNeMo graph DataPipe examples](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/minimal/datapipes/README.html)
make variable-sized graph batching visible: concatenate nodes, offset edges,
retain graph membership, and reduce back to one result per graph. This is a
better mental model than saying that the GPU "processes everything at once."

### Keep the workload fixed while teaching acceleration

NVIDIA's [Accelerated Python SVD lesson](https://github.com/NVIDIA/accelerated-computing-hub/blob/3419e2b3f40087fcbffd862f5a2669d4fdf918ec/tutorials/accelerated-python/notebooks/fundamentals/04__numpy_to_cupy__svd_reconstruction.ipynb)
shows a recognizable result first, then uses the same problem to reveal that a
GPU can lose at small sizes and win after the workload grows. Its
[power-iteration lessons](https://github.com/NVIDIA/accelerated-computing-hub/blob/3419e2b3f40087fcbffd862f5a2669d4fdf918ec/tutorials/accelerated-python/notebooks/fundamentals/06__asynchrony__power_iteration.ipynb)
also keep the numerical input fixed when comparing implementations. ALCHEMI
should compare the same coordinates, dtype, requested outputs, and stopping
conditions on CPU and GPU.

### Separate computational, performance, and scientific correctness

NVIDIA's [Warp Ising-model lesson](https://github.com/NVIDIA/accelerated-computing-hub/blob/3419e2b3f40087fcbffd862f5a2669d4fdf918ec/tutorials/warp/notebooks/02__ising_model.ipynb)
uses a fast but physically wrong GPU implementation to motivate the corrected
algorithm, then checks the result against an analytical solution. This is a
useful precedent for the nonmonotonic D3 result: an inconvenient result is
evidence to interpret, not a defect to edit out of the story.

### Make exercises small and self-checking

[GPU Puzzles](https://github.com/srush/GPU-Puzzles/blob/b3c4b237d7f0dc6d82055b753c8ea6e0cbb845eb/GPU_puzzlers.ipynb)
gives each exercise one task, a small expected code change, a visual mapping,
and immediate pass/fail feedback. Good ALCHEMI exercises are similarly narrow:
add one candidate, predict its batch membership, or find the first batch size
where GPU throughput exceeds CPU throughput.

### Use frequent, narrow learner checks

The [Carpentries exercise guidance](https://carpentries.github.io/lesson-development-training/instructor/formative-assessment.html)
recommends focused exercises and partially completed examples for novices. A
custom wrapper should be a small skeleton with one meaningful gap, not a blank
class or a 60-line implementation to read passively.

### Reuse mechanics, not prose

These sources have different content and code licenses. Their teaching moves
can inform our design, but no wording, figures, datasets, or code should be
copied without an asset-level license check and attribution plan. The NVIDIA
tutorial should use original prose and original figures.

This applies even to NVIDIA-hosted examples. At the revisions reviewed, the
Accelerated Computing Hub marks notebook code Apache-2.0 but written material
CC BY-NC-SA 4.0; CUDA-Q Academic similarly separates Apache-2.0 code from
CC BY-NC content; the inspected NVDLI notebook repository has no explicit
license. Those examples inform the lesson mechanics only. GPU Puzzles is MIT,
but there is no reason to copy its implementation into this tutorial.

## Core design principles

### 1. One notebook answers one question

Bad:

> Explore atomistic simulation, GPUs, batching, wrappers, dispersion,
> electrostatics, and adsorption.

Better:

> Evaluate the same set of structures one at a time and as one GPU batch. Check
> that the energies agree, then measure when batching becomes faster.

An API inventory is a reference page, not a tutorial narrative. The separate
`TOOLKIT_API_CURRICULUM.md` should remain the curriculum and coverage ledger.

### 2. Put a useful result before the architecture

Within five to ten minutes, the learner should have:

- seen the structure;
- loaded an exact checkpoint;
- evaluated one energy and maximum force;
- read the units and the boundary of the claim.

Only then unpack `AtomicData`, neighbors, model configuration, or batching.

Prototype order at the time of review:

> simulation overview → LJ benchmark → batch internals → 60-line wrapper →
> molecular result

Better order:

> structure → one energy → several related structures → one batch → why it
> works → how fast it is

### 3. Reuse one scientific object

The structure family should be the narrative spine. Do not use argon for the
CPU/GPU section, DESS66 for composition, and Cu/CO for batching. Generate a
scientifically meaningful series from the opening structure and keep using it.

If a second object is unavoidable, say why the model or boundary condition
changes and make that handoff the lesson. Never switch models silently.

### 4. Use a repeatable learning loop

Each section should follow this small grammar:

```markdown
## Batch eight structures

**Question:** Does batching change the predicted energy?

**Before you run:** Which array tells us which structure owns each atom?

[one short code cell]

**Observed:** one small table, plot, or rendered structure

> **Check:** one executable assertion or one-answer question
>
> **Why:** no more than two concrete bullets
>
> **Try it — 3 min:** change one scientifically meaningful input
```

The prediction makes the output matter. The modification makes the learner use
the idea rather than replay it.

### 5. Give each cell one job

A learner should be able to describe a cell with one verb: load, convert,
evaluate, compare, plot, or save. Setup belongs in a small import cell or a
tested helper. Scientific decisions stay visible in the notebook.

Do not bury these choices in helpers:

- model and checkpoint;
- D3 or electrostatic convention, parameter-file identity, and download
  behavior;
- charge, spin, cell, and PBC;
- cutoff and neighbor convention;
- eager versus compiled execution boundary and numerical parity tolerances;
- frozen atoms or other constraints;
- stopping criterion;
- output and reference units.

### 6. Prove batching preserves the answer before timing it

The batching section should first compare a serial loop with one `Batch` and
assert numerical agreement. Then show only the membership information learners
need: `batch_idx`, `batch_ptr`, `num_graphs`, and unpacking.

The section should end with a scientific output—such as the same energy curve
from both paths—not a printout of tensor shapes.

### 7. Let the CPU/GPU result be measured

Use the exact scientific workload from the preceding section. Report:

- first-call time separately;
- warm-up policy;
- explicit CUDA synchronization;
- what is inside and outside the timed region;
- latency and structures per second;
- CPU and GPU results across several batch sizes;
- hardware, dtype, package version, and checkpoint.

The question is "Where is the crossover on this machine?" not "Why is the GPU
always faster?" A CPU win at batch size one is a useful result.

### 8. Put the model contract beside the loader

Show a compact card containing:

- exact checkpoint and checksum;
- code and weight versions;
- code and weight licenses separately;
- training data and electronic-structure target;
- intended systems and supported elements;
- cutoff, PBC support, charge/spin assumptions, and dtype;
- whether dispersion or electrostatics is already represented;
- requested outputs.

Do not defer the applicability domain to a warning near the end.

### 9. Make every composition step earn its place

Before adding a term, answer four questions:

1. What target did the base model learn?
2. Was this term excluded or range-separated during training?
3. What boundary condition and parameter convention does the added term use?
4. What matched reference will test the composed result?

Plot the components because they explain the model convention, not because a
stacked bar chart looks physical. Do not treat fitted components as unique
observables.

### 10. Use scientific checks, not success messages

Prefer checks such as:

- serial and batched energy agreement;
- force equals the negative energy gradient;
- translation, rotation, or permutation invariance where applicable;
- charge conservation;
- energy conservation or optimizer convergence;
- agreement with an independently sourced reference;
- unchanged frozen atoms;
- initial and final structure inspection.

Keep deep regression gates in tests or a validation notebook. Expose only the
check that teaches the current concept.

### 11. Write callouts that help the next action

Useful labels:

- Before you run
- What changed?
- Check
- Why
- Boundary of this example
- Try it
- In practice

Avoid defensive walls of "guardrails" and meta-claims. A callout should change
what the learner predicts, runs, or concludes.

### 12. End with an artifact and a transfer task

The current Part 1 should save:

- the source and evaluated structures;
- a labeled energy/force table;
- the loop-versus-batch equivalence result;
- the measured CPU/GPU scaling data;
- a manifest with model, software, hardware, units, and provenance.

The final exercise should replace or perturb one supported structure and rerun
the same function. A downstream tutorial should consume these artifacts rather than rebuild
them through hidden setup.

### 13. Make every wait visible and semantically honest

Any cell measured at five seconds or longer on the target hardware should show
a styled progress card before expensive imports, model loading, compilation,
serial loops, relaxation, or dynamics begins. The card should expose a real
unit—checks, model calls, optimizer steps, or dynamics steps—plus elapsed time
and a final `COMPLETE` or `ACTION NEEDED` state. Preserve the final widget state
in the executed notebook so readers do not see an empty placeholder.
Live cards should enter `RUNNING` on their first visible frame; reserve the
timeless `STAGE` badge for static navigation cards.

For early convergence, distinguish work used from the declared limit. A FIRE2
card should say, for example, `181 steps used · limit 5,000` and show task
completion; a 3.6% rail with a `COMPLETE` badge looks contradictory. Static
stage-navigation cards should use a timeless `STAGE` label rather than remain
stuck at `READY` in an executed notebook.

### 14. Hide tutorial mechanics; expose Toolkit decisions

The notebook should contain no substantial helper definitions. Put structure
builders, plotting, signal processing, persistence, timing harnesses, and
presentation code in focused `aux/` modules. Keep `aux/__init__.py` empty of
re-exports so it never becomes a competing tutorial API, and document each
module's one responsibility.

Keep the learner-facing Toolkit path in executable cells: `AtomicData`,
`Batch`, neighbor construction, model wrappers and configuration,
`PipelineGroup`/`PipelineModelWrapper`, relaxation, dynamics stages, hooks,
segmented reductions, and persistence. The separation is not about hiding
code; it is about spending notebook attention on the ecosystem being taught.

## Superseded Part 3 learner narrative

This storyboard is retained because its single-object, early-result, batching,
and model-composition sequence informed the Part 1 rebuild. Its former Part 3
placement, runtime target, DESS66 scope, and two-level “matched” reference plan
are superseded by the current Part 1 contract at the top of this document. The
broader Part 3 research and runtime-validation harness itself remains.

Working title:

> **Build and batch a molecular interaction potential**

Central question:

> Can we build a complete molecular potential from the checkpoint-declared
> residual, predicted-charge electrostatics, and pairwise D3 terms—and evaluate
> one water interaction curve in a GPU batch without changing its answers?

Target: 25–30 core cells, 40–50 minutes, one optional performance extension.

### Opening contract

**What you will do**

- Evaluate one molecular dimer with an explicitly pinned AIMNet2 checkpoint.
- Evaluate a separation curve in a loop and as one batch.
- Prove that the answers agree and measure the CPU/GPU crossover.
- Compose the model's residual, D3, and finite-system Coulomb terms.
- Compare the complete model endpoint with pinned canonical B97-3c reference
  calculations.
- Save a reusable result table, structures, and provenance.

**What you will learn**

- How ASE structures become `AtomicData` and `Batch` objects.
- Why batching changes execution but not the underlying calculation.
- What a Toolkit model adapter declares and what it does not guarantee.
- How to benchmark asynchronous GPU work honestly.
- Why D3 and Coulomb are part of this checkpoint's scientific contract.
- Why a point-charge component is not a uniquely defined DFT observable.

**What you will need**

- one GPU, with a tested CPU fallback;
- the prewarmed, pinned MIT-licensed AIMNet2-2025 checkpoint;
- selected, redistributable DESS66x8 geometries;
- cached B97-3c reference values with a reproducible calculation manifest;
- approximately 40–50 minutes.

### Storyboard

1. **See the destination**
   - Render the structure family and show the final table/plot learners will
     create.
   - State the one scientific question and the expected runtime.

2. **Evaluate one structure**
   - Show one dimer before computing.
   - ASE `Atoms` → `AtomicData` → neighbors → exact AIMNet2 checkpoint →
     complete interaction energy and maximum force.
   - Put the model card directly below the loader.

3. **Turn the dimer into a separation curve**
   - Load or construct the eight frozen-monomer separations.
   - Evaluate the complete potential in a short serial loop.
   - Plot interaction energy against separation.

4. **Evaluate the same series as one batch**
   - Build `Batch.from_data_list`.
   - Visualize ragged membership once.
   - Assert serial/batch agreement and reproduce the same interaction curve.

5. **Find the CPU/GPU crossover**
   - Use additional DESS66x8 dimers or controlled repeats to form batch sizes
     such as 1, 8, 32, 128, and 528.
   - Separate first call from synchronized steady-state inference.
   - Plot latency and structures per second; interpret the measured crossover.

6. **Inspect and modify the model contract**
   - Request energy and forces explicitly.
   - Show cutoff, neighbor convention, PBC, dtype, and supported outputs.
   - Ask the learner to predict which data must change for a different output.

7. **Open the complete model**
   - Show the AIMNet residual and predicted charges.
   - Add the checkpoint-declared pairwise D3(BJ) term.
   - Add the official `simple` finite-cluster all-pairs Coulomb treatment.
   - Check charges, analytic two-particle Coulomb, forces, and parity with the
     native AIMNet calculator.
   - Display residual, residual + Coulomb, residual + D3, and the complete
     model together so each explicit term can be removed independently.

8. **Compare the complete endpoint**
   - Compare residual + all-pairs Coulomb + pairwise D3 with full canonical
     B97-3c as an endpoint/reference comparison.
   - Keep the incomplete variants visible as labeled ablations, not additional
     DFT levels or independent production potentials.
   - Do not label a two-body-D3-subtracted B97-3c value as a matched reference
     for residual + Coulomb; canonical B97-3c also contains ATM and gCP, and no
     identical separable checkpoint partition has been established.

9. **Package the result for the next part**
   - A small function accepts structures and a model and returns a labeled
     table.
   - Save the structures, table, timing data, and provenance manifest.
   - End with one three-minute structure modification.

### Historical structure-selection gates

Before implementation, the candidate structure family had to pass these gates:

- AIMNet-supported elements, charge state, and molecular domain;
- one meaningful coordinate or condition to scan;
- fast enough for serial and batched live execution;
- enough work to show a measurable GPU crossover after controlled replication;
- a full canonical B97-3c endpoint for every endpoint-fidelity claim;
- a licensed high-level reference for any external accuracy comparison;
- visual inspection of every source geometry and boundary condition.

The historical leading candidate was a small set of **DESS66x8
noncovalent-interaction curves**. The implemented Part 1 instead keeps a water
structure family from the dimer curve through relaxed clusters and IR; repeated
and mixed water graphs provide the performance workloads.

The DESS66 acquisition note below the old plan is not a current Part 1
dependency. Part 1 loads its separately computed, checksummed B97-3c dimer and
frozen-monomer endpoint values while computing AIMNet live.

## Where D3, electrostatics, adsorption, and wrappers belong

The tutorial sequence should be allowed to specialize.

### Current Part 1: molecular composition, relaxation, and GPU execution

- one AIMNet2 molecular interaction energy;
- a water-dimer separation curve and water-cluster continuation;
- serial versus `Batch` equivalence;
- CPU/GPU crossover;
- eager changing-size scans followed by a default-compiled fixed 42-atom batch
  with eager/compiled/repeat parity gates;
- checkpoint and model contract;
- predicted charges and `simple` finite-system all-pairs Coulomb;
- pairwise D3(BJ), official-calculator implementation parity, and a full
  canonical B97-3c endpoint/reference comparison, using an explicit
  checksummed D3 parameter file with automatic download disabled;
- batched FIRE2, predicted-charge IR dynamics, gates, and persistence;
- reusable artifacts.

### Part 2: materials adsorption and live relaxation

- choose and pin an MIT-licensed MACE materials checkpoint;
- generate adsorption candidates;
- native Toolkit hooks and FIRE2;
- convergence and frozen-atom checks;
- model-relative ranking;
- no standalone charge-Coulomb term;
- reference calculation handoff and explicit prescreening language.

### Later part: periodic or advanced long-range physics

- DSF as a scalable finite-cutoff alternative for larger finite systems;
- Ewald/PME on a genuinely periodic, charge-aware model and system;
- direct Coulomb versus periodic electrostatics chosen by boundary condition;
- dependent `PipelineGroup` composition and autograd;
- charged-cell and slab caveats.

This split exposes more of the Toolkit while keeping each notebook teachable.
It also prevents an unsupported "universal model" claim.

## Writing and visual style

### Prefer concrete sentences

Current:

> Research mode: live calculations, executable checks, explicit provenance.

Rewrite:

> Every energy below is calculated when you run the notebook. Stored reference
> values are used only for comparison.

Current:

> Accuracy comes from the complete setup.

Rewrite:

> The same coordinates can give different energies if the model, charge, cell,
> or boundary conditions change. We keep those choices fixed below.

Current:

> A hollow-site winner is not an experimental discovery.

Rewrite:

> Use this ranking to choose candidates for DFT. It is not a validated
> adsorption-site prediction.

Current:

> What this establishes

Rewrite:

> The batched and serial energies agree to numerical precision. We can now add
> candidates without changing the calculation itself.

Avoid filler such as "dive into," "unlock," "seamlessly," "powerful,"
"robust," "simply," "obviously," and "just." Do not restate code line by line.
Name the observation, the reason it matters, and the next action.

### Replace generic figure placeholders with teaching briefs

The Part 1 foundations sequence needs three core explanatory visuals:

1. **The calculation loop with ownership labels**
   - structure → neighbors → model → energy/forces;
   - label which step belongs to ASE, Toolkit Core, Toolkit-Ops, and the model;
   - caption answers: "What happens when I call the model?"

2. **Three ragged structures in one batch**
   - concatenated atoms, offset neighbors, `batch_idx`, and per-graph reduction;
   - caption answers: "How can different-sized structures share one call?"

3. **Measured CPU/GPU crossover**
   - latency and throughput against batch size;
   - include hardware and timing boundary in the figure itself;
   - caption answers: "When did the GPU become the faster choice here?"

Every visual needs a source, a concrete caption, an accessibility description,
and a human inspection sign-off. Decorative CPU-versus-GPU core diagrams do
not replace a measured plot.

## Review rubric

A Part 1 candidate is ready for learner testing only if every answer is yes.

### Narrative

- Can a learner state the notebook's question in one sentence?
- Does one scientific object persist from first result to final artifact?
- Is there a useful numerical result within ten minutes?
- Does every section change or inspect that result?

### Toolkit learning

- Are `AtomicData`, `Batch`, model configuration, neighbors, and unpacking
  visible in learner code?
- Does batching answer the scientific question rather than exist as a detached
  benchmark?
- Is each public versus private/tutorial-helper boundary named correctly?
- Are Toolkit conversions, neighbors, composition, relaxation, dynamics, and
  hooks visible rather than wrapped by tutorial helpers?
- Does `aux/` have focused modules and no broad package-level re-export API?

### Learner experience

- Does every target-hardware wait of at least five seconds show a styled card
  before the wait begins?
- Are final widget states preserved in the executed notebook?
- Do early-converged tasks distinguish steps used from their maximum budget?

### Scientific validity

- Are model target, domain, checkpoint, units, charge/spin, PBC, and reference
  convention explicit?
- Is every added energy term compatible with the model target?
- Are accuracy claims supported by a matched, independently sourced reference?
- Are structures, convergence behavior, and invariants inspected?
- For fused dynamics, are stage statuses counted and validated against the
  exact declared schedule before any trajectory-derived claim is shown?

### Performance validity

- Are warm-up, synchronization, transfer, compilation, and timed boundaries
  explicit?
- Do variable-size workloads remain eager unless dynamic-shape compilation was
  validated, and is a fixed compiled workload checked against eager and repeat
  calls before timing or dynamics?
- Are latency and throughput both reported?
- Is a small-batch CPU win allowed and explained?

### Usability

- Does each core cell have one purpose and finish quickly?
- Are outputs adjacent to the code and explanation that produced them?
- Are exercises narrow, meaningful, and runnable without hidden setup?
- Can the learner recover from a restart by running top to bottom?

### Licensing and provenance

- Are code, checkpoint, downloaded parameter tensor, dataset, and figure
  licenses checked separately?
- Are exact versions, hashes, citations, and download behavior recorded?
- Are restricted weights and data excluded from the image and repository?
- Has NVIDIA legal approved any non-permissive required dependency?

## Superseded pre-implementation checklist

These were decisions for the earlier DESS66/Part-3 proposal. They remain useful
as provenance but are not an open checklist for the implemented Part 1
notebook; current validation status belongs in `PART1_DESIGN_HANDOFF.md` and the
run manifest.

1. Acquire and checksum DESS66x8; select a small set of complementary
   interaction curves after visual and runtime review.
2. Pin the exact AIMNet2-2025 ensemble member, metadata, and checkpoint hash.
3. Generate B97-3c dimer and frozen-monomer references on the exact tutorial
   geometries. Record the code, version, full canonical method convention,
   numerical thresholds, inputs, outputs, and hashes. Do not infer a separable
   checkpoint-equivalent target by subtracting only pairwise D3.
4. Cross-check the reference protocol against a second trusted implementation
   on at least one dimer before publishing the table.
5. Reduce the direct Coulomb wrapper to the smallest readable implementation
   that retains charge, force, batching, and boundary-condition checks.
6. Move CO/Cu(111), FIRE2, and MACE to Part 2. Choose explicitly between
   `medium-0b3` and `medium-mpa-0` there; do not use `medium` or the default
   implicitly.
7. Keep Part 2 adsorption framed as prescreening and pair it with a
   reference-validation plan.
8. Obtain written commercial terms before considering MACE-MH-1, MACE-POLAR,
   or any other ASL checkpoint.
