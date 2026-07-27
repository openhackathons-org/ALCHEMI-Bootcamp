# Domain-decomposition result set

This directory is reserved for the recorded H100 results used by the final
Part 1 scaling lesson. Until a complete result set passes the checks below, the
notebook displays `NOT REPORTED`.

The workload starts from one checked, three-dimensionally periodic box
containing 128 phenol and 128 N-methylacetamide molecules (3,200 atoms). Its
input total-charge target is zero.
The molecular geometries come from NCI Atlas system `1.041`, which is evaluated
earlier in the notebook. Packmol placed the two kinds of molecule independently
when this base box was prepared. The count is a 1:1 composition count, not a
count of bound dimers. The checked files live in `prebuilt_base_box/`.

Every larger input is a deterministic integer supercell of that same base box.
This keeps the density, composition, and periodic contacts fixed while the atom
count changes. It also avoids spending tutorial or H100 time packing giant
boxes. These repeated inputs are controlled scaling workloads, not independent
liquid configurations, equilibrated mixtures, or predicted formulations.
The base Packmol input used a 2.0 Å distance tolerance and 0.001 Å optimizer
precision; the saved bundle records the periodic contact check.

The composed periodic model is:

```text
AIMNet2 checkpoint base -> predicted charges -> PME
finite-cutoff, tapered D3(BJ) ----------------> total energy and forces
```

Here the checkpoint base is `E_NN - E_Coulomb^SR`. Full PME restores the
declared Coulomb contribution without counting the short-range term twice.

The AIMNet2 component is NCI ensemble member 0,
`aimnet2-wb97m-d3_0`, with its checkpoint-declared D3(BJ) damping parameters.
The tutorial uses a 15 Å cutoff and tapers the final 20%, from 12 to 15 Å.
Reference D3 normally uses a much longer, untapered range, so this finite-cutoff
term should not be described as identical to reference D3. This is the same
model used for the finite NCI interaction curves, not the B97-3c-2025
checkpoint used in the opening water example.
The model architecture accepts periodic cells, but this checkpoint has not
been validated for a dense, heterogeneous condensed-phase mixture. This stage
demonstrates model composition and multi-GPU execution; it does not report
condensed-phase material properties.

The AIMNet2-to-PME path uses one dependent `PipelineGroup` with autograd so the
forces include the response of predicted charges to atomic positions. D3 uses
a separate direct-force group. The run calls the public PME estimator with a
12 Å real-space cutoff, a `1e-4` estimator target, and a `1.0` mesh safety
factor. It passes the returned splitting parameter and grid dimensions
explicitly to the PME wrapper; mesh spacing is therefore derived for each box,
not fixed in advance. Separate neighbor lists serve the different component
cutoffs, and `compile=False` remains fixed across GPU counts. The 19 Å D3 halo
combines the 15 Å cutoff with a 4 Å coordination-number margin used by Toolkit
0.2 multi-GPU tests. That 4 Å is extra communication depth, not another
interaction cutoff or a universal guarantee. This result set accepts it only
after the same-input force checks pass on two and four GPUs.

The pinned Toolkit 0.2 ordinary one-GPU path and multi-GPU `DomainParallel`
path do not reduce energy in the same way. The result set therefore does not
claim total-energy parity across one, two, and four GPUs. It checks
multi-GPU forces against the one-GPU force result, checks distributed energies
against a 2-GPU distributed reference, and keeps the raw one-GPU-to-multi-GPU
energy offset only as a diagnostic.

## Required measurements

- A predeclared, cold one-GPU atom-count ladder. Every size runs in a fresh
  process with no warmup. Every attempted size is retained, and the ladder
  stops only after the first genuine CUDA out-of-memory failure.
- The same geometry and settings for each one-GPU and multi-GPU comparison.
- A fixed 51,200-atom numerical check that fits on one GPU. Every force
  component from the 2- and 4-GPU runs must agree with the one-GPU
  result. Force acceptance is componentwise:
  `abs(delta) <= atol + rtol * abs(reference)`.
- The 2-GPU energy is the distributed reference for that fixed case. The
  4-GPU energy must agree with it within `1e-4 eV/atom`. The raw
  one-GPU-to-multi-GPU energy offsets are recorded as diagnostics and do not
  determine acceptance.
- Available finite float32 predicted charges on every successful one-GPU
  capacity case. Each result records the input target, predicted sum, residual,
  absolute residual per atom, absolute-charge statistics, dtype, shape, and
  tensor hash. The residual is a diagnostic, not a large-system pass limit.
  Toolkit passes the returned charge tensor to PME without another
  renormalization.
- A separate one-GPU fixed-charge PME-versus-Ewald check on the checked
  3,200-atom base box. It uses `estimate_ewald_parameters` at `2e-5`; the charge
  array must be identical in both solvers. The pass limits are declared before
  the H100 results are inspected: `|Σq − Qtarget| <= 1e-4 e`,
  `|ΔE| / atom <= 1e-4 eV`, and `max |ΔF| <= 5e-3 eV/Å`.
- The first planned size that naturally fails on one GPU, retried unchanged on
  two and four GPUs. These are cold out-of-memory retries: one
  public `partition` → `run` → `gather` workflow, no warmup, and a fresh
  process for each GPU count. At least one unchanged retry must succeed before
  the result set can be installed.
- A separate same-input timing series on one, two, and four GPUs. It
  uses the largest input that succeeded in the one-GPU capacity ladder. Each
  GPU count runs one warmup workflow and five measured workflows. Every
  workflow creates and enters a fresh `DomainParallel` wrapper because the
  public `partition` method is called once per wrapper.
- Every force component from each 2- and 4-GPU timing row must agree with
  the one-GPU timing row. The 4-GPU timing energy must agree with the
  2-GPU timing energy within `1e-4 eV/atom`. One-GPU-to-multi-GPU energy
  offsets remain diagnostic only.
- Toolkit 0.2 performs an automatic initial force evaluation before the first
  multi-rank step; its one-rank pass-through does not. To compare equal work,
  each timed one-rank workflow requests two `BaseDynamics` steps, while each
  multi-rank workflow requests one step after the automatic evaluation. Both
  paths therefore perform two model evaluations. `BaseDynamics` does not move
  the atoms in this single-point workload.
- The timer begins after `DomainParallel` construction and context entry, a
  rank barrier, and CUDA synchronization. It covers one public `partition` →
  `run` → `gather` workflow and the final CUDA synchronization. The recorded
  sample is the slowest rank's elapsed time. Loading the checked base box,
  constructing its supercell, model loading, `Batch` construction,
  host-to-device transfer, output checks, wrapper cleanup, and file writes
  remain outside the timer.
- All five measured sample times, with their median, first quartile, third
  quartile, and interquartile range. Every GPU count must have
  `IQR / median <= 0.10`; otherwise the timing series is rejected. Only the
  complete one-, two-, and four-GPU series is used for speedup and
  parallel efficiency. Every GPU count is reported; no fastest-looking point
  is selected. The cold capacity, fixed numerical check, and out-of-memory retry
  checks are not used for scaling claims.
- When two or more unchanged out-of-memory retries succeed, their energy and
  forces must also agree with each other. With only one successful retry,
  that cross-check is not applicable.
- Automatically derived spatial cell grids and rank layouts, owned atom
  counts, and peak memory for every rank. `DomainConfig.grid_dims=None` lets
  Toolkit choose both grids from the actual cell shape and cutoff; it does not
  use the rank layout as `grid_dims`. The campaign records the chosen
  `cells_per_dim` and `rank_grid` instead of assuming a cubic layout.
- Exact model, D3, PME, checked base-box, supercell-repeat, source, input,
  hardware, and driver versions, with result checksums. The base-box manifest
  also records the Packmol version and settings used once during preparation.

In the Toolkit 0.2 version used here, each rank contributes its local charges
to the PME mesh and the mesh is combined across ranks. Every rank then runs the
full reciprocal FFT and holds its workspace. Domain decomposition can reduce
atom-local AIMNet2, neighbor, D3, and real-space PME work; it does not divide
every allocation by the number of GPUs. A reciprocal-PME allocation failure
must not be described as rescued by spatial decomposition. For this reason,
an unchanged case that runs out of memory on one GPU may still run out of
memory on some or all multi-GPU runs. If none of the unchanged retries
succeeds, the result set remains incomplete rather than substituting a smaller
or differently configured input.

The public `DomainParallel` result exposes owned atoms through its returned
local `Batch`. It does not expose exact halo counts, and this distributed
composite returns energy and forces rather than the intermediate predicted
charges. Those fields remain unavailable unless a later public API adds them.
The Stage 7 input has a total-charge target of zero. AIMNet2 returns float32
atomic charges, and its internal charge correction also reduces in float32.
Re-summing the returned charges in float64 can expose a small residual for a
large system. The campaign records that residual and passes the returned
charges to PME without another adjustment. Toolkit 0.2 does not carry the input
system charge into each GPU region, and AIMNet2 defaults a missing charge to
zero; this exact
example must not be reused for a charged box. `gather`
reconstructs pre-existing atom fields but not per-system values such as charge,
stress, virial, dipole, graph embeddings, `info`, or custom metadata. Read the
globally reduced energy from the local result; use `gather` for atom fields.
Gathered atom rows are rank-contiguous rather than source-ordered. The campaign
keeps `source_atom_id` outside the Toolkit `Batch`, uses
`SpatialPartitioner.assign_atoms_to_ranks` to reproduce Toolkit's stable
rank-contiguous scatter order, and restores source order before comparing
forces. Multi-GPU runs request one step, so no deferred atom migration changes
that initial ownership before `gather`.

The distributed result does not expose the rank-local predicted charges for a
second comparison. Rank consistency is therefore checked with source-ordered
forces and the globally reduced distributed energies. The separate 3,200-atom
PME-versus-Ewald calculation is the only place where
`|Σq − Qtarget| <= 1e-4 e` is an acceptance rule.

The fixed 51,200-atom check is a 2 × 2 × 4 supercell. With the declared
cutoff, Toolkit is expected to choose 1 × 1 × 2 ranks on two GPUs and
1 × 1 × 4 on four GPUs. These are expectations
for planning, not hard-coded acceptance values. Each recorded run must report
its actual layout, and `require_nondegenerate=True` rejects it if any rank's
halo covers the full structure.

## Checked base box

```text
prebuilt_base_box/
├── manifest.json
├── structure.extxyz
├── preview.png
└── SHA256SUMS
```

The loader checks all four files before the notebook or campaign uses the box.
`part1_domain_plan.py prepare` then repeats `structure.extxyz` by the declared
integer factors. It never runs Packmol.

## Recorded bundle layout

```text
recorded/
├── manifest.json
├── campaign-summary.json
├── raw-results.jsonl
├── capacity.csv
├── parity.csv
├── distributed.csv
├── SHA256SUMS
├── structures/
├── logs/
├── plans/
├── producers/
├── source-inputs/
└── job-records/
```

`job-records/` keeps the inputs, per-case JSON, force arrays, rank records,
logs, and checksum lists produced by the one-, two-, and four-GPU
jobs. `plans/` contains portable copies of the capacity, selection, and
distributed plans. The result-producing scripts and the packaged NCI input
are included so that the saved tables can be audited without access to the
original Compute Lab directories.

Downloaded AIMNet2 weights and Toolkit-Ops' generated D3 parameter cache are
not redistributed here. The manifest records their exact SHA-256 identities
as external files. Their redistribution terms require a separate release
review.

Until `recorded/` exists, the loader returns the explicit `NOT REPORTED` state.
It rejects incomplete, mixed, or checksum-invalid result sets and does not
estimate absent timings, remove failed rows, accept undeclared files, or follow
paths outside the bundle.
