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

The 3,200-atom box is used for the one-GPU PME-versus-Ewald check. The
multi-GPU comparison uses one deterministic 51,200-atom integer supercell of
that box on 1, 2, and 4 H100s. This keeps the density, composition, and
periodic contacts fixed without spending tutorial or H100 time packing a giant
box. The repeated supercell is a controlled compute input, not an independent
liquid configuration, equilibrated mixture, or predicted formulation.
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
a separate direct-force group. The public PME estimator derives the splitting
parameter and grid from the box, a 12 Å real-space cutoff, and the declared
accuracy settings. Component cutoffs, compilation, and halo depth remain fixed
across GPU counts; the same-input force checks decide whether that setup is
accepted.

The pinned Toolkit 0.2 ordinary one-GPU path and multi-GPU `DomainParallel`
path do not reduce energy in the same way. The result set therefore does not
claim total-energy parity across one, two, and four GPUs. It checks
multi-GPU forces against the one-GPU force result, checks distributed energies
against a 2-GPU distributed reference, and keeps the raw one-GPU-to-multi-GPU
energy offset only as a diagnostic.

## Required measurements

- One fixed 51,200-atom input on 1, 2, and 4 H100s. It is a 2 x 2 x 4
  supercell of the checked base box. Geometry, model, dtype, cutoffs, requested
  outputs, and source files must match across all three runs.
- One `DomainParallel` context per GPU count. The structure is partitioned
  once, evaluated once for initialization and warm-up, evaluated three more
  times for measurement, and gathered once.
- Every measured call is `domain.run(local, n_steps=1)`. Because
  `BaseDynamics` has no integration update, this is a fixed-structure
  energy-and-force evaluation, not molecular dynamics. `DomainParallel` may
  wrap coordinates to equivalent periodic images.
- The first multi-rank warm-up also performs Toolkit's automatic force
  initialization. This extra work is recorded and excluded from the measured
  passes. Each measured pass performs exactly one complete AIMNet2 + PME + D3
  evaluation on every GPU count.
- Each timer starts after a rank barrier and CUDA synchronization. It covers one
  public `domain.run(...)` call and the final CUDA synchronization. The saved
  value is the slowest rank time. Partitioning, warm-up, the max-rank time
  reduction, output checks, the final gather, and file writes are outside.
- All three raw pass times and their median. The notebook reports every value;
  it does not select the fastest pass or describe three short samples as a
  general scaling benchmark.
- Every force component from the 2- and 4-GPU runs must agree with the one-GPU
  result. Force acceptance is componentwise:
  `abs(delta) <= atol + rtol * abs(reference)`.
- The one-GPU path returns a `torch.float32` energy. The pinned multi-GPU path
  returns `torch.float64` after Toolkit's distributed reduction. The median of
  the three measured energies represents each GPU layout. On 2 and 4 GPUs, the
  energy range across the three passes must be no larger than
  `1e-4 eV/atom`.
- The 2-GPU median energy is the distributed reference. The 4-GPU median must
  agree with it within `1e-4 eV/atom`. Raw one-GPU-to-multi-GPU offsets and
  the one-GPU pass range are diagnostics because the pinned one- and
  multi-GPU paths reduce energy differently.
- Finite float32 predicted charges from the one-GPU run. The result records
  the input target, predicted sum, residual, absolute residual per atom,
  absolute-charge statistics, dtype, shape, and tensor hash. The residual is a
  diagnostic, not a 51,200-atom pass limit. Toolkit passes the returned charge
  tensor to PME without another renormalization.
- A separate one-GPU fixed-charge PME-versus-Ewald check on the checked
  3,200-atom base box. It uses `estimate_ewald_parameters` at `2e-5`; the charge
  array must be identical in both solvers. The declared limits are
  `|Σq - Qtarget| <= 1e-4 e`, `|ΔE| / atom <= 1e-4 eV`, and
  `max |ΔF| <= 5e-3 eV/Å`.
- Automatically derived spatial cell grids and rank layouts, owned atom
  counts, and peak memory for every rank. `DomainConfig.grid_dims=None` lets
  Toolkit choose both grids from the actual cell shape and cutoff; it does not
  use the rank layout as `grid_dims`. The campaign records the chosen
  `cells_per_dim` and `rank_grid` instead of assuming a cubic layout.
- An exact input-file and input-tensor identity check, plus a
  minimum-image-displacement check through warm-up, all three measured passes,
  and the final gather. The maximum displacement must remain within `1e-4 Å`.
- Exact model, D3, PME, checked base-box, supercell-repeat, source, input,
  hardware, and driver versions, with result checksums. The base-box manifest
  also records the Packmol version and settings used once during preparation.

In the Toolkit 0.2 version used here, each rank contributes its local charges
to the PME mesh and the mesh is combined across ranks. Every rank then runs the
full reciprocal FFT and holds its workspace. Domain decomposition can reduce
atom-local AIMNet2, neighbor, D3, and real-space PME work; it does not divide
every allocation by the number of GPUs. This short run does not search for a
memory limit or deliberately trigger an out-of-memory failure.

Toolkit 0.2 does not expose halo counts or the intermediate multi-rank charges
from this composed model. The large-box run therefore checks source-ordered
forces and distributed energies; only the 3,200-atom PME-versus-Ewald check
uses `|Σq − Qtarget| <= 1e-4 e` as an acceptance rule. The large-box charge
residual is reported without adjustment.

This example is neutral and must not be reused for a charged box. The globally
reduced energy stays on the local result, while `gather` reconstructs atom
fields. The campaign carries stable atom IDs and fixed reference coordinates
as public custom node properties, then restores input order and checks
PBC-equivalent positions after the final gather.

The fixed 51,200-atom input is a 2 x 2 x 4 supercell. With the declared
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
`part1_domain_plan.py prepare` repeats `structure.extxyz` by the declared
integer factors, wraps the expanded coordinates into the primary periodic
cell, and records that operation. It never runs Packmol.

## Recorded bundle layout

```text
recorded/
├── manifest.json
├── raw-results.jsonl
├── distributed.csv
├── electrostatics-validation.json
├── SHA256SUMS
└── job-records/
    ├── gpus-01/
    ├── gpus-02/
    └── gpus-04/
```

`job-records/` keeps the inputs, per-case JSON, force arrays, rank records,
logs, plans, runtime records, and checksum lists produced by the one-, two-,
and four-GPU jobs. The manifest records the exact source revisions and producer
file hashes. Portable result rows replace Compute Lab paths with those checked
identities.

Downloaded AIMNet2 weights and Toolkit-Ops' generated D3 parameter cache are
not redistributed here. The manifest records their exact SHA-256 identities
as external files. Their redistribution terms require a separate release
review.

Until `recorded/` exists, the loader returns the explicit `NOT REPORTED` state.
It rejects incomplete, mixed, or checksum-invalid result sets and does not
estimate absent timings, remove failed rows, accept undeclared files, or follow
paths outside the bundle.
