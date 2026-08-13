# N08 worklog — Domain decomposition

## 2026-08-12 — initial curriculum brief

Status: technically validated draft — human cell review required

- Outcome: prepare an MLIP for Toolkit domain decomposition and profile one
  calculation across multiple GPUs.
- Core APIs: the current public `DistributedManager`, `DomainConfig`,
  `DomainParallel`, `partition`, `run`, and `gather` path.
- Validation: prove one-GPU correctness first, then compare two or more GPUs
  where hardware permits.
- Example source: use the official bring-your-own-model route at the frozen
  Toolkit pin.
- Continuity: reuse Parts 03 and 05; Part 06 supplies recommended performance
  literacy.

## 2026-08-13 — approved deep-dive design

### Learner and scope

- Audience: a scientist who can already build a `Batch`, configure a model,
  and run one `BaseDynamics` step, but has not distributed one periodic system.
- Learner outcome: explain ownership and halos, execute the supported
  world-size-one control path, and judge whether an externally recorded
  1/2/4-GPU campaign is both comparable and trustworthy.
- Teaching boundary: domain decomposition shards one large system spatially.
  It is not data parallelism and it is not distributed pipeline parallelism.
- Live path: a 32-atom periodic argon crystal with the built-in
  `LennardJonesModelWrapper`. The one-process fallback is deliberately small,
  deterministic, checkpoint-free, and CPU-runnable.
- Scale-out path: the checked 3,200-atom periodic phenol/NMA box from the
  existing Part 1 data bundle, deterministically repeated to 51,200 atoms and
  evaluated with the current pinned AIMNet2 adapter. The notebook does not
  launch the distributed job.

### Evidence decision

- The available H100 records use older Toolkit/Toolkit-Ops commits. They are
  retained only as methodology and are not plotted or relabelled as current
  results.
- No current-pin 1/2/4-GPU campaign is available in this environment. The
  learner-facing result state therefore begins as `NOT REPORTED`.
- A Part-08-owned external runner and campaign manifest will require exact
  Toolkit/Toolkit-Ops commits, input and checkpoint SHA-256 values, rank-local
  owned-atom counts, gathered source-atom IDs, force/energy parity, timings,
  and hardware/runtime provenance before any scaling plot is allowed.
- Timing is descriptive only: warm-up is excluded, ranks are synchronized,
  and the median of repeated model evaluations is recorded. A single campaign
  cannot establish general scaling.

### Notebook arc

1. Motivate decomposition with a spatial ownership/halo diagram.
2. Build and inspect one periodic argon `Batch`.
3. Execute `DistributedManager → mesh → DomainConfig → DomainParallel →
   partition → run → gather`.
4. Prove the world-size-one invariant: all atoms remain owned by rank 0,
   `gather` returns the full batch on rank 0, and no scaling claim follows.
5. Explain model distribution specs, output reductions, migration, and hook
   scopes without exposing private implementation.
6. Inspect the exact phenol/NMA campaign specification and the current
   `NOT REPORTED` evidence gate.
7. Load and validate cached records only when all current-pin, checksum,
   ownership, parity, and completeness checks pass; otherwise draw no plot.
8. Connect to Core, GPU pipelines, dynamics, and R&D; link UMA as a separate
   environment rather than executing it in the lock.

### Test-first acceptance matrix

- Notebook schema validates; code parses; learner cells remain short and
  progressive; stable cell IDs are unique.
- The visible notebook contains every public API in the required sequence,
  initializes and cleans up the manager safely, and contains no SSH,
  `torchrun`, subprocess, shell, prompt, or private-attribute access.
- World-size-one behavior is exercised on CPU with the pinned runtime.
- Input building, SHA-256 checks, source-atom reordering, evidence validation,
  ownership accounting, parity metrics, and plot gating have unit tests.
- The external runner is syntax checked and statically required to serialize
  campaign cases, synchronize ranks, record provenance, and emit per-case
  checksums.
- Final checks include Part-08 tests, repository notebook lint/design/schema
  checks, fresh-kernel execution, HTML render inspection, and two prose/design
  revision passes.

## 2026-08-13 — implementation and final review

### Revision passes

1. Scientific and technical review:
   - aligned the live Lennard-Jones cutoff and skin at 4.5 Å and 0.25 Å,
     keeping the cutoff below half the 10.52 Å cell length;
   - clarified that the ownership sketch isolates one boundary while periodic
     faces exchange with wrapped neighbors;
   - checked world-size-one gather and exact energy/force parity against the
     direct `BaseDynamics` path; and
   - kept old H100 records out of plots because their Toolkit pins are stale.
2. Pacing and rendered-design review:
   - retained 57 progressive cells with 18 lines or fewer per code cell;
   - reshaped the base-box identity into a two-column inspection table;
   - updated the API callout to the current shared template; and
   - added renderer-visible `aria-label` and SVG `title` text to both figures.

### Cell-review index

- 00–07 (`c4d8a176`–`b74bfd4e`): banner, goal, course map, setup, and installed
  pin check.
- 08–12 (`4cad07e7`–`11fd0023`): ownership, ghosts, halo width, and one-step
  mental model.
- 13–22 (`664cb2d2`–`f11ab8e9`): periodic argon control, model distribution
  contract, and direct reference.
- 23–35 (`d8141eb2`–`bbb07278`): visible public API path, cleanup, gather
  semantics, assertions, and parity figure.
- 36–38 (`5abf243d`–`11a74cff`): multi-rank changes, hook scopes, and separation
  from distributed pipeline parallelism.
- 39–50 (`20d3688d`–`916ea024`): exact phenol/NMA input, evidence gate,
  external runner, parity policy, and scaling limits.
- 51–56 (`4b3c6c19`–`9647a8bd`): bounded halo exercise, UMA route, recap, and
  human-review gate.

### Evidence provenance

- Runtime pins: Toolkit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`;
  Toolkit-Ops `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
- Base box SHA-256:
  `5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da`;
  base manifest SHA-256:
  `ea30e3f12f042f98f136147e783b56ab2e0da622f3486718b9fec69f3cde74b4`.
- Generated 51,200-atom tensor SHA-256:
  `56b9d1c71c9c392a2e12ad8149f3ca0cb0ab816fd4926af42fd264e8874d9a36`;
  AIMNet2 checkpoint SHA-256:
  `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`.
- Current-pin 1/2/4-GPU evidence remains `NOT REPORTED`. No scaling or
  multi-GPU parity values are claimed.

### Verification record

- `./scripts/v3-run pytest -q notebooks/08-domain-decomposition/tests`:
  28 passed; warnings were limited to upstream Torch deprecations and expected
  no-CUDA/NVML world-size-one notices.
- `./scripts/v3-run ruff check notebooks/08-domain-decomposition`: passed.
- Notebook schema, stable IDs, code syntax, and all 57 execution counts:
  validated.
- Fresh-kernel execution: passed with the checked `NOT REPORTED` output and no
  traceback or Warp CUDA stream noise.
- HTML render: passed without missing-alt warnings; both figures were inspected
  at teaching width. The render artifact is isolated under
  `/tmp/603ce216-4b94-47a4-b105-a33b00147be4/`.
- Human review remains required for cell-by-cell scientific copy and for any
  future external 1/2/4-GPU campaign.

### Coordinator note

The root `.venv` is outside Part 08 ownership. It was left in place and requires
later coordinator review; no cleanup or further root-environment use was
attempted.
