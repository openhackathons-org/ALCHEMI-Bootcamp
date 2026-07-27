# Runtime Snapshot

> **Historical validation record.** This file combines older environment notes
> with run history and is not the current environment specification. Use
> `build/Dockerfile` and `build/requirements.txt` for the current Part 1 build.
> Exact run details remain here temporarily until they are organized beside
> their saved outputs.

The opening sections point to the current build files and record the exact
versions used by older saved runs. Each run result below applies only to its
recorded source and environment.

## Declared image base

- Base image: `nvidia/cuda:13.2.0-runtime-ubuntu24.04`
- Conda environment: `/opt/conda/envs/alchemi-playbook`
- Python: `3.12.13`
- Torch wheel source: `https://download.pytorch.org/whl/cu130`
- Jupyter kernel configured by the Dockerfile: `alchemi-main` (`ALCHEMI Main`)
- Configured Jupyter server default kernel: `alchemi-main`

The Dockerfile removes the default `python3`, `ovito-pro`, and old
`alchemi-barebones` kernelspecs so notebooks bind to one playbook kernel. It
also configures Jupyter to disable its synthetic native kernel and use
`alchemi-main` by default.

## Pinned Git snapshots

- `nvalchemi-toolkit`: `331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409` (`0.2.0`)
- `nvalchemi-toolkit-ops`: `e8e7a7464f6745277a156a3d6f433d06b58c60e3` (`v0.4.0`)

`build/requirements.txt` is the source of truth for these pins. The Dockerfile
installs with `uv pip install --no-sources-package nvalchemi-toolkit-ops` so
Toolkit's internal `tool.uv.sources` revision cannot replace the directly
tested Toolkit-Ops commit. The current Dockerfile is configured to exclude
`orb-models` and `loguru`, then run `uv pip check`; a clean image build is
still pending. The old shared environment used
`build/overrides.txt` to force Orb 0.7.0 past its declared Toolkit-Ops `<0.4`
limit. That file is retained only as a historical record and is not read by
the current Docker or Slurm setup.

Part 1 normally lets Core `compute_neighbors` and model-generated hooks follow
each wrapper's declared neighbor configuration. Toolkit has multiple search
implementations; Core selects a compatible one from the model requirements.
For fixed coordinates, the tutorial uses one-shot `compute_neighbors` calls
with the model's cutoff, output format, and full- or half-list setting. When
coordinates move, model-generated hooks also use the configured skin to decide
when to rebuild the list.

## Historical shared-environment package versions

The list below describes earlier saved runs. It is not the current Part 1
package specification. In particular, `orb-models` belonged to the old shared
environment and is intentionally absent from the current Part 1 image. The
retained Orb notebook needs its separate historical environment.

- `torch`: `2.12.0+cu130`
- `nvalchemi-toolkit`: `0.2.0` from the pinned Git commit above
- `nvalchemi-toolkit-ops`: `0.4.0` from the pinned Git commit above
- `aimnet`: `0.2.0`
- `nvidia-physicsnemo`: `2.1.1`
- `warp-lang`: `1.13.0`
- `orb-models`: `0.7.0`
- `sevenn`: `0.13.0`
- `ovito`: `3.15.4`
- `mace-torch`: `0.3.15`
- `cuequivariance`: `0.10.0`
- `cuequivariance-torch`: `0.10.0`
- `cuequivariance-ops-cu13`: `0.10.0`
- `cuequivariance-ops-torch-cu13`: `0.10.0`
- `ase`: `3.29.0`
- `pymatgen`: `2026.5.4`
- `pydantic`: `2.13.4`
- `jupyterlab`: `4.6.1`
- `ipykernel`: `7.3.0`

## Build entrypoints

- Conda package spec: `build/environment.yml`
- Python/runtime package spec: `build/requirements.txt`
- Docker recipe: `build/Dockerfile`
- Compose service: `build/docker-compose.yml`

## Part 1 runtime checks

### Current-source Stage 7 domain setup

The Stage 7 periodic box is an unequilibrated scaling input, not a liquid or
formulation result. PME setup uses the public `estimate_pme_parameters`
function with a fixed 12 Å real-space cutoff, accuracy `1e-4`, and mesh safety
factor `1.0`. Its estimator-derived `alpha` and power-of-two mesh dimensions
are passed explicitly to `PMEModelWrapper`. A fixed-charge check on the
3,200-atom validation box builds its direct Ewald reference independently with
`estimate_ewald_parameters` at accuracy `2e-5`.

No complete checked H100 Stage 7 result set is installed. Capacity, timing,
speedup, and efficiency remain **NOT REPORTED**.

On 2026-07-24, the 3,200-atom input-preparation path was run locally with
Packmol 21.2.1. Two independent runs with the declared seed produced the same
structure SHA-256,
`5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da`.
An independent read of the saved structure confirmed 128 phenol molecules,
128 N-methylacetamide molecules, full three-dimensional periodic boundaries,
a 32.877998637 Å cubic cell, a construction density of exactly
1.000000000000 g/cm³, zero net charge, and a shortest periodic
intermolecular distance of 2.197807386 Å. The required lower bound was
1.999 Å.

Using that structure and Toolkit-Ops commit `e8e7a74`,
`estimate_pme_parameters` resolved `alpha = 0.252904534340 Å⁻¹`, a
`64 × 64 × 64` mesh, and 0.513718724251 Å spacing for the declared 12 Å
cutoff and `1e-4` target. The independent `estimate_ewald_parameters` call at
`2e-5` resolved a 15.894538879395 Å real-space cutoff, below the
16.438999318447 Å half-box length. These checks validate input construction
and parameter resolution only. They do not replace the H100 model,
electrostatics, force, or multi-GPU checks.

### Focused NCI Stage 3 run with the current Toolkit versions

Compute Lab job `3222436` ran the six Stage 3 cells on one NVIDIA H100 NVL with
Toolkit Core `331d6b2`, Toolkit-Ops `e8e7a74`, Torch `2.12.0+cu130`, and CUDA
`13.0`. It completed with exit `0:0` in `3:15`. The reportable code time is
`22.643220 s`, the CUDA-synchronized sum of the six cells, not scheduler time.

The measurement includes NCI data loading, four cold AIMNet model loads and
first calls, one shared D3 pass, composed-model checks, an independent force
check, reference analysis, table construction, and plotting. Checkpoint files
were already resolved and copied to node-local storage. Kernel startup, earlier
notebook cells, downloads, environment setup, and result packaging are
excluded. The run evaluated 90 graphs containing 1,140 atoms.

The complete model stayed below `0.368655 kcal/mol` MAE against DFT-D3 and
`0.351350 kcal/mol` against CCSD(T)/CBS for each of the three selected curves.
Its Toolkit analytic force differed from the official AIMNet2 analytic force by
`2.980e-7 eV/Å`. The official analytic force differed from a 0.003 Å central
finite difference of the official complete total energy by `2.545e-4 eV/Å`.
Charge conservation, component sum, graph-order, net-force, force-route, and
complete-model MAE checks passed.

The timed source notebook SHA-256 is
`9cbb4c88a1483a83d04c82d9fb144fe992f7a0cc4f96ceed3ca51210d976fb11`.
The local saved bundle is
`part-1-scalable-atomistic-workflows/outputs/h100-nci-stage3-3222436/`; all
files pass its `SHA256SUMS` check. This focused result does not replace the
required end-to-end run of the complete seven-stage notebook with the current
Toolkit versions.

The current tutorial uses the AIMNet checkpoint calculator's finite all-pairs
`simple` Coulomb path and a separate custom adapter for raw SevenNet-Omni 0.13.
The adapter selects the `mpa` task and evaluates energy and forces for a
fixed-geometry Cu(111) panel. Toolkit supplies a separate PBE-D3(BJ) term. The
surface batch is periodic in x and y only; isolated-molecule references are
nonperiodic. The official 103,162,838-byte checkpoint is pinned at SHA-256
`ca81bd3aac9fc2696c93dd386615f5a0fe41b92ab9ed7f69fa9526baaa9bab64`;
the SevenNet software is MIT and the checkpoint record is CC BY 4.0 (Figshare
DOI 10.6084/m9.figshare.30399814). Full-notebook job `3189534` below used an
older source and Core pin. CL job `3175650` validates the immediately preceding
OrbMol-v2 source only. Older DSF checks and MACE-era jobs below are retained as
historical development records.

### Historical accepted SevenNet-source Part 1 H100 run

CL job `3189534` ran the then-current 47-cell notebook on one NVIDIA H100 NVL. It
finished `COMPLETED` with exit `0:0` in `15:37`; all 47 code cells ran with no
error outputs. The external validator and portable checksum verification
passed. The dynamics cell took `598.6 s` (9.98 min) for exactly 5,000 NVT steps
followed by 20,000 NVE steps. The run used the installed pinned Toolkit Core
`80aab5c` and Toolkit-Ops `e8e7a74` directly. It did not replace runtime
attributes or patch source files on the compute node.

The production record contains 20,000 dipole and position frames over 10 ps.
Differencing the dipole gives 19,999 current samples. The saved spectrum uses
exactly two complete 5 ps Hann windows with 50% overlap and a `6.6713 cm⁻¹`
frequency grid. This short record supports a qualitative teaching spectrum;
it is not a trajectory-length convergence result.

- Source notebook SHA-256:
  `33cd752d0508794e566c9a0108a44bf1aeb2ce9ea1db6779738e086fa665ac56`.
- Executed notebook SHA-256:
  `ce6622ad95d712251ed5dee4b76db59c98f837aa55ebbf46ea868d53e62b04b0`.
- Calculation validator-report SHA-256:
  `a1d7d7b4ed06a0506a224743e183e39aa0b4bea0b8a11fe4c56f0a27da9d870b`.
- Reviewed-notebook validator-report SHA-256:
  `3c7522600e396b375373b94b41c9670036f55e992a16d6e44fd88a3805bbcbc0`.
- Reviewed notebook SHA-256:
  `894fbd2d6758c80423819dbff7be453d567280cd12c26ca42418c6be402ba94f`.
- Standalone reviewed HTML SHA-256:
  `5133e753eb7474256d21e8887c3fa59505e267b5dc73a252973accedd5b9bab1`.
- Trajectory SHA-256:
  `5234a7ae1eb151e94646d4a20f765d2790d0efa47b530d334cb501ae66d4dc54`.
- Run-manifest SHA-256:
  `708685811e7f685fce80052f3bfb4985c8e4f47098455d4d147964be534ee948`.
- Portable checksum-index SHA-256:
  `288704fc0a58d5b4ed8f7b3a71eebb0aaa5ad095db16059a970263063a77e`.
- Reviewed checksum-index SHA-256:
  `7bdf300da0b3f739867444218e51d79d79299f962868f258a57cf9f1f00d3cca`.
- Reviewed-validation checksum-index SHA-256:
  `d459209ea4761931116a26da23304c10ea72d1693e1ac65d82b831c3fae886d7`.
- The run manifest inventories 60 calculation outputs totaling 15,554,260
  bytes.

The complete Toolkit model's eight-point water-dimer MAE against full B97-3c
was `0.632837 kJ/mol`. The harmonic calculation passed all eight numerical
checks and gave a six-mode frequency MAE of `22.2530 cm⁻¹` against B97-3c.
The SevenNet graph mapping passed. The largest direct differences were
`8.4771e-7 eV/atom` in energy and `9.3937e-5 eV/Å` in a force component.

Every saved cluster frame remained oxygen-connected. The H₆ and D₆
initial-ring fractions were `0.99685` and `0.96810`, and the maximum energy
excursion was `0.379774 meV/atom`. H₂O/D₂O missed the 20% temperature-pair
check with a relative difference of `0.227301`. H₆/D₆ passed that check at
`0.136539` but did not retain the initial ring in every frame. All four
MD-derived quantitative comparisons remain **NOT REPORTED**.

The accepted bundle is
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3189534/`. It
contains the executed notebook, reviewed notebook, standalone HTML, validator
report, checksum indexes, and saved calculation outputs. Automated checks do
not replace a human review of the rendered notebook, which remains pending.
Compute Lab job `3190092` independently validated the Markdown-only reviewed
copy in the same pinned H100 environment and completed with exit `0:0`.

### Historical accepted OrbMol-source Part 1 H100 run

CL job `3175650` ran the previous complete 45-cell notebook on one NVIDIA H100 NVL. It
finished `COMPLETED` with exit `0:0` in `24:52`; the exact 5,000-step NVT +
50,000-step NVE cell took `1208.5 s`. The 2,048-system inflight cell took
`15.2 s`; this one execution is not a benchmark. All 45 code cells executed
with no error outputs. The external validator and portable checksum
verification passed. The complete Toolkit model's eight-point water-dimer MAE
against full B97-3c was `0.633 kJ/mol`.

- Source notebook SHA-256:
  `f32b9f253f41c6b8f1d614bf4bc03aa188328be26a136afa8c5601c7e0a8cfad`.
- Executed notebook SHA-256:
  `d6fd94aded92d1e19f15f4fb6294551a6fed9322b52faac5b4738eece7aa1e58`.
- Calculation validator-report SHA-256:
  `3a0e3d50c183532664625d86b6511199c006fef8faa978b9b64a89fa6feef3c7`.
- Reviewed notebook SHA-256:
  `2830333cfa00dc566b2e206b18214133f5ac80eefc0083580359d7303cb09cf7`.
- Reviewed validator-report SHA-256:
  `41849f9a7a23fa04605e6642f477a9955e52fd9ba31feddb21e5fd3b1cfd01b9`.
- Reviewed portable-checksum-index SHA-256:
  `698e9b6aafe9f058de9ff7f20ae8167243082b5f30aeb9b35faeac3fce880de5`.
- Standalone reviewed HTML SHA-256:
  `cd024829d3f675fb5b6a8d1aad38fe83b6c142016f1eba8e8cdc96945f841e56`.
- Trajectory SHA-256:
  `43d8b06b762d3b442c604fc3202b1a08f4943228d8fe64a96a715022d069ae51`.
- Run-manifest SHA-256:
  `d83534f7f70839bc534d3ea4a6d7a85054300f5978bac2b53191db7d5d9dc53e`.
- The manifest inventories 64 output artifacts totaling 36,830,973 bytes.

The OrbMol-v2 table contains 16 rows across three graphs: 14 required exact
mapping checks passed, while two rows are informational. The largest
Toolkit-versus-native differences were `0.0001830657323201 eV/atom` in energy
and `0.0005995631217956 eV/Å` in a force component. The same adapter ran FIRE2
on two graphs for 99 steps and finished at `0.0495131158989658 eV/Å` maximum
force.

The harmonic calculation passed all eight numerical checks, reached
`1.1697242689681489e-5 eV/Å`, and gave a six-mode frequency MAE of
`22.245965851892645 cm⁻¹` against B97-3c. Every cluster frame remained
oxygen-connected. Initial-ring fractions were `0.61552` for H₆ and `0.36392`
for D₆. Maximum energy excursion was `0.3255208333 meV/atom`. All four
MD-derived quantitative comparisons remain **NOT REPORTED** because the
applicable temperature or topology checks did not all pass.

CL test job `3176022` separately passed 451 tests and two subtests. Pytest time
was `62.90 s`; scheduler wall time was `1:10`; the job completed with exit
`0:0`, with no skips or failures. Its 16 warnings were upstream deprecation
warnings. It supersedes the 449-test pre-release-copy run `3175651`.

The accepted bundle for that saved source is
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3175650/`.
It includes the executed notebook, a Markdown-reviewed notebook and HTML,
validator reports, checksum indexes, and the saved run. Automated review and
validation found all six images embedded, all 45 progress cards marked
`COMPLETE`, all six top-level stages, two readable tables with no ellipses or
`NaN` values, and confirmed that both relative links resolve. Browser
automation was unavailable, so human review of the rendered notebook remains
pending.

Job `3174963` is the immediately preceding display-audit run. It completed all
45 cells and passed its scientific checks, but its rendered tables contained
truncated rows. Its bundle remains at
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3174963/`; it is
historical rather than the release display record.

Development job `3173975` completed that 45-cell scientific workload but failed
the old validator because topology values were compared at different
floating-point precision. Job `3174716` stopped during setup while Orb's unused
stress path was being disabled for the finite energy/force adapter. Job
`3174859` stopped during wrapper construction because generic `cuda` and
explicit `cuda:0` were treated as different devices. Regression tests cover the
two implementation fixes. These three jobs are diagnostic history, not accepted
results.

### Historical pre-OrbMol Part 1 H100 run

CL job `3149917` ran the complete 43-cell notebook on an NVIDIA H100 NVL. It
finished `COMPLETED` with exit `0:0` in `24:47`; the full 5,000-step NVT +
50,000-step NVE cell took `1195.6 s`. All 43 code cells executed with no error
outputs. The external validator passed, independently reproduced all seven
derived IR and topology tables with zero numerical difference, and verified
the portable bundle checksums.

- Executed source notebook SHA-256:
  `e603c629208d7ba325e4de203a0efc73a2b601513ffdcf4a3cc0327c6cfe7a13`.
- Executed notebook SHA-256:
  `0bae4aa14e31225194d2080f4ad720ff044f0bf29b880e3b42bdabf48f159e1a`.
- Trajectory SHA-256:
  `41ac428b3a6a025be44cb681628769d685f3f069bd969623fc3b4840fe6ec0df`.
- Run-manifest SHA-256:
  `204afb9219d02942162d04e20724975a52a0f16575245f4d283f50ae85f153a7`.
- Validator-report SHA-256:
  `54dda51528684d47eca1792ae321701e3cd68104b3bd7f06a44d67f2116d6e6b`.
- The manifest inventory contains 57 output files totaling 36,971,662 bytes;
  the trajectory contains 50,000 frames for four systems and 42 atoms.

The historical Markdown-only presentation revision had SHA-256
`9f9a6da1e98b2c0464f85f43a1298e6674a72ba76631b76eb795f2b980783f47`.
It has the same 43 executable cells, byte for byte, plus six Markdown-only
clarifications about harmonic intensities, the three distributed-pipeline
blockers, and the MACE constructor example. The local review notebook combines those
Markdown cells with the validated outputs and records job `3149917` in its
metadata. This is a presentation refresh, not a claim that the whole historical
notebook file was the exact file sent to the H100.
The reviewed-current notebook SHA-256 is
`ec65ae1d63857e4de715920226d679c217c0655a68bdd658f0e078572c440a76`.
This remains the reviewed bundle for that saved pre-OrbMol source only.

The final harmonic AIMNet + Coulomb + D3 comparison passed all eight numerical
checks and reported a six-mode frequency MAE of `22.253 cm⁻¹` against B97-3c.
The MD trajectory stayed covalently intact and oxygen-connected, and maximum
energy excursion stayed below `0.326 meV/atom`. All four MD comparison rows
remain not reported: H₂O/D₂O missed the 20% temperature-pair tolerance, and
the initial cyclic ring did not persist in every H₆ or D₆ frame. Distributed
timing also remains **NOT REPORTED** for the three reasons documented below.

Job `3149881` was the immediately preceding staging failure. It exited `1:0`
after `02:26` because its incomplete source archive omitted the repository
root `LICENSE`, which the checked B97-3c reference loader requires. It stopped
before dynamics; this was a packaging error, not a model or scientific result.

### Historical independent staged-source H100 follow-up

CL job `3150048` independently ran the exact staged source that added the
stricter distributed-transfer preflight. It finished `COMPLETED` with exit
`0:0` in `24:00`; all 43 code cells executed with zero errors. The exact
5,000-step NVT + 50,000-step NVE cell took `1194.2 s`, and the 2,048-system
inflight example took `18.9 s`. All eight harmonic checks passed, with a
six-mode AIMNet + Coulomb + D3 frequency MAE of `22.246 cm⁻¹` against B97-3c.
The seven-graph MACE-MP-0b2 scan also passed.

- Executed source notebook SHA-256:
  `eb1f2d4400b00e038d5d4986242214ceaea90f1fbcb2275c401d11243900173c`.
- Executed notebook SHA-256:
  `ab4e58a9565e63845bd28b4990d9b62896408399401ed3c835cf67cecdb36b5f`.
- Trajectory SHA-256:
  `f4cd8be861876244dd547d8dd74abd24b3a1174215f40e295e14939bc95131a9`.
- Run-manifest SHA-256:
  `0ee3a2f6e6c74fde90f78fb8978b7c7b628515df2aef2d250c883508c1e52122`.
- Validator-report SHA-256:
  `c7f6e1f0fe246d55a8b973779aa4ca9a518b70c41634926499e642ea098433d9`.
- The manifest inventory contains 57 files totaling 36,904,592 bytes; all
  portable checksum checks passed.

The independent bundle is
`part-1-scalable-atomistic-workflows/outputs/h100-remaster-3150048/`.
This run confirms the stricter preflight source, but it does not report
distributed timing: the stock transfer and overlap problems remain. The
then-tracked notebook differed from the staged source only in non-scientific
text and used the reviewed `3149917` outputs. It is not acceptance evidence for
the later OrbMol-v2 source or the current SevenNet-Omni source.

### Focused H100 composition diagnostics

CL jobs `3148246` and `3148360` ran focused checks on an NVIDIA H100 NVL with
Core `80aab5c`, Toolkit-Ops `e8e7a74`, AIMNet `0.2.0`, and Torch
`2.12.0+cu130`. Both completed with exit code `0:0`.

- Toolkit and the official AIMNet calculator agreed within `7.27e-4 eV` for
  absolute energy, `8.12e-4 eV` for `E(AB) - E(A) - E(B)`, `1.19e-6 eV/Å`
  for forces, and `8.94e-8 e` for charges.
- For one atomic force component, a 0.003 Å central difference of the official
  total energy differed from the official analytic force by `2.17e-4 eV/Å`.
  The Toolkit analytic force differed from the official force by
  `2.98e-7 eV/Å`.
- A Toolkit total-energy step sweep did not converge: its best observed error
  was `1.10e-2 eV/Å`. Neighbor counts were unchanged at both endpoints, and a
  rigid-monomer interaction-energy check showed the same lost precision.
- Source inspection found the cause in the pinned Core loader: it converts the
  checkpoint's float64 constant atomic reference-energy table to float32. The
  final reported energy is later promoted to float64, but its low bits are
  already lost. Those constants have zero coordinate derivative, so this does
  not explain away a force mismatch; it makes the Toolkit absolute-energy
  finite difference numerically unsuitable.

The notebook therefore uses the fixed 0.003 Å official-energy finite
difference as an independent numerical reference, records its error against
both official and Toolkit analytic forces, and requires both errors to remain
below `0.002 eV/Å`. It does not choose a favorable displacement after seeing
the result. A full Toolkit energy/force finite-difference check should return
after Core preserves the checkpoint's mixed precision.

### Historical AIMNet DSF development checks

- The custom Toolkit residual + DSF + D3 pipeline matched AIMNet's official
  DSF+D3 calculator on `{H2O, D2O, (H2O)6, (D2O)6}` to `1.8e-6 eV` in energy,
  `7.2e-7 eV/Å` in force, and `3.0e-8 e` in charge.
- A 15-step fused NVT→NVE control-flow test captured charges from
  `batch.charges` at `AFTER_STEP` with no second model call. Maximum
  graph-charge error was `9e-8 e`. This is validation only; the notebook has
  no reduced execution mode.
- The integrated float64 path converged the generated hexamer in 181 FIRE2
  steps, then reproduced exactly 5 NVT + 10 NVE steps on a fresh rerun. A
  separate 100 NVT + 1,000 NVE development check kept maximum total-energy
  excursion below `0.085 meV/atom` across all four graphs.
- The custom H/D masses were consumed by the integrators; the model inputs and
  potential predictions remained isotope-invariant.
- The deterministic cyclic hexamer seed reached `fmax < 0.01 eV/Å` in 181
  FIRE2 steps.
- A synthetic 3657 cm⁻¹ dipole recovered its peak within 1.2 cm⁻¹ using the
  5 ps Welch estimator (Fourier-bin spacing: 6.7 cm⁻¹; the Hann response is
  broader).

The local environment could not exercise the target compiled neighbor-hook
path because it lacks Python development headers. Its uncompiled RTX timing is
not a production estimate and did not change the then-current 5,000-step NVT +
50,000-step NVE workload.

### Historical MACE wrapper checks

The Part 1 MACE lesson was also exercised on CPU with Torch `2.12.0+cpu`,
Toolkit Core `80aab5c`, Toolkit-Ops `e8e7a74`, and `mace-torch 0.3.15`.

- Checkpoint: MACE-MP-0b2 medium, 79,462,798 bytes, SHA-256
  `a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf`.
- Seven conventional eight-atom diamond-silicon cells were packed into one
  `Batch` (`7` graphs, `56` atoms).
- One `MACEWrapper` call returned finite energy `(7, 1)`, forces `(56, 3)`,
  and stress `(7, 3, 3)`.
- Toolkit and upstream MACE agreed on CPU to `3.81e-6 eV` for energy,
  `7.93e-7 eV/Å` for forces, and `1.04e-7 eV/Å³` for stress.
- The lowest sampled relative energy and the sampled point closest to zero
  mean normal stress both occurred at `a = 5.50 Å`.

Historical job `3149917` also exercised this path on the H100. The wrapper agreed with
upstream MACE within `7.63e-6 eV` for energy, `6.48e-7 eV/Å` for forces, and
`4.10e-8 eV/Å³` for stress. cuEquivariance and the standard CUDA path agreed
within `3.81e-6 eV`, `6.72e-7 eV/Å`, and `8.20e-8 eV/Å³`, respectively. The
notebook loaded MACE in `7.3 s` and evaluated the seven-cell batch in `7.9 s`.
Historical job `3150048` independently completed the same seven-graph notebook scan.

These checks cover the pinned checkpoint, periodic COO-neighbor path, output
shapes, and CUDA implementation agreement. They are not a silicon accuracy
benchmark.

### DistributedPipeline preflight

**Current-source note, 2026-07-21:** `build/requirements.txt` now selects Core
`331d6b2`. That revision preserves segmented buffer capacity and waits for an
asynchronous send before reusing its storage. It still copies only float32
segmented fields in `Batch.put`, so integer fields such as `atomic_numbers`
still fail the full-dtype preflight. The classic pipeline also retains its
per-iteration completion `all_reduce`. The run record below remains tied to
`80aab5c`.

The planned water campaign was checked against the unmodified Toolkit Core
`0.2.0-rc` head at `80aab5c` at that time. Three problems prevented a valid
result:

- `Batch.put` skips tensor fields that are not stored as float32, including
  `atomic_numbers`. The receiving rank therefore does not have a scientifically
  valid water batch. [PR #130](https://github.com/NVIDIA/nvalchemi-toolkit/pull/130)
  proposed a fix but was closed without being merged.
- A preallocated `Batch.empty` buffer accepts its first `put`, then later writes
  can silently do nothing because the first write trimmed `batch_ptr`.
  [Issue #136](https://github.com/NVIDIA/nvalchemi-toolkit/issues/136) records
  the reproducer; [PR #140](https://github.com/NVIDIA/nvalchemi-toolkit/pull/140)
  is open and unmerged. Rebuilding the pipeline for each 512-system run does
  not avoid this when FIRE2 finishes systems in more than one subset.
- The classic `DistributedPipeline.run()` performs a blocking completion
  `all_reduce` after every iteration. `synchronized=False` removes the optional
  extra barrier, not this completion check. Rank 0 therefore cannot start the
  next FIRE2 batch while rank 1 processes the current dynamics batch, and no
  FIRE2/dynamics overlap has been demonstrated.

The selected unmodified official Core still does not contain all three fixes. Local patches
are not accepted as tutorial results, so distributed timing, throughput,
speedup, and efficiency remain **NOT REPORTED**.

Once an official Core revision preserves every required field, supports
repeated buffer writes, and demonstrates real stage overlap and clean
termination, the planned measurement keeps one pipeline active for each route.
The two-GPU sampler owns all 8,192 systems; the two upstream samplers in the
four-GPU route own 4,096 systems each, with at most 512 active per pair. Before
CUDA or NCCL setup, the producer's CPU preflight checks the first `put`, a second
distinct `put`, a distinct `put` after `zero()`, and exact float, integer, and
boolean fields. The H100 run then checks distinct completed system IDs and
measured overlap on two and four GPUs. Final systems are collected with
Toolkit's `ConvergedSnapshotHook`, and each measured repeat includes setup plus
pipeline fill and drain.

### Superseded H100 diagnostic

CL job `3064655` was initially recorded as a complete H100 validation, but a
later route audit found that pinned `FusedStage` had supplied its default
force-convergence condition to the NVT stage. The already-relaxed batch met
that condition after roughly one update, so the run migrated to NVE instead of
performing the declared 5,000-step warm-up. Its 50,000 saved production frames,
timings, temperatures, and spectral results are retained only as diagnostic
history; they do not validate the current IR science.

### Previous notebook source accepted on H100

CL job `3095065` ran the then-current learner source on an NVIDIA H100 NVL and
finished `COMPLETED` with exit code `0:0` in `24:37`. All 33/33 code cells
executed without an error output, all 17/17 persisted progress cards finished
as `COMPLETE`, and the dynamics cell took `1208.4 s`. The live recorder,
notebook assertion, run manifest, and external validator all recorded the exact
route:

```text
status_0_warmup_steps       5000
status_1_production_steps  50000
```

The accepted source notebook SHA-256 is
`6d7060ceefd52740bd09947e9862cb26a6677fd1e0d98099a05c454fd47b0e7c`;
the complete trajectory SHA-256 is
`54a5d36c04a5b518792bf09b5c6ae138d082665997404d3adb8025e253edb2db`.
Portable bundle checksums passed. The trajectory contains 50,000 dipole,
charge, kinetic-energy, total-energy, and position samples for all four graphs.

That run predates the harmonic-IR cells, the former MACE wrapper lesson,
inflight example, stock-0.2 distributed API lesson, and historical OrbMol-v2
adapter. Its dynamics evidence remains valid for that saved source hash, but it
is not acceptance evidence for the current remaster. Job `3175650` is the
accepted calculation for the later OrbMol source; jobs `3149917` and `3150048`
are historical pre-OrbMol records. None validates the current SevenNet-Omni
adapter or Cu(111) panel.

The Toolkit 3N mean NVE temperatures were `28.4476 K` for H₂O, `35.7652 K`
for D₂O, `99.5318 K` for (H₂O)₆, and `111.0687 K` for (D₂O)₆.
The H₂O/D₂O isotope row failed the declared 20% thermal-state screen. Both
hexamers remained covalently intact and oxygen-connected, but neither initial
directed hydrogen-bond ring persisted in every frame: the H₆ and D₆
initial-ring fractions were `0.57806` and `0.49414`, respectively. The H₆/D₆
row was **NOT REPORTED** because the topology check failed, while both
cluster–monomer rows failed their thermal-state and topology checks. All four
quantitative comparison rows were therefore **NOT REPORTED**. The executed
notebook still supports qualitative inspection of separate finite-temperature
AIMNet MD, double-harmonic B97-3c, and observed-position lanes; it does not
define a shared intensity scale or IR MAE.

This saved run applies only to its recorded source. Current SevenNet-source job
`3189534` supersedes it as the accepted historical execution. The present
source does not redistribute AIMNet, SevenNet, or generated D3 cache files.
It still needs a clean image rebuild, a full H100 run with the current Toolkit
versions, the recorded `DomainParallel` campaign, and human review of the
executed notebook. The
separate `DistributedPipeline` timing remains `NOT REPORTED` until stock Core
passes the transfer checks; that deferred timing is not a Part 1 release
requirement.
