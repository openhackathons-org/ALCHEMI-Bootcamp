# Part 1 rebuild — predicted-charge water IR

This directory contains the integrated water-dimer-to-IR Part 1 rebuild from
[`PART1_DESIGN_HANDOFF.md`](../PART1_DESIGN_HANDOFF.md). It is intentionally
separate from the legacy adsorption notebook, which is not part of this
molecular-model lesson.

Open [`alchemi-water-ir.ipynb`](alchemi-water-ir.ipynb) in the unified ALCHEMI
container. The notebook runs the full 5,000-step NVT warmup and 50,000-step NVE
production trajectory; it does not shorten the calculation for the development
GPU.

## Code organization

The notebook shows the Toolkit path directly: ASE structures become
`AtomicData`, structures become a `Batch`, neighbors are built, model wrappers
are configured, the residual/finite-system-all-pairs-Coulomb/D3 pipeline is
assembled, and dynamics hooks are registered in visible cells. Supporting
structure builders, scientific checks, spectrum analysis, plotting, artifact
I/O, and progress-card HTML live in focused [`aux/`](aux/) modules. `aux` has no
broad package-level re-exports; see [`aux/README.md`](aux/README.md) for the
module boundary.

Key implementation choices:

- AIMNet2-2025 B97-3c-derived residual + published pairwise D3(BJ) + the
  checkpoint calculator's default `simple` finite-system all-pairs Coulomb
  (no spatial cutoff and no periodic images);
- eager execution for the variable-size dimer scan, followed by default
  `torch.compile` only on the fixed 42-atom isotope × cluster batch;
- compiled energy, forces, and charges must agree with the eager result and
  with a second synchronized compiled call before FIRE2 or dynamics begins;
- DSF is a scalable finite-cutoff alternative for larger finite systems;
  Ewald and PME are periodic methods, and none of the three replaces `simple`
  in this vacuum-cluster endpoint;
- H/D isotope substitution changes masses only;
- one shared model call per fused dynamics step;
- predicted charges are read from `batch.charges` at `AFTER_STEP` without a
  second forward pass;
- NVT uses a never-passing force-convergence criterion, so only its exact
  5,000-step limit can advance `FusedStage` to NVE;
- the recorder and external validator require exactly 5,000 status-0 warm-up
  calls followed by 50,000 status-1 production calls;
- total dipole sampled every 0.5 fs;
- 5 ps, 50%-overlapped Welch dipole-current spectrum;
- float64 dynamics with all 50,000 position frames retained;
- after FIRE2, the final force and isotope checks run before the validated
  batch is written once through `ZarrData` and replayed; no repeated
  convergence-snapshot writes are attached to standalone FIRE;
- post-run charge, temperature, energy-drift, O–H, and oxygen-skeleton checks;
- checksummed Psi4 B97-3c harmonic references, raw DFT sticks, a 5 ps
  Hann-window resolution envelope, and continuous-mass H→D mode mapping.

The deterministic cyclic hexamer is generated in code and FIRE2-relaxed before
dynamics. No third-party coordinate asset is redistributed.

## D3 runtime asset

The notebook supplies the Toolkit D3 wrapper with an explicit `param_file` and
`auto_download=False`. By default it reads
`~/.cache/nvalchemiops/dftd3_parameters.pt`; deployments can set
`ALCHEMI_D3_PARAM_FILE` to another prewarmed path. The file must have SHA-256:

```text
b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84
```

The notebook fails before model composition if the file is absent or its hash
differs. This repository does not bundle the parameter tensor while its
redistribution rights remain unconfirmed; environment setup must provision it.

## Compilation acceptance gate

Changing graph sizes can trigger shape-specific compilation, so the dimer scan
and editable trial remain eager. The long-running model is compiled only for
the fixed 42-atom batch. Before that model is used, the notebook requires:

```text
compiled vs eager   energy < 5e-6 eV; forces < 5e-6 eV/Å; charges < 2e-7 e
compiled repeat     energy < 2e-6 eV; forces < 2e-6 eV/Å; charges < 1e-7 e
```

Each comparison uses the same coordinates, neighbors, requested outputs, and
a synchronized call. Compilation is accepted only as an execution change; it
must not change the model contract.

## Accepted H100 run — 2026-07-13

CL job `3087665` completed with exit 0 on an NVIDIA H100 80 GB. It executed all
31 code cells with zero error outputs, and all 14 live progress cards persisted
as `COMPLETE`. The NVT stage received
`ConvergenceHook.from_fmax(threshold=-1.0)`, so its declared step count was the
sole NVT→NVE transition condition. The live recorder, notebook assertion, run
manifest, and external validator all recorded:

```text
status_0_warmup_steps       5000
status_1_production_steps  50000
```

The dynamics cell took 1,232.4 s (20.54 min); the complete scheduler job took
23:02. The accepted source notebook SHA-256 is
`5403dfcd42bb707e15527a443e76edaec38fe38a8888ab8d527433b1dbf8efc8`;
the complete trajectory SHA-256 is
`ca2251061694e067f317fdb01d044897c8d913aff67e90e8f88ea5aaa6597f88`.
All portable artifact checksums pass. The complete accepted bundle is in
the intentionally gitignored local validation path
`outputs/h100-remaster-3087665/`; it is not distributed in a fresh clone.

The current learner-facing notebook is a presentation revision with SHA-256
`81124de2e95e709a527522d026288a2c98d7e41b90ce7c4dd93e17a557b5a667`.
Relative to the accepted H100 source, all 31 scientific code cells are
unchanged; the sole changed code cell only relabels its CPU/GPU conclusion from
`CHECK` to `RESULT — OBSERVED`. Markdown, banner, progress semantics, and plot
styling changed. This exact presentation revision has local contract tests but
has not been rerun on H100.

The earlier RTX 4000 SFF Ada control-flow test remains development provenance;
it is not a reduced tutorial mode and is no longer the acceptance evidence.

### What the accepted run supports

The complete residual + Coulomb + D3 curve reaches `0.6481 kJ mol⁻¹` MAE
against the full canonical B97-3c water-dimer curve. The ablations are not a
monotonic correction ladder:

```text
residual only                         8.1981 kJ mol⁻¹
residual + D3; Coulomb omitted        9.2579 kJ mol⁻¹
residual + Coulomb; D3 omitted        2.5363 kJ mol⁻¹
residual + Coulomb + D3               0.6481 kJ mol⁻¹
```

Explicit electrostatics is the dominant missing term for this hydrogen-bonded
scan. All four curves are shown against full B97-3c for context, but only the
complete Toolkit model is interpreted as the endpoint/reference comparison.
The partial-model MAEs are ablation distances that mix omitted physics with ML
error; they are not matched-level accuracy estimates.

The same run measured the CPU/GPU crossover rather than assuming it. CPU was
faster for one graph (`2.90` vs `4.49 ms`), while the H100 was faster by batch
32 and evaluated batch 128 in `4.51 ms` versus `26.88 ms` on CPU. For the same
40-graph heterogeneous workload, one mixed GPU call took `4.47 ms`; three
homogeneous bucket calls took `13.46 ms`. Launch savings dominated bucketing
for this measured workload.

### Scientific reporting boundary

The exact route produced inspectable predicted-charge spectra, but all four
comparative headline IR quantities were correctly withheld:

- H₂O/D₂O failed the 20% thermal-pair gate (`0.22714` relative difference);
- H₆/D₆ passed thermal matching (`0.08939`) but failed initial-ring
  persistence;
- both cluster-minus-monomer comparisons failed thermal and topology gates;
- cyclic-hexamer DFT overlays are therefore withheld.

Both clusters remained covalently intact and oxygen-connected. The original
ring did not persist in every frame: H₆ retained it for `0.85718` of frames and
first lost it at `1.2325 ps`; D₆ retained it for `0.97788` and first lost it at
`5.2825 ps`. Raw MD band centroids remain observations, not validated shifts:
H₂O `3827.91`, D₂O `2677.25`, H₆ `3434.60`, and D₆ `2489.99 cm⁻¹`.

### Superseded 2026-07-10 DSF run — diagnostic only

CL job `3064655` requested 55,000 updates, but pinned `FusedStage` supplied its
default force convergence to NVT. The already-relaxed batch satisfied it after
roughly one update, so the workflow migrated to NVE instead of completing the
nominal 5,000-step warm-up. The 50,000 saved frames therefore do not constitute
a valid 5,000-step NVT + 50,000-step NVE trajectory.

The old dynamics cell took 1268.6 s and the complete job took 25:33. Those
numbers, the executed-notebook hashes, and artifact hashes are retained only as
provenance and rough performance diagnostics for the defective route. The
former temperature, energy-drift, topology, spectrum, isotope, and cluster
interpretations are invalid and are not evidence for the current tutorial.

The old DSF-era composition independently matched AIMNet's official DSF+D3
calculator to `1.8e-6 eV` in energy, `7.2e-7 eV/Å` in force, and `3.0e-8 e` in
charge. These are historical implementation diagnostics only; they do not
validate the current `simple`+D3 composition or either dynamics route.

## Independent harmonic references

The checksummed H₂O/D₂O and cyclic H₆/D₆ endpoint references were generated at
canonical B97-3c/def2-mTZVP. The H₂O unscaled harmonic modes are `1709.54`,
`3743.13`, and `3853.99 cm⁻¹`; changing hydrogen masses only gives D₂O modes
at `1251.19`, `2698.32`, and `2823.06 cm⁻¹`. The 18-atom cyclic reference also
passed the gradient, finite-difference symmetry, minimum, covalent-topology,
and hydrogen-bond-ring gates for both isotope mass sets. See
[`reference/`](reference/) for the reproducible generator, full provenance,
four-system plot, and license boundary.

Canonical B97-3c contains D3(BJ)-ATM, while AIMNetCentral's explicit D3 layer
contains pairwise C6/C8 terms only. Public AIMNet2 materials do not establish
whether ATM remained in the learned residual. The DFT calculation is therefore
shown as a full endpoint reference, not exact term-by-term parity.
