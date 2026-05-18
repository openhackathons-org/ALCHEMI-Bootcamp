# Batching Grid Decision - 2026-05-12

Historical note: this document records the timing and chemistry evidence that
motivated a reduced teaching grid during the 2026-05-12 benchmark pass. The
active notebook control surface has since been simplified to two scopes,
`RUN_SCOPE = "short"` and `RUN_SCOPE = "full"`, plus independent
precomputed-output toggles.

## Goal

Make the tutorial runnable in a teaching session without losing the central story:
batched simulations let us search a chemically meaningful adsorption parameter
space instead of trusting one hand-picked starting geometry.

The tested grid had two notebook scopes plus one temporary reduced benchmark
slice at the time of this run:

- `RUN_SCOPE = "short"`: one CO/Cu(111) check with 4 starts.
- reduced teaching slice: one metal plus two oxides, 9 adsorbate/surface pairs,
  and 55 starts.
- `RUN_SCOPE = "full"`: the complete grid for the notebook's current surfaces,
  231 starting geometries.

## Chemistry Grid

The reduced teaching slice keeps:

- one pure metal surface: Cu(111);
- two oxide surfaces: alpha-Al2O3(0001) and rutile TiO2(110);
- three closed-shell adsorbates: CO, H2O, CH3OH;
- programmatic slabs with explicit Miller indices;
- multiple sites on every surface.

The reduced teaching slice cuts:

- in-plane rotations from `(0, 60, 120)` to `(0)`;
- H2O orientations from three to `O-down` and `flat`;
- CH3OH orientations to `O-down`.

This keeps site competition visible and avoids spending most of the session on
symmetry/rotation variants.

## Structure Generation APIs

No chemistry is token-written. Structures are generated with:

- ASE metal slab builder: `ase.build.fcc111` through
  `helpers.build_cu111_slab`;
- pymatgen `SlabGenerator` through `helpers.build_slab` for oxide Miller
  surfaces;
- `helpers.build_alpha_alumina_0001_slab` for Al2O3(0001);
- `helpers.build_tio2_110_slab` for TiO2(110);
- `helpers.build_config_grid` for site x orientation x rotation x height
  adsorption starts.

TiO2 and ZrO2 now use a generic oxide site finder that selects representative
top-cation, top-oxygen, bridge, and non-duplicate hollow sites from the
generated slab geometry. TiO2 is included in the notebook. ZrO2 remains
available as a programmatic option, but it is not part of the reduced teaching
grid yet.

## Toolkit APIs

The execution path remains Toolkit-only. The notebook uses:

- `AtomicData.from_atoms`;
- `Batch.from_data_list`;
- `MACEWrapper.from_checkpoint`;
- `ConvergenceHook`;
- `FIRE2`;
- Toolkit neighbor hooks and `FreezeAtomsHook` through the existing backend.

No Toolkit API changes were made. The notebook only changes how structures are
grouped before calling the existing batch relaxation path.

## ws-loc Hardware

Benchmarks were run on:

- host: `aad51f7-lcedt` (`ws-loc`);
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition;
- visible memory: 94.97 GiB;
- model: `medium-mpa-0`;
- D3: disabled.

CUDA setup note: the benchmark commands needed
`LD_LIBRARY_PATH=/home/nfedik/projects/tutorials/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH`
so Torch/e3nn could find `libnvrtc-builtins.so.13.0`.

## H2O Saturation

The molecule-only H2O benchmark finds the GPU throughput knee. It is not the
adsorption runtime model, but it is the cleanest way to show batching.

| H2O batch | median structures/s | peak VRAM |
|---:|---:|---:|
| 512 | 532 | 2.48 GB |
| 1024 | 823 | 4.91 GB |
| 2048 | 839 | 9.77 GB |
| 4096 | 878 | 19.48 GB |
| 8192 | 866 | 38.90 GB |
| 12288 | 860 | 58.33 GB |
| 16384 | OOM | - |

Conclusion: throughput is saturated around the 1024-4096 region. Chasing full
VRAM does not improve throughput for this example and eventually OOMs.

## Adsorption Batch Size

Real adsorption batches behave differently from H2O gas because periodic slabs
have larger graph neighborhoods and longer relaxations.

Al2O3(0001)/H2O, fixed 20 steps:

| batch | structures/s | peak VRAM |
|---:|---:|---:|
| 12 | 3.75 | 9.61 GB |
| 24 | 4.64 | 19.20 GB |
| 36 | 4.59 | 28.78 GB |
| 72 | 4.52 | 57.49 GB |
| 96 | 4.46 | 76.63 GB |

TiO2(110)/H2O, fixed 20 steps:

| batch | structures/s | peak VRAM |
|---:|---:|---:|
| 12 | 5.75 | 4.86 GB |
| 24 | 8.88 | 9.68 GB |
| 48 | 8.92 | 19.30 GB |
| 96 | 8.81 | 38.56 GB |

Conclusion: for adsorption, batch 24 is a good tutorial default. Larger
batches consume much more memory without a meaningful throughput gain.

## Homogeneous vs Mixed Batches

Toolkit uses graph-style batching, not dense coordinate padding to the largest
structure. Mixed atom counts are allowed. However, FIRE2 convergence is
batch-level: a difficult structure can keep the whole batch running, even if
other structures already reached the force threshold.

Therefore the notebook keeps homogeneous chunks by adsorbate/surface pair.
That gives stable memory behavior, clearer progress bars, and easier recovery
if one class is slow. Mixed batches are not used in the tutorial run.

## Slow Classes

The prior 252-structure full panel showed that Al2O3 was the bottleneck:

- Cu total: 84 configs, about 4.7 min;
- Pd total: 84 configs, about 4.9 min;
- Al2O3 total: 84 configs, about 48.1 min.

All final Al2O3 structures converged after reruns, so the oxide is not
chemically unusable. It is just the slow class. The worst cases were mainly
H2O/Al2O3 and CH3OH/Al2O3 O-down starts on Al-top/hollow-like sites.

Live convergence checks on the RTX PRO 6000:

| slice | configs | steps to all converged | wall time | peak VRAM |
|---|---:|---:|---:|---:|
| TiO2(110)/H2O full H2O grid | 27 | 555 | 84.85 s | 10.89 GB |
| Al2O3(0001)/H2O reduced slice | 8 | 272 | 25.05 s | 6.48 GB |

TiO2 is a useful second oxide: smaller than Al2O3, still chemically relevant,
and programmatically built from a different Miller index.

## Notebook Changes

- Superseded: the notebook no longer uses a separate teaching-session scope. The
  active control panel exposes only `short` and `full`.
- `TOOLKIT_N_STEPS` is now 1000 as a tutorial cap; structures stop earlier
  when `fmax <= 0.05 eV/A`.
- `ADSORPTION_CHUNK_SIZE = 24` for full adsorption chunks.
- Adsorption relaxation progress is shown per configuration, not only per
  pair.
- H2O speedup sweep has a progress bar.
- Gas-reference relaxations have a progress bar.
- Run metadata now records run scope, chunk size, surfaces, compositions,
  Miller indices, site/orientation filters, rotations, and heights.
