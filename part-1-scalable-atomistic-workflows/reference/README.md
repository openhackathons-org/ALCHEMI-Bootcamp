# B97-3c harmonic IR reference generator

This directory generates a separately computed, inspectable quantum-chemistry
reference for the water IR tutorial. It does not import or modify the AIMNet
notebook.

The clean Part 1 image includes the consumer bundles and every file named by
their published checksum indexes. Unindexed intermediate arrays and working
files remain in the full source checkout and are not needed by the learner
notebook.

The sibling `experimental_water_fundamentals/` bundle is deliberately smaller:
six observed gas-phase H₂-¹⁶O/D₂-¹⁶O positions transcribed from
[Dinu et al., Table 1](https://doi.org/10.1021/acs.jpca.9b07221), published
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). That table
attributes the underlying measurements to Toth's
[H₂-¹⁶O stretch](https://doi.org/10.1006/jmsp.1998.7771),
[H₂-¹⁶O bend](https://doi.org/10.1006/jmsp.1998.7611), and
[D₂-¹⁶O](https://doi.org/10.1006/jmsp.1999.7815) studies. Only the six numeric
positions and source metadata are included. No table image, article text,
experimental spectrum, or intensity data is redistributed.

The calculation is deliberately explicit:

1. optimize H₂O, the tutorial's deterministic cyclic `(H₂O)₆` seed, or one
   supplied XYZ at canonical B97-3c/def2-mTZVP;
2. displace every Cartesian coordinate by `±0.005 bohr`;
3. finite-difference the **full B97-3c gradients** and electronic dipoles;
4. diagonalize the symmetrized Hessian with H masses, then repeat the analysis
   with D masses without another electronic calculation;
5. save the raw samples, Hessian, dipole derivatives, normal modes, topology
   checks, minimum diagnostics, and a hashed manifest.

There is no atom-count cutoff or development-machine shortcut. The built-in
18-atom hexamer requires 108 displaced gradient evaluations plus the reference
gradient after optimization.

## Environment

Create a separate open-source quantum-chemistry environment:

```bash
mamba env create -f environment.yml
conda activate alchemi-b973c-ir
```

Psi4 is LGPL-3.0. `dftd3-python` (the Python API plus `s-dftd3` backend)
and `mctc-gcp` are LGPL-3.0-or-later.
The generator does not redistribute their binaries. Check the
installed package metadata if these outputs will be used outside the tutorial.

## Run

From this directory:

```bash
python b97_3c_ir.py \
  --system h2o \
  --output outputs/h2o-b973c \
  --threads 8 \
  --memory "8 GB"
```

The built-in cyclic seed is numerically identical to
`aux.structures.make_cyclic_water_hexamer()` before optimization:

```bash
python b97_3c_ir.py \
  --system cyclic-h6-seed \
  --output outputs/cyclic-h6-seed-b973c \
  --threads 32 \
  --memory "64 GB"
```

To evaluate the exact geometry saved by the AIMNet workflow, provide its
single-frame XYZ or extxyz instead:

```bash
python b97_3c_ir.py \
  --xyz ../outputs/water_hexamer_relaxed.extxyz \
  --output outputs/aimnet-relaxed-h6-b973c \
  --threads 32 \
  --memory "64 GB"
```

The output directory must be new or empty. Existing results are never
overwritten. geomeTRIC with translation/rotation internal coordinates (`tric`)
is the default for cluster robustness. Use `--coordsys cart` to restart a
difficult geometry in Cartesian coordinates, or `--optimizer optking` for the
native Psi4 optimizer. A failed optimization writes
`optimization_failed_last.xyz`; pass that file back with `--xyz` and select a
new output directory. `PK` integrals are the reproducible default;
`--scf-type df` is available when the density-fitting approximation is
explicitly acceptable. Ordinary DIIS remains enabled, but the initial ADIIS
accelerator is disabled by default: one Psi4 1.11 cluster optimization raised
an ADIIS minimizer exception, while isolated replays of that same geometry
converged to the same energy with either ADIIS or plain DIIS. Select it
explicitly with `--scf-initial-accelerator ADIIS` if needed; the choice and all
other SCF settings are recorded in `run_config.json`.

Each output directory owns a separate `psi4_scratch/` directory. This prevents
concurrent calculations from colliding through Psi4's default `/tmp` PSIO/DIIS
files. Scratch isolation changes no electronic-structure setting.

The standalone generator does not import `aux` or ASE; it carries a small
NumPy-only copy of the seed construction so the Psi4 environment stays
independent. Two legacy path labels remain inside `b97_3c_ir.py` because that
file's exact hash is an anchor in the published H6 provenance reconstruction.
They are descriptive strings, not imports or runtime dependencies.

## Why the basis is explicit

Psi4 normally treats B97-3c as an integrated-basis method. In Psi4 1.11 the
composite finite-difference planner can lose that implicit basis. This driver
therefore calls `b97-3c/def2-mtzvp` and also sets `basis=def2-mTZVP`.
def2-mTZVP is the canonical B97-3c basis, so this is a driver workaround, not a
different level of theory. In the local Psi4 1.11 check, explicit and implicit
single-point energies agreed to `2.8e-14 Eh`; the implicit composite frequency
path failed while the explicit path worked.

## Artifacts

- `input.xyz`, `optimized.xyz`: structures before and after B97-3c optimization;
- `optimization_history.npz`, `optimization_trajectory.xyz`: every retained
  optimizer step for inspection or restart;
- `finite_difference_samples.npz`: every `+h/-h` energy, full gradient, and
  dipole;
- `hessian_raw_Eh_per_bohr2.npy`: unsymmetrized Cartesian finite-difference
  Hessian;
- `hessian_symmetric_Eh_per_bohr2.npy`: matrix used for mode analysis;
- `dipole_derivative_3n_by_3_au.npy`: `d(mu_xyz)/d(R_Axyz)`;
- `modes_h.npz`, `modes_d.npz`: mass-weighted and Cartesian eigenvectors plus
  frequencies and intensities;
- `modes_h.csv`, `modes_d.csv`: plot-ready stick tables;
- `artifacts/{h2o,d2o}/` or `artifacts/{h6,d6}/`: immutable v1 consumer
  bundles, each containing a schema-versioned `manifest.json` and numeric-only
  `ir_arrays.npz` with geometry, masses, Hessian, dipole derivative,
  frequencies, modes, and intensities;
- `diagnostics.json`: gradient, imaginary modes, covalent topology, hydrogen
  bonds, and mass-only isotope checks;
- `run_config.json`, `psi4.out`, `manifest.json`: method, versions, complete
  engine log, and artifact hashes.

The provenance manifests recursively cover the stable files in each retained
run directory, including nested consumer bundles; transient Psi4
`psi.*.clean` sentinels and scratch files are excluded. The H₆ run also
records the exact calculation-source and environment hashes in
`artifacts/provenance/h6/source_provenance.json`; its small reconstruction
patch reproduces the exact calculation generator hash from the current
post-run source. The earlier H₂O calculation retains complete numerical
provenance, but its exact generator source hash was not recorded; that
limitation is explicit in `artifacts/provenance/h2o/source_provenance.json`.

The topology graph is a diagnostic only. It never changes the optimization or
truncates the calculation. Hydrogen bonds use a declared geometric definition
and are reported separately from covalent O-H bonds.

Consumer bundles are withheld unless all gates pass by default:

- maximum reference-gradient component ≤ `1e-5 Eh/bohr`;
- raw finite-difference Hessian max-antisymmetry / max-Hessian ≤ `1e-3`;
- no vibrational imaginary frequency above `10 cm⁻¹` for either isotope;
- preserved covalent water topology and mass-only H/D substitution;
- for the six-water tutorial artifact, a single directed H-bond ring before
  and after optimization.

The manifest embeds these results plus optimizer, SCF, grid, D3(BJ)-ATM, and
gCP provenance. `--allow-invalid-reference` is diagnostic-only: it writes a
bundle marked failed, and the tutorial loader rejects it.

After both monomer and hexamer bundles are present in `artifacts/`, render the
raw sticks, mode-character colors, and the 5 ps Hann-window resolution response:

```bash
python plot_artifacts.py
```

The curve is a finite-window resolution transform, not fitted Gaussian or
Lorentzian broadening. Every panel is normalized independently; the stored
stick intensities remain in km/mol.

The 2026-07-13 figure is a presentation-only rerender with the shared Part 1
plot theme. The checksummed frequency and intensity arrays are unchanged; the
render timestamp, package versions, source hashes, and original reference job
ID are recorded in `artifacts/plot_provenance.json`.

From `reference/artifacts/`, `sha256sum -c SHA256SUMS` verifies the
published bundles, provenance manifests, and rendered four-system figure with
portable relative paths.

## H₂O validation

Using `--optimizer optking`, the manual Cartesian implementation was checked
against Psi4 1.11's built-in
3-point finite-difference frequency driver at the same optimized geometry and
`0.005 bohr` step:

| mode | manual cm⁻¹ | Psi4 driver cm⁻¹ | Δ cm⁻¹ | manual km/mol | Psi4 driver km/mol |
|---:|---:|---:|---:|---:|---:|
| 1 | 1709.5413 | 1709.6038 | -0.0625 | 75.638706 | 75.638450 |
| 2 | 3743.1350 | 3743.1593 | -0.0243 | 0.169252 | 0.169310 |
| 3 | 3853.9864 | 3854.0103 | -0.0239 | 34.581555 | 34.581430 |

This validates the Cartesian finite-difference assembly, not B97-3c's accuracy
against experiment. These are unscaled, 0 K, double-harmonic sticks. The
tutorial trajectory is a finite-temperature, anharmonic predicted-charge
spectrum, so compare band regions, isotope shifts, and mode character—not
one-to-one peak equality.

## Cyclic H₆/D₆ validation

The full 18-atom calculation was run on the intended compute system with
Psi4 1.11, 32 CPU threads, `160 GB` declared memory, PK integrals, the
`99 × 590` DFT grid, and isolated node-local scratch. All 108 displaced
gradients plus the reference gradient completed at the declared `0.005 bohr`
step. The consumer gate passed:

- maximum reference-gradient component: `2.24e-6 Eh/bohr`;
- raw Hessian antisymmetry ratio: `5.07e-4`;
- no significant imaginary modes for H₆ or D₆;
- covalent water topology and the directed six-water ring preserved by the
  B97-3c optimization;
- identical geometry, Hessian, and dipole derivative used for both isotope
  mass analyses.

The immutable consumer IDs are `b97-3c-h6-10138fbfcda149e3` and
`b97-3c-d6-75a130196113794d`. The complete H₆ provenance—including every
finite-difference sample, Hessian, dipole derivative, engine log, and artifact
hash—is stored in `artifacts/provenance/h6/`. `plot_artifacts.py` renders the
four-system result in `artifacts/b97_3c_ir_reference.png` and labels the
cluster panels as the specific cyclic local minimum.

## B97-3c and AIMNet are endpoint-comparable, not term-by-term identical

Canonical Psi4 B97-3c contains the modified B97 functional, D3(BJ)-ATM, and the
geometrical counterpoise/short-range basis correction. The AIMNet checkpoint
used in the tutorial exposes a checkpoint base,
`E_base = E_NN - E_Coulomb^SR`, and asks the wrapper to add matched
**two-body** D3(BJ) and full Coulomb terms. Therefore:

- use this result as a full B97-3c endpoint reference;
- do not subtract Psi4's aggregate dispersion correction and label the
  remainder “the AIMNet checkpoint base”;
- do not assume the canonical ATM/gCP partition matches the checkpoint's
  two-body external-D3 partition;
- note that D3/gCP affect the finite-difference Hessian but have no direct
  electronic dipole derivative in this double-harmonic treatment.

The original B97-3c definition is Brandenburg *et al.*, J. Chem. Phys. 148,
064104 (2018), <https://doi.org/10.1063/1.5012601>. Psi4's integrated 3c method
documentation is <https://psicode.org/psi4manual/1.11.x/gcp.html>.

## Tests

The fast tests exercise geometry generation, XYZ handling, topology checks,
and the Cartesian finite-difference tensor orientation without importing
Psi4:

```bash
python -m unittest discover -s tests -v
```
