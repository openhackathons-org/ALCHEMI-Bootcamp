# Part 1 NCI Atlas reference data

This is the calculation record for the focused NCI Atlas set used in Stage 3
of the active seven-stage Part 1. It explains the selected reference levels,
interaction-energy definition, measured local checks, and remaining H100 run.
DESS is not included in the learner notebook.

The later IR stages use a separate B97-3c harmonic reference and selected
observed gas-phase band positions. Those answer a different question and do
not share an accuracy metric with the NCI interaction curves.

The historical focused NCI run passed on one H100 with the current Toolkit
source pins. Its six code cells took `22.643220 s`, measured with CUDA
synchronization. Complete-notebook job `3315568` later measured its eight-cell
Stage 3 at `22.988 s` and all notebook code at `798.752 s`. The current learner
cleanup has ten Stage 3 code cells, so its exact-source rerun remains pending.

## Selected data and model

Frozen-monomer dimer interaction curves are the scientific setting for Stage 3
of the active Part 1.

- Model: four-member `aimnet2-wb97m-d3_0` through `_3` ensemble; member 0 is
  retained for the explicit composed-pipeline numerical check.
- Near-matched DFT reference: NCI Atlas absolute dimer and frozen-monomer
  energies at ωB97M-D3(BJ)/def2-TZVPPD.
- Independent reference: NCI Atlas CCSD(T)/CBS interaction energies on the
  same geometries.
- License: NCI Atlas data are CC BY 4.0. Any redistributed subset must retain
  attribution, source revision, and method metadata.

The model was trained at ωB97M-D3/def2-TZVPP with the two-body D3(BJ) term
removed from the training labels and restored at inference. The NCI DFT data
use the same functional and D3(BJ) convention with a diffuse-augmented basis.
This is a near-matched teacher comparison, not an identical level of theory.

## Interaction-energy convention

For every method or model stage, evaluate the dimer and both frozen monomers:

```text
Delta E_int = E(AB) - E(A) - E(B)
```

Do not replace this with `dimer - 2 * monomer` except for a symmetric
homodimer whose two frozen monomer geometries are identical.

The prototype was designed to show three AIMNet stages:

```text
core                 E_NN - E_Coulomb(short range)
core + Coulomb       complete no-D3 AIMNet energy
core + Coulomb + D3  complete AIMNet energy
```

The checkpoint's embedded `SRCoulomb` module is a subtraction. Adding the
full nonperiodic point-charge Coulomb energy is therefore the intended
reconstruction, not double counting.

## Comparison rules

```text
Model result                       Reference                         Use
core                               full CCSD(T)/CBS                  incomplete ablation only
core + Coulomb                     DFT-D3(BJ) minus the same D3      algebraic bookkeeping only
core + Coulomb + D3(BJ)            full DFT-D3(BJ)                   near-matched reference comparison
core + Coulomb + D3(BJ)            CCSD(T)/CBS                       independent reference comparison
```

There is no unique DFT quantity corresponding to "DFT with Coulomb removed."
The model's predicted-charge Coulomb term is an architectural decomposition,
not a quantum-mechanical energy-decomposition analysis. The core-only result
must not be presented as a standalone electronic-structure method.

The no-D3 DFT column is constructed by subtracting the tutorial model's D3
term from the full DFT-D3 endpoint energies. Its error is therefore
algebraically identical to the complete-model error against full DFT-D3. It
checks term accounting only. The CCSD(T)/CBS comparisons provide the
independent evidence for how the explicit Coulomb and D3 additions change the
interaction curves.

## Measured three-curve set

All figures below are live four-member ensemble-mean results over the ten NCI
Atlas separation points. MAE values are in kcal/mol against CCSD(T)/CBS.

```text
Subset       ID       System                         core    + Coulomb    + D3    full vs DFT
HB375x10     1.041    phenol - N-methylacetamide    16.48       2.83      0.20       0.15
D442x10      1.07.74  propyne - methyl azide         6.10       2.48      0.35       0.37
IHB100x10    08.007   ammonia - benzoate             14.39       1.21      0.10       0.11
```

At the nominal equilibrium separation:

```text
System                         CCSD(T)    NCI DFT    core     + Coulomb    full
phenol - N-methylacetamide       -11.89     -11.87   -30.58       -8.57    -11.80
propyne - methyl azide            -3.31      -3.38     3.40       -0.70     -3.08
ammonia - benzoate               -10.24     -10.20   -27.78       -8.81    -10.13
```

The corresponding equilibrium interaction corrections are:

```text
System                         Coulomb correction    D3(BJ) correction
phenol - N-methylacetamide             +22.00                -3.23
propyne - methyl azide                  -4.10                -2.38
ammonia - benzoate                     +18.98                -1.33
```

The signs are model-decomposition results. They should not be relabeled as
physical electrostatic or dispersion energy components of CCSD(T).

For these curves, the NCI DFT versus CCSD(T)/CBS MAEs are 0.08, 0.09, and
0.08 kcal/mol, respectively. This makes the teacher-level and independent
comparisons mutually interpretable without claiming that they are identical.

The ensemble-mean full-model interaction-energy spread, averaged over each
curve, is 0.23, 0.40, and 0.46 kcal/mol, respectively. These are ensemble
disagreements, not calibrated statistical confidence intervals.

## Runtime checks completed before the merge

- Screened all 375 HB375x10 equilibrium complexes.
- Screened the 302 D442x10 equilibrium complexes supported by the checkpoint's
  element set; unsupported noble-gas cases were rejected before inference.
- Screened all 100 IHB100x10 equilibrium complexes.
- Repeated the equilibrium screen for all four checkpoint members.
- Re-evaluated the selected systems at all ten separations for all four
  checkpoint members and selected on ensemble mean plus spread.
- Verified that every calculator stage receives a fresh input mapping. The
  native AIMNet calculator mutates its input dictionary.
- Compared the Toolkit reconstruction with the maintained native AIMNet
  calculator. Per-graph stage energies agreed within `2e-6` eV; the tested
  interaction energy agreed within `2.7e-5` kcal/mol.
- Executed the complete nonvisual computation path on an RTX 4000 SFF Ada using the
  exact pinned Core/Ops source trees, four local official checkpoints, and the
  Toolkit D3 cache. All batching, charge, curve, pipeline, graph-order, force,
  and finite-difference checks passed. This validation used local Torch 2.11;
  the target image pins Torch 2.12.
- Independently executed the replacement Warp Tape graphs against the pinned
  Toolkit/Ops sources with Warp 1.13.0. One heterogeneous model call recorded
  one LJ kernel; three homogeneous calls produced a repeated 3× kernel cluster.
  Inline Graphviz rendering still needed to be checked in the exact image.

## Current publication status

- **Complete:** package only the selected NCI records, with CC BY attribution,
  source Git revision, method, units, fragment charges, and a checksum.
- **Complete in the notebook:** pass fresh data to every model stage, check
  charge conservation, component sums, graph-order agreement, and force finite
  differences. The earlier Toolkit/native calculator agreement result remains a
  separate validation result.
- **Complete:** calculate and plot all ten separation points for all three
  systems; no conclusion is inferred from equilibrium alone.
- **Complete:** periodic Ewald/PME are absent because these are finite
  complexes. The later SevenNet section changes model domain for Cu surfaces
  and does not reuse this molecular checkpoint.

Work that remains before release:

1. Run the complete visible notebook with the exact image pins (Torch 2.12,
   current Toolkit and Toolkit-Ops commits, Warp 1.13.0).
2. Complete the rendered learner review.

## Sources

- NCI Atlas repository and license: <https://github.com/Honza-R/NCIAtlas>
- NCI Atlas total-energy/gradient data:
  <https://github.com/Honza-R/NCIAtlas/tree/main/gradient/wB97M-D3BJ_def2-TZVPPD>
- AIMNet2 model card: <https://huggingface.co/isayevlab/aimnet2-wb97m-d3>
- AIMNet2 method and training convention:
  <https://doi.org/10.1039/D4SC08572H>
