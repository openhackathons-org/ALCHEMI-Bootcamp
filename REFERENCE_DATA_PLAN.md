# Part 3 reference data and validation plan

Status: four-member ensemble-screened candidate set, 2026-07-10.

## Decision

Keep frozen-monomer dimer interaction curves as the scientific spine of the
Part 3 prototype.

- Model: four-member `aimnet2-wb97m-d3_0` through `_3` ensemble; member 0 is
  retained for the explicit composed-pipeline parity check.
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

## Energy convention

For every method or model stage, evaluate the dimer and both frozen monomers:

```text
Delta E_int = E(AB) - E(A) - E(B)
```

Do not replace this with `dimer - 2 * monomer` except for a symmetric
homodimer whose two frozen monomer geometries are identical.

The notebook will show three AIMNet stages:

```text
core                 E_NN - E_Coulomb(short range)
core + Coulomb       complete no-D3 AIMNet energy
core + Coulomb + D3  complete AIMNet energy
```

The checkpoint's embedded `SRCoulomb` module is a subtraction. Adding the
full nonperiodic point-charge Coulomb energy is therefore the intended
reconstruction, not double counting.

## What may be compared

```text
Model result                       Reference                         Use
core                               full CCSD(T)/CBS                  incomplete ablation only
core + Coulomb                     DFT-D3(BJ) minus matched D3       teacher-level accuracy
core + Coulomb + D3(BJ)            full DFT-D3(BJ)                   teacher-level accuracy
core + Coulomb + D3(BJ)            CCSD(T)/CBS                       external scientific accuracy
```

There is no unique DFT quantity corresponding to "DFT with Coulomb removed."
The model's predicted-charge Coulomb term is an architectural decomposition,
not a quantum-mechanical energy-decomposition analysis. The core-only result
must not be presented as a standalone electronic-structure method.

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

## Runtime checks already passed

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
  and finite-difference gates passed. This validation used local Torch 2.11;
  the target image pins Torch 2.12.
- Independently executed the replacement Warp Tape graphs against the pinned
  Toolkit/Ops sources with Warp 1.13.0. One heterogeneous model call recorded
  one LJ kernel; three homogeneous calls produced a repeated 3× kernel cluster.
  Inline Graphviz rendering remains part of the exact-image release gate.

## Publication gate status

- **Complete:** package only the selected NCI records, with CC BY attribution,
  source Git revision, method, units, fragment charges, and a checksum.
- **Complete in the notebook:** pass fresh data to every model stage, check
  charge conservation, component sums, graph-order parity, and force finite
  differences. The earlier Toolkit/native calculator parity result remains a
  separate validation result.
- **Complete:** calculate and plot all ten separation points for all three
  systems; no conclusion is inferred from equilibrium alone.
- **Complete:** adsorption and periodic Ewald/PME are absent. They belong to
  later parts with their own model-domain and reference validation.

Remaining before publication:

1. Rerun the visible notebook in the exact image pins (Torch 2.12, Toolkit and
   Toolkit-Ops commits, Warp 1.13.0) with AIMNet and D3 caches prewarmed.
2. Complete the marked visual and reference-language reviews.

## Sources

- NCI Atlas repository and license: <https://github.com/Honza-R/NCIAtlas>
- NCI Atlas total-energy/gradient data:
  <https://github.com/Honza-R/NCIAtlas/tree/main/gradient/wB97M-D3BJ_def2-TZVPPD>
- AIMNet2 model card: <https://huggingface.co/isayevlab/aimnet2-wb97m-d3>
- AIMNet2 method and training convention:
  <https://doi.org/10.1039/D4SC08572H>
