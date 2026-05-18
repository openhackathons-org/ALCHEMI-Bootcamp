# MACE Checkpoint Sweep for OC20Dense Ranking

Date: 2026-05-15

## Goal

Select a MACE model for the tutorial validation section using a comparison that
matches the scientific question: can the model preserve the relative adsorption
landscape for many candidate geometries of the same surface and adsorbate?

The test uses 92 released OC20Dense DFT-final geometries for one fixed system:

- `system_id`: `72_7104_115`
- adsorbate: `*NH3`
- neutral label used in the local gas-control check: `NH3`
- DFT rank-1 geometry: `rand27`, `sid=3469`
- released DFT adsorption energy at rank 1: `-2.049713755 eV`

## Metric

Use the released DFT rank-1 geometry as the shared anchor:

```text
DFT gap  = E_DFT(row)  - E_DFT(DFT-rank-1 geometry)
MACE gap = E_MACE(row) - E_MACE(on the DFT-rank-1 geometry)
error    = MACE gap - DFT gap
```

This is intentionally not `MACE(row) - DFT(row)` total-energy parity. Within
one fixed system, the slab and adsorbate references are shared across all
geometries, so this metric tests the relative landscape and ranking without
requiring a gas/surface offset convention to match OC20 exactly.

This is also intentionally not a model-shifted metric where MACE chooses its own
minimum as zero. The reference stays the DFT rank-1 geometry.

## What Was Run

All generated artifacts are local and ignored:

`part-1-batched-adsorption/outputs/explorations/mace_checkpoint_sweep/`

Swept checkpoint groups:

- MACE-MP / MACE-MPA family: `small`, `medium`, `large`, `small-0b`,
  `medium-0b`, `small-0b2`, `medium-0b2`, `medium-0b3`, `large-0b2`,
  `medium-mpa-0`
- MACE-OMAT / MACE-MATPES family: `small-omat-0`, `medium-omat-0`,
  `mace-matpes-pbe-0`, `mace-matpes-r2scan-0`
- MACE-MH default wrapper checks: `mh-0`, `mh-1`
- Explicit MACE-MH head checks: `mh-0:oc20_usemppbe`,
  `mh-1:oc20_usemppbe`, `mh-1:omat_pbe`, `mh-1:matpes_r2scan`,
  `mh-0:omat_pbe`

The installed MACE registry was checked in
`.venv-toolkit/lib/python3.12/site-packages/mace/calculators/foundations_models.py`.
The sweep covers every key listed in its `mace_mp_urls` dictionary for the
periodic foundation-model path. Molecular-only models such as MACE-OFF/OMOL
are separate calculator entry points and were not treated as periodic
adsorbate-surface candidates here.

The normal Toolkit backend now supports explicit MACE-MH heads through the
`TOOLKIT_HEAD` environment variable. This was needed because multi-head MACE
models only receive a head choice when the input dictionary carries a `head`
tensor.

## Results

Best explicit-head result:

- `mh-1:oc20_usemppbe`
- RMSE: `0.0716 eV`
- MAE: `0.0598 eV`
- bias: `-0.0458 eV`
- Spearman rank correlation: `0.943`
- DFT rank-1 geometry recovered as MACE rank 1
- MACE top geometry is the DFT rank-1 geometry

After the provenance-guarded notebook rerun, the visible validation cell reports
the same result within numerical noise: RMSE `0.0716 eV`, MAE `0.0598 eV`,
bias `-0.0458 eV`, Spearman `0.943`, and DFT rank 1 recovered as MACE rank 1.
Benchmark chunks and output rows now stamp checkpoint, head, device, dtype, and
D3 state so stale outputs from another model cannot be silently reused.

Best simpler single-head/default Toolkit result:

- `medium-mpa-0`
- RMSE: `0.178 eV`
- MAE: `0.150 eV`
- bias: `-0.145 eV`
- Spearman rank correlation: `0.786`
- DFT rank-1 geometry is MACE rank 6
- MACE top geometry is DFT rank 3, only `0.0128 eV` above the DFT minimum

The larger older MACE-MP variants did not improve this case. For example,
`large-0b2` gives RMSE `0.511 eV` and picks a DFT rank-23 geometry as its top
configuration.

Full result tables:

- `outputs/explorations/mace_checkpoint_sweep/all_checkpoint_and_head_rank1_anchored_summary.csv`
- `outputs/explorations/mace_checkpoint_sweep/checkpoint_rank1_anchored_summary_all_mace_mp_sorted.csv`
- `outputs/explorations/mace_checkpoint_sweep/mh_head_checks/mh_explicit_head_rank1_anchored_summary.csv`

## Absolute Eads Control

A neutral-gas subtraction was tested for the best explicit head:

```text
Eads_defined = E(adslab) - E(clean slab) - E(NH3 gas)
```

For `mh-1:oc20_usemppbe`, this produced:

- surface energy: `-322.935272 eV`
- NH3 gas energy: `-19.298113 eV`
- absolute defined Eads MAE: `0.868 eV`
- absolute defined Eads RMSE: `0.869 eV`
- absolute defined Eads bias: `+0.868 eV`

Conclusion: do not present neutral-gas absolute Eads parity as the tutorial
validation result. It is a useful control because it shows the reference
convention matters. The notebook now skips relaxed-adslab Eads for the 92-row
NH3 ranking check unless the relaxed-adslab table has matching checkpoint/head
provenance. For absolute adsorption-energy claims, follow AdsorbML: rank with
ML, then use DFT single-points or DFT relaxations for the final energy when
strict accuracy is required.

## Literature Framing

AdsorbML frames the practical task as an adsorption-configuration search. It
uses ML relaxations to rank many candidate geometries, then optionally sends the
top `k` structures to DFT single-point or DFT relaxation follow-up. The paper's
reported success criterion is finding a configuration within `0.1 eV` of the
OC20Dense DFT minimum, not treating direct ML adsorption energy as final truth.

Useful anchors from the AdsorbML paper:

- The abstract reports a balanced option finding the lowest-energy
  configuration `87.36%` of the time with about `2000x` compute speedup.
- Table 2 reports direct ML-predicted-energy success rates and MAEs for OC20
  models. The best direct row there is eSCN-MD-Large: success `56.52%`, energy
  MAE `0.1739 eV`.
- The algorithm section states that ML energies are used to rank configurations,
  while the final energy in hybrid strategies comes from DFT.

Useful anchors from MACE sources:

- MACE foundation-model docs list MACE-MH-1/0 as multi-head models covering
  OMAT, OMOL, OC20, and MATPES for materials, molecules, and surfaces.
- The MACE-MH-1 model card lists `oc20_usemppbe` as the surface catalysis /
  adsorbates head and reports surface benchmark MAEs, including OC20 adsorption
  MAE `0.138 eV`.

Sources:

- AdsorbML paper: https://www.nature.com/articles/s41524-023-01121-5
- AdsorbML Table 2: https://www.nature.com/articles/s41524-023-01121-5/tables/2
- MACE foundation-model docs: https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html
- MACE-MH-1 model card: https://huggingface.co/mace-foundations/mace-mh-1

## Tutorial Recommendation

Use one of two paths:

1. Conservative/simple default: keep `medium-mpa-0` for the tutorial runtime
   path and present its `0.178 eV` DFT-rank-1 anchored RMSE as a compact
   validation check. This stays close to the existing notebook setup.
2. Stronger validation path: use `mh-1` with `TOOLKIT_HEAD=oc20_usemppbe` in
   the validation section. This gives the best result found so far
   (`0.0716 eV` RMSE) and cleanly recovers the DFT minimum, but the notebook
   must explain the multi-head model/head choice explicitly.

Either way, keep the tutorial story as:

- Generate many candidate adsorption geometries.
- Batch the candidate evaluation on GPU through the Toolkit.
- Validate the selected model on a released DFT-backed adsorption landscape.
- Use model ranking to build a shortlist; use DFT or domain review for strict
  final thermodynamic claims.

The qualitative AdsorbML-style relaxation-landing fallback is not needed for
the current inclusion decision because the fixed-geometry quantitative metric
now has a strong option. It remains a useful future visual/trajectory section:
launch several starts in batch and classify whether they land in the same DFT
minimum, a near-equivalent minimum, a model-only candidate, or a failure.

## Reproduction Commands

Common cache/runtime exports used during the sweep:

```bash
export XDG_CACHE_HOME=/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/.xdg-cache
export WARP_CACHE_DIR=/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/.warp-cache
export MPLCONFIGDIR=/home/nfedik/projects/tutorials/part-1-batched-adsorption/outputs/.matplotlib-cache
export LD_LIBRARY_PATH=/home/nfedik/projects/tutorials/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib:/home/nfedik/projects/tutorials/.venv-toolkit/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
```

MACE-MP sweep pattern:

```bash
base=part-1-batched-adsorption/outputs/explorations/mace_checkpoint_sweep
for ckpt in small medium large small-0b medium-0b small-0b2 medium-0b2 medium-0b3 medium-mpa-0 large-0b2; do
  safe="${ckpt//[^A-Za-z0-9._-]/_}"
  TOOLKIT_CHECKPOINT="$ckpt" OC20DENSE_SP_CHUNK_SIZE=12 \
    .venv-toolkit/bin/python part-1-batched-adsorption/scripts/run_oc20dense_dft_final_single_points.py \
      --toolkit-root part-1-batched-adsorption/outputs/oc20dense_known_examples \
      --dft-check-dir "$base/fixed_inputs/dft_reference_checks" \
      --outdir "$base/$safe/dft_final_single_points" \
      --systems 72_7104_115 \
      --chunk-size 12 \
      --force
done
```

Best explicit-head normal Toolkit backend reproduction:

```bash
TOOLKIT_CHECKPOINT=mh-1 TOOLKIT_HEAD=oc20_usemppbe OC20DENSE_SP_CHUNK_SIZE=8 \
  .venv-toolkit/bin/python part-1-batched-adsorption/scripts/run_oc20dense_dft_final_single_points.py \
    --toolkit-root part-1-batched-adsorption/outputs/oc20dense_known_examples \
    --dft-check-dir part-1-batched-adsorption/outputs/explorations/mace_checkpoint_sweep/fixed_inputs/dft_reference_checks \
    --outdir part-1-batched-adsorption/outputs/explorations/mace_checkpoint_sweep/mh-1-oc20_usemppbe-toolkit-head/dft_final_single_points \
    --systems 72_7104_115 \
    --chunk-size 8 \
    --force
```

Verification run after code changes:

```bash
.venv-toolkit/bin/python -m py_compile \
  part-1-batched-adsorption/helpers/relaxation_backends.py \
  part-1-batched-adsorption/scripts/run_oc20dense_known_examples.py \
  part-1-batched-adsorption/scripts/run_toolkit_full_panel.py \
  part-1-batched-adsorption/scripts/run_toolkit_step_diagnostics.py \
  part-1-batched-adsorption/scripts/run_oc20dense_dft_final_single_points.py \
  part-1-batched-adsorption/scripts/run_oc20dense_mace_adsorption_energies.py \
  part-1-batched-adsorption/scripts/summarize_oc20dense_accuracy.py

.venv-toolkit/bin/python -m pytest -q \
  part-1-batched-adsorption/tests/test_relaxation_backends.py \
  part-1-batched-adsorption/tests/test_oc20dense_benchmark.py
```
