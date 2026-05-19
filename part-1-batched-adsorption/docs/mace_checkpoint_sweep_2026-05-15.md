# Open-MACE Checkpoint Note for OC20Dense Ranking

Date: 2026-05-15, revised 2026-05-18

This note records the validation metric used by the tutorial without shipping
license-gated model outputs.

## Active Tutorial Metric

The compact validation check uses 92 released OC20Dense DFT-relaxed final
geometries for one fixed NH3/surface system:

- `system_id`: `72_7104_115`
- adsorbate: `*NH3`
- neutral label used in local controls: `NH3`
- DFT rank-1 geometry: `rand27`, `sid=3469`
- released DFT adsorption energy at rank 1: `-2.049713755 eV`

The relative-energy metric uses the released DFT rank-1 geometry as the shared
anchor:

```text
DFT gap  = E_DFT(row)  - E_DFT(DFT-rank-1 geometry)
MACE gap = E_MACE(row) - E_MACE(on the DFT-rank-1 geometry)
error    = MACE gap - DFT gap
```

This is not absolute `MACE(row) - DFT(row)` total-energy parity. Within one
fixed system, the slab and adsorbate references are shared across all
geometries, so this metric tests the relative landscape and ranking without
requiring a gas/surface offset convention to match OC20 exactly.

This is also not a model-shifted metric where MACE chooses its own minimum as
zero. The reference stays the DFT rank-1 geometry.

## Open-Model Baseline

The active shipped tutorial reports the open `medium-mpa-0` baseline:

- DFT-rank-1 anchored RMSE: `0.178 eV`
- MAE: `0.150 eV`
- bias: `-0.145 eV`
- Spearman rank correlation: `0.786`
- DFT rank-1 geometry is MACE rank 6
- MACE top geometry is DFT rank 3, only `0.0128 eV` above the DFT minimum

The tutorial should present this as a compact model-sanity check, not as a
universal accuracy claim. It applies to this fixed OC20Dense NH3/surface slice
and this relative-ranking metric.

## License Policy

MACE-MH-1 and related surface-specific heads are license-gated and are not part
of the runnable NVIDIA tutorial path. Do not ship their computed tables,
figures, cached outputs, or metric summaries in this repository. They may be
mentioned only as optional follow-up models for users whose license review
permits separate testing.

## Tutorial Recommendation

- Use `medium-mpa-0` as the open default for the validation and adsorption
  screen.
- Compare open MACE model sizes in the batch-size calibration so readers see
  the model-size, VRAM, and throughput trade-off without moving to a
  license-gated checkpoint.
- Use the ranking check to motivate shortlist generation; use DFT or
  project-specific expert review for strict thermodynamic claims.
