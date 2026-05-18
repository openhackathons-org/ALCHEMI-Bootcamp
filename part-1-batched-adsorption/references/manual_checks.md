# Manual Reference And Statement Checks

This checklist controls which statements may appear as strict scientific claims in the tutorial. A row in `helpers/references.py` should remain `context` until the corresponding evidence is checked here and recorded in `manifest.yml`.

## Global Claims

- [x] Download official open MACE arXiv PDF.
  - Saved as `references/pdfs/mace_foundation_model_arxiv_2401.00096.pdf`.
  - SHA256 recorded in `manifest.yml`.

- [x] Download official open AdsorbML PDF.
  - Saved as `references/pdfs/adsorbml_2023_npj.pdf`.
  - SHA256 recorded in `manifest.yml`.

- [ ] Verify the AdsorbML single-start reliability number from the paper or SI.
  - Current prose mentions roughly 50 percent.
  - Record exact figure/table/SI section and definition of the sampled starting configuration.

- [ ] Verify the AdsorbML multi-start or exhaustive-search reliability number.
  - Current prose mentions above 87 percent.
  - Record whether this refers to ML-relaxed, DFT-relaxed, IS2RE, or another benchmark setting.

- [x] Verify MACE-MPA-0 OC157 relative-energy MAD value.
  - Cached arXiv v3 foundation-model supplement reports MACE-MPA-0+D3 at 0.28 eV MAD on the OC157 molecule-surface relative-energy task.
  - The older 0.42 eV / 121-of-157 numbers correspond to the earlier arXiv/model-naming version and must not be used for the MPA-0 run.
- [x] Verify MACE-MP-0b3 OC157 MAD value.
  - Cached arXiv v3 foundation-model supplement reports MACE-MP-0b3+D3 at 0.38 eV MAD and 126/157 correct lowest-DFT-configuration identifications.
  - This is provenance for the optional literature MAD guide only. It is not
    the active tutorial run label; the runnable workflow uses MACE-MPA-0 with
    D3 disabled unless explicitly reconfigured.

- [ ] Download or manually retrieve restricted PDFs where official direct download failed.
  - ACS OC20 PDF returned HTTP 403.
  - ACS OC22 PDF returned HTTP 403.
  - APS CO/Pd PDF returned HTTP 403.
  - Science water/Ru PDF returned HTTP 403.
  - IOP OVITO URL returned HTML rather than a PDF.

- [ ] Verify that the runnable Toolkit path is using the model checkpoint, model head, D3 setting, PBC, active-mask convention, and optimizer described by the notebook.
  - Evidence should come from the notebook preflight output, cached run metadata, and Toolkit-side run logs.

- [ ] Rewrite or cite broad industrial-adoption claims.
  - If no public source is available, use "representative industrially relevant problem classes" rather than naming companies as active users of this workflow.

## Pair-Level Reference Checks

### CO / Cu(111)

- [ ] Confirm top-site preference from a direct low-coverage CO/Cu(111) source.
- [ ] Confirm exact adsorption energy, functional, dispersion, slab layers, coverage, and sign convention.
- [ ] Decide whether OC20 contains a matching record and record the ID.

### H2O / Cu(111)

- [ ] Replace Feibelman-as-context with a direct single-water Cu(111) reference if strict validation is desired.
- [ ] Separate single-molecule low-coverage adsorption from clusters, bilayers, and dissociation.

### CH3OH / Cu(111)

- [ ] Verify the current `10.1006/jcat.2002.3586` DOI, title, surface, and adsorption-mode relevance.
- [ ] Record exact adsorption energy and modeling details before using as parity.

### CO / Pd(111)

- [ ] Verify whether Hammer/Morikawa/Norskov resolves fcc-hollow vs hcp-hollow for the tutorial coverage.
- [ ] Find or extract a matching OC20 record for strict energy comparison.

### H2O / Pd(111)

- [ ] Find a direct single-water Pd(111) low-coverage reference.
- [ ] Confirm whether the final site should be top, bridge, or another motif under the tutorial convention.

### CH3OH / Pd(111)

- [ ] Verify Desai/Neurock or another primary source for molecular methanol adsorption on Pd(111).
- [ ] Record DOI, surface coverage, and energy convention.

### CO / alpha-Al2O3(0001)

- [ ] Verify the exact alpha-Al2O3(0001) termination used in the tutorial.
- [ ] Confirm whether molecular CO binding to Al-top is a published low-energy reference for that termination.

### H2O / alpha-Al2O3(0001)

- [ ] Separate molecular adsorption from dissociative adsorption and hydroxylated surfaces.
- [ ] Verify whether the tutorial slab termination is chemically stable under water.

### CH3OH / alpha-Al2O3(0001)

- [ ] Confirm molecular methanol adsorption mode and hydrogen-bond motif.
- [ ] Decide whether this remains the discrepancy case after exact reference data are checked.

### NH3 Context

- [ ] Decide whether NH3 is part of the active panel or only a phenomenon icon/context example.
- [ ] If active, verify NH3/Cu(111), NH3/Pd(111), and NH3/alpha-Al2O3(0001) references with exact slab/coverage details.
- [ ] If context only, move active validation rows out of the main AdsorbML table or mark them explicitly as optional.

## Geometry And Code Checks

- [ ] Verify Cu and Pd slabs are true (111) surfaces.
- [ ] Verify alpha-Al2O3 slab is true (0001) and report termination.
- [ ] Verify layer count and active-mask frozen-layer convention.
- [ ] Verify fcc/hcp site labels against second-layer stacking.
- [ ] Verify adsorbate starting geometries have no unphysical overlaps.
- [ ] Verify final-site classification is robust to periodic wrapping.

## Link Checking

- [ ] Run a link checker against markdown and notebook references.
- [ ] Ignore localhost links and internal deployment URLs.
- [ ] Record any DOI redirects or access restrictions.

Suggested ignore file: `references/link_check_ignore.txt`.
