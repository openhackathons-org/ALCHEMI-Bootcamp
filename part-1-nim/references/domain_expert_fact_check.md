# Domain Expert Fact-Check Packet

This file is the handoff checklist for validating AdsorbML/MACE adsorption
energy numbers before any row is promoted from `context` to `near-strict` or
`strict`.

## Non-Negotiable Comparability Fields

For every reference energy, record these fields in `manifest.yml` before using
it for parity statistics:

- Exact source record ID, table, figure, or SI page.
- Adsorbate identity, charge/spin if applicable, and gas-phase reference.
- Surface material, Miller index, slab termination, slab layers, and supercell.
- Coverage in adsorbates per surface cell or ML.
- Frozen-layer convention and whether clean slab, gas adsorbate, and adsorbed
  system use the same calculator settings.
- Exchange-correlation functional, +U settings if any, dispersion correction,
  pseudopotentials/PAW set, k-point mesh, and relaxation thresholds.
- Energy sign convention mapped to the tutorial convention:
  `E_ads = E_slab+ads - E_clean_slab - E_gas_ads`, negative exothermic.
- Whether the reported structure is molecular, dissociated, hydroxylated,
  clustered, bilayer, defect-bound, or otherwise outside the tutorial model.

## Primary Model And Dataset References

- Batatia et al., "A foundation model for atomistic materials chemistry",
  arXiv:2401.00096 / DOI: 10.48550/arXiv.2401.00096. Use the exact current
  version and table/SI location for MACE-MP-0, MACE-MP-0b3, MACE-MPA-0, S24,
  and OC157 uncertainty values.
- ACEsuit `mace-foundations` repository. Use this for current model naming and
  training-set distinctions: MACE-MP-0b3 is MPTrj-only, MACE-MPA-0 is
  MPTrj + sAlex.
- Lan et al., "AdsorbML: a leap in efficiency for adsorption energy
  calculations using generalizable machine learning potentials",
  DOI: 10.1038/s41524-023-01121-5. Use this for configuration-search
  methodology, dense reference data, success-rate definitions, and the
  adsorption-energy formula.
- Chanussot et al., "Open Catalyst 2020 (OC20) Dataset and Community
  Challenges", DOI: 10.1021/acscatal.0c04525. Candidate source for exact
  Cu(111)/Pd(111) metal-surface rows.
- Tran et al., "The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide
  Electrocatalysts", DOI: 10.1021/acscatal.2c05426. Candidate source for exact
  oxide rows, but only after matching alpha-Al2O3(0001) termination and
  molecular-vs-dissociative state.
- Grimme et al., "Effect of the damping function in dispersion corrected
  density functional theory", DOI: 10.1002/jcc.21759. Use for DFT-D3(BJ)
  method citation.

## Pair-Level Reference Candidates

### CO / Cu(111)

- Verify against an exact OC20 or AdsorbML dense record first.
- Supporting site-preference context: Ren, Rinke, and Scheffler,
  "Exploring the random phase approximation: Application to CO adsorbed on
  Cu(111)", DOI: 10.1103/PhysRevB.80.045402.
- Supporting functional-dependence context: Stroppa et al., "CO adsorption on
  metal surfaces: A hybrid functional study with plane-wave basis set",
  DOI: 10.1103/PhysRevB.76.195440.

### H2O / Cu(111)

- Do not use Feibelman Ru(0001) as a Cu(111) parity reference. It is only
  mechanistic/motif context.
- Find an exact single-water, low-coverage Cu(111) periodic slab row before
  using any energy. Exclude clusters, bilayers, dissociation, and ice-like
  overlayers unless the notebook model is changed to match them.

### CH3OH / Cu(111)

- Greeley and Mavrikakis, "Methanol Decomposition on Cu(111): A DFT Study",
  DOI: 10.1006/jcat.2002.3586. Verify the table value, coverage, slab, and
  whether molecular methanol adsorption is reported separately from
  decomposition intermediates.

### CO / Pd(111)

- Verify against an exact OC20 or AdsorbML dense record first.
- Hammer, Morikawa, and Norskov, "CO Chemisorption at Metal Surfaces and
  Overlayers", DOI: 10.1103/PhysRevLett.76.2141. Good qualitative hollow-site
  context; check whether fcc vs hcp is resolved for the tutorial coverage.

### H2O / Pd(111)

- Find an exact single-water Pd(111) low-coverage reference before using an
  energy. Exclude bilayer/cluster references unless the notebook model changes.

### CH3OH / Pd(111)

- Schennach, Eichler, and Rendulic, "Adsorption and Desorption of Methanol on
  Pd(111) and on a Pd/V Surface Alloy", DOI: 10.1021/jp021841q. Useful
  experimental/context source; still needs mapping to electronic adsorption
  energy convention.
- If a DFT parity row is desired, locate a periodic DFT source with molecular
  CH3OH/Pd(111), explicit coverage, and gas-phase reference.

### CO / alpha-Al2O3(0001)

- Casarin, Maccato, and Vittadini, "Theoretical study of the chemisorption of
  CO on Al2O3(0001)", DOI: 10.1021/ic000506i. Verify cluster vs periodic model
  before comparing to the tutorial slab.
- Rohmann, Metson, and Idriss, "A DFT study on carbon monoxide adsorption onto
  hydroxylated alpha-Al2O3(0001) surfaces", DOI: 10.1039/C4CP01373E. Use only
  as hydroxylated-surface context unless the tutorial slab is hydroxylated.

### H2O / alpha-Al2O3(0001)

- Shapovalov and Truong, "Ab Initio Study of Water Adsorption on
  alpha-Al2O3 (0001) Crystal Surface", DOI: 10.1021/jp001399g.
- Moskaleva et al., "Elastic polarizable environment cluster embedding
  approach for water adsorption on the alpha-Al2O3(0001) surface",
  DOI: 10.1039/B407082H.
- Hass et al., "First-Principles Molecular Dynamics Simulations of H2O on
  alpha-Al2O3 (0001)", DOI: 10.1021/jp000040p.
- Expert must separate molecular, 1,2-dissociative, 1,4-dissociative,
  hydroxylated, and ambient-water cases.

### CH3OH / alpha-Al2O3(0001)

- Nishimura, Gibbons, and Tro, "Desorption Kinetics of Methanol from
  Al2O3(0001)", DOI: 10.1021/jp981624i. Experimental context only unless
  converted carefully to the tutorial electronic adsorption convention.
- Ethanol/alcohol-on-alumina DFT literature may help with motif checking, but
  it is not a methanol parity source unless adsorbate identity and model match.

## Visualization References

- Stukowski, "Visualization and analysis of atomistic simulation data with
  OVITO", DOI: 10.1088/0965-0393/18/1/015012.
- OVITO Python API `AnariRenderer`: this is the Python renderer class for
  NVIDIA VisRTX/ANARI final-frame rendering.
- OVITO `create_ipywidget`: when a `Pipeline` is passed, the widget shows that
  pipeline in an ad-hoc viewport; this avoids multiple widgets sharing and
  clearing the global scene.

## Current Repo Audit Status

- Active tutorial cache is empty: no nine-pair adsorption energies were present
  under `cached_responses/adsorption-search` during this audit.
- Local shell lacked `ase`, `pymatgen`, `pytest`, `ipywidgets`, and a complete
  OVITO notebook environment, so live geometry tests and widget smoke tests
  were not run here.
- Docker was not installed and no BGR service was live at `localhost:8000`, so
  NIM/MACE energies were not generated in this audit.
