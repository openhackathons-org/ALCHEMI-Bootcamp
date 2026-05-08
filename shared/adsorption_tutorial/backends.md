# Backend Adapter Contract

The adsorption tutorial has one scientific panel and two execution backends:

- `bgr_nim`: Part 1, HTTP requests to the ALCHEMI Batch Geometry Relaxation NIM.
- `toolkit`: Part 2, local ALCHEMI Toolkit calculators/optimizers.

Both backends must use the same host/adsorbate panel in `contract.py` and emit
the same result columns. A result table from either backend should be valid
input to the same plotting, validation, and expert-review code.

## BGR NIM Adapter

Current owner: `part-1-nim`.

Responsibilities:

- Build canonical slabs and adsorbate starting configurations.
- Convert ASE structures to BGR request payloads.
- Submit clean-slab, gas-phase, and slab+adsorbate relaxations through
  `/v1/infer`, or replay cached JSON responses.
- Emit one row per relaxed starting configuration with `backend = "bgr_nim"`.
- Preserve NIM metadata: model name/version, D3 setting, PBC flag, optimizer
  preset, and `opttol`.

## Toolkit Adapter

Current owner: shared adapter in `part-1-nim/helpers/relaxation_backends.py`,
with the fuller narrative living in `part-2-toolkit`.

Responsibilities:

- Reuse the same canonical slabs and starting configurations.
- Run clean-slab, gas-phase, and slab+adsorbate optimizations through the native
  ALCHEMI Toolkit API: `AtomicData`, `Batch.from_data_list`, `MACEWrapper`,
  `PipelineModelWrapper`, and `FIRE2`.
- Freeze inactive slab atoms by translating the shared active mask into Toolkit
  `atom_categories` and `FreezeAtomsHook`; do not silently ignore constraints.
- Emit one row per relaxed starting configuration with `backend = "toolkit"`.
- Record calculator name, checkpoint/model version, dispersion treatment,
  precision/device, optimizer, force threshold, and cell/PBC handling.
- If the `toolkit` backend is explicitly selected and the required native
  package/API is unavailable, fail immediately with a backend-unavailable error.
  It must not redirect to BGR NIM or cache replay.
- BGR parity requires explicit DFT-D3(BJ) damping parameters from verified
  runtime metadata or documentation. Do not hide unverified D3 defaults in the
  adapter.

## Shared Analysis Boundary

These operations must be backend-neutral:

- adsorption energy:
  `E_ads = E_slab+ads - E_clean_slab - E_gas_ads`, negative exothermic;
- final-site classification from relaxed geometry, not starting label;
- convergence, force, desorption, and dissociation filtering before selecting
  the batch minimum;
- strict-vs-context reference gating;
- plots and domain-expert fact-check tables.

## Promotion Rule

The toolkit version should not be advertised as equivalent to the BGR NIM
version until both backends produce the required result schema for at least the
small-panel smoke case and agree on:

- slab atom count/layer count/termination;
- gas reference energies from the same calculator convention;
- clean slab energies from the same calculator convention;
- final-site classifications;
- adsorption energies within the documented model/backend tolerance.
