# Execution Adapter Notes

The active Part 1 tutorial exposes one execution path: native ALCHEMI Toolkit
batch relaxation on the GPU.

The shared science contract remains backend-neutral on purpose. A future
service/API tutorial can reuse the same panel, reference manifest, geometry
checks, and result schema, but it should live as a separate route and prove
equivalence before it is presented beside the Toolkit notebook.

## Toolkit Adapter

Current owner: `part-1-batched-adsorption/helpers/relaxation_backends.py`, with
the teaching narrative in the Part 1 notebook.

Responsibilities:

- Reuse the canonical slabs and starting configurations from the shared panel.
- Run clean-slab, gas-phase, and slab+adsorbate optimizations through native
  ALCHEMI Toolkit APIs: `AtomicData`, `Batch.from_data_list`, `MACEWrapper`,
  `PipelineModelWrapper`, and `FIRE2`.
- Freeze inactive slab atoms by translating the shared active mask into Toolkit
  `atom_categories` and `FreezeAtomsHook`; do not silently ignore constraints.
- Emit one row per relaxed starting configuration with `backend = "toolkit"`.
- Record calculator name, checkpoint/model version, dispersion treatment,
  precision/device, optimizer, force threshold, and cell/PBC handling.
- If the required Toolkit package/API is unavailable, fail with a clear
  Toolkit-unavailable error.

## Shared Analysis Boundary

These operations must stay independent of the execution adapter:

- adsorption energy:
  `E_ads = E_slab+ads - E_clean_slab - E_gas_ads`, negative exothermic;
- final-site classification from relaxed geometry, not starting label;
- convergence, force, desorption, and dissociation filtering before selecting
  the batch minimum;
- strict-vs-context reference gating;
- plots and domain-expert fact-check tables.

## Future Service/API Promotion Rule

A service/API route should not be advertised as equivalent to the Toolkit
teaching version until both routes produce the required result schema for at
least the short smoke case and agree on:

- slab atom count/layer count/termination;
- gas reference energies from the same calculator convention;
- clean slab energies from the same calculator convention;
- final-site classifications;
- adsorption energies within the documented model/route tolerance.
