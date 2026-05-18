# Next Plan - Miller-Index Search and Visual Review

## Correction

The completed 2026-05-12 ws-loc artifact run regenerated trajectories and logs
for the current fixed-surface panel and OC20Dense validation set. It did **not**
implement the requested material-specific Miller-index sweep.

That means the current full-panel run answers:

> For the fixed surfaces Cu(111), Pd(111), and Al2O3(0001), how do different
> starting sites, orientations, rotations, and heights relax under the Toolkit
> MACE workflow?

It does **not** yet answer:

> For one material, how does the best adsorption structure and energy change
> across Miller-index surfaces?

## Scope To Add

Add one material-specific Miller-index search to the tutorial. Default proposal:
rutile TiO2 because the notebook already has programmatic TiO2(110) support and
oxide surfaces make the search chemically richer than another pure metal.

Candidate material grid:

- material: rutile TiO2
- Miller indices: `(110)`, `(100)`, `(101)`, `(001)` if slab generation and
  surface sizes are stable
- adsorbates: H2O first, then CO and CH3OH if runtime remains reasonable
- starts per surface: representative cation top, oxygen top, bridge, hollow
- orientations: chemically meaningful subset, not every rotation by default
- heights: one default height first, optional height sensitivity after
  convergence/timing is understood

The tutorial question becomes:

> Once we choose a material, batching lets us search over exposed crystal faces,
> adsorption sites, and molecular starts instead of committing to one
> hand-picked slab and one hand-picked initial geometry.

## Implementation Plan

1. Add a Miller-index panel builder that takes a material name and list of
   Miller indices, then generates slabs programmatically with pymatgen/ASE
   helpers. No token-written XYZ structures.
2. Validate each generated slab: composition, vacuum, tags/frozen layers,
   surface atom classes, site count, and atom count.
3. Build a compact teaching grid for TiO2/H2O across Miller indices.
4. Run batch-size and step-count timing on the actual TiO2 Miller grid, not
   only on H2O gas or one surface.
5. Save all initial/final structures, full Toolkit trajectories, and
   energy/force logs for every generated start.
6. Add a notebook section that ranks by adsorption energy within the same
   model/reference convention and shows which Miller index/site/orientation
   produced the best relaxed structure.
7. Add a visual-review table with a small number of curated trajectory paths:
   easy convergence, difficult/rerun convergence, and cross-Miller-index
   winners.
8. Keep the current OC20Dense validation as the model/tooling check, but make
   the Miller-index sweep the tutorial's discovery-style example.

## Reporting Rule

Any runtime summary must state:

- whether it is full-pipeline, validation-only, notebook-only, or panel-only;
- number of structures;
- number of surfaces and Miller indices;
- chunk size;
- step cap and rerun cap;
- convergence count;
- whether trajectories/logs were written.
