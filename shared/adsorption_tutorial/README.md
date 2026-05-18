# Shared Adsorption Tutorial Contract

This directory records the scientific contract for the Toolkit teaching path.

Current state:

- Part 1 implements the AdsorbML-style adsorption workflow through the
  ALCHEMI Toolkit.
- Part 2 currently contains a general Toolkit sandbox and a separate
  melting-point SLC notebook; it is not the active adsorption implementation.
- `contract.py` is the canonical Python representation of the active panel and
  required result schema.
- `panel.yml` is the human-readable equivalent for docs and cross-language
  tooling.
- `backends.md` is retained as historical adapter context; the current tutorial
  notebook is Toolkit-only.

The intended sync rule is simple: the scientific panel, reference manifest,
geometry checks, and analysis definitions must be shared before a service/API
version is presented as equivalent to the Toolkit teaching version.

## Required Shared Surfaces

- Cu(111)
- Cu(100)
- Cu(110)
- rutile TiO2(110)
- rutile TiO2(100)
- rutile TiO2(101)
- TiN(001)
- TiN(110)
- TiN(210)

## Required Shared Adsorbates

- CO
- H2O
- NH3
- CH3OH

## Benchmark-Only Adsorbates

- N2 appears in the OC20Dense closed-shell verification slice.
- NH3 is now part of the active teaching panel.

## Required Shared Checks

- Miller/facet verification.
- Layer count and vacuum verification.
- Active-mask/frozen-layer verification.
- Adsorption-site classification.
- Adsorption-energy sign convention.
- Reference-scope guard: `context` rows do not enter strict parity statistics.

## Execution Boundary

The shared science should not depend on incidental implementation details.

- Toolkit backend: native calculator/dynamics APIs on the GPU.

The current notebook emits the shared analysis schema:

- backend
- host
- adsorbate
- label
- starting site
- starting orientation
- relaxed final site
- `E_ads` in eV
- convergence flag
- maximum force in eV/A
- geometry status
- reliable-for-minimum flag
- reference scope
- validation status

## Bootcamp Structure

OpenHackathons-style bootcamp repositories keep the learning path explicit:
top-level README, environment/setup notes, and notebooks that can be run inside
a configured GPU environment or cluster allocation. This repo follows that
pattern with separate tutorial folders and one shared science contract:

- `part-1-batched-adsorption/`: Toolkit adsorption-search tutorial.
- `part-2-toolkit/`: general Toolkit sandbox and related case-study material.
- `shared/adsorption_tutorial/`: reusable scientific contract and reference
  review packet.
