# Shared Adsorption Tutorial Contract

This directory records the cross-notebook contract for keeping a BGR NIM
version and a toolkit version of the adsorption tutorial synchronized.

Current state:

- Part 1 implements the AdsorbML-style adsorption workflow through the BGR NIM.
- Part 2 is not yet the full toolkit counterpart of Part 1; it currently
  contains a minimal toolkit sandbox and a separate melting-point SLC notebook.
- `contract.py` is the canonical Python representation of the active panel and
  required result schema.
- `panel.yml` is the human-readable equivalent for docs and cross-language
  tooling.
- `backends.md` defines the BGR/toolkit adapter boundary.

The intended sync rule is simple: the scientific panel, reference manifest,
geometry checks, and analysis definitions must be shared before a toolkit
version is presented as equivalent to the BGR NIM version.

## Required Shared Surfaces

- Cu(111)
- Pd(111)
- alpha-Al2O3(0001)

## Required Shared Adsorbates

- CO
- H2O
- CH3OH

## Optional Context Adsorbate

- NH3, used only for first-binding-context examples unless promoted into the active panel with references.

## Required Shared Checks

- Miller/facet verification.
- Layer count and vacuum verification.
- Active-mask/frozen-layer verification.
- Adsorption-site classification.
- Adsorption-energy sign convention.
- Reference-scope guard: `context` rows do not enter strict parity statistics.

## Backend Boundary

The shared science should not depend on the execution backend.

- NIM backend: HTTP BGR requests and cached JSON replay.
- Toolkit backend: local calculator/dynamics APIs.

Both backends should emit the same analysis schema:

- backend
- host
- adsorbate
- label
- starting site
- starting orientation
- relaxed final site
- adsorption energy in eV
- convergence flag
- maximum force in eV/A
- geometry status
- reliable-for-minimum flag
- reference scope
- validation status

## Bootcamp Structure

OpenHackathons-style bootcamp repositories keep the learning path explicit:
top-level README, deploy/container files, and notebooks that can be run inside a
container or cluster allocation. This repo follows that pattern with separate
backend folders and one shared science contract:

- `part-1-nim/`: BGR NIM implementation and cached-response replay.
- `part-2-toolkit/`: toolkit implementation space.
- `shared/adsorption_tutorial/`: reusable scientific contract and reference
  review packet.
