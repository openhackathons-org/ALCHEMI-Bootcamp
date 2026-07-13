# Internal support modules

`aux/` contains implementation details that would distract from the tutorial's
Toolkit data flow. It is deliberately **not** a second user-facing API:
`aux/__init__.py` re-exports nothing, and notebook cells import only the focused
functions they need.

The notebook keeps these public Toolkit operations visible:

- `AtomicData.from_atoms(...)` and `Batch.from_data_list(...)`;
- neighbor construction and the resulting batch fields;
- model wrappers and model configuration;
- `PipelineGroup` and `PipelineModelWrapper` composition;
- relaxation, fused dynamics, and hook registration.

The internal modules have narrow responsibilities:

- `structures.py` — deterministic ASE structures and isotope masses; it stops
  before conversion to Toolkit data;
- `electrostatics.py` — the custom finite-system, no-cutoff all-pairs Coulomb
  model wrapper;
- `checkpoint.py` — AIMNet checkpoint resolution and metadata validation;
- `capture.py` — GPU-resident predicted-charge trajectory recording;
- `diagnostics.py`, `topology.py`, and `spectra.py` — scientific checks and
  signal processing;
- `analysis.py` and `plotting.py` — result tables plus the shared figure size,
  isotope typography, palette, axes, and publication-ready plot builders;
- `benchmarking.py` — timing harnesses around notebook-constructed batches;
- `hooks.py` — notebook progress rendering adapted to Toolkit's public hook
  lifecycle; it never changes simulation state;
- `artifacts.py` — trajectory/structure serialization and deterministic run
  manifests;
- `reference_data.py` — exact live-geometry verification for the B97-3c dimer
  CSV before it enters a comparison plot;
- `reference/` — dependency-light readers and mode analysis for the immutable
  B97-3c reference bundles;
- `runtime.py` — environment checks;
- `ui.py` — the single 880 px notebook presentation system: accessible hero,
  lesson summary, semantic stage headings, callouts, visual-review slots, and
  live progress cards. It contains no Toolkit or scientific computation.

No module in `aux/` constructs the learner-facing Toolkit batch or hides the
potential pipeline assembled in the notebook.

Two artifact-boundary APIs are intentionally narrow:

- `write_water_run_manifest(output_dir, provenance=..., settings=...,
  gates=...)` writes `water_run_manifest.json`. It JSON-normalizes NumPy and
  path values and records the byte size and SHA-256 of every other file in the
  run-specific output directory, including nested store chunks, in
  deterministic relative-path order.
- `load_verified_b97_3c_dimer_reference(csv_path)` returns a pandas
  `DataFrame` only after authenticating the sibling `SHA256SUMS` and
  `manifest.json`, matching recorded source hashes to the live canonical
  driver/builder, and checking the ordered requested/measured O--O grid plus
  all AB/A/B geometry hashes against rebuilt structures. This proves bundle
  integrity and geometry identity; it does not recompute or independently
  validate the B97-3c energies.
