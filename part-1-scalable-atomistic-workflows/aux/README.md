# Part 1 support code

`aux/` contains tested support code that would interrupt the notebook's Toolkit
learning path. It is not a second user-facing API. The package `__init__.py`
does not re-export helpers, and notebook cells import only the focused functions
they need.

## What remains visible in the notebook

The notebook directly shows:

- `AtomicData` and `Batch` construction;
- model adapters and model configuration;
- requested outputs and neighbor construction;
- model composition and dependent data flow;
- relaxation, dynamics, and hook registration;
- Packmol settings and the conversion of its periodic box to one Toolkit graph;
- `PMEModelWrapper`, `DomainConfig`, and `DomainParallel` construction;
- the public partition, run, and gather sequence for multi-GPU domain execution;
- inflight and distributed pipeline construction;
- model calls, saved-result locations, and the checks used to interpret them.

Support code must not hide or rename these decisions.

## Module groups

| Group | Files | Responsibility |
|---|---|---|
| Structures | `structures.py`, `adsorption.py`, `adsorption_visualization.py` | Deterministic ASE structures, isotope variants, surface starting geometries, and OVITO display helpers |
| Models | `checkpoint.py`, `electrostatics.py`, `models/` | Checkpoint verification, model-specific tensor conversion, custom adapters, and native-versus-adapter checks |
| Configuration | `composition_config.py`, `nci_config.py`, `harmonic_config.py`, `workflow_config.py` | Shared named settings used by the notebook and validators |
| Interaction references | `nci_atlas.py`, `nci_plotting.py`, `reference_data.py` | Checksummed reference loading, ASE structure conversion, AB/A/B reductions, metrics, and shared curve plotting |
| Numerical checks | `composition_checks.py`, `nci_validation.py`, `numerical_checks.py`, `precision.py`, `diagnostics.py`, `topology.py` | Small calculations and tables that test agreement, precision, invariants, temperature, energy, and structure |
| Harmonic and spectral analysis | `harmonic_ir.py`, `harmonic_workflow.py`, `spectra.py`, `analysis.py` | Finite differences, modes, dipole derivatives, spectrum processing, and comparison tables |
| Batching campaigns | `inflight.py`, `pipeline_campaign_results.py`, `pipeline_campaign_view.py`, `benchmarking.py` | Inflight preparation, strict recorded-result loading, benchmark summaries, and failure retention |
| Single-system scaling | `domain/config.py`, `domain/packing.py`, `domain/results.py` | One versioned settings source, NCI-derived Packmol box construction, structure checks, and strict saved H100 domain-decomposition result views |
| Recording and files | `capture.py`, `artifacts.py`, `run_output.py`, `reference_data.py`, `experimental_reference.py`, `reference/` | Trajectory capture, saved tables and arrays, checksums, reference loading, and replay |
| Notebook presentation | `framework_comparison.py`, `ui.py`, `plotting.py`, `results_summary.py` | Small comparison tables, shared cards, callouts, figures, colors, and final result summaries |
| Runtime integration | `runtime.py`, `hooks.py` | Environment checks and the small tutorial progress hook |

## Helper design rules

- Give each module one clear responsibility.
- Keep behavior-changing settings in visible notebook configuration or one
  documented configuration module.
- Pass settings into helpers; do not repeat hidden defaults.
- Keep model and Toolkit calls visible unless the helper itself is the concept
  being tested.
- Keep AIMNet2-to-PME data flow, the independent D3 branch, `DomainConfig`, and
  the `DomainParallel` call sequence visible. The `domain/` helpers may prepare and
  check Packmol structures or read saved results, but must not hide those calls.
- Call the Packmol input value a construction density. It is not an equilibrated
  or predicted density.
- Return ordinary values, tables, figures, or Toolkit objects. Do not invent a
  parallel workflow interface.
- Preserve per-system identity and complete model outputs.
- Validate inputs and fail with a direct message.
- Do not use runtime patches, private Toolkit fields, or global state changes.
- Add type hints and docstrings to maintained public helper functions.
- Test scientific transforms, model mappings, and saved-file round trips.

## Custom model adapters

The maintained adapter source lives in `models/`. The notebook displays the
small Toolkit-facing class and explains:

- declared inputs, outputs, precision, periodicity, and neighbors;
- mapping from `Batch` fields to the native model;
- one native model call;
- mapping of energy and forces back to Toolkit names.

Long tensor validation and graph assembly may remain in helper functions. The
notebook must still compare the Toolkit adapter with the model's native path
before using it in a workflow.

## Notebook presentation

`ui.py` is the single implementation of the 880 px notebook presentation
system. It provides the hero, stage cards, progress cards, callouts, complete
tables, process diagrams, and accessible figure embedding. It contains no
scientific calculation or Toolkit workflow logic.

`plotting.py` owns the shared plot palette, line styles, grid, and figure
layout. Plot functions return figure objects; the notebook decides when to
display and save them.

The complete visual and writing rules are in
[ALCHEMI_TUTORIAL_PRINCIPLES.md](../../ALCHEMI_TUTORIAL_PRINCIPLES.md).
