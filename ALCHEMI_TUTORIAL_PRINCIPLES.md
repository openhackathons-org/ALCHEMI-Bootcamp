# ALCHEMI tutorial principles and visual style

This guide applies
[TUTORIAL_DESIGN_PRINCIPLES.md](TUTORIAL_DESIGN_PRINCIPLES.md) to ALCHEMI
Toolkit tutorials. It covers curriculum choices, public API exposure, live
compute, scientific care, and the shared notebook style.

ALCHEMI is the main subject. Scientific examples make Toolkit behavior visible
and useful, but these tutorials are not intended to teach a full scientific
field.

## 1. Plan broad Toolkit coverage across the series

Start with the reusable Toolkit actions learners should be able to perform.
Across the tutorial series, cover the important capability groups:

- represent structures with `AtomicData` and `Batch`;
- inspect system identity and recover per-system results;
- choose and build neighbor data;
- configure models and request supported outputs;
- compose independent contributions or connect dependent model stages;
- run relaxation or dynamics with hooks;
- compare serial, batched, inflight, and distributed execution;
- adapt an external model through Toolkit's public model interface; and
- save, reload, and analyze results.

The separate [Toolkit API curriculum](TOOLKIT_API_CURRICULUM.md) contains the
exact API inventory. This guide defines how to teach it.

A single notebook should not contain every capability. For each part, name:

- primary capabilities taught in depth;
- earlier capabilities reused for practice;
- new capabilities introduced briefly; and
- capabilities deferred to another part.

Choose scientific examples to serve that coverage. Keep an example while it
helps comparisons, and switch when another model, system, or workload is needed
to teach an important Toolkit feature. State why the example changes and which
Toolkit concepts carry over.

## 2. Establish the common Toolkit Core path early

Most Core tutorials begin with this neutral path before introducing optional
branches:

```text
structure -> AtomicData -> Batch -> chosen Toolkit operation -> per-system results
```

The chosen operation may call a model, build neighbors, run dynamics, compose
models, process an inflight queue, or execute distributed stages. Do not imply
that every workflow requires all of those branches.

A tutorial focused directly on Toolkit-Ops may instead begin with tensors and
one public kernel call. It should still produce an inspectable result early and
explain how the operation fits into a Core workflow.

Name the owner of each part:

- ASE or pymatgen creates and manipulates source structures.
- Toolkit Core supplies the shared data, model, pipeline, simulation, and
  `DistributedPipeline` APIs.
- Toolkit-Ops supplies accelerated neighbor, interaction, and segmented
  operations.
- PyTorch supplies tensors, automatic differentiation, compilation, and the
  communication backend used by distributed execution.
- External packages supply learned models and checkpoints.
- Tutorial helpers supply preparation, analysis, and presentation code.

Assume Python plus computational chemistry or machine-learning experience, but
no prior ALCHEMI knowledge. Define a Toolkit term at first use, then show it in
code.

The opening should produce one real Toolkit result quickly. In a Core lesson,
construct or load a structure, convert it, make a one-system batch, run a
documented operation, and inspect a named output with its units. Give only the
scientific background needed to judge that result.

## 3. Keep reusable Toolkit APIs visible

For APIs used in a tutorial, keep the reusable product path in learner code.

| Keep visible | Move to focused helpers |
|---|---|
| `AtomicData` conversion and important fields | structure-generation boilerplate |
| `Batch` construction, membership, selection, and unpacking | downloads and large input tables |
| model adapter and `model_config` choices | checkpoint location and checksum code |
| requested outputs and neighbor settings | repetitive table construction |
| neighbor calls or model neighbor hooks | plot implementation and presentation HTML |
| model composition and dependent wiring | signal processing and reference parsing |
| relaxation, dynamics, and hook registration | repeated format conversion |
| inflight or distributed pipeline construction | assembly of recorded benchmark results |
| Toolkit save and reload calls | display-only formatting |

Helpers must not become a second tutorial-only workflow API. Use documented
public APIs. Do not patch objects at runtime or replace the intended Toolkit
execution API with a tutorial-only loop.

## 4. Explain model support and model changes

When a model is first loaded, show a compact model card with:

- model and checkpoint name;
- code and weight source and licenses;
- training target and demonstrated application domain;
- supported elements, charge, spin, cell, and periodic-boundary behavior;
- dtype and neighbor requirements;
- available and requested outputs; and
- whether dispersion or long-range electrostatics is already represented, or
  that the available documentation does not establish this.

Model capability is not model accuracy. Periodic support alone does not show
that a checkpoint is accurate for surfaces.

If a new example needs another model, state why the earlier model is unsuitable,
what the new model supports, and which Toolkit interface remains unchanged. A
model switch should teach a meaningful domain or integration difference, not
repeat the same adapter with another checkpoint.

### Composition

Before adding a model contribution, answer:

1. What target did the base model learn?
2. Is the contribution already represented in that target?
3. Which boundary condition and parameter convention does it use?
4. Which independent comparison checks the complete result?

Add independent contributions directly. Use explicit pipeline wiring when one
stage produces an input for another stage and derivatives must follow that
connection. Do not double count dispersion or electrostatics.

An ablation shows what changes when a component is omitted. It is not
automatically a sequence of increasingly accurate methods. Compare compatible
geometries, quantities, units, and energy definitions. Keep electronic-
structure, finite-temperature simulation, and experimental comparisons
separate when they do not share an observable or intensity scale.

### Custom model adapters

Use a custom adapter to answer a real product question: how can a learner use a
model for which Toolkit has no built-in adapter?

Show the smallest complete path:

1. declare supported inputs, outputs, precision, periodicity, and neighbors;
2. map `Batch` fields to the model input;
3. call the external model;
4. map its results to Toolkit names and shapes;
5. use the adapter in a normal Toolkit workflow; and
6. compare its outputs with the model's native call.

Keep the class interface and important mapping visible. Move long conversions
and presentation code into helpers. Request only outputs the model supplies.

## 5. Teach data identity and execution modes precisely

A `Batch` holds the systems active in one operation. A workflow may reuse or
rebuild it across many calls. Before timing batched work, compare it with the
equivalent individual calls and verify agreement.

Show:

- how atoms map to systems;
- how variable-sized systems share a batch;
- how results map back to the source systems through order, graph membership,
  labels, or explicit IDs;
- whether each output is per atom or per system; and
- how to select and recover an individual result.

Homogeneous and heterogeneous batches represent the same scientific work in
different layouts. Homogeneous buckets can reduce shape variation or capacity
waste. Heterogeneous batches can reduce calls and keep a varied queue moving.
Measure both with the same structures, and report structures per second plus
atoms per second when sizes vary.

Keep these execution concepts separate:

- `FusedStage` runs compatible dynamics methods on one device while sharing one
  active `Batch` and one model evaluation per step. Each system's `status`
  selects the method that updates it, so systems can advance at different
  times. It is not the same as completing one full stage before starting the
  next, and hook timing differs because the model call is shared. It takes the
  model from the first sub-stage, so `+` is not a model-switching mechanism.
- Inflight execution uses `SizeAwareSampler` and bounded active batches. As
  systems finish, they leave and queued systems enter, subject to whichever
  graph, atom, or edge limits apply.
- A distributed pipeline, written as `stage_a | stage_b`, places stages on
  different workers and transfers data between them through
  `DistributedPipeline`.
- Domain-parallel execution in `nvalchemi.distributed` partitions one large
  spatial system across devices. It is separate from `DistributedPipeline`,
  which streams batches through stages and does not automatically split one
  model evaluation across GPUs.

For inflight work, show the input queue, active limits, refill events, stable
IDs, completion, and failure handling. For distributed work, show stage
ownership, transferred data, worker batch size, global concurrency, setup,
fill, steady work, drain, and result collection.

Use the public API with a small live example. When learners cannot request
several GPUs, provide recorded multi-GPU results for analysis and identify the
hardware, software, workload, warm-up, repetitions, and timing boundaries.

## 6. Measure and interpret Toolkit behavior

For every requested output, show:

- `active_outputs` or the equivalent declared output set;
- its shape and whether it is per atom or per system;
- its units;
- how it maps back to the source system; and
- the check or interpretation required by the lesson.

For performance comparisons, keep structures, model, dtype, outputs, and
neighbor settings fixed. Time neighbor construction, compilation, transfers,
and result collection separately when they matter. Report response time,
structures per second, and atoms per second as appropriate. Preserve failed
measurements and cases where the simpler approach wins. Record repetitions,
summary statistic, spread, hardware, and relevant software versions.

Save enough information to recover per-system results and recreate the main
figures without repeating the longest calculation.

## 7. Keep helper code small and maintainable

Organize helpers by one clear responsibility, such as structures, model
conversion, analysis, plotting, persistence, benchmarking, or notebook UI.

- Give public helper functions type hints and docstrings.
- Test scientific transformations and saved-file round trips.
- Keep package re-exports minimal so `aux` does not resemble Toolkit itself.
- Keep behavior-changing choices in visible notebook configuration.
- Do not duplicate hidden defaults across cells and helpers.
- Test the exact adapter or helper excerpt shown to learners.

## 8. Use the tutorial visual system

Use the tutorial UI and plotting helpers. Do not hand-code one-off notebook HTML
for banners, cards, callouts, tables, figures, or progress.

### Opening summary and hero

- Use the 2880 x 1440 hero source, rendered at 880 px wide.
- Use a dark background, title, short subtitle, and part badge over a dark
  left-to-right gradient.
- Use NVIDIA green `#76B900` as an accent, not body text.
- Add alternative text that describes the image content.
- Place the `DO / LEARN / NEED` summary cards immediately after the hero.

### Stage and progress cards

Each numbered stage begins with one stage card containing `STAGE n OF N`, a
short title that names the task or idea, one concrete outcome, the neutral
`STAGE` label, and the 6 px green position rail. Opening setup or notebook-map
sections do not need a stage number.

Every executable cell displays one progress card. Imports or setup required to
construct the card may appear first. Use real units such as files, structures,
model calls, optimizer steps, or dynamics steps. Use the states `READY`,
`RUNNING`, `COMPLETE`, and `ACTION NEEDED` consistently, show elapsed time, and
preserve the final state in the executed notebook.

Stage position is not execution progress. Do not leave a static stage card in a
misleading running state. If work completes before a maximum step count, report
steps used and the limit, while showing the work as complete.

### Callouts

| Label | Use |
|---|---|
| `BEFORE YOU RUN` | prediction or input to inspect before execution |
| `WHAT TO CHECK` | expected relationship, trend, or numerical check |
| `NOTE` | limitation or distinction that affects interpretation |
| `RESULT` | direct observation from the preceding output |
| `CHECK PASSED` | a named check that passed |
| `NOT REPORTED` | result intentionally not claimed, with the reason |
| `NEEDS ATTENTION` | failed check or required action |

Use familiar labels and ordinary technical language throughout.

### Plots, tables, and headings

- Use blue `#276FBF` for live simulation, orange `#D95F02` for
  electronic-structure references, green `#4D7C0F` for experiment, text
  `#111827`, and grid `#E5E7EB`.
- Add line styles and markers so meaning does not depend on color.
- Put units in axes and table headings.
- Show complete values and visible failure reasons.
- Use explicit missing-value text instead of an unexplained `NaN`.
- Give each figure useful alternative text plus a nearby caption or
  interpretation that answers a learner question.
- Use diagrams only when they clarify ownership, data flow, or execution order.
- Use short, task-led headings and direct sentences. Do not restate code line by
  line.

## Review checklist

Always check:

- ALCHEMI is the primary learning subject.
- The part's primary, practised, introduced, and deferred capabilities are clear.
- The first public Toolkit result appears near the beginning.
- Scientific detail is sufficient for safe interpretation but does not become
  the main lesson.
- Important Toolkit APIs and result-affecting choices remain visible.
- Every requested output is shown with shape, level, units, identity, and an
  interpretation.
- Every executable cell uses the shared progress card.
- Cards, callouts, plots, tables, headings, and colors use the shared helpers.
- The notebook runs from top to bottom in a fresh session.
- Software, checkpoint, dataset, and figure licenses are checked separately.

When the topic is taught, also check:

- individual and batched outputs agree before performance is discussed;
- model composition does not double count contributions;
- inflight and distributed execution preserve system identity and failures;
- benchmarks use equal work and honest timing; and
- a custom adapter agrees with the external model's native call.
