# ALCHEMI tutorial guide

This is the single source of truth for the teaching, writing, visual design,
scientific care, and review of this tutorial series. The
[Toolkit API reference](TOOLKIT_API_REFERENCE.md) owns exact public names,
signatures, shapes, and release-specific behavior.

Other files have narrower jobs:

- [`shared/README.md`](shared/README.md) documents shared assets and helper
  implementations.
- [`environment/README.md`](environment/README.md) documents the frozen runtime.
- [`WORKLOG.md`](WORKLOG.md) and notebook worklogs record current work, measured
  results, and local decisions.

Keep all tutorial policy in this document. Add a durable rule here, an API fact
to the technical reference, or a notebook-specific decision to its worklog.

## 1. Purpose and audience

ALCHEMI is a collection of tools for building atomistic simulation software.
The tutorials teach scientists and developers how to combine those tools into
their own data, model, and simulation workflows.

The public API is the subject. Chemistry gives each API a concrete input,
output, and scientific check. Application results support that API lesson.

Assume an audience that knows Python and basic computational chemistry or
materials science. Define Toolkit terms when they first appear. Explain the
reason for an abstraction through the operation it enables.

Use sources in this order:

1. pinned Toolkit and Toolkit-Ops documentation and examples;
2. the public API and source installed in the frozen course environment;
3. official upstream documentation for ASE, pymatgen, PyTorch, JAX, Warp, and
   other surrounding projects; and
4. measured notebook output from a fresh kernel.

Engineering examples establish the intended path. Source inspection resolves
release-specific details when prose and implementation differ.

### ALCHEMI product tracks

- **Toolkit** supplies reusable Python objects and interfaces for atomic data,
  batches, models, hooks, dynamics, loading, training, and distributed work.
- **Toolkit-Ops** supplies accelerated atomistic operations through framework
  bindings. State which binding a lesson uses and verify the implementation
  path before describing it.
- **NIM** packages supported domain workflows behind service APIs. NIM lessons
  teach requests, responses, configuration, deployment, and observability.
- **Surrounding scientific code** connects through documented public extension
  points such as readers, model adapters, hooks, and workflow classes.

Each lesson follows one product track closely enough that learners can reuse
its public calls in their own project.

## 2. Course structure

The core sequence introduces one reusable idea at a time.

Before drafting a part, divide its content into three groups:

1. **essential** capabilities that produce the lesson outcome;
2. **supporting** concepts needed to understand that path; and
3. **advanced** capabilities owned by a later or optional lesson.

Track each capability across the course as **introduced**, **practised**, or
**extended**. A later notebook should name the earlier capability it reuses and
the new operation it adds.

Treat product-gallery examples as source material, not as a ready-made lesson
sequence. A gallery often demonstrates the complete API in one page. Keep an
operation in the live lesson when its output helps learners understand the
current object or prepares the next operation. Link the remaining API breadth
as a reference for later exploration.

| Part | Focus | Learner outcome |
|---|---|---|
| 01 | `AtomicData` and `Batch` | Convert structures, validate fields, batch unequal systems, and recover per-system data. |
| 02 | Data loading with Zarr | Read individual records and stream batches through the supported loader path. |
| 03 | Model interfaces and composition | Wrap a model, declare runtime metadata, and combine dependent and independent terms. |
| 04 | Hooks | Add observation and control at named workflow stages. |
| 05 | `BaseDynamics` | Build optimization, molecular dynamics, and screening workflows from a shared execution abstraction. |
| 06 | GPU pipelines and profiling | Connect stages on one GPU, compile supported work, and measure the complete execution path. |
| 07 | Training and fine-tuning | Configure reproducible training and extend it through public strategies and hooks. |
| 08 | Domain decomposition | Prepare a model for spatial work and run one large system across multiple GPUs. |

Parts 01–05 form the fundamentals course. Parts 06–08 are advanced Toolkit
lessons. End-application notebooks may combine several parts after the API
foundations are established.

Part 01 introduces ALCHEMI and the course. Later notebooks use a short folded
orientation, link the exact earlier lesson they reuse, and state the one new
capability they add. Every notebook rebuilds its inputs from a fresh kernel.

The shared course map is an orientation graphic. It uses a compact vertical
course spine and six larger capability groups. The current part and its primary
capability use NVIDIA green. Course-map copy stays short enough to fit
fixed-size boxes. Each lesson-to-capability relationship has its own rounded
route and arrowhead. Capability arrows show real dependencies: fundamentals
support every downstream path; data management feeds model use, simulation,
and model development; models and simulation feed multi-GPU execution.
Notebook navigation sits in ordinary Markdown links below the map.
Keep documentation links visually prominent and repeat them as ordinary
Markdown below the map. Show unpublished course notebooks as muted,
non-clickable status text until they are ready for learners.
Capability cards use one vertical gap. Their height grows by one text-line step
for each additional body line; heading, body, line spacing, and bottom padding
stay aligned across the column.
Use the generated SVG assets described in [`shared/README.md`](shared/README.md)
so all notebooks show the same geometry, connectors, icons, and typography.
Render the SVG with Markdown image syntax or an HTML `img`, both of which work
reliably in VS Code and exported notebooks. Use an interactive `object` only
after checking it in every target renderer.
For hover navigation in a notebook, keep the diagram as the visual layer and
place ordinary HTML controls above it. SVG anchors can be treated as image
metadata by notebook renderers. Let mouse hover close on pointer exit; reserve
persistent focus states for keyboard navigation.

## 3. Lesson design

Start with a useful result or a clear scientific question. Then rebuild the
result from the smallest familiar object and introduce each Toolkit abstraction
when it becomes necessary.

A strong lesson usually follows this cycle:

1. show the question or useful result;
2. connect it to an object the learner already knows;
3. introduce one new public object or operation;
4. run it and inspect the smallest useful output;
5. explain the shape, ownership, units, or lifecycle that the output reveals;
6. inspect or modify one meaningful property to learn how the object behaves;
7. reuse the same operation at a larger scale or in a new composition;
8. give a bounded exercise with a visible success signal; and
9. recap the capability and where the course uses it next.

A short exercise should change an input to the public API being taught and
inspect that API's result. Supporting libraries may prepare the input. Keep the
learner's action centered on the product object.

When an API returns new values, show how those values enter reusable workflow
state. Name whether the call updates its input or returns a separate result,
keep the public registration call visible, and inspect the relevant fields
before and after. This is especially important for model outputs, hooks, and
dynamics state.

Change one important condition at a time. Keep the input fixed when comparing
execution strategies, the operation fixed when comparing data layouts, the
interface fixed when changing models, and the workload fixed when measuring an
optimization. When a new capability needs a different scientific example,
explain why the example changed and which product concepts remain the same.

Close each notebook with a short **Recap**:

- **Core concepts** states the objects, operations, and mental model the
  learner can now use.
- **How we will use this** names the later data, model, or workflow lessons that
  carry those objects forward.

Group exact API names under those outcomes. A recap should read as a course
handoff, with a link to the next notebook.

An opening performance result may use one tested helper to keep the pitch to a
single code cell and a polished output. Rebuild every central public operation
visibly later. State the compared work, timed boundary, batch size, device,
hardware, warm-up, repetitions, and spread. Labels such as “sequential
one-system calls” and “one batched call” should describe the actual operations.

Teach a public API completely enough to reuse:

- name the object or function;
- show the public import and call;
- explain the input object and required fields;
- inspect the returned or updated object;
- state shapes, units, device, and ownership where they matter;
- show configuration through the supported public method;
- demonstrate one documented validation boundary or failure when validation is
  a lesson outcome; and
- show the next operation the result enables.

Build validation examples from a realistic construction step, reader boundary,
field update, or documented invariant. Claim only the invariant that the
demonstrated validator checks.

Choose fields and edits that have a clear scientific or software meaning. A
system identifier, atom mask, energy, force, or temperature teaches ownership.
Arbitrary placeholder values teach only the method signature.

### Stable data language

Use this ladder consistently:

1. a **structure** or **molecule** is the familiar scientific object;
2. `AtomicData` stores one structure as a validated atomic graph;
3. atoms are graph nodes and per-atom fields are node data;
4. neighbor construction supplies the interaction edges required by a model;
5. `Batch` packs several `AtomicData` graphs into one graph-aware tensor
   container; and
6. `batch_idx` and `batch_ptr` describe graph membership and boundaries inside
   that packed container.

Use “molecule” or “structure” while discussing chemistry. Use “graph” while
explaining `AtomicData`, field levels, batching, neighbors, or model inputs.
Reserve “identity” for an explicit system ID, label, or metadata mapping that
the lesson carries and checks.

Toolkit provides public conversions from ASE `Atoms` and, when the optional
dependency is installed, pymatgen `Structure` and `Molecule`. File formats are
read by ASE or pymatgen before conversion. Teach a hand-built structure only
when constructing it reveals fields or invariants needed by the lesson.

When a library bridge is the lesson, start from an object built or read through
that library's documented path. Show direct tensor construction beside the
bridge only when the comparison explains which fields the converter carries,
validates, or places on the target device.

## 4. Notebook construction

Each code cell should perform one observable action. Most learner cells should
contain 1–5 lines. Keep visible code at 20 lines or fewer unless the learner is
writing a complete adapter, reader, hook, or workflow class whose shape is the
lesson.

Separate these actions into separate cells:

- computation;
- result shaping;
- DataFrame display;
- plotting; and
- interpretation in Markdown.

Keep imports and setup in one small collapsed cell. Hide path discovery,
checksums, warning configuration, repeated synchronization, repeated timing,
presentation layout, and asset loading in tested notebook-local helpers. Keep
public construction, configuration, execution, inspection, selection, and
recovery calls visible.

Name helpers for their work: `presentation`, `benchmark`, `data`, `models`, or
`validation`. Choose task names in place of names such as `lesson`,
`start_lesson`, or `primer`.

Use short inline comments only when they explain a scientific choice, tensor
level, unit conversion, framework boundary, or non-obvious API constraint.

Once `AtomicData` and `Batch` move to the lesson device, keep the main Toolkit
path on that device. CPU work belongs in input preparation or a clearly named
CPU/GPU benchmark.

Optional material must be a complete mini-lesson with a question, executable
work, inspection, and conclusion. Give it a descriptive title such as
“Optional: trusted-data loading.” Include the full sequence or leave the topic
for its owning lesson.

## 5. Visual teaching system

Visuals remain part of the teaching. Every visual answers a learner question,
shows one relationship, and is followed by a one-sentence takeaway.

Use the shared assets and implementation patterns in
[`shared/README.md`](shared/README.md):

- the ALCHEMI banner at the top of each notebook;
- a compact ALCHEMI ecosystem explanation and links in the first course
  introduction;
- the generated course-map SVG;
- MatterViz for interactive molecular or material structures;
- plain Pandas for tables; and
- plain Matplotlib with `shared/alchemi-dark.mplstyle` for quantitative plots.

An ecosystem graphic is optional. Include it when the course creator requests
it. Treat its removal by an educator as a design decision and update any checks
that expected the asset.

Use a light, compact molecular thumbnail when several molecules need visual
orientation. Use MatterViz when rotation, depth, periodicity, or atom selection
supports the lesson. Keep the viewer focused on the structure; move embedding,
HTML, captions, and fallbacks into a presentation helper.

Use NVIDIA green for the primary Toolkit result. Use muted neutral colors for
context and one restrained comparison color when needed. Plots include units,
readable labels, and a short question immediately before the figure.

### Diagrams

Use generated SVG for shared maps or architecture diagrams that need fixed box
sizes, aligned icons, and controlled routing. Mermaid suits compact local flows
when it renders consistently in the target notebook and export. Check that
render before keeping the source. Use a local SVG when Mermaid layout or export
changes the meaning or breaks the figure.

Flowcharts use:

- a top-to-bottom reading order when the concept is sequential;
- solid fills to establish hierarchy;
- NVIDIA green fill for the current path;
- charcoal or warm-neutral fills for other nodes;
- NVIDIA Sans with Arial and system fallbacks;
- equal-sized rounded boxes when nodes have the same role;
- quiet borders and muted connectors;
- clear arrowheads with consistent direction;
- angled or gently rounded connector routes; and
- short edge labels that state the relationship.

Use flat fills, restrained borders, one consistent icon family, and edge labels
that state the relationship. Keep gradients, shadows, badges, and emoji outside
technical diagrams.

Course-map icons use one quiet line weight, one simple metaphor, and the fewest
marks needed for recognition. They support the lesson title. Size them for
recognition at notebook width and check that their main marks remain distinct
after export. Capability connectors keep the overview compact: one independent
route per relationship, including each branch from a lesson that enables two
capabilities. Detailed API dependencies belong in the notebook that teaches
them. Route around collisions and do not merge relationships into shared rails.

### Callouts

Use two callouts:

- **Highlight** for one idea that deserves a pause. It uses a warm, neutral
  surface and must fit both light and dark notebook themes.
- **ALCHEMI Toolkit API** for a public call learners should reuse. It uses the
  NVIDIA signature treatment, shows the exact name, input, and result, and fits
  inside the notebook content width.

Each API card shows one signature. Put related constructors, conversion
methods, and variants in ordinary Markdown or a compact reference table.

Write results, notes, limitations, transitions, and exercises as ordinary
Markdown. Use Rich progress only for work with a visible wait.
Put fields shared by every progress row in one short heading. Keep row labels
stable while work runs, and begin each row with the device and workload shape.
Show CPU and GPU routes as separate tasks. Animate only the task currently
executing; use a quiet mark for pending tasks and a check for completed tasks.
Refresh long repeated work at least every 100 completed structures; keep one
indivisible batched model call as one task unit so the display matches the
measured operation. Size the completed/total column to its widest value and
right-align it so counts with different totals end on the same vertical line.

## 6. Writing standard

Write for computational chemistry and scientific software practitioners.

Name operations and relationships precisely:

- Check the official documentation and installed signature before naming an
  operation.
- Distinguish constructors, conversion class methods, readers, adapters,
  bindings, wrappers, and workflows.
- Use **same** for an identical interface or behavior. Use **similar** or
  **parallel** when different source types or methods produce the same target
  object.
- Prefer the API name and a direct verb such as *constructs*, *converts*,
  *reads*, *packs*, or *returns*. Use metaphors such as *bridge*, *handoff*, and
  *path* only when they clarify a real boundary.
- Repeat the clearest technical term instead of rotating synonyms.

- Use short paragraphs and familiar words from computational science/chemistry.
- Put the actor before the action.
- State what the learner can now do.
- Define a term at first use and keep its name stable.
- Preserve exact API names, shapes, units, devices, and measured values.
- Explain a choice where the learner makes it.
- Use bullets for genuine lists and tables for repeated fields.
- Keep the opening short enough to reach the first executable result quickly.
- Remove generic enthusiasm, promotional claims, motivational preambles,
  inflated transitions, and conclusions that merely restate headings.

Use source comments such as `REVISE: replace after rendered review` for open
editorial or visual decisions that need an educator’s attention. Remove or
resolve them before release.

## 7. Scientific and performance care

Separate four kinds of claims:

1. **execution:** the cell ran and returned the expected keys or shapes;
2. **numerical correctness:** two supported routes agree within a stated
   tolerance;
3. **scientific meaning:** the result supports a bounded chemical or physical
   interpretation; and
4. **performance:** a named operation ran faster or used resources differently
   under a reported setup.

State model source, license, supported chemical scope relevant to the example,
requested outputs, units, precision, and neighbor requirements before the first
scientific interpretation. A valid adapter demonstrates interface correctness;
scientific suitability needs a separate check.

For batched work, check graph counts, atom counts, boundaries, field levels,
and per-graph recovery before making a speed claim. For composition, check
component closure, output shapes, and any native-model parity the lesson claims.

A performance lesson should:

1. name the question;
2. hold the scientific work fixed;
3. name setup, transfer, compilation, warm-up, synchronization, and timed
   boundaries;
4. report hardware, software identity, workload size, and batch size;
5. repeat measurements and report a representative value with spread;
6. report latency and a workload-normalized rate when useful;
7. display the data before plotting it;
8. use a plot that answers the stated question; and
9. preserve cases where the measured approach loses or fails; and
10. save only reviewed outputs.

When target hardware is unavailable during a workshop, run the same public API
on the available hardware and provide clearly labelled recorded results from
the target system. Record the target hardware, software versions, workload,
warm-up, repetitions, and timed boundaries with those results.

Routine setup output should report the execution class, such as `CUDA GPU` or
`CPU`. Report the exact hardware model with benchmarks and recorded performance
results. When the public API supports CPU, a small CPU fallback can serve as a
quick walkthrough; label CUDA as the course and performance target.

Use familiar decimal units in learner-facing figures. For tutorial-scale timing,
plot seconds on a linear axis and write the elapsed value above each bar. Put a
shared live elapsed timer above multi-task progress rows.

Teach `torch.compile` with GPU pipelines and profiling in Part 06, where setup,
graph capture, parity, and timing can be explained together.

Check software, model weights, datasets, and visual assets separately for
license and redistribution terms. A software license does not establish the
terms for a linked checkpoint, dataset, or figure.

## 8. Authoring and review

Before implementation, record a concise design brief in the notebook worklog:

- lesson outcome and prior knowledge;
- cell and visual sequence;
- visible public APIs;
- helper boundaries;
- structures, model or service, outputs, shapes, units, and scope;
- expected runtime;
- bounded exercise; and
- static, numerical, fresh-kernel, and rendered checks.

Review in three passes.

### Structure

- The opening reaches a useful result quickly.
- One main action appears in each code cell.
- Public API calls remain visible.
- Computation, shaping, display, plotting, and interpretation are separate.
- Terms follow the stable object ladder.
- Later parts link prior teaching and add one new capability.

### Execution

- The frozen runtime check passes.
- The complete notebook namespace parses.
- Scoped tests pass.
- A fresh kernel executes every cell in order.
- Numerical claims match saved output.
- Runtime and hardware are recorded for measured results.

### Rendered learner review

- The banner, course map, diagrams, viewers, tables, and plots render at normal
  teaching width.
- Callouts fit the content column in light and dark themes.
- Text density supports a live workshop pace.
- Links, alt text, captions, units, and exercises are usable.
- A fresh reader can explain what each object is and what the next cell enables.

Treat educator feedback as a delta:

1. identify what the draft led the learner to do or infer;
2. identify what the educator wanted the learner to see, decide, or reuse; and
3. extract the general teaching rule.

Put durable rules in this guide. Put API facts in the technical reference. Put
cell IDs, molecules, timings, contributor names, and one-off repairs in the
notebook worklog. Test a new general rule on another lesson before treating it
as stable.

## 9. Teaching foundations

### Design the learning activity before the cells

Give each notebook one dominant role. A tutorial guides a first successful
path, a how-to solves a named task, an explanation builds a mental model, and a
reference lists facts for lookup. Link to another document type when the live
lesson would otherwise become a catalogue.

Use the ADDIE cycle at course scale:

1. **Analyze** the audience, prior knowledge, hardware, time, and scientific
   task.
2. **Design** the learner outcome, the evidence that demonstrates it, and the
   sequence of API decisions.
3. **Develop** the smallest runnable cells, visuals, exercises, and supporting
   helpers.
4. **Implement** the lesson in the real workshop environment with a fresh
   kernel and representative hardware.
5. **Evaluate** learner work, educator feedback, execution evidence, and the
   rendered notebook, then revise the next release.

Use a short notebook learning cycle:

1. show one complete worked path when the API has several connected steps;
2. ask the learner to predict one shape, field, boundary, or result;
3. give them a one- to five-line editable cell;
4. let the output provide immediate feedback;
5. explain the result in one short paragraph; and
6. offer one bounded variation that reuses the public API.

Keep each exercise next to the code it practises. State the task, name the
editable value, and give a visible success signal. A portable answer can use
native notebook HTML:

```html
<details>
<summary>Check the answer</summary>

The expected result is ... because ...

</details>
```

The answer follows the learner attempt and stays closed by default. Use
`nbgrader` when a course needs collected, scored assignments. Pilot richer
MyST exercises in a separate environment before adding a notebook extension to
the frozen workshop runtime.

These choices follow the
[Diátaxis documentation framework](https://diataxis.fr/), the
[Jupyter for Education guide](https://jupyter4edu.github.io/jupyter-edu-book/),
and ADDIE's iterative instructional-design structure. They guide the teaching
decision; they do not replace scientific and API review.

### Start with the Toolkit examples

The [Toolkit examples gallery](https://nvidia.github.io/nvalchemi-toolkit/examples/)
is the first teaching reference for Toolkit lessons. The examples show API
paths, vocabulary, extension points, and working combinations reviewed by the
engineering team. Inspect the relevant examples before designing a notebook.

Reuse a focused API sequence, scientific setup, or check when it fits the
lesson. Confirm it against the pinned course version, then reshape the pacing
for a live workshop. A gallery script may cover several features at once; a
tutorial notebook can teach the same calls through smaller cells and more
inspection.

| Course need | Toolkit examples to inspect |
|---|---|
| Atomic data and batching | [AtomicData and Batch](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/01_data_structures.html) and [ASE integration](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/03_ase_integration.html) |
| Optimization and molecular dynamics | [FIRE geometry optimization](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/02_geometry_optimization.html), [NVE](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/04_nve_energy_conservation.html), and the [basic examples gallery](https://nvidia.github.io/nvalchemi-toolkit/examples/basic/index.html) |
| Zarr data and trajectories | [Writing and replaying trajectories with Zarr](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/02_trajectory_zarr_io.html) |
| Model composition | [Additive model composition](https://nvidia.github.io/nvalchemi-toolkit/examples/advanced/07_composable_model_composition.html) and the [AIMNet2, Ewald, and DFTD3 pipeline](https://nvidia.github.io/nvalchemi-toolkit/examples/advanced/08_aimnet2_ewald_pipeline.html) |
| Hooks | [Writing a custom hook](https://nvidia.github.io/nvalchemi-toolkit/examples/advanced/02_custom_hook.html) |
| Custom dynamics | [Building a custom integrator](https://nvidia.github.io/nvalchemi-toolkit/examples/advanced/05_custom_integrator.html) |
| GPU pipelines and inflight work | [Multi-stage dynamics with FusedStage](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/01_multistage_pipeline.html) and [inflight batching](https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/04_inflight_batching.html) |
| Distributed execution | [Distributed pipeline examples](https://nvidia.github.io/nvalchemi-toolkit/examples/distributed/index.html) for rank ownership, buffers, sinks, logging, and profiling |

Toolkit-Ops lessons should begin with the
[Toolkit-Ops examples gallery](https://nvidia.github.io/nvalchemi-toolkit-ops/examples/index.html)
and follow the framework binding used by the course environment.

### External tutorials to explore

The links below are a starting set. For each new lesson, inspect a few current
tutorials that match its object, workflow, or learner action. Look for useful
pacing, executable feedback, visual explanations, and exercises. Add a durable
rule to this guide after the same lesson-design need appears across multiple
notebooks.

Check licenses before reusing text, code, figures, data, or other assets.

| Source | Teaching pattern used here |
|---|---|
| [Jupyter for Education](https://jupyter4edu.github.io/jupyter-edu-book/) | Choose the notebook's role, use worked examples before variations, and place a prompt, editable cell, feedback, and answer close together. |
| [Diátaxis](https://diataxis.fr/) | Keep tutorial, how-to, explanation, and reference needs distinct so the live lesson remains focused. |
| [ASE Atoms tutorial](https://ase-workshop-2023.github.io/tutorial/02-the-atoms-object/index.html) | Begin with a familiar scientific object, inspect it through small executable steps, and use structure graphics to support spatial reasoning. |
| [SymPy introductory tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html) | Teach core operations before domain-scale problems so learners can combine the primitives themselves. |
| [Dask overview](https://ncar.github.io/dask-tutorial/notebooks/00-dask-overview.html) | Establish the execution model and the reason for scaling before detailed APIs. |
| [xarray SciPy workshop](https://tutorial.xarray.dev/workshops/scipy2026/index.html#schedule) | Organize a longer course as a visible sequence of focused, executable lessons. |
| [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) and [Learning PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples) | Show a complete path early, then keep one problem stable while introducing tensors, differentiation, modules, and optimizers. |
| [TorchSim high-level tutorial](https://torchsim.github.io/torch-sim/tutorials/high_level_tutorial.html) | Produce a useful simulation result before opening lower-level state and batching details. |
| [BioNeMo recipes](https://docs.nvidia.com/bionemo-framework/latest/main/recipes/recipes/index.html) | Keep reusable API choices visible and accept local repetition when another abstraction would hide the lesson. |
| [PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html), [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), and [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) | Establish correctness, define the timed boundary, and reason about compute, memory, and communication before interpreting performance. |
| [Warp Ising-model tutorial](https://github.com/NVIDIA/accelerated-computing-hub/blob/aabe760ce8b71c6ca5993513fa51bbfb7db37d29/tutorials/warp/notebooks/02__ising_model.ipynb) | Show a flawed result, correct the method, and compare the correction with an independent expectation. |
| [University of Amsterdam pipeline-parallelism notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html) | Use diagrams and timelines to make data placement, work, and idle time visible. |

## Official product references

- [ALCHEMI developer hub](https://developer.nvidia.com/cuda/cuda-x-libraries/alchemi)
- [Toolkit documentation](https://nvidia.github.io/nvalchemi-toolkit/)
- [Toolkit examples](https://nvidia.github.io/nvalchemi-toolkit/examples/)
- [Toolkit source](https://github.com/NVIDIA/nvalchemi-toolkit)
- [Toolkit-Ops documentation](https://nvidia.github.io/nvalchemi-toolkit-ops/)
- [Toolkit-Ops examples](https://nvidia.github.io/nvalchemi-toolkit-ops/examples/index.html)
- [Toolkit-Ops source](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
