# Fundamental principles for hands-on product tutorials

These principles apply to executable tutorials for technical products,
libraries, and platforms used in scientific computing, machine learning, and
accelerated computing.

A tutorial may teach a scientific method, a software product, or both. Decide
which is primary. In a product tutorial, the product is the curriculum. The
application gives its APIs a concrete purpose, but it should not limit useful
product coverage.

## 1. Define what the learner is learning

Write the main learning outcomes as actions the learner can repeat after the
tutorial.

For a product tutorial, prefer outcomes such as:

- represent an input with the product's data model;
- run, configure, and inspect a normal workflow;
- choose between execution modes;
- measure performance correctly;
- connect an external component through a public extension API; and
- recover, save, and reuse the outputs.

The application has supporting outcomes: understand the input, recognize a
reasonable result, and know the limits of the example. Background theory should
be limited to what the learner needs for those decisions.

Before writing, divide the product capabilities into three groups:

1. essential capabilities the tutorial or series must teach;
2. supporting capabilities that make the main path understandable; and
3. advanced or unrelated capabilities that will be linked or taught later.

This prevents both a narrow demo that misses the product and an unstructured
tour of API names.

## 2. Use applications as teaching vehicles

A clear application gives the learner something real to inspect. It does not
have to carry every product capability.

Keep one example while it helps the learner compare methods or execution
choices. Change the example when a capability genuinely needs a different
model, data shape, scale, or operating condition. Explain the transition:

- what the first example could not show;
- why the new example is suitable; and
- which product concepts remain unchanged.

A tutorial series may therefore use one anchor example plus a few short,
purposeful examples. That is better than forcing every feature into one
scientific story or changing examples without explanation.

The [PyTorch beginner path](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
uses a complete machine-learning workflow for continuity, while PyTorch recipes
use shorter independent problems for focused features. Both structures work
because the teaching goal is clear.

## 3. Show a useful product workflow early

Let the learner reach a real output before explaining every component in depth.
The first workflow should use the documented public path and be small enough to
run during the opening of the session.

Then repeat a short learning cycle:

1. name the question or capability;
2. ask the learner what they expect;
3. run one focused operation;
4. inspect the relevant output;
5. explain the result; and
6. reuse the capability in the next step.

The
[TorchSim high-level tutorial](https://torchsim.github.io/torch-sim/tutorials/high_level_tutorial.html)
starts with a working simulation interface before its lower-level tutorials
expose state and batching details. PyTorch Quickstart similarly reaches a
prediction before the beginner series examines each stage separately.

## 4. Teach breadth as a progression, not a catalogue

Broad product exposure is a valid tutorial goal. Give that breadth an order.
A useful progression is:

```text
run it -> inspect it -> configure it -> scale it -> extend it -> reuse it
```

Each section should add an action learners can use again and produce evidence
that it worked. An API does not earn a section merely because it exists.
It should help learners complete one of the stated outcomes.

Use a capability more than once when possible. The first use establishes the
syntax; the second use shows how it fits a different operation or input. A
single unexplained call is exposure, not learning.

Across a series, mark capabilities as introduced, practised, or extended. This
allows wide coverage without making every notebook contain every feature.

## 5. Make one important change at a time

Keep the surrounding conditions stable when making a comparison:

- keep the input fixed when changing execution strategy;
- keep the operation fixed when comparing data layouts;
- keep the interface fixed when changing models;
- keep the workload fixed when measuring an optimization.

Sometimes the example itself must change. In that case, keep the product idea
stable and say why the new input is required.

[Learning PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples)
uses the same curve-fitting problem while introducing tensors, automatic
differentiation, modules, optimizers, and custom modules. The
[NERSC Deep Learning at Scale tutorial](https://github.com/NERSC/dl-at-scale-training)
keeps one weather workload while moving from a single-GPU baseline through
profiling and distributed execution.

## 6. Keep the product interface visible

Show the public APIs and choices learners are expected to reuse. Keep inputs,
requested outputs, important settings, units, and execution choices near the
code that uses them.

Move supporting work into small, well-named helpers when it obscures the
product lesson. Common examples are downloads, large fixture construction,
plot styling, presentation HTML, and repetitive result formatting.

Do not hide the central product call behind a tutorial-only wrapper. Do not use
private fields, runtime patches, or an ad-hoc replacement for a public workflow
the tutorial is meant to teach.

The
[BioNeMo recipes](https://docs.nvidia.com/bionemo-framework/latest/main/recipes/recipes/index.html)
favor readable, self-contained examples with important design choices visible.
They accept some repetition when abstraction would make the educational path
harder to follow.

## 7. Inspect and explain every important result

A successful cell shows that code ran. It does not establish correctness,
scientific quality, or better performance.

Use the check needed for the lesson:

- structural: shapes, labels, ordering, placement, or identity;
- numerical: agreement, round trips, invariants, or convergence;
- scientific: theory, trusted calculations, or measured data;
- performance: equivalent results before and after an optimization.

Show the evidence beside the operation that produced it. Then state the limited
conclusion it supports.

The
[Warp Ising-model tutorial](https://github.com/NVIDIA/accelerated-computing-hub/blob/aabe760ce8b71c6ca5993513fa51bbfb7db37d29/tutorials/warp/notebooks/02__ising_model.ipynb)
shows a flawed parallel method, corrects it, and compares the corrected behavior
with an analytical result. TorchSim's
[integrator analysis](https://torchsim.github.io/torch-sim/tutorials/integrator_tests_analysis.html)
connects numerical output to expected physical relationships.

## 8. Treat performance as an experiment

Performance teaching should answer when an approach helps, not promise that it
always wins.

1. Establish a correct baseline.
2. Keep the requested work fixed.
3. State what is and is not timed.
4. Separate setup and first-call costs from repeated execution.
5. Synchronize asynchronous devices correctly.
6. Measure relevant workload sizes.
7. Repeat measurements and report the summary statistic and spread.
8. Record the hardware and relevant software versions.
9. Report response time and throughput when they answer different questions.
10. Profile before explaining a bottleneck.
11. Preserve failures and cases where the proposed method loses.

The [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
demonstrates misleading timing before correcting it. The
[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
connects observed time to operators, memory use, and an execution trace.

For distributed work, show total input, work per worker, communication, idle
time, and result collection. Distinguish a fixed-total-work test from one that
increases work with the number of workers. The
[JAX Scaling Book](https://jax-ml.github.io/scaling-book/) teaches learners to
reason from compute, memory, and communication before profiling real models.

## 9. Design for live learning

Give each code cell one main job and follow it with the output or interpretation
that makes the job meaningful. Place short results throughout the tutorial
instead of putting all computation into one long block.

A useful visual answers a question, for example:

- What changed after this operation?
- Which result belongs to which input?
- Where is time being spent?
- How does the output compare with a reference?

The
[University of Amsterdam pipeline-parallelism notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html)
visualizes device placement and idle time. PyTorch TensorBoard tutorials ask
learners to inspect graphs, embeddings, images, and training curves rather than
using them as decoration.

Use short exercises for meaningful choices: change an input, select a mode,
predict an output shape, explain a measured crossover, or extend a working
interface. Give quick evidence of success.

If target hardware or a full workload is not available live, teach the same
public API with a short run and provide clearly identified results from the
target environment for analysis. Do not silently alter the declared workload.

## 10. Finish with reuse and navigation

End by asking the learner to use the product workflow on a related input or
configuration. This checks whether they can transfer the product skill rather
than repeat the notebook.

Summarize:

- the public APIs they can now use;
- the choices they learned to make;
- the outputs and checks they can interpret;
- the capabilities intentionally left for later; and
- the next focused tutorial or reference page.

Leave a useful result, such as a table, model, trajectory, profile, or processed
dataset, that supports the transfer task or later work.

## Writing standard

- Use direct sentences and short, task-led headings.
- State what the learner should inspect, not only what the code does.
- Keep code, output, and interpretation together.
- Put units in labels and table headings.
- Give every figure a concrete caption and useful alternative text.
- Do not rely on color alone to communicate meaning.
- Remove setup noise that obscures the lesson.
- Use familiar labels such as settings, source, check, result, and saved file.
- Avoid filler such as "dive into", "unlock", "seamlessly", "powerful",
  "simply", and "obviously".

## Review questions

- Is it clear whether the primary subject is the product, method, or science?
- Do the learning outcomes cover the important product capabilities?
- Does each application serve a stated learning outcome?
- Does the learner obtain a useful product result early?
- Is broad coverage organized as a progression rather than an API list?
- Does each section add one transferable action?
- Are public APIs and result-affecting choices visible?
- Is every important output inspected and interpreted?
- Are correctness, scientific comparison, and performance measured separately?
- Can the learner apply the product workflow to a related case?

## Sources and reuse

The examples above were reviewed for teaching methods, not as text, code,
figures, or data to copy. Repository licenses include BSD 3-Clause for PyTorch
Tutorials, MIT for TorchSim, Apache 2.0 for Warp, BioNeMo Framework, and the
Accelerated Computing Hub code, and MIT for the JAX Scaling Book and University
of Amsterdam notebooks. Accelerated Computing Hub written material and graphics
use CC BY-NC-SA 4.0. The NERSC Deep Learning at Scale repository did not show
an explicit license during this review.

Check the license of each code sample, figure, dataset, and model before reuse.
A repository's software license does not automatically cover linked data,
model weights, or third-party assets.
